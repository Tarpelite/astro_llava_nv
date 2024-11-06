import os
import argparse
import logging
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from tqdm.auto import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, accuracy_score, f1_score
# import wandb
import re

from models_astro_ultra_qwen2 import AstroQwen2VLForConditionalGeneration
from dataset_ultra_qwen2vl import Qwen2VLTrainingDataset, Qwen2VLEvaluationDataset, collate_fn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_regression_data", type=str, required=True)
    parser.add_argument("--train_classification_data", type=str, required=True)
    parser.add_argument("--eval_regression_data", type=str, required=True)
    parser.add_argument("--eval_classification_data", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    # Evaluation and logging
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    args = parser.parse_args()

    # 构建数据字典
    args.train_data = {
        "regression": args.train_regression_data,
        "classification": args.train_classification_data
    }
    args.eval_data = {
        "regression": args.eval_regression_data,
        "classification": args.eval_classification_data
    }
    
    return args

def prepare_model_for_training(model):
    """冻结基础模型参数,只训练新增参数"""
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 分别处理Module和Parameter
    trainable_modules = [
        model.spec_projector,
        model.struc_projector,
        model.spec_norm,
        model.struc_norm,
        model.num_head
    ]
    
    trainable_parameters = [
        model.spec_scale,
        model.struc_scale,
        model.lm_weight,
        model.regression_weight
    ]
    
    # 添加每一层的expert参数
    for layer in model.model.layers:
        if hasattr(layer, 'moe'):  # 确保layer有moe属性
            trainable_modules.extend([
                layer.moe.router,
                layer.moe.experts[0],  # EuclideanFFN
                layer.moe.experts[1],  # HyperbolicFFN
                layer.moe.experts[2],  # SphericalFFN
            ])
            trainable_parameters.append(layer.moe.temperature)
    
    # 设置Module的参数为可训练
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True
            
    # 设置Parameter为可训练
    for param in trainable_parameters:
        param.requires_grad = True
            
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {trainable_params}")
    
    return model

def extract_number(text: str) -> float:
    try:
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        return float('nan')
    except:
        return float('nan')


def evaluate_regression(model, eval_dataloader, processor, args):
    """评估回归任务"""
    model.eval()
    task_metrics = {}
    
    for task in model.regression_tasks:
        predictions = []
        labels = []
        
        for batch in eval_dataloader:
            with torch.no_grad():
                # 第一步:获取token生成结果
                generated_ids = model.generate(
                    **batch["processed_inputs"],
                    max_new_tokens=5,
                    do_sample=False
                )
                generated_text = processor.batch_decode(generated_ids)
                
                # 第二步:对于数值预测使用回归头
                for text in generated_text:
                    if " num" in text:
                        # 获取最后一个token的logits
                        last_token_logits = model.get_last_token_logits(generated_ids)
                        # 添加特征logits
                        logits = last_token_logits + model.get_feature_logits(batch)
                        # 使用回归头预测
                        pred = model.num_head(logits).squeeze()
                        predictions.append(pred.item())
                    else:
                        # 从生成文本中提取数值
                        pred = extract_number(text)
                        predictions.append(pred)
                        
                labels.extend(batch['labels'].cpu().numpy())
        
        # 计算指标
        mse = np.mean((np.array(predictions) - np.array(labels)) ** 2)
        r2 = r2_score(labels, predictions)
        
        task_metrics[task] = {
            'mse': mse,
            'r2': r2
        }
        
        # 记录到tensorboard
        args.writer.add_scalar(f'{task}/mse', mse, args.global_step)
        args.writer.add_scalar(f'{task}/r2', r2, args.global_step)
    
    return task_metrics

def evaluate_classification(model, eval_dataloader, args):
    """评估分类任务"""
    model.eval()
    task_metrics = {}
    
    def extract_class_label(text):
        """从生成的文本中提取分类标签"""
        # 寻找(a), (b), (c)等模式
        matches = re.search(r'\(([a-z])\)', text.lower())
        if matches:
            label_char = matches.group(1)
            # 将a,b,c转换为0,1,2
            return ord(label_char) - ord('a')
        return None
    
    for task in model.classification_tasks:
        predictions = []
        labels = []
        
        for batch in eval_dataloader:
            with torch.no_grad():
                # 使用生成来获取答案
                generated_ids = model.generate(
                    **batch["processed_inputs"],
                    max_new_tokens=5,
                    do_sample=False
                )
                generated_text = args.processor.batch_decode(generated_ids)
                
                # 从生成的文本中提取预测标签
                for text in generated_text:
                    pred_label = extract_class_label(text)
                    if pred_label is not None:
                        predictions.append(pred_label)
                    else:
                        # 如果无法提取标签，添加一个默认值或跳过
                        predictions.append(-1)
                
                labels.extend(batch['labels'].cpu().numpy())
        
        # 过滤掉无效预测
        valid_indices = np.array(predictions) != -1
        predictions = np.array(predictions)[valid_indices]
        labels = np.array(labels)[valid_indices]
        
        if len(predictions) > 0:
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
        else:
            acc = 0
            f1 = 0
            
        task_metrics[task] = {
            'accuracy': acc,
            'f1': f1
        }
        
        # 记录到tensorboard
        args.writer.add_scalar(f'{task}/accuracy', acc, args.global_step)
        args.writer.add_scalar(f'{task}/f1', f1, args.global_step)
    
    return task_metrics

def train():
    args = parse_args()


    
    # 初始化accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化tensorboard
    if accelerator.is_main_process:
        args.writer = SummaryWriter(args.output_dir)
        # wandb.init(project="astro_qwen2", name=args.run_name)
    
    # 加载模型和数据
    min_pixels = 110*110*3
    max_pixels = 144*144*3
    processor = AutoProcessor.from_pretrained(args.model_path,min_pixels=min_pixels, max_pixels=max_pixels)
    args.processor = processor
    model = AstroQwen2VLForConditionalGeneration.from_pretrained(args.model_path)
    model = prepare_model_for_training(model)
    
    train_dataset = Qwen2VLTrainingDataset(
        hdf5_paths=args.train_data,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor
    )
    
    eval_dataset = Qwen2VLEvaluationDataset(
        hdf5_paths=args.eval_data,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        max_regression_samples=args.max_eval_samples
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collate_fn
    )
    
    # 准备优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_update_steps = len(train_dataloader) * args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_update_steps
    )
    
    # 准备训练
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    # 训练循环
    global_step = 0
    best_metrics = {
        'mse': float('inf'),
        'r2': float('-inf'),
        'accuracy': 0,
        'f1': 0
    }
    
    for epoch in range(args.num_train_epochs):
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # import pudb;pu.db;
                outputs = model(**batch["processed_inputs"], return_dict=True)
                loss = outputs.loss
                print(loss)
                accelerator.backward(loss)
                
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # 记录训练loss
                    if global_step % args.logging_steps == 0:
                        args.writer.add_scalar('train/loss', loss.item(), global_step)
                        
                    # 评估
                    if global_step % args.eval_steps == 0:
                        reg_metrics = evaluate_regression(
                            model, eval_dataloader, processor, args
                        )
                        cls_metrics = evaluate_classification(
                            model, eval_dataloader, args
                        )
                        
                        # 更新最佳指标并保存模型
                        avg_mse = np.mean([m['mse'] for m in reg_metrics.values()])
                        avg_r2 = np.mean([m['r2'] for m in reg_metrics.values()])
                        avg_acc = np.mean([m['accuracy'] for m in cls_metrics.values()])
                        avg_f1 = np.mean([m['f1'] for m in cls_metrics.values()])
                        
                        if avg_mse < best_metrics['mse']:
                            best_metrics['mse'] = avg_mse
                            accelerator.save_state(f"{args.output_dir}/best_mse")
                            
                        if avg_r2 > best_metrics['r2']:
                            best_metrics['r2'] = avg_r2
                            accelerator.save_state(f"{args.output_dir}/best_r2")
                            
                        if avg_acc > best_metrics['accuracy']:
                            best_metrics['accuracy'] = avg_acc
                            accelerator.save_state(f"{args.output_dir}/best_accuracy")
                            
                        if avg_f1 > best_metrics['f1']:
                            best_metrics['f1'] = avg_f1
                            accelerator.save_state(f"{args.output_dir}/best_f1")
                    
                    # 定期保存checkpoint
                    if global_step % args.save_steps == 0:
                        accelerator.save_state(
                            f"{args.output_dir}/checkpoint-{global_step}"
                        )
    
    # 保存最终模型
    accelerator.save_state(f"{args.output_dir}/final")
    
    if accelerator.is_main_process:
        args.writer.close()
        # wandb.finish()

if __name__ == "__main__":
    train()