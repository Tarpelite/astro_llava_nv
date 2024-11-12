import os
import argparse
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
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
import math
from tqdm import tqdm

from models_astro_ultra_qwen2 import AstroQwen2VLForConditionalGeneration
from dataset_ultra_qwen2vl import Qwen2VLTrainingDataset, Qwen2VLEvaluationDataset, collate_fn, Qwen2VLClassificationEvaluationDataset, Qwen2VLRegressionEvaluationDataset

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
    # parser.add_argument("--eval_regression_data", type=str, required=True)
    # parser.add_argument("--eval_classification_data", type=str, required=True)
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
    

    parser.add_argument("--regression_eval_dir", type=str, required=True,
                       help="Directory containing split regression evaluation files")
    parser.add_argument("--classification_eval_dir", type=str, required=True,
                       help="Path to classification evaluation HDF5 file")
    parser.add_argument("--eval_regression_tasks", nargs='+', 
                       help="Specific regression tasks to evaluate")
    parser.add_argument("--eval_classification_tasks", nargs='+',
                       help="Specific classification tasks to evaluate")
    parser.add_argument("--samples_per_regression_task", type=int, default=None,
                       help="Number of samples to evaluate per regression task")
    parser.add_argument("--samples_per_classification_task", type=int, default=None,
                       help="Number of samples to evaluate per classification task")

    args = parser.parse_args()

    # 构建数据字典
    args.train_data = {
        "regression": args.train_regression_data,
        "classification": args.train_classification_data
    }
    # args.eval_data = {
    #     "regression": args.eval_regression_data,
    #     "classification": args.eval_classification_data
    # }
    
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


def evaluate_regression(model, eval_dataloader, processor, args, global_step):
    """评估回归任务"""
    model.eval()
    task_metrics = {}
    task_predictions = defaultdict(list)
    task_labels = defaultdict(list)
    
    print(f"Evaluating regression tasks, total batches: {len(eval_dataloader)}")
    for batch in eval_dataloader:
        with torch.no_grad():
            batch_tasks = batch['task_types']
            # print(batch["text_sequences"][0][0])
            
            # 一次只生成一个token
            for i in range(5):  # 最多生成5个token
                if i == 0:
                    # 第一个token
                    outputs = model.generate(
                        **batch["processed_inputs"],
                        max_new_tokens=1,
                        do_sample=False,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    current_ids = outputs.sequences
                    current_hidden = outputs.hidden_states[-1][-1]  # 最后一层的最后一个token的hidden states
                else:
                    # 后续token，需要使用之前的结果作为输入
                    new_inputs = batch["processed_inputs"].copy()
                    new_inputs["input_ids"] = current_ids
                    new_inputs["attention_mask"] = torch.ones_like(current_ids)
                    
                    outputs = model.generate(
                        **new_inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    current_ids = outputs.sequences
                    current_hidden = outputs.hidden_states[-1][-1]
                
                # 检查每个样本是否生成了" num"
                for batch_idx, (ids, task) in enumerate(zip(current_ids, batch_tasks)):
                    last_token = ids[-1].item()
                    if last_token == model.num_token_id:
                        # 如果是" num"，立即使用regression head进行预测
                        pred = model.predict_number(
                            input_ids=ids.unsqueeze(0),
                            hidden_states=current_hidden[batch_idx].unsqueeze(0),
                            spec_features=batch["processed_inputs"].get('spec_features')[batch_idx:batch_idx+1] if 'spec_features' in batch["processed_inputs"] else None,
                            euc_features=batch["processed_inputs"].get('euc_features')[batch_idx:batch_idx+1] if 'euc_features' in batch["processed_inputs"] else None,
                            hyp_features=batch["processed_inputs"].get('hyp_features')[batch_idx:batch_idx+1] if 'hyp_features' in batch["processed_inputs"] else None,
                            sph_features=batch["processed_inputs"].get('sph_features')[batch_idx:batch_idx+1] if 'sph_features' in batch["processed_inputs"] else None
                        )
                        task_predictions[task].append(pred.item())
                        task_labels[task].append(batch['answers'][batch_idx])
                        
                        # 从当前batch中移除已处理的样本
                        mask = torch.ones(len(batch_tasks), dtype=torch.bool)
                        mask[batch_idx] = False
                        batch_tasks = [t for t, m in zip(batch_tasks, mask) if m]
                        for k in batch["processed_inputs"]:
                            if isinstance(batch["processed_inputs"][k], torch.Tensor):
                                batch["processed_inputs"][k] = batch["processed_inputs"][k][mask]
                
                # 如果所有样本都已处理完，提前退出
                if len(batch_tasks) == 0:
                    break
            
            # 对于未生成" num"的样本，添加nan
            for task in batch_tasks:
                task_predictions[task].append(float('nan'))
                task_labels[task].append(batch['answers'][0])  # 注意这里可能需要调整
    
    # 计算每个任务的指标
    for task in task_predictions.keys():
        predictions = np.array(task_predictions[task])
        labels = np.array(task_labels[task])
        
        # 计算指标
        mse = np.mean((predictions - labels) ** 2)
        flag = np.any(np.isnan(predictions))
        r2 = 0 if flag else r2_score(labels, predictions)
        
        print(f"\nTask: {task}")
        print(f"Samples: {len(predictions)}")
        print(f"Predictions (first 3): {predictions[:3]}")
        print(f"Labels (first 3): {labels[:3]}")
        print(f"MSE: {mse:.6f}, R2: {r2:.6f}")
        
        task_metrics[task] = {
            'mse': mse,
            'r2': r2
        }
        
        # 记录到tensorboard
        args.writer.add_scalar(f'{task}/mse', mse, global_step)
        args.writer.add_scalar(f'{task}/r2', r2, global_step)
    
    return task_metrics

def evaluate_classification(model, eval_dataloader, args, global_step):
    """评估分类任务"""
    model.eval()
    task_metrics = {}
    task_predictions = defaultdict(list)
    task_labels = defaultdict(list)
    
    def extract_answer(text: str) -> str:
        """
        提取<|im_start|>assistant和<|im_end|>之间的回答内容
        """
        pattern = r'<\|im_start\|>assistant\s*(.*?)<\|im_end\|>'
        match = re.search(pattern, text, re.DOTALL)  # re.DOTALL让.也能匹配换行符
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_class_label(answer: str) -> int:
        """从回答中提取分类标签
        
        支持的模式包括:
        - 完整括号: (a), (b), (c)
        - 左括号: (a, (b, (c
        - 混合模式: (a), (b, (c)
        
        Args:
            answer: str, 包含答案的字符串
        
        Returns:
            int: 提取的类别标签(0-25对应a-z), 提取失败返回-1
        """
        if not answer:  # 如果答案为空
            return -1
                
        # 匹配以下模式:
        # 1. (a) - 完整括号
        # 2. (a  - 只有左括号
        # 标准化为小写并移除空白字符
        answer = answer.lower().strip()
        
        # 查找所有可能的模式
        # (?:\)|\b) 表示匹配右括号或者词边界
        matches = re.search(r'\(([a-z])(?:\)|\b)', answer)
        
        if matches:
            return ord(matches.group(1)) - ord('a')
        return -1
    
    print(f"Evaluating classification tasks, total batches: {len(eval_dataloader)}")
    for batch in eval_dataloader:
        with torch.no_grad():
            batch_tasks = batch['task_types']
            
            # 生成预测结果
            generated_ids = model.generate(
                **batch["processed_inputs"],
                max_new_tokens=5,
                do_sample=False
            )
            generated_texts = args.processor.batch_decode(generated_ids)
            
            # 处理每个样本的预测结果
            for i, (text, task) in enumerate(zip(generated_texts, batch_tasks)):
                # 提取assistant的回答
                answer = extract_answer(text)
                print(f"Task: {task}")
                print(f"Full text: {text}")
                print(f"Extracted answer: {answer}")
                
                # 从回答中提取标签
                pred_label = extract_class_label(answer)
                task_predictions[task].append(pred_label)
                task_labels[task].append(batch['answers'][i])
    
    # 计算每个任务的指标
    for task in task_predictions.keys():
        predictions = np.array(task_predictions[task])
        labels = np.array(task_labels[task])
        
        # 过滤掉无效预测
        valid_indices = predictions != -1
        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]
        
        if len(valid_predictions) > 0:
            acc = accuracy_score(valid_labels, valid_predictions)
            f1 = f1_score(valid_labels, valid_predictions, average='weighted')
        else:
            acc = 0
            f1 = 0
            
        print(f"\nTask: {task}")
        print(f"Total samples: {len(predictions)}")
        print(f"Valid samples: {len(valid_predictions)}")
        print(f"Invalid samples: {len(predictions) - len(valid_predictions)}")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        task_metrics[task] = {
            'accuracy': acc,
            'f1': f1,
            'valid_ratio': len(valid_predictions) / len(predictions)
        }
        
        # 记录到tensorboard
        args.writer.add_scalar(f'{task}/accuracy', acc, global_step)
        args.writer.add_scalar(f'{task}/f1', f1, global_step)
        args.writer.add_scalar(f'{task}/valid_ratio', len(valid_predictions) / len(predictions), global_step)
    
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
    
    # eval_dataset = Qwen2VLEvaluationDataset(
    #     hdf5_paths=args.eval_data,
    #     image_dir=args.image_dir,
    #     template_path=args.template_path,
    #     processor=processor,
    #     max_regression_samples=args.max_eval_samples
    # )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=args.per_device_eval_batch_size,
    #     collate_fn=collate_fn
    # )
    # 替换原来的eval_dataset初始化
    

    eval_regression_dataset = Qwen2VLRegressionEvaluationDataset(
        eval_dir=args.regression_eval_dir,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        max_samples_per_task=args.samples_per_regression_task,
        selected_tasks=args.eval_regression_tasks
    )

    eval_classification_dataset = Qwen2VLClassificationEvaluationDataset(
        eval_dir=args.classification_eval_dir,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        max_samples_per_task=args.samples_per_classification_task,
        selected_tasks=args.eval_classification_tasks
    )

    # 创建两个dataloader
    eval_regression_dataloader = DataLoader(
        eval_regression_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collate_fn
    )

    eval_classification_dataloader = DataLoader(
        eval_classification_dataset,
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
    model, optimizer, train_dataloader, eval_regression_dataloader, eval_classification_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader,  eval_regression_dataloader, eval_classification_dataloader
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
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            with accelerator.accumulate(model):
                # import pudb;pu.db;
                # if batch["task_types"][0][0] == 1:
                #     print(batch["text_sequences"][0][0])
                outputs = model(**batch["processed_inputs"], return_dict=True)
                loss = outputs.loss
                # print(loss)
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
                            model, eval_regression_dataloader, processor, args, global_step
                        )
                        cls_metrics = evaluate_classification(
                            model, eval_classification_dataloader, args, global_step
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