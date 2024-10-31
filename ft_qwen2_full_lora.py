import os
import argparse
import logging
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, get_linear_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from tqdm.auto import tqdm
import numpy as np
import re
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from models_astro_qwen2 import AstroQwen2VLFull
from dataset_full_qwen2vl import Qwen2VLDataset

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
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2VL model with LoRA")
    # Original arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_hdf5_path", type=str, required=True)
    parser.add_argument("--eval_hdf5_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # Add LoRA specific arguments
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    return parser.parse_args()

def extract_number_from_text(text: str) -> float:
    try:
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        return float('nan')
    except:
        return float('nan')



def prepare_model_for_training(model, args):
    """
    Prepare the model for training by adding LoRA adapters and setting up trainable parameters
    """
    # Define the target modules for LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ]
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Set specific modules to train in fp32
    for name, param in model.named_parameters():
        if any(module_name in name for module_name in [
            "vision_model",
            "spec_projector",
            "struc_projector"
        ]):
            param.requires_grad = True
            # Convert to fp32 for these specific modules
            param.data = param.data.to(torch.float32)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def evaluate_model(
    model,
    eval_dataset,
    processor,
    task_type: str,
    batch_size: int,
    accelerator: Accelerator
) -> Tuple[float, float]:
    """
    Evaluate model on a specific task.
    Returns MSE and R² score for the task.
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    # 设置数据集为单任务评估模式
    eval_dataset.num_questions = 1
    eval_dataset.task_types = [task_type]
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=4
    )
    
    eval_dataloader = accelerator.prepare(eval_dataloader)
    
    logger.info(f"Evaluating task: {task_type}")
    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = batch['processed_inputs']
            
            # 生成预测结果
            generated_ids = accelerator.unwrap_model(model).generate(
                **inputs,
                max_new_tokens=10,
                num_beams=1,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None
            )
            
            # 裁剪生成的id序列
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # 解码输出
            decoded_outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # 提取真实值
            for ans in batch['answers']:
                ground_truths.extend(ans)
            
            # 提取预测值
            for output in decoded_outputs:
                print(output)
                pred_value = extract_number_from_text(output)
                predictions.append(pred_value)
    
    # 过滤掉NaN值
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) 
                   if not (np.isnan(p) or np.isnan(g))]
    print(valid_pairs)
    
    if not valid_pairs:
        logger.warning(f"No valid predictions for task {task_type}")
        return float('inf'), float('nan')
    
    pred_array = np.array([p for p, _ in valid_pairs])
    gt_array = np.array([g for _, g in valid_pairs])
    
    # 计算MSE
    mse = np.mean((pred_array - gt_array) ** 2)
    
    # 计算R²分数
    ss_res = np.sum((gt_array - pred_array) ** 2)
    ss_tot = np.sum((gt_array - np.mean(gt_array)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    
    return mse, r2

def run_evaluation(
    model,
    eval_dataset,
    processor,
    eval_tasks: List[str],
    batch_size: int,
    accelerator: Accelerator,
    global_step: int
) -> Tuple[float, float]:
    """
    运行所有任务的评估并记录结果
    """
    logger.info(f"Starting evaluation at step {global_step}")
    model.eval()
    
    task_metrics = {}
    for task in eval_tasks:
        mse, r2 = evaluate_model(
            model=model,
            eval_dataset=eval_dataset,
            processor=processor,
            task_type=task,
            batch_size=batch_size,
            accelerator=accelerator
        )
        
        task_metrics[task] = {'mse': mse, 'r2': r2}
        
        # 记录每个任务的指标
        accelerator.log({
            f"eval/{task}/mse": mse,
            f"eval/{task}/r2": r2
        }, step=global_step)
        
        logger.info(f"Task {task} - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # 计算平均指标
    avg_mse = np.mean([metrics['mse'] for metrics in task_metrics.values()])
    avg_r2 = np.mean([metrics['r2'] for metrics in task_metrics.values() 
                      if not np.isnan(metrics['r2'])])
    
    # 记录平均指标
    accelerator.log({
        "eval/average/mse": avg_mse,
        "eval/average/r2": avg_r2
    }, step=global_step)
    
    logger.info(f"Average metrics - MSE: {avg_mse:.4f}, R²: {avg_r2:.4f}")
    
    return avg_mse, avg_r2


def train():
    args = parse_args()
    
    # Configure project
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.output_dir,
    )
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=["tensorboard", "all"],
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    
    accelerator.init_trackers(
        project_name="qwen2vl_lora_training",
        config=vars(args)
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # Load processor and model
    min_pixels = 110*110*3
    max_pixels = 110*110*3
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    model = AstroQwen2VLFull.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )
    
    # Prepare model with LoRA and set up trainable parameters
    model = prepare_model_for_training(model, args)
    
    # Initialize datasets
    train_dataset = Qwen2VLDataset(
        hdf5_path=args.train_hdf5_path,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        split="train",
        num_questions=1,
        max_length=512
    )
    
    eval_dataset = Qwen2VLDataset(
        hdf5_path=args.eval_hdf5_path,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        split="eval",
        num_questions=1,
        max_length=512,
        max_samples=args.eval_samples
    )
    
    # Initialize dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4
    )
    
    # Create optimizer groups with different learning rates
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(module_name in n for module_name in [
                          "vision_model", "spec_projector", "struc_projector"
                      ]) and p.requires_grad],
            "lr": args.learning_rate * 0.1,  # Lower learning rate for vision and projector parameters
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(module_name in n for module_name in [
                          "vision_model", "spec_projector", "struc_projector"
                      ]) and p.requires_grad],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training loop
    eval_tasks = [
        'task1_redshift',
        'task2_log_mstar',
        'task2_z_mw',
        'task2_sSFR',
        'task2_tage_mw'
    ]
    
    global_step = 0
    best_avg_mse = float('inf')
    best_avg_r2 = float('-inf')
    
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}"
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch['processed_inputs'])
                loss = outputs.loss
                print(loss)
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                global_step += 1
                
                # Logging and evaluation logic remains the same
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / args.logging_steps
                    accelerator.log(
                        {
                            "train_loss": avg_loss,
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )
                    total_loss = 0
                
                # 在training循环中的评估部分（替换原有的evaluation注释）
                if global_step % args.eval_steps == 0:
                    logger.info(f"Running evaluation at step {global_step}")
                    # 运行评估
                    avg_mse, avg_r2 = run_evaluation(
                        model=model,
                        eval_dataset=eval_dataset,
                        processor=processor,
                        eval_tasks=eval_tasks,
                        batch_size=args.eval_batch_size,
                        accelerator=accelerator,
                        global_step=global_step
                    )
                    
                    # 更新最佳指标并保存模型
                    save_model = False
                    if avg_mse < best_avg_mse:
                        best_avg_mse = avg_mse
                        save_model = True
                        logger.info(f"New best MSE: {best_avg_mse:.4f}")
                    
                    if avg_r2 > best_avg_r2:
                        best_avg_r2 = avg_r2
                        save_model = True
                        logger.info(f"New best R²: {best_avg_r2:.4f}")
                    
                    if save_model and accelerator.is_main_process:
                        logger.info("Saving new best model")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            os.path.join(args.output_dir, "best_model"),
                            save_function=accelerator.save
                        )
                    
                    # 恢复训练模式
                    model.train()
                
                # Save checkpoints
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    # Save both the LoRA adapters and the full model
                    unwrapped_model.save_pretrained(
                        os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                        save_function=accelerator.save
                    )
        
        progress_bar.close()
    
    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(args.output_dir, "final_model"),
            save_function=accelerator.save
        )
    
    accelerator.end_training()

if __name__ == "__main__":
    train()