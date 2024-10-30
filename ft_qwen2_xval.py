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

from models_astro_qwen2 import AstroQwen2VL
from dataset_vision_qwen2vl_xval import Qwen2VLDataset

torch.set_float32_matmul_precision('high')

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
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2VL model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--train_hdf5_path",
        type=str,
        required=True,
        help="Path to training HDF5 file",
    )
    parser.add_argument(
        "--eval_hdf5_path",
        type=str,
        required=True,
        help="Path to evaluation HDF5 file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        required=True,
        help="Path to template file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=100,
        help="Number of samples to use for evaluation",
    )
    # Add other training arguments from original script
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
    return parser.parse_args()

def extract_number_from_text(text: str) -> float:
    """Extract the numerical value from the model's output text."""
    try:
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        return float('nan')
    except:
        return float('nan')

def evaluate_model(
    model: Qwen2VLForConditionalGeneration,
    eval_dataset: Qwen2VLDataset,
    processor,
    task_type: str,
    batch_size: int,
    accelerator: Accelerator
) -> Tuple[float, float]:
    """
    Evaluate the model on a specific task.
    Returns both MSE and R2 score.
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    # Set dataset for single task evaluation
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
    
    print("evaluate on task {}".format(task_type))
    # import pudb;pu.db;
    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = batch['processed_inputs']
            
            # Generate outputs
            output = accelerator.unwrap_model(model)(
                **inputs)
            
            decoded_outputs = output.num.detach().cpu().tolist()
            
            # Extract ground truth values
            for ans in batch['answers'].detach().cpu().tolist():
                ground_truths.extend(ans)
                # print(ans)
            
            # Extract predicted values
            for output in decoded_outputs:
                # print(output)
                pred_value = output
                predictions.append(pred_value)
    
    # Filter out NaN values
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) 
                   if not (np.isnan(p) or np.isnan(g))]
    print(valid_pairs)
    
    if not valid_pairs:
        return float('inf'), float('nan')
    
    pred_array = np.array([p for p, _ in valid_pairs])
    gt_array = np.array([g for _, g in valid_pairs])
    
    # Calculate MSE
    mse = np.mean((pred_array - gt_array) ** 2)
    
    # Calculate R² score
    ss_res = np.sum((gt_array - pred_array) ** 2)
    ss_tot = np.sum((gt_array - np.mean(gt_array)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    
    return mse, r2

def get_optimizer(model, args):
    # 将参数分成两组
    num_head_params = []
    other_params = []
    
    # 遍历模型参数
    for name, param in model.named_parameters():
        if 'num_head.weight' in name:
            num_head_params.append(param)
        else:
            other_params.append(param)
    
    # 创建参数组，为num_head.weight使用更大的学习率
    param_groups = [
        {
            'params': other_params,
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay
        },
        {
            'params': num_head_params,
            'lr': args.learning_rate * 2,  # 10倍于基础学习率
            'weight_decay': args.weight_decay
        }
    ]
    
    # 初始化optimizer
    optimizer = torch.optim.AdamW(param_groups)
    
    return optimizer

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
    
    # Initialize logging
    accelerator.init_trackers(
        project_name="qwen2vl_training",
        config=vars(args)
    )
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    set_seed(args.seed)
    
    # Load processor and model
    min_pixels = 110*110*3
    max_pixels = 110*110*3
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    model = AstroQwen2VL.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )
    
    # Initialize datasets
    train_dataset = Qwen2VLDataset(
        hdf5_path=args.train_hdf5_path,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        split="train",
        num_questions=1,
        max_length=256
    )
    
    eval_dataset = Qwen2VLDataset(
        hdf5_path=args.eval_hdf5_path,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        split="eval",
        num_questions=1,
        max_length=256,
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
    
    # Initialize optimizer
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay
    # )

    optimizer = get_optimizer(model, args)
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Initialize learning rate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Tasks to evaluate
    eval_tasks = [
        'task1_redshift',
        'task2_log_mstar',
        'task2_z_mw',
        'task2_sSFR',
        'task2_tage_mw'
    ]
    
    # Training loop
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
                
                # Log training metrics
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
                
                # Evaluate model
                if global_step % args.eval_steps == 0:
                    logger.info(f"Evaluating at step {global_step}...")
                    model.eval()
                    task_metrics = {}
                    
                    for task in eval_tasks:
                        mse, r2 = evaluate_model(
                            model=model,
                            eval_dataset=eval_dataset,
                            processor=processor,
                            task_type=task,
                            batch_size=args.eval_batch_size,
                            accelerator=accelerator
                        )
                        task_metrics[task] = {'mse': mse, 'r2': r2}
                        
                        # Log metrics under 'eval' category
                        accelerator.log({
                            f"eval/metrics/{task}/mse": mse,
                            f"eval/metrics/{task}/r2": r2
                        }, step=global_step)
                    
                    # Calculate and log average metrics
                    avg_mse = np.mean([metrics['mse'] for metrics in task_metrics.values()])
                    avg_r2 = np.mean([metrics['r2'] for metrics in task_metrics.values() 
                                    if not np.isnan(metrics['r2'])])
                    
                    accelerator.log({
                        "eval/average/mse": avg_mse,
                        "eval/average/r2": avg_r2
                    }, step=global_step)
                    
                    # Save best model based on both metrics
                    if (avg_mse < best_avg_mse or avg_r2 > best_avg_r2) and accelerator.is_main_process:
                        best_avg_mse = min(best_avg_mse, avg_mse)
                        best_avg_r2 = max(best_avg_r2, avg_r2)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            os.path.join(args.output_dir, "best_model"),
                            save_function=accelerator.save
                        )
                    
                    model.train()
                        
                # Save regular checkpoint
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
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