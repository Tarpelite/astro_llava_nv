import os
import argparse
import logging
from typing import Dict


import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, MllamaForConditionalGeneration, get_linear_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed,ProjectConfiguration
from tqdm.auto import tqdm

from dataset_vision_only  import MLLamaDataset

torch.set_float32_matmul_precision('high')

# 设置基础日志配置
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("training.log")  # 同时保存到文件
    ]
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MLLama cross-attention layers")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/model",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="/path/to/train.hdf5",
        help="Path to HDF5 data file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/path/to/images",
        help="Directory containing images",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="template_vision_only.json",
        help="Path to template file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to apply",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Number of steps between logging updates",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of steps between model saves",
    )
    return parser.parse_args()

def train():
    args = parse_args()
     # 配置 ProjectConfiguration 用于正确设置日志
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.output_dir,
    )
    # 初始化 accelerator 时添加日志配置
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=["tensorboard", "all"],  # 同时启用所有日志
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    # 启动 accelerator 日志系统
    accelerator.init_trackers(
        project_name="mllama_training",
        config=vars(args),
        init_kwargs={
            "tensorboard": {
                "flush_secs": 120,  # 每120秒刷新一次tensorboard
            }
        }
    )

    
    # Make output directory if it doesn't exist
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
   

    # Set seed
    set_seed(args.seed)
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = MllamaForConditionalGeneration.from_pretrained(args.model_path)
    
    # Freeze layers
    cross_attention_layers = model.config.text_config.cross_attention_layers
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        param.requires_grad = False
        for x in cross_attention_layers:
            if str(x) in name.split(".") and ("vision_model" not in name.split(".")):
                param.requires_grad = True
                trainable_params += param.numel()
                if accelerator.is_main_process:
                    logger.info(f"Training parameter: {name}")
                break


    if accelerator.is_main_process:
        logger.info(f"Total parameters: {all_params}")
        logger.info(f"Trainable parameters: {trainable_params}")
        logger.info(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%")

    # Initialize dataset and dataloader
    train_dataset = MLLamaDataset(
        hdf5_path=args.hdf5_path,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        split="train",
        num_questions=1,
        max_length=512
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=MLLamaDataset.collate_fn,
        num_workers=4
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Calculate number of training steps
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
    
    # Initialize training state
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Training loop
    global_step = 0
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
                total_loss += loss.detach().float()
                print(loss)
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                global_step += 1
                
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
                
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                            save_function=accelerator.save
                        )
        
        progress_bar.close()
        
        # Save model at the end of each epoch
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}"),
                save_function=accelerator.save
            )
    
    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(args.output_dir, "final_model"),
            save_function=accelerator.save
        )
    
    # 关闭 trackers
    accelerator.end_training()

if __name__ == "__main__":
    train()