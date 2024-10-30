import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Tuple
import logging
import re
from dataset_vision_qwen2vl import Qwen2VLDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2VL model on different tasks")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Qwen2VL model",
    )
    parser.add_argument(
        "--hdf5_path",
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
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
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

def evaluate_by_task(
    args,
    model: Qwen2VLForConditionalGeneration,
    processor,
    eval_dataset: Dataset,
    task_type: str,
    batch_size: int,
    device: torch.device
) -> Tuple[float, List[Tuple[float, float]]]:
    """Evaluate the model on a specific task."""
    model.eval()
    predictions = []
    ground_truths = []
    pred_gt_pairs = []
    
    # Create dataloader for single task
    eval_dataset.num_questions = 1
    eval_dataset.task_types = [task_type]
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=args.num_workers
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {task_type}"):
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch['processed_inputs'].items()}
            
            import pudb;pu.db;
            # Generate outputs
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                num_beams=1,
                do_sample=False,
            )
            
            # Trim generated ids to only include new tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # Decode outputs
            decoded_outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Extract ground truth values
            for ans in batch['answers']:
                ground_truths.append(ans[0])
            
            # Extract predicted values
            for output in decoded_outputs:
                pred_value = extract_number_from_text(output)
                print(output)
                predictions.append(pred_value)
                print(f"Prediction: {pred_value}")
                
    # Filter out any NaN values
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) 
                   if not (np.isnan(p) or np.isnan(g))]
    
    if not valid_pairs:
        logger.warning(f"No valid predictions for task {task_type}")
        return float('inf'), []
    
    pred_array = np.array([p for p, _ in valid_pairs])
    gt_array = np.array([g for _, g in valid_pairs])
    
    # Calculate MSE
    mse = np.mean((pred_array - gt_array) ** 2)
    
    return mse, valid_pairs

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )
    model.to(device)
    
    # Initialize dataset
    min_pixels = 110*110*3
    max_pixels = 110*110*3
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    
    eval_dataset = Qwen2VLDataset(
        hdf5_path=args.hdf5_path,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        split="eval",
        num_questions=1,
        max_length=512,
        max_samples=None
    )
    
    # Tasks to evaluate
    tasks = [
        'task1_redshift',
        'task2_log_mstar',
        'task2_z_mw',
        'task2_sSFR',
        'task2_tage_mw'
    ]
    
    # Evaluate each task
    results = {}
    all_predictions = {}
    
    for task in tasks:
        logger.info(f"\nEvaluating task: {task}")
        mse, pred_gt_pairs = evaluate_by_task(
            args,
            model=model,
            processor=processor,
            eval_dataset=eval_dataset,
            task_type=task,
            batch_size=args.batch_size,
            device=device
        )
        
        results[task] = mse
        all_predictions[task] = pred_gt_pairs
        
        logger.info(f"MSE for {task}: {mse:.6f}")
        
        # Save predictions for this task
        with open(os.path.join(args.output_dir, f"{task}_predictions.txt"), 'w') as f:
            f.write(f"Predicted\tGround Truth\n")
            for pred, gt in pred_gt_pairs:
                f.write(f"{pred:.6f}\t{gt:.6f}\n")
    
    # Save overall results
    with open(os.path.join(args.output_dir, "evaluation_results.txt"), 'w') as f:
        f.write("Task\tMSE\n")
        for task, mse in results.items():
            f.write(f"{task}\t{mse:.6f}\n")
    
    logger.info("\nEvaluation Complete!")
    logger.info("Summary of Results:")
    for task, mse in results.items():
        logger.info(f"{task}: MSE = {mse:.6f}")

if __name__ == "__main__":
    main()