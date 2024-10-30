import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoProcessor, MllamaForConditionalGeneration
from tqdm import tqdm
from dataset_vision_only import MLLamaDataset
import logging
import re
from typing import Dict, List, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLLama model on different tasks")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model",
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
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    return parser.parse_args()

def extract_number_from_text(text: str) -> float:
    """Extract the numerical value from the model's output text."""
    # Assuming the format is something like "The value is [NUMBER]"
    try:
        # Look for a decimal number in the text
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        return float('nan')
    except:
        return float('nan')

def decode_outputs(processor, output_ids: torch.Tensor) -> List[str]:
    """Decode the model output ids to text."""
    decoded_outputs = []
    for output_id in output_ids:
        # Remove padding tokens
        output_id = output_id[output_id != processor.tokenizer.pad_token_id]
        decoded_text = processor.decode(output_id, skip_special_tokens=True)
        decoded_outputs.append(decoded_text)
    return decoded_outputs

def evaluate_by_task(
    model: MllamaForConditionalGeneration,
    processor,
    eval_dataset: MLLamaDataset,
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
        collate_fn=MLLamaDataset.collate_fn,
        num_workers=4
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {task_type}"):
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch['processed_inputs'].items()}
            
            # Generate outputs using greedy decoding
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs['pixel_values'],
                aspect_ratio_ids=inputs['aspect_ratio_ids'],  # Added this
                aspect_ratio_mask=inputs['aspect_ratio_mask'],  # Added this
                max_new_tokens=10,
                num_beams=1,  # greedy decoding
                do_sample=False,
                output_scores=False,
                return_dict_in_generate=True,
            ).sequences
            
            # Decode outputs
            decoded_outputs = decode_outputs(processor, generated_ids)

            # import pudb;pu.db
            
            # Extract ground truth values from the original text sequences
            for ans in batch['answers']:
                # ground_truth = extract_number_from_text(ans)  # [0] because num_questions=1
                ground_truths.append(ans[0])
                print("ground_truths", ans[0])
            
            # Extract predicted values
            for output in decoded_outputs:
                # import pudb;pu.db;
                print(output)
                pred_value = extract_number_from_text(output)
                predictions.append(pred_value)
                print("pred_values", pred_value)
                
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
    model = MllamaForConditionalGeneration.from_pretrained(args.model_path).bfloat16()
    model.to(device)
    
    # Initialize dataset
    eval_dataset = MLLamaDataset(
        hdf5_path=args.hdf5_path,
        image_dir=args.image_dir,
        template_path=args.template_path,
        processor=processor,
        split="eval",
        num_questions=1,  # Will be set per task
        max_length=512,
        max_samples=100
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