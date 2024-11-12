import os
import numpy as np
from astropy.table import Table, vstack
from collections import defaultdict
from typing import Dict, List, Set
import shutil
from tqdm import tqdm

def combine_hdf5_files(input_files: List[str]) -> Table:
    """
    Combine multiple HDF5 files into a single Table.
    
    Args:
        input_files: List of paths to input HDF5 files
        
    Returns:
        Combined astropy Table
    """
    print("Combining input files...")
    tables = []
    for file_path in tqdm(input_files):
        try:
            table = Table.read(file_path)
            tables.append(table)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            continue
    
    if not tables:
        raise ValueError("No valid input files found")
    
    print("Stacking tables...")
    combined_table = vstack(tables)
    print(f"Combined data contains {len(combined_table)} samples")
    return combined_table

def split_classification_dataset(
    input_hdf5_paths: List[str],
    output_dir: str,
    classification_tasks: List[str] = [
        'smooth',
        'disk-edge-on',
        'spiral-arms',
        'bar',
        'bulge-size',
        'how-rounded',
        'edge-on-bulge',
        'spiral-winding',
        'spiral-arm-count',
        'merging'
    ]
):
    """
    Split classification evaluation dataset into separate HDF5 files for each task.
    Only keep samples where class != -1.
    
    Args:
        input_hdf5_paths: List of paths to the input HDF5 files containing all classification tasks
        output_dir: Directory where the split HDF5 files will be saved
        classification_tasks: List of classification task names
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(input_hdf5_paths)} input files...")
    # Combine all input files
    data = combine_hdf5_files(input_hdf5_paths)
    total_samples = len(data)
    
    # Common fields that should be included in each task-specific file
    common_fields = [
        'iauname', 'ra', 'dec', 'redshift', 'index', 'file', 'image', 
        'image_feature', 'sphembeddings', 'hypembeddings', 'eucembeddings'
    ]
    
    # Process each classification task
    task_stats = {}
    for task in classification_tasks:
        print(f"\nProcessing task: {task}")
        
        # Skip if task column doesn't exist
        if task not in data.colnames:
            print(f"Warning: Task {task} not found in dataset")
            continue
        
        # Filter out samples where class == -1
        valid_mask = data[task] != -1
        valid_data = data[valid_mask]
            
        # Create task-specific table
        task_table = Table()
        
        # Copy common fields for valid samples only
        for field in common_fields:
            if field in data.colnames:  # Only copy fields that exist
                task_table[field] = valid_data[field]
            else:
                print(f"Warning: Field {field} not found in dataset")
            
        # Add task-specific label column
        task_table['class'] = valid_data[task]
        
        # Calculate statistics
        class_counts = defaultdict(int)
        valid_samples = len(task_table)
        for label in task_table['class']:
            class_counts[label] += 1
        
        # Print statistics
        print(f"Statistics for {task}:")
        print(f"Total samples in original data: {total_samples}")
        print(f"Valid samples after filtering: {valid_samples}")
        print("Class distribution:")
        for label, count in sorted(class_counts.items()):
            percentage = (count / valid_samples) * 100 if valid_samples > 0 else 0
            print(f"  Class {label}: {count} samples ({percentage:.2f}%)")
        
        # Store statistics
        task_stats[task] = {
            'total_samples': total_samples,
            'valid_samples': valid_samples,
            'class_distribution': dict(class_counts)
        }
        
        # Save task-specific file
        output_file = os.path.join(output_dir, f"eval_task3_{task}.hdf5")
        print(f"Saving to: {output_file}")
        task_table.write(output_file, format='hdf5', path='data', overwrite=True)
        
    return task_stats

def verify_split_files(
    output_dir: str,
    original_files: List[str],
    classification_tasks: List[str]
):
    """
    Verify that the split files contain the correct data and match the original files,
    considering only valid samples (class != -1).
    
    Args:
        output_dir: Directory containing the split HDF5 files
        original_files: List of paths to the original HDF5 files
        classification_tasks: List of classification task names
    """
    print("\nVerifying split files...")
    
    # Read and combine original data
    original_data = combine_hdf5_files(original_files)
    
    for task in classification_tasks:
        if task not in original_data.colnames:
            print(f"Warning: Task {task} not found in original dataset")
            continue
            
        task_file = os.path.join(output_dir, f"eval_task3_{task}.hdf5")
        if not os.path.exists(task_file):
            print(f"Warning: File not found for task {task}")
            continue
            
        print(f"\nVerifying {task}...")
        task_data = Table.read(task_file)
        
        # Get valid samples from original data
        valid_mask = original_data[task] != -1
        valid_original = original_data[valid_mask]
        
        # Verify sample count
        assert len(task_data) == len(valid_original), \
            f"Sample count mismatch for {task}"
            
        # Verify common fields
        common_fields = [
            'iauname', 'ra', 'dec', 'redshift', 'index', 'file', 'image',
            'image_feature', 'sphembeddings', 'hypembeddings', 'eucembeddings'
        ]
        
        for field in common_fields:
            if field not in valid_original.colnames or field not in task_data.colnames:
                print(f"Warning: Field {field} not found in one or both datasets")
                continue
                
            try:
                np.testing.assert_array_equal(
                    task_data[field],
                    valid_original[field],
                    err_msg=f"Data mismatch in {field} for {task}"
                )
            except AssertionError as e:
                print(f"Warning: Verification failed for {field} in {task}")
                print(str(e))
                continue
        
        # Verify class labels
        np.testing.assert_array_equal(
            task_data['class'],
            valid_original[task],
            err_msg=f"Class label mismatch for {task}"
        )
        
        print(f"Verification passed for {task}")

def main():
    # Configure paths
    base_dir = "/mnt/data/CVPR2025/task1_data/data1109/final"
    
    # Define input files
    input_files = [
        os.path.join(base_dir, f"classifications_test_{str(i).zfill(4)}.hdf5")
        for i in range(1, 4)  # 1到3的文件
    ]
    output_dir = os.path.join(base_dir, "classification_eval_split")
    
    # Define classification tasks
    classification_tasks = [
        'smooth',
        'disk-edge-on',
        'spiral-arms',
        'bar',
        'bulge-size',
        'how-rounded',
        'edge-on-bulge',
        'spiral-winding',
        'spiral-arm-count',
        'merging'
    ]
    
    # Split the dataset
    task_stats = split_classification_dataset(input_files, output_dir, classification_tasks)
    
    # Save statistics to a file
    stats_file = os.path.join(output_dir, "task_statistics.txt")
    print(f"\nSaving statistics to {stats_file}")
    with open(stats_file, 'w') as f:
        for task, stats in task_stats.items():
            f.write(f"\n{task}:\n")
            f.write(f"Total samples: {stats['total_samples']}\n")
            f.write(f"Valid samples: {stats['valid_samples']}\n")
            f.write("Class distribution:\n")
            for label, count in sorted(stats['class_distribution'].items()):
                percentage = (count / stats['valid_samples']) * 100
                f.write(f"  Class {label}: {count} samples ({percentage:.2f}%)\n")
    
    # Verify the split files
    verify_split_files(output_dir, input_files, classification_tasks)

if __name__ == "__main__":
    main()