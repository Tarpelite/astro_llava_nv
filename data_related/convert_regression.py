import os
import h5py
import numpy as np
from astropy.table import Table
from collections import defaultdict

def split_regression_dataset(
    input_hdf5_path: str,
    output_dir: str,
    task_mapping: dict = {
        'task1_redshift': 'redshift',
        'task2_log_mstar': 'LOG_MSTAR',
        'task2_z_mw': 'Z_MW',
        'task2_tage_mw': 'TAGE_MW',
        'task2_sSFR': 'sSFR'
    }
):
    """
    Split regression evaluation dataset into separate HDF5 files for each task.
    
    Args:
        input_hdf5_path: Path to the input HDF5 file containing all regression tasks
        output_dir: Directory where the split HDF5 files will be saved
        task_mapping: Dictionary mapping task names to their corresponding column names
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input HDF5 file using astropy Table
    print(f"Reading input file: {input_hdf5_path}")
    data = Table.read(input_hdf5_path)
    
    # Get common columns that should be included in all task-specific files
    common_columns = [
        'TARGETID',
        'eucembeddings',
        'hypembeddings',
        'sphembeddings'
    ]
    
    # Add spectrum_feature if it exists
    if 'spectrum_feature' in data.colnames:
        common_columns.append('spectrum_feature')
    
    # Process each regression task
    for task_name, column_name in task_mapping.items():
        if column_name not in data.colnames:
            print(f"Warning: Column {column_name} not found in data. Skipping task {task_name}")
            continue
            
        output_file = os.path.join(output_dir, f"eval_{task_name}.hdf5")
        print(f"Creating task-specific file for {task_name}: {output_file}")
        
        # Create a subset of the data with only the required columns
        task_columns = common_columns + [column_name]
        task_data = data[task_columns]
        
        # Save to new HDF5 file
        task_data.write(output_file, format='hdf5', path='data', overwrite=True)
        
        # Print basic statistics
        print(f"  - Number of samples: {len(task_data)}")
        print(f"  - Target column stats ({column_name}):")
        print(f"    - Mean: {np.mean(task_data[column_name]):.4f}")
        print(f"    - Std: {np.std(task_data[column_name]):.4f}")
        print(f"    - Min: {np.min(task_data[column_name]):.4f}")
        print(f"    - Max: {np.max(task_data[column_name]):.4f}")
        print()

def verify_split_files(
    output_dir: str,
    task_mapping: dict,
    original_file: str
):
    """
    Verify that the split files contain the correct data and match the original file.
    
    Args:
        output_dir: Directory containing the split HDF5 files
        task_mapping: Dictionary mapping task names to their corresponding column names
        original_file: Path to the original HDF5 file
    """
    print("Verifying split files...")
    
    # Read original file
    original_data = Table.read(original_file)
    
    for task_name, column_name in task_mapping.items():
        task_file = os.path.join(output_dir, f"eval_{task_name}.hdf5")
        if not os.path.exists(task_file):
            print(f"Warning: File not found for task {task_name}")
            continue
            
        # Read task-specific file
        task_data = Table.read(task_file)
        
        # Verify number of samples
        assert len(task_data) == len(original_data), \
            f"Sample count mismatch for {task_name}"
            
        # Verify target values
        np.testing.assert_array_almost_equal(
            task_data[column_name],
            original_data[column_name],
            decimal=6,
            err_msg=f"Target values mismatch for {task_name}"
        )
        
        # Verify embeddings
        for embedding_type in ['eucembeddings', 'hypembeddings', 'sphembeddings']:
            np.testing.assert_array_almost_equal(
                task_data[embedding_type],
                original_data[embedding_type],
                decimal=6,
                err_msg=f"Embedding mismatch ({embedding_type}) for {task_name}"
            )
        
        print(f"Verification passed for {task_name}")

def main():
    # Configure paths
    base_dir = "/mnt/data/CVPR2025/task1_data/data1105/regression"
    input_file = os.path.join(base_dir, "test_no_classification_addfeat.hdf5")
    output_dir = os.path.join(base_dir)
    
    # Define task mapping
    task_mapping = {
        'task1_redshift': 'redshift',
        'task2_log_mstar': 'LOG_MSTAR',
        'task2_z_mw': 'Z_MW',
        'task2_tage_mw': 'TAGE_MW',
        'task2_sSFR': 'sSFR'
    }
    
    # Split the dataset
    split_regression_dataset(input_file, output_dir, task_mapping)
    
    # Verify the split files
    verify_split_files(output_dir, task_mapping, input_file)

if __name__ == "__main__":
    main()