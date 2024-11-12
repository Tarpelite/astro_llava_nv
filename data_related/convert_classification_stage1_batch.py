import os
import h5py
import numpy as np
from astropy.table import Table
from tqdm import tqdm
from collections import defaultdict
from typing import List

def combine_and_reorganize_classification_data(input_files: List[str], output_path: str):
    """
    重新组织多个分类数据集文件，将多分类任务拆分为单独的样本
    
    Args:
        input_files: 输入文件路径列表
        output_path: 输出文件路径
    """
    # 定义任务字段
    task_fields = [
        'smooth', 'disk-edge-on', 'spiral-arms', 'bar', 
        'bulge-size', 'how-rounded', 'edge-on-bulge', 
        'spiral-winding', 'spiral-arm-count', 'merging'
    ]
    
    # 准备新的数据结构
    new_data = defaultdict(list)
    stats = defaultdict(lambda: defaultdict(int))
    
    # 处理每个输入文件
    for input_file in input_files:
        print(f"\nProcessing file: {input_file}")
        data = Table.read(input_file)
        
        print("Processing samples...")
        for row in tqdm(data):
            # 获取基础信息
            iauname = row['iauname']
            image = row['image']
            sph_emb = row['sphembeddings']
            hyp_emb = row['hypembeddings']
            euc_emb = row['eucembeddings']
            
            # 对每个任务字段处理
            for task in task_fields:
                class_label = int(row[task])
                if class_label != -1:  # 只处理有效的分类
                    new_data['iauname'].append(iauname)
                    new_data['image'].append(image)
                    new_data['task_type'].append(task)
                    new_data['class'].append(class_label)
                    new_data['sphembeddings'].append(sph_emb)
                    new_data['hypembeddings'].append(hyp_emb)
                    new_data['eucembeddings'].append(euc_emb)
                    
                    # 统计类别分布
                    stats[task][class_label] += 1
    
    # 转换为numpy数组，特别处理字符串类型
    processed_data = {
        'iauname': np.array(new_data['iauname'], dtype='S64'),
        'image': np.stack(new_data['image']),
        'task_type': np.array(new_data['task_type'], dtype='S32'),
        'class': np.array(new_data['class'], dtype=np.int32),
        'sphembeddings': np.stack(new_data['sphembeddings']),
        'hypembeddings': np.stack(new_data['hypembeddings']),
        'eucembeddings': np.stack(new_data['eucembeddings'])
    }
    
    # 保存到新的hdf5文件
    print("\nSaving processed data...")
    with h5py.File(output_path, 'w') as f:
        for key, value in processed_data.items():
            if key in ['iauname', 'task_type']:
                dt = h5py.special_dtype(vlen=str)
                dset = f.create_dataset(key, shape=(len(value),), dtype=dt)
                dset[:] = [v.decode('utf-8') if isinstance(v, bytes) else v for v in value]
            else:
                f.create_dataset(key, data=value)
    
    # 打印统计信息
    print("\nDataset statistics:")
    print(f"Total samples: {len(processed_data['iauname'])}")
    print("\nSamples per task and class:")
    for task in task_fields:
        total_task_samples = sum(stats[task].values())
        if total_task_samples > 0:
            print(f"\n{task}:")
            print(f"Total samples: {total_task_samples}")
            for class_label in sorted(stats[task].keys()):
                count = stats[task][class_label]
                percentage = (count / total_task_samples) * 100
                print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")

def verify_processed_data(output_path: str):
    """
    验证处理后的数据集
    """
    print("\nVerifying processed data...")
    with h5py.File(output_path, 'r') as f:
        required_fields = ['iauname', 'image', 'task_type', 'class', 
                         'sphembeddings', 'hypembeddings', 'eucembeddings']
        for field in required_fields:
            assert field in f, f"Missing field: {field}"
        
        n_samples = len(f['iauname'])
        for field in required_fields:
            assert len(f[field]) == n_samples, \
                f"Inconsistent number of samples in {field}: {len(f[field])} vs {n_samples}"
        
        print("Data verification completed successfully!")

if __name__ == "__main__":
    base_dir = "/mnt/data/CVPR2025/task1_data/data1109/final"
    
    # 处理训练集
    print("Processing training set...")
    train_input_files = [
        os.path.join(base_dir, f"classifications_train_{str(i).zfill(4)}.hdf5")
        for i in range(1, 4)  # 1到3的文件
    ]
    train_output_path = os.path.join(base_dir, "train_processed.hdf5")
    combine_and_reorganize_classification_data(train_input_files, train_output_path)
    verify_processed_data(train_output_path)
    
    # 处理测试集
    print("\nProcessing test set...")
    test_input_files = [
        os.path.join(base_dir, f"classifications_test_{str(i).zfill(4)}.hdf5")
        for i in range(1, 4)  # 1到3的文件
    ]
    test_output_path = os.path.join(base_dir, "test_processed.hdf5")
    combine_and_reorganize_classification_data(test_input_files, test_output_path)
    verify_processed_data(test_output_path)