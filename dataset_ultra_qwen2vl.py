import os
import json
from PIL import Image
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.table import Table
import random
from transformers import AutoProcessor
import numpy as np
from qwen_vl_utils import process_vision_info
from collections import defaultdict
import h5py

class Qwen2VLBaseDataset(Dataset):
    """Base dataset class with common functionality"""
    def __init__(
        self,
        template_path: str,
        processor,
        max_length: int = 512,
    ):
        self.processor = processor
        self.max_length = max_length
        
        # Load question templates
        with open(template_path, 'r') as f:
            self.templates = json.load(f)
            
        self.regression_tasks = [
            'task1_redshift',
            'task2_log_mstar',
            'task2_z_mw',
            'task2_tage_mw',
            'task2_sSFR'
        ]
        
        self.classification_tasks = [
            'task3_smooth',
            'task3_disk_edge_on',
            'task3_spiral_arms',
            'task3_bar',
            'task3_bulge_size',
            'task3_how_rounded',
            'task3_edge_on_bulge',
            'task3_spiral_winding',
            'task3_spiral_arm_count',
            'task3_merging'
        ]
        
        self.answer_mapping = {
            'task1_redshift': 'redshift',
            'task2_log_mstar': 'LOG_MSTAR',
            'task2_z_mw': 'Z_MW',
            'task2_tage_mw': 'TAGE_MW',
            'task2_sSFR': 'sSFR'
        }

    def _load_regression_image(self, image_dir: str, target_id: str) -> Image.Image:
        image_path = os.path.join(image_dir, f"{target_id}.png")
        return Image.open(image_path)
    
    def _load_classification_image(self, image_data: np.ndarray) -> Image.Image:
        img_array_transposed = np.transpose(image_data, (1, 2, 0))
        if img_array_transposed.max() <= 1.0:
            img_array_transposed = (img_array_transposed * 255).astype(np.uint8)
        else:
            img_array_transposed = img_array_transposed.astype(np.uint8)
        return Image.fromarray(img_array_transposed)
    
    def _get_value(self, row_data: Dict, key: str) -> float:
        value = row_data[key]
        if isinstance(value, np.ndarray):
            value = value.item()
        return float(value)
    
    def _construct_messages(self, image: Image.Image, task_type: str, row_data: Dict, is_train: bool) -> Tuple[List[Dict], float]:
        template = self.templates[task_type]
        
        if task_type in self.regression_tasks:
            answer_key = self.answer_mapping[task_type]
            answer = self._get_value(row_data, answer_key)
            # answer_text = f"{answer:.6f}"
            answer_text = " num"
        else:
            task_key = task_type.replace('task3_', '').replace("_", "-")
            answer = int(row_data[task_key])
            if answer == -1:  # Skip invalid classifications
                return None, None
            answer_text = chr(97 + answer)  # Convert 0->a, 1->b, 2->c, etc.

        if is_train:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": template}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": template}
                    ]
                }
            ]
        
        return messages, answer

    def process_inputs(self, messages: List[Dict], features: Dict, is_train: bool) -> Dict:
        add_generation_prompt = False if is_train else True
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "").replace(
            " <|image_pad|> ", "<|vision_start|><|image_pad|><|vision_end|>"
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        processor_kwargs = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "return_tensors": "pt",
        }
        
        if is_train:
            processor_kwargs["text_kwargs"] = {
                "max_length": self.max_length,
                "padding": "max_length",
                "padding_side": "right",
                "truncation": True
            }
            
        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Add features
        for feat_name, feat_value in features.items():
            if feat_value is not None:
                inputs[feat_name] = torch.tensor(feat_value)
                
        return inputs, text

class Qwen2VLTrainingDataset(Qwen2VLBaseDataset):
    def __init__(
        self,
        hdf5_paths: Dict[str, str],
        image_dir: str,
        template_path: str,
        processor,
        num_questions: int = 1,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        super().__init__(template_path, processor, max_length)
        
        self.image_dir = image_dir
        self.num_questions = num_questions
        self.max_samples = max_samples
        
        # 区分两种数据格式的读取
        # 回归数据使用 astropy Table
        self.regression_data = Table.read(hdf5_paths["regression"]) if "regression" in hdf5_paths else None
        # 分类数据使用 h5py
        self.classification_data = h5py.File(hdf5_paths["classification"], 'r') if "classification" in hdf5_paths else None
        
        # 获取数据集长度
        self.regression_len = len(self.regression_data) if self.regression_data is not None else 0
        self.classification_len = len(self.classification_data['iauname']) if self.classification_data is not None else 0
        
        # Combine all available tasks
        self.task_types = []
        if self.regression_data is not None:
            self.task_types.extend(self.regression_tasks)
        if self.classification_data is not None:
            self.task_types.extend(self.classification_tasks)

    def __len__(self) -> int:
        total_length = self.regression_len + self.classification_len
        if self.max_samples is not None:
            return min(self.max_samples, total_length)
        return total_length

    def __getitem__(self, idx: int) -> Dict:
        if idx < self.regression_len:
            # 使用原来的回归数据获取方式
            row = self.regression_data[idx]
            target_id = str(row['TARGETID'])
            image = self._load_regression_image(self.image_dir, target_id)
            features = {
                'euc_features': row["eucembeddings"],
                'hyp_features': row["hypembeddings"],
                'sph_features': row["sphembeddings"],
                'spec_features': row["spectrum_feature"] if "spectrum_feature" in row.colnames else None
            }
            available_tasks = [t for t in self.task_types if t in self.regression_tasks]
            task = random.choice(available_tasks)
            messages, answer = self._construct_messages_regression(image, task, row)
            task_type_id = 0 # 回归任务
        else:
            # 使用新的分类数据获取方式
            idx = idx - self.regression_len
            # 从h5py文件中读取数据
            target_id = str(self.classification_data['iauname'][idx].decode('utf-8'))
            image_data = self.classification_data['image'][idx]
            image = self._load_classification_image(image_data)
            task_type = str(self.classification_data['task_type'][idx].decode('utf-8'))
            class_label = int(self.classification_data['class'][idx])
            
            features = {
                'euc_features': self.classification_data["eucembeddings"][idx],
                'hyp_features': self.classification_data["hypembeddings"][idx],
                'sph_features': self.classification_data["sphembeddings"][idx],
                'spec_features': None
            }
            
            task_type = f"task3_{task_type}"
            messages = self._construct_messages_classification(image, task_type, task_type, class_label)
            answer = None
            task_type_id = 1 # 分类任务

        # Process inputs
        inputs, text = self.process_inputs(messages, features, is_train=True)
        inputs["task_type_id"] = torch.tensor(task_type_id)
            
        return {
            'target_id': target_id,
            'processed_inputs': [inputs],
            'text_sequences': [text],
            'answers': [answer],
            "task_type":[task_type_id],
            'raw_image': image
        }

    def _construct_messages_regression(self, image, task, row):
        template = self.templates[task]
        answer_key = self.answer_mapping[task]
        answer = self._get_value(row, answer_key)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": template}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": " num"}]
            }
        ]
        return messages, answer

    def _construct_messages_classification(self, image, task_type, task_key, class_label):
        template = self.templates[task_type.replace("-", "_")]
        answer_text = chr(97 + class_label)  # Convert 0->a, 1->b, etc.
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": template}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "({})".format(answer_text)}]
            }
        ]
        return messages

    def __del__(self):
        # 只需要关闭h5py文件
        if hasattr(self, 'classification_data') and self.classification_data is not None:
            self.classification_data.close()






class Qwen2VLEvaluationDataset(Qwen2VLBaseDataset):
   def __init__(
       self,
       hdf5_paths: Dict[str, str],
       image_dir: str,
       template_path: str,
       processor,
       max_length: int = 512,
       max_regression_samples: Optional[int] = None
   ):
       super().__init__(template_path, processor, max_length)
       
       self.image_dir = image_dir
       self.max_regression_samples = max_regression_samples
       
       # 区分两种数据格式的读取
       # 回归数据使用 astropy Table
       self.regression_data = Table.read(hdf5_paths["regression"]) if "regression" in hdf5_paths else None
       # 分类数据使用 h5py
       self.classification_data = h5py.File(hdf5_paths["classification"], 'r') if "classification" in hdf5_paths else None
       
       # Prepare evaluation samples
       self._prepare_evaluation_samples()

   def _prepare_evaluation_samples(self):
       self.eval_samples = []
       
       # Process regression samples with limit
       if self.regression_data is not None:
           regression_indices = list(range(len(self.regression_data)))
           if self.max_regression_samples:
               regression_indices = regression_indices[:self.max_regression_samples]
           
           for idx in regression_indices:
               for task in self.regression_tasks:
                   self.eval_samples.append(('regression', idx, task))
       
       # 处理分类样本 - 现在每个样本都是一个任务
       if self.classification_data is not None:
           for idx in range(len(self.classification_data['iauname'])):
               task_type = str(self.classification_data['task_type'][idx].decode('utf-8'))
               task = f"task3_{task_type}"
               self.eval_samples.append(('classification', idx, task))

   def __len__(self) -> int:
       return len(self.eval_samples)

   def __getitem__(self, idx: int) -> Dict:
       data_type, data_idx, task = self.eval_samples[idx]
       
       if data_type == 'regression':
           # 使用原来的回归数据获取方式
           row = self.regression_data[data_idx]
           target_id = str(row['TARGETID'])
           image = self._load_regression_image(self.image_dir, target_id)
           features = {
               'euc_features': row["eucembeddings"],
               'hyp_features': row["hypembeddings"],
               'sph_features': row["sphembeddings"],
               'spec_features': row["spectrum_feature"] if "spectrum_feature" in row.colnames else None
           }
           messages, answer = self._construct_messages_regression(image, task, row)
           task_type_id = 0
           
       else:  # classification
           # 从h5py文件中读取分类数据
           target_id = str(self.classification_data['iauname'][data_idx].decode('utf-8'))
           image_data = self.classification_data['image'][data_idx]
           image = self._load_classification_image(image_data)
           class_label = int(self.classification_data['class'][data_idx])
           
           features = {
               'euc_features': self.classification_data["eucembeddings"][data_idx],
               'hyp_features': self.classification_data["hypembeddings"][data_idx],
               'sph_features': self.classification_data["sphembeddings"][data_idx],
               'spec_features': None
           }
           
           messages = self._construct_messages_classification(image, task, task, class_label)
           answer = class_label
           task_type_id = 1
       
       inputs, text = self.process_inputs(messages, features, is_train=False)
       inputs["task_type_id"] = torch.tensor(task_type_id)

       return {
           'target_id': target_id,
           'processed_inputs': [inputs],
           'text_sequences': [text],
           'answers': [answer],
           'raw_image': image,
           'task_type': task
       }

   def _construct_messages_regression(self, image, task, row):
       template = self.templates[task]
       answer_key = self.answer_mapping[task]
       answer = self._get_value(row, answer_key)
       
       messages = [
           {
               "role": "user",
               "content": [
                   {"type": "image", "image": image},
                   {"type": "text", "text": template}
               ]
           }
       ]
       return messages, answer

   def _construct_messages_classification(self, image, task_type, task_key, class_label):
       template = self.templates[task_type.replace("-", "_")]
       messages = [
           {
               "role": "user",
               "content": [
                   {"type": "image", "image": image},
                   {"type": "text", "text": template}
               ]
           }
       ]
       return messages

   def __del__(self):
       # 只需要关闭h5py文件
       if hasattr(self, 'classification_data') and self.classification_data is not None:
           self.classification_data.close()

class Qwen2VLRegressionEvaluationDataset(Qwen2VLBaseDataset):
    def __init__(
        self,
        eval_dir: str,
        image_dir: str,
        template_path: str,
        processor,
        max_length: int = 512,
        max_samples_per_task: Optional[int] = None,
        selected_tasks: Optional[List[str]] = None
    ):
        super().__init__(template_path, processor, max_length)
        
        self.image_dir = image_dir
        self.max_samples_per_task = max_samples_per_task
        self.selected_tasks = selected_tasks
        
        # 加载每个任务的数据
        self.task_data = {}
        tasks_to_load = self.selected_tasks if self.selected_tasks is not None else self.regression_tasks
        
        for task in tasks_to_load:
            task_file = os.path.join(eval_dir, f"eval_{task}.hdf5")
            if os.path.exists(task_file):
                print(f"Loading data for task {task} from {task_file}")
                self.task_data[task] = Table.read(task_file)
                print(f"Loaded {len(self.task_data[task])} samples for {task}")
            else:
                print(f"Warning: No data file found for task {task} at {task_file}")
        
        # 准备评估样本
        self._prepare_evaluation_samples()

    def _prepare_evaluation_samples(self):
        self.eval_samples = []
        
        # 为每个已加载的任务准备样本
        for task, data in self.task_data.items():
            indices = list(range(len(data)))
            if self.max_samples_per_task:
                random.shuffle(indices)
                indices = indices[:self.max_samples_per_task]
            
            for idx in indices:
                self.eval_samples.append((task, idx))

    def __len__(self):
        return len(self.eval_samples)

    def __getitem__(self, idx: int) -> Dict:
        task, data_idx = self.eval_samples[idx]
        
        # 从对应任务的数据中获取样本
        row = self.task_data[task][data_idx]
        target_id = str(row['TARGETID'])
        
        # 读取图像
        image = self._load_regression_image(self.image_dir, target_id)
        
        # 获取目标值
        answer_key = self.answer_mapping[task]
        answer = self._get_value(row, answer_key)
        
        # 准备特征
        features = {
            'euc_features': row["eucembeddings"],
            'hyp_features': row["hypembeddings"],
            'sph_features': row["sphembeddings"],
            'spec_features': row["spectrum_feature"] if "spectrum_feature" in row.colnames else None
        }
        
        # 构建消息
        template = self.templates[task]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": template}
                ]
            }
        ]
        
        # 处理输入
        inputs, text = self.process_inputs(messages, features, is_train=False)
        inputs["task_type_id"] = torch.tensor(0)  # 回归任务

        return {
            'target_id': target_id,
            'processed_inputs': [inputs],
            'text_sequences': [messages[0]["content"][1]["text"]],
            'answers': [answer],
            'raw_image': image,
            'task_type': task
        }

class Qwen2VLClassificationEvaluationDataset(Qwen2VLBaseDataset):
    def __init__(
        self,
        eval_dir: str,
        image_dir: str,
        template_path: str,
        processor,
        max_length: int = 512,
        max_samples_per_task: Optional[int] = None,
        selected_tasks: Optional[List[str]] = None
    ):
        super().__init__(template_path, processor, max_length)
        
        self.image_dir = image_dir
        self.max_samples_per_task = max_samples_per_task
        self.selected_tasks = selected_tasks
        
        # 加载每个任务的数据
        self.task_data = {}
        tasks_to_load = self.selected_tasks if self.selected_tasks is not None else [
            task.replace('task3_', '') for task in self.classification_tasks
        ]
        
        for task in tasks_to_load:
            task_file = os.path.join(eval_dir, f"eval_task3_{task}.hdf5")
            if os.path.exists(task_file):
                print(f"Loading data for task {task} from {task_file}")
                self.task_data[task] = Table.read(task_file)
                print(f"Loaded {len(self.task_data[task])} samples for {task}")
            else:
                print(f"Warning: No data file found for task {task} at {task_file}")
        
        # 准备评估样本
        self._prepare_evaluation_samples()

    def _prepare_evaluation_samples(self):
        self.eval_samples = []
        
        # 为每个已加载的任务准备样本
        for task, data in self.task_data.items():
            # 过滤掉无效标签的样本
            # print(data.keys())
            # task="class"
            valid_indices = [i for i in range(len(data)) if data["class"][i] != -1]
            
            if self.max_samples_per_task and len(valid_indices) > self.max_samples_per_task:
                random.shuffle(valid_indices)
                valid_indices = valid_indices[:self.max_samples_per_task]
            
            for idx in valid_indices:
                self.eval_samples.append((f"task3_{task}", idx))

    def __getitem__(self, idx: int) -> Dict:
        task, data_idx = self.eval_samples[idx]
        task_name = task.replace('task3_', '')
        
        # 获取数据
        # print("task:{}".format(task))
        # print("task_name:{}".format(task_name))
        row = self.task_data[task_name][data_idx]
        target_id = str(row['iauname'])
        
        # 读取图像数据
        image = Image.fromarray(row['image'].transpose(1, 2, 0).astype(np.uint8))
        
        # 获取标签
        class_label = int(row["class"])
        
        # 准备特征
        features = {
            'euc_features': row["eucembeddings"],
            'hyp_features': row["hypembeddings"],
            'sph_features': row["sphembeddings"],
            'spec_features': None
        }
        
        # 构建消息
        template = self.templates[task.replace("-", "_")]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": template}
                ]
            }
        ]
        
        # 处理输入
        inputs, text = self.process_inputs(messages, features, is_train=False)
        inputs["task_type_id"] = torch.tensor(1)  # 分类任务

        return {
            'target_id': target_id,
            'processed_inputs': [inputs],
            'text_sequences': [text],
            'answers': [class_label],
            'raw_image': image,
            'task_type': task
        }

    def __len__(self) -> int:
        return len(self.eval_samples)



def collate_fn(batch: List[Dict]) -> Dict:
    target_ids = [item['target_id'] for item in batch]
    all_text_sequences = [item['text_sequences'] for item in batch]
    all_answers = [item['answers'] for item in batch]
    
    # Add task_type if it exists (for evaluation)
    task_types = [item.get('task_type') for item in batch if 'task_type' in item]
    
    all_processed_inputs = []
    for item in batch:
        all_processed_inputs.extend(item['processed_inputs'])
        
    batch_inputs = {
        'input_ids': torch.stack([inputs['input_ids'] for inputs in all_processed_inputs]),
        'attention_mask': torch.stack([inputs['attention_mask'] for inputs in all_processed_inputs]),
        'pixel_values': torch.stack([inputs['pixel_values'] for inputs in all_processed_inputs]),
        'image_grid_thw': torch.stack([inputs['image_grid_thw'] for inputs in all_processed_inputs]),
        "task_type_id": torch.stack([inputs["task_type_id"] for inputs in all_processed_inputs])
    }
    
    # Add optional features if they exist
    for feat_name in ['euc_features', 'hyp_features', 'sph_features', 'spec_features']:
        if all(feat_name in inputs for inputs in all_processed_inputs):
            batch_inputs[feat_name] = torch.stack([inputs[feat_name] for inputs in all_processed_inputs])

    labels = batch_inputs["input_ids"].clone()
    input_ids = batch_inputs["input_ids"]

    batch_size = input_ids.size(0)
    for i in range(batch_size):
        tokens = input_ids[i].tolist()
        
        try:
            answer_start = -1
            answer_end = -1
            for j in range(len(tokens) - 1):
                if tokens[j] == 151644 and tokens[j+1] == 77091:
                    answer_start = j + 2
                    for k in range(answer_start, len(tokens)):
                        if tokens[k] == 151645:
                            answer_end = k
                            break
                    break
                
            if answer_start != -1:
                labels[i, :answer_start] = -100
                labels[i, (answer_end+1):] = -100
            else:
                labels[i, :] = -100
                
        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
            labels[i, :] = -100

    batch_inputs["labels"] = labels
    batch_inputs["answers"] = all_answers
    

    output = {
        'target_ids': target_ids,
        'text_sequences': all_text_sequences,
        'answers': all_answers,
        'processed_inputs': batch_inputs
    }
    
    # Add task_types for evaluation if they exist
    if task_types:
        output['task_types'] = task_types
        
    return output

# 测试脚本
import os
from transformers import AutoProcessor
from torch.utils.data import DataLoader

def test_datasets():
    # Initialize processor
    min_pixels = 110*110*3
    max_pixels = 144*144*3
    processor = AutoProcessor.from_pretrained("/mnt/data/CVPR2025/task1_data/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    
    # 设置路径
    base_dir = "/mnt/data/CVPR2025/task1_data/"
    regression_train = os.path.join(base_dir, "train_no_classification_addfeat.hdf5")
    regression_test = os.path.join(base_dir, "test_no_classification_addfeat.hdf5")
    classification_train = os.path.join(base_dir, "classifications/train_no_classification_addfeat_task3.hdf5")
    classification_test = os.path.join(base_dir, "classifications/test_no_classification_addfeat_task3.hdf5")
    image_dir = os.path.join(base_dir, "images/images")
    template_path = "./template_ultra_qwen2vl_classification.json"
    
    # Test Training Dataset
    print("\nTesting Training Dataset...")
    train_dataset = Qwen2VLTrainingDataset(
        hdf5_paths={
            "regression": regression_train,
            "classification": classification_train
        },
        image_dir=image_dir,
        template_path=template_path,
        processor=processor,
        num_questions=1,
        max_length=512,
        max_samples=100  # 用于测试的小样本
    )
    
    print(f"Training dataset length: {len(train_dataset)}")
    
    # Test single training sample
    train_sample = train_dataset[0]
    print("\nTraining sample contents:")
    print(f"Target ID: {train_sample['target_id']}")
    print(f"Number of questions: {len(train_sample['processed_inputs'])}")
    print(f"Text sequences: {train_sample['text_sequences']}")
    print(f"Answers: {train_sample['answers']}")
    
    # Test Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print("\nTesting Training DataLoader:")
    train_batch = next(iter(train_loader))
    print(f"Batch keys: {train_batch.keys()}")
    print(f"Processed inputs keys: {train_batch['processed_inputs'].keys()}")
    print(f"Input IDs shape: {train_batch['processed_inputs']['input_ids'].shape}")
    
    # Test Evaluation Dataset
    print("\nTesting Evaluation Dataset...")
    eval_dataset = Qwen2VLEvaluationDataset(
        hdf5_paths={
            "regression": regression_test,
            "classification": classification_test
        },
        image_dir=image_dir,
        template_path=template_path,
        processor=processor,
        max_length=512,
        max_regression_samples=10  # 限制回归任务的样本数
    )
    
    print(f"Evaluation dataset length: {len(eval_dataset)}")
    
    # Test single evaluation sample
    eval_sample = eval_dataset[0]
    print("\nEvaluation sample contents:")
    print(f"Target ID: {eval_sample['target_id']}")
    print(f"Task type: {eval_sample['task_type']}")
    print(f"Text sequence: {eval_sample['text_sequences'][0]}")
    print(f"Answer: {eval_sample['answers'][0]}")
    
    # Test Evaluation DataLoader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,  # 评估时不打乱顺序
        collate_fn=collate_fn
    )
    
    print("\nTesting Evaluation DataLoader:")
    eval_batch = next(iter(eval_loader))
    print(f"Batch keys: {eval_batch.keys()}")
    print(f"Task types in batch: {eval_batch['task_types']}")
    
    # 测试按任务类型统计样本数
    print("\nSample counts by task type:")
    task_counts = defaultdict(int)
    for i in range(len(eval_dataset)):
        task_type = eval_dataset[i]['task_type']
        task_counts[task_type] += 1
    
    for task, count in task_counts.items():
        print(f"{task}: {count} samples")

if __name__ == "__main__":
    test_datasets()