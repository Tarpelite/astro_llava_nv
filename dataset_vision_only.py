import os
import json
from PIL import Image
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.table import Table
import random
from transformers import AutoProcessor, MllamaForConditionalGeneration
import numpy as np

class MLLamaDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        image_dir: str,
        template_path: str,
        processor,
        split: str = "train",
        num_questions: int = 5,
        max_length: int = 512,
        max_samples=None
    ):
        self.split = split
        self.image_dir = image_dir
        self.processor = processor
        self.num_questions = num_questions
        self.max_length = max_length
        self.max_samples = max_samples
        
        # 加载元数据
        self.data = Table.read(hdf5_path)
        
        # 加载问题模板
        with open(template_path, 'r') as f:
            self.templates = json.load(f)
            
        self.task_types = list(self.templates.keys())
        
        self.answer_mapping = {
            'task1_redshift': 'redshift',
            'task2_log_mstar': 'LOG_MSTAR',
            'task2_z_mw': 'Z_MW',
            'task2_tage_mw': 'TAGE_MW',
            'task2_sSFR': 'sSFR'
        }
        
    def __len__(self) -> int:
        if self.max_samples is not None:
            return self.max_samples
        else:
            return len(self.data)
    
    def _load_image(self, target_id: str) -> Image.Image:
        image_path = os.path.join(self.image_dir, f"{target_id}.png")
        return Image.open(image_path)
    
    def _get_value(self, row_data: Dict, key: str) -> float:
        value = row_data[key]
        if isinstance(value, np.ndarray):
            value = value.item()
        return float(value)
    
    def _construct_text_sequence(self, row_data: Dict, task_type: str) -> str:
        template = self.templates[task_type]  
        answer_key = self.answer_mapping[task_type]
        answer = self._get_value(row_data, answer_key)
        if self.split == "train":
            text = template.replace("[ANS]", f"{answer:.6f}")
        else:
            text = template.replace("[ANS]", "")
        return text, answer
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.data[idx]
        target_id = str(row['TARGETID'])
        
        # 加载图片
        image = self._load_image(target_id)
        # print(image.size)
        
        # 随机选择任务类型
        selected_tasks = random.sample(self.task_types, self.num_questions)
        
        # 构造文本序列
        text_sequences = []
        answers = []
        for task in selected_tasks:
            text, ans = self._construct_text_sequence(row, task)
            text_sequences.append(text)
            answers.append(ans)
        
        # 使用processor处理输入
        processed_samples = []
        padding_side = "left" if self.split == "eval" else "right"
        for text, ans in zip(text_sequences, answers):
            inputs = self.processor(
                images=image,
                text=text,
                text_kwargs={
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": self.max_length,
                    "return_tensors": "pt",
                    "padding_side": padding_side
                },
                # images_kwargs={
                #     "size": {"height": 110, "width": 110}  # 设置图片尺寸
                # },
                common_kwargs={
                    "return_tensors": "pt"
                }
            )
            
            # 移除batch维度
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            processed_samples.append(inputs)
        
        # import pudb;pu.db;
        # print(text_sequences)
            
        return {
            'target_id': target_id,
            'processed_inputs': processed_samples,
            'text_sequences': text_sequences,
            "answers": answers,
            'raw_image': image
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        target_ids = [item['target_id'] for item in batch]
        all_text_sequences = [item['text_sequences'] for item in batch]
        all_answers = [item["answers"] for item in batch]
        
        all_processed_inputs = []
        for item in batch:
            all_processed_inputs.extend(item['processed_inputs'])
            # print(item['processed_inputs'][0].keys())
            
        batch_inputs = {
            'input_ids': torch.stack([inputs['input_ids'] for inputs in all_processed_inputs]),
            # 'labels': torch.stack([inputs['input_ids'] for inputs in all_processed_inputs]),
            'attention_mask': torch.stack([inputs['attention_mask'] for inputs in all_processed_inputs]),
            'pixel_values': torch.stack([inputs['pixel_values'] for inputs in all_processed_inputs]),
            'aspect_ratio_ids': torch.stack([inputs['aspect_ratio_ids'] for inputs in all_processed_inputs]),
            'aspect_ratio_mask': torch.stack([inputs['aspect_ratio_mask'] for inputs in all_processed_inputs]),
        }

        labels = batch_inputs["input_ids"].clone()
        input_ids = batch_inputs["input_ids"]
        # 对每个序列找到"A:"的位置并设置labels
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # 将input_ids转换为列表以便搜索
            tokens = input_ids[i].tolist()
            
            # 在模型的词表中找到"A:"的token_ids
            # 注：这里需要根据实际的tokenization结果调整
            # 可以通过打印tokens和对应的解码结果来确定具体的token_ids
            try:
                # 可能需要调整这个逻辑来匹配实际的tokenization
                answer_start = -1
                for j in range(len(tokens) - 1):
                    if tokens[j] == 128003:  # 假设"A"的token_id是65，":"的token_id是58
                        answer_start = j + 1  # +2 是为了跳过"A:"
                        break
                
                if answer_start != -1:
                    # 将答案之前的部分设为-100
                    labels[i, :answer_start] = -100
                else:
                    # 如果没找到"A:"，整个序列都设为-100
                    labels[i, :] = -100
                    
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                labels[i, :] = -100  # 发生错误时，将整个序列设为-100
        


        # Note the pad should not be calculated
        # labels[labels==128004] = -100
        # print(labels[0])
        batch_inputs["labels"] = labels
        
        if 'cross_attention_mask' in all_processed_inputs[0]:
            batch_inputs['cross_attention_mask'] = torch.stack([
                inputs['cross_attention_mask'] for inputs in all_processed_inputs
            ])
            
        return {
            'target_ids': target_ids,
            'text_sequences': all_text_sequences,
            'answers':all_answers,
            'processed_inputs': batch_inputs
        }

if __name__ == "__main__":
    checkpoint = "/mnt/data/CVPR2025/task1_data/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = MllamaForConditionalGeneration.from_pretrained(checkpoint)

    # model.bfloat16()
    
    dataset = MLLamaDataset(
        hdf5_path="/mnt/data/CVPR2025/task1_data/train_no_classification.hdf5",
        image_dir="/mnt/data/CVPR2025/task1_data/images/images/",
        template_path="template_vision_only.json",
        processor=processor,
        split="train",
        num_questions=1,
        max_length=512
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=MLLamaDataset.collate_fn,
        num_workers=1
    )
   
    # model.train()
    for batch in dataloader:
        # print(batch['processed_inputs']['input_ids'][0].shape)  # 打印输入序列的形状
        # print(batch['processed_inputs']['input_ids'][0][:200])
        # print(batch['processed_inputs']['attention_mask'][0][:200])
        # print(batch['processed_inputs']['cross_attention_mask'])

        with torch.no_grad():
            outputs = model(**batch['processed_inputs'])
            loss = outputs.loss
            print(loss)
        # loss.backward()
        
        break