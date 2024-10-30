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

class Qwen2VLDataset(Dataset):
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
        
        # Load metadata
        self.data = Table.read(hdf5_path)
        
        # Load question templates
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
            return min(self.max_samples, len(self.data))
        return len(self.data)
    
    def _load_image(self, target_id: str) -> Image.Image:
        image_path = os.path.join(self.image_dir, f"{target_id}.png")
        return Image.open(image_path)
    
    def _get_value(self, row_data: Dict, key: str) -> float:
        value = row_data[key]
        if isinstance(value, np.ndarray):
            value = value.item()
        return float(value)
    
    def _construct_messages(self, image: Image.Image, task_type: str, row_data: Dict) -> Tuple[List[Dict], float]:
        template = self.templates[task_type]
        answer_key = self.answer_mapping[task_type]
        answer = self._get_value(row_data, answer_key)

        # if self.split == "train":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": template
                    },
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": " num"
                    }
                ]
            }
        ]
        
        # else:
        #     messages = [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "image",
        #                     "image": image,
        #                 },
        #                 {
        #                     "type": "text", 
        #                     "text": template
        #                 },
        #             ],
        #         }
        #     ]
        
        return messages, answer
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.data[idx]
        target_id = str(row['TARGETID'])
        
        # Load image
        image = self._load_image(target_id)
        
        # Randomly select task types
        selected_tasks = random.sample(self.task_types, self.num_questions)
        
        # Construct messages for each task
        all_messages = []
        answers = []
        text_sequences = []
        
        for task in selected_tasks:
            messages, answer = self._construct_messages(image, task, row)
            all_messages.append(messages)
            answers.append(answer)
            text_sequences.append(messages[0]["content"][1]["text"])
        
        # Process inputs
        processed_samples = []
        for messages in all_messages:
            # Prepare text using chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Move image_pad to correct position
            text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "").replace(
                " <|image_pad|> ", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Process with processor
            if self.split == "train":
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    # padding=True,
                    return_tensors="pt",
                    text_kwargs={
                        "max_length" : self.max_length,
                        "padding": "max_length",
                        "padding_side":"right",
                        "truncation": True
                    }
                    
                )
            else:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    # padding=True,
                    return_tensors="pt",
                    # text_kwargs={
                    #     "max_length" : self.max_length,
                    #     "padding": "max_length",
                    #     "padding_side":"right",
                    #     "truncation": True
                    # }
                    
                )
            
            # Remove batch dimension
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            # print(inputs.keys())
            processed_samples.append(inputs)
            
        return {
            'target_id': target_id,
            'processed_inputs': processed_samples,
            'text_sequences': text_sequences,
            'answers': answers,
            'raw_image': image
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        target_ids = [item['target_id'] for item in batch]
        all_text_sequences = [item['text_sequences'] for item in batch]
        all_answers = torch.tensor([item['answers'] for item in batch])
        
        all_processed_inputs = []
        for item in batch:
            all_processed_inputs.extend(item['processed_inputs'])
            
        batch_inputs = {
            'input_ids': torch.stack([inputs['input_ids'] for inputs in all_processed_inputs]),
            'attention_mask': torch.stack([inputs['attention_mask'] for inputs in all_processed_inputs]),
            'pixel_values': torch.stack([inputs['pixel_values'] for inputs in all_processed_inputs]),
            'image_grid_thw': torch.stack([inputs['image_grid_thw'] for inputs in all_processed_inputs]),
           
        }

        # Create special token mask for token_id 1629
        batch_size, seq_length = batch_inputs['input_ids'].shape
        special_token_mask = torch.zeros((batch_size, seq_length), dtype=torch.bool)
        
        for i in range(batch_size):
            # Find positions of special token 1629
            special_positions = (batch_inputs['input_ids'][i] == 1629).nonzero(as_tuple=True)[0]
            if len(special_positions) > 0:
                special_token_mask[i, special_positions] = True # for token shift
        
        batch_inputs['num_token_mask'] = special_token_mask


        labels = batch_inputs["input_ids"].clone()
        input_ids = batch_inputs["input_ids"]

        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # 将input_ids转换为列表以便搜索
            tokens = input_ids[i].tolist()
            # print(tokens)
            
            # 在模型的词表中找到"A:"的token_ids
            # 注：这里需要根据实际的tokenization结果调整
            # 可以通过打印tokens和对应的解码结果来确定具体的token_ids
            try:
                # 可能需要调整这个逻辑来匹配实际的tokenization
                answer_start = -1
                answer_end = -1
                
                for j in range(len(tokens) - 1):
                    if tokens[j] == 151644 and tokens[j+1] ==  77091:  # 假设"A"的token_id是65，":"的token_id是58
                        answer_start = j + 2  # +2 是为了跳过"A:"
                        for k in range(answer_start, len(tokens)):
                            if tokens[k] == 151645:
                                answer_end = k
                                break
                        break
                    
                if answer_start != -1:
                    # 将答案之前的部分设为-100
                    labels[i, :answer_start] = -100
                    labels[i, (answer_end+1):] = -100
                else:
                    # 如果没找到"A:"，整个序列都设为-100
                    labels[i, :] = -100
                    
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                labels[i, :] = -100  # 发生错误时，将整个序列设为-100

            # print(answer_start)
            # print(answer_end)
            # print(labels[i])

            
        batch_inputs["labels"] = labels
        batch_inputs["answers"] = all_answers
        
        
        return {
            'target_ids': target_ids,
            'text_sequences': all_text_sequences,
            'answers': all_answers,
            'processed_inputs': batch_inputs
        }