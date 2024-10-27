from PIL import Image
import requests
from transformers import AutoProcessor, MllamaForConditionalGeneration

checkpoint="/mnt/data/CVPR2025/task1_data/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

prompt = "<|image|>If I had to write a haiku for this one"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# import pudb;pu.db;
inputs = processor(text=prompt, images=image, return_tensors="pt")

# import pudb;pu.db;
output = model.generate(**inputs, max_new_tokens=15)

