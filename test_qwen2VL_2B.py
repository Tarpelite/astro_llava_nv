from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

checkpoint = "/mnt/data/CVPR2025/task1_data/Qwen2-VL-2B-Instruct"
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16
)

image_path = "/mnt/data/CVPR2025/task1_data/images/images/39633123209644949.png"
image_path = "/mnt/data/CVPR2025/task1_data/images/images/39633140951548502.png"
image = Image.open(image_path)

model.bfloat16()
model.cuda()
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     checkpoint,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
# processor = AutoProcessor.from_pretrained(checkpoint)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 110*110*3
max_pixels = 110*110*3
processor = AutoProcessor.from_pretrained(checkpoint, min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "Redshift is a measure of how much the wavelength of light from a galaxy has been stretched due to the expansion of the universe. It provides crucial information on the galaxy's velocity and distance and can be analyzed through multiple perspectives. Specifically, the <|image_pad|> utilizes celestial image data to obtain observational information such as morphology and luminosity, providing an initial estimate of distance.By integrating the information from these tokens, tell me the redshift value of the celestial object."},
        ],
    }
]

# import pudb;pu.db;

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "").replace(" <|image_pad|> ", "<|vision_start|><|image_pad|><|vision_end|>") #移动image_pad到对应的位置
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=10)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
