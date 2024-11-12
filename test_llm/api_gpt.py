# import openai

# def query_gpt4(question):
#     openai.api_key = "sk-V0yOhUMybwo7B8o008ILF2IkbqxBnAg1hnTcrRP7DU7NXYDp"
#     openai.base_url = 'https://4.0.wokaai.com/v1/'  # 如果你使用的是自定义的 API URL，这可以保持

#     try:
#         # 使用正确的 API 方法
#         response = openai.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant"},
#                 {"role": "user", "content": question}
#             ]
#         )

#         # 输出返回的内容
#         # print(response)

#         # 通过正确的访问方式获取生成的内容
#         generated_text = response.choices[0].message.content
#         return generated_text
#     except Exception as e:
#         return str(e)

# # 提问并获取答案
# question = "Say this is a test"
# answer = query_gpt4(question)
# print(answer)

from astropy.table import Table

union_path = '/home/qhd/AstroCLIP/datasets/final/train_no_classification.hdf5'
tab = Table.read(union_path)
print(tab.colnames)
# print(tab['targetid'].data[0])
# print(len(tab['spectrum_feature'].data[0]))
# print(tab['redshift'].data[0])


import openai
import base64
from PIL import Image
import io

from PIL import Image
import io
import clip
import torch
import random


def get_image_features(image_path):
    """使用 CLIP 提取图像特征"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features.flatten().tolist()  # 返回一维特征向量


def query_gpt4(question, image_path, spectrum_data):
    openai.api_key = "sk-gS5bjcRT6X1SNXyyVHFKVFs9KXSq7iNg2oie480RYj6BWzkB"
    openai.base_url = 'https://4.0.wokaai.com/v1/'  # 如果你使用的是自定义的 API URL，这可以保持

    try:
        # 将图片转换为base64
        image_base64 = get_image_features(image_path)

        # 将图片、spectrum数据和问题作为消息一起传递
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an astronomer who can help me with the analysis of astronomical problems."},
                {"role": "user",
                 "content": f"Stellar mass refers to the total mass of all the stars in a galaxy. Mass-weighted stellar metallicity measures the abundance of elements heavier than hydrogen and helium in a galaxy's stars, weighted by their mass.Mass-weighted galaxy age refers to the average age of stars within a galaxy.The specific star-formation rate (sSFR) is the rate of star formation per unit stellar mass in a galaxy. Here is the spectrum token : {spectrum_data} and image token: {image_base64}. {question}"}
            ]
        )

        # 输出返回的内容
        generated_text = response.choices[0].message.content
        return generated_text
    except Exception as e:
        return str(e)


id = random.randint(0, len(tab['targetid'].data))
print(id)
targetid = tab['targetid'].data[id]
# 图片路径
image_path = f"/home/qhd/AstroCLIP/datasets/images/{targetid}.png"  # 请替换为图片的实际路径

# Spectrum数据（假设它是一个长度为1024的列表）
spectrum_data = tab['spectrum_feature'].data[id]  # 用你的数据替换这里的示例数据

# 提问并获取答案
question = "You must give an answer. Whether it is correct is not important. The value is an answer Please output the Stellar mass value1, Z_MW value2, sSFR value3, tage_mw value4 in the following format: LOG_MSTAR: [value1], Z_MW: [value2], sSFR: [value3], TAGE_MW[value4] . Other answers is useless."
answer = query_gpt4(question, image_path, spectrum_data)
print(answer)
import re


def extract_stel(text):
    # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
    match = re.search(r'LOG_MSTAR: \s*([0-9]+\.?[0-9]*)', text)
    if match:
        # 提取匹配的数字
        return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    else:
        match = re.search(r'LOG_MSTAR: \s*\[([0-9]+\.?[0-9]*)\]', text)
        if match:
            # 提取匹配的数字
            return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
        return None  # 如果没有匹配到，返回 None


def extract_zmw(text):
    # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
    match = re.search(r'Z_MW: \s*([0-9]+\.?[0-9]*)', text)
    if match:
        # 提取匹配的数字
        return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    else:
        match = re.search(r'Z_MW: \s*\[([0-9]+\.?[0-9]*)\]', text)
        if match:
            # 提取匹配的数字
            return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
        return None  # 如果没有匹配到，返回 None


# 示例文本

def extract_ssfr(text):
    # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
    match = re.search(r'sSFR: \s*([0-9]+\.?[0-9]*)', text)
    if match:
        # 提取匹配的数字
        return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    else:
        match = re.search(r'sSFR: \s*\[([0-9]+\.?[0-9]*)\]', text)
        if match:
            # 提取匹配的数字
            return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
        return None  # 如果没有匹配到，返回 None


# 示例文本


def extract_tage(text):
    # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
    match = re.search(r'TAGE_MW: \s*([0-9]+\.?[0-9]*)', text)
    if match:
        # 提取匹配的数字
        return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    else:
        match = re.search(r'TAGE_MW: \s*\[([0-9]+\.?[0-9]*)\]', text)
        if match:
            # 提取匹配的数字
            return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
        return None  # 如果没有匹配到，返回 None


# 提取数字
numbers1 = extract_stel(answer)
numbers2 = extract_ssfr(answer)
numbers3 = extract_zmw(answer)
numbers4 = extract_tage(answer)
# 输出提取的数字
print(numbers1)

import json
import os


def save_redshift_to_json(file_path, record_id, zmw_value, tamw_value, mst_value, ssfr_value):
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 文件存在，读取现有数据
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        # 文件不存在，初始化一个空的字典
        data = {}

    # 按照固定格式添加新的记录
    if str(record_id) not in data:
        data[str(record_id)] = {}

    # 将所有属性添加到同一个 record_id 的字典下
    data[str(record_id)] = {
        'Z_MW': zmw_value,
        'TAGE_MW': tamw_value,
        'LOG_MSTAR': mst_value,
        'sSFR': ssfr_value
    }
    # 将数据写入 JSON 文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# 示例：保存数据
file_path = 'task2_log_mstar_gpt4o.json'

redshift_value = numbers1
if (numbers1):
    save_redshift_to_json(file_path, id, numbers3, numbers4, numbers1, numbers2)