import anthropic
import anthropic
import requests
import json

from astropy.table import Table

# union_path = '/home/qhd/AstroCLIP/datasets/final/train_no_classification.hdf5'
# tab = Table.read(union_path)
# print(tab.colnames)
union_path = '/home/qhd/AstroCLIP/datasets/task3/classifications_train_new1.hdf5'
tab = Table.read(union_path)
print(tab.colnames)
print(tab['smooth'])

# import openai
# import base64
# from PIL import Image
# import io

# from PIL import Image
# import io
# import clip
# import torch
# import random
# def get_image_features(image_path):
#     """使用 CLIP 提取图像特征"""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device)

#     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image)

#     return image_features.flatten().tolist()  # 返回一维特征向量


# def get_chat_gpt_response(question, image_path, spectrum_data):
#     url = "https://4.0.wokaai.com/v1/chat/completions"
#     headers ={
#         "Authorization": "sk-gS5bjcRT6X1SNXyyVHFKVFs9KXSq7iNg2oie480RYj6BWzkB",
#         "Content-Type": "application/json"
#     }
#     image_base64 = get_image_features(image_path)

#     # data = {
#     #     "model": "claude-3-5-sonnet-20240620",
#     #     "messages": [{"role":"system", "content": "You are an astronomer who can help me with the analysis of astronomical problems."},
#     #         {"role": "user", "content": f"Stellar mass refers to the total mass of all the stars in a galaxy. Mass-weighted stellar metallicity measures the abundance of elements heavier than hydrogen and helium in a galaxy's stars, weighted by their mass.Mass-weighted galaxy age refers to the average age of stars within a galaxy.The specific star-formation rate (sSFR) is the rate of star formation per unit stellar mass in a galaxy. Here is the spectrum token : {spectrum_data} and image token: {image_base64}. {question}"}]
#     # }
#     data = {
#         "model": "claude-3-5-sonnet-20240620",
#         "messages": [{"role":"system", "content": "You are an astronomer who can help me with the analysis of astronomical problems."},
#             {"role": "user", "content": f"Redshift is a measure of how much the wavelength of light from a galaxy has been stretched due to the expansion of the universe. Here is the spectrum token : {spectrum_data} and image token: {image_base64}. {question}"}]
#     }
#     response = requests.post(url,headers=headers,json=data)
#     print(json.loads(response.text))
#     return response
# # question = "You must give an answer. Whether it is correct is not important. The value is an answer Please output the Stellar mass value1, Z_MW value2, sSFR value3, tage_mw value4 in the following format: LOG_MSTAR: [value1], Z_MW: [value2], sSFR: [value3], TAGE_MW[value4] . Other answers is useless."
# question = "You must give an answer. Whether it is correct is not important. The value is an answer Please output the Redshift value in the following format: redshift: [value]. Other answers is useless."
# id = random.randint(0, len(tab['targetid'].data))
# print(id)
# targetid = tab['targetid'].data[id]
# # 图片路径
# image_path = f"/home/qhd/AstroCLIP/datasets/images/{targetid}.png"  # 请替换为图片的实际路径
# # Spectrum数据（假设它是一个长度为1024的列表）
# spectrum_data = tab['spectrum_feature'].data[id]  # 用你的数据替换这里的示例数据

# resp =get_chat_gpt_response(question, image_path, spectrum_data)

# response_data = json.loads(resp.text)
# print(response_data['choices'][0]['message']['content'])
# answer = response_data['choices'][0]['message']['content']
# import re
# def extract_stel(text):
#     # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
#     match = re.search(r'LOG_MSTAR: \s*([0-9]+\.?[0-9]*)', text)
#     if match:
#         # 提取匹配的数字
#         return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#     else:
#         match = re.search(r'LOG_MSTAR: \s*\[([0-9]+\.?[0-9]*)\]', text)
#         if match:
#         # 提取匹配的数字
#             return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#         return None  # 如果没有匹配到，返回 None

# def extract_zmw(text):
#     # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
#     match = re.search(r'Z_MW: \s*([0-9]+\.?[0-9]*)', text)
#     if match:
#         # 提取匹配的数字
#         return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#     else:
#         match = re.search(r'Z_MW: \s*\[([0-9]+\.?[0-9]*)\]', text)
#         if match:
#         # 提取匹配的数字
#             return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#         return None  # 如果没有匹配到，返回 None
# # 示例文本

# def extract_ssfr(text):
#     # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
#     match = re.search(r'sSFR: \s*([0-9]+\.?[0-9]*)', text)
#     if match:
#         # 提取匹配的数字
#         return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#     else:
#         match = re.search(r'sSFR: \s*\[([0-9]+\.?[0-9]*)\]', text)
#         if match:
#         # 提取匹配的数字
#             return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#         return None  # 如果没有匹配到，返回 None
# # 示例文本


# def extract_tage(text):
#     # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
#     match = re.search(r'TAGE_MW: \s*([0-9]+\.?[0-9]*)', text)
#     if match:
#         # 提取匹配的数字
#         return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#     else:
#         match = re.search(r'TAGE_MW: \s*\[([0-9]+\.?[0-9]*)\]', text)
#         if match:
#         # 提取匹配的数字
#             return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#         return None  # 如果没有匹配到，返回 None

# def extract_redshift(text):
#     # 使用正则表达式提取 "Predicted Redshift:" 后面的数字
#     match = re.search(r'redshift: \s*([0-9]+\.?[0-9]*)', text)
#     if match:
#         # 提取匹配的数字
#         return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#     else:
#         match = re.search(r'redshift: \s*\[([0-9]+\.?[0-9]*)\]', text)
#         if match:
#         # 提取匹配的数字
#             return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#         return None  # 如果没有匹配到，返回 None
# # 提取数字
# # numbers1 = extract_stel(answer)
# # numbers2 = extract_ssfr(answer)
# # numbers3 = extract_zmw(answer)
# # numbers4 = extract_tage(answer)
# # # 输出提取的数字
# # print(numbers1)
# number =extract_redshift(answer)
# print(number)
# import json
# import os

# def save_redshift_to_json(file_path, record_id, value):
#     # 检查文件是否存在
#     if os.path.exists(file_path):
#         # 文件存在，读取现有数据
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#     else:
#         # 文件不存在，初始化一个空的字典
#         data = {}

#     # 按照固定格式添加新的记录
#     if str(record_id) not in data:
#         data[str(record_id)] = {}

# # 将所有属性添加到同一个 record_id 的字典下
#     # data[str(record_id)] = {
#     # 'Z_MW': zmw_value,
#     # 'TAGE_MW': tamw_value,
#     # 'LOG_MSTAR': mst_value,
#     # 'sSFR': ssfr_value
#     # }
#     data[str(record_id)] = {
#     'redshift': value

#     }
#     # 将数据写入 JSON 文件
#     with open(file_path, 'w') as f:
#         json.dump(data, f, indent=4)

# # 示例：保存数据
# file_path = 'task1_redshift_claud3.5.json'

# redshift_value = number
# if(number):
#     save_redshift_to_json(file_path, id, number)