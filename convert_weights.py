import torch
from transformers import MllamaConfig, MllamaForConditionalGeneration
from transformers.models.mllama.modeling_mllama import MllamaCrossAttentionDecoderLayer

def create_new_config(original_config):
    new_config = MllamaConfig.from_dict(original_config.to_dict())
    
    # 添加新模态的配置
    new_config.structure_hidden_size = original_config.text_config.hidden_size
    new_config.spectrum_hidden_size = original_config.text_config.hidden_size
    
    # 为每个原始cross attention层添加两个新层
    new_config.text_config.cross_attention_layers = [
        layer for layer in original_config.text_config.cross_attention_layers
        for _ in range(3)
    ]
    
    return new_config

def convert_weights(original_model, new_config):
    new_model = MllamaForConditionalGeneration(new_config)
    
    # 复制所有共享的权重
    new_model.load_state_dict(original_model.state_dict(), strict=False)
    
    # 复制并初始化新的cross attention层
    for i, layer in enumerate(new_model.language_model.model.layers):
        if isinstance(layer, MllamaCrossAttentionDecoderLayer):
            original_layer = original_model.language_model.model.layers[i // 3]
            if i % 3 == 0:
                # 原始图像cross attention层
                layer.load_state_dict(original_layer.state_dict(), strict=False)
            else:
                # 新的结构和光谱cross attention层
                # 使用原始层的权重进行初始化
                layer.cross_attn.load_state_dict(original_layer.cross_attn.state_dict(), strict=False)
                # 重新初始化gate参数
                torch.nn.init.normal_(layer.cross_attn_attn_gate, mean=0.0, std=new_config.text_config.initializer_range)
                torch.nn.init.normal_(layer.cross_attn_mlp_gate, mean=0.0, std=new_config.text_config.initializer_range)
    
    return new_model

def convert_mllama_weights(checkpoint_path, output_path):
    # 加载原始模型和配置
    original_model = MllamaForConditionalGeneration.from_pretrained(checkpoint_path)
    original_config = original_model.config
    
    # 创建新配置
    new_config = create_new_config(original_config)
    
    # 转换权重
    new_model = convert_weights(original_model, new_config)
    
    # 保存新模型
    new_model.save_pretrained(output_path)
    print(f"New model saved to {output_path}")

# 使用示例
checkpoint_path = "path/to/original/mllama/checkpoint"
output_path = "path/to/new/mllama/checkpoint"
convert_mllama_weights(checkpoint_path, output_path)