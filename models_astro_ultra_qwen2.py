import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLPreTrainedModel,
    Qwen2VLModel,
    Qwen2VLConfig,
    Qwen2VLDecoderLayer,
    _prepare_4d_causal_attention_mask_with_cache_position
)
from transformers.cache_utils import Cache, StaticCache




@dataclass
class AstroQwen2VLOutput(ModelOutput):
    """
    Outputs for MoE enhanced Qwen2VL model
    """
    loss: Optional[torch.FloatTensor] = None 
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_logits: Optional[List[torch.FloatTensor]] = None
    regression_value: Optional[torch.FloatTensor] = None




class ExpertFFN(nn.Module):
    """Base class for FFN experts"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class EuclideanFFN(ExpertFFN):
    """Standard Euclidean FFN matching original architecture"""
    def __init__(self, config):
        super().__init__(config)
        # 保持与原始FFN相同的参数命名
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

class HyperbolicFFN(ExpertFFN):
    """Hyperbolic FFN with Lorentz model"""
    def __init__(self, config):
        super().__init__(config)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states):
        # Compute in Lorentz model
        norm = torch.norm(hidden_states, dim=-1, keepdim=True)
        direction = hidden_states / (norm + 1e-8)
        
        # Standard FFN computation
        gate_output = self.act_fn(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)
        
        # Apply hyperbolic scaling
        return output * direction * self.scale

class SphericalFFN(ExpertFFN):
    """Spherical FFN with von Mises-Fisher distribution"""
    def __init__(self, config):
        super().__init__(config)
        self.kappa = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states):
        # Project to unit sphere
        h_norm = F.normalize(hidden_states, dim=-1)
        
        # Standard FFN computation
        gate_output = self.act_fn(self.gate_proj(h_norm))
        up_output = self.up_proj(h_norm)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)
        
        # Apply von Mises-Fisher scaling
        return F.normalize(output, dim=-1) * self.kappa


class AstroAdapter(nn.Module):
    """MoE adapter combining three geometric experts"""
    def __init__(self, config, orig_mlp_state=None):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, 3)  # Route to 3 experts
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self.experts = nn.ModuleList([
            EuclideanFFN(config),
            HyperbolicFFN(config),
            SphericalFFN(config)
        ])

        # Initialize Euclidean FFN with original weights if provided
        if orig_mlp_state is not None:
            self.init_euclidean_from_pretrained(orig_mlp_state)
    
    def init_euclidean_from_pretrained(self, orig_state):
        """Initialize Euclidean FFN with pretrained weights"""
        euclidean_ffn = self.experts[0]
        
        # Map original parameter names to new names
        param_mapping = {
            'gate_proj.weight': 'gate_proj.weight',
            'up_proj.weight': 'up_proj.weight',
            'down_proj.weight': 'down_proj.weight'
        }
        
        # Copy weights
        for orig_name, new_name in param_mapping.items():
            if orig_name in orig_state:
                getattr(euclidean_ffn, new_name.split('.')[0]).weight.data.copy_(
                    orig_state[orig_name]
                )

    def forward(self, hidden_states):
        # Get routing weights
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        router_probs = F.softmax(router_logits / self.temperature, dim=-1)
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(hidden_states)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, seq_len, hidden_size]
        
        # Reshape router_probs to match expert_outputs dimensions
        # [batch_size, seq_len, num_experts] -> [batch_size, num_experts, seq_len, 1]
        router_probs = router_probs.permute(0, 2, 1).unsqueeze(-1)
        
        # Combine expert outputs
        # expert_outputs: [batch_size, num_experts, seq_len, hidden_size]
        # router_probs: [batch_size, num_experts, seq_len, 1]
        combined_output = torch.sum(expert_outputs * router_probs, dim=1)  # [batch_size, seq_len, hidden_size]
        
        return combined_output, router_logits


class AstroQwen2VLDecoderLayer(Qwen2VLDecoderLayer):
    """Modified Qwen2VL decoder layer with MoE FFN"""
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        
        # Store original FFN weights before replacing
        orig_mlp_state = {} 
        for name, param in self.mlp.named_parameters():
            orig_mlp_state[name] = param.data.clone()
        
        # Remove original FFN
        delattr(self, "mlp")
        
        # Create MoE adapter with initialized Euclidean FFN
        self.moe = AstroAdapter(config, orig_mlp_state)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = self_attn_outputs[0]
        hidden_states = residual + hidden_states

        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + self_attn_outputs[1:]
        if output_attentions:
            outputs += (router_logits,)

        return outputs


class AstroQwen2VLModel(Qwen2VLModel):
    """Qwen2VL model with MoE enhanced decoder layers alternating with original layers"""
    def __init__(self, config):
        super().__init__(config)
        
        # Create a mixture of original and MoE layers
        self.layers = nn.ModuleList([
            AstroQwen2VLDecoderLayer(config, i) if i % 4 == 0  # 每隔4层使用MoE
            else Qwen2VLDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        
        # Extract router logits only from MoE layers
        router_logits = []
        if kwargs.get("output_attentions", False):
            for i, layer_outputs in enumerate(outputs.attentions):
                if i % 4 == 0:  # 只从MoE层收集router_logits
                    if len(layer_outputs) > 2:
                        router_logits.append(layer_outputs[-1])
                    
        outputs.router_logits = router_logits if router_logits else None
        return outputs
    
class AstroQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """Main model class with MoE enhancement and multi-modal projectors"""
    def __init__(self, config):
        super().__init__(config)
        
        # Replace standard Qwen2VL model with MoE version
        delattr(self, "model")
        self.model = AstroQwen2VLModel(config)
        
        # Add projectors for spectral and structural features
        self.spec_projector = nn.Linear(1024, config.hidden_size)
        self.struc_projector = nn.Linear(256, config.hidden_size)

        self.spec_token_id = 73780 # spectrum token is Ã
        self.euc_token_id = 79607 # euc token is þ
        self.hyp_token_id = 65013 # hyp token is æ
        self.sph_token_id = 38118 # sph token is ø
        self.num_token_id = 1629 # num token is num

        self.spec_norm = nn.LayerNorm(config.hidden_size)
        self.struc_norm = nn.LayerNorm(config.hidden_size)

        self.spec_scale = nn.Parameter(torch.ones(1) * 0.1)  # 初始值设为0.1
        self.struc_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 添加损失权重参数
        self.lm_weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.regression_weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
        # Add regression head
        self.num_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1)
        )

        self.regression_loss = nn.SmoothL1Loss()  # 使用Huber Loss提高稳定性
        
        # Initialize new components
        self._init_new_weights()
        self._init_multimodal_weights()

    def _init_new_weights(self):
        """Initialize weights for new components"""
        for module in [self.spec_projector, self.struc_projector, self.num_head]:
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def _init_multimodal_weights(self):
        """使用float32进行权重初始化"""
        # 确保在float32下进行初始化
        with torch.no_grad():
            # 使用正交初始化
            spec_weight = torch.empty(self.spec_projector.weight.shape, dtype=torch.float32)
            struc_weight = torch.empty(self.struc_projector.weight.shape, dtype=torch.float32)
            
            nn.init.orthogonal_(spec_weight)
            nn.init.orthogonal_(struc_weight)
            
            # 缩放权重
            spec_weight.mul_(0.1)
            struc_weight.mul_(0.1)
            
            # 赋值回projector
            self.spec_projector.weight.data.copy_(spec_weight)
            self.struc_projector.weight.data.copy_(struc_weight)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos = None,
        image_grid_thw = None,
        video_grid_thw = None,
        spec_features=None,
        euc_features = None,
        hyp_features = None,
        sph_features = None,
        answers=None,
        task_type_id=None,
        **kwargs
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            model_dtype = inputs_embeds.dtype  # 获取模型当前使用的dtype
            # Process spectral features

            spec_embeds = None
            if spec_features is not None:
                # 转换到float32进行处理
                spec_features = spec_features.to(torch.float32)
                spec_features = F.normalize(spec_features, p=2, dim=-1)
                
                # 投影和归一化（在float32下）
                with torch.amp.autocast("cuda", enabled=False):
                    spec_embeds = self.spec_projector.to(torch.float32)(spec_features)
                    spec_embeds = self.spec_norm.to(torch.float32)(spec_embeds)
                    spec_embeds = spec_embeds * self.spec_scale.to(torch.float32)
                
                # 转回模型dtype并替换嵌入
                spec_embeds = spec_embeds.to(model_dtype)
                spec_mask = (input_ids == self.spec_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(spec_mask, spec_embeds)
            
            euc_embeds = None
            # Process structural features
            if euc_features is not None:
                # 转换到float32进行处理
                euc_features = euc_features.to(torch.float32)
                euc_features = F.normalize(euc_features, p=2, dim=-1)
                
                # 投影和归一化（在float32下）
                with torch.amp.autocast("cuda", enabled=False):
                    euc_embeds = self.struc_projector.to(torch.float32)(euc_features)
                    euc_embeds = self.struc_norm.to(torch.float32)(euc_embeds)
                    euc_embeds = euc_embeds * self.struc_scale.to(torch.float32)
                
                # 转回模型dtype并替换嵌入
                euc_embeds = euc_embeds.to(model_dtype)
                euc_mask = (input_ids == self.euc_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(euc_mask, euc_embeds)

            hyp_embeds = None
            if hyp_features is not None:
                # 转换到float32进行处理
                hyp_features = hyp_features.to(torch.float32)
                hyp_features = F.normalize(hyp_features, p=2, dim=-1)
                
                # 投影和归一化（在float32下）
                with torch.amp.autocast("cuda", enabled=False):
                    hyp_embeds = self.struc_projector.to(torch.float32)(hyp_features)
                    hyp_embeds = self.struc_norm.to(torch.float32)(hyp_embeds)
                    hyp_embeds = hyp_embeds * self.struc_scale.to(torch.float32)
                
                # 转回模型dtype并替换嵌入
                hyp_embeds = hyp_embeds.to(model_dtype)
                hyp_mask = (input_ids == self.hyp_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(hyp_mask, hyp_embeds)

            sph_embeds = None
            if sph_features is not None:
                # 转换到float32进行处理
                sph_features = sph_features.to(torch.float32)
                sph_features = F.normalize(sph_features, p=2, dim=-1)
                
                # 投影和归一化（在float32下）
                with torch.amp.autocast("cuda", enabled=False):
                    sph_embeds = self.struc_projector.to(torch.float32)(sph_features)
                    sph_embeds = self.struc_norm.to(torch.float32)(sph_embeds)
                    sph_embeds = sph_embeds * self.struc_scale.to(torch.float32)
                
                # 转回模型dtype并替换嵌入
                sph_embeds = sph_embeds.to(model_dtype)
                sph_mask = (input_ids == self.sph_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(sph_mask, sph_embeds)
            
            image_embeds = None
            # Process vision features (from parent class)
            if pixel_values is not None:
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            inputs_embeds = torch.clip(inputs_embeds, -100, 100)  # 防止数值溢出

        # Forward through model
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            # **kwargs
        )

        # Get language modeling logits
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Get regression value if needed
        regression_value = None
        # if answers is not None:
        #     regression_value = self.regression_head(hidden_states[:, -1])

        # Calculate losses
        loss = None

        if labels is not None:
            
            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Add regression loss if applicable
            if answers is not None and task_type_id == 0:
                # find the num token in label
                shift_hidden_states = hidden_states[...,:-1, :].contiguous()
                num_mask = (shift_labels == self.num_token_id).unsqueeze(-1).expand_as(shift_hidden_states)
                num_logits = shift_hidden_states[num_mask].view(-1, shift_hidden_states.shape[-1]) 
                if euc_embeds is not None:
                    # import pudb;pu.db;
                    num_logits += euc_embeds
                if hyp_embeds is not None:
                    num_logits += hyp_embeds
                
                if spec_embeds is not None:
                    num_logits += spec_embeds

                if sph_embeds is not None:
                    num_logits += sph_embeds
                
                # 计算回归值（在float32精度下）
                with torch.amp.autocast("cuda", enabled=False):
                    num_logits = num_logits.to(torch.float32)
                    regression_value = self.num_head(num_logits)
                
                # 确保在相同设备和dtype上计算
                answers = torch.tensor(answers).to(regression_value.device)
                answers = answers.to(torch.float32)

                #  # 计算回归损失
                # if regression_value.shape[0] == 0:
                #     import pudb; pu.db;
                print("regress_value: {} answers:{}".format(regression_value, answers))
                reg_loss = self.regression_loss(regression_value.squeeze().view(-1), answers.view(-1))
                
                # 使用动态权重合并损失
                lm_weight = F.softplus(self.lm_weight)  # 确保权重为正
                regression_weight = F.softplus(self.regression_weight)
                
                # 归一化权重
                total_weight = lm_weight + regression_weight
                lm_weight = lm_weight / total_weight
                regression_weight = regression_weight / total_weight


                loss = lm_weight* loss + regression_weight *reg_loss

        if not return_dict:
            output = (logits, regression_value) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return AstroQwen2VLOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            regression_value=regression_value
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        spec_features=None,
        euc_features = None,
        hyp_features = None,
        sph_features = None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "spec_features": spec_features,
                "euc_features": euc_features,
                "hyp_features": hyp_features,
                "sph_features": sph_features,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs

    def predict_number(
            self,
            input_ids: torch.LongTensor,
            hidden_states: torch.FloatTensor,
            spec_features: Optional[torch.FloatTensor] = None,
            euc_features: Optional[torch.FloatTensor] = None, 
            hyp_features: Optional[torch.FloatTensor] = None,
            sph_features: Optional[torch.FloatTensor] = None
            ) -> torch.FloatTensor:
            """
            Predict numerical values by combining token logits with multimodal features
            
            Args:
                input_ids: Input token IDs including " num" token
                hidden_states: Hidden states from model forward pass
                spec_features: Optional spectral features 
                euc_features: Optional Euclidean features
                hyp_features: Optional hyperbolic features 
                sph_features: Optional spherical features
                
            Returns:
                Predicted numerical value
            """
           
            # Find positions of " num" token (token_id = 1629)
            num_positions = (input_ids == self.num_token_id).nonzero()
            
            if len(num_positions) == 0:
                raise ValueError("No ' num' token found in input_ids")
            
            # Get hidden states for tokens before " num"
            # num_positions[0] gives batch index, num_positions[1] gives position
            batch_idx = num_positions[0,0]
            pos_idx = num_positions[0,1] - 1  # Get previous token
            
            # Get hidden states for the relevant token
            try:
                token_hidden = hidden_states[batch_idx, pos_idx]
            except Exception as e:
                print("Hidden States Shape:{} batch_idx:{} pos_idx:{}".format(hidden_states.shape, batch_idx, pos_idx))
                print(e)
            
            # Combine with multimodal features if provided
            combined_features = token_hidden
            
            model_dtype = combined_features.dtype

            if euc_features is not None:
                euc_features = euc_features.to(torch.float32)
                euc_features = F.normalize(euc_features, p=2, dim=-1)
                
                # 投影和归一化（在float32下）
                with torch.amp.autocast("cuda", enabled=False):
                    euc_embeds = self.struc_projector.to(torch.float32)(euc_features)
                    euc_embeds = self.struc_norm.to(torch.float32)(euc_embeds)
                    euc_embeds = euc_embeds * self.struc_scale.to(torch.float32)
                
                # 转回模型dtype并替换嵌入
                euc_embeds = euc_embeds.to(model_dtype)
                combined_features = combined_features + euc_embeds.squeeze()
                
            if hyp_features is not None:
                hyp_features = hyp_features.to(torch.float32)
                hyp_features = F.normalize(hyp_features, p=2, dim=-1)
                
                # 投影和归一化（在float32下）
                with torch.amp.autocast("cuda", enabled=False):
                    hyp_embeds = self.struc_projector.to(torch.float32)(hyp_features)
                    hyp_embeds = self.struc_norm.to(torch.float32)(hyp_embeds)
                    hyp_embeds = hyp_embeds * self.struc_scale.to(torch.float32)
                
                # 转回模型dtype并替换嵌入
                hyp_embeds = hyp_embeds.to(model_dtype)
                combined_features = combined_features + hyp_embeds.squeeze()
                
            if spec_features is not None:
                spec_features = spec_features.to(torch.float32)
                spec_features = F.normalize(spec_features, p=2, dim=-1)
                
                # 投影和归一化（在float32下）
                with torch.amp.autocast("cuda", enabled=False):
                    spec_embeds = self.spec_projector.to(torch.float32)(spec_features)
                    spec_embeds = self.spec_norm.to(torch.float32)(spec_embeds)
                    spec_embeds = spec_embeds * self.spec_scale.to(torch.float32)
                
                # 转回模型dtype并替换嵌入
                spec_embeds = spec_embeds.to(model_dtype)
                combined_features = combined_features + spec_embeds.squeeze()
                
            if sph_features is not None:
                sph_features = sph_features.to(torch.float32)
                sph_features = F.normalize(sph_features, p=2, dim=-1)
                
                # 投影和归一化（在float32下）
                with torch.amp.autocast("cuda", enabled=False):
                    sph_embeds = self.struc_projector.to(torch.float32)(sph_features)
                    sph_embeds = self.struc_norm.to(torch.float32)(sph_embeds)
                    sph_embeds = sph_embeds * self.struc_scale.to(torch.float32)
                
                # 转回模型dtype并替换嵌入
                sph_embeds = sph_embeds.to(model_dtype)
                combined_features = combined_features + sph_embeds.squeeze()
            
            # Use num_head to get prediction
            # Convert to float32 for better precision
            with torch.amp.autocast("cuda", enabled=False):
                combined_features = combined_features.to(torch.float32)
                prediction = self.num_head(combined_features)
            
            return prediction.squeeze()

if __name__ == "__main__":
    checkpoint = "/mnt/data/CVPR2025/task1_data/Qwen2-VL-2B-Instruct"
    model = AstroQwen2VLForConditionalGeneration.from_pretrained(checkpoint)