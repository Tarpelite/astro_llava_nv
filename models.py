import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os
from transformers import MllamaConfig, MllamaForConditionalGeneration
from typing import List, Optional, Tuple, Union, Dict

from transformers.models.mllama.configuration_mllama import MllamaTextConfig,MllamaVisionConfig
from transformers.models.mllama.modeling_mllama import MllamaCrossAttentionDecoderLayer, MllamaPreTrainedModel, MllamaSelfAttentionDecoderLayer, MllamaTextRMSNorm, MllamaRotaryEmbedding, _prepare_4d_causal_attention_mask_with_cache_position,MllamaVisionModel,_prepare_cross_attention_mask

from transformers.utils import logging
from transformers.modeling_rope_utils import rope_config_validation
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.generation import GenerationMixin

from spec_models import SpecFormer
from structure_models import StructureEncoder
from transformers.configuration_utils import PretrainedConfig


class AstrollavaTextConfig(PretrainedConfig):
    model_type = "mllama_text_model"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        hidden_act: str = "silu",
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 14_336,
        structure_output_dim: int=1024,
        spectrum_output_dim: int=1024,
        rope_theta: float = 500_000,
        rope_scaling: Optional[Dict] = None,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        cross_attention_layers: Optional[List[int]] = None,
        structure_cross_attention_layers: Optional[List[int]] = None,
        spectrum_cross_attention_layers: Optional[List[int]] = None,
        vision_cross_attention_layers:Optional[List[int]] = None,
        dropout: float = 0,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        pad_token_id: Optional[int] = 128004,
        **kwargs,
    ):
        if cross_attention_layers is None:
            cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.cross_attention_layers = cross_attention_layers
        # add cross attn for graph structure and spectrum encoders
        self.structure_cross_attention_layers = structure_cross_attention_layers
        self.spectrum_cross_attention_layers = spectrum_cross_attention_layers 
        self.vision_cross_attention_layers = vision_cross_attention_layers

        self.hidden_size = hidden_size
        self.structure_output_dim = structure_output_dim
        self.spectrum_output_dim = spectrum_output_dim

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mllama":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
           print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class AstrollavaConfig(MllamaConfig):
    model_type = "mllama"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=128256,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = MllamaVisionConfig()
            print("vision_config is None, using default mllama vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = MllamaVisionConfig(**vision_config)
        elif isinstance(vision_config, MllamaVisionConfig):
            self.vision_config = vision_config

        self.image_token_index = image_token_index

        if text_config is None:
            self.text_config = AstrollavaTextConfig()
            print("text_config is None, using default mllama text config")
        elif isinstance(text_config, dict):
            self.text_config = AstrollavaTextConfig(**text_config)
        elif isinstance(text_config, AstrollavaTextConfig):
            self.text_config = text_config

        super().__init__(**kwargs)

class AstroMllamaTextModel(MllamaPreTrainedModel):
    config_class = MllamaTextConfig
    base_model_prefix = "language_model.model"

    def __init__(self, config: MllamaTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size + 8, config.hidden_size, self.padding_idx)
        self.cross_attention_layers = config.cross_attention_layers
        self.structure_cross_attention_layers = config.structure_cross_attention_layers
        self.spectrum_cross_attention_layers = config.spectrum_cross_attention_layers
        self.vision_cross_attention_layers =  config.vision_cross_attention_layers


        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.cross_attention_layers:
                layers.append(MllamaCrossAttentionDecoderLayer(config, layer_idx))
            else:
                layers.append(MllamaSelfAttentionDecoderLayer(config, layer_idx))

        self.layers = nn.ModuleList(layers)
        self.norm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MllamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # @add_start_docstrings_to_model_forward(MLLAMA_TEXT_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class="MllamaTextConfig")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # cross_attention_states: Optional[torch.FloatTensor] = None,
        # cross_attention_mask: Optional[torch.Tensor] = None,
        strurcture_attention_states: Optional[torch.FloatTensor] = None,
        structure_attention_mask: Optional[torch.Tensor] = None,
        spectrum_attention_states:  Optional[torch.FloatTensor] = None,
        spectrum_attention_mask: Optional[torch.Tensor] = None,
        vision_attention_states:  Optional[torch.FloatTensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, MllamaTextModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaTextModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> text = "<|image|>If I had to write a haiku for this one"
        >>> inputs = processor(text=text, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 13, 4096])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            print(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have cross attention states
            # or cached cross attention states.
            is_cross_attention_layer = idx in self.cross_attention_layers
            is_cross_attention_cache_empty = past_key_values is None or (
                past_key_values is not None and past_key_values.get_seq_length(idx) == 0
            )
            # By tianyu ,control the cross attention states flow in 
            # [Start]
            if idx in self.structure_cross_attention_layers:
                is_structure_cross_attention_layer = True
                cross_attention_states = strurcture_attention_states
                cross_attention_mask = structure_attention_mask
            elif idx in self.spectrum_cross_attention_layers:
                is_spectrum_cross_attention_layer = True
                cross_attention_states = spectrum_attention_states
                cross_attention_mask = spectrum_attention_mask
            elif idx in self.vision_cross_attention_layers:
                is_vision_cross_attention_layer = True
                cross_attention_states = vision_attention_states
                cross_attention_mask = vision_attention_mask
            # [End]
            
            if is_cross_attention_layer and cross_attention_states is None and is_cross_attention_cache_empty:
                continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    cross_attention_states,
                    cross_attention_mask,
                    causal_mask,
                    full_text_row_masked_out_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    cross_attention_states=cross_attention_states,
                    cross_attention_mask=cross_attention_mask,
                    attention_mask=causal_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # TODO: we have only SDPA currently and there's a bug when attn-bias is passed. Need to add eager attn and return the line
        # self.config._attn_implementation == "sdpa" and
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class AstroMllamaForCausalLM(MllamaPreTrainedModel, GenerationMixin):
    config_class = MllamaTextConfig
    base_model_prefix = "language_model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.model = AstroMllamaTextModel._from_config(self.text_config, attn_implementation=config._attn_implementation)
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
         # cross_attention_states: Optional[torch.FloatTensor] = None,
        # cross_attention_mask: Optional[torch.Tensor] = None,
        structure_attention_states: Optional[torch.FloatTensor] = None,
        structure_attention_mask: Optional[torch.Tensor] = None,
        spectrum_attention_states:  Optional[torch.FloatTensor] = None,
        spectrum_attention_mask: Optional[torch.Tensor] = None,
        vision_attention_states:  Optional[torch.FloatTensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MllamaForCausalLM

        >>> model = MllamaForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
        >>> tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision")

        >>> prompt = "If I had to write a haiku, it would be:"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
        >>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(result)
        If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
        I love the idea of snowflakes gently falling, each one
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            # cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # cross_attention_mask=cross_attention_mask,
            structure_attention_states = structure_attention_states,
            structure_attention_mask = structure_attention_mask,
            spectrum_attention_states = spectrum_attention_states,
            spectrum_attention_mask = spectrum_attention_mask,
            vision_attention_states = vision_attention_states,
            vision_attention_mask = vision_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
        num_logits_to_keep=None,
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

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

class AstroMllamaForConditionalGeneration(MllamaPreTrainedModel, GenerationMixin):
    def __init__(self, config: AstrollavaConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        # self.vision_model = MllamaVisionModel._from_config(
        #     config.vision_config, attn_implementation=config._attn_implementation
        # )
        self.language_model = AstroMllamaForCausalLM._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )

        # self.structure_model = StructureEncoder(
        #     in_dim = 1024,
        #     hidden_dim= 512,
        #     out_dim = 256
        # )

        # structure modal projector
        self.structure_modal_projector = nn.Linear(
            256,
            config.text_config.hidden_size,
            bias = True,
        )

        # spectrum modal projector
        self.spectrum_modal_projector =  nn.Linear(
            1024, # hard coded
            config.text_config.hidden_size,
            bias = True,
        )

        # vision modal projector
        # self.multi_modal_projector = nn.Linear(
        #     config.vision_config.vision_output_dim,
        #     config.text_config.hidden_size,
        #     bias=True,
        # )
       
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        spec_input: Optional[torch.tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # cross_attention_states: Optional[torch.FloatTensor] = None,
        # cross_attention_mask: Optional[torch.Tensor] = None,
        vision_attention_states: Optional[torch.FloatTensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        structure_attention_states: Optional[torch.FloatTensor] = None,
        structure_attention_mask: Optional[torch.Tensor] = None,
        spectrum_attention_states:  Optional[torch.FloatTensor] = None,
        spectrum_attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
  
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        #     )

        # if pixel_values is not None and inputs_embeds is not None:
        #     raise ValueError(
        #         "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        #     )

        # if pixel_values is not None and vision_attention_states is not None:
        #     raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        # if pixel_values is not None:
        #     if aspect_ratio_ids is None:
        #         raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            
        #     # get vision tokens from vision model
        #     vision_outputs = self.vision_model(
        #         pixel_values=pixel_values,
        #         aspect_ratio_ids=aspect_ratio_ids,
        #         aspect_ratio_mask=aspect_ratio_mask,
        #         output_hidden_states=output_hidden_states,
        #         output_attentions=output_attentions,
        #         return_dict=return_dict,
        #     )
        #     vision_attention_states = vision_outputs[0]
        #     vision_attention_states = self.multi_modal_projector(vision_attention_states).reshape(
        #         -1, vision_attention_states.shape[-2], self.hidden_size
        #     )

        structure_attention_states = self.structure_modal_projector
        (structure_attention_states).reshape(-1, structure_attention_states.shape[-2], self.hidden_size)

        spectrum_attention_states = self.spectrum_modal_projector
        (spectrum_attention_states).reshape(-1, spectrum_attention_states.shape[-2], self.hidden_size)
        
        # vision_attention_states = self.multi_modal_projector(vision_attention_states).reshape(
        #     -1, vision_attention_states.shape[-2], self.hidden_size
        # )
        # vision states has been processed

        if vision_attention_mask is not None:
            vision_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                vision_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        # if vision_attention_mask is not None and cache_position is not None:
        #     vision_attention_mask = vision_attention_mask[:, :, cache_position]
        #     full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]
        
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # cross_attention_states=cross_attention_states,
            # cross_attention_mask=cross_attention_mask,
            structure_attention_states = structure_attention_states,
            structure_attention_mask = structure_attention_mask,
            spectrum_attention_states = spectrum_attention_states,
            spectrum_attention_mask = spectrum_attention_mask,
            vision_attention_states = vision_attention_states,
            vision_attention_mask = vision_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,

        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
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

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if (input_ids == self.config.image_token_index).any():
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        return model_kwargs

if __name__ == "__main__":
    # Test load specformer
    # spec_model_state_dict  = torch.load("/mnt/data/CVPR2025/task1_data/specformer/specformer.ckpt", weights_only=False)
    # spec_model = SpecFormer(
    #         embed_dim = 768,
    #         input_dim= 22,
    #         num_layers= 6,
    #         num_heads= 6,
    #         max_len= 800,
    #         dropout= 0.
    #     )
    # print(spec_model.load_state_dict(spec_model_state_dict['state_dict'])) # <All keys matched successfully>

    # Test load llama-3.1-v-11B
    checkpoint="/mnt/data/CVPR2025/task1_data/Llama-3.2-11B-Vision-Instruct/original/consolidated.pth"
    mllama_state_dict = torch.load(checkpoint, weights_only=False)
 
    # Make new model
    config = AstrollavaConfig.from_pretrained("/mnt/data/CVPR2025/task1_data/Llama-3.2-11B-Vision-Instruct/")
    
    



