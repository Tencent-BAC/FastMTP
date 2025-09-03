from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithMTP, CausalLMOutputWithPastMTP
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Attention,
                                                      Qwen2ForCausalLM,
                                                      Qwen2MLP, Qwen2Model,
                                                      Qwen2RMSNorm)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import check_model_inputs

from .configuration_mimo import MiMoConfig


def roll_tensor(tensor, shifts=-1, dims=-1, fill_num=0.0):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.

    e.g. [1,2,3] -> [2,3,1] -> [2,3,0]
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(fill_num)
    return rolled_tensor, rolled_tensor.sum()


class MiMoMTPLayers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.token_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.final_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config)

    def forward(self, input_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values: Optional[Cache]=None,
                    output_attentions: Optional[bool]=False,
                    use_cache: Optional[bool]=False,
                    cache_position=None,
                    position_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    **kwargs):
        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(torch.cat([previous_hidden_states, input_embeds], dim=-1))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states,
                                          position_embedding,
                                          attention_mask,
                                          past_key_values=past_key_values,
                                          cache_position=cache_position,
                                          position_ids=position_ids,
                                          output_attentions=output_attentions,
                                          use_cache=use_cache,
                                          **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class MiMoModel(Qwen2Model):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        self.config = config
        self.mtp_layers = nn.ModuleList([MiMoMTPLayers(config) for _ in range(config.num_nextn_predict_layers)])

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # Sequence of hidden-states at the output of the last layer of the model.
        hidden_states_main_model = hidden_states # (batch_size, sequence_length, hidden_size)
        hidden_states_mtp = ()
        for step in range(self.config.num_speculative_steps):
            # Calc logits for the current Multi-Token Prediction (MTP) layers.
            input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)

            # embedding
            input_embeds = self.embed_tokens(input_ids)

            # norm, linear projection and transformer
            hidden_states = self.mtp_layers[0](
                input_embeds=input_embeds,
                hidden_states=hidden_states,
                attention_mask=causal_mask_mapping["full_attention"],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embedding=position_embeddings,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states_mtp += (hidden_states,)

        return BaseModelOutputWithMTP(
            last_hidden_state=hidden_states_main_model,
            past_key_values=past_key_values if use_cache else None,
            hidden_states_mtp=hidden_states_mtp,
        )


class MiMoForCausalLM(Qwen2ForCausalLM):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.config = config
        self.model = MiMoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        loss_ntp = loss

        hidden_states_mtp = outputs.hidden_states_mtp
        assert len(self.config.mtp_loss_step_weights) == self.config.num_speculative_steps, \
            f"Expected {self.config.num_speculative_steps} MTP loss weights, but got {len(self.config.mtp_loss_step_weights)}"
        assert sum(self.config.mtp_loss_step_weights) == 1.0, \
            f"Expected MTP loss weights of all steps to sum to 1.0, but got {sum(self.config.mtp_loss_step_weights)}"
        
        mtp_loss_sum = None
        mtp_loss = ()

        for layer_number in range(self.config.num_speculative_steps):
            # mtp output
            hidden_states_current_step = hidden_states_mtp[layer_number]
            mtp_logits = self.lm_head(hidden_states_current_step[:, slice_indices, :])

            # Calc loss for the current Multi-Token Prediction (MTP) layers.
            if labels is not None:
                labels, _ = roll_tensor(labels, shifts=-1, dims=-1, fill_num=-100)
                mtp_loss_step = self.loss_function(logits=mtp_logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                mtp_loss += (mtp_loss_step.detach().clone(),)
                
                if mtp_loss_sum is None:
                    mtp_loss_sum = mtp_loss_step * self.config.mtp_loss_step_weights[layer_number]
                else:
                    mtp_loss_sum += mtp_loss_step * self.config.mtp_loss_step_weights[layer_number]

        loss = self.config.ntp_loss_weight * loss_ntp + self.config.mtp_loss_weight * mtp_loss_sum if mtp_loss_sum is not None else loss_ntp

        return CausalLMOutputWithPastMTP(
            loss=loss,
            loss_ntp=loss_ntp,
            logits=logits,
            loss_mtp=mtp_loss_sum,
            loss_mtp_all=mtp_loss,
        ) 

__all__ = ["MiMoModel", "MiMoForCausalLM", "MiMoMTPLayers"]
