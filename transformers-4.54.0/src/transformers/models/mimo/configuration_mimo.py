from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MiMoConfig(Qwen2Config):
    model_type = "mimo"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        *args,
        num_nextn_predict_layers=1,
        num_speculative_steps=1,
        mtp_loss_step_weights=[1.0],
        ntp_loss_weight=0.0,
        mtp_loss_weight=1.0,
        **kwargs
    ):
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_speculative_steps = num_speculative_steps
        self.mtp_loss_step_weights = mtp_loss_step_weights
        self.ntp_loss_weight = ntp_loss_weight
        self.mtp_loss_weight = mtp_loss_weight
        super().__init__(
            *args,
            **kwargs,
        )


__all__ = ["MiMoConfig"]
        