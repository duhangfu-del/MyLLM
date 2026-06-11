import torch
import torch.nn as nn
from .layers.rms_norm import RMSNorm
from .layers.attention import GroupedQueryAttention
from .layers.feedforward import SwiGLUFFN
from typing import Optional, Tuple


class DecoderLayer(nn.Module):
    """单个 Decoder 层：Pre-Norm 自注意力 + Pre-Norm FFN，均带残差连接"""

    def __init__(self, config):
        """
        Args:
            config: MiniMindConfig 对象
        """
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ):
        """
        Returns:
            hidden_states: (batch, seq_len, hidden_size)
            present_key_value: 如果 use_cache=True，返回本层的 KV cache
        """
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value