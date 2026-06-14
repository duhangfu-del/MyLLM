"""
分组查询注意力 (GQA) + KV Cache 支持
训练时：不启用 cache，正常并行计算
推理时：传入 past_key_value，仅对新 token 计算，并返回更新后的 cache
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding, apply_rotary_pos_emb
from typing import Optional, Tuple

try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(config)
        self.use_flash = config.use_flash_attn and FLASH_AVAILABLE

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            past_key_value: 上一轮的 KV cache，为 (key_cache, value_cache)
            use_cache: 是否返回本轮更新后的 KV cache
        Returns:
            attn_output: (batch, seq_len, hidden_size)
            present_key_value: 如果 use_cache=True，返回 (k_cache, v_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ===== 1. 投影 =====
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 重塑为多头：q -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # k,v -> (batch, num_kv_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ===== 2. RoPE（仅对新 token 的 q,k 施加） =====
        past_len = past_key_value[0].size(2) if past_key_value is not None else 0
        cos, sin = self.rotary_emb(q, seq_len=seq_len, position_offset=past_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ===== 3. 拼接历史 KV cache =====
        if past_key_value is not None:
            # 解包上一轮的 cache，形状均为 (batch, num_kv_heads, past_len, head_dim)
            k_cache, v_cache = past_key_value
            # 沿序列维拼接
            k = torch.cat([k_cache, k], dim=2)   # (batch, num_kv_heads, past_len+seq_len, head_dim)
            v = torch.cat([v_cache, v], dim=2)

        # 保存本轮更新后的 cache（仅在 use_cache=True 时）
        present_key_value = (k, v) if use_cache else None

        # ===== 4. 扩展 KV 头以匹配 Q 头（GQA） =====
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # ===== 5. 注意力计算 =====
        if self.use_flash and seq_len == hidden_states.size(1):  # Flash Attn 要求 q 和 k 的 seq 维相同
            # 注意：当使用 cache 时，q 的 seq 可能为 1，而 k 的 seq 为 total_len，此时 flash_attn 仍可处理
            # 但需要设置 causal=False，因为我们只计算当前 q 对所有历史 k 的注意力
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            # 若使用 cache，关闭 causal；否则开启
            is_causal = (past_key_value is None)
            attn_output = flash_attn_func(q_fa, k_fa, v_fa, causal=is_causal)
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
        else:
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            # 遮罩处理：如果没有使用 cache（即训练或首次推理），需加上三角因果遮罩
            if past_key_value is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, k.size(2), device=hidden_states.device, dtype=torch.bool),
                    diagonal=1 + k.size(2) - seq_len  # 让当前 token 只能看到自身及左侧
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # ===== 6. 输出投影 =====
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value