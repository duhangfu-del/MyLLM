"""
RoPE: Rotary Position Embedding
实现思路：
1. 根据 config 中的 hidden_size / num_attention_heads 得到 head_dim
2. 根据 head_dim 和 rope_theta 预计算频率的倒数 inv_freq
3. 对于任意序列长度，计算 cos 和 sin 缓存在 buffer 中
4. 在前向传播时，将 cos/sin 应用到 query 和 key 上
"""
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """RoPE 位置编码（不参与训练，仅提供正余弦缓存）"""
    
    def __init__(self, config):
        """
        Args:
            config: MiniMindConfig 对象，需包含：
                - hidden_size
                - num_attention_heads
                - max_seq_len
                - rope_theta
        """
        super().__init__()
        # 每个注意力头的维度
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_seq_len
        theta = config.rope_theta
        
        # 计算频率的倒数 inv_freq: shape (dim // 2,)
        # 形式为 1.0 / (theta^(2i/dim))，i = 0,1,...,dim/2-1
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        # 将 inv_freq 注册为 buffer，不参与梯度
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 初始化 cos/sin 缓存（使用 config 中的 max_seq_len）
        self._set_cos_sin(seq_len=self.max_seq_len)

    def _set_cos_sin(self, seq_len: int):
        """根据指定的序列长度预计算 cos 和 sin 并存入 buffer"""
        # 生成位置索引 [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        # 外积得到每个位置与每个频率的乘积，shape (seq_len, dim//2)
        freqs = torch.outer(t, self.inv_freq)
        # 将 freqs 重复拼接成 (seq_len, dim)，因为每两个元素用同一个频率
        emb = torch.cat((freqs, freqs), dim=-1)
        # 扩展维度以匹配 q/k 的维度 (batch, head, seq, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None, position_offset: int = 0):
        """
        返回当前序列长度对应的 cos 和 sin
        Args:
            x: 用于获取设备与数据类型的输入张量（形状无关）
            seq_len: 实际序列长度，默认取 x 的序列长度
            position_offset: 位置偏移量，用于 KV-cache 增量解码时指定起始位置
        Returns:
            cos: shape (1, 1, seq_len, dim)
            sin: shape (1, 1, seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[-2]  # 假设 x 维度为 (batch, head, seq, dim)
        # 如果序列长度超过缓存，重新计算
        total_len = seq_len + position_offset
        if total_len > self.max_seq_len:
            self._set_cos_sin(seq_len=total_len)
        return (
            self.cos_cached[:, :, position_offset:position_offset + seq_len, :].to(x.dtype),
            self.sin_cached[:, :, position_offset:position_offset + seq_len, :].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将张量沿最后一维分成两半，并将前一半取负后与后一半交叉拼接"""
    # 将最后一维分为前半部分 x1 和后半部分 x2
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 拼接：[-x2, x1]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对 query 和 key 应用旋转变换
    公式: q_rot = q * cos + rotate_half(q) * sin
          k_rot = k * cos + rotate_half(k) * sin
    Args:
        q: query 张量，形状 (batch, num_heads, seq_len, head_dim)
        k: key 张量，形状同上
        cos, sin: 预计算的位置编码，形状 (1, 1, seq_len, head_dim)
    Returns:
        (q_rot, k_rot)
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot