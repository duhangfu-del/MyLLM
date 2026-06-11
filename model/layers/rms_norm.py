"""
RMSNorm: Root Mean Square Layer Normalization
与 LayerNorm 的区别：不减去均值，只除以均方根，计算量更小，在 LLM 中广泛使用。
"""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm: 对输入张量最后一维进行归一化。
    y = x / sqrt(mean(x^2) + eps) * γ
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size: 隐藏层维度，也即归一化的特征维度
            eps: 防止除零的极小值
        """
        super().__init__()
        # 可学习的缩放参数 γ
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_size)
        Returns:
            归一化后的张量，形状同输入
        """
        # 计算每个位置向量的均方根倒数
        # x.pow(2): 逐元素平方
        # .mean(-1, keepdim=True): 沿最后一维求均值，保持维度以便广播
        # torch.rsqrt: 求倒数平方根（更稳定）
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # 进行缩放并乘以可学习权重 γ
        return x * rms * self.weight