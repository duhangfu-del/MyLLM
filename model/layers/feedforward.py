"""
SwiGLU FFN 实现
SwiGLU(x) = (xW_gate ⊙ SiLU(xW_up)) W_down
其中 SiLU(x) = x * sigmoid(x)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """带 SwiGLU 激活的门控前馈层"""
    
    def __init__(self, config):
        """
        Args:
            config: MiniMindConfig 对象，包含 hidden_size, intermediate_size
        """
        super().__init__()
        hidden_size = config.hidden_size          # 输入 / 输出维度
        intermediate_size = config.intermediate_size  # 门控中间层维度
        
        # 门控线性投影（用于 SiLU 激活的门控信号）
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 上采样线性投影
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 下采样线性投影，恢复为 hidden_size
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (batch, seq_len, hidden_size)
        Returns:
            输出张量，形状 (batch, seq_len, hidden_size)
        """
        # gate: 门控信号，经过 SiLU 激活
        gate = F.silu(self.gate_proj(x))
        # up: 上采样后的特征
        up = self.up_proj(x)
        # 逐元素相乘后，投影回 hidden_size
        return self.down_proj(gate * up)