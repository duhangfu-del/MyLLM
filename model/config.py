"""
模型配置类
集中管理所有超参数，其他模块通过传入该配置对象进行初始化。
"""
from dataclasses import dataclass

@dataclass
class MiniMindConfig:
    """MiniMind 模型的超参数配置"""
    
    # 词表大小，BBPE 分词器训练后得到 15k
    vocab_size: int = 15000
    
    # 隐藏层维度（也即 token embedding 维度）
    hidden_size: int = 768
    
    # Decoder 层数（总共 12 个 Decoder Layer）
    num_layers: int = 12
    
    # Query 的注意力头数
    num_attention_heads: int = 12
    
    # Key / Value 的注意力头数（分组查询注意力，4 个 KV 头）
    num_kv_heads: int = 4
    
    # SwiGLU FFN 的中间层维度（一般取 hidden_size 的 8/3 向上取整）
    intermediate_size: int = 2048
    
    # 最大序列长度（用于 RoPE 缓存预计算）
    max_seq_len: int = 2048
    
    # RoPE 的基础频率 θ（原论文为 10000）
    rope_theta: float = 10000.0
    
    # 是否使用 Flash Attention（需要安装 flash-attn 库）
    use_flash_attn: bool = False
    
    # 是否进行输入嵌入与输出头权重绑定（weight tying）
    tie_word_embeddings: bool = True
    
    # RMSNorm 中的 eps（防止除零）
    rms_norm_eps: float = 1e-6
    
    # Dropout 概率
    dropout: float = 0.0