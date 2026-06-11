"""
权重绑定（Weight Tying）工具函数
将语言模型头的权重与词嵌入矩阵的权重进行绑定，减少参数量并提升性能。
"""

def tie_word_embeddings(model):
    """
    将 lm_head 的权重设置为 embed_tokens 的权重（共享内存）
    注意：此操作应当在模型初始化时调用一次。
    Args:
        model: MiniMindForCausalLM 实例
    """
    model.lm_head.weight = model.embed_tokens.weight