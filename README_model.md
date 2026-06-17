# MiniMind

从零实现的 Decoder-only Transformer 中文语言模型，87M 参数，支持预训练和 SFT 微调版本。

## 模型架构

| 参数 | 值 |
|------|-----|
| 参数量 | 87M |
| 层数 | 12 |
| 隐藏维度 | 768 |
| 注意力头数 | 12 (Q) / 4 (KV, GQA) |
| FFN 中间层 | 2048 (SwiGLU) |
| 位置编码 | RoPE (θ=100000) |
| 归一化 | RMSNorm |
| 词表大小 | 15K (BBPE) |
| 最大序列长度 | 512 |

## 权重文件

```
checkpoints/
├── pretrain/
│   ├── step_98000.pt     # 预训练中期
│   ├── step_184000.pt    # 预训练后期
│   └── step_184914.pt    # 预训练最终
└── sft/
    ├── step_315000.pt    # SFT 中期
    ├── step_330000.pt    # SFT 后期
    └── step_338100.pt    # SFT 最终（推荐用于对话）
```

## 使用方法

### 环境

```bash
pip install torch tokenizers
```

### 加载模型

```python
import json
import torch
from tokenizers import Tokenizer
from model.config import MiniMindConfig
from model.transformer import MiniMindForCausalLM

# 加载分词器
tokenizer = Tokenizer.from_file("tokenizer.json")

# 加载配置
with open("model_config.json") as f:
    cfg = json.load(f)
cfg["vocab_size"] = tokenizer.get_vocab_size()
config = MiniMindConfig(**cfg)

# 加载模型
model = MiniMindForCausalLM(config)
checkpoint = torch.load("step_338100.pt", map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)
model.eval()
```

### 多轮对话

项目代码见 [GitHub](https://github.com/duhangfu-del/MyLLM)：

```bash
# 标准推理
python chat.py --checkpoint step_338100.pt --mode sft

# KV-Cache 加速推理
python chat_kvcache.py --checkpoint step_338100.pt --mode sft
```

## 训练

| 阶段 | 学习率 | Batch Size | Epochs | 数据 |
|------|--------|-----------|--------|------|
| 预训练 | 1e-3 | 48×3 | 3 | SpongeBobPRO 中文语料 |
| SFT | 2e-5 | 32×4 | 3 | Belle 3.5M 中文指令 |

- 优化器：AdamW
- 学习率调度：warmup (3%) + 余弦退火
- 混合精度：bfloat16
