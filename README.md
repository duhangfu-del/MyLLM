# MyLLM

从零实现的 Decoder-only Transformer 语言模型，支持预训练、SFT 微调和多轮对话推理。

## 模型架构

| 参数 | 值 |
|------|-----|
| 参数量 | 87M |
| 层数 | 12 |
| 隐藏维度 | 768 |
| 注意力头数 | 12 (Q) / 4 (KV) |
| FFN 中间层 | 2048 (SwiGLU) |
| 位置编码 | RoPE (θ=100000) |
| 归一化 | RMSNorm |
| 词表大小 | 15K (BBPE) |
| 最大序列长度 | 512 |
| 权重绑定 | 是 |

架构特性：分组查询注意力 (GQA)、SwiGLU 激活函数、RoPE 旋转位置编码、pre-norm 结构。

## 项目结构

```
MyLLM/
├── model/                    # 模型定义
│   ├── config.py             # 模型超参数配置
│   ├── transformer.py        # 完整模型 (MiniMindForCausalLM)
│   ├── decoder_layer.py      # Decoder Layer
│   └── layers/               # 底层组件
│       ├── attention.py      # GQA 注意力
│       ├── feedforward.py    # SwiGLU FFN
│       ├── rope.py           # RoPE 位置编码
│       └── rms_norm.py       # RMS 归一化
├── training/                 # 训练脚本
│   ├── pretrain/             # 预训练
│   ├── sft/                  # SFT 微调
│   └── utils/                # 工具函数 (lr_scheduler, checkpoint, logger 等)
├── data/                     # 数据处理
│   ├── Pretrain_dataset.py   # 预训练数据集
│   ├── sft_dataset.py        # SFT 数据集
│   └── tokenizer/            # BBPE 分词器
├── evaluation/               # 评测
│   └── benchmarks/           # C3, XCOPA 等中文基准测试
├── configs/                  # 配置文件
│   ├── model_config.json     # 模型结构配置
│   ├── pretrain_config.yaml  # 预训练超参数
│   └── sft_config.yaml       # SFT 超参数
├── chat.py                   # 标准多轮对话推理
├── chat_kvcache.py           # KV-Cache 加速推理
└── checkpoints/              # 模型权重
```

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA (推荐)

```bash
pip install torch tokenizers pyyaml swanlab
```

### 多轮对话

```bash
# 标准推理
python chat.py --checkpoint checkpoints/sft/step_338100.pt --mode sft

# KV-Cache 加速推理（更快）
python chat_kvcache.py --checkpoint checkpoints/sft/step_338100.pt --mode sft
```

参数说明：
- `--checkpoint`：模型权重路径
- `--mode sft`：SFT 对话模式（带历史记录）；`pretrain`：补全模式
- `--system`：添加系统提示词
- `--temperature 0.7`：生成温度
- `--max_new_tokens 256`：最大生成长度

### 预训练

```bash
python training/pretrain/train_pretrain.py --config configs/pretrain_config.yaml
```

预训练数据：SpongeBobPRO 中文语料，预处理为 512 长度的 token 序列。

### SFT 微调

```bash
python training/sft/train_sft.py --config configs/sft_config.yaml
```

SFT 数据：Belle 3.5M 中文指令数据，使用 ChatML 格式。

### 评测

```bash
python evaluation/benchmarks/sft_eval.py --checkpoint checkpoints/sft/step_338100.pt
```

支持 C3 和 XCOPA 中文基准测试，以及 LLM-as-Judge 自动评分。

## 训练配置

| 阶段 | 学习率 | Batch Size | Epochs | 数据 |
|------|--------|-----------|--------|------|
| 预训练 | 1e-3 | 48×3 (accum) | 3 | SpongeBobPRO |
| SFT | 2e-5 | 32×4 (accum) | 3 | Belle 3.5M |

- 优化器：AdamW (weight_decay=0.1/0.01)
- 学习率调度：warmup (3%) + 余弦退火
- 混合精度：bfloat16
- 实验管理：SwanLab

## 下载

### 模型权重

**SDK 下载：**
```bash
pip install modelscope
```
```python
from modelscope import snapshot_download
model_dir = snapshot_download('DuhangFu/MiniMind')
```

**Git 下载：**
```bash
git clone https://www.modelscope.cn/DuhangFu/MiniMind.git
```

### 数据

包含预训练语料、SFT 训练数据、Benchmark 评测数据、Tokenizer 文件。

**SDK 下载：**
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('DuhangFu/MiniMind')
```

**Git 下载：**
```bash
git lfs install
git clone https://www.modelscope.cn/datasets/DuhangFu/MiniMind.git
```

