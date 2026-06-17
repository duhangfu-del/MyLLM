# MiniMind 数据集

本数据集包含 MiniMind 项目所需的全部数据：预训练语料、SFT 指令数据、评测基准和分词器。

## 内容

```
├── pretrain/
│   ├── SpongeBobPRO_pretrain_512_final.bin   # 预训练二进制数据 (token ids)
│   └── SpongeBobPRO_pretrain_512_final.meta   # 元信息文件
├── sft/
│   ├── sft_train.jsonl                         # SFT 训练数据 (ChatML 格式)
│   └── train_3.5M_CN.json                      # Belle 3.5M 原始中文指令数据
├── benchmark/
│   ├── clue_c3_eval_500.jsonl                  # C3 中文阅读理解评测 (500 条)
│   ├── xcopa_zh_merged.jsonl                   # XCOPA 中文因果推理评测
│   └── 100miniSponge.jsonl                     # LLM-as-Judge 评测数据
└── tokenizer/
    ├── tokenizer.json                          # BBPE 分词器
    ├── vocab.json                              # 词表
    ├── merges.txt                              # BPE 合并规则
    └── tokenizer_config.json                   # 分词器配置
```

## 数据详情

### 预训练数据

- 来源：SpongeBobPRO 中文语料
- 格式：预处理后的 token id 二进制文件 + Arrow meta 文件
- 每条样本长度：512 tokens
- 词表大小：15,000 (BBPE)

### SFT 数据

- 来源：Belle 3.5M 中文指令数据集
- 格式：ChatML (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`)
- `sft_train.jsonl` 为预处理后的训练格式，`train_3.5M_CN.json` 为原始数据

### 评测数据

- **C3**：中文阅读理解选择题，500 条
- **XCOPA**：中文因果推理，判断原因/结果
- **100miniSponge**：用于 LLM-as-Judge 自动评分的开放式问答

### 分词器

- BBPE (Byte-level BPE) 分词器
- 词表大小：15,000
- 使用 HuggingFace `tokenizers` 库训练

## 使用方法

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
encoded = tokenizer.encode("你好世界")
print(encoded.ids)    # token ids
print(encoded.tokens) # token 文本
```

## 项目代码

[GitHub - MyLLM](https://github.com/duhangfu-del/MyLLM)
