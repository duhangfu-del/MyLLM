"""
SFT 对话数据集

功能：
1. 从 ChatML JSONL 文件中加载对话数据
2. 将对话转换为模型所需的输入序列
3. 生成 labels，仅对 assistant 回复部分计算损失（user 部分被忽略）
4. 支持动态批次填充

数据格式要求：
每行一个 JSON 对象，包含 "conversations" 字段，该字段是一个列表，
列表中每个元素为 {"role": "user"/"assistant", "content": "对话内容"}。
例如：
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助你的？"}]}
"""

import torch
from torch.utils.data import Dataset
import json
from tokenizers import Tokenizer


class SFTDataset(Dataset):
    """SFT 数据集类"""

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 512):
        """
        Args:
            data_path:   训练数据文件路径（JSONL 格式）
            tokenizer:   已训练好的 tokenizers.Tokenizer 实例
            max_seq_len: 最大序列长度，超出的部分将被截断
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 读取所有对话样本
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():                              # 跳过空行
                    item = json.loads(line)                   # 解析 JSON
                    self.samples.append(item['conversations']) # 只取对话列表

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回第 idx 个样本的处理结果

        返回格式：一个元组 (input_ids, labels)
        - input_ids: 编码后的 token 序列（LongTensor）
        - labels: 与 input_ids 等长，仅 assistant 部分为真实 token id，其余为 -100（忽略损失）
        """
        conversations = self.samples[idx]  # 获取对话列表

        # ========== 1. 构建 ChatML 格式的完整文本 ==========
        prompt = ""
        for turn in conversations:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                # 用户消息：<|im_start|>user\n 消息内容 <|im_end|>\n
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                # 助手消息：<|im_start|>assistant\n 消息内容 <|im_end|>\n
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            # 如果未来有 system 角色，可在此处扩展

        # ========== 2. 编码为 token IDs ==========
        enc = self.tokenizer.encode(prompt)
        input_ids = enc.ids[:self.max_seq_len]  # 截断到最大长度

        # ========== 3. 生成 labels：仅 assistant 部分参与损失计算 ==========
        # 初始化为 -100，PyTorch 的交叉熵损失会忽略值为 -100 的位置
        labels = [-100] * len(input_ids)

        offset = 0  # 当前已处理的 token 位置偏移
        for turn in conversations:
            role = turn['role']
            content = turn['content']

            # 重新构建当前轮次的文本片段，以便独立编码
            if role == 'user':
                segment = f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                segment = f"<|im_start|>assistant\n{content}<|im_end|>\n"
            else:
                continue  # 未知角色跳过

            seg_enc = self.tokenizer.encode(segment)
            seg_len = len(seg_enc.ids)

            # 如果是 assistant 的消息，将其对应位置的 labels 设为真实的 token id
            if role == 'assistant':
                for i in range(offset, min(offset + seg_len, len(labels))):
                    labels[i] = input_ids[i]   # 填入真实 token

            offset += seg_len
            if offset >= len(labels):          # 如果总长度已超过截断长度，停止处理后续轮次
                break

        # 转换为 PyTorch 张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels


def sft_collate_fn(batch):
    """
    批次整理函数：动态填充变长序列

    Args:
        batch: 由 __getitem__ 返回的 (input_ids, labels) 组成的列表

    Returns:
        input_ids: 填充后的 batch (batch_size, max_seq_len_in_batch)
        labels:    填充后的 labels，填充值为 -100（忽略损失）
    """
    input_ids_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # 注意：这里的 pad_token_id 必须与 tokenizer 中 <pad> 的 ID 一致
    # 你的 tokenizer 中 <pad> 的 ID 为 5
    pad_token_id = 5

    # 以批内最大长度进行填充
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100   # labels 的填充值始终为 -100
    )
    return input_ids, labels