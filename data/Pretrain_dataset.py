"""
预训练数据集加载类

功能：
1. 从 .bin + .meta 文件中加载已经预处理好的 token 数据。
2. 使用内存映射（np.memmap）避免将整个文件读入内存，大幅节省内存。
3. 每个样本长度固定为 512，无需 padding，可直接堆叠成 batch。
4. 返回 (input_ids, labels) 元组，便于 DataLoader 直接解包。

数据来源：SpongeBobPRO 预训练数据，已通过预处理脚本将所有文本拼接并切割成固定长度 512 的块。
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json


class PretrainDataset(Dataset):
    """
    预训练数据集

    期望的元数据文件（.meta）格式：
    {
        "seq_len": 512,
        "num_chunks": 2958637,
        "dtype": "uint16",
        "vocab_size": 15000,
        ...
    }

    二进制文件（.bin）是一个连续存储的二维数组，形状为 [num_chunks, seq_len]，
    数据类型为 meta 中指定的 dtype。
    """

    def __init__(self, meta_path: str, bin_path: str):
        """
        Args:
            meta_path: 元数据文件路径（JSON 格式），包含：
                       - seq_len: 每条样本的 token 数量（固定 512）
                       - num_chunks: 样本总数
                       - dtype: 二进制文件中 token id 的数据类型（如 'uint16'）
            bin_path:  二进制文件路径，存储所有样本的 token id。
        """
        # 读取元信息
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        self.seq_len = self.meta['seq_len']            # 每条样本的长度，应为 512
        self.num_samples = self.meta['num_chunks']     # 总样本数
        self.dtype = np.dtype(self.meta['dtype'])      # 数据类型，如 np.uint16

        # 使用内存映射打开二进制文件
        # 'r' 表示只读，shape 指定为二维数组，系统不会一次性加载全部数据到内存。
        self.data = np.memmap(
            bin_path,
            dtype=self.dtype,
            mode='r',
            shape=(self.num_samples, self.seq_len)
        )

    def __len__(self):
        """返回数据集样本总数"""
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回第 idx 个样本

        返回格式：一个元组 (input_ids, labels)
        - input_ids: 长度为 seq_len 的一维 tensor，dtype=torch.long
        - labels: 与 input_ids 完全相同，因为预训练目标是下一个 token 预测，
                  后续在训练循环中会通过 shift 操作自动对齐。
        """
        # 从内存映射中读取第 idx 行，转换为普通的 Python 列表，再转为 tensor（避免 numpy 兼容问题）
        tokens = self.data[idx].tolist()  # 直接转为 list
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()  # 标签与输入完全相同
        return input_ids, labels