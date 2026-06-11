"""
训练辅助工具：
- 分布式初始化与清理
- 主进程判断
- 日志封装（统一格式）
- SkipBatchSampler（支持精确从某个 step 续训）
"""
import os
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import math

def is_main_process():
    """判断当前进程是否为主进程（rank 0）"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

def init_distributed_mode():
    """
    初始化分布式训练环境（从 torchrun 自动读取环境变量）
    返回 local_rank，单卡时返回 0
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank, world_size, local_rank = 0, 1, 0
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class Logger:
    """统一日志打印，主进程才打印，避免多卡输出重复"""
    def __init__(self, tag=''):
        self.tag = tag
    def __call__(self, msg):
        if is_main_process():
            prefix = f'[{self.tag}] ' if self.tag else ''
            print(f'{prefix}{msg}')

class SkipBatchSampler(Sampler):
    """
    支持跳过若干 step 的批次采样器（用于精确续训）
    - 可传入 DistributedSampler（多卡）或普通索引列表（单卡）
    - skip 参数表示要跳过的样本数（step*batch_size）
    """
    def __init__(self, sampler, batch_size, skip=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip = skip

    def __iter__(self):
        if isinstance(self.sampler, Sampler):
            iterator = iter(self.sampler)
        else:
            iterator = iter(self.sampler)  # 单卡时传入的是 indices 列表
        # 跳过前 skip 个样本（对应已训练的步数*batch_size）
        for _ in range(self.skip):
            try:
                next(iterator)
            except StopIteration:
                break
        # 按照 batch_size 分组产出批次索引
        batch = []
        for idx in iterator:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        if isinstance(self.sampler, Sampler):
            total = len(self.sampler)
        else:
            total = len(self.sampler)
        remaining = max(0, total - self.skip)
        return math.ceil(remaining / self.batch_size)