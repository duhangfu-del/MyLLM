"""
MiniMind 预训练脚本（单卡，百分比 warmup + 余弦退火至 0）
- 自动按总步数比例 warmup（默认 3%）
- 余弦退火到 0，无需手动调整步数
- 日志中显示 global step 和 optimizer step
- 兼容最新 PyTorch AMP 接口
"""

import os
import sys
import time
import math
import yaml
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.config import MiniMindConfig
from model.transformer import MiniMindForCausalLM
from data.Pretrain_dataset import PretrainDataset
from evaluation.benchmarks.evaluator import run_benchmark

try:
    import swanlab
except ImportError:
    swanlab = None


# ----------------------------------------------------------------------
#  学习率工具（百分比 warmup + 余弦退火至 0）
# ----------------------------------------------------------------------
def get_lr(current_step, total_steps, peak_lr, warmup_ratio=0.03):
    """
    返回当前步数的学习率
    - 前 warmup_ratio * total_steps 步：线性从 0 升至 peak_lr
    - 剩余步数：余弦退火到 0
    """
    warmup_steps = int(total_steps * warmup_ratio)
    if current_step < warmup_steps:
        # 线性 warmup
        return peak_lr * (current_step + 1) / max(1, warmup_steps)
    else:
        # 余弦退火到 0
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------
#  配置与优化器
# ----------------------------------------------------------------------
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_optimizer(model, cfg):
    no_decay = ['bias', 'LayerNorm.weight', 'rms_norm.weight']
    grouped = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg['training']['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    lr = float(cfg['training']['learning_rate'])
    return AdamW(grouped, lr=lr, betas=(0.9, 0.95))


# ----------------------------------------------------------------------
#  Checkpoint 操作
# ----------------------------------------------------------------------
def save_checkpoint(model, optimizer, scaler, global_step, cfg, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{global_step}.pt")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'global_step': global_step,
        'config': cfg
    }
    torch.save(checkpoint, path)
    # 保留最近几个
    ckpts = sorted([f for f in os.listdir(save_dir) if f.endswith('.pt')])
    for old in ckpts[:-cfg['log'].get('save_total_limit', 3)]:
        os.remove(os.path.join(save_dir, old))
    print(f"Checkpoint saved: {path}")


def load_checkpoint(resume_path, model, optimizer, scaler):
    ckp = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(ckp['model_state_dict'])
    optimizer.load_state_dict(ckp['optimizer'])
    if ckp.get('scaler') and scaler:
        scaler.load_state_dict(ckp['scaler'])
    return ckp.get('global_step', 0)


def get_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None
    ckpts = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    if not ckpts:
        return None
    latest = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    return os.path.join(save_dir, latest)


# ----------------------------------------------------------------------
#  主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---------- 分词器 ----------
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer_files/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # ---------- 模型 ----------
    with open(cfg['model_config'], 'r') as f:
        model_cfg_dict = json.load(f)
    model_cfg_dict['vocab_size'] = vocab_size
    model_config = MiniMindConfig(**model_cfg_dict)
    model = MiniMindForCausalLM(model_config).to(device)

    # ---------- 混合精度（修复警告：使用 torch.amp.GradScaler('cuda', ...)） ----------
    use_amp = cfg['amp']['use_amp']
    amp_dtype = torch.bfloat16 if cfg['amp']['amp_dtype'] == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # ---------- 数据集 ----------
    dataset = PretrainDataset(cfg['data']['meta_path'], cfg['data']['bin_path'])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # ---------- 优化器 ----------
    optimizer = create_optimizer(model, cfg)
    accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)

    # ---------- 步数计算 ----------
    global_steps_per_epoch = len(dataloader)                     # micro batch 数
    optimizer_steps_per_epoch = global_steps_per_epoch // accumulation_steps
    total_optimizer_steps = optimizer_steps_per_epoch * cfg['training']['num_epochs']
    total_global_steps = global_steps_per_epoch * cfg['training']['num_epochs']
    print(f"Global steps per epoch: {global_steps_per_epoch}, total: {total_global_steps}")
    print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}, total: {total_optimizer_steps}")

    # 学习率参数
    peak_lr = float(cfg['training']['learning_rate'])
    warmup_ratio = float(cfg['training'].get('warmup_ratio', 0.03))

    # ---------- 断点续训 ----------
    global_step = 0
    if args.resume:
        global_step = load_checkpoint(args.resume, model, optimizer, scaler)
        print(f"Resumed from checkpoint, global_step={global_step}")
    else:
        latest_ckpt = get_latest_checkpoint('checkpoints/pretrain')
        if latest_ckpt:
            global_step = load_checkpoint(latest_ckpt, model, optimizer, scaler)
            print(f"Auto-resumed from {latest_ckpt}, global_step={global_step}")

    # ---------- SwanLab ----------
    swanlab_run = None
    if cfg['swanlab'].get('use_swanlab', False) and swanlab is not None:
        swanlab.init(
            project=cfg['swanlab']['project'],
            experiment_name=cfg['swanlab']['run_name'],
            logdir=cfg['swanlab'].get('log_dir', 'logs/swanlab')
        )
        swanlab_run = swanlab
        print("SwanLab enabled.")

    # ---------- 第零步评测 ----------
    print("Running Step 0 evaluation...")
    model.eval()
    c3_path = "evaluation/benchmarks/data/clue_c3_eval_500.jsonl"
    xcopa_path = "evaluation/benchmarks/data/xcopa_zh_merged.jsonl"
    eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
    print(f"Step 0 results: {eval_results}")
    if swanlab_run:
        swanlab_run.log({
            "benchmark/c3_acc": eval_results.get('c3_accuracy', 0),
            "benchmark/xcopa_acc": eval_results.get('xcopa_accuracy', 0)
        }, step=0)
    model.train()

    # ---------- 训练循环 ----------
    print("Starting training...")
    optimizer_step = 0   # 提前定义，避免 UnboundLocalError
    for epoch in range(cfg['training']['num_epochs']):
        epoch_loss = 0.0
        log_loss = 0.0
        start = time.time()

        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 前向（使用 torch.amp.autocast('cuda', ...) 修复警告）
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss = loss / accumulation_steps

            # 反向
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积更新
            if (step + 1) % accumulation_steps == 0:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    optimizer.step()

                # 手动设置学习率（基于 optimizer step）
                optimizer_step = global_step // accumulation_steps
                lr = get_lr(optimizer_step, total_optimizer_steps, peak_lr, warmup_ratio)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.zero_grad(set_to_none=True)

            real_loss = loss.item() * accumulation_steps
            epoch_loss += real_loss
            log_loss += real_loss
            global_step += 1

            # ---------- 日志（显示总步数） ----------
            if global_step % cfg['log']['log_interval'] == 0:
                avg_loss = log_loss / cfg['log']['log_interval']
                optimizer_step = global_step // accumulation_steps
                elapsed = time.time() - start
                print(f"Epoch {epoch+1}/{cfg['training']['num_epochs']} | "
                      f"Step {global_step}/{total_global_steps} (opt {optimizer_step}/{total_optimizer_steps}) | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")
                if swanlab_run:
                    swanlab_run.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "optimizer_step": optimizer_step
                    }, step=global_step)
                log_loss = 0.0
                start = time.time()

            # ---------- 保存 checkpoint ----------
            if global_step % cfg['log']['save_interval'] == 0:
                save_checkpoint(model, optimizer, scaler, global_step, cfg, 'checkpoints/pretrain')

            # ---------- 评测 ----------
            if global_step % cfg['log']['eval_interval'] == 0:
                model.eval()
                eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
                print(f"Eval at step {global_step}: {eval_results}")
                if swanlab_run:
                    swanlab_run.log({
                        "benchmark/c3_acc": eval_results.get('c3_accuracy', 0),
                        "benchmark/xcopa_acc": eval_results.get('xcopa_accuracy', 0)
                    }, step=global_step)
                model.train()

            # 终止条件：optimizer step 达到总步数
            if optimizer_step >= total_optimizer_steps:
                break

        if optimizer_step >= total_optimizer_steps:
            break

    # 最终保存
    save_checkpoint(model, optimizer, scaler, global_step, cfg, 'checkpoints/pretrain')
    print("Training finished.")


if __name__ == '__main__':
    main()