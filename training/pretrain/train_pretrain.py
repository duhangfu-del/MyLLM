"""
MiniMind 预训练脚本（单卡版，可直接运行）
使用混合精度、梯度累积、SwanLab（可选）、C3/XCOPA 评测
"""

import os
import sys
import time
import yaml
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR

# 确保项目根目录在 path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.config import MiniMindConfig
from model.transformer import MiniMindForCausalLM
from data.Pretrain_dataset import PretrainDataset
from evaluation.benchmarks.evaluator import run_benchmark

# SwanLab 可选
try:
    import swanlab
except ImportError:
    swanlab = None


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
    return AdamW(grouped, lr=cfg['training']['learning_rate'], betas=(0.9, 0.95))


def create_scheduler(optimizer, cfg, total_steps):
    warmup_steps = cfg['training']['warmup_steps']
    min_lr = cfg['training']['min_lr']

    # 线性 warmup：学习率从 0 线性增长到初始 lr
    def lr_lambda_warmup(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)   # step=0 时 lr=0，逐步增长到 1
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda_warmup)
    # 余弦退火到 min_lr
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
    # 拼接两个调度器
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return scheduler


def save_checkpoint(model, optimizer, scheduler, scaler, global_step, cfg, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{global_step}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'global_step': global_step,
        'config': cfg
    }, path)
    # 保留最近的 3 个
    ckpts = sorted([f for f in os.listdir(save_dir) if f.endswith('.pt')])
    for old in ckpts[:-3]:
        os.remove(os.path.join(save_dir, old))
    print(f"Checkpoint saved: {path}")


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

    # 分词器
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer_files/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # 模型
    with open(cfg['model_config'], 'r') as f:
        model_cfg_dict = json.load(f)
    model_cfg_dict['vocab_size'] = vocab_size
    model = MiniMindForCausalLM(MiniMindConfig(**model_cfg_dict)).to(device)

    # 混合精度（修复新版 API）
    use_amp = cfg['amp']['use_amp']
    amp_dtype = torch.bfloat16 if cfg['amp']['amp_dtype'] == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # 数据集
    dataset = PretrainDataset(cfg['data']['meta_path'], cfg['data']['bin_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['training']['batch_size'],
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # 优化器与调度器
    optimizer = create_optimizer(model, cfg)
    total_steps = cfg['training']['max_steps']
    scheduler = create_scheduler(optimizer, cfg, total_steps)

    global_step = 0
    if args.resume:
        ckp = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckp['model_state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
        if ckp['scaler'] is not None and scaler is not None:
            scaler.load_state_dict(ckp['scaler'])
        global_step = ckp['global_step']
        print(f"Resumed from step {global_step}")

    # SwanLab
    swanlab_run = None
    if cfg['swanlab']['use_swanlab'] and swanlab is not None:
        swanlab.init(project=cfg['swanlab']['project'],
                     experiment_name=cfg['swanlab']['run_name'],
                     logdir=cfg['swanlab'].get('log_dir', 'logs/swanlab'))
        swanlab_run = swanlab
        print("SwanLab enabled.")

    # Step 0 评测
    print("Running Step 0 evaluation...")
    model.eval()
    c3_path = "evaluation/benchmarks/data/clue_c3_eval_500.jsonl"
    xcopa_path = "evaluation/benchmarks/data/xcopa_zh_merged.jsonl"
    res = run_benchmark(model, tokenizer, c3_path, xcopa_path)
    print(f"Step 0 results: {res}")
    if swanlab_run:
        swanlab_run.log({"benchmark/c3_acc": res.get('c3_accuracy', 0),
                         "benchmark/xcopa_acc": res.get('xcopa_accuracy', 0)}, step=0)
    model.train()

    # 训练循环
    print("Starting training...")
    for epoch in range(cfg['training']['num_epochs']):
        epoch_loss = 0.0
        log_loss = 0.0
        start = time.time()
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                   shift_labels.view(-1), ignore_index=-100)
                acc = cfg['training'].get('gradient_accumulation_steps', 1)
                loss = loss / acc

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % acc == 0:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            real_loss = loss.item() * acc
            epoch_loss += real_loss
            log_loss += real_loss
            global_step += 1

            if global_step % cfg['log']['log_interval'] == 0:
                avg_loss = log_loss / cfg['log']['log_interval']
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start
                print(f"Epoch {epoch+1}/{cfg['training']['num_epochs']} | "
                      f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
                if swanlab_run:
                    swanlab_run.log({"train/loss": avg_loss, "train/lr": lr}, step=global_step)
                log_loss = 0.0
                start = time.time()

            if global_step % cfg['log']['save_interval'] == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, global_step, cfg, 'checkpoints/pretrain')

            if global_step % cfg['log']['eval_interval'] == 0:
                model.eval()
                res = run_benchmark(model, tokenizer, c3_path, xcopa_path)
                print(f"Eval at step {global_step}: {res}")
                if swanlab_run:
                    swanlab_run.log({"benchmark/c3_acc": res.get('c3_accuracy', 0),
                                     "benchmark/xcopa_acc": res.get('xcopa_accuracy', 0)}, step=global_step)
                model.train()

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    save_checkpoint(model, optimizer, scheduler, scaler, global_step, cfg, 'checkpoints/pretrain')
    print("Training finished.")


if __name__ == '__main__':
    main()