"""
MiniMind SFT 训练脚本（单卡，集成 LLM-as-Judge Benchmark，支持自动续训）

核心功能：
1. 从指定的预训练权重加载模型，仅微调 assistant 部分（通过 labels 中的 -100 实现）
2. 使用外部模块 `training.utils.lr_scheduler` 中的 `get_lr` 实现学习率调度：
   - 百分比 warmup（默认 3% 总步数），线性从 0 升至峰值
   - 余弦退火至 0
3. 支持 SwanLab 实时监控训练过程，每次评测直接上传 8 个指标：
   - fluency_avg3 / fluency_pass3
   - factuality_avg3 / factuality_pass3
   - instruction_following_avg3 / instruction_following_pass3
   - mean_avg3 / mean_pass3
4. 自动保存检查点，限制保留数量，保存完整的训练状态（含 optimizer_step）
5. 支持断点续训：自动检测 `checkpoints/sft` 下最新的 checkpoint 并恢复训练状态
6. 第 0 步评测仅在全新训练时执行，续训时跳过
7. 训练中定期评测（每 eval_interval 步），训练结束后进行最终评测

用法：
    python training/sft/train_sft.py --config configs/sft_config.yaml
"""

import os
import sys
import time
import math
import glob
import yaml
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 将项目根目录添加到 Python 搜索路径，确保可以导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.config import MiniMindConfig
from model.transformer import MiniMindForCausalLM
from data.sft_dataset import SFTDataset, sft_collate_fn

# 导入预训练脚本中使用的学习率调度函数（放在 training/utils/lr_scheduler.py 中）
from training.utils.lr_scheduler import get_lr

try:
    from evaluation.benchmarks.sft_eval import run_sft_benchmark
except ImportError:
    run_sft_benchmark = None
    print("Warning: sft_eval module not found. Benchmark will be disabled.")

try:
    import swanlab
except ImportError:
    swanlab = None


# ==============================================================================
#  配置文件加载
# ==============================================================================
def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ==============================================================================
#  优化器构建
# ==============================================================================
def create_optimizer(model, cfg):
    """
    创建 AdamW 优化器
    对 bias 和归一化层的参数不施加权重衰减（weight_decay）
    """
    no_decay = ['bias', 'LayerNorm.weight', 'rms_norm.weight']
    grouped = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': cfg['training']['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    lr = float(cfg['training']['learning_rate'])
    return AdamW(grouped, lr=lr, betas=(0.9, 0.95))


# ==============================================================================
#  检查点保存
# ==============================================================================
def save_checkpoint(model, optimizer, scaler, global_step, optimizer_step, cfg, save_dir):
    """
    保存训练检查点，包含模型、优化器、scaler、当前步数等信息。
    同时清理旧检查点，只保留最近 save_total_limit 个。
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{global_step}.pt")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'global_step': global_step,
        'optimizer_step': optimizer_step,
        'config': cfg
    }
    torch.save(checkpoint, path)

    # 清理旧文件
    ckpts = sorted(glob.glob(os.path.join(save_dir, "step_*.pt")))
    save_limit = cfg['log'].get('save_total_limit', 3)
    for old in ckpts[:-save_limit]:
        os.remove(old)
    print(f"Checkpoint saved: {path}")


# ==============================================================================
#  断点续训：加载最新检查点
# ==============================================================================
def load_latest_checkpoint(save_dir, model, optimizer, scaler):
    """
    从指定目录加载最新的 checkpoint（文件名按 step 排序取最大），恢复训练状态。
    返回 (global_step, optimizer_step)。若目录为空则返回 (0, 0)。
    """
    ckpts = sorted(glob.glob(os.path.join(save_dir, "step_*.pt")))
    if not ckpts:
        return 0, 0
    latest = ckpts[-1]
    print(f"Resuming from checkpoint: {latest}")
    ckp = torch.load(latest, map_location='cpu')
    model.load_state_dict(ckp['model_state_dict'])
    optimizer.load_state_dict(ckp['optimizer'])
    if scaler is not None and ckp.get('scaler') is not None:
        scaler.load_state_dict(ckp['scaler'])
    return ckp['global_step'], ckp.get('optimizer_step', 0)


# ==============================================================================
#  主训练函数
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sft_config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置随机种子，保证可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---------- 加载分词器 ----------
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer_files/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # ---------- 构建模型 ----------
    with open(cfg['model_config'], 'r') as f:
        model_cfg_dict = json.load(f)
    model_cfg_dict['vocab_size'] = vocab_size
    model_config = MiniMindConfig(**model_cfg_dict)
    model = MiniMindForCausalLM(model_config).to(device)

    # 加载预训练权重（基座模型）
    pretrain_path = cfg['pretrain_checkpoint']
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained weights from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        # 兼容多种保存格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Error: Pretrain checkpoint {pretrain_path} not found!")
        return

    # ---------- 混合精度 ----------
    use_amp = cfg['amp']['use_amp']
    amp_dtype = torch.bfloat16 if cfg['amp']['amp_dtype'] == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # ---------- 数据集 ----------
    dataset = SFTDataset(
        data_path=cfg['data']['train_path'],
        tokenizer=tokenizer,
        max_seq_len=cfg['data']['max_seq_len']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        collate_fn=sft_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # ---------- 优化器 ----------
    optimizer = create_optimizer(model, cfg)
    accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)

    # ---------- 计算总步数 ----------
    global_steps_per_epoch = len(dataloader)                      # 每个 epoch 的 micro batch 数
    optimizer_steps_per_epoch = global_steps_per_epoch // accumulation_steps
    total_optimizer_steps = optimizer_steps_per_epoch * cfg['training']['num_epochs']  # 总优化器步数
    total_global_steps = global_steps_per_epoch * cfg['training']['num_epochs']        # 总全局步数

    print(f"Global steps per epoch: {global_steps_per_epoch}, total: {total_global_steps}")
    print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}, total: {total_optimizer_steps}")

    # 学习率相关参数
    peak_lr = float(cfg['training']['learning_rate'])
    warmup_ratio = float(cfg['training'].get('warmup_ratio', 0.03))

    # ---------- DeepSeek API Key 与 benchmark 数据路径 ----------
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", cfg.get('deepseek_api_key', None))
    benchmark_data_path = cfg.get('benchmark', {}).get('data_path', 'evaluation/benchmarks/data/100miniSponge.jsonl')

    # ---------- 断点续训：尝试加载最新检查点 ----------
    # 如果 checkpoint 目录存在且包含文件，则恢复训练；否则从头开始
    save_dir = 'checkpoints/sft'
    global_step, optimizer_step = load_latest_checkpoint(save_dir, model, optimizer, scaler)

    # ---------- SwanLab ----------
    swanlab_run = None
    if cfg['swanlab'].get('use_swanlab', False) and swanlab is not None:
        swanlab.init(
            project=cfg['swanlab']['project'],
            experiment_name=cfg['swanlab']['run_name'],
            logdir=cfg['swanlab'].get('log_dir', 'logs/swanlab_sft')
        )
        swanlab_run = swanlab
        print("SwanLab enabled.")

    # ========== 第零步评测（仅在全新训练时执行） ==========
    # 如果 global_step == 0，说明是从头开始训练，此时进行初始评测；否则跳过（续训）
    if global_step == 0 and run_sft_benchmark is not None and deepseek_api_key:
        print("Running Step 0 benchmark evaluation...")
        model.eval()
        try:
            bench_results = run_sft_benchmark(
                model, tokenizer,
                data_path=benchmark_data_path,
                api_key=deepseek_api_key,
                device=device
            )
            print(f"Step 0 Benchmark Results: {bench_results}")
            if swanlab_run:
                swanlab_run.log(bench_results, step=0)
        except Exception as e:
            print(f"Step 0 benchmark failed: {e}")
        model.train()

    # ========== 训练循环 ==========
    print("Starting SFT training...")
    # 如果续训，global_step 和 optimizer_step 已经恢复；否则为 0
    # 计算起始 epoch（粗略计算，因为续训时可能 epoch 未完成）
    start_epoch = global_step // global_steps_per_epoch if global_steps_per_epoch > 0 else 0

    for epoch in range(start_epoch, cfg['training']['num_epochs']):
        epoch_loss = 0.0
        log_loss = 0.0
        start_time = time.time()

        # 若续训，需要跳过当前 epoch 中已经完成的步数
        skip_steps = global_step % global_steps_per_epoch if epoch == start_epoch else 0

        for step, (input_ids, labels) in enumerate(dataloader):
            # 跳过已完成的步数（续训场景）
            if skip_steps > 0:
                skip_steps -= 1
                continue

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # ---------- 前向传播 ----------
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

            # ---------- 反向传播 ----------
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ---------- 梯度累积更新 ----------
            if (step + 1) % accumulation_steps == 0:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    optimizer.step()

                # 更新优化器步数并计算当前学习率
                optimizer_step = (epoch * global_steps_per_epoch + step) // accumulation_steps
                lr = get_lr(optimizer_step, total_optimizer_steps, peak_lr, warmup_ratio)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.zero_grad(set_to_none=True)

            # 记录真实的损失（还原缩放）
            real_loss = loss.item() * accumulation_steps
            epoch_loss += real_loss
            log_loss += real_loss
            global_step = epoch * global_steps_per_epoch + step + 1

            # ---------- 日志打印 ----------
            # 注意：此时 lr 变量在累积步时已被赋值，但在非累积步时可能未定义。
            # 为了保证日志打印不出错，我们始终使用 optimizer.param_groups[-1]['lr'] 来获取当前学习率。
            current_lr = optimizer.param_groups[-1]['lr']

            if global_step % cfg['log']['log_interval'] == 0:
                avg_loss = log_loss / cfg['log']['log_interval']
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{cfg['training']['num_epochs']} | "
                      f"Step {global_step}/{total_global_steps} (opt {optimizer_step}/{total_optimizer_steps}) | "
                      f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
                if swanlab_run:
                    swanlab_run.log({
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "optimizer_step": optimizer_step
                    }, step=global_step)
                log_loss = 0.0
                start_time = time.time()

            # ---------- 保存检查点 ----------
            if global_step % cfg['log']['save_interval'] == 0:
                save_checkpoint(model, optimizer, scaler, global_step, optimizer_step, cfg, save_dir)

            # ---------- 定期评测 ----------
            if global_step % cfg['log']['eval_interval'] == 0 and deepseek_api_key and run_sft_benchmark is not None:
                print("Running SFT benchmark evaluation...")
                model.eval()
                try:
                    bench_results = run_sft_benchmark(
                        model, tokenizer,
                        data_path=benchmark_data_path,
                        api_key=deepseek_api_key,
                        device=device
                    )
                    print(f"Benchmark Results at step {global_step}: {bench_results}")
                    if swanlab_run:
                        swanlab_run.log(bench_results, step=global_step)
                except Exception as e:
                    print(f"Benchmark evaluation failed: {e}")
                model.train()

            if global_step >= total_global_steps:
                break

        if global_step >= total_global_steps:
            break

    # ========== 训练结束后最终评测 ==========
    if run_sft_benchmark is not None and deepseek_api_key:
        print("Running final benchmark evaluation...")
        model.eval()
        try:
            final_results = run_sft_benchmark(
                model, tokenizer,
                data_path=benchmark_data_path,
                api_key=deepseek_api_key,
                device=device
            )
            print(f"Final Benchmark Results: {final_results}")
            if swanlab_run:
                swanlab_run.log(final_results, step=global_step)
        except Exception as e:
            print(f"Final benchmark failed: {e}")

    # 最终保存
    save_checkpoint(model, optimizer, scaler, global_step, optimizer_step, cfg, save_dir)
    print("SFT training finished.")


if __name__ == '__main__':
    main()