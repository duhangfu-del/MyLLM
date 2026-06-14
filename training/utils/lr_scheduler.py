import math

def get_lr(current_step, total_steps, peak_lr, warmup_ratio=0.03):
    """
    按 current_step 计算当前学习率
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