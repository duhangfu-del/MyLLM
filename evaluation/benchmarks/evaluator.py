"""
Benchmark 评测模块（适配自研模型和 tokenizers.Tokenizer）
支持 C3 和 XCOPA 数据集，精确计算选项部分 loss
"""

import json
import torch
import torch.nn.functional as F


def eval_multiple_choice(model, tokenizer, context, choices, label_idx, max_length=512):
    """
    多选题评测：计算每个选项的困惑度（只计算选项部分），选择 loss 最低的选项
    """
    # 单独编码 context（不添加特殊 token）
    context_enc = tokenizer.encode(context)
    context_ids = context_enc.ids

    losses = []
    for choice in choices:
        # 编码 choice
        choice_enc = tokenizer.encode(choice)
        choice_ids = choice_enc.ids

        # 直接在 token 维度拼接，避免 BPE 边界合并问题
        full_ids = context_ids + choice_ids
        full_ids = full_ids[:max_length]          # 截断

        # 获取模型所在设备
        device = next(model.parameters()).device
        input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_ids)[0]          # 模型返回 (logits, present_key_values)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_all = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        # choice 在 shift 后的起始索引 = len(context_ids) - 1
        choice_start = max(0, len(context_ids) - 1)

        if choice_start < loss_all.numel():
            choice_loss = loss_all[choice_start:].mean().item()
        else:
            choice_loss = loss_all.mean().item()   # 截断严重时的回退

        losses.append(choice_loss)

    pred_idx = losses.index(min(losses))
    return 1 if pred_idx == label_idx else 0


def eval_c3(model, tokenizer, data_path):
    correct = 0
    total = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            context_text = ''.join(data['context'])
            question = data['question']
            choices = data['choice']
            answer = data['answer']
            if answer not in choices:
                continue
            label_idx = choices.index(answer)
            full_context = context_text + question
            result = eval_multiple_choice(model, tokenizer, full_context, choices, label_idx)
            correct += result
            total += 1
    return correct / total if total > 0 else 0


def eval_xcopa(model, tokenizer, data_path):
    correct = 0
    total = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            premise = data['premise']
            choices = [data['choice1'], data['choice2']]
            label_idx = data['label']
            question_type = data['question']
            if question_type == 'cause':
                context = f"{premise}这是因为："
            else:
                context = f"{premise}所以："
            result = eval_multiple_choice(model, tokenizer, context, choices, label_idx)
            correct += result
            total += 1
    return correct / total if total > 0 else 0


def run_benchmark(model, tokenizer, c3_path, xcopa_path):
    from torch.nn.parallel import DistributedDataParallel
    # 解包 DDP 和 torch.compile
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)

    raw_model.eval()

    results = {}
    print("\n" + "=" * 60)
    print("开始 Benchmark 评测")
    print("=" * 60)

    try:
        print(f"评测 C3 数据集: {c3_path}")
        c3_acc = eval_c3(raw_model, tokenizer, c3_path)
        results['c3_accuracy'] = c3_acc
        print(f"✓ C3 Accuracy: {c3_acc:.4f} ({c3_acc*100:.2f}%)")
    except Exception as e:
        print(f"✗ C3 evaluation failed: {e}")
        results['c3_accuracy'] = 0.0

    try:
        print(f"评测 XCOPA 数据集: {xcopa_path}")
        xcopa_acc = eval_xcopa(raw_model, tokenizer, xcopa_path)
        results['xcopa_accuracy'] = xcopa_acc
        print(f"✓ XCOPA Accuracy: {xcopa_acc:.4f} ({xcopa_acc*100:.2f}%)")
    except Exception as e:
        print(f"✗ XCOPA evaluation failed: {e}")
        results['xcopa_accuracy'] = 0.0

    print("=" * 60 + "\n")
    raw_model.train()
    return results