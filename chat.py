"""
MiniMind 多轮对话推理脚本（修复版）

改进：
- 禁用 KV Cache，每次传入完整序列，与评测环境完全一致
- 输入过长时从头部截断（保留开头的系统提示和最近的对话）
- 默认不添加系统提示（因为训练数据无 system 角色）
- 可通过 --system 启用系统提示（可选）

用法：
    python chat.py --checkpoint checkpoints/sft/step_338100.pt --mode sft
"""

import torch
import argparse
from tokenizers import Tokenizer
from model.config import MiniMindConfig
from model.transformer import MiniMindForCausalLM

DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的AI助手，请用中文回答问题。"

CHAT_TEMPLATE_START = "<|im_start|>system\n{system_prompt}<|im_end|>\n"
CHAT_TEMPLATE_USER = "<|im_start|>user\n{content}<|im_end|>\n"
CHAT_TEMPLATE_ASSISTANT = "<|im_start|>assistant\n{content}<|im_end|>\n"


def build_chat_prompt(history, user_msg, mode, use_system=False, system_prompt=DEFAULT_SYSTEM_PROMPT):
    if mode == "pretrain":
        return user_msg

    prompt = ""
    if use_system:
        prompt += CHAT_TEMPLATE_START.format(system_prompt=system_prompt)

    for turn in history:
        if turn["role"] == "user":
            prompt += CHAT_TEMPLATE_USER.format(content=turn["content"])
        elif turn["role"] == "assistant":
            prompt += CHAT_TEMPLATE_ASSISTANT.format(content=turn["content"])

    prompt += CHAT_TEMPLATE_USER.format(content=user_msg)
    prompt += "<|im_start|>assistant\n"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--mode', type=str, default='sft', choices=['sft', 'pretrain'])
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--system', action='store_true', help='添加系统提示（默认关闭）')
    parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer_files/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    model_config = MiniMindConfig(vocab_size=vocab_size)
    model = MiniMindForCausalLM(model_config)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    model.eval()
    print(f"✅ 模型已加载，参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"📝 模式: {'多轮对话' if args.mode == 'sft' else '文本续写'}"
          f"{' (系统提示已启用)' if args.system and args.mode == 'sft' else ''}")
    print("💬 开始对话（输入 'exit' 退出）\n")

    history = [] if args.mode == 'sft' else None

    while True:
        try:
            user_input = input("👤 User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        prompt = build_chat_prompt(history, user_input, args.mode,
                                   use_system=args.system, system_prompt=args.system_prompt)

        enc = tokenizer.encode(prompt)
        # 截断：保留开头（系统提示+早期对话）和末尾的最近对话，但确保不超出模型限制
        max_len = model_config.max_seq_len - args.max_new_tokens
        if len(enc.ids) > max_len:
            # 从头部截断：删除最早的对话内容，保留系统提示（如果有）和最近的几轮
            system_len = 0
            if args.system:
                system_prompt_text = CHAT_TEMPLATE_START.format(system_prompt=args.system_prompt)
                system_enc = tokenizer.encode(system_prompt_text)
                system_len = len(system_enc.ids)
            # 保留系统提示，然后从剩余部分的前端删除多余的 token
            prefix_ids = enc.ids[:system_len]
            rest_ids = enc.ids[system_len:]
            # 保留 rest_ids 的最后 (max_len - system_len) 个
            keep_rest = rest_ids[-(max_len - system_len):]
            enc.ids = prefix_ids + keep_rest
        input_ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(args.device)

        eos_id = tokenizer.token_to_id("<|im_end|>") or tokenizer.token_to_id("<eos>")
        if eos_id is None:
            print("⚠ 未找到结束 token，将使用最大长度截断")
            eos_id = -1  # 不可能匹配，但保留代码结构

        print("🤖 Assistant: ", end="", flush=True)

        # 生成（与 sft_eval.py 相同的简单循环，无 KV Cache）
        generated = input_ids.clone()
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                logits, _ = model(generated)  # 不使用 cache，传入完整序列
                next_logits = logits[:, -1, :] / args.temperature
            if args.top_k > 0:
                v, _ = torch.topk(next_logits, min(args.top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_id:
                break
            new_text = tokenizer.decode([next_token.item()])
            print(new_text, end="", flush=True)
            generated = torch.cat([generated, next_token], dim=-1)
        print()

        # 提取回复并更新历史
        response = tokenizer.decode(generated[0, input_ids.size(1):].tolist())
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        if args.mode == 'sft':
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()