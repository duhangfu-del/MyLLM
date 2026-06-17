"""
MiniMind 多轮对话推理脚本（支持 KV Cache，需配合修改后的模型文件）
用法：
    python chat_kvcache.py --checkpoint checkpoints/sft/step_338100.pt --mode sft
"""

import json
import torch
import argparse
from tokenizers import Tokenizer
from model.config import MiniMindConfig
from model.transformer import MiniMindForCausalLM

SYSTEM_PROMPT = "你是一个乐于助人的AI助手，请用中文回答问题。"

CHAT_TEMPLATE_START = "<|im_start|>system\n{system_prompt}<|im_end|>\n"
CHAT_TEMPLATE_USER = "<|im_start|>user\n{content}<|im_end|>\n"
CHAT_TEMPLATE_ASSISTANT = "<|im_start|>assistant\n{content}<|im_end|>\n"

def build_chat_prompt(history, user_msg, use_system=False, system_prompt=SYSTEM_PROMPT):
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
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mode', type=str, default='sft', choices=['sft', 'pretrain'])
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--system', action='store_true', help='添加系统提示')
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer_files/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    with open("configs/model_config.json", 'r') as f:
        model_cfg_dict = json.load(f)
    model_cfg_dict['vocab_size'] = vocab_size
    model_config = MiniMindConfig(**model_cfg_dict)
    model = MiniMindForCausalLM(model_config).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"✅ 模型已加载，参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    history = [] if args.mode == 'sft' else None
    eos_id = tokenizer.token_to_id("<|im_end|>") or tokenizer.token_to_id("<eos>")

    while True:
        try:
            user_input = input("👤 User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        if args.mode == 'pretrain':
            prompt = user_input
        else:
            prompt = build_chat_prompt(history, user_input, use_system=args.system)

        enc = tokenizer.encode(prompt)
        max_len = 512 - args.max_new_tokens  # 与训练时 max_seq_len 保持一致
        token_ids = enc.ids
        if len(token_ids) > max_len:
            token_ids = token_ids[-max_len:]   # 尾部截断，保留最近对话
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(args.device)

        print("🤖 Assistant: ", end="", flush=True)

        generated = input_ids.clone()
        past_key_values = None
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                current_input = generated if past_key_values is None else generated[:, -1:]
                logits, past_key_values = model(
                    current_input,
                    past_key_values=past_key_values,
                    use_cache=True
                )
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
        response = tokenizer.decode(generated[0, input_ids.size(1):].tolist())
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        if args.mode == 'sft':
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()