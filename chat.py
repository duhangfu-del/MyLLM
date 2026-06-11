"""
MiniMind 多轮对话推理脚本（适配自研模型与 tokenizers 分词器）

功能：
1. 加载自研 MiniMindForCausalLM 模型及 tokenizers 分词器
2. 采用 ChatML 模板构建多轮对话输入
3. 使用 KV Cache 加速自回归生成
4. 支持 top‑k 采样和温度控制
5. 流式逐 token 输出，模拟打字效果
6. 支持预训练续写模式（可选）

用法：
    python chat.py --checkpoint checkpoints/pretrain/global_step_1000.pt
"""

import torch
import argparse
from tokenizers import Tokenizer

# 导入自研模型（假设 chat.py 放在项目根目录）
from model.config import MiniMindConfig
from model.transformer import MiniMindForCausalLM


# ==============================================================================
#  对话模板（ChatML 风格）
#  需与训练时使用的 special token 保持一致
# ==============================================================================
SYSTEM_PROMPT = "你是一个乐于助人的AI助手，请用中文回答问题。"

CHAT_TEMPLATE_START = "<|im_start|>system\n{system_prompt}<|im_end|>\n"
CHAT_TEMPLATE_USER = "<|im_start|>user\n{content}<|im_end|>\n"
CHAT_TEMPLATE_ASSISTANT = "<|im_start|>assistant\n{content}<|im_end|>\n"


def build_chat_prompt(history: list, user_msg: str, mode: str = "sft") -> str:
    """
    根据对话历史构建当前轮次的输入字符串。
    mode = "sft"   : 拼装完整多轮对话，最后加上 assistant 起始标记
    mode = "pretrain" : 仅将用户输入作为续写前缀
    """
    if mode == "pretrain":
        # 预训练续写：直接把用户输入作为续写开头
        return user_msg

    # SFT 多轮对话拼接
    prompt = CHAT_TEMPLATE_START.format(system_prompt=SYSTEM_PROMPT)
    for turn in history:
        if turn["role"] == "user":
            prompt += CHAT_TEMPLATE_USER.format(content=turn["content"])
        elif turn["role"] == "assistant":
            prompt += CHAT_TEMPLATE_ASSISTANT.format(content=turn["content"])
    # 当前用户输入
    prompt += CHAT_TEMPLATE_USER.format(content=user_msg)
    # 加上 assistant 起始标记，等待模型生成回复
    prompt += "<|im_start|>assistant\n"
    return prompt


def main():
    parser = argparse.ArgumentParser(description="MiniMind Chat")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径（.pt 文件）')
    parser.add_argument('--mode', type=str, default='sft', choices=['sft', 'pretrain'],
                        help='对话模式：sft 多轮对话，pretrain 续写')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='每次回复最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--top_k', type=int, default=50,
                        help='top‑k 采样（0 表示禁用）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # -------------------- 1. 加载分词器 --------------------
    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer_files/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # -------------------- 2. 构建模型并加载权重 --------------------
    model_config = MiniMindConfig(vocab_size=vocab_size)
    model = MiniMindForCausalLM(model_config)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # 兼容保存格式：支持直接 state_dict 或带 'model_state_dict' 键的字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ 未指定 checkpoint，使用随机初始化权重。")
    model = model.to(args.device)
    model.eval()
    print(f"✅ 模型已加载，参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"📝 模式: {'多轮对话' if args.mode == 'sft' else '文本续写'}")
    print("💬 开始对话（输入 'exit' 退出）\n")

    # -------------------- 3. 对话历史（仅 SFT 模式使用） --------------------
    history = [] if args.mode == 'sft' else None

    while True:
        try:
            user_input = input("👤 User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        if user_input.lower() in ["exit", "quit"]:
            print("👋 再见！")
            break
        if not user_input:
            continue

        # 构建输入文本
        prompt = build_chat_prompt(history, user_input, args.mode)

        # 编码并截断（保留生成空间）
        enc = tokenizer.encode(prompt)
        max_len = model_config.max_seq_len - args.max_new_tokens
        if len(enc.ids) > max_len:
            print(f"[警告] 输入过长 ({len(enc.ids)} tokens)，已截断")
            enc.ids = enc.ids[-max_len:]
        input_ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(args.device)

        # -------------------- 4. 自回归生成（使用模型内置的 generate） --------------------
        # eos_token_id 可设为 <|im_end|>，若无则置 None 用最大长度截断
        eos_id = tokenizer.token_to_id("<|im_end|>")
        if eos_id is None:
            eos_id = tokenizer.token_to_id("<eos>")  # 回退到 <eos>

        print("🤖 Assistant: ", end="", flush=True)

        # 我们将在生成过程中逐步打印 token
        generated = input_ids.clone()
        past_key_values = None
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                logits, past_key_values = model(
                    generated[:, -1:] if past_key_values is not None else generated,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_logits = logits[:, -1, :] / args.temperature

            # top‑k 过滤
            if args.top_k > 0:
                v, _ = torch.topk(next_logits, min(args.top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 判断是否结束
            if next_token.item() == eos_id:
                break

            # 将新 token 解码并打印
            new_text = tokenizer.decode([next_token.item()])
            print(new_text, end="", flush=True)

            generated = torch.cat([generated, next_token], dim=-1)

        print()  # 换行

        # 更新历史（SFT 模式）
        response = tokenizer.decode(generated[0, input_ids.size(1):].tolist())
        if args.mode == 'sft':
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()