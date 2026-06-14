"""
SFT Benchmark 评测（LLM-as-Judge）
每次评测：对每道题生成 3 次回答，通过 DeepSeek API 评分后计算：
- fluency_avg3 / fluency_pass3
- factuality_avg3 / factuality_pass3
- instruction_following_avg3 / instruction_following_pass3
- mean_avg3 / mean_pass3
"""

import json
import time
import torch
from openai import OpenAI
from tokenizers import Tokenizer


JUDGE_PROMPT = """请根据问题对以下回答进行评分（0-1 二值）：

【问题】{question}
【回答】{response}

请从三个维度评分（0=不通过，1=通过）：
1. fluency: 回答是否流畅、语言自然
2. factuality: 回答是否准确、符合事实
3. instruction_following: 是否正确理解并遵循了指令并回答用户问题

请务必严格，如果无法判断，则视为不通过。

以 JSON 格式输出：
```json
{{
  "fluency": 0 或 1,
  "factuality": 0 或 1,
  "instruction_following": 0 或 1
}}
```"""


def generate_response(model, tokenizer, question, device, max_new_tokens=256):
    """
    让模型根据问题生成回答（ChatML 格式）
    每次生成带有随机性，因此多次调用会得到不同回答
    """
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer.encode(prompt)
    input_ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(device)

    generated = input_ids.clone()
    eos_id = tokenizer.token_to_id("<|im_end|>")

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(generated)
            next_logits = logits[:, -1, :] / 0.7           # temperature = 0.7
            v, _ = torch.topk(next_logits, 50)
            next_logits[next_logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)
        if eos_id is not None and next_token.item() == eos_id:
            break

    output_ids = generated[0, input_ids.size(1):].tolist()
    response = tokenizer.decode(output_ids)
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    return response


def evaluate_with_judge(question, response, api_key, base_url="https://api.deepseek.com"):
    """调用 DeepSeek 对单条回答进行评分，返回 (fluency, factuality, instruction_following)"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = JUDGE_PROMPT.format(question=question, response=response)

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(completion.choices[0].message.content)
        return (
            result.get("fluency", 0),
            result.get("factuality", 0),
            result.get("instruction_following", 0)
        )
    except Exception as e:
        print(f"Judge API error: {e}")
        return 0, 0, 0


def run_sft_benchmark(model, tokenizer, data_path, api_key, device):
    """
    一次 Benchmark 评测（共 100 题，每题生成 3 次回答）
    返回直接可用于 SwanLab 的 8 个指标：
        fluency_avg3, fluency_pass3,
        factuality_avg3, factuality_pass3,
        instruction_following_avg3, instruction_following_pass3,
        mean_avg3, mean_pass3
    """
    # 读取所有问题
    with open(data_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line)['prompt'] for line in f if line.strip()]

    total = len(questions)

    # 存储每个问题三个维度的 avg3 和 pass3（先累积每个问题的值）
    fl_avg_list, fl_pass_list = [], []
    fa_avg_list, fa_pass_list = [], []
    in_avg_list, in_pass_list = [], []

    print(f"Running SFT benchmark on {total} questions, 3 generations each...")

    for i, q in enumerate(questions):
        print(f"[{i+1}/{total}] {q[:30]}...")
        # 对同一个问题生成 3 次回答并评分
        fl_scores, fa_scores, in_scores = [], [], []
        for _ in range(3):
            response = generate_response(model, tokenizer, q, device)
            f, fa, ins = evaluate_with_judge(q, response, api_key)
            fl_scores.append(f)
            fa_scores.append(fa)
            in_scores.append(ins)
            time.sleep(0.3)  # 轻微延迟，避免 API 限速

        # 计算该题目的 avg3 (三个分数的平均值)
        fl_avg = sum(fl_scores) / 3
        fa_avg = sum(fa_scores) / 3
        in_avg = sum(in_scores) / 3

        # 计算该题目的 pass3 (三次中有任意一次 ≥ 0.5 则通过，记为1，否则0)
        fl_pass = 1 if any(s >= 0.5 for s in fl_scores) else 0
        fa_pass = 1 if any(s >= 0.5 for s in fa_scores) else 0
        in_pass = 1 if any(s >= 0.5 for s in in_scores) else 0

        fl_avg_list.append(fl_avg)
        fl_pass_list.append(fl_pass)
        fa_avg_list.append(fa_avg)
        fa_pass_list.append(fa_pass)
        in_avg_list.append(in_avg)
        in_pass_list.append(in_pass)

    # 对所有问题取平均
    fluency_avg3 = sum(fl_avg_list) / total
    fluency_pass3 = sum(fl_pass_list) / total
    factuality_avg3 = sum(fa_avg_list) / total
    factuality_pass3 = sum(fa_pass_list) / total
    instruction_following_avg3 = sum(in_avg_list) / total
    instruction_following_pass3 = sum(in_pass_list) / total

    mean_avg3 = (fluency_avg3 + factuality_avg3 + instruction_following_avg3) / 3
    mean_pass3 = (fluency_pass3 + factuality_pass3 + instruction_following_pass3) / 3

    results = {
        "fluency_avg3": fluency_avg3,
        "fluency_pass3": fluency_pass3,
        "factuality_avg3": factuality_avg3,
        "factuality_pass3": factuality_pass3,
        "instruction_following_avg3": instruction_following_avg3,
        "instruction_following_pass3": instruction_following_pass3,
        "mean_avg3": mean_avg3,
        "mean_pass3": mean_pass3
    }

    print(f"Benchmark results: {results}")
    return results