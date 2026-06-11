"""
MiniMind 完整模型（Decoder-only Transformer）
支持：训练前向、带 KV Cache 的逐 token 生成
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from .config import MiniMindConfig
from .layers.rms_norm import RMSNorm
from .decoder_layer import DecoderLayer


class MiniMindForCausalLM(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            past_key_values: 长度为 num_layers 的列表，每个元素是该层上一轮的 (k_cache, v_cache)
            use_cache: 是否返回更新后的 cache

        Returns:
            logits: (batch, seq_len, vocab_size)
            present_key_values: 如果 use_cache=True，返回新的 cache 列表
        """
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache
            )
            if use_cache:
                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, present_key_values

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=50, eos_token_id=None):
        """
        自回归生成，使用 KV Cache 加速
        """
        self.eval()
        generated_ids = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                current_input = generated_ids[:, -1:]
            else:
                current_input = generated_ids

            logits, past_key_values = self(
                current_input,
                past_key_values=past_key_values,
                use_cache=True
            )

            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        return generated_ids