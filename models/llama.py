import torch as t
from torch import nn
from einops import rearrange, einsum
from typing import Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class Attention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, head_embd: Optional[int] = None):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_embd = n_embd // n_heads if head_embd is None else head_embd
        self.total_head_embed = self.head_embd * self.n_heads
        self.qkv_proj = nn.Linear(n_embd, 3 * self.total_head_embed)
        self.output_proj = nn.Linear(self.total_head_embed, n_embd)
        self.attn_pattern = nn.Identity()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """(batch, seq, n_embd) -> (batch, seq, n_embd)"""
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b s (t h e) -> t b h s e", t=3, h=self.n_heads)
        q_k = (q @ k.transpose(-2, -1)) / (self.head_embd ** 0.5)
        mask = t.full((q_k.shape[-1], q_k.shape[-1]), -1e4).triu(1)
        attn = nn.functional.softmax(q_k.tril() + mask, dim=-1)
        attn = self.attn_pattern(attn)
        z = einsum(attn, v, "b h r c, b h c e -> b h r e")
        z = rearrange(z, "b h s e -> b s (h e)")
        return self.output_proj(z)

class GPT2Block(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, ln_eps: float):
        super().__init__()
        self.ln1 = nn.LayerNorm((n_embd), eps=ln_eps)
        self.attn = Attention(n_embd, n_heads, None)
        self.ln2 = nn.LayerNorm((n_embd), eps=ln_eps)
        self.linear1 = nn.Linear(n_embd, 4*n_embd)
        self.linear2 = nn.Linear(4*n_embd, n_embd)
        self.mlp_act = nn.Identity()
        self.input_resid = nn.Identity()
        self.final_resid = nn.Identity()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """(batch, seq, n_embd) -> (batch, seq, n_embd)"""
        x = self.input_resid(x)
        attn_out = self.attn(self.ln1(x))
        mid = x + attn_out
        mlp_act = nn.functional.gelu(self.linear1(self.ln2(mid)), approximate="tanh")
        mlp_act = self.mlp_act(mlp_act)
        mlp_out = self.linear2(mlp_act)
        final_resid = self.final_resid(mid + mlp_out)
        return final_resid

class GPT2(nn.Module):
    def __init__(self, vocab_size, max_pos, n_embd, n_layers, n_heads, ln_eps):
        super().__init__()
        self.vocab_size, self.max_pos, self.n_embd, self.n_layers, self.n_heads, self.ln_eps = \
            vocab_size, max_pos, n_embd, n_layers, n_heads, ln_eps
        self.tok_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_pos, n_embd)
        self.blocks = nn.Sequential(
            *[GPT2Block(n_embd, n_heads, ln_eps) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm((n_embd), ln_eps)

    def forward(self, x:t.Tensor) -> t.Tensor:
        """(batch, seq, [embed]), int64 -> (batch, seq, vocab_size), float32"""
        pos_vector = t.stack([t.arange(x.shape[1]) for _ in range(x.shape[0])])
        x = self.tok_embed(x) + self.pos_embed(pos_vector)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return einsum(x, self.tok_embed.weight, "b s e, v e -> b s v")