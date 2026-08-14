"""A standard modern transformer trained to be the riscformer: pre-norm,
SwiGLU MLPs, learned absolute positional embeddings over the 34 fixed slots,
per-token sigmoid bit heads. Same I/O contract as the constructed core."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_TOKENS, FEAT, INSTR_TGT


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Block(nn.Module):
    def __init__(self, d, heads, ff_mult=4):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.heads = heads
        self.n2 = nn.RMSNorm(d)
        h = int(d * ff_mult * 2 / 3)
        self.w1 = nn.Linear(d, h, bias=False)   # gate
        self.w3 = nn.Linear(d, h, bias=False)   # value
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, -1).transpose(1, 2)
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)      # full attention, no mask
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        y = self.n2(x)
        x = x + self.w2(F.silu(self.w1(y)) * self.w3(y))
        return x


class LearnedRiscformer(nn.Module):
    def __init__(self, d=384, layers=16, heads=6):
        super().__init__()
        self.inp = nn.Linear(FEAT, d)
        self.pos = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.reg_head = nn.Linear(d, 32)          # shared across register slots
        self.instr_head = nn.Linear(d, INSTR_TGT)

    def forward(self, feats):
        x = self.inp(feats) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        reg_logits = self.reg_head(x[:, 2:, :])   # (B, 32, 32)
        instr_logits = self.instr_head(x[:, 1, :])
        return reg_logits, instr_logits
