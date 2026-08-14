"""Per-instruction analysis of a trained checkpoint: which mnemonics does the
model actually know, and where do the bits go wrong?

    uv run --group learned python -m learned.analyze --ckpt learned/ckpt.pt
"""
import argparse
import random

import numpy as np
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rv32 import enc
from rv32.ref import MASK32
from . import data as D
from .model import LearnedRiscformer, pick_device


def kinds(rng):
    r = lambda: rng.randrange(32)
    imm = lambda: rng.randrange(-2048, 2048)
    sh = lambda: rng.randrange(32)
    u20 = lambda: rng.getrandbits(20)
    return {
        "add":    lambda: enc.add(r(), r(), r()),
        "sub":    lambda: enc.sub(r(), r(), r()),
        "and":    lambda: enc.and_(r(), r(), r()),
        "or":     lambda: enc.or_(r(), r(), r()),
        "xor":    lambda: enc.xor(r(), r(), r()),
        "sll":    lambda: enc.sll(r(), r(), r()),
        "srl":    lambda: enc.srl(r(), r(), r()),
        "sra":    lambda: enc.sra(r(), r(), r()),
        "slt":    lambda: enc.slt(r(), r(), r()),
        "sltu":   lambda: enc.sltu(r(), r(), r()),
        "mul":    lambda: enc.mul(r(), r(), r()),
        "mulh":   lambda: enc.mulh(r(), r(), r()),
        "mulhu":  lambda: enc.mulhu(r(), r(), r()),
        "addi":   lambda: enc.addi(r(), r(), imm()),
        "andi":   lambda: enc.andi(r(), r(), imm()),
        "slli":   lambda: enc.slli(r(), r(), sh()),
        "srai":   lambda: enc.srai(r(), r(), sh()),
        "lui":    lambda: enc.lui(r(), u20()),
        "auipc":  lambda: enc.auipc(r(), u20()),
        "jal":    lambda: enc.jal(r(), rng.randrange(-(1 << 19), 1 << 19) * 2),
        "jalr":   lambda: enc.jalr(r(), r(), imm()),
        "beq":    lambda: enc.beq(r(), r(), rng.randrange(-2048, 2048) & ~1),
        "blt":    lambda: enc.blt(r(), r(), rng.randrange(-2048, 2048) & ~1),
        "lw":     lambda: enc.lw(r(), r(), imm()),
        "lb":     lambda: enc.lb(r(), r(), imm()),
        "sw":     lambda: enc.sw(r(), r(), imm()),
    }


def eval_batch(model, dev, rng, instr_fn, n=256, state_fn=None):
    F, R, I, M = [], [], [], []
    saved = D.sample_instr
    try:
        D.sample_instr = lambda _rng: instr_fn()
        if state_fn is not None:
            saved_v = D.sample_value
            D.sample_value = state_fn
        for _ in range(n):
            f, r_, i, m = D.make_example(rng)
            F.append(f); R.append(r_); I.append(i); M.append(m)
        if state_fn is not None:
            D.sample_value = saved_v
    finally:
        D.sample_instr = saved
    feats = torch.from_numpy(np.stack(F)).to(dev)
    reg_tgt = torch.from_numpy(np.stack(R)).to(dev)
    it = torch.from_numpy(np.stack(I)).to(dev)
    im = torch.from_numpy(np.stack(M)).to(dev)
    with torch.no_grad():
        rl, il = model(feats)
    rb = (rl > 0) == (reg_tgt > 0.5)
    ib = ((il > 0) == (it > 0.5)) | (im == 0)
    exact = (rb.all(dim=(1, 2)) & ib.all(dim=1)).float().mean().item()
    wrong_reg = (~rb).float().sum(dim=(1, 2)).mean().item()
    wrong_ins = ((~ib).float().sum(dim=1)).mean().item()
    return exact, wrong_reg, wrong_ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="learned/ckpt.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dev = pick_device()
    ck = torch.load(args.ckpt, map_location=dev)
    a = ck["args"]
    model = LearnedRiscformer(a["d"], a["layers"], a["heads"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    rng = random.Random(args.seed)
    print(f"ckpt step {ck['step']}  d={a['d']} layers={a['layers']}\n")
    print(f"{'kind':8s} {'exact':>7s} {'~wrong reg bits':>16s} {'~wrong out bits':>16s}")
    rows = []
    for name, fn in kinds(rng).items():
        e, wr, wi = eval_batch(model, dev, rng, fn, args.n)
        rows.append((e, name, wr, wi))
    for e, name, wr, wi in sorted(rows, reverse=True):
        print(f"{name:8s} {e:7.3f} {wr:16.2f} {wi:16.2f}")

    # carry-length probe: a + b with a forced carry run of length k
    print("\nADD by forced carry-run length (a=0..01..1 (k ones), b=1):")
    for k in (0, 4, 8, 16, 24, 31):
        def state_fn(_rng, k=k):
            return (1 << k) - 1
        def instr_fn():
            rd = rng.randrange(1, 32)
            return enc.add(rd, rng.randrange(1, 32), 0) if False else enc.addi(rd, rng.randrange(1, 32), 1)
        e, wr, _ = eval_batch(model, dev, rng, instr_fn, args.n, state_fn=state_fn)
        print(f"  carry-run {k:2d}: exact {e:.3f}  wrong reg bits {wr:.2f}")


if __name__ == "__main__":
    main()
