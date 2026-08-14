"""Survival test: run the *trained* model as the CPU against the reference
emulator, instruction by instruction, and report how long it lasts.

    uv run --group learned python -m learned.lockstep --ckpt learned/ckpt.pt

The constructed core's score on this metric is "infinite" (bit-exact by
construction). A trained model earns a survival curve instead.
"""
import argparse
import random

import numpy as np
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rv32.ref import Bus, RefCPU, MASK32
from rv32 import enc
from .data import (N_TOKENS, FEAT, T_NPC, T_ADDR, T_WDATA, T_F3, T_PLF,
                   T_PLRD, CLS_ORDER, CLS_OPC)
from .model import LearnedRiscformer, pick_device


class TrainedCPU:
    """Same harness contract as the constructed core, model swapped."""

    def __init__(self, model, bus, pc, dev):
        self.model, self.bus, self.pc, self.dev = model, bus, pc, dev
        self.regs = [0] * 32
        self.plf, self.plrd, self.pld = 0, 0, 0

    @staticmethod
    def _bits(v):
        return [(v >> i) & 1 for i in range(32)]

    @staticmethod
    def _val(bits):
        v = 0
        for i, b in enumerate(bits):
            if b:
                v |= 1 << i
        return v

    def step(self):
        inst = self.bus.read(self.pc, 4, False)
        feats = np.zeros((1, N_TOKENS, FEAT), dtype=np.float32)
        feats[0, 0, :32] = self._bits(self.pc)
        feats[0, 0, 32] = self.plf
        if self.plf:
            feats[0, 0, 33 + self.plrd] = 1.0
            feats[0, 0, 65:97] = self._bits(self.pld)
        feats[0, 1, :32] = self._bits(inst)
        for r in range(32):
            feats[0, 2 + r, :32] = self._bits(self.regs[r])
        with torch.no_grad():
            reg_l, ins_l = self.model(torch.from_numpy(feats).to(self.dev))
        regs = (reg_l[0] > 0).cpu().numpy()
        ins = (ins_l[0] > 0).cpu().numpy()
        self.regs = [self._val(regs[r]) for r in range(32)]
        npc = self._val(ins[T_NPC:T_NPC + 32])
        f3 = int(np.argmax(ins_l[0, T_F3:T_F3 + 8].cpu().numpy()))
        op = inst & 0x7F
        self.plf = 0
        if op == 0x23:          # store (trust decode from the *instruction*,
            addr = self._val(ins[T_ADDR:T_ADDR + 32])   # data/addr from model)
            data = self._val(ins[T_WDATA:T_WDATA + 32])
            self.bus.write(addr, 1 << (f3 & 3), data)
        elif op == 0x03:        # load -> pending
            addr = self._val(ins[T_ADDR:T_ADDR + 32])
            data = self.bus.read(addr, 1 << (f3 & 3), f3 < 4)
            if ins[T_PLF]:
                self.plf = 1
                self.plrd = int(np.argmax(ins_l[0, T_PLRD:T_PLRD + 32].cpu().numpy()))
                self.pld = data
        self.pc = npc & MASK32

    def effective_regs(self):
        r = list(self.regs)
        if self.plf:
            r[self.plrd] = self.pld
        return r


def random_program(rng, n):
    """Branch-light random program that stays in bounds."""
    words = []
    for _ in range(n):
        from .data import sample_instr
        w = sample_instr(rng)
        if (w & 0x7F) in (0x63, 0x6F, 0x67, 0x73):   # keep control flow simple
            w = enc.addi(rng.randrange(32), rng.randrange(32),
                         rng.randrange(-2048, 2048))
        words.append(w)
    words.append(enc.jal(0, -4 * n))
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="learned/ckpt.pt")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    dev = pick_device()
    ck = torch.load(args.ckpt, map_location=dev)
    a = ck["args"]
    model = LearnedRiscformer(a["d"], a["layers"], a["heads"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"device: {dev}, ckpt step {ck['step']}")

    rng = random.Random(args.seed)
    survivals = []
    for t in range(args.trials):
        words = random_program(rng, 64)
        blob = b"".join(w.to_bytes(4, "little") for w in words)

        def mk():
            b = Bus()
            b.load_blob(0x1000, blob)
            b.load_blob(0x8000, bytes(rng.getrandbits(8) for _ in range(4096)))
            return b

        ref = RefCPU(mk(), pc=0x1000)
        tm = TrainedCPU(model, mk(), 0x1000, dev)
        for r in range(1, 32):
            v = rng.getrandbits(32)
            ref.regs[r] = v
            tm.regs[r] = v
        n = 0
        while n < args.max_steps:
            ref.step()
            tm.step()
            n += 1
            if tm.pc != ref.pc or tm.effective_regs() != ref.regs:
                break
        survivals.append(n)
        print(f"trial {t}: survived {n} instructions")
    s = sorted(survivals)
    print(f"\nmedian survival: {s[len(s)//2]}, mean: {sum(s)/len(s):.0f}, "
          f"max: {s[-1]} (constructed core: unbounded)")


if __name__ == "__main__":
    main()
