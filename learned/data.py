"""Training data for the learned riscformer: single-step (state, instruction)
-> next-state examples with the exact same I/O contract as the constructed
core (pending-load scheme included), generated from the reference emulator.

Every example is i.i.d. — the recurrence lives in the harness, so no
backprop-through-time is ever needed. States are sampled uniformly plus an
adversarial mixture (long carries, sign boundaries, powers of two), because
real program traces underrepresent exactly the values that break carry
generalization.
"""
import random

import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rv32.ref import Bus, RefCPU, MASK32
from rv32 import enc

# token slots: 0 = CTL, 1 = INSTR, 2..33 = x0..x31
N_TOKENS = 34
FEAT = 97          # CTL uses all 97: PC(32) PLF(1) PLRD(32) PLD(32)

SPECIALS = [0, 1, 2, 3, 0xFFFFFFFF, 0xFFFFFFFE, 0x80000000, 0x7FFFFFFF,
            0x0000FFFF, 0xFFFF0000, 0x00010000, 0x55555555, 0xAAAAAAAA,
            0x0000FF00, 0x80000001]


def sample_value(rng):
    p = rng.random()
    if p < 0.35:
        return rng.getrandbits(32)
    if p < 0.55:
        return rng.choice(SPECIALS)
    if p < 0.70:
        return rng.getrandbits(rng.randrange(1, 33))     # low-entropy
    if p < 0.85:
        v = 1 << rng.randrange(32)                        # powers of two +/- eps
        return (v + rng.choice([-1, 0, 1])) & MASK32
    return (rng.getrandbits(16) << rng.randrange(0, 17)) & MASK32


def sample_instr(rng):
    r = lambda: rng.randrange(32)
    imm12 = lambda: rng.randrange(-2048, 2048)
    p = rng.random()
    if p < 0.30:   # OP incl. M multiplies
        op = rng.choice([enc.add, enc.sub, enc.sll, enc.slt, enc.sltu, enc.xor,
                         enc.srl, enc.sra, enc.or_, enc.and_,
                         enc.mul, enc.mulh, enc.mulhsu, enc.mulhu])
        return op(r(), r(), r())
    if p < 0.50:   # OP-IMM
        q = rng.random()
        if q < 0.7:
            op = rng.choice([enc.addi, enc.slti, enc.sltiu, enc.xori,
                             enc.ori, enc.andi])
            return op(r(), r(), imm12())
        return rng.choice([enc.slli, enc.srli, enc.srai])(r(), r(), rng.randrange(32))
    if p < 0.62:   # LOAD
        return rng.choice([enc.lb, enc.lh, enc.lw, enc.lbu, enc.lhu])(r(), r(), imm12())
    if p < 0.72:   # STORE
        return rng.choice([enc.sb, enc.sh, enc.sw])(r(), r(), imm12())
    if p < 0.82:   # BRANCH (even byte offsets)
        return rng.choice([enc.beq, enc.bne, enc.blt, enc.bge, enc.bltu,
                           enc.bgeu])(r(), r(), rng.randrange(-2048, 2048) & ~1)
    if p < 0.87:
        return enc.lui(r(), rng.getrandbits(20))
    if p < 0.92:
        return enc.auipc(r(), rng.getrandbits(20))
    if p < 0.96:
        return enc.jal(r(), rng.randrange(-(1 << 19), 1 << 19) * 2)
    if p < 0.995:
        return enc.jalr(r(), r(), imm12())
    return enc.ecall()


class _RecBus:
    """Feeds the instruction at PC, returns random data for loads, records."""

    def __init__(self, rng, pc, inst):
        self.rng, self.pc, self.inst = rng, pc, inst
        self.raddr = self.rsize = self.rsigned = None
        self.waddr = self.wsize = self.wdata = None
        self.rdata = None

    def read(self, addr, size, signed):
        if addr == self.pc and size == 4 and not signed:
            return self.inst
        self.raddr, self.rsize, self.rsigned = addr, size, signed
        raw = self.rng.getrandbits(8 * size)
        self.rdata = raw
        from rv32.ref import sext
        return sext(raw, size * 8) & MASK32 if signed else raw

    def write(self, addr, size, value):
        self.waddr, self.wsize, self.wdata = addr, size, value & MASK32


CLS_ORDER = ["OP", "OPIMM", "LOAD", "STORE", "BR", "LUI", "AUIPC", "JAL",
             "JALR", "SYS", "FENCE"]
CLS_OPC = {0x33: 0, 0x13: 1, 0x03: 2, 0x23: 3, 0x63: 4, 0x37: 5, 0x17: 6,
           0x6F: 7, 0x67: 8, 0x73: 9, 0x0F: 10}

# INSTR-token target layout
T_NPC, T_ADDR, T_WDATA = 0, 32, 64
T_F3, T_CLS, T_PLF, T_PLRD = 96, 104, 115, 116
INSTR_TGT = 148


def make_example(rng):
    pc = (rng.getrandbits(24) << 2) & MASK32
    regs = [0] + [sample_value(rng) for _ in range(31)]
    plf = rng.random() < 0.3
    plrd = rng.randrange(1, 32) if plf else 0
    pld = sample_value(rng) if plf else 0
    inst = sample_instr(rng)

    eff = list(regs)
    if plf:
        eff[plrd] = pld

    bus = _RecBus(rng, pc, inst)
    cpu = RefCPU(bus, pc=pc)
    cpu.regs = list(eff)
    cpu.step()

    op = inst & 0x7F
    rd = (inst >> 7) & 0x1F
    is_load = op == 0x03
    tgt_regs = list(cpu.regs)
    plf_n, plrd_n = 0, 0
    if is_load and rd != 0:
        tgt_regs[rd] = eff[rd]          # load lands via the pending path
        plf_n, plrd_n = 1, rd

    # ---- features ----
    feats = np.zeros((N_TOKENS, FEAT), dtype=np.float32)
    for i in range(32):
        feats[0, i] = (pc >> i) & 1
    feats[0, 32] = float(plf)
    if plf:
        feats[0, 33 + plrd] = 1.0
        for i in range(32):
            feats[0, 65 + i] = (pld >> i) & 1
    for i in range(32):
        feats[1, i] = (inst >> i) & 1
    for r in range(32):
        for i in range(32):
            feats[2 + r, i] = (regs[r] >> i) & 1

    # ---- targets ----
    reg_tgt = np.zeros((32, 32), dtype=np.float32)
    for r in range(32):
        for i in range(32):
            reg_tgt[r, i] = (tgt_regs[r] >> i) & 1
    it = np.zeros(INSTR_TGT, dtype=np.float32)
    im = np.zeros(INSTR_TGT, dtype=np.float32)   # supervision mask
    for i in range(32):
        it[T_NPC + i] = (cpu.pc >> i) & 1
    im[T_NPC:T_NPC + 32] = 1
    if bus.raddr is not None or bus.waddr is not None:
        addr = bus.raddr if bus.raddr is not None else bus.waddr
        for i in range(32):
            it[T_ADDR + i] = (addr >> i) & 1
        im[T_ADDR:T_ADDR + 32] = 1
    if bus.waddr is not None:
        for i in range(32):
            it[T_WDATA + i] = (bus.wdata >> i) & 1
        im[T_WDATA:T_WDATA + 32] = 1
    it[T_F3 + ((inst >> 12) & 7)] = 1
    im[T_F3:T_F3 + 8] = 1
    it[T_CLS + CLS_OPC[op]] = 1
    im[T_CLS:T_CLS + 11] = 1
    it[T_PLF] = plf_n
    im[T_PLF] = 1
    if plf_n:
        it[T_PLRD + plrd_n] = 1
    if is_load:
        im[T_PLRD:T_PLRD + 32] = 1
    return feats, reg_tgt, it, im


def batch(rng, n):
    F, R, I, M = [], [], [], []
    for _ in range(n):
        f, r, i, m = make_example(rng)
        F.append(f); R.append(r); I.append(i); M.append(m)
    return (np.stack(F), np.stack(R), np.stack(I), np.stack(M))
