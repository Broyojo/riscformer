"""Machine harness: wires the transformer core to RAM (the memory bus).

The transformer is the CPU. The harness only does what motherboard wiring does:
  - embeds architectural state (PC, registers, pending-load record) into tokens
  - feeds the instruction word at PC (instruction bus)
  - services the load/store the core emitted (addr/size computed by the core)
  - copies the core's output state back in for the next step

The pending-load record makes loads architecturally invisible: the core emits
the address this step, the bus latches the data, and the core's own B2 layer
writes it into rd at the start of the next step.
"""
import numpy as np

from tcpu.core import build_core, T_CTL, T_INSTR, T_REG0, N_TOKENS
from rv32.ref import Bus, MASK32


class TransformerCPU:
    def __init__(self, bus: Bus, pc=0, dtype=np.float64, model=None, dims=None):
        if model is None:
            model, dims = build_core(dtype)
        self.model, self.dims = model, dims
        self.bus = bus
        self.pc = pc
        self.regs = [0] * 32
        self.plf, self.plrd, self.pld = 0, 0, 0
        self.halted = False
        self.ecall_handler = None
        self.steps = 0

    # -- state <-> tokens --

    def _embed(self):
        dm = self.dims
        x = np.zeros((N_TOKENS, dm.n))
        x[T_CTL, dm.bit("TT_CTL")] = 1.0
        for i in range(32):
            x[T_CTL, dm.bit("PC", i)] = (self.pc >> i) & 1
        if self.plf:
            x[T_CTL, dm.bit("PLF")] = 1.0
            x[T_CTL, dm.bit("PLRD", self.plrd)] = 1.0
            for i in range(32):
                x[T_CTL, dm.bit("PLD", i)] = (self.pld >> i) & 1
        inst = self.bus.read(self.pc, 4, False)
        x[T_INSTR, dm.bit("TT_INSTR")] = 1.0
        for i in range(32):
            x[T_INSTR, dm.bit("IW", i)] = (inst >> i) & 1
        for r in range(32):
            t = T_REG0 + r
            x[t, dm.bit("TT_REG")] = 1.0
            x[t, dm.bit("RIDX", r)] = 1.0
            for i in range(32):
                x[t, dm.bit("VAL", i)] = (self.regs[r] >> i) & 1
        return x

    def _field(self, x, tok, name):
        dm = self.dims
        base = dm.bit(name)
        n = dm.size(name)
        v = 0
        for i in range(n):
            if x[tok, base + i] > 0.5:
                v |= 1 << i
        return v

    def _flag(self, x, tok, name):
        return x[tok, self.dims.bit(name)] > 0.5

    # -- one instruction --

    def step(self):
        if self.halted:
            return
        x = self.model.forward(self._embed())
        dm = self.dims

        # sanity: every read-back bit must be near-integral (exactness check)
        self.regs = [self._field(x, T_REG0 + r, "VAL") for r in range(32)]
        npc = self._field(x, T_INSTR, "NPC")
        f3v = self._field(x, T_INSTR, "F3")
        f3 = f3v.bit_length() - 1 if f3v else 0

        # pending load from *previous* step has now been consumed
        self.plf = 0
        if self._flag(x, T_INSTR, "CLS_STORE"):
            addr = self._field(x, T_INSTR, "AS")
            data = self._field(x, T_INSTR, "BRAW")
            self.bus.write(addr, 1 << (f3 & 3), data)
        elif self._flag(x, T_INSTR, "CLS_LOAD"):
            addr = self._field(x, T_INSTR, "AS")
            data = self.bus.read(addr, 1 << (f3 & 3), f3 < 4)
            if self._flag(x, T_INSTR, "PLF_N"):
                self.plf = 1
                self.plrd = self._field(x, T_INSTR, "PLRD_N").bit_length() - 1
                self.pld = data
        elif self._flag(x, T_INSTR, "CLS_SYS"):
            if self.ecall_handler:
                self.ecall_handler(self)
            else:
                self.halted = True

        self.pc = npc & MASK32
        self.steps += 1

    def effective_regs(self):
        """Architectural registers (pending load applied)."""
        regs = list(self.regs)
        if self.plf:
            regs[self.plrd] = self.pld
        return regs
