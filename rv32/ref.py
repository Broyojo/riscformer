"""Reference RV32I emulator — ground truth for verifying the transformer core.

Shares the bus model with the transformer harness: word-addressable RAM dict,
subword access (size/sign-extend) performed by the bus.
"""

MASK32 = 0xFFFFFFFF


def sext(v, bits):
    m = 1 << (bits - 1)
    return ((v & ((1 << bits) - 1)) ^ m) - m


class Bus:
    """Byte-addressable sparse RAM. Subword sizing/sign-extension happens here,
    like a byte-enabled memory controller."""

    def __init__(self):
        self.mem = bytearray()
        self.base = 0

    def load_blob(self, addr, data: bytes):
        end = addr + len(data)
        if not self.mem:
            self.base = addr & ~0xFFF
            self.mem = bytearray(end - self.base + 0x100000)
        need = end - self.base
        if need > len(self.mem):
            self.mem.extend(b"\x00" * (need - len(self.mem) + 0x100000))
        self.mem[addr - self.base:end - self.base] = data

    def read(self, addr, size, signed):
        off = addr - self.base
        if off < 0 or off + size > len(self.mem):
            return 0
        v = int.from_bytes(self.mem[off:off + size], "little")
        if signed:
            v = sext(v, size * 8) & MASK32
        return v

    def write(self, addr, size, value):
        off = addr - self.base
        if off < 0:
            return
        if off + size > len(self.mem):
            self.mem.extend(b"\x00" * (off + size - len(self.mem) + 0x100000))
        self.mem[off:off + size] = (value & ((1 << (size * 8)) - 1)).to_bytes(size, "little")


class RefCPU:
    def __init__(self, bus: Bus, pc=0):
        self.bus = bus
        self.pc = pc
        self.regs = [0] * 32
        self.halted = False
        self.ecall_handler = None  # callable(cpu) -> None

    def step(self):
        if self.halted:
            return
        inst = self.bus.read(self.pc, 4, False)
        op = inst & 0x7F
        rd = (inst >> 7) & 0x1F
        f3 = (inst >> 12) & 7
        rs1 = (inst >> 15) & 0x1F
        rs2 = (inst >> 20) & 0x1F
        f7 = inst >> 25
        a = self.regs[rs1]
        b = self.regs[rs2]
        imm_i = sext(inst >> 20, 12)
        npc = (self.pc + 4) & MASK32
        wr = None

        if op == 0x33 and (f7 & 1):  # M extension
            sa, sb = sext(a, 32), sext(b, 32)
            if f3 == 0:
                wr = (a * b) & MASK32
            elif f3 == 1:
                wr = ((sa * sb) >> 32) & MASK32
            elif f3 == 2:
                wr = ((sa * b) >> 32) & MASK32
            elif f3 == 3:
                wr = ((a * b) >> 32) & MASK32
            elif f3 == 4:
                wr = MASK32 if b == 0 else (
                    0x80000000 if (sa, sb) == (-0x80000000, -1)
                    else (abs(sa) // abs(sb) * (1 if (sa < 0) == (sb < 0) else -1)) & MASK32)
            elif f3 == 5:
                wr = MASK32 if b == 0 else (a // b) & MASK32
            elif f3 == 6:
                wr = a if b == 0 else (
                    0 if (sa, sb) == (-0x80000000, -1)
                    else (abs(sa) % abs(sb) * (1 if sa >= 0 else -1)) & MASK32)
            elif f3 == 7:
                wr = a if b == 0 else (a % b) & MASK32
        elif op == 0x33:  # OP
            if f3 == 0:
                wr = (a - b if f7 & 0x20 else a + b) & MASK32
            elif f3 == 1:
                wr = (a << (b & 31)) & MASK32
            elif f3 == 2:
                wr = 1 if sext(a, 32) < sext(b, 32) else 0
            elif f3 == 3:
                wr = 1 if a < b else 0
            elif f3 == 4:
                wr = a ^ b
            elif f3 == 5:
                wr = (sext(a, 32) >> (b & 31)) & MASK32 if f7 & 0x20 else a >> (b & 31)
            elif f3 == 6:
                wr = a | b
            elif f3 == 7:
                wr = a & b
        elif op == 0x13:  # OP-IMM
            sh = imm_i & 31
            if f3 == 0:
                wr = (a + imm_i) & MASK32
            elif f3 == 1:
                wr = (a << sh) & MASK32
            elif f3 == 2:
                wr = 1 if sext(a, 32) < imm_i else 0
            elif f3 == 3:
                wr = 1 if a < (imm_i & MASK32) else 0
            elif f3 == 4:
                wr = (a ^ imm_i) & MASK32
            elif f3 == 5:
                wr = (sext(a, 32) >> sh) & MASK32 if (inst >> 30) & 1 else a >> sh
            elif f3 == 6:
                wr = (a | imm_i) & MASK32
            elif f3 == 7:
                wr = (a & imm_i) & MASK32
        elif op == 0x03:  # LOAD
            addr = (a + imm_i) & MASK32
            size = 1 << (f3 & 3)
            wr = self.bus.read(addr, size, f3 < 4)
        elif op == 0x23:  # STORE
            imm_s = sext(((inst >> 25) << 5) | ((inst >> 7) & 0x1F), 12)
            addr = (a + imm_s) & MASK32
            self.bus.write(addr, 1 << (f3 & 3), b)
        elif op == 0x63:  # BRANCH
            imm_b = sext((((inst >> 31) & 1) << 12) | (((inst >> 7) & 1) << 11) |
                         (((inst >> 25) & 0x3F) << 5) | (((inst >> 8) & 0xF) << 1), 13)
            taken = {0: a == b, 1: a != b,
                     4: sext(a, 32) < sext(b, 32), 5: sext(a, 32) >= sext(b, 32),
                     6: a < b, 7: a >= b}.get(f3, False)
            if taken:
                npc = (self.pc + imm_b) & MASK32
        elif op == 0x37:  # LUI
            wr = inst & 0xFFFFF000
        elif op == 0x17:  # AUIPC
            wr = (self.pc + (inst & 0xFFFFF000)) & MASK32
        elif op == 0x6F:  # JAL
            imm_j = sext((((inst >> 31) & 1) << 20) | (((inst >> 12) & 0xFF) << 12) |
                         (((inst >> 20) & 1) << 11) | (((inst >> 21) & 0x3FF) << 1), 21)
            wr = npc
            npc = (self.pc + imm_j) & MASK32
        elif op == 0x67:  # JALR
            wr = npc
            npc = (a + imm_i) & MASK32 & ~1
        elif op == 0x73:  # SYSTEM
            if self.ecall_handler:
                self.ecall_handler(self)
            else:
                self.halted = True
        elif op == 0x0F:  # FENCE
            pass
        else:
            raise ValueError(f"illegal instruction {inst:#010x} at pc={self.pc:#x}")

        if wr is not None and rd != 0:
            self.regs[rd] = wr & MASK32
        self.pc = npc
