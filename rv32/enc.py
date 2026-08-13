"""RV32I instruction encoders (for building test programs without a toolchain)."""


def _r(op, rd, f3, rs1, rs2, f7):
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def _i(op, rd, f3, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def _s(op, f3, rs1, rs2, imm):
    imm &= 0xFFF
    return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 0x1F) << 7) | op


def _b(op, f3, rs1, rs2, imm):
    imm &= 0x1FFF
    return (((imm >> 12) & 1) << 31) | (((imm >> 5) & 0x3F) << 25) | (rs2 << 20) | \
        (rs1 << 15) | (f3 << 12) | (((imm >> 1) & 0xF) << 8) | (((imm >> 11) & 1) << 7) | op


def _u(op, rd, imm):
    return (imm & 0xFFFFF000) | (rd << 7) | op


def _j(op, rd, imm):
    imm &= 0x1FFFFF
    return (((imm >> 20) & 1) << 31) | (((imm >> 1) & 0x3FF) << 21) | (((imm >> 11) & 1) << 20) | \
        (((imm >> 12) & 0xFF) << 12) | (rd << 7) | op


# R-type
def add(rd, a, b): return _r(0x33, rd, 0, a, b, 0)
def sub(rd, a, b): return _r(0x33, rd, 0, a, b, 0x20)
def sll(rd, a, b): return _r(0x33, rd, 1, a, b, 0)
def slt(rd, a, b): return _r(0x33, rd, 2, a, b, 0)
def sltu(rd, a, b): return _r(0x33, rd, 3, a, b, 0)
def xor(rd, a, b): return _r(0x33, rd, 4, a, b, 0)
def srl(rd, a, b): return _r(0x33, rd, 5, a, b, 0)
def sra(rd, a, b): return _r(0x33, rd, 5, a, b, 0x20)
def or_(rd, a, b): return _r(0x33, rd, 6, a, b, 0)
def and_(rd, a, b): return _r(0x33, rd, 7, a, b, 0)


# I-type ALU
def addi(rd, a, imm): return _i(0x13, rd, 0, a, imm)
def slti(rd, a, imm): return _i(0x13, rd, 2, a, imm)
def sltiu(rd, a, imm): return _i(0x13, rd, 3, a, imm)
def xori(rd, a, imm): return _i(0x13, rd, 4, a, imm)
def ori(rd, a, imm): return _i(0x13, rd, 6, a, imm)
def andi(rd, a, imm): return _i(0x13, rd, 7, a, imm)
def slli(rd, a, sh): return _i(0x13, rd, 1, a, sh)
def srli(rd, a, sh): return _i(0x13, rd, 5, a, sh)
def srai(rd, a, sh): return _i(0x13, rd, 5, a, sh | 0x400)


# Loads / stores
def lb(rd, base, imm): return _i(0x03, rd, 0, base, imm)
def lh(rd, base, imm): return _i(0x03, rd, 1, base, imm)
def lw(rd, base, imm): return _i(0x03, rd, 2, base, imm)
def lbu(rd, base, imm): return _i(0x03, rd, 4, base, imm)
def lhu(rd, base, imm): return _i(0x03, rd, 5, base, imm)
def sb(base, src, imm): return _s(0x23, 0, base, src, imm)
def sh(base, src, imm): return _s(0x23, 1, base, src, imm)
def sw(base, src, imm): return _s(0x23, 2, base, src, imm)


# Branches (imm = byte offset from this instruction)
def beq(a, b, imm): return _b(0x63, 0, a, b, imm)
def bne(a, b, imm): return _b(0x63, 1, a, b, imm)
def blt(a, b, imm): return _b(0x63, 4, a, b, imm)
def bge(a, b, imm): return _b(0x63, 5, a, b, imm)
def bltu(a, b, imm): return _b(0x63, 6, a, b, imm)
def bgeu(a, b, imm): return _b(0x63, 7, a, b, imm)


def lui(rd, imm20): return _u(0x37, rd, imm20 << 12)
def auipc(rd, imm20): return _u(0x17, rd, imm20 << 12)
def jal(rd, imm): return _j(0x6F, rd, imm)
def jalr(rd, base, imm): return _i(0x67, rd, 0, base, imm)
def ecall(): return 0x00000073
def ebreak(): return 0x00100073
def nop(): return addi(0, 0, 0)
