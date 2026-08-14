"""Differential verification: the transformer core must match the reference
emulator's architectural state (PC + all registers) after every instruction."""
import random
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rv32 import enc
from rv32.ref import Bus, RefCPU
from harness import TransformerCPU
from tcpu.core import build_core

BASE = 0x1000


@pytest.fixture(scope="module")
def core():
    return build_core(np.float64)


def run_diff(core, words, max_steps, data=None, data_addr=None):
    model, dims = core
    blob = b"".join(w.to_bytes(4, "little") for w in words)

    def mk_bus():
        bus = Bus()
        bus.load_blob(BASE, blob)
        if data is not None:
            bus.load_blob(data_addr, data)
        return bus

    ref = RefCPU(mk_bus(), pc=BASE)
    tf = TransformerCPU(mk_bus(), pc=BASE, model=model, dims=dims)
    for step in range(max_steps):
        if ref.halted or tf.halted:
            break
        ref.step()
        tf.step()
        assert tf.pc == ref.pc, f"step {step}: pc {tf.pc:#x} != ref {ref.pc:#x}"
        eff = tf.effective_regs()
        for r in range(32):
            assert eff[r] == ref.regs[r], (
                f"step {step}: x{r} {eff[r]:#x} != ref {ref.regs[r]:#x} "
                f"(pc was {ref.pc:#x})")
    assert ref.halted == tf.halted
    return ref, tf


def test_ref_sanity():
    words = [enc.addi(1, 0, 7), enc.addi(2, 0, 5), enc.add(3, 1, 2), enc.ecall()]
    bus = Bus()
    bus.load_blob(BASE, b"".join(w.to_bytes(4, "little") for w in words))
    cpu = RefCPU(bus, pc=BASE)
    for _ in range(10):
        if cpu.halted:
            break
        cpu.step()
    assert cpu.regs[3] == 12


def test_alu_torture(core):
    rng = random.Random(1234)
    words = []
    # seed registers x1..x10 with interesting 32-bit values
    seeds = [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x12345678,
             0xDEADBEEF, 0xFFFF8000, 2, 31]
    for r, v in enumerate(seeds, start=1):
        hi = (v + 0x800) >> 12 & 0xFFFFF
        words.append(enc.lui(r, hi))
        words.append(enc.addi(r, r, ((v & 0xFFF) ^ 0x800) - 0x800))
    rops = [enc.add, enc.sub, enc.sll, enc.slt, enc.sltu, enc.xor,
            enc.srl, enc.sra, enc.or_, enc.and_]
    iops = [enc.addi, enc.slti, enc.sltiu, enc.xori, enc.ori, enc.andi]
    for _ in range(60):
        if rng.random() < 0.6:
            words.append(rng.choice(rops)(rng.randrange(1, 16),
                                          rng.randrange(0, 16),
                                          rng.randrange(0, 16)))
        elif rng.random() < 0.7:
            words.append(rng.choice(iops)(rng.randrange(1, 16),
                                          rng.randrange(0, 16),
                                          rng.randrange(-2048, 2048)))
        else:
            sh = rng.randrange(0, 32)
            words.append(rng.choice([enc.slli, enc.srli, enc.srai])(
                rng.randrange(1, 16), rng.randrange(0, 16), sh))
    words.append(enc.ecall())
    run_diff(core, words, 200)


def test_x0_immutable(core):
    words = [enc.addi(0, 0, 55), enc.addi(1, 0, 3), enc.add(0, 1, 1),
             enc.lui(0, 0xFF), enc.ecall()]
    ref, tf = run_diff(core, words, 10)
    assert tf.effective_regs()[0] == 0


def test_branches_and_jumps(core):
    words = [
        enc.addi(1, 0, 10),        # x1 = 10 (counter)
        enc.addi(2, 0, 0),         # x2 = sum
        # loop: x2 += x1; x1 -= 1; bne x1, x0, loop
        enc.add(2, 2, 1),
        enc.addi(1, 1, -1),
        enc.bne(1, 0, -8),
        enc.blt(2, 0, 16),         # not taken (x2=55 > 0)
        enc.jal(5, 12),            # jump +12, link in x5
        enc.addi(6, 0, 111),       # skipped
        enc.addi(6, 0, 222),       # skipped
        enc.auipc(7, 0),           # landed here
        enc.jalr(8, 5, 0),         # jump back to link (addi 111)
        enc.addi(9, 0, 42),
        enc.ecall(),
    ]
    ref, tf = run_diff(core, words, 100)
    assert tf.effective_regs()[2] == 55


def test_signed_unsigned_compare(core):
    words = [
        enc.addi(1, 0, -1),          # x1 = 0xFFFFFFFF
        enc.addi(2, 0, 1),
        enc.slt(3, 1, 2),            # -1 < 1 signed -> 1
        enc.sltu(4, 1, 2),           # huge < 1 unsigned -> 0
        enc.bltu(2, 1, 8),           # taken: 1 < 0xFFFFFFFF
        enc.addi(5, 0, 99),          # skipped
        enc.bge(2, 1, 8),            # taken: 1 >= -1 signed
        enc.addi(6, 0, 99),          # skipped
        enc.ecall(),
    ]
    ref, tf = run_diff(core, words, 20)
    eff = tf.effective_regs()
    assert eff[3] == 1 and eff[4] == 0 and eff[5] == 0 and eff[6] == 0


def test_memory(core):
    daddr = 0x4000
    words = [
        enc.lui(1, daddr >> 12),      # x1 = data base
        enc.addi(2, 0, -1),           # 0xFFFFFFFF
        enc.sw(1, 2, 0),
        enc.lw(3, 1, 0),              # x3 = 0xFFFFFFFF
        enc.add(4, 3, 3),             # use loaded value immediately (hazard)
        enc.addi(5, 0, 0x7A),
        enc.sb(1, 5, 5),              # byte store
        enc.lb(6, 1, 5),              # 0x7A sign-pos
        enc.lb(7, 1, 0),              # 0xFF -> -1
        enc.lbu(8, 1, 0),             # 0xFF
        enc.lh(9, 1, 0),              # 0xFFFF -> -1
        enc.lhu(10, 1, 0),            # 0xFFFF
        enc.sh(1, 5, 8),
        enc.lhu(11, 1, 8),
        enc.lw(12, 1, 4),             # word containing the stored byte
        enc.ecall(),
    ]
    ref, tf = run_diff(core, words, 30, data=bytes(64), data_addr=daddr)
    eff = tf.effective_regs()
    assert eff[3] == 0xFFFFFFFF
    assert eff[7] == 0xFFFFFFFF and eff[8] == 0xFF
    assert eff[9] == 0xFFFFFFFF and eff[10] == 0xFFFF


def test_load_use_and_back_to_back(core):
    daddr = 0x4000
    data = (12345).to_bytes(4, "little") + (99999).to_bytes(4, "little")
    words = [
        enc.lui(1, daddr >> 12),
        enc.lw(2, 1, 0),
        enc.lw(3, 1, 4),          # back-to-back loads
        enc.add(4, 2, 3),         # both used right away
        enc.lw(2, 1, 4),          # overwrite x2
        enc.sw(1, 2, 8),          # store loaded value immediately
        enc.lw(5, 1, 8),
        enc.beq(5, 2, 8),         # taken
        enc.addi(6, 0, 1),        # skipped
        enc.ecall(),
    ]
    ref, tf = run_diff(core, words, 20, data=data, data_addr=daddr)
    eff = tf.effective_regs()
    assert eff[4] == 12345 + 99999
    assert eff[5] == 99999 and eff[6] == 0


def test_fib_function_calls(core):
    # iterative fib(12) with a call/return, stack in memory
    sp = 0x8000
    words = [
        enc.lui(2, sp >> 12),         # x2 = sp
        enc.addi(10, 0, 12),          # a0 = 12
        enc.jal(1, 12),               # call fib
        enc.add(20, 10, 0),           # x20 = result
        enc.ecall(),
        # fib: a=0 (x5), b=1 (x6); loop n times
        enc.addi(5, 0, 0),
        enc.addi(6, 0, 1),
        enc.beq(10, 0, 24),           # while n != 0
        enc.add(7, 5, 6),
        enc.add(5, 6, 0),
        enc.add(6, 7, 0),
        enc.addi(10, 10, -1),
        enc.jal(0, -20),
        enc.add(10, 5, 0),            # return a
        enc.jalr(0, 1, 0),
    ]
    ref, tf = run_diff(core, words, 200)
    assert tf.effective_regs()[20] == 144


def test_c_engine_matches_reference():
    """The C engine must match the reference emulator step-for-step."""
    from tcpu.fast import FastCPU
    words = []
    words.append(enc.lui(1, 0x12345))
    words.append(enc.addi(1, 1, 0x678 - 4096))
    words.append(enc.addi(2, 0, -1))
    words.append(enc.sra(3, 1, 2))
    body = [enc.add(4, 4, 1), enc.slli(1, 1, 1), enc.xor(5, 4, 1),
            enc.sltu(6, 5, 4), enc.bne(6, 0, 8), enc.sub(4, 4, 5)]
    words += body
    words.append(enc.jal(0, -4 * len(body)))
    blob = b"".join(w.to_bytes(4, "little") for w in words)

    bus = Bus()
    bus.load_blob(BASE, blob)
    ref = RefCPU(bus, pc=BASE)
    cpu = FastCPU(ram_base=0, ram_size=1 << 20, pc=BASE)
    cpu.write_blob(BASE, blob)
    for n in range(300):
        ref.step()
        cpu.run(1)
        assert cpu.pc == ref.pc, n
        for r in range(32):
            assert cpu.reg(r) == ref.regs[r], (n, r)


def test_m_extension_multiply(core):
    """All four M-extension multiplies, adversarial values, vs reference."""
    seeds = [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0xFFFF8000,
             0x00010001, 0xDEADBEEF]
    words = []
    for r, v in enumerate(seeds, start=1):
        hi = ((v + 0x800) >> 12) & 0xFFFFF
        words.append(enc.lui(r, hi))
        words.append(enc.addi(r, r, ((v & 0xFFF) ^ 0x800) - 0x800))
    for op in (enc.mul, enc.mulh, enc.mulhsu, enc.mulhu):
        for a in range(1, 9):
            for b in range(1, 9):
                words.append(op(9 + (a + b) % 3, a, b))
    words.append(enc.ecall())
    run_diff(core, words, 300)


def test_c_engine_multiply():
    from tcpu.fast import FastCPU
    words = [enc.lui(1, 0x89ABD), enc.addi(1, 1, 0xDEF - 4096),
             enc.addi(2, 0, -3)]
    body = [enc.mul(3, 1, 2), enc.mulh(4, 1, 2), enc.mulhu(5, 1, 2),
            enc.mulhsu(6, 1, 2), enc.add(1, 1, 3), enc.xor(2, 2, 4)]
    words += body
    words.append(enc.jal(0, -4 * len(body)))
    blob = b"".join(w.to_bytes(4, "little") for w in words)
    bus = Bus()
    bus.load_blob(BASE, blob)
    ref = RefCPU(bus, pc=BASE)
    cpu = FastCPU(ram_base=0, ram_size=1 << 20, pc=BASE)
    cpu.write_blob(BASE, blob)
    for n in range(400):
        ref.step()
        cpu.run(1)
        assert cpu.pc == ref.pc, n
        for r in range(32):
            assert cpu.reg(r) == ref.regs[r], (n, r)
