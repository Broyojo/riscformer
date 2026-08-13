"""Demo: the transformer CPU runs a string-printing loop over the ecall bridge.

    .venv/bin/python run_demo.py
"""
import sys
import time

import numpy as np

from rv32 import enc
from rv32.ref import Bus
from harness import TransformerCPU

BASE = 0x1000
DATA = 0x4000
MSG = b"hello from a transformer running RISC-V\n"


def ecall(cpu):
    a0, a7 = cpu.effective_regs()[10], cpu.effective_regs()[17]
    if a7 == 1:
        sys.stdout.write(chr(a0 & 0xFF))
        sys.stdout.flush()
    elif a7 == 93:
        cpu.halted = True


def main():
    words = [
        enc.lui(5, DATA >> 12),      # x5 = &msg
        # loop: a0 = *x5; if a0 == 0 exit; putchar; x5++
        enc.lbu(10, 5, 0),
        enc.beq(10, 0, 20),
        enc.addi(17, 0, 1),
        enc.ecall(),
        enc.addi(5, 5, 1),
        enc.jal(0, -20),
        enc.addi(17, 0, 93),         # exit
        enc.ecall(),
    ]
    bus = Bus()
    bus.load_blob(BASE, b"".join(w.to_bytes(4, "little") for w in words))
    bus.load_blob(DATA, MSG + b"\x00")

    cpu = TransformerCPU(bus, pc=BASE)
    cpu.ecall_handler = ecall
    print(f"transformer core: {len(cpu.model.blocks)} blocks, "
          f"d_model={cpu.dims.n}, {cpu.model.n_params():,} params")
    t = time.time()
    while not cpu.halted:
        cpu.step()
    dt = time.time() - t
    print(f"[{cpu.steps} instructions, {cpu.steps / dt:.0f} instr/s]")


if __name__ == "__main__":
    main()
