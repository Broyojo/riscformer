"""Reproducible C-engine benchmark on the real Doom boot instruction mix.

    uv run python bench.py [--steps 2000000]
"""
import argparse
import pathlib
import time

from tcpu.fast import FastCPU, R_ECALL
from verify_doom import MiniHost, DOOM, RAM_BASE, RAM_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    args = ap.parse_args()
    blob = (DOOM / "build" / "doom.bin").read_bytes()
    cpu = FastCPU(ram_base=RAM_BASE, ram_size=RAM_SIZE, pc=RAM_BASE)
    cpu.write_blob(RAM_BASE, blob)
    host = MiniHost("bench")

    # warm up (first 100k instructions, includes cold caches)
    while cpu.steps < 100_000:
        r = cpu.run(100_000 - cpu.steps)
        if r == R_ECALL:
            host.service(cpu.reg, cpu.setreg, cpu.read_blob, cpu.write_blob,
                         cpu.steps)
    t0 = time.perf_counter()
    start = cpu.steps
    while cpu.steps < start + args.steps:
        r = cpu.run(start + args.steps - cpu.steps)
        if r == R_ECALL:
            host.service(cpu.reg, cpu.setreg, cpu.read_blob, cpu.write_blob,
                         cpu.steps)
    dt = time.perf_counter() - t0
    n = cpu.steps - start
    print(f"{n:,} instructions in {dt:.2f}s -> {n/dt:,.0f} instr/s "
          f"({1e6*dt/n:.1f} us/instr)")


if __name__ == "__main__":
    main()
