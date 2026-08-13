"""Differentially verify the transformer C engine against the reference
emulator on the *actual Doom binary* — every instruction, mirrored hosts.

    .venv/bin/python verify_doom.py --steps 2000000
"""
import argparse
import pathlib
import time

from rv32.ref import Bus, RefCPU
from tcpu.fast import FastCPU, R_ECALL, R_EBREAK, R_ILLEGAL, R_STEPS

ROOT = pathlib.Path(__file__).resolve().parent
DOOM = ROOT / "doom"
RAM_BASE = 0x80000000
RAM_SIZE = 64 << 20


class MiniHost:
    """Deterministic ecall services, one instance per CPU (identical behavior)."""

    def __init__(self, tag):
        self.tag = tag
        self.files = {}
        self.next_fd = 3
        self.frames = 0
        self.out = []

    def service(self, regs_get, regs_set, mem_read, mem_write, steps):
        a0, a1, a2, a7 = regs_get(10), regs_get(11), regs_get(12), regs_get(17)
        ret = 0
        if a7 == 64:
            if a0 in (1, 2):
                self.out.append(mem_read(a1, a2))
                ret = a2
            else:
                ret = -1
        elif a7 == 63:
            f = self.files.get(a0)
            if f is None:
                ret = -1
            else:
                data = f.read(a2)
                mem_write(a1, data)
                ret = len(data)
        elif a7 == 1024:
            path = bytearray()
            addr = a0
            while True:
                b = mem_read(addr, 64)
                i = b.find(0)
                if i >= 0:
                    path += b[:i]
                    break
                path += b
                addr += 64
            name = pathlib.Path(path.decode()).name
            target = DOOM / name
            if a1 != 0 or not target.exists():
                ret = -1
            else:
                fd = self.next_fd
                self.next_fd += 1
                self.files[fd] = open(target, "rb")
                ret = fd
        elif a7 == 57:
            f = self.files.pop(a0, None)
            if f:
                f.close()
            ret = 0
        elif a7 == 62:
            f = self.files.get(a0)
            if f is None:
                ret = -1
            else:
                off = a1 - (1 << 32) if a1 & 0x80000000 else a1
                f.seek(off, a2)
                ret = f.tell()
        elif a7 == 93:
            return None
        elif a7 == 2001:
            ret = steps // 3000
        elif a7 == 2002:
            ret = 0xFFFFFFFF
        elif a7 == 2003:
            self.frames += 1
        regs_set(10, ret & 0xFFFFFFFF)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    args = ap.parse_args()
    blob = (DOOM / "build" / "doom.bin").read_bytes()

    bus = Bus()
    bus.load_blob(RAM_BASE, b"\x00")
    bus.mem = bytearray(RAM_SIZE)
    bus.base = RAM_BASE
    bus.load_blob(RAM_BASE, blob)
    ref = RefCPU(bus, pc=RAM_BASE)
    ref_steps = [0]
    ref_host = MiniHost("ref")

    def ref_ecall(cpu):
        r = ref_host.service(
            lambda r_: cpu.regs[r_],
            lambda r_, v: cpu.regs.__setitem__(r_, v),
            lambda a, n: bytes(bus.mem[a - bus.base:a - bus.base + n]),
            lambda a, d: bus.mem.__setitem__(slice(a - bus.base, a - bus.base + len(d)), d),
            ref_steps[0] + 1)  # ref services mid-step; tf after step completes
        if r is None:
            cpu.halted = True

    ref.ecall_handler = ref_ecall

    tf = FastCPU(ram_base=RAM_BASE, ram_size=RAM_SIZE, pc=RAM_BASE)
    tf.write_blob(RAM_BASE, blob)
    tf_host = MiniHost("tf")

    t0 = time.time()
    n = 0
    while n < args.steps:
        ref.step()
        ref_steps[0] += 1
        r = tf.run(1)
        if r == R_ECALL:
            ok = tf_host.service(tf.reg, tf.setreg, tf.read_blob, tf.write_blob,
                                 tf.steps)
            if ok is None:
                break
        elif r in (R_EBREAK, R_ILLEGAL):
            raise SystemExit(f"tf stopped: {r} at pc={tf.pc:#x} n={n}")
        n += 1
        if tf.pc != ref.pc:
            raise SystemExit(f"PC DIVERGED at step {n}: tf={tf.pc:#x} ref={ref.pc:#x}")
        for r_ in range(32):
            a, b = tf.reg(r_), ref.regs[r_]
            if a != b:
                raise SystemExit(
                    f"REG DIVERGED at step {n}: x{r_} tf={a:#x} ref={b:#x} pc={ref.pc:#x}")
        if n % 200_000 == 0:
            rate = n / (time.time() - t0)
            print(f"{n:,} instructions identical ({rate:,.0f}/s, "
                  f"pc={ref.pc:#x})", flush=True)
    print(f"\nVERIFIED: {n:,} Doom instructions bit-identical "
          f"(transformer C engine vs reference emulator)")
    print(f"guest console bytes: {sum(len(o) for o in tf_host.out)}, "
          f"frames signaled: {tf_host.frames}")


if __name__ == "__main__":
    main()
