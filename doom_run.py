"""Run Doom on the transformer CPU.

    .venv/bin/python doom_run.py [--max-steps N] [--frames-dir DIR]

The C engine executes the transformer (bit-exact, verified against the
reference emulator); this script is only the "motherboard": it services the
ecall host bridge (WAD file IO, console output, clock, keys) and saves each
frame the game draws as PNG.
"""
import argparse
import pathlib
import struct
import sys
import time

from tcpu.fast import FastCPU, R_STEPS, R_ECALL, R_EBREAK, R_ILLEGAL

ROOT = pathlib.Path(__file__).resolve().parent
DOOM = ROOT / "doom"
RAM_BASE = 0x80000000
RAM_SIZE = 64 << 20


def load_palette():
    data = (DOOM / "doom1.wad").read_bytes()
    magic, numlumps, dirofs = struct.unpack_from("<4sII", data, 0)
    assert magic == b"IWAD"
    for i in range(numlumps):
        ofs, size, name = struct.unpack_from("<II8s", data, dirofs + 16 * i)
        if name.rstrip(b"\x00") == b"PLAYPAL":
            return data[ofs:ofs + 768]
    raise RuntimeError("PLAYPAL not found")


class Host:
    def __init__(self, cpu, frames_dir):
        self.cpu = cpu
        self.files = {}
        self.next_fd = 3
        self.palette = load_palette()
        self.frames_dir = pathlib.Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.frame_no = 0
        self.keys = []          # queue of (pressed, doomkey)

    def cstr(self, addr):
        out = bytearray()
        while True:
            b = self.cpu.read_blob(addr, 64)
            i = b.find(0)
            if i >= 0:
                out += b[:i]
                return out.decode(errors="replace")
            out += b
            addr += 64

    def ecall(self):
        cpu = self.cpu
        a0, a1, a2, a7 = (cpu.reg(10), cpu.reg(11), cpu.reg(12), cpu.reg(17))
        ret = 0
        if a7 == 64:            # write(fd, buf, len)
            data = cpu.read_blob(a1, a2)
            if a0 in (1, 2):
                sys.stdout.write(data.decode(errors="replace"))
                sys.stdout.flush()
                ret = a2
            else:
                ret = -1
        elif a7 == 63:          # read(fd, buf, len)
            f = self.files.get(a0)
            if f is None:
                ret = -1
            else:
                data = f.read(a2)
                cpu.write_blob(a1, data)
                ret = len(data)
        elif a7 == 1024:        # open(path, flags)
            path = self.cstr(a0)
            name = pathlib.Path(path).name
            target = DOOM / name
            if a1 != 0 or not target.exists():
                ret = -1        # write access or missing file -> deny
            else:
                fd = self.next_fd
                self.next_fd += 1
                self.files[fd] = open(target, "rb")
                ret = fd
        elif a7 == 57:          # close
            f = self.files.pop(a0, None)
            if f:
                f.close()
            ret = 0
        elif a7 == 62:          # lseek(fd, off, whence)
            f = self.files.get(a0)
            if f is None:
                ret = -1
            else:
                off = a1 - (1 << 32) if a1 & 0x80000000 else a1
                f.seek(off, a2)
                ret = f.tell()
        elif a7 == 93:          # exit
            print(f"\n[guest exit({a0})]")
            return False
        elif a7 == 2001:        # ticks_ms: virtual 3 MIPS clock
            ret = cpu.steps // 3000
        elif a7 == 2002:        # getkey
            ret = self.keys.pop(0) if self.keys else 0xFFFFFFFF
        elif a7 == 2003:        # frame ready: a0 = 8bpp buffer, a1 = npixels
            self.save_frame(a0, a1)
        else:
            print(f"\n[unknown ecall a7={a7}]")
            ret = -1
        cpu.setreg(10, ret & 0xFFFFFFFF)
        return True

    def save_frame(self, addr, npix):
        from PIL import Image
        idx = self.cpu.read_blob(addr, npix)
        img = Image.frombytes("P", (320, 200), idx)
        img.putpalette(self.palette)
        img = img.convert("RGB").resize((640, 400), Image.NEAREST)
        path = self.frames_dir / f"frame_{self.frame_no:05d}.png"
        img.save(path)
        self.frame_no += 1
        print(f"\n[frame {self.frame_no} saved -> {path} "
              f"({self.cpu.steps:,} instructions)]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=10**12)
    ap.add_argument("--frames-dir", default=str(DOOM / "frames"))
    ap.add_argument("--status-every", type=int, default=2_000_000)
    args = ap.parse_args()

    cpu = FastCPU(ram_base=RAM_BASE, ram_size=RAM_SIZE, pc=RAM_BASE)
    cpu.write_blob(RAM_BASE, (DOOM / "build" / "doom.bin").read_bytes())
    host = Host(cpu, args.frames_dir)

    t0 = time.time()
    last_report = 0
    while cpu.steps < args.max_steps:
        r = cpu.run(200_000)
        if r == R_ECALL:
            if not host.ecall():
                break
        elif r == R_EBREAK:
            print(f"\n[ebreak at pc={cpu.pc:#x} after {cpu.steps:,} steps]")
            break
        elif r == R_ILLEGAL:
            print(f"\n[illegal instruction at pc={cpu.pc:#x} "
                  f"steps={cpu.steps:,} ra={cpu.reg(1):#x}]")
            break
        if cpu.steps - last_report >= args.status_every:
            last_report = cpu.steps
            rate = cpu.steps / (time.time() - t0)
            print(f"[{cpu.steps / 1e6:.1f}M instr, {rate / 1000:.1f}k/s, "
                  f"pc={cpu.pc:#x}, frames={host.frame_no}]", flush=True)
    rate = cpu.steps / (time.time() - t0)
    print(f"\n[done: {cpu.steps:,} instructions, {host.frame_no} frames, "
          f"{rate / 1000:.1f}k instr/s]")


if __name__ == "__main__":
    main()
