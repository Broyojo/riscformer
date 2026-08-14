"""ctypes wrapper for the C execution engine (tcpu/engine.c)."""
import ctypes
import os
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
LIB = pathlib.Path(os.environ.get("TCPU_LIB", HERE / "libtcpu.dylib"))
SRC = HERE / "engine.c"
WEIGHTS = HERE / "weights.bin"

R_STEPS, R_ECALL, R_EBREAK, R_ILLEGAL = 0, 1, 2, 3


def _build_lib():
    if LIB.exists() and LIB.stat().st_mtime > SRC.stat().st_mtime:
        return
    subprocess.run(["cc", "-O2", "-shared", "-o", str(LIB), str(SRC)], check=True)


def _build_weights():
    if WEIGHTS.exists():
        return
    from .core import build_core
    from .export import export_weights
    model, dims = build_core(np.float64)
    export_weights(model, dims, str(WEIGHTS))


class FastCPU:
    """Same machine as harness.TransformerCPU, executed by the C engine."""

    def __init__(self, ram_base=0x0, ram_size=1 << 24, pc=0):
        _build_lib()
        _build_weights()
        lib = ctypes.CDLL(str(LIB))
        lib.tf_load.argtypes = [ctypes.c_char_p]
        lib.tf_init_ram.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.tf_reset.argtypes = [ctypes.c_uint32]
        lib.tf_run.argtypes = [ctypes.c_int64]
        lib.tf_run.restype = ctypes.c_int
        lib.tf_pc.restype = ctypes.c_uint32
        lib.tf_steps.restype = ctypes.c_uint64
        lib.tf_getreg.argtypes = [ctypes.c_int]
        lib.tf_getreg.restype = ctypes.c_uint32
        lib.tf_setreg.argtypes = [ctypes.c_int, ctypes.c_uint32]
        lib.tf_ram.restype = ctypes.POINTER(ctypes.c_uint8)
        assert lib.tf_load(str(WEIGHTS).encode()) == 0
        assert lib.tf_init_ram(ram_base, ram_size) == 0
        lib.tf_reset(pc)
        self.lib = lib
        self.ram_base = ram_base
        self.ram_size = ram_size

    # -- memory --
    def write_blob(self, addr, data: bytes):
        off = addr - self.ram_base
        assert 0 <= off and off + len(data) <= self.ram_size
        ctypes.memmove(ctypes.addressof(self.lib.tf_ram().contents) + off,
                       data, len(data))

    def read_blob(self, addr, n) -> bytes:
        off = addr - self.ram_base
        buf = ctypes.create_string_buffer(n)
        ctypes.memmove(buf, ctypes.addressof(self.lib.tf_ram().contents) + off, n)
        return buf.raw

    # -- state --
    @property
    def pc(self):
        return self.lib.tf_pc()

    @property
    def steps(self):
        return self.lib.tf_steps()

    def reg(self, r):
        return self.lib.tf_getreg(r)

    def setreg(self, r, v):
        self.lib.tf_setreg(r, v & 0xFFFFFFFF)

    def run(self, max_steps):
        return self.lib.tf_run(max_steps)

    def prof(self):
        buf = (ctypes.c_double * 8)()
        self.lib.tf_prof(buf)
        return list(buf)

    def prof_reset(self):
        self.lib.tf_prof_reset()
