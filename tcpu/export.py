"""Export constructed transformer weights to a binary blob for the C engine.

Format (little-endian):
  magic "TCPU1\0\0\0", int32 D, int32 n_blocks
  dim table: int32 count; per field: int32 namelen, bytes name, int32 base, int32 size
  per block:
    int32 n_heads
    per head: sp(Wq), sp(Wk), sp(Wv), sp(Wo)
    int32 H (0 -> no MLP)
    if H: sp(W1), float32 b1[H], sp(W2)
  sp(M): int32 rows, cols, nnz; int32 rowptr[rows+1]; int32 col[nnz]; float32 val[nnz]
"""
import struct

import numpy as np


def _write_sparse(f, M):
    rows, cols = M.shape
    rowptr = [0]
    colidx = []
    vals = []
    for r in range(rows):
        nz = np.nonzero(M[r])[0]
        colidx.extend(int(c) for c in nz)
        vals.extend(float(M[r, c]) for c in nz)
        rowptr.append(len(colidx))
    f.write(struct.pack("<iii", rows, cols, len(colidx)))
    f.write(np.asarray(rowptr, dtype=np.int32).tobytes())
    f.write(np.asarray(colidx, dtype=np.int32).tobytes())
    f.write(np.asarray(vals, dtype=np.float32).tobytes())


def export_weights(model, dims, path):
    with open(path, "wb") as f:
        f.write(b"TCPU1\x00\x00\x00")
        f.write(struct.pack("<ii", dims.n, len(model.blocks)))
        f.write(struct.pack("<i", len(dims.fields)))
        for name, (base, size) in dims.fields.items():
            nb = name.encode()
            f.write(struct.pack("<i", len(nb)))
            f.write(nb)
            f.write(struct.pack("<ii", base, size))
        for b in model.blocks:
            f.write(struct.pack("<i", len(b.heads)))
            for (Wq, Wk, Wv, Wo) in b.heads:
                for M in (Wq, Wk, Wv, Wo):
                    _write_sparse(f, M)
            if b.W1 is None:
                f.write(struct.pack("<i", 0))
            else:
                H = b.W1.shape[1]
                f.write(struct.pack("<i", H))
                _write_sparse(f, b.W1)
                f.write(np.asarray(b.b1, dtype=np.float32).tobytes())
                _write_sparse(f, b.W2)
