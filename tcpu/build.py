"""Weight-construction framework: compiles logic gates and attention routing
into exact transformer weights.

Bit convention: every logical signal occupies one residual dimension holding
exactly 0.0 or 1.0 (small attention noise is squashed by the next gate).

Gate primitive: a linear-threshold unit "fire iff sum(w*x) >= theta", where the
design guarantees sum(w*x) is integer-valued (or half-integer theta is used
against near-integer sums), so preact = sum - (theta-1) is <=0 when false and
>=1 when true. The exact 0/1 step is built from two ReLUs:
    step(p) = relu(p) - relu(p - 1)
which is *exactly* 0 or 1 whenever p <= 0 or p >= 1 — no approximation.

Attention primitive: routing channels within a head. Channel j contributes
q_j * k_j to the score; queries are scaled by C. A key match of weight 1 gives
score C, weight 2 gives 2C, no match 0. Softmax puts all mass on the max row
score up to leakage ~T * e^-C (< 4e-12 for C=30), squashed by the next gate.
"""
import numpy as np

C = 30.0    # attention sharpness
BIG = 64.0  # gating weight to restrict a gate to one token type


class Gate:
    __slots__ = ("terms", "theta", "outs")

    def __init__(self, terms, theta, outs):
        self.terms = terms  # list[(dim, weight)]
        self.theta = theta  # fires iff sum >= theta
        self.outs = outs    # list[(dim, weight)] added to residual when fired


class HeadSpec:
    def __init__(self):
        self.channels = []  # list[(q_terms, k_terms)]
        self.vmap = []      # list[(src_dim, dst_dim, weight)]

    def channel(self, q_terms, k_terms):
        self.channels.append((q_terms, k_terms))

    def route(self, src_dim, dst_dim, w=1.0):
        self.vmap.append((src_dim, dst_dim, w))


class BlockSpec:
    def __init__(self):
        self.heads = []
        self.gates = []

    def head(self):
        h = HeadSpec()
        self.heads.append(h)
        return h

    def gate(self, terms, theta, outs):
        self.gates.append(Gate(terms, theta, outs))

    def when(self, token_dim, terms, theta, outs):
        """gate() restricted to tokens where token_dim == 1."""
        self.gates.append(Gate(terms + [(token_dim, BIG)], theta + BIG, outs))


class DimMap:
    def __init__(self):
        self.n = 0
        self.fields = {}

    def alloc(self, name, size=1):
        assert name not in self.fields, name
        self.fields[name] = (self.n, size)
        self.n += size
        return self.fields[name][0]

    def bit(self, name, i=0):
        base, size = self.fields[name]
        assert 0 <= i < size, (name, i, size)
        return base + i

    def size(self, name):
        return self.fields[name][1]


class CoreBuilder:
    def __init__(self):
        self.dims = DimMap()
        self.blocks = []

    def block(self):
        b = BlockSpec()
        self.blocks.append(b)
        return b

    def materialize(self, dtype=np.float64):
        from .nn import Block, Model
        D = self.dims.n
        blocks = []
        for spec in self.blocks:
            heads = []
            for h in spec.heads:
                nch = len(h.channels)
                Wq = np.zeros((D, nch), dtype)
                Wk = np.zeros((D, nch), dtype)
                for j, (qt, kt) in enumerate(h.channels):
                    for d, w in qt:
                        Wq[d, j] += w * C
                    for d, w in kt:
                        Wk[d, j] += w
                nv = len(h.vmap)
                Wv = np.zeros((D, nv), dtype)
                Wo = np.zeros((nv, D), dtype)
                for j, (src, dst, w) in enumerate(h.vmap):
                    Wv[src, j] = 1.0
                    Wo[j, dst] = w
                heads.append((Wq, Wk, Wv, Wo))
            ng = len(spec.gates)
            if ng:
                H = 2 * ng
                W1 = np.zeros((D, H), dtype)
                b1 = np.zeros(H, dtype)
                W2 = np.zeros((H, D), dtype)
                for j, g in enumerate(spec.gates):
                    for d, w in g.terms:
                        W1[d, 2 * j] += w
                        W1[d, 2 * j + 1] += w
                    b1[2 * j] = -(g.theta - 1.0)
                    b1[2 * j + 1] = -g.theta
                    for d, w in g.outs:
                        W2[2 * j, d] += w       # relu(p)
                        W2[2 * j + 1, d] += -w  # -relu(p-1) => exact step
            else:
                W1 = b1 = W2 = None
            blocks.append(Block(heads, W1, b1, W2))
        return Model(blocks, D, dtype)
