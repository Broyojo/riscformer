"""Plain transformer runtime. Nothing CPU-specific here: blocks of multi-head
softmax attention + ReLU MLP, both residual. Weights come from tcpu.build.

forward() is one pass over a token sequence (T, D). The CPU harness calls it
once per instruction, feeding the output state back in — autoregression over
machine steps rather than text tokens.
"""
import numpy as np


class Block:
    def __init__(self, heads, W1, b1, W2):
        self.heads = heads  # list of (Wq (D,dk), Wk (D,dk), Wv (D,dv), Wo (dv,D))
        self.W1, self.b1, self.W2 = W1, b1, W2

    def forward(self, x):
        if self.heads:
            acc = None
            for Wq, Wk, Wv, Wo in self.heads:
                q = x @ Wq
                k = x @ Wk
                v = x @ Wv
                scores = q @ k.T                      # (T, T); scale folded into Wq
                scores -= scores.max(axis=1, keepdims=True)
                w = np.exp(scores)
                w /= w.sum(axis=1, keepdims=True)
                out = (w @ v) @ Wo
                acc = out if acc is None else acc + out
            x = x + acc
        if self.W1 is not None and self.W1.shape[1] > 0:
            h = np.maximum(x @ self.W1 + self.b1, 0.0)
            x = x + h @ self.W2
        return x


class Model:
    def __init__(self, blocks, d_model, dtype=np.float64):
        self.blocks = blocks
        self.d_model = d_model
        self.dtype = dtype

    def forward(self, x):
        x = x.astype(self.dtype, copy=True)
        for b in self.blocks:
            x = b.forward(x)
        return x

    def n_params(self):
        n = 0
        for b in self.blocks:
            for h in b.heads:
                n += sum(w.size for w in h)
            for w in (b.W1, b.b1, b.W2):
                if w is not None:
                    n += w.size
        return n
