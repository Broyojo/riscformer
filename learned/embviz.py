"""Visualize the learned geometry: pairwise |cos| similarity of (a) the 34
learned positional embeddings, (b) the 97 input-feature embeddings (columns
of the shared input projection).

    uv run --group learned --with matplotlib python -m learned.embviz
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import LearnedRiscformer, pick_device


def abscos(M, center=False):
    """M: (n, d) -> (n, n) matrix of |cos(x_i, x_j)|. center=True removes the
    shared mean direction first (learned embeddings carry a global component
    that inflates every pairwise similarity)."""
    if center:
        M = M - M.mean(axis=0)
    X = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return np.abs(X @ X.T)


def design_similarity(pos_l, pos_c):
    """RSA: pearson r between centered learned slot similarities and the
    hand-designed ones, plus the two group means (reg-reg vs role-distinct)."""
    iu = np.triu_indices(pos_l.shape[0], 1)
    sl = abscos(pos_l, center=True)[iu]
    sc = abscos(pos_c)[iu]
    r = np.corrcoef(sl, sc)[0, 1]
    reg = sl[np.isclose(sc, 0.5)]
    oth = sl[np.isclose(sc, 0.0)]
    return r, reg.mean(), oth.mean()


def constructed_vectors():
    """The hand-designed core's counterparts, built from its real dim registry:
    slot identity = TT_* flag (+ RIDX one-hot for registers); each input
    feature owns a dedicated residual dimension (payload bits map to VAL/IW/PC
    depending on slot type -- all disjoint, so features are exactly one-hot)."""
    from tcpu.core import build_core
    _, dm = build_core()
    D = dm.n
    pos = np.zeros((34, D))
    pos[0, dm.bit("TT_CTL")] = 1
    pos[1, dm.bit("TT_INSTR")] = 1
    for r in range(32):
        pos[2 + r, dm.bit("TT_REG")] = 1
        pos[2 + r, dm.bit("RIDX", r)] = 1
    feat = np.zeros((97, D))
    for i in range(32):
        feat[i, dm.bit("VAL", i)] = 1
        feat[i, dm.bit("IW", i)] = 1
        feat[i, dm.bit("PC", i)] = 1
    feat[32, dm.bit("PLF")] = 1
    for k in range(32):
        feat[33 + k, dm.bit("PLRD", k)] = 1
    for i in range(32):
        feat[65 + i, dm.bit("PLD", i)] = 1
    return pos, feat


def _panel(ax, M, title, kind, fig):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title, fontsize=10)
    if kind == "pos":
        ticks = [0, 1] + list(range(2, 34, 4))
        labels = ["CTL", "INSTR"] + [f"x{r - 2}" for r in range(2, 34, 4)]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
    else:
        b = [0, 32, 33, 65, 97]
        for edge in b[1:-1]:
            ax.axhline(edge - 0.5, color="w", lw=0.6, alpha=0.6)
            ax.axvline(edge - 0.5, color="w", lw=0.6, alpha=0.6)
        mid = [(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)]
        names = ["payload 0-31", "PLF", "PLRD", "PLD"]
        ax.set_xticks(mid); ax.set_xticklabels(names, rotation=30, fontsize=7)
        ax.set_yticks(mid); ax.set_yticklabels(names, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="learned/ckpt.pt")
    ap.add_argument("--out", default="learned/embeddings.png")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    model = LearnedRiscformer(a["d"], a["layers"], a["heads"])
    model.load_state_dict(ck["model"])
    pos_l = model.pos.detach().numpy()
    feat_l = model.inp.weight.detach().numpy().T

    pos_c, feat_c = constructed_vectors()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    fig.suptitle("embedding geometry  |cos(x,y)|  --  constructed (top) vs "
                 f"learned (bottom, ckpt step {ck['step']}, d={a['d']})")
    _panel(axes[0, 0], abscos(pos_c),
           "constructed: slot identity (TT_* + RIDX), d=3176", "pos", fig)
    _panel(axes[0, 1], abscos(feat_c),
           "constructed: input features (dedicated dims)", "feat", fig)
    _panel(axes[1, 0], abscos(pos_l),
           f"learned: positional embeddings, d={a['d']}", "pos", fig)
    _panel(axes[1, 1], abscos(feat_l),
           f"learned: input-feature embeddings, d={a['d']}", "feat", fig)
    r, mreg, moth = design_similarity(pos_l, pos_c)
    fig.text(0.5, 0.005,
             f"design similarity (centered RSA): r = {r:.3f}   "
             f"reg-reg pairs {mreg:.3f} (design 0.5)   "
             f"role-distinct {moth:.3f} (design 0.0)",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")
    print(f"design similarity (centered RSA): r = {r:.3f}  "
          f"reg-reg {mreg:.3f} vs role-distinct {moth:.3f}")


if __name__ == "__main__":
    main()
