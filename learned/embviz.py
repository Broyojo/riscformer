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


def abscos(M):
    """M: (n, d) -> (n, n) matrix of |cos(x_i, x_j)|."""
    X = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return np.abs(X @ X.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="learned/ckpt.pt")
    ap.add_argument("--out", default="learned/embeddings.png")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    model = LearnedRiscformer(a["d"], a["layers"], a["heads"])
    model.load_state_dict(ck["model"])

    pos = model.pos.detach().numpy()                  # (34, d)
    feat = model.inp.weight.detach().numpy().T        # (97, d): feature f -> column

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.suptitle(f"learned embedding geometry  |cos(x,y)|   "
                 f"(ckpt step {ck['step']}, d={a['d']})")

    im0 = axes[0].imshow(abscos(pos), vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("positional embeddings (34 slots)")
    ticks = [0, 1] + list(range(2, 34, 4))
    labels = ["CTL", "INSTR"] + [f"x{r - 2}" for r in range(2, 34, 4)]
    axes[0].set_xticks(ticks); axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(abscos(feat), vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("input-feature embeddings (97 features)")
    b = [0, 32, 33, 65, 97]
    for edge in b[1:-1]:
        axes[1].axhline(edge - 0.5, color="w", lw=0.6, alpha=0.6)
        axes[1].axvline(edge - 0.5, color="w", lw=0.6, alpha=0.6)
    mid = [(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)]
    names = ["payload bits 0-31", "PLF", "PLRD one-hot", "PLD bits"]
    axes[1].set_xticks(mid); axes[1].set_xticklabels(names, rotation=30, fontsize=8)
    axes[1].set_yticks(mid); axes[1].set_yticklabels(names, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
