"""Train the learned riscformer on i.i.d. single-step examples.

    uv run --group learned python -m learned.train --steps 20000

Telemetry to watch: `carry-tail` — per-bit accuracy of the register targets
bucketed by bit position. The high bits only go exact if the model actually
learns carry propagation instead of interpolating it.
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import batch
from .model import LearnedRiscformer, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="learned/ckpt.pt")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    dev = pick_device()
    print(f"device: {dev}")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = LearnedRiscformer(args.d, args.layers, args.heads).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params: {nparams:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup))

    t0 = time.time()
    for step in range(1, args.steps + 1):
        feats, reg_tgt, instr_tgt, instr_mask = batch(rng, args.batch)
        feats = torch.from_numpy(feats).to(dev)
        reg_tgt = torch.from_numpy(reg_tgt).to(dev)
        instr_tgt = torch.from_numpy(instr_tgt).to(dev)
        instr_mask = torch.from_numpy(instr_mask).to(dev)

        reg_logits, instr_logits = model(feats)
        loss_reg = F.binary_cross_entropy_with_logits(reg_logits, reg_tgt)
        loss_instr = (F.binary_cross_entropy_with_logits(
            instr_logits, instr_tgt, reduction="none") * instr_mask).sum() \
            / instr_mask.sum().clamp(min=1)
        loss = loss_reg + loss_instr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                rb = (reg_logits > 0) == (reg_tgt > 0.5)
                ib = ((instr_logits > 0) == (instr_tgt > 0.5)) | (instr_mask == 0)
                bit_acc = rb.float().mean().item()
                exact = (rb.all(dim=(1, 2)) & ib.all(dim=1)).float().mean().item()
                # carry tail over the *written* register only: recover rd and
                # opcode from the instruction bits, keep reg-writing classes
                import numpy as _np
                iw = feats[:, 1, :32].cpu().numpy().astype(_np.int64)
                inst = (iw * (1 << _np.arange(32))).sum(axis=1)
                opc = inst & 0x7F
                rd = (inst >> 7) & 0x1F
                writes = _np.isin(opc, [0x33, 0x13, 0x37, 0x17, 0x6F, 0x67])
                sel = writes & (rd != 0)
                if sel.any():
                    idx = _np.nonzero(sel)[0]
                    rdb = rb[idx, rd[idx], :].float().mean(dim=0).cpu().numpy()
                    tail = " ".join(f"{rdb[i:i+4].mean():.3f}"
                                    for i in range(0, 32, 4))
                else:
                    tail = "n/a"
                rate = step * args.batch / (time.time() - t0)
                print(f"[{step:6d}] loss {loss.item():.4f}  bit {bit_acc:.5f}  "
                      f"exact-instr {exact:.4f}  ex/s {rate:,.0f}\n"
                      f"         rd-tail(b0..b31) {tail}", flush=True)
            torch.save({"model": model.state_dict(),
                        "args": vars(args), "step": step}, args.ckpt)
    print("done")


if __name__ == "__main__":
    main()
