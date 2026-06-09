"""Dump per-manifold reconstruction for the best STE hard-gate run -> npz.

For each of the 8 types (variant 2), on real L0=4 mixtures, compute the manifold's
ground-truth contribution and its reconstruction by (a) the single best-capturing atom
and (b) the full model (union of all active atoms) -- all projected into the manifold's
own V_i coordinates. This makes the captured_single vs captured_recon gap visual: where
the single atom recovers the shape (dedication) vs where only the union does (splitting),
and where the union is polluted by cross-talk from co-active atoms.

Does the decode forward -> run on jag. plot_recon.py renders locally from the npz.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import json
from pathlib import Path

import numpy as np
import torch

from manifold_ae.manifold_zoo import ManifoldZoo, TYPES
from reeval_capture import build

CKPT = "/nlp/scr/nathu/sae-manifold/runs/2026-06-09_jump_ste_sweep/lam02_pre1e3/ckpt.pt"
OUT = Path(__file__).parent
VARIANT, N, L0, CAP = 2, 4000, 4, 500
DEV = "cuda" if torch.cuda.is_available() else "cpu"

m, d = build(CKPT)
scale = d["scale"]
zoo = ManifoldZoo(d=256, seed=0)
rng = np.random.default_rng(123)
x, masks, truth, theta = zoo.sample(N, L0, rng, return_truth=True, return_theta=True)
xt = torch.tensor(x, device=DEV) / scale
with torch.no_grad():
    xh, z, g, active, dec, l0 = m.forward_jump(xt, theta=0.5)
    xhat_nb = (active.unsqueeze(-1) * dec).sum(1)            # (N,256) full recon - b_dec

by_name = {inst.name: inst for inst in zoo.instances}
arrays, meta = {}, {}
for name, di, ki, *_ in TYPES:
    inst = by_name[f"{name}_{VARIANT}"]
    idx = zoo.instances.index(inst)
    rows = np.where(masks[:, idx])[0]
    if len(rows) > CAP:
        rows = rng.choice(rows, CAP, replace=False)
    Vi = torch.tensor(inst.V, device=DEV)                    # (ki,256)
    m_i = torch.tensor(truth[rows, idx], device=DEV) / scale  # (n,256)
    den = (m_i ** 2).sum(-1, keepdim=True).clamp_min(1e-9)
    contrib = active[rows].unsqueeze(-1) * dec[rows]         # (n,N,256)
    fa = (((contrib - m_i.unsqueeze(1)) ** 2).sum(-1) / den).mean(0)   # (N,) per-atom FVU
    best = int(fa.argmin())

    tgt = (m_i @ Vi.t()).cpu().numpy()                       # (n,ki) target in V_i coords
    sgl = (contrib[:, best] @ Vi.t()).cpu().numpy()          # best single atom
    full = (xhat_nb[rows] @ Vi.t()).cpu().numpy()            # union / full model
    arrays[f"{name}__target"] = tgt
    arrays[f"{name}__single"] = sgl
    arrays[f"{name}__full"] = full
    arrays[f"{name}__theta"] = theta[rows, idx, :max(di, 1)]
    full_fvu = float((((full - tgt) ** 2).sum(-1)
                      / (tgt ** 2).sum(-1).clip(1e-9)).mean())
    meta[name] = dict(di=int(di), ki=int(ki), best_rank=int(m.ranks[best].item()),
                      single_fvu=float(fa[best]), full_fvu=full_fvu)

np.savez(OUT / "per_manifold.npz", **arrays)
json.dump(meta, open(OUT / "per_manifold_meta.json", "w"), indent=1)
print("saved", OUT / "per_manifold.npz")
print(json.dumps(meta, indent=1))
