"""Dump best-single-atom FVU and full-model FVU for ALL 48 instances (8 types x 6
variants) of the best STE hard-gate run -> fvu_48.json. Run on jag; plot_fvu_48.py
renders the 8x6 strip plot locally.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import json
from pathlib import Path

import numpy as np
import torch

from manifold_ae.manifold_zoo import ManifoldZoo
from reeval_capture import build

CKPT = "/nlp/scr/nathu/sae-manifold/runs/2026-06-09_jump_ste_sweep/lam02_pre1e3/ckpt.pt"
OUT = Path(__file__).parent
N, L0 = 6000, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"

m, d = build(CKPT)
scale = d["scale"]
zoo = ManifoldZoo(d=256, seed=0)
x, masks, truth = zoo.sample(N, L0, np.random.default_rng(123), return_truth=True)
xt = torch.tensor(x, device=DEV) / scale
with torch.no_grad():
    xh, z, g, active, dec, l0 = m.forward_jump(xt, theta=0.5)
    xhat_nb = (active.unsqueeze(-1) * dec).sum(1)

out = []
for idx, inst in enumerate(zoo.instances):
    rows = np.where(masks[:, idx])[0]
    if len(rows) == 0:
        continue
    Vi = torch.tensor(inst.V, device=DEV)
    m_i = torch.tensor(truth[rows, idx], device=DEV) / scale
    den = (m_i ** 2).sum(-1, keepdim=True).clamp_min(1e-9)
    contrib = active[rows].unsqueeze(-1) * dec[rows]
    single = float((((contrib - m_i.unsqueeze(1)) ** 2).sum(-1) / den).mean(0).min())
    tgt, rec = m_i @ Vi.t(), xhat_nb[rows] @ Vi.t()
    full = float((((tgt - rec) ** 2).sum(-1) / (tgt ** 2).sum(-1).clamp_min(1e-9)).mean())
    out.append(dict(type=inst.type, variant=inst.name.split("_")[-1],
                    ki=int(inst.ki), di=int(inst.di), single=single, full=full))

json.dump(out, open(OUT / "fvu_48.json", "w"), indent=1)
print(f"saved {len(out)} instances")
for t in dict.fromkeys(r["type"] for r in out):
    ss = [r["single"] for r in out if r["type"] == t]
    fs = [r["full"] for r in out if r["type"] == t]
    print(f"  {t:11s} single<0.05: {sum(s<0.05 for s in ss)}/6   full<0.05: {sum(f<0.05 for f in fs)}/6")
