"""Rank-distribution stats of the most promising 48-zoo run (K8/lr3e-4/lam0.003/1x).

Loads the checkpoint, computes:
  (1) learned per-atom rank histogram (all 64 atoms vs the active subset),
  (2) per manifold TYPE: oracle min-capturing rank vs the learned rank of the dedicating atom
      (for instances captured single-atom, FVU<0.05), alongside di (intrinsic) and ki (embedding).
Prints a table and saves rank_dist.png. Read-only analysis off the saved ckpt.
"""
import json
import sys, os
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from manifold_ae.manifold_sae import ManifoldSAE
from manifold_ae.manifold_zoo import ManifoldZoo
from manifold_ae.eval_and_viz import compute_capture

RUN = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-09_paper48_stage2_lr3e-4/k8_lam0.003_rev1x")
ORACLE = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-09_oracle_48/metrics.json")
OUT = Path(__file__).parent
DEV = "cpu"

ck = torch.load(RUN / "ckpt.pt", map_location=DEV)
m = ManifoldSAE(d_model=256, rank_dist=ck["pool"], enc_dims=tuple(ck["enc_dims"]),
                jump_eps=ck["jump_eps"], learn_rank=ck["learn_rank"]).to(DEV)
m.load_state_dict(ck["state_dict"]); m.eval()
zoo = ManifoldZoo(d=256, seed=0, variants_per_type=6)

# (1) per-atom learned rank + active mask (fire on an L0=4 mixture)
ar = m.atom_ranks().round().long().clamp(0, m.max_rank)
rng = np.random.default_rng(0)
x, _ = zoo.sample(4000, 4, rng)
with torch.no_grad():
    _, _, _, active, _, _ = m.forward_jump(torch.tensor(x, device=DEV) / ck["scale"])
ever = (active.sum(0) > 0).cpu()
hist_all = torch.bincount(ar, minlength=m.max_rank + 1).tolist()
hist_act = torch.bincount(ar[ever], minlength=m.max_rank + 1).tolist()
print(f"=== {RUN.name} ===")
print(f"per-atom rank hist (0..{m.max_rank})  all-{m.n_atoms}atoms : {hist_all}")
print(f"                                      active-{int(ever.sum())}    : {hist_act}  (dead {int((~ever).sum())})")
print(f"mean rank: all={ar.float().mean():.2f}  active={ar[ever].float().mean():.2f}")

# (2) per-type: oracle min-cap vs learned dedicating-atom rank for captured instances
res, _ = compute_capture(m, zoo, ck["scale"], n=4000, n_iso=1500, l0=4, want_tri=False)
oracle = json.load(open(ORACLE)) if ORACLE.exists() else {}
def omincap(name):
    d = oracle.get(name)
    return d.get("min_capturing_rank") if isinstance(d, dict) else None
by = defaultdict(lambda: dict(ki=0, di=0, n=0, cap=0, lr=[], ocap=[]))
for r in res["per_instance"]:
    t = by[r["type"]]; t["ki"], t["di"], t["n"] = r["ki"], r["di"], t["n"] + 1
    oc = omincap(r["name"])
    if oc is not None: t["ocap"].append(oc)
    if r["single"] < 0.05:
        t["cap"] += 1; t["lr"].append(r["best_rank"])
print(f"\nper-type (captured single-atom, FVU<0.05):  total single={res['captured_single']}/48")
print("%-11s %2s %2s  %-8s  %-8s  %-14s"%("type","di","ki","#cap/n","oracle_r","learned_r(mean)"))
for t in ["circle","sphere","torus","mobius","swiss_roll","helix","flat_disk","segment"]:
    d = by[t]
    if not d["n"]: continue
    omc = f"{np.mean(d['ocap']):.1f}" if d["ocap"] else "?"
    lr = f"{np.mean(d['lr']):.1f}" if d["lr"] else "--"
    print("%-11s %2d %2d  %-8s  %-8s  %-14s"%(t, d["di"], d["ki"], f"{d['cap']}/{d['n']}", omc, lr))

# ---- plot ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
xr = np.arange(m.max_rank + 1)
a1.bar(xr - 0.18, hist_all, 0.36, label=f"all {m.n_atoms}", color="C7")
a1.bar(xr + 0.18, hist_act, 0.36, label=f"active {int(ever.sum())}", color="C0")
a1.set_xlabel("learned rank (# on-dims)"); a1.set_ylabel("# atoms")
a1.set_title(f"{RUN.name}: per-atom rank dist  (mean active {ar[ever].float().mean():.2f})")
a1.legend(fontsize=8); a1.set_xticks(xr)
types = [t for t in ["circle","sphere","torus","mobius","swiss_roll","helix","flat_disk","segment"] if by[t]["n"]]
oc = [np.mean(by[t]["ocap"]) if by[t]["ocap"] else 0 for t in types]
lr = [np.mean(by[t]["lr"]) if by[t]["lr"] else 0 for t in types]
xt = np.arange(len(types))
a2.bar(xt - 0.2, oc, 0.4, label="oracle min-cap", color="crimson", alpha=0.8)
a2.bar(xt + 0.2, lr, 0.4, label="learned (dedicating)", color="C0")
for i, t in enumerate(types):
    a2.annotate(f"ki{by[t]['ki']}", (i, max(oc[i], lr[i]) + 0.05), ha="center", fontsize=7, color="grey")
a2.set_xticks(xt); a2.set_xticklabels(types, rotation=40, ha="right", fontsize=8)
a2.set_ylabel("rank"); a2.set_title("Per-type: oracle min-cap vs learned dedicating-atom rank")
a2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "rank_dist.png", dpi=140); plt.close(fig)
print(f"\nwrote {OUT/'rank_dist.png'}")
