"""Per-family capture scatter (single best-atom FVU only) for the 4 grid runs closest to the
dedicated operating point atoms/sample = 4, drawn from the lambda=0.001 (blue) and lambda=0.003
(orange) rows. Same pane structure as the per-run rank_vs_fvu figure, but ONE metric -- so color
identifies the RUN instead of the lens.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
RUNS = Path("/nlp/scr/nathu/sae-manifold/runs")

cands = []
for d in list((RUNS / "2026-06-09_paper48_lamatom_floor").glob("lamatom*_s*0")) + \
         list((RUNS / "2026-06-09_paper48_lam_lamatom_grid").glob("lam0.001_*")) + \
         [RUNS / "2026-06-09_paper48_floor/k8_lam0.003_rev1x"]:
    mp = d / "metrics.json"
    if not mp.exists():
        continue
    m = json.load(open(mp))
    if m.get("lam") not in (0.001, 0.003) or m.get("atoms_per_sample_mean") is None:
        continue
    cands.append((abs(m["atoms_per_sample_mean"] - 4.0), m))
cands.sort(key=lambda t: t[0])
chosen = [m for _, m in cands[:4]]

fams = {}
for m0 in chosen[0]["per_instance"]:
    fams.setdefault(m0["type"], m0)
names = sorted(fams)
fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), squeeze=False, sharey=True)
rng = np.random.default_rng(0)
for ax, fam in zip(axes.flat, names):
    for ci, m in enumerate(chosen):
        for r in m["per_instance"]:
            if r["type"] != fam:
                continue
            ax.scatter(r["best_rank"] + rng.uniform(-0.18, 0.18), max(r["single"], 1e-4),
                       color=f"C{ci}", s=26, alpha=0.85, linewidths=0)
    for th, ls in [(0.05, "--"), (0.1, ":")]:
        ax.axhline(th, color="red", ls=ls, lw=0.9, alpha=0.6)
    f0 = fams[fam]
    ax.set_yscale("log"); ax.set_ylim(8e-5, 12)
    ax.set_title(f"{fam} — {f0['di']}D manifold in {f0['ki']}D subspace", fontsize=8)
    ax.set_xticks(range(0, 7)); ax.grid(alpha=0.2)
for ax in axes[:, 0]:
    ax.set_ylabel("single best-atom FVU (log)")
for ax in axes[-1]:
    ax.set_xlabel("best atom rank")
handles = [plt.Line2D([], [], marker="o", ls="", color=f"C{ci}",
                      label=f"lam{m['lam']:g}_atom{m.get('lam_atom', 0) or 0:g} "
                            f"({m['atoms_per_sample_mean']:.1f} atoms/spl, "
                            f"{m['captured_single']} atoms learned/48)")
           for ci, m in enumerate(chosen)]
fig.legend(handles=handles, loc="upper right", fontsize=8)
fig.suptitle("Single-atom capture, the 4 runs nearest atoms/sample = 4 (lambda 0.001 + 0.003 rows)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT_DIR / "capture_scatter_near4.png", dpi=150)
print("chosen:", [(f"lam{m['lam']:g}_atom{m.get('lam_atom',0) or 0:g}",
                   round(m["atoms_per_sample_mean"], 2)) for m in chosen])
