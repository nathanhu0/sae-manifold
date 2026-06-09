"""Clean manifold-SAE figure: ONLY oracle frontier + linear baseline + the STE hard-gate
sweep (the pure-binary JumpReLU we're tuning). The soft-gate zoo is dropped on purpose
(see memory plot-only-oracle-linear-ste).

Panel A: FVU vs L0 (rank-units). For the STE runs FVU is the deployable number (gate is
binary, soft==hard). Panel B: manifolds captured by a single atom /48.

STE runs are read from RUN_GLOB (metrics_v2.json or metrics.json, whichever exists).
Pass the sweep dir as argv[1]; defaults to the 2026-06-09 STE sweep.
"""
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/nlp/scr/nathu/sae-manifold/runs")
OUT_DIR = Path(__file__).parent
STE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS / "2026-06-09_jump_ste_sweep"


def oracle_frontier():
    d = json.load(open(RUNS / "2026-06-08_oracle/metrics.json"))
    types = list(d)
    ropts = sorted(int(r) for r in d[types[0]]["fvu"])
    fvu_t = {t: {int(r): f for r, f in d[t]["fvu"].items()} for t in types}
    pts = []
    for combo in itertools.product(ropts, repeat=len(types)):
        pts.append((4.0 * np.mean(combo),
                    float(np.mean([fvu_t[t][r] for t, r in zip(types, combo)]))))
    pts.sort()
    front, best = [], np.inf
    for ru, fv in pts:
        if fv < best - 1e-12:
            front.append((ru, fv)); best = fv
    return front


def load_metrics(run):
    for fn in ("metrics_v2.json", "metrics.json"):
        if (run / fn).exists():
            return json.load(open(run / fn))
    return None


def ste_runs():
    """Each STE run -> dict(ru, fvu_deployable, captured_single, lam, lam_preact).
    A connected line traces the SPARSITY sweep (lam) only; lam_preact -> separate lines."""
    out = []
    for run in sorted(STE_DIR.glob("*/")):
        m = load_metrics(run)
        if m is None:
            continue
        ru = m.get("act_rank_hard", m.get("act_rank"))
        fvu = m.get("fvu_hard", m.get("fvu"))               # binary gate: hard==soft
        if ru is None or fvu is None:
            continue
        out.append(dict(ru=ru, fvu=fvu, lam=m.get("lam"),
                        pre=m.get("lam_preact"),
                        cap=m.get("captured_single_hard", m.get("n_captured"))))
    return out


fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))

front = oracle_frontier()
axA.plot([p[0] for p in front], [p[1] for p in front], "-*", color="crimson", lw=1.8,
         ms=11, markeredgecolor="black", markeredgewidth=0.5, zorder=5,
         label="oracle frontier (dedicated)")

lin = [json.load(open(RUNS / f"2026-06-08_budget_sweep/linear_k{b}/metrics.json"))
       for b in [3, 4, 5, 6, 7, 8, 9, 12, 16]]
axA.plot([d["units_per_sample"] for d in lin], [d["fvu"] for d in lin], "--o",
         color="gray", lw=1.5, ms=4, zorder=2, label="linear SAE (baseline)")

ste = ste_runs()
if ste:
    # one line per lam_preact (the non-sparsity HP); each line sweeps lam (sparsity).
    pres = sorted(set(r["pre"] for r in ste), key=lambda v: (v is None, v))
    for j, pre in enumerate(pres):
        g = sorted([r for r in ste if r["pre"] == pre], key=lambda r: r["ru"])
        c = plt.cm.tab10(j)
        lab = f"STE hard gate (λ_preact={pre:g})" if pre is not None else "STE hard gate"
        axA.plot([r["ru"] for r in g], [r["fvu"] for r in g], "-D", color=c, lw=1.2, ms=7,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=6, label=lab)
        gc = [r for r in g if r["cap"] is not None]
        if gc:
            axB.plot([r["ru"] for r in gc], [r["cap"] for r in gc], "-D", color=c,
                     lw=1.2, ms=7, label=lab)
else:
    axA.text(0.5, 0.5, "no STE runs yet\n(point argv[1] at the sweep dir)",
             transform=axA.transAxes, ha="center", va="center", fontsize=10, color="C0")

axA.set_yscale("log")
axA.set_xlabel("L0  (rank-units)"); axA.set_ylabel("reconstruction FVU (log)")
axA.set_title("A. FVU vs L0"); axA.grid(True, which="both", alpha=0.25)
axA.legend(fontsize=9, loc="upper right")

axB.axhline(48, color="crimson", ls=":", lw=1.2, label="oracle 48/48")
axB.set_xlabel("L0  (rank-units)"); axB.set_ylabel("captured by a single atom (/48)")
axB.set_title("B. Dedication"); axB.grid(True, alpha=0.25)
axB.legend(fontsize=9, loc="upper left")

fig.suptitle("Manifold-SAE: oracle vs linear vs STE hard gate (toy zoo, 48 instances)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "clean.png", dpi=150)
print("saved", OUT_DIR / "clean.png", "| STE runs:", len(ste))
