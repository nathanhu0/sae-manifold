"""FVU vs L0 (rank-units) across every mechanism sweep + hard-TopK + linear + oracle.

The money plot: x = L0 (rank-units = sum of active-atom ranks, the genuine budget
axis the oracle and hard-TopK live on), y = reconstruction FVU (log). Each soft-gate
mechanism is a lambda-sweep point set; hard rank-budget TopK and the linear SAE are
budget-swept curves; the oracle is the assembled ceiling (each manifold at its minimal
capturing rank).

Reads ONLY saved metrics.json (no recompute). Three schema versions are normalized to
(atoms, rank_units, fvu) per run. Output saved alongside via OUT_DIR.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/nlp/scr/nathu/sae-manifold/runs")
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(p):
    return json.load(open(RUNS / p / "metrics.json"))


# ---- soft-gate mechanism sweeps: (family dir, lambda subdirs, label, fvu key) ----
MECHS = [
    ("2026-06-08_jumprelu", ["lam005", "lam02", "lam05", "lam1"], "JumpReLU (STE)", "fvu"),
    ("2026-06-08_jumprelu_thetasched", ["lam002", "lam005", "lam01", "lam02"], "JumpReLU + theta-anneal", "fvu"),
    ("2026-06-08_implicit_lasso", ["lam001", "lam003", "lam006", "lam01"], "implicit-lasso (sigmoid)", "fvu"),
    ("2026-06-08_implicit_lasso_temp", ["lam001", "lam003", "lam006", "lam01"], "implicit + tau-anneal", "fvu"),
    ("2026-06-08_leaky_hard", ["lam001", "lam003", "lam006", "lam01"], "leaky-hard clamp", "fvu_soft"),
]


def mech_point(d, fvu_key):
    """Normalize one mechanism metrics dict -> (atoms, rank_units, fvu)."""
    atoms = d["act_atoms"]
    rank_units = d.get("act_rank", d.get("act_rank_soft"))
    return atoms, rank_units, d[fvu_key]


# ---- budget sweep: hard rank-budget TopK (manifold) + linear baseline ----
def budget_points(prefix, budgets):
    pts = []
    for b in budgets:
        d = load(f"2026-06-08_budget_sweep/{prefix}{b}")
        rf = d.get("rank_firing")
        if rf:                                              # manifold: rank-weighted
            atoms = sum(rf.values())
            rank_units = sum(int(r) * c for r, c in rf.items())
        else:                                               # linear: all rank-1
            atoms = rank_units = d["units_per_sample"]
        pts.append((atoms, rank_units, d["fvu"]))
    return pts


MAN_B = [3, 4, 5, 6, 7, 8, 9, 12, 16]
LIN_B = [3, 4, 5, 6, 7, 8, 9, 12, 16]


# ---- oracle frontier: enumerate every per-manifold rank assignment, take the ----
# ---- lower-left (Pareto) envelope of (rank-units, FVU). Orthogonal subspaces + ----
# ---- RMS-1 norm => mixture FVU = mean_t fvu_t(r_t); 4 of 8 active => rank-units ----
# ---- = 4 * mean_t r_t. This is the rigorous dedicated-atom ceiling.            ----
def oracle_frontier():
    import itertools
    d = load("2026-06-08_oracle")
    types = list(d)
    rank_opts = sorted(int(r) for r in d[types[0]]["fvu"])      # [1,2,3,4]
    fvu_t = {t: {int(r): f for r, f in d[t]["fvu"].items()} for t in types}
    pts = []
    for combo in itertools.product(rank_opts, repeat=len(types)):
        ru = 4.0 * np.mean(combo)
        fv = float(np.mean([fvu_t[t][r] for t, r in zip(types, combo)]))
        pts.append((ru, fv, combo))
    # Pareto-min frontier: keep points not dominated (lower ru AND lower fvu)
    pts.sort(key=lambda p: (p[0], p[1]))
    front, best = [], np.inf
    for ru, fv, combo in pts:
        if fv < best - 1e-12:
            front.append((ru, fv, combo))
            best = fv
    return front, types


# ============================ build + plot ============================
fig, ax = plt.subplots(figsize=(9, 6.5))
rows = []  # for CSV

# baselines first (lines, behind)
for prefix, budgets, lbl, color, ls in [
    ("manifold_R", MAN_B, "hard rank-budget TopK (manifold)", "k", "-"),
    ("linear_k", LIN_B, "linear SAE (baseline)", "gray", "--"),
]:
    pts = budget_points(prefix, budgets)
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.plot(xs, ys, ls, color=color, lw=1.6, marker="o", ms=4, label=lbl, zorder=2)
    for (a, ru, fv), b in zip(pts, budgets):
        rows.append((lbl, b, a, ru, fv))

# mechanism sweeps (point sets, ordered by lambda -> faint connector)
cmap = plt.cm.tab10
for i, (fam, subs, lbl, fkey) in enumerate(MECHS):
    pts = [mech_point(load(f"{fam}/{s}"), fkey) for s in subs]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    c = cmap(i)
    ax.plot(xs, ys, "-", color=c, lw=0.8, alpha=0.4, zorder=3)
    ax.scatter(xs, ys, color=c, s=55, label=lbl, zorder=4, edgecolor="white", linewidth=0.6)
    for (a, ru, fv), s in zip(pts, subs):
        rows.append((lbl, s, a, ru, fv))

# oracle frontier (dedicated-atom ceiling, Pareto-min over all rank assignments)
front, otypes = oracle_frontier()
oxs = [p[0] for p in front]
oys = [p[1] for p in front]
ax.plot(oxs, oys, "-*", color="crimson", lw=1.8, ms=12, zorder=6,
        markeredgecolor="black", markeredgewidth=0.5,
        label="oracle frontier (dedicated atoms)")
for ru, fv, combo in front:
    rows.append(("oracle frontier", "+".join(f"{t[:3]}{r}" for t, r in zip(otypes, combo)),
                 4.0, ru, fv))
print("oracle frontier (rank-units, FVU):")
for ru, fv, combo in front:
    print(f"  {ru:5.1f}  {fv:.5f}  ranks={dict(zip([t[:4] for t in otypes], combo))}")

ax.set_yscale("log")
ax.set_xlabel("L0  (rank-units = sum of active-atom ranks)")
ax.set_ylabel("reconstruction FVU  (log scale)")
ax.set_title("Manifold-SAE: FVU vs L0 across gating mechanisms (toy zoo, 48 instances)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT_DIR / "fvu_vs_l0.png", dpi=150)
print("saved", OUT_DIR / "fvu_vs_l0.png")

# ---- CSV ----
import csv
with open(OUT_DIR / "fvu_vs_l0.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "point", "atoms", "rank_units", "fvu"])
    for r in rows:
        w.writerow(r)
print("saved", OUT_DIR / "fvu_vs_l0.csv")
knee = min(front, key=lambda p: p[0] if p[1] < 0.005 else 1e9)
print(f"\noracle knee: {knee[0]:.1f} rank-units, FVU={knee[1]:.4f}")
