"""Honest FVU-vs-L0 + dedication, from the uniform re-eval (metrics_v2.json).

Panel A (FVU vs L0): for each manifold-SAE method plot the THRESHOLDED FVU (binarize
gate>0.5 -> the deployable discrete code) as the solid marker, with the OPERATING FVU
(the method's own forward, soft for implicit/leaky) as a faint ghost + connector. The
soft->hard jump shows how much each method leans on continuous gate values. Hard
rank-budget TopK and JumpReLU barely move (real hard selection); implicit/leaky jump
up toward the oracle. Oracle frontier (dedicated atoms) + linear baseline for reference.

Panel B (dedication): captured-by-single-atom / 48 vs L0 -- does ONE atom reconstruct a
manifold's ground-truth contribution (<0.05 FVU)? Higher = more dedicated. captured_recon
(union, projected onto V_i) shown faint; the recon<single inversions reflect cross-talk
(co-active atoms spill into V_i), flagged in the writeup.

x-axis = L0 in rank-units (sum of active-atom ranks), the budget axis the oracle lives on.
"""
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/nlp/scr/nathu/sae-manifold/runs")
OUT_DIR = Path(__file__).parent

MECHS = [
    ("2026-06-08_jumprelu", ["lam005", "lam02", "lam05", "lam1"], "JumpReLU (STE)"),
    ("2026-06-08_jumprelu_thetasched", ["lam002", "lam005", "lam01", "lam02"], "JumpReLU + theta"),
    ("2026-06-08_implicit_lasso", ["lam001", "lam003", "lam006", "lam01"], "implicit-lasso"),
    ("2026-06-08_implicit_lasso_temp", ["lam001", "lam003", "lam006", "lam01"], "implicit + tau"),
    ("2026-06-08_leaky_hard", ["lam001", "lam003", "lam006", "lam01"], "leaky-hard clamp"),
]
MAN_B = [3, 4, 5, 6, 7, 8, 9, 12, 16]


def load_v2(p):
    return json.load(open(RUNS / p / "metrics_v2.json"))


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


# ============================ figure ============================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 7.2))
cmap = plt.cm.tab10
rows = []

# --- oracle frontier (Panel A) ---
front = oracle_frontier()
axA.plot([p[0] for p in front], [p[1] for p in front], "-*", color="crimson",
         lw=1.8, ms=11, zorder=6, markeredgecolor="black", markeredgewidth=0.5,
         label="oracle frontier (dedicated)")

# --- linear baseline (Panel A): hard by construction, from saved metrics ---
lin = []
for b in MAN_B:
    d = json.load(open(RUNS / f"2026-06-08_budget_sweep/linear_k{b}/metrics.json"))
    lin.append((d["units_per_sample"], d["fvu"]))
axA.plot([p[0] for p in lin], [p[1] for p in lin], "--", color="gray", lw=1.5,
         marker="o", ms=4, zorder=2, label="linear SAE (baseline)")

# --- hard rank-budget TopK (both panels) ---
topk = [load_v2(f"2026-06-08_budget_sweep/manifold_R{b}") for b in MAN_B]
tk_x = [d["act_rank_hard"] for d in topk]
axA.plot(tk_x, [d["fvu_hard"] for d in topk], "-o", color="black", lw=1.6, ms=4,
         zorder=5, label="hard rank-budget TopK")
axB.plot(tk_x, [d["captured_single_hard"] for d in topk], "-o", color="black",
         lw=1.6, ms=4, zorder=5, label="hard rank-budget TopK")
for b, d in zip(MAN_B, topk):
    rows.append(("hard_topk", b, d["act_rank_hard"], d["fvu_soft"], d["fvu_hard"],
                 d["captured_recon"], d["captured_single_hard"]))

# --- mechanism sweeps: hard primary + soft ghost (Panel A); captured (Panel B) ---
for i, (fam, subs, lbl) in enumerate(MECHS):
    ds = [load_v2(f"{fam}/{s}") for s in subs]
    c = cmap(i)
    xh = [d["act_rank_hard"] for d in ds]
    xs = [d["act_rank_soft"] for d in ds]
    # connector soft<->hard FVU
    for d in ds:
        axA.plot([d["act_rank_soft"], d["act_rank_hard"]], [d["fvu_soft"], d["fvu_hard"]],
                 "-", color=c, lw=0.5, alpha=0.35, zorder=3)
    axA.scatter(xs, [d["fvu_soft"] for d in ds], color=c, s=28, alpha=0.35,
                marker="o", zorder=3, edgecolor="none")          # ghost: operating
    axA.scatter(xh, [d["fvu_hard"] for d in ds], color=c, s=60, zorder=4,
                marker="D", edgecolor="white", linewidth=0.6, label=lbl)  # solid: thresholded
    axB.plot(xh, [d["captured_single_hard"] for d in ds], "-D", color=c, ms=6,
             lw=0.8, label=lbl)
    axB.scatter(xh, [d["captured_recon"] for d in ds], color=c, s=22, alpha=0.4,
                marker="o", zorder=2)                            # faint: recon (union)
    for s, d in zip(subs, ds):
        rows.append((lbl, s, d["act_rank_hard"], d["fvu_soft"], d["fvu_hard"],
                     d["captured_recon"], d["captured_single_hard"]))

# anchor annotations
axA.annotate("oracle knee\n7 units / FVU 0.002", (front[6][0], front[6][1]),
             textcoords="offset points", xytext=(6, -22), fontsize=8, color="crimson")
tk12 = topk[MAN_B.index(12)]
axA.annotate("TopK R12: hard FVU 0.008\n(soft 0.004 -- tiny gap)",
             (tk12["act_rank_hard"], tk12["fvu_hard"]), textcoords="offset points",
             xytext=(-30, -34), fontsize=8)
axB.annotate("TopK R12:\n19/48 dedicated", (tk12["act_rank_hard"], tk12["captured_single_hard"]),
             textcoords="offset points", xytext=(-70, 6), fontsize=8)

axA.set_yscale("log")
axA.set_xlabel("L0  (rank-units = sum of active-atom ranks)")
axA.set_ylabel("reconstruction FVU  (log)")
axA.set_title("A. FVU vs L0  (solid=thresholded/deployable, ghost=operating)")
axA.grid(True, which="both", alpha=0.25)
axA.legend(fontsize=7.5, loc="lower left", framealpha=0.95)

axB.axhline(48, color="crimson", ls=":", lw=1.2, label="oracle (48/48 by construction)")
axB.set_xlabel("L0  (rank-units)")
axB.set_ylabel("manifolds captured by a SINGLE atom  (/48)")
axB.set_title("B. Dedication  (diamond+line=single atom, faint dot=union/recon)")
axB.grid(True, alpha=0.25)
axB.legend(fontsize=7.5, loc="upper right", framealpha=0.95)

fig.suptitle("Manifold-SAE on the toy zoo (48 instances): reconstruction vs dedication",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "fvu_vs_l0_hard.png", dpi=150)
print("saved", OUT_DIR / "fvu_vs_l0_hard.png")

with open(OUT_DIR / "fvu_vs_l0_hard.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "point", "rank_units_hard", "fvu_operating", "fvu_thresholded",
                "captured_recon_/48", "captured_single_hard_/48"])
    for r in rows:
        w.writerow(r)
print("saved", OUT_DIR / "fvu_vs_l0_hard.csv")
