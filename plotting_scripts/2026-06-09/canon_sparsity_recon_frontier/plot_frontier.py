"""Sparsity vs reconstruction for the canonical-gate sweep on the 16-manifold zoo.

x = active rank-budget (Sum_i rank_i * active_i per sample, the quantity lambda penalizes)
y = reconstruction error FVU(mix) on real mixtures  (top panel)
    captured manifolds /16, single-atom vs full-model union  (bottom panel)

The lambda sweep (lr 1e-3, 20k) is the sparsity-only frontier -> ONE connected line.
LR variations (lambda 0.003) and the longer 60k run vary a DIFFERENT hparam -> separate markers.
Oracle = assembled dedicated atoms (one funnel/manifold at its minimal capturing rank):
act_rank ~ 0.25*(2*14)=7.0, FVU ~ 0.002, derived from runs/2026-06-08_oracle/oracle_table.md.
(Linear baseline on THIS dataset not yet computed -- the 06-08 linear runs are the old
fixed-L0 48-zoo, not comparable to independent p_active presence; omitted, not faked.)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-09_jump_canon_sweep")
OUT_DIR = Path(__file__).parent
# Measured per-instance oracle (runs/2026-06-09_oracle_simplified): dedicated atom/instance.
# Expected assembled act_rank = 0.25 * sum_i rank_i; FVU = mean of per-instance isolated FVUs (est.).
# Two operating points: sparsest-capturing (min rank, FVU<0.05) and near-floor (min rank ~ best FVU).
ORACLE_FRONTIER = [(5.5, 0.0108), (7.5, 0.0024)]   # (act_rank, FVU)
DIR_RE = re.compile(r"lam([\d.]+)_lr([\de.-]+)_steps(\d+)k")
OPT_SWEEP = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-09_opt_sweep")  # 100k LR/sched/preact sweep
OPT_RE = re.compile(r"lr([\de.-]+)_(\w+?)_preact([\de.-]+)_(lam\w+)")
LEARNRANK = Path("/nlp/scr/nathu/sae-manifold/runs/2026-06-09_learnrank_sweep2")  # per-dim learned rank, K8
LRNK_RE = re.compile(r"k8_lam([\d.]+)_rev([\de.-]+)")


def load_opt():
    runs = []
    if not OPT_SWEEP.exists():
        return runs
    for d in sorted(OPT_SWEEP.iterdir()):
        m = OPT_RE.match(d.name); mj = d / "metrics.json"
        if not (m and mj.exists()):
            continue
        r = json.load(open(mj))
        runs.append(dict(lr=m.group(1), sched=m.group(2), preact=m.group(3), lamsched=m.group(4),
                         fvu=r["fvu"], act_rank=r["act_rank"], cs=r["captured_single"], cf=r["captured_full"]))
    return runs


def load_learnrank():
    """Per-dim learned-rank sweep2 (K8, 32 atoms). act_rank here = avg active TOTAL rank/sample,
    same x-axis as the fixed-rank runs (lower = the model pruned its own dims). cs<8 flags the
    over-prune collapse corners (rev << sparsity force)."""
    runs = []
    if not LEARNRANK.exists():
        return runs
    for d in sorted(LEARNRANK.iterdir()):
        m = LRNK_RE.match(d.name); mj = d / "metrics.json"
        if not (m and mj.exists()):
            continue
        r = json.load(open(mj))
        runs.append(dict(lam=float(m.group(1)), rev=float(m.group(2)),
                         fvu=r["fvu"], act_rank=r["act_rank"],
                         cs=r["captured_single"], cf=r["captured_full"]))
    return runs


def load():
    runs = []
    for d in sorted(SWEEP.iterdir()):
        m = DIR_RE.match(d.name)
        mj = d / "metrics.json"
        if not (m and mj.exists()):
            continue
        r = json.load(open(mj))
        runs.append(dict(lam=float(m.group(1)), lr=m.group(2), steps=int(m.group(3)) * 1000,
                         fvu=r["fvu"], act_rank=r["act_rank"],
                         cs=r["captured_single"], cf=r["captured_full"], n=r["n_inst"]))
    return runs


def main():
    runs = load()
    is_lr1e3 = lambda r: r["lr"] in ("1e-3", "0.001")
    lam_line = sorted([r for r in runs if is_lr1e3(r) and r["steps"] == 20000], key=lambda r: r["act_rank"])
    lr_pts = [r for r in runs if abs(r["lam"] - 0.003) < 1e-9 and r["steps"] == 20000 and not is_lr1e3(r)]
    steps_pts = [r for r in runs if r["steps"] == 60000]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.2, 9), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.25, 1]))

    # ---- top: FVU vs sparsity ----
    xs = [r["act_rank"] for r in lam_line]
    ax.plot(xs, [r["fvu"] for r in lam_line], "o-", color="C0", lw=2, ms=8, zorder=3,
            label="lambda sweep (lr 1e-3, 20k)")
    for r in lam_line:
        ax.annotate(f"lambda={r['lam']:g}", (r["act_rank"], r["fvu"]), fontsize=7.5,
                    xytext=(4, 6), textcoords="offset points", color="C0")
    for r in lr_pts:
        ax.scatter(r["act_rank"], r["fvu"], marker="D", s=80, color="C1", zorder=4,
                   edgecolor="white", linewidth=0.6)
        ax.annotate(f"lr={r['lr']}", (r["act_rank"], r["fvu"]), fontsize=7.5,
                    xytext=(4, -11), textcoords="offset points", color="C1")
    for r in steps_pts:
        ax.scatter(r["act_rank"], r["fvu"], marker="s", s=80, color="C2", zorder=4,
                   edgecolor="white", linewidth=0.6, label="60k steps (lambda 0.003)")
    ox = [p[0] for p in ORACLE_FRONTIER]; oy = [p[1] for p in ORACLE_FRONTIER]
    ax.plot(ox, oy, "*-", color="crimson", ms=20, lw=1.5, zorder=5, markeredgecolor="white",
            markeredgewidth=0.8, label="oracle frontier (per-instance, est.)")
    ax.axhline(min(oy), color="crimson", ls=":", lw=1, alpha=0.5)
    ax.scatter([], [], marker="D", s=80, color="C1", label="lr sweep (lambda 0.003, 20k)")
    opt = [o for o in load_opt() if o["fvu"] < 0.3]    # drop the diverged lr1e-2 run (off-scale)
    if opt:
        ax.scatter([o["act_rank"] for o in opt], [o["fvu"] for o in opt], marker="P", s=85,
                   color="purple", zorder=4, edgecolor="white", linewidth=0.5,
                   label="100k opt-sweep (lambda 0.003)")
    lrnk = load_learnrank()
    lrnk_ok = [r for r in lrnk if r["cs"] >= 8]      # stable (rev ~0.8x sparsity): dedicated + pruned
    lrnk_bad = [r for r in lrnk if r["cs"] < 8]       # collapsed (rev << sparsity): frozen -> spanning
    if lrnk_ok:
        ax.scatter([r["act_rank"] for r in lrnk_ok], [r["fvu"] for r in lrnk_ok], marker="h", s=120,
                   color="teal", zorder=6, edgecolor="white", linewidth=0.8,
                   label="learn-rank K8 (stable, rev~0.8x sparsity)")
        for r in lrnk_ok:
            ax.annotate(f"lam{r['lam']:g}/rev{r['rev']:g}", (r["act_rank"], r["fvu"]), fontsize=7,
                        xytext=(5, 5), textcoords="offset points", color="teal")
    if lrnk_bad:
        ax.scatter([r["act_rank"] for r in lrnk_bad], [r["fvu"] for r in lrnk_bad], marker="h", s=95,
                   facecolor="none", edgecolor="teal", linewidth=1.3, zorder=5,
                   label="learn-rank K8 (collapsed, rev<<sparsity)")
    ax.set_yscale("log")
    ax.set_ylabel("reconstruction error  FVU(mix)  (log)")
    ax.set_title("Canonical-gate sweep: sparsity vs reconstruction (16-manifold zoo)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25, which="both")

    # ---- bottom: dedication (captured /16) vs sparsity ----
    n = lam_line[0]["n"] if lam_line else 16
    ax2.plot(xs, [r["cf"] for r in lam_line], "^--", color="C0", lw=1.6, ms=9, alpha=0.8,
             label="full model (union)")
    ax2.plot(xs, [r["cs"] for r in lam_line], "o-", color="C0", lw=2, ms=8,
             label="best single atom")
    for r in lr_pts + steps_pts:
        c = "C1" if r in lr_pts else "C2"
        ax2.scatter(r["act_rank"], r["cf"], marker="^", s=70, color=c, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax2.scatter(r["act_rank"], r["cs"], marker="o", s=70, color=c, edgecolor="white", linewidth=0.5)
    ax2.axhline(n, color="gray", ls=":", lw=1, alpha=0.6)
    ax2.annotate(f"all {n}", (xs[-1], n), fontsize=7.5, xytext=(2, 3), textcoords="offset points", color="gray")
    ax2.scatter([ORACLE_FRONTIER[0][0]], [n], marker="*", s=320, color="crimson", zorder=6,
                edgecolor="white", linewidth=0.8, label=f"oracle ({n}/{n} single @ rank {ORACLE_FRONTIER[0][0]})")
    for o in opt:
        ax2.scatter(o["act_rank"], o["cf"], marker="^", s=55, color="purple", alpha=0.65, edgecolor="white", linewidth=0.4)
        ax2.scatter(o["act_rank"], o["cs"], marker="P", s=70, color="purple", edgecolor="white", linewidth=0.4)
    for i, r in enumerate(lrnk_ok):
        ax2.scatter(r["act_rank"], r["cf"], marker="^", s=70, color="teal", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax2.scatter(r["act_rank"], r["cs"], marker="h", s=110, color="teal", edgecolor="white", linewidth=0.7,
                    zorder=6, label="learn-rank single (stable)" if i == 0 else None)
    for r in lrnk_bad:
        ax2.scatter(r["act_rank"], r["cs"], marker="h", s=85, facecolor="none", edgecolor="teal", linewidth=1.3)
    ax2.set_ylim(-0.5, n + 1)
    ax2.set_xlabel("active rank-budget  (Sum_i rank_i * active_i per sample)")
    ax2.set_ylabel(f"manifolds captured  (/{n}, FVU<0.05)")
    ax2.set_title("Dedication gap: union captures many, no single atom does  (blue=lambda line; orange/green=lr/60k)")
    ax2.legend(loc="center right", fontsize=8.5)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = OUT_DIR / "sparsity_recon_frontier.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    print(f"lambda line: {[(r['lam'], round(r['act_rank'],1), round(r['fvu'],4), r['cs'], r['cf']) for r in lam_line]}")
    print(f"lr pts:      {[(r['lr'], round(r['act_rank'],1), round(r['fvu'],4), r['cs'], r['cf']) for r in lr_pts]}")
    print(f"60k pts:     {[(round(r['act_rank'],1), round(r['fvu'],4), r['cs'], r['cf']) for r in steps_pts]}")
    for r in sorted(lrnk, key=lambda r: r["act_rank"]):
        tag = "stable" if r["cs"] >= 8 else "collapsed"
        print("learn-rank:  lam%-6g rev%-6g  act_rank=%4.1f  FVU=%.4f  single=%2d/16  full=%2d/16  [%s]"
              % (r["lam"], r["rev"], r["act_rank"], r["fvu"], r["cs"], r["cf"], tag))


if __name__ == "__main__":
    main()
