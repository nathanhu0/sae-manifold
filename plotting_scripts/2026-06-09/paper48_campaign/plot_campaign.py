"""Paper 48-zoo (L0=4) campaign plots: learn-rank LR x K x lambda x preact.

Reads the run dirs (skips any not yet finished) and writes 3 figures into this folder:
  fig1_frontier.png  : FVU(mix) vs act_rank (top) + single-capture/48 vs act_rank (bottom),
                       Stage-2 grid colored by LR (3e-4 vs 1e-3), shape by K (K8 o / K4 ^ open),
                       LR-sweep markers, 48-zoo oracle floor if available. Best cell annotated.
  fig2_heatmap.png   : single-capture/48 over (lambda x preact-mult) for K8 @ the best LR.
  fig3_datascale.png : single-capture & FVU vs training steps (data points) @ K8/best-LR.

x = act_rank = avg active TOTAL rank/sample (sum_i active_i*rank_i); lower = the model
self-pruned. Capture is /48 (single-atom isolated FVU<0.05) -- the dedication metric.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/nlp/scr/nathu/sae-manifold/runs")
OUT = Path(__file__).parent
GRID_RE = re.compile(r"k(\d+)_lam([\d.]+)_rev([\d.]+)x")


def load_grid(base, lr):
    rows = []
    d0 = RUNS / base
    if not d0.exists():
        return rows
    for d in sorted(d0.glob("k*")):
        m = GRID_RE.match(d.name); mj = d / "metrics.json"
        if not (m and mj.exists()):
            continue
        r = json.load(open(mj))
        # recompute capture at BOTH thresholds from per_instance single-atom FVU (the 0.05 cliff
        # hides near-misses like segment@0.09); also the continuous mean single FVU.
        singles = [p["single"] for p in r.get("per_instance", [])]
        cs05 = sum(s < 0.05 for s in singles) if singles else r["captured_single"]
        cs10 = sum(s < 0.10 for s in singles) if singles else None
        ms = float(np.mean(singles)) if singles else r.get("mean_single_fvu")
        rows.append(dict(K=int(m.group(1)), lam=float(m.group(2)), mult=float(m.group(3)),
                         rev=r.get("lam_preact_dim"), lr=lr, fvu=r["fvu"], act_rank=r["act_rank"],
                         cs=cs05, cs10=cs10, mean_single=ms, cf=r["captured_full"], n=r["n_inst"]))
    return rows


def load_lrsweep():
    rows = []
    d0 = RUNS / "2026-06-09_paper48_lrsweep"
    if not d0.exists():
        return rows
    for d in sorted(d0.glob("k*_lr*")):
        m = re.match(r"k(\d+)_lr([\de.-]+)", d.name); mj = d / "metrics.json"
        if not (m and mj.exists()):
            continue
        r = json.load(open(mj))
        rows.append(dict(K=int(m.group(1)), lr=m.group(2), fvu=r["fvu"], act_rank=r["act_rank"],
                         cs=r["captured_single"], cf=r["captured_full"], n=r["n_inst"]))
    return rows


def load_oracle(L0=4):
    """Per-instance oracle -> assembled reference at L0=4: act_rank = (L0/N)*sum_i min-cap-rank_i,
    FVU = mean per-instance floor (FVU at min-capturing rank). metrics.json is keyed by instance."""
    mj = RUNS / "2026-06-09_oracle_48" / "metrics.json"
    if not mj.exists():
        return None
    o = json.load(open(mj))
    sum_r, floor = 0, []
    for d in o.values():
        if not (isinstance(d, dict) and isinstance(d.get("fvu"), dict)):
            continue
        fv = {int(k): v for k, v in d["fvu"].items()}
        cap = next((r for r in sorted(fv) if fv[r] < 0.05), max(fv))
        sum_r += cap; floor.append(fv[cap])
    if not floor:
        return None
    return dict(assembled_act_rank=L0 / len(floor) * sum_r, assembled_fvu=float(np.mean(floor)))


def main():
    g34 = load_grid("2026-06-09_paper48_stage2_lr3e-4", "3e-4")
    g13 = load_grid("2026-06-09_paper48_stage2", "1e-3")
    gfl = load_grid("2026-06-09_paper48_floor", "3e-4+floor")
    allg = g34 + g13 + gfl
    lrs = load_lrsweep()
    n = (allg or lrs or [{"n": 48}])[0]["n"]
    print(f"loaded: stage2 lr3e-4={len(g34)} lr1e-3={len(g13)} floor={len(gfl)}  lrsweep={len(lrs)}  N={n}")
    if not allg and not lrs:
        print("no data yet"); return

    # ---------- fig1: frontier ----------
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.4, 9), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.1, 1]))
    style = {"3e-4": dict(c="C0"), "1e-3": dict(c="C3"), "3e-4+floor": dict(c="C2")}
    for lr, rows in (("3e-4", g34), ("1e-3", g13), ("3e-4+floor", gfl)):
        for K, mk, fill in ((8, "o", True), (4, "^", False)):
            pts = [r for r in rows if r["K"] == K]
            if not pts:
                continue
            c = style[lr]["c"]
            kw = dict(marker=mk, s=80, edgecolor=c, linewidth=1.3, zorder=4, label=f"K{K} {lr}")
            kw["c" if fill else "facecolor"] = c if fill else "none"
            ax.scatter([p["act_rank"] for p in pts], [p["fvu"] for p in pts], **kw)
            ax2.scatter([p["act_rank"] for p in pts], [p["cs"] for p in pts], **kw)
            # cap@0.10 (faint hollow, above) + connector: the vertical band = near-misses (0.05<FVU<0.10)
            for p in pts:
                if p.get("cs10") is not None:
                    ax2.plot([p["act_rank"]] * 2, [p["cs"], p["cs10"]], c=c, lw=0.8, alpha=0.4, zorder=2)
                    ax2.scatter(p["act_rank"], p["cs10"], marker=mk, s=34, facecolor="none",
                                edgecolor=c, linewidth=0.8, alpha=0.55, zorder=3)
    for r in lrs:  # LR-sweep core as small grey diamonds
        ax.scatter(r["act_rank"], r["fvu"], marker="D", s=28, c="grey", alpha=0.5, zorder=2)
        ax2.scatter(r["act_rank"], r["cs"], marker="D", s=28, c="grey", alpha=0.5, zorder=2)
    o = load_oracle()
    if o and "assembled_act_rank" in o:
        ax.scatter([o["assembled_act_rank"]], [o.get("assembled_fvu", o.get("fvu", 0.003))],
                   marker="*", s=320, c="crimson", zorder=6, edgecolor="white", label="oracle (48-zoo)")
    if allg:
        best = max(allg, key=lambda r: (r["cs"], -r["fvu"]))
        ax2.annotate(f"best: {best['lr']} lam{best['lam']:g} {best['mult']:g}x\n"
                     f"{best['cs']}/{n}@.05  ({best.get('cs10','?')}/{n}@.10)\n"
                     f"mean single FVU {best.get('mean_single', float('nan')):.3f}",
                     (best["act_rank"], best["cs"]), fontsize=7.5, xytext=(8, -40),
                     textcoords="offset points", color=style[best["lr"]]["c"],
                     arrowprops=dict(arrowstyle="->", color=style[best["lr"]]["c"], lw=1))
    ax.set_yscale("log"); ax.set_ylabel("FVU(mix) (log)")
    ax.set_title(f"Paper 48-zoo (L0=4), learn-rank: sparsity vs reconstruction  (filled=K8, open=K4)")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2); ax.grid(alpha=0.25, which="both")
    ax2.axhline(n, color="gray", ls=":", lw=1, alpha=0.6)
    ax2.set_xlabel("active rank-budget  (sum_i active_i*rank_i / sample)")
    ax2.set_ylabel(f"single-atom captured /{n}  (filled <0.05, hollow <0.10)")
    ax2.set_ylim(-1, n + 2); ax2.grid(alpha=0.25); ax2.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig1_frontier.png", dpi=140); plt.close(fig)

    # ---------- fig2: K8 single-capture heatmap over (lam x preact-mult) at best LR ----------
    g = g34 if g34 else g13; lr = "3e-4" if g34 else "1e-3"
    k8 = [r for r in g if r["K"] == 8]
    if k8:
        lams = sorted({r["lam"] for r in k8}); mults = sorted({r["mult"] for r in k8})
        M = np.full((len(mults), len(lams)), np.nan)
        for r in k8:
            M[mults.index(r["mult"]), lams.index(r["lam"])] = r["cs"]
        fig, axh = plt.subplots(figsize=(6, 4.5))
        im = axh.imshow(M, origin="lower", aspect="auto", cmap="viridis")
        axh.set_xticks(range(len(lams))); axh.set_xticklabels([f"{l:g}" for l in lams])
        axh.set_yticks(range(len(mults))); axh.set_yticklabels([f"{m:g}x" for m in mults])
        axh.set_xlabel("lambda (sparsity)"); axh.set_ylabel("preact mult (x rule lam/30)")
        axh.set_title(f"K8 lr{lr}: single-capture /{n} over lambda x preact")
        for i in range(len(mults)):
            for j in range(len(lams)):
                if not np.isnan(M[i, j]):
                    axh.text(j, i, f"{int(M[i, j])}", ha="center", va="center",
                             color="white" if M[i, j] < np.nanmax(M) * 0.6 else "black", fontsize=10)
        fig.colorbar(im, label=f"single /{n}"); fig.tight_layout()
        fig.savefig(OUT / "fig2_heatmap.png", dpi=140); plt.close(fig)

    # ---------- fig3: data-scale ----------
    ds = RUNS / "2026-06-09_paper48_datascale"
    dpts = []
    if ds.exists():
        for d in sorted(ds.glob("k8_*")):
            mj = d / "metrics.json"
            st = re.search(r"steps(\d+)", d.name)
            if mj.exists() and st:
                r = json.load(open(mj)); dpts.append((int(st.group(1)), r["captured_single"], r["fvu"]))
    # include the 150k point from the LR-core winner (k8 lr3e-4) if present
    core = RUNS / "2026-06-09_paper48_lrsweep" / "k8_lr3e-4" / "metrics.json"
    if core.exists():
        r = json.load(open(core)); dpts.append((150000, r["captured_single"], r["fvu"]))
    if len(dpts) >= 2:
        dpts = sorted(set(dpts))
        steps = [p[0] / 1000 for p in dpts]
        fig, axd = plt.subplots(figsize=(6.5, 4))
        axd.plot(steps, [p[1] for p in dpts], "o-", c="C0", label=f"single /{n}")
        axd.set_xlabel("training steps (x1000)  ~ data points seen"); axd.set_ylabel(f"single /{n}", color="C0")
        axd.set_title("Data-scaling: more training data vs dedication/recon  (K8, lr3e-4, lam0.003)")
        axt = axd.twinx(); axt.plot(steps, [p[2] for p in dpts], "s--", c="C3", label="FVU")
        axt.set_ylabel("FVU(mix)", color="C3"); axt.set_yscale("log")
        fig.tight_layout(); fig.savefig(OUT / "fig3_datascale.png", dpi=140); plt.close(fig)

    print(f"wrote figures to {OUT}")
    # best per series, reported at BOTH thresholds + mean single FVU (the 0.05 cliff hides near-misses)
    for label, g in (("lr3e-4", g34), ("lr1e-3", g13), ("lr3e-4+FLOOR", gfl)):
        if not g:
            continue
        b = max(g, key=lambda r: (r["cs"], -r["fvu"]))
        print(f"BEST {label:14s}: K{b['K']} lam{b['lam']:g} {b['mult']:g}x -> "
              f"single {b['cs']}/{n}@.05  {b.get('cs10','?')}/{n}@.10  "
              f"mean_single_FVU={b.get('mean_single', float('nan')):.3f}  FVU(mix)={b['fvu']:.4f}  act_rank={b['act_rank']:.1f}")


if __name__ == "__main__":
    main()
