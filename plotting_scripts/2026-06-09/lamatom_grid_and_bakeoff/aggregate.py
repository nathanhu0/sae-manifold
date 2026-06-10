"""Aggregate the 2026-06-09 lam_atom / lambda-grid / gate-bakeoff / fixed-pool campaigns.

Reads each run's metrics.json (written by manifold_ae.eval_and_viz, re-evaled post-campaign) and
builds the final figures + tables:
  1. pareto_rank_fvu.png / pareto_atoms_fvu.png  - act_rank (and atoms/sample) vs mixture FVU
  2. dose_response.png                           - single/tiled/full/in-mixture vs lam_atom per lambda row
  3. heatmaps.png                                - lambda x lam_atom, one panel per capture lens
  4. fvu_ecdf_best.png                           - per-instance single-FVU ECDFs, best cells vs baseline
  5. per_family_best.png                         - per-family capture + mean single FVU + mean used rank
  6. bakeoff.md / fixedpool.md / master_table.md - tables (always incl. atoms/sample + act_rank + dead)

Tolerates missing runs (still training) and missing atoms_per_sample_mean (pre-re-eval evals).
Run (CPU, local): source .venv/bin/activate && python plotting_scripts/2026-06-09/lamatom_grid_and_bakeoff/aggregate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
RUNS = Path("/nlp/scr/nathu/sae-manifold/runs")
N_INST = 48

# (group, label, run_dir) -- baseline points at the floor winner's freshest eval (report_test).
SPECS = []
SPECS.append(("baseline", "lam0.003_atom0", RUNS / "2026-06-09_paper48_floor/k8_lam0.003_rev1x/report_test"))
for la in ["0.0003", "0.001", "0.003", "0.01", "0.03", "0.06", "0.1"]:
    d1 = RUNS / f"2026-06-09_paper48_lamatom_floor/lamatom{la}_seed0"
    d2 = RUNS / f"2026-06-09_paper48_lamatom_floor/lamatom{la}_s0"   # early submissions used _s0
    SPECS.append(("grid", f"lam0.003_atom{la}", d1 if d1.exists() else d2))
for lam in ["0.001", "0.01"]:
    for la in ["0", "0.0001", "0.0003", "0.0006", "0.001", "0.002", "0.003", "0.01", "0.03"]:
        SPECS.append(("grid", f"lam{lam}_atom{la}",
                      RUNS / f"2026-06-09_paper48_lam_lamatom_grid/lam{lam}_lamatom{la}"))
for arm in ["sigmoid_revival_seed0", "sigmoid_norevival_seed0", "rect_norevival_seed0",
            "sigmoid_norevival_eps0.5_seed0", "sigmoid_norevival_eps1.0_seed0",
            "sigmoid_norevival_eps4.0_seed0", "sigmoid_norevival_lamatom0.01_seed0"]:
    SPECS.append(("bakeoff", arm.replace("_seed0", ""), RUNS / f"2026-06-09_paper48_gate_bakeoff/{arm}"))
for arm in ["uniform_lamatom0", "uniform_lamatom0.01", "oracle_lamatom0", "oracle_lamatom0.01",
            "surplus_lamatom0", "surplus_lamatom0.01"]:
    SPECS.append(("fixedpool", arm, RUNS / f"2026-06-09_paper48_fixed_pool/{arm}"))


def load_rows():
    rows = []
    for group, label, d in SPECS:
        mp = d / "metrics.json"
        if not mp.exists():
            print(f"  [skip] {label} (no metrics.json yet: {d})")
            continue
        m = json.load(open(mp))
        # fallback: older standalone evals lack the hyperparam extras -- parse from the label
        lam, lam_atom = m.get("lam"), m.get("lam_atom")
        if lam is None and "lam" in label:
            lam = float(label.split("lam")[1].split("_")[0])
        if lam_atom is None and "atom" in label:
            lam_atom = float(label.split("atom")[1].split("_")[0])
        rows.append(dict(
            group=group, label=label, dir=str(d),
            lam=lam if lam is not None else 0.003, lam_atom=lam_atom or 0.0,
            gate_grad=m.get("gate_grad", "rect"), lam_preact=m.get("lam_preact"),
            learn_rank=m.get("learn_rank", True),
            fvu=m["fvu"], act_rank=m["act_rank"],
            atoms=m.get("atoms_per_sample_mean"),          # None until the re-eval pass
            dead=m["dead_atoms"],
            single=m["captured_single"], tiled=m["captured_tiled"], full=m["captured_full"],
            inmix=m.get("inmix_captured_single"),
            mean_single_fvu=m["mean_single_fvu"],
            per_family=m["per_family"],
            per_instance=m["per_instance"],
            inst_single=[r["single"] for r in m["per_instance"]],
            inst_full=[r["full"] for r in m["per_instance"]]))
    return rows


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def master_table(rows):
    L = ["| group | run | lam | lam_atom | atoms/spl | act_rank | dead | FVU | single | tiled | full | in-mix |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["group"], r["lam"] or 0, r["lam_atom"])):
        L.append(f"| {r['group']} | {r['label']} | {fmt(r['lam'],4)} | {fmt(r['lam_atom'],4)} | "
                 f"{fmt(r['atoms'],1)} | {r['act_rank']:.1f} | {r['dead']} | {r['fvu']:.4f} | "
                 f"{r['single']} | {r['tiled']} | {r['full']} | {fmt(r['inmix'],0)} |")
    (OUT_DIR / "master_table.md").write_text("\n".join(L) + "\n")


LENSES = [("single", "single (1 dedicated atom)"), ("tiled", "tiled (clean atlas)"),
          ("full", "full (union)"), ("inmix", "in-mixture single (deployment)")]
LAM_COLORS = {0.001: "C0", 0.003: "C1", 0.01: "C2"}


def dose_response(rows):
    grid = [r for r in rows if r["group"] in ("grid", "baseline") and r["learn_rank"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (key, title) in zip(axes.flat, LENSES):
        for lam in sorted({r["lam"] for r in grid}):
            rs = sorted([r for r in grid if r["lam"] == lam], key=lambda r: r["lam_atom"])
            x = [max(r["lam_atom"], 2e-5) for r in rs]              # 0 plotted at the left edge
            y = [r[key] if r[key] is not None else np.nan for r in rs]
            ax.plot(x, y, "o-", color=LAM_COLORS.get(lam, "k"), label=f"lambda={lam}")
        ax.set_xscale("log"); ax.set_ylim(0, N_INST + 2)
        ax.axhline(N_INST, color="0.8", lw=0.8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(f"captured / {N_INST}")
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("lam_atom (log; 0 shown at left edge)")
    axes[0][0].legend(fontsize=9)
    fig.suptitle("lam_atom dose-response per lambda row (capture @0.05)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT_DIR / "dose_response.png", dpi=150); plt.close(fig)


def heatmaps(rows):
    grid = [r for r in rows if r["group"] in ("grid", "baseline") and r["learn_rank"]]
    lams = sorted({r["lam"] for r in grid})
    las = sorted({r["lam_atom"] for r in grid})
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (key, title) in zip(axes, LENSES):
        M = np.full((len(lams), len(las)), np.nan)
        for r in grid:
            if r[key] is not None:
                M[lams.index(r["lam"]), las.index(r["lam_atom"])] = r[key]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=N_INST, origin="lower")
        ax.set_xticks(range(len(las)), [f"{v:g}" for v in las], rotation=45, fontsize=8)
        ax.set_yticks(range(len(lams)), [f"{v:g}" for v in lams], fontsize=8)
        ax.set_xlabel("lam_atom"); ax.set_ylabel("lambda")
        ax.set_title(title, fontsize=10)
        for i in range(len(lams)):
            for j in range(len(las)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] < N_INST * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"lambda x lam_atom capture (/{N_INST}) @0.05 — ridge vertical = absolute lam_atom matters, "
                 "diagonal = ratio matters", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT_DIR / "heatmaps.png", dpi=150); plt.close(fig)


def paretos(rows):
    """Sparsity-reconstruction frontiers with MEASURED sparsity on x (atoms firing / sample, and
    rank-dof / sample) -- lam_atom is an annotation along each lambda row's curve, never an axis."""
    for key, fname, xlabel, mark_l0 in [
            ("atoms", "pareto_atoms_fvu.png", "atoms firing / sample (measured)", True),
            ("act_rank", "pareto_rank_fvu.png", "active rank-dof / sample (measured)", False)]:
        rs = [r for r in rows if r[key] is not None]
        if not rs:
            print(f"  [skip] {fname} (no {key} yet — run the re-eval pass)")
            continue
        fig, ax = plt.subplots(figsize=(8.5, 6))
        for lam in sorted({r["lam"] for r in rs if r["group"] in ("grid", "baseline")}):
            seq = sorted([r for r in rs if r["group"] in ("grid", "baseline") and r["lam"] == lam],
                         key=lambda r: r["lam_atom"])
            if not seq:
                continue
            ax.plot([r[key] for r in seq], [r["fvu"] for r in seq], "o-", lw=1.2,
                    color=LAM_COLORS.get(lam, "k"), label=f"lambda={lam} (lam_atom 0 -> {seq[-1]['lam_atom']:g})")
            for r in seq:
                ax.annotate(f"{r['lam_atom']:g}", (r[key], r["fvu"]), fontsize=6, alpha=0.8,
                            xytext=(3, 3), textcoords="offset points")
        for grp, c, mk, lbl in [("bakeoff", "C3", "s", "gate bake-off"), ("fixedpool", "C4", "D", "fixed pool")]:
            gs = [r for r in rs if r["group"] == grp]
            if gs:
                ax.scatter([r[key] for r in gs], [r["fvu"] for r in gs], c=c, marker=mk, s=34,
                           alpha=0.85, label=lbl)
                for r in gs:
                    ax.annotate(r["label"], (r[key], r["fvu"]), fontsize=5, alpha=0.6,
                                xytext=(3, -6), textcoords="offset points")
        if mark_l0:
            ax.axvline(4.0, color="0.7", ls="--", lw=1, label="data L0 = 4 (dedicated code)")
        ax.set_xlabel(xlabel); ax.set_ylabel("mixture FVU (log)"); ax.set_yscale("log")
        ax.set_title("Sparsity vs reconstruction — lam_atom annotated along each lambda row")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(OUT_DIR / fname, dpi=150); plt.close(fig)


def pick_best(rows):
    """Cells for the deep-dive: baseline + each lambda row's best-tiled cell (+ best fixed pool if any)."""
    chosen = [r for r in rows if r["group"] == "baseline"]
    for lam in sorted({r["lam"] for r in rows if r["group"] == "grid"}):
        rs = [r for r in rows if r["group"] == "grid" and r["lam"] == lam]
        if rs:
            chosen.append(max(rs, key=lambda r: (r["tiled"], -(r["fvu"]))))
    fp = [r for r in rows if r["group"] == "fixedpool"]
    if fp:
        chosen.append(max(fp, key=lambda r: (r["tiled"], -(r["fvu"]))))
    return chosen


def fvu_ecdf(rows):
    """Expand the discrete captured counts: per-instance single-FVU ECDF per chosen setting."""
    chosen = pick_best(rows)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, r in enumerate(chosen):
        v = np.sort(np.clip(r["inst_single"], 1e-4, 10))
        ax.step(v, np.arange(1, len(v) + 1), where="post", color=f"C{i}",
                label=f"{r['label']} (single {r['single']}, tiled {r['tiled']})")
    for th, ls in [(0.05, "--"), (0.1, ":")]:
        ax.axvline(th, color="red", ls=ls, lw=1, alpha=0.7)
    ax.set_xscale("log"); ax.set_xlabel("per-manifold best-single-atom FVU (log)")
    ax.set_ylabel(f"# manifolds with FVU <= x (of {N_INST})")
    ax.set_title("Capture as a curve: per-instance single-atom FVU ECDF\n(red lines = 0.05 / 0.10 thresholds; "
                 "the discrete counts are these curves' level-crossings)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fvu_ecdf_best.png", dpi=150); plt.close(fig)


def per_family(rows):
    chosen = pick_best(rows)
    fams = sorted(chosen[0]["per_family"])
    panels = [("captured_single", "captured single (/6)"), ("captured_tiled", "captured tiled (/6)"),
              ("mean_single_fvu", "mean single FVU"), ("mean_rank", "mean used atom rank")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    w = 0.8 / len(chosen)
    xs = np.arange(len(fams))
    for ax, (key, title) in zip(axes.flat, panels):
        for i, r in enumerate(chosen):
            vals = [r["per_family"][f][key] for f in fams]
            ax.bar(xs + i * w, vals, w, color=f"C{i}", label=r["label"])
        ax.set_xticks(xs + 0.4 - w / 2, fams, rotation=30, fontsize=8)
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.2, axis="y")
        if key == "mean_single_fvu":
            ax.set_yscale("log"); ax.axhline(0.05, color="red", ls="--", lw=1)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Per-family deep dive: best cell per lambda row vs baseline", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT_DIR / "per_family_best.png", dpi=150); plt.close(fig)


FAMS = ["circle", "sphere", "torus", "mobius", "swiss_roll", "helix", "flat_disk", "segment"]
FAM_COLOR = {f: plt.get_cmap("tab10")(i) for i, f in enumerate(FAMS)}
RANK_LENSES = [("single", "single-atom FVU (isolated)"), ("cond_fvu", "conditional FVU (own chart)"),
               ("inmix_fvu", "in-mixture single FVU (deployment)"), ("full", "full/union FVU (separate criterion)")]


def rank_vs_fvu(rows):
    """The cutoff-free view: per manifold, x = dedicating atom's rank, y = FVU under each lens
    (log). Thresholds are reference lines, not classifications. One row per chosen run."""
    chosen = pick_best(rows)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(chosen), len(RANK_LENSES),
                             figsize=(4.4 * len(RANK_LENSES), 3.4 * len(chosen)),
                             squeeze=False, sharex=True, sharey=True)
    for i, r in enumerate(chosen):
        for j, (key, title) in enumerate(RANK_LENSES):
            ax = axes[i][j]
            missing = 0
            for m in r["per_instance"]:
                v = m.get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    missing += 1
                    continue
                x = m["best_rank"] + rng.uniform(-0.18, 0.18)
                ax.scatter(x, np.clip(v, 1e-4, 10), s=22, color=FAM_COLOR[m["type"]],
                           alpha=0.85, linewidths=0)
            for th, ls in [(0.05, "--"), (0.1, ":")]:
                ax.axhline(th, color="red", ls=ls, lw=0.9, alpha=0.6)
            ax.set_yscale("log"); ax.set_xticks(range(0, 9))
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{r['label']}\nFVU (log)", fontsize=8)
            if i == len(chosen) - 1:
                ax.set_xlabel("dedicating atom rank")
            if missing and i == 0 and j == 2:
                ax.text(0.5, 0.95, f"({missing} runs lack inmix_fvu — pre-re-eval)",
                        transform=ax.transAxes, fontsize=7, ha="center", va="top")
    handles = [plt.Line2D([], [], marker="o", ls="", color=FAM_COLOR[f], label=f) for f in FAMS]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Per-manifold rank vs FVU (cutoff-free; red lines = 0.05 / 0.10 reference)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "rank_vs_fvu_best.png", dpi=150); plt.close(fig)


def recon_vs_lamatom(rows):
    """The per-atom penalty's direct cost: reconstruction vs lam_atom, one line per lambda row.
    Left: mixture FVU (deployment reconstruction). Right: mean per-manifold single-atom FVU
    (dedication quality, continuous)."""
    grid = [r for r in rows if r["group"] in ("grid", "baseline") and r["learn_rank"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, title in [(axes[0], "fvu", "mixture FVU (reconstruction)"),
                           (axes[1], "mean_single_fvu", "mean single-atom FVU (dedication quality)")]:
        for lam in sorted({r["lam"] for r in grid}):
            rs = sorted([r for r in grid if r["lam"] == lam], key=lambda r: r["lam_atom"])
            x = [max(r["lam_atom"], 2e-5) for r in rs]
            ax.plot(x, [r[key] for r in rs], "o-", color=LAM_COLORS.get(lam, "k"), label=f"lambda={lam}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("lam_atom (log; 0 at left edge)"); ax.set_ylabel("FVU (log)")
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.suptitle("Reconstruction cost of the per-atom penalty", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT_DIR / "recon_vs_lamatom.png", dpi=150); plt.close(fig)


def side_tables(rows):
    for group, fname in [("bakeoff", "bakeoff.md"), ("fixedpool", "fixedpool.md")]:
        rs = [r for r in rows if r["group"] == group] + [r for r in rows if r["group"] == "baseline"]
        L = ["| run | atoms/spl | act_rank | dead | FVU | single | tiled | full | in-mix |",
             "|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            L.append(f"| {r['label']} | {fmt(r['atoms'],1)} | {r['act_rank']:.1f} | {r['dead']} | "
                     f"{r['fvu']:.4f} | {r['single']} | {r['tiled']} | {r['full']} | {fmt(r['inmix'],0)} |")
        (OUT_DIR / fname).write_text("\n".join(L) + "\n")


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} runs")
    master_table(rows)
    dose_response(rows)
    heatmaps(rows)
    paretos(rows)
    fvu_ecdf(rows)
    per_family(rows)
    recon_vs_lamatom(rows)
    side_tables(rows)
    # NB the cutoff-free rank-vs-FVU view is a PER-RUN figure now (rank_vs_fvu.png in each run
    # dir, built by eval_and_viz._rank_fvu_fig); the aggregation links the best runs' reports.
    print(f"wrote figures + tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
