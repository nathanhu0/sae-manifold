"""Re-evaluate every paper-48 campaign checkpoint with the CORRECTED aggregate-FVU metric.

Background: the isolated single/full/per-atom FVU previously used a per-sample mean of ratios
mean_n[||r_n||^2/||x_n||^2], which explodes on origin-crossing manifolds. Commit 432af3a fixed it
to aggregate sum_n||r_n||^2 / sum_n||x_n||^2. Only the floor-winner ckpt was re-evaluated after the
fix; the other ~60 ckpts still carry stale per-sample metrics.json. This script reruns eval_and_viz
on EVERY ckpt (rebuilding the exact model/zoo/scale/presence from the ckpt), backing up each stale
metrics.json -> metrics_persample_buggy.json, regenerating corrected metrics.json + plots in place,
and writing a master old-vs-new comparison.

Run on jag GPU (fast): ebatch reeval slconf40s "PYTHONUNBUFFERED=1 PYTHONPATH=. python reeval_campaign.py"
"""
import glob
import json
import shutil
import sys
import traceback
from pathlib import Path

import torch

from manifold_ae.manifold_zoo import ManifoldZoo
from manifold_ae.manifold_sae import ManifoldSAE
from manifold_ae.eval_and_viz import eval_and_viz

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUN_GLOB = "/nlp/scr/nathu/sae-manifold/runs/2026-06-09_paper48_*/*/ckpt.pt"
OUT_DIR = Path("/juice2/u/nathu/sae-manifold/plotting_scripts/2026-06-09/paper48_campaign/reeval_corrected")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# hparam fields worth preserving into the regenerated metrics.json
EXTRA_KEYS = ["lam", "lam_preact", "jump_eps", "lr", "lr_schedule", "lam_warmup_steps",
              "lam_decorr", "learn_rank", "lam_preact_dim", "l0", "lam_atom", "l0_rank_floor",
              "p_active", "variants_per_type"]


def reeval_one(ckpt_path):
    d = Path(ckpt_path).parent
    mfile = d / "metrics.json"
    old = json.load(open(mfile)) if mfile.exists() else {}
    # preserve the FIRST on-disk metrics (the per-sample buggy ones) exactly once
    backup = d / "metrics_persample_buggy.json"
    if mfile.exists() and not backup.exists():
        shutil.copy(mfile, backup)

    ck = torch.load(ckpt_path, map_location=DEV)
    m = ManifoldSAE(d_model=256, rank_dist=ck["pool"], enc_dims=ck["enc_dims"],
                    jump_eps=ck["jump_eps"], learn_rank=ck["learn_rank"]).to(DEV)
    m.load_state_dict(ck["state_dict"]); m.eval()
    zoo = ManifoldZoo(d=256, seed=0, variants_per_type=ck["variants_per_type"])
    scale = ck["scale"]
    l0 = ck.get("l0", None)
    pa = None if l0 is not None else ck.get("p_active", 0.25)
    extra = {k: ck.get(k) for k in EXTRA_KEYS}
    res = eval_and_viz(m, zoo, scale, d, p_active=pa, l0=l0, extra=extra)
    return old, res


def main():
    ckpts = sorted(glob.glob(RUN_GLOB))
    print(f"[reeval] device={DEV}  {len(ckpts)} checkpoints\n", flush=True)
    rows = []
    for i, cp in enumerate(ckpts):
        rel = "/".join(Path(cp).parts[-3:-1])   # run/cell
        print(f"\n===== [{i+1}/{len(ckpts)}] {rel} =====", flush=True)
        try:
            old, new = reeval_one(cp)
            rows.append(dict(
                run=Path(cp).parts[-3], cell=Path(cp).parts[-2],
                old_single=old.get("captured_single"), new_single=new["captured_single"],
                old_full=old.get("captured_full"), new_full=new["captured_full"],
                old_mean_single=old.get("mean_single_fvu"), new_mean_single=new["mean_single_fvu"],
                fvu_mix=new["fvu"], act_rank=new["act_rank"], dead=new["dead_atoms"]))
        except Exception:
            print(f"  !! FAILED on {cp}", flush=True); traceback.print_exc()
            rows.append(dict(run=Path(cp).parts[-3], cell=Path(cp).parts[-2], error=True))

    # master comparison
    json.dump(rows, open(OUT_DIR / "reeval_comparison.json", "w"), indent=1)
    lines = ["# Campaign re-eval: per-sample (buggy) vs aggregate (fixed) FVU\n",
             "| run | cell | single old->new | full old->new | mean_single old->new | fvu_mix | act_rank | dead |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['run']} | {r['cell']} | ERROR | | | | | |"); continue
        def ms(v): return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        lines.append(f"| {r['run']} | {r['cell']} | {r['old_single']}->{r['new_single']} | "
                     f"{r['old_full']}->{r['new_full']} | {ms(r['old_mean_single'])}->{ms(r['new_mean_single'])} | "
                     f"{r['fvu_mix']:.4f} | {r['act_rank']:.1f} | {r['dead']} |")
    (OUT_DIR / "reeval_comparison.md").write_text("\n".join(lines) + "\n")
    print(f"\n[reeval] DONE. Wrote {OUT_DIR/'reeval_comparison.md'} and .json", flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
