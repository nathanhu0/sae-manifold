"""Oracle: fit a DEDICATED atom to EACH instance of the zoo (no mixture, no sparsity).

For every instance in the zoo (variants_per_type variants/type) and each rank in
{1,2,3,4}, train a single manifold atom (the same funnel as the mixture SAE) directly
on that instance's isolated samples, gate-free, MSE. This is the CEILING for
per-manifold capture: the best a correctly-sized dedicated atom can do, with no
allocation / sharing / sparsity competition.

Per-INSTANCE (not per-type): the simplified zoo has variants_per_type variants/type,
each with its OWN random ambient embedding V_i, so a dedicated atom must be fit to the
specific instance. These atoms are exactly what the warm-start experiment loads into the
pool, and assembling them measures the true achievable (act_rank, FVU) on this dataset.

Writes oracle_table.md + metrics.json + per-(instance,rank) checkpoints.

Run (jag): ebatch oracle_simpl slconf/slconf40s \
  "PYTHONUNBUFFERED=1 PYTHONPATH=. python run_oracle.py --out-dir <d>"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manifold_ae.manifold_zoo import ManifoldZoo
from manifold_ae.manifold_sae import ManifoldSAE

D, STEPS, LR, BATCH = 256, 4000, 1e-3, 2048
RANKS = [1, 2, 3, 4]
ENC_DIMS = (128, 64, 32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def recon(m, x):
    z, g = m.encode(x - m.b_dec)
    return m.decode_all(z)[:, 0] + m.b_dec          # single atom, gate-free


def fit(inst, r, rng, steps):
    torch.manual_seed(0)
    m = ManifoldSAE(d_model=D, rank_dist={r: 1}, enc_dims=ENC_DIMS).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    # normalize to per-element RMS ~1: a unit-norm vector spread over D dims has
    # per-element magnitude ~1/sqrt(D); at that scale gradients fall below Adam's
    # eps and training stalls. (The mixture trainer divides by `scale` for this.)
    _, a0 = inst.sample_full(4096, np.random.default_rng(5))
    scale = float(np.sqrt((a0 ** 2).mean()))
    for _ in range(steps):
        _, amb = inst.sample_full(BATCH, rng)
        xt = torch.tensor(amb, device=DEV) / scale
        loss = ((recon(m, xt) - xt) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    _, amb = inst.sample_full(4096, rng)
    xt = torch.tensor(amb, device=DEV) / scale
    with torch.no_grad():
        fvu = (((recon(m, xt) - xt) ** 2).sum() / (xt ** 2).sum()).item()
    return m, fvu, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-per-type", type=int, default=2)   # 2 -> 16 instances
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out-dir", default="/nlp/scr/nathu/sae-manifold/runs/2026-06-09_oracle_simplified")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    zoo = ManifoldZoo(d=D, seed=0, variants_per_type=a.variants_per_type)
    rng = np.random.default_rng(1)
    table = {}
    for inst in zoo.instances:
        table[inst.name] = dict(type=inst.type, di=int(inst.di), ki=int(inst.ki), fvu={}, scale={})
        for r in RANKS:
            m, fvu, scale = fit(inst, r, rng, a.steps)
            torch.save({"state_dict": m.state_dict(), "rank": r, "name": inst.name,
                        "type": inst.type, "scale": scale, "enc_dims": ENC_DIMS},
                       out / f"{inst.name}_r{r}.pt")
            table[inst.name]["fvu"][r] = fvu
            table[inst.name]["scale"][r] = scale
            print(f"  {inst.name:13s} rank={r} (d{inst.di} k{inst.ki})  FVU={fvu:.4f}", flush=True)
        best_r = min(RANKS, key=lambda r: table[inst.name]["fvu"][r])
        # minimal capturing rank: smallest rank with FVU < 0.05
        cap_r = next((r for r in RANKS if table[inst.name]["fvu"][r] < 0.05), best_r)
        table[inst.name]["best_rank"] = best_r
        table[inst.name]["min_capturing_rank"] = cap_r
        print(f"[{inst.name}] best rank={best_r} FVU={table[inst.name]['fvu'][best_r]:.4f} "
              f"min-capturing={cap_r}", flush=True)

    json.dump(table, open(out / "metrics.json", "w"), indent=2)
    lines = ["# Oracle (per-instance): dedicated atom per zoo instance, FVU by rank\n",
             "A single funnel atom trained directly on each instance, gate-free. Best-possible",
             "per-manifold capture (no mixture/sharing/sparsity). min-cap = smallest rank with FVU<0.05.\n",
             "| instance | d_i | k_i | rank1 | rank2 | rank3 | rank4 | best | min-cap |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, t in table.items():
        f = t["fvu"]
        cells = " | ".join(f"{f[r]:.3f}" + ("*" if r == t["ki"] else "") for r in RANKS)
        br = t["best_rank"]
        lines.append(f"| {name} | {t['di']} | {t['ki']} | {cells} | r{br}={f[br]:.3f} | r{t['min_capturing_rank']} |")
    tot_min_rank = sum(t["min_capturing_rank"] for t in table.values())
    lines.append(f"\n(* = embedding dim k_i)")
    lines.append(f"\nSum of min-capturing ranks over all {len(table)} instances = {tot_min_rank} "
                 f"-> assembled oracle expected act_rank at p_active=0.25 = {0.25 * tot_min_rank:.2f}")
    (out / "oracle_table.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\n[done] oracle_table.md + metrics.json + {len(table) * len(RANKS)} atoms -> {out}", flush=True)


if __name__ == "__main__":
    main()
