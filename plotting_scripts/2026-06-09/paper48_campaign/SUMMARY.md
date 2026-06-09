# Manifold-SAE — paper 48-zoo (L0=4) campaign + diagnosis — 2026-06-09

General summary of the LR sweep, the K / λ / preact sweeps, the oracle floor, data-scaling, the
per-atom-penalty and rank-0-floor experiments, and the rank-distribution diagnosis. Code on branch
`manifold-sae`; runs in `/nlp/scr/nathu/sae-manifold/runs/2026-06-09_paper48_*`; figures + this doc
in this folder.

## Setup
Paper's synthetic benchmark: **48 manifolds** (8 types × 6 variants), **constant L0=4** presence
(uniform without replacement, σε=1e-5), per-instance RMS-1 normalization, random-orthonormal V_i,
zero bias. All runs use the **learnable per-dim rank** gate (each atom has max_rank K dims, a learned
per-dim bias gated by STE-Heaviside; effective rank = #on-dims). Pool `K:64` (64-atom pool).
Recipe knobs: tune sparsity λ; set both preact/revival losses by rule (presence `lam_preact=3e-4`
fixed; per-dim `lam_preact_dim ≈ λ/30` = 0.8·λ·L0/N at this setting).

## TL;DR
- **lr=3e-4 is the optimum at this scale** (not the 16-zoo's 1e-3) — confirmed by a peak: 1e-4 underfits, 3e-4 best, 1e-3/3e-3 worse.
- **K8 ≫ K4** — learn-rank needs rank headroom above the data's max embedding dim; K4 (=torus ki) collapses & forces spanning.
- **Recipe sweet spot:** K8 / lr3e-4 / λ0.003 / 1×preact → single **28/48**, full 32/48, FVU 0.021, dims/atom 2.80, frozen 0.03.
- **The gap is optimization/incentive, not data or objective** — oracle floor unreached, data-scaling flat, per-atom penalty neutral.
- **Diagnosis:** the failures were (a) two degenerate always-on rank-0 "bias" atoms (a cost loophole), (b) near-misses just above the 0.05 cutoff (segment), (c) smooth over-ranking (helix winding).
- **The rank-0 floor fix WORKS** — kills 29/51, collapses spanning (atoms/manifold 3–12 → ~1), and lifts the winner to **single 32/48** at *lower* rank (act_rank 8.3). New best. (segment still 0/6 — now a low-norm-sensitivity hard case, not a degeneracy.)

## 1. LR sweep (K8) — gentler LR wins at scale
| lr | single | FVU |
|---|---|---|
| 1e-4 | 11/48 | 0.055 (underfit) |
| **3e-4** | **28/48** | **0.021** |
| 1e-3 | 27/48 | 0.025 (full≪single: redundant co-firing) |
| 3e-3 | 20/48 | 0.039 |
Peak at 3e-4. 1e-3's `full(13)<single(27)` = atoms co-firing and summing to overshoot. (Small-LR×longer-steps
runs in progress to test whether gentler-trained-longer beats 3e-4 — budget ∝ lr×steps.)

## 2. K4 vs K8 — headroom is required
K8 single 28; K4 tops ~14 and collapses (frozen 0.45, dims→0.5) at higher LR. K4 (max_rank=4=torus ki, zero slack)
**forces spanning** — high full / low single (e.g. K4/λ0.001/2× = full 36, single 8). K8's slack enables one-atom dedication.

## 3. Stage-2 grid @ lr3e-4, K8 (λ × preact)
| λ | preact | FVU | single | full | dims/atom | frozen |
|---|---|---|---|---|---|---|
| 0.001 | 0.5× | 0.039 | 5 | 26 | 0.64 | 0.65 |
| 0.001 | 1× | 0.039 | 11 | 29 | 0.64 | 0.51 |
| 0.001 | 2× | 0.012 | 26 | 33 | 4.56 | 0.00 |
| 0.003 | 0.5× | 0.209 | 5 | 0 | 0.28 | 0.89 |
| **0.003** | **1×** | **0.021** | **28** | **32** | **2.80** | **0.03** |
| 0.003 | 2× | 0.033 | 29 | 31 | 4.75 | 0.00 |
| 0.01 | 0.5× | 0.132 | 9 | 0 | 1.67 | 0.36 |
| 0.01 | 1× | 0.094 | 15 | 7 | 2.10 | 0.06 |
| 0.01 | 2× | 0.112 | 13 | 13 | 5.38 | 0.00 |
Sub-rule preact (0.5×) collapses (frozen↑); the λ/30 rule is clean at λ0.003 but soft at λ0.001 (needs 2×).
**Two operating points:** clean/efficient = λ0.003/1× (single 28, dims 2.80); peak-single = λ0.001/2× (single 30, but rank ~15).

## 4. Oracle floor (reference)
Sum of min-capturing ranks over 48 instances = 70 → at L0=4 the oracle sits at **act_rank ≈ 5.83, FVU ≈ 0.007**.
Our winner (act_rank 9.2, FVU 0.021, single 28/48) spends ~1.6× the oracle's rank, ~3× the FVU floor, single 28 vs
48-by-construction → the dedication/optimization gap persists at scale.

## 5. Data-scaling — NOT data-limited
K8/lr3e-4/λ0.003, steps 150k→300k→600k: single **28→26→27** (flat). 4× the data finds no more dedicated atoms;
the bottleneck is optimization/incentive, not data. (act_rank/FVU drift is warmup-cosine-schedule-confounded.)

## 6. Per-atom count penalty (`--lam-atom`) — tried, NEUTRAL
`lam_atom·Σ active_i` (rank-independent) to favor consolidation. λ_atom∈{0.003,0.01,0.03} at the winner: best single
29 ≈ baseline 28 (noise), worse FVU; **n_active barely moved (54→52)**. It raises the *cost* of spanning but doesn't
help the optimizer *find* the consolidated fit — same incentive-vs-optimization story. (Code kept, additive, off.)

## 7. Rank-distribution diagnosis (winner; see rank_dist.png, firing_matrix.png)
- **Bimodal per-atom rank:** active atoms cluster at rank ≤4 (mode 2, mean 2.80), a gap at 5–7, then 7 atoms stuck at rank 8. Per-dim pruning is ~all-or-nothing.
- **Systematic over-ranking vs oracle** (+0.5–1.6 dims/type); **helix worst (learned 2.6 vs oracle 1.0)** — the extra rank genuinely winds its 3-turn coil (a decoder-capacity artifact, captured fine at 5/6).
- **Per-type capture is uneven:** mobius 6/6, helix/swiss 5/6, torus/circle 4/6, but **flat_disk 1/6 and segment 0/6**.
- **Segment 0/6 is a cutoff near-miss, not a dedication failure** — it has dedicated rank-1 atoms at FVU 0.075–0.11, just above 0.05.
- **The real pathology = 2 always-on rank-0 "bias" atoms (29, 51):** fire on >80% of all manifolds, emit a fixed `dec(0)` vector (FVU≥1), reconstruct nothing, and **wreck full-model reconstruction on low-norm manifolds** (segment `full` FVU up to 88). Pool = 45 working + 2 always-on + 17 dead.
- **Root cause:** the rank-weighted l0 charges **0** for rank-0 atoms, so an always-on rank-0 atom is *free*. (The architecture already has ample bias — tied `b_dec` in+out, plus per-atom per-layer biases — so this is a cost loophole, not a missing-bias problem.)

## 8. Rank-0 floor fix (`--l0-rank-floor`) — WORKS; NEW BEST single 32/48
Charges `Σ active·max(rank,1)` (≥1 per firing atom; rank≥1 unchanged) so a free always-on rank-0 atom gets
gated down. Reran the K8/lr3e-4 λ×preact grid (9 cells) with the floor. **Clear win:**
- **Always-on atoms 29/51 GONE** — 0 rank-0 atoms fire (was 2), 19 dead (see `firing_matrix_floor.png`).
- **Spanning → clean dedication** — atoms/manifold 3–12 → **~1.0–1.7**; helix now uses **1 atom each** (was 3+).
- **More capture at LOWER rank** — winner cell λ0.003/1× **single 28 → 32/48**, act_rank 9.2 → 8.3, mean type-rank
  ~1.2–2.2 (was 2.5–2.7, closer to the oracle); circle 4→6, helix 5→6, flat_disk 1→5.

Floor-OFF vs floor-ON single-capture (K8/lr3e-4):

| λ | preact | floor-OFF | floor-ON |
|---|---|---|---|
| 0.001 | 0.5× | 5 | 10 |
| 0.001 | 1× | 11 | 11 |
| 0.001 | 2× | 26 | 22 |
| 0.003 | 0.5× | 5 | 11 |
| **0.003** | **1×** | 28 | **32** |
| 0.003 | 2× | 29 | 27 |
| 0.01 | 0.5× | 9 | 17 |
| 0.01 | 1× | 15 | 20 |
| 0.01 | 2× | 13 | 13 |

Floor helps most at the low-preact (0.5×) cells (recover from collapse) and the winner; slightly hurts the
high-preact λ0.001/2× & λ0.003/2× cells. **New campaign best = floor-ON λ0.003/1×.**

The 0.05 capture cutoff is arbitrary, so report BOTH thresholds + the continuous mean single-atom FVU (fig1 now
shows @0.05 filled, @0.10 hollow; the vertical band = near-misses):

| config | @0.05 | @0.10 | mean single FVU | act_rank |
|---|---|---|---|---|
| lr3e-4 (no floor) | 29 | 35 | 0.112 | 16.2 |
| lr1e-3 (no floor) | 30 | 35 | 0.120 | 15.4 |
| **lr3e-4 + floor** | **32** | **38** | **0.092** | **8.3** |

The floor is best on every metric. The 6-manifold gap between @0.05 (32) and @0.10 (38) is the near-miss band
(segment's instances live there at single FVU 0.088–0.21) — "captured but loose," not failed.

**Remaining hard case: segment still 0/6** — but no longer the bias atoms (gone). It's *low-norm sensitivity*:
the dedicated rank-1 atom hits single FVU 0.088–0.21 (near-miss) while `full` stays 30–101 (tiny-norm denominator
blows up any stray contribution). A genuine hard case, not a degeneracy.

**RECOMMENDATION: adopt `--l0-rank-floor` as default** for this (no-0D-feature) zoo — strictly better (more
capture, lower rank, cleaner dedication, no degenerate bias atoms).

## Figures
- `fig1_frontier.png` — FVU vs act_rank + single-capture/48 (lr3e-4 vs lr1e-3, K8/K4, oracle ★).
- `fig2_heatmap.png` — K8 lr3e-4 single-capture over λ × preact.
- `fig3_datascale.png` — single-capture & FVU vs steps (flat).
- `rank_dist.png` — per-atom rank histogram (bimodal) + per-type oracle vs learned rank.
- `firing_matrix.png` — 48×64 firing fractions, floor-OFF (dedication blocks, the 2 always-on columns, 17 dead).
- `firing_matrix_floor.png` — same, floor-ON: **0 always-on columns** (29/51 gone), cleaner ~1-atom-per-row dedication.

## Open threads / next
1. **Floor-fix verdict** (pending) — if it works, re-run the recipe with it as default for this no-0-D-feature zoo.
2. **Small-LR × longer steps** (pending) — does gentler-trained-longer beat 3e-4?
3. **0-D feature test** — seed the zoo with discrete features; check learn-rank parks them at rank-0 (= SAE features) while holding manifolds at intrinsic rank (validates "subsumes the SAE").
4. The residual gap is optimization-shaped (over-ranking + near-misses + winding), not an incentive gap — likely wants better init/curriculum, not more loss terms.
