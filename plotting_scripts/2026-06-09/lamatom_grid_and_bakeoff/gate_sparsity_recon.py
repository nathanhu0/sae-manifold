"""Sparsity vs reconstruction for THE GATE QUESTION ONLY: is the sigmoid-surrogate backward
better than the default rect (Heaviside + rectangular STE)? All arms share the floor recipe
(lambda 0.003, pool 8:64 learn-rank, L0=4). Color = gate backward; filled = revival losses on,
open = revival off (eps annotated on the no-revival sigmoid arms).
X = measured sparsity (atoms firing/sample when available, else active rank-dof/sample).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
RUNS = Path("/nlp/scr/nathu/sae-manifold/runs")
ARMS = [  # (label, dir, gate, revival)
    ("rect + revival (default)", "2026-06-09_paper48_floor/k8_lam0.003_rev1x", "rect", True),
    ("sigmoid + revival", "2026-06-09_paper48_gate_bakeoff/sigmoid_revival_seed0", "sigmoid", True),
    ("rect, no revival", "2026-06-09_paper48_gate_bakeoff/rect_norevival_seed0", "rect", False),
    ("sigmoid, no revival (eps 0.5)", "2026-06-09_paper48_gate_bakeoff/sigmoid_norevival_eps0.5_seed0", "sigmoid", False),
    ("sigmoid, no revival (eps 1)", "2026-06-09_paper48_gate_bakeoff/sigmoid_norevival_eps1.0_seed0", "sigmoid", False),
    ("sigmoid, no revival (eps 2)", "2026-06-09_paper48_gate_bakeoff/sigmoid_norevival_seed0", "sigmoid", False),
    ("sigmoid, no revival (eps 4)", "2026-06-09_paper48_gate_bakeoff/sigmoid_norevival_eps4.0_seed0", "sigmoid", False),
]
COLOR = {"rect": "C0", "sigmoid": "C1"}

fig, ax = plt.subplots(figsize=(7.5, 5.2))
use_atoms = True
pts = []
for label, rel, gate, revival in ARMS:
    d = RUNS / rel
    try:
        m = json.load(open(d / "metrics.json"))
    except Exception:
        m = json.load(open(d / "metrics_pre_retrofit.json"))
    x = m.get("atoms_per_sample_mean")
    if x is None:
        x, use_atoms = m["act_rank"], False
    pts.append((label, x, m["fvu"], gate, revival, m["act_rank"]))
if not use_atoms:        # fall back uniformly to act_rank so the axis is consistent
    pts = [(l, ar, f, g, r, ar) for (l, x, f, g, r, ar) in pts]
for label, x, fvu, gate, revival, _ in pts:
    ax.scatter(x, fvu, s=70, color=COLOR[gate], marker="o",
               facecolors=COLOR[gate] if revival else "none", linewidths=1.6, zorder=3)
    ax.annotate(label, (x, fvu), fontsize=7.5, xytext=(6, -2), textcoords="offset points")
ax.set_xlabel("atoms firing / sample (measured)" if use_atoms else "active rank-dof / sample (measured)")
if use_atoms:
    ax.axvline(4.0, color="0.75", ls="--", lw=1)
    ax.text(4.05, ax.get_ylim()[1] * 0.9, "data L0 = 4", fontsize=8, color="0.4")
ax.set_ylabel("mixture FVU (log)"); ax.set_yscale("log")
ax.set_title("Gate backward: sigmoid surrogate vs default rect STE\n(filled = revival on, open = revival off; floor recipe)")
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT_DIR / "gate_sparsity_recon.png", dpi=150)
print("wrote gate_sparsity_recon.png  (x =", "atoms/sample" if use_atoms else "act_rank", ")")
