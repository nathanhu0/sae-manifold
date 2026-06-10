"""Simplest gate A/B: vanilla rect (Heaviside + rectangular STE, the floor winner) vs vanilla
sigmoid-surrogate gate -- same recipe, revival on, default jump_eps. Reconstruction (train MSE)
over training, parsed from the slurm step logs; final mixture-FVU eval as end markers.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
LOGS = {"rect STE (current)": ("/nlp/scr/nathu/slurm/15838692.out", 0.0249, "C0"),
        "sigmoid surrogate": ("/nlp/scr/nathu/slurm/15842386.out", 0.0367, "C1")}

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for label, (path, final_fvu, c) in LOGS.items():
    steps, recon = [], []
    for line in open(path):
        m = re.match(r"step\s+(\d+) recon=([0-9.]+)", line)
        if m:
            steps.append(int(m.group(1))); recon.append(float(m.group(2)))
    ax.plot(steps, recon, color=c, lw=1.4, label=f"{label} — final eval FVU {final_fvu:.3f}")
    ax.scatter([steps[-1]], [recon[-1]], color=c, s=40, zorder=3)
ax.set_yscale("log")
ax.set_xlabel("training step"); ax.set_ylabel("train reconstruction MSE (log)")
ax.set_title("Gate backward A/B (vanilla, revival on, eps 2.0): rect STE vs sigmoid surrogate")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT_DIR / "gate_ab_recon.png", dpi=150)
print("wrote", OUT_DIR / "gate_ab_recon.png")
