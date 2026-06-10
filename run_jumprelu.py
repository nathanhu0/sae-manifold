"""Back-compat shim: the trainer is now train_manifold_sae.py. The JumpReLU name was stale --
the gate is a pure binary presence gate (Heaviside at 0, no learned theta, no magnitude
pass-through); only the STE backward descends from JumpReLU, and --gate-grad sigmoid replaces
even that. Kept so older command lines keep working while the 2026-06-09 fleet lands; delete after.
"""
from train_manifold_sae import main

if __name__ == "__main__":
    main()
