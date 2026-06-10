"""Smoke test + documentation of the manifold-SAE evaluation suite (48-manifold, L0=4 setting).

Runs the full capture suite on the canonical floor-winner checkpoint and asserts every metric is
present and internally consistent. Doubles as the spec for "what the eval suite reports."

Run: PYTHONPATH=. python eval_smoke_test.py
"""
import math
import tempfile

import torch

from manifold_ae.eval_and_viz import compute_capture, load_checkpoint

CKPT = "/nlp/scr/nathu/sae-manifold/runs/2026-06-09_paper48_floor/k8_lam0.003_rev1x/ckpt.pt"

# --- THE EVAL SUITE: every metric compute_capture/eval_and_viz reports, grouped ---------------------
SUITE = {
    "deployment (real L0-mixtures)": ["fvu", "act_rank", "active_count_mean", "dead_atoms",
                                      "inmix_captured_single", "inmix_mean_fvu"],
    "isolated capture (the 2 standards)": ["captured_single", "captured_tiled", "tiled_charts_mean",
                                           "tiled_charts_hist", "mean_single_fvu", "mean_cond_fvu"],
    "diagnostics": ["captured_full", "captures_by_thresh", "per_family", "eval_mode", "presence",
                    "eval_n", "eval_n_iso", "thresh"],
}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, zoo, scale, l0, ck = load_checkpoint(CKPT, dev)
    res, _ = compute_capture(model, zoo, scale, l0=l0, want_tri=True)
    N = res["n_inst"]

    print("=== eval suite on floor winner (k8_lam0.003_rev1x) ===")
    for group, keys in SUITE.items():
        print(f"\n[{group}]")
        for k in keys:
            assert k in res, f"MISSING metric: {k}"
            v = res[k]
            print(f"  {k:24s} = {v if not isinstance(v, float) else round(v, 4)}")

    th, loose = res["thresh"], res["captures_by_thresh"]
    checks = [
        ("n_inst == 48", N == 48),
        ("0 <= single <= tiled <= 48", 0 <= res["captured_single"] <= res["captured_tiled"] <= N),
        ("tiled <= full (clean-atlas subset of union)", res["captured_tiled"] <= res["captured_full"]),
        ("single @0.10 >= single @0.05", loose[f"{2*th:.3f}"]["single"] >= loose[f"{th:.3f}"]["single"]),
        ("in-mixture single in [0,48]", 0 <= res["inmix_captured_single"] <= N),
        ("in-mixture ~ isolated (|diff|<=8)", abs(res["inmix_captured_single"] - res["captured_single"]) <= 8),
        ("no NaN in headline FVUs", not any(math.isnan(res[k]) for k in ["fvu", "mean_single_fvu"])),
        ("act_rank > 0", res["act_rank"] > 0),
        ("per_family covers 8 types", len(res["per_family"]) == 8),
    ]
    print("\n=== assertions ===")
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    # also confirm the file-writing path runs end to end
    with tempfile.TemporaryDirectory() as td:
        from manifold_ae.eval_and_viz import eval_and_viz
        eval_and_viz(model, zoo, scale, td, l0=l0)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    assert ok, "smoke test failed"


if __name__ == "__main__":
    main()
