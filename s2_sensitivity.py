"""S2.4 — block-length sensitivity of the f*(w_crash) threshold.

Runs ``s1.py`` four times with the PR+Kou engine at mean_block_length ∈
{2, 4, 8, 16}, collects the f*(w_crash) summary from each JSON, and
reports the stability of the interior-band threshold.

Acceptance (brief §S2.4): the threshold (w_crash where f* first leaves the
low boundary) is stable to within ±0.05 across these mean block lengths.
If it isn't, the bootstrap is the wrong primitive for this horizon and we
revisit. Either way, REPORT honestly — do not tune.

Usage:
    uv run python s2_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Reuse the s1 entry point so the pipeline definition lives in one place.
from s1 import RESULTS_DIR, main as s1_main

BLOCK_LENGTHS = (2.0, 4.0, 8.0, 16.0)
ENGINE = "politis_romano_kou"
ACCEPTANCE_BAND = 0.05  # ±0.05 in w_crash per brief §S2.4


def run_one(bl: float) -> dict:
    """Run s1.py at one mean_block_length, return the loaded JSON payload."""
    out_name = f"sens_blk{int(bl) if float(bl).is_integer() else bl}"
    argv = [
        "--engine", ENGINE,
        "--mean-block-length", str(bl),
        "--out", out_name,
    ]
    print(f"\n────── mean_block_length = {bl}  ({out_name}) ──────")
    rc = s1_main(argv)
    if rc != 0:
        raise RuntimeError(f"s1.main failed (rc={rc}) at bl={bl}")
    path = RESULTS_DIR / f"{out_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[S2.4] block-length sensitivity of f*(w_crash) threshold")
    print(f"       engine={ENGINE}  block_lengths={BLOCK_LENGTHS}")
    print(f"       acceptance: max−min(threshold) ≤ ±{ACCEPTANCE_BAND}")

    payloads = {bl: run_one(bl) for bl in BLOCK_LENGTHS}

    print("\n────── SENSITIVITY SUMMARY ──────")
    print(f"  {'block':>6} | {'threshold':>10} | {'interior_band':>16} | {'verdict':>10}")
    rows = []
    thresholds = []
    for bl, payload in payloads.items():
        fs = payload["f_star_summary"]
        thr = fs["threshold_leaves_low_boundary"]
        band = fs["interior_band"]
        v = payload["verdict"]["verdict"]
        thresholds.append(thr if thr is not None else float("nan"))
        band_str = f"[{band[0]:.3f},{band[1]:.3f}]" if band else "NONE"
        thr_str = f"{thr:.3f}" if thr is not None else "N/A"
        print(f"  {bl:>6.1f} | {thr_str:>10} | {band_str:>16} | {v:>10}")
        rows.append({
            "block_length": bl,
            "threshold": thr,
            "interior_band": band,
            "verdict": v,
            "f_star_low_weight": fs["f_star_low_weight"],
            "f_star_high_weight": fs["f_star_high_weight"],
        })

    finite = [t for t in thresholds if t is not None and t == t]
    if not finite:
        print("\n[VERDICT] no finite thresholds — interior band absent at all "
              "block lengths. Honest result, not a tuning failure.")
        stable = False
        spread = None
    else:
        spread = float(max(finite) - min(finite))
        stable = spread <= ACCEPTANCE_BAND
        print(f"\n[VERDICT] threshold spread = {spread:.4f}  "
              f"(acceptance ≤ {ACCEPTANCE_BAND}) → "
              f"{'STABLE ✓' if stable else 'UNSTABLE ✗'}")
        if not stable:
            print("          unstable means: bootstrap may be wrong primitive "
                  "for this horizon; revisit per brief §S2.4 acceptance.")

    out_path = RESULTS_DIR / "s2_block_sensitivity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "engine": ENGINE,
                "block_lengths": list(BLOCK_LENGTHS),
                "acceptance_band": ACCEPTANCE_BAND,
            },
            "rows": rows,
            "summary": {
                "thresholds": thresholds,
                "spread": spread,
                "stable": stable,
            },
        }, f, indent=2)
    print(f"[saved] {out_path}")
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
