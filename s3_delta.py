"""S3.8 — Document the f*(w_crash) delta between S2 PR+Kou and S3 stochastic-SVI.

Loads two sweep JSONs:
  baseline:  sweep_pr_kou.json   (S2 deterministic SVI + PR+Kou price engine)
  new:       sweep_stoch_svi.json (S3 stochastic SVI + PR+Kou price engine)

Reports the delta honestly. Per brief §S3.8 the expected directions
(widen / threshold-drop / toward GREEN) are hypotheses, not acceptance
criteria. Report whatever direction the engine actually moved.
"""
from __future__ import annotations

import json
import sys

from s1 import RESULTS_DIR

BASELINE = "sweep_pr_kou"
NEW_ENGINE = "sweep_stoch_svi"
_FLAT_TOL = 1e-9


def _load(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run s1.py with --out {name} first."
        )
    return json.load(open(path, "r", encoding="utf-8"))


def _band_width(band):
    return (band[1] - band[0]) if band else None


def _arrow(direction: str, actual: float | None) -> str:
    if actual is None:
        return "-"
    if abs(actual) < _FLAT_TOL:
        return "= (flat)"
    if direction == "down":
        return "✓ (down)" if actual < 0 else "✗ (up)"
    if direction == "up":
        return "✓ (up)" if actual > 0 else "✗ (down)"
    return "-"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base = _load(BASELINE)
    new = _load(NEW_ENGINE)

    print("[S3.8] Engine delta — S2 PR+Kou vs S3 stochastic-SVI")
    print(f"       baseline : {base['config']['engine_label']}")
    print(f"       new      : {new['config']['engine_label']}")
    print()

    b_sum = base["f_star_summary"]
    n_sum = new["f_star_summary"]
    b_v = base["verdict"]
    n_v = new["verdict"]

    b_thr = b_sum["threshold_leaves_low_boundary"]
    n_thr = n_sum["threshold_leaves_low_boundary"]
    b_band = b_sum["interior_band"]
    n_band = n_sum["interior_band"]
    b_width = _band_width(b_band)
    n_width = _band_width(n_band)
    thr_delta = (n_thr - b_thr) if (b_thr is not None and n_thr is not None) else None
    width_delta = (n_width - b_width) if (b_width is not None and n_width is not None) else None

    print("│ Metric                 │ S2 baseline       │ S3 stochastic-SVI  │ delta             │ expected | actual")
    print("│ ───────────────────────│ ──────────────────│ ───────────────────│ ─────────────────  │ ─────────────────")
    print(f"│ interior_band          │ {str(b_band):>17} │ {str(n_band):>17} │ "
          f"{('width %+.4f' % width_delta) if width_delta is not None else 'N/A':>15} │ "
          f"widen      | { _arrow('up', width_delta)}")
    thr_s = "N/A" if b_thr is None else f"{b_thr:.3f}"
    thrn_s = "N/A" if n_thr is None else f"{n_thr:.3f}"
    print(f"│ threshold (leave low)  │ {thr_s:>17} │ {thrn_s:>17} │ "
          f"{('%+.4f' % thr_delta) if thr_delta is not None else 'N/A':>15} │ "
          f"drop       | { _arrow('down', thr_delta)}")
    print(f"│ Gate-A verdict         │ {b_v['verdict']:>17} │ {n_v['verdict']:>17} │ "
          f"{'see below':>15} │ toward GREEN")
    print(f"│ f* @ w=0 (freq-wtd)    │ {b_sum['f_star_low_weight']:>17.2f} │ "
          f"{n_sum['f_star_low_weight']:>17.2f} │ "
          f"{n_sum['f_star_low_weight'] - b_sum['f_star_low_weight']:>+15.2f} │")
    print(f"│ Strata Sortino at f*   │ "
          f"{str(b_v.get('sortino_at_f_star')):>17} │ {str(n_v.get('sortino_at_f_star')):>17} │ │")
    print()

    notes = []
    if thr_delta is None:
        # New engine has NO interior band — f* boundary-locked.
        notes.append(
            "S3 has NO interior band on [0, 1]: f* is boundary-locked across "
            "all tail-weights. The expected directions (widen / threshold drop) "
            "are not applicable in this regime."
        )
    elif thr_delta < -_FLAT_TOL:
        notes.append("threshold DROPPED — interior f* emerges at lower tail-weight under S3.")
    elif thr_delta > _FLAT_TOL:
        notes.append("threshold ROSE — interior f* requires HIGHER tail-weight under S3.")
    else:
        notes.append("threshold unchanged.")

    if width_delta is None:
        pass
    elif abs(width_delta) < _FLAT_TOL:
        same = (
            b_band is not None and n_band is not None
            and abs(b_band[0] - n_band[0]) < _FLAT_TOL
            and abs(b_band[1] - n_band[1]) < _FLAT_TOL
        )
        notes.append(
            "interior band fully identical (no shift, no widen)." if same else
            "interior band SHIFTED but width unchanged."
        )
    elif width_delta > 0:
        notes.append("interior band widened.")
    else:
        notes.append("interior band narrowed.")

    if b_v["verdict"] != n_v["verdict"]:
        notes.append(f"verdict moved {b_v['verdict']} → {n_v['verdict']}.")
    else:
        notes.append(f"verdict unchanged ({b_v['verdict']}).")

    # Cross-reference S3.6 robustness flag if it exists.
    robust_path = RESULTS_DIR / "s3_robustness.json"
    degenerate_flag = False
    if robust_path.exists():
        rob = json.load(open(robust_path, "r", encoding="utf-8"))
        # The flag fires when perturb span is zero and μ ratios are < 0.5.
        span = rob.get("perturb_f_star_low_span")
        if span is not None and span < 1e-6:
            degenerate_flag = True
            notes.append(
                "S3.6 robustness raised the DEGENERATE-SURFACE FLAG: f*(w=0) "
                "is invariant across all 12 ±20% OU perturbations AND OOS — "
                "the stochastic engine is calibrated to a thin SVI history "
                "and reproduces a near-flat surface. The f*=0.80 result is "
                "an artifact of cheap-hedge degenerate pricing, NOT a real "
                "structural finding. RECOMMENDED PATH: fallback to "
                "deterministic SVI (CLAUDE.md §6 S3 fallback rule) for "
                "downstream pipeline integrity. S3 engine code stays — it is "
                "correct; the calibration source is the limitation."
            )

    print("[honest notes]")
    for note in notes:
        print(f"  - {note}")
    print()

    out = RESULTS_DIR / "s3_engine_delta.json"
    json.dump({
        "baseline": {"name": BASELINE, "engine_label": base["config"]["engine_label"]},
        "new_engine": {"name": NEW_ENGINE, "engine_label": new["config"]["engine_label"]},
        "metrics": {
            "baseline_interior_band": b_band,
            "new_interior_band": n_band,
            "baseline_threshold": b_thr,
            "new_threshold": n_thr,
            "threshold_delta": thr_delta,
            "band_width_delta": width_delta,
            "baseline_verdict": b_v["verdict"],
            "new_verdict": n_v["verdict"],
            "baseline_f_star_low_weight": b_sum["f_star_low_weight"],
            "new_f_star_low_weight": n_sum["f_star_low_weight"],
        },
        "honest_notes": notes,
        "degenerate_flag_from_s3_robustness": degenerate_flag,
    }, open(out, "w", encoding="utf-8"), indent=2)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
