"""S4.8 — Document the f*(w_crash) delta between S2 PR+Kou and S4 full model.

Loads:
  baseline:  sweep_pr_kou.json   (S2 — deterministic SVI + PR+Kou,
                                   single-cycle, single-strike hedge)
  new:       s4_sweep.json       (S4 — deterministic SVI + PR+Kou,
                                   multi-cycle, DN ladder, bucket)

Per brief §S4.8: REPORT HONESTLY whichever direction the delta moves.
The expected directions (widen / threshold-drop) are hypotheses; the
brief explicitly accepts unchanged / narrowed / no-interior as the
honest outcome if that's what S4 produces.
"""
from __future__ import annotations

import json
import sys

from s1 import RESULTS_DIR

BASELINE = "sweep_pr_kou"
NEW_ENGINE = "s4_sweep"
_FLAT_TOL = 1e-9


def _load(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run the corresponding orchestrator first."
        )
    return json.load(open(path, "r", encoding="utf-8"))


def _band_width(band):
    return (band[1] - band[0]) if band else None


def _arrow(direction, actual):
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

    print("[S4.8] Engine delta — S2 PR+Kou (single-cycle, single-strike) vs "
          "S4 full model (multi-cycle, DN ladder, bucket)")
    print(f"       baseline : {base['config']['engine_label']}")
    print(f"       new      : {new['config']['engine_label']}")
    print()

    b_sum = base["f_star_summary"]
    n_sum = new.get("f_star_summary")

    b_thr = b_sum["threshold_leaves_low_boundary"]
    n_thr = n_sum["threshold_leaves_low_boundary"] if n_sum else None
    b_band = b_sum["interior_band"]
    n_band = n_sum["interior_band"] if n_sum else None
    b_width = _band_width(b_band)
    n_width = _band_width(n_band)
    thr_delta = (n_thr - b_thr) if (b_thr is not None and n_thr is not None) else None
    width_delta = (n_width - b_width) if (b_width is not None and n_width is not None) else None

    print("│ Metric                 │ S2 baseline       │ S4 full model     │ delta             │ expected | actual")
    print("│ ───────────────────────│ ──────────────────│ ───────────────── │ ─────────────────  │ ─────────────────")
    print(f"│ interior_band          │ {str(b_band):>17} │ {str(n_band):>17} │ "
          f"{('width %+.4f' % width_delta) if width_delta is not None else 'N/A':>17} │ "
          f"widen      | { _arrow('up', width_delta)}")
    thr_s = "N/A" if b_thr is None else f"{b_thr:.3f}"
    thrn_s = "N/A" if n_thr is None else f"{n_thr:.3f}"
    print(f"│ threshold (leave low)  │ {thr_s:>17} │ {thrn_s:>17} │ "
          f"{('%+.4f' % thr_delta) if thr_delta is not None else 'N/A':>17} │ "
          f"drop       | { _arrow('down', thr_delta)}")
    b_v = base["verdict"]["verdict"]
    n_v = new["verdict"]["verdict"]
    print(f"│ Gate-A verdict         │ {b_v:>17} │ {n_v:>17} │ "
          f"{'see below':>17} │ toward GREEN")
    print(f"│ f*@w=0 (freq-wtd)      │ {b_sum['f_star_low_weight']:>17.2f} │ "
          f"{(n_sum['f_star_low_weight'] if n_sum else float('nan')):>17.2f} │ │")
    print()

    notes = []
    if n_band is None and b_band is not None:
        notes.append(
            "S4 has NO interior band on [0, 1]: f* is boundary-locked across "
            "all tail-weights (S2 baseline had an interior band [0.250, 0.325])."
        )
    elif thr_delta is None:
        notes.append("threshold/band comparison N/A.")
    elif thr_delta < -_FLAT_TOL:
        notes.append("threshold DROPPED — interior f* emerges at lower tail-weight under S4.")
    elif thr_delta > _FLAT_TOL:
        notes.append("threshold ROSE — interior f* requires higher tail-weight under S4.")
    else:
        notes.append("threshold unchanged.")

    if b_v == n_v:
        notes.append(f"verdict unchanged ({b_v}).")
    else:
        notes.append(f"verdict moved {b_v} → {n_v}.")

    # Honest strategic-note framing (per brief).
    notes.append(
        "Brief strategic-note context: S2 (rigorous tail engine) and S3 "
        "(stochastic SVI, degenerate-flag → fallback) did not shift Gate-A. "
        "S4 with the FULL machinery (DN ladder + multi-cycle + bucket + "
        "continuous κ) is the third structural lever. Whatever the result, "
        "the value of Strata still sits in the tail (quantified crash "
        "protection) + the R3 liquidity escape-hatch (NOT in single-cycle "
        "PnL). The interior f* remains conditional on tail-aversion; under "
        "crash-averse capital — the §2A target market — the f*(w) "
        "methodology is the deliverable."
    )

    print("[honest notes]")
    for note in notes:
        print(f"  - {note}")
    print()

    out = RESULTS_DIR / "s4_engine_delta.json"
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
            "baseline_verdict": b_v,
            "new_verdict": n_v,
            "baseline_f_star_low_weight": b_sum["f_star_low_weight"],
            "new_f_star_low_weight": n_sum["f_star_low_weight"] if n_sum else None,
        },
        "honest_notes": notes,
    }, open(out, "w", encoding="utf-8"), indent=2)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
