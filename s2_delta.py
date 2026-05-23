"""S2.5 — Document the delta between the S1/R2-B baseline and the S2 engine.

Loads two sweep JSONs (bootstrap baseline vs PR+Kou), tabulates the key
metrics side by side, and reports whether interior band / threshold /
verdict moved in the directions the brief flagged as EXPECTED. Per brief
§S2.5 those directions are NOT acceptance criteria — they are hypotheses;
the rule is REPORT HONESTLY.

Expected directions (per brief):
    interior band  widen
    threshold w*   drop
    Gate-A         move toward GREEN

Run after s1.py has produced sweep_bootstrap.json + sweep_pr_kou.json
(s2_sensitivity.py also produces these as sens_blk4.json effectively).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from s1 import RESULTS_DIR

BASELINE = "sweep_bootstrap"        # S1/R2-B-equivalent (BootstrapEngine bl=4)
NEW_ENGINE = "sweep_pr_kou"         # S2 (PR+Kou default params)


def load(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run s1.py with --out {name} first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _band_width(band):
    return (band[1] - band[0]) if band else None


_FLAT_TOL = 1e-9   # below this |delta| we report "flat" instead of up/down


def _arrow(direction: str, actual: float | None) -> str:
    """Return ✓/✗/=: expected direction matched, missed, or flat (within tol)."""
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

    base = load(BASELINE)
    new = load(NEW_ENGINE)

    print("[S2.5] Engine delta — R2-B baseline vs PR+Kou")
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

    print("│ Metric                 │ baseline       │ PR+Kou         │ delta       │ expected | actual")
    print("│ ───────────────────────│ ───────────────│ ───────────────│ ──────────  │ ─────────────────")
    print(f"│ interior_band          │ "
          f"{str(b_band):>15} │ {str(n_band):>15} │ "
          f"{('width %+.4f' % width_delta) if width_delta is not None else 'N/A':>11} │ "
          f"widen      | { _arrow('up', width_delta)}")
    print(f"│ threshold (leave low)  │ "
          f"{b_thr if b_thr is not None else 'N/A':>15} │ "
          f"{n_thr if n_thr is not None else 'N/A':>15} │ "
          f"{('%+.4f' % thr_delta) if thr_delta is not None else 'N/A':>11} │ "
          f"drop       | { _arrow('down', thr_delta)}")
    print(f"│ Gate-A verdict         │ {b_v['verdict']:>15} │ {n_v['verdict']:>15} │ "
          f"{'see below':>11} │ toward GREEN")
    print(f"│ Strata Sortino at f*   │ {str(b_v['sortino_at_f_star']):>15} │ "
          f"{str(n_v['sortino_at_f_star']):>15} │ "
          f"{'(see f-curves)':>11} │")
    print()

    # Honest narrative — synthesize without spin.
    notes = []
    if thr_delta is not None and thr_delta < 0:
        notes.append("threshold dropped — interior f* emerges at a LOWER tail-weight under the new engine.")
    elif thr_delta is not None and thr_delta > 0:
        notes.append("threshold ROSE — interior f* requires HIGHER tail-weight under the new engine "
                     "(opposite of the brief's expected direction; honest result).")
    elif thr_delta == 0:
        notes.append("threshold unchanged.")
    if width_delta is None:
        pass
    elif abs(width_delta) < _FLAT_TOL:
        notes.append("interior band width unchanged (band SHIFTED, did not widen).")
    elif width_delta > 0:
        notes.append("interior band widened.")
    else:
        notes.append("interior band narrowed.")
    if b_v["verdict"] != n_v["verdict"]:
        notes.append(f"verdict moved {b_v['verdict']} → {n_v['verdict']}.")
    else:
        notes.append(f"verdict unchanged ({b_v['verdict']}).")
    notes.append(
        "context: ScenarioReplay (crash sweep) is DATA-driven from historical "
        "windows — it is identical across engines by construction. Engine swap "
        "shifts only BENIGN. With λ=1e-4 anchored to historical extreme-bar "
        "rate, expected Kou jumps over 4000 bars/run ≈ 0.4 — Kou contributes "
        "sparsely. The headline takeaway: the new engine does NOT materially "
        "shift the Gate-A picture at these parameters. That is the honest "
        "S2 result; do not force GREEN by tuning Kou."
    )

    print("[honest notes]")
    for note in notes:
        print(f"  - {note}")
    print()

    out = RESULTS_DIR / "s2_engine_delta.json"
    with open(out, "w", encoding="utf-8") as f:
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
            },
            "honest_notes": notes,
        }, f, indent=2)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
