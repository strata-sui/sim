"""S5R3.5-final — re-derive verdict + gate_b from cached s5_gate_b.json.

The full sweep is ~10 min wall-clock; the verdict synthesis is sub-second.
This helper avoids re-running when only the verdict logic / structure has
changed (option-a rename, p01-disjunctive crash protection — both in
s5_main.py).

It:
  1. Loads `data/s1_results/s5_gate_b.json`.
  2. Re-builds `verdict` (option-a structure: strict_baseline_pass boolean
     + _gate_a_label diagnostic) and `gate_b` (disjunctive crash-protection
     synthesis) from the cached `benign_results` and `crash_results`.
  3. Writes the JSON back in-place.

The benign/crash results themselves are NOT recomputed — those come from
the upstream sweep and must already be sane (verified by S5R3.4 rerun).
Hero chart regeneration is NOT in scope here (the cached PNG is still
valid; only the verdict text/structure changed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gate_a import crash_protection_summary, gate_a_decide
from eval.objective import f_star_curve, summarize_f_star_curve
from s5_main import W_CRASH_GRID, RESULTS_DIR, _gate_b_verdict

J_PATH = RESULTS_DIR / "s5_gate_b.json"


def _coerce_f_keys(d: dict) -> dict:
    """JSON deserializes f-grid keys as strings; gate_a + objective want floats."""
    out = {}
    for strat, fmap in d.items():
        out[strat] = {float(k): v for k, v in fmap.items()}
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not J_PATH.exists():
        print(f"[FAIL] {J_PATH} missing — run s5_main.py first")
        return 1

    print(f"[load] {J_PATH}")
    data = json.load(open(J_PATH, "r", encoding="utf-8"))

    benign_results = _coerce_f_keys(data["benign_results"])
    crash_results = (
        _coerce_f_keys(data["crash_results"])
        if data.get("crash_results") else None
    )

    # ---- Re-derive verdict (option-a structure) -----------------------
    print("[re-derive] gate_a → strict_baseline_pass boolean")
    gate_a_raw = gate_a_decide(benign_results)
    verdict = {
        "strict_baseline_pass": gate_a_raw["verdict"] == "GREEN",
        "_gate_a_label": gate_a_raw["verdict"],
        "f_star": gate_a_raw["f_star"],
        "sortino_at_f_star": gate_a_raw["sortino_at_f_star"],
        "beats_baselines": gate_a_raw["beats_baselines"],
        "thresholds": gate_a_raw["thresholds"],
        "reasoning": gate_a_raw["reasoning"],
    }
    print(f"           strict_baseline_pass = {verdict['strict_baseline_pass']}")
    print(f"           _gate_a_label        = {verdict['_gate_a_label']}")

    # ---- Crash protection (cached) ------------------------------------
    print("[re-derive] crash_protection_summary")
    protection = (
        crash_protection_summary(crash_results) if crash_results else None
    )
    if protection is not None:
        print(f"           best_f                = {protection['best_protection_f']}")
        print(f"           best_sortino_uplift   = {protection['best_sortino_uplift']:+.4f}")
        print(f"           best_p01_reduction    = {protection['best_p01_reduction']:+.4f}")
        print(f"           protection_visible (s)= {protection['protection_visible']}")

    # ---- f*(w) curve --------------------------------------------------
    print("[re-derive] f_star_curve + summary")
    if crash_results is not None:
        curve = f_star_curve(benign_results, crash_results, W_CRASH_GRID)
        fstar_summary = summarize_f_star_curve(curve)
    else:
        curve = fstar_summary = None

    # ---- Gate-B synthesis (new disjunctive logic) ---------------------
    print("[re-derive] gate_b synthesis (option-a + disjunctive)")
    r3_path = RESULTS_DIR / "s5_r3_verification.json"
    gate_b = _gate_b_verdict(verdict, protection, fstar_summary, r3_path)
    print(f"           verdict_gate_b = {gate_b['verdict_gate_b']}")
    print(f"           {gate_b['reasoning']}")

    # ---- Write back ---------------------------------------------------
    data["verdict"] = verdict
    data["crash_protection"] = protection
    data["f_star_curve"] = curve
    data["f_star_summary"] = fstar_summary
    data["gate_b"] = gate_b

    json.dump(data, open(J_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\n[saved] {J_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
