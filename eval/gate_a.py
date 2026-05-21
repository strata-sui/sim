"""Gate-A decision logic with R2 asymmetric guard (CLAUDE.md §6).

R2 asymmetric guard verbatim:

    "Gate A may declare RED only if no viable f* exists EVEN WITH
     (i) a conservative-but-not-pessimal trader-flow model,
     (ii) the withdrawal-limiter liquidity proxy present, AND
     (iii) acknowledging deterministic-SVI compresses the f* signal.
     Risks 4.2 (limiter ignored) and 4.3 (deterministic SVI) BOTH bias
     toward false-RED — so a RED under deterministic-SVI-only triggers
     a quick stochastic-SVI spot-check BEFORE any pivot, never an
     immediate pivot. Continue-checking is cheap; a wrong pivot is the
     single most expensive mistake in the project."

S1 deterministic-SVI consequence (per spec §6 Gate-A behavior): a RED
verdict here must be treated as PROVISIONAL — full S4 continuous-κ
spot-check is required before any concept pivot.

S1 simplification artifact warning: the current ``raw_plp`` baseline shows
infinite Sortino in the f-sweep because the S1 settle_path assumes
non-hedge UP/DN positions net to zero. Real-world raw PLP IS risky (the
entire "is PLP safe?" thesis CLAUDE.md §2A). The Strata-vs-fixed_hedge_naive
comparison remains valid since both strategies share the simplification.
Per-strike StrikeMatrix at S4 will fix this and let raw_plp show realistic
downside. Gate A reasoning explicitly flags this.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

import math


class GateAVerdict(Enum):
    GREEN = "GREEN"        # f* zone with materially better Sortino than baselines
    MARGINAL = "MARGINAL"  # positive but no clear margin OR baselines artifact
    RED = "RED"            # no viable f* (PROVISIONAL — needs stochastic-SVI check)


# Numeric thresholds — design choices at S1, NOT data-calibrated.
# Documented here so they appear in the Gate A reasoning output.
GREEN_SORTINO_MIN = 0.50          # Strata best Sortino at f* clearly net-positive
BASELINE_BEATEN_MARGIN = 0.30     # margin over each FINITE baseline at f*


def _argmax_sortino(by_f: dict) -> Tuple[float, float]:
    """Return (f_star, best_sortino) ignoring ±inf when finite values exist.

    All-inf case (e.g., S1 raw_plp simplification artifact): pick the lowest f
    (smaller Strata footprint preferred — more conservative deployment).
    """
    f_values = sorted(by_f.keys())
    sortinos = [by_f[f]["sortino"] for f in f_values]
    finite = [(i, s) for i, s in enumerate(sortinos) if math.isfinite(s)]
    if finite:
        i_best, s_best = max(finite, key=lambda x: x[1])
        return float(f_values[i_best]), float(s_best)
    # All-inf fallback.
    return float(f_values[0]), float("inf")


def gate_a_decide(
    sweep_results: Dict[str, dict],
    strata_name: str = "strata",
    baselines: Tuple[str, ...] = ("raw_plp", "fixed_hedge_naive"),
    green_min: float = GREEN_SORTINO_MIN,
    baseline_margin: float = BASELINE_BEATEN_MARGIN,
) -> dict:
    """Apply Gate-A logic to f-sweep results.

    Args:
        sweep_results:    output of ``eval.f_sweep.run_sweep``.
        strata_name:      key for the Strata strategy in the sweep dict.
        baselines:        baseline strategy names to compare against.
        green_min:        minimum Sortino at f* for GREEN.
        baseline_margin:  required Sortino margin over each finite baseline.

    Returns:
        dict with verdict, f_star, sortino_at_f_star, beats_baselines, reasoning.
    """
    if strata_name not in sweep_results:
        raise KeyError(
            f"strata strategy '{strata_name}' not in sweep_results keys: "
            f"{list(sweep_results.keys())}"
        )

    f_star, best_sortino = _argmax_sortino(sweep_results[strata_name])

    # Compute margin over each baseline at f_star.
    beats_baselines: Dict[str, float] = {}
    for b in baselines:
        if b not in sweep_results or f_star not in sweep_results[b]:
            beats_baselines[b] = float("nan")
            continue
        b_sortino = sweep_results[b][f_star]["sortino"]
        if not math.isfinite(b_sortino):
            # Baseline at +inf is the S1 raw_plp artifact (no downside) —
            # treat as "baseline is artifact; can't fairly compare margin".
            beats_baselines[b] = float("nan")
        else:
            beats_baselines[b] = float(best_sortino - b_sortino)

    # All finite baseline margins must pass for GREEN.
    finite_margins = [m for m in beats_baselines.values() if math.isfinite(m)]
    margin_ok = all(m >= baseline_margin for m in finite_margins) if finite_margins else False

    if not math.isfinite(best_sortino):
        # Strata in all-inf zone (S1 artifact) — treat as MARGINAL pending S4.
        verdict = GateAVerdict.MARGINAL
        reasoning = (
            f"Strata Sortino is non-finite at f={f_star:.2f} (S1 simplification "
            f"artifact: non-hedge UP/DN nets to 0 in settle_path). Cannot pass "
            f"Gate A strictly until S4 per-strike StrikeMatrix fixes this. "
            f"R2 guard: hold MARGINAL, proceed to continuous-κ spot-check."
        )
    elif best_sortino >= green_min and margin_ok:
        verdict = GateAVerdict.GREEN
        beat_strs = ", ".join(
            f"{b}=+{m:.2f}" for b, m in beats_baselines.items() if math.isfinite(m)
        ) or "no finite baselines (artifact)"
        reasoning = (
            f"GREEN. Strata Sortino={best_sortino:.3f} at f*={f_star:.2f} "
            f">= threshold {green_min:.2f}; beats finite baselines by required "
            f"margin {baseline_margin:.2f} ({beat_strs}). Proceed to S2 (rigorous "
            f"price engine), S3 (stochastic SVI), then unlock Move."
        )
    elif best_sortino >= 0.0:
        verdict = GateAVerdict.MARGINAL
        reasoning = (
            f"MARGINAL. Strata Sortino={best_sortino:.3f} at f*={f_star:.2f} "
            f"is positive but does not clear GREEN threshold "
            f"({best_sortino:.3f} < {green_min:.2f}) or baseline margin "
            f"({baseline_margin:.2f}). R2 guard: trigger stochastic-SVI + "
            f"continuous-κ spot-check before any reframe. Continue-checking is "
            f"cheap; coarsened knobs 4-6 bias toward false-RED/MARGINAL."
        )
    else:
        verdict = GateAVerdict.RED
        reasoning = (
            f"RED (PROVISIONAL). Strata best Sortino={best_sortino:.3f} at "
            f"f*={f_star:.2f}. R2 asymmetric guard MANDATES a stochastic-SVI "
            f"spot-check AND continuous-κ spot-check at Conservative anchor "
            f"BEFORE any concept pivot. Deterministic-SVI compresses the f* "
            f"signal (risk 4.3) and binary κ_strike biases toward false-RED "
            f"(spec §6 S1 mitigation rule). Continue-checking is cheap; a "
            f"wrong pivot on day 4 is the most expensive mistake in the project."
        )

    return {
        "verdict": verdict.value,
        "f_star": float(f_star),
        "sortino_at_f_star": (
            float(best_sortino) if math.isfinite(best_sortino) else None
        ),
        "beats_baselines": beats_baselines,
        "thresholds": {
            "green_sortino_min": green_min,
            "baseline_beaten_margin": baseline_margin,
        },
        "reasoning": reasoning,
    }
