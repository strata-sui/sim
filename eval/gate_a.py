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

Dual-distribution Gate (Task C/D): the benign bootstrap is the aggregate
risk-adjusted view (it samples crash blocks at their natural ~0.12%
frequency, so the rare hedge payoff barely moves benign Sortino — the
hedge reads as a small drag, which is honest). The conditional-crash
sweep (``eval.scenarios``) is where the hedge's protection is designed to
show; ``crash_protection_summary`` quantifies it. Per CLAUDE.md §3 a DN
hedge is negative-EV by construction (its value is distributional, not
aggregate-mean), so a benign drag + crash protection is the CORRECT,
honest result — NOT a failure, and NOT grounds to force GREEN.
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
        # Differentiate failure mode: Sortino-below-threshold vs margin-failed.
        sortino_passes = best_sortino >= green_min
        if sortino_passes and not margin_ok:
            failed_finite = [
                f"{b}={m:+.3f}"
                for b, m in beats_baselines.items()
                if math.isfinite(m) and m < baseline_margin
            ]
            explanation = (
                f"Sortino={best_sortino:.3f} clears the {green_min:.2f} "
                f"threshold but FAILS baseline margin (need +{baseline_margin:.2f} "
                f"over each finite baseline; got: {', '.join(failed_finite) or 'none'})"
            )
        elif not sortino_passes:
            explanation = (
                f"Sortino={best_sortino:.3f} does NOT clear the {green_min:.2f} "
                f"GREEN threshold"
            )
        else:
            explanation = (
                f"Sortino={best_sortino:.3f} but GREEN criteria not satisfied"
            )
        reasoning = (
            f"MARGINAL at f*={f_star:.2f}: {explanation}. R2 guard: trigger "
            f"stochastic-SVI + continuous-κ spot-check before any reframe. "
            f"Continue-checking is cheap; coarsened knobs 4-6 bias toward "
            f"false-RED/MARGINAL."
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


def crash_protection_summary(
    crash_results: Dict[str, dict],
    strata_name: str = "strata",
    baseline_name: str = "raw_plp",
) -> dict:
    """Quantify hedge protection on the conditional-crash distribution.

    The hedge is designed for the tail; this is where its value must show.
    For each f, computes Strata-vs-baseline deltas (all signed so POSITIVE =
    Strata better in the crash):

        sortino_uplift     = strata.sortino    − baseline.sortino
        mean_loss_reduction = strata.mean       − baseline.mean
        p01_reduction      = strata.p01_return − baseline.p01_return
                             (positive ⇒ Strata's tail loss is shallower)

    ``protection_visible`` is True iff at the best-protection f the hedge
    BOTH improves risk-adjusted return (sortino_uplift > 0) AND truncates
    the tail (p01_reduction > 0) on the crash distribution.

    Args:
        crash_results: ``run_sweep`` output on the ScenarioReplay crash paths.
        strata_name:   Strata strategy key.
        baseline_name: unhedged baseline key (default raw_plp).

    Returns:
        dict with per-f deltas + a best-protection summary + flag.
    """
    if strata_name not in crash_results or baseline_name not in crash_results:
        raise KeyError(
            f"need both '{strata_name}' and '{baseline_name}' in crash_results"
        )

    per_f: Dict[float, dict] = {}
    best_f = None
    best_uplift = -math.inf
    for f in sorted(crash_results[strata_name].keys()):
        st = crash_results[strata_name][f]
        bl = crash_results[baseline_name][f]
        sortino_uplift = float(st["sortino"] - bl["sortino"])
        mean_loss_reduction = float(st["mean_return"] - bl["mean_return"])
        p01_reduction = float(st["p01_return"] - bl["p01_return"])
        per_f[f] = {
            "sortino_uplift": sortino_uplift,
            "mean_loss_reduction": mean_loss_reduction,
            "p01_reduction": p01_reduction,
            "strata_sortino": float(st["sortino"]),
            "baseline_sortino": float(bl["sortino"]),
            "strata_mean": float(st["mean_return"]),
            "baseline_mean": float(bl["mean_return"]),
            "strata_p01": float(st["p01_return"]),
            "baseline_p01": float(bl["p01_return"]),
        }
        if sortino_uplift > best_uplift:
            best_uplift = sortino_uplift
            best_f = f

    best = per_f[best_f]
    protection_visible = (
        best["sortino_uplift"] > 0.0 and best["p01_reduction"] > 0.0
    )
    return {
        "per_f": per_f,
        "best_protection_f": float(best_f),
        "best_sortino_uplift": float(best["sortino_uplift"]),
        "best_mean_loss_reduction": float(best["mean_loss_reduction"]),
        "best_p01_reduction": float(best["p01_reduction"]),
        "protection_visible": bool(protection_visible),
    }
