"""Tail-weighted blended objective + f*(w_crash) curve (S1 Task R2-B).

The benign and crash f-sweeps are each MONOTONIC in opposite directions
(benign Sortino decreases in f → favors low f; crash Sortino increases in
f → favors high f). Neither alone has an interior maximum. An interior f*
("hump") only exists for a COMBINED objective that trades the two off:

    J(f; w_crash) = (1 − w_crash) · Sortino_benign(f)
                  +      w_crash  · Sortino_crash(f)

``w_crash`` is the investor's tail-weight (risk-aversion CHOICE, not a data
fact). The frequency-weighted value of w_crash is tiny (crashes are ~0.1%
of paths), so the objective ≈ benign and f* sits at the low boundary. An
interior f* emerges only when w_crash is raised well above crash frequency
— i.e., for a strongly crash-averse investor.

This module quantifies exactly WHERE that happens: ``f_star_curve`` returns
f*(w_crash) over a grid, and ``interior_band`` reports the w_crash range (if
any) for which f* is strictly interior. This is the f*-methodology
deliverable (CLAUDE.md §2A headline), stated CORRECTLY as conditional —
NOT as an unconditional "guaranteed interior hump".
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _sorted_f_sortino(by_f: Dict, metric: str = "sortino") -> Tuple[np.ndarray, np.ndarray]:
    """Return (f_grid, metric_values) sorted ascending by f."""
    fs = sorted(by_f.keys(), key=float)
    f_grid = np.array([float(f) for f in fs], dtype=float)
    vals = np.array([by_f[f][metric] for f in fs], dtype=float)
    return f_grid, vals


def blended_objective(
    benign_results: Dict[str, dict],
    crash_results: Dict[str, dict],
    w_crash: float,
    strata_name: str = "strata",
) -> Tuple[np.ndarray, np.ndarray]:
    """J(f; w_crash) over the shared f-grid for one strategy.

    Returns (f_grid, J_values). Raises if the two sweeps disagree on the grid.
    """
    if not 0.0 <= w_crash <= 1.0:
        raise ValueError(f"w_crash must be in [0, 1], got {w_crash}")
    fb, ben = _sorted_f_sortino(benign_results[strata_name])
    fc, cra = _sorted_f_sortino(crash_results[strata_name])
    if fb.shape != fc.shape or not np.allclose(fb, fc):
        raise ValueError("benign and crash f-grids must match")
    J = (1.0 - w_crash) * ben + w_crash * cra
    return fb, J


def f_star_at_weight(
    benign_results: Dict[str, dict],
    crash_results: Dict[str, dict],
    w_crash: float,
    strata_name: str = "strata",
) -> dict:
    """Optimal f and whether it is interior at a single tail-weight.

    Returns dict {w_crash, f_star, J_at_f_star, is_interior}.
    ``is_interior`` = f* strictly between the grid endpoints.
    """
    f_grid, J = blended_objective(
        benign_results, crash_results, w_crash, strata_name
    )
    i = int(np.argmax(J))
    is_interior = 0 < i < len(f_grid) - 1
    return {
        "w_crash": float(w_crash),
        "f_star": float(f_grid[i]),
        "J_at_f_star": float(J[i]),
        "is_interior": bool(is_interior),
    }


def f_star_curve(
    benign_results: Dict[str, dict],
    crash_results: Dict[str, dict],
    w_grid: Sequence[float],
    strata_name: str = "strata",
) -> List[dict]:
    """f*(w_crash) over a tail-weight grid (list of f_star_at_weight dicts)."""
    return [
        f_star_at_weight(benign_results, crash_results, float(w), strata_name)
        for w in w_grid
    ]


def interior_band(curve: List[dict]) -> Optional[Tuple[float, float]]:
    """Lowest + highest w_crash in ``curve`` for which f* is strictly interior.

    Returns (w_low, w_high) or None if f* is never interior on the grid.
    The band is the tail-weight range where the interior-f* ("hump") exists;
    its lower edge is the THRESHOLD at which f* leaves the low boundary.
    """
    interior_ws = [pt["w_crash"] for pt in curve if pt["is_interior"]]
    if not interior_ws:
        return None
    return (min(interior_ws), max(interior_ws))


def summarize_f_star_curve(curve: List[dict]) -> dict:
    """Compact, honest summary for the report + JSON.

    Reports the interior band (or its absence), the boundary f* values, and
    the threshold tail-weight at which f* first leaves the low boundary.
    """
    band = interior_band(curve)
    w_sorted = sorted(curve, key=lambda p: p["w_crash"])
    f0 = w_sorted[0]["f_star"]   # f* at w_crash = grid min (≈ frequency-weighted)
    f1 = w_sorted[-1]["f_star"]  # f* at w_crash = grid max (full crash weight)
    # Threshold: first w where f* differs from the low-w boundary value.
    leave_low = next(
        (p["w_crash"] for p in w_sorted if p["f_star"] != f0), None
    )
    return {
        "interior_band": list(band) if band is not None else None,
        "interior_exists": band is not None,
        "f_star_low_weight": float(f0),
        "f_star_high_weight": float(f1),
        "threshold_leaves_low_boundary": (
            float(leave_low) if leave_low is not None else None
        ),
    }
