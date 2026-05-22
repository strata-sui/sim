"""Plotting helpers for S1 artifacts (headless / Agg backend).

Kept dependency-light: matplotlib only. Functions write PNGs and return the
output path — no display, safe under ``uv run`` in CI / headless shells.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402


def plot_f_star_vs_tailweight(
    curve: List[dict],
    out_path: str | Path,
    title: str = "f*(w_crash) — optimal LP-share vs tail-weight",
) -> Path:
    """Plot f*(w_crash) from ``eval.objective.f_star_curve`` output.

    Shades the interior-f* band (where the optimum leaves the boundaries) so
    the conditional nature of the interior optimum is visually unmistakable.

    Args:
        curve:    list of dicts with ``w_crash``, ``f_star``, ``is_interior``.
        out_path: PNG destination (parent dirs created).
        title:    chart title.

    Returns:
        The output Path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ws = [pt["w_crash"] for pt in curve]
    fstars = [pt["f_star"] for pt in curve]
    interior_ws = [pt["w_crash"] for pt in curve if pt["is_interior"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(ws, fstars, where="post", color="#1f77b4", linewidth=2,
            label="f* (argmax J)")
    ax.scatter(ws, fstars, color="#1f77b4", s=18, zorder=3)

    if interior_ws:
        ax.axvspan(min(interior_ws), max(interior_ws), color="#ffcc00",
                   alpha=0.25,
                   label=f"interior f* band  w∈[{min(interior_ws):.2f}, "
                         f"{max(interior_ws):.2f}]")

    ax.set_xlabel("tail-weight  w_crash  (crash-scenario weight in objective)")
    ax.set_ylabel("optimal LP-share  f*")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    # Annotate the frequency-weighted region (crashes ≈ 0.1% → tiny w_crash).
    ax.annotate(
        "frequency-weighted\n(crash ≈0.1%) → f* low boundary",
        xy=(0.02, fstars[0]), xytext=(0.12, 0.15), fontsize=8,
        arrowprops=dict(arrowstyle="->", color="grey"), color="grey",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
