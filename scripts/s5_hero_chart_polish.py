"""S6.4 — pitch-deck-ready hero chart re-render from cached Gate-B JSON.

Loads `data/s1_results/s5_gate_b.json` and re-renders the hero chart
with the s6_brief.md S6.4 locked caption + Strata word-mark + 3 export
sizes (1920×1080 PNG, 3840×2160 hi-DPI PNG, vector SVG). No re-run of
the 10-min Monte-Carlo sweep — strictly a presentation polish.

Color discipline (per brief): a SINGLE 2-color palette only —
Strata-blue (#1f4e79) for the strata curves, ember-red (#b22222) for
the crash overlays. Greys for axis / grid. NO chart-junk; information
density > visual flourish.

Output (overwrites in place):
    data/s1_results/s5_gate_b_hero.png       1920×1080  (web / README)
    data/s1_results/s5_gate_b_hero_hi.png    3840×2160  (pitch deck)
    data/s1_results/s5_gate_b_hero.svg                  (vector / slides)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from s5_main import RESULTS_DIR

J_PATH = RESULTS_DIR / "s5_gate_b.json"

# ★ S6.4 LOCKED caption — must match s6_brief.md byte-for-byte.
LOCKED_CAPTION = (
    "Sortino vs LP-share f, benign + crash anchors, f*(w_crash) inset. "
    "Gate-B MARGINAL — honest, post-accounting-fix, 10k MC. Three "
    "structural levers tested; value sits in tail + R3 + f*(w) "
    "methodology."
)

# Single 2-color palette (no decorative chart-junk).
STRATA_BLUE = "#1f4e79"
EMBER_RED = "#b22222"
GREY_AXIS = "#555555"
GREY_GRID = "#dddddd"


def _coerce_f_keys(d: dict) -> dict:
    """JSON serializes f-grid keys as strings; coerce back to floats."""
    out = {}
    for strat, fmap in d.items():
        out[strat] = {float(k): v for k, v in fmap.items()}
    return out


def _plot(data: dict, out_png: Path, out_hi: Path, out_svg: Path) -> None:
    benign = _coerce_f_keys(data["benign_results"])
    crash = _coerce_f_keys(data["crash_results"]) if data.get("crash_results") else None
    fstar_curve = data.get("f_star_curve")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )

    # ----- Left: Sortino vs f, benign + crash overlay -----------------
    f_grid = sorted(benign["strata"].keys())
    b_strata = [benign["strata"][f]["sortino"] for f in f_grid]
    b_raw = [benign["raw_plp"][f]["sortino"] for f in f_grid]

    ax1.plot(f_grid, b_strata, "-o", color=STRATA_BLUE, lw=2.2, ms=6,
             label="Strata — BENIGN (10k MC)")
    ax1.plot(f_grid, b_raw, "--o", color=STRATA_BLUE, alpha=0.45, lw=1.5,
             ms=5, label="raw_PLP — BENIGN")

    if crash is not None:
        c_strata = [crash["strata"][f]["sortino"] for f in f_grid]
        c_raw = [crash["raw_plp"][f]["sortino"] for f in f_grid]
        ax1.plot(f_grid, c_strata, "-s", color=EMBER_RED, lw=2.2, ms=6,
                 label="Strata — CRASH (5,900 historical)")
        ax1.plot(f_grid, c_raw, "--s", color=EMBER_RED, alpha=0.45, lw=1.5,
                 ms=5, label="raw_PLP — CRASH")

    ax1.axhline(0.0, color=GREY_AXIS, lw=0.7, ls=":")
    ax1.set_xlabel("LP-share fraction  f", fontsize=12)
    ax1.set_ylabel("Sortino", fontsize=12)
    ax1.set_title("f-Curve  ·  Sortino vs f  ·  benign + crash anchors",
                  fontsize=13, pad=10)
    ax1.grid(True, alpha=0.45, color=GREY_GRID)
    ax1.legend(fontsize=9, loc="best", framealpha=0.92)
    ax1.tick_params(colors=GREY_AXIS, labelsize=10)
    for spine in ax1.spines.values():
        spine.set_color(GREY_AXIS)

    # ----- Right: f*(w_crash) tail-weight curve -----------------------
    if fstar_curve:
        ws = [pt["w_crash"] for pt in fstar_curve]
        fstars = [pt["f_star"] for pt in fstar_curve]
        interior_ws = [pt["w_crash"] for pt in fstar_curve if pt["is_interior"]]
        ax2.step(ws, fstars, where="post", color=STRATA_BLUE, lw=2.4,
                 label="f*(w_crash)")
        ax2.scatter(ws, fstars, color=STRATA_BLUE, s=22, zorder=5)
        if interior_ws:
            ax2.axvspan(
                min(interior_ws), max(interior_ws),
                color="#ffcc66", alpha=0.32,
                label=f"interior  w ∈ [{min(interior_ws):.2f}, "
                      f"{max(interior_ws):.2f}]",
            )
        else:
            # Post-S5R3: no interior band. Annotate the actual finding.
            ax2.annotate(
                "f* = 0.05 across all w_crash\n"
                "(conservative LP-share IS the answer)",
                xy=(0.5, 0.05), xycoords="data",
                xytext=(0.55, 0.45), textcoords="data",
                fontsize=10, color=STRATA_BLUE,
                arrowprops=dict(arrowstyle="->", color=STRATA_BLUE, lw=1.2),
            )
    else:
        ax2.text(0.5, 0.5, "f*(w_crash) curve unavailable\n(no crash sweep)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color=GREY_AXIS)
    ax2.set_xlabel("tail-weight  w_crash", fontsize=12)
    ax2.set_ylabel("optimal LP-share  f*", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.02, 1.0)
    ax2.set_title("f*(w_crash)  ·  methodology curve", fontsize=13, pad=10)
    ax2.grid(True, alpha=0.45, color=GREY_GRID)
    ax2.legend(fontsize=9, loc="best", framealpha=0.92)
    ax2.tick_params(colors=GREY_AXIS, labelsize=10)
    for spine in ax2.spines.values():
        spine.set_color(GREY_AXIS)

    # ----- Strata word-mark + locked caption --------------------------
    fig.suptitle("STRATA  ·  Sui Overflow 2026  ·  DeepBook track",
                 fontsize=15, fontweight="bold", color=STRATA_BLUE, y=0.98)
    fig.text(
        0.5, 0.018, LOCKED_CAPTION,
        ha="center", va="bottom", fontsize=10, color=GREY_AXIS,
        wrap=True,
    )

    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.94))

    # ----- Export 3 sizes ---------------------------------------------
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight",
                facecolor="white")  # 1920×1080-ish at 16×9 figsize × 120dpi
    fig.savefig(out_hi, dpi=240, bbox_inches="tight",
                facecolor="white")  # 3840×2160 at 16×9 figsize × 240dpi
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                facecolor="white")  # vector
    plt.close(fig)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not J_PATH.exists():
        print(f"[FAIL] {J_PATH} missing — run s5_main.py first")
        return 1

    data = json.load(open(J_PATH, "r", encoding="utf-8"))

    out_png = RESULTS_DIR / "s5_gate_b_hero.png"
    out_hi = RESULTS_DIR / "s5_gate_b_hero_hi.png"
    out_svg = RESULTS_DIR / "s5_gate_b_hero.svg"

    _plot(data, out_png, out_hi, out_svg)

    print(f"[saved] {out_png}  ({out_png.stat().st_size:,} bytes)")
    print(f"[saved] {out_hi}   ({out_hi.stat().st_size:,} bytes)")
    print(f"[saved] {out_svg}  ({out_svg.stat().st_size:,} bytes)")
    print()
    print(f"locked caption (byte-match required at masterplanner review):")
    print(f"  {LOCKED_CAPTION!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
