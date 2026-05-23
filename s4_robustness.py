"""S4 robustness — anti-overfit ±20% perturbation discipline (acceptance #8).

Fast variant: BENIGN-ONLY sweep (skips the heavy crash sweep which is
already covered in s4_main / s4_delta). The benign f*@w=0 is the
primary parameter-sensitivity probe — if perturbations leave it
invariant we raise the degenerate flag (per S3 precedent); if they
shift it, the engine is structurally responsive at the conservative
anchor.

This script loads BTC returns + builds the price engine + generates
paths ONCE, then iterates the run_strategy_s4 call over the
perturbation grid. Each perturbation is just a fresh re-run of the
benign sweep at the perturbed params — ~3-5s each, total < 1 min.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from engine.loader import load_range
from engine.price_engine_pr_kou import PolitisRomanoKouEngine
from engine.resample import log_returns, resample_klines
from engine.svi_det import ANCHOR_BTC_2026_05_15
from eval.f_sweep import _rolling_vol, compute_metrics
from eval.strategy import ALL_STRATEGIES, run_strategy_s4
from model.dn_ladder import DEFAULT_MONEYNESS_HI, DEFAULT_MONEYNESS_LO
from model.trader_flow import CONSERVATIVE_ANCHOR
from s1 import RESULTS_DIR

# Reduced MC for fast perturbation sweep.
N_PATHS_ROBUST = 100
N_CYCLES = 10
PATH_STEPS = 4
TOTAL_STEPS = N_CYCLES * PATH_STEPS
INIT_PRICE = 60_000.0
F_GRID = (0.05, 0.10, 0.20, 0.40, 0.80)   # coarser than s4_main for speed
TOTAL_POOL_CAPITAL = 1_000_000.0

BASELINE_PARAMS = dict(
    ladder_size=5,
    moneyness_lo=DEFAULT_MONEYNESS_LO,
    moneyness_hi=DEFAULT_MONEYNESS_HI,
    kappa_alpha=2.0, kappa_beta=5.0,
    bucket_capacity_factor=1.0, bucket_refill_factor=1.0,
)

PERTURBATIONS = [
    ("baseline",            {}),
    ("ladder_size_3",       {"ladder_size": 3}),
    ("ladder_size_7",       {"ladder_size": 7}),
    ("kappa_alpha_-20",     {"kappa_alpha": 1.6}),
    ("kappa_alpha_+20",     {"kappa_alpha": 2.4}),
    ("kappa_beta_-20",      {"kappa_beta": 4.0}),
    ("kappa_beta_+20",      {"kappa_beta": 6.0}),
    ("bucket_cap_-20",      {"bucket_capacity_factor": 0.8}),
    ("bucket_cap_+20",      {"bucket_capacity_factor": 1.2}),
    ("bucket_refill_-20",   {"bucket_refill_factor": 0.8}),
    ("bucket_refill_+20",   {"bucket_refill_factor": 1.2}),
]


def _sweep_benign(price_paths, log_paths, realized_vols, sigma_long_run, params) -> dict:
    """Run the benign f-sweep ONLY for the given perturbed params."""
    results = {s.name: {} for s in ALL_STRATEGIES}
    n_paths = price_paths.shape[0]
    for f in F_GRID:
        strata_capital = f * TOTAL_POOL_CAPITAL
        other_lp = (1.0 - f) * TOTAL_POOL_CAPITAL
        for strat in ALL_STRATEGIES:
            pnls = np.empty(n_paths, dtype=float)
            for i in range(n_paths):
                out = run_strategy_s4(
                    strategy=strat,
                    strata_capital=strata_capital,
                    other_lp_initial=other_lp,
                    price_path=price_paths[i],
                    log_returns=log_paths[i],
                    realized_vols=realized_vols[i],
                    sigma_long_run=sigma_long_run,
                    svi_params=ANCHOR_BTC_2026_05_15,
                    anchor=CONSERVATIVE_ANCHOR,
                    n_cycles=N_CYCLES, path_steps=PATH_STEPS,
                    seed=42 + i,
                    **params,
                )
                pnls[i] = out["strata_pnl_total"]
            results[strat.name][f] = compute_metrics(pnls, capital=strata_capital)
    return results


def _argmax_f(results: dict, strategy: str = "strata") -> tuple[float, float]:
    """Best (f, Sortino) for one strategy."""
    best_f, best_s = None, -float("inf")
    for f, m in results[strategy].items():
        if m["sortino"] > best_s:
            best_s = m["sortino"]
            best_f = f
    return best_f, best_s


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"[S4 robustness] ±20% perturbation on ladder/κ/bucket — BENIGN only")
    print(f"                n_paths={N_PATHS_ROBUST}  n_cycles={N_CYCLES}  "
          f"f_grid={F_GRID}")
    print()

    print("[load] BTCUSDT 15m 2020-01..2026-04")
    df = load_range("BTCUSDT", "1m", "2020-01", "2026-04")
    r = log_returns(resample_klines(df, 15)).values
    print(f"       {len(r):,} log returns")

    print("[engine] PR+Kou (S2 baseline price layer)")
    pe = PolitisRomanoKouEngine(log_returns=r, mean_block_length=4.0, seed=42)
    sim = pe.simulate(n_paths=N_PATHS_ROBUST, n_steps=TOTAL_STEPS, init_price=INIT_PRICE)
    price_paths = sim["price"]
    log_paths = sim["log_return"]
    sigma_long_run = pe.sigma_historical
    realized_vols = _rolling_vol(log_paths, window=8, fallback=sigma_long_run)
    print()

    rows = []
    print(f"  {'label':>22}  {'f*_strata':>10}  {'sortino@f*':>12}")
    for label, perturb in PERTURBATIONS:
        params = {**BASELINE_PARAMS, **perturb}
        try:
            results = _sweep_benign(
                price_paths, log_paths, realized_vols, sigma_long_run, params,
            )
        except ValueError as exc:
            # Some perturbations are structurally invalid (e.g. bucket
            # refill > capacity violates the TokenBucket invariant). The
            # honest discipline is to report them as skipped, NOT drop
            # them silently.
            print(f"  {label:>22}  SKIPPED  (invalid: {exc})")
            rows.append({
                "label": label, "params": params,
                "skipped_reason": str(exc),
            })
            continue
        fstar, sortino_at_fstar = _argmax_f(results, "strata")
        rows.append({
            "label": label,
            "params": params,
            "f_star": fstar,
            "sortino_at_f_star": sortino_at_fstar,
        })
        print(f"  {label:>22}  {fstar:>10.2f}  {sortino_at_fstar:>+12.4f}")

    f_stars = [r["f_star"] for r in rows if "f_star" in r]
    span = float(max(f_stars) - min(f_stars)) if f_stars else 0.0
    print(f"\n  f* span across perturbations: {span:.3f}")
    if span < 1e-6:
        print("  ⚠ DEGENERATE-FLAG: f* invariant across all ±20% perturbations "
              "— engine result independent of these parameters at the "
              "conservative anchor. Disclose per brief acceptance #8.")
        degenerate = True
    else:
        print("  ✓ engine RESPONDS to parameter perturbations (f* shifts under "
              "±20%) — engine is structurally responsive, NOT degenerate.")
        degenerate = False

    out = RESULTS_DIR / "s4_robustness.json"
    json.dump({
        "config": {
            "n_paths_robust": N_PATHS_ROBUST,
            "n_cycles": N_CYCLES,
            "f_grid": list(F_GRID),
            "sweep_mode": "benign_only",
            "note": (
                "Crash sweep skipped (heavy + already covered in "
                "s4_engine_delta). Benign f*@conservative-anchor is the "
                "primary parameter-sensitivity probe."
            ),
        },
        "rows": rows,
        "f_star_span": span,
        "degenerate_flag": degenerate,
    }, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
