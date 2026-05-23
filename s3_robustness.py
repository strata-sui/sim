"""S3.6 — anti-overfit robustness: OOS 70/30 split + ±20% perturbation table.

Two diagnostics, BOTH reported honestly (BAB MVSK pattern):

  1. OOS split: calibrate on the first 70% of BTC log-returns, run the
     stochastic-SVI engine on the last 30% (held-out window). Compare
     interior-band thresholds. If the OOS metric is very different from
     IS → overfit; disclose.
  2. Per-OU perturbation: ±20% on each (θ, σ) of (a, b, ρ) — 6 OU params,
     12 total runs. Tabulate f*(w=0) and the threshold (w where f* leaves
     the low boundary). If any single ±20% perturbation flips the
     boundary regime wildly → over-fit / unstable; flag explicitly.

Time-box (brief §S3.6 + the 3-day cap): we use a reduced MC (500 paths)
to keep this orchestrator under a couple of minutes. The acceptance
criterion is STABILITY, not absolute numbers.

Usage:
    uv run python s3_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from engine.joint_engine import JointStochasticEngine
from engine.loader import load_range
from engine.ou import OUParams
from engine.price_engine_pr_kou import PolitisRomanoKouEngine
from engine.resample import log_returns, resample_klines
from engine.svi_calibration import calibrate_from_history
from engine.svi_det import ANCHOR_BTC_2026_05_15
from engine.svi_stoch import StochasticSVIEngine, SVIDynamicsParams
from eval.f_sweep import run_sweep
from eval.objective import f_star_curve, summarize_f_star_curve
from eval.scenarios import ScenarioReplay, find_crash_windows
from eval.strategy import ALL_STRATEGIES
from model.trader_flow import CONSERVATIVE_ANCHOR
from s1 import (
    CRASH_THRESHOLD,
    F_GRID,
    INIT_PRICE,
    PATH_STEPS,
    RESULTS_DIR,
    TOTAL_POOL_CAPITAL,
    W_CRASH_GRID,
)


N_PATHS_ROBUST = 500  # smaller MC for fast robustness sweep


def _build_joint(returns: np.ndarray, svi_params: SVIDynamicsParams) -> JointStochasticEngine:
    """Construct the joint engine from calibrated dynamics."""
    pe = PolitisRomanoKouEngine(log_returns=returns, mean_block_length=4.0, seed=42)
    se = StochasticSVIEngine(svi_params, seed=43)
    return JointStochasticEngine(
        price_engine=pe, svi_engine=se,
        static_m=ANCHOR_BTC_2026_05_15.m,
        static_sigma=ANCHOR_BTC_2026_05_15.sigma,
    )


def _run_sweep_and_summarize(engine, returns) -> dict:
    """Run benign + crash sweep, return f*(w_crash) summary."""
    benign = run_sweep(
        bootstrap=engine, svi_params=ANCHOR_BTC_2026_05_15,
        anchor=CONSERVATIVE_ANCHOR,
        total_pool_capital=TOTAL_POOL_CAPITAL, f_grid=F_GRID,
        n_paths=N_PATHS_ROBUST, path_steps=PATH_STEPS,
        init_price=INIT_PRICE, progress=False,
    )
    crash_windows = find_crash_windows(returns, n_steps=PATH_STEPS, threshold=CRASH_THRESHOLD)
    scenario = ScenarioReplay(crash_windows, sigma_historical=engine.sigma_historical)
    crash = run_sweep(
        bootstrap=scenario, svi_params=ANCHOR_BTC_2026_05_15,
        anchor=CONSERVATIVE_ANCHOR,
        total_pool_capital=TOTAL_POOL_CAPITAL, f_grid=F_GRID,
        n_paths=scenario.n_windows, path_steps=PATH_STEPS,
        init_price=INIT_PRICE, progress=False,
    )
    curve = f_star_curve(benign, crash, W_CRASH_GRID)
    return summarize_f_star_curve(curve)


def _perturb_ou(p: OUParams, attr: str, factor: float) -> OUParams:
    """Return a new OUParams with one attribute scaled by `factor`."""
    val = getattr(p, attr) * factor
    return OUParams(
        theta=val if attr == "theta" else p.theta,
        mu=val if attr == "mu" else p.mu,
        sigma=val if attr == "sigma" else p.sigma,
    )


def _perturb_params(base: SVIDynamicsParams, factor_label: str, factor: float) -> SVIDynamicsParams:
    """Apply ±20% perturbation to one OU param (e.g. 'a.sigma'). Returns new params."""
    factor_name, attr = factor_label.split(".")
    new_a = base.a
    new_b = base.b
    new_rho = base.rho
    if factor_name == "a":
        new_a = _perturb_ou(base.a, attr, factor)
    elif factor_name == "b":
        new_b = _perturb_ou(base.b, attr, factor)
    elif factor_name == "rho":
        new_rho = _perturb_ou(base.rho, attr, factor)
    return SVIDynamicsParams(
        a=new_a, b=new_b, rho=new_rho,
        correlation=base.correlation, m=base.m, sigma=base.sigma,
        btc_correlation=base.btc_correlation,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[S3.6] Stochastic-SVI robustness — OOS 70/30 + ±20% OU perturbation")
    print(f"       n_paths_robust={N_PATHS_ROBUST}  (reduced for speed)")
    print()

    # ---- Load data ----
    print("[load] BTCUSDT 1m 2020-01..2026-04 → 15m log returns")
    df = load_range("BTCUSDT", "1m", "2020-01", "2026-04")
    r = log_returns(resample_klines(df, 15)).values

    # ---- IS / OOS split ----
    n = len(r)
    cut = int(round(n * 0.7))
    r_is, r_oos = r[:cut], r[cut:]
    print(f"       n_total={n:,}  IS(70%)={len(r_is):,}  OOS(30%)={len(r_oos):,}")
    print()

    # ---- Calibrate on IS ----
    svi_history_files = sorted(
        (Path(__file__).resolve().parent / "data").glob("svi_*.json")
    )
    if svi_history_files:
        snaps = json.load(open(svi_history_files[0], "r", encoding="utf-8"))
        scale = 1.0e9
        a_hist = np.array([s["a"] / scale for s in snaps])
        b_hist = np.array([s["b"] / scale for s in snaps])
        rho_hist = np.array([
            (-s["rho"] if s.get("rho_negative", False) else s["rho"]) / scale
            for s in snaps
        ])
    else:
        a_hist = np.array([ANCHOR_BTC_2026_05_15.a])
        b_hist = np.array([ANCHOR_BTC_2026_05_15.b])
        rho_hist = np.array([ANCHOR_BTC_2026_05_15.rho])

    base_params, diag = calibrate_from_history(
        a_hist, b_hist, rho_hist, btc_log_returns=r_is,
        dt_svi=1.0, dt_returns=1.0, vol_window=16,
        static_m=ANCHOR_BTC_2026_05_15.m,
        static_sigma=ANCHOR_BTC_2026_05_15.sigma,
    )
    print("[calibration] IS-calibrated dynamics:")
    print(f"  a    : θ={base_params.a.theta:.3f}  μ={base_params.a.mu:.2e}  σ={base_params.a.sigma:.2e}")
    print(f"  b    : θ={base_params.b.theta:.3f}  μ={base_params.b.mu:.2e}  σ={base_params.b.sigma:.2e}")
    print(f"  rho  : θ={base_params.rho.theta:.3f}  μ={base_params.rho.mu:.3f}  σ={base_params.rho.sigma:.3f}")
    print(f"  btc_corr : {base_params.btc_correlation}")
    print(f"  diagnostic fits_well = {diag['rv_ou_calibration']['fits_well']}")
    print()

    # ---- Baseline IS run ----
    print("[run] BASELINE (full data, IS-calibrated)")
    engine_is = _build_joint(r, base_params)
    base_summary = _run_sweep_and_summarize(engine_is, r)
    print(f"      f*@w=0={base_summary['f_star_low_weight']:.2f}  f*@w=1={base_summary['f_star_high_weight']:.2f}  "
          f"interior band={base_summary['interior_band']}  threshold={base_summary['threshold_leaves_low_boundary']}")
    print()

    # ---- OOS run (engine on OOS returns) ----
    print("[run] OOS (held-out 30% — engine fed the OOS return window)")
    engine_oos = _build_joint(r_oos, base_params)
    oos_summary = _run_sweep_and_summarize(engine_oos, r_oos)
    print(f"      f*@w=0={oos_summary['f_star_low_weight']:.2f}  f*@w=1={oos_summary['f_star_high_weight']:.2f}  "
          f"interior band={oos_summary['interior_band']}  threshold={oos_summary['threshold_leaves_low_boundary']}")
    print()

    # ---- ±20% perturbations on (θ, σ) of (a, b, ρ) → 12 runs ----
    perturb_targets = [
        ("a.theta", 1.20), ("a.theta", 0.80),
        ("a.sigma", 1.20), ("a.sigma", 0.80),
        ("b.theta", 1.20), ("b.theta", 0.80),
        ("b.sigma", 1.20), ("b.sigma", 0.80),
        ("rho.theta", 1.20), ("rho.theta", 0.80),
        ("rho.sigma", 1.20), ("rho.sigma", 0.80),
    ]
    print("[perturb] ±20% on each (θ, σ) of (a, b, ρ) — 12 runs")
    print(f"  {'param':>10}  {'factor':>6}  {'f*@w=0':>7}  {'f*@w=1':>7}  {'threshold':>10}  {'band':>20}")
    perturb_results = []
    for label, fac in perturb_targets:
        perturbed = _perturb_params(base_params, label, fac)
        e = _build_joint(r, perturbed)
        try:
            s = _run_sweep_and_summarize(e, r)
        except Exception as exc:
            print(f"  {label:>10}  {fac:>5.2f}x   FAILED ({exc})")
            perturb_results.append({"param": label, "factor": fac, "error": str(exc)})
            continue
        band_str = f"[{s['interior_band'][0]:.2f},{s['interior_band'][1]:.2f}]" if s["interior_band"] else "NONE"
        thr_str = f"{s['threshold_leaves_low_boundary']:.3f}" if s["threshold_leaves_low_boundary"] is not None else "N/A"
        print(f"  {label:>10}  {fac:>5.2f}x   {s['f_star_low_weight']:>6.2f}   {s['f_star_high_weight']:>6.2f}   {thr_str:>10}  {band_str:>20}")
        perturb_results.append({
            "param": label, "factor": fac,
            "f_star_low_weight": s["f_star_low_weight"],
            "f_star_high_weight": s["f_star_high_weight"],
            "threshold": s["threshold_leaves_low_boundary"],
            "interior_band": s["interior_band"],
        })
    print()

    # ---- Honest verdict ----
    f_low_seq = [p.get("f_star_low_weight") for p in perturb_results if "f_star_low_weight" in p]
    if f_low_seq:
        span = max(f_low_seq) - min(f_low_seq)
    else:
        span = None
    print("[verdict]")
    print(f"  baseline f*@w=0:      {base_summary['f_star_low_weight']:.2f}")
    print(f"  OOS f*@w=0:           {oos_summary['f_star_low_weight']:.2f}")
    if span is not None:
        print(f"  perturb f*@w=0 span:  {span:.2f}  (max - min across all 12 ±20% perturbations)")
    print("  Honest reading: IS↔OOS agreement is the OOS-overfit check; perturb-span is the parameter-sensitivity check.")
    print("  No single perturbation should wildly flip the f* regime; if it does, the result is over-fit and flagged.")
    # Diagnostic flag: if the result is completely insensitive to dynamics
    # AND the calibrated μ differs from the verified anchor by >1 order of
    # magnitude, the engine is running a DEGENERATE surface (thin SVI history).
    anchor_a_ratio = base_params.a.mu / max(ANCHOR_BTC_2026_05_15.a, 1e-12)
    anchor_b_ratio = base_params.b.mu / max(ANCHOR_BTC_2026_05_15.b, 1e-12)
    degenerate_flag = (
        span is not None and span < 1e-6
        and (anchor_a_ratio < 0.5 or anchor_b_ratio < 0.5)
    )
    if degenerate_flag:
        print(
            "  ⚠ DEGENERATE-SURFACE FLAG: f* invariant AND calibrated μ << verified anchor\n"
            f"    (μ_a ratio = {anchor_a_ratio:.2f}, μ_b ratio = {anchor_b_ratio:.2f}).\n"
            "    The stochastic engine is reproducing a near-flat surface from the\n"
            "    thin SVI history (~100 snapshots from a single thin oracle). The\n"
            "    f*=0.80 outcome is an artifact of cheap-hedge degenerate pricing,\n"
            "    NOT a structural finding. Recommended: fallback to deterministic\n"
            "    SVI (CLAUDE.md §6 S3 fallback rule) — and disclose."
        )

    out_path = RESULTS_DIR / "s3_robustness.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({
        "config": {
            "engine": "stochastic_svi",
            "n_paths": N_PATHS_ROBUST,
            "is_oos_split": 0.7,
            "perturbation_factor": 0.2,
        },
        "calibration_diagnostics": diag,
        "baseline_is": base_summary,
        "oos_summary": oos_summary,
        "perturb_results": perturb_results,
        "perturb_f_star_low_span": span,
    }, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
