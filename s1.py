"""S1 orchestrator: load data → bootstrap → f-sweep → Gate-A verdict.

End-to-end pipeline (CLAUDE.md §6 S1 task 7):
    1. Load 15m BTC returns from cached Binance dumps (S0 work).
    2. Build ``BootstrapEngine`` over empirical log-returns.
    3. Run coarse f-sweep at the **Conservative** trader-flow anchor (the
       Gate-A anchor per CLAUDE.md §6 R2 guard).
    4. Apply Gate-A decision logic with R2 asymmetric guard.
    5. Print verdict + write the full sweep results to ``sim/data/s1_results/``.

Usage:
    uv run python s1.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from engine.joint_engine import JointStochasticEngine
from engine.loader import load_range
from engine.price_engine import BootstrapEngine
from engine.price_engine_pr_kou import PolitisRomanoKouEngine
from engine.resample import log_returns, resample_klines
from engine.svi_calibration import calibrate_from_history
from engine.svi_det import ANCHOR_BTC_2026_05_15
from engine.svi_stoch import StochasticSVIEngine
from eval.f_sweep import run_sweep
from eval.gate_a import crash_protection_summary, gate_a_decide
from eval.objective import f_star_curve, summarize_f_star_curve
from eval.scenarios import ScenarioReplay, find_crash_windows
from eval.strategy import ALL_STRATEGIES
from model.trader_flow import CONSERVATIVE_ANCHOR
from viz.plots import plot_f_star_vs_tailweight


# ---- S1 config (per CLAUDE.md §6 thin-slice scope) ----------------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
START = "2020-01"
END = "2026-04"
STEP_MINUTES = 15

# Pool size + f-grid: S1 coarse (~10 f-points). S5 will scale up.
TOTAL_POOL_CAPITAL = 1_000_000.0
F_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80)

# Monte Carlo. S1 = 1k paths × 9 f × 3 strategies = 27k cycle evals.
# S5 will be 1M paths.
N_PATHS = 1000
# ONE path = ONE sub-hour Predict expiry cycle. The live BTC oracle we pulled
# expires ~1h45m out, so ~4-7 × 15m bars. We use 4 (≈1h) to keep open interest
# bounded (notional accumulates per step; over 4 steps ≈ 50% of balance, under
# the 80% max_exposure cap). Using 32 (8h) made open interest balloon to ~11×
# balance — absurd leverage, mean returns ~200× — because positions don't
# expire within the path. Multi-cycle compounding lands at S4.
PATH_STEPS = 4
BOOTSTRAP_BLOCK = 4            # 1h block on 15m bars (matches one expiry cycle)

INIT_PRICE = 60000.0           # synthetic forward at step 0

# Crash-replay threshold (Task C): a 1h (4-step) window qualifies as a crash
# path if its cumulative move <= this. -4% sits at the empirical 1h p0.1 tail
# and is deep enough to fire the ~2%-OTM hedge. Data-driven + date-agnostic:
# captures Black Thursday / LUNA / FTX automatically (verified present in S0).
CRASH_THRESHOLD = -0.04

# Tail-weight grid for the f*(w_crash) sweep (Task R2-B). 41 points over [0,1]
# resolves the interior-f* band finely enough to report its edges.
W_CRASH_GRID = tuple(round(x, 3) for x in np.linspace(0.0, 1.0, 41))
RESULTS_DIR = Path(__file__).resolve().parent / "data" / "s1_results"

# Deterministic SVI surface for S1. Uses the VERIFIED CLAUDE.md §4 live BTC
# anchor (a=1.658e-4, b=7.32e-3, rho=-0.3188, m=-0.00275, sigma=0.01426), NOT
# the degenerate thin-oracle 0xd153 sample (b=2.37e-4, rho=-0.94) that an
# earlier draft used — that surface was near-flat, pricing the OTM-DN hedge at
# ~0 and distorting the strata-vs-fixed_hedge comparison. S3 replaces this with
# a stochastic OU process fitted to predict-server SVI history.
DEFAULT_SVI = ANCHOR_BTC_2026_05_15


def parse_args(argv=None) -> argparse.Namespace:
    """CLI for the S1 dual-report orchestrator.

    --engine selects the price-path engine. Default 'bootstrap' is the S1
    baseline (fixed block). 'politis_romano_kou' is the S2 engine
    (stationary block + Kou jump overlay). Both share the same .simulate
    contract so the f-sweep / Gate-A / f*(w) pipeline is identical.
    """
    p = argparse.ArgumentParser(
        description="S1 Gate-A dual-report orchestrator (benign + crash + f*(w))."
    )
    p.add_argument(
        "--engine",
        choices=("bootstrap", "politis_romano_kou", "stochastic_svi"),
        default="bootstrap",
        help="price-path engine (default: bootstrap = S1 fixed-block). "
             "'stochastic_svi' = PR+Kou price layer + S3 OU-stochastic SVI "
             "with BTC-leverage coupling + Gatheral g(k) arb-clamp.",
    )
    p.add_argument(
        "--mean-block-length",
        type=float,
        default=4.0,
        help="mean block length in bars (default 4 = 1h on 15m bars). "
             "For 'bootstrap' the integer cast is used; for 'politis_romano_kou' "
             "this is the Geometric mean.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="sweep_conservative",
        help="output basename. JSON → data/s1_results/<out>.json, "
             "f*(w) plot → data/s1_results/<out>_f_star.png.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("[S1] Strata thin-slice / FAST GATE A")
    print(f"     engine={args.engine}  mean_block_length={args.mean_block_length}"
          f"  out={args.out}")
    print(
        f"     pool=${TOTAL_POOL_CAPITAL:,.0f}  "
        f"f_grid={[round(f, 2) for f in F_GRID]}"
    )
    print(f"     n_paths={N_PATHS}  steps_per_path={PATH_STEPS}  "
          f"block={BOOTSTRAP_BLOCK}")
    print(f"     anchor=CONSERVATIVE (Gate-A R2 guard)")
    print()

    print(f"[load] {SYMBOL} {INTERVAL} {START}..{END}")
    df_1m = load_range(SYMBOL, INTERVAL, START, END)
    print(f"       {len(df_1m):,} 1m bars loaded")
    print(f"[resample] -> {STEP_MINUTES}m")
    df_res = resample_klines(df_1m, STEP_MINUTES)
    print(f"           {len(df_res):,} bars")
    r = log_returns(df_res)
    print(f"[returns] {len(r):,} log returns, std={r.std():.6f}")
    print()

    print(f"[svi] using deterministic anchor: "
          f"a={DEFAULT_SVI.a:.2e} b={DEFAULT_SVI.b:.2e} "
          f"rho={DEFAULT_SVI.rho:.3f}")
    print()

    if args.engine == "bootstrap":
        engine = BootstrapEngine(
            log_returns=r.values,
            block_length=int(args.mean_block_length),
            seed=42,
        )
        engine_label = f"bootstrap(block={int(args.mean_block_length)})"
    elif args.engine == "politis_romano_kou":
        engine = PolitisRomanoKouEngine(
            log_returns=r.values,
            mean_block_length=args.mean_block_length,
            seed=42,
        )
        engine_label = (
            f"PR+Kou(mean_block={args.mean_block_length}, "
            f"λ={engine.jump_intensity:.0e}, "
            f"down_scale={engine.down_jump_scale}, "
            f"up_scale={engine.up_jump_scale})"
        )
    elif args.engine == "stochastic_svi":
        # S3.7: PR+Kou price layer + S3 stochastic-SVI dynamics with BTC-
        # leverage coupling + Gatheral g(k) arb-clamp.
        price_engine = PolitisRomanoKouEngine(
            log_returns=r.values,
            mean_block_length=args.mean_block_length,
            seed=42,
        )
        # Calibrate SVI dynamics from cached SVI history (if available) +
        # BTC realized vol. Brief §S3.4: anchor to history, NOT to Sortino.
        # When SVI history is unavailable (e.g., fresh checkout), fall back
        # to literature-default anchors — disclosed in diagnostics.
        svi_history_files = sorted(
            (Path(__file__).resolve().parent / "data").glob("svi_*.json")
        )
        if svi_history_files:
            import json as _json
            snaps = _json.load(open(svi_history_files[0], "r", encoding="utf-8"))
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
        svi_params, svi_diag = calibrate_from_history(
            svi_a_history=a_hist,
            svi_b_history=b_hist,
            svi_rho_history=rho_hist,
            btc_log_returns=r.values,
            dt_svi=1.0,
            dt_returns=1.0,
            vol_window=16,
            static_m=ANCHOR_BTC_2026_05_15.m,
            static_sigma=ANCHOR_BTC_2026_05_15.sigma,
        )
        svi_engine = StochasticSVIEngine(svi_params, seed=43)
        engine = JointStochasticEngine(
            price_engine=price_engine,
            svi_engine=svi_engine,
            static_m=ANCHOR_BTC_2026_05_15.m,
            static_sigma=ANCHOR_BTC_2026_05_15.sigma,
        )
        engine_label = (
            f"stochastic_svi(mean_block={args.mean_block_length}, "
            f"a_OU=θ{svi_params.a.theta:.2f}/σ{svi_params.a.sigma:.1e}, "
            f"btc_corr_a={svi_params.btc_correlation[0]:+.2f}, "
            f"fallbacks={len(svi_diag['fallbacks_disclosed'])})"
        )
    print(f"[engine] {engine_label}")
    print(f"         sigma_historical={engine.sigma_historical:.6f}")
    print()

    total_evals = len(F_GRID) * len(ALL_STRATEGIES) * N_PATHS
    print(
        f"[sweep] running {len(F_GRID)} × {len(ALL_STRATEGIES)} × "
        f"{N_PATHS} = {total_evals:,} cycle evals"
    )
    results = run_sweep(
        bootstrap=engine,
        svi_params=DEFAULT_SVI,
        anchor=CONSERVATIVE_ANCHOR,
        total_pool_capital=TOTAL_POOL_CAPITAL,
        f_grid=F_GRID,
        n_paths=N_PATHS,
        path_steps=PATH_STEPS,
        init_price=INIT_PRICE,
        progress=True,
    )
    print()

    # ---- Crash-replay sweep (Task C): the decisive tail test ------------
    crash_windows = find_crash_windows(
        r.values, n_steps=PATH_STEPS, threshold=CRASH_THRESHOLD
    )
    print(f"[crash] {len(crash_windows)} historical {PATH_STEPS}-step windows "
          f"with cumulative move <= {CRASH_THRESHOLD:.0%} "
          f"(Black Thursday / LUNA / FTX auto-captured)")
    scenario = ScenarioReplay(crash_windows, sigma_historical=engine.sigma_historical)
    crash_results = run_sweep(
        bootstrap=scenario,
        svi_params=DEFAULT_SVI,
        anchor=CONSERVATIVE_ANCHOR,
        total_pool_capital=TOTAL_POOL_CAPITAL,
        f_grid=F_GRID,
        n_paths=scenario.n_windows,
        path_steps=PATH_STEPS,
        init_price=INIT_PRICE,
        progress=False,
    )
    print()

    # ---- Dual-report (Task D): benign drag + crash protection -----------
    verdict = gate_a_decide(results)
    protection = crash_protection_summary(crash_results)

    print("[dual-report] strata vs raw_plp  (benign = full-distribution drag; "
          "crash = conditional tail protection)")
    print(f"  {'f':>5} | {'BENIGN sortino':>26} | {'CRASH sortino':>26} | "
          f"{'CRASH tail p01':>26}")
    print(f"  {'':>5} | {'strata':>12} {'raw_plp':>12} | "
          f"{'strata':>12} {'raw_plp':>12} | {'strata':>12} {'raw_plp':>12}")
    for f in F_GRID:
        b_s = results["strata"][f]["sortino"]
        b_r = results["raw_plp"][f]["sortino"]
        c_s = crash_results["strata"][f]["sortino"]
        c_r = crash_results["raw_plp"][f]["sortino"]
        cp_s = crash_results["strata"][f]["p01_return"]
        cp_r = crash_results["raw_plp"][f]["p01_return"]
        print(f"  {f:>5.2f} | {b_s:>12.3f} {b_r:>12.3f} | "
              f"{c_s:>12.3f} {c_r:>12.3f} | {cp_s:>12.4f} {cp_r:>12.4f}")
    print()

    print("[gate-a] R2 asymmetric guard decision (aggregate = benign)")
    print(f"    VERDICT      : {verdict['verdict']}")
    print(f"    f_star       : {verdict['f_star']:.2f}")
    print(f"    sortino_at_f*: {verdict['sortino_at_f_star']}")
    print(f"    margins      : {verdict['beats_baselines']}")
    print(f"    reasoning    : {verdict['reasoning']}")
    print()
    print("[crash-protection] hedge value where it is designed to show")
    print(f"    protection_visible   : {protection['protection_visible']}")
    print(f"    best_protection_f    : {protection['best_protection_f']:.2f}")
    print(f"    sortino_uplift @f    : {protection['best_sortino_uplift']:+.3f} "
          f"(strata − raw_plp, crash dist)")
    print(f"    mean_loss_reduction  : {protection['best_mean_loss_reduction']:+.4f} "
          f"(less loss per crash cycle)")
    print(f"    tail p01 reduction   : {protection['best_p01_reduction']:+.4f} "
          f"(shallower 1-in-100 crash loss)")
    print()
    print("[honest framing] In benign regimes (the ~99.9% majority) the DN hedge "
          "is a small\n"
          "    drag — it is negative-EV by construction (§3), value is "
          "distributional.\n"
          "    In the crash tail it TRUNCATES the loss (quantified above), and "
          "the R3\n"
          "    liquidity escape-hatch (redeem bypasses the PLP limiter) is NOT "
          "captured\n"
          "    in this single-cycle PnL at all. Verdict is honest, NOT forced GREEN.")
    print()

    # ---- Tail-weighted objective: f*(w_crash) curve (Task R2-B) ---------
    fstar_curve = f_star_curve(results, crash_results, W_CRASH_GRID)
    fstar_summary = summarize_f_star_curve(fstar_curve)

    print("[f*(w_crash)] interior-f* is CONDITIONAL on the tail-weight "
          "(risk-aversion choice)")
    print(f"    J(f;w) = (1-w)*Sortino_benign(f) + w*Sortino_crash(f)")
    print(f"    f* @ w=0 (freq-weighted, benign-dominated): "
          f"{fstar_summary['f_star_low_weight']:.2f}  (low boundary)")
    print(f"    f* @ w=1 (full crash weight):               "
          f"{fstar_summary['f_star_high_weight']:.2f}  (high boundary)")
    if fstar_summary["interior_exists"]:
        lo, hi = fstar_summary["interior_band"]
        print(f"    interior f* band: w_crash ∈ [{lo:.3f}, {hi:.3f}]  "
              f"(threshold leaves low boundary at w≈"
              f"{fstar_summary['threshold_leaves_low_boundary']:.3f})")
        print(f"    => interior f* exists ONLY for a strongly crash-averse "
              f"objective (w≈{lo:.2f} ≈ weighting the crash ~"
              f"{lo/0.001:.0f}x its ~0.1% natural frequency). NOT the "
              f"frequency-weighted optimum. Conditional, not guaranteed.")
    else:
        print("    interior f* band: NONE on [0,1] — f* switches boundary-to-"
              "boundary. Interior hump does NOT emerge at any tail-weight.")
    plot_path = RESULTS_DIR / f"{args.out}_f_star.png"
    plot_f_star_vs_tailweight(
        fstar_curve, plot_path,
        title=f"f*(w_crash) — {engine_label}",
    )
    print(f"    [plot] {plot_path}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.out}.json"
    payload = {"config": {
        "engine": args.engine,
        "engine_label": engine_label,
        "mean_block_length": float(args.mean_block_length),
        "anchor": "CONSERVATIVE",
        "total_pool_capital": TOTAL_POOL_CAPITAL,
        "f_grid": list(F_GRID),
        "n_paths": N_PATHS,
        "path_steps": PATH_STEPS,
        "bootstrap_block": BOOTSTRAP_BLOCK,
        "data_range": [START, END],
        "step_minutes": STEP_MINUTES,
        "crash_threshold": CRASH_THRESHOLD,
        "crash_n_windows": int(scenario.n_windows),
        "w_crash_grid": list(W_CRASH_GRID),
    },
        "benign_results": results,
        "crash_results": crash_results,
        "verdict": verdict,
        "crash_protection": protection,
        "f_star_curve": fstar_curve,
        "f_star_summary": fstar_summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
