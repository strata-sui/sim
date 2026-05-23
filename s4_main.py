"""S4.7/S4.8 — Run the full S4 model and produce the dual benign+crash sweep
+ f*(w_crash) curve. Mirrors s1.py's reporting structure but uses
``run_strategy_s4`` (DN ladder + multi-cycle + token-bucket).

Engine for PRICE PATHS: PolitisRomanoKouEngine (S2 PR+Kou, the baseline
S4 is compared against). Per-strike DN pricing uses the DETERMINISTIC
SVI anchor (S3 fallback canonical — brief §S4 guardrails forbid the
S3 stochastic engine because of its degenerate flag).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from engine.loader import load_range
from engine.price_engine_pr_kou import PolitisRomanoKouEngine
from engine.resample import log_returns, resample_klines
from engine.svi_det import ANCHOR_BTC_2026_05_15
from eval.f_sweep import _rolling_vol, compute_metrics
from eval.gate_a import crash_protection_summary, gate_a_decide
from eval.objective import f_star_curve, summarize_f_star_curve
from eval.scenarios import ScenarioReplay, find_crash_windows
from eval.strategy import (
    ALL_STRATEGIES,
    run_strategy_s4,
)
from model.dn_ladder import DEFAULT_LADDER_SIZE, DEFAULT_MONEYNESS_HI, DEFAULT_MONEYNESS_LO
from model.trader_flow import CONSERVATIVE_ANCHOR
from viz.plots import plot_f_star_vs_tailweight

# ---- S4 config -----------------------------------------------------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
START = "2020-01"
END = "2026-04"
STEP_MINUTES = 15

TOTAL_POOL_CAPITAL = 1_000_000.0
F_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80)
N_PATHS = 500              # smaller than S1 to fit multi-cycle compute budget
N_CYCLES = 10              # S4 default
PATH_STEPS_PER_CYCLE = 4   # 1h cycles on 15m bars
TOTAL_PATH_STEPS = N_CYCLES * PATH_STEPS_PER_CYCLE
INIT_PRICE = 60_000.0

CRASH_THRESHOLD = -0.04
W_CRASH_GRID = tuple(round(x, 3) for x in np.linspace(0.0, 1.0, 41))

LADDER_SIZE = DEFAULT_LADDER_SIZE
MONEYNESS_LO = DEFAULT_MONEYNESS_LO
MONEYNESS_HI = DEFAULT_MONEYNESS_HI
KAPPA_ALPHA = 2.0
KAPPA_BETA = 5.0
BUCKET_CAPACITY_FACTOR = 1.0   # default = S1-equivalent (capacity = θ)
BUCKET_REFILL_FACTOR = 1.0

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "s1_results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="S4 dual sweep + f*(w) curve.")
    p.add_argument("--out", default="s4_sweep")
    p.add_argument("--n-paths", type=int, default=N_PATHS)
    p.add_argument("--n-cycles", type=int, default=N_CYCLES)
    p.add_argument("--ladder-size", type=int, default=LADDER_SIZE)
    p.add_argument("--kappa-alpha", type=float, default=KAPPA_ALPHA)
    p.add_argument("--kappa-beta", type=float, default=KAPPA_BETA)
    p.add_argument("--bucket-capacity", type=float, default=BUCKET_CAPACITY_FACTOR)
    p.add_argument("--bucket-refill", type=float, default=BUCKET_REFILL_FACTOR)
    return p.parse_args(argv)


def _sweep_one_engine(
    engine, returns, args, label,
) -> dict:
    """Run the (n_paths × f × strategy) sweep on a price-path engine."""
    sim = engine.simulate(
        n_paths=args.n_paths, n_steps=TOTAL_PATH_STEPS, init_price=INIT_PRICE,
    )
    price_paths = sim["price"]
    log_paths = sim["log_return"]
    sigma_long_run = engine.sigma_historical
    realized_vols = _rolling_vol(log_paths, window=8, fallback=sigma_long_run)

    results = {s.name: {} for s in ALL_STRATEGIES}
    for f in F_GRID:
        strata_capital = f * TOTAL_POOL_CAPITAL
        other_lp = (1.0 - f) * TOTAL_POOL_CAPITAL
        for strat in ALL_STRATEGIES:
            pnls = np.empty(args.n_paths, dtype=float)
            for i in range(args.n_paths):
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
                    n_cycles=args.n_cycles,
                    path_steps=PATH_STEPS_PER_CYCLE,
                    ladder_size=args.ladder_size,
                    moneyness_lo=MONEYNESS_LO,
                    moneyness_hi=MONEYNESS_HI,
                    kappa_alpha=args.kappa_alpha,
                    kappa_beta=args.kappa_beta,
                    bucket_capacity_factor=args.bucket_capacity,
                    bucket_refill_factor=args.bucket_refill,
                    seed=42 + i,
                )
                pnls[i] = out["strata_pnl_total"]
            results[strat.name][f] = compute_metrics(pnls, capital=strata_capital)
        print(f"  [{label}] f={f:.2f}  raw_plp.sortino="
              f"{results['raw_plp'][f]['sortino']:+.3f}  "
              f"strata.sortino={results['strata'][f]['sortino']:+.3f}")
    return results


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    print(f"[S4] full model — ladder + multi-cycle + bucket")
    print(f"     n_paths={args.n_paths}  n_cycles={args.n_cycles}  "
          f"path_steps_per_cycle={PATH_STEPS_PER_CYCLE}  "
          f"total_steps={args.n_cycles * PATH_STEPS_PER_CYCLE}")
    print(f"     ladder={args.ladder_size}  κ~Beta({args.kappa_alpha},{args.kappa_beta})  "
          f"bucket(cap={args.bucket_capacity}, refill={args.bucket_refill})")
    print()

    print("[load] BTCUSDT 15m 2020-01..2026-04")
    df = load_range(SYMBOL, INTERVAL, START, END)
    r = log_returns(resample_klines(df, STEP_MINUTES)).values
    print(f"       {len(r):,} 15m log returns")
    print()

    print("[engine] PolitisRomanoKouEngine (S2 baseline price layer)")
    price_engine = PolitisRomanoKouEngine(
        log_returns=r, mean_block_length=4.0, seed=42,
    )

    print("\n[sweep BENIGN]")
    benign_results = _sweep_one_engine(price_engine, r, args, label="BENIGN")
    print()

    print("[crash] finding historical crash windows")
    crash_windows = find_crash_windows(
        r, n_steps=TOTAL_PATH_STEPS, threshold=CRASH_THRESHOLD,
    )
    print(f"       {len(crash_windows)} {TOTAL_PATH_STEPS}-step windows with cumulative ≤ {CRASH_THRESHOLD:.0%}")
    if len(crash_windows) == 0:
        print("       NO crash windows at this horizon — skipping crash sweep.")
        crash_results = None
    else:
        scenario = ScenarioReplay(crash_windows, sigma_historical=price_engine.sigma_historical)
        # Replay all available crash windows.
        scen_args = argparse.Namespace(**vars(args))
        scen_args.n_paths = scenario.n_windows
        print("\n[sweep CRASH]")
        crash_results = _sweep_one_engine(scenario, r, scen_args, label="CRASH ")
    print()

    print("[gate-a]")
    verdict = gate_a_decide(benign_results)
    protection = crash_protection_summary(crash_results) if crash_results else None
    print(f"  benign verdict: {verdict['verdict']}  f*={verdict['f_star']}  "
          f"sortino={verdict['sortino_at_f_star']}")
    if protection is not None:
        print(f"  crash protection: visible={protection['protection_visible']}  "
              f"best_f={protection['best_protection_f']}  "
              f"sortino_uplift={protection['best_sortino_uplift']:+.3f}")
    print()

    print("[f*(w_crash)]")
    if crash_results is not None:
        curve = f_star_curve(benign_results, crash_results, W_CRASH_GRID)
        fstar_summary = summarize_f_star_curve(curve)
        print(f"  interior band: {fstar_summary['interior_band']}  "
              f"threshold={fstar_summary['threshold_leaves_low_boundary']}")
        plot_path = RESULTS_DIR / f"{args.out}_f_star.png"
        plot_f_star_vs_tailweight(
            curve, plot_path,
            title=f"f*(w_crash) — S4 (ladder×{args.ladder_size}, "
                  f"{args.n_cycles}-cycle, bucket)",
        )
        print(f"  [plot] {plot_path}")
    else:
        curve, fstar_summary = None, None
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.out}.json"
    payload = {
        "config": {
            "engine_label": (
                f"S4(ladder={args.ladder_size}, "
                f"n_cycles={args.n_cycles}, "
                f"κ~Beta({args.kappa_alpha},{args.kappa_beta}), "
                f"bucket(cap={args.bucket_capacity},refill={args.bucket_refill}))"
            ),
            "n_paths": args.n_paths,
            "n_cycles": args.n_cycles,
            "path_steps_per_cycle": PATH_STEPS_PER_CYCLE,
            "f_grid": list(F_GRID),
            "ladder_size": args.ladder_size,
            "moneyness_lo": MONEYNESS_LO,
            "moneyness_hi": MONEYNESS_HI,
            "kappa_alpha": args.kappa_alpha,
            "kappa_beta": args.kappa_beta,
            "bucket_capacity_factor": args.bucket_capacity,
            "bucket_refill_factor": args.bucket_refill,
            "crash_threshold": CRASH_THRESHOLD,
            "crash_n_windows": int(len(crash_windows)),
        },
        "benign_results": benign_results,
        "crash_results": crash_results,
        "verdict": verdict,
        "crash_protection": protection,
        "f_star_curve": curve,
        "f_star_summary": fstar_summary,
    }
    json.dump(payload, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
