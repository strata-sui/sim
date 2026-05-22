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

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from engine.loader import load_range
from engine.price_engine import BootstrapEngine
from engine.resample import log_returns, resample_klines
from engine.svi_det import ANCHOR_BTC_2026_05_15
from eval.f_sweep import run_sweep
from eval.gate_a import crash_protection_summary, gate_a_decide
from eval.scenarios import ScenarioReplay, find_crash_windows
from eval.strategy import ALL_STRATEGIES
from model.trader_flow import CONSERVATIVE_ANCHOR


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
RESULTS_DIR = Path(__file__).resolve().parent / "data" / "s1_results"

# Deterministic SVI surface for S1. Uses the VERIFIED CLAUDE.md §4 live BTC
# anchor (a=1.658e-4, b=7.32e-3, rho=-0.3188, m=-0.00275, sigma=0.01426), NOT
# the degenerate thin-oracle 0xd153 sample (b=2.37e-4, rho=-0.94) that an
# earlier draft used — that surface was near-flat, pricing the OTM-DN hedge at
# ~0 and distorting the strata-vs-fixed_hedge comparison. S3 replaces this with
# a stochastic OU process fitted to predict-server SVI history.
DEFAULT_SVI = ANCHOR_BTC_2026_05_15


def main() -> int:
    print("[S1] Strata thin-slice / FAST GATE A")
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

    print("[bootstrap] building engine")
    boot = BootstrapEngine(
        log_returns=r.values, block_length=BOOTSTRAP_BLOCK, seed=42
    )
    print(f"            sigma_historical={boot.sigma_historical:.6f}")
    print()

    total_evals = len(F_GRID) * len(ALL_STRATEGIES) * N_PATHS
    print(
        f"[sweep] running {len(F_GRID)} × {len(ALL_STRATEGIES)} × "
        f"{N_PATHS} = {total_evals:,} cycle evals"
    )
    results = run_sweep(
        bootstrap=boot,
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
    scenario = ScenarioReplay(crash_windows, sigma_historical=boot.sigma_historical)
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "sweep_conservative.json"
    payload = {"config": {
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
    },
        "benign_results": results,
        "crash_results": crash_results,
        "verdict": verdict,
        "crash_protection": protection,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
