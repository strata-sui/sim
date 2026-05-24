"""S5R3.2 diagnostic — trace per-cycle accounting on a pathological seed.

Hunts for seeds in [42, 42+200) that produce the most pathological
multi-cycle accounting under run_strategy_s4 on PR+Kou paths, then
dumps the cycle-by-cycle state to localize the blow-up.

Hypothesis (to confirm): the PLP balance can go arbitrarily negative
because the pool pays out all ITM contracts on settle even when the
cumulative payout exceeds the pool's balance. Real-world this would
default; in-sim we just let balance → -∞. Then share_price (nav/shares)
becomes massively negative, and LP-leg PnL = shares × (sp − 1) drives
strata_pnl_total to physically meaningless trillions on a $50k deposit.

Run via:
    uv run python sim/scripts/diag_s5_blowup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.loader import load_range
from engine.price_engine_pr_kou import PolitisRomanoKouEngine
from engine.resample import log_returns, resample_klines
from engine.svi_det import ANCHOR_BTC_2026_05_15
from eval.f_sweep import _rolling_vol
from eval.strategy import RAW_PLP, STRATA, run_strategy_s4
from model.dn_ladder import DEFAULT_MONEYNESS_HI, DEFAULT_MONEYNESS_LO
from model.trader_flow import CONSERVATIVE_ANCHOR

N_CYCLES = 10
PATH_STEPS = 4
TOTAL_STEPS = N_CYCLES * PATH_STEPS
INIT_PRICE = 60_000.0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[load] BTCUSDT 15m 2020-01..2026-04")
    df = load_range("BTCUSDT", "1m", "2020-01", "2026-04")
    r = log_returns(resample_klines(df, 15)).values
    pe = PolitisRomanoKouEngine(log_returns=r, mean_block_length=4.0, seed=42)

    sim = pe.simulate(n_paths=200, n_steps=TOTAL_STEPS, init_price=INIT_PRICE)
    price_paths = sim["price"]
    log_paths = sim["log_return"]
    sigma_long_run = pe.sigma_historical
    realized_vols = _rolling_vol(log_paths, window=8, fallback=sigma_long_run)

    f = 0.05
    strata_capital = f * 1_000_000.0
    other_lp = (1.0 - f) * 1_000_000.0

    # Find worst seed (raw_plp; the brief's diagnostic).
    pnls = np.empty(200, dtype=float)
    for i in range(200):
        out = run_strategy_s4(
            strategy=RAW_PLP,
            strata_capital=strata_capital,
            other_lp_initial=other_lp,
            price_path=price_paths[i],
            log_returns=log_paths[i],
            realized_vols=realized_vols[i],
            sigma_long_run=sigma_long_run,
            svi_params=ANCHOR_BTC_2026_05_15,
            anchor=CONSERVATIVE_ANCHOR,
            n_cycles=N_CYCLES, path_steps=PATH_STEPS,
            ladder_size=5,
            moneyness_lo=DEFAULT_MONEYNESS_LO,
            moneyness_hi=DEFAULT_MONEYNESS_HI,
            kappa_alpha=2.0, kappa_beta=5.0,
            bucket_capacity_factor=1.0,
            bucket_refill_factor=1.0,
            seed=42 + i,
        )
        pnls[i] = out["strata_pnl_total"]

    worst_idx = int(pnls.argmin())
    worst_pnl = float(pnls[worst_idx])
    print(f"\n[scan] raw_plp 200 seeds at f={f:.2f}, strata_capital=${strata_capital:,.0f}")
    print(f"       worst seed = {worst_idx} (path idx, not the +42 offset)")
    print(f"       worst PnL  = ${worst_pnl:,.2f}")
    print(f"       deposit    = ${strata_capital:,.2f}")
    print(f"       ratio      = {worst_pnl / strata_capital:.2f}× of deposit")
    print(f"       pnls.mean  = ${float(pnls.mean()):,.2f}")
    print(f"       pnls.min   = ${float(pnls.min()):,.2f}")
    print(f"       pnls.max   = ${float(pnls.max()):,.2f}")

    if worst_pnl > -strata_capital:
        print(f"\n[OK?] Worst PnL is sane (≥ -deposit). No blow-up at this scan.")
        return 0

    # Drill into the pathological path.
    print(f"\n[drill] tracing cycle-by-cycle accounting on worst seed")
    out = run_strategy_s4(
        strategy=RAW_PLP,
        strata_capital=strata_capital,
        other_lp_initial=other_lp,
        price_path=price_paths[worst_idx],
        log_returns=log_paths[worst_idx],
        realized_vols=realized_vols[worst_idx],
        sigma_long_run=sigma_long_run,
        svi_params=ANCHOR_BTC_2026_05_15,
        anchor=CONSERVATIVE_ANCHOR,
        n_cycles=N_CYCLES, path_steps=PATH_STEPS,
        ladder_size=5,
        moneyness_lo=DEFAULT_MONEYNESS_LO,
        moneyness_hi=DEFAULT_MONEYNESS_HI,
        kappa_alpha=2.0, kappa_beta=5.0,
        bucket_capacity_factor=1.0,
        bucket_refill_factor=1.0,
        seed=42 + worst_idx,
    )
    print(f"  total_pnl = ${out['strata_pnl_total']:,.2f}")
    print(f"  lp_leg    = ${out['lp_leg_pnl']:,.2f}")
    print(f"  hedge_leg = ${out['hedge_leg_pnl']:,.2f}")
    print(f"  final_sp  = {out['final_share_price']:.6f}  (init=1.0)")
    print(f"  per-cycle:")
    print(f"    {'cyc':>3} {'settle_px':>10} {'sp_after':>12} {'f_after':>9}")
    for c in out["cycles"]:
        print(
            f"    {c['cycle_index']:>3} {c['settle_price']:>10.1f} "
            f"{c['share_price_after_cycle']:>12.4f} {c['f_after']:>9.4f}"
        )

    # Price-path summary
    p = price_paths[worst_idx]
    print(f"\n  price path: open={p[0]:.0f} close={p[-1]:.0f}  "
          f"min={p.min():.0f}  max={p.max():.0f}  "
          f"return={p[-1]/p[0] - 1.0:+.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
