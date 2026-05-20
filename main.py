"""Strata sim — S0 orchestrator.

Runs the S0 data pipeline:
    1. Load BTC 1m klines from Binance Vision (auto-downloads to sim/data/)
    2. Resample to `STEP_MINUTES` (default 15m, matches Gate-A config)
    3. Compute log returns
    4. Run data-sanity gate (fat tails + volatility clustering)

Usage:
    uv run python main.py
"""
from __future__ import annotations

from engine.loader import load_range
from engine.resample import log_returns, resample_klines
from engine.sanity import gate_data_sanity

# ---- S0 config (per CLAUDE.md §5) ---------------------------------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

# Covers regime variety + named crashes (LUNA May-2022, FTX Nov-2022).
# Extend later when more history is needed.
START = "2022-01"
END = "2024-12"

# CLAUDE.md §5 default; true cadence is per-oracle runtime, verify live later.
STEP_MINUTES = 15


def main() -> int:
    print(f"[S0] Strata data pipeline")
    print(f"     symbol={SYMBOL} interval={INTERVAL} range={START}..{END}")
    print(f"     resample step={STEP_MINUTES}m")
    print()

    print(f"[load] {SYMBOL} {INTERVAL} {START}..{END}")
    df_1m = load_range(SYMBOL, INTERVAL, START, END)
    print(f"       {len(df_1m):,} 1m bars loaded")

    print(f"[resample] -> {STEP_MINUTES}m")
    df_res = resample_klines(df_1m, STEP_MINUTES)
    print(f"           {len(df_res):,} bars")

    print(f"[returns] log returns")
    r = log_returns(df_res)
    print(f"          {len(r):,} returns, std={r.std():.6f}")

    print(f"[gate] data sanity")
    result = gate_data_sanity(r)
    s = result["stats"]
    print(f"       kurtosis_excess = {s['kurtosis_excess']:>8.2f}    (need >= 5)   "
          f"-> fat_tail_ok = {result['fat_tail_ok']}")
    print(f"       acf|r|[10]      = {result['acf_abs'][9]:>8.3f}    (need >= 0.05) "
          f"-> clustering_ok = {result['clustering_ok']}")
    print(f"       worst bar       = {s['min']:>8.4f}")
    print(f"       best bar        = {s['max']:>8.4f}")
    print()

    if result["pass"]:
        print("[S0] GATE PASSED — data is fat-tailed and shows clustering.")
        print("     Ready for S1 (Thin Slice / FAST GATE A).")
        return 0
    else:
        print("[S0] GATE FAILED.")
        print("     Check data resolution / source / range before proceeding to S1.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
