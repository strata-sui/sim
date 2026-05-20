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
from engine.predict_server import PredictServerClient, PredictServerError
from engine.resample import log_returns, resample_klines
from engine.sanity import gate_data_sanity
from engine.svi_history import pull_btc_svi_summary

# ---- S0 config (per CLAUDE.md §5) ---------------------------------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

# Covers regime variety + named crashes (LUNA May-2022, FTX Nov-2022) and
# extends through latest complete month on Binance Vision (Apr 2026 as of
# this commit). The loader gracefully skips months not yet published.
START = "2022-01"
END = "2026-04"

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

    if not result["pass"]:
        print("[S0] GATE FAILED.")
        print("     Check data resolution / source / range before proceeding to S1.")
        return 1

    print("[S0] GATE PASSED — data is fat-tailed and shows clustering.")
    print()

    # Pull SVI history from predict-server testnet for S3 OU calibration anchor.
    # Non-blocking: failure is logged but doesn't fail S0.
    print("[svi] fetching SVI history from predict-server testnet")
    try:
        client = PredictServerClient()
        srv_status = client.status()
        print(f"      server status: {srv_status}")
        summary = pull_btc_svi_summary()
        print(f"      BTC oracle: {summary['oracle_id']}")
        print(f"        status   = {summary['oracle_status']}")
        print(f"        expiry   = {summary['expiry']}")
        print(f"        snapshots= {summary['history_len']:,}")
        print(f"        cached   = {summary['cache_path']}")
        if summary["latest"] is not None:
            print(f"        latest   = {summary['latest']}")
    except PredictServerError as e:
        print(f"      ⚠ SVI fetch failed (non-blocking): {e}")
    except Exception as e:
        print(f"      ⚠ SVI fetch error (non-blocking): {e!r}")

    print()
    print("     Ready for S1 (Thin Slice / FAST GATE A).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
