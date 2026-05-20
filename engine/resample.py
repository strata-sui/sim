"""Resample 1-minute kline data to coarser timesteps.

Uses pandas .resample with OHLCV semantics:
    open  = first
    high  = max
    low   = min
    close = last
    volume / num_trades = sum

Per CLAUDE.md §5: `step_minutes` is a config parameter (default 15m for Gate-A
validation). The true DeepBook Predict expiry cadence is a per-oracle runtime
parameter — once verified live, re-run with the matching step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "num_trades": "sum",
}


def resample_klines(df_1m: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    """Resample 1m klines to step_minutes klines.

    Args:
        df_1m: DataFrame indexed by tz-aware datetime, columns
               open/high/low/close/volume/num_trades.
        step_minutes: target resolution in minutes. Must be >=1.

    Returns:
        DataFrame at the coarser resolution, NaN bars dropped.
    """
    if step_minutes < 1:
        raise ValueError(f"step_minutes must be >= 1, got {step_minutes}")
    rule = f"{step_minutes}min"
    return df_1m.resample(rule).agg(_OHLCV_AGG).dropna()


def log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Compute log returns from a price column.

    Returns a Series of length len(df)-1, indexed at the close-time
    of each return (i.e., shifted forward by one bar).
    """
    prices = df[price_col].values
    if len(prices) < 2:
        raise ValueError("Need at least 2 bars to compute returns")
    r = np.log(prices[1:] / prices[:-1])
    return pd.Series(r, index=df.index[1:], name="log_return")
