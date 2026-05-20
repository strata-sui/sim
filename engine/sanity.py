"""S0 data-sanity gate for the BTC return series.

For Gate-A validity, two empirical properties must hold on the historical
return series BEFORE the simulator can claim fat-tail-bootstrap calibration:

  1. Fat tails  — excess kurtosis > 3 (crypto sub-hour typically 5–30).
  2. Volatility clustering — ACF of |r| at lag 10 > ~0.05 (slow decay).

If either fails: STOP. Either the data resolution is wrong, the source is
corrupted, or the wrong asset/range was loaded. Fixing this is cheaper
than discovering it inside S5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def basic_stats(returns: pd.Series) -> dict:
    """Distributional summary of a returns series."""
    arr = returns.dropna().values
    if len(arr) == 0:
        raise ValueError("empty return series")
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "skew": float(stats.skew(arr)),
        "kurtosis_excess": float(stats.kurtosis(arr)),  # excess (Normal = 0)
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p99": float(np.percentile(arr, 99)),
    }


def acf_abs_returns(returns: pd.Series, lags: int = 20) -> np.ndarray:
    """ACF of |r - mean(|r|)| at lags 1..lags (no lag 0).

    Slow decay confirms volatility clustering — needed to justify block
    bootstrap in S2. IID-like ACF (rapid decay to 0) means the bootstrap
    will under-state sustained drawdowns.
    """
    r = returns.dropna().values
    a = np.abs(r) - np.mean(np.abs(r))
    n = len(a)
    var = float(np.dot(a, a) / n)
    if var == 0.0:
        return np.zeros(lags)
    out = np.empty(lags)
    for k in range(1, lags + 1):
        out[k - 1] = float(np.dot(a[:n - k], a[k:]) / ((n - k) * var))
    return out


def gate_data_sanity(
    returns: pd.Series,
    kurt_min: float = 5.0,
    acf_lag: int = 10,
    acf_min: float = 0.05,
) -> dict:
    """S0 gate. Returns a dict with `pass` and the underlying stats.

    Args:
        returns: log return series.
        kurt_min: minimum acceptable excess kurtosis. 5 is a conservative
                  floor for crypto sub-hour (typical 10–30).
        acf_lag: ACF lag to check (default 10).
        acf_min: minimum ACF(|r|) at that lag.

    Returns:
        dict {
            "pass": bool,
            "fat_tail_ok": bool,
            "clustering_ok": bool,
            "stats": {...},
            "acf_abs": [acf_1, acf_2, ..., acf_20],
        }
    """
    s = basic_stats(returns)
    acf = acf_abs_returns(returns, lags=max(20, acf_lag))
    fat_tail_ok = s["kurtosis_excess"] >= kurt_min
    clustering_ok = float(acf[acf_lag - 1]) >= acf_min
    return {
        "pass": bool(fat_tail_ok and clustering_ok),
        "fat_tail_ok": bool(fat_tail_ok),
        "clustering_ok": bool(clustering_ok),
        "stats": s,
        "acf_abs": acf.tolist(),
        "config": {
            "kurt_min": kurt_min,
            "acf_lag": acf_lag,
            "acf_min": acf_min,
        },
    }
