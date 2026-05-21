"""Binance BTCUSDT 1m kline loader (data.binance.vision).

Reads kline CSVs from the public Binance Vision monthly dumps. Supports
auto-download into the local data/ folder. No auth required.

Binance 1m CSV columns (12, in order):
    open_time, open, high, low, close, volume, close_time,
    quote_asset_volume, num_trades, taker_buy_base_volume,
    taker_buy_quote_volume, ignore

Reference: https://github.com/binance/binance-public-data
"""
from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _month_url(symbol: str, interval: str, year: int, month: int) -> str:
    return (
        f"{BINANCE_VISION_BASE}/{symbol}/{interval}/"
        f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    )


def _month_zip_path(
    symbol: str, interval: str, year: int, month: int, data_dir: Path
) -> Path:
    return data_dir / f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"


def download_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    data_dir: Optional[Path] = None,
) -> Path:
    """Download a single month's kline zip. Idempotent (skips if file exists).

    Raises FileNotFoundError if Binance Vision returns 404 (month not yet
    published — e.g. current month before close, or future month). Cleans up
    any partial file before re-raising.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = _month_zip_path(symbol, interval, year, month, data_dir)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = _month_url(symbol, interval, year, month)
    print(f"  ↓ {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as e:
        if dest.exists():
            dest.unlink()  # cleanup partial
        if e.code == 404:
            raise FileNotFoundError(
                f"Binance Vision 404 for {url} — month likely not yet published"
            ) from e
        raise
    return dest


def load_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    data_dir: Optional[Path] = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load a single month's klines as a tz-aware DataFrame indexed by open_time."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    zpath = _month_zip_path(symbol, interval, year, month, data_dir)
    if not zpath.exists():
        if not auto_download:
            raise FileNotFoundError(zpath)
        download_month(symbol, interval, year, month, data_dir)

    with zipfile.ZipFile(zpath) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)

    # Binance Vision switched kline timestamp unit from ms → µs in 2025-01.
    # Auto-detect by magnitude (ms ≈ 1e12-1e13, µs ≈ 1e15-1e16) so the same
    # loader handles pre-2025 and post-2025 dumps uniformly. Failing to do
    # this parses µs values as ms → timestamps land in year ~57000 → any
    # resample tries to span 55k years → 14 GiB OOM in pandas binner.
    sample_ts = int(df["open_time"].iloc[0])
    ts_unit = "us" if sample_ts > 10**14 else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=ts_unit, utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit=ts_unit, utc=True)
    df = df.set_index("open_time").sort_index()
    return df[
        ["open", "high", "low", "close", "volume", "num_trades"]
    ].astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "num_trades": int,
        }
    )


def load_range(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    data_dir: Optional[Path] = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load all months inclusive between start and end (YYYY-MM strings).

    Example: load_range("BTCUSDT", "1m", "2022-01", "2024-12")
    """
    s_year, s_month = map(int, start.split("-"))
    e_year, e_month = map(int, end.split("-"))
    if (s_year, s_month) > (e_year, e_month):
        raise ValueError(f"start {start} is after end {end}")

    frames = []
    y, m = s_year, s_month
    while (y, m) <= (e_year, e_month):
        try:
            frames.append(
                load_month(symbol, interval, y, m, data_dir, auto_download)
            )
        except FileNotFoundError as e:
            # Month not yet published on Binance Vision. Stop iterating —
            # subsequent months won't exist either.
            print(f"  ⚠ skipping {y:04d}-{m:02d}: {e}")
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
    if not frames:
        raise RuntimeError(
            f"No data loaded for {symbol} {interval} in {start}..{end}"
        )
    return pd.concat(frames).sort_index()
