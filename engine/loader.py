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
    """Download a single month's kline zip. Idempotent (skips if file exists)."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = _month_zip_path(symbol, interval, year, month, data_dir)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = _month_url(symbol, interval, year, month)
    print(f"  ↓ {url}")
    urllib.request.urlretrieve(url, dest)
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

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
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
        frames.append(
            load_month(symbol, interval, y, m, data_dir, auto_download)
        )
        m += 1
        if m > 12:
            m = 1
            y += 1
    return pd.concat(frames).sort_index()
