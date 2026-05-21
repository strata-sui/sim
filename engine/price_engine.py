"""Block-bootstrap price engine (S1 form).

S1 thin-slice form: **fixed block length, wrap-around indexing** over an
empirical log-return series. Produces vectorized paths with realistic
fat tails (sampled from history) and partial volatility-clustering
preservation (within-block returns stay contiguous).

S2 will replace this with:
  * Politis-Romano stationary bootstrap (random block length, geometric
    distribution) — preserves clustering more faithfully.
  * Kou double-exponential jump overlay — calibrated so 1-in-1000 sub-hour
    jumps >= historical worst with conservative inflation.

S1 design rationale (CLAUDE.md §6 task table):
  * Block-length >= 1 preserves the autocorrelation of |r| at lags <
    block_length (volatility clustering).
  * Wrap-around avoids edge bias when (n_blocks · block_length) > N.
  * Vectorized indexing: one ndarray gather per (n_paths × n_blocks).
  * Single seed -> deterministic paths (S6 reproducibility).

Output per call (vectorized over paths):
    log_return         shape (n_paths, n_steps)
    standardized_shock shape (n_paths, n_steps)   = log_return / σ_historical
    price              shape (n_paths, n_steps+1) col 0 = init_price

σ_historical = stdev of input log_returns. Constant for S1; S3 stochastic
SVI engine will couple step-wise vol to an OU process.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BootstrapEngine:
    """Wrap-around fixed-block bootstrap of an empirical log-return series.

    Args:
        log_returns:  1-D ndarray of historical log returns (e.g., 15m BTC).
        block_length: fixed block length. Default 20 ≈ 5h on 15m bars,
                      which captures most intra-day vol clustering.
        seed:         master RNG seed for reproducibility.

    Raises:
        ValueError: invalid inputs (1-D requirement, length, block_length).
    """

    log_returns: np.ndarray
    block_length: int = 20
    seed: int = 0
    # Derived (set in __post_init__):
    _sigma: float = field(init=False, repr=False)
    _n: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.log_returns = np.asarray(self.log_returns, dtype=float)
        if self.log_returns.ndim != 1:
            raise ValueError(
                f"log_returns must be 1-D, got shape {self.log_returns.shape}"
            )
        if self.block_length < 1:
            raise ValueError(
                f"block_length must be >= 1, got {self.block_length}"
            )
        if len(self.log_returns) < self.block_length:
            raise ValueError(
                f"len(log_returns)={len(self.log_returns)} < "
                f"block_length={self.block_length}"
            )
        self._sigma = float(np.std(self.log_returns, ddof=1))
        self._n = len(self.log_returns)

    @property
    def sigma_historical(self) -> float:
        """Stdev of the input log-return series (ddof=1)."""
        return self._sigma

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        init_price: float = 100000.0,
    ) -> dict:
        """Simulate n_paths × n_steps via wrap-around block bootstrap.

        Vectorized: one numpy gather over the (n_paths, n_blocks,
        block_length) index tensor. Memory cost ≈
        ``n_paths · n_steps · 8 bytes`` for the float64 return matrix.

        Args:
            n_paths:    number of independent paths.
            n_steps:    number of return-steps per path.
            init_price: price at step 0. ``price[:, 0] == init_price``.

        Returns:
            dict with keys ``log_return``, ``standardized_shock``, ``price``.
        """
        if n_paths < 1 or n_steps < 1:
            raise ValueError(f"n_paths and n_steps must be >= 1")
        rng = np.random.default_rng(self.seed)

        # Pick enough whole blocks to cover n_steps, then trim.
        n_blocks = (n_steps + self.block_length - 1) // self.block_length

        # (n_paths, n_blocks) random start indices in [0, _n).
        starts = rng.integers(0, self._n, size=(n_paths, n_blocks))

        # Build (n_paths, n_blocks, block_length) index tensor via broadcasting.
        # Wrap-around with modulo so a block that runs off the end loops back.
        offsets = np.arange(self.block_length)
        idx = (starts[:, :, None] + offsets[None, None, :]) % self._n

        # Gather. Result shape (n_paths, n_blocks, block_length).
        blocks = self.log_returns[idx]

        # Flatten time axis and trim to n_steps.
        log_ret = blocks.reshape(n_paths, n_blocks * self.block_length)[:, :n_steps]

        # Standardized shock (constant σ — S3 will animate this).
        std_shock = log_ret / self._sigma

        # Cumulative log-price -> price path.
        log_cumsum = np.cumsum(log_ret, axis=1)
        price = np.empty((n_paths, n_steps + 1), dtype=float)
        price[:, 0] = init_price
        price[:, 1:] = init_price * np.exp(log_cumsum)

        return {
            "log_return": log_ret,
            "standardized_shock": std_shock,
            "price": price,
        }
