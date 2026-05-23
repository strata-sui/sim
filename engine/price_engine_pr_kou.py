"""Politis–Romano stationary block bootstrap + Kou jump overlay (S2).

Drop-in replacement for ``engine.price_engine.BootstrapEngine`` (same
``.simulate(n_paths, n_steps, init_price)`` signature, same output dict
``{log_return, standardized_shock, price}``, same ``.sigma_historical``
property), so ``s1.py`` / ``run_sweep`` / Gate-A / f*(w_crash) all keep
running unchanged at the caller boundary.

Two layers (CLAUDE.md §5 simulator design; brief §S2.1, §S2.2):

  1. **Politis–Romano stationary block bootstrap** (Politis & Romano, 1994).
     Random block length L ~ Geometric(p) with p = 1/mean_block_length.
     Wrap-around. Kills the fixed-block periodicity artifact in S1's
     ``BootstrapEngine`` and gives true stationary resampling.

  2. **Kou double-exponential jump overlay** (Kou, 2002). Compound-Poisson
     jumps with intensity ``jump_intensity`` per step; jump-size is
     asymmetric double-exponential (down-jump scale ≥ up-jump scale by
     construction). Adds tail mass for stress scenarios beyond historical.

Vectorization (no per-path Python loop): the bootstrap uses a
restart-decision tensor + ``np.maximum.accumulate`` to vectorize the
"most-recent restart" lookup; the Kou overlay is fully ndarray.

Reference (Politis–Romano):
    Politis, D. N. and Romano, J. P. (1994). "The Stationary Bootstrap."
    Journal of the American Statistical Association 89(428): 1303–1313.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PolitisRomanoKouEngine:
    """Stationary block bootstrap + (optional) Kou jump overlay.

    Args:
        log_returns:        1-D ndarray of historical log returns (e.g. 15m BTC).
        mean_block_length:  expected block length (bars). Geometric distribution
                            ⇒ p_restart = 1 / mean_block_length. Default 4
                            matches the S1 fixed-block baseline.
        jump_intensity:     Poisson rate of Kou jumps per step. Default 1e-4
                            (≈1 jump per 10k bars). Calibration anchor:
                            historically (BTCUSDT 2020-01..2026-04, ~221k
                            15m bars), the top-1-in-10k bars are the
                            “extreme” cluster — single-bar |move| > ~5%
                            (Black Thursday, LUNA, FTX, etc.) — so the
                            jump frequency is anchored to OBSERVED extreme-
                            bar rate, NOT tuned for Sortino. Set 0.0 to
                            disable Kou and run pure Politis–Romano.
        jump_prob_down:     fraction of jumps that are negative. Default 0.65
                            (slight crash bias consistent with crypto leverage
                            effect; not tuned for Sortino).
        down_jump_scale:    mean magnitude of down-jumps. Default 0.12 ≈
                            historical worst 15m bar (−12.7%); rationale:
                            anchor scale to observed extreme, NOT to outcome.
        up_jump_scale:      mean magnitude of up-jumps. Default 0.09 → ratio
                            0.12/0.09 = 1.33× ≥ the brief's 1.2× asymmetry floor.
        seed:               master RNG seed.
    """

    log_returns: np.ndarray
    mean_block_length: float = 4.0
    # S2.2 default: Kou ON, anchored to historical extreme-bar rate.
    #   λ = 1e-4  (~1 in 10k bars), p_down = 0.65 (leverage tilt),
    #   1/α_down = 0.12 (mean down-jump ≈ historical worst 15m bar),
    #   1/α_up   = 0.09 (asymmetry 0.12/0.09 = 1.33× ≥ brief's 1.2× floor).
    # Anchor = historical extremes + conservative inflation. NOT tuned to Sortino.
    jump_intensity: float = 1.0e-4
    jump_prob_down: float = 0.65
    down_jump_scale: float = 0.12
    up_jump_scale: float = 0.09
    seed: int = 0
    _sigma: float = field(init=False, repr=False)
    _n: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.log_returns = np.asarray(self.log_returns, dtype=float)
        if self.log_returns.ndim != 1:
            raise ValueError(
                f"log_returns must be 1-D, got shape {self.log_returns.shape}"
            )
        if self.mean_block_length < 1.0:
            raise ValueError(
                f"mean_block_length must be >= 1, got {self.mean_block_length}"
            )
        if self.jump_intensity < 0.0:
            raise ValueError(
                f"jump_intensity must be >= 0, got {self.jump_intensity}"
            )
        if not 0.0 <= self.jump_prob_down <= 1.0:
            raise ValueError(
                f"jump_prob_down ∈ [0,1] required, got {self.jump_prob_down}"
            )
        if self.down_jump_scale <= 0 or self.up_jump_scale <= 0:
            raise ValueError("jump scales must be > 0")
        if self.down_jump_scale < 1.2 * self.up_jump_scale:
            raise ValueError(
                f"asymmetry guardrail: down_jump_scale "
                f"({self.down_jump_scale}) must be >= 1.2 * up_jump_scale "
                f"({self.up_jump_scale}) per brief §S2.2"
            )
        self._sigma = float(np.std(self.log_returns, ddof=1))
        self._n = int(len(self.log_returns))

    @property
    def sigma_historical(self) -> float:
        return self._sigma

    # ---- Layer 1: Politis–Romano stationary block bootstrap ----------

    def _bootstrap_indices(
        self, rng: np.random.Generator, n_paths: int, n_steps: int
    ) -> np.ndarray:
        """Vectorized stationary block bootstrap index tensor.

        Per-step restart probability p = 1 / mean_block_length. At each step
        a Bernoulli(p) trial decides "restart" (new random start index) vs
        "continue" (advance previous index by 1). Geometric block length by
        construction. Wrap-around with modulo.

        Returns:
            (n_paths, n_steps) int index array into ``log_returns``.
        """
        p_restart = 1.0 / float(self.mean_block_length)
        restart = rng.random((n_paths, n_steps)) < p_restart
        restart[:, 0] = True  # always restart at t=0
        restart_starts = rng.integers(0, self._n, size=(n_paths, n_steps))
        step_idx = np.tile(np.arange(n_steps), (n_paths, 1))
        restart_at_step = np.where(restart, step_idx, -1)
        # Cumulative max gives the most-recent True step index per row.
        active_step = np.maximum.accumulate(restart_at_step, axis=1)
        active_starts = np.take_along_axis(restart_starts, active_step, axis=1)
        offsets = step_idx - active_step
        return (active_starts + offsets) % self._n

    # ---- Layer 2: Kou double-exponential jump overlay ----------------

    def _kou_jumps(
        self, rng: np.random.Generator, shape: tuple
    ) -> np.ndarray:
        """Compound-Poisson asymmetric double-exponential jumps.

        Per step: Bernoulli(jump_intensity) selects bars that have a jump.
        Sign is Bernoulli(jump_prob_down); magnitude is Exponential with the
        signed scale. Returns an additive-on-log-scale shock array.
        """
        if self.jump_intensity <= 0.0:
            return np.zeros(shape, dtype=float)
        jump_mask = rng.random(shape) < self.jump_intensity
        is_down = rng.random(shape) < self.jump_prob_down
        mag_down = rng.exponential(scale=self.down_jump_scale, size=shape)
        mag_up = rng.exponential(scale=self.up_jump_scale, size=shape)
        signed = np.where(is_down, -mag_down, mag_up)
        return np.where(jump_mask, signed, 0.0)

    # ---- Public: simulate paths --------------------------------------

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        init_price: float = 100_000.0,
    ) -> dict:
        """Simulate n_paths × n_steps using PR-bootstrap + Kou overlay."""
        if n_paths < 1 or n_steps < 1:
            raise ValueError("n_paths and n_steps must be >= 1")
        rng = np.random.default_rng(self.seed)

        idx = self._bootstrap_indices(rng, n_paths, n_steps)
        log_ret = self.log_returns[idx]
        log_ret = log_ret + self._kou_jumps(rng, (n_paths, n_steps))

        std_shock = log_ret / self._sigma
        log_cumsum = np.cumsum(log_ret, axis=1)
        price = np.empty((n_paths, n_steps + 1), dtype=float)
        price[:, 0] = init_price
        price[:, 1:] = init_price * np.exp(log_cumsum)
        return {
            "log_return": log_ret,
            "standardized_shock": std_shock,
            "price": price,
        }
