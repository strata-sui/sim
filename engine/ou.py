"""Single-factor Ornstein–Uhlenbeck process — Euler-Maruyama discretization.

    dx_t = θ (μ − x_t) dt + σ dW_t

Discrete update (Higham 2001):
    x_{t+dt} = x_t + θ (μ − x_t) dt + σ √dt · Z_t,    Z_t ~ N(0, 1)

Closed-form moments (used for test pinning):
    E[x_t | x_0]      = μ + (x_0 − μ) · exp(−θ t)
    Var[x_t | x_0]    = σ²/(2θ) · (1 − exp(−2θ t))
    Corr(x_t, x_{t+τ}) = exp(−θ τ)                              (stationary)
    Long-run: x_∞ ~ N(μ, σ²/(2θ))

The simulator accepts EXTERNAL innovations so a higher-level multi-factor
engine can build a correlated Brownian-innovation vector once (via Cholesky)
and feed correlated shocks into several OU processes simultaneously. This
is how S3.2 (multi-factor SVI) and S3.3 (BTC-return leverage coupling) attach
to this primitive without changing it.

Reference:
    Higham, D. J. (2001). "An Algorithmic Introduction to Numerical
    Simulation of Stochastic Differential Equations." SIAM Review 43(3),
    525–546.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OUParams:
    """Ornstein–Uhlenbeck parameters: dx = θ(μ−x)dt + σ dW.

    Attrs:
        theta: mean-reversion speed, > 0. Larger ⇒ faster snap to μ.
        mu:    long-run mean.
        sigma: instantaneous noise scale, > 0.
    """

    theta: float
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.theta <= 0.0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

    @property
    def long_run_variance(self) -> float:
        return self.sigma * self.sigma / (2.0 * self.theta)

    @property
    def long_run_std(self) -> float:
        return float(np.sqrt(self.long_run_variance))


def simulate_ou(
    params: OUParams,
    n_paths: int,
    n_steps: int,
    dt: float = 1.0,
    x0: float | np.ndarray | None = None,
    innovations: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Simulate ``n_paths × (n_steps + 1)`` OU paths via Euler-Maruyama.

    Args:
        params:      OU parameters (validated).
        n_paths:     number of independent paths.
        n_steps:     number of evolution steps; output has n_steps+1 values per
                     path (col 0 = initial x0).
        dt:          time-step size.
        x0:          initial value (scalar or shape (n_paths,)). Defaults to μ.
        innovations: optional (n_paths, n_steps) array of N(0, 1) shocks. If
                     None, drawn from ``np.random.default_rng(seed)``. Used by
                     S3.2/S3.3 to plug in correlated innovations.
        seed:        RNG seed (used only when innovations is None).

    Returns:
        (n_paths, n_steps+1) ndarray of paths.
    """
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be >= 1")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    if innovations is None:
        rng = np.random.default_rng(seed)
        innov = rng.standard_normal((n_paths, n_steps))
    else:
        innov = np.asarray(innovations, dtype=float)
        if innov.shape != (n_paths, n_steps):
            raise ValueError(
                f"innovations shape {innov.shape} != "
                f"expected {(n_paths, n_steps)}"
            )

    x = np.empty((n_paths, n_steps + 1), dtype=float)
    if x0 is None:
        x[:, 0] = params.mu
    elif np.isscalar(x0):
        x[:, 0] = float(x0)
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.shape != (n_paths,):
            raise ValueError(
                f"x0 shape {x0_arr.shape} != expected ({n_paths},)"
            )
        x[:, 0] = x0_arr

    drift_coef = params.theta * dt
    diff_coef = params.sigma * float(np.sqrt(dt))
    # Sequential per-step update — OU is inherently autoregressive in time.
    for t in range(n_steps):
        x[:, t + 1] = (
            x[:, t]
            + drift_coef * (params.mu - x[:, t])
            + diff_coef * innov[:, t]
        )
    return x


def expected_path_mean(
    params: OUParams, x0: float, t_grid: np.ndarray
) -> np.ndarray:
    """Closed-form ``E[x_t | x_0] = μ + (x_0 − μ) · e^{−θ t}`` on a t-grid."""
    t = np.asarray(t_grid, dtype=float)
    return params.mu + (x0 - params.mu) * np.exp(-params.theta * t)


def expected_path_variance(
    params: OUParams, t_grid: np.ndarray
) -> np.ndarray:
    """Closed-form ``Var[x_t | x_0] = σ²/(2θ)·(1 − e^{−2θ t})`` on a t-grid."""
    t = np.asarray(t_grid, dtype=float)
    return params.long_run_variance * (1.0 - np.exp(-2.0 * params.theta * t))
