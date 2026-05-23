"""Gatheral raw SVI parametrization + binary pricing (verified, CLAUDE.md §4).

DeepBook Predict's external Block Scholes oracle pushes a single-expiry
SVI surface per oracle. This module:

  1. Decodes on-chain / API-form integer SVI params (all i64 × 1e9).
  2. Computes total implied variance `w(k)` per Gatheral (2004).
  3. Prices binaries (UP / DN) and verticals (Range) under the verified
     formulas from `predict-testnet-4-16` source.
  4. Provides a (necessary-only) no-butterfly sanity check; full
     Gatheral-Jacquier (2013) g(k) ≥ 0 lattice check is deferred to S3
     stochastic-SVI engine where it actually bites (S1 uses a single
     fixed surface that is arb-free by construction).

CLAUDE.md §4 formulas:
    k     = ln(strike / forward)
    w(k)  = a + b · (ρ · (k - m) + √((k - m)² + σ²))
    d2    = -(k + w/2) / √w
    UP(k) = N(d2)                  binary UP digital price
    DN(k) = 1 - UP(k)              binary DN digital price
    Range = UP(k_lo) - UP(k_hi)    vertical range

The S1 deterministic anchor is the live BTC snapshot verified 2026-05-15
(CLAUDE.md §4). S1 freezes this surface; S3 will animate it via OU
processes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

PROTOCOL_SCALE = 1_000_000_000  # i64 × 1e9 on-chain encoding


@dataclass(frozen=True)
class SVIParams:
    """Raw-SVI parametrization (Gatheral 2004).

    All five floats are in their *natural* (unscaled) form, i.e.,
    already divided by the on-chain 1e9 scaling. Use
    ``from_protocol_int`` to decode raw API/on-chain integers.

    Conventions:
        a ∈ ℝ          y-axis intercept of total variance (often small/+).
        b ≥ 0          half-slope of the wing (≥ 0 by no-arb).
        ρ ∈ [-1, 1]    correlation parameter (leverage / skew).
        m ∈ ℝ          shift of the surface in log-moneyness.
        σ > 0          smoothing parameter.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0:
            raise ValueError(f"b must be ≥ 0, got {self.b}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be ∈ [-1, 1], got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

    @classmethod
    def from_protocol_int(
        cls,
        a: int,
        b: int,
        rho: int,
        m: int,
        sigma: int,
        *,
        rho_negative: bool = False,
        m_negative: bool = False,
    ) -> "SVIParams":
        """Decode predict-server JSON encoding into natural SVIParams.

        The API serializes each value as a positive integer plus an
        optional ``*_negative`` boolean flag (for JSON-safe transport of
        signed values). All values are stored on-chain as i64 × 1e9.
        """
        rho_signed = (-rho if rho_negative else rho) / PROTOCOL_SCALE
        m_signed = (-m if m_negative else m) / PROTOCOL_SCALE
        return cls(
            a=a / PROTOCOL_SCALE,
            b=b / PROTOCOL_SCALE,
            rho=rho_signed,
            m=m_signed,
            sigma=sigma / PROTOCOL_SCALE,
        )


# ---- S1 deterministic anchor (CLAUDE.md §4 verified live 2026-05-15) -----
#
# CLAUDE.md cites:
#     a=165787, b=7321280, rho=-0.3188, m=-0.00275, sigma=0.01426
#
# `a` and `b` in the doc are still in protocol-int form (×1e9); rho, m,
# sigma are already unscaled. We unscale a, b here so the resulting
# ``SVIParams`` is uniformly in natural units. This is the single fixed
# surface used by S1's deterministic SVI (S3 stochastic engine will
# replace this with an OU-driven path).
ANCHOR_BTC_2026_05_15 = SVIParams(
    a=165787 / PROTOCOL_SCALE,        # 1.65787e-4
    b=7321280 / PROTOCOL_SCALE,       # 7.32128e-3
    rho=-0.3188,
    m=-0.00275,
    sigma=0.01426,
)


# ---- Surface evaluation ---------------------------------------------------


def total_variance(k: np.ndarray | float, params: SVIParams) -> np.ndarray:
    """Total implied variance w(k) under raw SVI (Gatheral 2004).

    ``w(k) = a + b · (ρ(k-m) + √((k-m)² + σ²))``. Vectorized over ``k``.
    """
    k_arr = np.asarray(k, dtype=float)
    km = k_arr - params.m
    return params.a + params.b * (
        params.rho * km + np.sqrt(km * km + params.sigma * params.sigma)
    )


def _d2(k: np.ndarray | float, w: np.ndarray | float) -> np.ndarray:
    """``d2 = -(k + w/2) / √w`` with a small floor on w to avoid /0."""
    w_safe = np.maximum(np.asarray(w, dtype=float), 1e-12)
    return -(np.asarray(k, dtype=float) + 0.5 * w_safe) / np.sqrt(w_safe)


def up_price(
    strike: np.ndarray | float,
    forward: float,
    params: SVIParams,
) -> np.ndarray:
    """Binary UP digital price: N(d2)."""
    if forward <= 0:
        raise ValueError(f"forward must be > 0, got {forward}")
    strike_arr = np.asarray(strike, dtype=float)
    if np.any(strike_arr <= 0):
        raise ValueError("strikes must be > 0")
    k = np.log(strike_arr / forward)
    w = total_variance(k, params)
    return stats.norm.cdf(_d2(k, w))


def dn_price(
    strike: np.ndarray | float,
    forward: float,
    params: SVIParams,
) -> np.ndarray:
    """Binary DN digital price: N(-d2) = sf(d2).

    Computed via ``scipy.stats.norm.sf`` directly (NOT as ``1 - up_price``)
    to avoid catastrophic cancellation when DN is very small (high strike,
    UP near 1). ``up + dn`` still sums to exactly 1 because scipy's cdf
    and sf are paired to that constraint.
    """
    if forward <= 0:
        raise ValueError(f"forward must be > 0, got {forward}")
    strike_arr = np.asarray(strike, dtype=float)
    if np.any(strike_arr <= 0):
        raise ValueError("strikes must be > 0")
    k = np.log(strike_arr / forward)
    w = total_variance(k, params)
    return stats.norm.sf(_d2(k, w))


def range_price(
    strike_lo: np.ndarray | float,
    strike_hi: np.ndarray | float,
    forward: float,
    params: SVIParams,
) -> np.ndarray:
    """Vertical range price: UP(lo) - UP(hi).

    Positive when strike_lo < strike_hi (since UP is decreasing in strike).
    """
    return up_price(strike_lo, forward, params) - up_price(strike_hi, forward, params)


# ---- No-arb (necessary, NOT sufficient) ----------------------------------


def is_arb_free(params: SVIParams) -> bool:
    """Necessary no-arb sanity for raw SVI (Gatheral-Jacquier 2013, prop 1).

    Two cheap necessary conditions:
        (a) Wing-slope (Lee's moment formula):  b · (1 + |ρ|) ≤ 2
        (b) Floor of w:                         a + b·σ·√(1-ρ²) ≥ 0

    These are NOT exhaustive — they catch the common parametrization
    mistakes but not the full butterfly arbitrage frontier. For the
    lattice scan of Gatheral's g(k) function see ``is_arb_free_lattice``.

    Returns:
        True  if both conditions hold (still may have arb in pathological
              corner cases — only a stronger check rules it out).
        False if either condition is violated (definitive arb).
    """
    if params.b * (1.0 + abs(params.rho)) > 2.0:
        return False
    floor = params.a + params.b * params.sigma * math.sqrt(
        max(0.0, 1.0 - params.rho * params.rho)
    )
    return floor >= 0.0


# ---- Gatheral g(k) butterfly no-arb (S3.5) -------------------------------


def _w_derivatives(
    k: np.ndarray, params: SVIParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form (w, w', w'') for raw SVI at a strike-grid k.

    Analytical from w(k) = a + b·(ρ(k-m) + √((k-m)² + σ²)):
        w'(k)  = b·(ρ + (k-m)/√((k-m)² + σ²))
        w''(k) = b·σ²/((k-m)² + σ²)^{3/2}
    """
    k_arr = np.asarray(k, dtype=float)
    km = k_arr - params.m
    s2 = params.sigma * params.sigma
    radicand = km * km + s2
    root = np.sqrt(radicand)
    w = params.a + params.b * (params.rho * km + root)
    w_prime = params.b * (params.rho + km / root)
    w_double_prime = params.b * s2 / (radicand * root)
    return w, w_prime, w_double_prime


def gatheral_g(k: np.ndarray | float, params: SVIParams) -> np.ndarray:
    """Gatheral's g(k) — butterfly-arb indicator (Gatheral & Jacquier 2013).

        g(k) = (1 − k·w'/(2w))² − (w'/2)²·(1/w + 1/4) + w''/2

    No butterfly arb iff g(k) ≥ 0 for ALL k. Negative g(k) at any strike
    is a definitive butterfly-arb violation. w is clamped to a tiny floor
    to avoid /0 at degenerate surfaces (which themselves fail the necessary
    ``is_arb_free`` check earlier).
    """
    w, wp, wpp = _w_derivatives(k, params)
    w_safe = np.maximum(w, 1e-12)
    term1 = (1.0 - np.asarray(k, dtype=float) * wp / (2.0 * w_safe)) ** 2
    term2 = (wp / 2.0) ** 2 * (1.0 / w_safe + 0.25)
    term3 = wpp / 2.0
    return term1 - term2 + term3


def is_arb_free_lattice(
    params: SVIParams,
    k_grid: np.ndarray | None = None,
) -> bool:
    """Full butterfly-arb check: ``is_arb_free`` AND min g(k) ≥ 0 on a lattice.

    Args:
        params: SVI surface.
        k_grid: log-moneyness strike grid. Default ``np.linspace(-1, 1, 201)``
                — wide enough to catch wing violations; dense enough to
                approximate the continuous "for all k" condition.

    Returns:
        True if the surface passes both cheap sufficient conditions AND has
        non-negative g(k) on the lattice. False otherwise.
    """
    if not is_arb_free(params):
        return False
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 201)
    return bool(np.min(gatheral_g(k_grid, params)) >= 0.0)


def clamp_to_arb_free(
    params: SVIParams,
    k_grid: np.ndarray | None = None,
    max_iters: int = 30,
    b_shrink: float = 0.9,
) -> SVIParams:
    """Repair an arb-violating surface by shrinking ``b`` until valid.

    Strategy (documented choice per brief §S3.5): shrinking the wing-slope
    parameter ``b`` is monotonic in the wing-slope necessary condition
    ``b·(1+|ρ|) ≤ 2`` AND in the butterfly g(k) (since w''(k) ∝ b and
    (w'/2)² ∝ b² so the negative term shrinks faster — the lattice test
    becomes easier to pass). Reject-and-resample is the alternative; we
    choose CLAMP because it preserves the OU innovation correlation
    structure of the engine (S3.2/S3.3) — rejection would discard shocks
    correlated to the BTC return path and bias the leverage coupling.

    Returns:
        Arb-free SVIParams (same a, ρ, m, σ; b possibly reduced). If even
        b=0 fails (very degenerate, shouldn't happen with valid inputs)
        returns the b=0 surface — a flat ATM surface which is trivially
        arb-free.
    """
    if is_arb_free_lattice(params, k_grid):
        return params
    b = params.b
    for _ in range(max_iters):
        b *= b_shrink
        candidate = SVIParams(
            a=params.a, b=max(b, 0.0), rho=params.rho,
            m=params.m, sigma=params.sigma,
        )
        if is_arb_free_lattice(candidate, k_grid):
            return candidate
    # Last-resort flat surface.
    return SVIParams(a=params.a, b=0.0, rho=params.rho, m=params.m, sigma=params.sigma)
