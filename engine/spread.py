"""DeepBook Predict spread formula (verified, CLAUDE.md §4).

The protocol has NO protocol fee — the spread IS pool revenue. Per-contract
spread on a binary UP/DN mint, from `predict-testnet-4-16` source:

    spread(p, util) = max(base_spread · √(p·(1-p)), min_spread)
                    + base_spread · util_mult · util²

Where
    p     ∈ (0, 1)              fair binary price (DN or UP) from SVI surface
    util  ∈ [0, max_exposure]   pool inventory utilization

Defaults (verified live 2026-05-15):
    base_spread  = 0.02     (2%)
    min_spread   = 0.005    (0.5%)
    util_mult    = 2.0
    max_exposure = 0.80
    ask_lo, ask_hi = 0.01, 0.99    binary ask-price bounds

Term decomposition (drives Strata thesis numerics):
    Term-1 (house edge): max(base · √(p(1-p)), min)
        · √(p(1-p)) maxes at p=0.5 → 0.5; so term-1 maxes at base · 0.5 = 1%.
        · At deep-OTM (p→0 or p→1), √(p(1-p)) → 0; floor min binds at 0.5%.
        · Per-trade house edge ∈ [0.5%, 1.0%].
    Term-2 (inventory premium): base · util_mult · util²
        · Convex in util. At util=0.80 (max_exposure): 0.02·2·0.64 = 2.56%.

PLP-leg revenue: per-contract expected gross = +spread (pool is structurally
short-vol/short-gamma counterparty). This module computes spread only;
P&L accounting lives in `model/plp.py`.

Effective ask price (one-sided binary, pool sells to trader):
    ask = clip(mid + spread, ask_lo, ask_hi)
"""
from __future__ import annotations

import numpy as np

# ---- Verified protocol defaults (CLAUDE.md §4) ---------------------------
BASE_SPREAD_DEFAULT = 0.02
MIN_SPREAD_DEFAULT = 0.005
UTIL_MULT_DEFAULT = 2.0
MAX_EXPOSURE_DEFAULT = 0.80
ASK_LO_DEFAULT = 0.01
ASK_HI_DEFAULT = 0.99


def spread_per_contract(
    p: np.ndarray | float,
    util: np.ndarray | float,
    base_spread: float = BASE_SPREAD_DEFAULT,
    min_spread: float = MIN_SPREAD_DEFAULT,
    util_mult: float = UTIL_MULT_DEFAULT,
) -> np.ndarray:
    """Per-contract spread on a binary UP/DN mint.

    Vectorized: ``p`` and ``util`` broadcast against each other.

    Args:
        p:           fair binary price ∈ (0, 1). Scalar or ndarray.
        util:        pool inventory utilization ∈ [0, max_exposure]. Scalar
                     or ndarray.
        base_spread: house-edge scaling (default 2%).
        min_spread:  house-edge floor (default 0.5%).
        util_mult:   inventory-premium multiplier (default 2.0).

    Returns:
        Spread (dimensionless), same shape as broadcast(p, util). Always
        non-negative; lower bound = min_spread (at util=0).
    """
    p_arr = np.asarray(p, dtype=float)
    u_arr = np.asarray(util, dtype=float)
    house_edge = np.maximum(
        base_spread * np.sqrt(p_arr * (1.0 - p_arr)),
        min_spread,
    )
    inventory_premium = base_spread * util_mult * u_arr * u_arr
    return house_edge + inventory_premium


def ask_price(
    mid: np.ndarray | float,
    util: np.ndarray | float,
    ask_lo: float = ASK_LO_DEFAULT,
    ask_hi: float = ASK_HI_DEFAULT,
    **spread_kwargs,
) -> np.ndarray:
    """Effective ask price for a binary contract.

    ``ask = clip(mid + spread, ask_lo, ask_hi)``. Pool is the sole
    counterparty so the trader pays the full spread (one-sided market).

    Args:
        mid:           fair binary price (UP or DN) ∈ (0, 1).
        util:          pool utilization ∈ [0, max_exposure].
        ask_lo, ask_hi: clipping bounds (defaults 1% / 99% per §4).
        **spread_kwargs: forwarded to ``spread_per_contract`` (e.g. to
                         override base_spread for sensitivity sweeps).

    Returns:
        Clipped ask price, same shape as broadcast(mid, util).
    """
    mid_arr = np.asarray(mid, dtype=float)
    s = spread_per_contract(mid_arr, util, **spread_kwargs)
    return np.clip(mid_arr + s, ask_lo, ask_hi)
