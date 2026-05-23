"""DN ladder shaped to the empirical PLP loss-onset band (S4.1).

Replaces the S1 single-strike OTM-DN hedge with a ladder of M strikes
distributed across the band where the PLP actually loses materially.

Matching algorithm (anti-cherry-pick, anchored to data NOT Sortino):

  * Strikes are uniform in log-moneyness across [m_lo, m_hi] where
        m_lo = 1 + p_PLP_0.1   (deeper loss-onset, ~1-in-1000 PLP tail)
        m_hi = 1 + p_PLP_1     (shallow loss-onset, ~1-in-100 PLP tail)
    On the 4-step (1h) bootstrap of BTCUSDT 2020-01..2026-04 the
    empirical PLP loss distribution gives p1 = -1.89% and p0.1 = -4.05%
    (S0 diagnostic turn — committed before any Sortino was computed).
    The ladder therefore covers EXACTLY the region where the PLP starts
    losing materially. NOT tuned to flatter Sortino.

  * Per-strike notional: UNIFORM across the ladder (notional_k =
    sleeve_budget / M / ask_k). Uniform-notional is the simplest
    defensible default; a sophisticated max_payout-prefix-density match
    is an S5 polish item. The brief allows "uniform in log-moneyness" as
    a principled choice.

API (frozen dataclass):
    DnLadder
        strikes              (M,) USD strikes
        moneyness            (M,) strike / forward
        notional_per_strike  (M,) contracts (= $1 payoff per contract if ITM)
        dn_mid_per_strike    (M,) fair DN price from the SVI surface
        ask_per_strike       (M,) effective per-contract ask (= mid + spread)
        premium_per_strike   (M,) premium paid per strike
        total_premium_paid   sum of premium_per_strike

    size_ladder(...)         builds a ladder for a target sleeve budget.

Matched single-strike behaviour: M=1 collapses to the S1 hedge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.spread import ask_price
from engine.svi_det import SVIParams, dn_price


# Default loss-onset band — verbatim from the S0 diagnostic (1h bootstrap of
# 2020-01..2026-04 BTCUSDT 15m). The band is pinned BEFORE Sortino was ever
# computed, so this is anti-cherry-pick by construction.
DEFAULT_MONEYNESS_LO = 1.0 - 0.0405   # PLP p0.1 (deeper, 1-in-1000 tail)
DEFAULT_MONEYNESS_HI = 1.0 - 0.0189   # PLP p1   (shallower, 1-in-100 tail)
DEFAULT_LADDER_SIZE = 5


@dataclass(frozen=True)
class DnLadder:
    """Frozen snapshot of an M-strike DN-binary ladder."""

    strikes: np.ndarray
    moneyness: np.ndarray
    notional_per_strike: np.ndarray
    dn_mid_per_strike: np.ndarray
    ask_per_strike: np.ndarray
    premium_per_strike: np.ndarray
    total_premium_paid: float

    def __post_init__(self) -> None:
        if self.strikes.ndim != 1:
            raise ValueError("strikes must be 1-D")
        m = len(self.strikes)
        for name, arr in [
            ("moneyness", self.moneyness),
            ("notional_per_strike", self.notional_per_strike),
            ("dn_mid_per_strike", self.dn_mid_per_strike),
            ("ask_per_strike", self.ask_per_strike),
            ("premium_per_strike", self.premium_per_strike),
        ]:
            if arr.shape != (m,):
                raise ValueError(
                    f"{name} shape {arr.shape} != strikes shape ({m},)"
                )

    @property
    def n_strikes(self) -> int:
        return len(self.strikes)

    @property
    def total_notional(self) -> float:
        return float(np.sum(self.notional_per_strike))

    def itm_mask(self, settle_price: float) -> np.ndarray:
        """Boolean (M,) mask: True where strike > settle (DN binary ITM)."""
        return settle_price < self.strikes

    def payoff(self, settle_price: float) -> float:
        """Total USD payoff from the ladder at settlement.

        Each ITM strike pays its full notional (1 contract = $1).
        """
        return float(np.sum(self.notional_per_strike[self.itm_mask(settle_price)]))


def size_ladder(
    sleeve_budget: float,
    forward: float,
    svi_params: SVIParams,
    util: float = 0.0,
    n_strikes: int = DEFAULT_LADDER_SIZE,
    moneyness_lo: float = DEFAULT_MONEYNESS_LO,
    moneyness_hi: float = DEFAULT_MONEYNESS_HI,
    base_spread: float | None = None,
    min_spread: float | None = None,
    util_mult: float | None = None,
) -> DnLadder:
    """Build the M-strike ladder for a target premium budget.

    Args:
        sleeve_budget:  total USD to deploy across all M strikes (typically
                        ``hedge_premium_target(...)`` from the S1 Task-A
                        offset rule).
        forward:        BTC forward price at hedge open.
        svi_params:     deterministic SVI surface (S3 fallback canonical).
        util:           pool utilization at open (drives spread term-2).
        n_strikes:      M (≥1). M=1 collapses to the S1 single-strike hedge.
        moneyness_lo/hi: ladder range. Defaults = PLP p0.1 / p1 loss-onset.

    Returns:
        DnLadder. Sum of premium_per_strike ≈ sleeve_budget (modulo
        spread per-strike differences across the range).
    """
    if sleeve_budget <= 0:
        raise ValueError(f"sleeve_budget must be > 0, got {sleeve_budget}")
    if n_strikes < 1:
        raise ValueError(f"n_strikes must be >= 1, got {n_strikes}")
    if not (0.0 < moneyness_lo <= moneyness_hi < 1.0):
        raise ValueError(
            f"need 0 < moneyness_lo ({moneyness_lo}) <= moneyness_hi "
            f"({moneyness_hi}) < 1 for OTM-DN ladder"
        )
    if forward <= 0:
        raise ValueError(f"forward must be > 0, got {forward}")

    # Uniform in log-moneyness — principled match to a multiplicative
    # interpretation of the PLP loss band.
    if n_strikes == 1:
        log_k = np.array([0.5 * (np.log(moneyness_lo) + np.log(moneyness_hi))])
    else:
        log_k = np.linspace(
            np.log(moneyness_lo), np.log(moneyness_hi), n_strikes
        )
    moneyness = np.exp(log_k)
    strikes = forward * moneyness

    dn_mid = np.asarray(dn_price(strikes, forward, svi_params), dtype=float)
    spread_kw = {}
    if base_spread is not None:
        spread_kw["base_spread"] = base_spread
    if min_spread is not None:
        spread_kw["min_spread"] = min_spread
    if util_mult is not None:
        spread_kw["util_mult"] = util_mult
    ask = np.asarray(ask_price(dn_mid, util, **spread_kw), dtype=float)
    # Defensive: ask must be > 0 by spread clip; guard anyway.
    if np.any(ask <= 0):
        raise ValueError(f"computed asks must be > 0, got {ask}")

    # Uniform sleeve allocation across the ladder.
    budget_per_strike = sleeve_budget / float(n_strikes)
    notional_per_strike = budget_per_strike / ask
    premium_per_strike = notional_per_strike * ask  # = budget_per_strike each

    return DnLadder(
        strikes=strikes,
        moneyness=moneyness,
        notional_per_strike=notional_per_strike,
        dn_mid_per_strike=dn_mid,
        ask_per_strike=ask,
        premium_per_strike=premium_per_strike,
        total_premium_paid=float(np.sum(premium_per_strike)),
    )
