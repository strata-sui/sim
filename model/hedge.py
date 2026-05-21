"""Single-strike OTM-DN hedge — S1 thin-slice form.

S1 simplification: one DN-binary at one OTM strike per path. S4 will
upgrade to the SVI-shaped DN ladder matched to vault ``max_payout``
prefix (CLAUDE.md §5).

Mechanics (CLAUDE.md §4):
  * Trader (here, Strata as manager) mints DN binary by paying
    ``ask = mid + spread`` per contract to the pool.
  * Binary pays $1 per contract on settlement if ``settle_price < strike``,
    else $0.
  * `redeem` on settle BYPASSES the PLP withdraw limiter — the
    "liquidity escape hatch" co-equal pitch pillar (CLAUDE.md §2A R3).

Sizing convention for S1 (single strike, single hedge):
    strike = forward * moneyness         (e.g. moneyness=0.85 → 15% OTM)
    dn_mid = DN(strike, forward, SVI)
    ask    = mid + spread(dn_mid, util)
    N      = sleeve_budget / ask         (full sleeve deployed)
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.spread import ask_price
from engine.svi_det import SVIParams, dn_price


@dataclass(frozen=True)
class HedgePosition:
    """Frozen-after-open snapshot of a single DN-binary hedge.

    Attrs:
        strike:        OTM strike chosen (USD).
        notional:      N contracts (1 contract = $1 payoff if ITM).
        premium_paid:  total USD paid to pool at open
                       (= notional * ask_per_contract).
        ask_per_contract: the per-contract ask used at open (mid + spread).
        dn_mid:        the fair DN mid at open (for diagnostics / mark).
    """

    strike: float
    notional: float
    premium_paid: float
    ask_per_contract: float
    dn_mid: float

    def is_in_the_money(self, settle_price: float) -> bool:
        """DN binary is ITM iff settlement < strike."""
        return settle_price < self.strike

    def payoff(self, settle_price: float) -> float:
        """Crash payoff in USD: notional if ITM, else 0.

        Bypasses PLP withdraw limiter per CLAUDE.md §4 — the manager
        receives this directly via `redeem`, not via the LP withdraw path.
        """
        return self.notional if self.is_in_the_money(settle_price) else 0.0

    def net_pnl(self, settle_price: float) -> float:
        """Net hedge PnL = payoff - premium_paid.

        Strata's hedge-leg PnL only. The full self-reference accounting
        (LP-leg dilution of premium + payoff) lives in strategy.py where
        the f-dilution + u(k)-vs-f formula from spec §1.2 applies.
        """
        return self.payoff(settle_price) - self.premium_paid


def size_hedge(
    sleeve_budget: float,
    forward: float,
    moneyness: float,
    util: float,
    svi_params: SVIParams,
    base_spread: float | None = None,
    min_spread: float | None = None,
    util_mult: float | None = None,
) -> HedgePosition:
    """Open one single-strike DN hedge that fully deploys ``sleeve_budget``.

    Args:
        sleeve_budget:  USD allocated to the hedge sleeve this roll.
        forward:        current forward price (BTC USD).
        moneyness:      strike-as-fraction-of-forward, ``∈ (0, 1)`` for OTM-DN
                        (e.g. 0.85 = strike 15% below forward).
        util:           pool utilization at open (drives spread term 2).
        svi_params:     current SVI surface for DN pricing.
        base_spread, min_spread, util_mult: optional spread-formula overrides.

    Returns:
        Frozen ``HedgePosition``. ``premium_paid == notional × ask_per_contract``
        by construction.

    Raises:
        ValueError: invalid inputs.
    """
    if sleeve_budget <= 0:
        raise ValueError(f"sleeve_budget must be > 0, got {sleeve_budget}")
    if not 0.0 < moneyness < 1.0:
        raise ValueError(
            f"moneyness must be ∈ (0, 1) for OTM-DN, got {moneyness}"
        )
    if forward <= 0:
        raise ValueError(f"forward must be > 0, got {forward}")

    strike = forward * moneyness
    dn_mid = float(dn_price(strike, forward, svi_params))

    spread_kwargs = {}
    if base_spread is not None:
        spread_kwargs["base_spread"] = base_spread
    if min_spread is not None:
        spread_kwargs["min_spread"] = min_spread
    if util_mult is not None:
        spread_kwargs["util_mult"] = util_mult

    ask = float(ask_price(dn_mid, util, **spread_kwargs))
    if ask <= 0:
        # Defensive — clipping floor in spread module enforces ask >= 0.01.
        raise ValueError(f"computed ask is non-positive ({ask})")

    notional = sleeve_budget / ask
    premium_paid = notional * ask

    return HedgePosition(
        strike=strike,
        notional=notional,
        premium_paid=premium_paid,
        ask_per_contract=ask,
        dn_mid=dn_mid,
    )
