"""3-account integration: PLP NAV + DN-ladder hedge + f-dilution.

Per CLAUDE.md §3 + ``specs/trader_flow_spec.md §1.2`` strike-local formulation.

The three accounts (CLAUDE.md §5 mandatory 3-account model):
    1. PLP NAV         — Strata as LP, shares NAV-proportional via ``PLPVault``.
    2. DN-ladder hedge — S1: single OTM-DN binary via ``HedgePosition``.
    3. f-dilution +
       limiter liquidity — Strata's net P&L weights both legs by the
                            strike-local share ``u(k)`` and the dynamic LP
                            fraction ``f`` (post LP-flight).

S1 simplification: this file establishes the **state container** and the
**bootstrap** only. Per-step evolution (``step_path``) and settlement
(``settle_path``) ship in follow-up atomic commits — each ≤ ~150 LoC, each
with smoke-test imports.

Net Strata hedge P&L on settlement at hedge strike k (derived in
``specs/trader_flow_spec.md §1.2``):

    net_hedge = N_U · (1 − f / u(k)) · (𝟙_crash − ask_per_contract)

Boundary checks (verify by inspection later in ``settle_path``):
    u(k) = 1  → net = N_U · (1 − f) · (…)   (CLAUDE.md §3 special case)
    u(k) = f  → net = N_U · 0 · (…) = 0     (pure wash; thesis-fatal regime)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.spread import spread_per_contract
from engine.svi_det import SVIParams, dn_price
from model.hedge import HedgePosition, size_hedge
from model.plp import PLPVault
from model.trader_flow import TraderFlowAnchor, lp_flight_step, trader_flow_step


# CLAUDE.md §2A: "85/15/5 is the ENVELOPE (max PLP / max hedge / reserve)" —
# these are MAX CAPS, not required actuals. Actual sleeve allocations must sum
# to 1.0 (deploy all of strata_capital) AND each fit under its cap. Defaults
# below pick a valid interior point: 80% PLP (under 85% cap), 15% hedge (at
# cap), 5% reserve (at cap). 80+15+5 = 1.00.
ENVELOPE_PLP_MAX = 0.85
ENVELOPE_HEDGE_MAX = 0.15
ENVELOPE_RESERVE_MAX = 0.05

DEFAULT_SLEEVE_PLP = 0.80
DEFAULT_SLEEVE_HEDGE = 0.15
DEFAULT_SLEEVE_RESERVE = 0.05

# OTM-DN hedge strike-as-fraction-of-forward (S1 default; S4 promotes to ladder).
DEFAULT_HEDGE_MONEYNESS = 0.85


@dataclass(frozen=True)
class AccountConfig:
    """Strata vault + market-structure config for one Monte-Carlo path.

    Attrs:
        strata_capital:    Strata's total capital deposit, USD.
        other_lp_initial:  Initial other-LP capital in the pool, USD.
                           Sets the starting ``f = strata_supply / (strata_supply
                           + other_lp_initial)``.
        sleeve_plp:        Fraction of ``strata_capital`` deployed to PLP supply.
        sleeve_hedge:      Fraction allocated to the hedge sleeve.
        sleeve_reserve:    Fraction held idle (drawdown buffer / roll friction).
        hedge_moneyness:   OTM-DN strike as fraction of forward, ∈ (0, 1).
    """

    strata_capital: float
    other_lp_initial: float
    sleeve_plp: float = DEFAULT_SLEEVE_PLP
    sleeve_hedge: float = DEFAULT_SLEEVE_HEDGE
    sleeve_reserve: float = DEFAULT_SLEEVE_RESERVE
    hedge_moneyness: float = DEFAULT_HEDGE_MONEYNESS

    def __post_init__(self) -> None:
        if self.strata_capital <= 0:
            raise ValueError(
                f"strata_capital must be > 0, got {self.strata_capital}"
            )
        if self.other_lp_initial < 0:
            raise ValueError(
                f"other_lp_initial must be >= 0, got {self.other_lp_initial}"
            )
        sleeve_sum = self.sleeve_plp + self.sleeve_hedge + self.sleeve_reserve
        if not 0.999 <= sleeve_sum <= 1.001:
            raise ValueError(
                f"sleeves must sum to 1.0 (deploy all capital), got "
                f"{sleeve_sum:.4f}"
            )
        if self.sleeve_plp > ENVELOPE_PLP_MAX + 1e-9:
            raise ValueError(
                f"sleeve_plp={self.sleeve_plp} exceeds envelope cap "
                f"{ENVELOPE_PLP_MAX}"
            )
        if self.sleeve_hedge > ENVELOPE_HEDGE_MAX + 1e-9:
            raise ValueError(
                f"sleeve_hedge={self.sleeve_hedge} exceeds envelope cap "
                f"{ENVELOPE_HEDGE_MAX}"
            )
        if self.sleeve_reserve > ENVELOPE_RESERVE_MAX + 1e-9:
            raise ValueError(
                f"sleeve_reserve={self.sleeve_reserve} exceeds envelope cap "
                f"{ENVELOPE_RESERVE_MAX}"
            )
        if not 0.0 < self.hedge_moneyness < 1.0:
            raise ValueError(
                f"hedge_moneyness must be ∈ (0, 1) for OTM-DN, got "
                f"{self.hedge_moneyness}"
            )


@dataclass
class AccountState:
    """Mutable 3-account state for one path.

    The dynamic LP fraction ``f`` lives here (computed live each step from
    ``strata_supply`` vs ``l_other``). The strike-local share ``u(k)`` is
    computed in ``settle_path`` from ``anchor.kappa_strike`` + ``f`` via
    ``trader_flow.u_strike_local``.
    """

    config: AccountConfig
    plp: PLPVault
    l_other: float                       # Other-LP capital in USD (V2 dynamic).
    strata_shares: float                 # PLP shares minted to Strata.
    strata_supply: float                 # USD Strata supplied to PLP.
    hedge_sleeve: float                  # USD allocated to hedge (pre-mint).
    reserve: float                       # USD held idle.
    hedge: Optional[HedgePosition] = None
    nav_high_watermark: float = 0.0      # For drawdown calc (drives LP-flight).

    @property
    def f(self) -> float:
        """Aggregate LP-share fraction: Strata supply / (Strata supply + L_other)."""
        total_lp = self.strata_supply + self.l_other
        if total_lp <= 0.0:
            return 0.0
        return self.strata_supply / total_lp

    @property
    def drawdown_pct(self) -> float:
        """PLP-NAV peak-to-trough drawdown, ∈ [0, 1]."""
        if self.nav_high_watermark <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.plp.nav / self.nav_high_watermark)


def initialize(config: AccountConfig) -> AccountState:
    """Open all three accounts.

    Order matters: other-LP supplies FIRST at NAV=1.0 (gets 1:1 shares),
    THEN Strata supplies at the same NAV=1.0 (also 1:1 in the empty-vault
    bootstrap). This sets ``f0 = strata_supply / (strata_supply + l_other)``
    cleanly at the configured ratio.

    Returns:
        Fully-initialized ``AccountState`` ready for ``step_path``.
    """
    plp = PLPVault()
    if config.other_lp_initial > 0:
        plp.supply(config.other_lp_initial)
    strata_supply = config.strata_capital * config.sleeve_plp
    strata_shares = plp.supply(strata_supply) if strata_supply > 0 else 0.0
    hedge_sleeve = config.strata_capital * config.sleeve_hedge
    reserve = config.strata_capital * config.sleeve_reserve
    return AccountState(
        config=config,
        plp=plp,
        l_other=config.other_lp_initial,
        strata_shares=strata_shares,
        strata_supply=strata_supply,
        hedge_sleeve=hedge_sleeve,
        reserve=reserve,
        hedge=None,
        nav_high_watermark=plp.nav,
    )


def open_hedge(
    state: AccountState,
    forward: float,
    svi_params: SVIParams,
) -> HedgePosition:
    """Mint the single-strike OTM-DN hedge using the full hedge sleeve.

    Effects (CLAUDE.md §4 verified mechanics):
        * Strata-as-trader pays ``premium_paid`` out of its hedge sleeve.
        * Pool ``balance``           += ``premium_paid``
        * Pool ``total_mtm``         += ``notional × dn_mid`` (fair-value liability)
        * Pool ``total_max_payout``  += ``notional``           (worst-case liability)
        * ``state.hedge_sleeve``     := 0  (fully deployed)
        * ``state.hedge``            := new ``HedgePosition``

    The premium flows into the pool that Strata is f-share of — this is the
    self-referential leg. Settlement applies the strike-local ``(u(k) − f)``
    formula in ``settle_path`` (separate atomic commit) to recover Strata's
    actual net hedge P&L.

    Raises:
        RuntimeError: if a hedge is already open (must be settled first).
        ValueError:   if ``hedge_sleeve`` is depleted.
    """
    if state.hedge is not None:
        raise RuntimeError(
            "hedge already open; settle/redeem before re-opening"
        )
    if state.hedge_sleeve <= 0:
        raise ValueError(
            f"hedge_sleeve depleted ({state.hedge_sleeve}); cannot open"
        )

    util = (
        state.plp.total_mtm / state.plp.balance
        if state.plp.balance > 0
        else 0.0
    )

    hedge = size_hedge(
        sleeve_budget=state.hedge_sleeve,
        forward=forward,
        moneyness=state.config.hedge_moneyness,
        util=util,
        svi_params=svi_params,
    )

    state.hedge = hedge
    state.hedge_sleeve = 0.0

    state.plp.receive_premium(hedge.premium_paid)
    state.plp.update_mtm(
        state.plp.total_mtm + hedge.notional * hedge.dn_mid
    )
    state.plp.update_max_payout(
        state.plp.total_max_payout + hedge.notional
    )

    # NAV high-water-mark snaps to post-open NAV (drawdown clock starts here).
    state.nav_high_watermark = max(state.nav_high_watermark, state.plp.nav)
    return hedge


# Per-step S1 simplification: model arriving trader notional as if it lands at
# a single representative fair price (near-ATM, p≈0.5). The proper per-strike
# StrikeMatrix lands at S4 (full DN ladder). For Gate-A f-sweep this proxy is
# sufficient — only the magnitudes of premium accrual and max_payout matter,
# not exact strike distribution.
REPRESENTATIVE_PRICE_S1 = 0.5


def step_path(
    state: AccountState,
    forward: float,
    realized_vol: float,
    sigma_long_run: float,
    log_return_recent: float,
    svi_params: SVIParams,
    anchor: TraderFlowAnchor,
    representative_p: float = REPRESENTATIVE_PRICE_S1,
) -> dict:
    """One step of joint evolution: trader flow → pool state → LP flight.

    Per CLAUDE.md §4 verified mechanics, this step does NOT settle any
    positions (settlement is end-of-cycle in ``settle_path``). Effects:

        1. ``trader_flow_step`` → arriving DN+UP notional (knobs 1–3).
        2. Pool premium accrual: ``balance += total_notional × spread(p_rep, util)``
        3. Pool liability marks:
           ``total_mtm        += total_notional × p_rep``      (fair value)
           ``total_max_payout += total_notional``              (worst case = $1/contract)
        4. NAV high-water-mark snapped to current NAV.
        5. LP flight (``lp_flight_step``, knobs 5–6): ``l_other`` decreases
           under drawdown subject to ``theta_limiter`` cap. This is the f→1
           drift mechanism that makes the self-reference flaw bite mid-event.

    The hedge is marked informationally (``hedge_mark`` in return dict) but
    its P&L is NOT realized until ``settle_path``.

    Returns:
        dict of step diagnostics (``f``, ``util``, ``drawdown``,
        ``hedge_mark``, ``nav``, ``balance``, ``l_other``, ``total_notional``).
    """
    util_pre = (
        state.plp.total_mtm / state.plp.balance
        if state.plp.balance > 0
        else 0.0
    )

    flow = trader_flow_step(
        pool_balance=state.plp.balance,
        realized_vol=realized_vol,
        sigma_long_run=sigma_long_run,
        log_return_recent=log_return_recent,
        anchor=anchor,
    )
    total_notional = float(flow["total_notional"])

    if total_notional > 0.0:
        sp = float(spread_per_contract(representative_p, util_pre))
        state.plp.receive_premium(total_notional * sp)
        state.plp.update_mtm(
            state.plp.total_mtm + total_notional * representative_p
        )
        state.plp.update_max_payout(
            state.plp.total_max_payout + total_notional
        )

    hedge_mark = 0.0
    if state.hedge is not None:
        hedge_mark = float(
            dn_price(state.hedge.strike, forward, svi_params)
        ) * state.hedge.notional

    state.nav_high_watermark = max(state.nav_high_watermark, state.plp.nav)

    dd = state.drawdown_pct
    state.l_other = float(lp_flight_step(state.l_other, dd, anchor))

    return {
        "f": state.f,
        "util": util_pre,
        "drawdown": dd,
        "hedge_mark": hedge_mark,
        "nav": state.plp.nav,
        "balance": state.plp.balance,
        "l_other": state.l_other,
        "total_notional": total_notional,
    }
