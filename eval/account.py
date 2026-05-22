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

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from engine.spread import spread_per_contract
from engine.svi_det import SVIParams, dn_price
from model.hedge import HedgePosition, size_hedge
from model.plp import PLPVault
from model.trader_flow import (
    TraderFlowAnchor,
    lp_flight_step,
    trader_flow_step,
    u_strike_local,
)


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
        for name, val in (
            ("sleeve_plp", self.sleeve_plp),
            ("sleeve_hedge", self.sleeve_hedge),
            ("sleeve_reserve", self.sleeve_reserve),
        ):
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        sleeve_sum = self.sleeve_plp + self.sleeve_hedge + self.sleeve_reserve
        if not 0.999 <= sleeve_sum <= 1.001:
            raise ValueError(
                f"sleeves must sum to 1.0 (deploy all capital), got "
                f"{sleeve_sum:.4f}"
            )
        # The CLAUDE.md §2A 85/15/5 envelope is a STRATA POLICY guideline, not
        # a hard system constraint — baseline strategies (raw_plp = 100% PLP,
        # fixed_hedge = etc.) deliberately violate it to be compared against
        # Strata's chosen allocation. Envelope caps documented via the
        # ENVELOPE_*_MAX constants above for reference; enforcement happens
        # at the strategy layer (eval/strategy.py), not here.
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
    # Path-aware non-hedge mint history for honest settlement: each entry is
    # (mint_spot, dn_notional, up_notional). At settle, a DN minted at
    # mint_spot is ITM iff settle < mint_spot; an UP iff settle > mint_spot.
    # Without this, clearing MTM at settle erased the pool's payout liability
    # → PLP became a money-printer (no downside). See settle_path.
    mint_history: List[Tuple[float, float, float]] = field(default_factory=list)

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
    dn_notional = float(flow["dn_notional"])
    up_notional = float(flow["up_notional"])
    total_notional = dn_notional + up_notional

    if total_notional > 0.0:
        # Pool receives the FULL ask (mid + spread) per contract — NOT just the
        # spread. The fair-value (mid) portion is a real liability the pool must
        # honor at settlement; recording only the spread + clearing MTM later
        # was the money-printer bug. NAV change at mint = total_notional × spread.
        sp = float(spread_per_contract(representative_p, util_pre))
        ask = representative_p + sp
        state.plp.receive_premium(total_notional * ask)
        state.plp.update_mtm(
            state.plp.total_mtm + total_notional * representative_p
        )
        state.plp.update_max_payout(
            state.plp.total_max_payout + total_notional
        )
        # Record for path-aware settlement (resolved in settle_path): DN/UP
        # notional minted at this step's spot (= forward).
        state.mint_history.append((float(forward), dn_notional, up_notional))

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


def settle_path(
    state: AccountState,
    settle_price: float,
    anchor: TraderFlowAnchor,
) -> dict:
    """Settle one cycle: resolve hedge strike + compute Strata net P&L.

    Per ``specs/trader_flow_spec.md §1.2`` strike-local formulation, Strata's
    total P&L decomposes equivalently in two ways (algebraically identical):

        (A) Combined-strike form:
            total = (N_U + N_O) · (u(k) − f) · (𝟙_crash − ask)

        (B) Share-price form (used here):
            total = strata_shares · (share_price_settle − 1.0)   ← LP leg
                  + N_U · (𝟙_crash − ask_per_contract)            ← direct hedge

    Form (B) is preferred because:
      * Robust under LP-flight: ``strata_shares`` is constant (Strata never
        withdraws mid-cycle in S1), even though ``f`` drifts as L_other shrinks.
      * Needs only ONE settlement event (the hedge strike clear), not a per-
        strike book.

    Effects on pool:
        * Pool pays ``N_total · 𝟙_crash`` at hedge strike, where
          ``N_total = N_U / u(k)`` is the implied total DN open interest
          (Strata's hedge + other DN buyers at the same strike).
        * Pool's ``total_mtm`` and ``total_max_payout`` clear to 0 at settle
          (all positions resolved).

    S1 simplification (documented): non-hedge positions assumed UP/DN
    approximately paired by Knob-3 mechanic ⇒ net pool cashflow from non-hedge
    settlement ≈ 0. Accumulated spread retained in balance. Full per-strike
    StrikeMatrix lands at S4.

    Returns:
        Diagnostic dict with crash flag, strike-local quantities, and the
        full Strata P&L breakdown.
    """
    # Path-aware non-hedge settlement (the money-printer fix): resolve every
    # trader-flow position recorded in mint_history. A DN minted at mint_spot
    # is ITM iff settle < mint_spot; an UP iff settle > mint_spot. Pool pays
    # $1 per ITM contract. This is what makes raw PLP genuinely risky — in a
    # crash, all DN minted at higher spots become ITM and the pool bleeds.
    non_hedge_payout = 0.0
    for mint_spot, dn_notional, up_notional in state.mint_history:
        if settle_price < mint_spot:
            non_hedge_payout += dn_notional
        elif settle_price > mint_spot:
            non_hedge_payout += up_notional
    if non_hedge_payout > 0.0:
        state.plp.pay_settlement(non_hedge_payout)

    # raw_plp baseline strategy has no hedge — settle LP-leg only.
    if state.hedge is None:
        state.plp.update_mtm(0.0)
        state.plp.update_max_payout(0.0)
        initial_share_price = 1.0
        share_price_settle = (
            state.plp.share_price
            if state.strata_shares > 0
            else initial_share_price
        )
        strata_pnl_lp_leg = state.strata_shares * (
            share_price_settle - initial_share_price
        )
        strata_terminal = (
            state.reserve + state.strata_shares * share_price_settle
        )
        return {
            "crash": False,
            "settle_price": float(settle_price),
            "strike": None,
            "f_settle": float(state.f),
            "u_k": None,
            "n_total_at_strike": 0.0,
            "pool_hedge_payout": 0.0,
            "nav_settle": float(state.plp.nav),
            "share_price_settle": float(share_price_settle),
            "strata_pnl_lp_leg": float(strata_pnl_lp_leg),
            "strata_pnl_direct_hedge": 0.0,
            "strata_pnl_total": float(strata_pnl_lp_leg),
            "strata_terminal": float(strata_terminal),
        }

    h = state.hedge
    f_settle = state.f
    u_k = float(u_strike_local(anchor.kappa_strike, f_settle))
    crash = 1.0 if settle_price < h.strike else 0.0

    # Pool's hedge-strike payout clears (N_U + N_O) at this strike on crash.
    if u_k > 0.0:
        n_total = h.notional / u_k
    else:
        n_total = 0.0
    pool_hedge_payout = n_total * crash
    state.plp.pay_settlement(pool_hedge_payout)

    # Clear all open MTM and max_payout (all positions resolved at end-of-cycle).
    state.plp.update_mtm(0.0)
    state.plp.update_max_payout(0.0)

    # Strata P&L via share-price form (B).
    initial_share_price = 1.0  # NAV-proportional, 1:1 at bootstrap by construction.
    share_price_settle = (
        state.plp.share_price if state.strata_shares > 0 else initial_share_price
    )
    strata_pnl_lp_leg = state.strata_shares * (
        share_price_settle - initial_share_price
    )
    strata_pnl_direct_hedge = h.notional * (crash - h.ask_per_contract)
    strata_pnl_total = strata_pnl_lp_leg + strata_pnl_direct_hedge

    # Terminal wealth (sanity diagnostic): reserve + LP shares + hedge payoff.
    # hedge_sleeve is 0 here (fully spent at open into pool.balance).
    strata_terminal = (
        state.reserve
        + state.strata_shares * share_price_settle
        + h.notional * crash
    )

    return {
        "crash": bool(crash),
        "settle_price": float(settle_price),
        "strike": float(h.strike),
        "f_settle": float(f_settle),
        "u_k": float(u_k),
        "n_total_at_strike": float(n_total),
        "pool_hedge_payout": float(pool_hedge_payout),
        "nav_settle": float(state.plp.nav),
        "share_price_settle": float(share_price_settle),
        "strata_pnl_lp_leg": float(strata_pnl_lp_leg),
        "strata_pnl_direct_hedge": float(strata_pnl_direct_hedge),
        "strata_pnl_total": float(strata_pnl_total),
        "strata_terminal": float(strata_terminal),
    }
