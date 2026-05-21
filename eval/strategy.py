"""Three strategies for the S1 coarse f-sweep.

Per CLAUDE.md §6 S1 task and spec §5: compare on the SAME price path while
varying Strata's LP-share fraction ``f`` (controlled outside this module by
the strata_capital vs other_lp_initial ratio). The f-Curve at the
Conservative trader-flow anchor is the hero artifact (CLAUDE.md §7).

Strategies:
    RAW_PLP            — 100% PLP supply, 0% hedge. Unprotected baseline;
                          shows the "is PLP safe?" question vividly.
    FIXED_HEDGE_NAIVE  — 85% PLP + 15% hedge, 0 reserve. At envelope cap on
                          hedge but no liquidity buffer. Strawman: "more
                          hedge is always better" hypothesis.
    STRATA             — 80% PLP + 15% hedge + 5% reserve. The chosen
                          design — interior allocation under the §2A
                          envelope, with the 5% reserve serving as the
                          R3 liquidity escape-hatch pillar.

In S1 the hedge is single-strike at fixed moneyness for both
FIXED_HEDGE_NAIVE and STRATA — the dynamic-SVI hedge ratio (§3 b/ρ-driven
allocation within the 15% sleeve) is S4+ work. So at S1 these two
strategies differ ONLY in reserve presence; that's expected and
diagnostically useful (separates "hedge size effect" from "reserve effect").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Tuple

import numpy as np

from engine.svi_det import SVIParams
from eval.account import (
    AccountConfig,
    initialize,
    open_hedge,
    settle_path,
    step_path,
)
from model.trader_flow import TraderFlowAnchor


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy = a (sleeve_plp, sleeve_hedge, sleeve_reserve, has_hedge) tuple."""

    name: str
    sleeve_plp: float
    sleeve_hedge: float
    sleeve_reserve: float
    has_hedge: bool


RAW_PLP: Final[StrategyConfig] = StrategyConfig(
    name="raw_plp",
    sleeve_plp=1.0,
    sleeve_hedge=0.0,
    sleeve_reserve=0.0,
    has_hedge=False,
)

FIXED_HEDGE_NAIVE: Final[StrategyConfig] = StrategyConfig(
    name="fixed_hedge_naive",
    sleeve_plp=0.85,
    sleeve_hedge=0.15,
    sleeve_reserve=0.0,
    has_hedge=True,
)

STRATA: Final[StrategyConfig] = StrategyConfig(
    name="strata",
    sleeve_plp=0.80,
    sleeve_hedge=0.15,
    sleeve_reserve=0.05,
    has_hedge=True,
)

ALL_STRATEGIES: Final[Tuple[StrategyConfig, ...]] = (
    RAW_PLP,
    FIXED_HEDGE_NAIVE,
    STRATA,
)


def run_strategy(
    strategy: StrategyConfig,
    strata_capital: float,
    other_lp_initial: float,
    price_path: np.ndarray,
    realized_vols: np.ndarray,
    sigma_long_run: float,
    log_returns: np.ndarray,
    svi_params: SVIParams,
    anchor: TraderFlowAnchor,
    hedge_moneyness: float = 0.85,
) -> dict:
    """Run one strategy on one price path; return Strata P&L + diagnostics.

    Cycle skeleton (S1 single-cycle simplification):
        1. ``initialize`` with the strategy's sleeve config.
        2. If ``has_hedge``: ``open_hedge`` at ``price_path[0]`` (forward).
        3. For each step t ∈ [0, T): ``step_path`` (trader flow + LP flight).
        4. ``settle_path`` at ``price_path[-1]``.

    Args:
        strategy:           One of ``RAW_PLP``, ``FIXED_HEDGE_NAIVE``, ``STRATA``.
        strata_capital:     Strata's deposit, USD.
        other_lp_initial:   Other-LP capital initial, USD. (strata_capital /
                            (strata_capital + other_lp_initial)) = initial f.
        price_path:         Length T+1 BTC prices. ``[0]`` = open forward,
                            ``[-1]`` = settlement price.
        realized_vols:      Length T rolling realized vol per step.
        sigma_long_run:     Long-run vol mean (for Knob-2 normalization).
        log_returns:        Length T per-step log returns (feeds informed bias).
        svi_params:         Deterministic SVI (S1; S3 will make path-dependent).
        anchor:             Trader-flow knob calibration band.
        hedge_moneyness:    OTM-DN strike-as-fraction-of-forward.

    Returns:
        ``settle_path`` output dict, plus ``"strategy_name"`` key.
    """
    if len(price_path) != len(log_returns) + 1:
        raise ValueError(
            f"price_path len ({len(price_path)}) must equal log_returns len "
            f"({len(log_returns)}) + 1"
        )
    if len(realized_vols) != len(log_returns):
        raise ValueError(
            f"realized_vols len ({len(realized_vols)}) must equal log_returns "
            f"len ({len(log_returns)})"
        )

    cfg = AccountConfig(
        strata_capital=strata_capital,
        other_lp_initial=other_lp_initial,
        sleeve_plp=strategy.sleeve_plp,
        sleeve_hedge=strategy.sleeve_hedge,
        sleeve_reserve=strategy.sleeve_reserve,
        hedge_moneyness=hedge_moneyness,
    )
    state = initialize(cfg)

    forward_open = float(price_path[0])
    if strategy.has_hedge:
        open_hedge(state, forward=forward_open, svi_params=svi_params)

    for t in range(len(log_returns)):
        step_path(
            state,
            forward=float(price_path[t]),
            realized_vol=float(realized_vols[t]),
            sigma_long_run=sigma_long_run,
            log_return_recent=float(log_returns[t]),
            svi_params=svi_params,
            anchor=anchor,
        )

    out = settle_path(
        state,
        settle_price=float(price_path[-1]),
        anchor=anchor,
    )
    out["strategy_name"] = strategy.name
    return out
