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

from engine.spread import ask_price
from engine.svi_det import SVIParams, dn_price
from eval.account import (
    AccountConfig,
    hedge_premium_target,
    initialize,
    open_hedge,
    settle_path,
    step_path,
)
from model.dn_ladder import (
    DEFAULT_LADDER_SIZE,
    DEFAULT_MONEYNESS_HI,
    DEFAULT_MONEYNESS_LO,
    size_ladder,
)
from model.token_bucket import TokenBucket
from model.trader_flow import (
    TraderFlowAnchor,
    sample_kappa_ladder,
    u_strike_local,
)


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
    hedge_moneyness: float = 0.98,  # PLP loss-onset, not Sortino-tuned (see account.py)
    svi_params_path: "list[SVIParams] | None" = None,  # S3.7: per-step SVI
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

    # S3.7: if a per-step SVI path is supplied, use that; else fall back to
    # the single constant svi_params (S1/S2 behaviour, backward-compatible).
    def _svi_at(t: int) -> SVIParams:
        return svi_params_path[t] if svi_params_path is not None else svi_params

    forward_open = float(price_path[0])
    if strategy.has_hedge:
        open_hedge(
            state, forward=forward_open, svi_params=_svi_at(0), anchor=anchor
        )

    for t in range(len(log_returns)):
        # Knob-3 informed bias must react to the PAST (already-realized) move,
        # NOT the future. price_path[t+1] = price_path[t]·exp(log_returns[t]),
        # so log_returns[t] is the t→t+1 move — using it here would hand the
        # informed trader perfect foresight of the next bar (catastrophic
        # look-ahead: drove raw_plp prob_loss to 0.91 and an impossible
        # negative benign carry, contradicting the verified §4 short-vol-with-
        # spread-carry profile). The move that JUST brought price to
        # price_path[t] is log_returns[t-1]; at t=0 there is no prior move so
        # the signal is neutral (balanced flow).
        recent = float(log_returns[t - 1]) if t > 0 else 0.0
        step_path(
            state,
            forward=float(price_path[t]),
            realized_vol=float(realized_vols[t]),
            sigma_long_run=sigma_long_run,
            log_return_recent=recent,
            svi_params=_svi_at(t),
            anchor=anchor,
        )

    out = settle_path(
        state,
        settle_price=float(price_path[-1]),
        anchor=anchor,
    )
    out["strategy_name"] = strategy.name
    return out


# ---- S4.3 — multi-cycle dynamics (state rolls across cycles) ------------


def run_strategy_multi_cycle(
    strategy: StrategyConfig,
    strata_capital: float,
    other_lp_initial: float,
    price_path: np.ndarray,
    realized_vols: np.ndarray,
    sigma_long_run: float,
    log_returns: np.ndarray,
    svi_params: SVIParams,
    anchor: TraderFlowAnchor,
    n_cycles: int = 10,
    path_steps: int = 4,
    hedge_moneyness: float = 0.98,
    svi_params_path: "list[SVIParams] | None" = None,
) -> dict:
    """Run N consecutive cycles with state continuation between them.

    Each MC path is sliced into n_cycles × path_steps bars; per cycle:
      1. Reset PER-CYCLE ephemera (hedge=None, mint_history=[]).
      2. Open hedge (if has_hedge) at the cycle's opening forward.
      3. Step path_steps times.
      4. Settle the cycle (PLP MTM and max_payout clear).
      5. Carry PERSISTENT state forward to the next cycle (PLP balance,
         strata_shares, l_other, nav high-water-mark, hedge_sleeve
         drawn down by the cycle's premium).

    Cross-cycle Strata P&L accounting (the subtle bit):
      * LP-leg: ``shares × (final_share_price − 1.0)`` computed ONCE at
        the end. Per-cycle settle_path's strata_pnl_lp_leg uses an
        implicit initial sp=1.0 that becomes stale after cycle-1, so the
        wrapper IGNORES per-cycle lp-leg pnl and computes it once on the
        terminal state.
      * Hedge-leg: ``Σ_cycles strata_pnl_direct_hedge``. Each cycle's
        hedge is opened/settled fresh, so per-cycle direct-hedge sums.

    Args:
        n_cycles:   number of cycles per path (default 10 ≈ 10 hours on 4-step
                    sub-hour cycles).
        path_steps: bars per cycle. price_path/log_returns/realized_vols are
                    sliced n_cycles × path_steps + 1, n_cycles × path_steps,
                    n_cycles × path_steps respectively.

    Returns:
        dict with strata_pnl_total + lp_leg_pnl + hedge_leg_pnl + cycles
        (per-cycle diagnostics) + n_cycles + strategy_name.

    Raises:
        ValueError on length mismatches.
    """
    expected_len = n_cycles * path_steps
    if len(log_returns) != expected_len:
        raise ValueError(
            f"log_returns len ({len(log_returns)}) != "
            f"n_cycles*path_steps ({expected_len})"
        )
    if len(price_path) != expected_len + 1:
        raise ValueError(
            f"price_path len ({len(price_path)}) != "
            f"expected ({expected_len + 1})"
        )
    if len(realized_vols) != expected_len:
        raise ValueError(
            f"realized_vols len ({len(realized_vols)}) != "
            f"expected ({expected_len})"
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

    def _svi_at(t_global: int) -> SVIParams:
        return svi_params_path[t_global] if svi_params_path is not None else svi_params

    initial_share_price = 1.0
    cycle_diagnostics: list[dict] = []
    hedge_pnl_total = 0.0

    for cycle in range(n_cycles):
        c_lo = cycle * path_steps
        c_hi = c_lo + path_steps
        cycle_prices = price_path[c_lo : c_hi + 1]
        cycle_log_rets = log_returns[c_lo : c_hi]
        cycle_rv = realized_vols[c_lo : c_hi]

        # Reset per-cycle ephemera ONLY. PLP balance, shares, l_other
        # persist (state continuation).
        state.hedge = None
        state.mint_history = []
        # (settle_path on the prior cycle already cleared total_mtm and
        # total_max_payout. mint_history is cleared here.)

        forward_open = float(cycle_prices[0])
        if strategy.has_hedge and state.hedge_sleeve > 0:
            open_hedge(
                state, forward=forward_open,
                svi_params=_svi_at(c_lo), anchor=anchor,
            )

        for t in range(path_steps):
            recent = float(cycle_log_rets[t - 1]) if t > 0 else 0.0
            step_path(
                state,
                forward=float(cycle_prices[t]),
                realized_vol=float(cycle_rv[t]),
                sigma_long_run=sigma_long_run,
                log_return_recent=recent,
                svi_params=_svi_at(c_lo + t),
                anchor=anchor,
            )

        cycle_out = settle_path(
            state, settle_price=float(cycle_prices[-1]), anchor=anchor,
        )
        hedge_pnl_total += float(cycle_out["strata_pnl_direct_hedge"])
        cycle_out["cycle_index"] = cycle
        cycle_diagnostics.append(cycle_out)

    final_share_price = (
        float(state.plp.share_price) if state.strata_shares > 0 else initial_share_price
    )
    lp_pnl_total = state.strata_shares * (final_share_price - initial_share_price)
    strata_pnl_total = lp_pnl_total + hedge_pnl_total

    return {
        "strata_pnl_total": float(strata_pnl_total),
        "lp_leg_pnl": float(lp_pnl_total),
        "hedge_leg_pnl": float(hedge_pnl_total),
        "final_share_price": final_share_price,
        "cycles": cycle_diagnostics,
        "n_cycles": n_cycles,
        "strategy_name": strategy.name,
    }


# ---- S4.5/S4.6 — full S4 model: ladder + multi-cycle + token-bucket -----


def run_strategy_s4(
    strategy: StrategyConfig,
    strata_capital: float,
    other_lp_initial: float,
    price_path: np.ndarray,
    realized_vols: np.ndarray,
    sigma_long_run: float,
    log_returns: np.ndarray,
    svi_params: SVIParams,
    anchor: TraderFlowAnchor,
    n_cycles: int = 10,
    path_steps: int = 4,
    ladder_size: int = DEFAULT_LADDER_SIZE,
    moneyness_lo: float = DEFAULT_MONEYNESS_LO,
    moneyness_hi: float = DEFAULT_MONEYNESS_HI,
    kappa_alpha: float = 2.0,
    kappa_beta: float = 5.0,
    bucket_capacity_factor: float = 1.0,
    bucket_refill_factor: float = 1.0,
    seed: int = 0,
) -> dict:
    """Full S4 model: DN ladder + Beta-κ + multi-cycle + token-bucket limiter.

    Combines S4.1 (ladder), S4.2 (Beta κ), S4.3 (multi-cycle), S4.4 (bucket),
    S4.6 (per-strike pricing from deterministic anchor — pass that as
    svi_params). Per-strike DN pricing is automatic: ``size_ladder`` calls
    ``engine.svi_det.dn_price`` vectorized over the ladder strikes.

    Bucket sizing: capacity = θ × bucket_capacity_factor, refill = θ ×
    bucket_refill_factor. Defaults (1.0, 1.0) reduce to the S1 hard-cap
    equivalent for regression.

    Per cycle (N cycles per path):
      1. Build the M-strike ladder via size_ladder, sized to a
         §3-offset-target premium budget (S1 Task A — generalized to the
         mean-κ u_k estimate).
      2. Sample per-strike κ from Beta(α, β) — diagnostic only at this
         S4 form; the LP-leg's f-share recapture is already
         automatic via the share-price accounting.
      3. Post premium / MTM / max_payout to the PLP vault state.
      4. Step path_steps via step_path(..., bucket=bucket).
      5. Settle the ladder inline (per-strike DN payoff sum) AND the
         non-hedge mint_history (path-aware settlement — the S2 money-
         printer fix carries through).
      6. Clear PLP MTM and max_payout for the next cycle.

    Returns dict with strata_pnl_total + lp_leg_pnl + hedge_leg_pnl +
    cycles + n_cycles + strategy_name + ladder_diagnostics.
    """
    expected_len = n_cycles * path_steps
    if len(log_returns) != expected_len:
        raise ValueError(
            f"log_returns len ({len(log_returns)}) != {expected_len}"
        )
    if len(price_path) != expected_len + 1:
        raise ValueError(f"price_path len wrong: need {expected_len + 1}")
    if len(realized_vols) != expected_len:
        raise ValueError(f"realized_vols len wrong: need {expected_len}")

    rng = np.random.default_rng(seed)
    cfg = AccountConfig(
        strata_capital=strata_capital,
        other_lp_initial=other_lp_initial,
        sleeve_plp=strategy.sleeve_plp,
        sleeve_hedge=strategy.sleeve_hedge,
        sleeve_reserve=strategy.sleeve_reserve,
        hedge_moneyness=moneyness_lo,  # placeholder; ladder overrides per-strike
    )
    state = initialize(cfg)

    # Build token bucket from anchor.theta_limiter + multipliers.
    theta = float(anchor.theta_limiter)
    bucket = TokenBucket(
        capacity=theta * bucket_capacity_factor,
        refill_rate=theta * bucket_refill_factor,
        tokens=theta * bucket_capacity_factor,
    )

    hedge_pnl_total = 0.0
    cycle_diagnostics: list[dict] = []

    for cycle in range(n_cycles):
        c_lo = cycle * path_steps
        c_hi = c_lo + path_steps
        cycle_prices = price_path[c_lo : c_hi + 1]
        cycle_log_rets = log_returns[c_lo : c_hi]
        cycle_rv = realized_vols[c_lo : c_hi]

        # Reset per-cycle ephemera.
        state.hedge = None
        state.mint_history = []

        ladder = None
        if strategy.has_hedge and state.hedge_sleeve > 0:
            forward_open = float(cycle_prices[0])
            util_pre = (
                state.plp.total_mtm / state.plp.balance
                if state.plp.balance > 0 else 0.0
            )
            f_pre = state.f
            # Use the mean κ for the u_k estimate driving the offset target.
            mean_kappa = kappa_alpha / (kappa_alpha + kappa_beta)
            u_k_mean = float(u_strike_local(mean_kappa, f_pre))
            # Conservatism: use the DEEPEST strike depth in the band so the
            # premium budget covers the deepest PLP loss the ladder targets.
            crash_depth = 1.0 - moneyness_lo
            mid_strike = forward_open * 0.5 * (moneyness_lo + moneyness_hi)
            mid_check = float(dn_price(mid_strike, forward_open, svi_params))
            ask_check = float(ask_price(mid_check, util_pre))
            premium_budget = hedge_premium_target(
                strata_supply=state.strata_supply,
                f=f_pre, u_k=u_k_mean, crash_depth=crash_depth,
                ask_per_contract=ask_check, sleeve_cap=state.hedge_sleeve,
            )
            if premium_budget > 0:
                # κ per strike — diagnostic; doesn't affect the pool-share
                # accounting because non-hedge flow is still ATM in S4 minimum.
                kappa_per_strike = sample_kappa_ladder(
                    ladder_size, alpha=kappa_alpha, beta=kappa_beta, rng=rng
                )
                ladder = size_ladder(
                    sleeve_budget=premium_budget,
                    forward=forward_open, svi_params=svi_params, util=util_pre,
                    n_strikes=ladder_size,
                    moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,
                )
                total_notional = float(np.sum(ladder.notional_per_strike))
                total_mtm_add = float(np.sum(
                    ladder.notional_per_strike * ladder.dn_mid_per_strike
                ))
                state.plp.receive_premium(ladder.total_premium_paid)
                state.plp.update_mtm(state.plp.total_mtm + total_mtm_add)
                state.plp.update_max_payout(
                    state.plp.total_max_payout + total_notional
                )
                state.hedge_sleeve -= ladder.total_premium_paid
                state.nav_high_watermark = max(
                    state.nav_high_watermark, state.plp.nav
                )

        for t in range(path_steps):
            recent = float(cycle_log_rets[t - 1]) if t > 0 else 0.0
            step_path(
                state,
                forward=float(cycle_prices[t]),
                realized_vol=float(cycle_rv[t]),
                sigma_long_run=sigma_long_run,
                log_return_recent=recent,
                svi_params=svi_params,
                anchor=anchor,
                bucket=bucket,
            )

        settle_price = float(cycle_prices[-1])
        non_hedge_payout = 0.0
        for mint_spot, dn_notional, up_notional in state.mint_history:
            if settle_price < mint_spot:
                non_hedge_payout += dn_notional
            elif settle_price > mint_spot:
                non_hedge_payout += up_notional
        if non_hedge_payout > 0.0:
            state.plp.pay_settlement(non_hedge_payout)

        ladder_payoff = 0.0
        ladder_premium = 0.0
        if ladder is not None:
            ladder_payoff = ladder.payoff(settle_price)
            ladder_premium = ladder.total_premium_paid
            state.plp.pay_settlement(ladder_payoff)
        state.plp.update_mtm(0.0)
        state.plp.update_max_payout(0.0)
        state.mint_history = []

        cycle_hedge_pnl = ladder_payoff - ladder_premium
        hedge_pnl_total += cycle_hedge_pnl
        cycle_diagnostics.append({
            "cycle_index": cycle,
            "settle_price": settle_price,
            "ladder_premium_paid": ladder_premium,
            "ladder_payoff": ladder_payoff,
            "ladder_strikes": (
                ladder.strikes.tolist() if ladder is not None else None
            ),
            "hedge_pnl_this_cycle": cycle_hedge_pnl,
            "f_after": state.f,
            "share_price_after_cycle": state.plp.share_price,
        })

    final_share_price = (
        float(state.plp.share_price) if state.strata_shares > 0 else 1.0
    )
    lp_pnl_total = state.strata_shares * (final_share_price - 1.0)
    strata_pnl_total = lp_pnl_total + hedge_pnl_total

    return {
        "strata_pnl_total": float(strata_pnl_total),
        "lp_leg_pnl": float(lp_pnl_total),
        "hedge_leg_pnl": float(hedge_pnl_total),
        "final_share_price": final_share_price,
        "cycles": cycle_diagnostics,
        "n_cycles": n_cycles,
        "strategy_name": strategy.name,
    }
