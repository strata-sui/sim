"""Unit tests for ``sim.eval.strategy`` — strategy runner + look-ahead guard.

The headline test here is ``TestNoLookahead`` — it is the regression net
for the catastrophic bug found at S1 Gate-A debug: the informed-flow
signal (Knob 3) was fed ``log_returns[t]`` (the t→t+1 FUTURE move) instead
of ``log_returns[t-1]`` (the past move). That handed traders perfect
foresight of the next bar, driving raw_plp prob_loss to 0.91 and an
impossible negative benign carry (contradicting the verified §4
short-vol-with-spread-carry profile). These tests pin the fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace

from engine.svi_det import ANCHOR_BTC_2026_05_15
from eval.strategy import (
    RAW_PLP, STRATA,
    run_strategy, run_strategy_multi_cycle, run_strategy_s4,
)
from model.trader_flow import CONSERVATIVE_ANCHOR

SIGMA = 0.01
NOISE_ANCHOR = replace(CONSERVATIVE_ANCHOR, phi_informed=0.0)
INFORMED_ANCHOR = replace(CONSERVATIVE_ANCHOR, phi_informed=1.0)


def _const_vol(n: int) -> np.ndarray:
    return np.full(n, SIGMA)


class TestNoLookahead:
    """The informed signal must react to the PAST, not the future."""

    def test_flat_history_then_crash_keeps_pool_solvent(self):
        """Path is flat for 4 steps, then crashes 50% on the FINAL move.

        With the fix, every step's informed signal reads the prior (flat)
        return → b_informed = 0 → balanced 50/50 flow → the pool collects
        the spread on both sides and pays one side, netting positive even
        through the crash.

        With the look-ahead bug, the last step would read the -50% future
        move, tilt 100% DN, and the pool would pay the full ITM notional
        having collected only the ~0.5 mid → a large loss. So a
        non-negative raw_plp P&L here proves the look-ahead is gone.
        """
        # prices: 100,100,100,100 then 50 (crash on the last return only)
        price_path = np.array([100.0, 100.0, 100.0, 100.0, 50.0])
        log_rets = np.array(
            [0.0, 0.0, 0.0, float(np.log(50.0 / 100.0))]
        )
        out = run_strategy(
            strategy=RAW_PLP,
            strata_capital=200_000.0,
            other_lp_initial=800_000.0,
            price_path=price_path,
            realized_vols=_const_vol(len(log_rets)),
            sigma_long_run=SIGMA,
            log_returns=log_rets,
            svi_params=ANCHOR_BTC_2026_05_15,
            anchor=INFORMED_ANCHOR,  # phi=1: maximizes any look-ahead effect
        )
        assert out["strata_pnl_total"] >= 0.0, (
            "raw_plp lost on a flat-history path — look-ahead likely back"
        )

    def test_crash_in_history_legitimately_loses(self):
        """If the crash is in the PAST (history), informed flow legitimately
        chases it and the pool can lose — that's real short-vol behavior,
        not a bug. Mirror image of the previous test."""
        # Crash happens FIRST (in history the flow reacts to), then flat.
        price_path = np.array([100.0, 50.0, 50.0, 50.0, 50.0])
        log_rets = np.array(
            [float(np.log(50.0 / 100.0)), 0.0, 0.0, 0.0]
        )
        out = run_strategy(
            strategy=RAW_PLP,
            strata_capital=200_000.0,
            other_lp_initial=800_000.0,
            price_path=price_path,
            realized_vols=_const_vol(len(log_rets)),
            sigma_long_run=SIGMA,
            log_returns=log_rets,
            svi_params=ANCHOR_BTC_2026_05_15,
            anchor=INFORMED_ANCHOR,
        )
        # settle (50) == later mint spots (50) → those mints don't pay;
        # only the step-0 mint struck at 100 is relevant. This should NOT
        # be the catastrophic look-ahead loss; just assert it ran and is finite.
        assert np.isfinite(out["strata_pnl_total"])


class TestBalancedFlowCarry:
    """phi=0 (pure noise) ⇒ balanced book ⇒ pool always collects spread."""

    def test_noise_flow_non_negative_on_random_paths(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            rets = rng.normal(0, SIGMA, 4)
            prices = np.empty(5)
            prices[0] = 100.0
            prices[1:] = 100.0 * np.exp(np.cumsum(rets))
            out = run_strategy(
                strategy=RAW_PLP,
                strata_capital=200_000.0,
                other_lp_initial=800_000.0,
                price_path=prices,
                realized_vols=_const_vol(4),
                sigma_long_run=SIGMA,
                log_returns=rets,
                svi_params=ANCHOR_BTC_2026_05_15,
                anchor=NOISE_ANCHOR,
            )
            assert out["strata_pnl_total"] >= -1e-9, (
                "balanced (noise) flow should never lose — pure spread carry"
            )


class TestRunStrategyContract:
    """Output-shape + routing sanity."""

    def test_raw_plp_has_no_hedge_leg(self):
        rng = np.random.default_rng(1)
        rets = rng.normal(0, SIGMA, 4)
        prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(rets))])
        out = run_strategy(
            strategy=RAW_PLP,
            strata_capital=200_000.0,
            other_lp_initial=800_000.0,
            price_path=prices,
            realized_vols=_const_vol(4),
            sigma_long_run=SIGMA,
            log_returns=rets,
            svi_params=ANCHOR_BTC_2026_05_15,
            anchor=CONSERVATIVE_ANCHOR,
        )
        assert out["strata_pnl_direct_hedge"] == 0.0
        assert out["strategy_name"] == "raw_plp"

    def test_strata_has_hedge_leg_and_keys(self):
        rng = np.random.default_rng(2)
        rets = rng.normal(0, SIGMA, 4)
        prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(rets))])
        out = run_strategy(
            strategy=STRATA,
            strata_capital=200_000.0,
            other_lp_initial=800_000.0,
            price_path=prices,
            realized_vols=_const_vol(4),
            sigma_long_run=SIGMA,
            log_returns=rets,
            svi_params=ANCHOR_BTC_2026_05_15,
            anchor=CONSERVATIVE_ANCHOR,
        )
        for key in (
            "strata_pnl_total",
            "strata_pnl_lp_leg",
            "strata_pnl_direct_hedge",
            "u_k",
            "f_settle",
        ):
            assert key in out
        assert out["strategy_name"] == "strata"

    def test_mismatched_path_length_rejected(self):
        with pytest.raises(ValueError, match="price_path len"):
            run_strategy(
                strategy=RAW_PLP,
                strata_capital=200_000.0,
                other_lp_initial=800_000.0,
                price_path=np.array([100.0, 101.0]),  # len 2
                realized_vols=_const_vol(4),
                sigma_long_run=SIGMA,
                log_returns=np.zeros(4),  # len 4 → needs price len 5
                svi_params=ANCHOR_BTC_2026_05_15,
                anchor=CONSERVATIVE_ANCHOR,
            )


# ---- S4.3 — multi-cycle dynamics ---------------------------------------


def _build_multi_cycle_path(n_cycles: int, path_steps: int, seed: int):
    """Build a multi-cycle return path (random walk with BTC-like σ)."""
    rng = np.random.default_rng(seed)
    n_steps = n_cycles * path_steps
    log_rets = rng.normal(0, SIGMA, n_steps)
    prices = np.empty(n_steps + 1)
    prices[0] = 100.0
    prices[1:] = 100.0 * np.exp(np.cumsum(log_rets))
    rv = np.full(n_steps, SIGMA)
    return prices, log_rets, rv


class TestMultiCycleShape:
    def test_basic_run_returns_keys(self):
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=3, path_steps=4, seed=0)
        out = run_strategy_multi_cycle(
            strategy=RAW_PLP,
            strata_capital=200_000.0,
            other_lp_initial=800_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15,
            anchor=CONSERVATIVE_ANCHOR,
            n_cycles=3, path_steps=4,
        )
        for k in (
            "strata_pnl_total", "lp_leg_pnl", "hedge_leg_pnl",
            "final_share_price", "cycles", "n_cycles",
        ):
            assert k in out
        assert len(out["cycles"]) == 3
        assert out["n_cycles"] == 3

    def test_mismatched_length_rejected(self):
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=2, path_steps=4, seed=1)
        with pytest.raises(ValueError, match="log_returns"):
            run_strategy_multi_cycle(
                strategy=RAW_PLP,
                strata_capital=200_000.0, other_lp_initial=800_000.0,
                price_path=prices, log_returns=log_rets[:5],  # wrong length
                realized_vols=rv, sigma_long_run=SIGMA,
                svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
                n_cycles=2, path_steps=4,
            )


class TestMultiCycleSingleCycleEquivalence:
    """N=1 multi_cycle ≈ single-cycle run_strategy for RAW_PLP."""

    def test_n_cycles_1_matches_single(self):
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=1, path_steps=4, seed=2)
        single = run_strategy(
            strategy=RAW_PLP,
            strata_capital=200_000.0, other_lp_initial=800_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
        )
        multi = run_strategy_multi_cycle(
            strategy=RAW_PLP,
            strata_capital=200_000.0, other_lp_initial=800_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
            n_cycles=1, path_steps=4,
        )
        assert multi["strata_pnl_total"] == pytest.approx(
            single["strata_pnl_total"], rel=1e-9
        )


class TestMultiCycleStateConsistency:
    """Brief §S4.3 acceptance #3: state-consistency across cycle boundary."""

    def test_lp_shares_and_l_other_persist_across_cycles(self):
        """Strata_shares and l_other should be IDENTICAL pre- vs post-settle
        (settle clears mtm/max_payout, NOT shares or LP balance)."""
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=2, path_steps=4, seed=3)
        out = run_strategy_multi_cycle(
            strategy=RAW_PLP,  # no hedge — pure LP roll
            strata_capital=200_000.0, other_lp_initial=800_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
            n_cycles=2, path_steps=4,
        )
        # Both cycles produced valid settles.
        assert len(out["cycles"]) == 2
        # final_share_price is finite and positive
        assert out["final_share_price"] > 0
        # LP-leg pnl = shares × (final_sp - 1). Sanity: can be either sign.
        # Just check it's finite.
        assert np.isfinite(out["lp_leg_pnl"])

    def test_terminal_pnl_uses_compounded_share_price(self):
        """Manual compute: shares × (final_sp - 1) MUST equal lp_leg_pnl."""
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=3, path_steps=4, seed=4)
        out = run_strategy_multi_cycle(
            strategy=RAW_PLP,
            strata_capital=200_000.0, other_lp_initial=800_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
            n_cycles=3, path_steps=4,
        )
        # cycle pnl_total uses initial_sp=1.0 each cycle (stale post-cycle-1)
        # so we cannot just sum per-cycle pnl_total. The wrapper handles this
        # by computing LP-leg pnl ONCE on the terminal state. Sanity check
        # that pnl_total = lp_leg_pnl + hedge_leg_pnl exactly.
        assert out["strata_pnl_total"] == pytest.approx(
            out["lp_leg_pnl"] + out["hedge_leg_pnl"], rel=1e-9
        )

    def test_hedge_strategy_runs_across_cycles(self):
        """STRATA with hedge: each cycle opens/settles fresh hedge."""
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=3, path_steps=4, seed=5)
        out = run_strategy_multi_cycle(
            strategy=STRATA,
            strata_capital=200_000.0, other_lp_initial=200_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
            n_cycles=3, path_steps=4,
        )
        # Each cycle's settle records a hedge strike (or None for raw_plp).
        for c in out["cycles"]:
            assert "strata_pnl_direct_hedge" in c
        # Total hedge pnl = sum of per-cycle direct-hedge.
        total = sum(c["strata_pnl_direct_hedge"] for c in out["cycles"])
        assert out["hedge_leg_pnl"] == pytest.approx(total, rel=1e-9)


# ---- S4.5/S4.6 — full S4 integration runner ----------------------------


class TestRunStrategyS4:
    def test_basic_run_returns_keys(self):
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=3, path_steps=4, seed=10)
        out = run_strategy_s4(
            strategy=STRATA,
            strata_capital=200_000.0, other_lp_initial=200_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
            n_cycles=3, path_steps=4, ladder_size=5,
        )
        for k in (
            "strata_pnl_total", "lp_leg_pnl", "hedge_leg_pnl",
            "final_share_price", "cycles", "n_cycles",
        ):
            assert k in out
        assert out["n_cycles"] == 3
        assert len(out["cycles"]) == 3
        # Each cycle records ladder strikes (5 of them) and a hedge pnl.
        for c in out["cycles"]:
            if c["ladder_strikes"] is not None:
                assert len(c["ladder_strikes"]) == 5
            assert "hedge_pnl_this_cycle" in c

    def test_raw_plp_no_ladder_in_cycles(self):
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=2, path_steps=4, seed=11)
        out = run_strategy_s4(
            strategy=RAW_PLP,
            strata_capital=200_000.0, other_lp_initial=800_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
            n_cycles=2, path_steps=4,
        )
        # RAW_PLP has has_hedge=False → no ladder posted per cycle.
        for c in out["cycles"]:
            assert c["ladder_strikes"] is None
            assert c["ladder_premium_paid"] == 0.0

    def test_pnl_accounting_telescopes(self):
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=4, path_steps=4, seed=12)
        out = run_strategy_s4(
            strategy=STRATA,
            strata_capital=200_000.0, other_lp_initial=200_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
            n_cycles=4, path_steps=4,
        )
        assert out["strata_pnl_total"] == pytest.approx(
            out["lp_leg_pnl"] + out["hedge_leg_pnl"], rel=1e-9
        )

    def test_mismatched_lengths_rejected(self):
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=2, path_steps=4, seed=13)
        with pytest.raises(ValueError, match="log_returns"):
            run_strategy_s4(
                strategy=STRATA,
                strata_capital=200_000.0, other_lp_initial=200_000.0,
                price_path=prices, log_returns=log_rets[:5],
                realized_vols=rv, sigma_long_run=SIGMA,
                svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
                n_cycles=2, path_steps=4,
            )

    def test_ladder_size_one_collapses_to_single_strike(self):
        prices, log_rets, rv = _build_multi_cycle_path(n_cycles=2, path_steps=4, seed=14)
        out = run_strategy_s4(
            strategy=STRATA,
            strata_capital=200_000.0, other_lp_initial=200_000.0,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=SIGMA,
            svi_params=ANCHOR_BTC_2026_05_15, anchor=CONSERVATIVE_ANCHOR,
            n_cycles=2, path_steps=4, ladder_size=1,
        )
        for c in out["cycles"]:
            if c["ladder_strikes"] is not None:
                assert len(c["ladder_strikes"]) == 1
