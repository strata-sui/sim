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
from eval.strategy import RAW_PLP, STRATA, run_strategy
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
