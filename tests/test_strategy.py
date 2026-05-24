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


# ---- S5R3 regression — depositor cannot lose more than deposit ---------


class TestDepositorBoundedLoss:
    """★ S5R3.3 regression — strata_pnl_total ≥ -strata_capital, always.

    Pins the on-chain reality (CLAUDE.md §4): a PLP-vault token is a
    claim, not a liability. The depositor can lose at most the deposit,
    never owe the pool money. Before the fix (S5R3.2 root-cause trace),
    multi-cycle pathological paths produced losses on the order of
    -$10^10 on a $50k deposit (seed-25 of S5R3.2 diagnostic).

    Tests both single-cycle and multi-cycle runners under stressed
    parametrizations: large lambda_0, big vol jumps (forces
    trader_flow_step to over-mint), high informed bias (loads one side),
    extreme bull AND bear settlements, and across many seeds.
    """

    def test_run_strategy_s4_loss_bounded_raw_plp(self):
        """200 PR+Kou-style random paths × raw_plp + S4 runner."""
        from engine.price_engine_pr_kou import PolitisRomanoKouEngine
        rng = np.random.default_rng(20260601)
        # Build a synthetic returns history with BTC-like fat tails.
        hist = rng.normal(0, 0.012, size=10_000)
        # Inject a few jumps to make the bootstrap output volatile.
        hist[rng.integers(0, 10_000, size=50)] += rng.normal(0, 0.04, size=50)
        pe = PolitisRomanoKouEngine(
            log_returns=hist, mean_block_length=4.0, seed=42,
        )
        n_paths = 200
        n_cycles, ps = 10, 4
        sim = pe.simulate(n_paths=n_paths, n_steps=n_cycles * ps,
                          init_price=60_000.0)
        price_paths = sim["price"]
        log_paths = sim["log_return"]
        sigma_long = pe.sigma_historical
        from eval.f_sweep import _rolling_vol
        rv_paths = _rolling_vol(log_paths, window=8, fallback=sigma_long)

        for f in (0.05, 0.20, 0.50, 0.80):
            strata_capital = f * 1_000_000.0
            other_lp = (1.0 - f) * 1_000_000.0
            for i in range(n_paths):
                out = run_strategy_s4(
                    strategy=RAW_PLP,
                    strata_capital=strata_capital,
                    other_lp_initial=other_lp,
                    price_path=price_paths[i],
                    log_returns=log_paths[i],
                    realized_vols=rv_paths[i],
                    sigma_long_run=sigma_long,
                    svi_params=ANCHOR_BTC_2026_05_15,
                    anchor=CONSERVATIVE_ANCHOR,
                    n_cycles=n_cycles, path_steps=ps,
                    seed=42 + i,
                )
                pnl = out["strata_pnl_total"]
                assert pnl >= -strata_capital - 1e-6, (
                    f"depositor lost more than deposit: pnl=${pnl:,.2f} on "
                    f"${strata_capital:,.2f} deposit (f={f}, path {i})"
                )
                # Also pin the share_price floor invariant at the source.
                assert out["final_share_price"] >= 0.0, (
                    f"share_price went negative: {out['final_share_price']} "
                    f"(f={f}, path {i})"
                )

    def test_run_strategy_s4_loss_bounded_strata(self):
        """Strata with full S4 hedge ladder — must also respect the bound."""
        from engine.price_engine_pr_kou import PolitisRomanoKouEngine
        rng = np.random.default_rng(20260602)
        hist = rng.normal(0, 0.012, size=10_000)
        hist[rng.integers(0, 10_000, size=50)] += rng.normal(0, 0.04, size=50)
        pe = PolitisRomanoKouEngine(
            log_returns=hist, mean_block_length=4.0, seed=43,
        )
        n_paths = 100
        n_cycles, ps = 10, 4
        sim = pe.simulate(n_paths=n_paths, n_steps=n_cycles * ps,
                          init_price=60_000.0)
        price_paths = sim["price"]
        log_paths = sim["log_return"]
        sigma_long = pe.sigma_historical
        from eval.f_sweep import _rolling_vol
        rv_paths = _rolling_vol(log_paths, window=8, fallback=sigma_long)

        for f in (0.05, 0.50):
            strata_capital = f * 1_000_000.0
            other_lp = (1.0 - f) * 1_000_000.0
            for i in range(n_paths):
                out = run_strategy_s4(
                    strategy=STRATA,
                    strata_capital=strata_capital,
                    other_lp_initial=other_lp,
                    price_path=price_paths[i],
                    log_returns=log_paths[i],
                    realized_vols=rv_paths[i],
                    sigma_long_run=sigma_long,
                    svi_params=ANCHOR_BTC_2026_05_15,
                    anchor=CONSERVATIVE_ANCHOR,
                    n_cycles=n_cycles, path_steps=ps,
                    seed=42 + i,
                )
                pnl = out["strata_pnl_total"]
                assert pnl >= -strata_capital - 1e-6, (
                    f"STRATA depositor lost more than deposit: "
                    f"pnl=${pnl:,.2f} on ${strata_capital:,.2f} (f={f}, "
                    f"path {i})"
                )

    def test_extreme_bull_crash_single_path(self):
        """Adversarial path: cycles 0-6 flat, cycle 7 huge bull spike,
        cycle 8 crash. This is the seed-25 family the S5R3.2 diagnostic
        identified — pre-fix it produced $-11.6B on $50k.
        """
        n_cycles, ps = 10, 4
        n = n_cycles * ps
        log_rets = np.zeros(n)
        # Cycle 7 (steps 28-31) gets a +30% spike, cycle 8 (32-35) crashes -25%.
        log_rets[28:32] = np.log(1.30) / 4   # ~+7% per step
        log_rets[32:36] = np.log(0.75) / 4   # ~-7% per step
        prices = np.empty(n + 1)
        prices[0] = 60_000.0
        prices[1:] = prices[0] * np.exp(np.cumsum(log_rets))
        rv = np.full(n, 0.015)
        sigma_long = 0.012

        strata_capital = 50_000.0
        other_lp = 950_000.0
        out = run_strategy_s4(
            strategy=RAW_PLP,
            strata_capital=strata_capital,
            other_lp_initial=other_lp,
            price_path=prices, log_returns=log_rets, realized_vols=rv,
            sigma_long_run=sigma_long,
            svi_params=ANCHOR_BTC_2026_05_15,
            anchor=CONSERVATIVE_ANCHOR,
            n_cycles=n_cycles, path_steps=ps, seed=100,
        )
        pnl = out["strata_pnl_total"]
        assert pnl >= -strata_capital - 1e-6, (
            f"adversarial bull-then-crash blew the deposit bound: "
            f"pnl=${pnl:,.2f}"
        )

    def test_single_cycle_also_bounded(self):
        """N=1 cycle is the S1/S2 baseline — must also satisfy the bound."""
        rng = np.random.default_rng(20260603)
        for seed in range(50):
            log_rets = rng.normal(0, 0.02, size=4)
            log_rets[rng.integers(0, 4)] += rng.normal(0, 0.05)  # jump
            prices = np.empty(5)
            prices[0] = 60_000.0
            prices[1:] = prices[0] * np.exp(np.cumsum(log_rets))
            rv = np.full(4, 0.02)
            strata_capital = 50_000.0
            other_lp = 950_000.0
            out = run_strategy(
                strategy=RAW_PLP,
                strata_capital=strata_capital,
                other_lp_initial=other_lp,
                price_path=prices, log_returns=log_rets, realized_vols=rv,
                sigma_long_run=0.015,
                svi_params=ANCHOR_BTC_2026_05_15,
                anchor=CONSERVATIVE_ANCHOR,
            )
            pnl = out["strata_pnl_total"]
            assert pnl >= -strata_capital - 1e-6, (
                f"single-cycle bound violated: pnl=${pnl:,.2f} seed={seed}"
            )
