"""Unit tests for ``sim.eval.account`` — principled hedge sizing (S1 Task A).

The headline is the §2A budget rule: ``open_hedge`` must NOT spend the full
15% sleeve per cycle. It sizes the per-cycle premium via the §3 offset
target and leaves the remainder idle. These tests pin that.
"""
from __future__ import annotations

import pytest

from engine.svi_det import ANCHOR_BTC_2026_05_15
from eval.account import (
    AccountConfig,
    hedge_premium_target,
    initialize,
    open_hedge,
)
from model.trader_flow import CONSERVATIVE_ANCHOR, u_strike_local


class TestHedgePremiumTarget:
    """N = f·E[PLP_loss]/(1 − f/u); premium = N·ask, capped at sleeve."""

    def test_offset_formula_matches_closed_form(self):
        """Premium = f·(supply·depth)/(1−f/u)·ask when below the cap."""
        supply, f, crash_depth, ask, sleeve = 800_000.0, 0.5, 0.02, 0.15, 150_000.0
        u_k = float(u_strike_local(0.10, f))  # conservative kappa
        prem = hedge_premium_target(supply, f, u_k, crash_depth, ask, sleeve)
        n_target = (f * supply * crash_depth) / (1.0 - f / u_k)
        assert prem == pytest.approx(n_target * ask)
        assert prem < sleeve  # the whole point: NOT full-sleeve

    def test_premium_far_below_full_sleeve(self):
        """In the normal regime the per-cycle premium is a small fraction of
        the sleeve (sleeve is a multi-roll budget)."""
        prem = hedge_premium_target(
            strata_supply=800_000.0,
            f=0.5,
            u_k=float(u_strike_local(0.10, 0.5)),
            crash_depth=0.02,
            ask_per_contract=0.15,
            sleeve_cap=150_000.0,
        )
        assert prem < 0.2 * 150_000.0  # well under 20% of the sleeve

    def test_near_wash_caps_at_sleeve(self):
        """As f → u(k) (kappa→1), the offset target diverges → cap binds."""
        # kappa=1 → u = f → 1 - f/u = 0 → degenerate
        u_k = float(u_strike_local(1.0, 0.5))  # = 0.5 = f
        prem = hedge_premium_target(
            strata_supply=800_000.0,
            f=0.5,
            u_k=u_k,
            crash_depth=0.02,
            ask_per_contract=0.15,
            sleeve_cap=150_000.0,
        )
        assert prem == pytest.approx(150_000.0)  # capped

    def test_premium_grows_with_f(self):
        """Higher f → larger PLP crash exposure → larger hedge premium."""
        kw = dict(strata_supply=800_000.0, crash_depth=0.02,
                  ask_per_contract=0.15, sleeve_cap=150_000.0)
        p_lo = hedge_premium_target(f=0.10, u_k=float(u_strike_local(0.10, 0.10)), **kw)
        p_hi = hedge_premium_target(f=0.50, u_k=float(u_strike_local(0.10, 0.50)), **kw)
        assert p_hi > p_lo

    def test_zero_ask_returns_zero(self):
        prem = hedge_premium_target(800_000.0, 0.5, 0.95, 0.02, 0.0, 150_000.0)
        assert prem == 0.0


class TestOpenHedgeBudget:
    """open_hedge deploys only the targeted premium; remainder stays idle."""

    def _cfg(self, **kw):
        base = dict(
            strata_capital=200_000.0,
            other_lp_initial=200_000.0,  # f0 = 0.5
            sleeve_plp=0.80,
            sleeve_hedge=0.15,
            sleeve_reserve=0.05,
            hedge_moneyness=0.98,
        )
        base.update(kw)
        return AccountConfig(**base)

    def test_does_not_spend_full_sleeve(self):
        state = initialize(self._cfg())
        sleeve_before = state.hedge_sleeve
        h = open_hedge(
            state, forward=60_000.0, svi_params=ANCHOR_BTC_2026_05_15,
            anchor=CONSERVATIVE_ANCHOR,
        )
        # Premium is a small fraction; most of the sleeve remains idle.
        assert h.premium_paid < sleeve_before
        assert state.hedge_sleeve > 0.0
        assert state.hedge_sleeve == pytest.approx(sleeve_before - h.premium_paid)

    def test_premium_equals_notional_times_ask(self):
        state = initialize(self._cfg())
        h = open_hedge(
            state, forward=60_000.0, svi_params=ANCHOR_BTC_2026_05_15,
            anchor=CONSERVATIVE_ANCHOR,
        )
        assert h.premium_paid == pytest.approx(h.notional * h.ask_per_contract)

    def test_pool_receives_premium(self):
        state = initialize(self._cfg())
        bal_before = state.plp.balance
        h = open_hedge(
            state, forward=60_000.0, svi_params=ANCHOR_BTC_2026_05_15,
            anchor=CONSERVATIVE_ANCHOR,
        )
        assert state.plp.balance == pytest.approx(bal_before + h.premium_paid)

    def test_double_open_rejected(self):
        state = initialize(self._cfg())
        open_hedge(state, forward=60_000.0, svi_params=ANCHOR_BTC_2026_05_15,
                   anchor=CONSERVATIVE_ANCHOR)
        with pytest.raises(RuntimeError, match="already open"):
            open_hedge(state, forward=60_000.0, svi_params=ANCHOR_BTC_2026_05_15,
                       anchor=CONSERVATIVE_ANCHOR)
