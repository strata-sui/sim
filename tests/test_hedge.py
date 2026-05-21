"""Unit tests for ``sim.model.hedge`` — single-strike OTM-DN hedge."""
from __future__ import annotations

import pytest

from engine.svi_det import ANCHOR_BTC_2026_05_15
from model.hedge import HedgePosition, size_hedge


FORWARD = 100_000.0


# ---- HedgePosition payoff -----------------------------------------------


class TestPayoff:
    def test_itm_pays_notional(self):
        h = HedgePosition(
            strike=85_000.0,
            notional=1000.0,
            premium_paid=50.0,
            ask_per_contract=0.05,
            dn_mid=0.04,
        )
        assert h.payoff(settle_price=80_000.0) == 1000.0  # ITM
        assert h.is_in_the_money(80_000.0)

    def test_otm_pays_zero(self):
        h = HedgePosition(
            strike=85_000.0,
            notional=1000.0,
            premium_paid=50.0,
            ask_per_contract=0.05,
            dn_mid=0.04,
        )
        assert h.payoff(settle_price=90_000.0) == 0.0  # OTM
        assert not h.is_in_the_money(90_000.0)

    def test_at_strike_is_otm(self):
        """Exactly-at-strike is NOT ITM (strict <)."""
        h = HedgePosition(85_000.0, 1000.0, 50.0, 0.05, 0.04)
        assert not h.is_in_the_money(85_000.0)
        assert h.payoff(85_000.0) == 0.0

    def test_net_pnl_in_crash(self):
        """Net hedge PnL = payoff - premium when ITM."""
        h = HedgePosition(85_000.0, 1000.0, 50.0, 0.05, 0.04)
        assert h.net_pnl(settle_price=70_000.0) == 950.0  # 1000 - 50

    def test_net_pnl_no_crash(self):
        """Net hedge PnL = -premium when OTM."""
        h = HedgePosition(85_000.0, 1000.0, 50.0, 0.05, 0.04)
        assert h.net_pnl(settle_price=110_000.0) == -50.0


# ---- size_hedge ---------------------------------------------------------


class TestSizeHedge:
    def test_full_budget_deployed(self):
        """notional * ask_per_contract ≈ sleeve_budget (full deploy)."""
        h = size_hedge(
            sleeve_budget=150_000.0,
            forward=FORWARD,
            moneyness=0.85,
            util=0.0,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        assert h.premium_paid == pytest.approx(150_000.0)
        assert h.notional * h.ask_per_contract == pytest.approx(150_000.0)

    def test_strike_at_moneyness(self):
        """strike = forward * moneyness."""
        h = size_hedge(
            sleeve_budget=1000.0,
            forward=FORWARD,
            moneyness=0.85,
            util=0.0,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        assert h.strike == pytest.approx(85_000.0)

    def test_deeper_otm_lower_ask_more_notional(self):
        """For fixed budget: deeper OTM (smaller moneyness) → lower DN mid
        → lower ask → more contracts. Verifies the convexity story behind
        the f-Curve hump."""
        h_shallow = size_hedge(
            sleeve_budget=1000.0,
            forward=FORWARD,
            moneyness=0.95,  # 5% OTM
            util=0.0,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        h_deep = size_hedge(
            sleeve_budget=1000.0,
            forward=FORWARD,
            moneyness=0.70,  # 30% OTM
            util=0.0,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        assert h_deep.notional > h_shallow.notional
        assert h_deep.ask_per_contract < h_shallow.ask_per_contract

    def test_util_inflates_ask(self):
        """Higher utilization → larger spread → fewer contracts for same budget."""
        h_low_util = size_hedge(
            sleeve_budget=1000.0,
            forward=FORWARD,
            moneyness=0.85,
            util=0.0,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        h_hi_util = size_hedge(
            sleeve_budget=1000.0,
            forward=FORWARD,
            moneyness=0.85,
            util=0.80,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        assert h_hi_util.ask_per_contract > h_low_util.ask_per_contract
        assert h_hi_util.notional < h_low_util.notional

    def test_dn_mid_recorded(self):
        h = size_hedge(
            sleeve_budget=1000.0,
            forward=FORWARD,
            moneyness=0.85,
            util=0.0,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        # dn_mid recorded should match a fresh price call
        from engine.svi_det import dn_price
        dn = float(dn_price(85_000.0, FORWARD, ANCHOR_BTC_2026_05_15))
        assert h.dn_mid == pytest.approx(dn)


class TestSizeHedgeValidation:
    def test_zero_budget_rejected(self):
        with pytest.raises(ValueError, match="sleeve_budget"):
            size_hedge(
                sleeve_budget=0.0,
                forward=FORWARD,
                moneyness=0.85,
                util=0.0,
                svi_params=ANCHOR_BTC_2026_05_15,
            )

    def test_atm_moneyness_rejected(self):
        with pytest.raises(ValueError, match="moneyness"):
            size_hedge(
                sleeve_budget=1000.0,
                forward=FORWARD,
                moneyness=1.0,
                util=0.0,
                svi_params=ANCHOR_BTC_2026_05_15,
            )

    def test_itm_moneyness_rejected(self):
        """moneyness > 1 would be ITM-DN, not the OTM hedge regime."""
        with pytest.raises(ValueError, match="moneyness"):
            size_hedge(
                sleeve_budget=1000.0,
                forward=FORWARD,
                moneyness=1.5,
                util=0.0,
                svi_params=ANCHOR_BTC_2026_05_15,
            )

    def test_zero_forward_rejected(self):
        with pytest.raises(ValueError, match="forward"):
            size_hedge(
                sleeve_budget=1000.0,
                forward=0.0,
                moneyness=0.85,
                util=0.0,
                svi_params=ANCHOR_BTC_2026_05_15,
            )


# ---- Round-trip: open + settle ------------------------------------------


class TestRoundTrip:
    def test_open_settle_crash_realizes_payoff(self):
        """Open hedge with sleeve, settle below strike → net payoff > 0."""
        h = size_hedge(
            sleeve_budget=10_000.0,
            forward=FORWARD,
            moneyness=0.85,
            util=0.0,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        # Crash: settle at $50k (well below 85k strike)
        net = h.net_pnl(settle_price=50_000.0)
        # net = notional - premium = N(1 - ask). Since ask < 1 by ask-bound
        # (clip [0.01, 0.99]), net is positive (and large for OTM where ask is small).
        assert net > 0
        assert net == h.notional - h.premium_paid

    def test_open_settle_no_crash_loses_premium(self):
        h = size_hedge(
            sleeve_budget=10_000.0,
            forward=FORWARD,
            moneyness=0.85,
            util=0.0,
            svi_params=ANCHOR_BTC_2026_05_15,
        )
        net = h.net_pnl(settle_price=120_000.0)  # No crash, stays above strike
        assert net == -h.premium_paid
