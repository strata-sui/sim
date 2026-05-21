"""Unit tests for ``sim.model.plp`` — verified PLP-vault accounting.

Pinned invariants (CLAUDE.md §4):
  * NAV = balance - total_mtm; share_price = NAV / shares.
  * available_for_withdraw = max(0, balance - total_max_payout).
  * Supply at bootstrap is 1:1; subsequent supplies are NAV-proportional.
  * Withdraw is blocked when amount > available (limiter binds) — this
    is the §2A R3 liquidity-escape-hatch motivator.
  * Premium / settlement / MTM updates do not silently corrupt state.
"""
from __future__ import annotations

import pytest

from model.plp import PLPVault, PLPWithdrawBlocked


# ---- Bootstrap + NAV ----------------------------------------------------


class TestBootstrap:
    def test_empty_vault_defaults(self):
        v = PLPVault()
        assert v.balance == 0.0
        assert v.total_mtm == 0.0
        assert v.total_max_payout == 0.0
        assert v.shares_outstanding == 0.0

    def test_share_price_bootstraps_at_one(self):
        v = PLPVault()
        assert v.share_price == 1.0

    def test_nav_equals_balance_minus_mtm(self):
        v = PLPVault(balance=1_000_000.0, total_mtm=50_000.0)
        assert v.nav == 950_000.0

    def test_share_price_after_one_supply(self):
        v = PLPVault()
        v.supply(100.0)
        assert v.share_price == 1.0  # 1:1 bootstrap


# ---- Supply ------------------------------------------------------------


class TestSupply:
    def test_one_to_one_bootstrap(self):
        v = PLPVault()
        shares = v.supply(1000.0)
        assert shares == 1000.0
        assert v.balance == 1000.0
        assert v.shares_outstanding == 1000.0

    def test_subsequent_supply_nav_proportional(self):
        """After bootstrap with 1000 supply, NAV grows by 100 of premium.
        New supply of 1000 should mint 1000 / (1100/1000) = ~909 shares."""
        v = PLPVault()
        v.supply(1000.0)
        v.receive_premium(100.0)  # NAV now 1100
        new_shares = v.supply(1000.0)
        assert new_shares == pytest.approx(1000.0 * 1000.0 / 1100.0)

    def test_supply_into_negative_nav_rejected(self):
        """If NAV is negative (mtm > balance), cannot supply meaningfully."""
        v = PLPVault(balance=500.0, total_mtm=1000.0, shares_outstanding=500.0)
        with pytest.raises(ValueError, match="non-positive NAV"):
            v.supply(100.0)

    def test_zero_supply_rejected(self):
        v = PLPVault()
        with pytest.raises(ValueError, match="supply amount"):
            v.supply(0.0)

    def test_negative_supply_rejected(self):
        v = PLPVault()
        with pytest.raises(ValueError, match="supply amount"):
            v.supply(-100.0)


# ---- Withdraw + limiter ------------------------------------------------


class TestWithdraw:
    def test_normal_withdraw(self):
        v = PLPVault()
        v.supply(1000.0)
        amount = v.withdraw(500.0)
        assert amount == 500.0
        assert v.shares_outstanding == 500.0
        assert v.balance == 500.0

    def test_withdraw_at_higher_share_price(self):
        """After premium gain, withdrawing 500 shares pays > 500 USD."""
        v = PLPVault()
        v.supply(1000.0)
        v.receive_premium(100.0)  # NAV 1100, share_price 1.10
        amount = v.withdraw(500.0)
        assert amount == pytest.approx(550.0)

    def test_zero_shares_rejected(self):
        v = PLPVault()
        v.supply(1000.0)
        with pytest.raises(ValueError, match="withdraw shares"):
            v.withdraw(0.0)

    def test_overdraw_rejected(self):
        v = PLPVault()
        v.supply(1000.0)
        with pytest.raises(ValueError, match="cannot withdraw"):
            v.withdraw(2000.0)

    def test_limiter_blocks_when_payout_eats_balance(self):
        """available = balance - total_max_payout. If total_max_payout
        equals balance, available = 0 -> withdraw blocked."""
        v = PLPVault()
        v.supply(1000.0)
        v.update_max_payout(1000.0)  # all balance pre-pledged to payouts
        with pytest.raises(PLPWithdrawBlocked):
            v.withdraw(500.0)

    def test_limiter_passes_when_room(self):
        v = PLPVault()
        v.supply(1000.0)
        v.update_max_payout(400.0)  # only 600 free
        amount = v.withdraw(500.0)
        assert amount == 500.0

    def test_limiter_exception_carries_diagnostics(self):
        v = PLPVault()
        v.supply(1000.0)
        v.update_max_payout(1000.0)
        try:
            v.withdraw(300.0)
            pytest.fail("expected PLPWithdrawBlocked")
        except PLPWithdrawBlocked as e:
            assert e.amount_requested == 300.0
            assert e.available == 0.0


# ---- Premium / settlement ---------------------------------------------


class TestPremiumSettlement:
    def test_receive_premium_increases_balance(self):
        v = PLPVault(balance=1000.0, shares_outstanding=1000.0)
        v.receive_premium(50.0)
        assert v.balance == 1050.0

    def test_zero_premium_ok(self):
        v = PLPVault()
        v.receive_premium(0.0)
        assert v.balance == 0.0

    def test_negative_premium_rejected(self):
        v = PLPVault()
        with pytest.raises(ValueError, match="premium amount"):
            v.receive_premium(-1.0)

    def test_pay_settlement_decreases_balance(self):
        v = PLPVault(balance=1000.0, shares_outstanding=1000.0)
        v.pay_settlement(300.0)
        assert v.balance == 700.0

    def test_negative_settlement_rejected(self):
        v = PLPVault(balance=1000.0)
        with pytest.raises(ValueError, match="settlement amount"):
            v.pay_settlement(-1.0)


# ---- MTM + max_payout --------------------------------------------------


class TestMtmUpdates:
    def test_mtm_drops_nav(self):
        v = PLPVault(balance=1000.0, shares_outstanding=1000.0)
        assert v.nav == 1000.0
        v.update_mtm(100.0)
        assert v.nav == 900.0

    def test_negative_mtm_rejected(self):
        v = PLPVault()
        with pytest.raises(ValueError, match="total_mtm"):
            v.update_mtm(-1.0)

    def test_max_payout_zero_initially(self):
        v = PLPVault(balance=1000.0)
        assert v.available_for_withdraw == 1000.0

    def test_max_payout_eats_available(self):
        v = PLPVault(balance=1000.0)
        v.update_max_payout(600.0)
        assert v.available_for_withdraw == 400.0

    def test_max_payout_exceeds_balance_clamps_at_zero(self):
        v = PLPVault(balance=1000.0)
        v.update_max_payout(2000.0)
        assert v.available_for_withdraw == 0.0

    def test_negative_max_payout_rejected(self):
        v = PLPVault()
        with pytest.raises(ValueError, match="total_max_payout"):
            v.update_max_payout(-1.0)


# ---- End-to-end conservation ------------------------------------------


class TestConservation:
    def test_supply_then_full_withdraw_returns_capital(self):
        """Cap-in == cap-out under quiescent state (no MTM, no payouts)."""
        v = PLPVault()
        shares = v.supply(1234.56)
        amount = v.withdraw(shares)
        assert amount == pytest.approx(1234.56)
        assert v.shares_outstanding == pytest.approx(0.0)
        assert v.balance == pytest.approx(0.0)

    def test_premium_revenue_lifts_share_price(self):
        """Premium accrues to LP share price (the PLP-yield mechanic)."""
        v = PLPVault()
        v.supply(1000.0)
        pre = v.share_price
        v.receive_premium(50.0)
        post = v.share_price
        assert post > pre
        assert post == pytest.approx(1.05)

    def test_settlement_drops_share_price(self):
        v = PLPVault()
        v.supply(1000.0)
        v.pay_settlement(50.0)
        assert v.share_price == pytest.approx(0.95)
