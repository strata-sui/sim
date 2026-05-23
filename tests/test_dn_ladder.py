"""Unit tests for ``sim.model.dn_ladder`` — multi-strike DN ladder (S4.1)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.svi_det import ANCHOR_BTC_2026_05_15
from model.dn_ladder import (
    DEFAULT_LADDER_SIZE,
    DEFAULT_MONEYNESS_HI,
    DEFAULT_MONEYNESS_LO,
    DnLadder,
    size_ladder,
)


FORWARD = 60_000.0


def _build_default(budget=10_000.0, n=5):
    return size_ladder(
        sleeve_budget=budget,
        forward=FORWARD,
        svi_params=ANCHOR_BTC_2026_05_15,
        util=0.0,
        n_strikes=n,
    )


class TestSizeLadderConstruction:
    def test_strikes_in_loss_onset_band(self):
        L = _build_default()
        # Strikes inside [m_lo * F, m_hi * F]
        assert np.all(L.strikes >= DEFAULT_MONEYNESS_LO * FORWARD - 1e-6)
        assert np.all(L.strikes <= DEFAULT_MONEYNESS_HI * FORWARD + 1e-6)
        assert L.n_strikes == 5

    def test_strikes_strictly_ascending(self):
        L = _build_default()
        assert np.all(np.diff(L.strikes) > 0)

    def test_uniform_log_moneyness_spacing(self):
        """log-strike spacing should be uniform (within numerical tolerance)."""
        L = _build_default(n=7)
        log_strikes = np.log(L.strikes)
        diffs = np.diff(log_strikes)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-9)

    def test_premium_sums_to_budget(self):
        budget = 12_345.67
        L = _build_default(budget=budget)
        assert L.total_premium_paid == pytest.approx(budget, rel=1e-9)

    def test_single_strike_collapses_to_s1_form(self):
        L = _build_default(n=1)
        assert L.n_strikes == 1
        # Premium = budget, single strike at midpoint of band.
        assert L.total_premium_paid == pytest.approx(10_000.0)

    def test_zero_budget_rejected(self):
        with pytest.raises(ValueError, match="sleeve_budget"):
            size_ladder(0.0, FORWARD, ANCHOR_BTC_2026_05_15)

    def test_bad_moneyness_range_rejected(self):
        with pytest.raises(ValueError, match="moneyness"):
            size_ladder(1.0, FORWARD, ANCHOR_BTC_2026_05_15,
                        moneyness_lo=0.95, moneyness_hi=0.90)
        with pytest.raises(ValueError, match="moneyness"):
            size_ladder(1.0, FORWARD, ANCHOR_BTC_2026_05_15,
                        moneyness_lo=1.05, moneyness_hi=1.10)

    def test_zero_n_strikes_rejected(self):
        with pytest.raises(ValueError, match="n_strikes"):
            size_ladder(1.0, FORWARD, ANCHOR_BTC_2026_05_15, n_strikes=0)


class TestPayoff:
    def test_settle_below_all_strikes_pays_full_notional(self):
        L = _build_default()
        # 50_000 is well below the lowest strike (~ 57570 = 0.9595 * 60000).
        assert L.payoff(50_000.0) == pytest.approx(L.total_notional)

    def test_settle_above_all_strikes_pays_zero(self):
        L = _build_default()
        assert L.payoff(70_000.0) == 0.0

    def test_payoff_monotonic_nonincreasing(self):
        L = _build_default()
        settles = np.linspace(50_000, 65_000, 50)
        payoffs = [L.payoff(s) for s in settles]
        assert all(p2 <= p1 for p1, p2 in zip(payoffs, payoffs[1:]))

    def test_partial_itm_pays_subset(self):
        L = _build_default(n=5)
        # Settle slightly above the lowest strike but below the highest.
        s = 0.5 * (L.strikes[0] + L.strikes[-1])
        mask = L.itm_mask(s)
        expected = float(np.sum(L.notional_per_strike[mask]))
        assert L.payoff(s) == pytest.approx(expected)


class TestDeeperLadderPaysMoreOnDeepCrash:
    def test_deeper_band_more_protection_on_deep_crash(self):
        """A ladder spanning a DEEPER OTM band must pay MORE on a deeper crash."""
        shallow = size_ladder(
            sleeve_budget=10_000.0, forward=FORWARD,
            svi_params=ANCHOR_BTC_2026_05_15,
            n_strikes=5, moneyness_lo=0.99, moneyness_hi=0.995,
        )
        deep = size_ladder(
            sleeve_budget=10_000.0, forward=FORWARD,
            svi_params=ANCHOR_BTC_2026_05_15,
            n_strikes=5, moneyness_lo=0.95, moneyness_hi=0.97,
        )
        # On a -5% crash (settle = 57k), the deep ladder is FULLY ITM, the
        # shallow one is FULLY OTM.
        assert deep.payoff(57_000.0) > shallow.payoff(57_000.0)
