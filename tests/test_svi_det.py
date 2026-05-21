"""Unit tests for ``sim.engine.svi_det`` — Gatheral raw SVI + binary pricing.

Numeric targets derived analytically. Anchor surface is the live BTC
snapshot from CLAUDE.md §4 (verified 2026-05-15).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.svi_det import (
    ANCHOR_BTC_2026_05_15,
    PROTOCOL_SCALE,
    SVIParams,
    dn_price,
    is_arb_free,
    range_price,
    total_variance,
    up_price,
)


class TestSVIParams:
    """Validation + on-chain integer decoding."""

    def test_negative_b_rejected(self):
        with pytest.raises(ValueError, match="b must be"):
            SVIParams(a=0.0, b=-0.001, rho=0.0, m=0.0, sigma=0.01)

    def test_rho_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="rho must be"):
            SVIParams(a=0.0, b=0.001, rho=1.5, m=0.0, sigma=0.01)
        with pytest.raises(ValueError, match="rho must be"):
            SVIParams(a=0.0, b=0.001, rho=-1.5, m=0.0, sigma=0.01)

    def test_zero_sigma_rejected(self):
        with pytest.raises(ValueError, match="sigma must be"):
            SVIParams(a=0.0, b=0.001, rho=0.0, m=0.0, sigma=0.0)

    def test_from_protocol_int_unsigned(self):
        """Decode positive integer encoding (no sign flags)."""
        p = SVIParams.from_protocol_int(
            a=165787, b=7321280, rho=318800000, m=2750000, sigma=14260000
        )
        assert p.a == pytest.approx(165787 / PROTOCOL_SCALE)
        assert p.b == pytest.approx(7321280 / PROTOCOL_SCALE)
        assert p.rho == pytest.approx(0.3188)
        assert p.m == pytest.approx(0.00275)
        assert p.sigma == pytest.approx(0.01426)

    def test_from_protocol_int_with_sign_flags(self):
        """Decode predict-server API form (positive int + *_negative flag)."""
        # Matches Albary's live oracle 0xd153 sample (S0 run, 2026-05-21):
        # rho=940152888, rho_negative=True ; m=1838397, m_negative=True
        p = SVIParams.from_protocol_int(
            a=20325,
            b=237390,
            rho=940152888,
            m=1838397,
            sigma=1665873,
            rho_negative=True,
            m_negative=True,
        )
        assert p.rho == pytest.approx(-0.940152888)
        assert p.m == pytest.approx(-0.001838397)
        assert p.sigma == pytest.approx(0.001665873)

    def test_anchor_is_arb_free(self):
        """The CLAUDE.md §4 anchor must pass the cheap no-arb check."""
        assert is_arb_free(ANCHOR_BTC_2026_05_15)


class TestTotalVariance:
    """w(k) = a + b·(ρ(k-m) + √((k-m)² + σ²))."""

    def test_at_k_equals_m(self):
        """At k=m: w = a + b·σ (since √(0 + σ²) = σ, ρ·0 = 0)."""
        p = ANCHOR_BTC_2026_05_15
        w = float(total_variance(p.m, p))
        assert w == pytest.approx(p.a + p.b * p.sigma)

    def test_nonnegative_on_anchor(self):
        """Anchor surface: w(k) ≥ 0 across a wide log-moneyness band."""
        k_grid = np.linspace(-0.5, 0.5, 200)
        w = total_variance(k_grid, ANCHOR_BTC_2026_05_15)
        assert np.all(w >= 0)

    def test_vectorized(self):
        k = np.array([-0.1, 0.0, 0.1])
        w = total_variance(k, ANCHOR_BTC_2026_05_15)
        assert w.shape == (3,)

    def test_increasing_in_b(self):
        """Higher wing-slope b → larger w (in expectation, since the √
        term is always positive)."""
        p1 = SVIParams(a=1e-4, b=0.005, rho=-0.3, m=0.0, sigma=0.01)
        p2 = SVIParams(a=1e-4, b=0.010, rho=-0.3, m=0.0, sigma=0.01)
        assert total_variance(0.1, p2) > total_variance(0.1, p1)


class TestBinaryPricing:
    """UP, DN, Range — verified DeepBook Predict formulas."""

    def test_up_plus_dn_equals_one(self):
        """Probability complement: UP + DN = 1 by construction."""
        forward = 100000.0
        strikes = np.array([80000.0, 100000.0, 120000.0])
        up = up_price(strikes, forward, ANCHOR_BTC_2026_05_15)
        dn = dn_price(strikes, forward, ANCHOR_BTC_2026_05_15)
        np.testing.assert_allclose(up + dn, 1.0, atol=1e-12)

    def test_up_atm_near_half(self):
        """At strike = forward (k=0), UP should be near 0.5 (small w → d2≈0)."""
        forward = 100000.0
        up = float(up_price(forward, forward, ANCHOR_BTC_2026_05_15))
        # Anchor has m=-0.00275 and ρ=-0.3188 introducing a small skew,
        # so UP(ATM) is slightly off 0.5 — pin a tight band.
        assert 0.45 < up < 0.55

    def test_up_decreasing_in_strike_tight(self):
        """UP(K) strictly decreasing in K (tight range avoids float saturation)."""
        forward = 100000.0
        strikes = np.linspace(80000.0, 120000.0, 50)
        up = up_price(strikes, forward, ANCHOR_BTC_2026_05_15)
        assert np.all(np.diff(up) < 0)

    def test_up_non_increasing_wide(self):
        """UP(K) non-increasing across a wide range (saturation at ends OK)."""
        forward = 100000.0
        strikes = np.linspace(50000.0, 200000.0, 50)
        up = up_price(strikes, forward, ANCHOR_BTC_2026_05_15)
        # At very low K, UP saturates to 1.0 → consecutive diffs == 0 allowed.
        # At very high K, UP saturates to 0.0 → ditto. Non-increasing is the
        # correct invariant.
        assert np.all(np.diff(up) <= 0)

    def test_dn_increasing_in_strike_tight(self):
        """DN(K) strictly increasing in K (tight range avoids saturation).

        Uses `norm.sf(d2)` directly (svi_det.dn_price) to preserve precision
        in the tail — `1 - up` would catastrophically cancel here.
        """
        forward = 100000.0
        strikes = np.linspace(80000.0, 120000.0, 50)
        dn = dn_price(strikes, forward, ANCHOR_BTC_2026_05_15)
        assert np.all(np.diff(dn) > 0)

    def test_dn_non_decreasing_wide(self):
        """DN(K) non-decreasing across a wide range (saturation OK)."""
        forward = 100000.0
        strikes = np.linspace(50000.0, 200000.0, 50)
        dn = dn_price(strikes, forward, ANCHOR_BTC_2026_05_15)
        assert np.all(np.diff(dn) >= 0)

    def test_up_in_unit_interval(self):
        """Probabilities live in [0, 1] for any sane strike (saturation OK)."""
        forward = 100000.0
        strikes = np.array([1000.0, 50000.0, 100000.0, 200000.0, 1e7])
        up = up_price(strikes, forward, ANCHOR_BTC_2026_05_15)
        assert np.all(up >= 0.0)
        assert np.all(up <= 1.0)

    def test_range_positive_when_lo_below_hi(self):
        """Range(lo, hi) > 0 for lo < hi (UP decreasing → diff positive)."""
        forward = 100000.0
        r = float(range_price(90000.0, 110000.0, forward, ANCHOR_BTC_2026_05_15))
        assert r > 0.0
        assert r < 1.0

    def test_range_decomposes_to_up_difference(self):
        forward = 100000.0
        lo, hi = 90000.0, 110000.0
        r = float(range_price(lo, hi, forward, ANCHOR_BTC_2026_05_15))
        u_lo = float(up_price(lo, forward, ANCHOR_BTC_2026_05_15))
        u_hi = float(up_price(hi, forward, ANCHOR_BTC_2026_05_15))
        assert r == pytest.approx(u_lo - u_hi)

    def test_deep_otm_up_near_zero(self):
        """UP(strike >> forward) → ~0 (very unlikely to expire above)."""
        forward = 100000.0
        up = float(up_price(forward * 100, forward, ANCHOR_BTC_2026_05_15))
        assert up < 0.01

    def test_deep_itm_up_near_one(self):
        """UP(strike << forward) → ~1 (almost certainly expires above)."""
        forward = 100000.0
        up = float(up_price(forward * 0.01, forward, ANCHOR_BTC_2026_05_15))
        assert up > 0.99


class TestPricingErrors:
    def test_forward_zero_rejected(self):
        with pytest.raises(ValueError, match="forward must be"):
            up_price(100.0, 0.0, ANCHOR_BTC_2026_05_15)

    def test_forward_negative_rejected(self):
        with pytest.raises(ValueError, match="forward must be"):
            up_price(100.0, -100.0, ANCHOR_BTC_2026_05_15)

    def test_negative_strike_rejected(self):
        with pytest.raises(ValueError, match="strikes must be"):
            up_price(-100.0, 100.0, ANCHOR_BTC_2026_05_15)


class TestNoArb:
    """Necessary no-butterfly conditions (Gatheral-Jacquier 2013)."""

    def test_wing_slope_violation_caught(self):
        """b·(1+|ρ|) > 2 → reject (Lee's moment formula)."""
        # b=1.5, |ρ|=0.5 → 1.5·1.5 = 2.25 > 2
        p = SVIParams(a=0.001, b=1.5, rho=-0.5, m=0.0, sigma=0.05)
        assert not is_arb_free(p)

    def test_floor_violation_caught(self):
        """a + b·σ·√(1-ρ²) < 0 → reject (negative variance frontier)."""
        # Construct a case where a is very negative, b·σ·√(1-ρ²) small.
        p = SVIParams(a=-0.1, b=0.001, rho=0.0, m=0.0, sigma=0.01)
        # a + 0.001·0.01·1 = -0.1 + 1e-5 = -0.0999 < 0
        assert not is_arb_free(p)

    def test_typical_calibration_passes(self):
        """Realistic SVI params pass."""
        p = SVIParams(a=0.001, b=0.05, rho=-0.3, m=0.0, sigma=0.1)
        assert is_arb_free(p)


class TestVectorizationAndPerformance:
    """Pricing must be vectorized for MC throughput (S5 1M paths)."""

    def test_thousand_strikes_vectorized(self):
        forward = 100000.0
        strikes = np.linspace(50000.0, 200000.0, 1000)
        up = up_price(strikes, forward, ANCHOR_BTC_2026_05_15)
        assert up.shape == (1000,)
        # All in [0,1]; ends may saturate at exactly 0 or 1 by float64 precision.
        assert np.all(up >= 0.0)
        assert np.all(up <= 1.0)
