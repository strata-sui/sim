"""Unit tests for ``sim.engine.spread`` — verified spread formula (CLAUDE.md §4).

Numeric targets are derived analytically from the formula:
    spread(p, util) = max(base · √(p(1-p)), min) + base · util_mult · util²

These tests serve a double purpose: they pin behavior AND they document
the term-decomposition that the Strata thesis depends on (PLP yield comes
from house-edge + inventory-premium; the f-Curve hump-shape is partly
driven by util² convexity).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.spread import (
    ASK_HI_DEFAULT,
    ASK_LO_DEFAULT,
    BASE_SPREAD_DEFAULT,
    MAX_EXPOSURE_DEFAULT,
    MIN_SPREAD_DEFAULT,
    UTIL_MULT_DEFAULT,
    ask_price,
    spread_per_contract,
)


class TestHouseEdge:
    """Term-1: max(base · √(p(1-p)), min)."""

    def test_at_p_half_zero_util(self):
        """p=0.5 maxes √(p(1-p)) at 0.5 → house edge = base · 0.5 = 1%."""
        s = float(spread_per_contract(p=0.5, util=0.0))
        assert s == pytest.approx(BASE_SPREAD_DEFAULT * 0.5)
        assert s == pytest.approx(0.01)

    def test_floor_binds_at_deep_otm(self):
        """At p=0.01, √(p(1-p))≈0.0995 → base·0.0995≈0.00199 < min=0.005 → floor binds."""
        s = float(spread_per_contract(p=0.01, util=0.0))
        assert s == pytest.approx(MIN_SPREAD_DEFAULT)

    def test_floor_binds_at_deep_itm(self):
        """Symmetric to deep-OTM (p=0.99)."""
        s = float(spread_per_contract(p=0.99, util=0.0))
        assert s == pytest.approx(MIN_SPREAD_DEFAULT)

    def test_symmetry_around_half(self):
        """House edge is symmetric in p around p=0.5."""
        s_lo = float(spread_per_contract(p=0.2, util=0.0))
        s_hi = float(spread_per_contract(p=0.8, util=0.0))
        assert s_lo == pytest.approx(s_hi)


class TestInventoryPremium:
    """Term-2: base · util_mult · util² (convex in util)."""

    def test_zero_at_zero_util(self):
        """At util=0, premium = 0; total = house_edge only."""
        s = float(spread_per_contract(p=0.5, util=0.0))
        assert s == pytest.approx(0.01)  # = house_edge only

    def test_at_max_exposure(self):
        """At util=0.80, premium = 0.02 · 2 · 0.64 = 0.0256 (2.56%)."""
        expected_premium = BASE_SPREAD_DEFAULT * UTIL_MULT_DEFAULT * (
            MAX_EXPOSURE_DEFAULT ** 2
        )
        assert expected_premium == pytest.approx(0.0256)
        # Total at p=0.5, util=0.80:
        s = float(spread_per_contract(p=0.5, util=MAX_EXPOSURE_DEFAULT))
        assert s == pytest.approx(0.01 + 0.0256)

    def test_monotonic_in_util(self):
        """Premium grows strictly with util at fixed p."""
        u = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        s = spread_per_contract(p=0.5, util=u)
        # Strictly increasing
        assert np.all(np.diff(s) > 0)

    def test_convexity_in_util(self):
        """Premium is convex in util (util² term). Discrete check: 2nd diff > 0."""
        u = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        s = spread_per_contract(p=0.5, util=u)
        second_diff = np.diff(s, n=2)
        assert np.all(second_diff > 0)


class TestVectorization:
    """Vectorized over (p, util) ndarrays."""

    def test_array_p_scalar_util(self):
        p = np.array([0.1, 0.5, 0.9])
        s = spread_per_contract(p=p, util=0.0)
        assert s.shape == (3,)
        # p=0.1 and p=0.9 → same (symmetric)
        assert s[0] == pytest.approx(s[2])

    def test_scalar_p_array_util(self):
        u = np.array([0.0, 0.4, 0.8])
        s = spread_per_contract(p=0.5, util=u)
        assert s.shape == (3,)
        assert s[0] < s[1] < s[2]

    def test_broadcasting_p_and_util(self):
        """p column × util row broadcasting."""
        p = np.array([[0.3], [0.5], [0.7]])
        u = np.array([[0.0, 0.4, 0.8]])
        s = spread_per_contract(p=p, util=u)
        assert s.shape == (3, 3)


class TestAskPrice:
    """ask = clip(mid + spread, ask_lo, ask_hi)."""

    def test_unclipped_normal_regime(self):
        """At p=0.5, util=0: ask = 0.5 + 0.01 = 0.51."""
        a = float(ask_price(mid=0.5, util=0.0))
        assert a == pytest.approx(0.51)

    def test_clip_lo_deep_otm(self):
        """Mid=0.001 (deep OTM) → ask = 0.001 + 0.005 ≈ 0.006 < 0.01 → floor."""
        a = float(ask_price(mid=0.001, util=0.0))
        assert a == ASK_LO_DEFAULT

    def test_clip_hi_deep_itm(self):
        """Mid=0.99 + spread > 0.99 → cap to 0.99."""
        a = float(ask_price(mid=0.99, util=0.5))
        assert a == ASK_HI_DEFAULT


class TestNumericPinning:
    """Pin specific numeric values that downstream f-Curve depends on."""

    def test_atm_zero_util(self):
        """ATM, no inventory: 1.0% per contract = pool's pure house edge."""
        s = float(spread_per_contract(p=0.5, util=0.0))
        assert s == pytest.approx(0.01)

    def test_atm_half_util(self):
        """ATM, 50% util: 1.0% + 0.02·2·0.25 = 1.0% + 1.0% = 2.0%."""
        s = float(spread_per_contract(p=0.5, util=0.5))
        assert s == pytest.approx(0.02)

    def test_otm_max_util(self):
        """Deep OTM at max_exposure: 0.5% (floor) + 2.56% = 3.06%."""
        s = float(spread_per_contract(p=0.02, util=0.80))
        assert s == pytest.approx(MIN_SPREAD_DEFAULT + 0.0256)


class TestEdgeCases:
    """Boundary conditions."""

    def test_p_zero(self):
        """p=0 → √(0·1)=0 → floor binds."""
        s = float(spread_per_contract(p=0.0, util=0.0))
        assert s == pytest.approx(MIN_SPREAD_DEFAULT)

    def test_p_one(self):
        """p=1 → √(1·0)=0 → floor binds."""
        s = float(spread_per_contract(p=1.0, util=0.0))
        assert s == pytest.approx(MIN_SPREAD_DEFAULT)

    def test_util_zero_p_zero(self):
        """Bottom-left corner: house_edge floor only."""
        s = float(spread_per_contract(p=0.0, util=0.0))
        assert s == pytest.approx(MIN_SPREAD_DEFAULT)

    def test_negative_spread_impossible(self):
        """Spread is non-negative everywhere."""
        p = np.linspace(0.001, 0.999, 50)
        u = np.linspace(0.0, MAX_EXPOSURE_DEFAULT, 50)
        s = spread_per_contract(p=p[:, None], util=u[None, :])
        assert np.all(s >= 0)
        assert np.all(s >= MIN_SPREAD_DEFAULT)
