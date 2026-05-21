"""Unit tests for ``sim.model.trader_flow`` — 6-knob exogenous model.

Pinned invariants:
  * Anchor numbers match ``specs/trader_flow_spec.md §5`` table verbatim.
  * Strike-local u(k) boundary cases: u(0,f) = 1, u(1,f) = f.
  * Volume scales correctly with λ₀, β_vol, vol-excess, balance.
  * DN fraction respects φ_informed (noise vs informed mixture).
  * LP flight is monotonic in drawdown and capped by θ_limiter.
  * Vectorization works for batched MC use.
  * Parameter validation rejects out-of-range knobs.
"""
from __future__ import annotations

import numpy as np
import pytest

from model.trader_flow import (
    CONSERVATIVE_ANCHOR,
    MAX_EXPOSURE,
    OPTIMISTIC_ANCHOR,
    PESSIMAL_ANCHOR,
    TraderFlowAnchor,
    lp_flight_step,
    trader_flow_step,
    u_strike_local,
)


# ---- Anchors -------------------------------------------------------------


class TestAnchors:
    """Anchor values must match spec §5 verbatim — the Gate baseline."""

    def test_conservative_values(self):
        a = CONSERVATIVE_ANCHOR
        assert a.lambda_0 == 0.15
        assert a.beta_vol == 3.0
        assert a.phi_informed == 0.30
        assert a.kappa_strike == 0.10
        assert a.gamma_flight == 3.0
        assert a.theta_limiter == 0.10

    def test_pessimal_values(self):
        a = PESSIMAL_ANCHOR
        assert a.lambda_0 == 0.005
        assert a.beta_vol == 0.0
        assert a.phi_informed == 1.0
        assert a.kappa_strike == 0.30
        assert a.gamma_flight == 10.0
        assert a.theta_limiter == 1.0

    def test_optimistic_values(self):
        a = OPTIMISTIC_ANCHOR
        assert a.lambda_0 == 0.50
        assert a.beta_vol == 10.0
        assert a.phi_informed == 0.10
        assert a.kappa_strike == 0.05
        assert a.gamma_flight == 0.5
        assert a.theta_limiter == 0.02

    def test_conservative_between_pessimal_and_optimistic(self):
        """Conservative should sit between Pessimal and Optimistic per knob.

        Direction depends on the knob (some grow toward optimism, others
        shrink). The test pins the monotonic ordering implicit in spec §5.
        """
        # λ₀ grows toward optimism (more trader volume = better hedge funding):
        assert PESSIMAL_ANCHOR.lambda_0 < CONSERVATIVE_ANCHOR.lambda_0 < OPTIMISTIC_ANCHOR.lambda_0
        # β_vol grows toward optimism:
        assert PESSIMAL_ANCHOR.beta_vol <= CONSERVATIVE_ANCHOR.beta_vol <= OPTIMISTIC_ANCHOR.beta_vol
        # φ_informed SHRINKS toward optimism (less crash-chasing = better for Strata):
        assert PESSIMAL_ANCHOR.phi_informed > CONSERVATIVE_ANCHOR.phi_informed > OPTIMISTIC_ANCHOR.phi_informed
        # κ_strike SHRINKS toward optimism:
        assert PESSIMAL_ANCHOR.kappa_strike > CONSERVATIVE_ANCHOR.kappa_strike > OPTIMISTIC_ANCHOR.kappa_strike
        # γ_flight SHRINKS:
        assert PESSIMAL_ANCHOR.gamma_flight > CONSERVATIVE_ANCHOR.gamma_flight > OPTIMISTIC_ANCHOR.gamma_flight
        # θ_limiter SHRINKS (lower = more throttle = better):
        assert PESSIMAL_ANCHOR.theta_limiter > CONSERVATIVE_ANCHOR.theta_limiter > OPTIMISTIC_ANCHOR.theta_limiter


class TestAnchorValidation:
    def test_negative_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_0"):
            TraderFlowAnchor(-0.1, 3.0, 0.3, 0.1, 3.0, 0.1)

    def test_lambda_above_one_rejected(self):
        with pytest.raises(ValueError, match="lambda_0"):
            TraderFlowAnchor(1.5, 3.0, 0.3, 0.1, 3.0, 0.1)

    def test_negative_beta_vol_rejected(self):
        with pytest.raises(ValueError, match="beta_vol"):
            TraderFlowAnchor(0.15, -1.0, 0.3, 0.1, 3.0, 0.1)

    def test_phi_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="phi_informed"):
            TraderFlowAnchor(0.15, 3.0, 1.5, 0.1, 3.0, 0.1)

    def test_kappa_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="kappa_strike"):
            TraderFlowAnchor(0.15, 3.0, 0.3, 1.5, 3.0, 0.1)

    def test_theta_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="theta_limiter"):
            TraderFlowAnchor(0.15, 3.0, 0.3, 0.1, 3.0, 1.5)


# ---- Strike-local u(k) ---------------------------------------------------


class TestUStrikeLocal:
    """u = 1 - kappa·(1-f): genuine-hedge condition u(k) > f."""

    def test_kappa_zero_means_strata_alone(self):
        """At κ=0, Umbra is the sole DN holder → u = 1."""
        for f in [0.05, 0.20, 0.50, 0.80, 0.95]:
            assert float(u_strike_local(0.0, f)) == pytest.approx(1.0)

    def test_kappa_one_means_pure_wash(self):
        """At κ=1, u = f → expected hedge = 0 per spec §1.2."""
        for f in [0.05, 0.20, 0.50, 0.80, 0.95]:
            assert float(u_strike_local(1.0, f)) == pytest.approx(f)

    def test_linear_interp(self):
        """u(0.5, 0.5) = 1 - 0.5*0.5 = 0.75."""
        u = float(u_strike_local(0.5, 0.5))
        assert u == pytest.approx(0.75)

    def test_always_above_or_equal_f(self):
        """u >= f always (since 1-κ(1-f) >= f ⟺ 1-f >= κ(1-f) ⟺ κ <= 1)."""
        kappa = np.linspace(0, 1, 50)
        f = np.linspace(0.05, 0.95, 50)
        u_grid = u_strike_local(kappa[:, None], f[None, :])
        assert np.all(u_grid >= f[None, :] - 1e-12)

    def test_vectorized_broadcast(self):
        kappa = np.array([0.0, 0.3, 1.0])
        f = 0.5
        u = u_strike_local(kappa, f)
        assert u.shape == (3,)
        np.testing.assert_allclose(u, [1.0, 0.85, 0.5])


# ---- Trader-flow step ----------------------------------------------------


class TestTraderFlowStep:
    def test_baseline_no_vol_excess(self):
        """At realized_vol == sigma_long_run: total_notional = λ₀·max_exposure·balance."""
        out = trader_flow_step(
            pool_balance=1_000_000.0,
            realized_vol=0.01,
            sigma_long_run=0.01,
            log_return_recent=0.0,
            anchor=CONSERVATIVE_ANCHOR,
        )
        expected = 0.15 * MAX_EXPOSURE * 1_000_000.0
        assert float(out["total_notional"]) == pytest.approx(expected)
        assert float(out["lambda_eff"]) == pytest.approx(0.15)

    def test_vol_excess_amplifies_volume(self):
        """At realized_vol = 2·σ̄: λ_eff = λ₀·(1 + β·1)."""
        out = trader_flow_step(
            pool_balance=1_000_000.0,
            realized_vol=0.02,
            sigma_long_run=0.01,
            log_return_recent=0.0,
            anchor=CONSERVATIVE_ANCHOR,
        )
        expected_lambda = 0.15 * (1.0 + 3.0 * 1.0)  # = 0.60
        assert float(out["lambda_eff"]) == pytest.approx(expected_lambda)

    def test_no_vol_excess_when_below_mean(self):
        """realized_vol < σ̄: vol_excess clamped to 0 → λ_eff = λ₀."""
        out = trader_flow_step(
            pool_balance=1_000_000.0,
            realized_vol=0.005,
            sigma_long_run=0.01,
            log_return_recent=0.0,
            anchor=CONSERVATIVE_ANCHOR,
        )
        assert float(out["lambda_eff"]) == pytest.approx(0.15)

    def test_pure_noise_flow_is_balanced(self):
        """φ_informed=0 → DN fraction = 0.5 regardless of recent return."""
        a = TraderFlowAnchor(0.15, 3.0, 0.0, 0.1, 3.0, 0.1)  # phi=0
        out = trader_flow_step(
            pool_balance=1_000_000.0,
            realized_vol=0.01,
            sigma_long_run=0.01,
            log_return_recent=-0.05,  # big crash
            anchor=a,
        )
        assert float(out["dn_fraction"]) == pytest.approx(0.5)

    def test_informed_crash_chase(self):
        """φ_informed=1 + big negative recent return → DN fraction > 0.5."""
        a = TraderFlowAnchor(0.15, 3.0, 1.0, 0.1, 3.0, 0.1)  # phi=1
        out = trader_flow_step(
            pool_balance=1_000_000.0,
            realized_vol=0.01,
            sigma_long_run=0.01,
            log_return_recent=-0.05,
            anchor=a,
        )
        # -log_return/sigma = 0.05/0.01 = 5 → clipped to 1 → b_informed=1
        # dn_frac = 0.5*(1+1) = 1.0
        assert float(out["dn_fraction"]) == pytest.approx(1.0)

    def test_informed_pump_chase(self):
        """φ_informed=1 + big positive recent return → DN fraction < 0.5."""
        a = TraderFlowAnchor(0.15, 3.0, 1.0, 0.1, 3.0, 0.1)
        out = trader_flow_step(
            pool_balance=1_000_000.0,
            realized_vol=0.01,
            sigma_long_run=0.01,
            log_return_recent=+0.05,
            anchor=a,
        )
        # b_informed clipped to -1 → dn_frac = 0
        assert float(out["dn_fraction"]) == pytest.approx(0.0)

    def test_notional_conservation(self):
        """dn_notional + up_notional == total_notional always."""
        out = trader_flow_step(
            pool_balance=1_000_000.0,
            realized_vol=0.02,
            sigma_long_run=0.01,
            log_return_recent=-0.01,
            anchor=CONSERVATIVE_ANCHOR,
        )
        total = float(out["total_notional"])
        s = float(out["dn_notional"]) + float(out["up_notional"])
        assert s == pytest.approx(total)

    def test_vectorized_over_paths(self):
        """Batched balance/realized_vol/log_return across paths."""
        bal = np.array([1e5, 1e6, 1e7])
        rv = np.array([0.005, 0.01, 0.03])
        lrr = np.array([0.0, 0.0, -0.05])
        out = trader_flow_step(
            pool_balance=bal,
            realized_vol=rv,
            sigma_long_run=0.01,
            log_return_recent=lrr,
            anchor=CONSERVATIVE_ANCHOR,
        )
        assert out["total_notional"].shape == (3,)
        # path 2 has both vol spike AND crash → biggest dn_notional
        assert out["dn_notional"][2] > out["dn_notional"][1]
        assert out["dn_notional"][2] > out["dn_notional"][0]

    def test_zero_sigma_long_run_rejected(self):
        with pytest.raises(ValueError, match="sigma_long_run"):
            trader_flow_step(
                pool_balance=1_000_000.0,
                realized_vol=0.01,
                sigma_long_run=0.0,
                log_return_recent=0.0,
                anchor=CONSERVATIVE_ANCHOR,
            )


# ---- LP flight -----------------------------------------------------------


class TestLPFlight:
    def test_zero_drawdown_no_flight(self):
        """drawdown=0 → no flight (L_other unchanged)."""
        l_new = float(lp_flight_step(1_000_000.0, 0.0, CONSERVATIVE_ANCHOR))
        assert l_new == pytest.approx(1_000_000.0)

    def test_negative_drawdown_treated_as_zero(self):
        """Negative drawdown (pool gained value) → no flight."""
        l_new = float(lp_flight_step(1_000_000.0, -0.10, CONSERVATIVE_ANCHOR))
        assert l_new == pytest.approx(1_000_000.0)

    def test_throttled_by_theta(self):
        """Big drawdown × big γ → desired > θ → capped at θ."""
        # γ·DD = 3.0·0.50 = 1.50, clip to 1.0, then min(1.0, θ=0.10) = 0.10
        l_new = float(lp_flight_step(1_000_000.0, 0.50, CONSERVATIVE_ANCHOR))
        # Exit rate = 0.10 → L_new = 900_000
        assert l_new == pytest.approx(900_000.0)

    def test_pessimal_unthrottled(self):
        """Pessimal θ=1.0: full desired flight applied."""
        # γ·DD = 10·0.05 = 0.50; θ=1.0 → actual = 0.50
        l_new = float(lp_flight_step(1_000_000.0, 0.05, PESSIMAL_ANCHOR))
        assert l_new == pytest.approx(500_000.0)

    def test_monotonic_in_drawdown(self):
        """Flight amount weakly increases with drawdown."""
        dd = np.linspace(0.0, 0.30, 50)
        l_new = lp_flight_step(np.full_like(dd, 1_000_000.0), dd, CONSERVATIVE_ANCHOR)
        # Non-increasing L_other = strictly increasing flight.
        assert np.all(np.diff(l_new) <= 0)

    def test_vectorized_paths(self):
        l = np.array([1e5, 1e6, 1e7])
        dd = np.array([0.0, 0.10, 0.30])
        l_new = lp_flight_step(l, dd, CONSERVATIVE_ANCHOR)
        assert l_new.shape == (3,)
        # path 0: no flight
        assert l_new[0] == pytest.approx(l[0])
        # path 2: more flight than path 1
        assert (l[2] - l_new[2]) / l[2] >= (l[1] - l_new[1]) / l[1]
