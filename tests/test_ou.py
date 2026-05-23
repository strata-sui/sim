"""Unit tests for ``sim.engine.ou`` — Ornstein–Uhlenbeck primitive (S3.1)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.ou import (
    OUParams,
    expected_path_mean,
    expected_path_variance,
    simulate_ou,
)


class TestOUParamsValidation:
    def test_theta_zero_rejected(self):
        with pytest.raises(ValueError, match="theta"):
            OUParams(theta=0.0, mu=0.0, sigma=1.0)

    def test_theta_negative_rejected(self):
        with pytest.raises(ValueError, match="theta"):
            OUParams(theta=-0.1, mu=0.0, sigma=1.0)

    def test_sigma_nonpositive_rejected(self):
        with pytest.raises(ValueError, match="sigma"):
            OUParams(theta=0.5, mu=0.0, sigma=0.0)
        with pytest.raises(ValueError, match="sigma"):
            OUParams(theta=0.5, mu=0.0, sigma=-1.0)

    def test_long_run_variance_formula(self):
        p = OUParams(theta=0.5, mu=0.0, sigma=2.0)
        # σ²/(2θ) = 4/1 = 4
        assert p.long_run_variance == pytest.approx(4.0)
        assert p.long_run_std == pytest.approx(2.0)


class TestSimulateShape:
    def test_output_shape(self):
        p = OUParams(theta=1.0, mu=0.0, sigma=1.0)
        x = simulate_ou(p, n_paths=10, n_steps=20)
        assert x.shape == (10, 21)

    def test_init_value_default_is_mu(self):
        p = OUParams(theta=0.5, mu=1.5, sigma=0.3)
        x = simulate_ou(p, n_paths=5, n_steps=10)
        np.testing.assert_allclose(x[:, 0], 1.5)

    def test_init_value_scalar(self):
        p = OUParams(theta=0.5, mu=0.0, sigma=0.3)
        x = simulate_ou(p, n_paths=5, n_steps=10, x0=7.0)
        np.testing.assert_allclose(x[:, 0], 7.0)

    def test_init_value_array(self):
        p = OUParams(theta=0.5, mu=0.0, sigma=0.3)
        x0 = np.array([1.0, 2.0, 3.0])
        x = simulate_ou(p, n_paths=3, n_steps=5, x0=x0)
        np.testing.assert_allclose(x[:, 0], x0)

    def test_init_value_wrong_shape_rejected(self):
        p = OUParams(theta=0.5, mu=0.0, sigma=0.3)
        with pytest.raises(ValueError, match="x0 shape"):
            simulate_ou(p, n_paths=3, n_steps=5, x0=np.zeros(7))

    def test_bad_innovations_shape_rejected(self):
        p = OUParams(theta=0.5, mu=0.0, sigma=0.3)
        with pytest.raises(ValueError, match="innovations shape"):
            simulate_ou(p, n_paths=3, n_steps=5, innovations=np.zeros((3, 10)))

    def test_negative_dt_rejected(self):
        p = OUParams(theta=0.5, mu=0.0, sigma=0.3)
        with pytest.raises(ValueError, match="dt"):
            simulate_ou(p, n_paths=3, n_steps=5, dt=-1.0)


class TestReproducibility:
    def test_same_seed_identical(self):
        p = OUParams(theta=0.5, mu=0.0, sigma=1.0)
        x1 = simulate_ou(p, n_paths=20, n_steps=50, seed=42)
        x2 = simulate_ou(p, n_paths=20, n_steps=50, seed=42)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seed_diverges(self):
        p = OUParams(theta=0.5, mu=0.0, sigma=1.0)
        x1 = simulate_ou(p, n_paths=20, n_steps=50, seed=1)
        x2 = simulate_ou(p, n_paths=20, n_steps=50, seed=2)
        assert not np.allclose(x1, x2)

    def test_external_innovations_override_seed(self):
        """Passing innovations bypasses the seed-RNG path."""
        p = OUParams(theta=0.5, mu=0.0, sigma=1.0)
        innov = np.ones((5, 20))  # deterministic
        x = simulate_ou(p, n_paths=5, n_steps=20, innovations=innov, seed=0)
        # All paths identical given identical innovations + default x0=μ=0.
        for i in range(5):
            np.testing.assert_allclose(x[i], x[0])


class TestStatisticalProperties:
    """Compare empirical to closed-form OU moments."""

    def test_deterministic_path_matches_closed_form(self):
        """With σ=0 the path is purely deterministic mean reversion.

        Use σ slightly > 0 (validation requires >0) + innovations=zeros to
        suppress noise — same effect."""
        p = OUParams(theta=0.5, mu=2.0, sigma=1.0)
        innov = np.zeros((1, 50))
        x = simulate_ou(
            p, n_paths=1, n_steps=50, dt=0.1, x0=0.0, innovations=innov
        )
        t = np.arange(51) * 0.1
        expected = expected_path_mean(p, x0=0.0, t_grid=t)
        # Euler-Maruyama has O(dt) bias for the drift; with dt=0.1 over 50 steps
        # the bias is small but non-zero. Tolerance ~5% relative on the path.
        np.testing.assert_allclose(x[0], expected, rtol=0.05, atol=1e-3)

    def test_long_run_mean(self):
        p = OUParams(theta=1.0, mu=3.0, sigma=0.5)
        x = simulate_ou(p, n_paths=2000, n_steps=500, dt=0.05, seed=7)
        # Burn in: drop first 200 steps then take steady-state samples.
        steady = x[:, 200:].ravel()
        np.testing.assert_allclose(steady.mean(), 3.0, atol=0.02)

    def test_long_run_variance(self):
        p = OUParams(theta=1.0, mu=0.0, sigma=1.0)
        # σ²/(2θ) = 0.5 → std ≈ 0.707
        x = simulate_ou(p, n_paths=2000, n_steps=1000, dt=0.05, seed=8)
        steady = x[:, 200:].ravel()
        np.testing.assert_allclose(steady.var(ddof=1), 0.5, rtol=0.10)

    def test_acf_matches_theory(self):
        """ACF(τ) of stationary OU = exp(−θτ)."""
        theta = 1.0
        p = OUParams(theta=theta, mu=0.0, sigma=1.0)
        x = simulate_ou(p, n_paths=200, n_steps=2000, dt=0.05, seed=9)
        # Stationary tail.
        x_st = x[:, 500:]
        for lag in (5, 10, 20):
            tau = lag * 0.05
            # Pool over paths.
            a = (x_st[:, :-lag] - x_st[:, :-lag].mean()).ravel()
            b = (x_st[:, lag:] - x_st[:, lag:].mean()).ravel()
            empirical = float(np.dot(a, b) / (np.std(a) * np.std(b) * len(a)))
            theoretical = float(np.exp(-theta * tau))
            assert abs(empirical - theoretical) < 0.05, (
                f"lag {lag}: emp={empirical:.3f} theo={theoretical:.3f}"
            )

    def test_mean_reverts_from_far_initial(self):
        """E[x_t | x_0 = 5] → μ as t grows. With μ=0 and θ=1, by t=5 mean ≪ 5."""
        p = OUParams(theta=1.0, mu=0.0, sigma=0.1)
        x = simulate_ou(p, n_paths=2000, n_steps=200, dt=0.05, x0=5.0, seed=10)
        # At t ≈ 5 (step 100): E ≈ 5·e^{−5} ≈ 0.034
        empirical_mean_at_t5 = float(x[:, 100].mean())
        assert abs(empirical_mean_at_t5) < 0.2


class TestClosedFormHelpers:
    def test_expected_path_mean_decay(self):
        p = OUParams(theta=2.0, mu=0.0, sigma=1.0)
        m = expected_path_mean(p, x0=10.0, t_grid=np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(m, [10.0, 10.0 * np.exp(-1.0), 10.0 * np.exp(-2.0)])

    def test_expected_path_variance_grows_to_long_run(self):
        p = OUParams(theta=1.0, mu=0.0, sigma=2.0)
        # σ²/(2θ) = 2
        v = expected_path_variance(p, t_grid=np.array([0.0, 1.0, 100.0]))
        assert v[0] == pytest.approx(0.0)
        assert v[1] < v[2]
        assert v[2] == pytest.approx(p.long_run_variance, rel=1e-3)
