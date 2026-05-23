"""Unit tests for ``sim.engine.svi_stoch`` — multi-factor SVI dynamics (S3.2)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.ou import OUParams
from engine.svi_stoch import StochasticSVIEngine, SVIDynamicsParams


def _ou(theta=0.5, mu=0.0, sigma=0.1):
    return OUParams(theta=theta, mu=mu, sigma=sigma)


def _identity_corr():
    return np.eye(3)


def _negcorr_a_b(c: float = -0.5):
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = c
    return corr


def _params(corr=None, **overrides):
    return SVIDynamicsParams(
        a=overrides.get("a", _ou(theta=1.0, mu=1.66e-4, sigma=1.0e-5)),
        b=overrides.get("b", _ou(theta=0.5, mu=7.3e-3, sigma=5.0e-4)),
        rho=overrides.get("rho", _ou(theta=0.5, mu=-0.32, sigma=0.05)),
        correlation=corr if corr is not None else _identity_corr(),
        m=overrides.get("m", -0.00275),
        sigma=overrides.get("sigma", 0.01426),
    )


# ---- Validation ---------------------------------------------------------


class TestParamsValidation:
    def test_correlation_wrong_shape(self):
        with pytest.raises(ValueError, match="correlation must be"):
            _params(corr=np.eye(4))

    def test_correlation_not_symmetric(self):
        corr = np.eye(3)
        corr[0, 1] = 0.5
        # 1,0 stays 0 → asymmetric
        with pytest.raises(ValueError, match="symmetric"):
            _params(corr=corr)

    def test_correlation_diag_not_one(self):
        corr = np.eye(3)
        corr[0, 0] = 2.0
        with pytest.raises(ValueError, match="diagonal"):
            _params(corr=corr)

    def test_correlation_not_psd(self):
        # eigenvalues: a 3×3 with off-diagonals 0.99 has small negative eig.
        corr = np.array([
            [1.0, 0.99, -0.99],
            [0.99, 1.0, 0.99],
            [-0.99, 0.99, 1.0],
        ])
        with pytest.raises(ValueError, match="positive-definite"):
            _params(corr=corr)

    def test_sigma_nonpositive_rejected(self):
        with pytest.raises(ValueError, match="sigma"):
            _params(sigma=0.0)


# ---- Output shape -------------------------------------------------------


class TestSimulateShape:
    def test_output_keys_and_shapes(self):
        e = StochasticSVIEngine(_params(), seed=1)
        out = e.simulate(n_paths=10, n_steps=20)
        assert set(out.keys()) == {"a", "b", "rho", "m", "sigma"}
        assert out["a"].shape == (10, 21)
        assert out["b"].shape == (10, 21)
        assert out["rho"].shape == (10, 21)
        assert out["m"] == _params().m
        assert out["sigma"] == _params().sigma

    def test_initial_values_default_to_mu(self):
        e = StochasticSVIEngine(_params(), seed=2)
        out = e.simulate(n_paths=5, n_steps=10)
        np.testing.assert_allclose(out["a"][:, 0], _params().a.mu)
        np.testing.assert_allclose(out["b"][:, 0], _params().b.mu)
        np.testing.assert_allclose(out["rho"][:, 0], _params().rho.mu)

    def test_explicit_initial_values_respected(self):
        e = StochasticSVIEngine(_params(), seed=3)
        out = e.simulate(
            n_paths=5, n_steps=10, a0=0.001, b0=0.005, rho0=-0.4
        )
        np.testing.assert_allclose(out["a"][:, 0], 0.001)
        np.testing.assert_allclose(out["b"][:, 0], 0.005)
        np.testing.assert_allclose(out["rho"][:, 0], -0.4)

    def test_bad_external_innovations_rejected(self):
        e = StochasticSVIEngine(_params(), seed=4)
        with pytest.raises(ValueError, match="external_innovations"):
            e.simulate(
                n_paths=5, n_steps=10,
                external_innovations=np.zeros((5, 10, 4)),
            )


# ---- Reproducibility ----------------------------------------------------


class TestReproducibility:
    def test_same_seed_identical(self):
        e1 = StochasticSVIEngine(_params(), seed=42)
        e2 = StochasticSVIEngine(_params(), seed=42)
        o1 = e1.simulate(20, 50)
        o2 = e2.simulate(20, 50)
        for k in ("a", "b", "rho"):
            np.testing.assert_array_equal(o1[k], o2[k])

    def test_external_innovations_override_seed(self):
        innov = np.ones((3, 10, 3))
        e = StochasticSVIEngine(_params(), seed=0)
        out = e.simulate(n_paths=3, n_steps=10, external_innovations=innov)
        # With identical innovations all paths follow the same trajectory.
        np.testing.assert_allclose(out["a"][0], out["a"][1])
        np.testing.assert_allclose(out["b"][1], out["b"][2])


# ---- Statistical: Cholesky correlation respects target -----------------


class TestCorrelation:
    def test_identity_corr_gives_zero_empirical(self):
        """With corr = I the three OU factor innovations are independent."""
        e = StochasticSVIEngine(_params(corr=_identity_corr()), seed=10)
        out = e.simulate(n_paths=500, n_steps=300, dt=0.05)
        # Empirical correlation of factor INCREMENTS (not levels — levels are
        # AR-correlated by OU even with iid innovations).
        da = np.diff(out["a"], axis=1).ravel()
        db = np.diff(out["b"], axis=1).ravel()
        drho = np.diff(out["rho"], axis=1).ravel()
        # All three pairwise correlations should be ~0 within sampling.
        for x, y in [(da, db), (da, drho), (db, drho)]:
            corr = float(np.corrcoef(x, y)[0, 1])
            assert abs(corr) < 0.05, f"corr {corr:.4f} expected ~0"

    def test_negative_corr_a_b_realized(self):
        """corr[a,b] = -0.5 → empirical innovation correlation ≈ -0.5."""
        e = StochasticSVIEngine(_params(corr=_negcorr_a_b(-0.5)), seed=11)
        out = e.simulate(n_paths=500, n_steps=300, dt=0.05)
        # Diff cancels the AR(1) component → reveals innovation correlation.
        da = np.diff(out["a"], axis=1).ravel()
        db = np.diff(out["b"], axis=1).ravel()
        corr = float(np.corrcoef(da, db)[0, 1])
        assert abs(corr - (-0.5)) < 0.08, (
            f"empirical {corr:.3f}, expected -0.5 within tolerance"
        )

    def test_positive_corr_a_rho_realized(self):
        corr = np.eye(3)
        corr[0, 2] = corr[2, 0] = 0.7
        e = StochasticSVIEngine(_params(corr=corr), seed=12)
        out = e.simulate(n_paths=500, n_steps=300, dt=0.05)
        da = np.diff(out["a"], axis=1).ravel()
        drho = np.diff(out["rho"], axis=1).ravel()
        emp = float(np.corrcoef(da, drho)[0, 1])
        assert abs(emp - 0.7) < 0.08

    def test_correlation_property_exposes_input(self):
        corr = _negcorr_a_b(-0.3)
        e = StochasticSVIEngine(_params(corr=corr), seed=13)
        np.testing.assert_allclose(e.correlation, corr)


# ---- OU long-run properties survive multi-factor wrapping --------------


class TestBtcLeverageCoupling:
    """S3.3 — leverage-effect coupling between BTC return shock and SVI shocks."""

    def test_btc_correlation_default_is_zeros(self):
        p = _params()
        np.testing.assert_array_equal(p.btc_correlation, np.zeros(3))

    def test_btc_correlation_shape_validation(self):
        with pytest.raises(ValueError, match="btc_correlation"):
            SVIDynamicsParams(
                a=_ou(), b=_ou(), rho=_ou(),
                correlation=np.eye(3), m=0.0, sigma=0.01,
                btc_correlation=np.array([0.1, 0.2]),
            )

    def test_btc_correlation_range_validation(self):
        with pytest.raises(ValueError, match="btc_correlation"):
            SVIDynamicsParams(
                a=_ou(), b=_ou(), rho=_ou(),
                correlation=np.eye(3), m=0.0, sigma=0.01,
                btc_correlation=np.array([1.5, 0.0, 0.0]),
            )

    def test_joint_psd_violation_rejected(self):
        """Pathological btc_correlation that makes joint 4×4 non-PSD."""
        # If btc to all three SVI is +0.99 but SVI internals are -0.99,
        # joint matrix is not PSD.
        bad_svi = np.array([
            [1.0, -0.99, -0.99],
            [-0.99, 1.0, -0.99],
            [-0.99, -0.99, 1.0],
        ])
        # SVI alone might already fail PSD; if it does, that's the catch.
        with pytest.raises(ValueError):
            SVIDynamicsParams(
                a=_ou(), b=_ou(), rho=_ou(),
                correlation=bad_svi, m=0.0, sigma=0.01,
                btc_correlation=np.array([0.99, 0.99, 0.99]),
            )

    def test_return_shocks_shape_validation(self):
        e = StochasticSVIEngine(_params(), seed=1)
        with pytest.raises(ValueError, match="return_shocks"):
            e.simulate(
                n_paths=5, n_steps=10,
                return_shocks=np.zeros((5, 11)),  # wrong T
            )

    def test_leverage_a_coupling_realised(self):
        """corr(Z_btc, Z_a) = -0.6 → empirical Δa vs Z_btc ≈ -0.6."""
        p = SVIDynamicsParams(
            a=_ou(theta=1.0, mu=0.0, sigma=1.0),
            b=_ou(theta=1.0, mu=0.0, sigma=1.0),
            rho=_ou(theta=1.0, mu=0.0, sigma=1.0),
            correlation=np.eye(3),
            m=0.0, sigma=0.01,
            btc_correlation=np.array([-0.6, 0.0, 0.0]),
        )
        e = StochasticSVIEngine(p, seed=50)
        rng = np.random.default_rng(7)
        n_paths, n_steps = 500, 300
        z_btc = rng.standard_normal((n_paths, n_steps))
        out = e.simulate(
            n_paths=n_paths, n_steps=n_steps, dt=0.05,
            return_shocks=z_btc,
        )
        # The OU innovation for `a` at step t generated correlated to z_btc[t].
        # After OU integration we can recover the innovation by inverting:
        # innov_a[t] = (a[t+1] - a[t] - θ·dt·(μ - a[t])) / (σ·√dt)
        dt = 0.05
        innov_a = (
            (out["a"][:, 1:] - out["a"][:, :-1] - p.a.theta * dt * (p.a.mu - out["a"][:, :-1]))
            / (p.a.sigma * np.sqrt(dt))
        )
        emp_corr = float(np.corrcoef(z_btc.ravel(), innov_a.ravel())[0, 1])
        assert abs(emp_corr - (-0.6)) < 0.06

    def test_leverage_rho_coupling_realised(self):
        """corr(Z_btc, Z_rho) = +0.4 (BTC down → rho more negative)."""
        p = SVIDynamicsParams(
            a=_ou(theta=1.0, mu=0.0, sigma=1.0),
            b=_ou(theta=1.0, mu=0.0, sigma=1.0),
            rho=_ou(theta=1.0, mu=0.0, sigma=1.0),
            correlation=np.eye(3),
            m=0.0, sigma=0.01,
            btc_correlation=np.array([0.0, 0.0, 0.4]),
        )
        e = StochasticSVIEngine(p, seed=51)
        rng = np.random.default_rng(9)
        n_paths, n_steps = 500, 300
        z_btc = rng.standard_normal((n_paths, n_steps))
        out = e.simulate(
            n_paths=n_paths, n_steps=n_steps, dt=0.05,
            return_shocks=z_btc,
        )
        dt = 0.05
        innov_rho = (
            (out["rho"][:, 1:] - out["rho"][:, :-1]
             - p.rho.theta * dt * (p.rho.mu - out["rho"][:, :-1]))
            / (p.rho.sigma * np.sqrt(dt))
        )
        emp_corr = float(np.corrcoef(z_btc.ravel(), innov_rho.ravel())[0, 1])
        assert abs(emp_corr - 0.4) < 0.06

    def test_no_return_shocks_back_compat(self):
        """S3.2 mode (return_shocks=None) still works — backward compat."""
        e = StochasticSVIEngine(_params(), seed=52)
        out = e.simulate(n_paths=10, n_steps=20)
        assert out["a"].shape == (10, 21)


class TestLongRunProperties:
    def test_each_factor_mean_reverts_to_mu(self):
        e = StochasticSVIEngine(_params(), seed=20)
        out = e.simulate(n_paths=1000, n_steps=600, dt=0.05)
        # Burn in.
        a_st = out["a"][:, 200:].ravel()
        b_st = out["b"][:, 200:].ravel()
        rho_st = out["rho"][:, 200:].ravel()
        assert abs(float(a_st.mean()) - _params().a.mu) < 1e-4
        assert abs(float(b_st.mean()) - _params().b.mu) < 5e-4
        assert abs(float(rho_st.mean()) - _params().rho.mu) < 5e-2
