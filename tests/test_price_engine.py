"""Unit tests for ``sim.engine.price_engine`` — block-bootstrap S1 form.

Pinned invariants:
  * Shape correctness for the three output arrays.
  * Reproducibility under same seed; divergence under different seeds.
  * Kurtosis preservation: bootstrapped paths retain ≥ 80% of historical
    excess kurtosis (CLAUDE.md §6 false-Gate guard #5: bootstrap must NOT
    destroy jump clustering).
  * Volatility clustering preservation at lags < block_length.
  * Price-path consistency: ``price[:, k] = init * exp(cumsum(log_ret))``.
  * Validation of bad inputs.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from engine.price_engine import BootstrapEngine


# ---- Fixtures ------------------------------------------------------------


@pytest.fixture
def synth_returns() -> np.ndarray:
    """High-kurtosis synthetic returns (t-dist with df=5, kurt = 6)."""
    rng = np.random.default_rng(42)
    return rng.standard_t(df=5, size=10_000) * 0.01


@pytest.fixture
def clustered_returns() -> np.ndarray:
    """Returns with engineered volatility clustering (GARCH-ish).

    Volatility regime alternates in blocks of 50 between σ_lo and σ_hi.
    Guarantees ACF(|r|) > 0 at lags < ~50.
    """
    rng = np.random.default_rng(7)
    n = 5000
    sigma = np.tile(np.repeat([0.005, 0.03], 50), n // 100 + 1)[:n]
    return rng.standard_normal(n) * sigma


# ---- Construction --------------------------------------------------------


class TestConstruction:
    def test_default_construction(self, synth_returns):
        e = BootstrapEngine(synth_returns)
        assert e.block_length == 20
        assert e.seed == 0
        assert e.sigma_historical > 0

    def test_two_d_input_rejected(self):
        bad = np.zeros((10, 10))
        with pytest.raises(ValueError, match="1-D"):
            BootstrapEngine(bad)

    def test_block_length_below_one_rejected(self, synth_returns):
        with pytest.raises(ValueError, match="block_length must be"):
            BootstrapEngine(synth_returns, block_length=0)

    def test_short_history_rejected(self):
        with pytest.raises(ValueError, match="block_length"):
            BootstrapEngine(np.zeros(5), block_length=20)

    def test_sigma_historical_matches_input(self, synth_returns):
        e = BootstrapEngine(synth_returns)
        np.testing.assert_allclose(
            e.sigma_historical, np.std(synth_returns, ddof=1)
        )


# ---- Shape + content -----------------------------------------------------


class TestSimulateShape:
    def test_output_shapes(self, synth_returns):
        e = BootstrapEngine(synth_returns, block_length=20, seed=1)
        out = e.simulate(n_paths=50, n_steps=200)
        assert out["log_return"].shape == (50, 200)
        assert out["standardized_shock"].shape == (50, 200)
        assert out["price"].shape == (50, 201)  # +1 for init column

    def test_init_price_in_column_zero(self, synth_returns):
        e = BootstrapEngine(synth_returns, seed=2)
        out = e.simulate(n_paths=10, n_steps=50, init_price=123.45)
        np.testing.assert_allclose(out["price"][:, 0], 123.45)

    def test_price_path_consistent_with_log_returns(self, synth_returns):
        """price[:, k+1] = init * exp(cumsum(log_ret[:, :k+1]))."""
        e = BootstrapEngine(synth_returns, seed=3)
        out = e.simulate(n_paths=5, n_steps=30, init_price=100.0)
        expected = 100.0 * np.exp(np.cumsum(out["log_return"], axis=1))
        np.testing.assert_allclose(out["price"][:, 1:], expected, rtol=1e-12)

    def test_standardized_shock_is_log_return_over_sigma(self, synth_returns):
        e = BootstrapEngine(synth_returns, seed=4)
        out = e.simulate(n_paths=5, n_steps=30)
        np.testing.assert_allclose(
            out["standardized_shock"],
            out["log_return"] / e.sigma_historical,
            rtol=1e-12,
        )

    def test_zero_paths_or_steps_rejected(self, synth_returns):
        e = BootstrapEngine(synth_returns)
        with pytest.raises(ValueError):
            e.simulate(n_paths=0, n_steps=10)
        with pytest.raises(ValueError):
            e.simulate(n_paths=10, n_steps=0)


# ---- Reproducibility -----------------------------------------------------


class TestReproducibility:
    def test_same_seed_identical_paths(self, synth_returns):
        e1 = BootstrapEngine(synth_returns, block_length=10, seed=42)
        e2 = BootstrapEngine(synth_returns, block_length=10, seed=42)
        out1 = e1.simulate(50, 100)
        out2 = e2.simulate(50, 100)
        np.testing.assert_array_equal(out1["log_return"], out2["log_return"])

    def test_different_seed_different_paths(self, synth_returns):
        e1 = BootstrapEngine(synth_returns, block_length=10, seed=1)
        e2 = BootstrapEngine(synth_returns, block_length=10, seed=2)
        out1 = e1.simulate(50, 100)
        out2 = e2.simulate(50, 100)
        assert not np.allclose(out1["log_return"], out2["log_return"])


# ---- Statistical properties (the Gate-A-critical invariants) ------------


class TestStatisticalProperties:
    def test_kurtosis_preserved(self, synth_returns):
        """Excess kurtosis of bootstrapped pool >= 80% of historical.

        Guard #5 in CLAUDE.md §6: a bootstrap that destroys fat tails
        produces false-GREEN Gate A. This test pins that.
        """
        base_kurt = stats.kurtosis(synth_returns)  # excess
        e = BootstrapEngine(synth_returns, block_length=20, seed=11)
        out = e.simulate(n_paths=500, n_steps=2000)
        sampled = out["log_return"].flatten()
        sampled_kurt = stats.kurtosis(sampled)
        assert sampled_kurt >= 0.8 * base_kurt, (
            f"kurtosis lost: base={base_kurt:.2f}, sampled={sampled_kurt:.2f}"
        )

    def test_mean_and_variance_close(self, synth_returns):
        """First two moments of the bootstrapped pool match input."""
        e = BootstrapEngine(synth_returns, block_length=20, seed=12)
        out = e.simulate(n_paths=200, n_steps=5000)
        sampled = out["log_return"].flatten()
        np.testing.assert_allclose(
            sampled.mean(), synth_returns.mean(), atol=0.001
        )
        np.testing.assert_allclose(
            sampled.std(ddof=1), synth_returns.std(ddof=1), rtol=0.05
        )

    def test_clustering_preserved_within_block(self, clustered_returns):
        """ACF(|r|) at lag 5 (< block_length=20) should be > 0 — within-block
        contiguity preserves clustering."""
        e = BootstrapEngine(clustered_returns, block_length=20, seed=13)
        out = e.simulate(n_paths=1, n_steps=4000)
        r = out["log_return"][0]
        ar = np.abs(r) - np.abs(r).mean()
        n = len(ar)
        var = float(np.dot(ar, ar) / n)
        acf5 = float(np.dot(ar[:-5], ar[5:]) / ((n - 5) * var))
        assert acf5 > 0.05

    def test_iid_block_gives_no_clustering(self, clustered_returns):
        """block_length=1 = IID bootstrap — clustering destroyed.

        Confirms the bootstrap mechanism (block-length matters); this
        test is the *negative* version of the previous one.
        """
        e = BootstrapEngine(clustered_returns, block_length=1, seed=14)
        out = e.simulate(n_paths=1, n_steps=4000)
        r = out["log_return"][0]
        ar = np.abs(r) - np.abs(r).mean()
        n = len(ar)
        var = float(np.dot(ar, ar) / n)
        acf5 = float(np.dot(ar[:-5], ar[5:]) / ((n - 5) * var))
        # IID -> ACF noisy around 0; should be small.
        assert abs(acf5) < 0.05


# ---- Vectorization smoke -------------------------------------------------


class TestVectorization:
    def test_large_path_count(self, synth_returns):
        """10k × 500 should finish quickly and have valid shape."""
        e = BootstrapEngine(synth_returns, block_length=20, seed=99)
        out = e.simulate(n_paths=10_000, n_steps=500)
        assert out["log_return"].shape == (10_000, 500)
        assert np.isfinite(out["log_return"]).all()
        assert (out["price"] > 0).all()
