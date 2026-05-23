"""Unit tests for ``sim.engine.price_engine_pr_kou`` — Politis–Romano + Kou.

S2.1 scope: Politis–Romano stationary block bootstrap (Kou disabled).
S2.2 scope: Kou jump overlay (added in a follow-up commit).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from engine.price_engine_pr_kou import PolitisRomanoKouEngine


@pytest.fixture
def synth_returns() -> np.ndarray:
    """High-kurtosis synthetic returns (df=5 t-dist scaled to BTC-like σ)."""
    rng = np.random.default_rng(11)
    return rng.standard_t(df=5, size=20_000) * 0.003


@pytest.fixture
def clustered_returns() -> np.ndarray:
    """Alternating-σ blocks → non-zero ACF(|r|) at lags shorter than block."""
    rng = np.random.default_rng(7)
    n = 6000
    sigma = np.tile(np.repeat([0.001, 0.02], 100), n // 200 + 1)[:n]
    return rng.standard_normal(n) * sigma


# ---- Construction -------------------------------------------------------


class TestConstruction:
    def test_defaults(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns)
        assert e.mean_block_length == 4.0
        assert e.jump_intensity == 0.0      # S2.1: Kou off
        assert e.sigma_historical > 0

    def test_two_d_input_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            PolitisRomanoKouEngine(np.zeros((10, 10)))

    def test_block_length_below_one_rejected(self, synth_returns):
        with pytest.raises(ValueError, match="mean_block_length"):
            PolitisRomanoKouEngine(synth_returns, mean_block_length=0.5)

    def test_negative_intensity_rejected(self, synth_returns):
        with pytest.raises(ValueError, match="jump_intensity"):
            PolitisRomanoKouEngine(synth_returns, jump_intensity=-0.1)

    def test_prob_down_out_of_range_rejected(self, synth_returns):
        with pytest.raises(ValueError, match="jump_prob_down"):
            PolitisRomanoKouEngine(synth_returns, jump_prob_down=1.5)

    def test_asymmetry_guard(self, synth_returns):
        """down_jump_scale must be >= 1.2 * up_jump_scale (brief §S2.2)."""
        with pytest.raises(ValueError, match="asymmetry guardrail"):
            PolitisRomanoKouEngine(
                synth_returns, down_jump_scale=0.08, up_jump_scale=0.09
            )


# ---- Output shape + interface compatibility ----------------------------


class TestInterface:
    """Same surface as BootstrapEngine — drop-in for run_sweep."""

    def test_output_keys(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns, seed=1)
        out = e.simulate(n_paths=10, n_steps=20)
        assert set(out.keys()) == {"log_return", "standardized_shock", "price"}

    def test_output_shapes(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns, seed=1)
        out = e.simulate(n_paths=10, n_steps=20)
        assert out["log_return"].shape == (10, 20)
        assert out["standardized_shock"].shape == (10, 20)
        assert out["price"].shape == (10, 21)

    def test_init_price_in_column_zero(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns, seed=1)
        out = e.simulate(n_paths=5, n_steps=10, init_price=42.0)
        np.testing.assert_allclose(out["price"][:, 0], 42.0)

    def test_price_consistent_with_log_returns(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns, seed=2)
        out = e.simulate(n_paths=3, n_steps=15, init_price=100.0)
        np.testing.assert_allclose(
            out["price"][:, 1:],
            100.0 * np.exp(np.cumsum(out["log_return"], axis=1)),
            rtol=1e-12,
        )

    def test_sigma_historical_matches_input(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns)
        assert e.sigma_historical == pytest.approx(
            float(np.std(synth_returns, ddof=1))
        )


# ---- Reproducibility ----------------------------------------------------


class TestReproducibility:
    def test_same_seed_identical(self, synth_returns):
        e1 = PolitisRomanoKouEngine(synth_returns, seed=42)
        e2 = PolitisRomanoKouEngine(synth_returns, seed=42)
        np.testing.assert_array_equal(
            e1.simulate(20, 50)["log_return"],
            e2.simulate(20, 50)["log_return"],
        )

    def test_different_seed_diverges(self, synth_returns):
        e1 = PolitisRomanoKouEngine(synth_returns, seed=1)
        e2 = PolitisRomanoKouEngine(synth_returns, seed=2)
        assert not np.allclose(
            e1.simulate(20, 50)["log_return"],
            e2.simulate(20, 50)["log_return"],
        )


# ---- Politis–Romano: random geometric block length ---------------------


class TestPolitisRomano:
    """The defining property: block length is random Geometric(1/mean_BL)."""

    def test_block_length_is_random_not_fixed(self, synth_returns):
        """Run-length distribution of consecutive equal-delta indices should
        match a Geometric (NOT a fixed constant). Sanity: at least two
        distinct run lengths appear."""
        e = PolitisRomanoKouEngine(
            synth_returns, mean_block_length=8.0, seed=3
        )
        # Pull internals: get the index tensor by reusing the public path.
        rng = np.random.default_rng(e.seed)
        idx = e._bootstrap_indices(rng, n_paths=1, n_steps=2000).ravel()
        # Compute run-lengths of step+1 contiguity (treating wrap as a break).
        diffs = np.diff(idx)
        is_continue = (diffs == 1)
        # Block lengths = lengths of contiguous-True runs (+1 to count the start).
        run_lengths = []
        cur = 1
        for c in is_continue:
            if c:
                cur += 1
            else:
                run_lengths.append(cur)
                cur = 1
        run_lengths.append(cur)
        run_lengths = np.array(run_lengths)
        assert len(np.unique(run_lengths)) >= 2  # NOT a fixed block

    def test_mean_block_length_matches_target(self, synth_returns):
        """Empirical mean block length should be close to the configured one."""
        target = 8.0
        e = PolitisRomanoKouEngine(
            synth_returns, mean_block_length=target, seed=4
        )
        rng = np.random.default_rng(e.seed)
        idx = e._bootstrap_indices(rng, n_paths=1, n_steps=20_000).ravel()
        diffs = np.diff(idx)
        is_continue = (diffs == 1)
        run_lengths = []
        cur = 1
        for c in is_continue:
            if c:
                cur += 1
            else:
                run_lengths.append(cur)
                cur = 1
        run_lengths.append(cur)
        empirical = float(np.mean(run_lengths))
        # Geometric mean = 1/p = mean_block_length. Tolerance ±15% on 20k steps.
        assert empirical == pytest.approx(target, rel=0.15)

    def test_smaller_mean_block_more_restarts(self, synth_returns):
        """A smaller mean_block_length → more restarts → shorter runs."""
        rng2 = np.random.default_rng(5)
        rng16 = np.random.default_rng(5)
        e2 = PolitisRomanoKouEngine(synth_returns, mean_block_length=2.0, seed=5)
        e16 = PolitisRomanoKouEngine(synth_returns, mean_block_length=16.0, seed=5)
        i2 = e2._bootstrap_indices(rng2, n_paths=1, n_steps=5000).ravel()
        i16 = e16._bootstrap_indices(rng16, n_paths=1, n_steps=5000).ravel()
        # Larger mean_block → more contiguous (diff==1) than smaller mean_block.
        cont_2 = (np.diff(i2) == 1).mean()
        cont_16 = (np.diff(i16) == 1).mean()
        assert cont_16 > cont_2


# ---- Statistical preservation (no jumps) -------------------------------


class TestStatisticalPreservation:
    def test_sigma_preserved_within_tolerance(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns, mean_block_length=4.0, seed=6)
        out = e.simulate(n_paths=300, n_steps=2000)
        sampled_sigma = float(np.std(out["log_return"].ravel(), ddof=1))
        np.testing.assert_allclose(
            sampled_sigma, e.sigma_historical, rtol=0.03
        )

    def test_kurtosis_preserved(self, synth_returns):
        base_kurt = stats.kurtosis(synth_returns)
        e = PolitisRomanoKouEngine(synth_returns, mean_block_length=4.0, seed=7)
        out = e.simulate(n_paths=400, n_steps=2000)
        sampled = out["log_return"].ravel()
        assert stats.kurtosis(sampled) >= 0.7 * base_kurt

    def test_clustering_preserved_within_block(self, clustered_returns):
        """ACF(|r|) at lag 3 (< mean_block_length=8) > 0."""
        e = PolitisRomanoKouEngine(
            clustered_returns, mean_block_length=8.0, seed=8
        )
        out = e.simulate(n_paths=1, n_steps=4000)
        r = out["log_return"][0]
        ar = np.abs(r) - np.abs(r).mean()
        n = len(ar)
        var = float(np.dot(ar, ar) / n)
        acf3 = float(np.dot(ar[:-3], ar[3:]) / ((n - 3) * var))
        assert acf3 > 0.05


# ---- Kou disabled by default at S2.1 -----------------------------------


class TestKouOff:
    def test_default_intensity_is_zero(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns)
        assert e.jump_intensity == 0.0

    def test_no_jumps_added_when_disabled(self, synth_returns):
        e = PolitisRomanoKouEngine(synth_returns, seed=9)
        rng = np.random.default_rng(e.seed)
        jumps = e._kou_jumps(rng, shape=(50, 100))
        assert np.all(jumps == 0.0)
