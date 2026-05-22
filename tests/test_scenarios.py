"""Unit tests for ``sim.eval.scenarios`` — crash-window replay (S1 Task C)."""
from __future__ import annotations

import numpy as np
import pytest

from eval.scenarios import ScenarioReplay, find_crash_windows


class TestFindCrashWindows:
    def test_finds_window_below_threshold(self):
        # A clear -10% drop over 2 bars sits in the middle of calm bars.
        r = np.array([0.0, 0.0, np.log(0.95), np.log(0.95), 0.0, 0.0])
        w = find_crash_windows(r, n_steps=2, threshold=-0.04)
        # The (0.95, 0.95) window cumulates to ~-9.75% <= -4% → qualifies.
        assert w.shape[1] == 2
        assert w.shape[0] >= 1

    def test_no_windows_when_calm(self):
        r = np.zeros(20)
        w = find_crash_windows(r, n_steps=4, threshold=-0.04)
        assert w.shape == (0, 4)

    def test_short_series_returns_empty(self):
        r = np.array([0.0, 0.0])
        w = find_crash_windows(r, n_steps=4, threshold=-0.04)
        assert w.shape == (0, 4)

    def test_threshold_is_inclusive_cumulative(self):
        # Exactly -4% over 4 bars (cumulative) should qualify.
        per_bar = np.log1p(-0.04) / 4.0
        r = np.full(4, per_bar)
        w = find_crash_windows(r, n_steps=4, threshold=-0.04)
        assert w.shape[0] == 1

    def test_deeper_threshold_fewer_windows(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 5000)
        w_shallow = find_crash_windows(r, 4, threshold=-0.02)
        w_deep = find_crash_windows(r, 4, threshold=-0.05)
        assert len(w_deep) <= len(w_shallow)

    def test_invalid_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            find_crash_windows(np.zeros(10), 0)


class TestScenarioReplay:
    def _windows(self):
        # 3 crash windows of length 4, each a sustained ~-1.5%/bar slide.
        per = np.log1p(-0.015)
        return np.full((3, 4), per)

    def test_simulate_shapes(self):
        sr = ScenarioReplay(self._windows(), sigma_historical=0.003)
        out = sr.simulate(n_paths=3, n_steps=4, init_price=60000.0)
        assert out["price"].shape == (3, 5)
        assert out["log_return"].shape == (3, 4)
        assert out["standardized_shock"].shape == (3, 4)

    def test_prices_reflect_crash(self):
        sr = ScenarioReplay(self._windows(), sigma_historical=0.003)
        out = sr.simulate(n_paths=3, n_steps=4, init_price=60000.0)
        # Every path ends well below init (sustained down move).
        assert np.all(out["price"][:, -1] < out["price"][:, 0])
        # ~ -1.5%*4 ≈ -5.9% final
        assert np.all(out["price"][:, -1] / out["price"][:, 0] < 0.95)

    def test_resample_when_more_paths_requested(self):
        sr = ScenarioReplay(self._windows(), sigma_historical=0.003)
        out = sr.simulate(n_paths=100, n_steps=4)
        assert out["price"].shape == (100, 5)

    def test_subset_when_fewer_paths_requested(self):
        sr = ScenarioReplay(self._windows(), sigma_historical=0.003)
        out = sr.simulate(n_paths=2, n_steps=4)
        assert out["price"].shape == (2, 5)

    def test_n_windows_exposed(self):
        sr = ScenarioReplay(self._windows(), sigma_historical=0.003)
        assert sr.n_windows == 3

    def test_empty_windows_raises(self):
        sr = ScenarioReplay(np.empty((0, 4)), sigma_historical=0.003)
        with pytest.raises(ValueError, match="no crash windows"):
            sr.simulate(n_paths=10, n_steps=4)

    def test_step_mismatch_raises(self):
        sr = ScenarioReplay(self._windows(), sigma_historical=0.003)
        with pytest.raises(ValueError, match="window length"):
            sr.simulate(n_paths=3, n_steps=8)

    def test_sigma_historical_property(self):
        sr = ScenarioReplay(self._windows(), sigma_historical=0.0042)
        assert sr.sigma_historical == 0.0042
