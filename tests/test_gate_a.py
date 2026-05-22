"""Unit tests for ``sim.eval.gate_a`` — verdict logic + crash-protection (Task D)."""
from __future__ import annotations

import pytest

from eval.gate_a import crash_protection_summary, gate_a_decide


def _metrics(sortino, mean, p01):
    return {
        "sortino": sortino, "mean_return": mean, "p01_return": p01,
        "std": 0.01, "downside_std": 0.01, "p05_return": p01, "prob_loss": 0.5,
        "min_return": p01, "max_loss_pnl": -1.0, "mean_pnl": mean, "n_paths": 100,
        "strata_capital": 1.0,
    }


class TestGateADecide:
    def test_green_when_beats_baselines(self):
        res = {
            "strata": {0.2: _metrics(1.2, 0.02, -0.02), 0.5: _metrics(1.0, 0.02, -0.02)},
            "raw_plp": {0.2: _metrics(0.5, 0.01, -0.04), 0.5: _metrics(0.5, 0.01, -0.04)},
            "fixed_hedge_naive": {0.2: _metrics(0.6, 0.01, -0.03), 0.5: _metrics(0.6, 0.01, -0.03)},
        }
        v = gate_a_decide(res)
        assert v["verdict"] == "GREEN"

    def test_marginal_when_below_margin(self):
        res = {
            "strata": {0.2: _metrics(0.60, 0.01, -0.03)},
            "raw_plp": {0.2: _metrics(0.58, 0.01, -0.04)},
            "fixed_hedge_naive": {0.2: _metrics(0.55, 0.01, -0.03)},
        }
        v = gate_a_decide(res)
        assert v["verdict"] == "MARGINAL"

    def test_red_when_negative(self):
        res = {
            "strata": {0.2: _metrics(-0.5, -0.01, -0.05)},
            "raw_plp": {0.2: _metrics(-0.3, -0.005, -0.04)},
            "fixed_hedge_naive": {0.2: _metrics(-0.6, -0.02, -0.06)},
        }
        v = gate_a_decide(res)
        assert v["verdict"] == "RED"


class TestCrashProtectionSummary:
    def test_protection_visible_when_strata_better_in_crash(self):
        """Strata less-negative Sortino + shallower tail than raw_plp in crash."""
        crash = {
            "strata": {
                0.05: _metrics(-1.23, -0.018, -0.044),
                0.80: _metrics(-0.66, -0.009, -0.034),  # best protection
            },
            "raw_plp": {
                0.05: _metrics(-1.27, -0.023, -0.055),
                0.80: _metrics(-1.27, -0.023, -0.055),
            },
        }
        s = crash_protection_summary(crash)
        assert s["protection_visible"] is True
        assert s["best_protection_f"] == 0.80
        assert s["best_sortino_uplift"] == pytest.approx(-0.66 - (-1.27))
        # mean loss reduced (strata loses less)
        assert s["best_mean_loss_reduction"] > 0
        # tail shallower (strata p01 less negative)
        assert s["best_p01_reduction"] > 0

    def test_protection_not_visible_when_hedge_useless(self):
        """If strata is identical to raw_plp in crash, no protection."""
        crash = {
            "strata": {0.5: _metrics(-1.27, -0.023, -0.055)},
            "raw_plp": {0.5: _metrics(-1.27, -0.023, -0.055)},
        }
        s = crash_protection_summary(crash)
        assert s["protection_visible"] is False

    def test_signs_positive_mean_strata_better(self):
        crash = {
            "strata": {0.5: _metrics(-0.9, -0.012, -0.038)},
            "raw_plp": {0.5: _metrics(-1.27, -0.023, -0.055)},
        }
        s = crash_protection_summary(crash)
        per = s["per_f"][0.5]
        assert per["sortino_uplift"] > 0
        assert per["mean_loss_reduction"] > 0
        assert per["p01_reduction"] > 0

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            crash_protection_summary({"strata": {0.5: _metrics(-1, -0.01, -0.05)}})
