"""Unit tests for ``sim.eval.objective`` — tail-weighted f*(w_crash) (Task R2-B)."""
from __future__ import annotations

import numpy as np
import pytest

from eval.objective import (
    blended_objective,
    f_star_at_weight,
    f_star_curve,
    interior_band,
    summarize_f_star_curve,
)


def _mk(f_grid, sortinos):
    """Build a sweep-results-shaped dict for one strategy."""
    return {"strata": {f: {"sortino": s} for f, s in zip(f_grid, sortinos)}}


# Benign monotonic DOWN, crash monotonic UP (the real S1 shape).
F = [0.05, 0.2, 0.5, 0.8]
BENIGN = _mk(F, [0.574, 0.542, 0.466, 0.375])
CRASH = _mk(F, [-1.233, -1.130, -0.891, -0.660])


class TestBlend:
    def test_w0_is_benign(self):
        f_grid, J = blended_objective(BENIGN, CRASH, w_crash=0.0)
        np.testing.assert_allclose(J, [0.574, 0.542, 0.466, 0.375])

    def test_w1_is_crash(self):
        _, J = blended_objective(BENIGN, CRASH, w_crash=1.0)
        np.testing.assert_allclose(J, [-1.233, -1.130, -0.891, -0.660])

    def test_w_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="w_crash"):
            blended_objective(BENIGN, CRASH, w_crash=1.5)

    def test_grid_mismatch_rejected(self):
        bad = _mk([0.1, 0.2], [0.5, 0.4])
        with pytest.raises(ValueError, match="f-grids must match"):
            blended_objective(BENIGN, bad, w_crash=0.5)


class TestFStarAtWeight:
    def test_low_weight_low_boundary(self):
        """At w=0 benign dominates → f* at low boundary, not interior."""
        r = f_star_at_weight(BENIGN, CRASH, w_crash=0.0)
        assert r["f_star"] == 0.05
        assert r["is_interior"] is False

    def test_high_weight_high_boundary(self):
        """At w=1 crash dominates → f* at high boundary, not interior."""
        r = f_star_at_weight(BENIGN, CRASH, w_crash=1.0)
        assert r["f_star"] == 0.8
        assert r["is_interior"] is False


class TestInteriorBand:
    def test_interior_emerges_in_a_band(self):
        """Across the full w grid an interior f* should appear at mid weights."""
        curve = f_star_curve(BENIGN, CRASH, np.linspace(0, 1, 21))
        band = interior_band(curve)
        # On the real S1 numbers an interior region exists around w≈0.25-0.30.
        assert band is not None
        w_lo, w_hi = band
        assert 0.0 < w_lo <= w_hi < 1.0

    def test_no_interior_when_one_curve_flat_dominates(self):
        """If both curves favor the SAME boundary, f* never goes interior."""
        # Both monotonic DOWN → blend always favors low f.
        benign = _mk(F, [0.6, 0.5, 0.4, 0.3])
        crash = _mk(F, [0.3, 0.2, 0.1, 0.0])
        curve = f_star_curve(benign, crash, np.linspace(0, 1, 11))
        assert interior_band(curve) is None

    def test_summary_reports_threshold_and_band(self):
        curve = f_star_curve(BENIGN, CRASH, np.linspace(0, 1, 21))
        s = summarize_f_star_curve(curve)
        assert s["interior_exists"] is True
        assert s["f_star_low_weight"] == 0.05
        assert s["f_star_high_weight"] == 0.8
        # f* leaves the low boundary at some positive tail-weight.
        assert s["threshold_leaves_low_boundary"] is not None
        assert 0.0 < s["threshold_leaves_low_boundary"] < 1.0

    def test_summary_no_interior_case(self):
        benign = _mk(F, [0.6, 0.5, 0.4, 0.3])
        crash = _mk(F, [0.3, 0.2, 0.1, 0.0])
        curve = f_star_curve(benign, crash, np.linspace(0, 1, 11))
        s = summarize_f_star_curve(curve)
        assert s["interior_exists"] is False
        assert s["interior_band"] is None
