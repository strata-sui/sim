"""Unit tests for ``sim.model.token_bucket`` — S4.4 limiter."""
from __future__ import annotations

import pytest

from model.token_bucket import TokenBucket, lp_flight_step_bucketed
from model.trader_flow import CONSERVATIVE_ANCHOR, PESSIMAL_ANCHOR, lp_flight_step


class TestTokenBucketConstruction:
    def test_capacity_nonpositive_rejected(self):
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(capacity=0.0, refill_rate=0.1, tokens=0.1)
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(capacity=-1.0, refill_rate=0.1, tokens=0.1)

    def test_refill_nonpositive_rejected(self):
        with pytest.raises(ValueError, match="refill_rate"):
            TokenBucket(capacity=1.0, refill_rate=0.0, tokens=0.1)

    def test_refill_exceeds_capacity_rejected(self):
        with pytest.raises(ValueError, match="refill_rate"):
            TokenBucket(capacity=0.1, refill_rate=0.5, tokens=0.1)

    def test_initial_tokens_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="initial tokens"):
            TokenBucket(capacity=1.0, refill_rate=0.5, tokens=-0.1)
        with pytest.raises(ValueError, match="initial tokens"):
            TokenBucket(capacity=1.0, refill_rate=0.5, tokens=2.0)


class TestTokenBucketConsume:
    def test_refill_caps_at_capacity(self):
        b = TokenBucket(capacity=1.0, refill_rate=0.5, tokens=1.0)
        # Already at cap, refill keeps tokens at cap.
        b.consume_per_step(0.0)
        assert b.tokens == 1.0

    def test_consume_drains_then_refills(self):
        b = TokenBucket(capacity=1.0, refill_rate=0.2, tokens=1.0)
        # Step 1: refill to 1.0 (already there), consume 0.8 → tokens=0.2
        assert b.consume_per_step(0.8) == pytest.approx(0.8)
        assert b.tokens == pytest.approx(0.2)
        # Step 2: refill to 0.4, consume 0.4 → tokens=0.0
        assert b.consume_per_step(0.4) == pytest.approx(0.4)
        assert b.tokens == pytest.approx(0.0)
        # Step 3: refill to 0.2, consume 1.0 → only 0.2 granted, tokens=0.0
        assert b.consume_per_step(1.0) == pytest.approx(0.2)
        assert b.tokens == pytest.approx(0.0)

    def test_burst_then_throttle(self):
        """Bucket with capacity > refill allows burst then throttles."""
        b = TokenBucket(capacity=1.0, refill_rate=0.1, tokens=1.0)
        # Burst: consume 1.0 once.
        assert b.consume_per_step(1.0) == pytest.approx(1.0)
        # Now throttled — only refill_rate per step.
        for _ in range(5):
            granted = b.consume_per_step(1.0)
            assert granted == pytest.approx(0.1)

    def test_negative_request_rejected(self):
        b = TokenBucket(capacity=1.0, refill_rate=0.5, tokens=0.5)
        with pytest.raises(ValueError, match="requested"):
            b.consume_per_step(-0.1)


class TestS1Equivalence:
    """Sanity: capacity = refill_rate = θ reproduces S1 hard-cap."""

    def test_factory_constructs_equivalent_bucket(self):
        b = TokenBucket.s1_hardcap_equivalent(0.10)
        assert b.capacity == 0.10
        assert b.refill_rate == 0.10
        assert b.tokens == 0.10

    def test_reduces_to_s1_hardcap(self):
        """For any drawdown sequence, S4 bucket matches S1 lp_flight_step."""
        anchor = CONSERVATIVE_ANCHOR
        l0 = 1_000_000.0
        # Drawdown sequence including small + large + zero values.
        dd_seq = [0.0, 0.05, 0.20, 0.0, 0.50, 0.10]
        # S1 path:
        l_s1 = l0
        s1_trace = []
        for dd in dd_seq:
            l_s1 = float(lp_flight_step(l_s1, dd, anchor))
            s1_trace.append(l_s1)
        # S4 bucket equivalent:
        b = TokenBucket.s1_hardcap_equivalent(anchor.theta_limiter)
        l_s4 = l0
        s4_trace = []
        for dd in dd_seq:
            l_s4 = lp_flight_step_bucketed(l_s4, dd, anchor, b)
            s4_trace.append(l_s4)
        # Match exactly (modulo float).
        for s1v, s4v in zip(s1_trace, s4_trace):
            assert s1v == pytest.approx(s4v)


class TestLpFlightStepBucketed:
    def test_no_drawdown_no_flight(self):
        b = TokenBucket.s1_hardcap_equivalent(0.10)
        l_new = lp_flight_step_bucketed(1e6, 0.0, CONSERVATIVE_ANCHOR, b)
        assert l_new == pytest.approx(1e6)

    def test_pessimal_anchor_with_large_bucket_allows_full_desired(self):
        """Pessimal θ=1.0 hard-cap, but if bucket capacity > 1.0 the desired
        is still capped at 1.0 by the clip in lp_flight_step_bucketed."""
        b = TokenBucket(capacity=2.0, refill_rate=2.0, tokens=2.0)
        # γ=10, DD=0.05 → desired = 0.5
        l_new = lp_flight_step_bucketed(1e6, 0.05, PESSIMAL_ANCHOR, b)
        assert l_new == pytest.approx(0.5e6)

    def test_throttling_under_sustained_drawdown(self):
        """Bucket throttles after burst; sustained drawdown sees actual flight
        drop to refill_rate per step."""
        anchor = CONSERVATIVE_ANCHOR  # γ=3, θ=0.10
        b = TokenBucket(capacity=0.5, refill_rate=0.05, tokens=0.5)
        l = 1e6
        rates = []
        for _ in range(8):
            l_new = lp_flight_step_bucketed(l, 0.5, anchor, b)  # huge DD
            rates.append((l - l_new) / l)
            l = l_new
        # First step: burst takes large fraction; later steps throttle to ~0.05.
        assert rates[-1] < rates[0]
        assert rates[-1] == pytest.approx(0.05, abs=0.001)
