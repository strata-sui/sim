"""S4.4 — Token-bucket limiter for LP-flight (continuous-θ_limiter form).

Replaces the S1 hard-cap θ with a classical token-bucket:

    tokens_t = min(capacity, tokens_{t-1} + refill_rate)
    actual_flight = min(desired_flight, tokens_t)
    tokens_t -= actual_flight

S1 limit case (sanity): capacity == refill_rate == θ. Each step starts
with tokens = θ (refilled instantly), so the bucket behaves identically
to the S1 hard-cap. The S4 form lets bucket capacity > refill rate, so
LP flight can BURST briefly (drain accumulated tokens) then throttle back
to refill_rate — matching the protocol's actual token-bucket withdraw
limiter semantics (CLAUDE.md §4 verified mechanics).

Brief §S4.4 acceptance:
  * Token-bucket refill + capacity parameters documented (here).
  * Reduces to S1 hard-cap as limit case (test_reduces_to_s1_hardcap).
"""
from __future__ import annotations

from dataclasses import dataclass

from model.trader_flow import TraderFlowAnchor


@dataclass
class TokenBucket:
    """Withdraw-rate limiter. ``tokens`` increments by ``refill_rate`` per step,
    capped at ``capacity``. ``consume_per_step(x)`` refills first then deducts
    ``min(x, tokens)`` and returns the actual amount granted.
    """

    capacity: float       # maximum tokens the bucket can hold
    refill_rate: float    # tokens added per simulation step
    tokens: float         # current token balance

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {self.capacity}")
        if self.refill_rate <= 0:
            raise ValueError(f"refill_rate must be > 0, got {self.refill_rate}")
        if self.refill_rate > self.capacity:
            raise ValueError(
                f"refill_rate ({self.refill_rate}) must be <= capacity "
                f"({self.capacity})"
            )
        if self.tokens < 0 or self.tokens > self.capacity:
            raise ValueError(
                f"initial tokens must be in [0, capacity={self.capacity}], "
                f"got {self.tokens}"
            )

    @classmethod
    def s1_hardcap_equivalent(cls, theta: float) -> "TokenBucket":
        """Construct a bucket that reproduces the S1 hard-cap exactly.

        capacity = refill_rate = θ → each step refills to θ instantly → the
        per-step cap is min(desired, θ), identical to ``lp_flight_step``.
        """
        return cls(capacity=theta, refill_rate=theta, tokens=theta)

    def consume_per_step(self, requested: float) -> float:
        """Refill then grant ``min(requested, tokens)``. Returns actual granted."""
        if requested < 0:
            raise ValueError(f"requested must be >= 0, got {requested}")
        self.tokens = min(self.capacity, self.tokens + self.refill_rate)
        granted = min(requested, self.tokens)
        self.tokens -= granted
        return granted


def lp_flight_step_bucketed(
    l_other: float,
    drawdown_pct: float,
    anchor: TraderFlowAnchor,
    bucket: TokenBucket,
) -> float:
    """One-step LP flight with continuous γ + token-bucket θ (S4.4 form).

    Knob 5 (continuous γ): desired = clip(γ · max(0, drawdown), 0, 1).
    Knob 6 (S4 bucket):    actual  = bucket.consume_per_step(desired).
    Returns new L_other (>= 0).
    """
    desired = max(0.0, anchor.gamma_flight * max(0.0, drawdown_pct))
    desired = min(desired, 1.0)
    actual = bucket.consume_per_step(desired)
    return float(l_other) * (1.0 - actual)
