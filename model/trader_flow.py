"""Trader-flow exogenous model — 6 knobs (canonical spec at
``../../specs/trader_flow_spec.md``, project root, local-only).

This is the single most load-bearing assumption of the simulator
(CLAUDE.md §3: "one assumption, two legs" — feeds BOTH PLP yield AND
hedge funding). Per S1 collapse rules (spec §6), knobs 1-3 are
continuous; knob 4 (``kappa_strike``) is a binary toggle; knob 5
(``gamma_flight``) is a coarse step; knob 6 (``theta_limiter``) is a
hard cap. Continuous strike-ladder + token-bucket limiter land at S4.

Three calibration anchors (spec §5):
    PESSIMAL_ANCHOR     — stress bound, NOT used for Gate. Should
                          produce f* ≈ no-zone (sanity check).
    CONSERVATIVE_ANCHOR — the Gate operating assumption (R2 guard).
    OPTIMISTIC_ANCHOR   — ceiling, disclosed not Gate.

Strike-local Umbra DN share at hedge strike (from spec §1.2):
    u(k) = N_U(k) / (N_U(k) + N_O(k))

S1 parametrization: ``u = 1 - kappa * (1 - f)`` — linear interp between
``u=1`` at kappa=0 (Strata alone) and ``u=f`` at kappa=1 (pure wash).
The spec's loose phrasing "kappa=0.30 → u ≈ f" is shorthand for
"heavy dilution"; this implementation puts the true wash at kappa=1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- Anchors (spec §5 numeric table) -------------------------------------


@dataclass(frozen=True)
class TraderFlowAnchor:
    """Six-knob exogenous trader-flow / LP-capital model.

    All knobs natural-units; see ``specs/trader_flow_spec.md §3`` for full
    parametrization + spec §5 for band-specific rationale.

    Attrs:
        lambda_0:       baseline target utilization ∈ (0, max_exposure].
        beta_vol:       elasticity of volume to realized-vol excess.
        phi_informed:   fraction of arriving flow that is crash-correlated.
        kappa_strike:   strike-concentration at hedge strike, ∈ [0, 1].
                        S1 binary toggle: {0, 0.30}; S4+ continuous.
        gamma_flight:   LP-flight beta vs PLP drawdown.
        theta_limiter:  max realized flight per step (limiter throttle).
    """

    lambda_0: float
    beta_vol: float
    phi_informed: float
    kappa_strike: float
    gamma_flight: float
    theta_limiter: float

    def __post_init__(self) -> None:
        if not 0.0 < self.lambda_0 <= 1.0:
            raise ValueError(f"lambda_0 must be ∈ (0, 1], got {self.lambda_0}")
        if self.beta_vol < 0:
            raise ValueError(f"beta_vol must be >= 0, got {self.beta_vol}")
        if not 0.0 <= self.phi_informed <= 1.0:
            raise ValueError(
                f"phi_informed must be ∈ [0, 1], got {self.phi_informed}"
            )
        if not 0.0 <= self.kappa_strike <= 1.0:
            raise ValueError(
                f"kappa_strike must be ∈ [0, 1], got {self.kappa_strike}"
            )
        if self.gamma_flight < 0:
            raise ValueError(f"gamma_flight must be >= 0, got {self.gamma_flight}")
        if not 0.0 <= self.theta_limiter <= 1.0:
            raise ValueError(
                f"theta_limiter must be ∈ [0, 1], got {self.theta_limiter}"
            )


# Pessimal — worst-case stress (spec §5 table column "Pessimal").
PESSIMAL_ANCHOR = TraderFlowAnchor(
    lambda_0=0.005,
    beta_vol=0.0,
    phi_informed=1.0,
    kappa_strike=0.30,    # high → u close to f → near-wash
    gamma_flight=10.0,
    theta_limiter=1.0,    # no throttle
)

# Conservative — the Gate-A operating anchor (R2 guard says no false-RED below this).
CONSERVATIVE_ANCHOR = TraderFlowAnchor(
    lambda_0=0.15,
    beta_vol=3.0,
    phi_informed=0.30,
    kappa_strike=0.10,
    gamma_flight=3.0,
    theta_limiter=0.10,
)

# Optimistic — ceiling (disclosed band end, never used as Gate anchor).
OPTIMISTIC_ANCHOR = TraderFlowAnchor(
    lambda_0=0.50,
    beta_vol=10.0,
    phi_informed=0.10,
    kappa_strike=0.05,
    gamma_flight=0.5,
    theta_limiter=0.02,
)


# ---- Mechanics ------------------------------------------------------------

# Default max_exposure from CLAUDE.md §4 verified protocol params.
MAX_EXPOSURE = 0.80


def u_strike_local(
    kappa_strike: float | np.ndarray,
    f: float | np.ndarray,
) -> np.ndarray:
    """Strike-local Umbra DN-ownership share at the hedge strike.

    Linear interp:
        u(kappa=0) = 1     Strata is the sole DN holder at the hedge
                           strike → maximum hedge genuineness.
        u(kappa=1) = f     Pure wash: Umbra's DN share matches its LP share
                           → ``(u - f) = 0`` → expected net hedge = 0 per
                           the spec §1.2 formula.

    Args:
        kappa_strike: strike-concentration knob ∈ [0, 1].
        f:            aggregate LP-share fraction.

    Returns:
        u ∈ [f, 1], same shape as broadcast(kappa, f).
    """
    k = np.asarray(kappa_strike, dtype=float)
    f_arr = np.asarray(f, dtype=float)
    return 1.0 - k * (1.0 - f_arr)


def trader_flow_step(
    pool_balance: float | np.ndarray,
    realized_vol: float | np.ndarray,
    sigma_long_run: float,
    log_return_recent: float | np.ndarray,
    anchor: TraderFlowAnchor,
    rho_info: float = 1.0,
    max_exposure: float = MAX_EXPOSURE,
) -> dict:
    """One-step exogenous trader flow against the pool.

    Implements knobs 1–3 (spec §3):
        Knob 1 (λ₀):       baseline volume = λ₀ · max_exposure · balance
        Knob 2 (β_vol):    volume scaled by (1 + β·max(0, (σ-σ̄)/σ̄))
        Knob 3 (φ_info):   DN fraction = φ·informed_bias + (1-φ)·0.5

    Args:
        pool_balance:      USD pool balance (≥ 0). Vectorizable.
        realized_vol:      rolling realized vol proxy this step.
        sigma_long_run:    long-run mean of realized_vol (> 0).
        log_return_recent: recent log-return — drives informed-bias.
        anchor:            6-knob calibration.
        rho_info:          informed reactivity (default 1.0).
        max_exposure:      pool exposure cap (default 0.80, §4).

    Returns:
        dict with vectorized arrays:
            ``dn_notional``    arriving DN-contract notional in $.
            ``up_notional``    arriving UP-contract notional in $.
            ``total_notional`` dn + up.
            ``dn_fraction``    direction split applied this step.
            ``lambda_eff``     λ₀ · (1 + β · vol_excess) — the
                               vol-elasticity-adjusted utilization target.
    """
    if sigma_long_run <= 0:
        raise ValueError(f"sigma_long_run must be > 0, got {sigma_long_run}")

    bal = np.asarray(pool_balance, dtype=float)
    rv = np.asarray(realized_vol, dtype=float)
    lrr = np.asarray(log_return_recent, dtype=float)

    # Knob 2: volume excess kicks in only when vol > long-run.
    vol_excess = np.maximum(0.0, (rv - sigma_long_run) / sigma_long_run)
    lambda_eff = anchor.lambda_0 * (1.0 + anchor.beta_vol * vol_excess)

    # Knob 1 (scaled by 2): arriving total notional.
    total_notional = lambda_eff * max_exposure * bal

    # Knob 3: directional bias. Informed flow is correlated -log_return:
    # crash (negative return) -> b_informed positive -> DN fraction > 0.5.
    b_informed = np.clip(-rho_info * lrr / sigma_long_run, -1.0, 1.0)
    dn_frac_informed = 0.5 * (1.0 + b_informed)
    dn_frac = (
        anchor.phi_informed * dn_frac_informed
        + (1.0 - anchor.phi_informed) * 0.5
    )

    dn_notional = dn_frac * total_notional
    up_notional = (1.0 - dn_frac) * total_notional
    return {
        "dn_notional": dn_notional,
        "up_notional": up_notional,
        "total_notional": total_notional,
        "dn_fraction": dn_frac,
        "lambda_eff": lambda_eff,
    }


def sample_kappa_ladder(
    n_strikes: int,
    alpha: float = 2.0,
    beta: float = 5.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """S4.2 — per-strike κ drawn from Beta(α, β), one value per ladder strike.

    Replaces the S1 binary {0, 0.30} toggle with a continuous distribution
    on [0, 1] per spec §3 Knob-4 continuous form + brief §S4.2.

    Default prior: Beta(α=2, β=5) → mean = α/(α+β) = 0.286, mode ≈ 0.17.
    Anchored to the same intuition as S1's conservative κ ≈ 0.10–0.30:
    most independent trader DN flow is near-ATM, so at OTM hedge strikes
    the local share dilution is on average modest. NOT optimized to the
    f-curve shape — this is an adversarial prior, NOT a tuned parameter.

    Args:
        n_strikes: ladder size (M).
        alpha, beta: Beta(α, β) shape parameters (both > 0).
        rng: optional numpy Generator for seeded draws.

    Returns:
        (n_strikes,) ndarray with values in (0, 1).
    """
    if n_strikes < 1:
        raise ValueError(f"n_strikes must be >= 1, got {n_strikes}")
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha, beta must be > 0; got α={alpha}, β={beta}")
    if rng is None:
        rng = np.random.default_rng()
    return rng.beta(alpha, beta, size=n_strikes)


def lp_flight_step(
    l_other: float | np.ndarray,
    drawdown_pct: float | np.ndarray,
    anchor: TraderFlowAnchor,
) -> np.ndarray:
    """One-step adverse LP flight under (gamma, theta) (knobs 5 + 6).

    desired_rate = clip(γ · max(0, drawdown_pct), 0, 1)
    actual_rate  = min(desired_rate, θ_limiter)        S1 hard-cap
    L_other_t+1  = L_other_t · (1 - actual_rate)

    Args:
        l_other:      other-LP capital before this step (>= 0).
        drawdown_pct: PLP-NAV drawdown ∈ [0, 1] (e.g. 0.10 = 10% DD).
        anchor:       calibration (uses γ_flight, θ_limiter only).

    Returns:
        new L_other (>= 0), same shape as broadcast(l_other, drawdown_pct).
    """
    l = np.asarray(l_other, dtype=float)
    dd = np.maximum(0.0, np.asarray(drawdown_pct, dtype=float))
    desired = np.clip(anchor.gamma_flight * dd, 0.0, 1.0)
    actual = np.minimum(desired, anchor.theta_limiter)
    return l * (1.0 - actual)
