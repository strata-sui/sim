"""Stochastic SVI dynamics — three coupled OU processes (S3.2 → S3.7).

Three SVI parameters evolve as correlated Ornstein–Uhlenbeck processes:

    a   — variance-intercept (ATM level)
    b   — wing slope
    ρ   — skew

m and σ (SVI smoothing parameter) remain STATIC in S3 (anchored to the
historical mean of the surface). S4+ can promote them too if needed.

The three OU innovation Brownians are correlated via a Cholesky factor of
a user-supplied 3×3 correlation matrix. Validation: Cholesky decomposition
fails if the matrix is not positive-definite, which is the PSD check the
brief asks for. The factor is reusable each step (homoscedastic Brownian
correlation, standard practice).

S3.3 extends this primitive to ALSO consume external BTC-return shocks
(leverage-effect coupling). S3.5 attaches the no-arbitrage gate. S3.7
wraps this with PolitisRomanoKouEngine into a unified joint engine that
emits (price_path, svi_path) for the strategy layer.

This module ships in two atomic units:
  * S3.2 (THIS file at commit time): SVIDynamicsParams + StochasticSVIEngine
    with correlated SVI innovations only.
  * S3.3 (next commit): add ``external_innovations`` channel for BTC-leverage
    coupling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.ou import OUParams, simulate_ou


@dataclass(frozen=True)
class SVIDynamicsParams:
    """Stochastic SVI parametrization: 3 OU + 3×3 correlation + 2 static.

    Attrs:
        a:           OUParams for the variance-intercept.
        b:           OUParams for the wing slope.
        rho:         OUParams for skew.
        correlation: 3×3 symmetric positive-definite correlation matrix
                     among (a, b, rho) innovation Brownians (in that order).
                     Diagonal must be 1.0.
        m:           static SVI shift parameter (kept fixed in S3).
        sigma:       static SVI smoothing parameter (kept fixed in S3, > 0).
    """

    a: OUParams
    b: OUParams
    rho: OUParams
    correlation: np.ndarray
    m: float
    sigma: float

    def __post_init__(self) -> None:
        corr = np.asarray(self.correlation, dtype=float)
        if corr.shape != (3, 3):
            raise ValueError(
                f"correlation must be (3,3), got {corr.shape}"
            )
        if not np.allclose(corr, corr.T, atol=1e-9):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-9):
            raise ValueError("correlation diagonal must be 1.0")
        # PSD test = Cholesky succeeds.
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "correlation must be positive-definite (Cholesky failed)"
            ) from exc
        # Freeze normalized copy so external mutation doesn't break things.
        object.__setattr__(self, "correlation", corr)
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")


class StochasticSVIEngine:
    """Multi-factor stochastic SVI: 3 coupled OU processes.

    Produces (a, b, rho) paths whose Brownian innovations are correlated per
    the user-supplied 3×3 correlation matrix. Initial values default to each
    OU's long-run mean μ.
    """

    def __init__(self, params: SVIDynamicsParams, seed: int = 0) -> None:
        self.params = params
        self.seed = int(seed)
        # Cholesky factor: L · L.T = correlation. Apply as Z_corr = Z · L.T.
        self._chol = np.linalg.cholesky(params.correlation)

    @property
    def correlation(self) -> np.ndarray:
        return self.params.correlation

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float = 1.0,
        a0: float | None = None,
        b0: float | None = None,
        rho0: float | None = None,
        external_innovations: np.ndarray | None = None,
    ) -> dict:
        """Simulate correlated (a, b, ρ) paths.

        Args:
            n_paths:              number of independent paths.
            n_steps:              number of evolution steps.
            dt:                   time-step size.
            a0, b0, rho0:         optional initial values (else each OU's μ).
            external_innovations: optional (n_paths, n_steps, 3) iid N(0,1)
                                  innovations BEFORE Cholesky correlation.
                                  Used by S3.3 to couple to a BTC-return shock
                                  channel (the BTC channel is folded into the
                                  upstream correlation matrix; here we just
                                  consume the SVI-side iid block).

        Returns:
            dict with arrays:
              'a':   (n_paths, n_steps+1)
              'b':   (n_paths, n_steps+1)
              'rho': (n_paths, n_steps+1)
              'm':   scalar (static)
              'sigma': scalar (static)
        """
        if n_paths < 1 or n_steps < 1:
            raise ValueError("n_paths and n_steps must be >= 1")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        if external_innovations is None:
            rng = np.random.default_rng(self.seed)
            iid = rng.standard_normal((n_paths, n_steps, 3))
        else:
            iid = np.asarray(external_innovations, dtype=float)
            if iid.shape != (n_paths, n_steps, 3):
                raise ValueError(
                    f"external_innovations shape {iid.shape} != "
                    f"expected {(n_paths, n_steps, 3)}"
                )

        # Apply Cholesky to correlate: Z_corr[..., :] = Z_iid[..., :] @ L.T
        # so Cov(Z_corr) = L · L.T = correlation. ✓
        correlated = iid @ self._chol.T

        a_paths = simulate_ou(
            self.params.a, n_paths, n_steps, dt=dt, x0=a0,
            innovations=correlated[:, :, 0],
        )
        b_paths = simulate_ou(
            self.params.b, n_paths, n_steps, dt=dt, x0=b0,
            innovations=correlated[:, :, 1],
        )
        rho_paths = simulate_ou(
            self.params.rho, n_paths, n_steps, dt=dt, x0=rho0,
            innovations=correlated[:, :, 2],
        )
        return {
            "a": a_paths,
            "b": b_paths,
            "rho": rho_paths,
            "m": self.params.m,
            "sigma": self.params.sigma,
        }
