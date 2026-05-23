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
    """Stochastic SVI parametrization: 3 OU + correlation + 2 static + leverage.

    Attrs:
        a:               OUParams for the variance-intercept.
        b:               OUParams for the wing slope.
        rho:             OUParams for skew.
        correlation:     3×3 symmetric positive-definite correlation matrix
                         among (a, b, rho) innovation Brownians (in that
                         order). Diagonal must be 1.0.
        m:               static SVI shift parameter (kept fixed in S3).
        sigma:           static SVI smoothing parameter (kept fixed, > 0).
        btc_correlation: 3-vector of correlations between BTC RETURN shock
                         (channel 0) and (a, b, rho) shocks. Default
                         (0, 0, 0) = no leverage coupling (back-compat).
                         Convention (empirical defaults derived in S3.4):
                           corr(Z_btc, Z_a)   < 0  (vol rises on down moves)
                           corr(Z_btc, Z_b)   < 0  (wing widens on crash)
                           corr(Z_btc, Z_rho) > 0  (skew steepens — Z_btc<0
                                                   ⇒ Z_rho<0 ⇒ rho more neg)
                         The full 4×4 joint correlation (BTC + 3 SVI) must
                         be PSD; validation = Cholesky on the joint matrix.
    """

    a: OUParams
    b: OUParams
    rho: OUParams
    correlation: np.ndarray
    m: float
    sigma: float
    btc_correlation: np.ndarray = None  # default set in __post_init__

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
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "correlation must be positive-definite (Cholesky failed)"
            ) from exc
        object.__setattr__(self, "correlation", corr)
        # BTC coupling (S3.3) — defaults to zeros = no leverage.
        if self.btc_correlation is None:
            btc = np.zeros(3, dtype=float)
        else:
            btc = np.asarray(self.btc_correlation, dtype=float)
            if btc.shape != (3,):
                raise ValueError(
                    f"btc_correlation must be (3,), got {btc.shape}"
                )
            if not np.all((-1.0 <= btc) & (btc <= 1.0)):
                raise ValueError("btc_correlation entries must be in [-1, 1]")
        # Joint 4×4 PSD check.
        joint = np.eye(4)
        joint[0, 1:] = btc
        joint[1:, 0] = btc
        joint[1:, 1:] = corr
        try:
            np.linalg.cholesky(joint)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "joint 4x4 (BTC + 3 SVI) correlation must be PSD "
                f"(check btc_correlation={btc} vs SVI corr)"
            ) from exc
        object.__setattr__(self, "btc_correlation", btc)
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
        # Cholesky for the SVI-only correlation (used when no return-shock
        # coupling is requested).
        self._chol = np.linalg.cholesky(params.correlation)
        # Cholesky for the CONDITIONAL covariance Z_s | Z_btc ~ N(c·Z_btc, R_ss - c·c^T)
        # used by S3.3 leverage coupling.
        c = params.btc_correlation
        cond_cov = params.correlation - np.outer(c, c)
        # PSD guaranteed by the joint-4x4 PSD check in __post_init__.
        self._chol_cond = np.linalg.cholesky(cond_cov)

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
        return_shocks: np.ndarray | None = None,
    ) -> dict:
        """Simulate correlated (a, b, ρ) paths.

        Two modes:
          * NO ``return_shocks`` (default, S3.2 mode): the three SVI shocks
            are correlated via the 3×3 ``correlation`` matrix only.
          * WITH ``return_shocks`` (S3.3 leverage mode): the three SVI shocks
            are sampled CONDITIONAL on the supplied BTC return shock:
              Z_s | Z_btc = c · Z_btc + L_cond · ε,    ε ~ N(0, I_3)
            where c = ``btc_correlation`` and L_cond·L_cond.T = R_ss − c·c.T.
            This realises the joint 4×4 (Z_btc, Z_a, Z_b, Z_rho) correlation
            structure WITHOUT modifying the supplied Z_btc (so the BTC price
            path stays whatever the price engine produced).

        Args:
            n_paths:              number of independent paths.
            n_steps:              number of evolution steps.
            dt:                   time-step size.
            a0, b0, rho0:         optional initial values (else each OU's μ).
            external_innovations: optional (n_paths, n_steps, 3) iid N(0,1).
                                  Used as the SVI iid block BEFORE the
                                  Cholesky / conditional transform.
            return_shocks:        optional (n_paths, n_steps) standardised
                                  BTC return shocks (the ``standardized_shock``
                                  field from PolitisRomanoKouEngine). Activates
                                  S3.3 leverage-effect coupling.

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

        if return_shocks is None:
            # S3.2 path: pure SVI-only correlation.
            correlated = iid @ self._chol.T
        else:
            z_btc = np.asarray(return_shocks, dtype=float)
            if z_btc.shape != (n_paths, n_steps):
                raise ValueError(
                    f"return_shocks shape {z_btc.shape} != "
                    f"expected {(n_paths, n_steps)}"
                )
            # Conditional sampling: Z_s = c·Z_btc + L_cond·ε
            c = self.params.btc_correlation
            conditional_mean = c[None, None, :] * z_btc[:, :, None]
            correlated = conditional_mean + iid @ self._chol_cond.T

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
