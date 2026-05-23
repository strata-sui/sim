"""S3.7 — Joint price + stochastic-SVI engine (drop-in for run_sweep).

Composes the price engine (PolitisRomanoKouEngine, S2) with the stochastic
SVI engine (S3.2 + S3.3 leverage coupling) and a per-step Gatheral g(k)
arb-clamp (S3.5). Returns the BootstrapEngine output schema **plus** a
``svi_path`` of per-path-per-step ``SVIParams`` so the strategy layer can
mark/price binaries against the actually-evolving surface.

The engine surfaces ``.sigma_historical`` so it remains a drop-in for
``eval.f_sweep.run_sweep`` (and therefore for ``s1.py``).
"""
from __future__ import annotations

import numpy as np

from engine.svi_det import SVIParams, clamp_to_arb_free
from engine.svi_stoch import StochasticSVIEngine


class JointStochasticEngine:
    """Wrap a price-path engine + a stochastic-SVI engine + the no-arb gate.

    The price engine must expose ``.simulate(n_paths, n_steps, init_price)``
    returning ``{price, log_return, standardized_shock}`` and a
    ``.sigma_historical`` property (any S1/S2 engine works).
    """

    def __init__(
        self,
        price_engine,
        svi_engine: StochasticSVIEngine,
        static_m: float | None = None,
        static_sigma: float | None = None,
        k_grid: np.ndarray | None = None,
    ) -> None:
        self.price_engine = price_engine
        self.svi_engine = svi_engine
        # m and sigma are static in S3 (per CLAUDE.md §5); default from the
        # SVI dynamics params if not overridden.
        self.static_m = (
            static_m if static_m is not None else svi_engine.params.m
        )
        self.static_sigma = (
            static_sigma if static_sigma is not None else svi_engine.params.sigma
        )
        self.k_grid = k_grid  # passed to clamp_to_arb_free

    @property
    def sigma_historical(self) -> float:
        return self.price_engine.sigma_historical

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        init_price: float = 60_000.0,
    ) -> dict:
        """Produce joint (price, svi) paths with leverage coupling + arb gate.

        Returns the price-engine output PLUS:
          ``svi_path``: list of length n_paths; each element is a list of
                        (n_steps+1) arb-clamped SVIParams.
        """
        price_out = self.price_engine.simulate(
            n_paths=n_paths, n_steps=n_steps, init_price=init_price
        )
        # Leverage coupling channel — feed the price-path standardized shocks
        # as the BTC-return channel for the conditional SVI sampler.
        svi_out = self.svi_engine.simulate(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=1.0,
            return_shocks=price_out["standardized_shock"],
        )
        a_path = svi_out["a"]      # (n_paths, n_steps+1)
        b_path = svi_out["b"]
        rho_path = svi_out["rho"]
        # Clip raw OU outputs to admissible SVIParams ranges before clamp.
        b_clip = np.maximum(b_path, 0.0)
        rho_clip = np.clip(rho_path, -0.999, 0.999)

        # Build per-step SVIParams + apply g(k) arb-clamp. Step-major loop
        # because clamp_to_arb_free is scalar (per surface); fine for the
        # MC sizes we run (paths × steps ≤ a few thousand).
        svi_path = [
            [
                clamp_to_arb_free(
                    SVIParams(
                        a=float(a_path[i, t]),
                        b=float(b_clip[i, t]),
                        rho=float(rho_clip[i, t]),
                        m=self.static_m,
                        sigma=self.static_sigma,
                    ),
                    k_grid=self.k_grid,
                )
                for t in range(n_steps + 1)
            ]
            for i in range(n_paths)
        ]
        return {
            **price_out,
            "svi_path": svi_path,
        }
