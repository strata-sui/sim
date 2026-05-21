"""Coarse f-sweep — S1 Gate-A primary artifact (CLAUDE.md §6 / §7).

For each (strategy, f) combo, runs N Monte-Carlo paths through the
``BootstrapEngine`` → step-evolution → ``settle_path`` pipeline and
aggregates Strata's per-cycle P&L into summary metrics. The f-Curve
(Sortino vs f) per strategy is the visual proof primitive — at S1 it is
COARSE (≤ 10 f-points × ≤ 10k paths). The 1M-path production sweep ships
at S5.

S1 simplifications (vs S4+):
    * Single-cycle: one ``open_hedge`` at path start, one ``settle_path``
      at path end. Multi-cycle rolling at S4.
    * Constant realized vol fallback (windowed rolling with sigma-historical
      pre-fill). The informed-bias signal (Knob 3) still varies per step
      via ``log_return_recent``.
    * Deterministic SVI surface across the whole sweep. Stochastic SVI at S3.

Output schema:
    results[strategy_name][f] = {
        "n_paths", "mean_return", "std", "sortino", "downside_std",
        "p01_return", "p05_return", "prob_loss", "min_return",
        "max_loss_pnl", "mean_pnl", "strata_capital",
    }
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from engine.price_engine import BootstrapEngine
from engine.svi_det import SVIParams
from eval.strategy import ALL_STRATEGIES, StrategyConfig, run_strategy
from model.trader_flow import TraderFlowAnchor


def compute_metrics(pnls: np.ndarray, capital: float) -> dict:
    """Cycle-return summary from a P&L sample.

    Returns are computed as ``pnls / capital`` (single-cycle return). Sortino
    is ``mean / downside_std`` with ``downside_std`` over realized losses
    only. ``inf`` if mean > 0 with no downside; ``0`` if mean ≤ 0 with no
    downside.
    """
    if len(pnls) == 0:
        raise ValueError("compute_metrics requires non-empty pnls")
    if capital <= 0:
        raise ValueError(f"capital must be > 0, got {capital}")
    returns = pnls / capital
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    downside = returns[returns < 0.0]
    if len(downside) > 1:
        downside_std = float(np.std(downside, ddof=1))
    elif len(downside) == 1:
        downside_std = float(abs(downside[0]))
    else:
        downside_std = 0.0
    if downside_std > 0:
        sortino = mean / downside_std
    elif mean > 0:
        sortino = float("inf")
    else:
        sortino = 0.0
    return {
        "n_paths": int(len(returns)),
        "mean_return": mean,
        "std": std,
        "sortino": sortino,
        "downside_std": downside_std,
        "p01_return": float(np.percentile(returns, 1)),
        "p05_return": float(np.percentile(returns, 5)),
        "prob_loss": float(np.mean(returns < 0.0)),
        "min_return": float(np.min(returns)),
        "max_loss_pnl": float(np.min(pnls)),
        "mean_pnl": float(np.mean(pnls)),
        "strata_capital": float(capital),
    }


def _rolling_vol(
    log_returns: np.ndarray,
    window: int,
    fallback: float,
) -> np.ndarray:
    """Rolling stdev along axis 1 with ``fallback`` for steps < window."""
    n_paths, n_steps = log_returns.shape
    out = np.full_like(log_returns, fallback)
    if window < 2:
        return out
    for t in range(window - 1, n_steps):
        out[:, t] = np.std(
            log_returns[:, t - window + 1 : t + 1], axis=1, ddof=1
        )
    return out


def run_sweep(
    bootstrap: BootstrapEngine,
    svi_params: SVIParams,
    anchor: TraderFlowAnchor,
    total_pool_capital: float,
    f_grid: Sequence[float],
    n_paths: int,
    path_steps: int,
    init_price: float = 60000.0,
    strategies: Sequence[StrategyConfig] = ALL_STRATEGIES,
    realized_vol_window: int = 8,
    progress: bool = False,
) -> dict:
    """Run the coarse f-sweep.

    For each ``f`` in ``f_grid``:
        ``strata_capital   = f · total_pool_capital``
        ``other_lp_initial = (1 − f) · total_pool_capital``

    For each strategy × path: ``run_strategy`` collects ``strata_pnl_total``.
    Metrics aggregated per (strategy, f).

    Args:
        bootstrap:           seeded ``BootstrapEngine`` over empirical log returns.
        svi_params:          deterministic SVI surface (S1 constant).
        anchor:              trader-flow band (Conservative for Gate-A primary).
        total_pool_capital:  size of the pool — Strata + other-LP sum.
        f_grid:              monotone-increasing fractions in (0, 1).
        n_paths:             paths per (strategy, f). S1 default ~10k.
        path_steps:          steps per path. 1 step = 1 ``step_minutes`` bar.
        init_price:          starting forward at step 0.
        strategies:          which strategies to evaluate.
        realized_vol_window: rolling-window length for realized vol (S1).
        progress:            print per-(strategy, f) progress lines.

    Returns:
        nested dict ``{strategy_name: {f: metrics_dict}}``.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if any(not (0.0 < f < 1.0) for f in f_grid):
        raise ValueError(f"f_grid entries must lie in (0, 1): {f_grid}")
    if total_pool_capital <= 0:
        raise ValueError(
            f"total_pool_capital must be > 0, got {total_pool_capital}"
        )

    # Vectorized path generation: one bootstrap call covers all paths.
    sim = bootstrap.simulate(
        n_paths=n_paths,
        n_steps=path_steps,
        init_price=init_price,
    )
    price_paths = sim["price"]
    log_paths = sim["log_return"]
    sigma_long_run = bootstrap.sigma_historical

    realized_vols = _rolling_vol(
        log_paths, window=realized_vol_window, fallback=sigma_long_run
    )

    results: dict = {s.name: {} for s in strategies}
    for f in f_grid:
        strata_capital = f * total_pool_capital
        other_lp = (1.0 - f) * total_pool_capital
        for strat in strategies:
            pnls = np.empty(n_paths, dtype=float)
            for i in range(n_paths):
                out = run_strategy(
                    strategy=strat,
                    strata_capital=strata_capital,
                    other_lp_initial=other_lp,
                    price_path=price_paths[i],
                    realized_vols=realized_vols[i],
                    sigma_long_run=sigma_long_run,
                    log_returns=log_paths[i],
                    svi_params=svi_params,
                    anchor=anchor,
                )
                pnls[i] = out["strata_pnl_total"]
            metrics = compute_metrics(pnls, capital=strata_capital)
            results[strat.name][f] = metrics
            if progress:
                print(
                    f"  f={f:.2f} {strat.name:20s}  "
                    f"sortino={metrics['sortino']:>7.3f}  "
                    f"mean={metrics['mean_return']:>7.4f}  "
                    f"p01={metrics['p01_return']:>7.4f}  "
                    f"prob_loss={metrics['prob_loss']:>5.2f}"
                )
    return results
