"""Named-scenario crash replay — the decisive Gate-A tail test (S1 Task C).

The benign bootstrap under-samples the crash tail, but the hedge is
DESIGNED for the tail. To assess protection honestly we run the f-sweep on
the empirical conditional-on-crash path distribution: every contiguous
``n_steps`` window of real 15m history whose cumulative move is <=
threshold. This is data-driven and date-agnostic — it automatically
captures Black Thursday (Mar-2020), LUNA (May-2022), and FTX (Nov-2022),
all verified present in the S0 data, without hand-picking dates (which
would invite a cherry-pick objection).

``ScenarioReplay`` is a drop-in for ``BootstrapEngine`` in
``eval.f_sweep.run_sweep`` (same ``.simulate()`` signature +
``.sigma_historical`` property), so the crash sweep reuses the exact same
strategy/accounting pipeline as the benign sweep — only the path
distribution differs.
"""
from __future__ import annotations

import numpy as np


def find_crash_windows(
    log_returns: np.ndarray,
    n_steps: int,
    threshold: float = -0.04,
) -> np.ndarray:
    """All contiguous ``n_steps`` windows with cumulative move <= ``threshold``.

    Args:
        log_returns: 1-D historical log-return series.
        n_steps:     window length (= path length for the sweep).
        threshold:   simple-return cutoff (e.g. -0.04 = a >=4% drop over the
                     window). Default -4% sits at the empirical 1h p0.1 tail.

    Returns:
        ``(n_windows, n_steps)`` array of qualifying windows. Windows overlap
        (a sustained crash like Black Thursday contributes many) — this is
        the honest conditional-on-crash distribution, not an i.i.d. sample.
    """
    r = np.asarray(log_returns, dtype=float)
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if len(r) < n_steps:
        return np.empty((0, n_steps), dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(r, n_steps)
    cum_log = windows.sum(axis=1)
    log_thr = np.log1p(threshold)
    return np.ascontiguousarray(windows[cum_log <= log_thr])


class ScenarioReplay:
    """Drop-in for ``BootstrapEngine`` that replays crash windows as paths."""

    def __init__(self, crash_windows: np.ndarray, sigma_historical: float) -> None:
        self._w = np.asarray(crash_windows, dtype=float)
        if self._w.ndim != 2:
            raise ValueError("crash_windows must be 2-D (n_windows, n_steps)")
        self._sigma = float(sigma_historical)
        self.n_windows = int(self._w.shape[0])

    @property
    def sigma_historical(self) -> float:
        return self._sigma

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        init_price: float = 60000.0,
    ) -> dict:
        """Return crash paths in the BootstrapEngine output schema.

        If ``n_paths`` exceeds the number of crash windows, windows are
        resampled with replacement (seeded) so the sweep can request a fixed
        path count; if fewer, the first ``n_paths`` are used.
        """
        if self.n_windows == 0:
            raise ValueError("no crash windows found for the given threshold")
        if self._w.shape[1] != n_steps:
            raise ValueError(
                f"window length {self._w.shape[1]} != n_steps {n_steps}"
            )
        if n_paths == self.n_windows:
            sel = self._w
        elif n_paths < self.n_windows:
            sel = self._w[:n_paths]
        else:
            rng = np.random.default_rng(0)
            sel = self._w[rng.integers(0, self.n_windows, size=n_paths)]

        log_ret = sel
        std_shock = log_ret / self._sigma
        log_cumsum = np.cumsum(log_ret, axis=1)
        price = np.empty((log_ret.shape[0], n_steps + 1), dtype=float)
        price[:, 0] = init_price
        price[:, 1:] = init_price * np.exp(log_cumsum)
        return {
            "log_return": log_ret,
            "standardized_shock": std_shock,
            "price": price,
        }
