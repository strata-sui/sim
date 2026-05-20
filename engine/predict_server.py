"""DeepBook Predict testnet `predict-server` HTTP client.

Public read-only API at `predict-server.testnet.mystenlabs.com`.

Endpoints used here (verified live 2026-05-15):
    GET /status
    GET /predicts/:id/oracles
    GET /oracles/:id/state
    GET /oracles/:id/svi          (full history)
    GET /oracles/:id/svi/latest
    GET /predicts/:id/vault/summary
    GET /predicts/:id/vault/performance

Usage:
    client = PredictServerClient()
    print(client.status())
    oracles = client.predict_oracles()
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

DEFAULT_BASE = "https://predict-server.testnet.mystenlabs.com"
DEFAULT_TIMEOUT = 15  # seconds

# Known testnet predict object id (per research, verified 2026-05-15).
TESTNET_PREDICT_ID = (
    "0xc8736204d12f0a7277c86388a68bf8a194b0a14c5538ad13f22cbd8e2a38028a"
)


class PredictServerError(RuntimeError):
    """Raised when predict-server returns an error or is unreachable."""


class PredictServerClient:
    """Minimal HTTP client over urllib for the testnet predict-server."""

    def __init__(
        self,
        base: str = DEFAULT_BASE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    # ---- internal -------------------------------------------------------
    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            raise PredictServerError(
                f"HTTP {e.code} for {url}: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise PredictServerError(f"Network error for {url}: {e.reason}") from e
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise PredictServerError(f"Invalid JSON from {url}: {e}") from e

    # ---- public endpoints ----------------------------------------------
    def status(self) -> dict:
        """Pipeline health (`max_time_lag_seconds`, etc.)."""
        return self._get("/status")

    def predict_oracles(self, predict_id: str = TESTNET_PREDICT_ID) -> list:
        """List of oracles for a predict object."""
        return self._get(f"/predicts/{predict_id}/oracles")

    def oracle_state(self, oracle_id: str) -> dict:
        """Current oracle state: spot/forward, latest SVI params, status."""
        return self._get(f"/oracles/{oracle_id}/state")

    def oracle_svi_history(self, oracle_id: str) -> list:
        """Full SVI update history for an oracle."""
        return self._get(f"/oracles/{oracle_id}/svi")

    def oracle_svi_latest(self, oracle_id: str) -> dict:
        """Most recent SVI update for an oracle."""
        return self._get(f"/oracles/{oracle_id}/svi/latest")

    def predict_vault_summary(
        self, predict_id: str = TESTNET_PREDICT_ID
    ) -> dict:
        """Vault aggregate state (NAV, MTM, utilization, withdraw availability)."""
        return self._get(f"/predicts/{predict_id}/vault/summary")

    def predict_vault_performance(
        self,
        predict_id: str = TESTNET_PREDICT_ID,
        range_: Optional[str] = None,
    ) -> dict:
        """PLP share-price time series. range_ ∈ {'1H','1D','1W','ALL', ...}."""
        params = {"range": range_} if range_ else None
        return self._get(
            f"/predicts/{predict_id}/vault/performance", params=params
        )
