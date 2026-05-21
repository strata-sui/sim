"""Fetch + cache DeepBook Predict SVI history for trader-flow / SVI calibration.

Cache format: JSON list of SVI snapshots per oracle, saved to
`sim/data/svi_<oracle_id>.json` (gitignored).

S0 use: pull the live BTC oracle's SVI history so S3's stochastic-SVI engine
can later calibrate its OU process parameters against an empirical anchor
(per the spec in `../specs/trader_flow_spec.md` §plan-integration / CLAUDE.md §5).

Note: testnet utilization ≈ 9×10⁻⁴ means the history can be short / sparse.
That is exactly the situation the parameterized-adversarial-prior stance was
designed for — see disclosure in `../specs/trader_flow_spec.md`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from engine.predict_server import (
    TESTNET_PREDICT_ID,
    PredictServerClient,
    PredictServerError,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def cache_path(oracle_id: str, data_dir: Optional[Path] = None) -> Path:
    """Local cache path for an oracle's SVI history."""
    d = Path(data_dir) if data_dir else DATA_DIR
    return d / f"svi_{oracle_id}.json"


def find_btc_oracle(
    client: Optional[PredictServerClient] = None,
    predict_id: str = TESTNET_PREDICT_ID,
) -> Optional[dict]:
    """Return a BTC oracle dict from the predict's oracle list, or None.

    Preference order:
        1. ACTIVE BTC oracle (live, accepting mints).
        2. Any BTC oracle (settled / pending / inactive) as a fallback so
           we can still pull historical SVI snapshots.
    """
    client = client or PredictServerClient()
    oracles = client.predict_oracles(predict_id)
    btc = [
        o for o in oracles
        if "BTC" in str(o.get("underlying_asset", "")).upper()
    ]
    if not btc:
        return None
    active = [o for o in btc if str(o.get("status", "")).lower() == "active"]
    return active[0] if active else btc[0]


def fetch_and_cache_svi_history(
    oracle_id: str,
    data_dir: Optional[Path] = None,
    force: bool = False,
) -> list:
    """Fetch SVI history for an oracle and persist it locally as JSON.

    Args:
        oracle_id: on-chain oracle object id.
        data_dir:  override default cache directory (default `sim/data/`).
        force:     re-download even if cached file exists.

    Returns:
        The SVI history list (each entry typically contains
        `{a, b, rho, m, sigma, timestamp, ...}`).
    """
    d = Path(data_dir) if data_dir else DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = cache_path(oracle_id, d)
    if path.exists() and not force:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    client = PredictServerClient()
    history = client.oracle_svi_history(oracle_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    return history


def pull_btc_svi_summary(predict_id: str = TESTNET_PREDICT_ID) -> dict:
    """End-to-end convenience: locate BTC oracle, cache its SVI history,
    return a summary suitable for logging from the S0 orchestrator.

    Returns a dict with `oracle_id`, `oracle_status`, `expiry`,
    `history_len`, `cache_path`, and the `latest` SVI params (or None).
    On any network/decoding failure, raises `PredictServerError`.
    """
    client = PredictServerClient()
    oracle = find_btc_oracle(client, predict_id)
    if oracle is None:
        raise PredictServerError(
            f"No BTC oracle found for predict {predict_id}"
        )
    oid = oracle.get("id") or oracle.get("oracle_id") or oracle.get("object_id")
    if not oid:
        raise PredictServerError(
            f"BTC oracle entry has no id field: keys={list(oracle.keys())}"
        )
    history = fetch_and_cache_svi_history(oid)
    latest = history[-1] if history else None
    return {
        "oracle_id": oid,
        "oracle_status": oracle.get("status"),
        "expiry": oracle.get("expiry"),
        "history_len": len(history),
        "cache_path": str(cache_path(oid)),
        "latest": latest,
    }
