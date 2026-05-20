# Strata Sim

NumPy-based Monte Carlo simulator for the **Strata** vault strategy on DeepBook Predict.

Strata is a structured vault built for Sui Overflow 2026 (DeepBook track). This repo contains the simulator that validates the strategy and produces the qualification artifact required by the track ("proper simulation result for a vault strategy").

## Status

Work in progress. Pre-Gate-A.

## Related repos

- [`strata-sui/contracts`](https://github.com/strata-sui/contracts) — Sui Move smart contracts
- [`strata-sui/frontend`](https://github.com/strata-sui/frontend) — Next.js dashboard

## Tech

- Python 3.13, [uv](https://docs.astral.sh/uv/) for project management
- NumPy / pandas / SciPy / matplotlib

## License

MIT. See [LICENSE](./LICENSE).
