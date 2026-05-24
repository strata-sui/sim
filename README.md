# Strata — Simulator

> *A liquidity vault on DeepBook Predict that earns yield AND hedges its own downside.*
> *Plus: a built-in calculator that quantifies your safe deposit size — the question DeepBook itself flags as gating serious LP participation.*

Strata is a structured vault built for **Sui Overflow 2026, DeepBook
track**. This repository is the simulator that validates the strategy
end-to-end and produces the qualification artifact required by the
track ("proper simulation result if you are building a vault strategy").

The Move contracts, frontend, and testnet integration live in separate
repositories ([`strata-sui/contracts`](https://github.com/strata-sui/contracts),
[`strata-sui/frontend`](https://github.com/strata-sui/frontend)) — the
simulator phase is closed here at tag `v0.1.0-simulator-closed`.

---

## Why this exists

DeepBook Predict's protocol-stated open question is *"is PLP safe?"* —
the gating question for serious LP capital (treasuries, DAOs, cautious
yield aggregators). Without a defensible answer, PLP stays shallow and
the prediction market stays small. Strata's contribution is a
methodology — the **optimal LP-share f\*** as a function of crash
aversion — backed by a simulator and an on-chain liquidity escape-hatch
that prior naive PLP-plus-hedge designs miss.

---

## What's inside

```
sim/
├── engine/                  # price + volatility surface engines
│   ├── loader.py            # Binance Vision 1m klines → DataFrame
│   ├── resample.py          # 1m → step_minutes bar resampler
│   ├── price_engine.py      # baseline block-bootstrap engine
│   ├── price_engine_pr_kou.py  # Politis–Romano bootstrap + Kou jumps
│   ├── svi_det.py           # Gatheral SVI surface + no-arb gate
│   ├── svi_stoch.py         # OU-process stochastic SVI dynamics
│   ├── svi_calibration.py   # method-of-moments OU calibration
│   ├── joint_engine.py      # PR+Kou × stochastic SVI joint sampler
│   └── spread.py            # verified DeepBook Predict spread formula
├── model/
│   ├── plp.py               # PLP-vault accounting (verified semantics)
│   ├── hedge.py             # OTM-DN binary hedge position
│   ├── dn_ladder.py         # multi-strike DN ladder (PLP loss-onset)
│   ├── token_bucket.py      # withdraw limiter
│   └── trader_flow.py       # 6-knob exogenous trader-flow model
├── eval/
│   ├── account.py           # 3-account state + step_path + settle_path
│   ├── strategy.py          # RAW_PLP / FIXED_HEDGE_NAIVE / STRATA runners
│   ├── f_sweep.py           # f-grid Monte Carlo sweep
│   ├── scenarios.py         # historical crash-window replay
│   ├── gate_a.py            # Gate-A verdict + crash protection summary
│   └── objective.py         # tail-weighted J(f;w) + f*(w_crash) curve
├── s1.py – s5_main.py       # phase orchestrators (S1 thin slice → S5 Gate B)
├── scripts/                 # diagnostic + reconcile helpers
├── tests/                   # 357 unit + regression tests
└── data/s1_results/         # local-only Gate B output (regenerable)
```

The simulator phase plan is documented at the project's local
`docs/` (not pushed) and was executed in time-boxed phases S0 → S5
plus the S5R3 fix pass.

---

## Quickstart

Requirements: Python 3.13, [`uv`](https://docs.astral.sh/uv/) 0.10+.

```bash
# install pinned dependencies
uv sync

# run the full test suite (357 tests should pass)
uv run pytest -q

# regenerate Gate-B Monte Carlo + hero chart (~10 minutes wall clock)
uv run python s5_main.py
# outputs:  data/s1_results/s5_gate_b.json
#           data/s1_results/s5_gate_b_hero.png
#           data/s1_results/s5_gate_b_hero_hi.png

# optional: re-derive verdict only (sub-second) from cached results
uv run python scripts/s5_reconcile_verdict.py

# optional: verify the 11 brief acceptance criteria on the JSON
uv run python scripts/verify_s5_acceptance.py
```

The fixed seed is `42` at the simulator entry point; the resulting
metrics are reproducible within the sampling-noise band documented in
`REPRODUCE.md`.

---

## Result summary

**Gate-B verdict: MARGINAL** (honest, post-accounting-fix, 10,000
benign Monte-Carlo paths + 5,900 historical crash windows; raw +4 %
moves and worse, since 2020).

A MARGINAL verdict is the deliberate, non-euphemized outcome of a
quantified discovery: the §3 self-reference mathematics show that when
the vault is simultaneously the dominant liquidity provider AND the
hedge buyer, the hedge premium and payout cycle inside its own balance
sheet. The aggregate Sortino margin cannot clear at the conservative
trader-flow anchor regardless of which structural engine lever is
swapped in (this is the empirical "three-lever exhaustion" result of
the S2/S3/S4 progression). Two value pillars still hold under the
correct framing — they are the basis of the submission.

### Three pillars (quantified, with evidence files)

| Pillar | Quantification | Evidence file |
|---|---|---|
| **Tail truncation** | Conditional-crash p01 tail reduction = **+0.0473** (monotonic-increasing in `f`); best `f` = 0.05. The hedge ladder truncates the left tail by design; quantified residual published per `gate_a.crash_protection_summary`. | `data/s1_results/s5_gate_b.json` (`crash_protection.best_p01_reduction`) |
| **R3 liquidity escape-hatch** | Empirical liquid-cash delta = **$383,063** under the canonical scenario (`available_for_withdraw = 0` pre-settle, hedge ladder payoff bypasses the limiter via the manager-side redeem path). Stress test (bypass disabled) collapses Strata to raw-PLP outcome — the model is responsive to the on-chain mechanic, not artefactual. | `data/s1_results/s5_r3_verification.json` |
| **f\*(w_crash) methodology** | Frequency-weighted optimal `f*` = 0.05 across all crash-weight tail-aversions at the 10k+5,900 resolution. The conservative LP-share IS the optimal-`f` recommendation, NOT a tail-aversion choice — a stronger and more defensible claim than the earlier conditional interior result. | `data/s1_results/s5_gate_b.json` (`f_star_summary`) |

The on-chain mechanic for pillar 2 is the verified
`predict::redeem_permissionless<DUSDC>(MarketKey, ManagerCap)` path
documented in the DeepBook Predict source (`predict-testnet-4-16`
branch).

---

## Honest disclosures

- **Sample size.** 10,000 benign Monte-Carlo paths and 5,900 historical
  ≥ 4 % 40-step crash windows. The brief's optimistic target was 1 M
  benign paths; 10k was reached at S4 fidelity (≈ 2.1 ms / path / cell)
  within the time-box. Sampling-noise band larger than at 1 M;
  conclusions qualitative. Disclosed verbatim in
  `s5_gate_b.json::config.fallback_disclosure`.
- **Single-anchor SVI surface.** The stochastic-SVI engine
  (`engine/svi_stoch.py`) is calibrated but degenerate at the thin
  cached oracle history; per the documented `S3 fallback rule`, the
  Gate-B sweep uses the deterministic anchor
  `ANCHOR_BTC_2026_05_15`. The stochastic engine is kept in-tree for
  when the calibration source enriches.
- **Three-lever exhaustion.** Across the three progressive structural
  engine refinements (S2 Politis–Romano + Kou tail engine recalibrated
  to brief's p001 ≤ −15 %, S3 stochastic SVI with degenerate-flag +
  fallback, S4 full multi-cycle DN-ladder + token-bucket), the
  frequency-weighted Gate-A aggregate did not clear at any lever.
  This is the mechanical consequence of the §3 self-reference math
  the project discovered and discloses — not a strategy failure.
- **MARGINAL verdict, not GREEN.** The verdict is stated cleanly and
  not euphemized. Approved diction throughout: "truncated tail",
  "quantified residual", "left-tail reduction". The downside is *not*
  eliminated; the residual tail is bounded and quantified.
- **Post-S5R3 accounting fix.** The multi-cycle accounting required a
  source-layer fix (depositor-bounded loss invariant: `share_price`
  floored at 0, plus the protocol `max_exposure = 0.80` cap enforced
  on per-step trader notional). The brief acceptance criteria are
  pinned by `tests/test_strategy.py::TestDepositorBoundedLoss` +
  `tests/test_plp.py::TestSharePriceFloor` regression tests, and the
  R3 + writeup-hooks artefacts are regression-protected by
  `tests/test_s5_regression_protect.py`.

---

## Tech

- Python 3.13 with `uv` for project management.
- NumPy / pandas / SciPy / matplotlib for the Monte-Carlo engine.
- 357 unit + regression tests, run in `< 6 s` on a developer laptop.

## Related repositories

- [`strata-sui/contracts`](https://github.com/strata-sui/contracts) —
  Sui Move smart contracts (Move phase, post-simulator).
- [`strata-sui/frontend`](https://github.com/strata-sui/frontend) —
  Next.js dashboard (frontend phase).

## License

MIT. See [`LICENSE`](./LICENSE).

## Citations

- **DeepBook Predict** — protocol brief +
  [`MystenLabs/deepbookv3`](https://github.com/MystenLabs/deepbookv3)
  branch `predict-testnet-4-16`, package `packages/predict/`. All
  contract mechanics cited in source are verified directly.
- **Politis & Romano (1994)** — "The Stationary Bootstrap",
  *Journal of the American Statistical Association* 89:1303–1313. Used
  by `engine/price_engine_pr_kou.py` for stationary block bootstrap.
- **Kou (2002)** — "A jump-diffusion model for option pricing",
  *Management Science* 48(8):1086–1101. Used for the double-exponential
  jump overlay in the price engine.
- **Gatheral (2004)** — "A parsimonious arbitrage-free implied
  volatility parameterization". Used by `engine/svi_det.py` for SVI
  surface pricing of binary DN options.
- **Gatheral & Jacquier (2014)** — "Arbitrage-free SVI volatility
  surfaces", *Quantitative Finance* 14(1):59–71. Used for the
  butterfly + calendar no-arbitrage gate per simulated step in
  `engine/svi_stoch.py`.
- **Aït-Sahalia, method-of-moments OU calibration** — used by
  `engine/svi_calibration.py`.
