# Reproduce — `strata-sui/sim`

> Goal: a fresh clone of this repository, three commands, identical
> results to the committed Gate-B JSON within the documented
> sampling-noise band.

The simulator phase closes at git tag
[`v0.1.0-simulator-closed`](https://github.com/strata-sui/sim/releases/tag/v0.1.0-simulator-closed).
This document is the reproduction baseline against which the Move
contract numbers (next phase) and any independent audit must agree.

---

## Requirements

| Tool | Version | Why |
|---|---|---|
| Python | 3.13 | `pyproject.toml` `requires-python = ">=3.13"` |
| `uv` | 0.10.4 or later | Pinned dependency resolution + virtualenv (`uv.lock` committed) |
| `git` | 2.30+ | Standard |
| Disk space | ~ 200 MB | Binance Vision BTCUSDT 1-minute klines 2020-01 → 2026-04 (downloaded once, cached in `data/cache/`) |
| Memory | 4 GB free | Monte-Carlo sweep peak working set |
| Wall clock | ~ 10 min | S5 Gate-B sweep at 10,000 benign paths on a developer laptop |

Platform: tested on Windows 11 + PowerShell 5.1, Linux + bash, and
macOS + zsh. No platform-specific code paths.

---

## The three commands

```bash
# 1. Install pinned dependencies (uv.lock is committed for determinism)
uv sync

# 2. Run the full test suite — must show 357 passed
uv run pytest -q

# 3. Regenerate the Gate-B sweep + hero chart (~ 10 min wall clock)
uv run python s5_main.py
```

The third command writes:

| File | Bytes (approx) | Purpose |
|---|---|---|
| `data/s1_results/s5_gate_b.json` | ~ 50 KB | Full sweep result + verdict + components |
| `data/s1_results/s5_gate_b_hero.png` | ~ 100 KB | 1920 × 1080 web / README hero |
| `data/s1_results/s5_gate_b_hero_hi.png` | ~ 290 KB | 3840 × 2160 pitch-deck hero |

For the polished pitch-deck variant + vector SVG, additionally run:

```bash
uv run python scripts/s5_hero_chart_polish.py    # < 5 s, regenerates 3 sizes + .svg
```

For verdict-only re-derivation (sub-second; useful when only the
synthesis logic changed but the underlying sweep is still good):

```bash
uv run python scripts/s5_reconcile_verdict.py
```

For acceptance-criteria audit against the brief's 11 checks:

```bash
uv run python scripts/verify_s5_acceptance.py
```

---

## Random seed

The Monte-Carlo seed is **locked at `42`** at every entry point:

| Entry point | Locked at | Notes |
|---|---|---|
| `s5_main.py` | `PolitisRomanoKouEngine(..., seed=42)` (price engine), `seed=42 + i` (per-path in `run_strategy_s4`) | Disclosed in `s5_gate_b.json::config` |
| `scripts/diag_s5_blowup.py` | `seed=42`, `seed=42 + i` | Diagnostic only |
| `tests/` | `np.random.default_rng(<seed>)` per-test, deterministic | 357 tests |

The post-S5R3 accounting fix produces identical PnL bounds across
re-runs at the locked seed; the Monte-Carlo `mean / Sortino / p01`
metrics are reproducible to within the **sampling-noise band**
documented below.

---

## Expected sampling-noise band (10k MC)

The Gate-B JSON is regenerated stochastically (10,000 paths × 9 f × 3
strategies = 270,000 path-strategy evaluations). Even at the locked
seed, expect minor drift if NumPy / SciPy ever update their RNG
internals across a major release. The acceptance-criteria
`scripts/verify_s5_acceptance.py` script asserts the structural
invariants that must NOT drift:

- per-cell `|max_loss_pnl| ≤ strata_capital` (depositor-bounded loss,
  pinned at the source layer by `PLPVault.share_price` floor + the
  `max_exposure = 0.80` cap on per-step trader notional)
- per-cell `max_drawdown ∈ [-1, 0]`
- per-cell Sortino in plausible range (|S| ≤ 5)
- `n_paths_benign ≥ 10,000` AND `n_paths_crash ≥ 5,000`
- exactly ONE labelled verdict in the JSON (`gate_b.verdict_gate_b`);
  the `verdict.strict_baseline_pass` field is a boolean by design
  (per `s5_round3_fix_brief.md` S5R3.5-final, option (a))

Cell-level Sortino numbers may drift by ± 0.02–0.05 across NumPy
versions; the labelled verdict (`MARGINAL`) and the structural
findings (best `p01_reduction` positive across f, `f* = 0.05` across
`w_crash`, R3 delta ≈ $383k) are stable.

---

## Data acquisition

The price-engine input is **BTCUSDT 1-minute klines from Binance
Vision**, resampled to 15-minute bars inside the simulator. Raw data
is too large to track in git (~ 150 MB compressed, ~ 6.3 years of
1-minute bars). The loader at `engine/loader.py` downloads-and-caches
the missing months automatically on first invocation.

URL pattern:

```
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip
```

Example concrete URL (Black Thursday month):

```
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2020-03.zip
```

Caching layout (under `data/cache/`):

```
data/cache/BTCUSDT/1m/BTCUSDT-1m-2020-01.zip
data/cache/BTCUSDT/1m/BTCUSDT-1m-2020-02.zip
...
data/cache/BTCUSDT/1m/BTCUSDT-1m-2026-04.zip
```

(Cache files are git-ignored — see `.gitignore`.)

### Integrity check

Binance Vision publishes SHA-256 checksums alongside each kline zip
(at the same URL with `.CHECKSUM` appended). The loader does NOT
auto-verify these by default (network reliability over correctness);
if an auditor wants to verify, the canonical commands are:

```bash
# fetch checksum
curl -fsSL \
  https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2020-03.zip.CHECKSUM \
  -o BTCUSDT-1m-2020-03.zip.CHECKSUM

# verify
sha256sum -c BTCUSDT-1m-2020-03.zip.CHECKSUM
```

If a month's zip fails this check, the simulator's input is
non-canonical — re-download and re-run. Strata's S0 layer (the loader
+ sanity gate) caught one Binance Vision schema change historically
(ms → µs kline timestamp unit switch in 2025-01); the loader handles
both via magnitude-detection. Any future schema change should be
caught by the same mechanism.

---

## What the simulator does NOT use

- **No live network at sweep time.** All price data is local; SVI
  surface uses the deterministic anchor `ANCHOR_BTC_2026_05_15`
  (committed constants in `engine/svi_det.py`). The S3 stochastic
  SVI engine path is in-tree but fall-back-disabled at S5 by design
  (degenerate calibration on thin oracle history).
- **No proprietary data.** Binance Vision is public + free; the
  predict-server SVI history snapshot used during S3 calibration is
  also public via the `predict-server.testnet.mystenlabs.com` API.
- **No GPU.** Pure NumPy; CPU-only.

---

## Verification script

After regenerating, run the acceptance verifier to confirm all 11
brief acceptance criteria still hold:

```bash
uv run python scripts/verify_s5_acceptance.py
```

Expected tail of output:

```
[VERIFY OK] all acceptance criteria satisfied
```

Any other final line is a regression. Investigate before publishing
or citing the resulting JSON.

---

## Re-deriving the writeup figures

The README's "Three pillars" table cites:

| Field | Source location |
|---|---|
| `crash_protection.best_p01_reduction` | `data/s1_results/s5_gate_b.json` |
| R3 liquid-cash delta | `data/s1_results/s5_r3_verification.json` (regenerate via `uv run python s5_r3_verification.py`) |
| `f_star_summary` | `data/s1_results/s5_gate_b.json` |

If any pillar number in the writeup ever disagrees with the JSON, the
JSON is canonical — re-render the README + SUBMISSION accordingly.
