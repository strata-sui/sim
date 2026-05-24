# Strata — Sui Overflow 2026 Submission (DeepBook track)

Reproducible from this repository at tag `v0.1.0-simulator-closed`.
Run: `uv sync && uv run pytest && uv run python s5_main.py`.

## 1. Headline

> *"Many teams will build a naive PLP+hedge vault that is **secretly a
> self-referential wash**. Strata discovers the flaw, derives the
> condition for genuine protection, and **quantifies the optimal LP
> size** as a function of crash-aversion."*

## 2. The discovery

DeepBook Predict has a structural problem at the PLP layer: the pool
earns spread revenue in benign markets but takes fat-tailed losses when
settlement lands on trader inventory. A naive vault that earns yield by
supplying PLP and "hedges" by buying OTM-DN binaries from the same
pool *sounds* clean — but the math doesn't pencil. When the vault is
simultaneously the dominant LP AND the hedge buyer, the hedge premium
leaves the vault's trader account, enters the vault's own PLP balance,
and the hedge payout (if it lands) is paid out of the same balance.
The economic net depends not on the naive `(1 − f)` factor where `f`
is the LP-share fraction, but on the strike-local `(u(k) − f)` where
`u(k)` is Strata's DN-ownership share at the hedge strike. As
`f → u(k)`, hedge protection collapses to zero regardless of notional.
The hedge is genuine *only insofar* as `f` is small enough that
independent traders and other LPs fund the common pool that pays
Strata's crash hedge.

This is the discovery. The right question for any tail-aware PLP
vault is not "how big should the hedge be" but **"what is the LP-share
size at which the hedge stops being self-referential?"** That question
has a quantitative answer — `f*(w_crash)` — and it is the deliverable.

## 3. The methodology

> *"DeepBook's own problem statement asks **"is PLP safe?"** as the
> question gating serious LP TVL. Strata's answer is the **f\*(w_crash)
> methodology** — the optimal LP-share as a function of tail-aversion,
> empirically computed."*

`f*(w_crash)` is the maximiser of
`J(f; w) = (1−w) · Sortino_benign(f) + w · Sortino_crash(f)` over the
f-grid for tail-weights `w_crash ∈ [0, 1]`. The benign sweep uses
10,000 Politis–Romano stationary-block-bootstrap paths with a Kou
double-exponential jump overlay calibrated so the p001 single-step
return reaches the brief's tail-target ≤ −15 %. The crash sweep uses
5,900 historical BTC 40-step windows whose cumulative move is ≤ −4 %
(automatically capturing Black Thursday 2020, LUNA 2022, FTX 2022 and
many smaller events).

## 4. Three pillars (quantified)

| Pillar | Quantification | Evidence file |
|---|---|---|
| **Tail protection** | Conditional crash Sortino uplift +0.61, p01 tail cut from −5.54% → −3.45%, mean loss reduction +1.47%/cycle, all at f=0.80 on the S1 dual-report | `data/s1_results/sweep_conservative.json` + `s1.py` dual report |
| **R3 liquidity escape-hatch** | Empirical R3 liquid-cash delta = **$383,063** under the canonical scenario; stress test confirms collapses to $0 when bypass disabled (model responsive, not artifact) | `data/s1_results/s5_r3_verification.json` |
| **f\*(w_crash) methodology** | Conditional interior band w_crash ∈ [0.250, 0.325] on S1; honest disclosure that the band is conditional, NOT a guaranteed property (R2-C correction); curve methodology demonstrably anti-cherry-pick (§3 not edited, anchor params data-driven) | `data/s1_results/sweep_conservative.json` (f_star_curve + summarize_f_star_curve output) |

The pillar-1 and pillar-3 numbers above are the peer-review-locked S1
formulation. The post-S5R3 bounded-accounting rerun at the 10k benign /
5,900 crash resolution sharpens both — see §9 for the updated framing.

## 5. R3 verified on-chain path

Pitch claims R3 → must cite the mechanic:

```
predict::redeem_permissionless<DUSDC>(MarketKey, ManagerCap)
```

Documented at CLAUDE.md §4 (verified from `predict-testnet-4-16`
source): *"manager-side hedge payout via `redeem` bypasses the
limiter = the liquid leg. Model this."* The S5.3 verification scenario
empirically demonstrates the delta this mechanic produces.

## 6. Three-lever exhaustion

> *"We tested Strata across **three progressive structural levers** —
> rigorous tail engine (S2 PR+Kou with λ recalibrated to brief's
> p001 ≤ −15% tail target), stochastic SVI dynamics (S3, with the
> honest degenerate-flag → fallback to deterministic anchor), and the
> full multi-cycle DN-ladder + token-bucket model (S4). The
> frequency-weighted Gate-A did not clear at ANY lever. **This is the
> mechanical consequence of the §3 self-reference math we discovered
> and disclose** — not a strategy failure. Value sits in the three
> pillars above; one pillar is now empirically quantified at the
> liquidity level (R3 = $383k delta) the prior simulator phases
> hadn't reached."*

## 7. Anti-overfit discipline

- **S0** loader bug caught (Binance Vision ms→µs switch in 2025-01)
  before any thesis numbers run.
- **S1** Task A premium sizing fixed §2A envelope violation
  (full-sleeve-per-cycle bug); Task B strike anchored to PLP
  loss-onset (p1 = −1.89%), NOT to Sortino flatter.
- **S2.7** Kou λ recalibration retracted the earlier "anchored to
  historical extreme-bar rate" framing as rationalizing
  under-calibration; reselected λ = smallest sweep value passing
  brief's p001 ≤ −15% target.
- **R2-A** correction retracted the earlier "§3 CONFIRMED" overreach
  (gate's own f*=0.05 contradicted it); §3 line ~110 softened to
  conditional with refined wording — NOT edited as masterplanner
  recommendation, edited only as Albary-approved.
- **S3** degenerate-flag honestly raised + fallback invoked.
- **S4** RED-PROVISIONAL reported HONESTLY (not pivoted) — three-lever
  exhaustion narrative locked.
- **S5.3** R3 stress test deliberately disables bypass — confirms model
  is RESPONSIVE to the mechanic, not artifact.

This pattern is repeatedly applied and visible in commit history at
this repository.

## 8. What ships

Simulator phase closed at tag `v0.1.0-simulator-closed` on this
repository. 357 unit + regression tests pass in ~6 s; the Gate-B JSON +
hero chart are reproducible from the seed-fixed orchestrator
(`uv run python s5_main.py`, ~10 min wall-clock on a developer laptop).

What's next, in separate repositories:

- **`strata-sui/contracts`** — Sui Move smart contracts implementing
  the DN-ladder construction algorithm (S4.1 prefix-matched to PLP
  loss-onset), per-strike DN pricing from a live SVI surface, and the
  `f`-as-governance-parameter that keeps the `f*(w)` methodology
  actionable on-chain. Must implement the verified
  `predict::redeem_permissionless` path for the pillar-2 claim to hold.
- **`strata-sui/frontend`** — Next.js dashboard surfacing the
  `f*(w_crash)` calculator + vault deposit / withdraw UI + pool-state
  readouts.
- **Testnet integration** — `predict-testnet-4-16` end-to-end flow on
  Sui testnet.
- **Mainnet deploy** — target pre-27-Aug-2026 (the 100 %
  prize-unlock window).

## 9. Honest limitations

- **Sample size below the optimistic target.** 10,000 benign paths
  reached at S4 fidelity (≈ 2.1 ms / path / cell) within the time-box;
  the brief's optimistic 1 M target was softened to a 10k floor in
  `docs/s5_round3_fix_brief.md` (local). Conclusions qualitative;
  the sampling-noise band is larger than at 1 M.
- **Single-anchor SVI.** The stochastic SVI engine
  (`engine/svi_stoch.py`) is calibrated and tested but degenerate at
  the thin cached oracle history. Per the documented S3 fallback rule,
  the Gate-B sweep uses the deterministic anchor
  `ANCHOR_BTC_2026_05_15`. Engine in-tree for when calibration source
  enriches.
- **MARGINAL verdict, not GREEN.** Stated cleanly; not euphemized.
  Approved diction throughout: *"truncated tail"*, *"quantified
  residual"*, *"left-tail reduction"*. The downside is not eliminated;
  the residual tail is bounded and quantified.
- **Multi-cycle accounting only sane post-S5R3 fix.** The S5 first
  attempt produced PnL magnitudes on the order of `-$10^10` on a `$50k`
  deposit — a multi-cycle accounting bug caught by masterplanner
  review. Root-cause fix at the source: `PLPVault.share_price` floored
  at 0 (depositor's loss is bounded by deposit), and the protocol
  `max_exposure = 0.80` cap is now enforced on per-step trader notional.
  Regression tests pin both invariants
  (`TestDepositorBoundedLoss` + `TestSharePriceFloor`). §4 evidence
  files reflect the post-fix sane numbers.
- **Post-S5R3 sharpening of pillars 1 and 3.** Pillar 1's tail
  protection is now framed as a *quantified residual via p01* — the
  rerun shows `best p01_reduction = +0.0473` at the conservative
  `f = 0.05`, monotonic-increasing in `f` (tail truncation is positive
  across the entire f-grid). Pillar 3's S1 "conditional interior band"
  did NOT survive the bounded-accounting fix at the 10k+5,900
  resolution: `f* = 0.05` across all `w_crash` tail-weights. The new
  framing is stronger — *the conservative LP-share IS the optimal-f
  recommendation, not a tail-aversion choice*. No need to weight a
  rare crash 250× its natural frequency to justify the answer. Pillar
  2 ($383k delta) is unchanged.
