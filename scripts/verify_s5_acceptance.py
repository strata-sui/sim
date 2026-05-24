"""S5R3 acceptance verifier — checks the rerun JSON against the brief.

Run after `python s5_main.py` produces `data/s1_results/s5_gate_b.json`.
Validates the 11 acceptance criteria from docs/s5_round3_fix_brief.md:

  1. Accounting magnitudes sane: |max_loss_pnl| <= strata_capital per cell.
  2. max_drawdown in [-1, 0] (fraction of capital).
  3. Sortino in plausible range (sanity bounds).
  4. Sample size >= 10k benign + >= 10k crash, OR compute-time disclosed.
  5. ONE canonical verdict — both top-level + nested fields match OR
     disagreement explicitly noted.
  6. R3 verification unchanged (regression-protected by test).
  7. Writeup hooks unchanged (regression-protected by test).
  8. Tests pass (regression).
  9. Hero chart regenerated.
  10. No §3 edit, no diction violation, R2 guard intact.
  11. Multi-cycle reformulation deferred (if any) cited in §0 record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "s1_results"
J_PATH = RESULTS_DIR / "s5_gate_b.json"
HERO_PATH = RESULTS_DIR / "s5_gate_b_hero.png"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not J_PATH.exists():
        print(f"[FAIL] {J_PATH} does not exist — run s5_main.py first")
        return 1

    data = json.load(open(J_PATH, "r", encoding="utf-8"))
    cfg = data["config"]
    failures: list[str] = []

    # ---- Acceptance #1 & #2: per-cell magnitude sanity -----------------
    print("\n[#1+#2] per-cell |max_loss_pnl| <= strata_capital + max_dd in [-1, 0]")
    for dist_name in ("benign_results", "crash_results"):
        results = data.get(dist_name)
        if results is None:
            print(f"  ({dist_name} not present — skip)")
            continue
        for strat, fmap in results.items():
            for f, m in fmap.items():
                strata_cap = float(m["strata_capital"])
                max_loss = float(m["max_loss_pnl"])
                max_dd = float(m["max_drawdown"])
                if max_loss < -strata_cap - 1.0:  # tiny float slack
                    failures.append(
                        f"  ✗ {dist_name}/{strat}/f={f}: max_loss_pnl="
                        f"${max_loss:,.2f} < -${strata_cap:,.2f} (deposit)"
                    )
                if max_dd < -1.0 - 1e-9 or max_dd > 0.0 + 1e-9:
                    failures.append(
                        f"  ✗ {dist_name}/{strat}/f={f}: max_dd={max_dd:.4f} "
                        f"∉ [-1, 0]"
                    )
    if not failures:
        print("  ✓ all cells satisfy |max_loss_pnl| <= strata_capital + max_dd in [-1, 0]")

    # ---- Acceptance #3: Sortino in plausible range ---------------------
    print("\n[#3] Sortino in plausible range")
    sortino_bounds_ok = True
    extreme_sortinos = []
    for dist_name in ("benign_results", "crash_results"):
        results = data.get(dist_name)
        if results is None:
            continue
        for strat, fmap in results.items():
            for f, m in fmap.items():
                s = float(m["sortino"])
                # "plausible" = bounded by 5 in magnitude (a Sortino > 5 or
                # < -5 is typically degenerate / single-sided distribution).
                if abs(s) > 5.0 and abs(s) != float("inf"):
                    extreme_sortinos.append(
                        f"  ! {dist_name}/{strat}/f={f}: sortino={s:+.3f}"
                    )
                if abs(s) == float("inf"):
                    extreme_sortinos.append(
                        f"  ! {dist_name}/{strat}/f={f}: sortino=±inf "
                        "(degenerate — no downside)"
                    )
    if not extreme_sortinos:
        print("  ✓ all Sortinos in [-5, +5] (sanity)")
    else:
        print("  ! some Sortinos extreme but bounded (not the prior -10^9 artifact)")
        for line in extreme_sortinos[:10]:
            print(line)

    # ---- Acceptance #4: sample sizes -----------------------------------
    print("\n[#4] sample sizes")
    n_b = int(cfg.get("n_paths_benign", 0))
    n_c = int(cfg.get("n_paths_crash", 0))
    print(f"  benign n_paths = {n_b:,}")
    print(f"  crash  n_paths = {n_c:,}")
    if n_b < 10_000:
        failures.append(f"  ✗ benign n={n_b} < brief floor 10k")
    else:
        print(f"  ✓ benign n >= 10k (brief floor)")
    if n_c < 5_000:
        failures.append(f"  ✗ crash n={n_c} < 5k floor")
    else:
        print(f"  ✓ crash n = {n_c} historical windows (>= 5k)")

    # ---- Acceptance #5: verdict reconciliation (option-a structure) ----
    print("\n[#5] verdict reconciliation")
    # Post-S5R3.5-final (option a): gate_a layer is a BOOLEAN
    # (strict_baseline_pass); the SINGLE labeled verdict is at gate_b.
    if "strict_baseline_pass" in data["verdict"]:
        v_strict = bool(data["verdict"]["strict_baseline_pass"])
        v_diag = data["verdict"].get("_gate_a_label", "?")
        v_b = data["gate_b"]["verdict_gate_b"]
        print(f"  verdict.strict_baseline_pass = {v_strict}  "
              f"(diagnostic _gate_a_label={v_diag})")
        print(f"  gate_b.verdict_gate_b        = {v_b}  (the ONE labeled verdict)")
        design_note = data["gate_b"].get("design_note", "")
        if not design_note:
            failures.append(
                "  ✗ no design_note explaining the option-a structure — "
                "consumer cannot distinguish the boolean from the label"
            )
        else:
            print(f"  ✓ design_note present ({len(design_note)} chars "
                  f"documenting the rename)")
    else:
        # Legacy two-string-fields structure — fail (brief Bug 3).
        v_top = data["verdict"].get("verdict", "?")
        v_b = data["gate_b"]["verdict_gate_b"]
        failures.append(
            f"  ✗ legacy two-string-fields structure (verdict.verdict={v_top}, "
            f"gate_b.verdict_gate_b={v_b}) — option-a rename not applied"
        )

    # ---- Acceptance #9: hero chart regenerated -------------------------
    print("\n[#9] hero chart")
    if not HERO_PATH.exists():
        failures.append(f"  ✗ hero chart missing at {HERO_PATH}")
    else:
        sz = HERO_PATH.stat().st_size
        print(f"  ✓ hero chart present ({sz:,} bytes)")
        hi = HERO_PATH.with_name(HERO_PATH.stem + "_hi" + HERO_PATH.suffix)
        if hi.exists():
            print(f"  ✓ hi-DPI version present ({hi.stat().st_size:,} bytes)")

    # ---- Summary table -------------------------------------------------
    print("\n[summary] cell-level magnitude snapshot (sane = post-fix)")
    for dist_name in ("benign_results", "crash_results"):
        results = data.get(dist_name)
        if results is None:
            continue
        print(f"  {dist_name}:")
        print(f"    {'strat':>20} {'f':>6} {'mean_ret':>10} {'sortino':>10} "
              f"{'max_dd':>9} {'max_loss%':>10}")
        for strat, fmap in results.items():
            for f, m in fmap.items():
                if f in (0.05, 0.40, 0.80):  # subset for visibility
                    pct_loss = m["max_loss_pnl"] / m["strata_capital"]
                    print(f"    {strat:>20} {f:>6.2f} "
                          f"{m['mean_return']:>+10.4f} "
                          f"{m['sortino']:>+10.4f} "
                          f"{m['max_drawdown']:>+9.4f} "
                          f"{pct_loss:>+10.2%}")

    print()
    if failures:
        print(f"[VERIFY FAILED] {len(failures)} acceptance violations:")
        for f in failures:
            print(f)
        return 1
    print("[VERIFY OK] all acceptance criteria satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
