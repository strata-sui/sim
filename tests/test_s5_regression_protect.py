"""S5R3.7 — regression-protect the R3 verification + writeup hooks artifacts.

Per docs/s5_round3_fix_brief.md:
  * `data/s1_results/s5_r3_verification.json` — passed peer review.
    Pillar 2 (R3 liquid-cash escape-hatch) empirical proof.
    MUST stay: limiter_binding True, liquid_cash_delta_R3 > 0.
  * `data/s1_results/s5_writeup_hooks.md` — peer-review-locked pitch
    language (discovery-first headline, 3-pillar table, BAB signatures).
    MUST stay (only cosmetic edits permitted; structural language is
    what the S6 writer consumes verbatim).

These tests are the regression-net so future S5/S6 refactors cannot
silently regress the two artifacts the brief calls "exemplary and stay."
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "s1_results"
R3_PATH = RESULTS_DIR / "s5_r3_verification.json"
WRITEUP_PATH = RESULTS_DIR / "s5_writeup_hooks.md"


def _r3_payload():
    if not R3_PATH.exists():
        pytest.skip(f"R3 verification not yet produced at {R3_PATH}")
    return json.loads(R3_PATH.read_text(encoding="utf-8"))


def _writeup_text():
    if not WRITEUP_PATH.exists():
        pytest.skip(f"writeup hooks file not yet produced at {WRITEUP_PATH}")
    return WRITEUP_PATH.read_text(encoding="utf-8")


# ---- R3 verification JSON -----------------------------------------------


class TestR3VerificationArtifact:
    """Pillar 2 (R3) empirical proof must remain intact post-S5R3 fixes."""

    def test_required_top_level_keys(self):
        data = _r3_payload()
        for k in (
            "scenario_config", "ladder", "pool_state_pre_settle",
            "r3_delta", "on_chain_mechanic", "honest_notes",
        ):
            assert k in data, f"missing required R3 top-level key: {k}"

    def test_limiter_is_binding_pre_settle(self):
        """The whole point of R3: PLP withdraw limiter has bound (= 0
        available cash through the PLP withdraw path)."""
        data = _r3_payload()
        pre = data["pool_state_pre_settle"]
        assert pre["limiter_binding"] is True, (
            f"R3 limiter no longer binding (={pre['limiter_binding']}); "
            "the entire pillar-2 narrative depends on this being True"
        )
        # And available_for_withdraw is structurally 0 (or very tiny).
        assert pre["available_for_withdraw"] == pytest.approx(0.0, abs=1e-6), (
            f"available_for_withdraw should be ~0 when limiter binding, "
            f"got {pre['available_for_withdraw']}"
        )

    def test_r3_liquid_cash_delta_positive(self):
        """Strata must access strictly more liquid cash than raw_plp."""
        data = _r3_payload()
        delta = float(data["r3_delta"]["liquid_cash_delta_R3"])
        assert delta > 0.0, (
            f"R3 liquid-cash delta should be > 0 to substantiate the "
            f"pillar-2 claim; got ${delta:,.2f}"
        )

    def test_r3_delta_magnitude_in_expected_range(self):
        """Sanity: the empirical delta was ~$383k. Wide band ±50% to
        survive seed variation across re-runs but catch any structural
        collapse."""
        data = _r3_payload()
        delta = float(data["r3_delta"]["liquid_cash_delta_R3"])
        assert 100_000 <= delta <= 1_000_000, (
            f"R3 delta out of plausible range (was $383k pre-fix); "
            f"got ${delta:,.2f}"
        )

    def test_stress_case_responsive(self):
        """Disabling the bypass MUST collapse Strata to raw_plp outcome —
        if it didn't, the R3 lift would be an artifact, not a real
        on-chain mechanic. This test pins the responsiveness check."""
        data = _r3_payload()
        stress = float(data["r3_delta"]["stress_strata_if_no_bypass"])
        raw = float(data["r3_delta"]["raw_plp_depositor_cash_at_freeze"])
        # In the stress (bypass disabled) Strata should collapse close to
        # raw_plp (the brief notes: ~$0 — the ladder cash can't escape).
        assert stress <= raw + 1.0, (
            f"stress (bypass off) Strata cash ${stress:,.2f} should be "
            f"≤ raw_plp ${raw:,.2f}; model not responsive to bypass"
        )

    def test_on_chain_mechanic_cited(self):
        data = _r3_payload()
        oc = data["on_chain_mechanic"]
        assert "contract_path" in oc and oc["contract_path"], (
            "R3 must cite a concrete contract path for the bypass mechanic"
        )

    def test_honest_notes_present(self):
        """At least one honest-disclosure note must be carried."""
        data = _r3_payload()
        notes = data["honest_notes"]
        assert isinstance(notes, list) and len(notes) >= 1


# ---- Writeup hooks markdown ---------------------------------------------


class TestWriteupHooksArtifact:
    """Peer-review-locked pitch language — S6 writer consumes verbatim."""

    def test_discovery_first_headline_present(self):
        text = _writeup_text()
        # The §0 / §2A locked headline framing — must remain.
        ok = (
            "discovery" in text.lower()
            or "discover" in text.lower()
            or "is PLP safe" in text  # the protocol-stated track-fit
        )
        assert ok, "discovery-first / track-fit framing missing from writeup"

    def test_three_pillars_present(self):
        """Tail protection + R3 escape-hatch + f*(w) methodology."""
        text = _writeup_text()
        text_lower = text.lower()
        # tail protection pillar
        assert "tail" in text_lower, "pillar 1 (tail protection) missing"
        # R3 pillar (liquidity escape-hatch)
        assert ("r3" in text_lower or "escape" in text_lower
                or "liquid" in text_lower), "pillar 2 (R3 liquidity) missing"
        # f*(w_crash) methodology pillar
        assert ("f*" in text or "optimal-f" in text_lower
                or "f-curve" in text_lower or "f_star" in text_lower or
                "f(w" in text), "pillar 3 (f*(w) methodology) missing"

    def test_diction_rule_documented(self):
        """CLAUDE.md §8 binding rule: NEVER write 'capped'/'loss-proof'/
        'can't lose'/'capped downside'. The hedge TRUNCATES the tail, leaves
        a QUANTIFIED RESIDUAL.

        The writeup hooks file is the BAB-signature locked-language file
        the S6 writer consumes verbatim — it both bans the prohibited
        diction AND documents the approved vocabulary. This test
        confirms the rule is captured (so a future edit can't silently
        drop it) AND the approved replacement vocabulary is present.
        """
        text = _writeup_text().lower()
        # The rule itself must be documented — the writer needs to know
        # what NOT to say.
        rule_mentions = [
            ("capped", "capped"),
            ("loss-proof", "loss-proof"),
            ("can't lose", "can't lose"),
        ]
        rule_documented = any(phrase in text for _, phrase in rule_mentions)
        assert rule_documented, (
            "writeup hooks must document the CLAUDE.md §8 diction rule "
            "(approved vs banned). None of the rule keywords found."
        )
        # And at least ONE approved-vocabulary phrase must be present.
        approved = [
            "truncate", "truncated", "residual tail",
            "quantified residual", "left-tail", "tail-shaped",
        ]
        approved_present = any(phrase in text for phrase in approved)
        assert approved_present, (
            "writeup hooks missing approved diction (truncated/residual/"
            "left-tail) — the CLAUDE.md §8 vocabulary is the BAB signature"
        )

    def test_self_reference_disclosed(self):
        """CLAUDE.md §3 + §7 BAB honest-disclosure signature must be
        the credibility anchor in the writeup."""
        text = _writeup_text().lower()
        assert (
            "self-ref" in text or "self ref" in text or
            "self-reference" in text or "wash" in text
        ), "self-reference disclosure missing — kills the BAB signature"

    def test_minimum_length(self):
        """Sanity floor against accidental truncation."""
        text = _writeup_text()
        assert len(text) >= 1000, (
            f"writeup hooks unexpectedly short ({len(text)} chars); "
            "likely truncated"
        )
