"""Unit tests for the S5.3 R3 escape-hatch scenario.

Per brief §S5 acceptance #7: "new tests for S5.3 R3 scenario".
"""
from __future__ import annotations

from s5_r3_verification import _rig_scenario, _verify_assertions


class TestR3Scenario:
    def test_limiter_binds_pre_settle(self):
        """A heavy trader-mint shock should drive available_for_withdraw → 0."""
        out = _rig_scenario(trader_mint_factor=1.5)
        assert out["pool_state_pre_settle"]["limiter_binding"], (
            "Scenario should rig the limiter to bind; got available="
            f"${out['pool_state_pre_settle']['available_for_withdraw']:,.2f}"
        )

    def test_r3_delta_is_positive_under_crash(self):
        """Hedge ladder pays out on a crash > strike depth → R3 delta > 0."""
        out = _rig_scenario(crash_pct=-0.08)
        assert out["r3_delta"]["liquid_cash_delta_R3"] > 0

    def test_no_crash_no_r3_delta(self):
        """Without a crash large enough to fire the ladder, R3 delta = 0
        (the ladder doesn't pay → strata cash = raw_plp cash)."""
        out = _rig_scenario(crash_pct=0.0)  # no move
        assert out["r3_delta"]["liquid_cash_delta_R3"] == 0.0

    def test_stress_no_bypass_collapses_to_raw_plp(self):
        """If the bypass were disabled, Strata's cash at freeze = raw_plp's
        cash at freeze. The model is RESPONSIVE to the bypass mechanic."""
        out = _rig_scenario()
        assert out["r3_delta"]["stress_strata_if_no_bypass"] == (
            out["r3_delta"]["raw_plp_depositor_cash_at_freeze"]
        )

    def test_verify_assertions_returns_notes(self):
        out = _rig_scenario()
        notes = _verify_assertions(out)
        assert isinstance(notes, list)
        assert any("limiter BINDING" in n for n in notes)
        assert any("R3 DELTA POSITIVE" in n for n in notes)
        assert any("STRESS" in n for n in notes)

    def test_contract_path_documented(self):
        """Pitch claims must cite the on-chain mechanic."""
        out = _rig_scenario()
        assert "predict::redeem_permissionless" in out["on_chain_mechanic"]["contract_path"]
