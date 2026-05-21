"""PLP-vault accounting — verified semantics from CLAUDE.md §4.

Implements the single embedded shared Vault state and the four state
transitions that matter for the S1 thin slice:

  1. ``supply(amount)``         supply DUSDC, mint shares (NAV-proportional).
  2. ``withdraw(shares)``       redeem shares for DUSDC (limiter-gated).
  3. ``receive_premium(amt)``   trader mint: pool collects spread.
  4. ``pay_settlement(amt)``    on resolve: pool pays winners.

Plus the two mark-to-market updates needed for NAV correctness:
  5. ``update_mtm(new_total_mtm)``
  6. ``update_max_payout(new_total_max_payout)``

Invariants (CLAUDE.md §4):
    vault_value (NAV)        = balance - total_mtm
    share_price (NAV/shares) = NAV / shares_outstanding  (1.0 at bootstrap)
    available_for_withdraw   = max(0, balance - total_max_payout)

Post-crash, ``total_max_payout`` can equal or exceed ``balance``, in which
case ``available_for_withdraw -> 0`` and PLP withdrawals are BLOCKED.
This is the liquidity escape-hatch driver (CLAUDE.md §2A R3): Strata's
hedge-leg ``redeem`` bypasses this limiter, providing the only liquid
exit. The hedge module (S1 task 6) will exercise that bypass.

S1 form: scalar (one-vault per path). Vectorization across paths is the
MC runner's job (S1 task 8). S5+ can promote this to ndarray state if
the runner becomes the bottleneck.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class PLPWithdrawBlocked(RuntimeError):
    """Raised when withdraw exceeds available (limiter binding).

    Carries `amount_requested` and `available` for the strategy layer to
    decide what to do (e.g., trigger hedge `redeem` bypass).
    """

    def __init__(self, amount_requested: float, available: float) -> None:
        super().__init__(
            f"PLP withdraw blocked: requested={amount_requested:.2f}, "
            f"available={available:.2f}"
        )
        self.amount_requested = amount_requested
        self.available = available


@dataclass
class PLPVault:
    """Single PLP-vault state (one path).

    Attrs:
        balance:           USD held by pool (premium IN - settlement OUT
                           - withdraw paid + supply received).
        total_mtm:         sum of mark-to-market on all open binary
                           positions held by traders. Higher MTM => pool
                           is closer to paying winners => NAV lower.
        total_max_payout:  sum of maximum possible payout on all open
                           positions (= notional). Drives the withdraw
                           limiter via ``available = balance - this``.
        shares_outstanding: total LP shares issued. share_price = NAV / this.
    """

    balance: float = 0.0
    total_mtm: float = 0.0
    total_max_payout: float = 0.0
    shares_outstanding: float = 0.0

    # ---- read-only views ---------------------------------------------

    @property
    def nav(self) -> float:
        """Vault NAV = balance - total_mtm (CLAUDE.md §4)."""
        return self.balance - self.total_mtm

    @property
    def share_price(self) -> float:
        """NAV per share. 1.0 bootstrap when no shares outstanding."""
        if self.shares_outstanding == 0.0:
            return 1.0
        return self.nav / self.shares_outstanding

    @property
    def available_for_withdraw(self) -> float:
        """``max(0, balance - total_max_payout)`` (CLAUDE.md §4 limiter).

        Post-crash this collapses toward 0. PLP withdrawals through
        ``withdraw()`` are blocked. Hedge-leg ``redeem`` (a separate
        contract path) bypasses this and is the liquid exit.
        """
        return max(0.0, self.balance - self.total_max_payout)

    # ---- supply / withdraw -------------------------------------------

    def supply(self, amount: float) -> float:
        """Supply ``amount`` DUSDC, return shares minted.

        Shares are NAV-proportional. At bootstrap (no shares) it is 1:1.
        Subsequently: ``shares = amount * shares_outstanding / NAV``.
        """
        if amount <= 0:
            raise ValueError(f"supply amount must be > 0, got {amount}")
        if self.shares_outstanding == 0.0:
            shares = amount
        else:
            if self.nav <= 0:
                raise ValueError(
                    f"cannot supply into a non-positive NAV ({self.nav})"
                )
            shares = amount * self.shares_outstanding / self.nav
        self.balance += amount
        self.shares_outstanding += shares
        return shares

    def withdraw(self, shares: float) -> float:
        """Burn ``shares``, return USD paid out. Raises if limiter binds."""
        if shares <= 0:
            raise ValueError(f"withdraw shares must be > 0, got {shares}")
        if shares > self.shares_outstanding:
            raise ValueError(
                f"cannot withdraw {shares} shares: only "
                f"{self.shares_outstanding} outstanding"
            )
        amount = shares * self.share_price
        avail = self.available_for_withdraw
        if amount > avail:
            raise PLPWithdrawBlocked(amount, avail)
        self.balance -= amount
        self.shares_outstanding -= shares
        return amount

    # ---- premium / settlement / MTM ----------------------------------

    def receive_premium(self, amount: float) -> None:
        """Pool receives ``amount`` DUSDC of premium (trader mint)."""
        if amount < 0:
            raise ValueError(f"premium amount must be >= 0, got {amount}")
        self.balance += amount

    def pay_settlement(self, amount: float) -> None:
        """Pool pays ``amount`` DUSDC on settlement."""
        if amount < 0:
            raise ValueError(f"settlement amount must be >= 0, got {amount}")
        self.balance -= amount

    def update_mtm(self, new_total_mtm: float) -> None:
        """Set MTM (the trader-side mark, NOT including settlements)."""
        if new_total_mtm < 0:
            raise ValueError(f"total_mtm must be >= 0, got {new_total_mtm}")
        self.total_mtm = new_total_mtm

    def update_max_payout(self, new_total_max_payout: float) -> None:
        """Set total_max_payout (drives the withdraw limiter)."""
        if new_total_max_payout < 0:
            raise ValueError(
                f"total_max_payout must be >= 0, got {new_total_max_payout}"
            )
        self.total_max_payout = new_total_max_payout
