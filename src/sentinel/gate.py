"""The prospective decision: should this remediation go out?

Detection after the fact is worth something -- you can at least go and
contest the chargeback. But the money is saved by not paying twice in the
first place, and that decision has to be made when the second payment is
still pending and the future is genuinely unknown.

The gate combines two things the merchant has never had together:

  observed   what the ledger already shows was returned on this order
  unobserved the hazard that a chargeback is *coming* and has not landed

and turns them into an expected-value comparison in rupees.

Deliberately not a model
------------------------
The decision itself is arithmetic over two model outputs, not a third
model. A learned policy here would be unauditable, would need its own
labelled decisions to train on, and would put a black box between a
customer and money they may be owed. Every output carries the arithmetic
that produced it -- see :attr:`Decision.rationale`.

The false-positive cost is unusually clean here and drives the whole
threshold. Wrongly holding a legitimate refund does not merely annoy
someone: a meaningful share of those customers go on to file the very
chargeback the system exists to prevent. A false positive *manufactures*
the loss. That is why :data:`P_ESCALATE_IF_BLOCKED` sits in the middle of
the cost model rather than in a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from .hazard import HazardModel, OrderHistory
from .ledger import Ledger, LedgerEntry
from .schema import Channel, rupees


class Verdict(StrEnum):
    ALLOW = "allow"
    HOLD = "hold"      # human queue: the arithmetic is too close to call
    BLOCK = "block"


@dataclass
class CostModel:
    """Merchant-specific economics. Every field is an assumption a merchant
    would calibrate from their own books; they are stated here rather than
    buried so that a reviewer can disagree with them numerically."""

    #: Agent time to handle a held refund and explain it.
    support_touch_paise: int = 120_00
    #: Share of wrongly-held legitimate refunds that escalate to a formal
    #: dispute. This is the term that makes a false positive expensive.
    p_escalate_if_blocked: float = 0.35
    #: Scheme + PSP fee on a dispute, win or lose.
    dispute_fee_paise: int = 150_000
    #: Goodwill/retention damage from wrongly withholding money owed.
    churn_paise: int = 800_00
    #: Share of duplicate payouts that would eventually be clawed back
    #: anyway through manual audit. Recovering something already recoverable
    #: is not a saving, so the benefit is discounted by it.
    natural_recovery_rate: float = 0.15
    #: Width of the indecision band around break-even, as a fraction of the
    #: larger cost. Inside it the gate refuses to decide and asks a human.
    hold_band: float = 0.25


@dataclass
class Decision:
    order_id: str
    verdict: Verdict
    p_duplicate: float
    expected_allow_cost_paise: int
    expected_block_cost_paise: int
    rationale: list[str] = field(default_factory=list)
    ledger_ratio_before: float = 0.0
    ledger_ratio_after: float = 0.0
    hazard_future_dispute: float = 0.0

    @property
    def net_benefit_paise(self) -> int:
        """Expected rupees saved by this decision vs. always allowing."""
        if self.verdict is Verdict.BLOCK:
            return self.expected_allow_cost_paise - self.expected_block_cost_paise
        return 0

    def explain(self) -> str:
        head = (f"[{str(self.verdict).upper():5s}] {self.order_id}  "
                f"p_dup={self.p_duplicate:.2f}  "
                f"allow={rupees(self.expected_allow_cost_paise)}  "
                f"block={rupees(self.expected_block_cost_paise)}")
        return "\n".join([head] + [f"    - {r}" for r in self.rationale])


class Gate:
    def __init__(self, cost: CostModel | None = None,
                 duplicate_threshold: float = 1.25):
        self.cost = cost or CostModel()
        self.threshold = duplicate_threshold

    def evaluate(self, entry: LedgerEntry, pending_value_paise: int,
                 pending_channel: Channel, history: OrderHistory | None = None,
                 hazard: HazardModel | None = None,
                 now: datetime | None = None) -> Decision:
        c = self.cost
        order_value = entry.order_value_paise or 1
        before = entry.returned_paise / order_value
        after = (entry.returned_paise + pending_value_paise) / order_value
        why: list[str] = [
            f"ledger shows {rupees(entry.returned_paise)} already returned "
            f"on a {rupees(entry.order_value_paise)} order ({before:.2f}x) "
            f"across {len(entry.events)} event(s)"
        ]

        # --- observed component -----------------------------------------
        # Discount by linkage confidence: a conclusion drawn from an
        # uncertain merge deserves an uncertain verdict.
        conf = entry.min_confidence if entry.events else 1.0
        p_over_observed = conf if after >= self.threshold else 0.0
        if after >= self.threshold:
            why.append(f"paying {rupees(pending_value_paise)} now would reach "
                       f"{after:.2f}x, over the {self.threshold:.2f}x line "
                       f"(linkage confidence {conf:.2f})")

        # --- unobserved component ---------------------------------------
        p_future = 0.0
        if hazard is not None and history is not None and history.order.is_card:
            elapsed = ((now or entry.events[-1].occurred_at if entry.events
                        else history.order.created_at) - history.order.created_at
                       ).total_seconds() / 86400.0
            p_future = hazard.p_dispute_within(history, from_day=elapsed)
            if p_future > 0.01:
                why.append(f"chargeback not yet arrived but hazard puts "
                           f"P(dispute in remaining window) at {p_future:.1%} "
                           f"({elapsed:.0f}d elapsed)")
                # Paying now, then being charged back later, also duplicates.
                would_exceed_later = (
                    entry.returned_paise + pending_value_paise
                    + order_value) / order_value >= self.threshold
                if would_exceed_later:
                    p_over_observed = max(
                        p_over_observed, p_future * conf)

        p_dup = min(1.0, p_over_observed)

        # --- economics ---------------------------------------------------
        over_if_allowed = max(0, entry.returned_paise + pending_value_paise
                              - order_value)
        allow_cost = int(p_dup * over_if_allowed
                         * (1.0 - c.natural_recovery_rate))

        p_legit = 1.0 - p_dup
        block_cost = int(p_legit * (
            c.support_touch_paise
            + c.p_escalate_if_blocked * (pending_value_paise + c.dispute_fee_paise)
            + c.churn_paise))
        why.append(f"if legitimate, holding costs {rupees(block_cost)} "
                   f"(support + {c.p_escalate_if_blocked:.0%} escalation risk "
                   f"+ retention)")

        bigger = max(allow_cost, block_cost, 1)
        if abs(allow_cost - block_cost) / bigger < c.hold_band:
            verdict = Verdict.HOLD
            why.append("costs within the indecision band -- routed to a human")
        elif allow_cost > block_cost:
            verdict = Verdict.BLOCK
        else:
            verdict = Verdict.ALLOW

        return Decision(entry.order_id, verdict, p_dup, allow_cost, block_cost,
                        why, before, after, p_future)


def replay(ledger: Ledger, gate: Gate,
           histories: dict[str, OrderHistory] | None = None,
           hazard: HazardModel | None = None) -> list[Decision]:
    """Backtest the gate over every second-and-later remediation.

    Each candidate is judged against the ledger *as it stood* immediately
    before that event, never against the finished picture. Judging against
    hindsight would make the gate look clairvoyant and tell the merchant
    nothing about what it will do tomorrow.
    """
    out: list[Decision] = []
    for oid, entry in ledger.entries.items():
        if len(entry.events) < 2:
            continue
        for ev in entry.events[1:]:
            prior = ledger.state_at(oid, ev.occurred_at)
            if not prior.events:
                continue
            out.append(gate.evaluate(
                prior, ev.value_paise, ev.channel,
                history=(histories or {}).get(oid), hazard=hazard,
                now=ev.occurred_at))
    return out
