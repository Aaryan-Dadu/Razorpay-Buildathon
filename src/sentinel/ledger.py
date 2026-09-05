"""The remediation ledger: one view of how whole a customer has been made.

This is the artefact the merchant does not currently have. Refunds live in
the PSP, disputes at the network, credits in the CRM, replacements in the
OMS. Nobody owns the join, so nobody can answer "how much has this order
already paid back?" -- and that unanswered question is the entire loss.

Two modes, deliberately separated:

``audit``       retrospective. Given everything observed, which orders were
                over-remediated? This is what produces precision/recall.

``gate``        prospective. A remediation is *about to* go out. Given the
                ledger so far and the hazard of a chargeback that has not
                arrived yet, should it? This is what actually saves money,
                and it is the harder of the two because the future is
                genuinely unknown at decision time.

Events the resolver abstained on do not vanish. They land in
:attr:`Ledger.exceptions` and are reported as an honest exception list
rather than being silently dropped into "no duplicate found".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .resolve.matcher import Method, Resolution
from .schema import Channel, Order, RemediationEvent, rupees


@dataclass(slots=True)
class LedgerEntry:
    """Everything known about one order's remediation history."""

    order_id: str
    order_value_paise: int
    events: list[RemediationEvent] = field(default_factory=list)
    #: Linkage confidence per event; a low-confidence link makes any
    #: conclusion drawn from this entry correspondingly weaker.
    confidences: list[float] = field(default_factory=list)

    @property
    def returned_paise(self) -> int:
        """Customer-facing value returned. A replacement counts at retail."""
        return sum(e.value_paise for e in self.events)

    @property
    def merchant_cost_paise(self) -> int:
        return sum(e.merchant_cost_paise for e in self.events)

    @property
    def exposure_ratio(self) -> float:
        if not self.order_value_paise:
            return 0.0
        return self.returned_paise / self.order_value_paise

    @property
    def over_paise(self) -> int:
        return max(0, self.returned_paise - self.order_value_paise)

    @property
    def min_confidence(self) -> float:
        return min(self.confidences) if self.confidences else 0.0

    @property
    def channels(self) -> set[Channel]:
        return {e.channel for e in self.events}

    def explain(self) -> str:
        """Human-readable trail. Every gate decision cites one of these."""
        lines = [
            f"order {self.order_id}  value {rupees(self.order_value_paise)}",
            f"  returned {rupees(self.returned_paise)} "
            f"({self.exposure_ratio:.2f}x)  "
            f"merchant cost {rupees(self.merchant_cost_paise)}",
        ]
        for e, c in zip(self.events, self.confidences):
            lines.append(
                f"  - {e.occurred_at:%Y-%m-%d} {str(e.channel):11s} "
                f"{rupees(e.value_paise):>14s}  link_conf={c:.2f}  {e.event_id}"
            )
        return "\n".join(lines)


@dataclass
class Ledger:
    entries: dict[str, LedgerEntry] = field(default_factory=dict)
    #: Events the resolver would not commit to. Human queue, not a silent drop.
    exceptions: list[tuple[RemediationEvent, Resolution]] = field(default_factory=list)

    @classmethod
    def build(cls, orders: list[Order], events: list[RemediationEvent],
              resolutions: list[Resolution]) -> "Ledger":
        by_id = {o.order_id: o for o in orders}
        ev_by_id = {e.event_id: e for e in events}
        res_by_id = {r.event_id: r for r in resolutions}

        led = cls()
        for e in events:
            r = res_by_id.get(e.event_id)
            if r is None or not r.linked or r.order_id not in by_id:
                led.exceptions.append(
                    (e, r or Resolution(e.event_id, None, 0.0, Method.ABSTAINED))
                )
                continue
            entry = led.entries.get(r.order_id)
            if entry is None:
                o = by_id[r.order_id]
                entry = LedgerEntry(o.order_id, o.amount_paise)
                led.entries[r.order_id] = entry
            entry.events.append(e)
            entry.confidences.append(r.confidence)

        for entry in led.entries.values():
            entry.events.sort(key=lambda e: e.occurred_at)
        return led

    # -- retrospective ----------------------------------------------------

    def audit(self, threshold: float = 1.25) -> dict[str, LedgerEntry]:
        """Orders whose observed remediation crosses the duplicate line."""
        return {oid: en for oid, en in self.entries.items()
                if en.exposure_ratio >= threshold}

    # -- prospective ------------------------------------------------------

    def state_at(self, order_id: str, when: datetime) -> LedgerEntry:
        """Ledger as it stood at ``when``.

        The gate must never see events that had not happened yet. Rebuilding
        the entry as-of decision time is the only way a backtest of the gate
        means anything.
        """
        base = self.entries.get(order_id)
        if base is None:
            return LedgerEntry(order_id, 0)
        keep = [(e, c) for e, c in zip(base.events, base.confidences)
                if e.occurred_at < when]
        out = LedgerEntry(order_id, base.order_value_paise)
        for e, c in keep:
            out.events.append(e)
            out.confidences.append(c)
        return out

    def summary(self) -> dict[str, float | int | str]:
        n_multi = sum(1 for e in self.entries.values() if len(e.events) > 1)
        over = sum(e.over_paise for e in self.entries.values())
        return {
            "orders_with_remediation": len(self.entries),
            "orders_multi_event": n_multi,
            "events_linked": sum(len(e.events) for e in self.entries.values()),
            "events_in_exception_queue": len(self.exceptions),
            "gross_over_remediation": rupees(over),
        }
