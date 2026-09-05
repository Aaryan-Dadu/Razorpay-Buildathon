"""Evaluation. Baselines, money, ablations, operating curve.

Rules this module enforces:

* Every headline number is computed on held-out orders the linkage model
  never trained on.
* Every detector is reported against the *same* universe, so precision is
  comparable across rows. Reporting one method on flagged-only and another
  on all orders would be quietly meaningless.
* Money is reported in merchant cost, net of the false-positive bill. A
  "recovered" figure that ignores what the false positives cost is a
  marketing number.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import timedelta

from .gate import Decision, Verdict, CostModel
from .schema import GroundTruth, Order, RemediationEvent, rupees


@dataclass(slots=True)
class PRF:
    tp: int
    fp: int
    fn: int
    n_universe: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(precision=self.precision, recall=self.recall, f1=self.f1)
        return d


def score(flagged: set[str], positives: set[str], universe: set[str]) -> PRF:
    flagged &= universe
    positives &= universe
    return PRF(tp=len(flagged & positives),
               fp=len(flagged - positives),
               fn=len(positives - flagged),
               n_universe=len(universe))


# ---------------------------------------------------------------------------
# baselines -- what a merchant can do without this system
# ---------------------------------------------------------------------------


def baseline_explicit_key(orders: list[Order], events: list[RemediationEvent],
                          universe: set[str]) -> set[str]:
    """Join on identifiers that literally exist in the records, flag >=2.

    This is today's state of the art at most merchants, and its ceiling is
    set by the fact that a chargeback carries neither an order_id nor a
    payment_id. It cannot see the most expensive pattern at all.
    """
    pay2ord = {o.payment_id: o.order_id for o in orders}
    groups: dict[str, list[RemediationEvent]] = defaultdict(list)
    for e in events:
        key = e.order_id_hint or (pay2ord.get(e.payment_id or "") if e.payment_id else None)
        if key:
            groups[key].append(e)
    return {k for k, v in groups.items() if len(v) >= 2} & universe


def baseline_customer_amount(orders: list[Order], events: list[RemediationEvent],
                             universe: set[str], window_days: int = 30,
                             tol: float = 0.05) -> set[str]:
    """Flag orders from one customer with similar amounts close in time.

    The obvious heuristic. It is destroyed by the twin-order and
    serial-complainer confusers, which is the point of including it.
    """
    by_cust: dict[str, list[Order]] = defaultdict(list)
    for o in orders:
        by_cust[o.customer_id].append(o)
    out: set[str] = set()
    for rows in by_cust.values():
        rows.sort(key=lambda o: o.created_at)
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if b.created_at - a.created_at > timedelta(days=window_days):
                    break
                if abs(a.amount_paise - b.amount_paise) <= a.amount_paise * tol:
                    out.add(a.order_id)
                    out.add(b.order_id)
    return out & universe


def baseline_oracle_count(truth: GroundTruth, events: list[RemediationEvent],
                          universe: set[str]) -> set[str]:
    """Perfect linkage, then simply count events.

    The most important baseline in the table. It isolates how much of the
    problem is linkage and how much is the reasoning on top: if this scores
    well, the ledger is unnecessary and a join would do.
    """
    groups: dict[str, int] = Counter(
        truth.event_to_order[e.event_id] for e in events
        if e.event_id in truth.event_to_order)
    return {k for k, n in groups.items() if n >= 2} & universe


# ---------------------------------------------------------------------------
# money
# ---------------------------------------------------------------------------


@dataclass
class MoneyResult:
    prevented_paise: int
    false_positive_cost_paise: int
    missed_paise: int
    n_block: int
    n_hold: int
    n_allow: int
    n_correct_block: int
    n_wrong_block: int

    @property
    def net_paise(self) -> int:
        return self.prevented_paise - self.false_positive_cost_paise

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(net_paise=self.net_paise,
                 prevented=rupees(self.prevented_paise),
                 false_positive_cost=rupees(self.false_positive_cost_paise),
                 missed=rupees(self.missed_paise),
                 net=rupees(self.net_paise))
        return d


def realised_fp_cost(pending_value_paise: int, cost: CostModel) -> int:
    """What a wrong block *actually* costs, with hindsight.

    This must not be the gate's own expected-cost estimate. That estimate is
    scaled by the gate's belief that the payout was legitimate, so a gate
    that is confidently wrong prices its own mistakes at nearly zero and
    reports a false-positive bill of a few rupees. Grading a model with its
    own confidence is circular. Once the label is known the payout was
    legitimate, so the full bill is due.
    """
    return int(cost.support_touch_paise
               + cost.p_escalate_if_blocked
               * (pending_value_paise + cost.dispute_fee_paise)
               + cost.churn_paise)


def money(decisions: list[Decision], truly_duplicated: set[str],
          pending_cost: dict[str, int], cost: CostModel,
          pending_value: dict[str, int] | None = None) -> MoneyResult:
    """Rupees actually saved by the gate's decisions, net of its mistakes.

    ``pending_cost`` maps order_id -> merchant cost of the remediation the
    gate was deciding about. A blocked duplicate saves that cost, discounted
    by the share that manual audit would have clawed back anyway. A blocked
    legitimate payout is charged the *realised* false-positive bill -- see
    :func:`realised_fp_cost` for why the gate's own estimate will not do.
    """
    pending_value = pending_value or {}
    prevented = fp_cost = missed = 0
    n_block = n_hold = n_allow = n_ok = n_bad = 0

    for d in decisions:
        is_dup = d.order_id in truly_duplicated
        c = pending_cost.get(d.order_id, 0)
        if d.verdict is Verdict.BLOCK:
            n_block += 1
            if is_dup:
                n_ok += 1
                prevented += int(c * (1.0 - cost.natural_recovery_rate))
            else:
                n_bad += 1
                fp_cost += realised_fp_cost(
                    pending_value.get(d.order_id, 0), cost)
        elif d.verdict is Verdict.HOLD:
            n_hold += 1
            # A hold is not free: it still costs an agent's time, but it
            # does not escalate or churn, because the customer is told.
            fp_cost += cost.support_touch_paise
        else:
            n_allow += 1
            if is_dup:
                missed += c

    return MoneyResult(prevented, fp_cost, missed,
                       n_block, n_hold, n_allow, n_ok, n_bad)


def threshold_sweep(ratios: dict[str, float], positives: set[str],
                    universe: set[str],
                    grid: tuple[float, ...] = tuple(
                        round(1.02 + 0.02 * i, 2) for i in range(50))
                    ) -> list[dict]:
    """Precision/recall across the duplicate-ratio line.

    Shows the choice of 1.25 is not load-bearing: anywhere in the flat
    region gives the same answer, which is what "not tuned to the test set"
    looks like when demonstrated instead of asserted.
    """
    rows = []
    for t in grid:
        flagged = {oid for oid, r in ratios.items() if r >= t}
        s = score(flagged, positives, universe)
        rows.append({"threshold": t, "precision": s.precision,
                     "recall": s.recall, "f1": s.f1,
                     "tp": s.tp, "fp": s.fp, "fn": s.fn})
    return rows
