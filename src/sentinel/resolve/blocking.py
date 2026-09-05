"""Candidate generation.

Scoring every (event, order) pair is 8k x 21k = 170M comparisons per run.
Blocking cuts that to a few dozen candidates per event by requiring at least
one cheap shared signal, then leaning on the pairwise model to rank within
the block.

Two invariants that matter for correctness, not just speed:

* **Causality.** An order can only remediate *after* it was placed. Orders
  created after the event are never candidates.
* **Lookback.** Card schemes cap dispute filing at ~150 days. Beyond the
  lookback a match is not merely unlikely, it is out of policy.

Recall of the blocker is an upper bound on recall of the whole resolver, so
it is measured explicitly in :func:`blocking_recall` and reported.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from ..schema import Order, RemediationEvent

#: Widest plausible gap between an order and a remediation on it.
#: 150d scheme dispute limit + 30d slack for slow support tickets.
MAX_LOOKBACK = timedelta(days=185)

#: Amount bucket for the *fallback* tier only. Rs 200 wide: selective
#: enough to keep blocks small, coarse enough to tolerate rounding.
#:
#: An earlier version used Rs 5,000 buckets on every event. Over an order
#: range of Rs 499..14,999 that put most of the book into one block and a
#: 20k-order run did not finish. Amount is now a last resort, never a
#: default -- see the tiering in :meth:`BlockingIndex.candidates`.
_AMT_BUCKET = 200_00


def _amount_keys(paise: int) -> list[str]:
    """Bucket an amount, plus neighbours, so boundary cases still block."""
    b = paise // _AMT_BUCKET
    return [f"amt:{b}", f"amt:{b - 1}", f"amt:{b + 1}"]


class BlockingIndex:
    """Inverted index from cheap signals to orders."""

    def __init__(self, orders: list[Order]):
        self.orders = {o.order_id: o for o in orders}
        self.index: dict[str, list[str]] = defaultdict(list)
        for o in orders:
            for k in self._order_keys(o):
                self.index[k].append(o.order_id)

    @staticmethod
    def _order_keys(o: Order) -> list[str]:
        keys = [f"email:{o.email}", f"phone:{o.phone}", f"pay:{o.payment_id}",
                f"ord:{o.order_id}"]
        if o.card_last4:
            keys.append(f"l4:{o.card_bin}:{o.card_last4}")
        if o.awb:
            keys.append(f"awb:{o.awb}")
        keys += _amount_keys(o.amount_paise)
        return keys

    @staticmethod
    def _strong_event_keys(e: RemediationEvent) -> list[str]:
        """Identifier-bearing keys only. Each is selective on its own."""
        keys: list[str] = []
        if e.payment_id:
            keys.append(f"pay:{e.payment_id}")
        if e.order_id_hint:
            keys.append(f"ord:{e.order_id_hint}")
        if e.email:
            keys.append(f"email:{e.email}")
        if e.phone:
            keys.append(f"phone:{e.phone}")
        if e.awb:
            keys.append(f"awb:{e.awb}")
        if e.card_last4:
            # All a dispute has. Selective enough alone: 10k tails x 5 BINs.
            keys.append(f"l4:{e.card_bin}:{e.card_last4}")
        return keys

    def candidates(self, e: RemediationEvent, limit: int = 120) -> list[Order]:
        """Tier 1: strong identifiers. Tier 2 (amount) fires only when tier
        1 is empty -- roughly the 10% of support credits logged with neither
        an email nor a phone number attached."""
        seen: set[str] = set()
        for k in self._strong_event_keys(e):
            seen.update(self.index.get(k, ()))
        if not seen:
            for k in _amount_keys(e.value_paise):
                seen.update(self.index.get(k, ()))
        out = []
        for oid in seen:
            o = self.orders[oid]
            gap = e.occurred_at - o.created_at
            if timedelta(0) <= gap <= MAX_LOOKBACK:   # causality + policy
                out.append(o)
        # Truncation order matters. Sorting by recency looks reasonable and
        # is actively wrong: a chargeback lands 8-150 days after its order,
        # so "most recent" throws away precisely the candidates disputes
        # need. Rank by value proximity instead, which is channel-neutral.
        #
        # The `order_id` tiebreaker is not cosmetic. `seen` is a set of
        # strings, and Python salts string hashing per process, so its
        # iteration order changes between runs. Python's sort is stable, so
        # ties silently inherited that varying order -- which changed which
        # candidates survived truncation, which row order the model trained
        # on, and therefore the reported metrics. Two identical invocations
        # disagreed by ~6 links. Sorting on a total order fixes it at the
        # source, without depending on PYTHONHASHSEED being set.
        out.sort(key=lambda o: (abs(o.amount_paise - e.value_paise),
                                o.order_id))
        return out[:limit]


def blocking_recall(index: BlockingIndex, events: list[RemediationEvent],
                    truth: dict[str, str], limit: int = 120) -> dict[str, float]:
    """Fraction of events whose true order survives blocking.

    This is the ceiling on resolver recall. Reporting it separately stops a
    matcher from being blamed for candidates it was never shown.
    """
    hit = miss = 0
    sizes = []
    for e in events:
        true_oid = truth.get(e.event_id)
        if true_oid is None:
            continue
        cands = index.candidates(e, limit)
        sizes.append(len(cands))
        if any(c.order_id == true_oid for c in cands):
            hit += 1
        else:
            miss += 1
    n = hit + miss
    return {
        "blocking_recall": hit / n if n else 0.0,
        "events_scored": n,
        "mean_block_size": sum(sizes) / len(sizes) if sizes else 0.0,
        "max_block_size": max(sizes) if sizes else 0,
    }
