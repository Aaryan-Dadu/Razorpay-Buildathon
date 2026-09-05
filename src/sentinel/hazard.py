"""Time-to-chargeback as survival, not classification.

The premise the rest of the industry gets wrong
-----------------------------------------------
A chargeback lands 8-150 days after the transaction. So at any moment, the
recent slice of the book contains orders that *will* be disputed and have
not been yet. Treating those as clean negatives -- which is what a plain
binary classifier on "was_charged_back" does -- means:

* the merchant's reported chargeback rate is biased **low**, systematically,
  and worst for the most recent (most decision-relevant) cohort;
* a model trained that way learns "recent order => safe", which is exactly
  backwards.

The fix is standard and underused: right-censoring. An order observed for
40 days without a dispute is not a negative, it is a survivor to day 40.

Implementation is discrete-time survival in person-period form: expand each
order into one row per 15-day period it was genuinely at risk, label the
period in which the dispute landed, and fit any binary learner to get the
hazard h(k | x). Survival is the running product of (1 - h). This buys
flexible features (a Cox model would not take "was refunded in period 2"
nearly as naturally) at the cost of a coarser clock.

:func:`censoring_bias` quantifies the gap between the naive rate and the
censoring-corrected one. That number is the argument for this whole module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .schema import Channel, Dataset, Order, PaymentMethod, RemediationEvent

#: Clock granularity. 15 days x 10 periods covers the 150-day scheme window.
PERIOD_DAYS = 15
N_PERIODS = 10
MAX_DAYS = PERIOD_DAYS * N_PERIODS

#: A cohort must have seen at least this share of the dispute window
#: before its rate is extrapolated. Below it, report "insufficient
#: exposure" rather than a ratio dominated by division noise.
MIN_EXPOSURE_TO_PROJECT = 0.25

PERIOD_FEATURES = [
    "period", "log_amount", "is_card", "is_upi", "is_cod",
    "n_events_before", "value_returned_ratio_before",
    "had_refund_before", "had_goodwill_before", "had_replacement_before",
    "days_observed",
]


@dataclass(slots=True)
class OrderHistory:
    """What was known about an order, and what eventually happened."""

    order: Order
    dispute_day: float | None      # days from order to chargeback, if seen
    observed_days: float           # how long we watched before the horizon
    events: list[RemediationEvent] # non-dispute remediations, time-ordered

    @property
    def censored(self) -> bool:
        return self.dispute_day is None


def build_histories(ds: Dataset, event_to_order: dict[str, str]
                    ) -> dict[str, OrderHistory]:
    """Assemble per-order survival records from resolved links.

    ``event_to_order`` is the *resolver's* output in production use, or
    ground truth when measuring the hazard model in isolation. Passing the
    resolver's links is the honest end-to-end setting: linkage error
    propagates into the hazard estimate, as it would in deployment.
    """
    by_order: dict[str, list[RemediationEvent]] = {}
    for e in ds.events:
        oid = event_to_order.get(e.event_id)
        if oid:
            by_order.setdefault(oid, []).append(e)

    out: dict[str, OrderHistory] = {}
    for o in ds.orders:
        evs = sorted(by_order.get(o.order_id, []), key=lambda e: e.occurred_at)
        disputes = [e for e in evs if e.channel is Channel.DISPUTE]
        d_day = ((disputes[0].occurred_at - o.created_at).total_seconds() / 86400.0
                 if disputes else None)
        observed = (ds.horizon - o.created_at).total_seconds() / 86400.0
        out[o.order_id] = OrderHistory(
            order=o,
            dispute_day=d_day,
            observed_days=max(0.0, observed),
            events=[e for e in evs if e.channel is not Channel.DISPUTE],
        )
    return out


def _row(h: OrderHistory, k: int) -> list[float]:
    """Features for order ``h`` in period ``k``, using only what was known
    at the *start* of that period. Anything later would be leakage."""
    cutoff_day = k * PERIOD_DAYS
    prior = [e for e in h.events
             if (e.occurred_at - h.order.created_at).total_seconds() / 86400.0
             < cutoff_day]
    returned = sum(e.value_paise for e in prior)
    m = h.order.method
    return [
        float(k),
        float(np.log1p(h.order.amount_paise)),
        1.0 if m is PaymentMethod.CARD else 0.0,
        1.0 if m is PaymentMethod.UPI else 0.0,
        1.0 if m is PaymentMethod.COD else 0.0,
        float(len(prior)),
        returned / h.order.amount_paise if h.order.amount_paise else 0.0,
        1.0 if any(e.channel is Channel.REFUND for e in prior) else 0.0,
        1.0 if any(e.channel is Channel.GOODWILL for e in prior) else 0.0,
        1.0 if any(e.channel is Channel.REPLACEMENT for e in prior) else 0.0,
        min(h.observed_days, float(MAX_DAYS)),
    ]


def person_period(histories: list[OrderHistory]
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Expand orders into (order, period) rows over their at-risk window.

    An order contributes rows only for periods it was actually observed
    through. That is the whole mechanism: a 20-day-old order contributes
    two rows, not ten, so it cannot be counted as a survivor of periods
    nobody watched.
    """
    X, y, ids = [], [], []
    for h in histories:
        if not h.order.is_card:
            continue                      # card rails only: see generate.py
        if h.dispute_day is not None:
            last = min(int(h.dispute_day // PERIOD_DAYS), N_PERIODS - 1)
            event_k = last
        else:
            last = min(int(h.observed_days // PERIOD_DAYS), N_PERIODS - 1)
            event_k = None
        for k in range(last + 1):
            X.append(_row(h, k))
            y.append(1 if (event_k is not None and k == event_k) else 0)
            ids.append(h.order.order_id)
    return (np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.int64), ids)


class HazardModel:
    """Discrete-time hazard h(k | x), fitted on person-period rows."""

    def __init__(self, seed: int = 0):
        self.clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=4,
            l2_regularization=1.0, random_state=seed)
        self.fitted = False
        self.n_rows = 0

    def fit(self, histories: list[OrderHistory]) -> "HazardModel":
        X, y, _ = person_period(histories)
        if len(X) == 0 or y.sum() == 0:
            raise ValueError("no dispute events in training window")
        self.clf.fit(X, y)
        self.fitted = True
        self.n_rows = len(X)
        return self

    def hazard_curve(self, h: OrderHistory) -> np.ndarray:
        X = np.asarray([_row(h, k) for k in range(N_PERIODS)], dtype=np.float64)
        return self.clf.predict_proba(X)[:, 1]

    def p_dispute_within(self, h: OrderHistory, days: int = MAX_DAYS,
                         from_day: float | None = None) -> float:
        """P(dispute arrives within ``days``), conditional on having
        survived to ``from_day``. This is the number the gate consumes."""
        haz = self.hazard_curve(h)
        start = int((from_day if from_day is not None else 0) // PERIOD_DAYS)
        end = min(N_PERIODS, int(np.ceil(days / PERIOD_DAYS)))
        surv = 1.0
        for k in range(max(0, start), end):
            surv *= (1.0 - float(haz[k]))
        return 1.0 - surv


# ---------------------------------------------------------------------------
# the argument for this module
# ---------------------------------------------------------------------------


def kaplan_meier(histories: list[OrderHistory]) -> tuple[np.ndarray, np.ndarray]:
    """Non-parametric survival, period by period. No features, no model.

    Kept as the honest baseline: if the learned hazard cannot beat this, the
    features are not earning their place.
    """
    at_risk = np.zeros(N_PERIODS)
    events = np.zeros(N_PERIODS)
    for h in histories:
        if not h.order.is_card:
            continue
        if h.dispute_day is not None:
            k = min(int(h.dispute_day // PERIOD_DAYS), N_PERIODS - 1)
            at_risk[: k + 1] += 1
            events[k] += 1
        else:
            k = min(int(h.observed_days // PERIOD_DAYS), N_PERIODS - 1)
            at_risk[: k + 1] += 1
    haz = np.divide(events, at_risk, out=np.zeros_like(events),
                    where=at_risk > 0)
    surv = np.cumprod(1.0 - haz)
    return haz, surv


def censoring_bias(histories: list[OrderHistory]) -> dict[str, float]:
    """Naive chargeback rate vs the censoring-corrected estimate.

    The naive rate divides observed disputes by all card orders, which
    silently asserts that every order has had its full 150 days to be
    disputed. Most have not. The gap is not noise -- it is a structural
    understatement, and it is largest exactly where decisions are being
    made today.
    """
    card = [h for h in histories if h.order.is_card]
    if not card:
        return {}
    observed = sum(1 for h in card if h.dispute_day is not None)
    naive = observed / len(card)

    _, surv = kaplan_meier(card)
    corrected = float(1.0 - surv[-1])

    fully_seen = [h for h in card if h.observed_days >= MAX_DAYS]
    mature = (sum(1 for h in fully_seen if h.dispute_day is not None)
              / len(fully_seen)) if fully_seen else float("nan")

    return {
        "naive_rate": naive,
        "km_corrected_rate": corrected,
        "mature_cohort_rate": mature,
        "understatement_x": corrected / naive if naive else float("nan"),
        "n_card_orders": len(card),
        "n_fully_observed": len(fully_seen),
        "pct_still_at_risk": sum(1 for h in card
                                 if h.observed_days < MAX_DAYS) / len(card),
    }


def cohort_bias(histories: list[OrderHistory],
                buckets: tuple[int, ...] = (30, 60, 90, 120, 150)
                ) -> list[dict[str, float]]:
    """Understatement of the naive rate, broken out by how long each cohort
    has been observed.

    This is the presentation that makes the point honestly. A single
    blended number hides the structure: mature cohorts are fine, and the
    naive rate is badly wrong exactly where the merchant is making today's
    decisions. Reporting the blend alone would be the same mistake the
    method is meant to correct.
    """
    card = [h for h in histories if h.order.is_card]
    _, surv = kaplan_meier(card)
    rows = []
    edges = (0,) + buckets + (10**6,)
    for lo, hi in zip(edges[:-1], edges[1:]):
        grp = [h for h in card if lo <= h.observed_days < hi]
        if len(grp) < 30:
            continue
        naive = sum(1 for h in grp if h.dispute_day is not None) / len(grp)
        # Fraction of the full 150-day dispute window this cohort has seen.
        k = min(N_PERIODS, max(1, int(np.mean([h.observed_days for h in grp])
                                      // PERIOD_DAYS)))
        seen_frac = float(1.0 - surv[k - 1]) / float(1.0 - surv[-1]) \
            if surv[-1] < 1.0 else 1.0
        # Dividing by a small exposure fraction amplifies noise without
        # bound: a cohort that has seen 9% of its dispute window will
        # happily "project" a 12% chargeback rate off three observations.
        # Below the floor the honest output is "cannot project yet", not a
        # number with four decimal places on it.
        projectable = seen_frac >= MIN_EXPOSURE_TO_PROJECT
        rows.append({
            "observed_days_lo": float(lo),
            "observed_days_hi": float(min(hi, 10**6)),
            "n_orders": float(len(grp)),
            "naive_rate": naive,
            "exposure_seen": seen_frac,
            "projectable": float(projectable),
            "projected_rate": (naive / seen_frac) if projectable else float("nan"),
            "understatement_x": (1.0 / seen_frac) if projectable else float("nan"),
        })
    return rows
