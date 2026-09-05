"""Two-stage linkage: exact keys, then a learned scorer over the residual.

Stage 1 is deterministic. If a refund carries a payment_id that matches an
order, that is not a prediction, it is a join, and putting a model in front
of it would be worse in every dimension: slower, less accurate, unauditable.

Stage 2 trains only on what stage 1 could *not* settle. Training on
everything would let `strong_payment_id` dominate the loss and leave the
model useless on precisely the population it exists to serve -- disputes
and support credits, which carry no strong key at all.

Stage 3 does not decide. It abstains. When the top candidate is weak, or
too close to the runner-up, the event is marked AMBIGUOUS and handed to
llm.py. Abstention is a feature: a wrong link here silently corrupts the
ledger downstream, and a held refund costs less than a bad merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..schema import Order, RemediationEvent
from .blocking import BlockingIndex
from .features import FEATURE_NAMES, pair_features


class Method(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    LLM = "llm"
    ABSTAINED = "abstained"


@dataclass(slots=True)
class Resolution:
    event_id: str
    order_id: str | None
    confidence: float
    method: Method
    margin: float = 0.0
    runner_up: str | None = None

    @property
    def linked(self) -> bool:
        return self.order_id is not None


#: Below this the top candidate is not trusted on its own.
CONF_FLOOR = 0.55
#: Top-1 must beat top-2 by this much, else the pair is genuinely ambiguous
#: (the twin-order confuser is built to produce exactly this situation).
MARGIN_FLOOR = 0.15


def deterministic_link(e: RemediationEvent,
                       candidates: list[Order]) -> Order | None:
    """Exact identifier agreement. No model, no threshold, fully auditable."""
    for o in candidates:
        if e.payment_id and e.payment_id == o.payment_id:
            return o
    for o in candidates:
        if e.order_id_hint and e.order_id_hint == o.order_id:
            return o
    for o in candidates:
        if e.awb and o.awb and e.awb == o.awb:
            return o
    return None


class LinkageModel:
    """Pairwise scorer, normalised within a block."""

    #: ``ctx_*`` is dropped by default, and that default was earned rather
    #: than assumed. Including it lifts raw linkage accuracy by ~0.5pp and
    #: costs 4-9pp of end-to-end detection *precision*, consistently across
    #: seeds 7/11/23. The cause is ``ctx_amount_rank``: candidates are
    #: truncated in amount-proximity order, so the model learns "rank 0 is
    #: usually right", which is a fact about the blocker rather than about
    #: the world. On genuinely ambiguous pairs -- the twin-order confusers --
    #: that produces confident wrong links, and a confident wrong link
    #: becomes a false duplicate flag downstream.
    #:
    #: Optimising the intermediate metric degraded the end-to-end one. The
    #: features stay in the codebase so the ablation can keep demonstrating
    #: it; they are simply off by default.
    DEFAULT_DROP: tuple[str, ...] = ("ctx",)

    def __init__(self, seed: int = 0,
                 drop_families: tuple[str, ...] | None = None):
        """``drop_families`` zeroes whole feature families ('strong',
        'weak', 'text', 'ctx') for ablation. Zeroing rather than removing
        keeps the vector width fixed, so the same feature indices mean the
        same thing in every ablation run and results stay comparable.

        Pass ``()`` explicitly to use every feature including ``ctx_*``.
        """
        if drop_families is None:
            drop_families = self.DEFAULT_DROP
        self.clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, max_depth=6,
            l2_regularization=1.0, random_state=seed,
        )
        self.fitted = False
        self.n_train_pairs = 0
        self.drop_families = tuple(drop_families)
        self._mask = np.array(
            [0.0 if n.split("_", 1)[0] in self.drop_families else 1.0
             for n in FEATURE_NAMES], dtype=np.float64)

    def _apply_mask(self, X: np.ndarray) -> np.ndarray:
        return X * self._mask if self.drop_families else X

    # -- training ---------------------------------------------------------

    @staticmethod
    def _pairs(index: BlockingIndex, events: list[RemediationEvent],
               truth: dict[str, str], residual_only: bool = True):
        X, y = [], []
        for e in events:
            true_oid = truth.get(e.event_id)
            if true_oid is None:
                continue
            cands = index.candidates(e)
            if not cands:
                continue
            if residual_only and deterministic_link(e, cands) is not None:
                continue          # stage 1 owns it; do not train on it
            for rank, o in enumerate(cands):
                X.append(pair_features(e, o, block_size=len(cands),
                                       amount_rank=rank))
                y.append(1 if o.order_id == true_oid else 0)
        return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)

    def fit(self, index: BlockingIndex, train_events: list[RemediationEvent],
            truth: dict[str, str]) -> "LinkageModel":
        X, y = self._pairs(index, train_events, truth)
        if len(X) == 0 or y.sum() == 0:
            raise ValueError("no trainable residual pairs -- check blocking")
        self.clf.fit(self._apply_mask(X), y)
        self.fitted = True
        self.n_train_pairs = len(X)
        return self

    # -- inference --------------------------------------------------------

    def rank(self, e: RemediationEvent,
             candidates: list[Order]) -> list[tuple[Order, float]]:
        X = np.asarray(
            [pair_features(e, o, block_size=len(candidates), amount_rank=r)
             for r, o in enumerate(candidates)], dtype=np.float64)
        p = self.clf.predict_proba(self._apply_mask(X))[:, 1]
        # Normalise within the block: exactly one candidate should be right,
        # so a candidate's score is only meaningful against its rivals.
        s = p.sum()
        norm = p / s if s > 0 else np.full_like(p, 1.0 / len(p))
        pairs = sorted(zip(candidates, norm), key=lambda t: -t[1])
        return pairs

    def importances(self) -> dict[str, float]:
        """Permutation importance is computed in report.py; this is the
        cheap structural view for sanity-checking feature plumbing."""
        return {}


def resolve_all(index: BlockingIndex, events: list[RemediationEvent],
                model: LinkageModel | None = None,
                conf_floor: float = CONF_FLOOR,
                margin_floor: float = MARGIN_FLOOR) -> list[Resolution]:
    """Run the full cascade over a set of events."""
    out: list[Resolution] = []
    for e in events:
        cands = index.candidates(e)
        if not cands:
            out.append(Resolution(e.event_id, None, 0.0, Method.ABSTAINED))
            continue

        hit = deterministic_link(e, cands)
        if hit is not None:
            out.append(Resolution(e.event_id, hit.order_id, 1.0,
                                  Method.DETERMINISTIC, margin=1.0))
            continue

        if model is None or not model.fitted:
            out.append(Resolution(e.event_id, None, 0.0, Method.ABSTAINED))
            continue

        ranked = model.rank(e, cands)
        top_o, top_p = ranked[0]
        second_p = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_p - second_p
        runner = ranked[1][0].order_id if len(ranked) > 1 else None

        if top_p >= conf_floor and margin >= margin_floor:
            out.append(Resolution(e.event_id, top_o.order_id, float(top_p),
                                  Method.MODEL, float(margin), runner))
        else:
            # Deliberate abstention -> llm.py gets a shot at it.
            out.append(Resolution(e.event_id, None, float(top_p),
                                  Method.ABSTAINED, float(margin), runner))
    return out


def temporal_split(events: list[RemediationEvent], frac: float = 0.7
                   ) -> tuple[list[RemediationEvent], list[RemediationEvent], datetime]:
    """Split by event time, not at random.

    A random split leaks: the same order's refund lands in train and its
    chargeback in test, so the model sees the answer. Splitting on the clock
    is also the only split that matches deployment -- you always train on
    the past and score the future.
    """
    ordered = sorted(events, key=lambda e: e.occurred_at)
    cut = int(len(ordered) * frac)
    boundary = ordered[cut].occurred_at
    return ordered[:cut], ordered[cut:], boundary


def order_temporal_split(orders: list[Order], events: list[RemediationEvent],
                         truth: dict[str, str], frac: float = 0.7
                         ) -> tuple[set[str], set[str], list[RemediationEvent], datetime]:
    """Split on *order* creation time, and return train events by lineage.

    Why not split the event stream directly: an order's refund and its
    chargeback are two events months apart. Cutting the event stream puts
    them on opposite sides of the boundary, so the ledger sees one event,
    scores the order single-remediated, and recall is understated for a
    reason that has nothing to do with the detector. Production never has
    that problem -- it sees every event that has arrived.

    So: orders are split by creation date, the linkage model trains only on
    events belonging to train-side orders, and detection is scored only on
    test-side orders using their full event history.

    Returns ``(train_order_ids, test_order_ids, train_events, cutoff)``.
    """
    ordered = sorted(orders, key=lambda o: o.created_at)
    cut = int(len(ordered) * frac)
    cutoff = ordered[cut].created_at
    train_ids = {o.order_id for o in ordered[:cut]}
    test_ids = {o.order_id for o in ordered[cut:]}
    train_events = [e for e in events if truth.get(e.event_id) in train_ids]
    return train_ids, test_ids, train_events, cutoff
