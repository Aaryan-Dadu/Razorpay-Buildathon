"""Core domain model for the remediation ledger.

Two hard rules encoded here:

1. Money is ``int`` paise everywhere. Never float. A duplicate-remediation
   detector that disagrees with finance by one paisa is a detector nobody
   will switch on.

2. Ground truth lives in :class:`GroundTruth`, never on the event objects
   themselves. The resolver is handed ``list[RemediationEvent]`` and cannot
   reach the answer even by accident. Evaluation hygiene has to be
   structural -- a comment saying "don't use this field" is not a control.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import StrEnum
from typing import Iterable


class Channel(StrEnum):
    """The ways value flows back to a customer.

    These live in four different systems at a real merchant, which is the
    entire reason duplicate remediation goes unnoticed: no shared key, no
    shared clock, no single owner of the join.
    """

    REFUND = "refund"            # PSP.            key: payment_id  (strong)
    DISPUTE = "dispute"          # card network.   key: ARN + PAN tail (weak)
    GOODWILL = "goodwill"        # CRM / support.  key: email/phone + prose (weak)
    REPLACEMENT = "replacement"  # OMS.            key: AWB / order_id (medium)


class LossKind(StrEnum):
    """Why the customer says they are owed something."""

    NOT_DELIVERED = "not_delivered"
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"
    NOT_AS_DESCRIBED = "not_as_described"
    UNAUTHORISED = "unauthorised"


class PaymentMethod(StrEnum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    COD = "cod"


@dataclass(frozen=True, slots=True)
class Order:
    """A settled order. The unit a loss attaches to."""

    order_id: str
    customer_id: str
    created_at: datetime
    amount_paise: int
    cogs_paise: int
    method: PaymentMethod
    payment_id: str
    email: str
    phone: str
    sku: str
    descriptor: str
    card_bin: str | None = None
    card_last4: str | None = None
    awb: str | None = None

    @property
    def is_card(self) -> bool:
        return self.method is PaymentMethod.CARD


@dataclass(frozen=True, slots=True)
class RemediationEvent:
    """One act of making a customer whole, as observed from one system.

    Identifier fields are deliberately sparse and channel-dependent. A
    dispute genuinely does not carry your ``order_id`` -- the network hands
    you an ARN, a masked PAN and a mangled descriptor string. Modelling that
    scarcity faithfully is what makes the resolution problem real rather
    than a dictionary lookup.
    """

    event_id: str
    channel: Channel
    occurred_at: datetime

    #: How whole the *customer* was made. Detecting duplication is a
    #: question about this number: a replacement makes them as whole as a
    #: full refund does, even though it costs the merchant only COGS.
    value_paise: int

    #: What the act cost the *merchant*. Refund -> the refund. Dispute ->
    #: amount + scheme fee. Replacement -> COGS + shipping. The money
    #: metric is denominated in this, never in ``value_paise``.
    merchant_cost_paise: int

    # --- identifiers, sparse by channel -------------------------------
    payment_id: str | None = None      # refund only
    order_id_hint: str | None = None   # replacement sometimes; goodwill rarely
    arn: str | None = None             # dispute only
    card_bin: str | None = None        # dispute, refund-on-card
    card_last4: str | None = None      # dispute, refund-on-card
    email: str | None = None           # goodwill
    phone: str | None = None           # goodwill
    awb: str | None = None             # replacement

    # --- unstructured, where the residual signal hides ----------------
    descriptor_text: str | None = None  # dispute: truncated/garbled merchant string
    free_text: str | None = None        # goodwill: the support agent's note

    def identifiers(self) -> dict[str, str]:
        """Non-null structured identifiers, for blocking and matching."""
        raw = {
            "payment_id": self.payment_id,
            "order_id_hint": self.order_id_hint,
            "arn": self.arn,
            "card_last4": self.card_last4,
            "email": self.email,
            "phone": self.phone,
            "awb": self.awb,
        }
        return {k: v for k, v in raw.items() if v}


@dataclass(frozen=True, slots=True)
class LossEvent:
    """The underlying thing that went wrong. Generator-side concept.

    A loss event is what remediation events *should* cluster onto. One loss,
    one making-whole. Two remediations on one loss is the leak we hunt.
    """

    loss_id: str
    order_id: str
    kind: LossKind
    opened_at: datetime


@dataclass
class GroundTruth:
    """Held separately from the observable stream. See module docstring.

    ``event_to_order`` is the entity-resolution answer key.
    ``duplicated_orders`` is the detection answer key.
    ``confuser_orders`` are orders deliberately built to look like
    duplicates without being any -- they are what stops a precision number
    from being a participation trophy.
    """

    event_to_order: dict[str, str] = field(default_factory=dict)
    event_to_loss: dict[str, str] = field(default_factory=dict)
    losses: dict[str, LossEvent] = field(default_factory=dict)
    duplicated_orders: set[str] = field(default_factory=set)
    confuser_orders: set[str] = field(default_factory=set)
    # order_id -> paise returned beyond the order value (0 when clean)
    over_remediation_paise: dict[str, int] = field(default_factory=dict)
    #: order_id -> which duplicate pattern produced it. Used only to break
    #: recall down by pattern in the report; never fed to the detector.
    duplicate_patterns: dict[str, str] = field(default_factory=dict)

    def cluster_of(self, order_id: str) -> set[str]:
        return {e for e, o in self.event_to_order.items() if o == order_id}


@dataclass
class Dataset:
    """Everything a run needs. ``truth`` must never be passed downstream."""

    orders: list[Order]
    events: list[RemediationEvent]
    truth: GroundTruth
    horizon: datetime            # observation cutoff -> right-censoring
    seed: int = 0

    def orders_by_id(self) -> dict[str, Order]:
        return {o.order_id: o for o in self.orders}

    def events_by_id(self) -> dict[str, RemediationEvent]:
        return {e.event_id: e for e in self.events}

    def observable(self) -> tuple[list[Order], list[RemediationEvent]]:
        """The only view the pipeline is allowed to consume."""
        return self.orders, self.events

    def summary(self) -> dict[str, int]:
        by_channel: dict[str, int] = {}
        for e in self.events:
            by_channel[str(e.channel)] = by_channel.get(str(e.channel), 0) + 1
        return {
            "orders": len(self.orders),
            "events": len(self.events),
            "losses": len(self.truth.losses),
            "duplicated_orders": len(self.truth.duplicated_orders),
            "confuser_orders": len(self.truth.confuser_orders),
            **{f"events_{k}": v for k, v in sorted(by_channel.items())},
        }


def rupees(paise: int) -> str:
    """Format paise for human-facing output. Indian grouping."""
    neg = paise < 0
    s = f"{abs(paise) // 100:,}"
    # regroup to lakh/crore style
    whole = str(abs(paise) // 100)
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"{'-' if neg else ''}Rs {s}.{abs(paise) % 100:02d}"


def to_records(items: Iterable) -> list[dict]:
    """Dataclass sequence -> JSON-ready dicts (enums and datetimes flattened)."""
    out = []
    for it in items:
        d = asdict(it)
        for k, v in list(d.items()):
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif isinstance(v, StrEnum):
                d[k] = str(v)
        out.append(d)
    return out
