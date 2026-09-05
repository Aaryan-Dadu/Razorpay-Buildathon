"""Synthetic four-system world with planted lineage and adversarial confusers.

Why this file is the most important one in the repo
---------------------------------------------------
Any detector scores well on a generator that only plants easy positives.
The number that means something is precision *in the presence of cases that
look exactly like positives and are not*. So this generator spends as much
effort building confusers as it does building duplicates:

    twin orders          same customer, same card, same day, near-equal amount
    split remediation    one loss, legitimately settled in two partial events
    serial complainer    many orders, each with one legitimate remediation
    make-good combo      replacement + small goodwill, still under order value
    shared card          two customers behind one PAN (family / corporate)

Each of those trips a naive rule (`>=2 events on a customer`, `same card and
amount inside 30 days`, `two channels touched`). They are labelled
``confuser`` in the ground truth so precision can be reported against them
specifically -- see :meth:`GroundTruth.confuser_orders`.

Labels are *derived from what was actually emitted*, never asserted. If a
pattern intends to duplicate but emits values summing under the order value,
it is not labelled a duplicate. That keeps the answer key honest even when
the emission code has a bug.
"""

from __future__ import annotations

import zlib

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .schema import (
    Channel,
    Dataset,
    GroundTruth,
    LossEvent,
    LossKind,
    Order,
    PaymentMethod,
    RemediationEvent,
)

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class WorldConfig:
    """Knobs for the simulated merchant. Defaults model a mid-size Indian
    D2C brand: card-heavy with meaningful UPI and COD share, ~4% loss rate,
    and a dispute pipeline that lags weeks behind the refund pipeline."""

    n_orders: int = 4_000
    n_customers: int = 1_400
    start: datetime = datetime(2026, 1, 1)
    days: int = 240              # order intake window
    #: Observation cutoff. Set equal to the intake window on purpose: a live
    #: merchant's book always runs up to today, so it always contains fresh
    #: orders whose disputes have not had time to arrive.
    #:
    #: An earlier config stopped intake 60 days before the horizon, which
    #: made almost every order mature and shrank the censoring effect to
    #: 1.03x -- an artefact of the setup, not a property of the world.
    #: With intake running to the horizon the age gradient is real, and
    #: :func:`sentinel.hazard.cohort_bias` reports it by cohort.
    horizon_days: int = 240

    # money (paise)
    order_min: int = 49_900
    order_max: int = 1_499_900
    cogs_ratio: float = 0.42
    dispute_fee: int = 150_000        # scheme + PSP representment fee
    replacement_ship: int = 9_000

    # rates
    loss_rate: float = 0.055          # orders that go wrong at all
    duplicate_rate: float = 0.26      # of losses, how many get paid twice
    confuser_rate: float = 0.22       # of clean orders, how many are traps

    method_mix: tuple[float, ...] = (0.46, 0.30, 0.09, 0.15)  # card/upi/nb/cod

    #: An order counts as duplicated when total value returned reaches this
    #: multiple of order value. Rationale for 1.25, which a reviewer will
    #: and should ask about:
    #:     two independent full remediations   -> ~2.00x
    #:     partial refund + full chargeback    -> ~1.50x
    #:     replacement + goodwill token        -> ~1.05x   (legitimate)
    #: The gap between 1.05 and 1.50 is wide, so the line sits at 1.25 and
    #: the result is insensitive to moving it anywhere in 1.10..1.40.
    #: :func:`sentinel.generate.threshold_sensitivity` demonstrates that.
    duplicate_threshold: float = 1.25

    # dispute timing: lognormal, long right tail, 10..150 days
    cb_log_mu: float = 3.65
    cb_log_sigma: float = 0.55

    seed: int = 7


_SKUS = [
    "KRT-BLU-M", "KRT-BLK-L", "SNK-WHT-9", "WCH-STL-42", "EAR-TWS-B",
    "BAG-TAN-OS", "JNS-IND-32", "SRE-RED-OS", "LMP-BRS-01", "MAT-YOG-6",
]
_SKU_NAME = {
    "KRT-BLU-M": "blue kurta", "KRT-BLK-L": "black kurta",
    "SNK-WHT-9": "white sneakers", "WCH-STL-42": "steel watch",
    "EAR-TWS-B": "wireless earbuds", "BAG-TAN-OS": "tan sling bag",
    "JNS-IND-32": "indigo jeans", "SRE-RED-OS": "red saree",
    "LMP-BRS-01": "brass lamp", "MAT-YOG-6": "yoga mat",
}
_CITIES = ["BLR", "MUM", "DEL", "HYD", "PNQ", "CHN", "JAI", "LKO", "IDR", "GHY"]
_FIRST = ["aarav", "diya", "vihaan", "ananya", "kabir", "isha", "rohan",
          "meera", "arjun", "sara", "dev", "nisha", "yash", "tara", "omkar"]
_LAST = ["sharma", "iyer", "khan", "reddy", "das", "mehta", "nair", "bose",
         "gill", "rao", "shetty", "verma", "joshi", "pillai", "chawla"]

#: Merchant descriptors as the *network* returns them: upper-cased,
#: truncated near 22 chars, prefix-mangled. This is the real reason a
#: dispute cannot be joined to an order by string equality.
_DESCRIPTOR_FORMS = [
    "RZP*{m}", "RAZORPAY*{m}", "RZPY {m} {c}", "{m}*RZP", "RZP {m}{c}",
]


@dataclass
class _Customer:
    customer_id: str
    email: str
    phone: str
    card_bin: str
    card_last4: str
    city: str


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------


class World:
    """Deterministic given ``config.seed``."""

    def __init__(self, config: WorldConfig | None = None):
        self.cfg = config or WorldConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._n = 0

    # -- small helpers ----------------------------------------------------

    def _uid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n:07d}"

    def _pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def _when(self, base: datetime, lo_h: float, hi_h: float) -> datetime:
        return base + timedelta(hours=float(self.rng.uniform(lo_h, hi_h)))

    def _cb_delay_days(self) -> float:
        """Time from loss to chargeback. Lognormal, clipped to scheme limits."""
        d = float(self.rng.lognormal(self.cfg.cb_log_mu, self.cfg.cb_log_sigma))
        return float(np.clip(d, 8.0, 150.0))

    # -- population -------------------------------------------------------

    def _customers(self) -> list[_Customer]:
        out = []
        for _ in range(self.cfg.n_customers):
            f, l = self._pick(_FIRST), self._pick(_LAST)
            n = int(self.rng.integers(1, 999))
            out.append(
                _Customer(
                    customer_id=self._uid("cust"),
                    email=f"{f}.{l}{n}@example.in",
                    phone=f"+9198{self.rng.integers(10**7, 10**8 - 1)}",
                    card_bin=str(self._pick(["414367", "552244", "601140",
                                             "459150", "512345"])),
                    card_last4=f"{self.rng.integers(0, 9999):04d}",
                    city=self._pick(_CITIES),
                )
            )
        # shared-card confuser: a few PANs belong to two customer ids
        n_shared = max(2, self.cfg.n_customers // 60)
        for i in range(n_shared):
            a, b = out[2 * i], out[2 * i + 1]
            out[2 * i + 1] = _Customer(b.customer_id, b.email, b.phone,
                                       a.card_bin, a.card_last4, a.city)
        return out

    def _descriptor(self, sku: str, city: str) -> str:
        """Statement descriptor, as the network would return it.

        Deterministic in (sku, city) on purpose. Drawing the form at random
        per order made the descriptor a near-unique key and handed the
        matcher a signal no real dispute carries: a merchant's descriptor is
        stable, which is exactly why it narrows candidates without
        identifying one.
        """
        m = sku.split("-")[0] + "FASHION"
        # zlib.crc32, not hash(): str hashing is salted per process, so
        # hash() would hand back a different descriptor on every run and
        # quietly destroy seed reproducibility.
        h = zlib.crc32(f"{sku}|{city}".encode())
        form = _DESCRIPTOR_FORMS[h % len(_DESCRIPTOR_FORMS)]
        return form.format(m=m, c=city).upper()[:22].strip()

    def _order(self, cust: _Customer, when: datetime,
               amount: int | None = None, sku: str | None = None) -> Order:
        amt = amount if amount is not None else int(
            self.rng.integers(self.cfg.order_min, self.cfg.order_max) // 100 * 100
        )
        method = PaymentMethod(
            ["card", "upi", "netbanking", "cod"][
                int(self.rng.choice(4, p=list(self.cfg.method_mix)))
            ]
        )
        s = sku or self._pick(_SKUS)
        oid = self._uid("ord")
        return Order(
            order_id=oid,
            customer_id=cust.customer_id,
            created_at=when,
            amount_paise=amt,
            cogs_paise=int(amt * self.cfg.cogs_ratio),
            method=method,
            payment_id=self._uid("pay"),
            email=cust.email,
            phone=cust.phone,
            sku=s,
            descriptor=self._descriptor(s, cust.city),
            card_bin=cust.card_bin if method is PaymentMethod.CARD else None,
            card_last4=cust.card_last4 if method is PaymentMethod.CARD else None,
            awb=f"AWB{self.rng.integers(10**9, 10**10 - 1)}",
        )

    # -- remediation emitters --------------------------------------------
    # Each returns events *without* ground truth; lineage is recorded by the
    # caller into GroundTruth.

    def _refund(self, o: Order, when: datetime, amount: int) -> RemediationEvent:
        return RemediationEvent(
            event_id=self._uid("rfnd"),
            channel=Channel.REFUND,
            occurred_at=when,
            value_paise=amount,
            merchant_cost_paise=amount,
            payment_id=o.payment_id,          # strong key: this one is easy
            order_id_hint=o.order_id,
            card_bin=o.card_bin,
            card_last4=o.card_last4,
        )

    def _dispute(self, o: Order, when: datetime, amount: int) -> RemediationEvent:
        """Carries no order_id and no payment_id -- by construction, because
        the card network does not have them. Only ARN, masked PAN, amount,
        timestamp and a mangled descriptor.

        Card rails only. A UPI collect or a COD parcel cannot produce a
        scheme chargeback; modelling one would hand the blocker a card tail
        that never existed and silently inflate dispute recall. UPI does
        have its own NPCI dispute path with different identifiers -- out of
        scope here, and noted as a limitation in the README."""
        assert o.is_card, f"dispute emitted on non-card order {o.order_id}"
        return RemediationEvent(
            event_id=self._uid("dspt"),
            channel=Channel.DISPUTE,
            occurred_at=when,
            value_paise=amount,
            merchant_cost_paise=amount + self.cfg.dispute_fee,
            arn=f"ARN{self.rng.integers(10**11, 10**12 - 1)}",
            card_bin=o.card_bin,
            card_last4=o.card_last4,
            descriptor_text=o.descriptor,
        )

    def _goodwill_amount(self, o: Order) -> int:
        """One draw, used by *every* goodwill emission.

        This exists because of a leak found during evaluation. Duplicate
        patterns used to credit the exact order value while confusers
        credited partials, so `weak_amount_exact` separated the classes
        perfectly and the linkage model scored 100% on duplicated orders
        against 85% on single ones. It had learned the generator, not the
        problem. An event's amount must not carry information about whether
        its order ends up duplicated, so the draw lives here and nowhere
        else.
        """
        if self.rng.random() < 0.55:
            return o.amount_paise                    # full make-whole
        frac = float(self.rng.uniform(0.30, 0.95))   # partial compensation
        return max(10_000, int(o.amount_paise * frac) // 100 * 100)

    def _goodwill(self, o: Order, when: datetime, amount: int,
                  kind: LossKind) -> RemediationEvent:
        item = _SKU_NAME.get(o.sku, "item")
        mention_order = self.rng.random() < 0.35   # agents often omit it
        body = self._ticket_text(kind, item, o.order_id if mention_order else None)
        return RemediationEvent(
            event_id=self._uid("gwil"),
            channel=Channel.GOODWILL,
            occurred_at=when,
            value_paise=amount,
            merchant_cost_paise=amount,
            email=o.email if self.rng.random() < 0.8 else None,
            phone=o.phone if self.rng.random() < 0.5 else None,
            order_id_hint=o.order_id if mention_order and self.rng.random() < 0.4 else None,
            free_text=body,
        )

    def _replacement(self, o: Order, when: datetime) -> RemediationEvent:
        """Customer is made whole at retail value; merchant pays COGS+ship."""
        return RemediationEvent(
            event_id=self._uid("repl"),
            channel=Channel.REPLACEMENT,
            occurred_at=when,
            value_paise=o.amount_paise,
            merchant_cost_paise=o.cogs_paise + self.cfg.replacement_ship,
            awb=f"AWB{self.rng.integers(10**9, 10**10 - 1)}",
            order_id_hint=o.order_id if self.rng.random() < 0.75 else None,
        )

    def _ticket_text(self, kind: LossKind, item: str, oid: str | None) -> str:
        ref = f" order {oid}" if oid else ""
        opts = {
            LossKind.NOT_DELIVERED: [
                f"cust says{ref} {item} never arrived, courier shows delivered. issued credit",
                f"nothing received{ref}. checked AWB, marked delivered but cust denies. goodwill given",
                f"{item} not delivered{ref}, cust very upset, escalated. approved credit",
            ],
            LossKind.DAMAGED: [
                f"{item} arrived damaged{ref}, photos attached. credited",
                f"broken on arrival{ref} - {item}. partial credit for inconvenience",
            ],
            LossKind.WRONG_ITEM: [
                f"wrong size sent{ref}, cust got different {item}. credited difference",
                f"received other product instead of {item}{ref}. adjusted",
            ],
            LossKind.NOT_AS_DESCRIBED: [
                f"{item} colour not as shown on site{ref}. goodwill issued",
                f"quality complaint{ref} on {item}. small credit to close ticket",
            ],
            LossKind.UNAUTHORISED: [
                f"cust claims card used without consent{ref}. refunded pending review",
                f"says did not place this{ref}. credited and flagged",
            ],
        }
        return self._pick(opts[kind])

    # -- lineage patterns -------------------------------------------------

    def _emit_single(self, o: Order, loss: LossEvent) -> list[RemediationEvent]:
        """Correct handling: one loss, one making-whole. The negative class."""
        r = self.rng.random()
        t = self._when(loss.opened_at, 2, 72)
        if r < 0.55:
            return [self._refund(o, t, o.amount_paise)]
        if r < 0.75:
            return [self._replacement(o, t)]
        if r < 0.90:
            if o.is_card:
                return [self._dispute(o, loss.opened_at
                                      + timedelta(days=self._cb_delay_days()),
                                      o.amount_paise)]
            return [self._refund(o, t, o.amount_paise)]
        return [self._goodwill(o, t, self._goodwill_amount(o), loss.kind)]

    def _emit_split(self, o: Order, loss: LossEvent) -> list[RemediationEvent]:
        """CONFUSER: one loss legitimately settled across two partial events.
        Sums to <= order value, so it is *not* a duplicate -- but it trips
        every 'two events on one order' heuristic."""
        part = int(o.amount_paise * float(self.rng.uniform(0.55, 0.7)) // 100 * 100)
        rest = o.amount_paise - part
        t = self._when(loss.opened_at, 2, 48)
        return [
            self._refund(o, t, part),
            self._goodwill(o, self._when(t, 6, 96), rest, loss.kind),
        ]  # rest = order value - part, so the pair sums to exactly 1.0x

    def _emit_make_good(self, o: Order, loss: LossEvent) -> list[RemediationEvent]:
        """CONFUSER: replacement plus a token apology credit. Two channels,
        two events, still legitimate."""
        t = self._when(loss.opened_at, 2, 48)
        token = int(min(20_000, o.amount_paise * 0.05) // 100 * 100)
        return [
            self._replacement(o, t),
            self._goodwill(o, self._when(t, 12, 120), token, loss.kind),
        ]

    def _emit_duplicate(self, o: Order, loss: LossEvent
                        ) -> tuple[list[RemediationEvent], str]:
        """The positive class. Five realistic ways a merchant pays twice."""
        card_only = ["refund_then_chargeback", "chargeback_then_goodwill",
                     "partial_refund_then_full_chargeback"]
        any_rail = ["replacement_then_refund", "double_goodwill"]
        pattern = self._pick(card_only + any_rail if o.is_card else any_rail)
        t0 = self._when(loss.opened_at, 2, 72)
        cb_at = loss.opened_at + timedelta(days=self._cb_delay_days())

        if pattern == "refund_then_chargeback":
            ev = [self._refund(o, t0, o.amount_paise),
                  self._dispute(o, cb_at, o.amount_paise)]
        elif pattern == "chargeback_then_goodwill":
            ev = [self._dispute(o, cb_at, o.amount_paise),
                  self._goodwill(o, self._when(cb_at, 24, 240),
                                 self._goodwill_amount(o), loss.kind)]
        elif pattern == "replacement_then_refund":
            ev = [self._replacement(o, t0),
                  self._refund(o, self._when(t0, 48, 480), o.amount_paise)]
        elif pattern == "partial_refund_then_full_chargeback":
            part = int(o.amount_paise * 0.5 // 100 * 100)
            ev = [self._refund(o, t0, part),
                  self._dispute(o, cb_at, o.amount_paise)]
        else:  # double_goodwill -- two agents, two tickets, one complaint
            # Two independent draws. Their *sum* crosses the duplicate line;
            # neither one on its own looks any different from a legitimate
            # single credit, which is what makes this pattern hard.
            ev = [self._goodwill(o, t0, self._goodwill_amount(o), loss.kind),
                  self._goodwill(o, self._when(t0, 18, 200),
                                 self._goodwill_amount(o), loss.kind)]
        return ev, pattern

    # -- assembly ---------------------------------------------------------

    def build(self) -> Dataset:
        cfg = self.cfg
        rng = self.rng
        horizon = cfg.start + timedelta(days=cfg.horizon_days)
        customers = self._customers()

        orders: list[Order] = []
        events: list[RemediationEvent] = []
        truth = GroundTruth()
        patterns: dict[str, str] = {}

        def attach(o: Order, loss: LossEvent, evs: list[RemediationEvent]):
            for e in evs:
                events.append(e)
                truth.event_to_order[e.event_id] = o.order_id
                truth.event_to_loss[e.event_id] = loss.loss_id
            truth.losses[loss.loss_id] = loss

        # Explicit categorical dispatch. An earlier version used nested
        # `roll < threshold` comparisons and two thresholds collided
        # exactly, silently starving the duplicate branch to zero. Drawing
        # a labelled category makes that class of bug impossible.
        p_twin = cfg.confuser_rate * 0.25
        p_legit = cfg.confuser_rate * 0.50
        p_loss = cfg.loss_rate
        cats = ["twin", "loss", "legit_multi", "clean"]
        probs = [p_twin, p_loss, p_legit, 1.0 - p_twin - p_loss - p_legit]
        assert min(probs) > 0, f"category probabilities must be positive: {probs}"

        for _ in range(cfg.n_orders):
            cust = self._pick(customers)
            when = cfg.start + timedelta(
                hours=float(rng.uniform(0, cfg.days * 24))
            )
            o = self._order(cust, when)
            orders.append(o)

            cat = cats[int(rng.choice(len(cats), p=probs))]

            # --- twin-order confuser: a second, near-identical order ------
            if cat == "twin":
                twin_amt = o.amount_paise + int(rng.integers(-200, 200)) * 100
                twin = self._order(cust, self._when(when, 0.5, 8),
                                   amount=max(10_000, twin_amt), sku=o.sku)
                orders.append(twin)
                for parent in (o, twin):
                    loss = LossEvent(self._uid("loss"), parent.order_id,
                                     LossKind(self._pick(list(LossKind))),
                                     self._when(parent.created_at, 24, 400))
                    attach(parent, loss, self._emit_single(parent, loss))
                    truth.confuser_orders.add(parent.order_id)
                continue

            if cat == "loss":
                kind = LossKind(self._pick(list(LossKind)))
                loss = LossEvent(self._uid("loss"), o.order_id, kind,
                                 self._when(when, 24, 500))
                if rng.random() < cfg.duplicate_rate:
                    evs, pat = self._emit_duplicate(o, loss)
                    patterns[o.order_id] = pat
                    attach(o, loss, evs)
                else:
                    attach(o, loss, self._emit_single(o, loss))
            elif cat == "legit_multi":
                # legitimate multi-event settlements: split / make-good
                kind = LossKind(self._pick(list(LossKind)))
                loss = LossEvent(self._uid("loss"), o.order_id, kind,
                                 self._when(when, 24, 400))
                emit = (self._emit_split if rng.random() < 0.6
                        else self._emit_make_good)
                attach(o, loss, emit(o, loss))
                truth.confuser_orders.add(o.order_id)

        # --- serial-complainer confuser ---------------------------------
        # One customer, many orders, each settled once and legitimately.
        for _ in range(max(3, cfg.n_customers // 90)):
            cust = self._pick(customers)
            for _ in range(int(rng.integers(4, 8))):
                when = cfg.start + timedelta(hours=float(rng.uniform(0, cfg.days * 24)))
                o = self._order(cust, when)
                orders.append(o)
                loss = LossEvent(self._uid("loss"), o.order_id,
                                 LossKind(self._pick(list(LossKind))),
                                 self._when(when, 24, 300))
                attach(o, loss, self._emit_single(o, loss))
                truth.confuser_orders.add(o.order_id)

        # --- right-censoring at the horizon ------------------------------
        # Events after the cutoff have not been observed yet. Dropping them
        # is what creates censored labels for hazard.py -- and what makes a
        # naive fraud rate computed on this data optimistically biased.
        visible = [e for e in events if e.occurred_at <= horizon]
        dropped = {e.event_id for e in events if e.occurred_at > horizon}
        for eid in dropped:
            truth.event_to_order.pop(eid, None)
            truth.event_to_loss.pop(eid, None)
        events = visible

        # --- derive labels from what was actually emitted ----------------
        # Never assert a label the data does not support.
        by_order: dict[str, list[RemediationEvent]] = {}
        for e in events:
            by_order.setdefault(truth.event_to_order[e.event_id], []).append(e)

        amounts = {o.order_id: o.amount_paise for o in orders}
        for oid, evs in by_order.items():
            returned = sum(e.value_paise for e in evs)
            over = returned - amounts[oid]
            if returned >= amounts[oid] * cfg.duplicate_threshold:
                truth.duplicated_orders.add(oid)
                truth.over_remediation_paise[oid] = over
                truth.confuser_orders.discard(oid)
            else:
                truth.over_remediation_paise[oid] = 0

        events.sort(key=lambda e: e.occurred_at)
        orders.sort(key=lambda o: o.created_at)
        truth.duplicate_patterns = {
            k: v for k, v in patterns.items() if k in truth.duplicated_orders
        }
        return Dataset(orders=orders, events=events, truth=truth,
                       horizon=horizon, seed=cfg.seed)
