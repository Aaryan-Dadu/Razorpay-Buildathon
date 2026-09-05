"""Pairwise (event, order) features for the linkage model.

Design rule: every feature here must be computable from data a merchant
actually holds at decision time. No feature may read GroundTruth, and none
may read the future -- a feature like "this order was later disputed" would
score beautifully offline and be unavailable in production.

Features split into three families, which is also how the ablation in
report.py carves them up:

    strong_*    exact identifier agreement (payment_id, order hint, AWB)
    weak_*      customer- and money-level agreement (card tail, amount, time)
    text_*      evidence recovered from prose and mangled descriptors
"""

from __future__ import annotations

import math
import re
from datetime import timedelta

from rapidfuzz import fuzz

from ..schema import Channel, Order, RemediationEvent

FEATURE_NAMES: list[str] = [
    # strong
    "strong_payment_id", "strong_order_hint", "strong_awb",
    # weak: identity
    "weak_email", "weak_phone", "weak_card_tail", "weak_card_bin",
    # weak: money
    "weak_amount_exact", "weak_amount_ratio", "weak_amount_ratio_dist",
    "weak_amount_plausible_partial", "weak_amount_over",
    # weak: time
    "weak_days_gap", "weak_log_days_gap", "weak_gap_in_dispute_window",
    "weak_gap_under_7d",
    # text
    "text_descriptor_sim", "text_mentions_order_id", "text_mentions_sku",
    # context
    "ctx_block_size", "ctx_amount_rank",
    "ctx_ch_refund", "ctx_ch_dispute", "ctx_ch_goodwill", "ctx_ch_replacement",
]

_SKU_WORDS = {
    "KRT-BLU-M": ("blue", "kurta"), "KRT-BLK-L": ("black", "kurta"),
    "SNK-WHT-9": ("white", "sneakers"), "WCH-STL-42": ("steel", "watch"),
    "EAR-TWS-B": ("wireless", "earbuds"), "BAG-TAN-OS": ("tan", "sling"),
    "JNS-IND-32": ("indigo", "jeans"), "SRE-RED-OS": ("red", "saree"),
    "LMP-BRS-01": ("brass", "lamp"), "MAT-YOG-6": ("yoga", "mat"),
}

#: Scheme dispute filing window, used as a soft feature not a hard filter.
_DISPUTE_LO, _DISPUTE_HI = 8, 150


def _sim(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def pair_features(e: RemediationEvent, o: Order, *, block_size: int = 1,
                  amount_rank: int = 0) -> list[float]:
    """Feature vector for one candidate pairing. Order matches FEATURE_NAMES."""
    gap: timedelta = e.occurred_at - o.created_at
    days = gap.total_seconds() / 86400.0

    ratio = e.value_paise / o.amount_paise if o.amount_paise else 0.0
    text = " ".join(filter(None, [e.free_text, e.descriptor_text])).lower()

    return [
        # -- strong -----------------------------------------------------
        1.0 if (e.payment_id and e.payment_id == o.payment_id) else 0.0,
        1.0 if (e.order_id_hint and e.order_id_hint == o.order_id) else 0.0,
        1.0 if (e.awb and o.awb and e.awb == o.awb) else 0.0,
        # -- weak: identity ---------------------------------------------
        1.0 if (e.email and e.email == o.email) else 0.0,
        1.0 if (e.phone and e.phone == o.phone) else 0.0,
        1.0 if (e.card_last4 and e.card_last4 == o.card_last4) else 0.0,
        1.0 if (e.card_bin and e.card_bin == o.card_bin) else 0.0,
        # -- weak: money ------------------------------------------------
        1.0 if abs(e.value_paise - o.amount_paise) <= 100 else 0.0,
        min(ratio, 3.0),
        abs(1.0 - min(ratio, 3.0)),
        1.0 if 0.10 <= ratio <= 0.95 else 0.0,
        1.0 if ratio > 1.02 else 0.0,
        min(days, 400.0),
        math.log1p(max(days, 0.0)),
        1.0 if _DISPUTE_LO <= days <= _DISPUTE_HI else 0.0,
        1.0 if 0 <= days <= 7 else 0.0,
        # -- text -------------------------------------------------------
        _sim(e.descriptor_text, o.descriptor),
        1.0 if (o.order_id.lower() in text) else 0.0,
        (sum(w in text for w in _SKU_WORDS.get(o.sku, ()))
         / max(1, len(_SKU_WORDS.get(o.sku, ("x",))))),
        # -- context ----------------------------------------------------
        math.log1p(block_size),
        math.log1p(amount_rank),
        1.0 if e.channel is Channel.REFUND else 0.0,
        1.0 if e.channel is Channel.DISPUTE else 0.0,
        1.0 if e.channel is Channel.GOODWILL else 0.0,
        1.0 if e.channel is Channel.REPLACEMENT else 0.0,
    ]


def feature_family(name: str) -> str:
    return name.split("_", 1)[0]


assert len(FEATURE_NAMES) == len(
    pair_features(
        RemediationEvent("e", Channel.REFUND, __import__("datetime").datetime(2026, 1, 2), 100, 100),
        Order("o", "c", __import__("datetime").datetime(2026, 1, 1), 100, 40,
              __import__("sentinel.schema", fromlist=["PaymentMethod"]).PaymentMethod.CARD,
              "p", "e@x.in", "+91", "KRT-BLU-M", "RZP*X"),
    )
), "FEATURE_NAMES out of sync with pair_features"
