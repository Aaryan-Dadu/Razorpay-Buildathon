"""Tests for the properties that make the evaluation trustworthy.

These are deliberately not "does it run" tests. Each one guards an
invariant that, if broken, would silently inflate a headline number --
which is the failure mode that matters for a system whose whole claim is
honest measurement.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.gate import CostModel, Gate, Verdict
from sentinel.generate import World, WorldConfig
from sentinel.hazard import (MIN_EXPOSURE_TO_PROJECT, N_PERIODS, PERIOD_DAYS,
                             build_histories, cohort_bias, kaplan_meier,
                             person_period)
from sentinel.ledger import Ledger, LedgerEntry
from sentinel.metrics import realised_fp_cost, score
from sentinel.resolve.blocking import MAX_LOOKBACK, BlockingIndex
from sentinel.resolve.matcher import (LinkageModel, Method, Resolution,
                                      deterministic_link, order_temporal_split,
                                      resolve_all)
from sentinel.schema import Channel, PaymentMethod, rupees


@pytest.fixture(scope="module")
def ds():
    return World(WorldConfig(n_orders=4000, seed=13)).build()


# --------------------------------------------------------------------------
# generator: the answer key must be trustworthy
# --------------------------------------------------------------------------

def test_generation_is_reproducible():
    a = World(WorldConfig(n_orders=1500, seed=3)).build()
    b = World(WorldConfig(n_orders=1500, seed=3)).build()
    assert [o.order_id for o in a.orders] == [o.order_id for o in b.orders]
    assert [e.event_id for e in a.events] == [e.event_id for e in b.events]
    assert a.truth.duplicated_orders == b.truth.duplicated_orders
    # Descriptors must survive process restarts: they are built from a
    # stable crc32, not from salted str hash().
    assert [o.descriptor for o in a.orders] == [o.descriptor for o in b.orders]


def test_different_seeds_give_different_worlds():
    a = World(WorldConfig(n_orders=1500, seed=3)).build()
    b = World(WorldConfig(n_orders=1500, seed=4)).build()
    assert a.truth.duplicated_orders != b.truth.duplicated_orders


def test_duplicate_label_is_derived_not_asserted(ds):
    """Every flagged order must actually have been over-remediated.

    The generator intends certain patterns to duplicate, but the label is
    computed from emitted values. If emission and intent ever disagree, the
    data wins.
    """
    amounts = {o.order_id: o.amount_paise for o in ds.orders}
    by_order: dict[str, int] = {}
    for e in ds.events:
        oid = ds.truth.event_to_order[e.event_id]
        by_order[oid] = by_order.get(oid, 0) + e.value_paise
    for oid in ds.truth.duplicated_orders:
        assert by_order[oid] >= amounts[oid] * 1.25, oid


def test_duplicates_and_confusers_are_disjoint(ds):
    assert not (ds.truth.duplicated_orders & ds.truth.confuser_orders)


def test_confusers_are_actually_hard(ds):
    """A confuser must trip the naive rule it exists to defeat."""
    multi = {oid for oid in ds.truth.confuser_orders
             if len(ds.truth.cluster_of(oid)) >= 2}
    assert len(multi) > 0.4 * len(ds.truth.confuser_orders), (
        "most confusers should carry >=2 events, else they defeat nothing")


def test_disputes_only_on_card_orders(ds):
    """A UPI collect or a COD parcel cannot produce a scheme chargeback.
    Emitting one would hand the blocker a card tail that never existed."""
    by_id = ds.orders_by_id()
    for e in ds.events:
        if e.channel is Channel.DISPUTE:
            o = by_id[ds.truth.event_to_order[e.event_id]]
            assert o.method is PaymentMethod.CARD, o.order_id
            assert e.card_last4 is not None


def test_disputes_carry_no_strong_key(ds):
    """The premise of the whole project: a chargeback cannot be joined."""
    for e in ds.events:
        if e.channel is Channel.DISPUTE:
            assert e.payment_id is None and e.order_id_hint is None


def test_censoring_actually_drops_events(ds):
    """Events past the horizon must be absent from both stream and truth."""
    for e in ds.events:
        assert e.occurred_at <= ds.horizon
    assert set(ds.truth.event_to_order) == {e.event_id for e in ds.events}


# --------------------------------------------------------------------------
# blocking
# --------------------------------------------------------------------------

def test_blocking_respects_causality(ds):
    """No candidate may post-date the event that supposedly remediates it."""
    idx = BlockingIndex(ds.orders)
    for e in ds.events[:400]:
        for o in idx.candidates(e):
            assert o.created_at <= e.occurred_at
            assert e.occurred_at - o.created_at <= MAX_LOOKBACK


def test_blocking_recall_is_high_enough_to_be_a_ceiling(ds):
    from sentinel.resolve.blocking import blocking_recall
    r = blocking_recall(BlockingIndex(ds.orders), ds.events,
                        ds.truth.event_to_order)
    assert r["blocking_recall"] > 0.90
    assert r["mean_block_size"] < 60


# --------------------------------------------------------------------------
# matcher
# --------------------------------------------------------------------------

def test_deterministic_link_is_exact_not_fuzzy(ds):
    """Stage 1 must never guess. Every link it makes has to be correct."""
    idx = BlockingIndex(ds.orders)
    checked = 0
    for e in ds.events:
        hit = deterministic_link(e, idx.candidates(e))
        if hit is not None:
            assert hit.order_id == ds.truth.event_to_order[e.event_id]
            checked += 1
    assert checked > 100


def test_temporal_split_has_no_lineage_leak(ds):
    tr, te, tr_ev, cutoff = order_temporal_split(
        ds.orders, ds.events, ds.truth.event_to_order, 0.7)
    assert not (tr & te)
    for e in tr_ev:
        assert ds.truth.event_to_order[e.event_id] in tr, (
            "a training event belongs to a test order -- that is the leak "
            "this split exists to prevent")


def test_model_abstains_rather_than_guessing_badly(ds):
    idx = BlockingIndex(ds.orders)
    tr, te, tr_ev, _ = order_temporal_split(
        ds.orders, ds.events, ds.truth.event_to_order, 0.7)
    m = LinkageModel(0).fit(idx, tr_ev, ds.truth.event_to_order)
    res = resolve_all(idx, ds.events, m)
    committed = [r for r in res if r.linked]
    correct = sum(1 for r in committed
                  if r.order_id == ds.truth.event_to_order.get(r.event_id))
    # Precision on committed links must beat coverage: abstention should be
    # buying accuracy, not just hiding failures.
    assert correct / len(committed) > 0.90
    assert any(r.method is Method.ABSTAINED for r in res)


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------

def _entry(events, order_value=100_000):
    en = LedgerEntry("ord_x", order_value)
    for e in events:
        en.events.append(e)
        en.confidences.append(1.0)
    return en


def test_state_at_excludes_the_future():
    """The gate must never be shown an event that had not happened yet."""
    from sentinel.schema import RemediationEvent
    t0 = datetime(2026, 3, 1)
    evs = [RemediationEvent(f"e{i}", Channel.REFUND, t0 + timedelta(days=i),
                            50_000, 50_000) for i in range(4)]
    led = Ledger(entries={"ord_x": _entry(evs)})
    at = led.state_at("ord_x", t0 + timedelta(days=2))
    assert [e.event_id for e in at.events] == ["e0", "e1"]
    assert at.returned_paise == 100_000


def test_exposure_ratio_and_over_paise_are_integer_exact():
    from sentinel.schema import RemediationEvent
    t0 = datetime(2026, 3, 1)
    evs = [RemediationEvent("a", Channel.REFUND, t0, 100_000, 100_000),
           RemediationEvent("b", Channel.DISPUTE, t0, 100_000, 250_000)]
    en = _entry(evs, order_value=100_000)
    assert en.returned_paise == 200_000
    assert en.over_paise == 100_000
    assert en.exposure_ratio == 2.0
    # merchant cost is not the same number as customer value
    assert en.merchant_cost_paise == 350_000


def test_unlinked_events_go_to_exceptions_not_silence(ds):
    res = [Resolution(e.event_id, None, 0.0, Method.ABSTAINED)
           for e in ds.events[:50]]
    led = Ledger.build(ds.orders, ds.events[:50], res)
    assert len(led.exceptions) == 50
    assert not led.entries


# --------------------------------------------------------------------------
# gate economics
# --------------------------------------------------------------------------

def test_realised_fp_cost_ignores_model_confidence():
    """The bill for a wrong block cannot be scaled by the gate's own belief.

    Pricing mistakes with the confidence of the thing that made them is
    circular: a confidently wrong gate would report a near-zero
    false-positive cost.
    """
    c = CostModel()
    cost = realised_fp_cost(100_000, c)
    expected = int(c.support_touch_paise
                   + c.p_escalate_if_blocked * (100_000 + c.dispute_fee_paise)
                   + c.churn_paise)
    assert cost == expected
    assert cost > c.churn_paise


def test_gate_blocks_an_obvious_double_payment():
    from sentinel.schema import RemediationEvent
    t0 = datetime(2026, 3, 1)
    en = _entry([RemediationEvent("a", Channel.REFUND, t0, 100_000, 100_000)],
                order_value=100_000)
    d = Gate().evaluate(en, 100_000, Channel.GOODWILL)
    assert d.verdict is Verdict.BLOCK
    assert d.ledger_ratio_after == 2.0
    assert d.rationale


def test_gate_allows_a_legitimate_partial_top_up():
    """Refund of part, then the rest. Sums to 1.0x -- not a duplicate."""
    from sentinel.schema import RemediationEvent
    t0 = datetime(2026, 3, 1)
    en = _entry([RemediationEvent("a", Channel.REFUND, t0, 60_000, 60_000)],
                order_value=100_000)
    d = Gate().evaluate(en, 40_000, Channel.GOODWILL)
    assert d.verdict is Verdict.ALLOW


def test_every_decision_carries_its_arithmetic():
    from sentinel.schema import RemediationEvent
    t0 = datetime(2026, 3, 1)
    en = _entry([RemediationEvent("a", Channel.REFUND, t0, 100_000, 100_000)])
    d = Gate().evaluate(en, 100_000, Channel.GOODWILL)
    assert "already returned" in d.explain()
    assert str(d.verdict).upper() in d.explain()


# --------------------------------------------------------------------------
# hazard / censoring
# --------------------------------------------------------------------------

def test_person_period_rows_stop_at_observation(ds):
    """An order observed 20 days contributes 2 rows, not 10. Otherwise it
    is silently counted as having survived periods nobody watched."""
    H = build_histories(ds, ds.truth.event_to_order)
    card = [h for h in H.values() if h.order.is_card]
    X, y, ids = person_period(card)
    counts: dict[str, int] = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    for h in card[:200]:
        n = counts.get(h.order.order_id, 0)
        if h.dispute_day is None:
            expected = min(int(h.observed_days // PERIOD_DAYS), N_PERIODS - 1) + 1
        else:
            expected = min(int(h.dispute_day // PERIOD_DAYS), N_PERIODS - 1) + 1
        assert n == expected, h.order.order_id


def test_survival_is_monotone_non_increasing(ds):
    H = build_histories(ds, ds.truth.event_to_order)
    _, surv = kaplan_meier(list(H.values()))
    assert all(surv[i] >= surv[i + 1] - 1e-12 for i in range(len(surv) - 1))
    assert 0.0 <= surv[-1] <= 1.0


def test_censoring_correction_is_not_below_naive(ds):
    """Accounting for orders still inside the dispute window can only raise
    the estimate, never lower it."""
    from sentinel.hazard import censoring_bias
    H = build_histories(ds, ds.truth.event_to_order)
    b = censoring_bias(list(H.values()))
    assert b["km_corrected_rate"] >= b["naive_rate"] - 1e-9


def test_low_exposure_cohorts_refuse_to_project(ds):
    H = build_histories(ds, ds.truth.event_to_order)
    for row in cohort_bias(list(H.values())):
        if row["exposure_seen"] < MIN_EXPOSURE_TO_PROJECT:
            assert row["projectable"] == 0.0
            assert row["projected_rate"] != row["projected_rate"]  # NaN


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

def test_score_handles_empty_and_perfect():
    assert score(set(), {"a"}, {"a", "b"}).recall == 0.0
    s = score({"a"}, {"a"}, {"a", "b"})
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0


def test_score_is_confined_to_the_universe():
    """A detector cannot get credit for flagging outside the test set."""
    s = score({"a", "zzz"}, {"a"}, {"a", "b"})
    assert s.fp == 0 and s.tp == 1


def test_rupee_formatting_uses_indian_grouping():
    assert rupees(0) == "Rs 0.00"
    assert rupees(123456789) == "Rs 12,34,567.89"
    assert rupees(-4999) == "-Rs 49.99"


def test_llm_stage_degrades_to_abstention_without_credentials():
    from sentinel.resolve.llm import LLMResolver
    r = LLMResolver(enabled=False)
    assert not r.available and r.disabled_reason
