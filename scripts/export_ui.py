#!/usr/bin/env python3
"""Export a real slice of a run for the UI. No synthetic-for-display data:
every entry, decision and rationale here came out of the pipeline."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.gate import CostModel, Gate, replay
from sentinel.generate import World, WorldConfig
from sentinel.hazard import (HazardModel, build_histories, censoring_bias,
                             cohort_bias, kaplan_meier, PERIOD_DAYS)
from sentinel.ledger import Ledger
from sentinel.metrics import score
from sentinel.resolve.blocking import BlockingIndex
from sentinel.resolve.matcher import (LinkageModel, Method, order_temporal_split,
                                      resolve_all)

SEED, N, TH = 7, 25000, 1.25
ds = World(WorldConfig(n_orders=N, seed=SEED)).build()
idx = BlockingIndex(ds.orders); T = ds.truth
tr_ids, te_ids, tr_ev, _ = order_temporal_split(ds.orders, ds.events, T.event_to_order, .7)
model = LinkageModel(SEED).fit(idx, tr_ev, T.event_to_order)
res = resolve_all(idx, ds.events, model)
led = Ledger.build(ds.orders, ds.events, res)
by_ord = ds.orders_by_id()

pos = T.duplicated_orders & te_ids
flagged = {o for o, e in led.entries.items() if e.exposure_ratio >= TH} & te_ids
S = score(flagged, pos, te_ids)

def entry_json(oid, verdict):
    en = led.entries[oid]; o = by_ord[oid]
    return {
        "order_id": oid, "verdict": verdict,
        "customer": o.customer_id, "sku": o.sku, "method": str(o.method),
        "placed_at": o.created_at.isoformat(),
        "order_value": en.order_value_paise,
        "returned": en.returned_paise,
        "merchant_cost": en.merchant_cost_paise,
        "ratio": round(en.exposure_ratio, 3),
        "pattern": T.duplicate_patterns.get(oid),
        "is_confuser": oid in T.confuser_orders,
        "events": [{
            "id": e.event_id, "channel": str(e.channel),
            "at": e.occurred_at.isoformat(),
            "days_after_order": round((e.occurred_at - o.created_at).total_seconds()/86400, 1),
            "value": e.value_paise, "cost": e.merchant_cost_paise,
            "conf": round(c, 3),
            "note": e.free_text, "descriptor": e.descriptor_text,
            "has_strong_key": bool(e.payment_id or e.order_id_hint or e.awb),
        } for e, c in zip(en.events, en.confidences)],
    }

cases = []
tp = sorted(flagged & pos, key=lambda o: -led.entries[o].over_paise)
# Lead with a chargeback case: the 8-150 day lag is the whole reason this
# loss is invisible, and it only shows on a timeline that spans it.
cb = [o for o in tp if any(e.channel.value == "dispute" for e in led.entries[o].events)]
lead = cb[:2]
for oid in lead + [o for o in tp if o not in lead][:4]:
    cases.append(entry_json(oid, "caught"))
for oid in sorted(pos - flagged)[:2]: cases.append(entry_json(oid, "missed"))
for oid in sorted(flagged - pos)[:2]: cases.append(entry_json(oid, "false_positive"))
# confusers correctly left alone: multi-event, legitimately under the line
clean = [o for o in (T.confuser_orders & te_ids) - flagged
         if o in led.entries and len(led.entries[o].events) >= 2]
for oid in sorted(clean, key=lambda o: -led.entries[o].exposure_ratio)[:4]:
    cases.append(entry_json(oid, "correctly_cleared"))

H = build_histories(ds, {r.event_id: r.order_id for r in res if r.linked})
haz, surv = kaplan_meier(list(H.values()))
hz = HazardModel(SEED).fit([h for h in H.values() if h.order.order_id in tr_ids])
gate = Gate(CostModel(), TH)
test_led = Ledger(entries={o: e for o, e in led.entries.items() if o in te_ids})
decs = replay(test_led, gate, H, hz)
dec_json = [{
    "order_id": d.order_id, "verdict": str(d.verdict),
    "p_duplicate": round(d.p_duplicate, 3),
    "allow_cost": d.expected_allow_cost_paise,
    "block_cost": d.expected_block_cost_paise,
    "ratio_before": round(d.ledger_ratio_before, 2),
    "ratio_after": round(d.ledger_ratio_after, 2),
    "hazard": round(d.hazard_future_dispute, 4),
    "rationale": d.rationale,
    "was_duplicate": d.order_id in T.duplicated_orders,
} for d in sorted(decs, key=lambda x: -x.expected_allow_cost_paise)[:8]]

ch_keys = {}
for e in ds.events:
    k = str(e.channel)
    d = ch_keys.setdefault(k, {"total": 0, "no_strong_key": 0})
    d["total"] += 1
    if not (e.payment_id or e.order_id_hint or e.awb): d["no_strong_key"] += 1

out = {
    "meta": {"seed": SEED, "orders": len(ds.orders), "events": len(ds.events),
             "duplicates": len(T.duplicated_orders), "confusers": len(T.confuser_orders),
             "test_orders": len(te_ids), "test_positives": len(pos)},
    "headline": {"precision": S.precision, "recall": S.recall, "f1": S.f1,
                 "tp": S.tp, "fp": S.fp, "fn": S.fn},
    "channel_keys": ch_keys,
    "cases": cases,
    "decisions": dec_json,
    "km": {"period_days": PERIOD_DAYS, "hazard": [round(float(x), 5) for x in haz],
           "survival": [round(float(x), 5) for x in surv]},
    "censoring": censoring_bias(list(H.values())),
    "cohorts": cohort_bias(list(H.values())),
    "exceptions": len(led.exceptions),
}
# NaN is not valid JSON and JSON.parse rejects it. Cohorts that refuse to
# project emit NaN by design, so map it to null on the way out.
import math
def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    return x
Path("reports/ui_data.json").write_text(
    json.dumps(clean(out), indent=1, default=str, allow_nan=False))
print("cases:", len(cases), "decisions:", len(dec_json))
print("headline:", {k: round(v,4) if isinstance(v,float) else v for k,v in out["headline"].items()})
print("channel_keys:", json.dumps(ch_keys))
print("bytes:", Path("reports/ui_data.json").stat().st_size)
