#!/usr/bin/env python3
"""Export a real slice of a run for the UI. No synthetic-for-display data:
every entry, decision and rationale here came out of the pipeline."""
from __future__ import annotations
import json, math, sys
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.gate import CostModel, Gate, Verdict, replay
from sentinel.generate import World, WorldConfig
from sentinel.hazard import (HazardModel, build_histories, censoring_bias,
                             cohort_bias, kaplan_meier, PERIOD_DAYS)
from sentinel.ledger import Ledger
from sentinel.metrics import realised_fp_cost, score
from sentinel.resolve.blocking import BlockingIndex
from sentinel.schema import Channel
from sentinel.resolve.matcher import (LinkageModel, Method, deterministic_link,
                                      order_temporal_split,
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
cm = CostModel()
gate = Gate(cm, TH)
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

# ---------------------------------------------------------------------------
# interactive payloads
# ---------------------------------------------------------------------------

ratios = {o: e.exposure_ratio for o, e in led.entries.items()}
over   = {o: e.over_paise for o, e in led.entries.items()}
cost   = {o: e.merchant_cost_paise for o, e in led.entries.items()}

# --- 1. threshold sweep, priced -------------------------------------------
sweep = []
for i in range(46):
    t = round(1.02 + 0.03 * i, 2)
    flag = {o for o, r in ratios.items() if r >= t} & te_ids
    tp, fp = flag & pos, flag - pos
    prevented = sum(int(over.get(o, 0) * (1 - cm.natural_recovery_rate)) for o in tp)
    fpbill = sum(realised_fp_cost(cost.get(o, 0), cm) for o in fp)
    p = len(tp) / len(flag) if flag else 1.0
    r = len(tp) / len(pos) if pos else 0.0
    sweep.append({"t": t, "p": round(p, 4), "r": round(r, 4),
                  "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
                  "tp": len(tp), "fp": len(fp),
                  "net": prevented - fpbill, "prevented": prevented, "bill": fpbill})

# --- 2. "you try it": the resolver's actual job, with its real candidates --
# Only events stage 1 could not settle -- the ones where a human has nothing
# but prose, an amount and a clock to go on.
puzzles = []
for e in ds.events:
    if len(puzzles) >= 4:
        break
    true_oid = T.event_to_order.get(e.event_id)
    if true_oid not in te_ids or e.channel is Channel.REFUND:
        continue
    cands = idx.candidates(e)
    if deterministic_link(e, cands) is not None or not (3 <= len(cands) <= 40):
        continue
    if not any(c.order_id == true_oid for c in cands):
        continue
    ranked = model.rank(e, cands)
    picked = {c.order_id: s for c, s in ranked}
    others = [c for c in cands if c.order_id != true_oid]
    others.sort(key=lambda c: -picked.get(c.order_id, 0))
    shown = [next(c for c in cands if c.order_id == true_oid)] + others[:3]
    if len(shown) < 4:
        continue
    shown.sort(key=lambda c: c.created_at)
    puzzles.append({
        "event": {"id": e.event_id, "channel": str(e.channel),
                  "at": e.occurred_at.isoformat(),
                  "value": e.value_paise, "note": e.free_text,
                  "descriptor": e.descriptor_text,
                  "email": e.email, "phone": e.phone,
                  "card": (e.card_bin or "") + "|" + (e.card_last4 or "")},
        "answer": true_oid,
        "model_pick": ranked[0][0].order_id,
        "model_conf": round(float(ranked[0][1]), 3),
        "n_real_candidates": len(cands),
        "options": [{"order_id": c.order_id, "at": c.created_at.isoformat(),
                     "value": c.amount_paise, "sku": c.sku,
                     "method": str(c.method), "email": c.email,
                     "descriptor": c.descriptor,
                     "card": (c.card_bin or "") + "|" + (c.card_last4 or ""),
                     "days": round((e.occurred_at - c.created_at).total_seconds() / 86400, 1),
                     "score": round(float(picked.get(c.order_id, 0)), 3)}
                    for c in shown],
    })

# --- 3. a playable slice of the stream ------------------------------------
# A realistic mix. Taking duplicates first made every bar in the stream go
# critical, which teaches the opposite of the truth: most multi-event orders
# are perfectly legitimate, and the alert only means something when it is
# rare among them.
multi = [o for o in te_ids if o in led.entries and len(led.entries[o].events) >= 2]
# `legit`, not `clean`: there is already a clean() helper for NaN scrubbing
# in this file, and shadowing it made the export die at the last line.
dups  = sorted((o for o in multi if o in pos),
               key=lambda o: led.entries[o].events[0].occurred_at)[:7]
legit = sorted((o for o in multi if o not in pos),
               key=lambda o: led.entries[o].events[0].occurred_at)[:19]
watch = dups + legit
t0 = min(led.entries[o].events[0].occurred_at for o in watch)
stream = []
for oid in watch:
    en = led.entries[oid]
    for ev in en.events:
        stream.append({
            "order_id": oid, "channel": str(ev.channel),
            "t": round((ev.occurred_at - t0).total_seconds() / 86400, 2),
            "value": ev.value_paise, "cost": ev.merchant_cost_paise,
            "order_value": en.order_value_paise,
            "note": ev.free_text, "key": bool(ev.payment_id or ev.order_id_hint or ev.awb),
            "dup": oid in T.duplicated_orders})
stream.sort(key=lambda r: r["t"])

# The orders nearest the duplicate line, for the scroll-scrubbed section on
# the landing page. Only these matter visually: everything far below the line
# never lights up at any threshold worth showing, so shipping all 7,935 would
# be weight without information.
near = sorted(((round(r, 3), 1 if o in pos else 0)
               for o, r in ratios.items() if o in te_ids and r >= 1.0),
              key=lambda x: -x[0])[:150]

extra = {"sweep": sweep, "puzzles": puzzles, "stream": stream, "near": near,
         "stream_orders": [{"order_id": o,
                            "value": led.entries[o].order_value_paise,
                            "dup": o in T.duplicated_orders,
                            "confuser": o in T.confuser_orders} for o in watch],
         "chosen_threshold": TH}
blob = json.loads(Path("reports/ui_data.json").read_text())
blob.update(clean(extra))
Path("reports/ui_data.json").write_text(json.dumps(blob, indent=1, default=str, allow_nan=False))
print("sweep:", len(sweep), "puzzles:", len(puzzles), "stream:", len(stream),
      "orders:", len(watch))
print("bytes:", Path("reports/ui_data.json").stat().st_size)
