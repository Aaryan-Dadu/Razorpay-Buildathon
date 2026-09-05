#!/usr/bin/env python3
"""Full evaluation. One command, every number in the README.

    python scripts/run_eval.py --orders 25000 --seed 7

Writes reports/results.json and prints the tables. Nothing here is
cherry-picked: the seed, the split and the universe are fixed before any
metric is computed, and every detector is scored on the same held-out
orders.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.gate import CostModel, Gate, replay
from sentinel.generate import World, WorldConfig
from sentinel.hazard import (HazardModel, build_histories, censoring_bias,
                             cohort_bias, kaplan_meier)
from sentinel.ledger import Ledger
from sentinel.metrics import (baseline_customer_amount, baseline_explicit_key,
                              baseline_oracle_count, money, score,
                              threshold_sweep)
from sentinel.resolve.blocking import BlockingIndex, blocking_recall
from sentinel.resolve.matcher import (LinkageModel, Method, order_temporal_split,
                                      resolve_all)
from sentinel.schema import rupees

THRESHOLD = 1.25


def hr(t): print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default="reports/results.json")
    ap.add_argument("--llm", action="store_true",
                    help="enable stage 3 (Claude on the residual). Makes "
                         "billed API calls; results are disk-cached.")
    ap.add_argument("--llm-limit", type=int, default=None,
                    help="cap stage-3 calls, for a cheap smoke run")
    a = ap.parse_args()

    t0 = time.time()
    R: dict = {"config": vars(a)}

    # ---- world -------------------------------------------------------
    ds = World(WorldConfig(n_orders=a.orders, seed=a.seed)).build()
    R["dataset"] = ds.summary()
    hr("DATASET")
    for k, v in ds.summary().items():
        print(f"  {k:26s} {v}")

    T = ds.truth
    tr_ids, te_ids, tr_ev, cutoff = order_temporal_split(
        ds.orders, ds.events, T.event_to_order, a.train_frac)
    print(f"  {'train/test orders':26s} {len(tr_ids)} / {len(te_ids)}"
          f"   cutoff {cutoff.date()}")

    # ---- linkage -----------------------------------------------------
    idx = BlockingIndex(ds.orders)
    br = blocking_recall(idx, ds.events, T.event_to_order)
    R["blocking"] = br
    hr("STAGE 1-2  LINKAGE")
    print(f"  blocking recall (ceiling)  {br['blocking_recall']:.4f}"
          f"   mean block {br['mean_block_size']:.1f}")

    model = LinkageModel(a.seed).fit(idx, tr_ev, T.event_to_order)
    res = resolve_all(idx, ds.events, model)
    E = {e.event_id: e for e in ds.events}

    # ---- stage 3: Claude on the residual -----------------------------
    from sentinel.resolve.llm import LLMResolver, apply_llm_stage
    n_abstained_before = sum(1 for r in res if not r.linked)
    llm = LLMResolver(enabled=a.llm or None) if a.llm else LLMResolver(enabled=False)
    if a.llm and llm.available:
        res = apply_llm_stage(res, E, lambda e: idx.candidates(e), llm,
                              limit=a.llm_limit)
        R["llm"] = {"enabled": True, **vars(llm.stats)}
        print(f"  stage 3 (Claude): {llm.stats.linked} linked, "
              f"{llm.stats.abstained} abstained, "
              f"{llm.stats.rejected_not_in_candidates} rejected out-of-set, "
              f"{llm.stats.errors} errors")
    else:
        reason = ("not requested (pass --llm to enable)" if not a.llm
                  else llm.disabled_reason or "unavailable")
        R["llm"] = {"enabled": False, "reason": reason,
                    "residual_left_to_humans": n_abstained_before}
        print(f"  stage 3 (Claude): DISABLED -- {reason}")
        print(f"    {n_abstained_before} residual events go to the human "
              f"queue instead")
    rmap = {r.event_id: r for r in res}

    per_ch: dict[str, list[int]] = {}
    per_me: dict[str, list[int]] = {}
    for eid, true_oid in T.event_to_order.items():
        if true_oid not in te_ids:
            continue
        r = rmap[eid]
        ok = r.linked and r.order_id == true_oid
        per_ch.setdefault(str(E[eid].channel), [0, 0])[0 if ok else 1] += 1
        per_me.setdefault(str(r.method), [0, 0])[0 if ok else 1] += 1
    R["linkage_by_channel"] = {k: {"n": sum(v), "acc": v[0] / sum(v)}
                               for k, v in per_ch.items()}
    R["linkage_by_method"] = {k: {"n": sum(v), "acc": v[0] / sum(v)}
                              for k, v in per_me.items()}
    print(f"  {'channel':14s} {'n':>6s} {'accuracy':>9s}")
    for k, v in sorted(per_ch.items()):
        print(f"  {k:14s} {sum(v):6d} {v[0] / sum(v):9.1%}")
    print(f"  {'-- by stage --':14s}")
    for k, v in sorted(per_me.items()):
        print(f"  {k:14s} {sum(v):6d} {v[0] / sum(v):9.1%}")

    # ---- ledger + detection -----------------------------------------
    led = Ledger.build(ds.orders, ds.events, res)
    R["ledger"] = led.summary()
    ratios = {oid: en.exposure_ratio for oid, en in led.entries.items()}
    pos = T.duplicated_orders & te_ids
    flagged = {oid for oid, r in ratios.items() if r >= THRESHOLD} & te_ids

    rows = []
    rows.append(("B0  do nothing", score(set(), pos, te_ids)))
    rows.append(("B1  explicit-key join, >=2",
                 score(baseline_explicit_key(ds.orders, ds.events, te_ids),
                       pos, te_ids)))
    rows.append(("B2  same customer + amount",
                 score(baseline_customer_amount(ds.orders, ds.events, te_ids),
                       pos, te_ids)))
    rows.append(("B3  ORACLE linkage, >=2",
                 score(baseline_oracle_count(T, ds.events, te_ids), pos, te_ids)))
    rows.append(("**  remediation ledger", score(flagged, pos, te_ids)))

    hr(f"DETECTION  (held-out orders only, n={len(te_ids)}, positives={len(pos)})")
    print(f"  {'method':30s} {'prec':>7s} {'recall':>7s} {'F1':>7s} "
          f"{'TP':>5s} {'FP':>5s} {'FN':>5s}")
    R["detection"] = {}
    for name, s in rows:
        print(f"  {name:30s} {s.precision:7.1%} {s.recall:7.1%} {s.f1:7.3f} "
              f"{s.tp:5d} {s.fp:5d} {s.fn:5d}")
        R["detection"][name.strip()] = s.as_dict()

    pt, ph = Counter(), Counter()
    for oid in pos:
        p_ = T.duplicate_patterns.get(oid, "?")
        pt[p_] += 1
        if oid in flagged:
            ph[p_] += 1
    print(f"\n  recall by duplicate pattern:")
    R["recall_by_pattern"] = {}
    for p_, n in pt.most_common():
        print(f"    {p_:38s} {ph[p_]:3d}/{n:3d} = {ph[p_] / n:6.1%}")
        R["recall_by_pattern"][p_] = {"hit": ph[p_], "n": n, "recall": ph[p_] / n}

    fp_ids = flagged - pos
    print(f"\n  false positives: {len(fp_ids)}  "
          f"({len(fp_ids & T.confuser_orders)} are planted confusers)")
    R["fp_confuser_share"] = (len(fp_ids & T.confuser_orders) / len(fp_ids)
                              if fp_ids else 0.0)

    # ---- threshold sensitivity --------------------------------------
    sweep = threshold_sweep(ratios, pos, te_ids)
    R["threshold_sweep"] = sweep
    best = max(sweep, key=lambda r: r["f1"])
    flat = [r for r in sweep if 1.10 <= r["threshold"] <= 1.40]
    hr("THRESHOLD SENSITIVITY")
    print(f"  best F1 {best['f1']:.3f} at {best['threshold']:.2f}; "
          f"chosen {THRESHOLD:.2f}")
    print(f"  F1 range across 1.10..1.40: "
          f"{min(r['f1'] for r in flat):.3f}..{max(r['f1'] for r in flat):.3f}"
          f"  -> the line is not load-bearing")

    # ---- ablation ----------------------------------------------------
    hr("ABLATION  (each row drops one feature family from linkage)")
    print(f"  {'features':28s} {'link acc':>9s} {'prec':>7s} {'recall':>7s} {'F1':>7s}")
    R["ablation"] = {}
    for label, drop in [("default (no ctx_*)", ("ctx",)),
                        ("+ ctx_*  [rejected]", ()),
                        ("no text_*", ("ctx", "text")),
                        ("no weak_*", ("ctx", "weak")),
                        ("no strong_*", ("ctx", "strong"))]:
        try:
            m2 = LinkageModel(a.seed, drop_families=drop).fit(
                idx, tr_ev, T.event_to_order)
        except ValueError:
            continue
        r2 = resolve_all(idx, ds.events, m2)
        ok = tot = 0
        for eid, true_oid in T.event_to_order.items():
            if true_oid not in te_ids:
                continue
            rr = {x.event_id: x for x in r2}[eid]
            tot += 1
            ok += 1 if (rr.linked and rr.order_id == true_oid) else 0
        l2 = Ledger.build(ds.orders, ds.events, r2)
        f2 = {o for o, en in l2.entries.items()
              if en.exposure_ratio >= THRESHOLD} & te_ids
        s2 = score(f2, pos, te_ids)
        print(f"  {label:28s} {ok / tot:9.1%} {s2.precision:7.1%} "
              f"{s2.recall:7.1%} {s2.f1:7.3f}")
        R["ablation"][label] = {"link_acc": ok / tot, **s2.as_dict()}
    print("\n  ctx_* is off by default: it buys ~0.5pp of linkage accuracy and\n"
          "  costs several points of detection precision, because\n"
          "  ctx_amount_rank leaks the blocker's truncation order. Optimising\n"
          "  the intermediate metric degraded the end-to-end one.\n")
    print("  note: dropping strong_* changes nothing, and should not. The\n"
          "  deterministic stage has already consumed every event carrying a\n"
          "  strong key, so the model only ever trains on rows where those\n"
          "  features are zero. That is the intended division of labour, and\n"
          "  the ablation is how you check it actually holds.")

    # ---- censoring ---------------------------------------------------
    H = build_histories(ds, {r.event_id: r.order_id for r in res if r.linked})
    hs = list(H.values())
    cb = censoring_bias(hs)
    R["censoring"] = cb
    hr("CENSORING  (why the merchant's own chargeback rate is wrong)")
    print(f"  naive rate            {cb['naive_rate']:.4f}")
    print(f"  KM-corrected rate     {cb['km_corrected_rate']:.4f}")
    print(f"  understatement        {cb['understatement_x']:.2f}x")
    print(f"  still inside window   {cb['pct_still_at_risk']:.1%} of card orders")
    print(f"\n  {'observed days':>16s} {'n':>6s} {'naive':>8s} {'exposure':>9s} "
          f"{'projected':>10s}")
    R["cohort_bias"] = cohort_bias(hs)
    for r in R["cohort_bias"]:
        hi = "+" if r["observed_days_hi"] > 1e5 else f"{r['observed_days_hi']:.0f}"
        proj = ("insufficient" if not r["projectable"]
                else f"{r['projected_rate']:.4f}")
        print(f"  {r['observed_days_lo']:.0f}-{hi:<11s} {r['n_orders']:6.0f} "
              f"{r['naive_rate']:8.4f} {r['exposure_seen']:9.3f} {proj:>10s}")

    haz = HazardModel(a.seed).fit([h for h in hs if h.order.order_id in tr_ids])

    # ---- gate --------------------------------------------------------
    cost = CostModel()
    gate = Gate(cost, THRESHOLD)
    test_led = Ledger(entries={o: e for o, e in led.entries.items()
                               if o in te_ids}, exceptions=led.exceptions)
    decisions = replay(test_led, gate, H, haz)
    pending, pending_val = {}, {}
    for oid, en in test_led.entries.items():
        if len(en.events) > 1:
            pending[oid] = sum(e.merchant_cost_paise for e in en.events[1:])
            pending_val[oid] = sum(e.value_paise for e in en.events[1:])
    mr = money(decisions, T.duplicated_orders, pending, cost, pending_val)
    R["gate"] = mr.as_dict()
    hr("GATE  (prospective: decided against the ledger as it stood)")
    print(f"  decisions            {len(decisions)}  "
          f"(block {mr.n_block} / hold {mr.n_hold} / allow {mr.n_allow})")
    print(f"  correct blocks       {mr.n_correct_block}")
    print(f"  wrong blocks         {mr.n_wrong_block}")
    print(f"  duplicate payout prevented   {rupees(mr.prevented_paise):>18s}")
    print(f"  false-positive bill          {rupees(mr.false_positive_cost_paise):>18s}")
    print(f"  missed (allowed, was dup)    {rupees(mr.missed_paise):>18s}")
    print(f"  {'NET':28s} {rupees(mr.net_paise):>18s}")

    # ---- exceptions --------------------------------------------------
    hr("EXCEPTION QUEUE  (what it could not resolve, and admits to)")
    exc = Counter(str(E[e.event_id].channel) for e, _ in led.exceptions
                  if e.event_id in E)
    print(f"  {len(led.exceptions)} events routed to a human "
          f"({len(led.exceptions) / len(ds.events):.1%} of stream)")
    for k, v in exc.most_common():
        print(f"    {k:14s} {v}")
    R["exceptions"] = {"total": len(led.exceptions), "by_channel": dict(exc)}

    R["runtime_sec"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {a.out}  ({R['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
