## The pages

**Landing** https://claude.ai/code/artifact/85f17b51-e6d9-4441-bb98-7a551ad83500

A canvas score where each column is one real order, its marks the systems it
touched, and the tie between them the join no merchant can write. Red ties are
the orders that paid twice; hover isolates one.

Then the page hands over its controls. Scrolling the pinned section *is* the
duplicate threshold: 150 held-out orders light up as caught, held or missed,
and the net swings from a loss of Rs 53,285 at 1.02x to Rs 3.02 lakh at the
shipped line. Recall stays above 90% the whole way down, which is exactly why
a recall number on its own would call the losing configuration a success.

**Instrument** https://claude.ai/code/artifact/08c0ed9d-9d72-4c48-8924-79ddfd945aa3

Three things you operate rather than read:

- **Try it yourself.** One real support credit and four of the real candidate
  orders the resolver faced. Nothing in the ticket names an order. Most people
  guess, and on the first round the linkage model guesses too, at 0.377 against
  0.363. It was choosing from fifteen candidates, not four.
- **Run the book forward.** Play 52 real remediation events across 52 days over
  26 orders. Seven of the twenty-six cross the line. Scrub in either direction.
- **Move the duplicate line.** Every reading is a real measurement at that
  threshold.

Every figure, case, alert and curve is exported from an actual run.

## Deploying

The site is two self-contained files. No build step on the host, no runtime
dependencies, nothing fetched but the two webfonts.

```bash
make web          # export the run, then build and gate both pages
vercel deploy     # or point any static host at web/
make serve        # http://localhost:8000
```

`vercel.json` sets `web/` as the output directory with clean URLs, so the
instrument is served at `/instrument` as well as `/instrument.html`.

`scripts/build_ui.py` writes two forms of each page: the artifact copy, which
links to the instrument's published URL, and the deployable copy, wrapped in a
document shell with the reset the artifact runtime otherwise supplies. It
refuses to write either unless the JavaScript parses, every data region
populates, and the behaviour checks pass in a real browser: 13 interaction
checks against the instrument's controls, 12 scroll checks against the pinned
section.

# Double-Dip Sentinel

**A merchant pays the same customer twice for the same order, and nobody notices.**

Razorpay AI Buildathon. Track 02, AI Risk Manager.

---

## The loss class nobody names

A refund goes out. Sixty days later a chargeback lands for the same
transaction. Nobody connects them, nobody contests it, and the merchant pays
twice plus a dispute fee. Or a dispute is filed, a support agent "resolves"
it with a goodwill credit, and the dispute *also* resolves in the customer's
favour. Or a replacement ships, and then a refund follows it.

Every fraud system watches bad money coming **in**. Almost nobody watches
good money going **out** twice, because refunds are treated as customer
service rather than as a payout.

### Why merchants structurally cannot see it

There are five ways to be made whole, and each lives in a different system
with a different key and a different clock:

| Path | System | Key it carries |
|---|---|---|
| Refund | PSP | `payment_id`, strong |
| Chargeback | card network | ARN + masked PAN, **no order id** |
| Goodwill credit | CRM / support | ticket id + prose |
| Replacement | OMS | AWB |

There is no shared key and no single owner of the join. So nobody can answer
the one question that matters: *how much has this customer already been made
whole for this order, across all channels?*

In this dataset, **54% of the remediation events on duplicated orders carry
no explicit key at all.** They are invisible to a SQL join. That is the gap.

**Prior art, honestly:** Verifi RDR and Ethoca alerts already cover exactly
one pair of this: refund to chargeback, on card rails. The problem is real
enough that Visa and Mastercard monetise a slice of it. Nothing covers
goodwill credits, replacements, partial refunds, UPI or COD, and nothing
builds the unified ledger. That is the gap this fills.

---

## Results

Held-out orders only (temporal split on order creation date; the linkage
model never saw these orders). 26,450 orders, 9,470 remediation events,
**268 planted duplicates against 5,595 planted confusers**, a 21:1
adversarial imbalance.

| Detector | Precision | Recall | F1 | TP | FP |
|---|---|---|---|---|---|
| B0 · do nothing | n/a | 0.0% | 0.000 | 0 | 0 |
| B1 · explicit-key join, ≥2 events *(today's state of the art)* | 14.4% | 25.8% | 0.185 | 16 | 95 |
| B2 · same customer + similar amount ≤30d | 0.6% | 21.0% | 0.012 | 13 | 2120 |
| B3 · **oracle linkage**, ≥2 events | 8.1% | 100% | 0.149 | 62 | 706 |
| **Remediation ledger** | **96.6%** | **90.3%** | **0.933** | 56 | 2 |

Across three seeds, so the headline is not one lucky draw:

| Seed | Precision | Recall | F1 | B1 F1 | Net saved |
|---|---|---|---|---|---|
| 7 | 96.6% | 90.3% | 0.933 | 0.185 | Rs 3,34,415 |
| 11 | 93.5% | 97.7% | 0.956 | 0.267 | Rs 2,74,702 |
| 23 | 98.7% | 91.8% | 0.951 | 0.304 | Rs 5,10,252 |

Every run is byte-reproducible: three different `PYTHONHASHSEED` values
produce identical dataset, linkage, detection, ablation, censoring and gate
numbers. That was not true until bug 4 below was found.

**B3 is the baseline that matters.** Hand a naive detector *perfect* entity
resolution and it still gets 8.1% precision, because most orders with two
remediation events are legitimate. The problem is not only linkage. It is
knowing which multi-event orders are actually duplicates.

Recall by duplicate pattern:

| Pattern | Recall |
|---|---|
| `refund_then_chargeback` | 6/6 |
| `partial_refund_then_full_chargeback` | 4/4 |
| `chargeback_then_goodwill` | 2/2 |
| `replacement_then_refund` | 20/21 |
| `double_goodwill` | 24/29 ← the hard one |

### Money, net of the false-positive bill

| | |
|---|---|
| Duplicate payout prevented | **Rs 3,39,104** |
| False-positive bill (realised) | Rs 4,689 |
| Missed (allowed, was a duplicate) | Rs 3,561 |
| **Net** | **Rs 3,34,415** |

55 correct blocks, 2 wrong blocks, 1 routed to a human, across 634 decisions.

**The false-positive cost is unusually clean here, and it drives everything:**

> Wrongly holding a legitimate refund does not merely annoy someone. A
> meaningful share of those customers go on to file the very chargeback the
> system exists to prevent. **A false positive manufactures the loss.**

So `p_escalate_if_blocked = 0.35` sits in the middle of the cost model, not
in a footnote, and the operating threshold is solvable arithmetic rather
than taste.

### The chargeback rate you are looking at is wrong

Chargebacks land 8–150 days after the transaction, so recent orders are
**right-censored**, not clean. Treating them as negatives biases the
reported rate low, worst exactly where today's decisions are made.

| | |
|---|---|
| Naive rate (observed ÷ all card orders) | 2.25% |
| Kaplan-Meier corrected | 2.82% |
| **Understatement** | **1.25×** |
| Card orders still inside the dispute window | 62.6% |

Reported by cohort age, because a blended number hides the structure:
mature cohorts are fine, young ones are badly wrong. Cohorts that have seen
under 25% of their dispute window report *"insufficient exposure"* rather
than a ratio dominated by division noise.

---

## Architecture

```
four disconnected event streams, no shared key
refunds · disputes · support tickets · fulfilment
      │
      ▼
[1] DETERMINISTIC JOIN     exact keys. no model. 1,587 events, 100% correct
      │  residual
      ▼
[2] LEARNED PAIRWISE SCORER   blocked candidates, GBM, abstains when unsure
      │  residual                                    720 events, 98.2% correct
      ▼
[3] CLAUDE ON THE PROSE     support-ticket reading comprehension. closed set.
      │  residual
      ▼
    HUMAN QUEUE             581 events (6.1%). named, counted, not hidden.
      │
      ▼
[4] REMEDIATION LEDGER      append-only: value returned vs order value
      ▼
[5] HAZARD MODEL            P(chargeback that has not arrived yet)
      ▼
[6] GATE                    expected-value arithmetic → ALLOW / HOLD / BLOCK
```

Blocking recall is 97.3% at a mean block size of 14, reported separately,
because it is the ceiling on everything downstream and the matcher should
not be blamed for candidates it was never shown.

### Where AI is used, and where it deliberately is not

| Component | Model? | Why |
|---|---|---|
| Exact-key join | **No** | It is a join. A model here is slower, worse, and unauditable. |
| Pairwise linkage over candidates | **GBM** | Structured features, needs calibration, runs on every event. |
| Support-ticket disambiguation | **Claude** | Prose is the only remaining evidence. Genuine reading comprehension. |
| Duplicate decision | **No** | Arithmetic over a ledger. |
| Gate verdict | **No** | An expected-value comparison a merchant must be able to audit. A hallucinated authorization to withhold money owed is worse than no system. |

Stage 3 runs on ~6% of the stream and has three guardrails, because it is
the one soft component: it picks from a **closed candidate set** (an
out-of-set id is discarded, never trusted), **abstention is a first-class
answer** in the schema, and results are **disk-cached** so evaluation is
reproducible and does not re-bill. Without credentials it disables itself
and the residual goes to humans. That is a supported configuration, and the
ablation measures exactly what it is worth.

---

## What broke, and how I got out

The honest version. Every one of these inflated a headline number in my
favour before it was found.

**1. Two thresholds collided and the positive class was empty.**
The twin-confuser branch fired at `confuser_rate × 0.25 = 0.055` and the
loss branch at `loss_rate = 0.055`. Identical. The twin branch `continue`d,
so the duplicate path was *never reached*, and the 173 "duplicates" I was
scoring against were make-good confusers tripping a too-tight tolerance.
Fixed by replacing nested `roll <` comparisons with an explicit categorical
draw, which makes that class of bug impossible rather than just absent.

**2. The linkage model learned my generator instead of the problem.**
Goodwill events on duplicated orders linked at 100%; on single orders,
85.2%. Cause: duplicate patterns credited the *exact* order value while
confusers credited partials, so `weak_amount_exact` separated the classes
perfectly. Fixed by drawing every goodwill amount from one shared
distribution regardless of pattern. An event's amount must carry no
information about whether its order ends up duplicated. Detection fell from
94.7%/100% to 91.4%/91.4%. Those are the real numbers.

**3. I priced my own mistakes with my own confidence.**
The false-positive bill came to Rs 12.71 against Rs 3.4L prevented. I was
charging each wrong block the gate's *expected* cost, which is scaled by the
gate's belief that the payout was legitimate, so a confidently wrong gate
prices its errors at nearly zero. Circular. Once the label is known, the
full bill is due: Rs 4,689.

**4. Identical invocations disagreed, because a `set` decided a tiebreak.**
Two runs of the same command with the same seed reported 724 and 718 model
links. Not OpenMP, and `random_state` was already pinned. The cause: the
blocker collects candidates into a `set[str]`, Python salts string hashing
per process, and my proximity sort is *stable*, so ties silently inherited
an iteration order that changed every run. That shifted which candidates
survived truncation and which row order the model trained on. Fixed with a
total order (`(distance, order_id)`) rather than by exporting
`PYTHONHASHSEED`, because a reproducibility claim should hold without
requiring the caller to set an environment variable. For a project whose
entire claim is honest measurement, unreproducible numbers were the worst
bug in the list.

**5. Optimising the intermediate metric degraded the end-to-end one.**
`ctx_*` features lifted raw linkage accuracy ~0.5pp and cost 4–9pp of
detection *precision*, consistently across seeds 7/11/23. `ctx_amount_rank`
leaks the blocker's truncation order, so the model learned "rank 0 is
usually right", which is a fact about my blocker rather than the world, and
it produced confident wrong links on exactly the twin-order confusers. Now off by
default; the features stay so the ablation keeps demonstrating it.

**6. Card chargebacks on UPI and COD orders.** A domain error that capped
dispute blocking recall at 50.3%: those orders have no card tail to block
on. Disputes are now card-only, with an assertion. Recall → 100%.

**7. Recency truncation discarded exactly what disputes needed.** Sorting
candidates by "most recent" is intuitive and wrong. A chargeback lands
8–150 days after its order, so recency threw away the right answer. Ranking
by value proximity is channel-neutral.

**8. `hash()` broke reproducibility.** String hashing is salted per process,
so my "deterministic" descriptor differed on every run. `zlib.crc32`.

**9. Rs 5,000 amount blocks over a Rs 499–14,999 range** put most of the
book in one block and the 20k run never finished. Blocking is now tiered:
strong identifiers first, amount only as a last resort.

**10. The LLM stage lied about being available** and would have made 466
doomed API calls one network round-trip at a time. `Anthropic()` resolves
credentials lazily. Now checked up front, and latched off after 3 auth
failures.

---

## Limitations

- **Synthetic data.** No public Indian dataset carries linked refund /
  dispute / CRM / OMS records. The mitigation is that ground truth here is
  *lineage*. Whether two events refer to one order is a fact, not a guessed
  label, and and that the generator spends as much effort on confusers as on
  positives. It is not a substitute for a merchant's real book.
- **UPI disputes are out of scope.** NPCI's dispute path carries different
  identifiers. Modelling card rails only is honest; extending is real work.
- **62 positives in the held-out split.** Enough for the headline, thin for
  per-pattern recall. `chargeback_then_goodwill` at 2/2 means little.
- **The gate's cost model is assumptions**, stated in `CostModel` rather
  than buried, so a merchant can disagree with them numerically.
- **Stage 3 is unmeasured here**, with no credentials on the build machine. The
  581 residual events go to humans, and that is what the reported numbers
  reflect. `--llm` enables it.

---

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_eval.py --orders 25000 --seed 7   # ~90s
.venv/bin/python -m pytest tests/ -q                            # 28 tests, ~5s
```

`--llm` enables stage 3 (billed, disk-cached). Everything else is offline
and deterministic given `--seed`.

### Layout

| Path | |
|---|---|
| [generate.py](src/sentinel/generate.py) | the world, the confusers, the answer key |
| [resolve/blocking.py](src/sentinel/resolve/blocking.py) | tiered candidate generation |
| [resolve/features.py](src/sentinel/resolve/features.py) | pairwise features |
| [resolve/matcher.py](src/sentinel/resolve/matcher.py) | deterministic + learned cascade |
| [resolve/llm.py](src/sentinel/resolve/llm.py) | Claude on the residual |
| [ledger.py](src/sentinel/ledger.py) | the artefact merchants don't have |
| [hazard.py](src/sentinel/hazard.py) | survival, censoring, cohort bias |
| [gate.py](src/sentinel/gate.py) | expected-value decisions |
| [metrics.py](src/sentinel/metrics.py) | baselines, money, sweeps |

---

## Why Razorpay specifically

A merchant structurally cannot build this: the join spans four systems they
own separately. **Razorpay can.** The refund, the dispute and the settlement
are on one side of the wire. The join that is impossible inside a merchant
is a table join inside a PSP.

That is the whole argument. It is not a project. It is a product only a
payments company is positioned to ship, and nobody currently owns it.

## The pages

**Landing:** https://claude.ai/code/artifact/85f17b51-e6d9-4441-bb98-7a551ad83500

A canvas score where each column is one real order, its marks the systems it
touched, and the tie between them the join no merchant can write. Red ties are
the orders that paid twice. Hover isolates one.

## The instrument

A published forensic view of the same pipeline output:
**https://claude.ai/code/artifact/08c0ed9d-9d72-4c48-8924-79ddfd945aa3**

Three things you operate rather than read:

- **Try it yourself.** One real support credit and four of the real candidate
  orders the resolver faced. Nothing in the ticket names an order. Most people
  guess, and on the first round the linkage model guesses too, at a margin of
  0.377 against 0.363. It was choosing from fifteen candidates, not four.
- **Run the book forward.** Play 52 real remediation events across 52 days over
  26 orders. Tracks fill as value flows back, the notch is the order value, and
  seven of the twenty-six cross the line. Scrub in either direction.
- **Move the duplicate line.** Every reading is a real measurement at that
  threshold. Drag it below about 1.1x and the net goes *negative*: the
  false-positive bill overtakes everything recovered, which is the argument
  against recall-only metrics made draggable.

Every figure, case, alert and curve is exported from an actual run. Nothing is
mocked. `scripts/build_ui.py` refuses to write the page unless it parses, every
data region populates, and thirteen interaction checks pass in a real browser.

```bash
.venv/bin/python scripts/export_ui.py   # run pipeline -> reports/ui_data.json
.venv/bin/python scripts/build_ui.py    # inline it -> ui/sentinel.html
```
