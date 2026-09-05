# Double-Dip Sentinel

**Razorpay AI Buildathon, Track 02: AI Risk Manager**

A detector, verifier and gate for one specific class of merchant loss: paying
the same customer back more than once for the same order.

---

## 1. The problem

### 1.1 The loss nobody has a word for

A refund goes out. Sixty days later a chargeback lands for the same
transaction. Nobody connects them, nobody contests it, and the merchant pays
twice plus a dispute fee.

Or a dispute is filed, a support agent "resolves" it with a goodwill credit,
and the dispute *also* resolves in the customer's favour.

Or a replacement ships, and a refund follows it anyway.

Every fraud system in production watches bad money coming **in**. Almost
nobody watches good money going **out** twice, because refunds are treated as
customer service rather than as a payout.

### 1.2 Why merchants structurally cannot see it

There are four ways a customer gets made whole, and each lives in a different
system, with a different key, on a different clock:

| Path | System | Key it carries |
|---|---|---|
| Refund | PSP | `payment_id`, strong |
| Chargeback | card network | ARN + masked PAN, **no order id** |
| Goodwill credit | CRM / support desk | ticket id, and whatever the agent typed |
| Replacement | OMS / fulfilment | AWB |

There is no shared key and no single owner of the join. So nobody can answer
the one question that decides everything: *how much has this order already
paid back?*

In our evaluation book, **272 of 272 chargebacks (100%) carry no order id**,
and **2,778 of 3,255 support credits (85%) carry no joinable key either**.
Across all remediation events on orders that were genuinely paid twice,
**54% cannot be joined by any query a merchant can write.**

### 1.3 Prior art, stated honestly

Visa's RDR and Mastercard's Ethoca alerts already cover exactly one pair of
this: refund against chargeback, on card rails. The problem is real enough
that both networks monetise a slice of it.

Nothing covers goodwill credits, replacements, partial refunds, UPI or COD,
and nothing builds the unified ledger. That gap is the project.

---

## 2. Approach

### 2.1 The core idea

Build the artefact that does not exist: **a remediation ledger**. One
append-only view, per order, of how much value has flowed back to that
customer across every channel, regardless of which system recorded it.

Everything else follows from having that number:

- **Detection** is then arithmetic. Total returned against order value.
- **Prevention** is a gate in front of the next payout, comparing the cost of
  paying against the cost of holding, in rupees.
- **Forecasting** needs a hazard model, because the most expensive channel
  (chargebacks) arrives 8 to 150 days late and has not landed yet.

### 2.2 Why this is a Razorpay product and not a merchant one

A merchant structurally cannot build this. The join spans four systems they
own separately, and the hardest side of it (the dispute) arrives from a
network that never had their order id.

**Razorpay can.** The refund, the dispute and the settlement are on one side
of the wire. The join that is impossible inside a merchant is a table join
inside a PSP. That asymmetry is the whole commercial argument.

---

## 3. Implementation

### 3.1 Pipeline

```
four disconnected event streams, no shared key
refunds  .  disputes  .  support tickets  .  fulfilment
      |
      v
[1] TIERED BLOCKING            strong identifiers first, amount as last resort
      |                        97.3% recall ceiling, mean block 14
      v
[2] DETERMINISTIC JOIN         exact keys. no model. 1,587 events, 100% correct
      |  residual
      v
[3] LEARNED PAIRWISE SCORER    gradient boosting over blocked candidates,
      |  residual              abstains when top-1 is weak or too close to top-2
      v
[4] CLAUDE ON THE PROSE        support-ticket reading comprehension, closed set
      |  residual
      v
    HUMAN QUEUE                568 events. named, counted, never hidden.
      |
      v
[5] REMEDIATION LEDGER         append-only. value returned vs order value.
      v
[6] HAZARD MODEL               discrete-time survival. P(chargeback not yet here)
      v
[7] GATE                       expected value in rupees -> allow / hold / block
```

### 3.2 Where AI is used, and where it deliberately is not

| Component | Model? | Reasoning |
|---|---|---|
| Exact-key join | **No** | It is a join. A model here is slower, less accurate and unauditable. |
| Pairwise linkage | **Gradient boosting** | Structured features, needs calibration, runs on every event. |
| Support-ticket disambiguation | **Claude** | Prose is the only remaining evidence. Genuine reading comprehension. |
| Duplicate decision | **No** | Arithmetic over a ledger. |
| Gate verdict | **No** | An expected-value comparison a merchant must be able to audit. A hallucinated authorisation to withhold money owed is worse than having no system. |

The language model runs on roughly 6% of the stream and carries three
guardrails, because it is the one soft component:

1. **Closed set.** It picks from candidates the blocker produced. An
   out-of-set id is discarded, never trusted, so a hallucinated order can
   never enter the ledger.
2. **Abstention is a first-class answer.** The schema permits "not enough
   evidence" and the prompt says so. Forcing a choice on a genuinely
   ambiguous ticket is strictly worse than the human queue.
3. **Disk cache**, so evaluation is reproducible and does not re-bill.

Without credentials the stage disables itself and the residual goes to
humans. That is a supported configuration, and the ablation measures exactly
what it is worth.

### 3.3 The evaluation, which is the actual contribution

Ground truth here is **lineage**, not a guessed label. Whether two events
refer to the same order is a fact, which is why synthetic data works honestly
in this problem where it would not in most.

The generator spends as much effort on **confusers** as on positives:

| Confuser | What it defeats |
|---|---|
| Twin orders | same customer, same card, same day, near-equal amount |
| Split remediation | one loss legitimately settled in two partial events |
| Serial complainer | many orders, each with one legitimate remediation |
| Make-good combo | replacement plus a token apology credit |
| Shared card | two customers behind one PAN |

The book carries **268 real duplicates against 5,595 planted confusers**, a
21:1 adversarial imbalance. Precision had to be earned against cases built to
look exactly like the thing being hunted.

---

## 4. Results

Held-out orders only. The split is temporal on order creation date, so the
linkage model never saw these orders.

| Detector | Precision | Recall | F1 | Caught | False alarms |
|---|---|---|---|---|---|
| Do nothing | n/a | 0.0% | 0.000 | 0 | 0 |
| Explicit-key join, what merchants run today | 14.4% | 25.8% | 0.185 | 16 | 95 |
| Same customer and amount inside 30 days | 0.6% | 21.0% | 0.012 | 13 | 2,120 |
| **Oracle linkage** then count events | 8.1% | 100% | 0.149 | 62 | 706 |
| **Remediation ledger** | **96.6%** | **90.3%** | **0.933** | 56 | 2 |

**The oracle row is the one that matters.** Hand a naive detector *perfect*
entity resolution and it still reaches 8.1% precision, because most orders
touching two channels are entirely legitimate. Linking is only half the
problem.

Stable across seeds: F1 of 0.933, 0.956 and 0.951 on seeds 7, 11 and 23.
Byte-reproducible across hash seeds.

### 4.1 Money, net of the false-positive bill

| | |
|---|---|
| Duplicate payout prevented | Rs 3,39,104 |
| False-positive bill, realised | Rs 4,689 |
| Missed, allowed and was a duplicate | Rs 3,561 |
| **Net** | **Rs 3,34,415** |

55 correct blocks, 2 wrong blocks, 1 routed to a human, across 634 decisions
on 25,000 orders.

### 4.2 The false-positive cost is structural here

> Wrongly holding a legitimate refund does not merely annoy someone. A share
> of those customers go on to file the very chargeback the system exists to
> prevent. **A false positive manufactures the loss.**

So `p_escalate_if_blocked = 0.35` sits inside the arithmetic rather than in a
footnote, and the operating threshold is solvable rather than a matter of
taste. Below roughly 1.10x the net goes **negative**: the bill overtakes
everything recovered and the merchant is worse off running the detector than
not. Recall stays above 90% the whole way down into that region, which is
precisely why a recall number on its own would call the losing configuration
a success.

### 4.3 The reported chargeback rate is wrong

Disputes land 8 to 150 days after the sale, so recent orders are not clean.
They are **unfinished**. Counting them as negatives biases the rate downward,
worst exactly where today's decisions are made.

| | |
|---|---|
| Naive rate, observed over all card orders | 2.25% |
| Kaplan-Meier corrected | 2.82% |
| **Understatement** | **1.25x** |
| Card orders still inside the dispute window | 62.6% |

Pull the observation date back to day 60 and the naive rate reads 0.75%
against a corrected 1.47%: **half the truth**, with nothing about the fraud
having changed.

Cohorts that have seen under 25% of their dispute window report
*insufficient exposure* rather than a ratio dominated by division noise.

---

## 5. What broke, and how it got fixed

Every one of these inflated a headline number in our favour before it was
found.

1. **Two thresholds collided and the positive class was empty.** The twin
   confuser branch and the loss branch both fired at 0.055, so the duplicate
   path was never reached and the "duplicates" being scored were confusers.
   Fixed by replacing nested threshold comparisons with an explicit
   categorical draw, which makes that class of bug impossible rather than
   merely absent.

2. **The model learned the generator instead of the problem.** Duplicate
   patterns credited the exact order value while confusers credited partials,
   so one feature separated the classes perfectly and goodwill linked at 100%
   on duplicated orders against 85% elsewhere. Fixed by drawing every credit
   amount from one shared distribution. Detection fell from 94.7%/100% to
   honest numbers.

3. **We priced our own mistakes with our own confidence.** The
   false-positive bill came to Rs 12.71, because each wrong block was charged
   the gate's *expected* cost, which is scaled by the gate's belief that the
   payout was legitimate. A confidently wrong gate prices its errors at
   nearly zero. Circular. Realised cost: Rs 4,689.

4. **Identical invocations disagreed.** A `set[str]` decided sort tiebreaks,
   Python salts string hashing per process, and the sort was stable. Fixed
   with a total order rather than by exporting `PYTHONHASHSEED`, because a
   reproducibility claim should not depend on the caller setting an
   environment variable.

5. **Optimising the intermediate metric degraded the end-to-end one.**
   Context features lifted raw linkage accuracy by 0.5pp and cost 4 to 9pp of
   detection *precision* across three seeds, because one of them leaked the
   blocker's truncation order. Off by default; kept so the ablation keeps
   demonstrating it.

6. **Card chargebacks on UPI and COD orders.** A domain error that capped
   dispute blocking recall at 50.3%, since those orders have no card tail to
   block on. Disputes are card-only now, with an assertion.

7. **A page shipped whose JavaScript did not parse.** It rendered perfectly
   and every data-driven region was silently empty. The build now runs
   `node --check`, loads the page in a real browser, and asserts every region
   populated.

8. **A timing constant shadowed by the chart's own variable** turned every
   animation delay into `NaN`, so `render` died partway and left a third of a
   panel blank with no error raised.

9. **Scroll work hidden inside `requestAnimationFrame`.** A headless check
   proved rAF never fired in that configuration, so the entire scroll UI sat
   inert. Anywhere rAF is throttled a real visitor would have seen the same.
   Now synchronous.

---

## 6. Limitations

- **Synthetic data.** No public Indian dataset carries linked refund,
  dispute, CRM and OMS records. The mitigation is that ground truth is
  lineage rather than a guessed label, and that the generator spends as much
  effort on confusers as on positives. It is not a substitute for a
  merchant's real book.
- **UPI disputes are out of scope.** NPCI's dispute path carries different
  identifiers. Card rails only is honest; extending is real work.
- **62 positives in the held-out split.** Enough for the headline, thin for
  per-pattern recall.
- **The gate's cost model is assumptions**, stated in `CostModel` rather than
  buried, so a reviewer can disagree with them numerically.
- **The Claude stage is unmeasured in the shipped numbers**, because the
  build machine had no credentials. The 568 residual events went to the human
  queue, and the reported figures reflect that.

---

## 7. Running it

```bash
make setup      # venv and dependencies
make test       # 28 tests, about 7 seconds
make eval       # full pipeline, every number in this document
make web        # build and gate both pages
make serve      # http://localhost:8000
```

Everything is deterministic given `--seed`. `--llm` enables the Claude stage;
it is billed and disk-cached.

### Repository layout

| Path | |
|---|---|
| `src/sentinel/generate.py` | the world, the confusers, the answer key |
| `src/sentinel/resolve/` | blocking, features, deterministic and learned matching, the Claude stage |
| `src/sentinel/ledger.py` | the artefact merchants do not have |
| `src/sentinel/hazard.py` | survival, censoring, cohort bias |
| `src/sentinel/gate.py` | expected-value decisions |
| `src/sentinel/metrics.py` | baselines, money, threshold sweeps |
| `scripts/run_eval.py` | one command, every number |
| `scripts/export_ui.py` | pipeline output to page payload |
| `scripts/build_ui.py` | builds and gates both pages |
| `web/` | the deployable site |

---

## 8. What to submit

The buildathon form asks for exactly twelve things. Here is what goes in each
field that concerns the build.

**Track.** 02, AI Risk Manager.

**Project name.** Double-Dip Sentinel.

**What it solves.** Suggested wording:

> Merchants pay the same customer back more than once for the same order: a
> refund followed by a chargeback, a dispute quietly settled with a goodwill
> credit, a replacement plus a refund. It is invisible because the four
> systems that pay money out share no key, and 54% of the events on orders
> that were paid twice carry no identifier any query can join on. Double-Dip
> Sentinel builds the remediation ledger that does not exist today, then puts
> a gate in front of the second payout. On held-out orders it reaches 96.6%
> precision at 90.3% recall against a 21:1 confuser imbalance, where the join
> a merchant can actually run today reaches 14.4% at 25.8%. Net Rs 3.34 lakh
> saved per 25,000 orders, after the false-positive bill.

**GitHub repo.** The public repository URL.

**Pitch video.** Five minutes, unlisted is fine. Script in
`docs/DEMO_SCRIPT.md`.

**What broke, and how you got out.** *They read this one first.* Lead with
number 3 from section 5 above: pricing your own mistakes with your own
confidence. Suggested wording:

> The false-positive bill came out at Rs 12.71 against Rs 3.4 lakh recovered,
> which I believed for about a minute. I was charging every wrong block the
> gate's own *expected* cost, and that estimate is scaled by the gate's
> belief that the payout was legitimate. So a gate that was confidently wrong
> priced its own mistakes at nearly zero. It was grading itself with its own
> confidence, which is circular, and it made the headline number look far
> better than it was. Once the label is known the payout was legitimate, so
> the full bill is due: Rs 4,689. The same class of error showed up twice
> more, once where the linkage model had learned my data generator rather
> than the problem, and once where two identical runs disagreed because a
> Python set decided a sort tiebreak. All three were caught by making the
> evaluation adversarial to itself rather than by testing that the code ran.

**Anything else you want us to see.** The two live pages, and section 5 of
this document.
