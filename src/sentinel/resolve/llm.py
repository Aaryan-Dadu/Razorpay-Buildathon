"""Stage 3: Claude on the residual only.

Where the LLM earns its place
-----------------------------
Stage 1 joins on exact keys. Stage 2 scores structured features. Both
abstain on roughly 5% of the stream, and that residual is almost entirely
support credits whose only distinguishing evidence is prose a human typed
into a ticket: "cust says the blue kurta never arrived, courier shows
delivered, issued credit". Matching that to one of eleven candidate orders
is a reading-comprehension problem. It is the one part of this pipeline a
language model is genuinely better at than a feature vector.

Where it does not go
--------------------
Nowhere else. Not in the join (that is exact-key equality). Not in the
duplicate decision (that is arithmetic over a ledger). Not in the gate
(that is an expected-value comparison whose inputs must be auditable). A
model in any of those places would be slower, less accurate, and impossible
to explain to a merchant asking why their refund was held.

Three guardrails, because the component is soft
-----------------------------------------------
1. **Closed set.** The model picks from candidates the blocker produced. A
   returned id that is not in that set is discarded, not trusted -- so a
   hallucinated order can never enter the ledger.
2. **Abstention is a first-class answer.** The schema permits "not enough
   evidence" and the prompt says so. Forcing a choice on a genuinely
   ambiguous ticket would be strictly worse than leaving it in the human
   queue, because a wrong link silently corrupts the ledger downstream.
3. **Disk cache.** Keyed by the hash of the exact prompt payload. Re-running
   the evaluation does not re-bill, and results stay reproducible.

Absent credentials the resolver disables itself and the pipeline runs
without stage 3. That is a supported configuration, not a failure -- the
ablation in the report measures exactly what it is worth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ..schema import Order, RemediationEvent
from .matcher import Method, Resolution

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"
CACHE_DIR = Path(".cache/llm")

SYSTEM = """You reconcile customer-service records for an Indian e-commerce \
merchant.

You are given ONE remediation event (a refund, a support credit, a \
replacement shipment, or a card chargeback) and a SHORT LIST of candidate \
orders it might belong to. Decide which candidate order the event refers \
to, or say you cannot tell.

How to weigh evidence:
- An order id or AWB quoted in the note is decisive.
- Product descriptions in the note ("blue kurta", "steel watch") should \
match the candidate's item.
- Amounts often match the order value exactly, but partial credits are \
common and normal. A partial amount is weak evidence, not disqualifying.
- The event must occur AFTER the order was placed. Candidates are already \
filtered for this, but check the gap is plausible: support credits usually \
land within days, card chargebacks 8-150 days later.
- Two candidates from the same customer, placed the same day for similar \
amounts, are genuinely ambiguous. That is what "cannot tell" is for.

Answer "cannot tell" whenever the evidence does not clearly favour one \
candidate. An incorrect link corrupts a financial ledger; leaving it for a \
human costs a few minutes. Prefer the human."""


class LinkVerdict(BaseModel):
    """Structured answer. ``order_id`` is None when the model abstains."""

    order_id: str | None = Field(
        default=None,
        description="The candidate order_id this event belongs to, copied "
                    "exactly. Null if the evidence does not clearly favour "
                    "one candidate.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0-1. Use below 0.6 when genuinely unsure.")
    reason: str = Field(
        description="One sentence citing the specific evidence used.")


@dataclass
class LLMStats:
    called: int = 0
    cached: int = 0
    linked: int = 0
    abstained: int = 0
    rejected_not_in_candidates: int = 0
    errors: int = 0


class LLMResolver:
    """Stage 3. Disabled and harmless when no credentials are present."""

    def __init__(self, model: str = MODEL, min_confidence: float = 0.6,
                 cache_dir: Path = CACHE_DIR, enabled: bool | None = None):
        self.model = model
        self.min_confidence = min_confidence
        self.cache_dir = cache_dir
        self.stats = LLMStats()
        self.client = None
        self.disabled_reason: str | None = None
        #: Consecutive auth failures before the stage latches itself off.
        #: Without this a credential-less run makes one doomed API call per
        #: abstained event -- hundreds of them -- and each one waits on a
        #: network round trip before failing. Fail fast, once.
        self._auth_failures = 0
        self._latched_off = False

        if enabled is False:
            self.disabled_reason = "explicitly disabled"
            log.info("LLM stage explicitly disabled")
            return
        try:
            import anthropic
        except ImportError as exc:
            self.disabled_reason = f"anthropic SDK not installed ({exc})"
            return

        client = anthropic.Anthropic()
        # `Anthropic()` resolves credentials lazily: it constructs happily
        # with none and only fails at call time. Checking it here means the
        # report can say "stage 3 disabled, no credentials" up front instead
        # of discovering it once per event.
        if not (getattr(client, "api_key", None)
                or getattr(client, "auth_token", None)
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or Path.home().joinpath(".config/anthropic").exists()):
            self.disabled_reason = (
                "no credentials (set ANTHROPIC_API_KEY or run `ant auth login`)")
            log.warning("LLM stage disabled: %s", self.disabled_reason)
            return
        self.client = client

    @property
    def available(self) -> bool:
        return self.client is not None and not self._latched_off

    # -- prompt ---------------------------------------------------------

    @staticmethod
    def _payload(e: RemediationEvent, cands: list[Order]) -> dict:
        return {
            "event": {
                "channel": str(e.channel),
                "occurred_at": e.occurred_at.isoformat(),
                "value_rupees": round(e.value_paise / 100, 2),
                "support_note": e.free_text,
                "card_descriptor": e.descriptor_text,
                "email": e.email,
                "phone": e.phone,
            },
            "candidates": [
                {
                    "order_id": o.order_id,
                    "placed_at": o.created_at.isoformat(),
                    "order_value_rupees": round(o.amount_paise / 100, 2),
                    "item": o.sku,
                    "payment_method": str(o.method),
                    "card_descriptor": o.descriptor,
                }
                for o in cands
            ],
        }

    def _cache_path(self, payload: dict) -> Path:
        h = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        return self.cache_dir / f"{self.model}-{h}.json"

    # -- inference ------------------------------------------------------

    def resolve_one(self, e: RemediationEvent,
                    cands: list[Order]) -> Resolution:
        abstain = Resolution(e.event_id, None, 0.0, Method.ABSTAINED)
        if not self.available or not cands:
            return abstain

        payload = self._payload(e, cands)
        cp = self._cache_path(payload)
        verdict: LinkVerdict | None = None

        if cp.exists():
            try:
                verdict = LinkVerdict.model_validate_json(cp.read_text())
                self.stats.cached += 1
            except Exception:                          # noqa: BLE001
                verdict = None

        if verdict is None:
            try:
                resp = self.client.messages.parse(
                    model=self.model,
                    max_tokens=1024,
                    # The system block is byte-identical on every call, so
                    # it caches; the per-event payload goes after it.
                    system=[{"type": "text", "text": SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user",
                               "content": json.dumps(payload, indent=2)}],
                    output_format=LinkVerdict,
                )
                verdict = resp.parsed_output
                self.stats.called += 1
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(verdict.model_dump_json())
            except Exception as exc:                   # noqa: BLE001
                # A stage-3 failure must degrade to abstention, never to a
                # guess and never to a crash: the human queue is the
                # designed fallback and it is always available.
                self.stats.errors += 1
                msg = str(exc)
                if "authentication" in msg.lower() or "api_key" in msg.lower():
                    self._auth_failures += 1
                    if self._auth_failures >= 3:
                        self._latched_off = True
                        self.disabled_reason = "authentication failed"
                        log.warning("LLM stage latched off after %d auth "
                                    "failures; remaining events go to the "
                                    "human queue", self._auth_failures)
                else:
                    log.warning("LLM call failed for %s: %s", e.event_id, exc)
                return abstain

        if verdict.order_id is None:
            self.stats.abstained += 1
            return abstain

        # Guardrail 1: closed set. Never trust an id we did not offer.
        allowed = {o.order_id for o in cands}
        if verdict.order_id not in allowed:
            self.stats.rejected_not_in_candidates += 1
            log.warning("LLM returned out-of-set order %s for %s",
                        verdict.order_id, e.event_id)
            return abstain

        if verdict.confidence < self.min_confidence:
            self.stats.abstained += 1
            return abstain

        self.stats.linked += 1
        return Resolution(e.event_id, verdict.order_id,
                          float(verdict.confidence), Method.LLM)


def apply_llm_stage(resolutions: list[Resolution],
                    events: dict[str, RemediationEvent],
                    candidates_for, resolver: LLMResolver,
                    limit: int | None = None) -> list[Resolution]:
    """Re-resolve abstained events through stage 3, leave the rest alone."""
    if not resolver.available:
        return resolutions
    out, n = [], 0
    for r in resolutions:
        if r.method is not Method.ABSTAINED or (limit is not None and n >= limit):
            out.append(r)
            continue
        e = events.get(r.event_id)
        if e is None:
            out.append(r)
            continue
        out.append(resolver.resolve_one(e, candidates_for(e)))
        n += 1
    return out
