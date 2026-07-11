"""Stage 8: Escalation routing. Rule-based. No model.

Contract:
    should_escalate(report: DocumentReport) -> EscalationDecision

After stages 1-7 run, we know:
  - axe-core serious/critical violation counts
  - mean stage-3 classification confidence (pre-Qwen) or override rate (post-Qwen)
  - how often stage-6 fell back to the offline-Qwen enrichment lane
    (the ``claude_fallback_count`` field keeps its legacy v1 name)
  - document size (page count, block count)

Route the document to one of three outcomes:
  "ship"         — outputs are clean enough to release without further work
  "llm_fallback" — send the document (or the uncertain regions) back
                   through the local offline-Qwen lane (higher K, hotter
                   sampling) for a second pass before ship. No external
                   LLM is involved — the pipeline runs entirely on local
                   models + deterministic code at runtime. The actual
                   retry call lives outside this module (see
                   ``theta.offline_retry.maybe_offline_retry``); this
                   verdict just signals the decision.
  "fail"         — we can't produce accessible output even after the
                   offline retry; block release.

NOTE: this module is the legacy v1 escalation policy. The shipped
cascade decides retry-vs-ship in Stage 12/13
(``theta.offline_retry`` + ``theta.exits.decide_exit``); the verdict
names here predate that and read ``llm_fallback`` for historical
reasons. There is no human-review exit (architecture §11: "No human
escalation"), and there has never been a frontier/external-LLM path —
the second pass is always the local offline-Qwen lane.

Thresholds are module-level so they can be tuned from config rather
than code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentReport:
    n_blocks: int
    n_axe_serious: int
    n_axe_critical: int
    mean_classification_confidence: float
    claude_fallback_count: int
    page_count: int
    # Post-Qwen: fraction of blocks where Qwen's final label differed from
    # DistilBERT's hint. None when Qwen did not run. Post-Qwen, mean_conf
    # is always 1.0 (Qwen doesn't emit logprobs back), so this is our
    # replacement uncertainty signal.
    qwen_override_rate: float | None = None


@dataclass
class EscalationDecision:
    verdict: str  # "ship" | "llm_fallback" | "fail"
    reasons: list[str] = field(default_factory=list)


# Conservative defaults. Document owners can tune per-tenant.
SERIOUS_VIOLATION_SHIP_MAX = 0
CRITICAL_VIOLATION_SHIP_MAX = 0
CONFIDENCE_SHIP_MIN = 0.75
CLAUDE_FALLBACK_PER_PAGE_MAX = 3
CRITICAL_VIOLATION_FAIL_MIN = 10       # above this, fail entirely
# Fraction of blocks where Qwen overrode DistilBERT. High override rate
# signals either DistilBERT is systematically mislabeling this document
# or Qwen itself is thrashing on chunk boundaries — either way worth a
# human look. Well-calibrated inference should sit well below this.
QWEN_OVERRIDE_SHIP_MAX = 0.35


def should_escalate(report: DocumentReport) -> EscalationDecision:
    reasons: list[str] = []

    if report.n_axe_critical >= CRITICAL_VIOLATION_FAIL_MIN:
        reasons.append(
            f"{report.n_axe_critical} critical axe violations "
            f"(>= {CRITICAL_VIOLATION_FAIL_MIN} = fail threshold)"
        )
        return EscalationDecision(verdict="fail", reasons=reasons)

    if report.n_axe_critical > CRITICAL_VIOLATION_SHIP_MAX:
        reasons.append(f"{report.n_axe_critical} critical axe violations "
                       f"(ship limit: {CRITICAL_VIOLATION_SHIP_MAX})")
    if report.n_axe_serious > SERIOUS_VIOLATION_SHIP_MAX:
        reasons.append(f"{report.n_axe_serious} serious axe violations "
                       f"(ship limit: {SERIOUS_VIOLATION_SHIP_MAX})")
    # Per-block confidence is meaningful pre-Qwen. Post-Qwen every block
    # is 1.0 by construction, so this check effectively stops firing —
    # see the override-rate check below, which replaces it.
    if report.mean_classification_confidence < CONFIDENCE_SHIP_MIN:
        reasons.append(
            f"mean classification confidence {report.mean_classification_confidence:.2f} "
            f"< {CONFIDENCE_SHIP_MIN}"
        )
    if (report.qwen_override_rate is not None
            and report.qwen_override_rate > QWEN_OVERRIDE_SHIP_MAX):
        reasons.append(
            f"Qwen overrode DistilBERT on "
            f"{report.qwen_override_rate*100:.0f}% of blocks "
            f"(> {QWEN_OVERRIDE_SHIP_MAX*100:.0f}% ship limit)"
        )
    claude_per_page = (report.claude_fallback_count / report.page_count
                       if report.page_count else 0)
    if claude_per_page > CLAUDE_FALLBACK_PER_PAGE_MAX:
        reasons.append(
            f"{claude_per_page:.1f} Claude fallbacks/page "
            f"(> {CLAUDE_FALLBACK_PER_PAGE_MAX})"
        )

    if reasons:
        return EscalationDecision(verdict="llm_fallback", reasons=reasons)
    return EscalationDecision(verdict="ship", reasons=["all thresholds met"])
