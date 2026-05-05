"""Worker W7 — InstructionalDepthValidator.

Per-page pedagogical-density gate for the Courseforge two-pass
authoring surface. Different surface from
``MinEdgeCountValidator`` (graph-completeness): this validator
audits authored-content depth — minimum concepts taught per page,
minimum examples per concept, minimum explanation tokens per concept.

Closes the GPT-feedback "no instructional-depth floors" gap (§1.6 of
``plans/gpt-feedback-w2-w7-execution-2026-05.md``). Different surface
from the graph-completeness ``MinEdgeCountValidator`` because a page
can score perfectly on edge counts while still emitting two-sentence
"concepts" with no examples and no explanation prose.

Inputs (``inputs`` dict):

    blocks: List[Block]
        Outline- or rewrite-tier ``Courseforge.scripts.blocks.Block``
        instances for a single page (or a flat list across pages — the
        validator groups by ``block.page_id``).

    thresholds: Optional[Dict[str, float]]
        Override the three default floors. Keys:
        ``min_concepts_per_page`` (default 2),
        ``min_examples_per_concept`` (default 1.0),
        ``min_explanation_tokens_per_concept`` (default 80).

    decision_capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance with a
        ``log_decision(decision_type, decision, rationale, **kw)``
        method. Exactly one ``instructional_depth_check`` event is
        emitted per ``validate()`` call with all three metrics on the
        rationale + ``ml_features`` dict.

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"instructional_depth"``).

GateIssue codes (severity ``critical`` per the user spec):

    - ``INSTRUCTIONAL_DEPTH_CONCEPTS_PER_PAGE_BELOW_THRESHOLD``
    - ``INSTRUCTIONAL_DEPTH_EXAMPLES_PER_CONCEPT_BELOW_THRESHOLD``
    - ``INSTRUCTIONAL_DEPTH_EXPLANATION_TOKENS_PER_CONCEPT_BELOW_THRESHOLD``

References:
    - ``plans/gpt-feedback-w2-w7-execution-2026-05.md`` — W7 spec.
    - ``lib/validators/min_edge_count.py`` — sibling
      graph-completeness gate with the symmetric below-floor pattern.
    - ``lib/validators/synthesis_diversity.py`` — capture-emit pattern
      mirrored here.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Import bridge for Block (mirror of concept_example_similarity.py).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover - import guard
    from blocks import Block  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for partial installs
    Block = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Minimum number of ``concept`` blocks per page. Per the W7 user spec.
DEFAULT_MIN_CONCEPTS_PER_PAGE: int = 2

#: Minimum ratio of ``example`` blocks to ``concept`` blocks per page.
#: 1.0 means every concept must have at least one accompanying example
#: on the same page.
DEFAULT_MIN_EXAMPLES_PER_CONCEPT: float = 1.0

#: Minimum average explanation-token count per concept. Tokens are
#: counted as whitespace-separated word tokens after HTML-stripping
#: the concept-body + adjacent prose surfaces (``explanation`` blocks
#: on the same page).
DEFAULT_MIN_EXPLANATION_TOKENS_PER_CONCEPT: int = 80

#: Cap on per-validate issue list to avoid runaway emit on many-page
#: corpora. Mirrors the cap used by sibling structural validators.
_ISSUE_LIST_CAP: int = 50


# ---------------------------------------------------------------------------
# HTML / token helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_TOKEN_RE = re.compile(r"\S+")


def _strip_html_text(s: str) -> str:
    """Strip HTML tags + collapse whitespace.

    Mirror of ``lib/validators/assessment.py::_strip_html_text`` so we
    don't pull a heavy import for one regex. Kept local because the
    sibling helper is in a class-heavy module.
    """
    if not s:
        return ""
    text = _HTML_TAG_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _count_word_tokens(text: str) -> int:
    """Whitespace-separated word-token count after HTML strip."""
    if not text:
        return 0
    stripped = _strip_html_text(text)
    if not stripped:
        return 0
    return len(_WORD_TOKEN_RE.findall(stripped))


def _extract_block_text_surface(block: Any) -> str:
    """Pull a textual surface out of a ``Block``-like instance.

    Outline-tier dict shape: joins ``content["body"]``, ``content["definition"]``,
    ``content["explanation"]``, and ``content["key_claims"]`` (list-of-str).
    Rewrite-tier str shape: returned as-is (HTML strip happens later).
    Anything else: empty string. Defensive — block_type=concept blocks
    can carry either shape depending on which tier emitted them.
    """
    content = getattr(block, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts: List[str] = []
        for key in ("body", "definition", "explanation", "description"):
            v = content.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        claims = content.get("key_claims")
        if isinstance(claims, list):
            for item in claims:
                if isinstance(item, str) and item.strip():
                    parts.append(item)
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Per-page metric computation
# ---------------------------------------------------------------------------


def _group_blocks_by_page(blocks: List[Any]) -> Dict[str, List[Any]]:
    """Group blocks by ``page_id``. Empty / missing page_id collapses
    to ``"__nopage__"`` so the validator still produces a result on
    legacy / unattributed inputs."""
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for b in blocks:
        page = getattr(b, "page_id", None) or "__nopage__"
        grouped[page].append(b)
    return grouped


def _compute_page_metrics(
    page_blocks: List[Any],
) -> Tuple[int, int, int]:
    """Return ``(n_concepts, n_examples, total_explanation_tokens)``
    for a single page's block list.

    The explanation-token count is the sum of word tokens across:
    1. every ``concept`` block's textual content surface, and
    2. every ``explanation`` block on the same page (treated as
       adjacent prose elaborating the concepts).

    The ratio metrics use the per-page concept count as the
    denominator; the validator handles the divide-by-zero case at the
    threshold-comparison site.
    """
    n_concepts = 0
    n_examples = 0
    explanation_tokens = 0
    for b in page_blocks:
        bt = getattr(b, "block_type", None)
        if bt == "concept":
            n_concepts += 1
            explanation_tokens += _count_word_tokens(
                _extract_block_text_surface(b)
            )
        elif bt == "example":
            n_examples += 1
        elif bt == "explanation":
            explanation_tokens += _count_word_tokens(
                _extract_block_text_surface(b)
            )
    return n_concepts, n_examples, explanation_tokens


# ---------------------------------------------------------------------------
# Decision-capture emit
# ---------------------------------------------------------------------------


def _emit_decision(
    capture: Optional[Any],
    *,
    passed: bool,
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    failure_codes: List[str],
    pages_audited: int,
) -> None:
    """Emit one ``instructional_depth_check`` event per validate() call.

    Mirrors the ``synthesis_diversity_check`` emit pattern: rationale
    interpolates the four signals (concepts/page, examples/concept,
    explanation tokens/concept, failure codes) so post-hoc replay
    reconstructs the verdict without re-reading the gate result.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{','.join(failure_codes) or 'unknown'}"
    rationale = (
        f"instructional_depth gate verdict: pages_audited={pages_audited}, "
        f"avg_concepts_per_page={metrics['avg_concepts_per_page']:.2f} "
        f"(floor={thresholds['min_concepts_per_page']}), "
        f"avg_examples_per_concept={metrics['avg_examples_per_concept']:.2f} "
        f"(floor={thresholds['min_examples_per_concept']:.2f}), "
        f"avg_explanation_tokens_per_concept="
        f"{metrics['avg_explanation_tokens_per_concept']:.1f} "
        f"(floor={thresholds['min_explanation_tokens_per_concept']}); "
        f"failure_codes={failure_codes or ['none']}."
    )
    ml_features = {
        "avg_concepts_per_page": float(metrics["avg_concepts_per_page"]),
        "avg_examples_per_concept": float(metrics["avg_examples_per_concept"]),
        "avg_explanation_tokens_per_concept": float(
            metrics["avg_explanation_tokens_per_concept"]
        ),
        "min_concepts_per_page": float(thresholds["min_concepts_per_page"]),
        "min_examples_per_concept": float(thresholds["min_examples_per_concept"]),
        "min_explanation_tokens_per_concept": float(
            thresholds["min_explanation_tokens_per_concept"]
        ),
        "pages_audited": int(pages_audited),
        "passed": bool(passed),
        "failure_codes": list(failure_codes),
    }
    try:
        capture.log_decision(
            decision_type="instructional_depth_check",
            decision=decision,
            rationale=rationale,
            ml_features=ml_features,
        )
    except Exception as exc:  # noqa: BLE001 - audit emit is best-effort
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "instructional_depth_check: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class InstructionalDepthValidator:
    """Per-page pedagogical-density gate (W7).

    Validator-protocol-compatible class. Audits authored Block lists
    for three depth metrics; each below-threshold metric emits a
    critical GateIssue with a code of the shape
    ``INSTRUCTIONAL_DEPTH_<METRIC>_BELOW_THRESHOLD``.
    """

    name = "instructional_depth"
    version = "1.0.0"

    def __init__(
        self,
        *,
        thresholds: Optional[Dict[str, float]] = None,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self._thresholds_override = thresholds
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        thresholds = self._resolve_thresholds(inputs)

        raw = inputs.get("blocks")
        if raw is None:
            issue = GateIssue(
                severity="critical",
                code="MISSING_BLOCKS_INPUT",
                message=(
                    "inputs['blocks'] is required; expected a list of "
                    "Courseforge Block instances."
                ),
            )
            _emit_decision(
                capture,
                passed=False,
                metrics={
                    "avg_concepts_per_page": 0.0,
                    "avg_examples_per_concept": 0.0,
                    "avg_explanation_tokens_per_concept": 0.0,
                },
                thresholds=thresholds,
                failure_codes=["MISSING_BLOCKS_INPUT"],
                pages_audited=0,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[issue],
            )
        if not isinstance(raw, list):
            issue = GateIssue(
                severity="critical",
                code="INVALID_BLOCKS_INPUT",
                message=(
                    f"inputs['blocks'] must be a list; got "
                    f"{type(raw).__name__}."
                ),
            )
            _emit_decision(
                capture,
                passed=False,
                metrics={
                    "avg_concepts_per_page": 0.0,
                    "avg_examples_per_concept": 0.0,
                    "avg_explanation_tokens_per_concept": 0.0,
                },
                thresholds=thresholds,
                failure_codes=["INVALID_BLOCKS_INPUT"],
                pages_audited=0,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[issue],
            )

        blocks: List[Any] = list(raw)
        if not blocks:
            # Empty-input parity with MinEdgeCountValidator: pass with
            # neutral score so a page with zero authored blocks doesn't
            # block the gate (the upstream phase will fail on its own
            # surface). The capture event records the no-op.
            metrics = {
                "avg_concepts_per_page": 0.0,
                "avg_examples_per_concept": 0.0,
                "avg_explanation_tokens_per_concept": 0.0,
            }
            _emit_decision(
                capture,
                passed=True,
                metrics=metrics,
                thresholds=thresholds,
                failure_codes=[],
                pages_audited=0,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        # ---------- Per-page metric computation ----------
        grouped = _group_blocks_by_page(blocks)
        per_page_records: List[Dict[str, Any]] = []
        total_concepts = 0
        total_examples = 0
        total_explanation_tokens = 0
        for page_id, page_blocks in grouped.items():
            n_concepts, n_examples, exp_tokens = _compute_page_metrics(
                page_blocks
            )
            total_concepts += n_concepts
            total_examples += n_examples
            total_explanation_tokens += exp_tokens
            per_page_records.append(
                {
                    "page_id": page_id,
                    "n_concepts": n_concepts,
                    "n_examples": n_examples,
                    "explanation_tokens": exp_tokens,
                }
            )

        pages_audited = len(grouped)

        # Aggregate metrics for the rationale + ml_features. Per-page
        # records drive the GateIssue emit so the operator sees which
        # page tripped each floor.
        if pages_audited == 0:  # pragma: no cover - covered by empty-list short-circuit
            avg_concepts_per_page = 0.0
        else:
            avg_concepts_per_page = total_concepts / pages_audited
        if total_concepts == 0:
            avg_examples_per_concept = 0.0
            avg_explanation_tokens_per_concept = 0.0
        else:
            avg_examples_per_concept = total_examples / total_concepts
            avg_explanation_tokens_per_concept = (
                total_explanation_tokens / total_concepts
            )

        # ---------- Per-page threshold checks ----------
        issues: List[GateIssue] = []
        failure_codes: List[str] = []

        below_concepts_pages = [
            r for r in per_page_records
            if r["n_concepts"] < thresholds["min_concepts_per_page"]
        ]
        if below_concepts_pages:
            failure_codes.append(
                "INSTRUCTIONAL_DEPTH_CONCEPTS_PER_PAGE_BELOW_THRESHOLD"
            )
            for r in below_concepts_pages[:_ISSUE_LIST_CAP]:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code=(
                            "INSTRUCTIONAL_DEPTH_CONCEPTS_PER_PAGE"
                            "_BELOW_THRESHOLD"
                        ),
                        message=(
                            f"page {r['page_id']!r} emits "
                            f"{r['n_concepts']} concept block(s); minimum "
                            f"floor is {thresholds['min_concepts_per_page']}."
                        ),
                        location=str(r["page_id"]),
                        suggestion=(
                            "Re-roll the page through the rewrite tier "
                            "with a prompt directive to author at least "
                            f"{thresholds['min_concepts_per_page']} "
                            "distinct concepts."
                        ),
                    )
                )

        # Examples-per-concept: only meaningful when there's at least
        # one concept on the page. Pages with zero concepts trip the
        # concepts-per-page floor first (above) and don't double-fire
        # an examples ratio code.
        below_examples_pages: List[Dict[str, Any]] = []
        for r in per_page_records:
            if r["n_concepts"] <= 0:
                continue
            ratio = r["n_examples"] / r["n_concepts"]
            if ratio < thresholds["min_examples_per_concept"]:
                r_with_ratio = dict(r)
                r_with_ratio["ratio"] = ratio
                below_examples_pages.append(r_with_ratio)
        if below_examples_pages:
            failure_codes.append(
                "INSTRUCTIONAL_DEPTH_EXAMPLES_PER_CONCEPT_BELOW_THRESHOLD"
            )
            for r in below_examples_pages[:_ISSUE_LIST_CAP]:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code=(
                            "INSTRUCTIONAL_DEPTH_EXAMPLES_PER_CONCEPT"
                            "_BELOW_THRESHOLD"
                        ),
                        message=(
                            f"page {r['page_id']!r} has "
                            f"{r['n_examples']} example(s) for "
                            f"{r['n_concepts']} concept(s) "
                            f"(ratio={r['ratio']:.2f}); minimum "
                            f"floor is "
                            f"{thresholds['min_examples_per_concept']:.2f}."
                        ),
                        location=str(r["page_id"]),
                        suggestion=(
                            "Author at least one example block per "
                            "concept on this page."
                        ),
                    )
                )

        below_tokens_pages: List[Dict[str, Any]] = []
        for r in per_page_records:
            if r["n_concepts"] <= 0:
                continue
            avg_tokens = r["explanation_tokens"] / r["n_concepts"]
            if (
                avg_tokens
                < thresholds["min_explanation_tokens_per_concept"]
            ):
                r_with_tokens = dict(r)
                r_with_tokens["avg_tokens"] = avg_tokens
                below_tokens_pages.append(r_with_tokens)
        if below_tokens_pages:
            failure_codes.append(
                "INSTRUCTIONAL_DEPTH_EXPLANATION_TOKENS_PER_CONCEPT"
                "_BELOW_THRESHOLD"
            )
            for r in below_tokens_pages[:_ISSUE_LIST_CAP]:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code=(
                            "INSTRUCTIONAL_DEPTH_EXPLANATION_TOKENS"
                            "_PER_CONCEPT_BELOW_THRESHOLD"
                        ),
                        message=(
                            f"page {r['page_id']!r} averages "
                            f"{r['avg_tokens']:.1f} explanation tokens "
                            f"per concept "
                            f"(total={r['explanation_tokens']} "
                            f"across {r['n_concepts']} concept(s)); "
                            f"minimum floor is "
                            f"{thresholds['min_explanation_tokens_per_concept']}."
                        ),
                        location=str(r["page_id"]),
                        suggestion=(
                            "Expand the concept-block bodies (or add "
                            "adjacent explanation blocks) so the page "
                            "averages at least "
                            f"{thresholds['min_explanation_tokens_per_concept']} "
                            "tokens per concept."
                        ),
                    )
                )

        passed = len(failure_codes) == 0
        # Score: harmonic-style aggregate over the three depth signals
        # normalised against thresholds. Capped at [0, 1]; mirrors the
        # convention used by min_edge_count.py.
        score = self._compute_score(
            avg_concepts_per_page=avg_concepts_per_page,
            avg_examples_per_concept=avg_examples_per_concept,
            avg_explanation_tokens_per_concept=avg_explanation_tokens_per_concept,
            thresholds=thresholds,
        )

        metrics = {
            "avg_concepts_per_page": avg_concepts_per_page,
            "avg_examples_per_concept": avg_examples_per_concept,
            "avg_explanation_tokens_per_concept": avg_explanation_tokens_per_concept,
        }
        _emit_decision(
            capture,
            passed=passed,
            metrics=metrics,
            thresholds=thresholds,
            failure_codes=failure_codes,
            pages_audited=pages_audited,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    # ---------------------------------------------------------------- #
    # Threshold + score helpers
    # ---------------------------------------------------------------- #

    def _resolve_thresholds(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Resolve thresholds via inputs > constructor > defaults."""
        defaults = {
            "min_concepts_per_page": float(DEFAULT_MIN_CONCEPTS_PER_PAGE),
            "min_examples_per_concept": float(DEFAULT_MIN_EXAMPLES_PER_CONCEPT),
            "min_explanation_tokens_per_concept": float(
                DEFAULT_MIN_EXPLANATION_TOKENS_PER_CONCEPT
            ),
        }
        if isinstance(self._thresholds_override, dict):
            for k, v in self._thresholds_override.items():
                if k in defaults and v is not None:
                    defaults[k] = float(v)
        override = inputs.get("thresholds")
        if isinstance(override, dict):
            for k, v in override.items():
                if k in defaults and v is not None:
                    defaults[k] = float(v)
        return defaults

    @staticmethod
    def _compute_score(
        *,
        avg_concepts_per_page: float,
        avg_examples_per_concept: float,
        avg_explanation_tokens_per_concept: float,
        thresholds: Dict[str, float],
    ) -> float:
        """Three signals normalised to [0, 1] and averaged."""
        def _ratio(value: float, floor: float) -> float:
            if floor <= 0:
                return 1.0
            return min(1.0, max(0.0, value / floor))

        s_concepts = _ratio(
            avg_concepts_per_page, thresholds["min_concepts_per_page"]
        )
        s_examples = _ratio(
            avg_examples_per_concept, thresholds["min_examples_per_concept"]
        )
        s_tokens = _ratio(
            avg_explanation_tokens_per_concept,
            thresholds["min_explanation_tokens_per_concept"],
        )
        return round((s_concepts + s_examples + s_tokens) / 3.0, 4)


__all__ = [
    "InstructionalDepthValidator",
    "DEFAULT_MIN_CONCEPTS_PER_PAGE",
    "DEFAULT_MIN_EXAMPLES_PER_CONCEPT",
    "DEFAULT_MIN_EXPLANATION_TOKENS_PER_CONCEPT",
]
