"""CB5a — ExampleCompletenessValidator.

Safety-net gate that catches ``example`` blocks emitted by the content
rewrite tier which carry only a *problem statement* with no *worked
solution* — e.g. a one-line stub like::

    Find the sum of 5/6 and 7/9

with no intermediate steps, no result, and no answer. The existing
``lib/validators/instructional_depth.py`` gate only *counts* example
blocks (examples-per-concept ratio) and audits token-length on concept /
explanation surfaces — it never looks inside an ``example`` body. So a
page can satisfy every instructional-depth floor while shipping a stub
example. This validator closes that blind spot.

Different surface from ``InstructionalDepthValidator`` (page-level
density) and from the four ``Courseforge.router.inter_tier_gates``
``Block*Validator`` adapters (CURIE / content_type / page_objectives /
source_refs structural seams): this validator audits ONLY the *content
completeness* of each ``example`` block's body.

Per the CB5a spec (``plans/finegrain/content-block-quality-2026-06.md``
§CB5a): for each ``block_type == "example"`` block, after HTML-stripping
the body assert BOTH

  (a) the body carries >= ``min_word_tokens`` (default ~40) word tokens,
      AND
  (b) *solution evidence* is present — more than one ``<p>`` in the
      block, OR a result/answer marker (an ``=`` sign, one of the words
      "therefore" / "so" / "thus" / "result", or a final answer line
      distinct from the leading problem statement).

A miss emits a single ``EXAMPLE_BLOCK_INCOMPLETE`` GateIssue carrying
``action="regenerate"`` (the model can author the missing solution on a
re-roll). Non-example blocks pass through untouched. Empty / no-example
input gracefully passes.

NO embeddings — pure HTML / text. Mirrors the str/dict content-shape
dispatch pattern of the post-rewrite ``Block*Validator`` chain
(``Courseforge/router/inter_tier_gates.py``): rewrite-tier blocks carry
``content`` as an HTML string; outline-tier blocks carry it as a dict.

Inputs (``inputs`` dict):

    blocks: List[Block]
        Rewrite- (or outline-) tier ``Courseforge.scripts.blocks.Block``
        instances. Both content shapes are handled.

    thresholds: Optional[Dict[str, float]]
        Override the body-length floor. Key: ``min_word_tokens``
        (default 40).

    decision_capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance. One
        ``example_completeness_check`` event is emitted per audited
        ``example`` block (dynamic signals: ``block_id``, ``word_count``,
        ``has_solution_evidence``, ``p_count``).

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"example_completeness"``).

References:
    - ``plans/finegrain/content-block-quality-2026-06.md`` §CB5a.
    - ``lib/validators/instructional_depth.py`` — sibling authored-depth
      gate (capture-emit + HTML-strip pattern mirrored here).
    - ``Courseforge/router/inter_tier_gates.py`` — str/dict content-shape
      dispatch pattern for the four post-rewrite Block validators.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Import bridge for Block (mirror of instructional_depth.py).
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

#: Minimum word-token count in an ``example`` body. A stub problem
#: statement ("Find the sum of 5/6 and 7/9") is ~7 tokens; a worked
#: example with steps + answer is comfortably above 40.
DEFAULT_MIN_WORD_TOKENS: int = 40

#: Result / answer marker words. Presence of any one (case-insensitive,
#: whole-word) is solution evidence.
_RESULT_MARKER_WORDS: Tuple[str, ...] = ("therefore", "so", "thus", "result")

#: Cap on per-validate issue list to avoid runaway emit on many-example
#: corpora. Mirrors the cap used by sibling structural validators.
_ISSUE_LIST_CAP: int = 50


# ---------------------------------------------------------------------------
# HTML / token helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_TOKEN_RE = re.compile(r"\S+")
_P_OPEN_RE = re.compile(r"<p\b", re.IGNORECASE)
# Whole-word result-marker match (case-insensitive). Built once.
_RESULT_MARKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _RESULT_MARKER_WORDS) + r")\b",
    re.IGNORECASE,
)


def _strip_html_text(s: str) -> str:
    """Strip HTML tags + collapse whitespace.

    Mirror of ``inter_tier_gates._strip_html`` / ``instructional_depth
    ._strip_html_text`` — just enough to surface text content for regex
    matching, not a full DOM parser.
    """
    if not s:
        return ""
    text = _HTML_TAG_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _count_word_tokens(text: str) -> int:
    """Whitespace-separated word-token count over an already-stripped
    text surface."""
    if not text:
        return 0
    return len(_WORD_TOKEN_RE.findall(text))


def _example_body_html(block: Any) -> Optional[str]:
    """Return the raw HTML/text surface for an ``example`` block.

    Rewrite-tier (str shape): ``block.content`` returned as-is (HTML).
    Outline-tier (dict shape): joins the prose-bearing keys
    (``body`` / ``definition`` / ``explanation`` / ``description``) plus
    any ``key_claims`` list-of-str so a dict-shaped example is auditable
    too. Anything else: ``None`` (no surface to audit).
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
        if parts:
            return "\n".join(parts)
        return ""
    return None


def _count_p_tags(html: str) -> int:
    """Count opening ``<p>`` tags in an HTML surface. Dict-shaped (non-
    HTML) bodies have zero — solution evidence then leans on the result
    markers / answer-line heuristics instead."""
    if not html:
        return 0
    return len(_P_OPEN_RE.findall(html))


_HEADING_BLOCK_RE = re.compile(
    r"<h[1-6]\b[^>]*>.*?</h[1-6]\s*>", re.IGNORECASE | re.DOTALL
)

# Rewrite-template worked-solution structure markers. The rewrite tier emits
# worked examples whose intermediate steps and final answer live in semantic
# ``<div class="step-row">`` / ``<div class="solution-line">`` containers
# rather than the ``<p>`` paragraphs the legacy heuristics count. When such a
# block carries exactly one trailing recap ``<p>`` and writes its math in
# prose (no literal ``=`` and none of the result-marker words), every <p>-based
# / equals / marker heuristic collapses and a fully-worked example is flagged
# incomplete (measured on the sample-course-a 7B export: 2 false positives, 0
# genuine stubs). A ``solution-line`` answer container OR more than one
# ``step-row`` is solution evidence — precise enough that a one-line stub
# (no answer container, no step rows) still flags honestly.
_SOLUTION_LINE_RE = re.compile(
    r'class\s*=\s*["\'][^"\']*\bsolution-line\b', re.IGNORECASE
)
_STEP_ROW_RE = re.compile(
    r'class\s*=\s*["\'][^"\']*\bstep-row\b', re.IGNORECASE
)


def _has_worked_solution_structure(raw: str) -> bool:
    """Heuristic: the rewrite tier's div-based worked-solution structure.

    Recognizes a worked solution whose steps + answer are emitted in
    ``<div class="step-row">`` / ``<div class="solution-line">`` containers
    (the rewrite-template shape the ``<p>``-counting heuristics predate).
    Solution evidence is present when the block carries an explicit answer
    container (``solution-line``) OR more than one step row (``step-row``).
    A bare one-line problem stub has neither, so this branch is purely
    additive — it never weakens detection of a genuinely incomplete example.
    """
    if not raw:
        return False
    if _SOLUTION_LINE_RE.search(raw):
        return True
    return len(_STEP_ROW_RE.findall(raw)) > 1


def _has_final_answer_line(raw: str) -> bool:
    """Heuristic: a final answer line distinct from the problem statement.

    HTML path: more than one non-empty ``<p>`` paragraph where the LAST
    differs from the FIRST (a worked example renders problem + steps +
    answer across multiple paragraphs). A single ``<p>`` stub — even when
    wrapped in ``<section>`` / ``<h3>`` heading chrome — has exactly one
    paragraph, so it is NOT solution evidence.

    Text path (dict / no-markup, no ``<p>``): heading blocks are removed
    first (a heading line is chrome, not an answer), then the body must
    have more than one non-trivial line with the last distinct from the
    first.
    """
    # HTML paragraph split (rewrite-tier shape) — authoritative when <p>
    # tags are present. Heading chrome never contributes a paragraph.
    if _P_OPEN_RE.search(raw or ""):
        paras = [
            _strip_html_text(p)
            for p in re.split(r"</p\s*>", raw, flags=re.IGNORECASE)
        ]
        paras = [p for p in paras if p]
        return len(paras) > 1 and paras[-1] != paras[0]
    # No <p> markup: strip heading blocks, then split the remaining
    # surface on tag boundaries / newlines.
    deheaded = _HEADING_BLOCK_RE.sub("\n", raw or "")
    line_surface = _HTML_TAG_RE.sub("\n", deheaded)
    lines = [ln.strip() for ln in line_surface.splitlines() if ln.strip()]
    return len(lines) > 1 and lines[-1] != lines[0]


def _assess_example(
    block: Any, min_word_tokens: int,
) -> Tuple[int, bool, int]:
    """Return ``(word_count, has_solution_evidence, p_count)`` for one
    ``example`` block.

    Solution evidence is present when ANY of:
      * more than one ``<p>`` in the block HTML, OR
      * an ``=`` sign in the stripped text (a result line), OR
      * a result-marker word ("therefore"/"so"/"thus"/"result"), OR
      * a final answer line distinct from the leading problem statement, OR
      * the rewrite tier's div-based worked-solution structure (a
        ``solution-line`` answer container or >1 ``step-row``).
    """
    raw = _example_body_html(block)
    if raw is None:
        return 0, False, 0
    # Preserve line structure for the answer-line heuristic before the
    # whitespace collapse, but count tokens / scan markers on the
    # tag-stripped surface.
    p_count = _count_p_tags(raw)
    tag_stripped = _HTML_TAG_RE.sub(" ", raw)
    collapsed = _WHITESPACE_RE.sub(" ", tag_stripped).strip()
    word_count = _count_word_tokens(collapsed)

    has_evidence = False
    if p_count > 1:
        has_evidence = True
    elif "=" in collapsed:
        has_evidence = True
    elif _RESULT_MARKER_RE.search(collapsed):
        has_evidence = True
    elif _has_final_answer_line(raw):
        has_evidence = True
    elif _has_worked_solution_structure(raw):
        has_evidence = True

    return word_count, has_evidence, p_count


# ---------------------------------------------------------------------------
# Decision-capture emit
# ---------------------------------------------------------------------------


def _emit_decision(
    capture: Optional[Any],
    *,
    block_id: str,
    block_type: str,
    word_count: int,
    has_solution_evidence: bool,
    p_count: int,
    min_word_tokens: int,
    passed: bool,
) -> None:
    """Emit one ``example_completeness_check`` event per audited example
    block.

    Rationale interpolates the dynamic signals (block_id, word_count,
    has_solution_evidence, p_count) so post-hoc replay reconstructs the
    verdict without re-reading the gate result.
    """
    if capture is None:
        return
    verdict = "complete" if passed else "incomplete"
    rationale = (
        f"example_completeness audit of block {block_id!r} "
        f"(block_type={block_type}): word_count={word_count} "
        f"(floor={min_word_tokens}), has_solution_evidence="
        f"{has_solution_evidence}, p_count={p_count} -> {verdict}. "
        f"A stub example carrying only a problem statement (no worked "
        f"solution) is flagged for rewrite-tier regeneration."
    )
    ml_features = {
        "block_id": block_id,
        "block_type": block_type,
        "word_count": int(word_count),
        "has_solution_evidence": bool(has_solution_evidence),
        "p_count": int(p_count),
        "min_word_tokens": int(min_word_tokens),
        "passed": bool(passed),
    }
    try:
        capture.log_decision(
            decision_type="example_completeness_check",
            decision=verdict,
            rationale=rationale,
            ml_features=ml_features,
        )
    except Exception as exc:  # noqa: BLE001 - audit emit is best-effort
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "example_completeness_check: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class ExampleCompletenessValidator:
    """CB5a — stub-example safety-net gate.

    Validator-protocol-compatible class. Audits every ``example`` block
    for body completeness (length floor + solution evidence). A stub
    example emits a critical ``EXAMPLE_BLOCK_INCOMPLETE`` GateIssue with
    ``action="regenerate"``. Non-example blocks, empty input, and
    no-example input pass.
    """

    name = "example_completeness"
    version = "0.1.0"

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
        min_word_tokens = self._resolve_min_word_tokens(inputs)

        raw = inputs.get("blocks")
        if raw is None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="MISSING_BLOCKS_INPUT",
                        message=(
                            "inputs['blocks'] is required; expected a list "
                            "of Courseforge Block instances."
                        ),
                    )
                ],
                action="regenerate",
            )
        if not isinstance(raw, list):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="INVALID_BLOCKS_INPUT",
                        message=(
                            f"inputs['blocks'] must be a list; got "
                            f"{type(raw).__name__}."
                        ),
                    )
                ],
                action="regenerate",
            )

        blocks: List[Any] = list(raw)

        # Empty input → graceful pass (no examples to audit). Mirrors the
        # InstructionalDepthValidator empty-input parity.
        example_blocks = [
            b for b in blocks
            if getattr(b, "block_type", None) == "example"
        ]
        if not example_blocks:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
                action=None,
            )

        issues: List[GateIssue] = []
        incomplete = 0
        for b in example_blocks:
            block_id = str(
                getattr(b, "block_id", None)
                or getattr(b, "stable_id", None)
                or "<unknown>"
            )
            word_count, has_evidence, p_count = _assess_example(
                b, min_word_tokens,
            )
            passed_block = (
                word_count >= min_word_tokens and has_evidence
            )
            _emit_decision(
                capture,
                block_id=block_id,
                block_type="example",
                word_count=word_count,
                has_solution_evidence=has_evidence,
                p_count=p_count,
                min_word_tokens=min_word_tokens,
                passed=passed_block,
            )
            if passed_block:
                continue
            incomplete += 1
            if len(issues) >= _ISSUE_LIST_CAP:
                continue
            reasons: List[str] = []
            if word_count < min_word_tokens:
                reasons.append(
                    f"body has {word_count} word token(s) "
                    f"(floor {min_word_tokens})"
                )
            if not has_evidence:
                reasons.append(
                    "no solution evidence (need >1 <p>, an '=' result, a "
                    "result marker, or a distinct final answer line)"
                )
            issues.append(
                GateIssue(
                    severity="critical",
                    code="EXAMPLE_BLOCK_INCOMPLETE",
                    message=(
                        f"example block {block_id!r} looks like a problem "
                        f"statement with no worked solution: "
                        f"{'; '.join(reasons)}."
                    ),
                    location=block_id,
                    suggestion=(
                        "Re-roll the block through the rewrite tier with a "
                        "directive to author the full worked solution "
                        "(intermediate steps + a final answer), not only "
                        "the problem statement."
                    ),
                )
            )

        passed = incomplete == 0
        total_examples = len(example_blocks)
        score = (
            round((total_examples - incomplete) / total_examples, 4)
            if total_examples
            else 1.0
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )

    # ---------------------------------------------------------------- #
    # Threshold helper
    # ---------------------------------------------------------------- #

    def _resolve_min_word_tokens(self, inputs: Dict[str, Any]) -> int:
        """Resolve the body-length floor via inputs > constructor >
        default."""
        value = float(DEFAULT_MIN_WORD_TOKENS)
        if isinstance(self._thresholds_override, dict):
            v = self._thresholds_override.get("min_word_tokens")
            if v is not None:
                value = float(v)
        override = inputs.get("thresholds")
        if isinstance(override, dict):
            v = override.get("min_word_tokens")
            if v is not None:
                value = float(v)
        try:
            iv = int(value)
        except (TypeError, ValueError):
            iv = DEFAULT_MIN_WORD_TOKENS
        return iv if iv > 0 else DEFAULT_MIN_WORD_TOKENS


__all__ = [
    "ExampleCompletenessValidator",
    "DEFAULT_MIN_WORD_TOKENS",
]
