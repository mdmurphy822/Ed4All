#!/usr/bin/env python3
"""
Trainforge Assessment Generator

Generates assessments from course content with decision capture for training data.

Pipeline Position:
    IMSCC Package → RAG Index → [Assessment Generator] → Validated Assessments

Decision Capture:
    All generation decisions logged for model training.
"""

import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

# Add project path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # → Ed4All/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

# Runtime import for the MLFeatures dataclass used by the
# assessment_template_skip decision-capture events. Best-effort: when
# lib.decision_capture isn't importable (minimal test harness), fall back to
# None so the runtime keeps working without the helper.
try:
    from lib.decision_capture import MLFeatures  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    MLFeatures = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Observable cognitive task-type axis, gated behind
# TRAINFORGE_COGNITIVE_TASK_TYPE. Default unset -> the
# QuestionData.cognitive_task_type field stays None and to_dict() omits it, so
# legacy assessment output is byte-identical. When the flag is truthy, each
# emitted question is tagged with the task verb detected in its stem.
_COGNITIVE_TASK_TYPE_ENV = "TRAINFORGE_COGNITIVE_TASK_TYPE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _cognitive_task_type_enabled() -> bool:
    """Return True when TRAINFORGE_COGNITIVE_TASK_TYPE is truthy."""
    return os.environ.get(_COGNITIVE_TASK_TYPE_ENV, "").strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- #
# Item-writing rules at generation
# --------------------------------------------------------------------------- #
#
# Rodriguez's meta-analysis: a 3-option MC (1 key + 2 functional distractors)
# discriminates as well as or better than 4/5-option items — the 3rd+
# distractors are non-functional. Default to 3 options total; an operator can
# restore the legacy 4-option layout with ED4ALL_ASSESSMENT_OPTION_COUNT=4.
# Parse-with-fallback: garbage / <3 / >6 → 3.
_OPTION_COUNT_ENV = "ED4ALL_ASSESSMENT_OPTION_COUNT"


def _resolve_option_count() -> int:
    """Total options per MC item (1 key + N-1 distractors); default 3.

    Reads ED4ALL_ASSESSMENT_OPTION_COUNT. Any non-integer, or a value outside
    the sane [3, 6] band, falls back to the 3-option default (parse-with-
    fallback — a partly-garbage override never yields a degenerate item).
    """
    raw = os.environ.get(_OPTION_COUNT_ENV, "").strip()
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 3
    if val < 3 or val > 6:
        return 3
    return val


# --------------------------------------------------------------------------- #
# Bounded LLM apply-arm (flag-gated, default OFF).
#
# Route apply word-problems + per-distractor misconception prose through the
# in-tree ``AssessmentGeneratorProvider.generate_assessments`` (local registry
# seats only), then run a MANDATORY three-stage verify chain before an
# LLM-drafted item ships:
#   (1) sympy re-check on numeric stems/keys — FAIL-CLOSED (a declared numeric
#       key that cannot be verified is dropped);
#   (2) score_groundedness entailment of stem+key vs the item's cited chunks —
#       require ENTAILED, drop unsupported/contradicted;
#   (3) Bloom trivote agreement before an apply/analyze claim stands — GPU-gated:
#       when the trivote is unavailable (< 2 independent voters) the item's
#       bloom CLAMPS to its structural ceiling instead of shipping unverified.
# The deterministic tier stays the offline fallback; this arm only ADDS a
# bounded count of triple-verified items. Default off → byte-identical.
# --------------------------------------------------------------------------- #
_APPLY_ARM_ENV = "ED4ALL_ASSESSMENT_APPLY_ARM"
_APPLY_ARM_MAX_ENV = "ED4ALL_ASSESSMENT_APPLY_ARM_MAX"
#: Providers the apply-arm refuses (ToS-restricted for the assessment corpus,
#: which feeds the SLM training surface — license-clean local seats only).
_APPLY_ARM_BARRED_PROVIDERS = frozenset({"anthropic", "claude_session"})
#: item_subtype for an LLM-drafted apply word-problem. Its structural ceiling
#: is ``apply``; a higher CLAIM is clamped when the trivote can't back it.
_APPLY_ARM_SUBTYPE = "apply_word_problem"


def _apply_arm_enabled() -> bool:
    """Return True when ED4ALL_ASSESSMENT_APPLY_ARM is truthy (default off)."""
    return os.environ.get(_APPLY_ARM_ENV, "").strip().lower() in _TRUTHY


def _apply_arm_max() -> int:
    """Bounded per-quiz apply-arm draft count; default 4 (parse-with-fallback).

    Any non-integer / non-positive override falls back to 4 so a partly-garbage
    value never yields an unbounded (or zero) LLM draft budget.
    """
    raw = os.environ.get(_APPLY_ARM_MAX_ENV, "").strip()
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 4
    return val if val > 0 else 4


def _apply_arm_provider_allowed(provider_name: str, roster: Any = None) -> bool:
    """License guard for the apply-arm provider seat.

    Refuses the ToS-restricted Anthropic seats outright. When a ``roster``
    (endpoint-registry-shaped mapping of ``provider -> {license verdict ...}``)
    is supplied AND carries a ``verdict`` of ``barred`` for this provider, the
    seat is refused too (multi-teacher roster license flow-down). Absent a
    roster / license field, a non-Anthropic local seat is allowed.
    """
    name = (provider_name or "").strip().lower()
    if not name or name in _APPLY_ARM_BARRED_PROVIDERS:
        return False
    if isinstance(roster, dict):
        entry = roster.get(name) or {}
        if isinstance(entry, dict):
            verdict = str(entry.get("verdict") or entry.get("license_verdict")
                          or "").strip().lower()
            if verdict == "barred":
                return False
    return True


#: Banned option surface forms (Haladyna–Downing–Rodriguez guidelines): the
#: complex-alternative crutches "None/All of the above" and absolute-term
#: options (always/never) that a test-wise learner exploits. Enforced at
#: ASSEMBLY so a banned string never reaches choices[] (the _negate_statement
#: always↔never swap is kept for its unit-tested verifiably-false contract, but
#: any option it produces carrying an absolute term is dropped here and the
#: caller draws a replacement / skips).
_RE_NOTA_AOTA = re.compile(
    r"\b(?:none|all|both|neither)\s+of\s+the\s+(?:above|following|other)\b",
    re.IGNORECASE,
)
_RE_ABSOLUTE_TERM = re.compile(r"\b(?:always|never)\b", re.IGNORECASE)


def _is_banned_option(text: str) -> bool:
    """True when option ``text`` is a banned MC surface form.

    Strips HTML before matching so ``<p>None of the above</p>`` is caught.
    Bans None/All-of-the-above complex alternatives and absolute-term
    (always/never) options.
    """
    plain = re.sub(r"<[^>]+>", " ", text or "")
    if _RE_NOTA_AOTA.search(plain):
        return True
    if _RE_ABSOLUTE_TERM.search(plain):
        return True
    return False


#: Named deterministic misconception TRANSFORMS (the DiVERT principle: a
#: functional distractor encodes a specific procedural error, not surface
#: plausibility). Each transform is a symbolic/text rewrite of a correct
#: worked-solution step; the tuple is (error_name, callable, human_note). The
#: callable returns the perturbed text OR None when it does not apply (so a
#: step that carries no matching structure is skipped, never mangled).
def _t_sign_drop(text: str):
    """Drop a leading unary minus / flip the first ``- N`` term to ``+ N``.

    Models the "dropped a negative sign" error. Applies to the first
    ``<op> <number>`` where op is a binary minus.
    """
    m = re.search(r"(?<=[\dA-Za-z\)])\s*-\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    return text[: m.start()] + " + " + m.group(1) + text[m.end():]


def _t_fail_distribute_negative(text: str):
    """Distribute a negative over only the FIRST term of a parenthesised sum.

    ``-(a + b)`` → ``-a + b`` (forgot to flip the second sign). Models the
    classic fail-to-distribute-negative error.
    """
    m = re.search(r"-\s*\(\s*([^()+]+?)\s*\+\s*([^()]+?)\s*\)", text)
    if not m:
        return None
    return text[: m.start()] + f"-{m.group(1).strip()} + {m.group(2).strip()}" + text[m.end():]


def _t_add_exponents_on_multiply(text: str):
    """Rewrite ``x^a * x^b`` as ``x^(a*b)`` instead of ``x^(a+b)``.

    Models the add-exponents-on-multiply / multiply-exponents confusion —
    here the perturbation multiplies where addition is correct.
    """
    m = re.search(r"([A-Za-z])\^(\d+)\s*\*\s*\1\^(\d+)", text)
    if not m:
        return None
    var, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    return text[: m.start()] + f"{var}^{a * b}" + text[m.end():]


def _t_left_to_right_not_pemdas(text: str):
    """Evaluate ``a + b * c`` strictly left-to-right → ``(a + b) * c``.

    Models the order-of-operations (ignore PEMDAS) error. Only rewrites the
    display form (a distractor STEP), not a computed value.
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    return text[: m.start()] + f"({m.group(1)} + {m.group(2)}) * {m.group(3)}" + text[m.end():]


#: (error_name, transform, instructor-facing misconception note).
_MISCONCEPTION_TRANSFORMS = (
    ("sign_drop", _t_sign_drop,
     "Dropped a negative sign when moving a term across the equals sign."),
    ("fail_to_distribute_negative", _t_fail_distribute_negative,
     "Distributed the negative over only the first term of the parentheses."),
    ("add_exponents_on_multiply", _t_add_exponents_on_multiply,
     "Multiplied the exponents instead of adding them when multiplying like bases."),
    ("left_to_right_not_pemdas", _t_left_to_right_not_pemdas,
     "Evaluated left-to-right instead of following order of operations (PEMDAS)."),
)


def _apply_first_misconception(text: str):
    """Return ``(perturbed_text, error_name, note)`` for the first applicable
    misconception transform, or ``None`` when none applies to ``text``.
    """
    for name, fn, note in _MISCONCEPTION_TRANSFORMS:
        out = fn(text)
        if out is not None and out.strip() and out.strip() != (text or "").strip():
            return out, name, note
    return None


# --------------------------------------------------------------------------- #
# Bloom-honesty ceiling — a type may CLAIM apply/analyze only if its structure
# supports it. Single-fact MC/TF/term-FIB/matching cap at understand; the
# constructed-response / multi-step types (MR, error-analysis, numeric,
# ordering, two-tier) may claim up to apply/analyze. An item's asserted
# bloom_level is CLAMPED to its subtype ceiling; clamps are recorded in the
# generation stats (deterministic — no LLM, so no DecisionCapture required).
# --------------------------------------------------------------------------- #
from lib.ontology.bloom import BLOOM_LEVELS as _BLOOM_LEVELS  # noqa: E402

_BLOOM_ORDER = {name: i for i, name in enumerate(_BLOOM_LEVELS)}

_SUBTYPE_BLOOM_CEILING: Dict[str, str] = {
    "mc_single": "understand",
    "tf": "understand",
    "fib_term": "understand",
    "matching_mc": "understand",
    "fib_numeric": "apply",
    "mc_multiple_response": "apply",
    "ordering_mc": "apply",
    "two_tier_answer": "analyze",
    "two_tier_reason": "analyze",
    "error_analysis": "analyze",
    # LLM-drafted apply word-problem: genuinely assesses apply, so a higher
    # CLAIM (analyze+) that the trivote can't back clamps down to here.
    "apply_word_problem": "apply",
}


def _clamp_bloom(bloom_level: str, item_subtype: Optional[str]) -> str:
    """Clamp ``bloom_level`` down to ``item_subtype``'s structural ceiling.

    Returns the (possibly lowered) level. Unknown subtype / level → unchanged
    (fail-open — never RAISES a claim, only lowers an over-claim). The item
    STATEMENT is untouched; only the asserted metadata level moves.
    """
    ceiling = _SUBTYPE_BLOOM_CEILING.get(item_subtype or "")
    if ceiling is None:
        return bloom_level
    cur = _BLOOM_ORDER.get((bloom_level or "").lower())
    cap = _BLOOM_ORDER.get(ceiling)
    if cur is None or cap is None:
        return bloom_level
    if cur > cap:
        return ceiling
    return bloom_level


# --------------------------------------------------------------------------- #
# Deterministic diversified-tier target mix (fractions sum to ~0.90; classic
# single-answer MC fills the remainder). Consumed by the mix PLANNER, which
# turns a question_count into a deterministic, reproducible list of item
# subtypes.
# --------------------------------------------------------------------------- #
_TARGET_MIX = (
    ("mc_multiple_response", 0.12),
    ("error_analysis", 0.15),
    ("fib_numeric", 0.15),
    ("matching_mc", 0.12),
    ("ordering_mc", 0.05),
    ("two_tier_answer", 0.08),
    ("tf_justified", 0.05),
    # Written-response constructed-answer types (~15% of the diversified mix):
    # the honest higher-Bloom vehicle (NO Bloom ceiling — see the ceiling table).
    ("short_answer", 0.09),
    ("extended_response", 0.06),
    # remainder → mc_single
)


#: Fixed 2/1/0 analytic score scale reused by every written-response criterion.
#: The descriptors are generic scaffold labels; the specific expectation lives
#: in the (source-derived) criterion text, so nothing content-specific is
#: fabricated here.
_WRITTEN_RUBRIC_LEVELS: List[Dict[str, Any]] = [
    {"score": 2, "descriptor": "Fully and accurately addresses this criterion."},
    {"score": 1, "descriptor": "Partially addresses this criterion."},
    {"score": 0, "descriptor": "Does not address this criterion."},
]


def plan_item_mix(question_count: int) -> List[str]:
    """Deterministic largest-remainder allocation of subtypes for a count.

    Returns a list of length ``question_count`` of item-subtype tokens per
    ``_TARGET_MIX``; classic ``mc_single`` fills the remainder. Deterministic +
    reproducible (no RNG) — the same count always yields the same plan, which
    keeps resume / replay stable. A count of 0 → [].
    """
    n = max(0, int(question_count))
    if n == 0:
        return []
    # Largest-remainder apportionment so the emitted plan matches the target
    # fractions as closely as an integer count allows.
    raw = [(name, frac * n) for name, frac in _TARGET_MIX]
    counts = {name: int(val) for name, val in raw}
    allocated = sum(counts.values())
    # Distribute leftover (from flooring) to the largest fractional parts first,
    # capping total diversified at ~90% so mc_single always gets a share.
    frac_parts = sorted(
        ((val - int(val), name) for name, val in raw), reverse=True
    )
    max_diversified = int(round(0.90 * n))
    i = 0
    while allocated < max_diversified and frac_parts and i < 4 * len(frac_parts):
        _part, name = frac_parts[i % len(frac_parts)]
        counts[name] = counts.get(name, 0) + 1
        allocated += 1
        i += 1
    counts["mc_single"] = n - allocated
    # Emit in a stable interleaved order (round-robin over the subtypes present)
    # so a small quiz still spans several types instead of clustering.
    order = [name for name, _ in _TARGET_MIX] + ["mc_single"]
    plan: List[str] = []
    pools = {name: counts.get(name, 0) for name in order}
    while len(plan) < n:
        progressed = False
        for name in order:
            if pools[name] > 0:
                plan.append(name)
                pools[name] -= 1
                progressed = True
                if len(plan) >= n:
                    break
        if not progressed:
            break
    return plan


@dataclass
class GenerationStats:
    """Deterministic-tier generation telemetry (no LLM).

    Records the realized item-subtype histogram, Bloom-ceiling clamps, and
    per-subtype skip reasons so an operator can audit WHY the realized mix
    diverges from the target without grepping decision JSONL.
    """
    subtype_counts: Dict[str, int] = field(default_factory=dict)
    bloom_clamps: List[Dict[str, str]] = field(default_factory=list)
    skips: List[Dict[str, str]] = field(default_factory=list)

    def record_item(self, subtype: str) -> None:
        self.subtype_counts[subtype] = self.subtype_counts.get(subtype, 0) + 1

    def record_clamp(
        self, question_id: str, subtype: str, before: str, after: str
    ) -> None:
        self.bloom_clamps.append({
            "question_id": question_id,
            "item_subtype": subtype,
            "from": before,
            "to": after,
        })

    def record_skip(self, subtype: str, reason: str) -> None:
        self.skips.append({"item_subtype": subtype, "reason": reason})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtype_counts": dict(self.subtype_counts),
            "bloom_clamp_count": len(self.bloom_clamps),
            "bloom_clamps": self.bloom_clamps[:20],
            "skip_count": len(self.skips),
            "skips": self.skips[:20],
        }


# Import content extractor for content-grounded generation
from Trainforge.generators.content_extractor import ContentExtractor

# Import leak checker for answer-leak detection
try:
    from lib.leak_checker import LeakChecker  # noqa: F401
    LEAK_CHECKER_AVAILABLE = True
except ImportError:
    LEAK_CHECKER_AVAILABLE = False

# Source of truth for the verb lists: schemas/taxonomies/bloom_verbs.json
# via lib.ontology.bloom. Only the verb portion is loaded from the canonical
# taxonomy; patterns + question_types below are still local.
from lib.ontology.bloom import get_verbs_list as _get_canonical_verbs_list  # noqa: E402

_CANONICAL_VERBS = _get_canonical_verbs_list()


# Bloom's Taxonomy levels with associated question patterns
BLOOM_LEVELS = {
    "remember": {
        "verbs": _CANONICAL_VERBS["remember"],
        "patterns": ["What is...?", "List the...", "Which of the following...?"],
        "question_types": ["multiple_choice", "true_false", "fill_in_blank"],
    },
    "understand": {
        "verbs": _CANONICAL_VERBS["understand"],
        "patterns": ["Explain why...", "Describe how...", "What does X mean?"],
        "question_types": ["multiple_choice", "short_answer", "fill_in_blank"],
    },
    "apply": {
        "verbs": _CANONICAL_VERBS["apply"],
        "patterns": ["How would you use...?", "Apply X to...", "Solve..."],
        "question_types": ["multiple_choice", "short_answer", "essay"],
    },
    "analyze": {
        "verbs": _CANONICAL_VERBS["analyze"],
        "patterns": ["Compare and contrast...", "What are the differences...", "Analyze..."],
        "question_types": ["multiple_choice", "essay", "short_answer"],
    },
    "evaluate": {
        "verbs": _CANONICAL_VERBS["evaluate"],
        "patterns": ["Evaluate the effectiveness...", "Justify your answer...", "Assess..."],
        "question_types": ["essay", "multiple_choice", "short_answer"],
    },
    "create": {
        "verbs": _CANONICAL_VERBS["create"],
        "patterns": ["Design a...", "Develop a plan for...", "Create..."],
        "question_types": ["essay", "short_answer"],
    },
}


@dataclass
class QuestionData:
    """A generated assessment question."""
    question_id: str
    question_type: str
    stem: str
    bloom_level: str
    objective_id: str
    choices: List[Dict[str, Any]] = field(default_factory=list)
    correct_answer: Optional[str] = None
    points: float = 1.0
    feedback: Optional[str] = None
    source_chunks: List[str] = field(default_factory=list)
    generation_rationale: Optional[str] = None
    # Observed Bloom level as classified by the BERT ensemble + boolean
    # alignment signal (observed == declared). Round-tripped unconditionally
    # through to_dict() — None is an acceptable shape for a not-yet-classified
    # question.
    observed_bloom_level: Optional[str] = None
    bloom_alignment: Optional[bool] = None
    # Observable cognitive task verb (classify / compute / debug / critique /
    # ...), orthogonal to bloom_level. Populated only when
    # TRAINFORGE_COGNITIVE_TASK_TYPE is on AND the question stem carries a
    # canonical task verb; otherwise stays None and is OMITTED from to_dict()
    # so legacy output is byte-identical.
    cognitive_task_type: Optional[str] = None
    # RESPONSE CARDINALITY + pedagogy discriminator. All three are additive +
    # OPTIONAL: a legacy single-answer item leaves them at their default and
    # to_dict() OMITS them, so the pre-feature serialized shape is
    # byte-identical.
    #
    # ``correct_answers`` carries the plural keys of a MULTIPLE-RESPONSE
    # (select-all) item — the six CC QTI 1.2 primitives include
    # cc.multiple_response, so this is portable. Single-answer items keep the
    # scalar ``correct_answer`` and leave this empty; when populated the keys
    # ALSO carry ``is_correct=True`` in ``choices`` so the MR item is
    # self-describing without this field.
    correct_answers: List[str] = field(default_factory=list)
    # ``item_subtype`` names the PEDAGOGY simulated within a portable QTI
    # primitive (mc_single, mc_multiple_response, tf, fib_numeric,
    # error_analysis, matching_mc, ordering_mc, two_tier_answer,
    # two_tier_reason). ``question_type`` stays one of the CC-portable
    # primitives (multiple_choice / multiple_response / true_false /
    # fill_in_blank); the subtype is metadata only. None → omitted.
    item_subtype: Optional[str] = None
    # ``linked_item_id`` binds the two tiers of a two-tier answer+reason item
    # (Treagust misconception diagnosis) — the answer tier points at its
    # reason tier and vice versa. None → omitted.
    linked_item_id: Optional[str] = None
    # ``rubric`` carries the PER-ITEM analytic rubric of a written-response item
    # (``short_answer`` / ``extended_response``), grounded by construction: a
    # ``{"criteria": [...], "deductions": [...]}`` dict where every criterion +
    # deduction cites the source chunk id it was derived from. Additive +
    # OPTIONAL: a non-written item leaves it None and to_dict() OMITS it, so the
    # pre-feature serialized shape stays byte-identical. Shape:
    #   {"criteria": [{"criterion": str, "cites": [chunk_id, ...],
    #                  "levels": [{"score": int, "descriptor": str}, ...]}, ...],
    #    "deductions": [{"error": str, "note": str, "points": float,
    #                    "cites": [chunk_id, ...]}, ...]}
    rubric: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "stem": self.stem,
            "bloom_level": self.bloom_level,
            "objective_id": self.objective_id,
            "choices": self.choices,
            "correct_answer": self.correct_answer,
            "points": self.points,
            "feedback": self.feedback,
            "source_chunks": self.source_chunks,
            "generation_rationale": self.generation_rationale,
            "observed_bloom_level": self.observed_bloom_level,
            "bloom_alignment": self.bloom_alignment,
        }
        # Additive + optional: omitted when unset so the flag-off serialized
        # shape is byte-identical to the pre-feature output.
        if self.cognitive_task_type is not None:
            d["cognitive_task_type"] = self.cognitive_task_type
        if self.correct_answers:
            d["correct_answers"] = self.correct_answers
        if self.item_subtype is not None:
            d["item_subtype"] = self.item_subtype
        if self.linked_item_id is not None:
            d["linked_item_id"] = self.linked_item_id
        if self.rubric is not None:
            d["rubric"] = self.rubric
        return d


@dataclass
class SkippedItem:
    """A question generation attempt that was skipped (no placeholder emitted).

    When source_chunks is missing or content extraction yields nothing usable
    for the requested question type, the generator emits a SkippedItem rather
    than a deterministic-template placeholder — placeholder strings shipped
    into the training corpus poison it. Downstream consumers (Trainforge
    synthesis, training data export) filter these out.
    """
    question_id: str
    question_type: str
    bloom_level: str
    objective_id: str
    reason: str  # "no_source_chunks" | "no_extractable_content"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "bloom_level": self.bloom_level,
            "objective_id": self.objective_id,
            "reason": self.reason,
        }


@dataclass
class AssessmentData:
    """A complete assessment with multiple questions."""
    assessment_id: str
    title: str
    course_code: str
    questions: List[QuestionData] = field(default_factory=list)
    objectives_targeted: List[str] = field(default_factory=list)
    bloom_levels: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "generated"
    skipped_items: List[SkippedItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Canonical source_coverage block for the assessment-items arrow.
        # items_attempted = questions + skipped_items (every slot the
        # generator tried to fill); items_emitted = len(questions). Drop
        # reasons pull from the SkippedItem.reason field.
        from lib.governance.source_coverage import build_source_coverage
        _drop_histogram: Dict[str, int] = {}
        for _s in self.skipped_items:
            _key = (_s.reason or "unknown").strip() or "unknown"
            _drop_histogram[_key] = _drop_histogram.get(_key, 0) + 1
        _items_attempted = len(self.questions) + len(self.skipped_items)
        source_coverage_block = build_source_coverage(
            consumed_count=_items_attempted,
            emitted_count=len(self.questions),
            drop_reasons=_drop_histogram,
            dropped_count=len(self.skipped_items),
            label=f"assessment_data:{self.assessment_id}",
        )
        return {
            "assessment_id": self.assessment_id,
            "title": self.title,
            "course_code": self.course_code,
            "questions": [q.to_dict() for q in self.questions],
            "objectives_targeted": self.objectives_targeted,
            "bloom_levels": self.bloom_levels,
            "created_at": self.created_at,
            "status": self.status,
            "question_count": len(self.questions),
            "total_points": sum(q.points for q in self.questions),
            "skipped_items_count": len(self.skipped_items),
            "skipped_items_summary": [s.to_dict() for s in self.skipped_items[:3]],
            "source_coverage": source_coverage_block,
        }


class AssessmentGenerator:
    """
    Generates assessments from course content with decision capture.

    Supports:
    - Multiple question types (MCQ, T/F, Fill-blank, Essay)
    - Bloom's taxonomy targeting
    - Learning objective alignment
    - Decision capture for training

    Usage:
        generator = AssessmentGenerator(capture=capture)
        assessment = generator.generate(
            course_code="INT_101",
            objective_ids=["LO-001", "LO-002"],
            bloom_levels=["understand", "apply"],
            question_count=10
        )
    """

    def __init__(
        self,
        capture: Optional["DecisionCapture"] = None,
        check_leaks: bool = True,
        rag: Optional[Any] = None,
        *,
        assessment_provider: Optional[Any] = None,
        groundedness_scorer: Optional[Callable[..., Any]] = None,
        bloom_zero_shot_fn: Optional[Callable[[str], Optional[str]]] = None,
        teacher_roster: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the assessment generator.

        Args:
            capture: Optional DecisionCapture for logging generation decisions
            check_leaks: If True, run leak checker on generated questions
            rag: Optional TrainforgeRAG instance for self-serving retrieval.
                 If provided and source_chunks is None during generation,
                 chunks will be retrieved automatically.
            assessment_provider: Optional pre-built ``AssessmentGeneratorProvider``
                for the apply-arm. Test seam — when None and the apply-arm
                is enabled, one is constructed from the environment.
            groundedness_scorer: Optional ``score_groundedness``-shaped callable
                ``(answer_text, cited_passages, *, split_claims=...) -> report``
                for the apply-arm verify chain. Test seam — when None the
                production ``lib.retrieval.groundedness.score_groundedness`` is
                lazy-imported at call time.
            bloom_zero_shot_fn: Optional zero-shot Bloom voter
                ``(text) -> Optional[level]`` for the apply-arm trivote. Test
                seam / GPU-gated — when None the trivote runs on the deterministic
                asserted + verb voters only (no model load).
            teacher_roster: Optional endpoint-registry-shaped license roster
                consulted by the apply-arm provider license guard.
        """
        self.capture = capture
        # Apply-arm injection seams (test-overridable; production defaults
        # lazy-resolve). None until first use.
        self._assessment_provider = assessment_provider
        self._groundedness_scorer = groundedness_scorer
        self._bloom_zero_shot_fn = bloom_zero_shot_fn
        self._teacher_roster = teacher_roster
        self.check_leaks = check_leaks and LEAK_CHECKER_AVAILABLE
        self._leak_checker = LeakChecker(strict_mode=False) if self.check_leaks else None
        # Diversity rotation. Always picking terms[0] / statements[0] collapses
        # a whole run onto ONE term (one unique stem, the same distractor on
        # every item), so rotate through distinct extracted targets before any
        # reuse.
        self._used_terms: set = set()
        self._used_statements: set = set()
        # Relationship / procedure / example analogs. The W10 phase hands the
        # SAME generator instance + the SAME corpus-wide chunk pool to every
        # per-TO quiz, so always picking relationships[0] / procedures[0] /
        # examples[0] collapses every quiz onto ONE stem (the book-1 canary's
        # 325-item exact-duplicate group). Rotation state spans the whole run.
        self._used_relationships: set = set()
        self._used_procedures: set = set()
        self._used_examples: set = set()
        # Per-Bloom-level question_type rotation cursor. Always firing
        # available_types[0] collapses every remember-level item onto
        # multiple_choice and leaves the essay / short_answer paths dead;
        # rotate across the level's available_types so a multi-item run spans
        # formats.
        self._type_rotation: Dict[str, int] = {}
        # Deterministic diversified-tier telemetry handle; set by
        # generate_diversified(), read by the builders when present.
        self._stats: Optional[GenerationStats] = None
        # Per-run distractor rotation cursor shared by the diversified builders.
        self._div_offset = 0
        self._content_extractor = ContentExtractor()
        self.rag = rag


    def _rotate_term(self, terms):
        """Pick the first extracted term not yet used this run; reset the
        tracker only when every distinct term has been consumed (diversity
        rotation — never collapses a run onto terms[0])."""
        for t in terms:
            key = str(getattr(t, "term", t)).strip().lower()
            if key and key not in self._used_terms:
                self._used_terms.add(key)
                return t
        self._used_terms.clear()
        t = terms[0]
        self._used_terms.add(str(getattr(t, "term", t)).strip().lower())
        return t

    def _rotate_unique(self, items, used: set, key_of):
        """Generic diversity rotation: first item whose key is unused this
        run; reset the tracker only when every distinct key is consumed
        (mirrors :meth:`_rotate_term` — never collapses a run onto
        ``items[0]``)."""
        for it in items:
            key = key_of(it)
            if key and key not in used:
                used.add(key)
                return it
        used.clear()
        it = items[0]
        used.add(key_of(it))
        return it

    def _rotate_relationship(self, relationships):
        """Relationship analog of :meth:`_rotate_term` — keyed on the
        (concept_a, concept_b) pair so distinct extracted relationships are
        consumed before any reuse."""
        return self._rotate_unique(
            relationships,
            self._used_relationships,
            lambda r: (
                str(getattr(r, "concept_a", "")).strip().lower(),
                str(getattr(r, "concept_b", "")).strip().lower(),
            ),
        )

    def _rotate_procedure(self, procedures):
        """Procedure analog of :meth:`_rotate_term` — keyed on title plus the
        first step so same-titled ("Process") procedures still rotate."""
        return self._rotate_unique(
            procedures,
            self._used_procedures,
            lambda p: (
                str(getattr(p, "title", "")).strip().lower(),
                str((getattr(p, "steps", None) or [""])[0])[:80].strip().lower(),
            ),
        )

    def _rotate_example(self, examples):
        """Example analog of :meth:`_rotate_term`."""
        return self._rotate_unique(
            examples,
            self._used_examples,
            lambda e: str(getattr(e, "description", ""))[:80].strip().lower(),
        )

    def _rotate_statement(self, statements):
        """Statement analog of :meth:`_rotate_term`."""
        for st in statements:
            key = str(getattr(st, "text", st))[:80].strip().lower()
            if key and key not in self._used_statements:
                self._used_statements.add(key)
                return st
        self._used_statements.clear()
        st = statements[0]
        self._used_statements.add(str(getattr(st, "text", st))[:80].strip().lower())
        return st

    def _apparatus_guard(self, result):
        """Chokepoint guard against apparatus text in answers/choices.

        Apparatus text ('Solution: r Check: ...', 'Show answer ...') reaches
        correct answers through three separate harvest paths (factual
        statements, term definitions, chunk misconception corrections).
        Filtering each path individually leaves gaps, so reject any assembled
        question whose answer/choice text is apparatus-flavored — converted to
        a SkippedItem so the fill loop draws a replacement, never a
        placeholder.
        """
        from Trainforge.generators.content_extractor import _is_apparatus_text
        if not isinstance(result, QuestionData):
            return result
        texts = [str(result.correct_answer or "")]
        texts += [str(c.get("text") or "") for c in (result.choices or [])
                  if isinstance(c, dict)]
        import re as _re
        # stem-bleed shape: a mid-text '?' followed by more prose means a
        # question fragment fused into the answer ("...= 7? Substitute 18...")
        _bleed = _re.compile(r"\?\s+\S")
        if any(_is_apparatus_text(t) or _bleed.search(t)
               for t in texts if t.strip()):
            return SkippedItem(
                question_id=result.question_id,
                question_type=result.question_type,
                bloom_level=result.bloom_level,
                objective_id=result.objective_id,
                reason="apparatus_contaminated_content",
            )
        return result

    def generate(
        self,
        course_code: str,
        objective_ids: List[str],
        bloom_levels: List[str],
        question_count: int = 10,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
        ensure_objective_coverage: bool = False,
        chunks_by_objective: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> AssessmentData:
        """
        Generate an assessment for the given objectives.

        Args:
            course_code: Course identifier
            objective_ids: Learning objectives to assess
            bloom_levels: Target Bloom's levels
            question_count: Number of questions to generate
            source_chunks: Optional content chunks from RAG retrieval
            ensure_objective_coverage: When True, swap the legacy
                objective-major x bloom-minor nested distribution (which
                exhausts ``question_count`` on the first 1-2 objectives)
                for a coverage-first pass — exactly ONE generation attempt
                per objective in ``objective_ids`` (bloom level rotating by
                position) BEFORE any count-driven fill, so every TO/CO gets
                an item even when ``question_count < len(objective_ids)``.
                Default False → byte-identical legacy distribution.
            chunks_by_objective: Optional ``{objective_id: [chunk, ...]}``
                scoping map consumed ONLY by the coverage-first pass: an
                objective present in the map grounds its attempt in its OWN
                cited chunks first, falling back to the full
                ``source_chunks`` pool on a SkippedItem (coverage beats
                scoping; the fallback pool is still real course chunks —
                anti-fabrication preserved).

        Returns:
            AssessmentData with generated questions
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        assessment_id = f"ASM-{course_code}-{session_id}"

        # Log generation decision with substantive rationale
        if self.capture:
            # Build pedagogical rationale
            level_distribution = ", ".join(bloom_levels)
            rationale = (
                f"Covering {len(objective_ids)} objectives ensures "
                f"learners demonstrate mastery across the full scope. "
                f"Bloom's levels [{level_distribution}] assess both "
                f"foundational and higher-order thinking. "
                f"{question_count} questions balance sampling density "
                f"with learner time constraints."
            )
            self.capture.log_decision(
                decision_type="assessment_planning",
                decision=(
                    f"Planning assessment with {question_count} "
                    f"questions covering {len(objective_ids)} objectives"
                ),
                rationale=rationale,
                alternatives_considered=[
                    {
                        "option": "fewer_questions",
                        "rejected_because": "Insufficient sampling",
                    },
                    {
                        "option": "more_questions",
                        "rejected_because": "Exceeds optimal length",
                    },
                ],
            )

        # Distribute questions across objectives and levels
        questions: List[QuestionData] = []
        skipped_items: List[SkippedItem] = []

        if ensure_objective_coverage:
            # Coverage-first distribution: objective-major single pass — one
            # attempt per objective with the bloom level rotating by
            # position, NEVER capped by question_count (coverage wins; the
            # caller sizes question_count >= len(objective_ids)). A skipped
            # attempt that used an objective-scoped chunk subset retries
            # ONCE against the full pool at the safest (first) bloom level
            # so a thin CO citation set doesn't cost the CO its item.
            covered_objectives: set = set()
            for _obj_idx, obj_id in enumerate(objective_ids):
                bloom_level = bloom_levels[_obj_idx % len(bloom_levels)]
                scoped_chunks = (chunks_by_objective or {}).get(obj_id)
                result = self._apparatus_guard(self._generate_question(
                    objective_id=obj_id,
                    bloom_level=bloom_level,
                    source_chunks=scoped_chunks or source_chunks,
                ))
                if isinstance(result, SkippedItem) and (
                    scoped_chunks or bloom_level != bloom_levels[0]
                ):
                    # Fallback attempt: full pool + first bloom level.
                    result = self._apparatus_guard(self._generate_question(
                        objective_id=obj_id,
                        bloom_level=bloom_levels[0],
                        source_chunks=source_chunks,
                    ))
                if isinstance(result, QuestionData):
                    questions.append(result)
                    covered_objectives.add(obj_id)
                elif isinstance(result, SkippedItem):
                    skipped_items.append(result)

            # The first coverage pass is deliberately strict about rejecting
            # contaminated or otherwise unusable source material.  A rejected
            # item must not, however, be replaced by a duplicate item for an
            # already-covered objective in the generic count-fill loop below.
            # That previously made the requested question *count* look
            # complete while leaving objectives unassessed.  Retry only the
            # uncovered objectives, against the full source pool, rotating
            # Bloom levels so the extractor advances to different grounded
            # material on each attempt.
            _coverage_retry_limit = max(4, len(bloom_levels) * 2)
            for obj_id in objective_ids:
                if obj_id in covered_objectives:
                    continue
                for _retry_idx in range(_coverage_retry_limit):
                    bloom_level = bloom_levels[
                        _retry_idx % len(bloom_levels)
                    ]
                    # The deterministic builders mine from the front of their
                    # chunk list. Retrying an unchanged pool can therefore
                    # reproduce the same apparatus-contaminated candidate on
                    # every attempt. Rotate the real source pool so each retry
                    # examines a different grounded starting point without
                    # fabricating or weakening the apparatus guard.
                    retry_chunks = source_chunks
                    if source_chunks:
                        offset = _retry_idx % len(source_chunks)
                        retry_chunks = [
                            *source_chunks[offset:],
                            *source_chunks[:offset],
                        ]
                    result = self._apparatus_guard(self._generate_question(
                        objective_id=obj_id,
                        bloom_level=bloom_level,
                        source_chunks=retry_chunks,
                    ))
                    if isinstance(result, QuestionData):
                        questions.append(result)
                        covered_objectives.add(obj_id)
                        break
                    if isinstance(result, SkippedItem):
                        skipped_items.append(result)
            if self.capture:
                uncovered = [
                    o for o in objective_ids if o not in covered_objectives
                ]
                self.capture.log_decision(
                    decision_type="assessment_planning",
                    decision=(
                        f"Coverage-first distribution produced "
                        f"{len(questions)} items over "
                        f"{len(covered_objectives)}/{len(objective_ids)} "
                        f"objectives"
                    ),
                    rationale=(
                        f"ensure_objective_coverage=True: one attempt per "
                        f"objective (bloom rotating over "
                        f"[{', '.join(bloom_levels)}]) so every TO/CO in "
                        f"{objective_ids[:3]}... yields an assessment item "
                        f"for the strict archival coverage gate; "
                        f"uncovered_after_pass={uncovered[:5]} "
                        f"(scoped-chunk map entries: "
                        f"{len(chunks_by_objective or {})})"
                    ),
                )
        else:
            questions_per_combo = max(1, question_count // (len(objective_ids) * len(bloom_levels)))

            for obj_id in objective_ids:
                for bloom_level in bloom_levels:
                    for _ in range(questions_per_combo):
                        if len(questions) >= question_count:
                            break

                        result = self._apparatus_guard(self._generate_question(
                            objective_id=obj_id,
                            bloom_level=bloom_level,
                            source_chunks=source_chunks,
                        ))
                        if isinstance(result, QuestionData):
                            questions.append(result)
                        elif isinstance(result, SkippedItem):
                            skipped_items.append(result)

                    if len(questions) >= question_count:
                        break
                if len(questions) >= question_count:
                    break

        # Fill remaining with cycling through objectives. Bound the loop so
        # an all-skip path (e.g. empty source_chunks) cannot spin forever.
        idx = 0
        max_attempts = question_count * max(1, len(objective_ids) * len(bloom_levels))
        attempt = 0
        while len(questions) < question_count and attempt < max_attempts:
            obj_id = objective_ids[idx % len(objective_ids)]
            bloom_level = bloom_levels[idx % len(bloom_levels)]

            result = self._apparatus_guard(self._generate_question(
                objective_id=obj_id,
                bloom_level=bloom_level,
                source_chunks=source_chunks,
            ))
            if isinstance(result, QuestionData):
                questions.append(result)
            elif isinstance(result, SkippedItem):
                skipped_items.append(result)
            idx += 1
            attempt += 1

        # Run leak checker on generated questions
        if self._leak_checker and questions:
            leak_questions = []
            for q in questions:
                leak_q = {"id": q.question_id}
                if q.correct_answer:
                    leak_q["correct_answer"] = q.correct_answer
                elif q.choices:
                    correct = [
                        c["text"] for c in q.choices if c.get("is_correct")
                    ]
                    if correct:
                        leak_q["correct_answers"] = correct
                leak_questions.append(leak_q)

            self._leak_checker.register_assessment(assessment_id, leak_questions)

            # Check each question's stem for answer leaks
            leaked_ids = set()
            for q in questions:
                result = self._leak_checker.check_prompt(
                    q.stem, assessment_id=assessment_id, question_id=q.question_id
                )
                if not result.passed:
                    leaked_ids.add(q.question_id)
                    logger.warning(
                        "Leak detected in %s: %d leaks found",
                        q.question_id, result.leak_count,
                    )

            # Remove leaked questions
            if leaked_ids:
                original_count = len(questions)
                questions = [q for q in questions if q.question_id not in leaked_ids]
                logger.warning(
                    "Removed %d/%d questions with answer leaks",
                    original_count - len(questions), original_count,
                )

                if self.capture:
                    self.capture.log_decision(
                        decision_type="leak_check_filtering",
                        decision=f"Removed {len(leaked_ids)} questions with answer leaks",
                        rationale=(
                            f"Leak checker detected answer content in question stems "
                            f"for questions {leaked_ids}. Removing to prevent training "
                            f"data contamination and maintain RAG corpus integrity."
                        ),
                    )

                # Leak filtering runs after the coverage-first pass, so it can
                # invalidate that pass by removing the only item for an
                # objective. Repair only those newly-uncovered objectives and
                # subject every replacement to the same leak check. If the
                # deterministic generator cannot produce a safe replacement,
                # fail loudly instead of returning an assessment that violates
                # the caller's explicit coverage contract.
                if ensure_objective_coverage:
                    retained_objectives = {
                        q.objective_id for q in questions
                    }
                    missing_objectives = [
                        oid for oid in objective_ids
                        if oid not in retained_objectives
                    ]
                    repair_attempt_limit = max(4, len(bloom_levels) * 2)

                    def _leak_registration_payload(
                        candidates: List[QuestionData],
                    ) -> List[Dict[str, Any]]:
                        payload: List[Dict[str, Any]] = []
                        for candidate in candidates:
                            item: Dict[str, Any] = {
                                "id": candidate.question_id,
                            }
                            if candidate.correct_answer:
                                item["correct_answer"] = (
                                    candidate.correct_answer
                                )
                            elif candidate.choices:
                                correct = [
                                    choice["text"]
                                    for choice in candidate.choices
                                    if choice.get("is_correct")
                                ]
                                if correct:
                                    item["correct_answers"] = correct
                            payload.append(item)
                        return payload

                    for objective_id in missing_objectives:
                        for repair_idx in range(repair_attempt_limit):
                            bloom_level = bloom_levels[
                                repair_idx % len(bloom_levels)
                            ]
                            replacement = self._apparatus_guard(
                                self._generate_question(
                                    objective_id=objective_id,
                                    bloom_level=bloom_level,
                                    source_chunks=source_chunks,
                                )
                            )
                            if not isinstance(replacement, QuestionData):
                                if isinstance(replacement, SkippedItem):
                                    skipped_items.append(replacement)
                                continue

                            self._leak_checker.register_assessment(
                                assessment_id,
                                _leak_registration_payload(
                                    [*questions, replacement]
                                ),
                            )
                            leak_result = self._leak_checker.check_prompt(
                                replacement.stem,
                                assessment_id=assessment_id,
                                question_id=replacement.question_id,
                            )
                            if leak_result.passed:
                                questions.append(replacement)
                                retained_objectives.add(objective_id)
                                break
                            logger.warning(
                                "Rejected leak-repair item %s for %s: "
                                "%d leak(s) found",
                                replacement.question_id,
                                objective_id,
                                leak_result.leak_count,
                            )

                    still_missing = [
                        oid for oid in objective_ids
                        if oid not in retained_objectives
                    ]
                    if still_missing:
                        raise RuntimeError(
                            "Assessment coverage could not be restored after "
                            "leak filtering for objective_ids="
                            f"{still_missing}"
                        )

        assessment = AssessmentData(
            assessment_id=assessment_id,
            title=f"Assessment: {course_code}",
            course_code=course_code,
            questions=questions,
            objectives_targeted=objective_ids,
            bloom_levels=bloom_levels,
            skipped_items=skipped_items,
        )

        # Surface skipped items in the rationale stream so an operator can see
        # why questions are missing without grepping decision JSONL files.
        if self.capture and skipped_items:
            preview = "; ".join(
                f"{s.question_id}={s.reason}@{s.objective_id}/{s.bloom_level}"
                for s in skipped_items[:3]
            )
            self.capture.log_decision(
                decision_type="assessment_template_skip",
                decision=(
                    f"Skipped {len(skipped_items)} question slots "
                    f"(no source content; would have emitted placeholders)"
                ),
                rationale=(
                    f"Skipped {len(skipped_items)} of "
                    f"{len(skipped_items) + len(questions)} attempted "
                    f"question slots because the deterministic template "
                    f"fallback path was killed in Worker W1; emitting "
                    f"placeholder strings ('Correct answer based on "
                    f"content', etc.) into training data poisons the "
                    f"corpus. First slots: {preview}"
                ),
            )

        # Log final assessment decision with substantive rationale
        if self.capture:
            type_counts = {}
            bloom_counts = {}
            for q in questions:
                type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
                bloom_counts[q.bloom_level] = bloom_counts.get(q.bloom_level, 0) + 1

            # Build comprehensive rationale
            type_summary = ", ".join(f"{count} {qtype}" for qtype, count in type_counts.items())
            bloom_summary = ", ".join(f"{count} at {level}" for level, count in bloom_counts.items())

            rationale = (
                f"Finalized assessment with balanced question distribution: {type_summary}. "
                f"Cognitive level coverage: {bloom_summary}. "
                f"This distribution ensures comprehensive assessment of learning objectives "
                f"while maintaining appropriate variety in question formats to reduce test-taking strategy effects. "
                f"The mix of question types accommodates diverse learner strengths and provides "
                f"multiple opportunities to demonstrate competency."
            )

            self.capture.log_decision(
                decision_type="assessment_generation",
                decision=f"Completed assessment {assessment_id} with {len(questions)} questions across {len(objective_ids)} objectives",
                rationale=rationale,
                alternatives_considered=[
                    {"option": "all_mcq", "rejected_because": "Homogeneous formats allow test-taking strategies and fail to assess constructed response skills"},
                    {"option": "all_essay", "rejected_because": "Excessive grading burden and potential learner fatigue without assessing factual recall"},
                ],
            )

        return assessment

    def _generate_question(
        self,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Generate a single question for the given objective and level.

        If source_chunks is None and self.rag is available, retrieves
        chunks automatically using the fallback chain.

        Args:
            objective_id: Learning objective ID
            bloom_level: Target Bloom's level
            source_chunks: Optional content chunks

        Returns:
            QuestionData on success; SkippedItem when source_chunks is
            empty / unavailable — never a templated placeholder.
        """
        # Self-serve retrieval if no chunks provided
        if source_chunks is None and self.rag is not None:
            try:
                chunks, _metrics = self.rag.retrieve_with_fallback(
                    objective_text=objective_id,
                    bloom_level=bloom_level,
                )
                source_chunks = [c.to_dict() for c in chunks]
                if self.capture and source_chunks:
                    self.capture.log_decision(
                        decision_type="chunk_retrieval",
                        decision=f"Auto-retrieved {len(source_chunks)} chunks for {objective_id}",
                        rationale=(
                            f"No source chunks provided; used RAG fallback chain "
                            f"to retrieve {len(source_chunks)} chunks for objective "
                            f"'{objective_id}' at Bloom level '{bloom_level}'"
                        ),
                    )
            except Exception as e:
                logger.warning("RAG retrieval failed for %s: %s", objective_id, e)

        question_id = f"Q-{str(uuid.uuid4())[:8]}"

        # Select question type based on Bloom's level. Rotate across the
        # level's available_types (instead of always [0]) so a multi-item run
        # spans formats and the essay / short_answer paths are reachable.
        level_config = BLOOM_LEVELS.get(bloom_level, BLOOM_LEVELS["understand"])
        available_types = level_config["question_types"]
        _cursor = self._type_rotation.get(bloom_level, 0)
        question_type = available_types[_cursor % len(available_types)]
        self._type_rotation[bloom_level] = _cursor + 1

        # Build pedagogical rationale for question type selection
        type_rationales = {
            "multiple_choice": (
                "because MCQ format allows efficient assessment of recognition and discrimination skills, "
                "and plausible distractors can target common misconceptions to provide diagnostic feedback"
            ),
            "true_false": (
                "because T/F format efficiently assesses factual recall and simple comprehension, "
                "though limited to binary distinctions which suits lower cognitive levels"
            ),
            "fill_in_blank": (
                "because fill-in-blank requires active recall rather than recognition, "
                "testing deeper retention of key terminology and concepts"
            ),
            "short_answer": (
                "because short answer requires learners to construct responses, "
                "demonstrating understanding through explanation rather than selection"
            ),
            "essay": (
                "because essay format allows learners to synthesize, evaluate, and create, "
                "demonstrating higher-order thinking that cannot be assessed through closed-format items"
            ),
        }

        # Log question type decision with substantive rationale
        if self.capture:
            base_rationale = type_rationales.get(question_type, "because it aligns with the target cognitive level")
            alternatives = [
                {"option": alt_type, "rejected_because": f"Less appropriate for {bloom_level} level cognitive demands"}
                for alt_type in available_types[1:3] if alt_type != question_type
            ]

            self.capture.log_decision(
                decision_type="question_type_selection",
                decision=f"Selected {question_type} for Bloom level '{bloom_level}' targeting objective {objective_id}",
                rationale=f"Chose {question_type} format {base_rationale}. This format is pedagogically appropriate for the '{bloom_level}' cognitive level.",
                alternatives_considered=alternatives if alternatives else None,
            )

        # Generate question based on type
        if question_type == "multiple_choice":
            result = self._generate_multiple_choice(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )
        elif question_type == "true_false":
            result = self._generate_true_false(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )
        elif question_type == "fill_in_blank":
            result = self._generate_fill_in_blank(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )
        elif question_type == "essay":
            result = self._generate_essay(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )
        else:
            result = self._generate_short_answer(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )

        # Skip-emit path. Log a structured skip event with interpolated
        # rationale (objective + bloom + reason + question type) so post-hoc
        # audits can replay why each slot was dropped.
        if isinstance(result, SkippedItem):
            if self.capture:
                self.capture.log_decision(
                    decision_type="assessment_template_skip",
                    decision=(
                        f"Skipped {question_type} for {objective_id} "
                        f"({bloom_level}) — reason={result.reason}"
                    ),
                    rationale=(
                        f"Refusing to emit placeholder for "
                        f"question_id={question_id} "
                        f"(type={question_type}, objective={objective_id}, "
                        f"bloom={bloom_level}, reason={result.reason}). "
                        f"Worker W1 killed the deterministic template "
                        f"fallback that previously emitted strings like "
                        f"'Correct answer based on content' into the "
                        f"training corpus. Caller should re-dispatch with "
                        f"valid source_chunks or accept the gap."
                    ),
                    ml_features=MLFeatures(
                        bloom_levels=[bloom_level],
                        component_types=[question_type],
                    ),
                )
            return result

        # Tag the emitted question with the observable cognitive task verb
        # detected in its stem. Gated on TRAINFORGE_COGNITIVE_TASK_TYPE:
        # default unset -> the field stays None and to_dict() omits it
        # (byte-identical legacy output). When on AND a canonical task verb is
        # present, stamp the field and emit one decision-capture event.
        if _cognitive_task_type_enabled():
            task_type: Optional[str] = None
            try:
                from lib.ontology.cognitive_task import detect_cognitive_task_type
                task_type = detect_cognitive_task_type(result.stem)
            except Exception as exc:  # noqa: BLE001 - best-effort, never break emit
                logger.debug("cognitive_task_type detection raised: %s", exc)
            if task_type:
                result.cognitive_task_type = task_type
                if self.capture:
                    stem_prefix = result.stem[:80].replace("\n", " ").strip()
                    try:
                        self.capture.log_decision(
                            decision_type="cognitive_task_type_detection",
                            decision=f"Assigned cognitive_task_type={task_type}",
                            rationale=(
                                f"detect_cognitive_task_type matched the "
                                f"canonical verb {task_type!r} as a whole word "
                                f"in the {question_type} question stem "
                                f"(prefix={stem_prefix!r}) for question "
                                f"{question_id} (objective={objective_id}, "
                                f"bloom={bloom_level!r}). Adds an axis "
                                "orthogonal to bloom_level so downstream "
                                "consumers can audit per-task diversity."
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "DecisionCapture.log_decision raised on "
                            "cognitive_task_type_detection: %s",
                            exc,
                        )

        # Log content grounding decision (real question emit path).
        if self.capture:
            self.capture.log_decision(
                decision_type="question_generation",
                decision=(
                    f"Generated {question_type} question {question_id} "
                    f"(content-grounded)"
                ),
                rationale=result.generation_rationale or "No rationale available",
            )

        return result

    def _generate_multiple_choice(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """Generate a multiple choice question from content.

        Returns QuestionData on success, or SkippedItem when source_chunks
        is missing / empty / yields nothing extractable. There is deliberately
        no template fallback — a placeholder answer string would pass the
        structural validators and poison the training corpus.
        """
        # Try content-grounded generation
        if source_chunks:
            terms = self._content_extractor.extract_key_terms(source_chunks)
            statements = self._content_extractor.extract_factual_statements(source_chunks)

            # Collect misconceptions from chunk metadata for distractor generation
            misconception_texts = []
            for chunk in source_chunks:
                for mc in chunk.get("misconceptions", []):
                    if isinstance(mc, dict) and mc.get("misconception"):
                        misconception_texts.append(mc["misconception"])

            if terms:
                # Use a key term: ask for its definition
                target = self._rotate_term(terms)
                # Bloom-verb-leading phrasing. "Which of the following best
                # describes X?" carries no canonical Bloom verb form
                # ("describes" is conjugated; lib/ontology/bloom.py matches
                # whole-word canonical verbs), so every term MCQ phrased that
                # way fires assessment_quality VERB_LESS_STEM. "Identify"
                # registers as remember-level — same question, detector-visible
                # verb.
                stem = (
                    f"<p>Identify the statement that best describes "
                    f"<em>{target.term}</em>.</p>"
                )

                correct_text = target.definition
                # Trim to reasonable length
                if len(correct_text) > 200:
                    correct_text = correct_text[:197] + "..."

                # Build distractors: prefer misconceptions, then other terms
                distractors = []

                # Strategy 1: Use actual misconceptions as distractors
                for mc_text in misconception_texts:
                    if len(distractors) >= 3:
                        break
                    if len(mc_text) > 200:
                        mc_text = mc_text[:197] + "..."
                    distractors.append(mc_text)

                # Strategy 2: Other terms' definitions.
                # Distractor rotation: a static terms[1:4] slice repeats the
                # same 3 distractors on nearly every question
                # (TEMPLATED_DISTRACTORS). Exclude the target, rotate the pool
                # start per question.
                _pool = [t for t in terms if t.term != target.term]
                self._distractor_offset = getattr(self, "_distractor_offset", 0) + 1
                _n = len(_pool)
                for _k in range(_n):
                    other = _pool[(self._distractor_offset + _k) % _n]
                    if len(distractors) >= 3:
                        break
                    if other.definition != target.definition:
                        d_text = other.definition
                        if len(d_text) > 200:
                            d_text = d_text[:197] + "..."
                        if d_text not in distractors:
                            distractors.append(d_text)

                # Strategy 3: Factual statements.
                # Rotate the statement pool for the same reason Strategy 2
                # rotates its term pool: iterating from statements[0] on every
                # question re-serves the same leading statements as distractors
                # across the whole quiz set (measured: 29 distractor strings
                # reused >=3x on a 313-item set). Reuses the shared
                # ``_distractor_offset`` cursor advanced by Strategy 2 above,
                # and skips duplicates already collected for THIS question.
                _sn = len(statements)
                for _k in range(_sn):
                    if len(distractors) >= 3:
                        break
                    stmt = statements[(self._distractor_offset + _k) % _sn]
                    if stmt.statement.lower() != target.definition.lower():
                        d_text = stmt.statement
                        if len(d_text) > 200:
                            d_text = d_text[:197] + "..."
                        if d_text not in distractors:
                            distractors.append(d_text)

                # No padded-distractor fallback. Padding the choices with a
                # template string ("A concept unrelated to <term> in this
                # context") trivially passes the structural validators (it is
                # a string and differs from the key) and then gets baked into
                # training pairs. Emit a SkippedItem plus a
                # ``distractor_padding_skipped`` event instead, so
                # insufficient-distractor questions surface in the audit log.
                if len(distractors) < 3:
                    if self.capture is not None:
                        try:
                            self.capture.log_decision(
                                decision_type="distractor_padding_skipped",
                                decision=(
                                    f"Skipped MCQ {question_id} (target term "
                                    f"'{target.term}'): only "
                                    f"{len(distractors)} plausible distractors "
                                    f"resolved (need 3); padded-fallback path "
                                    f"eliminated in W2.D."
                                ),
                                rationale=(
                                    f"distractor_padding_skipped on "
                                    f"question_id={question_id!r}, "
                                    f"target_term={target.term!r}, "
                                    f"objective_id={objective_id!r}, "
                                    f"bloom_level={bloom_level!r}, "
                                    f"resolved_distractors={len(distractors)}, "
                                    f"required=3, "
                                    f"reason=padded_distractor_fallback_eliminated."
                                ),
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "DecisionCapture.log_decision raised on "
                                "distractor_padding_skipped: %s",
                                exc,
                            )
                    return SkippedItem(
                        question_id=question_id,
                        question_type="multiple_choice",
                        bloom_level=bloom_level,
                        objective_id=objective_id,
                        reason="padded_distractor_fallback_eliminated",
                    )

                choices = [
                    {"text": f"<p>{correct_text}</p>", "is_correct": True},
                ]
                for d in distractors[:3]:
                    choices.append({"text": f"<p>{d}</p>", "is_correct": False})

                return QuestionData(
                    question_id=question_id,
                    question_type="multiple_choice",
                    stem=stem,
                    bloom_level=bloom_level,
                    objective_id=objective_id,
                    choices=choices,
                    points=2.0,
                    feedback=f"<p>{target.context_sentence}</p>",
                    source_chunks=[target.source_chunk_id],
                    generation_rationale=(
                        f"MCQ grounded in key term '{target.term}' at "
                        f"{bloom_level} level; distractors from related content"
                    ),
                )

            elif statements and len(statements) >= 4:
                # Use a factual statement: ask which is true. Carry the central
                # idea in the stem (Haladyna rule 1) — a bare "Which statement
                # is correct?" is a non-problem. Anchor on the statement's key
                # subject so the stem names what is being assessed.
                correct_stmt = self._rotate_statement(statements)
                _subject = (getattr(correct_stmt, "key_subject", "") or "").strip()
                if _subject:
                    stem = (
                        f"<p>Which of the following statements about "
                        f"<em>{_subject}</em> is accurate?</p>"
                    )
                else:
                    stem = (
                        "<p>Which of the following statements is best supported "
                        "by the source material?</p>"
                    )

                # A regex-negated "false" distractor can be accidentally TRUE
                # (double-negation of an already-negative source, or a no-op
                # negation). Verify each negation is non-trivially false
                # and DROP the ones that aren't, drawing from ALL remaining
                # factual statements so a bad negation is regenerated rather
                # than silently emitted as a true option.
                distractor_texts: List[str] = []
                dropped_negations = 0
                for other in statements[1:]:
                    if len(distractor_texts) >= 3:
                        break
                    negated = self._safe_negate_statement(other.statement)
                    if negated is None:
                        dropped_negations += 1
                        continue
                    distractor_texts.append(negated)

                if dropped_negations and self.capture is not None:
                    try:
                        self.capture.log_decision(
                            decision_type="distractor_generation",
                            decision=(
                                f"Dropped {dropped_negations} unverifiable "
                                f"negated distractor(s) for MCQ {question_id}"
                            ),
                            rationale=(
                                f"negation_rejected on question_id="
                                f"{question_id!r}, objective_id="
                                f"{objective_id!r}, bloom_level={bloom_level!r}: "
                                f"{dropped_negations} regex-negated distractor(s) "
                                f"were not verifiably false (double-negation / "
                                f"vacuous negation) and were dropped; "
                                f"{len(distractor_texts)} verifiably-false "
                                f"distractor(s) kept."
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "DecisionCapture.log_decision raised on "
                            "distractor_generation (negation_rejected): %s",
                            exc,
                        )

                # Need >=2 verifiably-false distractors for a non-degenerate
                # MCQ (1 correct + 2 distractors). Fewer → skip-emit rather
                # than ship a quiz whose "wrong" options may be true.
                if len(distractor_texts) < 2:
                    return SkippedItem(
                        question_id=question_id,
                        question_type="multiple_choice",
                        bloom_level=bloom_level,
                        objective_id=objective_id,
                        reason="negation_unverifiable",
                    )

                choices = [
                    {"text": f"<p>{correct_stmt.statement}</p>", "is_correct": True},
                ]
                for d in distractor_texts:
                    choices.append({"text": f"<p>{d}</p>", "is_correct": False})

                return QuestionData(
                    question_id=question_id,
                    question_type="multiple_choice",
                    stem=stem,
                    bloom_level=bloom_level,
                    objective_id=objective_id,
                    choices=choices,
                    points=2.0,
                    feedback=f"<p>{correct_stmt.statement}</p>",
                    source_chunks=[correct_stmt.source_chunk_id],
                    generation_rationale=(
                        f"MCQ using correct-statement selection at {bloom_level} "
                        f"level; distractors are verifiably-false negated "
                        f"content statements (W7.4 correctness-checked)"
                    ),
                )

        # Skip-emit rather than a templated placeholder: returning a
        # SkippedItem keeps placeholder strings out of the training corpus;
        # the caller filters Skips and surfaces a count.
        reason = (
            "no_source_chunks"
            if not source_chunks
            else "no_extractable_content"
        )
        return SkippedItem(
            question_id=question_id,
            question_type="multiple_choice",
            bloom_level=bloom_level,
            objective_id=objective_id,
            reason=reason,
        )

    def _generate_true_false(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """Generate a true/false question from content.

        Returns QuestionData on success, or SkippedItem when no chunks
        are available / no factual statement can be extracted. There is
        deliberately no ``"Statement about ... content."`` placeholder.
        """
        if source_chunks:
            statements = self._content_extractor.extract_factual_statements(source_chunks)

            if statements:
                stmt = self._rotate_statement(statements)
                # Randomly decide true vs false (use question_id hash for determinism)
                make_false = hash(question_id) % 2 == 0

                # A regex-negated T/F "false" stem can be accidentally TRUE.
                # Verify the negation is non-trivially false; if it isn't,
                # fall back to the TRUE variant rather than ship a "false" item
                # whose stem may be true (never emit an unverifiable false).
                negated: Optional[str] = None
                if make_false:
                    negated = self._safe_negate_statement(stmt.statement)
                    if negated is None:
                        if self.capture is not None:
                            try:
                                self.capture.log_decision(
                                    decision_type="question_generation",
                                    decision=(
                                        f"T/F {question_id}: negation "
                                        f"unverifiable — emitting TRUE variant"
                                    ),
                                    rationale=(
                                        f"negation_rejected on question_id="
                                        f"{question_id!r}, objective_id="
                                        f"{objective_id!r}, subject="
                                        f"{stmt.key_subject!r}: regex negation of "
                                        f"the source statement was not verifiably "
                                        f"false (double-negation / vacuous "
                                        f"negation); fell back to the TRUE T/F "
                                        f"variant so no accidentally-true 'false' "
                                        f"item ships."
                                    ),
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "DecisionCapture.log_decision raised on "
                                    "question_generation (negation_rejected): %s",
                                    exc,
                                )
                        make_false = False

                if make_false:
                    return QuestionData(
                        question_id=question_id,
                        question_type="true_false",
                        stem=f"<p>{negated}</p>",
                        bloom_level=bloom_level,
                        objective_id=objective_id,
                        choices=[
                            {"text": "True", "is_correct": False},
                            {"text": "False", "is_correct": True},
                        ],
                        correct_answer="False",
                        points=1.0,
                        feedback=f"<p>The correct statement is: {stmt.statement}</p>",
                        source_chunks=[stmt.source_chunk_id],
                        generation_rationale=(
                            f"T/F (false) at {bloom_level} level; verifiably-false "
                            f"negated factual statement about '{stmt.key_subject}' "
                            f"(W7.4 correctness-checked)"
                        ),
                    )
                else:
                    return QuestionData(
                        question_id=question_id,
                        question_type="true_false",
                        stem=f"<p>{stmt.statement}</p>",
                        bloom_level=bloom_level,
                        objective_id=objective_id,
                        choices=[
                            {"text": "True", "is_correct": True},
                            {"text": "False", "is_correct": False},
                        ],
                        correct_answer="True",
                        points=1.0,
                        feedback=f"<p>This is correct. {stmt.statement}</p>",
                        source_chunks=[stmt.source_chunk_id],
                        generation_rationale=(
                            f"T/F (true) at {bloom_level} level; factual statement "
                            f"about '{stmt.key_subject}'"
                        ),
                    )

        # Skip-emit rather than a templated placeholder.
        reason = (
            "no_source_chunks"
            if not source_chunks
            else "no_extractable_content"
        )
        return SkippedItem(
            question_id=question_id,
            question_type="true_false",
            bloom_level=bloom_level,
            objective_id=objective_id,
            reason=reason,
        )

    def _generate_fill_in_blank(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """Generate a fill-in-the-blank question from content.

        Returns QuestionData on success, or SkippedItem when no chunks
        are available / no key term can be blanked. There is deliberately no
        ``"The key concept from ... is _______."`` placeholder.
        """
        if source_chunks:
            terms = self._content_extractor.extract_key_terms(source_chunks)

            if terms:
                target = self._rotate_term(terms)
                # Replace the term in the context sentence with a blank
                blanked = re.sub(
                    re.escape(target.term),
                    "_______",
                    target.context_sentence,
                    count=1,
                    flags=re.IGNORECASE,
                )
                # Only use if the blank actually replaced something
                if "_______" in blanked and blanked != target.context_sentence:
                    return QuestionData(
                        question_id=question_id,
                        question_type="fill_in_blank",
                        stem=f"<p>Complete the following: {blanked}</p>",
                        bloom_level=bloom_level,
                        objective_id=objective_id,
                        correct_answer=target.term,
                        points=1.0,
                        feedback=f"<p>The answer is <strong>{target.term}</strong>. {target.context_sentence}</p>",
                        source_chunks=[target.source_chunk_id],
                        generation_rationale=(
                            f"Fill-in-blank at {bloom_level} level; blanked term "
                            f"'{target.term}' from source content"
                        ),
                    )

        # Skip-emit rather than a templated placeholder.
        reason = (
            "no_source_chunks"
            if not source_chunks
            else "no_extractable_content"
        )
        return SkippedItem(
            question_id=question_id,
            question_type="fill_in_blank",
            bloom_level=bloom_level,
            objective_id=objective_id,
            reason=reason,
        )

    def _generate_essay(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """Generate an essay question from content.

        Returns QuestionData on success, or SkippedItem when no chunks
        are available / no relationship or example can be extracted. There is
        deliberately no ``"...concepts from ... and provide examples."``
        placeholder.
        """
        verb = level_config["verbs"][0]

        if source_chunks:
            relationships = self._content_extractor.extract_relationships(source_chunks)
            examples = self._content_extractor.extract_examples(source_chunks)

            if relationships:
                rel = self._rotate_relationship(relationships)
                stem = (
                    f"<p>{verb.capitalize()} the relationship between "
                    f"<em>{rel.concept_a}</em> and <em>{rel.concept_b}</em>. "
                    f"Support your analysis with specific examples from the course material.</p>"
                )
                # Build rubric points from content
                rubric_points = [f"Explains connection between {rel.concept_a} and {rel.concept_b}"]
                if examples:
                    rubric_points.append(f"Uses relevant examples (e.g., {examples[0].description[:80]}...)")
                for other_rel in relationships[1:3]:
                    rubric_points.append(f"Addresses: {other_rel.full_statement[:80]}...")
                rubric_text = "</li><li>".join(rubric_points)
                feedback = f"<p>A strong response should:</p><ul><li>{rubric_text}</li></ul>"

                chunk_ids = list({rel.source_chunk_id})
                if examples:
                    chunk_ids.append(examples[0].source_chunk_id)

                return QuestionData(
                    question_id=question_id,
                    question_type="essay",
                    stem=stem,
                    bloom_level=bloom_level,
                    objective_id=objective_id,
                    points=10.0,
                    feedback=feedback,
                    source_chunks=chunk_ids[:3],
                    generation_rationale=(
                        f"Essay at {bloom_level} level; explores relationship "
                        f"between '{rel.concept_a}' and '{rel.concept_b}'"
                    ),
                )

            elif examples:
                ex = self._rotate_example(examples)
                stem = (
                    f"<p>{verb.capitalize()} the following scenario: "
                    f"<em>{ex.description}</em>. "
                    f"What principles does this illustrate, and how could "
                    f"they be applied in a different context?</p>"
                )
                return QuestionData(
                    question_id=question_id,
                    question_type="essay",
                    stem=stem,
                    bloom_level=bloom_level,
                    objective_id=objective_id,
                    points=10.0,
                    feedback="<p>A complete response should identify the underlying principles illustrated by this example and propose a novel application.</p>",
                    source_chunks=[ex.source_chunk_id],
                    generation_rationale=(
                        f"Essay at {bloom_level} level; based on example from content"
                    ),
                )

        # Skip-emit rather than a templated placeholder.
        reason = (
            "no_source_chunks"
            if not source_chunks
            else "no_extractable_content"
        )
        return SkippedItem(
            question_id=question_id,
            question_type="essay",
            bloom_level=bloom_level,
            objective_id=objective_id,
            reason=reason,
        )

    def _generate_short_answer(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """Generate a short answer question from content.

        Returns QuestionData on success, or SkippedItem when no chunks
        are available / no procedure/relationship/term can be extracted. There
        is deliberately no ``"Briefly ... the key points from ..."``
        placeholder.
        """
        verb = level_config["verbs"][0]

        if source_chunks:
            relationships = self._content_extractor.extract_relationships(source_chunks)
            procedures = self._content_extractor.extract_procedures(source_chunks)
            terms = self._content_extractor.extract_key_terms(source_chunks)

            if procedures:
                proc = self._rotate_procedure(procedures)
                stem = (
                    f"<p>Briefly {verb} the steps involved in "
                    f"<em>{proc.title.lower()}</em>.</p>"
                )
                model_answer = "; ".join(proc.steps[:4])
                return QuestionData(
                    question_id=question_id,
                    question_type="short_answer",
                    stem=stem,
                    bloom_level=bloom_level,
                    objective_id=objective_id,
                    correct_answer=model_answer,
                    points=5.0,
                    feedback=f"<p>Key steps: {model_answer}</p>",
                    source_chunks=[proc.source_chunk_id],
                    generation_rationale=(
                        f"Short answer at {bloom_level} level; asks about procedure "
                        f"'{proc.title}' ({len(proc.steps)} steps)"
                    ),
                )

            elif relationships:
                rel = self._rotate_relationship(relationships)
                stem = (
                    f"<p>Briefly {verb} the relationship between "
                    f"<em>{rel.concept_a}</em> and <em>{rel.concept_b}</em>.</p>"
                )
                return QuestionData(
                    question_id=question_id,
                    question_type="short_answer",
                    stem=stem,
                    bloom_level=bloom_level,
                    objective_id=objective_id,
                    correct_answer=rel.full_statement,
                    points=5.0,
                    feedback=f"<p>{rel.full_statement}</p>",
                    source_chunks=[rel.source_chunk_id],
                    generation_rationale=(
                        f"Short answer at {bloom_level} level; relationship between "
                        f"'{rel.concept_a}' and '{rel.concept_b}'"
                    ),
                )

            elif terms:
                target = self._rotate_term(terms)
                stem = (
                    f"<p>In your own words, briefly {verb} what "
                    f"<em>{target.term}</em> means and why it is significant.</p>"
                )
                return QuestionData(
                    question_id=question_id,
                    question_type="short_answer",
                    stem=stem,
                    bloom_level=bloom_level,
                    objective_id=objective_id,
                    correct_answer=target.definition,
                    points=5.0,
                    feedback=f"<p>{target.context_sentence}</p>",
                    source_chunks=[target.source_chunk_id],
                    generation_rationale=(
                        f"Short answer at {bloom_level} level; defines term "
                        f"'{target.term}'"
                    ),
                )

        # Skip-emit rather than a templated placeholder.
        reason = (
            "no_source_chunks"
            if not source_chunks
            else "no_extractable_content"
        )
        return SkippedItem(
            question_id=question_id,
            question_type="short_answer",
            bloom_level=bloom_level,
            objective_id=objective_id,
            reason=reason,
        )

    #: Negation cue tokens. When the SOURCE claim already carries one,
    #: a naive regex negation risks a double-negative (net-TRUE) statement, so
    #: the correctness check rejects it. When the NEGATED string carries two or
    #: more, the negation itself double-negated (also risks net-TRUE).
    _NEGATION_CUES: frozenset = frozenset({
        "not", "no", "never", "none", "cannot", "cant", "wont", "dont",
        "doesnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
        "without", "neither", "nor", "nothing", "nowhere", "nobody",
    })

    @staticmethod
    def _negation_cue_count(text: str) -> int:
        """Count negation cue tokens in ``text`` (apostrophes stripped).

        ``don't`` / ``isn't`` normalise to ``dont`` / ``isnt`` so the
        contraction forms are counted alongside the bare ``not``.
        """
        toks = re.findall(r"[a-z']+", text.lower())
        cues = 0
        for tok in toks:
            bare = tok.replace("'", "")
            if bare in AssessmentGenerator._NEGATION_CUES:
                cues += 1
        return cues

    @classmethod
    def _negation_is_verifiably_false(
        cls, original: str, negated: str
    ) -> bool:
        """Deterministic check that ``negated`` is non-trivially false.

        ``_negate_statement`` flips a claim's polarity with a regex; with ZERO
        correctness check a regex-negated statement can be accidentally TRUE
        (a double-negation of an already-negative source, or a no-op that
        changed nothing). This gate rejects those so the caller drops the bad
        negation and regenerates from another statement.

        Rejects (→ ``False``) when:
          * the negation is empty / whitespace, or (case-insensitively) equal
            to the source — a VACUOUS negation that changed nothing;
          * the SOURCE claim already carries a negation cue (not/never/no/…) —
            negating an already-negative statement risks a net-TRUE
            double-negative that can't be cheaply verified as false;
          * the NEGATED string carries ≥2 negation cues, or an adjacent
            ``not … not`` — the negation introduced a DOUBLE negation.

        Accepts (→ ``True``) a single-cue insert (``is`` → ``is not``), a
        polar-qualifier swap that adds one cue (``always`` → ``never``), or a
        cue-free polarity swap (``increases`` → ``decreases``,
        ``before`` → ``after``) — each is a checkable falsification of a
        positive source claim.
        """
        if not negated or not negated.strip():
            return False
        if negated.strip().lower() == (original or "").strip().lower():
            return False
        # Source already negative → double-negation / unverifiable-truth risk.
        if cls._negation_cue_count(original or "") >= 1:
            return False
        # Negation introduced a double negation.
        if cls._negation_cue_count(negated) >= 2:
            return False
        if re.search(r"\bnot\s+not\b", negated.lower()):
            return False
        return True

    @classmethod
    def _safe_negate_statement(cls, statement: str) -> Optional[str]:
        """Negate ``statement`` and return it ONLY if verifiably false.

        Wraps :meth:`_negate_statement` with the correctness check.
        Returns ``None`` when the produced negation could be accidentally
        TRUE (so the caller drops it and regenerates), else the negated text.
        """
        negated = cls._negate_statement(statement)
        if cls._negation_is_verifiably_false(statement, negated):
            return negated
        return None

    @staticmethod
    def _negate_statement(statement: str) -> str:
        """Negate a factual statement for T/F false items or MCQ distractors.

        Uses simple verb-aware negation: inserts 'not' after the first
        auxiliary/copula verb, or swaps key qualifiers.

        NOTE: this is the RAW negation — it has no correctness check and can
        emit a statement that is accidentally TRUE. Callers that need a
        verifiably-false statement (T/F "false" stems, negated MCQ
        distractors) MUST route through :meth:`_safe_negate_statement`.
        """
        # Try qualifier swaps first (more natural sounding)
        swaps = [
            (r"\balways\b", "never"),
            (r"\bnever\b", "always"),
            (r"\ball\b", "no"),
            (r"\bno\b", "all"),
            (r"\bincreases?\b", "decreases"),
            (r"\bdecreases?\b", "increases"),
            (r"\bmore\b", "less"),
            (r"\bless\b", "more"),
            (r"\bbefore\b", "after"),
            (r"\bafter\b", "before"),
        ]
        for pattern, replacement in swaps:
            if re.search(pattern, statement, re.IGNORECASE):
                return re.sub(pattern, replacement, statement, count=1, flags=re.IGNORECASE)

        # Insert 'not' after auxiliary/copula verbs
        negation_targets = [
            (r"\b(is)\b", r"\1 not"),
            (r"\b(are)\b", r"\1 not"),
            (r"\b(was)\b", r"\1 not"),
            (r"\b(were)\b", r"\1 not"),
            (r"\b(has)\b", r"\1 not"),
            (r"\b(have)\b", r"\1 not"),
            (r"\b(can)\b", r"\1not"),
            (r"\b(will)\b", r"\1 not"),
            (r"\b(does)\b", r"\1 not"),
            (r"\b(do)\b", r"\1 not"),
            (r"\b(should)\b", r"\1 not"),
            (r"\b(would)\b", r"\1 not"),
            (r"\b(provides?)\b", r"does not provide"),
            (r"\b(requires?)\b", r"does not require"),
            (r"\b(includes?)\b", r"does not include"),
        ]
        for pattern, replacement in negation_targets:
            if re.search(pattern, statement, re.IGNORECASE):
                return re.sub(pattern, replacement, statement, count=1, flags=re.IGNORECASE)

        # Last resort: prepend "It is not true that"
        return f"It is not true that {statement[0].lower()}{statement[1:]}"

    # ------------------------------------------------------------------ #
    # Deterministic diversified tier
    # ------------------------------------------------------------------ #

    def _clamp_and_record(
        self, question_id: str, subtype: str, bloom_level: str
    ) -> str:
        """Clamp ``bloom_level`` to ``subtype``'s ceiling; record any clamp."""
        clamped = _clamp_bloom(bloom_level, subtype)
        if clamped != bloom_level and self._stats is not None:
            self._stats.record_clamp(question_id, subtype, bloom_level, clamped)
        return clamped

    @staticmethod
    def _trim(text: str, cap: int = 200) -> str:
        text = (text or "").strip()
        return text if len(text) <= cap else text[: cap - 3] + "..."

    def _assemble_single_key_choices(
        self,
        correct_text: str,
        distractor_specs: List[Any],
        option_count: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Assemble a 1-key MC choice list honoring the item-writing rules.

        ``distractor_specs`` is a list of ``str`` or ``(text, note)`` tuples.
        Bans NOTA/AOTA + absolute-term options, drops duplicates / options
        equal to the key, caps distractors to ``option_count - 1``. Returns
        None when fewer than 2 functional distractors survive (→ caller skips).
        """
        key = self._trim(correct_text)
        if not key or _is_banned_option(key):
            return None
        seen = {re.sub(r"<[^>]+>", " ", key).strip().lower()}
        kept: List[Dict[str, Any]] = []
        for spec in distractor_specs:
            if len(kept) >= option_count - 1:
                break
            if isinstance(spec, tuple):
                d_text, note = spec[0], spec[1]
            else:
                d_text, note = spec, None
            d_text = self._trim(d_text)
            norm = re.sub(r"<[^>]+>", " ", d_text).strip().lower()
            if not d_text or not norm or norm in seen:
                continue
            if _is_banned_option(d_text):
                continue
            seen.add(norm)
            choice: Dict[str, Any] = {"text": f"<p>{d_text}</p>", "is_correct": False}
            if note:
                choice["misconception_note"] = note
            kept.append(choice)
        if len(kept) < 2:
            return None
        return [{"text": f"<p>{key}</p>", "is_correct": True}] + kept

    def build_multiple_response(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """MULTIPLE-RESPONSE select-all — genuine apply.

        From a worked-example chunk with >=3 solution steps: the real steps are
        the keys ("which of the following ARE steps in solving X"); distractors
        are misconception-transformed steps + foreign steps. Portable via
        cc.multiple_response.
        """
        subtype = "mc_multiple_response"
        if not source_chunks:
            return SkippedItem(question_id, "multiple_response", bloom_level,
                               objective_id, "no_source_chunks")
        procedures = self._content_extractor.extract_procedures(source_chunks)
        target = next((p for p in procedures if len(p.steps) >= 3), None)
        if target is None:
            return SkippedItem(question_id, "multiple_response", bloom_level,
                               objective_id, "no_multistep_procedure")
        keys = [self._trim(s) for s in target.steps[:4] if s.strip()]
        keys = [k for k in keys if k and not _is_banned_option(k)]
        if len(keys) < 2:
            return SkippedItem(question_id, "multiple_response", bloom_level,
                               objective_id, "insufficient_real_steps")
        key_norms = {re.sub(r"<[^>]+>", " ", k).lower().strip() for k in keys}
        distractors: List[Dict[str, Any]] = []
        # Misconception-transformed steps.
        for step in target.steps:
            if len(distractors) >= 3:
                break
            out = _apply_first_misconception(step)
            if out is None:
                continue
            d_text, _err, note = out
            d_text = self._trim(d_text)
            norm = re.sub(r"<[^>]+>", " ", d_text).lower().strip()
            if norm in key_norms or _is_banned_option(d_text):
                continue
            key_norms.add(norm)
            distractors.append({"text": f"<p>{d_text}</p>", "is_correct": False,
                                "misconception_note": note})
        # Foreign steps from OTHER procedures (real text, not fabricated).
        for p in procedures:
            if p is target:
                continue
            for step in p.steps:
                if len(distractors) >= 3:
                    break
                d_text = self._trim(step)
                norm = re.sub(r"<[^>]+>", " ", d_text).lower().strip()
                if not d_text or norm in key_norms or _is_banned_option(d_text):
                    continue
                key_norms.add(norm)
                distractors.append({"text": f"<p>{d_text}</p>", "is_correct": False})
        if len(distractors) < 2:
            return SkippedItem(question_id, "multiple_response", bloom_level,
                               objective_id, "insufficient_distractors")
        clamped = self._clamp_and_record(question_id, subtype, bloom_level)
        choices = [{"text": f"<p>{k}</p>", "is_correct": True} for k in keys]
        choices.extend(distractors)
        return QuestionData(
            question_id=question_id,
            question_type="multiple_response",
            stem=(f"<p>Select <strong>all</strong> of the following that are "
                  f"correct steps in {self._trim(target.title.lower(), 80)}.</p>"),
            bloom_level=clamped,
            objective_id=objective_id,
            choices=choices,
            correct_answers=keys,
            item_subtype=subtype,
            points=float(len(keys)),
            feedback=("<p>The correct steps are those drawn directly from the "
                      "worked solution.</p>"),
            source_chunks=[target.source_chunk_id],
            generation_rationale=(
                f"Multiple-response select-all from {len(keys)} real steps of "
                f"'{target.title}'; {len(distractors)} distractors "
                f"(misconception-transformed + foreign steps)"
            ),
        )

    def build_error_analysis(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """ERROR-ANALYSIS MC — analyze level.

        Perturb ONE step of a worked solution with a named misconception
        transform; render the full (perturbed) solution and ask which step
        contains the error. The misconception_note on the key names the error.
        """
        subtype = "error_analysis"
        if not source_chunks:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "no_source_chunks")
        procedures = self._content_extractor.extract_procedures(source_chunks)
        target = next((p for p in procedures if len(p.steps) >= 3), None)
        if target is None:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "no_multistep_procedure")
        # Find the first step a misconception transform applies to.
        perturb_idx = None
        note = None
        for idx, step in enumerate(target.steps):
            out = _apply_first_misconception(step)
            if out is not None:
                perturb_idx, _perturbed, note = idx, out[0], out[1]
                break
        if perturb_idx is None:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "no_perturbable_step")
        _out = _apply_first_misconception(target.steps[perturb_idx])
        perturbed_text = _out[0]
        # Render the solution with the perturbed step in place.
        rendered_steps: List[str] = []
        for i, step in enumerate(target.steps):
            body = perturbed_text if i == perturb_idx else step
            rendered_steps.append(f"Step {i + 1}: {self._trim(body, 160)}")
        solution_html = "".join(f"<li>{s}</li>" for s in rendered_steps)
        stem = (
            f"<p>A student solved <em>{self._trim(target.title.lower(), 80)}</em> "
            f"as follows:</p><ol>{solution_html}</ol>"
            f"<p>Which step contains the error?</p>"
        )
        # Options = the step labels; key = the perturbed step's label.
        choices: List[Dict[str, Any]] = []
        for i in range(len(target.steps)):
            is_key = (i == perturb_idx)
            choice: Dict[str, Any] = {
                "text": f"<p>Step {i + 1}</p>", "is_correct": is_key,
            }
            if is_key and note:
                choice["misconception_note"] = note
            choices.append(choice)
        clamped = self._clamp_and_record(question_id, subtype, bloom_level)
        return QuestionData(
            question_id=question_id,
            question_type="multiple_choice",
            stem=stem,
            bloom_level=clamped,
            objective_id=objective_id,
            choices=choices,
            correct_answer=f"Step {perturb_idx + 1}",
            item_subtype=subtype,
            points=3.0,
            feedback=f"<p>Step {perturb_idx + 1} is incorrect. {note}</p>",
            source_chunks=[target.source_chunk_id],
            generation_rationale=(
                f"Error-analysis MC; perturbed step {perturb_idx + 1} of "
                f"'{target.title}' with misconception '{note}'"
            ),
        )

    def build_numeric_fib(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """NUMERIC-FIB — real apply/compute, sympy-VERIFIED key.

        Harvests a single-variable equation from a worked-example chunk, solves
        it with sympy, and ships ONLY when the computed key checks out under
        substitution. Unverifiable candidates → SkippedItem with a reason.
        """
        subtype = "fib_numeric"
        if not source_chunks:
            return SkippedItem(question_id, "fill_in_blank", bloom_level,
                               objective_id, "no_source_chunks")
        try:
            import sympy  # type: ignore
            from lib.validators.worked_example_math import (
                _iter_fragments, _split_equation_sides, _parse_side,
            )
        except Exception:  # noqa: BLE001 — sympy is an optional extra
            return SkippedItem(question_id, "fill_in_blank", bloom_level,
                               objective_id, "sympy_missing")

        # Assemble candidate (fragment, chunk_id) pairs in a stable order:
        # first the marked-math harvest per chunk, THEN — only when
        # ED4ALL_ASSESSMENT_NUMERIC_RECOVERY is on — the plain-text equations
        # recovered from Solution/Check/Step-N apparatus regions. Recovery must
        # be appended AFTER the marked-math candidates so the flag-off path
        # stays bitwise-identical (recovery returns []).
        candidates: List[Tuple[str, str]] = []
        for chunk in source_chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            html = chunk.get("text", "") or ""
            for frag, _ctx in _iter_fragments(html):
                candidates.append((frag, chunk_id))
        candidates.extend(
            self._content_extractor.recover_numeric_equation_candidates(
                source_chunks
            )
        )

        for frag, chunk_id in candidates:
            sides = _split_equation_sides(frag)
            if sides is None or len(sides) != 2:
                continue
            lhs = _parse_side(sides[0])
            rhs = _parse_side(sides[1])
            if lhs is None or rhs is None:
                continue
            free = (lhs.free_symbols | rhs.free_symbols)
            if len(free) != 1:
                continue
            var = next(iter(free))
            # Skip a trivial ``var = number`` restatement (nothing to solve).
            expr = sympy.simplify(lhs - rhs)
            if expr == 0:
                continue
            try:
                sols = sympy.solve(sympy.Eq(lhs, rhs), var)
            except Exception:  # noqa: BLE001
                continue
            numeric_sols = [s for s in sols
                            if getattr(s, "free_symbols", set()) == set()
                            and getattr(s, "is_real", None) is not False]
            if len(numeric_sols) != 1:
                continue
            sol = numeric_sols[0]
            # VERIFY by substitution — ship only if it checks out.
            try:
                residual = sympy.simplify(expr.subs(var, sol))
            except Exception:  # noqa: BLE001
                continue
            if residual != 0:
                continue
            # Render a clean numeric key.
            try:
                fval = float(sol)
                key = str(int(fval)) if fval == int(fval) else str(fval)
            except Exception:  # noqa: BLE001
                key = str(sol)
            clamped = self._clamp_and_record(question_id, subtype, bloom_level)
            eq_disp = f"{sides[0].strip()} = {sides[1].strip()}"
            return QuestionData(
                question_id=question_id,
                question_type="fill_in_blank",
                stem=(f"<p>Solve for <em>{var}</em>: "
                      f"<code>{eq_disp}</code>. Enter the value of "
                      f"<em>{var}</em>.</p>"),
                bloom_level=clamped,
                objective_id=objective_id,
                correct_answer=key,
                item_subtype=subtype,
                points=2.0,
                feedback=(f"<p>{var} = {key} (verified by substitution). "
                          f"Set the accepted-answer tolerance in the LMS "
                          f"editor.</p>"),
                source_chunks=[chunk_id],
                generation_rationale=(
                    f"Numeric FIB; sympy-solved {eq_disp} → {var}={key}, "
                    f"verified residual=0"
                ),
            )
        if self._stats is not None:
            self._stats.record_skip(subtype, "numeric_unverifiable")
        return SkippedItem(question_id, "fill_in_blank", bloom_level,
                           objective_id, "numeric_unverifiable")

    def build_matching_mc(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """MATCHING-as-MC — one MC per term, distractors = sibling defs.

        Homogeneous by construction: every option is a definition drawn from
        the same section's term set.
        """
        subtype = "matching_mc"
        if not source_chunks:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "no_source_chunks")
        terms = self._content_extractor.extract_key_terms(source_chunks)
        if len(terms) < 3:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "insufficient_terms")
        target = self._rotate_term(terms)
        self._div_offset += 1
        pool = [t for t in terms if t.term != target.term
                and t.definition.strip().lower() != target.definition.strip().lower()]
        n = len(pool)
        distractor_specs: List[str] = []
        for k in range(n):
            other = pool[(self._div_offset + k) % n]
            distractor_specs.append(other.definition)
        option_count = _resolve_option_count()
        choices = self._assemble_single_key_choices(
            target.definition, distractor_specs, option_count
        )
        if choices is None:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "insufficient_sibling_defs")
        clamped = self._clamp_and_record(question_id, subtype, bloom_level)
        return QuestionData(
            question_id=question_id,
            question_type="multiple_choice",
            stem=(f"<p>Which definition best matches the term "
                  f"<em>{target.term}</em>?</p>"),
            bloom_level=clamped,
            objective_id=objective_id,
            choices=choices,
            item_subtype=subtype,
            points=1.0,
            feedback=f"<p>{self._trim(target.context_sentence)}</p>",
            source_chunks=[target.source_chunk_id],
            generation_rationale=(
                f"Matching-as-MC for term '{target.term}'; "
                f"{len(choices) - 1} sibling-definition distractors"
            ),
        )

    def build_ordering_mc(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """ORDERING-as-MC — pick the correct order of steps.

        True sequence + 2 deterministic plausible permutations. Used sparingly
        (5% of the mix).
        """
        subtype = "ordering_mc"
        if not source_chunks:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "no_source_chunks")
        procedures = self._content_extractor.extract_procedures(source_chunks)
        target = next((p for p in procedures if len(p.steps) >= 3), None)
        if target is None:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "no_multistep_procedure")
        steps = [self._trim(s, 90) for s in target.steps[:4] if s.strip()]
        if len(steps) < 3:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "insufficient_steps")

        def _render(order: List[int]) -> str:
            return " → ".join(f"({j + 1}) {steps[o]}" for j, o in enumerate(order))

        correct_order = list(range(len(steps)))
        # Two deterministic plausible permutations: swap first two; reverse.
        perm_swap = correct_order.copy()
        perm_swap[0], perm_swap[1] = perm_swap[1], perm_swap[0]
        perm_rev = list(reversed(correct_order))
        distractor_specs: List[str] = []
        correct_render = _render(correct_order)
        for perm in (perm_swap, perm_rev):
            if perm != correct_order:
                distractor_specs.append(_render(perm))
        option_count = min(3, _resolve_option_count())
        choices = self._assemble_single_key_choices(
            correct_render, distractor_specs, max(option_count, 3)
        )
        if choices is None:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "insufficient_permutations")
        clamped = self._clamp_and_record(question_id, subtype, bloom_level)
        return QuestionData(
            question_id=question_id,
            question_type="multiple_choice",
            stem=(f"<p>Which of the following shows the correct order of steps "
                  f"to {self._trim(target.title.lower(), 80)}?</p>"),
            bloom_level=clamped,
            objective_id=objective_id,
            choices=choices,
            item_subtype=subtype,
            points=2.0,
            feedback="<p>The correct sequence follows the worked solution.</p>",
            source_chunks=[target.source_chunk_id],
            generation_rationale=(
                f"Ordering-as-MC over {len(steps)} steps of '{target.title}'"
            ),
        )

    def build_two_tier(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """TWO-TIER answer+reason — Treagust misconception diagnosis.

        Returns a LIST of two linked QuestionData: an answer tier (a term MC)
        and a reason tier whose distractors are misconception statements. The
        two are linked via ``linked_item_id``.
        """
        if not source_chunks:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "no_source_chunks")
        terms = self._content_extractor.extract_key_terms(source_chunks)
        if len(terms) < 3:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "insufficient_terms")
        # Harvest misconception statements for the reason tier's distractors.
        misconception_texts: List[str] = []
        for chunk in source_chunks:
            for mc in (chunk.get("misconceptions") or []):
                if isinstance(mc, dict) and mc.get("misconception"):
                    misconception_texts.append(mc["misconception"])
        target = self._rotate_term(terms)
        self._div_offset += 1
        pool = [t for t in terms if t.term != target.term]
        # Answer tier — a term MC.
        n = len(pool)
        ans_distractors = [pool[(self._div_offset + k) % n].definition
                           for k in range(n)]
        option_count = _resolve_option_count()
        ans_choices = self._assemble_single_key_choices(
            target.definition, ans_distractors, option_count
        )
        if ans_choices is None:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "insufficient_sibling_defs")
        # Reason tier — key is the grounded rationale; distractors are
        # misconception statements (falling back to transformed sibling defs).
        reason_key = self._trim(target.context_sentence)
        reason_specs: List[str] = list(misconception_texts)
        if len(reason_specs) < 2:
            for t in pool:
                out = _apply_first_misconception(t.definition)
                if out is not None:
                    reason_specs.append(out[0])
                if len(reason_specs) >= 3:
                    break
        # Last-resort: sibling definitions as (wrong-because-off-topic) reasons.
        if len(reason_specs) < 2:
            reason_specs.extend(t.definition for t in pool)
        reason_choices = self._assemble_single_key_choices(
            reason_key, reason_specs, option_count
        )
        if reason_choices is None:
            return SkippedItem(question_id, "multiple_choice", bloom_level,
                               objective_id, "insufficient_reason_distractors")
        ans_id = question_id
        reason_id = f"{question_id}-reason"
        ans_clamped = self._clamp_and_record(ans_id, "two_tier_answer", bloom_level)
        reason_clamped = self._clamp_and_record(
            reason_id, "two_tier_reason", bloom_level)
        answer_item = QuestionData(
            question_id=ans_id,
            question_type="multiple_choice",
            stem=(f"<p><strong>Tier 1.</strong> Which definition best matches "
                  f"<em>{target.term}</em>?</p>"),
            bloom_level=ans_clamped,
            objective_id=objective_id,
            choices=ans_choices,
            item_subtype="two_tier_answer",
            points=1.0,
            feedback=f"<p>{self._trim(target.context_sentence)}</p>",
            source_chunks=[target.source_chunk_id],
            linked_item_id=reason_id,
            generation_rationale=(
                f"Two-tier answer tier for term '{target.term}'"),
        )
        reason_item = QuestionData(
            question_id=reason_id,
            question_type="multiple_choice",
            stem=(f"<p><strong>Tier 2.</strong> What is the best reason your "
                  f"Tier 1 answer for <em>{target.term}</em> is correct?</p>"),
            bloom_level=reason_clamped,
            objective_id=objective_id,
            choices=reason_choices,
            item_subtype="two_tier_reason",
            points=2.0,
            feedback=f"<p>{self._trim(target.context_sentence)}</p>",
            source_chunks=[target.source_chunk_id],
            linked_item_id=ans_id,
            generation_rationale=(
                f"Two-tier reason tier; distractors are misconception "
                f"statements for term '{target.term}'"),
        )
        return [answer_item, reason_item]

    def build_tf_justified(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """True/False (subtype ``tf``) — cheap variety.

        Reuses the verified-negation T/F path and stamps the subtype + Bloom
        ceiling. The linked short-FIB justification tier is deliberately not
        emitted — TF alone is portable and honest.
        """
        result = self._generate_true_false(
            question_id, objective_id, bloom_level,
            BLOOM_LEVELS.get(bloom_level, BLOOM_LEVELS["understand"]),
            source_chunks,
        )
        if isinstance(result, QuestionData):
            result.item_subtype = "tf"
            result.bloom_level = self._clamp_and_record(
                question_id, "tf", bloom_level)
        return result

    def build_mc_single(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """Classic single-answer MC (subtype ``mc_single``) — fills the mix
        remainder. Reuses the content-grounded MC path and stamps the subtype
        + Bloom ceiling; banned options are already excluded by the MC path's
        own distractor synthesis plus the assembly ban.
        """
        result = self._generate_multiple_choice(
            question_id, objective_id, bloom_level,
            BLOOM_LEVELS.get(bloom_level, BLOOM_LEVELS["understand"]),
            source_chunks,
        )
        if isinstance(result, QuestionData):
            # Drop any banned option the legacy path may have admitted.
            result.choices = [c for c in result.choices
                              if c.get("is_correct")
                              or not _is_banned_option(c.get("text", ""))]
            result.item_subtype = "mc_single"
            result.bloom_level = self._clamp_and_record(
                question_id, "mc_single", bloom_level)
        return result

    # ------------------------------------------------------------------ #
    # Written-response constructed-answer tier (per-item analytic rubrics)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rubric_criterion(
        criterion: str, cites: List[str]
    ) -> Dict[str, Any]:
        """One analytic-rubric criterion row citing its source chunk id(s)."""
        return {
            "criterion": criterion,
            "cites": [c for c in cites if c],
            "levels": [dict(lv) for lv in _WRITTEN_RUBRIC_LEVELS],
        }

    def _misconception_deductions(
        self, texts: List[str], chunk_id: str
    ) -> List[Dict[str, Any]]:
        """Build 'common error' deduction rows from the misconception transforms
        applicable to ``texts`` (each cites ``chunk_id``). Nothing is invented —
        a transform fires only on structure it genuinely matches; a chunk-authored
        misconception string is passed through verbatim when no transform applies.
        """
        deductions: List[Dict[str, Any]] = []
        seen: set = set()
        for t in texts:
            out = _apply_first_misconception(t or "")
            if out is None:
                continue
            _perturbed, err, note = out
            if err in seen:
                continue
            seen.add(err)
            deductions.append({
                "error": err,
                "note": note,
                "points": -1.0,
                "cites": [chunk_id] if chunk_id else [],
            })
            if len(deductions) >= 3:
                break
        return deductions

    def build_short_answer(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """SHORT-ANSWER — 2-4 sentence constructed response.

        Drafted from explanation/concept chunks: a SPECIFIC ``Explain why <claim>``
        stem grounded in a real factual statement (never a bare "discuss X").
        Carries a per-item analytic rubric whose criteria are derived from the
        source chunk's key concepts + the claim itself (each criterion cites the
        chunk id); deductions are drawn from the item's misconception transforms.
        NO Bloom ceiling (an honest higher-Bloom vehicle).
        """
        subtype = "short_answer"
        if not source_chunks:
            return SkippedItem(question_id, "essay", bloom_level,
                               objective_id, "no_source_chunks")
        statements = self._content_extractor.extract_factual_statements(source_chunks)
        terms = self._content_extractor.extract_key_terms(source_chunks)
        claim = None
        claim_chunk = ""
        if statements:
            st = self._rotate_statement(statements)
            claim = self._trim(st.statement, 220)
            claim_chunk = st.source_chunk_id
        elif terms:
            t0 = self._rotate_term(terms)
            claim = self._trim(
                f"{t0.term} is {t0.definition}", 220)
            claim_chunk = t0.source_chunk_id
        if not claim or not claim.strip():
            return SkippedItem(question_id, "essay", bloom_level,
                               objective_id, "no_specific_claim")
        # Rubric criteria: the claim itself + up to two source key concepts.
        criteria: List[Dict[str, Any]] = [
            self._rubric_criterion(
                f"Correctly explains why the following holds: {claim}",
                [claim_chunk]),
        ]
        cited: set = {claim_chunk} if claim_chunk else set()
        for t in terms[:2]:
            criteria.append(self._rubric_criterion(
                f"Uses the concept '{t.term}' accurately "
                f"({self._trim(t.definition, 120)})",
                [t.source_chunk_id]))
            if t.source_chunk_id:
                cited.add(t.source_chunk_id)
        criteria.append(self._rubric_criterion(
            "Response is 2-4 sentences, clear, and free of major errors.",
            [claim_chunk] if claim_chunk else sorted(cited)[:1]))
        # Deductions from misconceptions relevant to this item's content.
        mc_texts: List[str] = [claim]
        for chunk in source_chunks:
            for mc in (chunk.get("misconceptions") or []):
                if isinstance(mc, dict) and mc.get("misconception"):
                    mc_texts.append(str(mc["misconception"]))
        deductions = self._misconception_deductions(
            mc_texts, claim_chunk or (sorted(cited)[0] if cited else ""))
        model_answer = claim
        # NO Bloom ceiling for written types — assert the requested level.
        return QuestionData(
            question_id=question_id,
            question_type="essay",
            stem=(f"<p>In 2-4 sentences, explain <strong>why</strong> the "
                  f"following is true, drawing on the reasoning in the course "
                  f"material: <em>{claim}</em></p>"),
            bloom_level=bloom_level,
            objective_id=objective_id,
            correct_answer=model_answer,
            item_subtype=subtype,
            points=5.0,
            feedback=(f"<p>A strong response explains: {model_answer}</p>"),
            source_chunks=sorted(cited) or ([claim_chunk] if claim_chunk else []),
            rubric={"criteria": criteria, "deductions": deductions},
            generation_rationale=(
                f"Short-answer constructed response grounded in a specific "
                f"claim from chunk {claim_chunk!r}; {len(criteria)}-criterion "
                f"analytic rubric, {len(deductions)} misconception deduction(s)"
            ),
        )

    def build_extended_response(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """EXTENDED-RESPONSE — "solve and show your work" / "compare X and Y".

        Prefers a worked-example chunk (>=2 steps) → a solve-and-justify prompt
        whose rubric criteria are the real solution steps (each cites the chunk)
        and whose deductions are the applicable misconception transforms;
        falls back to a term-pair chunk → a compare/contrast prompt whose rubric
        criteria are the two grounded definitions + a contrast criterion. NO
        Bloom ceiling (an honest higher-Bloom vehicle).
        """
        subtype = "extended_response"
        if not source_chunks:
            return SkippedItem(question_id, "essay", bloom_level,
                               objective_id, "no_source_chunks")
        procedures = self._content_extractor.extract_procedures(source_chunks)
        target = next((p for p in procedures if len(p.steps) >= 2), None)
        if target is not None:
            steps = [self._trim(s, 160) for s in target.steps if s.strip()]
            criteria = [
                self._rubric_criterion(
                    f"Shows and justifies the step: {s}",
                    [target.source_chunk_id])
                for s in steps
            ]
            criteria.append(self._rubric_criterion(
                "Presents the work in a clear, logically-ordered sequence.",
                [target.source_chunk_id]))
            deductions = self._misconception_deductions(
                list(target.steps), target.source_chunk_id)
            model_answer = "".join(f"<li>{self._trim(s, 200)}</li>"
                                   for s in target.steps)
            return QuestionData(
                question_id=question_id,
                question_type="essay",
                stem=(f"<p>Solve <em>{self._trim(target.title.lower(), 90)}</em>. "
                      f"<strong>Show your work</strong> step by step and justify "
                      f"each step you take.</p>"),
                bloom_level=bloom_level,
                objective_id=objective_id,
                item_subtype=subtype,
                points=10.0,
                feedback=(f"<p>A complete solution:</p><ol>{model_answer}</ol>"),
                source_chunks=[target.source_chunk_id],
                rubric={"criteria": criteria, "deductions": deductions},
                generation_rationale=(
                    f"Extended-response solve-and-show-work over "
                    f"{len(steps)} real steps of '{target.title}'; "
                    f"per-step analytic rubric, {len(deductions)} deduction(s)"
                ),
            )
        # Fallback — compare/contrast a term pair from the same section.
        terms = self._content_extractor.extract_key_terms(source_chunks)
        if len(terms) < 2:
            return SkippedItem(question_id, "essay", bloom_level,
                               objective_id, "no_procedure_or_term_pair")
        a = self._rotate_term(terms)
        b = next((t for t in terms if t.term != a.term), None)
        if b is None:
            return SkippedItem(question_id, "essay", bloom_level,
                               objective_id, "no_second_term")
        cited = sorted({a.source_chunk_id, b.source_chunk_id} - {""})
        criteria = [
            self._rubric_criterion(
                f"Correctly defines '{a.term}' "
                f"({self._trim(a.definition, 120)}).", [a.source_chunk_id]),
            self._rubric_criterion(
                f"Correctly defines '{b.term}' "
                f"({self._trim(b.definition, 120)}).", [b.source_chunk_id]),
            self._rubric_criterion(
                f"Identifies at least one substantive difference and one "
                f"connection between '{a.term}' and '{b.term}'.", cited),
        ]
        mc_texts = [a.definition, b.definition]
        for chunk in source_chunks:
            for mc in (chunk.get("misconceptions") or []):
                if isinstance(mc, dict) and mc.get("misconception"):
                    mc_texts.append(str(mc["misconception"]))
        deductions = self._misconception_deductions(
            mc_texts, cited[0] if cited else "")
        return QuestionData(
            question_id=question_id,
            question_type="essay",
            stem=(f"<p><strong>Compare and contrast</strong> <em>{a.term}</em> "
                  f"and <em>{b.term}</em>. Explain how they differ and where they "
                  f"connect, using specific examples from the course material.</p>"),
            bloom_level=bloom_level,
            objective_id=objective_id,
            item_subtype=subtype,
            points=10.0,
            feedback=(f"<p>{a.term}: {self._trim(a.definition)}</p>"
                      f"<p>{b.term}: {self._trim(b.definition)}</p>"),
            source_chunks=cited,
            rubric={"criteria": criteria, "deductions": deductions},
            generation_rationale=(
                f"Extended-response compare/contrast of '{a.term}' vs "
                f"'{b.term}'; grounded-definition analytic rubric, "
                f"{len(deductions)} deduction(s)"
            ),
        )

    #: Subtype → builder-method name dispatch for the diversified tier.
    _SUBTYPE_BUILDERS = {
        "mc_multiple_response": "build_multiple_response",
        "error_analysis": "build_error_analysis",
        "fib_numeric": "build_numeric_fib",
        "matching_mc": "build_matching_mc",
        "ordering_mc": "build_ordering_mc",
        "two_tier_answer": "build_two_tier",
        "tf_justified": "build_tf_justified",
        "short_answer": "build_short_answer",
        "extended_response": "build_extended_response",
        "mc_single": "build_mc_single",
    }

    def _build_by_subtype(
        self, subtype: str, question_id: str, objective_id: str,
        bloom_level: str, source_chunks: Optional[List[Dict[str, Any]]],
    ):
        """Dispatch to the builder for ``subtype``; returns QuestionData,
        a List[QuestionData] (two-tier), or SkippedItem."""
        method_name = self._SUBTYPE_BUILDERS.get(subtype, "build_mc_single")
        return getattr(self, method_name)(
            question_id, objective_id, bloom_level, source_chunks)

    def generate_diversified(
        self,
        course_code: str,
        objective_ids: List[str],
        bloom_levels: List[str],
        question_count: int = 10,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
        chunks_by_objective: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        ensure_objective_coverage: bool = False,
    ) -> AssessmentData:
        """Assemble a deterministic diversified assessment per the target mix.

        Plans the item-subtype mix (``plan_item_mix``), dispatches each planned
        slot to its builder (grounded to source chunk ids), and falls back to
        classic single-answer MC on a SkippedItem so per-CO coverage still
        holds. The realized subtype histogram + Bloom clamps + skips are
        recorded on ``self._stats`` (also returned via ``last_generation_stats``).
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        assessment_id = f"ASM-{course_code}-{session_id}"
        self._stats = GenerationStats()
        questions: List[QuestionData] = []
        skipped: List[SkippedItem] = []

        def _emit(result, subtype: str) -> None:
            items = result if isinstance(result, list) else [result]
            any_q = False
            for it in items:
                # Apparatus backstop — the SINGLE assembly seam for this tier.
                # ``generate()`` guards each ``_generate_question`` call site,
                # but the diversified builders reached ``questions`` ungated,
                # so turning ED4ALL_ASSESSMENT_DIVERSIFIED on routed generation
                # around the guard entirely: a real build shipped 159/626
                # distractors and 30/305 correct answers carrying figure
                # captions, HOW-TO banners and glyph alt-text. Guarding here
                # (rather than per builder) makes the backstop universal by
                # construction — a new subtype builder cannot bypass it.
                it = self._apparatus_guard(it)
                if isinstance(it, QuestionData):
                    questions.append(it)
                    self._stats.record_item(it.item_subtype or subtype)
                    any_q = True
                elif isinstance(it, SkippedItem):
                    skipped.append(it)
                    self._stats.record_skip(subtype, it.reason)
            return any_q

        def _scoped(obj_id: str) -> Optional[List[Dict[str, Any]]]:
            return (chunks_by_objective or {}).get(obj_id) or source_chunks

        plan = plan_item_mix(question_count)

        if ensure_objective_coverage:
            # One item per objective first (coverage wins), subtype rotating
            # over the planned mix by position.
            for idx, obj_id in enumerate(objective_ids):
                subtype = plan[idx % len(plan)] if plan else "mc_single"
                qid = f"Q-{str(uuid.uuid4())[:8]}"
                result = self._build_by_subtype(
                    subtype, qid, obj_id, bloom_levels[idx % len(bloom_levels)],
                    _scoped(obj_id))
                produced = _emit(result, subtype)
                if not produced:
                    # Fall back to classic MC so the CO still gets an item.
                    fb = self.build_mc_single(
                        f"Q-{str(uuid.uuid4())[:8]}", obj_id,
                        bloom_levels[0], _scoped(obj_id))
                    _emit(fb, "mc_single")

            # Repair genuine builder/fallback misses BEFORE count-fill. A
            # numeric fill starts again at objective zero and can otherwise
            # conceal missing tail objectives behind early duplicates.
            covered = {q.objective_id for q in questions}
            retry_limit = max(4, len(bloom_levels) * 2)
            for objective_index, objective_id in enumerate(objective_ids):
                if objective_id in covered:
                    continue
                objective_chunks = _scoped(objective_id)
                for retry_index in range(retry_limit):
                    subtype = (
                        plan[(objective_index + retry_index + 1) % len(plan)]
                        if plan else "mc_single"
                    )
                    bloom = bloom_levels[
                        (objective_index + retry_index) % len(bloom_levels)
                    ]
                    retry_chunks = objective_chunks
                    if objective_chunks:
                        offset = retry_index % len(objective_chunks)
                        retry_chunks = [
                            *objective_chunks[offset:],
                            *objective_chunks[:offset],
                        ]
                    produced = _emit(
                        self._build_by_subtype(
                            subtype,
                            f"Q-{str(uuid.uuid4())[:8]}",
                            objective_id,
                            bloom,
                            retry_chunks,
                        ),
                        subtype,
                    )
                    if not produced:
                        produced = _emit(
                            self.build_mc_single(
                                f"Q-{str(uuid.uuid4())[:8]}",
                                objective_id,
                                bloom,
                                retry_chunks,
                            ),
                            "mc_single",
                        )
                    if produced:
                        covered.add(objective_id)
                        break
            missing = [
                objective_id for objective_id in objective_ids
                if objective_id not in covered
            ]
            if missing:
                reasons = {
                    objective_id: sorted({
                        item.reason for item in skipped
                        if item.objective_id == objective_id
                    })
                    for objective_id in missing
                }
                raise RuntimeError(
                    "Diversified assessment could not produce a grounded "
                    f"item for objective_ids={missing}; reasons={reasons}"
                )

        # Count-driven fill across the planned mix.
        pos = 0
        attempts = 0
        max_attempts = max(1, question_count) * 4
        while len(questions) < question_count and attempts < max_attempts:
            subtype = plan[pos % len(plan)] if plan else "mc_single"
            obj_id = objective_ids[pos % len(objective_ids)]
            bloom = bloom_levels[pos % len(bloom_levels)]
            qid = f"Q-{str(uuid.uuid4())[:8]}"
            result = self._build_by_subtype(subtype, qid, obj_id, bloom, _scoped(obj_id))
            produced = _emit(result, subtype)
            if not produced:
                fb = self.build_mc_single(
                    f"Q-{str(uuid.uuid4())[:8]}", obj_id, bloom,
                    _scoped(obj_id))
                _emit(fb, "mc_single")
            pos += 1
            attempts += 1

        # Trim to the requested count. A two-tier builder contributes two
        # records for one objective; a naive prefix slice can therefore remove
        # the only item for objectives near the end of the coverage-first
        # pass. Reserve the first item for every requested objective before
        # filling remaining slots with the earliest extras.
        if len(questions) > question_count:
            if ensure_objective_coverage:
                first_index_by_objective: Dict[str, int] = {}
                for question_index, question in enumerate(questions):
                    first_index_by_objective.setdefault(
                        question.objective_id, question_index
                    )
                reserved = {
                    first_index_by_objective[objective_id]
                    for objective_id in objective_ids
                    if objective_id in first_index_by_objective
                }
                keep_indices = set(reserved)
                for question_index in range(len(questions)):
                    if len(keep_indices) >= question_count:
                        break
                    keep_indices.add(question_index)
                questions = [
                    question for question_index, question in enumerate(questions)
                    if question_index in keep_indices
                ]
            else:
                questions = questions[:question_count]

        # Bounded LLM apply-arm (flag-gated, default OFF). Additive on top of
        # the deterministic tier; every drafted item passes the sympy /
        # groundedness / trivote verify chain before it ships.
        if _apply_arm_enabled():
            apply_items = self._run_apply_arm(
                course_code=course_code,
                objective_ids=objective_ids,
                bloom_levels=bloom_levels,
                source_chunks=source_chunks,
                chunks_by_objective=chunks_by_objective,
            )
            for it in apply_items:
                questions.append(it)
                self._stats.record_item(it.item_subtype or _APPLY_ARM_SUBTYPE)

        if self.capture:
            self.capture.log_decision(
                decision_type="assessment_generation",
                decision=(
                    f"Diversified assessment {assessment_id}: "
                    f"{len(questions)} items across "
                    f"{len(self._stats.subtype_counts)} subtypes"),
                rationale=(
                    f"Deterministic diversified tier realized subtype mix "
                    f"{self._stats.subtype_counts} (target plan={dict((s, plan.count(s)) for s in set(plan))}); "
                    f"{self._stats.to_dict()['bloom_clamp_count']} Bloom-ceiling "
                    f"clamps, {len(skipped)} skips; all items grounded to source "
                    f"chunk ids, no LLM in the loop"),
            )

        self.last_generation_stats = self._stats
        return AssessmentData(
            assessment_id=assessment_id,
            title=f"Assessment: {course_code}",
            course_code=course_code,
            questions=questions,
            objectives_targeted=objective_ids,
            bloom_levels=bloom_levels,
            skipped_items=skipped,
        )

    # ------------------------------------------------------------------ #
    # Bounded LLM apply-arm (verify-gated). All methods below are only
    # reached when ED4ALL_ASSESSMENT_APPLY_ARM is on (checked in
    # generate_diversified), so the default path never touches them.
    # ------------------------------------------------------------------ #

    def _resolve_apply_arm_provider(self) -> Optional[Any]:
        """Resolve the apply-arm provider seat (injected or env-built), or None.

        License-gated: refuses Anthropic seats and roster-barred teachers, so the
        LLM-drafted assessment corpus (which feeds the SLM training surface)
        stays license-clean by construction. Returns None (arm skipped) on any
        resolution / license failure — the arm is ADDITIVE, never fatal.
        """
        if self._assessment_provider is not None:
            return self._assessment_provider
        provider_name = (
            os.environ.get("TRAINFORGE_ASSESSMENT_PROVIDER") or "local"
        ).strip().lower()
        if not _apply_arm_provider_allowed(provider_name, self._teacher_roster):
            logger.warning(
                "apply-arm: provider %r is license-barred (Anthropic / "
                "roster-barred teacher); skipping the arm — the assessment "
                "corpus stays license-clean.", provider_name,
            )
            return None
        try:
            from Trainforge.generators._assessment_provider import (
                AssessmentGeneratorProvider,
            )
            return AssessmentGeneratorProvider(
                provider=provider_name, capture=self.capture,
            )
        except Exception as exc:  # noqa: BLE001 — arm is best-effort
            logger.warning(
                "apply-arm: provider construction failed (%s); skipping arm.",
                exc,
            )
            return None

    def _run_apply_arm(
        self,
        *,
        course_code: str,
        objective_ids: List[str],
        bloom_levels: List[str],
        source_chunks: Optional[List[Dict[str, Any]]],
        chunks_by_objective: Optional[Dict[str, List[Dict[str, Any]]]],
    ) -> List[QuestionData]:
        """Draft a bounded count of apply items via the provider, verify each.

        Every drafted item runs the mandatory three-stage verify chain
        (sympy → groundedness → trivote-clamp) before it ships. Emits ONE
        ``assessment_generation`` DecisionCapture event per provider draft call
        whose rationale interpolates the runtime provider/model + the verify
        outcome counts. Returns the surviving (verified) ``QuestionData`` list;
        an empty list on any failure (the arm is additive).
        """
        provider = self._resolve_apply_arm_provider()
        if provider is None:
            return []
        max_items = _apply_arm_max()

        # Index every available chunk by id for cited-passage resolution.
        chunk_by_id: Dict[str, Dict[str, Any]] = {}
        pools: List[List[Dict[str, Any]]] = [source_chunks or []]
        pools += list((chunks_by_objective or {}).values())
        provider_chunks: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for pool in pools:
            for c in pool or []:
                cid = c.get("id") or c.get("chunk_id")
                if isinstance(cid, str) and cid.strip():
                    cid = cid.strip()
                    chunk_by_id.setdefault(cid, c)
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        provider_chunks.append(c)

        objectives = [
            {"id": oid, "statement": oid}
            for oid in objective_ids
            if isinstance(oid, str) and oid.strip()
        ]
        if not provider_chunks or not objectives:
            return []

        provider_name = getattr(provider, "_provider", "?")
        provider_model = getattr(provider, "_model", None)
        try:
            result = provider.generate_assessments(
                chunks=provider_chunks,
                objectives=objectives,
                target_count=max_items,
                course_code=str(course_code),
                bloom_levels=["apply"],
            )
        except Exception as exc:  # noqa: BLE001 — provider is best-effort
            logger.warning(
                "apply-arm: provider dispatch failed (%s); 0 items.", exc,
            )
            self._emit_apply_arm_decision(
                course_code=course_code, provider=provider_name,
                model=provider_model, drafted=0, shipped=0, dropped_sympy=0,
                dropped_grounding=0, dropped_other=0, clamped=0,
                error=str(exc)[:120],
            )
            return []

        drafted = result.get("questions") or []
        shipped: List[QuestionData] = []
        d_sympy = d_ground = d_other = clamped = 0
        for idx, q in enumerate(drafted[:max_items], start=1):
            qd, reason, was_clamped = self._verify_apply_item(
                q, idx, chunk_by_id,
            )
            if qd is None:
                if reason == "sympy_unverified":
                    d_sympy += 1
                elif reason and reason.startswith("grounding"):
                    d_ground += 1
                else:
                    d_other += 1
                if self._stats is not None:
                    self._stats.record_skip(_APPLY_ARM_SUBTYPE, reason or "dropped")
                continue
            if was_clamped:
                clamped += 1
            shipped.append(qd)

        self._emit_apply_arm_decision(
            course_code=course_code, provider=provider_name,
            model=provider_model, drafted=len(drafted), shipped=len(shipped),
            dropped_sympy=d_sympy, dropped_grounding=d_ground,
            dropped_other=d_other, clamped=clamped, error=None,
        )
        return shipped

    def _verify_apply_item(
        self,
        q: Dict[str, Any],
        idx: int,
        chunk_by_id: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[QuestionData], Optional[str], bool]:
        """Run the mandatory verify chain on one drafted item.

        Returns ``(QuestionData|None, drop_reason|None, was_bloom_clamped)``.
        Order: (1) sympy re-check on numeric stems/keys — FAIL-CLOSED;
        (2) groundedness entailment — require entailed; (3) Bloom trivote clamp.
        """
        stem = str(q.get("stem") or "").strip()
        oid = str(q.get("objective_id") or "").strip()
        claimed_bloom = str(q.get("bloom_level") or "apply").strip().lower()
        correct = q.get("correct_answer")
        src_ids = [
            s.strip() for s in (q.get("source_chunks") or [])
            if isinstance(s, str) and s.strip()
        ]
        if not stem or not oid:
            return None, "missing_stem_or_objective", False

        cited = [chunk_by_id[s] for s in src_ids if s in chunk_by_id]

        # (1) sympy re-check on numeric stems/keys — fail-closed.
        if self._answer_is_numeric(correct):
            if not self._sympy_verify_numeric(stem, cited, str(correct)):
                return None, "sympy_unverified", False

        # (2) groundedness entailment — require entailed.
        if not cited:
            return None, "grounding_no_cited_chunks", False
        verdict = self._groundedness_entailed(stem, correct, cited, src_ids)
        if verdict is not True:
            return None, f"grounding_{verdict}", False

        # (3) Bloom trivote clamp.
        final_bloom, was_clamped = self._trivote_bloom(stem, claimed_bloom)

        choices = q.get("choices")
        norm_choices: List[Dict[str, Any]] = []
        if isinstance(choices, list):
            for c in choices:
                if isinstance(c, dict):
                    norm_choices.append({
                        "text": str(c.get("text") or ""),
                        "is_correct": bool(c.get("is_correct")),
                    })
        qtype = str(q.get("question_type") or "multiple_choice").strip().lower()
        feedback = q.get("feedback")
        # A choice item's key lives in ``choices``; only a keyed non-choice item
        # (FIB / numeric) carries the scalar ``correct_answer``.
        scalar_key = (
            str(correct) if (correct is not None and not norm_choices) else None
        )
        return (
            QuestionData(
                question_id=f"Q-APPLY-{str(uuid.uuid4())[:8]}",
                question_type=qtype,
                stem=stem,
                bloom_level=final_bloom,
                objective_id=oid,
                choices=norm_choices,
                correct_answer=scalar_key,
                points=2.0,
                feedback=str(feedback) if isinstance(feedback, str) else None,
                item_subtype=_APPLY_ARM_SUBTYPE,
                source_chunks=src_ids,
                generation_rationale=(
                    f"LLM apply-arm item {idx}: sympy-"
                    f"{'verified' if self._answer_is_numeric(correct) else 'na'}, "
                    f"groundedness entailed vs {len(cited)} cited chunk(s), "
                    f"bloom {'clamped→'+final_bloom if was_clamped else final_bloom}"
                    f" (trivote)"
                ),
            ),
            None,
            was_clamped,
        )

    @staticmethod
    def _answer_is_numeric(val: Any) -> bool:
        """True when ``val`` reads as a number / simple fraction."""
        if val is None:
            return False
        s = str(val).strip()
        if not s:
            return False
        if re.fullmatch(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?", s):
            return True
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    def _sympy_verify_numeric(
        self,
        stem: str,
        cited: List[Dict[str, Any]],
        key: str,
    ) -> bool:
        """Fail-closed sympy re-check of a declared numeric key.

        Harvests equation candidates from the stem + cited chunk texts, solves
        each single-variable equation, and returns True iff SOME solved value
        equals the declared numeric ``key``. No equation / no match → False
        (the anti-hallucination contract: an unverifiable numeric claim is
        dropped, never shipped).
        """
        try:
            import sympy  # type: ignore
            from lib.validators.worked_example_math import (
                _iter_fragments, _split_equation_sides, _parse_side,
            )
            from Trainforge.generators.content_extractor import (
                extract_plaintext_equations,
                _strip_html as _cx_strip_html,
            )
        except Exception:  # noqa: BLE001 — sympy is an optional extra
            return False
        try:
            target = sympy.Rational(str(key).strip())
        except Exception:  # noqa: BLE001
            try:
                target = sympy.nsimplify(float(str(key).strip()))
            except Exception:  # noqa: BLE001
                return False

        texts = [stem] + [str(c.get("text") or "") for c in (cited or [])]
        frags: List[str] = []
        for text in texts:
            if not text:
                continue
            for frag, _ctx in _iter_fragments(text):
                frags.append(frag)
            # Plain-text (prose-embedded) equations — the word-problem shape.
            frags.extend(extract_plaintext_equations(_cx_strip_html(text)))

        for frag in frags:
            sides = _split_equation_sides(frag)
            if sides is None or len(sides) != 2:
                continue
            lhs = _parse_side(sides[0])
            rhs = _parse_side(sides[1])
            if lhs is None or rhs is None:
                continue
            free = lhs.free_symbols | rhs.free_symbols
            if len(free) != 1:
                continue
            var = next(iter(free))
            try:
                sols = sympy.solve(sympy.Eq(lhs, rhs), var)
            except Exception:  # noqa: BLE001
                continue
            for s in sols:
                if getattr(s, "free_symbols", set()):
                    continue
                try:
                    if sympy.simplify(s - target) == 0:
                        return True
                except Exception:  # noqa: BLE001
                    continue
        return False

    def _groundedness_entailed(
        self,
        stem: str,
        correct: Any,
        cited: List[Dict[str, Any]],
        src_ids: List[str],
    ) -> Any:
        """Return True when stem+key is ENTAILED by the cited chunks.

        Otherwise returns a short reason token (``unavailable`` / ``no_passages``
        / ``contradicted`` / ``unsupported`` / ``error``). Uses the injected
        ``groundedness_scorer`` when present (test seam); else lazy-imports the
        production ``score_groundedness``. GPU-gated: an unavailable scorer
        drops the item (we cannot confirm ``entailed``).
        """
        scorer = self._groundedness_scorer
        if scorer is None:
            try:
                from lib.retrieval.groundedness import score_groundedness
                scorer = score_groundedness
            except Exception:  # noqa: BLE001
                return "unavailable"
        passages = [
            str(c.get("text") or "") for c in cited
            if str(c.get("text") or "").strip()
        ]
        if not passages:
            return "no_passages"
        answer = stem + (f" Answer: {correct}" if correct is not None else "")
        try:
            report = scorer(answer, passages, split_claims=False)
        except TypeError:
            # Lenient signature for an injected fake scorer.
            report = scorer(answer, passages)
        except Exception:  # noqa: BLE001
            return "error"
        if not getattr(report, "available", True):
            return "unavailable"
        if int(getattr(report, "contradicted_count", 0) or 0) > 0:
            return "contradicted"
        rate = float(getattr(report, "groundedness_rate", 0.0) or 0.0)
        return True if rate >= 1.0 else "unsupported"

    def _trivote_bloom(self, stem: str, claimed_bloom: str) -> Tuple[str, bool]:
        """Bloom trivote clamp for an apply/analyze claim.

        Returns ``(final_bloom, was_clamped)``. A claim at understand or below
        is untouched. For a higher-order claim, the trivote runs over
        asserted + verb (+ injected zero-shot when GPU-present); when the
        asserted level is supported by ≥1 independent voter it stands, otherwise
        (disagreement OR fewer than 2 voters present — trivote unavailable) the
        item's bloom CLAMPS to its structural ceiling instead of shipping an
        unverified claim.
        """
        asserted = (claimed_bloom or "").strip().lower()
        if _BLOOM_ORDER.get(asserted, 0) <= _BLOOM_ORDER.get("understand", 1):
            return asserted or "apply", False
        try:
            from lib.classifiers.bloom_zero_shot import (
                verb_vote, trivote_disagreement,
            )
        except Exception:  # noqa: BLE001
            clamped = _clamp_bloom(asserted, _APPLY_ARM_SUBTYPE)
            return clamped, clamped != asserted
        verb = verb_vote(stem)[0]
        zshot = (
            self._bloom_zero_shot_fn(stem) if self._bloom_zero_shot_fn else None
        )
        td = trivote_disagreement(asserted, zshot, verb)
        if td.get("asserted_supported"):
            return asserted, False
        # Unavailable (skipped, <2 voters) OR disagreement → clamp to ceiling.
        clamped = _clamp_bloom(asserted, _APPLY_ARM_SUBTYPE)
        return clamped, clamped != asserted

    def _emit_apply_arm_decision(
        self,
        *,
        course_code: str,
        provider: str,
        model: Optional[str],
        drafted: int,
        shipped: int,
        dropped_sympy: int,
        dropped_grounding: int,
        dropped_other: int,
        clamped: int,
        error: Optional[str],
    ) -> None:
        """Emit one DecisionCapture event per apply-arm draft call.

        Rationale interpolates dynamic per-call signals (runtime provider/model,
        drafted/shipped counts, per-stage drop counts, bloom clamps) so a
        post-hoc audit can replay WHY each LLM-drafted apply item did / didn't
        ship — per the LLM call-site instrumentation contract.
        """
        if not self.capture:
            return
        rationale = (
            f"apply-arm draft call course={course_code} provider={provider} "
            f"model={model}: drafted={drafted}, shipped={shipped}, "
            f"dropped_sympy={dropped_sympy}, dropped_grounding={dropped_grounding}, "
            f"dropped_other={dropped_other}, bloom_clamped={clamped}; every "
            f"shipped item passed sympy(numeric)+groundedness-entailment+trivote"
            + (f"; error={error}" if error else "")
        )
        try:
            self.capture.log_decision(
                decision_type="assessment_generation",
                decision=(
                    f"apply_arm:{course_code}:shipped={shipped}/{drafted}"
                ),
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 — audit emit is best-effort
            logger.debug("apply-arm decision emit failed: %s", exc)

    def generate_for_objective(
        self,
        objective: Dict[str, Any],
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
        question_count: int = 3,
    ) -> List[QuestionData]:
        """
        Generate questions for a single learning objective.

        Args:
            objective: Learning objective data with id and text
            bloom_level: Target Bloom's level
            source_chunks: Content chunks from RAG
            question_count: Number of questions

        Returns:
            List of QuestionData
        """
        obj_id = objective.get("id", "LO-001")

        questions: List[QuestionData] = []
        skipped: List[SkippedItem] = []
        for _ in range(question_count):
            result = self._apparatus_guard(self._generate_question(
                objective_id=obj_id,
                bloom_level=bloom_level,
                source_chunks=source_chunks,
            ))
            if isinstance(result, QuestionData):
                questions.append(result)
            elif isinstance(result, SkippedItem):
                skipped.append(result)

        if self.capture:
            self.capture.log_decision(
                decision_type="objective_assessment",
                decision=f"Generated {len(questions)} questions for objective {obj_id}",
                rationale=(
                    f"Created {len(questions)} assessment items targeting the '{bloom_level}' cognitive level "
                    f"because this ensures thorough coverage of objective {obj_id}. "
                    f"Multiple questions per objective increase measurement reliability and allow "
                    f"learners multiple opportunities to demonstrate competency, reducing the impact "
                    f"of any single item on overall assessment outcomes."
                ),
                alternatives_considered=[
                    {"option": "single_question", "rejected_because": "Single item provides insufficient reliability for competency determination"},
                ],
            )

        return questions


def generate_assessment(
    course_code: str,
    objective_ids: List[str],
    bloom_levels: List[str],
    question_count: int = 10,
    capture: Optional["DecisionCapture"] = None,
) -> AssessmentData:
    """
    Convenience function to generate an assessment.

    Args:
        course_code: Course identifier
        objective_ids: Learning objectives to assess
        bloom_levels: Target Bloom's levels
        question_count: Number of questions
        capture: Optional DecisionCapture

    Returns:
        AssessmentData with generated questions
    """
    generator = AssessmentGenerator(capture=capture)
    return generator.generate(
        course_code=course_code,
        objective_ids=objective_ids,
        bloom_levels=bloom_levels,
        question_count=question_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test generation
    assessment = generate_assessment(
        course_code="TEST_101",
        objective_ids=["LO-001", "LO-002", "LO-003"],
        bloom_levels=["remember", "understand", "apply"],
        question_count=9,
    )

    print(f"Generated: {assessment.assessment_id}")
    print(f"Questions: {len(assessment.questions)}")
    print(f"Total points: {sum(q.points for q in assessment.questions)}")
    print(json.dumps(assessment.to_dict(), indent=2))
