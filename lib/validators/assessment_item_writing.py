"""Assessment-quality overhaul — deterministic Haladyna item-writing linter.

The W10 ``assessment_synthesis`` phase synthesizes learner-facing quizzes and
emits them as QTI 1.2 XML into ``<export>/06_assessments/`` — the surface an
LMS actually imports. Those items are gated for QTI well-formedness
(``qti_well_formed``), objective grounding (``assessment_objective_alignment``),
and distractor degeneracy (``synthesized_quiz_distractor``), but NOT for the
mechanical item-writing science (Haladyna–Downing–Rodriguez 2002) that
separates a fair MCQ from a leaky / cued / vacuous one. This validator closes
that gap with a deterministic, zero-LLM linter over the shipped product.

Surface decision (justified)
----------------------------
The **primary parse surface is the QTI 1.2 XML** globbed from
``inputs["qti_dir"]`` (the ``06_assessments/`` sibling), exactly like the
already-shipping ``SynthesizedQuizDistractorValidator`` and ``qti_well_formed``.
Rationale:

* The QTI XML is the ALWAYS-PRESENT, canonical, shipped-product surface — the
  whole point of the overhaul (plan §A.1) is that the LMS-imported cartridge is
  today *less* gated than the internal authored blocks. The generator's live
  ``.assessments_checkpoint.jsonl`` sidecar is DELETED after a successful run
  (``MCP/tools/pipeline_tools.py``), so it is not a reliable gate surface.
* Every mechanical rule (stem/option text, unemphasized negatives, None/All of
  the above, absolute terms, longest-option-is-key cue, option overlap, a/an
  agreement, single-key) is fully computable from the QTI item shape the
  emitter produces (``Courseforge/scripts/packaging/qti_emitter.py``): the item's
  ``cc_profile`` + ``response_lid@rcardinality`` carry the multiple-response
  discriminator, so the single-key rule's ``mc_multiple_response`` exemption
  resolves off the XML alone.

The **Bloom-honesty rule (§A.3 / work-item 4) requires per-item asserted
``bloom_level`` + ``item_subtype``, which the QTI XML does NOT carry** (the
emitter writes only ``cc_profile`` + ``cc_weighting`` into ``qtimetadata``).
So that rule runs ONLY when a structured item list is supplied via
``inputs["assessment_items"]`` (a list of ``QuestionData.to_dict()`` dicts) or
``inputs["assessment_items_path"]`` (a JSON sidecar of the same). On the
QTI-only production surface it degrades to a single info-severity
``ITEM_WRITING_NO_BLOOM_META`` note that documents the seam. **Seam to
activate in production (deferred, out of this round's scope):** either the
emitter must stamp ``cc_bloom_level`` + ``cc_item_subtype`` into the item's
``qtimetadata`` (a generator/emitter change), OR the ``assessment_synthesis``
phase handler must persist a stable ``06_assessments/assessment_items.json``
sidecar of the emitted ``QuestionData.to_dict()`` items (a phase-handler
change). Neither is done here — this validator is wired to consume that
surface the moment it lands, and the structural ceiling check is fully
unit-tested against in-memory items. Trivote (generator-claim + DeBERTa +
verb-ontology) agreement is a further deferred arm (needs a GPU); this
validator implements only the deterministic STRUCTURAL ceiling check.

Rules (each a coded, warning-severity GateIssue)
------------------------------------------------
* ``ITEM_STEM_TOO_SHORT_VS_OPTIONS`` — the stem carries the central idea only
  when it is not dwarfed by the options; flags a stem materially shorter than
  the mean option (ratio floor, ``ED4ALL_ASSESSMENT_STEM_OPTION_RATIO``).
* ``ITEM_STEM_GENERIC_NONPROBLEM`` — "Which statement is correct?"-class stems
  that pose no problem.
* ``ITEM_STEM_UNEMPHASIZED_NEGATIVE`` — a negative (``not``/``except``/
  ``never``/``neither``) in the stem that is neither UPPERCASED nor wrapped in
  emphasis markup (``<strong>``/``<em>``/``<b>``/``<u>``).
* ``ITEM_OPTION_NONE_ALL_OF_ABOVE`` — a None/All-of-the-above option.
* ``ITEM_OPTION_ABSOLUTE_TERM`` — an option built on an absolute qualifier
  (``always``/``never``) — the specious cue the generator's ``_negate_statement``
  literally inserts.
* ``ITEM_LONGEST_OPTION_IS_KEY`` — the correct option is the conspicuously
  longest (length-parity cue, ``ED4ALL_ASSESSMENT_LONGEST_MARGIN``).
* ``ITEM_OPTIONS_OVERLAP`` / ``ITEM_OPTIONS_DUPLICATE`` — near-synonymous
  (token-Jaccard floor, ``ED4ALL_ASSESSMENT_OPTION_OVERLAP``) or byte-identical
  options.
* ``ITEM_OPTION_ARTICLE_DISAGREEMENT`` — stem ends with ``a``/``an`` but an
  option's initial sound disagrees (grammatical cue).
* ``ITEM_NON_MR_MULTIPLE_KEYS`` — an auto-gradable choice item that is NOT
  multiple-response (``cc.multiple_response`` / ``rcardinality="Multiple"`` /
  ``item_subtype == "mc_multiple_response"``) must carry exactly one key.
* ``ITEM_BLOOM_CEILING_EXCEEDED`` — (structured items only) the asserted
  ``bloom_level`` exceeds the ``item_subtype``'s structural ceiling.

Severity contract
-----------------
ALL issues are **warning-severity day-1** (``passed`` is always True,
critical-count 0) with a ``# TODO(calibration)`` deferred critical-flip after a
>=2-corpus false-positive measurement (the WS3/W4 deferred-flip pattern; this
is NOT the IB3 fastest-flip exception). Item-writing rules are heuristics;
warning-first avoids blocking a real build on a debatable cue.

Input contract (``validate(inputs) -> GateResult``)
--------------------------------------------------
* ``inputs["qti_dir"]`` — directory globbed for ``*.xml`` (the wiring form).
* ``inputs["qti_paths"]`` — explicit XML file paths.
* ``inputs["qti_strings"]`` — ``[{"id","xml"}, ...]`` (or bare XML), in-memory.
* ``inputs["assessment_items"]`` — OPTIONAL list of ``QuestionData.to_dict()``
  dicts (carries ``item_subtype`` / ``bloom_level`` / per-choice
  ``is_correct``); enables the Bloom-honesty ceiling rule + richer signals.
* ``inputs["assessment_items_path"]`` — OPTIONAL JSON path resolving to either
  a list of item dicts or ``{"questions": [...]}`` / ``{"items": [...]}``.

When neither a QTI document nor a structured item resolves, the gate is a
graceful info-severity no-op (``ITEM_WRITING_NO_INPUT``) — the QTI-existence
axis is owned by ``qti_well_formed``.

References
----------
* ``lib/validators/synthesized_quiz_distractor.py`` — the QTI-surface
  distractor gate this mirrors (input contract, warning-day-1, GateResult).
* ``lib/validators/qti_well_formed.py`` — ``_iter_local`` / ``_localname`` /
  ``_extract_item_cc_profile`` QTI helpers reused here.
* ``Trainforge/generators/assessment/generator.py`` —
  ``_SUBTYPE_BLOOM_CEILING`` (the ceiling table this reads) + the
  ``item_subtype`` universe.
"""

from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.validators.qti_well_formed import (
    _extract_item_cc_profile,
    _iter_local,
    _localname,
)

logger = logging.getLogger(__name__)

# Cap the emitted issue list so a uniformly-broken quiz batch doesn't drown the
# gate report (mirrors the sibling distractor validators' cap).
_ISSUE_LIST_CAP: int = 50

_DECISION_TYPE: str = "validation_result"

# cc_profile values whose items carry a distractor/option set + require a key.
_CHOICE_PROFILES = frozenset({
    "cc.multiple_choice.v0p1",
    "cc.multiple_response.v0p1",
    "cc.true_false.v0p1",
})
_TRUE_FALSE_PROFILE = "cc.true_false.v0p1"
_MULTIPLE_RESPONSE_PROFILE = "cc.multiple_response.v0p1"
_ESSAY_PROFILE = "cc.essay.v0p1"
_AUTO_GRADABLE_PROFILES = _CHOICE_PROFILES | frozenset({"cc.fib.v0p1"})

# Question-type / item-subtype tokens that carry a WRITTEN constructed response
# (an ungraded essay-profile item with a per-item analytic rubric instead of a
# key). ``short_answer`` / ``extended_response`` are the assessment-quality
# overhaul's written types; ``essay`` is the legacy essay path.
_WRITTEN_QUESTION_TYPES = frozenset({"essay", "short_answer"})
_WRITTEN_SUBTYPES = frozenset({"short_answer", "extended_response"})

# Bloom order (low → high). Local copy so the module never needs the ontology
# at import time; mirrors ``lib/ontology/bloom.py::BLOOM_LEVELS`` ordering.
_BLOOM_ORDER: Dict[str, int] = {
    "remember": 0,
    "understand": 1,
    "apply": 2,
    "analyze": 3,
    "evaluate": 4,
    "create": 5,
}

# Fallback ceiling table — MIRRORS
# ``Trainforge.generators.assessment.generator._SUBTYPE_BLOOM_CEILING``. The
# live table is imported lazily (see ``_subtype_bloom_ceiling``); this copy is
# the last-resort default so the check works even if the generator import
# fails (offline / slim install).
_FALLBACK_SUBTYPE_CEILING: Dict[str, str] = {
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
}

# Generic "non-problem" stems that pose no actual question (Haladyna 4).
_GENERIC_STEM_RE = re.compile(
    r"which\s+(of\s+the\s+following\s+)?"
    r"(statements?|options?|answers?|choices?)\s+(is|are)\s+"
    r"(correct|true|right|accurate|false|incorrect)\b",
    re.IGNORECASE,
)
# Whole-word negatives that must be emphasized when they appear in a stem.
_NEGATIVE_RE = re.compile(r"\b(not|except|never|neither|none)\b", re.IGNORECASE)
# Emphasis markup around/near the negative (any presence in the raw stem HTML).
_EMPHASIS_TAG_RE = re.compile(r"<\s*(strong|em|b|u)\b", re.IGNORECASE)
# Bare / generic written-response stem: an open verb followed by a single
# short object and nothing else (e.g. "Discuss X.", "Describe photosynthesis").
_BARE_WRITTEN_STEM_RE = re.compile(
    r"^\s*(discuss|describe|explain|summarize|elaborate\s+on|"
    r"write\s+about|talk\s+about)\s+[\w-]+[\s.?!]*$",
    re.IGNORECASE,
)
# None/All-of-the-above option.
_NONE_ALL_OF_ABOVE_RE = re.compile(
    r"\b(none|all|both)\s+of\s+the\s+(above|following)\b", re.IGNORECASE
)
# Absolute-qualifier options (the specious cue).
_ABSOLUTE_TERM_RE = re.compile(r"\b(always|never)\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_VOWEL_SOUND_RE = re.compile(r"^[aeiou]", re.IGNORECASE)


def _strip_html(text: str) -> str:
    """Drop HTML tags + collapse whitespace to a clean text surface."""
    no_tags = _TAG_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", no_tags).strip()


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _resolve_float_env(name: str, default: float) -> float:
    """Parse-with-fallback float env resolver (garbage / non-positive → default)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    if val <= 0:
        return default
    return val


# A2 — Bloom-trivote seam (flag-gated, default OFF). The structural ceiling
# check above is ALWAYS the default. When ``ED4ALL_ASSESSMENT_ITEM_TRIVOTE`` is
# truthy this validator ADDITIONALLY runs the interpretable trivote — the
# item's asserted bloom_level vs the deterministic verb-ontology voter (+ an
# injected zero-shot voter when a GPU-present caller supplies one via
# ``inputs['zero_shot_bloom_fn']``) — and flags ``ITEM_BLOOM_TRIVOTE_UNSUPPORTED``
# (WARNING) when the generator's own Bloom claim is NOT backed by ≥1 independent
# voter. Off → the trivote arm never runs (byte-identical).
_ITEM_TRIVOTE_ENV = "ED4ALL_ASSESSMENT_ITEM_TRIVOTE"
_TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on"})


def _item_trivote_enabled() -> bool:
    """Return True when ED4ALL_ASSESSMENT_ITEM_TRIVOTE is truthy (default off)."""
    return os.environ.get(_ITEM_TRIVOTE_ENV, "").strip().lower() in _TRUTHY_TOKENS


def _subtype_bloom_ceiling() -> Dict[str, str]:
    """Return the live ``_SUBTYPE_BLOOM_CEILING`` (generator) or the fallback.

    Lazy import so ``lib`` never hard-depends on ``Trainforge`` at module load;
    offline-safe (the generator import is pure-Python + fast).
    """
    try:
        from Trainforge.generators.assessment.generator import (  # noqa: WPS433
            _SUBTYPE_BLOOM_CEILING,
        )
        if isinstance(_SUBTYPE_BLOOM_CEILING, dict) and _SUBTYPE_BLOOM_CEILING:
            return dict(_SUBTYPE_BLOOM_CEILING)
    except Exception as exc:  # noqa: BLE001 — degrade to the local mirror
        logger.debug(
            "assessment_item_writing: generator ceiling import failed (%s); "
            "using local fallback table.", exc,
        )
    return dict(_FALLBACK_SUBTYPE_CEILING)


# --------------------------------------------------------------------------- #
# Normalized item shape — one dict per emitted question, surfaced from EITHER
# the QTI XML or a structured item list. Keys:
#   id, cc_profile, item_subtype, bloom_level, question_type, stem_html,
#   stem_text, is_choice, is_multiple_response, options[{id, text, is_correct}],
#   correct_count.
# --------------------------------------------------------------------------- #
def _first_material_text(parent: ET.Element) -> str:
    """Return the text of ``parent``'s DIRECT ``material/mattext`` (stem)."""
    for child in list(parent):
        if _localname(child.tag) == "material":
            for mat in child.iter():
                if _localname(mat.tag) == "mattext":
                    return mat.text or ""
    return ""


def _item_from_xml(item: ET.Element) -> Dict[str, Any]:
    """Normalize one QTI ``<item>`` element into the shared item dict."""
    item_id = item.get("ident") or "<no-ident>"
    cc_profile = _extract_item_cc_profile(item) or ""

    presentation = next(_iter_local(item, "presentation"), None)
    response_lid = next(_iter_local(item, "response_lid"), None)
    rcardinality = (
        (response_lid.get("rcardinality") or "").strip()
        if response_lid is not None
        else ""
    )

    # Stem: the presentation's DIRECT material/mattext (NOT an option's).
    stem_html = _first_material_text(presentation) if presentation is not None else ""

    # Options: response_lid → render_choice → response_label → material/mattext.
    options: List[Dict[str, Any]] = []
    if response_lid is not None:
        for label in _iter_local(response_lid, "response_label"):
            oid = label.get("ident") or ""
            otext = _first_material_text(label)
            options.append({"id": oid, "text": otext, "is_correct": False})

    # Correct keys: every varequal token under resprocessing.
    correct_ids: List[str] = []
    resprocessing = next(_iter_local(item, "resprocessing"), None)
    if resprocessing is not None:
        for tok in _iter_local(resprocessing, "varequal"):
            if tok.text is not None and tok.text.strip():
                correct_ids.append(tok.text.strip())
    correct_set = set(correct_ids)
    for opt in options:
        if opt["id"] and opt["id"] in correct_set:
            opt["is_correct"] = True

    is_choice = cc_profile in _CHOICE_PROFILES or response_lid is not None
    is_mr = (
        cc_profile == _MULTIPLE_RESPONSE_PROFILE
        or rcardinality.lower() == "multiple"
    )
    is_tf = cc_profile == _TRUE_FALSE_PROFILE
    # correct_count: prefer the marked options; fall back to the raw key count.
    marked = sum(1 for o in options if o["is_correct"])
    correct_count = marked if marked else len(correct_set)
    return {
        "id": str(item_id),
        "cc_profile": cc_profile,
        "item_subtype": None,
        "bloom_level": None,
        "question_type": None,
        "stem_html": stem_html,
        "stem_text": _strip_html(stem_html),
        "is_choice": is_choice,
        "is_auto_gradable": cc_profile in _AUTO_GRADABLE_PROFILES,
        "is_multiple_response": is_mr,
        "is_true_false": is_tf,
        "is_written": cc_profile == _ESSAY_PROFILE,
        "is_structured": False,
        "rubric": None,
        "source_chunks": [],
        "options": options,
        "correct_count": correct_count,
    }


def _item_from_dict(q: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one ``QuestionData.to_dict()`` dict into the shared item dict."""
    qtype = str(q.get("question_type") or "").strip().lower()
    subtype = q.get("item_subtype")
    subtype = str(subtype).strip() if isinstance(subtype, str) else None
    stem_html = str(q.get("stem") or "")
    raw_choices = q.get("choices") or []
    options: List[Dict[str, Any]] = []
    for i, ch in enumerate(raw_choices):
        if isinstance(ch, dict):
            oid = str(ch.get("id") or ch.get("ident") or ch.get("label") or f"c{i}")
            otext = ch.get("text")
            if otext is None:
                otext = ch.get("content", ch.get("value", ""))
            is_correct = bool(ch.get("is_correct"))
        else:
            oid = f"c{i}"
            otext = str(ch)
            is_correct = False
        options.append({"id": oid, "text": str(otext), "is_correct": is_correct})

    # correct_count: marked choices ∪ plural correct_answers ∪ scalar answer.
    marked = sum(1 for o in options if o["is_correct"])
    plural = q.get("correct_answers") or []
    if isinstance(plural, list) and plural:
        correct_count = max(marked, len(plural))
    elif marked:
        correct_count = marked
    elif q.get("correct_answer") not in (None, ""):
        correct_count = 1
    else:
        correct_count = 0

    is_mr = qtype == "multiple_response" or subtype == "mc_multiple_response"
    is_tf = qtype == "true_false" or subtype == "tf"
    is_choice = bool(options) and qtype in {
        "multiple_choice", "multiple_response", "true_false",
    }
    is_auto = qtype in {
        "multiple_choice", "multiple_response", "true_false", "fill_in_blank",
    }
    return {
        "id": str(q.get("question_id") or "<no-id>"),
        "cc_profile": "",
        "item_subtype": subtype,
        "bloom_level": (str(q.get("bloom_level")).strip().lower()
                        if q.get("bloom_level") else None),
        "question_type": qtype,
        "stem_html": stem_html,
        "stem_text": _strip_html(stem_html),
        "is_choice": is_choice,
        "is_auto_gradable": is_auto,
        "is_multiple_response": is_mr,
        "is_true_false": is_tf,
        "is_written": (
            qtype in _WRITTEN_QUESTION_TYPES
            or (subtype in _WRITTEN_SUBTYPES if subtype else False)
        ),
        "is_structured": True,
        "rubric": q.get("rubric") if isinstance(q.get("rubric"), dict) else None,
        "source_chunks": [
            str(c) for c in (q.get("source_chunks") or []) if c
        ],
        "options": options,
        "correct_count": correct_count,
    }


class AssessmentItemWritingValidator:
    """Deterministic Haladyna item-writing linter for the W10 QTI surface."""

    name = "assessment_item_writing"
    version = "0.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")
        issues: List[GateIssue] = []

        items, structured, input_issues = self._collect_items(inputs)
        issues.extend(input_issues)

        if not items:
            if not input_issues:
                issues.append(
                    GateIssue(
                        severity="info",
                        code="ITEM_WRITING_NO_INPUT",
                        message=(
                            "AssessmentItemWritingValidator found no items "
                            "(inputs['qti_dir'] / ['qti_paths'] / "
                            "['qti_strings'] / ['assessment_items']); linter "
                            "skipped."
                        ),
                    )
                )
            return self._result(gate_id, issues, 0, 0, capture)

        # Bloom-honesty seam note: only structured items carry the metadata.
        if not structured:
            issues.append(
                GateIssue(
                    severity="info",
                    code="ITEM_WRITING_NO_BLOOM_META",
                    message=(
                        "Bloom-honesty ceiling check skipped: the QTI surface "
                        "carries no per-item bloom_level / item_subtype "
                        "metadata. Supply inputs['assessment_items'] (or "
                        "persist 06_assessments/assessment_items.json) to "
                        "activate ITEM_BLOOM_CEILING_EXCEEDED."
                    ),
                )
            )

        ratio_floor = _resolve_float_env(
            "ED4ALL_ASSESSMENT_STEM_OPTION_RATIO", 0.5
        )
        overlap_floor = _resolve_float_env(
            "ED4ALL_ASSESSMENT_OPTION_OVERLAP", 0.80
        )
        longest_margin = _resolve_float_env(
            "ED4ALL_ASSESSMENT_LONGEST_MARGIN", 1.5
        )
        ceiling = _subtype_bloom_ceiling()

        # A2 — Bloom-trivote seam (flag-gated). The zero-shot voter is GPU-gated,
        # so it is supplied by the caller via inputs (default None → the trivote
        # runs on the deterministic asserted + verb voters only, fully offline).
        trivote_on = _item_trivote_enabled()
        zero_shot_fn = (
            inputs.get("zero_shot_bloom_fn") if trivote_on else None
        )

        items_audited = 0
        items_flagged = 0
        for item in items:
            items_audited += 1
            before = len(issues)
            self._audit_item(
                item,
                issues,
                ratio_floor=ratio_floor,
                overlap_floor=overlap_floor,
                longest_margin=longest_margin,
                ceiling=ceiling,
                trivote_on=trivote_on,
                zero_shot_fn=zero_shot_fn,
            )
            if len(issues) > before:
                items_flagged += 1

        return self._result(gate_id, issues, items_audited, items_flagged, capture)

    # ------------------------------------------------------------- per-item

    def _audit_item(
        self,
        item: Dict[str, Any],
        issues: List[GateIssue],
        *,
        ratio_floor: float,
        overlap_floor: float,
        longest_margin: float,
        ceiling: Dict[str, str],
        trivote_on: bool = False,
        zero_shot_fn: Optional[Any] = None,
    ) -> None:
        loc = item["id"]

        def _add(severity: str, code: str, message: str, suggestion: str = "") -> None:
            if len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity=severity,
                        code=code,
                        message=message,
                        location=loc,
                        suggestion=suggestion or None,
                    )
                )

        stem_text = item["stem_text"]
        stem_html = item["stem_html"]
        options = item["options"]
        opt_texts = [_strip_html(o["text"]) for o in options]
        opt_texts = [t for t in opt_texts if t]

        # --- Bloom-honesty ceiling (structured items only). ---------------
        subtype = item.get("item_subtype")
        bloom = item.get("bloom_level")
        if subtype and bloom:
            cap = ceiling.get(subtype)
            cur_o = _BLOOM_ORDER.get(bloom)
            cap_o = _BLOOM_ORDER.get(cap) if cap else None
            if cap_o is not None and cur_o is not None and cur_o > cap_o:
                _add(
                    "warning",
                    "ITEM_BLOOM_CEILING_EXCEEDED",
                    (
                        f"item {loc!r} asserts bloom_level={bloom!r} but its "
                        f"item_subtype={subtype!r} structurally supports at "
                        f"most {cap!r}; the Bloom claim is not backed by the "
                        f"item's structure."
                    ),
                    suggestion=(
                        "Relabel the item to its structural ceiling, or emit a "
                        "type that genuinely assesses the claimed level "
                        "(numeric-solve / ordering / error-analysis)."
                    ),
                )

        # --- A2 Bloom-trivote arm (flag-gated). ---------------------------
        # Independent of the structural ceiling above: does the generator's OWN
        # Bloom claim survive the interpretable trivote? Runs only when
        # ED4ALL_ASSESSMENT_ITEM_TRIVOTE is on AND the item carries an asserted
        # bloom_level. The verb voter is deterministic + offline; the zero-shot
        # voter is supplied by a GPU-present caller (else the trivote runs on
        # asserted + verb, and a single present voter structured-skips —
        # never fabricating a verdict).
        if trivote_on and bloom:
            self._audit_bloom_trivote(
                loc=loc,
                stem_text=item.get("stem_text") or "",
                asserted=bloom,
                zero_shot_fn=zero_shot_fn,
                _add=_add,
            )

        # --- Single-key rule (all auto-gradable items). -------------------
        if item.get("is_auto_gradable") and item.get("is_choice"):
            if not item["is_multiple_response"]:
                if item["correct_count"] != 1:
                    _add(
                        "warning",
                        "ITEM_NON_MR_MULTIPLE_KEYS",
                        (
                            f"item {loc!r} is a single-answer item but carries "
                            f"{item['correct_count']} correct key(s); a "
                            f"non-multiple-response item must have exactly 1 "
                            f"key."
                        ),
                        suggestion=(
                            "Mark exactly one option correct, or emit the item "
                            "as multiple_response (cc.multiple_response)."
                        ),
                    )

        # --- Written-response (essay / short_answer / extended_response). --
        if item.get("is_written"):
            self._audit_written(item, _add)

        # Text rules below apply to CHOICE items with real options only.
        if not item.get("is_choice") or not opt_texts:
            self._audit_stem_negative(stem_text, stem_html, _add)
            return

        # --- Stem: negative-word emphasis. --------------------------------
        self._audit_stem_negative(stem_text, stem_html, _add)

        # --- Stem: generic non-problem. -----------------------------------
        if _GENERIC_STEM_RE.search(stem_text):
            _add(
                "warning",
                "ITEM_STEM_GENERIC_NONPROBLEM",
                (
                    f"item {loc!r} stem is a generic non-problem "
                    f"({stem_text[:60]!r}); the stem should pose the actual "
                    f"question, not ask which option is correct."
                ),
                suggestion="Rewrite the stem to carry the central idea.",
            )

        # --- Stem: too short vs options (stem carries the central idea). --
        # Skip true/false (options 'True'/'False' are trivially short).
        if not item["is_true_false"] and len(opt_texts) >= 3:
            mean_opt = sum(len(t) for t in opt_texts) / len(opt_texts)
            if mean_opt > 0 and len(stem_text) < ratio_floor * mean_opt:
                _add(
                    "warning",
                    "ITEM_STEM_TOO_SHORT_VS_OPTIONS",
                    (
                        f"item {loc!r} stem ({len(stem_text)} chars) is much "
                        f"shorter than the mean option ({mean_opt:.0f} chars); "
                        f"the stem should carry the central idea, not the "
                        f"options."
                    ),
                    suggestion="Move the shared option content into the stem.",
                )

        # --- Options: None/All-of-the-above + absolute terms. -------------
        for otext in opt_texts:
            if _NONE_ALL_OF_ABOVE_RE.search(otext):
                _add(
                    "warning",
                    "ITEM_OPTION_NONE_ALL_OF_ABOVE",
                    (
                        f"item {loc!r} carries a None/All-of-the-above option "
                        f"({otext[:50]!r}); these degrade discrimination."
                    ),
                    suggestion="Replace with a substantive distractor.",
                )
                break
        for otext in opt_texts:
            if _ABSOLUTE_TERM_RE.search(otext):
                _add(
                    "warning",
                    "ITEM_OPTION_ABSOLUTE_TERM",
                    (
                        f"item {loc!r} carries an absolute-qualifier option "
                        f"({otext[:50]!r}); 'always'/'never' options are a "
                        f"specious test-taking cue."
                    ),
                    suggestion="Rephrase the option without the absolute term.",
                )
                break

        # --- Longest-option-is-key length cue. ----------------------------
        self._audit_longest_key(item, options, longest_margin, _add)

        # --- Duplicate / overlapping options. -----------------------------
        self._audit_option_overlap(opt_texts, overlap_floor, _add)

        # --- Article (a/an) agreement. ------------------------------------
        self._audit_article_agreement(stem_text, opt_texts, _add)

    @staticmethod
    def _audit_bloom_trivote(
        *,
        loc: str,
        stem_text: str,
        asserted: str,
        zero_shot_fn: Optional[Any],
        _add,
    ) -> None:
        """A2 — flag-gated Bloom-trivote arm.

        Consumes the interpretable trivote voters from
        ``lib.classifiers.bloom_zero_shot``: the generator's OWN asserted level,
        the deterministic verb-ontology voter over the stem, and — when a
        GPU-present caller injects one — a zero-shot voter. Flags
        ``ITEM_BLOOM_TRIVOTE_UNSUPPORTED`` (WARNING) only when the asserted level
        is present, ≥1 independent voter is present, and NONE of the independent
        voters agree (the ``asserted_unsupported`` signal). A single-present-
        voter case structured-skips (never a fabricated verdict), and the
        structural ceiling check remains the always-on default.
        """
        try:
            from lib.classifiers.bloom_zero_shot import (
                verb_vote, trivote_disagreement,
            )
        except Exception as exc:  # noqa: BLE001 — offline / slim install
            logger.debug(
                "assessment_item_writing: trivote import failed (%s); skipped.",
                exc,
            )
            return
        verb = verb_vote(stem_text)[0]
        zshot = None
        if zero_shot_fn is not None:
            try:
                zshot = zero_shot_fn(stem_text)
            except Exception as exc:  # noqa: BLE001 — GPU voter best-effort
                logger.debug(
                    "assessment_item_writing: zero-shot voter raised (%s); "
                    "trivote runs on asserted + verb only.", exc,
                )
        td = trivote_disagreement(asserted, zshot, verb)
        if td.get("asserted_unsupported"):
            independent = td.get("independent_levels") or []
            _add(
                "warning",
                "ITEM_BLOOM_TRIVOTE_UNSUPPORTED",
                (
                    f"item {loc!r} asserts bloom_level={asserted!r} but the "
                    f"independent Bloom voters {independent!r} do not agree "
                    f"(trivote disagreement_score="
                    f"{td.get('disagreement_score')}); the generator's own "
                    f"Bloom claim is not backed by an interpretable check."
                ),
                suggestion=(
                    "Relabel the item to the majority independent level "
                    f"({td.get('majority_level')!r}), or rewrite the stem so its "
                    "verb genuinely signals the claimed level."
                ),
            )

    @staticmethod
    def _audit_written(item: Dict[str, Any], _add) -> None:
        """Written-response item-writing rules (essay / short_answer /
        extended_response).

        * ``ITEM_WRITTEN_STEM_NOT_SPECIFIC`` — the prompt is a bare open verb +
          single object ("Discuss X.") or falls below a token floor
          (``ED4ALL_ASSESSMENT_WRITTEN_MIN_TOKENS``, default 6). A written item
          must pose a SPECIFIC task, not a one-word topic dump. Runs on both the
          QTI and structured surfaces.
        * ``ITEM_WRITTEN_RUBRIC_MISSING`` — a written item must carry a per-item
          analytic rubric (structured surface only — the QTI XML carries none).
        * ``ITEM_WRITTEN_RUBRIC_CRITERION_UNGROUNDED`` — every rubric criterion
          must resolve at least one source chunk id, and each cited id must be in
          the item's ``source_chunks`` (grounded-by-construction contract).
        """
        loc = item["id"]
        stem_text = item["stem_text"]
        min_tokens = int(_resolve_float_env(
            "ED4ALL_ASSESSMENT_WRITTEN_MIN_TOKENS", 6.0))
        token_count = len(_tokens(stem_text))
        if _BARE_WRITTEN_STEM_RE.match(stem_text) or token_count < min_tokens:
            _add(
                "warning",
                "ITEM_WRITTEN_STEM_NOT_SPECIFIC",
                (
                    f"written item {loc!r} prompt is not specific "
                    f"({stem_text[:60]!r}, {token_count} tokens); a "
                    f"constructed-response prompt must pose a specific task, "
                    f"not a bare topic (e.g. 'Discuss X')."
                ),
                suggestion=(
                    "Ground the prompt in a specific claim / procedure / term "
                    "pair (e.g. 'Explain why <claim>' or 'Compare X and Y')."
                ),
            )

        # Rubric rules — structured surface only (QTI carries no rubric).
        if not item.get("is_structured"):
            return
        rubric = item.get("rubric")
        criteria = (rubric or {}).get("criteria") if isinstance(rubric, dict) else None
        if not criteria:
            _add(
                "warning",
                "ITEM_WRITTEN_RUBRIC_MISSING",
                (
                    f"written item {loc!r} carries no per-item analytic rubric; "
                    f"a constructed-response item must ship a rubric so it can "
                    f"be graded consistently."
                ),
                suggestion=(
                    "Attach a rubric with criteria (each citing a source chunk) "
                    "and score levels."
                ),
            )
            return
        source_ids = set(item.get("source_chunks") or [])
        for row in criteria:
            if not isinstance(row, dict):
                continue
            cites = [str(c) for c in (row.get("cites") or []) if c]
            grounded = bool(cites) and (
                not source_ids or all(c in source_ids for c in cites)
            )
            if not grounded:
                crit = str(row.get("criterion") or "")[:60]
                _add(
                    "warning",
                    "ITEM_WRITTEN_RUBRIC_CRITERION_UNGROUNDED",
                    (
                        f"written item {loc!r} rubric criterion {crit!r} does "
                        f"not resolve a source chunk id (cites={cites!r}); every "
                        f"criterion must cite a chunk the item is grounded in."
                    ),
                    suggestion=(
                        "Derive each criterion from the item's source chunks and "
                        "cite the chunk id it came from."
                    ),
                )

    @staticmethod
    def _audit_stem_negative(stem_text: str, stem_html: str, _add) -> None:
        m = _NEGATIVE_RE.search(stem_text)
        if not m:
            return
        word = m.group(1)
        # Emphasized when the matched word is UPPERCASE in the stem OR the raw
        # stem HTML carries an emphasis tag.
        uppercased = word.upper() in stem_text
        has_emphasis = bool(_EMPHASIS_TAG_RE.search(stem_html))
        if uppercased or has_emphasis:
            return
        _add(
            "warning",
            "ITEM_STEM_UNEMPHASIZED_NEGATIVE",
            (
                f"stem contains the negative {word!r} without emphasis; "
                f"unemphasized negatives are easily missed."
            ),
            suggestion=(
                "Uppercase the negative (NOT/EXCEPT) or wrap it in "
                "<strong>/<em>."
            ),
        )

    @staticmethod
    def _audit_longest_key(
        item: Dict[str, Any], options: List[Dict[str, Any]],
        longest_margin: float, _add,
    ) -> None:
        # Only meaningful for a single-key item with >=3 options.
        keys = [o for o in options if o["is_correct"]]
        if len(keys) != 1 or len(options) < 3:
            return
        lens = sorted(
            ((len(_strip_html(o["text"])), o["is_correct"]) for o in options),
            key=lambda t: t[0],
            reverse=True,
        )
        longest_len, longest_is_key = lens[0]
        second_len = lens[1][0]
        if not longest_is_key or second_len <= 0:
            return
        if longest_len >= longest_margin * second_len:
            _add(
                "warning",
                "ITEM_LONGEST_OPTION_IS_KEY",
                (
                    f"item {item['id']!r} has its correct option as the "
                    f"conspicuously longest ({longest_len} vs {second_len} "
                    f"chars); option length is a specious cue to the key."
                ),
                suggestion=(
                    "Balance option lengths — trim the key or lengthen the "
                    "distractors."
                ),
            )

    @staticmethod
    def _audit_option_overlap(
        opt_texts: List[str], overlap_floor: float, _add,
    ) -> None:
        n = len(opt_texts)
        reported_dup = False
        reported_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = opt_texts[i], opt_texts[j]
                if a.strip().lower() == b.strip().lower():
                    if not reported_dup:
                        reported_dup = True
                        _add(
                            "warning",
                            "ITEM_OPTIONS_DUPLICATE",
                            (
                                f"two options are identical ({a[:50]!r}); "
                                f"options must be distinct."
                            ),
                            suggestion="Deduplicate the option set.",
                        )
                    continue
                if not reported_overlap and _jaccard(a, b) >= overlap_floor:
                    reported_overlap = True
                    _add(
                        "warning",
                        "ITEM_OPTIONS_OVERLAP",
                        (
                            f"two options overlap heavily "
                            f"({a[:40]!r} / {b[:40]!r}); near-synonymous "
                            f"options blur the discrimination."
                        ),
                        suggestion="Make the distractors semantically distinct.",
                    )

    @staticmethod
    def _audit_article_agreement(
        stem_text: str, opt_texts: List[str], _add,
    ) -> None:
        m = re.search(r"\b(a|an)\s*$", stem_text.strip(), re.IGNORECASE)
        if not m:
            return
        article = m.group(1).lower()
        for otext in opt_texts:
            first = _tokens(otext)
            if not first:
                continue
            starts_vowel = bool(_VOWEL_SOUND_RE.match(first[0]))
            # 'an' expects a vowel sound; 'a' expects a consonant sound.
            if (article == "an") != starts_vowel:
                _add(
                    "warning",
                    "ITEM_OPTION_ARTICLE_DISAGREEMENT",
                    (
                        f"stem ends with the article {article!r} but option "
                        f"{otext[:40]!r} disagrees in a/an; grammatical "
                        f"agreement cues the key."
                    ),
                    suggestion=(
                        "Reword the stem to end before the article, e.g. "
                        "'... is a(n):'."
                    ),
                )
                return

    # ------------------------------------------------------------- inputs

    def _collect_items(
        self, inputs: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], bool, List[GateIssue]]:
        """Resolve normalized items from the structured or QTI surface.

        Returns ``(items, structured, issues)`` where ``structured`` is True
        when the items came from the metadata-rich ``assessment_items`` surface
        (so the Bloom-honesty rule can run).
        """
        issues: List[GateIssue] = []

        # (1) Structured items win (metadata-rich).
        struct = inputs.get("assessment_items")
        if struct is None and inputs.get("assessment_items_path"):
            struct, load_issues = self._load_items_path(
                inputs["assessment_items_path"]
            )
            issues.extend(load_issues)
        if isinstance(struct, list) and struct:
            items = [
                _item_from_dict(q) for q in struct if isinstance(q, dict)
            ]
            if items:
                return items, True, issues

        # (2) QTI XML documents.
        documents, doc_issues = self._collect_documents(inputs)
        issues.extend(doc_issues)
        items: List[Dict[str, Any]] = []
        for label, xml in documents:
            try:
                root = ET.fromstring(xml)
            except ET.ParseError:
                # Parse errors are owned by qti_well_formed; skip, don't
                # double-fail.
                continue
            for elem in _iter_local(root, "item"):
                norm = _item_from_xml(elem)
                norm["id"] = f"{label}#{norm['id']}"
                items.append(norm)
        return items, False, issues

    @staticmethod
    def _load_items_path(
        path_val: Any,
    ) -> Tuple[Optional[List[Dict[str, Any]]], List[GateIssue]]:
        issues: List[GateIssue] = []
        try:
            path = Path(str(path_val))
            if not path.exists():
                return None, issues
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="ITEM_WRITING_UNREADABLE_ITEMS",
                    message=(
                        f"Could not read assessment_items_path {path_val}: "
                        f"{exc}"
                    ),
                    location=str(path_val),
                )
            )
            return None, issues
        if isinstance(data, list):
            return data, issues
        if isinstance(data, dict):
            for key in ("items", "questions"):
                val = data.get(key)
                if isinstance(val, list):
                    return val, issues
        return None, issues

    @staticmethod
    def _collect_documents(
        inputs: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, str]], List[GateIssue]]:
        """Normalise the three QTI input forms into ``[(label, xml_str), ...]``.

        Mirrors ``SynthesizedQuizDistractorValidator._collect_documents`` — never
        emits a critical missing-input issue (the QTI-existence axis is owned by
        ``qti_well_formed``).
        """
        documents: List[Tuple[str, str]] = []
        issues: List[GateIssue] = []

        raw_strings = inputs.get("qti_strings")
        if raw_strings:
            for idx, entry in enumerate(raw_strings):
                if isinstance(entry, dict):
                    lbl = str(entry.get("id", f"qti_strings[{idx}]"))
                    xml = entry.get("xml", "")
                else:
                    lbl = f"qti_strings[{idx}]"
                    xml = entry
                documents.append((lbl, xml if isinstance(xml, str) else ""))

        path_candidates: List[Path] = []
        raw_paths = inputs.get("qti_paths")
        if raw_paths:
            path_candidates.extend(Path(p) for p in raw_paths)

        raw_dir = inputs.get("qti_dir")
        if raw_dir:
            qti_dir = Path(raw_dir)
            if qti_dir.exists() and qti_dir.is_dir():
                path_candidates.extend(sorted(qti_dir.rglob("*.xml")))

        for path in path_candidates:
            try:
                documents.append((str(path), path.read_text(encoding="utf-8")))
            except OSError as exc:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="ITEM_WRITING_UNREADABLE_FILE",
                        message=f"Could not read QTI file {path}: {exc}",
                        location=str(path),
                    )
                )

        documents = [(lbl, xml) for lbl, xml in documents if xml and xml.strip()]
        return documents, issues

    # ------------------------------------------------------------- result

    def _result(
        self,
        gate_id: str,
        issues: List[GateIssue],
        items_audited: int,
        items_flagged: int,
        capture: Any,
    ) -> GateResult:
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0  # warning-day-1 → always True
        warn_count = sum(1 for i in issues if i.severity == "warning")
        score = (
            1.0
            if items_audited == 0
            else round(1.0 - (items_flagged / items_audited), 4)
        )
        self._emit_decision(
            capture,
            items_audited=items_audited,
            items_flagged=items_flagged,
            warn_count=warn_count,
            passed=passed,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        items_audited: int,
        items_flagged: int,
        warn_count: int,
        passed: bool,
    ) -> None:
        if capture is None:
            return
        rationale = (
            f"Haladyna item-writing linter: {items_audited} item(s) audited, "
            f"{items_flagged} flagged with {warn_count} warning-severity "
            f"item-writing issue(s) (stem central-idea / unemphasized "
            f"negative / None-All-of-above / absolute term / longest-key cue / "
            f"option overlap / a-an agreement / single-key / bloom-ceiling). "
            f"All warning-severity day-1 (deferred critical-flip); "
            f"passed={passed}."
        )
        try:
            capture.log_decision(
                decision_type=_DECISION_TYPE,
                decision=(
                    f"assessment_item_writing:{items_flagged}/"
                    f"{items_audited}_flagged"
                ),
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 — capture must not break the gate
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "assessment_item_writing: %s", exc,
            )


__all__ = ["AssessmentItemWritingValidator"]
