"""W10 Phase 1 — QTI 1.2 / discussion (imsdt) / assignment emitter.

A PURE DETERMINISTIC transform from the Trainforge assessment shapes
(``AssessmentData`` / ``QuestionData`` or their ``to_dict()`` JSON) to the
canonical IMS Common Cartridge resource XML:

* QTI 1.2 assessment XML (``questestinterop`` → ``assessment`` → ``section`` →
  ``item``), matching the Brightspace-tested Pattern-15 shape in
  ``Courseforge/docs/guides/troubleshooting.md`` and round-tripping the in-tree
  ``Trainforge/parsers/qti_parser.py::QTIParser``.
* IMS discussion-topic XML (``imsdt_v1p3``).
* Assignment learning-application-resource XML (``cc_extensions/assignment``).

NO LLM. NO packaging. NO UI. The emitter never invents content — it only
reshapes the generator's output (anti-fabrication contract). Where the
generator's output cannot satisfy an auto-gradable QTI shape (e.g. a numeric
fill-in whose ``correct_answer`` is not a canonical number), the item is
*downgraded* to a still-valid shape (per spec §9.1) rather than fabricating a
key.

Public API (per spec §3):

* ``question_to_qti_item(q: dict) -> ET.Element`` — one ``<item>``.
* ``assessment_to_qti(a: dict) -> str`` — one ``<questestinterop>`` document.
* ``discussion_to_imsdt(prompt: dict) -> str`` — one ``<topic>`` document.
* ``assignment_to_resource(task: dict) -> str`` — one ``<assignment>`` document.

The inputs accept either the dataclass instances (anything exposing
``to_dict()``) or the plain dict form, so callers can hand the emitter the
``AssessmentData.to_dict()`` JSON read across a phase boundary.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

__all__ = [
    "question_to_qti_item",
    "assessment_to_qti",
    "assessment_to_objectbank",
    "BANK_METADATA_FIELDS",
    "discussion_to_imsdt",
    "assignment_to_resource",
    "resolve_cc_profile",
    "itemfeedback_enabled",
    "QTI_NS",
    "IMSDT_NS",
    "ASSIGNMENT_NS",
    "CC_EXAM_PROFILE",
    "SUPPORTED_ITEM_PROFILES",
]

# ── Namespaces (match the vendored XSD targetNamespaces) ────────────────────
QTI_NS = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
IMSDT_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"
ASSIGNMENT_NS = "http://www.imsglobal.org/xsd/imscc_extensions/assignment"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

# ── cc_profile mapping (spec §2.2) ──────────────────────────────────────────
CC_EXAM_PROFILE = "cc.exam.v0p1"

_CC_PROFILE: Dict[str, str] = {
    "multiple_choice": "cc.multiple_choice.v0p1",
    "multiple_response": "cc.multiple_response.v0p1",
    "true_false": "cc.true_false.v0p1",
    "fill_in_blank": "cc.fib.v0p1",
    # ``pattern_match`` is the sixth CC QTI 1.2 profile (optional). It is the
    # effective type a ``fill_in_blank`` resolves to when it declares MULTIPLE
    # accepted answer strings (``correct_answers``) — a single response_str /
    # render_fib whose resprocessing ORs one varequal per accepted string. It
    # is never a generator ``question_type``; only ``_resolved_item_type``
    # produces it, so it has no place in the question_type→profile table beyond
    # this self-map used by ``resolve_cc_profile(effective_type)``.
    "pattern_match": "cc.pattern_match.v0p1",
    "short_answer": "cc.essay.v0p1",
    "essay": "cc.essay.v0p1",
}

# The full set of profiles the emitter can produce (the §7 qti_well_formed gate
# audits item profiles against this set).
SUPPORTED_ITEM_PROFILES = frozenset(_CC_PROFILE.values())

# Auto-gradable types get a resolvable answer key. ``short_answer``/``essay``
# are emitted but ungraded (no respcondition).
_AUTOGRADABLE = frozenset({
    "multiple_choice", "multiple_response", "true_false",
    "fill_in_blank", "pattern_match",
})

_HTML_TEXTTYPE = "text/html"

# ── itemfeedback gate (assessment-quality overhaul) ─────────────────────────
# Per-distractor <itemfeedback> (misconception note) + one Solution-type
# <itemfeedback> (worked solution) are DEFAULT ON — this recovers content the
# generator already authors (``feedback`` + per-choice ``misconception_note``)
# and is spec-legal QTI 1.2 (``itemfeedback`` is a per-item child, not a
# profile, so ``qti_well_formed`` passes unchanged). Kill-switch:
# ``ED4ALL_ASSESSMENT_ITEMFEEDBACK=0`` (parse-with-fallback: explicit falsey
# ``0``/``false``/``no``/``off`` → off; unset / garbage / truthy → on), so an
# off value emits byte-identically to the pre-feature output.
_ITEMFEEDBACK_ENV = "ED4ALL_ASSESSMENT_ITEMFEEDBACK"
_FALSEY = frozenset({"0", "false", "no", "off"})


def itemfeedback_enabled() -> bool:
    """Resolve the ``ED4ALL_ASSESSMENT_ITEMFEEDBACK`` gate (default ON).

    Read each call so tests can ``setenv``-toggle it (mirrors the default-ON
    ``COURSEFORGE_PAGE_MATHJAX`` parse contract).
    """
    raw = os.environ.get(_ITEMFEEDBACK_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSEY

# Numeric canonical-form check for the fill_in_blank emit/downgrade decision
# (spec §9.1): an integer or simple decimal literal, optionally signed.
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\.\d+|\d+\.?|\d+)$")


def resolve_cc_profile(question_type: str) -> str:
    """Map a generator ``question_type`` to its QTI ``cc_profile`` string.

    Unknown / unsupported types collapse to the essay profile (emitted but
    ungraded) rather than raising — the emitter never drops an item silently.
    """
    return _CC_PROFILE.get((question_type or "").strip().lower(), "cc.essay.v0p1")


# ── helpers ─────────────────────────────────────────────────────────────────
def _as_dict(obj: Any) -> Dict[str, Any]:
    """Accept a dataclass (``to_dict()``) or a plain dict."""
    if isinstance(obj, dict):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    raise TypeError(f"expected dict or object with to_dict(), got {type(obj)!r}")


def _ncname(raw: str, *, fallback: str) -> str:
    """Coerce an arbitrary string into an XML NCName (xs:ID-safe).

    ``ident`` attributes on ``assessment`` / ``item`` / ``response_lid`` /
    ``section`` are ``xs:ID`` in the XSD — they must be valid NCNames and unique
    within the document. We replace illegal chars with ``_`` and ensure a
    letter/underscore lead char.
    """
    text = (raw or "").strip()
    if not text:
        text = fallback
    # Replace any char not valid in an NCName body with underscore.
    text = re.sub(r"[^0-9A-Za-z_.\-]", "_", text)
    # NCName cannot start with a digit, '-' or '.'.
    if not text or not (text[0].isalpha() or text[0] == "_"):
        text = f"_{text}"
    return text


def _choice_ident(raw: str, index: int) -> str:
    """Stable per-choice response_label ident (xs:string — laxer than ID)."""
    text = (raw or "").strip()
    if text:
        return text
    return chr(ord("A") + index) if index < 26 else f"opt_{index}"


def _material(parent: ET.Element, text: str) -> ET.Element:
    """Append ``<material><mattext texttype="text/html">…</mattext></material>``."""
    material = ET.SubElement(parent, "material")
    mattext = ET.SubElement(material, "mattext")
    mattext.set("texttype", _HTML_TEXTTYPE)
    mattext.text = text or ""
    return material


def _qtimetadata_field(parent: ET.Element, label: str, entry: str) -> None:
    field = ET.SubElement(parent, "qtimetadatafield")
    ET.SubElement(field, "fieldlabel").text = label
    ET.SubElement(field, "fieldentry").text = entry


def _normalize_choices(q: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of ``{id, text, is_correct}`` choices from the generator shape.

    ``QuestionData.choices`` is ``List[Dict]``; entries may carry ``id``/``text``
    (or ``label``/``content``) or be bare strings. The generator's per-choice
    ``is_correct`` flag is carried through (assessment-tail fix 2026-07-19):
    ``AssessmentGenerator`` MCQs mark the key via ``is_correct`` and leave
    ``correct_answer`` unset, so dropping the flag here silently emitted every
    MCQ ungraded (``QTI_NO_ANSWER_KEY``).
    """
    out: List[Dict[str, Any]] = []
    for i, choice in enumerate(q.get("choices") or []):
        note: Any = None
        if isinstance(choice, dict):
            cid = str(choice.get("id") or choice.get("ident") or choice.get("label") or "").strip()
            ctext = str(
                choice.get("text")
                if choice.get("text") is not None
                else choice.get("content", choice.get("value", ""))
            )
            is_correct = bool(choice.get("is_correct"))
            # Per-distractor targeted-misconception note (assessment-quality
            # overhaul) — carried through for the per-choice <itemfeedback> emit.
            raw_note = choice.get("misconception_note")
            if raw_note is not None:
                note = str(raw_note).strip() or None
        else:
            cid = ""
            ctext = str(choice)
            is_correct = False
        out.append({
            "id": _choice_ident(cid, i),
            "text": ctext,
            "is_correct": is_correct,
            "note": note,
        })
    return out


def _is_canonical_number(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text or not _NUMERIC_RE.match(text):
        return False
    try:
        Decimal(text)
    except (InvalidOperation, ValueError):
        return False
    return True


def _tf_choices() -> List[Dict[str, str]]:
    return [{"id": "True", "text": "True"}, {"id": "False", "text": "False"}]


_TAG_RE = re.compile(r"<[^>]+>")


def _norm_answer_text(text: str) -> str:
    """Normalize a choice text / answer token for matching.

    Strips HTML tags (generator choice texts arrive wrapped in ``<p>…</p>``
    while ``correct_answer`` is usually the bare text), collapses whitespace,
    and lowercases — so the two forms compare equal.
    """
    return " ".join(_TAG_RE.sub(" ", text or "").split()).lower()


def _correct_choice_ids(q: Dict[str, Any], choices: List[Dict[str, Any]]) -> List[str]:
    """Resolve the correct choice ident(s).

    Resolution ladder (assessment-tail fix 2026-07-19):

    1. ``correct_answer`` — a choice id, a choice text, or (for
       multiple_response) a list / comma-separated string of either. Text
       matching is HTML-stripped + whitespace-normalized so a bare-text
       answer matches a ``<p>``-wrapped choice body.
    2. Per-choice ``is_correct`` flags — the shape ``AssessmentGenerator``
       MCQs actually emit (``correct_answer`` unset). Never fabricates: only
       choices the generator itself flagged are used.

    An empty return means genuinely no key — the caller emits the item
    ungraded (anti-fabrication contract, unchanged).
    """
    raw = q.get("correct_answer")
    tokens: List[str] = []
    if raw is not None:
        if isinstance(raw, (list, tuple, set)):
            tokens = [str(t).strip() for t in raw]
        else:
            text = str(raw).strip()
            # multiple_response answers are often comma-separated.
            tokens = [t.strip() for t in text.split(",")] if "," in text else [text]
        tokens = [t for t in tokens if t]

    # Plural ``correct_answers`` is the canonical MULTIPLE-RESPONSE key list
    # (assessment-quality overhaul) — the generator sets it (as choice TEXTS)
    # and leaves scalar ``correct_answer`` unset. Harvest it alongside the
    # scalar so an MR item resolves even when its choices don't carry
    # ``is_correct`` flags. Set-membership resolution only (never fabricates).
    plural = q.get("correct_answers")
    if isinstance(plural, (list, tuple, set)):
        for t in plural:
            s = str(t).strip()
            if s and s not in tokens:
                tokens.append(s)

    by_id = {c["id"]: c["id"] for c in choices}
    by_id_ci = {c["id"].lower(): c["id"] for c in choices}
    by_text_ci = {c["text"].strip().lower(): c["id"] for c in choices}
    by_text_norm = {_norm_answer_text(c["text"]): c["id"] for c in choices}

    resolved: List[str] = []
    for tok in tokens:
        if tok in by_id:
            cid = by_id[tok]
        elif tok.lower() in by_id_ci:
            cid = by_id_ci[tok.lower()]
        elif tok.strip().lower() in by_text_ci:
            cid = by_text_ci[tok.strip().lower()]
        elif _norm_answer_text(tok) and _norm_answer_text(tok) in by_text_norm:
            cid = by_text_norm[_norm_answer_text(tok)]
        else:
            continue
        if cid not in resolved:
            resolved.append(cid)

    if resolved:
        return resolved

    # Fallback: the generator's own per-choice is_correct flags.
    return [c["id"] for c in choices if c.get("is_correct")]


def _accepted_fib_answers(q: Dict[str, Any]) -> List[str]:
    """Collect a fill-in's accepted answer strings.

    Prefers the explicit plural ``correct_answers`` (the multiple-accepted
    signal); falls back to the single scalar ``correct_answer``. Deliberately
    does NOT split ``correct_answer`` on commas — a thousands-separated numeric
    key (``1,000``) must stay a single accepted string, so ONLY the plural
    field promotes an item to the multiple-accepted (pattern_match) path.
    """
    out: List[str] = []
    plural = q.get("correct_answers")
    if isinstance(plural, (list, tuple, set)):
        for t in plural:
            s = str(t).strip()
            if s and s not in out:
                out.append(s)
    if not out:
        raw = q.get("correct_answer")
        if raw is not None:
            s = str(raw).strip()
            if s:
                out.append(s)
    return out


def _resolved_item_type(q: Dict[str, Any]) -> str:
    """Resolve the effective item type, applying the §9.1 numeric downgrade.

    A ``fill_in_blank`` with MULTIPLE accepted answer strings resolves to
    ``pattern_match`` (cc.pattern_match.v0p1) — the sixth CC profile — whose
    resprocessing ORs one varequal per accepted string. A single-answer
    ``fill_in_blank`` whose ``correct_answer`` is not a canonical
    integer/decimal string is downgraded to ``multiple_choice`` when choices
    exist, else to an (ungraded) ``short_answer`` — never silently dropped.
    """
    qtype = (q.get("question_type") or "").strip().lower()
    if qtype == "fill_in_blank":
        accepted = _accepted_fib_answers(q)
        if len(accepted) > 1:
            return "pattern_match"
        # A single accepted canonical-numeric key (from the scalar OR the
        # plural-of-one field) stays an exact-match fib.
        single = accepted[0] if accepted else None
        if single is not None and _is_canonical_number(single):
            return "fill_in_blank"
        if not _is_canonical_number(q.get("correct_answer")):
            if q.get("choices"):
                return "multiple_choice"
            return "short_answer"
    return qtype or "short_answer"


# ── per-type presentation + resprocessing builders ──────────────────────────
def _build_choice_item(
    item: ET.Element,
    q: Dict[str, Any],
    choices: List[Dict[str, str]],
    *,
    cardinality: str,
    response_ident: str,
    points: float,
) -> None:
    """Build a response_lid (render_choice) item + its resprocessing key."""
    presentation = ET.SubElement(item, "presentation")
    _material(presentation, str(q.get("stem", "")))

    response_lid = ET.SubElement(presentation, "response_lid")
    response_lid.set("ident", response_ident)
    response_lid.set("rcardinality", cardinality)
    render_choice = ET.SubElement(response_lid, "render_choice")
    for choice in choices:
        label = ET.SubElement(render_choice, "response_label")
        label.set("ident", choice["id"])
        _material(label, choice["text"])

    correct_ids = _correct_choice_ids(q, choices)
    if not correct_ids:
        # No resolvable key — still emit a well-formed presentation but no
        # respcondition (ungraded). Anti-fabrication: never invent a key.
        return

    resprocessing = ET.SubElement(item, "resprocessing")
    outcomes = ET.SubElement(resprocessing, "outcomes")
    decvar = ET.SubElement(outcomes, "decvar")
    decvar.set("varname", "SCORE")
    decvar.set("vartype", "Decimal")
    decvar.set("minvalue", "0")
    decvar.set("maxvalue", _fmt_points(points))

    respcondition = ET.SubElement(resprocessing, "respcondition")
    conditionvar = ET.SubElement(respcondition, "conditionvar")
    if cardinality == "Multiple" and len(correct_ids) > 1:
        and_el = ET.SubElement(conditionvar, "and")
        for cid in correct_ids:
            ve = ET.SubElement(and_el, "varequal")
            ve.set("respident", response_ident)
            ve.text = cid
    else:
        for cid in correct_ids:
            ve = ET.SubElement(conditionvar, "varequal")
            ve.set("respident", response_ident)
            ve.text = cid
    setvar = ET.SubElement(respcondition, "setvar")
    setvar.set("varname", "SCORE")
    setvar.set("action", "Set")
    setvar.text = _fmt_points(points)


def _build_fib_item(
    item: ET.Element,
    q: Dict[str, Any],
    *,
    response_ident: str,
    points: float,
) -> None:
    """Build a numeric fill-in (response_str + render_fib) item + key."""
    presentation = ET.SubElement(item, "presentation")
    _material(presentation, str(q.get("stem", "")))

    response_str = ET.SubElement(presentation, "response_str")
    response_str.set("ident", response_ident)
    response_str.set("rcardinality", "Single")
    ET.SubElement(response_str, "render_fib")

    # Prefer an accepted-answer harvest (scalar ``correct_answer`` OR a
    # single-entry plural ``correct_answers``) so a fib keyed only via the
    # plural field still emits its real key instead of the literal "None".
    accepted = _accepted_fib_answers(q)
    answer = accepted[0] if accepted else str(q.get("correct_answer") or "").strip()
    resprocessing = ET.SubElement(item, "resprocessing")
    outcomes = ET.SubElement(resprocessing, "outcomes")
    decvar = ET.SubElement(outcomes, "decvar")
    decvar.set("varname", "SCORE")
    decvar.set("vartype", "Decimal")
    decvar.set("minvalue", "0")
    decvar.set("maxvalue", _fmt_points(points))

    respcondition = ET.SubElement(resprocessing, "respcondition")
    conditionvar = ET.SubElement(respcondition, "conditionvar")
    varequal = ET.SubElement(conditionvar, "varequal")
    varequal.set("respident", response_ident)
    varequal.text = answer
    setvar = ET.SubElement(respcondition, "setvar")
    setvar.set("varname", "SCORE")
    setvar.set("action", "Set")
    setvar.text = _fmt_points(points)


def _build_pattern_match_item(
    item: ET.Element,
    q: Dict[str, Any],
    *,
    response_ident: str,
    points: float,
) -> None:
    """Build a pattern-match fill-in (response_str + render_fib) item + key.

    cc.pattern_match.v0p1 — the sixth CC profile — carries MULTIPLE accepted
    answer strings: the resprocessing ORs one ``varequal`` (case-insensitive)
    per accepted string, so any accepted form scores. Used when a
    ``fill_in_blank`` declares ``correct_answers`` with >1 entry (e.g. a
    fraction whose accepted forms are ``1/2`` / ``0.5``). No portable numeric
    tolerance exists — the instructor key carries the LMS-tolerance note.
    """
    presentation = ET.SubElement(item, "presentation")
    _material(presentation, str(q.get("stem", "")))

    response_str = ET.SubElement(presentation, "response_str")
    response_str.set("ident", response_ident)
    response_str.set("rcardinality", "Single")
    ET.SubElement(response_str, "render_fib")

    accepted = _accepted_fib_answers(q)
    resprocessing = ET.SubElement(item, "resprocessing")
    outcomes = ET.SubElement(resprocessing, "outcomes")
    decvar = ET.SubElement(outcomes, "decvar")
    decvar.set("varname", "SCORE")
    decvar.set("vartype", "Decimal")
    decvar.set("minvalue", "0")
    decvar.set("maxvalue", _fmt_points(points))

    respcondition = ET.SubElement(resprocessing, "respcondition")
    conditionvar = ET.SubElement(respcondition, "conditionvar")
    # OR one varequal per accepted string (case-insensitive → ``case="No"``).
    or_el = ET.SubElement(conditionvar, "or")
    for ans in accepted:
        varequal = ET.SubElement(or_el, "varequal")
        varequal.set("respident", response_ident)
        varequal.set("case", "No")
        varequal.text = ans
    setvar = ET.SubElement(respcondition, "setvar")
    setvar.set("varname", "SCORE")
    setvar.set("action", "Set")
    setvar.text = _fmt_points(points)


def _build_essay_item(item: ET.Element, q: Dict[str, Any], *, response_ident: str) -> None:
    """Build an ungraded short_answer/essay (response_str render_fib) item."""
    presentation = ET.SubElement(item, "presentation")
    _material(presentation, str(q.get("stem", "")))
    response_str = ET.SubElement(presentation, "response_str")
    response_str.set("ident", response_ident)
    response_str.set("rcardinality", "Single")
    render_fib = ET.SubElement(response_str, "render_fib")
    render_fib.set("rows", "6")
    render_fib.set("columns", "60")
    # No resprocessing — essay/short_answer are manually graded.


def _fmt_points(points: float) -> str:
    """Format a point value as an xs:decimal-safe string."""
    try:
        dec = Decimal(str(points))
    except (InvalidOperation, ValueError):
        return "1"
    if dec == dec.to_integral_value():
        return str(int(dec))
    return format(dec.normalize(), "f")


def _find_scoring_respcondition(item: ET.Element) -> Optional[ET.Element]:
    """Return the resprocessing ``respcondition`` that carries a ``setvar``.

    That is the answer-key-scoring condition (the one the LMS awards SCORE on);
    the Solution feedback is linked from it. Feedback-only respconditions carry
    no ``setvar``. Returns ``None`` for an ungraded item (no resprocessing).
    """
    resprocessing = item.find("resprocessing")
    if resprocessing is None:
        return None
    for respcondition in resprocessing.findall("respcondition"):
        if respcondition.find("setvar") is not None:
            return respcondition
    return None


def _apply_itemfeedback(
    item: ET.Element,
    q: Dict[str, Any],
    choices: List[Dict[str, Any]],
    *,
    response_ident: str,
) -> None:
    """Emit per-distractor + worked-solution ``<itemfeedback>`` (default ON).

    Recovers content the generator already authors but the pre-feature emitter
    dropped at the XML boundary:

    * One ``<itemfeedback>`` carrying a ``<solution>`` for the item's
      ``feedback`` (worked solution). When the item is graded, the scoring
      ``respcondition`` gains a ``<displayfeedback feedbacktype="Solution">``
      linking to it (the spec-required linkage).
    * One ``<itemfeedback>`` with a ``<flow_mat>`` per choice carrying a
      ``misconception_note``. Each is linked from a NEW feedback-only
      ``respcondition`` (``continue="Yes"``, no ``setvar``) whose
      ``varequal`` matches that choice — the canonical QTI 1.2 targeted-feedback
      shape.

    XSD ordering: ``itemfeedback`` elements are appended to ``item`` LAST
    (after ``resprocessing``), matching ``Item.Type``; feedback respconditions
    append after the scoring one; ``displayfeedback`` appends after ``setvar``.
    No-op (byte-identical) when the gate is off or the item carries no
    authored feedback.
    """
    if not itemfeedback_enabled():
        return

    feedback_text = q.get("feedback")
    feedback_text = str(feedback_text).strip() if feedback_text is not None else ""
    noted = [(c["id"], c["note"]) for c in choices if c.get("note")]
    if not feedback_text and not noted:
        return

    qid = item.get("ident", "item")
    resprocessing = item.find("resprocessing")

    # ---- Worked-solution feedback (Solution-type itemfeedback).
    if feedback_text:
        fb_ident = _ncname(f"{qid}_fb_solution", fallback="fb_solution")
        scoring = _find_scoring_respcondition(item)
        if scoring is not None:
            df = ET.SubElement(scoring, "displayfeedback")
            df.set("feedbacktype", "Solution")
            df.set("linkrefid", fb_ident)
        itemfeedback = ET.SubElement(item, "itemfeedback")
        itemfeedback.set("ident", fb_ident)
        solution = ET.SubElement(itemfeedback, "solution")
        solutionmaterial = ET.SubElement(solution, "solutionmaterial")
        flow_mat = ET.SubElement(solutionmaterial, "flow_mat")
        _material(flow_mat, feedback_text)

    # ---- Per-distractor targeted-misconception feedback (Response-type).
    seen_idents: set = set()
    for cid, note in noted:
        fb_ident = _ncname(f"{qid}_fb_{cid}", fallback="fb_choice")
        # Guarantee a distinct ident per feedback block within the item.
        base = fb_ident
        n = 1
        while fb_ident in seen_idents:
            fb_ident = f"{base}_{n}"
            n += 1
        seen_idents.add(fb_ident)

        if resprocessing is not None:
            respcondition = ET.SubElement(resprocessing, "respcondition")
            respcondition.set("continue", "Yes")
            conditionvar = ET.SubElement(respcondition, "conditionvar")
            varequal = ET.SubElement(conditionvar, "varequal")
            varequal.set("respident", response_ident)
            varequal.text = cid
            df = ET.SubElement(respcondition, "displayfeedback")
            df.set("feedbacktype", "Response")
            df.set("linkrefid", fb_ident)

        itemfeedback = ET.SubElement(item, "itemfeedback")
        itemfeedback.set("ident", fb_ident)
        flow_mat = ET.SubElement(itemfeedback, "flow_mat")
        _material(flow_mat, str(note))


# ── public API ──────────────────────────────────────────────────────────────
def question_to_qti_item(q: Any) -> ET.Element:
    """Transform one ``QuestionData`` (dict or dataclass) into a ``<item>`` element.

    The returned element is namespace-naked (no ``xmlns``) so it can be nested
    inside a ``<section>`` of an ``assessment_to_qti`` document where the
    default namespace is already declared on the root. Callers wanting a
    standalone item should reparent it under a namespaced root.
    """
    q = _as_dict(q)
    qid = _ncname(str(q.get("question_id", "")), fallback="question")
    effective_type = _resolved_item_type(q)
    profile = resolve_cc_profile(effective_type)
    points = q.get("points", 1.0)
    try:
        points = float(points)
    except (TypeError, ValueError):
        points = 1.0

    item = ET.Element("item")
    item.set("ident", qid)
    item.set("title", str(q.get("objective_id") or qid))

    # itemmetadata → qtimetadata → cc_profile + cc_weighting
    itemmetadata = ET.SubElement(item, "itemmetadata")
    qtimetadata = ET.SubElement(itemmetadata, "qtimetadata")
    _qtimetadata_field(qtimetadata, "cc_profile", profile)
    _qtimetadata_field(qtimetadata, "cc_weighting", _fmt_points(points))
    # Honor the diversified-tier pedagogy discriminator (assessment-quality
    # overhaul) as machine-readable metadata provenance. ``item_subtype``
    # (error_analysis / matching_mc / ordering_mc / two_tier_answer /
    # two_tier_reason / mc_multiple_response / fib_numeric / …) names the
    # pedagogy simulated within the portable CC primitive; ``linked_item_id``
    # binds the two tiers of a two-tier answer↔reason item. Both are additive:
    # an extra ``qtimetadatafield`` is XSD-valid (QTIMetadata allows unbounded
    # fields) and legacy items (no subtype) emit byte-identically.
    subtype = q.get("item_subtype")
    if subtype:
        _qtimetadata_field(qtimetadata, "item_subtype", str(subtype))
    linked = q.get("linked_item_id")
    if linked:
        _qtimetadata_field(qtimetadata, "linked_item_id", str(linked))

    # Per-item response ident is namespaced under the item id so response_lid
    # idents (xs:ID) stay unique across the document.
    response_ident = _ncname(f"{qid}_response", fallback="response")

    # The choice list used for feedback linkage must match what the builder
    # emitted, so compute it once for the choice-based types.
    fb_choices: List[Dict[str, Any]] = []
    if effective_type == "multiple_choice":
        fb_choices = _normalize_choices(q)
        _build_choice_item(
            item, q, fb_choices,
            cardinality="Single", response_ident=response_ident, points=points,
        )
    elif effective_type == "multiple_response":
        fb_choices = _normalize_choices(q)
        _build_choice_item(
            item, q, fb_choices,
            cardinality="Multiple", response_ident=response_ident, points=points,
        )
    elif effective_type == "true_false":
        fb_choices = _normalize_choices(q) or _tf_choices()
        _build_choice_item(
            item, q, fb_choices,
            cardinality="Single", response_ident=response_ident, points=points,
        )
    elif effective_type == "pattern_match":
        _build_pattern_match_item(item, q, response_ident=response_ident, points=points)
    elif effective_type == "fill_in_blank":
        _build_fib_item(item, q, response_ident=response_ident, points=points)
    else:  # short_answer / essay / unknown → ungraded essay shape
        _build_essay_item(item, q, response_ident=response_ident)

    # Per-distractor + worked-solution feedback (default ON; kill-switch
    # ED4ALL_ASSESSMENT_ITEMFEEDBACK=0). Appended LAST so itemfeedback lands
    # after resprocessing per Item.Type.
    _apply_itemfeedback(item, q, fb_choices, response_ident=response_ident)

    return item


def assessment_to_qti(a: Any) -> str:
    """Transform one ``AssessmentData`` (dict or dataclass) into a QTI document string.

    Produces ``<questestinterop>`` → ``<assessment>`` (with the
    ``cc_profile=cc.exam.v0p1`` ``qtimetadata``) → ``<section>`` → ``<item>+``,
    matching the Pattern-15 worked example and round-tripping ``QTIParser``.
    """
    a = _as_dict(a)
    assess_id = _ncname(str(a.get("assessment_id", "")), fallback="assessment")
    title = str(a.get("title") or assess_id)

    ET.register_namespace("", QTI_NS)
    ET.register_namespace("xsi", _XSI_NS)
    root = ET.Element(f"{{{QTI_NS}}}questestinterop")
    root.set(
        f"{{{_XSI_NS}}}schemaLocation",
        f"{QTI_NS} http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd",
    )

    assessment = ET.SubElement(root, "assessment")
    assessment.set("ident", assess_id)
    assessment.set("title", title)

    qtimetadata = ET.SubElement(assessment, "qtimetadata")
    _qtimetadata_field(qtimetadata, "cc_profile", CC_EXAM_PROFILE)
    _qtimetadata_field(qtimetadata, "qmd_assessmenttype", "Examination")

    section = ET.SubElement(assessment, "section")
    section.set("ident", "root_section")

    for q in a.get("questions") or []:
        item = question_to_qti_item(q)
        section.append(item)

    return _serialize(root)


#: Per-item ``qtimetadata`` fields the objectbank stamps so a bank consumer can
#: SELECT items without re-parsing prose. Ordered; ``(fieldlabel, source key)``.
#: Only fields whose source value is present + non-empty are emitted, so an item
#: lacking (say) ``item_subtype`` simply carries fewer fields — never a blank.
BANK_METADATA_FIELDS: List[tuple] = [
    ("ed4all_objective_id", "objective_id"),
    ("ed4all_bloom_level", "bloom_level"),
    ("ed4all_item_subtype", "item_subtype"),
    ("ed4all_question_type", "question_type"),
    ("ed4all_cognitive_task_type", "cognitive_task_type"),
]


def _stamp_bank_metadata(item: ET.Element, q: Dict[str, Any]) -> None:
    """Append the queryable bank fields to an item's ``itemmetadata/qtimetadata``.

    Reuses the element ``question_to_qti_item`` already built (which always
    emits ``itemmetadata`` → ``qtimetadata`` carrying ``cc_profile``), so the
    bank item is the EXACT same item shape an exam carries, plus additive
    selection metadata. Idempotent per call: a field already present under the
    same ``fieldlabel`` is not duplicated.
    """
    itemmeta = item.find("itemmetadata")
    if itemmeta is None:
        itemmeta = ET.SubElement(item, "itemmetadata")
    qtimeta = itemmeta.find("qtimetadata")
    if qtimeta is None:
        qtimeta = ET.SubElement(itemmeta, "qtimetadata")

    existing = {
        (f.findtext("fieldlabel") or "").strip()
        for f in qtimeta.findall("qtimetadatafield")
    }
    for label, key in BANK_METADATA_FIELDS:
        if label in existing:
            continue
        raw = q.get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        _qtimetadata_field(qtimeta, label, value)


def assessment_to_objectbank(bank: Any) -> str:
    """Transform a pool of questions into a QTI 1.2 ``<objectbank>`` document.

    This is the ITEM-BANK surface: the storage format for a course's whole
    question pool, as opposed to :func:`assessment_to_qti`, which emits one
    fixed ``<assessment>`` (an exam). ``objectbank`` is the QTI 1.2 container
    for a reusable item pool, so a per-week quiz becomes a SELECTION from the
    bank rather than the bank itself.

    Input accepts the same shapes as :func:`assessment_to_qti` — a dataclass
    exposing ``to_dict()`` or the plain dict — reading ``questions`` (or
    ``items``). Recognised keys: ``bank_id`` / ``assessment_id`` (ident) and
    ``title``.

    Every item is built by the SAME :func:`question_to_qti_item` the exam path
    uses, then stamped with the additive :data:`BANK_METADATA_FIELDS`
    ``qtimetadata`` (objective id, Bloom level, item subtype, question type,
    cognitive task type) so a consumer can select by objective or Bloom level
    without re-parsing the stem. No content is invented — a field absent on the
    question is simply not emitted (anti-fabrication contract).
    """
    bank = _as_dict(bank)
    bank_id = _ncname(
        str(bank.get("bank_id") or bank.get("assessment_id") or ""),
        fallback="objectbank",
    )
    title = str(bank.get("title") or bank_id)

    ET.register_namespace("", QTI_NS)
    ET.register_namespace("xsi", _XSI_NS)
    root = ET.Element(f"{{{QTI_NS}}}questestinterop")
    root.set(
        f"{{{_XSI_NS}}}schemaLocation",
        f"{QTI_NS} http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd",
    )

    objectbank = ET.SubElement(root, "objectbank")
    objectbank.set("ident", bank_id)

    bank_meta = ET.SubElement(objectbank, "qtimetadata")
    _qtimetadata_field(bank_meta, "cc_profile", CC_EXAM_PROFILE)
    _qtimetadata_field(bank_meta, "ed4all_bank_title", title)

    questions = bank.get("questions")
    if questions is None:
        questions = bank.get("items") or []
    for q in questions or []:
        q_dict = _as_dict(q)
        item = question_to_qti_item(q_dict)
        _stamp_bank_metadata(item, q_dict)
        objectbank.append(item)

    return _serialize(root)


def discussion_to_imsdt(prompt: Any) -> str:
    """Transform a discussion prompt into an IMS discussion-topic (imsdt) document.

    Input dict keys: ``title`` (or ``stem``/``objective_id`` fallback) and
    ``text`` (or ``prompt``/``stem``/``body``). The text is emitted as
    ``texttype="text/html"``.
    """
    prompt = _as_dict(prompt)
    title = str(
        prompt.get("title")
        or prompt.get("stem")
        or prompt.get("objective_id")
        or "Discussion"
    ).strip() or "Discussion"
    body = str(
        prompt.get("text")
        if prompt.get("text") is not None
        else prompt.get("prompt", prompt.get("body", prompt.get("stem", "")))
    )

    ET.register_namespace("", IMSDT_NS)
    root = ET.Element(f"{{{IMSDT_NS}}}topic")
    ET.SubElement(root, "title").text = title
    text_el = ET.SubElement(root, "text")
    text_el.set("texttype", _HTML_TEXTTYPE)
    text_el.text = body

    return _serialize(root)


def assignment_to_resource(task: Any) -> str:
    """Transform an assignment task into a learning-application-resource document.

    Input dict keys: ``title`` (or ``objective_id``), ``text``/``prompt``/``body``
    (learner-facing task), and optional ``instructor_text``/``points``.
    """
    task = _as_dict(task)
    ident = _ncname(
        str(task.get("assignment_id") or task.get("question_id") or task.get("objective_id") or ""),
        fallback="assignment",
    )
    title = str(task.get("title") or task.get("objective_id") or ident).strip() or ident
    body = str(
        task.get("text")
        if task.get("text") is not None
        else task.get("prompt", task.get("body", task.get("stem", "")))
    )
    instructor_text = str(task.get("instructor_text") or "Grade against the stated objective.")

    ET.register_namespace("", ASSIGNMENT_NS)
    root = ET.Element(f"{{{ASSIGNMENT_NS}}}assignment")
    root.set("identifier", ident)

    ET.SubElement(root, "title").text = title
    text_el = ET.SubElement(root, "text")
    text_el.set("texttype", _HTML_TEXTTYPE)
    text_el.text = body
    instr_el = ET.SubElement(root, "instructor_text")
    instr_el.set("texttype", _HTML_TEXTTYPE)
    instr_el.text = instructor_text

    points = task.get("points")
    if points is not None:
        gradable = ET.SubElement(root, "gradable")
        gradable.set("points_possible", _fmt_points_decimal(points))
        gradable.text = "true"

    return _serialize(root)


def _fmt_points_decimal(points: Any) -> str:
    try:
        return _fmt_points(float(points))
    except (TypeError, ValueError):
        return "1"


def _serialize(root: ET.Element) -> str:
    """Serialize an element tree to a UTF-8 XML string with declaration."""
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body
