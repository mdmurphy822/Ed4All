"""Studio learner quiz service — locate + parse QTI, strip the key, server-grade.

W10 Phase 3 SERVICE LAYER (spec: ``plans/finegrain/w10-assessments-qti-discussions-2026-06.md``
§5.1 + §11 Phase 3). The route wiring + front-end are a LATER wave; this module
is the pure-ish back end the routes will call.

What it does (and the contract it upholds):

1. **Locate** a course's packaged QTI XML. ``locate_quiz_xml`` scans the LibV2
   archive — the ``06_assessments/`` sibling first (loose ``*.xml`` per the
   §4 packaging convention), then inside the packaged cartridge
   (``source/imscc/*.imscc`` — ``_IMSCC_ROOTS``). It **fails soft**: a course
   with no quizzes returns ``[]``, never raises. Tests inject XML directly
   (path / bytes / str) so they never need a real archived course.
2. **Parse** with the in-tree ``Trainforge.parsers.qti_parser.QTIParser`` —
   imported + reused, NOT reimplemented.
3. **Learner JSON** (``to_learner_json``) returns
   ``{assessment_id, title, questions:[{question_id, stem, type,
   choices:[{id,text}]}]}`` — and OMITS the answer key entirely. There is no
   ``correct_answer`` / ``correct_response`` / per-choice ``is_correct`` and no
   answer-revealing feedback anywhere in this payload. The answer key NEVER
   leaves this module toward the learner.
4. **Server-side grade** (``grade_submission``) — given learner selections + the
   parsed assessment (which carries the key), returns
   ``{score, max_score, per_item:[{question_id, correct, feedback}]}``. Grading
   is deterministic + STATELESS (no attempt persistence — spec §10).
5. **Numeric equivalence** (spec §9.1): numeric fill-in is graded with
   ``fractions.Fraction`` equivalence (``0.5 == 1/2 == .50``), reusing the
   precedent parser ``lib.validators.assessment_numeric_equivalence._parse_numeric``.

Rendering note: the question ``stem`` is returned as RAW text. The route /
render layer (a later wave) HTML-escapes it on render — this module does no
escaping, matching the ``QuestionData.stem`` → server-render contract in §5.1.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import xml.etree.ElementTree as ET

# Reuse the in-tree QTI 1.2 parser (do NOT reimplement QTI parsing).
from Trainforge.parsers.qti_parser import QTIAssessment, QTIParser

# Reuse the Fraction-equivalence numeric parser precedent (spec §9.1).
from lib.validators.assessment_numeric_equivalence import _parse_numeric

# Reuse the audited slug shape + the IMSCC archive convention.
from gui.services.imscc_service import (
    _IMSCC_ROOTS,
    _SLUG_RE,
    _libv2_root,
    _validate_slug,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Where the pre-packaging assessment_synthesis phase writes loose QTI XML
# (§4 packaging convention — sibling of the content dir, surfaced in the LibV2
# archive). Scanned before the packaged cartridge.
_ASSESSMENT_ROOTS = ("06_assessments",)

# QTI item profiles this service treats as auto-gradable (spec §2.2). Others
# (essay / short_answer) parse + render but are not auto-graded.
_NUMERIC_TYPES = frozenset({"fill_in_blank"})


class QuizServiceError(Exception):
    """Raised on a malformed / unparseable quiz or an invalid slug.

    ``status`` + ``code`` mirror the ``SourcePageError`` (status, code, detail)
    contract the routes map, so the later route wave can surface a clean HTTP
    error without re-classifying.
    """

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- #
# Locate (fail-soft archive discovery)
# --------------------------------------------------------------------------- #


def locate_quiz_xml(
    slug: str, libv2_root: Optional[Path] = None
) -> List[bytes]:
    """Return the raw QTI XML bytes for every quiz packaged for ``slug``.

    Fail-soft by contract: a course with no archived quizzes (no
    ``06_assessments/`` XML, no QTI resource in the cartridge) returns ``[]``
    rather than raising. An invalid slug shape DOES raise (defence in depth via
    ``_validate_slug``), so a traversal attempt never silently scans off-root.

    Discovery order:

    1. ``LibV2/courses/<slug>/06_assessments/*.xml`` (loose, pre-packaging form).
    2. QTI resources inside ``LibV2/courses/<slug>/source/imscc/*.imscc``
       (member names whose content roots as ``<questestinterop>``).
    """
    root = Path(libv2_root) if libv2_root is not None else _libv2_root()
    # _validate_slug raises on a malformed slug / escaping path; a missing course
    # dir 404s — translate that to fail-soft empty (no quizzes for an unknown
    # course is "no quizzes", not an error on the locate surface).
    try:
        course_dir = _validate_slug(slug, root)
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status", None)
        if status == 404:
            return []
        # 422 (malformed slug) is a real caller error — re-raise as ours.
        if status == 422:
            raise QuizServiceError(422, "invalid_course_id", str(exc)) from None
        return []

    blobs: List[bytes] = []

    # 1) Loose 06_assessments/*.xml
    for rel in _ASSESSMENT_ROOTS:
        adir = course_dir / rel
        if not adir.is_dir():
            continue
        for xml_path in sorted(adir.glob("*.xml")):
            try:
                data = xml_path.read_bytes()
            except OSError:
                continue
            if _looks_like_qti(data):
                blobs.append(data)

    # 2) QTI members inside the packaged cartridge.
    cartridge = _find_cartridge(course_dir)
    if cartridge is not None:
        blobs.extend(_qti_members_from_cartridge(cartridge))

    return blobs


def _find_cartridge(course_dir: Path) -> Optional[Path]:
    """First ``*.imscc`` cartridge under the IMSCC roots, or None (fail-soft)."""
    for rel in _IMSCC_ROOTS:
        rdir = course_dir / rel
        if not rdir.is_dir():
            continue
        cartridges = sorted(rdir.glob("*.imscc"))
        if cartridges:
            return cartridges[0]
    return None


def _qti_members_from_cartridge(cartridge: Path) -> List[bytes]:
    """Return the bytes of every cartridge member that looks like QTI XML.

    Reads members in-memory (never extracts the untrusted archive to disk).
    Fail-soft: a corrupt zip / unreadable member is skipped, never raised.
    """
    out: List[bytes] = []
    try:
        with zipfile.ZipFile(cartridge) as zf:
            for name in sorted(zf.namelist()):
                if not name.lower().endswith(".xml"):
                    continue
                try:
                    data = zf.read(name)
                except (KeyError, OSError, zipfile.BadZipFile):
                    continue
                if _looks_like_qti(data):
                    out.append(data)
    except (zipfile.BadZipFile, OSError):
        return out
    return out


def _looks_like_qti(data: bytes) -> bool:
    """Cheap sniff: does the XML root tag end in ``questestinterop``?

    Avoids a full parse on every cartridge member; the manifest XML + content
    HTML are skipped without raising.
    """
    try:
        head = data[:4096].decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return False
    return "questestinterop" in head.lower()


# --------------------------------------------------------------------------- #
# Parse (reuse the in-tree QTIParser)
# --------------------------------------------------------------------------- #


def parse_quiz(source: Union[str, bytes, Path]) -> QTIAssessment:
    """Parse a single QTI document into a ``QTIAssessment``.

    ``source`` is a direct XML string, raw bytes, or a filesystem path — the
    path/bytes injection that lets tests feed a hand-authored fixture without a
    real archived course. Raises ``QuizServiceError(422, ...)`` on malformed XML
    so the route layer can surface a clean 422.
    """
    xml_text = _coerce_xml_text(source)
    try:
        assessment = QTIParser().parse_string(xml_text)
    except ValueError as exc:  # parser wraps ET.ParseError as ValueError
        raise QuizServiceError(422, "invalid_quiz_xml", str(exc)) from exc

    # The in-tree parser reads ``ident``/``title`` off the document ROOT
    # (``<questestinterop>``), which carries neither — the id + title live on the
    # nested ``<assessment>`` element. Backfill them from there so the learner
    # JSON + grader carry the real assessment identity (the parser falls back to
    # "unknown" / "Untitled Assessment" otherwise).
    ident, title = _assessment_identity(xml_text)
    if ident:
        assessment.id = ident
    if title:
        assessment.title = title
    return assessment


def _assessment_identity(xml_text: str) -> tuple:
    """Return ``(ident, title)`` read off the ``<assessment>`` element.

    Empty strings when the element / attribute is absent. Namespace-agnostic
    (matches the local tag name), mirroring the parser's tag-handling style.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return "", ""
    for el in root.iter():
        if _local(el.tag) == "assessment":
            return el.get("ident", ""), el.get("title", "")
    return "", ""


def _coerce_xml_text(source: Union[str, bytes, Path]) -> str:
    """Normalise a path / bytes / str source into XML text."""
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if isinstance(source, bytes):
        return source.decode("utf-8")
    if isinstance(source, str):
        # A bare path string (no XML markup) is read from disk; otherwise the
        # string IS the XML. The QTI root tag is unambiguous, so dispatch on it.
        if "<" not in source and os.path.exists(source):
            return Path(source).read_text(encoding="utf-8")
        return source
    raise QuizServiceError(
        422, "invalid_quiz_source", f"unsupported quiz source type: {type(source).__name__}"
    )


# --------------------------------------------------------------------------- #
# Learner-facing JSON (OMITS the answer key)
# --------------------------------------------------------------------------- #


def to_learner_json(assessment: QTIAssessment) -> Dict[str, Any]:
    """Return the learner-facing quiz JSON — WITHOUT any answer key.

    Shape::

        {
          "assessment_id": "...",
          "title": "...",
          "questions": [
            {"question_id": "...", "stem": "...", "type": "...",
             "choices": [{"id": "...", "text": "..."}]}
          ]
        }

    KEY ANTI-LEAK CONTRACT: no ``correct_answer`` / ``correct_response`` /
    per-choice ``is_correct`` / answer-revealing ``feedback`` field appears
    anywhere in the returned dict. The choice list is rebuilt field-by-field
    (only ``id`` + ``text``) so the parser's ``is_correct`` flag is structurally
    impossible to leak. ``stem`` is RAW text — the renderer escapes it.
    """
    questions: List[Dict[str, Any]] = []
    for q in assessment.questions:
        questions.append(
            {
                "question_id": q.id,
                "stem": q.stem,
                "type": q.type,
                # Rebuild choices from id+text ONLY — never copy is_correct.
                "choices": [{"id": c.id, "text": c.text} for c in q.choices],
            }
        )
    return {
        "assessment_id": assessment.id,
        "title": assessment.title,
        "questions": questions,
    }


# --------------------------------------------------------------------------- #
# Answer-key extraction (server-side only — never returned to the learner)
# --------------------------------------------------------------------------- #


# QTI 1.2 ``<varequal>`` carries one correct response id (a choice ident for
# choice items, or a literal answer value for fib items). Namespace-agnostic
# local-name match mirrors the in-tree parser's ``.tag.lower()`` style.
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _correct_response_ids(item: ET.Element) -> List[str]:
    """Return every ``varequal`` value under a positive-score ``respcondition``.

    The in-tree parser keeps only the LAST ``varequal`` per item (single-correct
    semantics), which loses ``multiple_response`` keys. This walk collects ALL
    correct response values from each scoring ``respcondition`` so the grader can
    enforce the all-or-nothing multi-select contract (spec §9.5).
    """
    correct: List[str] = []
    for resp in item.iter():
        if _local(resp.tag) != "resprocessing":
            continue
        for cond in resp.iter():
            if _local(cond.tag) != "respcondition":
                continue
            setvars: List[ET.Element] = []
            varequals: List[ET.Element] = []
            for child in cond.iter():
                lname = _local(child.tag)
                if lname == "setvar":
                    setvars.append(child)
                elif lname == "varequal":
                    varequals.append(child)
            # A respcondition is a SCORING condition when it sets a positive
            # SCORE. Mirrors the parser's ``score > 0`` gate.
            scoring = False
            for sv in setvars:
                txt = (sv.text or "0").strip()
                try:
                    if float(txt) > 0:
                        scoring = True
                        break
                except (ValueError, TypeError):
                    continue
            # If there's no setvar at all but there IS a varequal, treat it as a
            # correct-answer condition (some emitters omit the score on the
            # correct branch). Conservative: only when no setvar present.
            if not setvars and varequals:
                scoring = True
            if not scoring:
                continue
            for ve in varequals:
                val = (ve.text or "").strip()
                if val:
                    correct.append(val)
    # De-dupe preserving order.
    seen: set = set()
    ordered: List[str] = []
    for v in correct:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _build_answer_key(xml_text: str) -> Dict[str, List[str]]:
    """Map ``question_id -> [correct response value, ...]`` from the QTI XML.

    Server-side only. The values are choice idents (mc / mresp / true_false) or
    literal answer strings (numeric fib). Returned to the GRADER, never to the
    learner JSON.
    """
    key: Dict[str, List[str]] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return key
    for item in root.iter():
        if _local(item.tag) != "item":
            continue
        ident = item.get("ident")
        if not ident:
            continue
        key[ident] = _correct_response_ids(item)
    return key


# --------------------------------------------------------------------------- #
# Grading (deterministic, stateless)
# --------------------------------------------------------------------------- #


def _normalize_selection(value: Any) -> List[str]:
    """Coerce a learner selection into a list of string response ids."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _numeric_match(submitted: str, key_values: List[str]) -> bool:
    """True if ``submitted`` is numerically equivalent to any key value.

    Uses ``fractions.Fraction`` equivalence (spec §9.1) so ``0.5`` / ``1/2`` /
    ``.50`` all match. Falls back to a case-insensitive string compare when a
    value isn't numerically parseable (e.g. a textual fib answer).
    """
    sub_val = _parse_numeric(submitted)
    for kv in key_values:
        key_val = _parse_numeric(kv)
        if sub_val is not None and key_val is not None:
            if sub_val == key_val:
                return True
        elif submitted.strip().casefold() == kv.strip().casefold():
            return True
    return False


def _grade_item(
    question: Any, selection: List[str], key_values: List[str]
) -> bool:
    """Grade one question. All-or-nothing for multi-select (spec §9.5)."""
    if not key_values:
        # No resolvable key (e.g. essay / short_answer): not auto-gradable →
        # never marked correct by the auto-grader.
        return False

    # Numeric fill-in detection is robust to the in-tree parser's type quirk:
    # the parser tags ``response_str`` items ``essay`` (only ``response_fib``
    # becomes ``fill_in_blank``), so a numeric FIB authored with ``response_str``
    # + ``render_fib`` arrives typed ``essay``. Treat a question as numeric-FIB
    # when it's an explicit fib type OR it has NO choices and every key value
    # parses as a number (so an essay with a free-text key isn't mis-graded).
    is_choice = bool(question.choices)
    numeric_key = not is_choice and all(
        _parse_numeric(kv) is not None for kv in key_values
    )
    if question.type in _NUMERIC_TYPES or numeric_key:
        # Numeric fib: any submitted value matching the key (Fraction-equivalent).
        if not selection:
            return False
        return any(_numeric_match(sel, key_values) for sel in selection)

    # Choice items (mc / mresp / true_false): exact SET equality of selected
    # response ids vs the correct id set. All-or-nothing — a multiple_response
    # answer that's missing one correct id OR includes one wrong id is incorrect
    # (no partial credit, spec §9.5).
    return set(selection) == set(key_values)


def grade_submission(
    assessment: QTIAssessment,
    submission: Dict[str, Any],
    *,
    quiz_xml: Optional[str] = None,
) -> Dict[str, Any]:
    """Server-side grade a learner submission against the parsed quiz.

    Args:
        assessment: the ``QTIAssessment`` returned by :func:`parse_quiz`.
        submission: ``{question_id: selection}`` where ``selection`` is a single
            response id / value (mc / tf / fib) or a list of ids (mresp).
        quiz_xml: the ORIGINAL QTI XML for richer multi-correct key extraction.
            Optional — when omitted, the key falls back to the parser's
            per-choice ``is_correct`` flags + ``correct_response`` (single-correct
            only; multiple_response then can't resolve its full key). Pass it for
            full fidelity (the route wave threads the located XML through here).

    Returns::

        {"score": int, "max_score": int,
         "per_item": [{"question_id": str, "correct": bool, "feedback": str}]}

    Deterministic + STATELESS — no attempt persistence (spec §10). The answer
    key is consumed internally and never surfaced in the result.
    """
    # Prefer the XML-derived multi-correct key; fall back to the parser flags.
    xml_key: Dict[str, List[str]] = (
        _build_answer_key(quiz_xml) if quiz_xml else {}
    )

    per_item: List[Dict[str, Any]] = []
    score = 0
    max_score = 0

    for q in assessment.questions:
        max_score += 1
        key_values = xml_key.get(q.id)
        if key_values is None:
            # Fall back to the parser's single-correct signals.
            key_values = [c.id for c in q.choices if c.is_correct]
            if not key_values and q.correct_response:
                key_values = [q.correct_response]

        selection = _normalize_selection(submission.get(q.id))
        correct = _grade_item(q, selection, key_values)
        if correct:
            score += 1
            feedback = "Correct."
        elif not key_values:
            feedback = "This question is not auto-graded."
        else:
            feedback = "Incorrect. Review the related material and try again."

        per_item.append(
            {"question_id": q.id, "correct": correct, "feedback": feedback}
        )

    return {"score": score, "max_score": max_score, "per_item": per_item}


__all__ = [
    "QuizServiceError",
    "locate_quiz_xml",
    "parse_quiz",
    "to_learner_json",
    "grade_submission",
]
