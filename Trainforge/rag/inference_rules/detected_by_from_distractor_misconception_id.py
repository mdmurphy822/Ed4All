"""Rule: derive ``detected-by-question`` edges from questions carrying a
``misconception_id`` pointer.

GPT Feedback (12 May 2026) item 4: when an assessment question /
distractor pair encodes an explicit misconception (the rejected branch
of a ``preference_pair`` shape, or the distractor of a generated
multiple-choice question), the question DETECTS that misconception.
This rule materializes that implicit pointer as an explicit typed
edge::

    misconception_id --detected-by-question--> question_id

Federation-by-convention: ``source`` is a misconception ID
(``mc_[0-9a-f]{16}``); ``target`` is a question ID (e.g. ``MCQ-<8hex>``,
``MRQ-<8hex>``, ``TF-<8hex>``, ``FIB-<8hex>`` per
``Trainforge/generators/question_factory.py``, or the deterministic
``q_<chunk_id>_<lo_ref>`` form per
``process_course.py::_build_questions_for_graph``). No new node types
are added to the concept-graph schema.

Confidence is ``1.0`` — the misconception reference is explicit.

**Signal-availability caveat.** Questions are not currently threaded
into ``build_semantic_graph``'s main call chain with the
``misconception_id`` field populated. Both
``preference_pair.schema.json::misconception_id`` and the
question-factory distractor surface carry the pointer, but
``_build_questions_for_graph`` (the call-site that constructs the
``questions=`` kwarg) does NOT yet propagate it. When the upstream
pipeline starts threading the field, this rule fires automatically;
until then it emits ``[]`` gracefully — the same shape as the
``misconception_of_from_misconception_ref`` rule prior to Wave-99 LO
backfill.

Deterministic: output sorted by (source, target); duplicates on the
same (misconception_id, question_id) pair are collapsed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

RULE_NAME = "detected_by_from_distractor_misconception_id"
RULE_VERSION = 1
EDGE_TYPE = "detected-by-question"


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    *,
    questions: List[Dict[str, Any]] | None = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``detected-by-question`` edges when questions declare a
    ``misconception_id`` pointer.

    Args:
        chunks: Unused; interface parity.
        course: Unused; interface parity.
        concept_graph: Unused; endpoints reference external namespaces.
        questions: Optional list of question dicts. Each may have ``id``
            and ``misconception_id``. When ``None`` or empty, emits no
            edges (current pipeline state — awaiting upstream wiring).

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    del chunks, course, concept_graph  # unused; interface parity

    if not questions:
        return []

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        q_id = q.get("id")
        mc_id = q.get("misconception_id")
        if not q_id or not mc_id:
            continue
        key = (mc_id, q_id)
        if key in seen:
            continue
        seen[key] = {
            "source": mc_id,
            "target": q_id,
            "type": EDGE_TYPE,
            "confidence": 1.0,
            "provenance": {
                "rule": RULE_NAME,
                "rule_version": RULE_VERSION,
                "evidence": {
                    "misconception_id": mc_id,
                    "question_id": q_id,
                },
            },
        }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))
