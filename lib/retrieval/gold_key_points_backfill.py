"""Gold-question expected_key_points backfill PROPOSAL builder.

Companion to the ``GOLD_QUESTION_METADATA_INCOMPLETE`` gold-validate warning
(``lib/retrieval/gold_set.py::_metadata_completeness_issues``): every non-refusal
v1.1 question needs >= 2 ``expected_key_points`` (the claim-level completeness
points scored at eval time by
``lib/retrieval/answer_scoring.py::score_key_point_coverage``). For a frozen
gold set whose answerable questions are missing them, this module drafts
2-4 key points PER question from that question's relevant-passage bodies via the
license-clean LOCAL provider, and writes a PROPOSAL artifact — it NEVER mutates
``gold_set.json`` (the frozen set is the canonical eval pin; the orchestrator
applies the proposal after operator review).

LICENSE POSTURE — same as the rest of the gold-authoring surface: the key-point
TEXT is drafted by the local provider only (NEVER an Anthropic surface). This
module writes the prompt + the deterministic scaffolding (which questions to
draft for, which passage bodies to ground in, schema-legal shape) — never the
key-point content itself.

Selection rule (mirrors ``_metadata_completeness_issues`` /
``_is_refusal_question``): a question is in scope iff it is NOT a refusal-style
entry AND carries fewer than 2 ``expected_key_points``. The drafted points are
grounded ONLY in the question's own cited passage bodies (so the key points are
answerable from the corpus the question cites, not invented).

Decision capture: drafting IS an LLM call path — one ``gold_key_points_backfill``
decision is emitted after the batch with a dynamic rationale (question ids,
model, drafted/failed split). A regression test asserts the capture fires.

Output: ``retrieval_eval/gold_key_points_proposal.json``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.retrieval.gold_set import (
    RETRIEVAL_EVAL_SUBDIR,
    _load_chunks_by_id,
    load_gold_set,
)

GOLD_KEY_POINTS_PROPOSAL_FILENAME = "gold_key_points_proposal.json"
KEY_POINTS_BACKFILL_PROMPT_VERSION = "gold-keypoints.v1"

# Schema bounds: expected_key_points is 2-6; we target 2-4.
_KEY_POINTS_MIN = 2
_KEY_POINTS_MAX = 4
# Cap per-passage body chars put in front of the model so a long passage
# doesn't blow the serving window.
_PASSAGE_BODY_CAP = 1200

# Refusal-style categories exempt from the key-point requirement (mirror of
# gold_set._is_refusal_question — kept local so this module has no private-name
# import coupling to the loader).
_REFUSAL_CATEGORIES = frozenset(
    {"off_topic", "off_topic_llm", "adjacent_domain", "out_of_scope_detail"}
)


# ---------------------------------------------------------------- data types


@dataclass
class KeyPointsProposal:
    """The proposed expected_key_points for one gold question."""

    question_id: str
    question_text: str
    proposed_key_points: List[str] = field(default_factory=list)
    rationale: str = ""
    source: str = "local-model"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "proposed_key_points": list(self.proposed_key_points),
            "rationale": self.rationale,
            "source": self.source,
        }
        if self.error:
            out["error"] = self.error
        return out


# ---------------------------------------------------------------- selection


def _is_refusal_question(q: Dict[str, Any]) -> bool:
    if q.get("is_refusal") is True or q.get("unanswerable") is True:
        return True
    cat = q.get("category")
    return isinstance(cat, str) and cat in _REFUSAL_CATEGORIES


def _needs_key_points(q: Dict[str, Any]) -> bool:
    """A question is in scope iff non-refusal AND has < 2 expected_key_points."""
    if _is_refusal_question(q):
        return False
    kps = q.get("expected_key_points")
    return not (isinstance(kps, list) and len(kps) >= _KEY_POINTS_MIN)


def _passage_bodies(
    question: Dict[str, Any], chunks_by_id: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, str]]:
    """``(chunk_id, body)`` for each resolvable relevant passage of ``question``."""
    out: List[Tuple[str, str]] = []
    for p in question.get("relevant_passages") or []:
        if not isinstance(p, dict):
            continue
        cid = p.get("chunk_id")
        if isinstance(cid, str) and cid in chunks_by_id:
            body = str(chunks_by_id[cid].get("text") or "")
            if body.strip():
                out.append((cid, body[:_PASSAGE_BODY_CAP]))
    return out


# ---------------------------------------------------------------- drafting


def _key_points_prompt(
    question_text: str, passage_bodies: Sequence[Tuple[str, str]]
) -> Tuple[str, str]:
    """``(system, user)`` prompts asking the model to draft 2-4 key points for
    ``question_text`` grounded ONLY in the supplied passage bodies."""
    system = (
        "You list the key points a correct answer to a course question must "
        "state. Output JSON only: a JSON array of 2 to 4 short factual points "
        "(each a brief phrase or clause, not a full paragraph). Draw EVERY point "
        "only from the supplied source passages — do not add facts not present "
        "in them. The points are the claim-level completeness checklist for the "
        "answer."
    )
    bodies = "\n\n".join(
        f"[passage {i + 1} — chunk {cid}]\n{body}"
        for i, (cid, body) in enumerate(passage_bodies)
    )
    user = (
        f"Question:\n{question_text}\n\n"
        f"Source passages the answer must be grounded in:\n{bodies}\n\n"
        'Output JSON array only: ["key point one", "key point two", ...] '
        "(2 to 4 points)."
    )
    return system, user


def _parse_key_points(text: str) -> List[str]:
    """Parse a model response into a list of key-point strings. Tolerates
    markdown-fenced / object-wrapped JSON (same 7B-Q4 drift the local provider
    handles)."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    points: List[str] = []
    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Recover the first JSON array embedded in prose.
        start = raw.find("[")
        end = raw.rfind("]")
        if 0 <= start < end:
            try:
                parsed = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, list):
        points = [str(p).strip() for p in parsed if str(p).strip()]
    elif isinstance(parsed, dict):
        for key in ("expected_key_points", "key_points", "points", "result", "answer"):
            v = parsed.get(key)
            if isinstance(v, list):
                points = [str(p).strip() for p in v if str(p).strip()]
                break
        if not points:
            # Degenerate keys-as-labels object: the json_mode envelope (or the
            # 7B-Q4) re-emits the prompt's example array literal as OBJECT KEYS
            # ("key point one": "<real point>", ...). Harvest the string VALUES
            # (the real content); fall back to the keys if values are empty.
            values = [str(v).strip() for v in parsed.values() if str(v).strip()]
            if len(values) >= _KEY_POINTS_MIN:
                points = values
            else:
                points = [str(k).strip() for k in parsed.keys() if str(k).strip()]
    return [p for p in points if p][:_KEY_POINTS_MAX]


def draft_key_points(
    question: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
) -> Tuple[List[str], str, Optional[str]]:
    """Draft 2-4 key points for one question. Returns
    ``(points, rationale, error)``. ``error`` is non-None on a parse/short
    failure (and ``points`` is then empty so the proposal is recorded, not
    silently dropped)."""
    qid = str(question.get("question_id") or "")
    qtext = str(question.get("question_text") or "")
    bodies = _passage_bodies(question, chunks_by_id)
    if not bodies:
        return [], "", f"no resolvable relevant-passage bodies for {qid}"
    system, user = _key_points_prompt(qtext, bodies)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        text = client.chat_completion(messages, max_tokens=384, temperature=0.2)
        points = _parse_key_points(text)
    except Exception as exc:  # noqa: BLE001 — record, don't drop
        return [], "", f"draft failed: {exc}"
    if len(points) < _KEY_POINTS_MIN:
        return (
            points, "",
            f"model returned {len(points)} point(s) (< {_KEY_POINTS_MIN}); "
            f"needs operator authoring",
        )
    cited = ", ".join(cid for cid, _ in bodies)
    rationale = (
        f"{len(points)} key point(s) drafted from {len(bodies)} cited passage "
        f"body/bodies [{cited}] for {question.get('question_type', '?')} "
        f"question {qid}; grounded only in the question's own relevant passages."
    )
    return points, rationale, None


def build_key_points_proposals(
    gold: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    capture: Optional[Any] = None,
    model_id: Optional[str] = None,
) -> List[KeyPointsProposal]:
    """Draft a key-points proposal per in-scope question (non-refusal, < 2
    expected_key_points). One ``gold_key_points_backfill`` decision is emitted
    after the batch with a dynamic rationale. NEVER an Anthropic surface."""
    resolved_model = model_id or getattr(client, "model", "local")
    out: List[KeyPointsProposal] = []
    drafted_ok = 0
    for q in gold.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id")
        if not isinstance(qid, str) or not _needs_key_points(q):
            continue
        points, rationale, error = draft_key_points(q, chunks_by_id, client=client)
        if error is None:
            drafted_ok += 1
        out.append(
            KeyPointsProposal(
                question_id=qid,
                question_text=str(q.get("question_text") or ""),
                proposed_key_points=points,
                rationale=rationale,
                source=f"{resolved_model}/{KEY_POINTS_BACKFILL_PROMPT_VERSION}",
                error=error,
            )
        )

    if capture is not None and out:
        failures = len(out) - drafted_ok
        ids = ", ".join(p.question_id for p in out[:8])
        more = "" if len(out) <= 8 else f" (+{len(out) - 8} more)"
        try:
            capture.log_decision(
                decision_type="gold_key_points_backfill",
                decision=(
                    f"drafted expected_key_points for {drafted_ok}/{len(out)} "
                    f"gold question(s) via local model {resolved_model}; "
                    f"{failures} needing operator authoring."
                ),
                rationale=(
                    f"expected_key_points backfill over {len(out)} answerable "
                    f"gold question(s) missing >= 2 key points [{ids}{more}] using "
                    f"license-clean local model {resolved_model} (prompt "
                    f"{KEY_POINTS_BACKFILL_PROMPT_VERSION}); each point drafted "
                    f"strictly from the question's own cited relevant-passage "
                    f"bodies (no invented facts); drafted_ok={drafted_ok}, "
                    f"failures={failures}. PROPOSAL ONLY — never auto-applied to "
                    f"the frozen gold_set.json."
                ),
            )
        except Exception:  # pragma: no cover — defensive
            pass
    return out


# ---------------------------------------------------------------- doc/IO


def build_proposal_doc(
    course_slug: str, proposals: List[KeyPointsProposal], *, model_id: str
) -> Dict[str, Any]:
    drafted_ok = sum(1 for p in proposals if not p.error)
    return {
        "schema_version": "1.0",
        "course_slug": course_slug,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_operator_note": (
            "PROPOSAL ONLY — this artifact is never auto-applied to gold_set.json "
            "(the frozen set is the canonical eval pin). Review each proposed "
            "expected_key_points set, then the orchestrator applies it."
        ),
        "method": "local-model",
        "model_id": model_id,
        "prompt_version": KEY_POINTS_BACKFILL_PROMPT_VERSION,
        "n_questions": len(proposals),
        "n_drafted_ok": drafted_ok,
        "proposals": [p.to_dict() for p in proposals],
    }


def generate_key_points_proposal(
    course_dir: Path,
    *,
    client: Any,
    capture: Optional[Any] = None,
    model_id: Optional[str] = None,
    write: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end: load the gold set + pinned chunkset, draft key points for
    every in-scope question, and (unless ``write=False``) write
    ``retrieval_eval/gold_key_points_proposal.json``. Returns
    ``(doc, written_path)``.

    Raises ``FileNotFoundError`` when the gold set / pinned chunkset is absent
    (the passage bodies are the grounding source)."""
    course_dir = Path(course_dir)
    gold, _ = load_gold_set(course_dir, verify=False)
    if not gold:
        raise FileNotFoundError(f"no gold set found under {course_dir}.")
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path")
    if not chunks_rel:
        raise FileNotFoundError(
            f"no chunkset pin in gold set at {course_dir}; cannot ground key points."
        )
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise FileNotFoundError(f"pinned chunkset not found at {chunks_path}.")
    chunks_by_id = _load_chunks_by_id(chunks_path)

    proposals = build_key_points_proposals(
        gold, chunks_by_id, client=client, capture=capture, model_id=model_id
    )
    resolved_model = model_id or getattr(client, "model", "local")
    doc = build_proposal_doc(
        gold.get("course_slug", course_dir.name), proposals, model_id=resolved_model
    )

    out_path: Optional[Path] = None
    if write:
        out_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / GOLD_KEY_POINTS_PROPOSAL_FILENAME
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc, out_path


__all__ = [
    "GOLD_KEY_POINTS_PROPOSAL_FILENAME",
    "KEY_POINTS_BACKFILL_PROMPT_VERSION",
    "KeyPointsProposal",
    "draft_key_points",
    "build_key_points_proposals",
    "build_proposal_doc",
    "generate_key_points_proposal",
]
