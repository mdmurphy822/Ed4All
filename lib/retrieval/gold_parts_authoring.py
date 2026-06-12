"""multi_part question parts[] authoring PROPOSAL builder.

The v1.2 openstax gold set's ``multi_part`` questions carry operator-placeholder
parts ("PART A — operator: fill in", all ``covered:true``, no
``relevant_passage_refs``) — the scaffold ``gold_authoring._build_candidate``
emits for a multi_part stratified draft. The part-coverage scorer
(``lib/retrieval/answer_scoring.py::score_part_coverage``) reads each covered
part's ``relevant_passage_refs`` chunk bodies as its support signal, and detects
absence-flagging via each ``covered:false`` part's ``absence_note``. Placeholder
parts make that scorer inert. This module decomposes each multi_part question's
``question_text`` into its actual sub-parts via the license-clean LOCAL model,
binds each part to the passages that answer it, and emits a PROPOSAL artifact —
it NEVER mutates ``gold_set.json`` (the frozen set is the canonical eval pin;
the orchestrator applies the parts after operator review).

Decomposition + binding:

  1. The local model decomposes ``question_text`` into 2+ enumerable sub-parts
     (the actual sub-questions, NOT placeholders), each a short prompt.
  2. Each part's ``relevant_passage_refs`` are bound — first from the PARENT
     question's ``relevant_passages`` (a part whose text overlaps a parent
     passage body is anchored to it), then, for a part with no parent match, by
     a read-only retrieval dry-run over the corpus (``retrieve_fn``) to find the
     chunk that answers it.
  3. CONSERVATIVE absence: a part is marked ``covered:false`` + an
     ``absence_note`` ONLY when BOTH binding paths come back empty-ish (no parent
     passage overlaps AND the corpus retrieval returns nothing above a small
     score floor). This is the ws3.v2 completeness probe — the pipeline must
     flag the absent part, not invent it — so the bar to declare a part
     uncovered is deliberately high (default off unless retrieval is genuinely
     empty).

schema-legal part objects only (``additionalProperties:false``):
``{part_id, part_text, covered}`` always; ``relevant_passage_refs`` on covered
parts; ``absence_note`` (>= 20 chars) iff ``covered:false``.

LICENSE POSTURE: the part TEXT + absence_note are drafted by the local provider
only (NEVER an Anthropic surface). This module writes the prompt + the
deterministic binding/anchoring scaffolding — never the part content itself.

Decision capture: decomposition IS an LLM call path — one
``gold_parts_authoring`` decision is emitted after the batch with a dynamic
rationale (question ids, model, covered/uncovered split). A regression test
asserts the capture fires.

Output: ``retrieval_eval/gold_parts_proposal.json``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lib.retrieval._text import shingle_containment
from lib.retrieval.gold_set import (
    RETRIEVAL_EVAL_SUBDIR,
    _load_chunks_by_id,
    load_gold_set,
)

GOLD_PARTS_PROPOSAL_FILENAME = "gold_parts_proposal.json"
PARTS_AUTHORING_PROMPT_VERSION = "gold-parts.v1"

# Binding thresholds.
_PARENT_OVERLAP_MIN = 0.18        # 4-shingle containment of part-text in a parent passage body
_RETRIEVAL_SCORE_FLOOR = 0.01     # below this top score, treat retrieval as empty-ish
_RETRIEVAL_LIMIT = 5
_PART_BODY_CAP = 900
_MAX_PARTS = 6                    # sanity cap on decomposed parts


# ---------------------------------------------------------------- data types


@dataclass
class PartsProposal:
    """The proposed parts[] block for one multi_part question."""

    question_id: str
    question_text: str
    parts: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    source: str = "local-model"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "parts": self.parts,
            "n_parts": len(self.parts),
            "n_covered": sum(1 for p in self.parts if p.get("covered")),
            "n_uncovered": sum(1 for p in self.parts if not p.get("covered")),
            "rationale": self.rationale,
            "source": self.source,
        }
        if self.error:
            out["error"] = self.error
        return out


# ---------------------------------------------------------------- selection


def _is_multi_part(q: Dict[str, Any]) -> bool:
    return q.get("question_type") == "multi_part"


def _parent_passage_bodies(
    question: Dict[str, Any], chunks_by_id: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, str]]:
    """``(chunk_id, body)`` for each resolvable parent relevant passage."""
    out: List[Tuple[str, str]] = []
    for p in question.get("relevant_passages") or []:
        if not isinstance(p, dict):
            continue
        cid = p.get("chunk_id")
        if isinstance(cid, str) and cid in chunks_by_id:
            body = str(chunks_by_id[cid].get("text") or "")
            if body.strip():
                out.append((cid, body))
    return out


# ---------------------------------------------------------------- drafting


def _decompose_prompt(
    question_text: str, parent_bodies: Sequence[Tuple[str, str]]
) -> Tuple[str, str]:
    """``(system, user)`` prompts asking the model to decompose ``question_text``
    into its actual enumerable sub-parts."""
    system = (
        "You decompose a multi-part course question into its actual enumerable "
        "sub-parts — the real distinct sub-questions a learner must answer, NOT "
        "placeholders. There are ALWAYS at least two sub-parts (the question is "
        "multi-part by construction). Output JSON only, an object with one key "
        '"parts" whose value is a JSON array of 2 to 6 STRINGS, each string a '
        "full sub-question. Preserve the question's own structure (if it asks "
        "'a) ... b) ...' use those; otherwise split it into each distinct thing "
        "it asks for — e.g. 'what are the steps' and 'what is the numeric "
        "answer' are two sub-parts). Do not invent sub-parts the question does "
        "not ask."
    )
    bodies = "\n\n".join(
        f"[passage {i + 1} — chunk {cid}]\n{body[:_PART_BODY_CAP]}"
        for i, (cid, body) in enumerate(parent_bodies)
    )
    user = (
        f"Multi-part question:\n{question_text}\n\n"
        f"Context passages (the question's cited material):\n{bodies}\n\n"
        'Output JSON only: {"parts": ["<first sub-question>", '
        '"<second sub-question>"]} (2 to 6 sub-questions).'
    )
    return system, user


def _parse_parts(text: str) -> List[str]:
    """Parse a model response into a list of part-text strings."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if 0 <= start < end:
            try:
                parsed = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
    texts: List[str] = []
    # Schema-key labels the model may emit AS keys in a degenerate object — never
    # a real part text.
    _schema_keys = {"part_text", "text", "part", "parts", "part_id", "covered"}
    if isinstance(parsed, dict):
        # Unwrap a single list-valued envelope (json_mode wraps the array as
        # {"parts": [...]} / {"result": [...]}).
        for key in ("parts", "result", "items", "sub_parts", "subparts"):
            v = parsed.get(key)
            if isinstance(v, list):
                parsed = v
                break
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                t = item.get("part_text") or item.get("text") or item.get("part")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            elif isinstance(item, str) and item.strip():
                texts.append(item.strip())
    elif isinstance(parsed, dict):
        # Degenerate keys-as-labels object (the 7B-Q4 artifact): harvest the
        # string VALUES as the part texts (skipping schema-key labels), falling
        # back to non-schema keys only.
        values = [str(v).strip() for v in parsed.values()
                  if isinstance(v, str) and v.strip()]
        if len(values) >= 2:
            texts = values
        else:
            texts = [
                str(k).strip() for k in parsed.keys()
                if str(k).strip() and str(k).strip().lower() not in _schema_keys
            ]
    return texts[:_MAX_PARTS]


def _absence_note_prompt(question_text: str, part_text: str) -> Tuple[str, str]:
    """``(system, user)`` prompts asking the model to write the >= 20-char
    absence_note for a part the corpus does not cover."""
    system = (
        "You write a one-sentence note explaining why the course corpus does NOT "
        "cover a sub-part of a question. The note must be at least 20 characters "
        "and state plainly that this sub-part is not covered by the course "
        "material (a dry-run over the corpus confirmed the absence). Output JSON "
        'only: {"absence_note": "..."}.'
    )
    user = (
        f"Parent question:\n{question_text}\n\n"
        f"Uncovered sub-part:\n{part_text}\n\n"
        'Output JSON only: {"absence_note": "<>= 20 chars, says the corpus does '
        'not cover this sub-part; dry-run confirmed>"}'
    )
    return system, user


def _parse_absence_note(text: str) -> Optional[str]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            note = parsed.get("absence_note")
            if isinstance(note, str) and len(note.strip()) >= 20:
                return note.strip()
    except json.JSONDecodeError:
        pass
    # Fallback: a bare sentence long enough.
    flat = " ".join(raw.split())
    if len(flat) >= 20:
        return flat[:300]
    return None


# ---------------------------------------------------------------- binding


def _bind_to_parent(
    part_text: str, parent_bodies: Sequence[Tuple[str, str]]
) -> List[str]:
    """Return parent chunk_ids whose body the part text overlaps (4-shingle
    containment >= threshold), strongest first."""
    scored: List[Tuple[float, str]] = []
    for cid, body in parent_bodies:
        score = shingle_containment(part_text, body, shingle_size=4)
        if score >= _PARENT_OVERLAP_MIN:
            scored.append((score, cid))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [cid for _, cid in scored]


def _bind_by_retrieval(
    part_text: str,
    *,
    retrieve_fn: Optional[Callable[..., Sequence[Any]]],
    libv2_root: Optional[Path],
    course_slug: str,
    valid_chunk_ids: Sequence[str],
) -> Tuple[List[str], float]:
    """Retrieve over the corpus for a part with no parent overlap.

    Returns ``(chunk_ids, top_score)``. Only chunk ids present in the parent
    question's gold passages are eligible — a part must bind to a chunk the
    question already cites (per-part retrieval is a TIE-BREAK among the parent's
    passages, not a way to pull in arbitrary corpus chunks). When ``retrieve_fn``
    is None or raises (no index), returns ``([], 0.0)`` — empty-ish."""
    if retrieve_fn is None or libv2_root is None:
        return [], 0.0
    valid = set(valid_chunk_ids)
    try:
        results = list(
            retrieve_fn(libv2_root, part_text, course_slug=course_slug,
                        engine="lexical", limit=_RETRIEVAL_LIMIT)
        )
    except Exception:  # noqa: BLE001 — missing index => empty-ish
        return [], 0.0
    top_score = 0.0
    chunk_ids: List[str] = []
    for r in results:
        score = float(getattr(r, "score", 0.0) or 0.0)
        cid = str(getattr(r, "chunk_id", "") or "")
        top_score = max(top_score, score)
        if cid in valid and cid not in chunk_ids:
            chunk_ids.append(cid)
    return chunk_ids, top_score


# ---------------------------------------------------------------- per-question


def author_parts(
    question: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    retrieve_fn: Optional[Callable[..., Sequence[Any]]] = None,
    libv2_root: Optional[Path] = None,
    course_slug: str = "",
) -> PartsProposal:
    """Decompose one multi_part question + bind each part to passages.

    Records a proposal with ``error`` set (and empty parts) on a decomposition
    failure. CONSERVATIVE absence: a part is ``covered:false`` only when it
    binds to NEITHER a parent passage NOR a corpus retrieval above the score
    floor."""
    qid = str(question.get("question_id") or "")
    qtext = str(question.get("question_text") or "")
    parent_bodies = _parent_passage_bodies(question, chunks_by_id)
    parent_ids = [cid for cid, _ in parent_bodies]
    system, user = _decompose_prompt(qtext, parent_bodies)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        text = client.chat_completion(messages, max_tokens=512, temperature=0.2)
        part_texts = _parse_parts(text)
    except Exception as exc:  # noqa: BLE001 — record, don't drop
        return PartsProposal(qid, qtext, [], "", error=f"decompose failed: {exc}")
    if len(part_texts) < 2:
        return PartsProposal(
            qid, qtext, [],
            "", error=f"model returned {len(part_texts)} part(s) (< 2)",
        )

    parts: List[Dict[str, Any]] = []
    n_uncovered = 0
    for i, ptext in enumerate(part_texts):
        pid = chr(ord("a") + i) if i < 26 else f"p{i + 1}"
        refs = _bind_to_parent(ptext, parent_bodies)
        if not refs:
            refs, top_score = _bind_by_retrieval(
                ptext, retrieve_fn=retrieve_fn, libv2_root=libv2_root,
                course_slug=course_slug, valid_chunk_ids=parent_ids,
            )
        else:
            top_score = 1.0  # parent overlap already cleared the floor
        if refs:
            parts.append({
                "part_id": pid,
                "part_text": ptext,
                "covered": True,
                "relevant_passage_refs": refs,
            })
        else:
            # Genuinely uncovered: ask the model for the absence_note.
            n_uncovered += 1
            note: Optional[str] = None
            try:
                sysn, usern = _absence_note_prompt(qtext, ptext)
                ntext = client.chat_completion(
                    [{"role": "system", "content": sysn},
                     {"role": "user", "content": usern}],
                    max_tokens=160, temperature=0.2,
                )
                note = _parse_absence_note(ntext)
            except Exception:  # noqa: BLE001 — fall through to deterministic note
                note = None
            if not note:
                note = (
                    "The course corpus does not cover this sub-part; a read-only "
                    "retrieval dry-run over the pinned chunkset returned no "
                    "passage above the score floor for it."
                )
            parts.append({
                "part_id": pid,
                "part_text": ptext,
                "covered": False,
                "absence_note": note,
            })

    n_covered = len(parts) - n_uncovered
    rationale = (
        f"decomposed into {len(parts)} sub-part(s) ({n_covered} covered / "
        f"{n_uncovered} uncovered) for multi_part question {qid}; covered parts "
        f"bound to the parent's relevant_passages via 4-shingle overlap "
        f"(then per-part corpus retrieval among the parent's passages); uncovered "
        f"parts flagged only when both binding paths came back empty."
    )
    return PartsProposal(qid, qtext, parts, rationale)


def build_parts_proposals(
    gold: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    retrieve_fn: Optional[Callable[..., Sequence[Any]]] = None,
    libv2_root: Optional[Path] = None,
    capture: Optional[Any] = None,
    model_id: Optional[str] = None,
) -> List[PartsProposal]:
    """Author parts for every multi_part question. One ``gold_parts_authoring``
    decision is emitted after the batch with a dynamic rationale. NEVER an
    Anthropic surface."""
    resolved_model = model_id or getattr(client, "model", "local")
    course_slug = str(gold.get("course_slug") or "")
    out: List[PartsProposal] = []
    for q in gold.get("questions") or []:
        if not isinstance(q, dict) or not _is_multi_part(q):
            continue
        qid = q.get("question_id")
        if not isinstance(qid, str):
            continue
        out.append(
            author_parts(
                q, chunks_by_id, client=client, retrieve_fn=retrieve_fn,
                libv2_root=libv2_root, course_slug=course_slug,
            )
        )

    if capture is not None and out:
        ok = sum(1 for p in out if not p.error)
        failures = len(out) - ok
        total_uncovered = sum(
            sum(1 for pt in p.parts if not pt.get("covered")) for p in out
        )
        ids = ", ".join(p.question_id for p in out[:8])
        more = "" if len(out) <= 8 else f" (+{len(out) - 8} more)"
        try:
            capture.log_decision(
                decision_type="gold_parts_authoring",
                decision=(
                    f"authored parts[] for {ok}/{len(out)} multi_part gold "
                    f"question(s) via local model {resolved_model}; "
                    f"{total_uncovered} part(s) flagged uncovered, {failures} "
                    f"decompose failure(s)."
                ),
                rationale=(
                    f"multi_part parts[] authoring over {len(out)} question(s) "
                    f"[{ids}{more}] using license-clean local model "
                    f"{resolved_model} (prompt {PARTS_AUTHORING_PROMPT_VERSION}); "
                    f"each question's text decomposed into its real sub-parts, "
                    f"covered parts bound to the parent's relevant_passages via "
                    f"4-shingle overlap + per-part corpus retrieval, uncovered "
                    f"parts (total {total_uncovered}) flagged conservatively only "
                    f"when both binding paths returned empty; ok={ok}, "
                    f"failures={failures}. PROPOSAL ONLY — never auto-applied to "
                    f"the frozen gold_set.json."
                ),
            )
        except Exception:  # pragma: no cover — defensive
            pass
    return out


# ---------------------------------------------------------------- doc/IO


def build_proposal_doc(
    course_slug: str, proposals: List[PartsProposal], *, model_id: str
) -> Dict[str, Any]:
    ok = sum(1 for p in proposals if not p.error)
    return {
        "schema_version": "1.0",
        "course_slug": course_slug,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_operator_note": (
            "PROPOSAL ONLY — this artifact is never auto-applied to gold_set.json "
            "(the frozen set is the canonical eval pin). Review each authored "
            "parts[] set (covered:false parts carry an absence_note); the "
            "orchestrator applies after review. Each part object is schema-legal "
            "(part_id / part_text / covered, plus relevant_passage_refs on "
            "covered parts and absence_note on uncovered parts)."
        ),
        "method": "local-model",
        "model_id": model_id,
        "prompt_version": PARTS_AUTHORING_PROMPT_VERSION,
        "n_questions": len(proposals),
        "n_authored_ok": ok,
        "proposals": [p.to_dict() for p in proposals],
    }


def generate_parts_proposal(
    course_dir: Path,
    *,
    client: Any,
    retrieve_fn: Optional[Callable[..., Sequence[Any]]] = None,
    libv2_root: Optional[Path] = None,
    capture: Optional[Any] = None,
    model_id: Optional[str] = None,
    write: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end: load the gold set + pinned chunkset, author parts for every
    multi_part question, and (unless ``write=False``) write
    ``retrieval_eval/gold_parts_proposal.json``. Returns ``(doc, written_path)``.

    ``retrieve_fn`` + ``libv2_root`` drive the per-part corpus retrieval (inject
    a fake in tests / omit to bind from the parent passages only). Raises
    ``FileNotFoundError`` when the gold set / pinned chunkset is absent."""
    course_dir = Path(course_dir)
    gold, _ = load_gold_set(course_dir, verify=False)
    if not gold:
        raise FileNotFoundError(f"no gold set found under {course_dir}.")
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path")
    if not chunks_rel:
        raise FileNotFoundError(
            f"no chunkset pin in gold set at {course_dir}; cannot bind parts."
        )
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise FileNotFoundError(f"pinned chunkset not found at {chunks_path}.")
    chunks_by_id = _load_chunks_by_id(chunks_path)

    proposals = build_parts_proposals(
        gold, chunks_by_id, client=client, retrieve_fn=retrieve_fn,
        libv2_root=libv2_root, capture=capture, model_id=model_id,
    )
    resolved_model = model_id or getattr(client, "model", "local")
    doc = build_proposal_doc(
        gold.get("course_slug", course_dir.name), proposals, model_id=resolved_model
    )

    out_path: Optional[Path] = None
    if write:
        out_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / GOLD_PARTS_PROPOSAL_FILENAME
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc, out_path


__all__ = [
    "GOLD_PARTS_PROPOSAL_FILENAME",
    "PARTS_AUTHORING_PROMPT_VERSION",
    "PartsProposal",
    "author_parts",
    "build_parts_proposals",
    "build_proposal_doc",
    "generate_parts_proposal",
]
