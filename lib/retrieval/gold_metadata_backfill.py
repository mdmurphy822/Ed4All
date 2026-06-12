"""Gold-question metadata backfill PROPOSAL builder (retrieval-answer-eval-set).

Companion to the ``GOLD_QUESTION_METADATA_INCOMPLETE`` gold-validate warning
(``lib/retrieval/gold_set.py``). For a gold set whose questions are missing the
v1.1 ``difficulty`` / ``expected_citation_population`` metadata, this module
builds a PROPOSAL artifact — it NEVER mutates ``gold_set.json`` (the frozen set
is the canonical eval pin; silently editing it would break comparability).

The proposal is rule-based + deterministic (no LLM, no network — same posture as
``gold_set.py`` / ``gold_coverage.py``):

  * ``expected_citation_population`` is inferred from the question's
    ``relevant_passages`` chunk populations (``_population_of_chunk``): all
    ``source`` → ``source``; all ``course`` → ``course``; a mix → ``both``.
  * ``difficulty`` is inferred from the §1.3 heuristic (``_difficulty_heuristic``)
    over the passage count + the number of distinct content weeks the passages
    span.

Each proposed field carries a ``source`` tag (``rule`` here — the schema also
admits ``local-model`` if a future arm drafts the value) and a human ``rationale``
naming the signals used, so the operator can audit before applying.

Output: ``retrieval_eval/gold_metadata_backfill_proposal.json``. Apply is an
explicit, separate operator action (out of this module's scope by design).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.retrieval.gold_authoring import _difficulty_heuristic
from lib.retrieval.gold_coverage import _population_of_chunk, _week_of_item_path
from lib.retrieval.gold_set import (
    RETRIEVAL_EVAL_SUBDIR,
    _load_chunks_by_id,
    load_gold_set,
)

GOLD_METADATA_BACKFILL_FILENAME = "gold_metadata_backfill_proposal.json"


@dataclass
class FieldProposal:
    """One proposed value for one missing metadata field."""

    field: str
    proposed: str
    rationale: str
    source: str = "rule"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "proposed": self.proposed,
            "rationale": self.rationale,
            "source": self.source,
        }


@dataclass
class QuestionProposal:
    """All missing-field proposals for one gold question."""

    question_id: str
    question_text: str
    proposals: List[FieldProposal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "question_id": self.question_id,
            "question_text": self.question_text,
        }
        # Flatten the common (difficulty / population) proposals to top-level
        # keys for an at-a-glance review, while keeping the full per-field block.
        for p in self.proposals:
            if p.field == "difficulty":
                out["proposed_difficulty"] = p.proposed
            elif p.field == "expected_citation_population":
                out["proposed_population"] = p.proposed
        out["fields"] = [p.to_dict() for p in self.proposals]
        out["rationale"] = "; ".join(
            f"{p.field}={p.proposed} ({p.rationale})" for p in self.proposals
        )
        out["source"] = (
            "rule" if all(p.source == "rule" for p in self.proposals)
            else "mixed"
        )
        return out


def _passage_chunks(
    question: Dict[str, Any], chunks_by_id: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in question.get("relevant_passages") or []:
        if not isinstance(p, dict):
            continue
        cid = p.get("chunk_id")
        if isinstance(cid, str) and cid in chunks_by_id:
            out.append(chunks_by_id[cid])
    return out


def propose_population(
    chunks: List[Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    """Infer ``expected_citation_population`` from the passage chunk
    populations. Returns ``(value, rationale)`` or None when there are no
    resolvable passage chunks (can't infer — left to the operator)."""
    if not chunks:
        return None
    pops = {_population_of_chunk(c) for c in chunks}
    if pops == {"source"}:
        value = "source"
    elif pops == {"course"}:
        value = "course"
    else:
        value = "both"
    rationale = (
        f"{len(chunks)} relevant passage(s) classify as {sorted(pops)} via "
        f"_population_of_chunk; "
        + {
            "source": "all original-document chunks → source.",
            "course": "all generated course-page chunks → course.",
            "both": "a mix of source + course chunks → both.",
        }[value]
    )
    return value, rationale


def propose_difficulty(
    question: Dict[str, Any], chunks: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """Infer ``difficulty`` via the §1.3 heuristic over passage count + the
    number of distinct content weeks the passages span. Returns
    ``(value, rationale)``."""
    qtype = str(question.get("question_type") or "factual_recall")
    n_passages = len(question.get("relevant_passages") or [])
    weeks = set()
    for c in chunks:
        wk = _week_of_item_path((c.get("source") or {}).get("item_path", ""))
        if wk:
            weeks.add(wk)
    n_weeks = len(weeks)
    value = _difficulty_heuristic(qtype, n_passages, n_weeks)
    rationale = (
        f"§1.3 heuristic on question_type={qtype}, n_passages={n_passages}, "
        f"distinct_weeks={n_weeks} → {value}."
    )
    return value, rationale


def build_backfill_proposals(
    gold: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> List[QuestionProposal]:
    """Build a backfill proposal per question missing difficulty and/or
    expected_citation_population. Questions already carrying both are skipped
    (nothing to propose). Pure + deterministic."""
    out: List[QuestionProposal] = []
    for q in gold.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id")
        if not isinstance(qid, str):
            continue
        chunks = _passage_chunks(q, chunks_by_id)
        proposals: List[FieldProposal] = []

        has_difficulty = isinstance(q.get("difficulty"), str) and q["difficulty"].strip()
        if not has_difficulty:
            value, rationale = propose_difficulty(q, chunks)
            proposals.append(FieldProposal("difficulty", value, rationale))

        has_population = (
            isinstance(q.get("expected_citation_population"), str)
            and q["expected_citation_population"].strip()
        )
        if not has_population:
            inferred = propose_population(chunks)
            if inferred is not None:
                value, rationale = inferred
                proposals.append(
                    FieldProposal("expected_citation_population", value, rationale)
                )

        if proposals:
            out.append(
                QuestionProposal(
                    question_id=qid,
                    question_text=str(q.get("question_text") or ""),
                    proposals=proposals,
                )
            )
    return out


def build_backfill_doc(
    course_slug: str, proposals: List[QuestionProposal]
) -> Dict[str, Any]:
    """Assemble the ``gold_metadata_backfill_proposal.json`` wrapper doc."""
    return {
        "schema_version": "1.0",
        "course_slug": course_slug,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_operator_note": (
            "PROPOSAL ONLY — this artifact is never auto-applied to gold_set.json "
            "(the frozen set is the canonical eval pin). Review each proposed "
            "difficulty / expected_citation_population, then apply by hand / via a "
            "separate explicit step."
        ),
        "method": "rule-based",
        "n_questions": len(proposals),
        "proposals": [p.to_dict() for p in proposals],
    }


def generate_backfill_proposal(
    course_dir: Path, *, write: bool = True
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end: load the gold set + its pinned chunkset, build the backfill
    proposals, and (unless ``write=False``) write
    ``retrieval_eval/gold_metadata_backfill_proposal.json``. Returns
    ``(doc, written_path)``.

    Raises ``FileNotFoundError`` when the gold set / pinned chunkset is absent
    (the chunkset is the source of truth for population inference)."""
    course_dir = Path(course_dir)
    gold, _ = load_gold_set(course_dir, verify=False)
    if not gold:
        raise FileNotFoundError(f"no gold set found under {course_dir}.")
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path")
    if not chunks_rel:
        raise FileNotFoundError(
            f"no chunkset pin in gold set at {course_dir}; cannot infer population."
        )
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise FileNotFoundError(f"pinned chunkset not found at {chunks_path}.")
    chunks_by_id = _load_chunks_by_id(chunks_path)

    proposals = build_backfill_proposals(gold, chunks_by_id)
    doc = build_backfill_doc(gold.get("course_slug", course_dir.name), proposals)

    out_path: Optional[Path] = None
    if write:
        out_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / GOLD_METADATA_BACKFILL_FILENAME
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc, out_path


__all__ = [
    "GOLD_METADATA_BACKFILL_FILENAME",
    "FieldProposal",
    "QuestionProposal",
    "propose_population",
    "propose_difficulty",
    "build_backfill_proposals",
    "build_backfill_doc",
    "generate_backfill_proposal",
]
