"""Gold-question difficulty REGRADE proposal builder.

The v1.2 openstax gold set is difficulty-skewed: the seed/template arms tagged
nearly every question ``easy`` (the §1.3 heuristic defaults a single-passage
question to ``easy``), so the coverage report's difficulty bands
(``gold_coverage.py::_DIFFICULTY_TARGET`` easy/medium/hard ~ 40/40/20, ±15) flag
``COVERAGE_DIFFICULTY_SKEWED``. This module re-grades the over-represented
``easy`` questions via the license-clean LOCAL model against a WRITTEN RUBRIC,
and emits a PROPOSAL artifact — it NEVER mutates ``gold_set.json`` (the frozen
set is the canonical eval pin; the orchestrator applies the regrade after
operator review).

RUBRIC (the model is asked to grade each question against THIS rubric; the
rationale must cite the question's own features):

  * **easy**  — single-passage recall / lookup: a verbatim or definitional fact
    answerable from ONE chunk with no procedure or synthesis.
  * **medium** — a multi-step procedure (an ordered method worked within a
    section) OR a synthesis WITHIN a single section/passage (combine 2+ facts
    that live together).
  * **hard**  — cross-section synthesis (the answer joins material from >= 2
    distinct sections/weeks), a multi_part question, or an error-analysis /
    "why is this wrong / what's the misconception" question.

HONEST DISTRIBUTION CONTRACT: this is a re-grade, NOT a quota filler. The model
grades each question on its merits; the tool does not force a question up to
``medium``/``hard`` to hit a band. A question the model judges genuinely easy
keeps ``easy`` (``proposed == current``), and that is recorded. The proposed
resulting distribution is reported alongside the bands so the operator sees
whether the honest regrade closes the skew or whether the gold set genuinely
needs more medium/hard AUTHORING (a separate step).

Only ``easy`` questions are in scope (the skew is the easy over-fill; the 1
medium + 11 hard are left untouched). The model sees the question text, type,
passage count, and the distinct content-week count of its cited passages (the
same deterministic signals the §1.3 heuristic uses) plus the passage bodies, so
its grade is grounded.

LICENSE POSTURE: the rationale TEXT is drafted by the local provider only (NEVER
an Anthropic surface). This module writes the rubric + prompt + the
deterministic feature scaffolding — never the grade rationale itself.

Decision capture: regrading IS an LLM call path — one ``gold_difficulty_regrade``
decision is emitted after the batch with a dynamic rationale (regraded counts,
model, before/after distribution). A regression test asserts the capture fires.

Output: ``retrieval_eval/gold_difficulty_regrade_proposal.json``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.retrieval.gold_coverage import _week_of_item_path
from lib.retrieval.gold_set import (
    RETRIEVAL_EVAL_SUBDIR,
    _load_chunks_by_id,
    load_gold_set,
)

GOLD_DIFFICULTY_REGRADE_PROPOSAL_FILENAME = "gold_difficulty_regrade_proposal.json"
DIFFICULTY_REGRADE_PROMPT_VERSION = "gold-regrade.v1"

_DIFFICULTIES = ("easy", "medium", "hard")
# Cap per-passage body chars put in front of the model.
_PASSAGE_BODY_CAP = 900
# Coverage-report difficulty bands (mirror of gold_coverage so the proposal can
# REPORT the proposed distribution vs the bands without importing private names
# the operator can't see in this artifact).
_DIFFICULTY_TARGET = {"easy": 0.40, "medium": 0.40, "hard": 0.20}
_DIFFICULTY_TOLERANCE = 0.15


# ---------------------------------------------------------------- data types


@dataclass
class RegradeRow:
    """One regrade verdict for one question."""

    question_id: str
    question_text: str
    current: str
    proposed: str
    rationale: str
    error: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.error is None and self.proposed != self.current

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "current": self.current,
            "proposed": self.proposed,
            "rationale": self.rationale,
            "changed": self.changed,
        }
        if self.error:
            out["error"] = self.error
        return out


# ---------------------------------------------------------------- features


def _passage_features(
    question: Dict[str, Any], chunks_by_id: Dict[str, Dict[str, Any]]
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """``(n_passages, n_distinct_weeks, [(chunk_id, body), ...])`` for the
    question's resolvable relevant passages — the deterministic signals the
    §1.3 heuristic uses, surfaced to the model alongside the bodies."""
    passages = [p for p in (question.get("relevant_passages") or []) if isinstance(p, dict)]
    weeks: set = set()
    bodies: List[Tuple[str, str]] = []
    for p in passages:
        cid = p.get("chunk_id")
        anchor = p.get("anchor") or {}
        wk = _week_of_item_path(str(anchor.get("item_path") or ""))
        if wk:
            weeks.add(wk)
        if isinstance(cid, str) and cid in chunks_by_id:
            chunk = chunks_by_id[cid]
            cw = _week_of_item_path((chunk.get("source") or {}).get("item_path", ""))
            if cw:
                weeks.add(cw)
            body = str(chunk.get("text") or "")
            if body.strip():
                bodies.append((cid, body[:_PASSAGE_BODY_CAP]))
    return len(passages), len(weeks), bodies


# ---------------------------------------------------------------- drafting


_RUBRIC_TEXT = (
    "RUBRIC:\n"
    "- easy: single-passage recall or lookup — a verbatim or definitional fact "
    "answerable from ONE chunk, no procedure, no synthesis.\n"
    "- medium: a multi-step procedure worked within a section, OR a synthesis "
    "WITHIN a single section (combine 2+ facts that live together).\n"
    "- hard: cross-section synthesis (answer joins >= 2 distinct sections/weeks), "
    "a multi-part question, or an error-analysis / misconception question."
)


def _regrade_prompt(
    question_text: str,
    question_type: str,
    n_passages: int,
    n_weeks: int,
    passage_bodies: Sequence[Tuple[str, str]],
) -> Tuple[str, str]:
    """``(system, user)`` prompts asking the model to grade ONE question against
    the rubric and justify with the question's own features."""
    system = (
        "You grade the difficulty of a course evaluation question against a "
        "fixed rubric. Output JSON only with keys: difficulty (exactly one of "
        '"easy", "medium", or "hard") and rationale (one sentence that cites the '
        "question's own features — its type, how many passages/sections it "
        "spans, and whether answering needs a procedure, a within-section "
        "synthesis, or a cross-section synthesis). Grade HONESTLY on the merits; "
        "do not inflate difficulty. " + _RUBRIC_TEXT
    )
    bodies = "\n\n".join(
        f"[passage {i + 1} — chunk {cid}]\n{body}"
        for i, (cid, body) in enumerate(passage_bodies)
    )
    user = (
        f"Question:\n{question_text}\n\n"
        f"Features: question_type={question_type}, "
        f"relevant_passage_count={n_passages}, distinct_content_weeks={n_weeks}.\n\n"
        f"Cited passages:\n{bodies}\n\n"
        'Output JSON only: {"difficulty": "easy|medium|hard", "rationale": "..."}'
    )
    return system, user


def _parse_regrade(text: str) -> Tuple[Optional[str], str]:
    """Parse a model response into ``(difficulty, rationale)``. Returns
    ``(None, "")`` when no valid difficulty is recoverable."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, dict):
        diff = str(parsed.get("difficulty") or "").strip().lower()
        rationale = str(parsed.get("rationale") or "").strip()
        if diff in _DIFFICULTIES:
            return diff, rationale
    # Bare-word fallback: scan for a difficulty token.
    m = re.search(r"\b(easy|medium|hard)\b", raw.lower())
    if m:
        return m.group(1), raw.strip()[:300]
    return None, ""


def regrade_question(
    question: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
) -> RegradeRow:
    """Regrade one ``easy`` question against the rubric. Records a row with
    ``error`` set (and ``proposed == current``) on a parse/draft failure."""
    qid = str(question.get("question_id") or "")
    qtext = str(question.get("question_text") or "")
    qtype = str(question.get("question_type") or "factual_recall")
    current = str(question.get("difficulty") or "easy")
    n_passages, n_weeks, bodies = _passage_features(question, chunks_by_id)
    system, user = _regrade_prompt(qtext, qtype, n_passages, n_weeks, bodies)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        text = client.chat_completion(messages, max_tokens=256, temperature=0.1)
        proposed, rationale = _parse_regrade(text)
    except Exception as exc:  # noqa: BLE001 — record, don't drop
        return RegradeRow(qid, qtext, current, current, "", error=f"draft failed: {exc}")
    if proposed is None:
        return RegradeRow(
            qid, qtext, current, current, "",
            error="model returned no parseable difficulty",
        )
    if not rationale:
        rationale = (
            f"graded {proposed} against the rubric for a {qtype} question "
            f"spanning {n_passages} passage(s) / {n_weeks} week(s)."
        )
    return RegradeRow(qid, qtext, current, proposed, rationale)


def build_regrade_rows(
    gold: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    capture: Optional[Any] = None,
    model_id: Optional[str] = None,
    regrade_from: str = "easy",
) -> List[RegradeRow]:
    """Regrade every ``regrade_from`` (default ``easy``) question against the
    rubric. One ``gold_difficulty_regrade`` decision is emitted after the batch
    with a dynamic rationale (before/after distribution). NEVER an Anthropic
    surface."""
    resolved_model = model_id or getattr(client, "model", "local")
    rows: List[RegradeRow] = []
    for q in gold.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id")
        if not isinstance(qid, str):
            continue
        if str(q.get("difficulty") or "easy") != regrade_from:
            continue
        rows.append(regrade_question(q, chunks_by_id, client=client))

    if capture is not None and rows:
        changed = sum(1 for r in rows if r.changed)
        failures = sum(1 for r in rows if r.error)
        before, after = projected_distribution(gold, rows)
        try:
            capture.log_decision(
                decision_type="gold_difficulty_regrade",
                decision=(
                    f"regraded {len(rows)} '{regrade_from}' gold question(s) via "
                    f"local model {resolved_model}; {changed} reclassified, "
                    f"{failures} failure(s)."
                ),
                rationale=(
                    f"Difficulty regrade over {len(rows)} '{regrade_from}' gold "
                    f"question(s) against the easy/medium/hard rubric using "
                    f"license-clean local model {resolved_model} (prompt "
                    f"{DIFFICULTY_REGRADE_PROMPT_VERSION}); each grade is "
                    f"feature-grounded (type + passage/week counts + bodies) and "
                    f"HONEST (no quota filling). Full-set distribution before="
                    f"{before} -> after={after}; changed={changed}, "
                    f"failures={failures}. PROPOSAL ONLY — never auto-applied to "
                    f"the frozen gold_set.json."
                ),
            )
        except Exception:  # pragma: no cover — defensive
            pass
    return rows


# ---------------------------------------------------------------- distribution


def _current_distribution(gold: Dict[str, Any]) -> Dict[str, int]:
    dist = {d: 0 for d in _DIFFICULTIES}
    for q in gold.get("questions") or []:
        if isinstance(q, dict):
            d = str(q.get("difficulty") or "").strip().lower()
            if d in dist:
                dist[d] += 1
    return dist


def projected_distribution(
    gold: Dict[str, Any], rows: Sequence[RegradeRow]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return ``(before, after)`` full-set difficulty distributions: ``before``
    is the gold set as-is; ``after`` applies every (non-error) proposed regrade.
    Pure — does not touch the gold set."""
    before = _current_distribution(gold)
    after = dict(before)
    for r in rows:
        if r.error or r.proposed == r.current:
            continue
        if r.current in after:
            after[r.current] -= 1
        if r.proposed in after:
            after[r.proposed] += 1
    return before, after


def _band_report(dist: Dict[str, int]) -> Dict[str, Any]:
    """Per-difficulty fraction + whether it sits inside the coverage band."""
    total = sum(dist.values()) or 1
    out: Dict[str, Any] = {}
    for d in _DIFFICULTIES:
        frac = dist.get(d, 0) / total
        target = _DIFFICULTY_TARGET[d]
        out[d] = {
            "count": dist.get(d, 0),
            "fraction": round(frac, 4),
            "band": [round(max(0.0, target - _DIFFICULTY_TOLERANCE), 2),
                     round(target + _DIFFICULTY_TOLERANCE, 2)],
            "in_band": abs(frac - target) <= _DIFFICULTY_TOLERANCE,
        }
    return out


# ---------------------------------------------------------------- doc/IO


def build_proposal_doc(
    course_slug: str,
    gold: Dict[str, Any],
    rows: List[RegradeRow],
    *,
    model_id: str,
    regrade_from: str = "easy",
) -> Dict[str, Any]:
    before, after = projected_distribution(gold, rows)
    changed = sum(1 for r in rows if r.changed)
    failures = sum(1 for r in rows if r.error)
    return {
        "schema_version": "1.0",
        "course_slug": course_slug,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_operator_note": (
            "PROPOSAL ONLY — this artifact is never auto-applied to gold_set.json "
            "(the frozen set is the canonical eval pin). Each row is an HONEST "
            "rubric regrade (no quota filling); the orchestrator applies after "
            "review. If the proposed distribution still falls outside the bands, "
            "the gap is real and needs more medium/hard question AUTHORING, not "
            "regrading."
        ),
        "method": "local-model",
        "model_id": model_id,
        "prompt_version": DIFFICULTY_REGRADE_PROMPT_VERSION,
        "rubric": _RUBRIC_TEXT,
        "regrade_scope": regrade_from,
        "n_regraded": len(rows),
        "n_changed": changed,
        "n_failures": failures,
        "distribution": {
            "before": before,
            "after": after,
            "before_bands": _band_report(before),
            "after_bands": _band_report(after),
        },
        "rows": [r.to_dict() for r in rows],
    }


def generate_difficulty_regrade_proposal(
    course_dir: Path,
    *,
    client: Any,
    capture: Optional[Any] = None,
    model_id: Optional[str] = None,
    regrade_from: str = "easy",
    write: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end: load the gold set + pinned chunkset, regrade every
    ``regrade_from`` question, and (unless ``write=False``) write
    ``retrieval_eval/gold_difficulty_regrade_proposal.json``. Returns
    ``(doc, written_path)``.

    Raises ``FileNotFoundError`` when the gold set / pinned chunkset is absent
    (the passage bodies + week signal ground the regrade)."""
    course_dir = Path(course_dir)
    gold, _ = load_gold_set(course_dir, verify=False)
    if not gold:
        raise FileNotFoundError(f"no gold set found under {course_dir}.")
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path")
    if not chunks_rel:
        raise FileNotFoundError(
            f"no chunkset pin in gold set at {course_dir}; cannot ground regrade."
        )
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise FileNotFoundError(f"pinned chunkset not found at {chunks_path}.")
    chunks_by_id = _load_chunks_by_id(chunks_path)

    rows = build_regrade_rows(
        gold, chunks_by_id, client=client, capture=capture,
        model_id=model_id, regrade_from=regrade_from,
    )
    resolved_model = model_id or getattr(client, "model", "local")
    doc = build_proposal_doc(
        gold.get("course_slug", course_dir.name), gold, rows,
        model_id=resolved_model, regrade_from=regrade_from,
    )

    out_path: Optional[Path] = None
    if write:
        out_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / GOLD_DIFFICULTY_REGRADE_PROPOSAL_FILENAME
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc, out_path


__all__ = [
    "GOLD_DIFFICULTY_REGRADE_PROPOSAL_FILENAME",
    "DIFFICULTY_REGRADE_PROMPT_VERSION",
    "RegradeRow",
    "regrade_question",
    "build_regrade_rows",
    "projected_distribution",
    "build_proposal_doc",
    "generate_difficulty_regrade_proposal",
]
