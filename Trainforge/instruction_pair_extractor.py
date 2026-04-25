#!/usr/bin/env python3
"""Trainforge — Real Q-A / task-solution / reasoning-chain extractor.

Where Wave 77 epsilon's :mod:`Trainforge.synthesize_training` wraps every
chunk in an "Explain X" template, this stage walks the *structure already
present* in chunks and lifts:

* genuine Q -> A pairs from ``assessment_item`` chunks (stem, correct
  answer marker, reasoning),
* task -> solution pairs from ``exercise`` chunks,
* step-by-step reasoning chains from ``example`` chunks that carry
  worked-example structure ("First, ...", "Step 1: ...", "Then, ...",
  "Finally, ..."),
* fallback concept-question pairs from ``explanation`` chunks (lower
  quality bucket, tagged as ``derived_from_explanation_template``),
* contrastive pairs from each chunk's misconceptions (distinguish-from
  + compare-and-contrast).

Every emitted pair carries the metadata downstream stratified
SFT/DPO training depends on (``objective_ids``, ``bloom_level``,
``difficulty``, ``chunk_type``, ``extraction_method``,
``source_chunk_id``, ``quality_score``).

This module is a *new*, additive extractor. It does not modify the
Wave 77 epsilon API. The two stages can run side by side; their pair
JSONLs are union-able by ``source_chunk_id`` + ``extraction_method``.

CLI:

.. code-block:: bash

    python -m Trainforge.instruction_pair_extractor \\
        --slug rdf-shacl-550 \\
        --methods assessment_item,exercise,example_reasoning,\\
explanation_template,misconception_distinguish,misconception_contrast \\
        --min-quality 0.7 \\
        --output ./out/

Outputs (under ``--output``):

* ``instruction_pairs.jsonl`` — one row per emitted pair (full metadata)
* ``reasoning_chains.jsonl`` — subset where ``reasoning_chain`` is set
  (chain-of-thought training)
* ``extraction_report.json`` — counts by method + tag distributions
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Make project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public extraction-method names (also accepted as CLI ``--methods`` tokens).
# ---------------------------------------------------------------------------

METHOD_ASSESSMENT_ITEM = "assessment_item"
METHOD_EXERCISE = "exercise"
METHOD_EXAMPLE_REASONING = "example_reasoning"
METHOD_EXPLANATION_TEMPLATE = "explanation_template"
METHOD_MISCONCEPTION_DISTINGUISH = "misconception_distinguish"
METHOD_MISCONCEPTION_CONTRAST = "misconception_contrast"

ALL_METHODS: Tuple[str, ...] = (
    METHOD_ASSESSMENT_ITEM,
    METHOD_EXERCISE,
    METHOD_EXAMPLE_REASONING,
    METHOD_EXPLANATION_TEMPLATE,
    METHOD_MISCONCEPTION_DISTINGUISH,
    METHOD_MISCONCEPTION_CONTRAST,
)


# ---------------------------------------------------------------------------
# Quality-score buckets per the spec.
# ---------------------------------------------------------------------------

_QUALITY_BY_METHOD: Dict[str, float] = {
    METHOD_ASSESSMENT_ITEM: 1.0,
    METHOD_EXERCISE: 1.0,
    METHOD_EXAMPLE_REASONING: 1.0,
    METHOD_EXPLANATION_TEMPLATE: 0.6,
    METHOD_MISCONCEPTION_DISTINGUISH: 0.9,
    METHOD_MISCONCEPTION_CONTRAST: 0.9,
}


# ---------------------------------------------------------------------------
# Regex toolkit
# ---------------------------------------------------------------------------

# Inline ``Show answer`` is the canonical Courseforge marker dropped between
# stems and explanations in formative quizzes. Wave 77 gamma observed it
# verbatim in the rdf-shacl-550 archive.
_SHOW_ANSWER_RE = re.compile(r"\bShow\s+answer\b", re.IGNORECASE)
# Generic answer markers used across exercise / assessment chunks.
_ANSWER_MARKERS = (
    "Show answer",
    "Answer:",
    "Correct:",
    "Correct answer:",
    "Solution:",
    "Sample solution:",
    "Expected output:",
    "Output:",
    "Sample output:",
)
_ANY_ANSWER_MARKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m).rstrip(":\\\\") for m in _ANSWER_MARKERS) + r")\s*:?",
    re.IGNORECASE,
)

# Numbered / ordinal step cues for reasoning-chain extraction.
_STEP_CUES = (
    r"^\s*Step\s+\d+\s*[:.\)]",
    r"^\s*Step\s+\d+\b",
    r"^\s*\d+\.\s+",
    r"^\s*\d+\)\s+",
    r"\bFirst,",
    r"\bSecond,",
    r"\bThird,",
    r"\bFourth,",
    r"\bThen,",
    r"\bNext,",
    r"\bAfter\s+that,",
    r"\bFinally,",
    r"\bIn\s+conclusion,",
    r"\bLastly,",
)
_STEP_SPLIT_RE = re.compile("|".join(_STEP_CUES), re.MULTILINE)

# Sentence boundary used to find leading "problem statement" of an example.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExtractionStats:
    """Counts returned from :func:`run_extraction`."""

    chunks_total: int = 0
    pairs_by_method: Dict[str, int] = field(default_factory=dict)
    pairs_emitted: int = 0
    pairs_filtered_quality: int = 0
    reasoning_chains_emitted: int = 0
    misconceptions_seen: int = 0

    def bump(self, method: str, n: int = 1) -> None:
        self.pairs_by_method[method] = self.pairs_by_method.get(method, 0) + n
        self.pairs_emitted += n

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "pairs_emitted": self.pairs_emitted,
            "pairs_filtered_quality": self.pairs_filtered_quality,
            "pairs_by_method": dict(self.pairs_by_method),
            "reasoning_chains_emitted": self.reasoning_chains_emitted,
            "misconceptions_seen": self.misconceptions_seen,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _normalize(text: Any) -> str:
    """Collapse internal whitespace; keep newlines if relevant for steps."""
    if not text:
        return ""
    s = str(text)
    # Collapse runs of spaces / tabs but keep paragraph breaks.
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def _flatten(text: Any) -> str:
    """Aggressive flatten — collapse newlines into spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _build_metadata(
    *,
    chunk: Dict[str, Any],
    extraction_method: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    md: Dict[str, Any] = {
        "objective_ids": list(chunk.get("learning_outcome_refs") or []),
        "bloom_level": chunk.get("bloom_level"),
        "bloom_level_secondary": chunk.get("bloom_level_secondary"),
        "difficulty": chunk.get("difficulty"),
        "chunk_type": chunk.get("chunk_type"),
        "extraction_method": extraction_method,
        "source_chunk_id": chunk.get("id") or chunk.get("chunk_id"),
        "quality_score": _QUALITY_BY_METHOD.get(extraction_method, 0.5),
    }
    if extra:
        md.update(extra)
    return md


def _shape_pair(
    *,
    instruction: str,
    output: str,
    chunk: Dict[str, Any],
    extraction_method: str,
    reasoning_chain: Optional[List[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble the canonical pair shape, or None on validation failure."""
    instruction = _flatten(instruction)
    output = _flatten(output)
    if not instruction or not output:
        return None
    pair: Dict[str, Any] = {
        "instruction": instruction,
        "output": output,
        "metadata": _build_metadata(
            chunk=chunk, extraction_method=extraction_method, extra=extra_meta
        ),
        "schema_version": "v1",
    }
    if reasoning_chain:
        pair["reasoning_chain"] = list(reasoning_chain)
    return pair


# ---------------------------------------------------------------------------
# Extractor 1: assessment_item -> Q -> A pairs
# ---------------------------------------------------------------------------


_LETTER_ANSWER_RE = re.compile(
    r"^\s*(?:\(?([A-Da-d])\)?|([A-Da-d])\.)\s+",
)
_TF_ANSWER_RE = re.compile(r"^\s*(True|False)\b", re.IGNORECASE)


def _split_questions_show_answer(text: str) -> List[Tuple[str, str]]:
    """Split a ``Show answer``-marked formative quiz into [(stem, answer)].

    The Courseforge convention is::

        intro... <stem1> Show answer <answer1+reasoning1> <stem2> Show answer ...

    Some chunks open with instructional copy that itself mentions the
    phrase "click Show answer to reveal" — we drop those non-marker
    occurrences before splitting.
    """
    # Remove non-marker occurrences ("then click Show answer to reveal").
    # Keep only "Show answer" tokens that look like delimiters between a
    # stem and an explanation: the next chars after the marker should NOT
    # be one of ("to reveal", "to confirm", "to check", "to see").
    cleaned = re.sub(
        r"\bclick\s+Show\s+answer\s+(?:to\s+(?:reveal|confirm|check|see)\b[^.]*\.?)",
        "click [reveal] ",
        text,
        flags=re.IGNORECASE,
    )
    parts = _SHOW_ANSWER_RE.split(cleaned)
    if len(parts) < 3:
        return []
    pairs: List[Tuple[str, str]] = []
    intro = parts[0]
    # Locate the first stem inside the intro: take the last paragraph that
    # plausibly ends with question content (contains '?' or list of
    # choices) — fall back to the last paragraph.
    intro_paragraphs = [p.strip() for p in intro.split("\n\n") if p.strip()]

    def _looks_like_stem(p: str) -> bool:
        if "?" in p:
            return True
        # Multi-choice list: ends with " D. ..." letter choice.
        if re.search(r"\bD\.\s+\S", p):
            return True
        return False

    stem_n = ""
    for p in reversed(intro_paragraphs):
        if _looks_like_stem(p):
            stem_n = p
            break
    if not stem_n and intro_paragraphs:
        stem_n = intro_paragraphs[-1]

    # Walk in 2-step strides.
    for idx in range(1, len(parts)):
        block = parts[idx]
        if idx == len(parts) - 1:
            answer_n = block.strip()
            next_stem = ""
        else:
            paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
            if len(paragraphs) >= 2:
                answer_n = "\n\n".join(paragraphs[:-1])
                next_stem = paragraphs[-1]
            else:
                m = _SENTENCE_END_RE.split(block, maxsplit=1)
                if len(m) == 2:
                    answer_n, next_stem = m[0].strip(), m[1].strip()
                else:
                    answer_n = block.strip()
                    next_stem = ""
        if stem_n and answer_n and _looks_like_stem(stem_n):
            pairs.append((stem_n, answer_n))
        stem_n = next_stem
    return pairs


def _split_questions_qa_marker(text: str) -> List[Tuple[str, str]]:
    """Split ``Q: ... A: ...`` style stubs (used by tests / minimal inputs)."""
    pattern = re.compile(
        r"Q\s*:\s*(?P<q>.+?)\s*\n\s*\n?\s*A\s*:\s*(?P<a>.+?)(?=(?:\n\s*Q\s*:)|$)",
        re.DOTALL | re.IGNORECASE,
    )
    out: List[Tuple[str, str]] = []
    for m in pattern.finditer(text):
        q = _flatten(m.group("q"))
        a = _flatten(m.group("a"))
        if q and a:
            out.append((q, a))
    return out


def extract_from_assessment_item(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = chunk.get("text") or ""
    if not text.strip():
        return []
    raw_pairs: List[Tuple[str, str]] = []
    if _SHOW_ANSWER_RE.search(text):
        raw_pairs = _split_questions_show_answer(text)
    if not raw_pairs:
        raw_pairs = _split_questions_qa_marker(text)
    out: List[Dict[str, Any]] = []
    for stem, answer in raw_pairs:
        # Best-effort: split answer into [letter/TF] + reasoning.
        answer_clean = answer.strip()
        reasoning = ""
        answer_token = ""
        m = _LETTER_ANSWER_RE.match(answer_clean)
        tf = _TF_ANSWER_RE.match(answer_clean)
        if m:
            answer_token = (m.group(1) or m.group(2) or "").upper()
            reasoning = answer_clean[m.end():].strip()
        elif tf:
            answer_token = tf.group(1).capitalize()
            reasoning = answer_clean[tf.end():].strip()
        else:
            # No leading letter / T/F — treat entire answer as the
            # answer; reasoning empty.
            answer_token = answer_clean
        # Compose per-spec: ``<answer_text> — Reasoning: <reasoning_text>``.
        # When there is no separable reasoning we omit the suffix to avoid
        # an empty "— Reasoning:" tail.
        if answer_token and reasoning:
            output_text = f"{answer_token}. — Reasoning: {reasoning}"
        elif answer_token:
            output_text = answer_token
        else:
            output_text = reasoning
        pair = _shape_pair(
            instruction=stem,
            output=output_text,
            chunk=chunk,
            extraction_method=METHOD_ASSESSMENT_ITEM,
            extra_meta={
                "has_reasoning": bool(reasoning),
                "answer_token": answer_token or None,
            },
        )
        if pair:
            out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Extractor 2: exercise -> task -> solution pairs
# ---------------------------------------------------------------------------


_TASK_HEADERS = (
    "Submission format:",
    "Assessment criteria:",
    "Common pitfalls",
)


def _split_task_solution(text: str) -> Optional[Tuple[str, str, str]]:
    """Find the first task-vs-solution boundary in an exercise chunk.

    Returns ``(task, solution, marker_label)`` so callers can record which
    marker fired (useful for stratified downstream analysis).
    """
    # Prefer explicit "Solution:" / "Output:" / "Sample output:" / "Show
    # answer". Order matters for overlapping prefixes (Sample solution
    # before Solution; Sample output before Output).
    for marker_re, marker in (
        (re.compile(r"\bSample\s+solution\s*:", re.IGNORECASE), "sample_solution"),
        (re.compile(r"\bSolution\s*:", re.IGNORECASE), "solution"),
        (re.compile(r"\bExpected\s+output\s*:", re.IGNORECASE), "expected_output"),
        (re.compile(r"\bSample\s+output\s*:", re.IGNORECASE), "sample_output"),
        (re.compile(r"\bOutput\s*:", re.IGNORECASE), "output"),
        (re.compile(r"\bShow\s+answer\b", re.IGNORECASE), "show_answer"),
    ):
        m = marker_re.search(text)
        if m:
            task = text[: m.start()].strip()
            solution = text[m.end():].strip()
            if task and solution:
                return task, solution, marker
    return None


def extract_from_exercise(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = chunk.get("text") or ""
    if not text.strip():
        return []
    out: List[Dict[str, Any]] = []
    explicit = _split_task_solution(text)
    if explicit:
        task, solution, marker = explicit
        pair = _shape_pair(
            instruction=task,
            output=solution,
            chunk=chunk,
            extraction_method=METHOD_EXERCISE,
            extra_meta={"split_marker": marker},
        )
        if pair:
            out.append(pair)
        return out
    # Fallback: split on "Submission format:" / "Assessment criteria:".
    # The assessment criteria section is *expected output*; the rest above
    # it is the task.
    for header in _TASK_HEADERS:
        idx = text.find(header)
        if idx > 100:  # require a non-trivial preamble
            task = text[:idx].strip()
            solution = text[idx:].strip()
            if task and solution:
                pair = _shape_pair(
                    instruction=task,
                    output=solution,
                    chunk=chunk,
                    extraction_method=METHOD_EXERCISE,
                    extra_meta={"split_marker": header.rstrip(":").lower().replace(" ", "_")},
                )
                if pair:
                    out.append(pair)
                return out
    # Last-resort fallback: treat the leading objective + first paragraph
    # as the task, the remainder as the worked walkthrough. We require the
    # text to be long enough that the split is meaningful.
    if len(text) >= 600:
        # Prefer paragraph boundary; otherwise sentence boundary.
        idx = text.find("\n\n")
        if idx == -1 or idx < 80:
            sentences = _SENTENCE_END_RE.split(text, maxsplit=2)
            if len(sentences) >= 2:
                task = " ".join(sentences[:1]).strip()
                solution = " ".join(sentences[1:]).strip()
            else:
                return out
        else:
            task = text[:idx].strip()
            solution = text[idx:].strip()
        if task and solution and len(solution) >= 200:
            pair = _shape_pair(
                instruction=(
                    f"Complete the following exercise: {task}"
                ),
                output=solution,
                chunk=chunk,
                extraction_method=METHOD_EXERCISE,
                extra_meta={"split_marker": "paragraph_fallback"},
            )
            if pair:
                out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Extractor 3: example -> reasoning chains
# ---------------------------------------------------------------------------


def _extract_steps(text: str) -> List[str]:
    """Pull ordered reasoning steps out of a worked example."""
    if not text:
        return []
    # Find each step marker position and take the slice from there to the
    # next marker.
    cues = list(_STEP_SPLIT_RE.finditer(text))
    if len(cues) < 2:
        return []
    steps: List[str] = []
    for i, m in enumerate(cues):
        start = m.start()
        end = cues[i + 1].start() if i + 1 < len(cues) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        # Strip leading bullet/number to keep just the step body.
        chunk = re.sub(r"^\s*(?:Step\s+\d+\s*[:.\)]?|\d+[.)]\s*|First,|Second,|Third,|Fourth,|Then,|Next,|After\s+that,|Finally,|In\s+conclusion,|Lastly,)\s*",
                       "", chunk, flags=re.IGNORECASE)
        chunk = _flatten(chunk)
        if len(chunk) >= 12:
            steps.append(chunk)
    # Require at least 2 distinct steps to call it a "chain".
    if len(steps) < 2:
        return []
    return steps


def _derive_problem_statement(chunk: Dict[str, Any], steps_start_in_text: str) -> str:
    """Compose a concise problem statement for an example chunk.

    Priority: chunk['summary'] -> first sentence of text -> first concept
    tag -> "Walk through the following worked example."
    """
    summary = _flatten(chunk.get("summary"))
    if summary:
        return summary
    # First couple of sentences from the chunk text (preceding the first
    # step cue).
    text = chunk.get("text") or ""
    cues = list(_STEP_SPLIT_RE.finditer(text))
    if cues:
        prelude = text[: cues[0].start()].strip()
    else:
        prelude = text
    if prelude:
        sentences = _SENTENCE_END_RE.split(prelude)
        if sentences:
            head = " ".join(sentences[:2]).strip()
            head = _flatten(head)
            if head:
                return head
    tags = chunk.get("concept_tags") or []
    if tags:
        return f"Walk through the worked example for {tags[0]}."
    return "Walk through the following worked example step by step."


def extract_from_example(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = chunk.get("text") or ""
    if not text.strip():
        return []
    steps = _extract_steps(text)
    if not steps:
        return []
    problem = _derive_problem_statement(chunk, text)
    solution = " ".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
    pair = _shape_pair(
        instruction=problem,
        output=solution,
        chunk=chunk,
        extraction_method=METHOD_EXAMPLE_REASONING,
        reasoning_chain=steps,
        extra_meta={"step_count": len(steps)},
    )
    return [pair] if pair else []


# ---------------------------------------------------------------------------
# Extractor 4: explanation -> concept-question template (low quality)
# ---------------------------------------------------------------------------


def _question_from_concept(chunk: Dict[str, Any]) -> Tuple[str, str]:
    """Build a (question, derivation_hint) for a generic concept-question pair."""
    tags = chunk.get("concept_tags") or []
    if tags:
        topic = str(tags[0]).replace("_", " ").replace("-", " ")
        return f"What is {topic}, and how does it work?", "concept_tags[0]"
    key_terms = chunk.get("key_terms") or []
    if key_terms and isinstance(key_terms[0], dict):
        term = key_terms[0].get("term")
        if term:
            return f"What is {term}, and how does it work?", "key_terms[0]"
    summary = _flatten(chunk.get("summary"))
    if summary:
        # Strip trailing period from the summary if there is one before
        # turning it into a question.
        head = summary.rstrip(".!?")
        return f"Explain the following idea: {head}.", "summary"
    # Last resort — use the first sentence.
    text = chunk.get("text") or ""
    sentences = _SENTENCE_END_RE.split(text, maxsplit=2)
    if sentences:
        first = _flatten(sentences[0])
        if first:
            return f"Explain the following: {first.rstrip('.!?')}.", "first_sentence"
    return ("Explain the concept in this section.", "fallback")


def _question_variants(chunk: Dict[str, Any], max_variants: int = 3) -> List[Tuple[str, str]]:
    """Yield up to ``max_variants`` distinct (question, derivation) tuples
    for an explanation chunk so we lift more than one training pair from
    chunks with rich metadata.
    """
    seen: List[Tuple[str, str]] = []
    seen_keys = set()

    def push(q: str, derivation: str) -> None:
        if not q:
            return
        key = q.strip().lower()
        if key in seen_keys:
            return
        seen_keys.add(key)
        seen.append((q, derivation))

    # Variant 1: first concept_tag.
    tags = [str(t).strip() for t in (chunk.get("concept_tags") or []) if t]
    if tags:
        topic = tags[0].replace("_", " ").replace("-", " ")
        push(f"What is {topic}, and how does it work?", "concept_tags[0]")
    # Variant 2: first key_term.
    key_terms = chunk.get("key_terms") or []
    if key_terms and isinstance(key_terms[0], dict):
        term = key_terms[0].get("term")
        if term:
            push(
                f"Define '{term}' as it is used in this material.",
                "key_terms[0]",
            )
    # Variant 3: summary as imperative.
    summary = _flatten(chunk.get("summary"))
    if summary:
        head = summary.rstrip(".!?")
        push(f"Explain the following idea: {head}.", "summary")
    # Variant 4: second concept_tag (if available).
    if len(tags) >= 2:
        topic2 = tags[1].replace("_", " ").replace("-", " ")
        push(
            f"Why does {topic2} matter in the context of this section?",
            "concept_tags[1]",
        )
    # Variant 5: first sentence of text.
    text = chunk.get("text") or ""
    sentences = _SENTENCE_END_RE.split(text, maxsplit=2)
    if sentences:
        first = _flatten(sentences[0]).rstrip(".!?")
        if first:
            push(f"Explain the following: {first}.", "first_sentence")
    return seen[:max_variants]


def extract_from_explanation(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = chunk.get("text") or ""
    if not text.strip():
        return []
    body = _flatten(text)
    out: List[Dict[str, Any]] = []
    variants = _question_variants(chunk, max_variants=3)
    if not variants:
        question, derivation = _question_from_concept(chunk)
        variants = [(question, derivation)]
    for question, derivation in variants:
        pair = _shape_pair(
            instruction=question,
            output=body,
            chunk=chunk,
            extraction_method=METHOD_EXPLANATION_TEMPLATE,
            extra_meta={
                "derived_from": "explanation_template",
                "question_derivation": derivation,
            },
        )
        if pair:
            out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Extractor 5: contrastive misconception pairs
# ---------------------------------------------------------------------------


def extract_from_misconceptions(
    chunk: Dict[str, Any],
    *,
    methods: Sequence[str],
) -> List[Dict[str, Any]]:
    misconceptions = chunk.get("misconceptions") or []
    if not misconceptions:
        return []
    out: List[Dict[str, Any]] = []
    concept_topic = ""
    tags = chunk.get("concept_tags") or []
    if tags:
        concept_topic = str(tags[0]).replace("_", " ").replace("-", " ")
    for misc in misconceptions:
        if not isinstance(misc, dict):
            continue
        statement = _flatten(misc.get("misconception"))
        correction = _flatten(misc.get("correction"))
        if not statement or not correction:
            continue
        topic_for_compare = concept_topic or "this concept"
        if METHOD_MISCONCEPTION_DISTINGUISH in methods:
            instruction = (
                f"Is the following claim correct? '{statement}'"
            )
            output = f"No. {correction}"
            pair = _shape_pair(
                instruction=instruction,
                output=output,
                chunk=chunk,
                extraction_method=METHOD_MISCONCEPTION_DISTINGUISH,
                extra_meta={
                    "misconception_statement": statement,
                    "correction_statement": correction,
                },
            )
            if pair:
                out.append(pair)
        if METHOD_MISCONCEPTION_CONTRAST in methods:
            instruction = (
                f"Compare {topic_for_compare} as commonly misunderstood "
                f"vs. correctly."
            )
            output = (
                f"Common misunderstanding: {statement} "
                f"Correct understanding: {correction}"
            )
            pair = _shape_pair(
                instruction=instruction,
                output=output,
                chunk=chunk,
                extraction_method=METHOD_MISCONCEPTION_CONTRAST,
                extra_meta={
                    "misconception_statement": statement,
                    "correction_statement": correction,
                    "topic": topic_for_compare,
                },
            )
            if pair:
                out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _read_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _resolve_corpus_path(slug: str, libv2_root: Optional[Path] = None) -> Path:
    """Locate ``corpus/chunks.jsonl`` for ``slug`` under LibV2."""
    root = libv2_root or (PROJECT_ROOT / "LibV2" / "courses")
    for candidate in (root / slug, root / f"{slug}-{slug}"):
        p = candidate / "corpus" / "chunks.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find chunks.jsonl for slug {slug!r} under {root}"
    )


def _slug_to_course_code(slug: str) -> str:
    """Best-effort coerce a course slug to the ``[A-Z]{2,8}_[0-9]{3}`` shape
    that ``DecisionCapture`` validates.

    ``rdf-shacl-550`` -> ``RDFSHACL_550``. Falls back to ``COURSE_000`` if
    the slug has no trailing ``-NNN`` segment.
    """
    parts = slug.split("-")
    if not parts:
        return "COURSE_000"
    tail = parts[-1]
    if tail.isdigit():
        head = "".join(p for p in parts[:-1] if p.isalpha()).upper()
        if not head:
            head = "COURSE"
        head = head[:8] if len(head) > 8 else head
        if len(head) < 2:
            head = (head + "XX")[:2]
        # Course code regex requires exactly NNN, so left-pad / truncate.
        digits = tail.zfill(3)[-3:]
        return f"{head}_{digits}"
    cleaned = re.sub(r"[^A-Za-z]", "", slug).upper()[:8] or "COURSE"
    if len(cleaned) < 2:
        cleaned = (cleaned + "XX")[:2]
    return f"{cleaned}_000"


def run_extraction(
    *,
    chunks_path: Path,
    output_dir: Path,
    methods: Sequence[str] = ALL_METHODS,
    min_quality: float = 0.0,
    course_code: str = "COURSE_000",
    capture: Optional[DecisionCapture] = None,
) -> Tuple[ExtractionStats, List[Dict[str, Any]]]:
    """Run the extractor over ``chunks_path`` and write outputs to ``output_dir``.

    Returns the ``ExtractionStats`` and the list of emitted pairs (post
    quality filter).
    """
    method_set = set(methods)
    unknown = method_set - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"Unknown extraction methods: {sorted(unknown)}")

    own_capture = False
    if capture is None:
        try:
            capture = DecisionCapture(
                course_code=course_code,
                phase="trainforge-question-generation",
                tool="trainforge",
                streaming=False,
            )
            own_capture = True
        except Exception as exc:  # pragma: no cover - non-conformant slug
            logger.warning(
                "DecisionCapture init failed for course_code=%r: %s; "
                "continuing without capture", course_code, exc,
            )
            capture = None

    chunks = _read_chunks(chunks_path)
    stats = ExtractionStats(chunks_total=len(chunks))
    pairs: List[Dict[str, Any]] = []

    for chunk in chunks:
        chunk_type = (chunk.get("chunk_type") or "").strip().lower()
        emitted: List[Dict[str, Any]] = []

        if chunk_type == "assessment_item" and METHOD_ASSESSMENT_ITEM in method_set:
            emitted.extend(extract_from_assessment_item(chunk))
        if chunk_type == "exercise" and METHOD_EXERCISE in method_set:
            emitted.extend(extract_from_exercise(chunk))
        if chunk_type == "example" and METHOD_EXAMPLE_REASONING in method_set:
            emitted.extend(extract_from_example(chunk))
        if chunk_type in {"explanation", "overview", "summary"} and METHOD_EXPLANATION_TEMPLATE in method_set:
            # All three types share the same expository structure
            # (LO + body + closing). Lifting them through the same
            # template keeps the ``derived_from=explanation_template``
            # quality bucket honest.
            emitted.extend(extract_from_explanation(chunk))
        if (
            METHOD_MISCONCEPTION_DISTINGUISH in method_set
            or METHOD_MISCONCEPTION_CONTRAST in method_set
        ):
            misc_pairs = extract_from_misconceptions(chunk, methods=methods)
            if misc_pairs:
                stats.misconceptions_seen += sum(
                    1 for p in misc_pairs
                    if p["metadata"]["extraction_method"] == METHOD_MISCONCEPTION_DISTINGUISH
                )
                emitted.extend(misc_pairs)

        for pair in emitted:
            qs = float(pair["metadata"].get("quality_score", 0.0))
            if qs < min_quality:
                stats.pairs_filtered_quality += 1
                continue
            method = pair["metadata"]["extraction_method"]
            if capture is not None:
                _capture_emit(capture, pair, chunk)
            stats.bump(method)
            if pair.get("reasoning_chain"):
                stats.reasoning_chains_emitted += 1
            pairs.append(pair)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "instruction_pairs.jsonl", pairs)
    chains = [p for p in pairs if p.get("reasoning_chain")]
    _write_jsonl(output_dir / "reasoning_chains.jsonl", chains)

    report = {
        "chunks_total": stats.chunks_total,
        "methods_requested": sorted(method_set),
        "min_quality": min_quality,
        "pairs_emitted": stats.pairs_emitted,
        "pairs_filtered_quality": stats.pairs_filtered_quality,
        "pairs_by_method": stats.pairs_by_method,
        "reasoning_chains_emitted": stats.reasoning_chains_emitted,
        "misconceptions_seen": stats.misconceptions_seen,
        "outputs": {
            "instruction_pairs": str(output_dir / "instruction_pairs.jsonl"),
            "reasoning_chains": str(output_dir / "reasoning_chains.jsonl"),
        },
    }
    (output_dir / "extraction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    if own_capture and capture is not None:
        try:
            capture.close()
        except Exception:  # pragma: no cover - close is best-effort
            pass

    return stats, pairs


def _capture_emit(
    capture: DecisionCapture,
    pair: Dict[str, Any],
    chunk: Dict[str, Any],
) -> None:
    """Log one ``instruction_pair_synthesis`` decision per emitted pair.

    Per root CLAUDE.md "LLM call-site instrumentation" rule, we capture a
    rationale that interpolates dynamic signals (chunk id, method, quality
    score, lo refs, lengths) so the decision is replayable post-hoc.
    """
    md = pair["metadata"]
    method = md["extraction_method"]
    chunk_id = md["source_chunk_id"]
    instr_len = len(pair.get("instruction") or "")
    out_len = len(pair.get("output") or "")
    chain = pair.get("reasoning_chain") or []
    rationale = (
        f"Wave 79 extractor lifted a {method} pair from chunk {chunk_id} "
        f"(chunk_type={md.get('chunk_type')}, lo_refs={md.get('objective_ids')}, "
        f"bloom={md.get('bloom_level')}, difficulty={md.get('difficulty')}). "
        f"Instruction length={instr_len} chars, output length={out_len} chars, "
        f"reasoning_chain_steps={len(chain)}, quality_score={md.get('quality_score')}. "
        f"Pair extracted from existing chunk structure rather than via "
        f"template wrap (Wave 77 epsilon stays for that)."
    )
    try:
        capture.log_decision(
            decision_type="instruction_pair_synthesis",
            decision=f"emit_{method}",
            rationale=rationale,
            operation="extract_instruction_pair",
            confidence=md.get("quality_score"),
            context=f"chunk_id={chunk_id} method={method}",
        )
    except Exception as exc:  # pragma: no cover - capture is best-effort
        logger.debug("capture emit failed for chunk %s: %s", chunk_id, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_methods(raw: Optional[str]) -> Tuple[str, ...]:
    if not raw:
        return ALL_METHODS
    methods = tuple(m.strip() for m in raw.split(",") if m.strip())
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown methods {unknown}; valid: {ALL_METHODS}"
        )
    return methods


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m Trainforge.instruction_pair_extractor",
        description=(
            "Extract genuine instruction-response and reasoning-chain pairs "
            "from a Trainforge-aligned chunks.jsonl. Beyond Wave 77 epsilon's "
            "template wrap."
        ),
    )
    parser.add_argument(
        "--slug",
        help="LibV2 course slug (looked up under LibV2/courses/<slug>/corpus).",
    )
    parser.add_argument(
        "--chunks",
        help="Direct path to a chunks.jsonl. Overrides --slug.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help=(
            "Comma-separated list of extraction methods to run. Defaults "
            "to all six."
        ),
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Drop pairs whose quality_score < this threshold (default 0.0).",
    )
    parser.add_argument(
        "--output",
        default="./out/",
        help="Directory to write extractor outputs into.",
    )
    parser.add_argument(
        "--course-code",
        help=(
            "Override the inferred course_code used for DecisionCapture. "
            "Must match ^[A-Z]{2,8}_[0-9]{3}$."
        ),
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Disable DecisionCapture emit (CI / sandbox mode).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.slug and not args.chunks:
        parser.error("one of --slug or --chunks is required")
    if args.chunks:
        chunks_path = Path(args.chunks)
    else:
        chunks_path = _resolve_corpus_path(args.slug)
    methods = _parse_methods(args.methods)
    course_code = args.course_code or (
        _slug_to_course_code(args.slug) if args.slug else "COURSE_000"
    )
    output_dir = Path(args.output)
    capture: Optional[DecisionCapture] = None
    if not args.no_capture:
        try:
            capture = DecisionCapture(
                course_code=course_code,
                phase="trainforge-question-generation",
                tool="trainforge",
                streaming=False,
            )
        except Exception as exc:
            logger.warning(
                "DecisionCapture init failed (course_code=%r): %s; "
                "continuing without capture", course_code, exc,
            )
            capture = None
    stats, pairs = run_extraction(
        chunks_path=chunks_path,
        output_dir=output_dir,
        methods=methods,
        min_quality=args.min_quality,
        course_code=course_code,
        capture=capture,
    )
    if capture is not None:
        try:
            capture.close()
        except Exception:  # pragma: no cover
            pass
    print(json.dumps(stats.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
