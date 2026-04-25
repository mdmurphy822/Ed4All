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
# Wave 81: template-aware extractors keyed on chunk_type values that are now
# propagated from Courseforge ``data-cf-template-type`` (Wave 79 C). Each
# extractor produces a richer pair shape than the legacy
# ``explanation_template`` fallback because the source HTML carries
# template-specific subsections (Steps / Inputs / Output for procedure,
# Your Task / Approach / Success Criteria for real_world_scenario, etc.).
METHOD_PROCEDURE = "procedure"
METHOD_REAL_WORLD_SCENARIO = "real_world_scenario"
METHOD_COMMON_PITFALL = "common_pitfall"
METHOD_COMMON_PITFALL_MULTI_ARM = "common_pitfall_multi_arm"
METHOD_PROBLEM_SOLUTION = "problem_solution"
METHOD_PROBLEM_SOLUTION_DPO = "problem_solution_dpo"

ALL_METHODS: Tuple[str, ...] = (
    METHOD_ASSESSMENT_ITEM,
    METHOD_EXERCISE,
    METHOD_EXAMPLE_REASONING,
    METHOD_EXPLANATION_TEMPLATE,
    METHOD_MISCONCEPTION_DISTINGUISH,
    METHOD_MISCONCEPTION_CONTRAST,
    METHOD_PROCEDURE,
    METHOD_REAL_WORLD_SCENARIO,
    METHOD_COMMON_PITFALL,
    METHOD_COMMON_PITFALL_MULTI_ARM,
    METHOD_PROBLEM_SOLUTION,
    METHOD_PROBLEM_SOLUTION_DPO,
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
    # Wave 81 template-aware buckets — generally higher than
    # explanation_template because the chunks carry strict subsection
    # structure (Inputs/Steps/Output etc.) that produces well-shaped pairs.
    METHOD_PROCEDURE: 1.0,
    METHOD_REAL_WORLD_SCENARIO: 0.95,
    METHOD_COMMON_PITFALL: 0.95,
    METHOD_COMMON_PITFALL_MULTI_ARM: 0.95,
    METHOD_PROBLEM_SOLUTION: 1.0,
    METHOD_PROBLEM_SOLUTION_DPO: 1.0,
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
# Wave 81 template-aware extractors
# ---------------------------------------------------------------------------
#
# These extractors run when chunk.chunk_type matches a Wave 79 C template
# label (procedure, real_world_scenario, common_pitfall, problem_solution).
# Each template emits a stable subsection grammar that we exploit to lift
# pairs richer than what the legacy explanation_template fallback produces.

# Each section header maps to the prefix string we expect (case-insensitive).
# Anchors are reused across templates because Courseforge content-generator
# templates emit the same section labels in upper- or title-case across
# pages.

_SECTION_HEADER_RES = {
    # procedure
    "inputs": re.compile(r"\bInputs?\b\s*:?", re.IGNORECASE),
    "steps": re.compile(r"\bSteps?\b\s*:?", re.IGNORECASE),
    "output": re.compile(r"\bOutput\b\s*:?", re.IGNORECASE),
    "worked_example": re.compile(r"\bWorked\s+Example\b", re.IGNORECASE),
    "when_to_use": re.compile(r"\bWhen\s+to\s+use\b", re.IGNORECASE),
    # real_world_scenario
    "scenario": re.compile(r"\bScenario\b\s*:?", re.IGNORECASE),
    "your_task": re.compile(r"\bYour\s+Task\b", re.IGNORECASE),
    "approach": re.compile(r"\bApproach\b", re.IGNORECASE),
    "success_criteria": re.compile(
        r"\bSuccess\s+Criteria\b", re.IGNORECASE
    ),
    # common_pitfall
    "what_looks_right": re.compile(
        r"\bWhat\s+looks\s+like\s+the\s+right\s+answer\b",
        re.IGNORECASE,
    ),
    "why_wrong": re.compile(r"\bWhy\s+it'?s\s+wrong\b", re.IGNORECASE),
    "right_approach": re.compile(
        r"\bThe\s+right\s+approach\b", re.IGNORECASE
    ),
    "quick_test": re.compile(r"\bQuick\s+test\b", re.IGNORECASE),
    # problem_solution
    "problem": re.compile(r"\bProblem\b", re.IGNORECASE),
    "walkthrough": re.compile(r"\bWalkthrough\b", re.IGNORECASE),
    "common_incorrect": re.compile(
        r"\bCommon\s+Incorrect\s+Approach\b", re.IGNORECASE
    ),
    "verification": re.compile(
        r"\bVerification\s+discipline\b", re.IGNORECASE
    ),
}


def _split_by_headers(text: str, header_keys: Sequence[str]) -> Dict[str, str]:
    """Split ``text`` into named blocks keyed by anchor matches.

    Returns a mapping ``key -> block_text`` for every key in ``header_keys``
    whose regex matched somewhere in ``text``. Keys whose anchor never fired
    are absent. Block text runs from just after the matched header to the
    next header (in source order, regardless of which key it belongs to) or
    to the end of the string.
    """
    if not text:
        return {}
    matches: List[Tuple[int, int, str]] = []
    for key in header_keys:
        rx = _SECTION_HEADER_RES.get(key)
        if not rx:
            continue
        m = rx.search(text)
        if m:
            matches.append((m.start(), m.end(), key))
    matches.sort(key=lambda t: t[0])
    out: Dict[str, str] = {}
    for i, (start, end, key) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        block = text[end:next_start].strip().lstrip(":").strip()
        if block:
            out[key] = block
    return out


def extract_from_procedure(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lift Inputs / Steps / Output / Worked Example into procedure pairs.

    Pair shape:
        instruction = "Given <inputs>, <procedure_name>: produce <output>."
        output      = step-by-step text (Steps section verbatim, optionally
                      with Worked Example appended).
    """
    text = chunk.get("text") or ""
    if not text.strip():
        return []
    blocks = _split_by_headers(
        text,
        ("when_to_use", "inputs", "steps", "output", "worked_example"),
    )
    inputs = blocks.get("inputs", "")
    steps = blocks.get("steps", "")
    output_blk = blocks.get("output", "")
    worked = blocks.get("worked_example", "")
    if not steps:
        return []
    procedure_name = _flatten(chunk.get("section_heading") or chunk.get("title") or "")
    if not procedure_name:
        # First sentence of text as fallback procedure name.
        first = _SENTENCE_END_RE.split(text, maxsplit=1)
        if first:
            procedure_name = _flatten(first[0])[:160]
    procedure_name = procedure_name or "Run the following procedure"
    inputs_phrase = _flatten(inputs) if inputs else "the appropriate inputs"
    output_phrase = _flatten(output_blk) if output_blk else "the documented output"
    instruction = (
        f"Given {inputs_phrase}, perform the procedure {procedure_name}: "
        f"produce {output_phrase}."
    )
    out_text = _flatten(steps)
    if worked:
        out_text = f"{out_text} Worked example: {_flatten(worked)}"
    pair = _shape_pair(
        instruction=instruction,
        output=out_text,
        chunk=chunk,
        extraction_method=METHOD_PROCEDURE,
        extra_meta={
            "has_inputs": bool(inputs),
            "has_output_section": bool(output_blk),
            "has_worked_example": bool(worked),
            "procedure_name": procedure_name,
        },
    )
    return [pair] if pair else []


def extract_from_real_world_scenario(
    chunk: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Lift Scenario / Your Task / Approach / Success Criteria.

    Pair shape:
        instruction = "<scenario context>. Task: <task statement>"
        output      = "<approach text>. Success criteria: <criteria>"
    """
    text = chunk.get("text") or ""
    if not text.strip():
        return []
    blocks = _split_by_headers(
        text,
        ("scenario", "your_task", "approach", "success_criteria"),
    )
    task = blocks.get("your_task", "")
    approach = blocks.get("approach", "")
    if not task or not approach:
        return []
    # Scenario context: the explicit Scenario block, falling back to whatever
    # text precedes the Your Task header.
    scenario = blocks.get("scenario", "")
    if not scenario:
        rx = _SECTION_HEADER_RES["your_task"]
        m = rx.search(text)
        if m:
            scenario = text[: m.start()].strip()
    scenario = _flatten(scenario)[:1200]
    if not scenario:
        return []
    criteria = blocks.get("success_criteria", "")
    instruction = f"{scenario}. Task: {_flatten(task)}"
    out_text = _flatten(approach)
    if criteria:
        out_text = f"{out_text} Success criteria: {_flatten(criteria)}"
    pair = _shape_pair(
        instruction=instruction,
        output=out_text,
        chunk=chunk,
        extraction_method=METHOD_REAL_WORLD_SCENARIO,
        extra_meta={
            "has_success_criteria": bool(criteria),
            "scenario_chars": len(scenario),
        },
    )
    return [pair] if pair else []


def extract_from_common_pitfall(
    chunk: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Lift the misconception-distinguish + multi-arm pair from a pitfall.

    Emits up to two pairs per chunk:
      1. ``common_pitfall`` — distinguish-style pair where the instruction
         restates the misconception and the output explains why it's wrong
         and what the correct approach is.
      2. ``common_pitfall_multi_arm`` — situation-aware pair where the
         instruction names the situation + the misconception and asks for
         the correct approach; the output is the right approach plus the
         underlying reasoning (Why it's wrong block).
    """
    text = chunk.get("text") or ""
    if not text.strip():
        return []
    blocks = _split_by_headers(
        text,
        (
            "what_looks_right",
            "why_wrong",
            "right_approach",
            "quick_test",
        ),
    )
    misconception = blocks.get("what_looks_right", "")
    why_wrong = blocks.get("why_wrong", "")
    right_approach = blocks.get("right_approach", "")
    if not misconception or not right_approach:
        return []
    misconception_f = _flatten(misconception)
    why_wrong_f = _flatten(why_wrong) if why_wrong else ""
    right_approach_f = _flatten(right_approach)
    quick_test_f = _flatten(blocks.get("quick_test", ""))

    out: List[Dict[str, Any]] = []
    # Pair 1: distinguish-style
    instr_distinguish = (
        f"Is this the right approach? '{misconception_f[:600]}'"
    )
    if why_wrong_f:
        out_distinguish = (
            f"No. {why_wrong_f} The correct approach: {right_approach_f}"
        )
    else:
        out_distinguish = f"No. The correct approach: {right_approach_f}"
    pair1 = _shape_pair(
        instruction=instr_distinguish,
        output=out_distinguish,
        chunk=chunk,
        extraction_method=METHOD_COMMON_PITFALL,
        extra_meta={
            "misconception_excerpt": misconception_f[:200],
            "has_why_wrong": bool(why_wrong_f),
            "has_quick_test": bool(quick_test_f),
        },
    )
    if pair1:
        out.append(pair1)

    # Pair 2: multi-arm — situation, common mistake, ask for correct approach.
    # The "situation" is the chunk's section heading or the first sentence
    # of misconception text.
    situation = _flatten(chunk.get("section_heading") or "")
    if not situation:
        first = _SENTENCE_END_RE.split(text, maxsplit=1)
        if first:
            situation = _flatten(first[0])[:200]
    situation = situation or "the following situation"
    instr_multi = (
        f"In {situation}, a common mistake is: {misconception_f[:500]}. "
        f"What is the correct approach and why?"
    )
    out_multi = right_approach_f
    if why_wrong_f:
        out_multi = (
            f"{right_approach_f} Reasoning: {why_wrong_f}"
        )
    pair2 = _shape_pair(
        instruction=instr_multi,
        output=out_multi,
        chunk=chunk,
        extraction_method=METHOD_COMMON_PITFALL_MULTI_ARM,
        extra_meta={
            "situation": situation[:200],
            "misconception_excerpt": misconception_f[:200],
        },
    )
    if pair2:
        out.append(pair2)
    return out


def extract_from_problem_solution(
    chunk: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Lift main pair + DPO-shape pair from a problem-solution walkthrough.

    Emits up to two pairs:
      1. ``problem_solution`` — main instruction-response pair
         (problem statement -> walkthrough).
      2. ``problem_solution_dpo`` — DPO-shape pair where the walkthrough
         is the ``chosen`` response and the ``Common Incorrect Approach``
         block is the ``rejected`` response.
    """
    text = chunk.get("text") or ""
    if not text.strip():
        return []
    blocks = _split_by_headers(
        text,
        ("problem", "walkthrough", "common_incorrect", "verification"),
    )
    problem = blocks.get("problem", "")
    walkthrough = blocks.get("walkthrough", "")
    counter = blocks.get("common_incorrect", "")
    if not problem or not walkthrough:
        return []
    problem_f = _flatten(problem)
    walkthrough_f = _flatten(walkthrough)
    verification_f = _flatten(blocks.get("verification", ""))
    out: List[Dict[str, Any]] = []
    # Main pair
    main_out = walkthrough_f
    if verification_f:
        main_out = f"{walkthrough_f} Verification: {verification_f}"
    pair_main = _shape_pair(
        instruction=problem_f,
        output=main_out,
        chunk=chunk,
        extraction_method=METHOD_PROBLEM_SOLUTION,
        extra_meta={
            "has_verification": bool(verification_f),
            "has_counter_example": bool(counter),
        },
    )
    if pair_main:
        out.append(pair_main)
    # DPO-shape pair: walkthrough = chosen, counter-example = rejected.
    if counter:
        counter_f = _flatten(counter)
        dpo_pair = _shape_dpo_pair(
            instruction=problem_f,
            chosen=walkthrough_f,
            rejected=counter_f,
            chunk=chunk,
            extraction_method=METHOD_PROBLEM_SOLUTION_DPO,
            extra_meta={
                "chosen_chars": len(walkthrough_f),
                "rejected_chars": len(counter_f),
            },
        )
        if dpo_pair:
            out.append(dpo_pair)
    return out


def _shape_dpo_pair(
    *,
    instruction: str,
    chosen: str,
    rejected: str,
    chunk: Dict[str, Any],
    extraction_method: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble a DPO-flavoured pair: ``chosen`` + ``rejected`` siblings
    alongside the canonical ``output`` field (set to ``chosen``).

    Downstream stratified DPO emitters look for ``chosen``/``rejected``
    keys; legacy SFT consumers continue to read ``output``. Both paths see
    the same record shape so we don't have to fork the JSONL stream.
    """
    instruction = _flatten(instruction)
    chosen = _flatten(chosen)
    rejected = _flatten(rejected)
    if not instruction or not chosen or not rejected:
        return None
    pair: Dict[str, Any] = {
        "instruction": instruction,
        "output": chosen,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": _build_metadata(
            chunk=chunk,
            extraction_method=extraction_method,
            extra=extra_meta,
        ),
        "schema_version": "v1",
    }
    return pair


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
        # Wave 81: template-aware extractors keyed on the four new
        # chunk_type values that are now propagated from Courseforge
        # data-cf-template-type (Wave 79 C). Each emits richer pairs than
        # the explanation_template fallback because the source HTML has
        # template-specific subsection structure.
        if chunk_type == "procedure" and METHOD_PROCEDURE in method_set:
            emitted.extend(extract_from_procedure(chunk))
        if (
            chunk_type == "real_world_scenario"
            and METHOD_REAL_WORLD_SCENARIO in method_set
        ):
            emitted.extend(extract_from_real_world_scenario(chunk))
        if chunk_type == "common_pitfall" and (
            METHOD_COMMON_PITFALL in method_set
            or METHOD_COMMON_PITFALL_MULTI_ARM in method_set
        ):
            for pair in extract_from_common_pitfall(chunk):
                method = pair["metadata"]["extraction_method"]
                if method in method_set:
                    emitted.append(pair)
        if chunk_type == "problem_solution" and (
            METHOD_PROBLEM_SOLUTION in method_set
            or METHOD_PROBLEM_SOLUTION_DPO in method_set
        ):
            for pair in extract_from_problem_solution(chunk):
                method = pair["metadata"]["extraction_method"]
                if method in method_set:
                    emitted.append(pair)
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
