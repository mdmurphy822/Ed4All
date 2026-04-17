#!/usr/bin/env python3
"""
Trainforge Instruction Pair Factory

Synthesizes SFT-style (prompt, completion) training pairs from an enriched
Trainforge chunk. This is the mock-provider path: deterministic templates,
no LLM call, no network.

Design constraints (see Worker C plan):
- One function = one pair. The stage composes many calls.
- Only chunks with non-empty ``learning_outcome_refs`` produce pairs.
- The prompt MUST NOT contain any 50+-char verbatim span from ``chunk.text``.
- Prompt is 40-400 chars; completion is 50-600 chars.
- Same chunk + same seed -> identical pair (deterministic).
- Emits the pair dict PLUS a ``quality`` dict with the gate numbers so the
  stage can log them verbatim in decision capture.

The factory never raises on a quality-gate miss: it returns the best pair
it could build and the quality dict documents which gates passed. The stage
decides whether to drop or keep.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Maximum allowed verbatim span (in chars) from chunk.text that may appear
# in the prompt. Hard limit from plan's quality gates.
MAX_VERBATIM_SPAN = 50

# Prompt and completion length gates (chars).
PROMPT_MIN, PROMPT_MAX = 40, 400
COMPLETION_MIN, COMPLETION_MAX = 50, 600


# ---------------------------------------------------------------------------
# Prompt template catalog
# ---------------------------------------------------------------------------
#
# Each template is keyed by (bloom_level, content_type). Values are a short
# instruction stem that the factory fills with the chunk's topic signal
# (first concept tag or LO ref). Templates are intentionally generic so they
# do not quote chunk text verbatim.
#
# Unknown combinations fall back to the ``_default`` row for the bloom level
# and finally to ``("understand", "_default")``.

_BLOOM_LEVELS = ("remember", "understand", "apply", "analyze", "evaluate", "create")

TEMPLATE_CATALOG: Dict[Tuple[str, str], str] = {
    # remember
    ("remember", "explanation"): "In one or two sentences, define the key term associated with {topic}.",
    ("remember", "example"):     "Name the core concept illustrated by the example related to {topic}.",
    ("remember", "procedure"):   "List the high-level steps of the procedure for {topic}.",
    ("remember", "comparison"):  "Name the two items being compared in the discussion of {topic}.",
    ("remember", "_default"):    "State the definition of the central concept behind {topic}.",

    # understand
    ("understand", "explanation"): "Explain in your own words what {topic} means and why it matters.",
    ("understand", "example"):     "Describe what the example of {topic} is meant to illustrate.",
    ("understand", "procedure"):   "Summarize, at a high level, the procedure that applies to {topic}.",
    ("understand", "comparison"):  "Describe the key similarity and the key difference in the comparison of {topic}.",
    ("understand", "_default"):    "Explain the core idea behind {topic} for a learner new to the topic.",

    # apply
    ("apply", "explanation"): "Describe a concrete situation where you would apply the concept of {topic}, and say why.",
    ("apply", "example"):     "Given a new scenario loosely related to {topic}, explain how the example's lesson would carry over.",
    ("apply", "procedure"):   "Walk through the procedure for {topic} as if guiding a colleague doing it for the first time.",
    ("apply", "comparison"):  "Choose between the two options in the comparison of {topic} for a specific use case and justify the choice.",
    ("apply", "_default"):    "Use the idea of {topic} to resolve a realistic problem; describe both the problem and your approach.",

    # analyze
    ("analyze", "explanation"): "Break the idea of {topic} into its component parts and explain how they relate.",
    ("analyze", "example"):     "Analyze what the example for {topic} reveals about the underlying concept.",
    ("analyze", "procedure"):   "Analyze the procedure for {topic}: where are the failure modes and why?",
    ("analyze", "comparison"):  "Analyze the trade-offs surfaced by the comparison of {topic}.",
    ("analyze", "_default"):    "Analyze the structure of {topic} and identify its most important relationships.",

    # evaluate
    ("evaluate", "explanation"): "Evaluate whether the core claim about {topic} is well supported, and say what would strengthen it.",
    ("evaluate", "example"):     "Evaluate how well the example for {topic} demonstrates the concept it is meant to show.",
    ("evaluate", "procedure"):   "Critique the procedure for {topic}: what are its strengths and limitations?",
    ("evaluate", "comparison"):  "Judge which side of the comparison of {topic} is better supported and explain your criteria.",
    ("evaluate", "_default"):    "Assess the effectiveness of {topic} against a clear criterion you state up front.",

    # create
    ("create", "explanation"): "Propose a new explanation or analogy for {topic} that would help a novice learner.",
    ("create", "example"):     "Invent a fresh example that illustrates {topic} in a context different from the one in the material.",
    ("create", "procedure"):   "Design a simplified variant of the procedure for {topic} suitable for a beginner.",
    ("create", "comparison"):  "Create a new axis of comparison that would be useful when discussing {topic}.",
    ("create", "_default"):    "Design a short activity that teaches {topic} to a learner new to the material.",
}


@dataclass
class InstructionSynthesisResult:
    """Result returned by :func:`synthesize_instruction_pair`."""

    pair: Optional[Dict[str, Any]]
    quality: Dict[str, Any]
    template_id: str
    rationale: str
    topic: str
    alternatives: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Flatten HTML to plain text. Deterministic; no external deps."""
    if not text:
        return ""
    s = _HTML_TAG_RE.sub(" ", text)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()


def _contains_verbatim_span(prompt: str, chunk_text: str, max_span: int = MAX_VERBATIM_SPAN) -> bool:
    """Return True if ``prompt`` contains any span of >= ``max_span`` consecutive
    chars that also appears in ``chunk_text``.

    Uses a sliding window over the *prompt* (the shorter string in practice).
    Both inputs are HTML-stripped lowercased for comparison.
    """
    if not prompt or not chunk_text:
        return False
    p = prompt.lower()
    c = _strip_html(chunk_text).lower()
    if len(p) < max_span or len(c) < max_span:
        return False
    for i in range(0, len(p) - max_span + 1):
        window = p[i:i + max_span]
        if window in c:
            return True
    return False


def _derive_topic(chunk: Dict[str, Any]) -> str:
    """Derive a short topic phrase for template filling.

    Priority: concept_tags[0] -> first key_term.term -> first LO ref.
    Never returns an empty string (caller guarantees LO refs exist).
    """
    tags = chunk.get("concept_tags") or []
    if tags:
        t = str(tags[0]).strip()
        if t:
            return t.replace("-", " ").replace("_", " ")
    key_terms = chunk.get("key_terms") or []
    if key_terms and isinstance(key_terms[0], dict):
        term = key_terms[0].get("term")
        if term:
            return str(term).strip()
    lo_refs = chunk.get("learning_outcome_refs") or []
    if lo_refs:
        return f"learning outcome {lo_refs[0]}"
    return "the course topic"


def _normalize_bloom(bloom: Optional[str]) -> str:
    if not bloom:
        return "understand"
    b = str(bloom).strip().lower()
    if b in _BLOOM_LEVELS:
        return b
    return "understand"


def _normalize_content_type(chunk: Dict[str, Any]) -> str:
    label = chunk.get("content_type_label")
    if label:
        return str(label).strip().lower()
    ct = chunk.get("chunk_type")
    if ct:
        return str(ct).strip().lower()
    return "explanation"


def _select_template(bloom: str, content_type: str) -> Tuple[str, str]:
    """Return (template_id, template_string). Falls back as documented above."""
    key = (bloom, content_type)
    if key in TEMPLATE_CATALOG:
        return f"{bloom}.{content_type}", TEMPLATE_CATALOG[key]
    key = (bloom, "_default")
    if key in TEMPLATE_CATALOG:
        return f"{bloom}._default", TEMPLATE_CATALOG[key]
    return "understand._default", TEMPLATE_CATALOG[("understand", "_default")]


def _build_completion(chunk: Dict[str, Any], topic: str, bloom: str, content_type: str, rng: random.Random) -> str:
    """Build a deterministic completion that is not a verbatim chunk quote.

    Draws its content from structured chunk metadata (key_terms, misconceptions,
    concept_tags) so the completion is grounded but paraphrased.
    """
    parts: List[str] = []

    key_terms = chunk.get("key_terms") or []
    if key_terms and isinstance(key_terms[0], dict):
        kt = key_terms[0]
        term = str(kt.get("term", "")).strip()
        definition = str(kt.get("definition", "")).strip()
        if term and definition:
            # Paraphrase envelope: wrap the definition in a declarative frame.
            parts.append(f"The central idea behind {topic} is captured by the term '{term}'. {definition}")
        elif term:
            parts.append(f"The central idea behind {topic} is captured by the term '{term}'.")

    tags = [str(t) for t in (chunk.get("concept_tags") or []) if t]
    if tags and not parts:
        joined = ", ".join(tags[:3])
        parts.append(f"The treatment of {topic} draws on the related concepts {joined}.")

    # Add a bloom-flavored closing sentence so completion length and tone vary.
    bloom_tails = {
        "remember": f"Learners should be able to recall and restate this about {topic} without aid.",
        "understand": f"Learners should be able to explain this about {topic} in their own words.",
        "apply": f"Learners should be able to use this about {topic} in a new but similar situation.",
        "analyze": f"Learners should be able to break this down and explain the parts of {topic}.",
        "evaluate": f"Learners should be able to judge the quality of claims about {topic} against clear criteria.",
        "create": f"Learners should be able to generate a fresh example or application of {topic}.",
    }
    parts.append(bloom_tails.get(bloom, bloom_tails["understand"]))

    # If still too short, add a content-type-specific tail.
    completion = " ".join(parts).strip()
    if len(completion) < COMPLETION_MIN:
        completion += (
            f" In context, this content was delivered as a '{content_type}' section of the course, "
            f"which shapes how the idea should be used and assessed."
        )

    # Soft-cap at COMPLETION_MAX: trim on a sentence boundary if possible.
    if len(completion) > COMPLETION_MAX:
        hard = completion[:COMPLETION_MAX]
        last_period = hard.rfind(". ")
        if last_period > COMPLETION_MIN:
            completion = hard[:last_period + 1]
        else:
            completion = hard.rstrip() + "..."

    # Deterministic micro-variation so same-seed+different-chunk pairs differ
    # even when everything else collapses to the same template.
    if rng.random() < 0.5:
        completion = completion.replace(" Learners should be able to", " A proficient learner can")

    return completion


def _pair_hash(chunk_id: str, seed: int) -> str:
    h = hashlib.sha256()
    h.update(chunk_id.encode("utf-8"))
    h.update(b"|")
    h.update(str(int(seed)).encode("utf-8"))
    return h.hexdigest()[:16]


def _seed_rng(chunk_id: str, seed: int) -> random.Random:
    """Seed an RNG from (chunk_id, seed) so each call is deterministic."""
    digest = _pair_hash(chunk_id, seed)
    return random.Random(int(digest, 16))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_instruction_pair(
    chunk: Dict[str, Any],
    seed: int,
    provider: str = "mock",
) -> InstructionSynthesisResult:
    """Synthesize one instruction pair from an enriched chunk.

    Args:
        chunk: Enriched chunk dict from corpus/chunks.jsonl. Must have
            ``learning_outcome_refs`` (enforced by caller).
        seed: Deterministic seed. Same chunk + same seed -> same pair.
        provider: "mock" (implemented) or "anthropic" (future; raises for now).

    Returns:
        InstructionSynthesisResult. ``pair`` is None if any hard gate failed;
        ``quality`` explains which.
    """
    if provider != "mock":
        raise NotImplementedError(
            f"instruction synthesis provider '{provider}' is not implemented; "
            f"only 'mock' is wired in this release."
        )

    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    lo_refs = list(chunk.get("learning_outcome_refs") or [])
    if not chunk_id or not lo_refs:
        # Defense-in-depth; the stage filters these out first.
        return InstructionSynthesisResult(
            pair=None,
            quality={"passed": False, "reason": "missing_chunk_id_or_lo_refs"},
            template_id="none",
            rationale="Chunk is missing id or learning_outcome_refs; no pair produced.",
            topic="",
        )

    rng = _seed_rng(chunk_id, seed)
    bloom = _normalize_bloom(chunk.get("bloom_level"))
    content_type = _normalize_content_type(chunk)
    topic = _derive_topic(chunk)
    template_id, template = _select_template(bloom, content_type)

    prompt = template.format(topic=topic)

    # Enforce prompt length gate with a safe filler if too short.
    if len(prompt) < PROMPT_MIN:
        prompt = prompt + f" Frame your answer for a learner at the '{bloom}' cognitive level."
    if len(prompt) > PROMPT_MAX:
        prompt = prompt[: PROMPT_MAX - 3].rstrip() + "..."

    chunk_text = str(chunk.get("text") or "")
    leaked = _contains_verbatim_span(prompt, chunk_text)
    if leaked:
        # Try a rewrite that cannot match: swap the topic for a generic phrase.
        alt_prompt = template.format(topic=f"the concept in chunk {chunk_id}")
        if not _contains_verbatim_span(alt_prompt, chunk_text):
            prompt = alt_prompt
            leaked = False

    completion = _build_completion(chunk, topic, bloom, content_type, rng)

    quality = {
        "prompt_len": len(prompt),
        "completion_len": len(completion),
        "prompt_len_ok": PROMPT_MIN <= len(prompt) <= PROMPT_MAX,
        "completion_len_ok": COMPLETION_MIN <= len(completion) <= COMPLETION_MAX,
        "no_verbatim_leakage": not leaked,
    }
    quality["passed"] = (
        quality["prompt_len_ok"]
        and quality["completion_len_ok"]
        and quality["no_verbatim_leakage"]
    )

    if not quality["passed"]:
        return InstructionSynthesisResult(
            pair=None,
            quality=quality,
            template_id=template_id,
            rationale=(
                f"Instruction pair gated out: prompt_len_ok={quality['prompt_len_ok']}, "
                f"completion_len_ok={quality['completion_len_ok']}, "
                f"no_verbatim_leakage={quality['no_verbatim_leakage']}."
            ),
            topic=topic,
        )

    pair = {
        "prompt": prompt,
        "completion": completion,
        "chunk_id": chunk_id,
        "lo_refs": lo_refs,
        "bloom_level": bloom,
        "content_type": content_type,
        "seed": int(seed),
        # ``decision_capture_id`` is filled in by the stage after it logs the
        # decision, because only the stage owns the capture handle.
        "decision_capture_id": "",
        "template_id": template_id,
        "provider": provider,
        "schema_version": "v1",
    }

    rationale = (
        f"Selected template '{template_id}' for bloom='{bloom}' content_type='{content_type}' "
        f"targeting topic='{topic}'. Completion grounded in key_terms/concept_tags with a "
        f"bloom-level-specific closing sentence. Verbatim-span check against chunk.text passed."
    )

    return InstructionSynthesisResult(
        pair=pair,
        quality=quality,
        template_id=template_id,
        rationale=rationale,
        topic=topic,
        alternatives=[
            f"apply._default (rejected: pair targets '{bloom}' level, not 'apply')",
            f"{bloom}._default (rejected: content-type-specific template '{template_id}' is more specific)",
        ],
    )


__all__ = [
    "synthesize_instruction_pair",
    "InstructionSynthesisResult",
    "TEMPLATE_CATALOG",
    "MAX_VERBATIM_SPAN",
    "PROMPT_MIN",
    "PROMPT_MAX",
    "COMPLETION_MIN",
    "COMPLETION_MAX",
]
