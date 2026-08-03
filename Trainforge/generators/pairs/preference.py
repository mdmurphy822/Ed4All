#!/usr/bin/env python3
"""
Trainforge Canonical Preference Pair Factory

Synthesizes DPO-style (prompt, chosen, rejected) preference pairs from an
enriched Trainforge chunk. Mock-provider path: deterministic, no LLM call.

Design constraints:
- One function = one pair. The stage composes many calls.
- Only chunks with non-empty ``learning_outcome_refs`` produce pairs.
- The ``rejected`` completion is drawn from ``chunk.misconceptions`` when
  present; otherwise it is rule-synthesized from a deterministic distractor
  transform on the ``chosen`` completion.
- ``chosen`` != ``rejected``; token-Jaccard delta between the two >= 0.3.
- Prompt is 40-400 chars, completions are 50-600 chars each.
- No 50+-char verbatim span from ``chunk.text`` in the prompt.
- Deterministic under (chunk_id, seed).
- Emits pair PLUS quality dict (same contract as instruction_factory).
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from lib.decision_capture import DecisionAlternative
from lib.ontology.slugs import deslugify_concept

logger = logging.getLogger(__name__)


MAX_VERBATIM_SPAN = 50
PROMPT_MIN, PROMPT_MAX = 40, 400
COMPLETION_MIN, COMPLETION_MAX = 50, 600
JACCARD_DELTA_MIN = 0.3


@dataclass
class PreferenceSynthesisResult:
    """Result returned by :func:`synthesize_preference_pair`."""

    pair: Optional[Dict[str, Any]]
    quality: Dict[str, Any]
    rationale: str
    source: str  # "misconception" or "rule_synthesized"
    misconception_id: Optional[str] = None
    alternatives: List[DecisionAlternative] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt templates (preference pairs use a single mature template family so
# the question is always the same across chosen and rejected -- that's the
# DPO invariant: shared prompt, competing completions).
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES = {
    "misconception": (
        "A learner new to the material says the following about {topic}. "
        "Briefly explain whether they are correct and why."
    ),
    "explanation": (
        "Explain the concept associated with {topic} clearly enough for a "
        "new learner to avoid the most common misunderstanding."
    ),
    "application": (
        "Describe how you would apply the idea behind {topic} in a short "
        "realistic scenario, and flag one wrong way to do it."
    ),
}


# ---------------------------------------------------------------------------
# Helpers (shared in spirit with instruction_factory; kept local to avoid a
# cross-module import cycle between two sibling factories).
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _strip_html(text: str) -> str:
    if not text:
        return ""
    s = _HTML_TAG_RE.sub(" ", text)
    s = html.unescape(s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def _clean_answer_text(text: str) -> str:
    cleaned = _strip_html(text)
    cleaned = re.sub(r"^(?:CO|TO)-\d+:\s*", "", cleaned)
    return cleaned.strip()


def _looks_like_fragment(text: str) -> bool:
    return str(text or "").strip().endswith(":")


# Assessment-scaffolding patterns (kept in lockstep with instruction_factory).
_ASSESSMENT_SCAFFOLD_PATTERNS = [
    re.compile(r'\bQuestion\s+\d+\s*\(\s*[A-Z]+-\d+\s*,?\s*Bloom\s*:', re.IGNORECASE),
    re.compile(r'\b(?:Q|Item)\s*\d+\s*\(\s*[A-Z]+-\d+\b'),
    re.compile(r'\b(?:Bloom|Cognitive)\s*:\s*(?:Remember|Understand|Apply|Analyze|Evaluate|Create)\)', re.IGNORECASE),
]


def _contains_assessment_scaffolding(text: str) -> bool:
    """True when text contains an assessment-outline marker."""
    if not text:
        return False
    for pat in _ASSESSMENT_SCAFFOLD_PATTERNS:
        if pat.search(text):
            return True
    return False


def _contains_verbatim_span(prompt: str, chunk_text: str, max_span: int = MAX_VERBATIM_SPAN) -> bool:
    if not prompt or not chunk_text:
        return False
    p = prompt.lower()
    c = _strip_html(chunk_text).lower()
    if len(p) < max_span or len(c) < max_span:
        return False
    for i in range(0, len(p) - max_span + 1):
        if p[i:i + max_span] in c:
            return True
    return False


def _tokenize(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _derive_topic(chunk: Dict[str, Any]) -> str:
    """Return a human-readable topic phrase, or ``""`` when none exists.

    Mirrors ``instruction_factory._derive_topic``: there is deliberately NO
    LO-id fallback. Interpolating an opaque learning-outcome identifier into a
    learner-facing prompt slot produces syntactically valid, semantically
    empty preference data. The caller treats ``""`` as an ineligibility
    signal (no design-intent fallbacks).
    """
    tags = chunk.get("concept_tags") or []
    if tags:
        # deslugify_concept strips a trailing ``-(co|to)-NN`` LO-ref suffix
        # before the hyphen-to-space transform, so ``property-paths-co-15``
        # becomes ``property paths`` instead of bleeding ``co 15`` artifact
        # tokens into the prompt. A plain hyphen-to-space replace would not.
        derived = deslugify_concept(str(tags[0])).strip()
        if derived:
            return derived
    key_terms = chunk.get("key_terms") or []
    if key_terms and isinstance(key_terms[0], dict):
        term = key_terms[0].get("term")
        if term and str(term).strip():
            return str(term).strip()
    return ""


def _seed_rng(chunk_id: str, seed: int) -> random.Random:
    h = hashlib.sha256()
    h.update(chunk_id.encode("utf-8"))
    h.update(b"|pref|")
    h.update(str(int(seed)).encode("utf-8"))
    return random.Random(int(h.hexdigest(), 16))


def _misconception_id(
    misconception_text: str,
    correction_text: str,
    bloom_level: Optional[str] = None,
) -> str:
    """Content-hash misconception ID, form ``mc_<16-hex-char sha256>``.

    Stable across runs and across chunk re-chunking (unlike a
    position-based ``{chunk_id}_mc_{index}`` scheme).

    ``bloom_level`` joins the seed so two misconceptions sharing statement
    + correction text but different Bloom cognitive demands emit distinct
    IDs. The seed has two forms:

    * ``{statement}|{correction}|{bloom_level}`` when a bloom level is
      supplied, and
    * ``{statement}|{correction}`` when it is not.

    The two forms matter: always appending the (empty) bloom segment would
    add a trailing ``|`` and silently rekey every misconception in a
    corpus that predates bloom levels. Outer whitespace is normalised but
    inner whitespace is preserved, so cosmetic edits do not churn IDs but
    real text edits do.
    """
    # Delegated to ``lib.ontology.misconception_id.canonical_mc_id`` so this
    # site, ``process_course._build_misconceptions_for_graph``, and
    # ``pedagogy_graph_builder._mc_id`` share one source of truth — the three
    # must stay in lock-step or IDs diverge between corpus and graph.
    from lib.ontology.misconception_id import canonical_mc_id
    return canonical_mc_id(misconception_text, correction_text, bloom_level)


def _clamp_length(text: str, lo: int, hi: int, pad_hint: str) -> str:
    """Pad with ``pad_hint`` if shorter than ``lo``; trim at sentence boundary
    if longer than ``hi``."""
    if len(text) < lo:
        text = (text + " " + pad_hint).strip()
    if len(text) > hi:
        hard = text[:hi]
        period = hard.rfind(". ")
        if period > lo:
            text = hard[:period + 1]
        else:
            text = hard.rstrip() + "..."
    return text


# ---------------------------------------------------------------------------
# Chosen/Rejected builders
# ---------------------------------------------------------------------------

def _build_chosen(
    chunk: Dict[str, Any],
    topic: str,
    misconception: Optional[Dict[str, Any]] = None,
    *,
    disallow_summary: bool = False,
) -> str:
    """Build the preferred (chosen) completion -- grounded and correct.

    ``disallow_summary=True`` skips the ``chunk.summary`` branch; callers
    use it to retry after a verbatim-leakage hit, since ``chunk.summary``
    is often a near-verbatim extract of the source.
    """
    parts: List[str] = []

    if misconception:
        correction = _clean_answer_text(str(misconception.get("correction", "")))
        if correction and not _looks_like_fragment(correction):
            return _clamp_length(
                correction,
                COMPLETION_MIN,
                COMPLETION_MAX,
                pad_hint=(
                    f"This correction matters for {topic} because it prevents "
                    f"the learner from applying the wrong mental model."
                ),
            )

    summary = _clean_answer_text(str(chunk.get("summary") or ""))
    if summary and not disallow_summary:
        parts.append(summary)

    key_terms = chunk.get("key_terms") or []
    if not parts and key_terms and isinstance(key_terms[0], dict):
        kt = key_terms[0]
        term = str(kt.get("term", "")).strip()
        definition = _clean_answer_text(str(kt.get("definition", "")).strip())
        if term and definition:
            parts.append(f"{term} is the key term for {topic}: {definition}")
        elif term:
            parts.append(f"{term} is the key term for {topic}.")

    tags = [str(t) for t in (chunk.get("concept_tags") or []) if t]
    if tags and not parts:
        joined = ", ".join(tags[:3])
        # Scaffolding rotation: same phrasing set and same chunk_id-hash
        # deterministic selection as instruction_factory, so the two
        # factories don't emit one identical stock sentence corpus-wide.
        scaffolds = [
            f"{topic} is best understood by tying {joined} to the underlying schema, not by listing labels.",
            f"A learner working on {topic} should ground each idea in {joined} rather than memorising surface terms.",
            f"To master {topic}, connect each piece to {joined} as concrete schema mechanics, not vocabulary.",
            f"{topic} is built from the interplay of {joined}; treat them as operative roles, not labels.",
        ]
        chunk_id_for_hash = str(chunk.get("id") or chunk.get("chunk_id") or topic)
        idx = int(
            hashlib.sha256(chunk_id_for_hash.encode("utf-8")).hexdigest(), 16
        ) % len(scaffolds)
        parts.append(scaffolds[idx])

    # Course-level grounding sentence so the answer reads as an explanation
    # rather than a bare fact.
    parts.append(
        f"A correct response describes {topic} accurately and notes at least one "
        f"common pitfall learners should avoid."
    )

    chosen = " ".join(parts).strip()
    chosen = _clamp_length(
        chosen,
        COMPLETION_MIN,
        COMPLETION_MAX,
        pad_hint=(
            f"Framing this around {topic} helps learners avoid common misunderstandings "
            f"and apply the concept correctly."
        ),
    )
    return chosen


def _build_rejected_from_misconception(misconception: Dict[str, Any], topic: str) -> str:
    """Wrap a misconception in first-person framing so it reads as a plausible
    but wrong answer (the thing DPO learns to down-weight)."""
    mc_text = str(misconception.get("misconception", "")).strip()
    if not mc_text:
        return ""
    rejected = (
        f"Yes, that's essentially right. In my experience with {topic}, {mc_text} "
        f"That's a fair summary and you can rely on it."
    )
    return _clamp_length(
        rejected,
        COMPLETION_MIN,
        COMPLETION_MAX,
        pad_hint=f"Overall, I'd say this framing of {topic} works for most practical cases.",
    )


_NEGATION_SWAPS = [
    (r"\baccurately\b", "loosely"),
    (r"\bcorrectly\b", "approximately"),
    (r"\bcorrect\b", "rough"),
    (r"\bgrounded\b", "loosely tied"),
    (r"\bbest captured\b", "vaguely suggested"),
    (r"\bidea\b", "vibe"),
    (r"\bdescribes\b", "alludes to"),
    (r"\bavoid\b", "embrace"),
    (r"\bpitfall\b", "habit"),
    (r"\bcommon\b", "rare"),
]


# Rotated phrasings for the fallback token-stuffing path. Used ONLY when
# the chunk's CURIE has no anchored FORM_DATA entry (degraded_placeholder
# OR non-manifest). The primary path embeds an actual definition sentence
# drawn from FORM_DATA — see the dispatch in
# ``_enforce_preserve_tokens_in_preference``.
_PROMPT_REFERENCE_PHRASINGS: List[str] = [
    " (Reference: {tokens}.)",
    " (Relevant terms: {tokens}.)",
    " (See: {tokens}.)",
    " (In context: {tokens}.)",
]
_CHOSEN_REFERENCE_PHRASINGS: List[str] = [
    " Canonical terms: {tokens}.",
    " The relevant terms are {tokens}.",
    " Key vocabulary: {tokens}.",
    " This concerns {tokens}.",
]


def _select_phrasing(phrasings: List[str], chunk_id: str) -> str:
    """Deterministic phrasing selection by chunk_id hash."""
    if not phrasings:
        return ""
    idx = int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest(), 16) % len(phrasings)
    return phrasings[idx]


def _chunk_id_hash_int(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest(), 16)


def _append_anchored_to_prompt(prompt: str, anchor_definition: str) -> str:
    """Append a recall-style hook embedding the anchored definition;
    length-clamped against ``PROMPT_MAX``."""
    suffix = f" Recall how {anchor_definition}"
    if not suffix.endswith("."):
        suffix = suffix + "."
    new_prompt = prompt.rstrip() + suffix
    if len(new_prompt) > PROMPT_MAX:
        budget = PROMPT_MAX - len(suffix)
        new_prompt = prompt[:max(budget, 0)].rstrip() + suffix
    return new_prompt


def _append_anchored_to_chosen(chosen: str, anchor_definition: str) -> str:
    """Append the full anchored definition to the chosen completion;
    length-clamped against ``COMPLETION_MAX``."""
    addition = f" {anchor_definition}".rstrip()
    if not addition.endswith("."):
        addition = addition + "."
    new_chosen = chosen.rstrip() + addition
    if len(new_chosen) > COMPLETION_MAX:
        budget = COMPLETION_MAX - len(addition)
        if budget < COMPLETION_MIN:
            new_chosen = (
                chosen[:max(COMPLETION_MIN - len(addition), 0)].rstrip()
                + addition
            )
        else:
            new_chosen = chosen[:budget].rstrip() + addition
    return new_chosen


def _emit_degraded_capture(
    capture: Optional[Any],
    *,
    curie: str,
    chunk_id: str,
    side: str,
) -> None:
    """Mirror of the instruction-factory helper. Emits
    ``form_data_degraded_placeholder_skipped`` when an anchored-injection
    attempt falls back to the token-stuffing path.
    """
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type="form_data_degraded_placeholder_skipped",
            decision=(
                f"Force-injection for CURIE {curie} on chunk {chunk_id} "
                f"({side}) fell back to legacy token-stuffing because "
                f"the FORM_DATA entry is degraded_placeholder or absent."
            ),
            rationale=(
                f"Wave 135a FORM_DATA contract: CURIE={curie} has "
                f"either no entry or anchored_status="
                f"'degraded_placeholder' in _RDF_SHACL_FALLBACK_FORM_DATA. "
                f"Wave 135b's anchored-injection path requires "
                f"anchored_status='complete' for the entry's definitions "
                f"to be embedded in the pair body. Operator backfill of "
                f"the entry's anchored content (definitions + "
                f"usage_examples) flips the status to 'complete' and "
                f"silently improves the trained adapter's anchored-"
                f"injection coverage on chunks containing this CURIE."
            ),
            context=f"chunk_id={chunk_id}; curie={curie}; side={side}",
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "form_data_degraded_placeholder_skipped capture failed: %s", exc
        )


def _enforce_preserve_tokens_in_preference(
    pair: Dict[str, Any],
    preserve_tokens: List[str],
    *,
    capture: Optional[Any] = None,
) -> Dict[str, Any]:
    """Anchored force-injection on prompt + ``chosen``.

    Mirrors the instruction-factory dispatcher (see that module's
    docstring for the full contract). NEVER touches the ``rejected``
    field — DPO needs the misconception signal to reach the trainer
    intact, and the rejected completion legitimately may not contain
    the literal CURIE.
    """
    if not preserve_tokens:
        return pair
    from Trainforge.generators.deterministic.schema_translation_generator import (
        _RDF_SHACL_FALLBACK_FORM_DATA,
        resolve_anchor_text_for_curie,
    )

    prompt = str(pair.get("prompt") or "")
    chosen = str(pair.get("chosen") or "")
    chunk_id = str(pair.get("chunk_id") or "")
    chunk_hash = _chunk_id_hash_int(chunk_id) if chunk_id else 0

    missing_prompt = [t for t in preserve_tokens if t and t not in prompt]
    missing_chosen = [t for t in preserve_tokens if t and t not in chosen]

    # ---- Prompt side -----------------------------------------------------
    degraded_prompt: List[str] = []
    for token in missing_prompt:
        prompt_anchor, _, status = resolve_anchor_text_for_curie(
            token, _RDF_SHACL_FALLBACK_FORM_DATA, chunk_hash,
        )
        if status == "anchored" and prompt_anchor:
            new_prompt = _append_anchored_to_prompt(prompt, prompt_anchor)
            if token in new_prompt:
                prompt = new_prompt
                pair.setdefault("preserve_tokens_anchored_prompt", []).append(token)
                continue
        degraded_prompt.append(token)
        logger.warning(
            "Wave 135b: prompt-side anchored-injection fell back to "
            "token-stuffing for CURIE %s on chunk %s (status=%s)",
            token, chunk_id, status,
        )
        _emit_degraded_capture(
            capture, curie=token, chunk_id=chunk_id, side="prompt",
        )

    if degraded_prompt:
        template = _select_phrasing(_PROMPT_REFERENCE_PHRASINGS, chunk_id)
        prompt_add = template.format(tokens=", ".join(degraded_prompt))
        new_prompt = prompt.rstrip() + prompt_add
        if len(new_prompt) > PROMPT_MAX:
            budget = PROMPT_MAX - len(prompt_add)
            new_prompt = prompt[:max(budget, 0)].rstrip() + prompt_add
        prompt = new_prompt
        pair.setdefault("preserve_tokens_injected_prompt", []).extend(degraded_prompt)
    pair["prompt"] = prompt

    # ---- Chosen side -----------------------------------------------------
    degraded_chosen: List[str] = []
    for token in missing_chosen:
        _, completion_anchor, status = resolve_anchor_text_for_curie(
            token, _RDF_SHACL_FALLBACK_FORM_DATA, chunk_hash,
        )
        if status == "anchored" and completion_anchor:
            new_chosen = _append_anchored_to_chosen(chosen, completion_anchor)
            if token in new_chosen:
                chosen = new_chosen
                pair.setdefault("preserve_tokens_anchored", []).append(token)
                continue
        degraded_chosen.append(token)
        logger.warning(
            "Wave 135b: chosen-side anchored-injection fell back to "
            "token-stuffing for CURIE %s on chunk %s (status=%s)",
            token, chunk_id, status,
        )
        _emit_degraded_capture(
            capture, curie=token, chunk_id=chunk_id, side="chosen",
        )

    if degraded_chosen:
        template = _select_phrasing(_CHOSEN_REFERENCE_PHRASINGS, chunk_id)
        addition = template.format(tokens=", ".join(degraded_chosen))
        new_chosen = chosen.rstrip() + addition
        if len(new_chosen) > COMPLETION_MAX:
            budget = COMPLETION_MAX - len(addition)
            if budget < COMPLETION_MIN:
                new_chosen = (
                    chosen[:max(COMPLETION_MIN - len(addition), 0)].rstrip()
                    + addition
                )
            else:
                new_chosen = chosen[:budget].rstrip() + addition
        chosen = new_chosen
        pair.setdefault("preserve_tokens_injected", []).extend(degraded_chosen)
    pair["chosen"] = chosen

    return pair


def _rule_synthesize_rejected(chosen: str, topic: str, rng: random.Random) -> str:
    """Deterministic distractor: rewrite ``chosen`` with negation swaps plus a
    confidently-wrong closing sentence. Keeps length in range and guarantees
    enough token turnover to hit the Jaccard delta gate."""
    rejected = chosen
    for pattern, replacement in _NEGATION_SWAPS:
        rejected = re.sub(pattern, replacement, rejected, flags=re.IGNORECASE)

    # Append a confidently-wrong closing to inject distinct tokens. The exact
    # filler is one of a few deterministic variants so same-seed runs are stable.
    fillers = [
        f"Honestly, you don't really need to worry about {topic} in most situations.",
        f"The details of {topic} aren't worth memorising; trust your gut on this.",
        f"Most experts agree {topic} is mainly a theoretical curiosity.",
    ]
    idx = rng.randrange(len(fillers))
    rejected = rejected.rstrip() + " " + fillers[idx]

    return _clamp_length(
        rejected,
        COMPLETION_MIN,
        COMPLETION_MAX,
        pad_hint=f"That's been my experience with {topic} and I stand by it.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_preference_pair(
    chunk: Dict[str, Any],
    seed: int,
    provider: str = "mock",
    misconception_index: int = 0,
    *,
    paraphrase_provider: Optional[Any] = None,
    preserve_tokens: Optional[List[str]] = None,
    capture: Optional[Any] = None,
) -> PreferenceSynthesisResult:
    """Synthesize one preference pair from an enriched chunk.

    Args:
        chunk: Enriched chunk dict. Must have non-empty ``learning_outcome_refs``.
        seed: Deterministic seed.
        provider: ``"mock"`` (deterministic), ``"anthropic"`` (accepted for
            back-compat but never paraphrased — see below),
            ``"claude_session"`` (paraphrase via the running Claude Code
            session), ``"together"``, or ``"local"``.
        misconception_index: Which misconception in the chunk to target.
            If the chunk has fewer than ``misconception_index+1`` misconceptions,
            falls back to rule-synthesized rejection.
        paraphrase_provider: Optional provider instance with a
            ``paraphrase_preference(draft, chunk) -> dict`` method. Used
            when ``provider`` is ``"claude_session"``, ``"together"``, or
            ``"local"``. For ``"together"`` / ``"local"`` a default
            instance is constructed when this is None; for
            ``"claude_session"`` the caller MUST supply the instance,
            because the provider needs an injected LocalDispatcher.
            ``"anthropic"`` never reaches the paraphrase step.

    Returns:
        PreferenceSynthesisResult. ``pair`` is None if a hard gate failed.
    """
    if provider not in ("mock", "anthropic", "claude_session", "together", "local"):
        raise NotImplementedError(
            f"preference synthesis provider '{provider}' is not implemented; "
            f"valid choices are 'mock', 'anthropic', 'claude_session', 'together', 'local'."
        )

    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    lo_refs = list(chunk.get("learning_outcome_refs") or [])
    if not chunk_id or not lo_refs:
        return PreferenceSynthesisResult(
            pair=None,
            quality={"passed": False, "reason": "missing_chunk_id_or_lo_refs"},
            rationale="Chunk is missing id or learning_outcome_refs; no pair produced.",
            source="none",
        )

    rng = _seed_rng(chunk_id, seed)
    topic = _derive_topic(chunk)
    if not topic:
        # No concept_tags and no key_terms -> no human-readable topic. This
        # is a deterministic pre-dispatch exclusion, not a quality judgment
        # on generated output; the LO-id template fill it replaces was never
        # usable training data.
        return PreferenceSynthesisResult(
            pair=None,
            quality={
                "passed": False,
                "ineligible": True,
                "reason": "no_derivable_topic",
                "content_sources": {
                    "concept_tags": len(chunk.get("concept_tags") or []),
                    "key_terms": len(chunk.get("key_terms") or []),
                },
            },
            rationale=(
                f"Chunk {chunk_id} has no concept_tags and no key_terms, so no "
                f"human-readable topic can be derived (lo_refs={lo_refs} are "
                "database keys, never a topic). Unit marked ineligible before "
                "provider dispatch."
            ),
            source="none",
        )
    misconceptions = chunk.get("misconceptions") or []
    normalised_mcs = [
        m for m in misconceptions
        if isinstance(m, dict) and str(m.get("misconception", "")).strip()
    ]

    # Choose prompt variant deterministically.
    prompt_template = _PROMPT_TEMPLATES["misconception"] if normalised_mcs else _PROMPT_TEMPLATES["explanation"]
    prompt = prompt_template.format(topic=topic)
    if len(prompt) < PROMPT_MIN:
        prompt = prompt + f" Keep your answer concise and aimed at a learner new to {topic}."
    if len(prompt) > PROMPT_MAX:
        prompt = prompt[: PROMPT_MAX - 3].rstrip() + "..."

    chunk_text = str(chunk.get("text") or "")
    if _contains_verbatim_span(prompt, chunk_text):
        # Rewrite topic generically to guarantee no leakage.
        prompt = prompt_template.format(topic=f"the concept in chunk {chunk_id}")

    source: str = "rule_synthesized"
    mc_id: Optional[str] = None
    rejected: str = ""
    selected_mc: Optional[Dict[str, Any]] = None

    if normalised_mcs:
        idx = max(0, min(misconception_index, len(normalised_mcs) - 1))
        selected_mc = normalised_mcs[idx]

    chosen = _build_chosen(chunk, topic, selected_mc)
    # Completion-side leakage retry (mirrors instruction_factory).
    # _build_chosen leans on chunk.summary, which is often a near-verbatim
    # extract; on a leak, retry skipping the summary branch.
    if _contains_verbatim_span(chosen, chunk_text):
        chosen = _build_chosen(chunk, topic, selected_mc, disallow_summary=True)
    # Assessment-scaffolding contamination retry: a chunk whose summary is an
    # assessment outline ("Question 1 (CO-07, Bloom: ...)") would otherwise
    # carry that scaffolding into both chosen and rejected.
    if _contains_assessment_scaffolding(chosen):
        chosen = _build_chosen(chunk, topic, selected_mc, disallow_summary=True)

    if selected_mc:
        rejected_candidate = _build_rejected_from_misconception(selected_mc, topic)
        if rejected_candidate and rejected_candidate != chosen:
            rejected = rejected_candidate
            source = "misconception"
            mc_id = _misconception_id(
                str(selected_mc.get("misconception", "")),
                str(selected_mc.get("correction", "")),
                # bloom_level participates in the seed (lower-cased by the
                # helper). Absent / None on corpora predating bloom levels,
                # which the helper's two-form seed handles.
                str(selected_mc.get("bloom_level") or ""),
            )

    if not rejected or rejected == chosen:
        rejected = _rule_synthesize_rejected(chosen, topic, rng)
        source = "rule_synthesized"
        mc_id = None

    # Measure gates.
    jaccard = _jaccard(chosen, rejected)
    # Jaccard delta interpretation: gate says chosen and rejected must differ.
    # We require 1 - jaccard >= 0.3  ==>  jaccard <= 0.7.
    jaccard_ok = (1.0 - jaccard) >= JACCARD_DELTA_MIN
    distinct_ok = chosen != rejected
    # The leak gate covers prompt and chosen against chunk_text; rejected is
    # synthetic / misconception-derived, so verbatim leak isn't meaningful
    # there. The scaffolding gate DOES cover rejected, because
    # _rule_synthesize_rejected derives from chosen via token swaps and the
    # question pattern survives negation, so scaffolding propagates into it.
    #
    # The chosen side is compared against the chunk MINUS this pair's own
    # authored correction. On the misconception branch `_build_chosen`
    # RETURNS that correction — it is the ground truth the pair exists to
    # teach the model to prefer — and the correction is by construction a
    # span of the chunk it was authored into, so an unqualified comparison
    # rejects every misconception-derived pair and the designed DPO path
    # emits nothing. Excluding one span is narrower than relaxing the check:
    # `chosen` is still leak-checked against the whole of the REST of the
    # chunk, and the prompt side is untouched.
    #
    # This is the only verbatim guard on the preference path —
    # `lib/validators/synthesis_leakage.py` audits `prompt` / `completion`
    # (instruction rows) and never reads `chosen` / `rejected` — so it is
    # deliberately narrowed here rather than deferred to a later gate.
    # Both authored sides are excluded, not just the correction: the rejected
    # side of a misconception pair is the authored CLAIM, which is likewise a
    # span of the chunk it was authored into.
    chosen_leak_source = chunk_text
    if selected_mc:
        for field in ("correction", "misconception", "statement"):
            authored = str(selected_mc.get(field) or "").strip()
            if authored:
                chosen_leak_source = chosen_leak_source.replace(authored, " ")
    leak_ok = (
        not _contains_verbatim_span(prompt, chunk_text)
        and not _contains_verbatim_span(chosen, chosen_leak_source)
    )
    assessment_ok = (
        not _contains_assessment_scaffolding(chosen)
        and not _contains_assessment_scaffolding(rejected)
    )
    prompt_ok = PROMPT_MIN <= len(prompt) <= PROMPT_MAX
    chosen_ok = COMPLETION_MIN <= len(chosen) <= COMPLETION_MAX
    rejected_ok = COMPLETION_MIN <= len(rejected) <= COMPLETION_MAX

    quality = {
        "prompt_len": len(prompt),
        "chosen_len": len(chosen),
        "rejected_len": len(rejected),
        "jaccard_similarity": round(jaccard, 4),
        "jaccard_delta": round(1.0 - jaccard, 4),
        "jaccard_delta_ok": jaccard_ok,
        "chosen_ne_rejected": distinct_ok,
        "no_verbatim_leakage": leak_ok,
        "no_assessment_scaffolding": assessment_ok,
        "prompt_len_ok": prompt_ok,
        "chosen_len_ok": chosen_ok,
        "rejected_len_ok": rejected_ok,
    }
    quality["passed"] = all([
        jaccard_ok, distinct_ok, leak_ok, assessment_ok,
        prompt_ok, chosen_ok, rejected_ok,
    ])

    rationale = (
        f"Preference pair source='{source}'; chosen is grounded in key_terms/concept_tags for "
        f"topic='{topic}'; rejected is "
        + ("drawn from chunk.misconceptions" if source == "misconception" else "rule-synthesized via deterministic negation swaps")
        + f". Jaccard delta={quality['jaccard_delta']} (gate >= {JACCARD_DELTA_MIN})."
    )

    if not quality["passed"]:
        return PreferenceSynthesisResult(
            pair=None,
            quality=quality,
            rationale=(
                f"Preference pair gated out: jaccard_delta_ok={jaccard_ok}, "
                f"chosen_ne_rejected={distinct_ok}, no_verbatim_leakage={leak_ok}, "
                f"no_assessment_scaffolding={assessment_ok}, "
                f"prompt_len_ok={prompt_ok}, chosen_len_ok={chosen_ok}, rejected_len_ok={rejected_ok}."
            ),
            source=source,
            misconception_id=mc_id,
        )

    pair = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "misconception_id": mc_id,
        "chunk_id": chunk_id,
        "lo_refs": lo_refs,
        "seed": int(seed),
        "decision_capture_id": "",
        "source": source,
        "rejected_source": source,
        "template_id": (
            "preference_misconception"
            if normalised_mcs
            else "preference_explanation"
        ),
        "content_type": str(
            chunk.get("content_type_label")
            or chunk.get("chunk_type")
            or "unknown"
        ),
        "bloom_level": str(chunk.get("bloom_level") or "unknown").lower(),
        "provider": provider,
        "schema_version": "v1",
    }

    # Paraphrase the deterministic draft via the selected LLM seat. Same
    # fail-loud contract as the instruction factory: missing key raises,
    # malformed JSON retries. ``anthropic`` is deliberately NOT in the
    # dispatch set — the Anthropic-SDK training path is gone (run_synthesis
    # fails closed on it before reaching here), so a
    # ``provider="anthropic"`` call falls through unparaphrased, mirroring
    # how ``mock`` skips the paraphrase block.
    if provider in ("claude_session", "together", "local"):
        provider_instance = paraphrase_provider
        if provider_instance is None:
            if provider == "together":
                # Default ON: route the hosted OpenAI-wire seat through the
                # registry-driven builder, which pins the leaf-exact knobs
                # (verbose prompts, preserve disabled, hard 60s timeout).
                # The per-vendor leaf below is the rollback path.
                from Trainforge.generators.providers._synthesis_provider import (
                    agnostic_synthesis_enabled,
                )
                if agnostic_synthesis_enabled():
                    from Trainforge.generators.providers._synthesis_provider import (
                        build_synthesis_provider,
                    )
                    provider_instance = build_synthesis_provider("together")
                else:
                    from Trainforge.generators.providers._together_provider import (
                        TogetherSynthesisProvider,
                    )
                    provider_instance = TogetherSynthesisProvider()
            elif provider == "local":
                # Default ON: route the local OpenAI-wire seat through the
                # registry-driven builder, which pins the leaf-exact knobs
                # (terse prompts, preserve enabled, hard 60s timeout). The
                # per-vendor leaf below is the rollback path.
                from Trainforge.generators.providers._synthesis_provider import (
                    agnostic_synthesis_enabled,
                )
                if agnostic_synthesis_enabled():
                    from Trainforge.generators.providers._synthesis_provider import (
                        build_synthesis_provider,
                    )
                    provider_instance = build_synthesis_provider("local")
                else:
                    from Trainforge.generators.providers._local_provider import (
                        LocalSynthesisProvider,
                    )
                    provider_instance = LocalSynthesisProvider()
            else:
                raise RuntimeError(
                    "provider='claude_session' requires paraphrase_provider "
                    "to be supplied; no lazy fallback because the provider "
                    "needs a LocalDispatcher injected by the caller."
                )
        # Same preserve-and-fallback contract as the instruction factory.
        # Preference pairs check ``chosen`` only — the rule-synthesized
        # rejection legitimately may not contain the literal CURIE.
        deterministic_draft = dict(pair)
        try:
            try:
                pair = provider_instance.paraphrase_preference(
                    pair, chunk, preserve_tokens=preserve_tokens or [],
                )
            except TypeError:
                pair = provider_instance.paraphrase_preference(pair, chunk)
        except Exception as exc:
            code = getattr(exc, "code", None)
            if preserve_tokens and code in (
                "surface_form_preservation_failed",
                "paraphrase_invalid_after_retry",
            ):
                pair = deterministic_draft
                pair["paraphrase_fallback_reason"] = code
            elif code == "provider_output_verbatim_leakage":
                return PreferenceSynthesisResult(
                    pair=None,
                    quality={
                        **quality,
                        "passed": False,
                        "no_verbatim_leakage": False,
                        "reason": code,
                    },
                    rationale=(
                        "Provider exhausted two bounded source-free leakage "
                        "rewrites; the candidate was rejected without a "
                        "deterministic fallback."
                    ),
                    source=source,
                    misconception_id=mc_id,
                )
            elif (
                isinstance(code, str)
                and code.startswith("staged_")
                and code.endswith("_invalid")
                and bool(
                    getattr(exc, "details", {}).get(
                        "terminal_content_rejection"
                    )
                )
            ):
                evidence = dict(getattr(exc, "details", {}) or {})
                if capture is not None:
                    capture.log_decision(
                        decision_type="preference_pair_generation",
                        decision=(
                            f"Rejected staged DPO candidate for chunk "
                            f"{chunk_id}: {code}"
                        ),
                        rationale=(
                            f"chunk_id={chunk_id}, stage="
                            f"{evidence.get('stage', 'unknown')}, code={code}, "
                            f"validation_error="
                            f"{evidence.get('validation_error', 'unknown')}, "
                            f"prompt_ref={evidence.get('prompt_ref', 'missing')}, "
                            f"response_ref="
                            f"{evidence.get('response_ref', 'missing')}; "
                            "bounded validator-specific repairs were exhausted "
                            "without substituting deterministic content."
                        ),
                        context=json.dumps(
                            {"code": code, **evidence},
                            sort_keys=True,
                        ),
                        task_id=f"{chunk_id}:preference:terminal-rejection",
                    )
                return PreferenceSynthesisResult(
                    pair=None,
                    quality={
                        **quality,
                        "passed": False,
                        "reason": code,
                        "rejection_evidence": evidence,
                    },
                    rationale=(
                        f"Staged provider exhausted bounded repairs at "
                        f"{evidence.get('stage', 'unknown')}; candidate was "
                        "terminally rejected without fallback."
                    ),
                    source=source,
                    misconception_id=mc_id,
                )
            else:
                raise

    # Force-inject preserve_tokens missing from prompt + chosen, dispatching
    # per-token on the FORM_DATA ``anchored_status`` discriminator:
    # ``"complete"`` -> embed an actual definition sentence; degraded /
    # non-manifest -> fall back to token-stuffing AND emit a
    # ``form_data_degraded_placeholder_skipped`` decision-capture event.
    if preserve_tokens:
        pair = _enforce_preserve_tokens_in_preference(
            pair, preserve_tokens, capture=capture,
        )

    # Same narrowing as the draft-stage gate above, for the same reason and
    # on the same one span: the chosen/rejected sides of a misconception pair
    # ARE the authored correction and claim, so comparing them against a
    # chunk that still contains those sentences rejects the pair for
    # containing its own ground truth. Both sides keep their full check
    # against the rest of the chunk; the prompt side is unchanged.
    final_leak = (
        _contains_verbatim_span(str(pair.get("prompt") or ""), chunk_text)
        or _contains_verbatim_span(
            str(pair.get("chosen") or ""), chosen_leak_source,
        )
        or _contains_verbatim_span(
            str(pair.get("rejected") or ""), chosen_leak_source,
        )
    )
    if final_leak:
        return PreferenceSynthesisResult(
            pair=None,
            quality={
                **quality,
                "passed": False,
                "no_verbatim_leakage": False,
                "reason": "provider_output_verbatim_leakage",
            },
            rationale=(
                "Final provider output contained a 50-character source span "
                "after paraphrase/preservation postprocessing; pair dropped."
            ),
            source=source,
            misconception_id=mc_id,
        )

    return PreferenceSynthesisResult(
        pair=pair,
        quality=quality,
        rationale=rationale,
        source=source,
        misconception_id=mc_id,
        alternatives=[
            {
                "option": "paraphrase-only rejection",
                "reason_rejected": "insufficient token turnover for DPO signal",
            },
            {
                "option": "prompt-swap rejection",
                "reason_rejected": "DPO requires shared prompt across chosen/rejected",
            },
        ],
    )


__all__ = [
    "synthesize_preference_pair",
    "PreferenceSynthesisResult",
    "JACCARD_DELTA_MIN",
    "MAX_VERBATIM_SPAN",
    "PROMPT_MIN",
    "PROMPT_MAX",
    "COMPLETION_MIN",
    "COMPLETION_MAX",
]
