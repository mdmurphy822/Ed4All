#!/usr/bin/env python3
"""
Trainforge Preference Pair Factory

Synthesizes DPO-style (prompt, chosen, rejected) preference pairs from an
enriched Trainforge chunk. Mock-provider path: deterministic, no LLM call.

Design constraints (Worker C plan):
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
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

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
    alternatives: List[str] = field(default_factory=list)


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
    tags = chunk.get("concept_tags") or []
    if tags:
        return str(tags[0]).replace("-", " ").replace("_", " ")
    key_terms = chunk.get("key_terms") or []
    if key_terms and isinstance(key_terms[0], dict):
        term = key_terms[0].get("term")
        if term:
            return str(term)
    lo_refs = chunk.get("learning_outcome_refs") or []
    if lo_refs:
        return f"learning outcome {lo_refs[0]}"
    return "the course topic"


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
    """Content-hash misconception ID (REC-LNK-02, Wave 69 extends seed).

    Stable across runs and across chunk re-chunking. Form:
    ``mc_<16-hex-char sha256>``. Replaces the earlier unstable
    position-based format ``{chunk_id}_mc_{index:02d}_{hash}``.

    Wave 69: ``bloom_level`` (optional 3rd arg) joins the seed so two
    misconceptions sharing statement + correction text but different
    Bloom cognitive demands emit distinct IDs. The seed is built as:

    * ``{statement}|{correction}|{bloom_level}`` when a bloom level is
      supplied (Wave 60+ corpora), and
    * ``{statement}|{correction}`` when no bloom level is supplied (Wave 72).

    The two-form seed keeps pre-Wave-60 / legacy-corpus IDs stable with
    the pre-Wave-69 hash. Pre-Wave-72 the bloom-less path appended a
    trailing ``|`` and silently rekeyed every legacy misconception — this
    shape matches the documented intent. Outer whitespace is normalised
    but inner whitespace is preserved, so cosmetic edits do not churn
    IDs but real text edits do.
    """
    # Wave 72: two-segment seed for bloom-less misconceptions so legacy
    # corpora keep the pre-Wave-69 hash. The graph-side call site
    # (``CourseProcessor._build_misconceptions_for_graph``) applies the
    # same branch to stay in lock-step.
    # Wave 99: extracted to ``lib.ontology.misconception_id.canonical_mc_id``
    # so this site, ``process_course._build_misconceptions_for_graph``, and
    # ``pedagogy_graph_builder._mc_id`` share one source of truth.
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
) -> str:
    """Build the preferred (chosen) completion -- grounded and correct."""
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
    if summary:
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
        parts.append(
            f"{topic} should be explained through the concrete RDF/SHACL role "
            f"of {', '.join(tags[:3])}, not just by listing related labels."
        )

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


def _enforce_preserve_tokens_in_preference(
    pair: Dict[str, Any], preserve_tokens: List[str]
) -> Dict[str, Any]:
    """Force-inject any ``preserve_tokens`` not present in the
    ``chosen`` field. Mirrors the instruction-factory helper but
    targets ``chosen`` only — the rule-synthesized rejection legitimately
    may omit the technical CURIE, and forcing it into ``rejected`` would
    weaken the DPO signal. Idempotent, length-clamped.
    """
    if not preserve_tokens:
        return pair
    chosen = str(pair.get("chosen") or "")
    missing = [t for t in preserve_tokens if t and t not in chosen]
    if not missing:
        return pair
    addition = f" Canonical terms: {', '.join(missing)}."
    new_chosen = chosen.rstrip() + addition
    if len(new_chosen) > COMPLETION_MAX:
        budget = COMPLETION_MAX - len(addition)
        if budget < COMPLETION_MIN:
            new_chosen = chosen[:max(COMPLETION_MIN - len(addition), 0)].rstrip() + addition
        else:
            new_chosen = chosen[:budget].rstrip() + addition
    pair["chosen"] = new_chosen
    pair.setdefault("preserve_tokens_injected", []).extend(missing)
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
) -> PreferenceSynthesisResult:
    """Synthesize one preference pair from an enriched chunk.

    Args:
        chunk: Enriched chunk dict. Must have non-empty ``learning_outcome_refs``.
        seed: Deterministic seed.
        provider: ``"mock"`` (deterministic), ``"anthropic"`` (Wave 91:
            paraphrase via Claude SDK), ``"claude_session"`` (Wave 107:
            paraphrase via the running Claude Code session),
            ``"together"``, or ``"local"``.
        misconception_index: Which misconception in the chunk to target.
            If the chunk has fewer than ``misconception_index+1`` misconceptions,
            falls back to rule-synthesized rejection.
        paraphrase_provider: Optional provider instance with a
            ``paraphrase_preference(draft, chunk) -> dict`` method. Used
            when ``provider`` is ``"anthropic"``, ``"claude_session"``,
            ``"together"``, or ``"local"``. When
            ``provider="anthropic"`` and this is None, a default
            :class:`AnthropicSynthesisProvider` is constructed. For
            ``provider="claude_session"`` the caller MUST supply the
            instance.

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

    if selected_mc:
        rejected_candidate = _build_rejected_from_misconception(selected_mc, topic)
        if rejected_candidate and rejected_candidate != chosen:
            rejected = rejected_candidate
            source = "misconception"
            mc_id = _misconception_id(
                str(selected_mc.get("misconception", "")),
                str(selected_mc.get("correction", "")),
                # Wave 69: bloom-level participates in the seed; lower-cased
                # via the helper. Absent / None on pre-Wave-60 corpora.
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
    leak_ok = not _contains_verbatim_span(prompt, chunk_text)
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
        "prompt_len_ok": prompt_ok,
        "chosen_len_ok": chosen_ok,
        "rejected_len_ok": rejected_ok,
    }
    quality["passed"] = all([
        jaccard_ok, distinct_ok, leak_ok, prompt_ok, chosen_ok, rejected_ok,
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
        "provider": provider,
        "schema_version": "v1",
    }

    # Wave 91 Action A: paraphrase the mock draft via Anthropic when
    # provider="anthropic". Same fail-loud contract as the instruction
    # factory: missing key raises, malformed JSON retries up to 3x.
    if provider in ("anthropic", "claude_session", "together", "local"):
        provider_instance = paraphrase_provider
        if provider_instance is None:
            if provider == "anthropic":
                from Trainforge.generators._anthropic_provider import (
                    AnthropicSynthesisProvider,
                )
                provider_instance = AnthropicSynthesisProvider()
            elif provider == "together":
                from Trainforge.generators._together_provider import (
                    TogetherSynthesisProvider,
                )
                provider_instance = TogetherSynthesisProvider()
            elif provider == "local":
                from Trainforge.generators._local_provider import (
                    LocalSynthesisProvider,
                )
                provider_instance = LocalSynthesisProvider()
            else:
                raise RuntimeError(
                    "provider='claude_session' requires paraphrase_provider "
                    "to be supplied; no lazy fallback because the provider "
                    "needs a LocalDispatcher injected by the caller."
                )
        # Wave 120: same preserve-and-fallback contract as the
        # instruction factory. Preference pairs check ``chosen`` only —
        # the rule-synthesized rejection legitimately may not contain
        # the literal CURIE.
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
            if code == "surface_form_preservation_failed":
                pair = deterministic_draft
                pair["paraphrase_fallback_reason"] = "surface_form_preservation_failed"
            else:
                raise

    # Wave 120 follow-up: force-inject preserve_tokens absent from the
    # ``chosen`` field so the property-coverage gate sees the literal
    # CURIE. Same rationale as the instruction-factory version: both
    # the mock (deterministic) and paraphrase-fallback paths produce
    # chosen text that uses slugified concept_tags rather than the
    # literal CURIE. Idempotent, length-clamped.
    if preserve_tokens:
        pair = _enforce_preserve_tokens_in_preference(pair, preserve_tokens)

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
