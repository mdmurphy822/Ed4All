"""Per-chunk summary generator for dense-retrieval recall augmentation.

This module produces a 2–3 sentence summary for every chunk. The summary
is designed to improve recall when used alongside (or instead of) the
chunk's raw text during retrieval. See the companion benchmark in
``Trainforge/rag/retrieval_benchmark.py`` for recall@k measurements.

Design
------
- **Deterministic extractive path (default).** A pure function over the
  chunk text. Picks the opening sentence (topic), and, when one exists,
  a sentence that bears an LO-tag token or overlaps with the chunk's
  key terms (signal). Assembles 2–3 sentences clamped to 40–400 chars.
- **Optional LLM path.** Guarded behind the ``mode="llm"`` kwarg and a
  caller-supplied callable. The deterministic extractive path MUST work
  end-to-end without the LLM being available; LLM is purely opt-in.

Contract
--------
``generate(text, key_terms=None, learning_outcome_refs=None, mode="extractive", llm_fn=None)``
returns a string ``summary`` where:

- 40 <= len(summary) <= 400
- ``len(summary) <= len(text)``: summaries never exceed raw chunk length
- Deterministic: same inputs produce the same output
- Never raises on degenerate input (empty text returns a short marker
  that still satisfies the length band by padding with a period run —
  tests don't exercise this path, but callers stay crash-free).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, List, Optional, Sequence

__all__ = ["generate", "SUMMARY_MIN_LEN", "SUMMARY_MAX_LEN"]


SUMMARY_MIN_LEN = 40
SUMMARY_MAX_LEN = 400

# Reuse the same sentence boundary regex as CourseProcessor._split_by_sentences
# for behavioral consistency across the pipeline.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

# Tokens used for LO-tag heuristic. These patterns are intentionally loose:
# any bare LO-ish id (e.g., "co-04", "LO-02", "to-01", "w02-co-02") should
# be discoverable as a signal marker anywhere in the text.
_LO_TOKEN_RE = re.compile(r"\b(?:[a-z]{1,3}-\d{1,3}|LO-?\d{1,3}|w\d{2}-[a-z]{1,3}-\d{1,3})\b", re.IGNORECASE)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using the pipeline's canonical regex.

    Empty strings and whitespace-only fragments are dropped.
    """
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _normalise_terms(key_terms: Optional[Sequence[Any]]) -> List[str]:
    """Flatten key_terms (mixed str / {term, definition} dicts) to a
    lowercase list of term strings. Safe for None.
    """
    if not key_terms:
        return []
    out: List[str] = []
    for kt in key_terms:
        if isinstance(kt, dict):
            term = kt.get("term")
            if term:
                out.append(str(term).lower().strip())
        elif isinstance(kt, str):
            out.append(kt.lower().strip())
    return [t for t in out if t]


def _normalise_los(learning_outcome_refs: Optional[Iterable[str]]) -> List[str]:
    if not learning_outcome_refs:
        return []
    return [str(x).lower().strip() for x in learning_outcome_refs if x]


def _score_sentence(
    sentence: str,
    idx: int,
    key_terms: Sequence[str],
    los: Sequence[str],
) -> float:
    """Higher score => better summary candidate.

    Heuristic:
      - Opening sentence gets a topic bonus (+2.0) so it's almost always chosen.
      - A sentence containing an LO id literally (e.g., "co-01") gets +3.0
        per distinct LO token seen — that's our "LO-tag-bearing sentence".
      - Each key-term substring match adds +1.0.
      - Very short (< 3 words) or very long (> 60 words) sentences are
        penalised (-0.5) to keep summaries readable.
    """
    lower = sentence.lower()
    score = 0.0

    if idx == 0:
        score += 2.0

    lo_hits = 0
    for lo in los:
        # Match both the literal LO id and any LO-shaped token
        if lo and lo in lower:
            lo_hits += 1
    # Also count generic LO-shaped tokens (e.g., uppercase "LO-02", "co-04")
    lo_hits += len(_LO_TOKEN_RE.findall(sentence))
    score += 3.0 * min(lo_hits, 3)  # cap so one sentence can't dominate

    for term in key_terms:
        if term and term in lower:
            score += 1.0

    wc = len(sentence.split())
    if wc < 3 or wc > 60:
        score -= 0.5

    return score


def _clamp_length(summary: str, max_text_len: int) -> str:
    """Enforce 40 <= len(summary) <= min(400, len(text)).

    When the candidate exceeds the upper bound, truncate on a word boundary
    and append ``...`` so readers see the elision; the truncated form still
    respects the upper bound.

    When the candidate is shorter than 40 chars, pad deterministically with
    a trailing period run. This path only triggers on near-empty chunks and
    is defensive; it keeps the function total.
    """
    cap = min(SUMMARY_MAX_LEN, max_text_len) if max_text_len > 0 else SUMMARY_MAX_LEN

    if len(summary) > cap:
        # Hard cap with a word-boundary-aware trim.
        trimmed = summary[: cap - 3].rstrip()
        # Back off to the last whitespace so we don't cut mid-word.
        space = trimmed.rfind(" ")
        if space > cap // 2:
            trimmed = trimmed[:space]
        summary = trimmed + "..."

    if len(summary) < SUMMARY_MIN_LEN:
        # Deterministic pad. Use dots so callers can detect padding if they care.
        pad = SUMMARY_MIN_LEN - len(summary)
        summary = summary + ("." * pad)

    return summary


def _extractive_summary(
    text: str,
    key_terms: Sequence[str],
    los: Sequence[str],
) -> str:
    """Deterministic 2–3 sentence summary.

    Strategy:
      1. Split into sentences.
      2. If the text is very short (<= 2 sentences), return it clamped.
      3. Otherwise pick the top-2 sentences by score (always biased toward
         the opener), preserving source order in the output.
      4. If adding a third sentence keeps us within the 400-char bound and
         extends coverage (new key terms or LOs), include it.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    if len(sentences) <= 2:
        return " ".join(sentences).strip()

    scored = [
        (_score_sentence(s, i, key_terms, los), i, s) for i, s in enumerate(sentences)
    ]
    # Stable sort: high score first; ties broken by source order so output
    # is deterministic under equal heuristic values.
    scored.sort(key=lambda t: (-t[0], t[1]))

    chosen_indices = sorted({scored[0][1], scored[1][1]})
    chosen = [sentences[i] for i in chosen_indices]

    # Try to add a 3rd sentence if we have budget AND it contributes new signal.
    current_text = " ".join(chosen)
    if len(current_text) < SUMMARY_MAX_LEN - 30:
        covered = current_text.lower()
        for score, i, s in scored[2:]:
            if i in chosen_indices:
                continue
            slower = s.lower()
            new_signal = any(t for t in key_terms if t and t in slower and t not in covered)
            new_lo = any(lo for lo in los if lo and lo in slower and lo not in covered)
            if new_signal or new_lo or score > 1.0:
                candidate = current_text + " " + s
                if len(candidate) <= SUMMARY_MAX_LEN:
                    chosen_indices = sorted(set(chosen_indices) | {i})
                    chosen = [sentences[j] for j in chosen_indices]
                break

    return " ".join(chosen).strip()


def generate(
    text: str,
    key_terms: Optional[Sequence[Any]] = None,
    learning_outcome_refs: Optional[Iterable[str]] = None,
    mode: str = "extractive",
    llm_fn: Optional[Callable[[str, Sequence[str], Sequence[str]], str]] = None,
) -> str:
    """Produce a 2–3 sentence summary for a chunk.

    Args:
        text: The chunk's raw ``text`` field.
        key_terms: Optional chunk ``key_terms`` list (dicts with ``term`` key
            or bare strings).
        learning_outcome_refs: Optional list of LO IDs the chunk covers.
        mode: ``"extractive"`` (default, deterministic) or ``"llm"``. The
            LLM path is purely opt-in; if ``mode="llm"`` but ``llm_fn`` is
            ``None`` the function falls back to extractive.
        llm_fn: Callable ``(text, key_terms, los) -> str`` invoked when
            ``mode="llm"``. The callable's output is length-clamped before
            return.

    Returns:
        A summary string of 40–400 chars, never longer than the input text
        (or padded to 40 chars in the rare near-empty case).
    """
    if not text:
        # Defensive: keep the function total. Pad a placeholder so length
        # invariants hold if a caller inexplicably feeds an empty chunk.
        return _clamp_length("No content available for this chunk.", 200)

    key_term_strs = _normalise_terms(key_terms)
    lo_strs = _normalise_los(learning_outcome_refs)

    if mode == "llm" and llm_fn is not None:
        try:
            summary = llm_fn(text, key_term_strs, lo_strs)
        except Exception:
            summary = _extractive_summary(text, key_term_strs, lo_strs)
        if not summary:
            summary = _extractive_summary(text, key_term_strs, lo_strs)
    else:
        summary = _extractive_summary(text, key_term_strs, lo_strs)

    return _clamp_length(summary, len(text))
