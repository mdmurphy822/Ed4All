"""Deterministic post-hoc CO citation RE-SELECTION (opt-in).

The stage-2 7B window synthesis cites ``source_chunk_ids`` per candidate
objective, but a real full-book audit measured ~40% of COs citing a
same-chapter NEIGHBOR chunk instead of the best supporter (7B citation
sloppiness), most COs citing exactly 1 chunk, and only a small fraction of
chunks cited at all. The existing Fix 1A machinery
(:func:`lib.objectives.objective_dedup.dedup_candidates`'s relevance-floor +
top-K prune) only prunes/ranks what the model CITED — it cannot re-select a
better chunk the model SAW but failed to cite.

This pass fixes that deterministically (zero LLM calls): for each canonical CO
it builds a candidate pool = (the CO's synthesis window chunks) ∪ (its CHAPTER
bucket, derived from its cited chunks' ``chapter_id``) ∪ (currently cited ids),
deduped in stable order (window first, then chapter, then cited) — ranks the
pool by cosine(statement embedding, chunk text embedding), and re-cites the
top-K above the relevance floor. The caller (the stage-2 hook) assembles the
widened window ∪ chapter pool and passes it via ``window_chunk_ids_by_co``;
this module then unions the currently-cited ids and applies the resolve filter.
Widening past the single resolved window closes the hole where the Fix 1A prune
strips a dedup-merged CO's right-window citation, locking pool resolution to the
WRONG window so re-selection could never reach the true supporter. The
relevance floor (``resolve_chunk_relevance_floor``, default 0.30) is the junk
guard for the wider pool.

Contracts:

- **Anti-fabrication:** the pool is strictly (window chunks the model saw) ∪
  (same-chapter real chunks) ∪ (already-cited); every id must resolve in
  ``chunks_by_id`` (unresolvable ids are dropped and counted) — no id is ever
  invented.
- **ALWAYS-KEEP-original:** the CO's original citation(s) are never STRIPPED
  by re-selection — this pass only ever IMPROVES provenance, it never removes a
  supporter the model already cited. Two arms enforce this: (1) if nothing in
  the pool clears the floor the original citation(s) are kept verbatim; (2) the
  ``keep_original`` guard (default **ON**) UNIONS every original citation that
  itself clears the floor and is not exercise-like into the kept set (so the
  cosine top-K only ADDS better supporters, it can never DROP a synthesis
  citation). Exercise-like / below-floor originals stay droppable (the pass's
  neighbor-fix + exercise-demote purpose). See ``keep_original`` below.
- **Zero-citation COs are skipped** (re-SELECTION, not initial citation — the
  chapter_fallback grounding mode emits citation-less COs by design).
- **Graceful degrade:** embed client absent → logged no-op.

Floor / cap reuse the Fix 1A knobs (`ED4ALL_OBJECTIVE_CHUNK_RELEVANCE_FLOOR` /
`ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE`). Gate:
``ED4ALL_OBJECTIVE_CITATION_RESELECT`` (default OFF, parse-with-fallback;
active only on the ``TEXTBOOK_SYNTHESIS_PROVIDER`` path, like its siblings).
Selects no provider/model → no ``docs/LICENSING.md`` row. Decision capture
reuses the existing ``objective_chunk_prune`` event (no new decision_type).

**Exercise-chunk demotion (ranking-quality bug fix).** A full-book audit
(course ``sample-full-obj-01``) found end-of-chapter EXERCISE chunks winning the
pure-cosine rank because their instruction line nearly quotes the CO statement
(e.g. a "In the following exercises, find the place value ... 1. 51,493 ⓐ …"
answer-list chunk out-cosining a skill-phrased CO). The re-selection then cites
an answer list instead of instructional prose, and downstream page-per-CO
grounding degenerates to objective echoes. When a chunk carries the Wave #22 pedagogical metadata
(``composite_unit`` / ``unit_roles``, harvested from the SemantiK
data-dart-unit / -flow / -opener attributes), :func:`_is_exercise_like_chunk`
uses it as the PRIMARY signal (``composite_unit == "exercise_set"`` or a
practice-dominant role → demote-class; ``worked_example`` / ``statement`` →
instructional-class). Chunks without that metadata (legacy / non-SemantiK)
fall back to the conservative,
high-precision TEXT heuristic :func:`_is_exercise_like` (leading "In the
following exercises", ≥3 circled
ⓐ-ⓔ answer glyphs, a dense run of ≥3 ``N.`` numbered answer markers, or a
"Practice Makes Perfect" / "Section Exercises" / "Review Exercises" banner —
the circled-glyph + banner signals mirror ``lib/chunk_heading_sanity.py``). The
ranking key becomes ``(exercise_like ASC, cosine DESC, pool order)`` so every
non-exercise chunk outranks every exercise-like chunk regardless of cosine;
within each group cosine order is unchanged. Demotion NEVER excludes: the floor
and top-K still apply to everyone, so an all-exercise above-floor pool still
cites (no starvation). Gated by ``ED4ALL_OBJECTIVE_RESELECT_EXERCISE_DEMOTE``,
default **ON** whenever re-selection itself is on — this is a bug-fix of rank
quality inside an already opt-in feature, so default-on is acceptable; opt out
with ``0`` / ``false`` / ``no`` / ``off`` to restore the pure-cosine rank.

**Keep-original guard (entailment-regression bug fix).** A full-book audit
(course ``sample-full-obj-01``) found the pure top-K REPLACE could STRIP the one
chunk that entails the CO statement. A chapter-level CO ("Solve linear
inequalities using the Subtraction and Addition Properties of Inequality")
cited the section chunk that literally states both *Inequality* properties, but
a FOREIGN-window chunk about solving *equations* (lexically similar via
"solve / simplify / properties") out-cosined it and, under REPLACE, evicted the
true supporter from the top-K — collapsing the per-LO NLI entailment gate to
0%. Because that gate scores the statement against the UNION of the cited
chunks' text (one hypothesis, windowed rescue), RETAINING the supporter — even
alongside the off-topic chunk — restores entailment. The guard
(``ED4ALL_OBJECTIVE_RESELECT_KEEP_ORIGINAL``, default **ON** whenever
re-selection is on) UNIONS every original citation that clears the floor and is
not exercise-like into the kept set: the cosine top-K may only ADD supporters,
never DROP a synthesis citation. The cap is widened to
``max(cap, n_protected_originals)`` so protected originals are never squeezed
out; exercise-like / below-floor originals remain droppable (so the
neighbor-fix + exercise-demote behavior is unchanged). Opt out with ``0`` /
``false`` / ``no`` / ``off`` to restore the pure-cosine REPLACE.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lib.embedding._math import cosine_similarity
# Exercise-marker constants consolidated into the shared apparatus lexicon
# (single source of truth). Imported here as byte-identical re-exports — no
# behavior change. ``_EXERCISE_BANNER_RE`` was formerly imported from
# ``lib.chunk_heading_sanity`` (which now itself re-exports the same object).
from lib.objectives.apparatus_lexicon import (
    EXERCISE_BANNER_RE as _EXERCISE_BANNER_RE,
    EXTRA_EXERCISE_BANNER_RE as _EXTRA_EXERCISE_BANNER_RE,
    FOLLOWING_EXERCISES_RE as _FOLLOWING_EXERCISES_RE,
)
from lib.objectives.objective_dedup import (
    resolve_chunk_relevance_floor,
    resolve_max_chunks_per_objective,
)

logger = logging.getLogger(__name__)

ENV_CITATION_RESELECT = "ED4ALL_OBJECTIVE_CITATION_RESELECT"
ENV_RESELECT_EXERCISE_DEMOTE = "ED4ALL_OBJECTIVE_RESELECT_EXERCISE_DEMOTE"
ENV_RESELECT_KEEP_ORIGINAL = "ED4ALL_OBJECTIVE_RESELECT_KEEP_ORIGINAL"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})

# ---- Exercise-likeness heuristic (conservative, high-precision) -----------
# Signal A — the OpenStax "In the following exercises" instruction line, near
# the START of the chunk (an exercise block leads with it). Positional so a
# passing mention deep inside prose does not trip it. ``_FOLLOWING_EXERCISES_RE``
# is imported from lib.objectives.apparatus_lexicon (byte-identical).
_EARLY_WINDOW_CHARS = 200

# Signal B — circled multiple-choice answer glyphs ⓐⓑⓒⓓⓔ (U+24D0..U+24D4);
# mirrors chunk_heading_sanity's circled-answer detection. Require >=3 so a
# stray glyph in prose is spared.
_CIRCLED_ANSWER_RE = re.compile("[ⓐ-ⓔ]")
_CIRCLED_ANSWER_MIN = 3

# Signal C — a DENSE run of >=3 "N." numbered answer-list markers ("1. 51,493
# 2. 3,491 3. 812"). Whitespace after the period spares decimals ("3.5") and
# section numbers ("1.1"); the negative-lookbehind spares thousands ("51,493").
# "Dense" = >=3 markers within a short character span, so scattered legit
# enumerations ("... step 1. ... much later ... 2. ...") do NOT trip.
_NUMBERED_RUN_RE = re.compile(r"(?<!\d)\d{1,3}\.\s")
_NUMBERED_RUN_MIN = 3
_NUMBERED_RUN_SPAN = 400

# Signal D — exercise-section banners. The shared ``_EXERCISE_BANNER_RE``
# already covers "EXERCISES Practice Makes Perfect" + "In the following
# exercises"; ``_EXTRA_EXERCISE_BANNER_RE`` extends it with the standalone
# "Practice Makes Perfect" / "Section Exercises" / "Review Exercises" headers.
# Both are imported from lib.objectives.apparatus_lexicon (byte-identical).


def _has_dense_numbered_run(text: str) -> bool:
    """Whether ``text`` carries >=3 ``N.`` markers within a short char span."""
    starts = [m.start() for m in _NUMBERED_RUN_RE.finditer(text)]
    if len(starts) < _NUMBERED_RUN_MIN:
        return False
    for i in range(len(starts) - _NUMBERED_RUN_MIN + 1):
        span = starts[i + _NUMBERED_RUN_MIN - 1] - starts[i]
        if span <= _NUMBERED_RUN_SPAN:
            return True
    return False


def _is_exercise_like(text: Optional[str]) -> bool:
    """Whether ``text`` is clearly an exercise / answer-list block (defensive).

    Conservative by design — returns ``True`` only on unambiguous exercise
    signals so instructional prose is NEVER demoted. Any one signal fires:

    A. "In the following exercises" near the start of the chunk.
    B. >=3 circled answer glyphs (ⓐ-ⓔ).
    C. a dense run of >=3 ``N.`` numbered answer-list markers.
    D. an exercise-section banner ("Practice Makes Perfect" / "Section
       Exercises" / "Review Exercises" / the shared ``_EXERCISE_BANNER_RE``).
    """
    if not text or not text.strip():
        return False
    stem = str(text)

    if _FOLLOWING_EXERCISES_RE.search(stem[:_EARLY_WINDOW_CHARS]):
        return True
    if len(_CIRCLED_ANSWER_RE.findall(stem)) >= _CIRCLED_ANSWER_MIN:
        return True
    if _has_dense_numbered_run(stem):
        return True
    if _EXERCISE_BANNER_RE.search(stem) or _EXTRA_EXERCISE_BANNER_RE.search(
        stem
    ):
        return True
    return False


# ---- Pedagogical-metadata primary signal (Wave #22 quick-wins) ------------
# When a chunk carries the additive ``composite_unit`` / ``unit_roles``
# metadata (harvested from the SemantiK data-dart-unit / -flow / -opener
# attributes by the parser + chunker), it is a FAR more reliable exercise/
# instructional signal than the text heuristic. It becomes the PRIMARY signal;
# the text heuristic (:func:`_is_exercise_like`) is the fallback for legacy /
# non-SemantiK chunks that carry no metadata (so behavior on those is
# byte-identical to before this wave).
#
#   * ``composite_unit == "exercise_set"`` OR a practice-dominant role
#     (``try_it`` / ``practice``) with no instructional signal -> demote-class.
#   * ``composite_unit == "worked_example"`` OR a ``worked_example`` /
#     ``statement`` role -> instructional-class (never demoted).
# Instructional signals take precedence: a worked example / statement is
# instructional prose even when it sits adjacent to practice material.
_PRACTICE_ROLES = frozenset({"try_it", "practice"})
_INSTRUCTIONAL_ROLES = frozenset({"worked_example", "statement"})


def _pedagogical_exercise_signal(chunk: Any) -> Optional[bool]:
    """Tri-state exercise signal from a chunk's pedagogical metadata.

    Returns ``True`` (exercise / demote-class), ``False`` (instructional /
    keep-class), or ``None`` (no pedagogical metadata → defer to the text
    heuristic). Pure read of the ``composite_unit`` / ``unit_roles``
    fields; a chunk missing both (legacy / non-SemantiK) yields ``None``.
    """
    if not isinstance(chunk, dict):
        return None
    unit = chunk.get("composite_unit")
    roles = chunk.get("unit_roles")
    role_set = {str(r) for r in roles} if isinstance(roles, list) else set()
    # Instructional wins over practice when both are present.
    if unit == "worked_example" or (role_set & _INSTRUCTIONAL_ROLES):
        return False
    if unit == "exercise_set":
        return True
    if role_set & _PRACTICE_ROLES:
        return True
    return None


def _is_exercise_like_chunk(chunk: Any) -> bool:
    """Exercise-likeness with the pedagogical metadata as the PRIMARY signal.

    Uses :func:`_pedagogical_exercise_signal` when the chunk carries the
    ``composite_unit`` / ``unit_roles`` metadata; falls back to the
    text heuristic :func:`_is_exercise_like` when it does not (behavior-
    preserving on legacy / non-SemantiK chunks).
    """
    signal = _pedagogical_exercise_signal(chunk)
    if signal is not None:
        return signal
    return _is_exercise_like(_chunk_text(chunk))


def resolve_reselect_exercise_demote(value: Optional[bool] = None) -> bool:
    """Resolve ``ED4ALL_OBJECTIVE_RESELECT_EXERCISE_DEMOTE`` (default **ON**).

    Explicit arg > env. Only the opt-out tokens ``0`` / ``false`` / ``no`` /
    ``off`` (any case) disable the exercise demotion; unset / anything else
    keeps the default-ON bug-fix (parse-with-fallback, inverse of the sibling
    default-OFF resolvers since this is a rank-quality fix, not a new feature).
    Consulted only when re-selection itself is enabled.
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(ENV_RESELECT_EXERCISE_DEMOTE)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSEY


def resolve_reselect_keep_original(value: Optional[bool] = None) -> bool:
    """Resolve ``ED4ALL_OBJECTIVE_RESELECT_KEEP_ORIGINAL`` (default **ON**).

    Explicit arg > env. Only the opt-out tokens ``0`` / ``false`` / ``no`` /
    ``off`` (any case) disable the keep-original union guard; unset / anything
    else keeps the default-ON bug-fix (parse-with-fallback, mirroring
    ``resolve_reselect_exercise_demote`` — this is a correctness fix inside an
    already opt-in feature). Consulted only when re-selection is enabled.
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(ENV_RESELECT_KEEP_ORIGINAL)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSEY


def resolve_citation_reselect(value: Optional[bool] = None) -> bool:
    """Resolve the ``ED4ALL_OBJECTIVE_CITATION_RESELECT`` gate (default OFF).

    Explicit arg > env. Truthy tokens ``1``/``true``/``yes``/``on`` (any
    case) enable; falsey / garbage / unset → off (parse-with-fallback,
    mirroring ``ED4ALL_OBJECTIVE_DISTINCT_SKILL_SPLIT``).
    """
    if value is not None:
        return bool(value)
    return (
        os.environ.get(ENV_CITATION_RESELECT, "").strip().lower() in _TRUTHY
    )


@dataclass
class ReselectResult:
    """Aggregate signals from one re-selection pass (COs mutated in place)."""

    available: bool = False           # False → no-op (disabled / no embed)
    reselected_count: int = 0         # COs whose cited set changed
    scanned_count: int = 0            # COs with ≥1 citation that were scored
    skipped_no_citation: int = 0      # zero-citation COs (skipped by design)
    citation_density_before: int = 0  # distinct cited chunk ids, before
    citation_density_after: int = 0   # distinct cited chunk ids, after
    pool_misses: int = 0              # pool ids not resolvable in chunks_by_id
    kept_original_below_floor: int = 0  # COs where nothing cleared the floor
    exercise_demoted_total: int = 0   # exercise-like chunks demoted in ranking
    kept_original_supporters: int = 0  # above-floor originals the guard retained
    per_co_changes: List[Dict[str, Any]] = field(default_factory=list)


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("body") or "")
    return ""


def _cosine(a: Any, b: Any) -> float:
    """np.ndarray-safe cosine via the shared ``lib.embedding._math`` helper.

    The real embed client returns numpy vectors, so truthiness (``not a``)
    and ``or []`` coercions raise "truth value of an array is ambiguous" —
    guard with explicit None/length checks instead.
    """
    if a is None or b is None:
        return 0.0
    try:
        if len(a) == 0 or len(b) == 0 or len(a) != len(b):
            return 0.0
    except TypeError:
        return 0.0
    return float(cosine_similarity(a, b))


def _distinct_cited(cos: List[Dict[str, Any]]) -> int:
    seen: set = set()
    for co in cos:
        for cid in co.get("source_chunk_ids") or []:
            cs = str(cid).strip()
            if cs:
                seen.add(cs)
    return len(seen)


def _rewrite_source_refs(
    co: Dict[str, Any], kept: List[str], chunks_by_id: Dict[str, Any]
) -> None:
    """Mirror the new citation set onto the on-disk ``source_refs`` shape.

    Candidates carry ``source_refs: [{ref: chapter_id, chunk_ids: [...]}]``
    alongside the flat ``source_chunk_ids`` (back-compat shape from
    ``_normalise_window_objectives_payload``). Derive the ref label(s) from the
    KEPT chunks' ``chapter_id`` in ``chunks_by_id`` — one ``{ref, chunk_ids}``
    entry per chapter, preserving kept order within each chapter and chapter
    first-seen order across entries. With the widened pool the kept set stays
    single-chapter in practice, but a multi-chapter keep must NOT smear one
    chapter label over foreign chunks. Fall back to the old first ref label on a
    single collapsed entry only when a kept chunk's ``chapter_id`` is
    unresolvable.
    """
    refs = co.get("source_refs")
    if not isinstance(refs, list):
        return
    old_label = ""
    for entry in refs:
        if isinstance(entry, dict) and str(entry.get("ref") or "").strip():
            old_label = str(entry.get("ref")).strip()
            break
    groups: Dict[str, List[str]] = {}
    order: List[str] = []
    for cid in kept:
        rec = chunks_by_id.get(cid)
        ch = (
            str(rec.get("chapter_id") or "").strip()
            if isinstance(rec, dict) else ""
        )
        if not ch:
            # Unresolvable chapter → defensively collapse to the old label.
            co["source_refs"] = [{"ref": old_label, "chunk_ids": list(kept)}]
            return
        if ch not in groups:
            groups[ch] = []
            order.append(ch)
        groups[ch].append(cid)
    if not groups:
        co["source_refs"] = [{"ref": old_label, "chunk_ids": list(kept)}]
        return
    co["source_refs"] = [
        {"ref": ch, "chunk_ids": list(groups[ch])} for ch in order
    ]


def _emit_reselect_capture(
    capture: Any,
    *,
    co_index: int,
    statement: str,
    old_ids: List[str],
    new_ids: List[str],
    old_best_cos: Optional[float],
    new_best_cos: Optional[float],
    pool_size: int,
    floor: float,
) -> None:
    """Best-effort ``objective_chunk_prune`` capture (reused decision_type)."""
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type="objective_chunk_prune",
            decision=(
                f"citation_reselect: CO[{co_index}] re-cited "
                f"{old_ids} -> {new_ids}"
            ),
            rationale=(
                f"Post-hoc citation re-selection for '{statement[:60]}': "
                f"ranked a pool of {pool_size} chunk(s) the model saw "
                f"(window ∪ chapter ∪ cited) by cosine to the CO "
                f"statement; old best cosine="
                f"{'n/a' if old_best_cos is None else f'{old_best_cos:.3f}'}, "
                f"new best cosine="
                f"{'n/a' if new_best_cos is None else f'{new_best_cos:.3f}'}, "
                f"floor={floor:.2f}. The 7B cited a weaker neighbor; the "
                f"re-selection cites the strongest real supporter(s) from "
                f"the same pool (anti-fabrication: pool ⊆ chunks seen)."
            ),
            alternatives_considered=[
                "keep the model's original citations (Fix 1A prune-only)",
            ],
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("citation_reselect capture failed (%s); continuing", exc)


def reselect_citations(
    cos: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    embed: Any,
    *,
    window_chunk_ids_by_co: Optional[Dict[int, List[str]]] = None,
    chapter_chunks_by_co: Optional[Dict[int, List[str]]] = None,
    floor: Optional[float] = None,
    max_chunks: Optional[int] = None,
    capture: Optional[Any] = None,
    enabled: Optional[bool] = None,
    exercise_demote: Optional[bool] = None,
    keep_original: Optional[bool] = None,
) -> ReselectResult:
    """Re-select each CO's cited chunks from the pool the model actually saw.

    Mutates the CO dicts IN PLACE (``source_chunk_ids`` + the mirrored
    ``source_refs``) and returns aggregate :class:`ReselectResult` signals.
    ``window_chunk_ids_by_co`` / ``chapter_chunks_by_co`` are keyed by the
    CO's INDEX in ``cos`` (CO ids are not minted yet at the hook point).

    No-op (``available=False``) when the gate is off or ``embed`` is None.
    """
    result = ReselectResult()
    if not resolve_citation_reselect(enabled):
        return result
    if embed is None:
        logger.info(
            "citation_reselect: embedding client unavailable — skipping the "
            "re-selection pass (deterministic no-op)."
        )
        return result

    resolved_floor = resolve_chunk_relevance_floor(floor)
    resolved_cap = resolve_max_chunks_per_objective(max_chunks)
    demote_exercises = resolve_reselect_exercise_demote(exercise_demote)
    keep_originals = resolve_reselect_keep_original(keep_original)
    window_map = window_chunk_ids_by_co or {}
    chapter_map = chapter_chunks_by_co or {}

    result.citation_density_before = _distinct_cited(cos)

    # ---- Build per-CO pools (anti-fabrication: ids must resolve). --------
    pools: Dict[int, List[str]] = {}
    for i, co in enumerate(cos):
        cited = [
            str(c).strip() for c in (co.get("source_chunk_ids") or [])
            if str(c).strip()
        ]
        if not cited:
            result.skipped_no_citation += 1
            continue
        raw_pool: List[str] = list(window_map.get(i) or [])
        if not raw_pool:
            raw_pool = list(chapter_map.get(i) or [])
        # Union with cited (dedup, stable order: pool first, then cited).
        seen: set = set()
        pool: List[str] = []
        for cid in [*raw_pool, *cited]:
            cs = str(cid).strip()
            if not cs or cs in seen:
                continue
            seen.add(cs)
            if cs not in chunks_by_id:
                result.pool_misses += 1
                continue
            pool.append(cs)
        if pool:
            pools[i] = pool

    if not pools:
        result.available = True
        result.citation_density_after = result.citation_density_before
        return result

    # ---- Batch-embed statements + unique pool chunk texts. ---------------
    unique_ids: List[str] = []
    _useen: set = set()
    for pool in pools.values():
        for cid in pool:
            if cid not in _useen:
                _useen.add(cid)
                unique_ids.append(cid)
    try:
        stmt_vecs = embed.encode_batch([
            str(cos[i].get("statement") or cos[i].get("text") or "") or " "
            for i in pools
        ])
        chunk_vecs = embed.encode_batch([
            _chunk_text(chunks_by_id.get(cid)) or " " for cid in unique_ids
        ])
    except Exception as exc:  # noqa: BLE001 — graceful degrade, mirror dedup
        logger.warning(
            "citation_reselect: encode_batch failed (%s) — skipping the "
            "re-selection pass.", exc,
        )
        return result
    vec_by_chunk = dict(zip(unique_ids, chunk_vecs))
    vec_by_co = dict(zip(pools.keys(), stmt_vecs))

    # ---- Rank + re-cite per CO. ------------------------------------------
    for i, pool in pools.items():
        co = cos[i]
        result.scanned_count += 1
        old_ids = [
            str(c).strip() for c in (co.get("source_chunk_ids") or [])
            if str(c).strip()
        ]
        svec = vec_by_co[i]
        scored = [
            (cid, _cosine(svec, vec_by_chunk.get(cid))) for cid in pool
        ]
        # Exercise-likeness per pool chunk. Wave #22 quick-wins: the chunk's
        # pedagogical metadata (composite_unit / unit_roles) is the
        # PRIMARY signal when present; the text heuristic is the fallback for
        # legacy / non-SemantiK chunks (behavior-preserving on those).
        ex_like = {
            cid: (
                demote_exercises
                and _is_exercise_like_chunk(chunks_by_id.get(cid))
            )
            for cid, _ in scored
        }
        # Stable sort: exercise-like LAST, then cosine desc, then pool order.
        # Every non-exercise chunk outranks every exercise-like chunk (demote,
        # never exclude — the floor + top-K below still gate everyone).
        order = sorted(
            range(len(scored)),
            key=lambda k: (
                1 if ex_like[scored[k][0]] else 0,
                -scored[k][1],
                k,
            ),
        )
        ranked = [scored[k] for k in order]
        above = [(cid, cs) for cid, cs in ranked if cs >= resolved_floor]
        # Count exercise-like chunks the demotion pushed BELOW a weaker
        # (lower-cosine) non-exercise chunk among the above-floor survivors —
        # i.e. the demotion actually reordered them. All-exercise pools => 0.
        demoted = 0
        if demote_exercises and above:
            above_ids = {cid for cid, _ in above}
            nonex_cos = [
                cs for cid, cs in scored
                if cid in above_ids and not ex_like[cid]
            ]
            if nonex_cos:
                min_nonex = min(nonex_cos)
                demoted = sum(
                    1 for cid, cs in scored
                    if cid in above_ids and ex_like[cid] and cs > min_nonex
                )
        result.exercise_demoted_total += demoted
        if not above:
            # ALWAYS-KEEP-original: nothing clears the floor → unchanged.
            result.kept_original_below_floor += 1
            continue
        # Cosine top-K picks (the pre-guard REPLACE set).
        top_picks = [cid for cid, _ in above[:max(1, resolved_cap)]]
        if keep_originals:
            # KEEP-ORIGINAL guard: never STRIP a synthesis citation. UNION every
            # original that itself clears the floor and is NOT exercise-like into
            # the kept set (in cosine order), then fill remaining cap with the
            # cosine top-K. The cap is widened to max(cap, n_protected) so a
            # protected original is never squeezed out. Exercise-like / below-
            # floor originals stay droppable (neighbor-fix + demote unchanged).
            above_ids = {cid for cid, _ in above}
            old_id_set = set(old_ids)
            protected = [
                cid for cid, _ in above          # cosine-ranked order
                if cid in old_id_set
                and cid in above_ids
                and not ex_like.get(cid, False)
            ]
            result.kept_original_supporters += len(protected)
            kept = list(protected)
            _kept_seen = set(kept)
            for cid in top_picks:
                if cid not in _kept_seen:
                    _kept_seen.add(cid)
                    kept.append(cid)
            kept = kept[:max(max(1, resolved_cap), len(protected))]
        else:
            kept = top_picks
        # PIN the Pass-C NLI-entailing chunk (``entailing_chunk_id``): the
        # keep-original guard above only protects ABOVE-FLOOR, non-exercise
        # originals, but the chunk that EARNED the CO's grounded verdict is
        # often a math-dense worked example — exactly the shape the cosine
        # floor and the exercise-demote strip (the sample-scan-01
        # objective_entailment gate failure class, 2026-07-04). Never strip
        # it: entailment evidence outranks cosine relevance. Cap widens by
        # <= 1 (mirrors the protected-original widening). ANTI-FABRICATION:
        # pinned only when it is an ORIGINAL citation of this CO.
        _pin = str(co.get("entailing_chunk_id") or "").strip()
        if _pin and _pin in set(old_ids) and _pin not in set(kept):
            kept = list(kept) + [_pin]
        # Set-based change detection: a cosine-reordered but SET-IDENTICAL keep
        # is a no-op for provenance — don't rewrite, don't count, don't emit a
        # capture. (A genuinely different set keeps its cosine-ranked order.)
        if set(kept) == set(old_ids):
            continue
        cos_by_id = dict(scored)
        old_best = max(
            (cos_by_id[c] for c in old_ids if c in cos_by_id), default=None
        )
        new_best = ranked[0][1] if ranked else None
        co["source_chunk_ids"] = list(kept)
        _rewrite_source_refs(co, kept, chunks_by_id)
        result.reselected_count += 1
        result.per_co_changes.append({
            "co_index": i,
            "old": old_ids,
            "new": list(kept),
            "old_best_cosine": old_best,
            "new_best_cosine": new_best,
            "exercise_demoted": demoted,
        })
        _emit_reselect_capture(
            capture,
            co_index=i,
            statement=str(co.get("statement") or ""),
            old_ids=old_ids,
            new_ids=list(kept),
            old_best_cos=old_best,
            new_best_cos=new_best,
            pool_size=len(pool),
            floor=resolved_floor,
        )

    result.available = True
    result.citation_density_after = _distinct_cited(cos)
    return result


__all__ = [
    "ENV_CITATION_RESELECT",
    "ENV_RESELECT_EXERCISE_DEMOTE",
    "ENV_RESELECT_KEEP_ORIGINAL",
    "ReselectResult",
    "reselect_citations",
    "resolve_citation_reselect",
    "resolve_reselect_exercise_demote",
    "resolve_reselect_keep_original",
]
