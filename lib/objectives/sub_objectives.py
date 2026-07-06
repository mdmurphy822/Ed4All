"""Concept / sub-objective layer derivation (finding 17).

Every synthesized chapter objective (CO) ships with a ``sub_objectives`` field,
but the 7B path leaves it EMPTY (0/50 on the measured corpus). A CO that bundles
several distinct sub-topics (e.g. "prime factorization" + "divisibility tests"
+ "GCF/LCM") reads as one flat bullet; the Sonnet baseline NESTS related
instruction blocks under a named concept so the Learning-Objectives map + the
course itself are 3-level (TO -> CO -> concept -> instruction block).

This module DERIVES that concept / sub-objective layer for each CO from its
ALREADY-GROUNDED source chunks. Two paths, both fail-soft to an empty list:

1. **Deterministic clustering (default).** The CO's cited source chunks are
   embedded and single-link-clustered by cosine (reusing the SAME
   ``cluster_by_cosine`` core the dedup / bottom-up-TO passes use). Each cluster
   of ≥1 chunk becomes one sub-objective: a short content-derived ``statement``
   (the most salient noun-phrase-ish span from the cluster's representative
   chunk) plus the cluster's ``source_chunk_ids`` — a STRICT SUBSET of the CO's
   own cited chunk ids (anti-fabrication: a sub-objective never references a
   chunk the CO did not already cite). When no embedder is available the path
   degrades to a single whole-CO sub-objective over the CO's chunk set (still
   3-level, just un-clustered) — never empty when the CO HAS grounded chunks,
   never fabricated.

2. **Optional bounded LLM call (behind a provider flag).** Gated EXACTLY like
   the objective-review pass — ``ED4ALL_SUB_OBJECTIVE_PROVIDER`` (unset/empty =
   off; a value in the supported set routes via that endpoint seat) +
   ``ED4ALL_SUB_OBJECTIVE_MODEL`` override. The model is shown ONLY the CO's
   statement + a bounded digest of its grounded chunk text + the chunk-id menu,
   and must return ``{statement, source_chunk_ids}`` sub-objectives whose
   ``source_chunk_ids`` are a SUBSET of the menu (any out-of-set id is dropped;
   a sub-objective left with zero valid ids is dropped). Provider error / unset
   key / unparseable response degrades to the deterministic path — never fails
   the phase.

Anti-fabrication contract (BOTH paths): a sub-objective's ``source_chunk_ids``
is always ``⊆`` the CO's own grounded chunk-id set. No chunk id is ever
invented; a derivation that cannot ground any sub-objective returns ``[]`` and
the CO keeps its empty ``sub_objectives`` (the byte-stable status quo).

Env gate (default OFF for the LLM arm; the deterministic arm runs whenever an
embedder is present):

* ``ED4ALL_SUB_OBJECTIVE_PROVIDER`` — unset/empty = deterministic only;
  ``nvidia`` = run the bounded LLM call via the NVIDIA seat (same construction
  as the objective-review pass).
* ``ED4ALL_SUB_OBJECTIVE_MODEL`` — model-id override (chain: explicit arg →
  this env → ``NVIDIA_LARGE_MODEL`` → the endpoint default).
* ``ED4ALL_SUB_OBJECTIVE_MAX`` — hard cap on sub-objectives derived per CO
  (default 5; garbage / <1 → default). Stops a chunk-rich CO from sprouting a
  dozen thin concepts.

Decision capture: one ``sub_objective_derivation`` event per CO with a dynamic,
replayable rationale (CO id, # chunks in, # sub-objectives out, path used,
embedder availability, model/provider when the LLM arm fired).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.generation.llm_checkpoint import CheckpointStore, compute_fingerprint
from lib.generation.stop_control import check_stop
# The raw-chrome junk-marker tuple is consolidated into the shared apparatus
# lexicon (single source of truth); imported here as a byte-identical re-export.
from lib.objectives.apparatus_lexicon import (
    JUNK_MARKERS as _JUNK_MARKERS,
    compile_profile as _compile_apparatus_profile,
)

logger = logging.getLogger(__name__)

#: Graceful-stop + resume-sidecar wiring (P4 item 6, Resolved Decision D5 —
#: full sidecars even for the bounded per-CO loop). One record per CO; the
#: sidecar lives beside ``synthesized_objectives.json`` (``01_learning_objectives/``).
#: ``checkpoint_path`` unwired → no sidecar (byte-identical default), but the
#: per-CO ``check_stop`` stays live via ``ED4ALL_RUN_ID``. The DETERMINISTIC
#: derivation is untouched — only the ``populate_sub_objectives`` loop gains the
#: stop check + sidecar. Family flag ``ED4ALL_GENERATION_CHECKPOINT`` (site_env=None).
_SUB_OBJECTIVES_SITE_ID = "sub_objectives"
#: Canonical sidecar basename (sibling of ``synthesized_objectives.json``).
SUB_OBJECTIVES_CHECKPOINT_NAME = ".sub_objectives_checkpoint.jsonl"


def _sub_objectives_store(path: Optional[Path]) -> CheckpointStore:
    """Fingerprinted per-CO resume sidecar for the sub-objective derivation."""
    return CheckpointStore(
        path, site_id=_SUB_OBJECTIVES_SITE_ID, site_env=None,
    )

# ---------------------------------------------------------------------------
# Env contract (parse-with-fallback, mirroring the other ED4ALL_* knobs).
# ---------------------------------------------------------------------------

ENV_SUB_OBJECTIVE_PROVIDER = "ED4ALL_SUB_OBJECTIVE_PROVIDER"
ENV_SUB_OBJECTIVE_MODEL = "ED4ALL_SUB_OBJECTIVE_MODEL"
ENV_SUB_OBJECTIVE_MAX = "ED4ALL_SUB_OBJECTIVE_MAX"
#: Defect-D deterministic statement-quality gate (default OFF → byte-identical
#: to the legacy salient-statement extraction). ON enables leading-boilerplate
#: prefix-strip, run-on skip-not-truncate, glossary ``term : definition`` →
#: ``"Define {term}"``, all-cluster-body scanning, Bloom-verb heading keep, and
#: within-CO statement dedupe. Parse-with-fallback, mirroring the other flags.
ENV_SUB_OBJECTIVE_QUALITY = "ED4ALL_SUB_OBJECTIVE_QUALITY"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Providers the LLM arm knows how to construct (the validation allowlist).
#: Mirrors ``objective_review._SUPPORTED_REVIEW_PROVIDERS`` — only the strong
#: hosted ``nvidia`` seat today. Every entry MUST be a registered endpoint name
#: in ``config/endpoints.yaml``.
_SUPPORTED_PROVIDERS = ("nvidia",)

#: Default cap on sub-objectives per CO. A CO bundles a handful of sub-topics;
#: more than this is almost always over-fragmentation of a single concept.
_DEFAULT_MAX_SUB_OBJECTIVES = 5

#: Cosine ≥ this single-link-clusters two of a CO's source chunks into one
#: sub-topic. Deliberately moderate — a CO's chunks are already topically
#: related (they all ground the SAME objective), so the floor separates
#: genuinely distinct sub-topics rather than near-duplicate restatements (the
#: 0.88 dedup floor would collapse everything into one cluster).
_CHUNK_CLUSTER_THRESHOLD = 0.55

#: Decision-capture event type (NEW enum member, registered in
#: ``schemas/events/decision_event.schema.json``).
_DECISION_TYPE = "sub_objective_derivation"

#: Per-request HTTP timeout default for the LLM arm (seconds). Mirrors the
#: review pass — a generous floor over the bare 60s client default.
_LLM_DEFAULT_TIMEOUT_SECONDS = 300.0

#: Output budget + temperature for the bounded LLM call.
_LLM_MAX_TOKENS = 4096
_LLM_TEMPERATURE = 0.2

#: How many chunks of a CO's grounding to show the LLM (bounded prompt).
_LLM_MAX_CHUNKS_IN_PROMPT = 12

#: Chars of each chunk body shown to the LLM (bounded prompt).
_LLM_CHUNK_BODY_CHARS = 600

#: Sentinel distinguishing "caller omitted ``embedder``" (→ auto-load the
#: default sentence-embedder) from an explicit ``embedder=None`` (→ run the
#: deterministic path WITHOUT clustering — a single un-clustered sub-objective,
#: never blind-clustered). The docstring contract requires this distinction;
#: ``None`` alone could not express "no embedder" because the lazy-load
#: previously fired on it.
_UNSET = object()


def resolve_sub_objective_provider(
    provider: Optional[str] = None,
) -> Optional[str]:
    """Resolve the LLM-arm provider: explicit arg → env → off.

    Returns the lower-cased provider name when recognised, else ``None`` (the
    deterministic-only default). Garbage / unknown / empty → ``None``
    (parse-with-fallback — an unknown provider never crashes a run; it just
    disables the LLM arm). Mirrors
    ``objective_review.resolve_objective_review_provider``.
    """
    raw = provider if provider is not None else os.environ.get(
        ENV_SUB_OBJECTIVE_PROVIDER
    )
    if not raw:
        return None
    candidate = str(raw).strip().lower()
    if candidate in _SUPPORTED_PROVIDERS:
        return candidate
    logger.warning(
        "%s=%r is not a supported sub-objective provider (%s); the "
        "deterministic clustering path will run instead.",
        ENV_SUB_OBJECTIVE_PROVIDER,
        raw,
        list(_SUPPORTED_PROVIDERS),
    )
    return None


def resolve_sub_objective_model(
    provider: str,
    model: Optional[str] = None,
) -> Optional[str]:
    """Resolve the LLM-arm model: explicit arg → env → registry default.

    Chain: explicit arg → ``ED4ALL_SUB_OBJECTIVE_MODEL`` → the endpoint row's
    ``model_env`` (e.g. ``NVIDIA_LARGE_MODEL``) → the endpoint ``default_model``.
    ``None`` lets the client fall back to its own resolution. Mirrors
    ``objective_review.resolve_objective_review_model`` (sans the review-pass
    70B-specific default — the sub-objective task is short, so the endpoint
    default is fine).
    """
    if model:
        return str(model).strip()
    env_model = os.environ.get(ENV_SUB_OBJECTIVE_MODEL)
    if env_model and env_model.strip():
        return env_model.strip()
    try:
        from lib.llm.endpoints import load_endpoint_registry  # noqa: PLC0415

        row = load_endpoint_registry().get(provider)
    except Exception:  # noqa: BLE001 — registry load issues surface elsewhere
        return None
    if not row:
        return None
    env_var = row.get("model_env")
    if env_var:
        env_default = os.environ.get(env_var)
        if env_default and env_default.strip():
            return env_default.strip()
    default_model = row.get("default_model")
    return str(default_model) if default_model else None


def resolve_max_sub_objectives(value: Optional[int] = None) -> int:
    """Resolve the per-CO sub-objective cap: explicit arg → env → default.

    Garbage / ``< 1`` → :data:`_DEFAULT_MAX_SUB_OBJECTIVES` (a misconfigured cap
    must never silently disable derivation or drop to zero). Parse-with-fallback,
    mirroring ``objective_dedup.resolve_max_chunks_per_objective``.
    """
    raw = value if value is not None else os.environ.get(ENV_SUB_OBJECTIVE_MAX)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_MAX_SUB_OBJECTIVES
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_SUB_OBJECTIVES
    if parsed < 1:
        return _DEFAULT_MAX_SUB_OBJECTIVES
    return parsed


def resolve_sub_objective_quality(value: Optional[bool] = None) -> bool:
    """Resolve ``ED4ALL_SUB_OBJECTIVE_QUALITY`` (default **OFF**).

    Explicit arg > env. Only the truthy tokens ``1`` / ``true`` / ``yes`` /
    ``on`` (any case) enable the Defect-D statement-quality path; unset /
    anything else keeps the byte-identical legacy extraction (parse-with-
    fallback, mirroring the sibling default-OFF resolvers).
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(ENV_SUB_OBJECTIVE_QUALITY)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _resolve_llm_timeout() -> float:
    """Per-request HTTP timeout for the LLM arm (parse-with-fallback)."""
    import math as _math  # noqa: PLC0415

    raw = os.environ.get("ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS")
    if not raw or not str(raw).strip():
        return _LLM_DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return _LLM_DEFAULT_TIMEOUT_SECONDS
    if not _math.isfinite(parsed) or parsed <= 0:
        return _LLM_DEFAULT_TIMEOUT_SECONDS
    return parsed


# ---------------------------------------------------------------------------
# CO grounding extraction (pure — chunk-id set the sub-objectives must stay in)
# ---------------------------------------------------------------------------


def _co_statement(co: Dict[str, Any]) -> str:
    val = co.get("statement")
    return str(val).strip() if isinstance(val, str) else ""


def _co_chunk_ids(co: Dict[str, Any]) -> List[str]:
    """Return the CO's grounded chunk-id set, de-duplicated in first-seen order.

    Reads BOTH the flat ``source_chunk_ids`` and the nested
    ``source_refs[].chunk_ids`` (the two shapes the synthesizer emits) so the
    full grounded set is recovered regardless of which the upstream pass wrote.
    This set is the ANTI-FABRICATION boundary: a derived sub-objective's
    ``source_chunk_ids`` is always a subset of it.
    """
    seen: Dict[str, None] = {}

    def _add(cid: Any) -> None:
        if isinstance(cid, str) and cid.strip():
            seen.setdefault(cid.strip(), None)

    for cid in co.get("source_chunk_ids") or []:
        _add(cid)
    for ref in co.get("source_refs") or []:
        if isinstance(ref, dict):
            for cid in ref.get("chunk_ids") or []:
                _add(cid)
    return list(seen.keys())


def _chunk_body(chunk: Any) -> str:
    """Body text of a chunk dict (``text`` / ``body`` keys), else ``""``."""
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("body") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Statement naming (content-derived, deterministic)
# ---------------------------------------------------------------------------

# A "concept-ish" span: 2–5 capitalized-or-content words, or a math-ish term.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: OpenStax / generic structural & pedagogical labels that lead a NON-concept
#: line (a callout / exercise / answer-key header, not a teachable concept). A
#: sentence whose first word is one of these is leaked source chrome, not an
#: objective.
_NON_CONCEPT_LEAD = frozenset(
    {
        "be", "try", "how", "example", "solution", "exercises", "exercise",
        "chapter", "section", "figure", "table", "learning", "objectives",
        "objective", "answer", "answers", "key", "glossary", "practice",
        "review", "note", "tip", "warning", "checklist", "step", "prepared",
    }
)

#: Substrings marking raw exercise / callout / answer-key / glyph fragments that
#: leak verbatim from OpenStax-style source chunks (e.g. "BE PREPARED : : 1.1",
#: "TRY IT : : 1.6", the circled-letter exercise glyphs). A sentence containing
#: any of these is not a concept statement. ``_JUNK_MARKERS`` is imported from
#: lib.objectives.apparatus_lexicon (byte-identical re-export).


def _looks_like_concept_sentence(s: str) -> bool:
    """True when ``s`` reads as a teachable concept sentence.

    Rejects the raw exercise / callout / answer-key / math fragments that leak
    verbatim from source chunks (the ``BE PREPARED`` / ``TRY IT`` / glyph /
    bare-equation noise). Pure + deterministic: a sentence qualifies only when it
    has ≥3 words, carries no junk marker, does not lead with a structural label
    or a digit, and is mostly letters (a concept reads as prose, not symbols).
    """
    text = (s or "").strip()
    if len(text.split()) < 3:
        return False
    if any(marker in text for marker in _JUNK_MARKERS):
        return False
    lead = text.split(" ", 1)[0].lower().strip(".,;:()")
    # A concept sentence opens on a real word — not a structural label, a bare
    # number, or a math token ("3+5=8 ...", "|n| ...", "5+3 ...").
    if lead in _NON_CONCEPT_LEAD or not lead[:1].isalpha():
        return False
    # Reject symbol-/math-dominated spans ("3+5=8 ...", "|n| ≥ 0 ..."): a concept
    # sentence is overwhelmingly letters + spaces.
    letters = sum(c.isalpha() or c.isspace() for c in text)
    if letters / max(len(text), 1) < 0.65:
        return False
    return True


def _salient_statement(body: str, fallback: str) -> str:
    """Derive a short, content-derived sub-objective statement from chunk text.

    Deterministic: scans the chunk body for the FIRST sentence that reads like a
    teachable concept (:func:`_looks_like_concept_sentence`) and trims it to a
    concise clause. Raw exercise / callout / answer-key fragments
    (``BE PREPARED``, ``TRY IT``, glyphs, bare math) are skipped; when no clean
    concept sentence is found the CO statement (``fallback``) is used, so a
    sub-objective is always a real objective — never leaked source chrome and
    never the historical ``"Understand: <junk>"`` template prefix. NOT a
    summarizer — a pure, replayable span extraction.
    """
    text = (body or "").strip()
    if not text:
        return fallback
    for cand in _SENTENCE_SPLIT.split(text):
        cand = re.sub(r"\s+", " ", cand).strip().rstrip(":;,. ")
        if not _looks_like_concept_sentence(cand):
            continue
        words = cand.split(" ")
        if len(words) > 14:
            cand = " ".join(words[:14])
        return cand
    return fallback


# ---------------------------------------------------------------------------
# Defect-D deterministic statement quality (flag-gated; default OFF)
# ---------------------------------------------------------------------------

#: A run-on beyond this word count is SKIPPED (never head-truncated mid-clause)
#: unless it ends at a real sentence boundary within the budget.
_MAX_QUALITY_WORDS = 18

#: Glossary shape: a short leading term (1-4 words) then ``: <definition>``.
_GLOSSARY_RE = re.compile(
    r"^\s*([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})\s*:\s*\S"
)


def _leads_with_bloom_verb(text: str) -> bool:
    """Whether ``text``'s first word is a canonical Bloom / action verb.

    Uses ``lib.ontology.bloom.get_all_verbs`` (single source of truth) — a
    heading is kept verbatim only when it opens on a real objective verb, so
    the banned ``"Understand: <x>"`` templating never reappears.
    """
    first = (text or "").strip().split(" ", 1)[0].lower().strip(".,;:()")
    if not first:
        return False
    from lib.ontology.bloom import get_all_verbs  # noqa: PLC0415

    return first in get_all_verbs()


def _glossary_define(text: str) -> Optional[str]:
    """Map a glossary ``term : definition`` line to ``"Define {term}"``.

    Returns ``None`` when ``text`` is not glossary-shaped (no short leading term
    followed by a colon + definition body) or the term carries a junk marker.
    """
    m = _GLOSSARY_RE.match(text or "")
    if not m:
        return None
    term = m.group(1).strip()
    if not term or any(mk in term for mk in _JUNK_MARKERS):
        return None
    return f"Define {term}"


def _salient_statement_quality(
    bodies: List[str], fallback: str, profile: Any
) -> str:
    """Quality salient-statement extraction over ALL cluster-member bodies.

    Deterministic, replayable. For each member body in order:

    1. strip a leading learning-objective boilerplate preamble (recover the
       tail after the colon; a pure-preamble line contributes nothing);
    2. a glossary ``term : definition`` line → ``"Define {term}"``;
    3. otherwise the FIRST sentence that either (a) leads with a canonical
       Bloom verb and is short + marker-free (kept verbatim), or (b) ends at a
       real sentence boundary within :data:`_MAX_QUALITY_WORDS` and reads as a
       concept sentence. A run-on with no in-budget boundary is SKIPPED — never
       head-truncated mid-clause.

    Falls back to ``fallback`` (the CO statement) only when no member body
    yields a clean statement.
    """
    for body in bodies:
        text = profile.strip_lo_boilerplate_prefix((body or "").strip())
        if not text:
            continue
        glossary = _glossary_define(text)
        if glossary:
            return glossary
        for raw in _SENTENCE_SPLIT.split(text):
            s = re.sub(r"\s+", " ", raw).strip()
            if not s:
                continue
            ends_clean = s.endswith((".", "!", "?"))
            core = s.rstrip(":;,. ")
            if not core:
                continue
            words = core.split(" ")
            # Bloom-verb heading — kept verbatim even without terminal
            # punctuation, provided it is short and marker-free.
            if (
                2 <= len(words) <= _MAX_QUALITY_WORDS
                and _leads_with_bloom_verb(core)
                and not any(mk in core for mk in _JUNK_MARKERS)
            ):
                return core
            # Otherwise demand a real boundary within budget; never truncate.
            if not ends_clean or len(words) > _MAX_QUALITY_WORDS:
                continue
            if not _looks_like_concept_sentence(core):
                continue
            return core
    return fallback


def _dedupe_within_co(
    subs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Collapse sub-objectives with an identical normalized statement.

    Merges the source_chunk_ids of duplicates into the first occurrence (no
    chunk id is ever dropped — anti-loss). Empty statements are never merged
    together.
    """
    out: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}
    for so in subs:
        stmt = str(so.get("statement", ""))
        key = re.sub(r"\s+", " ", stmt.strip().lower())
        ids = list(so.get("source_chunk_ids", []))
        if key and key in index:
            existing = out[index[key]]
            seen = set(existing["source_chunk_ids"])
            for cid in ids:
                if cid not in seen:
                    existing["source_chunk_ids"].append(cid)
                    seen.add(cid)
            continue
        if key:
            index[key] = len(out)
        out.append({"statement": stmt, "source_chunk_ids": ids})
    return out


# ---------------------------------------------------------------------------
# Deterministic clustering path
# ---------------------------------------------------------------------------


def _derive_deterministic(
    *,
    co: Dict[str, Any],
    chunk_ids: List[str],
    chunks_by_id: Dict[str, Any],
    embedder: Optional[Any],
    max_sub_objectives: int,
    quality: bool = False,
    profile: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Cluster the CO's source chunks into named sub-objectives.

    Returns a list of ``{statement, source_chunk_ids}`` dicts whose
    ``source_chunk_ids`` partition (a subset of) the CO's grounded chunk set.
    Degrades:

    * no chunk bodies available → single whole-CO sub-objective over ``chunk_ids``
      (still 3-level, un-clustered);
    * no embedder → single whole-CO sub-objective (we never cluster blind);
    * ``< 2`` chunks → single sub-objective.

    Never fabricates a chunk id; never returns more than ``max_sub_objectives``.

    ``quality`` (Defect D, default OFF) routes to the parallel
    :func:`_derive_deterministic_quality` path. When OFF the body below is the
    byte-identical legacy extraction.
    """
    if quality:
        prof = profile if profile is not None else _compile_apparatus_profile()
        return _derive_deterministic_quality(
            co=co,
            chunk_ids=chunk_ids,
            chunks_by_id=chunks_by_id,
            embedder=embedder,
            max_sub_objectives=max_sub_objectives,
            profile=prof,
        )
    co_stmt = _co_statement(co)
    # Pair each chunk id with its body; keep order stable.
    bodies: List[str] = [_chunk_body(chunks_by_id.get(cid)) for cid in chunk_ids]

    def _single(ids: List[str], body: str) -> List[Dict[str, Any]]:
        if not ids:
            return []
        return [
            {
                "statement": _salient_statement(body, co_stmt or "Key concept"),
                "source_chunk_ids": list(ids),
            }
        ]

    if len(chunk_ids) < 2 or embedder is None:
        # One sub-objective over the whole grounded set — still 3-level, and
        # never blind-clustered. Name it from the first non-empty body.
        first_body = next((b for b in bodies if b), "")
        return _single(chunk_ids, first_body)

    # Embed each chunk body; a chunk with no body can't be clustered on content,
    # so it rides into the nearest cluster's leftover bucket (kept, never lost).
    vecs: List[Optional[List[float]]] = []
    for body in bodies:
        if not body:
            vecs.append(None)
            continue
        try:
            v = embedder.encode(body)
            vecs.append(list(v) if v is not None else None)
        except Exception as exc:  # noqa: BLE001 — degrade to no-cluster
            logger.warning("sub_objectives: chunk encode failed: %s", exc)
            vecs.append(None)

    embeddable_idx = [i for i, v in enumerate(vecs) if v is not None]
    if len(embeddable_idx) < 2:
        first_body = next((b for b in bodies if b), "")
        return _single(chunk_ids, first_body)

    from lib.objectives.objective_dedup import cluster_by_cosine  # noqa: PLC0415

    sub_vecs = [vecs[i] for i in embeddable_idx]  # type: ignore[index]
    clusters, _max_cos, _near = cluster_by_cosine(
        sub_vecs, _CHUNK_CLUSTER_THRESHOLD
    )

    # Map local cluster member indices back to GLOBAL chunk indices.
    global_clusters: List[List[int]] = [
        [embeddable_idx[local] for local in members] for members in clusters
    ]
    # Chunks with no body (no vector) were excluded from clustering — fold each
    # into the FIRST cluster so its id is never dropped (anti-loss).
    no_body_idx = [i for i, v in enumerate(vecs) if v is None]
    if no_body_idx and global_clusters:
        global_clusters[0].extend(no_body_idx)
    elif no_body_idx:  # no clusters at all (shouldn't happen with ≥2 embeddable)
        global_clusters = [no_body_idx]

    sub_objectives: List[Dict[str, Any]] = []
    for members in global_clusters:
        members_sorted = sorted(set(members))
        ids = [chunk_ids[i] for i in members_sorted]
        # Representative body = the first non-empty body in the cluster.
        rep_body = next((bodies[i] for i in members_sorted if bodies[i]), "")
        sub_objectives.append(
            {
                "statement": _salient_statement(rep_body, co_stmt or "Key concept"),
                "source_chunk_ids": ids,
            }
        )

    # Cap: when there are more clusters than the budget, merge the smallest
    # tail clusters into the last kept one (chunk ids are never dropped).
    if len(sub_objectives) > max_sub_objectives:
        kept = sub_objectives[: max_sub_objectives - 1]
        overflow_ids: List[str] = []
        overflow_body = ""
        for extra in sub_objectives[max_sub_objectives - 1:]:
            overflow_ids.extend(extra["source_chunk_ids"])
            if not overflow_body:
                overflow_body = extra["statement"]
        kept.append(
            {
                "statement": _salient_statement(
                    overflow_body, co_stmt or "Additional concepts"
                ),
                "source_chunk_ids": overflow_ids,
            }
        )
        sub_objectives = kept

    return sub_objectives


def _derive_deterministic_quality(
    *,
    co: Dict[str, Any],
    chunk_ids: List[str],
    chunks_by_id: Dict[str, Any],
    embedder: Optional[Any],
    max_sub_objectives: int,
    profile: Any,
) -> List[Dict[str, Any]]:
    """Defect-D statement-quality variant of :func:`_derive_deterministic`.

    Same clustering + anti-loss + cap semantics; the ONLY differences are the
    statement derivation (:func:`_salient_statement_quality`, scanning ALL
    cluster-member bodies with boilerplate-strip / glossary / run-on-skip /
    Bloom-heading rules) and a final within-CO statement dedupe. A separate
    function so the flag-OFF path in :func:`_derive_deterministic` stays
    byte-identical.
    """
    co_stmt = _co_statement(co)
    bodies: List[str] = [_chunk_body(chunks_by_id.get(cid)) for cid in chunk_ids]

    def _single(ids: List[str], member_bodies: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        return [
            {
                "statement": _salient_statement_quality(
                    member_bodies, co_stmt or "Key concept", profile
                ),
                "source_chunk_ids": list(ids),
            }
        ]

    if len(chunk_ids) < 2 or embedder is None:
        return _dedupe_within_co(_single(chunk_ids, bodies))

    vecs: List[Optional[List[float]]] = []
    for body in bodies:
        if not body:
            vecs.append(None)
            continue
        try:
            v = embedder.encode(body)
            vecs.append(list(v) if v is not None else None)
        except Exception as exc:  # noqa: BLE001 — degrade to no-cluster
            logger.warning("sub_objectives: chunk encode failed: %s", exc)
            vecs.append(None)

    embeddable_idx = [i for i, v in enumerate(vecs) if v is not None]
    if len(embeddable_idx) < 2:
        return _dedupe_within_co(_single(chunk_ids, bodies))

    from lib.objectives.objective_dedup import cluster_by_cosine  # noqa: PLC0415

    sub_vecs = [vecs[i] for i in embeddable_idx]  # type: ignore[index]
    clusters, _max_cos, _near = cluster_by_cosine(
        sub_vecs, _CHUNK_CLUSTER_THRESHOLD
    )
    global_clusters: List[List[int]] = [
        [embeddable_idx[local] for local in members] for members in clusters
    ]
    no_body_idx = [i for i, v in enumerate(vecs) if v is None]
    if no_body_idx and global_clusters:
        global_clusters[0].extend(no_body_idx)
    elif no_body_idx:
        global_clusters = [no_body_idx]

    sub_objectives: List[Dict[str, Any]] = []
    for members in global_clusters:
        members_sorted = sorted(set(members))
        ids = [chunk_ids[i] for i in members_sorted]
        member_bodies = [bodies[i] for i in members_sorted]
        sub_objectives.append(
            {
                "statement": _salient_statement_quality(
                    member_bodies, co_stmt or "Key concept", profile
                ),
                "source_chunk_ids": ids,
            }
        )

    # Cap: merge the smallest tail clusters into the last kept one, preserving
    # the first overflow cluster's (already-clean) statement; ids never dropped.
    if len(sub_objectives) > max_sub_objectives:
        kept = sub_objectives[: max_sub_objectives - 1]
        overflow = sub_objectives[max_sub_objectives - 1:]
        overflow_ids: List[str] = []
        for extra in overflow:
            overflow_ids.extend(extra["source_chunk_ids"])
        kept.append(
            {
                "statement": overflow[0]["statement"],
                "source_chunk_ids": overflow_ids,
            }
        )
        sub_objectives = kept

    return _dedupe_within_co(sub_objectives)


# ---------------------------------------------------------------------------
# Optional bounded LLM path
# ---------------------------------------------------------------------------


def _build_llm_client(
    provider: str,
    *,
    model: Optional[str],
    client: Optional[Any] = None,
) -> Any:
    """Construct the OpenAI-compatible client via the unified endpoint registry.

    The SAME constructor the objective-review pass + the ``nvidia`` content-gen
    provider route through. Raises (``EndpointKeyRequired`` etc.) when the key
    is unset and no test client is injected — the caller catches it and degrades
    to the deterministic path.
    """
    from lib.llm.endpoints import (  # noqa: PLC0415
        build_openai_compatible_client,
    )

    resolved_model = resolve_sub_objective_model(provider, model)
    return build_openai_compatible_client(
        provider,
        model=resolved_model,
        capture=None,
        provider_label=provider,
        client=client,
        json_mode=True,
        timeout=_resolve_llm_timeout(),
    )


def _render_llm_prompt(
    *,
    co_statement: str,
    chunk_ids: List[str],
    chunks_by_id: Dict[str, Any],
    max_sub_objectives: int,
) -> str:
    """Render the bounded sub-objective prompt for ONE CO."""
    menu = []
    for cid in chunk_ids[:_LLM_MAX_CHUNKS_IN_PROMPT]:
        body = _chunk_body(chunks_by_id.get(cid))[:_LLM_CHUNK_BODY_CHARS]
        menu.append({"chunk_id": cid, "text": body})
    payload = {
        "objective": co_statement,
        "source_chunks": menu,
    }
    return (
        "You are an instructional designer. Break the LEARNING OBJECTIVE below "
        "into its distinct sub-topics (concepts). Each sub-topic is grounded in "
        "ONE OR MORE of the provided source chunks.\n\n"
        f"Return at most {max_sub_objectives} sub-objectives. For each, give a "
        "short content-derived 'statement' (one clause naming the concept) and "
        "a 'source_chunk_ids' list. EVERY id in 'source_chunk_ids' MUST be a "
        "'chunk_id' that appears in 'source_chunks' below — never invent an id. "
        "Do not repeat the whole objective; name the SPECIFIC sub-topic.\n\n"
        'Return ONLY JSON of the form: {"sub_objectives": [{"statement": '
        '"...", "source_chunk_ids": ["..."]}, ...]}\n\n'
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _parse_llm_sub_objectives(
    raw_text: str,
    allowed_ids: set,
    max_sub_objectives: int,
) -> Optional[List[Dict[str, Any]]]:
    """Parse + anti-fabrication-filter the LLM response, or ``None``.

    Drops any ``source_chunk_ids`` not in ``allowed_ids`` (the CO's own grounded
    set); a sub-objective left with zero valid ids is dropped. Returns ``None``
    when nothing parseable / nothing grounded survives (caller falls back).
    """
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        try:
            parsed = json.loads(text[start:end])
        except (ValueError, TypeError):
            return None
    if not isinstance(parsed, dict):
        return None
    items = parsed.get("sub_objectives")
    if not isinstance(items, list):
        return None

    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        stmt = item.get("statement")
        if not isinstance(stmt, str) or not stmt.strip():
            continue
        raw_ids = item.get("source_chunk_ids") or []
        # ANTI-FABRICATION: keep only ids the CO actually cited; preserve order.
        kept_ids: List[str] = []
        seen: set = set()
        for cid in raw_ids:
            if isinstance(cid, str) and cid.strip():
                key = cid.strip()
                if key in allowed_ids and key not in seen:
                    kept_ids.append(key)
                    seen.add(key)
        if not kept_ids:
            continue  # a sub-objective with no grounded chunk is dropped
        out.append(
            {"statement": stmt.strip(), "source_chunk_ids": kept_ids}
        )
        if len(out) >= max_sub_objectives:
            break
    return out or None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def derive_sub_objectives_for_co(
    *,
    co: Dict[str, Any],
    chunks_by_id: Dict[str, Any],
    embedder: Optional[Any] = _UNSET,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    max_sub_objectives: Optional[int] = None,
    capture: Optional[Any] = None,
    quality: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Derive the concept / sub-objective layer for ONE chapter objective.

    Returns a list of ``{statement, source_chunk_ids}`` sub-objectives whose
    ``source_chunk_ids`` are a STRICT SUBSET of the CO's own grounded chunk-id
    set (anti-fabrication). NEVER raises — any error logs a warning and returns
    ``[]`` (the CO keeps its empty ``sub_objectives``, byte-stable status quo).

    Path selection:

    * ``ED4ALL_SUB_OBJECTIVE_PROVIDER`` set (and the client constructs) → the
      bounded LLM arm runs first; its output is anti-fabrication-filtered. An
      empty / unparseable / errored LLM result degrades to the deterministic
      clustering path.
    * otherwise → the deterministic clustering path.

    A CO with NO grounded chunk ids returns ``[]`` (nothing to ground a
    sub-objective on — never fabricated).

    Args:
        co: the chapter-objective dict (read-only here; the caller writes the
            returned list onto ``co['sub_objectives']``).
        chunks_by_id: ``{chunk_id: chunk_dict}`` for chunk-body lookup. A
            missing id simply yields an empty body (still grounded by id).
        embedder: CO-chunk encoder (test seam); defaults to
            ``try_load_embedder()``. ``None`` → the deterministic path degrades
            to a single un-clustered sub-objective.
        provider / model: explicit overrides for the LLM arm (else env-resolved).
        client: test seam for the OpenAI-compatible HTTP client.
        max_sub_objectives: per-CO cap (else env-resolved, default 5).
        capture: ``DecisionCapture`` — one ``sub_objective_derivation`` event.
    """
    co_id = str(co.get("id") or "")
    chunk_ids = _co_chunk_ids(co)
    cap = resolve_max_sub_objectives(max_sub_objectives)
    resolved_quality = resolve_sub_objective_quality(quality)

    if not chunk_ids:
        _emit_decision(
            capture,
            co_id=co_id,
            chunk_count=0,
            sub_count=0,
            path="none",
            embedder_available=embedder is not None,
            provider=None,
            model=None,
        )
        return []

    # Lazy-load the embedder (test seam wins) — needed by BOTH the deterministic
    # clustering and as the degrade target of the LLM arm. An OMITTED embedder
    # (the ``_UNSET`` sentinel) auto-loads the default; an EXPLICIT ``None``
    # means "no embedder" → the deterministic path degrades to a single
    # un-clustered sub-objective (the documented contract).
    if embedder is _UNSET:
        try:
            from lib.embedding.sentence_embedder import (  # noqa: PLC0415
                try_load_embedder,
            )

            scorer_embedder = try_load_embedder()
        except Exception as exc:  # noqa: BLE001
            logger.warning("sub_objectives: embedder load failed: %s", exc)
            scorer_embedder = None
    else:
        scorer_embedder = embedder

    allowed_ids = set(chunk_ids)
    path_used = "deterministic"
    resolved_provider = resolve_sub_objective_provider(provider)
    resolved_model: Optional[str] = None
    sub_objectives: Optional[List[Dict[str, Any]]] = None

    if resolved_provider is not None:
        resolved_model = resolve_sub_objective_model(resolved_provider, model)
        try:
            oa_client = _build_llm_client(
                resolved_provider, model=resolved_model, client=client
            )
            raw_text = oa_client.chat_completion(
                [
                    {
                        "role": "user",
                        "content": _render_llm_prompt(
                            co_statement=_co_statement(co),
                            chunk_ids=chunk_ids,
                            chunks_by_id=chunks_by_id,
                            max_sub_objectives=cap,
                        ),
                    }
                ],
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE,
            )
            parsed = _parse_llm_sub_objectives(raw_text, allowed_ids, cap)
            if parsed:
                sub_objectives = parsed
                path_used = "llm"
            else:
                logger.warning(
                    "sub_objectives: LLM arm for CO %s returned no grounded "
                    "sub-objectives (provider=%s); using deterministic path.",
                    co_id, resolved_provider,
                )
        except Exception as exc:  # noqa: BLE001 — degrade, never fail the phase
            logger.warning(
                "sub_objectives: LLM arm for CO %s failed (provider=%s, "
                "model=%s): %s. Falling back to deterministic clustering.",
                co_id, resolved_provider, resolved_model, exc,
            )

    if sub_objectives is None:
        try:
            sub_objectives = _derive_deterministic(
                co=co,
                chunk_ids=chunk_ids,
                chunks_by_id=chunks_by_id,
                embedder=scorer_embedder,
                max_sub_objectives=cap,
                quality=resolved_quality,
            )
        except Exception as exc:  # noqa: BLE001 — never fail the phase
            logger.warning(
                "sub_objectives: deterministic derivation for CO %s failed: "
                "%s. Returning empty sub_objectives.",
                co_id, exc,
            )
            sub_objectives = []

    # Final anti-fabrication backstop: a returned id MUST be in the CO's set.
    cleaned: List[Dict[str, Any]] = []
    for so in sub_objectives or []:
        ids = [
            cid
            for cid in so.get("source_chunk_ids", [])
            if cid in allowed_ids
        ]
        if not ids:
            continue
        cleaned.append(
            {"statement": str(so.get("statement", "")).strip(),
             "source_chunk_ids": ids}
        )

    _emit_decision(
        capture,
        co_id=co_id,
        chunk_count=len(chunk_ids),
        sub_count=len(cleaned),
        path=path_used if cleaned else "deterministic",
        embedder_available=scorer_embedder is not None,
        provider=resolved_provider,
        model=resolved_model,
    )
    return cleaned


def populate_sub_objectives(
    *,
    chapter_objectives: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    embedder: Optional[Any] = _UNSET,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    capture: Optional[Any] = None,
    checkpoint_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, int]:
    """Populate ``sub_objectives`` on every CO in ``chapter_objectives`` IN PLACE.

    ``chapter_objectives`` is the on-disk group shape
    (``[{chapter, objectives: [CO, ...]}]``) OR a flat ``[CO, ...]`` list — both
    are accepted. Each CO's ``sub_objectives`` is OVERWRITTEN only when the
    derivation yields ≥1 grounded sub-objective; a CO that yields none keeps its
    existing (empty) ``sub_objectives`` unchanged. Returns a counters dict
    (``cos_seen`` / ``cos_populated`` / ``sub_objectives_total``).

    A per-CO derivation failure logs a warning and leaves THAT CO untouched.

    Graceful stop (P4 item 6 / D5): a pending stop sentinel raises
    ``GracefulStopRequested`` at the per-CO boundary (checked at the loop top,
    BEFORE deriving that CO — and before the per-CO ``except`` guard, so it is
    never swallowed). Every already-derived CO is on the resume sidecar
    (``checkpoint_path``, when supplied); a resume replays those without
    re-deriving (skipping the LLM arm's call when a provider is set). The
    sidecar is removed once every CO is processed. The deterministic default
    path (no provider) is byte-identical — the sidecar only caches results and
    adds no output.

    Args (stop/resume additions):
        checkpoint_path: per-CO resume sidecar path (natural sibling of
            ``synthesized_objectives.json``). ``None`` disables the sidecar.
        run_id: explicit run id for the stop-sentinel probe (else
            ``ED4ALL_RUN_ID``).
    """
    counters = {"cos_seen": 0, "cos_populated": 0, "sub_objectives_total": 0}

    def _iter_cos() -> List[Dict[str, Any]]:
        cos: List[Dict[str, Any]] = []
        for entry in chapter_objectives or []:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("objectives"), list):
                for co in entry["objectives"]:
                    if isinstance(co, dict):
                        cos.append(co)
            elif entry.get("id"):  # flat CO list
                cos.append(entry)
        return cos

    # Resolve the embedder ONCE (shared across COs) when not injected. An
    # OMITTED embedder (the ``_UNSET`` sentinel) auto-loads the default; an
    # EXPLICIT ``None`` means "no embedder" (single un-clustered sub-objective).
    if embedder is _UNSET:
        try:
            from lib.embedding.sentence_embedder import (  # noqa: PLC0415
                try_load_embedder,
            )

            shared_embedder = try_load_embedder()
        except Exception as exc:  # noqa: BLE001
            logger.warning("sub_objectives: embedder load failed: %s", exc)
            shared_embedder = None
    else:
        shared_embedder = embedder

    store = _sub_objectives_store(checkpoint_path)
    cache = store.load()
    # Resolve the provider/model ONCE for the fingerprint (they don't change
    # per CO). The provider axis in the fingerprint keeps the deterministic and
    # LLM-arm records from colliding for the same CO.
    resolved_provider = resolve_sub_objective_provider(provider)
    resolved_model = (
        resolve_sub_objective_model(resolved_provider, model)
        if resolved_provider is not None else None
    )
    cap = resolve_max_sub_objectives(None)
    # Defect-D quality flag on the fingerprint so a flag flip invalidates cached
    # per-CO derivations (a resume never replays a stale legacy statement).
    resolved_quality = resolve_sub_objective_quality(None)

    for _idx, co in enumerate(_iter_cos()):
        # Stop at the per-CO unit boundary — BEFORE this CO's derivation AND
        # before the per-CO ``except`` guard, so a graceful stop is never
        # swallowed. Sequential loop → a plain raise is correct.
        check_stop(_SUB_OBJECTIVES_SITE_ID, counters["cos_seen"], run_id=run_id)
        counters["cos_seen"] += 1
        unit_id = str(co.get("id") or f"co#{_idx}")
        fingerprint = compute_fingerprint({
            "stmt": _co_statement(co),
            "chunk_ids": sorted(_co_chunk_ids(co)),
            "provider": resolved_provider or "",
            "model": resolved_model or "",
            "cap": cap,
            "quality": resolved_quality,
        })
        reused = store.reuse(unit_id, fingerprint, cache=cache)
        if reused is not None:
            derived = [
                s for s in (reused.get("sub_objectives") or [])
                if isinstance(s, dict)
            ]
        else:
            try:
                derived = derive_sub_objectives_for_co(
                    co=co,
                    chunks_by_id=chunks_by_id,
                    embedder=shared_embedder,
                    provider=provider,
                    model=model,
                    client=client,
                    capture=capture,
                    quality=resolved_quality,
                )
            except Exception as exc:  # noqa: BLE001 — isolate per-CO failure
                logger.warning(
                    "sub_objectives: derivation raised for CO %s: %s. Leaving "
                    "its sub_objectives unchanged.",
                    co.get("id"), exc,
                )
                continue
            # Checkpoint the derived result (even an empty list is a valid
            # outcome) so a stop at the next loop-top leaves this CO recoverable
            # and a resume skips its (possibly LLM-backed) derivation.
            store.append(unit_id, fingerprint, {"sub_objectives": derived})
        if derived:
            co["sub_objectives"] = derived
            counters["cos_populated"] += 1
            counters["sub_objectives_total"] += len(derived)

    # Every CO processed without a stop → drop the resume sidecar.
    store.remove()
    return counters


def _emit_decision(
    capture: Optional[Any],
    *,
    co_id: str,
    chunk_count: int,
    sub_count: int,
    path: str,
    embedder_available: bool,
    provider: Optional[str],
    model: Optional[str],
) -> None:
    """Emit one ``sub_objective_derivation`` decision per CO (dynamic rationale)."""
    if capture is None:
        return
    decision = f"sub_objectives:{co_id or 'CO'}:{sub_count}from{chunk_count}:{path}"
    rationale = (
        f"Derived the concept / sub-objective layer for {co_id or 'a CO'} "
        f"(finding 17): {chunk_count} grounded source chunk(s) clustered into "
        f"{sub_count} sub-objective(s) via the {path} path; "
        f"embedder_available={embedder_available}, provider={provider or 'none'}, "
        f"model={model or 'none'}. source_chunk_ids are a strict subset of the "
        f"CO's cited chunks (anti-fabrication)."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the phase
        logger.debug(
            "DecisionCapture.log_decision raised on sub_objective_derivation: "
            "%s", exc,
        )


__all__ = [
    "derive_sub_objectives_for_co",
    "populate_sub_objectives",
    "resolve_sub_objective_provider",
    "resolve_sub_objective_model",
    "resolve_max_sub_objectives",
    "resolve_sub_objective_quality",
    "ENV_SUB_OBJECTIVE_PROVIDER",
    "ENV_SUB_OBJECTIVE_MODEL",
    "ENV_SUB_OBJECTIVE_MAX",
    "ENV_SUB_OBJECTIVE_QUALITY",
]
