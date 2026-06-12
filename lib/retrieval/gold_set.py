"""Retrieval-eval gold-set loader + validator (Phase IA WS1.4).

Deterministic, read-only, stdlib + repo-internal deps only (no LLM, no
network, no DecisionCapture — this whole workstream is deterministic).

A gold set lives at ``LibV2/courses/<slug>/retrieval_eval/gold_set.json`` and
is the curated standard the retrieval evals score against. Canonical shape:
``schemas/retrieval/gold_set.schema.json``. The loader provides three layers
of checking, all surfaced as :class:`GoldSetIssue` records (mirroring the
``GateIssue`` shape used by ``lib/validators/chunkset_manifest.py`` but kept
dependency-light — no ``MCP.hardening`` import so the gold-set contract stays
usable from the WS2 embedding-eval harness without pulling the MCP surface):

  1. **Structural / schema** — validates against the canonical JSON Schema via
     ``jsonschema`` when importable, with a structural fallback otherwise
     (same graceful-degrade arm as ``chunkset_manifest.py``).
  2. **Chunkset pin** (``verify=True``) — recomputes the SHA-256 of the pinned
     ``chunks.jsonl`` and hard-errors on a mismatch
     (``GOLD_SET_CHUNKSET_SHA_MISMATCH``). The WS2 eval harness consumes this
     same contract: an eval must refuse to run against a corpus that has
     drifted from the bytes the gold set was authored against.
  3. **Answer resolution** (``verify=True``) — every ``relevant_passages[].
     chunk_id`` must exist in the pinned chunkset
     (``GOLD_SET_UNKNOWN_CHUNK_ID``) and every ``anchor.text_quote`` must be
     normalized-contained in its cited chunk's text
     (``GOLD_SET_QUOTE_NOT_IN_CHUNK``). Question ids must be unique
     (``GOLD_SET_DUPLICATE_QUESTION_ID``).

Warning-severity issues (do not fail the load, surface for authoring hygiene):

  * ``GOLD_SET_TYPE_IMBALANCE`` — any question_type < 10% of questions; fires
    only once the set is at scale (``len(questions) >= 30``) so a 10-question
    seed set isn't flagged for an imbalance that's expected pre-scale-up.
  * ``GOLD_SET_AMBIGUOUS_QUOTE`` — a text_quote that is normalized-contained
    in more than 3 chunks of the pinned chunkset; a quote that matches many
    chunks inflates retrieval credit (the sweep/eval match on text_quote), so
    authoring should pick distinctive sentences.

Text normalization is shared with ``citation_anchor`` via
``lib.retrieval._text`` so the gold-set loader, the citation anchor, and the
chunk-sweep scorer all measure containment identically.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.retrieval._text import normalize_ws
from lib.utils import sha256_file

# ---------------------------------------------------------------- constants

# Course-dir-relative subdir for the gold set (master plan § WS1.4; kept
# separate from the SLM-eval ``eval/`` dir).
RETRIEVAL_EVAL_SUBDIR = "retrieval_eval"
GOLD_SET_FILENAME = "gold_set.json"

# Supported gold-set schema versions. The loader validates a doc against the
# schema whose ``const`` schema_version matches the doc's declared version
# (dual-version acceptance — v1.0 const-pinned/frozen, v1.1 additive).
_SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")
_DEFAULT_SCHEMA_VERSION = "1.0"

# Schema filename per version (relative to schemas/retrieval/).
_SCHEMA_FILENAME_BY_VERSION = {
    "1.0": "gold_set.schema.json",
    "1.1": "gold_set.schema.v1_1.json",
}

# Question-type taxonomy. v1.0 = the first three; v1.1 adds procedural +
# multi_part. The structural fallback accepts the union so a v1.1 doc isn't
# spuriously rejected when jsonschema is absent (the per-version schema files
# are the authoritative enum when jsonschema is importable).
_QUESTION_TYPES_V1_0 = ("factual_recall", "conceptual_synthesis", "where_covered")
_QUESTION_TYPES_V1_1 = (
    "factual_recall",
    "procedural",
    "conceptual_synthesis",
    "multi_part",
    "where_covered",
)
_QUESTION_TYPES = _QUESTION_TYPES_V1_1  # union — used by the type-imbalance + fallback

# Warning thresholds.
_TYPE_IMBALANCE_MIN_QUESTIONS = 30   # only flag imbalance at scale
_TYPE_IMBALANCE_FRACTION = 0.10      # any type < 10% of questions
_AMBIGUOUS_QUOTE_MAX_CHUNKS = 3      # quote in > 3 chunks -> ambiguous

# 64-char lowercase hex (mirrors the schema regex; duplicated for the
# structural-fallback path when jsonschema isn't installed).
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QUESTION_ID_RE = re.compile(r"^gq-[a-z0-9-]+-\d{4}$")


# ---------------------------------------------------------------- issue type


@dataclass(frozen=True)
class GoldSetIssue:
    """A single gold-set validation finding.

    Mirrors ``MCP.hardening.validation_gates.GateIssue`` (code/severity/
    message) but is dependency-light so the gold-set contract is importable
    from the WS2 eval harness without the MCP surface.
    """

    code: str
    severity: str  # "critical" | "warning"
    message: str
    question_id: Optional[str] = None


# Canonical issue-code set (documented here so consumers can enumerate them).
GOLD_SET_ISSUE_CODES = frozenset(
    {
        # critical
        "GOLD_SET_NOT_FOUND",
        "GOLD_SET_INVALID_JSON",
        "GOLD_SET_SCHEMA_VIOLATION",
        "GOLD_SET_CHUNKSET_SHA_MISMATCH",
        "GOLD_SET_CHUNKSET_NOT_FOUND",
        "GOLD_SET_UNKNOWN_CHUNK_ID",
        "GOLD_SET_QUOTE_NOT_IN_CHUNK",
        "GOLD_SET_CONTENT_SHA_MISMATCH",
        "GOLD_SET_DUPLICATE_QUESTION_ID",
        # warning
        "GOLD_SET_TYPE_IMBALANCE",
        "GOLD_SET_AMBIGUOUS_QUOTE",
        "GOLD_QUESTION_METADATA_INCOMPLETE",
    }
)

# The v1.1 metadata fields a fully-authored answerable gold question carries.
# Missing any of these surfaces a warning (never a critical) so a partially
# authored / legacy-seed set still loads + evals: the frozen v1.1 set has a
# handful of seed questions missing difficulty / expected_citation_population,
# and flipping this critical would break the staleness / eval path. The backfill
# proposal (libv2 gold-metadata-backfill) is the remediation surface.
_METADATA_REQUIRED_FIELDS = (
    "question_type",
    "difficulty",
    "expected_citation_population",
)

_CRITICAL_CODES = frozenset(
    {
        "GOLD_SET_NOT_FOUND",
        "GOLD_SET_INVALID_JSON",
        "GOLD_SET_SCHEMA_VIOLATION",
        "GOLD_SET_CHUNKSET_SHA_MISMATCH",
        "GOLD_SET_CHUNKSET_NOT_FOUND",
        "GOLD_SET_UNKNOWN_CHUNK_ID",
        "GOLD_SET_QUOTE_NOT_IN_CHUNK",
        "GOLD_SET_CONTENT_SHA_MISMATCH",
        "GOLD_SET_DUPLICATE_QUESTION_ID",
    }
)


# ---------------------------------------------------------------- schema load


def doc_schema_version(gold: Dict[str, Any]) -> str:
    """Return the doc's declared schema_version, defaulting to the v1.0 base.

    Unknown versions fall through to the v1.0 schema so an out-of-range
    version surfaces as a const-mismatch GOLD_SET_SCHEMA_VIOLATION rather than
    silently selecting no schema.
    """
    declared = gold.get("schema_version") if isinstance(gold, dict) else None
    if isinstance(declared, str) and declared in _SUPPORTED_SCHEMA_VERSIONS:
        return declared
    return _DEFAULT_SCHEMA_VERSION


def chunk_content_sha256(chunk: Dict[str, Any]) -> str:
    """Canonical content hash for a chunk's normalized text.

    ``sha256(normalize_ws(chunk["text"]).lower())`` — the same normalization
    the quote-containment check uses, so the v1.1 ``anchor.content_sha256``
    re-resolution key and the text_quote fallback measure identically. Stable
    across a pure chunk-id renumber (the union-corpus rebuild case) because it
    keys on content, not position.
    """
    normalized = normalize_ws(chunk.get("text", "")).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _resolve_schema_path(version: str = _DEFAULT_SCHEMA_VERSION) -> Optional[Path]:
    """Locate the version-matched gold-set schema by walking up."""
    filename = _SCHEMA_FILENAME_BY_VERSION.get(version, _SCHEMA_FILENAME_BY_VERSION[_DEFAULT_SCHEMA_VERSION])
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "schemas" / "retrieval" / filename
        if candidate.exists():
            return candidate
    return None


def _validate_against_schema(gold: Dict[str, Any]) -> List[GoldSetIssue]:
    """Schema-validate the doc against the schema matching its schema_version.
    jsonschema when importable; structural fallback otherwise (graceful-
    degrade, mirrors chunkset_manifest.py)."""
    version = doc_schema_version(gold)
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return _structural_fallback(gold)

    schema_path = _resolve_schema_path(version)
    if not schema_path:
        return _structural_fallback(gold)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _structural_fallback(gold)

    issues: List[GoldSetIssue] = []
    validator = jsonschema.Draft202012Validator(schema)
    for err in validator.iter_errors(gold):
        location = ".".join(str(p) for p in err.absolute_path) or "<root>"
        # Best-effort question_id attribution: if the error path runs
        # through questions[i], pull that question's id.
        qid = _question_id_from_error_path(gold, list(err.absolute_path))
        issues.append(
            GoldSetIssue(
                code="GOLD_SET_SCHEMA_VIOLATION",
                severity="critical",
                message=f"schema check at {location}: {err.message}",
                question_id=qid,
            )
        )
    return issues


def _question_id_from_error_path(
    gold: Dict[str, Any], path: List[Any]
) -> Optional[str]:
    if len(path) >= 2 and path[0] == "questions" and isinstance(path[1], int):
        try:
            q = gold["questions"][path[1]]
            if isinstance(q, dict):
                qid = q.get("question_id")
                return qid if isinstance(qid, str) else None
        except (KeyError, IndexError, TypeError):
            return None
    return None


def _structural_fallback(gold: Dict[str, Any]) -> List[GoldSetIssue]:
    """Lightweight structural check covering the schema's load-bearing
    clauses for the jsonschema-absent path."""
    issues: List[GoldSetIssue] = []

    if not isinstance(gold, dict):
        return [
            GoldSetIssue(
                code="GOLD_SET_SCHEMA_VIOLATION",
                severity="critical",
                message=f"gold set root is not a JSON object (got {type(gold).__name__}).",
            )
        ]

    if gold.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        issues.append(
            GoldSetIssue(
                code="GOLD_SET_SCHEMA_VIOLATION",
                severity="critical",
                message=(
                    f"schema_version must be one of {_SUPPORTED_SCHEMA_VERSIONS}; "
                    f"got {gold.get('schema_version')!r}."
                ),
            )
        )
    if not isinstance(gold.get("course_slug"), str) or not gold.get("course_slug"):
        issues.append(
            GoldSetIssue(
                code="GOLD_SET_SCHEMA_VIOLATION",
                severity="critical",
                message="course_slug must be a non-empty string.",
            )
        )

    chunkset = gold.get("chunkset")
    if not isinstance(chunkset, dict):
        issues.append(
            GoldSetIssue(
                code="GOLD_SET_SCHEMA_VIOLATION",
                severity="critical",
                message="chunkset pin must be an object with kind/chunks_path/chunks_sha256.",
            )
        )
    else:
        if chunkset.get("kind") not in ("dart", "imscc", "corpus"):
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message=f"chunkset.kind must be one of dart/imscc/corpus; got {chunkset.get('kind')!r}.",
                )
            )
        if not isinstance(chunkset.get("chunks_path"), str) or not chunkset.get("chunks_path"):
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message="chunkset.chunks_path must be a non-empty string.",
                )
            )
        sha = chunkset.get("chunks_sha256")
        if not (isinstance(sha, str) and _SHA256_RE.match(sha)):
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message=f"chunkset.chunks_sha256 must be 64-char lowercase hex; got {sha!r}.",
                )
            )

    if not isinstance(gold.get("frozen"), bool):
        issues.append(
            GoldSetIssue(
                code="GOLD_SET_SCHEMA_VIOLATION",
                severity="critical",
                message="frozen must be a boolean.",
            )
        )

    questions = gold.get("questions")
    if not isinstance(questions, list) or not questions:
        issues.append(
            GoldSetIssue(
                code="GOLD_SET_SCHEMA_VIOLATION",
                severity="critical",
                message="questions must be a non-empty array.",
            )
        )
        return issues

    for q in questions:
        if not isinstance(q, dict):
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message="each question must be an object.",
                )
            )
            continue
        qid = q.get("question_id")
        if not (isinstance(qid, str) and _QUESTION_ID_RE.match(qid)):
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message=f"question_id must match ^gq-[a-z0-9-]+-\\d{{4}}$; got {qid!r}.",
                    question_id=qid if isinstance(qid, str) else None,
                )
            )
        if not (isinstance(q.get("question_text"), str) and len(q.get("question_text", "")) >= 10):
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message="question_text must be a string of >= 10 chars.",
                    question_id=qid if isinstance(qid, str) else None,
                )
            )
        if q.get("question_type") not in _QUESTION_TYPES:
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message=f"question_type must be one of {_QUESTION_TYPES}; got {q.get('question_type')!r}.",
                    question_id=qid if isinstance(qid, str) else None,
                )
            )
        # v1.1: multi_part questions must carry a parts[] block.
        if q.get("question_type") == "multi_part":
            parts = q.get("parts")
            if not isinstance(parts, list) or len(parts) < 2:
                issues.append(
                    GoldSetIssue(
                        code="GOLD_SET_SCHEMA_VIOLATION",
                        severity="critical",
                        message="multi_part question must carry a parts[] array of >= 2 parts.",
                        question_id=qid if isinstance(qid, str) else None,
                    )
                )
        passages = q.get("relevant_passages")
        if not isinstance(passages, list) or not passages:
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message="relevant_passages must be a non-empty array.",
                    question_id=qid if isinstance(qid, str) else None,
                )
            )
            continue
        if not any(
            isinstance(p, dict) and p.get("relevance") == "primary" for p in passages
        ):
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_SCHEMA_VIOLATION",
                    severity="critical",
                    message="at least one relevant_passage must have relevance:primary.",
                    question_id=qid if isinstance(qid, str) else None,
                )
            )
        for p in passages:
            if not isinstance(p, dict):
                continue
            anchor = p.get("anchor")
            if not isinstance(anchor, dict):
                issues.append(
                    GoldSetIssue(
                        code="GOLD_SET_SCHEMA_VIOLATION",
                        severity="critical",
                        message="passage anchor must be an object.",
                        question_id=qid if isinstance(qid, str) else None,
                    )
                )
                continue
            tq = anchor.get("text_quote")
            if not (isinstance(tq, str) and len(tq) >= 40):
                issues.append(
                    GoldSetIssue(
                        code="GOLD_SET_SCHEMA_VIOLATION",
                        severity="critical",
                        message="anchor.text_quote must be a string of >= 40 chars.",
                        question_id=qid if isinstance(qid, str) else None,
                    )
                )
    return issues


# ---------------------------------------------------------------- chunk index


def _load_chunks_by_id(chunks_path: Path) -> Dict[str, Dict[str, Any]]:
    """Index a chunks.jsonl by chunk id. Last write wins on duplicate ids
    (the validator does not police chunkset internal dup-ids; that is the
    chunkset manifest's job)."""
    chunks: Dict[str, Dict[str, Any]] = {}
    with chunks_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            cid = rec.get("id")
            if isinstance(cid, str):
                chunks[cid] = rec
    return chunks


def _quote_in_chunk(text_quote: str, chunk: Dict[str, Any]) -> bool:
    """text_quote is a verbatim chunk substring; verify normalized
    containment (whitespace-collapsed, case-insensitive). Shared
    normalization with citation_anchor via lib.retrieval._text."""
    chunk_text = normalize_ws(chunk.get("text", "")).lower()
    return normalize_ws(text_quote).lower() in chunk_text


def _quote_chunk_match_count(
    text_quote: str, chunks_by_id: Dict[str, Dict[str, Any]]
) -> int:
    """How many chunks in the pinned set contain this normalized quote."""
    needle = normalize_ws(text_quote).lower()
    if not needle:
        return 0
    return sum(
        1
        for c in chunks_by_id.values()
        if needle in normalize_ws(c.get("text", "")).lower()
    )


# ---------------------------------------------------------------- validators


def validate_gold_set(
    gold: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    chunks_sha256: str,
) -> List[GoldSetIssue]:
    """Validate a loaded gold-set doc against an in-memory chunkset index.

    Runs schema validation, then (given the pinned chunkset's recomputed
    SHA + chunk index) the pin / chunk-id / quote-containment / dup-id
    checks plus the two warning checks. Pure function — no I/O — so the
    WS2 eval harness can re-validate against an index it already holds.
    """
    issues: List[GoldSetIssue] = []
    issues.extend(_validate_against_schema(gold))

    # Pin check: the gold set's declared sha must match the chunkset bytes
    # the caller recomputed.
    declared = (gold.get("chunkset") or {}).get("chunks_sha256")
    if isinstance(declared, str) and isinstance(chunks_sha256, str):
        if declared != chunks_sha256:
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_CHUNKSET_SHA_MISMATCH",
                    severity="critical",
                    message=(
                        f"pinned chunks_sha256 ({declared[:16]}...) does not match "
                        f"on-disk chunkset ({chunks_sha256[:16]}...). The chunkset "
                        f"has drifted from the bytes this gold set was authored "
                        f"against; re-author or re-pin before running an eval."
                    ),
                )
            )

    questions = gold.get("questions")
    if not isinstance(questions, list):
        return issues

    # Duplicate-id check.
    seen_ids: Dict[str, int] = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id")
        if isinstance(qid, str):
            seen_ids[qid] = seen_ids.get(qid, 0) + 1
    for qid, count in seen_ids.items():
        if count > 1:
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_DUPLICATE_QUESTION_ID",
                    severity="critical",
                    message=f"question_id {qid!r} appears {count} times; ids must be unique.",
                    question_id=qid,
                )
            )

    # Per-passage chunk-id existence + quote containment + ambiguity.
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id") if isinstance(q.get("question_id"), str) else None
        for p in q.get("relevant_passages", []) or []:
            if not isinstance(p, dict):
                continue
            cid = p.get("chunk_id")
            anchor = p.get("anchor") or {}
            text_quote = anchor.get("text_quote")
            if not isinstance(cid, str) or cid not in chunks_by_id:
                issues.append(
                    GoldSetIssue(
                        code="GOLD_SET_UNKNOWN_CHUNK_ID",
                        severity="critical",
                        message=f"chunk_id {cid!r} not present in the pinned chunkset.",
                        question_id=qid,
                    )
                )
                # Can't check quote containment without the chunk.
                continue
            # v1.1: when the anchor carries content_sha256, it must match the
            # cited chunk's recomputed content hash. A mismatch means the
            # chunk_id resolves but its CONTENT drifted from authoring — exactly
            # what a stale pin after a re-chunk looks like at the chunk level.
            declared_csha = anchor.get("content_sha256")
            if isinstance(declared_csha, str) and declared_csha:
                actual_csha = chunk_content_sha256(chunks_by_id[cid])
                if declared_csha != actual_csha:
                    issues.append(
                        GoldSetIssue(
                            code="GOLD_SET_CONTENT_SHA_MISMATCH",
                            severity="critical",
                            message=(
                                f"anchor.content_sha256 ({declared_csha[:16]}...) does not "
                                f"match chunk {cid!r}'s recomputed content hash "
                                f"({actual_csha[:16]}...); re-pin with `libv2 gold-repin`."
                            ),
                            question_id=qid,
                        )
                    )
            if isinstance(text_quote, str) and text_quote:
                if not _quote_in_chunk(text_quote, chunks_by_id[cid]):
                    issues.append(
                        GoldSetIssue(
                            code="GOLD_SET_QUOTE_NOT_IN_CHUNK",
                            severity="critical",
                            message=(
                                f"text_quote (normalized) not contained in chunk "
                                f"{cid!r}: {text_quote[:60]!r}..."
                            ),
                            question_id=qid,
                        )
                    )
                else:
                    match_count = _quote_chunk_match_count(text_quote, chunks_by_id)
                    if match_count > _AMBIGUOUS_QUOTE_MAX_CHUNKS:
                        issues.append(
                            GoldSetIssue(
                                code="GOLD_SET_AMBIGUOUS_QUOTE",
                                severity="warning",
                                message=(
                                    f"text_quote is contained in {match_count} chunks "
                                    f"(> {_AMBIGUOUS_QUOTE_MAX_CHUNKS}); pick a more "
                                    f"distinctive sentence to avoid eval credit "
                                    f"inflation. quote: {text_quote[:60]!r}..."
                                ),
                                question_id=qid,
                            )
                        )

    # Type-imbalance warning (at scale only).
    issues.extend(_type_imbalance_issues(questions))

    # Metadata-completeness warnings (per-question; never fail the load). The
    # difficulty / expected_citation_population / expected_key_points fields are
    # v1.1 additions, so a v1.0 doc (which has no such fields by schema) is not
    # flagged — only a v1.1 doc is expected to carry them.
    if doc_schema_version(gold) == "1.1":
        issues.extend(_metadata_completeness_issues(questions))

    return issues


def _is_refusal_question(q: Dict[str, Any]) -> bool:
    """A gold question is treated as a refusal-style entry (exempt from the
    expected_key_points completeness check) when it is explicitly flagged
    unanswerable. Gold sets are answerable by construction, but a future
    deliberately-uncovered entry may carry such a marker — so the check is
    forward-compatible rather than assuming every question is answerable."""
    if q.get("is_refusal") is True or q.get("unanswerable") is True:
        return True
    cat = q.get("category")
    return isinstance(cat, str) and cat in (
        "off_topic", "off_topic_llm", "adjacent_domain", "out_of_scope_detail",
    )


def _metadata_completeness_issues(questions: List[Any]) -> List[GoldSetIssue]:
    """Warn on questions missing v1.1 metadata.

    Required for every question: question_type, difficulty,
    expected_citation_population. Required for non-refusal questions:
    expected_key_points. Each missing-field set emits ONE
    GOLD_QUESTION_METADATA_INCOMPLETE warning naming the absent fields, so a
    backfill pass (or operator) can see exactly what to fill. Warning-severity
    by contract — a partially-authored / legacy-seed set still loads + evals.
    """
    issues: List[GoldSetIssue] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id") if isinstance(q.get("question_id"), str) else None
        missing: List[str] = []
        for field_name in _METADATA_REQUIRED_FIELDS:
            value = q.get(field_name)
            if not (isinstance(value, str) and value.strip()):
                missing.append(field_name)
        if not _is_refusal_question(q):
            kps = q.get("expected_key_points")
            if not (isinstance(kps, list) and len(kps) >= 2):
                missing.append("expected_key_points")
        if missing:
            issues.append(
                GoldSetIssue(
                    code="GOLD_QUESTION_METADATA_INCOMPLETE",
                    severity="warning",
                    message=(
                        f"question is missing v1.1 metadata field(s): "
                        f"{', '.join(missing)}. Backfill before freeze "
                        f"(libv2 gold-metadata-backfill proposes difficulty / "
                        f"expected_citation_population)."
                    ),
                    question_id=qid,
                )
            )
    return issues


def _type_imbalance_issues(questions: List[Any]) -> List[GoldSetIssue]:
    n = len(questions)
    if n < _TYPE_IMBALANCE_MIN_QUESTIONS:
        return []
    counts = {t: 0 for t in _QUESTION_TYPES}
    for q in questions:
        if isinstance(q, dict):
            qt = q.get("question_type")
            if qt in counts:
                counts[qt] += 1
    issues: List[GoldSetIssue] = []
    floor = _TYPE_IMBALANCE_FRACTION * n
    for qt, c in counts.items():
        if c < floor:
            issues.append(
                GoldSetIssue(
                    code="GOLD_SET_TYPE_IMBALANCE",
                    severity="warning",
                    message=(
                        f"question_type {qt!r} has {c}/{n} questions "
                        f"(< {_TYPE_IMBALANCE_FRACTION:.0%}); the gold set is "
                        f"type-imbalanced at scale."
                    ),
                )
            )
    return issues


# ---------------------------------------------------------------- loader


def _resolve_chunkset_path(course_dir: Path, gold: Dict[str, Any]) -> Path:
    """Resolve the pinned chunks.jsonl path (course-dir-relative)."""
    rel = (gold.get("chunkset") or {}).get("chunks_path") or ""
    return course_dir / rel


def load_gold_set(
    course_dir: Path, *, verify: bool = True
) -> Tuple[Dict[str, Any], List[GoldSetIssue]]:
    """Load ``<course_dir>/retrieval_eval/gold_set.json``.

    When ``verify=True`` (default) also opens the pinned chunkset, recomputes
    its SHA-256, indexes chunk ids, and runs the pin / chunk-id / quote
    checks. When ``verify=False`` only the file load + schema validation run
    (no chunkset I/O) — useful for a fast structural lint.

    Returns ``(gold_doc, issues)``. ``gold_doc`` is ``{}`` only when the file
    is missing or unparseable (the corresponding critical issue is emitted).
    Callers fail closed on any issue whose code is in
    :data:`_CRITICAL_CODES` (helper :func:`has_critical_issues`).
    """
    course_dir = Path(course_dir)
    gold_path = course_dir / RETRIEVAL_EVAL_SUBDIR / GOLD_SET_FILENAME

    if not gold_path.exists() or not gold_path.is_file():
        return {}, [
            GoldSetIssue(
                code="GOLD_SET_NOT_FOUND",
                severity="critical",
                message=f"gold set not found at {gold_path}.",
            )
        ]

    try:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [
            GoldSetIssue(
                code="GOLD_SET_INVALID_JSON",
                severity="critical",
                message=f"gold set failed to parse: {exc.__class__.__name__}: {exc}",
            )
        ]

    if not verify:
        return gold, _validate_against_schema(gold)

    # Schema-validate first; if the chunkset pin itself is malformed the
    # downstream chunkset I/O would be noise — but we still attempt it when
    # the pin fields are present and well-shaped.
    schema_issues = _validate_against_schema(gold)

    chunks_path = _resolve_chunkset_path(course_dir, gold)
    if not chunks_path.exists() or not chunks_path.is_file():
        return gold, schema_issues + [
            GoldSetIssue(
                code="GOLD_SET_CHUNKSET_NOT_FOUND",
                severity="critical",
                message=(
                    f"pinned chunkset not found at {chunks_path}; cannot verify "
                    f"chunk_ids / quotes / sha pin."
                ),
            )
        ]

    try:
        actual_sha = sha256_file(chunks_path)
        chunks_by_id = _load_chunks_by_id(chunks_path)
    except (OSError, json.JSONDecodeError) as exc:
        return gold, schema_issues + [
            GoldSetIssue(
                code="GOLD_SET_CHUNKSET_NOT_FOUND",
                severity="critical",
                message=(
                    f"failed to read/parse pinned chunkset at {chunks_path}: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            )
        ]

    # validate_gold_set re-runs schema validation; dedupe by skipping the
    # schema pass here and merging (validate_gold_set is the full pass).
    issues = validate_gold_set(gold, chunks_by_id, actual_sha)
    return gold, issues


def has_critical_issues(issues: List[GoldSetIssue]) -> bool:
    """True iff any issue is critical-severity (or carries a critical code)."""
    return any(
        i.severity == "critical" or i.code in _CRITICAL_CODES for i in issues
    )


def critical_issues(issues: List[GoldSetIssue]) -> List[GoldSetIssue]:
    return [
        i for i in issues if i.severity == "critical" or i.code in _CRITICAL_CODES
    ]


__all__ = [
    "GoldSetIssue",
    "GOLD_SET_ISSUE_CODES",
    "RETRIEVAL_EVAL_SUBDIR",
    "GOLD_SET_FILENAME",
    "load_gold_set",
    "validate_gold_set",
    "has_critical_issues",
    "critical_issues",
    "doc_schema_version",
    "chunk_content_sha256",
    "_quote_in_chunk",
    "_quote_chunk_match_count",
    "_load_chunks_by_id",
]
