"""Deterministic Bloom-label harvester.

Given a Courseforge project/export path (and, optionally, a LibV2 course dir),
walk three artifact families and emit one JSONL label row per artifact-asserted
Bloom claim:

* **objectives** — ``synthesized_objectives.json`` / ``objectives.json``
  (TO + CO ``statement`` + ``bloom_level``).
* **blocks** — ``blocks_final.jsonl`` / ``blocks_outline*.jsonl`` (block text
  surface + the block's declared ``bloom_level`` / ``target_bloom``).
* **assessments** — anything question-shaped under a ``06_assessments/`` dir
  (``stem`` / ``question_text`` + ``bloom_level``), incl. the
  ``.assessments_checkpoint.jsonl`` sidecar (rows are ``assessment.to_dict()``).

No LLM, no model load, no network. Deterministic transformation: the same
inputs always yield the same rows. De-dup is by ``content_sha256`` (over the
normalized text + asserted level) both WITHIN this run and ACROSS runs (the
existing store's shas are loaded first), so re-harvesting a course is a cheap
no-op.

Contract:

* **Skip-not-fail** — a missing artifact family is COUNTED and REPORTED
  (``HarvestResult.missing_artifacts``), never raised. A structurally
  malformed row (bad JSON, non-dict) is skipped and counted
  (``malformed_skipped``).
* **Store location** — ``state/bloom_labels/labels.jsonl`` (a subdirectory
  under ``state/`` — covered by the ``.gitignore`` ``state/*/*`` catch-all, so
  the harvested corpus is never committed).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from lib.paths import STATE_PATH

logger = logging.getLogger(__name__)

#: Schema version stamped on every row (bump on a breaking row-shape change).
SCHEMA_VERSION: str = "1.0"

#: Constant license tag recorded on every harvested label.
#:
#: The labels are Bloom levels asserted by the local NVIDIA open-model seat
#: (Super-120B / Nemotron) — an inference-only, on-device generation. Per
#: ``docs/LICENSING.md`` the trained-model licensing exposure comes from the
#: SYNTHESIS provider that authors training-data pairs, not from these
#: read-only metadata reads; this tag records the generating surface so a later
#: consumer can filter by provenance. It is a documented CONSTANT, not derived.
LICENSE_TAG: str = "nvidia-open-model-local"

#: Canonical Bloom levels — mirrors :data:`lib.ontology.bloom.BLOOM_LEVELS`
#: (inlined import-light; a label whose level is not one of these is skipped).
_BLOOM_LEVELS: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)
_BLOOM_LEVEL_SET = frozenset(_BLOOM_LEVELS)

#: Default label store — a SUBDIR under ``state/`` (root-level ``state/`` files
#: are NOT covered by the ``.gitignore`` catch-all; a subdir is).
DEFAULT_STORE_PATH: Path = STATE_PATH / "bloom_labels" / "labels.jsonl"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

#: Artifact-level metadata keys that (when present) name the generating model.
_PROVENANCE_KEYS: Tuple[str, ...] = (
    "model",
    "llm_model",
    "model_id",
    "provider",
    "llm_provider",
    "generated_by",
)


@dataclass
class BloomLabel:
    """One harvested Bloom-label row (serialized as a JSONL line)."""

    text: str
    bloom_level: str
    artifact_kind: str  # objective | block | assessment_item
    source_id: str
    model_provenance: str
    license_tag: str
    content_sha256: str
    harvested_at: str  # ISO-8601 timestamp (informational; NOT a dedupe key)
    run_id: str
    schema_version: str = SCHEMA_VERSION

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HarvestResult:
    """Summary of a single :func:`harvest_bloom_labels` invocation."""

    store_path: str
    rows_added: int = 0
    duplicates_skipped: int = 0
    malformed_skipped: int = 0
    dry_run: bool = False
    per_artifact_counts: Dict[str, int] = field(default_factory=dict)
    missing_artifacts: List[str] = field(default_factory=list)
    scanned_files: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# text / hashing helpers
# ---------------------------------------------------------------------------
def _strip_html(value: str) -> str:
    """Strip HTML tags and collapse whitespace (cheap surface extractor)."""
    if not value:
        return ""
    return _WHITESPACE_RE.sub(" ", _HTML_TAG_RE.sub(" ", value)).strip()


def _normalize_text(value: Any) -> str:
    """Coerce a text surface to a normalized, whitespace-collapsed string."""
    if not isinstance(value, str):
        return ""
    return _WHITESPACE_RE.sub(" ", value).strip()


def _canonical_bloom(value: Any) -> Optional[str]:
    """Return the canonical lowercase Bloom level, or ``None`` if not canonical.

    Tolerates ``bloom_level`` (snake_case) and ``bloomLevel`` (camelCase) at the
    call site; here it only validates a single already-extracted value.
    """
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    return s if s in _BLOOM_LEVEL_SET else None


def _bloom_from(obj: Dict[str, Any], *keys: str) -> Optional[str]:
    """First canonical Bloom level found across ``keys`` (snake/camel tolerant)."""
    for key in keys:
        level = _canonical_bloom(obj.get(key))
        if level is not None:
            return level
    return None


def _content_sha256(text: str, bloom_level: str) -> str:
    """Stable dedupe key over the normalized text + asserted level.

    Same text with a DIFFERENT asserted level is a distinct label (the
    disagreement signal cares about the pairing), so the level is folded in.
    """
    payload = f"{bloom_level}\x00{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _provenance_from_doc(doc: Any, override: Optional[str]) -> str:
    """Resolve model provenance for an artifact.

    Priority: caller ``override`` > an explicit model/provider key on the
    artifact doc > ``"unknown"``. Deliberately conservative — we never guess a
    model name from a method/mint field.
    """
    if override:
        return override
    if isinstance(doc, dict):
        for key in _PROVENANCE_KEYS:
            val = doc.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return "unknown"


# ---------------------------------------------------------------------------
# artifact walkers — each yields (text, bloom_level, source_id, provenance)
# ---------------------------------------------------------------------------
def _flatten_objectives(payload: Any) -> List[Dict[str, Any]]:
    """Pull a flat list of LO dicts out of the accepted container shapes.

    Mirrors ``lib.validators.abcd_objective._flatten_objectives`` (kept inline
    to stay import-light): bare list; ``learning_outcomes``;
    ``terminal_objectives`` / ``terminal_outcomes`` +
    ``chapter_objectives`` / ``component_objectives`` (the latter may be flat LO
    dicts OR ``{"chapter": ..., "objectives": [...]}`` groups).
    """
    if isinstance(payload, list):
        return [lo for lo in payload if isinstance(lo, dict)]
    if not isinstance(payload, dict):
        return []
    flat = payload.get("learning_outcomes")
    if isinstance(flat, list):
        return [lo for lo in flat if isinstance(lo, dict)]
    out: List[Dict[str, Any]] = []
    for key in ("terminal_objectives", "terminal_outcomes"):
        block = payload.get(key)
        if isinstance(block, list):
            out.extend(lo for lo in block if isinstance(lo, dict))
    for key in ("chapter_objectives", "component_objectives"):
        block = payload.get(key)
        if not isinstance(block, list):
            continue
        for entry in block:
            if not isinstance(entry, dict):
                continue
            nested = entry.get("objectives")
            if isinstance(nested, list):
                out.extend(lo for lo in nested if isinstance(lo, dict))
            else:
                out.append(entry)
    return out


def _walk_objectives_file(
    path: Path, provenance_override: Optional[str], counters: "_Counters"
) -> Iterator[Tuple[str, str, str, str]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("bloom-labels: unreadable objectives file %s: %s", path, exc)
        counters.malformed += 1
        return
    provenance = _provenance_from_doc(doc, provenance_override)
    for idx, lo in enumerate(_flatten_objectives(doc)):
        if not isinstance(lo, dict):
            counters.malformed += 1
            continue
        level = _bloom_from(lo, "bloom_level", "bloomLevel")
        if level is None:
            continue
        text = _normalize_text(lo.get("statement") or lo.get("text"))
        if not text:
            continue
        source_id = str(lo.get("id") or lo.get("objective_id") or f"obj_{idx}")
        yield text, level, source_id, provenance


def _walk_blocks_file(
    path: Path, provenance_override: Optional[str], counters: "_Counters"
) -> Iterator[Tuple[str, str, str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("bloom-labels: unreadable blocks file %s: %s", path, exc)
        counters.malformed += 1
        return
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            block = json.loads(line)
        except ValueError:
            counters.malformed += 1
            continue
        if not isinstance(block, dict):
            counters.malformed += 1
            continue
        # Declared level first (what the gate audits), then the planner's
        # target_bloom assertion as a fallback.
        level = _bloom_from(block, "bloom_level", "bloomLevel", "target_bloom")
        if level is None:
            continue
        content = block.get("content")
        if isinstance(content, dict):
            text = _normalize_text(
                content.get("statement") or content.get("text")
            )
        elif isinstance(content, str):
            text = _strip_html(content)
        else:
            text = ""
        if not text:
            continue
        source_id = str(block.get("block_id") or f"block_{idx}")
        yield text, level, source_id, provenance_override or "unknown"


_QUESTION_TEXT_KEYS: Tuple[str, ...] = ("stem", "question_text", "text", "prompt")


def _iter_question_dicts(obj: Any) -> Iterator[Dict[str, Any]]:
    """Yield every question-shaped dict reachable in ``obj``.

    Question-shaped == a dict carrying a text key AND a ``bloom_level``. Also
    descends into ``questions`` / ``items`` lists and ``assessments`` /
    ``assessment`` containers (so an ``assessment.to_dict()`` row, a bare
    ``{questions: [...]}`` doc, or a top-level list all work).
    """
    if isinstance(obj, dict):
        has_text = any(isinstance(obj.get(k), str) for k in _QUESTION_TEXT_KEYS)
        if has_text and "bloom_level" in obj:
            yield obj
        for key in ("questions", "items", "assessments"):
            child = obj.get(key)
            if isinstance(child, list):
                for entry in child:
                    yield from _iter_question_dicts(entry)
        nested = obj.get("assessment")
        if isinstance(nested, dict):
            yield from _iter_question_dicts(nested)
    elif isinstance(obj, list):
        for entry in obj:
            yield from _iter_question_dicts(entry)


def _walk_assessment_file(
    path: Path, provenance_override: Optional[str], counters: "_Counters"
) -> Iterator[Tuple[str, str, str, str]]:
    """Harvest question stems + bloom from a JSON or JSONL assessment file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("bloom-labels: unreadable assessment file %s: %s", path, exc)
        counters.malformed += 1
        return
    # Parse as a single JSON doc; fall back to line-delimited JSONL.
    docs: List[Any] = []
    try:
        docs.append(json.loads(raw))
    except ValueError:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except ValueError:
                counters.malformed += 1
    seq = 0
    for doc in docs:
        provenance = _provenance_from_doc(doc, provenance_override)
        for q in _iter_question_dicts(doc):
            level = _bloom_from(q, "bloom_level", "bloomLevel")
            if level is None:
                continue
            text = ""
            for key in _QUESTION_TEXT_KEYS:
                text = _strip_html(_normalize_text(q.get(key)))
                if text:
                    break
            if not text:
                continue
            source_id = str(q.get("question_id") or q.get("id") or f"q_{seq}")
            seq += 1
            yield text, level, source_id, provenance


# ---------------------------------------------------------------------------
# file discovery
# ---------------------------------------------------------------------------
_OBJECTIVE_NAMES = {"synthesized_objectives.json", "objectives.json"}


def _find_files(roots: Iterable[Path]) -> Dict[str, List[Path]]:
    """Discover the three artifact families under the candidate roots.

    Returns ``{kind: [paths]}`` with paths sorted + de-duplicated so a run is
    deterministic regardless of filesystem walk order.
    """
    objectives: List[Path] = []
    blocks: List[Path] = []
    assessments: List[Path] = []
    seen: set = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = list(root.rglob("*"))
        for p in candidates:
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            name = p.name
            if name in _OBJECTIVE_NAMES:
                objectives.append(p)
                seen.add(rp)
            elif name.endswith(".jsonl") and (
                "blocks_final" in name or "blocks_outline" in name
            ):
                blocks.append(p)
                seen.add(rp)
            elif "06_assessments" in p.parts and (
                name.endswith(".json") or name.endswith(".jsonl")
            ):
                assessments.append(p)
                seen.add(rp)
            elif name == ".assessments_checkpoint.jsonl":
                assessments.append(p)
                seen.add(rp)
    return {
        "objective": sorted(objectives),
        "block": sorted(blocks),
        "assessment_item": sorted(assessments),
    }


# ---------------------------------------------------------------------------
# store I/O
# ---------------------------------------------------------------------------
def _load_existing_shas(store_path: Path) -> set:
    """Load ``content_sha256`` values already in the store (cross-run dedupe)."""
    shas: set = set()
    if not store_path.exists():
        return shas
    try:
        for line in store_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a corrupt existing line is ignored, not counted
            sha = row.get("content_sha256") if isinstance(row, dict) else None
            if isinstance(sha, str):
                shas.add(sha)
    except OSError as exc:
        logger.warning("bloom-labels: unreadable store %s: %s", store_path, exc)
    return shas


@dataclass
class _Counters:
    malformed: int = 0


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def harvest_bloom_labels(
    project_path: str | os.PathLike[str],
    *,
    course_path: str | os.PathLike[str] | None = None,
    store_path: str | os.PathLike[str] | None = None,
    run_id: Optional[str] = None,
    model_provenance: Optional[str] = None,
    dry_run: bool = False,
) -> HarvestResult:
    """Harvest artifact-asserted Bloom labels into the JSONL store.

    Args:
        project_path: A Courseforge project/export dir (or a single artifact
            file). Walked recursively for objectives / blocks / assessments.
        course_path: Optional additional LibV2 course dir (archive-form
            objectives, assessments).
        store_path: Override the default ``state/bloom_labels/labels.jsonl``.
        run_id: Provenance run id stamped on each row; defaults to
            ``ED4ALL_RUN_ID`` or ``"unknown"``.
        model_provenance: Force the ``model_provenance`` field (else resolved
            from artifact metadata, else ``"unknown"``).
        dry_run: Compute + report counts but write NOTHING to the store.

    Returns:
        A :class:`HarvestResult` with rows added / duplicates skipped /
        malformed skipped / per-artifact counts / missing-artifact families.
    """
    project = Path(project_path)
    roots: List[Path] = [project]
    if course_path:
        roots.append(Path(course_path))

    store = Path(store_path) if store_path else DEFAULT_STORE_PATH
    resolved_run_id = run_id or os.environ.get("ED4ALL_RUN_ID") or "unknown"
    harvested_at = datetime.now(timezone.utc).isoformat()

    result = HarvestResult(store_path=str(store), dry_run=dry_run)
    counters = _Counters()

    files = _find_files(roots)
    walkers = {
        "objective": _walk_objectives_file,
        "block": _walk_blocks_file,
        "assessment_item": _walk_assessment_file,
    }

    seen_shas = _load_existing_shas(store)
    new_rows: List[BloomLabel] = []

    for kind, walker in walkers.items():
        paths = files.get(kind, [])
        if not paths:
            result.missing_artifacts.append(kind)
            continue
        result.per_artifact_counts.setdefault(kind, 0)
        for path in paths:
            result.scanned_files.append(str(path))
            for text, level, source_id, provenance in walker(
                path, model_provenance, counters
            ):
                sha = _content_sha256(text, level)
                if sha in seen_shas:
                    result.duplicates_skipped += 1
                    continue
                seen_shas.add(sha)
                new_rows.append(
                    BloomLabel(
                        text=text,
                        bloom_level=level,
                        artifact_kind=kind,
                        source_id=source_id,
                        model_provenance=provenance,
                        license_tag=LICENSE_TAG,
                        content_sha256=sha,
                        harvested_at=harvested_at,
                        run_id=resolved_run_id,
                    )
                )
                result.per_artifact_counts[kind] += 1
                result.rows_added += 1

    result.malformed_skipped = counters.malformed

    if new_rows and not dry_run:
        store.parent.mkdir(parents=True, exist_ok=True)
        with store.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row.to_row(), ensure_ascii=False) + "\n")

    return result
