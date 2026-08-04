"""
LibV2 Storage Integration for Ed4All

Provides a unified storage interface for all Ed4All components to interact
with the LibV2 RAG library. This consolidates:
- IMSCC package outputs from Courseforge
- Training captures/Claude decisions from all tools
- RAG corpus chunks from Trainforge
- Source materials (textbooks, objectives)
"""

import fcntl
import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import paths from centralized module
from .paths import (
    LIBV2_CATALOG,
    LIBV2_COURSES,
    LIBV2_ONTOLOGY,
    LIBV2_PATH,
    LIBV2_SCHEMA,
    libv2_path,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
LIBV2_ROOT = LIBV2_PATH


class LibV2StorageError(Exception):
    """Raised when LibV2 storage operations fail."""
    pass


# ===== Phase 7c back-compat shim =====
#
# Phase 7c renamed ``LibV2/courses/<slug>/corpus/`` to
# ``LibV2/courses/<slug>/imscc_chunks/`` so the directory name reflects the
# IMSCC-derived chunkset (symmetric to the new ``dart_chunks/`` directory
# emitted by the Phase 7b chunker). All NEW writes target ``imscc_chunks/``
# (canonical path); reads attempt ``imscc_chunks/`` first and fall back to
# ``corpus/`` with a deprecation warning so unprovisioned LibV2 archives
# keep working through one migration cycle. The shim is dropped in Phase 8
# once the staged-chunkset backfill command has migrated all archives.

IMSCC_CHUNKS_DIRNAME = "imscc_chunks"
# ``semantik_chunks/`` is the ACTIVE emit dir. ``dart_chunks/`` is a PURE
# legacy READ fallback for archives produced before the SemantiK rename (and
# for un-migrated on-disk courses; the operator backfill physically renames
# them). A DeprecationWarning on the ``dart_chunks/`` fallback (matching the
# ``corpus/`` precedent below) is deferred — flipping it on would warn on every
# legacy course; it can land once corpora migrate.
LEGACY_CHUNKS_DIRNAME = "dart_chunks"
SEMANTIK_CHUNKS_DIRNAME = "semantik_chunks"
LEGACY_CORPUS_DIRNAME = "corpus"

# Canonical chunkset-dir-name <-> manifest ``chunkset_kind`` discriminator.
# Single source of truth (lib/ is the lower layer): ``vector_index.py`` imports
# this rather than redeclaring the mapping, so the manifest's ``chunkset_kind``
# and the on-disk directory name can never drift apart between the build path
# (vector_index) and the query path (resolve_chunks_path_for_query below).
DIRNAME_TO_CHUNKSET_KIND = {
    IMSCC_CHUNKS_DIRNAME: "imscc",
    SEMANTIK_CHUNKS_DIRNAME: "semantik",
    LEGACY_CHUNKS_DIRNAME: "dart",
    LEGACY_CORPUS_DIRNAME: "corpus-legacy",
}
CHUNKSET_KIND_TO_DIRNAME = {v: k for k, v in DIRNAME_TO_CHUNKSET_KIND.items()}

# Vector-index directory + manifest filename. Plain string constants live here
# (the numpy-free lower layer) so the pure-lexical / engine="auto" code paths
# can test for index presence without importing the numpy-laden
# ``vector_index`` module. ``vector_index.py`` re-exports these as its public
# ``VECTOR_INDEX_DIRNAME`` / ``MANIFEST_FILENAME`` (single source of truth).
VECTOR_INDEX_DIRNAME = "vector_index"
VECTOR_INDEX_MANIFEST_FILENAME = "manifest.json"


def has_vector_index(course_dir: Union[str, Path]) -> bool:
    """Does ``<course_dir>/vector_index/manifest.json`` exist?

    The honest ``engine="auto"`` resolution signal: a built semantic index
    leaves a ``manifest.json`` provenance file behind. A cheap stat — no load,
    no numpy, no torch — so a deps-slim operator box can take the lexical path
    without importing the embedding stack.
    """
    manifest = Path(course_dir) / VECTOR_INDEX_DIRNAME / VECTOR_INDEX_MANIFEST_FILENAME
    return manifest.is_file()


def resolve_auto_engine(engine: Optional[str], course_dir: Union[str, Path]) -> str:
    """Resolve a requested retrieval engine to a concrete one.

    ``"auto"`` (or an empty/None value) resolves to ``"hybrid-rrf"`` when the
    course has a vector index, else ``"lexical"``. hybrid-rrf — BM25 fused with
    semantic via reciprocal-rank fusion — is the benchmark-selected engine:
    pure semantic never beat the BM25 baseline, so ``auto`` routes through the
    fused engine rather than the semantic arm alone whenever an index exists.

    Any explicit engine passes through verbatim. That is load-bearing: an
    explicit ``semantic`` against a missing index must surface as a typed error
    from the pipeline (``SemanticIndexMissing``), never a silent downgrade to
    BM25 — the anti-silent-degradation contract.

    This is the SINGLE resolver for the ``auto`` seam. Both entry points that
    expose it — the ``libv2 answer-grounded`` CLI and the GUI learner ask
    service — call it, because two independent copies of these three lines had
    already drifted to different engines behind the same flag name.
    """
    # Strip BEFORE defaulting: a whitespace-only string is truthy, so
    # ``(engine or "auto").strip()`` would yield "" and pass an empty engine
    # straight through to the pipeline. Blank in any form means auto.
    requested = (engine or "").strip().lower()
    if requested and requested != "auto":
        return requested
    return "hybrid-rrf" if has_vector_index(course_dir) else "lexical"


def resolve_imscc_chunks_dir(
    course_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> Path:
    """Return the chunkset directory for a LibV2 course.

    Resolution order (precedence):

    1. ``<course_dir>/imscc_chunks`` — canonical IMSCC-derived chunkset.
    2. ``<course_dir>/semantik_chunks`` — ratified DART->semantik purge name
       for the staged chunkset. As of Stage 3c this is the ACTIVE emit dir
       (``_run_dart_chunking`` writes it), so NEW conversions resolve here
       (resolving here does NOT warn).
    3. ``<course_dir>/dart_chunks`` — legacy pre-Stage-3c staged chunkset. A
       first-class READ fallback for courses produced by the DART -> chunking
       path before Stage 3c (resolving here does NOT warn — the deprecation
       warn on this fallback is deferred; see the LEGACY_CHUNKS_DIRNAME note).
    4. ``<course_dir>/corpus`` — legacy pre-Phase-7c name; resolving here
       emits a DeprecationWarning naming the migration script.

    When ``filename`` is provided, prefer the first candidate directory
    that actually CONTAINS that file. This is what makes retrieval work
    for archives where a scaffold ``imscc_chunks/`` directory exists
    (created at provisioning time) but holds no ``chunks.jsonl`` — the
    real chunks live in ``dart_chunks/`` or legacy ``corpus/``. The
    plain directory-existence resolution (``filename=None``) is preserved
    for callers that only need the directory itself.

    When no candidate matches, returns the canonical ``imscc_chunks/``
    path so callers receive a clean ``FileNotFoundError`` on subsequent
    reads.

    Use this for any read operation that targets the chunkset directory
    itself (e.g. ``chunks.jsonl``, ``chunks.json``, ``corpus_stats.json``,
    ``.teaching_role_checkpoint.jsonl``).

    Legacy ``corpus/`` fallback (retained): kept until every archive has
    migrated off the legacy ``corpus/`` layout. Production call sites and
    on-disk courses still depend on it (tier-2 test discovery routes
    through this resolver), so it is load-bearing — do not remove it on a
    phase schedule.
    """
    course_dir = Path(course_dir)
    canonical = course_dir / IMSCC_CHUNKS_DIRNAME
    semantik = course_dir / SEMANTIK_CHUNKS_DIRNAME
    dart = course_dir / LEGACY_CHUNKS_DIRNAME
    legacy = course_dir / LEGACY_CORPUS_DIRNAME

    def _present(candidate: Path) -> bool:
        # File-aware mode: a directory only "counts" when it holds the
        # requested file. Directory mode: bare directory existence.
        if filename is not None:
            return (candidate / filename).exists()
        return candidate.exists()

    if _present(canonical):
        return canonical
    # Dual-READ: PREFER the ratified ``semantik_chunks/`` name (the ACTIVE emit
    # dir), then fall back to the legacy ``dart_chunks/`` for legacy archives
    # (no deprecation warn — deferred; see the LEGACY_CHUNKS_DIRNAME note).
    # Legacy archives have no semantik_chunks/ dir, so they still resolve to
    # dart_chunks/ exactly as before.
    if _present(semantik):
        return semantik
    if _present(dart):
        return dart
    if _present(legacy):
        warnings.warn(
            f"Phase 7c deprecation: {course_dir.name} still uses "
            f"{LEGACY_CORPUS_DIRNAME}/; run "
            "LibV2/tools/libv2/scripts/ops/backfill_legacy_chunks.py to migrate.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return canonical


def resolve_imscc_chunks_path(
    course_dir: Union[str, Path],
    filename: str = "chunks.jsonl",
) -> Path:
    """Return the path to a file inside the resolved chunkset dir.

    Convenience wrapper over :func:`resolve_imscc_chunks_dir` for the
    common case of reading ``chunks.jsonl`` (or ``chunks.json``). The
    filename is threaded into the resolver so the chosen directory is the
    one that actually contains the requested file — canonical
    ``imscc_chunks/`` first, then ``dart_chunks/``, then legacy
    ``corpus/`` (deprecation-warned).

    Legacy ``corpus/`` fallback (retained): kept until every archive has
    migrated off the legacy ``corpus/`` layout. Production call sites and
    on-disk courses still depend on it (tier-2 test discovery routes
    through this resolver), so it is load-bearing — do not remove it on a
    phase schedule.
    """
    return resolve_imscc_chunks_dir(course_dir, filename=filename) / filename


def resolve_staged_chunks_dir(
    course_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> Path:
    """Resolve the STAGED (DART-lineage) chunkset directory.

    Precedence: ``semantik_chunks/`` -> ``dart_chunks/`` -> legacy ``corpus/``.

    Unlike :func:`resolve_imscc_chunks_dir`, this does NOT prefer an
    ``imscc_chunks/`` sibling — callers here specifically want the chunkset the
    ``chunking`` workflow phase emitted from staged DART/SemantiK HTML (via
    ``MCP/tools/pipeline_tools.py::_run_dart_chunking``), which a
    dual-chunkset course (one that ALSO ran IMSCC packaging) carries alongside
    an unrelated ``imscc_chunks/``.

    ``semantik_chunks/`` is the ACTIVE emit dir; ``dart_chunks/`` is the legacy
    read fallback for legacy archives.
    When ``filename`` is given, prefer the first candidate dir that actually
    contains it (mirrors :func:`resolve_imscc_chunks_dir`). When no candidate
    matches, returns the canonical ``semantik_chunks/`` path so a subsequent
    read raises a clean ``FileNotFoundError`` naming the ratified location.
    """
    course_dir = Path(course_dir)

    def _present(candidate: Path) -> bool:
        if filename is not None:
            return (candidate / filename).exists()
        return candidate.exists()

    for name in (
        SEMANTIK_CHUNKS_DIRNAME,
        LEGACY_CHUNKS_DIRNAME,
        LEGACY_CORPUS_DIRNAME,
    ):
        candidate = course_dir / name
        if _present(candidate):
            return candidate
    return course_dir / SEMANTIK_CHUNKS_DIRNAME


def resolve_staged_chunks_path(
    course_dir: Union[str, Path],
    filename: str = "chunks.jsonl",
) -> Path:
    """Path to ``filename`` inside the resolved staged (DART-lineage) chunkset.

    Convenience wrapper over :func:`resolve_staged_chunks_dir` for the common
    ``chunks.jsonl`` read. Threads ``filename`` into the resolver so the chosen
    directory is the one that actually contains it (``semantik_chunks/`` first,
    then ``dart_chunks/``, then legacy ``corpus/``).
    """
    return resolve_staged_chunks_dir(course_dir, filename=filename) / filename


def resolve_chunks_path_for_query(
    course_dir: Union[str, Path],
    filename: str = "chunks.jsonl",
) -> tuple[Path, str]:
    """Resolve the chunkset file the live vector index was built over.

    Returns ``(chunks_path, resolution)`` where ``resolution`` is one of
    ``"manifest"`` (the path came from the vector-index manifest's
    ``chunkset_kind``) or ``"precedence"`` (the path came from the
    precedence resolver :func:`resolve_imscc_chunks_path`).

    Rationale (retrieval-stack fix): a course can carry several chunksets at
    once — e.g. an ``imscc_chunks/`` chunkset AND a ``corpus/`` union that is
    a superset of it. The on-device vector index is built over exactly ONE of
    them and records which in ``vector_index/manifest.json::chunkset_kind``.
    The QUERY path (semantic hydration + BM25 streaming) must read back the
    SAME chunkset, or every id the index has that the precedence-resolved file
    lacks is silently DROPPED at hydration (a union/corpus-legacy index becomes
    unserveable — its dart-range hits vanish). The directory-precedence
    resolver (imscc_chunks -> dart_chunks -> corpus) is the wrong oracle here;
    the manifest is authoritative.

    Resolution order:

    1. If ``vector_index/manifest.json`` exists and carries a known
       ``chunkset_kind``, map it to the chunkset dir
       (``imscc`` -> ``imscc_chunks/``, ``dart`` -> ``dart_chunks/``,
       ``corpus-legacy`` -> ``corpus/``) and return that file with
       ``resolution="manifest"`` — even if the precedence resolver would have
       chosen a different dir.
    2. If the mapped file is MISSING (manifest points at a chunkset that no
       longer exists on disk), log a warning and fall back to the precedence
       resolver. The semantic engine's own sha check will fail closed on this
       mismatch anyway (``SemanticIndexStale``); the warning makes the
       lexical-arm fallback observable rather than silent.
    3. If no index / manifest exists (or its ``chunkset_kind`` is absent /
       unknown), fall back to the precedence resolver — so lexical-only
       courses with no vector index keep working byte-identically to today.

    Use this ONLY on the retrieval/query hydration path. Non-query callers
    (chunk-count listings, validators, training synthesis, eval) must stay on
    :func:`resolve_imscc_chunks_path` — they have no index to honor and the
    precedence semantics are correct for them.
    """
    course_dir = Path(course_dir)
    manifest_path = course_dir / "vector_index" / "manifest.json"

    if manifest_path.is_file():
        kind: Optional[str] = None
        try:
            with manifest_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                raw_kind = data.get("chunkset_kind")
                kind = str(raw_kind) if raw_kind else None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "vector-index manifest unreadable for %s (%s); falling back "
                "to chunkset precedence resolution.",
                course_dir.name,
                exc,
            )
            kind = None

        if kind and kind in CHUNKSET_KIND_TO_DIRNAME:
            mapped = course_dir / CHUNKSET_KIND_TO_DIRNAME[kind] / filename
            if mapped.exists():
                return mapped, "manifest"
            logger.warning(
                "vector-index manifest for %s pins chunkset_kind=%r "
                "(-> %s) but %s is missing; falling back to chunkset "
                "precedence resolution. The semantic engine will fail its "
                "sha check on this mismatch.",
                course_dir.name,
                kind,
                CHUNKSET_KIND_TO_DIRNAME[kind],
                mapped,
            )

    return resolve_imscc_chunks_path(course_dir, filename=filename), "precedence"


class LibV2Storage:
    """
    Unified storage interface for LibV2.

    Provides path management and directory creation for:
    - Catalog entries (packages, training captures, assessments)
    - Course content (corpus, sources)

    Usage:
        storage = LibV2Storage("TST_908")
        storage.ensure_directories()

        # Get paths
        package_path = storage.get_package_output_path("20260102_143000")
        training_path = storage.get_training_capture_path("courseforge", "content-generator")
        chunks_path = storage.get_chunks_path()
    """

    def __init__(
        self,
        course_id: str,
        course_slug: Optional[str] = None,
        auto_create: bool = False,
        libv2_root: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize LibV2 storage for a course.

        Args:
            course_id: Course identifier (e.g., "TST_908")
            course_slug: URL-friendly slug (defaults to lowercase course_id with hyphens)
            auto_create: If True, create directories immediately
            libv2_root: Explicit LibV2 root override. Precedence (high → low):
                explicit ``libv2_root`` kwarg > ``ED4ALL_LIBV2_ROOT`` env var
                > ``PROJECT_ROOT / "LibV2"`` default. Resolved at construction
                time (not import time) via :func:`lib.paths.libv2_path` so a
                test that threads a tmp root — or sets the env var — never
                creates skeleton dirs in the real in-tree ``LibV2/``.
        """
        if not course_id:
            raise LibV2StorageError("course_id is required")

        self.course_id = course_id
        self.course_slug = course_slug or self._generate_slug(course_id)

        # Resolve the LibV2 root with the documented precedence:
        # explicit kwarg > ED4ALL_LIBV2_ROOT > PROJECT_ROOT/LibV2 default.
        if libv2_root:
            self.libv2_root = Path(libv2_root)
        else:
            self.libv2_root = libv2_path()
        catalog_root = self.libv2_root / "catalog"
        courses_root = self.libv2_root / "courses"

        # Catalog paths (per-course metadata, packages, training captures)
        self.catalog_path = catalog_root / course_id
        self.packages_path = self.catalog_path / "packages"
        self.training_path = self.catalog_path / "training"
        self.assessments_path = self.catalog_path / "assessments"
        self.metadata_path = self.catalog_path / "metadata.json"

        # Course content paths (imscc_chunks, sources)
        # Phase 7c rename: corpus/ -> imscc_chunks/. New writes target
        # ``imscc_chunks_path``. Reads via the ``corpus_path`` legacy
        # attribute resolve dynamically (see the property below) so a
        # legacy archive with only ``corpus/`` still resolves correctly.
        self.course_path = courses_root / self.course_slug
        self.imscc_chunks_path = self.course_path / IMSCC_CHUNKS_DIRNAME
        self.sources_path = self.course_path / "sources"
        self.concept_graph_path = self.course_path / "concept_graph"
        self.manifest_path = self.course_path / "manifest.json"

        if auto_create:
            self.ensure_directories()

    @property
    def corpus_path(self) -> Path:
        """Phase 7c back-compat alias. Resolves dynamically.

        Returns ``imscc_chunks/`` when present (canonical post-Phase 7c),
        or legacy ``corpus/`` (with deprecation warning) for unprovisioned
        archives, or the new path when neither exists.
        """
        return resolve_imscc_chunks_dir(self.course_path)

    @staticmethod
    def _generate_slug(course_id: str) -> str:
        """Generate URL-friendly slug from course ID."""
        return course_id.lower().replace("_", "-").replace(" ", "-")

    def ensure_directories(self) -> None:
        """Create all required directories for this course."""
        directories = [
            # Catalog structure
            self.packages_path,
            self.training_path / "courseforge",
            self.training_path / "trainforge",
            self.training_path / "semantik",
            self.assessments_path,
            # Course content structure
            self.imscc_chunks_path,
            self.sources_path / "textbooks",
            self.sources_path / "objectives",
            self.sources_path / "supplements",
            self.concept_graph_path,
        ]

        for path in directories:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory {path}: {e}")
                raise LibV2StorageError(f"Failed to create directory: {path}") from e

    # ===== Package Management =====

    def get_package_output_path(self, version: Optional[str] = None) -> Path:
        """
        Get path for IMSCC package output.

        Args:
            version: Version string (e.g., "20260102_143000") or None for "latest"

        Returns:
            Path to the package file
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.packages_path / f"{self.course_id}_{version}.imscc"

    def get_latest_package_path(self) -> Optional[Path]:
        """Get path to the most recent package, if any."""
        if not self.packages_path.exists():
            return None

        packages = sorted(
            self.packages_path.glob(f"{self.course_id}_*.imscc"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return packages[0] if packages else None

    def list_packages(self) -> List[Dict[str, Any]]:
        """List all packages with metadata."""
        if not self.packages_path.exists():
            return []

        packages = []
        for pkg in self.packages_path.glob(f"{self.course_id}_*.imscc"):
            stat = pkg.stat()
            packages.append({
                "path": str(pkg),
                "filename": pkg.name,
                "version": pkg.stem.replace(f"{self.course_id}_", ""),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return sorted(packages, key=lambda x: x["modified"], reverse=True)

    # ===== Training Capture Management =====

    def get_training_capture_path(self, tool: str, phase: Optional[str]) -> Path:
        """
        Get path for decision capture files.

        Args:
            tool: Tool name (courseforge, trainforge, dart)
            phase: Phase name (e.g., "content-generator", "question-generation").
                ``None`` is permitted by the canonical decision-event schema for
                tool-level captures that aren't scoped to a single phase (e.g.
                the orchestrator's phase_start emits before any phase has been
                selected); routes to ``phase_unknown/`` so the directory shape
                stays consistent.

        Returns:
            Path to the phase directory (created if needed)
        """
        # Normalize phase name (ensure hyphen-separated). ``phase=None`` is a
        # valid schema value — route to ``phase_unknown`` rather than crashing
        # on the ``.replace`` call.
        normalized_phase = phase.replace("_", "-") if phase else "unknown"

        phase_dir = self.training_path / tool / f"phase_{normalized_phase}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir

    def get_capture_file_path(
        self,
        tool: str,
        phase: str,
        session_id: Optional[str] = None
    ) -> Path:
        """
        Get path for a specific capture file.

        Args:
            tool: Tool name
            phase: Phase name
            session_id: Optional session ID (defaults to timestamp)

        Returns:
            Path to the JSONL capture file
        """
        phase_dir = self.get_training_capture_path(tool, phase)
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return phase_dir / f"decisions_{session_id}.jsonl"

    def list_training_captures(self, tool: Optional[str] = None) -> Dict[str, List[Path]]:
        """
        List all training capture files.

        Args:
            tool: Filter by tool name, or None for all tools

        Returns:
            Dict mapping tool/phase to list of capture file paths
        """
        captures = {}
        tools = [tool] if tool else ["courseforge", "trainforge", "semantik", "dart"]

        for t in tools:
            tool_path = self.training_path / t
            if not tool_path.exists():
                continue

            for phase_dir in tool_path.iterdir():
                if phase_dir.is_dir() and phase_dir.name.startswith("phase_"):
                    key = f"{t}/{phase_dir.name}"
                    captures[key] = list(phase_dir.glob("*.jsonl"))

        return captures

    # ===== RAG Corpus Management =====

    def get_chunks_path(self) -> Path:
        """Get path to ``chunks.jsonl`` for the IMSCC chunkset.

        Reads via the Phase 7c shim — returns the legacy ``corpus/``
        location with a deprecation warning if the archive hasn't been
        migrated yet, otherwise returns the new ``imscc_chunks/`` path.
        """
        return resolve_imscc_chunks_path(self.course_path)

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the corpus."""
        chunks_path = self.get_chunks_path()
        if not chunks_path.exists():
            return {"exists": False, "chunk_count": 0, "size_bytes": 0}

        chunk_count = 0
        with open(chunks_path, encoding='utf-8') as f:
            for _ in f:
                chunk_count += 1

        return {
            "exists": True,
            "chunk_count": chunk_count,
            "size_bytes": chunks_path.stat().st_size,
            "modified": datetime.fromtimestamp(chunks_path.stat().st_mtime).isoformat(),
        }

    def append_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Append a chunk to the IMSCC chunkset.

        Writes to the canonical ``imscc_chunks/chunks.jsonl`` path.

        Args:
            chunk: Chunk data with id, chunk_type, text, source, etc.
        """
        # Always write to the canonical (new) location even if the legacy
        # corpus/ directory exists alongside — back-compat is read-only.
        chunks_path = self.imscc_chunks_path / "chunks.jsonl"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        with open(chunks_path, 'a', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(chunk) + '\n')
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # ===== Source Material Management =====

    def get_textbooks_path(self) -> Path:
        """Get path for textbook storage."""
        return self.sources_path / "textbooks"

    def get_objectives_path(self) -> Path:
        """Get path for learning objectives."""
        return self.sources_path / "objectives"

    def get_supplements_path(self) -> Path:
        """Get path for supplementary materials."""
        return self.sources_path / "supplements"

    def list_sources(self) -> Dict[str, List[str]]:
        """List all source materials."""
        sources = {}
        for category in ["textbooks", "objectives", "supplements"]:
            cat_path = self.sources_path / category
            if cat_path.exists():
                sources[category] = [f.name for f in cat_path.iterdir() if f.is_file()]
            else:
                sources[category] = []
        return sources

    # ===== Assessment Management =====

    def get_assessments_path(self) -> Path:
        """Get path for generated assessments."""
        return self.assessments_path

    def save_assessment(self, assessment: Dict[str, Any], assessment_id: str) -> Path:
        """
        Save an assessment to storage.

        Args:
            assessment: Assessment data
            assessment_id: Unique assessment identifier

        Returns:
            Path to saved assessment file
        """
        self.assessments_path.mkdir(parents=True, exist_ok=True)
        assessment_path = self.assessments_path / f"{assessment_id}.json"

        with open(assessment_path, 'w', encoding='utf-8') as f:
            json.dump(assessment, f, indent=2)

        return assessment_path

    # ===== Metadata Management =====

    def get_metadata(self) -> Dict[str, Any]:
        """Load course metadata from catalog."""
        if self.metadata_path.exists():
            with open(self.metadata_path, encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save course metadata to catalog."""
        self.catalog_path.mkdir(parents=True, exist_ok=True)

        # Merge with existing metadata
        existing = self.get_metadata()
        existing.update(metadata)
        existing["updated_at"] = datetime.now().isoformat()

        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2)

    def get_manifest(self) -> Dict[str, Any]:
        """Load course manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save course manifest.

        W0.7: written via temp-file + ``os.replace`` so a crash (or a
        concurrent reader / validator) mid-write never observes a truncated,
        unparseable ``manifest.json``.
        """
        self.course_path.mkdir(parents=True, exist_ok=True)

        manifest["updated_at"] = datetime.now().isoformat()

        tmp_path = self.manifest_path.with_name(
            f"{self.manifest_path.name}.{os.getpid()}.tmp"
        )
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, self.manifest_path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    # ===== Utility Methods =====

    def exists(self) -> bool:
        """Check if this course exists in LibV2."""
        return self.catalog_path.exists() or self.course_path.exists()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of this course's storage."""
        return {
            "course_id": self.course_id,
            "course_slug": self.course_slug,
            "catalog_exists": self.catalog_path.exists(),
            "course_exists": self.course_path.exists(),
            "packages": self.list_packages(),
            "corpus": self.get_corpus_stats(),
            "sources": self.list_sources(),
            "training_captures": {
                k: len(v) for k, v in self.list_training_captures().items()
            },
        }

    def __repr__(self) -> str:
        return f"LibV2Storage(course_id='{self.course_id}', course_slug='{self.course_slug}')"


# ===== Convenience Functions =====

def get_course_storage(course_id: str, auto_create: bool = True) -> LibV2Storage:
    """
    Get LibV2Storage instance for a course.

    Args:
        course_id: Course identifier
        auto_create: Whether to create directories automatically

    Returns:
        Configured LibV2Storage instance
    """
    storage = LibV2Storage(course_id, auto_create=auto_create)
    return storage


def list_all_courses() -> List[Dict[str, Any]]:
    """List all courses in LibV2 catalog."""
    courses = []

    catalog_root = libv2_path() / "catalog"
    if not catalog_root.exists():
        return courses

    for course_dir in catalog_root.iterdir():
        if course_dir.is_dir() and not course_dir.name.startswith("."):
            storage = LibV2Storage(course_dir.name)
            courses.append({
                "course_id": course_dir.name,
                "catalog_path": str(course_dir),
                "has_packages": storage.packages_path.exists() and any(storage.packages_path.iterdir()),
                "has_training": storage.training_path.exists(),
            })

    return courses


def validate_libv2_structure() -> Dict[str, bool]:
    """Validate that LibV2 directory structure exists.

    Resolves the LibV2 root via :func:`lib.paths.libv2_path` so an
    ``ED4ALL_LIBV2_ROOT`` override is honored at call time.
    """
    root = libv2_path()
    return {
        "LIBV2_ROOT": root.exists(),
        "LIBV2_CATALOG": (root / "catalog").exists(),
        "LIBV2_COURSES": (root / "courses").exists(),
        "LIBV2_ONTOLOGY": LIBV2_ONTOLOGY.exists(),
        "LIBV2_SCHEMA": LIBV2_SCHEMA.exists(),
    }


OBJECTIVES_ARCHIVE_FILENAME = "objectives.json"


def project_objectives_for_archive(synthesized: Dict[str, Any]) -> Dict[str, Any]:
    """Project a ``synthesized_objectives.json`` dict to the canonical
    LibV2-archive ``objectives.json`` (Worker-A) shape.

    The archive's ``objectives.json`` is the PRIMARY objectives source for
    :class:`lib.validators.libv2.packet_integrity.PacketIntegrityValidator`
    (``_load_objectives`` reads ``archive_root / "objectives.json"`` before
    falling back to ``course.json``). That validator's ``co_has_parent``
    rule requires every component objective to carry a non-empty
    ``parent_terminal`` resolving to an existing terminal — a back-pointer
    the Trainforge-flattened ``course.json`` drops on disk. This projection
    restores it from the synthesized objectives' ``terminal_id`` (and its
    aliases) so the strict packet-integrity gate at ``libv2_archival`` can
    actually pass on a pipeline-built archive.

    Reuses the canonical CO→TO back-pointer resolution
    (:func:`lib.ontology.terminal_coverage.co_parent_terminal`) and the
    canonical chapter-objective flattener
    (:func:`lib.ontology.terminal_coverage.flatten_chapter_objectives`),
    so the three on-disk ``chapter_objectives`` shapes (group-of-groups,
    flat list, LibV2 ``component_objectives`` alias) all normalise here.

    Accepts both the Courseforge synthesized form
    (``terminal_objectives`` + ``chapter_objectives``) and the LibV2
    archive form (``terminal_outcomes`` + ``component_objectives``).

    Returns a dict shaped
    ``{"terminal_outcomes": [...], "component_objectives": [...]}``. The
    projection is anti-fabrication: it never invents an objective; a CO
    whose synthesized record carries no resolvable terminal back-pointer
    is emitted verbatim WITHOUT a ``parent_terminal`` (the validator then
    flags it honestly rather than us papering over a real upstream gap).
    """
    # Lazy import keeps libv2_storage importable in CLI-only environments
    # that don't pull the ontology package eagerly.
    from lib.ontology.terminal_coverage import (
        co_parent_terminal,
        flatten_chapter_objectives,
    )

    if not isinstance(synthesized, dict):
        return {"terminal_outcomes": [], "component_objectives": []}

    terminal_raw = (
        synthesized.get("terminal_objectives")
        or synthesized.get("terminal_outcomes")
        or []
    )
    terminal_outcomes: List[Dict[str, Any]] = [
        dict(t) for t in terminal_raw if isinstance(t, dict)
    ]

    chapter_raw = (
        synthesized.get("chapter_objectives")
        or synthesized.get("component_objectives")
        or synthesized.get("component_outcomes")
        or []
    )
    component_objectives: List[Dict[str, Any]] = []
    for co in flatten_chapter_objectives(chapter_raw):
        entry = dict(co)
        # ``co_parent_terminal`` lower-cases for case-insensitive matching;
        # preserve the canonical terminal id casing where possible so the
        # archived back-pointer reads naturally, falling back to the
        # lower-cased resolution otherwise.
        parent = co_parent_terminal(entry)
        if parent and not (entry.get("parent_terminal") or "").strip():
            entry["parent_terminal"] = parent
        component_objectives.append(entry)

    return {
        "terminal_outcomes": terminal_outcomes,
        "component_objectives": component_objectives,
    }


# Module-level validation
def _validate_libv2_paths():
    """Validate LibV2 paths at module load."""
    if not LIBV2_ROOT.exists():
        logger.warning(f"LibV2 root not found: {LIBV2_ROOT}")
    else:
        logger.debug(f"LibV2 root found: {LIBV2_ROOT}")

_validate_libv2_paths()


__all__ = [
    'LibV2Storage',
    'LibV2StorageError',
    'LIBV2_ROOT',
    'LIBV2_CATALOG',
    'LIBV2_COURSES',
    'LIBV2_ONTOLOGY',
    'LIBV2_SCHEMA',
    'get_course_storage',
    'list_all_courses',
    'validate_libv2_structure',
    'resolve_imscc_chunks_dir',
    'resolve_imscc_chunks_path',
    'resolve_staged_chunks_dir',
    'resolve_staged_chunks_path',
    'resolve_chunks_path_for_query',
    'SEMANTIK_CHUNKS_DIRNAME',
    'LEGACY_CHUNKS_DIRNAME',
    'DIRNAME_TO_CHUNKSET_KIND',
    'CHUNKSET_KIND_TO_DIRNAME',
    'VECTOR_INDEX_DIRNAME',
    'VECTOR_INDEX_MANIFEST_FILENAME',
    'has_vector_index',
    'resolve_auto_engine',
    'OBJECTIVES_ARCHIVE_FILENAME',
    'project_objectives_for_archive',
]
