"""
LibV2 Storage Integration for Ed4All

Provides a unified storage interface for all Ed4All components to interact
with the LibV2 RAG library. This consolidates:
- IMSCC package outputs from Courseforge
- Training captures/Claude decisions from all tools
- RAG corpus chunks from Trainforge
- Source materials (textbooks, objectives)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import paths from centralized module
from .paths import (
    LIBV2_CATALOG,
    LIBV2_COURSES,
    LIBV2_ONTOLOGY,
    LIBV2_PATH,
    LIBV2_SCHEMA,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
LIBV2_ROOT = LIBV2_PATH


class LibV2StorageError(Exception):
    """Raised when LibV2 storage operations fail."""
    pass


class LibV2Storage:
    """
    Unified storage interface for LibV2.

    Provides path management and directory creation for:
    - Catalog entries (packages, training captures, assessments)
    - Course content (corpus, sources)

    Usage:
        storage = LibV2Storage("INT_101")
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
        auto_create: bool = False
    ):
        """
        Initialize LibV2 storage for a course.

        Args:
            course_id: Course identifier (e.g., "INT_101")
            course_slug: URL-friendly slug (defaults to lowercase course_id with hyphens)
            auto_create: If True, create directories immediately
        """
        if not course_id:
            raise LibV2StorageError("course_id is required")

        self.course_id = course_id
        self.course_slug = course_slug or self._generate_slug(course_id)

        # Catalog paths (per-course metadata, packages, training captures)
        self.catalog_path = LIBV2_CATALOG / course_id
        self.packages_path = self.catalog_path / "packages"
        self.training_path = self.catalog_path / "training"
        self.assessments_path = self.catalog_path / "assessments"
        self.metadata_path = self.catalog_path / "metadata.json"

        # Course content paths (corpus, sources)
        self.course_path = LIBV2_COURSES / self.course_slug
        self.corpus_path = self.course_path / "corpus"
        self.sources_path = self.course_path / "sources"
        self.concept_graph_path = self.course_path / "concept_graph"
        self.manifest_path = self.course_path / "manifest.json"

        if auto_create:
            self.ensure_directories()

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
            self.training_path / "dart",
            self.assessments_path,
            # Course content structure
            self.corpus_path,
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

    def get_training_capture_path(self, tool: str, phase: str) -> Path:
        """
        Get path for decision capture files.

        Args:
            tool: Tool name (courseforge, trainforge, dart)
            phase: Phase name (e.g., "content-generator", "question-generation")

        Returns:
            Path to the phase directory (created if needed)
        """
        # Normalize phase name (ensure hyphen-separated)
        normalized_phase = phase.replace("_", "-")

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
        tools = [tool] if tool else ["courseforge", "trainforge", "dart"]

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
        """Get path for RAG corpus chunks."""
        return self.corpus_path / "chunks.jsonl"

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the corpus."""
        chunks_path = self.get_chunks_path()
        if not chunks_path.exists():
            return {"exists": False, "chunk_count": 0, "size_bytes": 0}

        chunk_count = 0
        with open(chunks_path, 'r') as f:
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
        Append a chunk to the corpus.

        Args:
            chunk: Chunk data with id, chunk_type, text, source, etc.
        """
        chunks_path = self.get_chunks_path()
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        with open(chunks_path, 'a') as f:
            f.write(json.dumps(chunk) + '\n')

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

        with open(assessment_path, 'w') as f:
            json.dump(assessment, f, indent=2)

        return assessment_path

    # ===== Metadata Management =====

    def get_metadata(self) -> Dict[str, Any]:
        """Load course metadata from catalog."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save course metadata to catalog."""
        self.catalog_path.mkdir(parents=True, exist_ok=True)

        # Merge with existing metadata
        existing = self.get_metadata()
        existing.update(metadata)
        existing["updated_at"] = datetime.now().isoformat()

        with open(self.metadata_path, 'w') as f:
            json.dump(existing, f, indent=2)

    def get_manifest(self) -> Dict[str, Any]:
        """Load course manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {}

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save course manifest."""
        self.course_path.mkdir(parents=True, exist_ok=True)

        manifest["updated_at"] = datetime.now().isoformat()

        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

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

    if not LIBV2_CATALOG.exists():
        return courses

    for course_dir in LIBV2_CATALOG.iterdir():
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
    """Validate that LibV2 directory structure exists."""
    return {
        "LIBV2_ROOT": LIBV2_ROOT.exists(),
        "LIBV2_CATALOG": LIBV2_CATALOG.exists(),
        "LIBV2_COURSES": LIBV2_COURSES.exists(),
        "LIBV2_ONTOLOGY": LIBV2_ONTOLOGY.exists(),
        "LIBV2_SCHEMA": LIBV2_SCHEMA.exists(),
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
]
