"""Course importer for LibV2."""

import hashlib
import json
import logging
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models.course import (
    ArxivMetadata,
    Classification,
    ContentProfile,
    CourseManifest,
    SLMProcessing,
    SourceArtifact,
    SourceArtifacts,
    SourceforgeManifest,
)
from .validator import ValidationError, validate_course_strict

logger = logging.getLogger(__name__)

# Wave 70 introduced a ``LIBV2_SHACL_IMPORT_STRICT`` gate here that routed
# ``CourseManifest.to_dict()`` through the Courseforge SHACL shapes. The
# gate was non-functional: the LibV2 manifest payload carries none of the
# ed4all: @type fields that ``courseforge_v1.shacl.ttl`` targets, so the
# expanded RDF graph had zero focus nodes and every NodeShape conformed
# vacuously. Wave 72 removes the dead call site. The lower-level
# ``_shacl_validator.validate_manifest_shacl`` helper is preserved — it
# works correctly on real Courseforge JSON-LD payloads (see
# ``schemas/tests/test_courseforge_shacl_shapes.py``) and stays available
# for any future caller that feeds it the right input.


def slugify(title: str, max_length: int = 50) -> str:
    """Convert a title to a URL-safe slug."""
    # Convert to lowercase
    slug = title.lower()

    # Remove articles from the beginning
    slug = re.sub(r"^(a|an|the)\s+", "", slug)

    # Replace special characters and spaces with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)

    # Remove leading/trailing hyphens
    slug = slug.strip("-")

    # Truncate to max length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")

    return slug


def ensure_unique_slug(slug: str, courses_dir: Path) -> str:
    """Ensure the slug is unique by appending a number if needed."""
    if not (courses_dir / slug).exists():
        return slug

    counter = 2
    while (courses_dir / f"{slug}-{counter}").exists():
        counter += 1

    return f"{slug}-{counter}"


def read_sourceforge_manifest(source_dir: Path) -> dict:
    """Read the manifest.json from Sourceforge output."""
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {source_dir}")

    with open(manifest_path) as f:
        return json.load(f)


def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return f"sha256:{sha256_hash.hexdigest()}"


def load_arxiv_metadata(arxiv_id: str, db_path: Path) -> Optional[ArxivMetadata]:
    """Load arxiv metadata from the papers.db SQLite database."""
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query for the paper - try exact match first, then prefix match
        cursor.execute(
            "SELECT * FROM papers WHERE arxiv_id = ? OR arxiv_id LIKE ?",
            (arxiv_id, f"{arxiv_id}%"),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        # Parse authors (stored as comma-separated string)
        authors_str = row["authors"] or ""
        authors = [a.strip() for a in authors_str.split(",") if a.strip()]

        # Parse categories (may be comma or space separated)
        categories_str = row["categories"] or ""
        # Handle both comma and space separated categories
        categories_str = categories_str.replace(",", " ")
        categories = [c.strip() for c in categories_str.split() if c.strip()]

        # Use primary_category from DB if available, otherwise first category
        primary_cat = row["primary_category"] or (categories[0] if categories else "")

        return ArxivMetadata(
            arxiv_id=row["arxiv_id"],
            title=row["title"] or "",
            authors=authors,
            abstract=row["abstract"] or "",
            categories=categories,
            primary_category=primary_cat,
            published_date=row["published_date"] or "",
            updated_date=row["updated_date"],
        )
    except (sqlite3.Error, KeyError):
        return None


def extract_content_profile(source_dir: Path, sf_manifest: dict) -> ContentProfile:
    """Extract content profile from Sourceforge output."""
    stats = sf_manifest.get("statistics", {})

    # Try to read corpus stats for more detail
    corpus_stats_path = source_dir / "corpus" / "corpus_stats.json"
    chunk_type_dist = {}
    difficulty_dist = {}

    if corpus_stats_path.exists():
        with open(corpus_stats_path) as f:
            corpus_stats = json.load(f)
            chunk_type_dist = corpus_stats.get("chunk_type_distribution", {})
            difficulty_dist = corpus_stats.get("difficulty_distribution", {})

    # Try training specs for token count
    training_specs_path = source_dir / "training_specs" / "dataset_config.json"
    total_tokens = 0
    if training_specs_path.exists():
        with open(training_specs_path) as f:
            training_specs = json.load(f)
            total_tokens = training_specs.get("statistics", {}).get("total_tokens", 0)

    return ContentProfile(
        total_chunks=stats.get("chunks", 0),
        total_tokens=total_tokens,
        total_concepts=stats.get("concepts", 0),
        language="en",
        difficulty_distribution=difficulty_dist,
        chunk_type_distribution=chunk_type_dist,
    )


def import_course(
    source_dir: Path,
    repo_root: Path,
    division: str,
    domain: str,
    subdomains: Optional[list[str]] = None,
    topics: Optional[list[str]] = None,
    secondary_domains: Optional[list[str]] = None,
    force: bool = False,
    imscc_path: Optional[Path] = None,
    slm_version: Optional[str] = None,
    slm_specialists: Optional[list[str]] = None,
    pdf_path: Optional[Path] = None,
    html_path: Optional[Path] = None,
    arxiv_id: Optional[str] = None,
    arxiv_db_path: Optional[Path] = None,
    strict_validation: bool = True,
) -> str:
    """
    Import a course from Sourceforge output into LibV2.

    Args:
        source_dir: Path to Sourceforge output directory
        repo_root: Path to LibV2 repository root
        division: STEM or ARTS
        domain: Primary domain
        subdomains: Optional list of subdomains
        topics: Optional list of topics
        secondary_domains: Optional list of secondary domains
        force: Overwrite existing course if True
        imscc_path: Optional path to source IMSCC package (stored in source/imscc/)
        slm_version: Optional SLM version used for processing
        slm_specialists: Optional list of SLM specialists that processed this course
        pdf_path: Optional path to original PDF (stored in source/pdf/)
        html_path: Optional path to DART accessible HTML (stored in source/html/)
        arxiv_id: Optional arxiv ID to load metadata from arxiv database
        arxiv_db_path: Optional path to arxiv papers.db SQLite database
        strict_validation: If True, run strict validation and fail on errors

    Returns:
        The slug of the imported course

    Raises:
        ValidationError: If strict_validation is True and validation fails
    """
    source_dir = Path(source_dir)
    repo_root = Path(repo_root)
    courses_dir = repo_root / "courses"

    # Validate source directory
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Read Sourceforge manifest
    sf_manifest_data = read_sourceforge_manifest(source_dir)

    # Generate slug
    title = sf_manifest_data.get("course_title", source_dir.name)
    slug = slugify(title)

    if not force:
        slug = ensure_unique_slug(slug, courses_dir)

    target_dir = courses_dir / slug

    # Check if already exists
    if target_dir.exists():
        if force:
            shutil.rmtree(target_dir)
        else:
            raise FileExistsError(f"Course already exists: {slug}")

    # Create target directory
    target_dir.mkdir(parents=True)

    # Copy Sourceforge output
    for subdir in ["corpus", "graph", "pedagogy", "training_specs", "quality"]:
        src = source_dir / subdir
        if src.exists():
            shutil.copytree(src, target_dir / subdir)

    # Copy original course.json if exists
    course_json = source_dir / "course.json"
    if course_json.exists():
        shutil.copy(course_json, target_dir / "course.json")

    # Create source/ directory structure and copy source artifacts
    source_base_dir = target_dir / "source"
    source_artifacts = None
    source_package_name = None  # Legacy field for backwards compatibility

    pdf_artifact = None
    html_artifact = None
    imscc_artifact = None
    now_iso = datetime.now().isoformat()

    # Handle PDF source
    if pdf_path:
        pdf_path = Path(pdf_path)
        if pdf_path.exists():
            pdf_dir = source_base_dir / "pdf"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            dest_pdf = pdf_dir / pdf_path.name
            shutil.copy(pdf_path, dest_pdf)
            pdf_artifact = SourceArtifact(
                filename=pdf_path.name,
                checksum=compute_file_checksum(dest_pdf),
                file_size=dest_pdf.stat().st_size,
                added_timestamp=now_iso,
            )

    # Handle HTML source (DART accessible output)
    if html_path:
        html_path = Path(html_path)
        if html_path.exists():
            html_dir = source_base_dir / "html"
            html_dir.mkdir(parents=True, exist_ok=True)
            dest_html = html_dir / html_path.name
            shutil.copy(html_path, dest_html)
            html_artifact = SourceArtifact(
                filename=html_path.name,
                checksum=compute_file_checksum(dest_html),
                file_size=dest_html.stat().st_size,
                added_timestamp=now_iso,
            )

    # Handle IMSCC source
    if imscc_path:
        imscc_path = Path(imscc_path)
        if imscc_path.exists():
            imscc_dir = source_base_dir / "imscc"
            imscc_dir.mkdir(parents=True, exist_ok=True)
            dest_imscc = imscc_dir / imscc_path.name
            shutil.copy(imscc_path, dest_imscc)
            source_package_name = imscc_path.name  # Legacy field
            imscc_artifact = SourceArtifact(
                filename=imscc_path.name,
                checksum=compute_file_checksum(dest_imscc),
                file_size=dest_imscc.stat().st_size,
                added_timestamp=now_iso,
            )

    # Build SourceArtifacts if any sources were provided
    if pdf_artifact or html_artifact or imscc_artifact:
        source_artifacts = SourceArtifacts(
            pdf=pdf_artifact,
            html=html_artifact,
            imscc=imscc_artifact,
        )

    # Load arxiv metadata if arxiv_id provided
    arxiv_metadata = None
    if arxiv_id:
        if arxiv_db_path:
            db_path = Path(arxiv_db_path)
            arxiv_metadata = load_arxiv_metadata(arxiv_id, db_path)
        else:
            logger.warning(f"arxiv_id '{arxiv_id}' provided but no arxiv_db_path specified; skipping arxiv metadata")

    # Create quality/ directory structure if not copied from source
    quality_dir = target_dir / "quality"
    quality_dir.mkdir(exist_ok=True)
    if not (quality_dir / "quality_report.json").exists():
        # Initialize empty quality report structure (only if source had none)
        quality_report = {
            "oscqr_score": None,
            "pattern_violations": [],
            "corrections": [],
            "last_evaluated": None,
        }
        with open(quality_dir / "quality_report.json", "w") as f:
            json.dump(quality_report, f, indent=2)

    # Create Sourceforge manifest object
    sf_manifest = SourceforgeManifest(
        sourceforge_version=sf_manifest_data.get("sourceforge_version", "unknown"),
        export_timestamp=sf_manifest_data.get("export_timestamp", datetime.now().isoformat()),
        course_id=sf_manifest_data.get("course_id", slug),
        course_title=title,
    )

    # Create classification
    classification = Classification(
        division=division.upper(),
        primary_domain=domain.lower(),
        secondary_domains=[d.lower() for d in (secondary_domains or [])],
        subdomains=[s.lower() for s in (subdomains or [])],
        topics=[t.lower() for t in (topics or [])],
    )

    # Extract content profile
    content_profile = extract_content_profile(source_dir, sf_manifest_data)

    # Create SLM processing info if version provided
    slm_processing = None
    if slm_version or slm_specialists:
        slm_processing = SLMProcessing(
            slm_version=slm_version,
            processing_timestamp=datetime.now().isoformat(),
            specialists_used=slm_specialists or [],
            generation=0,  # Initial import is generation 0
        )

    # Create LibV2 manifest (initially without validation — written first so validator can find it)
    manifest = CourseManifest(
        libv2_version="1.2.0",  # Bumped for source artifacts + arxiv metadata
        slug=slug,
        import_timestamp=datetime.now().isoformat(),
        sourceforge_manifest=sf_manifest,
        classification=classification,
        content_profile=content_profile,
        quality_metadata={
            "validation_status": "pending",
            "validation_errors": [],
            "validation_warnings": [],
        },
        provenance={
            k: v for k, v in {
                "source_path": str(source_dir),
                "import_pipeline_version": "1.2.0",
                "imscc_source": str(imscc_path) if imscc_path else None,
                "pdf_source": str(pdf_path) if pdf_path else None,
                "html_source": str(html_path) if html_path else None,
            }.items() if v is not None
        },
        slm_processing=slm_processing,
        source_package=source_package_name,
        source_artifacts=source_artifacts,
        arxiv_metadata=arxiv_metadata,
    )

    # Write manifest (must exist before validation runs)
    manifest_path = target_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    # Run validation AFTER manifest is written so validator can find it
    validation_result = validate_course_strict(target_dir, repo_root)
    validation_status = "validated" if validation_result.valid else "failed"
    validation_timestamp = datetime.now().isoformat()

    if strict_validation and not validation_result.valid:
        # Cleanup and fail
        logger.error(f"Validation failed for {slug}: {validation_result.errors}")
        shutil.rmtree(target_dir)
        raise ValidationError(
            f"Import validation failed for {slug}: {len(validation_result.errors)} errors",
            errors=validation_result.errors,
        )

    # Update manifest with validation results
    manifest.quality_metadata = {
        "validation_status": validation_status,
        "last_validated": validation_timestamp,
        "validation_errors": validation_result.errors,
        "validation_warnings": validation_result.warnings,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return slug


def list_importable_courses(sourceforge_output_dir: Path) -> list[dict]:
    """List courses available for import from Sourceforge output directory."""
    output_dir = Path(sourceforge_output_dir)
    courses = []

    for item in output_dir.iterdir():
        if item.is_dir():
            manifest_path = item / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                courses.append({
                    "path": str(item),
                    "course_id": manifest.get("course_id", item.name),
                    "course_title": manifest.get("course_title", "Unknown"),
                    "chunks": manifest.get("statistics", {}).get("chunks", 0),
                })

    return courses
