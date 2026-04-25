"""Course importer for LibV2."""

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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

# Wave 70 — opt-in strict SHACL gate on the import path. Off by default
# (log a warning on non-conform); ``LIBV2_SHACL_IMPORT_STRICT=1`` promotes
# a violation to a hard ValueError so CI / regression corpora catch drift
# immediately. Mirrors the truthy-string convention used elsewhere in
# Ed4All (see schemas/ONTOLOGY.md § 12).
_SHACL_STRICT_ENV = "LIBV2_SHACL_IMPORT_STRICT"
_TRUTHY = {"1", "true", "yes", "on"}


def _shacl_strict_flag() -> bool:
    return os.environ.get(_SHACL_STRICT_ENV, "").strip().lower() in _TRUTHY


def _shacl_validate_manifest(
    manifest_dict: dict, strict: bool
) -> Tuple[bool, str]:
    """Run the Courseforge SHACL shapes against a manifest-shaped dict.

    The loader + validator live in ``_shacl_validator`` so the heavy
    pyld/pyshacl/rdflib deps are imported lazily — a bare LibV2 install
    without the RDF toolchain still gets a clean import (with an
    info-level note that validation was skipped).

    Args:
        manifest_dict: The manifest to validate (expanded via the
            Courseforge @context inside the validator).
        strict: When True, raises ``ValueError`` with the SHACL report
            on non-conform. When False, logs a warning and returns the
            report so the import proceeds.

    Returns:
        ``(conforms, report_text)``. ``conforms=True`` means the payload
        satisfied every shape. On missing deps, returns
        ``(True, "<skipped: pyld/pyshacl/rdflib not installed>")`` — the
        import continues, and logs capture the skip.
    """
    # Lazy import so ``libv2 import`` doesn't pay for pyld at CLI startup.
    try:
        from ._shacl_validator import ShaclDepsMissing, validate_manifest_shacl
    except ImportError as exc:  # defensive — shouldn't fire, vendored in-tree
        logger.info(
            "SHACL manifest validation skipped — vendored validator import "
            "failed: %s",
            exc,
        )
        return True, f"<skipped: validator import failed: {exc}>"

    try:
        conforms, report = validate_manifest_shacl(manifest_dict)
    except ShaclDepsMissing as exc:
        logger.info(
            "SHACL manifest validation skipped — pyld/pyshacl/rdflib not "
            "installed (%s). Install the RDF toolchain to enable the gate.",
            exc,
        )
        return True, f"<skipped: deps missing: {exc}>"
    except Exception as exc:  # pragma: no cover - defensive
        # Any unexpected SHACL failure is a warning, not a crash — we
        # must not break imports on validator bugs.
        logger.warning(
            "SHACL manifest validation errored out (%s: %s); treating as "
            "non-conform but not strict-failing.",
            type(exc).__name__,
            exc,
        )
        return False, f"<validator error: {type(exc).__name__}: {exc}>"

    if not conforms:
        if strict:
            raise ValueError(
                "LibV2 SHACL manifest validation failed (strict mode):\n"
                + report
            )
        logger.warning(
            "LibV2 SHACL manifest validation produced violations (lenient "
            "mode — import proceeds):\n%s",
            report,
        )
    return conforms, report


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


def derive_course_slug(
    course_code: Optional[str],
    course_title: Optional[str],
    fallback: Optional[str] = None,
    max_length: int = 50,
) -> str:
    """Derive a LibV2 course slug from ``course_code`` + ``course_title``.

    Bug observed (2026-04-24): ``python -m Trainforge.process_course
    --import-to-libv2`` produced ``rdf-shacl-550-rdf-shacl-550`` because
    Courseforge writes the IMSCC manifest title as ``f"{course_code}:
    {course_title}"`` and Trainforge's IMSCC parser falls back to
    ``course_code`` when the manifest carries no usable title — so the
    title round-tripped as ``"RDF_SHACL_550: RDF_SHACL_550"`` and
    ``slugify`` doubled the code.

    This helper collapses that pattern: when ``course_title`` starts with
    ``course_code`` (with optional ``:`` / whitespace separator), strip the
    prefix before slugifying so we never emit ``code-slug-code-slug``. The
    resulting slug is ``slugify(f"{course_code} {stripped_title}")`` when a
    distinct title remains, else just ``slugify(course_code)``.

    Args:
        course_code: Stable course identifier (e.g. ``"RDF_SHACL_550"``).
        course_title: Human-friendly title from the source manifest.
        fallback: Used when both code + title are empty (e.g. the source
            directory name).
        max_length: Maximum slug length.

    Returns:
        A URL-safe slug. Never returns ``""`` — falls back to ``fallback``
        (or ``"course"``) when both inputs are empty.
    """
    code = (course_code or "").strip()
    title = (course_title or "").strip()

    # Dedupe: strip any leading ``{code}`` / ``{code}:`` / ``{code} ``
    # prefix from the title before concatenating. Case-insensitive
    # because IMSCC titles are emitted with original case but downstream
    # slugify lowercases everything anyway.
    stripped_title = title
    if code and title:
        # Match the code at the start, optionally followed by ``:`` and
        # whitespace, or just whitespace. Repeat — Courseforge has been
        # observed to double-prefix in some manifests.
        prefix_re = re.compile(
            r"^\s*" + re.escape(code) + r"\s*[:\-]?\s*",
            flags=re.IGNORECASE,
        )
        prev = None
        while stripped_title and stripped_title != prev:
            prev = stripped_title
            stripped_title = prefix_re.sub("", stripped_title, count=1).strip()

    if code and stripped_title:
        # Distinct title remains — concatenate code + title for a richer
        # slug. ``slugify`` collapses adjacent separators.
        return slugify(f"{code} {stripped_title}", max_length=max_length)

    if code:
        # Title was empty or fully redundant with code.
        return slugify(code, max_length=max_length)

    if title:
        # No code provided — slug from title alone (legacy callers).
        return slugify(title, max_length=max_length)

    if fallback:
        return slugify(fallback, max_length=max_length)

    return "course"


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

    # Generate slug. Use the dedupe-aware helper so titles like
    # ``"RDF_SHACL_550: RDF_SHACL_550"`` (Courseforge IMSCC manifest +
    # Trainforge fallback) don't collapse into ``rdf-shacl-550-rdf-shacl-550``.
    title = sf_manifest_data.get("course_title", source_dir.name)
    course_code = sf_manifest_data.get("course_id") or ""
    slug = derive_course_slug(
        course_code=course_code,
        course_title=title,
        fallback=source_dir.name,
    )

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

    # Wave 75 — copy objectives.json sidecar if Trainforge emitted it.
    # Carries the full TO-/CO- hierarchy (terminal_outcomes[] +
    # component_objectives[] with parent_terminal back-pointers) so
    # downstream chunk ``learning_outcome_refs`` can resolve against
    # ALL outcomes, not just the terminal ones declared on course.json.
    # Optional: pre-Wave-75 archives don't carry one and that's still
    # valid (LibV2 retrieval keeps falling back to course.json).
    objectives_json = source_dir / "objectives.json"
    if objectives_json.exists():
        shutil.copy(objectives_json, target_dir / "objectives.json")

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
    manifest_payload = manifest.to_dict()
    with open(manifest_path, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    # Wave 70 — SHACL gate on the manifest. Strict mode (env-controlled)
    # raises; lenient mode logs and continues. Missing RDF deps skip
    # silently. Raised as ValueError so the caller can catch it distinct
    # from JSON Schema ValidationError below.
    _shacl_validate_manifest(
        manifest_payload,
        strict=_shacl_strict_flag(),
    )

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
