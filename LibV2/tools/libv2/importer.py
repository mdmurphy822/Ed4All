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

# Wave 93 — canonical list of top-level subdirs to mirror from a Trainforge /
# Sourceforge build into ``LibV2/courses/<slug>/``. Adding ``"models"`` here
# means trained adapters land alongside corpus / graph / training_specs as
# first-class artifacts. Keep the order stable; tests pin both membership
# and presence of ``"models"`` (Wave 93).
#
# Phase 7c: ``imscc_chunks`` replaces the legacy ``corpus`` directory. Both
# names appear in the list so the importer copies whichever the build emits
# (back-compat for unprovisioned Trainforge runs that still write to
# ``corpus/``). Phase 8 drops ``corpus``.
_COPIED_SUBDIRS: list[str] = [
    "imscc_chunks",
    "corpus",  # Phase 7c back-compat — drop in Phase 8.
    "graph",
    "pedagogy",
    "training_specs",
    "quality",
    "models",
]

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
    # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
    corpus_stats_path = source_dir / "imscc_chunks" / "corpus_stats.json"
    if not corpus_stats_path.exists():
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

    # Copy Sourceforge output. ``_COPIED_SUBDIRS`` is the canonical
    # source of truth (Wave 93) — it includes ``models`` so adapter
    # runs survive a re-import.
    for subdir in _COPIED_SUBDIRS:
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


def _validate_model_pointers(pointers: dict) -> None:
    """Validate ``_pointers.json`` payload against the Wave 93 schema.

    Best-effort: when ``jsonschema`` isn't installed, fall back to a
    structural check (top-level required keys + history entry shape).
    Raises :class:`ValueError` on failure so callers fail loud.
    """
    if not isinstance(pointers, dict):
        raise ValueError(
            f"_pointers.json must be a dict, got {type(pointers).__name__}"
        )
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "schemas" / "models" / "model_pointers.schema.json"
    )
    try:
        import jsonschema  # type: ignore
    except ImportError:  # pragma: no cover - jsonschema is in baseline deps
        # Minimal structural check
        for key in ("current", "history"):
            if key not in pointers:
                raise ValueError(f"_pointers.json missing required key: {key}")
        if not isinstance(pointers["history"], list):
            raise ValueError("_pointers.json::history must be a list")
        for entry in pointers["history"]:
            if not isinstance(entry, dict):
                raise ValueError("_pointers.json::history entries must be objects")
            for key in ("model_id", "promoted_at"):
                if key not in entry:
                    raise ValueError(
                        f"_pointers.json::history entry missing key: {key}"
                    )
        return

    if not schema_path.exists():
        # Schema isn't in this checkout — fall back to structural check
        for key in ("current", "history"):
            if key not in pointers:
                raise ValueError(f"_pointers.json missing required key: {key}")
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(pointers, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(
            f"_pointers.json failed schema check: {exc.message} "
            f"(at {'.'.join(str(p) for p in exc.absolute_path)})"
        ) from exc


def _read_pointers_file(pointers_path: Path) -> dict:
    """Read ``_pointers.json``, returning a fresh skeleton if it doesn't exist."""
    if not pointers_path.exists():
        return {"current": None, "history": []}
    with open(pointers_path) as f:
        data = json.load(f)
    # Defensive: tolerate legacy half-built files
    data.setdefault("current", None)
    data.setdefault("history", [])
    return data


def _write_pointers_file(pointers_path: Path, pointers: dict) -> None:
    """Atomic-write ``_pointers.json`` after schema validation.

    Atomicity matters because callers may inspect the file mid-promotion
    (e.g. CLI tab-completion); the tmp+rename pattern means readers
    see either the old or new payload, never a half-written one.
    """
    _validate_model_pointers(pointers)
    pointers_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointers_path.with_suffix(pointers_path.suffix + ".tmp")
    tmp.write_text(json.dumps(pointers, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(pointers_path)


def import_model(
    course_slug: str,
    run_dir: Path,
    repo_root: Optional[Path] = None,
    promote: bool = False,
    promoted_by: Optional[str] = None,
) -> Path:
    """Import a trained-adapter run dir into a LibV2 course's ``models/`` dir.

    Wave 93 — wires :class:`Trainforge.training.runner.TrainingRunner.run`'s
    output (which writes ``adapter.safetensors`` + ``model_card.json`` +
    ``eval_report.json`` + ``training_run.jsonl`` into a run dir) into
    LibV2 as a first-class artifact.

    Args:
        course_slug: The LibV2 course slug to attach the model to.
        run_dir: Path to the run dir produced by ``TrainingRunner.run()``.
            Must contain ``model_card.json``.
        repo_root: Path to LibV2 repository root. Defaults to a search
            from ``Path.cwd()`` for a directory containing ``courses/``.
        promote: If True, set the new ``model_id`` as the course's
            current model in ``_pointers.json`` and demote the previous
            current.
        promoted_by: Optional actor identifier recorded in the history
            entry when ``promote=True``.

    Returns:
        The path to the imported ``models/<model_id>/`` directory.

    Raises:
        FileNotFoundError: ``run_dir`` or ``model_card.json`` missing,
            or ``courses/<slug>/`` does not exist.
        ValidationError: The model card fails the
            :class:`LibV2ModelValidator` critical-severity check.
        FileExistsError: The model_id directory already exists in the
            target course's ``models/`` dir.
    """
    # Local import keeps the validator dep optional for lighter callers
    # that only use ``import_course``.
    from lib.validators.libv2_model import LibV2ModelValidator

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    card_path = run_dir / "model_card.json"
    if not card_path.exists():
        raise FileNotFoundError(
            f"Run dir missing model_card.json: {card_path}. "
            f"This file is emitted by Trainforge.training.runner.TrainingRunner.run()."
        )

    if repo_root is None:
        # Search upwards for a courses/ directory
        cur = Path.cwd()
        while cur != cur.parent:
            if (cur / "courses").exists():
                repo_root = cur
                break
            cur = cur.parent
        if repo_root is None:
            raise FileNotFoundError(
                "Could not locate LibV2 repo_root from cwd; pass repo_root explicitly."
            )
    repo_root = Path(repo_root)

    course_dir = repo_root / "courses" / course_slug
    if not course_dir.exists():
        raise FileNotFoundError(
            f"LibV2 course slug not found: {course_dir}. "
            f"Import the course first via ``libv2 import``."
        )

    # Read the model card to mint the target dir + populate manifest
    with open(card_path) as f:
        card = json.load(f)
    model_id = card.get("model_id")
    if not model_id:
        raise ValueError(f"Model card at {card_path} is missing 'model_id'.")

    # Validate critical-severity issues before doing any disk work
    validator = LibV2ModelValidator()
    result = validator.validate({
        "model_card_path": str(card_path),
        "model_dir": str(run_dir),
        "course_dir": str(course_dir),
    })
    critical_issues = [i for i in result.issues if i.severity == "critical"]
    if critical_issues:
        codes = [i.code for i in critical_issues]
        messages = [f"  - {i.code}: {i.message}" for i in critical_issues]
        raise ValidationError(
            f"Model card validation failed for {model_id}: {len(critical_issues)} "
            f"critical issue(s) ({', '.join(codes)}).\n" + "\n".join(messages),
            errors=[i.message for i in critical_issues],
        )

    # Stage the run dir into models/<model_id>/
    models_root = course_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    target = models_root / model_id
    if target.exists():
        raise FileExistsError(
            f"Model already exists in course: {target}. "
            f"Models are content-addressed by provenance — re-running with the "
            f"same provenance produces the same model_id; remove the existing "
            f"directory if you intend to overwrite."
        )
    shutil.copytree(run_dir, target)

    # Promotion handling: write _pointers.json (validated against the schema)
    pointers_path = models_root / "_pointers.json"
    if promote:
        now_iso = datetime.now().isoformat()
        pointers = _read_pointers_file(pointers_path)

        # Demote any currently-promoted entry whose demoted_at is None
        for entry in pointers.get("history", []):
            if entry.get("demoted_at") is None and entry.get("model_id") != model_id:
                entry["demoted_at"] = now_iso

        # Append the new entry
        pointers["history"].append({
            "model_id": model_id,
            "promoted_at": now_iso,
            "promoted_by": promoted_by,
            "demoted_at": None,
        })
        pointers["current"] = model_id
        _write_pointers_file(pointers_path, pointers)

        # Update CourseManifest.slm_processing from the new card
        _update_manifest_slm_processing(course_dir, card)

    return target


def _update_manifest_slm_processing(course_dir: Path, card: dict) -> None:
    """Refresh ``manifest.json::slm_processing`` from a model card.

    Populates the existing :class:`SLMProcessing` dataclass slot with
    fields drawn from the card:
    - ``slm_version`` ← ``base_model.name`` + ``model_id`` suffix
    - ``processing_timestamp`` ← ``card.created_at``
    - ``generation`` ← previous generation + 1 (or 0 on first import)
    - ``parent_version`` ← previous slm_version (None on first import)

    Wave 93: this is the first wave that actually writes to this slot;
    the field has been carried by the manifest since v1.2.0 but
    nothing has populated it.
    """
    manifest_path = course_dir / "manifest.json"
    if not manifest_path.exists():
        # No manifest to update — happens for some test fixtures and
        # for hand-built course skeletons. The new model still lands;
        # only the manifest mirror is skipped.
        logger.warning(
            "import_model: course %s has no manifest.json; skipping "
            "slm_processing update.", course_dir.name,
        )
        return

    with open(manifest_path) as f:
        manifest_dict = json.load(f)

    # Compute slm_version label
    base_name = (card.get("base_model") or {}).get("name") or "unknown"
    model_id = card.get("model_id") or ""
    slm_version = f"{base_name}/{model_id}"

    prev = manifest_dict.get("slm_processing") or {}
    prev_generation = int(prev.get("generation", 0)) if isinstance(prev, dict) else 0
    prev_version = prev.get("slm_version") if isinstance(prev, dict) else None

    new_slm = {
        "slm_version": slm_version,
        "processing_timestamp": card.get("created_at") or datetime.now().isoformat(),
        "specialists_used": (
            list(prev.get("specialists_used", [])) if isinstance(prev, dict) else []
        ),
        "generation": prev_generation + 1 if prev_version else 0,
        "parent_version": prev_version,
    }
    manifest_dict["slm_processing"] = new_slm

    # Atomic write
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)


def list_course_models(course_slug: str, repo_root: Path) -> dict:
    """Return a summary of all imported models for a course.

    Wave 93 — backs ``libv2 models list <slug>``. Returns a dict
    mapping ``model_id`` to its top-level card metadata + an
    ``is_current`` flag derived from ``_pointers.json::current``.
    """
    course_dir = Path(repo_root) / "courses" / course_slug
    if not course_dir.exists():
        raise FileNotFoundError(f"Course not found: {course_dir}")

    models_root = course_dir / "models"
    if not models_root.exists():
        return {"current": None, "models": []}

    pointers = _read_pointers_file(models_root / "_pointers.json")
    current = pointers.get("current")

    models: list[dict] = []
    for entry in sorted(models_root.iterdir()):
        if not entry.is_dir():
            continue
        card_path = entry / "model_card.json"
        if not card_path.exists():
            continue
        try:
            with open(card_path) as f:
                card = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping malformed model card at %s: %s", card_path, exc)
            continue
        eval_path = entry / "eval_report.json"
        eval_summary = None
        if eval_path.exists():
            try:
                with open(eval_path) as f:
                    eval_summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                eval_summary = None
        models.append({
            "model_id": card.get("model_id", entry.name),
            "is_current": card.get("model_id") == current,
            "base_model": card.get("base_model"),
            "adapter_format": card.get("adapter_format"),
            "created_at": card.get("created_at"),
            "eval_scores": (eval_summary or card.get("eval_scores") or {}),
            "path": str(entry),
        })
    return {"current": current, "models": models}


def promote_model(
    course_slug: str,
    model_id: str,
    repo_root: Path,
    promoted_by: Optional[str] = None,
) -> Path:
    """Flip ``_pointers.json::current`` to a new model_id.

    Wave 93 — backs ``libv2 models promote <slug> <model_id>``.
    Demotes the previous current (sets its ``demoted_at``) and appends
    a new history entry. No-ops + returns the pointer path when
    ``model_id`` is already current.

    Raises:
        FileNotFoundError: course or model directory missing.
    """
    course_dir = Path(repo_root) / "courses" / course_slug
    models_root = course_dir / "models"
    target = models_root / model_id
    if not target.exists():
        raise FileNotFoundError(
            f"Model not found in course {course_slug}: {target}"
        )
    if not (target / "model_card.json").exists():
        raise FileNotFoundError(
            f"Model dir {target} is missing model_card.json"
        )

    pointers_path = models_root / "_pointers.json"
    pointers = _read_pointers_file(pointers_path)

    if pointers.get("current") == model_id:
        # Idempotent — current already points here. Still ensure schema-clean.
        _write_pointers_file(pointers_path, pointers)
        return pointers_path

    now_iso = datetime.now().isoformat()
    for entry in pointers.get("history", []):
        if entry.get("demoted_at") is None and entry.get("model_id") != model_id:
            entry["demoted_at"] = now_iso
    pointers["history"].append({
        "model_id": model_id,
        "promoted_at": now_iso,
        "promoted_by": promoted_by,
        "demoted_at": None,
    })
    pointers["current"] = model_id
    _write_pointers_file(pointers_path, pointers)

    # Mirror the card into manifest.json::slm_processing
    with open(target / "model_card.json") as f:
        card = json.load(f)
    _update_manifest_slm_processing(course_dir, card)

    return pointers_path


def get_model_eval_report(
    course_slug: str,
    model_id: str,
    repo_root: Path,
) -> Optional[dict]:
    """Return the cached ``eval_report.json`` for a model, or ``None``.

    Wave 93 — backs ``libv2 models eval <slug> <model_id>``. The
    actual eval-callable bridge from a saved adapter to the eval
    harness is deferred to a follow-up wave (see Wave 92's deferred
    items in ``plans/slm-training-2026-04-26.md``); this function
    only surfaces the cached report.
    """
    course_dir = Path(repo_root) / "courses" / course_slug
    eval_path = course_dir / "models" / model_id / "eval_report.json"
    if not eval_path.exists():
        return None
    try:
        with open(eval_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read eval_report.json at %s: %s", eval_path, exc)
        return None


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
