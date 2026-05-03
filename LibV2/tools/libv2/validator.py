"""Validation utilities for LibV2."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import jsonschema
except ImportError:
    jsonschema = None


class ValidationError(Exception):
    """Raised when validation fails in strict mode."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def _resolve_project_schemas_dir(repo_root: Path) -> Path:
    """Resolve the project-root /schemas/ directory from an arbitrary repo_root.

    Schemas and ontology previously lived under ``LibV2/schema/`` and
    ``LibV2/ontology/``; they are now unified under ``<project-root>/schemas/``.
    This helper resolves that location regardless of whether the caller
    supplied a LibV2 directory or the project root as ``repo_root``.
    """
    try:
        from lib.paths import SCHEMAS_PATH  # type: ignore
        if SCHEMAS_PATH.exists():
            return SCHEMAS_PATH
    except Exception:
        pass
    # If repo_root looks like LibV2 (has courses/), project root is its parent
    if (repo_root / "courses").exists() and (repo_root.parent / "schemas").exists():
        return repo_root.parent / "schemas"
    # Fallback: assume repo_root IS the project root
    return repo_root / "schemas"


def load_schema(repo_root: Path, schema_name: str) -> Optional[dict]:
    """Load a JSON schema from the project-root schemas/library directory."""
    schema_path = _resolve_project_schemas_dir(repo_root) / "library" / schema_name
    if schema_path.exists():
        with open(schema_path) as f:
            return json.load(f)
    return None


def validate_json_schema(data: dict, schema: dict, context: str = "") -> ValidationResult:
    """Validate data against a JSON schema."""
    result = ValidationResult(valid=True)

    if jsonschema is None:
        result.add_warning("jsonschema not installed, skipping schema validation")
        return result

    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        result.add_error(f"{context}: {e.message}")
    except jsonschema.SchemaError as e:
        result.add_error(f"{context}: Invalid schema - {e.message}")

    return result


def validate_course_structure(course_dir: Path) -> ValidationResult:
    """Validate the structure of a course directory."""
    result = ValidationResult(valid=True)

    # Required files
    required_files = [
        "manifest.json",
    ]

    # Required directories
    # Phase 7c: corpus/ renamed to imscc_chunks/. Either is accepted as
    # the IMSCC chunkset directory; legacy archives that still carry
    # corpus/ keep validating until backfill_dart_chunks.py migrates them.
    required_dirs = [
        "graph",
    ]

    # Optional directories
    optional_dirs = [
        "pedagogy",
        "training_specs",
    ]

    # Check required files
    for filename in required_files:
        if not (course_dir / filename).exists():
            result.add_error(f"Missing required file: {filename}")

    # Check required directories
    for dirname in required_dirs:
        if not (course_dir / dirname).exists():
            result.add_error(f"Missing required directory: {dirname}")
        elif not (course_dir / dirname).is_dir():
            result.add_error(f"Expected directory but found file: {dirname}")

    # Phase 7c IMSCC chunkset dir — accept either imscc_chunks/ (new) or
    # corpus/ (legacy). Missing both is the error.
    imscc_chunks_dir = course_dir / "imscc_chunks"
    legacy_corpus_dir = course_dir / "corpus"
    chunkset_dir: Optional[Path] = None
    if imscc_chunks_dir.exists() and imscc_chunks_dir.is_dir():
        chunkset_dir = imscc_chunks_dir
    elif legacy_corpus_dir.exists() and legacy_corpus_dir.is_dir():
        chunkset_dir = legacy_corpus_dir
    else:
        result.add_error("Missing required directory: imscc_chunks (or legacy corpus)")

    # Check optional directories
    for dirname in optional_dirs:
        if not (course_dir / dirname).exists():
            result.add_warning(f"Missing optional directory: {dirname}")

    # Check IMSCC chunkset contents
    if chunkset_dir is not None:
        if not (chunkset_dir / "chunks.json").exists() and not (chunkset_dir / "chunks.jsonl").exists():
            result.add_error(
                f"Missing chunks.json or chunks.jsonl in {chunkset_dir.name}/"
            )

    # Check graph contents
    graph_dir = course_dir / "graph"
    if graph_dir.exists():
        if not (graph_dir / "concept_graph.json").exists():
            result.add_warning("Missing concept_graph.json in graph/")

    return result


def validate_course_manifest(course_dir: Path, repo_root: Path) -> ValidationResult:
    """Validate the course manifest against the schema."""
    result = ValidationResult(valid=True)

    manifest_path = course_dir / "manifest.json"
    if not manifest_path.exists():
        result.add_error("manifest.json not found")
        return result

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON in manifest.json: {e}")
        return result

    # Load and validate against schema
    schema = load_schema(repo_root, "course_manifest.schema.json")
    if schema:
        schema_result = validate_json_schema(manifest, schema, "manifest.json")
        result.merge(schema_result)

    # Additional semantic validations
    if "classification" in manifest:
        classification = manifest["classification"]
        division = classification.get("division", "")
        if division not in ["STEM", "ARTS"]:
            result.add_error(f"Invalid division: {division} (must be STEM or ARTS)")

    # Validate content profile matches actual content
    if "content_profile" in manifest:
        profile = manifest["content_profile"]
        # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
        chunks_path = course_dir / "imscc_chunks" / "chunks.json"
        if not chunks_path.exists():
            chunks_path = course_dir / "corpus" / "chunks.json"
        if chunks_path.exists():
            try:
                with open(chunks_path) as f:
                    chunks = json.load(f)
                actual_count = len(chunks) if isinstance(chunks, list) else 0
                declared_count = profile.get("total_chunks", 0)
                if actual_count != declared_count:
                    result.add_warning(
                        f"Chunk count mismatch: manifest says {declared_count}, "
                        f"actual is {actual_count}"
                    )
            except json.JSONDecodeError:
                result.add_error(
                    f"Invalid JSON in {chunks_path.parent.name}/chunks.json"
                )

    return result


# Wave 76 — Domain aliases. The canonical taxonomy (schemas/taxonomies/
# taxonomy.json) keys domains in slug form (``computer-science``), but
# manifests have shipped human-readable forms (``computer science``,
# ``Computer Science``) for as long as the importer has been running.
# Rather than rewrite every existing manifest, the validator accepts a
# small set of canonical aliases per domain and matches case-insensitively.
# Additions here are append-only — never remove an alias because doing so
# retroactively invalidates archived courses.
_DOMAIN_ALIASES: dict[str, list[str]] = {
    # STEM
    "physics": ["physics"],
    "chemistry": ["chemistry"],
    "biology": ["biology"],
    "mathematics": ["mathematics", "math", "maths"],
    "computer-science": [
        "computer-science",
        "computer science",
        "computing",
        "cs",
        "software-engineering",
        "software engineering",
        "information-systems",
        "information systems",
    ],
    "engineering": ["engineering"],
    "medicine": [
        "medicine",
        "medicine and health sciences",
        "medicine & health sciences",
        "health sciences",
    ],
    "environmental-science": [
        "environmental-science",
        "environmental science",
    ],
    "data-science": ["data-science", "data science"],
    "educational-technology": [
        "educational-technology",
        "educational technology",
        "edtech",
    ],
    # ARTS (defensive — accept slug + display form for any registered domain)
    "literature": ["literature"],
    "history": ["history"],
    "philosophy": ["philosophy"],
    "music": ["music"],
    "visual-arts": ["visual-arts", "visual arts"],
    "performing-arts": ["performing-arts", "performing arts"],
    "languages": ["languages"],
    "religion": ["religion"],
}


def _domain_matches(declared: str, canonical: str) -> bool:
    """Case-insensitive match of a declared manifest domain against a
    canonical taxonomy slug, expanded via :data:`_DOMAIN_ALIASES`.

    Wave 76 — fixes a false-negative where ``"computer science"`` (space
    form, the natural-language form ed-tech tooling emits) failed against
    the slug-form ``computer-science`` taxonomy key.
    """
    if not declared:
        return False
    declared_norm = declared.strip().lower()
    canonical_norm = canonical.strip().lower()
    if declared_norm == canonical_norm:
        return True
    # Slug ↔ space variant (e.g. "computer-science" ↔ "computer science")
    if declared_norm.replace("-", " ") == canonical_norm.replace("-", " "):
        return True
    aliases = _DOMAIN_ALIASES.get(canonical_norm, [])
    return any(declared_norm == a.strip().lower() for a in aliases)


def validate_taxonomy_compliance(course_dir: Path, repo_root: Path) -> ValidationResult:
    """Validate that course classification uses valid taxonomy terms."""
    result = ValidationResult(valid=True)

    # Load taxonomy (now lives under <project-root>/schemas/taxonomies/)
    taxonomy_path = _resolve_project_schemas_dir(repo_root) / "taxonomies" / "taxonomy.json"
    if not taxonomy_path.exists():
        result.add_warning("taxonomy.json not found, skipping taxonomy validation")
        return result

    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    # Load course manifest
    manifest_path = course_dir / "manifest.json"
    if not manifest_path.exists():
        return result

    with open(manifest_path) as f:
        manifest = json.load(f)

    classification = manifest.get("classification", {})
    division = classification.get("division", "")
    domain = classification.get("primary_domain", "")

    # Validate division
    if division and division not in taxonomy.get("divisions", {}):
        result.add_error(f"Unknown division: {division}")
        return result

    # Validate domain (Wave 76 — case-insensitive, alias-aware)
    if division and domain:
        division_data = taxonomy["divisions"].get(division, {})
        domains = division_data.get("domains", {})
        if not any(_domain_matches(domain, canonical) for canonical in domains):
            result.add_error(f"Unknown domain '{domain}' in division '{division}'")

    return result


def validate_dataset_config_constraints(course_dir: Path) -> ValidationResult:
    """
    Enforce dataset_config.json constraints on chunks.

    Validates:
    - min_tokens: Chunks below threshold are errors
    - max_tokens: Chunks above threshold are errors
    - require_concepts: Empty concept_tags are errors if true
    """
    result = ValidationResult(valid=True)

    config_path = course_dir / "training_specs" / "dataset_config.json"
    # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
    chunks_path = course_dir / "imscc_chunks" / "chunks.json"
    if not chunks_path.exists():
        chunks_path = course_dir / "corpus" / "chunks.json"

    if not config_path.exists():
        result.add_warning("No dataset_config.json found, skipping constraint validation")
        return result

    if not chunks_path.exists():
        result.add_error("imscc_chunks/chunks.json (or legacy corpus/chunks.json) not found")
        return result

    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON in dataset_config.json: {e}")
        return result

    try:
        with open(chunks_path) as f:
            chunks = json.load(f)
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON in chunks.json: {e}")
        return result

    if not isinstance(chunks, list):
        result.add_error("chunks.json must be a list")
        return result

    filtering = config.get("filtering", {})
    min_tokens = filtering.get("min_tokens", 0)
    max_tokens = filtering.get("max_tokens", float("inf"))
    require_concepts = filtering.get("require_concepts", False)

    violations = {"min_tokens": 0, "max_tokens": 0, "empty_concepts": 0}

    for chunk in chunks:
        chunk_id = chunk.get("id", "unknown")
        tokens = chunk.get("tokens_estimate", 0)

        if min_tokens > 0 and tokens < min_tokens:
            violations["min_tokens"] += 1
            if violations["min_tokens"] <= 3:  # Only report first 3
                result.add_error(
                    f"Chunk {chunk_id}: {tokens} tokens < min {min_tokens}"
                )

        if tokens > max_tokens:
            violations["max_tokens"] += 1
            if violations["max_tokens"] <= 3:
                result.add_error(
                    f"Chunk {chunk_id}: {tokens} tokens > max {max_tokens}"
                )

        if require_concepts:
            concept_tags = chunk.get("concept_tags", [])
            if not concept_tags:
                violations["empty_concepts"] += 1
                if violations["empty_concepts"] <= 3:
                    result.add_error(
                        f"Chunk {chunk_id}: empty concept_tags (require_concepts=true)"
                    )

    # Summary if more violations than reported
    if violations["min_tokens"] > 3:
        result.add_error(
            f"... and {violations['min_tokens'] - 3} more min_tokens violations"
        )
    if violations["max_tokens"] > 3:
        result.add_error(
            f"... and {violations['max_tokens'] - 3} more max_tokens violations"
        )
    if violations["empty_concepts"] > 3:
        result.add_error(
            f"... and {violations['empty_concepts'] - 3} more empty concept_tags"
        )

    return result


def _count_total_learning_outcomes(course_dir: Path) -> Optional[int]:
    """Count terminal + component outcomes for the minimum-coverage rule.

    Wave 76 — Pre-Wave-76 the validator counted only ``course.json::
    learning_outcomes[]`` and rejected courses where the count fell
    below 10. Two failure modes:
      1. course.json was emitted before Wave 75's component-objective
         merge, so it held only the 7 terminal LOs even though the
         course had 36 outcomes total.
      2. course.json held the full 36 but a follow-on filter (now
         removed) dropped entries with ``type == "component"``.

    The fix counts the union of:
      - ``course.json::learning_outcomes[]`` (Wave 75+ canonical: holds
        terminal AND component LOs in one flat list).
      - As a fallback, ``objectives.json::terminal_outcomes[]`` +
        ``objectives.json::component_objectives[]`` (or the legacy
        ``terminal_objectives`` / ``chapter_objectives`` keys), which
        is what Wave 75 Worker A's emit guarantees even when course.json
        is mid-migration.

    Returns the total outcome count, or ``None`` when neither file exists
    / is parseable. Caller is responsible for raising on that condition.
    """
    ids: set[str] = set()

    course_json_path = course_dir / "course.json"
    if course_json_path.exists():
        try:
            with open(course_json_path) as f:
                course = json.load(f)
            for lo in course.get("learning_outcomes", []) or []:
                if isinstance(lo, dict):
                    lo_id = (lo.get("id") or "").strip().lower()
                    if lo_id:
                        ids.add(lo_id)
        except (json.JSONDecodeError, OSError):
            pass

    objectives_path = course_dir / "objectives.json"
    if objectives_path.exists():
        try:
            with open(objectives_path) as f:
                objectives = json.load(f)
            terminal_list = (
                objectives.get("terminal_outcomes")
                or objectives.get("terminal_objectives")
                or []
            )
            component_list = (
                objectives.get("component_objectives")
                or objectives.get("chapter_objectives")
                or []
            )
            for to in terminal_list or []:
                if isinstance(to, dict):
                    lo_id = (to.get("id") or "").strip().lower()
                    if lo_id:
                        ids.add(lo_id)
            for ch in component_list or []:
                if not isinstance(ch, dict):
                    continue
                # Both nested chapters and flat component objectives.
                if "objectives" in ch and isinstance(ch.get("objectives"), list):
                    inner = ch["objectives"]
                else:
                    inner = [ch]
                for obj in inner:
                    if isinstance(obj, dict):
                        lo_id = (obj.get("id") or "").strip().lower()
                        if lo_id:
                            ids.add(lo_id)
        except (json.JSONDecodeError, OSError):
            pass

    if not ids and not course_json_path.exists():
        return None
    return len(ids)


def validate_learning_outcomes(course_dir: Path) -> ValidationResult:
    """
    Validate learning outcome coverage.

    Requirements:
    - Minimum 10 course-level learning outcomes (terminal + component
      combined, Wave 76).
    - Warning above 60 (Wave 76 — bumped from 25 because Wave-75 archives
      legitimately ship 30-40 LOs once components are counted).
    - Warning if <50% of chunks have learning_outcome_refs.
    """
    result = ValidationResult(valid=True)

    course_json_path = course_dir / "course.json"
    # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
    chunks_path = course_dir / "imscc_chunks" / "chunks.json"
    if not chunks_path.exists():
        chunks_path = course_dir / "corpus" / "chunks.json"

    # Check course-level outcomes (terminal + component union)
    total = _count_total_learning_outcomes(course_dir)
    if total is None:
        result.add_warning("course.json not found, skipping outcome validation")
    else:
        if total < 10:
            result.add_error(
                f"Course has {total} learning outcomes, minimum is 10"
            )
        elif total > 60:
            result.add_warning(
                f"Course has {total} learning outcomes, recommended max is 60"
            )

    # Check chunk-level outcome refs
    if chunks_path.exists():
        try:
            with open(chunks_path) as f:
                chunks = json.load(f)

            if isinstance(chunks, list) and chunks:
                chunks_with_outcomes = sum(
                    1 for c in chunks if c.get("learning_outcome_refs")
                )
                coverage = chunks_with_outcomes / len(chunks)

                if coverage < 0.5:
                    result.add_warning(
                        f"Only {coverage:.1%} of chunks have learning_outcome_refs "
                        f"({chunks_with_outcomes}/{len(chunks)})"
                    )
        except json.JSONDecodeError:
            pass  # Already reported in other validators

    return result


def validate_concept_tags(
    course_dir: Path,
    max_unique_tags: int = 800,
) -> ValidationResult:
    """
    Validate concept tags against controlled vocabulary constraints.

    Requirements:
    - Max 800 unique concept tags per course (prevents explosion)
    - Tags must be lowercase-hyphenated format (1-4 words)
    - No markdown, special chars, or stopwords
    """
    result = ValidationResult(valid=True)

    # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
    chunks_path = course_dir / "imscc_chunks" / "chunks.json"
    if not chunks_path.exists():
        chunks_path = course_dir / "corpus" / "chunks.json"
    if not chunks_path.exists():
        return result

    try:
        with open(chunks_path) as f:
            chunks = json.load(f)
    except json.JSONDecodeError:
        return result  # Already reported elsewhere

    if not isinstance(chunks, list):
        return result

    all_tags = set()
    invalid_format_tags = []

    # Pattern: lowercase, hyphenated, 1-4 words, letters/numbers/hyphens only
    valid_tag_pattern = re.compile(r'^[a-z][a-z0-9]*(-[a-z0-9]+){0,3}$')

    for chunk in chunks:
        tags = chunk.get("concept_tags", [])
        for tag in tags:
            all_tags.add(tag)
            # Check format
            if not valid_tag_pattern.match(tag):
                if tag not in [t[0] for t in invalid_format_tags]:
                    invalid_format_tags.append((tag, chunk.get("id", "unknown")))

    # Check vocabulary size
    if len(all_tags) > max_unique_tags:
        result.add_error(
            f"Concept vocabulary explosion: {len(all_tags)} unique tags "
            f"(max {max_unique_tags})"
        )

    # Check format violations
    if invalid_format_tags:
        result.add_warning(
            f"Found {len(invalid_format_tags)} tags with invalid format "
            "(expected lowercase-hyphenated)"
        )
        # Show first 5 examples
        for tag, chunk_id in invalid_format_tags[:5]:
            result.add_warning(f"  Invalid tag '{tag}' in chunk {chunk_id}")
        if len(invalid_format_tags) > 5:
            result.add_warning(f"  ... and {len(invalid_format_tags) - 5} more")

    return result


def validate_course(course_dir: Path, repo_root: Path) -> ValidationResult:
    """Run all validations on a course."""
    result = ValidationResult(valid=True)

    # Structure validation
    struct_result = validate_course_structure(course_dir)
    result.merge(struct_result)

    # Manifest validation
    manifest_result = validate_course_manifest(course_dir, repo_root)
    result.merge(manifest_result)

    # Taxonomy validation
    taxonomy_result = validate_taxonomy_compliance(course_dir, repo_root)
    result.merge(taxonomy_result)

    return result


def validate_course_strict(course_dir: Path, repo_root: Path) -> ValidationResult:
    """
    Run all validations including strict checks.

    This includes:
    - Standard structure/manifest/taxonomy checks
    - Dataset config constraints (min/max tokens, require_concepts)
    - Learning outcomes (10-25 per course)
    - Concept tag governance (max 800, format)
    """
    result = validate_course(course_dir, repo_root)

    # Dataset config constraints
    config_result = validate_dataset_config_constraints(course_dir)
    result.merge(config_result)

    # Learning outcomes
    outcomes_result = validate_learning_outcomes(course_dir)
    result.merge(outcomes_result)

    # Concept tags
    concepts_result = validate_concept_tags(course_dir)
    result.merge(concepts_result)

    return result


def validate_repository(repo_root: Path) -> dict[str, ValidationResult]:
    """Validate all courses in the repository."""
    results = {}
    courses_dir = repo_root / "courses"

    if not courses_dir.exists():
        return results

    for course_dir in courses_dir.iterdir():
        if course_dir.is_dir() and not course_dir.name.startswith("."):
            results[course_dir.name] = validate_course(course_dir, repo_root)

    return results


def validate_indexes(repo_root: Path) -> ValidationResult:
    """Validate that indexes are consistent with courses."""
    result = ValidationResult(valid=True)

    catalog_path = repo_root / "catalog" / "master_catalog.json"
    courses_dir = repo_root / "courses"

    if not catalog_path.exists():
        result.add_warning("master_catalog.json not found")
        return result

    with open(catalog_path) as f:
        catalog = json.load(f)

    catalog_slugs = {c["slug"] for c in catalog.get("courses", [])}

    # Check all courses are in catalog
    if courses_dir.exists():
        for course_dir in courses_dir.iterdir():
            if course_dir.is_dir() and not course_dir.name.startswith("."):
                if course_dir.name not in catalog_slugs:
                    result.add_error(f"Course '{course_dir.name}' not in catalog")

    # Check all catalog entries exist
    for slug in catalog_slugs:
        if not (courses_dir / slug).exists():
            result.add_error(f"Catalog entry '{slug}' has no course directory")

    return result
