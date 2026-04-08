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


def load_schema(repo_root: Path, schema_name: str) -> Optional[dict]:
    """Load a JSON schema from the schema directory."""
    schema_path = repo_root / "schema" / schema_name
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
    required_dirs = [
        "corpus",
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

    # Check optional directories
    for dirname in optional_dirs:
        if not (course_dir / dirname).exists():
            result.add_warning(f"Missing optional directory: {dirname}")

    # Check corpus contents
    corpus_dir = course_dir / "corpus"
    if corpus_dir.exists():
        if not (corpus_dir / "chunks.json").exists() and not (corpus_dir / "chunks.jsonl").exists():
            result.add_error("Missing chunks.json or chunks.jsonl in corpus/")

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
                result.add_error("Invalid JSON in corpus/chunks.json")

    return result


def validate_taxonomy_compliance(course_dir: Path, repo_root: Path) -> ValidationResult:
    """Validate that course classification uses valid taxonomy terms."""
    result = ValidationResult(valid=True)

    # Load taxonomy
    taxonomy_path = repo_root / "ontology" / "taxonomy.json"
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

    # Validate domain
    if division and domain:
        division_data = taxonomy["divisions"].get(division, {})
        domains = division_data.get("domains", {})
        if domain not in domains:
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
    chunks_path = course_dir / "corpus" / "chunks.json"

    if not config_path.exists():
        result.add_warning("No dataset_config.json found, skipping constraint validation")
        return result

    if not chunks_path.exists():
        result.add_error("corpus/chunks.json not found")
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


def validate_learning_outcomes(course_dir: Path) -> ValidationResult:
    """
    Validate learning outcome coverage.

    Requirements:
    - 10-25 course-level learning outcomes
    - Warning if <50% of chunks have learning_outcome_refs
    """
    result = ValidationResult(valid=True)

    course_json_path = course_dir / "course.json"
    chunks_path = course_dir / "corpus" / "chunks.json"

    # Check course-level outcomes
    if course_json_path.exists():
        try:
            with open(course_json_path) as f:
                course = json.load(f)

            outcomes = course.get("learning_outcomes", [])
            if len(outcomes) < 10:
                result.add_error(
                    f"Course has {len(outcomes)} learning outcomes, minimum is 10"
                )
            elif len(outcomes) > 25:
                result.add_warning(
                    f"Course has {len(outcomes)} learning outcomes, recommended max is 25"
                )
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in course.json: {e}")
    else:
        result.add_warning("course.json not found, skipping outcome validation")

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
