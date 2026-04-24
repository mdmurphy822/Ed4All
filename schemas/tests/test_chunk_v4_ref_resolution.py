"""Wave 74 Session 3 — chunk_v4 $ref resolution regression.

Guards against a production bug that killed the Trainforge assessment
phase today: ``Trainforge/process_course.py::_validate_chunk`` crashed
on the first chunk with
``jsonschema.exceptions._RefResolutionError: Unresolvable JSON pointer:
'$defs/Source'`` because the legacy ``RefResolver`` wiring in
``_load_chunk_validator`` resolved inline ``#/$defs/Source`` against a
stale base URI when another code path had already accessed the
deprecated ``.resolver`` attribute on the cached validator.

Contract locked by this suite:

- every ``$ref`` value reachable from ``chunk_v4.schema.json`` (inline
  ``#/$defs/...`` and external absolute URIs) resolves against the
  schemas under ``schemas/`` — no ``_RefResolutionError``
- the canonical Trainforge production path
  (``process_course._validate_chunk``) validates a real LibV2 chunk
  without exception
- Draft 2020-12 meta-validation of the schema itself still passes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401

        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _load_id_store() -> Dict[str, Dict[str, Any]]:
    """Load every schema under schemas/ keyed by its $id."""
    store: Dict[str, Dict[str, Any]] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            store[sid] = s
    return store


def _iter_refs(node: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], str]]:
    """Walk a schema recursively yielding (json-pointer-path, $ref-value)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str):
                yield (path, v)
            else:
                yield from _iter_refs(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _iter_refs(v, path + (str(i),))


def _sample_chunk_with_source_references() -> Dict[str, Any]:
    """A chunk payload that exercises inline + external $ref paths.

    Touches ``$defs/Source`` (inline) AND
    ``source_reference.schema.json`` (external) AND
    ``bloom_verbs.schema.json`` (external) AND
    ``content_type.schema.json`` (external).
    """
    return {
        "id": "sample_101_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample.",
        "html": "<p>Sample.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "SAMPLE_101",
            "module_id": "m1",
            "lesson_id": "l1",
            "source_references": [
                {"sourceId": "dart:slug#s0_c0", "role": "primary"},
            ],
        },
        "concept_tags": [],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 1,
        "word_count": 1,
        "bloom_level": "understand",
    }


# --------------------------------------------------------------------- #
# Meta-validation + ref-walk
# --------------------------------------------------------------------- #


def test_chunk_v4_is_valid_draft_2020_12():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)


def test_every_ref_in_chunk_v4_resolves():
    """Walk every $ref reachable from chunk_v4 and assert it resolves
    against the local schema store. Catches regressions where a $ref
    points at a $defs entry that no longer exists (the bug that
    crashed today's pipeline with
    ``Unresolvable JSON pointer: '$defs/Source'``).
    """
    _require_jsonschema()
    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    id_store = _load_id_store()

    try:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012
    except ImportError:  # pragma: no cover
        pytest.skip("referencing library not installed")

    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_store.items()
    ]
    registry = Registry().with_resources(resources)
    base_uri = schema.get("$id", "")
    resolver = registry.resolver(base_uri=base_uri)

    failures: List[str] = []
    for path, ref in _iter_refs(schema):
        try:
            resolved = resolver.lookup(ref)
            assert resolved.contents is not None
        except Exception as exc:  # pragma: no cover — failure path
            failures.append(
                f"  {'/'.join(path) or '<root>'}: $ref={ref!r} -> "
                f"{type(exc).__name__}: {exc}"
            )

    assert not failures, (
        "chunk_v4 contains unresolvable $refs:\n" + "\n".join(failures)
    )


def test_source_defs_ref_is_present_and_inline():
    """The ``#/$defs/Source`` $ref at ``properties.source`` must point
    at an inline ``$defs.Source`` entry. Prevents silent regressions
    where Source gets moved external (breaking inline resolvers) or
    the $defs entry gets deleted (the bug observed today).
    """
    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    src_ref = schema["properties"]["source"].get("$ref")
    assert src_ref == "#/$defs/Source", f"unexpected $ref: {src_ref!r}"
    assert "Source" in schema.get("$defs", {}), (
        "$defs/Source missing — chunk_v4 references it at "
        "properties.source.$ref but the inline definition is absent"
    )


# --------------------------------------------------------------------- #
# End-to-end: production code path validates a real chunk cleanly
# --------------------------------------------------------------------- #


def test_process_course_validate_chunk_does_not_raise_on_real_chunk():
    """Calls ``Trainforge.process_course._validate_chunk`` on a real
    chunk payload. Must NOT raise ``_RefResolutionError`` (the symptom
    from today's RDF_SHACL_KG run).
    """
    _require_jsonschema()
    # Defer import so the test is robust when process_course's imports
    # fail on a stripped-down sandbox.
    try:
        from Trainforge.process_course import _validate_chunk
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"process_course not importable: {exc}")

    chunk = _sample_chunk_with_source_references()
    # Should return None (valid) or a plain string (validation error).
    # Must NOT raise — that's the regression we're guarding against.
    result = _validate_chunk(chunk)
    assert result is None or isinstance(result, str), (
        f"_validate_chunk returned unexpected type: {type(result).__name__}"
    )


def test_process_course_validator_resolves_all_refs():
    """Validator built by ``_load_chunk_validator`` must not raise
    ``_RefResolutionError`` when iterating errors — repeated calls
    (which mimic the ``_write_chunks`` loop) must stay clean.
    """
    _require_jsonschema()
    try:
        from Trainforge.process_course import _load_chunk_validator
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"process_course not importable: {exc}")

    validator = _load_chunk_validator()
    if validator is None:  # pragma: no cover — optional dep missing
        pytest.skip("chunk validator unavailable (jsonschema absent)")

    # Run the validator over a chunk — must not raise.
    chunk = _sample_chunk_with_source_references()
    errors = list(validator.iter_errors(chunk))
    # A correctly-shaped chunk has zero errors.
    assert errors == [], [e.message for e in errors]

    # Second call exercises the cached validator — also must not raise
    # (the production bug crashed on the FIRST chunk, but a regression
    # where cache poisoning only shows up on call N>1 would slip past
    # a single-shot test).
    errors_second = list(validator.iter_errors(chunk))
    assert errors_second == []


@pytest.mark.parametrize(
    "corpus_relpath",
    [
        "LibV2/courses/rdf-shacl-kg/corpus/chunks.jsonl",
        "LibV2/courses/best-practices-in-digital-web-design-for-accessibi/corpus/chunks.jsonl",
    ],
)
def test_process_course_validates_real_corpus_chunks(corpus_relpath: str):
    """Regression: real LibV2 chunks (from the failed run's corpus and
    a known-clean corpus) validate without raising.
    """
    _require_jsonschema()
    try:
        from Trainforge.process_course import _validate_chunk
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"process_course not importable: {exc}")

    corpus_path = PROJECT_ROOT / corpus_relpath
    if not corpus_path.exists():
        pytest.skip(f"corpus fixture missing: {corpus_path}")

    lines = corpus_path.read_text().splitlines()
    if not lines:
        pytest.skip(f"corpus empty: {corpus_path}")

    # Validate at least the first 5 chunks — the production bug
    # crashed on the FIRST chunk, so this is the canonical reproducer.
    for i, line in enumerate(lines[:5]):
        if not line.strip():
            continue
        chunk = json.loads(line)
        # Must not raise.
        result = _validate_chunk(chunk)
        assert result is None or isinstance(result, str), (
            f"chunk {i} in {corpus_relpath}: unexpected return type "
            f"{type(result).__name__}"
        )
