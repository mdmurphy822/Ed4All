"""Worker I — chunk_v4 + courseforge_jsonld_v1 schema validation tests.

Regression tests for the two knowledge-layer schemas authored in Wave 1.2 of
the KG-quality review (plans/kg-quality-review-2026-04):

  - schemas/knowledge/chunk_v4.schema.json — formalizes the Trainforge chunk
    node shape. Validation hook is wired into Trainforge/process_course.py
    ::_write_chunks and gated by TRAINFORGE_VALIDATE_CHUNKS=true (fail-closed)
    with a warn-log default.
  - schemas/knowledge/courseforge_jsonld_v1.schema.json — formalizes the
    Courseforge JSON-LD contract. Not hooked into runtime validation yet;
    this file only tests the schema + real-page validation.

Both schemas $ref Worker F's taxonomy files (merged in PR #20) for enum
safety. Tests build a local schema store so $id URIs resolve offline.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Project root (Ed4All/). This file lives at
# Ed4All/Trainforge/tests/test_chunk_validation.py → parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"
JSONLD_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "courseforge_jsonld_v1.schema.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_jsonschema():
    """Skip the test if jsonschema isn't installed."""
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover — test-env bootstrap
        pytest.skip("jsonschema not installed")


def _build_validator(schema_path: Path):
    """Build a Draft202012Validator with a RefResolver populated from every
    $id in schemas/ so Worker F taxonomy references resolve offline.
    """
    jsonschema = _require_jsonschema()
    from jsonschema import Draft202012Validator, RefResolver

    with open(schema_path) as f:
        schema = json.load(f)
    store: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            store[sid] = s
    resolver = RefResolver.from_schema(schema, store=store)
    return schema, Draft202012Validator(schema, resolver=resolver)


def _find_real_chunks_jsonl() -> Optional[Path]:
    """Locate a real chunks.jsonl from LibV2 for regression testing.

    Prefers the WCAG-201 / best-practices course referenced in the sub-plan.
    Returns None if no corpus is present in the checkout.
    """
    candidates = [
        PROJECT_ROOT
        / "LibV2"
        / "courses"
        / "best-practices-in-digital-web-design-for-accessibi"
        / "corpus"
        / "chunks.jsonl",
        PROJECT_ROOT
        / "LibV2"
        / "courses"
        / "foundations-of-digital-pedagogy"
        / "corpus"
        / "chunks.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: first corpus/chunks.jsonl we find anywhere under LibV2
    for p in (PROJECT_ROOT / "LibV2" / "courses").rglob("chunks.jsonl"):
        if p.is_file():
            return p
    return None


def _find_wcag_jsonld_pages() -> List[Path]:
    """Locate WCAG_201 course HTML pages with embedded JSON-LD."""
    root = (
        PROJECT_ROOT
        / "Courseforge"
        / "exports"
        / "WCAG_201_COURSE"
        / "03_content_development"
    )
    if not root.exists():
        return []
    return sorted(root.rglob("*.html"))


_JSONLD_RE = re.compile(
    r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
    re.S,
)


def _extract_jsonld(html: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON-LD block from a page."""
    m = _JSONLD_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _make_valid_chunk() -> Dict[str, Any]:
    """Construct a minimal chunk_v4-compliant record for positive tests."""
    return {
        "id": "test_course_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk text.",
        "html": "<p>Sample chunk text.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "m1",
            "lesson_id": "l1",
        },
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 3,
        "word_count": 3,
        "bloom_level": "understand",
    }


# ---------------------------------------------------------------------------
# Schema self-validation (sanity: the schema itself parses as a valid JSON
# Schema Draft 2020-12 document).
# ---------------------------------------------------------------------------


def test_chunk_schema_self_valid():
    jsonschema = _require_jsonschema()
    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    jsonschema.Draft202012Validator.check_schema(schema)


def test_jsonld_schema_self_valid():
    jsonschema = _require_jsonschema()
    with open(JSONLD_SCHEMA_PATH) as f:
        schema = json.load(f)
    jsonschema.Draft202012Validator.check_schema(schema)


# ---------------------------------------------------------------------------
# Regression: existing production chunks/JSON-LD validate against the new
# schemas at or above the master-plan's 95% threshold.
# ---------------------------------------------------------------------------


def test_existing_libv2_chunks_validate():
    _require_jsonschema()
    chunks_path = _find_real_chunks_jsonl()
    if chunks_path is None:
        pytest.skip("No LibV2 chunks.jsonl present in this checkout")

    _, validator = _build_validator(CHUNK_SCHEMA_PATH)
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    assert chunks, f"chunks.jsonl at {chunks_path} is empty"

    valid = 0
    failures: List[Tuple[str, str]] = []
    for c in chunks:
        errs = list(validator.iter_errors(c))
        if not errs:
            valid += 1
        else:
            path = ".".join(str(p) for p in errs[0].absolute_path) or "root"
            failures.append((c.get("id", "?"), f"{path}: {errs[0].message}"))

    ratio = valid / len(chunks)
    # Per master plan: "Expected: ≥95% valid; remaining failures documented
    # as known gaps for later waves."
    assert ratio >= 0.95, (
        f"chunk_v4 regression: {valid}/{len(chunks)} valid ({ratio:.1%}); "
        f"first failures: {failures[:3]}"
    )


def test_existing_wcag_jsonld_validates():
    _require_jsonschema()
    pages = _find_wcag_jsonld_pages()
    if not pages:
        pytest.skip("No WCAG_201 course export present in this checkout")

    _, validator = _build_validator(JSONLD_SCHEMA_PATH)
    total = 0
    valid = 0
    failures: List[Tuple[str, str]] = []
    for page in pages:
        data = _extract_jsonld(page.read_text())
        if data is None:
            continue
        total += 1
        errs = list(validator.iter_errors(data))
        if not errs:
            valid += 1
        elif len(failures) < 3:
            path = ".".join(str(p) for p in errs[0].absolute_path) or "root"
            failures.append((page.name, f"{path}: {errs[0].message}"))

    assert total > 0, "Expected at least one page with embedded JSON-LD"
    ratio = valid / total
    # JSON-LD is the emit-only contract; we expect 100% on a clean run.
    # Tolerate the same 95% threshold as chunks for parity.
    assert ratio >= 0.95, (
        f"JSON-LD regression: {valid}/{total} valid ({ratio:.1%}); "
        f"first failures: {failures[:3]}"
    )


# ---------------------------------------------------------------------------
# Production hook behaviour: _validate_chunk + _write_chunks env-var gate.
# ---------------------------------------------------------------------------


def test_validate_chunk_passes_on_valid_sample():
    _require_jsonschema()
    from Trainforge.process_course import _validate_chunk

    chunk = _make_valid_chunk()
    assert _validate_chunk(chunk) is None


def test_validate_chunk_catches_missing_source_course_id():
    _require_jsonschema()
    from Trainforge.process_course import _validate_chunk

    chunk = _make_valid_chunk()
    del chunk["source"]["course_id"]
    err = _validate_chunk(chunk)
    assert err is not None
    assert "course_id" in err


def test_validate_chunk_catches_wrong_schema_version():
    _require_jsonschema()
    from Trainforge.process_course import _validate_chunk

    chunk = _make_valid_chunk()
    chunk["schema_version"] = "v3"
    err = _validate_chunk(chunk)
    assert err is not None


def test_write_chunks_strict_mode_raises(monkeypatch, tmp_path):
    """TRAINFORGE_VALIDATE_CHUNKS=true + malformed chunk → ValueError."""
    _require_jsonschema()
    import Trainforge.process_course as pc

    # Build a minimal processor-like shim that exposes only what
    # _write_chunks touches: corpus_dir and capture.log_decision.
    class _StubCapture:
        def log_decision(self, **kwargs):
            pass

    class _Stub:
        pass

    stub = _Stub()
    stub.corpus_dir = tmp_path
    stub.capture = _StubCapture()

    bad = _make_valid_chunk()
    del bad["source"]["course_id"]

    monkeypatch.setenv("TRAINFORGE_VALIDATE_CHUNKS", "true")
    with pytest.raises(ValueError, match="chunk_v4 validation failed"):
        pc.CourseProcessor._write_chunks(stub, [bad])


def test_write_chunks_default_warns(monkeypatch, tmp_path, caplog):
    """Env unset → warn-only; chunks still written; no raise."""
    _require_jsonschema()
    import Trainforge.process_course as pc

    class _StubCapture:
        def __init__(self):
            self.decisions = []

        def log_decision(self, **kwargs):
            self.decisions.append(kwargs)

    class _Stub:
        pass

    stub = _Stub()
    stub.corpus_dir = tmp_path
    stub.capture = _StubCapture()

    bad = _make_valid_chunk()
    del bad["source"]["course_id"]

    monkeypatch.delenv("TRAINFORGE_VALIDATE_CHUNKS", raising=False)
    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        # Should NOT raise
        pc.CourseProcessor._write_chunks(stub, [bad])

    # Warning was emitted
    assert any(
        "chunk_v4 validation" in rec.message for rec in caplog.records
    ), f"Expected chunk_v4 validation warning; got {[r.message for r in caplog.records]}"

    # Files were still written
    assert (tmp_path / "chunks.jsonl").exists()
    assert (tmp_path / "chunks.json").exists()


def test_write_chunks_valid_chunks_pass_strict(monkeypatch, tmp_path):
    """Valid chunks under strict mode → no raise, files written."""
    _require_jsonschema()
    import Trainforge.process_course as pc

    class _StubCapture:
        def log_decision(self, **kwargs):
            pass

    class _Stub:
        pass

    stub = _Stub()
    stub.corpus_dir = tmp_path
    stub.capture = _StubCapture()

    monkeypatch.setenv("TRAINFORGE_VALIDATE_CHUNKS", "true")
    pc.CourseProcessor._write_chunks(stub, [_make_valid_chunk()])
    assert (tmp_path / "chunks.jsonl").exists()
    assert (tmp_path / "chunks.json").exists()


# ---------------------------------------------------------------------------
# Wave 3 / Worker M — Case preservation for learning_outcome_refs (A3)
# ---------------------------------------------------------------------------
#
# The opt-in env var ``TRAINFORGE_PRESERVE_LO_CASE=true`` stops
# ``CourseProcessor._extract_objective_refs`` from lowercasing
# structured LO ids at ingest. Default stays lowercase for backward-
# compat with existing LibV2 chunks; the default flips in Wave 4's
# structural migration. See plans/kg-quality-review-2026-04/
# worker-m-subplan.md §2.
# ---------------------------------------------------------------------------


class _LOStub:
    """Minimal LearningObjective look-alike for _extract_objective_refs.

    The production code reads ``lo.id`` via ``hasattr`` (see
    ``process_course.py::_extract_objective_refs``); we only need that
    attribute to exercise the case-normalisation branch.
    """

    def __init__(self, obj_id: str):
        self.id = obj_id


def _call_extract_objective_refs(obj_ids):
    """Run ``CourseProcessor._extract_objective_refs`` on a synthetic item.

    Uses a SimpleNamespace shim so we don't need a full processor
    instance — the method only touches ``self.WEEK_PREFIX_RE`` and
    ``self.OBJECTIVE_CODE_RE`` (both class attrs on CourseProcessor).
    """
    from types import SimpleNamespace

    import Trainforge.process_course as pc

    item = {
        "learning_objectives": [_LOStub(x) for x in obj_ids],
        "key_concepts": [],
        "sections": [],
        "objective_refs": [],
    }
    stub = SimpleNamespace(
        WEEK_PREFIX_RE=pc.CourseProcessor.WEEK_PREFIX_RE,
        OBJECTIVE_CODE_RE=pc.CourseProcessor.OBJECTIVE_CODE_RE,
    )
    return pc.CourseProcessor._extract_objective_refs(stub, item)


def test_preserve_case_flag_off_lowercases(monkeypatch):
    """Default env (unset) → refs lowercased for backward-compat."""
    monkeypatch.delenv("TRAINFORGE_PRESERVE_LO_CASE", raising=False)
    refs = _call_extract_objective_refs(["TO-01"])
    assert refs == ["to-01"], (
        f"default env must lowercase LO refs; got {refs}"
    )


def test_preserve_case_flag_on_preserves(monkeypatch):
    """TRAINFORGE_PRESERVE_LO_CASE=true → refs preserve source casing."""
    monkeypatch.setenv("TRAINFORGE_PRESERVE_LO_CASE", "true")
    refs = _call_extract_objective_refs(["TO-01"])
    assert refs == ["TO-01"], (
        f"flag=true must preserve case; got {refs}"
    )


def test_preserve_case_flag_non_true_values_lowercase(monkeypatch):
    """Only the literal string 'true' enables preservation (case-insensitive)."""
    for val in ("false", "0", "", "1", "yes"):
        monkeypatch.setenv("TRAINFORGE_PRESERVE_LO_CASE", val)
        refs = _call_extract_objective_refs(["TO-01"])
        assert refs == ["to-01"], (
            f"TRAINFORGE_PRESERVE_LO_CASE={val!r} should NOT enable "
            f"preservation; got {refs}"
        )


def test_preserve_case_flag_on_still_strips_week_prefix(monkeypatch):
    """Week prefix (W01-, w01-) still stripped regardless of case flag."""
    monkeypatch.setenv("TRAINFORGE_PRESERVE_LO_CASE", "true")
    refs = _call_extract_objective_refs(["W03-CO-05"])
    # WEEK_PREFIX_RE is case-insensitive → W03- stripped; CO-05 stays
    # uppercase because preserve_case is on.
    assert refs == ["CO-05"], (
        f"week prefix should strip; casing should preserve; got {refs}"
    )
