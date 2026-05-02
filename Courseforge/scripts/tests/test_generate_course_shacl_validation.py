"""Wave 68 — emit-time SHACL validation in generate_course.py.

Companion to ``test_generate_course_jsonld_validation.py`` (Wave 49).
The Wave 49 validator catches schema-shape drift at emit time; Wave 68
adds a second, RDF-native validation layer that runs the same payload
through the Wave 62 @context + Wave 63/67 SHACL shapes. The two layers
are complementary — SHACL catches IRI-level closed-set constraints
(e.g., typo'd bloomLevel IRIs, missing Section headings) that the JSON
Schema's flat enum checks can't express.

These tests cover:

1. Well-formed ``generate_week`` round-trip has ``conforms=True`` and
   no WARNING logs.
2. With ``COURSEFORGE_ENFORCE_SHACL=1`` and an LO carrying a deliberately
   malformed bloomLevel IRI (outside the canonical 6-concept SKOS set),
   the emit raises ``ValueError``.
3. Without the env var (default), the same malformed emit logs a
   WARNING and does NOT raise — preserves back-compat.
4. Graceful-degrade: when pyld / pyshacl / rdflib raise ``ImportError``
   the helper becomes a no-op (returns ``(True, "")``) and emit still
   succeeds. Simulated via monkeypatching the dep-availability check.
5. Load-caching: the ``@context`` document and SHACL shapes graph are
   each loaded exactly once per process via ``functools.lru_cache``.
   Verified by inspecting ``cache_info()`` hits after two calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Skip the whole module if the RDF stack isn't importable — consistent
# with schemas/tests/test_courseforge_shacl_shapes.py. Production envs
# have all three pinned via pyproject.toml; CI skips gracefully when
# they're missing.
pytest.importorskip(
    "pyld",
    reason="pyld is required for SHACL tests; install with `pip install pyld`.",
)
pytest.importorskip(
    "pyshacl",
    reason="pyshacl is required for SHACL tests; install with `pip install pyshacl`.",
)
pytest.importorskip("rdflib", reason="rdflib comes with pyshacl.")

import generate_course  # noqa: E402
from generate_course import (  # noqa: E402
    _ENFORCE_SHACL_ENV,
    _validate_page_jsonld_shacl,
    _validate_page_jsonld_shacl_at_emit,
    generate_week,
)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture
def well_formed_metadata() -> dict:
    """Minimally-valid CourseModule payload. Mirrors the shape
    ``_build_page_metadata`` produces for an overview page with no
    optional arrays set. Should pass both JSON Schema and SHACL."""
    return {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": "SAMPLE_101",
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01_intro",
    }


@pytest.fixture
def bad_bloom_iri_metadata() -> dict:
    """CourseModule payload with a LO that carries a bloomLevel IRI
    whose local fragment is typo'd ("aplly" instead of "apply"). The
    Wave 67 ``sh:in`` check against the canonical 6-concept set fires;
    the older Wave 63 prefix pattern would have passed this. Used to
    validate the SHACL layer catches drift the JSON Schema can't
    express (JSON Schema checks only the wire-format enum, not the
    expanded IRI)."""
    return {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": "SAMPLE_101",
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01_bad_bloom",
        "learningObjectives": [
            {
                "@type": "LearningObjective",
                "id": "https://example.org/los/CO-01",
                "statement": "Apply the framework to sample data.",
                # Full IRI form bypasses the @context enum mapping so
                # the expanded graph carries a non-canonical IRI.
                "ed4all:bloomLevel": {
                    "@id": "https://ed4all.dev/vocab/bloom#aplly"
                },
            }
        ],
    }


@pytest.fixture
def week_data() -> dict:
    """Minimal week-data fixture that mirrors the Wave 49 test fixture
    for apples-to-apples emit coverage."""
    return {
        "week_number": 3,
        "title": "Visual Perception",
        "objectives": [
            {
                "id": "CO-03",
                "statement": "Apply color contrast rules",
                "bloom_level": "apply",
            },
        ],
        "overview_text": ["Intro paragraph."],
        "readings": ["Ch. 5 pp. 80-92"],
        "content_modules": [
            {
                "title": "POUR Principles",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": ["POUR stands for ..."],
                    },
                ],
            }
        ],
        "activities": [
            {
                "title": "Color Audit",
                "description": "Evaluate contrast on a real page.",
                "bloom_level": "apply",
            },
        ],
        "key_takeaways": ["POUR is the accessibility foundation."],
        "reflection_questions": ["Which principle feels most challenging?"],
    }


# ---------------------------------------------------------------------- #
# 1. Well-formed metadata conforms; generate_week round-trip is clean
# ---------------------------------------------------------------------- #


def test_well_formed_metadata_conforms(well_formed_metadata):
    """A minimally-valid CourseModule payload validates against SHACL."""
    conforms, text = _validate_page_jsonld_shacl(well_formed_metadata)
    assert conforms, f"Well-formed metadata failed SHACL:\n{text}"


def test_well_formed_metadata_emits_no_warning(
    well_formed_metadata, caplog, monkeypatch,
):
    """Well-formed payload must not log a SHACL warning."""
    monkeypatch.delenv(_ENFORCE_SHACL_ENV, raising=False)
    with caplog.at_level(logging.WARNING, logger=generate_course.logger.name):
        _validate_page_jsonld_shacl_at_emit(
            well_formed_metadata, page_id=well_formed_metadata["pageId"],
        )
    assert not any(
        "SHACL validation failed" in rec.getMessage()
        for rec in caplog.records
    ), f"Unexpected SHACL warning on valid metadata: {caplog.records!r}"


def test_real_generate_week_output_passes_shacl(
    tmp_path, week_data, monkeypatch, caplog,
):
    """A real ``generate_week`` emit run with SHACL enforcement ON
    must NOT raise and must NOT log any SHACL warning. Proves the
    emit path stays clean under the hardened Wave 67 shapes."""
    monkeypatch.setenv(_ENFORCE_SHACL_ENV, "1")
    with caplog.at_level(logging.WARNING, logger=generate_course.logger.name):
        # Must not raise.
        generate_week(
            week_data, tmp_path / "out", "SAMPLE_101", source_module_map=None,
        )
    assert not any(
        "SHACL validation failed" in rec.getMessage()
        for rec in caplog.records
    ), f"generate_week tripped SHACL: {caplog.records!r}"


# ---------------------------------------------------------------------- #
# 2. Enforcement flag: truthy -> ValueError
# ---------------------------------------------------------------------- #


def test_malformed_metadata_raises_when_enforced(
    bad_bloom_iri_metadata, monkeypatch,
):
    """With ``COURSEFORGE_ENFORCE_SHACL=1`` a typo'd bloomLevel IRI
    must raise ``ValueError``. This is the IRI-level drift the Wave
    49 JSON-Schema layer cannot catch."""
    monkeypatch.setenv(_ENFORCE_SHACL_ENV, "1")
    with pytest.raises(ValueError) as excinfo:
        _validate_page_jsonld_shacl_at_emit(
            bad_bloom_iri_metadata, page_id="week_01_content_01_bad_bloom",
        )
    msg = str(excinfo.value)
    assert "week_01_content_01_bad_bloom" in msg
    assert "failed SHACL validation" in msg


def test_enforcement_flag_truthy_values(bad_bloom_iri_metadata, monkeypatch):
    """All four documented truthy spellings trigger the raise path,
    matching the Wave 49 ``_ENFORCE_TRUTHY_VALUES`` frozenset."""
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv(_ENFORCE_SHACL_ENV, val)
        with pytest.raises(ValueError):
            _validate_page_jsonld_shacl_at_emit(
                bad_bloom_iri_metadata, page_id=f"p-{val}",
            )


# ---------------------------------------------------------------------- #
# 3. Default (unset): malformed metadata logs a WARNING and does NOT raise
# ---------------------------------------------------------------------- #


def test_malformed_metadata_logs_warning_when_unenforced(
    bad_bloom_iri_metadata, caplog, monkeypatch,
):
    """Default (env unset) must log a WARNING and not raise — legacy
    corpora with known shape quirks don't block CI on the day Wave 68
    lands."""
    monkeypatch.delenv(_ENFORCE_SHACL_ENV, raising=False)
    with caplog.at_level(logging.WARNING, logger=generate_course.logger.name):
        result = _validate_page_jsonld_shacl_at_emit(
            bad_bloom_iri_metadata, page_id="week_01_content_01_warn",
        )
    assert result is None
    fired = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING
        and "SHACL validation failed" in rec.getMessage()
        and "week_01_content_01_warn" in rec.getMessage()
    ]
    assert fired, (
        "Expected WARNING log mentioning page_id + SHACL failure; got "
        f"{caplog.records!r}"
    )


def test_enforcement_flag_falsy_values(
    bad_bloom_iri_metadata, caplog, monkeypatch,
):
    """Empty / other values should WARN, not raise."""
    for val in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(_ENFORCE_SHACL_ENV, val)
        # Must not raise.
        _validate_page_jsonld_shacl_at_emit(
            bad_bloom_iri_metadata, page_id=f"p-{val or 'empty'}",
        )


# ---------------------------------------------------------------------- #
# 4. End-to-end wire-up: _wrap_page invokes the SHACL layer
# ---------------------------------------------------------------------- #


def test_wrap_page_raises_when_strict_and_emit_drifts(
    tmp_path, week_data, monkeypatch,
):
    """Positive wire-up test for the SHACL layer on the emit path:
    force ``_build_page_metadata`` to inject a typo'd bloomLevel IRI,
    flip the SHACL enforcement flag, and confirm ``generate_week``
    surfaces the SHACL failure at the emit site (not dead code)."""
    monkeypatch.setenv(_ENFORCE_SHACL_ENV, "1")
    # Wave 49 enforcement OFF so we isolate SHACL as the failure.
    monkeypatch.delenv("COURSEFORGE_ENFORCE_JSONLD_SCHEMA", raising=False)

    real_build = generate_course._build_page_metadata

    def _broken_build_page_metadata(*args, **kwargs):
        meta = real_build(*args, **kwargs)
        # Inject a typo'd bloomLevel IRI via the full-IRI form so the
        # expanded graph carries a non-canonical IRI — SHACL's sh:in
        # check fires, JSON Schema's enum check doesn't.
        los = meta.get("learningObjectives")
        if los:
            los[0]["ed4all:bloomLevel"] = {
                "@id": "https://ed4all.dev/vocab/bloom#aplly"
            }
        return meta

    monkeypatch.setattr(
        generate_course, "_build_page_metadata", _broken_build_page_metadata,
    )
    with pytest.raises(ValueError, match="failed SHACL validation"):
        generate_week(
            week_data, tmp_path / "out", "SAMPLE_101", source_module_map=None,
        )


# ---------------------------------------------------------------------- #
# 5. Graceful-degrade: ImportError disables the SHACL layer
# ---------------------------------------------------------------------- #


def test_shacl_validation_noop_when_deps_missing(
    bad_bloom_iri_metadata, monkeypatch,
):
    """When pyld / pyshacl / rdflib aren't importable, the helper
    must become a no-op (returns ``(True, "")``) so emit still
    succeeds on thinly-dependencied environments.

    Simulated by monkeypatching ``_shacl_deps_available`` to return
    False. This mirrors the real fallback branch — a production env
    missing the RDF stack would hit the same ``return True, ""`` path.
    """
    monkeypatch.setattr(
        generate_course, "_shacl_deps_available", lambda: False,
    )
    conforms, text = _validate_page_jsonld_shacl(bad_bloom_iri_metadata)
    assert conforms is True
    assert text == ""


def test_wrap_page_succeeds_when_shacl_deps_missing(
    tmp_path, week_data, monkeypatch,
):
    """End-to-end graceful-degrade: with ``COURSEFORGE_ENFORCE_SHACL=1``
    but SHACL deps "unavailable", ``generate_week`` still succeeds —
    the helper's no-op path short-circuits before the enforcement
    check."""
    monkeypatch.setenv(_ENFORCE_SHACL_ENV, "1")
    monkeypatch.setattr(
        generate_course, "_shacl_deps_available", lambda: False,
    )
    # Must not raise.
    generate_week(
        week_data, tmp_path / "out", "SAMPLE_101", source_module_map=None,
    )


# ---------------------------------------------------------------------- #
# 6. Load-caching: shapes graph + @context document load once per process
# ---------------------------------------------------------------------- #


def test_context_and_shapes_graph_are_lru_cached(well_formed_metadata):
    """The ``@context`` document and SHACL shapes graph are each wired
    to ``functools.lru_cache(maxsize=1)``. Second call on identical args
    must hit the cache, proving a single disk-read per process.

    Guards against a refactor that inlines the file-read inside the
    per-page hot path (SHACL on every emitted HTML page in a 15-week
    course would re-read the shapes file ~120 times without the
    cache)."""
    # Clear caches to get a clean baseline.
    generate_course._load_shacl_context.cache_clear()
    generate_course._load_shacl_shapes_graph.cache_clear()

    # Two validation calls back-to-back; both should reuse the cached
    # context + shapes after the first.
    _validate_page_jsonld_shacl(well_formed_metadata)
    _validate_page_jsonld_shacl(well_formed_metadata)

    ctx_info = generate_course._load_shacl_context.cache_info()
    shapes_info = generate_course._load_shacl_shapes_graph.cache_info()

    # Exactly one miss (first call populates the cache) and at least
    # one hit (subsequent calls reuse).
    assert ctx_info.misses == 1, (
        f"Expected exactly one @context disk-read; got {ctx_info!r}"
    )
    assert ctx_info.hits >= 1, (
        f"Expected >=1 @context cache hit after second call; got {ctx_info!r}"
    )
    assert shapes_info.misses == 1, (
        f"Expected exactly one shapes-graph disk-read; got {shapes_info!r}"
    )
    assert shapes_info.hits >= 1, (
        f"Expected >=1 shapes-graph cache hit after second call; got "
        f"{shapes_info!r}"
    )


def test_context_cache_survives_many_calls(well_formed_metadata):
    """Thirty validation calls still produce exactly one disk-read for
    each of the two cached resources. Closer to the emit hot-path than
    the two-call smoke test above."""
    generate_course._load_shacl_context.cache_clear()
    generate_course._load_shacl_shapes_graph.cache_clear()

    for _ in range(30):
        _validate_page_jsonld_shacl(well_formed_metadata)

    assert generate_course._load_shacl_context.cache_info().misses == 1
    assert generate_course._load_shacl_shapes_graph.cache_info().misses == 1


# ---------------------------------------------------------------------- #
# 7. Phase 2 Subtask 14: Block touched_by cardinality + Touch shape
# ---------------------------------------------------------------------- #


def _block_payload(touched_by: list | None = None) -> dict:
    """Build a CourseModule payload carrying one Block with
    ``@type: "Block"`` so the cfshapes:BlockShape (sh:targetClass
    ed4all:Block) fires during SHACL validation. Phase 2 emits
    ``blocks[]`` as a top-level optional array; in the JSON-LD context
    the array is mapped via ``ed4all:hasBlock`` to a set of nodes,
    each typed as ``ed4all:Block`` via the ``@type: "Block"`` alias.
    """
    block = {
        "@type": "Block",
        "blockId": "week_01_content_01_intro#objective_TO-01_0",
        "blockType": "objective",
        "sequence": 0,
    }
    if touched_by is not None:
        block["touchedBy"] = touched_by
    return {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": "SAMPLE_101",
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01_intro",
        "blocks": [block],
    }


def test_block_touched_by_cardinality_validated_by_shacl():
    """Phase 2 Subtask 14: assert SHACL enforces the Block + Touch
    contract.

    Two cases:
      1. A Block with empty ``touchedBy[]`` (the cardinality minimum is
         0) conforms.
      2. A Block carrying a Touch missing ``decisionCaptureId`` (Wave
         112 invariant; SHACL pins it via ``sh:minCount 1``) does NOT
         conform — the violation message references the Touch shape.

    Skip cleanly if the RDF stack isn't importable (mirrors the
    module-level ``pytest.importorskip`` pattern at the top of this
    file).
    """
    # Case 1: empty touchedBy[] is valid (sh:minCount 0).
    valid_payload = _block_payload(touched_by=[])
    conforms, text = _validate_page_jsonld_shacl(valid_payload)
    assert conforms, (
        f"Block with empty touchedBy[] should validate; SHACL said:\n{text}"
    )

    # Case 2: Touch missing decisionCaptureId fires the TouchShape
    # cardinality violation. Note: per the SHACL Touch shape, ``tier``,
    # ``provider``, and ``decisionCaptureId`` are each required (minCount 1);
    # leaving ``decisionCaptureId`` off (while keeping the other two)
    # isolates the Wave 112 invariant as the failure reason.
    bad_touch = {
        "@type": "Touch",
        "model": "qwen2.5-14b",
        "provider": "local",
        "tier": "outline",
        "timestamp": "2026-05-02T00:00:00Z",
        # decisionCaptureId deliberately omitted — Wave 112 invariant
        # requires it to be present + non-empty.
        "purpose": "draft",
    }
    invalid_payload = _block_payload(touched_by=[bad_touch])
    conforms, text = _validate_page_jsonld_shacl(invalid_payload)
    assert not conforms, (
        "Touch missing decisionCaptureId should fail SHACL; "
        f"got conforms=True. Results:\n{text}"
    )
    # Sanity: the violation should reference the missing predicate or
    # the TouchShape directly.
    assert (
        "decisionCaptureId" in text
        or "TouchShape" in text
        or "decision_capture_id" in text.lower()
    ), (
        "Expected SHACL violation text to mention decisionCaptureId / "
        f"TouchShape; got:\n{text}"
    )
