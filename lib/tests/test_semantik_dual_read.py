"""DART->semantik naming purge, Stage 1 (dual-READ) regression net.

The purge ratified (2026-07-11) ``semantik`` as the target vocabulary:
``semantik:{slug}#{id}`` sourceIds, ``data-semantik-*`` HTML attrs,
``semantik_chunks/`` on-disk chunkset dir, ``chunkset_kind: "semantik"``,
``source_semantik_html_sha256`` / ``semantik_chunks_sha256`` manifest fields.

Stage 1 widens every READER to accept BOTH the legacy ``dart`` form AND the new
``semantik`` form. EMITTERS ARE UNCHANGED this stage. These tests assert the
dual-read: a ``semantik``-form input validates/harvests/resolves EXACTLY like
its legacy ``dart``-form counterpart, and the legacy form is unchanged
(byte-identical acceptance).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# 1. sourceId regexes (item 1) — the canonical anchored patterns.
# ---------------------------------------------------------------------------

def test_source_id_re_accepts_both_prefixes_identically():
    from lib.validators.source_refs import SOURCE_ID_RE

    assert SOURCE_ID_RE.match("dart:photosynthesis#sec_01")
    assert SOURCE_ID_RE.match("semantik:photosynthesis#sec_01")
    # Neither prefix -> still rejected (no widening beyond the two prefixes).
    assert SOURCE_ID_RE.match("imscc:photosynthesis#sec_01") is None
    assert SOURCE_ID_RE.match("photosynthesis#sec_01") is None


def test_inter_tier_gates_source_id_re_dual_read():
    from Courseforge.router.inter_tier_gates import _SOURCE_ID_RE

    assert _SOURCE_ID_RE.match("dart:a#b")
    assert _SOURCE_ID_RE.match("semantik:a#b")


def test_pipeline_tools_canonical_source_id_re_dual_read():
    from MCP.tools.pipeline_tools import _CANONICAL_DART_SOURCE_ID_RE

    assert _CANONICAL_DART_SOURCE_ID_RE.match("dart:a#b")
    assert _CANONICAL_DART_SOURCE_ID_RE.match("semantik:a#b")


def test_chunk_window_slug_capture_dual_read():
    from lib.objectives.chunk_window import _SOURCE_ID_RE

    dart = _SOURCE_ID_RE.match("dart:my-doc#s3_c0")
    sem = _SOURCE_ID_RE.match("semantik:my-doc#s3_c0")
    assert dart and sem
    assert dart.group("slug") == sem.group("slug") == "my-doc"


def test_lo_map_builder_ref_dual_read():
    from lib.objectives.lo_map_builder import _DART_REF_RE

    dart = _DART_REF_RE.match("dart:doc#anchor_1")
    sem = _DART_REF_RE.match("semantik:doc#anchor_1")
    assert dart and sem
    assert (dart.group("src"), dart.group("anchor")) == (
        sem.group("src"),
        sem.group("anchor"),
    )


def test_provenance_resolution_split_token_dual_read():
    from lib.aggregators.provenance_resolution import _split_token

    assert _split_token("dart:stem#anchor") == ("stem", "anchor")
    assert _split_token("semantik:stem#anchor") == ("stem", "anchor")
    assert _split_token("imscc:stem#anchor") is None


def test_answer_render_strip_source_prefix_dual_read():
    from gui.services.answer_render import _strip_source_prefix

    assert _strip_source_prefix("dart:doc#block") == "doc#block"
    assert _strip_source_prefix("semantik:doc#block") == "doc#block"
    assert _strip_source_prefix("other:doc#block") is None


# ---------------------------------------------------------------------------
# 2. Schema pattern (item 3).
# ---------------------------------------------------------------------------

def test_source_reference_schema_accepts_both_prefixes():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        PROJECT_ROOT / "schemas" / "knowledge" / "source_reference.schema.json"
    )
    schema = json.loads(schema_path.read_text())

    for prefix in ("dart", "semantik"):
        ref = {"sourceId": f"{prefix}:photosynthesis#sec_01", "role": "primary"}
        jsonschema.validate(ref, schema)  # must not raise

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"sourceId": "imscc:x#y", "role": "primary"}, schema
        )


# ---------------------------------------------------------------------------
# 3. data-semantik-* HTML-attr harvest (item 2) — identical to data-dart-*.
# ---------------------------------------------------------------------------

def test_harvest_source_refs_semantik_attr_equals_dart_attr():
    from Trainforge.chunker import harvest_dart_source_refs

    dart_html = '<section data-dart-block-id="s3_c0" data-dart-pages="3-5">x</section>'
    semantik_html = (
        '<section data-semantik-block-id="s3_c0" '
        'data-semantik-pages="3-5">x</section>'
    )
    dart_refs = harvest_dart_source_refs(dart_html)
    semantik_refs = harvest_dart_source_refs(semantik_html)

    assert dart_refs == semantik_refs
    assert dart_refs[0]["block_id"] == "s3_c0"
    assert dart_refs[0]["pages"] == [3, 4, 5]


def test_dart_markers_attr_presence_dual_read():
    from lib.validators.dart_markers import (
        _DATA_DART_SOURCE_RE,
        _DATA_DART_BLOCK_ID_RE,
    )

    assert _DATA_DART_SOURCE_RE.search('data-dart-source="claude_llm"')
    assert _DATA_DART_SOURCE_RE.search('data-semantik-source="claude_llm"')
    assert _DATA_DART_BLOCK_ID_RE.search('data-dart-block-id="s0"')
    assert _DATA_DART_BLOCK_ID_RE.search('data-semantik-block-id="s0"')


# ---------------------------------------------------------------------------
# 4. Chunks-dir resolver (item 4) — prefers semantik_chunks over dart_chunks.
# ---------------------------------------------------------------------------

def test_resolve_chunks_dir_prefers_semantik(tmp_path):
    from lib.libv2_storage import (
        resolve_imscc_chunks_dir,
        SEMANTIK_CHUNKS_DIRNAME,
        DART_CHUNKS_DIRNAME,
    )

    course = tmp_path / "course"
    (course / SEMANTIK_CHUNKS_DIRNAME).mkdir(parents=True)
    (course / DART_CHUNKS_DIRNAME).mkdir(parents=True)
    (course / SEMANTIK_CHUNKS_DIRNAME / "chunks.jsonl").write_text("{}\n")
    (course / DART_CHUNKS_DIRNAME / "chunks.jsonl").write_text("{}\n")

    resolved = resolve_imscc_chunks_dir(course, filename="chunks.jsonl")
    assert resolved.name == SEMANTIK_CHUNKS_DIRNAME


def test_resolve_chunks_dir_dart_fallback_byte_identical(tmp_path):
    # No semantik_chunks/ present -> resolves to dart_chunks/ exactly as before
    # (no DeprecationWarning on the dart fallback this stage).
    import warnings

    from lib.libv2_storage import (
        resolve_imscc_chunks_dir,
        DART_CHUNKS_DIRNAME,
    )

    course = tmp_path / "course"
    (course / DART_CHUNKS_DIRNAME).mkdir(parents=True)
    (course / DART_CHUNKS_DIRNAME / "chunks.jsonl").write_text("{}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        resolved = resolve_imscc_chunks_dir(course, filename="chunks.jsonl")
    assert resolved.name == DART_CHUNKS_DIRNAME


# ---------------------------------------------------------------------------
# 5. Chunkset-manifest validator + schema (item 5).
# ---------------------------------------------------------------------------

def test_chunkset_manifest_allows_semantik_kind():
    from lib.validators.chunkset_manifest import (
        _ALLOWED_CHUNKSET_KINDS,
        _CONDITIONAL_SOURCE_FIELD,
    )

    assert "semantik" in _ALLOWED_CHUNKSET_KINDS
    assert "dart" in _ALLOWED_CHUNKSET_KINDS  # legacy still accepted
    assert _CONDITIONAL_SOURCE_FIELD["semantik"] == "source_semantik_html_sha256"


def test_chunkset_manifest_schema_accepts_semantik():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        PROJECT_ROOT / "schemas" / "library" / "chunkset_manifest.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    sha = "a" * 64

    semantik_manifest = {
        "chunks_sha256": sha,
        "chunker_version": "v4",
        "chunkset_kind": "semantik",
        "source_semantik_html_sha256": sha,
    }
    jsonschema.validate(semantik_manifest, schema)  # must not raise

    # A semantik chunkset missing its conditional source-SHA is rejected.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {
                "chunks_sha256": sha,
                "chunker_version": "v4",
                "chunkset_kind": "semantik",
            },
            schema,
        )


# ---------------------------------------------------------------------------
# 6. LibV2 course-manifest validator (item 5) — accepts semantik_chunks_sha256.
# ---------------------------------------------------------------------------

def test_libv2_manifest_accepts_semantik_chunks_sha256(tmp_path):
    from lib.validators.libv2.manifest import LibV2ManifestValidator

    sha = "b" * 64
    course = tmp_path / "course"
    (course / "semantik_chunks").mkdir(parents=True)
    chunks = course / "semantik_chunks" / "chunks.jsonl"
    chunks.write_text("{}\n")

    import hashlib

    real_sha = hashlib.sha256(chunks.read_bytes()).hexdigest()

    # Legacy field name still accepted.
    dart_issues = LibV2ManifestValidator._check_dart_chunks_sha256(
        {"dart_chunks_sha256": real_sha}, course
    )
    # New field name accepted identically.
    sem_issues = LibV2ManifestValidator._check_dart_chunks_sha256(
        {"semantik_chunks_sha256": real_sha}, course
    )
    assert not [i for i in dart_issues if i.severity == "critical"]
    assert not [i for i in sem_issues if i.severity == "critical"]

    # BOTH absent -> the MISSING critical fires.
    missing = LibV2ManifestValidator._check_dart_chunks_sha256({}, course)
    assert any(i.code == "MISSING_DART_CHUNKS_SHA256" for i in missing)


def test_course_manifest_schema_anyof_accepts_either_sha():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        PROJECT_ROOT / "schemas" / "library" / "course_manifest.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    sha = "c" * 64
    base = {
        "libv2_version": "1.0.0",
        "slug": "test-course",
        "import_timestamp": "2026-07-11T00:00:00Z",
        "sourceforge_manifest": {
            "sourceforge_version": "1.0.0",
            "export_timestamp": "2026-07-11T00:00:00Z",
            "course_id": "x",
            "course_title": "X",
        },
        "classification": {"division": "STEM", "primary_domain": "p"},
        "content_profile": {"total_chunks": 1, "total_tokens": 1},
        "imscc_chunks_sha256": sha,
        "concept_graph_sha256": sha,
    }
    # Legacy dart field satisfies the anyOf.
    jsonschema.validate({**base, "dart_chunks_sha256": sha}, schema)
    # New semantik field satisfies the anyOf.
    jsonschema.validate({**base, "semantik_chunks_sha256": sha}, schema)
    # Neither -> anyOf fails.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(base, schema)


# ---------------------------------------------------------------------------
# 7. Phase-name checkpoint alias (item 6).
# ---------------------------------------------------------------------------

def test_checkpoint_load_dual_read_phase_alias(tmp_path):
    from MCP.hardening.checkpoint import CheckpointManager

    mgr = CheckpointManager(tmp_path)
    mgr.start_phase(
        run_id="R1",
        workflow_id="W1",
        phase_name="dart_conversion",
        phase_index=0,
        task_ids=["t1"],
    )
    # A resume that asks under the FUTURE renamed phase key still finds the
    # checkpoint written under the legacy name.
    cp = mgr.load_checkpoint("semantik_conversion")
    assert cp is not None
    assert cp.phase_name == "dart_conversion"

    # And the legacy lookup is unchanged.
    assert mgr.load_checkpoint("dart_conversion") is not None
    # An unrelated phase with no checkpoint still returns None.
    assert mgr.load_checkpoint("packaging") is None


# ---------------------------------------------------------------------------
# 8. Conversion PHASE rename (task #19, Stage 3d):
#    dart_conversion -> semantik_conversion.
#    NEW runs DECLARE/EMIT ``semantik_conversion``; OLD runs (whose persisted
#    phase_outputs are keyed ``dart_conversion``) still RESUME via a dual-read.
# ---------------------------------------------------------------------------

def test_conversion_phase_declared_semantik_in_config():
    """The textbook_to_course conversion phase is EMITTED as
    ``semantik_conversion`` — the legacy ``dart_conversion`` name is gone from
    the declared phase list (readers still accept it; emitters do not)."""
    from MCP.core.workflow_runner import _load_workflows_config

    cfg = _load_workflows_config()
    tbc = cfg["workflows"]["textbook_to_course"]
    names = [p["name"] for p in tbc["phases"]]
    assert "semantik_conversion" in names
    assert "dart_conversion" not in names
    # staging still depends on the (renamed) conversion phase.
    staging = next(p for p in tbc["phases"] if p["name"] == "staging")
    assert "semantik_conversion" in staging["depends_on"]


def _dual_read_runner():
    from unittest.mock import MagicMock

    from MCP.core.config import WorkflowConfig
    from MCP.core.workflow_runner import WorkflowRunner

    return WorkflowRunner(
        executor=MagicMock(),
        config=WorkflowConfig(description="t", phases=[]),
    )


def test_route_params_resolves_legacy_dart_conversion_phase_output():
    """RESUME COMPAT: staging's ``dart_html_paths`` selector now references the
    renamed ``semantik_conversion`` phase output, but an OLD paused run
    persisted its conversion output under the legacy ``dart_conversion`` key.
    ``_route_params`` must still resolve it (dual-read)."""
    runner = _dual_read_runner()
    routed = runner._route_params(
        "staging",
        workflow_params={"run_id": "R1", "course_name": "C1"},
        phase_outputs={"dart_conversion": {"output_paths": "a.html,b.html"}},
    )
    assert routed["dart_html_paths"] == "a.html,b.html"


def test_route_params_resolves_new_semantik_conversion_phase_output():
    """A NEW run keys the conversion output under ``semantik_conversion``; the
    same staging selector resolves it directly."""
    runner = _dual_read_runner()
    routed = runner._route_params(
        "staging",
        workflow_params={"run_id": "R1", "course_name": "C1"},
        phase_outputs={"semantik_conversion": {"output_paths": "a.html,b.html"}},
    )
    assert routed["dart_html_paths"] == "a.html,b.html"


def test_extract_phase_outputs_keyed_under_both_conversion_names():
    """``_extract_phase_outputs`` performs its per-PDF output_paths collection
    for BOTH the declared ``semantik_conversion`` and (dual-read) the legacy
    ``dart_conversion`` phase name."""
    from unittest.mock import MagicMock

    runner = _dual_read_runner()

    def _result(path):
        r = MagicMock()
        r.status = "COMPLETE"
        r.result = {"output_path": path, "html_path": path}
        return r

    results = {"t0": _result("a.html"), "t1": _result("b.html")}
    for phase_name in ("semantik_conversion", "dart_conversion"):
        extracted = runner._extract_phase_outputs(phase_name, results)
        assert extracted["output_paths"] == "a.html,b.html"
        assert extracted["html_paths"] == "a.html,b.html"
