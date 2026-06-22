"""SemantiK migration §4.5 item 5 — end-to-end chunk-accompanying-data
harvest regression (the m5 gate).

Asserts the six chunk-root enrichment fields survive HTML -> chunk, built on a
REAL adapter-output HTML fixture (``lib/semantik/adapter.normalize_cascade_to_ed4all``
run over the P2a synthetic cascade fixture):

  * ``source_block_role`` / ``source_block_confidence`` / ``wcag_block_status``
    survive the HTML -> chunk boundary via the new same-element harvest regexes
    (``Trainforge/chunker/helpers.py`` :: ``_DATA_DART_BLOCK_ROLE_RE`` /
    ``_DATA_DART_CONFIDENCE_RE`` / ``_DATA_DART_WCAG_RE``).
  * ``semantic_preservation_score`` / ``certification_status`` come through from
    the DOC-LEVEL signal stamped on the CourseProcessor instance (mirrors
    ``source_document_sha256``).
  * ``figure_alt`` is recovered DOM-side for the figure chunk.

Plus a back-compat assertion: a LEGACY HTML fixture WITHOUT the new attrs (and
WITHOUT the doc-level signals) still chunks, and NONE of the six fields appear.

Built + run with NO models / GPU:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest Trainforge/chunker/tests/test_semantik_chunk_enrichment.py -q
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from Trainforge.chunker import ChunkerContext, chunk_content
from Trainforge.chunker.helpers import harvest_dart_source_refs
from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)


# ---------------------------------------------------------------------------
# Real adapter-output HTML fixture (§4.5 item 5: run the adapter's normalizer
# on a synthetic cascade, then chunk the result).
# ---------------------------------------------------------------------------


class _SyntheticCascade:
    """Duck-typed PipelineV2Result + normalized IR (mirrors the P2a fixture).

    Two chapters; one figure block carrying alt-text; banded confidences; per-
    block role + wcag; a headingless block (positional sid, omitted role/wcag);
    a non-content answer-key heading that the adapter filters.
    """

    def __init__(self, *, exit_action: str = "ship_with_flag"):
        self.pdf = "sample_text_ch1.pdf"
        self.wcag_status = "passed"
        self.exit_action = exit_action
        self.theta_score = 0.91
        self.flags = ["theta_low"]
        self.lane_used = "fast-lane"
        self.lang = "en"
        self.chapters = [
            _AdapterChapter(
                title="Chapter 1: Foundations",
                blocks=[
                    _AdapterBlock(
                        html="<p>" + ("Algebra builds on arithmetic. " * 30) + "</p>",
                        region_kind="paragraph",
                        raw_block_index=0,
                        raw_text="Algebra builds on arithmetic. " * 30,
                        heading_text="Introduction to Algebra",
                        pages=[1],
                        confidence=0.83,  # bands to 0.8
                        block_role="body",
                        wcag_status="passed",
                    ),
                    _AdapterBlock(
                        html=(
                            '<figure><img src="fig1.png" '
                            'alt="A number line from -5 to 5">'
                            "<figcaption>Figure 1.1 The number line "
                            "from negative five to five.</figcaption></figure>"
                            "<p>" + ("The number line orders integers. " * 25)
                            + "</p>"
                        ),
                        region_kind="figure",
                        raw_block_index=2,
                        raw_text=(
                            "Figure 1.1 The number line. "
                            + ("The number line orders integers. " * 25)
                        ),
                        heading_text="Number Line",
                        pages=[3],
                        confidence=1.0,  # bands to omitted
                        block_role="figure",
                        wcag_status="flagged",
                        figure_alt="A number line from -5 to 5",
                    ),
                ],
            ),
            _AdapterChapter(
                title="Chapter 2: Linear Equations",
                blocks=[
                    _AdapterBlock(
                        html="<p>" + ("Solve for x in 2x + 3 = 7. " * 30) + "</p>",
                        region_kind="paragraph",
                        raw_block_index=10,
                        raw_text="Solve for x in 2x + 3 = 7. " * 30,
                        heading_text="Solving Linear Equations",
                        pages=[5],
                        confidence=0.61,  # bands to 0.6
                        block_role="body",
                        wcag_status="passed",
                    ),
                ],
            ),
        ]


def _adapter_html(exit_action: str = "ship_with_flag") -> Dict[str, Any]:
    return normalize_cascade_to_ed4all(
        _SyntheticCascade(exit_action=exit_action), pdf_stem="sample_text_ch1"
    )


# ---------------------------------------------------------------------------
# CourseProcessor-backed _create_chunk (bypasses __init__, mirrors
# Trainforge/tests/test_provenance.py).
# ---------------------------------------------------------------------------


def _make_processor(
    *,
    semantic_preservation_score: float | None = None,
    certification_status: str | None = None,
):
    from Trainforge.process_course import CourseProcessor

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "SEMANTIK_TEST"
    proc.MAX_CHUNK_SIZE = CourseProcessor.MAX_CHUNK_SIZE
    proc.TARGET_CHUNK_SIZE = CourseProcessor.TARGET_CHUNK_SIZE
    proc.MIN_CHUNK_SIZE = CourseProcessor.MIN_CHUNK_SIZE
    proc._all_concept_tags = set()
    proc.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": defaultdict(int),
        "difficulty_distribution": defaultdict(int),
    }
    # Doc-level signals (the P3 seam sets these from the sidecar /
    # PipelineV2Result). Omit-when-absent on a legacy run.
    if semantic_preservation_score is not None:
        proc._semantic_preservation_score = semantic_preservation_score
    if certification_status is not None:
        proc._certification_status = certification_status
    # Stub the __init__-dependent extraction helpers.
    proc._extract_concept_tags = lambda t, i: []  # type: ignore[method-assign]
    proc._determine_difficulty = lambda t, i: "foundational"  # type: ignore[method-assign]
    proc._extract_objective_refs = lambda i, **kw: []  # type: ignore[method-assign]
    proc._extract_section_metadata = (  # type: ignore[method-assign]
        lambda i, h, **kw: (
            None, None, [], {"key_claims": [], "objective_alignment": []},
            {
                "content_type_label": "none",
                "key_terms": "none",
                "key_claims": "none",
                "objective_alignment": "none",
            },
        )
    )
    return proc


def _item(raw_html: str, *, chunk_type_hint: str = "explanation") -> Dict[str, Any]:
    return {
        "module_id": "mod1",
        "module_title": "Mod 1",
        "item_id": "sample_text_ch1",
        "title": "OpenStax Algebra Ch.1",
        "resource_type": "page",
        "item_path": "sample_text_ch1.html",
        "raw_html": raw_html,
        "sections": [],
        "learning_objectives": [],
        "courseforge_metadata": None,
        "misconceptions": [],
    }


def _chunk_real_adapter_html(
    out: Dict[str, Any],
    *,
    semantic_preservation_score: float | None = None,
    certification_status: str | None = None,
) -> List[Dict[str, Any]]:
    proc = _make_processor(
        semantic_preservation_score=semantic_preservation_score,
        certification_status=certification_status,
    )
    result = chunk_content(
        [_item(out["html"])],
        "SEMANTIK_TEST",
        ctx=ChunkerContext(create_chunk=proc._create_chunk),
    )
    return result.chunks


# ---------------------------------------------------------------------------
# (m5) End-to-end harvest — all six fields survive HTML -> chunk.
# ---------------------------------------------------------------------------


def test_html_harvested_fields_survive_to_chunk():
    """source_block_role / source_block_confidence / wcag_block_status survive
    the HTML -> chunk boundary via the new same-element harvest regexes."""
    out = _adapter_html()
    chunks = _chunk_real_adapter_html(
        out,
        semantic_preservation_score=0.91,
        certification_status="flagged",
    )
    assert chunks, "adapter HTML produced no chunks"

    # At least one body chunk carries the harvested role/confidence/wcag.
    roles = {c.get("source_block_role") for c in chunks if "source_block_role" in c}
    assert "body" in roles, f"no body role harvested onto a chunk: {roles}"

    confidences = [
        c["source_block_confidence"]
        for c in chunks
        if "source_block_confidence" in c
    ]
    assert confidences, "no source_block_confidence harvested onto any chunk"
    assert all(0.0 <= v <= 1.0 for v in confidences)
    # The banded 0.8 / 0.6 values come through (the 1.0 figure is omitted).
    assert 0.8 in confidences or 0.6 in confidences

    wcag_values = {
        c["wcag_block_status"] for c in chunks if "wcag_block_status" in c
    }
    assert wcag_values, "no wcag_block_status harvested onto any chunk"
    assert wcag_values <= {"passed", "flagged", "skipped"}


def test_doc_level_signals_stamped_on_every_chunk():
    """semantic_preservation_score + certification_status come through from the
    doc-level instance signal — same value on EVERY chunk (mirrors
    source_document_sha256)."""
    out = _adapter_html(exit_action="ship_with_flag")
    chunks = _chunk_real_adapter_html(
        out,
        semantic_preservation_score=0.91,
        certification_status="flagged",
    )
    assert chunks
    for c in chunks:
        assert c.get("semantic_preservation_score") == 0.91
        assert c.get("certification_status") == "flagged"


def test_figure_alt_recovered_for_figure_chunk():
    """figure_alt is recovered DOM-side for the figure chunk."""
    out = _adapter_html()
    chunks = _chunk_real_adapter_html(out)
    alts = [c["figure_alt"] for c in chunks if c.get("figure_alt")]
    assert alts, "no figure_alt recovered from any chunk"
    # figcaption text is preferred over the <img alt>.
    assert any("number line" in a.lower() for a in alts)


def test_all_six_fields_present_on_real_adapter_corpus():
    """The full m5 gate: every one of the six §4 fields appears at least once
    across the chunks of a real adapter-output document."""
    out = _adapter_html()
    chunks = _chunk_real_adapter_html(
        out,
        semantic_preservation_score=0.91,
        certification_status="flagged",
    )
    present: set = set()
    for c in chunks:
        for field in (
            "source_block_role",
            "source_block_confidence",
            "wcag_block_status",
            "figure_alt",
            "semantic_preservation_score",
            "certification_status",
        ):
            if field in c:
                present.add(field)
    missing = {
        "source_block_role",
        "source_block_confidence",
        "wcag_block_status",
        "figure_alt",
        "semantic_preservation_score",
        "certification_status",
    } - present
    assert not missing, f"these §4 fields never survived to a chunk: {missing}"


# ---------------------------------------------------------------------------
# Back-compat: legacy HTML WITHOUT the new attrs (and no doc-level signal)
# still chunks, fields simply absent.
# ---------------------------------------------------------------------------


_LEGACY_HTML = (
    "<html><body>"
    '<section class="dart-section" data-dart-block-id="s1" data-dart-pages="4">'
    "<h3>Legacy Section</h3>"
    "<p>" + ("Ordinary legacy prose with no SemantiK enrichment. " * 30) + "</p>"
    "</section>"
    "</body></html>"
)


def test_legacy_html_chunks_without_the_six_fields():
    """A legacy DART HTML fixture WITHOUT the new attrs (and with no doc-level
    signals on the instance) still chunks; NONE of the six §4 fields appear."""
    # The legacy HTML harvests block-id + pages, but NONE of the enrichment.
    refs = harvest_dart_source_refs(_LEGACY_HTML)
    assert refs and refs[0]["block_id"] == "s1"
    assert "block_role" not in refs[0]
    assert "confidence" not in refs[0]
    assert "wcag_status" not in refs[0]

    proc = _make_processor()  # no doc-level signals
    result = chunk_content(
        [_item(_LEGACY_HTML)],
        "SEMANTIK_TEST",
        ctx=ChunkerContext(create_chunk=proc._create_chunk),
    )
    assert result.chunks, "legacy HTML produced no chunks"
    for c in result.chunks:
        for field in (
            "source_block_role",
            "source_block_confidence",
            "wcag_block_status",
            "figure_alt",
            "semantic_preservation_score",
            "certification_status",
        ):
            assert field not in c, f"legacy chunk leaked {field}"


# ---------------------------------------------------------------------------
# Schema: a chunk with AND without the six fields validates against chunk_v4.
# ---------------------------------------------------------------------------


def _build_chunk_validator():
    """Validator over chunk_v4 with all sibling schemas resolvable offline.

    Mirrors ``schemas/tests/test_chunk_source_references.py::_build_validator``
    (RefResolver + a ``$id`` -> content store globbed from ``schemas/``) — the
    canonical project pattern for validating against chunk_v4's taxonomy
    ``$ref``s without network access.
    """
    jsonschema = pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator, RefResolver

    schemas_dir = PROJECT_ROOT / "schemas"
    schema = json.loads(
        (schemas_dir / "knowledge" / "chunk_v4.schema.json").read_text(
            encoding="utf-8"
        )
    )
    store: Dict[str, Any] = {}
    for p in schemas_dir.rglob("*.json"):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = doc.get("$id") if isinstance(doc, dict) else None
        if sid:
            store[sid] = doc
    resolver = RefResolver.from_schema(schema, store=store)
    return jsonschema, Draft202012Validator(schema, resolver=resolver)


def test_enriched_and_legacy_chunks_validate_against_chunk_v4():
    jsonschema, validator = _build_chunk_validator()

    # Build a minimal valid chunk, then add / omit the six fields.
    base = {
        "id": "semantik_test_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "x",
        "html": "<p>x</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "SEMANTIK_TEST",
            "module_id": "m1",
            "lesson_id": "l1",
        },
        "concept_tags": [],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 1,
        "word_count": 1,
        "bloom_level": "understand",
    }

    # (1) legacy chunk (no §4 fields) validates.
    validator.validate(base)

    # (2) fully-enriched chunk validates.
    enriched = dict(base)
    enriched["id"] = "semantik_test_chunk_00002"
    enriched["source_block_role"] = "body"
    enriched["source_block_confidence"] = 0.8
    enriched["wcag_block_status"] = "flagged"
    enriched["figure_alt"] = "A number line from -5 to 5"
    enriched["semantic_preservation_score"] = 0.91
    enriched["certification_status"] = "non_certified"
    validator.validate(enriched)

    # (3) an out-of-enum certification_status is rejected (the enum bites).
    bad = dict(enriched)
    bad["certification_status"] = "bogus_status"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)
