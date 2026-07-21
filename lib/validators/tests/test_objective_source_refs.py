"""Wave 1.6 W1.6.C — tests for ``ObjectiveSourceRefValidator``.

Cases covered (mirrors plan §2 Fix 1.6.C):

1. ``test_legacy_string_list_all_resolved`` — legacy ``List[str]`` shape
   with every entry in the union universe → PASS, action=None.
2. ``test_legacy_string_list_one_unresolved`` — legacy ``List[str]``
   shape with one ref unresolved → WARNING
   ``OBJECTIVE_SOURCE_REF_UNRESOLVED``, action=``regenerate``.
3. ``test_structured_all_resolved`` — structured shape, all refs +
   chunk_ids resolved → PASS.
4. ``test_structured_ref_unresolved`` — structured shape, ``ref`` not
   in textbook structure → WARNING
   ``OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE``.
5. ``test_structured_chunk_id_unresolved`` — structured shape, one
   ``chunk_ids[*]`` unresolved → WARNING
   ``OBJECTIVE_CHUNK_NOT_IN_DART_MANIFEST``.
6. ``test_empty_source_refs_on_co`` — empty ``source_refs[]`` on a CO
   → WARNING ``OBJECTIVE_MISSING_SOURCE_REFS``, action=``regenerate``.
7. ``test_empty_source_refs_on_to_silent_pass_default`` — empty
   ``source_refs[]`` on a TO with ``require_to_attribution=False``
   (default) → silent pass.
8. ``test_empty_source_refs_on_to_warn_when_required`` — empty
   ``source_refs[]`` on a TO with ``require_to_attribution=True`` →
   WARNING fires.
9. ``test_no_universes_graceful_degrade`` — both inputs absent →
   graceful-degrade, ``OBJECTIVE_SOURCE_REFS_NO_UNIVERSE`` warning,
   ``passed=True``, ``action=None``.
10. ``test_decision_capture_emits_per_lo`` — per-LO decision-capture
    event fires with the seven required signals.
11. ``test_libv2_archive_form_flattens_identically`` — LibV2-archive
    form (``terminal_outcomes`` / ``component_objectives``) flattens
    identically to the Courseforge form (``terminal_objectives`` /
    ``chapter_objectives``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Repo root on path for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.objective_source_refs import (  # noqa: E402
    ObjectiveSourceRefValidator,
)


# --------------------------------------------------------------------- #
# Stub DecisionCapture — records calls so tests can assert log_decision
# was invoked with the expected decision_type + signals.
# --------------------------------------------------------------------- #


class _StubDecisionCapture:
    """Minimal DecisionCapture stand-in.

    Records every ``log_decision(...)`` invocation in ``self.events``
    so tests can assert which decision types fired and what signals
    landed in the rationale + context.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        context: Any = None,
        alternatives_considered: Any = None,
        **kwargs: Any,
    ) -> None:
        self.events.append({
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
            "context": context,
            "alternatives_considered": alternatives_considered,
            **kwargs,
        })


# --------------------------------------------------------------------- #
# Fixture builders.
# --------------------------------------------------------------------- #


def _write_textbook_structure(tmp_path: Path, *, chapter_ids, section_map=None) -> Path:
    """Write a minimal textbook_structure.json carrying the given IDs.

    ``chapter_ids`` is an iterable of chapter IDs; ``section_map`` is an
    optional dict {chapter_id: [section_id, ...]} to attach sections.
    """
    section_map = section_map or {}
    chapters = []
    for cid in chapter_ids:
        sections = [
            {"id": sid, "headingText": f"Section {sid}"}
            for sid in section_map.get(cid, [])
        ]
        chapters.append({
            "id": cid,
            "headingText": f"Chapter {cid}",
            "sections": sections,
        })
    payload = {
        "course_name": "TEST_COURSE",
        "chapters": chapters,
    }
    p = tmp_path / "textbook_structure.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _write_semantik_chunks(tmp_path: Path, chunks: List[Dict[str, Any]]) -> Path:
    """Write a minimal semantik_chunks/manifest.json + sibling chunks.jsonl.

    Returns the manifest path. The validator harvests chunk-ids from
    the SIBLING chunks.jsonl per the documented contract; the manifest
    itself just needs to exist.
    """
    chunks_dir = tmp_path / "semantik_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = chunks_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")
    manifest_path = chunks_dir / "manifest.json"
    # Minimal manifest payload — the validator's only contract on this
    # file is existence (the parent-dir lookup uses it as the anchor
    # for chunks.jsonl).
    manifest_path.write_text(
        json.dumps({
            "chunks_sha256": "0" * 64,
            "chunker_version": "v4",
            "chunkset_kind": "semantik",
            "source_semantik_html_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )
    return manifest_path


def _write_objectives(
    tmp_path: Path,
    *,
    terminal: List[Dict[str, Any]],
    chapter: List[Dict[str, Any]],
    libv2_form: bool = False,
) -> Path:
    """Write a minimal synthesized_objectives.json.

    When ``libv2_form=True``, emits the LibV2-archive form
    (``terminal_outcomes`` / ``component_objectives``) instead of the
    Courseforge form (``terminal_objectives`` / ``chapter_objectives``)
    so the round-trip-shape test can pin equivalence.
    """
    if libv2_form:
        payload = {
            "terminal_outcomes": terminal,
            "component_objectives": chapter,
        }
    else:
        payload = {
            "terminal_objectives": terminal,
            "chapter_objectives": chapter,
        }
    p = tmp_path / "synthesized_objectives.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# --------------------------------------------------------------------- #
# Cases.
# --------------------------------------------------------------------- #


def test_legacy_string_list_all_resolved(tmp_path: Path) -> None:
    """Case 1: legacy List[str], every entry resolves → PASS."""
    ts = _write_textbook_structure(
        tmp_path,
        chapter_ids=["ch1", "ch2"],
        section_map={"ch1": ["sec1a"]},
    )
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [
            {"id": "course_chunk_00001", "source": {}},
        ],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples.",
            "source_refs": ["ch1", "sec1a"],
        }],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
    })

    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


def test_legacy_string_list_one_unresolved(tmp_path: Path) -> None:
    """Case 2: legacy List[str] with one ref unresolved → WARNING + regenerate."""
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [{"id": "course_chunk_00001", "source": {}}],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples.",
            "source_refs": ["ch1", "ch_DOES_NOT_EXIST"],
        }],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
    })

    # Warning-only severity → passed=True, but action=regenerate.
    assert result.passed is True
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_SOURCE_REF_UNRESOLVED" in codes
    miss = next(
        i for i in result.issues
        if i.code == "OBJECTIVE_SOURCE_REF_UNRESOLVED"
    )
    assert miss.severity == "warning"
    assert "ch_DOES_NOT_EXIST" in miss.message
    assert "CO-01" in miss.location


def test_structured_all_resolved(tmp_path: Path) -> None:
    """Case 3: structured shape, all refs + chunk_ids resolved → PASS."""
    ts = _write_textbook_structure(
        tmp_path,
        chapter_ids=["ch1"],
        section_map={"ch1": ["sec1a"]},
    )
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [
            {
                "id": "course_chunk_00001",
                "source": {
                    "source_references": [
                        {"sourceId": "semantik:rdf11_primer#s4"},
                        {"sourceId": "semantik:rdf11_primer#s5"},
                    ],
                },
            },
        ],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples.",
            "source_refs": [
                {
                    "ref": "ch1",
                    "chunk_ids": [
                        "semantik:rdf11_primer#s4",
                        "semantik:rdf11_primer#s5",
                    ],
                },
            ],
        }],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
    })

    assert result.passed is True
    assert result.action is None
    assert result.issues == []


def test_structured_ref_unresolved(tmp_path: Path) -> None:
    """Case 4: structured shape, ref not in textbook structure → WARNING."""
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [
            {
                "id": "course_chunk_00001",
                "source": {
                    "source_references": [{"sourceId": "semantik:foo#s1"}],
                },
            },
        ],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples.",
            "source_refs": [
                {
                    "ref": "ch_NOT_IN_TEXTBOOK",
                    "chunk_ids": ["semantik:foo#s1"],
                },
            ],
        }],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
    })

    assert result.passed is True
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE" in codes
    miss = next(
        i for i in result.issues
        if i.code == "OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE"
    )
    assert miss.severity == "warning"
    assert "ch_NOT_IN_TEXTBOOK" in miss.message


def test_structured_no_ref_grounded_chunks_passes(tmp_path: Path) -> None:
    """FIX A: a terminal objective's structured ``{chunk_ids}`` entry that omits
    the ``ref`` key (the post-fix shape of ``_annotate_terminals_with_children``)
    PASSES — its grounded chunk_ids resolve and the impossible self-referential
    textbook-universe check is skipped.

    Before FIX A the annotator minted ``ref=<this TO's own id>`` (e.g. TO-01
    citing ref="TO-01"), which can never resolve against the chapter/section
    universe → ``OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE`` fired on EVERY TO.
    The fix omits ``ref`` entirely; this test pins that the no-``ref`` shape is
    accepted while the chunk_ids grounding is still enforced.
    """
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [
            {
                "id": "course_chunk_00001",
                "source": {
                    "source_references": [
                        {"sourceId": "semantik:rdf11_primer#s4"},
                        {"sourceId": "semantik:rdf11_primer#s5"},
                    ],
                },
            },
        ],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[{
            "id": "TO-01",
            "statement": "Master RDF foundations.",
            # FIX A shape: NO self-referential ``ref`` — only grounded chunk_ids.
            "source_refs": [
                {
                    "chunk_ids": [
                        "semantik:rdf11_primer#s4",
                        "semantik:rdf11_primer#s5",
                    ],
                },
            ],
        }],
        chapter=[],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
        "require_to_attribution": True,
    })

    # No self-referential ref → no OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE,
    # and the grounded chunk_ids resolve → clean pass.
    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


def test_structured_no_ref_unresolved_chunk_still_flagged(tmp_path: Path) -> None:
    """FIX A guard: omitting ``ref`` does NOT weaken chunk_ids enforcement — an
    unresolvable chunk_id on a no-``ref`` entry still fires the chunk-miss code.
    """
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [{"id": "course_chunk_00001",
          "source": {"source_references": [{"sourceId": "semantik:rdf11_primer#s4"}]}}],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[{
            "id": "TO-01",
            "statement": "Master RDF foundations.",
            "source_refs": [{"chunk_ids": ["semantik:rdf11_primer#sNOPE"]}],
        }],
        chapter=[],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
        "require_to_attribution": True,
    })

    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_CHUNK_NOT_IN_DART_MANIFEST" in codes
    # No textbook-structure miss — there is no ref to (fail to) resolve.
    assert "OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE" not in codes


def test_structured_chunk_id_unresolved(tmp_path: Path) -> None:
    """Case 5: structured shape, one chunk_id unresolved.

    Still emits the per-LO WARNING ``OBJECTIVE_CHUNK_NOT_IN_DART_MANIFEST``,
    but now ALSO trips the GAP-1 aggregate CRITICAL ``ORPHANED_CITATIONS``
    net (unresolvable rate > 0 with a live chunkset) → passed=False,
    action=block.
    """
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [
            {
                "id": "course_chunk_00001",
                "source": {
                    "source_references": [{"sourceId": "semantik:rdf11_primer#s4"}],
                },
            },
        ],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples.",
            "source_refs": [
                {
                    "ref": "ch1",
                    "chunk_ids": [
                        "semantik:rdf11_primer#s4",        # resolves
                        "semantik:rdf11_primer#sNOPE",     # does not resolve
                    ],
                },
            ],
        }],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
    })

    # GAP-1 net trips on any orphan when a live chunkset is present.
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues]
    # Per-LO warning detail still present.
    assert "OBJECTIVE_CHUNK_NOT_IN_DART_MANIFEST" in codes
    miss = next(
        i for i in result.issues
        if i.code == "OBJECTIVE_CHUNK_NOT_IN_DART_MANIFEST"
    )
    assert miss.severity == "warning"
    assert "semantik:rdf11_primer#sNOPE" in miss.message
    # Aggregate CRITICAL orphan net.
    orphan = next(i for i in result.issues if i.code == "ORPHANED_CITATIONS")
    assert orphan.severity == "critical"
    assert "1 of 2" in orphan.message  # 1 orphaned of 2 cited
    assert "semantik:rdf11_primer#sNOPE" in orphan.message


def test_orphaned_citations_all_renumbered_fires_critical(tmp_path: Path) -> None:
    """GAP 1: chunkset renumbered after synthesis → 100% orphaned citations
    trips CRITICAL ORPHANED_CITATIONS with counts + sample ids.

    This is the exact split-brain scenario the net exists for: the
    objectives cite chunk_ids ``...#s4``/``...#s5`` but the CURRENT
    chunkset was regenerated and now only contains ``...#s40``/``...#s41``,
    so every cited id is orphaned.
    """
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    # Renumbered chunkset — none of the originally-cited ids survive.
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [
            {
                "id": "course_chunk_00099",
                "source": {
                    "source_references": [
                        {"sourceId": "semantik:rdf11_primer#s40"},
                        {"sourceId": "semantik:rdf11_primer#s41"},
                    ],
                },
            },
        ],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[],
        chapter=[
            {
                "id": "CO-01",
                "statement": "Apply RDF triples.",
                "source_refs": [
                    {"ref": "ch1", "chunk_ids": ["semantik:rdf11_primer#s4"]},
                ],
            },
            {
                "id": "CO-02",
                "statement": "Model graphs.",
                "source_refs": [
                    {"ref": "ch1", "chunk_ids": ["semantik:rdf11_primer#s5"]},
                ],
            },
        ],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
    })

    assert result.passed is False
    assert result.action == "block"
    orphan = next(i for i in result.issues if i.code == "ORPHANED_CITATIONS")
    assert orphan.severity == "critical"
    # 2 of 2 cited chunk_ids orphaned → 100%.
    assert "2 of 2" in orphan.message
    assert "100.0%" in orphan.message
    # Both affected objectives counted.
    assert "2 objective(s)" in orphan.message
    # Sample ids present.
    assert "semantik:rdf11_primer#s4" in orphan.message
    assert "semantik:rdf11_primer#s5" in orphan.message


def test_orphaned_citations_missing_chunkset_graceful_skip(tmp_path: Path) -> None:
    """GAP 1: no chunkset input → orphan net skips (graceful, existing
    behavior). With only a textbook_structure universe, chunk_ids can't
    be resolved so the net stays silent — passed=True.
    """
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    objectives = _write_objectives(
        tmp_path,
        terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples.",
            "source_refs": [
                {"ref": "ch1", "chunk_ids": ["semantik:rdf11_primer#s4"]},
            ],
        }],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        # No dart_chunks_manifest_path — chunk_ids universe unavailable.
    })

    codes = [i.code for i in result.issues]
    assert "ORPHANED_CITATIONS" not in codes
    assert result.passed is True


def test_empty_source_refs_on_co(tmp_path: Path) -> None:
    """Case 6: empty source_refs on CO → WARNING + regenerate."""
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [{"id": "c1", "source": {}}],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples.",
            "source_refs": [],
        }],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
    })

    assert result.passed is True
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_MISSING_SOURCE_REFS" in codes


def test_empty_source_refs_on_to_silent_pass_default(tmp_path: Path) -> None:
    """Case 7: empty source_refs on TO + require_to_attribution=False → silent pass."""
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [{"id": "c1", "source": {}}],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[{
            "id": "TO-01",
            "statement": "Build accessible web content.",
            # source_refs absent → empty shape.
        }],
        chapter=[],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
        # require_to_attribution defaults to False.
    })

    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


def test_empty_source_refs_on_to_warn_when_required(tmp_path: Path) -> None:
    """Case 8: empty source_refs on TO + require_to_attribution=True → WARNING."""
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [{"id": "c1", "source": {}}],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[{
            "id": "TO-01",
            "statement": "Build accessible web content.",
            "source_refs": [],
        }],
        chapter=[],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
        "require_to_attribution": True,
    })

    assert result.passed is True
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_MISSING_SOURCE_REFS" in codes
    miss = next(
        i for i in result.issues
        if i.code == "OBJECTIVE_MISSING_SOURCE_REFS"
    )
    assert miss.severity == "warning"
    assert "TO-01" in miss.location


def test_no_universes_graceful_degrade(tmp_path: Path) -> None:
    """Case 9: both inputs absent → graceful-degrade warning, passed=True."""
    objectives = _write_objectives(
        tmp_path,
        terminal=[{
            "id": "TO-01",
            "statement": "Build accessible web content.",
            "source_refs": ["ch1"],
        }],
        chapter=[],
    )

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        # No textbook_structure_path, no dart_chunks_manifest_path.
    })

    assert result.passed is True
    assert result.action is None
    assert len(result.issues) == 1
    assert result.issues[0].code == "OBJECTIVE_SOURCE_REFS_NO_UNIVERSE"
    assert result.issues[0].severity == "warning"


def test_decision_capture_emits_per_lo(tmp_path: Path) -> None:
    """Case 10: per-LO decision-capture event with the seven required signals."""
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [
            {
                "id": "c1",
                "source": {
                    "source_references": [{"sourceId": "semantik:foo#s1"}],
                },
            },
        ],
    )
    objectives = _write_objectives(
        tmp_path,
        terminal=[{
            "id": "TO-01",
            "statement": "Build it.",
            "source_refs": [
                {"ref": "ch1", "chunk_ids": ["semantik:foo#s1"]},
            ],
        }],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples.",
            "source_refs": ["ch1"],
        }],
    )
    capture = _StubDecisionCapture()

    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
        "decision_capture": capture,
    })

    assert result.passed is True
    # One event per LO; both are objective_source_ref_check.
    assert len(capture.events) == 2
    types = {e["decision_type"] for e in capture.events}
    assert types == {"objective_source_ref_check"}

    # Pull the TO-01 event and confirm the seven required signals
    # appear in the rationale + context.
    to_event = next(
        e for e in capture.events if "TO-01" in (e.get("context") or "")
    )
    rationale = to_event["rationale"]
    context = to_event["context"]
    # Rationale must be ≥ 20 chars per project decision-capture contract.
    assert len(rationale) >= 20
    # Each of the seven required signals must surface in the context.
    for needle in (
        "lo_id=TO-01",
        "hierarchy_level=terminal",
        "declared_refs_count=",
        "unresolved_refs_count=",
        "declared_chunk_ids_count=",
        "unresolved_chunk_ids_count=",
        "attribution_shape=structured_per_ref",
    ):
        assert needle in context, (
            f"signal {needle!r} missing from context: {context!r}"
        )

    # CO event uses the legacy_string_list shape — confirm it.
    co_event = next(
        e for e in capture.events if "CO-01" in (e.get("context") or "")
    )
    assert "hierarchy_level=chapter" in co_event["context"]
    assert "attribution_shape=legacy_string_list" in co_event["context"]


def test_libv2_archive_form_flattens_identically(tmp_path: Path) -> None:
    """Case 11: LibV2-archive form flattens identically to Courseforge form.

    Asserts that an LO set written under ``terminal_outcomes`` /
    ``component_objectives`` (LibV2 archive) is audited identically to
    the same payload written under ``terminal_objectives`` /
    ``chapter_objectives`` (Courseforge synthesized form). Same gate
    result on both surfaces — proves
    ``_flatten_objectives`` reuse is shape-tolerant.
    """
    ts = _write_textbook_structure(tmp_path, chapter_ids=["ch1"])
    chunks_manifest = _write_semantik_chunks(
        tmp_path,
        [{"id": "c1", "source": {}}],
    )
    chapter_payload = [{
        "id": "CO-01",
        "statement": "Apply RDF triples.",
        "source_refs": ["ch1"],
    }]
    terminal_payload = [{
        "id": "TO-01",
        "statement": "Build accessible web content.",
    }]

    cf_path = tmp_path / "cf.json"
    cf_path.write_text(json.dumps({
        "terminal_objectives": terminal_payload,
        "chapter_objectives": chapter_payload,
    }), encoding="utf-8")

    libv2_path = tmp_path / "libv2.json"
    libv2_path.write_text(json.dumps({
        "terminal_outcomes": terminal_payload,
        "component_objectives": chapter_payload,
    }), encoding="utf-8")

    common_inputs = {
        "textbook_structure_path": str(ts),
        "dart_chunks_manifest_path": str(chunks_manifest),
    }

    cf_result = ObjectiveSourceRefValidator().validate({
        **common_inputs,
        "synthesized_objectives_path": str(cf_path),
    })
    libv2_result = ObjectiveSourceRefValidator().validate({
        **common_inputs,
        "synthesized_objectives_path": str(libv2_path),
    })

    # Same passed / action / issue codes on both surfaces.
    assert cf_result.passed == libv2_result.passed
    assert cf_result.action == libv2_result.action
    assert (
        sorted(i.code for i in cf_result.issues)
        == sorted(i.code for i in libv2_result.issues)
    )
    # And both must be the no-issue happy path (the chapter-level CO-01
    # resolves against ch1; TO-01 is silently passed by the
    # require_to_attribution=False default).
    assert cf_result.passed is True
    assert cf_result.action is None
    assert cf_result.issues == []
