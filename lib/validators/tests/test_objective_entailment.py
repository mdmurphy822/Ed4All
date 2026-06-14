"""W4 — ``ObjectiveEntailmentValidator`` tests.

Covers the per-LO NLI entailment gate at course_planning. Injects a
deterministic fake NLI via the ``nli=...`` test seam.

Test cases (per ``plans/finegrain/w4-nli-grounding-gate.md`` §5.2):

1. LO statement entailed by its cited chunk → PASS.
2. LO statement NOT entailed (fabricated objective with valid-resolving
   chunk_ids) → OBJECTIVE_UNENTAILED, regenerate.
3. LO contradicted → OBJECTIVE_CONTRADICTED.
4. Legacy List[str] / empty refs → SKIPPED; TO empty-refs honors
   require_to_entailment.
5. Graceful-degrade (NLI absent → warning passed=True; strict → raise) +
   no-universe (no manifest → warning OBJECTIVE_NO_GROUNDING_SOURCE).
6. id→text loader resolves by top-level id AND source_references[].sourceId.
7. Decision-capture fires with ≥20-char rationale + dynamic signals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.validators.objective_entailment import (  # noqa: E402
    ObjectiveEntailmentValidator,
    _CODE_NLI_DEPS_MISSING,
    _CODE_OBJECTIVE_CONTRADICTED,
    _CODE_OBJECTIVE_NO_GROUNDING_SOURCE,
    _CODE_OBJECTIVE_UNENTAILED,
    _DECISION_TYPE,
    _load_dart_chunks_text_map,
)


# --------------------------------------------------------------------- #
# Fake NLI + fixture writers
# --------------------------------------------------------------------- #


class _FakeNli:
    _revision = "fake-nli-rev-1"
    device = "cpu"

    def __init__(self) -> None:
        self.batch_calls: List[List[Tuple[str, str]]] = []

    def score_batch(self, *, pairs: List[Tuple[str, str]]) -> List[NliScore]:
        self.batch_calls.append(list(pairs))
        out: List[NliScore] = []
        for _premise, hypothesis in pairs:
            if "[ENTAIL]" in hypothesis:
                out.append(NliScore(entailment=0.95, neutral=0.03, contradiction=0.02))
            elif "[CONTRA]" in hypothesis:
                out.append(NliScore(entailment=0.05, neutral=0.10, contradiction=0.85))
            else:
                out.append(NliScore(entailment=0.10, neutral=0.85, contradiction=0.05))
        return out


class _CaptureSpy:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def _write_dart_chunks(tmp_path: Path, chunks: List[Dict[str, Any]]) -> Path:
    """Write dart_chunks/manifest.json + sibling chunks.jsonl (with bodies)."""
    chunks_dir = tmp_path / "dart_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")
    manifest_path = chunks_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "chunks_sha256": "0" * 64,
            "chunker_version": "v4",
            "chunkset_kind": "dart",
        }),
        encoding="utf-8",
    )
    return manifest_path


def _write_objectives(
    tmp_path: Path, *, terminal: List[Dict[str, Any]], chapter: List[Dict[str, Any]],
) -> Path:
    p = tmp_path / "synthesized_objectives.json"
    p.write_text(
        json.dumps({
            "terminal_objectives": terminal,
            "chapter_objectives": chapter,
        }),
        encoding="utf-8",
    )
    return p


_CHUNK_BODY = (
    "RDF triples are subject-predicate-object statements that form the "
    "foundation of the semantic web data model."
)


def _chunk(chunk_id: str, source_id: str) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "text": _CHUNK_BODY,
        "source": {"source_references": [{"sourceId": source_id}]},
    }


# --------------------------------------------------------------------- #
# Test 1: LO entailed → PASS
# --------------------------------------------------------------------- #


def test_lo_entailed_passes(tmp_path: Path) -> None:
    manifest = _write_dart_chunks(tmp_path, [_chunk("c1", "dart:rdf#s1")])
    objectives = _write_objectives(
        tmp_path, terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "Apply RDF triples to model semantic web data [ENTAIL].",
            "source_refs": [{"ref": "ch1", "chunk_ids": ["dart:rdf#s1"]}],
        }],
    )
    result = ObjectiveEntailmentValidator(nli=_FakeNli()).validate({
        "synthesized_objectives_path": str(objectives),
        "dart_chunks_manifest_path": str(manifest),
    })
    assert result.passed is True
    assert result.action is None
    assert not any(
        i.code in (_CODE_OBJECTIVE_UNENTAILED, _CODE_OBJECTIVE_CONTRADICTED)
        for i in result.issues
    )


# --------------------------------------------------------------------- #
# Test 2: LO unentailed (fabricated, valid-resolving chunk_ids) → FAIL
# --------------------------------------------------------------------- #


def test_lo_unentailed_fails(tmp_path: Path) -> None:
    manifest = _write_dart_chunks(tmp_path, [_chunk("c1", "dart:rdf#s1")])
    objectives = _write_objectives(
        tmp_path, terminal=[],
        chapter=[{
            "id": "CO-01",
            # No [ENTAIL] marker → fake NLI returns low entailment.
            "statement": "Compose interpretive jazz music from RDF triples.",
            "source_refs": [{"ref": "ch1", "chunk_ids": ["dart:rdf#s1"]}],
        }],
    )
    result = ObjectiveEntailmentValidator(nli=_FakeNli()).validate({
        "synthesized_objectives_path": str(objectives),
        "dart_chunks_manifest_path": str(manifest),
    })
    unentailed = [
        i for i in result.issues if i.code == _CODE_OBJECTIVE_UNENTAILED
    ]
    assert len(unentailed) == 1
    assert unentailed[0].severity == "critical"
    assert result.passed is False
    assert result.action == "regenerate"


# --------------------------------------------------------------------- #
# Test 3: LO contradicted
# --------------------------------------------------------------------- #


def test_lo_contradicted(tmp_path: Path) -> None:
    manifest = _write_dart_chunks(tmp_path, [_chunk("c1", "dart:rdf#s1")])
    objectives = _write_objectives(
        tmp_path, terminal=[],
        chapter=[{
            "id": "CO-01",
            "statement": "RDF triples have no subject or predicate at all [CONTRA].",
            "source_refs": [{"ref": "ch1", "chunk_ids": ["dart:rdf#s1"]}],
        }],
    )
    result = ObjectiveEntailmentValidator(nli=_FakeNli()).validate({
        "synthesized_objectives_path": str(objectives),
        "dart_chunks_manifest_path": str(manifest),
    })
    contradicted = [
        i for i in result.issues if i.code == _CODE_OBJECTIVE_CONTRADICTED
    ]
    assert len(contradicted) == 1
    assert result.passed is False
    assert result.action == "regenerate"


# --------------------------------------------------------------------- #
# Test 4: Legacy / empty refs skipped; TO honors require_to_entailment
# --------------------------------------------------------------------- #


def test_legacy_and_empty_refs_skipped(tmp_path: Path) -> None:
    manifest = _write_dart_chunks(tmp_path, [_chunk("c1", "dart:rdf#s1")])
    objectives = _write_objectives(
        tmp_path,
        terminal=[{"id": "TO-01", "statement": "Master RDF.", "source_refs": []}],
        chapter=[
            # Legacy List[str] source_refs → skipped (resolution gate owns it).
            {"id": "CO-01", "statement": "Apply RDF.", "source_refs": ["ch1"]},
            # Empty refs on a CO → skipped (no premise).
            {"id": "CO-02", "statement": "Use RDF.", "source_refs": []},
        ],
    )
    result = ObjectiveEntailmentValidator(nli=_FakeNli()).validate({
        "synthesized_objectives_path": str(objectives),
        "dart_chunks_manifest_path": str(manifest),
    })
    # Nothing was audited (all skipped) → pass, no entailment issues.
    assert result.passed is True
    assert not any(
        i.code in (_CODE_OBJECTIVE_UNENTAILED, _CODE_OBJECTIVE_CONTRADICTED)
        for i in result.issues
    )


# --------------------------------------------------------------------- #
# Test 5: Graceful-degrade
# --------------------------------------------------------------------- #


def test_nli_deps_missing_warns_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.classifiers import nli_classifier as mod

    monkeypatch.setattr(
        mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)

    manifest = _write_dart_chunks(tmp_path, [_chunk("c1", "dart:rdf#s1")])
    objectives = _write_objectives(
        tmp_path, terminal=[],
        chapter=[{
            "id": "CO-01", "statement": "Apply RDF triples.",
            "source_refs": [{"ref": "ch1", "chunk_ids": ["dart:rdf#s1"]}],
        }],
    )
    result = ObjectiveEntailmentValidator().validate({
        "synthesized_objectives_path": str(objectives),
        "dart_chunks_manifest_path": str(manifest),
    })
    assert any(i.code == _CODE_NLI_DEPS_MISSING for i in result.issues)
    assert result.passed is True
    assert result.action is None


def test_strict_mode_missing_deps_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.classifiers import nli_classifier as mod

    monkeypatch.setattr(
        mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    manifest = _write_dart_chunks(tmp_path, [_chunk("c1", "dart:rdf#s1")])
    objectives = _write_objectives(
        tmp_path, terminal=[],
        chapter=[{
            "id": "CO-01", "statement": "Apply RDF triples.",
            "source_refs": [{"ref": "ch1", "chunk_ids": ["dart:rdf#s1"]}],
        }],
    )
    with pytest.raises(RuntimeError) as excinfo:
        ObjectiveEntailmentValidator().validate({
            "synthesized_objectives_path": str(objectives),
            "dart_chunks_manifest_path": str(manifest),
        })
    assert "TRAINFORGE_REQUIRE_EMBEDDINGS" in str(excinfo.value)


def test_no_universe_warns_passes(tmp_path: Path) -> None:
    """No dart_chunks_manifest_path → OBJECTIVE_NO_GROUNDING_SOURCE warning."""
    objectives = _write_objectives(
        tmp_path, terminal=[],
        chapter=[{
            "id": "CO-01", "statement": "Apply RDF triples [ENTAIL].",
            "source_refs": [{"ref": "ch1", "chunk_ids": ["dart:rdf#s1"]}],
        }],
    )
    result = ObjectiveEntailmentValidator(nli=_FakeNli()).validate({
        "synthesized_objectives_path": str(objectives),
    })
    assert any(
        i.code == _CODE_OBJECTIVE_NO_GROUNDING_SOURCE for i in result.issues
    )
    assert result.passed is True
    assert result.action is None


# --------------------------------------------------------------------- #
# Test 6: id→text loader resolves both id forms
# --------------------------------------------------------------------- #


def test_id_text_loader_both_forms(tmp_path: Path) -> None:
    manifest = _write_dart_chunks(
        tmp_path,
        [{
            "id": "course_chunk_00001",
            "text": _CHUNK_BODY,
            "source": {"source_references": [{"sourceId": "dart:rdf#s1"}]},
        }],
    )
    text_map = _load_dart_chunks_text_map(manifest)
    # Resolves by top-level id AND by source_references[].sourceId.
    assert text_map.get("course_chunk_00001") == _CHUNK_BODY
    assert text_map.get("dart:rdf#s1") == _CHUNK_BODY


# --------------------------------------------------------------------- #
# Test 7: Decision-capture fires with dynamic signals
# --------------------------------------------------------------------- #


def test_decision_capture_dynamic_signals(tmp_path: Path) -> None:
    manifest = _write_dart_chunks(tmp_path, [_chunk("c1", "dart:rdf#s1")])
    objectives = _write_objectives(
        tmp_path, terminal=[],
        chapter=[{
            "id": "CO-01", "statement": "Apply RDF triples [ENTAIL].",
            "source_refs": [{"ref": "ch1", "chunk_ids": ["dart:rdf#s1"]}],
        }],
    )
    capture = _CaptureSpy()
    ObjectiveEntailmentValidator(nli=_FakeNli()).validate({
        "synthesized_objectives_path": str(objectives),
        "dart_chunks_manifest_path": str(manifest),
        "decision_capture": capture,
    })
    assert len(capture.events) == 1
    evt = capture.events[0]
    assert evt["decision_type"] == _DECISION_TYPE
    rationale = evt["rationale"]
    assert len(rationale) >= 20
    assert "CO-01" in rationale
    assert "groundedness_rate=" in rationale
    assert "fake-nli-rev-1" in rationale
