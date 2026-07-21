"""Write-time citation SANITIZER invariants.

``lib/objectives/citation_sanitize.py::sanitize_synthesized_objectives`` drops
every ``source_refs`` / ``source_chunk_ids`` citation that does not resolve
against the current chunkset / textbook-structure universe (the fabricated
topic-label / statement-echo citations the local synthesizer emits), converting
a would-be ``objective_source_refs`` ORPHANED_CITATIONS critical block into the
benign no-grounding warning path. Hermetic — no torch/numpy/LLM/network.

The integration case runs ``ObjectiveSourceRefValidator`` on a fabricated-id
objectives doc (asserting it BLOCKS), sanitizes the doc, then re-validates
(asserting it PASSES) — proving the sanitizer unblocks the gate.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.objectives.citation_sanitize import (  # noqa: E402
    ENV_SANITIZE_CITATIONS,
    build_chunk_universe,
    resolve_sanitize_citations,
    sanitize_synthesized_objectives,
)
from lib.validators.objective_source_refs import (  # noqa: E402
    ObjectiveSourceRefValidator,
)


# --------------------------------------------------------------------------- #
# Universe fixtures.
# --------------------------------------------------------------------------- #
_REAL_CHUNKS: List[Dict[str, Any]] = [
    {
        "id": "test_course_ch01_chunk_00001",
        "source": {
            "source_references": [
                {"sourceId": "semantik:test-course#s1"},
            ]
        },
    },
    {
        "id": "test_course_ch01_chunk_00002",
        "source": {},
    },
]
_CHUNK_UNIVERSE = build_chunk_universe(_REAL_CHUNKS)
_STRUCTURE_UNIVERSE = {"ch01", "sec01a"}


def _real_chunk_id() -> str:
    return "test_course_ch01_chunk_00001"


def _real_source_id() -> str:
    return "semantik:test-course#s1"


# --------------------------------------------------------------------------- #
# build_chunk_universe.
# --------------------------------------------------------------------------- #
def test_build_chunk_universe_harvests_id_and_sourceid() -> None:
    assert _real_chunk_id() in _CHUNK_UNIVERSE
    assert "test_course_ch01_chunk_00002" in _CHUNK_UNIVERSE
    assert _real_source_id() in _CHUNK_UNIVERSE
    assert len(_CHUNK_UNIVERSE) == 3


# --------------------------------------------------------------------------- #
# resolve_sanitize_citations — default ON, opt-out only.
# --------------------------------------------------------------------------- #
def test_resolver_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_SANITIZE_CITATIONS, raising=False)
    assert resolve_sanitize_citations() is True


@pytest.mark.parametrize("token", ["0", "false", "no", "off", "OFF", "False"])
def test_resolver_opt_out(monkeypatch: pytest.MonkeyPatch, token: str) -> None:
    monkeypatch.setenv(ENV_SANITIZE_CITATIONS, token)
    assert resolve_sanitize_citations() is False


@pytest.mark.parametrize("token", ["1", "true", "on", "garbage", ""])
def test_resolver_non_falsey_stays_on(
    monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv(ENV_SANITIZE_CITATIONS, token)
    assert resolve_sanitize_citations() is True


def test_explicit_arg_wins() -> None:
    assert resolve_sanitize_citations(False) is False
    assert resolve_sanitize_citations(True) is True


# --------------------------------------------------------------------------- #
# Structured chunk_ids sanitation.
# --------------------------------------------------------------------------- #
def test_structured_fabricated_chunk_id_dropped_real_kept() -> None:
    doc = {
        "chapter_objectives": [{
            "id": "CO-01",
            "statement": "Round whole numbers.",
            "source_refs": [{
                "ref": "ch01",
                "chunk_ids": [_real_chunk_id(), "Round Whole Numbers"],
            }],
            "source_chunk_ids": [_real_chunk_id(), "Round Whole Numbers"],
        }],
    }
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=True,
    )
    assert result.available is True
    assert result.los_mutated == 1
    assert result.structured_chunk_ids_dropped == 1
    assert result.flat_chunk_ids_dropped == 1
    co = doc["chapter_objectives"][0]
    assert co["source_refs"][0]["chunk_ids"] == [_real_chunk_id()]
    assert co["source_chunk_ids"] == [_real_chunk_id()]


def test_fully_fabricated_structured_entry_dropped() -> None:
    doc = {
        "terminal_objectives": [{
            "id": "TO-01",
            "statement": "Use place value to round.",
            "source_refs": [{
                "ref": "Order of Operations",          # not in structure
                "chunk_ids": ["Order of Operations"],  # not in chunk universe
            }],
        }],
    }
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=True,
    )
    assert result.entries_dropped == 1
    assert doc["terminal_objectives"][0]["source_refs"] == []


def test_valid_ref_keeps_entry_with_emptied_chunk_ids() -> None:
    doc = {
        "chapter_objectives": [{
            "id": "CO-01",
            "statement": "x",
            "source_refs": [{"ref": "ch01", "chunk_ids": ["Divisibility Tests"]}],
        }],
    }
    sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=True,
    )
    # ref resolves → entry retained, chunk_ids emptied (no orphan, ref warns only).
    assert doc["chapter_objectives"][0]["source_refs"] == [
        {"ref": "ch01", "chunk_ids": []}
    ]


def test_sourceid_form_resolves() -> None:
    doc = {
        "chapter_objectives": [{
            "id": "CO-01",
            "statement": "x",
            "source_refs": [{"ref": "ch01", "chunk_ids": [_real_source_id()]}],
        }],
    }
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=True,
    )
    assert result.los_mutated == 0
    assert doc["chapter_objectives"][0]["source_refs"][0]["chunk_ids"] == [
        _real_source_id()
    ]


# --------------------------------------------------------------------------- #
# Legacy List[str] sanitation.
# --------------------------------------------------------------------------- #
def test_legacy_string_fabricated_dropped_real_kept() -> None:
    doc = {
        "terminal_objectives": [{
            "id": "TO-01",
            "statement": "Use place value to round.",
            "source_refs": [
                "ch01",                          # resolves (structure)
                _real_chunk_id(),                # resolves (chunk)
                "Use place value to round...",   # fabricated statement-echo
            ],
        }],
    }
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=True,
    )
    assert result.legacy_strings_dropped == 1
    assert doc["terminal_objectives"][0]["source_refs"] == [
        "ch01", _real_chunk_id()
    ]


def test_legacy_all_fabricated_becomes_empty() -> None:
    doc = {
        "chapter_objectives": [{
            "id": "CO-01",
            "statement": "x",
            "source_refs": ["Round Whole Numbers", "Order of Operations"],
        }],
    }
    sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=True,
    )
    assert doc["chapter_objectives"][0]["source_refs"] == []


# --------------------------------------------------------------------------- #
# Group-of-groups chapter_objectives shape.
# --------------------------------------------------------------------------- #
def test_group_of_groups_shape_handled() -> None:
    doc = {
        "chapter_objectives": [{
            "chapter_title": "Week 1",
            "objectives": [{
                "id": "CO-01",
                "statement": "x",
                "source_chunk_ids": [_real_chunk_id(), "Fabricated Label"],
            }],
        }],
    }
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=True,
    )
    assert result.flat_chunk_ids_dropped == 1
    inner = doc["chapter_objectives"][0]["objectives"][0]
    assert inner["source_chunk_ids"] == [_real_chunk_id()]


# --------------------------------------------------------------------------- #
# No-op guards (byte-identical).
# --------------------------------------------------------------------------- #
def test_disabled_is_byte_identical() -> None:
    doc = {
        "chapter_objectives": [{
            "id": "CO-01",
            "source_refs": [{"ref": "ch01", "chunk_ids": ["Fabricated"]}],
            "source_chunk_ids": ["Fabricated"],
        }],
    }
    before = copy.deepcopy(doc)
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=False,
    )
    assert result.available is False
    assert doc == before


def test_empty_universe_is_noop() -> None:
    doc = {
        "chapter_objectives": [{
            "id": "CO-01",
            "source_refs": [{"ref": "anything", "chunk_ids": ["anything"]}],
            "source_chunk_ids": ["anything"],
        }],
    }
    before = copy.deepcopy(doc)
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=set(),
        structure_universe=set(),
        enabled=True,
    )
    assert result.available is False
    assert doc == before


def test_healthy_corpus_is_byte_identical() -> None:
    doc = {
        "terminal_objectives": [{
            "id": "TO-01",
            "source_refs": [{"ref": "ch01", "chunk_ids": [_real_chunk_id()]}],
        }],
        "chapter_objectives": [{
            "id": "CO-01",
            "source_refs": ["ch01"],
            "source_chunk_ids": [_real_chunk_id()],
        }],
    }
    before = copy.deepcopy(doc)
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        enabled=True,
    )
    assert result.available is True
    assert result.los_mutated == 0
    assert doc == before


# --------------------------------------------------------------------------- #
# Decision capture.
# --------------------------------------------------------------------------- #
class _StubCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def test_capture_fires_on_mutation() -> None:
    cap = _StubCapture()
    doc = {
        "chapter_objectives": [{
            "id": "CO-01",
            "source_chunk_ids": [_real_chunk_id(), "Fabricated"],
        }],
    }
    sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        capture=cap,
        enabled=True,
    )
    assert len(cap.events) == 1
    ev = cap.events[0]
    assert ev["decision_type"] == "objective_chunk_prune"
    assert len(ev["rationale"]) >= 20


def test_no_capture_when_nothing_dropped() -> None:
    cap = _StubCapture()
    doc = {
        "chapter_objectives": [{
            "id": "CO-01",
            "source_chunk_ids": [_real_chunk_id()],
        }],
    }
    sanitize_synthesized_objectives(
        doc,
        chunk_universe=_CHUNK_UNIVERSE,
        structure_universe=_STRUCTURE_UNIVERSE,
        capture=cap,
        enabled=True,
    )
    assert cap.events == []


# --------------------------------------------------------------------------- #
# Integration: sanitizer unblocks ObjectiveSourceRefValidator.
# --------------------------------------------------------------------------- #
def _write_chunks(tmp_path: Path) -> Path:
    chunks_dir = tmp_path / "semantik_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in _REAL_CHUNKS) + "\n",
        encoding="utf-8",
    )
    manifest = chunks_dir / "manifest.json"
    manifest.write_text(
        json.dumps({
            "chunks_sha256": "0" * 64,
            "chunker_version": "v4",
            "chunkset_kind": "semantik",
        }),
        encoding="utf-8",
    )
    return manifest


def _write_structure(tmp_path: Path) -> Path:
    p = tmp_path / "textbook_structure.json"
    p.write_text(
        json.dumps({
            "chapters": [{
                "id": "ch01",
                "sections": [{"id": "sec01a"}],
            }],
        }),
        encoding="utf-8",
    )
    return p


def _fabricated_doc() -> Dict[str, Any]:
    return {
        "terminal_objectives": [{
            "id": "TO-01",
            "statement": "Use place value to round whole numbers.",
            "source_refs": [{
                "ref": "Round Whole Numbers",
                "chunk_ids": ["Round Whole Numbers", "Order of Operations"],
            }],
        }],
        "chapter_objectives": [{
            "id": "CO-01",
            "statement": "Round a whole number to a given place.",
            "source_refs": [{
                "ref": "ch01",
                "chunk_ids": ["Divisibility Tests"],
            }],
        }],
    }


def test_integration_sanitizer_unblocks_gate(tmp_path: Path) -> None:
    manifest = _write_chunks(tmp_path)
    structure = _write_structure(tmp_path)

    # 1. Fabricated citations → gate BLOCKS (ORPHANED_CITATIONS critical).
    doc = _fabricated_doc()
    obj_path = tmp_path / "synthesized_objectives.json"
    obj_path.write_text(json.dumps(doc), encoding="utf-8")
    pre = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(obj_path),
        "textbook_structure_path": str(structure),
        "dart_chunks_manifest_path": str(manifest),
    })
    assert pre.passed is False
    assert any(i.code == "ORPHANED_CITATIONS" for i in pre.issues)

    # 2. Sanitize the SAME doc against the SAME universe the gate uses.
    chunk_universe = build_chunk_universe(_REAL_CHUNKS)
    from lib.validators.objective_source_refs import (
        _load_textbook_structure_universe,
    )
    structure_universe = _load_textbook_structure_universe(structure)
    result = sanitize_synthesized_objectives(
        doc,
        chunk_universe=chunk_universe,
        structure_universe=structure_universe,
        enabled=True,
    )
    assert result.available is True
    assert result.los_mutated == 2

    # 3. Re-validate the sanitized doc → gate PASSES (no critical).
    obj_path.write_text(json.dumps(doc), encoding="utf-8")
    post = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(obj_path),
        "textbook_structure_path": str(structure),
        "dart_chunks_manifest_path": str(manifest),
    })
    assert post.passed is True
    assert not any(i.severity == "critical" for i in post.issues)
