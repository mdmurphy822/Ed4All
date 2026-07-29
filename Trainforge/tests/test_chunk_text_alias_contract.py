"""Regression net: the chunk text-field alias contract.

``Trainforge/synthesis_eligibility.py`` used to read a chunk's prose as
``chunk.get("text") or ""``. That single expression collapsed two facts that
are not the same:

* the chunk **declares itself empty** — a real, gate-able disposition, and
* the mapping **has no prose field at all** — shape drift.

A chunk carrying its prose under the ``content`` alias therefore scored as
empty prose, and the content gate excluded it under a fabricated
``degenerate_source_stem`` reason: a chunk full of real prose reported as
auto-generated slot-filler residue. Two MCP dispatch tests
(``test_synthesize_training_dispatches_in_live_config`` /
``..._schema_parity``) had been failing on exactly that, emitting a zero-byte
``instruction_pairs.jsonl``.

Every assertion below drives a PRODUCTION entry point —
``Trainforge.synthesize_training._pair_eligibility_for_mode`` (the seam
``run_synthesis`` itself calls) and the ``synthesize_training`` tool-registry
callable — never the leaf ``content_gate_eligibility``. A leaf test would not
have caught this: the leaf was doing exactly what it was told.

Fixtures are written inline from SHAPES only — no course slug, corpus path,
chunk id, or book title from any real corpus.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from Trainforge.synthesis_eligibility import (
    CHUNK_TEXT_FIELDS,
    ChunkTextContractError,
    resolve_chunk_text,
)
from Trainforge.synthesize_training import _pair_eligibility_for_mode

# Prose long enough to clear the content gate's 40-word floor on its own, so
# the fixtures below isolate the ALIAS variable rather than riding on
# ``key_terms``.
_PROSE = (
    "A knowledge graph organises information as nodes joined by typed edges. "
    "Each node stands for one entity in the domain and each edge names the "
    "relation that holds between two of them, so the same structure records "
    "both what exists and how the pieces connect. Reading an edge in the "
    "reverse direction answers a different question about the same pair of "
    "entities, which is why the direction is stored rather than inferred."
)


def _chunk(text_field: str | None = "text", **overrides: Any) -> Dict[str, Any]:
    """A minimal eligible chunk whose prose sits under ``text_field``.

    ``text_field=None`` builds the drift shape: a mapping carrying no prose
    field under any accepted alias.
    """
    chunk: Dict[str, Any] = {
        "id": "chunk-0001",
        "chunk_type": "explanation",
        "bloom_level": "understand",
        "learning_outcome_refs": ["co-01"],
    }
    if text_field is not None:
        chunk[text_field] = _PROSE
    chunk.update(overrides)
    return chunk


# --------------------------------------------------------------------------- #
# The defect, at the production seam
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alias", CHUNK_TEXT_FIELDS)
def test_prose_under_any_accepted_alias_is_eligible(alias: str) -> None:
    """Prose is prose under every accepted alias — never slot-filler residue."""
    verdict = _pair_eligibility_for_mode(_chunk(alias), kind="instruction")
    assert verdict.eligible, (
        f"prose carried under {alias!r} was excluded as {verdict.reason!r}; "
        f"the alias set {list(CHUNK_TEXT_FIELDS)} is a READ contract and "
        f"every member must resolve identically"
    )


def test_content_alias_is_not_reported_as_a_degenerate_stem() -> None:
    """Pin the exact false verdict, not merely 'it was ineligible'.

    ``degenerate_source_stem`` asserts a positive finding — that the chunk's
    leading sentence is template residue. Reaching it because the prose was
    never read is a fabricated verdict, so the reason is asserted by name.
    """
    verdict = _pair_eligibility_for_mode(_chunk("content"), kind="instruction")
    assert verdict.reason != "degenerate_source_stem"
    assert verdict.eligible


def test_canonical_text_field_wins_over_an_alias() -> None:
    """``text`` is canonical; an alias is consulted only when it is absent."""
    chunk = _chunk("text", content="unrelated alias prose that must not win")
    assert resolve_chunk_text(chunk) == _PROSE


def test_a_blank_canonical_field_falls_through_to_a_populated_alias() -> None:
    """Precedence is 'first field carrying prose', not 'first field present'."""
    chunk = _chunk("content", text="   ")
    assert resolve_chunk_text(chunk) == _PROSE
    assert _pair_eligibility_for_mode(chunk, kind="instruction").eligible


# --------------------------------------------------------------------------- #
# The failure mode: declared-empty is a verdict, missing is drift
# --------------------------------------------------------------------------- #


def test_declared_empty_chunk_stays_an_honest_gate_verdict() -> None:
    """An explicitly blank chunk is empty — the gate owns it, not an exception.

    This is the half of the contract that must NOT become loud: turning a real
    disposition into a crash would take the content gate's whole population
    down with it.
    """
    chunk = _chunk(None, text="", concept_tags=[], key_terms=[])
    verdict = _pair_eligibility_for_mode(chunk, kind="instruction")
    assert not verdict.eligible
    assert verdict.reason == "chunk_carries_no_groundable_content"


def test_a_chunk_with_no_prose_field_at_all_fails_loudly() -> None:
    """Shape drift raises; it is never defaulted to empty prose.

    Silently scoring an absent field as ``""`` is what manufactured the false
    ``degenerate_source_stem`` verdict in the first place, so the missing-field
    case must be unrepresentable as a verdict.
    """
    chunk = _chunk(None, key_terms=[{"term": "knowledge graph"}])
    with pytest.raises(ChunkTextContractError) as excinfo:
        _pair_eligibility_for_mode(chunk, kind="instruction")
    message = str(excinfo.value)
    # The message must identify the drifting producer, not just complain.
    for field in CHUNK_TEXT_FIELDS:
        assert field in message
    assert "chunk-0001" in message


def test_resolver_rejects_a_non_mapping_rather_than_returning_empty() -> None:
    with pytest.raises(ChunkTextContractError):
        resolve_chunk_text(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# End-to-end: the dispatch that was emitting a zero-byte corpus
# --------------------------------------------------------------------------- #


def test_alias_keyed_chunk_emits_real_instruction_rows_end_to_end(
    tmp_path: Path,
) -> None:
    """The registry dispatch must write pair ROWS, not just create the file.

    ``instruction_pairs.jsonl`` existed all along — it was zero bytes. Asserting
    on parsed row count (and on the row carrying this chunk's id) is what
    distinguishes 'the phase dispatched' from 'the phase produced training
    data'.
    """
    corpus_dir = tmp_path / "trainforge"
    (corpus_dir / "corpus").mkdir(parents=True)
    chunk = {
        "id": "chunk-alias-0001",
        "course_id": "DEMO_101",
        "section_id": "sec_01",
        # The alias shape — the whole point of the fixture.
        "content": _PROSE,
        "learning_outcome_refs": ["TO-01"],
        "bloom_level": "understand",
        "content_type_label": "explanation",
        "key_terms": [
            {"term": "knowledge graph", "definition": "a structured representation"},
        ],
    }
    (corpus_dir / "corpus" / "chunks.jsonl").write_text(
        json.dumps(chunk) + "\n", encoding="utf-8"
    )

    pipeline_tools = importlib.import_module("MCP.tools.pipeline_tools")
    registry = pipeline_tools._build_tool_registry()
    result = json.loads(
        asyncio.run(
            registry["synthesize_training"](
                corpus_dir=str(corpus_dir),
                course_code="DEMO_101",
                provider="mock",
                seed=7,
            )
        )
    )

    assert result.get("success") is True, result
    rows = [
        json.loads(line)
        for line in Path(result["instruction_pairs_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert rows, (
        "alias-keyed chunk produced an empty instruction corpus — the chunk "
        "was excluded before dispatch, which is the defect this test pins"
    )
    assert any(row.get("chunk_id") == "chunk-alias-0001" for row in rows), (
        f"no emitted pair traces back to the source chunk: {rows!r}"
    )
