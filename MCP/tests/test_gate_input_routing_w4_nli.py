"""W4 — gate-input-builder regression tests for the NLI grounding gates.

Locks in two things the W4 posture inversion depends on:

1. The §0.1 fix: ``claim_support`` (ClaimSupportValidator) — which had NO
   registered gate-input builder and therefore ran against an empty
   ``source_chunks`` premise map (a silent no-op) — is NOW registered, along
   with the two new NLI validators (``block_prose_entailment`` /
   ``objective_entailment``). Without this, promoting the NLI gates to critical
   would fail-close every run against an empty premise.

2. The §3.5 authoritative chunk-body fallback builds a non-empty id→TEXT
   premise map from the SemantiK chunkset ``chunks.jsonl`` (the staging
   manifest's ``files[].text`` is best-effort and often absent), so the NLI
   gates have a real premise to entail against.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from MCP.hardening.gate_input_routing import (
    _build_source_chunks_from_dart_jsonl,
    default_router,
)


# --------------------------------------------------------------------- #
# §0.1 — claim_support is NOW registered (was the silent no-op)
# --------------------------------------------------------------------- #


def test_claim_support_now_registered() -> None:
    """Regression-guard the §0.1 bug: claim_support had no builder and ran
    against an empty premise. It must now resolve to a real builder."""
    r = default_router()
    assert (
        "lib.validators.claim_support.ClaimSupportValidator" in r.builders
    ), "claim_support must be registered (the §0.1 silent-no-op fix)"
    # The builder must be the rewrite-block + source_chunks builder so the
    # NLI premise map is populated (not __no_builder_registered__).
    inputs, missing = r.build(
        "lib.validators.claim_support.ClaimSupportValidator", {}, {},
    )
    assert missing != ["__no_builder_registered__"]


def test_block_prose_entailment_registered() -> None:
    r = default_router()
    assert (
        "lib.validators.block_prose_entailment.BlockProseEntailmentValidator"
        in r.builders
    )
    inputs, missing = r.build(
        "lib.validators.block_prose_entailment.BlockProseEntailmentValidator",
        {}, {},
    )
    assert missing != ["__no_builder_registered__"]


def test_objective_entailment_registered() -> None:
    r = default_router()
    assert (
        "lib.validators.objective_entailment.ObjectiveEntailmentValidator"
        in r.builders
    )
    inputs, missing = r.build(
        "lib.validators.objective_entailment.ObjectiveEntailmentValidator",
        {}, {},
    )
    # Resolves via the objective_source_refs builder; missing only when no
    # objectives path is threaded (structured skip), never no-builder.
    assert missing != ["__no_builder_registered__"]


# --------------------------------------------------------------------- #
# §3.5 — authoritative source_chunks from SemantiK chunks.jsonl (non-empty)
# --------------------------------------------------------------------- #


def _write_semantik_chunks(tmp_path: Path, chunks: List[Dict[str, Any]]) -> str:
    chunks_dir = tmp_path / "semantik_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_jsonl = chunks_dir / "chunks.jsonl"
    with chunks_jsonl.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")
    return str(chunks_jsonl)


def test_source_chunks_from_chunks_jsonl_non_empty(tmp_path: Path) -> None:
    """The premise map is populated from the chunkset, keyed on BOTH the
    chunk id AND each source_references[].sourceId."""
    body = "RDF triples are subject-predicate-object statements."
    chunks_path = _write_semantik_chunks(
        tmp_path,
        [{
            "id": "course_chunk_00001",
            "text": body,
            "source": {"source_references": [{"sourceId": "semantik:rdf#s1"}]},
        }],
    )
    phase_outputs = {"chunking": {"semantik_chunks_path": chunks_path}}
    text_map = _build_source_chunks_from_dart_jsonl(phase_outputs)
    assert text_map, "premise map must be non-empty (the §0.1/§3.5 precondition)"
    assert text_map.get("course_chunk_00001") == body
    assert text_map.get("semantik:rdf#s1") == body


def test_source_chunks_from_chunks_jsonl_empty_when_absent() -> None:
    """No chunks path → empty map (graceful, validators degrade)."""
    assert _build_source_chunks_from_dart_jsonl({}) == {}
