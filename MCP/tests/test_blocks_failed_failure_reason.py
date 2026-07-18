"""blocks_failed.jsonl ``failure_reason`` composer.

The rewrite / inter-tier ``blocks_failed.jsonl`` rows used to carry the bare
Block projection with no failure attribution (implicit ``null`` reason). The
``_compose_block_failure_reason`` helper builds a real reason from the per-block
failing-gate signal: the failing gate(s) located at the block (the CURIE gate's
message carries the dropped tokens) + the regen attempts consumed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _compose_block_failure_reason  # noqa: E402


def _block(block_id: str = "page#concept_x_0", attempts: int = 0):
    return SimpleNamespace(block_id=block_id, validation_attempts=attempts)


def _gate(gate_id: str, passed: bool, *issues):
    return {"gate_id": gate_id, "passed": passed, "issues": list(issues)}


def _issue(code: str, message: str, location: str, severity: str = "critical"):
    return {
        "code": code,
        "message": message,
        "location": location,
        "severity": severity,
    }


def test_curie_preservation_reason_includes_tokens_and_attempts():
    """A CURIE-anchoring gate failure surfaces the dropped tokens (from the
    issue message) plus the regen attempts consumed."""
    blk = _block(attempts=2)
    gate_results = [
        _gate(
            "rewrite_curie_anchoring",
            False,
            _issue(
                "OUTLINE_BLOCK_CURIE_NOT_ANCHORED",
                "Block 'page#concept_x_0' CURIE sh:NodeShape not anchored "
                "in the emitted HTML surface",
                "page#concept_x_0",
            ),
        ),
    ]
    reason = _compose_block_failure_reason(blk, gate_results)
    assert reason is not None
    assert "rewrite_curie_anchoring" in reason
    assert "OUTLINE_BLOCK_CURIE_NOT_ANCHORED" in reason
    assert "sh:NodeShape" in reason
    assert "validation_attempts=2" in reason


def test_other_failure_reason_is_gate_code_plus_message_head():
    """A non-CURIE gate failure yields the gate code (failure class) + the
    message head — the analog of an exception class + message head."""
    blk = _block()
    gate_results = [
        _gate(
            "rewrite_source_refs",
            False,
            _issue(
                "EMPTY_SOURCE_REFS",
                "Block 'page#concept_x_0' carries no data-cf-source-ids",
                "page#concept_x_0",
            ),
        ),
    ]
    reason = _compose_block_failure_reason(blk, gate_results)
    assert reason == (
        "rewrite_source_refs:EMPTY_SOURCE_REFS — Block 'page#concept_x_0' "
        "carries no data-cf-source-ids"
    )


def test_multiple_failing_gates_joined():
    blk = _block(attempts=1)
    gate_results = [
        _gate(
            "rewrite_content_type",
            False,
            _issue("BAD_CT", "bad content type", "page#concept_x_0"),
        ),
        _gate(
            "rewrite_source_refs",
            False,
            _issue("EMPTY_SOURCE_REFS", "no source ids", "page#concept_x_0"),
        ),
    ]
    reason = _compose_block_failure_reason(blk, gate_results)
    assert "rewrite_content_type:BAD_CT" in reason
    assert "rewrite_source_refs:EMPTY_SOURCE_REFS" in reason
    assert "; " in reason
    assert "validation_attempts=1" in reason


def test_passing_gates_and_other_block_issues_ignored():
    """Issues on OTHER blocks and passing gates do not leak into this block's
    reason; no located issue → explicit None (honest 'not attributable')."""
    blk = _block(block_id="page#concept_x_0")
    gate_results = [
        _gate("rewrite_curie_anchoring", True),  # passed → ignored
        _gate(
            "rewrite_source_refs",
            False,
            _issue("EMPTY_SOURCE_REFS", "no source ids", "page#OTHER_block_9"),
        ),
    ]
    assert _compose_block_failure_reason(blk, gate_results) is None


def test_malformed_shapes_never_raise():
    blk = _block()
    assert _compose_block_failure_reason(blk, []) is None
    assert _compose_block_failure_reason(blk, [{"passed": False}]) is None
    assert _compose_block_failure_reason(
        blk, [{"gate_id": "g", "passed": False, "issues": ["not-a-dict"]}]
    ) is None
    # Block with no id → None, never raises.
    assert _compose_block_failure_reason(
        SimpleNamespace(), [{"gate_id": "g", "passed": False, "issues": []}]
    ) is None
