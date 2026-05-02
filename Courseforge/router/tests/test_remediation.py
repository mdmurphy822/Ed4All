"""Tests for ``Courseforge/router/remediation.py`` (Phase 3.5 Subtask 4).

Exercises the generalized remediation-suffix builder per
`plans/phase3_5_post_rewrite_validation.md` §A Subtask 4. Coverage:

- :func:`_append_remediation_for_gates` emits one block per failure.
- Directive lookup honours the per-gate-id directives table; unknown
  gate_ids fall back to the generic ``"Re-emit correctly per the
  {validator_name} contract"`` line.
- Long issue messages are truncated at the
  :data:`_MAX_ISSUE_MESSAGE_CHARS` budget (200 chars) so the suffix
  size stays bounded on multi-failure blocks.
- :func:`_append_preserve_remediation` emits a single remediation
  block naming each missing token verbatim.
- :func:`_missing_preserve_tokens` accepts ``content: Any``: str
  searches the full body, dict searches the keys named in ``in_keys``.
- Pass-action results are no-ops — the suffix builder doesn't fire
  when no actionable failure exists.

Reuses the ``GateResult`` / ``GateIssue`` constructor pattern from
:mod:`Courseforge.router.tests.test_validator_action` so the fixtures
stay parallel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.remediation import (  # noqa: E402
    _MAX_ISSUE_MESSAGE_CHARS,
    _REMEDIATION_DIRECTIVES_BY_GATE_ID,
    _append_preserve_remediation,
    _append_remediation_for_gates,
    _missing_preserve_tokens,
)
from MCP.hardening.validation_gates import GateIssue, GateResult  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _gate_result(
    *,
    gate_id: str,
    issues: List[GateIssue],
    passed: bool = False,
    action: str = "regenerate",
    validator_name: str = "TestValidator",
) -> GateResult:
    """Build a canned :class:`GateResult` for the suffix builder."""
    return GateResult(
        gate_id=gate_id,
        validator_name=validator_name,
        validator_version="1.0",
        passed=passed,
        issues=list(issues),
        action=action,
    )


# ---------------------------------------------------------------------------
# _append_remediation_for_gates
# ---------------------------------------------------------------------------


def test_append_remediation_for_gates_emits_one_block_per_failure():
    """Two failures → two remediation blocks, both reachable via
    substring match against the canonical suffix shape."""
    failures = [
        _gate_result(
            gate_id="outline_curie_anchoring",
            issues=[
                GateIssue(
                    severity="critical",
                    code="CURIE_DROPPED",
                    message="sh:NodeShape was dropped from the curies array",
                ),
            ],
        ),
        _gate_result(
            gate_id="outline_content_type",
            issues=[
                GateIssue(
                    severity="critical",
                    code="CONTENT_TYPE_INVALID",
                    message="content_type='widget' not in canonical 8-value enum",
                ),
            ],
        ),
    ]
    out = _append_remediation_for_gates("base prompt", failures)
    assert "base prompt" in out
    assert "Your previous attempt failed validation" in out
    assert "[outline_curie_anchoring]" in out
    assert "sh:NodeShape was dropped" in out
    assert "[outline_content_type]" in out
    assert "widget" in out


def test_append_remediation_for_gates_uses_directive_table_lookup():
    """Each failure block carries the directive looked up from the
    per-gate-id table — verbatim wording for the canonical 8 keys."""
    failures = [
        _gate_result(
            gate_id="rewrite_source_refs",
            issues=[
                GateIssue(
                    severity="critical",
                    code="SOURCE_REF_MISSING",
                    message="block has no data-cf-source-ids",
                ),
            ],
        ),
    ]
    out = _append_remediation_for_gates("p", failures)
    expected_directive = _REMEDIATION_DIRECTIVES_BY_GATE_ID[
        "rewrite_source_refs"
    ]
    assert expected_directive in out


def test_append_remediation_for_gates_falls_back_to_generic_directive():
    """An unknown gate_id falls back to the generic ``"Re-emit correctly
    per the {validator_name} contract"`` line so the suffix is never
    silent on a failure."""
    failures = [
        _gate_result(
            gate_id="unknown_gate_id_not_in_table",
            issues=[
                GateIssue(
                    severity="critical",
                    code="GENERIC_FAIL",
                    message="something went wrong",
                ),
            ],
            validator_name="MyOddValidator",
        ),
    ]
    out = _append_remediation_for_gates("p", failures)
    assert "MyOddValidator contract" in out
    assert "Re-emit correctly per the MyOddValidator contract" in out


def test_append_remediation_for_gates_truncates_long_issue_messages():
    """An issue message longer than :data:`_MAX_ISSUE_MESSAGE_CHARS`
    is truncated with an ellipsis so the suffix size stays bounded
    on multi-failure blocks."""
    long_msg = "X" * 500
    failures = [
        _gate_result(
            gate_id="outline_curie_anchoring",
            issues=[
                GateIssue(
                    severity="critical",
                    code="CURIE_DROPPED",
                    message=long_msg,
                ),
            ],
        ),
    ]
    out = _append_remediation_for_gates("p", failures)
    # The full 500-char message is NOT in the suffix.
    assert long_msg not in out
    # The truncated form (199 X's + ellipsis) IS in the suffix.
    truncated = "X" * (_MAX_ISSUE_MESSAGE_CHARS - 1) + "…"
    assert truncated in out


# ---------------------------------------------------------------------------
# _append_preserve_remediation
# ---------------------------------------------------------------------------


def test_append_preserve_remediation_emits_token_list():
    """The preserve-token remediation names each missing token verbatim
    and reuses the canonical "did not include the required" phrase the
    rewrite-provider regression suite substring-matches."""
    out = _append_preserve_remediation(
        "prompt body",
        ["sh:NodeShape", "rdfs:subClassOf"],
    )
    assert "prompt body" in out
    assert "did not include the required" in out
    assert "sh:NodeShape" in out
    assert "rdfs:subClassOf" in out
    # Default in_keys=("body",) interpolates field name.
    assert "body" in out


def test_append_preserve_remediation_empty_tokens_is_noop():
    """No missing tokens → prompt passes through unchanged."""
    out = _append_preserve_remediation("the original prompt", [])
    assert out == "the original prompt"


# ---------------------------------------------------------------------------
# _missing_preserve_tokens
# ---------------------------------------------------------------------------


def test_missing_preserve_tokens_str_content_searches_full_body():
    """When ``content`` is a string, the function searches the full
    body for each token via substring match."""
    html = (
        "<section><p>The <code>sh:NodeShape</code> constrains the "
        "focus node.</p></section>"
    )
    missing = _missing_preserve_tokens(html, ["sh:NodeShape"])
    assert missing == []

    missing = _missing_preserve_tokens(
        "<section><p>The shape constrains.</p></section>",
        ["sh:NodeShape", "rdfs:subClassOf"],
    )
    assert "sh:NodeShape" in missing
    assert "rdfs:subClassOf" in missing


def test_missing_preserve_tokens_dict_content_searches_in_keys():
    """When ``content`` is a dict, the function searches the keys
    named in ``in_keys`` for each token."""
    parsed = {
        "prompt": "Define sh:NodeShape verbatim.",
        "completion": "Answer: it is a node shape.",
    }
    # Token only appears in "prompt" — searching just "completion"
    # reports it missing.
    missing = _missing_preserve_tokens(
        parsed, ["sh:NodeShape"], in_keys=("completion",)
    )
    assert missing == ["sh:NodeShape"]

    # Searching both keys finds the token.
    missing = _missing_preserve_tokens(
        parsed, ["sh:NodeShape"], in_keys=("prompt", "completion")
    )
    assert missing == []


def test_missing_preserve_tokens_empty_tokens_returns_empty_list():
    """Empty tokens list → empty result regardless of content shape."""
    assert _missing_preserve_tokens("anything", []) == []
    assert _missing_preserve_tokens({"body": "anything"}, []) == []
    assert _missing_preserve_tokens(None, []) == []


# ---------------------------------------------------------------------------
# Pass-action no-op semantics
# ---------------------------------------------------------------------------


def test_remediation_for_pass_action_is_noop():
    """A GateResult whose ``action="pass"`` doesn't trigger remediation
    — the prompt passes through unchanged."""
    failures = [
        _gate_result(
            gate_id="outline_curie_anchoring",
            issues=[],
            passed=True,
            action="pass",
        ),
    ]
    out = _append_remediation_for_gates("clean prompt", failures)
    assert out == "clean prompt"


def test_remediation_for_legacy_passed_result_is_noop():
    """Legacy validators leaving ``action=None`` and ``passed=True``
    (the legacy "pass" interpretation) are no-ops."""
    legacy_passing = GateResult(
        gate_id="outline_curie_anchoring",
        validator_name="LegacyValidator",
        validator_version="1.0",
        passed=True,
        issues=[],
        action=None,  # legacy
    )
    out = _append_remediation_for_gates("clean prompt", [legacy_passing])
    assert out == "clean prompt"


def test_remediation_for_empty_failures_list_is_noop():
    """An empty failures list returns the original prompt unchanged."""
    out = _append_remediation_for_gates("untouched", [])
    assert out == "untouched"


def test_remediation_filters_pass_results_among_mixed_batch():
    """When the batch carries a mix of pass + fail results, only the
    failures contribute remediation blocks."""
    failures = [
        _gate_result(
            gate_id="outline_curie_anchoring",
            issues=[],
            passed=True,
            action="pass",
        ),
        _gate_result(
            gate_id="outline_content_type",
            issues=[
                GateIssue(
                    severity="critical",
                    code="CT_INVALID",
                    message="content_type missing",
                ),
            ],
            action="regenerate",
        ),
    ]
    out = _append_remediation_for_gates("p", failures)
    # Only one failure block, naming the non-passing gate_id.
    assert "[outline_content_type]" in out
    assert "[outline_curie_anchoring]" not in out


# ---------------------------------------------------------------------------
# Directive table integrity
# ---------------------------------------------------------------------------


def test_directive_table_carries_all_eight_canonical_gate_ids():
    """The directives table covers the 4 outline + 4 rewrite gate IDs
    declared in the Phase 3.5 plan §A Subtask 1 contract."""
    expected = {
        "outline_curie_anchoring",
        "outline_content_type",
        "outline_page_objectives",
        "outline_source_refs",
        "rewrite_curie_anchoring",
        "rewrite_content_type",
        "rewrite_page_objectives",
        "rewrite_source_refs",
    }
    assert expected.issubset(set(_REMEDIATION_DIRECTIVES_BY_GATE_ID.keys()))
