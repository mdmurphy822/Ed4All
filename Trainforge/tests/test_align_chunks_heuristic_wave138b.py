"""Wave 138b — _heuristic_role content_type_label-aware extension.

The Wave 138a TeachingRoleAlignmentEvaluator surfaced a systematic
underlabeling of ``content_type_label="real_world_scenario"`` /
``"scenario"`` chunks: 0/8 transfer rate on rdf-shacl-551-2 vs an
expected ≥70% share. The 4-role LLM curriculum-alignment enum
(``introduce`` / ``elaborate`` / ``reinforce`` / ``synthesize``) cannot
emit ``transfer`` or ``assess`` by design — those are heuristic-only.
Without the new branch in ``align_chunks._heuristic_role``, scenario
chunks fall through to the LLM and never get ``transfer``.

These tests pin the extension's behavior:

- New ``content_type_label`` rules return the right deterministic role.
- Ordering preserves the legacy ``chunk_type=assessment_item`` precedence.
- Chunks without a recognized signal still fall through to legacy
  ``resource_type`` checks (no behavioral regression).
- Decision capture fires once per heuristic-extension fire with the
  expected ``decision_type="teaching_role_heuristic_extended"`` value
  and the chunk_id / role / rule interpolated into the rationale.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.align_chunks import _heuristic_role  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingCapture:
    """Stand-in for ``DecisionCapture`` — records every decision call."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def _chunk(
    chunk_id: str = "chunk_001",
    *,
    chunk_type: str = "explanation",
    content_type_label: str = "",
    resource_type: str = "",
    position_in_module: int = 1,
) -> Dict[str, Any]:
    """Build a minimal chunk dict matching the shape ``_heuristic_role`` reads."""
    return {
        "id": chunk_id,
        "chunk_type": chunk_type,
        "content_type_label": content_type_label,
        "source": {
            "resource_type": resource_type,
            "position_in_module": position_in_module,
        },
    }


# ---------------------------------------------------------------------------
# Wave 138b — content_type_label-aware rules
# ---------------------------------------------------------------------------


def test_heuristic_classifies_real_world_scenario_as_transfer() -> None:
    """``content_type_label="real_world_scenario"`` → ``transfer``."""
    chunk = _chunk(
        "chunk_rws_001",
        chunk_type="explanation",
        content_type_label="real_world_scenario",
    )
    assert _heuristic_role(chunk) == "transfer"


def test_heuristic_classifies_scenario_content_type_as_transfer() -> None:
    """``content_type_label="scenario"`` → ``transfer`` (the
    rdf-shacl-551-2 chunker emits ``"scenario"`` not the longer
    ``"real_world_scenario"`` form for these chunks)."""
    chunk = _chunk(
        "chunk_scen_001",
        chunk_type="explanation",
        content_type_label="scenario",
    )
    assert _heuristic_role(chunk) == "transfer"


def test_heuristic_classifies_self_check_as_assess() -> None:
    """``content_type_label="self_check"`` → ``assess`` (the LLM enum
    cannot emit ``assess`` so this branch is load-bearing)."""
    chunk = _chunk(
        "chunk_sc_001",
        chunk_type="explanation",
        content_type_label="self_check",
    )
    assert _heuristic_role(chunk) == "assess"


def test_heuristic_classifies_assessment_content_type_as_assess() -> None:
    """``content_type_label="assessment"`` → ``assess`` even when
    ``chunk_type`` is not ``assessment_item`` (e.g. an explanatory
    preamble inside an assessment block)."""
    chunk = _chunk(
        "chunk_a_001",
        chunk_type="explanation",
        content_type_label="assessment",
    )
    assert _heuristic_role(chunk) == "assess"


def test_heuristic_classifies_summary_content_type_as_synthesize() -> None:
    """``content_type_label="summary"`` → ``synthesize``. Mirrors the
    legacy ``resource_type="summary"`` branch but fires earlier on the
    content_type_label signal so chunker-emitted summary content
    routes correctly even without the wrapper page metadata."""
    chunk = _chunk(
        "chunk_sm_001",
        chunk_type="explanation",
        content_type_label="summary",
    )
    assert _heuristic_role(chunk) == "synthesize"


# ---------------------------------------------------------------------------
# Ordering — legacy chunk_type=assessment_item must still win
# ---------------------------------------------------------------------------


def test_heuristic_assessment_item_chunk_type_still_takes_precedence() -> None:
    """``chunk_type="assessment_item"`` returns ``assess`` regardless of
    ``content_type_label``. The Wave 138b branches are inserted AFTER
    the legacy ``assessment_item`` check, so a contradictory
    ``content_type_label="summary"`` annotation cannot flip an
    assessment item to ``synthesize``."""
    chunk = _chunk(
        "chunk_x",
        chunk_type="assessment_item",
        content_type_label="summary",
    )
    assert _heuristic_role(chunk) == "assess"


def test_heuristic_returns_none_for_chunks_without_recognized_signal() -> None:
    """A chunk with no ``chunk_type``, no relevant ``content_type_label``,
    and no ``resource_type`` signal must still fall through to ``None``
    so the LLM / mock path can classify it. Regression-protects the
    pre-Wave-138b legacy path."""
    chunk = _chunk(
        "chunk_unknown",
        chunk_type="explanation",
        content_type_label="motivation",  # no rule
        resource_type="",
    )
    assert _heuristic_role(chunk) is None


def test_heuristic_legacy_resource_type_summary_still_works() -> None:
    """The legacy ``resource_type="summary"`` branch fires when there's
    no ``content_type_label`` signal — regression-protects pre-Wave-138b
    callers that don't carry a content_type_label."""
    chunk = _chunk(
        "chunk_legacy",
        chunk_type="explanation",
        content_type_label="",
        resource_type="summary",
    )
    assert _heuristic_role(chunk) == "synthesize"


def test_heuristic_legacy_application_resource_type_still_returns_transfer() -> None:
    """The legacy ``resource_type="application"`` branch returns
    ``transfer`` for chunks without a content_type_label hint."""
    chunk = _chunk(
        "chunk_legacy_app",
        chunk_type="explanation",
        content_type_label="",
        resource_type="application",
    )
    assert _heuristic_role(chunk) == "transfer"


# ---------------------------------------------------------------------------
# Decision capture wiring
# ---------------------------------------------------------------------------


def test_heuristic_extended_emits_decision_capture_for_scenario() -> None:
    """A ``content_type_label="scenario"`` chunk fires exactly one
    ``teaching_role_heuristic_extended`` decision event with the
    chunk_id / role / rule interpolated into the rationale."""
    capture = _RecordingCapture()
    chunk = _chunk(
        "chunk_scen_001",
        chunk_type="explanation",
        content_type_label="scenario",
    )
    role = _heuristic_role(chunk, capture=capture)
    assert role == "transfer"
    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "teaching_role_heuristic_extended"
    assert event["decision"] == "role=transfer"
    rationale = event["rationale"]
    assert "chunk_scen_001" in rationale
    assert "transfer" in rationale
    assert "scenario" in rationale
    metadata = event.get("metadata") or {}
    assert metadata.get("rule") == "content_type_label_scenario"
    assert metadata.get("role") == "transfer"
    assert metadata.get("chunk_id") == "chunk_scen_001"


def test_heuristic_extended_does_not_capture_when_legacy_path_fires() -> None:
    """The capture only fires on the new content_type_label-aware
    branches. Legacy ``chunk_type=assessment_item`` /
    ``resource_type=summary`` fires must NOT emit the new event —
    they're not the new heuristic and would muddy the audit trail."""
    capture = _RecordingCapture()
    # Legacy assessment_item path
    chunk_a = _chunk(
        "chunk_a",
        chunk_type="assessment_item",
        content_type_label="",
    )
    assert _heuristic_role(chunk_a, capture=capture) == "assess"
    # Legacy resource_type=summary path
    chunk_s = _chunk(
        "chunk_s",
        chunk_type="explanation",
        content_type_label="",
        resource_type="summary",
    )
    assert _heuristic_role(chunk_s, capture=capture) == "synthesize"
    # Neither legacy path emitted a decision event.
    assert capture.events == []


def test_heuristic_extended_capture_is_optional_argument() -> None:
    """Callers that don't thread a capture through must not break.
    ``_heuristic_role(chunk)`` with no capture kwarg should still
    return the new role for content_type_label=scenario."""
    chunk = _chunk(
        "chunk_no_cap",
        chunk_type="explanation",
        content_type_label="scenario",
    )
    # Default-arg call — no capture passed.
    assert _heuristic_role(chunk) == "transfer"


def test_heuristic_extended_capture_failure_does_not_block_classification() -> None:
    """If the capture's ``log_decision`` raises, the classification
    must still return the role. Decision capture is observability —
    a logging failure cannot cost a chunk its teaching role."""

    class _BrokenCapture:
        def log_decision(self, **kwargs: Any) -> None:
            raise RuntimeError("capture is broken")

    chunk = _chunk(
        "chunk_broken_cap",
        chunk_type="explanation",
        content_type_label="scenario",
    )
    # Must not raise; must still classify.
    assert _heuristic_role(chunk, capture=_BrokenCapture()) == "transfer"
