#!/usr/bin/env python3
"""Wave 116 — incremental sidecar write coverage for ``run_synthesis``.

Covers two contracts on the ``.jsonl.in_progress`` sidecar files that
``Trainforge/synthesize_training.py::run_synthesis`` writes
incrementally as pairs are emitted:

  1. On a clean exit (no exception, no budget cap), the sidecars MUST
     be deleted after the atomic final ``_write_jsonl`` so the
     ``training_specs/`` directory is left in a tidy state.

  2. On a ``SynthesisBudgetExceeded`` early-exit, the sidecars MUST
     stay on disk so the operator has inspectable partial output and
     can resume from the synthesis cache on a re-run with a higher
     ``--max-dispatches``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest  # noqa: F401  -- imported for symmetry with sibling tests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import run_synthesis  # noqa: E402


FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)


def _make_working_copy(tmp_path: Path) -> Path:
    """Copy the read-only fixture into tmp so run_synthesis can write."""
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
        dst / "training_specs" / "instruction_pairs.jsonl.in_progress",
        dst / "training_specs" / "preference_pairs.jsonl.in_progress",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def test_sidecar_written_incrementally_and_cleaned_up_on_success(
    tmp_path: Path,
) -> None:
    """A clean ``run_synthesis`` invocation MUST leave no sidecars on
    disk after writing the final atomic JSONL artifacts.

    Uses ``provider="mock"`` so the test is fully deterministic and
    needs no LLM. The fixture has 15 chunks — enough that the
    incremental-write path is exercised many times during the run.
    """
    working = _make_working_copy(tmp_path)
    inst_progress = (
        working / "training_specs" / "instruction_pairs.jsonl.in_progress"
    )
    pref_progress = (
        working / "training_specs" / "preference_pairs.jsonl.in_progress"
    )
    inst_final = working / "training_specs" / "instruction_pairs.jsonl"
    pref_final = working / "training_specs" / "preference_pairs.jsonl"

    # Pre-condition: no sidecars exist before the call.
    assert not inst_progress.exists()
    assert not pref_progress.exists()

    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    # Post-condition: final files exist and carry the synthesized pairs.
    assert inst_final.exists()
    assert pref_final.exists()
    assert stats.instruction_pairs_emitted > 0
    assert stats.preference_pairs_emitted > 0
    # Final files have at least one line per emitted pair.
    inst_lines = [
        l for l in inst_final.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    pref_lines = [
        l for l in pref_final.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert len(inst_lines) == stats.instruction_pairs_emitted
    assert len(pref_lines) == stats.preference_pairs_emitted

    # Post-condition: sidecars cleaned up on clean exit.
    assert not inst_progress.exists(), (
        "Wave 116 contract: instruction sidecar must be deleted on a "
        "clean run; found it lingering at "
        f"{inst_progress}"
    )
    assert not pref_progress.exists(), (
        "Wave 116 contract: preference sidecar must be deleted on a "
        "clean run; found it lingering at "
        f"{pref_progress}"
    )

    # Capped flag is False on a healthy run (no budget cap was set).
    assert stats.capped_at_max_dispatches is False


def test_sidecar_preserved_on_budget_exceeded(tmp_path: Path) -> None:
    """When the chunk loop raises ``SynthesisBudgetExceeded``, the
    sidecars MUST be preserved so the operator can inspect partial
    output. This test simplifies the FakeLocalDispatcher path by using
    a stub paraphrase provider that raises after one successful emit:
    the assertion is on the SIDECAR being preserved, not on the exact
    machinery that triggered the budget cap.
    """
    from Trainforge.tests._synthesis_fakes import (
        FakeLocalDispatcher,
        make_instruction_response,
        make_preference_response,
    )

    # Wave 112 Task 4: outputs must respect _validate_lengths floors
    # (PROMPT_MIN=40, COMPLETION_MIN=50) so they don't fail the
    # length-clamp before the sidecar is exercised.
    _ok_p = "Paraphrased prompt explaining RDFS in detail for the learner."
    _ok_c = (
        "Paraphrased completion grounded in the source chunk text "
        "covering RDFS and SHACL contracts in sufficient detail."
    )

    async def agent_tool(*, task_params, **_kw):
        if task_params["kind"] == "instruction":
            return make_instruction_response(prompt=_ok_p, completion=_ok_c)
        return make_preference_response(prompt=_ok_p, chosen=_ok_c, rejected=_ok_c)

    dispatcher = FakeLocalDispatcher(agent_tool=agent_tool)
    working = _make_working_copy(tmp_path)
    inst_progress = (
        working / "training_specs" / "instruction_pairs.jsonl.in_progress"
    )
    pref_progress = (
        working / "training_specs" / "preference_pairs.jsonl.in_progress"
    )

    # Pre-condition: no sidecars exist before the call.
    assert not inst_progress.exists()
    assert not pref_progress.exists()

    # ``max_dispatches=1`` triggers SynthesisBudgetExceeded after the
    # first dispatch; per Wave 111 / Phase E, run_synthesis catches
    # the exception, writes pilot_progress.json, and returns a
    # SynthesisStats with capped_at_max_dispatches=True.
    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="claude_session",
        seed=11,
        dispatcher=dispatcher,
        max_dispatches=1,
    )

    # Post-condition: budget-cap path was taken.
    assert stats.capped_at_max_dispatches is True

    # Post-condition: pilot_progress.json snapshot exists.
    progress_path = working / "training_specs" / "pilot_progress.json"
    assert progress_path.exists()

    # Post-condition: sidecars preserved on early-exit. At least one
    # of the two MUST exist — exactly which depends on whether the
    # budget-exceeded raise fired during the instruction or
    # preference dispatch on the very first chunk. The contract is
    # that the sidecar is NOT unlinked on an early-exit path.
    assert inst_progress.exists() or pref_progress.exists(), (
        "Wave 116 contract: at least one sidecar must be preserved "
        "for postmortem inspection on a SynthesisBudgetExceeded exit; "
        "neither was found"
    )
    # The instruction sidecar should have at least one emitted pair
    # written before the budget tripped (max_dispatches=1 means the
    # FIRST instruction call succeeded; the second / preference call
    # is what raises). If it doesn't exist, the test still passes by
    # the OR above — but if it does, it must have content.
    if inst_progress.exists():
        content = inst_progress.read_text(encoding="utf-8")
        # File handle was flushed after the successful emit, so the
        # written pair JSON should be visible regardless of whether
        # the loop ran to completion.
        if stats.instruction_pairs_emitted > 0:
            assert content.strip(), (
                "instruction sidecar exists but is empty; the flush() "
                "after each append was not exercised"
            )
