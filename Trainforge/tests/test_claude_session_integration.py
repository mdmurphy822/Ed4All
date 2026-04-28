"""End-to-end smoke test for --provider claude_session.

Exercises the path: run_synthesis -> ClaudeSessionProvider (with cache and
capture wired) -> instruction_factory / preference_factory -> JSONL
output. The dispatcher is a FakeLocalDispatcher whose agent_tool returns
deterministic paraphrases. The test asserts both the JSONL carries
provider='claude_session' and that the synthesis_provider_call decision
event fires.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import run_synthesis
from Trainforge.tests._synthesis_fakes import (
    FakeLocalDispatcher,
    make_instruction_response,
    make_preference_response,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "mini_course_training"


class _ListCapture:
    """DecisionCapture stand-in that records every ``log_decision`` call.

    Mirrors enough of the real DecisionCapture surface that the synthesis
    stage can introspect ``self.decisions[-1]["event_id"]`` (used to
    populate ``decision_capture_id`` on each emitted pair) without
    crashing.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.decisions: list[dict] = []

    def log_decision(self, **kwargs: object) -> None:
        event = dict(kwargs)
        event_id = f"event_{len(self.events):06d}"
        event["event_id"] = event_id
        self.events.append(event)
        self.decisions.append(event)


def _make_working_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def test_full_synthesis_with_claude_session_emits_provider_tag_and_capture(
    tmp_path: Path,
) -> None:
    async def agent_tool(*, task_params, **_kw) -> dict:
        kind = task_params["kind"]
        chunk_id = task_params.get("chunk_id", "")
        if kind == "instruction":
            return make_instruction_response(
                prompt=(
                    "Explain the role of the topic from chunk_id="
                    f"{chunk_id} for an RDF/SHACL learner."
                ),
                completion=(
                    "RDFS describes vocabulary semantics — class hierarchy and "
                    "property domains — in a way downstream RDF processors can "
                    f"reason about. [{chunk_id}]"
                ),
            )
        return make_preference_response(
            prompt="Which option is correct about the chunk topic?",
            chosen=(
                "RDFS describes vocabulary semantics; SHACL validates RDF "
                "graphs against shape constraints."
            ),
            rejected=(
                "RDFS validates RDF graphs against shape constraints; SHACL "
                "describes vocabulary semantics."
            ),
        )

    dispatcher = FakeLocalDispatcher(agent_tool=agent_tool)
    capture = _ListCapture()

    working = _make_working_copy(tmp_path)
    cache_path = working / "training_specs" / ".synthesis_cache.jsonl"

    run_synthesis(
        corpus_dir=working,
        course_code="TEST_END2END",
        provider="claude_session",
        seed=11,
        capture=capture,
        dispatcher=dispatcher,
        cache_path=cache_path,
    )

    inst_rows = [
        json.loads(line)
        for line in (working / "training_specs" / "instruction_pairs.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    pref_rows = [
        json.loads(line)
        for line in (working / "training_specs" / "preference_pairs.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]

    assert inst_rows, "no instruction pairs emitted"
    assert pref_rows, "no preference pairs emitted"
    assert all(r["provider"] == "claude_session" for r in inst_rows)
    assert all(r["provider"] == "claude_session" for r in pref_rows)

    # synthesis_provider_call fired at least once per call type:
    inst_caps = [
        e for e in capture.events if e.get("decision") == "claude_session::instruction"
    ]
    pref_caps = [
        e for e in capture.events if e.get("decision") == "claude_session::preference"
    ]
    assert inst_caps, "no instruction decision events captured"
    assert pref_caps, "no preference decision events captured"

    # Cache populated:
    assert cache_path.exists()
    cache_rows = [
        json.loads(line)
        for line in cache_path.read_text().splitlines()
        if line.strip()
    ]
    assert cache_rows
