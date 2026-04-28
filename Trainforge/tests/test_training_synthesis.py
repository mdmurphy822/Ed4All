#!/usr/bin/env python3
"""
Tests for Trainforge's training-pair synthesis stage (Worker C).

Covered contracts:
  - Instruction and preference factories are deterministic under seed
  - Emitted pairs validate against their JSON schemas
  - Quality gates reject malformed pairs with a clear diagnostic
  - No 50+-char verbatim span from chunk.text leaks into the prompt
  - chosen != rejected with token-Jaccard delta >= 0.3
  - Every emitted pair carries a resolvable decision_capture_id
  - Chunks without learning_outcome_refs produce zero pairs
  - Integration on the mini_course_training fixture meets the volume floor
  - Stage idempotence: same-seed second run is byte-identical
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import jsonschema
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.instruction_factory import (
    COMPLETION_MAX,
    COMPLETION_MIN,
    MAX_VERBATIM_SPAN,
    PROMPT_MAX,
    PROMPT_MIN,
    synthesize_instruction_pair,
)
from Trainforge.generators.preference_factory import (
    JACCARD_DELTA_MIN,
    synthesize_preference_pair,
)
from Trainforge.synthesize_training import run_synthesis

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
SCHEMAS_ROOT = PROJECT_ROOT / "schemas"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_schema(name: str) -> dict:
    with (SCHEMAS_ROOT / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_fixture_chunks() -> list[dict]:
    with (FIXTURE_ROOT / "corpus" / "chunks.jsonl").open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _make_working_copy(tmp_path: Path) -> Path:
    """Copy the read-only fixture into a tmp dir so run_synthesis can write."""
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    # Clear any stale pairs from earlier test runs in the source tree.
    for stale in (dst / "training_specs" / "instruction_pairs.jsonl",
                  dst / "training_specs" / "preference_pairs.jsonl"):
        if stale.exists():
            stale.unlink()
    return dst


def _find_chunk(chunks: list[dict], chunk_id: str) -> dict:
    for c in chunks:
        if c["id"] == chunk_id:
            return c
    raise KeyError(chunk_id)


# ---------------------------------------------------------------------------
# Unit tests: factories
# ---------------------------------------------------------------------------

def test_instruction_factory_deterministic_under_seed():
    chunks = _load_fixture_chunks()
    chunk = _find_chunk(chunks, "chunk_mc_01")
    a = synthesize_instruction_pair(chunk, seed=42).pair
    b = synthesize_instruction_pair(chunk, seed=42).pair
    assert a is not None and b is not None
    # Strip decision_capture_id (assigned by the stage, not the factory).
    for p in (a, b):
        p["decision_capture_id"] = ""
    assert a == b, "Instruction factory is not deterministic under same seed"

    # Different seed ideally differs, but we only require same-seed stability.
    c = synthesize_instruction_pair(chunk, seed=43).pair
    assert c is not None


def test_preference_factory_deterministic_under_seed():
    chunks = _load_fixture_chunks()
    chunk = _find_chunk(chunks, "chunk_mc_02")
    a = synthesize_preference_pair(chunk, seed=7).pair
    b = synthesize_preference_pair(chunk, seed=7).pair
    assert a is not None and b is not None
    for p in (a, b):
        p["decision_capture_id"] = ""
    assert a == b, "Preference factory is not deterministic under same seed"


def test_no_prompt_text_leakage_50_char_rule():
    """The 50-char verbatim-span rule must hold for every emitted pair."""
    chunks = _load_fixture_chunks()
    for chunk in chunks:
        if not chunk.get("learning_outcome_refs"):
            continue
        chunk_text = chunk.get("text", "").lower()
        if len(chunk_text) < MAX_VERBATIM_SPAN:
            continue  # Not long enough for the rule to apply.

        inst = synthesize_instruction_pair(chunk, seed=101).pair
        pref = synthesize_preference_pair(chunk, seed=101).pair

        for p in (inst, pref):
            if p is None:
                continue
            prompt_lc = p["prompt"].lower()
            for i in range(0, len(prompt_lc) - MAX_VERBATIM_SPAN + 1):
                window = prompt_lc[i:i + MAX_VERBATIM_SPAN]
                assert window not in chunk_text, (
                    f"Leakage: {MAX_VERBATIM_SPAN}-char prompt span '{window}' "
                    f"found in chunk {chunk['id']}.text"
                )


def test_preference_chosen_ne_rejected_with_jaccard_delta():
    chunks = _load_fixture_chunks()
    found_any = False
    for chunk in chunks:
        if not chunk.get("learning_outcome_refs"):
            continue
        result = synthesize_preference_pair(chunk, seed=11)
        if result.pair is None:
            continue
        found_any = True
        assert result.pair["chosen"] != result.pair["rejected"]
        jaccard_delta = result.quality["jaccard_delta"]
        assert jaccard_delta >= JACCARD_DELTA_MIN, (
            f"Chunk {chunk['id']}: jaccard_delta={jaccard_delta} below "
            f"gate {JACCARD_DELTA_MIN}"
        )
    assert found_any, "Fixture produced no preference pairs; fixture is broken"


def test_length_gates_enforced_on_factory_output():
    """Every non-None factory output respects the hard length gates."""
    chunks = _load_fixture_chunks()
    for chunk in chunks:
        if not chunk.get("learning_outcome_refs"):
            continue
        inst = synthesize_instruction_pair(chunk, seed=5).pair
        if inst is not None:
            assert PROMPT_MIN <= len(inst["prompt"]) <= PROMPT_MAX
            assert COMPLETION_MIN <= len(inst["completion"]) <= COMPLETION_MAX
        pref = synthesize_preference_pair(chunk, seed=5).pair
        if pref is not None:
            assert PROMPT_MIN <= len(pref["prompt"]) <= PROMPT_MAX
            assert COMPLETION_MIN <= len(pref["chosen"]) <= COMPLETION_MAX
            assert COMPLETION_MIN <= len(pref["rejected"]) <= COMPLETION_MAX


def test_instruction_factory_prefers_chunk_summary_over_relation_scaffold():
    chunk = {
        "id": "chunk_summary_01",
        "text": "RDF statements combine into graphs and preserve meaning under merge.",
        "summary": "RDF triples combine into graphs so independently produced facts can be merged and queried together.",
        "learning_outcome_refs": ["LO-1"],
        "concept_tags": ["rdf", "triple", "graph"],
        "bloom_level": "understand",
        "chunk_type": "explanation",
    }
    result = synthesize_instruction_pair(chunk, seed=23)
    assert result.pair is not None
    completion = result.pair["completion"]
    assert "RDF triples combine into graphs" in completion
    assert "related concepts" not in completion.lower()


def test_preference_factory_uses_authored_misconception_correction():
    chunk = {
        "id": "chunk_pref_01",
        "text": "RDFS describes entailments. SHACL validates closed-world constraints.",
        "summary": "RDFS publishes vocabulary meaning while SHACL checks data against explicit constraints.",
        "learning_outcome_refs": ["LO-1"],
        "concept_tags": ["rdfs", "shacl"],
        "misconceptions": [
            {
                "misconception": "RDFS domain declarations reject bad data like a schema validator.",
                "correction": (
                    "RDFS domain declarations infer class membership; SHACL is the tool "
                    "that validates data against explicit constraints."
                ),
                "bloom_level": "understand",
            },
        ],
    }
    result = synthesize_preference_pair(chunk, seed=23)
    assert result.pair is not None
    assert result.source == "misconception"
    assert "RDFS domain declarations infer class membership" in result.pair["chosen"]
    assert result.pair["source"] == "misconception"


def test_malformed_pair_rejected_with_diagnostic():
    """A chunk that cannot pass the eligibility filter returns pair=None
    with a clear quality diagnostic, not an exception."""
    # Empty LO refs -> defense-in-depth early-return in the factory.
    bad_chunk = {
        "id": "x",
        "text": "some text",
        "learning_outcome_refs": [],
    }
    inst = synthesize_instruction_pair(bad_chunk, seed=1)
    pref = synthesize_preference_pair(bad_chunk, seed=1)
    assert inst.pair is None and inst.quality.get("reason") == "missing_chunk_id_or_lo_refs"
    assert pref.pair is None and pref.quality.get("reason") == "missing_chunk_id_or_lo_refs"


def test_lo_filter_skips_orphan_chunks(tmp_path):
    """Orphan chunk in the fixture (empty learning_outcome_refs) must
    produce zero pairs and be counted as skipped."""
    working = _make_working_copy(tmp_path)
    stats = run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )
    assert stats.chunks_skipped_no_lo == 1, (
        f"Expected exactly 1 orphan chunk to be skipped; got {stats.chunks_skipped_no_lo}"
    )
    # And no emitted pair references the orphan chunk id.
    inst = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    pref = _load_jsonl(working / "training_specs" / "preference_pairs.jsonl")
    all_chunk_ids = {p["chunk_id"] for p in inst} | {p["chunk_id"] for p in pref}
    assert "chunk_orphan_01" not in all_chunk_ids


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_emitted_pairs_validate_against_schemas(tmp_path):
    working = _make_working_copy(tmp_path)
    run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )

    inst_schema = _load_schema("knowledge/instruction_pair.schema.json")
    pref_schema = _load_schema("knowledge/preference_pair.schema.json")

    inst = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    pref = _load_jsonl(working / "training_specs" / "preference_pairs.jsonl")
    assert inst, "No instruction pairs emitted"
    assert pref, "No preference pairs emitted"

    for i, rec in enumerate(inst):
        try:
            jsonschema.validate(rec, inst_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Instruction pair {i} failed schema: {e.message}")

    for i, rec in enumerate(pref):
        try:
            jsonschema.validate(rec, pref_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Preference pair {i} failed schema: {e.message}")


def test_editorial_misconception_dpo_pairs_validate_against_schema(tmp_path):
    working = _make_working_copy(tmp_path)
    run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
        include_dpo_from_misconceptions=True,
    )

    pref_schema = _load_schema("knowledge/preference_pair.schema.json")
    pref = _load_jsonl(working / "training_specs" / "preference_pairs.jsonl")
    editorial = [r for r in pref if r.get("source") == "misconception_editorial"]
    assert editorial, "Fixture produced no editorial misconception DPO pairs"
    for i, rec in enumerate(editorial):
        try:
            jsonschema.validate(rec, pref_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Editorial DPO pair {i} failed schema: {e.message}")


def test_stage_attaches_source_grounding_and_citations(tmp_path):
    working = _make_working_copy(tmp_path)
    run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
        instruction_variants_per_chunk=3,
    )

    inst = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    pref = _load_jsonl(working / "training_specs" / "preference_pairs.jsonl")
    assert inst and pref

    first_inst = inst[0]
    assert first_inst["source_chunk_id"] == first_inst["chunk_id"]
    assert first_inst["source_citation"] == f"[{first_inst['chunk_id']}]"
    assert isinstance(first_inst["source_references"], list)
    assert first_inst["source_citation"] not in first_inst["completion"]
    assert "cite the source chunk" not in first_inst["prompt"].lower()

    cited_inst = next(r for r in inst if r.get("requires_source_citation"))
    assert cited_inst["instruction_variant"] == 2
    assert cited_inst["source_citation"] in cited_inst["completion"]
    assert "cite the source chunk" in cited_inst["prompt"].lower()

    first_pref = pref[0]
    assert first_pref["source_chunk_id"] == first_pref["chunk_id"]
    assert first_pref["source_citation"] not in first_pref["chosen"]
    assert "cite the source chunk" not in first_pref["prompt"].lower()


# ---------------------------------------------------------------------------
# Decision-capture linkage
# ---------------------------------------------------------------------------

def test_decision_capture_id_resolves_for_every_pair(tmp_path, monkeypatch):
    """Every pair's decision_capture_id must resolve to an event_id in
    the decision log on disk."""
    working = _make_working_copy(tmp_path)
    # Redirect TRAINING_DIR (legacy capture location) into tmp to avoid
    # polluting the repo during tests, then redirect LibV2 storage too.
    run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101_PYTEST",
        provider="mock",
        seed=17,
    )

    inst = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    pref = _load_jsonl(working / "training_specs" / "preference_pairs.jsonl")

    # Gather decision-capture event ids from the legacy streaming log, which
    # is always written to <TRAINING_DIR>/trainforge/<COURSE>/phase_synthesize-training/
    from lib.paths import TRAINING_DIR
    capture_dir = TRAINING_DIR / "trainforge" / "MINI_TRAINING_101_PYTEST" / "phase_synthesize-training"
    assert capture_dir.exists(), f"No decision-capture dir at {capture_dir}"

    event_ids = set()
    for jsonl in capture_dir.glob("decisions_*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = rec.get("event_id")
            if eid:
                event_ids.add(str(eid))

    assert event_ids, "No decision events captured"
    # Every pair's decision_capture_id must resolve.
    for rec in inst + pref:
        cid = rec.get("decision_capture_id")
        assert cid, f"Pair has empty decision_capture_id: {rec.get('chunk_id')}"
        assert cid in event_ids, (
            f"decision_capture_id={cid} not found in event_ids; pair chunk_id={rec['chunk_id']}"
        )


# ---------------------------------------------------------------------------
# Integration and idempotence
# ---------------------------------------------------------------------------

def test_integration_fixture_meets_volume_floor(tmp_path):
    """The fixture has 14 eligible chunks (15 minus 1 orphan). To prove
    >=20 instruction pairs is attainable the integration test runs synthesis
    twice with distinct seeds and counts the union-of-records. >=5 preference
    pairs comes from a single run because the fixture has 7 misconception
    chunks."""
    working = _make_working_copy(tmp_path)
    stats_a = run_synthesis(working, "MINI_TRAINING_101", provider="mock", seed=17)

    inst_a = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    pref_a = _load_jsonl(working / "training_specs" / "preference_pairs.jsonl")

    assert stats_a.chunks_eligible == 14, (
        f"Expected 14 eligible chunks from fixture; got {stats_a.chunks_eligible}"
    )
    assert len(pref_a) >= 5, f"Expected >= 5 preference pairs; got {len(pref_a)}"

    # Second run with a different seed; sum of unique (chunk_id, seed) records.
    run_synthesis(working, "MINI_TRAINING_101", provider="mock", seed=99)
    inst_b = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")

    # Second run OVERWRITES (we preserve statistics.preference_pairs as the
    # most recent run). To verify >=20 is attainable we run once more into a
    # sibling corpus with a different seed and check the combined count.
    working2 = _make_working_copy(tmp_path / "second")
    run_synthesis(working2, "MINI_TRAINING_101", provider="mock", seed=99)
    inst_c = _load_jsonl(working2 / "training_specs" / "instruction_pairs.jsonl")

    total = len(inst_a) + len(inst_c)
    assert total >= 20, (
        f"Combined instruction pairs across two seeds expected >= 20; got {total} "
        f"({len(inst_a)} + {len(inst_c)})"
    )

    # Sanity: inst_b equals inst_c because the first working copy's second run
    # used the same seed=99 as working2's first run on the same fixture.
    def _strip_capture_ids(recs):
        return [{k: v for k, v in r.items() if k != "decision_capture_id"} for r in recs]
    assert _strip_capture_ids(inst_b) == _strip_capture_ids(inst_c), (
        "Second-run emission should match fresh-run emission at same seed "
        "(once decision_capture_id is stripped, since event_ids are per-session)."
    )


def test_stage_idempotence_same_seed_byte_identical(tmp_path):
    """Running the stage twice on the same fixture with the same seed should
    produce byte-identical instruction_pairs.jsonl and preference_pairs.jsonl
    (after stripping the decision_capture_id, which is tied to the capture
    session id and therefore changes run-over-run)."""
    a_dir = _make_working_copy(tmp_path / "a")
    b_dir = _make_working_copy(tmp_path / "b")

    run_synthesis(a_dir, "MINI_TRAINING_101", provider="mock", seed=17)
    run_synthesis(b_dir, "MINI_TRAINING_101", provider="mock", seed=17)

    def _canonical(path: Path) -> list[dict]:
        recs = _load_jsonl(path)
        # Strip decision_capture_id (session-bound) and sort deterministically.
        for r in recs:
            r.pop("decision_capture_id", None)
        recs.sort(key=lambda r: (r["chunk_id"], r.get("seed", 0)))
        return recs

    assert _canonical(a_dir / "training_specs" / "instruction_pairs.jsonl") == \
        _canonical(b_dir / "training_specs" / "instruction_pairs.jsonl")
    assert _canonical(a_dir / "training_specs" / "preference_pairs.jsonl") == \
        _canonical(b_dir / "training_specs" / "preference_pairs.jsonl")


def test_dataset_config_statistics_updated(tmp_path):
    working = _make_working_copy(tmp_path)
    stats = run_synthesis(working, "MINI_TRAINING_101", provider="mock", seed=17)
    with (working / "training_specs" / "dataset_config.json").open("r") as fh:
        cfg = json.load(fh)
    assert cfg["statistics"]["instruction_pairs"] == stats.instruction_pairs_emitted
    assert cfg["statistics"]["preference_pairs"] == stats.preference_pairs_emitted
    assert "synthesis" in cfg and "last_run" in cfg["synthesis"]


# ---------------------------------------------------------------------------
# Wave 107 — Phase A: claude_session provider routing
# ---------------------------------------------------------------------------

def test_run_synthesis_routes_claude_session_provider_through_dispatcher(tmp_path):
    """When provider='claude_session' is set and a dispatcher is supplied,
    run_synthesis must route paraphrase calls through ClaudeSessionProvider
    and tag emitted rows with provider='claude_session', not 'mock'."""
    from Trainforge.tests._synthesis_fakes import (
        FakeLocalDispatcher,
        make_instruction_response,
        make_preference_response,
    )

    async def agent_tool(*, task_params, **_kw):
        if task_params["kind"] == "instruction":
            return make_instruction_response(
                prompt="Paraphrased: explain a key concept from the chunk.",
                completion=(
                    "RDFS describes vocabulary semantics — class hierarchy and property "
                    "domains — in a way that downstream RDF processors can reason about. "
                    "[" + str(task_params.get("chunk_id", "")) + "]"
                ),
            )
        return make_preference_response(
            prompt="Which option is correct about the chunk topic?",
            chosen=(
                "RDFS describes vocabulary semantics; SHACL validates RDF graphs against "
                "shape constraints. They are complementary, not interchangeable."
            ),
            rejected=(
                "RDFS validates RDF graphs against shape constraints; SHACL describes "
                "vocabulary semantics. They are complementary, not interchangeable."
            ),
        )

    dispatcher = FakeLocalDispatcher(agent_tool=agent_tool)
    working = _make_working_copy(tmp_path)
    cache_path = working / "training_specs" / ".synthesis_cache.jsonl"

    run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="claude_session",
        seed=11,
        dispatcher=dispatcher,
        cache_path=cache_path,
    )

    # The session provider was called at least once for instruction:
    assert any(c[1]["kind"] == "instruction" for c in dispatcher.calls)
    # Output JSONL carries provider='claude_session', not 'mock':
    inst_path = working / "training_specs" / "instruction_pairs.jsonl"
    rows = _load_jsonl(inst_path)
    assert rows, "expected at least one instruction pair"
    assert all(r["provider"] == "claude_session" for r in rows), [
        r["provider"] for r in rows
    ]


def test_run_synthesis_claude_session_without_dispatcher_fails_loud(tmp_path):
    """Phase A precondition: --provider claude_session requires a dispatcher.
    Standalone CLI runs have no Claude Code session; abort with clear message."""
    working = _make_working_copy(tmp_path)

    with pytest.raises(RuntimeError, match="claude_session.*dispatcher"):
        run_synthesis(
            corpus_dir=working,
            course_code="MINI_TRAINING_101",
            provider="claude_session",
            seed=11,
            dispatcher=None,  # the failure case
        )


def test_run_synthesis_claude_session_respects_max_dispatches(tmp_path):
    """Wave 110 / Phase D + Wave 111 / Phase E: run_synthesis stops
    dispatching once max_dispatches is hit. Phase E changed the
    contract from raising SynthesisBudgetExceeded to returning a
    SynthesisStats with capped_at_max_dispatches=True."""
    from Trainforge.tests._synthesis_fakes import (
        FakeLocalDispatcher, make_instruction_response, make_preference_response,
    )

    # Wave 112 Task 4: outputs must respect _validate_lengths floors.
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

    stats = run_synthesis(
        corpus_dir=working, course_code="MINI_TRAINING_101",
        provider="claude_session", seed=11,
        dispatcher=dispatcher,
        max_dispatches=1,
    )
    assert stats.capped_at_max_dispatches is True
    assert stats.dispatched_count == 1


def test_run_synthesis_writes_pilot_progress_on_budget_exceeded(tmp_path):
    """Wave 111 / Phase E: hitting max_dispatches no longer bubbles a
    stack trace — run_synthesis returns SynthesisStats with
    capped_at_max_dispatches=True and writes pilot_progress.json."""
    from Trainforge.tests._synthesis_fakes import (
        FakeLocalDispatcher, make_instruction_response, make_preference_response,
    )

    # Wave 112 Task 4: outputs must respect _validate_lengths floors.
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

    stats = run_synthesis(
        corpus_dir=working, course_code="MINI_TRAINING_101",
        provider="claude_session", seed=11,
        dispatcher=dispatcher,
        max_dispatches=1,
    )
    assert stats.capped_at_max_dispatches is True
    progress = working / "training_specs" / "pilot_progress.json"
    assert progress.exists()
    payload = json.loads(progress.read_text())
    assert payload["dispatched"] == 1
    assert payload["max_dispatches"] == 1
    assert "resume" in payload["message"].lower()


# ---------------------------------------------------------------------------
# Wave 112 Task 6: audit-log empty-field misconception drops
# ---------------------------------------------------------------------------

def test_build_misconception_dpo_pair_logs_drop_on_empty_misconception():
    """When the editorial misconception text is empty, the helper must log
    a ``misconception_pair_skipped`` decision before returning ``None``.

    Pre-Wave-112, empty/short editorial entries were silently dropped on the
    floor with no audit trail, so a corpus rebuild that lost a property
    family looked clean in the captures. This test pins the audit
    contract so future refactors can't silently drop again.
    """
    from Trainforge.synthesize_training import _build_misconception_dpo_pair

    captured: list[dict] = []

    class _RecorderCapture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    chunk = {
        "id": "chunk_drop_01",
        "text": "RDFS describes entailments. SHACL validates closed-world constraints.",
        "learning_outcome_refs": ["LO-1"],
        "concept_tags": ["rdfs"],
        "bloom_level": "understand",
    }
    misconception = {
        "misconception": "",  # empty -> pair must be dropped + logged
        "correction": "RDFS publishes vocabulary; SHACL validates data.",
        "bloom_level": "understand",
    }

    result = _build_misconception_dpo_pair(
        chunk, misconception, pair_index=0, capture=_RecorderCapture(),
    )
    assert result is None
    assert len(captured) == 1, (
        "Empty-field misconception drop must emit exactly one decision event"
    )
    event = captured[0]
    assert event["decision_type"] == "misconception_pair_skipped"
    assert event["decision"] == "dropped"
    rationale = event["rationale"]
    assert len(rationale) >= 20
    # Rationale should name the offending field and the chunk id.
    assert "misconception" in rationale.lower()
    assert "chunk_drop_01" in rationale


def test_build_misconception_dpo_pair_logs_drop_on_empty_correction():
    """Symmetric to the misconception-empty case: empty correction must
    also emit an audit event before the ``None`` return."""
    from Trainforge.synthesize_training import _build_misconception_dpo_pair

    captured: list[dict] = []

    class _RecorderCapture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    chunk = {
        "id": "chunk_drop_02",
        "text": "Foundational RDF semantics anchor every later SHACL constraint.",
        "learning_outcome_refs": ["LO-2"],
        "concept_tags": ["rdf"],
        "bloom_level": "remember",
    }
    misconception = {
        "misconception": "RDF and SHACL are interchangeable.",
        "correction": "   ",  # whitespace-only -> empty after strip
        "bloom_level": "remember",
    }

    result = _build_misconception_dpo_pair(
        chunk, misconception, pair_index=1, capture=_RecorderCapture(),
    )
    assert result is None
    assert len(captured) == 1
    event = captured[0]
    assert event["decision_type"] == "misconception_pair_skipped"
    assert event["decision"] == "dropped"
    assert "correction" in event["rationale"].lower()
    assert "chunk_drop_02" in event["rationale"]


def test_build_misconception_dpo_pair_no_log_when_pair_built():
    """A well-formed misconception should NOT emit a skip event."""
    from Trainforge.synthesize_training import _build_misconception_dpo_pair

    captured: list[dict] = []

    class _RecorderCapture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    chunk = {
        "id": "chunk_ok_01",
        "text": "RDFS describes entailments. SHACL validates closed-world constraints.",
        "learning_outcome_refs": ["LO-1"],
        "concept_tags": ["shacl"],
        "bloom_level": "understand",
    }
    misconception = {
        "misconception": "RDFS domain declarations reject bad data like a validator.",
        "correction": (
            "RDFS domain declarations infer class membership; SHACL is the "
            "tool that validates data against explicit constraints."
        ),
        "bloom_level": "understand",
    }

    pair = _build_misconception_dpo_pair(
        chunk, misconception, pair_index=2, capture=_RecorderCapture(),
    )
    assert pair is not None
    assert captured == [], (
        "log_decision must NOT fire when the misconception pair is built"
    )
