"""Tests for the Marketable-v1 A5 corpus-generalization defaults-on path.

``WorkflowRunner._apply_corpus_generalization_defaults`` turns the
corpus-generalization feature set ON for ``textbook_to_course`` /
``course_generation`` runs so a fresh CLI/GUI run gets page-level concept
tags, the measured graph-shaping quartet, dynamic CURIEs (via three-stage
synthesis), and LO-refs by default — the features the product is sold on.

These tests drive the helper directly with the env scrubbed, so they assert
flag resolution (env unset -> new default; explicit legacy value -> honored)
without any orchestrator / LLM / live provider. A workflow-level test asserts
the auto-on set is carried for a real ``run_workflow`` env.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from MCP.core.workflow_runner import (
    _CORPUS_GENERALIZATION_ENV_DEFAULTS,
    _DISABLE_CORPUS_GENERALIZATION_ENV,
    _TEXTBOOK_SYNTHESIS_PROVIDER_ENV,
    _TRAINFORGE_SYNTHESIS_PROVIDER_ENV,
    WorkflowRunner,
)

# Every env var this helper can set — scrubbed per test for a known baseline.
_ALL_A5_ENVS = (
    *_CORPUS_GENERALIZATION_ENV_DEFAULTS.keys(),
    _TEXTBOOK_SYNTHESIS_PROVIDER_ENV,
    _TRAINFORGE_SYNTHESIS_PROVIDER_ENV,
)


def _make_runner() -> WorkflowRunner:
    return WorkflowRunner(executor=MagicMock(), config=MagicMock())


def _clear_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _ALL_A5_ENVS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv(_DISABLE_CORPUS_GENERALIZATION_ENV, raising=False)


# --------------------------------------------------------------------------
# Master opt-out: a deterministic fixture-contract run skips the whole set.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on"])
def test_disable_env_skips_entire_a5_set(monkeypatch, truthy):
    """The master opt-out suppresses every A5 env — the measured graph-shaping
    flags AND the licensing-sensitive synthesis-provider envs — so a
    deterministic fixture-contract run (the e2e pipeline test) never dispatches
    live-LLM textbook synthesis mid-pipeline."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv(_DISABLE_CORPUS_GENERALIZATION_ENV, truthy)
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert applied == {}
    for env_var in _ALL_A5_ENVS:
        assert os.environ.get(env_var) is None


# --------------------------------------------------------------------------
# Measured-config compliance: chunk-local tags must NOT be in the auto-on set.
# --------------------------------------------------------------------------


def test_chunk_local_tags_is_not_in_the_auto_on_set():
    """Measured-best config wins: PAGE-level tags. Chunk-local tags fragment
    the graph, so the flag must never be auto-enabled."""
    assert "TRAINFORGE_CHUNK_LOCAL_TAGS" not in _CORPUS_GENERALIZATION_ENV_DEFAULTS


def test_measured_graph_shaping_quartet_is_complete():
    """prune + fragment-filter + merge + fan-out-cap all present, page-level
    plus the recovery paths."""
    expected = {
        "TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS": "true",
        "TRAINFORGE_SEED_TECH_CONCEPTS": "true",
        "TRAINFORGE_FILTER_FRAGMENT_CONCEPTS": "true",
        "TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE": "true",
        "TRAINFORGE_MERGE_DUPLICATE_CONCEPTS": "true",
        "TRAINFORGE_INTRA_CHUNK_LINKS": "true",
        "TRAINFORGE_RELATED_FANOUT_CAP": "8",
        "TRAINFORGE_NORMALIZE_LABELS": "true",
        # W3 — cooccurrence pair-counting aggregated to the page level.
        "TRAINFORGE_COOCCURRENCE_GROUP_BY": "page",
        # M4 — degenerate-grouping guard for the page aggregation above (steps
        # page->section->chunk when the corpus collapses into <3 real groups, so
        # a single-lesson_id PDF still yields cooccurrence edges).
        "TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK": "true",
        "TRAINFORGE_DROP_FRONTMATTER": "true",
        "TRAINFORGE_LEXICAL_CONCEPT_SEEDS": "true",
        "TRAINFORGE_OBJECTIVE_QUALITY_GATE": "true",
        # Vendor-parity D1 — page-level key-concept fallback for markup-less
        # GLM-OCR accessible HTML (fills EMPTY key_concepts only; page-level,
        # never chunk-local). See lib/ontology/page_concept_fallback.py.
        "TRAINFORGE_PAGE_CONCEPT_FALLBACK": "true",
        # Vendor-parity D4 — relocate stranded next-section heading tails
        # ("...Figure. 1.1 EXERCISES" -> marker opens the following chunk).
        # See Trainforge/chunker/stranded_heading_tails.py.
        "TRAINFORGE_RELOCATE_STRANDED_HEADINGS": "true",
        # Defensive heading-sanity filter — repairs a chunk's section_heading to
        # its nearest clean ancestor when the upstream classifier mis-tagged
        # answer-key / exercise / numeric noise as a heading (chunk + retrieval
        # display quality; see lib/chunk_heading_sanity.py).
        "TRAINFORGE_HEADING_SANITY_FILTER": "true",
        # Campaign-validated 2026-07 promotions — deterministic, no LLM
        # provider/model selection (see workflow_runner comments per flag).
        "ED4ALL_OBJECTIVE_CITATION_RESELECT": "true",
        "ED4ALL_OBJECTIVE_DEDUP_LEXICAL": "true",
        "ED4ALL_CHUNK_ROLE_DIVERSIFY": "true",
        "ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE": "true",
        # Campaign flag-coverage audit 2026-07-22 promotions — PORTABLE
        # (deterministic / warning-day-1 / read-only or a KG-shaping arm; none
        # selects an LLM provider/model/seat). Owner-validated on book 1.
        "TRAINFORGE_PREREQ_LO_ADJACENT_ONLY": "true",
        "ED4ALL_KG_PREREQ_HEALTH": "1",
        "ED4ALL_CONCEPT_COVERAGE": "1",
        "ED4ALL_INTELLIGENCE_RUBRIC": "1",
        "ED4ALL_RETRIEVAL_INTERLEAVE": "1",
        "ED4ALL_TRIANGLE_FLOOR": "1",
        "ED4ALL_WORKED_EXAMPLE_FLOOR": "1",
        "ED4ALL_BLOOM_SPREAD_FLOOR": "1",
        "ED4ALL_OBJECTIVE_SPECIFICITY": "1",
        "ED4ALL_OBJECTIVE_BLOOM_RELEVEL": "1",
        "ED4ALL_TO_SOURCE_GROUNDING": "1",
        "ED4ALL_BLOOM_DISTRIBUTION": "1",
        "ED4ALL_EMBED_OVERFLOW_GUARD": "1",
        "ED4ALL_KEY_TERMS_PAGE": "1",
        "ED4ALL_FAQ_PAGE": "1",
        "ED4ALL_KEYTERM_DEF_QUALITY": "1",
        "ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE": "1",
        "ED4ALL_BLOCK_QUALITY_RUBRIC": "1",
        "ED4ALL_BLOCK_QUALITY_SHADOW": "1",
        "TRAINFORGE_EDGE_NLI": "1",
        "TRAINFORGE_CONTRADICTED_EDGE_POLICY": "decay",
    }
    assert _CORPUS_GENERALIZATION_ENV_DEFAULTS == expected


# --------------------------------------------------------------------------
# env unset -> new effective default applied.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("workflow_type", ["textbook_to_course", "course_generation"])
def test_env_unset_gets_new_defaults(monkeypatch, workflow_type):
    _clear_envs(monkeypatch)
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults(workflow_type)

    import os

    # Every measured-best flag now resolves to its new default.
    for env_var, value in _CORPUS_GENERALIZATION_ENV_DEFAULTS.items():
        assert os.environ.get(env_var) == value
        assert applied[env_var] == value
    # Fan-out cap is the measured operator value.
    assert os.environ.get("TRAINFORGE_RELATED_FANOUT_CAP") == "8"
    # Chunk-local tags stays unset -> page-level tags emit.
    assert os.environ.get("TRAINFORGE_CHUNK_LOCAL_TAGS") is None


def test_synthesis_provider_defaults_to_local_when_no_llm_provider(monkeypatch):
    _clear_envs(monkeypatch)
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(_TEXTBOOK_SYNTHESIS_PROVIDER_ENV) == "local"
    assert applied[_TEXTBOOK_SYNTHESIS_PROVIDER_ENV] == "local"


def test_training_synthesis_provider_defaults_to_local_when_no_llm_provider(monkeypatch):
    """Marketable-v1 D4 — the TRAINING-PAIR synthesis provider is the CLI mirror
    of the GUI authoring-route fill: it must default to the license-clean
    ``local`` so a fresh CLI run never routes training-pair synthesis through
    the Claude Code subagent by default."""
    _clear_envs(monkeypatch)
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "local"
    assert applied[_TRAINFORGE_SYNTHESIS_PROVIDER_ENV] == "local"


def test_synthesis_provider_resolves_from_llm_provider(monkeypatch):
    """The run's global routing provider wins over the license-clean default —
    for BOTH the textbook-structure and the training-pair synthesis envs."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "together")
    runner = _make_runner()

    runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(_TEXTBOOK_SYNTHESIS_PROVIDER_ENV) == "together"
    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "together"


def test_training_synthesis_provider_explicit_value_honored(monkeypatch):
    """setdefault — an operator pinning the training-synthesis provider (even to
    an Anthropic-family value, which the run_synthesis gate then guards) is
    honored verbatim; this helper only fills an unset env."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV, "anthropic")
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "anthropic"
    assert _TRAINFORGE_SYNTHESIS_PROVIDER_ENV not in applied


# --------------------------------------------------------------------------
# LICENSING — a restricted LLM_PROVIDER never AUTO-resolves the training seat
# to a ToS-restricted provider (the corpus the SLM is a derivative work of).
# --------------------------------------------------------------------------


def test_nvidia_llm_provider_pins_training_seat_local_textbook_follows(monkeypatch):
    """LLM_PROVIDER=nvidia — the AUTO-resolved TRAINING-PAIR seat is pinned to
    the license-clean ``local`` (NVIDIA-hosted Llama-3.3 is ToS-restricted for
    training data), while the textbook seat still follows the authoring
    provider (nvidia). Only the training seat is guarded."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "local"
    assert applied[_TRAINFORGE_SYNTHESIS_PROVIDER_ENV] == "local"
    assert os.environ.get(_TEXTBOOK_SYNTHESIS_PROVIDER_ENV) == "nvidia"
    assert applied[_TEXTBOOK_SYNTHESIS_PROVIDER_ENV] == "nvidia"


def test_anthropic_llm_provider_pins_training_seat_local(monkeypatch):
    """LLM_PROVIDER=anthropic — the AUTO-resolved training seat is pinned to
    ``local`` too (Anthropic Commercial/Consumer Terms restrict training-data
    use); the textbook seat still follows authoring."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "local"
    assert applied[_TRAINFORGE_SYNTHESIS_PROVIDER_ENV] == "local"
    assert os.environ.get(_TEXTBOOK_SYNTHESIS_PROVIDER_ENV) == "anthropic"


def test_corpus_gen_then_authoring_route_training_seat_stays_local(monkeypatch):
    """Real run_workflow ORDERING: corpus-generalization runs FIRST, then
    authoring-route fill. With LLM_PROVIDER=nvidia + COURSEFORGE_TWO_PASS=true,
    the training seat must be ``local`` at the END of both passes — corpus-gen's
    license-clean pin holds because the authoring-route fill is a setdefault
    that no longer touches the already-set seat."""
    _clear_envs(monkeypatch)
    monkeypatch.delenv("COURSEFORGE_TWO_PASS", raising=False)
    monkeypatch.delenv("TEXTBOOK_SYNTHESIS_MODEL", raising=False)
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEPLANNER_PROVIDER", raising=False)
    monkeypatch.delenv("TRAINFORGE_ASSESSMENT_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    runner = _make_runner()

    # Order mirrors WorkflowRunner.run_workflow.
    runner._apply_corpus_generalization_defaults("textbook_to_course")
    runner._apply_authoring_route_env("textbook_to_course")

    import os

    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "local"


def test_explicit_nvidia_training_seat_export_honored_verbatim(monkeypatch):
    """REGRESSION — an EXPLICIT operator export of the training seat to nvidia
    is honored verbatim (the setdefault skip wins; the licensing guard only
    touches the AUTO-resolved seat). The downstream run_synthesis gate is the
    fail-closed backstop for a direct export."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV, "nvidia")
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "nvidia"
    assert _TRAINFORGE_SYNTHESIS_PROVIDER_ENV not in applied


def test_together_llm_provider_training_seat_unchanged(monkeypatch):
    """BYTE-IDENTICAL guard — a license-clean LLM_PROVIDER (together) is NOT in
    the restricted set, so the training seat resolves to ``together`` exactly as
    before. Only anthropic/nvidia auto-resolve changes."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "together")
    runner = _make_runner()

    runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "together"
    assert os.environ.get(_TEXTBOOK_SYNTHESIS_PROVIDER_ENV) == "together"


# --------------------------------------------------------------------------
# explicit legacy value -> honored verbatim (setdefault semantics).
# --------------------------------------------------------------------------


def test_explicit_legacy_value_is_honored(monkeypatch):
    """An operator pinning a flag OFF keeps it off — only unset envs fill."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv("TRAINFORGE_MERGE_DUPLICATE_CONCEPTS", "false")
    monkeypatch.setenv("TRAINFORGE_RELATED_FANOUT_CAP", "0")
    monkeypatch.setenv(_TEXTBOOK_SYNTHESIS_PROVIDER_ENV, "anthropic")
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get("TRAINFORGE_MERGE_DUPLICATE_CONCEPTS") == "false"
    assert os.environ.get("TRAINFORGE_RELATED_FANOUT_CAP") == "0"
    assert os.environ.get(_TEXTBOOK_SYNTHESIS_PROVIDER_ENV) == "anthropic"
    # The honored flags are NOT reported as applied by this call.
    assert "TRAINFORGE_MERGE_DUPLICATE_CONCEPTS" not in applied
    assert "TRAINFORGE_RELATED_FANOUT_CAP" not in applied
    assert _TEXTBOOK_SYNTHESIS_PROVIDER_ENV not in applied
    # ...but the un-pinned flags still get their defaults.
    assert os.environ.get("TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS") == "true"


# --------------------------------------------------------------------------
# Campaign-validated 2026-07 promotions: the four deterministic ED4ALL_* flags.
# --------------------------------------------------------------------------

_CAMPAIGN_2026_07_FLAGS = (
    "ED4ALL_OBJECTIVE_CITATION_RESELECT",
    "ED4ALL_OBJECTIVE_DEDUP_LEXICAL",
    "ED4ALL_CHUNK_ROLE_DIVERSIFY",
    "ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE",
)


@pytest.mark.parametrize("workflow_type", ["textbook_to_course", "course_generation"])
def test_campaign_2026_07_flags_auto_on_for_pipeline(monkeypatch, workflow_type):
    """The four campaign-validated deterministic flags fill "true" on a
    pipeline run (setdefault applied when the env is unset)."""
    _clear_envs(monkeypatch)
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults(workflow_type)

    import os

    for env_var in _CAMPAIGN_2026_07_FLAGS:
        assert os.environ.get(env_var) == "true"
        assert applied[env_var] == "true"


@pytest.mark.parametrize("env_var", _CAMPAIGN_2026_07_FLAGS)
def test_campaign_2026_07_explicit_env_value_honored(monkeypatch, env_var):
    """setdefault — an operator (or the live campaign env) pinning a promoted
    flag, on OR off, is honored verbatim; the helper never overwrites."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv(env_var, "0")
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(env_var) == "0"
    assert env_var not in applied


@pytest.mark.parametrize("workflow_type", ["rag_training", "trainforge_train"])
def test_campaign_2026_07_flags_untouched_for_non_pipeline(monkeypatch, workflow_type):
    """Non-pipeline workflows (and bare library calls that never reach
    run_workflow) keep the legacy default-off contract for the promoted set."""
    _clear_envs(monkeypatch)
    runner = _make_runner()

    runner._apply_corpus_generalization_defaults(workflow_type)

    import os

    for env_var in _CAMPAIGN_2026_07_FLAGS:
        assert os.environ.get(env_var) is None


# --------------------------------------------------------------------------
# Campaign flag-coverage audit 2026-07-22 promotions: the 21 PORTABLE
# owner-validated flags now auto-on for pipeline runs. Values are taken
# verbatim from the campaign env (ED4ALL_* booleans as "1",
# TRAINFORGE_PREREQ_LO_ADJACENT_ONLY as "true", the edge policy as "decay").
# --------------------------------------------------------------------------

_CAMPAIGN_2026_07_22_FLAGS = {
    "TRAINFORGE_PREREQ_LO_ADJACENT_ONLY": "true",
    "ED4ALL_KG_PREREQ_HEALTH": "1",
    "ED4ALL_CONCEPT_COVERAGE": "1",
    "ED4ALL_INTELLIGENCE_RUBRIC": "1",
    "ED4ALL_RETRIEVAL_INTERLEAVE": "1",
    "ED4ALL_TRIANGLE_FLOOR": "1",
    "ED4ALL_WORKED_EXAMPLE_FLOOR": "1",
    "ED4ALL_BLOOM_SPREAD_FLOOR": "1",
    "ED4ALL_OBJECTIVE_SPECIFICITY": "1",
    "ED4ALL_OBJECTIVE_BLOOM_RELEVEL": "1",
    "ED4ALL_TO_SOURCE_GROUNDING": "1",
    "ED4ALL_BLOOM_DISTRIBUTION": "1",
    "ED4ALL_EMBED_OVERFLOW_GUARD": "1",
    "ED4ALL_KEY_TERMS_PAGE": "1",
    "ED4ALL_FAQ_PAGE": "1",
    "ED4ALL_KEYTERM_DEF_QUALITY": "1",
    "ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE": "1",
    "ED4ALL_BLOCK_QUALITY_RUBRIC": "1",
    "ED4ALL_BLOCK_QUALITY_SHADOW": "1",
    "TRAINFORGE_EDGE_NLI": "1",
    "TRAINFORGE_CONTRADICTED_EDGE_POLICY": "decay",
}


@pytest.mark.parametrize("workflow_type", ["textbook_to_course", "course_generation"])
def test_campaign_2026_07_22_flags_auto_on_for_pipeline(monkeypatch, workflow_type):
    """Every 2026-07-22 promotion fills its campaign value on a pipeline run
    (setdefault applied when the env is unset)."""
    _clear_envs(monkeypatch)
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults(workflow_type)

    import os

    for env_var, value in _CAMPAIGN_2026_07_22_FLAGS.items():
        assert os.environ.get(env_var) == value
        assert applied[env_var] == value


@pytest.mark.parametrize("env_var", sorted(_CAMPAIGN_2026_07_22_FLAGS))
def test_campaign_2026_07_22_explicit_env_value_honored(monkeypatch, env_var):
    """OPERATOR-OVERRIDE-WINS — setdefault: an operator pinning a promoted flag
    OFF ("0") — or to any explicit value — is honored verbatim; the helper never
    overwrites and never reports it as applied."""
    _clear_envs(monkeypatch)
    monkeypatch.setenv(env_var, "0")
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults("textbook_to_course")

    import os

    assert os.environ.get(env_var) == "0"
    assert env_var not in applied


@pytest.mark.parametrize("workflow_type", ["rag_training", "trainforge_train"])
def test_campaign_2026_07_22_flags_untouched_for_non_pipeline(
    monkeypatch, workflow_type
):
    """BARE-LIBRARY-CALL-UNCHANGED — non-pipeline workflows (and any bare
    lib/Trainforge call that never reaches run_workflow) keep the legacy
    default-off contract for the whole 2026-07-22 promotion set."""
    _clear_envs(monkeypatch)
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults(workflow_type)

    import os

    assert applied == {}
    for env_var in _CAMPAIGN_2026_07_22_FLAGS:
        assert os.environ.get(env_var) is None


# --------------------------------------------------------------------------
# scope: non-pipeline workflows are a no-op.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "workflow_type", ["rag_training", "trainforge_train"]
)
def test_non_pipeline_workflow_is_noop(monkeypatch, workflow_type):
    _clear_envs(monkeypatch)
    runner = _make_runner()

    applied = runner._apply_corpus_generalization_defaults(workflow_type)

    import os

    assert applied == {}
    for env_var in _ALL_A5_ENVS:
        assert os.environ.get(env_var) is None


# --------------------------------------------------------------------------
# Workflow-level: run_workflow carries the auto-on set for a real run.
# Mirrors A3's run_service / guardrail integration-test pattern: drive
# run_workflow far enough to reach the helper, then halt deterministically.
# --------------------------------------------------------------------------


class _Halt(Exception):
    """Sentinel raised right after the A5 helper fires to stop run_workflow."""


async def test_run_workflow_applies_corpus_generalization_for_pipeline(
    tmp_path, monkeypatch
):
    import json
    import os

    from MCP.core import workflow_runner as wr

    _clear_envs(monkeypatch)

    # Redirect workflow state to a scratch dir + seed a real state file.
    monkeypatch.setattr(wr, "STATE_PATH", tmp_path)
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(parents=True)
    workflow_id = "WF-A5-TEST"
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "type": "textbook_to_course",
                "params": {"course_name": "A5_TEST"},
                "phase_outputs": {},
            }
        )
    )

    runner = _make_runner()
    # Config returns a truthy workflow config so run_workflow proceeds past
    # the get_workflow guard to the A5 helper.
    runner.config.get_workflow.return_value = MagicMock(phases=[])
    # Stop deterministically right after the A5 helper runs (it is invoked
    # before _topological_sort).
    monkeypatch.setattr(runner, "_restore_resume_phase_outputs", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner, "_topological_sort", lambda *_a, **_k: (_ for _ in ()).throw(_Halt())
    )

    with pytest.raises(_Halt):
        await runner.run_workflow(workflow_id)

    # The blessed auto-on set was carried into the run env.
    assert os.environ.get("TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS") == "true"
    assert os.environ.get("TRAINFORGE_RELATED_FANOUT_CAP") == "8"
    assert os.environ.get("TRAINFORGE_NORMALIZE_LABELS") == "true"
    assert os.environ.get(_TEXTBOOK_SYNTHESIS_PROVIDER_ENV) == "local"
    # D4: training-pair synthesis defaults to the license-clean provider too —
    # a real CLI run no longer routes training-pair synthesis through Claude.
    assert os.environ.get(_TRAINFORGE_SYNTHESIS_PROVIDER_ENV) == "local"
    # Page-level tags: chunk-local stays off.
    assert os.environ.get("TRAINFORGE_CHUNK_LOCAL_TAGS") is None
