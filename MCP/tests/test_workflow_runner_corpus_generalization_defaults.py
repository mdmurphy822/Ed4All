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
        "TRAINFORGE_LEXICAL_CONCEPT_SEEDS": "true",
        "TRAINFORGE_OBJECTIVE_QUALITY_GATE": "true",
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
# scope: non-pipeline workflows are a no-op.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "workflow_type", ["intake_remediation", "batch_dart", "rag_training", "trainforge_train"]
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
