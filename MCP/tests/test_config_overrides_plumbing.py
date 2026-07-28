"""Regression — ``--config-overrides`` reaches the ``training`` phase.

``Trainforge/training/configs/nemotron3-nano-30b.yaml`` ships
``dpo_learning_rate: null`` deliberately, and
``Trainforge/training/peft_trainer.py`` RAISES rather than reusing the SFT
rate, telling the operator to "supply the selected value through
--config-overrides". No such route existed through the pipeline:
``_get_phase_param_routing("training")`` carried only ``course_code`` and
``base_model``, so DPO on that base could not be started by ``ed4all run`` at
all.

These tests pin the whole route: CLI flag -> parse-time validation ->
workflow_params -> the phase's ``inputs_from`` block -> the tool schema ->
``run_training`` -> ``TrainingRunner``. With the flag absent nothing is
recorded and the per-base YAML stays the sole source, so an unflagged run is
byte-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402

PARAM = "config_overrides"


def _training_phase() -> Dict[str, Any]:
    data = yaml.safe_load(
        (PROJECT_ROOT / "config" / "workflows.yaml").read_text(encoding="utf-8")
    )
    for phase in data["workflows"]["textbook_to_course"]["phases"]:
        if phase.get("name") == "training":
            return phase
    raise AssertionError("training phase missing from workflows.yaml")


# --------------------------------------------------------------------- #
# Route                                                                  #
# --------------------------------------------------------------------- #


def test_workflows_yaml_routes_the_param_from_workflow_params() -> None:
    routes = _training_phase().get("inputs_from") or []
    matches = [r for r in routes if r.get("param") == PARAM]
    assert matches, (
        f"the training phase must declare an inputs_from route for {PARAM}; "
        "without it the CLI flag can never reach the phase handler and the "
        "Nemotron Nano DPO path stays unreachable"
    )
    assert matches[0]["source"] == "workflow_params"
    assert matches[0]["key"] == PARAM


def test_phase_param_routing_resolves_the_param() -> None:
    """Through the production routing resolver, not the raw YAML."""
    from MCP.core.workflow_runner import _get_phase_param_routing

    routing = _get_phase_param_routing("training")
    assert routing.get(PARAM) == ("workflow_params", PARAM)
    # The pre-existing routes must survive alongside it.
    assert routing.get("course_code") == ("workflow_params", "course_name")
    assert routing.get("base_model") == ("workflow_params", "base_model")


def test_tool_schema_lists_the_param_as_optional_with_no_default() -> None:
    from MCP.core.tool_schemas import get_defaults, get_optional_params

    assert PARAM in get_optional_params("run_training")
    assert PARAM not in get_defaults("run_training"), (
        "a default would inject the key on every dispatch; an absent key must "
        "leave the per-base YAML untouched"
    )


# --------------------------------------------------------------------- #
# Handler                                                                #
# --------------------------------------------------------------------- #


class _RecordingRunner:
    """Stand-in for ``TrainingRunner`` that records kwargs and stops."""

    last_kwargs: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = dict(kwargs)
        raise RuntimeError("stop-after-capture")


@pytest.fixture()
def recorded_runner(monkeypatch: pytest.MonkeyPatch):
    import Trainforge.training as training

    _RecordingRunner.last_kwargs = {}
    monkeypatch.setattr(training, "TrainingRunner", _RecordingRunner)
    return _RecordingRunner


async def _dispatch(**kwargs: Any) -> Dict[str, Any]:
    handler = _build_tool_registry()["run_training"]
    return json.loads(await handler(course_name="demo-course", **kwargs))


@pytest.mark.asyncio
async def test_handler_forwards_a_validated_override_dict(
    recorded_runner: Any,
) -> None:
    """The happy path the whole feature exists for."""
    await _dispatch(
        base_model="nemotron3-nano-30b",
        config_overrides={"dpo_learning_rate": 5e-7},
    )
    assert recorded_runner.last_kwargs["config_overrides"] == {
        "dpo_learning_rate": 5e-7
    }


@pytest.mark.asyncio
async def test_handler_accepts_an_inline_spec_and_coerces_types(
    recorded_runner: Any,
) -> None:
    """Routed params can arrive as strings; the parser owns the cast."""
    await _dispatch(
        base_model="nemotron3-nano-30b",
        config_overrides="dpo_learning_rate=5e-7,epochs=2",
    )
    forwarded = recorded_runner.last_kwargs["config_overrides"]
    assert forwarded == {"dpo_learning_rate": 5e-7, "epochs": 2}
    assert isinstance(forwarded["dpo_learning_rate"], float)
    assert isinstance(forwarded["epochs"], int)


@pytest.mark.asyncio
async def test_handler_accepts_a_yaml_file_path(
    recorded_runner: Any, tmp_path: Path,
) -> None:
    """The standalone contract (a file path) still resolves."""
    path = tmp_path / "overrides.yaml"
    path.write_text("dpo_learning_rate: 5.0e-7\n", encoding="utf-8")
    await _dispatch(
        base_model="nemotron3-nano-30b", config_overrides=str(path),
    )
    assert recorded_runner.last_kwargs["config_overrides"] == {
        "dpo_learning_rate": 5e-7
    }


@pytest.mark.asyncio
async def test_handler_fails_closed_on_an_unknown_key(
    recorded_runner: Any,
) -> None:
    """Never dropped, never a new attribute — the phase fails."""
    envelope = await _dispatch(
        base_model="nemotron3-nano-30b",
        config_overrides={"dpo_lernin_rate": 5e-7},
    )
    assert envelope["success"] is False
    assert envelope["error_type"] == "invalid_config_overrides"
    assert "dpo_lernin_rate" in envelope["error"]
    # The supported set is named so the operator can fix it in one pass.
    assert "dpo_learning_rate" in envelope["error"]
    assert not recorded_runner.last_kwargs, "no runner may be constructed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec",
    [
        "dpo_learning_rate=-1",       # out of range
        "dpo_learning_rate=0",        # out of range (a zero LR trains nothing)
        "dpo_learning_rate=fast",     # not a number
        "epochs=1.5",                 # fractional integer
        "lora_dropout=2",             # outside [0, 1]
        "use_4bit=maybe",             # not a boolean
        "base_model=qwen2.5-1.5b",    # locked: owned by --base-model
    ],
)
async def test_handler_fails_closed_on_a_garbage_value(
    recorded_runner: Any, spec: str,
) -> None:
    envelope = await _dispatch(
        base_model="nemotron3-nano-30b", config_overrides=spec,
    )
    assert envelope["success"] is False
    assert envelope["error_type"] == "invalid_config_overrides"
    assert not recorded_runner.last_kwargs, "no runner may be constructed"


@pytest.mark.asyncio
async def test_handler_omits_the_kwarg_entirely_without_the_flag(
    recorded_runner: Any,
) -> None:
    """Byte-identical default: nothing overrides the per-base YAML."""
    await _dispatch(base_model="nemotron3-nano-30b")
    assert recorded_runner.last_kwargs["config_overrides"] is None


# --------------------------------------------------------------------- #
# CLI                                                                    #
# --------------------------------------------------------------------- #


_COMMON = dict(
    corpus="in.pdf",
    course_name="DEMO",
    weeks=None,
    no_assessments=False,
    assessment_count=50,
    bloom_levels="remember,understand,apply,analyze",
    priority="normal",
    objectives_path=None,
)


def test_cli_flag_populates_the_workflow_param() -> None:
    from cli.commands.run import _build_workflow_params

    absent = _build_workflow_params("textbook_to_course", **_COMMON)
    assert PARAM not in absent, (
        "an unflagged run must not record the key at all, so the per-base "
        "training YAML remains the sole source"
    )

    explicit = _build_workflow_params(
        "textbook_to_course",
        config_overrides={"dpo_learning_rate": 5e-7},
        **_COMMON,
    )
    assert explicit[PARAM] == {"dpo_learning_rate": 5e-7}
    # Must stay JSON-serializable — it is persisted into workflow state and
    # re-read on resume.
    json.dumps(explicit[PARAM])


def test_cli_validates_the_spec_at_parse_time() -> None:
    from cli.commands.run import _validate_config_overrides

    assert _validate_config_overrides(None) is None
    assert _validate_config_overrides("dpo_learning_rate=5e-7") == {
        "dpo_learning_rate": 5e-7
    }


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ("dpo_lernin_rate=5e-7", "Unknown TrainingConfig override key"),
        ("dpo_learning_rate=-3", "must be greater than 0"),
        ("dpo_learning_rate=fast", "expected a number"),
        ("base_model=qwen2.5-1.5b", "--base-model"),
        ("not-a-file.yaml", "neither an existing file"),
    ],
)
def test_cli_exits_two_before_writing_any_state(spec: str, needle: str) -> None:
    """Parse-time failure: seconds, not six hours into a training run."""
    from click.testing import CliRunner

    from cli.commands.run import run_command

    result = CliRunner().invoke(
        run_command,
        [
            "textbook-to-course", "--corpus", "in.pdf",
            "--course-name", "DEMO",
            "--config-overrides", spec,
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "--config-overrides" in result.output
    assert needle in result.output


def test_resume_repins_the_persisted_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume-time flag must patch persisted params, like --base-model."""
    import lib.paths as paths
    from cli.commands import run as run_mod

    workflows = tmp_path / "workflows"
    workflows.mkdir()
    wf_id = "WF-TEST-CONFIG-OVERRIDES"
    (workflows / f"{wf_id}.json").write_text(
        json.dumps({"workflow_id": wf_id, "params": {"course_name": "DEMO"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "STATE_PATH", tmp_path)

    def _read() -> Dict[str, Any]:
        return json.loads(
            (workflows / f"{wf_id}.json").read_text(encoding="utf-8")
        )["params"]

    # Absent flag leaves the persisted value alone (stickiness contract).
    assert run_mod._apply_resume_config_overrides_override(wf_id, None) is None
    assert PARAM not in _read()

    applied = run_mod._apply_resume_config_overrides_override(
        wf_id, {"dpo_learning_rate": 5e-7},
    )
    assert applied == {"dpo_learning_rate": 5e-7}
    assert _read()[PARAM] == {"dpo_learning_rate": 5e-7}
    # ...and the pre-existing params survive the patch.
    assert _read()["course_name"] == "DEMO"
