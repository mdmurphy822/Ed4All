"""Regression — ``instruction_variants_per_chunk`` reaches ``run_synthesis``.

Reject-mined DPO negatives (``Trainforge/synthesis_reject_mining.py``) have
STRUCTURALLY ZERO yield unless a chunk holds at least two instruction units:
one unit can never hold both an accepted anchor and a rejected unit to pair it
against. The knob that controls that count was settable ONLY through the
standalone ``Trainforge/synthesize_training.py --instruction-variants-per-chunk``
argparse flag — ``_synthesize_training`` forwarded a fixed kwarg list that
omitted it, and neither ``config/workflows.yaml`` nor ``cli/`` mentioned it — so
the documented shadow-measurement path was unreachable through ``ed4all run``.

These tests pin the whole route: CLI flag -> workflow_params -> the phase's
``inputs_from`` block -> the tool schema -> ``run_synthesis``. The default stays
1, so an unflagged run is byte-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402

PARAM = "instruction_variants_per_chunk"


def _training_synthesis_phase() -> Dict[str, Any]:
    data = yaml.safe_load(
        (PROJECT_ROOT / "config" / "workflows.yaml").read_text(encoding="utf-8")
    )
    for phase in data["workflows"]["textbook_to_course"]["phases"]:
        if phase.get("name") == "training_synthesis":
            return phase
    raise AssertionError("training_synthesis phase missing from workflows.yaml")


def test_workflows_yaml_routes_the_param_from_workflow_params() -> None:
    routes = _training_synthesis_phase().get("inputs_from") or []
    matches = [r for r in routes if r.get("param") == PARAM]
    assert matches, (
        f"training_synthesis must declare an inputs_from route for {PARAM}; "
        "without it the CLI flag can never reach the phase handler"
    )
    assert matches[0]["source"] == "workflow_params"
    assert matches[0]["key"] == PARAM


def test_phase_param_routing_resolves_the_param() -> None:
    """Through the production routing resolver, not the raw YAML."""
    from MCP.core.workflow_runner import _get_phase_param_routing

    routing = _get_phase_param_routing("training_synthesis")
    assert routing.get(PARAM) == ("workflow_params", PARAM)


def test_tool_schema_lists_the_param_as_optional_with_default_one() -> None:
    from MCP.core.tool_schemas import get_defaults, get_optional_params

    assert PARAM in get_optional_params("synthesize_training")
    assert get_defaults("synthesize_training")[PARAM] == 1


class _RecordingRunSynthesis:
    """Stand-in for ``run_synthesis`` that records the kwarg and stops."""

    def __init__(self) -> None:
        self.kwargs: Dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        raise RuntimeError("stop-after-capture")


@pytest.fixture()
def recorded_run_synthesis(monkeypatch: pytest.MonkeyPatch):
    import Trainforge.synthesize_training as st

    recorder = _RecordingRunSynthesis()
    monkeypatch.setattr(st, "run_synthesis", recorder)
    return recorder


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "course"
    (corpus / "imscc_chunks").mkdir(parents=True)
    (corpus / "imscc_chunks" / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    return corpus


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("passed", "expected"),
    [
        (None, 1),      # absent -> byte-identical legacy default
        (3, 3),         # explicit operator value forwarded verbatim
        ("4", 4),       # routed params arrive as strings; parse-with-fallback
        ("nonsense", 1),
        (0, 1),         # non-positive clamps up rather than disabling synthesis
    ],
)
async def test_registry_handler_forwards_the_param(
    tmp_path: Path,
    recorded_run_synthesis: _RecordingRunSynthesis,
    passed: Any,
    expected: int,
) -> None:
    handler = _build_tool_registry()["synthesize_training"]
    kwargs: Dict[str, Any] = {
        "corpus_dir": str(_corpus(tmp_path)),
        "course_code": "DEMO",
        "provider": "mock",
    }
    if passed is not None:
        kwargs[PARAM] = passed

    await handler(**kwargs)

    assert recorded_run_synthesis.kwargs, (
        "run_synthesis was never reached; the handler bailed before dispatch"
    )
    assert recorded_run_synthesis.kwargs[PARAM] == expected


def test_cli_flag_populates_the_workflow_param() -> None:
    from cli.commands.run import _build_workflow_params

    common = dict(
        corpus="in.pdf",
        course_name="DEMO",
        weeks=None,
        no_assessments=False,
        assessment_count=50,
        bloom_levels="remember,understand,apply,analyze",
        priority="normal",
        objectives_path=None,
    )

    absent = _build_workflow_params("textbook_to_course", **common)
    assert PARAM not in absent, (
        "an unflagged run must not record the key at all, so the handler's own "
        "default of 1 applies"
    )

    explicit = _build_workflow_params(
        "textbook_to_course", instruction_variants_per_chunk=2, **common
    )
    assert explicit[PARAM] == 2


def test_cli_rejects_a_variant_count_below_one() -> None:
    from click.testing import CliRunner

    from cli.commands.run import run_command

    result = CliRunner().invoke(
        run_command,
        [
            "textbook-to-course", "--corpus", "in.pdf",
            "--course-name", "DEMO",
            "--instruction-variants-per-chunk", "0",
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "instruction-variants-per-chunk" in result.output
