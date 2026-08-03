"""Protect the public pipeline runbooks from operational and privacy drift."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from click.testing import CliRunner

from cli.main import cli

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNBOOKS = (
    PROJECT_ROOT / "docs" / "operations" / "pipeline-invocation.md",
    PROJECT_ROOT / "docs" / "operations" / "full-run-playbook.md",
)
FORBIDDEN_PUBLIC_TERMS = (
    "modernbert",
    "council",
    "rtx 3070",
    "runpod",
    "unverified",
)


def _normalized(path: Path) -> str:
    """Return case-folded prose with whitespace normalized."""
    return " ".join(path.read_text(encoding="utf-8").casefold().split())


def _local_markdown_targets(path: Path) -> list[Path]:
    """Resolve local Markdown link targets while excluding URLs and anchors."""
    text = path.read_text(encoding="utf-8")
    raw_targets = re.findall(r"(?<!!)\[[^]]*]\(([^)]+)\)", text)
    targets: list[Path] = []
    for raw_target in raw_targets:
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        target = target.split("#", 1)[0]
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        targets.append((path.parent / target).resolve())
    return targets


def test_public_runbooks_preserve_current_pipeline_contract() -> None:
    for path in RUNBOOKS:
        text = _normalized(path)

        assert "private" in text
        assert any(
            marker in text
            for marker in (
                "not tracked",
                "never tracked",
                "git-ignored",
                "do not commit",
            )
        )
        assert "glm-ocr" in text
        assert "enrich" in text
        assert "super" in text
        assert "heading judge" in text or "heading-judge" in text

        assert "ed4all stop" in text
        assert "--resume" in text
        assert "gate" in text and "fail" in text
        assert "threshold" in text and "severity" in text
        assert "cause" in text or "fix the source" in text

        for forbidden in FORBIDDEN_PUBLIC_TERMS:
            assert forbidden not in text
        assert "private opt-in" not in text
        assert "private opt in" not in text

        assert "installation.md" in text
        assert "licensing.md" in text
        assert "gates.md" in text

    assert "pipeline-invocation.md" in _normalized(RUNBOOKS[1])


def test_public_runbook_local_links_resolve() -> None:
    for runbook in RUNBOOKS:
        missing = [
            target
            for target in _local_markdown_targets(runbook)
            if not target.is_file()
        ]
        assert not missing, f"broken local links in {runbook}: {missing}"


def test_documented_entry_points_exist_in_live_cli_and_workflow_config() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "config" / "workflows.yaml").read_text(encoding="utf-8")
    )
    assert "textbook_to_course" in config["workflows"]

    runner = CliRunner()
    root_help = runner.invoke(cli, ["--help"])
    run_help = runner.invoke(cli, ["run", "--help"])
    stop_help = runner.invoke(cli, ["stop", "--help"])

    assert root_help.exit_code == 0
    assert "run" in root_help.output and "stop" in root_help.output
    assert run_help.exit_code == 0
    for option in ("--corpus", "--course-name", "--resume", "--stop-after"):
        assert option in run_help.output
    assert stop_help.exit_code == 0
