"""Protect the public Courseforge installation and routing documentation."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
README = PROJECT_ROOT / "Courseforge" / "README.md"
ARCHITECTURE = PROJECT_ROOT / "Courseforge" / "architecture.md"
WORKFLOWS = PROJECT_ROOT / "config" / "workflows.yaml"


def _text(path: Path) -> str:
    """Read one public document as normalized UTF-8 text."""
    return path.read_text(encoding="utf-8")


def _prose(path: Path) -> str:
    """Collapse Markdown whitespace for prose-level contract checks."""
    return " ".join(_text(path).split())


def _textbook_phases() -> dict[str, dict]:
    """Index the canonical textbook workflow phases by name."""
    config = yaml.safe_load(WORKFLOWS.read_text(encoding="utf-8"))
    phases = config["workflows"]["textbook_to_course"]["phases"]
    return {phase["name"]: phase for phase in phases}


def test_readme_installs_conversion_dependencies_for_end_to_end_use() -> None:
    """Keep the quick start consistent with its conversion promise."""
    text = _text(README)
    assert 'pip install -e ".[full,semantik]"' in text
    assert "document-conversion dependencies" in text


def test_architecture_matches_mutually_exclusive_authoring_routes() -> None:
    """Pin the public routing description to the workflow predicates."""
    phases = _textbook_phases()
    assert phases["content_generation"]["enabled_when_env"] == (
        "COURSEFORGE_TWO_PASS!=true"
    )
    for name in (
        "content_generation_outline",
        "inter_tier_validation",
        "content_generation_rewrite",
        "post_rewrite_validation",
    ):
        assert phases[name]["enabled_when_env"] == "COURSEFORGE_TWO_PASS=true"

    prose = _prose(ARCHITECTURE)
    assert "mutually exclusive" in prose
    assert "do not combine" in prose
    assert "both authoring contracts" in prose


def test_stage_examples_include_private_safe_identifying_arguments() -> None:
    """Require complete stage-command templates without concrete identities."""
    text = _text(ARCHITECTURE)
    for command in (
        "courseforge-outline",
        "courseforge-validate",
        "courseforge-rewrite",
        "courseforge",
    ):
        expected = (
            f"ed4all run {command} --corpus <private-source-path> "
            "--course-name <private-course-name>"
        )
        assert expected in text


def test_public_docs_keep_generated_course_material_private() -> None:
    """Keep source identity and generated artifacts outside the release."""
    combined = f"{_text(README)}\n{_text(ARCHITECTURE)}".lower()
    assert "gitignored `courseforge/exports/`" in combined
    assert "private source material" in combined
    assert "operator course names" in combined
