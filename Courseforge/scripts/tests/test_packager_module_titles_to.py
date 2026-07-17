"""
Tests for ED4ALL_IMSCC_MODULE_TITLES=to — terminal-objective module titles.

Opt-in flag on the IMSCC packager: when the env var resolves to the exact
token ``to``, top-level organization groups are titled
``"Module {N}: {topic}"`` from ``terminal_objectives[N-1]`` in the export's
``01_learning_objectives/synthesized_objectives.json`` (topic =
``anchor_module_title``, falling back to a truncated ``statement``).

Asserted here:
    1. Env unset → byte-identical legacy "Week N" / "Week N: <h1 title>"
       titles.
    2. Flag on + objectives fixture → "Module 2: <anchor title>" (and the
       statement-truncation fallback when ``anchor_module_title`` is absent).
    3. Flag on + missing / unparseable objectives file → silent legacy
       fallback (packaging never fails over a title).
    4. Any non-"to" token → legacy behavior (parse-with-fallback).
"""

import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from package_multifile_imscc import package_imscc  # noqa: E402

_CC_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"

_OBJECTIVES = {
    "course_title": "Mini Course",
    "terminal_objectives": [
        {
            "id": "TO-01",
            "statement": "Analyze foundational structures.",
            "anchor_module_title": "Foundations of Algebra",
        },
        {
            "id": "TO-02",
            "statement": "Evaluate advanced applications.",
            "anchor_module_title": "Linear Equations and Graphs",
        },
    ],
}


def _write_week(content_dir: Path, week_num: int, h1_title: str) -> None:
    week_dir = content_dir / f"week_{week_num:02d}"
    week_dir.mkdir(parents=True, exist_ok=True)
    (week_dir / f"week_{week_num:02d}_overview.html").write_text(
        "<!DOCTYPE html><html><head></head><body>"
        f"<h1>Week {week_num} Overview: {h1_title}</h1>"
        "<p>content</p></body></html>",
        encoding="utf-8",
    )


@pytest.fixture
def export_root(tmp_path):
    """Export-shaped tree: 03_content_development/week_* + objectives dir."""
    content_dir = tmp_path / "03_content_development"
    _write_week(content_dir, 1, "Getting Started")
    _write_week(content_dir, 2, "Going Deeper")
    return tmp_path


def _write_objectives(export_root: Path, doc) -> Path:
    obj_dir = export_root / "01_learning_objectives"
    obj_dir.mkdir(parents=True, exist_ok=True)
    path = obj_dir / "synthesized_objectives.json"
    path.write_text(
        doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8"
    )
    return path


def _package_and_read_week_titles(export_root: Path, tmp_path: Path):
    """Run the packager and return the WEEK_* item titles from the manifest."""
    content_dir = export_root / "03_content_development"
    output = tmp_path / "out.imscc"
    package_imscc(
        content_dir, output, "TEST_101", "Test Course", skip_validation=True
    )
    with zipfile.ZipFile(output) as zf:
        manifest = zf.read("imsmanifest.xml").decode("utf-8")
    root = ET.fromstring(manifest)
    titles = {}
    for item in root.iter(f"{{{_CC_NS}}}item"):
        ident = item.get("identifier", "")
        if ident.startswith("WEEK_"):
            titles[ident] = item.find(f"{{{_CC_NS}}}title").text
    return titles


class TestModuleTitlesLegacyDefault:
    def test_env_unset_yields_legacy_week_titles(
        self, export_root, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("ED4ALL_IMSCC_MODULE_TITLES", raising=False)
        _write_objectives(export_root, _OBJECTIVES)  # present but ignored
        titles = _package_and_read_week_titles(export_root, tmp_path)
        assert titles == {
            "WEEK_1": "Week 1: Getting Started",
            "WEEK_2": "Week 2: Going Deeper",
        }

    def test_non_to_token_yields_legacy_week_titles(
        self, export_root, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ED4ALL_IMSCC_MODULE_TITLES", "modules")
        _write_objectives(export_root, _OBJECTIVES)
        titles = _package_and_read_week_titles(export_root, tmp_path)
        assert titles["WEEK_1"] == "Week 1: Getting Started"
        assert titles["WEEK_2"] == "Week 2: Going Deeper"


class TestModuleTitlesToMode:
    def test_flag_on_titles_from_anchor_module_title(
        self, export_root, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ED4ALL_IMSCC_MODULE_TITLES", "to")
        _write_objectives(export_root, _OBJECTIVES)
        titles = _package_and_read_week_titles(export_root, tmp_path)
        assert titles["WEEK_1"] == "Module 1: Foundations of Algebra"
        assert titles["WEEK_2"] == "Module 2: Linear Equations and Graphs"
        # No calendar concept survives in the group titles.
        assert not any(t.startswith("Week") for t in titles.values())

    def test_flag_on_statement_fallback_when_no_anchor_title(
        self, export_root, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ED4ALL_IMSCC_MODULE_TITLES", "to")
        doc = {
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Analyze foundational structures."},
                {
                    "id": "TO-02",
                    "statement": (
                        "Evaluate advanced applications of linear systems "
                        "across graphing, substitution, and elimination "
                        "strategies in real-world modeling contexts."
                    ),
                },
            ]
        }
        _write_objectives(export_root, doc)
        titles = _package_and_read_week_titles(export_root, tmp_path)
        assert titles["WEEK_1"] == "Module 1: Analyze foundational structures."
        # Long statement truncates at a word boundary with an ellipsis.
        assert titles["WEEK_2"].startswith("Module 2: Evaluate advanced")
        assert titles["WEEK_2"].endswith("…")
        assert len(titles["WEEK_2"]) < len("Module 2: ") + 82

    def test_flag_on_ordinal_beyond_terminals_falls_back_per_group(
        self, export_root, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ED4ALL_IMSCC_MODULE_TITLES", "to")
        _write_objectives(
            export_root,
            {"terminal_objectives": [_OBJECTIVES["terminal_objectives"][0]]},
        )
        titles = _package_and_read_week_titles(export_root, tmp_path)
        assert titles["WEEK_1"] == "Module 1: Foundations of Algebra"
        assert titles["WEEK_2"] == "Week 2: Going Deeper"  # legacy fallback


class TestModuleTitlesFallbacks:
    def test_flag_on_missing_objectives_file_falls_back_legacy(
        self, export_root, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ED4ALL_IMSCC_MODULE_TITLES", "to")
        # No 01_learning_objectives/ dir written at all.
        titles = _package_and_read_week_titles(export_root, tmp_path)
        assert titles == {
            "WEEK_1": "Week 1: Getting Started",
            "WEEK_2": "Week 2: Going Deeper",
        }

    def test_flag_on_unparseable_objectives_falls_back_legacy(
        self, export_root, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ED4ALL_IMSCC_MODULE_TITLES", "to")
        _write_objectives(export_root, "{not json")
        titles = _package_and_read_week_titles(export_root, tmp_path)
        assert titles["WEEK_1"] == "Week 1: Getting Started"
        assert titles["WEEK_2"] == "Week 2: Going Deeper"
