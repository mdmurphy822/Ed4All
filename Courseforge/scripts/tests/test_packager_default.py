"""
Tests for Worker L (REC-CTR-03) — packager default-on + workflow gate.

Validates that:
    1. Without ``--objectives`` and WITH ``course.json`` at content-dir root,
       validation auto-discovers and runs.
    2. ``--skip-validation`` (skip_validation=True) still bypasses.
    3. Validation failure raises ``SystemExit(2)`` even under auto-discovery.
    4. Without any objectives source (no arg, no course.json), a warning is
       printed and packaging proceeds — backward-compat.
    5. The new ``PageObjectivesValidator`` class returns a ``GateResult`` of
       the expected shape for clean, violating, and no-objectives inputs.
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from package_multifile_imscc import package_imscc  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture helpers (shared with test_packager_validation_gate.py style)
# ---------------------------------------------------------------------------

_OBJECTIVES = {
    "course_title": "Mini Course",
    "description": "Fixture",
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Terminal 1", "bloomLevel": "evaluate"},
    ],
    "chapter_objectives": [
        {
            "chapter": "Week 1-2: Foundations",
            "objectives": [
                {"id": "CO-01", "statement": "Foundation 1", "bloomLevel": "understand"},
                {"id": "CO-02", "statement": "Foundation 2", "bloomLevel": "apply"},
            ],
        },
        {
            "chapter": "Week 3-4: Advanced",
            "objectives": [
                {"id": "CO-03", "statement": "Advanced 1", "bloomLevel": "analyze"},
                {"id": "CO-04", "statement": "Advanced 2", "bloomLevel": "evaluate"},
            ],
        },
    ],
}


def _page_html(lo_ids):
    """Emit a minimal HTML page carrying one JSON-LD learningObjectives block."""
    los = [{"id": x, "statement": f"stub for {x}"} for x in lo_ids]
    ld = json.dumps({"@context": "x", "@type": "LearningResource", "learningObjectives": los})
    return (
        '<!DOCTYPE html><html><head>'
        '<script type="application/ld+json">' + ld + '</script>'
        '</head><body><p>content</p></body></html>'
    )


@pytest.fixture
def content_dir_with_courseJson(tmp_path):
    """Content dir containing week_* subdirs + course.json at root.

    The validator's auto-discovery hits the course.json at content-dir
    root. The fixture writes the canonical objectives there so tests can
    exercise the default-on path without passing ``objectives_path``.
    """
    (tmp_path / "week_01").mkdir()
    (tmp_path / "week_03").mkdir()
    (tmp_path / "course.json").write_text(
        json.dumps(_OBJECTIVES), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def content_dir_no_courseJson(tmp_path):
    """Content dir with week_* subdirs but NO course.json (auto-discovery miss)."""
    (tmp_path / "week_01").mkdir()
    (tmp_path / "week_03").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Packager default behavior
# ---------------------------------------------------------------------------

class TestPackagerDefaultOn:
    """Exercises the default-on behavior flipped in Worker L."""

    def test_validation_runs_by_default_with_auto_discovery(
        self, content_dir_with_courseJson, tmp_path, capsys,
    ):
        """Valid content + auto-discovered course.json ⇒ validation runs, package produced."""
        (content_dir_with_courseJson / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01", "CO-02"]), encoding="utf-8",
        )
        (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-03", "CO-04"]), encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_courseJson, output, "TEST_101", "Test Course",
        )
        captured = capsys.readouterr().out
        assert "Auto-discovered objectives" in captured
        assert "All week pages pass per-week LO contract" in captured
        assert output.exists(), "package must be produced when validation passes"

    def test_skip_validation_bypasses(
        self, content_dir_with_courseJson, tmp_path, capsys,
    ):
        """skip_validation=True bypasses validation even with auto-discoverable course.json."""
        # Intentionally VIOLATING page — would fail if validation ran.
        (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-01"]),  # CO-01 belongs to weeks 1-2
            encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_courseJson, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        captured = capsys.readouterr().out
        assert "SKIPPED (per --skip-validation)" in captured
        # Auto-discovery must NOT fire when skip_validation is set.
        assert "Auto-discovered objectives" not in captured
        assert output.exists(), "--skip-validation must allow packaging to proceed"

    def test_validation_fails_on_broken_lo(
        self, content_dir_with_courseJson, tmp_path, capsys,
    ):
        """Violating LO + auto-discovered course.json ⇒ SystemExit(2)."""
        (content_dir_with_courseJson / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01"]), encoding="utf-8",
        )
        # Fabricated week-local ID — exact shape of the pre-fix defect.
        (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
            _page_html(["W03-CO-01"]), encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        with pytest.raises(SystemExit) as excinfo:
            package_imscc(
                content_dir_with_courseJson, output, "TEST_101", "Test Course",
            )
        assert excinfo.value.code == 2
        assert not output.exists(), "packager must not create the zip on validation failure"
        captured = capsys.readouterr().out
        assert "Auto-discovered objectives" in captured
        assert "REFUSING TO PACKAGE" in captured

    def test_no_objectives_no_autodiscovery_warns(
        self, content_dir_no_courseJson, tmp_path, capsys,
    ):
        """No course.json + no objectives arg ⇒ warning, packaging proceeds."""
        # Intentionally VIOLATING page; with no objectives source, validation
        # cannot run at all, and packaging MUST still succeed (backward-compat
        # for callers that never wired the flag).
        (content_dir_no_courseJson / "week_03" / "week_03_overview.html").write_text(
            _page_html(["W03-CO-01"]), encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_no_courseJson, output, "TEST_101", "Test Course",
        )
        captured = capsys.readouterr().out
        assert "WARNING: no objectives file found" in captured
        assert "REFUSING TO PACKAGE" not in captured
        assert output.exists(), (
            "missing objectives alone must never hard-fail; "
            "packaging must proceed with a warning"
        )


# ---------------------------------------------------------------------------
# PageObjectivesValidator wrapper
# ---------------------------------------------------------------------------

class TestPageObjectivesValidator:
    """Direct tests of the orchestrator-gate wrapper."""

    def test_page_objectives_validator_returns_validation_result(
        self, content_dir_with_courseJson,
    ):
        """Clean content ⇒ GateResult(passed=True, no critical issues)."""
        from lib.validators.page_objectives import PageObjectivesValidator

        (content_dir_with_courseJson / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01", "CO-02"]), encoding="utf-8",
        )
        result = PageObjectivesValidator().validate({
            "content_dir": content_dir_with_courseJson,
        })
        assert result.passed is True
        assert result.gate_id == "page_objectives"
        assert result.validator_name == "page_objectives"
        assert result.critical_count == 0

    def test_page_objectives_validator_returns_critical_on_violation(
        self, content_dir_with_courseJson,
    ):
        """Violating content ⇒ GateResult(passed=False, critical issue emitted)."""
        from lib.validators.page_objectives import PageObjectivesValidator

        (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
            _page_html(["W03-CO-01"]),  # fabricated ID
            encoding="utf-8",
        )
        result = PageObjectivesValidator().validate({
            "content_dir": content_dir_with_courseJson,
        })
        assert result.passed is False
        assert result.critical_count >= 1
        codes = {issue.code for issue in result.issues}
        assert "LO_SPECIFICITY_VIOLATION" in codes

    def test_page_objectives_validator_no_objectives_warns(
        self, content_dir_no_courseJson,
    ):
        """No objectives available ⇒ passed=True with a NO_OBJECTIVES_FILE warning."""
        from lib.validators.page_objectives import PageObjectivesValidator

        (content_dir_no_courseJson / "week_01" / "week_01_overview.html").write_text(
            "<html><body>no JSON-LD here</body></html>", encoding="utf-8",
        )
        result = PageObjectivesValidator().validate({
            "content_dir": content_dir_no_courseJson,
        })
        assert result.passed is True
        codes = {issue.code for issue in result.issues}
        assert "NO_OBJECTIVES_FILE" in codes
        # Warning, not critical - orchestrator must not block on this.
        severities = {issue.severity for issue in result.issues}
        assert "critical" not in severities


# ---------------------------------------------------------------------------
# Wave 3 / Worker M — course_metadata.json stub inclusion in IMSCC zip
# ---------------------------------------------------------------------------
#
# Closes the Wave 2 integration gap: Worker J's course_metadata.json
# classification stub was emitted alongside the IMSCC but never bundled
# inside it. Trainforge's consume already handled both zip-root and
# sibling paths, so the gap was latent — but zip-root is the canonical
# self-contained delivery. Behavior: additive, no env var, no-op when
# the stub file is absent.
# ---------------------------------------------------------------------------


class TestPackagerStubInclusion:
    """Wave 3 / Worker M: course_metadata.json bundled at zip root."""

    def test_packager_includes_course_metadata_when_present(
        self, content_dir_with_courseJson, tmp_path,
    ):
        """Stub file at content-dir root → bundled at zip root."""
        # Valid pages so per-week LO validation passes.
        (content_dir_with_courseJson / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01", "CO-02"]), encoding="utf-8",
        )
        (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-03", "CO-04"]), encoding="utf-8",
        )
        # Stub body is arbitrary JSON; the packager does not parse it,
        # only bundles it. Shape mirrors Worker J's emit contract.
        stub_payload = {
            "courseCode": "TEST_101",
            "classification": {
                "division": "STEM",
                "primary_domain": "computer-science",
                "subdomains": ["web-development"],
                "topics": ["rest-apis"],
            },
        }
        (content_dir_with_courseJson / "course_metadata.json").write_text(
            json.dumps(stub_payload), encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_courseJson, output, "TEST_101", "Test Course",
        )
        assert output.exists(), "package must be produced"

        with zipfile.ZipFile(output) as zf:
            names = zf.namelist()
        assert "course_metadata.json" in names, (
            f"expected course_metadata.json at zip root; got {names}"
        )
        # Sanity: manifest + html files still present.
        assert "imsmanifest.xml" in names
        assert any(n.endswith(".html") for n in names)

    def test_packager_skips_stub_when_absent(
        self, content_dir_with_courseJson, tmp_path,
    ):
        """No stub file → zip contains manifest + html only; no course_metadata.json."""
        (content_dir_with_courseJson / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01", "CO-02"]), encoding="utf-8",
        )
        (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-03", "CO-04"]), encoding="utf-8",
        )
        # Explicitly NO course_metadata.json (backward-compat path).
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_courseJson, output, "TEST_101", "Test Course",
        )
        assert output.exists(), "package must be produced without stub"

        with zipfile.ZipFile(output) as zf:
            names = zf.namelist()
        assert "imsmanifest.xml" in names
        assert "course_metadata.json" not in names, (
            f"stub absent at source must NOT appear in zip; got {names}"
        )

    def test_packager_stub_inclusion_logs_in_summary(
        self, content_dir_with_courseJson, tmp_path, capsys,
    ):
        """Summary print line reflects stub inclusion when it was bundled."""
        (content_dir_with_courseJson / "week_01" / "week_01_overview.html").write_text(
            _page_html(["TO-01", "CO-01", "CO-02"]), encoding="utf-8",
        )
        (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
            _page_html(["TO-01", "CO-03", "CO-04"]), encoding="utf-8",
        )
        (content_dir_with_courseJson / "course_metadata.json").write_text(
            json.dumps({"courseCode": "TEST_101"}), encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_courseJson, output, "TEST_101", "Test Course",
        )
        captured = capsys.readouterr().out
        assert "course_metadata.json" in captured, (
            f"summary must mention the stub when bundled; got:\n{captured}"
        )
