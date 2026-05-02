"""Phase 2 Subtask 28: --emit-mode {full|outline} CLI flag verification.

Drives :func:`generate_course.generate_course` against an inline fixture
in two modes back-to-back and asserts:

  * full mode (default) emits content / activity / self-check / discussion
    HTML bodies — the snapshot suite already covers byte-stability for
    full mode; this test is a sanity check on the same fixture.
  * outline mode suppresses content/activity/self-check/discussion HTML
    bodies (no `data-cf-content-type`, no flip-cards, no callouts, no
    self-check radio inputs, no activity-card divs).
  * outline mode keeps the outline-tier shell (chrome footer/header,
    objectives section + objective `<li>`s, summary key takeaways,
    overview readings list).
  * outline mode stamps `course_metadata.json::blocks_summary.outline_only=true`
    even without --division/--primary-domain (so the packager + Trainforge
    consumer can detect the outline shape).
  * full mode does NOT stamp `blocks_summary` (preserves legacy stub
    shape — backward compat).
  * `emit_mode` is plumbed through the function signature so external
    callers (the MCP runtime) can drive outline mode without going via
    the CLI.

These assertions run end-to-end via the real `generate_course()` entry
point — no per-renderer mocking — so they catch regressions in the
plumbing as well as the renderer-level filters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


_COURSE_DATA = {
    "course_code": "OUTLINE_101",
    "course_title": "Outline Mode Test Course",
    "weeks": [
        {
            "week_number": 1,
            "title": "Foundations",
            "estimated_hours": "3-4",
            "objectives": [
                {
                    "id": "TO-01",
                    "statement": "Define the foundational concepts.",
                    "bloom_level": "remember",
                    "bloom_verb": "define",
                },
                {
                    "id": "CO-01",
                    "statement": "Explain how the foundational concepts relate.",
                    "bloom_level": "understand",
                    "bloom_verb": "explain",
                },
            ],
            "overview_text": [
                "This week introduces the foundational concepts you will build "
                "on in subsequent weeks.",
            ],
            "readings": [
                "Chapter 1: Foundations (pp. 1-25)",
            ],
            "content_modules": [
                {
                    "title": "Core Concepts",
                    "sections": [
                        {
                            "heading": "What is a Concept?",
                            "content_type": "definition",
                            "paragraphs": [
                                "A concept is a mental abstraction representing "
                                "a class of objects, ideas, or events that share "
                                "common features."
                            ],
                            "key_terms": ["Concept"],
                            "flip_cards": [
                                {
                                    "term": "Schema",
                                    "definition": "A cognitive framework.",
                                },
                            ],
                            "callouts": [
                                {
                                    "type": "info",
                                    "title": "Note",
                                    "body": "Concepts vs. percepts.",
                                },
                            ],
                        },
                    ],
                    "misconceptions": [
                        {
                            "misconception": "All concepts are concrete.",
                            "correction": "Concepts can be abstract or concrete.",
                        },
                    ],
                },
            ],
            "activities": [
                {
                    "title": "Concept Mapping",
                    "description": "Create a concept map relating five core terms.",
                    "objective_ref": "CO-01",
                    "bloom_level": "apply",
                    "type": "application",
                },
            ],
            "self_check_questions": [
                {
                    "question": "Which best describes a concept?",
                    "options": [
                        {"text": "A specific object", "correct": False, "feedback": "Too narrow."},
                        {"text": "A mental abstraction", "correct": True, "feedback": "Yes."},
                        {"text": "A physical event", "correct": False, "feedback": "No."},
                        {"text": "A unique percept", "correct": False, "feedback": "No."},
                    ],
                    "objective_ref": "TO-01",
                    "bloom_level": "remember",
                },
            ],
            "key_takeaways": [
                "Concepts are mental abstractions.",
                "Schemas organize concepts into frameworks.",
            ],
            "discussion": {
                "prompt": "Discuss a concept that surprised you this week.",
            },
        }
    ],
}


@pytest.fixture
def course_data_path(tmp_path: Path) -> Path:
    p = tmp_path / "course_data.json"
    p.write_text(json.dumps(_COURSE_DATA), encoding="utf-8")
    return p


def _read_week_files(out: Path) -> dict[str, str]:
    week_dir = out / "week_01"
    return {
        f.name: f.read_text(encoding="utf-8") for f in sorted(week_dir.glob("*.html"))
    }


_MAIN_RE = __import__("re").compile(
    r"<main[^>]*>(.*?)</main>", __import__("re").DOTALL
)


def _main_body(html: str) -> str:
    """Return the inner HTML of the <main> tag.

    All assertions about HTML body content go through this so tests
    don't accidentally match a class name that lives in the inline
    <style> block in <head> (which carries `.flip-card`, `.callout`,
    `.activity-card`, `.discussion-prompt` selectors regardless of
    page content). Falls back to the full HTML if no <main> match
    (which would itself be a regression worth flagging).
    """
    m = _MAIN_RE.search(html)
    if m is None:
        raise AssertionError(f"<main> not found in HTML:\n{html[:500]}")
    return m.group(1)


class TestEmitModeFullDefault:
    def test_full_mode_renders_content_bodies(self, course_data_path, tmp_path):
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "full_out"
        generate_course(str(course_data_path), str(out), emit_mode="full")
        files = _read_week_files(out)
        # Content page exists and carries content-type / flip-card / callout.
        content_keys = [k for k in files if "content" in k]
        assert len(content_keys) == 1
        content_body = _main_body(files[content_keys[0]])
        assert "data-cf-content-type=" in content_body
        assert "flip-card" in content_body
        # Self-check page renders radio inputs.
        sc_body = _main_body(files["week_01_self_check.html"])
        assert 'class="self-check"' in sc_body
        # Application page renders activity cards.
        app_body = _main_body(files["week_01_application.html"])
        assert "activity-card" in app_body
        # Discussion page renders prompt.
        disc_body = _main_body(files["week_01_discussion.html"])
        assert "discussion-prompt" in disc_body

    def test_full_mode_does_not_stamp_blocks_summary(
        self, course_data_path, tmp_path
    ):
        """Backward compat: full mode without classification → no stub."""
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "full_out"
        generate_course(str(course_data_path), str(out), emit_mode="full")
        # No stub written when classification is empty AND mode is full.
        assert not (out / "course_metadata.json").exists()


class TestEmitModeOutline:
    def test_outline_mode_suppresses_content_bodies(
        self, course_data_path, tmp_path
    ):
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "outline_out"
        generate_course(str(course_data_path), str(out), emit_mode="outline")
        files = _read_week_files(out)
        # Content page: NO content-type / flip-card / callout in <main> body.
        content_keys = [k for k in files if "content" in k]
        assert len(content_keys) == 1
        content_body = _main_body(files[content_keys[0]])
        assert "data-cf-content-type=" not in content_body, (
            f"content-type leaked into outline-mode content page:\n"
            f"{content_body[:500]}"
        )
        assert "flip-card" not in content_body
        # `.callout` lives in CSS (head) but no <div class="callout"> in body.
        assert 'class="callout' not in content_body

    def test_outline_mode_suppresses_self_check_bodies(
        self, course_data_path, tmp_path
    ):
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "outline_out"
        generate_course(str(course_data_path), str(out), emit_mode="outline")
        files = _read_week_files(out)
        sc_body = _main_body(files["week_01_self_check.html"])
        # No radio inputs / self-check item wrappers in outline mode.
        assert 'class="self-check"' not in sc_body
        assert '<input type="radio"' not in sc_body
        # No "Self-Check" heading either — the entire body is suppressed.
        assert "Self-Check: Test Your Understanding" not in sc_body

    def test_outline_mode_suppresses_activity_cards(
        self, course_data_path, tmp_path
    ):
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "outline_out"
        generate_course(str(course_data_path), str(out), emit_mode="outline")
        files = _read_week_files(out)
        app_body = _main_body(files["week_01_application.html"])
        # CSS class `.activity-card` lives in head <style>; assert no
        # `<div class="activity-card …">` wrapper appears in <main>.
        assert 'class="activity-card' not in app_body
        assert "Learning Activities" not in app_body

    def test_outline_mode_suppresses_discussion_body(
        self, course_data_path, tmp_path
    ):
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "outline_out"
        generate_course(str(course_data_path), str(out), emit_mode="outline")
        files = _read_week_files(out)
        disc_body = _main_body(files["week_01_discussion.html"])
        # CSS class `.discussion-prompt` lives in head; assert the
        # `<div class="discussion-prompt …">` wrapper is absent in body.
        assert 'class="discussion-prompt"' not in disc_body
        assert "Discussion Forum" not in disc_body

    def test_outline_mode_keeps_objectives(self, course_data_path, tmp_path):
        """Objectives are an outline-tier block_type — must persist."""
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "outline_out"
        generate_course(str(course_data_path), str(out), emit_mode="outline")
        files = _read_week_files(out)
        overview_body = _main_body(files["week_01_overview.html"])
        # Objectives section + per-LO list items survive.
        assert 'class="objectives"' in overview_body
        assert 'data-cf-objective-id="TO-01"' in overview_body
        assert 'data-cf-objective-id="CO-01"' in overview_body
        # And overview readings list still renders (overview body chrome).
        assert "Readings" in overview_body

    def test_outline_mode_keeps_summary_takeaways(
        self, course_data_path, tmp_path
    ):
        """Summary key takeaways are summary_takeaway block_type — outline-tier."""
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "outline_out"
        generate_course(str(course_data_path), str(out), emit_mode="outline")
        files = _read_week_files(out)
        summary_body = _main_body(files["week_01_summary.html"])
        assert "Key Takeaways" in summary_body
        assert "Concepts are mental abstractions" in summary_body

    def test_outline_mode_keeps_template_chrome(
        self, course_data_path, tmp_path
    ):
        """Chrome (header/footer/skip-link) is a chrome block_type — outline-tier."""
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "outline_out"
        generate_course(str(course_data_path), str(out), emit_mode="outline")
        files = _read_week_files(out)
        for name, html in files.items():
            assert 'data-cf-role="template-chrome"' in html, (
                f"chrome role missing on outline-mode page {name}"
            )

    def test_outline_mode_stamps_blocks_summary_outline_only(
        self, course_data_path, tmp_path
    ):
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "outline_out"
        generate_course(str(course_data_path), str(out), emit_mode="outline")
        stub_path = out / "course_metadata.json"
        assert stub_path.exists(), (
            "outline mode must emit course_metadata.json stub even without "
            "--division (downstream packager reads blocks_summary.outline_only)"
        )
        stub = json.loads(stub_path.read_text(encoding="utf-8"))
        assert stub.get("blocks_summary", {}).get("outline_only") is True
        # Course code + title still populated.
        assert stub["course_code"] == "OUTLINE_101"

    def test_outline_mode_invalid_value_raises(self, course_data_path, tmp_path):
        from generate_course import generate_course  # noqa: E402

        out = tmp_path / "bogus_out"
        with pytest.raises(ValueError, match="emit_mode"):
            generate_course(str(course_data_path), str(out), emit_mode="bogus")
