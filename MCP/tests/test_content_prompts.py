"""Wave 34 tests: prompt builders for mailbox-brokered subagent tasks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.orchestrator.content_prompts import (  # noqa: E402
    build_alt_text_prompt,
    build_content_generation_prompt,
    build_synthesize_training_prompt,
)


class TestContentGenerationPrompt:
    def test_includes_chapter_html_and_lo_list(self, tmp_path: Path):
        los = [
            {"id": "TO-01", "statement": "Explain Newton's first law"},
            {"id": "TO-02", "statement": "Apply F=ma to solved problems"},
        ]
        prompt = build_content_generation_prompt(
            week_n=3,
            chapter_html="<section><h1>Chapter 3</h1><p>Motion</p></section>",
            planned_los=los,
            output_dir=tmp_path / "weeks" / "week_3",
            chapter_id="ch-03",
        )
        assert "week 3" in prompt.lower() or "week_3" in prompt
        assert "<chapter-html>" in prompt
        assert "Chapter 3" in prompt  # chapter HTML content present
        assert "TO-01" in prompt
        assert "TO-02" in prompt
        assert "Newton" in prompt  # LO statement text present
        # Output dir is referenced so the subagent knows where to write
        assert str(tmp_path / "weeks" / "week_3") in prompt

    def test_schema_contract_listed(self, tmp_path: Path):
        prompt = build_content_generation_prompt(
            week_n=1,
            chapter_html="x" * 100,
            planned_los=[{"id": "TO-01", "statement": "foo"}],
            output_dir=tmp_path,
        )
        # Enforces the 4-page contract and the data-cf-* attribute rule.
        for required_token in (
            "overview",
            "content",
            "application",
            "summary",
            "data-cf-role",
            "data-cf-source-ids",
            "data-cf-objective-ids",
            "source_refs",  # the gate name
        ):
            assert required_token in prompt, f"expected {required_token!r} in prompt"

        # Return shape: status + outputs.pages list
        assert '"status": "ok"' in prompt
        assert '"pages"' in prompt
        assert "week_1_overview.html" in prompt

    def test_truncates_long_chapter_html(self, tmp_path: Path):
        big_html = "a" * 100_000
        prompt = build_content_generation_prompt(
            week_n=2,
            chapter_html=big_html,
            planned_los=["TO-01"],
            output_dir=tmp_path,
            max_chapter_chars=500,
        )
        assert "[truncated]" in prompt
        # The full 100k payload must NOT have been embedded.
        assert prompt.count("a") < 2000


class TestAltTextPrompt:
    def test_includes_caption_and_context(self):
        prompt = build_alt_text_prompt(
            figure_b64="AAAA",
            caption="Figure 2.1: Projectile motion parabola",
            context="The projectile rises to apex then falls symmetrically.",
            figure_id="fig-02-01",
        )
        assert "Figure 2.1" in prompt
        assert "apex" in prompt
        assert "fig-02-01" in prompt
        assert "AAAA" in prompt
        assert "<figure-base64>" in prompt

    def test_alt_text_contract_present(self):
        prompt = build_alt_text_prompt(
            figure_b64="BBBB",
            caption="chart",
            context="ctx",
        )
        for token in (
            "alt_text",
            "decorative",
            "confidence",
            "180 characters",
            "image of",  # the "don't say 'image of'" rule
        ):
            assert token in prompt
        assert '"status": "ok"' in prompt

    def test_raw_bytes_trigger_size_hint_not_payload(self):
        """If caller passes figure_bytes without b64, the prompt should
        carry only a size hint — not the raw binary."""
        fake_bytes = b"\x89PNG\r\n" + b"\x00" * 500
        prompt = build_alt_text_prompt(
            figure_bytes=fake_bytes,
            caption="c",
            context="c",
        )
        assert "size hint" in prompt
        assert str(len(fake_bytes)) in prompt
        # No raw binary sneaked into the prompt string.
        assert "\x89PNG" not in prompt


class TestSynthesizeTrainingPrompt:
    def test_includes_chunk_text_and_lo_refs(self):
        prompt = build_synthesize_training_prompt(
            chunk_text="A catalyst lowers activation energy without being consumed.",
            lo_refs=[{"id": "TO-04", "statement": "Describe catalyst action"}],
            chunk_id="ch-04-para-07",
            content_type="explanation",
            bloom_level="understand",
        )
        assert "catalyst lowers activation energy" in prompt
        assert "TO-04" in prompt
        assert "ch-04-para-07" in prompt
        assert "explanation" in prompt
        assert "understand" in prompt

    def test_return_shape_mentions_both_pairs(self):
        prompt = build_synthesize_training_prompt(
            chunk_text="x",
            lo_refs=["TO-01"],
        )
        assert "instruction_pair" in prompt
        assert "preference_pair" in prompt
        assert "chosen" in prompt
        assert "rejected" in prompt
        # Schema contract: preference_pair prompt min length (40 chars)
        assert "40 characters" in prompt


class TestNoCorpusSpecificLeak:
    """Canon rule: prompt builders must not bake corpus-specific
    identifiers into the returned string. Callers provide everything.
    """

    @pytest.mark.parametrize(
        "builder_call",
        [
            # content-gen with neutral tokens
            lambda: build_content_generation_prompt(
                week_n=1,
                chapter_html="neutral html",
                planned_los=["LO_ALPHA"],
                output_dir=Path("/tmp/out"),
            ),
            # alt-text
            lambda: build_alt_text_prompt(
                figure_b64="ZZZZ",
                caption="a caption",
                context="neutral context",
            ),
            # training synth
            lambda: build_synthesize_training_prompt(
                chunk_text="neutral chunk",
                lo_refs=["LO_ALPHA"],
            ),
        ],
    )
    def test_no_corpus_identifier_leaks(self, builder_call):
        prompt = builder_call()
        # Sentinel corpus tokens we've used historically in tests/docs —
        # none of these should appear unless the caller passed them.
        for forbidden in (
            "PHYS_101",
            "BIO_201",
            "CHEM_101",
            "INT_101",
            "SYNTH_101",
        ):
            assert forbidden not in prompt, (
                f"builder leaked {forbidden!r} into the prompt"
            )


class TestReturnContractMatchesDispatcherShape:
    """The dispatcher's mailbox envelope parser expects either a
    ``result`` dict containing ``status`` and ``phase_name`` / ``outputs``
    fields, or a ``raw`` JSON string parseable into the same. The
    prompt contract must therefore mention ``status`` + some form of
    ``outputs`` / payload keys."""

    def test_content_generation_prompt_pins_status_and_outputs(self, tmp_path: Path):
        prompt = build_content_generation_prompt(
            week_n=1,
            chapter_html="x",
            planned_los=["TO-01"],
            output_dir=tmp_path,
        )
        assert '"status"' in prompt
        assert "outputs" in prompt

    def test_alt_text_prompt_pins_status_and_alt_text(self):
        prompt = build_alt_text_prompt(
            figure_b64="AAAA",
            caption="c",
            context="c",
        )
        assert '"status"' in prompt
        assert "alt_text" in prompt

    def test_training_prompt_pins_status_and_pairs(self):
        prompt = build_synthesize_training_prompt(
            chunk_text="x",
            lo_refs=["TO-01"],
        )
        assert '"status"' in prompt
        assert "instruction_pair" in prompt
        assert "preference_pair" in prompt
