"""Placeholder-free content-generation tests.

Covers the "no placeholder generations" directive for
``MCP.tools._content_gen_helpers``:

  * When the source corpus lacks real Learning-Objective sections,
    :func:`synthesize_objectives_from_topics` must NOT emit the legacy
    "Apply concepts from X to analyze real-world examples" / "Describe X
    and explain the core ideas" / "Differentiate key aspects of Y"
    templates.
  * When the source corpus lacks real misconception/correction pairs,
    :func:`_build_misconceptions_for_week` must return an empty list
    (NOT the legacy "Students often assume X is a single idea" template).
  * When the source corpus lacks real exercise/review-question markers,
    :func:`_build_self_check_questions` must return an empty list (NOT
    the legacy "Which of the following best describes X?" template).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402


# ---------------------------------------------------------------------- #
# Banned phrases — the exact templates the user directive called out.
# ---------------------------------------------------------------------- #

BANNED_TEMPLATE_PHRASES = [
    "Apply concepts from ",
    "analyze real-world examples",
    "explain the core ideas presented in the source",
    "Differentiate key aspects of ",
    "Which of the following best describes ",
    "Students often assume ",
    "is a single idea with a single definition",
    "A minor footnote only mentioned in passing",
    "It replaces the earlier material entirely",
]


def _assert_no_banned_phrases(payload: str):
    """Scan a JSON-serialisable payload for any banned template phrase."""
    for phrase in BANNED_TEMPLATE_PHRASES:
        assert phrase not in payload, (
            f"Banned placeholder phrase leaked into emitted payload: "
            f"{phrase!r}"
        )


# ---------------------------------------------------------------------- #
# Topic fixtures
# ---------------------------------------------------------------------- #


def _topic_without_extractions(heading: str) -> dict:
    """Topic dict where the source had NO LO section, NO misconception
    pairs, and NO exercises."""
    return {
        "heading": heading,
        "paragraphs": [
            "This section describes the basic concept. It consists of "
            "several related ideas and technical vocabulary students must "
            "become familiar with to follow later material."
        ],
        "key_terms": ["Basic Concept"],
        "source_file": "chapter01",
        "word_count": 40,
        "extracted_lo_statements": [],
        "extracted_misconceptions": [],
        "extracted_questions": [],
    }


def _topic_with_extractions(heading: str) -> dict:
    """Topic dict where the source HAD real LOs, misconceptions, and
    exercises that the extractor successfully pulled out."""
    return {
        "heading": heading,
        "paragraphs": ["Body paragraph for the topic." * 3],
        "key_terms": ["Photosynthesis"],
        "source_file": "chapter01",
        "word_count": 50,
        "extracted_lo_statements": [
            "Describe the two stages of photosynthesis and where each occurs.",
            "Explain why chlorophyll reflects green light.",
        ],
        "extracted_misconceptions": [
            {
                "misconception": "Plants get their food from the soil.",
                "correction": "Plants produce their own food through "
                              "photosynthesis; soil provides water and "
                              "minerals only.",
            }
        ],
        "extracted_questions": [
            {
                "question": "Explain in one sentence why chloroplasts "
                            "reflect green light.",
                "bloom_level": "understand",
                "options": [],
            }
        ],
    }


# ---------------------------------------------------------------------- #
# synthesize_objectives_from_topics
# ---------------------------------------------------------------------- #


class TestObjectiveSynthesisNoPlaceholders:
    def test_empty_corpus_returns_empty_lists(self):
        terminal, chapter = _cgh.synthesize_objectives_from_topics([], 4)
        assert terminal == []
        assert chapter == []

    def test_no_extracted_los_uses_real_heading_not_template(self):
        topics = [_topic_without_extractions("Introduction to Photosynthesis")]
        terminal, chapter = _cgh.synthesize_objectives_from_topics(topics, 1)
        # Must emit SOMETHING (a real heading IS source content).
        assert len(terminal) >= 1
        for entry in terminal + chapter:
            stmt = entry["statement"]
            # The heading IS the source; it must appear verbatim-ish.
            # Banned template phrases must NOT appear.
            for phrase in BANNED_TEMPLATE_PHRASES:
                assert phrase not in stmt, (
                    f"Banned template {phrase!r} in objective statement: {stmt!r}"
                )

    def test_extracted_los_emitted_verbatim(self):
        topics = [_topic_with_extractions("Photosynthesis")]
        terminal, chapter = _cgh.synthesize_objectives_from_topics(topics, 1)
        combined = terminal + chapter
        statements = {e["statement"] for e in combined}
        # Both real LO statements must appear.
        assert any(
            "two stages of photosynthesis" in s.lower() for s in statements
        )
        assert any(
            "chlorophyll reflects green" in s.lower() for s in statements
        )
        # Banned template phrases must NOT.
        for entry in combined:
            for phrase in BANNED_TEMPLATE_PHRASES:
                assert phrase not in entry["statement"]


# ---------------------------------------------------------------------- #
# _build_misconceptions_for_week
# ---------------------------------------------------------------------- #


class TestMisconceptionBuilderNoPlaceholders:
    def test_empty_topics_returns_empty(self):
        assert _cgh._build_misconceptions_for_week([]) == []

    def test_topics_without_extracted_misconceptions_returns_empty(self):
        topics = [
            _topic_without_extractions("Introduction to Photosynthesis"),
            _topic_without_extractions("The Calvin Cycle"),
        ]
        result = _cgh._build_misconceptions_for_week(topics)
        assert result == [], (
            f"Expected empty list when no real misconceptions; got {result}"
        )

    def test_topics_with_extracted_misconceptions_passes_them_through(self):
        topics = [_topic_with_extractions("Photosynthesis")]
        result = _cgh._build_misconceptions_for_week(topics)
        assert len(result) == 1
        assert result[0]["misconception"] == (
            "Plants get their food from the soil."
        )
        assert "photosynthesis" in result[0]["correction"].lower()

    def test_no_synthesized_generic_misconception_leaked(self):
        """The legacy generic 'Students often assume X is a single idea'
        template must never appear."""
        topics = [_topic_without_extractions("Anything")]
        result = _cgh._build_misconceptions_for_week(topics)
        for m in result:
            for phrase in BANNED_TEMPLATE_PHRASES:
                assert phrase not in m.get("misconception", "")
                assert phrase not in m.get("correction", "")


# ---------------------------------------------------------------------- #
# _build_self_check_questions
# ---------------------------------------------------------------------- #


class TestSelfCheckBuilderNoPlaceholders:
    def test_empty_topics_returns_empty(self):
        assert _cgh._build_self_check_questions([], []) == []

    def test_topics_without_extracted_questions_returns_empty(self):
        topics = [_topic_without_extractions("Anything")]
        objectives = [
            {"id": "TO-01", "statement": "Demo objective", "bloom_level": "apply"}
        ]
        result = _cgh._build_self_check_questions(topics, objectives)
        assert result == [], (
            f"Expected empty list without real exercises; got {result}"
        )

    def test_topics_with_extracted_questions_pass_through(self):
        topics = [_topic_with_extractions("Photosynthesis")]
        objectives = [
            {"id": "TO-01", "statement": "Demo objective", "bloom_level": "apply"}
        ]
        result = _cgh._build_self_check_questions(topics, objectives)
        assert len(result) == 1
        assert "chloroplasts" in result[0]["question"].lower()
        assert result[0].get("objective_ref") == "TO-01"

    def test_no_which_of_the_following_template_emitted(self):
        """The legacy 'Which of the following best describes X' template
        must never be produced here."""
        topics = [_topic_without_extractions("Photosynthesis")]
        objectives = [
            {"id": "TO-01", "statement": "Objective", "bloom_level": "apply"}
        ]
        result = _cgh._build_self_check_questions(topics, objectives)
        for q in result:
            for phrase in BANNED_TEMPLATE_PHRASES:
                assert phrase not in q["question"]


# ---------------------------------------------------------------------- #
# Source-extraction invariants
# ---------------------------------------------------------------------- #


class TestSourceExtractionInvariants:
    def test_extract_los_returns_empty_for_non_lo_text(self):
        text = (
            "The weather today is sunny. Photosynthesis occurs in plants. "
            "The Calvin cycle is one of two stages."
        )
        assert _cgh.extract_learning_objectives(text) == []

    def test_extract_los_captures_bulleted_section(self):
        text = (
            "Chapter Objectives\n"
            "After reading this chapter you will be able to:\n"
            "- Describe the role of chloroplasts in photosynthesis.\n"
            "- Explain the energetic coupling between stages.\n"
            "- Compare the light and dark reactions.\n"
            "\n"
            "Introduction\n"
            "Photosynthesis is..."
        )
        los = _cgh.extract_learning_objectives(text)
        assert len(los) >= 3
        assert any("chloroplasts" in s.lower() for s in los)

    def test_extract_misconceptions_requires_paired_shape(self):
        paired = (
            "Misconception: Plants eat soil. Correction: Plants produce "
            "their own food via photosynthesis."
        )
        assert len(_cgh.extract_misconceptions(paired)) == 1

        unpaired_only_misc = "Common misconception: plants eat soil."
        assert _cgh.extract_misconceptions(unpaired_only_misc) == []

    def test_extract_questions_requires_exercise_marker(self):
        text_with = "Review question 2.3. Why do plants appear green to the eye?"
        assert len(_cgh.extract_self_check_questions(text_with)) == 1

        text_without = "Why do plants appear green to the eye?"
        assert _cgh.extract_self_check_questions(text_without) == []


# ---------------------------------------------------------------------- #
# Grep-hardening: the banned source-level f-strings must not reappear.
# ---------------------------------------------------------------------- #


def test_source_file_contains_no_banned_fstring_templates():
    """Static check: grep the source file for the exact template phrases
    the task directive called out. Any reappearance fails this test.
    """
    source_path = PROJECT_ROOT / "MCP" / "tools" / "_content_gen_helpers.py"
    src = source_path.read_text(encoding="utf-8")
    banned_needles = [
        "Which of the following best describes",
        "Apply concepts from ",
        "Describe {",  # matches old f-string fragment
        "Differentiate key aspects of",
        "Students often assume",
        "is a single idea with a single definition",
    ]
    for needle in banned_needles:
        # Allow the needle to appear inside a comment or docstring that
        # *describes* what's banned (we document the removal). Strict
        # check: no *code* occurrence outside comments.
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Skip docstrings. Very crude: lines inside triple-quoted
            # strings — we just check that if the needle appears, the
            # line is a comment or an assertion/docstring reference. To
            # keep this test precise without full parsing, we accept
            # occurrences INSIDE a string literal on a non-code line
            # that happens to mention the banned phrase. The simple
            # rule: flag only f-string / string-format occurrences,
            # i.e. lines that contain an f-string opening (``f"``) AND
            # the banned phrase.
            if needle in line and 'f"' in line and not stripped.startswith('"""'):
                pytest.fail(
                    f"Banned placeholder template reintroduced in "
                    f"_content_gen_helpers.py: {line.strip()!r}"
                )
