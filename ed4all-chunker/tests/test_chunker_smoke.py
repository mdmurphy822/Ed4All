"""Smoke tests for the lifted ``ed4all_chunker`` package surface.

Phase 7a Subtask 5. These tests exercise the public API of the
package independently of Trainforge so a regression in the lifted
chunker proper, the lifted helpers, or the lifted boilerplate
detector is caught at the package boundary — not deferred to the
Trainforge regression suite (which would mask a packaging bug
behind in-tree call sites).

Self-containment contract: nothing in this file imports anything
from ``Trainforge`` or from the parent project. The package's
declared in-function lazy imports
(``Trainforge.parsers.html_content_parser`` /
``Trainforge.parsers.xpath_walker``) are NOT exercised here — those
fire only on call paths that need plain-text extraction or xpath
resolution, neither of which the smoke contracts tested here
require. See the package's chunker module docstring for the
rationale.

Test surface (per Subtask 5 spec):
    1. Empty-input contract — ``chunk_content([], 'TEST_101', ctx=None)``
       returns an empty result and tuple-unpacks cleanly.
    2. Non-empty input WITHOUT ``ctx`` raises
       :class:`ChunkerContextRequired` (loud-fail, no silent drop).
    3. Non-empty input WITH ``ctx`` dispatches to ``ctx.create_chunk``
       with the expected kwargs.
    4. ``pages_with_misconceptions`` is populated from input items
       whose ``misconceptions`` field is truthy.
    5. ``MIN_CHUNK_SIZE`` / ``MAX_CHUNK_SIZE`` / ``TARGET_CHUNK_SIZE``
       constants match the Trainforge baseline (100 / 800 / 500).
    6. ``CANONICAL_CHUNK_TYPES`` is a frozenset with the expected
       canonical chunk-type values.
    7. ``TRAINFORGE_CONTENT_HASH_IDS`` env var toggles between
       position-based and content-hash chunk IDs.

Plus boilerplate + helpers smoke (folded into one file per the
Subtask 5 plan-spec note that authoring as one file is acceptable):
    8. ``strip_boilerplate`` round-trip per the plan's verification
       snippet.
    9. ``detect_repeated_ngrams`` finds repeated spans in fixture text.
    10. ``BoilerplateConfig`` instantiates with defaults.
    11. ``type_from_resource`` known-mapping coverage.
    12. ``strip_assessment_feedback`` strips ``<div class="sc-feedback">``.
    13. ``strip_feedback_from_text`` strips ``Correct.`` / ``Incorrect.``
        line markers.
    14. ``extract_section_html`` returns ``""`` when heading is missing.
    15. Package-level re-export sanity check — every name in
        ``ed4all_chunker.__all__`` is importable from the top.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from ed4all_chunker import (
    BoilerplateConfig,
    CANONICAL_CHUNK_TYPES,
    ChunkContentResult,
    ChunkerContext,
    ChunkerContextRequired,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    TARGET_CHUNK_SIZE,
    chunk_content,
    detect_repeated_ngrams,
    extract_section_html,
    strip_assessment_feedback,
    strip_boilerplate,
    strip_feedback_from_text,
    type_from_resource,
)
from ed4all_chunker.chunker import _generate_chunk_id


# ---------------------------------------------------------------------------
# Test 1: empty-input contract
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_result_without_ctx() -> None:
    """``chunk_content([], 'TEST_101')`` returns empty without needing ctx."""

    result = chunk_content([], course_code="TEST_101", ctx=None)

    assert isinstance(result, ChunkContentResult)
    assert result.chunks == []
    assert result.pages_with_misconceptions == set()


def test_empty_input_result_supports_tuple_unpack() -> None:
    """Empty result tuple-unpacks to ``(chunks, pages_with_misconceptions)``."""

    chunks, pages_with_misconceptions = chunk_content(
        [], course_code="TEST_101", ctx=None
    )

    assert chunks == []
    assert pages_with_misconceptions == set()


# ---------------------------------------------------------------------------
# Test 2: non-empty input without ctx raises
# ---------------------------------------------------------------------------


def test_non_empty_input_without_ctx_raises_loud_fail() -> None:
    """Non-empty input without ``ctx`` MUST raise — silent no-op is forbidden."""

    parsed_items: List[Dict[str, Any]] = [
        {
            "module_id": "m1",
            "item_id": "i1",
            "module_title": "M1",
            "title": "I1",
            "resource_type": "content",
            "raw_html": "<p>" + ("word " * 50) + "</p>",
            "sections": [],
        }
    ]

    with pytest.raises(ChunkerContextRequired):
        chunk_content(parsed_items, course_code="TEST_101", ctx=None)


# ---------------------------------------------------------------------------
# Test 3: non-empty input with ctx dispatches to create_chunk
# ---------------------------------------------------------------------------


class _RecordingContext:
    """Minimal ChunkerContext callback recorder for dispatch assertions.

    The chunker calls ``ctx.create_chunk(**kwargs)`` once per resolved
    chunk; this recorder captures every kwarg dict and returns a
    minimal chunk dict so the chunker's follows-chunk linkage walks
    forward without exploding.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        # The chunker reads ``["id"]`` off the returned dict to set
        # ``follows_chunk_id`` for the next chunk in the same lesson,
        # so the returned shape MUST carry ``id`` even though the
        # production callback returns a richer dict.
        return {"id": kwargs["chunk_id"], "text": kwargs["text"]}


def test_non_empty_input_with_ctx_dispatches_to_create_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``chunk_content`` calls ``ctx.create_chunk`` once per resolved chunk.

    Force position-based chunk IDs (``TRAINFORGE_CONTENT_HASH_IDS`` unset)
    so the asserted ID format is deterministic. The single short item has
    no sections, so the loop falls into the no-sections branch and
    dispatches exactly once.
    """

    monkeypatch.delenv("TRAINFORGE_CONTENT_HASH_IDS", raising=False)

    recorder = _RecordingContext()
    ctx = ChunkerContext(create_chunk=recorder)

    parsed_items: List[Dict[str, Any]] = [
        {
            "module_id": "m1",
            "item_id": "i1",
            "module_title": "Module One",
            "title": "Item One",
            "resource_type": "content",
            "raw_html": "<p>"
            + " ".join(f"word{n}" for n in range(50))
            + "</p>",
            "sections": [],
        }
    ]

    result = chunk_content(
        parsed_items, course_code="TEST_101", ctx=ctx
    )

    assert len(recorder.calls) == 1, (
        f"expected one create_chunk dispatch, got {len(recorder.calls)}"
    )

    call = recorder.calls[0]
    # Required kwargs the chunker promises to pass through:
    expected_kwargs = {
        "chunk_id",
        "text",
        "html",
        "item",
        "section_heading",
        "chunk_type",
        "follows_chunk_id",
        "position_in_module",
        "html_xpath",
        "char_span",
        "section_source_ids",
        "merged_headings",
    }
    assert expected_kwargs.issubset(call.keys()), (
        "create_chunk dispatch missing kwargs: "
        f"{expected_kwargs - set(call.keys())}"
    )

    # Position-based chunk IDs use the format
    # ``{course_code.lower()}_chunk_{NNNNN}`` per ``_generate_chunk_id``.
    assert call["chunk_id"] == "test_101_chunk_00001"
    assert call["section_heading"] == "Item One"
    assert call["follows_chunk_id"] is None  # first chunk in the lesson
    assert call["position_in_module"] == 0

    # Result carries one chunk + zero misconception pages (input had no
    # ``misconceptions`` field).
    assert len(result.chunks) == 1
    assert result.pages_with_misconceptions == set()


# ---------------------------------------------------------------------------
# Test 4: pages_with_misconceptions populated from input
# ---------------------------------------------------------------------------


def test_pages_with_misconceptions_populated_from_input() -> None:
    """Items with truthy ``misconceptions`` field flow into the result set.

    No ``ctx`` is needed here: the population happens at the top of
    ``chunk_content`` via a set comprehension over ``parsed_items``,
    and we can short-circuit the per-item loop by triggering the
    ``ChunkerContextRequired`` raise after the comprehension fires.
    To assert the set without raising, we use a no-op ctx and a
    single item with no sections AND empty raw_html (which the loop
    then drops in the ``if text.strip():`` guard).
    """

    recorder = _RecordingContext()
    ctx = ChunkerContext(create_chunk=recorder)

    parsed_items: List[Dict[str, Any]] = [
        {
            "module_id": "m1",
            "item_id": "page_with_mc",
            "module_title": "M1",
            "title": "Page A",
            "resource_type": "content",
            "raw_html": "",
            "sections": [],
            "misconceptions": [{"text": "students confuse X with Y"}],
        },
        {
            "module_id": "m1",
            "item_id": "page_without_mc",
            "module_title": "M1",
            "title": "Page B",
            "resource_type": "content",
            "raw_html": "",
            "sections": [],
        },
        {
            "module_id": "m2",
            "item_id": "page_empty_mc",
            "module_title": "M2",
            "title": "Page C",
            "resource_type": "content",
            "raw_html": "",
            "sections": [],
            # Falsy misconceptions — should NOT register in the set.
            "misconceptions": [],
        },
    ]

    result = chunk_content(
        parsed_items, course_code="TEST_101", ctx=ctx
    )

    assert result.pages_with_misconceptions == {"page_with_mc"}
    # Empty raw_html means no chunks materialise.
    assert result.chunks == []
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# Test 5: chunk-size constants match Trainforge baseline
# ---------------------------------------------------------------------------


def test_chunk_size_constants_match_trainforge_baseline() -> None:
    """Ensure the Trainforge-baseline chunk-size constants survived the lift."""

    assert MIN_CHUNK_SIZE == 100
    assert MAX_CHUNK_SIZE == 800
    assert TARGET_CHUNK_SIZE == 500


# ---------------------------------------------------------------------------
# Test 6: CANONICAL_CHUNK_TYPES shape
# ---------------------------------------------------------------------------


def test_canonical_chunk_types_is_frozenset_with_expected_members() -> None:
    """Mirror ``Trainforge/process_course.py:103`` byte-for-byte."""

    assert isinstance(CANONICAL_CHUNK_TYPES, frozenset)
    expected = {
        "assessment_item",
        "overview",
        "summary",
        "exercise",
        "explanation",
        "example",
        "procedure",
        "real_world_scenario",
        "common_pitfall",
        "problem_solution",
    }
    assert set(CANONICAL_CHUNK_TYPES) == expected


# ---------------------------------------------------------------------------
# Test 7: TRAINFORGE_CONTENT_HASH_IDS env var toggles ID format
# ---------------------------------------------------------------------------


def test_chunk_id_position_based_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default mode emits zero-padded ordinal IDs."""

    monkeypatch.delenv("TRAINFORGE_CONTENT_HASH_IDS", raising=False)
    chunk_id = _generate_chunk_id(
        prefix="test_101_chunk_",
        start_id=1,
        text="ignored under position mode",
        source_locator="m1/i1",
    )
    assert chunk_id == "test_101_chunk_00001"


def test_chunk_id_content_hash_when_env_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TRAINFORGE_CONTENT_HASH_IDS=true`` emits a 16-hex-char digest."""

    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "true")
    chunk_id = _generate_chunk_id(
        prefix="test_101_chunk_",
        start_id=1,
        text="some text used as the content-hash payload",
        source_locator="m1/i1",
    )
    assert chunk_id.startswith("test_101_chunk_")
    digest = chunk_id[len("test_101_chunk_"):]
    assert len(digest) == 16, f"expected 16-char digest, got {len(digest)}"
    assert all(c in "0123456789abcdef" for c in digest), (
        "digest must be lowercase hex"
    )

    # Re-encoding the same payload must produce the same ID
    # (deterministic — the regression we want to catch is a future
    # change to the salt / schema-version that breaks LibV2 corpora).
    again = _generate_chunk_id(
        prefix="test_101_chunk_",
        start_id=999,  # ignored under content-hash mode
        text="some text used as the content-hash payload",
        source_locator="m1/i1",
    )
    assert chunk_id == again


def test_chunk_id_content_hash_distinct_for_different_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different text ⇒ different content-hash IDs."""

    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "true")
    a = _generate_chunk_id("p_", 1, "alpha", "m1/i1")
    b = _generate_chunk_id("p_", 1, "beta", "m1/i1")
    assert a != b


# ---------------------------------------------------------------------------
# Test 8: strip_boilerplate round-trip (Subtask 2 verification snippet)
# ---------------------------------------------------------------------------


def test_strip_boilerplate_simple_round_trip() -> None:
    """Mirrors the Subtask 2 plan-cited verification snippet."""

    out, removed = strip_boilerplate("hello world", ["world"])
    assert out == "hello"
    assert removed == 1


def test_strip_boilerplate_no_spans_returns_input_unchanged() -> None:
    out, removed = strip_boilerplate("hello world", [])
    assert out == "hello world"
    assert removed == 0


# ---------------------------------------------------------------------------
# Test 9: detect_repeated_ngrams finds repeated spans
# ---------------------------------------------------------------------------


def test_detect_repeated_ngrams_returns_repeated_span() -> None:
    """When the same 4-gram appears in every doc it must be detected."""

    repeated = " ".join(["common", "footer", "text", "appears"] * 4)
    documents = [
        f"Page one content {repeated} epilogue alpha.",
        f"Page two content {repeated} epilogue beta.",
        f"Page three content {repeated} epilogue gamma.",
    ]

    spans = detect_repeated_ngrams(documents, n=4, min_doc_frac=0.5)
    assert spans, "expected at least one repeated span"
    assert any("common footer text appears" in span for span in spans)


def test_detect_repeated_ngrams_empty_input_returns_empty() -> None:
    assert detect_repeated_ngrams([], n=10, min_doc_frac=0.3) == []


# ---------------------------------------------------------------------------
# Test 10: BoilerplateConfig defaults
# ---------------------------------------------------------------------------


def test_boilerplate_config_instantiates_with_defaults() -> None:
    cfg = BoilerplateConfig()
    assert cfg.min_ngram_tokens == 15
    assert cfg.min_doc_frac == 0.30


def test_boilerplate_config_accepts_overrides() -> None:
    cfg = BoilerplateConfig(min_ngram_tokens=8, min_doc_frac=0.5)
    assert cfg.min_ngram_tokens == 8
    assert cfg.min_doc_frac == 0.5


# ---------------------------------------------------------------------------
# Test 11: type_from_resource known-mapping coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resource_type, expected",
    [
        ("quiz", "assessment_item"),
        ("overview", "overview"),
        ("summary", "summary"),
        ("discussion", "exercise"),
        ("application", "exercise"),
        ("unknown_resource_type", "explanation"),  # default fallback
        ("", "explanation"),
    ],
)
def test_type_from_resource_mapping(resource_type: str, expected: str) -> None:
    assert type_from_resource(resource_type) == expected


# ---------------------------------------------------------------------------
# Test 12: strip_assessment_feedback removes Courseforge feedback markup
# ---------------------------------------------------------------------------


def test_strip_assessment_feedback_removes_sc_feedback_div() -> None:
    html = (
        '<div class="question">What is 2+2?</div>'
        '<div class="sc-feedback">The correct answer is 4 because...</div>'
        '<label data-correct="true">4</label>'
    )
    cleaned = strip_assessment_feedback(html)
    assert "sc-feedback" not in cleaned
    assert "The correct answer is 4" not in cleaned
    assert 'data-correct=' not in cleaned
    # Non-feedback content survives.
    assert "What is 2+2?" in cleaned


def test_strip_assessment_feedback_idempotent_on_clean_html() -> None:
    html = "<p>plain content with no feedback markers</p>"
    assert strip_assessment_feedback(html) == html


# ---------------------------------------------------------------------------
# Test 13: strip_feedback_from_text removes feedback line markers
# ---------------------------------------------------------------------------


def test_strip_feedback_from_text_drops_correct_incorrect_lines() -> None:
    text = (
        "Question: What is 2+2?\n"
        "Correct. Four is the right answer.\n"
        "Incorrect. Try again.\n"
        "Final score: 1/2"
    )
    cleaned = strip_feedback_from_text(text)
    assert "Correct." not in cleaned
    assert "Incorrect." not in cleaned
    assert "What is 2+2?" in cleaned
    assert "Final score" in cleaned


# ---------------------------------------------------------------------------
# Test 14: extract_section_html boundary cases
# ---------------------------------------------------------------------------


def test_extract_section_html_returns_empty_when_heading_missing() -> None:
    assert extract_section_html("<p>no heading here</p>", "Missing") == ""


def test_extract_section_html_returns_empty_for_empty_inputs() -> None:
    assert extract_section_html("", "Heading") == ""
    assert extract_section_html("<h2>Heading</h2>", "") == ""


def test_extract_section_html_returns_just_heading_when_outside_section() -> None:
    """Heading outside any ``<section>`` returns just the heading element."""

    html = "<h1>Page Title</h1><p>body</p>"
    out = extract_section_html(html, "Page Title")
    assert out == "<h1>Page Title</h1>"


def test_extract_section_html_returns_full_section_when_inside_section() -> None:
    """Heading inside a ``<section>`` returns the enclosing section."""

    html = (
        "<section><h2>Inside</h2><p>section body</p></section>"
        "<section><h2>Other</h2></section>"
    )
    out = extract_section_html(html, "Inside")
    assert out.startswith("<section>")
    assert out.endswith("</section>")
    assert "section body" in out
    assert "Other" not in out


# ---------------------------------------------------------------------------
# Test 15: package re-exports — every __all__ name is importable
# ---------------------------------------------------------------------------


def test_package_reexports_all_declared_names() -> None:
    import ed4all_chunker as pkg

    for name in pkg.__all__:
        assert hasattr(pkg, name), (
            f"ed4all_chunker.__all__ declares {name!r} but it is not "
            "an attribute of the package — re-export drift."
        )


# ---------------------------------------------------------------------------
# Self-containment sentinel — pure ``import ed4all_chunker`` does NOT load
# Trainforge transitively
# ---------------------------------------------------------------------------


def test_pure_import_does_not_load_trainforge_modules() -> None:
    """``import ed4all_chunker`` MUST NOT eagerly load any Trainforge module.

    The package's two declared lazy imports
    (``Trainforge.parsers.html_content_parser`` inside
    :func:`ed4all_chunker.helpers.extract_plain_text`; and
    ``Trainforge.parsers.xpath_walker`` inside
    :func:`ed4all_chunker.chunker.chunk_text_block`) are CALL-time
    lazy imports — they fire only when those specific functions are
    invoked. Pure ``import ed4all_chunker`` against a fresh
    interpreter must come up clean: no Trainforge modules in
    ``sys.modules``.

    Verifying via subprocess so prior tests in this same pytest
    session (which DO drive the chunker through call paths that
    legitimately fire the lazy imports) don't poison the
    ``sys.modules`` snapshot.

    If a future change in the package eagerly imports Trainforge
    (e.g. a top-of-module import of ``html_content_parser``), this
    sentinel catches it at the package boundary instead of deferring
    the regression to the parent project's regression suite.
    """

    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import ed4all_chunker  # noqa: F401\n"
                "loaded = [m for m in sys.modules if m.startswith('Trainforge')]\n"
                "print(';'.join(sorted(loaded)))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = [m for m in completed.stdout.strip().split(";") if m]
    assert loaded == [], (
        "Pure ``import ed4all_chunker`` MUST NOT load Trainforge "
        f"modules; subprocess found: {loaded}"
    )
