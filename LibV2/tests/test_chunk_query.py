"""Engine-layer tests for ``LibV2.tools.chunk_query`` (Wave 77 Worker β)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from LibV2.tools.chunk_query import (
    BLOOM_LEVELS,
    CHUNK_TYPES,
    DIFFICULTY_LEVELS,
    SORT_KEYS,
    ChunkQueryError,
    MalformedArchiveError,
    QueryFilter,
    UnknownSlugError,
    _build_to_to_cos,
    _expand_outcomes,
    parse_csv,
    parse_week_spec,
    query_chunks,
    validate_choice,
)


# ---------------------------------------------------------------------- #
# Synthetic archive builder                                               #
# ---------------------------------------------------------------------- #


def _make_archive(courses_root: Path, slug: str, chunks: List[Dict[str, Any]],
                  objectives: Dict[str, Any] = None) -> Path:
    root = courses_root / slug
    (root / "corpus").mkdir(parents=True)
    with (root / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    if objectives is not None:
        (root / "objectives.json").write_text(
            json.dumps(objectives), encoding="utf-8"
        )
    return root


def _basic_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "id": "c01",
            "chunk_type": "explanation",
            "difficulty": "foundational",
            "bloom_level": "remember",
            "text": "Intro chunk about RDF triples.",
            "word_count": 100,
            "learning_outcome_refs": ["co-01"],
            "source": {"module_id": "week_01_overview"},
        },
        {
            "id": "c02",
            "chunk_type": "example",
            "difficulty": "intermediate",
            "bloom_level": "apply",
            "text": "Example with sh:minCount usage.",
            "word_count": 150,
            "learning_outcome_refs": ["co-02", "to-01"],
            "source": {"module_id": "week_03_content_01"},
        },
        {
            "id": "c03",
            "chunk_type": "exercise",
            "difficulty": "intermediate",
            "bloom_level": "apply",
            "text": "Exercise: write a SHACL shape.",
            "word_count": 200,
            "learning_outcome_refs": ["co-16"],
            "source": {"module_id": "week_07_application"},
        },
        {
            "id": "c04",
            "chunk_type": "exercise",
            "difficulty": "advanced",
            "bloom_level": "analyze",
            "text": "Analyze the constraint violation.",
            "word_count": 250,
            "learning_outcome_refs": ["co-17"],
            "source": {"module_id": "week_07_application"},
        },
        {
            "id": "c05",
            "chunk_type": "assessment_item",
            "difficulty": "advanced",
            "bloom_level": "evaluate",
            "text": "Question on SHACL features.",
            "word_count": 50,
            "learning_outcome_refs": ["to-04"],
            "source": {"module_id": "week_08_assessment"},
        },
    ]


def _basic_objectives() -> Dict[str, Any]:
    return {
        "terminal_outcomes": [{"id": "to-01"}, {"id": "to-04"}],
        "component_objectives": [
            {"id": "co-01", "parent_terminal": "to-01"},
            {"id": "co-02", "parent_terminal": "to-01"},
            {"id": "co-16", "parent_terminal": "to-04"},
            {"id": "co-17", "parent_terminal": "to-04"},
        ],
    }


# ---------------------------------------------------------------------- #
# Constants                                                               #
# ---------------------------------------------------------------------- #


def test_chunk_types_constant_is_canonical():
    # The 6 ontology-canonical chunk types.
    assert set(CHUNK_TYPES) == {
        "explanation",
        "example",
        "exercise",
        "assessment_item",
        "overview",
        "summary",
    }


def test_bloom_levels_in_pedagogical_order():
    assert BLOOM_LEVELS == (
        "remember",
        "understand",
        "apply",
        "analyze",
        "evaluate",
        "create",
    )


def test_difficulty_levels_three_tier():
    assert set(DIFFICULTY_LEVELS) == {
        "foundational",
        "intermediate",
        "advanced",
    }


def test_sort_keys_exposes_supported_set():
    assert set(SORT_KEYS) == {"week", "chunk_id", "word_count", "bloom"}


# ---------------------------------------------------------------------- #
# Parsing helpers                                                         #
# ---------------------------------------------------------------------- #


def test_parse_week_spec_single():
    assert parse_week_spec("7") == (7, 7)


def test_parse_week_spec_range():
    assert parse_week_spec("1-12") == (1, 12)


def test_parse_week_spec_rejects_inverted():
    with pytest.raises(ValueError):
        parse_week_spec("10-3")


def test_parse_week_spec_rejects_empty():
    with pytest.raises(ValueError):
        parse_week_spec("")


def test_parse_csv_returns_none_for_none():
    assert parse_csv(None) is None


def test_parse_csv_strips_and_drops_empty():
    assert parse_csv("a, b ,, c") == ["a", "b", "c"]


def test_parse_csv_returns_none_for_all_empty():
    assert parse_csv(",,") is None


def test_validate_choice_accepts_known():
    validate_choice(["apply"], BLOOM_LEVELS, "--bloom")


def test_validate_choice_rejects_unknown():
    with pytest.raises(ValueError):
        validate_choice(["bogus"], BLOOM_LEVELS, "--bloom")


def test_validate_choice_no_op_on_empty():
    validate_choice(None, BLOOM_LEVELS, "--bloom")
    validate_choice([], BLOOM_LEVELS, "--bloom")


# ---------------------------------------------------------------------- #
# Outcome rollup helpers                                                  #
# ---------------------------------------------------------------------- #


def test_build_to_to_cos_handles_missing_objectives():
    assert _build_to_to_cos(None) == {}
    assert _build_to_to_cos({}) == {}


def test_build_to_to_cos_groups_by_parent():
    objs = _basic_objectives()
    mapping = _build_to_to_cos(objs)
    assert mapping["to-01"] == {"co-01", "co-02"}
    assert mapping["to-04"] == {"co-16", "co-17"}


def test_expand_outcomes_includes_to_and_children():
    mapping = _build_to_to_cos(_basic_objectives())
    expanded = _expand_outcomes(["to-04"], mapping)
    assert expanded == {"to-04", "co-16", "co-17"}


def test_expand_outcomes_passes_co_through():
    mapping = _build_to_to_cos(_basic_objectives())
    expanded = _expand_outcomes(["co-16"], mapping)
    assert expanded == {"co-16"}


def test_expand_outcomes_handles_unknown_to():
    mapping = _build_to_to_cos(_basic_objectives())
    # to-99 has no children — only itself is returned.
    expanded = _expand_outcomes(["to-99"], mapping)
    assert expanded == {"to-99"}


# ---------------------------------------------------------------------- #
# query_chunks core behavior                                              #
# ---------------------------------------------------------------------- #


def test_query_chunks_unknown_slug(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    with pytest.raises(UnknownSlugError):
        query_chunks("no-such", QueryFilter(), courses_root=courses_root)


def test_query_chunks_missing_chunks_file_raises(tmp_path: Path):
    courses_root = tmp_path / "courses"
    (courses_root / "demo").mkdir(parents=True)
    with pytest.raises(MalformedArchiveError):
        query_chunks("demo", QueryFilter(), courses_root=courses_root)


def test_query_chunks_empty_filter_returns_all(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks("demo", QueryFilter(), courses_root=courses_root)
    assert result.total_matches == 5
    assert result.returned == 5


def test_query_chunks_chunk_type_filter(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(chunk_types=["exercise"]),
        courses_root=courses_root,
    )
    assert result.total_matches == 2
    assert {c["id"] for c in result.chunks} == {"c03", "c04"}


def test_query_chunks_multi_value_chunk_type(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(chunk_types=["exercise", "assessment_item"]),
        courses_root=courses_root,
    )
    assert result.total_matches == 3


def test_query_chunks_bloom_and_difficulty_compose(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(bloom_levels=["apply"], difficulties=["intermediate"]),
        courses_root=courses_root,
    )
    assert result.total_matches == 2
    assert {c["id"] for c in result.chunks} == {"c02", "c03"}


def test_query_chunks_week_single(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(week_min=7, week_max=7),
        courses_root=courses_root,
    )
    assert result.total_matches == 2
    assert all(
        c["source"]["module_id"].startswith("week_07") for c in result.chunks
    )


def test_query_chunks_week_range(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(week_min=1, week_max=3),
        courses_root=courses_root,
    )
    # Week 1 (c01) + week 3 (c02) = 2 chunks.
    assert result.total_matches == 2


def test_query_chunks_module_exact(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(modules=["week_07_application"]),
        courses_root=courses_root,
    )
    assert result.total_matches == 2


def test_query_chunks_outcome_rollup_for_to(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(outcomes=["to-04"]),
        courses_root=courses_root,
    )
    # to-04 itself (c05) + co-16 (c03) + co-17 (c04) = 3.
    assert result.total_matches == 3
    assert "co-16" in result.expanded_outcomes
    assert "to-04" in result.expanded_outcomes


def test_query_chunks_outcome_co_passes_through(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(outcomes=["co-16"]),
        courses_root=courses_root,
    )
    assert result.total_matches == 1
    assert result.chunks[0]["id"] == "c03"


def test_query_chunks_outcome_without_objectives_file(tmp_path: Path):
    """Missing objectives.json → TO query matches only direct TO refs."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks())  # No objectives.
    result = query_chunks(
        "demo",
        QueryFilter(outcomes=["to-04"]),
        courses_root=courses_root,
    )
    assert result.total_matches == 1  # Only c05 directly tagged with to-04.


def test_query_chunks_text_substring_case_insensitive(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(text_substring="SH:MINCOUNT"),
        courses_root=courses_root,
    )
    assert result.total_matches == 1
    assert result.chunks[0]["id"] == "c02"


def test_query_chunks_limit_and_offset(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(limit=2, offset=1),
        courses_root=courses_root,
    )
    assert result.total_matches == 5  # Total before slicing.
    assert result.returned == 2


def test_query_chunks_default_sort_week_asc(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks("demo", QueryFilter(), courses_root=courses_root)
    ids = [c["id"] for c in result.chunks]
    # Week order: c01(1), c02(3), c03(7), c04(7), c05(8). c03 before c04 by id.
    assert ids == ["c01", "c02", "c03", "c04", "c05"]


def test_query_chunks_sort_word_count(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(sort_key="word_count"),
        courses_root=courses_root,
    )
    word_counts = [c["word_count"] for c in result.chunks]
    assert word_counts == sorted(word_counts)


def test_query_chunks_sort_bloom(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    result = query_chunks(
        "demo",
        QueryFilter(sort_key="bloom"),
        courses_root=courses_root,
    )
    blooms = [c["bloom_level"] for c in result.chunks]
    # remember < apply < analyze < evaluate.
    indexed = [BLOOM_LEVELS.index(b) for b in blooms]
    assert indexed == sorted(indexed)


def test_query_chunks_invalid_sort_key_raises(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    with pytest.raises(ChunkQueryError):
        query_chunks(
            "demo",
            QueryFilter(sort_key="bogus"),
            courses_root=courses_root,
        )


def test_query_chunks_week_with_no_module_id(tmp_path: Path):
    """Chunks without a parseable week are excluded from week filters."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    chunks = _basic_chunks() + [
        {
            "id": "orphan",
            "chunk_type": "explanation",
            "difficulty": "foundational",
            "bloom_level": "remember",
            "text": "no module",
            "word_count": 10,
            "learning_outcome_refs": [],
            "source": {"module_id": "syllabus"},  # No week prefix.
        }
    ]
    _make_archive(courses_root, "demo", chunks, _basic_objectives())
    # With no week filter, orphan is included.
    base = query_chunks("demo", QueryFilter(), courses_root=courses_root)
    assert base.total_matches == 6
    # With a week filter, orphan is excluded.
    filtered = query_chunks(
        "demo",
        QueryFilter(week_min=1, week_max=12),
        courses_root=courses_root,
    )
    assert filtered.total_matches == 5
    assert "orphan" not in {c["id"] for c in filtered.chunks}


def test_query_chunks_full_compose(tmp_path: Path):
    """All filters AND together."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_archive(courses_root, "demo", _basic_chunks(), _basic_objectives())
    # week 7 + exercise + apply → only c03.
    result = query_chunks(
        "demo",
        QueryFilter(
            chunk_types=["exercise"],
            bloom_levels=["apply"],
            week_min=7,
            week_max=7,
        ),
        courses_root=courses_root,
    )
    assert result.total_matches == 1
    assert result.chunks[0]["id"] == "c03"
