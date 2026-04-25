"""Wave 76 — chunk module_id + bloom_level + chunks.json/.jsonl parity tests.

Exercises the three small data-hygiene fixes added to
``Trainforge/process_course.py`` in Wave 76:

  * ``canonicalize_bloom_level`` — split compound values
    (``"remember-apply"``) into primary (HIGHER) + secondary (LOWER).
  * ``normalize_module_id`` — lift short slugs (``"application"``,
    ``"summary"``) to canonical ``"week_NN_<slot>"`` form when the
    week number is recoverable from the chunk's ``item_path`` /
    ``week_num``.
  * ``_assert_chunk_files_parity`` — the round-trip check that
    chunks.json contains the exact same chunk list as chunks.jsonl.

The rdf-shacl-550 corpus surfaced concrete examples of each gap;
these tests guard against regressions.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import (  # noqa: E402  (after sys.path tweak)
    _assert_chunk_files_parity,
    canonicalize_bloom_level,
    normalize_module_id,
)


# ---------------------------------------------------------------------------
# canonicalize_bloom_level
# ---------------------------------------------------------------------------


def test_bloom_compound_remember_apply():
    primary, secondary = canonicalize_bloom_level("remember-apply")
    assert primary == "apply"
    assert secondary == "remember"


def test_bloom_compound_understand_analyze():
    primary, secondary = canonicalize_bloom_level("understand-analyze")
    assert primary == "analyze"
    assert secondary == "understand"


def test_bloom_compound_apply_analyze():
    primary, secondary = canonicalize_bloom_level("apply-analyze")
    assert primary == "analyze"
    assert secondary == "apply"


def test_bloom_compound_analyze_evaluate():
    primary, secondary = canonicalize_bloom_level("analyze-evaluate")
    assert primary == "evaluate"
    assert secondary == "analyze"


def test_bloom_single_apply_passthrough():
    primary, secondary = canonicalize_bloom_level("apply")
    assert primary == "apply"
    assert secondary is None


def test_bloom_single_create_passthrough():
    primary, secondary = canonicalize_bloom_level("create")
    assert primary == "create"
    assert secondary is None


def test_bloom_invalid_value_does_not_crash(caplog):
    """Unknown values pass through; caller decides whether to keep / drop."""
    primary, secondary = canonicalize_bloom_level("synthesize")
    assert primary == "synthesize"
    assert secondary is None


def test_bloom_empty_returns_none():
    assert canonicalize_bloom_level("") == (None, None)
    assert canonicalize_bloom_level(None) == (None, None)
    assert canonicalize_bloom_level("   ") == (None, None)


def test_bloom_compound_case_insensitive():
    primary, secondary = canonicalize_bloom_level("Remember-Apply")
    assert primary == "apply"
    assert secondary == "remember"


def test_bloom_three_part_compound():
    """Three-part compound — keep highest as primary, lowest as secondary."""
    primary, secondary = canonicalize_bloom_level("remember-apply-evaluate")
    assert primary == "evaluate"
    assert secondary == "remember"


# ---------------------------------------------------------------------------
# normalize_module_id
# ---------------------------------------------------------------------------


def test_module_id_short_slug_with_week_num():
    new_mid, changed = normalize_module_id("application", week_num=3)
    assert new_mid == "week_03_application"
    assert changed is True


def test_module_id_short_slug_with_item_path():
    new_mid, changed = normalize_module_id(
        "application", item_path="week_04/application.html"
    )
    assert new_mid == "week_04_application"
    assert changed is True


def test_module_id_short_slug_with_zero_padding():
    new_mid, changed = normalize_module_id("summary", week_num=12)
    assert new_mid == "week_12_summary"
    assert changed is True


def test_module_id_already_prefixed_unchanged():
    new_mid, changed = normalize_module_id(
        "week_01_overview", item_path="week_01/week_01_overview.html"
    )
    assert new_mid == "week_01_overview"
    assert changed is False


def test_module_id_no_week_info_returns_unchanged(caplog):
    """No item_path + no week_num + no prefix → keep as-is, don't drop."""
    new_mid, changed = normalize_module_id("summary")
    assert new_mid == "summary"
    assert changed is False


def test_module_id_empty_returns_unchanged():
    new_mid, changed = normalize_module_id("", week_num=3)
    assert new_mid == ""
    assert changed is False
    new_mid, changed = normalize_module_id(None, week_num=3)
    assert new_mid is None
    assert changed is False


def test_module_id_week_num_zero_or_negative_no_normalize():
    """Bogus week numbers must not produce ``week_00_<slot>``."""
    new_mid, changed = normalize_module_id("application", week_num=0)
    assert new_mid == "application"
    assert changed is False
    new_mid, changed = normalize_module_id("application", week_num=-1)
    assert new_mid == "application"
    assert changed is False


def test_module_id_item_path_takes_precedence_when_week_num_missing():
    new_mid, changed = normalize_module_id(
        "content_05", item_path="week_07/content_05.html"
    )
    assert new_mid == "week_07_content_05"
    assert changed is True


# ---------------------------------------------------------------------------
# chunks.json / chunks.jsonl round-trip parity
# ---------------------------------------------------------------------------


def _write_pair(tmp_path: Path, chunks):
    jsonl_path = tmp_path / "chunks.jsonl"
    json_path = tmp_path / "chunks.json"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, indent=2, ensure_ascii=False)
    return jsonl_path, json_path


def test_parity_check_passes_on_matched_files(tmp_path):
    chunks = [
        {"id": "a_chunk_00001", "text": "first"},
        {"id": "a_chunk_00002", "text": "second"},
    ]
    jsonl_path, json_path = _write_pair(tmp_path, chunks)
    _assert_chunk_files_parity(jsonl_path, json_path)


def test_parity_check_fails_on_line_count_mismatch(tmp_path):
    chunks = [{"id": "x", "text": "ok"}]
    jsonl_path, json_path = _write_pair(tmp_path, chunks)
    # Inject an extra line into the .jsonl only.
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "y", "text": "extra"}) + "\n")
    with pytest.raises(RuntimeError, match="line count mismatch"):
        _assert_chunk_files_parity(jsonl_path, json_path)


def test_parity_check_fails_on_content_drift(tmp_path):
    chunks = [
        {"id": "a", "text": "one"},
        {"id": "b", "text": "two"},
    ]
    jsonl_path, json_path = _write_pair(tmp_path, chunks)
    # Mutate only the .json file's second chunk so contents diverge.
    with json_path.open("r", encoding="utf-8") as fh:
        json_chunks = json.load(fh)
    json_chunks[1]["text"] = "DRIFTED"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(json_chunks, fh, indent=2, ensure_ascii=False)
    with pytest.raises(RuntimeError, match="chunk index 1 differs"):
        _assert_chunk_files_parity(jsonl_path, json_path)


def test_parity_check_fails_on_non_list_json(tmp_path):
    chunks = [{"id": "a", "text": "ok"}]
    jsonl_path, json_path = _write_pair(tmp_path, chunks)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"chunks": chunks}, fh)  # wrong shape
    with pytest.raises(RuntimeError, match="expected top-level list"):
        _assert_chunk_files_parity(jsonl_path, json_path)


def test_parity_check_round_trip_after_normalization(tmp_path):
    """End-to-end: write a normalized list, read back from both files,
    confirm equality. Mirrors the in-pipeline assertion in _write_chunks.
    """
    chunks = []
    for idx in range(50):
        chunks.append({
            "id": f"sample_chunk_{idx:05d}",
            "schema_version": "v4",
            "bloom_level": "apply" if idx % 2 == 0 else "analyze",
            "source": {
                "course_id": "TEST_101",
                "module_id": f"week_{(idx // 10) + 1:02d}_content_{(idx % 10) + 1:02d}",
                "lesson_id": f"lesson_{idx}",
            },
            "text": f"chunk text {idx}",
        })
    jsonl_path, json_path = _write_pair(tmp_path, chunks)
    _assert_chunk_files_parity(jsonl_path, json_path)
    # Round-trip read both back independently and compare.
    jl = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                jl.append(json.loads(line))
    with json_path.open("r", encoding="utf-8") as fh:
        js = json.load(fh)
    assert jl == js == chunks
