"""Wave 69 — misconception ID seed includes bloom_level.

Pre-Wave-69 the misconception content-hash used
``sha256(statement|correction)``. That collapsed two misconceptions
with identical text but different Bloom cognitive demands (e.g., an
"apply"-level misread vs an "analyze"-level misread of the same
concept) to the same ID, losing the Wave 60 emit that distinguishes
them. Wave 69 extends the seed to
``sha256(statement|correction|bloom_level)``.

Covers both hash call sites (they must stay in lock-step so a chunk's
``misconceptions[*].bloom_level`` hashes the same whether the ID is
minted by ``process_course._build_misconceptions_for_graph`` or by
``preference_factory._misconception_id``):

* IDs differ when Bloom levels differ (same statement + correction)
* IDs are stable across calls (same inputs → same output)
* IDs match the schema pattern ``^mc_[0-9a-f]{16}$``
* bloom_level absence ("no Wave 60 emit") feeds an empty segment — the
  two legacy-corpus misconceptions with the same text collapse to the
  same ID (expected: that's the pre-Wave-60 behavior)
* The graph-side helper (``_build_misconceptions_for_graph``) picks up
  bloom_level from chunk misconception dicts
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.generators.preference_factory import _misconception_id  # noqa: E402
from Trainforge.process_course import CourseProcessor  # noqa: E402


_ID_PATTERN = re.compile(r"^mc_[0-9a-f]{16}$")


def _bare_processor() -> CourseProcessor:
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "TST_101"
    return proc


# ---------------------------------------------------------------------------
# _misconception_id helper
# ---------------------------------------------------------------------------


def test_bloom_level_param_changes_id():
    """Same text, different bloom_level → different ID."""
    a = _misconception_id("belief", "correction", "apply")
    b = _misconception_id("belief", "correction", "analyze")
    assert a != b
    assert _ID_PATTERN.match(a) and _ID_PATTERN.match(b)


def test_bloom_level_seed_is_stable_across_calls():
    """Same (statement, correction, bloom_level) always hashes the same."""
    a = _misconception_id("belief", "correction", "apply")
    b = _misconception_id("belief", "correction", "apply")
    assert a == b


def test_bloom_level_case_insensitive_normalization():
    """Upper / lower / mixed Bloom levels normalize to the same ID.

    The helper lowercases internally so camelCase input from JSON-LD
    (pre-normalization) and snake_case internal values hash identically.
    """
    lowered = _misconception_id("b", "c", "apply")
    upper = _misconception_id("b", "c", "APPLY")
    mixed = _misconception_id("b", "c", "Apply")
    assert lowered == upper == mixed


def test_bloom_level_whitespace_stripped():
    """Leading/trailing whitespace on bloom_level is normalized out."""
    base = _misconception_id("b", "c", "apply")
    padded = _misconception_id("b", "c", " apply ")
    assert base == padded


def test_missing_bloom_level_matches_empty_string():
    """None, missing 3rd arg, and empty string all collapse to the same ID.

    This defines the "legacy corpus" path — pre-Wave-60 misconceptions
    without a bloom_level field produce IDs stable with each other (but
    different from Wave-60+ IDs even if text matches, per wave-break).
    """
    none_arg = _misconception_id("b", "c", None)
    empty = _misconception_id("b", "c", "")
    default = _misconception_id("b", "c")
    assert none_arg == empty == default


def test_bloom_level_absent_differs_from_bloom_level_present():
    """Pre-Wave-69 IDs (no bloom seed) differ from Wave-69 IDs with bloom.

    Confirms the documented breaking change: rebuilding a corpus under
    Wave 69 will re-hash any misconception that newly carries a
    bloom_level. Corpora with no Wave-60 emit stay stable.
    """
    legacy = _misconception_id("b", "c")
    wave69 = _misconception_id("b", "c", "apply")
    assert legacy != wave69


def test_bloom_less_path_matches_pre_wave_69_two_field_seed():
    """Wave 72 regression: bloom-less path hashes a 2-field seed.

    Pre-Wave-72 the seed was always ``{statement}|{correction}|{bloom_level}``
    — when ``bloom_level`` was empty the trailing pipe still contributed to
    the hash, rekeying every legacy / pre-Wave-60 misconception. Wave 72
    switches to a 2-field seed when no bloom is supplied so legacy corpora
    retain their pre-Wave-69 IDs. Asserted directly against the canonical
    2-field sha256 so the seed shape can't silently drift again.

    Whitespace-only bloom (``" "``, ``"\\t"``) must also take the 2-field
    path — ``.strip()`` inside the helper collapses them to empty before
    the branch, so a cosmetically-stripped emit cannot rekey legacy IDs.
    """
    import hashlib

    expected = "mc_" + hashlib.sha256(b"b|c").hexdigest()[:16]
    assert _misconception_id("b", "c") == expected
    assert _misconception_id("b", "c", None) == expected
    assert _misconception_id("b", "c", "") == expected
    assert _misconception_id("b", "c", " ") == expected
    assert _misconception_id("b", "c", "\t") == expected
    assert _misconception_id("b", "c", "   ") == expected


# ---------------------------------------------------------------------------
# process_course._build_misconceptions_for_graph picks up bloom_level
# ---------------------------------------------------------------------------


def _fixture_chunk(bloom_level: str | None) -> dict:
    """Chunk with one misconception; bloom_level optional.

    Represents a chunk whose ``misconceptions[]`` came through the
    ``html_content_parser`` normalizer (so keys are already snake_case
    and Bloom is lowercased).
    """
    mc = {
        "misconception": "Students believe X is always Y.",
        "correction": "X is Z except when W holds.",
    }
    if bloom_level is not None:
        mc["bloom_level"] = bloom_level
    return {
        "id": "chunk_01",
        "concept_tags": ["my-concept"],
        "misconceptions": [mc],
    }


def test_graph_helper_emits_distinct_ids_for_bloom_distinct_misconceptions():
    """Two misconceptions with identical text but different Bloom levels
    produce distinct IDs via the graph-build path."""
    proc = _bare_processor()
    # Two chunks, each carrying the same statement + correction but
    # different Bloom level. Pre-Wave-69 they collapse; Wave-69+ they
    # stay distinct.
    chunks = [
        _fixture_chunk("apply"),
        _fixture_chunk("analyze"),
    ]
    # Rename the second chunk so the dedup-by-ID set doesn't drop it
    # before we can inspect both.
    chunks[1] = dict(chunks[1])
    chunks[1]["id"] = "chunk_02"

    entities = proc._build_misconceptions_for_graph(chunks)
    assert len(entities) == 2, [e["id"] for e in entities]
    ids = {e["id"] for e in entities}
    assert len(ids) == 2

    # bloom_level propagates onto the entity.
    by_bloom = {e["bloom_level"]: e for e in entities}
    assert set(by_bloom) == {"apply", "analyze"}
    # Content-hash id shape preserved.
    for e in entities:
        assert _ID_PATTERN.match(e["id"])


def test_graph_helper_collapses_bloom_less_duplicates():
    """Two misconceptions with identical text AND no bloom_level (legacy)
    still collapse to the same ID. Guards against seed drift in the
    legacy path."""
    proc = _bare_processor()
    chunks = [
        _fixture_chunk(None),
        _fixture_chunk(None),
    ]
    chunks[1] = dict(chunks[1])
    chunks[1]["id"] = "chunk_02"
    entities = proc._build_misconceptions_for_graph(chunks)
    # Same text, same (empty) bloom → one dedup.
    assert len(entities) == 1
    e = entities[0]
    # bloom_level field elided when absent.
    assert "bloom_level" not in e


def test_graph_helper_and_preference_factory_agree_on_id():
    """The two hash call sites must produce the same ID for the same
    (statement, correction, bloom_level). Guards against seed drift
    between ``_build_misconceptions_for_graph`` and
    ``preference_factory._misconception_id``."""
    proc = _bare_processor()
    chunk = _fixture_chunk("apply")
    (entity,) = proc._build_misconceptions_for_graph([chunk])
    direct_id = _misconception_id(
        "Students believe X is always Y.",
        "X is Z except when W holds.",
        "apply",
    )
    assert entity["id"] == direct_id


def test_graph_helper_lowercases_mixed_case_bloom_for_lockstep():
    """Wave 72 regression: graph-side normalization matches preference_factory.

    ``preference_factory._misconception_id`` applies ``.strip().lower()`` to
    ``bloom_level`` before the seed branch. The graph-side call site
    (``_build_misconceptions_for_graph``) must do the same, otherwise a
    chunk constructed directly (bypassing the html_content_parser
    normalizer that canonicalizes Bloom to lowercase) with ``"Apply"`` or
    ``"APPLY"`` or ``" apply "`` would hash to a different ID on the
    graph side than on the preference_factory side — silently breaking
    cross-call-site identity for the same conceptual misconception.
    """
    proc = _bare_processor()
    canonical_id = _misconception_id(
        "Students believe X is always Y.",
        "X is Z except when W holds.",
        "apply",
    )
    for raw_bloom in ("apply", "Apply", "APPLY", " apply ", "\tApply\n"):
        (entity,) = proc._build_misconceptions_for_graph(
            [_fixture_chunk(raw_bloom)]
        )
        assert entity["id"] == canonical_id, raw_bloom
