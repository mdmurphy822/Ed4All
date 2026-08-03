"""Unit tests for the lossless split/merge-aware global aligner.

Covers the Phase-0 contract of ``data/alignment/structure_align.py``:

  * exact 1:1 MATCH
  * SPLIT (1 pdf -> 2 gold) recovers BOTH gold roles
  * MERGE (2 pdf -> 1 gold)
  * GOLD_GAP / PDF_GAP accounting
  * the never-drop invariant (assert_complete passes on a real result; a
    deliberately-broken ledger fails closed)
  * confidence is higher for clean matches than for low-sim ones
  * role-space projection maps to the active vocab + records exclusions
  * segmentation does NOT merge across pages
  * regression: the new aligner recovers what the old greedy 0.30-Jaccard
    loop matched PLUS at least one block the greedy version would have dropped

No GPU, no model, no IO — pure data structures + numpy. This file inlines the
SemantiK-root sys.path bootstrap (mirroring ``semantik_structure/tests/conftest.py``)
so ``from data.alignment.structure_align import ...`` resolves when pytest collects it
directly, without adding a conftest.py (Phase-0 ships exactly two new files).
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../SemantiK/data/tests/test_structure_align.py
#   parents[0]=tests parents[1]=data parents[2]=SemantiK
_SEMANTIK_ROOT = Path(__file__).resolve().parents[2]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))

import pytest  # noqa: E402

from data.alignment.structure_align import (  # noqa: E402
    SCHEMA_VERSION,
    AlignLedger,
    FBView,
    GoldView,
    LedgerIncompleteError,
    align_blocks,
    project_rows,
    sim,
)
from semantik_structure.text_utils import jaccard_overlap  # noqa: E402

# The active head vocabulary the trainer uses (mirror of
# build_structure_data.ROLE_NAMES — kept local so the test is independent of
# the builder's heavy import chain).
ACTIVE_VOCAB = (
    "paragraph",
    "heading",
    "list_item",
    "form_label",
    "blockquote",
    "code_block",
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _pdf(text: str, idx: int, *, page: int = 1) -> FBView:
    # A trivially distinct bbox per index keeps provenance inspectable.
    return FBView(
        text=text,
        bbox=(0.0, float(idx) * 10.0, 100.0, float(idx) * 10.0 + 8.0),
        page=page,
        order_index=idx,
        layout=[float(idx)] * 20,
    )


def _gold(text: str, role: str, idx: int, *, tag: str | None = None) -> GoldView:
    if tag is None:
        tag = {"heading": "h2", "paragraph": "p", "list_item": "li"}.get(role, "p")
    return GoldView(text=text, tag=tag, role=role, gold_index=idx)


def _kinds(result) -> list[str]:
    return [r.kind for r in result.rows]


# ---------------------------------------------------------------------------
# sim()
# ---------------------------------------------------------------------------


def test_sim_ordering_and_bounds():
    assert sim("alpha beta gamma", "alpha beta gamma") == pytest.approx(1.0)
    partial = sim("alpha beta gamma", "alpha beta delta")
    disjoint = sim("alpha beta gamma", "xxx yyy zzz")
    assert disjoint == 0.0
    assert 0.0 < partial < 1.0
    assert sim("", "anything") == 0.0
    # Directional containment lifts a subset pair above its bare Jaccard.
    assert sim("alpha beta", "alpha beta gamma delta") > jaccard_overlap(
        "alpha beta", "alpha beta gamma delta"
    )


# ---------------------------------------------------------------------------
# Core moves
# ---------------------------------------------------------------------------


def test_exact_one_to_one_match():
    pdf = [_pdf("the quick brown fox", 0)]
    gold = [_gold("the quick brown fox", "paragraph", 0)]
    res = align_blocks(pdf, gold)
    assert _kinds(res) == ["matched"]
    row = res.rows[0]
    assert row.role == "paragraph"
    assert row.text == "the quick brown fox"
    assert row.confidence == pytest.approx(1.0)
    assert row.align["gold_indices"] == [0]
    assert row.provenance["pdf_block_indices"] == [0]
    assert row.schema_version == SCHEMA_VERSION
    res.ledger.assert_complete(1, 1)
    assert res.ledger.counts["matched"] == 1


def test_split_one_pdf_into_two_gold():
    # One PDF block whose text is the concatenation of two gold elements with
    # DIFFERENT roles — the greedy loop can recover at most one.
    pdf = [_pdf("the quick brown fox jumps over the lazy dog", 0)]
    gold = [
        _gold("the quick brown fox", "heading", 0),
        _gold("jumps over the lazy dog", "paragraph", 1),
    ]
    res = align_blocks(pdf, gold)
    assert _kinds(res) == ["split", "split"]
    roles = {r.role for r in res.rows}
    assert roles == {"heading", "paragraph"}
    # Both rows share the single source pdf block + the full gold-index set.
    for r in res.rows:
        assert r.provenance["pdf_block_indices"] == [0]
        assert r.align["gold_indices"] == [0, 1]
        assert r.kind == "split"
    # Each split row carries the recovered gold member's own text.
    texts = {r.text for r in res.rows}
    assert texts == {"the quick brown fox", "jumps over the lazy dog"}
    res.ledger.assert_complete(1, 2)
    assert res.ledger.counts["split"] == 1


def test_merge_two_pdf_into_one_gold():
    pdf = [_pdf("alpha beta", 0), _pdf("gamma delta", 1)]
    gold = [_gold("alpha beta gamma delta", "paragraph", 0)]
    res = align_blocks(pdf, gold)
    assert _kinds(res) == ["merge"]
    row = res.rows[0]
    assert row.role == "paragraph"
    assert row.text == "alpha beta gamma delta"
    assert row.provenance["pdf_block_indices"] == [0, 1]
    assert row.align["gold_indices"] == [0]
    res.ledger.assert_complete(2, 1)
    assert res.ledger.counts["merge"] == 1


def test_gold_gap_recorded_not_dropped():
    pdf = [_pdf("hello world", 0)]
    gold = [
        _gold("hello world", "paragraph", 0),
        _gold("entirely unrelated orphan gold", "heading", 1),
    ]
    res = align_blocks(pdf, gold)
    assert res.ledger.counts["matched"] == 1
    assert res.ledger.counts["gold_gap"] == 1
    # The orphan gold yields NO training row but IS accounted.
    assert res.ledger.gold_buckets[1] == "gold_gap"
    res.ledger.assert_complete(1, 2)


def test_pdf_gap_recorded_not_dropped():
    pdf = [_pdf("match this text", 0), _pdf("orphan unrelated content", 1)]
    gold = [_gold("match this text", "paragraph", 0)]
    res = align_blocks(pdf, gold)
    assert res.ledger.counts["matched"] == 1
    assert res.ledger.counts["pdf_gap"] == 1
    assert res.ledger.pdf_buckets[1] == "pdf_gap"
    res.ledger.assert_complete(2, 1)


# ---------------------------------------------------------------------------
# Never-drop invariant
# ---------------------------------------------------------------------------


def test_assert_complete_passes_on_real_result():
    pdf = [_pdf("alpha beta", 0), _pdf("gamma", 1), _pdf("noise here", 2)]
    gold = [
        _gold("alpha beta", "paragraph", 0),
        _gold("gamma", "heading", 1),
        _gold("lonely gold element", "paragraph", 2),
    ]
    res = align_blocks(pdf, gold)
    # Every pdf position and every gold position landed in exactly one bucket.
    assert set(res.ledger.pdf_buckets) == {0, 1, 2}
    assert set(res.ledger.gold_buckets) == {0, 1, 2}
    res.ledger.assert_complete(3, 3)  # must not raise


def test_assert_complete_fails_closed_on_broken_ledger():
    led = AlignLedger()
    led.record_move("matched", pdf_positions=[0], gold_positions=[0])
    # pdf position 1 deliberately unaccounted.
    with pytest.raises(LedgerIncompleteError):
        led.assert_complete(2, 1)


def test_double_record_fails_closed():
    led = AlignLedger()
    led.record_move("matched", pdf_positions=[0], gold_positions=[0])
    with pytest.raises(LedgerIncompleteError):
        led.record_move("pdf_gap", pdf_positions=[0])


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_higher_for_clean_than_low_sim_rows():
    # A clean 1:1 match next to a weaker (but still above-floor) match: the
    # composite confidence ranks the clean match strictly higher.
    pdf = [
        _pdf("apple apple apple banana", 0),
        _pdf("apple banana cherry date", 1),
    ]
    gold = [
        _gold("apple apple apple banana", "paragraph", 0),
        _gold("apple banana cherry fig", "paragraph", 1),
    ]
    res = align_blocks(pdf, gold)
    assert _kinds(res) == ["matched", "matched"]
    clean, weaker = res.rows[0], res.rows[1]
    assert clean.confidence == pytest.approx(1.0)
    assert clean.confidence > weaker.confidence
    assert weaker.confidence > 0.30  # still a confident-enough match


def test_low_sim_match_is_flagged_in_ledger():
    # A below-floor association the global structure still selects (here a
    # low-confidence SPLIT, where the gap alternative for 3 positions is more
    # expensive than the split) is EMITTED but flagged low_sim — never
    # silently dropped, never silently passed off as confident.
    pdf = [_pdf("alpha", 0)]
    gold = [
        _gold("alpha extra one two three", "paragraph", 0),
        _gold("six seven eight nine ten", "heading", 1),
    ]
    res = align_blocks(pdf, gold)
    assert _kinds(res) == ["split", "split"]
    assert all(r.confidence < 0.30 for r in res.rows)
    assert all(r.align["low_sim"] is True for r in res.rows)
    assert res.ledger.counts["low_sim"] == 1
    res.ledger.assert_complete(1, 2)


# ---------------------------------------------------------------------------
# Role-space projection
# ---------------------------------------------------------------------------


def test_project_rows_maps_active_and_records_excluded():
    pdf = [
        _pdf("a real paragraph here", 0),
        _pdf("a figure caption text", 1),
    ]
    gold = [
        _gold("a real paragraph here", "paragraph", 0),
        # figure_caption is in the FULL role space but NOT the active head.
        _gold("a figure caption text", "figure_caption", 1, tag="figcaption"),
    ]
    res = align_blocks(pdf, gold)
    assert {r.role for r in res.rows} == {"paragraph", "figure_caption"}

    kept = project_rows(res.rows, ACTIVE_VOCAB, ledger=res.ledger)
    assert [r.role for r in kept] == ["paragraph"]
    assert kept[0].labels["structural_role"] == ACTIVE_VOCAB.index("paragraph")
    assert kept[0].align["projected"] is True
    # The excluded role is RECORDED, not silently dropped.
    assert res.ledger.counts["role_filtered"] == 1
    assert res.ledger.excluded_roles["figure_caption"] == 1
    assert res.ledger.projected_roles["paragraph"] == 1


def test_projected_row_is_trainer_readable():
    # The v3 row must stay backward-readable by train_structure.load_split,
    # which requires labels.structural_role (int) + layout + text.
    pdf = [_pdf("a heading line", 0)]
    gold = [_gold("a heading line", "heading", 0, tag="h2")]
    res = align_blocks(pdf, gold)
    kept = project_rows(res.rows, ACTIVE_VOCAB)
    row = kept[0].to_row()
    assert "structural_role" in row["labels"]
    assert isinstance(row["labels"]["structural_role"], int)
    assert len(row["layout"]) == 20
    assert row["text"] == "a heading line"
    assert row["schema_version"] == SCHEMA_VERSION
    # Additive blocks are present and JSON-serializable.
    import json

    json.dumps(row)
    assert "provenance" in row and "align" in row


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_segmentation_does_not_merge_across_pages():
    # Two pages, each with a within-page 2->1 MERGE. Segmentation must keep
    # each merge's source blocks on a SINGLE page (never fuse page1+page2).
    pdf = [
        _pdf("apple red", 0, page=1),
        _pdf("very sweet", 1, page=1),
        _pdf("banana yellow", 2, page=2),
        _pdf("also sweet", 3, page=2),
    ]
    gold = [
        _gold("apple red very sweet", "paragraph", 0),
        _gold("banana yellow also sweet", "paragraph", 1),
    ]
    res = align_blocks(pdf, gold)
    merges = [r for r in res.rows if r.kind == "merge"]
    assert len(merges) == 2
    for r in merges:
        pages = {pdf[i].page for i in r.provenance["pdf_block_indices"]}
        assert len(pages) == 1, "a merge spanned two pages"
    res.ledger.assert_complete(4, 2)

    # Contrast: the SAME four blocks on ONE page + one combined gold DO fuse
    # into a single cross-block merge — proving the page boundary (not the
    # data) is what prevented the cross-page merge above.
    pdf_one_page = [
        _pdf("apple red", 0, page=1),
        _pdf("very sweet", 1, page=1),
        _pdf("banana yellow", 2, page=1),
        _pdf("also sweet", 3, page=1),
    ]
    gold_one = [_gold("apple red very sweet banana yellow also sweet", "paragraph", 0)]
    res2 = align_blocks(pdf_one_page, gold_one)
    big = [r for r in res2.rows if r.kind == "merge"]
    assert len(big) == 1
    assert res2.rows[0].provenance["pdf_block_indices"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Regression vs the old greedy 0.30-Jaccard loop
# ---------------------------------------------------------------------------


def _greedy_align(pdf: list[FBView], gold: list[GoldView]) -> set[int]:
    """Faithful re-implementation of the old greedy loop
    (build_structure_data.process_pair ~:576-593): monotonic cursor, sliding
    window=8, 0.30 Jaccard floor. Returns the set of MATCHED gold indices."""
    matched: set[int] = set()
    cursor = 0
    window = 8
    for block in pdf:
        text = block.text.strip()
        if not text:
            continue
        best_idx, best_score = -1, 0.0
        for j in range(max(0, cursor - 2), min(len(gold), cursor + window)):
            s = jaccard_overlap(text, gold[j].text)
            if s > best_score:
                best_score, best_idx = s, j
        if best_idx < 0 or best_score < 0.30:
            continue
        matched.add(best_idx)
        cursor = max(cursor, best_idx + 1)
    return matched


def test_recovers_greedy_matches_plus_a_dropped_block():
    # One PDF block fuses two gold elements. The greedy loop matches the block
    # to ONE gold and advances its cursor, DROPPING the second gold's
    # supervision. The DP SPLITs and recovers both.
    pdf = [_pdf("alpha beta gamma delta", 0)]
    gold = [
        _gold("alpha beta", "heading", 0),
        _gold("gamma delta", "paragraph", 1),
    ]

    greedy_matched = _greedy_align(pdf, gold)
    assert len(greedy_matched) == 1  # greedy recovers exactly one gold

    res = align_blocks(pdf, gold)
    dp_covered = {
        gi for r in res.rows for gi in r.align["gold_indices"]
    }
    # Recovers everything the greedy matched ...
    assert greedy_matched <= dp_covered
    # ... PLUS at least one gold the greedy version dropped.
    assert len(dp_covered) > len(greedy_matched)
    assert dp_covered == {0, 1}
    res.ledger.assert_complete(1, 2)
