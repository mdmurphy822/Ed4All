"""Wave 5 (W5.F) — chunker merge-path provenance preservation tests.

Pre-W5.F, ``merge_small_sections`` carried only ``source_references``,
``template_type``, and merged headings across the buffer when adjacent
sections collapsed into a single chunk. The W5.A parser-side carry
landed ``key_claims`` (W1.5 per-claim attribution) and
``objective_alignment`` (W1.7 per-objective tri-axis audit) on
``ContentSection``, but every merge boundary silently dropped both
fields — ~30-50% of chunks on small-section corpora (the RDF/SHACL
calibration corpus in particular) lost the audit signal.

W5.F extends the merger:

* ``MergedSectionResult`` dataclass replaces the legacy 5-tuple return.
  ``__iter__`` yields exactly the legacy 5 elements so every existing
  destructure site keeps working byte-identical to pre-W5.F.
* ``merged_key_claims`` — concatenation of every section's
  ``key_claims`` array (per-claim entries are independent so concat
  is safe).
* ``merged_objective_alignment`` — union deduped by ``objective_id``
  with first-seen-wins (mirrors ``merge_section_source_ids``).
* ``chunk_text_block`` accepts the new kwargs and forwards them to
  the ``ctx.create_chunk`` callback so ``_create_chunk`` (W5.B's
  stamp site) can union them with the section_metadata_extras
  contribution and emit the merged audit fields on the final chunk.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Trainforge.chunker.chunker import (  # noqa: E402
    ChunkerContext,
    MergedSectionResult,
    chunk_text_block,
    merge_small_sections,
)
from Trainforge.parsers.html_content_parser import ContentSection  # noqa: E402


# --------------------------------------------------------------------- #
# Section helpers
# --------------------------------------------------------------------- #


def _section(
    heading: str,
    content: str,
    *,
    key_claims: List[Dict[str, Any]] = None,
    objective_alignment: List[Dict[str, Any]] = None,
    source_references: List[str] = None,
    template_type: str = None,
) -> ContentSection:
    """Build a ContentSection with optional W5.A audit fields."""
    return ContentSection(
        heading=heading,
        level=2,
        content=content,
        word_count=len(content.split()),
        source_references=list(source_references or []),
        template_type=template_type,
        key_claims=list(key_claims or []),
        objective_alignment=list(objective_alignment or []),
    )


# --------------------------------------------------------------------- #
# MergedSectionResult dataclass contract
# --------------------------------------------------------------------- #


class TestMergedSectionResultBackCompat:
    """Pin the legacy 5-element destructure contract."""

    def test_dataclass_iter_yields_5_elements(self):
        result = MergedSectionResult(
            heading="H",
            combined_text="text",
            chunk_type="explanation",
            merged_source_ids=["dart:a"],
            merged_headings=["H"],
            merged_key_claims=[{"claim": "x"}],
            merged_objective_alignment=[{"objective_id": "TO-01"}],
        )
        # Tuple-unpack contract: exactly 5 elements.
        elements = list(result)
        assert len(elements) == 5
        assert elements[0] == "H"
        assert elements[1] == "text"
        assert elements[2] == "explanation"
        assert elements[3] == ["dart:a"]
        assert elements[4] == ["H"]

    def test_legacy_destructure_5_tuple_works(self):
        result = MergedSectionResult(
            heading="H",
            combined_text="text",
            chunk_type="explanation",
            merged_source_ids=[],
            merged_headings=[],
        )
        # Existing call sites do `heading, text, ctype, sids, mhs = result`.
        heading, text, ctype, sids, mhs = result
        assert heading == "H"
        assert text == "text"
        assert ctype == "explanation"
        assert sids == []
        assert mhs == []

    def test_indexing_returns_legacy_position(self):
        result = MergedSectionResult(
            heading="H",
            combined_text="text",
            chunk_type="explanation",
            merged_source_ids=["dart:a"],
            merged_headings=["H"],
            merged_key_claims=[{"claim": "x"}],
        )
        assert result[3] == ["dart:a"]
        assert result[4] == ["H"]

    def test_len_returns_5(self):
        result = MergedSectionResult(
            heading="H",
            combined_text="t",
            chunk_type="explanation",
        )
        assert len(result) == 5


# --------------------------------------------------------------------- #
# merge_small_sections — W5.F union behavior
# --------------------------------------------------------------------- #


class TestMergeSmallSectionsAuditFields:
    def test_unions_key_claims_across_sections(self):
        """Two sections with different key_claims → merged result
        carries the concat (per-claim entries are independent)."""
        sections = [
            _section(
                "Sec A",
                "alpha alpha alpha",
                key_claims=[
                    {"claim": "X is Y", "source_chunk_ids": ["dart:a#1"]},
                    {"claim": "P is Q", "source_chunk_ids": ["dart:a#2"]},
                ],
            ),
            _section(
                "Sec B",
                "beta beta beta",
                key_claims=[
                    {"claim": "Z is W", "source_chunk_ids": ["dart:b#1"]},
                ],
            ),
        ]
        merged = merge_small_sections(sections, max_chunk_size=1000)
        assert len(merged) == 1
        result = merged[0]
        assert isinstance(result, MergedSectionResult)
        # All three claims survive concat; first-section's pair anchors
        # at indices 0/1, second-section's claim at index 2.
        assert len(result.merged_key_claims) == 3
        claims = [entry["claim"] for entry in result.merged_key_claims]
        assert claims == ["X is Y", "P is Q", "Z is W"]
        # Source-chunk attribution preserved per-entry.
        assert result.merged_key_claims[0]["source_chunk_ids"] == ["dart:a#1"]
        assert result.merged_key_claims[2]["source_chunk_ids"] == ["dart:b#1"]

    def test_dedupes_objective_alignment_first_seen_wins(self):
        """Two sections with overlapping objective_alignment (same
        objective_id, different status) → merged result's entry is
        the first-seen one (mirrors merge_section_source_ids)."""
        sections = [
            _section(
                "Sec A",
                "alpha alpha alpha",
                objective_alignment=[
                    {
                        "objective_id": "TO-05",
                        "status": "delivered",
                        "declared_bloom": "apply",
                    },
                    {
                        "objective_id": "TO-06",
                        "status": "delivered",
                        "declared_bloom": "understand",
                    },
                ],
            ),
            _section(
                "Sec B",
                "beta beta beta",
                objective_alignment=[
                    # Same objective_id as Sec A's first entry — should
                    # be DEDUPED (first-seen wins).
                    {
                        "objective_id": "TO-05",
                        "status": "missing",
                        "declared_bloom": "remember",
                    },
                    # New objective — should be appended.
                    {
                        "objective_id": "TO-07",
                        "status": "delivered",
                        "declared_bloom": "analyze",
                    },
                ],
            ),
        ]
        merged = merge_small_sections(sections, max_chunk_size=1000)
        assert len(merged) == 1
        result = merged[0]
        # Three distinct objective_ids: TO-05 (first-seen), TO-06, TO-07.
        oids = [entry["objective_id"] for entry in result.merged_objective_alignment]
        assert oids == ["TO-05", "TO-06", "TO-07"]
        # First-seen-wins: TO-05's entry MUST carry Sec A's status,
        # NOT Sec B's overriding status.
        to5_entry = next(
            e for e in result.merged_objective_alignment
            if e["objective_id"] == "TO-05"
        )
        assert to5_entry["status"] == "delivered"
        assert to5_entry["declared_bloom"] == "apply"

    def test_back_compat_no_audit_fields(self):
        """A legacy ContentSection without key_claims /
        objective_alignment → merged result's new positions are []
        (empty list), NOT None."""
        sections = [
            _section("Sec A", "alpha alpha alpha"),
            _section("Sec B", "beta beta beta"),
        ]
        merged = merge_small_sections(sections, max_chunk_size=1000)
        assert len(merged) == 1
        result = merged[0]
        # Defensive default: empty list, never None. Downstream stamp
        # sites can no-op on `if merged.merged_key_claims:` without a
        # None-check guard.
        assert result.merged_key_claims == []
        assert result.merged_objective_alignment == []
        # Confirm not None.
        assert result.merged_key_claims is not None
        assert result.merged_objective_alignment is not None

    def test_duck_typed_section_without_audit_attrs(self):
        """Section-like object that doesn't even DECLARE the new attrs
        (legacy DART-side ContentSection-like, pre-W5.A parser) →
        getattr default kicks in, no AttributeError, fields are []."""
        legacy_section = SimpleNamespace(
            heading="Legacy",
            level=2,
            content="legacy content here",
            word_count=3,
            source_references=[],
            template_type=None,
            # No key_claims, no objective_alignment attrs at all.
        )
        merged = merge_small_sections([legacy_section], max_chunk_size=1000)
        assert len(merged) == 1
        result = merged[0]
        assert result.merged_key_claims == []
        assert result.merged_objective_alignment == []

    def test_per_section_buffer_resets_across_flush(self):
        """When the buffer flushes (next section would exceed
        MAX_CHUNK_SIZE), the new buffer's audit fields start fresh —
        no leak from the previous flush."""
        # Word counts: 400 + 400 → first chunk; second 400 starts new
        # buffer. Use max=800 so first two merge.
        big = " ".join(["w"] * 400)
        sections = [
            _section(
                "A",
                big,
                key_claims=[{"claim": "claim_A"}],
                objective_alignment=[{"objective_id": "TO-01"}],
            ),
            _section(
                "B",
                big,
                key_claims=[{"claim": "claim_B"}],
                objective_alignment=[{"objective_id": "TO-02"}],
            ),
            _section(
                "C",
                big,
                key_claims=[{"claim": "claim_C"}],
                objective_alignment=[{"objective_id": "TO-03"}],
            ),
        ]
        merged = merge_small_sections(sections, max_chunk_size=800)
        # A+B merge (400+400=800); C overflows -> new chunk.
        assert len(merged) == 2
        # First merged chunk: A + B claims/alignment unioned.
        assert [e["claim"] for e in merged[0].merged_key_claims] == [
            "claim_A", "claim_B"
        ]
        assert [
            e["objective_id"] for e in merged[0].merged_objective_alignment
        ] == ["TO-01", "TO-02"]
        # Second merged chunk: ONLY C's claims/alignment — no leak from
        # the previous buffer.
        assert [e["claim"] for e in merged[1].merged_key_claims] == ["claim_C"]
        assert [
            e["objective_id"] for e in merged[1].merged_objective_alignment
        ] == ["TO-03"]

    def test_non_dict_objective_alignment_entries_silently_dropped(self):
        """Defensive against malformed payloads: non-dict entries in
        objective_alignment are silently filtered at dedupe time."""
        sections = [
            _section(
                "Sec A",
                "alpha alpha alpha",
                objective_alignment=[
                    {"objective_id": "TO-05", "status": "delivered"},
                    "malformed_string_entry",  # non-dict — should be dropped
                    None,  # also dropped
                ],
            ),
        ]
        merged = merge_small_sections(sections, max_chunk_size=1000)
        assert len(merged) == 1
        # Only the dict entry survives.
        assert len(merged[0].merged_objective_alignment) == 1
        assert merged[0].merged_objective_alignment[0]["objective_id"] == "TO-05"

    def test_objective_alignment_entry_without_objective_id_dropped(self):
        """An entry missing the objective_id key can't be deduped, so
        it's silently dropped (defensive against malformed payloads
        that slip past schema validation)."""
        sections = [
            _section(
                "Sec A",
                "alpha alpha alpha",
                objective_alignment=[
                    {"status": "delivered"},  # no objective_id — dropped
                    {"objective_id": "TO-05", "status": "delivered"},
                ],
            ),
        ]
        merged = merge_small_sections(sections, max_chunk_size=1000)
        assert merged[0].merged_objective_alignment == [
            {"objective_id": "TO-05", "status": "delivered"}
        ]


# --------------------------------------------------------------------- #
# chunk_text_block — kwargs threading
# --------------------------------------------------------------------- #


class TestChunkTextBlockThreadsAuditFields:
    """Pin: chunk_text_block accepts the new kwargs and forwards them
    to the ctx.create_chunk callback."""

    def _capture_callback(self, captured: List[Dict[str, Any]]):
        """Build a create_chunk callback that records every kwargs dict."""

        def _callback(**kwargs):
            captured.append(kwargs)
            # Return a minimal chunk dict so the chunker's emit loop
            # works (it reads chunk["id"] downstream).
            return {
                "id": kwargs["chunk_id"],
                "text": kwargs["text"],
                "key_claims": kwargs.get("merged_key_claims"),
                "objective_alignment": kwargs.get("merged_objective_alignment"),
            }

        return _callback

    def test_threads_merged_audit_fields_into_chunk(self):
        """Round-trip: pass merged_key_claims + merged_objective_alignment
        to chunk_text_block; assert the callback receives them."""
        captured: List[Dict[str, Any]] = []
        ctx = ChunkerContext(create_chunk=self._capture_callback(captured))

        merged_key_claims = [
            {"claim": "X is Y", "source_chunk_ids": ["dart:a#1"]},
        ]
        merged_objective_alignment = [
            {"objective_id": "TO-05", "status": "delivered"},
        ]
        item = {
            "module_id": "mod_01",
            "item_id": "item_01",
            "raw_html": "<html><body><p>alpha beta gamma</p></body></html>",
            "title": "Page Title",
        }
        chunks = chunk_text_block(
            text="alpha beta gamma",
            html="<p>alpha beta gamma</p>",
            item=item,
            heading="Sec",
            chunk_type="explanation",
            prefix="test_chunk_",
            start_id=1,
            ctx=ctx,
            merged_key_claims=merged_key_claims,
            merged_objective_alignment=merged_objective_alignment,
        )
        assert len(chunks) == 1
        # The callback observed the new kwargs.
        assert len(captured) == 1
        assert captured[0]["merged_key_claims"] == merged_key_claims
        assert captured[0]["merged_objective_alignment"] == merged_objective_alignment
        # And the chunk dict carries them through the test callback.
        assert chunks[0]["key_claims"] == merged_key_claims
        assert chunks[0]["objective_alignment"] == merged_objective_alignment

    def test_omits_kwargs_when_none(self):
        """Back-compat: when merged_key_claims / merged_objective_alignment
        are None / empty, the kwargs are NOT passed to the callback so
        legacy callbacks (pre-W5.B) keep working without a TypeError."""
        captured: List[Dict[str, Any]] = []

        def _legacy_callback(**kwargs):
            # Legacy callback signature — would raise TypeError if W5.F
            # kwargs were always passed. Verify they aren't.
            assert "merged_key_claims" not in kwargs
            assert "merged_objective_alignment" not in kwargs
            captured.append(kwargs)
            return {"id": kwargs["chunk_id"], "text": kwargs["text"]}

        ctx = ChunkerContext(create_chunk=_legacy_callback)
        item = {
            "module_id": "mod_01",
            "item_id": "item_01",
            "raw_html": "<html><body><p>alpha beta gamma</p></body></html>",
            "title": "Page Title",
        }
        # Don't pass the new kwargs at all.
        chunks = chunk_text_block(
            text="alpha beta gamma",
            html="<p>alpha beta gamma</p>",
            item=item,
            heading="Sec",
            chunk_type="explanation",
            prefix="test_chunk_",
            start_id=1,
            ctx=ctx,
        )
        assert len(chunks) == 1

    def test_legacy_callback_falls_back_when_kwargs_unsupported(self):
        """When a legacy callback (no merged_key_claims kwarg) is wired,
        chunk_text_block catches the TypeError and re-dispatches without
        the W5.F kwargs. Pre-W5.B callbacks keep working byte-identical."""
        captured: List[Dict[str, Any]] = []

        def _strict_legacy_callback(
            chunk_id, text, html, item, section_heading, chunk_type,
            follows_chunk_id=None, position_in_module=0,
            html_xpath=None, char_span=None,
            section_source_ids=None, merged_headings=None,
        ):
            # No merged_key_claims / merged_objective_alignment — this
            # is the pre-W5.B callback signature.
            captured.append({
                "chunk_id": chunk_id,
                "text": text,
                "section_heading": section_heading,
            })
            return {"id": chunk_id, "text": text}

        ctx = ChunkerContext(create_chunk=_strict_legacy_callback)
        item = {
            "module_id": "mod_01",
            "item_id": "item_01",
            "raw_html": "<html><body><p>alpha beta gamma</p></body></html>",
            "title": "Page Title",
        }
        # PASS the W5.F kwargs — the dispatcher should fall back.
        chunks = chunk_text_block(
            text="alpha beta gamma",
            html="<p>alpha beta gamma</p>",
            item=item,
            heading="Sec",
            chunk_type="explanation",
            prefix="test_chunk_",
            start_id=1,
            ctx=ctx,
            merged_key_claims=[{"claim": "X"}],
            merged_objective_alignment=[{"objective_id": "TO-05"}],
        )
        assert len(chunks) == 1
        # Legacy callback was invoked successfully — no TypeError surfaced.
        assert captured[0]["section_heading"] == "Sec"


# --------------------------------------------------------------------- #
# Round-trip end-to-end: merge → chunk → CourseProcessor stamp
# --------------------------------------------------------------------- #


class TestEndToEndChunkStamping:
    """Round-trip: ContentSections → merge → chunk_text_block →
    CourseProcessor._create_chunk → chunk dict carries the merged
    audit fields. Verifies W5.F + W5.B compose correctly."""

    def _make_processor(self):
        from Trainforge.process_course import CourseProcessor

        proc = object.__new__(CourseProcessor)
        proc.capture = SimpleNamespace(
            run_id="test_run_w5f",
            log_decision=lambda **kwargs: None,
        )
        proc.course_code = "test_101"
        proc.stats = {
            "total_chunks": 0,
            "total_words": 0,
            "total_tokens_estimate": 0,
            "chunk_types": defaultdict(int),
            "difficulty_distribution": defaultdict(int),
            "modules_processed": 0,
            "quizzes_processed": 0,
            "sections_processed": 0,
        }
        proc._boilerplate_spans = []
        proc._all_concept_tags = set()
        proc.domain_concept_seeds = []
        proc.objectives = None
        proc.OBJECTIVE_CODE_RE = CourseProcessor.OBJECTIVE_CODE_RE
        proc.WEEK_PREFIX_RE = CourseProcessor.WEEK_PREFIX_RE
        proc.NON_CONCEPT_TAGS = CourseProcessor.NON_CONCEPT_TAGS
        proc.MIN_CHUNK_SIZE = 100
        proc.MAX_CHUNK_SIZE = 800
        proc.TARGET_CHUNK_SIZE = 500
        return proc

    def test_merge_then_chunk_carries_merged_audit_fields(self):
        """Round-trip: two sections each with their own audit fields →
        merge → chunk_text_block via CourseProcessor → chunk dict has
        the unioned key_claims AND objective_alignment."""
        proc = self._make_processor()

        sections = [
            _section(
                "Sec A",
                "alpha " * 50,
                key_claims=[{"claim": "X is Y", "source_chunk_ids": ["dart:a#1"]}],
                objective_alignment=[
                    {"objective_id": "TO-05", "status": "delivered"},
                ],
            ),
            _section(
                "Sec B",
                "beta " * 50,
                key_claims=[{"claim": "Z is W", "source_chunk_ids": ["dart:b#1"]}],
                objective_alignment=[
                    {"objective_id": "TO-06", "status": "delivered"},
                ],
            ),
        ]
        # Merge: both fit in MAX_CHUNK_SIZE=800 → one merged result.
        merged = proc._merge_small_sections(sections)
        assert len(merged) == 1
        result = merged[0]
        # Sanity: the merger collected both sections' fields.
        assert len(result.merged_key_claims) == 2
        assert len(result.merged_objective_alignment) == 2

        # Now thread through chunk_text_block + _create_chunk.
        item = {
            "module_id": "mod_01",
            "module_title": "Module 1",
            "item_id": "item_01",
            "title": "Page Title",
            "resource_type": "content",
            "raw_html": "<html><body><p>alpha beta gamma</p></body></html>",
            "item_path": "week_01/page.html",
            "courseforge_metadata": None,
            "sections": [],
            "learning_objectives": [],
            "misconceptions": [],
        }
        chunks = proc._chunk_text_block(
            text=result.combined_text,
            html="<p>" + result.combined_text + "</p>",
            item=item,
            heading=result.heading,
            chunk_type=result.chunk_type,
            prefix="test_chunk_",
            start_id=1,
            section_source_ids=result.merged_source_ids,
            merged_headings=result.merged_headings,
            merged_key_claims=result.merged_key_claims,
            merged_objective_alignment=result.merged_objective_alignment,
        )
        # One chunk emitted (text under MAX_CHUNK_SIZE).
        assert len(chunks) == 1
        chunk = chunks[0]
        # Chunk carries the merged audit fields.
        assert "key_claims" in chunk
        assert "objective_alignment" in chunk
        assert len(chunk["key_claims"]) == 2
        claims = [c["claim"] for c in chunk["key_claims"]]
        assert "X is Y" in claims and "Z is W" in claims
        oids = [a["objective_id"] for a in chunk["objective_alignment"]]
        assert oids == ["TO-05", "TO-06"]

    def test_back_compat_legacy_sections_no_stamp(self):
        """When sections carry no audit fields, the chunk dict doesn't
        carry the keys at all (not even as []). Pre-W5.A baseline."""
        proc = self._make_processor()

        sections = [
            _section("Sec A", "alpha " * 50),
            _section("Sec B", "beta " * 50),
        ]
        merged = proc._merge_small_sections(sections)
        item = {
            "module_id": "mod_01",
            "module_title": "Module 1",
            "item_id": "item_01",
            "title": "Page Title",
            "resource_type": "content",
            "raw_html": "<html><body><p>alpha beta</p></body></html>",
            "item_path": "week_01/page.html",
            "courseforge_metadata": None,
            "sections": [],
            "learning_objectives": [],
            "misconceptions": [],
        }
        chunks = proc._chunk_text_block(
            text=merged[0].combined_text,
            html="<p>x</p>",
            item=item,
            heading=merged[0].heading,
            chunk_type=merged[0].chunk_type,
            prefix="test_chunk_",
            start_id=1,
            section_source_ids=merged[0].merged_source_ids,
            merged_headings=merged[0].merged_headings,
            merged_key_claims=merged[0].merged_key_claims,
            merged_objective_alignment=merged[0].merged_objective_alignment,
        )
        assert len(chunks) == 1
        # Pre-W5.A back-compat: keys ABSENT, not present-with-empty-list.
        assert "key_claims" not in chunks[0]
        assert "objective_alignment" not in chunks[0]
