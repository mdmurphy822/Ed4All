"""Wave 8 — DART per-block source-provenance emission.

These tests exercise the Wave 8 changes in
``DART/multi_source_interpreter.py``:

* ``auto_synthesize_section`` emits the new per-section record shape with
  ``section_id`` / ``page_range`` / ``provenance`` block while preserving
  the legacy ``sources_used`` field.
* Per-matcher functions (``synthesize_contacts``, ``synthesize_systems_table``,
  ``synthesize_roster``) emit ``block_id`` + provenance envelopes on every
  block, with the canonical 5-value confidence scale.
* ``generate_html_from_synthesized`` emits ``data-dart-*`` attributes on
  ``<section>`` and ``.contact-card`` wrappers. Attributes are scoped per
  design-doc P2 (no per-``<p>`` / ``<tr>`` bloat in prose) and respect the
  "omit data-dart-pages when unknown" + "omit data-dart-confidence when
  1.0" rules.
* Legacy ``claude_processor`` renderer stamps ``data-dart-source="claude_llm"``
  on section wrappers.

Minimal synthetic fixtures are used instead of the full PDF extraction
pipeline — the goal is to verify the emit-layer contract, not to
re-exercise pdftotext / pdfplumber / OCR upstream.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from DART.multi_source_interpreter import (
    CONFIDENCE_DERIVATION,
    CONFIDENCE_DIRECT_TABLE,
    CONFIDENCE_NAME_PATTERN,
    CONFIDENCE_PROXIMITY,
    SOURCE_PDFPLUMBER,
    SOURCE_PDFTOTEXT,
    SOURCE_SYNTHESIZED,
    _build_dart_attrs,
    _format_pages_attr,
    _make_block_id,
    _page_range_list,
    auto_synthesize_section,
    generate_html_from_synthesized,
    match_email_by_name,
    match_phone_by_proximity,
    synthesize_contacts,
    synthesize_roster,
    synthesize_systems_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _contacts_ctx(page_range=(3, 4)):
    """Minimal context for a contacts section with two identifiable people."""
    return {
        "section_type": "contacts",
        "section_title": "Campus Contacts",
        "page_range": page_range,
        "sources": {
            "pdftotext": [
                "Jane Doe", "IT Coordinator",
                "jane.doe@campus.edu", "518-555-1234",
                "Help Desk", "help@campus.edu", "518-555-9999",
            ],
            "tables": [
                {"page": 3, "headers": ["Jane Doe", "Help Desk"], "rows": []},
            ],
            "ocr": [],
        },
        "entities": {
            "phones": ["518-555-1234", "518-555-9999"],
            "emails": ["jane.doe@campus.edu", "help@campus.edu"],
            "urls": [],
            "names": ["Jane Doe"],
        },
    }


def _systems_ctx(page_range=(5, 6)):
    """Minimal 3-column systems table context."""
    return {
        "section_type": "systems",
        "section_title": "Campus Systems",
        "page_range": page_range,
        "sources": {
            "pdftotext": ["Campus Email", "https://mail.campus.edu"],
            "tables": [{
                "page": 5,
                "headers": ["System", "Students", "Faculty"],
                "rows": [
                    ["Campus Email", "student.campus.edu", "faculty.campus.edu"],
                    ["LMS", "lms.campus.edu", "lms-admin.campus.edu"],
                ],
            }],
            "ocr": [],
        },
        "entities": {"phones": [], "emails": [], "urls": [], "names": []},
    }


def _roster_ctx(page_range=(4, 5)):
    return {
        "section_type": "roster",
        "section_title": "Course / Roster",
        "page_range": page_range,
        "sources": {
            "pdftotext": [
                "Semester: Fall 2025",
                "Course Code: BIO 101",
                "Instructor: Dr. Smith",
            ],
            "tables": [],
            "ocr": [],
        },
        "entities": {"phones": [], "emails": [], "urls": [], "names": []},
    }


def _prose_ctx(page_range=(2, 3)):
    return {
        "section_type": "guest",
        "section_title": "Guest / Observer",
        "page_range": page_range,
        "sources": {
            "pdftotext": [
                "Guests may attend without credentials.",
                "Contact the registrar for details.",
            ],
            "tables": [],
            "ocr": [],
        },
        "entities": {"phones": [], "emails": [], "urls": [], "names": []},
    }


# ---------------------------------------------------------------------------
# auto_synthesize_section: per-section shape
# ---------------------------------------------------------------------------


class TestSectionShape:
    """Per-section record shape — verify the Wave 8 contract."""

    def test_section_has_required_wave8_fields(self):
        section = auto_synthesize_section(_contacts_ctx(), section_index=3)
        assert section["section_id"] == "s3"
        assert section["section_type"] == "contacts"
        assert section["section_title"] == "Campus Contacts"
        assert section["page_range"] == [3, 4]
        assert "provenance" in section
        assert "data" in section

    def test_section_provenance_block_shape(self):
        section = auto_synthesize_section(_contacts_ctx(), section_index=0)
        prov = section["provenance"]
        assert set(prov.keys()) == {"sources", "strategy", "confidence"}
        assert all(
            s in ("pdftotext", "pdfplumber", "ocr", "synthesized")
            for s in prov["sources"]
        )
        assert 0.0 <= prov["confidence"] <= 1.0
        assert isinstance(prov["strategy"], str)
        assert prov["strategy"]  # non-empty

    def test_legacy_sources_used_retained(self):
        """Back-compat: sources_used dict must still be present."""
        section = auto_synthesize_section(_contacts_ctx(), section_index=0)
        assert "sources_used" in section
        assert isinstance(section["sources_used"], dict)

    def test_prose_section_paragraph_provenance(self):
        section = auto_synthesize_section(_prose_ctx(), section_index=7)
        assert section["section_id"] == "s7"
        data = section["data"]
        # Back-compat flat paragraphs preserved
        assert "paragraphs" in data
        assert data["paragraphs"]
        # New paragraph_provenance envelope list
        assert "paragraph_provenance" in data
        envelopes = data["paragraph_provenance"]
        assert len(envelopes) >= 1
        for env in envelopes:
            assert "block_id" in env
            assert "provenance" in env
            p = env["provenance"]
            assert p["source"] == SOURCE_PDFTOTEXT
            assert 0.0 <= p["confidence"] <= 1.0
            assert p["pages"] == [2, 3]


# ---------------------------------------------------------------------------
# Per-block envelopes
# ---------------------------------------------------------------------------


class TestContactsProvenance:
    def test_contacts_carry_block_id_and_name_provenance(self):
        section = auto_synthesize_section(_contacts_ctx(), section_index=0)
        contacts = section["data"]["contacts"]
        assert contacts, "Expected at least one contact"
        for c in contacts:
            assert "block_id" in c
            assert c["block_id"].startswith("s0_c")
            assert "name_provenance" in c
            np = c["name_provenance"]
            assert np["source"] in (SOURCE_PDFPLUMBER, SOURCE_PDFTOTEXT, SOURCE_SYNTHESIZED)
            assert 0.0 <= np["confidence"] <= 1.0

    def test_contact_email_envelope_has_method(self):
        section = auto_synthesize_section(_contacts_ctx(), section_index=0)
        contacts = section["data"]["contacts"]
        jane = next(c for c in contacts if "Jane" in c.get("name", ""))
        assert "email_provenance" in jane
        env = jane["email_provenance"]
        assert env["source"] == SOURCE_PDFTOTEXT
        assert env["method"] in ("name_pattern", "proximity", "special_case")
        # Name-pattern match should score 0.8 per the canonical scale.
        assert env["confidence"] == CONFIDENCE_NAME_PATTERN

    def test_contact_legacy_plain_fields_preserved(self):
        """Plain string fields stay on contact dicts for back-compat."""
        section = auto_synthesize_section(_contacts_ctx(), section_index=0)
        jane = section["data"]["contacts"][0]
        assert jane["name"]  # plain string
        assert isinstance(jane["email"], str)
        assert isinstance(jane["phone"], str)

    def test_derivation_path_uses_low_confidence(self):
        """Email-only synthesis path drops to CONFIDENCE_DERIVATION."""
        ctx = {
            "section_type": "contacts",
            "section_title": "Campus Contacts",
            "page_range": (3, 4),
            "sources": {
                "pdftotext": ["jdoe@campus.edu"],
                "tables": [],  # no headers -> derivation path triggers
                "ocr": [],
            },
            "entities": {
                "phones": [],
                "emails": ["jdoe@campus.edu"],
                "urls": [],
                "names": [],  # no names either
            },
        }
        section = auto_synthesize_section(ctx, section_index=0)
        contacts = section["data"]["contacts"]
        assert contacts, "Derivation path should produce a synthesized contact"
        name_env = contacts[0]["name_provenance"]
        assert name_env["source"] == SOURCE_SYNTHESIZED
        assert name_env["confidence"] == CONFIDENCE_DERIVATION
        assert name_env["method"] == "email_local_part"


class TestSystemsProvenance:
    def test_systems_rows_carry_block_id_and_provenance(self):
        section = auto_synthesize_section(_systems_ctx(), section_index=2)
        rows = section["data"]["rows"]
        assert len(rows) >= 1
        for row in rows:
            assert "block_id" in row
            assert row["block_id"].startswith("s2_r")
            assert "provenance" in row
            p = row["provenance"]
            # Direct-table extraction at page 5 -> pdfplumber + confidence 1.0.
            assert p["source"] == SOURCE_PDFPLUMBER
            assert p["confidence"] == CONFIDENCE_DIRECT_TABLE
            assert p["pages"] == [5]

    def test_systems_fallback_uses_pdftotext_proximity(self):
        """When no structured 3-col table matches, fallback uses pdftotext."""
        ctx = _systems_ctx()
        # Strip out the structured table — this should force the fallback.
        ctx["sources"]["tables"] = []
        # Fallback looks for known labels in the pdftotext stream.
        ctx["sources"]["pdftotext"] = ["Campus Email https://mail.campus.edu"]
        section = auto_synthesize_section(ctx, section_index=0)
        rows = section["data"]["rows"]
        if rows:
            env = rows[0]["provenance"]
            assert env["source"] == SOURCE_PDFTOTEXT
            assert env["confidence"] == CONFIDENCE_PROXIMITY


class TestRosterProvenance:
    def test_roster_carries_pair_provenance_side_channel(self):
        section = auto_synthesize_section(_roster_ctx(), section_index=4)
        data = section["data"]
        # Back-compat pairs list must still be present (List[Tuple]).
        assert "pairs" in data
        # Wave 8 addition: pair_provenance list with per-pair envelopes.
        assert "pair_provenance" in data
        envelopes = data["pair_provenance"]
        for env in envelopes:
            assert "block_id" in env
            assert env["block_id"].startswith("s4_p")
            assert env["provenance"]["source"] in (
                SOURCE_PDFTOTEXT, SOURCE_SYNTHESIZED, SOURCE_PDFPLUMBER,
            )

    def test_roster_tuple_shape_preserved(self):
        """Renderer still sees List[Tuple[label, value]] for pairs."""
        section = auto_synthesize_section(_roster_ctx(), section_index=0)
        pairs = section["data"]["pairs"]
        for p in pairs:
            assert isinstance(p, tuple)
            assert len(p) == 2


# ---------------------------------------------------------------------------
# match_* functions: direct 3-tuple return contract
# ---------------------------------------------------------------------------


class TestMatcherReturnShape:
    def test_match_email_returns_email_method_confidence(self):
        email, method, conf = match_email_by_name(
            "Jane Doe",
            ["jdoe@campus.edu"],
            "Jane Doe jdoe@campus.edu",
        )
        assert email == "jdoe@campus.edu"
        assert method == "name_pattern"
        assert conf == CONFIDENCE_NAME_PATTERN

    def test_match_email_returns_empty_tuple_on_miss(self):
        email, method, conf = match_email_by_name(
            "Nobody", [], "text",
        )
        assert email == ""
        assert method == ""
        assert conf == 0.0

    def test_match_phone_returns_proximity_confidence(self):
        phone, method, conf = match_phone_by_proximity(
            "Jane Doe", "", ["518-555-1234"], "Jane Doe 518-555-1234",
        )
        assert phone == "518-555-1234"
        assert method == "proximity"
        assert conf == CONFIDENCE_PROXIMITY


# ---------------------------------------------------------------------------
# data-dart-* HTML emission
# ---------------------------------------------------------------------------


def _synthesized_doc(sections):
    return {
        "campus_code": "TEST",
        "campus_name": "Test Campus",
        "sections": sections,
    }


class TestHtmlAttributes:
    def test_section_carries_data_dart_attributes(self):
        section = auto_synthesize_section(_contacts_ctx(), section_index=3)
        doc = _synthesized_doc([section])
        html = generate_html_from_synthesized(doc)

        section_match = re.search(r"<section[^>]*id=\"s0\"[^>]*>", html)
        assert section_match, "Expected a <section> tag with id=s0 in output"
        tag = section_match.group(0)
        assert 'data-dart-block-id="s3"' in tag
        assert 'data-dart-source="' in tag
        assert 'data-dart-pages="3-4"' in tag
        assert 'data-dart-strategy="' in tag

    def test_contact_card_carries_data_dart_attributes(self):
        section = auto_synthesize_section(_contacts_ctx(), section_index=0)
        doc = _synthesized_doc([section])
        html = generate_html_from_synthesized(doc)

        card_match = re.search(r"<div class=\"contact-card [^>]*>", html)
        assert card_match, "Expected a contact-card div in output"
        tag = card_match.group(0)
        assert "data-dart-block-id=" in tag
        assert 'data-dart-source="' in tag

    def test_systems_section_lists_multi_sources(self):
        section = auto_synthesize_section(_systems_ctx(), section_index=0)
        doc = _synthesized_doc([section])
        html = generate_html_from_synthesized(doc)
        # Structured table -> only pdfplumber contributed; no data-dart-sources
        # expected because it's single-source.
        section_match = re.search(r"<section[^>]*id=\"s0\"[^>]*>", html)
        assert section_match
        tag = section_match.group(0)
        assert 'data-dart-source="pdfplumber"' in tag

    def test_confidence_omitted_when_one_point_zero(self):
        """Per spec: don't emit data-dart-confidence when value is 1.0."""
        attrs = _build_dart_attrs(
            block_id="s0", source=SOURCE_PDFPLUMBER, confidence=1.0,
        )
        assert "data-dart-confidence" not in attrs
        assert 'data-dart-block-id="s0"' in attrs

    def test_confidence_emitted_two_decimals(self):
        attrs = _build_dart_attrs(
            block_id="s0", source=SOURCE_PDFTOTEXT, confidence=0.873,
        )
        assert 'data-dart-confidence="0.87"' in attrs

    def test_pages_omitted_when_empty(self):
        """Per spec: don't lie — omit data-dart-pages when unknown."""
        attrs = _build_dart_attrs(block_id="s0", source="pdftotext", pages=[])
        assert "data-dart-pages" not in attrs


class TestPageFormatting:
    @pytest.mark.parametrize(
        "pages, expected",
        [
            ([3], "3"),
            ([3, 4, 5], "3-5"),
            ([3, 5, 7], "3,5,7"),
            ([5, 3, 4], "3-5"),  # sort + range
            ([], ""),
            ([0], ""),            # zero invalid
            ([-1, 2], "2"),       # negatives dropped
        ],
    )
    def test_format_pages_attr(self, pages, expected):
        assert _format_pages_attr(pages) == expected


class TestPageRangeList:
    @pytest.mark.parametrize(
        "pr, expected",
        [
            ((3, 4), [3, 4]),
            ((3, 3), [3]),
            ([5, 7], [5, 6, 7]),
            (None, []),
            ((0, 0), []),
            ((5, 3), []),  # end < start
        ],
    )
    def test_page_range_list(self, pr, expected):
        assert _page_range_list(pr) == expected


class TestBlockIdGeneration:
    def test_positional_block_id_default(self, monkeypatch):
        """With flag off, block_id stays positional."""
        monkeypatch.delenv("TRAINFORGE_CONTENT_HASH_IDS", raising=False)
        bid = _make_block_id("s3", "s3_c0", "content", "Jane Doe")
        assert bid == "s3_c0"

    def test_content_hash_block_id_under_flag(self, monkeypatch):
        """With flag on, block_id switches to 16-hex content hash."""
        monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "1")
        bid = _make_block_id("s3", "s3_c0", "content", "Jane Doe")
        assert re.fullmatch(r"[0-9a-f]{16}", bid), f"got {bid!r}"

    def test_content_hash_stable_across_calls(self, monkeypatch):
        monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "1")
        a = _make_block_id("s3", "s3_c0", "content", "Jane Doe")
        b = _make_block_id("s3", "s3_c0", "content", "Jane Doe")
        assert a == b


# ---------------------------------------------------------------------------
# Legacy claude_processor HTML: minimal data-dart-source stamp
# ---------------------------------------------------------------------------


class TestLegacyClaudeProcessorStamp:
    def test_claude_llm_section_stamp_present(self):
        """converter._generate_html_from_structure stamps <section> wrappers."""
        from DART.pdf_converter.claude_processor import (
            DocumentStructure,
            StructuredBlock,
        )
        from DART.pdf_converter.converter import PDFToAccessibleHTML

        doc = DocumentStructure(
            title="Test Paper",
            authors=["A. Author"],
            abstract="An abstract.",
            blocks=[
                StructuredBlock(block_type="heading", content="Introduction",
                                heading_level=2),
                StructuredBlock(block_type="paragraph",
                                content="This is the introduction text " * 3),
                StructuredBlock(block_type="heading", content="References",
                                heading_level=2),
            ],
            metadata={},
        )
        conv = PDFToAccessibleHTML()
        html_out = conv._generate_html_from_structure(doc)
        # Every section wrapper must carry the legacy provenance stamp.
        assert 'data-dart-source="claude_llm"' in html_out
        # Abstract section also stamped.
        assert html_out.count('data-dart-source="claude_llm"') >= 2


# ---------------------------------------------------------------------------
# Round-trip: JSON sidecar + HTML stay internally consistent
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_section_id_matches_html_block_id(self):
        section = auto_synthesize_section(_contacts_ctx(), section_index=5)
        html = generate_html_from_synthesized(
            _synthesized_doc([section])
        )
        # Section's section_id (s5) flows to data-dart-block-id on the
        # <section> wrapper even though the DOM id stays "s0" (position).
        assert 'data-dart-block-id="s5"' in html

    def test_contact_block_ids_are_stable_within_section(self):
        section = auto_synthesize_section(_contacts_ctx(), section_index=0)
        ids = [c["block_id"] for c in section["data"]["contacts"]]
        assert len(ids) == len(set(ids)), "block_ids must be unique"

    def test_all_provenance_confidences_in_canonical_scale(self):
        """Every confidence value in the envelope must be one of our 5 points."""
        section = auto_synthesize_section(_contacts_ctx(), section_index=0)
        valid = {
            CONFIDENCE_DIRECT_TABLE,
            CONFIDENCE_NAME_PATTERN,
            CONFIDENCE_PROXIMITY,
            CONFIDENCE_DERIVATION,
            0.2,  # OCR fallback
        }
        for c in section["data"]["contacts"]:
            for key in ("name_provenance", "email_provenance",
                        "phone_provenance", "title_provenance"):
                env = c.get(key)
                if env:
                    assert env["confidence"] in valid, (
                        f"Confidence {env['confidence']} not in canonical scale"
                    )
