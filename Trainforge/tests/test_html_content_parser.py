"""Wave 5 W5.A: HTMLContentParser carries W1.5 ``key_claims`` and W1.7
``objective_alignment`` from JSON-LD ``blocks[]`` projections.

The parser-side half of the chunker ingestion mirror. Without this,
every downstream consumer (``Trainforge/process_course.py``, the
NLI claim-support gate, the ``BlockObjectiveDeliveryValidator``
tri-axis check) silently drops the audit signals emitted by the
Phase-2 ``Block`` dataclass on the wire.

Surfaces touched:

* ``ContentSection`` dataclass — two new optional fields
  (``key_claims``, ``objective_alignment``) defaulted to ``[]`` so
  legacy emit paths never see ``None``.
* ``HTMLContentParser._content_sections_from_blocks`` — copies
  ``block.get("keyClaims", [])`` → ``key_claims`` and
  ``block.get("objectiveAlignment", [])`` → ``objective_alignment``
  with defensive dict-only filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.parsers.html_content_parser import (  # noqa: E402
    ContentSection,
    HTMLContentParser,
)


# ---------------------------------------------------------------------------
# W1.5 keyClaims projection
# ---------------------------------------------------------------------------


def test_content_sections_from_blocks_carries_key_claims():
    """Block carrying structured ``keyClaims`` lands on the section's
    ``key_claims`` field with claim + source_chunk_ids preserved.
    """
    parser = HTMLContentParser()
    blocks = [
        {
            "blockId": "page_a#explanation_intro_0",
            "blockType": "explanation",
            "contentTypeLabel": "explanation",
            "keyTerms": [],
            "templateType": "explanation",
            "keyClaims": [
                {
                    "claim": "X is Y",
                    "source_chunk_ids": ["dart:chunk_A"],
                },
                {
                    "claim": "Y implies Z",
                    "source_chunk_ids": ["dart:chunk_A", "dart:chunk_B"],
                },
            ],
        }
    ]
    sections = parser._content_sections_from_blocks(blocks)
    assert len(sections) == 1
    section = sections[0]
    assert isinstance(section.key_claims, list)
    assert len(section.key_claims) == 2
    first = section.key_claims[0]
    assert first["claim"] == "X is Y"
    assert first["source_chunk_ids"] == ["dart:chunk_A"]
    second = section.key_claims[1]
    assert second["claim"] == "Y implies Z"
    assert second["source_chunk_ids"] == ["dart:chunk_A", "dart:chunk_B"]


def test_content_sections_from_blocks_filters_non_dict_key_claims():
    """Defensive: malformed entries (strings, ints, None) are silently
    dropped. Mirrors the dict-coerce posture in
    ``_extract_blocks_from_jsonld``.
    """
    parser = HTMLContentParser()
    blocks = [
        {
            "blockId": "page_a#explanation_intro_0",
            "blockType": "explanation",
            "keyClaims": [
                {"claim": "valid", "source_chunk_ids": ["dart:chunk_A"]},
                "string-not-dict",
                42,
                None,
                {"claim": "also valid", "source_chunk_ids": []},
            ],
        }
    ]
    sections = parser._content_sections_from_blocks(blocks)
    assert len(sections) == 1
    assert len(sections[0].key_claims) == 2
    assert sections[0].key_claims[0]["claim"] == "valid"
    assert sections[0].key_claims[1]["claim"] == "also valid"


# ---------------------------------------------------------------------------
# W1.7 objectiveAlignment projection
# ---------------------------------------------------------------------------


def test_content_sections_from_blocks_carries_objective_alignment():
    """Block carrying structured ``objectiveAlignment`` lands on the
    section's ``objective_alignment`` field with objective_id, status,
    and declared_bloom preserved.
    """
    parser = HTMLContentParser()
    blocks = [
        {
            "blockId": "page_b#explanation_intro_0",
            "blockType": "explanation",
            "contentTypeLabel": "explanation",
            "objectiveAlignment": [
                {
                    "objective_id": "TO-05",
                    "status": "delivered",
                    "declared_bloom": "apply",
                },
                {
                    "objective_id": "CO-12",
                    "status": "partial",
                    "declared_bloom": "understand",
                },
            ],
        }
    ]
    sections = parser._content_sections_from_blocks(blocks)
    assert len(sections) == 1
    section = sections[0]
    assert isinstance(section.objective_alignment, list)
    assert len(section.objective_alignment) == 2
    first = section.objective_alignment[0]
    assert first["objective_id"] == "TO-05"
    assert first["status"] == "delivered"
    assert first["declared_bloom"] == "apply"
    second = section.objective_alignment[1]
    assert second["objective_id"] == "CO-12"
    assert second["status"] == "partial"
    assert second["declared_bloom"] == "understand"


def test_content_sections_from_blocks_filters_non_dict_objective_alignment():
    """Defensive: malformed entries silently dropped."""
    parser = HTMLContentParser()
    blocks = [
        {
            "blockId": "page_b#explanation_intro_0",
            "blockType": "explanation",
            "objectiveAlignment": [
                {"objective_id": "TO-01", "status": "delivered"},
                "TO-02",
                None,
                123,
                {"objective_id": "TO-03", "status": "partial"},
            ],
        }
    ]
    sections = parser._content_sections_from_blocks(blocks)
    assert len(sections) == 1
    assert len(sections[0].objective_alignment) == 2
    ids = [oa["objective_id"] for oa in sections[0].objective_alignment]
    assert ids == ["TO-01", "TO-03"]


# ---------------------------------------------------------------------------
# Back-compat: legacy blocks (pre-W1.5/W1.7) emit empty lists, not None
# ---------------------------------------------------------------------------


def test_content_sections_from_blocks_back_compat_legacy_blocks():
    """Legacy block (no ``keyClaims`` / ``objectiveAlignment``) projects
    to a ``ContentSection`` with empty lists — never ``None``.

    The defensive default matters because downstream consumers iterate
    these fields; ``None`` would raise ``TypeError`` and silently break
    every legacy corpus rebuilt against the new chunker schema.
    """
    parser = HTMLContentParser()
    blocks = [
        {
            "blockId": "page_legacy#explanation_intro_0",
            "blockType": "explanation",
            "contentTypeLabel": "explanation",
            "keyTerms": ["term_a", "term_b"],
            "templateType": "explanation",
            # NB: no ``keyClaims`` / ``objectiveAlignment`` keys at all,
            # mirroring the pre-W1.5 / pre-W1.7 emit shape.
        }
    ]
    sections = parser._content_sections_from_blocks(blocks)
    assert len(sections) == 1
    section = sections[0]
    # Critical: lists, not None. Default-factory contract.
    assert section.key_claims == []
    assert section.objective_alignment == []
    assert section.key_claims is not None
    assert section.objective_alignment is not None
    # Legacy fields still populate as before.
    assert section.content_type == "explanation"
    assert section.key_terms == ["term_a", "term_b"]
    assert section.template_type == "explanation"


def test_content_section_dataclass_default_factory_emits_empty_lists():
    """Direct dataclass construction without the new fields yields
    empty lists, not ``None``. Belt-and-braces: covers callers that
    instantiate ``ContentSection`` directly (a handful of tests + the
    legacy DOM-walk path in ``_extract_sections``).
    """
    section = ContentSection(
        heading="Heading", level=2, content="body", word_count=1
    )
    assert section.key_claims == []
    assert section.objective_alignment == []
    # field(default_factory=list) ⇒ each instance owns its own list.
    other = ContentSection(
        heading="Other", level=2, content="body", word_count=1
    )
    assert section.key_claims is not other.key_claims
    assert section.objective_alignment is not other.objective_alignment


# ---------------------------------------------------------------------------
# Combined projection: blocks carrying BOTH W1.5 and W1.7 audit fields
# ---------------------------------------------------------------------------


def test_content_sections_from_blocks_carries_both_audit_fields():
    """A block carrying BOTH ``keyClaims`` and ``objectiveAlignment``
    projects to a section that surfaces both audit arrays — the
    common case once W1.5 + W1.7 emit lands in Courseforge.
    """
    parser = HTMLContentParser()
    blocks = [
        {
            "blockId": "page_combined#explanation_intro_0",
            "blockType": "explanation",
            "keyClaims": [
                {"claim": "Foo", "source_chunk_ids": ["dart:chunk_X"]},
            ],
            "objectiveAlignment": [
                {
                    "objective_id": "TO-07",
                    "status": "delivered",
                    "declared_bloom": "analyze",
                },
            ],
        }
    ]
    sections = parser._content_sections_from_blocks(blocks)
    assert len(sections) == 1
    section = sections[0]
    assert len(section.key_claims) == 1
    assert section.key_claims[0]["claim"] == "Foo"
    assert len(section.objective_alignment) == 1
    assert section.objective_alignment[0]["objective_id"] == "TO-07"
