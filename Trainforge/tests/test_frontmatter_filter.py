"""Unit tests for the front-matter / donor contamination classifier.

Fixtures are lifted from (or faithfully model) a real OpenStax Elementary
Algebra 7B build's contaminated chunkset
(``LibV2/courses/<course-slug>/dart_chunks/chunks.jsonl`` —
``_chunk_00001`` cover/authors/copyright, ``_chunk_00002`` donor list + TOC +
marketing). The negative fixtures model real algebra content, INCLUDING the
"Chapter 1 Foundations" running-header case that is the documented
false-positive trap (68/72 real chunks contain "foundation").
"""

from __future__ import annotations

from Trainforge.chunker.frontmatter import (
    DEFAULT_COVER_REGION_CHUNKS,
    FrontmatterVerdict,
    classify_frontmatter,
    partition_frontmatter,
)


# ---------------------------------------------------------------------------
# Real contaminated fixtures (verbatim head of the actual chunks).
# ---------------------------------------------------------------------------

# ``_chunk_00001`` — cover / contributing authors / copyright / ISBN / CC.
CHUNK_00001_COVER = (
    "Elementary Algebra 2e SENIOR CONTRIBUTING AUTHORS LYNN MARECEK, SANTA "
    "ANA COLLEGE MARYANNE ANTHONY-SMITH, SANTA ANA COLLEGE ANDREA HONEYCUTT "
    "MATHIS, NORTHEAST MISSISSIPPI COMMUNITY COLLEGE OpenStax Rice University "
    "6100 Main Street MS-375 Houston, Texas 77005 To learn more about "
    "OpenStax, visit https://openstax.org. Individual print copies and bulk "
    "orders can be purchased through our website. If you redistribute this "
    "textbook in a digital format (including but not limited to PDF and HTML), "
    "then you must retain on every page the following attribution: “Access "
    "for free at openstax.org.” Trademarks The OpenStax name, OpenStax "
    "logo, OpenStax book covers, OpenStax CNX name, OpenStax CNX logo, "
    "Connexions name, Connexions logo, Rice University name, and Rice "
    "University logo are not subject to the Creative Commons license."
)

# ``_chunk_00002`` — donor list ("Leon Lowenstein Foundation, Inc.", named
# individual donors) + marketing web-view blurb + Table of Contents.
CHUNK_00002_DONOR_TOC = (
    "Leon Lowenstein Foundation, Inc. Tammy and Guillermo Treviño Study "
    "where you want, what you want, when you want. When you access your book "
    "in our web view, you can use our new online highlighting and note-taking "
    "features to create your own study guides. Our books are free and "
    "flexible, forever. Get started at openstax.org/details/books/"
    "sample-algebra-2e Access. The future of education. openstax.org "
    "Table of Contents Preface 1 1 Foundations 5 1.1 Introduction to Whole "
    "Numbers 5 1.2 Use the Language of Algebra 21 1.3 Add and Subtract "
    "Integers 41 1.4 Multiply and Divide Integers 64 1.5 Visualize Fractions "
    "79 1.6 Add and Subtract Fractions 95 1.7 Decimals 111 1.8 The Real "
    "Numbers 130 1.9 Properties of Real Numbers 146 1.10 Systems of "
    "Measurement 165"
)

# ``_chunk_00004`` — "Key Features" preface (book features marketing).
CHUNK_00004_KEY_FEATURES = (
    "Key Features and Boxes Examples Each learning objective is supported by "
    "one or more worked examples that demonstrate the problem-solving "
    "approaches that students must master. Get started in our web view. "
    "Additional Resources Student and Instructor Resources We have compiled "
    "additional resources for both students and instructors."
)


# ---------------------------------------------------------------------------
# Real / realistic negative fixtures — genuine algebra content.
# ---------------------------------------------------------------------------

# The critical false-positive trap: a real worked-math chunk whose running
# header carries "Chapter 1 Foundations" (the algebra chapter title — NOT a
# donor). MUST NOT be classified front-matter.
CHUNK_FOUNDATIONS_HEADER_MATH = (
    "Chapter 1 Foundations 11 A number is a multiple of n if it is the "
    "product of a counting number and n. Another way to say that 15 is a "
    "multiple of 3 is to say that 15 is divisible by 3. That means that when "
    "we divide 3 into 15, we get a counting number. In fact, 15 ÷ 3 is 5, "
    "so 15 is 5 · 3. EXAMPLE 1.7 Find the prime factorization of 48."
)

# A worked example with circled sub-questions + numbered EXAMPLE marker.
CHUNK_WORKED_EXAMPLE = (
    "The digit 7 is in the ten-thousands place. The digit 8 is in the "
    "thousands place. EXAMPLE 1.1 In the number 63,407,218, find the place "
    "value of each digit: ⓐ7 ⓑ0 ⓒ1 ⓓ6 ⓔ3 "
    "Solution The 7 is in the ten-thousands place."
)

# A TRY IT exercise block.
CHUNK_TRY_IT = (
    "Since 3 is less than 5, we leave the 0 as is, and then replace the "
    "digits to the right with zeros. So, 100,000 is 103,978 rounded to the "
    "nearest ten thousand. Chapter 1 Foundations 11 TRY IT : : 1.9 Round "
    "206,981 to the nearest: ⓐ hundred ⓑ thousand ⓒ ten thousand."
)

# Plain prose that happens to mention a few math vocabulary words but carries
# NO worked computation — still real content (definition-style), must be kept.
CHUNK_DEFINITION_PROSE = (
    "The prime factorization of a number is the product of prime numbers that "
    "equals the number. To find the prime factorization of a composite "
    "number, find any two factors of the number and use them to create two "
    "branches. If a factor is prime, that branch is complete."
)


# ---------------------------------------------------------------------------
# Positive: contaminated chunks ARE classified front-matter.
# ---------------------------------------------------------------------------


def test_cover_authors_copyright_is_frontmatter():
    chunk = {"id": "c_00001", "text": CHUNK_00001_COVER}
    verdict = classify_frontmatter(chunk, position=0)
    assert verdict.is_frontmatter is True
    assert "copyright_attribution" in verdict.matched_signatures
    assert "author_contributor_list" in verdict.matched_signatures
    assert verdict.has_math_content is False


def test_donor_list_toc_marketing_is_frontmatter():
    chunk = {"id": "c_00002", "text": CHUNK_00002_DONOR_TOC}
    verdict = classify_frontmatter(chunk, position=1)
    assert verdict.is_frontmatter is True
    # Multi-signal: donor + marketing + TOC.
    assert "donor_acknowledgement" in verdict.matched_signatures
    assert "marketing_web_view" in verdict.matched_signatures
    assert "table_of_contents" in verdict.matched_signatures
    assert verdict.has_math_content is False


def test_key_features_preface_is_frontmatter():
    chunk = {"id": "c_00004", "text": CHUNK_00004_KEY_FEATURES}
    verdict = classify_frontmatter(chunk, position=3)
    assert verdict.is_frontmatter is True
    assert "book_features" in verdict.matched_signatures


def test_isbn_creative_commons_strong_signal_in_cover_region():
    chunk = {
        "id": "c_iso",
        "text": (
            "Print ISBN-13: 978-1-947172-26-5 This work is licensed under a "
            "Creative Commons Attribution 4.0 International License."
        ),
    }
    verdict = classify_frontmatter(chunk, position=0)
    assert verdict.is_frontmatter is True
    assert "copyright_attribution" in verdict.matched_signatures


# ---------------------------------------------------------------------------
# Negative: real algebra content is NOT classified front-matter.
# ---------------------------------------------------------------------------


def test_foundations_header_with_math_is_not_frontmatter():
    """THE false-positive guard: 'Chapter 1 Foundations' header + real math."""
    chunk = {"id": "c_math", "text": CHUNK_FOUNDATIONS_HEADER_MATH}
    # Even forced into the cover region, the math veto must win.
    verdict = classify_frontmatter(chunk, position=0)
    assert verdict.is_frontmatter is False
    assert verdict.has_math_content is True


def test_worked_example_is_not_frontmatter():
    chunk = {"id": "c_ex", "text": CHUNK_WORKED_EXAMPLE}
    verdict = classify_frontmatter(chunk, position=0)
    assert verdict.is_frontmatter is False
    assert verdict.has_math_content is True


def test_try_it_exercise_is_not_frontmatter():
    chunk = {"id": "c_try", "text": CHUNK_TRY_IT}
    verdict = classify_frontmatter(chunk, position=2)
    assert verdict.is_frontmatter is False
    assert verdict.has_math_content is True


def test_definition_prose_is_not_frontmatter():
    """Math-vocabulary prose with no marketing signatures stays kept."""
    chunk = {"id": "c_def", "text": CHUNK_DEFINITION_PROSE}
    verdict = classify_frontmatter(chunk, position=12)
    assert verdict.is_frontmatter is False


def test_bare_foundation_word_is_not_a_donor_signal():
    """A chunk whose only 'foundation' is the chapter title is not donor."""
    chunk = {
        "id": "c_fnd",
        "text": (
            "Figure 1.1 In order to be structurally sound, the foundation of "
            "a building must be carefully constructed. Foundations are the "
            "starting point for the rest of the structure. The foundation of "
            "algebra is the real number system."
        ),
    }
    verdict = classify_frontmatter(chunk, position=20)
    # Outside cover region, no math, but also no real front-matter signal:
    # the bare "foundation"/"Foundations" must NOT trip donor_acknowledgement.
    assert "donor_acknowledgement" not in verdict.matched_signatures
    assert verdict.is_frontmatter is False


def test_section_title_couplet_is_not_donor():
    """'Add and Subtract Integers' is a section title, not a donor name."""
    chunk = {
        "id": "c_sec",
        "text": (
            "This foundation page lists the chapter sections: Add and "
            "Subtract Integers, Multiply and Divide Integers, Expressions "
            "and Equations In Context."
        ),
    }
    verdict = classify_frontmatter(chunk, position=30)
    assert "donor_acknowledgement" not in verdict.matched_signatures


# ---------------------------------------------------------------------------
# Threshold semantics.
# ---------------------------------------------------------------------------


def test_single_weak_signal_outside_cover_region_is_kept():
    """One signature past the cover region needs corroboration to drop."""
    chunk = {
        "id": "c_weak",
        "text": (
            "The course covers the additional resources available to "
            "instructors who adopt this material in their classrooms."
        ),
    }
    # 'additional resources' = book_features (a SUPPORTING signal, not strong).
    verdict = classify_frontmatter(
        chunk, position=DEFAULT_COVER_REGION_CHUNKS + 5
    )
    assert verdict.is_frontmatter is False


def test_single_strong_signal_inside_cover_region_drops():
    chunk = {
        "id": "c_strong",
        "text": (
            "Table of Contents Preface 1 1 Foundations 5 1.1 Introduction to "
            "Whole Numbers 5 1.2 Use the Language of Algebra 21 1.3 Add and "
            "Subtract Integers 41 1.4 Multiply and Divide Integers 64"
        ),
    }
    verdict = classify_frontmatter(chunk, position=2)
    assert verdict.is_frontmatter is True
    assert "table_of_contents" in verdict.matched_signatures


def test_math_veto_overrides_multiple_signatures():
    """Even a chunk hitting many marketing signatures is kept if it has math."""
    chunk = {
        "id": "c_mixed",
        "text": (
            CHUNK_00002_DONOR_TOC
            + " EXAMPLE 1.5 Compute 15 ÷ 3 = 5 and verify 5 · 3 = 15."
        ),
    }
    verdict = classify_frontmatter(chunk, position=1)
    assert verdict.has_math_content is True
    assert verdict.is_frontmatter is False
    # Signatures still recorded for audit even though the chunk is kept.
    assert verdict.matched_signatures


# ---------------------------------------------------------------------------
# partition_frontmatter — ordered list partition + verdict stream.
# ---------------------------------------------------------------------------


def test_partition_drops_only_contaminated_chunks():
    chunks = [
        {"id": "c_00001", "text": CHUNK_00001_COVER},
        {"id": "c_00002", "text": CHUNK_00002_DONOR_TOC},
        {"id": "c_math", "text": CHUNK_FOUNDATIONS_HEADER_MATH},
        {"id": "c_ex", "text": CHUNK_WORKED_EXAMPLE},
        {"id": "c_def", "text": CHUNK_DEFINITION_PROSE},
    ]
    kept, dropped = partition_frontmatter(chunks)
    dropped_ids = {v.chunk_id for v in dropped}
    kept_ids = {c["id"] for c in kept}
    assert dropped_ids == {"c_00001", "c_00002"}
    assert kept_ids == {"c_math", "c_ex", "c_def"}
    # Order preserved among kept chunks.
    assert [c["id"] for c in kept] == ["c_math", "c_ex", "c_def"]


def test_partition_verdicts_carry_replayable_rationale():
    chunks = [{"id": "c_00001", "text": CHUNK_00001_COVER}]
    _, dropped = partition_frontmatter(chunks)
    assert len(dropped) == 1
    rationale = dropped[0].rationale()
    assert "c_00001" in rationale
    assert "front-matter" in rationale.lower()
    # Rationale interpolates dynamic signals (the matched signature names).
    assert "copyright_attribution" in rationale
    assert len(rationale) >= 20  # decision-capture rationale floor


def test_empty_input_partition():
    kept, dropped = partition_frontmatter([])
    assert kept == []
    assert dropped == []


def test_verdict_is_dataclass_with_expected_fields():
    v = classify_frontmatter({"id": "x", "text": "plain text"}, position=50)
    assert isinstance(v, FrontmatterVerdict)
    assert v.is_frontmatter is False
    assert v.signature_count == 0
    assert v.chunk_id == "x"
