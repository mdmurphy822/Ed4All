"""Unit tests for scripts/harness/gold_compare.py.

All fixtures are tiny synthetic chunk_v4 / HTML blobs built in-test -- no course
data is read from disk (repo rule: tracked tests must not reference course data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import gold_compare as gc  # noqa: E402


# --------------------------------------------------------------------------
# synthetic fixture builders
# --------------------------------------------------------------------------
def _chunk(cid, text, chapter, section, *, heading=""):
    return {
        "id": cid,
        "schema_version": "v4",
        "text": text,
        "source": {
            "item_path": f"synthetic-book-ch{chapter:02d}_accessible.html",
            "module_id": f"synthetic-book-ch{chapter:02d}_accessible",
            "section_heading": heading,
            "source_references": [
                {
                    "sourceId": f"semantik:synthetic-book-ch{chapter:02d}_accessible#{section}",
                    "role": "primary",
                    "extractor": "synthesized",
                }
            ],
        },
        "concept_tags": [],
    }


# distinct, shingle-rich gold sections so 5-gram containment is meaningful
SEC_A_TEXT = (
    "The counting numbers are one two three and so on used to count objects. "
    "Whole numbers add zero to the counting numbers forming a complete set."
)
SEC_B_TEXT = (
    "A variable is a letter that stands for a number we do not yet know. "
    "An algebraic expression combines variables and constants using operations."
)
SEC_C_TEXT = (
    "Prime factorization writes a composite number as a product of prime numbers. "
    "The least common multiple is the smallest shared multiple of two numbers."
)


def build_gold(tmp_path) -> Path:
    chunks = [
        _chunk("g1", SEC_A_TEXT, 1, "1-1-whole-numbers", heading="Whole Numbers"),
        _chunk("g2", SEC_B_TEXT, 1, "1-2-variables", heading="Variables"),
        _chunk("g3", SEC_C_TEXT, 1, "1-3-prime-factorization", heading="Prime Factorization"),
        _chunk("g4", SEC_A_TEXT + " Extra chapter two prose about integers and their opposites on a number line.", 2, "2-1-integers", heading="Integers"),
    ]
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# helper-level tests
# --------------------------------------------------------------------------
def test_tokenize_and_ngrams():
    toks = gc.tokenize("Hello, WORLD! 42x")
    assert toks == ["hello", "world", "42x"]
    assert gc.ngram_set(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}
    assert gc.ngram_set(["a"], 2) == set()


def test_strip_html_and_headings():
    html = "<h1>Title</h1><p>Body <b>text</b> here.</p><script>junk()</script><h2>Sub</h2>"
    text, headings = gc.strip_html(html)
    assert "junk" not in text
    assert "Body text here." in text.replace("  ", " ")
    assert (1, "Title") in headings
    assert (2, "Sub") in headings


def test_strip_html_excludes_sr_only_and_hidden():
    """Screen-reader-only / hidden a11y labels must NOT reach extracted text.

    SemantiK's gold-shell emits ``<p class="sr-only" hidden>Paragraph block</p>``
    structural labels as an accessibility surface. Counting them as content
    massively inflates the junk rate (thousands of "Paragraph block" tokens).
    """
    html = (
        "<h1>Real Heading</h1>"
        '<p id="s0" class="sr-only" hidden>Paragraph block</p>'
        "<p>Genuine body prose that is content.</p>"
        '<span class="visually-hidden">Metadata drop block</span>'
        '<div hidden>bare-hidden reveal content</div>'
        '<span aria-hidden="true">decorative label</span>'
        "<p>More real prose here.</p>"
    )
    text, headings = gc.strip_html(html)
    # a11y labels are excluded from the text projection
    assert "Paragraph block" not in text
    assert "Metadata drop block" not in text
    assert "bare-hidden reveal content" not in text
    assert "decorative label" not in text
    # genuine content + headings survive
    assert "Genuine body prose that is content." in text.replace("  ", " ")
    assert "More real prose here." in text.replace("  ", " ")
    assert (1, "Real Heading") in headings
    # a sr-only label token contributes ZERO junk against an empty gold vocab
    tokens = gc.tokenize(text)
    assert "block" not in tokens  # "Paragraph block"/"List block" labels gone


def test_chapter_from_text():
    assert gc.chapter_from_text("sample-algebra-2e-ch07_accessible.html") == 7
    assert gc.chapter_from_text("foo-ch01-bar") == 1
    assert gc.chapter_from_text("no-chapter-here") is None


def test_section_slug_and_title():
    src = {"source_references": [{"sourceId": "semantik:x#1-2-use-the-language", "role": "primary"}]}
    assert gc.section_slug_from_source(src) == "1-2-use-the-language"
    assert gc.section_title("1-2-use-the-language") == "use the language"


# --------------------------------------------------------------------------
# majority-section anchoring (FIX 4: gold chunks carry MULTIPLE refs; the FIRST
# ref lumps 40-50% of a chapter onto one anchor and poisons bleed detection)
# --------------------------------------------------------------------------
def _src_refs(*fragments, heading=""):
    """A source dict with one source_reference per given section fragment.

    ``None`` fragments produce a ref with no sourceId (skipped by the majority
    function); non-None fragments become ``semantik:algch1#<fragment>`` sourceIds.
    """
    refs = []
    for frag in fragments:
        if frag is None:
            refs.append({"role": "supporting"})
        else:
            refs.append({"sourceId": f"semantik:algch1#{frag}", "role": "supporting"})
    return {"source_references": refs, "section_heading": heading}


def test_majority_section_picks_most_frequent_fragment():
    # [1-1, 1-2, 1-2] -> majority is 1-2, NOT the first ref (1-1)
    src = _src_refs("1-1", "1-2", "1-2")
    assert gc.majority_section_slug_from_source(src) == "1-2"


def test_majority_section_tie_breaks_to_first_ref():
    # [1-1, 1-2] tie (1 each) -> first ref in document order wins
    src = _src_refs("1-1", "1-2")
    assert gc.majority_section_slug_from_source(src) == "1-1"


def test_majority_section_skips_refs_without_sourceid():
    # a ref with no sourceId is skipped; 2-3 wins the remaining majority
    src = _src_refs(None, "2-3", "2-3", None, "2-4")
    assert gc.majority_section_slug_from_source(src) == "2-3"


def test_majority_section_falls_back_to_heading_slug():
    # zero usable sourceIds -> section_heading slug (same as section_slug_from_source)
    src = _src_refs(None, None, heading="Prime Factorization")
    assert gc.majority_section_slug_from_source(src) == "prime-factorization"
    # no refs at all, no heading -> "unknown"
    assert gc.majority_section_slug_from_source({"source_references": []}) == "unknown"


def test_majority_section_handles_sourceid_without_fragment():
    # a bare sourceId (no '#') uses the whole id as its fragment
    src = {"source_references": [{"sourceId": "semantik:algch1"}, {"sourceId": "semantik:algch1"}]}
    assert gc.majority_section_slug_from_source(src) == "semantik:algch1"


def _multi_ref_chunk(cid, text, chapter, fragments):
    """A gold chunk_v4 blob carrying MULTIPLE section refs (the lumping shape)."""
    return {
        "id": cid,
        "schema_version": "v4",
        "text": text,
        "source": {
            "item_path": f"synthetic-book-ch{chapter:02d}_accessible.html",
            "module_id": f"synthetic-book-ch{chapter:02d}_accessible",
            "section_heading": "",
            "source_references": [
                {
                    "sourceId": f"semantik:synthetic-book-ch{chapter:02d}_accessible#{frag}",
                    "role": "primary" if i == 0 else "supporting",
                    "extractor": "synthesized",
                }
                for i, frag in enumerate(fragments)
            ],
        },
        "concept_tags": [],
    }


def test_load_gold_uses_majority_section(tmp_path):
    # gold chunk whose FIRST ref is 1-1 but whose MAJORITY is 1-2 -> anchored 1-2
    obj = _multi_ref_chunk("g1", SEC_A_TEXT, 1, ["1-1", "1-2", "1-2"])
    p = tmp_path / "gold.jsonl"
    p.write_text(json.dumps(obj), encoding="utf-8")
    gold = gc.load_gold(p)
    assert gold[1].chunks[0].section == "1-2"


# --------------------------------------------------------------------------
# oversized-section down-weighting in the discriminative map (FIX 4 part 2)
# --------------------------------------------------------------------------
SEC_D_TEXT = (
    "A fraction represents part of a whole and has a numerator and a denominator. "
    "Equivalent fractions name the same value using different numerator denominator pairs."
)
SEC_E_TEXT = (
    "An exponent tells how many times to multiply a base by itself in repeated form. "
    "Scientific notation writes very large numbers as a coefficient times a power of ten."
)


def test_oversized_section_shingles_excluded_from_discriminative():
    # 8-chunk chapter: section 1-1 holds 4 chunks (50% > 25%) = lumping residue;
    # its sole-owned shingles must be EXCLUDED from the discriminative map.
    chunks = [
        gc.GoldChunk("a1", SEC_A_TEXT, 1, "1-1-whole-numbers", "Whole Numbers"),
        gc.GoldChunk("a2", SEC_A_TEXT, 1, "1-1-whole-numbers", "Whole Numbers"),
        gc.GoldChunk("a3", SEC_A_TEXT, 1, "1-1-whole-numbers", "Whole Numbers"),
        gc.GoldChunk("a4", SEC_A_TEXT, 1, "1-1-whole-numbers", "Whole Numbers"),
        gc.GoldChunk("b1", SEC_B_TEXT, 1, "1-2-variables", "Variables"),
        gc.GoldChunk("c1", SEC_C_TEXT, 1, "1-3-prime-factorization", "Prime"),
        gc.GoldChunk("d1", SEC_D_TEXT, 1, "1-4-fractions", "Fractions"),
        gc.GoldChunk("e1", SEC_E_TEXT, 1, "1-5-exponents", "Exponents"),
    ]
    gch = gc.GoldChapter(chapter=1, chunks=chunks)
    gc._build_gold_caches(gch)
    owners = set(gch.discriminative.values())
    # the oversized 1-1 bucket contributes NO discriminative shingles
    assert "1-1-whole-numbers" not in owners
    # the other (right-sized) sections still own discriminative shingles
    assert "1-2-variables" in owners
    assert "1-3-prime-factorization" in owners


def test_small_chapter_single_section_not_gutted():
    # 3-chunk chapter (< OVERSIZED_SECTION_MIN_CHUNKS): the guard must NOT fire,
    # so a dominant small-chapter section keeps its discriminative shingles.
    chunks = [
        gc.GoldChunk("a1", SEC_A_TEXT, 1, "1-1-whole-numbers", "Whole Numbers"),
        gc.GoldChunk("a2", SEC_A_TEXT, 1, "1-1-whole-numbers", "Whole Numbers"),
        gc.GoldChunk("b1", SEC_B_TEXT, 1, "1-2-variables", "Variables"),
    ]
    gch = gc.GoldChapter(chapter=1, chunks=chunks)
    gc._build_gold_caches(gch)
    owners = set(gch.discriminative.values())
    assert "1-1-whole-numbers" in owners  # small-chapter guard preserved it


def test_load_gold_maps_chapters_and_sections(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    assert set(gold) == {1, 2}
    assert len(gold[1].chunks) == 3
    assert "1-1-whole-numbers" in gold[1].section_shingles
    assert gold[1].vocab  # non-empty


# --------------------------------------------------------------------------
# metric tests
# --------------------------------------------------------------------------
def test_recall_perfect_self_comparison(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    cand_text = "\n".join(c.text for c in gch.chunks)
    m = gc.metric_recall(gch, cand_text)
    assert m["recall_mean"] == 1.0
    assert m["recall_min"] == 1.0


def test_recall_drops_when_section_missing(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    # candidate only contains section A -> B and C gold chunks unrecovered
    m = gc.metric_recall(gch, SEC_A_TEXT)
    assert m["recall_mean"] < 0.5
    assert m["worst_10"][0]["recall"] == 0.0


def test_junk_rate_flags_ocr_garbage(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    clean = "\n".join(c.text for c in gch.chunks)
    assert gc.metric_junk(gch, clean)["junk_rate"] == 0.0
    dirty = clean + " qwxzptr blargh xzzzt vvvbbb nonsenseword"
    assert gc.metric_junk(gch, dirty)["junk_rate"] > 0.0


def test_junk_rate_exempts_folded_math_words(tmp_path):
    """Folded LaTeX command words are legitimate math, never junk — even when the
    GOLD reference chapter expressed no such notation (e.g. gold dropped every
    radical but the candidate correctly emits ``\\sqrt`` → folded ``sqrt``)."""
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    clean = "\n".join(c.text for c in gch.chunks)
    # sqrt/frac/cdot are in MATH_WORDS and absent from this gold's vocab.
    assert "sqrt" in gc.MATH_WORDS and "sqrt" not in gch.vocab
    mathy = clean + (" \\sqrt{2} \\cdot \\sqrt{5} = \\sqrt{10}" * 20)
    # Every added OOV-shaped token is an exempt math word → junk stays ~0.
    assert gc.metric_junk(gch, mathy)["junk_rate"] == 0.0
    # Real garbage still counts even amid the math.
    assert gc.metric_junk(gch, mathy + " qwxzptr blargh xzzzt")["junk_rate"] > 0.0


def test_mishandled_detects_mojibake_and_hyphen():
    text = "This equa- tion has a mojibake Ã© artifact and â€™ smart quote."
    m = gc.metric_mishandled(text)
    assert m["mojibake_hits"] >= 1
    assert m["broken_hyphen_hits"] >= 1
    assert m["mojibake_per_1k_chars"] > 0


def test_mishandled_garbage_wordshape():
    assert gc._looks_like_garbage("qwrtplkj")  # no vowels
    assert gc._looks_like_garbage("baaaad")     # 3+ repeat
    assert not gc._looks_like_garbage("equation")
    assert not gc._looks_like_garbage("cat")     # too short


def test_repetition_detects_duplicated_block(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    phrase = "this is a repeated twelve gram block that appears many times over again "
    cand = phrase * 6
    m = gc.metric_repetition(gch, cand, [])
    assert m["repeated_12gram_blocks"] > 0
    assert m["repeated_12gram_excess"] > 3


def test_repetition_near_duplicate_chunks(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    cc1 = gc.CandChunk("c1", SEC_A_TEXT, 1)
    cc2 = gc.CandChunk("c2", SEC_A_TEXT + " tiny", 1)  # near-identical
    m = gc.metric_repetition(gch, cc1.text + cc2.text, [cc1, cc2])
    assert m["near_duplicate_chunk_pairs"] == 1


def test_bleed_detects_cross_section_chunk(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    # a candidate chunk that fuses section A + section B text = a bleed
    bleeder = gc.CandChunk("b1", SEC_A_TEXT + " " + SEC_B_TEXT, 1)
    clean = gc.CandChunk("b2", SEC_C_TEXT, 1)
    m = gc.metric_bleed(gch, [bleeder, clean])
    assert m["bleed_candidates"] >= 1
    assert any(bl["candidate_id"] == "b1" for bl in m["bleeds"])


def test_bleed_boundary_duplication(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    shared = "The least common multiple is the smallest shared multiple of two numbers."
    c1 = gc.CandChunk("c1", "Some intro prose about numbers here. " + shared, 1)
    c2 = gc.CandChunk("c2", shared + " Then more prose continues afterward here.", 1)
    m = gc.metric_bleed(gch, [c1, c2])
    assert m["boundary_duplicate_sentences"] >= 1


def test_structure_missing_and_extra_headings(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    gch = gold[1]
    headings = [(1, "Whole Numbers"), (2, "Completely Unrelated Appendix")]
    m = gc.metric_structure(gch, headings)
    missing_titles = {x["title"] for x in m["missing_sections"]}
    assert "variables" in missing_titles or "prime factorization" in missing_titles
    assert any("Appendix" in h for h in m["extra_headings"])
    assert m["level_histogram"] == {"h1": 1, "h2": 1}


# --------------------------------------------------------------------------
# end-to-end evaluate() + verdict tests
# --------------------------------------------------------------------------
def test_evaluate_self_comparison_is_pass(tmp_path):
    gold_path = build_gold(tmp_path)
    gold = gc.load_gold(gold_path)
    cand = gc.load_candidate_chunks(gold_path, gold)  # candidate == gold
    res = gc.evaluate(gold, cand, "chunks", None)
    assert res["overall"]["overall_verdict"] == "PASS"
    assert res["overall"]["mean_recall"] == 1.0
    assert res["overall"]["mean_junk_rate"] == 0.0


def test_evaluate_missing_chapter_fails(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    cand: dict = {}  # nothing converted
    res = gc.evaluate(gold, cand, "chunks", [1])
    assert res["verdicts"]["1"]["verdict"] == "FAIL"
    assert "chapter_absent_from_candidate" in res["verdicts"]["1"]["triggers"]


def test_html_mode_chapter_from_filename(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    html = "<html><body><h1>Whole Numbers</h1><p>" + SEC_A_TEXT + "</p>" \
           "<h1>Variables</h1><p>" + SEC_B_TEXT + "</p>" \
           "<h1>Prime Factorization</h1><p>" + SEC_C_TEXT + "</p></body></html>"
    d = tmp_path / "cand_html"
    d.mkdir()
    (d / "synthetic-book-ch01_accessible.html").write_text(html, encoding="utf-8")
    cand = gc.load_candidate_html(d, {})
    assert 1 in cand
    res = gc.evaluate(gold, cand, "html", [1])
    r1 = res["per_chapter"]["1"]
    assert r1["recall"]["recall_mean"] == 1.0
    assert len(r1["structure"]["missing_sections"]) == 0


def test_explicit_map_override(tmp_path):
    d = tmp_path / "html"
    d.mkdir()
    p = d / "weird-name-no-chapter.html"
    p.write_text("<h1>Whole Numbers</h1><p>" + SEC_A_TEXT + "</p>", encoding="utf-8")
    cand = gc.load_candidate_html(None, {3: p})
    assert 3 in cand


def test_fallback_chapter_assignment_by_text(tmp_path):
    gold = gc.load_gold(build_gold(tmp_path))
    # candidate chunk with NO chNN signal anywhere -> best-text-match to ch1
    obj = {
        "id": "x1",
        "text": SEC_B_TEXT,
        "source": {"item_path": "opaque.html", "module_id": "opaque", "source_references": []},
        "concept_tags": [],
    }
    p = tmp_path / "cand.jsonl"
    p.write_text(json.dumps(obj), encoding="utf-8")
    cand = gc.load_candidate_chunks(p, gold)
    # SEC_B_TEXT is unique to chapter 1 -> should land there via fallback
    assert 1 in cand
    assert cand[1].fallback_chunks == 1


def test_parse_chapters_ranges():
    assert gc.parse_chapters("1,2,5-7") == [1, 2, 5, 6, 7]
    assert gc.parse_chapters(None) is None


def test_main_smoke_writes_json(tmp_path, capsys):
    gold_path = build_gold(tmp_path)
    out = tmp_path / "report.json"
    rc = gc.main([
        "--gold", str(gold_path),
        "--candidate-chunks", str(gold_path),
        "--json-out", str(out),
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["schema_version"] == gc.SCHEMA_VERSION
    assert report["primary_mode"] == "chunks"
    assert report["overall"]["overall_verdict"] == "PASS"
    assert "generated_at" in report


# --------------------------------------------------------------------------
# math-aware folding tests (LaTeX ↔ unicode ↔ plain-text)
# --------------------------------------------------------------------------
def _tk(s):
    """fold_math + tokenize, for asserting cross-representation equality."""
    return gc.tokenize(gc.fold_math(s))


def test_fold_math_reconciles_latex_unicode_and_plain():
    # sqrt: LaTeX command and unicode radical meet at the word "sqrt"
    assert _tk(r"$\sqrt{h}$") == _tk("√h") == ["sqrt", "h"]
    # frac: structural command dropped, argument survives to match plain a/b
    assert _tk(r"\frac{a}{b}") == _tk("a/b") == ["a", "b"]
    # cdot: command and unicode middle-dot both fold to "cdot"
    assert _tk(r"a \cdot b") == _tk("a · b") == ["a", "cdot", "b"]
    # relations
    assert _tk(r"x \neq y") == _tk("x ≠ y") == ["x", "neq", "y"]
    assert _tk(r"p \geq q") == _tk("p ≥ q") == ["p", "geq", "q"]
    assert _tk(r"p \leq q") == _tk("p ≤ q") == ["p", "leq", "q"]
    # superscript folds to a digit, then digit-letter split reconciles the two
    assert _tk("x^2") == _tk("x²") == ["x", "2"]


def test_fold_math_digit_letter_fusion_normalized_both_sides():
    # the VLM fuses "9c"/"36r"; the gold spaces them -> must match after fold
    assert _tk("9c") == _tk("9 c") == ["9", "c"]
    assert _tk("36r") == _tk("36 r") == ["36", "r"]


def test_fold_math_is_noop_on_plain_english():
    for s in [SEC_A_TEXT, SEC_B_TEXT, SEC_C_TEXT,
              "The quick brown fox jumps over 12 lazy dogs."]:
        # no backslash / math-unicode / digit-letter adjacency => byte-stable
        assert gc.fold_math(s) == s
        assert gc.tokenize(gc.fold_math(s)) == gc.tokenize(s)


def test_count_math_folds_reports_per_side():
    c = gc.count_math_folds(r"\sqrt{h}=\frac{a}{b}\cdot 9c")
    assert c["latex_commands"] == 3          # \sqrt \frac \cdot
    assert c["digit_letter_fusions"] == 1    # 9c
    g = gc.count_math_folds("√h and a · b with 9 c")
    assert g["unicode_symbols"] == 2         # √ and ·
    assert g["latex_commands"] == 0
    # plain english => all zero
    assert gc.count_math_folds(SEC_A_TEXT) == {
        "latex_commands": 0, "unicode_symbols": 0, "digit_letter_fusions": 0,
        "part_markers": 0,
    }


def test_looks_like_garbage_exempts_folded_math_words():
    # folded LaTeX command words are legitimate math, never garble
    for w in ["sqrt", "frac", "cdot", "neq", "geq", "leq"]:
        assert not gc._looks_like_garbage(w)


def test_looks_like_garbage_fixes_consonant_run_false_positives():
    # 5-consonant English clusters must PASS (were falsely flagged before)
    for w in ["offspring", "strengths", "twelfths", "scratched"]:
        assert not gc._looks_like_garbage(w)


def test_looks_like_garbage_still_flags_true_garble():
    # no-vowel garble, triple-repeat, and 6+ consonant-run-with-vowel garble
    assert gc._looks_like_garbage("qwrtplkj")   # no vowel
    assert gc._looks_like_garbage("baaaad")      # 3+ repeat
    assert gc._looks_like_garbage("aqvbcdfg")    # vowel + 7-consonant run
    assert gc._looks_like_garbage("xzzzt")       # no vowel + repeat


def _build_math_gold(tmp_path) -> Path:
    """Gold chunk that writes math in UNICODE / plain text (the vendor form)."""
    text = (
        "The length is √h divided by 4 for every shape we examine here. "
        "To combine terms multiply a · b and observe that x ≠ y always holds. "
        "The fraction a/b uses 9 c over the shared base of the whole figure."
    )
    p = tmp_path / "math_gold.jsonl"
    p.write_text(json.dumps(_chunk("m1", text, 1, "1-1-radicals", heading="Radicals")),
                 encoding="utf-8")
    return p


# the SAME content the VLM fused-extraction arm emits: LaTeX + fused digits
_MATH_CAND_LATEX = (
    r"The length is $\sqrt{h}$ divided by 4 for every shape we examine here. "
    r"To combine terms multiply $a \cdot b$ and observe that $x \neq y$ always holds. "
    r"The fraction $\frac{a}{b}$ uses 9c over the shared base of the whole figure."
)


def test_junk_rate_not_inflated_by_latex_vs_unicode_gold(tmp_path):
    gold = gc.load_gold(_build_math_gold(tmp_path))
    gch = gold[1]
    m = gc.metric_junk(gch, _MATH_CAND_LATEX)
    # LaTeX control words (sqrt/cdot/neq/frac) and "9c" fold to the gold vocab,
    # so a faithful-but-LaTeX candidate is NOT junk.
    assert m["junk_rate"] == 0.0


def test_recall_recovers_math_dense_chunk(tmp_path):
    gold = gc.load_gold(_build_math_gold(tmp_path))
    gch = gold[1]
    m = gc.metric_recall(gch, _MATH_CAND_LATEX)
    # math-dense recall is fully recovered once notation is folded
    assert m["recall_mean"] == 1.0


def test_garbage_words_not_inflated_by_latex(tmp_path):
    # a candidate dense in \sqrt must not read as thousands of garbage words
    cand = (r"$\sqrt{x}$ " * 30) + "and some perfectly ordinary readable prose here."
    m = gc.metric_mishandled(cand)
    assert m["garbage_words"] == 0
    assert m["garbage_word_rate"] == 0.0


def test_garbage_words_still_detects_real_ocr_garble():
    # folding must not blunt true garble detection
    cand = "ordinary prose then qwrtplkj vvvbbb xzzzt garbled ocr output"
    m = gc.metric_mishandled(cand)
    assert m["garbage_words"] >= 3


def test_fold_math_part_markers_cross_representation():
    # circled letter (gold), parenthesized ASCII (VLM candidate), and the bare
    # plain letter all fold to the SAME token.
    assert _tk("ⓐ") == _tk("(a)") == _tk("a") == ["a"]
    assert _tk("Ⓑ") == _tk("(B)") == _tk("b") == ["b"]
    # a whole exercise part line: "ⓐ 4 ( 6 − 11 )" vs "(a) 4 (6-11)"
    assert _tk("ⓐ 4 ( 6 − 11 )") == _tk("(a) 4 (6-11)") == ["a", "4", "6", "11"]


def test_fold_math_part_markers_noop_on_plain_prose():
    # "a" is a common English word; folding "(a)" must not corrupt real words or
    # touch multi-char paren content ("(see fig)", "(6-11)").
    plain = "The answer is a whole number, so (see below) for the details here."
    assert gc.fold_math(plain) == plain  # no single-letter paren / circled char
    # a multi-letter paren group is left untouched (only single letters fold)
    assert "(6-11)" in gc.fold_math("Simplify (6-11) using the rule.")
    # counts: circled + single-letter paren markers only
    c = gc.count_math_folds("ⓐ x ⓑ y (c) z (see) (6-11)")
    assert c["part_markers"] == 3   # ⓐ ⓑ (c); NOT (see) or (6-11)


def _build_marker_gold(tmp_path) -> Path:
    """Gold exercise chunk rendering part markers as UNICODE circled letters."""
    text = (
        "In the following exercises simplify each expression completely. "
        "ⓐ 4 ( 6 − 11 ) ⓑ 2 ( 5 − 12 ) ⓒ 3 ( 8 − 15 ) using the distributive rule. "
        "Then check each answer against the provided solution key for accuracy."
    )
    p = tmp_path / "marker_gold.jsonl"
    p.write_text(json.dumps(_chunk("mk1", text, 1, "1-1-exercises", heading="Exercises")),
                 encoding="utf-8")
    return p


# the SAME exercise, but the VLM emits ASCII parenthesized markers.
_MARKER_CAND_PAREN = (
    "In the following exercises simplify each expression completely. "
    "(a) 4 (6-11) (b) 2 (5-12) (c) 3 (8-15) using the distributive rule. "
    "Then check each answer against the provided solution key for accuracy."
)


def test_recall_marker_only_difference_scores_perfect(tmp_path):
    gold = gc.load_gold(_build_marker_gold(tmp_path))
    gch = gold[1]
    m = gc.metric_recall(gch, _MARKER_CAND_PAREN)
    # the ONLY difference is marker representation (ⓐ vs (a)) -> full recall
    assert m["recall_mean"] == 1.0


def test_normalization_block_present_in_report(tmp_path):
    gold_path = _build_math_gold(tmp_path)
    gold = gc.load_gold(gold_path)
    # candidate chunk carrying the LaTeX form
    obj = {
        "id": "c1", "text": _MATH_CAND_LATEX,
        "source": {
            "item_path": "book-ch01_accessible.html", "module_id": "book-ch01",
            "source_references": [
                {"sourceId": "semantik:book-ch01#1-1-radicals", "role": "primary"}
            ],
        },
        "concept_tags": [],
    }
    cp = tmp_path / "cand.jsonl"
    cp.write_text(json.dumps(obj), encoding="utf-8")
    cand = gc.load_candidate_chunks(cp, gold)
    res = gc.evaluate(gold, cand, "chunks", [1])
    norm = res["per_chapter"]["1"]["normalization"]
    assert norm["candidate"]["latex_commands"] >= 4   # sqrt cdot neq frac
    assert norm["candidate"]["digit_letter_fusions"] >= 1  # 9c
    assert norm["gold"]["unicode_symbols"] >= 3        # √ · ≠
    # overall rollup sums the per-chapter counts
    ov_norm = res["overall"]["normalization"]
    assert ov_norm["candidate"]["latex_commands"] >= 4
    assert ov_norm["gold"]["unicode_symbols"] >= 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
