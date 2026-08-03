"""Synthetic-fixture tests for scripts/harness/structure_scorecard.py.

All fixtures are inline synthetic HTML strings -- NO reference to inputs/ or any
course-data path (those are gitignored, internal-validation only). Each scored
dimension gets a good fixture (PASS) and a violating fixture (WARN/FAIL), plus
threshold-logic and directory-aggregation coverage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import structure_scorecard as ss  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders -- minimal well-formed accessible-HTML skeletons.
# ---------------------------------------------------------------------------
def _doc(body_inner: str, *, lang="en", title="A Test Document",
         head_extra="", html_open='<html lang="en">') -> str:
    if lang is None:
        html_open = "<html>"
    elif lang != "en":
        html_open = f'<html lang="{lang}">'
    title_tag = f"<title>{title}</title>" if title is not None else ""
    return (
        "<!DOCTYPE html>\n"
        f"{html_open}\n<head>\n<meta charset='utf-8'>\n{title_tag}\n{head_extra}\n</head>\n"
        "<body>\n"
        '<a href="#main-content" class="skip-link">Skip to main content</a>\n'
        '<nav aria-label="Table of contents"><a href="#s1">Section 1</a></nav>\n'
        '<main id="main-content">\n'
        f"{body_inner}\n"
        "</main>\n</body>\n</html>\n"
    )


# A document that fails BOTH deliver (undelivered table role) and cleanliness
# (heavy LaTeX leakage) -> composite well below the FAIL band.
BAD_BODY = (
    '<h1 id="s1">T</h1>'
    '<section data-dart-block-role="table"><p>not a table</p></section>'
    + "".join(f"<p>Row {i} \\textbf{{x}} \\begin{{eq}} \\hline \\frac{{a}}{{b}}</p>" for i in range(15))
)


def _score(html):
    return ss.score_html(html)


# ---------------------------------------------------------------------------
# A fully-clean document should PASS every scored dimension.
# ---------------------------------------------------------------------------
GOOD_BODY = """
<h1 id="s1">Main Title</h1>
<h2>Section A</h2>
<p>Some clean prose with no leaked markup at all.</p>
<h3>Sub A.1</h3>
<ul><li>first</li><li>second</li></ul>
<table><tr><th>Col</th></tr><tr><td>Val</td></tr></table>
<figure><img src="d.png" alt="A descriptive alternative text for the diagram."><figcaption>Cap</figcaption></figure>
"""


def test_clean_document_passes_all():
    r = _score(_doc(GOOD_BODY))
    assert r.verdict == "PASS"
    for name in ["deliver", "heading", "landmarks", "cleanliness", "image_a11y", "language"]:
        assert r.dimensions[name].verdict == "PASS", (name, r.dimensions[name].score)


# ---------------------------------------------------------------------------
# Dimension 1: DELIVER-WHAT-YOU-DECLARE
# ---------------------------------------------------------------------------
def test_deliver_annotation_arm_delivered():
    body = (
        '<h1 id="s1">T</h1>'
        '<section data-dart-block-role="list"><ul><li>a</li></ul></section>'
        '<section data-dart-block-role="table"><table><tr><td>x</td></tr></table></section>'
    )
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.details["arm"] == "annotation"
    assert d.score == 1.0
    assert d.verdict == "PASS"


def test_deliver_annotation_arm_undelivered_fails():
    # Declares list/table/definition_list roles but bodies are just <p> -- the
    # rerender_v5 defect signature.
    body = (
        '<h1 id="s1">T</h1>'
        '<section data-dart-block-role="list"><p>a; b; c</p></section>'
        '<section data-dart-block-role="table"><p>col col</p></section>'
        '<section data-dart-block-role="definition_list"><p>term: def</p></section>'
    )
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.details["arm"] == "annotation"
    assert d.details["delivered_total"] == 0
    assert d.score == 0.0
    assert d.verdict == "FAIL"


def test_deliver_annotation_arm_partial():
    body = (
        '<h1 id="s1">T</h1>'
        '<section data-dart-block-role="list"><ul><li>a</li></ul></section>'
        '<section data-dart-block-role="list"><p>not a list</p></section>'
    )
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.score == pytest.approx(0.5)
    assert d.details["per_role"]["list"] == {"declared": 2, "delivered": 1}


def test_deliver_counts_demoted_declarations_report_only():
    # Blind-spot guard: a block honestly reconciled from a declared role to
    # paragraph carries data-dart-demoted-role. The scorecard COUNTS it (so a
    # mass demotion is visible) but never lets it into the score.
    body = (
        '<h1 id="s1">T</h1>'
        '<section data-dart-block-role="table"><table><tr><td>x</td></tr></table></section>'
        '<p data-dart-demoted-role="table">was a table, now prose debris</p>'
        '<p data-dart-demoted-role="list">was a list, now prose</p>'
    )
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.details["arm"] == "annotation"
    # The one delivered table still scores a perfect deliver rate ...
    assert d.score == 1.0
    # ... but the demotions are surfaced so the regression is not invisible.
    assert d.details["demoted_declarations"] == 2
    assert d.details["demoted_by_role"] == {"table": 1, "list": 1}


def test_deliver_generic_arm_counts_demotions():
    # No live declarations (generic arm) but demoted-role stamps present.
    body = (
        '<h1 id="s1">T</h1>'
        '<p data-dart-demoted-role="table">ex table</p>'
        '<p>clean prose</p>'
    )
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.details["arm"] == "generic"
    assert d.details["demoted_declarations"] == 1


def test_deliver_generic_arm_orphan_table_run_flagged():
    # Three consecutive pipe rows in prose with NO real <table> -> orphan shape.
    body = (
        '<h1 id="s1">T</h1>'
        "<p>| a | b |</p><p>| c | d |</p><p>| e | f |</p>"
    )
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.details["arm"] == "generic"
    assert d.details["orphan_pipe_table_runs"] >= 1
    assert d.score < 1.0


def test_deliver_generic_arm_orphan_list_run_flagged():
    body = (
        '<h1 id="s1">T</h1>'
        "<p>- one</p><p>- two</p><p>- three</p>"
    )
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.details["arm"] == "generic"
    assert d.details["orphan_list_runs"] >= 1
    assert d.score < 1.0


def test_deliver_generic_arm_no_signals_is_na():
    body = '<h1 id="s1">T</h1><p>Just prose, nothing structural.</p>'
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.details["arm"] == "generic"
    assert d.score == 1.0
    assert any("N/A" in n for n in d.notes)


def test_deliver_generic_two_isolated_pipes_not_flagged():
    # Bra-ket / conditional-probability style: isolated pipe lines, not a run.
    body = (
        '<h1 id="s1">T</h1>'
        "<p>The state |00> maps to |11> under the operator.</p>"
        "<p>We compute P(x | y) for the model.</p>"
    )
    d = _score(_doc(body)).dimensions["deliver"]
    assert d.details["orphan_pipe_table_runs"] == 0
    assert d.score == 1.0


# ---------------------------------------------------------------------------
# Dimension 2: HEADING HIERARCHY
# ---------------------------------------------------------------------------
def test_heading_clean_passes():
    body = '<h1 id="s1">T</h1><h2>A</h2><h3>B</h3><h2>C</h2>'
    d = _score(_doc(body)).dimensions["heading"]
    assert d.score == 1.0


def test_heading_multiple_h1_penalised():
    body = '<h1 id="s1">One</h1><h1>Two</h1><h2>A</h2>'
    d = _score(_doc(body)).dimensions["heading"]
    assert d.details["h1_count"] == 2
    assert d.score < 1.0


def test_heading_level_skip_penalised():
    body = '<h1 id="s1">T</h1><h2>A</h2><h4>skip</h4>'
    d = _score(_doc(body)).dimensions["heading"]
    assert d.details["level_skips"]
    assert d.score < 1.0


def test_heading_empty_and_markup_penalised():
    body = '<h1 id="s1">T</h1><h2>   </h2><h2>\\textbf{leak}</h2>'
    d = _score(_doc(body)).dimensions["heading"]
    assert d.details["empty_headings"] == 1
    assert d.details["markup_in_headings"] == 1
    assert d.score < 1.0


def test_heading_none_is_warn():
    body = "<p>no headings here</p>"
    d = _score(_doc(body)).dimensions["heading"]
    assert d.details["heading_count"] == 0
    assert d.verdict == "WARN"


# ---------------------------------------------------------------------------
# Dimension 3: LANDMARKS & NAV
# ---------------------------------------------------------------------------
def test_landmarks_clean_passes():
    d = _score(_doc('<h1 id="s1">T</h1><p>x</p>')).dimensions["landmarks"]
    assert d.score == 1.0
    assert d.details["skip_target_resolves"] is True


def test_landmarks_broken_skip_target():
    html = (
        "<!DOCTYPE html><html lang='en'><head><title>T</title></head><body>"
        '<a href="#nope" class="skip-link">Skip to main content</a>'
        '<main id="main-content"><h1 id="s1">T</h1></main></body></html>'
    )
    d = _score(html).dimensions["landmarks"]
    assert d.details["skip_target_resolves"] is False
    assert d.score < 1.0


def test_landmarks_missing_main_and_dup_ids():
    html = (
        "<!DOCTYPE html><html lang='en'><head><title>T</title></head><body>"
        '<h1 id="dup">T</h1><p id="dup">x</p></body></html>'
    )
    d = _score(html).dimensions["landmarks"]
    assert d.details["main_count"] == 0
    assert d.details["duplicate_ids"] == ["dup"]
    assert d.score < 1.0


def test_landmarks_unlabelled_nav():
    html = (
        "<!DOCTYPE html><html lang='en'><head><title>T</title></head><body>"
        '<a href="#m" class="skip-link">Skip to main content</a>'
        "<nav><a href='#s1'>x</a></nav>"
        '<main id="m"><h1 id="s1">T</h1></main></body></html>'
    )
    d = _score(html).dimensions["landmarks"]
    assert d.details["nav_labelled"] == 0
    assert d.score < 1.0


# ---------------------------------------------------------------------------
# Dimension 4: TEXT CLEANLINESS
# ---------------------------------------------------------------------------
def test_cleanliness_clean_passes():
    d = _score(_doc(GOOD_BODY)).dimensions["cleanliness"]
    assert d.verdict == "PASS"


def test_cleanliness_latex_leak_fails():
    body = '<h1 id="s1">T</h1>' + "".join(
        f"<p>Row {i} \\textbf{{bold}} \\begin{{eq}} \\hline value</p>" for i in range(20)
    )
    d = _score(_doc(body)).dimensions["cleanliness"]
    assert d.details["latex_commands"] > 0
    assert d.verdict in ("WARN", "FAIL")
    assert d.score < 0.9


def test_cleanliness_replacement_char_and_entities():
    body = '<h1 id="s1">T</h1>' + "".join(
        f"<p>bad char � and &amp;nbsp; artifact {i}</p>" for i in range(30)
    )
    d = _score(_doc(body)).dimensions["cleanliness"]
    assert d.details["replacement_chars"] > 0
    assert d.score < 1.0


def test_cleanliness_inline_math_not_penalised():
    # $..$ math is report-only (dim 6), must NOT count as cleanliness leakage.
    body = '<h1 id="s1">T</h1>' + "".join(
        f"<p>The value $\\alpha_{i} = \\frac{{1}}{{2}}$ holds.</p>" for i in range(20)
    )
    d = _score(_doc(body)).dimensions["cleanliness"]
    assert d.details["latex_commands"] == 0
    assert d.verdict == "PASS"


def test_cleanliness_isolated_pipes_not_penalised():
    body = '<h1 id="s1">T</h1>' + "".join(
        f"<p>state |0{i}> under P(x | y) probability</p>" for i in range(20)
    )
    d = _score(_doc(body)).dimensions["cleanliness"]
    assert d.details["pipe_table_lines"] == 0


def test_cleanliness_pipe_table_run_penalised():
    rows = "".join(f"<p>| a{i} | b{i} |</p>" for i in range(6))
    body = f'<h1 id="s1">T</h1>{rows}'
    d = _score(_doc(body)).dimensions["cleanliness"]
    assert d.details["pipe_table_lines"] >= 6


def test_cleanliness_long_delimited_inline_math_not_penalised():
    # A long (>300 char) but properly $-delimited answer-key span is math, not
    # visible leakage — the uncapped sequential strip must remove it entirely.
    long_math = "$" + " ".join(f"\\sqrt{{{i}}} \\approx {i}.0" for i in range(60)) + "$"
    body = '<h1 id="s1">T</h1>' + f"<p>See: {long_math} done.</p>"
    d = _score(_doc(body)).dimensions["cleanliness"]
    assert d.details["latex_commands"] == 0
    assert d.verdict == "PASS"


def test_cleanliness_display_math_and_mixed_dollars_not_penalised():
    # $$..$$ display stripped before single $..$ (sequential passes), so a $
    # adjacent to a $$ pair cannot desync and expose the enclosed commands.
    body = '<h1 id="s1">T</h1>' + "".join(
        f"<p>$$\\sqrt{{{i}}} = {i}$$ and $\\frac{{1}}{{{i+1}}}$ ok</p>" for i in range(20)
    )
    d = _score(_doc(body)).dimensions["cleanliness"]
    assert d.details["latex_commands"] == 0
    assert d.verdict == "PASS"


# ---------------------------------------------------------------------------
# Dimension 5: IMAGE / FIGURE A11Y
# ---------------------------------------------------------------------------
def test_image_good_alt_passes():
    body = '<h1 id="s1">T</h1><img src="a.png" alt="A thorough description of the chart contents.">'
    d = _score(_doc(body)).dimensions["image_a11y"]
    assert d.score == pytest.approx(1.0, abs=0.16)  # alt-ok arm dominates
    assert d.details["good_alt"] == 1


def test_image_missing_alt_fails():
    body = '<h1 id="s1">T</h1><img src="a.png">'
    d = _score(_doc(body)).dimensions["image_a11y"]
    assert d.details["missing_alt"] == 1
    assert d.verdict == "FAIL"


def test_image_trivial_alt_flagged():
    body = '<h1 id="s1">T</h1><img src="a.png" alt="image"><img src="b.png" alt="b.png">'
    d = _score(_doc(body)).dimensions["image_a11y"]
    assert d.details["trivial_alt"] == 2
    assert d.score < 0.9


def test_image_decorative_ok():
    body = '<h1 id="s1">T</h1><img src="deco.png" alt="" role="presentation">'
    d = _score(_doc(body)).dimensions["image_a11y"]
    assert d.details["decorative"] == 1
    assert d.score == pytest.approx(1.0, abs=0.16)


def test_image_none_is_na():
    d = _score(_doc('<h1 id="s1">T</h1><p>x</p>')).dimensions["image_a11y"]
    assert d.details["img_count"] == 0
    assert d.score == 1.0


def test_figure_without_caption_dinged():
    body = '<h1 id="s1">T</h1><figure><img src="a.png" alt="A long descriptive alt here."></figure>'
    d = _score(_doc(body)).dimensions["image_a11y"]
    assert d.details["figures_with_caption"] == 0
    assert d.details["figure_pair_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Dimension 6: MATH (report-only)
# ---------------------------------------------------------------------------
def test_math_report_only_not_in_composite():
    body = '<h1 id="s1">T</h1>' + "".join(f"<p>$x_{i}$</p>" for i in range(10))
    r = _score(_doc(body))
    assert r.dimensions["math"].scored is False
    assert r.dimensions["math"].details["raw_math_runs"] > 0
    # composite should still be high (report-only math does not drag it down)
    assert r.composite > 0.9


def test_math_mathml_present():
    body = '<h1 id="s1">T</h1><math><mi>x</mi></math>'
    d = _score(_doc(body)).dimensions["math"]
    assert d.details["mathml_elements"] == 1


# ---------------------------------------------------------------------------
# Dimension 8: PEDAGOGY (report-only) -- pedagogical-unit tally
# ---------------------------------------------------------------------------
# LEGACY-COMPAT: these fixtures annotate with ``data-dart-unit`` to exercise the
# scorecard's legacy dual-read fallback; live SemantiK output uses
# ``data-semantik-unit`` (covered by test_pedagogy_counts_semantik_units below).
def test_pedagogy_counts_semantik_units_primary():
    body = (
        '<h1 id="s1">T</h1>'
        '<section data-semantik-unit="worked_example"><p>a</p></section>'
        '<section data-semantik-unit="exercise_set"><p>b</p></section>'
    )
    d = _score(_doc(body)).dimensions["pedagogy"]
    assert d.details["semantik_unit_total"] == 2
    assert d.details["semantik_units_by_type"] == {
        "exercise_set": 1,
        "worked_example": 1,
    }


def test_pedagogy_counts_dart_units():
    body = (
        '<h1 id="s1">T</h1>'
        '<section data-dart-unit="section_opener"><p>a</p></section>'
        '<section data-dart-unit="worked_example"><p>b</p></section>'
        '<section data-dart-unit="worked_example"><p>c</p></section>'
        '<section data-dart-unit="worked_example"><p>d</p></section>'
        '<section data-dart-unit="exercise_set"><p>e</p></section>'
        '<section data-dart-unit="section_opener"><p>f</p></section>'
    )
    d = _score(_doc(body)).dimensions["pedagogy"]
    assert d.scored is False
    assert d.verdict == "REPORT"
    assert d.details["semantik_unit_total"] == 6
    assert d.details["distinct_unit_types"] == 3
    assert d.details["semantik_units_by_type"] == {
        "exercise_set": 1,
        "section_opener": 2,
        "worked_example": 3,
    }
    # by-type keys are emitted in sorted order for determinism.
    assert list(d.details["semantik_units_by_type"]) == sorted(d.details["semantik_units_by_type"])


def test_pedagogy_empty_and_whitespace_units_skipped():
    body = (
        '<h1 id="s1">T</h1>'
        '<section data-dart-unit="">skipped</section>'
        '<section data-dart-unit="   ">also skipped</section>'
        '<section data-dart-unit="worked_example">kept</section>'
    )
    d = _score(_doc(body)).dimensions["pedagogy"]
    assert d.details["semantik_unit_total"] == 1
    assert d.details["semantik_units_by_type"] == {"worked_example": 1}


def test_pedagogy_no_units_is_noop_not_penalty():
    # An exemplar-shaped doc with NO data-dart-unit annotations: empty count
    # map, still report-only, and the notes announce the facet is N/A.
    body = '<h1 id="s1">T</h1><p>Just prose, nothing pedagogical.</p>'
    r = _score(_doc(body))
    d = r.dimensions["pedagogy"]
    assert d.scored is False
    assert d.verdict == "REPORT"
    assert d.details["semantik_unit_total"] == 0
    assert d.details["semantik_units_by_type"] == {}
    assert d.details["distinct_unit_types"] == 0
    assert any("N/A" in n for n in d.notes)


def test_pedagogy_report_only_does_not_alter_composite():
    # Adding pedagogy must not change the scored composite: it is excluded from
    # the weights and from the composite denominator. Prove the composite equals
    # the weighted mean of the SCORED dims alone.
    r = _score(_doc(GOOD_BODY))
    assert r.dimensions["pedagogy"].scored is False
    assert "pedagogy" not in ss.CONFIG["weights"]

    weights = ss.CONFIG["weights"]
    num = sum(w * r.dimensions[n].score for n, w in weights.items() if r.dimensions[n].scored)
    den = sum(w for n, w in weights.items() if r.dimensions[n].scored)
    assert r.composite == pytest.approx(num / den)


def test_pedagogy_never_drags_composite_down():
    # A doc that carries data-dart-unit annotations still gets a PASS/FAIL
    # composite driven only by the scored dims -- pedagogy's placeholder score
    # is non-load-bearing.
    body = GOOD_BODY + '<section data-dart-unit="worked_example"><p>x</p></section>'
    r = _score(_doc(body))
    assert r.dimensions["pedagogy"].details["semantik_unit_total"] == 1
    assert r.verdict == "PASS"
    # The pedagogy dim is present but absent from the scored-weight config.
    assert "pedagogy" not in ss.CONFIG["weights"]


def test_pedagogy_in_report_order_not_scored_order():
    assert "pedagogy" in ss._REPORT_ORDER
    assert "pedagogy" not in ss._SCORED_ORDER


# ---------------------------------------------------------------------------
# Dimension 7: LANGUAGE / META
# ---------------------------------------------------------------------------
def test_language_clean_passes():
    d = _score(_doc('<h1 id="s1">T</h1>')).dimensions["language"]
    assert d.score == 1.0


def test_language_missing_lang_and_title():
    html = (
        "<!DOCTYPE html><html><head></head><body>"
        '<main id="m"><h1 id="s1">T</h1></main></body></html>'
    )
    d = _score(html).dimensions["language"]
    assert d.details["lang"] in (None, "")
    assert d.details["title_present"] is False
    assert d.score == 0.0


# ---------------------------------------------------------------------------
# Threshold / verdict logic
# ---------------------------------------------------------------------------
def test_verdict_bands():
    assert ss._verdict(0.95) == "PASS"
    assert ss._verdict(0.90) == "PASS"
    assert ss._verdict(0.80) == "WARN"
    assert ss._verdict(0.70) == "WARN"
    assert ss._verdict(0.50) == "FAIL"


def test_composite_excludes_report_only_dims():
    # math + meta-desc/author never enter the weighted composite denominator.
    _score(_doc(GOOD_BODY))
    weights = ss.CONFIG["weights"]
    assert "math" not in weights
    assert set(weights) <= {"deliver", "heading", "landmarks", "cleanliness", "image_a11y", "language"}


# ---------------------------------------------------------------------------
# Directory aggregation + CLI
# ---------------------------------------------------------------------------
def test_directory_aggregation(tmp_path):
    good = tmp_path / "good.html"
    bad = tmp_path / "bad.html"
    good.write_text(_doc(GOOD_BODY), encoding="utf-8")
    bad.write_text(_doc(BAD_BODY), encoding="utf-8")
    files = ss._collect_html(tmp_path)
    assert len(files) == 2
    results = [ss.score_file(p) for p in files]
    agg = ss.aggregate(results)
    assert agg["file_count"] == 2
    assert agg["verdict_counts"]["PASS"] >= 1
    assert agg["verdict_counts"]["FAIL"] >= 1
    assert agg["worst_files"][0]["path"].endswith("bad.html")


def test_cli_json_out_and_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.html"
    bad.write_text(
        _doc(BAD_BODY),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    rc = ss.main(["--html", str(bad), "--json-out", str(out)])
    assert rc == 1  # a FAIL file -> non-zero exit when not a baseline run
    report = json.loads(out.read_text())
    assert report["tool"] == "structure_scorecard"
    assert report["files"][0]["dimensions"]["deliver"]["verdict"] == "FAIL"


def test_cli_baseline_never_fails_process(tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text(
        _doc(BAD_BODY),
        encoding="utf-8",
    )
    rc = ss.main(["--html", str(bad), "--baseline"])
    assert rc == 0  # baseline reports the gap but never fails the process


def test_cli_missing_target(tmp_path, capsys):
    rc = ss.main(["--html", str(tmp_path / "nope")])
    assert rc == 2
