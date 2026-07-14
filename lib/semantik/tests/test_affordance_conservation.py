"""T0 — SEMANTIK_AFFORDANCE_GATE regression net.

Covers, per check class: fires on a synthetic zero-emit-with-evidence pair; is
SILENT on 0-evidence/0-emit and on a healthy pair; FIRES on an over-emit
(fabrication) in BOTH count checks; the false-positive regressions (currency
prose, a headerless markdown grid, an OCR pipe-dash garble); the byte-identical
off-path; ``not_evaluated`` never folding to a green sidecar; and — the
load-bearing part — the RELATIONAL invariants against a real pre-fix conversion
+ a real non-math control corpus.

The real-artifact lanes are OPERATOR-POINTED and SLUG-AGNOSTIC (env vars +
glob discovery, see below) and SKIP when unset — tracked test code never names a
course or carries a machine-absolute path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lib.semantik.affordance_conservation import (
    _math_evidence,
    audit_affordances,
    resolve_affordance_gate_mode,
    resolve_section_recall_min,
)

# ---------------------------------------------------------------------------
# REAL-ARTIFACT lanes — operator-pointed, SLUG-AGNOSTIC.
#
# House doctrine: tracked code NEVER hardcodes a course slug and never carries a
# machine-absolute path. So the real-artifact lanes are pointed at by env vars
# and the artifact BASE is DISCOVERED by glob (``*_accessible.cascade_ir.json``),
# never named. Unset / absent → the lane SKIPS (the CI shape).
#
#   SEMANTIK_AFFORDANCE_KNOWN_BAD_DIR  a dir holding a PRE-fix conversion
#                                      (source notation shipped, no <math>/<th>)
#   SEMANTIK_AFFORDANCE_CONTROL_DIR    a dir holding a NON-MATH, NON-TABLE
#                                      conversion (the domain-agnosticism control)
#
# Assertions are RELATIONAL INVARIANTS (evidence>0 & emitted==0 → critical), not
# corpus-specific magic numbers — those would only hold for one book, and pinning
# them here is what would smuggle a corpus dependency into a domain-agnostic gate.
# ---------------------------------------------------------------------------
def _discover(env: str) -> Path | None:
    raw = (os.environ.get(env) or "").strip()
    if not raw:
        return None
    root = Path(raw)
    if not root.is_dir():
        return None
    hits = sorted(root.glob("*_accessible.cascade_ir.json"))
    if not hits:
        return None
    base = hits[0].with_suffix("")  # strip .json -> ..._accessible.cascade_ir
    base = base.with_suffix("")  # strip .cascade_ir -> ..._accessible
    return base if base.with_suffix(".html").is_file() else None


_KNOWN_BAD = _discover("SEMANTIK_AFFORDANCE_KNOWN_BAD_DIR")
_CONTROL = _discover("SEMANTIK_AFFORDANCE_CONTROL_DIR")


def _ir(*texts: str) -> dict:
    return {
        "region_provenance": [
            {"region_index": i, "region_kind": "paragraph", "raw_text": t}
            for i, t in enumerate(texts)
        ]
    }


def _page(body: str, *, title: str = "Algebra Basics One") -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


def _load(base: Path) -> tuple[dict, str]:
    ir = json.loads(base.with_suffix(".cascade_ir.json").read_text(encoding="utf-8"))
    html = base.with_suffix(".html").read_text(encoding="utf-8")
    return ir, html


# ---------------------------------------------------------------------------
# Flag resolution — default OFF.
# ---------------------------------------------------------------------------
def test_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_AFFORDANCE_GATE", raising=False)
    assert resolve_affordance_gate_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_gate_truthy_set(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_AFFORDANCE_GATE", val)
    assert resolve_affordance_gate_mode() is True


@pytest.mark.parametrize("val", ["", "0", "false", "off", "garbage"])
def test_gate_falsey_and_garbage_off(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_AFFORDANCE_GATE", val)
    assert resolve_affordance_gate_mode() is False


def test_section_recall_min_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("SEMANTIK_AFFORDANCE_SECTION_RECALL_MIN", raising=False)
    assert resolve_section_recall_min() == 0.80
    for bad in ("", "abc", "nan", "-1", "2.0", "inf"):
        monkeypatch.setenv("SEMANTIK_AFFORDANCE_SECTION_RECALL_MIN", bad)
        assert resolve_section_recall_min() == 0.80
    monkeypatch.setenv("SEMANTIK_AFFORDANCE_SECTION_RECALL_MIN", "0.5")
    assert resolve_section_recall_min() == 0.5


# ---------------------------------------------------------------------------
# math — CLASS-B (missing transform).
# ---------------------------------------------------------------------------
def test_math_fires_on_zero_emit_with_evidence():
    """Evidence in the source, ZERO <math> emitted, no emitter self-report."""
    ir = _ir(r"Simplify $\frac{3}{4}$ and then $x^2 + 1$.")
    report = audit_affordances(ir, _page(r"<p>Simplify $\frac{3}{4}$ and then $x^2 + 1$.</p>"))
    math = report["checks"]["math"]
    assert math["verdict"] == "fail"
    assert math["severity"] == "critical"
    assert math["evidence_convertible"] == 2
    assert math["emitted_math_elements"] == 0
    assert report["document_passed"] is False


def test_math_silent_on_zero_evidence_zero_emit():
    ir = _ir("Plain prose with no mathematics at all.")
    report = audit_affordances(ir, _page("<p>Plain prose with no mathematics at all.</p>"))
    assert report["checks"]["math"]["verdict"] == "pass"
    assert report["checks"]["math"]["severity"] is None


def test_math_silent_on_healthy_pair():
    ir = _ir(r"Simplify $\frac{3}{4}$.")
    html = _page('<p>Simplify <math display="inline" alttext="\\frac{3}{4}"><mrow/></math>.</p>')
    report = audit_affordances(ir, html)
    assert report["checks"]["math"]["verdict"] == "pass"
    assert report["document_passed"] is True


def test_math_zero_emit_with_self_report_is_documented_refusal_not_critical():
    """The emitter RAN and declined everything → a loud WARNING, never a CRITICAL.

    A fail-soft decline is a documented refusal; only a pass that never RAN is
    the CLASS-B defect. This is the branch that keeps a correct fail-soft render
    from failing its own gate.
    """
    ir = _ir(r"Simplify $x^2$.")  # convertible evidence...
    report = audit_affordances(
        ir,
        _page(r"<p>Simplify $x^2$.</p>"),  # ...but no <math> emitted...
        # ...and the emitter says it ran and declined.
        emitter_report={"math_spans_converted": 0, "math_spans_declined": 1},
    )
    math = report["checks"]["math"]
    assert math["evidence_convertible"] == 1
    assert math["emitted_math_elements"] == 0
    assert math["verdict"] == "refused"
    assert math["severity"] == "warning"
    # A refusal is loud but must NOT fail the document closed.
    assert math["severity"] != "critical"


# ---------------------------------------------------------------------------
# table_headers — CLASS-A (destruction).
# ---------------------------------------------------------------------------
# The production shape: VLM fusion FLATTENS the pipe grid onto ONE line (verified
# against the real ch01 raw_text), which is what makes the separator row's
# `R*n_cols + (R-1)` token arithmetic the load-bearing topology signal.
_PIPE_TABLE = (
    "| Expression | Words | | --- | --- | | 3 + 4 | the sum | "
    "| 5 - 2 | the difference |"
)


def test_table_headers_fires_on_zero_emit_with_evidence():
    ir = _ir(_PIPE_TABLE)
    # The separator row was DESTROYED before parse_table ran -> headerless soup.
    html = _page("<table><tr><td>Expression</td><td>Words</td></tr></table>")
    report = audit_affordances(ir, html)
    check = report["checks"]["table_headers"]
    assert check["evidence_reconstructible_tables"] == 1
    assert check["emitted_thead_blocks"] == 0
    assert check["verdict"] == "fail"
    assert check["severity"] == "critical"


def test_table_headers_silent_on_healthy_pair():
    ir = _ir(_PIPE_TABLE)
    html = _page(
        "<table><thead><tr><th scope='col'>Expression</th>"
        "<th scope='col'>Words</th></tr></thead>"
        "<tbody><tr><td>3 + 4</td><td>the sum</td></tr></tbody></table>"
    )
    report = audit_affordances(ir, html)
    assert report["checks"]["table_headers"]["verdict"] == "pass"


def test_table_headers_over_emit_is_flagged():
    """A <thead> with NO admissible evidence — a fabricated header misleads AT."""
    ir = _ir("Just prose, no table declared.")
    html = _page("<table><thead><tr><th>Fabricated</th></tr></thead></table>")
    check = audit_affordances(ir, html)["checks"]["table_headers"]
    assert check["verdict"] == "over_emit"
    assert check["severity"] == "warning"


# ---------------------------------------------------------------------------
# FALSE-POSITIVE regressions (the domain-agnosticism guarantees).
# ---------------------------------------------------------------------------
def test_currency_prose_yields_zero_math_evidence_and_zero_residue():
    """'costs $5 and $10' must NOT be read as math.

    A naive ``\\$[^$]+\\$`` counter false-fires here; the reused guards
    (``(?<!\\\\)`` lookbehind + ``_is_math_content``) do not.
    """
    text = "The item costs $5 and the other costs $10. Tickets are $5 to enter and $3 for parking."
    ir = _ir(text)
    report = audit_affordances(ir, _page(f"<p>{text}</p>"))
    assert report["checks"]["math"]["evidence_convertible"] == 0
    assert report["checks"]["math"]["severity"] is None
    assert report["checks"]["residue"]["math_source_spans"] == 0
    assert report["checks"]["residue"]["verdict"] == "pass"
    assert report["document_passed"] is True


def test_headerless_markdown_grid_yields_zero_table_evidence():
    """No separator row -> no topology declaration -> zero evidence, no fire."""
    ir = _ir("| a | b |\n| c | d |")
    report = audit_affordances(ir, _page("<table><tr><td>a</td><td>b</td></tr></table>"))
    check = report["checks"]["table_headers"]
    assert check["evidence_reconstructible_tables"] == 0
    assert check["severity"] is None


def test_ocr_garble_pipe_dash_run_is_not_table_evidence():
    """A spurious pipe-dash-pipe run in OCR prose is refused by the topology
    parser — which is exactly why the evidence predicate is the emitter's own
    accept and not ``has_separator_row``. (Measured: 1 such block on the real
    non-table medical scan.)"""
    ir = _ir("OECD (2013a) Skills Outlook | --- | garbled ocr tail | --- |")
    check = audit_affordances(ir, _page("<p>prose</p>"))["checks"]["table_headers"]
    assert check["evidence_reconstructible_tables"] == 0
    assert check["severity"] is None


def test_converted_math_alttext_and_annotation_are_not_residue():
    """A CORRECT <math> carries its LaTeX twice by design (alttext + x-tex
    annotation). Counting those as residue would report a perfect document as
    broken."""
    html = _page(
        '<p><math display="inline" alttext="\\frac{1}{2}"><semantics><mrow/>'
        '<annotation encoding="application/x-tex">\\frac{1}{2}</annotation>'
        "</semantics></math></p>"
    )
    check = audit_affordances(_ir(r"$\frac{1}{2}$"), html)["checks"]["residue"]
    assert check["math_source_spans"] == 0
    assert check["latex_control_words"] == 0
    assert check["verdict"] == "pass"


# ---------------------------------------------------------------------------
# residue / title_purity / sections / images.
# ---------------------------------------------------------------------------
def test_residue_critical_when_transform_never_ran():
    ir = _ir(r"$\frac{3}{4}$")
    check = audit_affordances(ir, _page(r"<p>$\frac{3}{4}$</p>"))["checks"]["residue"]
    assert check["verdict"] == "fail"
    assert check["severity"] == "critical"
    assert check["math_source_spans"] == 1
    assert check["residue_convertible"] == 1
    assert check["transform_ran"] is False
    assert "never ran" in check["message"]


def test_residue_declined_span_is_warning_not_critical_WITHOUT_a_self_report():
    """A span the fail-soft emitter DECLINES ships verbatim BY DESIGN — and the
    RESIDUE ITSELF says so.

    D1 regression. The verdict is adjudicated per-span by the emitter's OWN accept
    predicate (``latex_to_mathml``), so it needs NO ``emitter_report`` at all. The
    old code inferred "did the transform run?" purely from the presence of the
    in-memory self-report and fired a CRITICAL without it.
    """
    ir = _ir(r"$\begin{array}{c} 1 \end{array}$")
    check = audit_affordances(
        ir, _page(r"<p>$\begin{array}{c} 1 \end{array}$</p><math><mi>x</mi></math>")
    )["checks"]["residue"]
    assert check["verdict"] == "declined_verbatim"
    assert check["severity"] == "warning"
    assert check["residue_convertible"] == 0
    assert check["residue_declined"] == 1


def test_residue_never_claims_the_transform_never_ran_over_emitted_math():
    """D1: the factually-false message. ``<math>`` in the HTML is DISPOSITIVE
    proof the transform ran, and the message must never contradict it."""
    ir = _ir(r"$\frac{3}{4}$")
    html = _page(r"<p>$\frac{3}{4}$</p>" + "<math><mi>x</mi></math>" * 50)
    check = audit_affordances(ir, html)["checks"]["residue"]
    assert check["transform_ran"] is True
    assert check["emitted_math_elements"] == 50
    assert "never ran" not in check["message"]
    # It IS still critical — a CONVERTIBLE span shipped literal is attrition —
    # but for the true reason: the transform ran and dropped it.
    assert check["severity"] == "critical"
    assert "DID run" in check["message"]


def test_residue_is_not_laundered_by_an_unrelated_decline_count():
    """D3: ``len(residue) <= declined`` was a COUNT compare masquerading as an
    identity compare — one unrelated decline could launder a genuinely-dropped
    span. Adjudication is now per-span, so a CONVERTIBLE span stays critical no
    matter what the self-report claims."""
    ir = _ir(r"$\frac{3}{4}$")
    check = audit_affordances(
        ir,
        _page(r"<p>$\frac{3}{4}$</p><math><mi>x</mi></math>"),
        emitter_report={"math_spans_converted": 100, "math_spans_declined": 99},
    )["checks"]["residue"]
    assert check["residue_convertible"] == 1
    assert check["severity"] == "critical"


def test_title_purity_flags_running_header_furniture():
    html = _page("<p>x</p>", title="Chapter 1 Foundations 55 ✓ Solution")
    check = audit_affordances(_ir("x"), html)["checks"]["title_purity"]
    assert check["verdict"] == "impure"
    assert check["severity"] == "warning"
    assert check["repair_delta"] is True
    assert check["sanitized"] == "Chapter 1 Foundations"


def test_title_purity_folio_arm_catches_the_long_tail_the_sanitizer_refuses():
    """Arm A (repair-delta) deliberately refuses a LONG furniture tail; arm B
    (a surviving folio token) is why that case is still caught."""
    title = "Chapter 1 Foundations 73 73 Be careful to get a and b in the right order!"
    check = audit_affordances(_ir("x"), _page("<p>x</p>", title=title))["checks"][
        "title_purity"
    ]
    assert check["repair_delta"] is False  # the sanitizer alone MISSES it
    assert check["folio_token"] is True  # arm B catches it
    assert check["verdict"] == "impure"


def test_title_purity_silent_on_clean_title():
    check = audit_affordances(_ir("x"), _page("<p>x</p>", title="Elementary Algebra 2e"))[
        "checks"
    ]["title_purity"]
    assert check["verdict"] == "pass"
    assert check["severity"] is None


def test_title_purity_sibling_disagreement_is_an_unwired_seam():
    check = audit_affordances(
        _ir("x"),
        _page("<p>x</p>", title="Chapter 1 Foundations"),
        sibling_title="Chapter 1 Whole Numbers",
    )["checks"]["title_purity"]
    assert check["sibling_agrees"] is False
    assert check["verdict"] == "impure"


def test_images_fires_when_candidates_exist_and_zero_img_shipped():
    """Evidence > 0 with ZERO <img> is MEASURED destruction — it must fire."""
    ir = {
        "region_provenance": [
            {"region_index": 0, "region_kind": "figure", "raw_text": ""},
        ]
    }
    check = audit_affordances(ir, _page("<p>no img</p>"))["checks"]["images"]
    assert check["verdict"] == "fail"
    assert check["severity"] == "warning"
    assert check["evidence_figure_regions"] == 1


def test_images_zero_evidence_is_shadow_and_says_it_is_not_an_all_clear():
    """The honest limit: a page-raster scan has NO sub-page image objects, so the
    evidence is ~0 by construction and a REAL figure gap is invisible."""
    check = audit_affordances(_ir("x"), _page("<p>no img</p>"))["checks"]["images"]
    assert check["verdict"] == "shadow"
    assert check["severity"] is None
    assert check["declared_intentional_drops"]  # the limit is STATED, not implied
    assert "NOT an all-clear" in check["message"]


# ---------------------------------------------------------------------------
# D2 — FABRICATION must be visible. Both count checks are TWO-SIDED.
# ---------------------------------------------------------------------------
def test_math_over_emit_against_zero_evidence_is_flagged():
    """D2: 501 <math> against ~zero convertible evidence used to return
    verdict='pass'. `latex_to_mathml(" and ")` really does emit
    <mi>a</mi><mi>n</mi><mi>d</mi>, so this axis DOES fabricate."""
    report = audit_affordances(
        _ir("plain prose with no math at all"),
        _page("<math><mi>a</mi></math>" * 501),
    )
    check = report["checks"]["math"]
    assert check["verdict"] == "over_emit"
    assert check["severity"] == "warning"
    assert check["emitted_math_elements"] == 501
    assert check["evidence_convertible"] == 0


def test_math_over_emit_against_nonzero_evidence_is_flagged():
    """Symmetric with _check_table_headers' over_emit arm."""
    check = audit_affordances(
        _ir(r"$\frac{3}{4}$"),
        _page("<math><mi>a</mi></math>" * 9),
    )["checks"]["math"]
    assert check["verdict"] == "over_emit"
    assert check["severity"] == "warning"
    assert "FABRICATED" in check["message"]


def test_math_conserved_pair_is_silent_no_over_emit_false_positive():
    """The over-emit arm is a COUNT arm: on a conserved document it is silent."""
    check = audit_affordances(
        _ir(r"$\frac{3}{4}$"), _page("<math><mi>x</mi></math>")
    )["checks"]["math"]
    assert check["verdict"] == "pass"
    assert check["severity"] is None


# ---------------------------------------------------------------------------
# #59 — the EVIDENCE side must REUSE the render's segmentation, never race it.
#
# The count arm FALSE-FIRED "FABRICATED" on a clean chapter (3,884 "convertible"
# vs 3,906 emitted <math>) because the evidence side re-segmented the RAW IR text
# with its own _MATH_SPAN_ANGLE_RE loop, while the render converts the ASSEMBLED
# BLOCK HTML (post sanitize_body_latex + wrap_bare_math + sanitize_math_spans)
# and adjudicates with the document prose_vocab. A second segmenter, drifting.
#
# These are RELATIONAL invariants on the MECHANISM (no corpus, no magic number).
# ---------------------------------------------------------------------------
def test_bare_math_the_render_wraps_is_counted_as_evidence():
    """THE #59 REGRESSION. ``wrap_bare_math`` wraps BARE (un-delimited) math into
    ``$…$`` at render time, so the render emits a <math> for it. The old raw-text
    rescan saw ZERO spans there (no ``$`` to find) → emitted(1) > convertible(0)
    → a phantom "1 FABRICATED". Nothing was fabricated."""
    from lib.semantik.math_fold import _MATH_SPAN_ANGLE_RE

    bare = r"The result is \sqrt{5} \approx 2.236 for this case."
    # Precondition: the OLD raw rescan is blind to this span -- that IS the bug.
    assert len(_MATH_SPAN_ANGLE_RE.findall(bare)) == 0

    spans, convertible = _math_evidence([bare])
    assert convertible == 1, "the render WRAPS this bare run; evidence must see it"
    assert spans == 1
    # End to end: one <math> against that evidence is CONSERVED, not fabricated.
    check = audit_affordances(_ir(bare), _page("<math><mi>x</mi></math>"))["checks"]["math"]
    assert check["verdict"] == "pass"
    assert check["severity"] is None


def test_variable_product_letter_run_is_not_a_fabrication():
    """``<mi>a</mi><mi>b</mi>`` is the CORRECT MathML for the product ``ab``. A
    pure letter-run span is a legitimate variable product, not prose mis-wrapped
    as math, and must land in the denominator (else it reads as an over-emit)."""
    spans, convertible = _math_evidence([r"Simplify the product $ab$ and $cd$ here."])
    assert (spans, convertible) == (2, 2)


def test_declared_drop_region_math_is_not_conservable_evidence():
    """A ``metadata_drop`` region (running header / folio) is destroyed BY DESIGN
    and can NEVER reach the learner page, so its spans were never conservable.
    Counting them manufactures a phantom shortfall (the mirror of the D2 sin).

    This NARROWS the denominator, so it can never LAUNDER an over-emit."""
    ir = {
        "region_provenance": [
            {"region_index": 0, "region_kind": "paragraph", "raw_text": r"$\frac{3}{4}$"},
            {"region_index": 1, "region_kind": "metadata_drop", "raw_text": r"$\frac{9}{8}$"},
        ]
    }
    check = audit_affordances(ir, _page("<math><mi>x</mi></math>"))["checks"]["math"]
    # ONLY the paragraph region is conservable -> 1 evidence, 1 emitted -> silent.
    assert check["evidence_convertible"] == 1
    assert check["verdict"] == "pass"


def test_over_emit_still_fires_after_the_segmentation_fix():
    """THE OTHER HALF OF THE FIX. Silencing the false positive must NOT disable
    the check — that would be a regression dressed as a fix. A genuinely
    fabricating render (501 <math> against ONE evidence span, the D2 case) must
    still be caught."""
    check = audit_affordances(
        _ir(r"The value is $x+1$ exactly."),
        _page("<math><mi>a</mi></math>" * 501),
    )["checks"]["math"]
    assert check["verdict"] == "over_emit"
    assert check["severity"] == "warning"
    assert check["emitted_math_elements"] == 501
    assert check["evidence_convertible"] == 1
    assert "FABRICATED" in check["message"]


# ---------------------------------------------------------------------------
# D4 — A GATE THAT CANNOT EVALUATE MUST NEVER CONTRIBUTE A GREEN.
# ---------------------------------------------------------------------------
def test_empty_ir_is_not_evaluated_and_poisons_document_passed_to_none():
    """D4 — THE founding sin, one layer up. No region universe = NO MEASUREMENT.
    It must NOT fold to document_passed=True."""
    report = audit_affordances({"region_provenance": []}, _page("<p>hi</p>"))
    assert report["checks"]["math"]["verdict"] == "not_evaluated"
    assert report["checks"]["table_headers"]["verdict"] == "not_evaluated"
    # Tri-state: None is neither a pass nor a fabricated failure.
    assert report["document_passed"] is None
    assert report["document_passed"] is not True
    assert report["not_evaluated_count"] == 2
    assert sorted(report["not_evaluated_checks"]) == ["math", "table_headers"]
    codes = {i["code"] for i in report["issues"]}
    assert "AFFORDANCE_MATH_NOT_EVALUATED" in codes
    assert "AFFORDANCE_TABLE_HEADERS_NOT_EVALUATED" in codes


def test_not_evaluated_de_certifies_the_quality_sidecar():
    """D4 at the ARTIFACT boundary — the layer an operator actually reads. The
    pre-fix fold wrote quality_score 1.0 / certified over a gate that never
    looked."""
    from MCP.tools.pipeline_tools import _apply_affordance_to_quality  # noqa: PLC0415

    report = audit_affordances({"region_provenance": []}, _page("<p>hi</p>"))
    quality = {
        "quality_score": 1.0,
        "compliant": True,
        "wcag_status": "passed",
        "certification_status": "certified",
        "issues": [],
        "flags": [],
    }
    _apply_affordance_to_quality(quality, report)

    assert quality["certification_status"] == "not_certified"
    assert "AFFORDANCE_NOT_EVALUATED" in quality["flags"]
    assert quality["affordance_conservation"]["document_passed"] is None
    assert quality["affordance_conservation"]["not_evaluated_checks"]
    # An unlooked gate has no evidence for a RED either — inventing one is the
    # mirror image of inventing a green (the D1 sin). It removes the VOUCH only.
    assert quality["quality_score"] == 1.0
    assert quality["compliant"] is True


def test_critical_de_certifies_and_zeroes_the_quality_score():
    from MCP.tools.pipeline_tools import _apply_affordance_to_quality  # noqa: PLC0415

    report = audit_affordances(_ir(r"$\frac{3}{4}$"), _page(r"<p>$\frac{3}{4}$</p>"))
    quality = {
        "quality_score": 1.0,
        "compliant": True,
        "certification_status": "certified",
        "issues": [],
        "flags": [],
    }
    _apply_affordance_to_quality(quality, report)
    assert quality["certification_status"] == "not_certified"
    assert quality["compliant"] is False
    assert quality["quality_score"] == 0.0
    assert "AFFORDANCE_CONSERVATION_FAILED" in quality["flags"]


@pytest.mark.parametrize("val", [None, "", "0", "false", "off", "garbage"])
def test_off_path_adds_no_key_to_the_quality_sidecar(monkeypatch, val):
    """The byte-identity contract, at the seam that decides it.

    The gate is wired behind ``if resolve_affordance_gate_mode():`` — so with the
    flag off (or garbage) the fold is NEVER reached and the ``affordance_conservation``
    key is simply ABSENT from quality.json / cascade_ir.json. Serialized-equal is
    the assertion, because "the key is absent" IS the byte-identity claim.
    """
    if val is None:
        monkeypatch.delenv("SEMANTIK_AFFORDANCE_GATE", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_AFFORDANCE_GATE", val)

    from MCP.tools.pipeline_tools import _apply_affordance_to_quality  # noqa: PLC0415

    quality = {"quality_score": 1.0, "certification_status": "certified", "issues": []}
    before = json.dumps(quality, sort_keys=True)

    # Mirror the production guard verbatim.
    if resolve_affordance_gate_mode():  # pragma: no cover - off by construction
        _apply_affordance_to_quality(quality, audit_affordances(_ir("x"), _page("<p>x</p>")))

    assert resolve_affordance_gate_mode() is False
    assert "affordance_conservation" not in quality
    assert json.dumps(quality, sort_keys=True) == before


def test_healthy_report_still_certifies():
    """The gate must not de-certify a document it MEASURED and passed."""
    from MCP.tools.pipeline_tools import _apply_affordance_to_quality  # noqa: PLC0415

    report = audit_affordances(_ir("plain prose"), _page("<p>plain prose</p>"))
    assert report["document_passed"] is True
    quality = {"certification_status": "certified", "issues": [], "flags": []}
    _apply_affordance_to_quality(quality, report)
    assert quality["certification_status"] == "certified"
    assert quality["flags"] == []



# ---------------------------------------------------------------------------
# REAL-ARTIFACT regression — the invariants this gate was built from.
#
# Slug-agnostic + operator-pointed (see _discover). These assert the RELATIONAL
# invariant, not a book's magic numbers. The exact measured counts for the
# validation corpus live in the landing commit message, where they belong.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_KNOWN_BAD is None, reason="no SEMANTIK_AFFORDANCE_KNOWN_BAD_DIR")
def test_real_known_bad_conversion_fires_on_every_loss_class():
    """THE acceptance test. A real PRE-fix conversion must fire on all of it."""
    ir, html = _load(_KNOWN_BAD)
    report = audit_affordances(ir, html)
    assert report["document_passed"] is False
    checks = report["checks"]

    # CLASS-B (MISSING TRANSFORM): convertible spans in the source, ZERO <math>.
    assert checks["math"]["evidence_convertible"] > 0
    assert checks["math"]["emitted_math_elements"] == 0
    assert checks["math"]["severity"] == "critical"

    # CLASS-B (residue): CONVERTIBLE source notation SHIPPED as literal text.
    assert checks["residue"]["residue_convertible"] > 0
    assert checks["residue"]["severity"] == "critical"
    assert checks["residue"]["transform_ran"] is False

    # CLASS-A (DESTRUCTION): reconstructible table topologies, ZERO <thead>.
    assert checks["table_headers"]["evidence_reconstructible_tables"] > 0
    assert checks["table_headers"]["emitted_thead_blocks"] == 0
    assert checks["table_headers"]["severity"] == "critical"

    # CLASS-C (UNWIRED REPAIR): the running-header furniture in the <title>.
    assert checks["title_purity"]["severity"] == "warning"
    assert checks["title_purity"]["repair_delta"] is True

    # CLASS-A: every figure region the extractor DID see, dropped before render.
    assert checks["images"]["evidence_figure_regions"] > 0
    assert checks["images"]["emitted_img_elements"] == 0
    assert checks["images"]["severity"] == "warning"

    # CLASS-A: the declared ordinal spine outruns the emitted headings.
    assert checks["sections"]["recall"] < checks["sections"]["recall_min"]
    assert checks["sections"]["severity"] == "warning"


@pytest.mark.skipif(_KNOWN_BAD is None, reason="no SEMANTIK_AFFORDANCE_KNOWN_BAD_DIR")
def test_real_post_fix_render_passes_WITHOUT_the_emitter_self_report(monkeypatch):
    """D1 REGRESSION on the real artifact — the FABRICATED critical.

    The old ``_check_residue`` inferred "did the transform run?" SOLELY from the
    in-memory ``emitter_report``. That report is in-memory BY DESIGN (it is what
    buys the byte-identical off-path), so it is absent on EVERY out-of-band audit
    of a persisted corpus — the natural operator path, and precisely how this gate
    gets validated. Auditing the post-fix render through the documented 2-arg API
    therefore returned document_passed=False, printing "the transform never ran"
    over a document full of <math>.

    So this deliberately calls the 2-arg API with NO emitter_report: the two
    PERSISTED artifacts must be enough.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from semantik_rerender import _chapters_from_ir  # noqa: PLC0415

    from lib.semantik.adapter import normalize_cascade_to_ed4all  # noqa: PLC0415

    monkeypatch.setenv("SEMANTIK_LATEX_MATHML", "1")
    monkeypatch.setenv("SEMANTIK_TABLE_STRUCTURE", "1")
    monkeypatch.setenv("SEMANTIK_TITLE_SANITIZE", "1")
    ir, _old = _load(_KNOWN_BAD)
    out = normalize_cascade_to_ed4all(_chapters_from_ir(ir), pdf_stem=ir["pdf"])

    # The free win: the emitters report their OWN conserved/declined counts.
    emitter = out["affordance_emitter_report"]
    assert emitter["math_spans_converted"] > 0
    assert emitter["tables_accepted"] > 0

    report = audit_affordances(ir, out["html"])  # <- no emitter_report
    assert report["critical_count"] == 0
    assert report["document_passed"] is True

    residue = report["checks"]["residue"]
    assert residue["residue_convertible"] == 0  # nothing the emitter could render
    assert residue["transform_ran"] is True
    assert "never ran" not in residue["message"]
    if residue["residue_declined"]:
        # Fail-soft declines ship verbatim BY DESIGN -> warning, never critical.
        assert residue["verdict"] == "declined_verbatim"
        assert residue["severity"] == "warning"

    # The verdict is IDENTICAL with the self-report fed in — it is corroboration,
    # never the basis. Grading one document two ways depending on how the audit
    # was invoked WAS the defect.
    corroborated = audit_affordances(ir, out["html"], emitter_report=emitter)
    assert corroborated["document_passed"] is True
    assert corroborated["critical_count"] == 0
    assert corroborated["checks"]["residue"]["verdict"] == residue["verdict"]

    # The transforms actually did their job.
    assert report["checks"]["math"]["emitted_math_elements"] > 0
    assert report["checks"]["table_headers"]["emitted_thead_blocks"] > 0
    assert report["checks"]["title_purity"]["repair_delta"] is False


@pytest.mark.skipif(_CONTROL is None, reason="no SEMANTIK_AFFORDANCE_CONTROL_DIR")
def test_real_non_math_corpus_is_silent_domain_agnosticism_proof():
    """The de-poisoning proof: a non-math, non-table corpus must produce ZERO
    evidence and therefore ZERO issues. If this ever fires, the gate has grown a
    corpus-specific assumption."""
    ir, html = _load(_CONTROL)
    report = audit_affordances(ir, html)
    assert report["document_passed"] is True
    assert report["critical_count"] == 0
    assert report["warning_count"] == 0
    assert report["checks"]["math"]["evidence_convertible"] == 0
    assert report["checks"]["table_headers"]["evidence_reconstructible_tables"] == 0
