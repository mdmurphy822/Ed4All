"""WCAG 2.2 A/AA per-success-criterion coverage map (Plan 12 A2).

The single source of truth for what SemantiK's conformance claim actually
covers: every Level A and AA success criterion in WCAG 2.2, with an
honest verdict on how (and whether) the pipeline verifies it. Embedded
per-document into the conformance-audit sidecar and rendered to
``docs/wcag_coverage.md`` by ``scripts/analysis/gen_wcag_coverage_doc.py``.

Status vocabulary (``STATUSES``):

* ``automated``      — machine-verified on every document by a named
                       gate/check; failure blocks shipping certified.
* ``partial``        — the machine-checkable facet is verified; the
                       semantic facet is not machine-verifiable (the
                       notes say exactly which facet is which).
* ``human-review``   — not machine-verified today; a conformance claim
                       for affected content requires human audit.
* ``template-scope`` — owned by the SemantiK page template (CSS/skip-UX),
                       not the document content this pipeline emits.
* ``not-applicable`` — the criterion targets content/behavior the
                       pipeline structurally cannot emit (no JS, no
                       media, no timing, single static document).

Honesty rules: a criterion is ``automated`` only if a check named in
``enforced_by`` runs on every document and its failure prevents a
certified exit. Anything verified for syntax but not meaning is
``partial`` — never ``automated``. WCAG 2.2 removed SC 4.1.1 (Parsing);
it is listed ``not-applicable`` with that note so reviewers checking
against a 2.1-era list see it accounted for rather than missing.
"""

from __future__ import annotations

STATUSES = (
    "automated",
    "partial",
    "human-review",
    "template-scope",
    "not-applicable",
)

#: WCAG 2.2 Level A = 31 criteria, Level AA = 24 (4.1.1 removed in 2.2;
#: carried as a 56th informational row).
N_LEVEL_A = 31
N_LEVEL_AA = 24

_NO_JS = "pipeline emits static HTML with no scripting"
_NO_MEDIA = "pipeline emits no audio/video content (PDF→static HTML)"

# fmt: off
COVERAGE: list[dict] = [
    # ----------------------------------------------------------- 1.x
    {"sc": "1.1.1", "name": "Non-text Content", "level": "A",
     "status": "partial",
     "enforced_by": ["axe:image-alt", "gates/mathml_check.py (alttext)",
                     "figure_captioner numeric guard + caption-first eval"],
     "notes": "Alt presence and MathML alttext are hard-gated; figure "
              "alt quality is bounded by the captioner's no-hallucination "
              "guard but semantic adequacy of any alt text is not "
              "machine-verifiable."},
    {"sc": "1.2.1", "name": "Audio-only and Video-only (Prerecorded)",
     "level": "A", "status": "not-applicable", "enforced_by": [],
     "notes": _NO_MEDIA},
    {"sc": "1.2.2", "name": "Captions (Prerecorded)", "level": "A",
     "status": "not-applicable", "enforced_by": [], "notes": _NO_MEDIA},
    {"sc": "1.2.3", "name": "Audio Description or Media Alternative "
     "(Prerecorded)", "level": "A", "status": "not-applicable",
     "enforced_by": [], "notes": _NO_MEDIA},
    {"sc": "1.2.4", "name": "Captions (Live)", "level": "AA",
     "status": "not-applicable", "enforced_by": [], "notes": _NO_MEDIA},
    {"sc": "1.2.5", "name": "Audio Description (Prerecorded)",
     "level": "AA", "status": "not-applicable", "enforced_by": [],
     "notes": _NO_MEDIA},
    {"sc": "1.3.1", "name": "Info and Relationships", "level": "A",
     "status": "partial",
     "enforced_by": ["axe:wcag22aa structure rules",
                     "gates/hard_region.py:_check_table_structure "
                     "(th scope/id)",
                     "gates/table_h43.py (complex-table id/headers "
                     "generation + gate)",
                     "gates/hard_document.py heading-tree + landmark "
                     "checks", "axe:label / fieldset rules"],
     "notes": "Structural encoding is hard-gated (tables incl. H43 "
              "id/headers on complex tables, headings, landmarks, "
              "labels). Whether a heading is *semantically* a heading "
              "is bounded by the calibrated is_heading head + "
              "plausible-heading guard, not proven."},
    {"sc": "1.3.2", "name": "Meaningful Sequence", "level": "A",
     "status": "partial",
     "enforced_by": ["theta flag reading_order_at_risk (Plan 12 B3)",
                     "Stage-2 reading-order skeleton → DOM order"],
     "notes": "DOM order follows the extraction reading order; "
              "correctness of that order is not machine-verifiable. "
              "Degraded regions adjacent to tables/figures/math raise "
              "the reading_order_at_risk flag on the sidecar."},
    {"sc": "1.3.3", "name": "Sensory Characteristics", "level": "A",
     "status": "human-review", "enforced_by": [],
     "notes": "Instructions referencing shape/position/color are source "
              "content reproduced verbatim; the transformation neither "
              "introduces nor removes them."},
    {"sc": "1.3.4", "name": "Orientation", "level": "AA",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No orientation-restricting CSS or viewport directives "
              "are emitted."},
    {"sc": "1.3.5", "name": "Identify Input Purpose", "level": "AA",
     "status": "human-review", "enforced_by": [],
     "notes": "autocomplete attributes for user-data fields are not "
              "emitted; relevant only when form deliverables collect "
              "information about the user (e.g. tax forms)."},
    {"sc": "1.4.1", "name": "Use of Color", "level": "A",
     "status": "automated",
     "enforced_by": ["emitter sets no colors",
                     "axe:link-in-text-block"],
     "notes": "Output conveys no information through color: the "
              "document content carries no color styling at all."},
    {"sc": "1.4.2", "name": "Audio Control", "level": "A",
     "status": "not-applicable", "enforced_by": [], "notes": _NO_MEDIA},
    {"sc": "1.4.3", "name": "Contrast (Minimum)", "level": "AA",
     "status": "template-scope", "enforced_by": [],
     "notes": "Document content sets no colors; the SemantiK page template "
              "owns the palette (≥7:1)."},
    {"sc": "1.4.4", "name": "Resize Text", "level": "AA",
     "status": "template-scope", "enforced_by": [],
     "notes": "No font-size CSS is emitted in content; template "
              "stylesheet responsibility."},
    {"sc": "1.4.5", "name": "Images of Text", "level": "AA",
     "status": "partial",
     "enforced_by": ["math regions emit MathML (not images)",
                     "figure router equation_or_table_image class "
                     "(Plan 12 V4.1, pending)"],
     "notes": "Equations ship as MathML, not images. Source figures "
              "that are images of text are carried as figures with alt; "
              "detection/remediation of text-bearing figures awaits the "
              "figure router."},
    {"sc": "1.4.10", "name": "Reflow", "level": "AA",
     "status": "template-scope", "enforced_by": [],
     "notes": "No fixed-width layout is emitted; data tables use the "
              "WCAG-permitted two-dimensional exception."},
    {"sc": "1.4.11", "name": "Non-text Contrast", "level": "AA",
     "status": "template-scope", "enforced_by": [],
     "notes": "Content emits no styled UI components or graphics "
              "requiring contrast; template owns control styling."},
    {"sc": "1.4.12", "name": "Text Spacing", "level": "AA",
     "status": "template-scope", "enforced_by": [],
     "notes": "No letter/word/line-spacing CSS is emitted that could "
              "break user overrides."},
    {"sc": "1.4.13", "name": "Content on Hover or Focus", "level": "AA",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No tooltips/popovers are emitted; " + _NO_JS + "."},
    # ----------------------------------------------------------- 2.x
    {"sc": "2.1.1", "name": "Keyboard", "level": "A",
     "status": "automated",
     "enforced_by": ["native elements only (no scripted widgets)",
                     "axe:wcag22aa"],
     "notes": "Only natively keyboard-operable HTML elements are "
              "emitted; no event handlers exist."},
    {"sc": "2.1.2", "name": "No Keyboard Trap", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "Focus traps require scripting; " + _NO_JS + "."},
    {"sc": "2.1.4", "name": "Character Key Shortcuts", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No keyboard shortcuts are implemented; " + _NO_JS + "."},
    {"sc": "2.2.1", "name": "Timing Adjustable", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No time limits exist; " + _NO_JS + "."},
    {"sc": "2.2.2", "name": "Pause, Stop, Hide", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No moving/blinking/auto-updating content is emitted."},
    {"sc": "2.3.1", "name": "Three Flashes or Below Threshold",
     "level": "A", "status": "not-applicable", "enforced_by": [],
     "notes": "No flashing content can be emitted (static document)."},
    {"sc": "2.4.1", "name": "Bypass Blocks", "level": "A",
     "status": "automated",
     "enforced_by": ["assembler/shell.py skip-link",
                     "gates/hard_document.py <main> landmark check"],
     "notes": "Skip link + single <main> landmark are emitted and "
              "hard-gated on every document."},
    {"sc": "2.4.2", "name": "Page Titled", "level": "A",
     "status": "automated",
     "enforced_by": ["gates/hard_document.py title check",
                     "axe:document-title",
                     "theta flag title_fabricated (provenance)"],
     "notes": "Non-empty <title> is hard-gated; a synthesized (vs "
              "extracted) title is flagged title_fabricated on the "
              "sidecar."},
    {"sc": "2.4.3", "name": "Focus Order", "level": "A",
     "status": "automated",
     "enforced_by": ["no tabindex emitted; focus order = DOM order"],
     "notes": "Focusable elements follow DOM order, which follows "
              "reading order (see SC 1.3.2 for the reading-order "
              "caveat)."},
    {"sc": "2.4.4", "name": "Link Purpose (In Context)", "level": "A",
     "status": "partial",
     "enforced_by": ["axe:link-name"],
     "notes": "Non-empty accessible link names are hard-gated; whether "
              "the text meaningfully describes the target is not "
              "machine-verifiable."},
    {"sc": "2.4.5", "name": "Multiple Ways", "level": "AA",
     "status": "not-applicable", "enforced_by": [],
     "notes": "Applies to pages within a set; each output is a single "
              "self-contained document. Heading ids support in-page "
              "navigation regardless."},
    {"sc": "2.4.6", "name": "Headings and Labels", "level": "AA",
     "status": "partial",
     "enforced_by": ["gates/hard_document.py heading-tree check",
                     "structure_graph plausible-heading guard + "
                     "calibrated is_heading (T=1.6553, thr 0.8)",
                     "form label/legend requirements"],
     "notes": "Hierarchy validity and label presence are hard-gated; "
              "descriptiveness of heading/label text is bounded by the "
              "calibrated detector, not proven."},
    {"sc": "2.4.7", "name": "Focus Visible", "level": "AA",
     "status": "template-scope", "enforced_by": [],
     "notes": "Focus styling is injected by the SemantiK page template; "
              "content emits no CSS that could suppress it."},
    {"sc": "2.4.11", "name": "Focus Not Obscured (Minimum)",
     "level": "AA", "status": "not-applicable", "enforced_by": [],
     "notes": "No sticky/overlay positioning is emitted that could "
              "obscure focused elements."},
    {"sc": "2.5.1", "name": "Pointer Gestures", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No gesture handlers; " + _NO_JS + "."},
    {"sc": "2.5.2", "name": "Pointer Cancellation", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No pointer event handlers; " + _NO_JS + "."},
    {"sc": "2.5.3", "name": "Label in Name", "level": "A",
     "status": "automated",
     "enforced_by": ["axe:label-content-name-mismatch",
                     "labels are plain text (visible text = accessible "
                     "name by construction)"],
     "notes": "No aria-label overrides of visible text are emitted."},
    {"sc": "2.5.4", "name": "Motion Actuation", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No motion-based input; " + _NO_JS + "."},
    {"sc": "2.5.7", "name": "Dragging Movements", "level": "AA",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No draggable interactions; " + _NO_JS + "."},
    {"sc": "2.5.8", "name": "Target Size (Minimum)", "level": "AA",
     "status": "partial",
     "enforced_by": ["TARGET_MIN_STYLE inline min 24px on form "
                     "controls (legacy emitter)"],
     "notes": "Inline min-size styles cover emitted form controls; "
              "links in prose fall under the inline-text exception. "
              "Re-verify when the v2 cascade gains a forms path."},
    # ----------------------------------------------------------- 3.x
    {"sc": "3.1.1", "name": "Language of Page", "level": "A",
     "status": "automated",
     "enforced_by": ["gates/hard_document.py lang_declared check",
                     "axe:html-has-lang/html-lang-valid"],
     "notes": "Non-empty <html lang> is hard-gated on every document."},
    {"sc": "3.1.2", "name": "Language of Parts", "level": "AA",
     "status": "human-review", "enforced_by": [],
     "notes": "Inline <span lang> for foreign-language phrases is not "
              "emitted (Plan 12 V3.4). Monolingual documents are "
              "unaffected; multilingual documents need human review."},
    {"sc": "3.2.1", "name": "On Focus", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No focus-triggered context changes; " + _NO_JS + "."},
    {"sc": "3.2.2", "name": "On Input", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No input-triggered context changes; " + _NO_JS + "."},
    {"sc": "3.2.3", "name": "Consistent Navigation", "level": "AA",
     "status": "not-applicable", "enforced_by": [],
     "notes": "Applies to repeated navigation across a set of pages; "
              "each output is a single document."},
    {"sc": "3.2.4", "name": "Consistent Identification", "level": "AA",
     "status": "not-applicable", "enforced_by": [],
     "notes": "Applies to components repeated across a set of pages; "
              "each output is a single document."},
    {"sc": "3.2.6", "name": "Consistent Help", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No help mechanism is provided within the documents."},
    {"sc": "3.3.1", "name": "Error Identification", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "Forms render structurally but are not processed; no "
              "input-error detection exists to describe."},
    {"sc": "3.3.2", "name": "Labels or Instructions", "level": "A",
     "status": "automated",
     "enforced_by": ["emitter requires label+name per control "
                     "(<label for>)", "fieldset/legend required",
                     "axe:label"],
     "notes": "Every emitted control carries a programmatic label; "
              "hard-gated."},
    {"sc": "3.3.3", "name": "Error Suggestion", "level": "AA",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No input-error handling exists (static forms)."},
    {"sc": "3.3.4", "name": "Error Prevention (Legal, Financial, "
     "Data)", "level": "AA", "status": "not-applicable",
     "enforced_by": [],
     "notes": "No submissions are processed."},
    {"sc": "3.3.7", "name": "Redundant Entry", "level": "A",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No multi-step processes exist within a document."},
    {"sc": "3.3.8", "name": "Accessible Authentication (Minimum)",
     "level": "AA", "status": "not-applicable", "enforced_by": [],
     "notes": "No authentication exists in the documents."},
    # ----------------------------------------------------------- 4.x
    {"sc": "4.1.2", "name": "Name, Role, Value", "level": "A",
     "status": "automated",
     "enforced_by": ["native semantic elements only (no custom "
                     "widgets)", "axe:wcag22aa ARIA rules"],
     "notes": "Names/roles come from native semantics; axe validates "
              "any ARIA attributes the assembler emits "
              "(aria-describedby, role=note banner)."},
    {"sc": "4.1.3", "name": "Status Messages", "level": "AA",
     "status": "not-applicable", "enforced_by": [],
     "notes": "No dynamic status updates; " + _NO_JS + "."},
    # Removed criterion — listed so 2.1-era checklists see it accounted for.
    {"sc": "4.1.1", "name": "Parsing (REMOVED in WCAG 2.2)",
     "level": "removed", "status": "not-applicable",
     "enforced_by": ["gates HTML5 well-formedness checks (still run)"],
     "notes": "SC 4.1.1 was removed in WCAG 2.2; SemantiK still hard-gates "
              "HTML5 well-formedness at region and document level."},
]
# fmt: on


def coverage_map() -> list[dict]:
    """The per-SC coverage rows, deep-copied so callers can't mutate
    the canonical table."""
    import copy

    return copy.deepcopy(COVERAGE)


def coverage_summary() -> dict[str, int]:
    """Status → count over the A/AA rows (the removed 4.1.1 excluded)."""
    out: dict[str, int] = {}
    for row in COVERAGE:
        if row["level"] == "removed":
            continue
        out[row["status"]] = out.get(row["status"], 0) + 1
    return out


def render_markdown() -> str:
    """Render the procurement-facing coverage document.

    ``docs/wcag_coverage.md`` is generated from this function by
    ``scripts/analysis/gen_wcag_coverage_doc.py`` — edit COVERAGE, regenerate,
    never hand-edit the doc (a test enforces sync).
    """
    lines = [
        "# SemantiK — WCAG 2.2 A/AA Coverage Map",
        "",
        "<!-- GENERATED by scripts/analysis/gen_wcag_coverage_doc.py from",
        "     semantik_structure/gates/wcag_coverage.py — do not hand-edit. -->",
        "",
        "Per-success-criterion statement of what SemantiK's automated "
        "conformance pipeline verifies, what it verifies only "
        "syntactically, what requires human review, what the page "
        "template owns, and what cannot occur in this output class. "
        "The same rows are embedded machine-readably in every "
        "document's `conformance_audit.json` sidecar.",
        "",
        "| Status | Meaning |",
        "| --- | --- |",
        "| `automated` | Machine-verified on every document; failure blocks a certified exit |",
        "| `partial` | Machine-checkable facet verified; semantic facet not machine-verifiable (see notes) |",
        "| `human-review` | Not machine-verified; conformance for affected content needs human audit |",
        "| `template-scope` | Owned by the SemantiK page template, not document content |",
        "| `not-applicable` | The output class structurally cannot contain the targeted content/behavior |",
        "",
        "## Summary",
        "",
    ]
    summary = coverage_summary()
    for status in STATUSES:
        lines.append(f"- **{status}**: {summary.get(status, 0)}")
    lines += [
        "",
        "## Criteria",
        "",
        "| SC | Name | Level | Status | Enforced by | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in COVERAGE:
        enforced = "; ".join(row["enforced_by"]) or "—"
        lines.append(
            f"| {row['sc']} | {row['name']} | {row['level']} | "
            f"`{row['status']}` | {enforced} | {row['notes']} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "COVERAGE",
    "STATUSES",
    "N_LEVEL_A",
    "N_LEVEL_AA",
    "coverage_map",
    "coverage_summary",
    "render_markdown",
]
