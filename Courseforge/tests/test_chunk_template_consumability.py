"""
Wave 79 Worker C — chunk-template consumability tests.

Verifies that the four canonical chunk templates documented in
``Courseforge/templates/chunk_templates.md`` produce HTML whose
metadata is locatable by Trainforge's ``HTMLContentParser`` (the same
extraction surface Wave 79 Worker A's instruction-pair pipeline runs
against). These tests assert the templates are CONSUMABLE — i.e.
Trainforge can find ``data-cf-template-type`` and the template-
specific sub-attributes deterministically without keyword guessing.

The tests are **forward-looking**: they pin the contract the future
content-generator subagent must honor. They do NOT exercise the
content-generator itself — that requires a real Anthropic dispatch
and is out of scope for Wave 79 Worker C.
"""

from __future__ import annotations

import re

from Trainforge.parsers.html_content_parser import HTMLContentParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_page(body_html: str, title: str = "Wave 79 Template Sample") -> str:
    """Wrap a fragment in a minimal HTML5 page so the parser sees a title."""
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\"><head>"
        f"<meta charset=\"UTF-8\"><title>{title}</title>"
        "</head><body>"
        f"{body_html}"
        "</body></html>"
    )


def _extract_attr(html: str, attr: str) -> list[str]:
    """Return all values of a given ``data-cf-*`` attribute."""
    return re.findall(rf'{re.escape(attr)}="([^"]*)"', html)


# ---------------------------------------------------------------------------
# Template 1: Real-World Scenario
# ---------------------------------------------------------------------------

TEMPLATE_1_HTML = """
<section data-cf-template-type="real_world_scenario"
         data-cf-scenario-domain="data_governance"
         data-cf-applicable-concepts="rdf-graph,shacl-shape"
         data-cf-expected-deliverable="A SHACL NodeShape that constrains the Customer class"
         data-cf-objective-id="CO-04"
         data-cf-bloom-level="apply"
         data-cf-content-type="example">
  <h3 data-cf-content-type="example"
      data-cf-key-terms="rdf-graph,shacl-shape">Scenario: Onboarding a new SHACL constraint</h3>
  <p>You're the data steward at FinServ Inc. The compliance team flagged that the
  customer master graph admits records missing tax-residency.</p>
  <h4>Your Task</h4>
  <p>Author a SHACL NodeShape that constrains every Customer to one taxResidency.</p>
  <h4>Approach</h4>
  <ol>
    <li>Identify the target class.</li>
    <li>Add a property shape with cardinality bounds.</li>
    <li>Constrain the value to ISO-3166 alpha-2.</li>
  </ol>
  <h4>Success Criteria</h4>
  <ul>
    <li>The shape rejects a Customer with zero taxResidency values.</li>
    <li>The shape accepts US, GB, JP and rejects USA, us.</li>
  </ul>
</section>
"""


def test_template_1_real_world_scenario_metadata_extractable():
    """Template 1 emits ``data-cf-template-type`` + scenario sub-fields."""
    page = _wrap_page(TEMPLATE_1_HTML)

    # 1) Template-type marker is present and unique.
    template_types = _extract_attr(page, "data-cf-template-type")
    assert template_types == ["real_world_scenario"]

    # 2) Scenario-specific sub-attributes are deterministically locatable.
    assert _extract_attr(page, "data-cf-scenario-domain") == ["data_governance"]
    assert _extract_attr(page, "data-cf-applicable-concepts") == [
        "rdf-graph,shacl-shape"
    ]
    deliverables = _extract_attr(page, "data-cf-expected-deliverable")
    assert len(deliverables) == 1
    assert deliverables[0].startswith("A SHACL NodeShape")

    # 3) Wave-stable attributes (objective + bloom + content_type) still apply.
    assert _extract_attr(page, "data-cf-objective-id") == ["CO-04"]
    assert _extract_attr(page, "data-cf-bloom-level") == ["apply"]

    # 4) Trainforge's parser finds the section, picks up content_type +
    #    key terms from the heading, and the chunk-extractor surface is
    #    populated.
    parsed = HTMLContentParser().parse(page)
    headings = [s.heading for s in parsed.sections]
    assert any(h.startswith("Scenario:") for h in headings)
    scenario_section = next(
        s for s in parsed.sections if s.heading.startswith("Scenario:")
    )
    assert scenario_section.content_type == "example"
    assert "rdf-graph" in scenario_section.key_terms
    assert "shacl-shape" in scenario_section.key_terms


# ---------------------------------------------------------------------------
# Template 2: Problem-Solution Walkthrough
# ---------------------------------------------------------------------------

TEMPLATE_2_HTML = """
<section data-cf-template-type="problem_solution"
         data-cf-problem-class="cardinality_constraint"
         data-cf-applicable-concepts="shacl-shape,property-path"
         data-cf-objective-id="CO-04"
         data-cf-bloom-level="apply"
         data-cf-content-type="example">
  <h3>Problem</h3>
  <p>A Customer may have multiple email addresses, but only one carries
  the primaryEmail flag. Write a SHACL shape that fails when more than
  one primaryEmail is present per customer.</p>
  <h3>Walkthrough</h3>
  <ol>
    <li><strong>Identify:</strong> the property targeted is primaryEmail.</li>
    <li><strong>Plan:</strong> use sh:maxCount 1 on the property path.</li>
    <li><strong>Execute:</strong> author the Turtle shape.</li>
    <li><strong>Verify:</strong> validate against fixtures.</li>
  </ol>
  <h3>Common Incorrect Approach</h3>
  <p data-cf-counter-example="true">Many learners try to enforce primary-email
  uniqueness with sh:maxCount 1 on the generic email property. This fails
  because the constraint fires on every email, not on the flagged primary.</p>
</section>
"""


def test_template_2_problem_solution_walkthrough_extractable():
    """Template 2: problem statement, walkthrough steps, and counter-example
    must all be parseable as separate locatable artifacts."""
    page = _wrap_page(TEMPLATE_2_HTML)

    # 1) Template-type marker.
    assert _extract_attr(page, "data-cf-template-type") == ["problem_solution"]

    # 2) Problem-class slug is locatable for the extractor.
    assert _extract_attr(page, "data-cf-problem-class") == [
        "cardinality_constraint"
    ]

    # 3) Counter-example paragraph carries the discriminator the DPO
    #    extractor uses to mint the "rejected" arm.
    counter_examples = _extract_attr(page, "data-cf-counter-example")
    assert counter_examples == ["true"]

    # 4) Trainforge's parser surfaces three distinct sections and the
    #    walkthrough's ordered list is preserved as text.
    parsed = HTMLContentParser().parse(page)
    headings = [s.heading for s in parsed.sections]
    assert "Problem" in headings
    assert "Walkthrough" in headings
    assert "Common Incorrect Approach" in headings

    walkthrough = next(s for s in parsed.sections if s.heading == "Walkthrough")
    # Each "Identify / Plan / Execute / Verify" step keyword appears in the
    # walkthrough's flattened text — confirms the ordered list is preserved.
    for step_keyword in ("Identify", "Plan", "Execute", "Verify"):
        assert step_keyword in walkthrough.content, (
            f"Walkthrough step '{step_keyword}' missing from extracted text"
        )


# ---------------------------------------------------------------------------
# Template 3: Common Pitfall
# ---------------------------------------------------------------------------

TEMPLATE_3_HTML = """
<section data-cf-template-type="common_pitfall"
         data-cf-pitfall-concept="rdf-blank-node"
         data-cf-confused-with="rdf-named-node"
         data-cf-objective-id="CO-02"
         data-cf-bloom-level="analyze"
         data-cf-content-type="explanation">
  <h3>Common Pitfall: treating blank nodes like named resources</h3>
  <p>When learners first model a complex object, they reach for a blank node.</p>
  <h4>What looks like the right answer</h4>
  <p data-cf-misconception="true">A blank node is just an anonymous URI;
  downstream consumers can dereference it the same way.</p>
  <h4>Why it's wrong</h4>
  <p>Blank-node identifiers are scoped to the graph that emits them.</p>
  <h4>The right approach</h4>
  <p>Mint a named node when the resource needs to be referenced from
  outside its immediate context.</p>
  <h4>Quick test</h4>
  <p>If anything outside this graph could need to link to this thing,
  mint a named node.</p>
</section>
"""


def test_template_3_common_pitfall_misconception_extractable():
    """Template 3: misconception paragraph carries
    ``data-cf-misconception="true"`` so Trainforge can mint a misconception
    KG node deterministically."""
    page = _wrap_page(TEMPLATE_3_HTML)

    # 1) Template-type marker.
    assert _extract_attr(page, "data-cf-template-type") == ["common_pitfall"]

    # 2) Pitfall vs. confused-with concept slugs are locatable.
    assert _extract_attr(page, "data-cf-pitfall-concept") == ["rdf-blank-node"]
    assert _extract_attr(page, "data-cf-confused-with") == ["rdf-named-node"]

    # 3) Misconception discriminator is on EXACTLY ONE paragraph and
    #    its text is extractable for KG node minting.
    misconception_markers = _extract_attr(page, "data-cf-misconception")
    assert misconception_markers == ["true"]

    misc_paragraphs = re.findall(
        r'<p[^>]*data-cf-misconception="true"[^>]*>(.*?)</p>',
        page,
        re.DOTALL,
    )
    assert len(misc_paragraphs) == 1
    assert "anonymous URI" in misc_paragraphs[0]

    # 4) The pitfall section is parseable as a top-level h3 with each
    #    sub-section (h4) showing up downstream.
    parsed = HTMLContentParser().parse(page)
    headings = [s.heading for s in parsed.sections]
    assert any("Common Pitfall" in h for h in headings)
    assert "What looks like the right answer" in headings
    assert "The right approach" in headings


# ---------------------------------------------------------------------------
# Template 4: Step-by-Step Procedure
# ---------------------------------------------------------------------------

TEMPLATE_4_HTML = """
<section data-cf-template-type="procedure"
         data-cf-procedure-name="validate_graph_against_shapes"
         data-cf-applicable-concepts="shacl-shape,validation-report"
         data-cf-objective-id="CO-05"
         data-cf-bloom-level="apply"
         data-cf-content-type="procedure">
  <h3 data-cf-content-type="procedure"
      data-cf-key-terms="shacl-shape,validation-report">Procedure: Validate an RDF graph against a SHACL shapes graph</h3>
  <h4>When to use</h4>
  <p>Run this procedure whenever you need to confirm a data graph
  conforms to a published shapes graph.</p>
  <h4>Inputs</h4>
  <ul>
    <li>A data graph in any RDF serialization.</li>
    <li>A shapes graph defining the constraints.</li>
    <li>A SHACL validator (e.g. pyshacl).</li>
  </ul>
  <h4>Steps</h4>
  <ol>
    <li>Load the data graph into the validator.</li>
    <li>Load the shapes graph the same way.</li>
    <li>Invoke validation. Capture the validation report graph.</li>
    <li>Inspect sh:conforms.</li>
    <li>If false, iterate through sh:result entries.</li>
  </ol>
  <h4>Output</h4>
  <p>A SHACL validation report graph (itself RDF).</p>
  <h4>Worked Example</h4>
  <p>Running pyshacl -s shapes.ttl data.ttl on the FinServ fixture
  yields sh:conforms false and one sh:MinCountConstraintComponent.</p>
</section>
"""


def test_template_4_procedure_steps_parseable():
    """Template 4: procedure steps must parse as an ordered list and the
    Inputs / Steps / Output / Worked Example sub-sections must be locatable."""
    page = _wrap_page(TEMPLATE_4_HTML)

    # 1) Template-type marker.
    assert _extract_attr(page, "data-cf-template-type") == ["procedure"]

    # 2) Procedure-name slug is locatable for the extractor.
    assert _extract_attr(page, "data-cf-procedure-name") == [
        "validate_graph_against_shapes"
    ]

    # 3) The Steps section is an ORDERED list (sequencing matters for
    #    procedural training pairs). The opening <ol> tag must appear
    #    before the worked example block so step order is preserved.
    steps_block = re.search(
        r'<h4>Steps</h4>\s*<ol>(.*?)</ol>',
        page,
        re.DOTALL,
    )
    assert steps_block is not None, "Procedure must use <ol> for Steps"
    step_items = re.findall(r'<li>(.*?)</li>', steps_block.group(1), re.DOTALL)
    assert len(step_items) >= 3, (
        f"Procedure must have at least 3 steps; got {len(step_items)}"
    )

    # 4) All four required procedure sub-sections are extracted as headings
    #    by Trainforge's parser.
    parsed = HTMLContentParser().parse(page)
    headings = [s.heading for s in parsed.sections]
    for required in ("Inputs", "Steps", "Output", "Worked Example"):
        assert required in headings, (
            f"Procedure sub-section '{required}' missing from parsed headings"
        )

    # 5) The procedure section's content_type is canonically "procedure",
    #    confirming the chunk_v4 content_type_label enum routing.
    procedure_section = next(
        s for s in parsed.sections if s.heading.startswith("Procedure:")
    )
    assert procedure_section.content_type == "procedure"
