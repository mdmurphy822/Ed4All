# Wave 79 Chunk Template Catalog

> **Forward-looking** — applies to FUTURE Courseforge content-generator
> runs. Existing exports (e.g. `Courseforge/exports/PROJ-RDF_SHACL_550-…`)
> are unchanged; this catalog is ADDITIVE guidance.

## Why this catalog exists

Wave 78's content-generator subagents produced grounded but
**explanation-heavy** chunks. Across the rdf-shacl-550 archive,
chunk-type distribution skewed:

| content_type_label | count | share |
|--------------------|------:|------:|
| `explanation`      |  153  |  70%  |
| `example`          |   25  |  11%  |
| `exercise`         |   14  |   6%  |
| `assessment_item`  |   14  |   6%  |
| (other)            |   13  |   6%  |
| **total**          |  219  | 100%  |

Downstream SLM training (Wave 79 Worker A's instruction-pair
extractor) wants ~3× the example/exercise/assessment ratio so the
synthesized SFT + DPO pairs cover task-oriented behavior, not just
explanatory recall. This file gives the content-generator subagent
**deterministic** per-template HTML so the parser can find each chunk
type by attribute, not by heuristic class-name guessing.

## How Trainforge consumes this

Each template prescribes a `data-cf-template-type` attribute on the
enclosing `<section>`. The Wave 79 Worker A extractor walks the parsed
section list (via `Trainforge/parsers/html_content_parser.py`) and
keys instruction/preference pairs off that attribute:

| `data-cf-template-type` | Wave 79 extractor mapping |
|-------------------------|---------------------------|
| `real_world_scenario`   | scenario → task → SFT pair |
| `problem_solution`      | problem → walkthrough → SFT pair; counter-example → DPO rejected arm |
| `common_pitfall`        | misconception → correction → DPO chosen vs. rejected pair |
| `procedure`             | inputs + steps + worked example → SFT procedural pair |

`data-cf-template-type` is ADDITIVE: the existing
`data-cf-content-type` / `data-cf-objective-id` / `data-cf-bloom-level`
/ `data-cf-source-ids` attributes (per `Courseforge/CLAUDE.md` §
"HTML Data Attributes") still apply on the same `<section>` and
remain authoritative for the chunk's content_type / LO / Bloom /
provenance metadata.

---

## Template 1 — Real-World Scenario

**Goal**: A bounded, evaluable task framed in a realistic professional
setting. Trainforge's extractor turns the scenario context into the
SFT prompt and the deliverable into the SFT response target.

### Required `data-cf-*` attributes

| Attribute | Required | Purpose |
|-----------|:--------:|---------|
| `data-cf-template-type="real_world_scenario"` | yes | Routes the section to the scenario extractor arm. |
| `data-cf-scenario-domain` | yes | Short slug of the professional context (e.g. `data_governance`, `cli_devops`, `clinical_triage`). |
| `data-cf-applicable-concepts` | yes | Comma-separated concept slugs the scenario exercises. |
| `data-cf-expected-deliverable` | yes | One-line description of what the learner is asked to produce. |
| `data-cf-objective-id` | yes | Canonical CO-NN / TO-NN reference. |
| `data-cf-bloom-level` | yes | Typically `apply` or `analyze`; `evaluate` for judgment-heavy scenarios. |

### Canonical HTML

```html
<section data-cf-template-type="real_world_scenario"
         data-cf-scenario-domain="data_governance"
         data-cf-applicable-concepts="rdf-graph,shacl-shape"
         data-cf-expected-deliverable="A SHACL NodeShape that constrains the Customer class"
         data-cf-objective-id="CO-04"
         data-cf-bloom-level="apply"
         data-cf-content-type="example">
  <h3>Scenario: Onboarding a new SHACL constraint at FinServ Inc.</h3>
  <p>You're the data steward at FinServ Inc., a mid-sized financial
  services firm. The compliance team has just flagged that the
  customer master graph admits records missing tax-residency. Records
  are stored as RDF triples in a graph database; downstream
  reporting tools assume every <code>:Customer</code> has exactly one
  <code>:taxResidency</code>. Without a constraint, ingestion
  silently accepts malformed records and the gap surfaces only at
  the regulator's quarterly audit.</p>
  <h4>Your Task</h4>
  <p>Author a SHACL NodeShape that constrains every
  <code>:Customer</code> to have exactly one <code>:taxResidency</code>
  property whose value is an ISO-3166 alpha-2 country code. Submit
  the shape as Turtle.</p>
  <h4>Approach</h4>
  <ol>
    <li>Identify the target class (<code>sh:targetClass :Customer</code>).</li>
    <li>Add a property shape for <code>:taxResidency</code> with
    <code>sh:minCount 1</code> and <code>sh:maxCount 1</code>.</li>
    <li>Constrain the value with <code>sh:datatype xsd:string</code>
    and <code>sh:pattern "^[A-Z]{2}$"</code>.</li>
    <li>Validate the shape against a known-good fixture before
    deploying it to staging.</li>
  </ol>
  <h4>Success Criteria</h4>
  <ul>
    <li>The shape rejects a <code>:Customer</code> with zero
    <code>:taxResidency</code> values.</li>
    <li>The shape rejects a <code>:Customer</code> with two
    <code>:taxResidency</code> values.</li>
    <li>The shape accepts <code>"US"</code>, <code>"GB"</code>,
    <code>"JP"</code> and rejects <code>"USA"</code>,
    <code>"us"</code>, <code>"123"</code>.</li>
  </ul>
</section>
```

---

## Template 2 — Problem-Solution Walkthrough

**Goal**: An explicit problem statement plus a stepwise walkthrough
that names the failure mode of the most common incorrect approach.
The counter-example block becomes the DPO **rejected** arm in
Wave 79's preference-pair pipeline; the walkthrough is the
**chosen** arm.

### Required `data-cf-*` attributes

| Attribute | Required | Purpose |
|-----------|:--------:|---------|
| `data-cf-template-type="problem_solution"` | yes | Routes to the problem-solution extractor arm. |
| `data-cf-problem-class` | yes | Slug grouping similar problems (e.g. `cardinality_constraint`, `recursive_descent_parse`). |
| `data-cf-applicable-concepts` | yes | Comma-separated concept slugs. |
| `data-cf-objective-id` | yes | Canonical CO-NN / TO-NN reference. |
| `data-cf-bloom-level` | yes | Typically `apply` or `analyze`. |

The counter-example paragraph MUST carry
`data-cf-counter-example="true"` so the DPO extractor can locate it
deterministically without keyword matching.

### Canonical HTML

```html
<section data-cf-template-type="problem_solution"
         data-cf-problem-class="cardinality_constraint"
         data-cf-applicable-concepts="shacl-shape,property-path"
         data-cf-objective-id="CO-04"
         data-cf-bloom-level="apply"
         data-cf-content-type="example">
  <h3 data-cf-content-type="example"
      data-cf-key-terms="shacl-shape,property-path">Problem</h3>
  <p>A <code>:Customer</code> may have multiple email addresses, but
  only one of them may carry the <code>:primaryEmail</code> flag.
  Given the data graph below, write a SHACL shape that fails when
  more than one <code>:primaryEmail</code> is present per customer.</p>
  <h3>Walkthrough</h3>
  <ol>
    <li><strong>Identify:</strong> the constraint targets the
    <code>:primaryEmail</code> property, with cardinality at most one
    per <code>:Customer</code> instance.</li>
    <li><strong>Plan:</strong> use a property shape with
    <code>sh:maxCount 1</code> on
    <code>sh:path :primaryEmail</code>, scoped via
    <code>sh:targetClass :Customer</code>.</li>
    <li><strong>Execute:</strong> author the Turtle shape and
    register it in the validation graph.</li>
    <li><strong>Verify:</strong> validate against a fixture with one
    primary email (passes) and a fixture with two (fails with
    <code>sh:MaxCountConstraintComponent</code>).</li>
  </ol>
  <h3>Common Incorrect Approach</h3>
  <p data-cf-counter-example="true">Many learners try to enforce
  primary-email uniqueness with <code>sh:minCount 0</code> and
  <code>sh:maxCount 1</code> on the <em>generic</em>
  <code>:email</code> property. This fails because the constraint
  fires on every email, not on the flagged primary — a customer
  with three secondary emails and zero primaries trips the shape.
  The walkthrough above succeeds because it scopes the cardinality
  bound to <code>:primaryEmail</code> specifically, leaving generic
  <code>:email</code> multi-valued as intended.</p>
</section>
```

---

## Template 3 — Common Pitfall

**Goal**: Name the misconception, expose the gap, and give the
right framing. The misconception paragraph becomes a first-class
KG node (`misconception-of` edge in `schemas/knowledge/misconception.schema.json`)
and the corresponding correction becomes the DPO **chosen** arm.

### Required `data-cf-*` attributes

| Attribute | Required | Purpose |
|-----------|:--------:|---------|
| `data-cf-template-type="common_pitfall"` | yes | Routes to the pitfall extractor arm. |
| `data-cf-pitfall-concept` | yes | Slug for the concept the learner is misapplying. |
| `data-cf-confused-with` | yes | Slug for the concept the learner is confusing it with. |
| `data-cf-objective-id` | yes | Canonical CO-NN / TO-NN reference. |
| `data-cf-bloom-level` | yes | Typically `analyze` or `evaluate`. |

The "what looks like the right answer" paragraph MUST carry
`data-cf-misconception="true"` so Trainforge can mint a
misconception node deterministically.

### Canonical HTML

```html
<section data-cf-template-type="common_pitfall"
         data-cf-pitfall-concept="rdf-blank-node"
         data-cf-confused-with="rdf-named-node"
         data-cf-objective-id="CO-02"
         data-cf-bloom-level="analyze"
         data-cf-content-type="explanation">
  <h3 data-cf-content-type="explanation"
      data-cf-key-terms="rdf-blank-node,rdf-named-node">Common Pitfall: treating blank nodes like named resources</h3>
  <p>When learners first model a complex object — say a customer's
  mailing address — they reach for a blank node because it "saves
  having to mint a URI." This works locally but fails the moment the
  same address has to be referenced from a second graph or a query
  result.</p>
  <h4>What looks like the right answer</h4>
  <p data-cf-misconception="true">"A blank node is just an
  anonymous URI; downstream consumers can dereference it the same
  way." This treats blank-node identity as portable across graph
  boundaries, which it isn't.</p>
  <h4>Why it's wrong</h4>
  <p>Blank-node identifiers are <em>scoped to the graph that emits
  them</em>. The same blank node serialized into two different stores
  becomes two distinct nodes; there is no global identity. Any
  cross-graph reference (federated SPARQL, IMSCC import, JSON-LD
  framing) loses the link.</p>
  <h4>The right approach</h4>
  <p>If the resource needs to be referenced from outside its
  immediate context — across graphs, across services, across
  serializations — mint a named node with a stable URI under a
  controlled namespace. Use blank nodes only for genuinely local
  structure (e.g. anonymous list cells, intermediate query
  variables) where identity is bounded by the surrounding graph.</p>
  <h4>Quick test</h4>
  <p>If you have to ask "could anything outside this graph need to
  link to this thing?" and the answer is anything other than a firm
  no, mint a named node.</p>
</section>
```

---

## Template 4 — Step-by-Step Procedure

**Goal**: A bounded, repeatable procedure with explicit inputs,
ordered steps, an output description, and one worked example. The
extractor turns the inputs + steps into a procedural SFT pair; the
worked example becomes a second SFT pair grounded in the same
procedure.

### Required `data-cf-*` attributes

| Attribute | Required | Purpose |
|-----------|:--------:|---------|
| `data-cf-template-type="procedure"` | yes | Routes to the procedure extractor arm. |
| `data-cf-procedure-name` | yes | Short noun phrase naming the procedure. |
| `data-cf-applicable-concepts` | yes | Comma-separated concept slugs. |
| `data-cf-objective-id` | yes | Canonical CO-NN / TO-NN reference. |
| `data-cf-bloom-level` | yes | Typically `apply`. |

### Canonical HTML

```html
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
  conforms to a published shapes graph — typically before
  promoting a dataset from staging to production, or before
  accepting an external partner's graph contribution. Do <em>not</em>
  use this procedure for free-form quality checks that aren't
  expressible as SHACL constraints.</p>
  <h4>Inputs</h4>
  <ul>
    <li>A data graph in any RDF serialization (Turtle, N-Triples,
    JSON-LD, RDF/XML).</li>
    <li>A shapes graph (typically Turtle) defining the constraints.</li>
    <li>A SHACL validator (e.g. <code>pyshacl</code>,
    <code>TopBraid</code>, <code>Apache Jena SHACL</code>).</li>
  </ul>
  <h4>Steps</h4>
  <ol>
    <li>Load the data graph into the validator (verify parse with no
    syntax errors).</li>
    <li>Load the shapes graph the same way (verify parse).</li>
    <li>Invoke validation. Capture the validation report graph.</li>
    <li>Inspect the report's <code>sh:conforms</code> boolean —
    <code>true</code> means no constraint violations.</li>
    <li>If <code>sh:conforms</code> is <code>false</code>, iterate
    through <code>sh:result</code> entries; each names the focus
    node, the violated constraint component, and a human-readable
    message.</li>
  </ol>
  <h4>Output</h4>
  <p>A SHACL validation report graph (itself RDF) with one
  <code>sh:ValidationResult</code> per violation, plus a top-level
  <code>sh:conforms</code> flag. A conforming graph yields a report
  with <code>sh:conforms true</code> and zero results.</p>
  <h4>Worked Example</h4>
  <p>Given the FinServ <code>:Customer</code> graph from Template 1
  and the <code>:taxResidency</code> shape from the same scenario,
  running <code>pyshacl -s shapes.ttl data.ttl</code> against a
  fixture with one tax-residency-less customer produces a report
  with <code>sh:conforms false</code> and a single
  <code>sh:MinCountConstraintComponent</code> result naming that
  customer's IRI as the focus node.</p>
</section>
```

---

## Per-week chunk mix targets (Wave 79)

| Chunk type | Wave 78 actual (rdf-shacl-550) | Wave 79 target (per week) |
|------------|-------------------------------:|--------------------------:|
| `explanation`              | ~13/wk | **4-5/wk** |
| `example`                  | ~2/wk  | **2-3/wk** |
| `procedure`                | rare   | **2/wk** |
| `real_world_scenario`      | rare   | **1-2/wk** |
| `common_pitfall`           | rare   | **1/wk** |
| `problem_solution`         | rare   | **1/wk** |
| `self-check` (interactive) | ~1/wk  | **1/wk** |
| `summary`                  | ~1/wk  | **1/wk** |
| `overview`                 | ~1/wk  | **1/wk** |
| **Total / week**           | ~8-10  | **~15-18** |

The new mix shifts the explanation : example : exercise ratio from
roughly **10 : 1.5 : 1** to roughly **3 : 2 : 1.5**. That delta is
what feeds Wave 79 Worker A's task-oriented training-pair budget.

## Authoring checklist (per template instance)

- [ ] `data-cf-template-type` is one of the four canonical values
      above.
- [ ] All template-specific required attributes are present.
- [ ] The section also carries the existing wave-stable attributes:
      `data-cf-objective-id`, `data-cf-bloom-level`,
      `data-cf-content-type`, and (when DART source material is
      available) `data-cf-source-ids` per `Courseforge/CLAUDE.md` §
      "HTML Data Attributes".
- [ ] Counter-example / misconception paragraphs (Templates 2 and 3)
      carry the `data-cf-counter-example="true"` /
      `data-cf-misconception="true"` markers respectively.
- [ ] Pattern 22 prevention still applies: each template instance
      lives alongside the surrounding 400+ word theoretical
      foundation; templates do not REPLACE explanation, they
      AUGMENT it.
