# DART/Semantic — Theta (post-WCAG document-intelligence score) investigation

> **STATUS (banner added 2026-06-17): SHIPPED.** Theta semantic-preservation v8 (full-FT
> DeBERTa-v3-small) is live in the runtime cascade. Retained as the investigation record.

**Plan version.** 2026-05-03
**Self-contained.** No reliance on prior conversation. Cross-references go to
`architecture.md`, `Plans/01_implementation_plan.md`, `docs/ontology.md`, and
the named v1 source files.
**Authoritative architecture.** `architecture.md` §1, §5, §6, §10.

---

## 0. Frame

Theta is a proposed **post-WCAG meaning-preservation score**. It runs *after*
the document-level hard gate has passed and the document-level soft reranker
has chosen an assembled HTML. It does not gate, it does not override the hard
gate, and it does not change WCAG conformance status. It augments the
ship-with-confidence exit with a numeric meaning-preservation report and may
trigger the offline-Qwen lane as a soft policy.

The key constraint that everything in this document must respect is the locked
DART principle (`architecture.md` §1):

> **Learned models are narrow candidate generators; deterministic code
> orchestrates, gates, and assembles.**

Anything that tries to wedge a single fat learned model into theta — across
eight conceptually different dimensions — violates that principle and is
rejected on those grounds.

---

## 1. Is theta one BERT, several learned scores, or a composite?

### 1.1 The three options stated honestly

**Option A — One end-to-end cross-encoder.** A single ModernBERT-base or
DeBERTa-v3-small cross-encoder takes `(source PDF text, assembled HTML)` and
emits `theta_score` plus the eight per-dimension scores from one shared
representation. Single training objective; one model on disk; ~600 MB warm.

- Pros: simple deployment, swaps in/out as one checkpoint, trivially batched.
- Cons (load-bearing):
  1. Eight conceptually different dimensions are squashed through one shared
     representation. Reference-integrity is a graph-traversal property
     (does `href="#sec-3-2"` resolve?). A learned head over a 768-d
     pooled vector cannot reliably emit that — it can only memorize the
     correlation patterns it saw in training.
  2. Calibration of individual dimensions is **not auditable**. A
     procurement reviewer asking "why did reference_integrity score 0.6?"
     gets "the model said so." This is exactly the failure mode that
     `architecture.md` §1 was written to forbid.
  3. Training labels for "semantic preservation" at document scale are
     extremely expensive. There is no ar5iv-style ground truth for "how
     well did this remediation preserve meaning."
  4. Wraps a single model around eight decisions — the opposite of the
     "narrow decision surface" rule.

**Option B — Per-dimension small models or rule-based scorers.** Each
dimension has its own scorer. Most are deterministic (graph traversal, regex,
counting). One or two are narrow learned models. Theta is a weighted sum or a
small calibrated combiner over the per-dimension scores.

- Pros:
  1. Reference integrity, fragmentation, hallucinated-structure, navigation
     clarity, structural coherence, cognitive load are **all expressible
     deterministically** with real signal — the source PDF, the assembled
     DOM, and the council's typed signals already give us everything we
     need for those six.
  2. The remaining one or two learned dimensions (semantic preservation,
     and possibly cognitive-load risk) are narrow — exactly the same
     "narrow learned model" pattern as the BERT council.
  3. Each per-dimension score is auditable. A failing
     reference_integrity score names *which* refs broke. A failing
     hallucinated-structure score names *which* generated tokens have no
     source antecedent.
  4. Versioning and weight-locking are tractable (Section 7.3).
- Cons:
  - More moving pieces; weight tuning across dimensions needs ground
    truth.
  - Composite score calibration drifts as upstream phases change.

**Option C — Wrapper over the existing Stage 11 doc-level soft reranker plus
1–2 extra learned scores.** Stage 11 already scores heading-tree balance,
landmark coverage, ref-link integrity, document-outline cleanliness — that
overlaps theta dimensions 2 (structural coherence), 3 (navigation clarity),
and 5 (reference integrity). Reuse Stage 11; add semantic-preservation and
cognitive-load on top.

- Pros: Maximum reuse, minimum new model count.
- Cons (load-bearing):
  1. Stage 11's training objective is **"pick the best assembly among
     gate-survivors"** (a relative pairwise objective). Theta needs an
     **absolute quality score** on the chosen assembly. The two
     objectives are different and a model trained on relative ranking
     does not transfer cleanly to absolute calibration. (Concretely:
     Stage 11 can put assembly-A above assembly-B without committing
     to whether either is "good.")
  2. `Plans/01_implementation_plan.md` DP-10.1 explicitly recommends
     starting Stage 11 **rule-based**, promoting to learned only if
     calibration justifies it. Conflating theta with Stage 11 forces
     theta into either (a) the rule-based scorer, which can't cover
     semantic preservation, or (b) a learned model with the wrong
     objective.
  3. Reusing Stage 11's emitted score as part of theta's composite is
     fine; *replacing* theta with Stage 11 is not.

### 1.2 Recommendation: Option B

Build theta as a **deterministic composite over per-dimension scorers, with
one and only one learned narrow scorer (semantic preservation)**. Reuse
Stage 11's intermediate per-axis scores as deterministic inputs to theta's
structural-coherence, navigation-clarity, and reference-integrity dimensions
(don't run those checks twice). Add narrow deterministic scorers for the
remaining four. Add a single small learned cross-encoder for semantic
preservation only.

This is the only option consistent with `architecture.md` §1.
The recommendation is locked and the rest of this document operationalizes it.

**Argument against Option A.** Squashing eight dimensions through one
representation makes per-dimension scores uncalibrated and unauditable, which
is exactly what a procurement-grade meaning-preservation report cannot be.

**Argument against Option C.** Stage 11's "pick a winner" objective does not
transfer to "score absolute quality." Reuse the *signals* Stage 11 computes,
not its trained head.

---

## 2. Per-dimension operationalization

For each of the eight dimensions, the recommendation is below. Output range
is `[0.0, 1.0]` for every score (1.0 = best). For deterministic scorers, the
formula is given in pseudocode. For the one learned scorer, the input/target
contract is given.

| # | Dimension | Learned or deterministic? | Inputs | Algorithm or model | Output range | Notes |
|---|---|---|---|---|---|---|
| 1 | semantic preservation | **learned** (narrow cross-encoder) | source PDF prose (per top-level section) + assembled HTML prose (per top-level section) | DeBERTa-v3-small cross-encoder, single regression head, score ∈ [0,1] per section, document score = length-weighted mean | [0,1] | Only learned dimension |
| 2 | structural coherence | deterministic | assembled DOM heading tree | reuse Stage 11 heading-tree balance score; details in §2.2 | [0,1] | |
| 3 | navigation clarity | deterministic | assembled DOM landmarks + headings | landmark coverage × heading-id density × ToC-target-resolution; §2.3 | [0,1] | Reuses Stage 11 landmark-coverage score |
| 4 | context continuity | deterministic | source per-section text vs. assembled per-section text + intra-section cross-refs | (a) section preservation rate, (b) intra-section ref resolution rate; §2.4 | [0,1] | |
| 5 | reference integrity | deterministic | assembled DOM `<a href="#…">` and target `id`s | resolved_refs / total_refs (with broken-ref enumeration); §2.5 | [0,1] | Reuses Stage 11 ref-link integrity |
| 6 | cognitive-load reduction | deterministic | assembled DOM run-length features | composite of paragraph-length distribution, heading density, list-coverage, run-length variance; §2.6 — emits **risk** enum {low, medium, high} alongside the score | [0,1] + enum | Output is opinionated; calibrated against held-out OpenStax (educational gold) |
| 7 | fragmentation / ambiguity penalty | deterministic | source layout (column flow, page boundaries) + assembled DOM | fraction of source paragraphs split across multiple `<p>` after assembly + fraction of source list items broken across multiple `<li>`; §2.7 | [0,1] | Penalty inverted to 1 − fragmentation |
| 8 | hallucinated-structure penalty | deterministic | source PDF text + assembled HTML, **with gap-fill provenance** | for every gap-fill-emitted token, require a substring/paraphrase anchor in the source; §2.8 | [0,1] | Specifically targets Stage 9b gap-fill outputs |

### 2.1 Semantic preservation (the only learned dimension)

```
Input:  (source_section_text, assembled_section_html)
Model:  DeBERTa-v3-small cross-encoder, one regression head
Output: score ∈ [0, 1]
Doc score: weighted mean over sections, weights = section_token_count
```

The input is **section-level**, not document-level — a 30-page PDF will not
fit in any cross-encoder context. Sections are defined by the assembled HTML's
H1/H2 boundaries. For each section: pair the source text under the matching
PDF heading-region with the assembled HTML's section subtree (text-only,
strip tags). Assembler already aligns these via `arxiv_sections.py`-style
matching.

**Supervision signal:** §3 below.

### 2.2 Structural coherence (deterministic)

Reuse Stage 11's heading-tree balance signal directly. Theta does not
recompute it; it consumes it from the soft-reranker's emitted state.

```
score_2 = stage11.heading_tree_balance  # already in [0, 1]
```

If Stage 11 is rule-based at the time theta lands (per DP-10.1), the score is
defined as:

```
balance = 1 - normalized_variance_of_subtree_depths(heading_tree)
where depths = [depth(h) for each heading h, normalized by max depth]
```

### 2.3 Navigation clarity (deterministic)

```
landmark_coverage = stage11.landmark_coverage   # fraction of expected landmarks
heading_id_density = headings_with_id / total_headings
toc_resolution = toc_links_resolved / toc_links_total  if toc_present else 1.0
score_3 = 0.5 * landmark_coverage + 0.3 * heading_id_density + 0.2 * toc_resolution
```

The 0.5/0.3/0.2 weighting is a default; tunable in `theta/config.yaml`.

### 2.4 Context continuity (deterministic)

```
# (a) Section preservation: every source section appears in the assembled doc
matched_sections = sum(1 for s in source_sections if has_match(s, assembled))
section_preservation = matched_sections / len(source_sections)

# (b) Intra-section cross-references: refs originating in section X to anchors
# in the same section X resolve at >= 99% (this is the easy case; if even
# this fails, something is structurally wrong).
intra_resolved = sum(1 for r in intra_section_refs if r.resolves)
intra_section_ref_rate = intra_resolved / max(1, len(intra_section_refs))

score_4 = 0.7 * section_preservation + 0.3 * intra_section_ref_rate
```

### 2.5 Reference integrity (deterministic)

Reuse Stage 11's ref-link integrity signal directly. Augment with a broken-ref
list for the `breakdown` field (Section 5):

```
total_refs = len(all_href_anchors_in_assembled)
broken_refs = [a for a in all_href_anchors if a.target_id not in document_ids]
score_5 = 1.0 - (len(broken_refs) / max(1, total_refs))
```

Carries forward the actual list of broken-ref ids in the report so a consumer
can find them.

### 2.6 Cognitive-load reduction (deterministic, opinionated)

The hardest one. Pick measurable signals; calibrate against OpenStax (a
deliberately accessible educational corpus is the closest thing to "gold
cognitive-load reduction" we have access to without licensing burden).

```
# All measured on the assembled doc.
para_lengths = [len(p.text) for p in assembled.find_all("p")]
para_p99 = percentile(para_lengths, 99)
median_para = median(para_lengths)
heading_density = len(headings) / max(1, len(paragraphs))   # ratio
list_coverage = sum(len(li) for li in lists) / max(1, sum(len(p) for p in paragraphs))
runlength_cv = stdev(para_lengths) / max(1, mean(para_lengths))   # CV

# Sub-scores (each 0..1; calibrated on OpenStax baselines):
s_para_length = clip(1 - (para_p99 - 600) / 1200, 0, 1)
s_heading_density = clip(heading_density / 0.10, 0, 1)   # >=10% of paragraphs is a heading
s_list_coverage = clip(list_coverage / 0.20, 0, 1)
s_runlength_cv = clip(1 - abs(runlength_cv - 0.6) / 0.6, 0, 1)

score_6 = 0.4 * s_para_length + 0.3 * s_heading_density + 0.2 * s_list_coverage + 0.1 * s_runlength_cv
risk_6 = "low" if score_6 >= 0.75 else "medium" if score_6 >= 0.55 else "high"
```

The constants (600, 1200, 0.10, 0.20, 0.6) are calibrated once on a held-out
OpenStax sample, then locked alongside the theta version. Re-tune only on a
ThetaConfig version bump (Section 7).

### 2.7 Fragmentation / ambiguity penalty (deterministic)

```
# Fragmentation = source structure that got split across more elements than warranted.
src_paragraphs = source_paragraph_count    # from layout extraction
asm_paragraphs = len(assembled.find_all("p"))
para_fragmentation = max(0, asm_paragraphs - src_paragraphs) / max(1, src_paragraphs)

src_list_items = source_list_item_count
asm_list_items = len(assembled.find_all("li"))
list_fragmentation = max(0, asm_list_items - src_list_items) / max(1, src_list_items)

score_7 = 1 - clip((para_fragmentation + list_fragmentation) / 2, 0, 1)
```

**Important guard.** A *legitimate* WCAG remediation may split one source
"paragraph" (which was a layout artifact spanning columns) into two semantic
paragraphs. This is correct, not a fragmentation failure. The deterministic
signal is intentionally noisy; theta's interpretation rule in §7.2 says
fragmentation alone never triggers a confidence-down action — only
fragmentation in conjunction with low semantic_preservation does.

### 2.8 Hallucinated-structure penalty (deterministic, gap-fill-aware)

The safety net for Stage 9b gap-fill outputs. Scope is narrow because
gap-fill scope is narrow (`architecture.md` §4): titles, citation/footnote
resolution, author/copyright/legal blocks.

```
# For each gap-fill-emitted span (carries provenance from Stage 9b):
hallucination_count = 0
for span in gap_fill_spans(assembled):
    # The substring-or-paraphrase contract:
    if span.kind == "title":
        if not has_substring_or_paraphrase(span.text, source.first_h1_or_abstract):
            hallucination_count += 1
    elif span.kind == "citation_resolution":
        # Generated href must match an existing target_id whose text
        # is also present in the source.
        if span.target_id not in document_ids:
            hallucination_count += 1
        elif not has_substring(document_text(span.target_id), source.text):
            hallucination_count += 1
    elif span.kind in {"author", "copyright", "legal"}:
        # All tokens must appear in the source (allow casing/whitespace).
        if not all_tokens_in(span.text, source.text):
            hallucination_count += 1

score_8 = 1 - hallucination_count / max(1, total_gap_fill_spans)
# If no gap-fill ran, score_8 = 1.0 (vacuously).
```

**This dimension is the load-bearing safety net for the gap-fill adapter.** If
gap-fill hallucinates a title or citation, theta catches it and the document
gets `ship_with_flag` (Section 4) regardless of any other dimension's score.

### 2.9 Composite

```
weights = {
    1: 0.30,  # semantic_preservation        (the dominant axis)
    2: 0.10,  # structural_coherence
    3: 0.10,  # navigation_clarity
    4: 0.10,  # context_continuity
    5: 0.10,  # reference_integrity
    6: 0.10,  # cognitive_load_reduction
    7: 0.05,  # fragmentation_penalty
    8: 0.15,  # hallucinated_structure_penalty   (load-bearing for gap-fill)
}
theta_score = sum(weights[i] * score_i for i in 1..8)
```

Weights sum to 1.0. Locked alongside theta version (Section 7.3).

---

## 3. Training data for the learned components

Only one dimension is learned: **semantic preservation** (§2.1).
Recommendation for the supervision signal — combine three sources, weighted:

### 3.1 Source A: Synthetic perturbations of WCAG-clean ar5iv pairs (primary, 70% of train data)

Take a WCAG-clean ar5iv pair `(source_pdf_text, ground_truth_html)`. Apply
calibrated perturbations to the HTML and label the score:

| Perturbation | Target score |
|---|---|
| identity (no change) | 1.0 |
| delete one random `<p>` | 0.85 |
| delete two random `<p>` | 0.75 |
| delete one full `<section>` | 0.50 |
| paraphrase one `<p>` (synonym substitution at 10%) | 0.92 |
| paraphrase one `<p>` (synonym substitution at 30%) | 0.78 |
| inject a fabricated `<p>` (out-of-source text) | 0.70 |
| inject a fabricated `<section>` | 0.40 |
| shuffle two adjacent `<section>`s | 0.85 (structure shifted but semantics intact) |
| shuffle non-adjacent sections (more disruptive) | 0.70 |

The numeric labels above are seed values; Phase A.1 (calibration) re-fits the
labels by training a model on these pairs and checking that two known-good
real outputs score higher than known-bad real outputs. This is rough
supervision — the goal is **monotonicity in the right direction**, not
calibrated absolute values.

Cheap (no human annotation), abundant (every ar5iv pair multiplies into
~10 perturbed pairs), and consistent with `architecture.md` §7.4's existing
ar5iv data infrastructure.

### 3.2 Source B: Bootstrap from offline-Qwen lane outputs (secondary, 20%)

Once the offline-Qwen lane (Phase 10) is running, every doc that fast-lane
*failed* and offline-lane *passed* gives a known-better-than-fast-lane
example. Pair `(source, offline_html)` as the higher-scoring twin of
`(source, fast_lane_html)` if the fast lane produced any output before
failing the gate.

This gives **pairwise** ranking signal at zero label cost. Cannot be the
primary source because it only exists after Phase 10 ships; theta's first
training run must rely on Source A alone.

### 3.3 Source C: Proxy from existing eval signals (tertiary, 10%)

`scripts/eval_v7_family.sh` already emits per-document axe verdicts and
diff-from-source proxies. A doc that passes axe with a smaller diff-from-
source is weak signal of higher semantic preservation. Use as a low-weight
training example only; this signal is noisy enough that it would dominate the
loss if weighted heavily.

### 3.4 Pairwise human preference — explicitly NOT in v1

Pairwise human preference (`(A, B, A_better) → A > B`) is the gold-standard
supervision signal for "how well did this remediation preserve meaning." It
is also the slowest and most expensive. v1 of theta does not use it.

If theta's calibration on Source A + B turns out poor (Q4 in §10), revisit:
budget ~500 pairwise preference labels on the broader eval corpus (arXiv +
OpenStax + IRS), at the cost of ~2 weeks of human time, and use them as a
held-out calibration set rather than a training set.

### 3.5 Corpus mix per dimension

Theta evaluates on **whatever the document is**, but its semantic-preservation
training data should reflect the same mix as the broader pipeline
(`architecture.md` §7.4): primarily ar5iv, but with OpenStax + IRS for
calibration breadth. Cognitive-load reduction (§2.6) is calibrated explicitly
on OpenStax because OpenStax is the only corpus in scope that is already
designed for cognitive-load reduction.

### 3.6 Data budget

- ar5iv perturbation pairs: ~5K source pairs × 10 perturbations = 50K labeled
  examples. Free.
- Offline-Qwen-lane bootstrap pairs: depends on Phase 10 throughput; expected
  ~500–2000 pairs per month once running.
- Eval-signal proxy: existing eval output, no new labeling.
- Pairwise human (deferred): 500 labels, deferred to v2 of theta if needed.

This is well within the 8K-pair arXiv budget — the perturbation strategy is a
multiplier, not a parallel data ask.

---

## 4. Pipeline placement and exit interaction

### 4.1 Confirmed placement

```
Stage 10: Document-level HARD gate         (eliminating)
Stage 11: Document-level soft reranker     (pick top-1 doc)
Stage 12 (NEW): ThetaEvaluator             (score the chosen doc)
Stage 13 (renumbered): Exit                (with theta in the report)
```

The user's proposed placement is correct; the only refinement is to make the
exit *Stage 13* (theta is its own stage, not folded into the exit decision).

### 4.2 When does theta run?

**Decision: theta runs on every doc that exits the fast-lane Stage 11, and on
every doc that exits the offline-lane Stage 11.** It does NOT run on docs
that take the non-certified-stamp exit (those, by definition, failed the
hard gate, and theta is undefined on a non-conformant doc).

Cost basis: theta's deterministic dimensions are O(milliseconds) on a fully
assembled DOM. The one learned dimension (semantic preservation) is a
DeBERTa-v3-small cross-encoder run on ~10 sections per doc — sub-second on
the dev GPU. Running theta on every passing doc is cheap and gives the
consumer a uniform report.

### 4.3 Does theta run on the non-certified-stamp exit?

**No.** Theta's contract assumes the doc passed WCAG. If a doc failed both
fast-lane and offline-lane, the doc's HTML may have structural defects that
break theta's deterministic checks (e.g., orphaned `<a href="#x">` to
non-existent ids). The non-certified stamp exit ships the HTML as-is with
the stamp; theta is omitted from the report and `theta_score: null`,
`wcag_status: "failed"` are emitted in the JSON.

### 4.4 Does theta-low trigger offline-Qwen retry?

**Yes, with a strict policy.** The rule:

```
if exit_lane == "fast" and theta_score < TAU_THETA_RETRY:
    if not previously_retried:
        rerun document through offline-Qwen lane
        re-evaluate theta
        compare:
            if offline_theta > fast_theta + DELTA_THETA_IMPROVE:
                ship offline output, action = ship_with_confidence
            else:
                ship the higher-theta output, action = ship_with_flag
                flag = "meaning-preservation review recommended"
```

Defaults: `TAU_THETA_RETRY = 0.70`, `DELTA_THETA_IMPROVE = 0.05`. Both in
`theta/config.yaml`, both versioned.

### 4.5 Edge case — fast-lane WCAG passes but offline-lane theta is higher

This is exactly the case §4.4 handles. The offline-lane output ships if
`offline_theta > fast_theta + DELTA_THETA_IMPROVE`. Otherwise the fast-lane
output ships (it passed WCAG; meaning-preservation is a soft signal and we
don't gratuitously re-do work).

### 4.6 Per-dimension thresholds

**Yes — alongside the composite threshold.** Specific per-dimension floors:

| Dimension | Floor | If below | Flag emitted |
|---|---|---|---|
| reference_integrity | 0.80 | flag "broken citations" with the broken-ref list | `broken_refs_present` |
| hallucinated_structure_penalty | 0.85 | flag "potential gap-fill hallucination" with span list | `gap_fill_review_recommended` |
| semantic_preservation | 0.65 | flag and trigger §4.4 retry policy | `meaning_preservation_low` |
| cognitive_load_risk == "high" | n/a | flag "cognitive load high" | `cognitive_load_high` |

Other dimensions do not have hard floors; their effect is via the composite.

The action enum is determined by the strictest signal:
- if any dimension floor is hit → at minimum `ship_with_flag`.
- if `theta_score < TAU_THETA_RETRY` and not retried → `retry_offline`.
- if `theta_score >= TAU_THETA_SHIP` (default 0.80) and no floor hit →
  `ship_with_confidence`.

### 4.7 Updated exit decision matrix

| WCAG hard gate | Lane | Theta | Floor breach | Action |
|---|---|---|---|---|
| pass | fast | ≥ 0.80 | none | ship_with_confidence |
| pass | fast | 0.70–0.80 | none | ship_with_flag (`meaning_preservation_borderline`) |
| pass | fast | < 0.70 | (any) | retry_offline (once) |
| pass | fast | any | floor breach | ship_with_flag (specific flag per §4.6) |
| pass | offline | ≥ 0.80 | none | ship_with_confidence |
| pass | offline | < 0.80 | none | ship_with_flag (`meaning_preservation_borderline`) |
| pass | offline | any | floor breach | ship_with_flag |
| fail | fast | n/a | n/a | offline-Qwen lane (existing) |
| fail | offline | n/a | n/a | non_certified_stamp; theta omitted |

`retry_offline` is a single retry; the second pass cannot loop back to retry
again. This caps cost.

---

## 5. Output schema

```json
{
  "schema_version": "theta/1.0",
  "wcag_status": "passed",                   // "passed" | "failed"
  "lane": "fast",                            // "fast" | "offline"
  "theta_score": 0.87,                       // null if wcag_status == "failed"
  "theta_version": "theta-config-1.0",       // pins weights + thresholds + scorer versions
  "dimensions": {
    "semantic_preservation": {
      "score": 0.94,
      "section_scores": [
        {"section_id": "abstract",  "score": 0.96},
        {"section_id": "sec-1",     "score": 0.93}
      ],
      "model_version": "theta-semantic-deberta-1.0"
    },
    "structural_coherence": {
      "score": 0.89,
      "source": "stage11.heading_tree_balance"
    },
    "navigation_clarity": {
      "score": 0.82,
      "landmark_coverage": 1.0,
      "heading_id_density": 0.75,
      "toc_resolution": null                 // null when no ToC present
    },
    "context_continuity": {
      "score": 0.86,
      "section_preservation": 0.95,
      "intra_section_ref_rate": 0.70
    },
    "reference_integrity": {
      "score": 0.91,
      "total_refs": 47,
      "broken_refs": ["sec-3-2", "fig-4"]
    },
    "cognitive_load_reduction": {
      "score": 0.78,
      "risk": "low"                           // enum: "low" | "medium" | "high"
    },
    "fragmentation_penalty": {
      "score": 0.92,
      "para_fragmentation": 0.04,
      "list_fragmentation": 0.03
    },
    "hallucinated_structure_penalty": {
      "score": 1.00,
      "gap_fill_spans_evaluated": 2,
      "hallucinated_spans": []
    }
  },
  "flags": [],                                // populated by §4.6 rules
  "action": "ship_with_confidence",           // enum, see §9
  "retry_history": []                         // [{lane: "fast", theta_score: 0.65, ...}] when retried
}
```

Notes on the schema:

- `schema_version` is the JSON envelope version. Bumped on shape changes.
- `theta_version` pins the *content* version: weights, thresholds, learned-
  model checkpoint hashes. A consumer can match
  `(theta_version, schema_version)` against a registry to know what produced
  the score.
- Every dimension carries a `score` plus its own breakdown fields. The
  breakdown fields are the audit trail — when reference_integrity is 0.6,
  the consumer sees *which* refs broke.
- `flags` is a list of string codes from a fixed enum (Section 9) so
  consumers can filter without parsing prose.
- `action` is the recommended downstream behavior. The consumer ultimately
  decides; theta only recommends.
- `retry_history` is empty unless the document was retried under §4.4. When
  retried, it carries the prior pass's full theta report so the consumer can
  see what changed.

---

## 6. Where in the build plan does theta land?

### 6.1 New phase

**Phase 11 — Theta evaluator.** Position after Phase 10. Hard prerequisites:

- Phase 9 (assembler with gap-fill provenance — §2.8 needs gap-fill spans
  tagged with their kind and source-anchor).
- Phase 10 (doc gate + Stage 11 soft reranker — theta consumes Stage 11's
  emitted axis scores).

Phase 11 is **not** a prerequisite for retiring v1. v1 retirement is
governed by `Plans/01_implementation_plan.md` §5.2 and does not depend on
theta. Theta is additive product surface, not a gate.

### 6.2 Complexity

**M** (~3 weeks). Breakdown:

- Deterministic dimensions (2, 3, 4, 5, 7, 8): ~1 week including unit
  tests against the four held-out arXiv papers.
- Semantic-preservation learned dimension: ~1.5 weeks including data
  generation (perturbation pipeline), training, and calibration on
  OpenStax held-out.
- Composite + config + JSON schema + integration into exit selector:
  ~3 days.
- Cognitive-load calibration on OpenStax held-out: ~2 days.

### 6.3 Earlier phases that need a small revision

- **Phase 0.** Adds `dart_semantic/theta/types.py` (full content in
  Section 9). Adds `theta_score: float | None` and `theta_report:
  ThetaReport | None` to whatever the doc-level result type is in
  `pipeline_v2.py`. Adds `ConfidenceAction` enum.
- **Phase 9.** Gap-fill spans must carry provenance: `kind`, `source_anchor`,
  `assembler_slot_id`. This is needed by §2.8 hallucinated-structure scoring.
  If Phase 9 has already shipped without this, add it as a small follow-up
  patch in Phase 11.
- **Phase 10.** Stage 11's soft reranker must emit its per-axis scores
  (heading-tree balance, landmark coverage, ref-link integrity), not just
  the chosen-doc decision. This is a small addition to the existing
  emitted state — theta consumes those scores instead of recomputing them.

### 6.4 Decision points specific to Phase 11

- **DP-11.1.** Cross-encoder family for semantic preservation: DeBERTa-v3-
  small (≈140 MB) vs. ModernBERT-base (≈600 MB shared with the council).
  Recommend **DeBERTa-v3-small** since it doesn't share the council's LoRA
  swap discipline and can be loaded as a separate small model without
  competing for backbone slots. Final decision deferred to Phase 11 start.
- **DP-11.2.** Theta weighting strategy: hand-tuned (current §2.9) vs. fit
  to held-out. Start hand-tuned. Promote to fit-on-held-out only if hand-
  tuned shows uncalibration on the held-out OpenStax + IRS sample.
- **DP-11.3.** Per-document-type thresholds vs. global threshold. **Start
  global.** Per-doc-type thresholds (academic vs. legal vs. educational) are
  a v2-of-theta concern — see Section 8 Q5.

---

## 7. Risks and edge cases

### 7.1 Theta inflation

If theta is reported alongside WCAG status, downstream consumers may start
quoting high theta as a public marketing claim ("99% meaning preservation").
This is a real risk: theta is opinionated, weight-tuned, and drifts with
training data.

**Position: theta is a developer-facing and consumer-facing internal
diagnostic, not a public marketing metric.** Specifically:

- The conformance statement template in `docs/ontology.md` §6 must NOT
  include theta. WCAG conformance is the public claim. Theta is the
  "we were extra careful" diagnostic.
- The product UI may show theta_score to consumers as a numeric or
  letter-grade equivalent ("A/B/C") — same way axe-core scores are exposed.
- Public marketing copy must not quote theta directly. The strongest
  permitted public claim is WCAG conformance plus a qualitative statement
  ("meaning-preservation diagnostics included").

This is a product policy. Theta cannot be the marketing axis because
nothing learned + composite + weight-tuned ever should be.

### 7.2 Theta-WCAG decoupling failures

A *valid* WCAG remediation may *legitimately* trigger a low theta score:

- Flattening a sidebar into linear flow is correct WCAG (see
  `docs/ontology.md` §2.3) but raises fragmentation_penalty (§2.7).
- Splitting a multi-column "paragraph" that PDF layout merged into one is
  correct, but raises para_fragmentation.
- Replacing an image-of-equation with native MathML
  (`docs/ontology.md` §2.10) changes the source-to-assembled token mix —
  semantic_preservation may drop slightly because the cross-encoder was
  not trained on MathML in HTML.

**Mitigation rules (locked into theta's interpretation logic):**

1. Fragmentation alone never triggers retry or flag. It contributes to the
   composite at weight 0.05; even a fragmentation_penalty of 0.0 only
   reduces theta_score by 0.05.
2. The retry rule (§4.4) requires `theta_score < TAU_THETA_RETRY` —
   a single low-weight dimension cannot push the composite under 0.70 by
   itself.
3. Per-dimension floors (§4.6) are deliberately set high enough that they
   trip on real failures, not legitimate remediation choices. The floor
   for hallucinated_structure_penalty is 0.85, which one or two
   gap-fill spans can clear; the floor for reference_integrity is 0.80,
   which a handful of legitimately-broken refs in a noisy source can
   clear.

### 7.3 Calibration drift

Theta is rule-based with hand-tuned weights, plus one learned cross-encoder.
All of these change under maintenance. Versioning strategy:

- `theta_version = "theta-config-{major}.{minor}"`. Every config change
  (weights, thresholds, sub-score constants) bumps minor; every algorithm
  change to a deterministic dimension bumps major.
- The learned semantic-preservation cross-encoder has its own version
  string `theta-semantic-deberta-{N}` referenced inside the report
  (Section 5).
- Both versions must appear in the report. A consumer can detect that
  theta scores are not directly comparable across versions and either
  re-evaluate or flag the inconsistency.
- A theta version is **locked** for the lifetime of a model release. Once
  shipped, it does not auto-update. Re-running theta on an old document
  with a newer version is allowed but produces a new report; the old
  report remains the canonical record for that release.
- Calibration suite: `scripts/eval_theta.sh` runs theta on a fixed
  20-document corpus (10 arXiv + 5 OpenStax + 5 IRS) every time theta is
  trained or its config changes. Score deltas above ±0.05 on any document
  block the release until justified.

### 7.4 Adversarial gap-fill hallucinations

Gap-fill outputs can hallucinate plausible-but-wrong titles or citations.
This is the explicit threat model for dimension #8 (§2.8). Specifically:

- **Title-inference hallucination.** Gap-fill produces a "plausible" title
  that has no anchor in the source. Detected by §2.8 substring-or-paraphrase
  check against `source.first_h1_or_abstract`. If failed, span counted
  as hallucinated; floor-breach flag set.
- **Citation-resolution hallucination.** Gap-fill resolves "Section 3.2" to
  an `id` that exists in the doc but whose target text is unrelated to the
  citation context. Detected by §2.8 substring check on
  `document_text(target_id)` against the source.
- **Author/copyright/legal block hallucination.** Gap-fill produces author
  names or copyright text not in source. Detected by the all-tokens-in
  check.

This is the load-bearing safety net for the gap-fill adapter. Without
theta's hallucinated_structure_penalty, the system has no quality check on
gap-fill outputs other than the WCAG hard gate, which does not detect
content fabrication.

### 7.5 Theta on the offline-Qwen lane

The offline lane uses higher candidate counts and looser sampling (`Plans/
01_implementation_plan.md` Q7). This raises hallucination risk. Theta's
hallucinated_structure_penalty floor (0.85) is the same on both lanes — the
offline lane does not get a more generous floor. If the offline lane
trips the floor, the doc still gets `ship_with_flag`. The offline lane is
not a "trust me harder" lane; it is a "try harder" lane.

### 7.6 Cost ceiling

Theta on every passing doc is cheap, but the §4.4 retry policy potentially
doubles the offline-Qwen lane cost. Cap: at most one retry per document.
If retry policy starts firing on >10% of docs, that is signal that
TAU_THETA_RETRY is too high; tune down rather than expand the retry budget.

---

## 8. Open questions for the user

Numbered. One-sentence decisions.

1. **Q1.** Confirm theta runs on every doc that passes WCAG (both lanes),
   not only on docs where Stage 11 had multiple candidate assemblies to
   choose from?
2. **Q2.** Confirm the §4.4 retry policy: theta-low fast-lane output triggers
   one and only one offline-Qwen retry, with the higher-theta output shipped?
3. **Q3.** Is theta a public-facing metric or internal-only? (Recommendation:
   internal/consumer-diagnostic; never on public marketing.)
4. **Q4.** Lock theta weights at model-release time, or update with each
   training run? (Recommendation: lock per release; re-run an old release's
   theta only on explicit request.)
5. **Q5.** Should per-dimension thresholds be per-document-type (academic
   vs. legal vs. educational) or one global threshold? (Recommendation:
   global in v1; per-type is a v2 concern.)
6. **Q6.** TAU_THETA_RETRY default: 0.70 acceptable, or tighter/looser?
7. **Q7.** TAU_THETA_SHIP default: 0.80 acceptable for `ship_with_confidence`?
8. **Q8.** Should the cognitive_load_risk enum be exposed in the consumer
   report, or hidden behind an internal-only flag? (Recommendation: expose
   the enum, hide the underlying numeric components.)
9. **Q9.** Confirm the eight-dimension list is exhaustive for v1? Specifically
   missing: (a) image alt-text quality, deferred because gap-fill does not
   handle alt-text in v1; (b) language-of-parts compliance (`docs/ontology.md`
   §2.10 gap), deferred because parsers do not yet propagate `lang=`.
10. **Q10.** Confirm the Phase 11 placement (after Phase 10) and that no
    earlier phase needs to be revised beyond the small additions in §6.3?

---

## 9. Implications for Phase 0 type definitions

Phase 0 currently lands `dart_semantic/assembler/types.py` and similar
package skeletons. Theta adds a new package:

```
dart_semantic/theta/
  __init__.py
  types.py
  config.yaml
```

### 9.1 `dart_semantic/theta/types.py`

```python
"""Theta (post-WCAG meaning-preservation) typed report.

The shape of this module is locked by Plans/03_theta_investigation.md §9.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ThetaDimension(str, Enum):
    SEMANTIC_PRESERVATION = "semantic_preservation"
    STRUCTURAL_COHERENCE = "structural_coherence"
    NAVIGATION_CLARITY = "navigation_clarity"
    CONTEXT_CONTINUITY = "context_continuity"
    REFERENCE_INTEGRITY = "reference_integrity"
    COGNITIVE_LOAD_REDUCTION = "cognitive_load_reduction"
    FRAGMENTATION_PENALTY = "fragmentation_penalty"
    HALLUCINATED_STRUCTURE_PENALTY = "hallucinated_structure_penalty"


class CognitiveLoadRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceAction(str, Enum):
    SHIP_WITH_CONFIDENCE = "ship_with_confidence"
    SHIP_WITH_FLAG = "ship_with_flag"
    RETRY_OFFLINE = "retry_offline"
    NON_CERTIFIED = "non_certified"


class ThetaFlag(str, Enum):
    BROKEN_REFS_PRESENT = "broken_refs_present"
    GAP_FILL_REVIEW_RECOMMENDED = "gap_fill_review_recommended"
    MEANING_PRESERVATION_LOW = "meaning_preservation_low"
    MEANING_PRESERVATION_BORDERLINE = "meaning_preservation_borderline"
    COGNITIVE_LOAD_HIGH = "cognitive_load_high"


@dataclass(frozen=True)
class SectionScore:
    section_id: str
    score: float


@dataclass(frozen=True)
class DimensionScore:
    dimension: ThetaDimension
    score: float                      # always [0.0, 1.0]
    breakdown: dict = field(default_factory=dict)   # dimension-specific fields


@dataclass(frozen=True)
class ThetaConfig:
    """Versioned thresholds + weights. Loaded from theta/config.yaml.

    Locked at model-release time. See Plans/03_theta_investigation.md §7.3.
    """
    version: str                              # e.g. "theta-config-1.0"
    weights: dict                             # {ThetaDimension: float}; sum to 1.0
    tau_theta_ship: float = 0.80
    tau_theta_retry: float = 0.70
    delta_theta_improve: float = 0.05
    floor_reference_integrity: float = 0.80
    floor_hallucinated_structure: float = 0.85
    floor_semantic_preservation: float = 0.65
    semantic_model_version: str = "theta-semantic-deberta-1.0"


@dataclass(frozen=True)
class ThetaReport:
    """The canonical theta report. JSON shape in §5."""
    schema_version: str                       # e.g. "theta/1.0"
    wcag_status: str                          # "passed" | "failed"
    lane: str                                 # "fast" | "offline"
    theta_score: float | None
    theta_version: str
    dimensions: dict                          # {ThetaDimension: DimensionScore}
    flags: list                               # list[ThetaFlag]
    action: ConfidenceAction
    retry_history: list = field(default_factory=list)   # list[ThetaReport]
```

### 9.2 `dart_semantic/theta/config.yaml`

```yaml
version: theta-config-1.0
weights:
  semantic_preservation: 0.30
  structural_coherence: 0.10
  navigation_clarity: 0.10
  context_continuity: 0.10
  reference_integrity: 0.10
  cognitive_load_reduction: 0.10
  fragmentation_penalty: 0.05
  hallucinated_structure_penalty: 0.15
tau_theta_ship: 0.80
tau_theta_retry: 0.70
delta_theta_improve: 0.05
floor_reference_integrity: 0.80
floor_hallucinated_structure: 0.85
floor_semantic_preservation: 0.65
semantic_model_version: theta-semantic-deberta-1.0
cognitive_load_calibration:
  para_p99_ceiling: 600
  para_p99_clip_range: 1200
  heading_density_target: 0.10
  list_coverage_target: 0.20
  runlength_cv_target: 0.6
```

### 9.3 Touch points in existing types

`dart_semantic/types.py` and (Phase 0) `dart_semantic/assembler/types.py`
gain:

```python
# in pipeline_v2 result shape
@dataclass
class PipelineResult:
    html: str
    wcag_status: str
    exit_action: ConfidenceAction      # from theta/types
    theta_report: ThetaReport | None   # None if non_certified or theta not run
    # ... existing fields
```

Phase 0 lands these as empty/None defaults so v1 path keeps working
unchanged. Phase 11 fills in `theta_report`.

---

## 10. Concrete recommendation

Build theta as **a deterministic composite over per-dimension scorers, with a
single narrow learned cross-encoder for semantic preservation**. Reuse Stage
11's already-emitted axis scores for structural coherence, navigation clarity,
and reference integrity — do not recompute them. Land theta as a new
**Phase 11** in the implementation plan, after Phase 10 ships. Phase 0 lands
the typed dataclasses (`ThetaReport`, `ThetaConfig`, `ThetaDimension`,
`ConfidenceAction`, `ThetaFlag`) so subsequent phases have the contract.

**Don't do these things:**

1. **Don't build theta as one end-to-end BERT** — eight conceptually distinct
   dimensions through one shared representation produces uncalibrated,
   unauditable scores. Violates `architecture.md` §1.
2. **Don't fold theta into Stage 11** — Stage 11's training objective is
   "pick the best assembly," theta's is "score absolute quality." Different
   objectives; reuse signals, not the trained head.
3. **Don't let theta be a public marketing number.** WCAG conformance is the
   public claim. Theta is a diagnostic; it must not become the marketing
   axis.
4. **Don't gate on theta.** Theta never overrides the WCAG hard gate; it can
   recommend retry or flag, never override. The hard gate's eliminating
   semantics are load-bearing for procurement.
5. **Don't include image alt-text quality or language-of-parts in v1
   theta.** Both are gap-analysis items in `docs/ontology.md` §7 that are
   not yet remediated upstream; theta cannot meaningfully score them
   until parsers and gap-fill can produce them.
6. **Don't add pairwise human preference labels in v1.** Synthetic
   perturbations of WCAG-clean ar5iv pairs (§3.1) plus offline-lane
   bootstrap pairs (§3.2) are sufficient for v1. Defer human preference to
   v2 of theta if calibration drift forces it.
7. **Don't run theta on non-certified-stamp output.** Theta is undefined
   on non-conformant HTML. Emit `theta_score: null, wcag_status: "failed"`.

---

*End of investigation.*
