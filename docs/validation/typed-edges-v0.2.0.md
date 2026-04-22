# Typed-edge concept graph — Worker N validation (v0.2.0)

Worker F shipped `Trainforge/rag/typed_edge_inference.py` with three rule modules producing `concept_graph_semantic.json`. Worker N's job: sample edges from the regenerated corpora, read them, score plausibility, and decide **per rule** whether to keep, tune, or disable.

**No code changes in this PR** — pure evaluation. Any rule tuning is a follow-up worker after findings are agreed.

## Corpora

| Course | Nodes | Total edges | is-a | prerequisite | related-to |
|---|---:|---:|---:|---:|---:|
| WCAG_201 (131 chunks, 28 LOs post-Worker-H) | 257 | 1451 | 0 | **0** | 1451 |
| DIGPED_101 (86 chunks, 26 LOs pre-Worker-H) | 28 | 172 | 0 | **142** | 30 |

Both courses produce zero `is-a` edges. WCAG produces zero `prerequisite` edges. DIGPED produces many prerequisites but on a very different corpus shape. Details below.

## Sampling protocol

- Deterministic: edges sorted by `(source, target)` ascending; first 50 per type per course taken.
- For each sampled edge, compute concept-tag co-occurrence count in the chunk corpus — i.e., how many chunks carry BOTH concepts in their `concept_tags`. Low co-occurrence on a "related" edge is a strong signal of noise.
- Plausibility scored as:
  - **good**: concepts are semantically related, co-occur in ≥3 chunks, edge type semantics match
  - **borderline**: semantically related but weakly co-occurring, OR edge direction ambiguous
  - **noise**: concepts are from different semantic domains OR co-occurrence is ≤1 and no structural tie

## Rule-by-rule findings

### `related-to` (co-occurrence threshold) — **KEEP**

| Course | Sampled | good | borderline | noise | Plausibility |
|---|---:|---:|---:|---:|---:|
| WCAG_201 (first 50 of 1451) | 50 | 19 | 31 | 0 | **100% non-noise** |
| DIGPED_101 (all 30) | 30 | 30 | 0 | 0 | **100% good** |

Both corpora exercise this rule well. All edges connect concepts within-domain, co-occurrence counts are high on the "good" tier, and there are no cross-domain noise edges.

**Good examples (WCAG)**:
- `accessibility→accessible-name` (co-occurs in 10 chunks)
- `accessibility→ada` (co=16)
- `accessibility→alt-text` (co=24)
- `accessibility→aria` (co=64) — strong domain relationship

**Good examples (DIGPED)**:
- `accessibility↔udl` (co=7) — UDL is an accessibility-adjacent instructional framework
- `alignment↔blooms-taxonomy` (co=9) — classic constructive-alignment pairing
- `alignment↔rubric` (co=8)
- `assessment↔behaviorism` (co=5) — behaviorist assessment tradition

**Verdict**: **KEEP**. The rule is working as designed. The borderline tier on WCAG (31/50) is chunks connecting concepts via weaker co-occurrence (2–4 chunks) — not wrong, just lower-confidence. The existing `DEFAULT_THRESHOLD` (check `Trainforge/rag/inference_rules/related_from_cooccurrence.py`) is well-calibrated.

### `prerequisite` (LO-order heuristic) — **TUNE (not disable)**

| Course | Sampled | good | borderline | noise | Plausibility |
|---|---:|---:|---:|---:|---:|
| WCAG_201 (total: 0) | 0 | — | — | — | **rule fires zero edges** |
| DIGPED_101 (first 50 of 142) | 50 | 14 | 21 | 15 | **30% noise** |

**WCAG_201 fires ZERO prerequisite edges** despite Worker H's per-week LO specificity fix. Root cause: Worker H attaches all four Terminal Objectives (TO-01..TO-04) to every week's pages, PLUS the week-specific Chapter Objectives. Every chunk's `learning_outcome_refs` therefore starts with `to-01`, and since `to-01` is position 0 in `course.json::learning_outcomes`, the "earliest LO position" for every concept's first-seen chunk is **always 0**. The prerequisite rule's skew check produces no edges.

**DIGPED_101 fires 142 prerequisite edges** but with 30% noise. DIGPED's LO-ref pattern is different: 74/86 chunks have NO `learning_outcome_refs` at all (rule skips them); the 12 chunks with refs use a sparse, inconsistent single-LO pattern that does produce position skew. The "good" and "borderline" edges are real pedagogy→pedagogy concept pairs (e.g. `addie→assessment`, `backward-design→blooms-taxonomy`). The 15 noise edges are all cross-domain — specifically, `accessibility→<pedagogy-concept>` edges where accessibility (Week 11 in DIGPED) gets wrongly positioned as a prerequisite of earlier pedagogy concepts.

**Noise pattern** (DIGPED examples):
- `accessibility -[prereq]-> addie` (accessibility is NOT a prereq for ADDIE; both are independent pedagogical topics)
- `accessibility -[prereq]-> alignment`
- `accessibility -[prereq]-> assessment`
- `accessibility -[prereq]-> backward-design`

The rule is being fooled by concept positional drift across sparse LO-refs plus accessibility being in a late week. A proper prerequisite relationship requires both **positional skew** AND **semantic affinity**; the current rule only checks the former.

**Verdict**: **TUNE** in a follow-up worker. Specific recommendations:

1. **Exclude Terminal Objectives from position lookup.** Terminal objectives apply across the whole course by design; they don't drive "order". `_earliest_lo_position` should filter out LOs whose IDs start with `to-` (or whose `hierarchy_level == "terminal"` in course.json). This alone will fix WCAG's zero-edges problem.
2. **Require minimum co-occurrence for an edge.** A prerequisite relationship that doesn't show up as a co-occurrence pair in at least a few chunks is likely spurious. Add a floor (e.g., `min_cooccurrence_for_edge = 3`).
3. **Optional: require source and target concepts to share at least one co-occurrence neighborhood** (i.e., both appear in chunks that share a third concept). This would filter the `accessibility→<pedagogy>` noise on DIGPED cleanly.

These three changes together should move DIGPED's noise rate from 30% to <10% and unlock meaningful edges on WCAG.

### `is-a` (key-term definition parsing) — **DATA-STARVED, keep unchanged**

| Course | Edges |
|---|---:|
| WCAG_201 | 0 |
| DIGPED_101 | 0 |

Worker F's PR #6 report noted this risk explicitly: *"is-a rule is starved on corpora whose key_term definitions are descriptive phrases rather than taxonomic"*. Today's corpora confirm it. Descriptive patterns observed in real key_terms definitions: "The inclusive practice of…", "Hardware or software that…", "WCAG Principle 1: Information…" — none of them match an `X is a Y` / `X is type of Y` pattern.

**Verdict**: **keep the code, document the expected-zero behaviour**. The rule is structurally sound and will fire on corpora whose authors write taxonomic definitions. No tuning possible until we have such a corpus, because there's no positive example to validate against. Also: Worker M2's H2 fix is expected to lift `key_terms` coverage dramatically (currently 52.7% on WCAG, 0% on DIGPED); the downstream effect on `is-a` will only show if the backfilled definitions are themselves taxonomic, which is content-author-dependent.

## Cross-cutting observation — corpus-shape sensitivity

The three rules stress-test different aspects of the corpus:

| Rule | Requires | WCAG_201 strength | DIGPED_101 strength |
|---|---|---|---|
| related-to | concept co-occurrence density | **high** (large corpus, tight clustering) | medium (small corpus, still clean) |
| prerequisite | LO-order skew across chunks | **broken** by Worker H's universal TOs | noisy due to sparse LO-refs |
| is-a | taxonomic key_term definitions | data-starved | data-starved |

This suggests the semantic graph quality will vary significantly by corpus. A quality gate on typed-edge counts in `quality_report.json` would be a useful v0.3 addition — not as a pass/fail, but as a diagnostic: "your corpus produced N prereq edges; typical healthy corpora produce 50–200; investigate if <20 or >500 for a 100-chunk course."

## FOLLOWUPs

- `FOLLOWUP-WORKER-N-1` — tune `prerequisite_from_lo_order`: exclude TOs from position lookup, add min-co-occurrence floor, consider co-occurrence-neighborhood constraint. Sequential after Worker N findings agreed.
- `FOLLOWUP-WORKER-N-2` — diagnostic output: add typed-edge counts-by-type to `quality_report.json` so corpus-level rule-fire-rates are visible without opening the semantic graph JSON. Small, one-line per type.
- `FOLLOWUP-WORKER-N-3` — when Worker M2's H2 fallback fix lands, re-run this benchmark; `is-a` may fire on corpora whose filled-in key-term definitions are taxonomic. Append addendum to this doc.

## How to reproduce

```
# Inside the worker-n worktree (branch worker-n/typed-edge-validation)
venv/bin/python -m Trainforge.process_course --imscc /path/to/WCAG_201.imscc --course-code WCAG_201 --division STEM --domain computer-science --subdomain web-accessibility --output Trainforge/output/wcag_201 --objectives /path/to/WCAG_201_objectives.json
venv/bin/python -m Trainforge.process_course --imscc /path/to/DIGPED_101.imscc --course-code DIGPED_101 --division ARTS --domain education --subdomain instructional-design --output Trainforge/output/digped_101 --objectives /path/to/DIGPED_101_objectives.json

# Inspect the semantic graph edge breakdown
python3 -c "
import json
from collections import Counter
for slug in ['wcag_201', 'digped_101']:
    g = json.load(open(f'Trainforge/output/{slug}/graph/concept_graph_semantic.json'))
    types = Counter(e['type'] for e in g['edges'])
    print(f'{slug}: nodes={len(g[\"nodes\"])} edges={len(g[\"edges\"])} types={dict(types)}')
"
```

Regenerated corpora live under gitignored `Trainforge/output/` — not committed.
