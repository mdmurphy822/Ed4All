# Summary vs. body retrieval — Worker O validation (v0.2.0)

Does Worker D's `chunk.summary` + `chunk.retrieval_text` earn its place in the v4 chunk schema? This report runs the recall@k benchmark over both existing courses using **hand-curated gold queries** (not LO-derived, which conflates retrieval quality with LO-tagging quality) and makes a shippable decision.

## Setup

- **Corpora.** COURSE_W (131 chunks, 20 hand-curated gold queries) and COURSE_D (86 chunks, 15 hand-curated gold queries). The gold queries live locally per the ADR-002 retrieval scope (not tracked in git); Worker O reads them from the user's local working tree.
- **Variants compared.** `text` (chunk body, baseline), `summary` (Worker D 2–3 sentence extract), `retrieval_text` (Worker D composed field = summary + joined key_terms, emitted only when the chunk carries key_terms).
- **Benchmark runner.** `Trainforge/rag/retrieval_benchmark.py::run_benchmark`, extended to accept `gold_queries_path` (new param; existing LO-derived path remains the default). Pure BM25 over the chosen text variant, `min_relevance=0.0` so top-k is comparable across variants without the production threshold unfairly hard-filtering shorter summary fields.

## Results

### COURSE_W (131 chunks × 20 hand-curated gold queries)

| Variant | recall@1 | recall@5 | recall@10 |
|---|---:|---:|---:|
| `text` (baseline) | 0.400 | 0.817 | 0.950 |
| `summary` | **0.450** | **0.858** | 0.900 |
| `retrieval_text` | **0.475** | **0.858** | 0.925 |

**Deltas vs `text` baseline:**

| Variant | Δ@1 (abs / rel) | Δ@5 (abs / rel) | Δ@10 (abs / rel) |
|---|---|---|---|
| `summary` | +0.050 / +12.5% | +0.042 / +5.1% | −0.050 / −5.3% |
| `retrieval_text` | **+0.075 / +18.8%** | +0.042 / +5.1% | −0.025 / −2.6% |

### COURSE_D (86 chunks × 15 hand-curated gold queries)

| Variant | recall@1 | recall@5 | recall@10 |
|---|---:|---:|---:|
| `text` (baseline) | 0.400 | **0.867** | **0.967** |
| `summary` | **0.467** | 0.700 | 0.867 |
| `retrieval_text` | **0.467** | 0.700 | 0.867 |

**Deltas vs `text` baseline:**

| Variant | Δ@1 (abs / rel) | Δ@5 (abs / rel) | Δ@10 (abs / rel) |
|---|---|---|---|
| `summary` | +0.067 / +16.7% | **−0.167 / −19.3%** | **−0.100 / −10.3%** |
| `retrieval_text` | +0.067 / +16.7% | **−0.167 / −19.3%** | **−0.100 / −10.3%** |

### Observations

- **Summary wins at top-1 on both courses** (+5 pts on COURSE_W, +7 pts on COURSE_D). That's the precision-first regime: if a consumer wants the single best chunk for a query, summary-based BM25 beats body-based BM25 by a clear margin.
- **Summary wins through recall@5 on COURSE_W but loses sharply on COURSE_D** (+4 pts vs. −17 pts). The cross-over point differs by course.
- **Recall@10 favors text on both courses.** Long summary chunks preserve more signal surface area; BM25's IDF weighting favors the richer vocabulary in body text when you widen the aperture.
- **`retrieval_text` is identical to `summary` on COURSE_D**, because COURSE_D has 0% `key_terms` coverage (confirmed by Worker M1's §4.4a diagnostic — COURSE_D's JSON-LD has no `sections` metadata, so no key terms ever reach chunks). The composed-field concept is sound; the COURSE_D signal here is a downstream symptom of the same H2 defect M1 surfaced.
- **`retrieval_text` beats `summary` on COURSE_W** at recall@1 (+2.5 pts over summary, +7.5 pts over body text). When `key_terms` are present, composing them into `retrieval_text` adds discriminative vocabulary that raw summary lacks.

## Decision: **KEEP ALL THREE, DOCUMENT THE PICKER**

Summary lift at recall@1 on both courses exceeds the +3-point threshold I set at the start, so neither field is dropped. But summary's regression at recall@5/10 on COURSE_D is large enough that it cannot be treated as a universal upgrade over body text — that would regress broad-aperture retrieval.

**What ships in v0.2.0 is the decision logic consumers need to pick the right variant**:

- **Use `retrieval_text` (when present) or `summary` (fallback) for top-1 / top-3 precision retrieval.** Query-answering, cite-the-best-chunk, training-pair grounding. Lift is +7–19% at recall@1.
- **Use `text` (chunk body) for top-10+ broad retrieval.** Synthesis, multi-concept aggregation, rerank candidates. Summary regresses meaningfully here on data-poor corpora.
- **Use `text` when `summary`/`retrieval_text` are empty/degenerate** — which is corpus-dependent. COURSE_D's summaries are weaker because COURSE_D's upstream metadata is thin (see Worker M1). Fix the enrichment pipeline (Worker M2) and the summary regression on COURSE_D likely shrinks.

This is documented in `docs/reference/reference-retrieval.md` so consumers don't re-derive it.

## Interaction with Worker M1 findings

Worker M1's §4.4a diagnostic shows **H2 dominates** — COURSE_D has 0% section-level JSON-LD metadata; COURSE_W has 47% missing. The weak summary signal on COURSE_D is partly a symptom: summary generation reads `key_terms`/LO refs/chunk text; with thin upstream metadata, summaries are generic and don't discriminate between chunks.

**Expected post-M2 trajectory**:
- COURSE_W `summary`/`retrieval_text` deltas should hold or improve (already net-positive).
- COURSE_D `summary`/`retrieval_text` should close the recall@5/10 gap as backfilled key_terms and content-type labels feed summary quality.
- If post-M2 COURSE_D still shows summary hurting recall@5+, that's a real signal to deprecate the field. For now the variant picker above is the right compromise.

## FOLLOWUPs

- **`FOLLOWUP-WORKER-O-1`** — re-run this benchmark after Worker M2 lands the H2 fallback fix. Record the post-fix COURSE_W + COURSE_D deltas in this doc as a v0.2.0-post-M2 addendum; update the variant-picker guidance if the cross-over point shifts.
- **`FOLLOWUP-WORKER-O-2`** — expose the variant picker as retriever flags on `libv2 retrieve` (`--variant text|summary|retrieval_text`) so consumers don't have to reach through the Python API. Worker J's retriever wiring is the natural place; small follow-up, not blocking.

## How to reproduce

```
# Inside the worker-o worktree (branch worker-o/summary-retrieval-benchmark)
python -c "
from pathlib import Path
from Trainforge.rag.retrieval_benchmark import run_benchmark
for slug, label in [
    ('course-w-accessibility', 'COURSE_W'),
    ('course-d-pedagogy', 'COURSE_D'),
]:
    chunks = Path(f'LibV2/courses/{slug}/corpus/chunks.jsonl')
    course = Path(f'LibV2/courses/{slug}/course.json')
    gold = Path(f'LibV2/courses/{slug}/retrieval/gold_queries.jsonl')
    r = run_benchmark(chunks, course, gold_queries_path=gold)
    print(label, r['variants'])
"
```

The slugs above (`course-w-accessibility`, `course-d-pedagogy`) are anonymized stand-ins — substitute your own local corpus slugs under `LibV2/courses/`. Corpora + gold queries live in the user's local working tree under gitignored `LibV2/courses/*/{corpus,retrieval}/` — not shipped.
