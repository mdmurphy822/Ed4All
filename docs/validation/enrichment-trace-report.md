# §4.4a Enrichment-coverage investigation — Worker M1 diagnostic

**Status:** diagnostic-only. No behavior change in this PR. The fix lives in a follow-up Worker M2 PR once the dominant hypothesis is agreed.

**Instrumentation:** `chunk["_metadata_trace"]` field records the source path for each enrichment field. `quality/metadata_trace_report.json` groups chunks by trace value. Parser-side flag `_jsonld_parse_failed` distinguishes H2 (tag absent / sections empty) from H5 (tag present but parse failed).

## Hypothesis reference (from VERSIONING.md §4.4a)

| Code | Hypothesis |
|---|---|
| H1 | heading-normalisation drift between Courseforge emit + Trainforge consume |
| H2 | JSON-LD sections genuinely absent on the page (tag missing, or `sections: []`) |
| H3 | `content_type_label` short-circuit at `_extract_section_metadata` gate — JSON-LD supplies contentType but not keyTerms, so data-cf-* fallback never runs |
| H4 | no-sections code path — chunk heading equals page title; JSON-LD sections keyed by section heading → structurally cannot match |
| H5 | JSON-LD `<script>` tag present but JSON parse failed; chunker treats as absent |

## Findings

### WCAG_201 (131 chunks, regenerated end-to-end through dev-v0.2.0 pipeline)

| Field | Populated | Trace breakdown |
|---|---|---|
| `content_type_label` | **69 / 131 (52.7%)** | 68 `jsonld_section_match` (51.9%) • **60 `none_no_jsonld_sections` [H2] (45.8%)** • 2 `none_heading_mismatch` [H1] (1.5%) • 1 `data_cf_fallback` (0.8%) |
| `key_terms` | 69 / 131 (52.7%) | same breakdown as `content_type_label` |
| `bloom_level` | 131 / 131 (100.0%) | 131 `section_jsonld` — no fallback needed |
| `misconceptions` | 71 / 131 (54.2%) | 71 `jsonld_page_misconceptions` (54.2%) • 60 `none` (45.8%) — page-level field; 60 chunks come from pages that genuinely had no misconceptions declared, so "none" here is accurate, not a drop |

### DIGPED_101 (86 chunks)

| Field | Populated | Trace breakdown |
|---|---|---|
| `content_type_label` | 0 / 86 (0.0%) | **86 `none_no_jsonld_sections` [H2] (100%)** |
| `key_terms` | 0 / 86 (0.0%) | 86 `none_no_jsonld_sections` [H2] (100%) |
| `bloom_level` | 86 / 86 (100.0%) | 75 `verbs` (87.2%) • 10 `section_jsonld` (11.6%) • 1 `default` (1.2%) — fallback chain carried the load |
| `misconceptions` | 0 / 86 (0.0%) | 86 `none` — pages lack the page-level `misconceptions` JSON-LD block entirely |

## Verdict

**H2 dominates.** On WCAG_201, 46% of chunks are missing JSON-LD `sections` metadata at the page level — the pages Courseforge emitted with JSON-LD `learningObjectives` but no `sections` array. On DIGPED_101, 100% of chunks hit H2 (the DIGPED course was last generated before Worker H's updated pipeline; its JSON-LD has no `sections` array on any page).

H1 contributes 1.5% on WCAG_201 (two chunks whose heading normalisation drifted between emit and consume). Negligible compared to H2.

**H3, H4, H5 did not fire on either corpus.** The short-circuit (H3) exists as a structural possibility — any page where JSON-LD supplies `contentType` without `keyTerms` would trip it — but none of the Courseforge-emitted pages in the two corpora hit that shape. The no-sections path (H4) didn't fire because chunks are not using the page-title fallback heading when JSON-LD sections exist. No silent parser failures (H5) were detected by the instrumented tag-present / metadata-present discriminator.

### Example: a page that contributes to H2

File: `week_01/week_01_application.html` (WCAG_201).
- JSON-LD block present; the `sections` array is literally empty (`"sections": []`).
- `_jsonld_parse_failed = False` (tag parsed fine; the key just contains no entries).
- Every chunk whose parent is this page falls into `none_no_jsonld_sections`.

The Courseforge generator emits `sections` when the course-data JSON declares a rich page structure (like the week's content pages with multiple `<h2>` subsections). Pages that are simpler in structure — week overviews, activities, self-checks, discussions, summaries — often emit an empty `sections` array. These are the 60 missing WCAG chunks.

## Recommended M2 fix target

**H2 — wire the fallback helpers.** VERSIONING.md §4.4a says verbatim: *"If the root cause is H2, fallbacks are appropriate."* The three helpers already exist at module scope in `Trainforge/process_course.py` and are unit-tested but deliberately unwired pending this investigation:

- `derive_bloom_from_verbs(text)` — already wired into `_create_chunk` for bloom fallback (Worker B). Not the gap.
- `extract_key_terms_from_html(html)` — not wired. Would populate `key_terms` from bold / `<dfn>` / definition lists when JSON-LD doesn't supply them.
- `extract_misconceptions_from_text(text)` — not wired. Would detect "Common mistake:" / "A common misconception:" prose patterns when JSON-LD page-level misconceptions are absent.

Worker M2 should:

1. **Wire `extract_key_terms_from_html` as a fallback inside `_extract_section_metadata`** — when JSON-LD sections don't match AND data-cf-* fallback yields nothing, call the HTML extractor against the raw HTML for the relevant section. Falls back to whole-page key-term extraction for H4-adjacent cases where the chunk is a page-level chunk.
2. **Wire `extract_misconceptions_from_text` as a fallback in `_create_chunk`** — when `item.get("misconceptions")` is empty, scan the chunk text; emit discovered misconceptions on the chunk (page-level grouping OK).
3. **Add a small `content_type_label` heuristic** — when section metadata is missing, derive `content_type_label` from `chunk_type` + resource_type (e.g., `exercise` → `example`, `summary` → `summary`). Five rules, one-line each. This closes ~45% of the gap with no Courseforge-side change.
4. **Leave the existing short-circuit at line 1218** (`if not content_type_label:`) alone — H3 doesn't fire empirically, and changing the gate semantics risks regressions in the 52% that currently work.
5. **Leave `_metadata_trace` in place behind an opt-in flag `--trace-enrichment`** so the diagnostic stays available for future investigations without bloating shipped chunk schemas.

Expected post-M2 coverage (order-of-magnitude estimate):
- WCAG_201 `key_terms`: 52.7% → ~85%+ (extractive key terms catch bold/strong terms across the 60 H2 chunks)
- DIGPED_101 `key_terms`: 0% → ~70%+ (similar extraction + no Courseforge regen required)
- `misconceptions`: limited by the actual presence of misconception prose; modest lift expected.

### Courseforge-side follow-up (out of scope for M2)

The canonical fix is to make Courseforge emit `sections` metadata on every generated page, not just content pages. That requires a Courseforge-side PR — tracked as **`FOLLOWUP-WORKER-M1-1`**. The Trainforge-side fallbacks M2 will ship are belt-and-suspenders that remain useful even after Courseforge ships the canonical fix, because they cover IMSCC packages not generated by Courseforge.

## How to reproduce

```
# Inside the worker-m1 worktree (branch worker-m1/44a-diagnostic)
rm -rf Trainforge/output/wcag_201 Trainforge/output/digped_101
venv/bin/python -m Trainforge.process_course \
  --imscc /path/to/WCAG_201.imscc \
  --course-code WCAG_201 --division STEM --domain computer-science \
  --output Trainforge/output/wcag_201 \
  --objectives /path/to/WCAG_201_objectives.json

# Read the trace report
python3 -c "
import json
r = json.load(open('Trainforge/output/wcag_201/quality/metadata_trace_report.json'))
for fname, fdata in r['fields'].items():
    print(f'{fname}: {fdata[\"populated_pct\"]:.1%}')
    for row in fdata['by_trace']:
        print(f'  {row[\"count\"]:4} [{row[\"hypothesis\"]:4}] {row[\"trace\"]}')"
```

Raw `metadata_trace_report.json` files live under gitignored `Trainforge/output/` — not committed.

## Followups

- `FOLLOWUP-WORKER-M1-1` — Courseforge-side canonical fix: emit `sections` metadata on every generated page (not just content pages). Separate Courseforge PR.
- `FOLLOWUP-WORKER-M1-2` — `misconceptions` coverage is limited by the actual presence of misconception prose in source content; an upstream task for the content-generator agent to include misconceptions in every page is a Courseforge-side improvement.
