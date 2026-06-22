# DART/Semantic — Completion Punchlist

**Created:** 2026-05-31
**Branch:** `snapshot/runtime-2026-05-28` (substantial uncommitted work)
**Purpose:** the ordered, dispatchable list of what stands between the current
state and "full completion" of the v2 13-stage pipeline. Audit basis:
`architecture.md`, `README.md`, `Plans/05`–`09`, the working tree, and a full
`pytest` run (548 pass / 5 skip / 1 fail @ 2026-05-31).

**Progress @ 2026-05-31 (post-batch):** R1–R7 DONE; R6 harness CODE done (GPU
run pending); **R9 was already wired** in committed `199c52f` (audit was wrong)
— now locked by 6 invariant tests. Full suite **584 pass / 5 skip / 0 fail**.
Remaining: **R8** theta retrain (GPU, lynchpin) → then the GPU *runs* of **R6**
(figure eval) and **R10** (e2e eval). All code is written; only GPU runs remain,
to be launched by the user. Per-task notes below.

This is a status snapshot, not new design. Design lives in `architecture.md`
and `Plans/03`/`04`; per-thread sequencing lives in `Plans/05`–`09`.

---

## State of the world (post-batch, 2026-05-31)

> The 5-line pre-session snapshot is preserved at the bottom (§"Original audit
> snapshot") for the record. Current reality:

1. **v2 is code-complete AND test-complete end-to-end.** Full suite **584 pass /
   5 skip / 0 fail**. All 4 Qwen GGUFs registered; `structure_v2/final` (already
   the live runtime model, `is_image_block` head firing) + `table_specialist/final_v5`
   (promoted, `LAYOUT_FEATURE_DIM` 15→17 load bug fixed) both live and gated.
2. **Plan 09 image alt-text is built, tested, and wired:** `image_extract.py` +
   `figure_captioner.py` (SmolVLM2-256M) have 17 unit tests; `extended_description`
   now emits to HTML via `aria-describedby`; the eval harness exists (R6) — only
   its GPU run is pending.
3. **The Stage-13 offline-Qwen lane is wired** (was never actually degraded —
   audit correction in R9) and now locked by 6 invariant tests.
4. **The ONE remaining model gap** is theta `semantic_preservation` — still the
   `stub_v1` 0.7 placeholder (mode-collapsed; needs a *retrain*, R8). Until it
   lands, every `theta_score` is partly synthetic and the `<0.70` offline-retry
   trigger can't fire in production.
5. **End-to-end eval evidence is still `mock`/`stub`-flagged** — the real-runtime
   corpus eval (R10) is the README "definition of done" and depends on R8.

---

## Required for completion

Ordered. "Dep" = hard dependency; otherwise parallelizable. Size S/M/L.

### R1 — Regenerate the gap-fill dataset (fix the failing test) ✅ DONE
> Regenerated (seed=42, 6 pair dirs). Total 11,292 → **31,212 rows**; suite 549
> pass. ⚠️ `citation_unresolved` ballooned 5,708 → **25,628 rows** (older
> extraction path in the stale file). This is exactly the
> `feedback_balance_dominant_category` footgun — **cap citation_unresolved
> before any gap_fill v2 retrain (N3)** or it will regress the minor kinds again.

- **Files:** `data/build_gap_fill_qwen_data.py` (run), `data/qwen_gap_fill_dataset/`
  (output), `tests/test_qwen_gap_fill_dataset_contract.py` (verify).
- **What:** the shipped dataset has only `variant_idx=2` for `(PMC1079975,
  citation_unresolved)`; regenerating with the current builder produces a
  contiguous `0..5` (verified by re-running `build_citation_rows_for_paper` on
  that pair). Root cause: stale data, NOT a test bug or split-logic bug. Back up
  the current dir to a `.bak` first (matches the existing `.bak` convention).
- **Acceptance:** `pytest tests/test_qwen_gap_fill_dataset_contract.py` is green;
  full suite is 549 pass / 5 skip / 0 fail.
- **Dep:** none. **Parallel:** yes. **Size:** S (CPU regen, minutes).

### R2 — Promote `structure_v2` to the runtime (lights up figure detection) ✅ DONE (no-op)
> `models/council/structure/final/` IS already the v2 model (table_region_pos_f1
> 0.962, is_image_block pos-F1 0.738); the old one is backed up at `final_pre_v2/`.
> `DEFAULT_ADAPTER_DIR` already pointed there — the figure path was live. No edit.

- **Files:** `dart_semantic/council/structure.py:46` (`DEFAULT_ADAPTER_DIR`) or
  the council registry; validate against `models/council/structure_v2/final/summary.json`.
- **What:** `structure_v2/final` is trained, passes Plan 07 §5 gates
  (test_table_region_pos_f1 **0.962** ≥ 0.924 baseline; test_role_macro_f1
  **0.854**; adds `is_image_block` pos-F1 0.738), and the runtime loader already
  supports the 5-head shape. Re-point the default/registry at it. This is the
  step that makes `kind="figure"` actually fire end-to-end (the whole Plan 09
  path is dead without it).
- **Acceptance:** council runtime loads `structure_v2`; existing council tests
  stay green; a cascade smoke on a figure-bearing PDF emits ≥1 `figure` region.
- **Dep:** none. **Parallel:** yes. **Size:** S.

### R3 — Promote `table_specialist/final_v5` to the runtime ✅ DONE
> Repointed `final` → `final_v5` (test_macro_f1 0.907, 8× more training data).
> Caught a latent bug: runtime `LAYOUT_FEATURE_DIM` was 15 but v5 (and the data
> builder) use 17 — a blind swap would have raised on load. Fixed 15 → 17.

- **Files:** `dart_semantic/council/table_specialist.py:35` (`DEFAULT_ADAPTER_DIR`).
- **What:** `final_v5` is trained on the PMC-inclusive cell-role set (Plan 06
  §0.2 plan-B), test_macro_f1 **0.907**; runtime still points at `final`.
  Re-point after a quick metric sanity check.
- **Acceptance:** runtime loads `final_v5`; table cell-role tests stay green.
- **Dep:** none. **Parallel:** yes. **Size:** S.

### R4 — Add tests for `image_extract.py` and `figure_captioner.py` ✅ DONE
> `tests/test_image_extract.py` (8 tests, real in-mem 1-page PDF, no model) +
> `tests/test_figure_captioner.py` (9 tests, SmolVLM2 loader stubbed to raise so
> no model can load). Covers no-op, payload attach, and the typed-raise paths.

- **Files:** new `tests/test_image_extract.py`, `tests/test_figure_captioner.py`.
- **What:** both new modules have zero test coverage. Minimal coverage:
  - `image_extract`: no-figure-regions is a no-op (no PDF opened); a figure
    region with a valid bbox gets `payload["image_png_bytes"]` (non-empty PNG)
    + `image_render_dpi`; a degenerate/out-of-page bbox raises `FigureRenderError`;
    non-figure regions pass through unchanged. Use a tiny synthetic 1-page PDF
    fixture (Chromium render or a checked-in fixture).
  - `figure_captioner`: no-figure-regions is a no-op (no model load — assert the
    `lru_cache` loader is not called, e.g. monkeypatch); a figure region missing
    `image_png_bytes` raises `FigureCaptionError`; with the SmolVLM2 call
    monkeypatched, `alt_text`/`extended_description` land on the payload and
    `run_extended=False` skips the extended call. Do NOT load the real model in CI.
- **Acceptance:** both files pass; no GPU/model download in the test run.
- **Dep:** none (R2 helps produce a realistic fixture but is not required).
  **Parallel:** yes. **Size:** M.

### R5 — Emit `extended_description` as `aria-describedby` (Plan 09 §4) ✅ DONE
> `fallback_figure` now emits `<img aria-describedby="dart-figdesc-{fb}">` + a
> sibling `<p id=...>` inside the `<figure>`. Mirrors the live form-field
> `help_text` idiom (NOT the unwired `enrich.summarize_table` v1 helper — noted
> for the doc). Absent ext desc → byte-identical to prior output. Axe gate passes.
> Also fixed the stale "Florence-2" comment at `cascade.py:286`.

- **Files:** `dart_semantic/assembler/fallbacks.py:237` (`fallback_figure`);
  mirror the existing `summarize_table` aria-describedby pattern in `enrich.py`.
- **What:** the captioner attaches `extended_description` to the figure payload,
  but `fallback_figure` emits only the `alt` attribute. Wire the extended text to
  an `aria-describedby` target (visually-hidden block or `<details>`), matching
  Plan 09 §4 and the table-summary pattern. Must pass the per-region axe gate.
- **Acceptance:** a figure with an `extended_description` emits a valid
  `aria-describedby` reference; the fragment passes axe wcag22aa (0 serious/critical).
- **Dep:** R4 (test the new emission). **Parallel with R1–R3.** **Size:** S.

### R6 — Figure-captioner eval harness (Plan 09 §5 / §8 step 4) 🟡 CODE DONE — GPU run pending
> `scripts/eval_figure_captioner.py` + `tests/test_eval_figure_captioner.py` (16
> tests, CPU-validated with a stubbed captioner). All four §5 gates implemented
> (axe pass/regression, decorative→alt="", caption-first, no-hallucination numeric
> check), stratified by source. Model injected behind a `caption_fn` seam.
> **User-launched GPU run:** `.venv/bin/python -m scripts.eval_figure_captioner --tag v1`
> (`--self-test` for the CPU stub path). Dataset `data/figure_alt_dataset/` carries
> no decorative/subtype fields → decorative is a rule, subtype falls back to source.

- **Files:** new `scripts/eval_figure_captioner.py`, modeled on
  `scripts/eval_qwen_table_adapter.py`; reads `data/figure_alt_dataset/`.
- **What:** the only Plan 09 acceptance gate with zero coverage. Build the
  harness: axe pass/regression on figure fragments, decorative→`alt=""`,
  caption-first precision, and the no-hallucination spot-check (generated
  description contains no numeric claim absent from caption), stratified by
  figure subtype. Write `data/eval_reports/figure_captioner_*.json`.
- **Acceptance:** harness runs on a held-out figure sample and emits a report
  with the §5 gate metrics; documents the SmolVLM2 verdict (ship / iterate).
- **Dep:** R2 (need real figure regions end-to-end), R5 (eval the full emission).
  GPU-serial (loads SmolVLM2) — do not run while another GPU job holds VRAM.
  **Size:** L.

### R7 — Regenerate `data/figure_catalog/coverage_report.json` ✅ DONE
> Builder has no no-fetch recount mode, so added `scripts/refresh_figure_coverage.py`
> (stdlib-only, zero network). `image_local` 0 → **20,847**; `needs_image_fetch`
> 20,922 → **75** (arxiv 3 / pmc 72 / openstax 0). Manifest-ok (20,560) matches
> on-disk exactly; surplus is 297 duplicate catalog rows sharing a fetched image.

- **Files:** `data/build_figure_catalog.py` (re-run the coverage pass) or a small
  recount; output `data/figure_catalog/coverage_report.json`.
- **What:** the report says `image_local: 0` / `needs_image_fetch: 20922` because
  it predates the image fetch. `data/figure_images/` now holds 2.0 GB across
  arxiv/pmc/openstax (20,961-line `_fetch_manifest.jsonl`). Recompute so
  `image_local` reflects reality (avoids a future worker re-fetching 2 GB).
- **Acceptance:** `image_local` > 0 and matches the on-disk/manifest count.
- **Dep:** none. **Parallel:** yes. **Size:** S.

### R8 — Train the theta `semantic_preservation` cross-encoder (de-stub Stage 12)
- **Files:** `scripts/train_semantic_preservation.py`,
  `data/build_semantic_preservation_dataset.py`,
  `dart_semantic/theta/semantic_preservation.py`,
  `dart_semantic/theta/evaluator.py`.
- **What:** Stage 12 dim 1 is a hard-coded `0.7` `stub_v1` (architecture §6.6;
  memory: theta v1 is mode-collapsed, val MAE 0.232 / spearman 0.21 — needs a
  *retrain*, not a rewire). Train the DeBERTa-v3-small cross-encoder, validate it
  clears the noise floor, and let the loader sentinel pick it up (it correctly
  falls back to `stub_v1` until then). This is the single biggest "not
  model-complete" gap; until it lands, every `theta_score` is partly synthetic.
- **Acceptance:** trained model loads without the `DART_ALLOW_THETA_STUB` opt-in;
  val spearman materially > 0.21 and MAE materially < 0.232; theta reports stop
  carrying the `stub_v1` method tag.
- **Dep:** none. GPU-serial. **Parallel** with all code tasks. **Size:** L.

### R9 — Wire the Stage-13 offline-Qwen lane (remove the two `*_v1` degradations) ✅ DONE (already wired; now tested)
> AUDIT CORRECTION: the offline lane was ALREADY wired in committed `199c52f` —
> `theta/offline_retry.py:maybe_offline_retry` does the re-run + higher-theta
> selection (keeps offline iff `offline_theta ≥ fast_theta + 0.05` and clears WCAG),
> `cascade.py` calls it before `decide_exit`, and neither `*_v1` flag is emitted
> anywhere. The two degradations described above never existed in this branch.
> Added 6 invariant tests to `tests/test_theta.py` locking the §7.5 rows, the
> "no `*_v1` flag in any exit" invariant, and the `StageThirteenStubRequired`
> no-silent-fallback guard. **Production caveat:** the `<0.70` retry trigger won't
> fire until R8 de-stubs theta (stub_v1 is constant ~0.7) — correct and expected.

- **Files:** `dart_semantic/theta/exits.py` (the `StageThirteenStubRequired`
  path + `THETA_LOW_NO_RETRY` / `OFFLINE_LANE_UNAVAILABLE_V1` flags in
  `theta/types.py`), `dart_semantic/cascade.py` (the `_run_inner(lane)`
  orchestrator already exists for offline retry — connect the trigger).
- **What:** architecture §7.5 — two rows are degraded because the offline lane
  isn't fully wired: `pass|fast|<0.70 → retry_offline` ships-with-flag
  (`theta_low_no_retry`) and `fail|fast → offline_qwen_lane` falls straight to
  `non_certified_stamp` (`offline_lane_unavailable_v1`). The cascade already has
  the offline-lane plumbing (`_run_inner("offline")`); finish the trigger +
  re-enable the canonical actions. Honor the no-silent-fallback invariant
  (the typed `StageThirteenStubRequired` is intentional until this lands).
- **Acceptance:** a doc with fast-lane theta < 0.70 actually re-runs offline and
  ships the higher-theta output; a fast-lane gate failure routes through the
  offline lane before any non-certified stamp; the two `*_v1` flags no longer fire.
- **Dep:** R8 is recommended first (offline retry is theta-driven; with the stub,
  theta is constant 0.7 and the retry trigger is meaningless). **Size:** L.

### R10 — Real-runtime end-to-end corpus eval (the README "definition of done")
- **Files:** `scripts/eval_full_cascade.py`; output `data/eval_reports/`.
- **What:** `README.md` states v2 is "non-production until a real-runtime corpus
  eval exists" — all current end-to-end evidence is `mock`/`stub`-flagged
  (`full_cascade_real_smoke*` are n=1–15 smokes). Run a real-runtime
  (`make_runtime("real")`, all 4 GGUFs) eval over a held-out corpus, with R8's
  theta de-stubbed, and record axe pass-rate / stage-7 + stage-10 violation
  rates / theta distribution. This is the gate that flips the README's
  "code-complete but not model-complete" caveat.
- **Acceptance:** a non-mock, non-stub `full_cascade_*` report over a multi-PDF
  corpus with stage-10 pass-rate and a real theta distribution.
- **Dep:** R2, R3, R8 (real models + real theta), ideally R9. GPU-serial.
  **Size:** L.

---

## Nice-to-have (not required for v1 completion)

### N1 — Structure `is_heading` calibration
- **Files:** `dart_semantic/structure_graph.py:588`, `scripts/calibrate_structure_heads.py`,
  `scripts/measure_stage5_heading_rate.py`.
- **What:** architecture §3.3 TODO — `is_heading` over-fires (20–38% of FBs at
  conf ≥ 0.5–0.9). Threshold raised 0.5→0.7 as partial mitigation; a temperature-
  scaling / recalibration pass is the real fix. Deferred to a future Phase-3c
  retrain. **Size:** M.

### N2 — DePlot chart-tier captioning (Plan 09 §3)
- **What:** SmolVLM2 emits generic descriptions; the chart-tier (DePlot →
  deterministic data-table summary) is deferred until a chart-vs-photo router
  exists and we want to compare against `data/figure_alt_dataset`. The §0.1
  strategic argument ("charts are the hard case") makes this the highest-value
  *quality* lever once the baseline ships. **Size:** L.

### N3 — gap_fill V2 remaining kinds (`copyright_block`, `legal_disclaimer`)
- **Files:** `data/build_gap_fill_qwen_data.py`, `dart_semantic/assembler/pass_9a.py`
  (detect), `dart_semantic/assembler/pass_9c.py` (`_splice_<kind>`),
  `dart_semantic/qwen_specialists/prompts.py`.
- **What:** Plan 08 — `citation_unresolved` is the lead (its rows exist in the
  dataset; see R1). `copyright_block` + `legal_disclaimer` are the companions
  (detect + extract + prompt + splice + retrain). Lower priority than the
  required list. **Size:** L (4-part change each + GPU retrain).

### N4 — Doc-level soft reranker / per-region top-2 fan-out (architecture §11 v2)
- **What:** explicitly deferred to v2 (combinatorial trap). Not part of v1
  completion. **Size:** L.

### N5 — Remove the superseded `enrich.py` figure stubs ✅ DONE (removed 2026-06-09)
- **Files:** `dart_semantic/enrich.py` `alt_text_for_image` /
  `describe_figure_extended` `NotImplementedError` stubs (was `:107,134`),
  `:57` TODO warning.
- **What:** the Stage-6b cascade path supersedes these v1-pipeline stubs. The
  `NotImplementedError` stubs were deleted 2026-06-09; figure alt-text now flows
  through the cascade (`figure_captioner.caption_figure_regions` →
  `assembler/fallbacks.fallback_figure`). Housekeeping only — the v1
  pipeline (`pipeline.py`) is the legacy path. **Size:** S.

---

## Remaining path to completion (GPU-only)

All code is written. R1–R5, R7, R9 are DONE; R6, R8, R10 are now also DONE.
**v1 completion is met.** Only the N-items (nice-to-haves) remain.

```
R8 (theta retrain) ──┬─► R10 (real e2e eval)   [R10 needs real theta]
                     │
R6 (figure eval) ────┘   [independent of R8 — can run first/standalone]
```

- **R8 ✅** — theta `semantic_preservation` retrain. v8 (full-FT
  DeBERTa-v3-small, cls, BCE) is live and loads without `DART_ALLOW_THETA_STUB`;
  it replaced the mode-collapsed v1.
- **R6 ✅** — figure-captioner eval ran: `figure_captioner_v1.json` (axe 1.0 /
  regression 0.0) and `figure_captioner_v2_guard.json` (no_hallucination 1.0,
  ship=true).
- **R10 ✅** — real-runtime corpus eval passed 2026-06-09
  (`data/eval_reports/full_cascade_real_v8_R10_postfix.json`: WCAG 3/3,
  `stage10_pass_rate` 1.0). Flips the README "not model-complete" caveat.

With these three done, "v1 completion" is met. The nice-to-haves (N1–N5) are
explicitly out of v1 scope.

---

## Original audit snapshot (pre-session, 2026-05-31, for the record)

1. v2 code-complete; 4 GGUFs registered; structure_v2 + table_v5 trained & gated.
2. Not model-complete: theta `stub_v1`, offline lane "unwired" (later found
   already wired), e2e eval mock/stub-flagged.
3. structure_v2 / table_v5 believed un-promoted (structure_v2 was in fact live).
4. Plan 09 built but untested/uneval'd; `extended_description` not emitted.
5. One failing test (`test_variant_idx_monotonic`) from a stale gap-fill dataset.
Suite at that point: 548 pass / 5 skip / 1 fail.
