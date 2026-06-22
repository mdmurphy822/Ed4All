# Plan 12 — Verifiable Accessible HTML Outputs: Long-Term Roadmap

**Created:** 2026-06-10
**Branch:** `snapshot/runtime-2026-05-28` (28 commits ahead of `main`)
**Basis:** code audit of `dart_semantic/gates/`, `dart_semantic/theta/`,
`architecture.md`, `docs/ontology.md`, `data/eval_reports/`, Plans/05–11,
and the R10 real-runtime eval (`full_cascade_real_v8_R10_postfix.json`).

---

## 0. The long-term goal, stated precisely

> **Every document DART ships carries machine-checkable evidence of its WCAG
> 2.2 AA conformance** — which gates ran, which passed, which were skipped and
> why, what the calibrated quality score was, and which success criteria are
> verified automatically vs. flagged for human review — over a corpus broad
> enough that the claim generalizes beyond arXiv papers.

This is the procurement-facing differentiator (compete on the accessibility
axis): not "our HTML passes axe" but "here is the per-document audit trail
proving it, and here is exactly what that proof does and does not cover."

**Where we are:** v1 completion was met 2026-06-09 (Plans/10 R1–R10 all done;
R10 real-runtime eval green: WCAG 3/3, stage10_pass_rate 1.0, theta v8 live,
offline lane firing). The three R10 follow-ups (mathml lxml soft-pass, table
15≠17 layout dim, report mislabel) were all **fixed** in commits `cc67da2`,
`69857b3`, `fffe8d9`, `6c3d50d` — none remain open. What remains is the gap
between *"the pipeline passed its eval"* and *"any third party can verify any
shipped document."*

---

## 1. Blocker inventory (what stands between here and the goal)

### Verified-current status corrections

Three things the docs misstate (all SHIPPED — do not re-scope them):

- **Theta `semantic_preservation` is live** (v8 full-FT DeBERTa-v3-small),
  not the `stub_v1` 0.7 — `architecture.md:586` is stale.
- **The Stage-13 offline-Qwen lane is wired** and fired twice in R10
  (`n_offline_retries: 2`) — `architecture.md:688-689` "v1 degradation" rows
  are stale.
- **Figure alt-text is shipped and verified** (SmolVLM2 captioner, numeric
  guard, no_hallucination 1.0, axe re-run passed).

### Open blockers, by tier

| # | Blocker | Evidence | Severity |
|---|---------|----------|----------|
| A1 | No per-document conformance audit artifact — gate-by-gate pass/**skip**/fail is never emitted per doc; skip counts exist only at corpus level (`cascade.py:_stage7_skip_distribution`) | `theta/types.py` ThetaReport has no gate log | **Core of the goal** |
| A2 | No per-SC coverage map — conformance statement (`ontology.md:186`) doesn't say which of the ~50 WCAG 2.2 AA criteria are auto-verified / partial / human-required / template-scope | `ontology.md` gap table is dev-facing only | **Core of the goal** |
| A3 | Theta is uncalibrated — uniform 1/8 weights placeholder (`theta/evaluator.py:224`), thresholds 0.70/0.80 never tuned on held-out (`architecture.md:517`), `hallucinated_structure_penalty` is a constant 0.85 when gaps resolved (`evaluator.py:532`) | R10 theta avg 0.721, min 0.688 — scores not yet meaningful as a quality claim | **High** — theta drives ship/retry decisions |
| B1 | Complex tables: no `headers`/`id` (H43) association — multi-header tables get `scope=` only or fail the gate; nothing generates `id`/`headers` pairs | `gates/hard_region.py:597-646`, `ontology.md:244` | High for STEM docs (SC 1.3.1) |
| B2 | Language of parts not emitted — no `lang` on runs/spans; multilingual docs fail SC 3.1.2 | `ontology.md:150,247` | High for multilingual; low for arXiv corpus |
| B3 | Reading order never re-validated after region drops/transforms — a dropped context region can silently break SC 1.3.2 | Stage 2 skeleton → Stage 9 emission, no gate between | Medium (flag-only fix is cheap) |
| B4 | Heading over-detection fix (Plans/11: T=1.6553, threshold 0.8) is code-complete but `data/extract_cache/` (1.6 GB) is keyed on PDF mtime, not code version — stale cache can serve pre-fix extractions | Plans/11 caveat | Medium — verify, then close |
| C1 | Eval corpus is n=3 arXiv PDFs — "WCAG 3/3" does not support a generalizable conformance claim | R10 report `per_pdf n=3` | **High** — evidence breadth |
| C2 | Zero non-arXiv/non-Wikipedia coverage in the e2e eval — no forms, scans, textbooks, gov docs | data-expansion plan unexecuted at eval level | High (long-term) |
| D1 | Math v2 retrain stalled at `checkpoint-1000` (no `final/`); last log 2026-06-09 22:19; GPU currently free (1.2/8 GB) | `models/qwen_specialists/math/v2/`, `data/logs/train_qwen_math_v2.log` | Medium — math regions ship via v1 adapter with 32.5% length-censoring meanwhile |
| D2 | SigLIP figure router: zero-shot taxonomy agreement 75.9% < 80% gate (map 0.40, equation_or_table_image 0.14); P2 visual spot-check (300 rows) not run; decorative class has zero rows (needs re-harvest) | `data/figure_embeddings/agreement_report.json` | Medium — gates the DePlot chart tier (alt-text *quality* lever) |
| D3 | Gap-fill v2: `citation_unresolved` at 25,628/31,212 rows (82%) must be capped before any retrain; copyright/legal kinds unimplemented | Plans/08, Plans/10 N3 + R1 warning | Low (explicit nice-to-have) |
| E1 | 28 commits (186 files, +49.5K lines) unmerged to `main` — the entire 2026-06 batch lives on one snapshot branch | `git log main..HEAD` | Hygiene, but real risk |
| E2 | `architecture.md` + `ontology.md` stale on theta v8, offline lane, figure captioner — canonical docs misstate shipped reality (this audit was misled by them) | `architecture.md:586,688-689` | Hygiene — cheap, do early |

---

## 2. The plan

Ordered by (value to the goal) × (unblocking effect). Tiers A and C *are* the
long-term goal; B closes real SC gaps; D are quality levers; E is hygiene that
protects everything else.

### Phase V0 — Protect and re-baseline (days)

1. **E1 — Merge `snapshot/runtime-2026-05-28` → `main`.** ✅ DONE 2026-06-10 (fast-forward, then all Plan-12 work committed on main). The branch is
   forward-building with no conflicts. Everything below builds on it.
2. **E2 — Doc reconciliation pass.** ✅ DONE 2026-06-10 (715d925). Update `architecture.md` §6.6 (theta v8
   live), §7.5 (offline lane wired — replace the two `*_v1` degradation rows
   with the canonical actions), and the figure-captioner status. Update
   `ontology.md` gap-table rows that R5/R6 closed. Acceptance: a fresh reader
   of `architecture.md` reaches the same world-state as `git log`.
3. **B4 — Close the heading-fix loop.** ✅ DONE 2026-06-10 (7b3081b: EXTRACT_CACHE_VERSION salt; cache had already been cleared for R10-postfix). Re-run the cascade smoke/eval with
   `--invalidate-extract-cache` (or clear `data/extract_cache/`), confirm
   corpus heading counts land near reference (1807.02622-style 95% reduction
   holds corpus-wide). Consider keying the cache on an extractor code-version
   salt so this class of staleness can't recur. Acceptance: post-cache-clear
   eval report with heading-rate column; cache key includes code version.
4. **D1 — Finish the math v2 train.** ⏸ USER-GATED GPU: resume command staged (checkpoint-1000; ~61-80h); first launch aborted correctly by the training-lock guard (GPU held by another project). GPU is free. Resume/relaunch
   `train_reasoner.py`-lane math v2 to `final/`, then eval at the 960/4096
   caps with the tokenizer-true length check (~2.9 chars/token for dense
   content). Acceptance: `models/qwen_specialists/math/v2/final/` + a
   `qwen_math_adapter_v2` eval report showing truncation materially below
   v1's, GGUF rebuilt and registered. (GPU-serial — nothing else on CUDA.)

### Phase V1 — The verifiability artifact (the heart of the goal; ~1–2 weeks)

5. **A1 — Per-document `conformance_audit.json` sidecar.** ✅ DONE 2026-06-10 (24e3342). Emit, for every
   document at Stage 13 (all exit modes, including non-certified):
   - Stage-7 per-region gate log: each check {name, passed, skipped,
     skip_reason, message}; aggregate skip counts per check (the corpus-level
     `_stage7_skip_distribution` logic, moved to per-doc).
   - Stage-10 doc-gate log: same shape, plus the axe ruleset/version and
     violation list (empty on pass).
   - Theta block: per-dimension score + **method tag** (model version vs.
     heuristic vs. stub) + flags + thresholds in force (config version).
   - Assembly provenance: gaps flagged/resolved/fallback, regions dropped
     (count + kinds), `data-dart-*` marker census, offline-retry history.
   - Pipeline provenance: model/adapter versions (council, Qwen GGUFs, theta,
     captioner), code version (git SHA), runtime mode.
   Design notes: this is mostly *surfacing* state the cascade already holds —
   no new models. Honor no-silent-fallbacks: a skipped check must say so; an
   audit writer failure must raise, not ship a doc without its audit.
   Acceptance: every e2e eval doc emits a sidecar; a test asserts schema
   completeness; a doc with skipped text-preserve regions shows them.
6. **A2 — WCAG 2.2 AA coverage map, emitted not just documented.** ✅ DONE 2026-06-10 (27b54b7; H43 row updated in e16fbee).
   - One canonical machine-readable table (`dart_semantic/gates/wcag_coverage.py`
     or data file) mapping every Level A/AA SC → {automated | partial |
     human-review | template-scope | not-applicable} with the enforcing
     check's name.
   - Embed the per-SC verdicts into the A1 sidecar; render a
     procurement-facing `docs/wcag_coverage.md` from the same source.
   - Be honest where verification is syntax-only (alt presence vs. alt
     quality, link non-emptiness vs. link purpose).
   Acceptance: every SC appears exactly once; the doc is generated, not
   hand-maintained; sidecar carries per-SC status.
7. **B3 — `reading_order_at_risk` flag (rides on A1).** ✅ DONE 2026-06-10 (704a1f0). When Stage 9 drops or
   falls back a region adjacent to a complex region (table/figure/math),
   record it in the sidecar and theta flags. Cheap instrumentation, no model.
   Acceptance: synthetic doc with a forced region drop carries the flag.

### Phase V2 — Calibrate the number we ship (1–2 weeks, partly GPU)

8. **A3 — Theta calibration.** ✅ DONE 2026-06-10/11 — part 1:
   `hallucinated_structure_penalty` de-stubbed via token anchoring
   (commit 081937d). Part 2: `scripts/calibrate_theta.py` fitted
   weights/taus/floors on 40 docs × 520 perturbation variants (fitted
   AUC 0.849 vs uniform 0.834); locked in `theta/config.yaml`
   (`theta-config-2.0`) with provenance; evaluator/exits/audit load
   from config (`ThetaConfigError` on missing — no silent default).
   Caveat recorded: taus clamped (synthetic clean sits above the
   real-pipeline distribution) — re-calibrate on the C1 corpus.
   - Build a held-out perturbation set (good HTML + controlled degradations:
     shuffled sections, dropped headings, broken refs, hallucinated structure)
     with target quality ranks.
   - Fit composite weights (replace uniform 1/8 with the `theta/config.yaml`
     schedule or a fitted one) and tune `TAU_THETA_RETRY` / `TAU_THETA_CONFIDENCE`
     / per-dimension floors against it; lock + version in config (the
     architecture's "lock at release time" discipline).
   - De-stub `hallucinated_structure_penalty`: gap-fill provenance now exists
     end-to-end, so token-level substring/paraphrase anchoring of resolved-gap
     text against source is implementable. Replace the constant 0.85.
   - Record calibration dataset + date in the config so the sidecar can cite it.
   Acceptance: weights/thresholds carry a calibration provenance block; the
   retry trigger demonstrably separates degraded from clean docs on held-out;
   no constant-valued dimension remains except declared template-scope ones.

### Phase V3 — Make the evidence generalize (2–4 weeks, interleaved)

9. **C1 — Scale the real-runtime eval corpus.** 🟡 PREP DONE 2026-06-10/11 (3daedc0 + c62c892: 56-entry manifest_v2, --manifest plumbing, runbook in data/eval_corpus/README.md). ⏸ The GPU run itself (~10-25h) is user-gated. n=3 → stratified 30–50 PDFs:
   arXiv across subject areas (the 7,480-PDF local repo), Wikipedia-derived,
   synthetic forms, OpenStax chapters. Reuse `eval_full_cascade.py`; run
   GPU-serial. Track stage-7/stage-10 pass rates, theta distribution,
   offline-retry rate, per-SC verdicts from A1. Acceptance: a single report
   the README can cite as the standing definition-of-done, refreshed on
   model promotions.
10. **C2 — Non-arXiv domains into the eval (data-expansion plan, eval-first).** ✅ DONE 2026-06-11 (c62c892: 18 US-gov-PD PDFs across forms/opinions/govinfo/scans strata).
    Add govinfo.gov / IRS-form / scanned-OCR documents to the corpus *as eval
    inputs first* (no training required to measure). Expect new failure modes
    (forms → form_label/fieldset paths; scans → Tesseract-quality floors);
    file follow-up plans per domain rather than fixing inline.
11. **B1 — Complex-table `headers`/`id` (H43).** ✅ DONE 2026-06-10 (e16fbee). Detect multi-header-row /
    dual-axis tables at assembly; generate `id` on `<th>` + `headers` on
    `<td>`; extend `_check_table_structure` to *require* H43 association for
    complex tables (gate currently can't even express the requirement).
    STEM corpora make this the highest-value B item. Acceptance: complex-table
    fixture round-trips with valid associations; axe + extended gate green;
    eval corpus table docs unaffected at the simple-table tier.
12. **B2 — Language of parts (SC 3.1.2).** ✅ SCOPED-DONE 2026-06-10: coverage map records SC 3.1.2 as human-review; implementation triggers when multilingual content enters the corpus (none in manifest_v2). Per-span language detection at
    enrich time (langdetect/lingua on CPU, threshold high to avoid false
    spans), `lang` field through the IR, `<span lang>` emission, sidecar
    notes spans tagged. Schedule when multilingual content enters the corpus
    (C2); until then mark the SC "partial — document-level lang only" in A2's
    map rather than pretending coverage.

### Phase V4 — Quality levers (background / opportunistic GPU)

13. **D2 — SigLIP router to P2.** ✅ DONE 2026-06-11: visual spot-check (eval-only, figure_router_spotcheck_v1.json), binary chart-vs-rest gate shipped (37dd0bb, offline P 1.000/R 0.769), decorative re-harvest + arXiv OAI-PMH license fix (125cb30). Unblocks N2 (DePlot chart-tier
    descriptions) — the alt-text *quality* lever for the hardest figure class.
    **5-class plumbing DONE 2026-06-11 (all CPU):** embeddings refreshed
    (license-stale rows pruned via `--prune-stale`; captionless splits
    embedded via `--embed-missing`), `mine_figure_labels.py` ran (150
    class-balanced frozen-eval + 500 quota/uncertainty train candidates,
    doc-disjoint), `train_figure_router.py` + `eval_figure_router.py`
    written (measured logreg recipe + CalibratedClassifierCV, gates acc≥0.85
    / mF1≥0.80 on the runtime view), `classify_subtype` wired fail-closed
    (typed `FigureRouterHeadMissing`; abstain→other at calibrated p<0.55).
    See `data/figure_labels/README.md`. ⚠ The 300-row /tmp truth set was
    LOST to the 2026-06-09 host reboot; 60 labels salvaged from the
    spot-check report into `seed_labels.jsonl` (all label artifacts now live
    in git). P3 consistency-gate kill recorded:
    `eval/results/figure_consistency_calibration.json`. OPEN USER DECISION:
    labeling budget, now ~650 figures (vision-worker build-time labeling,
    policy-approved 2026-06-09) → then train + eval + the head goes live.
14. **D3 — Gap-fill v2.** 🟡 Cap DONE (d2b609f) + copyright/legal 4-part plumbing DONE 2026-06-11 (1158e80: detect/prompt/splice/data, dataset 15,600 rows across 5 kinds; legal_disclaimer starved at 176 rows — noted). ⏸ The retrain itself is user-gated GPU. Cap `citation_unresolved` (~3–4K rows, parity with
    other kinds) per the balance discipline **before** any retrain; then the
    4-part plumbing (detect/extract/prompt/splice) for `copyright_block` +
    `legal_disclaimer`; retrain + per-kind eval. Becomes more valuable once
    C2 brings gov/legal docs in.
15. **N1 leftover — only if V0.3 measurement says so.** Plans/11 shipped
    temperature scaling; if corpus-wide heading rates still over-fire on the
    bigger V3 corpus, schedule the Structure retrain then, not now.

---

## 3. Dependencies

```
E1 merge ─► everything
E2 docs ──► (independent, do first)
B4 cache ─► C1 corpus eval        D1 math v2 ─► C1 (math regions at v2)
A1 sidecar ─► A2 map ─► C1 report cites per-SC verdicts
A1 ─► B3 flag
A3 theta calibration ─► C1 (theta distribution is meaningful)
C1 ─► C2 domains ─► (B2 lang, D3 gap-fill priority calls)
D2 router ─► N2 DePlot (out of scope here)
GPU-serial chain (one at a time): D1 math v2 → A3 calibration runs → C1/C2 evals → D2/D3
```

### Definition of done for the long-term goal

A stratified ≥30-document real-runtime eval, spanning at least three source
domains, where **every** document — including failures — emits a
`conformance_audit.json` whose per-SC coverage map, gate logs, and
calibrated theta score a third party could use to independently confirm the
WCAG 2.2 AA claim, with no stubbed dimension and no silent skip.

---

## 4. Explicitly out of scope (so it stays out)

- Doc-level soft reranker / top-2 fan-out (architecture v2 deferral, N4).
- Human escalation paths (locked: four exit actions, no human).
- BERT-TableDetector (retired 2026-05-05; do not scope back in).
- Alt-text *semantic adequacy* beyond the numeric-guard + no-hallucination
  checks — declared "partial / human-review" in the A2 map, not solved.
- External LLMs anywhere in the runtime (locked).
