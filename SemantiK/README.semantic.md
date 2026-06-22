# Semantic — DART structure pipeline

Turns arbitrary PDFs into WCAG 2.2 AA-conformant accessible HTML for DART, on
the principle that **learned models are narrow candidate generators;
deterministic code orchestrates, gates, and assembles.** No external LLMs at
runtime. No human in the loop.

> **Target architecture lives in [`architecture.md`](architecture.md).** It
> covers the BERT council, the Qwen specialist adapters, the two-tier
> validation gate, and the three exit modes (ship / offline-Qwen lane /
> non-certified stamp). Read that first if you want the canonical design.
>
> **Two pipelines coexist in this repo:**
>
> - **v1** — the deterministic 8-stage pipeline described below, with one
>   trained block-classifier at stage 3 (`dart_semantic/pipeline.py`, also
>   reachable as `pipeline_v2.run(pdf, mode="v1")`). The classifier is the
>   seed of `BERT-Structure` in the target design.
> - **v2** — the full 13-stage cascade from `architecture.md`, now wired
>   end-to-end in `dart_semantic/cascade.py` and reachable as
>   `pipeline_v2.run(pdf, mode="v2")`. Every stage (council → cross-BERT
>   reranker → structure graph → Qwen specialists → two-tier gates →
>   assembler → ThetaEvaluator → exit decider) is implemented in code.
>
> **v2 is model-complete.** A real-runtime corpus eval passed 2026-06-09
> (`data/eval_reports/full_cascade_real_v8_R10_postfix.json`: WCAG 3/3,
> `stage10_pass_rate` 1.0), with the theta `semantic_preservation` dimension
> running live on the v8 full-FT DeBERTa-v3-small head (no stub) and the
> offline-Qwen lane firing. Earlier `*_under_mock` evidence in
> `data/eval_reports/` is superseded by this real-runtime run.

## v1 reference path (legacy seed — 8 stages, one trained model)

> The v2 BERT-council + Qwen-specialist cascade described above is the canonical
> architecture (see [`architecture.md`](architecture.md)). The v1 path below is
> the original single-classifier seed: still runnable and used for data-gen, but
> no longer the system's primary path. The training command in this section
> builds/trains the v1 classifier specifically.


```
1. extract       pikepdf + pypdfium2 + pdfplumber + Tesseract  no model
2. features      per-block layout features (size/gap/caps…)   no model
3. classify      7 rules + distilbert role classifier         ONE model ← trained
4. hierarchy     font-stack heading depth + list nesting      no model
5. ontology_map  (role, depth) → HTML via config              no model
6. enrich        language detect + stubbed vision calls       specialized tools
7. validate      axe-core wcag22aa ruleset                    no model
8. escalate      rule-based ship / llm_fallback / fail        no model
```

Every intermediate block is inspectable; every structural choice traces back
to either a rule, an algorithm, or a classifier confidence.

> **Note on the earlier "zero-violation on four arXiv papers" result.** That
> result was contaminated: v1 used to silently flatten header-less tables into
> `<p class="unlabeled-table">`, so axe-core had no `<table>` to flag. That
> silent fallback has been removed — a header-less table now raises
> `MappingError` and the document is dropped with a stage-5 error (per the
> no-silent-fallbacks invariant). The four reference papers all hit that path,
> so they now drop rather than ship a degraded result. Honest table
> remediation is a v2 (Qwen table specialist) responsibility.

The historical refactor that produced this layout is documented in
[`docs/refactor_plan.md`](docs/refactor_plan.md). The target architecture
(BERT council + Qwen specialists) supersedes it; see
[`architecture.md`](architecture.md).

## Hardware

RTX 3070 8 GB (WSL2 Ubuntu 24.04, Python 3.12). CUDA 12.1+ for the stage-3 classifier. Stages 1/2/4/5/6/7/8 run on CPU.

## Setup

```bash
sudo apt install tesseract-ocr libasound2t64
# Note: Poppler (GPL-2.0) and PyMuPDF/MuPDF (AGPL-3.0) are deliberately NOT
# dependencies. All PDF handling uses pypdfium2 (Apache-2), pdfplumber
# (MIT), pikepdf (MPL-2.0), and Tesseract (Apache-2).
./setup.sh
source .venv/bin/activate
```

## Running

### On a real PDF

```bash
# With the trained classifier (expected path):
python scripts/infer_pdf.py path/to/doc.pdf --save /tmp/result.json

# Without the classifier (rules-only + paragraph defaults):
python scripts/infer_pdf.py path/to/doc.pdf --no-classifier
```

Outputs: role distribution, axe-core result, escalation verdict, and optionally the full resolved-block list saved as JSON.

### Training the classifier

```bash
# 1. Build per-block training data from existing pair files:
python data/build_classifier_data.py

# 2. Train:
python train_classifier.py                  # ~12-15 min on a 3070
# Adapter saved to models/classifier/final/
```

### Generating training data

The pair files under `data/pairs/` and `data/synthetic/` are produced by:

```bash
python scripts/pair_from_wikipedia.py --titles data/seeds/seed_titles.txt --workers 4
python scripts/pair_from_arxiv.py --limit 200 --workers 4
python scripts/gen_synthetic_forms.py --n 200
```

Each script emits `{input_ocr, output_html}` pairs (no JSON target). `data/build_classifier_data.py` derives per-block `(features, role)` labels from them.

## Layout

```
dart_semantic/
  extract.py          stage 1: thin adapter over extract_shared.py
  extract_shared.py   stage 1 core: parallel pikepdf + pypdfium2 +
                      pdfplumber + Tesseract → standardized per-page JSON
  features.py         stage 2: RawBlock → FeatureBlock
  classify.py         stage 3: Role enum + rules + classifier hook
  hierarchy.py        stage 4: depth resolution algorithms
  ontology_map.py     stage 5: role → HTML assembly (supersedes emit_html)
  enrich.py           stage 6: language detect + vision stubs
  validate.py         stage 7: axe-core runner
  escalate.py         stage 8: ship / llm_fallback / fail rules
  pipeline.py         orchestrator — run_pipeline(pdf)
  types.py            dataclasses flowing between stages
  parse_wikipedia.py  data-gen: Wikipedia REST HTML → legacy IR
  parse_ar5iv.py      data-gen: ar5iv HTML → legacy IR
  arxiv_sections.py   data-gen: PDF bookmarks → section page ranges
  arxiv_license.py    commercial-OK license filter
  worker_pool.py      ProcessPoolExecutor helper (per-worker HtmlValidator)
  emit_html.py        [LEGACY] used by data-gen scripts for ground-truth HTML
  ir.py               [LEGACY] tree IR used by emit_html
data/
  build_classifier_data.py    pair files → per-block training JSONL
  synthetic/                  synthetic pair files (gitignored)
  pairs/                      wikipedia + arxiv pairs (gitignored)
  classifier_dataset/         train/val/test JSONL (gitignored)
scripts/
  pair_from_wikipedia.py      generate Wikipedia pairs
  pair_from_arxiv.py          generate arXiv pairs (local PDFs + ar5iv HTML)
  gen_synthetic_forms.py      synthetic form generator
  infer_pdf.py                run full pipeline on a real PDF
eval/
  compare_classifiers.py      side-by-side checkpoint comparison on real PDFs
architecture.md               target architecture (BERT council + Qwen specialists)
docs/
  ontology.md                 standards-grounded structural ontology
  refactor_plan.md            8-stage refactor plan (historical)
  bloat_audit.md              code-cleanup audit (historical)
train_classifier.py           stage-3 classifier training
```

## License stance

Training data sourced only under commercial-permissive licenses (CC-BY, CC-BY-SA, CC0, ODC-By, public domain, arXiv). No LLM-API-derived labels — labels are mechanically extracted from ground-truth HTML tags. Positioning line for procurement: *"DART's structure model is trained exclusively on public and synthetic data, with every training example validated against WCAG 2.2 AA before inclusion."*

See `architecture.md` for the target pipeline (BERT council + Qwen specialist adapters + two-tier validation gate) and `docs/ontology.md` for the exact standards mapping: WAI-ARIA 1.2 APG, HTML Living Standard, W3C ARIA in HTML, PDF/UA (ISO 14289), EPUB Accessibility 1.1, WCAG 2.2 AA.
