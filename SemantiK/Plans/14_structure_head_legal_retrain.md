# Plan 14 — Structure-head legal retrain (kill is_heading / list / blockquote FPs on legal text)

**Status:** SCOPED 2026-06-16 (GPU training is user-gated). Authored from the
loop investigation that drove single-column FRAGMENT to 0 on whitepaper +
encyclopedia + 3 opinions, leaving a residual on citation-dense legal opinions.

## 1. Problem & root cause (quantified)

The Structure council head (`models/council/structure/final`, ModernBERT-base +
LoRA, 5 heads) over-fires three false positives on legal-opinion text, each of
which splits a flowing paragraph and orphans the continuation as a FRAGMENT:

| FP | example | residual frags (8-opinion tally) |
|----|---------|-------|
| `is_heading` on mid-paragraph BODY line | "…procedural due process claim. Following" | 3 (after the Pass-2 plain-heading guard caught the rest) |
| `structural_role=list_item` on citation/enumeration prose | "65 n.3 (explaining…", inline "(1)…and (2)…" | 2 (blocks the continuation bridge) |
| `structural_role=blockquote` on body narrative | "…His mental health also" | 2 (after the Pass-5 flow-through override caught the rest) |

**Root cause (confirmed):** the trained dataset (`data/structure_dataset`,
74,867 examples) contains **zero legal opinions** — sources are wikipedia /
openstax / federal_register / gutenberg / forms / arxiv / synthetic. blockquote
is only 1.8% of labels. The head has **never seen legal-opinion structure**
(citation-dense prose, footnote stamps, lettered/roman section headings,
indented quotes), so legal body lines fall outside the textbook "paragraph"
distribution it learned and trip heading/list/blockquote. The deterministic
`structure_graph.py` guards added in the loop compensate for the SAFE subset;
the residual needs the head itself to stop mislabeling — i.e. a retrain with
legal hard-negatives.

structure_graph guards already shipped (keep them — they generalize): Pass-2
multi-line heading absorption + style-distinct guard + plain-heading FP guard;
Pass-5 blockquote-FP flow-through override; list wrapped-tail / not_same gate;
cross-page split; page-height margin gate; running-header page-# gate.

## 2. Data finding — courtlistener ground-truth is too FLAT to use as-is

`data/pairs/courtlistener/*.json` (532 pairs) carry `input_ocr` (with
`[size=… top]` / `[gap=…]` layout markers) AND `output_html` ground truth — but
the HTML is **flat**: exactly one `<h1>` (case caption) per doc and **every
other block is `<p>`** (across 40 sampled: h2–h6 = 0, blockquote = 0, li = 0).

Labels in `data/build_structure_data.py` are derived from HTML tags
(`TAG_TO_ROLE`: h1–6→heading, p→paragraph, li→list_item, blockquote→blockquote;
`is_heading=1` iff h1–6). So ingesting courtlistener as-is would teach:
- ✅ legal body / citations / narrative → paragraph, not_heading  (the negatives we want)
- ❌ legal SECTION HEADINGS ("DISCUSSION", "I. Procedural History", "A. Duration")
  → paragraph, **is_heading=0** — suppressing real headings. WCAG-worse than a
  fragment (a vanished heading breaks screen-reader navigation).

**WORSE THAN FLAT — heading text is GLUED INTO the body block.** Calibrating the
heading regexes against real data, every match looks like
`<p>II. STANDARD OF REVIEW In analyzing Rosa's sentencing challenges, we…</p>` —
the section heading is **merged into the following body paragraph**, not a
separate node. The `input_ocr` does the same ("…remand for resentencing in that
case. I. BACKGROUND We recap the key facts…"). So courtlistener's pre-generated
OCR **and** HTML never separate legal headings from body. There is no clean `<p>`
to retag — you would have to SPLIT each block at the heading boundary first.

**Conclusion: the courtlistener `output_html` is unusable as heading ground
truth.** Retagging it is not enough; the heading/body separation itself is wrong.

**Viable path — construct ground truth from DART's OWN extraction.** DART's
`extract_shared`→`featurize` DOES separate "I. BACKGROUND" from "We recap…" into
distinct FeatureBlocks (that is exactly what `structure_graph` measures). So
build legal training labels by:
1. Run DART extraction on each courtlistener PDF (`data/cache/courtlistener/*.pdf`)
   → correctly-separated FeatureBlocks.
2. Deterministically pseudo-label each block: heading regex (roman/lettered/
   allcaps-banner, short + set-off via `relative_font_ratio`/`is_bold`/gap) →
   heading; citation/enumeration/body → paragraph; indented multi-line run →
   blockquote; everything else paragraph.
3. **AUDIT** (manual Claude review of ~20 docs): every real section heading
   labeled heading, no body line labeled heading, citations labeled paragraph.
   Iterate the rules until clean. This is the bulk of the work and the gate.
4. Emit as the per-pair JSONL the builder expects (text, layout, labels{...}),
   OR as a synthetic structured `output_html` the existing builder can align.

This is partly circular (heuristics labeling data to train a head that replaces
heuristics), BUT the heuristics here run on CORRECTLY-SEPARATED blocks (where a
regex on the isolated "I. BACKGROUND" block is reliable), unlike the glued
OCR/HTML — and the audit gate bounds the circularity. The payoff: the head
learns the legal-text *distribution* (citation density, footnote stamps, lettered
headings) which generalizes beyond the specific regexes.

## 3. Re-annotation (the new work) — `data/annotate_courtlistener_structure.py` (TODO)

Produce corrected `output_html` (or a sidecar label JSON) for each courtlistener
pair, deterministic + auditable:

- **Headings** → wrap in `<h2>`/`<h3>` when a block matches a legal-heading
  signal AND is short (≤ ~12 words) AND is set off (the OCR `[size=lg|xl]` /
  `[bold]` / `top`/`gap=lg` markers, OR a heading regex):
  - roman section: `^(?:[IVXLC]+\.)\s` ("I.", "II.")
  - lettered subsection: `^[A-Z]\.\s` ("A.", "B.")
  - allcaps banners: `^(DISCUSSION|BACKGROUND|CONCLUSION|ANALYSIS|OPINION|ORDER|FACTS?|PROCEDURAL HISTORY)\b`
  - numbered: `^\d+\.\s` (guard against citations — require short + set-off)
  Level: h2 for roman/allcaps, h3 for lettered/numbered.
- **Block quotes** → `<blockquote>` when the OCR shows an indented run (left
  margin > body median) of ≥2 lines, OR a `[quote]`-style marker if present.
- **Everything else stays `<p>`** — these are the hard negatives (body lines
  with internal periods, citations "65 n.3 …", inline enumerations) that teach
  the head NOT to fire heading/list/blockquote.
- **Audit gate:** run the annotator, then sample 20 docs and manually verify
  (Claude review) that (a) every real section heading got `<h_>`, (b) no body
  line got `<h_>`, (c) citations stayed `<p>`. Iterate the regex/markers until
  clean BEFORE building the dataset. Do NOT train on un-audited pseudo-labels.

Fallback if rule-based annotation is too noisy: bootstrap from the CURRENT
(guard-corrected) `build_structure_graph` output as pseudo-labels, but only
after the same manual audit — it is circular and must be spot-checked.

## 3b. STATUS — constructor BUILT + AUDITED (2026-06-17)

`data/build_courtlistener_pseudo_structure.py` is implemented and the audit
gate is GREEN:
- Runs DART `extract_shared_cached` on `data/cache/courtlistener/*.pdf`, mirrors
  build_structure_data.py's span extraction + `compute_span_layout_features`
  (schema-identical: same keys, same 5 label heads, layout dim 20 — verified).
- Deterministic labels: `_is_heading` = roman/lettered/numbered/banner prefix,
  ≤12 words, NOT sentence-ending (`_ends_sentence_loose` handles curly quotes),
  banner ≤5 words; real `•` bullets → list_item; everything else → paragraph.
- Audit (`--audit N`) on 12 opinions: real section headings correctly tagged
  ("I. Background", "II. Standard of Review", "A. Facial Challenge",
  "1. Plain Text of the Constitution", "OPINION"); ZERO body→heading FPs after
  two fixes the audit caught (case-insensitive banner matching common words
  "Order/argument/conclusion"; sentence-ending lines "opinion."/"1. …probative.");
  captions/party-names/dates correctly stay paragraph; no real headings missed.
- `--build --para-cap 6000` writes ONE balanced file
  `data/structure_dataset/per_pair_legal/courtlistener_pseudo.jsonl`: **8,895
  examples (6000 paragraph + 2617 heading + 278 list_item)** from 523 opinions,
  with **9 eval opinions held out** (`--holdout`, the loop's measured set).
- Wired into `build_structure_data.py` via `--include-legal-pseudo` (appends the
  pre-balanced legal set AFTER source-capping so its rare legal headings are not
  re-subsampled away; typed SystemExit if the file is missing).

## 4. Build + balance (legal set already balanced in 3b)

Rebuild the full dataset, appending the pre-balanced legal set (regenerates
train/val/test.jsonl). Use a fresh `--out-dir` to PRESERVE the current dataset
until the retrain is validated:
```
python data/build_structure_data.py --include-legal-pseudo \
  --out-dir data/structure_dataset_legal
```
- The 8,895 legal examples (≈10.6% of the resulting ~84k corpus) are appended
  after capping, NOT re-capped — so the 2,617 legal headings survive (a uniform
  source-cap would drop them). The textbook/wiki/arxiv sources keep their
  existing per_pair alignment; only the split is recomputed.
- Verify the new `coverage_report.json`: blockquote/code/form counts must NOT
  drop vs current; legal heading count (numbered/lettered) > 0; source
  `courtlistener_pseudo` present.
- NOTE: a plain `build_structure_data.py` run re-aligns ALL pairs (slow). If
  per_pair is already current, a merge-only path (read per_pair + append legal
  + stratified_split) is enough — see if a `--skip-align` shortcut is worth
  adding before the full run.

## 5. Train (existing `train_structure.py`; GPU-gated — user go/no-go)

```
CUDA_VISIBLE_DEVICES=0 python train_structure.py \
  --dataset-dir data/structure_dataset --output-dir models/council/structure \
  --base-model answerdotai/ModernBERT-base \
  --epochs 12 --patience 4 --lr 2e-4 --batch-size 16 --max-length 192 \
  --lora-r 16 --lora-alpha 32 --weight-cap 30.0 --is-heading-sampler-cap 8.0 \
  --snapshot-policy weighted_macro
```
- 12 epochs / patience 4 (the 6-epoch default under-trained is_heading — see
  feedback "6-epoch undertraining"; Plan 13 used 12/4).
- Respect the CUDA-context guard: no second CUDA process (no `pytest tests/`)
  while training (see feedback_train_cuda_context_guard).
- Then recalibrate: `python scripts/calibrate_structure_heads.py`.

## 6. Eval gates (must ALL hold to ship)

1. `train_structure.py` test metrics: `role_macro_f1 ≥ 0.906` (current), and
   `is_heading_pos_f1` ≥ current (do NOT trade heading recall away).
2. `scripts/eval_cascade_structure_to_semantic.py --mode endtoend`: per-source
   P/R/F1 not regressed on textbook/wiki sources.
3. **The real gate — real-runtime FRAGMENT on a held-out opinion set** (the
   loop's metric): the 9-doc deterministic baseline (memory) must improve
   toward 0 with **NO heading loss** — re-measure heading counts too (shades
   must keep its 39; opinions keep their real DISCUSSION/I./A. headings).
   Hold out ~5 courtlistener opinions from training for this.
4. Regression: shades=0, all wiki=0, no new FRAGMENTs anywhere.

## 7. Risks

- **Heading suppression** (the #1 risk): if annotation under-marks real legal
  headings, the head learns to flatten legal docs. Mitigated by the audit gate
  (§3) + the is_heading_pos_f1 and heading-count eval gates (§6).
- **Swamping minor classes**: mitigated by the 8k cap + coverage_report check.
- **Determinism**: council inference is deterministic (verified) — A/B
  FRAGMENT against the immediate prior checkpoint, never a stale number.
- **Cost**: ~hours on the 8GB card; user-gated.

## 8. Open decisions for the user
- Approve the GPU training run (§5) once the dataset audit (§3) is green.
- Annotation approach: rule-based (§3 preferred) vs bootstrap-from-pipeline.
- Whether to also annotate blockquotes now or defer (headings are the priority;
  blockquote FPs are largely handled by the Pass-5 flow-through guard already).
