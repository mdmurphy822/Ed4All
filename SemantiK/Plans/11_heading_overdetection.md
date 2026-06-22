# Plan 11 — Heading over-detection correction

**Status:** investigated 2026-06-08. **Stages 1 + 2 + 3 IMPLEMENTED + validated 2026-06-08** —
the blocker is RESOLVED and the heading outline now recovers real section structure (gate PASS).
Stage 4 (retrain) deferred — not needed. ⚠ Stage 3 requires an extract-cache invalidation to take
effect globally (see Stage 3 below). Blocker for math-heavy papers passing the document
WCAG `heading_tree` gate (R10 finding 1B — see `Plans/10_completion_punchlist.md`,
memory `project_r10_corpus_eval.md`).

## Stage 1 result (validated)
`dart_semantic/structure_graph.py:_plausible_heading()` guard wired into the Pass-2 heading
promotion. On `1807.02622v2` (mock-runtime smoke, real Stage-5 council + real doc gate):
**heading count 234 → 17 (−93%); `heading_tree` gate FAILED → PASSED; document `wcag_status`
failed → PASSED** (all 6 checks green). Real headings (title, `Abstract:`, `3. An Information
Inequality`, `6. Conclusion`, `References`) survive. Unit tests in `tests/test_structure_graph.py`
(`test_plausible_heading_keeps_real` / `_drops_garbage`). The gate depends only on the heading
SET + normalized levels (Stage 5), identical mock vs real, so the mock validation is faithful.

## Problem
On `1807.02622v2` the assembled doc emitted **234 `<hN>` headings**, almost all false. The
document gate fails `heading_tree: level skip` — not mathml/html5 (those pass). The per-region
level normalizer (`assembler/heading_tree.py`) works; the failure is the *volume/order* of
garbage headings. Masking via level-renormalization was rejected (would ship 234 fake headings,
anti-accessible, violates the no-silent-fallbacks rule).

## Findings (quantified, 234 headings)
| Category | Count |
|---|---|
| Empty / whitespace-only | 60 |
| Single/double-char fragments (".", "v", "1") | 23 |
| Math-glyph fragments ("∑ ∑", "(cid:112)") | ~41 |
| Long body sentences (>12 words) | 37 |
| Sentence fragments (end in `. , :`) | 19 |
| Concatenated-no-space ("frameworkforderiving…") | 15+ |
| Genuinely heading-shaped | **~6** |

Level dist `{1:2, 2:37, 3:33, 4:34, 5:23, 6:105}` — level-6 (body-font) dominates → the
`is_heading` head fires on blocks with **no font elevation**. Many numbered matches are
**bibliography entries** mis-read as numbered sections.

**Verdict: (A) Structure-head over-prediction dominates (~80%); (B) space-losing extraction amplifies (~15).**

## Decision points (file:line)
- **Promotion (only predicate):** `dart_semantic/structure_graph.py:658-679` — `is_heading` top-1
  == "heading" AND `conf >= is_heading_threshold`. **No text-shape guard.**
- **Threshold:** `structure_graph.py:559` default **0.7**, no runtime override (live as-is).
- **Level hint:** `structure_graph.py:294-337` (`_level_hint`, font-ratio + numbered-section).
- **Emission:** `assembler/pass_9a.py:266-302` — iterates every `kind=="heading"`, no filter.
- **Known soft penalty (not a drop):** `soft_reranker/document.py:141-166` already penalizes
  empty/≤2-char headings — proves the signal is known but never hard-drops.
- **Extraction space loss:** `extract_shared.py:250-290` (pypdfium2 rect text) — orphan blocks
  carry concatenated text; no glyph-gap space recovery. pdfplumber path (line 348) preserves spaces.

## Correction plan (staged)
1. **[deterministic-code, S, highest impact]** Add `_plausible_heading(text, level_hint)` guard at
   `structure_graph.py:665` (drop empty, ≤2-char, >~12-word, math-shaped, concatenated-no-space>25
   unless real numbered-section prefix). Removes ~195/234 (~83%) → gate passes. Reuse the
   empty/short logic from `soft_reranker/document.py:160-161`. Re-route dropped heading FBs to
   prose (no silent loss). Validate: `scripts/measure_stage5_heading_rate.py` + re-run R10 smoke.
2. **[deterministic-code, S] — DONE + validated 2026-06-08.** Ran `calibrate_structure_heads.py`
   on `val.jsonl` (n=12348): fit **T=1.6553** (guards pass: NLL +0.0225, ECE 5.68×), wrote
   `is_heading_temperature` into `models/council/structure/final/heads.pt` (backup `.bak` kept;
   runtime already plumbs it at `council/structure.py:286,456`; Semantic uses RAW is_heading so
   no cascade skew). Raised `build_structure_graph` default `is_heading_threshold` 0.7 → **0.8**
   — calibrated max-F1: **P=0.926 / R=0.902 / F1=0.914** (vs raw-0.7 P=0.872/R=0.935). Validated:
   1807.02622 mock smoke heading **17→11** (234→11 = 95% total), gate still PASS; 116 tests green.
   Reversible (restore `.bak` + revert 0.8). Residual 11 are mostly concatenated-text body lines
   → Stage 3 target. Recall cost ~10% real headings on this hard paper (acceptable per val R=0.902).
3. **[extraction, M] — DONE + validated 2026-06-08.** Root cause was NOT pypdfium2 (its
   `get_text_bounded` preserves spaces) — it was **pdfplumber** `extract_words(use_text_flow=True)`
   with the default 3pt `x_tolerance`, which never splits LaTeX PDFs whose inter-word gaps are
   <3pt (no space glyph). The merge prefers pdfplumber (for font info), so the concatenated text
   won. Fix: added `x_tolerance_ratio=_X_TOLERANCE_RATIO` (0.15, font-scaled) to the
   `_pdfplumber_page` `extract_words` call (`extract_shared.py`), with a `TypeError` fallback for
   older pins. Validated on 1807.02622: the merged block went `"frameworkforderiving…"` →
   `"framework for deriving various EPIs…"`, and the **heading outline RECOVERED real structure**
   (mock smoke heading 11→20 but now readable: "2. Preliminary Definitions and Properties",
   "4. First Version of the Rényi EPI", "5. / 5.1. / 5.2.", "6. Conclusion" — previously
   concatenated garbage or missing). Gate still PASS (n=20). 29 extract tests pass. The count
   *rising* is real-heading recovery, not regression — a genuine a11y/navigability win.
   **⚠ Cache caveat:** `data/extract_cache` (1820 files / 1.6G) is keyed on PDF mtime, not code
   version, so it's STALE for previously-extracted PDFs. The fix only applies to fresh extractions.
   Run eval with `--invalidate-extract-cache`, or clear the cache, for the fix to take effect
   globally (incl. a re-run of R10).
4. **[retrain, L, deferred — last resort]** Structure `is_heading` retrain with math/reference hard
   negatives. Only if 1-2 leave residuals. Cap dominant paragraph class first
   (`feedback_balance_dominant_category`). RTX 3070 8GB constraint.

## Risks & guards
- Don't drop long *real* headings ("Keywords:", "Abstract:") — keep word cap generous (~12-15),
  exempt numbered-section prefixes, special-case the title (level 1, large font).
- Never drop the only h1 (would flip to a different gate failure) — always keep highest-font/first.
- Watch `measure_stage5_heading_rate.py` count-vs-reference: must drop on math/physics WITHOUT
  falling below reference on prose/form. `_score_outline_cleanliness` rising = free sentinel.

**Recommended order:** 1 → 2 → (3 if needed) → (4 last resort).
