# DART/Semantic — Assembler Layer Investigation (Stages 9–12)

> **STATUS (banner added 2026-06-17): SHIPPED.** The assembler layer (stages 9–12) is
> implemented and exercised by the R10 real-runtime corpus eval. Historical design record.

**Document version:** 2026-05-03
**Scope:** Phase 9 (deterministic assembler), Phase 9b (Qwen-GapFill), Phase 9c (merge-back), Phase 10 (doc-level hard gate + soft reranker + exits).
**Authoritative target:** `architecture.md` §2 (Stage 9–12), §4 (Qwen specialists, §4.2), §5 (two-tier gate), §6 (exits), §10 (locked choices).
**WCAG/standards mapping:** `docs/ontology.md` §2 + §3 + §7.
**Self-contained.** Future readers do not need conversation history. Cite-only file references are absolute paths.

This document is the architectural deep-dive on the bottom half of the pipeline. It is read by Phase 0 (to land type definitions) and again by Phase 9–10 (to land implementation). It is not a redesign — it pins down algorithms, edge cases, and decisions that the architecture and plan name but do not specify.

---

## Executive map (what each section answers)

| § | Question |
|---|---|
| 1 | What does Phase 9a do, concretely, for each of its six sub-tasks? |
| 2 | How does the assembler decide a slot is a `GapSlot` rather than empty/missing? |
| 3 | How does the gap-fill Qwen pass invoke (K, selection, failure mode)? |
| 4 | How does merge-back splice and re-stabilize the document? |
| 5 | What does the document-level hard gate check, and how strict is each check? |
| 6 | When does the document-level soft reranker fire, and how does it score? |
| 7 | What does each of the three exits emit? |
| 8 | How much v1 code is salvageable per module? |
| 9 | What decisions does the user need to answer? |
| 10 | What types must Phase 0 land in `dart_semantic/assembler/types.py`? |

---

## Section 1 — Deterministic assembler (Phase 9a)

The assembler runs after Phase 8 has selected the per-region top-1. Its input is **a list of per-region top-1 candidates plus the BERT council state** (the typed signals the BERTs produced). Its output is **either a single assembled HTML string + a list of zero `GapSlot`s** (clean exit to Stage 10), **or** an HTML draft + a non-empty list of `GapSlot`s (which routes through Phase 9b before Stage 10).

Sub-tasks fire in a fixed order so each can rely on prior sub-task output:

```
1. reading-order DOM placement   → ordered region stream
2. heading-hierarchy normalize   → consistent <h1..h6> tree
3. list-continuation             → merged <ul>/<ol>
4. ARIA landmark wiring          → <main>/<nav>/<aside>/<footer>
5. doc shell                     → DOCTYPE/<html lang>/<head>/<title>/skip-link
6. reference resolution          → <a href="#sec-...">
7. gap detection                 → emit GapSlot list
```

Each sub-task is below.

### 1.1 Heading hierarchy normalization

**Inputs.**
- Per-region top-1 HTML fragments where Structure has called `heading` and the multi-head has emitted a heading-level prediction in {1..6}.
- The cross-BERT reranker's `Routing.structure_role` (so we know which regions are headings).
- BERT-Semantic's top-k for `title` doc-role (the document title block, if any, claims H1).
- The reading-order skeleton (Stage 2) so headings are walked in document order.

**Algorithm — "promote the first, demote forward, never skip."**

```
last_emitted = 0
saw_h1       = False
title_index  = first region where Semantic top-1 == "title"

for region in document_order(headings):
    raw = region.predicted_heading_level   # from BERT-Structure heading-level head
    conf = region.predicted_heading_confidence

    # Force-claim H1 from Semantic.title once.
    if region.index == title_index and not saw_h1:
        emitted = 1
    elif not saw_h1:
        # No Semantic.title gating; first heading is the H1 unconditionally.
        emitted = 1
    else:
        # Standard demote-forward rule (matches v1 emit_html policy).
        # Allow at most last_emitted + 1; clamp to [1, 6].
        emitted = min(max(raw, 1), min(6, last_emitted + 1))

    region.heading_level = emitted
    last_emitted = emitted
    saw_h1 = True
```

**Key choice — demote-to-fit, never promote.** Promoting to fit (raising H4 → H2) destroys upstream evidence; if Structure said H4 with high confidence and the prior sibling was H1, the document is genuinely skipping a level. We demote-to-fit (H4 → H2 in that case) because demotion preserves Structure's *relative* call (still a subordinate heading) without claiming evidence we don't have.

**Edge case — Structure says H4 high-confidence, context says H2.** The algorithm emits H2 (`min(4, last_emitted + 1) = min(4, 2) = 2`). High confidence cannot win against the no-skip rule because the no-skip rule is a WCAG hard requirement (SC 1.3.1, technique H42). We log the demotion as a soft signal that may feed Phase 11's heading-tree-balance score.

**Edge case — no title, no headings at all.** Document has no `<h1>`. The legacy `ontology_map.emit_html` raises `MappingError` here. The new assembler **does not raise**: it flags `GapSlot(kind=missing_title, context=<first paragraph + doc metadata>)` and continues. Phase 9b's gap-fill produces the `<h1>`; merge-back splices it.

**Reuse of v1.** `/home/user/Projects/Semantic/dart_semantic/ir.py:234-263` (`normalize_heading_levels`) implements this algorithm on `ir.Document`. Lift the algorithm; rewrite the I/O for the new region-stream input.

### 1.2 ARIA landmark wiring

**Inputs.**
- BERT-Semantic top-k for each region. The doc-role labels are
  `{title, author, abstract, body, citation, footer, legal, metadata}`
  (`architecture.md:195`).
- A confidence threshold τ_landmark (recommended 0.70; final value is part of Phase 7-style threshold tuning per `architecture.md` §6).

**Mapping (Semantic doc-role → landmark).**

| Semantic top-1 | Landmark wrapping |
|---|---|
| `title` | none directly (becomes `<h1>` inside `<main>`) |
| `author` | `<address>` inside `<main>` (NOT `<aside>` — author byline is primary content) |
| `abstract` | none directly (`<p>` or `<section aria-labelledby=>` if a heading precedes it) |
| `body` | none (default placement inside `<main>`) |
| `citation` | accumulated into a single `<section epub:type="bibliography">` at end of `<main>` |
| `footer` | `<footer>` at body scope (= role=contentinfo); ONLY if confidence ≥ τ_landmark |
| `legal` | `<aside class="legal">` (NOT `<footer>` — legal isn't always page contentinfo) |
| `metadata` | dropped from visible output (matches v1 `ontology_map.py:65-68`) |

**The footer rule (concrete).** A region is wired into `<footer>` only when:

1. BERT-Semantic top-1 == `footer` AND
2. Confidence ≥ τ_landmark (default 0.70) AND
3. The region is in the bottom 25% of its containing page (geometric corroboration from Stage 2) AND
4. No other heading or main-content region appears below it on the same page.

A low-confidence footer call on body text is **not** wired into `<footer>`. It falls through to default `<p>` inside `<main>`. This is the conservative posture — under-wiring landmarks is a comprehension penalty; over-wiring landmarks (especially `<footer>` and `<aside>`) is a screen-reader-navigation regression because landmark-mode users skip those regions.

**`<main>`.** Always emitted; always wraps everything that is not explicitly a body-scope `<header>`/`<footer>`/`<nav>`/`<aside>`. Per `docs/ontology.md` §5 first-rule-of-ARIA, `<main>` does NOT carry `role="main"`.

**`<nav>`.** Emitted when a contiguous run of regions are all classified by Semantic as `body` AND their text matches a TOC pattern (a v1 BlockType.TOC_ITEM heuristic — series of short hyperlinked lines or numbered short lines that map to subsequent headings). Concretely: `<nav aria-labelledby="toc-heading">`. Skip the `<nav>` wrapper when only one TOC-ish line is present (false positive cost too high).

**`<aside>` and v1.** Per `docs/ontology.md` §7, the v1 IR has no `ir.Aside`. Recommendation: **emit `<aside>` only for the `legal` doc-role**, not for footnotes or sidebars in v1. Adding `<aside>` for footnotes requires an ontology decision and is best deferred to v2 (this is Section 9 Q3).

### 1.3 List continuation across layout/column breaks

**The problem.** A list interrupted by a column break or a page break frequently arrives as two separate "list" regions in the Stage-2 reading order. Each per-region top-1 emits a self-contained `<ul>` or `<ol>`. The naive concatenation produces two adjacent `<ul>`s when one should exist.

**Concrete heuristic — merge if all four hold:**

1. Same list kind (both `<ul>`, both `<ol>`, both `<dl>`).
2. Same predicted list-marker style (BERT-Structure's list-nesting head exposes marker family: bullet • | dash – | numeric 1. | alpha a. | roman i.). For numeric/alpha/roman, the second region's first marker must be the next-in-sequence (e.g. region 1 ends "3.", region 2 starts "4.").
3. Adjacent in reading order — no intervening region of any other kind UNLESS the intervening region is a `figure_caption`/`figure`/`table` block AND the geometric column/page break pattern says the figure is interruption-only. Concretely: the figure must sit on the same column boundary or bridge a page break with no heading.
4. No heading between them at any level.

**If only (1)–(3) hold but (4) fails (a heading interrupted the list):** never merge. The heading defines a new sub-section; the list under it is logically a new list.

**Algorithm sketch.**

```
def merge_lists(regions):
    out = []
    pending_list = None
    for r in regions:
        if r.kind == "heading":
            flush(pending_list, out); pending_list = None
            out.append(r)
        elif r.kind == "list":
            if pending_list and continuation_compatible(pending_list, r):
                pending_list.items.extend(r.items)
            else:
                flush(pending_list, out); pending_list = r
        elif r.kind in ("figure", "table"):
            # Interruption-tolerant if continuation_compatible would still hold
            # for the next list seen. Buffer the figure but keep pending_list alive.
            out.append(r)
        else:
            flush(pending_list, out); pending_list = None
            out.append(r)
    flush(pending_list, out)
    return out
```

**Reuse of v1.** `/home/user/Projects/Semantic/dart_semantic/hierarchy.py:109-139` (`_resolve_list_groups`) groups contiguous `LIST_ITEM` blocks by indent_bucket. Salvageable for the within-a-region nesting; the across-region continuation is new.

### 1.4 Reference resolution

**Scope.** Convert intra-document plain-text references to anchor links:

- `"Section 3.2"` → `<a href="#sec-3-2">Section 3.2</a>` (when the document has a heading whose section-number prefix matches `3.2`).
- `"[12]"` → `<a href="#ref-12">[12]</a>` (when the document has a `REFERENCE` block tagged with reference_number 12).
- `"Figure 5"` → `<a href="#fig-5">Figure 5</a>` (when a `figure_caption` exists with that label).
- `"Table 2"` → `<a href="#tab-2">Table 2</a>`.

**Algorithm.**

1. **Pass 1: build a reference index.** Walk the document. For each heading, extract the section number prefix (the v1 `_leading_section_number` regex in `emit_blocks.py:227-240` is the model — extend to recognize letter+decimal+roman mixes like "A.3.2.iv"). For each REFERENCE block, extract its reference_number. For each Figure/Table caption, extract "Figure N" / "Table N". Build a dict mapping the canonical key to a generated anchor id (`sec-3-2`, `ref-12`, `fig-5`, `tab-2`).

2. **Pass 2: regex-find candidate references.** Run a conservative regex set over each region's plain text (skip headings/captions themselves):
   - `\bSection\s+([0-9A-Z]+(?:\.[0-9A-Za-z]+)*)\b`
   - `\bSec\.?\s+([0-9A-Z]+(?:\.[0-9A-Za-z]+)*)\b`
   - `\bAppendix\s+([A-Z](?:\.[0-9]+)*)\b`
   - `\bFigure\s+(\d+)\b` / `\bFig\.?\s+(\d+)\b`
   - `\bTable\s+(\d+)\b` / `\bTab\.?\s+(\d+)\b`
   - `\[(\d+(?:,\s*\d+)*)\]` (numeric citations, possibly multi)

3. **Pass 3: resolve, splice, or flag.** For each match:
   - **Resolves cleanly** (target exists in the index): wrap match in `<a href="#anchor">match-text</a>`.
   - **Ambiguous text** ("see above", "the previous section", "the foregoing"): leave as plain text. Do NOT flag as `GapSlot` — the slot is genuinely under-specified in the source and gap-fill cannot do better than rules.
   - **Target missing** (regex matched "Section 4.2" but no heading with section_number "4.2" exists): flag `GapSlot(kind=citation_unresolved, context={match_text, surrounding_50_chars, candidate_targets_with_close_section_numbers})`. The gap-fill specialist may either (a) propose the nearest-match anchor, or (b) confirm "leave as plain text" and the merge-back skips the splice.

**Why flag, not silently leave plain text.** A "Section 4.2" reference whose target doesn't exist may signal an upstream parsing error (a heading was missed). Flagging gives gap-fill a chance to resolve OR confirms the document is genuinely incomplete. Either is more informative than silence.

### 1.5 Doc shell construction

**Fixed.**
- `<!DOCTYPE html>` (always).
- `<meta charset="utf-8">` in `<head>`.
- `<main>` body wrapper (always).
- Skip-link `<a class="skip-link" href="#main-content">Skip to main content</a>` immediately after `<body>` open. The skip-link target is `id="main-content"` placed on `<main>`. This is borrowed from the DART downstream `wcag_enhancer._add_skip_link` pattern referenced in `docs/ontology.md` §4 + §7. The Semantic emitter has not historically emitted this; the new assembler does because it owns full doc shell now. (See `docs/ontology.md` §7 row "Skip link" for the v1 split-of-responsibility — we are *folding the responsibility back upstream*.)

**Variable — and where each value comes from.**

| Slot | Source ladder (first hit wins) |
|---|---|
| `<html lang>` | (1) PDF metadata `Lang` if set and BCP-47-shaped. (2) `lingua` language detection over the body text (≥ 200 chars sampled). (3) Default `"en"`. **No GapSlot.** Language detection is robust enough that this never needs Qwen. |
| `<title>` | (1) Region with BERT-Semantic top-1 == `title` AND confidence ≥ 0.70. (2) Document's first `<h1>`. (3) PDF metadata `/Title` if non-empty and not a known boilerplate string ("untitled", "document1", filename-shaped). (4) **GapSlot(missing_title)** — context = first 1000 chars of body. |
| `<head>` `<meta name="description">` | Optional. Skip in v1; not required for WCAG. |

**Why language has no GapSlot but title does.** Language detection on body text is high-precision and the cost of a wrong language tag is low (a screen-reader picks the wrong voice but the content is still readable). Title is high-stakes (WCAG SC 2.4.2 hard fail if empty), high-variance (many PDFs have no clean title), and gap-fill has clear fragments-of-context to work from.

### 1.6 Reading-order DOM placement

**Inputs.** The Stage-2 reading-order skeleton — a sequence of region-IDs in canonical reading order across columns and pages.

**Algorithm.** The assembler **does not re-derive reading order**. It walks the Stage-2 sequence and emits regions in that order, with the structural transformations from §1.1–§1.4 applied along the way. Multi-column documents have already been linearized by Stage 2.

**Key edge case — when per-region top-1 changes a region's role.** Stage 2 produced reading order assuming visual flow. If Phase 8's top-1 reclassified region N from `paragraph` to `heading`, that does NOT move the region in DOM order. It changes the *element wrapping* the region's text but keeps document position. Reading order is a layout-derived skeleton; structural role is independent.

**Consequence.** The "DOM position N is now a heading" event is the trigger for §1.1 (heading hierarchy normalization) — the heading is inserted into the heading-level tree at the document-order position, and surrounding heading-level claims may demote-forward.

---

## Section 2 — Gap detection (the bridge from 9a to 9b)

Gap detection is the **last** sub-task of Phase 9a. The assembler enumerates only the four supported `GapSlot` kinds (architecture §4 narrow scope — `architecture.md:501-503` locks this); any other "missing" slot is filled by a deterministic fallback (e.g. empty title → "Untitled document"; missing language → "en"; ambiguous reference → leave as plain text).

| Gap kind | Detection trigger | Context to ship to gap-fill |
|---|---|---|
| **`missing_title`** | After §1.5's title ladder runs and produces no candidate. Concretely: no Semantic.title with conf ≥ 0.70, no `<h1>` emitted by §1.1, PDF metadata `/Title` is empty/boilerplate. | `{first_paragraph_text: str (≤500 chars), first_h1_text: str|None (if §1.1 produced any heading at all, even non-title), pdf_metadata_subject: str|None, doc_lang: str, source_corpus_hint: str (e.g. "arxiv"|"openstax"|"unknown")}` |
| **`citation_unresolved`** | §1.4 found a "Section X.Y" or "[N]" or "Figure N" or "Table N" pattern whose target does not exist in the document index, AND at least one near-miss target exists (Levenshtein distance ≤ 2 on the section number, or |N - target_N| ≤ 1 for numerics). If no near-miss exists at all, do NOT flag — the reference is genuinely external/dangling and gap-fill won't help. | `{match_text: str, surrounding_50_chars: str, candidate_targets: list[{key: str, anchor_id: str, snippet: str}], reference_kind: Literal["section","figure","table","numbered_citation"]}` |
| **`author_block`** | BERT-Semantic top-1 == `author` AND prose adapter's per-region top-1 is a generic `<p>` (i.e., not wrapped in `<address>`). Author blocks have specific markup conventions (`<address>` with comma-separated affiliations) that the prose adapter is the wrong objective for. | `{raw_text: str, surrounding_context: str (≤300 chars before and after), semantic_confidence: float, structure_role_topk: list[(role, conf)], doc_corpus_hint: str}` |
| **`copyright_block`** | BERT-Semantic top-1 == `legal` (sub-flag = copyright via heuristic: text matches `r"\b(©|copyright|\(c\)|all rights reserved)\b"i`) AND prose adapter top-1 doesn't already wrap in `<aside>` or `<footer>`. | `{raw_text: str, has_copyright_year_match: bool, semantic_confidence: float, surrounding_context: str}` |
| **`legal_disclaimer`** | BERT-Semantic top-1 == `legal` AND copyright sub-flag is FALSE (i.e., disclaimer/license/terms language). Same gating as copyright. | Same shape as `copyright_block` plus a `legal_subkind` discriminator field. |

**Threshold notes.**

- The Semantic confidence threshold for triggering author/copyright/legal gap-fill is τ_gap = 0.60 (deliberately lower than the τ_landmark=0.70 used for landmark wiring). Rationale: a low-confidence Semantic call still warrants asking gap-fill "what does this look like as `<address>`?" — the wrong answer falls through to the prose top-1; the right answer remediates.
- These thresholds are placeholder defaults; tune in Phase 7-style threshold tuning once gap-fill training data exists.

**Gap-fill input prompt schema (concrete proposal).**

```
SYSTEM: You produce a single HTML5 fragment that satisfies WCAG 2.2 AA for the
named slot kind. Output only the fragment, no commentary.

USER: slot_kind = <one of: missing_title | citation_unresolved | author_block |
copyright_block | legal_disclaimer>

context:
  <a JSON object matching the per-kind schema in the table above>

Produce the HTML fragment.
```

**Gap-fill output schema.** A single HTML fragment per generation. K candidates per gap → K fragments. Per-gap soft scoring (§3) selects one. **Each candidate is independently passed through the per-region hard gate** (Section 3 below argues for this). Fragments that fail the gate are dropped before scoring.

---

## Section 3 — Phase 9b: Qwen-GapFill invocation

### 3.1 K candidates per gap

**Recommendation: K=4 fast lane / K=8 offline lane** — same as Plan DP-5.2 for region specialists.

Reasoning against a lower K for gap-fill:
- Gap-fill outputs are *narrower* than prose remediations (a title is one line; an author block is 2–4 lines), but they are also *higher-stakes*: a wrong title is permanently visible, a wrong author byline misattributes work. Diversity of candidates is exactly the failsafe against a single confidently-wrong generation.
- The marginal cost of K=4 over K=2 on a region with 20 tokens of input and maybe 50 tokens of output is small. The marginal benefit is the per-gap soft scorer having genuine choice.
- Doc-level frequency of gaps is low (most documents have 0–2 gaps). Total gap-fill generation cost is `K * n_gaps`, which is small even at K=8 in the offline lane.

**Sampling parameters.** Higher temperature than prose (e.g. 0.8 vs. 0.6) to encourage diversity in a smaller output surface; top-p 0.95.

### 3.2 Per-gap selection (Phase 9c hand-off)

**Recommendation: rule-based per-gap soft scorer in v1, NOT the per-region cross-encoder soft reranker.**

Per-gap scoring signals (concrete formulas):

```
score = w_axe * pass_per_region_hard_gate
      + w_html5 * pass_html5_validator
      + w_kind  * kind_specific_fit_score
      + w_len   * length_fit_score
      - w_diff  * diff_from_context_penalty
```

- `pass_per_region_hard_gate`: 1.0 if the candidate fragment passes axe + html5validator + (text-preservation if applicable to the kind), else 0.0. (See §3.4 below.)
- `kind_specific_fit_score`:
  - `missing_title`: keyword overlap with first paragraph (Jaccard, ≤ 1.0); penalty if length > 200 chars; bonus if no question mark.
  - `citation_unresolved`: 1.0 if the chosen anchor target exists in the doc index; 0.0 otherwise.
  - `author_block`: 1.0 if root element is `<address>`; 0.5 if `<p>`; 0.0 if anything else.
  - `copyright_block`: 1.0 if root element is `<aside>` or `<footer>` and contains `©` or "copyright"; lower otherwise.
- `length_fit_score`: piecewise — full credit when fragment length is within the expected band for the kind (titles 5–150 chars; author blocks 20–300 chars; copyright 30–250 chars); decays outside the band.
- `diff_from_context_penalty`: edit distance from the raw text the slot replaced, capped — heavily penalize hallucinated content not justified by context.

**Why rule-based, not the per-region soft reranker.** Per-region soft reranker (Phase 8) is trained on cross-encoder scoring of "fit to surrounding region". Gap-fill outputs are categorically different (titles, address blocks, citations) and the cross-encoder has not seen them in training. Rule-based scoring is calibratable to the small per-kind label set and avoids a trained model with no training data.

### 3.3 Adapter loading economy

The gap-fill adapter is **one** adapter that handles all five slot kinds via the prompt's `slot_kind` field. Only one adapter load happens, regardless of how many distinct kinds are gapped.

If the document has zero gaps, **Pass 4 is skipped entirely** — no adapter load, no Qwen invocation. This is the architecture's locked behavior (`architecture.md:280-281`).

If the document has 5 gaps of 3 kinds, the assembler:
1. Loads the gap-fill adapter (one-time cost ~5s including 4-bit re-quant).
2. Batches all 5 gaps into a single generation call (with the adapter still resident).
3. Each prompt encodes its `slot_kind`. The adapter has been trained on per-kind labels and produces appropriate output.
4. K candidates per gap → 5 × 4 = 20 fragments emitted.
5. Adapter is released (frees VRAM for the doc-level gate's axe-core context if needed).

This matches the serial-adapter discipline locked by `feedback_qwen_build_serial.md`.

### 3.4 Failure mode: gap-fill produces nothing valid

**Architecture.md is silent on whether gap-fill outputs go through the per-region hard gate.** Recommendation: **YES, gate them.** Reasons:

1. Gap-fill outputs are HTML fragments. The same axe + html5validator + text-preservation contract that Stage 7 enforces on prose/table/math regions applies. Skipping the gate would create an asymmetry where gap-fill outputs are trusted on shape but other Qwen outputs are not.
2. The gate is cheap (~200ms/fragment in the dev-machine target).
3. A gate failure on all K candidates is a hard failure mode the assembler must handle. Without the gate, hard failures slip through to the doc-level gate, where the failure manifests as "the whole doc fails" with a much harder root-cause analysis.

**When all K candidates fail the per-region hard gate.** Concrete fallback ladder:

| Slot kind | Deterministic fallback when all K fail |
|---|---|
| `missing_title` | Use first `<h1>` text; if no `<h1>`, use `"Untitled document"` (matches v1 `ontology_map.py:46`). |
| `citation_unresolved` | Leave the original plain text. No splice. |
| `author_block` | Wrap raw text in `<p>` (no `<address>`). Visible content preserved; semantics weaker. |
| `copyright_block` | Wrap raw text in `<aside class="copyright">`. |
| `legal_disclaimer` | Wrap raw text in `<aside class="legal">`. |

Gap-fill failure on `missing_title` does NOT cause the doc to fail the doc-level gate (because `"Untitled document"` is still non-empty `<title>`). It does cost the doc a "ship-with-confidence" stamp (the confidence reporter records `missing_title_fallback=true`). Phase 11 may demote this doc among multiple assemblies if any are available.

---

## Section 4 — Phase 9c: Deterministic merge-back

### 4.1 Splice semantics per slot kind

| Slot kind | DOM position | Wrapping element |
|---|---|---|
| `missing_title` | `<head><title>...</title></head>` AND, if no `<h1>` exists, also `<body><main><h1>...</h1>...</main></body>` (FIRST element under `<main>`, before any other body content). The `<h1>` and `<title>` text may differ if gap-fill chose differently for each (rare — usually the same text). | `<title>` and `<h1>` (no class, no role). |
| `citation_unresolved` | In-place replacement of the matched plain-text span (e.g., "Section 4.2") with the gap-fill's chosen `<a href="#sec-4-2">` (or whatever target gap-fill confirmed). The preceding/following text is preserved unchanged. | `<a>` (only). |
| `author_block` | In-place replacement of the original region's wrapper. The region's content was previously wrapped as `<p>` by the prose top-1; the splice replaces that `<p>` with the gap-fill `<address>`. | `<address>` (root); inner content per gap-fill. |
| `copyright_block` | In-place replacement, but the slot is also wrapped in the document's body-scope `<footer>` if it isn't already inside one. (If multiple `legal`/`copyright` slots exist, they all collect into a single `<footer>` at the end of `<body>`.) | `<aside>` inside `<footer>`. |
| `legal_disclaimer` | Same as copyright. | `<aside>` (inside `<footer>` if a footer exists; otherwise inline at original position). |

**Key choice — `<address>` for author byline, not `<aside>`.** The HTML5 living standard reserves `<address>` for contact information of the article's author. This is exactly the author-byline use case. Using `<aside>` would mark author info as tangential; that is wrong (author attribution is core to scholarly content).

**Cross-references — replace plain text, not edit existing `<a>`.** The pre-9c document has `<a>` tags only for explicit href-bearing source spans (e.g., URLs in body text). Cross-reference resolution always operates on plain text matches. There is no need to *edit* an existing `<a>`; we *create* a new `<a>` wrapper around the matched span.

### 4.2 Re-running 9a checks

**The trigger condition.** Re-run the named 9a sub-task if and only if its inputs changed:

| 9a sub-task | Re-run when |
|---|---|
| Heading hierarchy normalization (§1.1) | A `missing_title` gap-fill produced a new `<h1>` AND it inserts at the front of the document. (The new H1 forces a re-walk because heading levels downstream may now legally be one deeper.) |
| Reference resolution (§1.4) | A `missing_title` gap-fill produced a new heading (because section anchors built from headings may now have more entries). OR a `citation_unresolved` gap-fill changed the document text (rare — it should only change anchors, not text). |
| Doc shell (§1.5) | Title slot was filled — `<title>` element value changes. (This is just a string substitution; full re-run not needed.) |
| ARIA landmarks (§1.2) | A `legal`/`copyright` gap-fill was added — the body-scope `<footer>` may need to be rewired around the new aside. |

Other 9a sub-tasks (list continuation, reading-order DOM placement) are **not** re-run because gap-fill outputs do not change list grouping or document-order skeleton.

### 4.3 Iteration vs. one-shot

**Recommendation: one-shot re-run (a single iteration), not iterate-until-stable.**

Reasons:
1. The re-run inputs that gap-fill can produce are bounded — gap-fill is narrow-scope (no recursive document generation). A `missing_title` fill can cause a re-run of heading hierarchy and reference resolution, but the re-run cannot itself produce new gaps because the gap-set is already determined by the first 9a pass.
2. A theoretical cycle — gap-fill produces new cross-references that match new gaps — is structurally ruled out: gap-fill for `missing_title` produces an `<h1>`/`<title>` only, not body text; `citation_unresolved` produces an `<a>` wrapper only; `author_block` / `copyright_block` / `legal_disclaimer` produce wrapper elements around existing text. None of these produce *new* unresolved cross-references that didn't exist before.
3. If gap-fill ever did somehow produce new unresolved cross-references, the safer behavior is "splice, leave the new dangling refs as plain text, ship doc-level gate" rather than re-enter gap-fill recursively.

**Hardening.** Add an assertion in 9c: after merge-back, any new `Section X.Y` or `[N]` patterns introduced by gap-fill must resolve against the post-9c index OR fall through as plain text. Never re-enter Phase 9b.

---

## Section 5 — Document-level hard gate (Stage 10)

### 5.1 Inputs

A single string of HTML representing one complete assembled document. (When multiple candidate assemblies exist — see §6 — each is gated independently.)

### 5.2 Concrete checks

**Eliminating-only.** Any single check fails → drop. No soft trade-offs. (Architecture §5.1.)

| Check | Pass criterion | Notes |
|---|---|---|
| **axe-core full doc** | Zero `serious` AND zero `critical` violations under `wcag22aa` ruleset. | Reuse `dart_semantic/validate.py:HtmlValidator` exactly as-is. The fragment-vs-doc difference is just the input shape (full doc vs. fragment-in-shell); the validator handles both. Tags accepted: `["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]`. |
| **html5validator** | Zero errors. **Warnings allowed.** | html5validator is conservative; rejecting on warnings would over-fire on legitimately valid HTML (e.g., `<section>` without explicit heading). Errors are the bar. |
| **Heading-tree valid** | (a) Exactly one `<h1>`. (b) No skipped levels (`<h2>` → `<h4>` is invalid). (c) First heading in document is `<h1>`. | Reuse algorithm from `/home/user/Projects/Semantic/dart_semantic/ir.py:234-263`. Implement as `validate_heading_tree(html: str) -> bool`. The legacy `emit_html.py:60-78` raises on `<h2>` → `<h4>` skip with `EmitterError("Heading level skipped: …")`. The doc-level gate translates that raise into a "drop". |
| **`<html lang>` declared** | `<html>` element has a non-empty `lang` attribute. **No BCP-47 syntactic check** (per `docs/ontology.md` §4: BCP-47 is an open registry; whitelist-validation rejects legitimate tags). | Just non-empty. |
| **`<title>` non-empty** | `<head><title>...</title></head>` exists and has non-whitespace text. | WCAG SC 2.4.2. |
| **`<main>` landmark present** | Document contains exactly one `<main>` element wrapping primary content. | Architecture §5 names "landmarks present" — `<main>` is the only required landmark. `<nav>`/`<aside>`/`<footer>` are NOT required by the gate (they are emitted only when Semantic predicted them; absence is a feature, not a fail). |

### 5.3 Eliminating semantics — proof obligation

The gate is implemented as a sequence of `pass_X(html) -> bool` checks; the gate result is `all(pass_X)`. There is no scoring, no thresholding within a single check (within axe-core, "zero serious + critical" is a 0/1 decision; minor + moderate are NOT counted toward the gate, matching `dart_semantic/validate.py:32-35`'s `REJECT_IMPACTS = {"serious", "critical"}`).

**Enforcement.** A unit test fixtures a doc with one `serious` axe violation and one excellent heading tree; gate result must be `False`. Another fixture with zero axe violations and a skipped heading level; gate result must also be `False`. (Failures must be independent; mixing must not pass.)

---

## Section 6 — Document-level soft reranker (Stage 11)

This is the most under-specified part of the architecture. Investigate concretely.

### 6.1 When does the soft reranker fire?

**Three plausible sources of multiple candidate assemblies:**

1. **Per-region top-2 → 2-doc fan-out.** Phase 8's per-region soft reranker emits top-2 instead of top-1 when the top-2 score gap is within ε. With M ambiguous regions, this gives up to 2^M doc variants.
2. **K-fold gap-fill candidates → K^M doc variants.** With M gaps and K candidates each, the cross product is K^M doc variants.
3. **Offline-Qwen lane re-runs.** When the fast lane fails the doc gate, the offline lane (larger Qwen / larger K) regenerates everything; both fast and offline outputs may now be available and need ranking.

**Recommendation: only source (3) is supported in v1.**

Reasoning against (1) and (2):

- **(1) is a combinatorial trap.** Even with ε small, M=20 regions with 2 candidates each = 1M variants. The soft reranker is rule-based (DP-10.1) and cheap, but the assembler's gap-detection + 9a re-run cost is non-trivial; running it 1M times per doc is infeasible. Capping fan-out at the top 5 regions by ambiguity score caps the variant count at 32, which is more tractable but still 32× the cost.
- **(2) is the same trap with a different shape.** Most docs have 0–2 gaps; K^M = K^2 = 16–64 variants. Closer to feasible than (1), but still an unbounded growth pattern as gap-rich docs (forms, contracts, multi-author papers) push M up.
- **(3) is naturally bounded.** At most one fast-lane assembly + one offline-lane assembly = 2 candidates. The soft reranker picks between them. Always cheap.

**Concrete v1 behavior:**
- Phase 8 emits exactly top-1 per region (not top-2). The soft reranker fires only when the offline-Qwen lane runs.
- When only the fast lane runs and passes the gate: ship that doc; soft reranker is a no-op (one candidate trivially wins).
- When the fast lane fails the gate and the offline lane runs and passes the gate: the soft reranker chooses between (a) the offline-lane output and (b) the fast-lane output IF the fast-lane output passed at least 5 of the 6 checks (a partial-pass that can be diffed against). In practice this often degenerates to "ship the offline lane" because the fast-lane output is genuinely failing.

This is conservative. It defers the combinatorial fan-out problem to v2 when more eval signal exists.

### 6.2 Score axes (rule-based)

Per Plan DP-10.1, start rule-based. Concrete formulas:

| Axis | Formula | Range |
|---|---|---|
| **heading_tree_balance** | `1 - normalized_variance(subtree_depths)` where `subtree_depths` is the list of depths from each `<h1>` down to its leaves under each top-level section. Even balance scores 1.0; pathological "all H6 under one H1" scores low. | [0, 1] |
| **landmark_coverage** | `n_landmarks_present / n_landmarks_expected`. Expected = `<main>` (always) + `<nav>` if doc has ≥3 TOC-pattern lines + `<footer>` if Semantic predicted any `legal`/`copyright`. | [0, 1] |
| **ref_link_integrity** | `n_resolving_anchors / max(1, n_anchors_emitted)`. An anchor `<a href="#X">` "resolves" when `X` is the `id` of a real element in the doc. Dangling anchors penalize. | [0, 1] |
| **outline_cleanliness** | Composite: `0.5 * (1 - empty_heading_rate) + 0.3 * (1 - duplicate_h1_rate) + 0.2 * (1 - orphan_caption_rate)`. Empty headings = `<h1></h1>`; duplicate H1 = >1 `<h1>` (always 0 if §5.2 passed); orphan caption = `<figcaption>` outside `<figure>`. | [0, 1] |

**Composite score.** `final = 0.30 * heading_tree_balance + 0.20 * landmark_coverage + 0.30 * ref_link_integrity + 0.20 * outline_cleanliness`. Weights are placeholders; tune on a 30-doc held-out set in Phase 10.

**Tie-break.** Smaller diff-from-source by character count wins. (Same tie-break rule as Phase 8 per the v1 plan, applied at doc level.)

### 6.3 Promote to learned cross-encoder?

Plan DP-10.1: only if rule-based shows poor calibration (inversion vs. final axe outcome on > 20% of cases). Defer the decision to Phase 10 measurement. Rule-based first.

---

## Section 7 — Exits (Stage 12)

### 7.1 ship-with-confidence

**Concrete output.**
- The assembled HTML, byte-for-byte as Stage 11 selected.
- A sidecar JSON `{document_id, exit: "ship-with-confidence", confidence: <float 0..1>, gate_results: {...}, soft_reranker_score: <float>}`.
- The HTML itself contains a machine-readable certification meta tag:
  `<meta name="dart-certification-status" content="certified">`.

**Confidence number.** Aggregate of:
- Mean per-region soft reranker score (Phase 8) → 0.40 weight.
- Doc-level soft reranker composite (§6.2) → 0.40 weight.
- Fraction of regions whose top-1 cross-BERT-reranker confidence exceeded τ_landmark → 0.20 weight.

This is a heuristic confidence, not a calibrated probability. Document it as such.

### 7.2 offline-Qwen lane

**Per Plan Q7 — pick the most plausible 8GB-feasible implementation.**

Recommendation: **same Qwen 3 4B base + same adapters, but K=8 candidates with looser sampling (temp 0.9 vs. 0.6, top-p 0.95)**, NOT a model upgrade.

Reasoning:
- Qwen 3 7B 4-bit is borderline-fits in 7.0 GB headroom; with axe-core's Chromium context concurrent, it WILL OOM. Documented in `feedback_qwen_build_serial.md`-adjacent constraints.
- The diversity gain from K=8 vs. K=4 is substantial; from temp 0.9 vs. 0.6 is also substantial; from a 4B → 7B upgrade is real but the cost is "doesn't run".
- A genuinely-larger model belongs in a "v2 cloud lane" — out of scope for v1 because of the no-external-LLM-runtime memory.

**Implementation.**
- `dart_semantic/qwen_specialists/offline_lane.py` accepts the failing-fast-lane document state (BERT outputs cached; per-region candidates re-generated).
- Re-runs Stages 6–9 with K=8 and temp 0.9 over only the regions whose fast-lane gate failed (BERT outputs are unchanged, so the council does not re-run; the cross-BERT reranker does not re-run; only the per-region Qwen specialists + per-region gate + soft reranker + assembler re-run).
- One re-entry only. If the offline lane also fails the doc gate, fall to non-certified stamp.

### 7.3 non-certified stamp

**Two parts: machine-readable and human-visible.**

**Machine-readable** (always):
- `<meta name="dart-certification-status" content="not-certified">` in `<head>`.
- A sidecar JSON `{document_id, exit: "non-certified", failed_checks: [...], confidence: 0.0}`.

**Human-visible** (always):
- A `<aside role="note" class="dart-uncertified-banner">` is inserted as the first child of `<main>`, before any other content:

  ```html
  <aside role="note" class="dart-uncertified-banner" aria-label="Accessibility certification status">
    <p><strong>Accessibility notice:</strong> This document failed automated WCAG 2.2 AA validation and is <em>not certified accessible</em>. Some content may not be readable by assistive technology. Contact the document publisher for an alternative format.</p>
  </aside>
  ```

**Why visible, not just machine-readable.** A consumer with no DART-aware tooling sees nothing if the stamp is metadata-only. WCAG conformance is a product claim; failure to meet it must be communicated to the actual reader. The `<aside role="note">` placement at the top of `<main>` is screen-reader-discoverable and visually obvious.

**Why `<aside role="note">` and not just text.** `role="note"` is the ARIA equivalent for parenthetical/marginal content (per ARIA 1.2 §5.5). It is explicitly distinguished from main content but still announced. This matches the user-facing semantics: "this is information about the document, not part of the document."

---

## Section 8 — Reuse from existing v1 code

### 8.1 `dart_semantic/ontology_map.py` — the legacy Stage 5 emitter

- **Salvageable:**
  - Doc shell construction (`emit_html()` lines 48-54): DOCTYPE, `<html lang>`, `<head>`, `<title>`, `<body>`, `<main>` boilerplate. Lift wholesale into the new `assembler/shell.py`.
  - Heading hierarchy demote-forward logic (lines 60-100): the algorithm itself ports cleanly, just with new I/O.
  - Table assembly (`_emit_table`, lines 240-308): the row-clustering and `<thead>`/`<tbody>`/`<th scope>` emission logic is reusable as a per-region table emitter. Note this is for the case where the Qwen table specialist doesn't already provide the full HTML; if it does, use the specialist's output directly.
  - List nesting (`_emit_list`, lines 193-237): the depth-based nesting algorithm is reusable.
  - HTML-escape helpers (`_text`, `_attr`, lines 347-352): direct reuse.
- **Deprecated:**
  - The whole role-dispatch loop (lines 60-172): replaced by region-stream walk where each region carries its own pre-emitted HTML from Phase 8 top-1. The assembler does not re-emit per-region HTML; it composes.
  - Skip-link absence: the v1 emitter doesn't emit a skip link (per `docs/ontology.md` §7 the responsibility was downstream). The new assembler adds one (§1.5).

### 8.2 `dart_semantic/hierarchy.py` — Stage 4 hierarchy resolver

- **Salvageable:**
  - `_resolve_heading_depths` (lines 74-104): font-size-stack heading depth algorithm. Useful as a *fallback* when BERT-Structure heading-level prediction is low-confidence (< 0.50). When confidence is high, use the BERT prediction directly.
  - `_resolve_list_groups` (lines 109-139): contiguous-LIST_ITEM grouping with indent_bucket-based depth. Reusable for within-a-region list grouping; needs extension for across-region continuation (§1.3).
  - `_resolve_table_groups` (lines 144-189): row clustering by y-overlap. Reusable as a fallback when BERT-TableSpecialist hasn't run yet or its output is incomplete.
- **Deprecated:**
  - The driver `resolve_hierarchy` (lines 38-69): the new assembler's Phase 9a is the driver. The hierarchy module becomes a library of pure helpers.

### 8.3 `dart_semantic/enrich.py` — Stage 6 content enrichment

- **Salvageable:**
  - `_lingua_detector` + `_detect_language` (lines 69-94): direct reuse for `<html lang>` detection in §1.5. This is the *only* part that survives.
  - `summarize_table` (lines 110-126): deterministic short table summary for `aria-describedby`. Useful in the assembler's `<caption>`-fallback path, but only when the table specialist Qwen produced no caption.
- **Deprecated:**
  - `enrich()` driver and the per-block iteration (lines 42-64): replaced by the Phase 9a sub-task pipeline.
  - `alt_text_for_image` and `describe_figure_extended` (lines 99-136): NotImplementedError stubs. The architecture explicitly excludes image alt-text from the gap-fill specialist's v1 scope (`architecture.md:502`); these stay as stubs and inform a v2 effort.
  - Slot-detector logic for FIGURE / TABLE: replaced by BERT-Semantic + the Qwen table/figure paths.

**Worth promoting from `enrich.py` into gap-detection (§2):** the language-detection cascade is the model. Other "slots" enrich.py recognizes (figure/table) are already handled by the Qwen prose/table/math specialists upstream.

### 8.4 `dart_semantic/escalate.py` — v1 tri-verdict router

- **Mapping to the new exits:**

| v1 verdict | New exit |
|---|---|
| `ship` | `ship-with-confidence` |
| `llm_fallback` | `offline-Qwen lane` (NB: v1's "frontier LLM" intent → v2 offline-LOCAL Qwen; the routing behavior is the same shape) |
| `fail` | `non-certified-stamp` |

- **Salvageable:**
  - The threshold-based reasoning structure (`should_escalate`, lines 69-110). The shape — collect reasons, decide verdict — is the right shape.
  - The tunable thresholds at module level (lines 56-66): same idea applies to the new exits' thresholds (gate-pass count, soft-reranker score floor, gap-fill failure count).
- **Deprecated:**
  - `mean_classification_confidence` and `qwen_override_rate` as inputs (lines 38-47): these were Stage-3-DistilBERT-specific and Stage-3b-Qwen-specific signals. The new exits use gate-pass and soft-reranker signals.
  - The verdict strings change (`ship` / `llm_fallback` / `fail` → `ship-with-confidence` / `offline-Qwen` / `non-certified`) but the structure is one-to-one.

**Recommendation:** rewrite `escalate.py` into `dart_semantic/exits.py` (per Plan §1.2), preserving the threshold-based reasoning skeleton and the dataclass-as-decision-record pattern.

### 8.5 `dart_semantic/validate.py` — axe-core validator

- **Salvageable: 100%.** The `HtmlValidator` class (lines 76-142) is the validator for **both** the per-region hard gate (Stage 7) and the doc-level hard gate (Stage 10). Same axe ruleset (`ACCEPT_TAGS`, lines 32). Same reject impacts (`REJECT_IMPACTS = {"serious", "critical"}`, line 35).
- **Reuse pattern:** in Stage 7 and Stage 10 alike, use `HtmlValidator` as a context manager around a batch of fragments / docs; spin up Chromium once per batch.
- **Plan §2.1 already names this:** "validate.py — Extend. HtmlValidator is reused by Phase 7 (per-region) and Phase 10 (per-document). The class itself does not change."

### 8.6 `dart_semantic/emit_html.py` — legacy IR-tree emitter

- **Salvageable:**
  - `_emit_table` (lines 150-178): full `<thead>`/`<tbody>`/`<th scope>` table emission with caption + colspan + rowspan. The most polished table emitter in v1; lift directly when the table-specialist Qwen output needs a deterministic fallback.
  - `_emit_form` (lines 220-236) and `_emit_field` (lines 239-286): full form emission with `<fieldset>`/`<legend>`/`<label for>`/required+aria-required/`<select>`/`<textarea>`/checkbox-radio groups + min-target-size inline style. Forms are out-of-scope for the v1 Qwen specialists (the architecture's four adapters are prose/table/math/gap-fill — no `form` adapter), so when the assembler encounters a form region it emits via this exact code path. This is a substantial piece of working WCAG-compliant emission.
  - `_emit_figure` (lines 196-206): `<figure><img alt><figcaption>` with mandatory alt-text. Lift as-is.
  - `_emit_blockquote`, `_emit_codeblock`, `_emit_definition_list` (lines 209-217, 138-147): one-line emitters. Reuse direct.
  - `_emit_runs` (lines 289-302): inline run emission with `<strong>`/`<em>`/`<code>`/`<a>`. Reuse direct.
- **Note on file status:** `emit_html.py` is marked "[LEGACY]" in its module docstring, but it remains the only working WCAG emitter for IR trees. Per `docs/bloat_audit.md` §4.1 it is preserved because the data ingesters (`scripts/pair_from_*.py`) build `ir.Document` and emit through it for ground-truth HTML. The new assembler reuses individual `_emit_*` functions but does not import the top-level `emit()` driver.

---

## Section 9 — Open questions / decisions for the user

Each numbered question is one sentence; user can answer in one sentence. These are decisions where the architecture and plan are silent and the recommendation is non-obvious.

1. **Should gap-fill outputs go through the per-region hard gate before reaching Phase 9c?** *(Investigation recommends YES; architecture is silent.)*

2. **Should Phase 9a iterate until stable when gap-fill changes the title or refs, or one-shot re-run only?** *(Investigation recommends one-shot.)*

3. **Should `<aside>` be emitted by the assembler in v1, given v1's IR has no `ir.Aside` and `<aside>` carries real ARIA navigation cost?** *(Investigation recommends emitting `<aside>` ONLY for `legal`/`copyright` and the `non-certified-stamp` banner; never for footnotes or sidebars in v1.)*

4. **Phase 11 soft reranker fan-out — is "v1 supports source (3) only" acceptable, deferring per-region top-2 fan-out and gap-fill cross-product to v2?** *(Investigation recommends yes — the combinatorial trap is real.)*

5. **Does `<header>` / `<footer>` at body scope belong to the Semantic emitter (`<footer>` wraps a `legal`/`copyright` slot), or is it a downstream-DART responsibility per `docs/ontology.md` §7?** *(Investigation assumes assembler owns it; the alternative — leave to DART downstream — would simplify the assembler but require a coordinated DART change.)*

6. **For the non-certified stamp, is the user-visible `<aside role="note">` banner inside `<main>` acceptable, or should it live outside `<main>` (e.g., as the first child of `<body>` before `<main>`)?** *(Investigation recommends inside `<main>` so it's covered by the skip-link; placing it outside risks AT skipping the warning.)*

7. **Soft-reranker rule-based weights (heading_tree_balance 0.30, landmark_coverage 0.20, ref_link_integrity 0.30, outline_cleanliness 0.20) — accept defaults pending Phase 10 tuning?** *(Investigation recommends accepting; numbers are placeholders.)*

8. **Confidence formula — is the heuristic (0.40·per-region soft + 0.40·doc soft + 0.20·cross-BERT) acceptable, or should we wait until calibration data exists before reporting any confidence number at all?** *(Investigation recommends shipping the heuristic with explicit "uncalibrated" labeling in the sidecar JSON.)*

9. **Should the `<title>` text and a synthesized `<h1>` text always match when both are gap-filled, or are they independent generations?** *(Investigation recommends matching: ask gap-fill once for "doc title", emit it both as `<title>` and `<h1>`. Independent generations introduce a needless inconsistency.)*

10. **Citation/footnote resolution — when the gap-fill specialist confirms "leave as plain text" (no clean target), should we emit any indication (e.g., a `<span class="unresolved-ref">`) or strictly no markup change?** *(Investigation recommends strictly no markup change — the WCAG bar doesn't require it and the visible cue would be noise. Logging the unresolved ref to the sidecar JSON suffices.)*

11. **Author block — `<address>` per HTML Living Standard's intent, or `<header>` (which can also carry author info on `<article>`-scoped content)?** *(Investigation recommends `<address>` for the v1 case where the document is a single article; `<header>` is scoped to `<article>` and v1 doesn't emit `<article>` wrappers.)*

12. **Offline-Qwen lane behavior — same-base + larger-K-and-temp confirmed (per Plan Q7) for v1, or is there a stronger candidate?** *(Investigation confirms the Plan Q7 default; no stronger candidate fits 8 GB with axe-core concurrent.)*

---

## Section 10 — Implications for Phase 0 type definitions

Phase 0 lands `dart_semantic/assembler/types.py`. The minimum viable type set is below; field names and types are specified concretely so Phase 0 can land them without re-investigating.

### 10.1 `GapKind` enum

```python
from enum import Enum

class GapKind(str, Enum):
    MISSING_TITLE       = "missing_title"
    CITATION_UNRESOLVED = "citation_unresolved"
    AUTHOR_BLOCK        = "author_block"
    COPYRIGHT_BLOCK     = "copyright_block"
    LEGAL_DISCLAIMER    = "legal_disclaimer"
```

### 10.2 `GapSlot` dataclass

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class GapSlot:
    kind: GapKind
    region_index: int                      # -1 for whole-doc slots (e.g. missing_title)
    context: dict[str, Any]                # per-kind shape; see §2 table
    detected_by: str                       # "9a:title_ladder" | "9a:reference_resolution" | "9a:semantic_landmark" | ...
    semantic_top1: str | None = None       # the BERT-Semantic doc-role that triggered this (if any)
    semantic_confidence: float | None = None
    fallback_html: str | None = None       # the deterministic fallback to splice if all K gap-fill candidates fail the per-region gate
```

The `context` dict is intentionally typed-loosely; per-kind concrete shape is documented in §2's table. Validation occurs at gap-fill prompt-construction time.

### 10.3 `CertificationStatus` enum

```python
class CertificationStatus(str, Enum):
    SHIP_WITH_CONFIDENCE = "ship-with-confidence"
    OFFLINE_QWEN_LANE    = "offline-qwen-lane"
    NON_CERTIFIED        = "non-certified"
```

### 10.4 `AssembledDoc` dataclass

```python
@dataclass
class AssembledDoc:
    html: str                              # the full doc HTML, post-9c (or post-9a if no gaps)
    gaps_found: list[GapSlot]              # what 9a flagged; empty list if 9a clean
    gaps_resolved: list[GapSlot]           # what 9c successfully filled (subset of gaps_found)
    gaps_fallback: list[GapSlot]           # what 9c filled with deterministic fallbacks
    heading_tree: list[tuple[int, str]]    # (level, text) in document order; for §6.2 scoring
    landmarks: dict[str, int]              # {"main": 1, "nav": 1, "footer": 0, "aside": 2, ...}
    anchors: dict[str, str]                # {"sec-3-2": "<h2 id='sec-3-2'>...</h2>", ...} for §6.2 ref-link-integrity
    region_provenance: list[int]           # index back into Phase 8 top-1 regions for each emitted block
    sub_task_log: dict[str, str]           # {"heading_normalize": "demoted 2 levels", "list_continue": "merged 3 lists", ...}
```

### 10.5 `DocCandidate` dataclass (for Phase 11 multi-assembly)

```python
@dataclass
class DocCandidate:
    assembled: AssembledDoc
    source: str                            # "fast-lane" | "offline-lane"
    gate_result: "GateResult"              # the doc-level hard gate verdict; types live in dart_semantic/gates/types.py
    soft_score: float | None = None        # composite score from §6.2; None until ranker runs
    soft_axes: dict[str, float] | None = None  # the per-axis breakdown; for diagnostics
    confidence: float | None = None        # final §7.1 confidence (set only when ranker has chosen)
```

The `GateResult` type is owned by the `gates` package (Plan Phase 0), not the `assembler` package. The forward-string annotation avoids a circular import at definition time.

### 10.6 `ExitDecision` dataclass

```python
@dataclass
class ExitDecision:
    status: CertificationStatus
    chosen: DocCandidate                   # the DocCandidate selected to ship (with stamp if non-certified)
    fast_lane_attempted: bool
    offline_lane_attempted: bool
    reasons: list[str] = field(default_factory=list)  # why this status, not the next-better one
    sidecar_json: dict[str, Any] = field(default_factory=dict)  # per-§7 schema
```

### 10.7 What Phase 0 does NOT need to land

- Concrete implementations of any of the sub-tasks (§1.1–§1.6).
- The gap-fill prompt schema (lives in `qwen_specialists/gap_fill.py`, Phase 6c).
- The doc-level hard-gate checks (live in `gates/hard_document.py`, Phase 10).
- The soft-reranker scoring functions (live in `gates/soft_document.py`, Phase 10).

Phase 0's scope is the type contract only. Implementations land in Phases 9 + 10.

---

## Cross-references

- **Architecture:** `/home/user/Projects/Semantic/architecture.md` §2 (pipeline), §4 (Qwen specialists), §5 (gates), §6 (exits), §10 (locked choices).
- **Plan:** `/home/user/Projects/Semantic/Plans/01_implementation_plan.md` Phase 9 (assembler), Phase 10 (gate + reranker + exits), Phase 6c (gap-fill).
- **Ontology:** `/home/user/Projects/Semantic/docs/ontology.md` §2 (per-component standards), §3 (PDF/UA → HTML5 map), §4 (under-specified choices), §5 (positive non-inclusions), §7 (gap analysis).
- **v1 reuse anchors:**
  - `/home/user/Projects/Semantic/dart_semantic/ontology_map.py:35-178` (legacy emitter; doc shell + heading + table + list logic).
  - `/home/user/Projects/Semantic/dart_semantic/hierarchy.py:74-189` (heading-stack, list-grouping, table-clustering algorithms).
  - `/home/user/Projects/Semantic/dart_semantic/enrich.py:69-94` (lingua language detection).
  - `/home/user/Projects/Semantic/dart_semantic/escalate.py:35-110` (tri-verdict shape and threshold module-level pattern).
  - `/home/user/Projects/Semantic/dart_semantic/validate.py:76-142` (HtmlValidator — direct reuse in both gates).
  - `/home/user/Projects/Semantic/dart_semantic/emit_html.py:150-286` (table, form, figure, list, code, blockquote emitters — direct reuse).
  - `/home/user/Projects/Semantic/dart_semantic/ir.py:234-263` (`normalize_heading_levels`).

*End of investigation.*
