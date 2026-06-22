# gap_fill V2 — light up the deferred GapKinds (lead: `citation_unresolved`)

> **STATUS (banner added 2026-06-17): FOLDED INTO Plan 12.** gap_fill V2 GapKinds are tracked
> as a sub-item of Plan 12 (§D); see Plan 12 for current status. Historical detail retained.

**Plan version:** 2026-05-27 (rev 1)
**Branch:** `main`
**Status:** PLANNED. Execution/sequencing plan — the *design* already exists
in [Plans/04](04_assembler_layer_investigation.md) (gap-detection table,
context shapes, the `slot_kind` prompt scaffold, per-kind scorers). V1 shipped
`missing_title` + `author_block` only; this turns on the three GapKinds that
Plans/04 designed but deferred. **Dataset build is CPU (do anytime); gap_fill
retrain is the GPU step (serial — behind the table-eval + `structure_v2`
queue).**

Cross-links: realizes the V2 scope deferred in
`data/qwen_gap_fill_dataset/coverage_report.json` ("notes.v1_scope") and the
gap-detection table in Plans/04. Reuses the same license-cleared pair sources
(`data/pairs/{arxiv,pmc,courtlistener,gutenberg,...}`) already paired for the
other adapters.

---

## 0. Current state (V1)

`GapKind` enum (`dart_semantic/assembler/types.py:13`) defines **5** kinds;
`pass_9a` detects **2**:

| GapKind | detect (9a) | dataset | splice (9c) |
|---|---|---|---|
| `missing_title` | ✅ | ✅ | ✅ |
| `author_block` | ✅ | ✅ | ✅ `_splice_author_block` |
| `citation_unresolved` | ⬜ | ⬜ | ⬜ |
| `copyright_block` | ⬜ | ⬜ | ⬜ |
| `legal_disclaimer` | ⬜ | ⬜ | ⬜ |

Dataset: 4,426 train / 610 val / 548 test (5 sources). **Volume is adequate —
not expanding existing kinds** (user decision 2026-05-27). The gap is *kind
coverage*, not row count. Note the V1 kind imbalance for context: `missing_title`
4,338 vs `author_block` 1,246 (`no_author` skipped 3,435 — Wikipedia/Gutenberg/
CourtListener lack ar5iv author markup).

## 1. Scope & sequencing (recommended)

Each new kind is a **4-part change**, not just data — the dataset rows are the
easy part:

1. **detect** — add the trigger to `pass_9a` (Plans/04's gap-detection table
   gives the exact rule + context shape per kind).
2. **extract** — `data/build_gap_fill_qwen_data.py` emits `(slot, target)` rows
   from the cleared pair sources (make the gap, keep the truth as target).
3. **prompt** — `dart_semantic/qwen_specialists/prompts.py:build_gap_fill_request`
   already carries a `slot_kind` field (Plans/04 prompt scaffold); confirm the
   SYSTEM line covers the new kind.
4. **splice** — `dart_semantic/assembler/pass_9c.py` needs a `_splice_<kind>`
   to merge the generated fragment back (V1 has `_splice_author_block`).

**Order:**

1. **`citation_unresolved`** ⭐ *lead* — highest value-to-effort.
2. **`copyright_block`** + **`legal_disclaimer`** — companions; share the
   legal/boilerplate sourcing (CourtListener, Gutenberg, gov) and the same
   `BERT-Semantic top-1 == legal` detection gate (copyright vs disclaimer split
   by a `©|copyright|all rights reserved` sub-flag — Plans/04).

## 2. Why `citation_unresolved` first

- **Ubiquitous** — inline `[N]` / "Section X.Y" / "Figure N" refs are in nearly
  every academic doc; a broken ref→target link is a real, common defect.
- **Clean ground truth, no new sourcing** — ar5iv (`<a href="#bib...">`) and
  PMC JATS (`<xref ref-type=...>`) carry the inline-cite→reference links
  natively. Make the gap by *stripping the anchor* around an inline cite;
  the **target is the restored `<a href="#anchor">[N]</a>`**. Same
  license-cleared arXiv/PMC pairs we already use (no arXiv-default-license
  exposure — these are the CC-cleared subset).
- **Short targets** — a single restored anchor; well within `max_len=512`.
- **WCAG win** — SC 1.3.1 (info & relationships) + 2.4.4 (link purpose).
- **Design already done** — Plans/04 specifies the detection trigger
  (pattern match + target-exists check + **near-miss gate**: only flag when a
  Levenshtein-≤2 / |ΔN|≤1 candidate exists, else the ref is genuinely dangling
  and gap-fill won't help) and the scorer (1.0 iff the chosen anchor target
  exists in the doc index).

## 3. Ground-truth sources per kind

| kind | sources | make-the-gap recipe | target |
|---|---|---|---|
| `citation_unresolved` | arxiv (ar5iv), pmc (JATS xref) | strip the `<a>`/`<xref>` wrapper around an inline cite; record the true anchor id | `<a href="#<anchor>">[N]</a>` (or "leave as plain text" when no valid target — the negative case) |
| `copyright_block` | gutenberg (rich © pages), courtlistener, gov | take a region whose text matches `©\|copyright\|all rights reserved`, present it unwrapped | `<aside>`/`<footer>` wrapped © block |
| `legal_disclaimer` | courtlistener (legal corpus), gov | legal/license/terms region, unwrapped | wrapped disclaimer block + `legal_subkind` |

**Balance discipline (mirror the table/structure work):** citation rows from
arXiv/PMC will be abundant; copyright/legal from Gutenberg/CourtListener thinner.
Cap the dominant kind (likely citation) so V2 doesn't swamp the V1 kinds or each
other — keep per-kind shares sane, the same downsample discipline applied in
Plans/06 §6 and the `structure_dataset_v2` merge. Include **negative/no-op rows**
for `citation_unresolved` (ref with no valid target → "leave as plain text") so
the adapter learns *not* to hallucinate anchors.

## 4. Acceptance gates (before retrain)

| gate | target |
|---|---|
| detection precision (9a) | high — must **not** over-flag valid text/refs; verify on a held-out doc set (Plans/04 §"do-NOT flag ambiguous refs": "see above"/"the foregoing" stay plain text) |
| per-kind scorer (Plans/04) | citation: anchor-target-exists rate; copyright/legal: root `∈{<aside>,<footer>}` + keyword present |
| schema | rows match the existing gap_fill builder contract; load through `data_loader` survey |
| target length | p99 within `max_len=512` (V1 targets are tiny; keep it so) |
| license | citation from the CC-cleared arXiv/PMC subset; copyright/legal from PD/CC sources only — **no arXiv-default, no CC-BY-SA** |
| end-to-end | `pass_9c` splices each kind without breaking fragment well-formedness or the axe gate |

## 5. FOLLOW-UP — build + retrain

**Step 1 — detection + extraction + splice (CPU, code).** Implement the 4-part
plumbing for `citation_unresolved` first; rebuild
`data/qwen_gap_fill_dataset` (new version dir, keep the current as `.bak` like
the existing `arxiv_only`/`multi_v1` backups). Then add copyright/legal.

**Step 2 — retrain gap_fill (GPU; serial).** The LAST of the 4 Qwen specialists
(prose ✅ / math ✅ / table ✅-pending-eval / gap_fill = this). Config already
in `train_config.py` (`max_len=512`, epochs 3, r=8, dropout 0.10) — **but its
"~1K rows" comment is stale (now 4,426+); re-check the overfit guard before
training**. Gated behind the table-eval and `structure_v2` retrain on the
serial GPU.

**Step 3 — eval.** Mirror the prose/table harness: id-stripped structural match
+ well-formedness + axe, **stratified by `slot_kind`** (so a strong
`missing_title` number can't mask a weak `citation_unresolved`), plus the
per-kind functional scorers (citation anchor-exists, copyright/legal wrapper).

## Do-NOT list
- Don't expand existing-kind row volume (4,426 is enough — user call); V2 is
  about *kind coverage*.
- Don't flag ambiguous/dangling refs ("see above", refs with no near-miss
  target) — leave plain text (Plans/04).
- License: citation only from CC-cleared arXiv/PMC; copyright/legal only from
  PD/CC sources. No arXiv-default, no CC-BY-SA.
- No silent fallbacks — unknown slot shape raises; a no-valid-target citation
  emits the explicit "leave as plain text" target, not a guessed anchor.
- Serial GPU only for the retrain (behind table-eval + `structure_v2`).
