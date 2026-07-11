# Technical Debt Register

**Purpose.** A single, living list of known technical debt so it stays visible and
prioritized instead of accumulating silently. This is the durable home for debt
items (task lists are ephemeral; this survives sessions).

**Maintenance protocol.**
- **Add** an item the moment debt is found or *created* — including debt you
  knowingly take on (log it here rather than leaving it silent).
- **Prevention first:** new work should land clean (scratch → gitignored dirs,
  migrations finish their naming, experimental flags carry a validate-or-remove
  intent, docs/counts stay in sync). Prevention beats cleanup.
- **Review** in the recurring hygiene audit (re-run the file-hygiene +
  dart-surface scans); promote/close items as they change.
- **Close** an item only when the debt is actually gone (not just planned).

**Legend.** Severity: 🔴 high · 🟠 med · 🟡 low. Effort: S (hours) · M (day) ·
L (days) · XL (multi-day/coordination). Status: `open` · `in-progress` ·
`parked` · `done`.

---

## Prioritized items

| # | Item | Category | Sev | Effort | Status | Pointer |
|---|------|----------|:---:|:------:|--------|---------|
| D1 | **DART naming purge** — migration removed the AGPL engine but left **799 tracked files / 8160 line-hits** of `dart` naming (21 `data-dart-*` attrs, `dart:` sourceIds, the 207-file `SemantiK/dart_semantic/` package, phase `dart_conversion`, agents, ~70 env tokens). Assessed 2026-07-11: stale branch (0 unique commits), 19 persisted courses + 230 checkpoints need migration (`migrate.py` is manifest-only → extend), `imscc_chunks/` rename is the blessed dual-read precedent. 5-stage no-stubs plan; blocked on owner Stage-0 naming ratification | naming/migration | 🟠 | XL | parked | `docs/dart-surface-inventory.md`, branch `refactor/dart-purge`, memory `project_dart_purge_assessment` |
| D2 | **Config / env-flag sprawl** — ~87 env flags, 5 dispatch paths, routing complexity; "one config + one gateway" never landed | config-sprawl | 🟠 | XL | open | memory `project_llm_management_redesign` |
| D3 | **`extract_cache` doesn't invalidate on fusion/extraction code change** — a code fix is silently masked by the cached extraction (cost real debug time 2026-07-11); cache key omits a code/version hash | correctness-footgun | 🟠 | M | open | `SemantiK/dart_semantic/extract_shared.py` `_compute_extract_cache_key` |
| D4 | **`SEMANTIK_VLM_ORDER_AUTHORITATIVE`** committed default-OFF with a *mis-designed trigger* (gates on divergence, which doesn't capture pure order-scramble) — dormant wrong code | code-quality | 🟠 | M | open | `SemantiK/dart_semantic/vlm_fusion.py` (commit `8cf0318f`) |
| D5 | **`vlm_extract` LLM call site lacks `DecisionCapture`** — violates the root CLAUDE.md LLM-call-site instrumentation contract (+ needs a docs/LICENSING.md seat row) | contract-gap | 🟠 | S–M | done | fixed 2026-07-11 commit `c0c5c70b` (per-live-page `structure_detection` capture, best-effort; LICENSING row already present) |
| D6 | **Fusion garbage-tails + phantom-chapter headings** — clean VLM block + appended tesseract garbage (`® -Iq\|…`); running page-headers minted as chapters | conversion-quality | 🟠 | M–L | done | fixed 2026-07-11 commit `2436e2e6` (garbage-tail filter + single-digit running-header regex; validated on full ch01 re-run) |
| D13 | **Chapter-title derivation loses the title word** — with the page-glued running-header region dropped (D6 fix), the doc `<h1>` falls back to bare "Chapter 1" instead of "Chapter 1 Foundations"; the title word survives elsewhere in the outline but `build_chapters_ir` doesn't recover it | conversion-quality | 🟡 | M | done | fixed 2026-07-11 (mines `metadata_drop` running-header text for a folio-stripped modal title, fallback-tier only; `lib/semantik/cascade_ir.py`) |
| D7 | **Scan multi-column reading-order** (broad) — the known-hard SemantiK structure problem; the number-fix resolved the *grid* case, but dense multi-column + heading hierarchy remain | conversion-quality | 🟠 | L | open | memory `plans-pointer-semantik-restructure` |
| D8 | **`runtime/` junk drawer** — Jun-8 scratch: `fix_*`/`complete_*` one-off scripts, ~25 `_res_`/`_fres_` result blobs, 11 stray logs, `qwen_test/` prototype (all gitignored → local disk only) | hygiene/scratch | 🟡 | S | open | `docs/file-audit-cleanup.md` |
| D9 | **Stray local logs / `.bak` / `.contaminated`** — `state/logs/*.contaminated`, `state/workflows/*.pre-remediation.bak`, 3× 0-byte root `.log` (gitignored) | hygiene/scratch | 🟡 | S | open | `docs/file-audit-cleanup.md` |
| D10 | **Doc/count drift risk** — the CLAUDE.md family carries many hand-maintained counts + flag tables (gate counts, flag indices) that can drift from code | docs | 🟡 | ongoing | open | `doc-sanitation-reviewer` agent, flag-doc-sync |
| D11 | **Test-bed representativeness** — small slices behave differently from full runs (a 3-page slice dropped content a 198-page run kept), so slice-based validation can mislead | process | 🟡 | ongoing | open | 2026-07-11 fusion-fix debugging |
| D12 | **Marketable-v2 open items** — LTI fork + pilot decisions, demo build un-run, chunker 10-vs-21 drift | product-open-items | 🟡 | var | open | memory `project_marketable_v2_implemented` |
| D14 | **Council BERT `merge_or_split` CUDA-OOMs under a vLLM GPU pin** — when Omni (vLLM) owns the whole card, the in-process ModernBERT structure-council `merge_or_split` step raises `CUDA error: out of memory`, is caught, and is silently SKIPPED — the run still reports `gates=pass`, so heading merge/split refinement degrades invisibly. Seen on the ch01-03 Spark run 2026-07-11 (WF-20260711-bd0d1eda). Fix belongs in task #10's per-model GPU lifecycle: either lease the card to the council BERT at its stage seam, or fall the merge_or_split BERT back to CPU like NLI (`ED4ALL_NLI_DEVICE=cpu`) instead of swallowing the OOM. | gpu-contention/silent-degradation | 🟠 | M | done | fixed 2026-07-11 commit `1144c6c0` — `runner.run_bert` CPU-fallback on CUDA OOM (detect OOM class, `force_to_cpu`, re-run, latch CPU process-wide, loud warning; 8-test guard). Kills the silent skip. |

---

## Notes on the highest-value items

- **D3 (`extract_cache` invalidation)** is small but insidious — it silently makes
  code changes look like no-ops on re-run. Fix: fold a hash of the extraction/fusion
  code (or a bumped `EXTRACT_SCHEMA_VERSION`) into `_compute_extract_cache_key`, so a
  code change busts the cache. Highest debt-per-effort ratio here.

- **D1 (DART purge)** and **D2 (config sprawl)** are the big structural ones; both
  have written plans and should be scheduled as dedicated passes, not squeezed
  between features. D1 must be staged (dual-read shim → migrate persisted LibV2
  corpora → flip emitters → tighten) — a big-bang rename breaks every existing course.

- **D4 / D5** are debt *this session created or left*: an unvalidated experimental
  flag and a missing instrumentation contract. Per the prevention rule, these should
  be closed (validate-or-remove D4; wire DecisionCapture for D5) rather than lingering.

---

*Seeded 2026-07-11 from the file-hygiene audit, the dart-surface inventory, and
open items in project memory. Keep it current — see the maintenance protocol above.*
