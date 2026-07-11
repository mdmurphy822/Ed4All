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
| D1 | **DART naming purge** — migration removed the AGPL engine but left ~450 files of `dart` naming (attrs, sourceIds, paths, phases, agents, env) | naming/migration | 🟠 | XL | parked | `docs/dart-surface-inventory.md`, branch `refactor/dart-purge` |
| D2 | **Config / env-flag sprawl** — ~87 env flags, 5 dispatch paths, routing complexity; "one config + one gateway" never landed | config-sprawl | 🟠 | XL | open | memory `project_llm_management_redesign` |
| D3 | **`extract_cache` doesn't invalidate on fusion/extraction code change** — a code fix is silently masked by the cached extraction (cost real debug time 2026-07-11); cache key omits a code/version hash | correctness-footgun | 🟠 | M | open | `SemantiK/dart_semantic/extract_shared.py` `_compute_extract_cache_key` |
| D4 | **`SEMANTIK_VLM_ORDER_AUTHORITATIVE`** committed default-OFF with a *mis-designed trigger* (gates on divergence, which doesn't capture pure order-scramble) — dormant wrong code | code-quality | 🟠 | M | open | `SemantiK/dart_semantic/vlm_fusion.py` (commit `8cf0318f`) |
| D5 | **`vlm_extract` LLM call site lacks `DecisionCapture`** — violates the root CLAUDE.md LLM-call-site instrumentation contract (+ needs a docs/LICENSING.md seat row) | contract-gap | 🟠 | S–M | open | task #15; `SemantiK/dart_semantic/vlm_extract.py` |
| D6 | **Fusion garbage-tails + phantom-chapter headings** — clean VLM block + appended tesseract garbage (`® -Iq\|…`); running page-headers minted as chapters | conversion-quality | 🟠 | M–L | open | task #20 |
| D7 | **Scan multi-column reading-order** (broad) — the known-hard SemantiK structure problem; the number-fix resolved the *grid* case, but dense multi-column + heading hierarchy remain | conversion-quality | 🟠 | L | open | memory `plans-pointer-semantik-restructure` |
| D8 | **`runtime/` junk drawer** — Jun-8 scratch: `fix_*`/`complete_*` one-off scripts, ~25 `_res_`/`_fres_` result blobs, 11 stray logs, `qwen_test/` prototype (all gitignored → local disk only) | hygiene/scratch | 🟡 | S | open | `docs/file-audit-cleanup.md` |
| D9 | **Stray local logs / `.bak` / `.contaminated`** — `state/logs/*.contaminated`, `state/workflows/*.pre-remediation.bak`, 3× 0-byte root `.log` (gitignored) | hygiene/scratch | 🟡 | S | open | `docs/file-audit-cleanup.md` |
| D10 | **Doc/count drift risk** — the CLAUDE.md family carries many hand-maintained counts + flag tables (gate counts, flag indices) that can drift from code | docs | 🟡 | ongoing | open | `doc-sanitation-reviewer` agent, flag-doc-sync |
| D11 | **Test-bed representativeness** — small slices behave differently from full runs (a 3-page slice dropped content a 198-page run kept), so slice-based validation can mislead | process | 🟡 | ongoing | open | 2026-07-11 fusion-fix debugging |
| D12 | **Marketable-v2 open items** — LTI fork + pilot decisions, demo build un-run, chunker 10-vs-21 drift | product-open-items | 🟡 | var | open | memory `project_marketable_v2_implemented` |

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
