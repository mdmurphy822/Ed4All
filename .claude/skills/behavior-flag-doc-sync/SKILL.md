---
name: behavior-flag-doc-sync
description: Audit Ed4All behavior-flag documentation against the actual code references. Use when adding/removing a behavior flag, or to spot drift between the per-subsystem CLAUDE.md flag tables and the codebase. Reports flags in code missing from docs and vice versa.
---

# behavior-flag-doc-sync

Detect drift between the documented behavior-flag tables — the
**Opt-In Behavior Flags** tables in the per-subsystem `CLAUDE.md` files
plus the root-owned cross-cutting table in
`docs/operations/behavior-flags.md` (split out of root `CLAUDE.md` for
context-size reasons; root keeps a one-line index) — and the actual
`os.environ` references in code.

## Documented flag prefixes (per subsystem)

| Prefix | Owner doc |
|--------|-----------|
| `TRAINFORGE_*` / `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` / `CURRICULUM_ALIGNMENT_*` / `WAVE18_*` / `NVIDIA_*` | `Trainforge/CLAUDE.md` |
| `SEMANTIK_*` (plus the single allowlisted `DART_THETA_DEVICE` legacy-compat env, documented in `SemantiK/CLAUDE.md` as the `SEMANTIK_THETA_DEVICE` fallback) <!-- legacy-token: allow --> | `SemantiK/CLAUDE.md` |
| `COURSEFORGE_*` / `COURSEPLANNER_*` / `TEXTBOOK_SYNTHESIS_*` | `Courseforge/CLAUDE.md` |
| `DECISION_*` / `ED4ALL_*` / `LOCAL_DISPATCHER_*` / `MCP_ORCHESTRATOR_*` / `LLM_*` (cross-cutting) | `docs/operations/behavior-flags.md` (canonical per-flag detail; root `CLAUDE.md` keeps a one-line index) |

(Per-flag rationale also lives in `schemas/ONTOLOGY.md` § 12.)

## Procedure

1. **Find every flag reference in code.** Use ripgrep with PCRE2 so the
   `\K` trick works for clean extraction:

   ```bash
   rg -nP --no-heading 'os\.environ(?:\.get)?\(["\x27](TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|NVIDIA_|SEMANTIK_|COURSEFORGE_|COURSEPLANNER_|TEXTBOOK_SYNTHESIS_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+' \
     lib/ MCP/ cli/ Trainforge/ LibV2/ Courseforge/ SemantiK/
   ```

   Also check `os.getenv(...)`:

   ```bash
   rg -nP --no-heading 'os\.getenv\(["\x27](TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|NVIDIA_|SEMANTIK_|COURSEFORGE_|COURSEPLANNER_|TEXTBOOK_SYNTHESIS_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+' \
     lib/ MCP/ cli/ Trainforge/ LibV2/ Courseforge/ SemantiK/
   ```

   Extract the unique set of flag names. Sort them.

2. **Extract the documented flag set.** In the subsystem `CLAUDE.md`
   files the tables live under `## Opt-In Behavior Flags`; flags are in
   the first column, wrapped in backticks. The root-owned cross-cutting
   flags' canonical per-flag detail lives in
   `docs/operations/behavior-flags.md` (the whole file is one table);
   root `CLAUDE.md` also carries a one-line index of the same names.
   Pull them across all sources:

   ```bash
   {
     # subsystem tables (awk-scoped to the Opt-In section) + root index
     for f in CLAUDE.md SemantiK/CLAUDE.md Courseforge/CLAUDE.md Trainforge/CLAUDE.md; do
       awk '/^## Opt-In Behavior Flags/,/^## [^O]/' "$f" \
         | grep -oP '`\K(TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|NVIDIA_|SEMANTIK_|COURSEFORGE_|COURSEPLANNER_|TEXTBOOK_SYNTHESIS_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+'
     done
     # root-owned cross-cutting canonical doc (whole-file table)
     grep -oP '`\K(DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+' docs/operations/behavior-flags.md
   } | sort -u
   ```

3. **Diff the two sets.** Report:

   - **Undocumented flags** — present in code but missing from every
     subsystem's table. These need a row added in the prefix-owner's
     CLAUDE.md.
   - **Stale documented flags** — present in some table but no longer
     referenced anywhere in code. These need either a removal or an
     explanation if they are intentionally reserved.
   - **Mis-located flags** — documented in the wrong subsystem (e.g. a
     `TRAINFORGE_*` row in `Courseforge/CLAUDE.md`).

4. **Cross-reference `schemas/ONTOLOGY.md` § 12** for additional
   rationale:

   ```bash
   awk '/^### Opt-in flags/,/^### [^O]/' schemas/ONTOLOGY.md \
     | grep -oP '`\K(TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|NVIDIA_|SEMANTIK_|COURSEFORGE_|COURSEPLANNER_|TEXTBOOK_SYNTHESIS_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+'
   ```

5. **Report**

   - List of undocumented flags (file:line references) and recommended
     owner-subsystem.
   - List of stale documented flags (which subsystem CLAUDE.md).
   - List of mis-located flags.
   - List of `ONTOLOGY.md § 12` ↔ subsystem-CLAUDE.md mismatches.
   - Suggested doc additions / removals / relocations.

## Constraints

- Read-only audit. **Do not edit `CLAUDE.md`, `ONTOLOGY.md`, or code.**
  Surface findings; let the author apply edits.
- Treat any `os.environ[...]` direct-subscript usage the same as
  `os.environ.get(...)` — both are flag reads.
