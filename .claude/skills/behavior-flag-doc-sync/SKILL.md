---
name: behavior-flag-doc-sync
description: Audit Ed4All behavior-flag documentation against the actual code references. Use when adding/removing a behavior flag, or to spot drift between the per-prefix behavior-flag reference docs and the codebase. Reports flags in code missing from docs and vice versa.
---

# behavior-flag-doc-sync

Detect drift between the documented behavior-flag tables — the four
per-prefix reference docs under `docs/operations/` — and the actual
`os.environ` references in code.

All four tables were split out of the `CLAUDE.md` family for context-size
reasons; each `CLAUDE.md` now carries only a pointer to its owner doc.

## Documented flag prefixes (per subsystem)

| Prefix | Owner doc |
|--------|-----------|
| `TRAINFORGE_*` / `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` / `CURRICULUM_ALIGNMENT_*` / `WAVE18_*` / `NVIDIA_*` | `docs/operations/behavior-flags-trainforge.md` |
| `SEMANTIK_*` (plus the single allowlisted `DART_THETA_DEVICE` legacy-compat env, documented as the `SEMANTIK_THETA_DEVICE` fallback) <!-- legacy-token: allow --> | `docs/operations/behavior-flags-semantik.md` |
| `COURSEFORGE_*` / `COURSEPLANNER_*` / `TEXTBOOK_SYNTHESIS_*` | `docs/operations/behavior-flags-courseforge.md` |
| `DECISION_*` / `ED4ALL_*` / `LOCAL_DISPATCHER_*` / `MCP_ORCHESTRATOR_*` / `LLM_*` (cross-cutting) | `docs/operations/behavior-flags.md` |

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

2. **Extract the documented flag set.** Each owner doc is a single
   whole-file table; flags are in the first column, wrapped in backticks.
   Pull them across all four:

   ```bash
   grep -hoP '`\K(TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|NVIDIA_|SEMANTIK_|COURSEFORGE_|COURSEPLANNER_|TEXTBOOK_SYNTHESIS_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+' \
     docs/operations/behavior-flags.md \
     docs/operations/behavior-flags-semantik.md \
     docs/operations/behavior-flags-trainforge.md \
     docs/operations/behavior-flags-courseforge.md \
     | sort -u
   ```

3. **Diff the two sets.** Report:

   - **Undocumented flags** — present in code but missing from every
     table. These need a row added in the prefix-owner's doc.
   - **Stale documented flags** — present in some table but no longer
     referenced anywhere in code. These need either a removal or an
     explanation if they are intentionally reserved.
   - **Mis-located flags** — documented under the wrong prefix owner (e.g. a
     `TRAINFORGE_*` row in `behavior-flags-courseforge.md`).

4. **Cross-reference `schemas/ONTOLOGY.md` § 12** for additional
   rationale:

   ```bash
   awk '/^### Opt-in flags/,/^### [^O]/' schemas/ONTOLOGY.md \
     | grep -oP '`\K(TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|NVIDIA_|SEMANTIK_|COURSEFORGE_|COURSEPLANNER_|TEXTBOOK_SYNTHESIS_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+'
   ```

5. **Report**

   - List of undocumented flags (file:line references) and recommended
     owner doc.
   - List of stale documented flags (which owner doc).
   - List of mis-located flags.
   - List of `ONTOLOGY.md § 12` ↔ owner-doc mismatches.
   - Suggested doc additions / removals / relocations.

## Constraints

- Read-only audit. **Do not edit the flag docs, `CLAUDE.md`, `ONTOLOGY.md`,
  or code.**
  Surface findings; let the author apply edits.
- Treat any `os.environ[...]` direct-subscript usage the same as
  `os.environ.get(...)` — both are flag reads.
