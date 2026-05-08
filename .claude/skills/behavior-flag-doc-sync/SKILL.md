---
name: behavior-flag-doc-sync
description: Audit Ed4All behavior-flag documentation against the actual code references. Use when adding/removing a behavior flag, or to spot drift between the per-subsystem CLAUDE.md flag tables and the codebase. Reports flags in code missing from docs and vice versa.
---

# behavior-flag-doc-sync

Detect drift between the **Opt-In Behavior Flags** tables in the
per-subsystem `CLAUDE.md` files (W-D11d C5: split out of root) and the
actual `os.environ` references in code.

## Documented flag prefixes (per subsystem)

| Prefix | Owner CLAUDE.md |
|--------|-----------------|
| `TRAINFORGE_*` / `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` / `CURRICULUM_ALIGNMENT_*` / `WAVE18_*` | `Trainforge/CLAUDE.md` |
| `DART_*` | `DART/CLAUDE.md` |
| `COURSEFORGE_*` | `Courseforge/CLAUDE.md` |
| `DECISION_*` / `ED4ALL_*` / `LOCAL_DISPATCHER_*` / `MCP_ORCHESTRATOR_*` / `LLM_*` (cross-cutting) | root `CLAUDE.md` |

(Per-flag rationale also lives in `schemas/ONTOLOGY.md` § 12.)

## Procedure

1. **Find every flag reference in code.** Use ripgrep with PCRE2 so the
   `\K` trick works for clean extraction:

   ```bash
   rg -nP --no-heading 'os\.environ(?:\.get)?\(["\x27](TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|DART_|COURSEFORGE_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+' \
     lib/ MCP/ cli/ Trainforge/ LibV2/ Courseforge/ DART/
   ```

   Also check `os.getenv(...)`:

   ```bash
   rg -nP --no-heading 'os\.getenv\(["\x27](TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|DART_|COURSEFORGE_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+' \
     lib/ MCP/ cli/ Trainforge/ LibV2/ Courseforge/ DART/
   ```

   Extract the unique set of flag names. Sort them.

2. **Extract the documented flag set from each subsystem CLAUDE.md.**
   The tables live under `## Opt-In Behavior Flags`. Flags are in the
   first column, wrapped in backticks. Pull them across all four files:

   ```bash
   for f in CLAUDE.md DART/CLAUDE.md Courseforge/CLAUDE.md Trainforge/CLAUDE.md; do
     awk '/^## Opt-In Behavior Flags/,/^## [^O]/' "$f" \
       | grep -oP '`\K(TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|DART_|COURSEFORGE_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+'
   done | sort -u
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
     | grep -oP '`\K(TRAINFORGE_|LOCAL_SYNTHESIS_|TOGETHER_|ANTHROPIC_SYNTHESIS_|CURRICULUM_ALIGNMENT_|WAVE18_|DART_|COURSEFORGE_|DECISION_|ED4ALL_|LOCAL_DISPATCHER_|MCP_ORCHESTRATOR_|LLM_)\w+'
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
