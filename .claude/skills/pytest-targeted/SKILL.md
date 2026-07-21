---
name: pytest-targeted
description: Run the targeted pytest suite for Ed4All subprojects. Use when verifying changes scoped to one or more subprojects (SemantiK, Courseforge, Trainforge, LibV2, lib, MCP, schemas) without running the full 3616-test suite.
disable-model-invocation: true
---

# pytest-targeted

Run the pytest suite scoped to a single subproject (or a curated
combination), instead of the full ~3600-test repo suite.

## Per-subproject test paths

| Subproject | Command |
|------------|---------|
| `semantik` | `pytest SemantiK/tests/ lib/semantik/tests/` |
| `courseforge` | `pytest Courseforge/tests/` *(verify directory exists first — Courseforge tests may live elsewhere; run `ls Courseforge/` to confirm)* |
| `trainforge` | `pytest Trainforge/tests/` |
| `libv2` | `pytest LibV2/tools/libv2/tests/` |
| `lib` | `pytest lib/tests/ lib/validators/tests/ lib/ontology/tests/` |
| `mcp` | `pytest MCP/tests/` |
| `schemas` | `pytest schemas/tests/` |
| `integration` | `pytest tests/integration/` |
| `wave85` | `pytest Trainforge/tests/ lib/validators/tests/ lib/ontology/tests/ LibV2/tools/libv2/tests/ schemas/tests/` |

The `wave85` combination is the canonical fast-verification suite for
recent waves (Trainforge + validators + ontology + LibV2 + schemas).

## Usage

The skill takes a subproject name as argument (one of the keys above).

Default flags: `-x --tb=short`

- `-x` — stop at first failure (fast feedback for targeted runs).
- `--tb=short` — short tracebacks (avoid wall-of-text on a fail).

Example invocations:

```bash
# SemantiK only
pytest SemantiK/tests/ lib/semantik/tests/ -x --tb=short

# lib (three subdirs)
pytest lib/tests/ lib/validators/tests/ lib/ontology/tests/ -x --tb=short

# Wave 85 verification combo
pytest Trainforge/tests/ lib/validators/tests/ lib/ontology/tests/ LibV2/tools/libv2/tests/ schemas/tests/ -x --tb=short
```

## Procedure

1. Resolve the requested subproject to its command from the table above.
2. Verify the test directory exists (`ls -d <path>`); if it doesn't,
   surface that immediately rather than running pytest with a missing
   path.
3. Run the command with `-x --tb=short` appended (unless the caller
   passed an override).
4. If the user adds a `-k <expr>` or `-m <marker>` flag, append it to
   the command verbatim.
5. Report pass / fail counts and the first failing test (when `-x`
   tripped). Do not paste the whole pytest output — keep the report
   short.

## Constraints

- Always run from the repo root (the directory containing the top-level `CLAUDE.md`).
- Do **not** run the full repo suite (`pytest`) from this skill — that's
  a different workflow.
- Do not modify code or fixtures.
