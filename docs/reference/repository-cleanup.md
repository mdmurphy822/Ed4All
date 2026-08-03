# Repository cleanup record

This record explains the public-safe dispositions made during the repository
organization pass. It intentionally excludes operator data, course names,
course slugs, local paths, run identifiers, and inventories of ignored files.

The placement rules live in
[`../architecture/repo-organization.md`](../architecture/repo-organization.md).
Installation and dependency acquisition live in
[`../operations/installation.md`](../operations/installation.md).

## Dispositions

| Area | Evidence reviewed | Disposition |
|---|---|---|
| Root operational scripts | Imports, command examples, tests, and history | Grouped under `scripts/ops/`, `scripts/harness/`, `scripts/integration/`, `scripts/codegen/`, or `scripts/archive/` by responsibility |
| SemantiK scripts | Python imports, shell entry points, tests, and subprocess paths | Grouped by analysis, calibration, datasets, evaluation, smoke testing, and training; the cascade entry point remains flat |
| SemantiK data utilities | Package imports, generated-artifact boundaries, and tests | Grouped into alignment, augmentation, builders, common helpers, and source adapters |
| SemantiK evaluation inputs | Script defaults, comments, fixtures, workflow guidance, and operator paths | Removed tracked corpus identities and benchmark defaults; diagnostics now require explicit operator-supplied inputs and fail loudly when absent |
| Courseforge scripts and docs | Import paths, packaging commands, tests, and user journeys | Grouped by packaging, rendering, validation, guides, and reference material |
| Trainforge synthesis modules | MCP dispatch, imports, CLIs, and compatibility tests | Canonical implementation moved to `Trainforge/synthesis/`; legacy import paths remain warning-emitting compatibility surfaces |
| Trainforge evaluation modules | Imports, entry points, configuration, and evaluation tests | Grouped by metrics, retrieval checks, and runners; compatibility is retained only for documented external entry points |
| Trainforge deterministic generators | Imports, package exports, synthesis wiring, operator scripts, tests, schemas, and documentation | Grouped under `Trainforge/generators/deterministic/`; all tracked callers use canonical paths and no legacy alias is required |
| Validator registry | Registry loading and validator discovery | Retained as one cohesive registry; no cosmetic split |
| Historical migrations | Git history, tests, and operator documentation | Retained in archive directories when replay or provenance value remains |
| One-off SemantiK diagnostics | Imports, entry points, configuration, tests, docs, and history | Removed only where every reviewed surface showed no live or compatibility use |

The verified SemantiK removals were
`SemantiK/scripts/analysis/_diag_merge_structure.py`,
`SemantiK/scripts/eval/_compare_semantic_adapter_on_test.py`,
`SemantiK/scripts/eval/_compare_structure_adapter_on_test.py`,
`SemantiK/scripts/eval/_compare_table_adapter_on_test.py`, and
`SemantiK/scripts/eval/_eval_legal_fragments.py`. Static and dynamic imports, command
and subprocess entry points, workflow configuration, package exports, tests,
documentation, and Git history showed no live or compatibility use. Their
removal is recorded in commit `91728357`.

## Dependency boundary

The public repository tracks declarations and instructions, not installed
dependency payloads. Lock files, requirement manifests, licensing notes, and
reproducible installation guidance remain public. Downloaded packages,
third-party source trees, standards payloads, virtual environments, caches,
browser bundles, native libraries, and model artifacts remain local and are
ignored.

Required external schemas are named and verified by the installation process.
If they are absent, validation fails with an actionable message; the runtime
does not silently select a weaker validation path.

## Privacy boundary

Tracked code, comments, docstrings, fixtures, paths, examples, schemas, and
documentation use placeholders or clearly synthetic identifiers. Operator
course material, course and directory names, run evidence, input/output
contents, machine details, and local execution plans stay outside the public
tree.

## Removal standard

A low text-reference count is not proof that code is dead. Removal requires a
review of static and dynamic imports, package exports, CLI and subprocess entry
points, workflow configuration, tests, user documentation, and Git history.
Candidates are classified as live code, compatibility surface, reusable
harness, historical campaign artifact, or dead code. Only the final category
is deleted.
