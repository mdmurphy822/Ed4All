# Apply interactive components

The Courseforge component applier is a standalone, deterministic utility for
transforming plain HTML sections into Bootstrap-compatible interactive
components. It selects transformations through content-pattern matching and
adds accessibility-oriented markup for accordions, timelines, callouts, flip
cards, knowledge checks, and related components.

It is not an MCP remediation target and does not call a model provider. Run it
directly when an operator wants this optional post-processing step.

> **Private by design:** input HTML, output HTML, component reports, course
> identifiers, and log contents are always private. Keep them in Git-ignored
> or external storage and never commit them.

## Installation

Follow [Installation and local dependencies](../../../docs/operations/installation.md).
The repository records dependency requirements but does not host installed
packages or caches.

## Process one file

Run from the repository root:

```bash
python3 Courseforge/scripts/ops/component_applier.py \
  --input <PRIVATE_INPUT_HTML> \
  --output <PRIVATE_OUTPUT_HTML>
```

## Process a directory

```bash
python3 Courseforge/scripts/ops/component_applier.py \
  --input-dir <PRIVATE_INPUT_DIRECTORY> \
  --output-dir <PRIVATE_OUTPUT_DIRECTORY>
```

Add `--json` to print the processing report as JSON:

```bash
python3 Courseforge/scripts/ops/component_applier.py \
  --input-dir <PRIVATE_INPUT_DIRECTORY> \
  --output-dir <PRIVATE_OUTPUT_DIRECTORY> \
  --json
```

The accepted `--mapping` option is reserved for command-line compatibility and
currently has no effect. Component selection remains deterministic.

## Output

The utility writes transformed HTML to the requested private destination and
prints a summary containing file and component counts. Runtime logging goes to
the repository's ignored runtime log area.

Review transformed pages for accessibility and LMS compatibility before use.
Automated markup does not replace human accessibility review or target-LMS
import testing.
