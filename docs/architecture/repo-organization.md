# Repository Organization

Ed4All is organized around four first-class products—**SemantiK**,
**Courseforge**, **Trainforge**, and **LibV2**—supported by shared platform
packages, contracts, tools, and private runtime storage. This document defines
where repository content belongs at every depth.

The executable policy is
[`repository-layout.json`](repository-layout.json), validated by
[`repository-layout.schema.json`](repository-layout.schema.json) and enforced
by the repository guards. This guide explains how contributors apply that
policy. Exact file inventories belong in the machine-readable policy and
allowlists, not in this document.

## 1. Authority and design principles

Use these sources together:

1. `repository-layout.json` classifies root paths, recursive roles, public
   documents, external-data areas, sentinels, and prohibited artifacts.
2. `ci/layout_allowlist.txt` supplies closed-root and flat-directory ratchets.
3. `ci/layout_guard.py` and `ci/guards/repository_policy.py` enforce the
   tracked-tree, release, privacy, and role-composition rules.
4. This document is the contributor-facing placement contract.

The organizing principles are:

- A directory has one clear role and may contain only compatible child roles.
- Product-specific behavior stays with its owning product; genuinely shared
  behavior belongs in a shared package.
- Mutable data, corpora, generated output, credentials, machine configuration,
  and local planning are never public source dependencies.
- Tests live beside the product or package they verify. Root `tests/` is for
  cross-product integration.
- Comments and docstrings explain current purpose, contracts, and non-obvious
  behavior. Authoring circumstances, worker attribution, and cleanup history
  belong in version control or local planning notes.
- New structure must make the repository easier to navigate without breaking
  supported imports, commands, dispatch names, schemas, or persisted state.

## 2. The schema: four zones, one allowlist

The repository root is closed. A new root directory or root file requires an
intentional policy and allowlist change in the same review.

| Zone | Purpose | Placement rule |
|---|---|---|
| **Product code** | `SemantiK/`, `Courseforge/`, `Trainforge/`, and `LibV2/` | Each product remains a first-class root and owns its implementation, documentation, tests, fixtures, templates, and product-local tools. |
| **Shared platform** | Shared libraries, orchestration, CLI, and GUI surfaces | Cross-cutting code lives in `lib/` or `MCP/`; `cli/` and `gui/` remain thin interfaces over reusable behavior. |
| **Contracts and infrastructure** | Configuration, schemas, CI policy, scripts, documentation, and integration tests | Content is tracked and hand-authored. Generated data and machine-specific configuration are prohibited. |
| **Private variable data** | Runtime state, corpus input, local plans, generated products, and operator-only material | Content is ignored and is never required to build, test, package, or document the public source tree. |

The four product roots are peers. Do not nest one beneath another, move their
implementation into a generic package tree, or create a second root for part
of a product. A product may use shared public contracts, but it must not import
a sibling product's private implementation details.

### 2.1 Role composition

`repository-layout.json` defines the allowed parent/child role matrix. The
common roles are:

- `subsystem` for a product boundary;
- `package` for importable implementation;
- `surface` for CLI and GUI entry layers;
- `tools` for maintained operational utilities;
- `contracts` for schemas and configuration;
- `tests`, `fixtures`, and `templates` for their named purposes;
- `docs` for reviewed public documentation; and
- `external-var` for private or generated data.

A path must resolve to exactly one role. Tests and fixtures receive recursive
overrides so they can mirror the code they validate. Product-local data areas
receive explicit `external-var` overrides and remain private even though their
parent product is public source.

### 2.2 Placement rules

- `lib/` contains reusable cross-cutting Python. New modules normally begin in
  `lib/<topic>/`; new flat `lib/*.py` files are prohibited by the name ratchet.
- `MCP/` contains orchestration, workflow execution, and tool exposure.
  Reusable validators and domain logic belong in `lib/` or the owning product.
- `cli/` and `gui/` translate user interaction into shared APIs. Business logic
  that must also work headlessly does not belong only in either surface.
- `config/` contains portable orchestration configuration and constraints.
  Secrets and per-machine values are private runtime configuration.
- `schemas/` contains JSON Schema, SHACL, ontology, and taxonomy contracts—not
  generated instances.
- `ci/` contains repository policy, guards, allowlists, and their tests.
  Operational runbooks belong in `docs/operations/`.
- `.claude/` and `.codex/` contain reviewed harness-level agent configuration,
  not pipeline agents or runtime state. Personal settings, sessions, caches,
  and worktrees remain ignored.
- `scripts/` contains maintained utilities organized by § 3. Shipped code must
  not import a script as a runtime dependency.
- `docs/` follows the publication contract in § 4.
- `tests/` contains only cross-product tests. Product and package tests stay
  with their owner.

Directory names are lowercase kebab-case or snake_case, matching their
siblings. The four product names and `MCP` are the intentional root naming
exceptions. Public Markdown filenames use kebab-case except established
entry-point names such as `README.md`, `ARCHITECTURE.md`, and `CLAUDE.md`.

## 3. `scripts/` taxonomy

Every directory named `scripts/` uses the same recursive organization model:

```text
scripts/
├── ops/          durable operator commands
├── harness/      repeatable measurement and quality harnesses
├── integration/  cross-component integration utilities
├── codegen/      contract and source generation maintenance
├── maintenance/  repository or product maintenance helpers, where needed
├── tests/        tests for the tracked script families
└── regression/   ignored local shelf for obsolete scripts
```

Only create the families a particular `scripts/` directory needs. Do not add
loose scripts beside them unless a supported entry point must remain there and
the layout policy explicitly permits it.

Use this placement test:

- A documented command used to operate the product belongs in `ops/`.
- A reproducible measurement intended for repeated use belongs in `harness/`.
- A utility coordinating multiple components belongs in `integration/`.
- A tool that regenerates tracked contracts belongs in `codegen/`.
- A one-off experiment or corpus-specific repair belongs in private runtime
  storage, not the source tree.

Every `scripts/` directory may contain `regression/`. The recursive ignore rule
keeps these shelves out of Git. Move a script there only after checking imports,
dynamic dispatch, commands, configuration, documentation, tests, and relevant
history. A regression-shelved script is not a supported interface or test
dependency. Never create a tracked `scripts/archive/` substitute.

## 4. `docs/` taxonomy

Public documentation uses four durable buckets:

- `docs/architecture/` — system design and architecture decisions;
- `docs/operations/` — installation, commands, configuration, and runbooks;
- `docs/validation/` — gates, validators, and quality contracts; and
- `docs/reference/` — stable indexes, compliance references, and technical
  detail that supports the other guides.

`docs/LICENSING.md` remains the root documentation licensing register. Avoid
single-file documentation directories; place a document in the closest
existing bucket.

Documentation is private by default. A document is public only when its exact
path appears in both `repository-layout.json`'s `public_docs` list and the
final-order `.gitignore` negation mirror. Directory-wide publication patterns
are prohibited. An ignored document must not become a code, build, test, link,
or packaging dependency.

Public documentation must use examples that cannot identify a real corpus,
course, run, person, machine, or network. Do not publish hardcoded course names
or slugs, local absolute paths, hostnames, LAN addresses, credentials, corpus
extracts, generated outputs, or comments describing private input. Use clearly
synthetic placeholders and portable environment-variable examples.

## 5. Runtime, dependency, and compatibility boundaries

### 5.1 Private runtime and corpus data

All corpus inputs and pipeline outputs are private. There is no public-corpus
opt-in path. The in-repository defaults separate source from variable data:

- `inputs/` stages private source material;
- `runtime/` contains mutable state, caches, captures, local configuration,
  diagnostics, generated previews, and experiments;
- product-local paths classified as `external-var` contain that product's
  generated or imported material; and
- `plan/` is the ignored operator planning workspace and never a build input.

Only policy-approved sentinel files may be tracked in an `external-var` tree.
`ED4ALL_HOME` may relocate runtime storage without changing the public source
layout. New mutable-data categories belong below an existing private boundary,
not at repository root.

The repository policy checks tracked files and unignored release candidates for
private tokens, course-shaped paths, nested repositories, generated artifacts,
oversized files, and secret-like content. Operators may extend the local
private-token vocabulary through `ED4ALL_PRIVATE_TOKEN_FILE`; that file must
itself remain ignored.

### 5.2 Dependencies

Git stores dependency declarations and installation instructions, not installed
dependencies. Do not commit virtual environments, package caches, downloaded
models, model weights, container layers, third-party source snapshots, or
generated dependency bundles. Declare Python dependencies in the appropriate
project metadata and explain supported installation paths in public operations
documentation. Vendored material requires an explicit licensing and repository
policy decision.

### 5.3 Compatibility shims

When reorganizing supported code, prefer updating callers to the canonical
path. A compatibility shim is justified only for a documented import, CLI,
module invocation, plugin hook, serialized identifier, or dynamic dispatch
surface that external users may rely on.

A shim must:

- re-export or delegate to one canonical implementation;
- preserve object identity and behavior where callers depend on them;
- contain no duplicate business logic;
- state the supported replacement path and deprecation policy; and
- have a focused compatibility test.

Private underscore modules and unexported internal helpers do not receive shims
without evidence of a supported consumer. Do not retain aliases solely because
a file once existed at another path.

## 6. Enforcement

Two complementary guards enforce organization:

1. `ci/layout_guard.py` reads tracked paths from `git ls-files` and enforces
   the closed root, the flat `lib/` module ratchet, the documentation taxonomy,
   loose root-script restrictions, the ban on tracked script archives, and
   recursive flat-file caps.
2. `ci/guards/repository_policy.py` validates recursive role classification,
   parent/child compatibility, public-document approval, external-data
   sentinels, generated and oversized artifacts, secret-like content, private
   path tokens, and nested source repositories.

Both checks are registered in `ci/integrity_check.py`. The JSON policy is
validated against its schema before classification is trusted.

### 6.1 Ratchets

Allowlist entries may only shrink, and numeric flat caps may only decrease.
Adding an allowlist entry or raising a cap is an exception that requires a
specific design justification in the same review. Renaming or moving content
must also remove stale allowances; a cap for a vanished directory is a policy
failure rather than harmless residue.

The guards intentionally evaluate tracked paths for layout and the broader set
of unignored release candidates for publication safety. Ignoring private data
is a safety boundary, not permission to force-add it.

## 7. Subsystem interior schema

The organization rules apply recursively within **SemantiK**, **Courseforge**,
**Trainforge**, **LibV2**, and every shared platform tree.

### 7.1 Cohesive packages and containers

Every non-test code directory is one of two shapes:

- A **cohesive package** contains peer modules implementing one contract. A
  flat module set can be correct when that set is itself the package registry.
- A **container** groups material with different purposes or lifetimes. It must
  use named child packages or script families instead of accumulating unrelated
  loose files.

When a new module does not clearly share the directory's existing contract,
create or use a focused child package. Do not use generic buckets such as
`misc`, `old`, `temp`, or `archive`.

### 7.2 Flat-file caps

`flatcap:<directory>=<count>` entries in `ci/layout_allowlist.txt` freeze the
number of loose code files in larger interior directories. The cap controls a
count rather than filenames so supported renames remain possible while
unstructured growth is blocked.

The count excludes:

| Exclusion | Reason |
|---|---|
| paths with a `tests` segment | Tests may mirror a flat implementation without discouraging coverage. |
| `__init__.py` | Package markers are structural, not feature modules. |
| Markdown files | Required package guides do not represent loose code growth. |

Contracts in `config/` and `schemas/` are not interior code containers, and
private variable-data trees are outside the tracked-file cap. Tests and
fixtures remain subject to recursive role and privacy policy even when exempt
from the numeric cap.

### 7.3 Documentation at product and package boundaries

Each first-class product has a public `README.md` and `architecture.md` that
match the root documentation's accessible style while remaining technically
specific to that product. Deeper directories need local documentation when
they expose a supported interface or when their contract cannot be understood
from the parent architecture guide. Avoid copying exact inventories or
duplicating flags, schemas, and commands owned by another canonical document;
link to the source of truth instead.

### 7.4 Contributor move procedure

Before moving code or documentation:

1. Identify the owning product or shared contract and the destination role.
2. Search static imports, dynamic imports, dispatch registries, CLI/module
   invocations, configuration, schemas, tests, public docs, and persisted path
   identifiers.
3. Decide whether the old path is a supported interface. Add a minimal shim
   only when § 5.3 requires one.
4. Move the implementation and update all in-repository consumers to the
   canonical path. Move or update its focused tests and documentation in the
   same change.
5. Recalculate applicable ratchets and reduce them. Remove stale allowlist,
   policy override, and ignore entries.
6. Run the focused tests, layout guard, recursive repository-policy guard,
   privacy checks, documentation-link checks, and the relevant integrity
   suite.
7. Inspect `git diff` and `git status` for accidental generated data, private
   names, local paths, dependency artifacts, and unrelated edits.

If ownership, compatibility, licensing, or publication status is unclear, stop
and resolve that design question before changing the tree.
