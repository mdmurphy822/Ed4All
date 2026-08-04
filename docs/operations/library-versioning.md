# LibV2 library-format versioning

Canonical reference for the **on-disk LibV2 course-layout contract version**,
stamped on every freshly emitted `course_manifest.json` as
`library_format_version` (start `"1.0"`). This doc defines the *upgrade
contract* — what an old-format (or old-chunker) course means, who checks it,
and what bumps the version. The `libv2 migrate` command provides the in-place
migration framework described below.

## The three version fields (do not conflate)

A LibV2 course manifest carries three independent version stamps:

| Field | Contract it versions | Bumped when | Producer |
|-------|----------------------|-------------|----------|
| `libv2_version` | The **manifest-document schema** (which keys the manifest JSON carries). | The manifest JSON shape changes. | `MCP/tools/pipeline_tools.py` archival (currently `"1.2.0"`). |
| `chunker_version` | The **chunk-emit contract** (the shape/semantics of `chunks.jsonl`). | The chunker's emit shape or semantics change. | `Trainforge.chunker.CHUNKER_SCHEMA_VERSION` (currently `"v4"`). |
| `library_format_version` | The **on-disk course-directory layout contract** (which subdirs/files a course dir is expected to contain, and how a consumer must read them). | The course-directory layout / read contract changes in a way that needs a migration. | `MCP/tools/pipeline_tools.py::LIBRARY_FORMAT_VERSION` (currently `"1.0"`). |

`chunker_version` is provenance-only at the manifest scope, but the **vector
index** load path already fails closed on a chunkset-SHA drift
(`SemanticIndexStale`) — a rebuild-only posture by design. `library_format_version`
sits one level up: it versions the *layout*, not the bytes.

## What an old-format / old-chunker course means

The governing rule is **serve read-only + warn; re-chunk to upgrade; never
silent**:

* **Missing `library_format_version`** (an unstamped legacy manifest)
  is treated as the **pre-1.0 baseline** layout. It is still servable — read
  paths must degrade gracefully — but a validator/fsck pass MUST surface a
  warning so an operator knows the course was written against an older layout.
* **An older `chunker_version`** than the current `CHUNKER_SCHEMA_VERSION`
  means the archived chunks were emitted by a superseded chunker. The course is
  still queryable, but any consumer that requires current-shape chunks (e.g. a
  fresh vector index) must **re-chunk** rather than silently reinterpret the old
  bytes. The `chunkset_manifest` gate already compares `chunker_version` against
  the in-repo constant.
* **An older `library_format_version`** than the current
  `LIBRARY_FORMAT_VERSION` means the course dir was laid out against a
  superseded contract. Two paths bring it current: the deterministic in-place
  `libv2 migrate` upgrader (§ "The migration framework"), or a full
  **re-chunk / re-emit** (a fresh `ed4all run` archival). Either way there is
  deliberately **no silent in-place rewrite** — `libv2 migrate` is dry-run by
  default and only writes on an explicit `--apply` (with a manifest backup +
  post-migration validate + rollback).

## Who checks it

| Checker | Behavior today |
|---------|----------------|
| `LibV2/tools/libv2/validator.py::validate_course_manifest` | REPORTS a warning when `library_format_version` is absent (pre-1.0 baseline). Report-only — never an error, never a block. |
| `lib/validators/libv2_manifest.py::LibV2ManifestValidator` (the `libv2_archival` gate) | Fresh archives always carry `library_format_version` because the emitter stamps it. The gate does not (yet) enforce a floor — the field is additive/optional in the schema so legacy manifests still validate. |
| `lib/libv2_fsck.py` | Version-awareness is a candidate follow-up; today fsck's manifest checks pass through. |

The field is **optional** in `schemas/library/course_manifest.schema.json`
(pattern `^\d+\.\d+$`), so every unstamped legacy manifest continues to validate
untouched.

## What bumps `library_format_version`

Bump the minor version (`1.0` → `1.1`) when the course-directory layout changes
in a **backward-compatible** way (a new optional subdir/file that older
consumers can ignore). Bump the major version (`1.x` → `2.0`) on a
**breaking** layout change — a rename or removal that would make an old
consumer misread a new course, or a new consumer misread an old course. A major
bump is the trigger for a new migration step: at that point an operator needs a
deterministic path from the old layout to the new one, registered in the
migration framework below.

Do **not** bump `library_format_version` for:

* manifest-JSON-only key additions → bump `libv2_version`;
* chunk-emit-shape changes → bump `chunker_version`.

## The migration framework

The upgrader has two pieces:

* **The framework** — `LibV2/tools/libv2/migrate.py`:
  * `detect_version(manifest)` — a manifest with no `library_format_version` is
    the pre-1.0 baseline, reported as the `LEGACY_VERSION` sentinel (`"legacy"`,
    deliberately not a real `^\d+\.\d+$` value so it never collides with a
    stamped one); any present value is the on-disk layout version.
  * `MigrationRegistry` — an ordered registry of `MigrationStep`s keyed
    `from_version -> to_version` (one deterministic successor per version). The
    registry's `target_version()` is the terminus of the chain from `legacy`;
    the module-level drift guard test asserts it equals
    `MCP/tools/pipeline_tools.py::LIBRARY_FORMAT_VERSION`.
  * `plan_course_migration()` / `apply_course_migration()` — the dry-run planner
    (never writes) and the apply engine (backup → transform → write → validate →
    rollback-on-failure).
  * `DEFAULT_REGISTRY` ships one baseline step: **`legacy -> 1.0`**, which stamps
    `library_format_version = "1.0"` (the pre-1.0 baseline directory layout *is*
    the 1.0 layout; only the explicit stamp was missing — the manifest
    "otherwise conforming" is enforced by the post-write validate pass, not by
    mutating any other key). A course already at `1.0` yields the empty
    "already current" plan (a no-op).

* **The CLI** — `libv2 migrate [SLUG] [--all] [--apply]`:
  * **Dry-run by default**: prints the per-course plan (`from_version ->
    current` plus the chained steps) and writes nothing.
  * `--apply` executes. On apply the engine (1) backs up `manifest.json` to a
    timestamped `manifest.json.<UTC-ts>.bak` sibling **before** any write,
    (2) applies the chained transforms and writes the migrated manifest,
    (3) re-runs the existing per-course LibV2 validate pass
    (`validator.py::validate_course` — structure + manifest-schema + taxonomy),
    and (4) **restores the manifest from the backup** if validation fails,
    reporting `rolled_back` + the validation errors and exiting non-zero.
  * `--all` iterates every course under `courses/` (`already current` courses
    are skipped as no-ops).

```bash
libv2 migrate <private-course-slug>               # dry-run plan for one course
libv2 migrate <private-course-slug> --apply       # migrate with backup, validation, and rollback
libv2 migrate --all                       # dry-run plan for every course
libv2 migrate --all --apply               # migrate every course
```

A full **re-chunk / re-emit** through `ed4all run` remains the alternative
upgrade path (it mints a fresh current-format course dir); `libv2 migrate` is
the in-place path for a manifest-only layout bump. Either way there is **no
silent in-place rewrite** — `migrate` only writes under an explicit `--apply`.

The whole-repo `lib/libv2_fsck.py::run_fsck` check is intentionally NOT the
per-migration validator: fsck is repo-scoped (blobs, catalog, run manifests) and
heavier, whereas an in-place manifest migration needs the per-course granularity
`validate_course` provides. `apply_course_migration` accepts an injectable
`validator` callable if a caller wants a stricter or lighter post-migration
check.
