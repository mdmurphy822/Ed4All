# LibV2 library-format versioning (OP4)

Canonical reference for the **on-disk LibV2 course-layout contract version**,
stamped on every freshly emitted `course_manifest.json` as
`library_format_version` (start `"1.0"`). This doc defines the *upgrade
contract* — what an old-format (or old-chunker) course means, who checks it,
and what bumps the version. The migration *framework* (an actual in-place
upgrader) is explicitly **deferred to v2.1**; today the contract is
report-only.

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

* **Missing `library_format_version`** (a manifest that predates the OP4 stamp)
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
  superseded contract. Until the v2.1 migration framework lands, the posture is:
  read what is safely readable, warn loudly, and require a **re-chunk / re-emit**
  (a fresh `ed4all run` archival) to bring the course to the current format.
  There is deliberately **no silent in-place rewrite**.

## Who checks it

| Checker | Behavior today |
|---------|----------------|
| `LibV2/tools/libv2/validator.py::validate_course_manifest` | REPORTS a warning when `library_format_version` is absent (pre-1.0 baseline). Report-only — never an error, never a block. |
| `lib/validators/libv2_manifest.py::LibV2ManifestValidator` (the `libv2_archival` gate) | Fresh archives always carry `library_format_version` because the emitter stamps it. The gate does not (yet) enforce a floor — the field is additive/optional in the schema so legacy manifests still validate. |
| `lib/libv2_fsck.py` | Version-awareness is a candidate follow-up; today fsck's manifest checks pass through. |

The field is **optional** in `schemas/library/course_manifest.schema.json`
(pattern `^\d+\.\d+$`), so every pre-OP4 manifest on disk continues to validate
untouched.

## What bumps `library_format_version`

Bump the minor version (`1.0` → `1.1`) when the course-directory layout changes
in a **backward-compatible** way (a new optional subdir/file that older
consumers can ignore). Bump the major version (`1.x` → `2.0`) on a
**breaking** layout change — a rename or removal that would make an old
consumer misread a new course, or a new consumer misread an old course. A major
bump is the trigger for the v2.1 migration framework: at that point an operator
needs a deterministic path from the old layout to the new one.

Do **not** bump `library_format_version` for:

* manifest-JSON-only key additions → bump `libv2_version`;
* chunk-emit-shape changes → bump `chunker_version`.

## Deferred: the migration framework (v2.1)

The actual upgrader — a `libv2 migrate <slug>` command that reads an
old-format course, transforms it to the current layout, re-stamps
`library_format_version`, and re-verifies the three-hash triangle — is
**out of scope for OP4** and deferred to v2.1. OP4 lands only:

1. the `library_format_version` **stamp** on fresh emits, and
2. this **contract** (serve read-only + warn; re-chunk to upgrade; never
   silent), plus the report-only validator awareness.

Until v2.1, "upgrading" an old-format course is a full re-run: re-chunk /
re-archive it through `ed4all run`, which mints a current-format course dir with
the current `library_format_version`.
