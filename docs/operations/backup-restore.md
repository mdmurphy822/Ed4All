# `ed4all backup` (OP3)

A **complete, restore-able snapshot** of every mutable Ed4All data directory —
and a verifier for it. Unlike [`support-bundle`](support-bundle.md) (a redacted
*diagnostic* slice), a backup is meant to bring a working system back, so it
includes secrets and is written owner-only `0600`.

```bash
# Create a backup of every resolved data dir (0600, includes secrets.json)
ed4all backup --output /secure/ed4all-backup.tar.gz

# Verify an existing archive (manifest sha256 + libv2 fsck)
ed4all backup --verify /secure/ed4all-backup.tar.gz
```

`--output` (create) and `--verify` are mutually exclusive.

## What's IN

Every directory in the `_DATA_DIR_KEYS` relocation set, **resolved through
`lib/paths` helpers** so `ED4ALL_HOME` and each per-dir override are honored —
paths are never hardcoded:

| Archive prefix | Resolver | Honors |
|----------------|----------|--------|
| `state/` | `ED4ALL_HOME/state` (home-aware, call-time) | `ED4ALL_HOME` |
| `libv2/` | `lib.paths.libv2_path()` | `ED4ALL_LIBV2_ROOT` → `ED4ALL_HOME` |
| `exports/` | `lib.paths.courseforge_exports_dir()` | `ED4ALL_HOME` |
| `training-captures/` | `lib.paths.get_training_captures_dir()` | `ED4ALL_TRAINING_CAPTURES_DIR` → `ED4ALL_HOME` |
| `semantik-output/` | `lib.paths.semantik_output_dir()` | `ED4ALL_HOME` (dual-reads a legacy `dart-output/` on a pre-task-#19 box; old backups whose manifest carries the `dart-output` key still restore — restore iterates the manifest keys generically) |
| `state-runs/` | `lib.paths.get_state_runs_dir()` | `ED4ALL_STATE_RUNS_DIR` — added **only** when the runs subtree is relocated OUTSIDE the state root (scattered layout) |

A missing directory is reported (`missing_dirs`) and skipped — not fatal.

`manifest.json` (at the archive root) records the resolved `dirs`, the
`missing_dirs`, and every member's `arcname` / `size` / `sha256`.

## What's OUT

* **Docker-only external stores** — the HuggingFace model cache and the ollama
  model store live outside the data-dir set and are **out of scope**. They are
  re-fetchable and large; back them up with your container/volume tooling, not
  this command. See [`docker.md`](docker.md).
* Nothing else is filtered — a backup is deliberately complete.

## Secret handling

A backup **includes `secrets.json`** on purpose: a restored system needs its
credentials to be functional. Consequences:

* The archive is chmod'd to **`0600`** (owner read/write only) and the command
  says so on stdout.
* Treat the file as a credential. **Do not commit it, do not share it**, store
  it in your secrets-grade location.

This is the deliberate inverse of `support-bundle`, which *drops* secrets so it
is safe to hand to a maintainer.

## Verification (`--verify`)

`--verify ARCHIVE` extracts the archive to a temp dir (refusing any member whose
path escapes the extract root) and:

1. **Recomputes every member's sha256** against the embedded `manifest.json`. A
   mismatch or a missing member fails verification — this is what catches a
   **corrupted member**. A tar/gzip stream that is itself unreadable fails
   loudly rather than raising.
2. **Runs the LibV2 fsck** (`lib/libv2_fsck.py`) over the extracted `libv2/`
   tree, folding any blob/catalog **error** into the verdict.

Exit code `0` = verified, `1` = failed (not restore-safe). The temp extraction
is a full copy of the archive; ensure enough scratch space for a large backup.

## Restore

Restore is a manual, deliberate step (there is no `--restore` — overwriting a
live data root should never be a single flag):

```bash
# 1. Verify first
ed4all backup --verify /secure/ed4all-backup.tar.gz

# 2. Extract into the target data root (ED4ALL_HOME, or the repo root)
tar -xzf /secure/ed4all-backup.tar.gz -C "$ED4ALL_HOME"

# 3. Re-run fsck against the live tree
ed4all fsck
```

The archive's top-level prefixes (`state/`, `libv2/`, …) match the `ED4ALL_HOME`
data layout, so extracting into `ED4ALL_HOME` reconstitutes the tree. For a
scattered layout (per-dir overrides), extract each prefix to its override target.
