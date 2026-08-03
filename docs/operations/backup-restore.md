# Backup and restore

`ed4all backup` creates a private recovery archive from Ed4All’s resolved data
directories and verifies archives before restoration. It is a data backup, not
an application image, dependency mirror, or model snapshot.

Backups may contain credentials, course content, generated outputs, learner or
operator activity, and model interaction records. Treat every archive as
sensitive even when its filename or manifest looks harmless.

## Private data scope

The command resolves storage through [`lib/paths.py`](../../lib/paths.py), so it
honors the active data root and supported per-directory overrides. Its scope
includes application state, the LibV2 course library, Courseforge exports,
training captures, SemantiK output, and a separately relocated run-state tree
when one is configured.

The archive includes credentials stored inside those directories. It is created
with owner-only file permissions, but it is a gzip-compressed tar archive—not an
encrypted archive.

The following remain outside the application backup:

- container images and writable container layers;
- model weights and model-server volumes;
- embedding, tokenizer, browser, and package caches;
- external databases or object stores not resolved by Ed4All path helpers; and
- source documents held outside the configured data directories.

Re-create excluded dependencies from approved upstream sources or protect them
with the backup facilities of the platform that owns them. Do not add dependency
payloads or model caches to the project repository.

## Create an archive

Quiesce writes before creating a recovery point: stop active workflows and
prevent uploads or library mutations. The backup is a filesystem walk, not a
transactional snapshot, so concurrent writes can produce an internally
inconsistent point in time.

Choose a new path in private, operator-controlled storage:

```bash
export ED4ALL_PRIVATE_BACKUP_DIR='<private-backup-directory>'
export ED4ALL_BACKUP="$ED4ALL_PRIVATE_BACKUP_DIR/ed4all-backup.tar.gz"
ed4all backup --output "$ED4ALL_BACKUP"
```

The command writes an embedded `manifest.json` containing archive member names,
sizes, and SHA-256 digests. It reports included and missing data roots and sets
the resulting archive mode to `0600` where the filesystem permits it.

There is no create-mode `--dry-run`, no incremental mode, and no built-in
retention policy. The output path is opened for writing and an existing file can
be replaced. Before running the command, confirm the target is the intended new
archive:

```bash
test ! -e "$ED4ALL_BACKUP"
```

A configured data directory that does not exist is reported and skipped. The
archive can still be created. Review that list: a missing directory may be
expected on a new installation, or it may indicate an incorrect environment.
The creator also skips a source file it cannot read, so quiescing the system and
performing a separate inventory or storage-level snapshot is important for
high-assurance recovery.

## Verify an archive

Verify immediately after creation and again before every restore:

```bash
ed4all backup --verify "$ED4ALL_BACKUP"
```

Verification safely extracts to a temporary directory, rejects archive members
that escape that directory, checks every manifest-listed member against its
SHA-256 digest, and runs a read-only LibV2 consistency check when the library is
present.

Missing archives, unreadable tar or gzip streams, unsafe paths, missing
manifests, missing members, digest mismatches, and LibV2 error findings make the
archive not restore-safe and return a nonzero status. If the LibV2 checker itself
cannot run, verification reports that it was skipped; repeat the check in a
working Ed4All environment before treating the archive as disaster-recovery
ready.

`--output` and `--verify` are mutually exclusive. The command has no mode that
repairs a damaged archive.

## Encryption and retention

Ed4All does not encrypt, upload, rotate, or expire backup archives. The operator
is responsible for:

- encrypting archives at rest and in transit with an approved system;
- controlling access to encryption keys separately from the archive;
- keeping multiple recovery points in failure-independent locations;
- defining retention and secure-deletion periods; and
- testing restores often enough to detect configuration or format drift.

File mode `0600` protects against other local users under normal Unix permission
semantics. It does not protect against privileged users, copied files, lost
media, remote storage exposure, or an unencrypted transfer.

## Restore safely

There is deliberately no `ed4all backup --restore` command. Restoration can
overwrite live state and therefore requires explicit staging and conflict
review.

First verify and extract into a new temporary directory:

```bash
ed4all backup --verify "$ED4ALL_BACKUP"
export ED4ALL_RESTORE_STAGE="$(mktemp -d)"
tar -xzf "$ED4ALL_BACKUP" -C "$ED4ALL_RESTORE_STAGE"
find "$ED4ALL_RESTORE_STAGE" -maxdepth 2 -mindepth 1 -print
```

Inspect `manifest.json`, the top-level data prefixes, available disk space, and
the destination configuration. Do not extract an unverified archive directly
over a live data root.

For a unified `ED4ALL_HOME` layout, preview file conflicts with `rsync` after
stopping Ed4All services:

```bash
test -n "$ED4ALL_HOME"
rsync -a --dry-run --itemize-changes --exclude manifest.json \
  "$ED4ALL_RESTORE_STAGE/" "$ED4ALL_HOME/"
```

The staging root also contains `manifest.json`; exclude it when applying the
restore because it describes the archive rather than live application state:

```bash
rsync -a --exclude manifest.json "$ED4ALL_RESTORE_STAGE/" "$ED4ALL_HOME/"
```

For a scattered layout, map each archive prefix to the corresponding active
path helper override instead of copying the staging root wholesale. If a
destination already contains useful state, preserve or rename it before the
copy. Neither tar nor rsync can decide which conflicting course, checkpoint, or
credential is authoritative.

Remove the dedicated staging directory with your platform’s approved temporary
file cleanup only after the restored system passes verification.

## Disaster-recovery verification

A successful archive checksum is necessary but not sufficient. Test the
restored system in isolation before returning it to service:

```bash
ed4all fsck
ed4all doctor
```

Then verify that:

- expected courses and run histories are present;
- credentials are loaded from the intended private store;
- manifests, blobs, and indexes refer to available artifacts;
- required model and dependency stores have been restored or rebuilt; and
- a synthetic, non-private smoke workflow can read and write the restored data
  root without modifying production state.

Record the archive identity, verification result, restore destination, software
revision, and smoke-test result in the operator’s private recovery log. A backup
is proven only by a successful restore exercise.

## Failure semantics

- Creation reports missing roots but does not treat every absent optional root
  as fatal.
- Verification fails closed on archive corruption, manifest mismatch, unsafe
  extraction paths, and LibV2 consistency errors.
- Verification never changes the archive or live data.
- Restore conflict resolution is manual and explicit.
- No backup failure authorizes falling back to an older unverified archive.

For container-specific volume exclusions, see [Docker deployment](docker.md).
For a redacted diagnostic artifact that is safer to share with a maintainer,
see [Support bundles](support-bundle.md).
