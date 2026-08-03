# Create a sanitized support bundle

`ed4all support-bundle` packages diagnostics for a maintainer without walking
course archives or Courseforge exports. Redaction reduces disclosure risk, but
it cannot prove that every included log or malformed file is safe. Always
inspect the archive inside your trust boundary before sharing it.

## Create the bundle

Bundle the newest private run and current environment diagnostics:

```bash
ed4all support-bundle
```

Select a run or output path explicitly:

```bash
ed4all support-bundle \
  --run-id <PRIVATE_RUN_ID> \
  --output <PRIVATE_BUNDLE_PATH>
```

The command prints the archive path and size. Missing optional state, such as a
run directory or GUI log directory, becomes a manifest warning so the remaining
diagnostics can still be assembled.

## Included diagnostics

| Archive path | Contents |
|---|---|
| `doctor.json` | Post-mortem diagnostics for an explicit run, or current environment diagnostics when no run is selected. |
| `run/<private-run-id>/` | One run's checkpoints, resource telemetry, usage records, decisions, and audit files. |
| `gui-logs/` | Available GUI console logs. |
| `manifest.json` | Included-member sizes and SHA-256 hashes plus warnings for missing or excluded material. |
| `captures/` | Decision captures, only when `--include-captures` is explicitly supplied. |

When no run is specified, the most recently modified run directory is selected.
This is a convenience rule, not proof that the selected run is the one you
intended; verify `manifest.json`.

Course archives and Courseforge exports are excluded by construction because
the command never walks those trees.

## Redaction boundary

The bundle drops recognized secret-only files, including common environment
files and private-key or certificate formats. In valid JSON files, values under
secret-shaped keys are replaced with `***REDACTED***`; empty secret values stay
`null` so configuration presence remains visible.

Redaction is deliberately conservative but not universal:

- plaintext logs and non-JSON files are copied as-is;
- malformed JSON is copied as-is because its structure cannot be inspected
  reliably;
- an unrecognized credential filename or key can evade pattern-based checks;
- run metadata can reveal private identifiers, model choices, timings, and
  infrastructure characteristics; and
- generated rationales can quote private source material.

Review both the archive members and their contents before transfer. Share a
bundle only through an approved private channel.

## Decision captures

Captures are excluded by default. Include them only when a maintainer needs the
decision trail and the source-content risk has been reviewed:

```bash
ed4all support-bundle \
  --run-id <PRIVATE_RUN_ID> \
  --include-captures \
  --output <PRIVATE_BUNDLE_PATH>
```

The command records an explicit warning in the manifest when captures are
included. Opting in does not sanitize free-text rationales.

## Verify before sharing

1. Open `manifest.json` and confirm the selected private run.
2. Review every warning and every included archive path.
3. Search extracted content for credentials, source text, personal data,
   course identifiers, local paths, host details, and endpoint information.
4. Confirm that decision captures were included only when intended.
5. Transfer the bundle through a private, access-controlled channel.

Do not use a support bundle as a backup. A backup intentionally includes a
broader restore surface and has different confidentiality requirements; see
[Backup and restore](backup-restore.md).
