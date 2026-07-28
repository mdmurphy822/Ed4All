# Fresh training-synthesis reset

Use the scoped reset tool when a new training-pair synthesis pass must not
reuse any disposition, output, journal, telemetry, or contract fingerprint
from an earlier pass. The tool does not start a model service or resume a
workflow.

The default invocation is read-only:

```bash
python3 scripts/prepare_fresh_training_synthesis.py \
  --workflow-state <workflow-state.json> \
  --training-specs-dir <training-specs> \
  --runs-dir <run-state-root>/runs \
  --upstream-input <chunks.jsonl> \
  --upstream-input <objectives.json>
```

Review the complete JSON plan, especially `artifacts_to_archive`,
`artifact_sha256`, `preserved_input_sha256`, and `phases_to_reset`. Apply that
reviewed scope by repeating the command with `--apply`. If any planned
artifact or preserved input changed between review and apply, the operation
fails before its first filesystem mutation.

## Result

The applied reset:

- moves pair outputs, `.in_progress` files, pair and generation checkpoints,
  terminal-rejection rows, the provider cache, seat-recovery evidence,
  synthesis telemetry, pilot outputs, the phase checkpoint, and a stale stop
  sentinel into a timestamped evidence archive;
- records a SHA-256 for every moved artifact and makes the evidence archive
  read-only;
- preserves and fingerprints upstream chunks, objectives, assessments, and
  other non-synthesis inputs;
- removes only `training_synthesis` and already-observed downstream phase
  state, leaving upstream phase outputs byte-for-byte unchanged; and
- writes `training_specs/.synthesis_fresh_start.json` with a new
  `fresh_start_id`, preserved-input hashes, and the evidence-manifest hash.

Synthesis code that requires a fresh pass must call
`Trainforge.synthesis_fresh_start.require_fresh_start_marker()` before loading
any resume state. The check fails loudly when the marker is absent, its
identity differs from the requested run, a preserved input changed, or any
old synthesis cache/checkpoint/output reappeared. New outputs created after
that check are normal; the check belongs at the pre-load boundary.

The archive is diagnostic evidence and is excluded from all future synthesis.
Do not move its files back into `training_specs`; start another scoped reset if
a later pass also needs a clean identity.
