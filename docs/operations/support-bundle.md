# `ed4all support-bundle` (OP1)

Assemble a **redacted diagnostics tarball** to hand a maintainer when a run
misbehaves. A support bundle carries the *diagnostic* surface of a deployment —
never the confidential surface, and never course content.

```bash
# Newest run + live-env doctor groups
ed4all support-bundle

# A specific run (adds the doctor POST-MORTEM group over its checkpoints/VRAM)
ed4all support-bundle --run-id WF-20260707-abc12345 -o /tmp/bundle.tar.gz

# Include decision captures (OFF by default — see "Secret handling")
ed4all support-bundle --run-id WF-... --include-captures
```

Prints the bundle path + size on completion.

## What's IN

| Member | Source | Notes |
|--------|--------|-------|
| `doctor.json` | `ed4all doctor`, run **in-process** at bundle time | Post-mortem group with `--run-id` (reads the run's checkpoints + VRAM trajectory off disk); otherwise the live-env `gpu` / `window` / `environment` groups. Never raises — a failure degrades to `{"error": ...}`. |
| `run/<run_id>/…` | `runtime/state/runs/<id>/` | Checkpoints, `vram_trajectory.jsonl`, `llm_usage.jsonl`, decisions, audit. With no `--run-id` the **newest** run dir is chosen. |
| `gui-logs/…` | `runtime/state/gui/logs/*.log` | The per-run consoles tailed by the GUI. |
| `captures/…` | `runtime/training-captures/**/*.jsonl` | **Only** under `--include-captures`. |
| `manifest.json` | generated | Every included file with its `size` + `sha256`, plus a `warnings[]` list recording anything withheld. |

## What's OUT

* **Course content** — `LibV2/courses/`, `Courseforge/exports/`. These trees are
  never walked, so they are excluded by construction.
* **Secret-only files** — `secrets.json`, `.env`, `.env.rendered`, `*.key`,
  `*.pem`, `*.crt` … are dropped outright wherever they appear (e.g. a
  `secrets.json` that slipped into a run dir). Each drop is logged as a
  `warnings[]` entry in the manifest.

## Secret handling

Defense in depth beyond the wave-1 settings/secrets sidecar split:

1. **Secret-only files are dropped** (never bundled). `secrets.json` and
   `settings.json` env values never enter a bundle.
2. **Every bundled `*.json` is walked** and any secret-shaped key
   (`*_API_KEY` / `*_KEY` / `*_TOKEN` / `*_SECRET` / `*_PASSWORD`, and the bare
   `authorization` / `password` / `secret` / `token` / `api_key` keys) has its
   value replaced with `"***REDACTED***"` (a set-but-empty value stays `null` so
   the "is it configured?" signal survives). This scrubs a stray credential in a
   config snapshot.
3. **Decision captures are opt-in.** Capture *rationales* interpolate real
   signals and can quote verbatim source text, so `runtime/training-captures/` is
   excluded unless you pass `--include-captures` — which also prints a review
   warning and records it in the manifest.

Even so, plaintext GUI logs are bundled as-is; skim a bundle before sharing it
outside your trust boundary.

## Related

* `ed4all doctor` — the diagnostics whose JSON is embedded (`cli/commands/doctor.py`).
* `ed4all backup` — a **complete, restore-able** snapshot (includes secrets;
  `0600`). See [`backup-restore.md`](backup-restore.md).
