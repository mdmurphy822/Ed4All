# Development-token statistics

The README's development-token footer is generated from local Claude and Codex
session logs that were run from the Ed4All repository. It is a fun measure of
development collaboration, not a measure of code quality, cost, authorship, or
model efficiency.

```bash
python3 scripts/ops/update_development_tokens.py
python3 scripts/ops/update_development_tokens.py --check
```

The updater deduplicates Claude streaming snapshots by session and message ID.
For Codex it uses the final cumulative token count in each session and
deduplicates active and archived copies by session ID. It writes only numeric
token categories, session and recorded-user-turn counts, a UTC timestamp, and a
fixed public scope label to
`docs/reference/development-token-stats.json`. Prompts, responses, session IDs,
usernames, hostnames, and local paths are never written to the repository.

## Metric definitions

- **Average tokens/session** is observed tokens divided by counted sessions.
- **Average session span** is the mean time between each session's first and
  last timestamp. It is not active keyboard time: pauses and background work
  inside that span remain included.
- **Input/read** is fresh input plus cache creation and cache reads for Claude,
  and total input for Codex. Claude exposes its cache fields as additive token
  categories. Codex reports cached input as a subset of input, so it must not be
  added to the Codex total a second time.
- **Output/write** is provider-reported output. Codex reasoning output is a
  reported subset of output, not an additional category. Claude's local logs do
  not expose a separate reasoning-token count.
- **Recorded user turns** counts explicit Claude user-role/prompt-history rows
  and explicit Codex user-message/turn records without retaining their content.
  These provider surfaces are similar but not identical. Historical coverage
  can also differ when prompt metadata outlives token-bearing transcripts, so
  the number is an observed activity indicator rather than a complete or
  cross-provider-normalized prompt count.

The README presents the metrics in one centered, native HTML table immediately
after the complete **What Ed4All does** section. Tinted KPI cells and labeled
section bands provide visual grouping, while every value remains text in the
table for accessibility and plain-renderer resilience. The display uses no
images, badges, SVG, Mermaid, custom CSS, or collapsible sections.

## Maintained lines of code and documentation

The dashboard also counts newline-delimited lines in files returned by
`git ls-files`. Each file belongs to exactly one category, with tests taking
precedence over docs, then tooling/config, source, and other text. Binary files
and paths for runtime state, generated course artifacts, build outputs, and
vendored dependencies are excluded. The result is a maintained-repository LOC
indicator, not a language-aware count of executable statements. Newly created
files enter this count after they are staged; rerun the updater after staging
the tracker itself.

## Include an aggregate from another machine

Run the updater from the Ed4All checkout on that machine and write a
numeric-only, local export:

```bash
python3 scripts/ops/update_development_tokens.py \
  --export-only <path-to-local-export.json>
```

The resulting file has this shape:

```json
{
  "schema_version": 2,
  "sources": {
    "claude": {
      "tokens": 0,
      "sessions": 0,
      "user_turns": 0,
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
      "cached_input_tokens": 0,
      "reasoning_output_tokens": 0,
      "duration_seconds": 0
    },
    "codex": {
      "tokens": 0,
      "sessions": 0,
      "user_turns": 0,
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
      "cached_input_tokens": 0,
      "reasoning_output_tokens": 0,
      "duration_seconds": 0
    }
  }
}
```

Then merge it while updating:

```bash
ED4ALL_TOKEN_STATS_EXPORT=<path-to-local-export.json> \
  python3 scripts/ops/update_development_tokens.py
```

The same environment variable must be present for `--check`. Extra export
fields are ignored, but the export should still contain no raw session data.
Copying this small aggregate is the supported direction-independent workflow;
the updater does not open an SSH connection or publish a machine address.

## Pre-push freshness check

Install the local hook once:

```bash
python3 scripts/ops/update_development_tokens.py --install-hook
```

The hook runs `--check` and rejects a push when the local numeric aggregates or
README footer are stale. It never updates or amends a commit. The installer
refuses to overwrite a pre-existing hook; in that case, add the documented
`--check` command to the existing hook manually.
