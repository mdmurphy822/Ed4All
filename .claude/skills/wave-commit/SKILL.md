---
name: wave-commit
description: Author a Wave-N commit message using the Ed4All convention. Use when the user asks to commit changes following the project's wave-numbered convention.
disable-model-invocation: true
---

# wave-commit

Author a commit message following the Ed4All `Wave NN:` convention and
create the commit.

## Convention

Subject line:

- `Wave NN: <subject>` — one-shot wave.
- `Wave NN Phase X: <subject>` — when the user is mid-wave and a phase
  number applies.

The next wave number comes from the latest existing wave in `git log`:

```bash
git log --oneline | grep -oP 'Wave \K\d+' | sort -rn | head -1
```

Add 1 for a new wave, or keep the same wave + bump the phase if the user
indicates they are continuing the in-progress wave.

Body:

- 1–3 short paragraphs explaining **what changed and why** (not a file
  list — the diff already shows that).
- Reference relevant CLAUDE.md sections, behavior flags, validators, or
  schemas when the change touches them.
- End with the trailer:

  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```

## Reference examples (from `git log`)

- `07c3f26 Wave 81: Trainforge chunker honors data-cf-template-type`
- `0579d7b Wave 81: bake Wave 75-78 enrichment into Trainforge emit pipeline`
- `fc2675d Wave 81: auto-extract retag vocabularies for full CO coverage`

Keep the subject under ~72 chars and use imperative mood ("honors",
"bake", "auto-extract").

## Workflow

1. Inspect the staged state:

   ```bash
   git status
   git diff --staged
   ```

   If nothing is staged, ask the user which files to stage (or
   `git add` specific files explicitly — never `git add -A` blindly).

2. Determine the next wave / phase number:

   ```bash
   git log --oneline | grep -oP 'Wave \K\d+' | sort -rn | head -1
   ```

   If the user said "continue wave NN" / "phase 2 of wave NN", reuse the
   wave number and add `Phase X`.

3. Draft the commit message in your head, then commit using a heredoc to
   preserve formatting:

   ```bash
   git commit -m "$(cat <<'EOF'
   Wave NN: <subject>

   <body paragraph 1>

   <body paragraph 2 if needed>

   Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
   EOF
   )"
   ```

4. Confirm:

   ```bash
   git status
   git log --oneline -1
   ```

## Constraints

- Never `--amend` an existing commit; always create a new one. If a
  pre-commit hook fails, fix the issue, re-stage, and create a fresh
  commit (the previous commit did **not** happen on hook failure, so
  amending would clobber the prior commit).
- Never `--no-verify` or `--no-gpg-sign` unless the user explicitly
  asks.
- Never push.
- Never stage `.env`, credentials, or large binaries. Add files by name
  rather than `git add -A`.
