# License-clean pipeline run

Use this runbook when Ed4All outputs may become courseware, training pairs, or
model artifacts. It keeps every authoring and synthesis surface on an explicitly
selected, license-cleared provider and fails loudly when that provider is not
available.

> **Corpus privacy is invariant.** Source books, converted content, course
> identifiers, generated courses, training pairs, adapters, and run captures
> are private. Keep them outside the repository and do not publish them.

The authoritative licensing policy is [`docs/LICENSING.md`](../LICENSING.md).
Review it, the selected model license, and the provider terms before every
production run. This document is an operational checklist, not legal advice.

## Supported provider posture

| Training-pair provider | Operational posture |
|---|---|
| `local` | Recommended license-clean route. Use an OpenAI-compatible server with a model whose license permits the intended output use. |
| `together` | Hosted OSS alternative. It is paid and networked; verify the current hosted-model license and provider terms before use. |
| `mock` | Plumbing tests only. Its template output is not a shippable training corpus. |
| `claude_session` | Not a license-clean default. It fails closed unless `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS` is explicitly truthy and requires a dispatcher. Use only with a separate agreement permitting derivative training. |
| `anthropic` | Removed from training-pair synthesis and rejected unconditionally. There is no acknowledgment bypass. |
| `nvidia` | Rejected unconditionally for training-pair synthesis. There is no acknowledgment bypass. |

The accepted truthy acknowledgment values for `claude_session` are `1`,
`true`, `yes`, and `on` (case-insensitive). The acknowledgment records an
operator decision; it does not change provider terms or make the output
license-clean.

## 1. Install and prepare

1. Install Ed4All and the required extras using the maintained
   [installation guide](installation.md). Do not vendor dependency source,
   environments, model weights, or caches into this repository.
2. Place the private input corpus in an operator-controlled location outside
   the repository.
3. Start an OpenAI-compatible local inference server. Choose its endpoint and
   model for the current deployment; this runbook intentionally does not pin a
   host, port, device, or model.
4. Confirm the selected model license permits the intended courseware and
   derivative-training use.
5. Confirm the server reports the exact model you intend to use. Staged local
   synthesis requires an explicit `LOCAL_SYNTHESIS_MODEL` and rejects a
   conflicting generic model selector.

## 2. Pin every authoring surface

Do not rely on inherited shell state or provider defaults. Pin the local route
explicitly for each surface whose generated text can flow into the course or
training corpus:

```bash
export TEXTBOOK_SYNTHESIS_PROVIDER=local
export COURSEPLANNER_PROVIDER=local
export COURSEFORGE_PROVIDER=local
export COURSEFORGE_TWO_PASS=true
export COURSEFORGE_OUTLINE_PROVIDER=local
export COURSEFORGE_REWRITE_PROVIDER=local
export CURRICULUM_ALIGNMENT_PROVIDER=local
export TRAINFORGE_ASSESSMENT_PROVIDER=local
export TRAINFORGE_SYNTHESIS_PROVIDER=local

export LOCAL_SYNTHESIS_BASE_URL="<OPENAI_COMPATIBLE_BASE_URL>"
export LOCAL_SYNTHESIS_MODEL="<LICENSE_CLEARED_MODEL_ID>"
```

Provide credentials through the deployment's secret manager if the local
endpoint requires authentication. Never write keys into tracked files, command
examples, run reports, or captured logs.

SemantiK conversion settings are deployment-specific. Select the current local
GLM-OCR, SDK, enrichment, and heading-judge path described in the
[pipeline invocation guide](pipeline-invocation.md); do not substitute a
historical conversion stack.

## 3. Validate, then run

First resolve the workflow without dispatching work:

```bash
ed4all run textbook-to-course \
  --corpus "<PRIVATE_CORPUS_PATH>" \
  --course-name "<PRIVATE_COURSE_NAME>" \
  --provider local \
  --mode local \
  --dry-run
```

Then run the same command without `--dry-run`:

```bash
ed4all run textbook-to-course \
  --corpus "<PRIVATE_CORPUS_PATH>" \
  --course-name "<PRIVATE_COURSE_NAME>" \
  --provider local \
  --mode local
```

Follow the stage, resume, stop, and timeout procedures in
[`pipeline-invocation.md`](pipeline-invocation.md). Run the validation gates
defined in `config/workflows.yaml`; stop at the first failed gate and fix its
cause. Never lower a threshold, downgrade severity, or switch providers as an
implicit fallback.

Before promotion, inspect the decision-capture and provider provenance for all
authoring and synthesis calls. A missing event, unexpected provider, unexpected
model, or server identity mismatch invalidates the run until investigated.

## Hosted OSS alternative

Together remains the implemented hosted alternative for training-pair
synthesis:

```bash
export TRAINFORGE_SYNTHESIS_PROVIDER=together
export TOGETHER_API_KEY="<FROM_SECRET_MANAGER>"
export TOGETHER_SYNTHESIS_MODEL="<LICENSE_CLEARED_HOSTED_MODEL_ID>"
```

This changes only the training-pair provider. Any other authoring surface stays
on the explicitly pinned provider shown above unless its own provider setting
is deliberately changed and reviewed. Because hosted catalogs and terms can
change, re-check both before dispatch rather than copying a historical model
name from documentation.

## Release checklist

- The corpus and every generated artifact remain private.
- Provider and model selections match the approved licensing record.
- No call used `mock`, `claude_session`, `anthropic`, or an unreviewed hosted
  provider.
- Decision captures account for every LLM call and contain no private content
  intended for publication.
- All required validation gates pass without relaxed thresholds or fallbacks.
- Only code and public-safe documentation are staged for GitHub.

## Related documentation

- [Licensing and provider policy](../LICENSING.md)
- [Installation and dependencies](installation.md)
- [Pipeline invocation, stop, and resume](pipeline-invocation.md)
- [Validation gates](../validation/gates.md)
