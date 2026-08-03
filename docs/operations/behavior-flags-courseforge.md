# Courseforge behavior flags

Public reference for Courseforge-owned environment controls. Provider and model
selection is also governed by [licensing policy](../LICENSING.md). Unless a row
says otherwise, two-pass controls apply only when `COURSEFORGE_TWO_PASS` is on.

Boolean controls accept `1`, `true`, `yes`, or `on` case-insensitively. Falsey
tokens are `0`, `false`, `no`, and `off`. Numeric resolvers use the stated
default when input is blank, malformed, or outside the accepted range.

## Pipeline and rendering

| Flag | Default | Current contract |
|---|---|---|
| `COURSEFORGE_TWO_PASS` | off | Runs outline, inter-tier validation, and rewrite phases instead of single-pass generation. Other values leave two-pass routing off. Source: `Courseforge/scripts/blocks.py`, `MCP/core/workflow_runner.py`. |
| `COURSEFORGE_ALLOW_TEMPLATE_EMITTER` | off | Permits deterministic template output when LLM authoring was requested. Unset makes the content-authorship gate block that condition. Source: `lib/validators/content_authorship.py`. |
| `COURSEFORGE_PROVIDER` | unset | Selects `anthropic`, `together`, `local`, or another registered OpenAI-compatible content provider. Unset uses deterministic content generation. Provider/model licensing applies. Source: `Courseforge/generators/_provider.py`, `MCP/tools/pipeline_tools.py`. |
| `COURSEFORGE_RUN_ID` | empty | Adds a run identifier to emitted block provenance. Empty omits it. Source: `Courseforge/scripts/rendering/generate_course.py`. |
| `COURSEFORGE_BLOCK_ROUTING_PATH` | `Courseforge/config/block_routing.yaml` | Selects the block routing policy. A missing file yields an empty policy and resolver defaults; an invalid file fails validation. Source: `Courseforge/router/policy.py`. |
| `COURSEFORGE_EMIT_BLOCKS` | off | Emits canonical `blocks[]`, provenance, content hashes, and block IDs in page JSON-LD. Other values preserve the standard page metadata shape. Source: `Courseforge/scripts/blocks.py`, `Courseforge/scripts/rendering/generate_course.py`. |
| `COURSEFORGE_PAGE_MATHJAX` | on | Includes MathJax configuration and the public loader in rendered pages. Explicit falsey tokens omit it; invalid values retain the enabled default. Source: `Courseforge/scripts/rendering/generate_course.py`. |
| `COURSEFORGE_ENFORCE_JSONLD_SCHEMA` | off | Makes invalid emitted JSON-LD fail page generation. Unset reports validation without blocking. Source: `Courseforge/scripts/rendering/generate_course.py`. |
| `COURSEFORGE_ENFORCE_SHACL` | off | Makes SHACL violations fail page generation. Unset reports violations without blocking. Source: `Courseforge/scripts/rendering/generate_course.py`. |

## Outline tier

| Flag | Default | Current contract |
|---|---|---|
| `COURSEFORGE_OUTLINE_PROVIDER` | `local` | Selects `local`, `together`, `anthropic`, or another registered OpenAI-compatible provider. Invalid providers fail during provider construction. Licensing applies. Source: `Courseforge/generators/outline/_outline_provider.py`. |
| `COURSEFORGE_OUTLINE_MODEL` | `qwen2.5:7b-instruct-q4_K_M` | Overrides the model used by the outline provider. The served endpoint must expose the selected identifier. Licensing applies. Source: `Courseforge/generators/outline/_outline_provider.py`. |
| `COURSEFORGE_OUTLINE_GRAMMAR_MODE` | provider-derived | Selects `gbnf`, `json_schema`, `json_object`, or `none`; unset chooses a provider-compatible mode. Unsupported values use provider detection. Source: `Courseforge/generators/outline/_outline_provider.py`. |
| `COURSEFORGE_OUTLINE_N_CANDIDATES` | `3` | Sets the positive candidate count per block. Invalid or non-positive values use `3`; routing policy overrides take precedence. Source: `Courseforge/router/router.py`. |
| `COURSEFORGE_OUTLINE_REGEN_BUDGET` | `10` | Caps outline regeneration attempts after validator rejection. Invalid or negative values use `10`; exhaustion marks the block for escalation. Source: `Courseforge/router/router.py`. |
| `COURSEFORGE_OUTLINE_MAX_CHUNKS` | `8` | Caps source chunks included in one outline prompt. Invalid or non-positive values use `8`. Source: `Courseforge/generators/outline/_outline_provider.py`. |
| `COURSEFORGE_OUTLINE_MAX_TOKENS` | `4096` | Sets the positive outline response-token ceiling. Invalid or non-positive values use `4096`; per-call and routing-policy values take precedence. Source: `Courseforge/generators/outline/_outline_provider.py`, `Courseforge/router/router.py`. |
| `COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE` | on | Fails an outline call when reported prompt usage indicates input truncation. Explicit falsey tokens disable it; missing usage does not fail the call. Source: `Courseforge/generators/outline/_outline_provider.py`. |
| `COURSEFORGE_OUTLINE_DISPATCH_BREAKER` | `5` | Stops the phase after this many consecutive dispatch failures; `0` disables the breaker. Invalid or negative values use `5`. Source: `MCP/tools/pipeline_tools.py`. |
| `COURSEFORGE_OUTLINE_CONCURRENCY` | `1` | Sets positive parallel outline dispatches. Invalid or non-positive values use `1`. Source: `MCP/tools/pipeline_tools.py`, `Courseforge/router/router.py`. |
| `COURSEFORGE_BLOCK_PLAN_CONCURRENCY` | inherits outline concurrency | Sets positive parallel block-plan warmups. Invalid or non-positive values fall through to `COURSEFORGE_OUTLINE_CONCURRENCY`, then `1`. Source: `MCP/tools/pipeline_tools.py`. |
| `COURSEFORGE_OUTLINE_CHECKPOINT` | on | Writes resumable per-block outline checkpoints and removes the sidecar after successful finalization. Explicit falsey tokens disable it; invalid values retain the enabled default. Source: `MCP/tools/pipeline_tools.py`, `lib/generation/llm_checkpoint.py`. |

## Rewrite tier

| Flag | Default | Current contract |
|---|---|---|
| `COURSEFORGE_REWRITE_PROVIDER` | `anthropic` | Selects `anthropic`, `together`, `local`, `claude_session`, or another registered compatible provider. Unsupported providers fail during construction. Licensing applies. Source: `Courseforge/generators/rewrite/_rewrite_provider.py`. |
| `COURSEFORGE_REWRITE_MODEL` | provider-derived (`claude-sonnet-4-6` for Anthropic) | Overrides the rewrite model. The served endpoint must expose the identifier. Licensing applies. Source: `Courseforge/generators/rewrite/_rewrite_provider.py`. |
| `COURSEFORGE_REWRITE_MAX_TOKENS` | `4096` provider / `6144` router fallback | Sets the positive rewrite response-token ceiling. Invalid or non-positive values use the owning resolver's default; routing policy and per-call values take precedence. Source: `Courseforge/generators/rewrite/_rewrite_provider.py`, `Courseforge/router/router.py`. |
| `COURSEFORGE_REWRITE_REGEN_BUDGET` | `10` | Caps rewrite remediation attempts after validator rejection. Invalid or negative values use `10`; exhaustion returns the marked best effort. Source: `Courseforge/router/router.py`. |
| `COURSEFORGE_REWRITE_CONCURRENCY` | `1` (`2` for an unpinned batched cloud lane) | Sets positive parallel rewrite work. Invalid values use the active lane's default. Source: `Courseforge/router/router.py`, `Courseforge/generators/rewrite/_rewrite_batch.py`. |
| `COURSEFORGE_REWRITE_BATCH` | provider-derived | Enables batched rewrite dispatch for supported cloud providers; unset enables it only for the cloud lane. Explicit falsey tokens disable it. Source: `Courseforge/generators/rewrite/_rewrite_batch.py`. |
| `COURSEFORGE_REWRITE_BATCH_SIZE` | `3` | Sets the positive blocks-per-request cap for batched rewriting. Invalid or non-positive values use `3`. Source: `Courseforge/generators/rewrite/_rewrite_batch.py`. |
| `COURSEFORGE_REWRITE_BATCH_K` | `1` | Requests candidate count for each batched block. Values are clamped to `1`; invalid or non-positive values use `1`. Source: `Courseforge/generators/rewrite/_rewrite_batch.py`. |
| `COURSEFORGE_REWRITE_CHECKPOINT` | on | Writes resumable per-block rewrite checkpoints and removes the sidecar after successful finalization. Explicit falsey tokens disable it; invalid values retain the enabled default. Source: `MCP/tools/pipeline_tools.py`, `lib/generation/llm_checkpoint.py`. |
| `COURSEFORGE_REWRITE_HTML_REPAIR` | off | Repairs balanced HTML structure deterministically before final rewrite output. Truthy tokens enable it; unrecoverable fragments remain unchanged for validation to reject. Source: `lib/utils/html_balance.py`, `MCP/tools/pipeline_tools.py`. |
| `ED4ALL_REWRITE_FIT_WINDOW` | off | Trims rewrite grounding to fit the configured serving window. Truthy tokens enable it; other values leave prompt assembly unchanged. Source: `Courseforge/generators/rewrite/_rewrite_fit_window.py`. |
| `ED4ALL_REWRITE_NUM_CTX` | `8192` | Sets the positive rewrite serving-window budget. Invalid or non-positive values use `8192`. Source: `Courseforge/generators/rewrite/_rewrite_fit_window.py`. |
| `ED4ALL_REWRITE_TRUNCATION_TRIPWIRE` | on | Fails rewrite calls when local or server usage signals show prompt truncation. Explicit falsey tokens disable it; absent usage leaves the server-side arm inactive. Source: `Courseforge/generators/rewrite/_rewrite_fit_window.py`. |
| `ED4ALL_REWRITE_MAX_ESCALATION_SHARE` | `0.5` | Fails a sufficiently large rewrite phase when marked blocks exceed this share. Values above `1` clamp to `1`; negative or invalid values use `0.5`. Source: `MCP/tools/pipeline_tools.py`. |
| `COURSEFORGE_CURIE_DETERMINISTIC` | off | Removes CURIEs from the model prompt and deterministically stamps vocabulary-resolved CURIEs after generation. Truthy tokens enable it; unresolved CURIEs are never fabricated. Source: `Courseforge/generators/rewrite/_rewrite_fit_window.py`, `Courseforge/generators/rewrite/_rewrite_provider.py`. |
| `COURSEFORGE_CURIE_PRESERVE_SKIP_WHEN_POSTMINT` | off | Immediately postmints missing enforceable CURIEs instead of spending a CURIE-preservation retry. Truthy tokens enable it; deterministic CURIE mode takes precedence. Source: `Courseforge/generators/rewrite/_rewrite_provider.py`. |

## Planning and textbook synthesis

| Flag | Default | Current contract |
|---|---|---|
| `COURSEPLANNER_PROVIDER` | `anthropic` | Selects the course-planning provider from Anthropic or the registered OpenAI-compatible providers. Unsupported values fail during construction. Licensing applies. Source: `Courseforge/generators/outline/_outliner_provider.py`. |
| `COURSEPLANNER_MODEL` | provider-derived | Overrides the course-planning model identifier. The served endpoint must expose the identifier. Licensing applies. Source: `Courseforge/generators/outline/_outliner_provider.py`. |
| `TEXTBOOK_SYNTHESIS_PROVIDER` | `anthropic` for direct construction; workflow default `local` | Selects the three-stage textbook synthesis provider. Explicit values override workflow defaults; unsupported values fail during construction. Licensing applies. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`, `MCP/core/workflow_runner.py`. |
| `TEXTBOOK_SYNTHESIS_MODEL` | provider-derived | Overrides the textbook synthesis model identifier. The served endpoint must expose the identifier. Licensing applies. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`. |
| `TEXTBOOK_SYNTHESIS_NUM_CTX` | `4096` | Sets the positive textbook synthesis context budget, falling back to the grounded-answer context setting and then `4096`. Invalid or non-positive values use that chain. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`. |
| `TEXTBOOK_SYNTHESIS_GRAMMAR_MODE` | provider-derived | Selects constrained decoding for textbook synthesis. Accepted modes match the outline grammar modes; invalid values use provider detection. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`. |
| `TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS` | `300` | Sets the positive request timeout in seconds. Invalid or non-positive values use `300`. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`. |
| `TEXTBOOK_SYNTHESIS_MAX_TOKENS` | `4096` | Sets the positive response-token ceiling. Invalid or non-positive values use `4096`; explicit call arguments take precedence. Source: `Courseforge/generators/outline/_textbook_synthesis_provider.py`. |

## Generation techniques

| Flag | Default | Current contract |
|---|---|---|
| `COURSEFORGE_BEST_OF_N` | `1` | Supplies the positive candidate count for either tier unless its tier-specific value is set. Invalid or non-positive values use `1`. Source: `lib/generation/technique_modes.py`. |
| `COURSEFORGE_REWRITE_N_CANDIDATES` | inherits `COURSEFORGE_BEST_OF_N` | Overrides the rewrite candidate count with a positive integer. Invalid values fall through to the umbrella value, then `1`. Source: `lib/generation/technique_modes.py`. |
| `COURSEFORGE_BEST_OF_N_SELECT_BY` | `gate_pass` | Selects `gate_pass` or `entailment_argmax`. Unsupported values use `gate_pass`; only validator-passing candidates are eligible. Source: `Courseforge/router/router.py`. |
| `COURSEFORGE_REWRITE_EARLY_EXIT` | off | Under `entailment_argmax`, stops after the first validator-passing candidate that clears the entailment floors. Truthy tokens enable it; otherwise all candidates are considered. Source: `Courseforge/router/router.py`. |
| `COURSEFORGE_SELF_VERIFY` | off | Enables the self-verification generation pass. Only canonical truthy tokens enable it. Source: `lib/generation/technique_modes.py`, `Courseforge/router/router.py`. |
| `COURSEFORGE_REFINE_ROUNDS` | `0` | Sets bounded positive refinement rounds. Invalid or non-positive values use `0`. Source: `lib/generation/technique_modes.py`, `Courseforge/router/router.py`. |
| `COURSEFORGE_CHUNK_SCOPED` | off | Restricts generation inputs to the selected chunk scope. Only canonical truthy tokens enable it. Source: `lib/generation/technique_modes.py`, `Courseforge/router/router.py`. |

## Licensing and related controls

Provider and model flags select systems whose terms determine whether generated
course content may later participate in training. Review
[`docs/LICENSING.md`](../LICENSING.md) before changing those values. Root-owned
generation controls are listed in [the root registry](behavior-flags.md).
