# Trainforge behavior flags

Public registry for Trainforge-owned environment configuration. Values shown as
“off” use the owning resolver's established falsey/default path; do not infer a
new fallback from this summary. Implementation and tests remain authoritative.

Truthy boolean resolvers accept `1`, `true`, `yes`, and `on` unless a row says
otherwise. Invalid operator input follows the documented parse-with-fallback
rule only where stated. Provider and model choices remain subject to
[`docs/LICENSING.md`](../LICENSING.md).

`TRAINFORGE_SYNTHESIS_RUN_CONTRACT_SHA256` and
`TRAINFORGE_SYNTHESIS_CONTRACT_COMPONENTS_SHA256` are generated outputs, not
configuration flags.

## Chunk and metadata emission

| Variable | Default and current contract |
|---|---|
| `TRAINFORGE_ALIGN_CHUNKS_MODEL` | Model for direct teaching-role classification. Precedence: `--llm-model`/kwarg, environment, built-in default. |
| `TRAINFORGE_CONTENT_HASH_IDS` | Off. Truthy uses content-hash chunk IDs stable across rechunking. |
| `TRAINFORGE_SCOPE_CONCEPT_IDS` | Off. Truthy scopes concept IDs as `{course_id}:{slug}`. |
| `TRAINFORGE_PRESERVE_LO_CASE` | Off. Truthy preserves emitted learning-outcome reference case. |
| `TRAINFORGE_VALIDATE_CHUNKS` | Off. Truthy validates every chunk against `schemas/knowledge/chunk_v4.schema.json`. |
| `TRAINFORGE_ENFORCE_CONTENT_TYPE` | Off. Truthy enforces the canonical content-type enum. |
| `TRAINFORGE_STRICT_EVIDENCE` | Off. Truthy removes the fallback-provenance evidence arm. |
| `TRAINFORGE_SOURCE_PROVENANCE` | Off. Truthy copies chunk `source.source_references[]` into evidence arms. |
| `TRAINFORGE_PROVENANCE_CORPUS` | Unset. Test-only absolute path consumed by `Trainforge/tests/test_provenance.py`; unset skips the local-corpus checks. Never commit its value. |
| `TRAINFORGE_REQUIRE_EMBEDDINGS` | Off. Missing optional embedding dependencies produce warning-only validator results; truthy raises `EmbedderDepsMissing`. |
| `TRAINFORGE_REQUIRE_BERT_ENSEMBLE` | Off compatibility switch. Truthy makes unavailable retired Bloom-classifier members raise `BertEnsembleDepsMissing`; it does not reactivate or provision them. |
| `TRAINFORGE_SEED_TECH_CONCEPTS` | Off for bare calls; pipeline workflows setdefault it on. Adds canonical technical anchors found in chunk text. |
| `TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS` | Off for bare calls; pipeline workflows setdefault it on. Removes domain-agnostic scaffolding tags before graph minting. |
| `TRAINFORGE_CHUNK_LOCAL_TAGS` | Off. Truthy derives tags from each chunk rather than shared page concepts. Deliberately not auto-enabled. |
| `TRAINFORGE_PAGE_CONCEPT_FALLBACK` | Off for bare calls; pipeline workflows setdefault it on. Deterministically fills empty page concept lists; derivation errors warn and preserve parser output. |
| `TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE` | Off for bare calls; pipeline workflows setdefault it on. Uses whole-word, title-aware chunk typing. |
| `TRAINFORGE_HEADING_SANITY_FILTER` | Off for bare calls; pipeline workflows setdefault it on. Repairs or rejects suspect promoted headings before chunk typing. |
| `TRAINFORGE_CHUNK_OVERLAP_WORDS` | `0`; unset, invalid, or non-positive → `0`. Positive integer prepends bounded verbatim overlap without crossing known flow boundaries. |
| `TRAINFORGE_RELOCATE_STRANDED_HEADINGS` | Off for bare calls; pipeline workflows setdefault it on. Moves a conservative trailing heading marker to its same-flow successor; errors warn and preserve chunks. |
| `TRAINFORGE_DROP_FRONTMATTER` | Off for bare calls; pipeline workflows setdefault it on. Drops high-confidence front matter with content vetoes; classifier errors keep all chunks. |
| `TRAINFORGE_DROP_APPARATUS_DUMPS` | Off. Truthy drops high-confidence answer-key and navigation dumps; classifier errors keep all chunks. |

## Knowledge graph and validation

| Variable | Default and current contract |
|---|---|
| `TRAINFORGE_MERGE_DUPLICATE_CONCEPTS` | Off for bare calls; pipeline workflows setdefault it on. Merges guarded lexical duplicate nodes. |
| `TRAINFORGE_RELATED_FANOUT_CAP` | Unset/`0` means no cap; pipeline workflows setdefault `8`. Positive integers cap per-node `related-to` fan-out after graph construction. |
| `TRAINFORGE_LEXICAL_CONCEPT_SEEDS` | Off for bare calls; pipeline workflows setdefault it on. Deterministically derives seeds only when the authored vocabulary is empty. |
| `TRAINFORGE_FILTER_FRAGMENT_CONCEPTS` | Off for bare calls; pipeline workflows setdefault it on. Filters fragmentary concept candidates while retaining valid multiword concepts. |
| `TRAINFORGE_NORMALIZE_LABELS` | Off for bare calls; pipeline workflows setdefault it on. Applies curated compound splitting before label title-casing. |
| `TRAINFORGE_INTRA_CHUNK_LINKS` | Off for bare calls; pipeline workflows setdefault it on. Adds bounded `related-to` links among grounded co-located concepts; errors leave the graph unchanged. |
| `TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE` | Off. Truthy lowers concept-node minting frequency from two occurrences to one. |
| `TRAINFORGE_COOCCURRENCE_GROUP_BY` | `chunk` for bare calls; accepted values `chunk`, `page`, `section`; pipeline workflows setdefault `page`. |
| `TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK` | Off for bare calls; pipeline workflows setdefault it on. Degenerate page/section grouping steps down to a finer real grouping unit. |
| `TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE` | Off. Truthy requires non-empty provenance on every graph edge. |
| `TRAINFORGE_OBJECTIVE_QUALITY_GATE` | Off for bare calls; pipeline workflows setdefault it on. Filters weak extracted objectives, fills uncovered weeks, and applies the current duration-aware terminal-objective ceiling. |
| `TRAINFORGE_VALIDATE_RULE_OUTPUTS` | Off. Truthy enables warning-severity rule-output regression validation against a supplied baseline. |
| `TRAINFORGE_PREREQ_LO_ADJACENT_ONLY` | Off for bare calls; pipeline workflows setdefault it on. Reduces transitive LO-order closure and demotes text-order conflicts. |
| `TRAINFORGE_PREREQ_DEFINITION_MENTION` | Off. Truthy emits grounded definition-before-use prerequisite edges; dispatch remains absent when off so graph identity is unchanged. |
| `TRAINFORGE_EMIT_TRIG` | Off. Truthy adds a named-graph TriG sibling; JSON output is unchanged. |
| `TRAINFORGE_TARGET_MODELS` | Comma-separated dataset-audit model list; unset/empty uses the implementation default. Whitespace is trimmed. |
| `TRAINFORGE_USE_SHACL_RULES` | Off. Truthy runs the SHACL rule projection for defined-by edges. |
| `TRAINFORGE_EDGE_NLI` | Off for bare calls; pipeline workflows setdefault it on. NLI may retract contradicted chunk-anchored edges; absent text lookup is a no-op and missing dependencies follow strict-mode policy. |
| `TRAINFORGE_EDGE_NLI_MAX_EDGES` | `500`; invalid or non-positive → `500`. No-op unless edge NLI is enabled. |
| `TRAINFORGE_CONTRADICTED_EDGE_POLICY` | Unset means stamp-only; accepted values `decay` and `retract`; pipeline workflows setdefault `decay`. Any other non-empty value raises before mutation. |
| `TRAINFORGE_EVAL_PROGRESS_EVERY` | `25`; invalid/non-positive warns and falls back to `25`. Logging cadence only; excluded from generation identity. |
| `WAVE18_COS_PER_WEEK` | `2`; invalid/non-integer → `2`. Divisor only for the no-terminal-objective week-count fallback; explicit pacing still wins. |
| `TRAINFORGE_SHACL_CLOSED_WORLD` | Off. Truthy merges the closed-world SHACL shapes and reports unminted predicates. |

## Provider and assessment routing

| Variable | Default and current contract |
|---|---|
| `ANTHROPIC_SYNTHESIS_MODEL` | Anthropic model override for remaining curriculum/assessment backends. It cannot restore the removed Anthropic training-pair route; `ANTHROPIC_API_KEY` is still required where used. |
| `TOGETHER_API_KEY` | Required for `--provider together`; missing credentials fail loudly with no mock fallback. |
| `TOGETHER_SYNTHESIS_MODEL` | Together model override; the resolved model is captured per call. License follows the selected model and Together terms. |
| `LOCAL_SYNTHESIS_BASE_URL` | Local OpenAI-compatible base URL. Uses the implementation default when unset; there is no implicit endpoint or lifecycle fallback. Keep real endpoints private. |
| `LOCAL_SYNTHESIS_MODEL` | Local model identity. Staged local synthesis requires an explicit value and exact `/v1/models` match before output creation; mismatch or unavailable identity fails loudly. |
| `LOCAL_SYNTHESIS_API_KEY` | Optional local-server credential. When unset, the client uses its stable local placeholder rather than requiring a secret. |
| `NVIDIA_API_KEY` | Credential for NVIDIA-backed content-generation routes, not Trainforge training-pair synthesis. Missing credentials fail at provider construction. |
| `NVIDIA_BASE_URL` | Optional NVIDIA content-generation endpoint override. Precedence: explicit/YAML, environment, provider default. Keep real endpoints private. |
| `NVIDIA_LARGE_MODEL` | Optional NVIDIA content-generation model override. Precedence: explicit/YAML, environment, provider default. |
| `CURRICULUM_ALIGNMENT_PROVIDER` | Accepted values `anthropic`, `together`, `local`. CLI `--curriculum-provider` wins; unset uses the established non-provider path. Licensing follows the selected backend. |
| `TRAINFORGE_ASSESSMENT_PROVIDER` | Assessment backend name: `anthropic`, `together`, `local`, or a registered OpenAI-compatible provider. Unset preserves subagent/deterministic routing. |
| `TRAINFORGE_ASSESSMENT_MODEL` | Assessment model override. Precedence: per-call kwarg, this variable, backend model variable/default. No effect without an assessment provider. |
| `ED4ALL_ASSESSMENT_PROSE_PROVIDER` | Unset disables optional assignment/discussion prose. A registered provider enables it; construction failure warns and skips prose while deterministic quizzes continue. Product-content licensing follows that provider. |
| `ED4ALL_ASSESSMENT_DIVERSIFIED` | Off. Truthy enables deterministic diversified and constructed-response items. Included in quiz resume identity; selects no provider. |

## Synthesis inputs, identity, and routing

| Variable | Default and current contract |
|---|---|
| `TRAINFORGE_SYNTHESIS_HOLDOUT_EXCLUSION` | Off. Truthy enables the fail-closed final holdout trust chain and requires both registry and manifest paths before any provider/output creation. |
| `TRAINFORGE_SYNTHESIS_HOLDOUT_REGISTRY` | Required with holdout exclusion. Path to final `training_input_exclusion.json`; missing, stale, candidate, or mutated input fails loudly. |
| `TRAINFORGE_SYNTHESIS_HOLDOUT_MANIFEST` | Required with holdout exclusion. Path to the final reviewed expanded-suite manifest; disagreement with the registry fails loudly. |
| `TRAINFORGE_SYNTHESIS_OBJECTIVES_PATH` | Optional authoritative objectives file. Explicit API `objectives_path=` wins; a configured missing file fails before output creation. |
| `TRAINFORGE_SYNTHESIS_FRESH_START_ID` | Optional run-scoped identity requiring a matching verified marker before resume/provider/output inspection. Explicit API value wins. |
| `TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS` | Required positive integer for staged synthesis; no default. Missing/invalid fails before output or dispatch. No-op outside staged synthesis. |
| `TRAINFORGE_SYNTHESIS_PROVIDER` | Training-pair backend: `mock`, `anthropic`, `claude_session`, `together`, or `local`. Pipeline workflows setdefault a configured provider or `local`; environment overrides the kwarg and local-agent dispatch. `anthropic` always fails closed; `claude_session` requires acknowledgment; `mock` is non-promotable. |
| `TRAINFORGE_SYNTHESIS_MODEL` | Conflict detector only. It never selects a model; during staged local synthesis, if set, it must equal `LOCAL_SYNTHESIS_MODEL` or the run fails before generation. |
| `TRAINFORGE_SYNTHESIS_MAX_CONCURRENT` | `1`; blank/invalid/non-positive → `1`; accepted `2..48`; above `48` raises. CLI/API alias `--max-concurrent` wins. `claude_session` requires `1`. Failures remain loud; no serial fallback. See the [benchmark method](super-synthesis-benchmark.md). |
| `TRAINFORGE_AGNOSTIC_SYNTHESIS` | Default on. Explicit `0`/`false`/`no`/`off` selects per-vendor rollback; unset/empty/other values select the registry-driven provider. Does not choose provider/model. |
| `TRAINFORGE_STAGED_SYNTHESIS_V4` | Off for bare calls; pipeline workflows setdefault it on. Truthy selects evidence-first staged generation with bounded repairs and fail-loud leakage exhaustion. Provider/licensing are unchanged. CLI contract: `--synthesis-contract staged-v4`. |
| `TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1` | Off. CLI `--synthesis-contract micro-v1` selects micro and disables V4 for that process; conflicting ambient values, simultaneous micro+V4, or variants greater than one fail loudly. Resume state is fingerprinted and stop-aware. |
| `TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS` | Acknowledgment only for `claude_session`; truthy allows that route only when the operator has suitable rights. It never unlocks removed `anthropic`, which always fails closed. |

Provider/model choices above are governed by
[`docs/LICENSING.md`](../LICENSING.md). The license-clean public default is a
permitted local model; never treat a successful benchmark or acknowledgment as
a licensing decision.

## Synthesis eligibility and pair policy

| Variable | Default and current contract |
|---|---|
| `TRAINFORGE_COGNITIVE_TASK_TYPE` | Unset/off. Truthy adds detected cognitive-task type to assessment output and DecisionCapture; chunk/misconception optional fields remain additive. |
| `TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD` | Off. Truthy records heuristic provenance and calibrates only when sufficient real learner responses exist; it never fabricates IRT parameters. |
| `TRAINFORGE_SYNTHESIS_CONTENT_GATE` | Default on. Only explicit `0`/`false`/`no`/`off` disables; unset/garbage stays on. Ineligible units are recorded before model dispatch and excluded from rejection denominators. |
| `TRAINFORGE_SYNTHESIS_MIN_PROSE_WORDS` | `40`; unset/invalid/non-positive → `40`. Prose-arm floor; no-op when the content gate is off. |
| `TRAINFORGE_SYNTHESIS_MIN_STEM_CONTENT_WORDS` | `3`; unset/invalid/non-positive → `3`. Distinct content-word floor; no-op when the content gate is off. |
| `TRAINFORGE_DPO_MINE_REJECTS` | `off`; accepted `off`, `shadow`, `on`. `shadow` captures and reports without emitting; `on` may emit only contract-compatible, same-anchor near-miss negatives. Incomplete passes do not mine. Changing mode rekeys runtime policy and may require a fresh start. See the [reject-mining guide](reject-mined-dpo-negatives.md). |
| `TRAINFORGE_DPO_MINE_MIN_SUPPORT` | Float `[0.0,1.0)`, default `0.50`; invalid/out-of-range → `0.50`. No-op when mining is off. |
| `TRAINFORGE_DPO_MINE_MIN_FAIL_ENTAILMENT` | Float `[0.0,1.0)`, default `0.15`; invalid/out-of-range → `0.15`. No-op when mining is off. |
| `TRAINFORGE_DPO_MINE_MAX_SKELETON_FREQ` | Positive integer, default `3`; invalid/non-positive → `3`. No-op when mining is off. |
| `TRAINFORGE_DPO_MINE_MAX_FRACTION` | Float `(0.0,1.0]`, default `0.25`; invalid/out-of-range → `0.25`. Hard cap relative to accepted anchors; no-op when mining is off. |
| `TRAINFORGE_BLOOM_WINDOWS` | Off. Truthy recovers per-card Bloom levels, applies an optional target-rung evidence ceiling, and renders rung annotations; invalid/falsey → off. It does not select a provider or alter the window contract version. |

## Resume and failure semantics

Concurrency and evaluation-report cadence are behavior-neutral operational
values and do not regenerate terminal work. Model, endpoint, generation
parameters, source/focus identity, and synthesis policy remain fingerprinted.
Generation-contract drift on a bound fresh-start marker raises
`FreshStartError`; duplicate terminal journal writes raise immediately. Do not
edit markers or journals to bypass either condition. Use the public
[`pipeline invocation guide`](pipeline-invocation.md#7-graceful-stop-resume-and-checkpoints)
for stop, resume, checkpoints, and forced replay.

## Canonical sources and tests

- Workflow auto-defaults: `MCP/core/workflow_runner.py`.
- Synthesis CLI and provider gates:
  `Trainforge/synthesis/synthesize_training.py`.
- Concurrency resolver: `Trainforge/synthesis/synthesis_concurrency.py` and
  `Trainforge/tests/test_synthesis_concurrency.py`.
- Reject mining: `Trainforge/synthesis/synthesis_reject_mining.py` and
  `Trainforge/tests/test_reject_mining_*.py`.
- Chunk options: `Trainforge/chunker/` and its tests.
- Graph options: `lib/ontology/`, `lib/aggregators/`, `lib/validators/`, and
  their colocated tests.
- Assessment routing: `Trainforge/generators/providers/_assessment_provider.py`
  and `Trainforge/tests/test_assessment_*`.
