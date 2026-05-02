# Phase 3 Plan — Two-pass execution + per-block model router

## 0. Audience and dependency footprint

This plan assumes Phase 1 (`plans/phase1_tos_unblock.md`, ToS-unblock via env-var provider swap; introduces `COURSEFORGE_PROVIDER`) and Phase 2 (`plans/phase2_intermediate_format.md`, stable intermediate format + `Block` dataclass) have shipped. The Block shape this plan references is the shape Phase 2 stabilises:

- `Block.id: str` — stable per-page-per-block identifier (e.g. `week01.module03.section02.callout01`)
- `Block.type: str` — one of an enum (`objective`, `section_intro`, `flip_card`, `self_check`, `activity`, `callout`, `summary`, `recap`, `assessment_item`, `reflection`, `key_term`, `misconception`, …)
- `Block.curies: List[str]` — extracted manifest CURIEs anchoring the block (analogue of Trainforge's `preserve_tokens`)
- `Block.bloom_level: str` — Bloom verb / level if applicable
- `Block.content_type: str` — taxonomy from `lib/validators/content_type.py` / `schemas/taxonomies/content_type.json`
- `Block.objective_refs: List[str]` — `TO-NN` / `CO-NN` IDs (`Courseforge/CLAUDE.md` LO-ID contract)
- `Block.source_refs: List[Dict]` — DART `sourceId` provenance (`schemas/knowledge/courseforge_jsonld_v1.schema.json`)
- `Block.outline: Optional[Dict]` — outline-tier draft (sentence skeleton, key claims, structural metadata)
- `Block.body: Optional[str]` — rewrite-tier final HTML / markdown
- `Block.touched_by: List[Dict]` — append-only provenance trail; each entry `{tier, provider, model, timestamp, decision_capture_id, retry_count}` (Phase 2 deliverable)
- `Block.status: Literal["empty","outlined","gated","rewritten","failed"]`

**If Phase 2 picks different field names** (e.g. `slug` instead of `id`, or `metadata` instead of separate `bloom_level` / `content_type`), Step 11 below ("Sequencing") includes an explicit reconciliation step at the start to update the router signatures before any execution work begins. The router code MUST consume the Phase 2 dataclass directly — never a parallel shape.

## 1. Goals and non-goals

**Goals:**
1. Make Courseforge content generation a two-pass pipeline: outline tier (small local model) → deterministic gates → rewrite tier (configurable cloud / large local model).
2. Per-block-type model routing configurable via env vars, with optional `block_routing.yaml` policy file overrides, and per-call kwargs winning over both.
3. Promote a small set of existing validators (Step 6) to fire BETWEEN tiers, on outline-tier Block objects rather than on final HTML.
4. Per-block decision-capture so a post-hoc audit can replay (block, tier, provider, model, prompt hash, token usage, retries) for every LLM call.
5. Re-execution entry point: re-route a subset of blocks (e.g. `--blocks assessments --model deepseek-v3`) without rebuilding upstream phases.
6. Feature-flagged rollout (`COURSEFORGE_TWO_PASS=true`) so the legacy single-pass path remains the default until the new path proves out.

**Non-goals (deferred to Phase 4 / 5):**
- Statistical-tier validators (DistilBERT-style classifiers) — Phase 3 only specifies the seam.
- CLI subcommands for per-block re-execution as user-facing UX — Phase 5 owns that.
- New validators net-new to Phase 3 (only existing validators are promoted).
- Replacing the Courseforge `content-generator` subagent's markdown spec — Phase 3 augments the dispatch path; the spec's deep-content directives stay.

## 2. Tier definitions

### 2.1 Outline tier

**Purpose:** Generate a structurally-correct draft block tagged with CURIEs, objective_refs, content_type, bloom_level, and a sentence-level skeleton — but NOT the final pedagogical depth. Use a 7B-class local model.

**Per-block-type output schema** (deserialises into `Block.outline`):

```json
{
  "block_id": "week01.module03.section02.callout01",
  "block_type": "callout",
  "content_type": "warning",
  "bloom_level": "understand",
  "objective_refs": ["CO-03"],
  "curies": ["sh:NodeShape", "rdfs:subClassOf"],
  "key_claims": [
    "A NodeShape constrains the structure of an RDF node.",
    "It composes with rdfs:subClassOf for hierarchical reuse."
  ],
  "section_skeleton": [
    {"role": "lede", "summary": "..."},
    {"role": "elaboration", "summary": "..."}
  ],
  "source_refs": [{"sourceId": "dart:rdf-shacl#blk_017", "role": "primary"}],
  "structural_warnings": []
}
```

**Model class:** `qwen2.5:7b-instruct-q4_K_M` on Ollama (sane default, matches Trainforge's `LOCAL_SYNTHESIS_MODEL` precedent at `Trainforge/generators/_local_provider.py:78`). Cheap, deterministic-ish at `temperature=0.0` with `format: "json"` Ollama mode.

**Prompt structure (skeleton):** system prompt names the block-type contract (per type, ≤80 words — mirrors the terse local-only system prompts at `Trainforge/generators/_local_provider.py:127-142`); user prompt carries `(LO context, source_chunks, parent_section_objectives, target_curies)`. JSON-mode and lenient JSON extraction reuse `Trainforge.generators._openai_compatible_client.OpenAICompatibleClient._extract_json_lenient` at `Trainforge/generators/_openai_compatible_client.py:426-492`.

**Length floors:** much smaller than rewrite — `key_claims[]` ≥ 1 entry per claim, ≥ 30 chars each; `section_skeleton[]` ≥ 1 entry. Per-block-type bounds live in a `_OUTLINE_KIND_BOUNDS` table mirroring `Trainforge/generators/_local_provider.py:145-150::DEFAULT_LOCAL_KIND_BOUNDS`.

**Failure mode:** outline-tier exhaustion (`MAX_PARSE_RETRIES`) raises `OutlineProviderError(code="outline_exhausted")`; the router marks the block `status="failed"` and emits a `block_outline_failed` decision event. The block is excluded from subsequent rewrite-tier dispatch but retained in the Block list so re-execution can target only failed blocks.

#### 2.1.1 Constrained decoding as the primary structural gate

A small constrained model is more reliable at the outline tier than a large unconstrained one — counterintuitive but consistently true for slot-filling against a fixed contract. The 7B's role is mechanical: emit a structurally-valid outline payload that it cannot violate at sample time. SHACL (Section 6 / Phase 4) repositions to the **secondary** gate that catches semantic violations the grammar can't express (cross-block referential integrity, CURIE consistency across siblings).

Enforcement layer order (pre-generation, sample-time):

1. **Grammar-constrained decoding** — primary gate. The token sampler literally cannot emit invalid structure.
2. **JSON Schema enforcement during sampling** — engine-side type / shape constraints.
3. **Regex constraints** on CURIE-shaped fields, Bloom verbs, identifier slots.
4. **Token-level vocabulary masking** for controlled-vocab slots (`content_type` enum, `bloom_level` enum, `block_type` enum).

Per-provider mechanism map (the outline tier's `OutlineProvider` selects the right mechanism based on the resolved provider string):

| Provider | Mechanism | Payload field | Reference |
|---|---|---|---|
| `local` (llama.cpp / Ollama) | GBNF grammar | `grammar: <gbnf-string>` (llama.cpp); `format: <json-schema-dict>` (Ollama, recent versions) | llama.cpp server API; Ollama 0.5+ |
| `local` (vLLM) | outlines integration | `guided_grammar` / `guided_json` / `guided_regex` | vLLM `extra_body` |
| `local` (LM Studio) | GBNF | same as llama.cpp | LM Studio LLM Server |
| `together` | JSON-mode + JSON Schema | `response_format: {type: "json_schema", json_schema: {...}}` | Together AI Chat Completions |
| `anthropic` | Prompt + JSON-mode (no sample-time grammar) | n/a — falls back to JSON-mode-only + lenient parse + remediation retry | Anthropic Messages API |

Anthropic doesn't expose a sample-time grammar surface, so the outline tier defaults to local providers. When `anthropic` is selected for the outline tier (rare; the ToS-clean default is `local`), the provider gracefully degrades to JSON-mode-only and relies entirely on the rewrite-tier-style preserve-tokens + remediation retry to recover from structural drift.

The existing `OpenAICompatibleClient.chat_completion` already accepts an `extra_payload: Dict[str, Any]` arg (`Trainforge/generators/_openai_compatible_client.py:209-211`), which carries grammar / guided-decoding fields through to the underlying HTTP body unchanged. Phase 3 wires `OutlineProvider` to populate that extra_payload from a per-block-type grammar map and from `BlockProviderSpec.extra_payload`.

A new env var `COURSEFORGE_OUTLINE_GRAMMAR_MODE` (`gbnf|json_schema|json_object|none`) selects the mechanism explicitly when an operator wants to override autodetection (e.g. force JSON-mode-only on a provider that does support grammar but the operator wants to A/B-test). Default behaviour: autodetect from `(provider, base_url, model)`.

~~Pre-feedback wording elsewhere in this plan that frames SHACL as the primary structural gate is superseded by this subsection — SHACL becomes the secondary semantic gate per Phase 4 §4.~~

### 2.2 Deterministic tier (between outline and rewrite)

This is the validation layer the proposal calls "deterministic gates that reject malformed outlines before the rewrite pass." See Section 6 for the specific validators promoted.

**Inputs:** the list of `Block` objects with populated `Block.outline` (and unpopulated `Block.body`).

**Outputs:** for each block: `(passed: bool, issues: List[GateIssue])`. Failed blocks are NOT sent to rewrite; they're flagged for re-execution at Phase 5's CLI surface, or fall back to a deterministic Python template (Phase 1's emit path).

**Decision capture:** one `validation_result` event per (block, gate) failure. (`validation_result` already exists in `decision_event.schema.json:134`.)

### 2.3 Statistical tier (Phase 4 seam, not Phase 3)

The router exposes one extension point for Phase 4: between deterministic gate and rewrite-tier dispatch, the router calls `self._statistical_filter(block)` which is a no-op shim in Phase 3. Phase 4 will plug in DistilBERT-style classifiers (e.g. content_type predictor, bloom-level predictor, ungroundedness detector). The shim signature: `_statistical_filter(block: Block) -> Tuple[bool, List[GateIssue]]`.

### 2.4 Rewrite tier

**Purpose:** Take the outline (which is structurally correct but lean) and rewrite for pedagogical depth — Pattern 22 prevention (`Courseforge/agents/content-generator.md:36-52`), 600+ words per substantive sub-module, scaffolded examples, accessibility-clean components.

**Provider:** configurable per block type. Default `together` (ToS-clean cloud), with the same `LOCAL_SYNTHESIS_*` / `TOGETHER_*` env-var precedence Trainforge already establishes.

**Prompt template (skeleton, per block type):**

System prompt: full pedagogical contract (Pattern 22 prevention, color palette, depth floor) — analogous to today's Courseforge content-generator subagent system block.

User prompt key sections:
1. The outline JSON object verbatim (with `key_claims`, `section_skeleton`, `curies`, `objective_refs`).
2. The specific instruction: `"Outline is structurally correct but generated by a smaller model. PRESERVE: factual claims (verbatim), CURIEs (verbatim), objective refs, source refs, structural metadata. REWRITE: for pedagogical depth, scaffolding, examples, voice. DO NOT add facts not in the outline's key_claims or in the source chunks."`
3. Source chunks (DART block content for `source_refs`).
4. Block-type-specific output shape contract (HTML with `data-cf-*` attributes per `Courseforge/CLAUDE.md` Metadata Output table).

**Verification:** rewrite-tier output is verified for CURIE preservation (every CURIE in `Block.outline.curies` MUST appear in `Block.body`) — direct port of `Trainforge/generators/_local_provider.py:548-564::_missing_preserve_tokens`. Failed preservation triggers a Wave-120-style remediation retry (`_local_provider.py:566-583::_append_preserve_remediation`), with the same retry-budget exhaustion path raising `RewriteProviderError(code="rewrite_curie_drop")`.

**Provenance update:** on success, append `{tier:"rewrite", provider, model, timestamp, decision_capture_id, retry_count}` to `Block.touched_by` and set `Block.status="rewritten"`.

## 3. Router architecture

New module: **`Courseforge/router/`** (package, peer to `Courseforge/scripts/`, `Courseforge/agents/`). Layout:

```
Courseforge/router/
├── __init__.py
├── router.py            # CourseforgeRouter, BlockProviderSpec
├── outline_provider.py  # OutlineProvider (composes OpenAICompatibleClient)
├── rewrite_provider.py  # RewriteProvider (composes OpenAICompatibleClient + Anthropic SDK direct path)
├── policy.py            # block_routing.yaml loader + schema validation
├── prompts/
│   ├── outline_system_prompts.py   # one terse system prompt per block_type
│   ├── rewrite_system_prompts.py   # one full pedagogical prompt per block_type
│   └── prompt_hashing.py           # SHA-256 of (system + user) for decision-capture provenance
└── tests/
    ├── test_router.py
    ├── test_outline_provider.py
    ├── test_rewrite_provider.py
    └── test_policy.py
```

### 3.1 `CourseforgeRouter` class

```python
class CourseforgeRouter:
    def __init__(
        self,
        *,
        policy: Optional[BlockRoutingPolicy] = None,   # parsed block_routing.yaml
        outline_provider: Optional[OutlineProvider] = None,
        rewrite_provider: Optional[RewriteProvider] = None,
        capture: Optional[DecisionCapture] = None,
        deterministic_gates: Optional[List[InterTierGate]] = None,
        statistical_filter: Optional[Callable[[Block], Tuple[bool, List[GateIssue]]]] = None,
    ) -> None: ...

    def route(self, block: Block, *, tier: Literal["outline", "rewrite"], **overrides) -> Block: ...
    def route_all(self, blocks: List[Block]) -> List[Block]: ...           # full two-pass on a page/week
    def reroute_failed(self, blocks: List[Block]) -> List[Block]: ...      # only blocks with status=="failed"
```

The dispatch shape mirrors `Trainforge/generators/_curriculum_provider.py:158-263::CurriculumAlignmentProvider.__init__` exactly: env-var resolution → optional kwarg override → instantiate the right backend class. Per-block-type generalisation is the only structural difference.

### 3.2 `BlockProviderSpec` dataclass

```python
@dataclass(frozen=True)
class BlockProviderSpec:
    block_type: str
    tier: Literal["outline", "rewrite"]
    provider: Literal["anthropic", "together", "local", "openai_compatible"]
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0   # 0.0 for outline, 0.4 for rewrite
    max_tokens: int = 2400
    json_mode: bool = True     # outline only; rewrite returns HTML
    extra_payload: Dict[str, Any] = field(default_factory=dict)
```

### 3.3 Policy resolution order

For a `(block_type, tier)` pair:

1. **Per-call kwargs** to `route(...)` (highest precedence; supports re-execution overrides).
2. **`block_routing.yaml`** entry matching `(block_type, tier)`.
3. **Tier-default env vars** — `COURSEFORGE_OUTLINE_*` for outline, `COURSEFORGE_REWRITE_*` for rewrite.
4. **`COURSEFORGE_PROVIDER`** (Phase 1 deliverable) as final fallback.
5. **Hardcoded class default** (`together` / `meta-llama/Llama-3.3-70B-Instruct-Turbo` for rewrite; `local` / `qwen2.5:7b-instruct-q4_K_M` for outline).

### 3.4 Backend instantiation

Same composition pattern as `Trainforge/generators/_curriculum_provider.py:211-262`:

- Provider in `("local", "together", "openai_compatible")` → `OpenAICompatibleClient` instance with the resolved `(base_url, model, api_key)` and `provider_label=provider_string`. The `extra_payload` field carries through to the embedded client's `chat_completion(...)` (already supported at `Trainforge/generators/_openai_compatible_client.py:209-211`).
- Provider `"anthropic"` → lazy `import anthropic`, instantiate `anthropic.Anthropic(api_key=...)`. Mirrors `_curriculum_provider.py:387-434`.

### 3.5 Decision-capture wiring

`OutlineProvider` and `RewriteProvider` each compose one `OpenAICompatibleClient`. The client's `llm_chat_call` event already exists (`_openai_compatible_client.py:534`). On top of that, the router emits ONE higher-level event per (block, tier) — per Section 9.

### 3.6 Self-consistency dispatch

Constrained decoding guarantees structural validity but doesn't guarantee semantic quality. The router pairs it with **self-consistency sampling**: generate N candidates per block, run the validator chain on each, return the first passer. Cheap on local hardware because the outline tier is 7B; catches tail-end failures without escalating to a stronger model.

```python
class CourseforgeRouter:
    def route_with_self_consistency(
        self,
        block: Block,
        *,
        n_candidates: int = 3,
        validators: Optional[List[InterTierGate]] = None,
    ) -> Block: ...
```

**Validator chain order** (cheapest gate first; later gates only run if earlier ones pass):

1. **Grammar / JSON Schema** — sample-time, always passes by construction. Listed first so the chain shape is uniform.
2. **SHACL** — semantic invariants the grammar can't express (Phase 4 §4).
3. **CURIE resolution** — every CURIE in `Block.outline.curies` must resolve against the concept-graph manifest (`schemas/knowledge/courseforge_jsonld_v1.schema.json`).
4. **Embedding similarity floor** — Phase 4 §2: every example block must clear cosine threshold against the concept it claims to illustrate; every assessment-Block stem must clear threshold against its declared objective. Sub-threshold returns `action="regenerate"` per §6.5.
5. **Round-trip check** — Phase 4 §5(a): extract objective from assessment item, verify it matches the source objective.

**Generation strategy:** candidates may be generated in parallel (all N requests fired concurrently to the local server, first-passer wins, others discarded) or sequentially (cheaper on token usage when first candidate usually passes). Default sequential; parallel mode opt-in via per-call kwarg.

**Decision-capture metadata** added to the `outline_block_call` event (Section 9.2):

- `n_candidates_requested`: int (the operator-supplied or env-var-resolved budget).
- `winning_candidate_index`: int (0-indexed; absent if all failed).
- `failed_candidate_count`: int.
- `validator_failure_distribution`: `Dict[validator_name, int]` — counts how many candidates failed each validator (informs Phase 4 calibration).

**Env var:** `COURSEFORGE_OUTLINE_N_CANDIDATES` (default `3`). Per-block-type override available in `block_routing.yaml` (`blocks.<type>.outline.n_candidates`).

### 3.7 Regeneration budget + escalation

Some block types exceed 7B competence — usually prereq inference and multi-step reasoning ones. Infinite local retry loops eat overnight runtime when a $0.001 API call resolves it. The router enforces a per-block regeneration budget, then escalates directly to the rewrite tier (skipping further outline retries).

**Per-block counter:** read from `Block.validation_attempts` (Phase 2 deliverable — see `plans/phase2_intermediate_format_detailed.md` Subtasks 3, 10, 13 for the dataclass field, JSON Schema field, and SHACL property respectively). Default `0`; the router increments on every failed validator pass within the self-consistency loop.

**Escalation trigger:**

```
if block.validation_attempts >= COURSEFORGE_OUTLINE_REGEN_BUDGET:
    block.escalation_marker = "outline_budget_exhausted"
    # short-circuit straight to rewrite tier; skip remaining outline retries
```

`Block.escalation_marker` is one of the markers Phase 2 declares in `_ESCALATION_MARKERS` (see `plans/phase2_intermediate_format_detailed.md` Subtask 3). Phase 3 sets `"outline_budget_exhausted"`; Phase 5 may set the others.

**Rewrite-tier prompt amendment:** when the rewrite tier sees a non-null `escalation_marker`, it switches to a richer prompt template:

```
The outline tier could not produce a valid {block_type} after {n} attempts (marker={escalation_marker}).
Synthesize from scratch using {source_chunks} and {objective_refs}, preserving CURIEs:
  {curies}
Do not introduce facts outside the supplied source chunks.
```

**Per-block-type immediate escalation:** `block_routing.yaml` schema gains an optional per-block-type `escalate_immediately: bool` flag. When true, the router skips the outline tier entirely and routes the block straight to rewrite with `escalation_marker="outline_skipped_by_policy"`. Use cases: prereq inference, multi-step reasoning, or any block-type whose outline-tier success rate is empirically low. Maintenance contract: when an operator flips a block-type to `escalate_immediately: true`, they must justify the swap in the YAML rationale comment block (operator-facing, Section 4 already shows the convention).

**Env var:** `COURSEFORGE_OUTLINE_REGEN_BUDGET` (default `3`).

## 4. `block_routing.yaml` schema

**Location convention:** `Courseforge/router/block_routing.yaml` (default), or any path supplied via `COURSEFORGE_BLOCK_POLICY` env var. JSON Schema lives at `schemas/courseforge/block_routing.schema.json`.

**Schema (Draft 2020-12):**

```yaml
# block_routing.yaml — Courseforge per-block-type model routing
version: 1
# Optional global defaults; otherwise the env-var fallback chain wins.
defaults:
  outline:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
    base_url: http://localhost:11434/v1
    temperature: 0.0
    max_tokens: 1200
  rewrite:
    provider: together
    model: meta-llama/Llama-3.3-70B-Instruct-Turbo
    temperature: 0.4
    max_tokens: 2400

# Per-block-type overrides. Keys are Block.type values.
blocks:
  assessment_item:
    rewrite:
      provider: anthropic
      model: claude-sonnet-4-6
      temperature: 0.2
      # Rationale comment (operator-facing): assessments are higher-stakes,
      # use the highest-quality available rewriter. Anthropic ToS forbids
      # using output as TRAINING data, but Courseforge output is course
      # content for learners, not training data — the ToS-clean constraint
      # the Trainforge synthesis pipeline carries doesn't apply here.

  flip_card:
    outline:
      # 7B is fine; flip cards are short.
      provider: local
      model: qwen2.5:7b-instruct-q4_K_M
    rewrite:
      provider: together
      model: Qwen/Qwen2.5-72B-Instruct-Turbo

  callout:
    rewrite:
      # Callouts are short — no need for a 70B
      provider: local
      model: qwen2.5:14b-instruct-q4_K_M

# Optional per-block hard-pin (highest specificity short of a per-call kwarg).
# Block.id glob match. Useful for "I want this one block to use deepseek".
overrides:
  - block_id: "week09.module*.assessment_item.*"
    rewrite:
      provider: openai_compatible
      base_url: https://api.deepinfra.com/v1/openai
      api_key_env: DEEPINFRA_API_KEY
      model: deepseek-ai/DeepSeek-V3
```

**Loader:** `Courseforge/router/policy.py::load_block_routing_policy(path: Optional[Path]) -> BlockRoutingPolicy`. Validates against `schemas/courseforge/block_routing.schema.json` using the existing jsonschema dependency (the same way `lib/validators/page_objectives.py` and friends validate inputs). Default behaviour when file is absent: log an INFO row, return an empty policy (env-var chain handles everything).

## 5. Env-var inventory

| Env var | Purpose | Default |
|---|---|---|
| `COURSEFORGE_TWO_PASS` | Master feature flag for the two-pass pipeline. When false, legacy single-pass content_generation runs unchanged. | `false` |
| `COURSEFORGE_OUTLINE_PROVIDER` | Tier-default outline backend (`local`/`together`/`anthropic`/`openai_compatible`). | `local` |
| `COURSEFORGE_OUTLINE_MODEL` | Tier-default outline model. | `qwen2.5:7b-instruct-q4_K_M` |
| `COURSEFORGE_OUTLINE_BASE_URL` | Tier-default outline base URL (only used when provider is `local`/`openai_compatible`). | `http://localhost:11434/v1` |
| `COURSEFORGE_OUTLINE_API_KEY` | Tier-default outline API key. | `local` (placeholder) |
| `COURSEFORGE_REWRITE_PROVIDER` | Tier-default rewrite backend. | `together` |
| `COURSEFORGE_REWRITE_MODEL` | Tier-default rewrite model. | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `COURSEFORGE_REWRITE_BASE_URL` | Tier-default rewrite base URL. | `https://api.together.xyz/v1` |
| `COURSEFORGE_REWRITE_API_KEY` | Tier-default rewrite API key. | from `TOGETHER_API_KEY` if provider=`together` |
| `COURSEFORGE_BLOCK_POLICY` | Path to `block_routing.yaml`. | `Courseforge/router/block_routing.yaml` |
| `COURSEFORGE_OUTLINE_N_CANDIDATES` | Self-consistency sample count per block at the outline tier (Section 3.6). | `3` |
| `COURSEFORGE_OUTLINE_REGEN_BUDGET` | Per-block regeneration budget before outline-tier escalation to rewrite (Section 3.7). | `3` |
| `COURSEFORGE_OUTLINE_GRAMMAR_MODE` | Constrained-decoding mechanism (`gbnf|json_schema|json_object|none`). Selects how `OutlineProvider` materialises the grammar in the request payload (Section 2.1.1). | autodetect from provider |
| `COURSEFORGE_PROVIDER` | (Phase 1) global Courseforge provider — final fallback for both tiers. | (Phase 1 default) |

Each row maps to a CLAUDE.md flag table addition (style: same as the `CURRICULUM_ALIGNMENT_PROVIDER` row at `CLAUDE.md:729`).

## 6. Phase wiring (recommendation: split)

**Recommendation:** SPLIT `content_generation` in `config/workflows.yaml` into three sequential phases under `textbook_to_course` (and mirror the split under `course_generation`):

```yaml
- name: content_generation_outline
  agents: [content-generator]            # same agent type — still routes via Wave-74 dispatch
  parallel: true
  max_concurrent: 10
  batch_by: week
  depends_on: [course_planning, source_mapping, staging]
  timeout_minutes: 30                    # shorter than legacy — outline is cheap
  description: Outline-tier draft block emission via 7B local model
  inputs_from: [...]
  outputs:
    - blocks_outline_path                # JSONL of Block objects (outline filled, body empty)
  validation_gates:
    - gate_id: outline_curie_anchoring
      validator: lib.validators.curie_anchoring.CurieAnchoringValidator
      severity: critical
    - gate_id: outline_content_type
      validator: lib.validators.content_type.ContentTypeValidator
      severity: critical
    - gate_id: outline_page_objectives
      validator: lib.validators.page_objectives.PageObjectivesValidator
      severity: critical
    - gate_id: outline_source_refs
      validator: lib.validators.source_refs.PageSourceRefValidator
      severity: critical

- name: content_generation_rewrite
  agents: [content-generator]
  parallel: true
  max_concurrent: 10
  batch_by: week
  depends_on: [content_generation_outline]
  timeout_minutes: 90
  description: Rewrite-tier pedagogical-depth pass on validated outlines
  inputs_from:
    - param: blocks_outline_path
      source: phase_outputs
      phase: content_generation_outline
      output: blocks_outline_path
  outputs:
    - content_paths
    - page_paths
    - content_dir
    - blocks_final_path
  validation_gates:
    # Existing gates currently on content_generation move here unchanged
    - gate_id: content_grounding         # final HTML check — unchanged
    - gate_id: content_structure
    - gate_id: source_refs               # also runs on outline; re-runs on final HTML
```

**Justification:**
1. **Failure isolation.** Outline-tier failures (cheap, fast) re-run independently without burning a rewrite-tier cloud call.
2. **Parallel-batch independence.** Outline can complete for all weeks before any rewrite kicks off, letting the inter-tier deterministic gates accumulate cross-block signals (e.g. consistent CURIE coverage across the course).
3. **Phase-level decision capture.** Each phase emits its own decision events under distinct phase tags (Section 9 adds `courseforge-content-generator-outline` / `-rewrite` to the decision_event.schema.json `phase` enum).
4. **Re-execution surface (Phase 5).** A future `--rerun-rewrite-only` CLI flag becomes a one-line `workflow_runner` change.
5. **Backward compat via flag.** When `COURSEFORGE_TWO_PASS=false`, the workflow runner skips the new phases and the legacy `content_generation` runs unchanged. This is achievable with a per-phase `enabled_when:` predicate on `config/workflows.yaml` (small change to `MCP/core/workflow_runner.py` to honour it).

The single-phase alternative (one `content_generation` phase with internal two-pass) was rejected because it muddles decision-capture phase tags, doubles the legacy phase's timeout, and forces re-execution to re-run the outline pass even when only rewrite needs redoing.

### 6.5 Validator output shape

Inter-tier validators return a structured signal the router consumes to decide whether to re-roll, escalate, or hard-fail:

```python
@dataclass(frozen=True)
class GateResult:
    passed: bool
    action: Literal["pass", "regenerate", "escalate", "block"]
    issues: List[GateIssue]
    score: Optional[float] = None     # cosine / shape-conformance / etc., when applicable
```

**Action semantics:**

| `action` | Router behaviour |
|---|---|
| `"pass"` | Block proceeds to next gate / next tier. |
| `"regenerate"` | Re-roll within the self-consistency loop (Section 3.6). Increments `Block.validation_attempts`. If budget exhausted (Section 3.7), the router promotes the action to `"escalate"`. |
| `"escalate"` | Short-circuit to the rewrite tier with `Block.escalation_marker` set. Skip remaining outline-tier retries. |
| `"block"` | Hard-fail. Used by deterministic gates where re-rolling cannot help (e.g. CURIE-resolution against a missing manifest entry; the LLM cannot invent a manifest row). Block is marked `status="failed"` and excluded from rewrite. |

**Per-validator default action mapping** (the Phase 4 detailed plan fully specifies this; Phase 3 only requires the seam):

| Validator | Failure default action |
|---|---|
| Grammar / JSON Schema (sample-time) | n/a (cannot fail at runtime) |
| `outline_shacl` | `regenerate` |
| `objective_assessment_similarity` (Phase 4) | `regenerate` |
| `concept_example_similarity` (Phase 4) | `regenerate` |
| `objective_roundtrip_similarity` (Phase 4) | `regenerate` |
| `bloom_classifier_disagreement` (Phase 4 deferred) | `regenerate` (when the deferred classifier lands) |
| CURIE-resolution-against-manifest | `block` (deterministic; re-rolling can't invent a missing manifest row) |
| `curie_anchoring` (Wave 135c, generalised) | `regenerate` (the LLM may have dropped a token; re-roll often recovers) |
| `content_type` enum | `regenerate` |
| `page_objectives` | `block` (LO coverage is structural, not regen-fixable) |
| `source_refs` against staging manifest | `block` (sourceId not in manifest is a structural error) |

The `Block.touched_by` provenance trail (Phase 2 deliverable) records the action chosen per gate fire, so an audit can replay the regen-vs-escalate-vs-block decision chain.

## 7. Inter-tier gate promotion

Validators that should fire AFTER outline / BEFORE rewrite, per `lib/validators/` directory listing:

| Validator | Today's input | Outline-tier input shape | Adapter / generalisation |
|---|---|---|---|
| `curie_anchoring.py` (Wave 135c) | instruction-pair JSONL | Block objects with `Block.outline.curies` and `Block.outline.key_claims` | **Generalise.** Accept a new `_validate_blocks(blocks)` entry that reads `(curies, key_claims)` from each block; the existing `_validate_pairs(pairs)` stays. Adapter shim in `Courseforge/router/inter_tier_gates.py`. |
| `content_type.py` (REC-VOC-03) | instruction_pair `content_type` field, retriever filter | Block.outline.content_type | **In-place generalise.** Already a pure enum check (`schemas/taxonomies/content_type.json`). Add a one-line caller in the router that calls `validate_content_type(block.outline.content_type)`. |
| `page_objectives.py` | content_dir / objectives_path (HTML scan) | List of (Block.objective_refs, Block.outline.key_claims) | **Adapter required.** Build a `_build_page_objectives_from_blocks(blocks_path)` adapter at `lib/validators/page_objectives.py::_build_page_objectives_from_blocks` parallel to the existing HTML-scanning builder. The validator core (LO-specificity check) stays unchanged; the adapter feeds it the synthetic input. |
| `source_refs.py` (Wave 9) | rendered HTML page paths | Block.outline.source_refs | **Adapter required.** Same shape: build `_build_page_source_refs_from_blocks(blocks_path)` so the existing `_resolve_against_manifest(...)` core works on outline-tier source_refs. |
| `content_grounding.py` (Wave 31) | rendered HTML | — | **Stays at rewrite tier.** This validator wants final HTML (paragraph-level word counts, ancestor walk). NOT promoted. |

The `Courseforge/router/inter_tier_gates.py` module wires the four promoted gates as `InterTierGate` instances, each taking `Block` and returning `GateResult`. The inter-tier gate phase runs AFTER `content_generation_outline` populates `blocks_outline_path` and BEFORE `content_generation_rewrite` consumes it. Failed blocks are dropped from the rewrite list and persisted with `status="failed"`.

## 8. Subagent vs. direct-call execution model

**Recommendation:** **Direct in-process router invocation** for both outline and rewrite tiers, NOT subagent dispatch.

**Trade-off analysis:**

Today, `content-generator` is a subagent in `AGENT_SUBAGENT_SET` at `MCP/core/executor.py:225-242`. Each weekly module is one Task call, dispatched via `LocalDispatcher.dispatch_task` at `MCP/orchestrator/local_dispatcher.py:227-307`. With 12 weeks × ~7 files each = ~84 tasks today.

If outline + rewrite each became subagents, we'd have ~840 tasks (84 files × ~10 blocks/file × 2 tiers), each crossing the mailbox bridge. That's a 10× explosion of mailbox round-trips for what is fundamentally a stateless `(prompt, model) -> response` operation already cleanly provided by `OpenAICompatibleClient`.

The Trainforge precedent is informative: `Trainforge/generators/_curriculum_provider.py` runs in-process via `align_chunks.py` even though `align_chunks` itself can be invoked in a subagent context. The provider class is a pure-Python LLM-routing helper. The router belongs at the same architectural layer.

**Concrete recommendation:**
- The `content-generator` subagent stays the orchestration entry point per phase task. Wave-74 dispatch is unchanged.
- INSIDE the subagent's tool path (`MCP/tools/pipeline_tools.py::generate_course_content`, the function `AGENT_TOOL_MAPPING["content-generator"]` resolves to per `executor.py:147`), the implementation acquires a `CourseforgeRouter` instance and calls `router.route_all(blocks)` directly.
- The decision capture chain stays clean: the subagent owns `phase_courseforge-content-generator-outline/`; the router emits per-block events under that phase via the capture handle the subagent passes in.

This keeps Wave 74's subagent-vs-tool classification stable, avoids 10× mailbox traffic, and preserves the option (Phase 5+) of moving individual rewrite calls back to subagents if a future Anthropic Claude Code session-driven rewrite path is desired (mirroring `Trainforge/generators/_claude_session_provider.py:107`).

## 9. Provenance, re-execution, and decision-capture event shapes

### 9.1 Re-execution entry point

New CLI: `python -m Courseforge.router.rerun_blocks --blocks-path <path/to/blocks.jsonl> --filter "block_type==assessment_item" --tier rewrite --provider anthropic --model claude-sonnet-4-6`.

Behaviour:
1. Load `blocks.jsonl` (Phase 2's intermediate format).
2. Filter blocks by `--filter` predicate (a small declarative DSL: `block_type==X`, `status==failed`, `block_id matches glob`).
3. For each matching block, set `Block.body=None` and `Block.status="outlined"`, then call `router.route(block, tier="rewrite", **per_call_overrides)`.
4. Append a new `touched_by` entry per re-route.
5. Atomically rewrite `blocks.jsonl` (tmp + rename) and re-run the rewrite-tier consumer (`generate_course.py::_render_*` against the new Block list) to produce final HTML.

Phase 5 promotes this to a first-class `ed4all run` subcommand. Phase 3 ships only the Python-module entry point.

### 9.2 New decision-capture events

Two new `decision_type` enum values added to `schemas/events/decision_event.schema.json:63-136`:

- `outline_block_call`
- `rewrite_block_call`

Two new `phase` enum values added to `schemas/events/decision_event.schema.json:53`:

- `courseforge-content-generator-outline`
- `courseforge-content-generator-rewrite`

**`outline_block_call` payload contract (router-emitted, ≥20-char rationale):**

```json
{
  "decision_type": "outline_block_call",
  "phase": "courseforge-content-generator-outline",
  "operation": "route_outline_block",
  "decision": "Outline tier emitted draft for block <id> via provider=<p>, model=<m>; key_claims=<n>, curies=<n>, retries=<n>.",
  "rationale": "Routing block_type=<type> outline through provider=<p> model=<m> base_url=<u> (chosen via <policy_source: env|yaml|kwarg|default>). Outline tier produces structurally-correct draft with CURIEs preserved; rewrite tier will expand for pedagogical depth. prompt_hash=<sha256[:12]>, prompt_tokens=<n>, completion_tokens=<n>, http_retries=<n>, parse_retries=<n>, json_mode=true.",
  "ml_features": {
    "block_id": "<id>",
    "block_type": "<type>",
    "tier": "outline",
    "provider": "<p>",
    "model": "<m>",
    "policy_source": "<env|yaml|kwarg|default>",
    "prompt_hash_12": "<12-char>",
    "token_usage": {"prompt": ..., "completion": ..., "total": ...}
  }
}
```

**`rewrite_block_call` payload** is structurally identical, with `tier="rewrite"`, additional fields `outline_block_id`, `curies_preserved_count`, `curies_required_count`, `pedagogical_depth_chars` (final body length).

The lower-level `llm_chat_call` event from `_openai_compatible_client.py:534` continues to fire underneath each router call, so a postmortem can drill from a router event to its underlying HTTP call.

## 10. Testing plan

Mirror the Trainforge testing pattern at `Trainforge/tests/test_curriculum_alignment_provider.py`:

1. **`Courseforge/router/tests/test_router.py`** — router dispatch unit tests:
   - `test_unknown_provider_raises_value_error` (Trainforge precedent: line 107).
   - `test_default_outline_provider_is_local_when_env_unset` / `test_env_var_selects_provider`.
   - `test_block_routing_yaml_overrides_env_var`.
   - `test_per_call_kwarg_overrides_yaml`.
   - `test_per_block_type_routing_dispatches_to_correct_provider` (table-driven across 5 block types × 2 tiers, mock providers).
   - `test_failed_outline_block_excluded_from_rewrite`.
   - `test_reroute_failed_only_touches_failed_blocks`.
2. **`test_outline_provider.py`** — uses `httpx.MockTransport` exactly like `Trainforge/tests/test_curriculum_alignment_provider.py:71-74`, asserts JSON-mode payload includes `format: "json"` and `response_format: {...}`, asserts lenient extraction recovers from markdown fences, asserts CURIE preservation gate fires.
3. **`test_rewrite_provider.py`** — same shape, plus Anthropic-SDK happy path with mock client (Trainforge precedent: `_curriculum_provider.py:387-434`).
4. **`test_policy.py`** — `block_routing.yaml` schema validation, default-when-absent, glob match in `overrides`.
5. **Integration test** at `tests/integration/test_courseforge_two_pass_end_to_end.py`:
   - Fixture: 1 mini course with 2 weeks, 4 block types.
   - Mock outline provider returns canned outline JSON; mock rewrite provider returns canned HTML.
   - Run `content_generation_outline` → inter-tier gates → `content_generation_rewrite`.
   - Assert: every emitted block has `touched_by` entries for both tiers, every CURIE in outline survives to final HTML, all decision-capture events validate against `decision_event.schema.json`.
6. **Decision-capture wiring test** (Trainforge precedent: lines after 119): mock capture handle, assert one `outline_block_call` and one `rewrite_block_call` event per block, both with rationale ≥20 chars.
7. **Regression test:** with `COURSEFORGE_TWO_PASS=false`, run the legacy `course_generation` workflow against a fixture course; assert byte-identical (or whitespace-tolerant) match to a pre-Phase-3 golden output.
8. **Schema-strict test** (`DECISION_VALIDATION_STRICT=true` env): assert every emitted event passes the strict validator. Closes the regression class Wave 120 Phase A re-fixed for the curriculum surface.

## 11. Sequencing

Subtasks in execution order. Items with `(parallel: A)` can be parallelised with same-tag items.

1. **Phase 2 reconciliation.** Read the actual Block dataclass from Phase 2's deliverable (likely `Courseforge/blocks/block.py`). Update this plan's signatures + the router's type hints to use the canonical shape. (1 step; trivially small.)
2. **Add the four new decision_type / phase enum values** to `schemas/events/decision_event.schema.json` + regenerate the strict-mode validator's frozen enum cache. (parallel: schemas)
3. **Land `block_routing.schema.json`** at `schemas/courseforge/block_routing.schema.json`. (parallel: schemas)
4. **Land `Courseforge/router/policy.py`** + `block_routing.yaml` default skeleton. (parallel: router-core)
5. **Land `Courseforge/router/outline_provider.py`** with `OpenAICompatibleClient` composition (mirror `_local_provider.py:174-270`). (parallel: router-core)
6. **Land `Courseforge/router/rewrite_provider.py`** with both `OpenAICompatibleClient` and Anthropic-SDK-direct paths (mirror `_curriculum_provider.py`). (parallel: router-core)
7. **Land `Courseforge/router/router.py::CourseforgeRouter`**. Depends on 4-6.
8. **Land `Courseforge/router/inter_tier_gates.py`** + adapter shims for `page_objectives.py` and `source_refs.py`. (parallel: gates)
9. **Generalise `lib/validators/curie_anchoring.py`** to accept a Block-list input mode. (parallel: gates)
10. **Wire two new phases into `config/workflows.yaml`** (`content_generation_outline` + `content_generation_rewrite`) under `textbook_to_course` AND `course_generation`, gated by `COURSEFORGE_TWO_PASS`. Update `MCP/core/workflow_runner.py`'s phase-input mapping. Depends on 7-9.
11. **Land the `MCP/tools/pipeline_tools.py::generate_course_content` two-pass path** that instantiates `CourseforgeRouter` and dispatches outline → gates → rewrite when the flag is on. Legacy single-pass path stays as the `else` branch. Depends on 10.
12. **Add the unit test suite** (Section 10 items 1-4). (parallel: tests-unit; can run alongside 8-11)
13. **Add the integration test** + decision-capture wiring test + strict-schema test. Depends on 11.
14. **Add the regression test** for legacy single-pass mode. Depends on 11.
15. **Document the seven new env vars in `CLAUDE.md`** flag table (mirror the `CURRICULUM_ALIGNMENT_PROVIDER` row).
16. **Land `Courseforge/router/rerun_blocks.py`** entry point (Section 9.1). (Phase 5 will wrap as a CLI subcommand.)

Approximate effort: 16 subtasks, ~9 parallelisable into three waves (schemas / router-core / gates / tests / wiring).

## 12. Risks and rollback

**Top 5 risks:**

1. **CURIE drop in rewrite tier.** Mitigated by Wave-120-style preserve-tokens directive + remediation retry (port `Trainforge/generators/_local_provider.py:548-583`). Rollback: per-block flag `COURSEFORGE_REWRITE_STRICT_PRESERVE=1` raises the floor to 1.0 (full preservation required); failures fall back to deterministic Python templates emitting outline-only content.
2. **Per-block decision-capture volume explosion.** ~10× event count on a 12-week course (~840 events instead of ~84). Mitigated by event compactness (no full prompt persisted; only `prompt_hash_12`). Rollback: env flag `COURSEFORGE_DECISION_CAPTURE_GRANULARITY=phase|block` defaults to `phase` initially.
3. **Outline-tier 7B drift on uncommon block types.** Mitigated by JSON-mode + lenient extraction + remediation retry (proven in Wave 113 / 114). Rollback: per-block-type override in `block_routing.yaml` flips the outline provider to a 14B model for the affected types.
4. **Inter-tier gate input-shape drift.** Adapters in Section 7 add a second entry point alongside the existing HTML path; both must stay in sync. Mitigated by parameterised tests over both input shapes. Rollback: revert the workflows.yaml split; the legacy single-phase content_generation gate panel still works because the legacy path runs the rewrite-tier-output gates unchanged.
5. **Operator confusion: which provider is each block-type using?** Mitigated by emitting at startup a compact summary table (per phase task) listing the resolved policy for every block type encountered. Rollback: none needed; this is purely additive observability.

**Feature-flag rollout:**

Stage 1 (week 1): `COURSEFORGE_TWO_PASS=false` default. Land all code, ship tests green. Operators opt in per-run.
Stage 2 (week 2): One full course (e.g. `rdf-shacl-551-2`) generated under the flag and human-reviewed end to end. Compare quality + provenance volume against the legacy run.
Stage 3 (week 3+): Default flips to `true` only after Stage 2 review passes; legacy path retained for one further wave as fallback.

## 13. Open questions

1. **Outline JSON shape: single uniform schema or per-block-type schema?** This plan proposes a uniform shape with optional fields. A per-type schema would catch malformed outlines earlier but multiplies prompt-template surface ~10× and complicates the CURIE-anchoring adapter. Recommend uniform-with-required-fields-per-type.
2. **Should the rewrite tier consume the source chunks directly, or only the outline + key_claims?** If only the outline, hallucination risk drops but prose quality may suffer (the rewriter doesn't see the source's actual examples). Recommend: rewrite tier sees BOTH outline AND source chunks, with the explicit instruction "do not introduce facts not in `key_claims` or in the source." This matches Trainforge's paraphrase contract at `Trainforge/generators/_local_provider.py:127-133`.
3. **`block_routing.yaml` location: per-course or global?** Per-course (`{project}/block_routing.yaml`) gives operators fine-grained control; global (`Courseforge/router/block_routing.yaml`) is simpler. Recommend global default with optional per-course override discovered via `{project_dir}/.courseforge/block_routing.yaml`.
4. **Should outline-tier failures emit a deterministic-template fallback Block, or simply fail the block?** Trainforge's Wave 135b force-injection precedent suggests deterministic fallback is safer. Phase 3 is silent on this; recommend deterministic fallback (so a failed outline doesn't cascade into a failed page) emit a `paraphrase_used_deterministic_draft`-shaped event.
5. **For the Anthropic rewrite path, do we plumb prompt-caching headers?** The `OpenAICompatibleClient` doesn't, the Anthropic SDK does. Phase 1 may already specify; if not, recommend enabling on chunks ≥ 1024 tokens to amortise cost.
6. **Do we want `block_routing.yaml` to be per-page-glob or only per-block-type?** This plan supports both via `overrides:` block-id glob. Confirm operator desire.
7. **Inter-tier gate phase: separate workflow phase, or in-process between outline and rewrite phases?** This plan recommends in-process inside `content_generation_outline` (gate runs at phase end, before phase-output emission), to avoid a third ~empty workflow phase. Confirm.
8. **Differentiated rewrite-tier prompts for budget-exhausted vs. immediately-escalated blocks?** Section 3.7 defines one richer prompt for both `escalation_marker="outline_budget_exhausted"` and `escalation_marker="outline_skipped_by_policy"`. A case can be made for differentiating: budget-exhausted blocks have a partial outline the rewriter can reference (signal that 7B got partway); skipped-by-policy blocks have nothing. The current plan elides the difference; confirm whether the rewrite-tier prompt template should branch on the marker.
9. **Per-validator vs. global validator-action signal?** Section 6.5's `GateResult.action` is per-validator. An alternative: validators emit only `(passed, score, issues)` and the router decides the action via a central policy table. Per-validator gives validators agency (e.g. CURIE-resolution can hard-block while embedding-similarity can regenerate); central policy is easier to change in one place. The current plan chooses per-validator with a default mapping table (Section 6.5) the operator can override via gate config. Confirm.

---

### Critical files for implementation

- `/home/user/Ed4All/Courseforge/router/router.py` (NEW)
- `/home/user/Ed4All/Courseforge/router/outline_provider.py` (NEW)
- `/home/user/Ed4All/Courseforge/router/rewrite_provider.py` (NEW)
- `/home/user/Ed4All/Courseforge/router/policy.py` (NEW)
- `/home/user/Ed4All/Courseforge/router/inter_tier_gates.py` (NEW)
- `/home/user/Ed4All/config/workflows.yaml` (split content_generation into outline + rewrite phases)
- `/home/user/Ed4All/schemas/events/decision_event.schema.json` (add two `decision_type` enum values + two `phase` enum values)
- `/home/user/Ed4All/MCP/tools/pipeline_tools.py` (wire `generate_course_content` to `CourseforgeRouter` when `COURSEFORGE_TWO_PASS=true`)

Reference (read-only) files cited heavily by this plan:

- `/home/user/Ed4All/Trainforge/generators/_curriculum_provider.py` (dispatch precedent)
- `/home/user/Ed4All/Trainforge/generators/_openai_compatible_client.py` (HTTP client)
- `/home/user/Ed4All/Trainforge/generators/_local_provider.py` (preserve-tokens + remediation retry)
- `/home/user/Ed4All/Trainforge/generators/_together_provider.py` (subclass-hook pattern)
- `/home/user/Ed4All/MCP/core/executor.py` (Wave-74 dispatch + AGENT_SUBAGENT_SET)
- `/home/user/Ed4All/MCP/orchestrator/local_dispatcher.py` (dispatch_task contract)
- `/home/user/Ed4All/lib/validators/curie_anchoring.py`, `content_type.py`, `page_objectives.py`, `source_refs.py`, `content_grounding.py`
- `/home/user/Ed4All/Courseforge/scripts/generate_course.py` (legacy emit path)
- `/home/user/Ed4All/Courseforge/agents/content-generator.md` (Pattern 22 prevention contract)
- `/home/user/Ed4All/Trainforge/tests/test_curriculum_alignment_provider.py` (testing pattern)
