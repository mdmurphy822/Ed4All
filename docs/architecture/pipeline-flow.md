# Pipeline Flow — how `textbook_to_course` actually executes

This document describes the pipeline as it *runs*, not as the phase list reads.
Every phase name, dependency edge, and filename below was verified against
`config/workflows.yaml`, the dispatch and runner code, and the per-phase
checkpoint/state records of a completed 21-phase production run.

Line references are to the code as of the commit that added this file; treat
them as pointers to the right function, not as exact addresses.

Related: [`ADR-001-pipeline-shape.md`](ADR-001-pipeline-shape.md) (why the shape
is what it is), [`aggregators.md`](aggregators.md) (the post-loop rollups),
[`../validation/gates.md`](../validation/gates.md) (the gate table).

---

## 1. End-to-end phase flow

`config/workflows.yaml::workflows.textbook_to_course.phases` declares **21
phases**. The runner does not walk them in list order — it topologically sorts
them on their *effective* `depends_on`
(`MCP/core/workflow_runner.py::_topological_sort` → `_effective_depends_on`),
and several of those dependency lists change shape based on the environment.

The diagram below shows the graph as it resolves under `COURSEFORGE_TWO_PASS=true`,
which is the configuration a real production run uses.

```mermaid
flowchart TD
    subgraph CONV["Conversion + ingest"]
        P0["semantik_conversion<br/><i>semantik-converter</i>"]
        P1["heading_judge<br/><i>agents: []</i>"]
        P2["staging<br/><i>textbook-stager</i>"]
        P3["chunking<br/><i>semantik-chunker</i>"]
        P4["objective_extraction<br/><i>textbook-ingestor</i>"]
        P5["source_mapping<br/><i>source-router</i>"]
    end

    subgraph PLAN["Planning + knowledge"]
        P6["course_planning<br/><i>course-outliner</i>"]
        P7["concept_extraction<br/><i>pedagogy-graph-builder</i>"]
    end

    subgraph GEN["Generation (two-pass tier — see §3)"]
        P8["content_generation<br/>SKIPPED<br/>enabled_when_env:<br/>COURSEFORGE_TWO_PASS!=true"]
        P9["content_generation_outline"]
        P10["inter_tier_validation<br/><i>agents: []</i>"]
        P11["content_generation_rewrite"]
        P12["assessment_synthesis<br/><i>agents: []</i> · optional"]
        P13["post_rewrite_validation<br/><i>agents: []</i>"]
    end

    subgraph PKG["Package + archive"]
        P14["packaging<br/><i>brightspace-packager</i>"]
        P15["imscc_chunking<br/><i>semantik-chunker</i>"]
        P16["trainforge_assessment<br/>optional"]
        P17["training_synthesis<br/>SKIPPED<br/>optional · --skip-training"]
        P18["libv2_archival<br/><i>libv2-archivist</i>"]
        P19["vector_indexing<br/>optional"]
    end

    subgraph TRAIN["Training tail — opt-in (--with-training)"]
        P20["training<br/><i>agents: []</i><br/>optional"]
        P21["post_training_validation<br/><i>agents: []</i> · gates only<br/>optional"]
        P22["evaluation<br/><i>agents: []</i><br/>optional"]
    end

    subgraph FIN["Finalize"]
        P23["finalization"]
    end

    P0 --> P1 --> P2 --> P3 --> P4 --> P5
    P5 --> P6
    P3 --> P6
    P6 --> P7
    P3 --> P7

    P6 -.->|"single-pass only"| P8

    P6 --> P9
    P5 --> P9
    P2 --> P9
    P7 --> P9
    P9 --> P10 --> P11
    P11 --> P12
    P3 --> P12
    P11 --> P13
    P12 --> P13

    P13 --> P14
    P12 --> P14
    P14 --> P15
    P14 --> P16
    P15 --> P16
    P16 -.-> P17
    P15 -.-> P17
    P14 --> P18
    P16 --> P18
    P17 --> P18
    P18 --> P19
    P19 -.->|"opt-in"| P20 -.-> P21 -.-> P22
    P22 --> P23

    classDef skipped fill:#eee,stroke:#999,stroke-dasharray: 4 3,color:#555
    class P8,P17,P20,P21,P22 skipped
```

`finalization` depends on `evaluation`, not `vector_indexing`, so it is
genuinely last. A default build still reaches it: a skipped phase stamps
`_completed` in `phase_outputs`, and `_dependencies_met` reads only that.

### 1.1 Non-obvious facts about the current shape

* **`semantik_conversion`** is the conversion phase name. A run paused under an
  older persisted phase key still resumes: `MCP/hardening/checkpoint.py` carries
  a bidirectional `_PHASE_NAME_ALIASES` map so a checkpoint written under either
  key is found under the other, and `MCP/core/workflow_runner.py` normalizes the
  legacy key in persisted `phase_outputs` (`_LEGACY_CONVERSION_PHASE`,
  `workflow_runner.py:372`). Nothing *writes* the legacy name — it is a
  read-only resume-compat alias.

* **`heading_judge`** is a real phase between conversion and staging, not an
  optional add-on. It is declared non-optional with `agents: []`, so the runner
  always dispatches it; the *skip* happens inside the handler
  (`_run_heading_judge`, `MCP/tools/pipeline_tools.py:20275`), which
  short-circuits with `success: true, skipped: true` when
  `SEMANTIK_HEADING_JUDGE` is off, and again when the corpus has no
  `*.glmocr_layout.json` sidecars (a born-digital corpus). This is
  skip-with-pass, not phase-skip: a checkpoint is still written.

* **`assessment_synthesis` runs BEFORE `post_rewrite_validation`** in
  `textbook_to_course` — but only under two-pass, and the ordering is inverted
  in `course_generation`. See §1.2.

### 1.2 The `assessment_synthesis` / `post_rewrite_validation` ordering inversion

Three phases carry a `depends_on_when_env` / `depends_on_when_env_value` pair.
When the predicate holds, the alt list **replaces** `depends_on` for both the
dependency check and the topological sort (`_effective_depends_on`,
`workflow_runner.py:7379-7396`).

In `textbook_to_course`:

| phase | static `depends_on` | under `COURSEFORGE_TWO_PASS=true` |
|---|---|---|
| `assessment_synthesis` | `content_generation`, `chunking` | `content_generation_rewrite`, `chunking` |
| `post_rewrite_validation` | `content_generation_rewrite` | `content_generation_rewrite`, **`assessment_synthesis`** |
| `packaging` | `content_generation`, `assessment_synthesis` | `post_rewrite_validation`, `assessment_synthesis` |

So in `textbook_to_course` the assessments are emitted **before** the final
validation pass, and `post_rewrite_validation` therefore observes them.

In `course_generation` the edge points the other way:

| phase | static `depends_on` | under `COURSEFORGE_TWO_PASS=true` |
|---|---|---|
| `assessment_synthesis` | `content_generation` | **`post_rewrite_validation`** |

`course_generation` also has no `depends_on_when_env` on
`post_rewrite_validation` at all. Its validation runs first; assessments follow.
The two workflows genuinely disagree on this ordering — do not port a diagram
from one to the other.

### 1.3 Optional and env-gated phases

Three mechanisms skip a phase entirely, all in
`workflow_runner.py::_should_skip_phase` (`:4997`):

**`enabled_when_env`** — a `"VAR=value"` / `"VAR!=value"` predicate evaluated
against the live environment (`_eval_enabled_when_env`, `:5227`). The literal
`true` matches any of `1` / `true` / `yes` / `on`, case-insensitively. A
*malformed* predicate returns `True` (enabled), so a typo surfaces as an
unexpected run rather than a silent no-op. Five phases carry one:

| phase | predicate |
|---|---|
| `content_generation` | `COURSEFORGE_TWO_PASS!=true` |
| `content_generation_outline` | `COURSEFORGE_TWO_PASS=true` |
| `inter_tier_validation` | `COURSEFORGE_TWO_PASS=true` |
| `content_generation_rewrite` | `COURSEFORGE_TWO_PASS=true` |
| `post_rewrite_validation` | `COURSEFORGE_TWO_PASS=true` |

`content_generation` and the four two-pass phases are mutually exclusive by
construction: exactly one of the two paths can satisfy its predicate.

**`optional: true` + a workflow param** — checked only for phases marked
optional:

| phase | skipped when |
|---|---|
| `assessment_synthesis` | `generate_assessments` is false (`--no-assessments`) |
| `trainforge_assessment` | `generate_assessments` is false |
| `training_synthesis` | `skip_training` is true (`--skip-training`) |
| `vector_indexing` | embedding stack unavailable, unless `TRAINFORGE_REQUIRE_EMBEDDINGS` |
| `training` | `with_training` is not true, or `skip_training` is true (`--skip-training` wins) |
| `post_training_validation` | same as `training` |
| `evaluation` | same as `training` |

The training-tail branch sits **after** the `not phase.optional` guard, and
`trainforge_train`'s same-named phases are not optional — so the standalone
training workflow is structurally immune to it, with no name-based
special-casing.

**The `courseforge_stage` whitelist** (`_should_skip_for_courseforge_stage`,
`:5198`) skips everything outside the named tier for the
`ed4all run courseforge-*` stage subcommands. It runs before the optional-phase
check, so it can skip non-optional phases too.

**Skips are not bare stubs.** When a phase skips, the runner *merges* the skip
markers into whatever was already in `phase_outputs[phase]` rather than
overwriting it (`workflow_runner.py:2273-2278`). That is why downstream
`inputs_from` references to a skipped `content_generation` still resolve —
pre-populated `project_id` / `content_dir` / `content_paths` survive alongside
`_skipped: true, _completed: true`.

### 1.4 How a phase can fail and the run continue

There are four distinct outcomes, and they are easy to confuse.

1. **A `warning`-severity gate fails.** Nothing happens to the run. The
   executor clears `gates_passed` **only** for a gate whose declared severity is
   `CRITICAL` (`MCP/core/executor.py:2377-2380`). A `severity: warning` gate
   that also declares `behavior.on_fail: block` still does not block —
   `on_fail: block` is not consulted at this seam. In the reference run,
   `content_grounding`, `wcag_compliance`, and `chunkset_drift` all failed this
   way and the run continued.

2. **A `critical` gate fails.** `gates_passed` goes false. The executor
   immediately stamps the phase checkpoint `status: "failed"` via
   `checkpoint_manager.fail_phase` (`executor.py:2414-2429`) — *regardless of
   what the runner decides next*. The runner then stops the workflow
   (`final_status = "FAILED"`) **unless** the phase is `optional: true`
   (`workflow_runner.py:2693`, `if not gates_passed and not getattr(phase, "optional", False)`).
   So an optional phase can fail a critical gate and the run marches on with a
   `failed` checkpoint on disk.

3. **`course_planning` specifically has a retry/fail-open path.** When
   `ED4ALL_PLANNING_GATE_RETRIES > 0` (default `0`), a gate failure routes into
   `_retry_course_planning_gates` (`workflow_runner.py:2711`), which re-rolls
   the nondeterministic objective synthesis with a per-attempt salt. On budget
   exhaustion it stamps `_planning_gate_retries_exhausted` and lets the run
   continue. It does **not** re-stamp the phase checkpoint, so the checkpoint
   can read `failed` while the workflow state reads `_gates_passed: true`.

4. **A task errors, times out, or returns `success=false`.** The runner stops
   the workflow unless the phase is optional (`workflow_runner.py:2680-2691`).

**Reading a completed run's state:** a checkpoint stamped `failed` does not mean
the run failed, and `config/workflows.yaml` — not the persisted checkpoint — is
authoritative for whether a gate blocks. The `severity` recorded on a persisted
gate result is back-filled from the YAML **only when the `GateResult` left it
`None`** (`executor.py:2394-2397`); validators that set their own severity
persist a value that can disagree with the config.

There is **no operator waiver feature for phase gates**. `GateManager` has a
`GateResult.waiver_info` surface, but nothing in the runner lets an operator
wave a failed phase through — a hand-edited state file is a hand-edited state
file, not a supported path.

---

## 2. Dispatch routing: phase name vs agent name

A phase declares `agents: [...]` in YAML, but the agent name is not always what
selects the handler. `MCP/core/executor.py:858-860` resolves in this order:

```mermaid
flowchart TD
    A["phase from config/workflows.yaml"] --> B{"phase name in<br/>_PHASE_TOOL_MAPPING?"}
    B -- yes --> C["tool = _PHASE_TOOL_MAPPING[phase]<br/>(7 phases — wins over the agent)"]
    B -- no --> D{"agent_type in<br/>AGENT_TOOL_MAPPING?"}
    D -- yes --> E["tool = AGENT_TOOL_MAPPING[agent_type]"]
    D -- no --> ERR["no tool name →<br/>ExecutionResult status=ERROR<br/>executor.py:862-870"]
    C --> INV["TaskExecutor._invoke_tool"]
    E --> INV
    INV --> J{"ED4ALL_AGENT_DISPATCH on<br/>AND dispatcher injected<br/>AND agent in AGENT_SUBAGENT_SET?"}
    J -- yes --> SUB["dispatcher.dispatch_task<br/>(subagent via mailbox bridge)"]
    J -- no --> G["_build_tool_registry lookup<br/>MCP/tools/pipeline_tools.py"]
```

Note the ordering: `ED4ALL_AGENT_DISPATCH` is **not** a fallback for a mapping
miss. A miss on both maps returns an `ExecutionResult` with `status="ERROR"`
(`executor.py:862-870`). The subagent fork happens *after* a tool name is
already resolved, inside `_invoke_tool` (`executor.py:1329-1338`), and only for
agents classified in `AGENT_SUBAGENT_SET`; several provider-set overrides
(`COURSEFORGE_PROVIDER`, the course-planner, Trainforge assessment/synthesis)
force the in-process path even when the flag is on.

`_PHASE_TOOL_MAPPING` (`executor.py:263`) has seven entries, all in
`textbook_to_course`:

| phase | tool | `agents:` in YAML | why the override exists |
|---|---|---|---|
| `heading_judge` | `run_heading_judge` | `[]` | only route |
| `content_generation_outline` | `run_content_generation_outline` | `["content-generator"]` | that agent otherwise maps to the single-pass `generate_course_content` |
| `inter_tier_validation` | `run_inter_tier_validation` | `[]` | only route |
| `content_generation_rewrite` | `run_content_generation_rewrite` | `["content-generator"]` | same agent, different tier |
| `assessment_synthesis` | `run_assessment_synthesis` | `[]` | only route |
| `post_rewrite_validation` | `run_post_rewrite_validation` | `[]` | only route |
| `imscc_chunking` | `run_imscc_chunking` | `["semantik-chunker"]` | that agent otherwise maps to the staged-HTML chunking tool; the override is what selects the IMSCC-side chunker |

Two consequences that are not obvious from the YAML:

* **The four `agents: []` phases have no fallback.**
  `workflow_runner._create_phase_tasks` synthesizes a virtual `phase-handler`
  task *only* for phases present in `_PHASE_TOOL_MAPPING`. Remove the mapping
  row and no task is created at all — the phase silently produces nothing
  rather than erroring.

* **`chunking` and `imscc_chunking` share one agent** (`semantik-chunker`) and
  differ only by the phase-name override. `chunking` falls through to the
  staged-HTML chunking tool and emits the staged-HTML chunkset with
  `chunkset_kind="semantik"`; `imscc_chunking` is overridden to
  `run_imscc_chunking` and emits the packaged-IMSCC chunkset with
  `chunkset_kind="imscc"`. The two chunksets are later compared by the
  `chunkset_drift` gate at `libv2_archival`.

The remaining phases route by agent name through `AGENT_TOOL_MAPPING`. That map
also carries a few live read-compat aliases — legacy agent names from before the
SemantiK rename that point at the current conversion and chunking tools — reached
by string from older persisted state, with no import edge to them.

---

## 3. The two-pass generation tier

Under `COURSEFORGE_TWO_PASS=true`, content authoring splits into an outline tier
(structure, no HTML body) and a rewrite tier (final HTML), with a validator-only
phase on each side.

```mermaid
flowchart TD
    OBJ["synthesized_objectives.json"] --> OUT
    KG["concept_graph_semantic.json<br/>domain_concept_vocabulary.json"] --> OUT
    SRC["source_module_map.json<br/>semantik_chunks/chunks.jsonl"] --> OUT

    OUT["<b>content_generation_outline</b><br/>CourseforgeRouter.route(tier='outline')<br/>via route_with_self_consistency"]

    OUT -->|"candidate fails<br/>validator chain"| REGEN["re-sample candidate<br/>+ remediation suffix"]
    REGEN --> OUT
    OUT -->|"validation_attempts &ge; regen budget<br/>(default 10)"| ESC["stamp escalation_marker=<br/>outline_budget_exhausted"]
    OUT -->|"spec.escalate_immediately<br/>(policy skip — 0 candidates)"| ESC

    OUT --> MINT["<b>_mint_outline_curies</b><br/>stamp per-course CURIEs on blocks<br/>whose curies[] is empty<br/>(no-op without a vocabulary file)"]
    ESC --> MINT
    MINT --> BOUT["01_outline/blocks_outline.jsonl<br/>+ outline_chunks.json<br/>+ outline_objectives.json"]

    BOUT --> ITV["<b>inter_tier_validation</b><br/>agents: [] · no LLM dispatch<br/>Courseforge/router/inter_tier_gates.py"]
    ITV --> BV["01_outline/blocks_validated.jsonl"]
    ITV --> BFAIL["01_outline/blocks_failed.jsonl"]
    ITV --> ITREP["&lt;export&gt;/02_validation_report/report.json<br/><i>(runner-written, sibling of the stage dir)</i>"]

    BV --> RW["<b>content_generation_rewrite</b><br/>route_rewrite_with_remediation"]
    ESC -.->|"escalation_marker present →<br/>_render_escalated_user_prompt<br/>(synthesize from chunks, not the draft)"| RW

    CACHE["failure-driven reuse:<br/>04_rewrite/blocks_final.jsonl<br/>+ .blocks_final_checkpoint.jsonl<br/>(fingerprint-matched hits skip dispatch)"] --> RW
    RW -->|"validator fail &ge; rewrite budget<br/>(default 10)"| ESC2["escalation_marker=<br/>validator_consensus_fail"]

    RW --> SWEEP["<b>final CURIE-anchoring sweep</b><br/>prose-mint a hidden data-cf-curie span<br/>for shipping blocks with no<br/>gate-extractable CURIE"]
    ESC2 --> SWEEP
    SWEEP --> BFIN["04_rewrite/blocks_final.jsonl<br/>+ 03_content_development/*.html"]

    BFIN --> PRV["<b>post_rewrite_validation</b><br/>agents: [] · the same validators<br/>re-run on rewrite-tier prose"]
    ASSESS["06_assessments/*.xml"] --> PRV
    PRV --> REPORT["04_rewrite/blocks_validated.jsonl<br/>04_rewrite/blocks_failed.jsonl<br/>04_rewrite/02_validation_report/report.json<br/><i>(report written by the runner, post-phase)</i>"]

    SP["content_generation (single-pass)<br/>enabled_when_env:<br/>COURSEFORGE_TWO_PASS!=true"]
    SP -.->|"mutually exclusive —<br/>never runs alongside these tiers"| OUT

    classDef skipped fill:#eee,stroke:#999,stroke-dasharray: 4 3,color:#555
    class SP skipped
```

### 3.1 Regeneration and escalation

Escalation is bounded, not open-ended. Inside
`CourseforgeRouter.route_with_self_consistency`, a cumulative
`validation_attempts` counter increments on every failed validator pass; when it
meets the resolved regeneration budget the loop breaks and the last candidate is
rebound with `escalation_marker="outline_budget_exhausted"`
(`CourseforgeRouter.route_with_self_consistency`, `router.py:1559`; the rebinding
`dataclasses.replace` calls at `router.py:1966` and `:1995`). Budget resolution
order, highest first:
per-call kwarg → `policy.regen_budget_by_block_type` →
`COURSEFORGE_OUTLINE_REGEN_BUDGET` → constructor attribute → module default
`_DEFAULT_OUTLINE_REGEN_BUDGET = 10` (`router.py:244`). The rewrite tier has a
symmetric `_DEFAULT_REWRITE_REGEN_BUDGET = 10` and stamps
`validator_consensus_fail` on exhaustion.

A second path reaches the same marker without generating anything: when the
routing policy sets `escalate_immediately` for a block type, the outline
dispatch is skipped entirely and the return block is stamped
`outline_budget_exhausted` (`router.py:1246`, `short_circuit_marker`). The two
paths are indistinguishable by marker — `_ESCALATION_MARKERS`
(`Courseforge/scripts/blocks.py:368`) is a closed frozenset that does not admit
a separate policy-skip value. The discriminator is a
`Touch(purpose="escalate_immediately")` audit record on the block.

The closed set has **ten** members:

| marker | fires when |
|---|---|
| `outline_budget_exhausted` | outline regen budget exhausted, **or** a policy `escalate_immediately` short-circuit |
| `validator_consensus_fail` | rewrite regen budget exhausted |
| `structural_unfixable` | a structural miss the regen loop cannot repair |
| `outline_dispatch_error` / `rewrite_dispatch_error` | a per-block provider failure inside `CourseforgeRouter.route_all` |
| `per_claim_attribution_unfixable` | outline budget exhausted purely on per-claim source-attribution misses |
| `block_objective_undelivered` | rewrite budget exhausted purely on block-objective delivery misses |
| `best_of_n_no_clean_candidate` | no best-of-N candidate cleared the verifier |
| `input_prompt_truncated` | the composed prompt exceeded the serving window |
| `rewrite_scaffold_overflow` | the rewrite scaffold exceeded its budget |

The two `*_dispatch_error` markers exist so a per-block provider failure
surfaces as an escalated block rather than a silently dropped one — before they
existed the packager-side `escalation_marker is not None` filter never saw the
block and the cartridge shipped without it.

The authoritative list is `_ESCALATION_MARKERS` in
`Courseforge/scripts/blocks.py:368`; `Block.__post_init__` (`blocks.py:1269`)
rejects any marker outside it, so this set is closed by construction.

Either way, the rewrite tier reads the marker and switches from
`RewriteProvider._render_user_prompt` to `_render_escalated_user_prompt`, which
synthesizes from source chunks and objectives directly rather than refining an
outline draft it does not trust.

### 3.2 Where cache reuse happens

The rewrite tier is the expensive phase, and it reuses aggressively
(`MCP/tools/pipeline_tools.py:18346-18420`):

* An existing `04_rewrite/blocks_final.jsonl` is read into a per-`block_id`
  cache. Only entries with real string content **and no escalation marker** are
  cached — a degraded block is deliberately left uncached so it gets another
  attempt.
* `blocks_final.jsonl` is written once at the very end, so a mid-loop kill
  leaves nothing there. The append-as-you-go
  `04_rewrite/.blocks_final_checkpoint.jsonl` sidecar covers that gap; its
  entries merge into the same cache.
* A sidecar entry is only honoured when its stamped **input fingerprint** matches
  the recomputed one for that `block_id`, so a model/provider swap or changed
  source chunks re-authors instead of silently reusing a stale rewrite. Entries
  with no stamp keep legacy `block_id`-only reuse.

Three additive eviction scopes let an operator force specific blocks to
re-author while every out-of-scope block keeps byte-identical reuse:
`--blocks` (per block *type*, wired as `target_block_ids`), `--block-ids`
(exact instances, `target_block_instance_ids`), and `--pages` (a `page_id` or a
module prefix, `target_page_ids`). All three unset means pure failure-driven
reuse. `--force` clears the crash sidecar but deliberately leaves the
`blocks_final.jsonl` cache intact, because the `--blocks` byte-identity
contract depends on it.

### 3.3 Where the CURIE mint and the anchor sweep sit

Two separate deterministic passes, at opposite ends of the tier:

* **Outline mint** — `_mint_outline_curies` (`pipeline_tools.py:16892`) runs at
  the *end* of the outline phase, after objective-ref repair and before
  `blocks_outline.jsonl` is written. A prose corpus carries no RDF CURIEs in its
  text, so every outline block would otherwise land with an empty
  `content["curies"]` and fail the anchoring gate outright. When a
  `domain_concept_vocabulary.json` exists for the course, this pass mints a
  per-course CURIE per vocabulary concept and stamps matches onto every block
  that *needs* a domain CURIE — which is broader than "empty": a block
  qualifies when `content["curies"]` is empty **or** when it carries only
  generic / non-domain CURIEs (none of them a key in the per-course minted
  map, e.g. the placeholder CURIEs a small model emits). In that second case
  the minted domain CURIE is **appended**; existing CURIEs are never deleted,
  and a block already carrying a real minted domain CURIE is left untouched.
  Matching runs over the block's `key_claims` text and, when that misses, over
  the block's grounded source-chunk text; a block with neither mints nothing
  and fails closed downstream. No vocabulary file means a complete no-op — the absence of
  the file is the gate, no behavior flag involved. The minter must only assign
  CURIEs that the gate's own anchoring rule would accept; a minter that stamps
  a CURIE the gate cannot anchor launders the wrong concept forward, because
  the rewrite tier force-injects the declared CURIE as a literal hidden span
  that then passes `post_rewrite_validation` by construction.

* **Final anchor sweep** — in `_run_content_generation_rewrite`
  (`pipeline_tools.py:19483-19540`), after rewrite and before the HTML
  tag-balance repair. For each shipping block with no gate-extractable CURIE in
  its body, it strips any junk `data-cf-curie` span, mints a CURIE from the
  block's **own prose** (surface-form match only — anti-fabrication), and
  appends it as a hidden `<span hidden data-cf-curie="...">`. It mirrors the
  gate's audit universe exactly, skipping unshipped escalation tombstones and
  deterministic template blocks. Idempotent, no LLM, no GPU.

---

## 4. Artifact flow

Filenames below are the ones the code writes. `<export>` is the Courseforge
project export dir; `<libv2_course>` is the LibV2 course dir.

```mermaid
flowchart TD
    PDF["corpus — source PDFs<br/>or publisher HTML"]
    PDF -->|"semantik_conversion"| HTML["{stem}_accessible.html<br/>+ {stem}.glmocr_layout.json<br/>+ quality sidecars"]
    HTML -->|"heading_judge"| HJ["judged {stem}_accessible.html<br/>({stem}_accessible.html.prejudge.bak kept)<br/>+ {stem}.heading_judgments.json<br/>+ {stem}.corrected_layout.json"]
    HJ -->|"staging"| STG["Courseforge staging dir<br/>+ role-tagged staging manifest"]

    STG -->|"chunking"| CH["&lt;libv2_course&gt;/semantik_chunks/<br/>chunks.jsonl + manifest.json"]
    STG -->|"objective_extraction"| TS["&lt;export&gt;/01_learning_objectives/<br/>textbook_structure.json"]
    TS -->|"source_mapping"| SM["&lt;export&gt;/source_module_map.json"]

    TS -->|"course_planning"| OBJ["&lt;export&gt;/01_learning_objectives/<br/>synthesized_objectives.json"]
    CH --> OBJ

    CH -->|"concept_extraction"| KG["&lt;libv2_course&gt;/concept_graph/<br/>concept_graph_semantic.json<br/>domain_concept_vocabulary.json<br/>manifest.json"]
    OBJ --> KG
```

```mermaid
flowchart TD
    OBJ["synthesized_objectives.json"] --> BO
    KG["concept_graph_semantic.json<br/>domain_concept_vocabulary.json"] --> BO
    SM["source_module_map.json"] --> BO
    CH["semantik_chunks/chunks.jsonl"] --> BO

    BO["&lt;export&gt;/01_outline/<br/>blocks_outline.jsonl<br/>outline_chunks.json<br/>outline_objectives.json"]
    BO -->|"inter_tier_validation"| BV["01_outline/blocks_validated.jsonl<br/>01_outline/blocks_failed.jsonl"]
    BV -->|"content_generation_rewrite"| BF["&lt;export&gt;/04_rewrite/blocks_final.jsonl<br/>+ &lt;export&gt;/03_content_development/*.html"]

    BF -->|"assessment_synthesis"| AS["&lt;export&gt;/06_assessments/<br/>week_NN_quiz.xml (QTI 1.2)<br/>week_NN_discussion.xml (imsdt)<br/>week_NN_assignment.xml<br/>+ manifest.json"]
    CH --> AS

    BF -->|"post_rewrite_validation"| VR["04_rewrite/blocks_validated.jsonl<br/>04_rewrite/blocks_failed.jsonl<br/>04_rewrite/02_validation_report/report.json"]
    AS --> VR

    VR -->|"packaging"| PK["&lt;export&gt;/05_final_package/&lt;course&gt;.imscc<br/>+ packaging_report.json"]
    AS --> PK
```

```mermaid
flowchart TD
    PK["&lt;course&gt;.imscc"] -->|"imscc_chunking"| IC["&lt;libv2_course&gt;/imscc_chunks/<br/>chunks.jsonl + manifest.json"]
    PK -->|"trainforge_assessment"| TA["&lt;export&gt;/trainforge/<br/>assessments.json, course.json,<br/>objectives.json, manifest.json"]
    IC --> TA

    TA -.->|"training_synthesis<br/>(optional — skipped by --skip-training)"| SFT["training_specs/<br/>instruction_pairs.jsonl (SFT)<br/>preference_pairs.jsonl (DPO)"]
    IC -.-> SFT

    TA -->|"libv2_archival"| LV["&lt;libv2_course&gt;/<br/>manifest.json (course manifest),<br/>objectives.json,<br/>source/, corpus/, graph/, quality/"]
    PK --> LV
    IC --> LV

    LV -->|"vector_indexing"| VI["&lt;libv2_course&gt;/vector_index/<br/>embeddings.npy, id_map.json,<br/>manifest.json"]
    VI -->|"finalization"| FIN["final package re-stamp<br/>+ training-capture export"]

    LV -->|"post-loop aggregators"| AG["coverage_map.json<br/>courseforge_promotion_chain_report.json<br/>build_cost_report.json<br/>quality/*.json"]
```

Notes on the chain:

* **The chunkset is produced twice.** `semantik_chunks/` comes from the staged
  HTML at `chunking`; `imscc_chunks/` comes from the packaged cartridge at
  `imscc_chunking`. The `chunkset_drift` gate at `libv2_archival` compares them.
  Both dir names are constants in `lib/libv2_storage.py` (`SEMANTIK_CHUNKS_DIRNAME`,
  `IMSCC_CHUNKS_DIRNAME`).
* **`project_id`** (emitted by `objective_extraction`) is the most fanned-out
  single output — it is threaded by `inputs_from` into course planning, concept
  extraction, both generation tiers, both validator phases, assessment
  synthesis, packaging, and finalization.
* `vector_index/` is byte-reproducible: the stated determinism contract is same
  machine + venv + provider + model + `device=cpu` + batch size ⇒
  `embeddings.npy` and `id_map.json` are byte-identical across rebuilds, and
  `manifest.json` is identical modulo the single optional `generated_at` field,
  which is excluded from every content hash
  (`LibV2/tools/libv2/vector_index.py:18-25`).
* The post-loop aggregators are best-effort: an aggregator failure logs a
  warning and never changes `final_status`. Several write nothing at all when
  their env flag is off — see [`aggregators.md`](aggregators.md).

---

## 5. Checkpoints, resume, and stop

### 5.1 Two state surfaces, and only one of them drives resume

| surface | written by | what it is |
|---|---|---|
| `state/runs/<RUN_ID>/checkpoints/<phase>_checkpoint.json` | `MCP/hardening/checkpoint.py::CheckpointManager` (`checkpoints_dir`, `checkpoint.py:103`), called from `executor.py::execute_phase` | per-phase execution record + the full gate chain |
| `state/workflows/<WORKFLOW_ID>.json` → `phase_outputs[<phase>]` | `workflow_runner.py::_save_workflow_state` | **what `--resume` actually reads** |

`--resume` skips a phase only when its recorded output has `_completed` **and**
`_gates_passed` is not `False` (`workflow_runner.py:2253-2257`). An absent
`_gates_passed` defaults to skip, for backward compatibility with older state
and with optional-phase skip markers. A phase whose gates failed is therefore
re-run on resume rather than marched past.

### 5.2 Per-unit resume sidecars

Phase checkpoints are coarse — they resume you to a phase boundary. The
expensive LLM phases additionally write fingerprinted per-unit sidecars, so a
kill mid-phase costs at most one in-flight call. All are dot-prefixed JSONL
next to the artifact they rebuild, and all are governed by the
`ED4ALL_GENERATION_CHECKPOINT` family with per-site overrides:

| phase | sidecar | site flag |
|---|---|---|
| `course_planning` | `01_learning_objectives/.stage2_windows_checkpoint.jsonl`, `.stage2_clusters_checkpoint.jsonl` | `ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT` |
| `concept_extraction` | `concept_graph/.concept_extraction_checkpoint.jsonl` | `ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT` |
| `content_generation_outline` | `01_outline/.blocks_outline_checkpoint.jsonl`, `.block_planner_weeks_checkpoint.jsonl` | `COURSEFORGE_OUTLINE_CHECKPOINT` |
| `content_generation_rewrite` | `04_rewrite/.blocks_final_checkpoint.jsonl` | `COURSEFORGE_REWRITE_CHECKPOINT` |
| `assessment_synthesis` | `06_assessments/.assessments_checkpoint.jsonl` | (`ED4ALL_GENERATION_CHECKPOINT` family) |
| `post_rewrite_validation` | `.prose_entailment_cache/` beside `blocks_final.jsonl` | `ED4ALL_VALIDATION_CHECKPOINT` |

Sidecar entries are content-addressed. The rewrite sidecar keys on a per-block
input fingerprint; the prose-entailment cache keys on a hash of the prose plus
sorted cited chunk ids/texts, the floors, the scorer version, the NLI model
revision, and the device class. Change any of those and the entry misses and
re-computes rather than serving a stale verdict.

### 5.3 Stop sentinels

`ed4all stop` writes a filesystem sentinel that long-running stages poll at
their unit boundaries (`lib/generation/stop_control.py`):

* run-scoped: `state/runs/<RUN_ID>/control/` (`_run_sentinel_path`, `:123`)
* global: `state/runs/STOP_ALL` (`_global_sentinel_path`, `:119`), operator-owned
  — it both pauses running work and blocks new/resumed runs until
  `ed4all stop --clear-all`

The in-flight unit finishes, checkpoints to its sidecar, and the phase pauses
(exit code 3). Resume with a **plain** `--resume`; `--force` clears the resume
sidecars, which is the opposite of what you want after a deliberate stop.

### 5.4 GPU lease hand-off

`_gpu_lifecycle_sweep` has three call sites, and which one fires tells you how
the phase ended:

* **`workflow_runner.py:2772`** — the normal boundary. Runs after task results
  are in, gates passed, and outputs are persisted, and before the next phase
  dispatches.
* **`:2229`** — a graceful stop observed *before* a phase started.
* **`:2486`** — a graceful stop that paused a phase mid-flight.

The two stop paths sweep deliberately, so a paused run leaves a clean card for
the resume. Nothing sweeps on a **failed** or **partial** phase — those `break`
out above the call — and nothing sweeps for a resume-skipped phase, which
`continue`s earlier still. The sweep is best-effort by construction, so a sweep
failure cannot change `final_status`.
Phases declare their vLLM seats via a `seats:` annotation in
`config/workflows.yaml`; see [`gpu-seat-residency.md`](gpu-seat-residency.md).
