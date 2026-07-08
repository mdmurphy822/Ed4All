"""
Workflow Runner - Executes multi-phase workflows end-to-end.

This module provides the missing orchestration layer that chains
workflow phases together, routing outputs from each phase into
the next phase's inputs.

Usage:
    runner = WorkflowRunner(executor, config)
    result = await runner.run_workflow(workflow_id)
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_LO_ID_RE = re.compile(r"^[a-zA-Z]{2,}-\d{2,}$")

from .config import OrchestratorConfig, WorkflowPhase
from .executor import (
    _PHASE_TOOL_MAPPING,
    AGENT_AUTHORING_PROVIDER_ENV_MAP,
    AGENT_PROVIDER_ENV_MAP,
    AGENT_SUBAGENT_SET,
    ExecutionResult,
    TaskExecutor,
    _agent_dispatch_enabled,
)

# VRAM-contention doctor — per-phase forensic free-VRAM trajectory. Imported
# at module scope (rather than lazily inside the hook) so a default-off run
# pays only the import cost and tests can patch these symbols on this module.
# The whole hook is gated behind ``vram_doctor_enabled()`` (default OFF) and
# wrapped best-effort, so the default path never calls ``snapshot_vram`` and a
# doctor failure can never affect the run. See lib/llm/vram_doctor.py.
from lib.llm.vram_doctor import (
    snapshot_vram,
    vram_doctor_enabled,
    write_trajectory_row,
)

# Graceful stop ("checkpoint on command"): the run loop probes a filesystem
# stop sentinel between phases and halts to a PAUSED status (never FAILED). The
# module is stdlib-only, selects no provider/model (no LICENSING / behavior-flag
# rows), and resolves its sentinel paths through ``lib.paths.get_state_runs_dir``
# — the SAME parent the executor writes phase checkpoints into. See
# lib/generation/stop_control.py.
from lib.generation.stop_control import (
    GLOBAL_SENTINEL_NAME,
    clear_stop,
    stop_requested,
)
from lib.paths import DART_PATH, get_state_runs_dir


class AuthoringProviderRouteError(RuntimeError):
    """Raised when a run would dispatch an LLM-needing phase to an
    unserviced mailbox (or a silent templated stub) instead of resolving
    its generation through the in-process provider lattice.

    Marketable-v1 A3 fail-fast guardrail: a GUI-launched / headless run
    must never enqueue a mailbox ``agent_task`` that nobody will service
    (the run would hang forever) and must never silently degrade an
    LLM-needing agent to a templated in-process stub. When neither a
    per-agent provider env nor an explicit session/stub opt-in is present,
    the run fails fast with an actionable message naming the fix.
    """

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# state/ is a relocatable data root: source it from lib.paths so an ED4ALL_HOME
# deployment lands workflow state under the mounted data root. Byte-stable to
# ``PROJECT_ROOT / "state"`` when ED4ALL_HOME is unset (lib.paths.STATE_PATH
# == PROJECT_ROOT/state in that case). Import is local so a bare lib.paths
# import failure (very unlikely) degrades to the in-tree default.
try:
    from lib.paths import STATE_PATH as _LIB_STATE_PATH

    STATE_PATH = _LIB_STATE_PATH
except Exception:  # noqa: BLE001 — defensive: fall back to in-tree default
    STATE_PATH = PROJECT_ROOT / "state"


# ------------------------------------------------------------------------
# Marketable-v1 A5 — corpus-generalization defaults-on for pipeline runs.
#
# The corpus-generalization features (dynamic CURIE minting, page-level
# concept tags, the measured-best graph-shaping quartet, three-stage
# textbook synthesis) are gated behind opt-in env flags that default OFF so
# *bare library calls* and *rebuilds of legacy calibration corpora* stay
# byte-identical. But a fresh CLI/GUI textbook run is exactly the case where
# these features are the product — without them a general textbook quietly
# misses LO-refs, concept tags, and minted CURIEs. So the WORKFLOW RUNNER
# turns the blessed set ON per-run (mechanism "auto-on per run"): both the
# CLI (``create_textbook_pipeline`` → orchestrator → ``run_workflow``) and
# the GUI (``run_service`` → orchestrator → ``run_workflow``) flow through
# here, while a direct ``lib.ontology`` / ``Trainforge.chunker`` call that
# never touches ``run_workflow`` keeps the legacy default-off contract.
#
# The MEASURED-BEST graph config wins over the roadmap prose: page-level
# concept tags (chunk-local tags fragment the graph), plus the
# prune+fragfilter+merge+cap quartet + tech-seed + content-aware typing +
# label-normalize + intra-chunk links + lexical-seed fallback + objective
# quality gate. ``TRAINFORGE_CHUNK_LOCAL_TAGS`` is deliberately ABSENT (it
# stays default-off → page-level tags are emitted).
#
# Each value is applied with setdefault semantics (only when the env is
# unset/empty) so an operator's explicit legacy value (e.g.
# ``TRAINFORGE_MERGE_DUPLICATE_CONCEPTS=false``) is honored verbatim. Read
# sites all consult ``os.environ`` at call time, so a run-scoped set takes
# effect for every downstream phase handler.
_CORPUS_GENERALIZATION_ENV_DEFAULTS: Dict[str, str] = {
    # Chunking stage (PAGE-level tags — TRAINFORGE_CHUNK_LOCAL_TAGS left unset)
    "TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS": "true",
    "TRAINFORGE_SEED_TECH_CONCEPTS": "true",
    "TRAINFORGE_FILTER_FRAGMENT_CONCEPTS": "true",
    "TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE": "true",
    # M1: drop front-matter / donor / marketing contamination chunks (cover,
    # author/copyright/ISBN/Creative-Commons block, donor acknowledgements,
    # Table of Contents, "Key Features"/"Additional Resources" preface) BEFORE
    # dedup/emit, so they can't be cited by objectives or pollute grounding. A
    # full-textbook PDF carries ~6-8 such chunks at its head that the curated
    # baseline never ingested. Multi-signal scored with a hard math-content
    # veto (a chunk with real equations/worked examples is NEVER dropped) and
    # the "Chapter 1 Foundations"-vs-donor-"Foundation, Inc." false-positive
    # guard. See Trainforge/chunker/frontmatter.py.
    "TRAINFORGE_DROP_FRONTMATTER": "true",
    # Knowledge-graph stage (measured-best shaping quartet)
    "TRAINFORGE_MERGE_DUPLICATE_CONCEPTS": "true",
    "TRAINFORGE_INTRA_CHUNK_LINKS": "true",
    "TRAINFORGE_RELATED_FANOUT_CAP": "8",
    "TRAINFORGE_NORMALIZE_LABELS": "true",
    # W3: cooccurrence pair-counting aggregated to the PAGE level (chunk-local
    # cooccurrence fragments the graph; page-level connects it). Node frequency
    # + occurrences stay chunk-level. "chunk" byte-stable legacy when unset.
    "TRAINFORGE_COOCCURRENCE_GROUP_BY": "page",
    # M4: degenerate-grouping guard for the PAGE aggregation above. When a DART
    # converter collapses an entire multi-chapter PDF into ONE ``lesson_id``,
    # every chunk folds into a single page-group → every cooccurrence pair lands
    # weight==1 → the related_from_cooccurrence rule (weight>=3) emits ZERO
    # edges and the KG backbone dies (measured 0 vs 556 on a full-course 7B
    # build). On a degenerate (<3 real groups) page/section level the guard
    # steps DOWN to a finer level (page→section→chunk) for pair-counting only —
    # real co-occurrence at a valid window, nodes/occurrences unchanged. No-op
    # (byte-stable) on any corpus that already resolves into ≥3 groups.
    "TRAINFORGE_COOCCURRENCE_GROUP_FALLBACK": "true",
    # Corpus-generalization recovery paths (general / non-RDF textbooks)
    "TRAINFORGE_LEXICAL_CONCEPT_SEEDS": "true",
    "TRAINFORGE_OBJECTIVE_QUALITY_GATE": "true",
    # Defensive heading-sanity filter: repair a chunk's section_heading to its
    # nearest clean ancestor heading when the upstream heading classifier
    # mis-tagged answer-key / exercise-prose / numeric noise as a heading. The
    # original noise text stays in the chunk body. Hardens chunk + retrieval
    # display quality against residual upstream heading mis-classification
    # (SemantiK retraining is the upstream root fix). See
    # lib/chunk_heading_sanity.py. Default-off (byte-stable) outside a run.
    "TRAINFORGE_HEADING_SANITY_FILTER": "true",
}

# The env var that selects the three-stage textbook-synthesis provider. It
# selects an LLM backend (licensing-sensitive), so its default-on value is
# resolved like the A3 authoring providers — ``LLM_PROVIDER`` (the run's
# global routing provider) > ``"local"`` (license-clean default) — rather
# than a hardcoded literal, and is applied with the same setdefault
# semantics. Turning synthesis on is what produces the Stage-3
# ``domain_concept_vocabulary.json`` that the outline/rewrite CURIE
# anchoring gates use as their ``minted_curie_map`` — i.e. dynamic CURIE
# minting rides on the synthesis provider being set; there is no separate
# CURIE behavior flag.
_TEXTBOOK_SYNTHESIS_PROVIDER_ENV = "TEXTBOOK_SYNTHESIS_PROVIDER"

# The env var that selects the Trainforge TRAINING-PAIR synthesis provider
# (``Trainforge/synthesize_training.py::run_synthesis`` via the
# ``training-synthesizer`` agent). Distinct from
# ``TEXTBOOK_SYNTHESIS_PROVIDER`` above: that selects the authoring-adjacent
# three-stage textbook-structure synthesis (concept vocabulary + objectives);
# THIS one selects the provider that paraphrases the chunk corpus into the
# instruction / preference pairs that LITERALLY become the SLM's training
# corpus (the trained adapter is a derivative work of these pairs). Per
# ``docs/LICENSING.md`` § "Synthesis providers" this surface must default to a
# license-clean provider and Claude must never author the pairs. It is
# resolved exactly like the A3 authoring providers — ``LLM_PROVIDER`` (the
# run's global routing provider) > ``"local"`` (license-clean default) — and
# applied with the same setdefault semantics, so a CLI ``ed4all run`` matches
# the GUI ``run_service._apply_authoring_route_env`` behavior (which already
# fills this env). Without this, a CLI run with ``TRAINFORGE_SYNTHESIS_PROVIDER``
# unset and ``ED4ALL_AGENT_DISPATCH=true`` would dispatch the
# ``training-synthesizer`` subagent to the Claude Code session — routing the
# training-pair corpus through a ToS-unclean provider by default.
_TRAINFORGE_SYNTHESIS_PROVIDER_ENV = "TRAINFORGE_SYNTHESIS_PROVIDER"

# W1 Gap A — providers that are local / ToS-clean OSS. Used by
# ``_apply_authoring_route_env`` to decide whether to redirect the two-pass
# block-routing policy to the all-local variant under the single switch.
# ``together`` is cloud but ToS-clean OSS; both keep the two-pass router off
# the anthropic ``large`` capability tier in the canonical
# ``block_routing.yaml``. Mirrors the resolved set the license-clean runbook
# (``docs/operations/license-clean-run.md``) documents.
_LOCAL_OSS_PROVIDERS: frozenset = frozenset({"local", "together"})

# W1 Gap A — the all-local block-routing variant the single switch points
# ``COURSEFORGE_BLOCK_ROUTING_PATH`` at when an all-local two-pass run is
# requested. Reuses the existing license-clean sibling rather than minting a
# third near-identical routing file (which would need its own schema
# regression + drift guard and immediately drift from the license-clean
# variant). Repo-root-relative; ``load_block_routing_policy`` resolves it.
_ALL_LOCAL_BLOCK_ROUTING_PATH = (
    "Courseforge/config/block_routing.license_clean.yaml"
)

# W1 Gap A — the tier-default rewrite/outline provider envs. Filling these
# under the single switch covers the no-policy-file edge case where the
# block-routing loader returns an empty policy and the router falls to
# ``COURSEFORGE_REWRITE_PROVIDER`` → ``_rewrite_provider.DEFAULT_PROVIDER``
# (``"anthropic"``). Both filled with setdefault semantics.
_TWO_PASS_TIER_PROVIDER_ENVS: Tuple[str, ...] = (
    "COURSEFORGE_REWRITE_PROVIDER",
    "COURSEFORGE_OUTLINE_PROVIDER",
)

_BLOCK_ROUTING_PATH_ENV = "COURSEFORGE_BLOCK_ROUTING_PATH"
_TWO_PASS_ENV = "COURSEFORGE_TWO_PASS"

# Hosted-large build profile SETUP — the block-routing variant the single
# switch points ``COURSEFORGE_BLOCK_ROUTING_PATH`` at on an explicit
# ``--provider nvidia`` (the vendor endpoint-registry key) two-pass run.
# Sibling of the all-local variant above; its ``large`` rewrite tier resolves
# ``provider: nvidia`` (the hosted large-model seat), with the outline-tier
# first draft staying on the local 7B Qwen. Repo-root-relative;
# ``load_block_routing_policy`` resolves it.
_HOSTED_LARGE_BLOCK_ROUTING_PATH = (
    "Courseforge/config/block_routing.nvidia_large.yaml"
)
# Deprecated compat alias (external scripts + tests reach the old name).
_NVIDIA_LARGE_BLOCK_ROUTING_PATH = _HOSTED_LARGE_BLOCK_ROUTING_PATH

# Hosted-large build profile SETUP — the canonical cloud-model env + its
# default large-model ID. The branch setdefaults NVIDIA_LARGE_MODEL to close
# the 30B-nano registry-default leak (config/endpoints.yaml ``nvidia``
# default_model is the nemotron-3-nano-30b, NOT the large model). The canonical
# cloud-model knob is NVIDIA_LARGE_MODEL / the YAML ``model`` field — NEVER
# COURSEFORGE_REWRITE_MODEL (that env's router projector fires for
# ``provider: local`` tiers ONLY, so it is dead on the cloud tier — see
# Courseforge/router/router.py:3223 + Trainforge/CLAUDE.md NVIDIA_LARGE_MODEL
# row).
_HOSTED_LARGE_MODEL_ENV = "NVIDIA_LARGE_MODEL"
_HOSTED_LARGE_MODEL_DEFAULT = "meta/llama-3.3-70b-instruct"
# Deprecated compat aliases (external scripts + tests reach the old names).
_NVIDIA_LARGE_MODEL_ENV = _HOSTED_LARGE_MODEL_ENV
_NVIDIA_LARGE_MODEL_DEFAULT = _HOSTED_LARGE_MODEL_DEFAULT

# Hosted-large build profile GAP-1 fix — the textbook-synthesis seat (the env
# the objective_extraction / course_planning / concept_extraction phases read,
# confirmed at Courseforge/generators/_textbook_synthesis_provider.py
# ENV_PROVIDER/ENV_MODEL). ``--provider`` only fills the four authoring envs
# and does NOT reach this seat, so WITHOUT this setdefault the synthesis phases
# would silently stay on the local 7B while the rest of the build ran on the
# hosted large seat.
_TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV = "TEXTBOOK_SYNTHESIS_PROVIDER"
_TEXTBOOK_SYNTHESIS_MODEL_ROUTE_ENV = "TEXTBOOK_SYNTHESIS_MODEL"

# Hosted-large build profile — the hosted cloud-seat endpoint-registry key.
_CLOUD_SEAT_PROVIDER = "nvidia"
# Deprecated compat alias (external scripts + tests reach the old name).
_NVIDIA_PROVIDER = _CLOUD_SEAT_PROVIDER

# Providers whose training-data outputs are ToS-restricted (corpus taint):
# selecting one for the SLM TRAINING-PAIR seat (TRAINFORGE_SYNTHESIS_PROVIDER)
# taints the corpus the adapter is a derivative work of. This set MIRRORS the
# licensing posture in ``docs/LICENSING.md`` § "Synthesis providers" (the single
# source of truth) AND the checked-in copy at
# ``lib/diagnostics/provider.py::_LICENSE_RESTRICTED`` — keep all three in sync;
# drift between them is a documentation bug (mirrors the project's
# doc-mirrored-constant convention). Used below to pin the AUTO-resolved
# training seat to "local" when LLM_PROVIDER resolves to a restricted provider,
# so the corpus-generalization defaults never silently route training-pair
# synthesis through Anthropic / NVIDIA-hosted Llama-3.3.
_LICENSE_RESTRICTED_SYNTHESIS = frozenset({"anthropic", "nvidia"})

# Master opt-out for the A5 corpus-generalization defaults-on path. When
# truthy, ``_apply_corpus_generalization_defaults`` returns early and sets
# NOTHING — neither the measured graph-shaping flags nor the
# licensing-sensitive synthesis-provider envs. This exists for deterministic
# fixture-contract runs (e.g. ``tests/integration/test_pipeline_end_to_end.py``)
# that take the ``LOCAL_DISPATCHER_ALLOW_STUB`` stub authoring route: the
# three-stage textbook synthesis the A5 set turns ON dispatches REAL local-LLM
# (Ollama) calls during ``objective_extraction`` / ``course_planning`` /
# ``concept_extraction``, which are nondeterministic and CI-infeasible. There
# is no "off" value for ``TEXTBOOK_SYNTHESIS_PROVIDER`` (any non-empty value
# fires the synthesis guard), so a single master switch is the clean knob.
# This does NOT change the product default (unset → full A5 auto-on); it is a
# test/deterministic-run companion to the stub opt-in, mirroring
# ``LOCAL_DISPATCHER_ALLOW_STUB`` semantics.
_DISABLE_CORPUS_GENERALIZATION_ENV = "ED4ALL_DISABLE_CORPUS_GENERALIZATION"

# Workflows that get the corpus-generalization auto-on treatment: the full
# textbook pipeline and its Courseforge stage aliases (which run through the
# same ``textbook_to_course`` config) plus ``course_generation``.
_CORPUS_GENERALIZATION_WORKFLOWS: frozenset = frozenset(
    {"textbook_to_course", "course_generation"}
)


def courseforge_exports_dir() -> Path:
    """Resolve the Courseforge exports dir for this module.

    Honors the ED4ALL_HOME relocatable data root (via ``lib.paths``) when set;
    otherwise resolves against this module's ``PROJECT_ROOT`` so a test that
    monkeypatches ``workflow_runner.PROJECT_ROOT`` (the long-standing seam for
    redirecting project exports into a ``tmp_path``) still works. Byte-stable to
    ``PROJECT_ROOT / "Courseforge" / "exports"`` when ED4ALL_HOME is unset.
    """
    try:
        from lib.paths import courseforge_exports_dir as _lib_resolver
        from lib.paths import ed4all_home

        if ed4all_home() is not None:
            return _lib_resolver()
    except Exception:  # noqa: BLE001 — defensive: fall back to in-tree default
        pass
    return PROJECT_ROOT / "Courseforge" / "exports"


WORKFLOWS_YAML_PATH = PROJECT_ROOT / "config" / "workflows.yaml"
WORKFLOWS_META_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "config" / "workflows_meta.schema.json"


# =============================================================================
# INTER-PHASE DATA ROUTING
# =============================================================================
# Defines how outputs from one phase become inputs to the next.
# Format: {phase_name: {param_name: (source_type, *source_path)}}
#   - ("workflow_params", key) => from workflow creation params
#   - ("phase_outputs", phase_name, key) => from a prior phase's extracted outputs
#   - ("literal", value) => hardcoded value
#
# REC-CTR-05 (Wave 6): Routing is now primarily defined in config/workflows.yaml
# via per-phase `inputs_from:` and `outputs:` blocks. The legacy dicts below
# act as backwards-compat fallbacks for phases whose YAML entries have not yet
# been annotated. `_load_workflows_config()` validates the YAML against
# schemas/config/workflows_meta.schema.json at module load time, so typos in
# gate IDs, phase names, severities, or inter-phase references are caught
# pre-flight.
# =============================================================================

_LEGACY_PHASE_PARAM_ROUTING: Dict[str, Dict[str, Tuple]] = {
    "dart_conversion": {
        # Task creation handled specially in _create_phase_tasks (one task per PDF)
        "course_code": ("workflow_params", "course_name"),
    },
    "staging": {
        "run_id": ("workflow_params", "run_id"),
        "dart_html_paths": ("phase_outputs", "dart_conversion", "output_paths"),
        "course_name": ("workflow_params", "course_name"),
    },
    "objective_extraction": {
        "course_name": ("workflow_params", "course_name"),
        "objectives_path": ("workflow_params", "objectives_path"),
        "duration_weeks": ("workflow_params", "duration_weeks"),
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
        # Wave 24: textbook-ingestor needs staging_dir so
        # extract_textbook_structure can walk the staged DART HTML.
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
    },
    "source_mapping": {
        # Wave 9: DART source-block -> Courseforge page routing.
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "textbook_structure_path": (
            "phase_outputs", "objective_extraction", "textbook_structure_path",
        ),
    },
    # Phase 7b ST 11: chunking phase — DART chunkset emit. Mirrors the
    # YAML routing at config/workflows.yaml::chunking. Phase 8 ST 3
    # adds the optional ``libv2_root`` workflow param so ops topologies
    # that mount LibV2 at a non-default location can override the
    # in-tree default via ``--libv2-root`` / ``ED4ALL_LIBV2_ROOT``.
    "chunking": {
        "course_name": ("workflow_params", "course_name"),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "libv2_root": ("workflow_params", "libv2_root"),
    },
    # Phase 6 ST 11: concept_extraction phase — pedagogy-graph builder.
    # Mirrors the YAML routing at config/workflows.yaml::concept_extraction.
    # Phase 7b ST 14.5 added the upstream dart_chunks_path consumption;
    # Phase 8 ST 3 adds the optional ``libv2_root`` workflow param.
    "concept_extraction": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "course_name": ("workflow_params", "course_name"),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "dart_chunks_path": (
            "phase_outputs", "chunking", "dart_chunks_path",
        ),
        # Three-stage textbook synthesis — Stage 3 (Wave C). Routes the
        # objective_extraction phase's textbook_structure.json so
        # _run_concept_extraction can read chapters[].chapter_text for
        # the per-chapter concept-synthesis calls when
        # TEXTBOOK_SYNTHESIS_PROVIDER is set. Mirrors the YAML routing at
        # config/workflows.yaml::concept_extraction; consulted as a
        # fallback when YAML lookup misses.
        "textbook_structure_path": (
            "phase_outputs", "objective_extraction", "textbook_structure_path",
        ),
        "libv2_root": ("workflow_params", "libv2_root"),
        # Objectives resolution candidate #1: the --reuse-objectives JSON
        # so a reuse run pins the LO-dependent typed-edge rules
        # (prerequisite_from_lo_order, targets_concept_from_lo) to the
        # operator's verbatim objectives doc. Mirrors the YAML routing at
        # config/workflows.yaml::concept_extraction.
        "objectives_path": ("workflow_params", "reuse_objectives_path"),
        # Objectives resolution candidate #2 (fresh-run path): the
        # synthesized_objectives.json emitted by course_planning. The
        # phase-ordering fix (Option A1) moved concept_extraction to run
        # AFTER course_planning, so fresh runs without --reuse-objectives
        # now get a real learning_outcomes ordering here (the
        # LO-dependent typed-edge rules fire and the
        # concept_objective_linker can populate keyConcepts[]). Mirrors
        # the YAML routing at config/workflows.yaml::concept_extraction.
        "synthesized_objectives_path": (
            "phase_outputs", "course_planning", "synthesized_objectives_path",
        ),
    },
    "course_planning": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "course_name": ("workflow_params", "course_name"),
        "objectives_path": ("workflow_params", "objectives_path"),
        "duration_weeks": ("workflow_params", "duration_weeks"),
        # Wave 40: route duration_weeks_explicit so _plan_course_structure's
        # config-over-kwargs precedence check activates on real runs.
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
        "source_module_map_path": (
            "phase_outputs", "source_mapping", "source_module_map_path",
        ),
        # Phase-ordering fix (Option A1): the concept_graph_path route was
        # deleted here. course_planning now runs BEFORE concept_extraction,
        # so the concept graph does not exist yet at planning time. The
        # concept_objective_linker pass that populated keyConcepts[] moved
        # into _run_concept_extraction (which now runs after planning).
    },
    "content_generation": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "source_module_map_path": (
            "phase_outputs", "source_mapping", "source_module_map_path",
        ),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        # Wave 40: same rationale as course_planning —
        # _generate_course_content's precedence check needs the flag.
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
    },
    # Phase 3 Subtask 5: input routing for the two-pass router phases.
    # Mirrors the legacy ``content_generation`` routing for the outline
    # tier; the rewrite tier additionally consumes
    # ``blocks_validated_path`` from the inter-tier validation phase.
    "content_generation_outline": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "source_module_map_path": (
            "phase_outputs", "source_mapping", "source_module_map_path",
        ),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
        # Worker W2 (validation-wiring fix): thread workflow_type so
        # ``_run_content_generation_outline`` can resolve the
        # ``inter_tier_validation`` phase's validation_gates from the
        # YAML spec, instantiate them, and pass the resolved validator
        # list into ``router.route_with_self_consistency``. Without this
        # thread, the outline phase falls back to the empty-validators
        # path (preserving pre-fix behavior on legacy direct calls).
        "workflow_type": ("workflow_params", "workflow_type"),
        # ``--force`` must beat the crash-resume sidecar: the outline
        # handler clears + ignores ``.blocks_outline_checkpoint.jsonl``
        # when force_rerun is set (fresh blocks still checkpoint).
        "force_rerun": ("workflow_params", "force_rerun"),
    },
    "inter_tier_validation": {
        "blocks_outline_path": (
            "phase_outputs", "content_generation_outline",
            "blocks_outline_path",
        ),
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
    },
    "content_generation_rewrite": {
        "blocks_validated_path": (
            "phase_outputs", "inter_tier_validation",
            "blocks_validated_path",
        ),
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "source_module_map_path": (
            "phase_outputs", "source_mapping", "source_module_map_path",
        ),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
        # Worker W3 (validation-wiring fix): thread workflow_type so
        # ``_run_content_generation_rewrite`` can resolve the
        # ``post_rewrite_validation`` phase's validation_gates from the
        # YAML spec, instantiate them, and pass the resolved validator
        # list into ``router.route_rewrite_with_remediation``. Without
        # this thread the rewrite phase falls back to the empty-
        # validators path (preserves pre-fix behavior on legacy direct
        # calls).
        "workflow_type": ("workflow_params", "workflow_type"),
        # Worker W3: rehydrate the W2-persisted outline sidecars so the
        # rewrite phase can pass per-block ``source_chunks`` +
        # ``objectives`` into the remediation loop instead of the
        # legacy ``[]`` defaults that broke the inter-tier seam.
        "outline_chunks_path": (
            "phase_outputs", "content_generation_outline",
            "outline_chunks_path",
        ),
        "outline_objectives_path": (
            "phase_outputs", "content_generation_outline",
            "outline_objectives_path",
        ),
        # ``--force`` must beat the crash-resume sidecar: the rewrite
        # handler clears + ignores ``.blocks_final_checkpoint.jsonl``
        # when force_rerun is set (fresh rewrites still checkpoint; the
        # separate blocks_final.jsonl --blocks byte-identity cache is
        # untouched).
        "force_rerun": ("workflow_params", "force_rerun"),
        # Phase 5 ST 1 (--blocks) — thread the parsed block-TYPE filter so
        # ``_run_content_generation_rewrite`` additively evicts cached
        # blocks whose ``block_type`` is in the set (re-rolling them even
        # after a prior successful rewrite). Unset (default) → the handler
        # reads ``None`` → byte-identical failure-driven cache reuse.
        "target_block_ids": ("workflow_params", "target_block_ids"),
    },
    "packaging": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "blocks_final_path": (
            "phase_outputs",
            "content_generation_rewrite",
            "blocks_final_path",
        ),
    },
    # Phase 7c ST 16: imscc_chunking phase — IMSCC chunkset emit
    # post-packaging. Mirrors the YAML routing at
    # config/workflows.yaml::imscc_chunking. Phase 8 ST 3 adds the
    # optional ``libv2_root`` workflow param so ops topologies that
    # mount LibV2 at a non-default location can override the in-tree
    # default via ``--libv2-root`` / ``ED4ALL_LIBV2_ROOT``.
    "imscc_chunking": {
        "course_name": ("workflow_params", "course_name"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
        "libv2_root": ("workflow_params", "libv2_root"),
    },
    "trainforge_assessment": {
        "course_id": ("workflow_params", "course_name"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
        "bloom_levels": ("workflow_params", "bloom_levels"),
        "question_count": ("workflow_params", "assessment_count"),
        # Wave 24: real TO/CO objective_ids come from course_planning
        # (was objective_extraction with phantom {COURSE}_OBJ_N IDs).
        "objective_ids": ("phase_outputs", "course_planning", "objective_ids"),
        # Phase 8 ST 2: route the upstream IMSCC chunkset path
        # written by ``imscc_chunking`` (Phase 7c ST 16) so
        # ``_run_trainforge_assessment`` can pass it to
        # ``CourseProcessor`` and short-circuit the in-process
        # ``_chunk_content`` rebuild. Mirrors the equivalent YAML
        # routing at ``config/workflows.yaml::trainforge_assessment``;
        # the legacy dict is consulted as a fallback when YAML
        # lookup misses (see ``_get_phase_param_routing``).
        "imscc_chunks_path": (
            "phase_outputs", "imscc_chunking", "imscc_chunks_path",
        ),
    },
    "libv2_archival": {
        "course_name": ("workflow_params", "course_name"),
        "domain": ("workflow_params", "domain"),
        "division": ("workflow_params", "division"),
        "pdf_paths": ("workflow_params", "pdf_paths"),
        "html_paths": ("phase_outputs", "dart_conversion", "output_paths"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
        # Phase 6 ST 18 / Phase 7c.5 / Phase 8 ST 5: thread the three
        # chunkset SHA-256s (concept graph from ``concept_extraction``,
        # DART chunkset from ``chunking``, IMSCC chunkset from
        # ``imscc_chunking``) so the LibV2 manifest carries each hash
        # and the ``libv2_manifest`` gate can cross-check on-disk
        # artifacts. Mirrors the YAML routing at
        # config/workflows.yaml::libv2_archival; the legacy dict is
        # consulted as a fallback when YAML lookup misses (see
        # ``_get_phase_param_routing``).
        "concept_graph_sha256": (
            "phase_outputs", "concept_extraction", "concept_graph_sha256",
        ),
        "dart_chunks_sha256": (
            "phase_outputs", "chunking", "dart_chunks_sha256",
        ),
        "imscc_chunks_sha256": (
            "phase_outputs", "imscc_chunking", "imscc_chunks_sha256",
        ),
        # Issue I1: explicit chunkset PATHS pin the fresh on-disk chunkset
        # as the archive copy source (vs the mtime heuristic). Mirrors the
        # YAML routing at config/workflows.yaml::libv2_archival.
        "imscc_chunks_path": (
            "phase_outputs", "imscc_chunking", "imscc_chunks_path",
        ),
        "dart_chunks_path": (
            "phase_outputs", "chunking", "dart_chunks_path",
        ),
        # B3 license/attribution plumbing: mirrors the YAML routing at
        # config/workflows.yaml::libv2_archival so the parity contract holds.
        "license_note": ("workflow_params", "license_note"),
        "attribution": ("workflow_params", "attribution"),
        # Objectives plumbing: thread the course_planning-emitted
        # synthesized_objectives.json so _archive_to_libv2 can project the
        # canonical archive-side objectives.json (with parent_terminal
        # back-pointers) the strict packet_integrity gate's co_has_parent
        # rule depends on.
        "synthesized_objectives_path": (
            "phase_outputs", "course_planning", "synthesized_objectives_path",
        ),
    },
    "finalization": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "course_slug": ("phase_outputs", "libv2_archival", "course_slug"),
    },
}

# Maps phase names to the keys extracted from their task results.
# After a phase completes, these fields are pulled from the result
# and stored in workflow state under phase_outputs[phase_name].
_LEGACY_PHASE_OUTPUT_KEYS: Dict[str, List[str]] = {
    # Wave 32 Deliverable B: surface html_path + html_paths (router
    # canonical keys) alongside the legacy output_path / output_paths
    # aliases so the DartMarkersValidator gate builder picks them up
    # without a router change. Pre-Wave-32 runs reported
    # ``dart_markers skipped — missing inputs: html_path`` because
    # ``_build_dart_markers`` looked for html_path but the phase only
    # surfaced output_path.
    "dart_conversion": [
        "output_path", "output_paths",
        "html_path", "html_paths",
        "success", "html_length",
    ],
    "staging": ["staging_dir", "staged_files", "file_count"],
    # Wave 24: objective_extraction no longer emits objective_ids; it
    # now emits textbook_structure_path + chapter_count + source_file_count
    # + duration_weeks (autoscaled when --weeks unset).
    # Real objective_ids surface from course_planning's synthesize step.
    "objective_extraction": [
        "project_id", "project_path", "textbook_structure_path",
        "chapter_count", "duration_weeks", "source_file_count",
    ],
    "source_mapping": ["source_module_map_path", "source_chunk_ids"],
    "course_planning": [
        "project_id", "synthesized_objectives_path",
        "objective_ids", "terminal_count", "chapter_count",
        # Stage-2 grounding + citation-reselect counters (dict; {} when
        # Stage-2 window synthesis did not run). Mirrors the YAML
        # course_planning outputs block so a post-run audit reads
        # phase_outputs.course_planning.grounding_signals directly.
        "grounding_signals",
    ],
    # Wave 32 Deliverable B: add page_paths + content_dir so the
    # ContentGroundingValidator + PageObjectivesValidator builders
    # can resolve inputs (pre-Wave-32 both gates silently skipped).
    "content_generation": [
        "project_id", "content_paths", "page_paths", "content_dir",
        "weeks_prepared",
        # Anti-silent-template guard provenance (ContentAuthorshipValidator).
        "generator_mode", "template_fallback_fired",
        "content_generation_provenance_path",
    ],
    # Phase 3 Subtask 5: two-pass router phase output declarations.
    # The outline tier emits a Block-list JSON sidecar (no HTML body);
    # the validation tier filters into pass/fail Block lists; the
    # rewrite tier emits the final HTML pages plus a final Block JSON
    # for downstream consumers (Trainforge ingest reads from the
    # rewrite-tier blocks_final_path when COURSEFORGE_TWO_PASS=true).
    "content_generation_outline": [
        "blocks_outline_path", "project_id", "weeks_prepared",
        # Worker W2 (validation-wiring fix): sidecars persisted next to
        # blocks_outline.jsonl so the rewrite phase can rehydrate the
        # chunks_lookup + objectives_payload that the outline tier
        # built (without re-walking staging / synthesized_objectives).
        "outline_chunks_path", "outline_objectives_path",
    ],
    "inter_tier_validation": [
        "blocks_validated_path", "blocks_failed_path",
    ],
    "content_generation_rewrite": [
        "content_paths", "page_paths", "content_dir",
        "blocks_final_path",
    ],
    # Phase 3.5 Subtask 12: post-rewrite validation phase output keys.
    # Mirrors inter_tier_validation's shape — emits
    # ``blocks_validated_path`` (rewrite-tier blocks that passed every
    # gate) and ``blocks_failed_path`` (rewrite-tier blocks that
    # tripped at least one gate). Packaging consumes blocks_validated_path
    # via the post_rewrite_validation -> packaging dependency chain
    # introduced in Subtask 10 + 11.
    "post_rewrite_validation": [
        "blocks_validated_path", "blocks_failed_path",
    ],
    # Wave 32 Deliverable B: surface imscc_path + content_dir so
    # IMSCCValidator + PageObjectivesValidator builders pick them up.
    "packaging": [
        "package_path", "libv2_package_path", "imscc_path",
        "content_dir", "project_id",
    ],
    # Wave 24: surface chunks_path + assessments_path for the
    # assessment_objective_alignment gate input builder.
    "trainforge_assessment": [
        "output_path", "assessments_path", "assessment_id",
        "question_count", "chunks_path",
    ],
    "libv2_archival": ["course_slug", "course_dir", "manifest_path"],
    "finalization": ["project_id", "package_path", "course_slug"],
}


# Backwards-compat: expose the legacy aliases. Callers outside this module
# historically imported these names directly. New code should call
# _get_phase_param_routing() / _get_phase_output_keys() or the YAML-first
# accessors, which respect per-phase YAML overrides.
PHASE_PARAM_ROUTING = _LEGACY_PHASE_PARAM_ROUTING
PHASE_OUTPUT_KEYS = _LEGACY_PHASE_OUTPUT_KEYS


# =============================================================================
# YAML-BASED PHASE ROUTING LOADER (REC-CTR-05)
# =============================================================================

# Module-level cache for loaded + validated workflows.yaml. Populated lazily
# by _load_workflows_config(). Reset for tests via _reset_workflows_cache().
_WORKFLOWS_CONFIG_CACHE: Optional[Dict[str, Any]] = None

# Track phases we've already warn-logged for fall-through to legacy defaults,
# to avoid log spam when the same phase fires repeatedly across a workflow.
_FALLBACK_LOGGED: set = set()


def _summarize_gate_failure(gate_results: Any) -> str:
    """Build a one-line human reason from a phase's failed validation gates.

    ``gate_results`` is the list of per-gate results (``GateResult`` dataclasses
    or already-dictified) the executor returns for a phase. We pick the failed
    gates (``passed`` is falsy), prefer ``critical`` over ``warning``, and render
    ``<gate_id> (<first issue>)`` joined for up to two gates so the GUI failure
    panel has a readable summary without re-walking the structured gate list.
    Returns a generic message when no structured detail is available.
    """
    failed: List[Tuple[str, str, str]] = []  # (severity, gate_id, first_issue)
    for gr in gate_results or []:
        if hasattr(gr, "gate_id"):
            passed = getattr(gr, "passed", True)
            gate_id = getattr(gr, "gate_id", "") or ""
            severity = getattr(gr, "severity", "warning") or "warning"
            issues = getattr(gr, "issues", None) or []
        elif isinstance(gr, dict):
            passed = gr.get("passed", True)
            gate_id = gr.get("gate_id", "") or ""
            severity = gr.get("severity", "warning") or "warning"
            issues = gr.get("issues") or []
        else:
            continue
        if passed:
            continue
        first_issue = str(issues[0]) if issues else ""
        failed.append((severity, gate_id, first_issue))
    if not failed:
        return "failed validation gates"
    # critical first, then warning; stable within severity.
    failed.sort(key=lambda t: 0 if t[0] == "critical" else 1)
    parts = []
    for _sev, gate_id, issue in failed[:2]:
        parts.append(f"{gate_id} ({issue})" if issue else gate_id)
    suffix = f" (+{len(failed) - 2} more)" if len(failed) > 2 else ""
    return "failed validation gate(s): " + ", ".join(parts) + suffix


def _reset_workflows_cache() -> None:
    """Clear the cached workflows config and fallback-log tracker.

    Primarily used by tests to force a reload after modifying the underlying
    YAML or schema on disk.
    """
    global _WORKFLOWS_CONFIG_CACHE
    _WORKFLOWS_CONFIG_CACHE = None
    _FALLBACK_LOGGED.clear()


def _load_workflows_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load and validate config/workflows.yaml against the meta-schema.

    Validates against schemas/config/workflows_meta.schema.json plus a
    cross-reference integrity check: any `inputs_from` entry with
    source=phase_outputs must reference a prior-phase output declared in
    that phase's `outputs:` list.

    Raises:
        ValueError: If workflows.yaml is missing, malformed, or fails
            meta-schema/cross-ref validation.

    Returns:
        The raw parsed YAML dict (already validated).
    """
    global _WORKFLOWS_CONFIG_CACHE
    if _WORKFLOWS_CONFIG_CACHE is not None and not force_reload:
        return _WORKFLOWS_CONFIG_CACHE

    if not WORKFLOWS_YAML_PATH.exists():
        raise ValueError(
            f"Workflows config not found: {WORKFLOWS_YAML_PATH}. "
            "workflow_runner requires config/workflows.yaml to load phase routing."
        )

    try:
        with open(WORKFLOWS_YAML_PATH) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {WORKFLOWS_YAML_PATH}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"workflows.yaml must be a mapping at the top level, got {type(data).__name__}"
        )

    # Meta-schema validation (REC-CTR-05). If jsonschema is not installed or
    # the schema file is missing, log a warning and skip — don't block
    # execution purely on meta-schema tooling availability.
    if WORKFLOWS_META_SCHEMA_PATH.exists():
        try:
            import jsonschema
            with open(WORKFLOWS_META_SCHEMA_PATH) as f:
                meta_schema = json.load(f)
            try:
                jsonschema.validate(data, meta_schema)
            except jsonschema.ValidationError as e:
                path = ".".join(str(p) for p in e.absolute_path)
                raise ValueError(
                    f"config/workflows.yaml failed meta-schema validation at '{path}': "
                    f"{e.message}"
                ) from e
        except ImportError:
            logger.warning(
                "jsonschema not installed; skipping workflows.yaml meta-schema validation. "
                "Install jsonschema to catch config typos pre-flight."
            )
    else:
        logger.warning(
            "Meta-schema not found at %s; skipping structural validation of workflows.yaml.",
            WORKFLOWS_META_SCHEMA_PATH,
        )

    # Cross-reference integrity: every phase_outputs input must resolve
    # to a prior phase's declared outputs.
    _validate_inputs_from_references(data)

    _WORKFLOWS_CONFIG_CACHE = data
    return data


def _validate_inputs_from_references(workflows_data: Dict[str, Any]) -> None:
    """Ensure `inputs_from: {source: phase_outputs,...}` references resolve.

    For each workflow, iterates phases in declared order and checks that any
    phase_outputs-sourced input refers to (phase, output) that was declared
    in a prior phase's `outputs:` list. Phases without an explicit `outputs:`
    block are treated as exposing the legacy output keys for that phase,
    preserving backwards compatibility.

    Raises:
        ValueError: On the first unresolved reference, with a clear message.
    """
    for wf_name, wf in (workflows_data.get("workflows") or {}).items():
        if not isinstance(wf, dict):
            continue
        seen_outputs: Dict[str, set] = {}
        for phase in wf.get("phases", []) or []:
            if not isinstance(phase, dict):
                continue
            phase_name = phase.get("name", "<unnamed>")
            for route in phase.get("inputs_from") or []:
                if not isinstance(route, dict):
                    continue
                if route.get("source") != "phase_outputs":
                    continue
                ref_phase = route.get("phase")
                ref_output = route.get("output")
                if ref_phase not in seen_outputs:
                    raise ValueError(
                        f"Workflow '{wf_name}' phase '{phase_name}' inputs_from "
                        f"references unknown or not-yet-declared phase '{ref_phase}'."
                    )
                if ref_output not in seen_outputs[ref_phase]:
                    raise ValueError(
                        f"Workflow '{wf_name}' phase '{phase_name}' inputs_from "
                        f"references '{ref_phase}.{ref_output}' but '{ref_phase}' does "
                        f"not declare '{ref_output}' in its outputs. "
                        f"Declared outputs: {sorted(seen_outputs[ref_phase])}"
                    )
            # Record this phase's declared outputs, falling back to legacy
            # keys so legacy phases still satisfy downstream references.
            declared = phase.get("outputs")
            if declared is None:
                declared = _LEGACY_PHASE_OUTPUT_KEYS.get(phase_name, [])
            seen_outputs[phase_name] = set(declared or [])


def _phase_yaml_block(phase_name: str) -> Optional[Dict[str, Any]]:
    """Locate the first phase entry matching `phase_name` across all workflows.

    Phase names are used as dict keys in the legacy dicts, so callers only
    have a phase name (not workflow+phase). If the same phase name appears in
    multiple workflows (e.g. `dart_conversion` in sibling workflows),
    the first YAML block with an `inputs_from:` or `outputs:` annotation
    wins. This preserves the prior implicit behavior where a phase had a
    single global routing signature.
    """
    try:
        data = _load_workflows_config()
    except ValueError:
        # Propagate to caller at first use; logged there.
        raise

    fallback: Optional[Dict[str, Any]] = None
    for wf in (data.get("workflows") or {}).values():
        if not isinstance(wf, dict):
            continue
        for phase in wf.get("phases", []) or []:
            if not isinstance(phase, dict):
                continue
            if phase.get("name") == phase_name:
                if phase.get("inputs_from") or phase.get("outputs"):
                    return phase
                if fallback is None:
                    fallback = phase
    return fallback


def _get_phase_param_routing(phase_name: str) -> Dict[str, Tuple]:
    """Return {param: (source_type, *path)} routing for a phase.

    Preference order:
      1. YAML `inputs_from:` block for this phase (REC-CTR-05).
      2. Legacy in-memory `_LEGACY_PHASE_PARAM_ROUTING` entry (warn once).
      3. Empty dict.
    """
    try:
        block = _phase_yaml_block(phase_name)
    except ValueError as e:
        logger.error("Failed to load workflows.yaml for phase routing: %s", e)
        block = None

    if block and block.get("inputs_from"):
        routing: Dict[str, Tuple] = {}
        for route in block["inputs_from"]:
            if not isinstance(route, dict):
                continue
            param = route.get("param")
            source = route.get("source")
            if not param or not source:
                continue
            if source == "workflow_params":
                routing[param] = ("workflow_params", route.get("key"))
            elif source == "phase_outputs":
                routing[param] = (
                    "phase_outputs",
                    route.get("phase"),
                    route.get("output"),
                )
            elif source == "literal":
                routing[param] = ("literal", route.get("value"))
        return routing

    # Fallback to legacy in-memory dict
    if phase_name in _LEGACY_PHASE_PARAM_ROUTING:
        if phase_name not in _FALLBACK_LOGGED:
            logger.warning(
                "Phase '%s' has no `inputs_from:` block in config/workflows.yaml; "
                "falling back to legacy in-memory routing. Annotate the phase to "
                "silence this warning.",
                phase_name,
            )
            _FALLBACK_LOGGED.add(phase_name)
        return _LEGACY_PHASE_PARAM_ROUTING[phase_name]

    return {}


def _get_phase_output_keys(phase_name: str) -> List[str]:
    """Return the list of output keys to extract from a phase's task results.

    Preference order:
      1. YAML `outputs:` block for this phase.
      2. Legacy in-memory `_LEGACY_PHASE_OUTPUT_KEYS` entry (warn once).
      3. Empty list.
    """
    try:
        block = _phase_yaml_block(phase_name)
    except ValueError as e:
        logger.error("Failed to load workflows.yaml for phase outputs: %s", e)
        block = None

    if block and block.get("outputs"):
        return list(block["outputs"])

    if phase_name in _LEGACY_PHASE_OUTPUT_KEYS:
        key = f"outputs:{phase_name}"
        if key not in _FALLBACK_LOGGED:
            logger.warning(
                "Phase '%s' has no `outputs:` block in config/workflows.yaml; "
                "falling back to legacy in-memory output keys.",
                phase_name,
            )
            _FALLBACK_LOGGED.add(key)
        return list(_LEGACY_PHASE_OUTPUT_KEYS[phase_name])

    return []


# Eager load + validate workflows.yaml at module import so typos surface
# before any workflow attempts to run. Tests that want a pristine config
# should call _reset_workflows_cache() after patching.
try:
    _load_workflows_config()
except ValueError as _e:
    # Log and re-raise so downstream imports see the error immediately.
    logger.error("workflows.yaml failed pre-flight validation: %s", _e)
    raise


# =============================================================================
# Wave 80 Worker A: --reuse-objectives helpers
# =============================================================================


def _normalize_to_courseforge_form(
    data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Normalize an objectives JSON into Courseforge synthesized form.

    Wave 80 Worker A. Accepts:

      * Courseforge synthesized form: ``terminal_objectives[]`` +
        ``chapter_objectives[]``. ``chapter_objectives`` may be a flat
        list of objective dicts OR the canonical
        ``[{"chapter": ..., "objectives": [...]}]`` group shape.
      * Wave 75 LibV2 archive form: ``terminal_outcomes[]`` +
        ``component_objectives[]`` (flat list with optional
        ``parent_terminal`` back-pointer).

    Returns a dict carrying:
      * ``terminal_objectives`` — list of terminal LO dicts (Courseforge
        shape: ``id``, ``statement``, etc.).
      * ``chapter_objectives`` — list of ``{"chapter": str,
        "objectives": [...]}`` groups (Courseforge shape).
      * ``course_name`` (best-effort, may be missing) and
        ``duration_weeks`` (best-effort, may be missing).

    Returns ``None`` when neither shape is present.
    """
    has_courseforge = (
        isinstance(data.get("terminal_objectives"), list)
        or isinstance(data.get("chapter_objectives"), list)
    )
    has_libv2 = (
        isinstance(data.get("terminal_outcomes"), list)
        or isinstance(data.get("component_objectives"), list)
    )
    if not (has_courseforge or has_libv2):
        return None

    if has_courseforge and not has_libv2:
        # Already in target form. Ensure chapter_objectives is in the
        # group shape ([{chapter, objectives}], not a flat list).
        terminal = list(data.get("terminal_objectives") or [])
        chapter_raw = list(data.get("chapter_objectives") or [])
        chapter_groups = _coerce_chapter_groups(chapter_raw)
        return {
            "terminal_objectives": terminal,
            "chapter_objectives": chapter_groups,
            "course_name": data.get("course_name"),
            "duration_weeks": data.get("duration_weeks"),
        }

    # LibV2 archive form (or mixed — we prefer libv2 keys when both
    # present, since the user explicitly handed us the archive shape).
    terminal_raw = list(data.get("terminal_outcomes") or [])
    components_raw = list(data.get("component_objectives") or [])

    # Map to Courseforge shape. LibV2 IDs are lowercase by default; we
    # preserve them verbatim — the LO ID regex accepts both cases.
    terminal_objectives: List[Dict[str, Any]] = []
    for to in terminal_raw:
        if not isinstance(to, dict) or "id" not in to:
            continue
        entry: Dict[str, Any] = {"id": to["id"]}
        for key in (
            "statement", "bloom_level", "bloom_verb",
            "cognitive_domain", "weeks",
        ):
            if to.get(key) is not None:
                entry[key] = to[key]
        terminal_objectives.append(entry)

    # Group component objectives by parent_terminal -> a chapter group
    # (one group per terminal). LibV2 stores the parent reverse-link as
    # ``parent_terminal``; Courseforge's content-generator only needs
    # the flat per-week shape, so we emit one group per CO with the
    # parent's id as the chapter label fallback.
    chapter_groups: List[Dict[str, Any]] = []
    for co in components_raw:
        if not isinstance(co, dict) or "id" not in co:
            continue
        obj: Dict[str, Any] = {"id": co["id"]}
        for key in (
            "statement", "bloom_level", "bloom_verb",
            "cognitive_domain", "week", "source_refs",
        ):
            if co.get(key) is not None:
                obj[key] = co[key]
        # Preserve the parent_terminal back-pointer so downstream
        # consumers (and our cross-validation below) can verify the
        # hierarchy.
        if co.get("parent_terminal"):
            obj["parent_terminal"] = co["parent_terminal"]
        # Emit as a per-CO group. Use ``Week N`` style label by index
        # to match _plan_course_structure's convention.
        chapter_groups.append({
            "chapter": f"Week {len(chapter_groups) + 1}",
            "objectives": [obj],
        })

    return {
        "terminal_objectives": terminal_objectives,
        "chapter_objectives": chapter_groups,
        "course_name": data.get("course_code") or data.get("course_name"),
        "duration_weeks": data.get("duration_weeks"),
    }


def _coerce_chapter_groups(
    chapter_raw: List[Any],
) -> List[Dict[str, Any]]:
    """Coerce a chapter_objectives list to the canonical group shape.

    Accepts the dual shapes already supported by
    ``_content_gen_helpers.load_objectives_json``:

      * Group shape: ``[{"chapter": str, "objectives": [...]}, ...]``.
      * Flat shape: ``[{"id": "co-01", ...}, ...]``.

    Always returns the group shape, one group per CO when the input
    was flat (so the hierarchy is preserved 1:1 without forcing a
    chapter assignment).
    """
    groups: List[Dict[str, Any]] = []
    flat_buffer: List[Dict[str, Any]] = []
    for entry in chapter_raw:
        if not isinstance(entry, dict):
            continue
        if "objectives" in entry and isinstance(entry["objectives"], list):
            groups.append({
                "chapter": entry.get("chapter") or f"Week {len(groups) + 1}",
                "objectives": list(entry["objectives"]),
            })
        else:
            flat_buffer.append(entry)
    # If we accumulated flat-shape entries (or the input had nothing
    # but flat entries), emit one group per CO so the hierarchy stays
    # 1:1 with the input.
    for flat in flat_buffer:
        groups.append({
            "chapter": f"Week {len(groups) + 1}",
            "objectives": [flat],
        })
    return groups


def _validate_reused_lo_coherence(
    terminal: List[Dict[str, Any]],
    chapter_flat: List[Dict[str, Any]],
) -> Optional[str]:
    """Return None on success; an error string on failure.

    Wave 80 Worker A. Cross-validates a reused objectives file:

    * Every LO ID matches ``^[a-zA-Z]{2,}-\\d{2,}$`` (mirrors
      ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` and
      ``lib/ontology/learning_objectives.py::validate_lo_id``).
    * No duplicate IDs (across terminal + chapter combined).
    * Every CO ``parent_terminal`` (or ``parent_to``) reference, when
      present, resolves to an existing TO ID.
    """
    seen_ids: set = set()
    terminal_ids: set = set()
    for to in terminal:
        to_id = (to or {}).get("id")
        if not to_id:
            return "terminal entry missing 'id' field"
        if not _LO_ID_RE.match(str(to_id)):
            return (
                f"terminal id {to_id!r} does not match LO id regex "
                f"^[a-zA-Z]{{2,}}-\\d{{2,}}$"
            )
        if to_id in seen_ids:
            return f"duplicate LO id {to_id!r}"
        seen_ids.add(to_id)
        terminal_ids.add(to_id)
    for co in chapter_flat:
        co_id = (co or {}).get("id")
        if not co_id:
            return "chapter/component entry missing 'id' field"
        if not _LO_ID_RE.match(str(co_id)):
            return (
                f"chapter id {co_id!r} does not match LO id regex "
                f"^[a-zA-Z]{{2,}}-\\d{{2,}}$"
            )
        if co_id in seen_ids:
            return f"duplicate LO id {co_id!r}"
        seen_ids.add(co_id)
        # Hierarchy back-pointer (when present) must resolve.
        parent = co.get("parent_terminal") or co.get("parent_to")
        if parent and parent not in terminal_ids:
            return (
                f"chapter {co_id!r} parent_terminal={parent!r} "
                f"does not reference a known TO id "
                f"(known: {sorted(terminal_ids)})"
            )
    return None


def _warn_on_source_map_mismatch(
    source_map_path: str,
    terminal: List[Dict[str, Any]],
    chapter_flat: List[Dict[str, Any]],
) -> None:
    """Best-effort warning when the reused LOs miss ids referenced in
    the source_module_map. Pure logging — never raises.

    This catches the case where a user supplies an objectives file
    that's been heavily edited (e.g. removed half the COs) while the
    upstream source_module_map still references the original IDs.
    Downstream content_generation will then emit pages referencing
    objective ids that don't resolve.
    """
    try:
        path = Path(source_map_path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return

    # Collect all LO ids referenced by the source map (best-effort —
    # the schema is a router output we don't want to over-couple to).
    referenced: set = set()
    if isinstance(data, dict):
        for entries in data.values():
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict):
                        for key in ("objective_ids", "lo_ids", "objectives"):
                            val = e.get(key)
                            if isinstance(val, list):
                                for v in val:
                                    if isinstance(v, str) and _LO_ID_RE.match(v):
                                        referenced.add(v)
                            elif isinstance(val, str) and _LO_ID_RE.match(val):
                                referenced.add(val)

    if not referenced:
        return

    available = {str(t.get("id")) for t in terminal if t.get("id")}
    available |= {str(c.get("id")) for c in chapter_flat if c.get("id")}
    # Case-insensitive comparison since the LO id regex allows mixed.
    available_lc = {x.lower() for x in available}
    missing = [
        rid for rid in referenced
        if rid.lower() not in available_lc
    ]
    if missing:
        logger.warning(
            "reuse_objectives: source_module_map references %d LO id(s) "
            "absent from the supplied objectives file. content_generation "
            "may emit pages with unresolved objective references. "
            "missing=%s",
            len(missing),
            sorted(missing)[:10],
        )


class WorkflowRunner:
    """
    Executes a multi-phase workflow end-to-end with inter-phase data routing.

    Bridges the gap between the workflow YAML definitions and the
    TaskExecutor's phase-level execution. Handles:
    - Phase dependency ordering (topological sort via depends_on)
    - Task creation for each phase from config + routed params
    - Inter-phase output-to-input data routing
    - Optional phase skipping
    - Workflow state persistence for crash recovery
    """

    def __init__(self, executor: TaskExecutor, config: OrchestratorConfig):
        self.executor = executor
        self.config = config

    async def _vram_doctor_snapshot(
        self,
        phase_name: str,
        when: str,
        event: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one best-effort VRAM-trajectory row for a phase boundary.

        A no-op unless ``ED4ALL_VRAM_DOCTOR`` is on (``vram_doctor_enabled``):
        the default-off path returns IMMEDIATELY — it never probes VRAM (no
        NVML read, no ollama HTTP call) AND spawns NO worker thread, so a
        default run is byte-identical and zero-overhead.

        When enabled, the actual snapshot+write — ``snapshot_vram`` does a
        BLOCKING ollama ``/api/ps`` HTTP round-trip + NVML probes — is offloaded
        to a worker thread via ``asyncio.to_thread`` so a slow/unreachable
        ollama (or a slow NVML probe) can NEVER block the async run loop at a
        phase boundary (this hook runs inline twice per phase). It snapshots
        free/total VRAM + the resident ollama models and appends a JSON line to
        ``state/runs/<run_id>/vram_trajectory.jsonl`` (the SAME run dir the
        executor writes its phase checkpoints into — ``run_id`` is the
        executor's ``run_id``, which resolves ``get_state_runs_dir() / run_id``
        identically to the checkpoint manager's ``run_path``). The whole hook is
        wrapped best-effort so a doctor failure (incl. a ``to_thread`` error)
        can NEVER perturb the run's control flow or ``final_status``.
        """
        try:
            if not vram_doctor_enabled():
                return
            run_id = getattr(self.executor, "run_id", None) or "unknown"
            await asyncio.to_thread(
                self._vram_doctor_snapshot_blocking,
                run_id,
                phase_name,
                when,
                event,
                extra,
            )
        except Exception as exc:  # noqa: BLE001 — observability must never crash the run
            logger.debug(
                "vram_doctor: per-phase trajectory hook failed for phase %r "
                "(%s); ignoring: %s",
                phase_name, when, exc,
            )

    @staticmethod
    def _vram_doctor_snapshot_blocking(
        run_id: str,
        phase_name: str,
        when: str,
        event: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Synchronous snapshot+write body, run off the event loop.

        Holds the blocking ``snapshot_vram`` (ollama HTTP + NVML) +
        ``write_trajectory_row`` (disk append). Invoked only via
        ``asyncio.to_thread`` from the enabled path of
        ``_vram_doctor_snapshot``; the caller wraps it best-effort, so this body
        does not need its own guard.
        """
        snapshot = snapshot_vram()
        write_trajectory_row(
            run_id,
            phase_name,
            when,
            snapshot,
            event=event,
            extra=extra,
        )

    async def _gpu_lifecycle_sweep(self, phase_name: str) -> None:
        """Deterministic phase-boundary GPU lease hand-off (best-effort).

        The lease side of the "VRAM doctor" story: after a phase completes
        SUCCESSFULLY (post task results + post gates + post checkpoint persist),
        release the resident local ollama generation model(s) + the torch
        allocator cache so the NEXT phase's model gets a clean card. This is the
        deterministic replacement for the shared-8GB contention heuristics
        (silent CUDA-OOM deaths, council+reviewer coexistence): every GPU model
        loads, does its job, and hands the card over at the boundary.

        A no-op unless ``ED4ALL_GPU_LIFECYCLE`` is on (default ON — the owner
        directive wants lease semantics AS the behavior; residency/timing only,
        never an output byte). When off, this returns IMMEDIATELY — no ollama
        HTTP call, no worker thread — so control flow is byte-identical.

        When on, the blocking sweep (ollama ``/api/ps`` + ``keep_alive:0``
        round-trip; a slow/unreachable ollama must not stall the async run loop
        at a phase boundary) is offloaded via ``asyncio.to_thread`` — the SAME
        proven pattern as ``_vram_doctor_snapshot``. The whole hook is wrapped
        best-effort so a sweep failure can NEVER perturb ``final_status`` or the
        phase results.

        Ordering: the caller runs the doctor ``"after"`` snapshot FIRST (records
        true end-of-phase residency) and this sweep AFTER, so a trajectory shows
        end-of-phase residency THEN the lease hand-off. Fires ONLY at a phase
        boundary for a phase that actually RAN + succeeded this session (never
        between tasks within a phase; never after a FAILED-phase break — those
        break out of the loop before this call; never on a resume-skipped
        phase, which ``continue``s earlier).
        """
        try:
            from lib.gpu_lifecycle import resolve_gpu_lifecycle_mode

            if not resolve_gpu_lifecycle_mode():
                return
            run_id = getattr(self.executor, "run_id", None) or "unknown"
            await asyncio.to_thread(self._gpu_lifecycle_sweep_blocking, run_id, phase_name)
        except Exception as exc:  # noqa: BLE001 — lease hand-off must never crash the run
            logger.debug(
                "gpu_lifecycle: phase-boundary sweep failed for phase %r "
                "(ignoring): %s",
                phase_name, exc,
            )

    @staticmethod
    def _gpu_lifecycle_sweep_blocking(run_id: str, phase_name: str) -> None:
        """Synchronous release body, run off the event loop.

        Releases the resident ollama model(s) + torch allocator cache, then —
        ONLY when ``ED4ALL_VRAM_DOCTOR`` is also on — appends a
        ``lifecycle_sweep`` event row (carrying the evicted model names) to the
        SAME ``state/runs/<run_id>/vram_trajectory.jsonl`` the doctor writes, so
        a trajectory shows the lease hand-offs inline with the residency
        snapshots. The trajectory write is best-effort; the release arms are
        themselves never-raising.
        """
        from lib.gpu_lifecycle import release_ollama_models, release_torch

        evicted = release_ollama_models(stage=f"phase:{phase_name}")
        release_torch(stage=f"phase:{phase_name}")

        try:
            if vram_doctor_enabled():
                snapshot = snapshot_vram()
                write_trajectory_row(
                    run_id,
                    phase_name,
                    "after",
                    snapshot,
                    event="lifecycle_sweep",
                    extra={"evicted_models": evicted},
                )
        except Exception as exc:  # noqa: BLE001 — observability must never crash the run
            logger.debug(
                "gpu_lifecycle: lifecycle_sweep trajectory row failed for "
                "phase %r (ignoring): %s",
                phase_name, exc,
            )

    async def run_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute all phases of a workflow in dependency order.

        Args:
            workflow_id: ID of the workflow to execute

        Returns:
            Dict with workflow_id, status, phase_results, and phase_outputs
        """
        # Load workflow state
        workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not workflow_path.exists():
            return {"error": f"Workflow not found: {workflow_id}"}

        with open(workflow_path) as f:
            workflow_state = json.load(f)

        workflow_type = workflow_state.get("type", "")
        workflow_params = workflow_state.get("params", {})
        if isinstance(workflow_params, str):
            workflow_params = json.loads(workflow_params)

        # Worker W2 (validation-wiring fix): thread workflow_type into
        # workflow_params so ``_route_params`` can route it through to
        # the ``content_generation_outline`` phase handler. The handler
        # uses it to resolve the YAML-declared ``inter_tier_validation``
        # gate chain into validator instances threaded into
        # ``router.route_with_self_consistency``.
        if workflow_type and "workflow_type" not in workflow_params:
            workflow_params["workflow_type"] = workflow_type

        # Load workflow config from YAML
        wf_config = self.config.get_workflow(workflow_type)
        if not wf_config:
            return {"error": f"Unknown workflow type: {workflow_type}"}

        # Marketable-v1 A5: turn the corpus-generalization feature set ON for
        # this run (textbook_to_course / course_generation) so a fresh
        # general-textbook run gets LO-refs + page-level concept tags +
        # dynamic CURIEs + three-stage synthesis by default — the features
        # the product is sold on. setdefault semantics keep an operator's
        # explicit legacy env value intact. Bare library calls that never
        # reach run_workflow keep the default-off legacy contract.
        self._apply_corpus_generalization_defaults(workflow_type)

        # W1 Gap C: fill the four blessed authoring-route provider envs
        # (COURSEFORGE_PROVIDER / COURSEPLANNER_PROVIDER /
        # TRAINFORGE_ASSESSMENT_PROVIDER / TRAINFORGE_SYNTHESIS_PROVIDER) so a
        # bare `ed4all run --provider local` is a true single switch and does
        # not hard-fail at `_enforce_authoring_provider_route`. Gap A (all-
        # local two-pass routing) is appended inside the same helper. The
        # CLI threads `--provider` into `workflow_params["provider"]`; absent
        # that, resolution falls to `LLM_PROVIDER` > `local`. setdefault
        # semantics keep any operator/GUI-pinned env intact.
        self._apply_authoring_route_env(
            workflow_type, str(workflow_params.get("provider", "") or "")
        )

        # Hosted-large build profile GAP-2 fix — ``--stop-after <phase>``. When
        # set, the phase loop halts cleanly AFTER the named phase completes, so a
        # ``--stop-after imscc_chunking`` slice stops BEFORE
        # trainforge_assessment / training_synthesis / libv2_archival /
        # finalization run. Default unset → no behaviour change (runs to
        # completion). The phase name is validated against this workflow's
        # phase list; an unknown name is a hard error (fail fast rather than
        # silently never halting).
        stop_after_phase = str(workflow_params.get("stop_after", "") or "").strip()
        if stop_after_phase:
            valid_phase_names = {p.name for p in (wf_config.phases or [])}
            if stop_after_phase not in valid_phase_names:
                return {
                    "error": (
                        f"--stop-after phase '{stop_after_phase}' is not a "
                        f"phase of workflow '{workflow_type}'. Valid phases: "
                        f"{sorted(valid_phase_names)}"
                    )
                }

        # Initialize phase outputs (may already exist from partial run)
        phase_outputs: Dict[str, Dict] = workflow_state.get("phase_outputs", {})

        # Wave 74 Session 3: honour --skip-dart by synthesising the
        # dart_conversion phase_output from an existing DART/output/
        # directory before the phase loop runs. Downstream phases
        # (staging, libv2_archival) then resolve their inputs_from
        # without dart_conversion actually executing.
        if workflow_params.get("skip_dart") and "dart_conversion" not in phase_outputs:
            synthesized = self._synthesize_dart_skip_output(workflow_params)
            if synthesized is not None:
                phase_outputs["dart_conversion"] = synthesized

        # Phase 5 Subtask 2: honour --outline / courseforge-* stage
        # subcommands by synthesising every upstream phase's
        # phase_output from the on-disk artifacts under the project
        # export + LibV2 course dir. The phase loop's _completed skip
        # check then short-circuits every upstream phase, so the
        # downstream target phase (typically content_generation_rewrite)
        # runs without re-dispatching the upstream chain.
        #
        # Resolution chain for the OUTLINE_DIR:
        #   1. Explicit ``outline_dir`` workflow param (Worker WA's
        #      forthcoming --outline CLI flag).
        #   2. ``courseforge_stage`` set => walk
        #      Courseforge/exports/PROJ-{COURSE_CODE}-* and pick the
        #      most-recently-modified project dir.
        #
        # Honours --force (Worker WA's ``force_rerun`` workflow param,
        # commit 96e1bde) by flipping _completed to False on every
        # synthesised entry so the phase loop re-runs them.
        outline_dir_resolved = self._resolve_outline_dir(workflow_params)
        if outline_dir_resolved is not None:
            # When a courseforge_stage whitelist is active, a two-pass
            # phase that is OUTSIDE the whitelist gets skipped by the loop
            # but a downstream WHITELISTED phase may still depend on its
            # on-disk output via ``inputs_from``. The clearest motivating
            # case: ``courseforge-validate`` activates
            # ``post_rewrite_validation`` (which needs
            # ``content_generation_rewrite.blocks_final_path``) but SKIPS
            # ``content_generation_rewrite`` itself — so its output must be
            # reconstructed from ``04_rewrite/blocks_final.jsonl`` on disk.
            # Pass the active whitelist down so the synthesizer reconstructs
            # the skipped-but-depended-on two-pass phases WITHOUT
            # reconstructing (and thus wrongly ``_completed``-skipping) the
            # phases that the stage actually intends to RE-RUN.
            stage_param = workflow_params.get("courseforge_stage")
            stage_active_phases = (
                self._resolve_courseforge_stage_active_phases(stage_param)
                if stage_param
                else None
            )
            try:
                outline_synth = self._synthesize_outline_output(
                    outline_dir_resolved,
                    stage_active_phases=stage_active_phases,
                )
            except Exception as e:  # noqa: BLE001 — defensive
                logger.error(
                    "outline reuse: synthesis raised %s; falling through",
                    e,
                )
                outline_synth = {}
            force_rerun = bool(workflow_params.get("force_rerun"))
            for phase_name, phase_out in outline_synth.items():
                if phase_name in phase_outputs:
                    continue
                if force_rerun:
                    phase_out = {**phase_out, "_completed": False}
                phase_outputs[phase_name] = phase_out

        # Resume restoration: when this is a --resume run, ``phase_outputs``
        # was reloaded from the persisted workflow-state JSON above. A
        # phase can be persisted as ``_completed=True`` while carrying an
        # EMPTY (or partial) extracted-output dict — e.g. when a prior run
        # marked the phase complete but ``_extract_phase_outputs`` captured
        # nothing (no COMPLETE task results, or a crash between checkpoint-
        # complete and state-save). Those completed phases are skipped by
        # the loop's ``_completed`` guard below, so their canonical output
        # keys are never re-derived and downstream phases (content gates,
        # ``imscc_chunking`` reading ``packaging.package_path``) find empty
        # inputs and fail. Reconstruct the missing keys from on-disk
        # artifacts BEFORE the loop runs. Strictly additive: only fills
        # keys absent from the recorded dict, never overwrites, and is a
        # no-op on a fresh (non-resume) run where ``phase_outputs`` is
        # empty.
        self._restore_resume_phase_outputs(phase_outputs)

        # Graceful-stop launch handshake ("checkpoint on command"). run_id
        # resolves from the executor (stop_control falls back to
        # ED4ALL_RUN_ID internally when this is None).
        _stop_run_id = getattr(self.executor, "run_id", None)

        # (f) STOP_ALL refusal (D9): the global, operator-owned sentinel is
        # NEVER auto-cleared. While it exists, refuse to start ANY run — fresh
        # or --resume — with a loud, actionable error naming the ONLY command
        # that clears it. This is the master "halt everything" switch; a run
        # must not sneak past it. Resolved through get_state_runs_dir() so a
        # non-repo-root CWD sees the same file the CLI wrote (risk R2).
        _global_sentinel = get_state_runs_dir() / GLOBAL_SENTINEL_NAME
        if _global_sentinel.exists():
            msg = (
                f"Refusing to start workflow {workflow_id}: global stop "
                f"sentinel present at {_global_sentinel}. Every run is halted "
                "until an operator clears it with `ed4all stop --clear-all`."
            )
            logger.error(msg)
            return {"error": msg}

        # (e) Clear this run's OWN stale run-scoped sentinel BEFORE the status
        # flips to RUNNING. A prior attempt under the same run_id could have
        # left <run_id>/control/STOP_REQUESTED behind; without this clear a
        # fresh start or a --resume would pause on the FIRST phase boundary.
        # Only a stop requested AFTER this point (during this run) trips the
        # between-phase probe below. The operator-owned global STOP_ALL is
        # untouched (include_global defaults False).
        clear_stop(_stop_run_id)

        # Update workflow status
        workflow_state["status"] = "RUNNING"
        workflow_state["started_at"] = datetime.now().isoformat()
        self._save_workflow_state(workflow_path, workflow_state)

        # Sort phases by dependency order
        sorted_phases = self._topological_sort(wf_config.phases)

        # Execute each phase
        all_results: Dict[str, Dict] = {}
        final_status = "COMPLETE"
        # Marketable-v1 A6 operator-failure-UX: when the loop breaks on a phase
        # failure, record WHICH phase failed and a short human reason so the GUI
        # can render an actionable failure panel instead of inferring the failing
        # phase from "last running" heuristics. Surfaced in both the persisted
        # workflow state and the returned payload. ``None`` on a clean run.
        failed_phase: Optional[str] = None
        failure_reason: Optional[str] = None
        # Graceful stop: which phase the run paused at (None on a clean or
        # failed run). Distinct from ``failed_phase`` — a pause is NOT a
        # failure; the phase is persisted resumable (``_completed=False``).
        paused_phase: Optional[str] = None

        # Wave1-I8 (Finding 7 of plans/dispatch-7-execution-inspection-2026-05.md):
        # emit one banner line per agent in ``AGENT_PROVIDER_ENV_MAP`` so
        # operators can see at workflow start whether each agent will
        # route to (a) an in-process local provider, (b) the Claude Code
        # subagent, or (c) the in-process stub fallback. Pure
        # observability — no behaviour change.
        self._emit_provider_banner()

        # Marketable-v1 A3 fail-fast guardrail: before running any phase,
        # verify every LLM-needing agent in the phases that will actually
        # run resolves its generation through the in-process provider
        # lattice (or has an explicit session / stub opt-in). Otherwise a
        # GUI-launched / headless run would enqueue a mailbox agent_task
        # nobody services (hang) or silently degrade to a templated stub.
        # Raises AuthoringProviderRouteError with an actionable message.
        self._enforce_authoring_provider_route(sorted_phases, workflow_params)

        # Bug B (resume stop-after integrity): ``--stop-after`` must halt the
        # run AFTER the named phase regardless of HOW that phase became
        # satisfied — executed this run, skipped-as-already-complete on
        # resume, skipped-optional, or reused (--reuse-objectives). The
        # in-loop stop check at the bottom of a normal phase execution is only
        # reached on the EXECUTE path; every early ``continue`` (skip paths)
        # bypasses it, which let a resumed run march past ``--stop-after
        # dart_conversion`` into staging→course_planning. This helper is
        # invoked at each such satisfied/continue point; it records the
        # deliberate halt marker + persists it and returns True so the caller
        # can ``break`` out of the phase loop.
        def _stop_after_now(pname: str, note: str) -> bool:
            if stop_after_phase and pname == stop_after_phase:
                logger.info(
                    "Stopping workflow after phase '%s' (--stop-after)%s; "
                    "skipping all subsequent phases.",
                    pname, note,
                )
                workflow_state["stopped_after"] = pname
                self._save_workflow_state(workflow_path, workflow_state)
                return True
            return False

        # Graceful-stop between-phase probe (modeled on ``_stop_after_now``).
        # At the TOP of every phase iteration — before the next phase
        # dispatches — probe the stop sentinel (run-scoped OR global STOP_ALL).
        # When set, the loop halts with ``final_status="PAUSED"`` so downstream
        # phases never run. Best-effort: stop_control already degrades OSError
        # to False; the extra guard keeps any unexpected error from crashing
        # the run.
        def _stop_requested_now() -> bool:
            try:
                return stop_requested(_stop_run_id)
            except Exception:  # noqa: BLE001 — a stop probe must never crash the run
                return False

        for phase_idx, phase in enumerate(sorted_phases):
            phase_name = phase.name

            # Graceful stop (a): halt cleanly BEFORE dispatching this phase
            # when a stop was requested mid-run. Downstream phases never run;
            # status = PAUSED (never FAILED). This phase was not dispatched, so
            # it is not ``_completed`` — a ``--resume`` re-enters here. The
            # previous phase already handed the GPU card over on its
            # success-path sweep, but run an explicit (idempotent, best-effort)
            # sweep so a pause landing right after a phase boundary still
            # leaves a clean card for the resume.
            if _stop_requested_now():
                logger.info(
                    "Graceful stop observed before phase '%s'; pausing "
                    "workflow (this and all subsequent phases skipped).",
                    phase_name,
                )
                final_status = "PAUSED"
                paused_phase = phase_name
                workflow_state["paused_phase"] = phase_name
                self._save_workflow_state(workflow_path, workflow_state)
                await self._gpu_lifecycle_sweep(phase_name)
                break

            # Skip already-completed phases (crash recovery).
            #
            # Resume-integrity invariant: a phase whose validation gates
            # FAILED must NOT be treated as resumable-complete. A prior
            # (buggy) run could persist a gate-failed phase as
            # ``_completed=True`` WITH ``_gates_passed=False`` (the stamp
            # at the bottom of this loop sets ``_completed=True``
            # unconditionally, then the workflow breaks on the gate
            # failure AFTER the state was already saved). Skipping such a
            # phase on ``--resume`` marches the workflow downstream on its
            # bad output. We therefore require BOTH ``_completed`` AND
            # gates-not-failed to skip.
            #
            # Backward-compat with old + benign state:
            #   * ``_gates_passed`` absent  -> default True -> skip
            #     (pre-``_gates_passed`` completed phases; optional-phase
            #     skip markers that stamp ``_completed`` but no gate flag).
            #   * ``_gates_passed is True``  -> skip (normal happy resume).
            #   * ``_gates_passed is False`` -> DO NOT skip -> re-run the
            #     phase (old buggy gate-failed state + new state alike).
            recorded_out = phase_outputs.get(phase_name)
            if (
                isinstance(recorded_out, dict)
                and recorded_out.get("_completed")
                and recorded_out.get("_gates_passed", True) is not False
            ):
                logger.info(f"Skipping already-completed phase: {phase_name}")
                # Bug B: --stop-after must halt AFTER this phase even when it
                # was satisfied by a skip-as-already-complete on resume, never
                # advance into downstream phases on the resumed run.
                if _stop_after_now(phase_name, "; already complete on resume"):
                    break
                continue

            # Check if this optional phase should be skipped
            if self._should_skip_phase(phase, workflow_params):
                logger.info(f"Skipping optional phase: {phase_name}")
                # Preserve pre-populated data (from _synthesize_outline_output) if present;
                # merge the skip markers in rather than overwriting. Downstream phases
                # pull keys like ``project_id`` via ``inputs_from``, so wiping the dict
                # breaks Phase-5 stage subcommands with --force.
                existing = phase_outputs.get(phase_name) or {}
                phase_outputs[phase_name] = {
                    **existing,
                    "_skipped": True,
                    "_completed": True,
                }
                workflow_state["phase_outputs"] = phase_outputs
                self._save_workflow_state(workflow_path, workflow_state)
                # Bug B: honour --stop-after when the named phase is an
                # optional phase that gets skipped (e.g. trainforge_assessment
                # under --no-assessments) — halt at its slot, don't run on.
                if _stop_after_now(phase_name, "; optional phase skipped"):
                    break
                continue

            # Check that all dependencies completed
            if not self._dependencies_met(phase, phase_outputs):
                logger.error(
                    f"Phase {phase_name} dependencies not met: {phase.depends_on}"
                )
                final_status = "FAILED"
                failed_phase = phase_name
                failure_reason = (
                    f"dependencies not met: {', '.join(phase.depends_on or [])}"
                )
                break

            # Wave 80 Worker A: honour --reuse-objectives by synthesising
            # the course_planning phase_output from the user-supplied
            # objectives JSON instead of dispatching the course-outliner
            # subagent. Stable across re-runs (no LLM nondeterminism),
            # preserving chunk learning_outcome_refs continuity. We do
            # this just-in-time (inside the phase loop) rather than
            # pre-loop because the synthesised output needs project_id
            # from objective_extraction, which hasn't run pre-loop.
            if (
                phase_name == "course_planning"
                and workflow_params.get("reuse_objectives_path")
            ):
                synthesized_planning = (
                    self._synthesize_course_planning_reuse_output(
                        workflow_params, phase_outputs,
                    )
                )
                if synthesized_planning is not None:
                    logger.info(
                        "course_planning: reusing user-supplied objectives "
                        "from %s; skipping course-outliner dispatch",
                        workflow_params.get("reuse_objectives_path"),
                    )
                    phase_outputs[phase_name] = synthesized_planning
                    workflow_state["phase_outputs"] = phase_outputs
                    self._save_workflow_state(workflow_path, workflow_state)
                    all_results[phase_name] = {
                        "task_count": 0,
                        "completed": 0,
                        "failed": 0,
                        "gates_passed": True,
                    }
                    # Bug B: --stop-after course_planning must halt even when
                    # planning was satisfied by --reuse-objectives (no dispatch).
                    if _stop_after_now(phase_name, "; objectives reused"):
                        break
                    continue
                else:
                    # Synthesis failed (e.g. project dir not yet created
                    # or objectives file unreadable). Surface as a hard
                    # failure: the user explicitly opted in to reuse, so
                    # silently falling back to a fresh LO mint would
                    # defeat the purpose.
                    logger.error(
                        "course_planning: --reuse-objectives synthesis "
                        "failed; aborting workflow"
                    )
                    final_status = "FAILED"
                    failed_phase = phase_name
                    failure_reason = (
                        "--reuse-objectives synthesis failed "
                        "(project dir not created or objectives file unreadable)"
                    )
                    break

            logger.info(f"Starting phase {phase_idx + 1}/{len(sorted_phases)}: {phase_name}")

            # Route parameters from workflow params + prior phase outputs
            routed_params = self._route_params(phase_name, workflow_params, phase_outputs)

            # Create tasks for this phase
            tasks = self._create_phase_tasks(
                workflow_id, phase, routed_params, workflow_params, phase_outputs
            )

            # Add tasks to workflow state
            workflow_state.setdefault("tasks", []).extend(tasks)
            self._save_workflow_state(workflow_path, workflow_state)

            # Get validation gate configs from phase
            gate_configs = getattr(phase, "validation_gates", None)

            # Execute the phase.
            #
            # Wave 23 Sub-task A: thread accumulated phase_outputs +
            # workflow_params through to the executor so the per-gate
            # input router can build validator-specific inputs. Without
            # these, every gate received a generic artifacts blob and
            # silently failed / skipped.
            # Wave 33 Bug B: hand the executor a way to extract the
            # current phase's outputs BEFORE the gate router runs.
            # Pre-Wave-33 extraction happened here (post-execute_phase)
            # so gate builders never saw the current phase's keys and
            # six gates silently skipped with "missing inputs: *".
            #
            # VRAM doctor: capture free-VRAM BEFORE the phase runs so a
            # crash / OOM mid-phase still leaves a forensic timeline. No-op
            # + zero-overhead unless ED4ALL_VRAM_DOCTOR is on.
            await self._vram_doctor_snapshot(phase_name, "before", "phase_start")
            results, gates_passed, gate_results = await self.executor.execute_phase(
                workflow_id=workflow_id,
                phase_name=phase_name,
                phase_index=phase_idx,
                tasks=tasks,
                gate_configs=gate_configs,
                max_concurrent=getattr(phase, "max_concurrent", 5),
                phase_outputs=phase_outputs,
                workflow_params=workflow_params,
                extract_phase_outputs_fn=self._extract_phase_outputs,
                # Plumb the per-phase YAML ``batch_timeout_minutes`` (e.g.
                # content_generation_rewrite's 240) into execute_phase so a
                # slow local-7B phase is not killed at the executor-wide
                # env/30-min fallback. None (no YAML value) is byte-stable.
                phase_batch_timeout_minutes=getattr(
                    phase, "batch_timeout_minutes", None
                ),
            )
            # VRAM doctor: capture free-VRAM AFTER the phase returns, stamped
            # with the gate verdict, so the trajectory shows the per-phase
            # delta (e.g. a model that loaded + never released the card).
            await self._vram_doctor_snapshot(
                phase_name, "after", "phase_end",
                extra={"phase_passed": bool(gates_passed)},
            )

            # Extract outputs from results
            extracted = self._extract_phase_outputs(phase_name, results)

            # Graceful stop (b): a phase interrupted mid-run comes back with
            # one or more PAUSED task results (the executor maps
            # GracefulStopRequested -> ExecutionResult(status="PAUSED"): no
            # retry, no poison record). This MUST be handled BEFORE the
            # anti-zombie (zero-success) and Bug-A (partial-completion) guards
            # below — both would otherwise stamp the phase FAILED, because a
            # paused phase has completed_count < len(tasks). Persist it
            # ``_completed=False, _paused=True`` so the completed-phase skip
            # guard re-runs it on ``--resume`` (its per-unit sidecars replay
            # the already-finished units — worst-case loss is one in-flight LLM
            # call). Run an explicit GPU lease sweep (the success-path sweep at
            # the bottom of the loop is never reached on this break) and halt
            # with status PAUSED (never FAILED).
            phase_paused = any(
                getattr(r, "status", None) == "PAUSED" for r in results.values()
            )
            if phase_paused:
                paused_count = sum(
                    1 for r in results.values()
                    if getattr(r, "status", None) == "PAUSED"
                )
                completed_now = sum(
                    1 for r in results.values() if r.status == "COMPLETE"
                )
                logger.info(
                    "Phase %s paused (graceful stop): %d completed, %d paused "
                    "of %d task(s); halting workflow with status PAUSED.",
                    phase_name, completed_now, paused_count, len(tasks),
                )
                extracted["_completed"] = False
                extracted["_paused"] = True
                extracted["_gates_passed"] = gates_passed
                extracted["_gate_results"] = list(gate_results or [])
                phase_outputs[phase_name] = extracted
                workflow_state["phase_outputs"] = phase_outputs
                workflow_state["paused_phase"] = phase_name
                self._save_workflow_state(workflow_path, workflow_state)
                all_results[phase_name] = {
                    "task_count": len(tasks),
                    "completed": completed_now,
                    "paused": paused_count,
                    "gates_passed": gates_passed,
                }
                final_status = "PAUSED"
                paused_phase = phase_name
                await self._gpu_lifecycle_sweep(phase_name)
                break

            # Anti-zombie guard: refuse to stamp ``_completed=True`` on a
            # non-optional phase that dispatched at least one task, had
            # ZERO of those tasks return a COMPLETE (success) result, AND
            # produced no canonical output keys. Pre-guard such phases
            # were persisted as ``_completed=True, keys=[]`` (the real
            # motivating run: a ``content_generation`` checkpoint with
            # ``tasks_completed: []`` + failed tasks). On ``--resume``
            # those empty-completed phases are skipped, starving every
            # downstream phase that needs a canonical key (content gates,
            # ``imscc_chunking`` needing ``packaging.package_path``). We
            # treat the phase as FAILED so the workflow stops here and a
            # later resume re-runs it instead of silently advancing.
            #
            # Carefully scoped so it does NOT fire on legitimate
            # no-output phases:
            #   * Validator-only phases (``agents: []`` → synthesised
            #     ``phase-handler`` task) that PASS their gates emit a
            #     COMPLETE result with no canonical keys — keyed on a
            #     COMPLETE result existing, those still complete.
            #   * Optional phases (``optional: true``, e.g.
            #     ``training_synthesis``) are excluded outright; they may
            #     be skipped/empty by design.
            #   * Phases that ran zero tasks (pure gate-chain phases) are
            #     excluded — there is nothing that "failed".
            #   * Gate failures are handled by the existing fail path
            #     below; this guard only covers the dispatched-but-all-
            #     failed-with-no-output case.
            any_task_succeeded = any(
                r.status == "COMPLETE" for r in results.values()
            )
            has_canonical_output = bool(extracted)
            if (
                not getattr(phase, "optional", False)
                and len(tasks) > 0
                and not any_task_succeeded
                and not has_canonical_output
            ):
                logger.error(
                    "phase %s marked failed: zero successful tasks and no "
                    "outputs (dispatched=%d, complete=0, keys=[]); refusing "
                    "to stamp _completed to avoid a passed-but-no-artifact "
                    "zombie that would be skipped on --resume",
                    phase_name,
                    len(tasks),
                )
                # Persist the phase as explicitly NOT completed so a
                # later resume re-runs it rather than skipping it.
                extracted["_completed"] = False
                extracted["_gates_passed"] = gates_passed
                extracted["_gate_results"] = list(gate_results or [])
                phase_outputs[phase_name] = extracted
                workflow_state["phase_outputs"] = phase_outputs
                self._save_workflow_state(workflow_path, workflow_state)
                all_results[phase_name] = {
                    "task_count": len(tasks),
                    "completed": 0,
                    "failed": sum(
                        1 for r in results.values()
                        if r.status in ("ERROR", "TIMEOUT", "FAILED")
                    ),
                    "gates_passed": gates_passed,
                }
                final_status = "FAILED"
                failed_phase = phase_name
                failure_reason = (
                    f"zero successful tasks and no outputs "
                    f"(dispatched={len(tasks)}, complete=0)"
                )
                break

            # Bug A (partial-artifact resume trap): extend the anti-zombie
            # contract above from "ZERO successful tasks" to "not EVERY
            # dispatched task succeeded". A non-optional multi-task phase where
            # SOME tasks COMPLETE-d but others failed / timed out / were
            # poisoned MUST NOT be persisted as ``_completed=True``. The live
            # motivating run: a 10-PDF ``dart_conversion`` batch where 1 PDF
            # converted (ch09) and 9 hit the 24000s batch timeout. The 1
            # success made ``any_task_succeeded`` True (so the guard above did
            # not fire) and produced canonical output keys, so the loop stamped
            # ``_completed=True, _gates_passed=True`` — and a subsequent
            # ``--resume`` saw a satisfied phase and SKIPPED it whole, so the 9
            # unfinished conversions never re-ran. Completeness is measured by
            # COMPLETE-count == dispatched-count (this also catches POISON_PILL,
            # which the ERROR/TIMEOUT/FAILED ``phase_failed`` check below
            # misses). Stamp ``_completed=False`` and fail the workflow here so
            # a later resume re-runs the phase (its per-task checkpoints let the
            # already-converted units be reused). Excludes optional phases
            # (partial by design), zero-task phases (nothing dispatched), and
            # fully-complete phases (byte-stable happy path).
            completed_count = sum(
                1 for r in results.values() if r.status == "COMPLETE"
            )
            if (
                not getattr(phase, "optional", False)
                and len(tasks) > 0
                and completed_count < len(tasks)
            ):
                logger.error(
                    "phase %s marked failed: partial completion "
                    "(dispatched=%d, complete=%d); refusing to stamp "
                    "_completed to avoid a partial-artifact zombie that "
                    "--resume would skip whole",
                    phase_name, len(tasks), completed_count,
                )
                extracted["_completed"] = False
                extracted["_gates_passed"] = gates_passed
                extracted["_gate_results"] = list(gate_results or [])
                phase_outputs[phase_name] = extracted
                workflow_state["phase_outputs"] = phase_outputs
                self._save_workflow_state(workflow_path, workflow_state)
                all_results[phase_name] = {
                    "task_count": len(tasks),
                    "completed": completed_count,
                    "failed": sum(
                        1 for r in results.values()
                        if r.status in ("ERROR", "TIMEOUT", "FAILED", "POISON_PILL")
                    ),
                    "gates_passed": gates_passed,
                }
                final_status = "FAILED"
                failed_phase = phase_name
                failure_reason = (
                    f"partial completion: {completed_count} of {len(tasks)} "
                    f"task(s) completed"
                )
                break

            extracted["_completed"] = True
            extracted["_gates_passed"] = gates_passed
            # Worker W5: stash the per-phase gate_results chain so the
            # post-loop ``courseforge_validation_report.json`` aggregator
            # can include phases that don't write their own report.json
            # (packaging, libv2_archival, etc.). The
            # ``_gate_results`` key is private (underscore-prefixed) and
            # gets stripped from the run_workflow return payload by the
            # existing dict comprehension below.
            extracted["_gate_results"] = list(gate_results or [])
            phase_outputs[phase_name] = extracted

            # Phase 5 Subtask 4: write the operator-facing
            # ``02_validation_report/report.json`` aggregation after
            # the ``inter_tier_validation`` and
            # ``post_rewrite_validation`` phases complete. The shipped
            # ``_run_inter_tier_validation`` helper writes JSONL only
            # (``blocks_validated_path`` + ``blocks_failed_path``); the
            # operator-facing structured per-block summary is a Phase 5
            # deliverable. Best-effort — failure to write the report
            # does NOT abort the workflow (it's an aggregation; the
            # raw JSONL is the source of truth).
            if phase_name in ("inter_tier_validation", "post_rewrite_validation"):
                try:
                    self._write_validation_report(
                        workflow_id=workflow_id,
                        phase_name=phase_name,
                        phase_output=extracted,
                        gate_results_list=gate_results,
                    )
                except Exception as exc:  # noqa: BLE001 — defensive
                    logger.warning(
                        "Phase 5 validation_report writer failed for "
                        "%s (non-fatal): %s",
                        phase_name, exc,
                    )

            # Persist phase outputs
            workflow_state["phase_outputs"] = phase_outputs
            self._save_workflow_state(workflow_path, workflow_state)

            all_results[phase_name] = {
                "task_count": len(tasks),
                "completed": sum(1 for r in results.values() if r.status == "COMPLETE"),
                # Wave 33 Bug C: count "FAILED" alongside "ERROR" and
                # "TIMEOUT" so tool envelopes with ``success=False``
                # surface in the phase summary instead of being
                # silently counted as completed.
                "failed": sum(
                    1 for r in results.values()
                    if r.status in ("ERROR", "TIMEOUT", "FAILED")
                ),
                "gates_passed": gates_passed,
            }

            # Check if phase failed
            # Wave 33 Bug C: include "FAILED" status so phases that had
            # every task return ``success=False`` envelopes stop the
            # workflow instead of advancing with a stale "12/12
            # complete" count.
            phase_failed = any(
                r.status in ("ERROR", "TIMEOUT", "FAILED")
                for r in results.values()
            )
            if phase_failed and not getattr(phase, "optional", False):
                logger.error(f"Phase {phase_name} failed, stopping workflow")
                final_status = "FAILED"
                failed_phase = phase_name
                failed_tasks = sum(
                    1 for r in results.values()
                    if r.status in ("ERROR", "TIMEOUT", "FAILED")
                )
                failure_reason = (
                    f"{failed_tasks} of {len(tasks)} task(s) failed"
                )
                break

            if not gates_passed and not getattr(phase, "optional", False):
                logger.error(f"Phase {phase_name} failed validation gates, stopping workflow")
                final_status = "FAILED"
                failed_phase = phase_name
                failure_reason = _summarize_gate_failure(gate_results)
                break

            # Deterministic GPU lease hand-off. The phase has now completed
            # SUCCESSFULLY (task results in, gates passed, outputs persisted), so
            # release the resident local model(s) + torch allocator cache before
            # the next phase dispatches — every GPU stage loads, runs, and hands
            # the card over. Ordered AFTER the doctor ``"after"`` snapshot above
            # (end-of-phase residency stays observable) and BEFORE the next
            # phase / any early-stop. No-op + zero-overhead unless
            # ED4ALL_GPU_LIFECYCLE is on (default ON); best-effort so a sweep
            # failure can never change final_status. Never fires for a
            # FAILED/partial phase (those break above) or a resume-skipped phase
            # (skipped earlier via ``continue``).
            await self._gpu_lifecycle_sweep(phase_name)

            # Hosted-large build profile GAP-2 fix — clean early-stop. The named
            # phase has now completed successfully (its outputs are persisted +
            # gates passed); halt before any later phase runs. Records a
            # logged reason + a ``stopped_after`` marker on the workflow state
            # so a resume / audit can see the deliberate halt (distinct from a
            # FAILED phase). final_status stays the loop's success value.
            if _stop_after_now(phase_name, ""):
                break

        # Finalize workflow state
        workflow_state["status"] = final_status
        workflow_state["completed_at"] = datetime.now().isoformat()
        # Marketable-v1 A6: persist the structured failure surface so the GUI
        # (and any resume / audit) can read which phase failed + why directly
        # from the workflow state file rather than inferring it.
        if failed_phase is not None:
            workflow_state["failed_phase"] = failed_phase
            workflow_state["failure_reason"] = failure_reason
        if paused_phase is not None:
            workflow_state["paused_phase"] = paused_phase
        self._save_workflow_state(workflow_path, workflow_state)

        # (d) Graceful-stop resume hint. The status was stamped PAUSED above
        # (never FAILED); tell the operator exactly how to pick the run back
        # up. The aggregators below still run (D4) — they are read-only
        # rollups and never alter ``final_status``.
        if final_status == "PAUSED":
            logger.info(
                "Workflow %s PAUSED by graceful stop%s. Resume with: "
                "ed4all run --resume %s",
                workflow_id,
                f" at phase '{paused_phase}'" if paused_phase else "",
                workflow_id,
            )

        # Worker W5 (GPT-feedback follow-up): post-loop aggregator that
        # walks every per-phase ``02_validation_report/report.json``
        # plus any in-memory ``_gate_results`` we just stashed and
        # writes a single top-level
        # ``<project_path>/courseforge_validation_report.json``. Best-
        # effort — aggregator failure (missing project_path, OSError
        # on write) does NOT alter ``final_status``; the per-phase
        # reports remain the source of truth.
        aggregator_path = self._maybe_write_courseforge_validation_report(
            workflow_id=workflow_id,
            workflow_params=workflow_params,
            phase_outputs=phase_outputs,
        )

        # GPT Feedback v2 Wave 2 (W2.B): post-loop TrainForge assessment-
        # quality aggregator. Walks ``phase_outputs`` for the
        # ``training_synthesis`` / ``trainforge_assessment`` /
        # ``libv2_archival`` ``_gate_results`` chains plus the
        # ``quality_report.json::assessments`` dimension and writes a
        # single top-level
        # ``<libv2_course>/quality/trainforge_assessment_quality_report.json``.
        # Best-effort — aggregator failure does NOT alter
        # ``final_status``; the per-phase reports remain authoritative.
        trainforge_aggregator_path = (
            self._maybe_write_trainforge_assessment_quality_report(
                workflow_id=workflow_id,
                workflow_params=workflow_params,
                phase_outputs=phase_outputs,
            )
        )

        # GPT Feedback v2 Wave 3 (W3.E): post-loop coverage-map aggregator.
        # Builds an objective-keyed table linking objectives -> chunks ->
        # questions -> training_pairs and writes a single top-level
        # ``<libv2_course>/coverage_map.json`` (or
        # ``<trainforge_dir>/coverage_map.json`` when archival hasn't run).
        # Best-effort — aggregator failure does NOT alter ``final_status``.
        coverage_map_path = self._maybe_write_coverage_map(
            workflow_id=workflow_id,
            workflow_params=workflow_params,
            phase_outputs=phase_outputs,
        )

        # IB6.6: post-loop block-quality rollup aggregator. Reads the
        # ``block_quality_rubric`` GateResult metadata stashed in the
        # post_rewrite_validation / inter_tier_validation ``_gate_results``
        # chain, rolls the per-block 8-dim 0-3 scores up to module + course
        # tiers (BOTH mean and per-dim minimum-floor paths + the 3 hard gates),
        # and writes ``<libv2_course>/block_quality_rollup_report.json``. Only
        # runs when ED4ALL_BLOCK_QUALITY_RUBRIC is on; best-effort — aggregator
        # failure does NOT alter ``final_status``.
        block_quality_rollup_path = self._maybe_write_block_quality_rollup(
            workflow_id=workflow_id,
            workflow_params=workflow_params,
            phase_outputs=phase_outputs,
        )

        # NVIDIA-KG item 3 (GPT-fb-12-may item 2 mirror): post-loop
        # edge-consensus aggregator. The ``concept_extraction`` phase
        # stamps ``edge_status`` + ``consensus_signals[]`` at authoring
        # time (``_run_concept_extraction``), but semantic graphs that
        # land via OTHER routes (the process_course / IMSCC path writing
        # ``<libv2_course>/graph/concept_graph_semantic.json``, or
        # pre-fix corpora under ``concept_graph/``) reach workflow end
        # un-stamped. Walk both layouts under the LibV2 course dir,
        # stamp any un-stamped graph in place, and write the sibling
        # ``edge_consensus_report.json``. Deterministic (cross-rule
        # matrix only — no LLM; NLI stays off unless
        # ``TRAINFORGE_EDGE_NLI``). Already-stamped graphs with an
        # existing sibling report are skipped untouched (idempotent
        # safety with the authoring-time wiring). Best-effort —
        # aggregator failure does NOT alter ``final_status``.
        edge_consensus_report_paths = self._maybe_write_edge_consensus_reports(
            workflow_id=workflow_id,
            workflow_params=workflow_params,
            phase_outputs=phase_outputs,
        )

        # GPT Feedback v2 Wave 3 (W3.G): post-loop master promotion-chain
        # aggregator (governance G1). Walks all 9 arrows of the
        # DART -> eval-report chain, reads each per-stage report best-
        # effort, and writes a single canonical
        # ``<libv2_course>/courseforge_promotion_chain_report.json``.
        # Anti-silent-degradation contract: missing per-stage reports
        # surface as fail-promotion-decision rows so an operator can see
        # the silent-skip class. Best-effort — aggregator failure does
        # NOT alter ``final_status``; the per-stage reports remain the
        # source of truth.
        promotion_chain_path = self._maybe_write_promotion_chain_report(
            workflow_id=workflow_id,
            workflow_params=workflow_params,
            phase_outputs=phase_outputs,
        )

        # Wave 4 (W4.1): post-loop concept-coverage aggregator. Concept-
        # graph analogue of coverage_map — per concept node, tallies which
        # pedagogical surfaces touch it (explained / defined-in-glossary /
        # assessed / demonstrated / prereq-scaffolded). Gated OFF by default
        # (ED4ALL_CONCEPT_COVERAGE) so a default run is byte-identical (no
        # file). Best-effort — aggregator failure does NOT alter
        # ``final_status``.
        concept_coverage_path = self._maybe_write_concept_coverage(
            workflow_id=workflow_id,
            workflow_params=workflow_params,
            phase_outputs=phase_outputs,
        )

        # Wave 4 (W4.6): post-loop course-intelligence-level aggregator. A
        # deterministic (no-model) 0-5 self-assessment tallying PRESENT
        # capability artifacts (key-terms glossary, concept graph,
        # assessment density, prereq cross-links, FAQ-if-present). Gated OFF
        # by default (ED4ALL_INTELLIGENCE_RUBRIC) so a default run is
        # byte-identical (no file). Best-effort — aggregator failure does
        # NOT alter ``final_status``.
        intelligence_level_path = self._maybe_write_intelligence_level(
            workflow_id=workflow_id,
            workflow_params=workflow_params,
            phase_outputs=phase_outputs,
        )

        # Roadmap T3: post-loop accessibility-conformance (ACR) aggregator.
        # Inverts the gate-level WCAG issue stream into a per-success-criterion
        # conformance table (supports / partially_supports / does_not_support /
        # not_evaluated) with EXPLICIT not_evaluated rows for criteria outside
        # automated static-HTML reach. Emitted at
        # ``<libv2_course>/quality/accessibility_conformance.json`` (trainforge-
        # dir fallback). Best-effort — aggregator failure does NOT alter
        # ``final_status``.
        accessibility_conformance_path = (
            self._maybe_write_accessibility_conformance(
                workflow_id=workflow_id,
                workflow_params=workflow_params,
                phase_outputs=phase_outputs,
            )
        )

        # Roadmap OP2: post-loop build-cost metering aggregator. Sums per-phase
        # wall-clock (checkpoints), GPU residency (vram_trajectory.jsonl), and
        # LLM calls/tokens (llm_usage.jsonl) into
        # ``<libv2_course>/build_cost_report.json`` (trainforge-dir fallback).
        # No LLM decisions — pure metering. Best-effort — aggregator failure
        # does NOT alter ``final_status``.
        build_cost_path = self._maybe_write_build_cost_report(
            workflow_id=workflow_id,
            workflow_params=workflow_params,
            phase_outputs=phase_outputs,
        )

        return {
            "workflow_id": workflow_id,
            "status": final_status,
            "failed_phase": failed_phase,
            "failure_reason": failure_reason,
            "paused_phase": paused_phase,
            "phase_results": all_results,
            "phase_outputs": {
                k: {pk: pv for pk, pv in v.items() if not pk.startswith("_")}
                for k, v in phase_outputs.items()
            },
            "courseforge_validation_report_path": (
                str(aggregator_path) if aggregator_path else None
            ),
            "trainforge_assessment_quality_report_path": (
                str(trainforge_aggregator_path)
                if trainforge_aggregator_path
                else None
            ),
            "coverage_map_path": (
                str(coverage_map_path) if coverage_map_path else None
            ),
            "block_quality_rollup_report_path": (
                str(block_quality_rollup_path)
                if block_quality_rollup_path
                else None
            ),
            "edge_consensus_report_paths": (
                [str(p) for p in edge_consensus_report_paths]
                if edge_consensus_report_paths
                else None
            ),
            "promotion_chain_report_path": (
                str(promotion_chain_path) if promotion_chain_path else None
            ),
            "concept_coverage_path": (
                str(concept_coverage_path) if concept_coverage_path else None
            ),
            "intelligence_level_report_path": (
                str(intelligence_level_path)
                if intelligence_level_path
                else None
            ),
            "accessibility_conformance_report_path": (
                str(accessibility_conformance_path)
                if accessibility_conformance_path
                else None
            ),
            "build_cost_report_path": (
                str(build_cost_path) if build_cost_path else None
            ),
        }

    # Env opt-ins that authorize the non-lattice dispatch paths. A run
    # that hits one of these has deliberately accepted the
    # mailbox-subagent (Claude session) or templated-stub path, so the
    # guardrail steps aside.
    _SESSION_SERVICED_ENVS = (
        # Operator is running inside a Claude Code session (or an
        # external servicer, e.g. scripts/mailbox_servicer.py) that will
        # drain the mailbox. Set this when a servicer is attached.
        "ED4ALL_MAILBOX_SERVICED",
    )
    _STUB_OPT_IN_ENV = "LOCAL_DISPATCHER_ALLOW_STUB"

    @staticmethod
    def _env_truthy(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in (
            "1", "true", "yes", "on",
        )

    def _llm_agents_in_phases(
        self, phases: List[WorkflowPhase], workflow_params: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Return ``(phase_name, agent_type)`` for every subagent-classified
        (LLM-needing) agent in the phases that will actually run.

        Skips optional / env-gated / stage-whitelisted phases via the same
        ``_should_skip_phase`` predicate the run loop uses, so a guardrail
        decision never trips on a phase the workflow won't execute (e.g.
        ``trainforge_assessment`` under ``--no-assessments``).
        """
        out: List[Tuple[str, str]] = []
        for phase in phases:
            if self._should_skip_phase(phase, workflow_params):
                continue
            for agent in phase.agents or []:
                if agent in AGENT_SUBAGENT_SET:
                    out.append((phase.name, agent))
        return out

    def _apply_authoring_route_env(
        self, workflow_type: str, provider_hint: str = ""
    ) -> Dict[str, str]:
        """Fill the four AGENT_AUTHORING_PROVIDER_ENV_MAP envs for a turnkey run.

        W1 Gap C. CLI parity with
        ``gui.services.run_service._apply_authoring_route_env``: a headless /
        CLI run with no Claude session draining the mailbox must resolve every
        LLM-needing authoring agent through the in-process provider lattice, or
        ``_enforce_authoring_provider_route`` fails it. A bare
        ``ed4all run textbook-to-course --provider local`` would otherwise hard-
        fail because ``COURSEFORGE_PROVIDER`` / ``COURSEPLANNER_PROVIDER`` /
        ``TRAINFORGE_ASSESSMENT_PROVIDER`` are unset (only
        ``TRAINFORGE_SYNTHESIS_PROVIDER`` is filled today by
        ``_apply_corpus_generalization_defaults``).

        setdefault semantics — an env already set (operator export, GUI
        settings ``model_routing``, or a prior overlay) is left intact.
        Resolution per unset env: ``provider_hint`` > ``LLM_PROVIDER`` >
        ``"local"`` (license-clean default; an air-gapped Ollama/vLLM lattice
        provider that needs no key). The implementation is structurally
        identical to the GUI helper (drift-guarded by a test) but lives in
        ``MCP/`` so the orchestrator never imports the opt-in ``gui`` extra.

        W1 Gap A (appended below): when the resolved provider is local /
        ToS-clean OSS AND the two-pass router is enabled, redirect the
        block-routing policy to the all-local variant so the ``large``
        capability tier (canonical: anthropic) routes local, and fill the
        tier-default rewrite/outline provider envs so a no-policy-file run
        still resolves local instead of the ``_rewrite_provider``
        ``DEFAULT_PROVIDER="anthropic"`` code fallback.

        Returns the env vars this call set (logging / tests). A no-op for every
        non-pipeline workflow (mirrors ``_apply_corpus_generalization_defaults``
        scope) and for a run where the operator already pinned every env.
        """
        if workflow_type not in _CORPUS_GENERALIZATION_WORKFLOWS:
            return {}

        resolved = (
            (provider_hint or "").strip()
            or os.environ.get("LLM_PROVIDER", "").strip()
            or "local"
        )

        # Hosted-large build profile GAP-3 fix (LICENSING). The Gap C loop below
        # fills ALL FOUR authoring envs — INCLUDING the
        # ``TRAINFORGE_SYNTHESIS_PROVIDER`` training seat — with the resolved
        # provider. For ``--provider nvidia`` that would route the SLM TRAINING
        # corpus through the NVIDIA-hosted Llama-3.3 (ToS-restricted at scale
        # for training data; the adapter is a derivative work of those
        # outputs). The build also ``--stop-after imscc_chunking`` so
        # training_synthesis never executes, but the ROUTING must never resolve
        # nvidia regardless. So for the nvidia provider ONLY, the training seat
        # is pinned LOCAL (license-clean) instead of nvidia. setdefault still
        # honors an explicit operator export. See docs/LICENSING.md.
        _training_seat_env = AGENT_AUTHORING_PROVIDER_ENV_MAP.get(
            "training-synthesizer"
        )

        applied: Dict[str, str] = {}
        # Gap C — the four blessed authoring-route envs.
        for env_var in AGENT_AUTHORING_PROVIDER_ENV_MAP.values():
            if os.environ.get(env_var, "").strip():
                continue
            # GAP-3 licensing guard: never route the training seat to nvidia.
            if resolved == _CLOUD_SEAT_PROVIDER and env_var == _training_seat_env:
                os.environ[env_var] = "local"
                applied[env_var] = "local"
                continue
            os.environ[env_var] = resolved
            applied[env_var] = resolved

        # Gap A — all-local two-pass routing. Only when the resolved authoring
        # provider is local / ToS-clean OSS AND the two-pass router is enabled.
        if resolved in _LOCAL_OSS_PROVIDERS and self._env_truthy(_TWO_PASS_ENV):
            # Redirect the block-routing policy to the all-local variant so the
            # `large` capability tier (canonical: anthropic) routes local.
            # setdefault: an operator who already pinned the path keeps it.
            if not os.environ.get(_BLOCK_ROUTING_PATH_ENV, "").strip():
                os.environ[_BLOCK_ROUTING_PATH_ENV] = _ALL_LOCAL_BLOCK_ROUTING_PATH
                applied[_BLOCK_ROUTING_PATH_ENV] = _ALL_LOCAL_BLOCK_ROUTING_PATH
            # Belt-and-braces: also fill the tier-default rewrite/outline envs
            # so a NO-policy-file run (loader returns empty policy) still
            # resolves local instead of the `_rewrite_provider`
            # DEFAULT_PROVIDER="anthropic" code fallback.
            for env_var in _TWO_PASS_TIER_PROVIDER_ENVS:
                if os.environ.get(env_var, "").strip():
                    continue
                os.environ[env_var] = resolved
                applied[env_var] = resolved

        # Hosted-large build profile SETUP — the THIRD branch (sibling of the
        # Gap A local/together branch above). Only on an explicit
        # ``--provider nvidia`` (the vendor endpoint-registry key) AND the
        # two-pass router enabled. Default-OFF: with no ``--provider nvidia``
        # this branch never fires, so every routing decision + emitted artifact
        # is byte-identical to today. All setdefault (so explicit per-phase
        # operator/GUI overrides always win). NOTHING dispatches to the cloud
        # seat here — this only fills routing envs; the run is gated on the
        # operator's explicit launch-script flip + a real key.
        if resolved == _CLOUD_SEAT_PROVIDER and self._env_truthy(_TWO_PASS_ENV):
            # (1) Redirect the block-routing policy to the hosted-large variant
            #     (outline tier stays local 7B; rewrite tier escalates to the
            #     hosted large seat). setdefault: an operator-pinned path wins.
            if not os.environ.get(_BLOCK_ROUTING_PATH_ENV, "").strip():
                os.environ[_BLOCK_ROUTING_PATH_ENV] = (
                    _HOSTED_LARGE_BLOCK_ROUTING_PATH
                )
                applied[_BLOCK_ROUTING_PATH_ENV] = (
                    _HOSTED_LARGE_BLOCK_ROUTING_PATH
                )
            # (2) Pin the cloud model. Closes the 30B-nano registry-default
            #     leak (config/endpoints.yaml ``nvidia`` default_model is the
            #     nano). The canonical cloud-model knob is NVIDIA_LARGE_MODEL /
            #     the YAML — NEVER COURSEFORGE_REWRITE_MODEL (dead on the cloud
            #     tier; router.py:3223).
            if not os.environ.get(_HOSTED_LARGE_MODEL_ENV, "").strip():
                os.environ[_HOSTED_LARGE_MODEL_ENV] = _HOSTED_LARGE_MODEL_DEFAULT
                applied[_HOSTED_LARGE_MODEL_ENV] = _HOSTED_LARGE_MODEL_DEFAULT
            # (3) GAP 1 fix — reach the synthesis phases (objective_extraction /
            #     course_planning / concept_extraction). ``--provider`` only
            #     fills the four authoring envs above; WITHOUT this the synthesis
            #     seat stays local 7B while the rest runs on the hosted large seat.
            if not os.environ.get(
                _TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV, ""
            ).strip():
                os.environ[_TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV] = (
                    _CLOUD_SEAT_PROVIDER
                )
                applied[_TEXTBOOK_SYNTHESIS_PROVIDER_ROUTE_ENV] = _CLOUD_SEAT_PROVIDER
                # Its model env too, so the synthesis seat resolves the hosted
                # large model (not the registry nano). setdefault — operator
                # override wins.
                if not os.environ.get(
                    _TEXTBOOK_SYNTHESIS_MODEL_ROUTE_ENV, ""
                ).strip():
                    os.environ[_TEXTBOOK_SYNTHESIS_MODEL_ROUTE_ENV] = (
                        _HOSTED_LARGE_MODEL_DEFAULT
                    )
                    applied[_TEXTBOOK_SYNTHESIS_MODEL_ROUTE_ENV] = (
                        _HOSTED_LARGE_MODEL_DEFAULT
                    )
            # (4) GAP 3 fix (LICENSING) — DELIBERATELY DO NOT point the training
            #     seat at nvidia. ``TRAINFORGE_SYNTHESIS_PROVIDER`` is left
            #     local/untouched by this branch so the SLM training corpus can
            #     NEVER route through Llama-3.3 (ToS-restricted at scale for
            #     training data; the build also --stop-after imscc_chunking so
            #     training_synthesis never executes). See docs/LICENSING.md.
            #
            # (5) OPTIONAL product toggles LEFT UNSET by this branch — separate
            #     user decisions (gated on the later RUN discussion):
            #       - ED4ALL_OBJECTIVE_REVIEW_PROVIDER (objective-review pass)
            #       - ED4ALL_DYNAMIC_BLOCK_PLAN / _PROVIDER (large-model planner)
            #       - COURSEFORGE_OUTLINE_PROVIDER=nvidia (outline tier on the
            #         hosted large seat; the YAML keeps the outline first draft
            #         on local 7B by default)

        if applied:
            logger.info(
                "W1 authoring-route fill for %s (provider=%s): set %s",
                workflow_type,
                resolved,
                ", ".join(f"{k}={v}" for k, v in sorted(applied.items())),
            )
        return applied

    def _apply_corpus_generalization_defaults(self, workflow_type: str) -> Dict[str, str]:
        """Turn the corpus-generalization feature set ON for a pipeline run.

        Marketable-v1 A5 (mechanism "auto-on per run"). For
        ``textbook_to_course`` / ``course_generation`` runs only, fill every
        corpus-generalization env flag that is currently unset/empty with its
        measured-best value, so a fresh CLI/GUI run gets page-level concept
        tags, the measured graph-shaping quartet (prune + fragment-filter +
        merge + fan-out cap), dynamic CURIEs, and three-stage textbook
        synthesis by default.

        * setdefault semantics — an operator's explicit value (legacy or
          otherwise) is honored verbatim; only an unset/empty env is filled.
        * MEASURED-best wins over the roadmap prose: PAGE-level tags are the
          default, so ``TRAINFORGE_CHUNK_LOCAL_TAGS`` is intentionally NOT in
          the default map (chunk-local tags fragment the graph).
        * The synthesis provider is licensing-sensitive, so it resolves like
          the A3 authoring providers: ``LLM_PROVIDER`` > ``"local"``.
        * Returns the env vars this call actually set (for logging / tests).
          A no-op for every non-pipeline workflow and for a run where the
          operator already pinned every flag.
        """
        if workflow_type not in _CORPUS_GENERALIZATION_WORKFLOWS:
            return {}

        # Master opt-out (deterministic fixture-contract runs): skip the whole
        # A5 set, including the licensing-sensitive synthesis-provider envs, so
        # the run stays fully deterministic with no live-LLM dispatch.
        if self._env_truthy(_DISABLE_CORPUS_GENERALIZATION_ENV):
            logger.info(
                "A5 corpus-generalization defaults-on SKIPPED for %s "
                "(%s is set).",
                workflow_type,
                _DISABLE_CORPUS_GENERALIZATION_ENV,
            )
            return {}

        applied: Dict[str, str] = {}
        for env_var, value in _CORPUS_GENERALIZATION_ENV_DEFAULTS.items():
            if os.environ.get(env_var, "").strip():
                continue
            os.environ[env_var] = value
            applied[env_var] = value

        # Both synthesis-provider envs are licensing-sensitive (they select an
        # LLM backend whose ToS decides whether the corpus is trainable), so
        # they resolve like the A3 authoring providers — LLM_PROVIDER > "local"
        # (license-clean default) — rather than a hardcoded literal.
        #   * TEXTBOOK_SYNTHESIS_PROVIDER — authoring-adjacent (concept
        #     vocabulary / objectives); A5.
        #   * TRAINFORGE_SYNTHESIS_PROVIDER — the training-PAIR corpus (the SLM
        #     is a derivative work of these pairs). Defaulting it here is the
        #     CLI-side mirror of the GUI run_service authoring-route fill, so a
        #     CLI run no longer routes training-pair synthesis through the
        #     Claude Code session by default. (Marketable-v1 D4.)
        for _provider_env in (
            _TEXTBOOK_SYNTHESIS_PROVIDER_ENV,
            _TRAINFORGE_SYNTHESIS_PROVIDER_ENV,
        ):
            if os.environ.get(_provider_env, "").strip():
                continue
            resolved_provider = os.environ.get("LLM_PROVIDER", "").strip() or "local"
            # LICENSING guard: the TRAINING-PAIR seat (the corpus the SLM adapter
            # is a derivative work of) must NEVER auto-resolve to a ToS-restricted
            # provider. For anthropic/nvidia (Anthropic Commercial/Consumer Terms;
            # NVIDIA-hosted Llama-3.3), pin the AUTO-resolved training seat to the
            # license-clean "local" instead. This runs FIRST in run_workflow, so
            # the seat is correct regardless of helper order (the later
            # _apply_authoring_route_env GAP-3 guard would otherwise be dead —
            # its setdefault short-circuit fires because this loop already set
            # the seat). The textbook seat still follows the resolved provider;
            # only the training seat is guarded. The setdefault skip ABOVE still
            # honors an explicit operator export verbatim. See docs/LICENSING.md.
            if (
                _provider_env == _TRAINFORGE_SYNTHESIS_PROVIDER_ENV
                and resolved_provider in _LICENSE_RESTRICTED_SYNTHESIS
            ):
                resolved_provider = "local"
            os.environ[_provider_env] = resolved_provider
            applied[_provider_env] = resolved_provider

        if applied:
            logger.info(
                "A5 corpus-generalization defaults-on for %s: set %s",
                workflow_type,
                ", ".join(f"{k}={v}" for k, v in sorted(applied.items())),
            )
        return applied

    def _enforce_authoring_provider_route(
        self,
        phases: List[WorkflowPhase],
        workflow_params: Dict[str, Any],
    ) -> None:
        """Fail fast unless every LLM-needing phase resolves generation via
        the in-process provider lattice (the blessed authoring route).

        Marketable-v1 A3. For each subagent-classified agent in the phases
        that will run, the dispatch path resolves (mirroring
        ``TaskExecutor._invoke_tool``) to one of:

        * **lattice** — the agent's ``<AGENT>_PROVIDER`` env
          (``AGENT_AUTHORING_PROVIDER_ENV_MAP``) is set, so the executor
          short-circuits subagent dispatch and runs the in-process tool,
          which routes through the OpenAI-compatible provider registry. OK.
        * **session subagent** — no provider env, but
          ``ED4ALL_AGENT_DISPATCH=true`` AND a servicer opt-in
          (``ED4ALL_MAILBOX_SERVICED``) signals an attached Claude session /
          external servicer that will drain the mailbox. OK (operator's
          responsibility).
        * **stub** — ``LOCAL_DISPATCHER_ALLOW_STUB`` accepts the templated
          in-process stub (tests / dry runs). OK.
        * **hang / silent-degrade** — none of the above. A run here would
          enqueue an unserviced mailbox task (hang) or fall to a templated
          stub for an LLM-needing agent (silent degradation). FAIL FAST.

        The error names the fix: set the agent's provider env, or run inside
        a Claude session with ``ED4ALL_AGENT_DISPATCH=true`` +
        ``ED4ALL_MAILBOX_SERVICED=1``.
        """
        offenders: List[Dict[str, Any]] = []
        seen: set = set()
        agent_dispatch = _agent_dispatch_enabled()
        serviced = any(self._env_truthy(e) for e in self._SESSION_SERVICED_ENVS)
        stub_ok = self._env_truthy(self._STUB_OPT_IN_ENV)

        for phase_name, agent in self._llm_agents_in_phases(phases, workflow_params):
            if agent in seen:
                continue
            seen.add(agent)

            provider_env = AGENT_AUTHORING_PROVIDER_ENV_MAP.get(agent)
            if provider_env and os.environ.get(provider_env, "").strip():
                continue  # lattice route — blessed
            if stub_ok:
                continue  # explicit templated-stub opt-in
            if agent_dispatch and serviced:
                continue  # session subagent route, servicer attached

            offenders.append(
                {
                    "phase": phase_name,
                    "agent": agent,
                    "provider_env": provider_env,
                }
            )

        if not offenders:
            return

        lines: List[str] = []
        for o in offenders:
            if o["provider_env"]:
                lines.append(
                    f"  - phase '{o['phase']}' agent '{o['agent']}': set "
                    f"{o['provider_env']}=local (or together / another "
                    f"registered provider) to route via the in-process lattice"
                )
            else:
                lines.append(
                    f"  - phase '{o['phase']}' agent '{o['agent']}': "
                    f"session-only agent — run inside a Claude session with "
                    f"ED4ALL_AGENT_DISPATCH=true and ED4ALL_MAILBOX_SERVICED=1"
                )

        raise AuthoringProviderRouteError(
            "Run would dispatch an LLM-needing phase to an unserviced "
            "mailbox (hang) or a templated stub (silent degradation). The "
            "supported turnkey authoring route resolves LLM generation "
            "through the in-process provider lattice. Fix one of:\n"
            + "\n".join(lines)
            + "\n\nOr, to run inside a Claude Code session that services the "
            "mailbox, set ED4ALL_AGENT_DISPATCH=true and "
            "ED4ALL_MAILBOX_SERVICED=1. For tests / dry runs only, set "
            "LOCAL_DISPATCHER_ALLOW_STUB=1 to accept templated stubs."
        )

    def _emit_provider_banner(self) -> None:
        """Wave1-I8: emit one log line per agent in
        ``AGENT_PROVIDER_ENV_MAP`` summarising how each will dispatch
        at workflow start.

        Resolution order (matches the executor's
        ``_dispatch_task_via_tool`` precedence):

        1. If ``AGENT_PROVIDER_ENV_MAP[agent]`` env var is set and non-
           empty → ``local-provider`` (executor short-circuits subagent
           dispatch and routes to the in-process Wave-D provider).
        2. Else if ``ED4ALL_AGENT_DISPATCH=true`` → ``subagent
           (claude)`` (executor routes through
           ``dispatcher.dispatch_task`` to the Claude Code subagent).
        3. Else → ``in-process-stub`` (executor falls through to
           ``tool_registry[tool_name](**mapped_params)``).

        Pure observability — no behaviour change.
        """
        agent_dispatch = (
            os.environ.get("ED4ALL_AGENT_DISPATCH", "").strip().lower() == "true"
        )
        for agent, env_var in AGENT_PROVIDER_ENV_MAP.items():
            env_value = os.environ.get(env_var, "").strip()
            if env_value:
                logger.info(
                    "[provider-banner] %s: local-provider               "
                    "%s=%s",
                    agent,
                    env_var,
                    env_value,
                )
            elif agent_dispatch:
                logger.info(
                    "[provider-banner] %s: subagent (claude)            "
                    "ED4ALL_AGENT_DISPATCH=true",
                    agent,
                )
            else:
                logger.info(
                    "[provider-banner] %s: in-process-stub              "
                    "ED4ALL_AGENT_DISPATCH=false",
                    agent,
                )

    def _maybe_write_courseforge_validation_report(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Worker W5 helper — write top-level aggregator JSON if possible.

        Resolves ``project_path`` from the standard upstream signal
        (``phase_outputs['objective_extraction']``: explicit
        ``project_path`` key first, else
        ``Courseforge/exports/<project_id>``). Returns ``None`` when
        no project_path can be resolved (no Courseforge phases ran)
        or when the aggregator raises during build/write.
        """
        try:
            project_path = self._resolve_courseforge_project_path(
                phase_outputs
            )
            if project_path is None:
                logger.debug(
                    "courseforge_validation_report: no project_path "
                    "resolvable; skipping aggregator (run_id=%s)",
                    workflow_id,
                )
                return None

            # Local import to keep the workflow_runner import-time
            # dependency surface unchanged for non-Courseforge runs.
            from lib.aggregators.courseforge_validation_report import (
                CourseforgeValidationReport,
            )

            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = CourseforgeValidationReport(
                project_path=project_path,
                phase_outputs=phase_outputs,
                course_code=course_code,
                run_id=workflow_id,
            )
            output_path = (
                project_path / "courseforge_validation_report.json"
            )
            aggregator.write(output_path)
            logger.info(
                "courseforge_validation_report: wrote %s "
                "(run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "courseforge_validation_report aggregator failed "
                "(non-fatal, run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    @staticmethod
    def _resolve_courseforge_project_path(
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Resolve the Courseforge project export root from phase_outputs.

        Mirrors the ``reuse_objectives`` resolution chain at
        ``WorkflowRunner._synthesize_outline_output`` so the
        aggregator picks up the same path the phase handlers wrote
        per-phase reports under.
        """
        oe = phase_outputs.get("objective_extraction") or {}
        explicit = oe.get("project_path")
        if explicit:
            return Path(explicit)
        project_id = oe.get("project_id")
        if project_id:
            return courseforge_exports_dir() / project_id
        return None

    def _maybe_write_trainforge_assessment_quality_report(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Worker W2.B helper — write the Trainforge assessment-quality JSON.

        Path resolution priority:

        1. ``phase_outputs.libv2_archival.course_dir`` — the canonical
           LibV2 course root. Output lands at
           ``<course_dir>/quality/trainforge_assessment_quality_report.json``.
        2. Fallback to
           ``phase_outputs.trainforge_assessment.trainforge_dir`` /
           ``phase_outputs.training_synthesis.corpus_dir`` — emit at the
           Trainforge workspace root when LibV2 archival hasn't yet
           completed (e.g. partial textbook_to_course run that stops
           before ``libv2_archival``).

        Returns ``None`` when neither resolves (no Trainforge surfaces
        ran) or when the aggregator raises during build/write. Best-
        effort posture matches the courseforge aggregator: failure
        logs a warning and never fails the workflow.
        """
        try:
            # Local import to keep the workflow_runner import-time
            # dependency surface unchanged for non-Trainforge runs.
            from lib.aggregators.trainforge_assessment_quality_report import (
                TrainforgeAssessmentQualityReport,
            )

            archival = phase_outputs.get("libv2_archival") or {}
            course_dir_str = archival.get("course_dir")
            output_path: Optional[Path] = None
            libv2_course_path: Optional[Path] = None
            trainforge_dir: Optional[Path] = None

            if course_dir_str:
                libv2_course_path = Path(course_dir_str)
                output_path = (
                    libv2_course_path
                    / "quality"
                    / "trainforge_assessment_quality_report.json"
                )

            # Resolve trainforge_dir from any contributing surface so
            # the aggregator can read quality_report.json::assessments
            # even when libv2_archival ran (the trainforge_dir may have
            # been wiped post-archival).
            ta = phase_outputs.get("trainforge_assessment") or {}
            tdir_str = ta.get("trainforge_dir")
            if not tdir_str:
                ts = phase_outputs.get("training_synthesis") or {}
                tdir_str = ts.get("corpus_dir") or ts.get("trainforge_dir")
            if tdir_str:
                trainforge_dir = Path(tdir_str)

            # Fallback: write to <trainforge_dir>/trainforge_assessment_quality_report.json
            # when libv2_archival didn't run.
            if output_path is None and trainforge_dir is not None:
                output_path = (
                    trainforge_dir
                    / "trainforge_assessment_quality_report.json"
                )

            if output_path is None:
                logger.warning(
                    "trainforge_assessment_quality_report: no "
                    "libv2_archival.course_dir / training_synthesis."
                    "corpus_dir / trainforge_assessment.trainforge_dir "
                    "resolvable; skipping aggregator (run_id=%s)",
                    workflow_id,
                )
                return None

            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = TrainforgeAssessmentQualityReport(
                phase_outputs=phase_outputs,
                course_code=course_code,
                run_id=workflow_id,
                libv2_course_path=libv2_course_path,
                trainforge_dir=trainforge_dir,
            )
            aggregator.write(output_path)
            logger.info(
                "trainforge_assessment_quality_report: wrote %s "
                "(run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "trainforge_assessment_quality_report aggregator failed "
                "(non-fatal, run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    def _maybe_write_block_quality_rollup(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """IB6.6 helper — write block_quality_rollup_report.json if possible.

        Only runs when ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is on (the IB6 keystone
        flag); default-off runs skip entirely (byte-stable). Resolves per-block
        8-dim scores from the ``block_quality_rubric`` GateResult metadata
        stashed in the validation-phase ``_gate_results`` chain. Output root:
        the LibV2 course dir, else the Courseforge project path. Best-effort —
        any failure logs a warning and never alters ``final_status``.
        """
        try:
            from lib.validators._block_rubric_helpers import (
                block_quality_rubric_enabled,
            )

            if not block_quality_rubric_enabled():
                return None

            from lib.aggregators.block_quality_rollup import (
                BlockQualityRollupAggregator,
            )

            output_root: Optional[Path] = None
            archival = phase_outputs.get("libv2_archival") or {}
            course_dir = archival.get("course_dir")
            if course_dir:
                output_root = Path(course_dir)
            if output_root is None:
                project_path = self._resolve_courseforge_project_path(
                    phase_outputs
                )
                if project_path is not None:
                    output_root = Path(project_path)
            if output_root is None:
                logger.debug(
                    "block_quality_rollup: no libv2 course / project path "
                    "resolvable; skipping aggregator (run_id=%s)",
                    workflow_id,
                )
                return None

            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = BlockQualityRollupAggregator(
                phase_outputs=phase_outputs,
                course_code=course_code,
                run_id=workflow_id,
            )
            if not aggregator.blocks:
                logger.debug(
                    "block_quality_rollup: no block_quality_rubric gate "
                    "result found; skipping aggregator (run_id=%s)",
                    workflow_id,
                )
                return None
            output_path = output_root / "block_quality_rollup_report.json"
            aggregator.write(output_path)
            logger.info(
                "block_quality_rollup: wrote %s (run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "block_quality_rollup aggregator failed "
                "(non-fatal, run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    def _maybe_write_coverage_map(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Worker W3.E helper — write top-level coverage_map.json if possible.

        Resolution priority for the output root:

        1. ``phase_outputs.libv2_archival.course_dir`` — the canonical
           LibV2 course root. Output lands at
           ``<course_dir>/coverage_map.json``.
        2. Fallback to
           ``phase_outputs.trainforge_assessment.trainforge_dir`` /
           ``phase_outputs.training_synthesis.corpus_dir`` — emit at the
           Trainforge workspace root when LibV2 archival hasn't yet
           completed (e.g. partial textbook_to_course run that stops
           before ``libv2_archival``).

        Returns ``None`` when neither resolves (no objective / chunk /
        assessment / pair surfaces ran) or when the aggregator raises
        during build/write. Best-effort posture matches the courseforge
        + trainforge aggregators: failure logs a warning and never
        fails the workflow.
        """
        try:
            # Local import to keep workflow_runner import-time deps
            # unchanged for non-Trainforge runs.
            from lib.aggregators.coverage_map import CoverageMapAggregator

            archival = phase_outputs.get("libv2_archival") or {}
            course_dir_str = archival.get("course_dir")
            output_path: Optional[Path] = None
            libv2_course_path: Optional[Path] = None
            trainforge_dir: Optional[Path] = None

            if course_dir_str:
                libv2_course_path = Path(course_dir_str)
                output_path = libv2_course_path / "coverage_map.json"

            ta = phase_outputs.get("trainforge_assessment") or {}
            tdir_str = ta.get("trainforge_dir")
            if not tdir_str:
                ts = phase_outputs.get("training_synthesis") or {}
                tdir_str = ts.get("corpus_dir") or ts.get("trainforge_dir")
            if tdir_str:
                trainforge_dir = Path(tdir_str)

            if output_path is None and trainforge_dir is not None:
                output_path = trainforge_dir / "coverage_map.json"

            if output_path is None:
                logger.debug(
                    "coverage_map: no libv2_archival.course_dir / "
                    "training_synthesis.corpus_dir / trainforge_assessment."
                    "trainforge_dir resolvable; skipping aggregator "
                    "(run_id=%s)",
                    workflow_id,
                )
                return None

            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = CoverageMapAggregator(
                phase_outputs=phase_outputs,
                course_code=course_code,
                run_id=workflow_id,
                libv2_course_path=libv2_course_path,
                trainforge_dir=trainforge_dir,
            )
            aggregator.write(output_path)
            logger.info(
                "coverage_map: wrote %s (run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "coverage_map aggregator failed (non-fatal, run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    def _maybe_write_concept_coverage(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Worker W4.1 helper — write top-level concept_coverage.json if enabled.

        Gated OFF by default via ``ED4ALL_CONCEPT_COVERAGE``: when the
        flag is falsey the helper short-circuits BEFORE resolving any
        path or constructing the aggregator, so a default run is
        byte-identical (no file emitted).

        Output root resolution (concept-graph-driven — needs a LibV2
        course dir or a direct concept_graph_path):

        1. ``phase_outputs.libv2_archival.course_dir`` — the canonical
           LibV2 course root. Output lands at
           ``<course_dir>/concept_coverage.json``.
        2. ``phase_outputs.concept_extraction.concept_graph_path`` — the
           concept-graph path emitted by the concept_extraction phase
           (covers partial runs that stop before ``libv2_archival``);
           output lands next to the graph.

        Returns ``None`` when the flag is off, neither source resolves,
        or the aggregator raises. Best-effort — failure logs a warning
        and never fails the workflow.
        """
        try:
            from lib.aggregators.concept_coverage import (
                ConceptCoverageAggregator,
                resolve_concept_coverage,
            )

            if not resolve_concept_coverage():
                return None

            archival = phase_outputs.get("libv2_archival") or {}
            course_dir_str = archival.get("course_dir")
            libv2_course_path: Optional[Path] = None
            concept_graph_path: Optional[Path] = None
            output_path: Optional[Path] = None

            if course_dir_str:
                libv2_course_path = Path(course_dir_str)
                output_path = libv2_course_path / "concept_coverage.json"

            ce = phase_outputs.get("concept_extraction") or {}
            ce_graph_str = ce.get("concept_graph_path")
            if ce_graph_str:
                concept_graph_path = Path(ce_graph_str)
                if output_path is None:
                    output_path = (
                        concept_graph_path.parent / "concept_coverage.json"
                    )

            if output_path is None:
                logger.debug(
                    "concept_coverage: no libv2_archival.course_dir / "
                    "concept_extraction.concept_graph_path resolvable; "
                    "skipping aggregator (run_id=%s)",
                    workflow_id,
                )
                return None

            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = ConceptCoverageAggregator(
                course_code=course_code,
                run_id=workflow_id,
                libv2_course_path=libv2_course_path,
                concept_graph_path=concept_graph_path,
            )
            aggregator.write(output_path)
            logger.info(
                "concept_coverage: wrote %s (run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "concept_coverage aggregator failed (non-fatal, "
                "run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    def _maybe_write_intelligence_level(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Worker W4.6 helper — write intelligence_level_report.json if enabled.

        Gated OFF by default via ``ED4ALL_INTELLIGENCE_RUBRIC``: when the
        flag is falsey the helper short-circuits BEFORE resolving any
        path or constructing the aggregator, so a default run is
        byte-identical (no file emitted).

        Output root resolution:

        1. ``phase_outputs.libv2_archival.course_dir`` — canonical LibV2
           course root; output lands at
           ``<course_dir>/intelligence_level_report.json``.
        2. Fallback to
           ``phase_outputs.trainforge_assessment.trainforge_dir`` /
           ``phase_outputs.training_synthesis.corpus_dir`` — emit at the
           Trainforge workspace root when archival hasn't run.

        Returns ``None`` when the flag is off, neither source resolves,
        or the aggregator raises. Best-effort — failure logs a warning
        and never fails the workflow.
        """
        try:
            from lib.aggregators.intelligence_level import (
                IntelligenceLevelAggregator,
                resolve_intelligence_rubric,
            )

            if not resolve_intelligence_rubric():
                return None

            archival = phase_outputs.get("libv2_archival") or {}
            course_dir_str = archival.get("course_dir")
            libv2_course_path: Optional[Path] = None
            trainforge_dir: Optional[Path] = None
            output_path: Optional[Path] = None

            if course_dir_str:
                libv2_course_path = Path(course_dir_str)
                output_path = (
                    libv2_course_path / "intelligence_level_report.json"
                )

            ta = phase_outputs.get("trainforge_assessment") or {}
            tdir_str = ta.get("trainforge_dir")
            if not tdir_str:
                ts = phase_outputs.get("training_synthesis") or {}
                tdir_str = ts.get("corpus_dir") or ts.get("trainforge_dir")
            if tdir_str:
                trainforge_dir = Path(tdir_str)

            if output_path is None and trainforge_dir is not None:
                output_path = (
                    trainforge_dir / "intelligence_level_report.json"
                )

            if output_path is None:
                logger.debug(
                    "intelligence_level: no libv2_archival.course_dir / "
                    "trainforge_dir resolvable; skipping aggregator "
                    "(run_id=%s)",
                    workflow_id,
                )
                return None

            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = IntelligenceLevelAggregator(
                course_code=course_code,
                run_id=workflow_id,
                libv2_course_path=libv2_course_path,
                trainforge_dir=trainforge_dir,
            )
            aggregator.write(output_path)
            logger.info(
                "intelligence_level: wrote %s (run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "intelligence_level aggregator failed (non-fatal, "
                "run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    def _maybe_write_promotion_chain_report(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Worker W3.G helper — write top-level promotion_chain_report if possible.

        Resolution priority for the LibV2 course root (the canonical emit
        location):

        1. ``phase_outputs.libv2_archival.course_dir`` — post-archival
           canonical surface. Output lands at
           ``<course_dir>/courseforge_promotion_chain_report.json``.
        2. Fallback to
           ``phase_outputs.training_synthesis.corpus_dir`` /
           ``phase_outputs.trainforge_assessment.trainforge_dir`` — the
           Trainforge workspace root when LibV2 archival hasn't run.

        The aggregator itself is best-effort over individual arrow reads
        (each missing per-stage report becomes a missing-stage-report
        row), so this wrapper only fails closed when no LibV2 / Trainforge
        root can be resolved at all. Best-effort posture matches the
        courseforge / trainforge / coverage-map aggregators: failure
        logs a warning and never fails the workflow.
        """
        try:
            from lib.aggregators.promotion_chain_report import (
                PromotionChainAggregator,
            )

            archival = phase_outputs.get("libv2_archival") or {}
            course_dir_str = archival.get("course_dir")
            output_path: Optional[Path] = None
            course_path: Optional[Path] = None
            trainforge_dir: Optional[Path] = None

            if course_dir_str:
                course_path = Path(course_dir_str)
                output_path = (
                    course_path
                    / "courseforge_promotion_chain_report.json"
                )

            ta = phase_outputs.get("trainforge_assessment") or {}
            tdir_str = ta.get("trainforge_dir")
            if not tdir_str:
                ts = phase_outputs.get("training_synthesis") or {}
                tdir_str = ts.get("corpus_dir") or ts.get("trainforge_dir")
            if tdir_str:
                trainforge_dir = Path(tdir_str)

            if output_path is None and trainforge_dir is not None:
                output_path = (
                    trainforge_dir
                    / "courseforge_promotion_chain_report.json"
                )

            if output_path is None:
                logger.debug(
                    "promotion_chain_report: no libv2_archival.course_dir "
                    "/ training_synthesis.corpus_dir / trainforge_assessment."
                    "trainforge_dir resolvable; skipping aggregator "
                    "(run_id=%s)",
                    workflow_id,
                )
                return None

            project_path = self._resolve_courseforge_project_path(
                phase_outputs
            )

            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = PromotionChainAggregator(
                course_path=course_path,
                project_path=project_path,
                trainforge_dir=trainforge_dir,
                course_code=course_code,
                run_id=workflow_id,
                phase_outputs=phase_outputs,
            )
            aggregator.write(output_path)
            logger.info(
                "promotion_chain_report: wrote %s "
                "(run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "promotion_chain_report aggregator failed "
                "(non-fatal, run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    def _maybe_write_accessibility_conformance(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Roadmap T3 helper — write accessibility_conformance.json if possible.

        Output root resolution (mirrors the promotion-chain report):

        1. ``phase_outputs.libv2_archival.course_dir`` — canonical LibV2
           course root; output lands at
           ``<course_dir>/quality/accessibility_conformance.json``.
        2. Fallback to
           ``phase_outputs.training_synthesis.corpus_dir`` /
           ``phase_outputs.trainforge_assessment.trainforge_dir`` — the
           Trainforge workspace root when archival hasn't run; output lands
           at ``<trainforge_dir>/quality/accessibility_conformance.json``.

        The aggregator inverts the WCAG gate stream deterministically, so it
        always produces a full A+AA table (even a run with zero accessibility
        gates emits a table where every criterion is supports / not_evaluated).
        Best-effort — failure logs a warning and never alters ``final_status``.
        """
        try:
            from lib.aggregators.accessibility_conformance import (
                AccessibilityConformanceAggregator,
            )

            archival = phase_outputs.get("libv2_archival") or {}
            course_dir_str = archival.get("course_dir")
            output_path: Optional[Path] = None

            if course_dir_str:
                output_path = (
                    Path(course_dir_str)
                    / "quality"
                    / "accessibility_conformance.json"
                )

            if output_path is None:
                ta = phase_outputs.get("trainforge_assessment") or {}
                tdir_str = ta.get("trainforge_dir")
                if not tdir_str:
                    ts = phase_outputs.get("training_synthesis") or {}
                    tdir_str = ts.get("corpus_dir") or ts.get("trainforge_dir")
                if tdir_str:
                    output_path = (
                        Path(tdir_str)
                        / "quality"
                        / "accessibility_conformance.json"
                    )

            if output_path is None:
                logger.debug(
                    "accessibility_conformance: no libv2_archival.course_dir "
                    "/ training_synthesis.corpus_dir / trainforge_assessment."
                    "trainforge_dir resolvable; skipping aggregator "
                    "(run_id=%s)",
                    workflow_id,
                )
                return None

            project_path = self._resolve_courseforge_project_path(
                phase_outputs
            )
            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = AccessibilityConformanceAggregator(
                phase_outputs=phase_outputs,
                project_path=project_path,
                course_code=course_code,
                run_id=workflow_id,
            )
            aggregator.write(output_path)
            logger.info(
                "accessibility_conformance: wrote %s "
                "(run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "accessibility_conformance aggregator failed "
                "(non-fatal, run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    def _maybe_write_build_cost_report(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Path]:
        """Roadmap OP2 helper — write build_cost_report.json if possible.

        Output root resolution (mirrors the coverage-map / promotion-chain
        reports):

        1. ``phase_outputs.libv2_archival.course_dir`` — canonical LibV2
           course root; output lands at
           ``<course_dir>/build_cost_report.json``.
        2. Fallback to
           ``phase_outputs.training_synthesis.corpus_dir`` /
           ``phase_outputs.trainforge_assessment.trainforge_dir``.

        The aggregator reads its cost inputs from ``state/runs/<run_id>/``
        (checkpoints / vram_trajectory.jsonl / llm_usage.jsonl); the GPU +
        LLM sections are omitted when their source files are absent. Pure
        metering (no LLM decisions). Best-effort — failure logs a warning and
        never alters ``final_status``.
        """
        try:
            from lib.aggregators.build_cost import BuildCostAggregator

            archival = phase_outputs.get("libv2_archival") or {}
            course_dir_str = archival.get("course_dir")
            output_path: Optional[Path] = None

            if course_dir_str:
                output_path = Path(course_dir_str) / "build_cost_report.json"

            if output_path is None:
                ta = phase_outputs.get("trainforge_assessment") or {}
                tdir_str = ta.get("trainforge_dir")
                if not tdir_str:
                    ts = phase_outputs.get("training_synthesis") or {}
                    tdir_str = ts.get("corpus_dir") or ts.get("trainforge_dir")
                if tdir_str:
                    output_path = Path(tdir_str) / "build_cost_report.json"

            if output_path is None:
                logger.debug(
                    "build_cost: no libv2_archival.course_dir / "
                    "training_synthesis.corpus_dir / trainforge_assessment."
                    "trainforge_dir resolvable; skipping aggregator "
                    "(run_id=%s)",
                    workflow_id,
                )
                return None

            course_code = (workflow_params or {}).get("course_name") or ""
            aggregator = BuildCostAggregator(
                course_code=course_code,
                run_id=workflow_id,
            )
            aggregator.write(output_path)
            logger.info(
                "build_cost: wrote %s (run_id=%s, course_code=%s)",
                output_path, workflow_id, course_code,
            )
            return output_path
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "build_cost aggregator failed (non-fatal, run_id=%s): %s",
                workflow_id, exc,
            )
            return None

    def _maybe_write_edge_consensus_reports(
        self,
        *,
        workflow_id: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> List[Path]:
        """NVIDIA-KG item 3 helper — stamp edge consensus on landed graphs.

        Mirror of the authoring-time wiring in
        ``MCP/tools/pipeline_tools.py::_run_concept_extraction`` for
        semantic graphs that exist on disk at workflow end but were NOT
        authored by that phase (process_course / IMSCC path, pre-fix
        corpora). LibV2 courses carry the semantic graph under BOTH
        layouts — ``graph/concept_graph_semantic.json`` (process_course
        route, e.g. the RDF/SHACL calibration corpus) and
        ``concept_graph/concept_graph_semantic.json``
        (``_run_concept_extraction`` route) — and some courses carry
        both, so every existing candidate is handled.

        Candidate-graph resolution priority:

        1. ``phase_outputs.libv2_archival.course_dir`` — the canonical
           LibV2 course root; probe ``graph/`` + ``concept_graph/``
           subdirs.
        2. ``phase_outputs.concept_extraction.concept_graph_path`` —
           direct path emitted by the concept_extraction phase (covers
           partial runs that stop before ``libv2_archival``).

        Per-graph behaviour (each graph fails soft independently):

        - Un-stamped (any dict edge missing a non-None ``edge_status``)
          → ``EdgeConsensusAggregator.apply_to_graph`` stamps in place
          (idempotent + deterministic per its contract; cross-rule
          matrix only, no LLM, NLI off unless ``TRAINFORGE_EDGE_NLI``),
          the graph is re-serialized with the pipeline convention
          (``indent=2, ensure_ascii=False``), and the sibling
          ``edge_consensus_report.json`` is (re)written so report and
          graph agree.
        - Fully stamped + sibling report present → skipped untouched
          (no byte/mtime churn; the report's ``generated_at`` would
          otherwise drift on every re-run). This is the idempotency
          contract with the authoring-time wiring.
        - Fully stamped but report missing → only the sibling report is
          written; the graph file is not rewritten.

        Returns the list of sibling-report paths that now exist for the
        handled graphs (written this run or pre-existing-and-skipped).
        Best-effort posture matches the other post-loop aggregators:
        any failure logs a warning and never fails the workflow.
        """
        report_paths: List[Path] = []
        try:
            # Local import to keep the workflow_runner import-time
            # dependency surface unchanged (matches the sibling
            # aggregator helpers above).
            from lib.aggregators.edge_consensus import (
                EdgeConsensusAggregator,
                load_chunk_text_lookup,
            )

            # ----------------------------------------------------------
            # Resolve candidate semantic-graph paths (both layouts).
            # ----------------------------------------------------------
            candidates: List[Path] = []
            seen: set = set()

            def _add_candidate(path: Path) -> None:
                try:
                    key = path.resolve()
                except OSError:
                    key = path
                if key in seen:
                    return
                seen.add(key)
                if path.is_file():
                    candidates.append(path)

            archival = phase_outputs.get("libv2_archival") or {}
            course_dir_str = archival.get("course_dir")
            if course_dir_str:
                for subdir in ("graph", "concept_graph"):
                    _add_candidate(
                        Path(course_dir_str)
                        / subdir
                        / "concept_graph_semantic.json"
                    )

            ce = phase_outputs.get("concept_extraction") or {}
            ce_graph_str = ce.get("concept_graph_path")
            if ce_graph_str:
                _add_candidate(Path(ce_graph_str))

            if not candidates:
                logger.debug(
                    "edge_consensus: no concept_graph_semantic.json "
                    "resolvable from libv2_archival.course_dir / "
                    "concept_extraction.concept_graph_path; skipping "
                    "aggregator (run_id=%s)",
                    workflow_id,
                )
                return report_paths

            course_code = (workflow_params or {}).get("course_name") or ""
            course_slug = ce.get("course_slug") or course_code

            for graph_path in candidates:
                # Per-graph fail-soft: one corrupt graph must not stop
                # the sibling layout from being stamped.
                try:
                    try:
                        graph = json.loads(
                            graph_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "edge_consensus: failed to read/parse %s "
                            "(non-fatal, run_id=%s): %s",
                            graph_path, workflow_id, exc,
                        )
                        continue
                    if not isinstance(graph, dict):
                        continue
                    edges = graph.get("edges")
                    if not isinstance(edges, list) or not edges:
                        logger.debug(
                            "edge_consensus: %s carries no edges; "
                            "skipping (run_id=%s)",
                            graph_path, workflow_id,
                        )
                        continue

                    dict_edges = [e for e in edges if isinstance(e, dict)]
                    fully_stamped = bool(dict_edges) and all(
                        e.get("edge_status") is not None for e in dict_edges
                    )
                    report_path = (
                        graph_path.parent / "edge_consensus_report.json"
                    )

                    if fully_stamped and report_path.is_file():
                        # Idempotency contract: already stamped by the
                        # authoring-time wiring (or a prior run) AND the
                        # sibling report exists — leave both untouched.
                        logger.debug(
                            "edge_consensus: %s already stamped with "
                            "sibling report; no-op (run_id=%s)",
                            graph_path, workflow_id,
                        )
                        report_paths.append(report_path)
                        continue

                    # TRAINFORGE_EDGE_NLI: build the chunk-text lookup from the
                    # course tree (graph lives at <course_dir>/{graph,
                    # concept_graph}/...). None when no chunkset is found / the
                    # flag is off → the NLI extension no-ops.
                    nli_chunk_lookup = load_chunk_text_lookup(
                        graph_path.parent.parent
                    )
                    aggregator = EdgeConsensusAggregator(
                        semantic_graph_path=graph_path,
                        course_slug=course_slug,
                        run_id=workflow_id,
                        chunk_text_lookup=nli_chunk_lookup,
                    )

                    if not fully_stamped:
                        # apply_to_graph is idempotent + deterministic
                        # (lib/aggregators/edge_consensus.py contract;
                        # pinned by test #6 in its test suite), so
                        # re-stamping a partially-stamped graph is safe.
                        aggregator.apply_to_graph(graph)
                        graph_path.write_text(
                            json.dumps(graph, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )

                    # write() re-reads the (now stamped) on-disk graph so
                    # the report and the graph agree — same ordering as
                    # the authoring-time wiring.
                    written = aggregator.write(report_path)
                    if written is not None:
                        report_paths.append(written)
                        logger.info(
                            "edge_consensus: %s %s; wrote %s "
                            "(run_id=%s, course_slug=%s)",
                            "stamped" if not fully_stamped
                            else "already stamped",
                            graph_path, written, workflow_id, course_slug,
                        )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "edge_consensus aggregator failed for %s "
                        "(non-fatal, run_id=%s): %s",
                        graph_path, workflow_id, exc,
                    )
            return report_paths
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "edge_consensus aggregator failed (non-fatal, run_id=%s): %s",
                workflow_id, exc,
            )
            return report_paths

    def _route_params(
        self,
        phase_name: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """
        Build task params for a phase by resolving the routing table.

        Args:
            phase_name: Name of the phase to build params for
            workflow_params: Original workflow creation params
            phase_outputs: Accumulated outputs from prior phases

        Returns:
            Dict of resolved parameter values
        """
        routing = _get_phase_param_routing(phase_name)
        params = {}

        for param_name, source_spec in routing.items():
            source_type = source_spec[0]

            if source_type == "workflow_params":
                key = source_spec[1]
                value = workflow_params.get(key)
                if value is not None:
                    # Handle list values that need comma-joining for tool params
                    if isinstance(value, list):
                        value = ",".join(str(v) for v in value)
                    params[param_name] = value

            elif source_type == "phase_outputs":
                phase_key = source_spec[1]
                output_key = source_spec[2]
                phase_data = phase_outputs.get(phase_key, {})
                value = phase_data.get(output_key)
                if value is not None:
                    if isinstance(value, list):
                        value = ",".join(str(v) for v in value)
                    params[param_name] = value

            elif source_type == "literal":
                params[param_name] = source_spec[1]

        return params

    def _resume_reusable_conversion_stems(
        self,
        phase_outputs: Optional[Dict[str, Any]],
        pdf_paths: List[Any],
    ) -> set:
        """PDF stems whose prior conversion is COMPLETE — for P6 resume reuse.

        Graceful-stop "checkpoint on command": ``dart_conversion`` is one
        SemantiK task per PDF (``max_concurrent: 1``, no per-unit sidecar — a
        chapter is either fully converted or not), so a stop mid-phase leaves
        the finished chapters' artifacts on disk and the in-flight chapter
        un-converted. On ``--resume`` this auto-enables per-PDF conversion reuse
        for the finished chapters so the run does not re-convert (hours of
        7B/OCR work) what already succeeded.

        RESUME-GATED (never fires on a fresh run): the ``dart_conversion`` phase
        is (re)created here only when it did NOT complete — the completed-phase
        skip guard skips a ``_completed`` phase — and a prior
        ``phase_outputs['dart_conversion']`` that is present but NOT
        ``_completed`` is the paused/partial resume signal. A fresh run has no
        such entry → empty set, so a stale ``{stem}_accessible.html`` left in
        the output dir by an unrelated prior run is never wrongly reused.

        COMPLETENESS SIGNAL = BOTH ``{stem}_accessible.html`` AND
        ``{stem}_accessible.quality.json`` in the conversion output dir (the
        ``DART_PATH/output`` the ``extract_and_convert_pdf`` seam writes to).
        The HTML is written first and the quality sidecar best-effort after
        (``_run_semantik_v2_conversion``), so requiring BOTH rejects a torn
        write (HTML present, sidecar missing) → that PDF is re-converted. The
        downstream ``_run_semantik_v2_conversion`` reuse arm then reads the
        prior HTML + sidecars back (cascade skipped); a PDF NOT in this set gets
        no ``reuse_conversion`` flag and is converted normally.
        """
        if not isinstance(phase_outputs, dict):
            return set()
        entry = phase_outputs.get("dart_conversion")
        if not isinstance(entry, dict) or entry.get("_completed"):
            return set()
        conv_dir = DART_PATH / "output"
        reusable: set = set()
        for pdf_path in pdf_paths:
            stem = Path(str(pdf_path)).stem
            if not stem:
                continue
            html = conv_dir / f"{stem}_accessible.html"
            quality = conv_dir / f"{stem}_accessible.quality.json"
            try:
                if html.is_file() and quality.is_file():
                    reusable.add(stem)
            except OSError:
                continue
        return reusable

    def _create_phase_tasks(
        self,
        workflow_id: str,
        phase: WorkflowPhase,
        routed_params: Dict[str, Any],
        workflow_params: Optional[Dict[str, Any]] = None,
        phase_outputs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create task dicts for a phase.

        Handles special cases:
        - dart_conversion: one task per PDF file
        - content_generation with batch_by=week: one task per week
        - Default: one task per agent in the phase

        Args:
            workflow_id: Parent workflow ID
            phase: Phase configuration
            routed_params: Parameters resolved from routing table
            workflow_params: Original workflow creation params

        Returns:
            List of task dicts ready for execute_phase()
        """
        tasks = []
        timestamp = datetime.now().strftime("%H%M%S")
        workflow_params = workflow_params or {}

        # Special case: dart_conversion creates one task per PDF
        if phase.name == "dart_conversion":
            pdf_paths = workflow_params.get("pdf_paths", [])
            if isinstance(pdf_paths, str):
                pdf_paths = [p.strip() for p in pdf_paths.split(",")]
            # Graceful-stop (P6) auto-reuse on paused/partial resume: the PDF
            # stems whose prior conversion is COMPLETE on disk, to be skipped
            # (reused) on this resume leg. Empty on a fresh run (self-gated) so
            # a stale artifact is never wrongly reused.
            auto_reuse_stems = self._resume_reusable_conversion_stems(
                phase_outputs, pdf_paths
            )
            for i, pdf_path in enumerate(pdf_paths):
                task_id = f"T-{phase.name}-{i}-{timestamp}"
                # SemantiK migration P3c — the registry tool
                # ``extract_and_convert_pdf`` flips the LIVE PDF→HTML
                # conversion to the SemantiK v2 seam ONLY when it sees
                # ``phase == "dart_conversion"`` (this textbook/course path).
                # Forward the seam params so the flipped
                # path threads canonical course code, figures dir, and the
                # ``--reuse-conversion`` pinned-HTML flag (§3.3a item 2).
                task_params = {
                    "pdf_path": pdf_path,
                    "course_code": workflow_params.get("course_name", ""),
                    "phase": phase.name,
                    "canonical_course_code": workflow_params.get(
                        "canonical_course_code"
                    ),
                }
                # ``reuse_conversion`` is set either GLOBALLY (--reuse-conversion
                # / ED4ALL_REUSE_CONVERSION) or PER-PDF on a paused/partial
                # resume for PDFs whose prior conversion completed (P6). Same
                # per-task flag / same ``_run_semantik_v2_conversion`` reuse
                # mechanism — no parallel plumbing.
                pdf_stem = Path(str(pdf_path)).stem
                if (
                    workflow_params.get("reuse_conversion")
                    or pdf_stem in auto_reuse_stems
                ):
                    task_params["reuse_conversion"] = True
                task = {
                    "id": task_id,
                    "agent_type": phase.agents[0],
                    "phase": phase.name,
                    "status": "PENDING",
                    "params": task_params,
                    "created_at": datetime.now().isoformat(),
                    "dependencies": [],
                }
                tasks.append(task)
            return tasks

        # Special case: batch_by week creates one task per week
        if phase.batch_by == "week":
            duration = workflow_params.get("duration_weeks", 12)
            for week in range(1, duration + 1):
                task_id = f"T-{phase.name}-w{week}-{timestamp}"
                task = {
                    "id": task_id,
                    "agent_type": phase.agents[0],
                    "phase": phase.name,
                    "status": "PENDING",
                    "params": {
                        **routed_params,
                        "week_range": f"{week}-{week}",
                    },
                    "created_at": datetime.now().isoformat(),
                    "dependencies": [],
                }
                tasks.append(task)
            return tasks

        # Default: one task per agent
        for agent_name in phase.agents:
            task_id = f"T-{phase.name}-{agent_name}-{timestamp}"
            task = {
                "id": task_id,
                "agent_type": agent_name,
                "phase": phase.name,
                "status": "PENDING",
                "params": routed_params.copy(),
                "created_at": datetime.now().isoformat(),
                "dependencies": [],
            }
            tasks.append(task)

        # Phase 4 Subtask 1 — synthesize a single virtual task for
        # phases that declare ``agents: []`` but ARE registered in
        # ``_PHASE_TOOL_MAPPING`` (e.g. ``inter_tier_validation``,
        # ``post_rewrite_validation``, plus the two-pass
        # outline/rewrite phases when wired without an explicit
        # agent). Without this fallback, the per-agent loop above
        # yields zero tasks, ``execute_phase`` runs the validation
        # gate chain only, and the dedicated phase-handler
        # (``run_inter_tier_validation`` / ``run_post_rewrite_validation``
        # / etc.) never lands its blocks-validated-and-persist
        # work to disk. The executor's ``_PHASE_TOOL_MAPPING.get``
        # path keys off ``phase.name`` so the placeholder
        # ``agent_type="phase-handler"`` is intentional — the agent
        # name is irrelevant on this routing path.
        if not tasks and _PHASE_TOOL_MAPPING.get(phase.name):
            task_id = f"T-{phase.name}-phase-handler-{timestamp}"
            tasks.append({
                "id": task_id,
                "agent_type": "phase-handler",
                "phase": phase.name,
                "status": "PENDING",
                "params": routed_params.copy(),
                "created_at": datetime.now().isoformat(),
                "dependencies": [],
            })

        return tasks

    def _extract_phase_outputs(
        self,
        phase_name: str,
        results: Dict[str, ExecutionResult],
    ) -> Dict[str, Any]:
        """
        Extract key output values from phase results for downstream routing.

        Args:
            phase_name: Name of the completed phase
            results: Dict of task_id -> ExecutionResult

        Returns:
            Dict of extracted output values
        """
        output_keys = _get_phase_output_keys(phase_name)
        extracted = {}

        for result in results.values():
            if result.status != "COMPLETE":
                continue

            result_data = result.result
            if not isinstance(result_data, dict):
                continue

            for key in output_keys:
                if key in result_data and key not in extracted:
                    extracted[key] = result_data[key]

        # Special handling: collect multiple output_paths into output_paths list
        if phase_name == "dart_conversion":
            paths = []
            for result in results.values():
                if result.status == "COMPLETE" and isinstance(result.result, dict):
                    path = (
                        result.result.get("output_path")
                        or result.result.get("html_path")
                    )
                    if path:
                        paths.append(path)
            if paths:
                joined = ",".join(paths)
                extracted["output_paths"] = joined
                # Wave 32 Deliverable B: alias as html_paths (router
                # canonical key) so DartMarkersValidator gate builder
                # picks it up without a router change.
                extracted["html_paths"] = joined
                # And surface a single representative html_path for
                # validators that only accept the scalar form.
                extracted.setdefault("html_path", paths[0])

        return extracted

    def _should_skip_phase(
        self, phase: WorkflowPhase, workflow_params: Dict[str, Any]
    ) -> bool:
        """Check if an optional phase should be skipped based on workflow params.

        Wave 74 Session 3: dart_conversion's --skip-dart path is
        handled upstream by pre-populating ``phase_outputs`` in
        ``run_workflow`` before the loop runs. The already-completed
        guard then skips execution naturally, preserving the
        synthesised output dict (this method would have overwritten it
        with a bare ``{"_skipped": True, "_completed": True}``).

        Phase 3 Subtask 1: phases may carry an ``enabled_when_env``
        predicate (``"VAR=value"`` or ``"VAR!=value"``); when present,
        the predicate is evaluated against the live environment and
        the phase skips when unsatisfied. This gate runs BEFORE the
        legacy optional-phase logic so a non-optional phase can still
        skip via the env predicate (e.g. the legacy
        ``content_generation`` phase carries
        ``enabled_when_env: "COURSEFORGE_TWO_PASS!=true"`` to disable
        itself when the new two-pass router is engaged).

        Phase 5 Subtask 4: when ``workflow_params['courseforge_stage']``
        is set (CLI plumbed by ``cli/commands/run.py`` for the four
        Phase 5 ``courseforge-*`` subcommands), phases NOT in the
        active-phase whitelist for that stage are skipped. Whitelist
        per stage:

        * ``courseforge_outline``: ``[content_generation_outline]``.
          Validate + rewrite + post_rewrite_validation skip.
        * ``courseforge_validate``: ``[inter_tier_validation,
          post_rewrite_validation]``. The two read-only validator
          phases run; outline + rewrite skip.
        * ``courseforge_rewrite``: ``[content_generation_rewrite,
          post_rewrite_validation]``. Outline + inter_tier_validation
          skip (the rewrite tier consumes the synthesizer-reconstructed
          ``inter_tier_validation`` output from disk via
          ``_synthesize_outline_output``).
        * ``courseforge`` / ``full`` (or absent — falls through to
          existing behaviour): all four two-pass phases run; nothing
          skipped from the courseforge whitelist.

        Upstream phases (dart_conversion, staging, chunking,
        objective_extraction, source_mapping, concept_extraction,
        course_planning) are also skipped because the runner
        pre-populates them via ``_synthesize_outline_output`` before
        the phase loop runs (their ``_completed=True`` guard at
        ``run_workflow:897`` already short-circuits them; this gate
        catches the case where the synthesizer didn't fire — e.g.
        operator passed only ``--course-name`` without setting up a
        prior project export). Downstream phases (packaging,
        imscc_chunking, trainforge_assessment, training_synthesis,
        libv2_archival, finalization) skip because the Phase 5 stage
        subcommands are scoped to the Courseforge two-pass surface
        only — operators who want post-rewrite phases should run the
        full ``ed4all run textbook-to-course`` pipeline.
        """
        predicate = getattr(phase, "enabled_when_env", None)
        if predicate:
            if not self._eval_enabled_when_env(predicate):
                # Wave1-I6 (Finding 9): make env-predicate skips
                # visible in the operator log. Without this, an
                # ``enabled_when_env`` phase silently no-ops when the
                # gating env var is unset, which is indistinguishable
                # from a crashed / mis-routed phase at triage time.
                if "!=" in predicate:
                    var_name = predicate.partition("!=")[0].strip()
                elif "=" in predicate:
                    var_name = predicate.partition("=")[0].strip()
                else:
                    var_name = ""
                actual_value = os.environ.get(var_name, "") if var_name else ""
                resolved = actual_value if actual_value else "unset"
                logger.info(
                    f"Skipping phase {phase.name}: enabled_when_env "
                    f"predicate {predicate} unsatisfied "
                    f"(resolved value: {resolved})"
                )
                return True

        # Phase 5 Subtask 4: courseforge_stage whitelist gate. Runs
        # BEFORE the optional-phase early-return below so non-optional
        # phases (e.g. dart_conversion, packaging) can still be
        # skipped when the operator scoped a stage subcommand to a
        # subset of the Courseforge surface.
        stage = workflow_params.get("courseforge_stage")
        if stage and self._should_skip_for_courseforge_stage(phase.name, stage):
            return True

        if not getattr(phase, "optional", False):
            return False

        # Skip the assessment phases if generate_assessments is False.
        # ``assessment_synthesis`` (W10 pre-packaging QTI/discussion/assignment
        # surface) AND ``trainforge_assessment`` (post-package assessment gen)
        # are both gated by --no-assessments / generate_assessments=false. The
        # W10 phase was added without updating this skip (it only named
        # trainforge_assessment), so a --no-assessments run still dispatched
        # assessment_synthesis and failed (CLAUDE.md documents it as
        # "skipped via generate_assessments=false").
        if phase.name in ("trainforge_assessment", "assessment_synthesis"):
            return not workflow_params.get("generate_assessments", True)

        # Skip training_synthesis if --skip-training was passed. This is the
        # canonical A/B audit posture: Qwen generates assessments (phase 14)
        # but Claude must NOT generate training pairs (phase 16) because that
        # would route synthesis through a non-license-clean provider. See
        # plans/algebra-textbook-kg-test-2026-05.md and docs/LICENSING.md.
        if phase.name == "training_synthesis":
            return bool(workflow_params.get("skip_training", False))

        # Marketable-v1 A2: vector_indexing runs by default but skips cleanly
        # (with a logged reason) when the embedding stack is unavailable UNLESS
        # the operator opts into strict mode via TRAINFORGE_REQUIRE_EMBEDDINGS.
        # This mirrors the embedding-stack graceful-degrade convention used by
        # the statistical-tier validators (lib/embedding/sentence_embedder.py::
        # is_strict_mode + the EMBEDDING_DEPS_MISSING warning contract): a slim
        # install without the [embedding] extra should not fail an otherwise
        # green textbook_to_course run. In strict mode the phase is NOT skipped
        # and run_vector_indexing fails closed if the backend is broken.
        if phase.name == "vector_indexing":
            return self._should_skip_vector_indexing()

        return False

    @staticmethod
    def _should_skip_vector_indexing() -> bool:
        """Skip vector_indexing when the embedding stack is unavailable.

        Returns True (skip) only when BOTH (a) strict mode is OFF
        (``TRAINFORGE_REQUIRE_EMBEDDINGS`` unset/falsey) and (b) the embedding /
        vector-index dependencies are not importable. In strict mode we never
        skip — the phase runs and ``run_vector_indexing`` fails closed on a
        broken backend (anti-silent-degradation contract).
        """
        try:
            from lib.embedding.sentence_embedder import is_strict_mode
        except Exception:  # noqa: BLE001 — embedding pkg itself unimportable
            is_strict_mode = lambda: False  # noqa: E731

        if is_strict_mode():
            return False

        # Cheap import-spec probe — never imports the heavy ML stack. The
        # provider/vector-index modules import-guard their heavy deps lazily, so
        # presence of the spec is the slim/full-install signal.
        import importlib.util

        for mod in ("lib.embedding.providers", "LibV2.tools.libv2.vector_index"):
            if importlib.util.find_spec(mod) is None:
                logger.info(
                    "Skipping optional phase vector_indexing: embedding "
                    "dependencies unavailable (%s not importable) and "
                    "TRAINFORGE_REQUIRE_EMBEDDINGS is not set. Install the "
                    "[embedding] extra or set TRAINFORGE_REQUIRE_EMBEDDINGS=true "
                    "to make indexing mandatory.",
                    mod,
                )
                return True
        return False

    # Phase 5 Subtask 4: per-stage active-phase whitelist. Source of
    # truth for the four ``courseforge-*`` subcommand handlers in
    # ``cli/commands/run.py``. Stage names accept both hyphenated
    # (``courseforge-rewrite``) and underscored (``courseforge_rewrite``)
    # spellings — ``run.py::_normalize_workflow`` already collapses
    # hyphens to underscores before passing the stage through, but
    # we accept both here for defence-in-depth.
    _COURSEFORGE_STAGE_ACTIVE_PHASES: Dict[str, frozenset] = {
        "courseforge_outline": frozenset({"content_generation_outline"}),
        "courseforge_validate": frozenset({
            "inter_tier_validation",
            "post_rewrite_validation",
        }),
        "courseforge_rewrite": frozenset({
            "content_generation_rewrite",
            "post_rewrite_validation",
        }),
        "courseforge": frozenset({
            "content_generation_outline",
            "inter_tier_validation",
            "content_generation_rewrite",
            "post_rewrite_validation",
        }),
    }

    @classmethod
    def _resolve_courseforge_stage_active_phases(
        cls, stage: str
    ) -> Optional[frozenset]:
        """Resolve a courseforge_stage name to its active-phase whitelist.

        Returns ``None`` when ``stage`` is unrecognised so the caller
        treats that as "no whitelist applied" and falls through to
        normal phase-loop semantics.
        """
        if not stage:
            return None
        normalized = stage.replace("-", "_").strip().lower()
        return cls._COURSEFORGE_STAGE_ACTIVE_PHASES.get(normalized)

    def _should_skip_for_courseforge_stage(
        self, phase_name: str, stage: str
    ) -> bool:
        """Return True if ``phase_name`` is NOT in the stage whitelist.

        Phase 5 Subtask 4: phases outside the four-phase Courseforge
        two-pass surface (``content_generation_outline``,
        ``inter_tier_validation``, ``content_generation_rewrite``,
        ``post_rewrite_validation``) are ALSO skipped when a stage is
        active because Phase 5 stage subcommands are scoped to the
        Courseforge surface only — pre-Courseforge phases pre-populate
        via ``_synthesize_outline_output``, post-Courseforge phases
        belong to the full ``textbook_to_course`` workflow.
        """
        active = self._resolve_courseforge_stage_active_phases(stage)
        if active is None:
            # Unknown stage — don't skip on behalf of a typo.
            return False
        # Phases inside the two-pass surface but outside the stage's
        # whitelist => skip.
        two_pass_surface = self._COURSEFORGE_STAGE_ACTIVE_PHASES["courseforge"]
        if phase_name in two_pass_surface:
            return phase_name not in active
        # Phases outside the two-pass surface entirely — pre-Courseforge
        # (synthesized via _synthesize_outline_output) and
        # post-Courseforge (out-of-scope for stage subcommands) — skip.
        return True

    @staticmethod
    def _eval_enabled_when_env(predicate: str) -> bool:
        """Evaluate an ``enabled_when_env`` predicate against ``os.environ``.

        Grammar (Phase 3 Subtask 1):
            "<NAME>=<value>"   -> True when ``os.environ[NAME] == value`` (case-insensitive)
            "<NAME>!=<value>"  -> True when ``os.environ[NAME] != value`` (case-insensitive)

        The literal ``true`` matches any of ``1`` / ``true`` / ``yes`` /
        ``on`` (case-insensitive), mirroring
        ``Courseforge/scripts/blocks.py::_EMIT_BLOCKS_TRUTHY`` at ``:40``
        so the two-pass-router gate is consistent with the Phase 2
        emit-blocks gate.

        Malformed predicates (no operator, empty NAME, etc.) return
        ``True`` so a typo doesn't silently skip a phase — the
        predicate is treated as "enabled by default" and surfaces the
        bug at YAML-load review time instead.
        """
        if not predicate or not isinstance(predicate, str):
            return True

        truthy = {"1", "true", "yes", "on"}

        # Order matters: check ``!=`` before ``=`` so the longer
        # operator wins.
        if "!=" in predicate:
            name, _, value = predicate.partition("!=")
            negate = True
        elif "=" in predicate:
            name, _, value = predicate.partition("=")
            negate = False
        else:
            return True

        name = name.strip()
        value = value.strip()
        if not name:
            return True

        env_value = os.environ.get(name, "").strip().lower()
        target = value.lower()

        if target == "true":
            matched = env_value in truthy
        else:
            matched = env_value == target

        return (not matched) if negate else matched

    def _synthesize_dart_skip_output(
        self, workflow_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build a dart_conversion phase_output from existing DART HTMLs.

        Walks ``workflow_params['dart_output_dir']`` for
        ``*_accessible.html`` files and returns a dict mirroring what
        ``_extract_phase_outputs`` would have produced on a live run:
        ``output_path``, ``output_paths``, ``html_path``, ``html_paths``,
        plus the ``_completed``/``_skipped``/``_gates_passed`` markers
        the phase loop expects.

        When the corpus params include explicit ``pdf_paths``, we emit
        one entry per PDF in corpus order so downstream staging's
        ``{stem}_accessible.html`` lookup matches the PDF ordering. If a
        PDF has no matching HTML we skip it silently — the CLI already
        warned at --skip-dart validation time.
        """
        from pathlib import Path as _Path

        dart_dir_str = workflow_params.get("dart_output_dir")
        if dart_dir_str:
            dart_dir = _Path(dart_dir_str)
            if not dart_dir.is_absolute():
                dart_dir = (PROJECT_ROOT / dart_dir_str).resolve()
        else:
            # No explicit override: use the (ED4ALL_HOME-aware) default DART
            # output dir so a relocated deployment finds its staged HTML.
            dart_dir = dart_output_dir()
        if not dart_dir.is_dir():
            logger.error(
                "skip_dart set but dart_output_dir is not a directory: %s",
                dart_dir,
            )
            return None

        # Order htmls by corpus PDF order when available; fall back to
        # a stable sort over the directory listing.
        pdf_paths = workflow_params.get("pdf_paths") or []
        if isinstance(pdf_paths, str):
            pdf_paths = [p.strip() for p in pdf_paths.split(",") if p.strip()]
        ordered_htmls: List[_Path] = []
        if pdf_paths:
            for pdf in pdf_paths:
                stem = _Path(pdf).stem
                candidate = dart_dir / f"{stem}_accessible.html"
                if candidate.exists():
                    ordered_htmls.append(candidate)
        if not ordered_htmls:
            ordered_htmls = sorted(dart_dir.glob("*_accessible.html"))

        if not ordered_htmls:
            logger.error(
                "skip_dart set but no ``*_accessible.html`` files found in %s",
                dart_dir,
            )
            return None

        path_strs = [str(p) for p in ordered_htmls]
        joined = ",".join(path_strs)
        logger.info(
            "skip_dart: synthesised dart_conversion phase_output "
            "from %d HTML(s) in %s",
            len(path_strs),
            dart_dir,
        )
        return {
            "output_path": path_strs[0],
            "output_paths": joined,
            "html_path": path_strs[0],
            "html_paths": joined,
            "success": True,
            "html_length": sum(
                (p.stat().st_size if p.exists() else 0) for p in ordered_htmls
            ),
            "_completed": True,
            "_skipped": True,
            "_gates_passed": True,
            "_skip_reason": "skip_dart=True; reused existing DART HTMLs",
        }

    def _synthesize_course_planning_reuse_output(
        self,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Dict[str, Any]]:
        """Build a ``course_planning`` phase_output from a reused LO file.

        Wave 80 Worker A. Loads the user-supplied objectives JSON
        (Courseforge synthesized form OR Wave 75 LibV2 archive form),
        normalizes to the Courseforge form expected by downstream
        consumers (content-generator, Trainforge CourseProcessor),
        cross-validates LO ID hierarchy / format / uniqueness, and
        writes the result to
        ``{project_path}/01_learning_objectives/synthesized_objectives.json``.

        Returns a dict mirroring what ``_extract_phase_outputs`` would
        have produced on a live run:

        * ``project_id`` — from the upstream ``objective_extraction``
          phase output (pre-conditions: that phase completed).
        * ``synthesized_objectives_path`` — absolute path to the file
          written into the project's
          ``01_learning_objectives/synthesized_objectives.json``.
        * ``objective_ids`` — comma-joined LO IDs (TO-NN + CO-NN).
        * ``terminal_count`` / ``chapter_count``.
        * ``_completed`` / ``_skipped`` / ``_gates_passed`` markers.

        Returns ``None`` (and logs at error level) when:

        * The reuse file is missing/unreadable/malformed (CLI already
          validates at parse time, but a race or manual workflow-state
          edit could still trip this).
        * The upstream ``objective_extraction`` phase did NOT produce a
          ``project_path`` / ``project_id`` we can resolve. Without a
          project to write into, the content-generator cannot pick up
          the objectives via ``project_config.json``.
        * Cross-validation fails (orphan parent_terminal references,
          malformed IDs, or duplicates).
        """
        from pathlib import Path as _Path

        reuse_path_str = workflow_params.get("reuse_objectives_path")
        if not reuse_path_str:
            return None

        reuse_path = _Path(reuse_path_str)
        if not reuse_path.is_file():
            logger.error(
                "reuse_objectives: file not found: %s", reuse_path,
            )
            return None
        try:
            raw = reuse_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError) as e:
            logger.error(
                "reuse_objectives: failed to parse %s: %s", reuse_path, e,
            )
            return None
        if not isinstance(data, dict):
            logger.error(
                "reuse_objectives: top-level JSON must be an object; "
                "got %s",
                type(data).__name__,
            )
            return None

        # Normalize into Courseforge synthesized form. Accept either
        # shape on input.
        normalized = _normalize_to_courseforge_form(data)
        if normalized is None:
            logger.error(
                "reuse_objectives: file does not match a recognised "
                "shape (Courseforge synthesized OR LibV2 archive). "
                "path=%s",
                reuse_path,
            )
            return None

        terminal: List[Dict[str, Any]] = normalized["terminal_objectives"]
        chapter_groups: List[Dict[str, Any]] = normalized["chapter_objectives"]

        if not terminal:
            logger.error(
                "reuse_objectives: zero terminal objectives in %s",
                reuse_path,
            )
            return None

        # Cross-validation. Flatten chapter groups for ID checks.
        chapter_flat: List[Dict[str, Any]] = []
        for group in chapter_groups:
            inner = group.get("objectives") or []
            for obj in inner:
                if isinstance(obj, dict):
                    chapter_flat.append(obj)

        validation_err = _validate_reused_lo_coherence(terminal, chapter_flat)
        if validation_err:
            logger.error(
                "reuse_objectives: cross-validation failed: %s",
                validation_err,
            )
            return None

        # Optional warning: compare against source_module_map if available.
        source_map_data = phase_outputs.get("source_mapping") or {}
        source_map_path = source_map_data.get("source_module_map_path")
        if source_map_path:
            _warn_on_source_map_mismatch(
                source_map_path, terminal, chapter_flat,
            )

        # Resolve project path / id from upstream objective_extraction.
        objective_extraction_out = phase_outputs.get(
            "objective_extraction"
        ) or {}
        project_id = objective_extraction_out.get("project_id")
        project_path_str = objective_extraction_out.get("project_path")
        if not project_path_str and project_id:
            project_path_str = str(
                courseforge_exports_dir() / project_id
            )
        if not project_path_str:
            logger.error(
                "reuse_objectives: cannot resolve project_path from "
                "upstream objective_extraction output. Did the phase "
                "complete? extracted=%s",
                objective_extraction_out,
            )
            return None

        project_path = _Path(project_path_str)
        if not project_path.is_dir():
            logger.error(
                "reuse_objectives: resolved project_path is not a "
                "directory: %s",
                project_path,
            )
            return None

        # Build the canonical synthesized JSON.
        course_name = (
            workflow_params.get("course_name")
            or normalized.get("course_name")
            or project_id
            or ""
        )
        duration_weeks = normalized.get("duration_weeks") or workflow_params.get(
            "duration_weeks",
        )

        lo_entries: List[Dict[str, Any]] = []
        for to in terminal:
            entry = dict(to)
            entry["hierarchy_level"] = "terminal"
            lo_entries.append(entry)
        for co in chapter_flat:
            entry = dict(co)
            entry["hierarchy_level"] = "chapter"
            lo_entries.append(entry)

        synthesized = {
            "course_name": course_name,
            "generated_from": str(reuse_path),
            "mint_method": "reuse_objectives",
            "duration_weeks": duration_weeks,
            "learning_outcomes": lo_entries,
            "terminal_objectives": [dict(t) for t in terminal],
            "chapter_objectives": chapter_groups,
            "synthesized_at": datetime.now().isoformat(),
        }

        # Write into the project directory. Use the canonical filename.
        objectives_out_dir = project_path / "01_learning_objectives"
        objectives_out_dir.mkdir(parents=True, exist_ok=True)
        objectives_out_path = (
            objectives_out_dir / "synthesized_objectives.json"
        )
        try:
            objectives_out_path.write_text(
                json.dumps(synthesized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(
                "reuse_objectives: failed to write %s: %s",
                objectives_out_path, e,
            )
            return None

        # Update project_config so downstream phases pick it up.
        config_path = project_path / "project_config.json"
        config_data: Dict[str, Any] = {}
        if config_path.exists():
            try:
                config_data = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                config_data = {}
        config_data["objectives_path"] = str(objectives_out_path)
        config_data["synthesized_objectives_path"] = str(objectives_out_path)
        # Provenance for the course_planning self-poisoning reuse guard
        # (MCP/tools/pipeline_tools.py::_plan_course_structure): this is the
        # operator ``--reuse-objectives`` path, so the pin is operator-
        # supplied and MAY short-circuit synthesis on a re-run.
        config_data["objectives_source"] = "operator"
        config_data["course_name"] = course_name
        config_data["project_id"] = project_id or project_path.name
        config_data["status"] = "planned"
        if duration_weeks is not None:
            config_data["duration_weeks"] = duration_weeks
        try:
            config_path.write_text(
                json.dumps(config_data, indent=2), encoding="utf-8",
            )
        except OSError as e:
            logger.warning(
                "reuse_objectives: failed to update project_config.json "
                "(non-fatal): %s",
                e,
            )

        objective_ids = [str(e["id"]) for e in lo_entries if e.get("id")]
        joined_ids = ",".join(objective_ids)

        # Phase-ordering fix (Option A1) companion: back-fill
        # learning_outcome_refs on the on-disk DART chunkset for reuse
        # runs too. On a live (non-reuse) course_planning,
        # _plan_course_structure runs this same backfill at its tail; the
        # reuse path short-circuits that helper, so without this call the
        # reused chunks keep empty learning_outcome_refs and the
        # downstream concept_extraction graph degrades to the
        # related_to-dominated shape. Best-effort: a failure logs a
        # warning and does not change the synthesized phase output.
        try:
            from MCP.tools.pipeline_tools import (
                _backfill_dart_chunk_lo_refs as _backfill_lo_refs,
            )

            # Match the slug derivation used by _plan_course_structure's
            # live backfill (NOT lib.ontology.slugs.canonical_slug, which
            # strips hyphens differently) so we re-open the same
            # <libv2>/courses/<slug>/dart_chunks/chunks.jsonl the chunking
            # phase wrote.
            course_slug = (
                str(course_name or "")
                .lower()
                .replace("_", "-")
                .replace(" ", "-")
            )
            if course_slug and objective_ids:
                _backfill_lo_refs(
                    course_slug=course_slug,
                    objective_ids=objective_ids,
                    libv2_root=workflow_params.get("libv2_root"),
                )
        except Exception as e:  # noqa: BLE001 — best-effort, never block reuse
            logger.warning(
                "reuse_objectives: DART chunk LO-ref backfill failed "
                "(non-fatal): %s",
                e,
            )

        logger.info(
            "reuse_objectives: synthesised course_planning phase_output "
            "with %d terminal + %d chapter objectives from %s",
            len(terminal), len(chapter_flat), reuse_path,
        )

        return {
            "project_id": project_id or project_path.name,
            "synthesized_objectives_path": str(objectives_out_path),
            "objective_ids": joined_ids,
            "terminal_count": len(terminal),
            "chapter_count": len(chapter_flat),
            "_completed": True,
            "_skipped": True,
            "_gates_passed": True,
            "_skip_reason": (
                "reuse_objectives=True; reused user-supplied "
                "objectives JSON"
            ),
        }

    def _resolve_outline_dir(
        self, workflow_params: Dict[str, Any]
    ) -> Optional[Path]:
        """Resolve the OUTLINE_DIR for ``_synthesize_outline_output``.

        Phase 5 Subtask 2. Resolution chain:

        * ``workflow_params["outline_dir"]`` — explicit operator-supplied
          project export path (Worker WA's forthcoming --outline flag).
        * ``workflow_params["courseforge_stage"]`` set (commit 96e1bde) =>
          walk ``Courseforge/exports/PROJ-{COURSE_NAME}-*`` and pick
          the most-recently-modified candidate.

        Returns ``None`` when neither route resolves a directory; the
        caller treats that as "not a stage subcommand run, fall
        through to normal full-pipeline execution."
        """
        explicit = workflow_params.get("outline_dir")
        if explicit:
            cand = Path(explicit)
            if cand.is_dir():
                return cand
            logger.warning(
                "outline reuse: outline_dir param=%r not a directory; "
                "falling through to courseforge_stage resolution",
                explicit,
            )

        stage = workflow_params.get("courseforge_stage")
        if not stage:
            return None
        course_name = workflow_params.get("course_name") or ""
        if not course_name:
            logger.warning(
                "outline reuse: courseforge_stage=%r set but course_name "
                "is empty; cannot resolve project dir",
                stage,
            )
            return None
        exports_root = courseforge_exports_dir()
        if not exports_root.is_dir():
            logger.warning(
                "outline reuse: %s not a directory; no project to "
                "resume from",
                exports_root,
            )
            return None
        prefix = f"PROJ-{course_name}-"
        candidates: List[Tuple[float, Path]] = []
        for cand in exports_root.iterdir():
            if not cand.is_dir():
                continue
            if not cand.name.startswith(prefix):
                continue
            candidates.append((cand.stat().st_mtime, cand))
        if not candidates:
            logger.warning(
                "outline reuse: no project dir under %s matches "
                "course_name=%r (prefix=%r)",
                exports_root, course_name, prefix,
            )
            return None
        candidates.sort(reverse=True)
        resolved = candidates[0][1]
        logger.info(
            "outline reuse: resolved courseforge_stage=%r project to %s "
            "(most recent of %d candidates)",
            stage, resolved, len(candidates),
        )
        return resolved

    def _synthesize_outline_output(
        self,
        outline_dir: Path,
        target_phases: Optional[List[str]] = None,
        stage_active_phases: Optional[frozenset] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Reconstruct phase_outputs for upstream phases from disk.

        Phase 5 Subtask 2. When an operator runs ``ed4all run
        courseforge-rewrite`` (or any of the new courseforge-* stage
        subcommands), the upstream phases — dart_conversion, staging,
        chunking, objective_extraction, source_mapping,
        concept_extraction, course_planning,
        content_generation_outline, inter_tier_validation — must already
        have run; their output artifacts live under the project export
        directory + the course's LibV2 directory. This synthesizer walks
        those locations and reconstructs the per-phase ``phase_outputs``
        dicts (matching the keys ``inputs_from`` references in
        ``config/workflows.yaml``) so the workflow runner's ``_completed``
        skip check at line 860 fires for every upstream phase. The
        rewrite tier (or any single-tier phase that depends on these
        upstream outputs) then runs without re-dispatching the upstream
        phases.

        ``outline_dir`` accepts either:

        * The Courseforge project export root, e.g.
          ``Courseforge/exports/PROJ-PHYS_101-20260502/``.
        * The ``01_outline/`` subdirectory inside that project, e.g.
          ``Courseforge/exports/PROJ-PHYS_101-20260502/01_outline``.

        In either case we resolve to the project_path. ``project_config.json``
        at the project root supplies course_name + staging_dir.

        Returns a dict keyed by phase_name; each value is a phase_outputs
        dict carrying ``_completed: True`` plus the canonical output
        keys that ``inputs_from`` for downstream phases pulls. When an
        upstream artifact is absent / unreadable, that phase is omitted
        from the returned dict (warning-logged) so the workflow runner's
        ``_dependencies_met`` check at line 1643 surfaces the gap as a
        normal dependency failure rather than a silent inconsistency.

        Recognized phase names (plan §5):

        * ``dart_conversion`` — synthesises ``output_paths`` from the
          staging manifest's HTML inputs (each staged ``*_accessible.html``
          maps back to a DART output).
        * ``staging`` — ``staging_dir`` from project_config.json.
        * ``chunking`` — reads ``LibV2/courses/<slug>/dart_chunks/
          manifest.json`` for ``dart_chunks_sha256`` + ``chunks.jsonl``
          path.
        * ``objective_extraction`` — reads
          ``<project>/01_learning_objectives/textbook_structure.json``.
        * ``source_mapping`` — reads ``<project>/source_module_map.json``.
        * ``concept_extraction`` — reads
          ``LibV2/courses/<slug>/concept_graph/manifest.json``.
        * ``course_planning`` — reads
          ``<project>/01_learning_objectives/synthesized_objectives.json``.
        * ``content_generation_outline`` — reads
          ``<project>/01_outline/blocks_outline.jsonl``.
        * ``inter_tier_validation`` — reads
          ``<project>/01_outline/blocks_validated.jsonl`` (+
          ``blocks_failed.jsonl``).

        ``target_phases`` filters which upstream phases to reconstruct;
        defaults to the full canonical list above. Unknown names are
        silently dropped (not an error).

        ``content_generation_rewrite`` — reads
        ``<project>/04_rewrite/blocks_final.jsonl`` and synthesises
        ``blocks_final_path``. ONLY reconstructed when a
        ``stage_active_phases`` whitelist is supplied AND
        ``content_generation_rewrite`` is OUTSIDE it (i.e. the active stage
        SKIPS the rewrite tier but a whitelisted phase — typically
        ``post_rewrite_validation`` under ``courseforge-validate`` — still
        needs its on-disk output via ``inputs_from``). It is deliberately
        NOT in the default ``canonical_phases`` set: on a normal
        full-pipeline run, or under ``courseforge`` / ``courseforge-rewrite``
        (where the rewrite tier is whitelisted and meant to RE-RUN),
        synthesising it as ``_completed=True`` would wrongly make the loop
        skip the live rewrite.
        """
        from pathlib import Path as _Path

        canonical_phases = [
            "dart_conversion",
            "staging",
            "chunking",
            "objective_extraction",
            "source_mapping",
            "concept_extraction",
            "course_planning",
            "content_generation_outline",
            "inter_tier_validation",
        ]
        # Conditionally reconstruct the rewrite-tier output for a stage
        # subcommand that SKIPS content_generation_rewrite but depends on
        # its blocks_final_path downstream (courseforge-validate).
        if (
            stage_active_phases is not None
            and "content_generation_rewrite" not in stage_active_phases
        ):
            canonical_phases.append("content_generation_rewrite")
        if target_phases is None:
            phases = list(canonical_phases)
        else:
            phases = [p for p in target_phases if p in canonical_phases]

        outline_dir = _Path(outline_dir)
        if outline_dir.name == "01_outline":
            project_path = outline_dir.parent
        else:
            project_path = outline_dir

        if not project_path.is_dir():
            logger.error(
                "outline reuse: project_path is not a directory: %s",
                project_path,
            )
            return {}

        # Load project_config.json — supplies course_name +
        # staging_dir + project_id.
        config_path = project_path / "project_config.json"
        config_data: Dict[str, Any] = {}
        if config_path.exists():
            try:
                config_data = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as e:
                logger.warning(
                    "outline reuse: project_config.json unreadable at %s: %s",
                    config_path, e,
                )

        course_name = (
            config_data.get("course_name")
            or project_path.name.split("-")[1] if "-" in project_path.name else ""
        )
        project_id = config_data.get("project_id") or project_path.name
        course_slug = (
            (course_name or "").lower().replace("_", "-").replace(" ", "-")
        )
        libv2_course_dir = (
            PROJECT_ROOT / "LibV2" / "courses" / course_slug
            if course_slug else None
        )

        synthesized: Dict[str, Dict[str, Any]] = {}

        # ----- staging -----------------------------------------------
        if "staging" in phases:
            staging_dir_str = config_data.get("staging_dir")
            staging_dir: Optional[Path] = None
            if staging_dir_str:
                cand = _Path(staging_dir_str)
                if cand.is_dir():
                    staging_dir = cand
            # Fall back: walk Courseforge/inputs/textbooks/ for the
            # most-recent staging dir whose manifest carries
            # course_name == course_name.
            if staging_dir is None:
                inputs_root = (
                    PROJECT_ROOT / "Courseforge" / "inputs" / "textbooks"
                )
                if inputs_root.is_dir() and course_name:
                    candidates = []
                    for cand in inputs_root.iterdir():
                        if not cand.is_dir():
                            continue
                        manifest = cand / "staging_manifest.json"
                        if not manifest.exists():
                            continue
                        try:
                            mdata = json.loads(
                                manifest.read_text(encoding="utf-8")
                            )
                        except (OSError, ValueError):
                            continue
                        if mdata.get("course_name") == course_name:
                            candidates.append((manifest.stat().st_mtime, cand))
                    if candidates:
                        candidates.sort(reverse=True)
                        staging_dir = candidates[0][1]

            if staging_dir is not None and staging_dir.is_dir():
                staged_files = sorted(
                    str(p) for p in staging_dir.glob("*.html")
                )
                synthesized["staging"] = {
                    "staging_dir": str(staging_dir),
                    "staged_files": staged_files,
                    "file_count": len(staged_files),
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": "outline reuse: staging_dir from project_config",
                }
            else:
                logger.warning(
                    "outline reuse: staging_dir not found for project %s "
                    "(config_data.staging_dir=%r); skipping staging "
                    "phase pre-population",
                    project_id, staging_dir_str,
                )

        # Resolve dart_html_paths from staging if available.
        # ----- dart_conversion ---------------------------------------
        if "dart_conversion" in phases and "staging" in synthesized:
            staged_files = synthesized["staging"].get("staged_files") or []
            if staged_files:
                synthesized["dart_conversion"] = {
                    "output_path": staged_files[0],
                    "output_paths": ",".join(staged_files),
                    "html_path": staged_files[0],
                    "html_paths": ",".join(staged_files),
                    "success": True,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: derived from staging manifest"
                    ),
                }

        # ----- chunking ----------------------------------------------
        if "chunking" in phases and libv2_course_dir is not None:
            chunks_dir = libv2_course_dir / "dart_chunks"
            chunks_path = chunks_dir / "chunks.jsonl"
            manifest_path = chunks_dir / "manifest.json"
            if chunks_path.exists() and manifest_path.exists():
                try:
                    cmanifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    sha256 = cmanifest.get("chunks_sha256") or ""
                    synthesized["chunking"] = {
                        "dart_chunks_path": str(chunks_path),
                        "dart_chunks_sha256": sha256,
                        "manifest_path": str(manifest_path),
                        "course_slug": course_slug,
                        "chunks_count": cmanifest.get("chunks_count", 0),
                        "_completed": True,
                        "_skipped": True,
                        "_gates_passed": True,
                        "_skip_reason": (
                            "outline reuse: read dart_chunks/manifest.json"
                        ),
                    }
                except (OSError, ValueError) as e:
                    logger.warning(
                        "outline reuse: chunking manifest unreadable at "
                        "%s: %s",
                        manifest_path, e,
                    )
            else:
                logger.warning(
                    "outline reuse: chunking artifacts missing under "
                    "%s; skipping chunking phase pre-population",
                    chunks_dir,
                )

        # ----- objective_extraction ----------------------------------
        if "objective_extraction" in phases:
            structure_path = (
                project_path / "01_learning_objectives"
                / "textbook_structure.json"
            )
            if structure_path.exists():
                chapter_count = 0
                duration_weeks = config_data.get("duration_weeks")
                try:
                    structure_data = json.loads(
                        structure_path.read_text(encoding="utf-8")
                    )
                    chapter_count = len(
                        structure_data.get("chapters") or []
                    )
                    if duration_weeks is None:
                        duration_weeks = structure_data.get("duration_weeks")
                except (OSError, ValueError) as e:
                    logger.warning(
                        "outline reuse: textbook_structure.json "
                        "unreadable: %s",
                        e,
                    )
                synthesized["objective_extraction"] = {
                    "project_id": project_id,
                    "project_path": str(project_path),
                    "textbook_structure_path": str(structure_path),
                    "chapter_count": chapter_count,
                    "duration_weeks": duration_weeks,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read textbook_structure.json"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: textbook_structure.json missing at "
                    "%s; skipping objective_extraction pre-population",
                    structure_path,
                )

        # ----- source_mapping ----------------------------------------
        if "source_mapping" in phases:
            map_path = project_path / "source_module_map.json"
            if map_path.exists():
                source_chunk_ids: List[str] = []
                try:
                    map_data = json.loads(
                        map_path.read_text(encoding="utf-8")
                    )
                    if isinstance(map_data, dict):
                        for week_entries in map_data.values():
                            if not isinstance(week_entries, list):
                                continue
                            for entry in week_entries:
                                if isinstance(entry, dict):
                                    cid = entry.get("chunk_id")
                                    if cid:
                                        source_chunk_ids.append(str(cid))
                except (OSError, ValueError) as e:
                    logger.warning(
                        "outline reuse: source_module_map.json "
                        "unreadable: %s",
                        e,
                    )
                staging_dir_str = (
                    synthesized.get("staging", {}).get("staging_dir") or ""
                )
                synthesized["source_mapping"] = {
                    "source_module_map_path": str(map_path),
                    "source_chunk_ids": sorted(set(source_chunk_ids)),
                    "staging_dir": staging_dir_str,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read source_module_map.json"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: source_module_map.json missing at "
                    "%s; skipping source_mapping pre-population",
                    map_path,
                )

        # ----- concept_extraction ------------------------------------
        if "concept_extraction" in phases and libv2_course_dir is not None:
            graph_dir = libv2_course_dir / "concept_graph"
            graph_path = graph_dir / "concept_graph_semantic.json"
            cmanifest_path = graph_dir / "manifest.json"
            if graph_path.exists():
                sha256 = ""
                if cmanifest_path.exists():
                    try:
                        cmanifest = json.loads(
                            cmanifest_path.read_text(encoding="utf-8")
                        )
                        sha256 = cmanifest.get("concept_graph_sha256") or ""
                    except (OSError, ValueError):
                        sha256 = ""
                synthesized["concept_extraction"] = {
                    "concept_graph_path": str(graph_path),
                    "concept_graph_sha256": sha256,
                    "course_slug": course_slug,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read concept_graph_semantic.json"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: concept_graph_semantic.json missing "
                    "at %s; skipping concept_extraction pre-population",
                    graph_path,
                )

        # ----- course_planning ---------------------------------------
        if "course_planning" in phases:
            objectives_path = (
                project_path / "01_learning_objectives"
                / "synthesized_objectives.json"
            )
            if objectives_path.exists():
                terminal_count = 0
                chapter_count = 0
                objective_ids: List[str] = []
                try:
                    odata = json.loads(
                        objectives_path.read_text(encoding="utf-8")
                    )
                    terminal = odata.get("terminal_objectives") or []
                    chapter_groups = odata.get("chapter_objectives") or []
                    terminal_count = len(terminal)
                    for to in terminal:
                        if isinstance(to, dict) and to.get("id"):
                            objective_ids.append(str(to["id"]))
                    for group in chapter_groups:
                        if not isinstance(group, dict):
                            continue
                        inner = group.get("objectives") or []
                        for co in inner:
                            if isinstance(co, dict) and co.get("id"):
                                objective_ids.append(str(co["id"]))
                                chapter_count += 1
                except (OSError, ValueError) as e:
                    logger.warning(
                        "outline reuse: synthesized_objectives.json "
                        "unreadable: %s",
                        e,
                    )
                synthesized["course_planning"] = {
                    "project_id": project_id,
                    "synthesized_objectives_path": str(objectives_path),
                    "objective_ids": ",".join(objective_ids),
                    "terminal_count": terminal_count,
                    "chapter_count": chapter_count,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read synthesized_objectives.json"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: synthesized_objectives.json missing "
                    "at %s; skipping course_planning pre-population",
                    objectives_path,
                )

        # ----- content_generation_outline ----------------------------
        outline_subdir = project_path / "01_outline"
        if "content_generation_outline" in phases:
            blocks_outline_path = outline_subdir / "blocks_outline.jsonl"
            if blocks_outline_path.exists():
                # Count weeks via a one-pass scan of the JSONL.
                weeks_seen: set = set()
                block_count = 0
                try:
                    with blocks_outline_path.open(
                        "r", encoding="utf-8"
                    ) as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            block_count += 1
                            try:
                                entry = json.loads(line)
                            except ValueError:
                                continue
                            wk = entry.get("week")
                            if wk is not None:
                                weeks_seen.add(wk)
                except OSError as e:
                    logger.warning(
                        "outline reuse: blocks_outline.jsonl unreadable: %s",
                        e,
                    )
                # Thread the W2-persisted outline sidecars so a stage
                # subcommand's rewrite tier rehydrates per-block source
                # chunks + objectives (the same keys the runner's
                # ``inputs_from`` maps for ``content_generation_rewrite``).
                # Without these, ``_load_outline_chunks`` falls through to an
                # empty ``chunks_lookup`` and the rewrite backstop's canonical
                # source-id resolution + domain-CURIE minting silently no-op
                # for EVERY block (the rewrite_grounding_missing path) — so a
                # ``courseforge-rewrite`` re-run loses all ``data-cf-source-ids``
                # grounding and curie anchoring that a full run produced.
                _outline_chunks_path = outline_subdir / "outline_chunks.json"
                _outline_objectives_path = (
                    outline_subdir / "outline_objectives.json"
                )
                synthesized["content_generation_outline"] = {
                    "blocks_outline_path": str(blocks_outline_path),
                    "project_id": project_id,
                    "weeks_prepared": len(weeks_seen),
                    "block_count": block_count,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read blocks_outline.jsonl"
                    ),
                }
                if _outline_chunks_path.exists():
                    synthesized["content_generation_outline"][
                        "outline_chunks_path"
                    ] = str(_outline_chunks_path)
                else:
                    logger.warning(
                        "outline reuse: outline_chunks.json missing at %s; "
                        "rewrite-tier source-id resolution + CURIE minting "
                        "will no-op (empty chunks_lookup)",
                        _outline_chunks_path,
                    )
                if _outline_objectives_path.exists():
                    synthesized["content_generation_outline"][
                        "outline_objectives_path"
                    ] = str(_outline_objectives_path)
            else:
                logger.warning(
                    "outline reuse: blocks_outline.jsonl missing at %s; "
                    "skipping content_generation_outline pre-population",
                    blocks_outline_path,
                )

        # ----- inter_tier_validation ---------------------------------
        if "inter_tier_validation" in phases:
            validated_path = outline_subdir / "blocks_validated.jsonl"
            failed_path = outline_subdir / "blocks_failed.jsonl"
            if validated_path.exists():
                synthesized["inter_tier_validation"] = {
                    "blocks_validated_path": str(validated_path),
                    "blocks_failed_path": str(failed_path)
                    if failed_path.exists()
                    else "",
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read blocks_validated.jsonl"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: blocks_validated.jsonl missing at "
                    "%s; skipping inter_tier_validation pre-population",
                    validated_path,
                )

        # ----- content_generation_rewrite ---------------------------
        # Only reachable when the stage whitelist SKIPS the rewrite tier
        # (courseforge-validate) — see the canonical_phases gate above.
        # post_rewrite_validation resolves its ``blocks_final_path``
        # ``inputs_from`` against this synthesised output, so without it
        # the phase-handler task receives no path and returns
        # ``{"error": "blocks_final_path is required"}`` → zero outputs →
        # the anti-zombie guard fails the workflow before the
        # validation-report writer runs.
        if "content_generation_rewrite" in phases:
            rewrite_subdir = project_path / "04_rewrite"
            blocks_final_path = rewrite_subdir / "blocks_final.jsonl"
            if blocks_final_path.exists():
                rewrite_block_count = 0
                try:
                    with blocks_final_path.open(
                        "r", encoding="utf-8"
                    ) as fh:
                        for line in fh:
                            if line.strip():
                                rewrite_block_count += 1
                except OSError as e:
                    logger.warning(
                        "outline reuse: blocks_final.jsonl unreadable: %s",
                        e,
                    )
                synthesized["content_generation_rewrite"] = {
                    "blocks_final_path": str(blocks_final_path),
                    "project_id": project_id,
                    "block_count": rewrite_block_count,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read 04_rewrite/blocks_final.jsonl"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: blocks_final.jsonl missing at %s; "
                    "skipping content_generation_rewrite pre-population "
                    "(post_rewrite_validation will fail loudly with "
                    "'blocks_final_path is required')",
                    blocks_final_path,
                )

        logger.info(
            "outline reuse: synthesised phase_outputs for %d phase(s) "
            "from %s: %s",
            len(synthesized), project_path, sorted(synthesized.keys()),
        )
        return synthesized

    # Phase 5 Subtask 4: validation-report writer schema version. Bumped
    # alongside any breaking change to the per-block summary shape;
    # consumers (operator-facing dashboards, dry-run preview tooling,
    # the Phase 6 ABCD concept-extractor's validator surface) should
    # gate on this field when reading the report.
    _VALIDATION_REPORT_SCHEMA_VERSION = "v2"

    def _write_validation_report(
        self,
        *,
        workflow_id: str,
        phase_name: str,
        phase_output: Dict[str, Any],
        gate_results_list: Optional[List[Dict[str, Any]]],
    ) -> Optional[Path]:
        """Aggregate inter-tier / post-rewrite gate results into ``report.json``.

        Phase 5 Subtask 4. The shipped phase helpers
        (``_run_inter_tier_validation``,
        ``_run_post_rewrite_validation``) emit JSONL only —
        ``blocks_validated.jsonl`` + ``blocks_failed.jsonl`` next to the
        consumed blocks file. The operator-facing structured summary
        (passed / failed / escalated counts plus a ``per_block`` array
        keyed by ``block_id``) is a Phase 5 deliverable that lives at:

        * ``{project_root}/02_validation_report/report.json`` for
          ``inter_tier_validation``.
        * ``{project_root}/04_rewrite/02_validation_report/report.json``
          for ``post_rewrite_validation``.

        Where ``project_root`` is derived from the
        ``blocks_validated_path`` extracted output (which lives at
        ``{project_root}/01_outline/blocks_validated.jsonl`` for the
        outline-tier inter_tier_validation phase, and at
        ``{project_root}/04_rewrite/blocks_validated.jsonl`` for the
        rewrite-tier post_rewrite_validation phase — matching how the
        rewrite tier writes its blocks JSONL into ``04_rewrite/``).

        Returns the report path on successful write, or ``None`` when
        the report could not be written (no ``blocks_validated_path``
        in the phase output, or filesystem error).

        Schema (matches plan §6 ``report.json``):

        ::

            {
              "run_id": "<workflow_id>",
              "phase": "<phase_name>",
              "schema_version": "v2",
              "total_blocks": <int>,
              "passed": <int>,
              "failed": <int>,
              "escalated": <int>,
              "curie_force_injected": <int>,
              "per_block": [
                {
                  "block_id": "<id>",
                  "block_type": "<type>",
                  "page": "<page_id|null>",
                  "week": <int|null>,
                  "status": "passed|failed|escalated",
                  "gate_results": [
                    {
                      "gate_id": "<id>",
                      "action": "<action|null>",
                      "passed": <bool>,         # PHASE-level pass/fail
                      "issue_count": <int>      # issues whose location == THIS block_id
                    },
                    ...
                  ],
                  "escalation_marker": "<marker|null>",
                  "curie_force_injected": true   # present only when set
                },
                ...
              ],
              "phase_level_gate_results": [
                {
                  "gate_id": "<id>",
                  "action": "<action|null>",
                  "passed": <bool>,
                  "issue_count": <int>,             # TOTAL issues for the gate
                  "unattributed_issue_count": <int> # issues whose location matched
                                                    # NO block_id (objective/page/None)
                },
                ...
              ]
            }

        Per-block ``issue_count`` attribution (schema v2 fix): a
        ``GateResult``'s ``issues[]`` each carry a ``location`` field
        whose meaning is gate-dependent — a BLOCK-level gate
        (``udl_coverage`` / ``qa_checklist`` / …) sets ``location`` to a
        ``block_id``; OBJECTIVE-level (``triangle_completeness`` ->
        ``CO-01``), MODULE/PAGE-level (``retrieval_presence`` ->
        ``week_01_content_01``), and summary issues set ``location`` to a
        non-block id or ``None``. The writer builds, ONCE, a
        ``gate_id -> Counter(location)`` map from
        ``gate_results_list``, then each block's ``gate_results[].
        issue_count`` counts ONLY the issues whose ``location`` equals
        THAT block's ``block_id``. Schema v1 (the bug this supersedes)
        attached the same phase-level ``issue_count`` to every block,
        smearing course-wide totals across all blocks and inflating the
        calibration harness's per-gate fire-rates toward 100%. The
        per-gate ``passed`` stays PHASE-level (a gate either passed or
        failed for the whole phase); only ``issue_count`` is now
        per-block.

        ``phase_level_gate_results`` (schema v2 addition): preserves the
        structural (objective / page / ``None``-location) issues that are
        NOT attributable to any single block. Per gate it carries the
        TOTAL ``issue_count`` plus ``unattributed_issue_count`` (issues
        whose ``location`` matched no block_id). This is where the
        calibration harness reads structural gates
        (``triangle_completeness`` / ``retrieval_presence`` /
        anatomy-summary issues) at the right granularity instead of from
        smeared per-block counts.

        R6 — ``curie_force_injected``: the top-level count + the
        per-block boolean flag mark blocks that PASSED the
        ``rewrite_curie_anchoring`` gate only because
        ``RewriteProvider`` force-injected their CURIE anchoring after
        the rewrite LLM exhausted its remediation budget. ``status``
        stays ``passed`` (the appended hidden span legitimately anchors
        them); the flag exists so operators can quantify the
        silent-degradation class instead of treating those blocks as
        clean rewrites. The per-block flag is emitted only when ``True``
        so clean-rewrite entries stay byte-stable.
        """
        validated_path_raw = (phase_output or {}).get(
            "blocks_validated_path"
        )
        if not validated_path_raw:
            logger.debug(
                "Phase 5 validation_report: no blocks_validated_path "
                "in %s phase_output; nothing to aggregate",
                phase_name,
            )
            return None

        validated_path = Path(validated_path_raw)
        # Project root is two levels up from blocks_validated.jsonl:
        # ``<project_root>/<stage_dir>/blocks_validated.jsonl``.
        if not validated_path.is_absolute():
            validated_path = Path(validated_path)

        # The blocks JSONL lives in either ``01_outline/`` (outline-tier
        # inter_tier_validation) or ``04_rewrite/`` (rewrite-tier
        # post_rewrite_validation). The report dir is sibling to the
        # blocks file's stage dir for inter_tier_validation, and lives
        # INSIDE the stage dir for post_rewrite_validation per plan §6
        # ("rewrite writes its own equivalent under
        # 04_rewrite/02_validation_report/report.json").
        stage_dir = validated_path.parent
        if phase_name == "inter_tier_validation":
            report_dir = stage_dir.parent / "02_validation_report"
        else:  # post_rewrite_validation
            report_dir = stage_dir / "02_validation_report"

        try:
            report_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Phase 5 validation_report: cannot create %s: %s",
                report_dir, exc,
            )
            return None

        # Load the validated + failed blocks JSONL to build per-block
        # records. Failed blocks set status='failed'; blocks with
        # ``escalation_marker`` set are reclassified as 'escalated'.
        validated_blocks: List[Dict[str, Any]] = []
        failed_blocks: List[Dict[str, Any]] = []

        if validated_path.exists():
            try:
                for line in validated_path.read_text(
                    encoding="utf-8"
                ).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        validated_blocks.append(json.loads(line))
                    except ValueError:
                        continue
            except OSError as exc:
                logger.warning(
                    "Phase 5 validation_report: blocks_validated.jsonl "
                    "unreadable at %s: %s",
                    validated_path, exc,
                )

        failed_path_raw = (phase_output or {}).get("blocks_failed_path")
        if failed_path_raw:
            failed_path = Path(failed_path_raw)
            if failed_path.exists():
                try:
                    for line in failed_path.read_text(
                        encoding="utf-8"
                    ).splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            failed_blocks.append(json.loads(line))
                        except ValueError:
                            continue
                except OSError as exc:
                    logger.warning(
                        "Phase 5 validation_report: blocks_failed.jsonl "
                        "unreadable at %s: %s",
                        failed_path, exc,
                    )

        # Aggregate counts. Escalated == failed-with-non-null
        # escalation_marker (plan §3 escalated_only path); plain failed
        # blocks have ``escalation_marker is None`` or absent.
        per_block: List[Dict[str, Any]] = []
        passed_count = 0
        failed_count = 0
        escalated_count = 0
        # R6 silent-degradation fix: count blocks whose CURIE anchoring
        # was force-injected by ``RewriteProvider`` after the rewrite
        # LLM exhausted the remediation budget. Such blocks legitimately
        # PASS the ``rewrite_curie_anchoring`` gate (the appended hidden
        # span anchors them), so they would otherwise be invisible in
        # the report — identical to a clean rewrite. The signal is the
        # ``data-cf-curie-forced`` boolean attribute the force-injected
        # span carries inside ``block.content`` (the one Block field
        # that survives every JSONL round trip).
        curie_force_injected_count = 0
        try:
            from Courseforge.generators._rewrite_provider import (
                html_has_forced_curie_marker as _has_forced_curie,
            )
        except Exception:  # noqa: BLE001 — best-effort; absence -> no marker
            def _has_forced_curie(_html: Any) -> bool:  # type: ignore[misc]
                return False

        # gate_results_list is the executor's emit; we attach a stable
        # per-gate chain (gate_id / action / passed / issue_count) to
        # every block's ``gate_results`` so the operator can introspect
        # each gate's findings WITHOUT re-running the validators.
        #
        # Schema v2 attribution fix: each ``GateResult.issues[]`` entry
        # carries a ``location`` field. For a BLOCK-level gate the
        # location IS the block_id; for OBJECTIVE / MODULE / summary
        # issues the location is a non-block id or ``None``. We build,
        # ONCE, a per-gate ``Counter`` over issue locations, then each
        # block looks up the count for ITS OWN block_id — so a gate's
        # per-block ``issue_count`` reflects only the issues that name
        # that block, not the course-wide total (the schema-v1 bug).
        from collections import Counter as _Counter
        from collections import OrderedDict

        # gate_id -> Counter(location -> n_issues). Insertion-ordered so
        # the per-block ``gate_results`` array preserves chain order.
        gate_location_counts: "OrderedDict[str, _Counter]" = OrderedDict()
        # Stable per-gate (action, passed) + total issue_count, for the
        # phase-level section and the per-block chain scaffold.
        gate_chain_meta: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for gr in gate_results_list or []:
            if not isinstance(gr, dict):
                continue
            gid = gr.get("gate_id")
            issues = gr.get("issues") or []
            counter = gate_location_counts.setdefault(gid, _Counter())
            for issue in issues:
                loc = issue.get("location") if isinstance(issue, dict) else None
                # ``None`` locations (summary issues) are tracked under a
                # sentinel so they roll up into the phase-level section
                # but never attach to a block.
                counter[loc] += 1
            # Keep the LAST (action, passed, total) for a gate_id if the
            # executor somehow emitted it twice; total is the true count.
            gate_chain_meta[gid] = {
                "gate_id": gid,
                "action": gr.get("action"),
                "passed": gr.get("passed"),
                "issue_count": len(issues),
            }

        # Pre-resolve the set of real block_ids so we can compute, per
        # gate, how many issues were NOT attributable to any block
        # (objective/page/None-location issues) for the phase-level
        # section. Built from the loaded validated + failed blocks.
        all_block_ids: set = set()
        for _blk in validated_blocks:
            _bid = _blk.get("block_id")
            if _bid is not None:
                all_block_ids.add(_bid)
        for _blk in failed_blocks:
            _bid = _blk.get("block_id")
            if _bid is not None:
                all_block_ids.add(_bid)

        def _block_gate_results(block_id: Any) -> List[Dict[str, Any]]:
            """Per-block gate chain: phase-level (action, passed) + the
            issue_count of issues whose ``location`` == this block_id."""
            chain: List[Dict[str, Any]] = []
            for gid, meta in gate_chain_meta.items():
                counter = gate_location_counts.get(gid)
                per_block_issue_count = (
                    counter.get(block_id, 0) if counter is not None else 0
                )
                chain.append({
                    "gate_id": meta["gate_id"],
                    "action": meta["action"],
                    "passed": meta["passed"],
                    "issue_count": per_block_issue_count,
                })
            return chain

        def _record_block(entry: Dict[str, Any], status: str) -> None:
            nonlocal passed_count, failed_count, escalated_count
            nonlocal curie_force_injected_count
            esc = entry.get("escalation_marker")
            if status == "failed" and esc:
                status = "escalated"
            if status == "passed":
                passed_count += 1
            elif status == "escalated":
                escalated_count += 1
            else:
                failed_count += 1
            # R6: a force-injected block keeps its (legitimate) verdict —
            # do NOT flip status — but is flagged so it is countable /
            # greppable in the report. The rewrite LLM provably could not
            # author this block cleanly across the full remediation
            # budget; the appended hidden span is what makes it pass.
            forced = bool(_has_forced_curie(entry.get("content")))
            if forced:
                curie_force_injected_count += 1
            block_record: Dict[str, Any] = {
                "block_id": entry.get("block_id"),
                "block_type": entry.get("block_type"),
                "page": entry.get("page_id"),
                "week": entry.get("week"),
                "status": status,
                "gate_results": _block_gate_results(entry.get("block_id")),
                "escalation_marker": esc,
            }
            if forced:
                block_record["curie_force_injected"] = True
            per_block.append(block_record)

        for entry in validated_blocks:
            _record_block(entry, "passed")
        for entry in failed_blocks:
            _record_block(entry, "failed")

        # W3.H sub-task H2: build the canonical source_coverage block.
        # The arrow is "blocks attempted -> blocks passing the rewrite
        # tier"; consumed_count = total_blocks (blocks attempted),
        # emitted_count = passed_count (blocks that passed validation).
        # Drop reasons walk the per_block array and bucket each
        # non-passing block by its escalation_marker:
        #   - validator_consensus_fail: every regen candidate failed
        #     validation (canonical marker minted by the router).
        #   - regen_budget: outline_budget_exhausted marker (regen cap
        #     hit with no validator-consensus-fail short-circuit).
        #   - escalate_immediately: per-block-type policy short-circuit
        #     (block.touched_by carries the escalate_immediately
        #     purpose tag — surfaced via Touch.purpose in the
        #     courseforge router contract; absence implies the block
        #     went through normal regen).
        # Plain-failed blocks (escalation_marker is None) fall into a
        # generic `validation_failed` bucket so the dropped_count ==
        # sum(drop_reasons.values()) invariant holds.
        from lib.governance.source_coverage import build_source_coverage
        _v_drop_reasons: Dict[str, int] = {}
        for blk in per_block:
            status = blk.get("status")
            if status == "passed":
                continue
            esc = blk.get("escalation_marker")
            if esc == "validator_consensus_fail":
                key = "validator_consensus_fail"
            elif esc == "outline_budget_exhausted":
                key = "regen_budget"
            elif esc == "rewrite_dispatch_error" or esc == "outline_dispatch_error":
                # Per-tier dispatch errors get their own bucket so
                # the master aggregator can disambiguate dispatch
                # failures from semantic exhaustion.
                key = "dispatch_error"
            elif esc == "structural_unfixable":
                key = "structural_unfixable"
            elif esc == "per_claim_attribution_unfixable":
                key = "per_claim_attribution_unfixable"
            elif esc:
                # Any other minted marker (escalate_immediately /
                # custom future markers) falls under the canonical
                # escalate_immediately key per plan §W3.H.
                key = "escalate_immediately"
            else:
                key = "validation_failed"
            _v_drop_reasons[key] = _v_drop_reasons.get(key, 0) + 1
        _v_total = passed_count + failed_count + escalated_count
        source_coverage_block = build_source_coverage(
            consumed_count=_v_total,
            emitted_count=passed_count,
            drop_reasons=_v_drop_reasons,
            dropped_count=_v_total - passed_count,
            label=f"two_pass_{phase_name}",
        )

        # Schema v2: the phase-level gate section preserves the
        # structural (objective / page / None-location) issues that are
        # NOT attributable to any single block. Per gate it carries the
        # TOTAL issue_count plus the count of issues whose ``location``
        # matched NO block_id — i.e. objective-scoped
        # (``triangle_completeness`` -> ``CO-01``), page/module-scoped
        # (``retrieval_presence`` -> ``week_01_content_01``), and
        # ``None``-location summary issues. The calibration harness reads
        # structural gates from HERE rather than from per-block counts.
        phase_level_gate_results: List[Dict[str, Any]] = []
        for gid, meta in gate_chain_meta.items():
            counter = gate_location_counts.get(gid)
            if counter is None:
                total = 0
                unattributed = 0
            else:
                total = sum(counter.values())
                unattributed = sum(
                    n for loc, n in counter.items()
                    if loc is None or loc not in all_block_ids
                )
            phase_level_gate_results.append({
                "gate_id": meta["gate_id"],
                "action": meta["action"],
                "passed": meta["passed"],
                "issue_count": total,
                "unattributed_issue_count": unattributed,
            })

        report = {
            "run_id": workflow_id,
            "phase": phase_name,
            "schema_version": self._VALIDATION_REPORT_SCHEMA_VERSION,
            "total_blocks": passed_count + failed_count + escalated_count,
            "passed": passed_count,
            "failed": failed_count,
            "escalated": escalated_count,
            # R6: blocks that PASSED only because RewriteProvider
            # force-injected their CURIE anchoring after the rewrite LLM
            # exhausted the remediation budget. A subset of ``passed`` —
            # NOT a separate status — surfaced so operators can quantify
            # the silent-degradation class. Per-block flag:
            # ``per_block[*].curie_force_injected``.
            "curie_force_injected": curie_force_injected_count,
            "per_block": per_block,
            "phase_level_gate_results": phase_level_gate_results,
            "source_coverage": source_coverage_block,
        }

        report_path = report_dir / "report.json"
        try:
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Phase 5 validation_report: cannot write %s: %s",
                report_path, exc,
            )
            return None

        logger.info(
            "Phase 5 validation_report: wrote %s "
            "(total=%d passed=%d failed=%d escalated=%d)",
            report_path, report["total_blocks"], passed_count,
            failed_count, escalated_count,
        )
        return report_path

    # Canonical output keys that, when ALL absent from a recorded
    # ``_completed`` phase output, signal the resume-restoration pass
    # should attempt to reconstruct that phase's outputs from disk.
    # Mirrors the load-bearing ``inputs_from`` consumers downstream: a
    # phase whose declared keys are all missing cannot satisfy any
    # downstream ``phase_outputs`` reference.
    _RESUME_RESTORE_REQUIRED_KEYS: Dict[str, List[str]] = {
        "objective_extraction": ["project_id", "project_path"],
        "staging": ["staging_dir"],
        "source_mapping": ["source_module_map_path"],
        "course_planning": ["synthesized_objectives_path"],
        "content_generation": ["content_dir"],
        "content_generation_rewrite": ["content_dir"],
        "packaging": ["package_path"],
    }

    def _restore_resume_phase_outputs(
        self, phase_outputs: Dict[str, Dict],
    ) -> None:
        """Backfill missing output keys for resumed ``_completed`` phases.

        Mutates ``phase_outputs`` in place. For each phase already marked
        ``_completed=True`` whose recorded dict is MISSING all of its
        canonical required output keys (see
        ``_RESUME_RESTORE_REQUIRED_KEYS``), reconstruct those keys from
        on-disk artifacts under the Courseforge project export +
        the LibV2 course dir and merge them in — filling ONLY keys that
        are absent (never overwriting a recorded value).

        Backward-compatibility contract:

        * No-op on a fresh (non-resume) run: ``phase_outputs`` is empty,
          so the loop body never executes.
        * No-op when every ``_completed`` phase already carries its
          required keys (the normal happy-resume path): nothing is
          reconstructed.
        * Best-effort: a phase whose artifacts are missing / unreadable
          is left exactly as recorded (warning-logged); the loop's
          ``_dependencies_met`` check then surfaces the gap as a normal
          dependency failure rather than a silent inconsistency.
        """
        if not phase_outputs:
            return

        # Identify completed phases whose required keys are all missing.
        needs_restore = []
        for phase_name, required in self._RESUME_RESTORE_REQUIRED_KEYS.items():
            recorded = phase_outputs.get(phase_name)
            if not isinstance(recorded, dict):
                continue
            if not recorded.get("_completed"):
                continue
            # Only restore when EVERY required key is absent or empty —
            # i.e. the recorded dict carries no usable routing signal for
            # this phase. A partially-populated dict is left untouched to
            # avoid clobbering a real (if sparse) prior extraction.
            if all(not recorded.get(k) for k in required):
                needs_restore.append(phase_name)

        if not needs_restore:
            return

        # Resolve the project export root from the (possibly partial)
        # objective_extraction output. Without it we cannot reconstruct
        # any Courseforge-export-rooted artifact.
        project_path = self._resolve_courseforge_project_path(phase_outputs)
        if project_path is None or not project_path.is_dir():
            logger.warning(
                "resume restore: %d completed phase(s) missing output "
                "keys but no resolvable project_path; leaving as-is "
                "(phases=%s)",
                len(needs_restore), needs_restore,
            )
            return

        logger.info(
            "resume restore: reconstructing on-disk outputs for "
            "completed-but-empty phase(s): %s (project=%s)",
            needs_restore, project_path,
        )

        reconstructed = self._reconstruct_resume_outputs(
            project_path, needs_restore,
        )

        for phase_name in needs_restore:
            recon = reconstructed.get(phase_name)
            if not recon:
                logger.warning(
                    "resume restore: could not reconstruct outputs for "
                    "phase '%s' from %s; leaving as-is (downstream "
                    "dependency check will surface the gap)",
                    phase_name, project_path,
                )
                continue
            recorded = phase_outputs[phase_name]
            # Fill only ABSENT keys; never overwrite recorded values or
            # the existing ``_completed`` / ``_gates_passed`` markers.
            for key, value in recon.items():
                if key.startswith("_"):
                    continue
                if not recorded.get(key):
                    recorded[key] = value
            recorded["_resume_restored"] = True
            logger.info(
                "resume restore: backfilled phase '%s' keys=%s",
                phase_name,
                sorted(k for k in recon if not k.startswith("_")),
            )

    def _reconstruct_resume_outputs(
        self, project_path: Path, phases: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Reconstruct per-phase output dicts from the project export.

        Best-effort, read-only. Returns ``{phase_name: {key: value}}``
        for whichever phases could be reconstructed; absent phases are
        simply omitted. Never raises (per-phase failures are
        warning-logged and skipped).
        """
        out: Dict[str, Dict[str, Any]] = {}

        config_data: Dict[str, Any] = {}
        config_path = project_path / "project_config.json"
        if config_path.exists():
            try:
                config_data = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as e:
                logger.warning(
                    "resume restore: project_config.json unreadable at "
                    "%s: %s", config_path, e,
                )
        course_name = config_data.get("course_name") or ""
        project_id = config_data.get("project_id") or project_path.name

        # ----- objective_extraction --------------------------------------
        if "objective_extraction" in phases:
            entry: Dict[str, Any] = {
                "project_id": project_id,
                "project_path": str(project_path),
            }
            ts = (
                project_path / "01_learning_objectives"
                / "textbook_structure.json"
            )
            if ts.exists():
                entry["textbook_structure_path"] = str(ts)
            out["objective_extraction"] = entry

        # ----- staging ---------------------------------------------------
        if "staging" in phases:
            staging_dir_str = config_data.get("staging_dir")
            if staging_dir_str and Path(staging_dir_str).is_dir():
                staging_dir = Path(staging_dir_str)
                staged = sorted(str(p) for p in staging_dir.glob("*.html"))
                out["staging"] = {
                    "staging_dir": str(staging_dir),
                    "staged_files": staged,
                    "file_count": len(staged),
                }

        # ----- source_mapping --------------------------------------------
        if "source_mapping" in phases:
            smap = project_path / "source_module_map.json"
            if smap.exists():
                out["source_mapping"] = {
                    "source_module_map_path": str(smap),
                }

        # ----- course_planning -------------------------------------------
        if "course_planning" in phases:
            synth = (
                project_path / "01_learning_objectives"
                / "synthesized_objectives.json"
            )
            if synth.exists():
                entry = {
                    "project_id": project_id,
                    "synthesized_objectives_path": str(synth),
                }
                # Recover objective_ids best-effort so downstream
                # trainforge_assessment routing resolves.
                try:
                    sdata = json.loads(synth.read_text(encoding="utf-8"))
                    ids: List[str] = []
                    for grp in ("terminal_objectives", "chapter_objectives"):
                        for o in sdata.get(grp) or []:
                            if isinstance(o, dict) and o.get("id"):
                                ids.append(o["id"])
                            elif isinstance(o, dict) and isinstance(
                                o.get("objectives"), list
                            ):
                                for sub in o["objectives"]:
                                    if isinstance(sub, dict) and sub.get("id"):
                                        ids.append(sub["id"])
                    if ids:
                        entry["objective_ids"] = ids
                except (OSError, ValueError):
                    pass
                out["course_planning"] = entry

        # ----- content_generation / content_generation_rewrite -----------
        content_dir = project_path / "03_content_development"
        for cg_phase in ("content_generation", "content_generation_rewrite"):
            if cg_phase in phases and content_dir.is_dir():
                pages = sorted(str(p) for p in content_dir.glob("**/*.html"))
                if pages:
                    out[cg_phase] = {
                        "project_id": project_id,
                        "content_dir": str(content_dir),
                        "content_paths": pages,
                        "page_paths": pages,
                    }

        # ----- packaging -------------------------------------------------
        if "packaging" in phases:
            final_dir = project_path / "05_final_package"
            pkg: Optional[Path] = None
            if final_dir.is_dir():
                # Prefer the course-named .imscc; fall back to any .imscc.
                named = final_dir / f"{course_name}.imscc"
                if named.exists():
                    pkg = named
                else:
                    candidates = sorted(final_dir.glob("*.imscc"))
                    if candidates:
                        pkg = candidates[0]
            if pkg is not None and pkg.exists():
                out["packaging"] = {
                    "project_id": project_id,
                    "package_path": str(pkg),
                    "libv2_package_path": str(pkg),
                    "imscc_path": str(pkg),
                    "content_dir": str(content_dir),
                }
            else:
                logger.warning(
                    "resume restore: no .imscc found under %s; cannot "
                    "reconstruct packaging output", final_dir,
                )

        return out

    def _dependencies_met(
        self, phase: WorkflowPhase, phase_outputs: Dict[str, Dict]
    ) -> bool:
        """Check that all phase dependencies have completed.

        Phase 3 Subtask 4: when a phase declares
        ``depends_on_when_env`` paired with
        ``depends_on_when_env_value`` and the predicate is satisfied,
        the alt list replaces ``depends_on`` for this check. Used by
        ``course_generation::packaging`` to switch from depending on
        the legacy ``content_generation`` to the rewrite tier
        ``content_generation_rewrite`` when ``COURSEFORGE_TWO_PASS=true``.
        """
        deps = self._effective_depends_on(phase)
        for dep in deps:
            dep_output = phase_outputs.get(dep, {})
            if not dep_output.get("_completed"):
                return False
        return True

    def _effective_depends_on(self, phase: WorkflowPhase) -> List[str]:
        """Resolve a phase's effective ``depends_on`` for the current env.

        Mirrors the env-aware switch in ``_dependencies_met``: when a
        phase declares ``depends_on_when_env`` paired with
        ``depends_on_when_env_value`` and the predicate is satisfied
        against the live environment, the alt list replaces the static
        ``depends_on``. Used by ``_topological_sort`` so the dispatch
        order matches the dependency check (Phase 3.5: packaging
        switches from depending on the legacy ``content_generation`` to
        the rewrite-tier ``post_rewrite_validation`` when
        ``COURSEFORGE_TWO_PASS=true``).
        """
        alt_pred = getattr(phase, "depends_on_when_env", None)
        alt_value = getattr(phase, "depends_on_when_env_value", None)
        if alt_pred and alt_value and self._eval_enabled_when_env(alt_pred):
            return list(alt_value)
        return list(phase.depends_on or [])

    def _topological_sort(self, phases: List[WorkflowPhase]) -> List[WorkflowPhase]:
        """
        Sort phases respecting depends_on ordering.

        Uses Kahn's algorithm for topological sort. Honors the same
        env-aware ``depends_on_when_env`` switch as ``_dependencies_met``
        so the queue order matches the dependency check at runtime.
        """
        phase_map = {p.name: p for p in phases}
        effective_deps = {p.name: self._effective_depends_on(p) for p in phases}
        in_degree = {p.name: 0 for p in phases}

        for name, deps in effective_deps.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_names = []

        while queue:
            # Pick the first available (stable sort)
            name = queue.pop(0)
            sorted_names.append(name)

            for other_name, deps in effective_deps.items():
                if name in deps:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0:
                        queue.append(other_name)

        # Detect circular dependencies
        if len(sorted_names) < len(phases):
            unresolved = {p.name for p in phases} - set(sorted_names)
            logger.error(f"Circular dependencies detected in phases: {unresolved}")
            raise ValueError(f"Circular dependencies detected: {unresolved}")

        return [phase_map[name] for name in sorted_names if name in phase_map]

    def _save_workflow_state(self, path: Path, state: Dict[str, Any]) -> None:
        """Persist workflow state to disk."""
        state["updated_at"] = datetime.now().isoformat()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save workflow state: {e}")
