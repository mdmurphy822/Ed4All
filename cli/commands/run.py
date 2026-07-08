"""
Canonical ``ed4all run`` CLI command (Wave 7).

This command is the single recommended entry point for running any Ed4All
workflow end-to-end. Wave 7 replaced the ad-hoc trio of
``ed4all textbook-to-course`` + ``create_textbook_pipeline_tool`` +
``run_textbook_pipeline_tool`` with a unified surface; Wave 28f
removed those predecessors entirely:

    ed4all run <workflow_name> [options]

Workflow names correspond to keys in ``config/workflows.yaml``
(``textbook_to_course``, ``course_generation``, ``rag_training``).
The command:

1. Parses CLI flags into workflow params.
2. Creates the workflow state (reuses the existing per-workflow creators).
3. Instantiates ``PipelineOrchestrator`` with the chosen mode + backend.
4. Calls ``.run(workflow_id)`` and streams / returns the result.

``--dry-run`` prints the planned phase sequence without executing anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from lib.generation import stop_control

logger = logging.getLogger(__name__)

#: Signals that request a graceful stop (AMENDMENT #11: ``signal.signal``, NOT
#: ``loop.add_signal_handler`` — the latter raises NotImplementedError on
#: Windows event loops). Precedent: ``cli/commands/mailbox_watch.py``.
_STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


# Mapping of canonical workflow names accepted by ``ed4all run`` to their
# "creator" functions (which produce a workflow_id + state JSON). Not every
# workflow has a creator wired up yet; ones that don't surface an error
# pointing the user at the stage-by-stage commands.
SUPPORTED_WORKFLOWS = {
    "textbook_to_course",
    "textbook-to-course",  # hyphenated alias
    "course_generation",
    "rag_training",
    # Wave 90 — post-import SLM adapter training stage. Generic
    # workflow path: ``ed4all run trainforge_train --course-name TST_101``
    # creates the workflow state via create_workflow_impl, then the
    # ``training`` phase dispatches Trainforge/train_course.py.
    "trainforge_train",
    # Phase 5 — Courseforge stage-by-stage subcommands. Each entry is
    # a thin alias over the underlying ``textbook_to_course`` workflow
    # state machine; the new ``courseforge-*`` subcommands re-execute
    # only their target Phase 3 tier(s) by pre-populating the upstream
    # phase_outputs (synthesised via ``_synthesize_outline_output`` —
    # Worker WB territory in ``MCP/core/workflow_runner.py``).
    "courseforge",
    "courseforge-outline",
    "courseforge-validate",
    "courseforge-rewrite",
}


# Phase 5 ST 3 — Courseforge stage-by-stage subcommands. These are
# thin aliases over the ``textbook_to_course`` workflow that re-execute
# only one or more Phase 3 tiers; the workflow runner pre-populates
# upstream phase_outputs so the legacy / completed phases skip and
# only the target tier executes.
COURSEFORGE_STAGE_SUBCOMMANDS = frozenset(
    {
        "courseforge",
        "courseforge_outline",
        "courseforge_validate",
        "courseforge_rewrite",
    }
)


def _normalize_workflow(name: str) -> str:
    return name.replace("-", "_").strip().lower()


DEFAULT_DART_OUTPUT_DIR = "DART/output"

# Phase 5 ST 1 — canonical 21-value Block-type enum from
# ``Courseforge/scripts/blocks.py:77``. Held here as a flat tuple so
# the CLI can validate ``--blocks`` tokens without importing the
# Courseforge module at parse time (avoids pulling the renderer
# dependency tree into ``ed4all run --help``). Re-validated against
# the canonical enum by the regression test in
# ``cli/tests/test_run_command.py``.
VALID_BLOCK_TYPES = (
    "objective",
    "concept",
    "example",
    "assessment_item",
    "explanation",
    "prereq_set",
    "activity",
    "misconception",
    "callout",
    "flip_card_grid",
    "self_check_question",
    "summary_takeaway",
    "reflection_prompt",
    "discussion_prompt",
    "chrome",
    "recap",
    "scenario",
    "problem",
    "vocab_card",
    "formula",
    "checklist",
    # I6 structured-content types.
    "table",
    "acronym",
    "key_idea",
    "resources",
    # IB5 framework-aligned pedagogical types (B02/B04/B05/B06) + the
    # FR-INT-01 B08 guided-practice fade block.
    "hook",
    "multimedia",
    "worked_example",
    "diagram",
    "guided_practice",
)


def _parse_blocks_filter(raw: Optional[str]) -> Optional[List[str]]:
    """Parse the ``--blocks`` flag into a list of block-type tokens.

    Phase 5 ST 1: accepts a comma-separated string of canonical
    ``Block.block_type`` values (NOT block IDs — block types from the
    21-singular ``BLOCK_TYPES`` enum at
    ``Courseforge/scripts/blocks.py:77``). Returns ``None`` when ``raw``
    is empty/None (caller treats that as "no filter — every block").

    Validation: each token MUST be in ``VALID_BLOCK_TYPES``; invalid
    tokens raise ``click.BadParameter`` with a friendly error listing
    valid tokens (Click formats it as a parse-time error and exits 2).

    Whitespace is stripped per token; empty tokens (e.g. trailing
    comma) are ignored. Duplicates are de-duplicated while preserving
    first-seen order so downstream consumers see a stable list.
    """
    if raw is None:
        return None
    parts = [tok.strip() for tok in raw.split(",")]
    parts = [tok for tok in parts if tok]
    if not parts:
        return None
    invalid = [tok for tok in parts if tok not in VALID_BLOCK_TYPES]
    if invalid:
        raise click.BadParameter(
            f"--blocks contains unknown block type(s): {invalid}. "
            f"Valid block types: {list(VALID_BLOCK_TYPES)}",
            param_hint="--blocks",
        )
    # De-duplicate while preserving order
    seen: set = set()
    out: List[str] = []
    for tok in parts:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _build_workflow_params(
    workflow: str,
    *,
    corpus: Optional[str],
    course_name: Optional[str],
    weeks: Optional[int],
    no_assessments: bool,
    assessment_count: int,
    bloom_levels: str,
    priority: str,
    objectives_path: Optional[str],
    skip_dart: bool = False,
    dart_output_dir: Optional[str] = None,
    reuse_objectives: Optional[str] = None,
    reuse_conversion: bool = False,
    target_block_ids: Optional[List[str]] = None,
    force_rerun: bool = False,
    courseforge_stage: Optional[str] = None,
    libv2_root: Optional[str] = None,
    skip_training: bool = False,
    provider: Optional[str] = None,
    stop_after: Optional[str] = None,
    license_note: Optional[str] = None,
    attribution: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the params dict for a workflow from CLI inputs.

    Wave 24 HIGH-6 / Wave 39: when --weeks is unset and workflow is
    textbook_to_course, we omit duration_weeks entirely so downstream
    phases auto-scale to max(8, chapter_count) once textbook_structure
    is known. Other workflows fall back to 12 (the historical default).
    """
    params: Dict[str, Any] = {
        "course_name": course_name,
        "generate_assessments": not no_assessments,
        "assessment_count": assessment_count,
        "bloom_levels": bloom_levels,
        "priority": priority,
    }
    if weeks is not None:
        params["duration_weeks"] = weeks
    elif workflow != "textbook_to_course":
        params["duration_weeks"] = 12
    # Wave 24: record whether --weeks was explicitly set. Downstream
    # phases (extract_textbook_structure / plan_course_structure) can
    # read this from project_config.json and auto-scale if needed.
    params["duration_weeks_explicit"] = weeks is not None

    if objectives_path:
        params["objectives_path"] = objectives_path

    if corpus:
        params["corpus"] = corpus
        # textbook_to_course expects pdf_paths specifically
        if workflow == "textbook_to_course":
            params["pdf_paths"] = corpus

    # Wave 74 Session 3: reuse existing DART HTML instead of re-converting.
    # When --skip-dart is set, the workflow runner synthesizes a
    # dart_conversion phase_output from the provided dir so staging's
    # inputs_from resolves without the phase actually executing.
    if skip_dart:
        params["skip_dart"] = True
        params["dart_output_dir"] = dart_output_dir or DEFAULT_DART_OUTPUT_DIR

    # Wave 80 Worker A: --reuse-objectives pins the course_planning phase
    # to a previously-synthesized objectives file instead of re-dispatching
    # the course-outliner subagent. Eliminates LLM-nondeterminism drift
    # across re-runs that breaks chunk learning_outcome_refs continuity.
    # The runner reads ``reuse_objectives_path`` and synthesizes the
    # course_planning phase output directly (Wave 74 --skip-dart pattern).
    if reuse_objectives:
        params["reuse_objectives_path"] = reuse_objectives

    # SemantiK §3.3a item 2: --reuse-conversion reuses a prior PDF→HTML
    # conversion instead of re-running the model-nondeterministic SemantiK v2
    # cascade (analogous to --reuse-objectives). The seam
    # ``_run_semantik_v2_conversion`` reads this (or ``ED4ALL_REUSE_CONVERSION``)
    # and, when prior ``{stem}_accessible.html`` + sidecars exist, skips the
    # cascade. Threaded as a bool param so the conversion phase honours it.
    if reuse_conversion:
        params["reuse_conversion"] = True

    # Phase 5 ST 1: --blocks plumbing. The CLI parses the comma-separated
    # block-type tokens via _parse_blocks_filter and threads them into
    # workflow params under ``target_block_ids``. The
    # ``content_generation_rewrite`` (and friends) phase reads this list
    # and re-rolls only blocks whose ``block_type`` is in the filter;
    # untouched blocks are byte-identical to the input.
    if target_block_ids:
        params["target_block_ids"] = list(target_block_ids)

    # Phase 5 ST 5: --force plumbing. When set, the workflow runner
    # ignores existing per-phase ``_completed`` checkpoints and
    # re-executes every phase that would otherwise short-circuit. Worker
    # WB on ``MCP/core/workflow_runner.py:860`` honours
    # ``force_rerun`` by stripping the ``_completed`` flag before the
    # phase loop's skip check fires.
    if force_rerun:
        params["force_rerun"] = True

    # Phase 5 ST 3: propagate the requested Courseforge stage subcommand
    # so the runner can pre-populate upstream phase_outputs and skip
    # everything but the target Phase 3 tier(s). Untyped here — the
    # runner-side validator (Worker WB) is the authoritative gatekeeper.
    if courseforge_stage:
        params["courseforge_stage"] = courseforge_stage

    # Phase 8 ST 3: surface the optional ``--libv2-root`` flag as a
    # workflow parameter so the runner's ``inputs_from`` chain can route
    # it into ``concept_extraction`` / ``chunking`` / ``imscc_chunking``
    # phases. Default unset → those helpers fall through to the env-var
    # / in-tree-default resolution chain in
    # ``MCP/tools/pipeline_tools.py::_resolve_libv2_root``.
    if libv2_root:
        params["libv2_root"] = libv2_root

    # --skip-training: suppress the training_synthesis phase (phase 16).
    # Required when the operator wants Qwen-generated assessments (phase 14)
    # but NOT Claude-generated training pairs — the canonical A/B audit
    # posture per plans/algebra-textbook-kg-test-2026-05.md. Without this
    # flag, training_synthesis fires as a Claude subagent and violates the
    # licensing posture documented in docs/LICENSING.md.
    if skip_training:
        params["skip_training"] = True

    # W1 Gap C (§2.3): thread the EXPLICIT run provider into params so the
    # runner's authoring-route fill (`_apply_authoring_route_env`) honors an
    # explicit `--api-provider` for the env-driven generation surfaces (not
    # just the orchestrator BackendSpec). Only an explicitly-passed provider
    # is threaded — the `_resolve_provider` `anthropic` fallback is NOT, so a
    # bare run with `--api-provider` unset still resolves the authoring envs
    # via `LLM_PROVIDER` > `local` (the documented single-switch default).
    # Injecting the fallback here would push `anthropic` onto the authoring
    # envs and regress the local default.
    if provider:
        params["provider"] = provider

    # Hosted-large build profile GAP-2 fix — thread --stop-after so the workflow
    # runner halts cleanly after the named phase completes (before later
    # phases run). Default unset → no behaviour change.
    if stop_after:
        params["stop_after"] = stop_after

    # B3: optional license/attribution declarations for the archived corpus.
    # Threaded into the libv2_archival phase (via the phase's workflow_params
    # inputs_from routing) where they land on course_manifest.json + drive the
    # NOTICE emit. Default unset → byte-identical manifest + no NOTICE.
    if license_note:
        params["license_note"] = license_note
    if attribution:
        params["attribution"] = attribution

    return params


def _discover_dart_htmls(dart_output_dir: str) -> List[Path]:
    """Return the list of ``*_accessible.html`` files in the given dir.

    Does not recurse — DART's canonical layout is flat.
    """
    root = Path(dart_output_dir)
    if not root.is_dir():
        return []
    return sorted(root.glob("*_accessible.html"))


def _corpus_pdf_basenames(corpus: Optional[str]) -> List[str]:
    """Return the list of PDF basenames (without .pdf) implied by ``corpus``.

    Accepts a single PDF path, a comma-separated list, or a directory.
    Silently returns an empty list when the corpus is None, not a PDF
    and not a directory — the caller treats ``[]`` as "no cross-check
    possible" rather than a hard failure.
    """
    if not corpus:
        return []
    # Directory
    path = Path(corpus)
    if path.is_dir():
        return [p.stem for p in sorted(path.glob("*.pdf"))]
    # Comma-separated or single file
    parts = [p.strip() for p in corpus.split(",") if p.strip()]
    stems: List[str] = []
    for part in parts:
        part_path = Path(part)
        if part_path.suffix.lower() == ".pdf":
            stems.append(part_path.stem)
    return stems


def _validate_skip_dart_inputs(
    *,
    dart_output_dir: str,
    corpus: Optional[str],
) -> Optional[str]:
    """Validate the --skip-dart inputs. Returns an error string on failure.

    * Dir must exist and contain at least one ``*_accessible.html`` file.
    * For each corpus PDF, emit a *warning* (not an error) when the
      matching ``{basename}_accessible.html`` is absent — caller may be
      running against a superset corpus or renamed outputs.
    """
    root = Path(dart_output_dir)
    if not root.is_dir():
        return (
            f"--skip-dart requires --dart-output-dir to point at an existing "
            f"directory; got: {dart_output_dir!r}"
        )
    htmls = _discover_dart_htmls(dart_output_dir)
    if not htmls:
        return (
            f"--skip-dart requires at least one ``*_accessible.html`` file "
            f"inside {dart_output_dir!r}; found none."
        )
    # Warn (don't fail) on corpus/output mismatches
    pdf_stems = _corpus_pdf_basenames(corpus)
    if pdf_stems:
        html_stems = {p.name.removesuffix("_accessible.html") for p in htmls}
        missing = [s for s in pdf_stems if s not in html_stems]
        if missing:
            click.secho(
                f"warning: --skip-dart is set but {len(missing)} corpus PDF(s) "
                f"have no matching ``*_accessible.html`` in "
                f"{dart_output_dir!r}: {missing}",
                fg="yellow",
            )
    return None


_LO_ID_PATTERN = __import__("re").compile(r"^[a-zA-Z]{2,}-\d{2,}$")


def _validate_reuse_objectives_file(path: str) -> Optional[str]:
    """Validate the ``--reuse-objectives`` file at parse time.

    Accepts either:
      * Courseforge synthesized form: ``terminal_objectives[]`` +
        ``chapter_objectives[]``.
      * Wave 75 LibV2 archive form: ``terminal_outcomes[]`` +
        ``component_objectives[]``.

    Returns ``None`` when the file is acceptable; otherwise returns a
    human-readable error string. Performs:

    * file-exists check
    * JSON parse
    * shape detection (at least one of the two recognised splits)
    * non-empty terminal list
    """
    p = Path(path)
    if not p.exists():
        return f"--reuse-objectives file not found: {path!r}"
    if not p.is_file():
        return f"--reuse-objectives must be a file (got a directory?): {path!r}"
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        return f"--reuse-objectives file unreadable: {path!r} ({e})"
    try:
        data = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return f"--reuse-objectives is not valid JSON: {path!r} ({e})"
    if not isinstance(data, dict):
        return (
            f"--reuse-objectives must be a JSON object at the top level; "
            f"got {type(data).__name__}"
        )

    # Detect shape
    has_courseforge = (
        isinstance(data.get("terminal_objectives"), list)
        or isinstance(data.get("chapter_objectives"), list)
    )
    has_libv2 = (
        isinstance(data.get("terminal_outcomes"), list)
        or isinstance(data.get("component_objectives"), list)
    )
    if not (has_courseforge or has_libv2):
        return (
            f"--reuse-objectives file does not match a recognised shape. "
            f"Expected either Courseforge form (terminal_objectives + "
            f"chapter_objectives) or LibV2 archive form (terminal_outcomes "
            f"+ component_objectives). Path: {path!r}"
        )

    # At least one terminal entry
    terminal: List[Any] = (
        data.get("terminal_objectives")
        or data.get("terminal_outcomes")
        or []
    )
    if not terminal:
        return (
            f"--reuse-objectives file has zero terminal objectives. "
            f"At least one terminal entry is required. Path: {path!r}"
        )

    return None


async def _create_textbook_workflow(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Delegate to the existing ``create_textbook_pipeline`` helper.

    Avoids duplicating all the state-setup boilerplate while we migrate.
    Returns the parsed JSON response.
    """
    from MCP.tools.pipeline_tools import create_textbook_pipeline

    # Wave 39 follow-up: propagate ``duration_weeks_explicit`` so the
    # extractor's auto-scale branch fires on real runs when ``--weeks``
    # was unset. Without this, omission of the key in params only
    # affects ``--dry-run`` output while live runs silently fell back
    # to the fixed 12-week default (PR #100 review finding).
    result = await create_textbook_pipeline(
        pdf_paths=params.get("pdf_paths", params.get("corpus", "")),
        course_name=params["course_name"],
        objectives_path=params.get("objectives_path"),
        duration_weeks=params.get("duration_weeks", 12),
        duration_weeks_explicit=params.get("duration_weeks_explicit", True),
        generate_assessments=params.get("generate_assessments", True),
        assessment_count=params.get("assessment_count", 50),
        bloom_levels=params.get("bloom_levels", "remember,understand,apply,analyze"),
        priority=params.get("priority", "normal"),
        skip_dart=bool(params.get("skip_dart", False)),
        dart_output_dir=params.get("dart_output_dir"),
        reuse_objectives_path=params.get("reuse_objectives_path"),
        reuse_conversion=bool(params.get("reuse_conversion", False)),
        courseforge_stage=params.get("courseforge_stage"),
        force_rerun=bool(params.get("force_rerun", False)),
        skip_training=bool(params.get("skip_training", False)),
        stop_after=params.get("stop_after"),
        provider=params.get("provider"),
        # Phase 5 ST 1 (--blocks): forward the parsed block-TYPE filter so
        # the rewrite phase additively evicts cached blocks of the named
        # types. Unset → None → byte-identical failure-driven cache reuse.
        target_block_ids=params.get("target_block_ids"),
    )
    return json.loads(result)


async def _create_generic_workflow(
    workflow: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a workflow through the orchestrator tools helper.

    For workflows that don't have a dedicated creator, we fall back to the
    generic ``create_workflow_impl`` path.
    """
    from MCP.tools.orchestrator_tools import create_workflow_impl

    raw = await create_workflow_impl(
        workflow_type=workflow,
        params=json.dumps(params),
        priority=params.get("priority", "normal"),
    )
    return json.loads(raw)


def _resolve_mode(mode: Optional[str]) -> str:
    if mode:
        return mode
    return os.environ.get("LLM_MODE", "local")


def _resolve_provider(provider: Optional[str]) -> str:
    if provider:
        return provider
    return os.environ.get("LLM_PROVIDER", "anthropic")


def _build_orchestrator(
    mode: str,
    *,
    provider: str,
    model: Optional[str],
):
    """Instantiate a PipelineOrchestrator for the chosen mode/provider."""
    from MCP.orchestrator import PipelineOrchestrator
    from MCP.orchestrator.llm_backend import BackendSpec

    spec = BackendSpec(
        mode=mode,
        provider=provider,
        model=model,
    )
    return PipelineOrchestrator(mode=mode, backend_spec=spec)


# ============================================================================
# Click command
# ============================================================================


@click.command("run")
@click.argument("workflow_name")
@click.option(
    "--corpus",
    type=click.Path(),
    help="Input material (PDF, directory of PDFs, IMSCC package)",
)
@click.option(
    "--course-name",
    help="Course identifier (e.g., PHYS_101). Required for most workflows.",
)
@click.option(
    "--mode",
    type=click.Choice(["local", "api"]),
    default=None,
    help="Execution mode. Default: env LLM_MODE or 'local'.",
)
@click.option(
    "--api-provider",
    "--provider",
    "api_provider",
    type=click.Choice(["anthropic", "openai", "nvidia"]),
    default=None,
    help=(
        "LLM provider for the run. ``anthropic`` / ``openai`` are the api-mode "
        "backends; ``nvidia`` (the vendor endpoint-registry key) selects the "
        "hosted large-model build profile (redirects the two-pass "
        "block-routing + synthesis seat to the hosted large seat; the training "
        "seat stays local for licensing). Also settable via env LLM_PROVIDER. "
        "Default: env LLM_PROVIDER or 'anthropic'. NB: ``nvidia`` requires "
        "COURSEFORGE_TWO_PASS=true and a NVIDIA_API_KEY; run --dry-run first "
        "for the routing preflight."
    ),
)
@click.option(
    "--model",
    help="Model identifier override. Default: per-provider default.",
)
@click.option(
    "--weeks",
    type=int,
    default=None,
    help=(
        "Course duration in weeks (workflow-dependent). When unset for "
        "textbook-to-course, auto-scales to max(8, chapter_count) once the "
        "textbook structure is known; otherwise defaults to 12."
    ),
)
@click.option(
    "--no-assessments",
    is_flag=True,
    help="Skip the Trainforge assessment phase (where applicable)",
)
@click.option(
    "--assessment-count",
    type=int,
    default=50,
    help="Number of questions to generate when assessments are enabled",
)
@click.option(
    "--bloom-levels",
    default="remember,understand,apply,analyze",
    help="Comma-separated Bloom taxonomy levels to target",
)
@click.option(
    "--priority",
    type=click.Choice(["low", "normal", "high"]),
    default="normal",
)
@click.option(
    "--objectives",
    type=click.Path(),
    help="Optional path to a learning-objectives file to merge",
)
@click.option(
    "--resume",
    "resume_run_id",
    default=None,
    help="Resume a prior run from its last checkpoint (provide run_id)",
)
@click.option(
    "--skip-dart",
    is_flag=True,
    default=False,
    help=(
        "Skip the dart_conversion phase and reuse existing DART HTML output. "
        "Useful when re-running textbook-to-course after tweaking downstream "
        "phases. Requires --dart-output-dir to contain ``*_accessible.html`` "
        "files (defaults to DART/output/)."
    ),
)
@click.option(
    "--dart-output-dir",
    type=click.Path(),
    default=None,
    help=(
        "Directory containing ``*_accessible.html`` files. Only consulted "
        "when --skip-dart is set. Defaults to DART/output/."
    ),
)
@click.option(
    "--reuse-objectives",
    type=click.Path(),
    default=None,
    help=(
        "Pin the course_planning phase to a previously-synthesized "
        "objectives JSON instead of re-dispatching the course-outliner "
        "subagent. Accepts Courseforge synthesized form "
        "(terminal_objectives + chapter_objectives) or Wave 75 LibV2 "
        "archive form (terminal_outcomes + component_objectives). Used "
        "for stable LO regens that need to preserve continuity with the "
        "existing chunks' learning_outcome_refs."
    ),
)
@click.option(
    "--reuse-conversion",
    "reuse_conversion",
    is_flag=True,
    default=False,
    help=(
        "SemantiK §3.3a — reuse a prior PDF→accessible-HTML conversion "
        "instead of re-running the (model-nondeterministic) SemantiK v2 "
        "cascade. When set AND prior ``{stem}_accessible.html`` + sidecars "
        "exist at the conversion output path, the cascade is skipped and the "
        "prior artifacts are reused (the model-nondeterminism guarantee on "
        "re-runs, analogous to --reuse-objectives). Mirrors the "
        "``ED4ALL_REUSE_CONVERSION`` env var (flag wins when both are set)."
    ),
)
@click.option(
    "--blocks",
    "blocks_filter",
    default=None,
    help=(
        "Phase 5 — comma-separated list of canonical Block types to "
        "filter on (per-block re-execution scope). Tokens must be from "
        "the 21-singular ``BLOCK_TYPES`` enum at "
        "``Courseforge/scripts/blocks.py`` (e.g. "
        "``--blocks assessment_item,example``). When set, Phase 5 "
        "rewrite-tier handlers re-roll only blocks whose ``block_type`` "
        "is in the filter; every other block is byte-identical to the "
        "input. Unknown tokens fail fast with a list of valid types."
    ),
)
@click.option(
    "--force",
    "force_rerun",
    is_flag=True,
    default=False,
    help=(
        "Phase 5 — re-run every phase even if a per-phase ``_completed`` "
        "checkpoint exists. Useful for the Phase 5 stage subcommands "
        "(``courseforge-outline`` / ``-validate`` / ``-rewrite``) when "
        "the operator wants to overwrite a prior completion. The "
        "workflow runner strips the ``_completed`` flag before its "
        "phase-skip check fires."
    ),
)
@click.option(
    "--libv2-root",
    "libv2_root",
    type=click.Path(),
    default=None,
    help=(
        "Phase 8 ST 3 — override the LibV2 root directory used by the "
        "Phase 6/7 chunkset + concept-graph helpers (concept_extraction, "
        "chunking, imscc_chunking) when persisting per-course artifacts "
        "under ``<libv2_root>/courses/<course_slug>/``. Resolution chain: "
        "this flag > ``ED4ALL_LIBV2_ROOT`` env var > the default in-tree "
        "``LibV2/`` directory. Useful for ops topologies that mount LibV2 "
        "at a non-default location (Docker volume / NFS / ConfigMap)."
    ),
)
@click.option(
    "--skip-training",
    "skip_training",
    is_flag=True,
    default=False,
    help=(
        "Suppress the training_synthesis phase (phase 16). Use when "
        "Trainforge assessment generation (phase 14) should run via a "
        "license-clean provider (e.g. local Qwen) but training-pair "
        "synthesis must not fire — the canonical A/B audit posture. "
        "Without this flag training_synthesis dispatches as a Claude "
        "subagent and violates the licensing posture in docs/LICENSING.md."
    ),
)
@click.option(
    "--stop-after",
    "stop_after",
    default=None,
    help=(
        "Halt the run cleanly AFTER the named phase completes, skipping all "
        "subsequent phases. E.g. ``--stop-after imscc_chunking`` stops before "
        "trainforge_assessment / training_synthesis / libv2_archival / "
        "finalization run — the canonical 'build a retrieval-ready course "
        "without training synthesis' slice. The phase name is validated "
        "against the workflow's phase list (unknown name → error). Default "
        "unset → runs to completion."
    ),
)
@click.option(
    "--license-note",
    "license_note",
    default=None,
    help=(
        "B3 — optional license declaration for the archived source corpus "
        "(e.g. ``--license-note 'CC-BY-4.0'``). Recorded on the LibV2 "
        "``course_manifest.json`` under ``license.spdx_or_name`` and mirrored "
        "into a human-readable NOTICE file in the course dir. Absent → the "
        "manifest + course dir are byte-identical to the pre-B3 layout."
    ),
)
@click.option(
    "--attribution",
    "attribution",
    default=None,
    help=(
        "B3 — optional source-attribution statement for the archived corpus "
        "(e.g. ``--attribution 'Access for free at openstax.org'``). Recorded "
        "on the manifest under ``attribution.statement`` and mirrored into the "
        "NOTICE file. Absent → byte-identical manifest."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the planned pipeline without executing",
)
@click.option(
    "--watch",
    is_flag=True,
    help="Stream phase transitions to stdout (Wave 7: logs only; token "
    "streaming lands in a later wave)",
)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output")
@click.pass_context
def run_command(
    ctx: click.Context,
    workflow_name: str,
    corpus: Optional[str],
    course_name: Optional[str],
    mode: Optional[str],
    api_provider: Optional[str],
    model: Optional[str],
    weeks: Optional[int],
    no_assessments: bool,
    assessment_count: int,
    bloom_levels: str,
    priority: str,
    objectives: Optional[str],
    resume_run_id: Optional[str],
    skip_dart: bool,
    dart_output_dir: Optional[str],
    reuse_objectives: Optional[str],
    reuse_conversion: bool,
    blocks_filter: Optional[str],
    force_rerun: bool,
    libv2_root: Optional[str],
    skip_training: bool,
    stop_after: Optional[str],
    license_note: Optional[str],
    attribution: Optional[str],
    dry_run: bool,
    watch: bool,
    output_json: bool,
) -> None:
    """Run an Ed4All workflow end-to-end (canonical entry point).

    Example:

    \b
        ed4all run textbook-to-course --corpus textbook.pdf --course-name PHYS_101
        ed4all run textbook-to-course --corpus ./pdfs/ --course-name BIO_201 --weeks 16
        ed4all run rag_training --corpus course.imscc --course-name CHEM_101

    Modes:

    \b
      local  (default)  Uses the current Claude Code session; no API key needed.
      api               Uses the Anthropic SDK directly; needs ANTHROPIC_API_KEY.

    See ``config/workflows.yaml`` for the full list of available workflows.
    """
    workflow = _normalize_workflow(workflow_name)

    if workflow not in {_normalize_workflow(w) for w in SUPPORTED_WORKFLOWS}:
        click.secho(
            f"Unknown workflow: {workflow_name}. "
            f"Choose from: {sorted(SUPPORTED_WORKFLOWS)}",
            fg="red",
        )
        sys.exit(2)

    # Phase 5 ST 3: track which Courseforge stage subcommand the user
    # requested (if any) before we alias the workflow back to
    # ``textbook_to_course``. The runner reads ``courseforge_stage``
    # from workflow params to know which Phase 3 tier to re-execute
    # while skipping the others via ``_synthesize_outline_output``
    # (Worker WB territory in MCP/core/workflow_runner.py).
    courseforge_stage: Optional[str] = None
    if workflow in COURSEFORGE_STAGE_SUBCOMMANDS:
        courseforge_stage = workflow
        workflow = "textbook_to_course"

    mode = _resolve_mode(mode)
    provider = _resolve_provider(api_provider)

    # Wave 74 Session 3: validate --skip-dart inputs BEFORE building
    # params so CLI fails fast with a clear error. --skip-dart is
    # currently only honoured by textbook_to_course; other workflows
    # don't have a dart_conversion phase to elide.
    if skip_dart and workflow != "textbook_to_course":
        click.secho(
            f"--skip-dart is only supported for workflow 'textbook_to_course'; "
            f"got '{workflow}'.",
            fg="red",
        )
        sys.exit(2)
    effective_dart_output_dir = dart_output_dir or (
        DEFAULT_DART_OUTPUT_DIR if skip_dart else None
    )
    if skip_dart:
        err = _validate_skip_dart_inputs(
            dart_output_dir=effective_dart_output_dir or DEFAULT_DART_OUTPUT_DIR,
            corpus=corpus,
        )
        if err:
            click.secho(err, fg="red")
            sys.exit(2)

    # Wave 80 Worker A: validate --reuse-objectives at parse time so
    # malformed / missing files fail fast with a clear error before any
    # workflow state is created.
    if reuse_objectives:
        if workflow != "textbook_to_course":
            click.secho(
                f"--reuse-objectives is only supported for workflow "
                f"'textbook_to_course'; got '{workflow}'.",
                fg="red",
            )
            sys.exit(2)
        err = _validate_reuse_objectives_file(reuse_objectives)
        if err:
            click.secho(err, fg="red")
            sys.exit(2)

    # Phase 5 ST 1: parse and validate the --blocks filter. Invalid
    # tokens raise click.BadParameter (Click prints + exits 2 itself).
    target_block_ids = _parse_blocks_filter(blocks_filter)

    params = _build_workflow_params(
        workflow,
        corpus=corpus,
        course_name=course_name,
        weeks=weeks,
        no_assessments=no_assessments,
        assessment_count=assessment_count,
        bloom_levels=bloom_levels,
        priority=priority,
        objectives_path=objectives,
        skip_dart=skip_dart,
        dart_output_dir=effective_dart_output_dir,
        reuse_objectives=reuse_objectives,
        reuse_conversion=reuse_conversion,
        target_block_ids=target_block_ids,
        force_rerun=force_rerun,
        courseforge_stage=courseforge_stage,
        libv2_root=libv2_root,
        skip_training=skip_training,
        stop_after=stop_after,
        license_note=license_note,
        attribution=attribution,
        # Pass the EXPLICIT flag (None when unset), not the
        # `_resolve_provider` anthropic-fallback value, so an unset provider
        # leaves the authoring-env resolution to LLM_PROVIDER > local.
        provider=api_provider,
    )

    # -------- dry-run: plan only, no side effects ------------------------
    if dry_run:
        plan = _dry_run_plan(workflow, params, mode=mode, provider=provider)
        if output_json:
            click.echo(json.dumps(plan, indent=2, default=str))
        else:
            _print_dry_run_plan(plan)
        return

    # -------- resume path ------------------------------------------------
    if resume_run_id:
        _resume_workflow(
            workflow_id=resume_run_id,
            mode=mode,
            provider=provider,
            model=model,
            output_json=output_json,
            watch=watch,
            # Trap-2 hardening: an explicit --stop-after on the resume
            # command overrides the persisted halt point.
            stop_after=stop_after,
            # I3: an explicit --reuse-objectives on resume pins the resumed
            # course_planning phase (validated at parse time above).
            reuse_objectives=reuse_objectives,
        )
        return

    # -------- create + run -----------------------------------------------
    if not course_name:
        click.secho(
            "--course-name is required unless --dry-run or --resume is used.",
            fg="red",
        )
        sys.exit(2)

    if (
        not corpus
        and workflow in {"textbook_to_course", "rag_training"}
        and not courseforge_stage
    ):
        click.secho(
            f"--corpus is required for workflow '{workflow}'.",
            fg="red",
        )
        sys.exit(2)

    exit_code = asyncio.run(
        _create_and_run(
            workflow=workflow,
            params=params,
            mode=mode,
            provider=provider,
            model=model,
            output_json=output_json,
            watch=watch,
        )
    )
    if exit_code:
        sys.exit(exit_code)


# ============================================================================
# Helpers
# ============================================================================


def _dry_run_plan(
    workflow: str,
    params: Dict[str, Any],
    *,
    mode: str,
    provider: str,
) -> Dict[str, Any]:
    """Build a dry-run plan dict (no side effects)."""
    try:
        from MCP.core.config import OrchestratorConfig

        config = OrchestratorConfig.load()
        wf = config.get_workflow(workflow)
        if wf is None:
            return {
                "workflow": workflow,
                "mode": mode,
                "provider": provider,
                "params": params,
                "error": f"Unknown workflow: {workflow}",
                "phases": [],
            }

        # Topologically sort phases (reuse WorkflowRunner logic)
        from MCP.core.workflow_runner import WorkflowRunner

        runner = WorkflowRunner(executor=None, config=config)
        sorted_phases = runner._topological_sort(wf.phases)

        # Respect --no-assessments by pruning the optional phase
        skip_trainforge = not params.get("generate_assessments", True)
        skip_training_flag = bool(params.get("skip_training", False))
        skip_dart_flag = bool(params.get("skip_dart", False))
        reuse_objectives_path = params.get("reuse_objectives_path")
        # Phase 5 ST 6: --blocks filter annotation. Only the rewrite-tier
        # phase consumes ``target_block_ids`` per Phase 5 §3 selection
        # algorithm ("rewrite tier only; validate ignores --blocks"). When
        # the filter is set, the rewrite phase is annotated with
        # ``<FILTERED:type1,type2>`` mirroring the existing ``<REUSED>``
        # annotation precedent at ``_dry_run_plan`` for --reuse-objectives.
        target_block_ids = params.get("target_block_ids")
        force_rerun_flag = bool(params.get("force_rerun", False))
        # Hosted-large build profile GAP-2 — the --stop-after halt phase
        # (annotated in the dry-run plan; phases after it are marked
        # SKIPPED-after-stop).
        stop_after_phase = str(params.get("stop_after", "") or "").strip()
        stop_reached = False
        # Phases that consume target_block_ids (single-source-of-truth list
        # for the dry-run annotation). Kept narrow because plan §3
        # explicitly scopes selection to the rewrite tier.
        block_filtered_phases = {"content_generation_rewrite"}
        phases = []
        for idx, phase in enumerate(sorted_phases):
            if skip_trainforge and phase.name == "trainforge_assessment":
                continue
            if skip_training_flag and phase.name == "training_synthesis":
                continue
            phase_entry = {
                "order": len(phases) + 1,
                "name": phase.name,
                "agents": list(phase.agents),
                "max_concurrent": getattr(phase, "max_concurrent", 5),
                "depends_on": list(phase.depends_on or []),
                "optional": bool(getattr(phase, "optional", False)),
            }
            # Wave 74 Session 3: mark dart_conversion as SKIPPED in the
            # dry-run plan when --skip-dart is set. The phase is still
            # listed so ordering is transparent, but its status reflects
            # that the runner will synthesize outputs from an existing
            # DART/output/ directory instead of executing it.
            if skip_dart_flag and phase.name == "dart_conversion":
                phase_entry["status"] = "SKIPPED"
                phase_entry["skip_reason"] = (
                    f"--skip-dart set; reusing HTML from "
                    f"{params.get('dart_output_dir', DEFAULT_DART_OUTPUT_DIR)!r}"
                )
            # Wave 80 Worker A: mark course_planning as REUSED in the
            # dry-run plan when --reuse-objectives is set. The phase is
            # still listed so ordering is transparent, but its status
            # reflects that the runner will copy the user's objectives
            # file into the project's 01_learning_objectives/ dir
            # instead of dispatching the course-outliner subagent.
            if reuse_objectives_path and phase.name == "course_planning":
                phase_entry["status"] = "REUSED"
                phase_entry["reuse_reason"] = (
                    f"--reuse-objectives set; reusing LOs from "
                    f"{reuse_objectives_path!r}"
                )
            # Phase 5 ST 6: --blocks filter annotation. Mark phases that
            # consume ``target_block_ids`` with ``<FILTERED:...>`` so the
            # operator sees which phases will run on a per-block scope.
            # Mirrors the ``<REUSED>`` precedent above.
            if target_block_ids and phase.name in block_filtered_phases:
                phase_entry["status"] = "FILTERED"
                phase_entry["blocks_filter"] = list(target_block_ids)
                phase_entry["filter_reason"] = (
                    f"--blocks set; re-rolling only block_type(s) "
                    f"{list(target_block_ids)!r}"
                )
            # Hosted-large build profile GAP-2 — --stop-after annotation. Phases
            # AFTER the named stop phase are marked as not-running; the stop
            # phase itself is the last to execute.
            if stop_after_phase:
                if stop_reached:
                    phase_entry["status"] = "SKIPPED_AFTER_STOP"
                    phase_entry["skip_reason"] = (
                        f"--stop-after {stop_after_phase!r}; halts before this "
                        f"phase"
                    )
                elif phase.name == stop_after_phase:
                    phase_entry["stop_after"] = True
                    stop_reached = True
            phases.append(phase_entry)

        plan_dict: Dict[str, Any] = {
            "workflow": workflow,
            "mode": mode,
            "provider": provider,
            "params": params,
            "phases": phases,
        }
        # Phase 5 ST 6: top-level summary fields for --blocks /
        # --force so JSON consumers see the run-wide flags without
        # re-reading params.
        if target_block_ids:
            plan_dict["blocks_filter"] = list(target_block_ids)
        if force_rerun_flag:
            plan_dict["force_rerun"] = True
        if stop_after_phase:
            plan_dict["stop_after"] = stop_after_phase
        # Hosted-large build profile SETUP — preflight ("wired but not firing"
        # proof). When --provider nvidia (the vendor endpoint-registry key),
        # resolve+assert the routing for every build phase WITHOUT dispatching.
        # Makes NO network call. Emit under the new ``cloud_seat_preflight`` key
        # AND the legacy ``nvidia_preflight`` key for one release so existing
        # dry-run consumers + doctor post-mortems keep reading it.
        if provider == "nvidia":
            _pf = _cloud_seat_preflight(params)
            plan_dict["cloud_seat_preflight"] = _pf
            plan_dict["nvidia_preflight"] = _pf
        return plan_dict
    except Exception as exc:  # noqa: BLE001 — dry-run shouldn't explode
        return {
            "workflow": workflow,
            "mode": mode,
            "provider": provider,
            "params": params,
            "error": f"plan build failed: {exc}",
            "phases": [],
        }


def _cloud_seat_preflight(params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve + assert the hosted large-model build-profile routing.

    The "wired but not firing" proof. Computes what every build seat WOULD
    resolve to under ``--provider nvidia`` (the vendor endpoint-registry key)
    + ``COURSEFORGE_TWO_PASS=true`` (using the same module constants the
    workflow runner uses), and emits pass/warn/error assertions WITHOUT making
    any network call or requiring a real key value (presence-only). Never
    dispatches to the cloud seat.

    Asserts:
      - NVIDIA_API_KEY present (presence only — no call).
      - the rewrite + synthesis tiers resolve to nvidia + the hosted large
        model (catch the 30B-nano leak).
      - TRAINFORGE_SYNTHESIS_PROVIDER resolves LOCAL (gap-3 licensing guard).
      - ED4ALL_ANSWER_PROVIDER resolves loopback.
      - WARN on a stale ``COURSEFORGE_*_PROVIDER=local`` env that would
        silently win over the nvidia YAML.
    """
    from MCP.core import workflow_runner as _wr

    checks: List[Dict[str, str]] = []

    def _check(level: str, name: str, detail: str) -> None:
        checks.append({"level": level, "name": name, "detail": detail})

    two_pass = str(os.environ.get("COURSEFORGE_TWO_PASS", "")).strip().lower() in (
        "1", "true", "yes", "on",
    )
    if not two_pass:
        _check(
            "error",
            "two_pass",
            "COURSEFORGE_TWO_PASS is not truthy — the nvidia routing branch "
            "only fires on the two-pass surface. Export COURSEFORGE_TWO_PASS=true.",
        )

    # 1. NVIDIA_API_KEY presence (NOT value, NEVER a call).
    if os.environ.get("NVIDIA_API_KEY", "").strip():
        _check("pass", "nvidia_api_key", "NVIDIA_API_KEY is present (presence-only check).")
    else:
        _check(
            "error",
            "nvidia_api_key",
            "NVIDIA_API_KEY is not set — required for the nvidia rewrite + "
            "synthesis tiers (no value/call made here, presence only).",
        )

    # 2. Rewrite tier — the block-routing YAML the branch would select.
    routing_path = os.environ.get("COURSEFORGE_BLOCK_ROUTING_PATH", "").strip()
    expected_routing = _wr._HOSTED_LARGE_BLOCK_ROUTING_PATH
    if routing_path and routing_path != expected_routing:
        _check(
            "warn",
            "block_routing_path",
            f"COURSEFORGE_BLOCK_ROUTING_PATH is pinned to {routing_path!r} "
            f"(would override the nvidia YAML {expected_routing!r}).",
        )
    else:
        try:
            from MCP.core.config import OrchestratorConfig  # noqa: F401
            from Courseforge.router.policy import load_block_routing_policy
            from pathlib import Path as _Path

            project_root = _Path(_wr.__file__).resolve().parents[2]
            policy = load_block_routing_policy(project_root / expected_routing)
            large = policy.capability_tiers.get("large", {}) if not policy.is_empty() else {}
            if str(large.get("provider")) == "nvidia":
                _check(
                    "pass",
                    "rewrite_tier",
                    f"rewrite 'large' tier resolves provider=nvidia via {expected_routing}.",
                )
            else:
                _check(
                    "error",
                    "rewrite_tier",
                    f"rewrite 'large' tier did NOT resolve nvidia in {expected_routing} "
                    f"(got {large.get('provider')!r}).",
                )
        except Exception as exc:  # noqa: BLE001 — preflight never crashes a plan
            _check("warn", "rewrite_tier", f"could not load nvidia routing YAML: {exc}")

    # 3. NVIDIA_LARGE_MODEL — catch the 30B-nano registry-default leak.
    large_model = os.environ.get("NVIDIA_LARGE_MODEL", "").strip()
    expected_model = _wr._HOSTED_LARGE_MODEL_DEFAULT
    resolved_model = large_model or expected_model  # branch setdefaults the large model
    if "nano" in resolved_model.lower() or "30b" in resolved_model.lower():
        _check(
            "error",
            "cloud_model",
            f"NVIDIA_LARGE_MODEL resolves to {resolved_model!r} — the 30B-nano "
            f"leak. The branch setdefaults the hosted large model "
            f"({expected_model!r}); a stale export is overriding it.",
        )
    else:
        _check(
            "pass",
            "cloud_model",
            f"cloud model resolves to {resolved_model!r} (large model, not the 30B nano).",
        )

    # 4. Synthesis seat — must resolve nvidia + the large model (GAP-1 surface).
    synth_provider = os.environ.get("TEXTBOOK_SYNTHESIS_PROVIDER", "").strip()
    # The branch setdefaults nvidia when unset; an explicit value wins.
    resolved_synth = synth_provider or "nvidia"
    if resolved_synth == "nvidia":
        _check(
            "pass",
            "synthesis_tier",
            "TEXTBOOK_SYNTHESIS_PROVIDER resolves nvidia (objective_extraction "
            "/ course_planning / concept_extraction reach the hosted large seat).",
        )
    else:
        _check(
            "warn",
            "synthesis_tier",
            f"TEXTBOOK_SYNTHESIS_PROVIDER is pinned to {resolved_synth!r} — the "
            f"synthesis phases would NOT route to nvidia (GAP-1 surface).",
        )

    # 5. GAP-3 licensing guard — TRAINFORGE_SYNTHESIS_PROVIDER must be LOCAL.
    train_provider = os.environ.get("TRAINFORGE_SYNTHESIS_PROVIDER", "").strip().lower()
    # The branch leaves it untouched; A5 defaults-on setdefaults it local.
    resolved_train = train_provider or "local"
    if resolved_train == "nvidia":
        _check(
            "error",
            "training_seat_licensing",
            "TRAINFORGE_SYNTHESIS_PROVIDER resolves nvidia — the SLM training "
            "corpus must NEVER route through Llama-3.3 (docs/LICENSING.md). The "
            "nvidia branch leaves this local; a stale export is overriding it.",
        )
    else:
        _check(
            "pass",
            "training_seat_licensing",
            f"TRAINFORGE_SYNTHESIS_PROVIDER resolves {resolved_train!r} (local — "
            f"training corpus stays off the cloud tier).",
        )

    # 6. Answer path — must resolve loopback (Phase IA wall).
    answer_provider = os.environ.get("ED4ALL_ANSWER_PROVIDER", "").strip().lower()
    resolved_answer = answer_provider or "local"
    if resolved_answer == "local":
        _check(
            "pass",
            "answer_loopback",
            "ED4ALL_ANSWER_PROVIDER resolves local/loopback (grounded-answer "
            "path never reaches the cloud).",
        )
    else:
        _check(
            "warn",
            "answer_loopback",
            f"ED4ALL_ANSWER_PROVIDER is {resolved_answer!r} — verify it resolves "
            f"to a loopback base_url (Phase IA: no cloud arm on the answer path).",
        )

    # 7. WARN on a stale COURSEFORGE_*_PROVIDER=local that would silently win.
    for env_name in ("COURSEFORGE_REWRITE_PROVIDER", "COURSEFORGE_OUTLINE_PROVIDER",
                     "COURSEFORGE_PROVIDER"):
        val = os.environ.get(env_name, "").strip().lower()
        if val == "local":
            _check(
                "warn",
                "stale_provider_env",
                f"{env_name}=local is exported — it would silently win over the "
                f"nvidia YAML (setdefault honors the explicit export). Unset it "
                f"for a pure-nvidia build.",
            )

    has_error = any(c["level"] == "error" for c in checks)
    has_warn = any(c["level"] == "warn" for c in checks)
    return {
        "provider": "nvidia",
        "verdict": "FAIL" if has_error else ("WARN" if has_warn else "PASS"),
        "checks": checks,
        "note": "No NVIDIA dispatch — routing resolved + asserted only.",
    }


# Deprecated compat alias — ``doctor`` imports this name and older callers /
# tests monkeypatch it. Points at the renamed ``_cloud_seat_preflight``.
_nvidia_preflight = _cloud_seat_preflight


def _print_dry_run_plan(plan: Dict[str, Any]) -> None:
    click.secho("Dry run — planned execution:", fg="cyan")
    click.echo(f"  Workflow:  {plan['workflow']}")
    click.echo(f"  Mode:      {plan['mode']}")
    click.echo(f"  Provider:  {plan['provider']}")
    if plan.get("error"):
        click.secho(f"  Error:     {plan['error']}", fg="red")
        return
    if plan.get("params", {}).get("course_name"):
        click.echo(f"  Course:    {plan['params']['course_name']}")
    if plan.get("params", {}).get("corpus"):
        click.echo(f"  Corpus:    {plan['params']['corpus']}")
    # Phase 5 ST 6: surface --blocks / --force at the top level so
    # operators see the run-wide flags without scanning every phase.
    if plan.get("blocks_filter"):
        click.echo(f"  Blocks:    {plan['blocks_filter']}")
    if plan.get("force_rerun"):
        click.echo(f"  Force:     re-run completed phases (--force)")
    if plan.get("stop_after"):
        click.echo(f"  StopAfter: {plan['stop_after']} (halts before later phases)")
    # Hosted-large build profile preflight summary. Read the new key first,
    # fall back to the legacy ``nvidia_preflight`` key (older plan dicts).
    preflight = plan.get("cloud_seat_preflight") or plan.get("nvidia_preflight")
    if preflight:
        verdict = preflight.get("verdict", "?")
        color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(verdict, "white")
        click.echo()
        click.secho(f"Hosted large-seat preflight: {verdict}", fg=color, bold=True)
        click.secho(f"  ({preflight.get('note', '')})", fg="white")
        for chk in preflight.get("checks", []):
            lvl = chk.get("level", "")
            sym = {"pass": "OK ", "warn": "!! ", "error": "XX "}.get(lvl, "   ")
            lcolor = {"pass": "green", "warn": "yellow", "error": "red"}.get(lvl, "white")
            click.secho(f"  {sym}{chk.get('name')}: {chk.get('detail')}", fg=lcolor)
    click.echo()
    click.secho("Phases:", fg="cyan")
    for phase in plan.get("phases", []):
        agents = ", ".join(phase["agents"])
        status = phase.get("status")
        # Phase 5 ST 6: status suffix carries the FILTERED block list
        # inline so a quick eyeball of the dry-run output shows which
        # phase will run scoped to which block types.
        if status == "FILTERED" and phase.get("blocks_filter"):
            status_suffix = f"  <FILTERED:{','.join(phase['blocks_filter'])}>"
        elif status:
            status_suffix = f"  <{status}>"
        else:
            status_suffix = ""
        click.echo(
            f"  {phase['order']}. {phase['name']}"
            f"  [agents={agents}, max_concurrent={phase['max_concurrent']}]"
            f"{status_suffix}"
        )
        skip_reason = phase.get("skip_reason")
        if skip_reason:
            click.echo(f"      reason: {skip_reason}")
        reuse_reason = phase.get("reuse_reason")
        if reuse_reason:
            click.echo(f"      reason: {reuse_reason}")
        filter_reason = phase.get("filter_reason")
        if filter_reason:
            click.echo(f"      reason: {filter_reason}")


def _any_gate_failed(result) -> bool:
    """Return True if any phase reported ``gates_passed=False``.

    Wave 29 Defect 3: phase_results is a ``{phase_name: {..., gates_passed:
    bool, ...}}`` mapping produced by ``WorkflowRunner.run_workflow``.
    The top-level workflow status can read ``COMPLETE`` even when gates
    failed (``optional`` phases bypass the stop-on-fail check), so we
    scan every phase directly.
    """
    if not result or not getattr(result, "phase_results", None):
        return False
    for info in result.phase_results.values():
        if not isinstance(info, dict):
            continue
        # ``gates_passed`` key may be absent on phases that emitted no
        # gates — treat absence as pass.
        if info.get("gates_passed") is False:
            return True
    return False


def _resolve_run_id_for_workflow(workflow_id: str) -> str:
    """Resolve the run_id the orchestrator/runner will use for a workflow.

    Mirrors ``PipelineOrchestrator._build_runner``: prefer the workflow's
    persisted ``params.run_id`` (textbook pipelines mint ``TTC_...``) and fall
    back to the workflow_id (generic workflows). Best-effort — an unreadable /
    malformed state file degrades to the workflow_id so the signal handler
    still writes SOME sentinel.
    """
    from lib.paths import STATE_PATH  # noqa: PLC0415 — per-call for test isolation

    try:
        state = json.loads(
            (STATE_PATH / "workflows" / f"{workflow_id}.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return workflow_id
    if not isinstance(state, dict):
        return workflow_id
    params = state.get("params")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    if isinstance(params, dict) and params.get("run_id"):
        return str(params["run_id"])
    return workflow_id


def _best_effort_mark_interrupted(workflow_id: str) -> None:
    """Stamp a still-``RUNNING`` workflow ``paused`` on a hard-kill signal.

    AMENDMENT #9: a hard kill (second signal) never lets the runner update the
    status, leaving an eternal-``RUNNING`` state file that ``ed4all stop`` would
    report as live. This best-effort backstop flips a still-RUNNING record to
    ``paused`` so the staleness heuristic + resume path see the truth. Swallows
    every error — it runs on the way to process termination.
    """
    from lib.paths import STATE_PATH  # noqa: PLC0415 — per-call for test isolation

    try:
        path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(state, dict) and str(state.get("status", "")).upper() == "RUNNING":
            state["status"] = "paused"
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            path.write_text(
                json.dumps(state, indent=2, default=str), encoding="utf-8"
            )
    except Exception:  # noqa: BLE001 — best-effort on the termination path
        pass


def _make_stop_signal_handler(run_id: str, workflow_id: str):
    """Build the SIGTERM/SIGINT handler for a run (AMENDMENT #11).

    First signal: write the run-scoped stop sentinel + log — the run finishes
    its current unit, checkpoints, and pauses (exit 3). Second signal: restore
    the DEFAULT disposition on both stop signals (so a further signal hard-kills)
    plus best-effort-mark the workflow interrupted, then re-raise so the default
    terminate disposition fires immediately. Exposed as a factory so the unit
    test can drive the handler directly without installing it or killing itself.
    """
    state = {"fired": False}

    def _handler(signum, frame=None):  # noqa: ARG001 — frame unused
        if not state["fired"]:
            state["fired"] = True
            try:
                stop_control.request_stop(
                    run_id, reason="operator_signal", source=f"signal:{signum}"
                )
            except Exception:  # noqa: BLE001 — a stop request must never crash the handler
                logger.warning("stop signal: request_stop failed", exc_info=True)
            click.secho(
                f"\nGraceful stop requested (signal {signum}); finishing the "
                f"current unit then pausing. Signal again to force-quit.",
                fg="yellow",
                err=True,
            )
            return
        # Second signal → hard kill: restore default disposition, mark the run
        # interrupted, then re-raise so the default terminate fires now.
        for sig in _STOP_SIGNALS:
            try:
                signal.signal(sig, signal.SIG_DFL)
            except (ValueError, OSError):  # pragma: no cover — non-main-thread
                pass
        _best_effort_mark_interrupted(workflow_id)
        signal.raise_signal(signum)

    return _handler


@contextlib.contextmanager
def _install_stop_signals(run_id: str, workflow_id: str):
    """Install the graceful-stop SIGTERM/SIGINT handler for the run's duration.

    Restores the prior handlers on exit. In a non-main thread (e.g. some test
    runners) ``signal.signal`` raises ``ValueError`` — swallowed, so the run
    proceeds without signal handling (tests drive the handler factory directly).
    """
    handler = _make_stop_signal_handler(run_id, workflow_id)
    previous: Dict[Any, Any] = {}
    for sig in _STOP_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread — skip installation
            pass
    try:
        yield handler
    finally:
        for sig, prev in previous.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):  # pragma: no cover
                pass


def _paused_exit_code(result, workflow_id: str, *, output_json: bool) -> Optional[int]:
    """Return exit code 3 (+ a resume hint) iff the run paused; else ``None``.

    A paused run halted at a unit boundary (``ed4all stop`` / SIGTERM / a
    timeout that drained to a checkpoint). It is neither success (0) nor a
    gate/status failure (2) — it is resumable. The resume hint is suppressed in
    ``--json`` mode so it never pollutes the JSON stdout.
    """
    if getattr(result, "status", None) != "paused":
        return None
    if not output_json:
        click.secho(
            f"Workflow {workflow_id} paused at a checkpoint (graceful stop).",
            fg="yellow",
        )
        click.echo(
            f"  Resume with: ed4all run <workflow> --resume {workflow_id}"
        )
    return 3


async def _create_and_run(
    *,
    workflow: str,
    params: Dict[str, Any],
    mode: str,
    provider: str,
    model: Optional[str],
    output_json: bool,
    watch: bool,
) -> int:
    """Create the workflow then run it through the orchestrator.

    Wave 29 Defect 3: now returns an int exit code rather than None.
    The top-level ``run_command`` propagates it via ``sys.exit``:

    * ``0`` — workflow completed successfully (all gates passed).
    * ``3`` — workflow PAUSED at a checkpoint (graceful stop via ``ed4all
      stop`` / SIGTERM / a timeout that drained to a checkpoint); resumable
      with ``--resume`` (D8).
    * ``2`` — workflow ran to completion but at least one gate failed
      **or** the workflow reported a non-ok status.
    * ``1`` — workflow couldn't be created / initialised (existing
      ``_emit_failure`` path, which calls ``sys.exit(1)`` directly).
    """
    if workflow == "textbook_to_course":
        created = await _create_textbook_workflow(params)
    else:
        created = await _create_generic_workflow(workflow, params)

    if "error" in created:
        _emit_failure(created, output_json=output_json)
        return 1  # unreachable — _emit_failure sys.exits — but keeps typing honest

    workflow_id = created.get("workflow_id")
    if not workflow_id:
        _emit_failure(
            {"error": "workflow creation returned no workflow_id", "detail": created},
            output_json=output_json,
        )
        return 1

    orchestrator = _build_orchestrator(mode, provider=provider, model=model)
    if watch:
        click.secho(
            f"Running workflow {workflow_id} ({workflow}) via {mode} mode...",
            fg="cyan",
        )

    # Graceful stop: the run_id the runner uses (params.run_id > workflow_id)
    # is what the run-scoped stop sentinel keys on. Install SIGTERM/SIGINT
    # handlers for the run's duration (first signal → sentinel + pause; second
    # → hard kill).
    run_id = created.get("run_id") or workflow_id
    with _install_stop_signals(run_id, workflow_id):
        result = await orchestrator.run(workflow_id)

    paused_code = _paused_exit_code(result, workflow_id, output_json=output_json)

    if output_json:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
    elif paused_code is None:
        if result.status == "ok":
            click.secho(f"Workflow {workflow_id} completed successfully.", fg="green")
        else:
            click.secho(
                f"Workflow {workflow_id} finished with status={result.status}.",
                fg="yellow",
            )
        if result.error:
            click.secho(f"  Error: {result.error}", fg="red")
        if result.phase_results:
            click.echo()
            click.echo("Phase summary:")
            for name, info in result.phase_results.items():
                click.echo(
                    f"  {name}: {info.get('completed', 0)}/{info.get('task_count', 0)}"
                    f" complete, gates={'pass' if info.get('gates_passed') else 'fail'}"
                )

    # Graceful stop wins over the gate/status collapse — a paused run is
    # resumable, not a failure.
    if paused_code is not None:
        return paused_code

    # Wave 29 Defect 3: exit code propagation.
    gates_failed = _any_gate_failed(result)
    if gates_failed or result.status != "ok":
        return 2
    return 0


def _resolve_resume_stop_after(
    explicit: Optional[str], persisted: Optional[str],
) -> Optional[str]:
    """Resolve the effective ``--stop-after`` for a ``--resume``.

    Trap-2 hardening precedence: an explicit ``--stop-after`` on the
    resume command wins; else the value persisted at workflow creation;
    else ``None`` (run to completion).
    """
    if explicit:
        return explicit
    return persisted or None


def _apply_resume_stop_after_override(
    workflow_id: str, explicit_stop_after: Optional[str],
) -> Optional[str]:
    """Persist an explicit resume-time ``--stop-after`` override.

    ``WorkflowRunner.run_workflow`` reads ``stop_after`` from the
    persisted workflow params, so when the operator passes an explicit
    ``--stop-after`` on a ``--resume`` we patch ``params.stop_after`` in
    the state file BEFORE the run so the new halt point takes effect. A
    no-op (the persisted value already applies) when no explicit override
    is supplied. Best-effort: a missing / unreadable / unwritable state
    file is left untouched and the persisted value continues to govern.
    Returns the effective stop_after, or ``None``.
    """
    from lib.paths import STATE_PATH  # noqa: PLC0415

    workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
    try:
        state = json.loads(workflow_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return explicit_stop_after or None
    if not isinstance(state, dict):
        return explicit_stop_after or None
    params = state.get("params")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    if not isinstance(params, dict):
        params = {}
    state["params"] = params
    persisted = params.get("stop_after")
    effective = _resolve_resume_stop_after(explicit_stop_after, persisted)
    if explicit_stop_after and effective != persisted:
        params["stop_after"] = effective
        try:
            workflow_path.write_text(
                json.dumps(state, indent=2, default=str), encoding="utf-8",
            )
        except OSError:
            # Non-fatal: fall back to whatever was already persisted.
            pass
    return effective


def _apply_resume_reuse_objectives_override(
    workflow_id: str, explicit_reuse_objectives: Optional[str],
) -> Optional[str]:
    """Persist an explicit resume-time ``--reuse-objectives`` override.

    Mirrors :func:`_apply_resume_stop_after_override`. The runner reads
    ``reuse_objectives_path`` from the persisted workflow params to
    synthesize the ``course_planning`` phase output from a pinned
    objectives JSON instead of re-dispatching the outliner; so when the
    operator passes ``--reuse-objectives`` on a ``--resume`` we patch
    ``params.reuse_objectives_path`` (as a resolved absolute path — the
    fresh-run contract) in the state file BEFORE the run. A no-op when no
    override is supplied. Best-effort: a missing / unreadable / unwritable
    state file is left untouched. Returns the effective path, or ``None``.
    """
    if not explicit_reuse_objectives:
        return None

    from lib.paths import STATE_PATH  # noqa: PLC0415

    workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
    try:
        state = json.loads(workflow_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    params = state.get("params")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    if not isinstance(params, dict):
        params = {}
    state["params"] = params
    resolved = str(Path(explicit_reuse_objectives).resolve())
    if params.get("reuse_objectives_path") != resolved:
        params["reuse_objectives_path"] = resolved
        try:
            workflow_path.write_text(
                json.dumps(state, indent=2, default=str), encoding="utf-8",
            )
        except OSError:
            # Non-fatal: fall back to whatever was already persisted.
            pass
    return resolved


def _resume_workflow(
    *,
    workflow_id: str,
    mode: str,
    provider: str,
    model: Optional[str],
    output_json: bool,
    watch: bool,
    stop_after: Optional[str] = None,
    reuse_objectives: Optional[str] = None,
) -> None:
    """Resume an existing workflow state through the orchestrator.

    Wave 29 Defect 3: ``--resume`` also honours the resumed workflow's
    final gate status — a resumed run that fails gates exits 2.

    Trap-2 hardening: an explicit ``--stop-after`` passed alongside
    ``--resume`` overrides the persisted halt point before the run; when
    unset, the value persisted at creation continues to govern.
    """

    async def _run() -> int:
        # Apply the resume-time --stop-after precedence BEFORE the run so
        # run_workflow (which reads stop_after from persisted params)
        # honors the effective halt point.
        _apply_resume_stop_after_override(workflow_id, stop_after)
        # I3: an explicit --reuse-objectives on the resume command pins the
        # resumed course_planning phase to a prior objectives JSON (parity
        # with the fresh-run --reuse-objectives). Patched into the persisted
        # params before the run; unset → persisted value governs.
        _apply_resume_reuse_objectives_override(workflow_id, reuse_objectives)
        orchestrator = _build_orchestrator(mode, provider=provider, model=model)
        if watch:
            click.secho(
                f"Resuming workflow {workflow_id} via {mode} mode...", fg="cyan"
            )
        # Graceful stop: install SIGTERM/SIGINT handlers for the resumed run's
        # duration, keyed on the persisted run_id (params.run_id > workflow_id).
        resume_run_id = _resolve_run_id_for_workflow(workflow_id)
        with _install_stop_signals(resume_run_id, workflow_id):
            result = await orchestrator.run(workflow_id)

        paused_code = _paused_exit_code(
            result, workflow_id, output_json=output_json
        )
        if output_json:
            click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        elif paused_code is None:
            status_color = "green" if result.status == "ok" else "yellow"
            click.secho(
                f"Workflow {workflow_id} resumed: status={result.status}",
                fg=status_color,
            )
            if result.error:
                click.secho(f"  Error: {result.error}", fg="red")

        # A paused resume is resumable again (exit 3), not a failure.
        if paused_code is not None:
            return paused_code

        gates_failed = _any_gate_failed(result)
        if gates_failed or result.status != "ok":
            return 2
        return 0

    exit_code = asyncio.run(_run())
    if exit_code:
        sys.exit(exit_code)


def _emit_failure(payload: Dict[str, Any], *, output_json: bool) -> None:
    if output_json:
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        click.secho(f"Error: {payload.get('error')}", fg="red")
        detail = payload.get("detail")
        if detail:
            click.echo(f"  detail: {detail}")
    sys.exit(1)


def register_run_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all run`` command to the top-level CLI group."""
    cli_group.add_command(run_command)
