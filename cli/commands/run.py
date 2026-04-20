"""
Canonical ``ed4all run`` CLI command (Wave 7).

This command is the single recommended entry point for running any Ed4All
workflow end-to-end. It replaces the ad-hoc trio of
``ed4all textbook-to-course`` + ``create_textbook_pipeline_tool`` +
``run_textbook_pipeline_tool`` with a unified surface:

    ed4all run <workflow_name> [options]

Workflow names correspond to keys in ``config/workflows.yaml``
(``textbook_to_course``, ``course_generation``, ``intake_remediation``,
``batch_dart``, ``rag_training``). The command:

1. Parses CLI flags into workflow params.
2. Creates the workflow state (reuses the existing per-workflow creators).
3. Instantiates ``PipelineOrchestrator`` with the chosen mode + backend.
4. Calls ``.run(workflow_id)`` and streams / returns the result.

``--dry-run`` prints the planned phase sequence without executing anything.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

logger = logging.getLogger(__name__)


# Mapping of canonical workflow names accepted by ``ed4all run`` to their
# "creator" functions (which produce a workflow_id + state JSON). Not every
# workflow has a creator wired up yet; ones that don't surface an error
# pointing the user at the stage-by-stage commands.
SUPPORTED_WORKFLOWS = {
    "textbook_to_course",
    "textbook-to-course",  # hyphenated alias
    "course_generation",
    "intake_remediation",
    "batch_dart",
    "rag_training",
}


def _normalize_workflow(name: str) -> str:
    return name.replace("-", "_").strip().lower()


def _build_workflow_params(
    workflow: str,
    *,
    corpus: Optional[str],
    course_name: Optional[str],
    weeks: int,
    no_assessments: bool,
    assessment_count: int,
    bloom_levels: str,
    priority: str,
    objectives_path: Optional[str],
) -> Dict[str, Any]:
    """Build the params dict for a workflow from CLI inputs."""
    params: Dict[str, Any] = {
        "course_name": course_name,
        "duration_weeks": weeks,
        "generate_assessments": not no_assessments,
        "assessment_count": assessment_count,
        "bloom_levels": bloom_levels,
        "priority": priority,
    }

    if objectives_path:
        params["objectives_path"] = objectives_path

    if corpus:
        params["corpus"] = corpus
        # textbook_to_course expects pdf_paths specifically
        if workflow == "textbook_to_course":
            params["pdf_paths"] = corpus

    return params


async def _create_textbook_workflow(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Delegate to the existing ``create_textbook_pipeline`` helper.

    Avoids duplicating all the state-setup boilerplate while we migrate.
    Returns the parsed JSON response.
    """
    from MCP.tools.pipeline_tools import create_textbook_pipeline

    result = await create_textbook_pipeline(
        pdf_paths=params.get("pdf_paths", params.get("corpus", "")),
        course_name=params["course_name"],
        objectives_path=params.get("objectives_path"),
        duration_weeks=params.get("duration_weeks", 12),
        generate_assessments=params.get("generate_assessments", True),
        assessment_count=params.get("assessment_count", 50),
        bloom_levels=params.get("bloom_levels", "remember,understand,apply,analyze"),
        priority=params.get("priority", "normal"),
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
    type=click.Choice(["anthropic", "openai"]),
    default=None,
    help="LLM provider for api mode. Default: env LLM_PROVIDER or 'anthropic'.",
)
@click.option(
    "--model",
    help="Model identifier override. Default: per-provider default.",
)
@click.option(
    "--weeks",
    type=int,
    default=12,
    help="Course duration in weeks (workflow-dependent)",
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
    weeks: int,
    no_assessments: bool,
    assessment_count: int,
    bloom_levels: str,
    priority: str,
    objectives: Optional[str],
    resume_run_id: Optional[str],
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

    mode = _resolve_mode(mode)
    provider = _resolve_provider(api_provider)

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
        )
        return

    # -------- create + run -----------------------------------------------
    if not course_name:
        click.secho(
            "--course-name is required unless --dry-run or --resume is used.",
            fg="red",
        )
        sys.exit(2)

    if not corpus and workflow in {"textbook_to_course", "batch_dart", "rag_training"}:
        click.secho(
            f"--corpus is required for workflow '{workflow}'.",
            fg="red",
        )
        sys.exit(2)

    asyncio.run(
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
        phases = []
        for idx, phase in enumerate(sorted_phases):
            if skip_trainforge and phase.name == "trainforge_assessment":
                continue
            phases.append(
                {
                    "order": len(phases) + 1,
                    "name": phase.name,
                    "agents": list(phase.agents),
                    "max_concurrent": getattr(phase, "max_concurrent", 5),
                    "depends_on": list(phase.depends_on or []),
                    "optional": bool(getattr(phase, "optional", False)),
                }
            )

        return {
            "workflow": workflow,
            "mode": mode,
            "provider": provider,
            "params": params,
            "phases": phases,
        }
    except Exception as exc:  # noqa: BLE001 — dry-run shouldn't explode
        return {
            "workflow": workflow,
            "mode": mode,
            "provider": provider,
            "params": params,
            "error": f"plan build failed: {exc}",
            "phases": [],
        }


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
    click.echo()
    click.secho("Phases:", fg="cyan")
    for phase in plan.get("phases", []):
        agents = ", ".join(phase["agents"])
        click.echo(
            f"  {phase['order']}. {phase['name']}"
            f"  [agents={agents}, max_concurrent={phase['max_concurrent']}]"
        )


async def _create_and_run(
    *,
    workflow: str,
    params: Dict[str, Any],
    mode: str,
    provider: str,
    model: Optional[str],
    output_json: bool,
    watch: bool,
) -> None:
    """Create the workflow then run it through the orchestrator."""
    if workflow == "textbook_to_course":
        created = await _create_textbook_workflow(params)
    else:
        created = await _create_generic_workflow(workflow, params)

    if "error" in created:
        _emit_failure(created, output_json=output_json)
        return

    workflow_id = created.get("workflow_id")
    if not workflow_id:
        _emit_failure(
            {"error": "workflow creation returned no workflow_id", "detail": created},
            output_json=output_json,
        )
        return

    orchestrator = _build_orchestrator(mode, provider=provider, model=model)
    if watch:
        click.secho(
            f"Running workflow {workflow_id} ({workflow}) via {mode} mode...",
            fg="cyan",
        )

    result = await orchestrator.run(workflow_id)

    if output_json:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        return

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


def _resume_workflow(
    *,
    workflow_id: str,
    mode: str,
    provider: str,
    model: Optional[str],
    output_json: bool,
    watch: bool,
) -> None:
    """Resume an existing workflow state through the orchestrator."""

    async def _run() -> None:
        orchestrator = _build_orchestrator(mode, provider=provider, model=model)
        if watch:
            click.secho(
                f"Resuming workflow {workflow_id} via {mode} mode...", fg="cyan"
            )
        result = await orchestrator.run(workflow_id)
        if output_json:
            click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            status_color = "green" if result.status == "ok" else "yellow"
            click.secho(
                f"Workflow {workflow_id} resumed: status={result.status}",
                fg=status_color,
            )
            if result.error:
                click.secho(f"  Error: {result.error}", fg="red")

    asyncio.run(_run())


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
