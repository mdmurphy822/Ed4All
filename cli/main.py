#!/usr/bin/env python3
"""
Ed4All CLI - Integrity checking and run management tools.

Phase 0 Hardening - Requirement 9: CLI Integrity Checks

Usage: python -m cli.main [command] [options]
   or: ed4all [command] [options]  (if installed)
"""

import sys
from pathlib import Path

# Add project root to path
_CLI_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CLI_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import click
except ImportError:
    print("Error: click library required. Install with: pip install click")
    sys.exit(1)

from lib.paths import LIBV2_PATH, STATE_PATH


@click.group()
@click.version_option(version="0.1.0", prog_name="ed4all")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """Ed4All integrity checking and run management tools."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


# Register Wave 7 canonical 'ed4all run' command. Imported lazily to keep
# legacy CLI paths working if the orchestrator package fails to import.
try:
    from cli.commands import register_run_command

    register_run_command(cli)
except ImportError as _run_import_err:  # pragma: no cover
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "cli.commands.run unavailable: %s", _run_import_err
    )


# =============================================================================
# VALIDATE-RUN COMMAND
# =============================================================================

@cli.command('validate-run')
@click.argument('run_id')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--fix', is_flag=True, help='Attempt to fix issues')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def validate_run(ctx, run_id: str, verbose: bool, fix: bool, output_json: bool):
    """
    Validate a run's integrity.

    Checks:
    - Manifest schema validity
    - Config lockfile integrity
    - Hash chain continuity
    - Artifact hashes
    - Decision schema compliance
    """
    import json

    from .validators.run_validator import RunValidator

    verbose = verbose or ctx.obj.get('verbose', False)
    validator = RunValidator(run_id)
    result = validator.validate(fix=fix)

    if output_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    if result.passed:
        click.secho(f"✓ Run {run_id} passed validation", fg='green')
    else:
        click.secho(f"✗ Run {run_id} failed validation", fg='red')

    if verbose or not result.passed:
        for issue in result.issues:
            color = 'red' if issue.severity == 'error' else 'yellow'
            click.secho(f"  [{issue.severity}] {issue.category}: {issue.message}", fg=color)
            if issue.path:
                click.echo(f"    Path: {issue.path}")

    click.echo(f"\nChecked: {result.checked_files} files")
    click.echo(f"Errors: {result.error_count}")
    click.echo(f"Warnings: {result.warning_count}")

    if fix and result.fixed_count > 0:
        click.secho(f"Fixed: {result.fixed_count} issues", fg='green')

    sys.exit(0 if result.passed else 1)


# =============================================================================
# SUMMARIZE-RUN COMMAND
# =============================================================================

@cli.command('summarize-run')
@click.argument('run_id')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['text', 'json', 'markdown']),
              default='text', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def summarize_run(ctx, run_id: str, output_format: str, output: str):
    """
    Generate a summary report for a run.

    Includes:
    - Run metadata and timing
    - Phase completion status
    - Decision statistics
    - Quality metrics
    - Artifact inventory
    """
    from .reporters.run_summarizer import RunSummarizer

    summarizer = RunSummarizer(run_id)
    report = summarizer.generate(format=output_format)

    if output:
        Path(output).write_text(report)
        click.secho(f"Report written to {output}", fg='green')
    else:
        click.echo(report)


# =============================================================================
# DIFF-RUNS COMMAND
# =============================================================================

@cli.command('diff-runs')
@click.argument('run_a')
@click.argument('run_b')
@click.option('--config-only', is_flag=True, help='Compare only configurations')
@click.option('--decisions-only', is_flag=True, help='Compare only decisions')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def diff_runs(ctx, run_a: str, run_b: str, config_only: bool,
              decisions_only: bool, output_json: bool):
    """
    Compare two runs.

    Shows differences in:
    - Configuration snapshots
    - Decision patterns
    - Outcomes and metrics
    """
    import json

    from .comparators.run_diff import RunDiff

    differ = RunDiff(run_a, run_b)

    if config_only:
        diff = differ.compare_configs()
    elif decisions_only:
        diff = differ.compare_decisions()
    else:
        diff = differ.compare_all()

    if output_json:
        click.echo(json.dumps(diff.to_dict(), indent=2))
    else:
        click.echo(diff.format())


# =============================================================================
# EXPORT-TRAINING COMMAND
# =============================================================================

@cli.command('export-training')
@click.argument('run_id')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['jsonl', 'alpaca', 'openai', 'dpo']),
              default='jsonl', help='Output format')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file path')
@click.option('--min-quality',
              type=click.Choice(['exemplary', 'proficient', 'developing']),
              default='proficient', help='Minimum quality level')
@click.option('--decision-types', '-t', multiple=True,
              help='Filter to specific decision types (can repeat)')
@click.option('--include-rejected', is_flag=True,
              help='Include rejected/negative examples')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output stats as JSON')
@click.pass_context
def export_training(ctx, run_id: str, output_format: str, output: str,
                   min_quality: str, decision_types: tuple, include_rejected: bool,
                   verbose: bool, output_json: bool):
    """
    Export training data from a run.

    Formats:
    - jsonl: Raw decision events
    - alpaca: Instruction format for fine-tuning
    - openai: OpenAI-compatible format
    - dpo: Direct Preference Optimization pairs
    """
    import json

    from .exporters.training_exporter import TrainingExporter

    verbose = verbose or ctx.obj.get('verbose', False)

    exporter = TrainingExporter(run_id)

    # Handle DPO separately
    if output_format == 'dpo':
        stats = exporter.export_dpo_pairs(
            output_path=Path(output),
            min_quality=min_quality
        )
    else:
        stats = exporter.export(
            output_path=Path(output),
            format=output_format,
            min_quality=min_quality,
            decision_types=list(decision_types) if decision_types else None,
            include_rejected=include_rejected
        )

    if output_json:
        click.echo(json.dumps(stats.to_dict(), indent=2))
        return

    if stats.warnings:
        for warning in stats.warnings:
            click.secho(f"Warning: {warning}", fg='yellow')

    click.secho(f"Exported {stats.exported_events} training examples to {output}", fg='green')

    if verbose:
        click.echo(f"\nTotal events processed: {stats.total_events}")
        click.echo(f"Filtered out: {stats.filtered_events}")
        click.echo("\nBy decision type:")
        for dtype, count in sorted(stats.by_decision_type.items()):
            click.echo(f"  {dtype}: {count}")
        click.echo("\nBy quality level:")
        for quality, count in sorted(stats.by_quality.items()):
            click.echo(f"  {quality}: {count}")


# =============================================================================
# FSCK COMMAND
# =============================================================================

@cli.command('fsck')
@click.option('--fix', is_flag=True, help='Attempt to fix issues')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def fsck(ctx, fix: bool, verbose: bool, output_json: bool):
    """
    Check LibV2 storage integrity.

    Validates:
    - Blob hash integrity
    - Catalog consistency
    - Run manifest validity
    - Symlink targets
    """
    import json

    from lib.libv2_fsck import LibV2Fsck

    verbose = verbose or ctx.obj.get('verbose', False)

    checker = LibV2Fsck(LIBV2_PATH)
    result = checker.check_all(fix=fix)

    if output_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    if result.passed:
        click.secho("✓ LibV2 integrity check passed", fg='green')
    else:
        click.secho("✗ LibV2 integrity check failed", fg='red')

    click.echo(f"\nChecked: {result.checked_files} files")
    click.echo(f"Errors: {result.error_count}")
    click.echo(f"Warnings: {result.warning_count}")

    if fix and result.fixed_count > 0:
        click.secho(f"Fixed: {result.fixed_count} issues", fg='green')

    if verbose:
        for issue in result.issues:
            color = 'red' if issue.severity == 'error' else 'yellow'
            click.secho(f"  [{issue.severity}] {issue.category}: {issue.message}", fg=color)
            if issue.path:
                click.echo(f"    Path: {issue.path}")

    sys.exit(0 if result.passed else 1)


# =============================================================================
# LIST-RUNS COMMAND
# =============================================================================

@cli.command('list-runs')
@click.option('--limit', '-n', default=20, help='Number of runs to show')
@click.option('--status', type=click.Choice(['all', 'completed', 'failed', 'running']),
              default='all', help='Filter by status')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def list_runs(ctx, limit: int, status: str, output_json: bool):
    """List recent runs."""
    import json

    runs_dir = STATE_PATH / "runs"
    if not runs_dir.exists():
        click.echo("No runs found")
        return

    runs = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True)[:limit]:
        if not run_dir.is_dir():
            continue

        manifest_path = run_dir / "run_manifest.json"
        run_info = {
            "run_id": run_dir.name,
            "created_at": None,
            "status": "unknown",
            "workflow_type": None,
        }

        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                run_info["created_at"] = manifest.get("created_at")
                run_info["workflow_type"] = manifest.get("workflow_type")
                run_info["status"] = manifest.get("status", "completed")
            except Exception:
                pass

        # Filter by status
        if status != 'all' and run_info['status'] != status:
            continue

        runs.append(run_info)

    if output_json:
        click.echo(json.dumps(runs, indent=2))
        return

    if not runs:
        click.echo("No runs found matching criteria")
        return

    click.echo(f"{'Run ID':<40} {'Status':<12} {'Workflow':<20} {'Created'}")
    click.echo("-" * 90)
    for run in runs:
        created = run['created_at'][:19] if run['created_at'] else 'N/A'
        workflow = run['workflow_type'] or 'N/A'
        click.echo(f"{run['run_id']:<40} {run['status']:<12} {workflow:<20} {created}")


# =============================================================================
# VERIFY-CHAIN COMMAND
# =============================================================================

@cli.command('verify-chain')
@click.argument('chain_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def verify_chain(ctx, chain_file: str, verbose: bool):
    """
    Verify a hash-chained event log.

    Checks hash chain integrity and reports any breaks.
    """
    from lib.hash_chain import HashChainedLog

    verbose = verbose or ctx.obj.get('verbose', False)

    chain = HashChainedLog(Path(chain_file))
    result = chain.verify()

    if result.valid:
        click.secho(f"✓ Hash chain verified: {result.event_count} events", fg='green')
    else:
        click.secho(f"✗ Hash chain broken at sequence {result.break_at_seq}", fg='red')
        click.echo(f"Error: {result.error}")

    if verbose:
        click.echo(f"\nFirst event: seq={result.first_seq}")
        click.echo(f"Last event: seq={result.last_seq}")
        click.echo(f"Total events: {result.event_count}")

    sys.exit(0 if result.valid else 1)


# =============================================================================
# TEXTBOOK-TO-COURSE COMMAND
# =============================================================================

@cli.command('textbook-to-course')
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--course-name', '-n', required=True, help='Course identifier (e.g., PHYS_101)')
@click.option('--objectives', '-o', type=click.Path(exists=True), help='Optional objectives file to merge')
@click.option('--weeks', '-w', type=int, default=12, help='Course duration in weeks')
@click.option('--no-assessments', is_flag=True, help='Skip Trainforge assessment generation')
@click.option('--assessment-count', type=int, default=50, help='Number of questions to generate')
@click.option('--bloom-levels', default='remember,understand,apply,analyze', help='Target Bloom levels (comma-separated)')
@click.option('--priority', type=click.Choice(['low', 'normal', 'high']), default='normal')
@click.option('--dry-run', is_flag=True, help='Show plan without executing')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def textbook_to_course(ctx, pdf_path, course_name, objectives, weeks,
                       no_assessments, assessment_count, bloom_levels,
                       priority, dry_run, output_json):
    """
    [DEPRECATED] Create a complete course from a PDF textbook.

    Superseded by ``ed4all run textbook-to-course`` (Wave 7). This command
    will be removed in the next cleanup cycle.

    Pipeline: DART (PDF->HTML) -> Courseforge (content) -> Trainforge (assessments)

    This command orchestrates the full textbook-to-course pipeline:

    \b
    1. DART converts PDF textbook(s) to accessible HTML
    2. Staged outputs are copied to Courseforge inputs
    3. Learning objectives are extracted from textbook content
    4. Course structure and modules are generated
    5. IMSCC package is created for LMS import
    6. (Optional) Trainforge generates assessments

    Examples:

    \b
        ed4all textbook-to-course physics.pdf -n PHYS_101
        ed4all textbook-to-course textbook.pdf -n CHEM_201 --weeks 16
        ed4all textbook-to-course book.pdf -n CS_101 --no-assessments
        ed4all textbook-to-course ./textbooks/ -n BIO_301  # Directory of PDFs
    """
    import asyncio
    import json as json_lib

    click.secho(
        "DEPRECATED: Use 'ed4all run textbook-to-course' instead. "
        "This command will be removed in the next cleanup cycle.",
        fg='yellow',
        err=True,
    )

    pdf_path = Path(pdf_path)

    # Dry run - show what would happen
    if dry_run:
        phases = [
            "dart_conversion",
            "staging",
            "objective_extraction",
            "course_planning",
            "content_generation",
            "packaging",
        ]
        if not no_assessments:
            phases.append("trainforge_assessment")
        phases.append("finalization")

        plan = {
            "workflow": "textbook_to_course",
            "pdf_paths": [str(pdf_path)],
            "course_name": course_name,
            "objectives_path": objectives,
            "duration_weeks": weeks,
            "generate_assessments": not no_assessments,
            "assessment_count": assessment_count if not no_assessments else None,
            "bloom_levels": bloom_levels.split(","),
            "priority": priority,
            "phases": phases
        }

        if output_json:
            click.echo(json_lib.dumps(plan, indent=2))
        else:
            click.secho("Dry run - would execute:", fg='cyan')
            click.echo(f"  PDF source:    {pdf_path}")
            click.echo(f"  Course name:   {course_name}")
            click.echo(f"  Duration:      {weeks} weeks")
            click.echo(f"  Assessments:   {'Yes (' + str(assessment_count) + ' questions)' if not no_assessments else 'No'}")
            click.echo(f"  Bloom levels:  {bloom_levels}")
            click.echo(f"  Priority:      {priority}")
            if objectives:
                click.echo(f"  Objectives:    {objectives}")
            click.echo()
            click.secho("Pipeline phases:", fg='cyan')
            for i, phase in enumerate(phases, 1):
                click.echo(f"  {i}. {phase}")
        return

    # Execute pipeline
    try:
        # Import standalone pipeline function
        sys.path.insert(0, str(_PROJECT_ROOT / "MCP"))
        from MCP.tools.pipeline_tools import create_textbook_pipeline

        async def run_pipeline():
            return await create_textbook_pipeline(
                pdf_paths=str(pdf_path),
                course_name=course_name,
                objectives_path=objectives,
                duration_weeks=weeks,
                generate_assessments=not no_assessments,
                assessment_count=assessment_count,
                bloom_levels=bloom_levels,
                priority=priority
            )

        result = asyncio.run(run_pipeline())
        result_data = json_lib.loads(result)

        if output_json:
            click.echo(json_lib.dumps(result_data, indent=2))
        elif "error" in result_data:
            click.secho(f"Error: {result_data['error']}", fg='red')
            sys.exit(1)
        else:
            click.secho("Pipeline created successfully!", fg='green')
            click.echo()
            click.echo(f"  Workflow ID: {result_data.get('workflow_id')}")
            click.echo(f"  Run ID:      {result_data.get('run_id')}")
            click.echo(f"  Status:      {result_data.get('status')}")
            click.echo()
            click.secho("Monitor progress with:", fg='cyan')
            click.echo(f"  ed4all summarize-run {result_data.get('run_id')}")

    except ImportError as e:
        click.secho(f"Import error: {e}", fg='red')
        click.echo("Ensure MCP tools are properly installed.")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {e}", fg='red')
        sys.exit(1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
