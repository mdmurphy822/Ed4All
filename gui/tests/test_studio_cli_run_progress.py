"""Static-asset contract: create.js degraded Build-progress for CLI-launched runs.

A run launched via ``ed4all run ...`` exists only in orchestrator state — the
GUI run registry (``state/gui/runs/``) never saw it, so ``GET /api/runs/<id>``
404s as ``unknown_run`` BY DESIGN (run_service reads only the GUI registry),
while ``GET /api/runs/<id>/progress`` (progress_service) deliberately accepts
the bare orchestrator workflow id ("CLI-launched runs are observable too").

The Build-progress page must therefore DEGRADE instead of hard-failing:

* on an unknown_run 404 it probes ``/progress`` once; if that also fails, the
  original error path is kept verbatim;
* on a successful probe it renders the shared stage-tracker rail + 3s poll
  (``mountStageRail`` — the same helper the normal path uses, so the
  terminal-stop logic exists exactly once) with NO explanatory banner;
* it honestly OMITS the WS log console and the cancel control — the GUI never
  launched this process, so ``state/gui/logs/<id>.log`` and the driver task do
  not exist (no stubs, no fabricated capabilities).

Client-side JS can't be driven headless here, so per ``gui/CLAUDE.md`` this is
asserted against the served static asset (the ``test_embed_widget.py``
pattern). Pure file-content assertions — no fastapi needed, no state touched,
no real run ids referenced.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CREATE_JS = REPO_ROOT / "gui" / "static" / "studio" / "create.js"


def _js() -> str:
    return CREATE_JS.read_text(encoding="utf-8")


def _top_level_function(js: str, name: str) -> str:
    """Slice a top-level ``function <name>(...) { ... }`` body out of the file.

    create.js declares its functions at column 0 and closes them with a
    column-0 ``}``, so the first ``\\n}`` after the declaration bounds the body.
    """
    start = js.index(f"function {name}(")
    end = js.index("\n}", start)
    return js[start:end]


# --------------------------------------------------------------------------- #
# The unknown_run fallback branch: probe /progress before giving up
# --------------------------------------------------------------------------- #


def test_unknown_run_404_probes_progress_before_erroring():
    js = _js()
    # The typed-error check on the api() ApiError surface (flat {error, detail}
    # body → e.error), scoped to the 404 so other failures keep the hard error.
    assert "e.status === 404 && e.error === 'unknown_run'" in js, (
        "renderProgress must special-case the unknown_run 404 from run_service"
    )
    body = _top_level_function(js, "renderProgress")
    # The one-shot probe of the progress endpoint inside the catch, feeding the
    # degraded renderer only on success.
    assert "/progress" in body and "renderCliRunProgress(" in body, (
        "the unknown_run branch must probe GET /api/runs/<id>/progress and "
        "hand off to the degraded CLI-run renderer when it answers"
    )
    # The original error path survives verbatim for every other failure.
    assert "toastErr(e, 'Could not load run')" in body


# --------------------------------------------------------------------------- #
# The degraded CLI-run page: rail only — no banner, no console, no cancel
# --------------------------------------------------------------------------- #


def test_cli_run_note_removed():
    """Owner direction: NO explanatory banner on the CLI-run page — it renders
    the rail/stats/output directly. The Run <id> meta line survives, and the
    page still adds no alert / live region of its own."""
    js = _js()
    body = _top_level_function(js, "renderCliRunProgress")
    assert "launched from the command line" not in body
    assert "cli-run-note" not in body
    assert "role: 'alert'" not in body
    assert "aria-live" not in body
    # The Run <id> meta line survives on the degraded page.
    assert "`Run ${runId}`" in body


def test_cli_run_page_omits_ws_console_and_cancel():
    js = _js()
    body = _top_level_function(js, "renderCliRunProgress")
    # No WS log console: the GUI log file for this run does not exist.
    assert "runProgressConsole(" not in body
    assert "openRunSocket(" not in body
    # No cancel affordance: there is no driver task to cancel.
    assert "onCancel" not in body
    assert "requestCancel" not in body
    assert "/cancel" not in body


def test_both_paths_share_the_stage_rail_helper():
    js = _js()
    # One definition + a call site in each of the two progress paths, so the
    # poll / terminal-stop logic is never duplicated.
    assert js.count("mountStageRail(") >= 3, (
        "mountStageRail must be defined once and used by both the normal and "
        "the degraded CLI-run progress paths"
    )
    helper = _top_level_function(js, "mountStageRail")
    # The shared loop stops itself at terminal statuses (both paths inherit it).
    # The terminal-status list is hoisted into module-level RAIL_TERMINAL_EFF /
    # RAIL_TERMINAL_RAW constants; the helper references them (never its own
    # inline copy) and calls clearInterval when a terminal status is observed.
    assert "RAIL_TERMINAL_EFF" in helper and "RAIL_TERMINAL_RAW" in helper, (
        "the shared helper must consult the terminal-status constants"
    )
    assert "clearInterval" in helper
    # The constants themselves must carry every terminal status the rail latches
    # on — asserted at their single point of definition so both paths inherit it.
    for const in ("RAIL_TERMINAL_EFF", "RAIL_TERMINAL_RAW"):
        decl_start = js.index(f"const {const} = [")
        decl = js[decl_start:js.index("]", decl_start)]
        for status in ("'completed'", "'failed'", "'cancelled'", "'paused'"):
            assert status in decl, f"{const} must include {status}"
    # The degraded page seeds the rail from the probe snapshot and starts the
    # same loop the normal path uses.
    cli_body = _top_level_function(js, "renderCliRunProgress")
    assert "railCtl.apply(snapshot)" in cli_body
    assert "railCtl.start()" in cli_body
