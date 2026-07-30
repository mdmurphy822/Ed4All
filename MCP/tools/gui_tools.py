"""GUI control-plane MCP tools — the Claude-interaction surface.

Registered via ``register_gui_tools(mcp)`` from ``MCP/server.py`` alongside the
other ``register_*`` tool modules. Every tool operates on the SAME ``runtime/state/gui/``
store the FastAPI GUI reads/writes, so a Claude Code session drives/observes
exactly what the GUI shows (the bidirectional bridge of build spec §8).

IMPORTANT: this module is **import-safe without FastAPI/uvicorn installed**. It
imports only ``gui.shared_state`` + ``gui.settings_store`` (foundation-core, no
web deps). ``gui.services.course_service`` is imported LAZILY inside the two
course tools so the module-level import never requires the ``gui`` extra; if
that service (or its deps) is unavailable a typed error is returned rather than
raising on import.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path so ``gui`` imports resolve the same way the other
# tool modules resolve ``lib`` / ``MCP`` packages.
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gui import settings_store, shared_state  # noqa: E402

logger = logging.getLogger(__name__)


def _err(message: str, **extra: Any) -> str:
    """Serialize a typed error record (never a fabricated success)."""
    payload = {"error": message}
    payload.update(extra)
    return json.dumps(payload)


def register_gui_tools(mcp):
    """Register the GUI control-plane tools with the MCP server."""

    @mcp.tool()
    async def gui_get_settings() -> str:
        """
        Return the masked GUI settings document.

        Reads the canonical ``runtime/state/gui/settings.json`` (falling back to catalog
        defaults if unset) and masks every ``*_API_KEY`` / ``*_KEY`` value to
        ``"set"`` / ``null`` so secrets never surface. This is exactly what the
        GUI's Settings tab displays.
        """
        try:
            masked = settings_store.mask_secrets(settings_store.load_settings())
            return json.dumps(masked, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_get_settings failed")
            return _err(str(e))

    @mcp.tool()
    async def gui_set_setting(path: str, value: Any) -> str:
        """
        Deep-patch a single setting at a dotted path; return the masked doc.

        Args:
            path: Dotted path into the settings doc, e.g.
                ``model_routing.global.provider`` or ``flags.COURSEFORGE_TWO_PASS``
                or ``env.ANTHROPIC_API_KEY``.
            value: New value to set at that path (string/number/bool/null).

        The patch is applied via ``settings_store.update_settings`` (deep-merge +
        validate + atomic save), so the GUI sees the change immediately. The
        returned doc is masked.
        """
        try:
            if not isinstance(path, str) or not path.strip():
                return _err("path must be a non-empty dotted string")
            keys = path.split(".")
            patch: dict = {}
            cursor = patch
            for key in keys[:-1]:
                cursor[key] = {}
                cursor = cursor[key]
            cursor[keys[-1]] = value
            saved = settings_store.update_settings(patch)
            return json.dumps(settings_store.mask_secrets(saved), indent=2)
        except ValueError as e:
            return _err(str(e), detail="validation_error")
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_set_setting failed")
            return _err(str(e))

    @mcp.tool()
    async def gui_list_runs() -> str:
        """
        List every run in the GUI run registry (newest first).

        Mirrors ``runtime/state/gui/runs/<run_id>.json`` — the same records the GUI's
        Activity / Runs views render and that ``gui_enqueue_run`` writes.
        """
        try:
            return json.dumps(shared_state.list_runs(), indent=2)
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_list_runs failed")
            return _err(str(e))

    @mcp.tool()
    async def gui_get_run(run_id: str) -> str:
        """
        Return a single run record from the GUI run registry.

        Args:
            run_id: The run identifier (as minted by ``gui_enqueue_run`` or the
                GUI backend).

        Returns the run JSON, or a typed ``not_found`` error if unknown.
        """
        try:
            record = shared_state.read_run(run_id)
            if record is None:
                return _err(f"no run registered with id {run_id!r}", detail="not_found")
            return json.dumps(record, indent=2)
        except ValueError as e:
            return _err(str(e), detail="validation_error")
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_get_run failed")
            return _err(str(e))

    @mcp.tool()
    async def gui_enqueue_run(
        workflow: str,
        course_name: str,
        corpus: str,
        options_json: str = "{}",
    ) -> str:
        """
        Enqueue a run request the GUI backend (or run_service) picks up.

        Writes a registry record with ``status="requested"`` (symmetric to the
        TaskMailbox bridge): a Claude Code session requests a run, the GUI side
        observes the request and launches it.

        Args:
            workflow: Workflow name (e.g. ``textbook_to_course``, ``courseforge-outline``).
            course_name: Target course name / code.
            corpus: Corpus path or upload id that will feed ``--corpus``.
            options_json: JSON object string of extra options (weeks, mode,
                provider, model, etc.). Defaults to ``"{}"``.

        Returns the stored run record (including the minted ``run_id``).
        """
        try:
            if not isinstance(workflow, str) or not workflow.strip():
                return _err("workflow must be a non-empty string")
            if not isinstance(course_name, str) or not course_name.strip():
                return _err("course_name must be a non-empty string")
            try:
                options = json.loads(options_json) if options_json else {}
            except json.JSONDecodeError as e:
                return _err(f"options_json is not valid JSON: {e}", detail="validation_error")
            if not isinstance(options, dict):
                return _err("options_json must decode to a JSON object")

            run_id = shared_state.new_run_id("REQ")
            record = {
                "run_id": run_id,
                "status": "requested",
                "source": "claude",
                "workflow": workflow,
                "course_name": course_name,
                "corpus": corpus,
                "options": options,
            }
            shared_state.register_run(record)
            stored = shared_state.read_run(run_id)
            return json.dumps(stored, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_enqueue_run failed")
            return _err(str(e))

    @mcp.tool()
    async def gui_list_courses() -> str:
        """
        List the courses the GUI knows about (Courseforge exports + LibV2).

        Lazily imports ``gui.services.course_service`` (FastAPI-free) so this
        tool module never requires the ``gui`` extra at import time. If the
        service is unavailable, a typed ``course_service_unavailable`` error is
        returned rather than raising.
        """
        try:
            try:
                from gui.services import course_service  # noqa: PLC0415
            except ImportError as e:
                return _err(str(e), detail="course_service_unavailable")
            return json.dumps(course_service.list_courses(), indent=2)
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_list_courses failed")
            return _err(str(e))

    @mcp.tool()
    async def gui_get_objectives(course: str) -> str:
        """
        Return the synthesized objectives doc for a course.

        Args:
            course: The course name or project id.

        Lazily imports ``gui.services.course_service``; returns a typed
        ``course_service_unavailable`` error if the service can't be loaded.
        """
        try:
            try:
                from gui.services import course_service  # noqa: PLC0415
            except ImportError as e:
                return _err(str(e), detail="course_service_unavailable")
            return json.dumps(course_service.get_objectives(course), indent=2)
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_get_objectives failed")
            return _err(str(e))

    @mcp.tool()
    async def gui_post_event(kind: str, payload_json: str = "{}") -> str:
        """
        Append a ``claude``-sourced event to the GUI activity feed.

        Shows up in the GUI's Activity tab — the human-visible side of the
        Claude<->GUI bridge.

        Args:
            kind: Event kind label (e.g. ``message``, ``note``, ``run_ready``).
            payload_json: JSON object string carrying the event payload.

        Returns the stored event record (with its ``seq`` + ``ts``).
        """
        try:
            try:
                payload = json.loads(payload_json) if payload_json else {}
            except json.JSONDecodeError as e:
                return _err(f"payload_json is not valid JSON: {e}", detail="validation_error")
            if not isinstance(payload, dict):
                return _err("payload_json must decode to a JSON object")
            record = shared_state.append_event("claude", kind, payload)
            return json.dumps(record, indent=2)
        except ValueError as e:
            return _err(str(e), detail="validation_error")
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_post_event failed")
            return _err(str(e))

    @mcp.tool()
    async def gui_read_events(since: int = 0) -> str:
        """
        Read GUI activity events with ``seq >= since`` (append order).

        Lets Claude observe human messages posted from the GUI's Activity tab
        (``gui``-sourced events), symmetric to ``gui_post_event``.

        Args:
            since: Return events whose ``seq`` is at least this value (0 = all).
        """
        try:
            events = shared_state.read_events(int(since))
            return json.dumps(events, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.exception("gui_read_events failed")
            return _err(str(e))

    logger.info("GUI tools registered")
