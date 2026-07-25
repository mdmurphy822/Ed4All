"""Operator-assistant MCP tools — the campaign surface for external clients.

Registered via ``register_assistant_tools(mcp)`` from ``MCP/server.py`` alongside
the other ``register_*`` tool modules. These are **thin wrappers** over the
already-hardened operator-assistant harness in ``lib/assistant`` — they expose
the assistant's BOUNDED campaign surface to an external MCP client (e.g. a
Claude Code session) WITHOUT widening it.

Design invariant (do not violate): the wrapper NEVER reimplements validation.
Every mutating campaign tool routes through
``lib.assistant.campaign_tools.dispatch_campaign_tool`` — the single validated
choke point that already enforces the per-tool argument whitelist, the
required-argument checks, redaction + output clipping, and the never-raise
self-diagnosis contract. Path confinement (``validate_corpus_path``), the flag
allowlist (``campaign_flags.validate_overlay``), plain-resume-only (``--force``
is not a parameter anywhere in the campaign layer and cannot be smuggled in),
and the single-owner ``preflight_launch`` gate are all inherited from the
underlying implementation — the MCP layer adds NOTHING to that surface.

``assistant_ask`` runs one engine turn in ``campaign-tick`` (READONLY) mode, so
an MCP client can OBSERVE + REPORT via the LLM but cannot mutate through it; the
deterministic mutation tools above are the only mutation path.

IMPORTANT: this module is import-safe. It defers every ``lib.assistant`` import
into the tool bodies (mirroring ``gui_tools``' lazy service import), so the
module never hard-fails at import and an unavailable dependency surfaces as a
typed error record rather than raising.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path so ``lib`` imports resolve the same way the other
# tool modules resolve ``lib`` / ``MCP`` packages.
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _err(message: str, **extra: Any) -> str:
    """Serialize a typed error record (never a fabricated success)."""
    payload = {"error": message}
    payload.update(extra)
    return json.dumps(payload)


def _dispatch(name: str, arguments: dict) -> str:
    """Route a campaign tool through the validated dispatch choke point.

    Delegates to ``lib.assistant.campaign_tools.dispatch_campaign_tool`` (looked
    up as a MODULE ATTRIBUTE so it stays monkeypatchable in tests), which owns
    the whitelist / required-arg / redaction / never-raise contract. The wrapper
    adds no validation of its own.
    """
    try:
        from lib.assistant import campaign_tools  # noqa: PLC0415
    except ImportError as exc:
        return _err(str(exc), detail="assistant_unavailable")
    return campaign_tools.dispatch_campaign_tool(name, arguments)


def register_assistant_tools(mcp):
    """Register the operator-assistant campaign tools with the MCP server."""

    @mcp.tool()
    async def assistant_campaign_queue() -> str:
        """
        Return the campaign book/corpus queue + manifest states.

        Organized view: manifest book counts + per-status slugs (pending in
        wave/pages order), prepared overlay names, and the count of OPEN review
        reports (with blocking book slugs). Read-only.
        """
        return _dispatch("campaign_queue", {})

    @mcp.tool()
    async def assistant_campaign_run_status(wf_id: str = "") -> str:
        """
        Return active/recent campaign run states (read-only monitor).

        Args:
            wf_id: Optional workflow id (``WF-YYYYMMDD-xxxxxxxx``). Omit to
                summarize every active ``state/workflows`` record + recent
                launches; pass one to get that record + the tail of its log.
                An invalid id is refused by the underlying validator.
        """
        return _dispatch("campaign_run_status", {"run_id": wf_id})

    @mcp.tool()
    async def assistant_campaign_prepare_run(
        corpus: str,
        overrides_json: str = "{}",
        note: str = "",
    ) -> str:
        """
        Prepare a run: write a VALIDATED env overlay (no script, no arbitrary file).

        Validates the corpus path (confinement) and the flag overrides against
        the campaign flag allowlist, then writes a fixed-shape JSON overlay under
        ``pending-runs/<name>.json``. Launch it later with
        ``assistant_campaign_launch_run``.

        Args:
            corpus: Corpus path to validate + record (feeds ``--corpus``).
            overrides_json: JSON object string of campaign flag overrides
                (the env overlay). Defaults to ``"{}"``. Each key/value is
                validated against the flag allowlist by the underlying tool.
            note: Optional free-text note recorded on the overlay.
        """
        try:
            overrides = json.loads(overrides_json) if overrides_json else {}
        except json.JSONDecodeError as exc:
            return _err(f"overrides_json is not valid JSON: {exc}", detail="validation_error")
        if not isinstance(overrides, dict):
            return _err("overrides_json must decode to a JSON object")
        args: dict = {"corpus": corpus, "env_overlay": overrides}
        if note:
            args["note"] = note
        return _dispatch("campaign_prepare_run", args)

    @mcp.tool()
    async def assistant_campaign_launch_run(prepared_id: str) -> str:
        """
        Launch a PREPARED run overlay by name (full preflight).

        Re-validates the (untrusted) on-disk overlay, runs the mandatory
        single-owner ``preflight_launch`` gate, then spawns the fixed launcher
        argv detached. All of that is inherited — the wrapper only passes the
        overlay name.

        Args:
            prepared_id: The prepared overlay name (as returned by
                ``assistant_campaign_prepare_run``).
        """
        return _dispatch("campaign_launch_run", {"name": prepared_id})

    @mcp.tool()
    async def assistant_campaign_resume_run(wf_id: str) -> str:
        """
        Resume a paused run with a PLAIN ``--resume`` (never ``--force``).

        ``--force`` is not a parameter here, is not in the underlying tool's
        argument whitelist, and cannot be added — a plain resume is the only
        possible outcome. Preflight (STOP_ALL + single-owner) is enforced by the
        underlying tool.

        Args:
            wf_id: The paused run id (``WF-YYYYMMDD-xxxxxxxx``).
        """
        return _dispatch("campaign_resume_run", {"run_id": wf_id})

    @mcp.tool()
    async def assistant_campaign_stop_run(wf_id: str) -> str:
        """
        Gracefully stop one run (drops the stop sentinel via ``ed4all stop``).

        Args:
            wf_id: The run id to stop (``WF-YYYYMMDD-xxxxxxxx``).
        """
        return _dispatch("campaign_stop_run", {"run_id": wf_id})

    @mcp.tool()
    async def assistant_campaign_report(kind: str, payload_json: str = "{}") -> str:
        """
        File a structured review-queue report for a human reviewer.

        Args:
            kind: Report kind (validated against the underlying ``REPORT_KINDS``
                enum — an unknown kind is refused).
            payload_json: JSON object string carrying the report body. Must
                include ``summary``; may include ``run_id``, ``phase``,
                ``error_class``, ``log_excerpt``, ``book_slug``. Any key outside
                that whitelist is dropped by the underlying dispatcher.
        """
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except json.JSONDecodeError as exc:
            return _err(f"payload_json is not valid JSON: {exc}", detail="validation_error")
        if not isinstance(payload, dict):
            return _err("payload_json must decode to a JSON object")
        args = {"kind": kind, **payload}
        return _dispatch("campaign_report", args)

    @mcp.tool()
    async def assistant_campaign_prepare_training(slug: str) -> str:
        """
        Validate a book's Stage-B LoRA-training readiness (mutates nothing).

        Runs every underlying prepare check loudly: slug in the campaign
        manifest with a trainable (built/review) status, non-empty instruction
        pairs, base-model registry resolution (unknown name = loud error, never
        a fallback), the reviewer's training-approval marker (the assistant
        never writes it), training-env readiness (transformers>=4.57 + Mamba
        kernels), STOP_ALL, and single-owner (no live build, no live training).

        Args:
            slug: The campaign book slug.
        """
        return _dispatch("campaign_prepare_training", {"slug": slug})

    @mcp.tool()
    async def assistant_campaign_launch_training(slug: str) -> str:
        """
        Launch a Stage-B LoRA training run for one book (full preflight).

        Re-runs EVERY prepare check (a stale prepare is never trusted), then
        docker-stops every registered vLLM seat and VERIFIES none still serves
        (fails loudly, no launch, if one does), then spawns the fixed argv
        ``ed4all run trainforge_train --course-code <slug> --base-model
        <validated>`` detached. All inherited from the underlying tool — the
        wrapper only passes the slug.

        Args:
            slug: The campaign book slug.
        """
        return _dispatch("campaign_launch_training", {"slug": slug})

    @mcp.tool()
    async def assistant_campaign_training_status(slug: str = "") -> str:
        """
        Return recent/active Stage-B training runs (read-only monitor).

        Summarizes the campaign's ``kind: "training"`` launch rows, each row's
        workflow record status, a bounded log tail for the newest, and the
        training-env readiness summary.

        Args:
            slug: Optional book slug to filter on. Omit for all training runs.
        """
        args = {"slug": slug} if slug else {}
        return _dispatch("campaign_training_status", args)

    @mcp.tool()
    async def assistant_seat_status() -> str:
        """
        Return the assistant's dynamically-resolved seat (read-only probe).

        Runs the same priority-walk seat resolution the assistant would use
        right now (``ED4ALL_SEAT_BASE_URLS`` priority seats first, else the
        ``ED4ALL_ASSISTANT_*`` fallback) and reports which seat / served model
        would answer, whether it is live, and how it resolved. No mutation, no
        autostart — a non-loopback config surfaces as a typed error.
        """
        try:
            from lib.assistant.client import (  # noqa: PLC0415
                AssistantProviderNotLocal,
                resolve_active_seat,
            )
        except ImportError as exc:
            return _err(str(exc), detail="assistant_unavailable")
        try:
            seat = resolve_active_seat(force=True)
        except AssistantProviderNotLocal as exc:
            return _err(str(exc), detail="provider_not_local")
        except Exception as exc:  # noqa: BLE001
            logger.exception("assistant_seat_status failed")
            return _err(str(exc))
        return json.dumps(
            {
                "seat_name": seat.seat_name,
                "base_url": seat.base_url,
                "model": seat.model,
                "live": bool(seat.live),
                "source": seat.source,
            },
            indent=2,
        )

    @mcp.tool()
    async def assistant_ask(question: str) -> str:
        """
        Ask the assistant one campaign question (READONLY ``campaign-tick`` turn).

        Runs a single engine turn in ``campaign-tick`` mode: the LLM may OBSERVE
        + REPORT (read-only campaign tools only — queue / run-status / report);
        every mutating campaign tool is refused on this path, so an MCP client
        cannot mutate the campaign through the LLM. Direct mutations go through
        the explicit ``assistant_campaign_*`` tools above.

        Args:
            question: The operator question / instruction for this turn.

        Returns a bounded JSON record: the reply, the tool names invoked, round
        + token counts, and the seat / model that answered. If no assistant seat
        is serving, a typed error with the start hint is returned (no autostart).
        """
        if not isinstance(question, str) or not question.strip():
            return _err("question must be a non-empty string")
        try:
            from lib.assistant.client import (  # noqa: PLC0415
                AssistantProviderNotLocal,
                AssistantSeatUnavailable,
            )
            from lib.assistant.engine import AssistantEngine  # noqa: PLC0415
        except ImportError as exc:
            return _err(str(exc), detail="assistant_unavailable")
        try:
            engine = AssistantEngine(mode="campaign-tick")
            # No autostart from an MCP call — surface a clean typed error if the
            # seat is down rather than spawning a container.
            engine.ensure_seat(allow_autostart=False)
            turn = engine.run_turn(question)
        except AssistantProviderNotLocal as exc:
            return _err(str(exc), detail="provider_not_local")
        except AssistantSeatUnavailable as exc:
            return _err(str(exc), detail="seat_unavailable")
        except Exception as exc:  # noqa: BLE001
            logger.exception("assistant_ask failed")
            return _err(str(exc))
        return json.dumps(
            {
                "reply": turn.reply,
                "mode": "campaign-tick",
                "rounds": turn.rounds,
                "tools_called": [t.get("tool") for t in (turn.tool_calls or [])],
                "prompt_tokens": turn.prompt_tokens,
                "completion_tokens": turn.completion_tokens,
                "seat_name": turn.seat_name,
                "model": turn.model,
            },
            indent=2,
        )

    logger.info("Assistant tools registered")
