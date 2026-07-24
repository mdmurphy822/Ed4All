"""UI-agnostic operator-assistant engine (consumed by the CLI verb and,
next, the Control-Plane GUI chat panel).

Design contract:

* ZERO UI coupling — no ``input()`` / ``print()``. One call
  (:meth:`AssistantEngine.run_turn`) takes the user message + the caller's
  conversation state and returns an :class:`AssistantTurn` (reply text,
  updated serializable message list, tool-call trace, token counts, the
  resolved seat/model that answered). A stateless HTTP endpoint can
  round-trip ``AssistantTurn.messages`` as-is.
* The ENGINE owns the sandbox, so every consumer inherits it identically
  and cannot widen it: the system prompt, the typed tool whitelist +
  dispatch (:mod:`lib.assistant.tools`, plus the campaign tool set in
  campaign mode — :mod:`lib.assistant.campaign_tools`), the loopback guard +
  dynamic seat resolution / lazy-start policy (:mod:`lib.assistant.client`,
  surfaced via :meth:`ensure_seat`), the per-turn
  :class:`~lib.decision_capture.DecisionCapture` wiring, and the hard
  :data:`MAX_TOOL_ROUNDS` cap.
* Project law: every LLM call site logs a decision. Each ``run_turn`` emits
  exactly one ``llm_chat_call`` decision whose rationale interpolates the
  live signals (mode, model id, resolved seat, seat URL, rounds, tool
  names, token counts).
* Campaign mode: the assistant ORGANIZES / ARRANGES / MONITORS the
  multi-book campaign — Claude (dev sessions) still FIXES and REVIEWS. The
  campaign system prompt states that division of labor and its hard
  constraints; the campaign tools are UNREACHABLE outside campaign mode
  (dispatch routes them only when ``mode == "campaign"``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lib.assistant.client import (
    AssistantClient,
    AssistantProviderNotLocal,
    AssistantSeatUnavailable,
    autostart_seat,
    resolve_assistant_autostart,
    seat_start_hint,
)

# T1's dynamic seat-resolution API (S1) lands in parallel in
# ``lib.assistant.client``. Import it directly; a small bridge keeps the
# engine importable during the parallel build (and until integration), when
# the sibling symbols may not be present yet. Once T1 has landed the real
# implementations are used unchanged.
try:  # pragma: no cover - exercised via the real symbols post-integration
    from lib.assistant.client import (
        ResolvedSeat,
        reset_seat_cache,
        resolve_active_seat,
    )
except ImportError:  # pragma: no cover - parallel-build bridge only
    from lib.assistant.client import (
        resolve_assistant_base_url,
        resolve_assistant_model,
        resolve_assistant_seat,
    )

    @dataclass(frozen=True)
    class ResolvedSeat:  # minimal S1-shaped bridge until client.py lands it
        seat_name: Optional[str]
        base_url: str
        model: str
        live: bool
        source: str

    def resolve_active_seat(
        *,
        force: bool = False,
        probe: Optional[Any] = None,
        model_reader: Optional[Any] = None,
    ) -> "ResolvedSeat":
        """Bridge: report the static assistant seat with a live probe."""
        from lib.assistant.client import seat_is_serving  # noqa: PLC0415

        base_url = resolve_assistant_base_url()
        return ResolvedSeat(
            resolve_assistant_seat(),
            base_url,
            resolve_assistant_model(),
            bool(seat_is_serving(base_url)),
            "fallback",
        )

    def reset_seat_cache() -> None:  # bridge no-op
        return None

# T3's campaign tool set (S4) lands in parallel in
# ``lib.assistant.campaign_tools``. Same bridge posture: import the real
# module when present, otherwise fall back to an empty catalog + a refusing
# dispatcher so the engine imports and operator/debug modes are unaffected.
try:  # pragma: no cover - exercised via the real module post-integration
    from lib.assistant.campaign_tools import (
        CAMPAIGN_READONLY_TOOL_SCHEMAS,
        CAMPAIGN_TOOL_REGISTRY,
        CAMPAIGN_TOOL_SCHEMAS,
        dispatch_campaign_tool,
    )
except ImportError:  # pragma: no cover - parallel-build bridge only
    CAMPAIGN_TOOL_REGISTRY: Dict[str, Any] = {}
    CAMPAIGN_TOOL_SCHEMAS: List[Dict[str, Any]] = []
    CAMPAIGN_READONLY_TOOL_SCHEMAS: List[Dict[str, Any]] = []

    def dispatch_campaign_tool(
        name: str, arguments: Dict[str, Any], *, readonly: bool = False
    ) -> str:
        return f"Refused: tool {name!r} is not in the campaign tool whitelist."

from lib.assistant.tools import READONLY_TOOL_SCHEMAS, TOOL_SCHEMAS, dispatch_tool

logger = logging.getLogger(__name__)

#: Hard cap on tool-call rounds per user turn — the model cannot loop the
#: whitelist indefinitely.
MAX_TOOL_ROUNDS = 6

#: One-line-per-tool catalog, grouped — this is the model's map of its own
#: whitelist, so keep every line short and literal (a 30B model follows it).
_TOOL_CATALOG = (
    "Your tools, grouped:\n"
    "STATUS: campaign_status (book campaign counts); run_status (recent "
    "runs); seat_status (vLLM seats serving/down).\n"
    "REPORTS: list_workflows (workflow phase lists); run_report <run_id> "
    "(one run's status + phases); gate_report <run_id> [phase] (validation "
    "gates + issue codes); tail_log <slug-or-run-id> [lines] (log tail, max "
    "200); doctor [group] (diagnostics); build_cost <slug> (per-phase "
    "wall-clock/tokens); aggregator_report <slug> <name> (promotion_chain, "
    "coverage_map, kg_quality, ...); flag_lookup <name> (behavior-flag "
    "semantics).\n"
    "LIBRARY: library_courses (LibV2 slugs + index presence); "
    "course_objectives <slug> (TO/CO summary); ask_course <slug> <question> "
    "(grounded course Q&A).\n"
    "SEAT-SETUP: seat_env_doctor (audit the seat-swap env vars); "
    "list_gpu_containers (docker container discovery); generate_seat_env "
    "(returns a ready-to-source env snippet — the USER saves and sources "
    "it; nothing is written).\n"
    "ACTIONS: start_next_book; start_book <slug>; resume_run <run_id>; "
    "stop_run <run_id>; pause_all; clear_stop_all (needs explicit human "
    "confirmation first); start_seat <seat>; stop_seat <seat>; "
    "support_bundle.\n"
    "HELP: get_help <topic> (run, resume, stop, campaign, seats, "
    "seat-setup, reports, library, debug)."
)

#: Campaign tool catalog — mirrors ``_TOOL_CATALOG`` style: the 7 S4 campaign
#: tools plus the still-available read-only base tools. One short literal line
#: per group for a 30B model.
_CAMPAIGN_TOOL_CATALOG = (
    "Your campaign tools, grouped:\n"
    "ORGANIZE: campaign_queue (book campaign counts + per-status slugs, "
    "pending-run overlay names, count of open review reports).\n"
    "ARRANGE: campaign_prepare_run <corpus> [env_overlay] [note] (writes a "
    "VALIDATED JSON env overlay under pending-runs/ — data only, never a "
    "script); campaign_launch_run <name> (launch a prepared overlay, "
    "detached, after the single-owner + STOP_ALL preflight).\n"
    "MONITOR: campaign_run_status [run_id] (read-only active-run + recent "
    "log-tail summary).\n"
    "RUN-CONTROL: campaign_resume_run <run_id> (PLAIN resume only — there is "
    "no --force); campaign_stop_run <run_id> (graceful ed4all stop).\n"
    "TRAIN: campaign_prepare_training <slug> (validate a book's LoRA-training "
    "readiness — manifest status, instruction pairs, base model, the "
    "reviewer's approval marker, training env, single-owner; mutates "
    "nothing); campaign_launch_training <slug> (re-validates everything, "
    "docker-stops every registered vLLM seat, verifies the card is free, "
    "then launches ed4all run trainforge_train detached); "
    "campaign_training_status [slug] (read-only training-run monitor).\n"
    "REVIEW: campaign_report <kind> <summary> [run_id] [phase] "
    "[error_class] [log_excerpt] [book_slug] (queue a run_failure / "
    "run_paused / gate_anomaly / assistant_error / campaign_note report for "
    "Claude to review).\n"
    "READ-ONLY BASE TOOLS stay available: campaign_status, run_status, "
    "seat_status, list_workflows, run_report, gate_report, tail_log, "
    "doctor, build_cost, aggregator_report, flag_lookup, library_courses, "
    "course_objectives, ask_course, get_help."
)

SYSTEM_PROMPT = (
    "You are the Ed4All operator assistant, a narrow helper for the Ed4All "
    "course-pipeline system. Your ONLY jobs: (1) answer basic questions "
    "about operating Ed4All (runs, resume/stop, the book campaign, vLLM "
    "seats, reports) using the get_help tool where useful; (2) report "
    "status and diagnostics via the STATUS/REPORTS/LIBRARY tools; (3) "
    "perform the whitelisted ACTIONS when the user asks (each action tool "
    "enforces its own safety guards and may refuse — relay the refusal). "
    + _TOOL_CATALOG + " "
    "REFUSE anything else — general coding, content generation, questions "
    "unrelated to Ed4All, requests to run shell commands or read/write "
    "files. You have NO shell and NO file access; only the listed tools. "
    "Never invent run ids, slugs, seat names, or statuses — check with a "
    "tool. Never call clear_stop_all with confirm=true unless the human "
    "explicitly confirmed in this conversation. Keep answers short and "
    "concrete."
)

DEBUG_SYSTEM_PROMPT = (
    "You are the Ed4All operator assistant in DEBUG MODE: a failed pipeline "
    "run's diagnostic context is provided below as a system context block. "
    "Your job is root-cause diagnosis of THAT failure: (1) explain the "
    "failure chain (which phase failed, which gates/errors fired, what the "
    "log shows) in plain language; (2) correlate gate issue codes and "
    "thresholds with their flag/threshold semantics — use flag_lookup for "
    "any ED4ALL_* flag you mention; (3) pull more evidence only through the "
    "tools (run_report, gate_report, tail_log, doctor, build_cost, "
    "aggregator_report); (4) propose a SMALL set of bounded next actions "
    "using ONLY the registered action tools (e.g. resume_run after a "
    "transient failure, start_book to retry, doctor to verify the "
    "environment) — never invent commands outside the whitelist, and let "
    "the human trigger mutating actions. If the failure mentions "
    "seat_schedule, a seat that could not be brought up coherent, or a "
    "seat/container mapping, run seat_env_doctor FIRST — a misconfigured "
    "seat-swap env is the usual root cause there. "
    + _TOOL_CATALOG + " "
    "You have NO shell and NO file access; only the listed tools. Never "
    "invent run ids, gate ids, or log lines — quote only what the context "
    "or a tool returned. Keep answers short, structured, and concrete."
)

CAMPAIGN_SYSTEM_PROMPT = (
    "You are the Ed4All campaign assistant. You ORGANIZE, ARRANGE, and "
    "MONITOR the multi-book course-build campaign: the assistant organizes "
    "the input queue, arranges runs as validated data, and monitors their "
    "progress. You do NOT fix anything. Claude (the developer sessions) "
    "FIXES errors and REVIEWS your reports — the division of labor is hard: "
    "you organize, Claude repairs. "
    "You NEVER author scripts or code, and you never write arbitrary files. "
    "Environment arrangements are validated JSON DATA overlays only, "
    "composed through campaign_prepare_run (an allowlisted key + a "
    "conservative value charset — never author a shell script or any file "
    "yourself). "
    "Resume is ALWAYS a plain resume (`ed4all run --resume <id>`): there is "
    "no --force — it does not exist for you and you must never ask for it. "
    "Your only run controls are plain resume and graceful stop. Anything "
    "beyond resume/stop — a failure to diagnose, a gate anomaly, a code or "
    "content repair — becomes a campaign_report queued for Claude to "
    "review; you never repair it yourself. "
    "You also run the campaign's Stage-B LoRA training. The per-book "
    "lifecycle is: build the course (campaign_launch_run) -> the build "
    "synthesizes instruction pairs -> a human/Claude reviewer approves "
    "training -> seat teardown -> train (campaign_launch_training) -> the "
    "training workflow's own eval + promotion review. HARD training rules: "
    "NEVER launch training while a build runs, and NEVER launch a build "
    "while training runs — the GPU is single-owner and the tools enforce it. "
    "The approval marker "
    "<campaign dir>/review-queue/approvals/<slug>.training-approved "
    "is written ONLY by the human/Claude reviewer — you NEVER write it, you "
    "have no tool that can, and you never claim it exists without "
    "campaign_prepare_training confirming it. Always run "
    "campaign_prepare_training first and relay any failing check; a "
    "training failure becomes a campaign_report, never a retry loop. "
    "Never invent run ids, book slugs, or paths — check with a tool "
    "(campaign_queue / campaign_run_status / campaign_training_status) "
    "before you name one. "
    + _CAMPAIGN_TOOL_CATALOG + " "
    "You have NO shell and NO file access beyond these tools; every tool "
    "enforces its own guards and may refuse — relay the refusal. Keep "
    "answers short and concrete."
)

#: Read-only campaign-tick tool catalog — the OBSERVE + REPORT slice exposed
#: to the pilot's per-tick review turn. Mirrors ``_CAMPAIGN_TOOL_CATALOG``
#: style; only read/status/report tools appear.
_CAMPAIGN_TICK_TOOL_CATALOG = (
    "Your tick tools (OBSERVE + REPORT only), grouped:\n"
    "ORGANIZE: campaign_queue (book campaign counts + per-status slugs, "
    "prepared overlays, open review-report count).\n"
    "MONITOR: campaign_run_status [run_id] (read-only active-run + log-tail "
    "summary); campaign_training_status [slug] (read-only LoRA-training runs "
    "+ env-readiness summary).\n"
    "REVIEW: campaign_report <kind> <summary> [run_id] [phase] [error_class] "
    "[log_excerpt] [book_slug] (queue a report for Claude to review).\n"
    "READ-ONLY BASE TOOLS also available: campaign_status, run_status, "
    "seat_status, list_workflows, run_report, gate_report, tail_log, doctor, "
    "build_cost, aggregator_report, flag_lookup, library_courses, "
    "course_objectives, ask_course, get_help."
)

CAMPAIGN_TICK_SYSTEM_PROMPT = (
    "You are the Ed4All campaign pilot's per-tick monitoring assistant. You "
    "OBSERVE and REPORT ONLY. The pilot's DETERMINISTIC policy — not you — "
    "owns every run mutation: bounded auto-resume, queue advance, and "
    "halt-on-failure. You have NO launch, resume, or stop tools this tick; "
    "you cannot start seats, prepare runs, or launch training — training is "
    "OBSERVABLE only, via the read-only campaign_training_status. Your job "
    "is to review the "
    "campaign snapshot, pull read-only status through the tools, and file a "
    "campaign_report for anything that needs Claude's attention — never to "
    "act on it yourself. Never invent run ids, book slugs, or paths — check "
    "with campaign_queue / campaign_run_status before naming one. "
    + _CAMPAIGN_TICK_TOOL_CATALOG + " "
    "Every tool enforces its own guards and may refuse — relay the refusal. "
    "Keep answers short and concrete."
)


@dataclass
class AssistantTurn:
    """Result of one user turn through the engine."""

    #: Final assistant reply text for display.
    reply: str
    #: Updated conversation state (plain dicts, JSON-serializable): the
    #: caller-supplied history plus this turn's user / assistant / tool
    #: messages. Pass it back as ``history`` on the next call. The system
    #: prompt is NOT included — the engine prepends it on the wire.
    messages: List[Dict[str, Any]] = field(default_factory=list)
    #: Tool-call trace: ``[{"tool": ..., "arguments": ..., "result": ...}]``.
    tool_calls: List[Dict[str, str]] = field(default_factory=list)
    #: Chat-completion rounds consumed (1 = no tool calls).
    rounds: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: The logical seat name that answered this turn (``None`` for a static
    #: client that never dynamically resolved). JSON-serializable.
    seat_name: Optional[str] = None
    #: The served model id that answered this turn (``None`` when the client
    #: carries no model). JSON-serializable.
    model: Optional[str] = None


class AssistantEngine:
    """The reusable chat/tool loop. One instance per consumer session.

    Args:
        client: Optional pre-built :class:`AssistantClient` (tests inject one
            with a fake transport). Default: a fresh client from the
            ``ED4ALL_ASSISTANT_*`` env resolution (loopback-enforced at
            construction).
        capture: Optional pre-built ``DecisionCapture``. Default: one is
            created lazily (course_code ``assistant``, phase
            ``assistant_session``, tool ``assistant``) on first turn.
        mode: ``"operator"`` (default), ``"debug"``, or ``"campaign"``. Debug
            mode swaps the system prompt to root-cause diagnosis and injects
            ``debug_context`` as the first assistant-visible context block on
            every wire call. Campaign mode swaps the system prompt to the
            organize/arrange/monitor role and additionally exposes the
            campaign tool set. Neither grants any capability beyond the
            relevant tool whitelist.
        debug_context: The composed context text from
            :func:`lib.assistant.debug_context.build_debug_context`
            (its ``"text"`` field). Ignored outside debug mode.
    """

    #: The supported engine modes. ``campaign-tick`` is the pilot's restricted
    #: OBSERVE + REPORT surface (read-only base tools + read-only campaign
    #: tools only); the interactive ``campaign`` session keeps the full set.
    MODES = ("operator", "debug", "campaign", "campaign-tick")

    def __init__(
        self,
        *,
        client: Optional[AssistantClient] = None,
        capture: Optional[Any] = None,
        mode: str = "operator",
        debug_context: Optional[str] = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(
                f"AssistantEngine mode must be one of {self.MODES}, got {mode!r}"
            )
        self.client = client if client is not None else AssistantClient()
        self._capture = capture
        self.mode = mode
        self.debug_context = str(debug_context) if debug_context else None

    @property
    def system_prompt(self) -> str:
        """The mode-selected system prompt (engine-owned, never caller-set)."""
        if self.mode == "campaign":
            return CAMPAIGN_SYSTEM_PROMPT
        if self.mode == "campaign-tick":
            return CAMPAIGN_TICK_SYSTEM_PROMPT
        if self.mode == "debug":
            return DEBUG_SYSTEM_PROMPT
        return SYSTEM_PROMPT

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """The mode-selected tool whitelist sent on the wire. Campaign mode
        adds the full campaign tool set; ``campaign-tick`` exposes only the
        read-only base tools + read-only campaign tools (no run/seat mutation);
        every other mode is base tools only."""
        if self.mode == "campaign":
            return [*TOOL_SCHEMAS, *CAMPAIGN_TOOL_SCHEMAS]
        if self.mode == "campaign-tick":
            return [*READONLY_TOOL_SCHEMAS, *CAMPAIGN_READONLY_TOOL_SCHEMAS]
        return TOOL_SCHEMAS

    def _wire_prefix(self) -> List[Dict[str, Any]]:
        """System prompt (+ the debug-context block in debug mode) — always
        prepended on the wire, never stored in the caller-held history."""
        prefix: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if self.mode == "debug" and self.debug_context:
            prefix.append(
                {
                    "role": "system",
                    "content": "FAILED-RUN DEBUG CONTEXT:\n" + self.debug_context,
                }
            )
        return prefix

    # ------------------------------------------------------------------ #
    # Seat policy
    # ------------------------------------------------------------------ #

    def ensure_seat(self, *, allow_autostart: bool = True) -> None:
        """Ensure an assistant seat is serving; lazy-start it when allowed.

        Policy (owned here so every consumer inherits it): resolve the active
        seat dynamically (S1 priority walk — Super when live, else nano, else
        the fallback env). If it is live, return. Otherwise, if
        ``allow_autostart`` AND ``ED4ALL_ASSISTANT_AUTOSTART`` is truthy,
        lazy-start the assistant seat via the lifecycle lib's coherent-start
        path (liveness ceiling + bounded coherence probes) and clear the seat
        cache so the next resolution sees the freshly-started seat. Otherwise
        raise :class:`AssistantSeatUnavailable` with the start instructions —
        never a silent degrade. A non-loopback config surfaces as
        :class:`AssistantProviderNotLocal` (re-raised, never swallowed).
        """
        try:
            seat = resolve_active_seat(force=True)
        except AssistantProviderNotLocal:
            raise
        if seat.live:
            return
        base_url = seat.base_url
        if allow_autostart and resolve_assistant_autostart():
            result = autostart_seat()
            if getattr(result, "ok", False):
                reset_seat_cache()
                return
            raise AssistantSeatUnavailable(
                f"Autostart of the assistant seat failed "
                f"(reason: {getattr(result, 'reason', 'unknown')}). "
                + seat_start_hint(base_url)
            )
        raise AssistantSeatUnavailable(seat_start_hint(base_url))

    # ------------------------------------------------------------------ #
    # Decision capture
    # ------------------------------------------------------------------ #

    def _get_capture(self):
        if self._capture is None:
            from lib.decision_capture import DecisionCapture  # noqa: PLC0415

            self._capture = DecisionCapture(
                course_code="assistant",
                phase="assistant_session",
                tool="assistant",
            )
        return self._capture

    def _resolved_seat_label(self) -> str:
        """The logical seat name that answered (or ``"static"``)."""
        seat = getattr(self.client, "last_seat", None)
        if seat is None:
            return "static"
        return getattr(seat, "seat_name", None) or "static"

    # ------------------------------------------------------------------ #
    # The turn loop
    # ------------------------------------------------------------------ #

    def run_turn(
        self,
        user_text: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AssistantTurn:
        """Run one user turn: chat → (tool rounds ≤ 6) → final reply.

        ``history`` is the caller-held message list from a prior
        :class:`AssistantTurn` (or ``None`` for a fresh conversation).
        """
        user_text = str(user_text)
        messages: List[Dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": user_text})

        tool_schemas = self.tool_schemas
        trace: List[Dict[str, str]] = []
        prompt_tokens = 0
        completion_tokens = 0
        rounds = 0
        reply: Optional[str] = None

        while rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            body = self.client.chat(
                [*self._wire_prefix(), *messages],
                tools=tool_schemas,
            )
            usage = body.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            message = body["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                reply = str(message.get("content") or "").strip()
                messages.append({"role": "assistant", "content": reply})
                break

            # Record the assistant message that requested the tools, then
            # answer every tool call through the whitelist dispatcher.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                name = str(fn.get("name") or "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                except (ValueError, TypeError) as exc:
                    result = (
                        f"Refused: malformed tool arguments for {name!r} "
                        f"({exc}). Send a JSON object."
                    )
                else:
                    # Campaign tools are UNREACHABLE outside the campaign
                    # modes: only in ``campaign`` / ``campaign-tick`` does a
                    # campaign-registered name route to the campaign
                    # dispatcher; every other name (and every other mode) goes
                    # to the base whitelist dispatcher, whose unknown-name
                    # refusal is the desired behavior. ``campaign-tick`` passes
                    # ``readonly=True`` so mutating tools are refused; full
                    # ``campaign`` mode passes ``campaign_mode=True`` so the
                    # base seat guard confines start_seat to the assistant's
                    # own chat seat. Operator/debug modes keep the exact
                    # 2-arg base-dispatch call.
                    campaign_tick = self.mode == "campaign-tick"
                    is_campaign_tool = name in CAMPAIGN_TOOL_REGISTRY
                    if self.mode == "campaign" and is_campaign_tool:
                        result = dispatch_campaign_tool(name, arguments)
                    elif campaign_tick and is_campaign_tool:
                        result = dispatch_campaign_tool(name, arguments, readonly=True)
                    elif self.mode == "campaign":
                        result = dispatch_tool(name, arguments, campaign_mode=True)
                    elif campaign_tick:
                        result = dispatch_tool(name, arguments, readonly=True)
                    else:
                        result = dispatch_tool(name, arguments)
                trace.append(
                    {
                        "tool": name,
                        "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or f"call_{len(trace)}"),
                        "content": result,
                    }
                )

        if reply is None:
            reply = (
                f"Tool-call round limit reached ({MAX_TOOL_ROUNDS} rounds) "
                f"without a final answer. Partial tool results: "
                + ("; ".join(t["tool"] for t in trace) or "none")
                + ". Please narrow the request."
            )
            messages.append({"role": "assistant", "content": reply})

        self._log_turn_decision(
            user_text=user_text,
            reply=reply,
            trace=trace,
            rounds=rounds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        seat = getattr(self.client, "last_seat", None)
        return AssistantTurn(
            reply=reply,
            messages=messages,
            tool_calls=trace,
            rounds=rounds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            seat_name=(seat.seat_name if seat is not None else None),
            model=getattr(self.client, "model", None),
        )

    def _log_turn_decision(
        self,
        *,
        user_text: str,
        reply: str,
        trace: List[Dict[str, str]],
        rounds: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """One ``llm_chat_call`` decision per turn — dynamic rationale, per
        the LLM call-site instrumentation contract. Capture failure is loud
        in the log but never eats the reply."""
        tool_names = [t["tool"] for t in trace]
        query_hash = hashlib.sha256(user_text.encode("utf-8")).hexdigest()[:12]
        seat_label = self._resolved_seat_label()
        try:
            self._get_capture().log_decision(
                decision_type="llm_chat_call",
                decision=(
                    f"assistant_turn: {rounds} round(s), "
                    f"{len(tool_names)} tool call(s) "
                    f"({', '.join(tool_names) or 'none'})"
                ),
                rationale=(
                    f"Operator-assistant turn (mode={self.mode}, seat "
                    f"{seat_label}, query sha {query_hash}) answered "
                    f"by model {self.client.model!r} at {self.client.base_url} "
                    f"(max_tokens={self.client.max_tokens}): {rounds} "
                    f"chat round(s), tools invoked: "
                    f"{', '.join(tool_names) or 'none'}; usage "
                    f"prompt_tokens={prompt_tokens}, "
                    f"completion_tokens={completion_tokens}; reply "
                    f"{len(reply)} chars."
                ),
                context=f"tool_trace={json.dumps(tool_names)}",
            )
        except Exception:  # noqa: BLE001 — telemetry must not eat the reply
            logger.exception("assistant: decision capture failed for turn")


__all__ = [
    "AssistantEngine",
    "AssistantTurn",
    "CAMPAIGN_SYSTEM_PROMPT",
    "CAMPAIGN_TICK_SYSTEM_PROMPT",
    "CAMPAIGN_READONLY_TOOL_SCHEMAS",
    "CAMPAIGN_TOOL_REGISTRY",
    "CAMPAIGN_TOOL_SCHEMAS",
    "DEBUG_SYSTEM_PROMPT",
    "MAX_TOOL_ROUNDS",
    "ResolvedSeat",
    "SYSTEM_PROMPT",
    "dispatch_campaign_tool",
    "reset_seat_cache",
    "resolve_active_seat",
]
