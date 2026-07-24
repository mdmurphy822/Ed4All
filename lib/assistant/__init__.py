"""Operator-assistant harness behind ``ed4all assistant``.

A sandboxed CLI chat backed by a LOCAL vLLM seat (model-agnostic in code;
the Nemotron nano seat is only the default deployment). Four modules, one
contract each:

* :mod:`lib.assistant.client` — the loopback-only OpenAI-compatible
  chat-completions client (``ED4ALL_ASSISTANT_*`` env knobs, conditional
  reasoning-thinking-off injection, seat liveness/autostart helpers).
* :mod:`lib.assistant.tools` — the TYPED tool whitelist (status / reports /
  library / seat-setup / actions / help families). The model can only reach
  the registered functions; every id/slug/enum value validates before a
  path or argv is built, every result is summarized (~4k-char cap) and
  secret-filtered. No shell, no arbitrary file access — any other requested
  tool name is refused back to the model.
* :mod:`lib.assistant.engine` — the UI-agnostic per-turn tool-call loop
  (:class:`~lib.assistant.engine.AssistantEngine`, capped at 6 rounds,
  ``operator`` / ``debug`` modes) + the mandatory
  :class:`lib.decision_capture.DecisionCapture` wiring (one
  ``llm_chat_call`` decision per assistant turn). The CLI verb and the GUI
  chat panel are both thin consumers of this engine.
* :mod:`lib.assistant.debug_context` — the bounded, redacted failed-run
  diagnostic snapshot injected by debug mode (``ed4all assistant --debug``).
"""

from .client import (  # noqa: F401
    ASSISTANT_SEAT_NAME,
    AssistantClient,
    AssistantClientError,
    AssistantProviderNotLocal,
    AssistantSeatUnavailable,
    resolve_assistant_autostart,
    resolve_assistant_base_url,
    resolve_assistant_max_tokens,
    resolve_assistant_model,
    resolve_assistant_seat,
    resolve_assistant_timeout,
    seat_is_serving,
)
from .engine import (  # noqa: F401
    CAMPAIGN_SYSTEM_PROMPT,
    CAMPAIGN_TOOL_REGISTRY,
    CAMPAIGN_TOOL_SCHEMAS,
    DEBUG_SYSTEM_PROMPT,
    MAX_TOOL_ROUNDS,
    SYSTEM_PROMPT,
    AssistantEngine,
    AssistantTurn,
    ResolvedSeat,
    dispatch_campaign_tool,
    resolve_active_seat,
)
from .tools import TOOL_REGISTRY, TOOL_SCHEMAS, dispatch_tool  # noqa: F401
