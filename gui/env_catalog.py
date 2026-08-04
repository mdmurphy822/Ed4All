"""Declarative env-knob / provider / base-model catalog (DATA ONLY).

The settings UI renders from this catalog so the frontend never hardcodes
knobs. Every entry is derived from the real codebase:

* ``PROVIDERS`` — the shared OpenAI-wire provider entries (``local`` /
  ``together`` / ``together-vision`` / ``groq`` / ``fireworks`` /
  ``deepseek``) are **DERIVED** from the canonical endpoint registry
  (``config/endpoints.yaml`` via ``lib/llm/endpoints.py``) — the SINGLE
  source of truth. This module no longer hand-maintains a second copy of
  their ``api_key_env`` / ``base_url`` / ``model_env`` / ``model_default``
  (W9.3-next: the historical drift source is eliminated by construction).
  Only GUI-facing metadata (``label`` and the ``local`` routing-dropdown
  ``vision_capable`` overlay) is added on top. The non-HTTP GUI-explicit
  seats — ``anthropic`` (SDK backend; its GUI ``model_env`` is the
  operator-facing ``LLM_MODEL`` knob, a documented divergence from the
  synthesis ``ANTHROPIC_SYNTHESIS_MODEL`` on the canonical row) and
  ``mock`` (a no-network convenience entry with no canonical endpoint) —
  stay explicit. **No invented providers** — a new OpenAI-wire provider is
  a one-row change in ``config/endpoints.yaml``, mirrored here for free.
* ``CATALOG`` covers every env var enumerated in the GUI build spec §4,
  grouped by ``category``.
* ``BASE_MODELS`` mirrors ``BaseModelRegistry.list_supported()`` when
  importable, else the documented static five.

This module imports cleanly WITHOUT FastAPI/uvicorn. The endpoint-registry
import (``lib.llm.endpoints``) pulls only stdlib + ``yaml`` + ``jsonschema``
+ ``lib.paths`` (no heavy ``MCP``/``Trainforge`` deps — those import FROM
the registry, not the reverse), and the base-model import is guarded. MCP
tools can therefore read the catalog without web deps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from lib.llm.endpoints import openai_compatible_legacy_registry

# ---------------------------------------------------------------------------
# Value grammars for knobs whose accepted set is not a finite enum.
#
# A catalog entry may declare EITHER ``enum`` (a closed list the UI renders as
# a <select>) OR ``pattern`` (a regex the UI renders as a validated text
# input). ``pattern`` exists because some knobs accept a parameterized token
# that no finite list can express — today only the torch device token, whose
# ``cuda:N`` form is accepted by BOTH
# ``lib/embedding/providers.py::normalize_device_token`` and the LibV2
# ``--device`` flag. Widening that entry to a free ``string`` would let the UI
# post ``auto`` (a value the resolver REJECTS by design — auto-detection is
# silent degradation), so the grammar is pinned here instead.
# ---------------------------------------------------------------------------

#: Torch device tokens accepted by ``normalize_device_token`` — ``cpu``,
#: ``cuda``, ``cuda:N``. Deliberately NOT ``auto``. Canonical lower-case form:
#: the resolver additionally accepts mixed case (it lower-cases first), but the
#: GUI offers only the canonical spelling, and the regex stays free of inline
#: flags so it is usable verbatim as an HTML ``pattern`` attribute.
DEVICE_TOKEN_PATTERN = r"^(?:cpu|cuda(?::\d+)?)$"

# ---------------------------------------------------------------------------
# Provider registry — DERIVED from the canonical endpoint registry.
#
# ``config/endpoints.yaml`` (loaded by ``lib/llm/endpoints.py``) is the ONE
# source of truth for the OpenAI-wire seats. Its
# ``openai_compatible_legacy_registry()`` projection carries the exact legacy
# per-entry shape (base_url_env / base_url_default / api_key_env /
# api_key_default / model_env / model_default / api_key_required + optional
# vision_capable / vision_capable_env / unverified), so the shared providers
# below are BUILT from that projection rather than re-typed here.
#
# GUI-facing overlay applied per derived provider:
#   * ``label``          human-facing string the settings UI renders.
#   * ``vision_capable`` (``local`` only) forced True so the routing dropdown
#     offers the local OpenAI-compatible seat (vLLM / Ollama / llama.cpp); the
#     real gate is LOCAL_VISION_CAPABLE + LOCAL_SYNTHESIS_MODEL at resolve
#     time. All other providers inherit the
#     canonical row's vision_capable (default False when the row omits it).
#
# GUI-explicit (NOT OpenAI-wire) seats stay hand-declared:
#   * ``anthropic`` — SDK backend (kind=anthropic on the canonical row, not
#     openai-wire); its GUI ``model_env`` is the operator-facing LLM_MODEL
#     knob (a documented divergence from ANTHROPIC_SYNTHESIS_MODEL on the
#     canonical row — see the drift-guard test).
#   * ``mock`` — no-network convenience entry with no canonical endpoint.
# ---------------------------------------------------------------------------

# Ordered shared OpenAI-wire providers the GUI exposes, with their GUI
# overlay. Order is preserved in the final PROVIDERS list. The canonical
# registry may carry additional openai_compatible rows (e.g. ``nvidia`` /
# ``nvidia-deepseek``) the GUI does not surface; they are intentionally NOT
# listed here so no default selection changes.
_DERIVED_PROVIDER_META: List[Dict[str, Any]] = [
    # ``local`` is a strict OpenAI-compatible server (TRT-LLM / vLLM /
    # llama.cpp / LM Studio) — no Ollama-native fallback. A deployment
    # overrides LOCAL_SYNTHESIS_BASE_URL to its own host:port. The
    # registry name stays ``"local"`` because Trainforge and the resolver all
    # key on that literal. The GUI marks it vision-capable for the routing
    # dropdown (overlay); the real enable is LOCAL_VISION_CAPABLE + a vision
    # LOCAL_SYNTHESIS_MODEL at resolve time.
    {
        "name": "local",
        "label": "Local model server (OpenAI-compatible: TRT-LLM, vLLM, llama.cpp)",
        "vision_capable": True,
    },
    {"name": "together", "label": "Together AI (text)"},
    {"name": "together-vision", "label": "Together AI (vision)"},
    {"name": "groq", "label": "Groq"},
    {"name": "fireworks", "label": "Fireworks"},
    {"name": "deepseek", "label": "DeepSeek"},
]

# GUI-explicit seats with no OpenAI-wire canonical projection.
_ANTHROPIC_PROVIDER: Dict[str, Any] = {
    "name": "anthropic",
    "label": "Anthropic (Claude)",
    "api_key_env": "ANTHROPIC_API_KEY",
    "base_url_env": None,
    "base_url_default": None,
    # GUI-facing model knob (LLM_MODEL) — deliberately NOT the canonical
    # ANTHROPIC_SYNTHESIS_MODEL synthesis env (documented divergence).
    "model_env": "LLM_MODEL",
    "model_default": "claude-opus-4-7",
    "api_key_required": True,
    "vision_capable": True,
    "unverified": False,
}
_MOCK_PROVIDER: Dict[str, Any] = {
    "name": "mock",
    "label": "Mock (no network)",
    "api_key_env": None,
    "base_url_env": None,
    "base_url_default": None,
    "model_env": None,
    "model_default": None,
    "api_key_required": False,
    "vision_capable": False,
    "unverified": False,
}


def _derive_shared_provider(
    meta: Dict[str, Any], legacy: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Build one GUI provider entry from the canonical legacy projection.

    ``meta`` supplies the GUI overlay (``name`` + ``label`` + optional
    ``vision_capable`` force). Every wire field (api_key_env / base_url /
    model_env / model_default / api_key_required) flows from the canonical
    row so the GUI can never drift from ``config/endpoints.yaml``.
    """
    name = meta["name"]
    row = legacy[name]
    entry: Dict[str, Any] = {
        "name": name,
        "label": meta["label"],
        "api_key_env": row.get("api_key_env"),
        "base_url_env": row.get("base_url_env"),
        "base_url_default": row.get("base_url_default"),
        "model_env": row.get("model_env"),
        "model_default": row.get("model_default"),
        "api_key_required": bool(row.get("api_key_required", False)),
        # GUI overlay wins when present; else the canonical flag (default
        # False when the row omits it — mirrors the legacy literal).
        "vision_capable": bool(
            meta["vision_capable"]
            if "vision_capable" in meta
            else row.get("vision_capable", False)
        ),
        "unverified": bool(row.get("unverified", False)),
    }
    # Only surface vision_capable_env when the canonical row declares it
    # (today: ``local`` → LOCAL_VISION_CAPABLE), matching the legacy shape.
    vision_env = row.get("vision_capable_env")
    if vision_env:
        entry["vision_capable_env"] = vision_env
    return entry


def _build_providers() -> List[Dict[str, Any]]:
    """Assemble PROVIDERS: anthropic (explicit) + derived shared + mock.

    Order matches the historical hand-maintained list exactly. Shared
    OpenAI-wire seats are DERIVED from the canonical registry; the two
    non-wire seats stay explicit.
    """
    legacy = openai_compatible_legacy_registry()
    providers: List[Dict[str, Any]] = [dict(_ANTHROPIC_PROVIDER)]
    for meta in _DERIVED_PROVIDER_META:
        providers.append(_derive_shared_provider(meta, legacy))
    providers.append(dict(_MOCK_PROVIDER))
    return providers


PROVIDERS: List[Dict[str, Any]] = _build_providers()


def provider_names() -> List[str]:
    """Return the ordered list of provider names."""
    return [p["name"] for p in PROVIDERS]


def provider(name: str) -> Optional[Dict[str, Any]]:
    """Return the provider registry entry for ``name`` (or ``None``)."""
    for entry in PROVIDERS:
        if entry["name"] == name:
            return entry
    return None


def provider_label(name: str) -> str:
    """Return the human-facing label for a provider name.

    Falls back to the raw name when the entry is unknown or carries no
    ``label`` (so the UI never renders an empty string).
    """
    entry = provider(name)
    if entry and entry.get("label"):
        return str(entry["label"])
    return name


def vision_provider_names() -> List[str]:
    """Return the ordered list of vision-capable provider names.

    Filters ``PROVIDERS`` on the ``vision_capable`` marker. These are the
    only providers a vision/VLM routing dropdown should offer:
    ``anthropic``, ``local`` (a local OpenAI-compatible vision seat — e.g.
    Ollama or vLLM — with LOCAL_VISION_CAPABLE=true), and ``together-vision``.
    These feed the SemantiK VLM seat selector (``SEMANTIK_VLM_PROVIDER``).
    """
    return [p["name"] for p in PROVIDERS if p.get("vision_capable")]


# ---------------------------------------------------------------------------
# Base models (training).
# ---------------------------------------------------------------------------
_STATIC_BASE_MODELS: List[str] = [
    "qwen2.5-1.5b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "smollm2-1.7b",
    "phi-3.5-mini",
]


def _load_base_models() -> List[str]:
    """Mirror ``BaseModelRegistry.list_supported()`` when importable.

    The import is guarded so this data module stays usable on a CPU-only /
    web-only install where ``Trainforge.training`` may not be importable.
    Falls back to the documented static five.
    """
    try:
        from Trainforge.training.base_models import (  # noqa: PLC0415
            BaseModelRegistry,
        )

        supported = list(BaseModelRegistry.list_supported())
        if supported:
            return supported
    except Exception:  # noqa: BLE001 — defensive; any import/runtime failure
        pass
    return list(_STATIC_BASE_MODELS)


BASE_MODELS: List[str] = _load_base_models()


# ---------------------------------------------------------------------------
# Env catalog — every knob the settings UI renders, grouped by category.
#
# Each entry: key, label, category, type (secret|enum|string|bool|number),
# default, enum (where applicable), help, applies_to.
# ---------------------------------------------------------------------------
def _build_catalog() -> List[Dict[str, Any]]:
    names = provider_names()
    vision_names = vision_provider_names()
    return [
        # ---------------------------------------------------------- credentials
        {
            "key": "ANTHROPIC_API_KEY",
            "label": "Anthropic API Key",
            "category": "credentials",
            "type": "secret",
            "default": None,
            "help": "Required for --mode api with the Anthropic backend.",
            "applies_to": "anthropic",
        },
        {
            "key": "TOGETHER_API_KEY",
            "label": "Together AI API Key",
            "category": "credentials",
            "type": "secret",
            "default": None,
            "help": "Required for the together / together-vision providers.",
            "applies_to": "together",
        },
        {
            "key": "GROQ_API_KEY",
            "label": "Groq API Key",
            "category": "credentials",
            "type": "secret",
            "default": None,
            "help": "Required for the groq provider (unverified registry entry).",
            "applies_to": "groq",
        },
        {
            "key": "FIREWORKS_API_KEY",
            "label": "Fireworks API Key",
            "category": "credentials",
            "type": "secret",
            "default": None,
            "help": "Required for the fireworks provider (unverified registry entry).",
            "applies_to": "fireworks",
        },
        {
            "key": "DEEPSEEK_API_KEY",
            "label": "DeepSeek API Key",
            "category": "credentials",
            "type": "secret",
            "default": None,
            "help": "Required for the deepseek provider (unverified registry entry).",
            "applies_to": "deepseek",
        },
        {
            "key": "LOCAL_SYNTHESIS_API_KEY",
            "label": "Local Server API Key",
            "category": "credentials",
            "type": "secret",
            "default": None,
            "help": (
                "Optional auth key for the local OpenAI-compatible server. "
                "Most local servers ignore it; set only when proxied."
            ),
            "applies_to": "local",
        },
        # -------------------------------------------------------- global routing
        {
            "key": "LLM_MODE",
            "label": "LLM Mode",
            "category": "global",
            "type": "enum",
            "default": "local",
            "enum": ["local", "api"],
            "help": (
                "local: use the current Claude Code session as the LLM. "
                "api: use the Anthropic SDK directly (needs ANTHROPIC_API_KEY)."
            ),
            "applies_to": "global",
        },
        {
            "key": "LLM_PROVIDER",
            "label": "LLM Provider",
            "category": "global",
            "type": "enum",
            "default": "anthropic",
            "enum": names,
            "help": "Provider used in api mode.",
            "applies_to": "global",
        },
        {
            "key": "LLM_MODEL",
            "label": "LLM Model",
            "category": "global",
            "type": "string",
            "default": None,
            "help": "Model ID override; per-provider default when unset.",
            "applies_to": "global",
        },
        {
            "key": "MCP_ORCHESTRATOR_LLM_MODEL",
            "label": "Orchestrator Model",
            "category": "global",
            "type": "string",
            "default": "claude-opus-4-7",
            "help": "Pins the Anthropic model used by the MCP orchestrator backend.",
            "applies_to": "global",
        },
        # ---------------------------------------------------------- local backend
        {
            "key": "LOCAL_SYNTHESIS_BASE_URL",
            "label": "Local Base URL",
            "category": "local",
            "type": "string",
            "default": "http://localhost:8000/v1",
            "help": (
                "Base URL of the local OpenAI-compatible server "
                "(TRT-LLM / vLLM / llama.cpp / LM Studio)."
            ),
            "applies_to": "local",
        },
        {
            "key": "LOCAL_SYNTHESIS_MODEL",
            "label": "Local Model",
            "category": "local",
            "type": "string",
            "default": "nemotron-3-nano-30b-a3b",
            "help": "Model identifier the local server expects.",
            "applies_to": "local",
        },
        # LOCAL_VISION_CAPABLE now lives in the ``vision`` category (single
        # canonical catalog entry) — see the vision block below.
        # --------------------------------------------------------------- together
        {
            "key": "TOGETHER_SYNTHESIS_MODEL",
            "label": "Together Text Model",
            "category": "together",
            "type": "string",
            "default": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "help": "Together AI text model (synthesis).",
            "applies_to": "together",
        },
        # TOGETHER_VISION_MODEL now lives in the ``vision`` category (single
        # canonical catalog entry) — see the vision block above.
        # --------------------------------------------------------------- vision
        # Vision / VLM routing. ``SEMANTIK_VLM_PROVIDER`` is the REAL knob the
        # SemantiK cascade consumes to select its VLM seat for per-page
        # extraction + figure alt-text (``semantik_structure/extract_shared.py
        # ::resolve_vlm_provider``). The enum is filtered to the vision-capable
        # provider universe (anthropic / together-vision / local — a local
        # OpenAI-compatible vision seat such as Ollama or vLLM) so the UI can't
        # offer a text-only backend that would fail the VLM seat's "fail at
        # startup, not mid-page" readiness gate.
        {
            "key": "SEMANTIK_VLM_PROVIDER",
            "label": "Vision Provider (SemantiK VLM / alt-text)",
            "category": "vision",
            "type": "enum",
            "default": None,
            "enum": vision_names,
            "help": (
                "Vision-capable LLM backend for the SemantiK VLM seat "
                "(per-page extraction + figure alt-text). Unset falls back to "
                "the local OpenAI-compatible VLM seat. Vision-capable options "
                "only: anthropic, together-vision, or local (a local vision "
                "model + LOCAL_VISION_CAPABLE=true — served by vLLM, Ollama, "
                "etc.)."
            ),
            "applies_to": "vision",
        },
        {
            "key": "SEMANTIK_VLM_MODEL",
            "label": "Vision Model (SemantiK VLM)",
            "category": "vision",
            "type": "string",
            "default": "qwen2.5vl:7b",
            "help": (
                "Model id the SemantiK VLM seat requests "
                "(semantik_structure/extract_shared.py::resolve_vlm_model). "
                "Default qwen2.5vl:7b (Qwen2.5-VL-7B-Instruct, Apache-2.0) on "
                "the local vision seat (e.g. Ollama or a vLLM seat)."
            ),
            "applies_to": "vision",
        },
        {
            "key": "LOCAL_VISION_CAPABLE",
            "label": "Local Vision Capable",
            "category": "vision",
            "type": "bool",
            "default": False,
            "help": (
                "Flip on when the local model server (vLLM, Ollama, etc.) has "
                "a vision model loaded (e.g. qwen2.5-vl, llama3.2-vision, "
                "llava:13b). This is how a LOCAL VLM is selected: set the local "
                "model (LOCAL_SYNTHESIS_MODEL) to the vision model AND flip this "
                "on. There is no separate LOCAL_VISION_MODEL env var in the code."
            ),
            "applies_to": "vision",
        },
        {
            "key": "TOGETHER_VISION_MODEL",
            "label": "Together Vision Model (VLM)",
            "category": "vision",
            "type": "string",
            "default": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "help": (
                "Vision model for the together-vision provider. Reuses "
                "TOGETHER_API_KEY. Set via vision routing when the vision "
                "provider is together-vision."
            ),
            "applies_to": "together-vision",
        },
        # ------------------------------------------------------------- courseforge
        {
            "key": "COURSEFORGE_TWO_PASS",
            "label": "Courseforge Two-Pass",
            "category": "courseforge",
            "type": "bool",
            "default": False,
            "help": "Enable the Courseforge outline -> validate -> rewrite two-pass pipeline.",
            "applies_to": "courseforge",
        },
        {
            "key": "COURSEFORGE_PROVIDER",
            "label": "Courseforge Provider",
            "category": "courseforge",
            "type": "enum",
            "default": None,
            "enum": names,
            "help": "Default LLM provider for Courseforge content generation.",
            "applies_to": "courseforge",
        },
        {
            "key": "COURSEFORGE_OUTLINE_PROVIDER",
            "label": "Outline Provider",
            "category": "courseforge",
            "type": "enum",
            "default": None,
            "enum": names,
            "help": "LLM provider for the Courseforge outline tier.",
            "applies_to": "courseforge_outline",
        },
        {
            "key": "COURSEFORGE_OUTLINE_MODEL",
            "label": "Outline Model",
            "category": "courseforge",
            "type": "string",
            "default": None,
            "help": "Model for the Courseforge outline tier.",
            "applies_to": "courseforge_outline",
        },
        {
            "key": "COURSEFORGE_REWRITE_PROVIDER",
            "label": "Rewrite Provider",
            "category": "courseforge",
            "type": "enum",
            "default": None,
            "enum": names,
            "help": "LLM provider for the Courseforge rewrite tier.",
            "applies_to": "courseforge_rewrite",
        },
        {
            "key": "COURSEFORGE_REWRITE_MODEL",
            "label": "Rewrite Model",
            "category": "courseforge",
            "type": "string",
            "default": None,
            "help": "Model for the Courseforge rewrite tier.",
            "applies_to": "courseforge_rewrite",
        },
        {
            "key": "COURSEPLANNER_PROVIDER",
            "label": "Course Planner Provider",
            "category": "courseforge",
            "type": "enum",
            "default": None,
            "enum": names,
            "help": "LLM provider for objective synthesis (course planning).",
            "applies_to": "courseplanner",
        },
        {
            "key": "COURSEPLANNER_MODEL",
            "label": "Course Planner Model",
            "category": "courseforge",
            "type": "string",
            "default": None,
            "help": "Model for objective synthesis (course planning).",
            "applies_to": "courseplanner",
        },
        {
            "key": "COURSEFORGE_BLOCK_ROUTING_PATH",
            "label": "Block Routing Path",
            "category": "courseforge",
            "type": "string",
            "default": None,
            "help": "Path to a per-block-type routing config for the rewrite tier.",
            "applies_to": "courseforge",
        },
        {
            "key": "TEXTBOOK_SYNTHESIS_PROVIDER",
            "label": "Textbook Synthesis Provider",
            "category": "courseforge",
            "type": "enum",
            "default": None,
            "enum": names,
            "help": "LLM provider for the three-stage textbook synthesis (domain concepts).",
            "applies_to": "textbook_synthesis",
        },
        {
            "key": "TEXTBOOK_SYNTHESIS_MODEL",
            "label": "Textbook Synthesis Model",
            "category": "courseforge",
            "type": "string",
            "default": None,
            "help": "Model for the three-stage textbook synthesis.",
            "applies_to": "textbook_synthesis",
        },
        # ------------------------------------------------------------- trainforge
        {
            "key": "TRAINFORGE_SYNTHESIS_PROVIDER",
            "label": "Synthesis Provider",
            "category": "trainforge",
            "type": "enum",
            "default": None,
            "enum": names,
            "help": "LLM provider for training-pair synthesis (license-clean providers only).",
            "applies_to": "trainforge_synthesis",
        },
        {
            "key": "TRAINFORGE_ASSESSMENT_PROVIDER",
            "label": "Assessment Provider",
            "category": "trainforge",
            "type": "enum",
            "default": None,
            "enum": names,
            "help": "LLM provider for Trainforge assessment generation.",
            "applies_to": "trainforge_assessment",
        },
        {
            "key": "TRAINFORGE_ASSESSMENT_MODEL",
            "label": "Assessment Model",
            "category": "trainforge",
            "type": "string",
            "default": None,
            "help": "Model for Trainforge assessment generation.",
            "applies_to": "trainforge_assessment",
        },
        {
            "key": "ANTHROPIC_SYNTHESIS_MODEL",
            "label": "Anthropic Synthesis Model",
            "category": "trainforge",
            "type": "string",
            "default": "claude-sonnet-4-6",
            "help": "Anthropic model for the chunk -> instruction-pair paraphrase pass.",
            "applies_to": "trainforge_synthesis",
        },
        {
            "key": "TRAINFORGE_REQUIRE_EMBEDDINGS",
            "label": "Require Embeddings",
            "category": "trainforge",
            "type": "bool",
            "default": False,
            "help": (
                "Fail closed when the optional [embedding] extras are absent "
                "(statistical-tier validators)."
            ),
            "applies_to": "trainforge",
        },
        {
            "key": "WAVE18_COS_PER_WEEK",
            "label": "Chapter Objectives / Week",
            "category": "trainforge",
            "type": "number",
            "default": 2,
            "help": "Objective-driven pacing target used to re-scale duration_weeks.",
            "applies_to": "courseplanner",
        },
        # --------------------------------------------------------- embedding
        # Retrieval-index embedding backend. SEPARATE registry from the LLM
        # ``PROVIDERS`` list (embeddings index corpus chunks; they are not
        # LLM chat/synthesis providers), so the enum is a literal triple, NOT
        # ``provider_names()``. Mirrors
        # ``lib/embedding/providers.py::_EMBEDDING_PROVIDERS``.
        {
            "key": "ED4ALL_EMBEDDING_PROVIDER",
            "label": "Embedding Provider",
            "category": "embedding",
            "type": "enum",
            "default": "st",
            "enum": ["st", "local-openai", "fake"],
            "help": (
                "Retrieval-index embedding backend: st (in-process "
                "sentence-transformers), local-openai (local /v1/embeddings "
                "server), or fake (deterministic test vectors). Separate from "
                "the LLM provider registry."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_MODEL",
            "label": "Embedding Model",
            "category": "embedding",
            "type": "string",
            "default": None,
            "help": (
                "Embedding model ID override (per-provider default when "
                "unset; e.g. BAAI/bge-large-en-v1.5 for st, nomic-embed-text for "
                "local-openai)."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_BASE_URL",
            "label": "Embedding Server Base URL",
            "category": "embedding",
            "type": "string",
            "default": "http://localhost:11434/v1",
            "help": (
                "Base URL of the local OpenAI-compatible /v1/embeddings "
                "server (local-openai provider only)."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_API_KEY",
            "label": "Embedding Server API Key",
            "category": "embedding",
            "type": "secret",
            "default": None,
            "help": (
                "Optional bearer token for the local embedding server "
                "(local-openai only; most local servers ignore it)."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_DEVICE",
            "label": "Embedding Device",
            "category": "embedding",
            # A ``pattern``, not an ``enum``: the resolver
            # (lib/embedding/providers.py::normalize_device_token) and the
            # LibV2 ``--device`` flag both accept a ``cuda:N`` token that a
            # two-value enum cannot express, so the enum silently made a valid
            # operator value unrepresentable in the GUI. Still CLOSED — the
            # pattern rejects everything outside cpu / cuda / cuda:N.
            "type": "string",
            "default": "cuda",
            "pattern": DEVICE_TOKEN_PATTERN,
            "help": (
                "Torch device for the in-process st embedder — index builds, "
                "query encoding, and the validator-tier embedder. Accepted "
                "tokens: cpu, cuda, cuda:N (e.g. cuda:1 to pin a second "
                "card); 'auto' is deliberately NOT accepted. Default cuda; "
                "cpu is a fully-supported EXPLICIT selection. There is no "
                "automatic CUDA->CPU fallback: on a GPU-less host you must "
                "select cpu here or the embedding paths raise. Recorded in "
                "the index manifest."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_DTYPE",
            "label": "Embedding Precision",
            "category": "embedding",
            "type": "enum",
            "default": "fp32",
            "enum": ["fp32", "bf16", "fp16"],
            "help": (
                "Encoder precision for the in-process st embedder "
                "(lib/embedding/providers.py::normalize_dtype_token). Default "
                "fp32 — byte-identical to the pre-seam load. Prefer bf16 over "
                "fp16 for the 1024-dim mean-pooled encoder (same bandwidth "
                "win, no overflow risk on the pooling sum). Recorded on the "
                "client fingerprint, so an index built at one precision is "
                "distinguishable from another."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_BATCH_SIZE",
            "label": "Embedding Batch Size",
            "category": "embedding",
            "type": "number",
            "default": 16,
            "help": "Encode batch size for the embedding client.",
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_CLIENT_CACHE",
            "label": "Embedding Client Cache",
            "category": "embedding",
            "type": "bool",
            "default": True,
            "help": (
                "Process-level resident embedding-client cache (default ON). "
                "Keeps the loaded encoder resident across calls instead of "
                "re-allocating ~1.25 GiB per query. An explicit falsey token "
                "(0/false/no/off) restores the historical "
                "construct-a-client-per-call behavior exactly."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_CLIENT_CACHE_MAX",
            "label": "Embedding Client Cache Size",
            "category": "embedding",
            "type": "number",
            "default": 2,
            "help": (
                "Maximum resident embedding clients held by the LRU cache "
                "(entries, not bytes — resident cost is dominated by encoder "
                "weights). Values below 1 and non-integers fall back to the "
                "default 2."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_NLI_DEVICE",
            "label": "NLI Device",
            "category": "embedding",
            "type": "enum",
            "default": "cpu",
            "enum": ["cpu", "cuda"],
            "help": (
                "Torch device for the in-process NLI classifier that scores "
                "groundedness/eval entailment. Default cpu for determinism; "
                "cuda for speed (~20-50x, fp16 on GPU). cuda requested on a "
                "CPU box falls back to cpu (no crash). Recorded for "
                "provenance."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_EMBEDDING_ALLOW_FAKE",
            "label": "Allow Fake Embedding Index",
            "category": "embedding",
            "type": "bool",
            "default": False,
            "help": (
                "Permit loading a vector index built with the fake provider "
                "in a production read path. Default off → fake-provider "
                "indexes are refused at query time."
            ),
            "applies_to": "retrieval",
        },
        # Query-side search tuning (LibV2/tools/libv2/retrieval/vector_index.py). Both
        # are latency/parity knobs: results are byte-identical whichever way
        # they are set, which is why a bad value degrades to the default
        # rather than failing retrieval.
        {
            "key": "ED4ALL_RETRIEVAL_BLAS_THREADS",
            "label": "Retrieval BLAS Threads",
            "category": "embedding",
            "type": "number",
            "default": 8,
            "help": (
                "BLAS thread-team cap around the search GEMV only (never the "
                "build-time encode). Default 8 — fanning a small query GEMV "
                "across every core costs more in thread wakeup than it "
                "recovers. 0 (or any non-positive value) disables the limiter "
                "and lets BLAS use its own team. Results are unaffected; only "
                "latency is. Requires threadpoolctl to take effect."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_RETRIEVAL_TOPK_LEGACY",
            "label": "Legacy Top-K Sort",
            "category": "embedding",
            "type": "bool",
            "default": False,
            "help": (
                "Restore the pre-argpartition full-Python top-k sort. Parity "
                "escape hatch only — the two paths are required to be "
                "output-identical and the test suite pins that; the legacy "
                "path is ~50x slower at library scale."
            ),
            "applies_to": "retrieval",
        },
        # --------------------------------------------------------- chunking
        # Chunkset emit knobs read by the ``chunking`` / ``imscc_chunking``
        # phases (MCP/tools/pipeline_tools.py) and by the chunker itself
        # (Trainforge/chunker/cross_course_dedup.py).
        {
            "key": "ED4ALL_CHUNK_DEDUP",
            "label": "Within-Package Chunk Dedup",
            "category": "chunking",
            "type": "bool",
            "default": False,
            "help": (
                "Drop exact-normalised repeated units within one chunkset. "
                "Default OFF — the pass DELETES content, so enabling it is a "
                "deliberate decision; off means a byte-identical chunkset "
                "with no hashing and an empty dedup ledger. Only an explicit "
                "truthy token (1/true/yes/on) enables it."
            ),
            "applies_to": "chunking",
        },
        {
            "key": "ED4ALL_CHUNK_DEDUP_MIN_TOKENS",
            "label": "Chunk Dedup Token Floor",
            "category": "chunking",
            "type": "number",
            "default": 8,
            "help": (
                "A repeat carrying fewer exact-normalised tokens than this is "
                "NEVER dropped (guards a short-but-legitimate recurring "
                "definition). 0 is honoured as an explicit 'no floor' choice; "
                "negative and non-integer values fall back to the default 8. "
                "Only consulted when the dedup pass is on."
            ),
            "applies_to": "chunking",
        },
        {
            "key": "ED4ALL_HTML_PARSE_WORKERS",
            "label": "Staged-HTML Parse Workers",
            "category": "chunking",
            "type": "number",
            "default": 10,
            "help": (
                "Process-pool size for the staged-HTML parse. 0 and 1 are "
                "HONORED as the byte-identical serial path (no pool "
                "constructed); pooled and serial emit are identical by "
                "construction. Otherwise clamped to min(value, file count, "
                "CPU count). Negative / non-integer values fall back to the "
                "default 10. Note: above 1 a graceful stop drains the "
                "in-flight files rather than a single unit."
            ),
            "applies_to": "chunking",
        },
        {
            "key": "ED4ALL_HTML_PARSE_START_METHOD",
            "label": "Parse Pool Start Method",
            "category": "chunking",
            "type": "enum",
            "default": "spawn",
            # ``fork`` is NOT offered: the chunking phases run in a
            # multi-threaded process that already holds a CUDA primary
            # context, where fork is undefined behaviour. The resolver warns
            # loudly and falls back to spawn — the GUI must not be able to
            # request it in the first place.
            "enum": ["spawn", "forkserver", "serial"],
            "help": (
                "multiprocessing start method for the staged-HTML parse pool. "
                "fork is deliberately NOT an accepted value: these phases run "
                "in a multi-threaded process holding a CUDA primary context, "
                "where fork deadlocks. Every accepted value emits identical "
                "chunks, so an unrecognized token degrades to spawn."
            ),
            "applies_to": "chunking",
        },
        {
            "key": "ED4ALL_HTML_ASSET_REJECT",
            "label": "Reject Binary Assets",
            "category": "chunking",
            "type": "bool",
            "default": True,
            "help": (
                "Byte-signature reject of non-HTML files (PNG/PDF/zip/font/…) "
                "that match the discovery glob. Default ON; an explicit "
                "falsey token (0/false/no/off) opts out. Either way the file "
                "is recorded in the per-file outcome ledger under a named "
                "drop reason — the flag only changes WHICH reason "
                "(asset_rejected vs a decode failure), never whether the drop "
                "is counted."
            ),
            "applies_to": "chunking",
        },
        # --------------------------------------------------------- answer
        # Grounded-answer backend (runtime Q&A inference over retrieved
        # passages). Resolves the W-D12 ``_OPENAI_COMPATIBLE_PROVIDERS``
        # registry but is LOOPBACK-ENFORCED: a non-loopback resolved
        # base_url raises AnswerProviderNotLocal. Phase IA has no cloud arm
        # on the answer path. See lib/retrieval/answer_backend.py.
        {
            "key": "ED4ALL_ANSWER_PROVIDER",
            "label": "Answer Provider",
            "category": "answer",
            "type": "string",
            "default": "local",
            "help": (
                "Grounded-answer backend (runtime Q&A inference). "
                "Loopback-only: a non-loopback resolved base_url raises "
                "AnswerProviderNotLocal (Phase IA: no cloud answer arm). "
                "Only 'local' is permitted today."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_ANSWER_MODEL",
            "label": "Answer Model",
            "category": "answer",
            "type": "string",
            "default": None,
            "help": (
                "Answer model ID override. Resolution chain: explicit arg "
                "> ED4ALL_ANSWER_MODEL > LOCAL_SYNTHESIS_MODEL > registry "
                "default (nemotron-3-nano-30b-a3b)."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_ANSWER_TIMEOUT_SECONDS",
            "label": "Answer Timeout (seconds)",
            "category": "answer",
            "type": "number",
            "default": 120,
            "help": (
                "Answer-client HTTP timeout (long passages, slow local GPU). "
                "Garbage values fall back to the default."
            ),
            "applies_to": "retrieval",
        },
        {
            "key": "ED4ALL_ANSWER_LIBRARY_WIDE",
            "label": "Library-Wide Ask",
            "category": "answer",
            "type": "boolean",
            "default": False,
            "help": (
                "Union retrieval across ALL indexed courses on the grounded-"
                "answer path (default off => single-course). Every citation keeps "
                "its source course's provenance. A per-request 'search all "
                "courses' toggle (ask + ask-jobs endpoints) WINS over this env "
                "default."
            ),
            "applies_to": "retrieval",
        },
    ]


CATALOG: List[Dict[str, Any]] = _build_catalog()


def by_category() -> Dict[str, List[Dict[str, Any]]]:
    """Group ``CATALOG`` entries by their ``category`` (insertion order)."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in CATALOG:
        grouped.setdefault(entry["category"], []).append(entry)
    return grouped


def catalog_keys() -> List[str]:
    """Return the set of env keys the catalog declares (validation helper)."""
    return [entry["key"] for entry in CATALOG]


def catalog_entry(key: str) -> Optional[Dict[str, Any]]:
    """Return the catalog entry for env ``key`` (or ``None`` when unknown)."""
    for entry in CATALOG:
        if entry["key"] == key:
            return entry
    return None


def value_is_valid(key: str, value: Any) -> bool:
    """True when ``value`` is an acceptable value for catalog ``key``.

    Enforces the two CLOSED value grammars a catalog entry may declare:

    * ``enum``    — membership in the declared list (this is what keeps
      ``ED4ALL_HTML_PARSE_START_METHOD=fork`` unrepresentable).
    * ``pattern`` — a full regex match (this is what makes
      ``ED4ALL_EMBEDDING_DEVICE=cuda:1`` representable while still rejecting
      ``auto``).

    An unset value (``None`` / blank) is always acceptable — "unset" means
    "let the resolver's own default win", never a fabricated selection. An
    unknown key and an entry declaring neither grammar return ``True``: this
    helper narrows, it never invents a constraint the code does not have.
    """
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    entry = catalog_entry(key)
    if entry is None:
        return True
    choices = entry.get("enum")
    if isinstance(choices, list) and choices:
        return text in [str(c) for c in choices]
    pattern = entry.get("pattern")
    if isinstance(pattern, str) and pattern:
        return re.match(pattern, text) is not None
    return True


# ---------------------------------------------------------------------------
# Model-routing -> env mapping.
#
# Maps each ``model_routing.<task>.<field>`` path to the canonical env var.
# Consumed by settings_store.render_env / routing_to_env. Tasks/fields not in
# this table are ignored on render (so an unknown key never invents a var).
# ---------------------------------------------------------------------------
ROUTING_ENV_MAP: Dict[str, Dict[str, str]] = {
    "global": {
        "mode": "LLM_MODE",
        "provider": "LLM_PROVIDER",
        "model": "LLM_MODEL",
    },
    # Canonical vision / VLM routing surface. ``vision.provider`` maps to
    # SEMANTIK_VLM_PROVIDER and ``vision.model`` to SEMANTIK_VLM_MODEL — the
    # real knobs the SemantiK VLM seat reads (extract_shared.py::
    # resolve_vlm_provider / resolve_vlm_model). The seat is provider-agnostic
    # and OpenAI-compatible, so the model is always SEMANTIK_VLM_MODEL
    # regardless of the selected provider.
    "vision": {
        "provider": "SEMANTIK_VLM_PROVIDER",
        "model": "SEMANTIK_VLM_MODEL",
    },
    "courseforge_outline": {
        "provider": "COURSEFORGE_OUTLINE_PROVIDER",
        "model": "COURSEFORGE_OUTLINE_MODEL",
    },
    "courseforge_rewrite": {
        "provider": "COURSEFORGE_REWRITE_PROVIDER",
        "model": "COURSEFORGE_REWRITE_MODEL",
    },
    "courseplanner": {
        "provider": "COURSEPLANNER_PROVIDER",
        "model": "COURSEPLANNER_MODEL",
    },
    "textbook_synthesis": {
        "provider": "TEXTBOOK_SYNTHESIS_PROVIDER",
        "model": "TEXTBOOK_SYNTHESIS_MODEL",
    },
    "trainforge_synthesis": {
        "provider": "TRAINFORGE_SYNTHESIS_PROVIDER",
        "model": "ANTHROPIC_SYNTHESIS_MODEL",
    },
    "trainforge_assessment": {
        "provider": "TRAINFORGE_ASSESSMENT_PROVIDER",
        "model": "TRAINFORGE_ASSESSMENT_MODEL",
    },
    "answer": {
        "provider": "ED4ALL_ANSWER_PROVIDER",
        "model": "ED4ALL_ANSWER_MODEL",
    },
}


def routing_to_env(model_routing: Dict[str, Any]) -> Dict[str, str]:
    """Flatten a ``model_routing`` block to an env var map.

    Walks ``ROUTING_ENV_MAP``; drops null / empty values. Unknown tasks or
    fields are skipped (never inventing an env var). Values are stringified.
    """
    out: Dict[str, str] = {}
    if not isinstance(model_routing, dict):
        return out
    for task, fields in ROUTING_ENV_MAP.items():
        block = model_routing.get(task)
        if not isinstance(block, dict):
            continue
        for field, env_key in fields.items():
            value = block.get(field)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            out[env_key] = text
    return out


# ---------------------------------------------------------------------------
# Default settings doc (the §3 schema with defaults baked in).
# ---------------------------------------------------------------------------
def default_settings() -> Dict[str, Any]:
    """Return a fresh default settings doc (version 1).

    A new dict on every call so callers can mutate freely.
    """
    return {
        "version": 1,
        "updated_at": None,
        "env": {},
        "model_routing": {
            # Default to "local" so a fresh GUI works key-free on first run
            # and agrees with the LLM_MODE catalog default (also "local").
            # Using "api" here required ANTHROPIC_API_KEY and broke first run.
            "global": {"mode": "local", "provider": "anthropic", "model": None},
            "vision": {"provider": None, "model": None},
            "courseforge_outline": {"provider": None, "model": None},
            "courseforge_rewrite": {"provider": None, "model": None},
            "courseplanner": {"provider": None, "model": None},
            "textbook_synthesis": {"provider": None, "model": None},
            "trainforge_synthesis": {"provider": None, "model": None},
            "trainforge_assessment": {"provider": None, "model": None},
        },
        "retrieval": {
            "top_k": 5,
            "min_grounding_cosine": 0.45,
            "require_embeddings": False,
        },
        "flags": {
            "COURSEFORGE_TWO_PASS": False,
            "TRAINFORGE_REQUIRE_EMBEDDINGS": False,
        },
        # The Studio Assistant chat panel. ``autostart`` governs whether a chat
        # request may ATTEMPT the engine's lazy seat-start path when the local
        # assistant seat is down (the engine's own ED4ALL_ASSISTANT_AUTOSTART
        # policy still decides whether a start actually occurs). Default off:
        # a down seat surfaces as a 503 with start instructions.
        "assistant": {
            "autostart": False,
        },
    }


__all__ = [
    "PROVIDERS",
    "provider_names",
    "provider",
    "provider_label",
    "vision_provider_names",
    "BASE_MODELS",
    "CATALOG",
    "DEVICE_TOKEN_PATTERN",
    "by_category",
    "catalog_keys",
    "catalog_entry",
    "value_is_valid",
    "ROUTING_ENV_MAP",
    "routing_to_env",
    "default_settings",
]
