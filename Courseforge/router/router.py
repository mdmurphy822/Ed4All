#!/usr/bin/env python3
"""Courseforge two-pass router (Phase 3 §3).

The router is the per-block dispatch surface for the Phase 3 two-pass
content pipeline. It chooses an LLM provider + model per ``(block_type,
tier)`` then dispatches to either the
:class:`Courseforge.generators._outline_provider.OutlineProvider`
(structural-skeleton draft, fast/cheap) or the
:class:`Courseforge.generators._rewrite_provider.RewriteProvider`
(pedagogical-depth HTML body, larger/Anthropic-default).

Resolution order for a per-block dispatch (Phase 3 §3.3):

1. Per-call ``**overrides`` (operator / test override).
2. Loaded ``block_routing.yaml`` policy entry (Subtask 34 — not yet
   wired in this Wave; the router falls through when the policy is
   absent, which is the Wave-N default).
3. Tier-default env vars (``COURSEFORGE_OUTLINE_PROVIDER`` /
   ``COURSEFORGE_OUTLINE_MODEL`` / ``COURSEFORGE_REWRITE_PROVIDER`` /
   ``COURSEFORGE_REWRITE_MODEL``).
4. Module-level :data:`_HARDCODED_DEFAULTS` table (one entry per
   ``(block_type, tier)`` pair, populated for every value in
   :data:`Courseforge.scripts.blocks.BLOCK_TYPES`).

Decision-event contract: every successful per-block dispatch emits one
``block_outline_call`` (outline tier) or ``block_rewrite_call`` (rewrite
tier) decision-capture event. The provider classes already emit one
event per LLM call; the router emits one additional ``policy_source``
audit event so a postmortem can reconstruct WHICH layer of the
resolution chain governed the dispatch.

Touch-attribution contract: when the router constructs a
:class:`Touch` directly (e.g. for the
``escalate_immediately`` short-circuit that skips the outline tier),
the ``provider`` field is mapped onto the canonical
:data:`Courseforge.scripts.blocks._TOUCH_PROVIDERS` set —
``"openai_compatible"`` collapses to ``"local"`` because both go
through the same OpenAICompatibleClient. Anthropic / together /
local map 1:1.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# ``blocks.py`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# import bridge used by the sibling provider modules so ``from blocks
# import Block`` resolves regardless of how this module is loaded
# (CLI, MCP tool, pytest).
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import BLOCK_TYPES, Block, Touch  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALLOWED_TIERS: Tuple[str, ...] = ("outline", "rewrite")
_ALLOWED_PROVIDERS: Tuple[str, ...] = (
    "anthropic",
    "together",
    "local",
    "openai_compatible",
)

# Per-tier env-var names mirroring the OutlineProvider / RewriteProvider
# constructor surfaces. Read at resolution time so tests can monkeypatch
# them without re-importing the router.
_ENV_OUTLINE_PROVIDER = "COURSEFORGE_OUTLINE_PROVIDER"
_ENV_OUTLINE_MODEL = "COURSEFORGE_OUTLINE_MODEL"
_ENV_REWRITE_PROVIDER = "COURSEFORGE_REWRITE_PROVIDER"
_ENV_REWRITE_MODEL = "COURSEFORGE_REWRITE_MODEL"
_ENV_LEGACY_PROVIDER = "COURSEFORGE_PROVIDER"

# Per-Phase-3 §4: outline tier defaults to a 7B local model; rewrite
# tier prefers a multi-step-reasoning Anthropic model for blocks that
# require deeper pedagogy (assessment items, prereq sets, misconceptions)
# and a larger local model for everything else. These values are
# starting points subject to Phase 4 calibration.
_DEFAULT_OUTLINE_MODEL = "qwen2.5:7b-instruct-q4_K_M"
_DEFAULT_REWRITE_MODEL_LOCAL = "qwen2.5:14b-instruct-q4_K_M"
_DEFAULT_REWRITE_MODEL_ANTHROPIC = "claude-sonnet-4-6"

# Block types that route through the Anthropic rewrite default — the
# multi-step-reasoning workloads where the higher-capability model
# noticeably improves emit quality.
_REWRITE_ANTHROPIC_BLOCK_TYPES: frozenset = frozenset(
    {
        "prereq_set",
        "assessment_item",
        "misconception",
    }
)


# ---------------------------------------------------------------------------
# Touch-provider mapping
# ---------------------------------------------------------------------------


def _collapse_to_touch_provider(provider: str) -> str:
    """Map a router provider tag onto the canonical Touch provider set.

    ``Courseforge/scripts/blocks.py::_TOUCH_PROVIDERS`` only allows
    ``{"anthropic","local","together","claude_session","deterministic"}``;
    the router supports a fourth value ``"openai_compatible"`` (operators
    pointing at a non-Ollama / non-Together OpenAI-compatible server).
    Collapse the new value onto ``"local"`` — both go through the same
    OpenAICompatibleClient so the audit trail's ``provider`` field stays
    informative without breaking Touch validation.
    """
    if provider == "openai_compatible":
        return "local"
    return provider


# ---------------------------------------------------------------------------
# BlockProviderSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockProviderSpec:
    """Per-(block_type, tier) provider configuration.

    Frozen because the router caches resolved specs by key and a stale
    cache entry would silently route to a wrong provider. Carries the
    minimum fields the dispatch path needs:

    - ``block_type`` / ``tier`` — identity (validated against the
      canonical sets in :func:`__post_init__`).
    - ``provider`` — one of ``{"anthropic","together","local","openai_compatible"}``.
    - ``model`` — provider-specific model id.
    - ``base_url`` — optional override for OpenAI-compatible backends.
    - ``api_key_env`` — optional env-var name the provider should read
      its API key from instead of the default
      (``ANTHROPIC_API_KEY`` / ``TOGETHER_API_KEY`` /
      ``LOCAL_SYNTHESIS_API_KEY``).
    - ``temperature`` / ``max_tokens`` — sampling knobs.
    - ``extra_payload`` — opaque dict merged into the OpenAI-compatible
      POST body (grammar / guided_json / response_format / format —
      see :meth:`_BaseLLMProvider._dispatch_call`).
    - ``escalate_immediately`` — when ``True`` the outline tier is
      skipped entirely and the rewrite tier is dispatched with
      ``escalation_marker="outline_skipped_by_policy"``. Used for
      blocks where the outline-tier model is known to fail the
      structural contract (operator opt-in via block_routing.yaml).
    """

    block_type: str
    tier: Literal["outline", "rewrite"]
    provider: Literal["anthropic", "together", "local", "openai_compatible"]
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2400
    extra_payload: Dict[str, Any] = field(default_factory=dict)
    escalate_immediately: bool = False

    def __post_init__(self) -> None:
        if self.tier not in _ALLOWED_TIERS:
            raise ValueError(
                f"BlockProviderSpec.tier must be one of "
                f"{list(_ALLOWED_TIERS)}; got {self.tier!r}"
            )
        if self.provider not in _ALLOWED_PROVIDERS:
            raise ValueError(
                f"BlockProviderSpec.provider must be one of "
                f"{list(_ALLOWED_PROVIDERS)}; got {self.provider!r}"
            )
        if not self.block_type:
            raise ValueError("BlockProviderSpec.block_type must be non-empty")
        if not self.model:
            raise ValueError("BlockProviderSpec.model must be non-empty")


# ---------------------------------------------------------------------------
# Hardcoded fallback table
# ---------------------------------------------------------------------------


def _build_hardcoded_defaults() -> Dict[Tuple[str, str], BlockProviderSpec]:
    """Build the per-(block_type, tier) hardcoded defaults table.

    Mirrors Phase 3 §4 pre-resolved decisions:
    - outline → 7B local Qwen for ALL block types.
    - rewrite → Anthropic Sonnet for ``prereq_set`` / ``assessment_item``
      / ``misconception``; 14B local Qwen for everything else.

    Loaded once at module import; the router consumes this table as the
    final fallback when neither per-call kwargs, YAML policy, nor env
    vars resolve a spec. Every value in
    :data:`Courseforge.scripts.blocks.BLOCK_TYPES` has both an outline
    and a rewrite entry.
    """
    table: Dict[Tuple[str, str], BlockProviderSpec] = {}
    for block_type in BLOCK_TYPES:
        table[(block_type, "outline")] = BlockProviderSpec(
            block_type=block_type,
            tier="outline",
            provider="local",
            model=_DEFAULT_OUTLINE_MODEL,
            temperature=0.0,
            max_tokens=1200,
        )
        if block_type in _REWRITE_ANTHROPIC_BLOCK_TYPES:
            table[(block_type, "rewrite")] = BlockProviderSpec(
                block_type=block_type,
                tier="rewrite",
                provider="anthropic",
                model=_DEFAULT_REWRITE_MODEL_ANTHROPIC,
                temperature=0.4,
                max_tokens=2400,
            )
        else:
            table[(block_type, "rewrite")] = BlockProviderSpec(
                block_type=block_type,
                tier="rewrite",
                provider="local",
                model=_DEFAULT_REWRITE_MODEL_LOCAL,
                temperature=0.4,
                max_tokens=2400,
            )
    return table


_HARDCODED_DEFAULTS: Dict[Tuple[str, str], BlockProviderSpec] = (
    _build_hardcoded_defaults()
)


__all__ = [
    "BlockProviderSpec",
    "_HARDCODED_DEFAULTS",
    "_collapse_to_touch_provider",
]
