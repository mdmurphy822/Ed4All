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
import datetime as _dt
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


# ---------------------------------------------------------------------------
# CourseforgeRouter
# ---------------------------------------------------------------------------


class CourseforgeRouter:
    """Two-pass dispatch surface for Phase 3 content generation.

    Owns three responsibilities:

    1. Resolve a :class:`BlockProviderSpec` for each ``(block, tier)``
       per Phase 3 §3.3 (per-call kwargs → YAML policy → env vars →
       hardcoded defaults).
    2. Lazy-instantiate the Outline / Rewrite providers on first use
       (so a router constructed for a YAML-only run that never calls
       a tier doesn't pay the import / construction cost).
    3. Dispatch per-block via :meth:`route` and per-list via
       :meth:`route_all` (two-pass over a Block list).

    Wave-N scope: the YAML policy lookup is a stub (returns ``None``)
    until Subtask 34 lands the loader. Self-consistency loop, regen
    budget, inter-tier validators, and gate plumbing land in Wave N+1
    (Subtasks 36-43). The router method signatures already accommodate
    those features so the Wave-N+1 fill-ins don't re-shape the public
    surface.
    """

    def __init__(
        self,
        *,
        policy: Optional[Any] = None,
        outline_provider: Optional[Any] = None,
        rewrite_provider: Optional[Any] = None,
        capture: Optional[Any] = None,
        deterministic_gates: Optional[List[Any]] = None,
        statistical_filter: Optional[Any] = None,
        n_candidates: Optional[int] = None,
        regen_budget: Optional[int] = None,
    ) -> None:
        # YAML policy (Subtask 34); ``None`` for Wave N — the loader
        # lands in a follow-up subtask. ``_resolve_spec`` skips the
        # policy lookup when this is None.
        self._policy = policy

        # Optional provider injections — when set, ``_get_outline_provider``
        # / ``_get_rewrite_provider`` short-circuit to these instances
        # instead of constructing one. Used by tests to inject fakes.
        self._outline_provider_override: Optional[Any] = outline_provider
        self._rewrite_provider_override: Optional[Any] = rewrite_provider

        self._capture = capture
        self._deterministic_gates: List[Any] = list(deterministic_gates or [])
        self._statistical_filter = statistical_filter
        self._n_candidates_override: Optional[int] = n_candidates
        self._regen_budget_override: Optional[int] = regen_budget

        # Lazy-instantiated provider cache. Keyed by spec hash so a per-
        # block-type route that resolves a different model than the
        # constructor-default doesn't reuse the wrong provider instance.
        self._provider_cache: Dict[Tuple[str, str, str], Any] = {}

    # ------------------------------------------------------------------
    # Spec resolution
    # ------------------------------------------------------------------

    def _resolve_spec(
        self,
        block: Block,
        tier: str,
        **overrides: Any,
    ) -> BlockProviderSpec:
        """Resolve the :class:`BlockProviderSpec` for ``(block, tier)``.

        Resolution order (Phase 3 §3.3):

        1. Per-call ``**overrides`` — when ``provider`` / ``model`` /
           ``base_url`` / ``temperature`` / ``max_tokens`` /
           ``escalate_immediately`` is supplied as a kwarg, the override
           wins outright. ``provider`` and ``model`` together fully
           specify the spec; partial overrides (e.g. ``provider`` only)
           merge over the next-most-specific source.
        2. YAML policy entry for ``(block.block_type, tier)`` — Wave N
           stub returns ``None``; Subtask 34 fills it in.
        3. Tier-default env vars
           (``COURSEFORGE_OUTLINE_PROVIDER`` / ``COURSEFORGE_OUTLINE_MODEL``
           / ``COURSEFORGE_REWRITE_PROVIDER`` / ``COURSEFORGE_REWRITE_MODEL``).
        4. Hardcoded defaults table (:data:`_HARDCODED_DEFAULTS`).
        """
        if tier not in _ALLOWED_TIERS:
            raise ValueError(
                f"_resolve_spec: tier must be one of "
                f"{list(_ALLOWED_TIERS)}; got {tier!r}"
            )

        # 4. Hardcoded default (always present — populated for every
        # value in BLOCK_TYPES at module import).
        baseline = _HARDCODED_DEFAULTS.get((block.block_type, tier))
        if baseline is None:
            # Fail-loud — an unknown block_type at the router level
            # means Block.__post_init__ accepted a value the router's
            # defaults don't cover, which is a bug.
            raise ValueError(
                f"_resolve_spec: no hardcoded default for "
                f"(block_type={block.block_type!r}, tier={tier!r})"
            )

        # Build the resolved spec by overlaying each layer in reverse
        # priority order so the highest-priority source wins.
        resolved: BlockProviderSpec = baseline

        # 3. Tier-default env vars.
        env_provider, env_model = self._read_tier_env(tier)
        env_overrides: Dict[str, Any] = {}
        if env_provider:
            env_overrides["provider"] = env_provider
        if env_model:
            env_overrides["model"] = env_model
        if env_overrides:
            resolved = self._apply_overrides(resolved, env_overrides)

        # 2. YAML policy — Wave N stub. When ``self._policy`` is non-None
        # and exposes a ``resolve(block_id, block_type, tier)`` method,
        # honour it. The loader (Subtask 34) returns a BlockProviderSpec
        # or None; None falls through to the next lower layer.
        policy_spec = self._policy_lookup(block, tier)
        if policy_spec is not None:
            resolved = policy_spec

        # 1. Per-call overrides (highest priority).
        if overrides:
            resolved = self._apply_overrides(resolved, overrides)

        return resolved

    @staticmethod
    def _read_tier_env(tier: str) -> Tuple[Optional[str], Optional[str]]:
        """Read tier-default env vars; ``(provider, model)``.

        Both values are ``None`` when the corresponding env var is unset
        or empty. The router treats blank strings the same as unset so
        an operator setting ``COURSEFORGE_OUTLINE_PROVIDER=""`` does not
        override the hardcoded default.
        """
        if tier == "outline":
            provider = os.environ.get(_ENV_OUTLINE_PROVIDER) or None
            model = os.environ.get(_ENV_OUTLINE_MODEL) or None
        else:
            provider = os.environ.get(_ENV_REWRITE_PROVIDER) or None
            model = os.environ.get(_ENV_REWRITE_MODEL) or None
        return (
            provider.strip() if provider else None,
            model.strip() if model else None,
        )

    def _policy_lookup(
        self, block: Block, tier: str
    ) -> Optional[BlockProviderSpec]:
        """Look up the YAML policy entry for ``(block_id, block_type, tier)``.

        Wave-N stub: when ``self._policy`` is ``None`` (default) or does
        not expose a ``resolve(...)`` method, returns ``None``. The
        Subtask-34 loader will return a frozen
        :class:`BlockRoutingPolicy` that exposes ``resolve``.
        """
        policy = self._policy
        if policy is None:
            return None
        resolve = getattr(policy, "resolve", None)
        if not callable(resolve):
            return None
        try:
            spec = resolve(block.block_id, block.block_type, tier)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "policy.resolve raised for (%s, %s, %s): %s",
                block.block_id, block.block_type, tier, exc,
            )
            return None
        if spec is None:
            return None
        if not isinstance(spec, BlockProviderSpec):
            logger.warning(
                "policy.resolve returned non-BlockProviderSpec %r; ignoring",
                type(spec).__name__,
            )
            return None
        return spec

    # ------------------------------------------------------------------
    # Provider construction (lazy)
    # ------------------------------------------------------------------

    def _get_outline_provider(self, spec: BlockProviderSpec) -> Any:
        """Return an OutlineProvider instance for ``spec``.

        Lazy-imports :class:`OutlineProvider` so the router module
        itself can be imported without pulling the provider's
        dependencies (httpx / anthropic SDK / OpenAICompatibleClient).
        Caches by ``(provider, model, base_url)`` so repeated dispatches
        for the same spec reuse the same instance — important because
        the OpenAI-compatible client maintains an internal connection
        pool we don't want to thrash.
        """
        if self._outline_provider_override is not None:
            return self._outline_provider_override
        cache_key = ("outline", spec.provider, spec.model)
        cached = self._provider_cache.get(cache_key)
        if cached is not None:
            return cached
        from Courseforge.generators._outline_provider import (  # noqa: PLC0415
            OutlineProvider,
        )
        # ``openai_compatible`` is the router's fourth provider value;
        # the OutlineProvider supports it natively (its
        # SUPPORTED_PROVIDERS includes it). Pass through unchanged so
        # the OutlineProvider can route through OpenAICompatibleClient
        # at a non-Ollama / non-Together base_url.
        provider_arg = spec.provider
        instance = OutlineProvider(
            provider=provider_arg if provider_arg != "openai_compatible" else "local",
            model=spec.model,
            base_url=spec.base_url,
            capture=self._capture,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )
        self._provider_cache[cache_key] = instance
        return instance

    def _get_rewrite_provider(self, spec: BlockProviderSpec) -> Any:
        """Return a RewriteProvider instance for ``spec``.

        Same lazy-import + cache strategy as :meth:`_get_outline_provider`.
        """
        if self._rewrite_provider_override is not None:
            return self._rewrite_provider_override
        cache_key = ("rewrite", spec.provider, spec.model)
        cached = self._provider_cache.get(cache_key)
        if cached is not None:
            return cached
        from Courseforge.generators._rewrite_provider import (  # noqa: PLC0415
            RewriteProvider,
        )
        # RewriteProvider's base only accepts {"anthropic","together","local"};
        # collapse "openai_compatible" to "local" so the constructor passes.
        provider_arg = (
            spec.provider if spec.provider != "openai_compatible" else "local"
        )
        instance = RewriteProvider(
            provider=provider_arg,
            model=spec.model,
            base_url=spec.base_url,
            capture=self._capture,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )
        self._provider_cache[cache_key] = instance
        return instance

    # ------------------------------------------------------------------
    # Per-block dispatch
    # ------------------------------------------------------------------

    def route(
        self,
        block: Block,
        *,
        tier: Literal["outline", "rewrite"],
        source_chunks: Optional[List[Any]] = None,
        objectives: Optional[List[Any]] = None,
        **overrides: Any,
    ) -> Block:
        """Dispatch a single Block through the chosen ``tier``.

        Steps:

        1. Resolve the :class:`BlockProviderSpec` via :meth:`_resolve_spec`.
        2. ``escalate_immediately`` short-circuit (outline tier only):
           when ``spec.escalate_immediately`` is True and ``tier`` is
           ``"outline"``, skip the LLM call entirely; return the block
           with ``escalation_marker="outline_skipped_by_policy"`` and a
           deterministic Touch entry so the rewrite tier sees the marker
           and routes through ``_render_escalated_user_prompt``.
        3. Lazy-instantiate the provider for ``spec`` via
           :meth:`_get_outline_provider` / :meth:`_get_rewrite_provider`.
        4. Dispatch to ``provider.generate_outline(block, ...)`` or
           ``provider.generate_rewrite(block, ...)``.
        5. Emit one ``block_outline_call`` / ``block_rewrite_call``
           decision-capture event with the resolved
           ``(provider, model, policy_source)`` audit metadata.

        Per Phase 3 §9.2 the provider classes already emit one event
        per LLM call; this router event is the additional
        policy-source audit event so a postmortem can reconstruct
        WHICH layer of the resolution chain governed the dispatch.
        """
        spec = self._resolve_spec(block, tier, **overrides)

        # 2. escalate_immediately short-circuit (outline tier only).
        if tier == "outline" and spec.escalate_immediately:
            timestamp = (
                _dt.datetime.now(_dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            touch_provider = _collapse_to_touch_provider(spec.provider)
            touch = Touch(
                model=spec.model,
                provider=touch_provider,
                tier="outline",
                timestamp=timestamp,
                decision_capture_id=self._build_router_capture_id(
                    block, "outline_skipped_by_policy"
                ),
                purpose="escalate_immediately",
            )
            short_circuited = dataclasses.replace(
                block,
                escalation_marker="outline_skipped_by_policy",
                touched_by=block.touched_by + (touch,),
            )
            self._emit_router_decision(
                tier=tier,
                spec=spec,
                block=block,
                policy_source=self._classify_policy_source(spec, overrides),
                outcome="short_circuited",
                extra_rationale=(
                    "spec.escalate_immediately=True; outline tier "
                    "skipped — block flagged for rewrite-tier escalation"
                ),
            )
            return short_circuited

        # 3 + 4. Lazy-instantiate + dispatch.
        if tier == "outline":
            provider_instance = self._get_outline_provider(spec)
            try:
                out = provider_instance.generate_outline(
                    block,
                    source_chunks=source_chunks or [],
                    objectives=objectives or [],
                )
                outcome = "success"
                err: Optional[str] = None
            except Exception as exc:
                outcome = "failed"
                err = f"{type(exc).__name__}: {exc}"[:200]
                self._emit_router_decision(
                    tier=tier,
                    spec=spec,
                    block=block,
                    policy_source=self._classify_policy_source(spec, overrides),
                    outcome=outcome,
                    extra_rationale=f"dispatch_error={err}",
                )
                raise
        else:  # rewrite
            provider_instance = self._get_rewrite_provider(spec)
            try:
                out = provider_instance.generate_rewrite(
                    block,
                    source_chunks=source_chunks or [],
                    objectives=objectives or [],
                )
                outcome = "success"
                err = None
            except Exception as exc:
                outcome = "failed"
                err = f"{type(exc).__name__}: {exc}"[:200]
                self._emit_router_decision(
                    tier=tier,
                    spec=spec,
                    block=block,
                    policy_source=self._classify_policy_source(spec, overrides),
                    outcome=outcome,
                    extra_rationale=f"dispatch_error={err}",
                )
                raise

        # 5. Emit the per-call router decision event.
        self._emit_router_decision(
            tier=tier,
            spec=spec,
            block=block,
            policy_source=self._classify_policy_source(spec, overrides),
            outcome=outcome,
        )
        return out

    # ------------------------------------------------------------------
    # Decision-capture helpers
    # ------------------------------------------------------------------

    def _classify_policy_source(
        self,
        spec: BlockProviderSpec,
        overrides: Dict[str, Any],
    ) -> str:
        """Classify which resolution layer produced the ``spec``.

        Returns one of ``"per_call"`` / ``"yaml_policy"`` / ``"env_var"``
        / ``"hardcoded_default"``. Used purely for decision-capture
        rationale; the router does NOT branch on this value.
        """
        if overrides:
            return "per_call"
        if self._policy is not None and getattr(self._policy, "resolve", None):
            try:
                if self._policy.resolve(
                    spec.block_type, spec.block_type, spec.tier
                ) is not None:
                    return "yaml_policy"
            except Exception:  # pragma: no cover — defensive
                pass
        env_provider, env_model = self._read_tier_env(spec.tier)
        if env_provider or env_model:
            return "env_var"
        return "hardcoded_default"

    def _build_router_capture_id(
        self, block: Block, marker: str
    ) -> str:
        """Build a deterministic capture ID for router-emitted Touches.

        Mirrors the ``in-memory:{id}`` form the providers fall back to
        when no DecisionCapture is wired. Including the block_id +
        marker keeps the Touch.decision_capture_id non-empty (Wave 112
        invariant) and traceable in postmortem logs.
        """
        return f"router:{block.block_id}:{marker}"

    def _emit_router_decision(
        self,
        *,
        tier: str,
        spec: BlockProviderSpec,
        block: Block,
        policy_source: str,
        outcome: str,
        extra_rationale: str = "",
    ) -> None:
        """Emit the per-route ``block_outline_call`` / ``block_rewrite_call``
        audit event.

        Swallows capture errors so a flaky capture handle never breaks
        the dispatch. The provider classes emit their own per-LLM-call
        event; this one is the router-layer audit event that records
        WHICH policy layer governed the dispatch (per Phase 3 §9.2).
        """
        if self._capture is None:
            return
        decision_type = (
            "block_outline_call" if tier == "outline" else "block_rewrite_call"
        )
        rationale_parts = [
            f"router_dispatch tier={tier}",
            f"block_id={block.block_id}",
            f"block_type={block.block_type}",
            f"page_id={block.page_id}",
            f"provider={spec.provider}",
            f"model={spec.model}",
            f"policy_source={policy_source}",
            f"outcome={outcome}",
        ]
        if spec.base_url:
            rationale_parts.append(f"base_url={spec.base_url}")
        if extra_rationale:
            rationale_parts.append(extra_rationale)
        rationale = "; ".join(rationale_parts)
        try:
            self._capture.log_decision(
                decision_type=decision_type,
                decision=(
                    f"router_route:{tier}:{block.block_type}:{block.block_id}"
                    f":{outcome}"
                ),
                rationale=rationale,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "router decision-capture emit failed: %s", exc
            )

    @staticmethod
    def _apply_overrides(
        baseline: BlockProviderSpec,
        overrides: Dict[str, Any],
    ) -> BlockProviderSpec:
        """Return a new BlockProviderSpec with ``overrides`` overlaid.

        ``overrides`` may carry any subset of the spec's fields. The
        ``block_type`` and ``tier`` fields are sticky — they cannot be
        overridden because they identify the spec, and the router's
        cache is keyed off them.
        """
        # Filter out keys we don't recognise so a typo doesn't silently
        # drop a value — TypeError on dataclasses.replace surfaces it.
        allowed = {
            "provider",
            "model",
            "base_url",
            "api_key_env",
            "temperature",
            "max_tokens",
            "extra_payload",
            "escalate_immediately",
        }
        clean: Dict[str, Any] = {
            k: v for k, v in overrides.items() if k in allowed and v is not None
        }
        if not clean:
            return baseline
        return dataclasses.replace(baseline, **clean)


__all__ = [
    "BlockProviderSpec",
    "CourseforgeRouter",
    "_HARDCODED_DEFAULTS",
    "_collapse_to_touch_provider",
]
