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

## Regeneration budget + escalation

Per Phase 3 §3.7 every block has a per-(block_type) regeneration
budget that bounds how many failed validator passes a block may
accumulate inside :meth:`CourseforgeRouter.route_with_self_consistency`
before the router stops sampling new outline candidates and hands the
block off to the rewrite tier with the canonical
``escalation_marker="outline_budget_exhausted"`` stamp. The budget is
how the router avoids burning indefinite outline-tier compute on a
block the outline model fundamentally cannot handle.

Two failure-mode paths feed the same downstream rewrite-tier branch:

1. **Budget-exhausted path** (Subtask 41 —
   :meth:`CourseforgeRouter.route_with_self_consistency`). The
   self-consistency loop bumps a cumulative ``validation_attempts``
   accumulator on every failed validator pass. When the accumulator
   meets or exceeds the resolved budget the loop breaks early; the
   last candidate is rebound via
   ``dataclasses.replace(..., escalation_marker="outline_budget_exhausted")``
   and a single ``block_escalation`` decision-capture event fires
   (``attempts=cumulative_count``, ``n_candidates=loop_iteration_index+1``).
2. **Policy-skip path** (Subtask 42 — :meth:`CourseforgeRouter.route`
   with ``tier="outline"`` and ``spec.escalate_immediately=True``).
   The outline LLM dispatch is skipped entirely — no candidates are
   ever generated. The same ``outline_budget_exhausted`` marker is
   stamped on the return block (because ``Block._ESCALATION_MARKERS``
   only admits ``{outline_budget_exhausted, structural_unfixable,
   validator_consensus_fail}``); policy-skip provenance is preserved
   on a ``Touch(purpose="escalate_immediately")`` audit record so a
   postmortem reader can distinguish the two paths. The same
   ``block_escalation`` event fires with ``attempts=0`` and
   ``n_candidates=0`` to signal the policy-skip discriminator.

NOTE on the marker-name deviation from Phase 3 §3.7. The plan text
discusses ``outline_skipped_by_policy`` as a separate marker for the
policy-skip path, but ``Block._ESCALATION_MARKERS`` does not include
that value (closing the marker set is a Phase-2 invariant). The
implementation collapses both paths onto the canonical
``outline_budget_exhausted`` marker; the policy-skip discriminator
lives on the ``Touch.purpose`` audit field, which is the canonical
provenance surface a postmortem reader consults. Both paths route
through the same rewrite-tier escalated-prompt branch so the
behavioural contract is preserved.

Budget resolution order (highest-priority first; mirrors the
n-candidates resolution chain):

1. Per-call ``regen_budget`` kwarg on
   :meth:`CourseforgeRouter.route_with_self_consistency`.
2. Policy fast-lookup map ``policy.regen_budget_by_block_type[block_type]``
   (Worker G's :class:`BlockRoutingPolicy` surface).
3. Env var ``COURSEFORGE_OUTLINE_REGEN_BUDGET`` (parsed as int; falls
   through silently on parse failure).
4. Constructor-time instance attribute
   :class:`CourseforgeRouter` ``regen_budget=...``.
5. Module-level default :data:`_DEFAULT_OUTLINE_REGEN_BUDGET` (3).

Rewrite-tier prompt branching: when the outline-tier hand-off carries
a non-None ``Block.escalation_marker``, the rewrite tier switches from
:meth:`RewriteProvider._render_user_prompt` to
:meth:`RewriteProvider._render_escalated_user_prompt` — a richer
prompt template that synthesises from source chunks + objectives
directly (rather than refining the outline draft) and re-asserts the
CURIE-preservation contract so any anchored vocabulary survives the
tier transition.

Cross-links: Phase 3 plan section §3.7 documents the contract;
Phase 5's ``--escalated-only`` CLI flag (planned) will let an operator
re-run the rewrite tier across only those blocks the corpus emitted
with a non-None ``escalation_marker``.
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
_ENV_OUTLINE_N_CANDIDATES = "COURSEFORGE_OUTLINE_N_CANDIDATES"
_ENV_OUTLINE_REGEN_BUDGET = "COURSEFORGE_OUTLINE_REGEN_BUDGET"

# Default outline-tier candidate count when neither per-call kwarg, policy
# entry, env var, nor instance attr resolves a value. Per Phase 3 §3.6
# the self-consistency loop dispatches up to N candidates and returns
# the first that passes the validator chain; N=3 balances latency
# against pass-rate at the 7B-class outline model.
_DEFAULT_OUTLINE_N_CANDIDATES = 3

# Default per-block regen budget when neither per-call kwarg, policy
# entry, env var, nor instance attr resolves a value. Per Phase 3 §3.7
# the budget is the number of failed validation attempts a block can
# accumulate inside the self-consistency loop before the router stamps
# ``escalation_marker="outline_budget_exhausted"`` and breaks early
# (escalating to the rewrite tier with an enriched prompt).
_DEFAULT_OUTLINE_REGEN_BUDGET = 3

# Subtask 48: validator-action priority. The router dispatches on the
# highest-priority action across all validator results in a candidate's
# gate-results list. Order: ``block`` is the most severe (the block is
# structurally unfixable — give up immediately); ``escalate`` is the
# rewrite-tier hand-off path; ``regenerate`` continues the self-
# consistency loop; ``pass`` is the steady-state success signal.
_ACTION_PRIORITY: Dict[str, int] = {
    "pass": 0,
    "regenerate": 1,
    "escalate": 2,
    "block": 3,
}

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

        # ------------------------------------------------------------------
        # Phase 3a four-layer precedence chain (Subtask 25).
        # ------------------------------------------------------------------
        # Resolution priority (highest → lowest):
        #
        #   1. Per-call kwargs (``**overrides``)
        #          — operator-explicit per-route override; wins outright.
        #          — applied LAST in this method so it overlays every
        #            other layer.
        #
        #   2. YAML policy (``self._policy.resolve(...)``)
        #          — operator-tunable ``Courseforge/config/block_routing.yaml``;
        #            wins over env vars + hardcoded defaults.
        #          — populated by the Subtask-34 loader; ``None`` when no
        #            YAML file is present (clean checkout) or when the
        #            policy doesn't carry an entry for this (block_id,
        #            block_type, tier) triple.
        #          — applied THIRD in this method (after the env-var
        #            overlay, BEFORE per-call overrides).
        #
        #   3. Tier-default env vars
        #          — ``COURSEFORGE_OUTLINE_PROVIDER`` /
        #            ``COURSEFORGE_OUTLINE_MODEL`` /
        #            ``COURSEFORGE_REWRITE_PROVIDER`` /
        #            ``COURSEFORGE_REWRITE_MODEL``.
        #          — supplements the baseline; doesn't replace it (env
        #            vars carry only ``provider`` / ``model``, not the
        #            full :class:`BlockProviderSpec` shape).
        #          — applied SECOND in this method (overlays the
        #            hardcoded baseline before the policy lookup).
        #
        #   4. Hardcoded defaults table (``_HARDCODED_DEFAULTS``)
        #          — final fallback; covers every (block_type, tier)
        #            value in :data:`Courseforge.scripts.blocks.BLOCK_TYPES`.
        #          — applied FIRST in this method as the baseline.
        #
        # The build order below is REVERSE priority so the highest-
        # priority source wins by overlaying the lower-priority layers.
        # Audit tests pinning this contract:
        # ``test_phase3a_env_var_overrides_hardcoded_default`` and
        # ``test_phase3a_yaml_wins_over_env_var`` in
        # ``Courseforge/router/tests/test_router.py``.

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
        remediation_suffix: Optional[str] = None,
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
            # ``Block.__post_init__`` validates ``escalation_marker``
            # against the canonical ``_ESCALATION_MARKERS`` set
            # (``{outline_budget_exhausted, structural_unfixable,
            # validator_consensus_fail}``). The rewrite provider's
            # ``_ESCALATION_MARKER_CONTEXT`` documents an additional
            # marker name (``outline_skipped_by_policy``) for the
            # router's policy-skip semantics, but Block validation
            # rejects it. Using ``outline_budget_exhausted`` keeps the
            # marker on Block-validated ground while preserving the
            # rewrite-tier escalated-prompt routing — the rewrite
            # provider treats both markers as "the outline tier did not
            # produce a valid emit; synthesize from source chunks".
            # The router's ``purpose="escalate_immediately"`` Touch and
            # the rationale on the ``block_outline_call`` decision
            # event preserve the policy-skip provenance for postmortem.
            short_circuit_marker = "outline_budget_exhausted"
            touch = Touch(
                model=spec.model,
                provider=touch_provider,
                tier="outline",
                timestamp=timestamp,
                decision_capture_id=self._build_router_capture_id(
                    block, short_circuit_marker
                ),
                purpose="escalate_immediately",
            )
            short_circuited = dataclasses.replace(
                block,
                escalation_marker=short_circuit_marker,
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
            # Subtask 42 + 43: emit a ``block_escalation`` decision-capture
            # event from the policy-skip path so the audit trail
            # surfaces a single canonical event for both budget-exhaustion
            # (Subtask 41) and policy-skip escalation classes. ``attempts``
            # is 0 because the outline tier never ran; ``n_candidates`` is
            # 0 for the same reason — both signal the policy-skip path
            # to a postmortem reader.
            self._emit_block_escalation(
                short_circuited,
                marker=short_circuit_marker,
                attempts=0,
                n_candidates=0,
            )
            return short_circuited

        # 3 + 4. Lazy-instantiate + dispatch.
        if tier == "outline":
            provider_instance = self._get_outline_provider(spec)
            # Phase 3.5 Subtask 18: thread the remediation_suffix
            # through to OutlineProvider.generate_outline so a re-rolled
            # candidate sees the prior failure context. The provider
            # widens the kwarg with a default of None (back-compat
            # preserved for callers that don't pass it).
            try:
                outline_kwargs: Dict[str, Any] = {
                    "source_chunks": source_chunks or [],
                    "objectives": objectives or [],
                }
                if remediation_suffix is not None:
                    outline_kwargs["remediation_suffix"] = remediation_suffix
                out = provider_instance.generate_outline(
                    block,
                    **outline_kwargs,
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
    # Two-pass dispatch over a Block list
    # ------------------------------------------------------------------

    def route_all(
        self,
        blocks: List[Block],
        *,
        source_chunks_by_block_id: Optional[Dict[str, List[Any]]] = None,
        objectives: Optional[List[Any]] = None,
    ) -> List[Block]:
        """Two-pass dispatch over an ordered list of Blocks.

        Returns the full list with input ordering preserved. Each block
        is routed through:

        1. **Outline tier** — :meth:`route` with ``tier="outline"`` per
           block. Single-candidate path for Wave N (the
           self-consistency loop is layered on top by Subtask 37 in
           Wave N+1). On dispatch failure the block is marked
           ``content="failed"``-style by capturing the exception and
           setting ``escalation_marker="outline_budget_exhausted"``;
           the failed block is included in the returned list but is
           NOT dispatched to the rewrite tier.
        2. **Inter-tier validation** — Wave N stub. The Wave N+1
           Subtasks 36/38 land the validator chain that decides whether
           a block proceeds to rewrite or is flagged failed. For
           Wave N every successful outline-tier emit proceeds to
           rewrite.
        3. **Rewrite tier** — :meth:`route` with ``tier="rewrite"``
           per surviving block. Failed-outline blocks skip this stage.

        ``source_chunks_by_block_id`` is an optional dict keyed by
        ``block.block_id`` carrying that block's pre-resolved source
        chunks; when absent the per-block source chunks are an empty
        list (the providers will note the absence in their prompts via
        ``_format_source_chunks(...)``).

        Wave-N constraints (per Worker 3D scope):
        - No self-consistency loop (single outline candidate per block).
        - No inter-tier validator chain (every successful outline
          dispatches straight to rewrite).
        - No regen budget tracking on the router level (the providers
          enforce their own parse-retry budgets internally).
        - Failed-outline blocks return early with an
          ``outline_budget_exhausted`` marker and skip the rewrite
          stage; the caller sees them in the returned list at their
          original position so downstream packaging can persist them
          for re-execution.
        """
        chunks_lookup: Dict[str, List[Any]] = source_chunks_by_block_id or {}
        objectives_list: List[Any] = list(objectives or [])

        # Pass 1: outline tier per block.
        outline_results: List[Tuple[int, Block, bool]] = []
        for idx, block in enumerate(blocks):
            block_chunks = chunks_lookup.get(block.block_id, [])
            try:
                outlined = self.route(
                    block,
                    tier="outline",
                    source_chunks=block_chunks,
                    objectives=objectives_list,
                )
                # When the outline tier short-circuited via
                # ``escalate_immediately``, ``outlined.escalation_marker``
                # is non-None — that's the signal the rewrite-tier
                # branch routes through ``_render_escalated_user_prompt``.
                # The block still proceeds to rewrite (the marker is
                # the routing signal, not a failure).
                outline_results.append((idx, outlined, True))
            except Exception as exc:
                logger.warning(
                    "route_all: outline tier failed for block_id=%s: %s",
                    block.block_id, exc,
                )
                # Mark the block as outline-failed so it's persisted
                # (with marker) for re-execution and skipped by the
                # rewrite pass. ``escalation_marker`` lives in the
                # canonical _ESCALATION_MARKERS set, which keeps
                # Block.__post_init__ from raising on the replace.
                failed = dataclasses.replace(
                    block,
                    escalation_marker="outline_budget_exhausted",
                )
                outline_results.append((idx, failed, False))

        # Pass 2 (Wave N stub): inter-tier validation. Every successful
        # outline emit proceeds to rewrite. Subtasks 36/38 in Wave N+1
        # plug in the validator chain.

        # Pass 3: rewrite tier per surviving block.
        rewrite_results: List[Tuple[int, Block]] = []
        for idx, outlined, ok in outline_results:
            if not ok:
                # Outline-failed blocks bypass rewrite; they ride the
                # return list at their original index.
                rewrite_results.append((idx, outlined))
                continue
            block_chunks = chunks_lookup.get(outlined.block_id, [])
            try:
                rewritten = self.route(
                    outlined,
                    tier="rewrite",
                    source_chunks=block_chunks,
                    objectives=objectives_list,
                )
                rewrite_results.append((idx, rewritten))
            except Exception as exc:
                logger.warning(
                    "route_all: rewrite tier failed for block_id=%s: %s",
                    outlined.block_id, exc,
                )
                # Rewrite failure: keep the outlined block (with marker)
                # so the caller can persist for re-execution.
                failed_rewrite = (
                    outlined
                    if outlined.escalation_marker is not None
                    else dataclasses.replace(
                        outlined,
                        escalation_marker="validator_consensus_fail",
                    )
                )
                rewrite_results.append((idx, failed_rewrite))

        # Reassemble ordered output.
        rewrite_results.sort(key=lambda pair: pair[0])
        return [b for _, b in rewrite_results]

    # ------------------------------------------------------------------
    # Self-consistency dispatch (Phase 3 §3.6 — Subtask 37)
    # ------------------------------------------------------------------

    def route_with_self_consistency(
        self,
        block: Block,
        *,
        n_candidates: Optional[int] = None,
        regen_budget: Optional[int] = None,
        validators: Optional[List[Any]] = None,
        source_chunks: Optional[List[Any]] = None,
        objectives: Optional[List[Any]] = None,
        fast_fail: bool = True,
        **overrides: Any,
    ) -> Block:
        """Sample N outline candidates and return the first that passes
        the validator chain.

        Per Phase 3 §3.6 self-consistency loop:

        1. Resolve ``n`` from arg → ``policy.n_candidates_by_block_type``
           → ``COURSEFORGE_OUTLINE_N_CANDIDATES`` env var → instance
           attribute → :data:`_DEFAULT_OUTLINE_N_CANDIDATES` (3).
        2. ``escalate_immediately`` short-circuit: when the resolved
           spec carries ``escalate_immediately=True`` the outline tier
           is skipped entirely; ``route(block, tier="outline", ...)``
           handles that path and returns a Block with the
           ``outline_skipped_by_policy`` provenance Touch — we delegate
           and return the result without entering the candidate loop.
        3. For ``i in range(n)``: dispatch one outline candidate via
           :meth:`route` (single ``tier="outline"`` call). Run the
           validator chain (Subtask 38) on the candidate. If all pass,
           return the block with a ``Touch(purpose="self_consistency_winner")``
           appended carrying the ``winning_candidate_index=i`` audit
           field on the Subtask-39 decision-capture event.
        4. Subtask 48: dispatch on the highest-priority action across
           all validator results in the gate-results list. Priority
           order ``block > escalate > regenerate > pass``:

           - ``block`` → stamp ``escalation_marker="structural_unfixable"``,
             break the loop immediately, return for downstream consumers
             to skip the rewrite tier entirely. Bypasses the regen budget
             because structural-unfixable is a "give up entirely" signal.
           - ``escalate`` → stamp
             ``escalation_marker="validator_consensus_fail"``, break the
             loop, return for the rewrite tier's enriched-prompt branch.
           - ``regenerate`` → bump the cumulative validation_attempts
             accumulator and continue the loop. On budget exhaustion
             (Subtask 41) stamp ``outline_budget_exhausted`` and break.
           - ``pass`` → return the candidate (winner branch above).

        5. If all N candidates fail every validator without triggering
           the block / escalate / budget-exhaustion paths, return the
           LAST candidate with ``validation_attempts=n``.
        6. Records per-candidate failure distribution into a local dict
           that's emitted on the audit event by Subtask 39, and per-
           validator action events via Subtask 47.

        Returns the winning block on success, or the last candidate
        with the appropriate escalation marker on failure.
        """
        # 1. Resolve n_candidates per the precedence chain.
        resolved_n = self._resolve_n_candidates(block, n_candidates)

        # Validators default to an empty list (Phase 4+ wires concrete
        # validators in via the inter_tier_validation phase / Subtask
        # 50). Empty list → first candidate "passes" trivially → loop
        # exits after one dispatch with no validation gating.
        validator_list: List[Any] = list(validators or [])

        # 2. Resolve the spec once — its ``escalate_immediately`` flag
        # short-circuits the outline tier per the contract in
        # :meth:`route`. We delegate so the policy-skip Touch +
        # decision-event emit live in one place.
        spec = self._resolve_spec(block, "outline", **overrides)
        if spec.escalate_immediately:
            return self.route(
                block,
                tier="outline",
                source_chunks=source_chunks,
                objectives=objectives,
                **overrides,
            )

        # 3a. Resolve the regen_budget per the precedence chain (Subtask 41).
        # When the budget is exhausted mid-loop the candidate is stamped
        # with ``escalation_marker="outline_budget_exhausted"`` and the
        # loop breaks early — the rewrite tier sees the marker and
        # routes through the escalated-prompt branch.
        resolved_budget = self._resolve_regen_budget(block, regen_budget)

        # 3. Sequential N-candidate loop.
        # ``failure_distribution`` keys are validator names; values are
        # per-validator failure counts across all N candidates.
        # ``last_candidate`` carries the most recent dispatch output so
        # the all-fail branch can return it with the correct
        # ``validation_attempts`` count. ``escalated`` carries the
        # mid-loop escalation block when the regen budget is exhausted
        # (Subtask 41); it short-circuits the post-loop return resolution.
        # ``blocked`` carries the mid-loop short-circuit block when a
        # validator returned action="block" (Subtask 48 — structural-
        # failure path that bypasses both the regen budget and the
        # rewrite tier; the block's escalation_marker is stamped
        # ``structural_unfixable`` so downstream consumers (route_all,
        # packaging) skip rewrite per Phase 4 §1.5 mapping).
        failure_distribution: Dict[str, int] = {}
        last_candidate: Optional[Block] = None
        winning_index: Optional[int] = None
        winner: Optional[Block] = None
        escalated: Optional[Block] = None
        blocked: Optional[Block] = None

        # ``cumulative_attempts`` tracks the running validation_attempts
        # count across candidates so the regen-budget check sees the
        # cumulative total (Subtask 41). Each candidate is a fresh
        # output from the outline provider — its own
        # ``validation_attempts`` field is typically 0 — so the budget
        # accumulator lives on the loop, not on the per-candidate
        # Block. The final ``last_candidate`` is rebound with the
        # cumulative count before being returned.
        cumulative_attempts = block.validation_attempts

        # Phase 3.5 Subtask 18: per-loop remediation suffix carried
        # over from the prior candidate's validator-chain failures.
        # ``None`` on the first iteration (no prior failures); set to
        # the canonical ``_append_remediation_for_gates`` output after
        # each failed validator chain so the next candidate sees what
        # went wrong on the previous attempt and the directive to fix it.
        remediation_suffix: Optional[str] = None

        for i in range(resolved_n):
            candidate = self.route(
                block,
                tier="outline",
                source_chunks=source_chunks,
                objectives=objectives,
                remediation_suffix=remediation_suffix,
                **overrides,
            )
            last_candidate = candidate

            # Phase 3.5 Subtask 17: _run_validator_chain now returns a
            # 3-tuple including the cumulatively-touched block. Each
            # validator append a Touch with tier="outline_val" so the
            # audit chain records the full validator pass/fail history.
            # We rebind ``candidate`` (and ``last_candidate``) to the
            # touched block so the audit chain survives every downstream
            # branch (winner, escalation, structural-block, all-fail).
            all_passed, gate_results, candidate = self._run_validator_chain(
                candidate, validator_list, fast_fail=fast_fail,
                validator_tier="outline_val",
            )
            last_candidate = candidate
            if all_passed:
                # Append a "self_consistency_winner" Touch so the audit
                # chain records WHICH candidate index won. Reuses the
                # short-circuit Touch construction pattern from
                # ``route``.
                timestamp = (
                    _dt.datetime.now(_dt.timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                touch = Touch(
                    model=spec.model,
                    provider=_collapse_to_touch_provider(spec.provider),
                    tier="outline",
                    timestamp=timestamp,
                    decision_capture_id=self._build_router_capture_id(
                        candidate, f"self_consistency_winner_{i}"
                    ),
                    purpose="self_consistency_winner",
                )
                winner = candidate.with_touch(touch)
                winning_index = i
                break

            # Failure: increment ``validation_attempts`` per Subtask 41
            # + bump the per-validator counter for the audit event.
            # ``GateResult.derive_default_action`` collapses legacy
            # validators (no ``action`` set) onto ``"block"`` on failure
            # — the failure-distribution dict counts the gate name
            # regardless of which discriminator surfaced it.
            from MCP.hardening.validation_gates import GateResult  # noqa: PLC0415

            # Subtask 48: compute the highest-priority action across
            # ALL gate_results. ``_ACTION_PRIORITY`` orders the actions
            # block > escalate > regenerate > pass; the router dispatches
            # on the most severe action in the result list.
            #
            # Back-compat note: legacy validators that don't set the
            # ``action`` field ALWAYS map to ``"regenerate"`` for the
            # router's loop-control semantics (continue the retry loop)
            # — NOT the ``"block"`` value ``derive_default_action``
            # returns. The block / escalate paths are opt-in: a Phase-3-
            # aware validator must explicitly set ``action="block"`` or
            # ``"escalate"`` to engage the structural-failure /
            # consensus-fail short-circuit. This keeps every pre-Phase-3
            # test (validators with passed=False / action=None) running
            # the legacy retry loop unchanged.
            highest_action = "pass"
            highest_priority = 0
            for gate_result in gate_results:
                explicit_action = getattr(gate_result, "action", None)
                if explicit_action is not None:
                    action = explicit_action
                else:
                    # Legacy validator: pass on success, regenerate on
                    # failure (retry-loop semantics).
                    action = "pass" if gate_result.passed else "regenerate"
                if action == "pass":
                    continue
                # Use validator_name when present; fall back to
                # gate_id for legacy validators that don't set the
                # human-readable name.
                gate_name = (
                    getattr(gate_result, "validator_name", None)
                    or getattr(gate_result, "gate_id", None)
                    or "unknown_validator"
                )
                failure_distribution[gate_name] = (
                    failure_distribution.get(gate_name, 0) + 1
                )
                priority = _ACTION_PRIORITY.get(action, 0)
                if priority > highest_priority:
                    highest_priority = priority
                    highest_action = action

            # Subtask 48: ``block`` action — structural failure that
            # bypasses both the regen-budget retry loop and the rewrite
            # tier. Stamp ``escalation_marker="structural_unfixable"``
            # so route_all (and the downstream packager) detect the
            # block as outline-failed and skip the rewrite pass; the
            # marker IS in the canonical ``_ESCALATION_MARKERS`` set.
            # The block does NOT contribute to ``cumulative_attempts``
            # because the regen budget is for retry-able failures —
            # ``block`` action is a "give up entirely" signal that
            # bypasses retry semantics.
            if highest_action == "block":
                blocked = dataclasses.replace(
                    last_candidate,
                    escalation_marker="structural_unfixable",
                    validation_attempts=cumulative_attempts + 1,
                )
                # Audit-trail: the block_validation_action events were
                # already emitted inside _run_validator_chain (Subtask
                # 47). Reuse ``_emit_block_escalation`` for the route-
                # level escalation event so the JSONL stream shows the
                # same event class for every escalation path
                # (regardless of which path landed it). ``attempts=0``
                # signals the structural-unfixable path because the
                # regen budget was NOT consumed.
                self._emit_block_escalation(
                    blocked,
                    marker="structural_unfixable",
                    attempts=0,
                    n_candidates=i + 1,
                )
                break

            # Subtask 48: ``escalate`` action — exit the candidate loop
            # immediately and route through the rewrite tier with the
            # ``validator_consensus_fail`` marker. The marker IS in the
            # canonical ``_ESCALATION_MARKERS`` set; the rewrite-tier
            # provider sees the marker and switches to the enriched
            # ``_render_escalated_user_prompt`` template per the contract
            # in the module docstring (lines 99-106).
            if highest_action == "escalate":
                escalated = dataclasses.replace(
                    last_candidate,
                    escalation_marker="validator_consensus_fail",
                    validation_attempts=cumulative_attempts + 1,
                )
                self._emit_block_escalation(
                    escalated,
                    marker="validator_consensus_fail",
                    attempts=escalated.validation_attempts,
                    n_candidates=i + 1,
                )
                break

            # Subtask 41 + 48: ``regenerate`` (or unset/derived
            # ``block`` from a legacy passed=False) → bump the
            # cumulative validation_attempts accumulator and continue
            # the self-consistency loop. Note that legacy validators
            # without an explicit action that fail map to ``"block"``
            # via derive_default_action, but we treat them as
            # regenerate-equivalent here (continue the loop) because
            # the legacy contract — pre-Phase-3 — ALWAYS expected
            # retry-on-failure behaviour. New Phase-3-aware validators
            # opt INTO the structural-unfixable path by setting
            # ``action="block"`` explicitly; the
            # ``derive_default_action`` collapse is a back-compat
            # bridge, not a structural-failure signal.
            cumulative_attempts += 1
            last_candidate = dataclasses.replace(
                last_candidate,
                validation_attempts=cumulative_attempts,
            )

            # Subtask 41: regen-budget check. When the bumped count meets
            # or exceeds the resolved budget, stamp the canonical
            # ``outline_budget_exhausted`` marker and break early.
            # ``Block.__post_init__`` validates the marker against the
            # canonical ``_ESCALATION_MARKERS`` set.
            if cumulative_attempts >= resolved_budget:
                escalated = dataclasses.replace(
                    last_candidate,
                    escalation_marker="outline_budget_exhausted",
                )
                self._emit_block_escalation(
                    escalated,
                    marker="outline_budget_exhausted",
                    attempts=escalated.validation_attempts,
                    n_candidates=i + 1,
                )
                break

            # Phase 3.5 Subtask 18: build the remediation suffix from
            # the prior validator-chain failures so the next outline
            # candidate sees what went wrong on this attempt and the
            # canonical directive to fix it. The empty-string base
            # plus the helper's per-failure block emit produces a
            # non-empty suffix only when at least one actionable
            # failure exists (the helper passes through when every
            # gate result has action="pass").
            from Courseforge.router.remediation import (  # noqa: PLC0415
                _append_remediation_for_gates,
            )
            built_suffix = _append_remediation_for_gates("", gate_results)
            # ``_append_remediation_for_gates`` returns "" unchanged when
            # no actionable failures exist; lstrip to drop the leading
            # "\n\n" the helper inserts so the OutlineProvider's
            # _render_user_prompt seam doesn't double-pad the prompt.
            remediation_suffix = (
                built_suffix.lstrip("\n") if built_suffix else None
            )

        # 4. Resolve the return-block.
        if winner is not None:
            outcome_block = winner
            failed_count = winning_index if winning_index is not None else 0
        elif blocked is not None:
            # Subtask 48: a validator returned action="block" mid-loop.
            # ``blocked`` carries the ``structural_unfixable`` marker;
            # downstream consumers detect the marker and skip the
            # rewrite tier entirely. ``failed_count`` reports the
            # cumulative attempts consumed before the block fired so
            # the audit event reflects the outline-tier compute spent.
            outcome_block = blocked
            failed_count = cumulative_attempts + 1
        elif escalated is not None:
            # Subtask 41 / Subtask 48: outline tier escalation. Either
            # the regen budget was exhausted (Subtask 41 marker
            # ``outline_budget_exhausted``) or a validator returned
            # action="escalate" (Subtask 48 marker
            # ``validator_consensus_fail``). Both paths route the
            # block through the rewrite tier's enriched-prompt branch.
            outcome_block = escalated
            failed_count = cumulative_attempts
        else:
            # All N candidates failed every validator but the regen
            # budget was higher than N (so no escalation marker fired).
            # ``last_candidate`` already carries the correct cumulative
            # ``validation_attempts`` count from the per-failure
            # increment above; no further bump needed.
            assert last_candidate is not None  # the loop ran at least once
            outcome_block = last_candidate
            failed_count = resolved_n

        # 5. Emit the per-self-consistency-loop decision-capture event
        # (Subtask 39 lands the ml_features payload). For Subtask 37
        # we already emit the audit event with the winning index +
        # failure distribution baked into the rationale string; the
        # Subtask 39 commit promotes the same data into the
        # structured ``ml_features`` payload for ML-trainability.
        self._emit_self_consistency_decision(
            block=block,
            spec=spec,
            n_candidates_requested=resolved_n,
            winning_candidate_index=winning_index,
            failed_candidate_count=failed_count,
            validator_failure_distribution=failure_distribution,
        )

        return outcome_block

    def _resolve_n_candidates(
        self, block: Block, override: Optional[int]
    ) -> int:
        """Resolve the per-block N-candidate count.

        Precedence (highest first):

        1. ``override`` arg (the per-call ``n_candidates`` kwarg).
        2. ``self._policy.n_candidates_by_block_type[block.block_type]``
           (Worker G's fast-lookup map on :class:`BlockRoutingPolicy`).
        3. ``COURSEFORGE_OUTLINE_N_CANDIDATES`` env var (parsed as int;
           silently falls through on parse failure).
        4. ``self._n_candidates_override`` (constructor-time instance
           attribute set via ``CourseforgeRouter(n_candidates=...)``).
        5. :data:`_DEFAULT_OUTLINE_N_CANDIDATES` (3).
        """
        # 1. Per-call kwarg.
        if isinstance(override, int) and override > 0:
            return override

        # 2. Policy fast-lookup map (Worker G).
        policy = self._policy
        if policy is not None:
            policy_map = getattr(policy, "n_candidates_by_block_type", None)
            if isinstance(policy_map, dict):
                policy_n = policy_map.get(block.block_type)
                if isinstance(policy_n, int) and policy_n > 0:
                    return policy_n

        # 3. Env var.
        env_value = os.environ.get(_ENV_OUTLINE_N_CANDIDATES)
        if env_value:
            try:
                env_n = int(env_value.strip())
                if env_n > 0:
                    return env_n
            except (TypeError, ValueError):
                # Parse failure falls through to the next layer.
                pass

        # 4. Constructor-time instance attribute.
        if (
            isinstance(self._n_candidates_override, int)
            and self._n_candidates_override > 0
        ):
            return self._n_candidates_override

        # 5. Hardcoded default.
        return _DEFAULT_OUTLINE_N_CANDIDATES

    def _resolve_regen_budget(
        self, block: Block, override: Optional[int]
    ) -> int:
        """Resolve the per-block regen budget (Subtask 41).

        Precedence (highest first):

        1. ``override`` arg (the per-call ``regen_budget`` kwarg).
        2. ``self._policy.regen_budget_by_block_type[block.block_type]``
           (Worker G's fast-lookup map on :class:`BlockRoutingPolicy`).
        3. ``COURSEFORGE_OUTLINE_REGEN_BUDGET`` env var (parsed as int;
           silently falls through on parse failure).
        4. ``self._regen_budget_override`` (constructor-time instance
           attribute set via ``CourseforgeRouter(regen_budget=...)``).
        5. :data:`_DEFAULT_OUTLINE_REGEN_BUDGET` (3).

        The budget is the number of failed validation passes a block
        can accumulate inside :meth:`route_with_self_consistency`
        before the router stamps
        ``escalation_marker="outline_budget_exhausted"`` and breaks
        early.
        """
        # 1. Per-call kwarg.
        if isinstance(override, int) and override > 0:
            return override

        # 2. Policy fast-lookup map (Worker G).
        policy = self._policy
        if policy is not None:
            policy_map = getattr(policy, "regen_budget_by_block_type", None)
            if isinstance(policy_map, dict):
                policy_b = policy_map.get(block.block_type)
                if isinstance(policy_b, int) and policy_b > 0:
                    return policy_b

        # 3. Env var.
        env_value = os.environ.get(_ENV_OUTLINE_REGEN_BUDGET)
        if env_value:
            try:
                env_b = int(env_value.strip())
                if env_b > 0:
                    return env_b
            except (TypeError, ValueError):
                # Parse failure falls through to the next layer.
                pass

        # 4. Constructor-time instance attribute.
        if (
            isinstance(self._regen_budget_override, int)
            and self._regen_budget_override > 0
        ):
            return self._regen_budget_override

        # 5. Hardcoded default.
        return _DEFAULT_OUTLINE_REGEN_BUDGET

    def _run_validator_chain(
        self,
        block: Block,
        validators: List[Any],
        *,
        fast_fail: bool = True,
        validator_tier: Literal["outline_val", "rewrite_val"] = "outline_val",
    ) -> Tuple[bool, List[Any], Block]:
        """Run an ordered chain of validators against ``block``.

        Cheapest-first ordering per Phase 3 §3.6 (the caller is
        responsible for pre-sorting ``validators`` into this order; the
        router walks the list as given so per-block-type weighting and
        Phase 4 / Phase 5 reordering land at the inter-tier-gate seam,
        not here):

        1. **Grammar / JSON Schema** — already enforced sample-time by
           the outline provider's constrained-decoding payload (Subtask
           18 ``OutlineProvider._build_grammar_payload`` emits the
           per-provider grammar / response_format / format dict). The
           validator-chain entry for this layer is listed for shape
           only; for Phase 3 it's a no-op shim so the chain stays
           stable when later phases promote the grammar check off
           sample-time.
        2. **SHACL** — Phase 4 seam. The Trainforge SHACL rule runner
           (``Trainforge/rag/shacl_rule_runner.py``) is the precedent;
           Phase 4 will introduce a ``BlockSHACLValidator`` shim that
           projects the block's JSON-LD entry into a SHACL-validatable
           graph and runs ``schemas/context/courseforge_v1.shacl-rules.ttl``
           against it. For Phase 3 it's a no-op shim.
        3. **CURIE resolution** — Phase 4 seam. The Trainforge
           ``CurieAnchoringValidator`` (``lib/validators/curie_anchoring.py``)
           is the precedent. Phase 3 no-op shim.
        4. **Embedding similarity** — Phase 4 seam. Reserved for the
           round-trip semantic-similarity check between the block's
           outline content and the source-chunk text. Phase 3 no-op
           shim.
        5. **Round-trip check** — Phase 4 seam. Reserved for the
           outline → rewrite → outline projection check. Phase 3
           no-op shim.

        For Phase 3 ALL non-grammar validators are no-op shims; the
        ones that actually fire are passed in via the ``validators``
        arg (Phase 4 + Phase 5 will populate the inter-tier validators
        from this list via Subtask 50).

        Each validator must implement
        ``validate(inputs: Dict[str, Any]) -> GateResult`` (the
        :class:`MCP.hardening.validation_gates.Validator` Protocol).
        The router invokes
        ``validator.validate({"block": block, "blocks": [block]})`` so
        per-block validators see a single-block input dict and
        Block-list-aware validators see a one-element list — both
        shapes work without forcing every validator to accept both.

        Returns ``(all_passed, [GateResult per validator], touched_block)``.
        When ``fast_fail=True`` (default) the loop stops at the first
        non-pass action; when ``False`` it collects every result so
        the caller can aggregate per-validator failures.

        Phase 3.5 Subtask 17: every validator (passing OR failing)
        appends one :class:`Touch` to the returned block carrying
        ``tier=validator_tier`` (``outline_val`` for the inter-tier
        validation seam, ``rewrite_val`` for the post-rewrite seam)
        and ``purpose="validation_pass"`` / ``"validation_fail"``
        per the validator's outcome. The Touch chain is the canonical
        audit-trail seam: a postmortem reader can reconstruct WHICH
        validators ran in WHICH order, which passed, and which
        failed, without re-parsing the GateResult list.

        Uses :meth:`MCP.hardening.validation_gates.GateResult.derive_default_action`
        to interpret ``GateResult.action``: legacy validators that
        leave ``action`` unset collapse to ``"pass"`` on success /
        ``"block"`` on failure (Worker J46 / Subtask 46 contract).
        Phase-3-aware validators emit ``action="regenerate"`` /
        ``"escalate"`` / ``"block"`` directly; the router treats any
        non-``"pass"`` action as a failure that increments the
        per-validator failure-distribution counter.
        """
        results: List[Any] = []
        all_passed = True
        # Phase 3.5 Subtask 17: cumulatively-touched block. Each
        # validator appends to this so the final return carries the
        # full pass/fail audit chain.
        touched_block: Block = block
        if not validators:
            # Empty chain → pass. Lets the self-consistency loop's
            # default behaviour be "first candidate wins" when no
            # validators are wired (Wave-N pre-Phase-4 shape).
            return True, results, touched_block

        # Build the input dict once per chain — both per-block and
        # Block-list-aware validators read from the same shape.
        inputs = {"block": block, "blocks": [block]}

        # Lazy-import the canonical action helper so the router module
        # itself can be imported without pulling MCP.hardening when
        # the caller never passes any validators (e.g. Wave-N tests).
        from MCP.hardening.validation_gates import GateResult  # noqa: PLC0415

        # Phase 3.5 Subtask 26: the outline-tier validator chain emits
        # block_validation_action events with ml_features.tier="outline";
        # the rewrite-tier chain (post_rewrite_validation phase) sets
        # ``validator_tier="rewrite_val"`` and the corresponding event
        # tier becomes ``"rewrite"``. Map outline_val/rewrite_val onto
        # the canonical tier label expected by the decision-event
        # consumer.
        event_tier: Literal["outline", "rewrite"] = (
            "rewrite" if validator_tier == "rewrite_val" else "outline"
        )

        for validator in validators:
            validate_fn = getattr(validator, "validate", None)
            if not callable(validate_fn):
                # Defensive: skip non-conforming validators instead of
                # blowing up the loop. Logged so a postmortem can spot
                # the misconfiguration.
                logger.warning(
                    "_run_validator_chain: validator %r missing validate(); skipping",
                    type(validator).__name__,
                )
                continue
            try:
                gate_result = validate_fn(inputs)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "_run_validator_chain: validator %r raised: %s",
                    type(validator).__name__, exc,
                )
                all_passed = False
                if fast_fail:
                    break
                continue

            results.append(gate_result)

            action = GateResult.derive_default_action(
                gate_result.passed, gate_result.action
            )

            # Phase 3.5 Subtask 17: append a per-validator Touch to
            # the cumulative chain. Tier is the validator-tier
            # (``outline_val`` / ``rewrite_val``); purpose disambiguates
            # pass vs fail so a downstream reader can filter the chain
            # by outcome without re-deriving from the GateResult list.
            gate_id_for_touch = (
                getattr(gate_result, "gate_id", "")
                or getattr(gate_result, "validator_name", "")
                or "unknown_gate"
            )
            touch = Touch(
                model="validator",
                provider="deterministic",
                tier=validator_tier,
                timestamp=(
                    _dt.datetime.now(_dt.timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                ),
                decision_capture_id=self._build_router_capture_id(
                    block, f"{validator_tier}:{gate_id_for_touch}"
                ),
                purpose=(
                    "validation_pass" if action == "pass" else "validation_fail"
                ),
            )
            touched_block = touched_block.with_touch(touch)

            if action != "pass":
                # Subtask 47: emit one block_validation_action event per
                # validator that returned a non-pass action. The
                # decision-capture stream carries the per-block-per-
                # validator non-pass chain so a postmortem reader sees
                # WHICH gate fired WHICH action with WHICH issues.
                #
                # Phase 3.5 Subtask 26: ml_features.tier — outline vs
                # rewrite — derived from validator_tier.
                self._emit_block_validation_action(
                    block,
                    gate_id=gate_id_for_touch,
                    action=action,
                    score=getattr(gate_result, "score", None),
                    issues=list(getattr(gate_result, "issues", []) or []),
                    tier=event_tier,
                )
                all_passed = False
                if fast_fail:
                    break

        return all_passed, results, touched_block

    def _emit_self_consistency_decision(
        self,
        *,
        block: Block,
        spec: BlockProviderSpec,
        n_candidates_requested: int,
        winning_candidate_index: Optional[int],
        failed_candidate_count: int,
        validator_failure_distribution: Dict[str, int],
    ) -> None:
        """Emit the per-loop ``block_outline_call`` audit event with
        self-consistency metadata in the structured ``ml_features``
        payload (Subtask 39).

        Per Phase 3 §3.6 the ml_features payload carries:

        - ``n_candidates_requested: int`` — N as resolved by
          :meth:`_resolve_n_candidates`.
        - ``winning_candidate_index: Optional[int]`` — 0-based index of
          the candidate that passed; ``None`` when all candidates
          failed every validator.
        - ``failed_candidate_count: int`` — number of candidates that
          failed before a winner emerged (or N when all failed).
        - ``validator_failure_distribution: Dict[str, int]`` — keyed by
          validator name (``GateResult.validator_name`` or
          ``gate_id`` fallback); values are per-validator failure
          counts across all dispatched candidates.

        The rationale string mirrors the ml_features payload so a
        human reading the JSONL stream sees the same data without
        having to project the ml_features dict.

        Schema-side: the ``decision_event.schema.json`` ``ml_features``
        block does not pin ``additionalProperties: false`` (verified at
        ``schemas/events/decision_event.schema.json:181-218``), so the
        new keys validate alongside the canonical
        ``pedagogy_pattern`` / ``engagement_patterns`` / ... fields
        without touching the schema. Subtask 7 in the next batch is
        the canonical home for any schema-side enum extension.
        """
        if self._capture is None:
            return
        outcome = (
            "winner_found"
            if winning_candidate_index is not None
            else "all_candidates_failed"
        )
        rationale_parts = [
            "router_self_consistency",
            f"block_id={block.block_id}",
            f"block_type={block.block_type}",
            f"page_id={block.page_id}",
            f"provider={spec.provider}",
            f"model={spec.model}",
            f"n_candidates_requested={n_candidates_requested}",
            f"winning_candidate_index={winning_candidate_index}",
            f"failed_candidate_count={failed_candidate_count}",
            f"outcome={outcome}",
        ]
        ml_features: Dict[str, Any] = {
            "n_candidates_requested": n_candidates_requested,
            "winning_candidate_index": winning_candidate_index,
            "failed_candidate_count": failed_candidate_count,
            "validator_failure_distribution": dict(validator_failure_distribution),
        }
        try:
            self._capture.log_decision(
                decision_type="block_outline_call",
                decision=(
                    f"self_consistency:{block.block_type}:{block.block_id}"
                    f":{outcome}"
                ),
                rationale="; ".join(rationale_parts),
                ml_features=ml_features,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "router self-consistency decision-capture emit failed: %s", exc
            )

    # ------------------------------------------------------------------
    # Decision-capture helpers
    # ------------------------------------------------------------------

    def _emit_block_escalation(
        self,
        block: Block,
        *,
        marker: str,
        attempts: int,
        n_candidates: int,
    ) -> None:
        """Emit one ``block_escalation`` decision-capture event.

        Per Phase 3 Subtask 43 this is the canonical seam for the
        ``block_escalation`` decision-event class. Wired from both
        escalation paths so a postmortem reader sees a single event
        type regardless of how the block reached the rewrite tier:

        - :meth:`route_with_self_consistency` (Subtask 41 path): emits
          when the outline regen budget is exhausted mid-loop;
          ``attempts`` is the cumulative validation_attempts count and
          ``n_candidates`` is ``i+1`` (the loop iteration index that
          triggered the budget check).
        - :meth:`route` (Subtask 42 path): emits when the
          ``escalate_immediately`` short-circuit fires; ``attempts=0``
          and ``n_candidates=0`` signal the policy-skip path (no
          outline candidates were ever dispatched).

        Per Phase 3 Subtask 43 the rationale string is at least 20
        characters and interpolates the dynamic signals
        (``block_id`` / ``block_type`` / ``marker`` / ``attempts`` /
        ``n_candidates``); the structured ``ml_features`` payload
        carries the same fields for ML-trainability. Capture errors
        are swallowed so a flaky capture handle never breaks the
        dispatch path.
        """
        if self._capture is None:
            return
        rationale = (
            f"Block {block.block_id} (block_type={block.block_type}) "
            f"escalated to rewrite tier with marker={marker} after "
            f"{attempts} validation attempts across {n_candidates} "
            f"candidates. Outline tier exhausted regen budget; rewrite "
            f"tier will receive an enriched prompt with full source "
            f"chunks + objective refs to author from scratch."
        )
        ml_features: Dict[str, Any] = {
            "block_id": block.block_id,
            "block_type": block.block_type,
            "marker": marker,
            "attempts": attempts,
            "n_candidates": n_candidates,
        }
        try:
            self._capture.log_decision(
                decision_type="block_escalation",
                decision=(
                    f"escalate:{block.block_type}:{block.block_id}:{marker}"
                ),
                rationale=rationale,
                ml_features=ml_features,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "router block_escalation decision-capture emit failed: %s", exc
            )

    def _emit_block_validation_action(
        self,
        block: Block,
        *,
        gate_id: str,
        action: str,
        score: Optional[float],
        issues: List[Any],
        tier: Literal["outline", "rewrite"] = "outline",
    ) -> None:
        """Emit one ``block_validation_action`` decision-capture event
        per validator that returned a non-``"pass"`` action (Subtask 47).

        Wired from :meth:`_run_validator_chain` (Subtask 38) on every
        validator result whose router-derived action is in
        ``{"regenerate","escalate","block"}``. The pass action is a
        steady-state signal (no event needed); only deviations emit so
        a postmortem reader sees the per-block-per-validator chain of
        non-pass discriminators.

        Per Phase 3 §10 the rationale string is ≥20 characters and
        interpolates the dynamic signals (``gate_id`` / ``action`` /
        ``score`` / top-3 issue codes + messages) so the JSONL stream
        carries enough context to reconstruct WHY the validator chose
        the action without re-running the validator. The structured
        ``ml_features`` payload mirrors the rationale fields plus the
        full ``issues`` list (truncated to top-3 to bound payload size)
        for ML-trainability.

        Phase 3.5 Subtask 26: the ``tier`` parameter records WHICH tier
        the validator chain was run against. The outline-tier
        :meth:`_run_validator_chain` call sites pass ``tier="outline"``
        (the default keeps the existing call sites byte-stable); Wave B
        Subtask 13's ``_run_post_rewrite_validation`` helper will pass
        ``tier="rewrite"`` so a postmortem reader can disambiguate
        outline-tier validator failures (which trigger
        regenerate / escalate / block per the loop semantics) from
        rewrite-tier validator failures (which surface as escalation
        events with ``tier="rewrite"`` provenance). The field threads
        into ``ml_features.tier`` so ML training can stratify by tier
        without reparsing the rationale string. Schema does NOT pin
        ``additionalProperties: false`` on ``ml_features``, so adding
        the field is backward-compatible — pre-Phase-3.5 captures
        without ``tier`` continue to deserialize.

        Capture errors are swallowed so a flaky capture handle never
        breaks the dispatch path (mirrors the pattern in the sibling
        ``_emit_block_escalation`` / ``_emit_self_consistency_decision``
        helpers).

        ``issues`` is typed ``List[Any]`` rather than ``List[GateIssue]``
        because the validators land per-issue dicts in some legacy
        paths; the helper handles both shapes via duck-typed attribute
        access.
        """
        if self._capture is None:
            return
        # Top-3 issue summary for the rationale string. Issues may be
        # GateIssue dataclass instances OR plain dicts (some legacy
        # validators return dicts); both shapes resolve via getattr +
        # dict.get fallback.
        top_issues: List[Dict[str, Any]] = []
        for issue in (issues or [])[:3]:
            code = (
                getattr(issue, "code", None)
                if not isinstance(issue, dict)
                else issue.get("code")
            ) or "unknown"
            message = (
                getattr(issue, "message", None)
                if not isinstance(issue, dict)
                else issue.get("message")
            ) or ""
            severity = (
                getattr(issue, "severity", None)
                if not isinstance(issue, dict)
                else issue.get("severity")
            ) or "unknown"
            top_issues.append(
                {"code": code, "severity": severity, "message": message}
            )
        issue_summary = "; ".join(
            f"{i['code']}({i['severity']}):{i['message']}" for i in top_issues
        ) or "no_issues_reported"
        score_repr = "n/a" if score is None else f"{score:.3f}"
        rationale = (
            f"Validator {gate_id} returned action={action} for block "
            f"{block.block_id} (block_type={block.block_type}); "
            f"score={score_repr}; top_issues=[{issue_summary}]. "
            f"Router will route per Phase 4 §1.5 mapping: "
            f"regenerate→retry, escalate→rewrite tier, block→fail."
        )
        ml_features: Dict[str, Any] = {
            "block_id": block.block_id,
            "block_type": block.block_type,
            "gate_id": gate_id,
            "action": action,
            "score": score,
            "issues_top3": top_issues,
            "issues_count": len(issues or []),
            # Phase 3.5 Subtask 26: tier provenance — outline (current
            # call site) vs rewrite (Wave B Subtask 13's
            # _run_post_rewrite_validation helper).
            "tier": tier,
        }
        try:
            self._capture.log_decision(
                decision_type="block_validation_action",
                decision=(
                    f"validation_action:{block.block_type}:{block.block_id}"
                    f":{gate_id}:{action}"
                ),
                rationale=rationale,
                ml_features=ml_features,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "router block_validation_action decision-capture emit failed: %s",
                exc,
            )

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
