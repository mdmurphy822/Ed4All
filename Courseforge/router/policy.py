#!/usr/bin/env python3
"""Block-routing policy loader for the Courseforge two-pass router.

Loads the operator-tunable ``Courseforge/config/block_routing.yaml``
file, validates it against
``schemas/courseforge/block_routing.schema.json``, and exposes a frozen
:class:`BlockRoutingPolicy` whose :meth:`resolve` method is consumed by
:meth:`Courseforge.router.router.CourseforgeRouter._resolve_spec` as
the second-priority resolver in the Phase 3 §3.3 dispatch chain (after
per-call kwargs, before tier-default env vars and the hardcoded
fallback table).

Resolution priority inside :meth:`BlockRoutingPolicy.resolve`:

1. ``overrides[]`` — first per-block_id glob match (Python ``fnmatch``).
2. ``blocks[block_type][tier]`` — per-block_type entry.
3. ``defaults[tier]`` — tier-level fallback.
4. ``None`` — caller falls through to env-var / hardcoded chain.

Path resolution inside :func:`load_block_routing_policy`:

1. ``path`` argument when provided.
2. ``COURSEFORGE_BLOCK_ROUTING_PATH`` env var.
3. :data:`_DEFAULT_POLICY_PATH` (``Courseforge/config/block_routing.yaml``).

When the resolved path doesn't exist on disk the loader returns an
empty :class:`BlockRoutingPolicy` and logs at INFO level — the policy
file is intentionally optional so a clean checkout (no operator
overrides) walks straight through to env vars + hardcoded defaults.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# The router lives next to this module; reuse its BlockProviderSpec so
# the loader emits the exact same frozen dataclass the dispatch path
# already consumes.
from Courseforge.router.router import BlockProviderSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Path is resolved relative to the repo root (the project's CWD when
# the canonical ``ed4all`` CLI runs). Tests that need a different path
# pass it explicitly to :func:`load_block_routing_policy`.
_DEFAULT_POLICY_PATH: Path = Path("Courseforge/config/block_routing.yaml")

_ENV_POLICY_PATH = "COURSEFORGE_BLOCK_ROUTING_PATH"

# Schema lives at the repo-root-relative path; resolved at load time
# via the module's parents chain so import order doesn't depend on CWD.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_SCHEMA_PATH: Path = (
    _REPO_ROOT / "schemas" / "courseforge" / "block_routing.schema.json"
)


# ---------------------------------------------------------------------------
# BlockRoutingPolicy dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockRoutingPolicy:
    """Frozen, immutable view of a loaded ``block_routing.yaml`` file.

    Frozen because the router caches the policy on construction and a
    silent mutation downstream would route subsequent dispatches to a
    different provider with no audit trail. ``defaults`` and ``blocks``
    contain :class:`BlockProviderSpec` instances ready to hand back
    from :meth:`resolve`; ``overrides`` retains the raw dict shape
    (block_id glob + spec dicts) so :meth:`resolve` can re-evaluate
    glob matches per call.

    Fields:

    - ``defaults`` — keyed by tier (``"outline"`` / ``"rewrite"``);
      each value is a fully constructed :class:`BlockProviderSpec`
      with ``block_type=""`` (the resolver fills in the per-block
      block_type at lookup time so the cached defaults are reusable
      across block types).
    - ``blocks`` — keyed by block_type then tier; each leaf is a
      :class:`BlockProviderSpec` already pinned to that block_type.
    - ``overrides`` — list of dicts, each carrying ``block_id`` plus
      optional ``outline`` / ``rewrite`` spec dicts and an optional
      ``escalate_immediately`` flag.
    - ``escalate_immediately_by_block_type`` — keyed by block_type;
      mirrors ``blocks[type].escalate_immediately`` for a fast lookup
      from :meth:`Courseforge.router.router.CourseforgeRouter._resolve_spec`.
    - ``n_candidates_by_block_type`` / ``regen_budget_by_block_type``
      — same fast-lookup pattern for the self-consistency knobs
      consumed by Subtasks 37 / 41.
    - ``regen_budget_rewrite_by_block_type`` (Phase 3.5 Subtask 21):
      per-block-type override for the rewrite-tier regen budget
      consumed by
      :meth:`Courseforge.router.router.CourseforgeRouter._resolve_rewrite_regen_budget`.
      Symmetric to ``regen_budget_by_block_type`` (which applies to
      the outline tier).
    """

    defaults: Dict[str, BlockProviderSpec] = field(default_factory=dict)
    blocks: Dict[str, Dict[str, BlockProviderSpec]] = field(default_factory=dict)
    overrides: List[Dict[str, Any]] = field(default_factory=list)
    escalate_immediately_by_block_type: Dict[str, bool] = field(
        default_factory=dict
    )
    n_candidates_by_block_type: Dict[str, int] = field(default_factory=dict)
    regen_budget_by_block_type: Dict[str, int] = field(default_factory=dict)
    # Phase 3.5 Subtask 21: rewrite-tier regen budget map.
    regen_budget_rewrite_by_block_type: Dict[str, int] = field(
        default_factory=dict
    )
    # Operator-supplied capability-tier table. Keys are operator-chosen
    # labels (e.g. ``"small"`` / ``"medium"`` / ``"large"``); values are
    # raw spec dicts (NOT projected ``BlockProviderSpec`` instances)
    # because the projector restamps each lookup with the caller's
    # ``block_type`` + ``tier`` at resolution time. Empty dict when the
    # YAML carries no ``capability_tiers`` block (legacy mode).
    capability_tiers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Per-(block_type, tier) capability-tier chains: the resolved list
    # of tier-name strings for blocks that carry a ``capability_tier``
    # reference. Empty when the YAML carries no chains. Used by the
    # router's :meth:`CourseforgeRouter._resolve_capability_tier_chain`
    # to walk a cascading-regen chain. Keys: ``(block_type, tier)``;
    # values: ordered list of tier-name strings (length ≥ 1).
    capability_tier_chain_by_block_type: Dict[
        tuple, List[str]
    ] = field(default_factory=dict)
    # Same shape for the tier-default chains under ``defaults[tier]``.
    # Keys: ``"outline"`` / ``"rewrite"``; values: ordered list of
    # tier-name strings.
    capability_tier_chain_by_default_tier: Dict[
        str, List[str]
    ] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True when the policy carries no defaults / blocks /
        overrides — i.e. the caller should fall straight through to
        the next resolver layer."""
        return (
            not self.defaults
            and not self.blocks
            and not self.overrides
            and not self.capability_tiers
        )

    def resolve(
        self,
        block_id: str,
        block_type: str,
        tier: str,
    ) -> Optional[BlockProviderSpec]:
        """Resolve a (block_id, block_type, tier) triple to a spec.

        Walks the four-step chain documented in the module docstring.
        Returns ``None`` when no match exists at any layer; the
        :meth:`Courseforge.router.router.CourseforgeRouter._resolve_spec`
        caller treats ``None`` as a fall-through signal.
        """
        # 1. overrides: first glob match wins
        for entry in self.overrides:
            pattern = entry.get("block_id", "")
            if not pattern:
                continue
            if not match_block_id_glob(block_id, pattern):
                continue
            spec_dict = entry.get(tier)
            if isinstance(spec_dict, dict):
                return _spec_from_dict(spec_dict, block_type=block_type, tier=tier)

        # 2. per-block_type entry
        per_type = self.blocks.get(block_type)
        if per_type is not None:
            spec = per_type.get(tier)
            if spec is not None:
                return spec

        # 3. tier-level default — re-stamp with the caller's block_type
        # because the cached default carries an empty block_type sentinel.
        default = self.defaults.get(tier)
        if default is not None:
            return dataclasses.replace(default, block_type=block_type)

        # 4. fall-through signal
        return None


# ---------------------------------------------------------------------------
# Glob helper
# ---------------------------------------------------------------------------


def match_block_id_glob(block_id: str, pattern: str) -> bool:
    """Match a block_id against a Python ``fnmatch``-style glob.

    Used by :meth:`BlockRoutingPolicy.resolve` for the per-block_id
    overrides surface. ``fnmatch`` is the right tool: it's stdlib,
    case-sensitive (block IDs in this project are lowercase + hyphens),
    and supports ``*`` / ``?`` / ``[seq]`` without dragging regex into
    the operator-facing YAML.
    """
    if not pattern:
        return False
    return fnmatch.fnmatchcase(block_id, pattern)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_block_routing_policy(
    path: Optional[Path] = None,
) -> BlockRoutingPolicy:
    """Load + validate ``block_routing.yaml`` and return a frozen policy.

    Path resolution: ``path`` arg → ``COURSEFORGE_BLOCK_ROUTING_PATH``
    env var → :data:`_DEFAULT_POLICY_PATH`. Missing file is non-fatal
    (returns an empty :class:`BlockRoutingPolicy`); malformed YAML or
    schema-violating content fails closed via the underlying YAML /
    jsonschema exceptions.
    """
    resolved = _resolve_policy_path(path)

    if not resolved.exists():
        logger.info(
            "block_routing policy file not found at %s; "
            "router will fall through to env vars + hardcoded defaults.",
            resolved,
        )
        return BlockRoutingPolicy()

    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        # Empty file is treated the same as a missing file.
        logger.info(
            "block_routing policy file %s is empty; "
            "router will fall through to env vars + hardcoded defaults.",
            resolved,
        )
        return BlockRoutingPolicy()

    if not isinstance(raw, dict):
        raise ValueError(
            f"block_routing policy at {resolved} must deserialize to a "
            f"mapping at the top level; got {type(raw).__name__}."
        )

    _validate_against_schema(raw, source_path=resolved)

    return _policy_from_dict(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_policy_path(path: Optional[Path]) -> Path:
    if path is not None:
        return Path(path)
    env_value = os.environ.get(_ENV_POLICY_PATH)
    if env_value:
        return Path(env_value)
    return _DEFAULT_POLICY_PATH


def _validate_against_schema(payload: Dict[str, Any], *, source_path: Path) -> None:
    """Validate the loaded YAML against the canonical JSON Schema.

    Imported lazily so the loader's import surface stays minimal when
    callers only want the dataclass shape (e.g. for type hints).
    """
    import jsonschema  # local import: not a hot-path dep

    if not _SCHEMA_PATH.exists():
        # Defensive — the schema is checked into the repo, so this
        # should never fire in practice. Logged at WARNING because a
        # missing schema means the operator gets no validation.
        logger.warning(
            "block_routing schema not found at %s; loading %s without validation.",
            _SCHEMA_PATH,
            source_path,
        )
        return

    with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)

    jsonschema.Draft202012Validator(schema).validate(payload)


def _policy_from_dict(raw: Dict[str, Any]) -> BlockRoutingPolicy:
    """Project a validated YAML payload into a frozen
    :class:`BlockRoutingPolicy`."""
    # Operator-supplied capability-tier table (top-level
    # ``capability_tiers`` block). Stored as raw dicts because the
    # projector restamps each lookup with the caller's block_type +
    # tier at resolution time. Empty dict when the YAML carries no
    # ``capability_tiers`` block.
    capability_tiers: Dict[str, Dict[str, Any]] = {}
    raw_capability_tiers = raw.get("capability_tiers") or {}
    if isinstance(raw_capability_tiers, dict):
        for tier_name, tier_spec in raw_capability_tiers.items():
            if not isinstance(tier_spec, dict):
                continue
            capability_tiers[tier_name] = dict(tier_spec)

    # Per-tier-default capability_tier chain map, populated below as
    # we walk ``defaults``.
    capability_tier_chain_by_default_tier: Dict[str, List[str]] = {}
    # Per-block-type capability_tier chain map, populated below as
    # we walk ``blocks``.
    capability_tier_chain_by_block_type: Dict[tuple, List[str]] = {}

    defaults: Dict[str, BlockProviderSpec] = {}
    for tier, spec_dict in (raw.get("defaults") or {}).items():
        if not isinstance(spec_dict, dict):
            continue
        # Phase 3a env-var-first contract (Subtask 23): when the YAML's
        # ``defaults[tier].model`` is the same hardcoded sentinel literal
        # the loader / hardcoded-defaults table ships with AND the
        # corresponding tier-default env var
        # (``COURSEFORGE_OUTLINE_MODEL`` / ``COURSEFORGE_REWRITE_MODEL``)
        # is set non-empty, the env var overrides the YAML value. Per-
        # block_type overrides in YAML still win over the env var
        # (operator-explicit > tier-default), which is enforced at the
        # ``BlockRoutingPolicy.resolve`` layer because per-block_type
        # entries are resolved before tier defaults.
        spec_dict = _maybe_apply_env_model_override(spec_dict, tier=tier)
        # Capture the capability-tier chain for the default tier when
        # present. The projector below honours both single-string and
        # list forms; we record the list form here so the router can
        # reconstruct the cascading chain.
        chain = _normalise_capability_tier_chain(
            spec_dict.get("capability_tier")
        )
        if chain:
            capability_tier_chain_by_default_tier[tier] = chain
        # block_type left empty here; resolve() restamps with the
        # caller's block_type before handing the spec back.
        defaults[tier] = _project_capability_aware_spec(
            spec_dict,
            block_type="",
            tier=tier,
            capability_tiers=capability_tiers,
        )

    blocks: Dict[str, Dict[str, BlockProviderSpec]] = {}
    escalate_map: Dict[str, bool] = {}
    n_candidates_map: Dict[str, int] = {}
    regen_budget_map: Dict[str, int] = {}
    # Phase 3.5 Subtask 21: rewrite-tier regen budget map.
    regen_budget_rewrite_map: Dict[str, int] = {}
    for block_type, entry in (raw.get("blocks") or {}).items():
        if not isinstance(entry, dict):
            continue
        per_tier: Dict[str, BlockProviderSpec] = {}
        for tier in ("outline", "rewrite"):
            spec_dict = entry.get(tier)
            if isinstance(spec_dict, dict):
                chain = _normalise_capability_tier_chain(
                    spec_dict.get("capability_tier")
                )
                if chain:
                    capability_tier_chain_by_block_type[
                        (block_type, tier)
                    ] = chain
                per_tier[tier] = _project_capability_aware_spec(
                    spec_dict,
                    block_type=block_type,
                    tier=tier,
                    capability_tiers=capability_tiers,
                )
        if per_tier:
            blocks[block_type] = per_tier
        if entry.get("escalate_immediately") is True:
            escalate_map[block_type] = True
        n_candidates = entry.get("n_candidates")
        if isinstance(n_candidates, int):
            n_candidates_map[block_type] = n_candidates
        regen_budget = entry.get("regen_budget")
        if isinstance(regen_budget, int):
            regen_budget_map[block_type] = regen_budget
        # Phase 3.5 Subtask 21: rewrite-tier per-block-type budget.
        regen_budget_rewrite = entry.get("regen_budget_rewrite")
        if isinstance(regen_budget_rewrite, int):
            regen_budget_rewrite_map[block_type] = regen_budget_rewrite

    overrides: List[Dict[str, Any]] = list(raw.get("overrides") or [])

    return BlockRoutingPolicy(
        defaults=defaults,
        blocks=blocks,
        overrides=overrides,
        escalate_immediately_by_block_type=escalate_map,
        n_candidates_by_block_type=n_candidates_map,
        regen_budget_by_block_type=regen_budget_map,
        regen_budget_rewrite_by_block_type=regen_budget_rewrite_map,
        capability_tiers=capability_tiers,
        capability_tier_chain_by_block_type=capability_tier_chain_by_block_type,
        capability_tier_chain_by_default_tier=capability_tier_chain_by_default_tier,
    )


def _normalise_capability_tier_chain(value: Any) -> List[str]:
    """Normalise a ``capability_tier`` value into an ordered chain list.

    Accepts the schema's ``oneOf`` (string OR non-empty list of
    strings). Returns ``[name]`` for the string form, the list itself
    for the list form, and ``[]`` when the value is absent or
    malformed (the projector falls back to the legacy spec path).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        out: List[str] = []
        for entry in value:
            if isinstance(entry, str) and entry:
                out.append(entry)
        return out
    return []


def _project_capability_aware_spec(
    spec_dict: Dict[str, Any],
    *,
    block_type: str,
    tier: str,
    capability_tiers: Dict[str, Dict[str, Any]],
) -> BlockProviderSpec:
    """Project a YAML spec dict into a :class:`BlockProviderSpec` honouring
    the optional ``capability_tier`` reference.

    When ``spec_dict["capability_tier"]`` is set:

    1. Resolve the FIRST tier name in the chain via
       ``capability_tiers[<name>]``. The first tier is the
       starting-point spec the dispatch path uses; the router walks
       the rest of the chain via
       :meth:`CourseforgeRouter._resolve_capability_tier_chain` when
       cascading-regen escalates.
    2. Fail-loud (``ValueError``) when the named tier is absent — an
       operator misconfig should not silently fall through to the
       hardcoded defaults table.
    3. Project the resolved tier dict into a :class:`BlockProviderSpec`.
    4. Merge sibling fields in the original spec_dict (everything
       except ``capability_tier``) over the resolved spec —
       sibling-explicit wins over tier-default within the same YAML
       entry.
    5. Stamp ``capability_tier_name`` with the resolved tier name so
       the audit trail records WHICH tier the dispatch fired against.

    When ``capability_tier`` is absent, falls through to the legacy
    :func:`_spec_from_dict` projector (back-compat).
    """
    chain = _normalise_capability_tier_chain(spec_dict.get("capability_tier"))
    if not chain:
        return _spec_from_dict(spec_dict, block_type=block_type, tier=tier)

    first_tier_name = chain[0]
    tier_spec = capability_tiers.get(first_tier_name)
    if tier_spec is None:
        raise ValueError(
            f"capability_tier {first_tier_name!r} referenced by "
            f"(block_type={block_type or '<default>'}, tier={tier!r}) "
            f"is not declared under top-level capability_tiers; "
            f"available={sorted(capability_tiers.keys())}"
        )
    # Build the resolved spec from the tier dict, then overlay any
    # sibling fields the operator authored on the same YAML entry
    # (sibling-explicit > tier-default within the same Spec). Drop
    # ``capability_tier`` itself so it doesn't leak into the kwargs.
    merged: Dict[str, Any] = dict(tier_spec)
    for key, value in spec_dict.items():
        if key == "capability_tier":
            continue
        merged[key] = value
    spec = _spec_from_dict(merged, block_type=block_type, tier=tier)
    # Stamp the operator-chosen label so the audit trail can introspect
    # which tier the dispatch path resolved.
    return dataclasses.replace(spec, capability_tier_name=first_tier_name)


# ---------------------------------------------------------------------------
# Phase 3a env-var-first override (Subtask 23)
# ---------------------------------------------------------------------------

# Sentinel YAML model literals that the shipped
# ``Courseforge/config/block_routing.yaml`` carries under
# ``defaults.outline.model`` / ``defaults.rewrite.model``. These mirror
# the hardcoded fallback table at
# ``Courseforge/router/router.py::_DEFAULT_OUTLINE_MODEL`` /
# ``_DEFAULT_REWRITE_MODEL_ANTHROPIC``. Keeping the strings in sync is
# checked by the Subtask-23 acceptance test
# (Courseforge/router/tests/test_router.py::test_phase3a_*).
_YAML_SENTINEL_OUTLINE_MODEL = "qwen2.5:7b-instruct-q4_K_M"
_YAML_SENTINEL_REWRITE_MODEL = "claude-sonnet-4-6"

_ENV_TIER_MODEL_VAR: Dict[str, str] = {
    "outline": "COURSEFORGE_OUTLINE_MODEL",
    "rewrite": "COURSEFORGE_REWRITE_MODEL",
}

_TIER_SENTINELS: Dict[str, str] = {
    "outline": _YAML_SENTINEL_OUTLINE_MODEL,
    "rewrite": _YAML_SENTINEL_REWRITE_MODEL,
}


def _maybe_apply_env_model_override(
    spec_dict: Dict[str, Any],
    *,
    tier: str,
) -> Dict[str, Any]:
    """Apply env-var-first override to a tier-default spec dict.

    Phase 3a §3.3 contract (Subtask 23): an operator who has set
    ``COURSEFORGE_OUTLINE_MODEL`` / ``COURSEFORGE_REWRITE_MODEL``
    expects the env var to win over the shipped YAML default. The
    YAML's ``defaults.{tier}.model`` is intentionally pinned to the
    hardcoded sentinel literal (so a clean checkout walks straight
    through to the hardcoded-defaults table); when an operator sets
    the tier-default env var, that intent should beat the sentinel.

    Override only fires when:

    1. ``tier`` is one of ``{"outline", "rewrite"}`` (else no-op).
    2. The YAML model field equals the sentinel literal (so an
       operator who explicitly pinned a non-sentinel model in YAML
       keeps that pin — operator-explicit > tier-default env var).
    3. The corresponding env var is set and non-empty.

    Returns a copy of ``spec_dict`` with ``model`` overridden when the
    above conditions hold; returns ``spec_dict`` unchanged otherwise
    (no allocation).

    Per-block_type overrides in ``blocks[block_type][tier]`` are NOT
    routed through this helper, so an operator explicitly pinning a
    per-block_type model in YAML continues to win over the env var.

    Audit trail: emits an ``INFO`` log line via the module logger
    when the override fires. Decision-capture has no surface here
    because the loader is module-level (no ``capture`` instance in
    scope); ``model_resolution_env_override`` is not in
    ``schemas/events/decision_event.schema.json``'s enum, so a JSONL
    audit event would fail strict validation. The structured log
    line carries ``tier``, ``yaml_model``, ``env_var``, and
    ``env_value`` fields so a postmortem reader can reconstruct the
    override without parsing the YAML or env state at the time.
    """
    if tier not in _ENV_TIER_MODEL_VAR:
        return spec_dict
    yaml_model = spec_dict.get("model")
    if yaml_model != _TIER_SENTINELS[tier]:
        return spec_dict
    env_var = _ENV_TIER_MODEL_VAR[tier]
    env_value = os.environ.get(env_var)
    if not env_value:
        return spec_dict
    env_value = env_value.strip()
    if not env_value:
        return spec_dict
    logger.info(
        "block_routing env-var-first override fired: "
        "tier=%s yaml_model=%r env_var=%s env_value=%r",
        tier, yaml_model, env_var, env_value,
    )
    overridden = dict(spec_dict)
    overridden["model"] = env_value
    return overridden


def _spec_from_dict(
    spec_dict: Dict[str, Any],
    *,
    block_type: str,
    tier: str,
) -> BlockProviderSpec:
    """Build a :class:`BlockProviderSpec` from a YAML dict, applying
    the same conventional defaults the hardcoded fallback table uses
    when fields are unspecified.

    The schema constrains ``provider`` and ``model`` shape so missing
    values here mean the operator left them out intentionally; we
    fall back to per-tier conventions (outline = local 7B Qwen at
    temperature 0.0 / 1200 tokens; rewrite = 0.4 / 2400 tokens).
    """
    provider = spec_dict.get("provider", "local")
    model = spec_dict.get(
        "model",
        "qwen2.5:7b-instruct-q4_K_M"
        if tier == "outline"
        else "qwen2.5:14b-instruct-q4_K_M",
    )
    temperature = spec_dict.get(
        "temperature", 0.0 if tier == "outline" else 0.4
    )
    max_tokens = spec_dict.get(
        "max_tokens", 1200 if tier == "outline" else 2400
    )
    # Optional per-tier-spec regen budget (consumed by the router's
    # cascading-regen helper). ``None`` when the operator didn't
    # specify one — the router falls back to
    # ``resolved_budget // len(tier_chain)``.
    regen_budget_value = spec_dict.get("regen_budget")
    regen_budget_int: Optional[int] = None
    if isinstance(regen_budget_value, int) and regen_budget_value > 0:
        regen_budget_int = int(regen_budget_value)
    return BlockProviderSpec(
        block_type=block_type or "_default",
        tier=tier,
        provider=provider,
        model=model,
        base_url=spec_dict.get("base_url"),
        api_key_env=spec_dict.get("api_key_env"),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        regen_budget=regen_budget_int,
    )


__all__ = [
    "BlockRoutingPolicy",
    "load_block_routing_policy",
    "match_block_id_glob",
]
