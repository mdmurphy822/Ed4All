"""Tests for :class:`Courseforge.router.router.CourseforgeRouter` (Phase 3 Subtask 32).

Exercises the two-pass dispatch surface that owns provider resolution
+ per-block routing + the outline/rewrite two-pass walk over a Block
list. Coverage:

- ``_resolve_spec`` priority chain: per-call kwargs → YAML policy →
  tier-default env vars → hardcoded defaults.
- Unknown provider value raises ``ValueError`` (caught by
  :class:`BlockProviderSpec.__post_init__`).
- :meth:`route` dispatches to the correct provider for the chosen tier
  and emits a ``block_outline_call`` / ``block_rewrite_call`` decision
  event with ``policy_source`` interpolated.
- :meth:`route_all` runs the full two-pass over a Block list, preserves
  ordering, excludes outline-failed blocks from the rewrite pass.
- ``escalate_immediately=True`` short-circuits the outline tier (no LLM
  call) and stamps ``escalation_marker="outline_skipped_by_policy"``.
- YAML policy override wins over env vars when both are set (binds the
  Subtask-34 contract that the Wave-N stub only honors a non-None
  ``policy.resolve(...)``).

Reuses the ``httpx.MockTransport`` fixture pattern from
``Courseforge/generators/tests/test_rewrite_provider.py`` for any test
that needs a wire-level fake; pure-router tests use injected fake
providers (no transport needed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.router import (  # noqa: E402
    BlockProviderSpec,
    CourseforgeRouter,
    _HARDCODED_DEFAULTS,
)
from blocks import Block  # noqa: E402  (Phase 2 intermediate format)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _block(
    *,
    block_type: str = "concept",
    block_id: str = "page1#concept_intro_0",
    content: Any = "hello",
    escalation_marker: Optional[str] = None,
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content=content,
        escalation_marker=escalation_marker,
    )


class _FakeProvider:
    """Stub provider exposing the two router-facing surfaces.

    ``record`` accumulates call kwargs for each method; tests inspect
    it to verify the router routed to the right tier + passed the
    right block-of-record.
    """

    def __init__(self, *, raise_on_outline: bool = False) -> None:
        self.outline_calls: List[Dict[str, Any]] = []
        self.rewrite_calls: List[Dict[str, Any]] = []
        self._raise_on_outline = raise_on_outline

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        **kwargs: Any,
    ) -> Block:
        self.outline_calls.append(
            {
                "block": block,
                "source_chunks": source_chunks,
                "objectives": objectives,
                **{k: v for k, v in kwargs.items() if k == "remediation_suffix"},
            }
        )
        if self._raise_on_outline:
            raise RuntimeError("outline forced failure")
        return block

    def generate_rewrite(
        self, block: Block, *, source_chunks: Any, objectives: Any
    ) -> Block:
        self.rewrite_calls.append(
            {
                "block": block,
                "source_chunks": source_chunks,
                "objectives": objectives,
            }
        )
        return block


class _FakeCapture:
    """Lightweight stand-in for :class:`DecisionCapture`."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class _StubPolicy:
    """Minimal policy stub exposing ``resolve(block_id, block_type, tier)``.

    Returns a pre-configured ``BlockProviderSpec`` (or ``None``) per the
    Subtask 34 contract the Wave-N stub honors.
    """

    def __init__(self, spec: Optional[BlockProviderSpec]) -> None:
        self._spec = spec
        self.calls: List[tuple] = []

    def resolve(self, block_id: str, block_type: str, tier: str) -> Any:
        self.calls.append((block_id, block_type, tier))
        return self._spec


# ---------------------------------------------------------------------------
# Spec resolution
# ---------------------------------------------------------------------------


def test_resolve_spec_per_call_kwargs_win_over_yaml_and_env(monkeypatch):
    """Per-call ``**overrides`` beat YAML policy and env vars."""
    monkeypatch.setenv("COURSEFORGE_OUTLINE_PROVIDER", "together")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_MODEL", "env-model")
    yaml_spec = BlockProviderSpec(
        block_type="concept",
        tier="outline",
        provider="anthropic",
        model="yaml-model",
    )
    r = CourseforgeRouter(policy=_StubPolicy(yaml_spec))
    spec = r._resolve_spec(
        _block(),
        "outline",
        provider="local",
        model="kwarg-model",
    )
    assert spec.provider == "local"
    assert spec.model == "kwarg-model"


def test_resolve_spec_yaml_overrides_env_var(monkeypatch):
    """YAML policy entry wins over tier-default env vars."""
    monkeypatch.setenv("COURSEFORGE_OUTLINE_PROVIDER", "together")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_MODEL", "env-model")
    yaml_spec = BlockProviderSpec(
        block_type="concept",
        tier="outline",
        provider="anthropic",
        model="yaml-model",
    )
    r = CourseforgeRouter(policy=_StubPolicy(yaml_spec))
    spec = r._resolve_spec(_block(), "outline")
    assert spec.provider == "anthropic"
    assert spec.model == "yaml-model"


def test_resolve_spec_env_var_overrides_default(monkeypatch):
    """Tier-default env var beats the hardcoded default when YAML
    policy is absent."""
    monkeypatch.setenv("COURSEFORGE_OUTLINE_PROVIDER", "together")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_MODEL", "env-model")
    r = CourseforgeRouter()  # no policy
    spec = r._resolve_spec(_block(), "outline")
    assert spec.provider == "together"
    assert spec.model == "env-model"


def test_resolve_spec_falls_through_to_hardcoded_default(monkeypatch):
    """No overrides + no YAML + no env vars → hardcoded default wins."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MODEL", raising=False)
    r = CourseforgeRouter()
    spec = r._resolve_spec(_block(block_type="objective"), "outline")
    expected = _HARDCODED_DEFAULTS[("objective", "outline")]
    assert spec.provider == expected.provider
    assert spec.model == expected.model


def test_unknown_provider_raises_value_error():
    """Per-call ``provider`` outside the allowed set surfaces via
    ``BlockProviderSpec.__post_init__``."""
    r = CourseforgeRouter()
    with pytest.raises(ValueError):
        r._resolve_spec(_block(), "outline", provider="bogus")


# ---------------------------------------------------------------------------
# Phase 3a env-var-first audit tests (Subtask 25)
# ---------------------------------------------------------------------------
#
# These tests pin the four-layer precedence chain comment block on
# ``CourseforgeRouter._resolve_spec``: per-call kwargs > YAML policy >
# env vars > hardcoded defaults. Cross-link: the inline comment block
# at ``Courseforge/router/router.py::_resolve_spec`` references
# ``test_phase3a_env_var_overrides_hardcoded_default`` and
# ``test_phase3a_yaml_wins_over_env_var`` by name; renaming or deleting
# either test must be paired with a router-side comment update.


def test_phase3a_env_var_overrides_hardcoded_default(monkeypatch):
    """Phase 3a §3.3 contract: env var beats the hardcoded default.

    Setup: no per-call kwargs, no YAML policy, both
    ``COURSEFORGE_OUTLINE_PROVIDER`` and ``COURSEFORGE_OUTLINE_MODEL``
    set. The resolved spec MUST carry the env-var values, NOT the
    hardcoded baseline. Symmetrically asserts the rewrite tier so a
    future schema change to the chain doesn't silently regress one
    tier while leaving the other untouched.
    """
    # Outline tier — env var must win over hardcoded.
    monkeypatch.setenv("COURSEFORGE_OUTLINE_PROVIDER", "together")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_MODEL", "env-outline-model")
    r = CourseforgeRouter()  # no policy, no per-call kwargs
    spec = r._resolve_spec(_block(block_type="concept"), "outline")
    # Hardcoded default for ("concept", "outline") is "local" /
    # "qwen2.5:7b-instruct-q4_K_M"; env var values must shadow both.
    hardcoded = _HARDCODED_DEFAULTS[("concept", "outline")]
    assert spec.provider != hardcoded.provider, (
        "env var should have shadowed hardcoded provider"
    )
    assert spec.model != hardcoded.model, (
        "env var should have shadowed hardcoded model"
    )
    assert spec.provider == "together"
    assert spec.model == "env-outline-model"

    # Rewrite tier — same contract under the rewrite env vars.
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_REWRITE_MODEL", "env-rewrite-model")
    spec_rw = r._resolve_spec(_block(block_type="concept"), "rewrite")
    hardcoded_rw = _HARDCODED_DEFAULTS[("concept", "rewrite")]
    assert spec_rw.provider != hardcoded_rw.provider or hardcoded_rw.provider == "local"
    assert spec_rw.model != hardcoded_rw.model
    assert spec_rw.provider == "local"
    assert spec_rw.model == "env-rewrite-model"


def test_phase3a_yaml_wins_over_env_var(monkeypatch):
    """Phase 3a §3.3 contract: YAML policy beats tier-default env vars.

    Setup: env vars AND YAML policy both set; no per-call kwargs. The
    YAML policy entry must win — operator-explicit YAML > tier-default
    env var (the operator who wrote the YAML file made an explicit
    per-block choice; the env var is a tier-default knob).

    The Phase 3a env-var-first override on the YAML LOADER (Subtask 23
    in ``Courseforge/router/policy.py::_maybe_apply_env_model_override``)
    fires only when the YAML's ``defaults[tier].model`` is the
    hardcoded sentinel literal — an operator-explicit non-sentinel
    value in YAML preserves operator intent. This test pins the
    DISPATCH-side YAML > env var contract; the loader-side
    env-var-first override is a separate orthogonal contract pinned by
    ``Courseforge/router/tests/test_policy.py`` and the inline
    ``policy.py`` doctest.
    """
    monkeypatch.setenv("COURSEFORGE_OUTLINE_PROVIDER", "together")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_MODEL", "env-model")
    yaml_spec = BlockProviderSpec(
        block_type="concept",
        tier="outline",
        provider="anthropic",
        model="explicit-yaml-model",
    )
    r = CourseforgeRouter(policy=_StubPolicy(yaml_spec))
    spec = r._resolve_spec(_block(block_type="concept"), "outline")
    # YAML wins over env var.
    assert spec.provider == "anthropic"
    assert spec.model == "explicit-yaml-model"
    # Env-var values were NOT applied.
    assert spec.provider != "together"
    assert spec.model != "env-model"


# ---------------------------------------------------------------------------
# Per-block dispatch
# ---------------------------------------------------------------------------


def test_route_dispatches_to_outline_provider_for_outline_tier(monkeypatch):
    """``tier="outline"`` routes to the outline provider's
    ``generate_outline``."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    fake = _FakeProvider()
    r = CourseforgeRouter(outline_provider=fake, rewrite_provider=_FakeProvider())
    out = r.route(_block(), tier="outline", source_chunks=[{"id": "c"}], objectives=[])
    assert len(fake.outline_calls) == 1
    assert len(fake.rewrite_calls) == 0
    assert isinstance(out, Block)


def test_route_dispatches_to_rewrite_provider_for_rewrite_tier(monkeypatch):
    """``tier="rewrite"`` routes to the rewrite provider's
    ``generate_rewrite``."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_PROVIDER", raising=False)
    fake = _FakeProvider()
    r = CourseforgeRouter(outline_provider=_FakeProvider(), rewrite_provider=fake)
    out = r.route(_block(), tier="rewrite", source_chunks=[], objectives=[])
    assert len(fake.outline_calls) == 0
    assert len(fake.rewrite_calls) == 1
    assert isinstance(out, Block)


def test_route_emits_block_outline_call_decision_event(monkeypatch):
    """Successful outline route emits one ``block_outline_call`` event
    whose rationale interpolates ``policy_source``, ``provider``, ``model``,
    and ``block_id``."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MODEL", raising=False)
    capture = _FakeCapture()
    r = CourseforgeRouter(
        outline_provider=_FakeProvider(),
        rewrite_provider=_FakeProvider(),
        capture=capture,
    )
    blk = _block(block_id="page1#concept_alpha_0")
    r.route(blk, tier="outline")
    # Filter to router-emitted events (decision_type=block_outline_call).
    events = [e for e in capture.events if e["decision_type"] == "block_outline_call"]
    assert len(events) == 1
    rationale = events[0]["rationale"]
    assert "policy_source=" in rationale
    assert "block_id=page1#concept_alpha_0" in rationale
    assert "provider=" in rationale
    assert "outcome=success" in rationale


def test_escalate_immediately_short_circuits_outline_tier(monkeypatch):
    """``spec.escalate_immediately=True`` on the outline tier skips the
    LLM call entirely, sets a Block-valid escalation marker, and
    appends a deterministic Touch with ``purpose="escalate_immediately"``
    so postmortem can distinguish a policy-skip from an outline failure.

    ``Block.__post_init__`` validates the marker against the canonical
    ``_ESCALATION_MARKERS`` set; the router uses
    ``outline_budget_exhausted`` (the closest semantic match within the
    allowed set) and surfaces the policy-skip provenance via the
    Touch's ``purpose`` field instead.
    """
    fake_outline = _FakeProvider()
    r = CourseforgeRouter(outline_provider=fake_outline)
    out = r.route(
        _block(),
        tier="outline",
        escalate_immediately=True,
    )
    # No LLM call.
    assert len(fake_outline.outline_calls) == 0
    # Marker stamped (Block-valid value).
    assert out.escalation_marker == "outline_budget_exhausted"
    # Touch appended with the policy-skip provenance in ``purpose``.
    assert len(out.touched_by) == 1
    touch = out.touched_by[0]
    assert touch.tier == "outline"
    assert touch.purpose == "escalate_immediately"


def test_per_block_type_dispatch_uses_yaml_override_when_present(monkeypatch):
    """When the YAML policy supplies a per-block-type spec, the router
    dispatches to whichever tier-provider matches that spec's tier.

    Concretely: a YAML spec for ``(concept, outline)`` that selects
    ``provider="anthropic"`` causes ``_resolve_spec`` to return that
    spec; the router still dispatches to the outline provider because
    the tier comes from the ``route(..., tier=...)`` kwarg, not from the
    spec. The ``policy_source`` audit field flips to ``yaml_policy``."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    yaml_spec = BlockProviderSpec(
        block_type="concept",
        tier="outline",
        provider="anthropic",
        model="yaml-model",
    )
    capture = _FakeCapture()
    r = CourseforgeRouter(
        policy=_StubPolicy(yaml_spec),
        outline_provider=_FakeProvider(),
        rewrite_provider=_FakeProvider(),
        capture=capture,
    )
    r.route(_block(), tier="outline")
    events = [
        e for e in capture.events if e["decision_type"] == "block_outline_call"
    ]
    assert len(events) == 1
    assert "policy_source=yaml_policy" in events[0]["rationale"]
    assert "model=yaml-model" in events[0]["rationale"]


# ---------------------------------------------------------------------------
# Two-pass walk
# ---------------------------------------------------------------------------


def test_route_all_runs_two_pass(monkeypatch):
    """``route_all`` walks every block through both tiers and preserves
    input ordering."""
    fake_outline = _FakeProvider()
    fake_rewrite = _FakeProvider()
    r = CourseforgeRouter(
        outline_provider=fake_outline,
        rewrite_provider=fake_rewrite,
    )
    b1 = _block(block_type="objective", block_id="page1#objective_a_0")
    b2 = _block(block_type="concept", block_id="page1#concept_b_0")
    b3 = _block(block_type="example", block_id="page1#example_c_0")
    out = r.route_all([b1, b2, b3])
    # All three blocks reach both tiers.
    assert len(fake_outline.outline_calls) == 3
    assert len(fake_rewrite.rewrite_calls) == 3
    # Ordering preserved.
    assert [b.block_id for b in out] == [
        "page1#objective_a_0",
        "page1#concept_b_0",
        "page1#example_c_0",
    ]


def test_route_all_excludes_failed_outline_from_rewrite(monkeypatch):
    """A block whose outline-tier dispatch raises is marked
    ``escalation_marker="outline_budget_exhausted"`` and is NOT
    passed to the rewrite tier; the failed block still appears in the
    returned list at its original index."""
    fake_outline = _FakeProvider(raise_on_outline=True)
    fake_rewrite = _FakeProvider()
    r = CourseforgeRouter(
        outline_provider=fake_outline,
        rewrite_provider=fake_rewrite,
    )
    b1 = _block(block_type="concept", block_id="page1#concept_failed_0")
    out = r.route_all([b1])
    # Outline was attempted.
    assert len(fake_outline.outline_calls) == 1
    # Rewrite was NOT attempted on the failed block.
    assert len(fake_rewrite.rewrite_calls) == 0
    # Block is returned with outline-failed marker.
    assert len(out) == 1
    assert out[0].escalation_marker == "outline_budget_exhausted"


def test_route_all_preserves_ordering_with_mixed_outcomes(monkeypatch):
    """A list with one failing and two passing outline dispatches
    returns blocks in input order; the failed block has the
    ``outline_budget_exhausted`` marker, the passing blocks reach
    the rewrite tier."""

    class _MixedOutline:
        """Outline provider that fails for ``block_id`` containing
        ``failure``; passes everything else through."""

        def __init__(self) -> None:
            self.calls: List[str] = []

        def generate_outline(
            self,
            block: Block,
            *,
            source_chunks: Any,
            objectives: Any,
            **kwargs: Any,
        ) -> Block:
            self.calls.append(block.block_id)
            if "failure" in block.block_id:
                raise RuntimeError("simulated outline failure")
            return block

    mixed = _MixedOutline()
    fake_rewrite = _FakeProvider()
    r = CourseforgeRouter(
        outline_provider=mixed,
        rewrite_provider=fake_rewrite,
    )
    b1 = _block(block_type="concept", block_id="page1#concept_a_0")
    b2 = _block(block_type="concept", block_id="page1#concept_failure_1")
    b3 = _block(block_type="concept", block_id="page1#concept_c_2")
    out = r.route_all([b1, b2, b3])
    assert [b.block_id for b in out] == [
        "page1#concept_a_0",
        "page1#concept_failure_1",
        "page1#concept_c_2",
    ]
    # Only the two surviving blocks hit rewrite.
    assert len(fake_rewrite.rewrite_calls) == 2
    rewrite_ids = [c["block"].block_id for c in fake_rewrite.rewrite_calls]
    assert "page1#concept_failure_1" not in rewrite_ids
    # Failed block carries the marker.
    failed = [b for b in out if "failure" in b.block_id][0]
    assert failed.escalation_marker == "outline_budget_exhausted"


# ---------------------------------------------------------------------------
# Provider lazy-instantiation
# ---------------------------------------------------------------------------


def test_provider_override_short_circuits_lazy_construction(monkeypatch):
    """When ``outline_provider`` / ``rewrite_provider`` are injected at
    construction time, ``_get_outline_provider`` / ``_get_rewrite_provider``
    return them directly without invoking the Provider constructors —
    important for tests that don't want to wire up the full
    OpenAICompatibleClient."""
    fake_outline = _FakeProvider()
    fake_rewrite = _FakeProvider()
    r = CourseforgeRouter(
        outline_provider=fake_outline,
        rewrite_provider=fake_rewrite,
    )
    spec_outline = BlockProviderSpec(
        block_type="concept",
        tier="outline",
        provider="local",
        model="bogus-not-imported",
    )
    spec_rewrite = BlockProviderSpec(
        block_type="concept",
        tier="rewrite",
        provider="local",
        model="bogus-not-imported",
    )
    assert r._get_outline_provider(spec_outline) is fake_outline
    assert r._get_rewrite_provider(spec_rewrite) is fake_rewrite
