"""Regression tests for the Qwen-7B audit plan §2 + §4 capability_tier
abstraction (provider-agnostic cascading-regen-at-higher-capability).

Coverage:

- §2.1 schema admits the new ``capability_tiers`` top-level table
  + per-Spec ``capability_tier`` field; legacy YAML without these
  fields validates and runs identically (§2.5 back-compat).
- §2.2 policy projector resolves a ``capability_tier: <name>``
  reference through the operator-supplied table, fails loud on
  missing tier names, merges sibling fields over the resolved spec.
- §4.1 / §4.2 router cascading-regen escalates through the chain on
  per-tier sub-budget exhaustion and emits one
  ``block_capability_escalation`` decision-capture event per
  transition.
- §4 chain exhaustion stamps the canonical
  ``outline_budget_exhausted`` / ``validator_consensus_fail`` markers
  rather than continuing to escalate.
- §3-§4 recorded transcripts: the §3.1-3.4 outline-tier surfaces
  recorded at ``runtime/qwen_test/surfaces.json`` are now expected to
  pass on attempt 1 with the new prompt + the capability-tier
  starting spec; the §3.5 token-stuffing rewrite-tier surface is
  rejected by the contextual gate.

Imports use the same project-root sys.path pattern as the existing
:mod:`Courseforge.router.tests.test_router` suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jsonschema  # noqa: E402

from Courseforge.router.policy import (  # noqa: E402
    BlockRoutingPolicy,
    _ENV_POLICY_PATH,
    _project_capability_aware_spec,
    load_block_routing_policy,
)
from Courseforge.router.router import (  # noqa: E402
    BlockProviderSpec,
    CourseforgeRouter,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
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


class _FakeCapture:
    """Minimal DecisionCapture-shaped fake. Records every event so the
    capability-escalation regression tests can introspect ml_features."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        ml_features: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                "ml_features": ml_features or {},
                **kwargs,
            }
        )


class _CountingProvider:
    """Outline / Rewrite stand-in. Counts dispatches and tags each
    candidate's content with the tier_name it ran under so the test
    can confirm the chain walked."""

    def __init__(self, *, fail_first_n: int = 0, tier_label: str = "") -> None:
        self.calls: List[Dict[str, Any]] = []
        self._fail_first_n = fail_first_n
        self._tier_label = tier_label

    def generate_outline(self, block: Block, **kwargs: Any) -> Block:
        self.calls.append({"block": block, **kwargs})
        return block

    def generate_rewrite(self, block: Block, **kwargs: Any) -> Block:
        self.calls.append({"block": block, **kwargs})
        return block


# ---------------------------------------------------------------------------
# §2.1 / §2.5 schema + back-compat
# ---------------------------------------------------------------------------


def test_legacy_yaml_without_capability_tiers_loads_unchanged(tmp_path):
    """A YAML file authored before the capability_tier feature MUST
    continue to validate and project into a BlockRoutingPolicy with
    legacy semantics."""
    yaml_body = """
version: 1
defaults:
  outline:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
  rewrite:
    provider: anthropic
    model: claude-sonnet-4-6
blocks:
  concept:
    rewrite:
      provider: anthropic
      model: claude-sonnet-4-6
"""
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml_body)
    policy = load_block_routing_policy(p)

    # No capability_tier surfaces fired.
    assert policy.capability_tiers == {}
    assert policy.capability_tier_chain_by_block_type == {}
    assert policy.capability_tier_chain_by_default_tier == {}
    # Defaults + blocks projected normally.
    assert policy.defaults["outline"].provider == "local"
    assert policy.defaults["rewrite"].model == "claude-sonnet-4-6"
    assert "concept" in policy.blocks
    # No tier_name stamped on legacy specs.
    assert policy.defaults["outline"].capability_tier_name is None


def test_capability_tiers_top_level_schema_admits_string_capability_tier(tmp_path):
    """Schema accepts the new top-level capability_tiers + per-Spec
    string-form capability_tier reference (§2.1)."""
    yaml_body = """
version: 1
capability_tiers:
  small:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
defaults:
  outline:
    capability_tier: small
"""
    p = tmp_path / "small.yaml"
    p.write_text(yaml_body)
    policy = load_block_routing_policy(p)
    assert "small" in policy.capability_tiers
    assert policy.capability_tier_chain_by_default_tier == {"outline": ["small"]}
    assert policy.defaults["outline"].provider == "local"
    assert policy.defaults["outline"].capability_tier_name == "small"


def test_capability_tier_list_form_admits_chain(tmp_path):
    """Schema accepts the list form on per-block-type entries; the
    projector records the chain on the policy for the router."""
    yaml_body = """
version: 1
capability_tiers:
  small:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
  medium:
    provider: local
    model: qwen2.5:14b-instruct-q4_K_M
blocks:
  assessment_item:
    rewrite:
      capability_tier: [small, medium]
"""
    p = tmp_path / "chain.yaml"
    p.write_text(yaml_body)
    policy = load_block_routing_policy(p)
    assert (
        policy.capability_tier_chain_by_block_type[
            ("assessment_item", "rewrite")
        ]
        == ["small", "medium"]
    )
    assert (
        policy.blocks["assessment_item"]["rewrite"].capability_tier_name
        == "small"
    )


def test_capability_tier_referencing_undeclared_name_fails_loud(tmp_path):
    """Referencing a tier name that isn't declared under the top-level
    capability_tiers table is operator misconfig — must raise rather
    than silently fall through (§2.2 fail-loud contract)."""
    yaml_body = """
version: 1
capability_tiers:
  small:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
blocks:
  assessment_item:
    rewrite:
      capability_tier: medium
"""
    p = tmp_path / "missing.yaml"
    p.write_text(yaml_body)
    with pytest.raises(ValueError, match="medium"):
        load_block_routing_policy(p)


def test_capability_tier_unknown_sibling_field_rejected(tmp_path):
    """The Spec definition is `additionalProperties: false`; an
    unknown sibling field next to capability_tier MUST fail closed
    against the schema."""
    yaml_body = """
version: 1
capability_tiers:
  small:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
defaults:
  outline:
    capability_tier: small
    bogus_field: 42
"""
    p = tmp_path / "bogus.yaml"
    p.write_text(yaml_body)
    with pytest.raises(jsonschema.ValidationError):
        load_block_routing_policy(p)


def test_capability_tier_sibling_field_overlays_resolved_tier(tmp_path):
    """When the Spec carries both capability_tier AND an explicit
    sibling field, the sibling-explicit field wins (§2.2 merge
    contract)."""
    yaml_body = """
version: 1
capability_tiers:
  medium:
    provider: local
    model: qwen2.5:14b-instruct-q4_K_M
    temperature: 0.4
blocks:
  concept:
    rewrite:
      capability_tier: medium
      temperature: 0.7
"""
    p = tmp_path / "overlay.yaml"
    p.write_text(yaml_body)
    policy = load_block_routing_policy(p)
    spec = policy.blocks["concept"]["rewrite"]
    # Sibling temperature wins over the medium tier's 0.4.
    assert spec.temperature == 0.7
    # Tier-supplied provider/model still pass through.
    assert spec.provider == "local"
    assert spec.model == "qwen2.5:14b-instruct-q4_K_M"
    # Audit trail records the tier_name regardless.
    assert spec.capability_tier_name == "medium"


# ---------------------------------------------------------------------------
# §4 router cascading-regen
# ---------------------------------------------------------------------------


def test_capability_chain_escalates_after_sub_budget_exhaustion():
    """Two-tier outline-tier chain ``[small, medium]``: when the
    per-tier sub-budget exhausts at the small tier, the next dispatch
    flips to the medium tier and emits one
    ``block_capability_escalation`` decision-capture event with
    rationale interpolating block_id + from_tier + to_tier + attempts."""
    capture = _FakeCapture()

    # Stub policy carrying a two-tier chain on the outline default.
    class _StubPolicy:
        capability_tiers = {
            "small": {
                "provider": "local",
                "model": "qwen2.5:7b-instruct-q4_K_M",
                "temperature": 0.0,
            },
            "medium": {
                "provider": "local",
                "model": "qwen2.5:14b-instruct-q4_K_M",
                "temperature": 0.4,
            },
        }
        capability_tier_chain_by_block_type: Dict[tuple, List[str]] = {}
        capability_tier_chain_by_default_tier: Dict[str, List[str]] = {
            "outline": ["small", "medium"],
        }
        n_candidates_by_block_type: Dict[str, int] = {}
        regen_budget_by_block_type: Dict[str, int] = {}
        regen_budget_rewrite_by_block_type: Dict[str, int] = {}

        def resolve(self, *args, **kwargs):
            return None

    # Validator that always returns regenerate so the loop walks the
    # whole budget; the chain escalation should fire mid-loop.
    class _AlwaysFail:
        def validate(self, inputs):
            from MCP.hardening.validation_gates import GateResult, GateIssue

            return GateResult(
                gate_id="x",
                validator_name="x",
                passed=False,
                action="regenerate",
                score=0.0,
                issues=[
                    GateIssue(code="X", severity="critical", message="forced fail")
                ],
            )

    outline_provider = _CountingProvider()
    router = CourseforgeRouter(
        policy=_StubPolicy(),
        outline_provider=outline_provider,
        capture=capture,
    )

    block = _block(block_type="concept")
    # n_candidates = resolved_budget so the loop iterates as many
    # times as the budget allows. Two-tier chain with default budget
    # 10 → sub-budget 5 each.
    out = router.route_with_self_consistency(
        block,
        n_candidates=10,
        regen_budget=10,
        validators=[_AlwaysFail()],
    )

    # Capability-escalation event fired at least once.
    cap_events = [
        e
        for e in capture.events
        if e["decision_type"] == "block_capability_escalation"
    ]
    assert len(cap_events) == 1, (
        f"expected exactly one capability-escalation event; got {len(cap_events)}"
    )
    ev = cap_events[0]
    assert ev["ml_features"]["from_tier"] == "small"
    assert ev["ml_features"]["to_tier"] == "medium"
    assert ev["ml_features"]["attempts"] >= 1
    # Plan: rationale ≥ 20 chars + interpolates dynamic signals.
    assert len(ev["rationale"]) >= 20
    assert block.block_id in ev["rationale"]
    assert "small" in ev["rationale"]
    assert "medium" in ev["rationale"]
    # Per-iteration overrides flipped the dispatch spec mid-loop.
    # First few calls: small tier (qwen 7b); later calls: medium tier.
    seen_models = {c["block"].block_id: [] for c in outline_provider.calls}
    # Dispatched models recorded via the iteration_overrides override
    # path — the FakeCapture sees the model on the route() event.
    route_events = [
        e
        for e in capture.events
        if e["decision_type"] == "block_outline_call"
        and "router_dispatch" in e["rationale"]
    ]
    early_model_lines = [
        e["rationale"] for e in route_events[: cap_events[0]["ml_features"]["attempts"]]
    ]
    later_model_lines = [
        e["rationale"]
        for e in route_events[cap_events[0]["ml_features"]["attempts"] :]
    ]
    assert any("7b-instruct" in r for r in early_model_lines), (
        f"expected small-tier dispatches first; got {early_model_lines}"
    )
    assert any("14b-instruct" in r for r in later_model_lines), (
        f"expected medium-tier dispatches after escalation; got {later_model_lines}"
    )
    # Outcome block carries the budget-exhausted marker once the chain
    # is fully consumed.
    assert out.escalation_marker == "outline_budget_exhausted"


def test_single_tier_chain_preserves_legacy_behaviour():
    """Length-1 capability chain (or no chain) MUST behave identically
    to the pre-§4 single-tier loop: no escalation event fires, the
    chain-exhausted path still stamps outline_budget_exhausted."""
    capture = _FakeCapture()

    class _AlwaysFail:
        def validate(self, inputs):
            from MCP.hardening.validation_gates import GateResult

            return GateResult(
                gate_id="x",
                validator_name="x",
                passed=False,
                action="regenerate",
                score=0.0,
            )

    outline_provider = _CountingProvider()
    router = CourseforgeRouter(
        outline_provider=outline_provider,
        capture=capture,
    )
    out = router.route_with_self_consistency(
        _block(),
        n_candidates=5,
        regen_budget=3,
        validators=[_AlwaysFail()],
    )

    cap_events = [
        e
        for e in capture.events
        if e["decision_type"] == "block_capability_escalation"
    ]
    assert cap_events == [], "no capability-escalation should fire on single-tier chain"
    assert out.escalation_marker == "outline_budget_exhausted"


def test_per_call_overrides_collapse_chain_to_single_tier():
    """Per-call overrides win outright per §2.2; the chain collapses
    to a single-element list (no mid-loop escalation)."""
    capture = _FakeCapture()

    class _StubPolicy:
        capability_tiers = {
            "small": {"provider": "local", "model": "model-a"},
            "medium": {"provider": "local", "model": "model-b"},
        }
        capability_tier_chain_by_block_type: Dict[tuple, List[str]] = {}
        capability_tier_chain_by_default_tier: Dict[str, List[str]] = {
            "outline": ["small", "medium"],
        }

        def resolve(self, *a, **kw):
            return None

    class _Provider:
        def __init__(self):
            self.calls: List[Dict[str, Any]] = []

        def generate_outline(self, block, **kwargs):
            self.calls.append(kwargs)
            return block

    p = _Provider()
    router = CourseforgeRouter(
        policy=_StubPolicy(),
        outline_provider=p,
        capture=capture,
    )

    chain = router._resolve_capability_tier_chain(
        _block(), "outline", provider="anthropic", model="cm-pinned"
    )
    assert len(chain) == 1
    assert chain[0].provider == "anthropic"
    assert chain[0].model == "cm-pinned"


def test_tier_sub_budget_default_split():
    """Default per-tier sub-budget = resolved_budget // len(chain),
    floor 1."""
    chain_3 = [
        BlockProviderSpec(
            block_type="x", tier="outline", provider="local", model=f"m{i}"
        )
        for i in range(3)
    ]
    assert CourseforgeRouter._tier_sub_budget(chain_3, 0, resolved_budget=12) == 4
    assert CourseforgeRouter._tier_sub_budget(chain_3, 1, resolved_budget=12) == 4
    # Single-element chain ⇒ full budget for that tier.
    chain_1 = [chain_3[0]]
    assert CourseforgeRouter._tier_sub_budget(chain_1, 0, resolved_budget=10) == 10
    # Floor of 1 even when budget < len.
    assert CourseforgeRouter._tier_sub_budget(chain_3, 0, resolved_budget=2) == 1


def test_tier_sub_budget_explicit_per_tier_override():
    """When a tier spec carries `regen_budget`, that value wins over
    the default split (operators can dial up the large-tier budget)."""
    chain = [
        BlockProviderSpec(
            block_type="x",
            tier="outline",
            provider="local",
            model="small",
            regen_budget=2,
        ),
        BlockProviderSpec(
            block_type="x",
            tier="outline",
            provider="local",
            model="large",
            regen_budget=8,
        ),
    ]
    assert CourseforgeRouter._tier_sub_budget(chain, 0, resolved_budget=10) == 2
    assert CourseforgeRouter._tier_sub_budget(chain, 1, resolved_budget=10) == 8


# ---------------------------------------------------------------------------
# §3 recorded-transcript replay (sanity)
# ---------------------------------------------------------------------------


_SURFACES_PATH = PROJECT_ROOT / "runtime" / "qwen_test" / "surfaces.json"


def _load_surfaces() -> Dict[str, Any]:
    if not _SURFACES_PATH.exists():
        pytest.skip(f"runtime/qwen_test/surfaces.json absent at {_SURFACES_PATH}")
    return json.loads(_SURFACES_PATH.read_text())


def test_recorded_outline_failure_strings_match_retry_directive_table():
    """The §1 failure validator strings recorded in surfaces.json each
    match a §3.6 retry-directive table pattern. Closes plan §5.2.1
    (replay the recorded transcripts and assert the new pipeline
    handles them)."""
    from Courseforge.generators._outline_provider import _match_retry_directive

    surfaces = _load_surfaces()
    outline_failures = [
        e for e in surfaces["outline"] if e.get("ok") is False
    ]
    if not outline_failures:
        pytest.skip("no recorded outline failures to replay")

    matched: List[str] = []
    for entry in outline_failures:
        # Pull the validator's last_error from the structured error
        # message recorded by the introspection harness. The
        # introspection script formats the error as
        # ``OutlineProviderError: ... (last_error="<msg>")``.
        err = entry.get("error", "") or ""
        if "last_error=" not in err:
            continue
        # Extract the inner last_error message.
        inner = err.split("last_error=", 1)[1]
        # Strip trailing closing paren + quote characters.
        inner = inner.rstrip(")")
        if inner.startswith('"') and inner.endswith('"'):
            inner = inner[1:-1]
        directive = _match_retry_directive(inner)
        matched.append((entry["block_type"], inner, directive))

    # Every recorded failure resolves to a directive (no silent fall-
    # throughs). Plan §3.6: four patterns seeded in the table
    # specifically to catch the §1 failure classes.
    for block_type, inner, directive in matched:
        assert directive is not None, (
            f"recorded {block_type} failure did not match any retry "
            f"directive pattern; last_error={inner!r}"
        )


def test_recorded_token_stuffing_rewrite_response_rejected_by_contextual_gate():
    """The §1.7 misconception rewrite-tier response (token-stuffing
    via `vocab="rdf:RDF"` attribute on an invented `<span>`) MUST be
    rejected by the new contextual CURIE-preservation gate."""
    from Courseforge.generators._rewrite_provider import _missing_preserve_curies

    surfaces = _load_surfaces()
    # Pull the misconception entry's wire transcripts; the second
    # response carries the recorded token-stuffing pattern.
    misc = next(
        (e for e in surfaces["rewrite"] if e["block_type"] == "misconception"),
        None,
    )
    if misc is None or len(misc.get("wire", [])) < 2:
        pytest.skip("no recorded misconception rewrite transcript with ≥2 attempts")
    stuffed_html = misc["wire"][1]["raw_response"]
    # The recorded stuffed response carries `vocab="rdf:RDF"` — a
    # CURIE in an attribute value. The contextual gate must reject.
    missing = _missing_preserve_curies(stuffed_html, ["rdf:RDF"])
    assert "rdf:RDF" in missing, (
        "contextual gate accepted a CURIE-in-attribute-value "
        "token-stuffing response; the gate's contract requires "
        "pedagogical-context occurrences only"
    )
