"""Tests for :mod:`Courseforge.router.policy` (Phase 3 Subtask 36).

Exercises the operator-tunable block-routing policy loader + the
:class:`BlockRoutingPolicy.resolve` resolution chain. Coverage:

- Missing policy file returns an empty policy (the "no operator
  overrides" mode the loader treats as non-fatal).
- Loaded YAML is validated against
  ``schemas/courseforge/block_routing.schema.json``; malformed
  payloads fail closed via ``jsonschema.ValidationError``.
- :meth:`BlockRoutingPolicy.resolve` walks the four-step chain
  documented in the module: per-block_id overrides (with fnmatch glob
  support) → blocks[type][tier] → defaults[tier] → None fall-through.
- :func:`match_block_id_glob` honors Python ``fnmatch`` semantics
  (``*`` / ``?`` / ``[seq]``).
- Loader honors ``COURSEFORGE_BLOCK_ROUTING_PATH`` env var as the
  override path source.
- Empty-file YAML is treated the same as a missing file (both produce
  an INFO log + empty policy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jsonschema  # noqa: E402

from Courseforge.router.policy import (  # noqa: E402
    BlockRoutingPolicy,
    _ENV_POLICY_PATH,
    load_block_routing_policy,
    match_block_id_glob,
)
from Courseforge.router.router import BlockProviderSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


_VALID_POLICY_BODY = """
version: 1
defaults:
  outline:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
    base_url: http://localhost:11434/v1
    temperature: 0.0
    max_tokens: 1200
  rewrite:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.4
    max_tokens: 2400
blocks:
  assessment_item:
    rewrite:
      provider: anthropic
      model: claude-sonnet-4-6
      temperature: 0.5
      max_tokens: 2400
  prereq_set:
    escalate_immediately: true
overrides:
  - block_id: "week_03_*#assessment_item_quiz_*"
    rewrite:
      provider: together
      model: meta-llama/Llama-3.3-70B-Instruct-Turbo
      temperature: 0.3
      max_tokens: 2000
"""


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_load_returns_empty_policy_when_file_absent(tmp_path, monkeypatch):
    # Force the loader off the repo's checked-in default by pointing
    # the env-var override at a definitely-missing tmp path.
    missing = tmp_path / "does_not_exist.yaml"
    monkeypatch.delenv(_ENV_POLICY_PATH, raising=False)

    policy = load_block_routing_policy(missing)

    assert isinstance(policy, BlockRoutingPolicy)
    assert policy.is_empty()
    assert policy.defaults == {}
    assert policy.blocks == {}
    assert policy.overrides == []


def test_load_validates_against_schema(tmp_path):
    yaml_path = _write_yaml(tmp_path / "policy.yaml", _VALID_POLICY_BODY)

    policy = load_block_routing_policy(yaml_path)

    assert not policy.is_empty()
    # defaults: both tiers projected to specs
    assert "outline" in policy.defaults
    assert "rewrite" in policy.defaults
    assert isinstance(policy.defaults["outline"], BlockProviderSpec)
    # blocks: per-block_type entry preserved
    assert "assessment_item" in policy.blocks
    assert policy.blocks["assessment_item"]["rewrite"].provider == "anthropic"
    # escalate_immediately captured into the fast-lookup map
    assert policy.escalate_immediately_by_block_type.get("prereq_set") is True
    # overrides preserved as raw dicts
    assert len(policy.overrides) == 1
    assert policy.overrides[0]["block_id"].startswith("week_03_")


def test_invalid_yaml_raises(tmp_path):
    bad = _write_yaml(
        tmp_path / "bad.yaml",
        # version: 999 violates `const: 1` in the schema.
        "version: 999\ndefaults: {}\n",
    )
    with pytest.raises(jsonschema.ValidationError):
        load_block_routing_policy(bad)


def test_invalid_yaml_raises_on_unknown_block_type(tmp_path):
    # Block type not in the canonical 16-value enum should fail closed.
    bad = _write_yaml(
        tmp_path / "bad_block.yaml",
        "version: 1\nblocks:\n  not_a_real_block_type:\n    outline:\n      provider: local\n      model: x\n",
    )
    with pytest.raises(jsonschema.ValidationError):
        load_block_routing_policy(bad)


def test_env_var_overrides_default_path(tmp_path, monkeypatch):
    yaml_path = _write_yaml(tmp_path / "from_env.yaml", _VALID_POLICY_BODY)
    monkeypatch.setenv(_ENV_POLICY_PATH, str(yaml_path))

    # Calling with no path arg should resolve to the env value.
    policy = load_block_routing_policy()

    assert not policy.is_empty()
    assert policy.blocks["assessment_item"]["rewrite"].temperature == 0.5


def test_empty_yaml_file_treated_as_empty_policy(tmp_path, monkeypatch):
    empty = _write_yaml(tmp_path / "empty.yaml", "")
    monkeypatch.delenv(_ENV_POLICY_PATH, raising=False)

    policy = load_block_routing_policy(empty)

    assert policy.is_empty()


# ---------------------------------------------------------------------------
# resolve() chain tests
# ---------------------------------------------------------------------------


def test_resolve_walks_overrides_first(tmp_path):
    yaml_path = _write_yaml(tmp_path / "policy.yaml", _VALID_POLICY_BODY)
    policy = load_block_routing_policy(yaml_path)

    spec = policy.resolve(
        block_id="week_03_module_01#assessment_item_quiz_42",
        block_type="assessment_item",
        tier="rewrite",
    )

    # The override entry pins `together`; the per-block_type entry pins
    # `anthropic`. Override must win.
    assert spec is not None
    assert spec.provider == "together"
    assert spec.model.startswith("meta-llama/")


def test_resolve_falls_through_to_blocks_then_defaults(tmp_path):
    yaml_path = _write_yaml(tmp_path / "policy.yaml", _VALID_POLICY_BODY)
    policy = load_block_routing_policy(yaml_path)

    # No override match for this block_id → blocks[assessment_item][rewrite].
    spec = policy.resolve(
        block_id="week_05_module_02#assessment_item_quiz_1",
        block_type="assessment_item",
        tier="rewrite",
    )
    assert spec is not None
    assert spec.provider == "anthropic"
    assert spec.temperature == 0.5  # the per-block_type override

    # No per-block_type entry for `concept` → defaults[outline] kicks in,
    # and the resolver re-stamps with the caller's block_type.
    spec_default = policy.resolve(
        block_id="week_01_module_01#concept_intro_1",
        block_type="concept",
        tier="outline",
    )
    assert spec_default is not None
    assert spec_default.provider == "local"
    assert spec_default.block_type == "concept"  # re-stamped, not "_default"


def test_resolve_returns_none_when_no_match(tmp_path):
    # Build a policy with neither defaults nor per-block_type for the
    # requested tier — resolve must return None so the router falls
    # through to env-var / hardcoded defaults.
    body = """
version: 1
defaults:
  outline:
    provider: local
    model: qwen2.5:7b-instruct-q4_K_M
"""
    yaml_path = _write_yaml(tmp_path / "outline_only.yaml", body)
    policy = load_block_routing_policy(yaml_path)

    spec = policy.resolve(
        block_id="anything",
        block_type="concept",
        tier="rewrite",  # no rewrite default + no per-block_type
    )
    assert spec is None


def test_resolve_returns_none_when_policy_is_empty():
    policy = BlockRoutingPolicy()
    assert policy.resolve("any-id", "concept", "outline") is None


# ---------------------------------------------------------------------------
# Glob tests
# ---------------------------------------------------------------------------


def test_block_id_glob_match_supports_star():
    assert match_block_id_glob(
        "week_03_module_01#assessment_item_quiz_42",
        "week_03_*#assessment_item_quiz_*",
    )
    assert not match_block_id_glob(
        "week_04_module_01#assessment_item_quiz_42",
        "week_03_*#assessment_item_quiz_*",
    )


def test_block_id_glob_match_supports_question_mark_and_seq():
    assert match_block_id_glob("page_a", "page_?")
    assert not match_block_id_glob("page_ab", "page_?")
    assert match_block_id_glob("page_3", "page_[0-9]")
    assert not match_block_id_glob("page_x", "page_[0-9]")


def test_block_id_glob_empty_pattern_never_matches():
    assert match_block_id_glob("anything", "") is False
