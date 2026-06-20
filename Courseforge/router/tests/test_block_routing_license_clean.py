"""Schema-validation regression for ``block_routing.license_clean.yaml`` (W-D11.F).

Asserts the opt-in license-clean sibling routing YAML at
``Courseforge/config/block_routing.license_clean.yaml`` parses, validates
against ``schemas/courseforge/block_routing.schema.json``, and round-trips
through :func:`Courseforge.router.policy.load_block_routing_policy` —
so a future schema change that breaks the canonical
``block_routing.yaml`` is also caught against the license-clean variant.

Why this test exists:

The license-clean variant is the file an operator points
``COURSEFORGE_BLOCK_ROUTING_PATH`` at when running a ToS-clean
pipeline (per ``docs/operations/license-clean-run.md``). Without a
regression here, a future edit to either the schema or the variant
could ship a malformed routing file that fails closed at pipeline
boot — a class of error that should be caught in CI.

Specific contract under test:

* The variant is YAML-parseable.
* It validates against ``schemas/courseforge/block_routing.schema.json``
  (Draft 2020-12, ``additionalProperties: false``).
* The ``large`` capability tier resolves to a ToS-clean provider
  (``local`` or ``together``) — never ``anthropic``. This is the
  load-bearing semantic difference between the variant and the
  canonical file.
* The 21-singular ``BLOCK_TYPES`` enum coverage matches the canonical
  file (no block_type is silently dropped in the variant).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jsonschema  # noqa: E402

from Courseforge.router.policy import (  # noqa: E402
    BlockRoutingPolicy,
    load_block_routing_policy,
)


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_LICENSE_CLEAN_PATH: Path = (
    PROJECT_ROOT / "Courseforge" / "config" / "block_routing.license_clean.yaml"
)
_CANONICAL_PATH: Path = (
    PROJECT_ROOT / "Courseforge" / "config" / "block_routing.yaml"
)
_SCHEMA_PATH: Path = (
    PROJECT_ROOT / "schemas" / "courseforge" / "block_routing.schema.json"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def license_clean_yaml() -> dict:
    assert _LICENSE_CLEAN_PATH.exists(), (
        f"License-clean variant not found at {_LICENSE_CLEAN_PATH}"
    )
    return yaml.safe_load(_LICENSE_CLEAN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def canonical_yaml() -> dict:
    assert _CANONICAL_PATH.exists(), (
        f"Canonical routing file not found at {_CANONICAL_PATH}"
    )
    return yaml.safe_load(_CANONICAL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def routing_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_license_clean_yaml_parses(license_clean_yaml: dict) -> None:
    """The license-clean variant must be YAML-parseable + non-empty."""
    assert isinstance(license_clean_yaml, dict)
    assert license_clean_yaml.get("version") == 1
    assert "capability_tiers" in license_clean_yaml
    assert "blocks" in license_clean_yaml


def test_license_clean_yaml_validates_against_schema(
    license_clean_yaml: dict, routing_schema: dict
) -> None:
    """Variant validates against the canonical Draft-2020-12 schema."""
    jsonschema.validate(license_clean_yaml, routing_schema)


def test_license_clean_large_tier_is_tos_clean(license_clean_yaml: dict) -> None:
    """The defining semantic difference: ``large`` tier must NOT be Anthropic.

    This is the load-bearing assertion that distinguishes the variant
    from the canonical file. If a future edit accidentally reverts the
    ``large`` tier back to ``anthropic``, this test fires and the
    file-rename / git-blame trail catches the regression.
    """
    tiers = license_clean_yaml.get("capability_tiers", {})
    large = tiers.get("large")
    assert large is not None, (
        "license-clean variant must define a `capability_tiers.large` entry"
    )
    provider = large.get("provider")
    assert provider in {"local", "together"}, (
        f"license-clean `large` tier must use a ToS-clean provider "
        f"(local|together); got {provider!r}. The whole point of the "
        f"variant is to swap claude-sonnet-4-6 for an Apache-2.0 / "
        f"Llama-Community-licensed alternative — see "
        f"docs/operations/license-clean-run.md and docs/LICENSING.md."
    )


def test_license_clean_loader_round_trip(monkeypatch) -> None:
    """The loader produces a non-empty :class:`BlockRoutingPolicy`."""
    # Don't let an existing env-var override sway the test — pin the
    # path explicitly through the load function's `path` arg.
    monkeypatch.delenv("COURSEFORGE_BLOCK_ROUTING_PATH", raising=False)

    policy = load_block_routing_policy(_LICENSE_CLEAN_PATH)

    assert isinstance(policy, BlockRoutingPolicy)
    assert not policy.is_empty(), (
        "license-clean variant loaded as an empty policy — schema "
        "validation likely silently failed"
    )
    # The variant overrides every per-block-type entry from the
    # canonical file; the loader must surface them.
    assert "assessment_item" in policy.blocks
    assert "concept" in policy.blocks


def test_license_clean_block_type_coverage_matches_canonical(
    license_clean_yaml: dict, canonical_yaml: dict
) -> None:
    """Variant covers exactly the same 21 ``BLOCK_TYPES`` entries.

    Drift here means a block_type was silently dropped from the
    variant — operators expecting the variant to be a full drop-in
    replacement for the canonical file would lose validator-matrix
    enforcement on the missing block_type.
    """
    license_clean_blocks = set(license_clean_yaml.get("blocks", {}).keys())
    canonical_blocks = set(canonical_yaml.get("blocks", {}).keys())
    assert license_clean_blocks == canonical_blocks, (
        f"license-clean block_types diverge from canonical:\n"
        f"  only in license-clean: {license_clean_blocks - canonical_blocks}\n"
        f"  only in canonical: {canonical_blocks - license_clean_blocks}"
    )
