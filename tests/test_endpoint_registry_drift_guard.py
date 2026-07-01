"""W9.3-lite: cross-registry DRIFT GUARD for the LLM endpoint catalogs.

The codebase keeps THREE hand-maintained descriptions of the LLM/API
endpoint set:

1. ``config/endpoints.yaml`` (loaded by ``lib/llm/endpoints.py``) — the
   CANONICAL source of truth. Transport + identity + provenance flow from
   one named row per endpoint.
2. ``MCP/orchestrator/llm_backend.py::_OPENAI_COMPATIBLE_PROVIDERS`` — a
   PROJECTION of (1) via ``_openai_compatible_legacy_registry()`` (derived,
   not independent).
3. ``gui/env_catalog.py::PROVIDERS`` — the GUI-rendered provider list.
   Its shared OpenAI-wire seats (``local`` / ``together`` /
   ``together-vision`` / ``groq`` / ``fireworks`` / ``deepseek``) are now
   **DERIVED** from (1) via ``openai_compatible_legacy_registry()``
   (W9.3-next), so it can no longer drift for those seats; only the two
   non-wire seats (``anthropic`` SDK + ``mock``) stay hand-declared.

This guard asserts a CONSISTENCY INVARIANT between the canonical registry
(1) and the GUI catalog (3): where they describe the SAME provider they
must AGREE (api_key_env, base_url, model_env), and the GUI catalog must
not invent an OpenAI-wire provider the canonical registry does not know.
It is a CONTRADICTION guard, not a completeness guard — the GUI catalog
may legitimately omit non-HTTP seats (``claude_session``) or an SDK/large
seat (``nvidia`` / ``nvidia-deepseek``) it does not surface, and may add
the ``mock`` convenience entry that is not an HTTP endpoint. What it may
NOT do is disagree with the canonical row for a provider they BOTH define.

W9.3-next STRENGTHENING: beyond the contradiction guards, this file now
asserts the shared OpenAI-wire seats are SOURCED from canonical — every
wire field on the GUI entry must equal the ``openai_compatible_legacy_registry()``
projection of the same row (drift made impossible by CONSTRUCTION, not
merely detected). The assertion tolerates the two documented GUI overlays
that are NOT wire config: ``label`` (GUI metadata) and the ``local``
``vision_capable`` routing-dropdown force.

CANONICAL SOURCE: ``config/endpoints.yaml`` (via ``lib/llm/endpoints.py``)
is authoritative. When these registries disagree, fix the GUI catalog to
match the canonical YAML, not the reverse. The full three-registry
consolidation onto ``lib/llm/endpoints.py`` is the deferred W9.3 epic;
this test is the drift TRIPWIRE + the sourced-from-canonical assertion.
"""

from __future__ import annotations

import pytest

from gui import env_catalog
from lib.llm.endpoints import (
    endpoint_names,
    load_endpoint_registry,
    openai_compatible_legacy_registry,
)

# The GUI catalog carries convenience entries that are intentionally NOT
# HTTP endpoints in the canonical registry.
_CATALOG_ONLY_ALLOWED = {"mock"}

# GUI-explicit seats that are deliberately NOT derived from an
# openai_compatible canonical row (so the sourced-from-canonical assertion
# below skips them): ``anthropic`` is an SDK (kind=anthropic) seat whose GUI
# ``model_env`` is the operator-facing LLM_MODEL knob, and ``mock`` has no
# canonical endpoint at all.
_NOT_WIRE_DERIVED = {"anthropic", "mock"}

# The wire fields the GUI entry must source verbatim from the canonical
# legacy projection (keyed GUI-field -> legacy-projection-field).
_SOURCED_WIRE_FIELDS = {
    "api_key_env": "api_key_env",
    "base_url_env": "base_url_env",
    "base_url_default": "base_url_default",
    "model_env": "model_env",
    "model_default": "model_default",
    "api_key_required": "api_key_required",
}


@pytest.fixture(scope="module")
def registry():
    return load_endpoint_registry()


@pytest.fixture(scope="module")
def catalog():
    return {p["name"]: p for p in env_catalog.PROVIDERS}


def test_gui_catalog_providers_are_registered(registry, catalog):
    """Every GUI-catalog provider (bar the allowed convenience entries)
    must be a real endpoint in the canonical registry — the GUI must not
    invent an OpenAI-wire provider the registry does not know."""
    unknown = set(catalog) - set(endpoint_names()) - _CATALOG_ONLY_ALLOWED
    assert not unknown, (
        f"gui/env_catalog defines provider(s) absent from the canonical "
        f"config/endpoints.yaml: {sorted(unknown)}. Add the endpoint row to "
        f"the canonical registry or remove the catalog entry."
    )


def test_shared_providers_agree_on_api_key_env(registry, catalog):
    """A provider defined in BOTH registries must agree on its api_key_env."""
    for name in sorted(set(catalog) & set(registry)):
        reg_key = registry[name].get("api_key_env")
        cat_key = catalog[name].get("api_key_env")
        assert reg_key == cat_key, (
            f"api_key_env drift for {name!r}: registry={reg_key!r} vs "
            f"gui/env_catalog={cat_key!r}. Fix the GUI catalog to match "
            f"the canonical config/endpoints.yaml."
        )


def test_shared_openai_endpoints_agree_on_model_env_and_base_url(registry, catalog):
    """For OpenAI-compatible seats defined in BOTH, model_env + base_url
    must agree.

    Scoped to ``kind == openai_compatible`` so the ``anthropic`` row (whose
    GUI ``model_env`` is deliberately the GUI-facing ``LLM_MODEL`` knob,
    not the synthesis ``ANTHROPIC_SYNTHESIS_MODEL``) is excluded — that is
    a documented divergence, not drift.
    """
    for name in sorted(set(catalog) & set(registry)):
        row = registry[name]
        if row.get("kind") != "openai_compatible":
            continue
        reg_model_env = row.get("model_env")
        cat_model_env = catalog[name].get("model_env")
        assert reg_model_env == cat_model_env, (
            f"model_env drift for {name!r}: registry={reg_model_env!r} vs "
            f"gui/env_catalog={cat_model_env!r}."
        )
        reg_base = row.get("base_url") or None
        cat_base = catalog[name].get("base_url_default") or None
        assert reg_base == cat_base, (
            f"base_url drift for {name!r}: registry={reg_base!r} vs "
            f"gui/env_catalog={cat_base!r}."
        )


# ---------------------------------------------------------------------------
# W9.3-next: sourced-from-canonical assertions (drift impossible by
# construction, not merely detected).
# ---------------------------------------------------------------------------
def _shared_wire_providers(catalog):
    """GUI providers that must be sourced from an openai_compatible row."""
    legacy = openai_compatible_legacy_registry()
    return {
        name: legacy[name]
        for name in sorted(set(catalog) & set(legacy))
        if name not in _NOT_WIRE_DERIVED
    }


def test_shared_wire_providers_are_sourced_from_canonical(catalog):
    """Every shared OpenAI-wire GUI seat's wire config must EQUAL the
    canonical ``openai_compatible_legacy_registry()`` projection.

    This is the strengthened invariant: because ``gui/env_catalog`` derives
    these entries from the projection, any future attempt to hand-edit one
    of the wire fields diverges from canonical and trips here — drift is
    impossible by construction. The two documented GUI overlays (``label``
    metadata and the ``local`` vision_capable routing force) are NOT wire
    fields and are intentionally excluded.
    """
    shared = _shared_wire_providers(catalog)
    assert shared, "expected at least one shared OpenAI-wire provider"
    for name, projected in shared.items():
        entry = catalog[name]
        for gui_field, legacy_field in _SOURCED_WIRE_FIELDS.items():
            expected = projected.get(legacy_field)
            # api_key_required is a bool on both sides; normalize.
            if legacy_field == "api_key_required":
                expected = bool(expected)
                actual = bool(entry.get(gui_field, False))
            else:
                actual = entry.get(gui_field)
            assert actual == expected, (
                f"{name!r}.{gui_field} is NOT sourced from canonical: "
                f"gui/env_catalog={actual!r} vs "
                f"config/endpoints.yaml projection={expected!r}. These GUI "
                f"seats must derive from openai_compatible_legacy_registry()."
            )


def test_shared_wire_providers_unverified_sourced_from_canonical(catalog):
    """The ``unverified`` audit flag must also trace to canonical.

    Canonical rows carry ``unverified`` only for the illustrative stub
    seats (groq / fireworks / deepseek); rows that omit it default to
    False on both sides.
    """
    shared = _shared_wire_providers(catalog)
    for name, projected in shared.items():
        expected = bool(projected.get("unverified", False))
        actual = bool(catalog[name].get("unverified", False))
        assert actual == expected, (
            f"{name!r}.unverified is NOT sourced from canonical: "
            f"gui/env_catalog={actual!r} vs projection={expected!r}."
        )


def test_local_vision_capable_env_sourced_from_canonical(catalog):
    """The ``local`` seat's ``vision_capable_env`` must trace to the
    canonical row (LOCAL_VISION_CAPABLE), the only wire-adjacent field on
    the vision overlay that IS sourced (the vision_capable *bool* is a GUI
    routing-dropdown force and is deliberately NOT asserted)."""
    if "local" not in catalog:
        pytest.skip("local seat not exposed by the GUI catalog")
    projected = openai_compatible_legacy_registry().get("local", {})
    assert catalog["local"].get("vision_capable_env") == projected.get(
        "vision_capable_env"
    ), "local.vision_capable_env must be sourced from config/endpoints.yaml"
