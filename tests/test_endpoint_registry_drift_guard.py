"""W9.3-lite: cross-registry DRIFT GUARD for the LLM endpoint catalogs.

The codebase keeps THREE hand-maintained descriptions of the LLM/API
endpoint set:

1. ``config/endpoints.yaml`` (loaded by ``lib/llm/endpoints.py``) — the
   CANONICAL source of truth. Transport + identity + provenance flow from
   one named row per endpoint.
2. ``MCP/orchestrator/llm_backend.py::_OPENAI_COMPATIBLE_PROVIDERS`` — a
   PROJECTION of (1) via ``_openai_compatible_legacy_registry()`` (derived,
   not independent).
3. ``gui/env_catalog.py::PROVIDERS`` — a HAND-MAINTAINED literal list the
   GUI renders (the module docstring notes it is "reproduced literally
   rather than importing" the registry). This is the one that DRIFTS.

This guard asserts a CONSISTENCY INVARIANT between the canonical registry
(1) and the hand-maintained GUI catalog (3): where they describe the SAME
provider they must AGREE (api_key_env, base_url, model_env), and the GUI
catalog must not invent an OpenAI-wire provider the canonical registry
does not know. It is a CONTRADICTION guard, not a completeness guard —
the GUI catalog may legitimately omit non-HTTP seats
(``claude_session``) or an SDK/large seat (``nvidia`` / ``nvidia-deepseek``)
it does not surface, and may add the ``mock`` convenience entry that is
not an HTTP endpoint. What it may NOT do is disagree with the canonical
row for a provider they BOTH define.

CANONICAL SOURCE: ``config/endpoints.yaml`` (via ``lib/llm/endpoints.py``)
is authoritative. When these registries disagree, fix the GUI catalog to
match the canonical YAML, not the reverse. The full three-registry
consolidation onto ``lib/llm/endpoints.py`` is the deferred W9.3 epic;
this test is only the drift TRIPWIRE.
"""

from __future__ import annotations

import pytest

from gui import env_catalog
from lib.llm.endpoints import endpoint_names, load_endpoint_registry

# The GUI catalog carries convenience entries that are intentionally NOT
# HTTP endpoints in the canonical registry.
_CATALOG_ONLY_ALLOWED = {"mock"}


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
