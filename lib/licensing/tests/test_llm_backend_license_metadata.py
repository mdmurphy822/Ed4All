"""SFT-C S6 — license_metadata_for_provider accessor tests.

Verifies the registry-field addition in
``MCP/orchestrator/llm_backend.py``: entries WITHOUT inline license fields
fall back to the roster; inline fields win; the accessor never mutates the
projected ``_OPENAI_COMPATIBLE_PROVIDERS`` dict. Offline (registry read
only, no network / model).
"""

from __future__ import annotations

import copy

from MCP.orchestrator import llm_backend
from MCP.orchestrator.llm_backend import (
    _OPENAI_COMPATIBLE_PROVIDERS,
    license_metadata_for_provider,
)


def test_unknown_provider_returns_none():
    assert license_metadata_for_provider("no-such-provider") is None


def test_local_provider_falls_back_to_roster():
    meta = license_metadata_for_provider("local")
    assert meta is not None
    # `local` resolves canonical pinned Nemotron Nano via the roster.
    assert meta["license_verdict"] == "safe"
    assert (
        meta["license_spdx"]
        == "LicenseRef-NVIDIA-Nemotron-OML-2025-12-15"
    )


def test_accessor_does_not_mutate_registry():
    before = copy.deepcopy(_OPENAI_COMPATIBLE_PROVIDERS)
    license_metadata_for_provider("local")
    license_metadata_for_provider("together")
    assert _OPENAI_COMPATIBLE_PROVIDERS == before


def test_inline_license_fields_win(monkeypatch):
    entry = {
        "base_url_env": None,
        "base_url_default": "http://x/v1",
        "api_key_env": None,
        "api_key_default": "k",
        "model_env": None,
        "model_default": "some-model",
        "api_key_required": False,
        "license_spdx": "MIT",
        "license_url": "https://example/license",
        "license_verdict": "safe",
        "license_obligations": ["retain MIT notice"],
    }
    monkeypatch.setitem(_OPENAI_COMPATIBLE_PROVIDERS, "fake-licensed", entry)
    meta = license_metadata_for_provider("fake-licensed")
    assert meta == {
        "license_spdx": "MIT",
        "license_url": "https://example/license",
        "license_verdict": "safe",
        "license_obligations": ["retain MIT notice"],
    }


def test_registry_projection_still_matches_source():
    # Guard: the license accessor + doc additions must not disturb the
    # byte-equality of the projected registry vs the YAML source.
    from lib.llm import endpoints as ep

    assert _OPENAI_COMPATIBLE_PROVIDERS == ep.openai_compatible_legacy_registry()
    assert not any(
        "license_spdx" in row for row in _OPENAI_COMPATIBLE_PROVIDERS.values()
    ), "no shipped YAML row declares inline license fields yet (roster is the source)"
