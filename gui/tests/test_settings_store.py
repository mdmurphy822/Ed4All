"""Tests for ``gui.settings_store`` — defaults, round-trip, render, mask, apply.

No fastapi needed (store is web-free). State is isolated via the ``state_dir``
fixture (``ED4ALL_STATE_RUNS_DIR`` -> tmp_path) so the real ``state/gui/`` is
never written.
"""

from __future__ import annotations

import json

import pytest

from gui import env_catalog, settings_store, shared_state


def test_load_returns_defaults_when_no_file(state_dir):
    doc = settings_store.load_settings()
    assert doc["version"] == 1
    assert doc["model_routing"]["global"]["mode"] == "local"
    assert doc["model_routing"]["global"]["provider"] == "anthropic"
    assert "flags" in doc and "retrieval" in doc
    # No settings.json should have been created merely by loading.
    assert not shared_state.settings_path().exists()


def test_save_load_round_trip(state_dir, sample_settings_doc):
    stored = settings_store.save_settings(sample_settings_doc)
    assert stored["updated_at"]  # stamped
    assert shared_state.settings_path().exists()

    reloaded = settings_store.load_settings()
    # Run-time consumers still get the raw secret (merged from the sidecar).
    assert reloaded["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"
    assert reloaded["model_routing"]["courseforge_outline"]["provider"] == "local"
    assert reloaded["flags"]["COURSEFORGE_TWO_PASS"] is True
    # settings.json holds NO plaintext secret value at rest.
    on_disk = json.loads(shared_state.settings_path().read_text())
    assert on_disk["env"]["ANTHROPIC_API_KEY"] is None
    assert "sk-ant-test-secret-value" not in shared_state.settings_path().read_text()


def test_save_rejects_unknown_provider(state_dir):
    bad = env_catalog.default_settings()
    bad["model_routing"]["courseforge_outline"]["provider"] = "not_a_real_provider"
    with pytest.raises(ValueError, match="not a registered"):
        settings_store.save_settings(bad)


def test_update_settings_deep_merges(state_dir, sample_settings_doc):
    settings_store.save_settings(sample_settings_doc)
    merged = settings_store.update_settings({"flags": {"TRAINFORGE_REQUIRE_EMBEDDINGS": True}})
    # Patch merged in; pre-existing flag preserved.
    assert merged["flags"]["TRAINFORGE_REQUIRE_EMBEDDINGS"] is True
    assert merged["flags"]["COURSEFORGE_TWO_PASS"] is True
    assert merged["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"


def test_render_env_maps_routing_flags_and_passthrough(state_dir, sample_settings_doc):
    rendered = settings_store.render_env(sample_settings_doc)
    # model_routing -> env
    assert rendered["LLM_MODE"] == "api"
    assert rendered["LLM_PROVIDER"] == "anthropic"
    assert rendered["COURSEFORGE_OUTLINE_PROVIDER"] == "local"
    assert rendered["COURSEFORGE_OUTLINE_MODEL"] == "qwen2.5:7b-instruct-q4_K_M"
    assert rendered["COURSEFORGE_REWRITE_PROVIDER"] == "anthropic"
    assert rendered["COURSEFORGE_REWRITE_MODEL"] == "claude-sonnet-4-6"
    # flags -> lowercase bool strings
    assert rendered["COURSEFORGE_TWO_PASS"] == "true"
    assert rendered["ED4ALL_ANSWER_LIBRARY_WIDE"] == "false"
    # retrieval.require_embeddings mirror
    assert rendered["TRAINFORGE_REQUIRE_EMBEDDINGS"] == "true"
    # explicit env passthrough of a catalog-known key
    assert rendered["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"


def test_render_env_drops_nulls_and_unknown_keys(state_dir):
    doc = env_catalog.default_settings()
    doc["env"]["NOT_A_CATALOG_KEY"] = "x"
    doc["env"]["LLM_MODEL"] = ""  # empty -> dropped
    rendered = settings_store.render_env(doc)
    assert "NOT_A_CATALOG_KEY" not in rendered
    assert "LLM_MODEL" not in rendered
    # null routing model slots are not emitted
    assert "COURSEFORGE_OUTLINE_MODEL" not in rendered


def test_mask_secrets_no_raw_api_key_leak(state_dir, sample_settings_doc):
    masked = settings_store.mask_secrets(sample_settings_doc)
    assert masked["env"]["ANTHROPIC_API_KEY"] == "set"
    # Non-secret env value untouched.
    assert masked["env"]["LLM_MODE"] == "api"
    # Deep-copy: original is unchanged.
    assert sample_settings_doc["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"
    # No raw secret anywhere in the masked serialization.
    assert "sk-ant-test-secret-value" not in json.dumps(masked)


def test_mask_secrets_empty_key_becomes_none(state_dir):
    doc = env_catalog.default_settings()
    doc["env"]["TOGETHER_API_KEY"] = "   "
    masked = settings_store.mask_secrets(doc)
    assert masked["env"]["TOGETHER_API_KEY"] is None


def test_apply_env_sets_os_environ(state_dir, sample_settings_doc, monkeypatch):
    # Ensure a clean slate for the keys we assert on.
    for key in (
        "ANTHROPIC_API_KEY",
        "COURSEFORGE_OUTLINE_PROVIDER",
        "COURSEFORGE_TWO_PASS",
        "TRAINFORGE_REQUIRE_EMBEDDINGS",
    ):
        monkeypatch.delenv(key, raising=False)

    applied = settings_store.apply_env(sample_settings_doc)
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"
    assert os.environ["COURSEFORGE_OUTLINE_PROVIDER"] == "local"
    assert os.environ["COURSEFORGE_TWO_PASS"] == "true"
    assert os.environ["TRAINFORGE_REQUIRE_EMBEDDINGS"] == "true"
    assert applied["COURSEFORGE_OUTLINE_MODEL"] == "qwen2.5:7b-instruct-q4_K_M"

    # .env.rendered is written and masks the secret value.
    rendered_path = shared_state.env_rendered_path()
    assert rendered_path.exists()
    body = rendered_path.read_text()
    assert "ANTHROPIC_API_KEY=set" in body
    assert "sk-ant-test-secret-value" not in body


# ------------------------------------------------------------- secrets at rest


def _secrets_file():
    return shared_state.settings_path().parent / "secrets.json"


def test_secret_at_rest_split_to_sidecar(state_dir, sample_settings_doc):
    """A set secret lands only in the 0600 sidecar, never plaintext in settings.json."""
    settings_store.save_settings(sample_settings_doc)

    settings_text = shared_state.settings_path().read_text()
    assert "sk-ant-test-secret-value" not in settings_text
    on_disk = json.loads(settings_text)
    assert on_disk["env"]["ANTHROPIC_API_KEY"] is None
    # Non-secret env value is untouched at rest.
    assert on_disk["env"]["LLM_MODE"] == "api"

    # The raw value lives in the sidecar, owner-only (0600).
    sidecar = _secrets_file()
    assert sidecar.exists()
    stored = json.loads(sidecar.read_text())
    assert stored["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"
    assert (sidecar.stat().st_mode & 0o777) == 0o600

    # The effective (loaded) doc still carries the raw value for run-time use.
    effective = settings_store.load_settings()
    assert effective["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"


def test_update_settings_preserves_sidecar_secret(state_dir, sample_settings_doc):
    """Patching a non-secret field must not lose the previously-stored secret."""
    settings_store.save_settings(sample_settings_doc)
    merged = settings_store.update_settings({"flags": {"TRAINFORGE_REQUIRE_EMBEDDINGS": True}})
    assert merged["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"
    assert "sk-ant-test-secret-value" not in shared_state.settings_path().read_text()


def test_unset_secret_removed_from_sidecar(state_dir, sample_settings_doc):
    """Clearing a secret drops it from the sidecar rather than leaving it stale."""
    settings_store.save_settings(sample_settings_doc)
    settings_store.update_settings({"env": {"ANTHROPIC_API_KEY": ""}})
    stored = json.loads(_secrets_file().read_text())
    assert "ANTHROPIC_API_KEY" not in stored
    effective = settings_store.load_settings()
    assert not (effective["env"].get("ANTHROPIC_API_KEY") or "").strip()


def test_legacy_plaintext_migrated_on_load(state_dir, sample_settings_doc):
    """An old-format settings.json with a plaintext secret is split on first load."""
    # Simulate a legacy file: raw plaintext secret written straight to settings.json.
    legacy = dict(sample_settings_doc)
    legacy["updated_at"] = "2026-01-01T00:00:00+00:00"
    shared_state._atomic_write_json(shared_state.settings_path(), legacy)
    assert not _secrets_file().exists()
    assert "sk-ant-test-secret-value" in shared_state.settings_path().read_text()

    # Loading transparently migrates: raw returned, plaintext scrubbed on disk.
    loaded = settings_store.load_settings()
    assert loaded["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"

    settings_text = shared_state.settings_path().read_text()
    assert "sk-ant-test-secret-value" not in settings_text
    assert json.loads(settings_text)["env"]["ANTHROPIC_API_KEY"] is None
    sidecar = json.loads(_secrets_file().read_text())
    assert sidecar["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"

    # Second load is a no-op (already migrated) and still returns the raw value.
    again = settings_store.load_settings()
    assert again["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-secret-value"


def test_mask_secrets_of_loaded_doc_shows_set(state_dir, sample_settings_doc):
    """GET masking is unchanged: the sidecar-merged doc masks a set secret to 'set'."""
    settings_store.save_settings(sample_settings_doc)
    masked = settings_store.mask_secrets(settings_store.load_settings())
    assert masked["env"]["ANTHROPIC_API_KEY"] == "set"
    assert "sk-ant-test-secret-value" not in json.dumps(masked)
