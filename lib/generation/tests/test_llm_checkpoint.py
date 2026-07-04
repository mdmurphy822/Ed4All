"""Unit tests for the shared fingerprinted LLM unit-checkpoint store.

``lib/generation/llm_checkpoint.py`` generalizes the stage-2 objective-
synthesis per-window resume sidecar into one mechanism shared by every
expensive per-unit LLM dispatch loop. These tests pin the store contract:
write/reuse/mismatch/torn-line/opt-out plus the family-vs-site flag
precedence. Hermetic (tmp dirs, no LLM, no network).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.generation.llm_checkpoint import (  # noqa: E402
    FAMILY_ENV,
    SCHEMA_VERSION,
    CheckpointStore,
    checkpoint_enabled,
    compute_fingerprint,
    hash_text,
)

_SITE_ENV = "ED4ALL_TEST_SITE_CHECKPOINT"


def _store(tmp_path: Path, *, site_env: str | None = _SITE_ENV) -> CheckpointStore:
    return CheckpointStore(
        tmp_path / ".units_checkpoint.jsonl",
        site_id="test_site",
        site_env=site_env,
    )


# --------------------------------------------------------------------------- #
# Flag gating: family default ON; site flag (when SET) wins over family.
# --------------------------------------------------------------------------- #
def test_family_default_on(monkeypatch):
    monkeypatch.delenv(FAMILY_ENV, raising=False)
    monkeypatch.delenv(_SITE_ENV, raising=False)
    assert checkpoint_enabled(_SITE_ENV) is True
    assert checkpoint_enabled(None) is True


def test_family_falsey_disables_all_sites(monkeypatch):
    monkeypatch.delenv(_SITE_ENV, raising=False)
    for tok in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv(FAMILY_ENV, tok)
        assert checkpoint_enabled(_SITE_ENV) is False
        assert checkpoint_enabled(None) is False


def test_site_flag_set_wins_over_family(monkeypatch):
    # Site ON beats family OFF.
    monkeypatch.setenv(FAMILY_ENV, "0")
    monkeypatch.setenv(_SITE_ENV, "1")
    assert checkpoint_enabled(_SITE_ENV) is True
    # Site OFF beats family ON (unset family = ON).
    monkeypatch.delenv(FAMILY_ENV, raising=False)
    monkeypatch.setenv(_SITE_ENV, "off")
    assert checkpoint_enabled(_SITE_ENV) is False


def test_garbage_tokens_parse_with_fallback(monkeypatch):
    monkeypatch.setenv(FAMILY_ENV, "garbage")
    monkeypatch.delenv(_SITE_ENV, raising=False)
    assert checkpoint_enabled(_SITE_ENV) is True  # garbage family → ON
    monkeypatch.setenv(_SITE_ENV, "garbage")
    assert checkpoint_enabled(_SITE_ENV) is True  # garbage site → ON


# --------------------------------------------------------------------------- #
# Fingerprint helper
# --------------------------------------------------------------------------- #
def test_fingerprint_deterministic_and_axis_sensitive():
    base = {"a": 1, "b": ["x", "y"], "sha": hash_text("body")}
    fp = compute_fingerprint(base)
    assert fp == compute_fingerprint(dict(base))  # key order irrelevant
    assert compute_fingerprint({**base, "a": 2}) != fp
    assert compute_fingerprint({**base, "sha": hash_text("body!")}) != fp


# --------------------------------------------------------------------------- #
# Store: write / reuse / mismatch
# --------------------------------------------------------------------------- #
def test_append_then_reuse_roundtrip(tmp_path):
    store = _store(tmp_path)
    fp = compute_fingerprint({"input": "one"})
    store.append("u1", fp, {"result": [1, 2, 3]})
    cache = store.load()
    assert store.reuse("u1", fp, cache=cache) == {"result": [1, 2, 3]}
    # Mismatched fingerprint → None (stale).
    assert store.reuse("u1", "deadbeef", cache=cache) is None
    # Unknown unit → None.
    assert store.reuse("u2", fp, cache=cache) is None


def test_last_line_wins_per_unit(tmp_path):
    store = _store(tmp_path)
    fp = compute_fingerprint({"i": 1})
    store.append("u1", fp, {"result": "old"})
    store.append("u1", fp, {"result": "new"})
    assert store.reuse("u1", fp) == {"result": "new"}


def test_reuse_or_compute_caches_and_skips_none(tmp_path):
    store = _store(tmp_path)
    fp = compute_fingerprint({"i": 1})
    calls = []

    def _fn():
        calls.append(1)
        return {"result": "computed"}

    assert store.reuse_or_compute("u1", fp, _fn) == {"result": "computed"}
    assert store.reuse_or_compute("u1", fp, _fn) == {"result": "computed"}
    assert len(calls) == 1  # second call served from the sidecar

    # A failing unit (fn → None) is NOT checkpointed → re-attempted.
    fails = []

    def _fail():
        fails.append(1)
        return None

    assert store.reuse_or_compute("u2", fp, _fail) is None
    assert store.reuse_or_compute("u2", fp, _fail) is None
    assert len(fails) == 2


def test_reserved_keys_protected(tmp_path):
    store = _store(tmp_path)
    fp = compute_fingerprint({"i": 1})
    store.append("u1", fp, {"fingerprint": "spoof", "result": "ok"})
    rec = store.load()["u1"]
    assert rec["fingerprint"] == fp  # envelope wins over payload collision
    assert store.reuse("u1", fp) == {"result": "ok"}


# --------------------------------------------------------------------------- #
# Torn-line tolerance / malformed lines
# --------------------------------------------------------------------------- #
def test_torn_trailing_line_tolerated(tmp_path):
    store = _store(tmp_path)
    fp = compute_fingerprint({"i": 1})
    store.append("u1", fp, {"result": "a"})
    store.append("u2", fp, {"result": "b"})
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "unit_id": "u3", "finger')
    cache = store.load()
    assert set(cache) == {"u1", "u2"}


def test_malformed_middle_line_and_schema_mismatch_skipped(tmp_path):
    store = _store(tmp_path)
    fp = compute_fingerprint({"i": 1})
    store.append("u1", fp, {"result": "a"})
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write(json.dumps({
            "schema_version": "v0-old", "unit_id": "u9",
            "fingerprint": fp, "result": "stale",
        }) + "\n")
    store.append("u2", fp, {"result": "b"})
    cache = store.load()
    assert set(cache) == {"u1", "u2"}
    assert cache["u1"]["schema_version"] == SCHEMA_VERSION


def test_load_missing_file_and_none_path(tmp_path):
    assert _store(tmp_path).load() == {}
    none_store = CheckpointStore(None, site_id="x", site_env=_SITE_ENV)
    assert none_store.load() == {}
    assert none_store.enabled is False
    none_store.append("u", "fp", {})  # no-op, no crash
    none_store.remove()  # no-op, no crash


# --------------------------------------------------------------------------- #
# Opt-out disables write and reuse; remove works regardless
# --------------------------------------------------------------------------- #
def test_flag_off_disables_write_and_reuse(tmp_path, monkeypatch):
    store = _store(tmp_path)
    fp = compute_fingerprint({"i": 1})
    store.append("u1", fp, {"result": "a"})  # enabled write

    monkeypatch.setenv(_SITE_ENV, "0")
    assert store.enabled is False
    assert store.load() == {}  # reuse disabled
    assert store.reuse("u1", fp) is None
    store.append("u2", fp, {"result": "b"})  # write disabled
    monkeypatch.delenv(_SITE_ENV, raising=False)
    assert set(store.load()) == {"u1"}  # u2 never landed


def test_family_off_disables_site_without_site_env(tmp_path, monkeypatch):
    store = CheckpointStore(
        tmp_path / ".cp.jsonl", site_id="family_only", site_env=None,
    )
    monkeypatch.setenv(FAMILY_ENV, "0")
    assert store.enabled is False
    store.append("u1", "fp", {"r": 1})
    assert not store.path.exists()


def test_remove_clears_sidecar_even_when_disabled(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.append("u1", "fp", {"r": 1})
    assert store.path.exists()
    monkeypatch.setenv(FAMILY_ENV, "0")
    monkeypatch.delenv(_SITE_ENV, raising=False)
    store.remove()  # --force clear must work even when the flag is off
    assert not store.path.exists()
    store.remove()  # idempotent on missing file
