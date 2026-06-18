"""Router + service tests for the Studio Create wizard + Settings page (C3).

Covers the new/extended server surfaces the Create wizard + Studio settings page
depend on:

* ``GET /api/settings/studio`` — the masked credentials/answer/global/local
  subset of the env catalog (no full operator catalog leak; secrets masked).
* ``POST /api/settings/test-provider`` — the cheap reachability probe the
  Settings "Test provider" button hits (local-provider arm stubbed so no real
  network call), asserting the ok / unreachable contract.
* ``run_service.PHASE_LABELS`` + ``/api/workflows`` phase ``label`` enrichment —
  the friendly phase names the progress checklist renders.
* ``run_service._poll_phase_progress`` — the live per-phase progress poller that
  appends ``[phase] <name> <state>`` lines the WS streams to the wizard.

Skipped on a default install with no ``fastapi`` (the ``gui`` extra is opt-in).
State is isolated via ``state_dir`` / ``libv2_root`` so the real ``state/gui/``
and ``LibV2/`` are never touched. Synthetic only — no real model / network.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402


@pytest.fixture
def client(state_dir, libv2_root):
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# GET /api/settings/studio — scoped subset
# --------------------------------------------------------------------------- #


def test_studio_settings_returns_scoped_subset(client, sample_settings_doc):
    from gui import settings_store

    settings_store.save_settings(sample_settings_doc)
    resp = client.get("/api/settings/studio")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Only Studio-relevant categories in the returned catalog.
    cats = {c["category"] for c in body["catalog"]}
    assert cats <= {"credentials", "global", "answer", "local"}
    assert "credentials" in cats and "global" in cats
    # The full operator catalog (dart / trainforge / embedding / courseforge)
    # is NOT returned here.
    assert "dart" not in cats and "trainforge" not in cats and "embedding" not in cats

    # Secret masked — no raw key leaks.
    assert body["env"].get("ANTHROPIC_API_KEY") == "set"
    assert "sk-ant-test-secret-value" not in resp.text

    # Read-only host/port echo present.
    assert body["host"]
    assert isinstance(body["port"], int)

    # Providers list present (the provider picker renders from it).
    names = {p["name"] for p in body["providers"]}
    assert "anthropic" in names and "local" in names


def test_studio_settings_only_studio_routing_tasks(client, sample_settings_doc):
    from gui import settings_store

    settings_store.save_settings(sample_settings_doc)
    body = client.get("/api/settings/studio").json()
    # Both Courseforge two-pass authoring tiers are in Studio scope and surfaced.
    assert "courseforge_outline" in body["model_routing"]
    assert "courseforge_rewrite" in body["model_routing"]
    # global (mode/provider/model) is in scope and surfaced.
    assert "global" in body["model_routing"]


def test_studio_settings_exposes_two_pass_flag_for_flow_tree(client, sample_settings_doc):
    """Phase 6: the Create-wizard AI-tier flow tree greys the Validate/Rewrite
    node off COURSEFORGE_TWO_PASS, so the studio payload must echo it (read-only,
    not a secret). No other operator flags leak."""
    from gui import settings_store

    settings_store.save_settings(sample_settings_doc)
    body = client.get("/api/settings/studio").json()
    assert "flags" in body, "studio payload must echo the scoped flags block"
    assert "COURSEFORGE_TWO_PASS" in body["flags"], "two-pass flag drives the flow tree"
    # Only the scoped flag is echoed — no DART_LLM_CLASSIFICATION etc.
    assert set(body["flags"]).issubset({"COURSEFORGE_TWO_PASS"})


# --------------------------------------------------------------------------- #
# POST /api/settings/test-provider — reachability probe (local arm stubbed)
# --------------------------------------------------------------------------- #


def test_test_provider_local_reachable(client, monkeypatch):
    """Local provider probe reports reachable when the stub returns HTTP 200."""
    import urllib.request

    class _Resp:
        status = 200

        def getcode(self):  # pragma: no cover - status attr wins
            return 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    resp = client.post("/api/settings/test-provider", json={"provider": "local"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "local"
    assert body["ok"] is True
    assert body["status"] == "reachable"


def test_test_provider_local_unreachable(client, monkeypatch):
    """Connection refused → ok=False / status=unreachable (clear failure)."""
    import urllib.request

    def _boom(*a, **k):
        raise OSError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    resp = client.post("/api/settings/test-provider", json={"provider": "local"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is False
    assert body["status"] == "unreachable"


def test_test_provider_unknown(client):
    resp = client.post("/api/settings/test-provider", json={"provider": "nope"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False and body["status"] == "unknown_provider"


def test_test_provider_missing_key_is_unauthorized(client, monkeypatch):
    """A hosted provider with no key reports missing_key (unauthorized arm)."""
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    resp = client.post("/api/settings/test-provider", json={"provider": "together"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False and body["status"] == "missing_key"


# --------------------------------------------------------------------------- #
# /api/workflows phase labels + PHASE_LABELS
# --------------------------------------------------------------------------- #


def test_workflows_carry_friendly_phase_labels(client):
    from gui.services import run_service

    body = client.get("/api/workflows").json()
    ttc = next(w for w in body["workflows"] if w["name"] == "textbook_to_course")
    by_name = {p["name"]: p for p in ttc["phases"]}
    # Every internal phase that has a friendly label exposes it.
    assert by_name["dart_conversion"]["label"] == run_service.PHASE_LABELS["dart_conversion"]
    assert by_name["vector_indexing"]["label"]  # the new vector_indexing phase is labelled
    # Optional phases keep their flag for the checklist.
    assert by_name["trainforge_assessment"]["optional"] is True


# --------------------------------------------------------------------------- #
# run_service._poll_phase_progress — live phase-progress log lines
# --------------------------------------------------------------------------- #


def test_poll_phase_progress_emits_friendly_lines(state_dir, monkeypatch):
    """The poller appends a friendly [phase] line per newly-completed phase."""
    from gui import shared_state
    from gui.services import run_service
    from lib.paths import STATE_PATH

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": run_id, "kind": "pipeline", "workflow": "textbook_to_course",
         "status": "running", "course_name": "T"}
    )
    workflow_id = "WF-TEST-0001"
    wf_dir = STATE_PATH / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    wf_file = wf_dir / f"{workflow_id}.json"

    async def _drive():
        # Start with one completed phase, then add a second + a skipped one.
        wf_file.write_text(json.dumps({
            "phase_outputs": {"dart_conversion": {"_completed": True}}
        }), encoding="utf-8")
        task = asyncio.ensure_future(run_service._poll_phase_progress(run_id, workflow_id))
        await asyncio.sleep(1.2)
        wf_file.write_text(json.dumps({
            "phase_outputs": {
                "dart_conversion": {"_completed": True},
                "staging": {"_completed": True},
                "trainforge_assessment": {"_completed": True, "_skipped": True},
            }
        }), encoding="utf-8")
        await asyncio.sleep(1.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(_drive())
    finally:
        wf_file.unlink(missing_ok=True)

    text, _ = shared_state.tail_log(run_id, 0)
    assert "[phase] dart_conversion done" in text
    assert "[phase] staging done" in text
    assert "[phase] trainforge_assessment skipped" in text
    # Friendly label is appended.
    assert "Convert textbook to accessible HTML" in text
