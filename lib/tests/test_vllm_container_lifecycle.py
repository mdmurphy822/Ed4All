"""Unit tests for the per-model vLLM container-lifecycle lease helper.

Contract under test (``lib/vllm_container_lifecycle.py``):

  * ``resolve_vllm_container_lifecycle_mode`` default-OFF / truthy-ON parse.
  * ``parse_container_registry`` good / garbage / empty.
  * Flag OFF → every verb is a pure no-op (NO docker subprocess, NO network).
  * ``ensure_serving`` already-ready → ``0.0`` without ``docker start``.
  * ``ensure_serving`` starts + polls (mocked probe + docker) → measured seconds.
  * The ``sg docker -c`` fallback fires on a permission error / perms-shaped rc.
  * ``release`` / ``release_all`` stop the mapped container(s).
  * ``record_load_event`` writes a valid JSONL row.

NO live docker / NO live network — subprocess and the readiness probe are
monkeypatched throughout; a test that leaked a real ``docker`` / ``urlopen``
call would fail its ``assert``.
"""

from __future__ import annotations

import json

import pytest

import lib.vllm_container_lifecycle as vcl


_REGISTRY = "http://localhost:8000=vllm-omni,http://localhost:8001=vllm-embed"


@pytest.fixture(autouse=True)
def _reset_warn_flag():
    """The one-time registry warning is process-global; reset it per test."""
    vcl._REGISTRY_WARNED = False
    yield
    vcl._REGISTRY_WARNED = False


# --------------------------------------------------------------------------
# resolve_vllm_container_lifecycle_mode
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("0", False),
        ("false", False),
        ("off", False),
        ("garbage", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("Yes", True),
        ("on", True),
    ],
)
def test_resolve_mode_default_off(value, expected):
    assert vcl.resolve_vllm_container_lifecycle_mode(value) is expected


def test_resolve_mode_reads_env(monkeypatch):
    monkeypatch.delenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, raising=False)
    assert vcl.resolve_vllm_container_lifecycle_mode() is False
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    assert vcl.resolve_vllm_container_lifecycle_mode() is True


# --------------------------------------------------------------------------
# parse_container_registry
# --------------------------------------------------------------------------
def test_parse_registry_good():
    reg = vcl.parse_container_registry(_REGISTRY)
    assert reg == {
        "http://localhost:8000": "vllm-omni",
        "http://localhost:8001": "vllm-embed",
    }


def test_parse_registry_trailing_slash_normalized():
    reg = vcl.parse_container_registry("http://localhost:8000/=vllm-omni")
    assert reg == {"http://localhost:8000": "vllm-omni"}


def test_parse_registry_empty():
    assert vcl.parse_container_registry(None) == {}
    assert vcl.parse_container_registry("") == {}
    assert vcl.parse_container_registry("   ") == {}


def test_parse_registry_garbage_keeps_valid_pairs():
    # A no-'=' token, an empty-side token, and a double-'=' token are dropped;
    # the one valid pair survives (fail-soft, never raises).
    reg = vcl.parse_container_registry(
        "nonsense,=vllm-x,http://h=,http://localhost:8000=vllm-omni,a=b=c"
    )
    assert reg == {"http://localhost:8000": "vllm-omni"}


def test_parse_registry_all_garbage():
    assert vcl.parse_container_registry("nonsense,also-bad") == {}


# --------------------------------------------------------------------------
# Flag OFF → every verb is a pure no-op (no docker, no network).
# --------------------------------------------------------------------------
def test_flag_off_noops(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    monkeypatch.delenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, raising=False)

    def _boom(*a, **k):
        raise AssertionError("subprocess must NOT be called when flag is off")

    def _boom_probe(*a, **k):
        raise AssertionError("network probe must NOT run when flag is off")

    monkeypatch.setattr(vcl.subprocess, "run", _boom)
    monkeypatch.setattr(vcl, "_probe_ready", _boom_probe)

    assert vcl.ensure_serving("http://localhost:8000") is None
    assert vcl.release("http://localhost:8000") is False
    assert vcl.release_all() == 0


# --------------------------------------------------------------------------
# ensure_serving
# --------------------------------------------------------------------------
def test_ensure_serving_unmapped_base_url(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    monkeypatch.setattr(
        vcl.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no docker for unmapped")),
    )
    assert vcl.ensure_serving("http://localhost:9999") is None


def test_ensure_serving_already_ready(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: True)

    def _no_docker(*a, **k):
        raise AssertionError("docker start must NOT run when already ready")

    monkeypatch.setattr(vcl.subprocess, "run", _no_docker)
    assert vcl.ensure_serving("http://localhost:8000") == 0.0


def test_ensure_serving_starts_and_polls(monkeypatch, tmp_path):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)

    # Probe: not-ready twice, then ready.
    probes = iter([False, False, True])
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: next(probes))
    # docker start succeeds.
    docker_calls = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: docker_calls.append(args) or True)
    # No real sleeps.
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a: None)

    load = vcl.ensure_serving(
        "http://localhost:8000", timeout_seconds=60, run_dir=tmp_path
    )
    assert isinstance(load, float) and load >= 0.0
    assert docker_calls == [["start", "vllm-omni"]]
    # Load event recorded because a container was started and run_dir is known.
    events = (tmp_path / "model_load_events.jsonl").read_text().strip().splitlines()
    assert len(events) == 1
    row = json.loads(events[0])
    assert row["container"] == "vllm-omni"
    assert row["base_url"] == "http://localhost:8000"


def test_ensure_serving_docker_start_fails(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: False)
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: False)
    assert vcl.ensure_serving("http://localhost:8000", timeout_seconds=1) is None


def test_ensure_serving_timeout(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: False)
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: True)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a: None)
    # timeout_seconds=0 → loop body never enters, returns None.
    assert vcl.ensure_serving("http://localhost:8000", timeout_seconds=0) is None


# --------------------------------------------------------------------------
# _run_docker + sg-docker fallback
# --------------------------------------------------------------------------
class _Proc:
    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_run_docker_success(monkeypatch):
    calls = []

    def _run(argv, **k):
        calls.append(argv)
        return _Proc(0)

    monkeypatch.setattr(vcl.subprocess, "run", _run)
    assert vcl._run_docker(["stop", "vllm-omni"]) is True
    assert calls == [["docker", "stop", "vllm-omni"]]


def test_run_docker_sg_fallback_on_permission_stderr(monkeypatch):
    calls = []

    def _run(argv, **k):
        calls.append(argv)
        if argv[0] == "docker":
            return _Proc(1, stderr="Got permission denied while trying to connect")
        # sg docker -c "docker stop vllm-omni"
        return _Proc(0)

    monkeypatch.setattr(vcl.subprocess, "run", _run)
    assert vcl._run_docker(["stop", "vllm-omni"]) is True
    assert calls[0][0] == "docker"
    assert calls[1][:3] == ["sg", "docker", "-c"]
    assert "docker stop vllm-omni" in calls[1][3]


def test_run_docker_sg_fallback_on_permission_error(monkeypatch):
    calls = []

    def _run(argv, **k):
        calls.append(argv)
        if argv[0] == "docker":
            raise PermissionError("denied")
        return _Proc(0)

    monkeypatch.setattr(vcl.subprocess, "run", _run)
    assert vcl._run_docker(["start", "vllm-omni"]) is True
    assert calls[1][:3] == ["sg", "docker", "-c"]


def test_run_docker_nonperms_failure_no_fallback(monkeypatch):
    calls = []

    def _run(argv, **k):
        calls.append(argv)
        return _Proc(1, stderr="No such container: vllm-omni")

    monkeypatch.setattr(vcl.subprocess, "run", _run)
    assert vcl._run_docker(["stop", "vllm-omni"]) is False
    # A non-perms failure does NOT trigger the sg fallback.
    assert len(calls) == 1


def test_run_docker_missing_cli(monkeypatch):
    def _run(argv, **k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(vcl.subprocess, "run", _run)
    assert vcl._run_docker(["stop", "vllm-omni"]) is False


# --------------------------------------------------------------------------
# release / release_all
# --------------------------------------------------------------------------
def test_release(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    seen = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: seen.append(args) or True)
    assert vcl.release("http://localhost:8000") is True
    assert seen == [["stop", "vllm-omni"]]


def test_release_unmapped(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    monkeypatch.setattr(
        vcl, "_run_docker",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no docker for unmapped")),
    )
    assert vcl.release("http://localhost:9999") is False


def test_release_all(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    seen = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: seen.append(args) or True)
    assert vcl.release_all() == 2
    assert ["stop", "vllm-omni"] in seen
    assert ["stop", "vllm-embed"] in seen


def test_release_all_partial(monkeypatch):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINER_LIFECYCLE, "on")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)

    def _run(args, **k):
        return args != ["stop", "vllm-embed"]  # embed stop fails

    monkeypatch.setattr(vcl, "_run_docker", _run)
    assert vcl.release_all() == 1


# --------------------------------------------------------------------------
# record_load_event
# --------------------------------------------------------------------------
def test_record_load_event_writes_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _REGISTRY)
    vcl.record_load_event(tmp_path, "http://localhost:8000/", 12.3456)
    vcl.record_load_event(tmp_path, "http://localhost:8001", 0.5)
    lines = (tmp_path / "model_load_events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["base_url"] == "http://localhost:8000"
    assert r0["container"] == "vllm-omni"
    assert r0["load_seconds"] == 12.346
    assert "ts" in r0
    r1 = json.loads(lines[1])
    assert r1["container"] == "vllm-embed"


def test_record_load_event_none_run_dir_skips():
    # Must not raise.
    vcl.record_load_event(None, "http://localhost:8000", 1.0)


# --------------------------------------------------------------------------
# coherence_probe — reasoning-seat contract (2026-07-21 regression)
#
# A reasoning-parsed seat (e.g. nemotron_v3) routes think-tokens into
# ``reasoning_content``; the probe must (a) request thinking OFF via
# ``chat_template_kwargs`` and (b) accept coherent ``reasoning_content`` when
# ``content`` is empty — otherwise a HEALTHY seat is declared mode-collapsed
# and cold-recreated in a loop (observed live on the Super seat, 2026-07-21).


class _FakeResp:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _probe_with(monkeypatch, message: dict, captured: dict):
    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("/v1/models"):
            return _FakeResp({"data": [{"id": "m1"}]})
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"choices": [{"message": message}]})

    monkeypatch.setattr(vcl.urllib.request, "urlopen", fake_urlopen)
    return vcl.coherence_probe("http://localhost:9999")


def test_coherence_probe_sends_thinking_off_kwargs(monkeypatch):
    captured: dict = {}
    ok = _probe_with(monkeypatch, {"content": "SEATOK"}, captured)
    assert ok is True
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["max_tokens"] >= 64  # 24 starved reasoning seats


def test_coherence_probe_accepts_reasoning_only_content(monkeypatch):
    captured: dict = {}
    ok = _probe_with(
        monkeypatch,
        {"content": None,
         "reasoning_content": "The user wants the word SEATOK, so I will reply."},
        captured,
    )
    assert ok is True


def test_coherence_probe_still_rejects_empty_both(monkeypatch):
    captured: dict = {}
    ok = _probe_with(monkeypatch, {"content": None, "reasoning_content": ""}, captured)
    assert ok is False
