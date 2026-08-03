"""Unit tests for the declarative SEAT SCHEDULE primitives.

Contract under test (``lib/vllm_container_lifecycle.py`` seat-schedule half):

  * ``resolve_seat_schedule_mode`` default-OFF / truthy-ON parse.
  * ``parse_seat_registry`` good / garbage / empty (``seat_name=base_url``).
  * ``resolve_seat_base_url`` / ``all_registered_seat_names``.
  * ``start_seat`` already-serving → 0.0; cold start + poll → seconds;
    unresolvable seat / missing container / start-fail / timeout → None.
  * ``stop_seat`` resolves + ``docker stop``; unresolvable → False.
  * ``coherence_probe`` sane content → True; empty / soup / no-choices / no
    model-id / HTTP error → False (NEVER raises).
  * ``_looks_coherent`` heuristic.

NO live docker / NO live network — subprocess + urlopen are monkeypatched.
"""

from __future__ import annotations

import io
import json

import pytest

import lib.vllm_container_lifecycle as vcl


_SEATS = "spark-super=http://localhost:8001,spark-glm=http://localhost:8002/"
_CONTAINERS = "http://localhost:8001=vllm-super,http://localhost:8002=vllm-glm"


@pytest.fixture(autouse=True)
def _reset_warn_flags():
    vcl._REGISTRY_WARNED = False
    vcl._SEAT_REGISTRY_WARNED = False
    vcl._LAUNCH_SPEC_WARNED = False
    yield
    vcl._REGISTRY_WARNED = False
    vcl._SEAT_REGISTRY_WARNED = False
    vcl._LAUNCH_SPEC_WARNED = False


@pytest.fixture
def _seat_env(monkeypatch):
    monkeypatch.setenv(vcl.ENV_SEAT_BASE_URLS, _SEATS)
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _CONTAINERS)


# --------------------------------------------------------------------------
# resolve_seat_schedule_mode
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False), ("", False), ("   ", False), ("0", False),
        ("false", False), ("off", False), ("garbage", False),
        ("1", True), ("true", True), ("TRUE", True), ("Yes", True), ("on", True),
    ],
)
def test_resolve_seat_schedule_mode(value, expected):
    assert vcl.resolve_seat_schedule_mode(value) is expected


# --------------------------------------------------------------------------
# parse_seat_registry / resolve_seat_base_url / all_registered_seat_names
# --------------------------------------------------------------------------
def test_parse_seat_registry_good():
    reg = vcl.parse_seat_registry(_SEATS)
    assert reg == {
        "spark-super": "http://localhost:8001",
        "spark-glm": "http://localhost:8002",   # trailing / stripped
    }


def test_parse_seat_registry_empty():
    assert vcl.parse_seat_registry("") == {}
    assert vcl.parse_seat_registry(None) == {}


def test_parse_seat_registry_partly_garbage():
    reg = vcl.parse_seat_registry("spark-super=http://x:1, bad, =y, z=, ok=http://y:2")
    assert reg == {"spark-super": "http://x:1", "ok": "http://y:2"}


def test_resolve_seat_base_url():
    assert vcl.resolve_seat_base_url("spark-super", value=_SEATS) == "http://localhost:8001"
    assert vcl.resolve_seat_base_url("nope", value=_SEATS) is None


def test_all_registered_seat_names():
    assert vcl.all_registered_seat_names(_SEATS) == {"spark-super", "spark-glm"}


# --------------------------------------------------------------------------
# _looks_coherent
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,ok",
    [
        ('{"seat_ok":true}', True), (' { "seat_ok": true } ', True),
        ("SEATOK", False), ("The answer is 42.", False),
        ('{"seat_ok":false}', False), ('{"ok":true}', False),
        ('{"seat_ok":true,"extra":1}', False), ('{"seat_ok":1}', False),
        (None, False), ("", False), ("    ", False),
        ("!!!!", False), ("aaaa", False), ("        ", False),
        (".", False),
    ],
)
def test_looks_coherent(text, ok):
    assert vcl._looks_coherent(text) is ok


# --------------------------------------------------------------------------
# start_seat
# --------------------------------------------------------------------------
def test_start_seat_already_serving(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "_probe_ready", lambda base_url, **k: True)
    calls = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: calls.append(args) or True)
    assert vcl.start_seat("spark-super") == 0.0
    assert calls == []  # no docker start when already serving


def test_start_seat_cold_start(monkeypatch, _seat_env):
    # not ready first probe, ready second → measured load
    probes = iter([False, True])
    monkeypatch.setattr(vcl, "_probe_ready", lambda base_url, **k: next(probes))
    started = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: started.append(args) or True)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)
    load = vcl.start_seat("spark-super")
    assert isinstance(load, float) and load >= 0.0
    assert started == [["start", "vllm-super"]]


def test_start_seat_unresolvable_seat(monkeypatch, _seat_env):
    # never touches docker/probe when the seat name is not mapped
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: pytest.fail("probed"))
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: pytest.fail("dockered"))
    assert vcl.start_seat("ghost") is None


def test_start_seat_no_container_mapping(monkeypatch):
    monkeypatch.setenv(vcl.ENV_SEAT_BASE_URLS, "spark-super=http://localhost:9999")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _CONTAINERS)  # 9999 not mapped
    assert vcl.start_seat("spark-super") is None


def test_start_seat_docker_start_fails(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: False)
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: False)
    assert vcl.start_seat("spark-super") is None


def test_start_seat_timeout(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: False)
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: True)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)
    assert vcl.start_seat("spark-super", timeout_seconds=0) is None


# --------------------------------------------------------------------------
# stop_seat
# --------------------------------------------------------------------------
def test_stop_seat(monkeypatch, _seat_env):
    calls = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: calls.append(args) or True)
    assert vcl.stop_seat("spark-glm") is True
    assert calls == [["stop", "vllm-glm"]]


def test_stop_seat_unresolvable(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: pytest.fail("dockered"))
    assert vcl.stop_seat("ghost") is False


# --------------------------------------------------------------------------
# coherence_probe
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen_factory(models_payload, chat_payload, *, chat_error=None):
    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if url.endswith("/v1/models"):
            return _FakeResp(models_payload)
        if url.endswith("/v1/chat/completions"):
            if chat_error is not None:
                raise chat_error
            return _FakeResp(chat_payload)
        raise AssertionError(f"unexpected url {url}")
    return _fake


_MODELS_OK = {"data": [{"id": "nemotron-super"}]}


def test_coherence_probe_ok(monkeypatch):
    chat = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": '{"seat_ok":true}',
            },
        }],
    }
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, chat))
    assert vcl.coherence_probe("http://localhost:8001") is True


def test_coherence_probe_empty_content(monkeypatch):
    chat = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, chat))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_null_content(monkeypatch):
    chat = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, chat))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_soup(monkeypatch):
    chat = {"choices": [{"message": {"role": "assistant", "content": "!!!!!!!!"}}]}
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, chat))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_no_choices(monkeypatch):
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, {"choices": []}))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_no_model_id(monkeypatch):
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory({"data": []}, {}))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_http_error(monkeypatch):
    monkeypatch.setattr(
        vcl.urllib.request, "urlopen",
        _fake_urlopen_factory(_MODELS_OK, {}, chat_error=OSError("connreset")),
    )
    assert vcl.coherence_probe("http://localhost:8001") is False


# --------------------------------------------------------------------------
# resolve_seat_load_timeout — env-tunable CEILING (default 1200s / 20 min)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 1200.0), ("", 1200.0), ("   ", 1200.0),
        ("garbage", 1200.0), ("0", 1200.0), ("-5", 1200.0),
        ("900", 900.0), ("1800", 1800.0), ("1200.5", 1200.5),
    ],
)
def test_resolve_seat_load_timeout(value, expected):
    assert vcl.resolve_seat_load_timeout(value) == expected


def test_resolve_seat_load_timeout_from_env(monkeypatch):
    monkeypatch.setenv(vcl.ENV_SEAT_LOAD_TIMEOUT_SECONDS, "1500")
    assert vcl.resolve_seat_load_timeout() == 1500.0
    monkeypatch.delenv(vcl.ENV_SEAT_LOAD_TIMEOUT_SECONDS, raising=False)
    assert vcl.resolve_seat_load_timeout() == 1200.0  # default 20 min


# --------------------------------------------------------------------------
# resolve_seat_coherence_attempts — BOUNDED coherence attempts (default 3)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 3), ("", 3), ("   ", 3), ("garbage", 3), ("0", 3), ("-2", 3),
        ("1.5", 3), ("1", 1), ("3", 3), ("5", 5),
    ],
)
def test_resolve_seat_coherence_attempts(value, expected):
    assert vcl.resolve_seat_coherence_attempts(value) == expected


def test_resolve_seat_coherence_attempts_from_env(monkeypatch):
    monkeypatch.setenv(vcl.ENV_SEAT_COHERENCE_ATTEMPTS, "4")
    assert vcl.resolve_seat_coherence_attempts() == 4
    monkeypatch.delenv(vcl.ENV_SEAT_COHERENCE_ATTEMPTS, raising=False)
    assert vcl.resolve_seat_coherence_attempts() == 3  # default


# --------------------------------------------------------------------------
# _coherence_check — bounded attempt count, not a ceiling
# --------------------------------------------------------------------------
def test_coherence_check_passes_first_attempt(monkeypatch):
    calls = {"n": 0}

    def _probe(base_url, **k):
        calls["n"] += 1
        return True

    monkeypatch.setattr(vcl, "coherence_probe", _probe)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)
    assert vcl._coherence_check("http://x", seat_name="s", attempts=3) is True
    assert calls["n"] == 1  # short-circuits on first coherent probe


def test_coherence_check_bounded_by_attempts(monkeypatch):
    calls = {"n": 0}
    slept = {"total": 0.0}

    def _probe(base_url, **k):
        calls["n"] += 1
        return False  # never coherent

    monkeypatch.setattr(vcl, "coherence_probe", _probe)
    monkeypatch.setattr(vcl.time, "sleep", lambda s: slept.__setitem__("total", slept["total"] + s))
    assert vcl._coherence_check("http://x", seat_name="s", attempts=3, interval=8.0) is False
    # EXACTLY `attempts` probes — never an unbounded ceiling-poll.
    assert calls["n"] == 3
    # Only (attempts-1) inter-attempt sleeps — far below any load ceiling.
    assert slept["total"] == 16.0


# --------------------------------------------------------------------------
# parse_seat_launch_specs / resolve_seat_launch_spec
# --------------------------------------------------------------------------
def test_parse_seat_launch_specs_script_paths():
    reg = vcl.parse_seat_launch_specs(
        "spark-super=/opt/seats/launch-super.sh,spark-glm=/opt/seats/launch-glm.sh"
    )
    assert reg == {
        "spark-super": "/opt/seats/launch-super.sh",
        "spark-glm": "/opt/seats/launch-glm.sh",
    }


def test_parse_seat_launch_specs_semicolon_command_with_commas():
    # A full command with commas needs the ';' separator; the spec keeps its '='.
    raw = "spark-super=docker run --name vllm-super --env A=1,B=2 img;spark-glm=/x.sh"
    reg = vcl.parse_seat_launch_specs(raw)
    assert reg["spark-super"] == "docker run --name vllm-super --env A=1,B=2 img"
    assert reg["spark-glm"] == "/x.sh"


def test_parse_seat_launch_specs_empty():
    assert vcl.parse_seat_launch_specs("") == {}
    assert vcl.parse_seat_launch_specs(None) == {}


def test_parse_seat_launch_specs_partly_garbage():
    reg = vcl.parse_seat_launch_specs("spark-super=/a.sh, noeq, =/b.sh, ok=/c.sh")
    assert reg == {"spark-super": "/a.sh", "ok": "/c.sh"}


def test_resolve_seat_launch_spec(monkeypatch):
    monkeypatch.setenv(vcl.ENV_SEAT_LAUNCH_SPECS, "spark-super=/opt/launch.sh")
    assert vcl.resolve_seat_launch_spec("spark-super") == "/opt/launch.sh"
    assert vcl.resolve_seat_launch_spec("nope") is None


# --------------------------------------------------------------------------
# recreate_seat — cold docker rm -f + relaunch + poll to live
# --------------------------------------------------------------------------
_LAUNCH = "spark-super=/opt/seats/launch-super.sh,spark-glm=/opt/seats/launch-glm.sh"


class _FakeDC:
    """DecisionCapture stand-in that records every log_decision call."""

    calls = []
    instances = []

    def __init__(self, course_code=None, phase=None, tool=None, **kw):
        self.course_code = course_code
        self.phase = phase
        self.tool = tool
        _FakeDC.instances.append(self)

    def log_decision(self, **kw):
        _FakeDC.calls.append(kw)

    def save(self):
        pass


@pytest.fixture
def _launch_env(monkeypatch, _seat_env):
    monkeypatch.setenv(vcl.ENV_SEAT_LAUNCH_SPECS, _LAUNCH)


@pytest.fixture
def _fake_capture(monkeypatch):
    _FakeDC.calls = []
    _FakeDC.instances = []
    import lib.decision_capture as dc_mod
    monkeypatch.setattr(dc_mod, "DecisionCapture", _FakeDC)
    return _FakeDC


def test_recreate_seat_happy_path(monkeypatch, _launch_env, _fake_capture):
    docker_calls, launch_calls = [], []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: docker_calls.append(args) or True)
    monkeypatch.setattr(vcl, "_run_launch_spec", lambda spec, **k: launch_calls.append(spec) or True)
    monkeypatch.setattr(vcl, "_probe_ready", lambda base_url, **k: True)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)

    load = vcl.recreate_seat("spark-super", timeout_seconds=5.0)
    assert isinstance(load, float) and load >= 0.0
    # docker rm -f of the mapped container fired, then the launch spec ran.
    assert docker_calls == [["rm", "-f", "vllm-super"]]
    assert launch_calls == ["/opt/seats/launch-super.sh"]


def test_recreate_seat_emits_decision_capture(
    monkeypatch, tmp_path, _launch_env, _fake_capture
):
    run_dir = tmp_path / "WF-real-run"
    monkeypatch.setenv("ED4ALL_PHASE_NAME", "training_synthesis")
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: True)
    monkeypatch.setattr(vcl, "_run_launch_spec", lambda spec, **k: True)
    monkeypatch.setattr(vcl, "_probe_ready", lambda base_url, **k: True)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)

    vcl.recreate_seat(
        "spark-super",
        reason="warm_start_incoherent",
        timeout_seconds=5.0,
        run_dir=run_dir,
    )

    assert len(_fake_capture.calls) == 1, "recreate must emit exactly one DecisionCapture"
    assert len(_fake_capture.instances) == 1
    assert _fake_capture.instances[0].course_code == "WF-real-run"
    assert _fake_capture.instances[0].phase == "training_synthesis"
    call = _fake_capture.calls[0]
    assert call["decision_type"] == "seat_cold_recreate"
    rationale = call["rationale"]
    assert "warm_start_incoherent" in rationale
    assert "spark-super" in rationale
    assert "http://localhost:8001" in rationale  # base_url interpolated
    assert len(rationale) >= 20  # project law: rationale ≥ 20 chars
    alternatives = call["alternatives_considered"]
    assert len(alternatives) == 2
    assert all(
        set(item) == {"option", "reason_rejected"} for item in alternatives
    )
    assert all("spark-super" in item["reason_rejected"] for item in alternatives)


def test_recreate_seat_no_launch_spec_returns_none(monkeypatch, _seat_env, _fake_capture):
    # No ED4ALL_SEAT_LAUNCH_SPECS → cannot cold-recreate.
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: pytest.fail("dockered"))
    monkeypatch.setattr(vcl, "_run_launch_spec", lambda *a, **k: pytest.fail("launched"))
    assert vcl.recreate_seat("spark-super", timeout_seconds=5.0) is None
    assert _fake_capture.calls == []  # no recreate attempted → no capture


def test_recreate_seat_launch_failure_returns_none(
    monkeypatch, tmp_path, _launch_env, _fake_capture
):
    run_dir = tmp_path / "WF-launch-failure"
    monkeypatch.setenv("ED4ALL_PHASE_NAME", "seat_schedule")
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: True)
    monkeypatch.setattr(vcl, "_run_launch_spec", lambda spec, **k: False)  # launch fails
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: pytest.fail("probed after failed launch"))
    assert vcl.recreate_seat(
        "spark-super",
        timeout_seconds=5.0,
        run_dir=run_dir,
    ) is None
    # A recreate was ATTEMPTED (rm + launch) so a capture still fires (launched=False).
    assert len(_fake_capture.calls) == 1
    assert len(_fake_capture.instances) == 1
    assert _fake_capture.instances[0].course_code == "WF-launch-failure"
    assert _fake_capture.instances[0].phase == "seat_schedule"
    assert "relaunch_ok=False" in _fake_capture.calls[0]["rationale"]


def test_recreate_seat_unresolvable_seat_returns_none(monkeypatch, _launch_env, _fake_capture):
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: pytest.fail("dockered"))
    assert vcl.recreate_seat("ghost", timeout_seconds=5.0) is None
    assert _fake_capture.calls == []


# --------------------------------------------------------------------------
# start_seat_coherent — warm start + coherence + cold-recreate self-heal
# --------------------------------------------------------------------------
def test_start_seat_coherent_warm_ok_no_recreate(monkeypatch, _launch_env):
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: 1.0)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: True)
    monkeypatch.setattr(vcl, "recreate_seat", lambda *a, **k: pytest.fail("recreated"))
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)

    res = vcl.start_seat_coherent("spark-super", timeout_seconds=0.0)
    assert res.ok is True
    assert res.recreated is False
    assert res.reason == "warm_start_coherent"


def test_start_seat_coherent_warm_incoherent_no_spec(monkeypatch, _seat_env):
    # No launch spec → cannot self-heal → not ok, no recreate.
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: 1.0)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: False)
    monkeypatch.setattr(vcl, "recreate_seat", lambda *a, **k: pytest.fail("recreated"))
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)

    res = vcl.start_seat_coherent("spark-super", timeout_seconds=0.0)
    assert res.ok is False
    assert res.recreated is False
    assert res.reason == "warm_incoherent_no_spec"


def test_start_seat_coherent_cold_recreate_heals(monkeypatch, _launch_env):
    # warm coherence FALSE (every bounded attempt), cold coherence TRUE (once the
    # recreate has run) → recreate fires, ends ok.
    state = {"recreated": False}
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: 1.0)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: state["recreated"])
    recreated = []

    def _fake_recreate(seat, **k):
        recreated.append(seat)
        state["recreated"] = True
        return 5.0

    monkeypatch.setattr(vcl, "recreate_seat", _fake_recreate)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)

    res = vcl.start_seat_coherent("spark-super", timeout_seconds=0.0)
    assert res.ok is True
    assert res.recreated is True
    assert res.reason == "cold_recreate_coherent"
    assert recreated == ["spark-super"]


def test_start_seat_coherent_warm_and_cold_both_incoherent(monkeypatch, _launch_env):
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: 1.0)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: False)  # never coherent
    monkeypatch.setattr(vcl, "recreate_seat", lambda seat, **k: 5.0)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)

    res = vcl.start_seat_coherent("spark-super", timeout_seconds=0.0)
    assert res.ok is False
    assert res.recreated is True
    assert res.reason == "still_incoherent_after_recreate"


def test_start_seat_coherent_recreate_failed(monkeypatch, _launch_env):
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: 1.0)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: False)
    monkeypatch.setattr(vcl, "recreate_seat", lambda seat, **k: None)  # recreate could not bring it up
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)

    res = vcl.start_seat_coherent("spark-super", timeout_seconds=0.0)
    assert res.ok is False
    assert res.recreated is True
    assert res.reason == "recreate_failed"


def test_start_seat_coherent_start_failed_no_probe(monkeypatch, _launch_env):
    # start_seat could not bring the seat live → never probe coherence / recreate.
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: None)
    monkeypatch.setattr(vcl, "coherence_probe", lambda *a, **k: pytest.fail("probed"))
    monkeypatch.setattr(vcl, "recreate_seat", lambda *a, **k: pytest.fail("recreated"))

    res = vcl.start_seat_coherent("spark-super", timeout_seconds=0.0)
    assert res.ok is False
    assert res.recreated is False
    assert res.reason == "start_failed"


def test_incoherent_recreates_after_bounded_attempts_not_full_ceiling(
    monkeypatch, _launch_env
):
    """A live-but-incoherent seat triggers the cold recreate after EXACTLY the
    bounded coherence attempts — it does NOT ride out the 1200s liveness ceiling.

    Uses a fake clock + attempt/sleep counters (NO real sleeps): liveness is
    immediate-200 (start_seat returns at once), coherence is always-false, and we
    assert the recreate fires after `attempts` probes with only bounded virtual
    sleep — far below the 1200s ceiling passed in.
    """
    monkeypatch.setenv(vcl.ENV_SEAT_COHERENCE_ATTEMPTS, "3")
    # Liveness resolves immediately (a genuinely-slow load would use the ceiling;
    # a mode-collapse must NOT).
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: 0.5)

    probe_calls = {"n": 0}
    state = {"recreated": False}

    def _probe(base_url, **k):
        probe_calls["n"] += 1
        return state["recreated"]  # incoherent until the recreate runs

    monkeypatch.setattr(vcl, "coherence_probe", _probe)

    recreate_at = {}

    def _fake_recreate(seat, **k):
        recreate_at["probes_before"] = probe_calls["n"]
        state["recreated"] = True
        return 4.0

    monkeypatch.setattr(vcl, "recreate_seat", _fake_recreate)

    # Fake clock: accumulate virtual time on sleep instead of really sleeping.
    virtual = {"t": 0.0}
    monkeypatch.setattr(vcl.time, "sleep", lambda s: virtual.__setitem__("t", virtual["t"] + s))

    # Pass the FULL 1200s liveness ceiling — proving coherence does NOT consume it.
    res = vcl.start_seat_coherent("spark-super", timeout_seconds=1200.0)

    assert res.ok is True and res.recreated is True
    assert res.reason == "cold_recreate_coherent"
    # Recreate fired after EXACTLY 3 warm coherence probes (the bounded attempts),
    # NOT after thousands of ceiling-poll iterations.
    assert recreate_at["probes_before"] == 3
    # Total virtual sleep is the 2 inter-attempt coherence waits (~16s), which is
    # DECISIVELY less than the 1200s liveness ceiling — the mode-collapse did not
    # wait out the ceiling.
    assert virtual["t"] == 16.0
    assert virtual["t"] < 1200.0
