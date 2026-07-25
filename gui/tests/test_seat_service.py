"""vLLM seat-monitor tests — service, endpoints, and the two mounted surfaces.

Hermetic by construction: the ``/v1/models`` probe and the ``docker ps`` read
are ALWAYS monkeypatched, so no network call, no docker invocation and no seat
mutation ever happens. The registries are plain env strings, so every test uses
INVENTED seat names — proving the surface is registry-driven and model-agnostic
(a new seat is an ``ED4ALL_SEAT_BASE_URLS`` entry, never code).

Covered:

* registry-driven rendering + registry ORDER (arbitrary names),
* the four states, incl. container-up-but-endpoint-silent → ``loading`` (a
  ~9-minute cold start must NOT render as an alarm) and docker-missing →
  ``unknown``, never a guessed ``down``,
* expected-seat derivation for all three ``seats:`` annotation cases
  (named / explicit ``[]`` / absent) against a canned config AND a read-only
  sanity check of the REAL ``config/workflows.yaml``,
* mismatch ONLY for a named seat that is neither live nor loading,
* the shared TTL cache (one probe round serves the dashboard AND a build page),
* transition tracking (first observation → null; unchanged preserves; change
  resets),
* never-raise on a broken registry / raising probe,
* router shapes + 404 + serve-mode mounts,
* the static-asset contract for seats.js / create.js / dashboard.js / the CSS
  (text-not-color-only a11y).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gui.services import seat_service

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "gui" / "static" / "studio"
SEATS_JS = STUDIO_DIR / "seats.js"
CREATE_JS = STUDIO_DIR / "create.js"
DASHBOARD_JS = STUDIO_DIR / "dashboard.js"
STUDIO_CSS = STUDIO_DIR / "studio.css"

# Deliberately NOT the deployment's real seat names: the whole surface must
# render whatever is registered.
SEAT_A = "atlas-large"
SEAT_B = "pebble-small"
URL_A = "http://localhost:9101/v1"
URL_B = "http://localhost:9102/v1"
CONTAINER_A = "vllm-atlas"
CONTAINER_B = "vllm-pebble"


@pytest.fixture(autouse=True)
def _reset():
    seat_service.reset_cache()
    yield
    seat_service.reset_cache()


@pytest.fixture
def registry(monkeypatch):
    """Two invented seats, both with a registered container."""
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", f"{SEAT_A}={URL_A},{SEAT_B}={URL_B}")
    monkeypatch.setenv(
        "ED4ALL_VLLM_CONTAINERS", f"{URL_A}={CONTAINER_A},{URL_B}={CONTAINER_B}"
    )


def _install(monkeypatch, *, live=(), running=(), probe_log=None):
    """Stub the probe + docker read. ``running=None`` means docker unavailable."""

    def _probe(base_url):
        if probe_log is not None:
            probe_log.append(base_url)
        return "some-model-id" if base_url in live else None

    monkeypatch.setattr(seat_service, "_probe_seat", _probe)
    monkeypatch.setattr(
        seat_service,
        "_docker_running_names",
        lambda: None if running is None else set(running),
    )


# ------------------------------------------------------------ registry-driven
def test_renders_every_registered_seat_in_registry_order(monkeypatch, registry):
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A, CONTAINER_B))
    payload = seat_service.seat_overview()
    assert [s["name"] for s in payload["seats"]] == [SEAT_A, SEAT_B]
    assert payload["registry_configured"] is True
    assert payload["seats"][0]["base_url"] == URL_A
    assert payload["seats"][0]["container"] == CONTAINER_A


def test_novel_seat_names_render_unchanged(monkeypatch):
    """A seat name the code has never seen must render — no hardcoded roster."""
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "quokka-xl=http://localhost:9999/v1")
    monkeypatch.delenv("ED4ALL_VLLM_CONTAINERS", raising=False)
    _install(monkeypatch, live=("http://localhost:9999/v1",), running=())
    payload = seat_service.seat_overview()
    assert [s["name"] for s in payload["seats"]] == ["quokka-xl"]
    assert payload["seats"][0]["state"] == seat_service.STATE_LIVE


def test_empty_registry_yields_no_seats_and_no_probing(monkeypatch):
    monkeypatch.delenv("ED4ALL_SEAT_BASE_URLS", raising=False)
    log = []
    _install(monkeypatch, live=(), running=(), probe_log=log)
    payload = seat_service.seat_overview()
    assert payload["seats"] == []
    assert payload["registry_configured"] is False
    assert log == [], "an empty registry must not probe anything"


# --------------------------------------------------------------- state matrix
def test_live_loading_down_classification(monkeypatch, registry):
    # A: answering /v1/models → live. B: container RUNNING but endpoint silent
    # → loading (the ~9-minute cold start), NOT down.
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A, CONTAINER_B))
    seats = {s["name"]: s for s in seat_service.seat_overview()["seats"]}
    assert seats[SEAT_A]["state"] == "live" and seats[SEAT_A]["live"] is True
    assert seats[SEAT_A]["model"] == "some-model-id"
    assert seats[SEAT_B]["state"] == "loading" and seats[SEAT_B]["live"] is False
    assert seats[SEAT_B]["model"] is None


def test_container_not_running_is_down(monkeypatch, registry):
    _install(monkeypatch, live=(), running=(CONTAINER_A,))
    seats = {s["name"]: s for s in seat_service.seat_overview()["seats"]}
    assert seats[SEAT_A]["state"] == "loading"  # container up, endpoint silent
    assert seats[SEAT_B]["state"] == "down"  # container absent from docker ps


def test_docker_unavailable_is_unknown_not_down(monkeypatch, registry):
    _install(monkeypatch, live=(), running=None)
    payload = seat_service.seat_overview()
    assert {s["state"] for s in payload["seats"]} == {"unknown"}
    assert payload["docker_available"] is None


def test_unregistered_container_is_unknown_not_down(monkeypatch):
    """Docker is readable but the seat has no container mapping → unknown."""
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", f"{SEAT_A}={URL_A}")
    monkeypatch.delenv("ED4ALL_VLLM_CONTAINERS", raising=False)
    _install(monkeypatch, live=(), running=(CONTAINER_A,))
    payload = seat_service.seat_overview()
    assert payload["seats"][0]["state"] == "unknown"
    assert payload["seats"][0]["container"] is None


def test_probe_url_never_doubles_the_v1_suffix(monkeypatch):
    """The registry convention is a bare base URL, but a ``/v1`` entry is
    tolerated rather than reported as a false ``down``."""
    import urllib.request

    seen = []

    class _Resp:
        status = 200

        def read(self, _n=None):
            return b'{"object":"list","data":[{"id":"m-1"}]}'

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _urlopen(req, timeout=None):  # noqa: ANN001
        seen.append(req.full_url)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert seat_service._probe_seat("http://host:9101") == "m-1"
    assert seat_service._probe_seat("http://host:9101/v1") == "m-1"
    assert seat_service._probe_seat("http://host:9101/v1/") == "m-1"
    assert seen == ["http://host:9101/v1/models"] * 3


def test_docker_binary_missing_returns_none_not_error(monkeypatch):
    """``_docker_running_names`` degrades to None (unknown) when docker is absent."""

    def _boom(*_a, **_kw):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert seat_service._docker_running_names() is None


def test_docker_permission_denied_retries_via_sg(monkeypatch):
    """A permission-shaped failure retries once through ``sg docker -c``."""
    calls = []

    class _Proc:
        def __init__(self, rc, out, err):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def _run(argv, **_kw):
        calls.append(argv)
        if argv[0] == "docker":
            return _Proc(1, "", "Got permission denied while trying to connect")
        return _Proc(0, f"{CONTAINER_A}\n", "")

    monkeypatch.setattr(subprocess, "run", _run)
    assert seat_service._docker_running_names() == {CONTAINER_A}
    assert calls[0][0] == "docker" and calls[1][0] == "sg"


# ------------------------------------------------ expected seats (3 cases)
_CANNED_WF = {
    "demo_flow": {
        "phases": [
            {"name": "needs_seat", "seats": [SEAT_A]},
            {"name": "seat_free", "seats": []},
            {"name": "no_opinion"},
            {"name": "malformed", "seats": "atlas-large"},
        ]
    }
}


@pytest.fixture
def canned_workflows(monkeypatch):
    monkeypatch.setattr(seat_service, "_workflows_config", lambda: _CANNED_WF)


def test_expected_seats_three_annotation_cases(canned_workflows):
    assert seat_service.phase_expected_seats("demo_flow", "needs_seat") == [SEAT_A]
    assert seat_service.phase_expected_seats("demo_flow", "seat_free") == []
    assert seat_service.phase_expected_seats("demo_flow", "no_opinion") is None
    # A malformed annotation is "no opinion", never a guessed expectation.
    assert seat_service.phase_expected_seats("demo_flow", "malformed") is None
    # Unknown workflow / phase / empty input → no opinion.
    assert seat_service.phase_expected_seats("nope", "needs_seat") is None
    assert seat_service.phase_expected_seats("demo_flow", "nope") is None
    assert seat_service.phase_expected_seats("", "") is None


def test_real_workflows_config_carries_both_annotation_shapes():
    """Read-only sanity: the shipped config really does declare both shapes."""
    workflows = seat_service._workflows_config()
    assert workflows, "config/workflows.yaml must be readable"
    named = seat_free = absent = 0
    for wf in workflows.values():
        for ph in (wf or {}).get("phases") or []:
            if "seats" not in ph:
                absent += 1
            elif ph.get("seats"):
                named += 1
            else:
                seat_free += 1
    assert named and seat_free and absent, (
        f"expected all three annotation cases in the shipped config "
        f"(named={named}, seat_free={seat_free}, absent={absent})"
    )


# ------------------------------------------------------------------ mismatch
def _stub_run(monkeypatch, *, phase, workflow="demo_flow", status="running"):
    from gui.services import progress_service

    monkeypatch.setattr(
        progress_service,
        "run_progress",
        lambda rid: {
            "run_id": rid,
            "workflow": workflow,
            "status": status,
            "effective_status": status,
            "current_phase": phase,
        },
    )


def test_mismatch_when_named_seat_is_down(monkeypatch, registry, canned_workflows):
    _install(monkeypatch, live=(URL_B,), running=(CONTAINER_B,))  # A down
    _stub_run(monkeypatch, phase="needs_seat")
    payload = seat_service.seat_overview("WF-20260722-abcdef01")
    assert payload["current_phase"] == "needs_seat"
    assert payload["expected_seats"] == [SEAT_A]
    assert payload["mismatch"] == [
        {"seat": SEAT_A, "state": "down", "base_url": URL_A}
    ]
    seats = {s["name"]: s for s in payload["seats"]}
    assert seats[SEAT_A]["expected"] is True and seats[SEAT_B]["expected"] is False


def test_no_mismatch_while_named_seat_is_loading(monkeypatch, registry, canned_workflows):
    """A cold start is progress, not an alarm."""
    _install(monkeypatch, live=(), running=(CONTAINER_A,))
    _stub_run(monkeypatch, phase="needs_seat")
    payload = seat_service.seat_overview("WF-20260722-abcdef01")
    assert payload["seats"][0]["state"] == "loading"
    assert payload["mismatch"] == []


def test_no_mismatch_when_named_seat_is_live(monkeypatch, registry, canned_workflows):
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A,))
    _stub_run(monkeypatch, phase="needs_seat")
    assert seat_service.seat_overview("WF-20260722-abcdef01")["mismatch"] == []


def test_seat_free_phase_never_mismatches(monkeypatch, registry, canned_workflows):
    _install(monkeypatch, live=(), running=())  # everything down
    _stub_run(monkeypatch, phase="seat_free")
    payload = seat_service.seat_overview("WF-20260722-abcdef01")
    assert payload["expected_seats"] == []
    assert payload["mismatch"] == []


def test_absent_annotation_never_mismatches(monkeypatch, registry, canned_workflows):
    _install(monkeypatch, live=(), running=())
    _stub_run(monkeypatch, phase="no_opinion")
    payload = seat_service.seat_overview("WF-20260722-abcdef01")
    assert payload["expected_seats"] is None
    assert payload["mismatch"] == []
    assert all(s["expected"] is False for s in payload["seats"])


def test_named_but_unregistered_seat_is_a_loud_mismatch(monkeypatch, canned_workflows):
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", f"{SEAT_B}={URL_B}")
    monkeypatch.delenv("ED4ALL_VLLM_CONTAINERS", raising=False)
    _install(monkeypatch, live=(URL_B,), running=())
    _stub_run(monkeypatch, phase="needs_seat")
    payload = seat_service.seat_overview("WF-20260722-abcdef01")
    assert payload["mismatch"] == [
        {"seat": SEAT_A, "state": "unregistered", "base_url": None}
    ]


def test_terminal_run_has_no_current_phase_and_no_mismatch(
    monkeypatch, registry, canned_workflows
):
    _install(monkeypatch, live=(), running=())
    _stub_run(monkeypatch, phase=None, status="completed")
    payload = seat_service.seat_overview("WF-20260722-abcdef01")
    assert payload["current_phase"] is None
    assert payload["expected_seats"] is None
    assert payload["mismatch"] == []


def test_unknown_run_returns_none(monkeypatch, registry):
    from gui.services import progress_service

    _install(monkeypatch, live=(), running=())
    monkeypatch.setattr(progress_service, "run_progress", lambda rid: None)
    assert seat_service.seat_overview("WF-nope") is None
    assert seat_service.seat_overview("") is None


def test_broken_run_resolution_degrades_to_unknown_run(monkeypatch, registry):
    from gui.services import progress_service

    def _boom(_rid):
        raise RuntimeError("state read exploded")

    _install(monkeypatch, live=(), running=())
    monkeypatch.setattr(progress_service, "run_progress", _boom)
    assert seat_service.seat_overview("WF-20260722-abcdef01") is None


# --------------------------------------------------------------- TTL cache
def test_ttl_cache_serves_repeat_calls_from_one_probe_round(monkeypatch, registry):
    log = []
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A,), probe_log=log)
    seat_service.seat_overview()
    seat_service.seat_overview()
    seat_service.seat_overview()
    assert len(log) == 2, f"expected ONE probe round (2 seats), got {log}"


def test_dashboard_and_build_page_share_one_probe_round(
    monkeypatch, registry, canned_workflows
):
    """The always-on monitor and a build page must not double the probe rate."""
    log = []
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A,), probe_log=log)
    _stub_run(monkeypatch, phase="needs_seat")
    seat_service.seat_overview()  # dashboard poll
    seat_service.seat_overview("WF-20260722-abcdef01")  # build-page poll
    assert len(log) == 2, f"the run-scoped view must reuse the shared cache: {log}"


def test_force_refresh_reprobes(monkeypatch, registry):
    log = []
    _install(monkeypatch, live=(), running=(), probe_log=log)
    seat_service.seat_overview()
    seat_service.seat_overview(force_refresh=True)
    assert len(log) == 4


def test_expired_cache_reprobes(monkeypatch, registry):
    log = []
    _install(monkeypatch, live=(), running=(), probe_log=log)
    seat_service.seat_overview()
    monkeypatch.setattr(seat_service, "CACHE_TTL_SECONDS", -1.0)
    seat_service.seat_overview()
    assert len(log) == 4


def test_run_scoped_payload_never_mutates_the_shared_cache(
    monkeypatch, registry, canned_workflows
):
    _install(monkeypatch, live=(), running=())
    _stub_run(monkeypatch, phase="needs_seat")
    seat_service.seat_overview("WF-20260722-abcdef01")
    globalp = seat_service.seat_overview()
    assert "expected" not in globalp["seats"][0], (
        "the run-scoped copy must not leak `expected` into the global payload"
    )
    assert "mismatch" not in globalp


# ------------------------------------------------------- transition tracking
def test_first_observation_reports_no_age(monkeypatch, registry):
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A,))
    payload = seat_service.seat_overview()
    for seat in payload["seats"]:
        assert seat["since"] is None and seat["since_ms"] is None


def test_unchanged_state_preserves_the_since_marker(monkeypatch, registry):
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A,))
    seat_service.seat_overview()
    marker = dict(seat_service._since[SEAT_A])
    seat_service.seat_overview(force_refresh=True)
    assert seat_service._since[SEAT_A]["iso"] == marker["iso"]
    assert seat_service._since[SEAT_A]["monotonic"] == marker["monotonic"]
    seat = seat_service.seat_overview(force_refresh=True)["seats"][0]
    assert seat["since"] == marker["iso"]
    assert isinstance(seat["since_ms"], int) and seat["since_ms"] >= 0


def test_state_change_resets_the_since_marker(monkeypatch, registry):
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A,))
    seat_service.seat_overview()
    first = dict(seat_service._since[SEAT_A])
    # The seat goes away mid-build — exactly the incident this panel exists for.
    _install(monkeypatch, live=(), running=())
    seat = seat_service.seat_overview(force_refresh=True)["seats"][0]
    assert seat["state"] == "down"
    assert seat["since_ms"] == 0, "a fresh transition starts its age at zero"
    assert seat_service._since[SEAT_A]["monotonic"] >= first["monotonic"]
    assert seat_service._since[SEAT_A]["state"] == "down"


def test_since_markers_are_pruned_when_a_seat_leaves_the_registry(monkeypatch):
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", f"{SEAT_A}={URL_A},{SEAT_B}={URL_B}")
    _install(monkeypatch, live=(), running=None)
    seat_service.seat_overview()
    assert set(seat_service._since) == {SEAT_A, SEAT_B}
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", f"{SEAT_A}={URL_A}")
    seat_service.seat_overview(force_refresh=True)
    assert set(seat_service._since) == {SEAT_A}


# ------------------------------------------------------------- never raises
def test_broken_registry_degrades_to_an_empty_overview(monkeypatch):
    import lib.vllm_container_lifecycle as vcl

    def _boom(*_a, **_kw):
        raise ValueError("registry exploded")

    monkeypatch.setattr(vcl, "parse_seat_registry", _boom)
    payload = seat_service.seat_overview()
    assert payload["seats"] == []
    assert payload["registry_configured"] is False
    assert "detail" in payload


def test_raising_probe_is_treated_as_not_serving(monkeypatch, registry):
    def _boom(_url):
        raise OSError("socket exploded")

    monkeypatch.setattr(seat_service, "_probe_seat", _boom)
    monkeypatch.setattr(seat_service, "_docker_running_names", lambda: None)
    payload = seat_service.seat_overview()
    assert {s["state"] for s in payload["seats"]} == {"unknown"}


def test_raising_container_registry_is_not_fatal(monkeypatch, registry):
    import lib.vllm_container_lifecycle as vcl

    monkeypatch.setattr(
        vcl, "parse_container_registry", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("x"))
    )
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A,))
    payload = seat_service.seat_overview()
    assert payload["seats"][0]["state"] == "live"
    assert payload["seats"][1]["state"] == "unknown"  # no container map → unknown


# =========================================================== router surface
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402


@pytest.fixture
def client(state_dir):
    return TestClient(create_app())


def test_get_api_seats_shape(client, monkeypatch, registry):
    _install(monkeypatch, live=(URL_A,), running=(CONTAINER_A,))
    resp = client.get("/api/seats")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [s["name"] for s in body["seats"]] == [SEAT_A, SEAT_B]
    assert set(body["seats"][0]) >= {
        "name", "base_url", "live", "state", "container", "model", "since", "since_ms",
    }
    assert "mismatch" not in body, "the global view carries no run-scoped keys"


def test_get_api_seats_run_shape(client, monkeypatch, registry, canned_workflows):
    _install(monkeypatch, live=(), running=())
    _stub_run(monkeypatch, phase="needs_seat")
    resp = client.get("/api/seats/run/WF-20260722-abcdef01")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["current_phase"] == "needs_seat"
    assert body["expected_seats"] == [SEAT_A]
    assert [m["seat"] for m in body["mismatch"]] == [SEAT_A]


def test_get_api_seats_run_unknown_is_typed_404(client, monkeypatch, registry):
    from gui.services import progress_service

    _install(monkeypatch, live=(), running=())
    monkeypatch.setattr(progress_service, "run_progress", lambda rid: None)
    resp = client.get("/api/seats/run/WF-does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"] == "run_not_found"


def test_router_failure_maps_to_typed_500(client, monkeypatch):
    monkeypatch.setattr(
        seat_service,
        "seat_overview",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    resp = client.get("/api/seats")
    assert resp.status_code == 500
    assert resp.json()["error"] == "seats_failed"


def test_seats_mounted_in_studio_not_learner(monkeypatch, state_dir, registry):
    _install(monkeypatch, live=(), running=())
    studio = TestClient(create_app(mode="studio"))
    assert studio.get("/api/seats").status_code == 200
    learner = TestClient(create_app(mode="learner"))
    assert learner.get("/api/seats").status_code == 404


def test_seats_endpoint_is_open_when_the_token_gate_is_armed(monkeypatch, state_dir, registry):
    """Parity with /api/health: the Studio shell polls it, so it stays open."""
    from gui import auth

    _install(monkeypatch, live=(), running=())
    monkeypatch.setenv("ED4ALL_GUI_TOKEN", "secret-token")
    assert auth.is_operator_path("/api/seats") is False
    client = TestClient(create_app())
    assert client.get("/api/seats").status_code == 200


# ==================================================== static-asset contract
def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_seat_js_is_registry_driven_and_names_no_seat():
    js = _read(SEATS_JS)
    assert "/api/seats" in js and "/api/seats/run/" in js
    for token in ("spark-super", "spark-nano", "spark-glm", "spark-qwen"):
        assert token not in js, f"seats.js must not hardcode the seat name {token!r}"
    for token in ("spark-super", "spark-nano"):
        assert token not in _read(REPO_ROOT / "gui" / "services" / "seat_service.py")


def test_seat_js_renders_state_as_text_not_color_only():
    js = _read(SEATS_JS)
    # Every state maps to a plain-language WORD rendered as text beside the pill.
    for word in ("serving", "starting up", "down", "unknown"):
        assert f"'{word}'" in js or f'"{word}"' in js
    assert "statusPill(" in js, "chips must reuse the shared pill kit"
    assert "seat-chip-state" in js and "seat-chip-name" in js
    # The mismatch line is a full sentence with a glyph, not a colored bar.
    assert "needs ${names}" in js and "not serving" in js
    assert "aria-hidden" in js


def test_seat_js_polls_and_stops():
    js = _read(SEATS_JS)
    assert "SEAT_POLL_MS = 15000" in js
    assert "setInterval(pollOnce, SEAT_POLL_MS)" in js
    assert "clearInterval(timer)" in js
    assert "SEAT_TERMINAL" in js, "a terminal run must latch and stop polling"


def test_create_js_mounts_the_seat_strip_on_both_progress_paths():
    js = _read(CREATE_JS)
    assert "import { mountSeatMonitor } from '/studio/seats.js';" in js
    # The real invariant: the seat strip is MOUNTED on both progress paths — the
    # full WS-console path (renderProgress) and the degraded CLI-run path
    # (renderCliRunProgress) — exactly once per path. That mount count is the
    # robust signal (start()/stop() below are called per-branch, so their raw
    # count over-counts renderProgress's two mutually-exclusive branches).
    assert js.count("mountSeatMonitor({ runId })") == 2, (
        "both the full and the degraded CLI-run progress paths mount the strip"
    )
    # Every mounted strip is started and every teardown stops its poll.
    # renderProgress starts/stops the monitor inside EACH of its two mutually-
    # exclusive branches (already-terminal refresh vs live WS stream), so the raw
    # call count (3) exceeds the mount count (2) — but starts and stops stay
    # balanced, and there is at least one of each per path.
    n_start = js.count("seatCtl.start()")
    n_stop = js.count("seatCtl.stop()")
    assert n_start == n_stop >= 2, (
        "seat poll start/stop must stay balanced and cover both paths "
        f"(start={n_start}, stop={n_stop}); both teardowns must stop the poll"
    )
    assert "v.appendChild(seatCtl.el)" in js


def test_dashboard_mounts_the_always_on_seat_section_and_tears_it_down():
    js = _read(DASHBOARD_JS)
    assert "import { renderSeatSection } from '/studio/seats.js';" in js
    # Mounted on BOTH dashboard paths: the populated view and the onboarding
    # (nothing built yet) view — the monitor must render with no build running.
    assert js.count("renderSeatSection()") == 2
    assert js.count(".stop()") == 2, "each path must return a teardown that stops the poll"
    assert "return () => seatsSection.stop();" in js
    assert "return () => seatsOnboarding.stop();" in js


def test_create_js_renders_seat_swap_activity_on_the_rail():
    """The stage rail surfaces an in-progress seat swap as its OWN annotation.

    text-not-color-only (a plain-language sentence + a glyph), applied after
    every rail.update() alongside the pages stat, and distinguishing the
    PATIENT cold-start from the ALARMING mismatch.
    """
    js = _read(CREATE_JS)
    assert "import { fmtAge } from '/studio/seats.js';" in js  # reuse the age fmt
    assert "function applySeatActivity(" in js
    assert "stats.seat_activity" in js
    assert "applySeatActivity(rail.el, p);" in js, "must run every rail poll"
    # Patient vs alarming both rendered as text (never color alone).
    assert "Swapping model seats" in js
    assert "not coming up" in js
    assert "stage-seatwait-concern" in js
    assert "aria-hidden" in js  # the glyph is decorative, the sentence carries it


def test_seat_swap_css_is_present_and_a_redundant_cue():
    css = _read(STUDIO_CSS)
    for cls in (".stage-seatwait", ".stage-seatwait.stage-seatwait-concern", ".stage-seatwait-glyph"):
        assert cls in css, f"{cls} must be styled"


def test_seat_css_states_are_a_redundant_cue_only():
    css = _read(STUDIO_CSS)
    for cls in (".seat-monitor", ".seat-strip", ".seat-chip", ".seat-mismatch", ".seat-note"):
        assert cls in css, f"{cls} must be styled"
    for state in ("live", "loading", "down", "unknown"):
        assert f'.seat-chip[data-state="{state}"]' in css
    # No display:none anywhere in the seat rules (a state must never be hidden).
    start = css.index(".seat-monitor")
    assert "display: none" not in css[start:start + 2000]


def test_seat_strip_markup_passes_the_wcag_gate():
    """Reconstruct the chip list exactly as seats.js builds it and gate it."""
    from lib.validators.wcag import IssueSeverity, WCAGValidator

    html = (
        '<!DOCTYPE html><html lang="en"><head><title>Build progress</title></head>'
        '<body><main><h1>Building your course</h1>'
        '<section class="seat-monitor" aria-label="Model seat status">'
        '<p class="seat-mismatch"><span class="seat-mismatch-glyph" aria-hidden="true">&#9888; </span>'
        '<span>Phase needs_seat needs atlas-large — not serving (atlas-large: down). '
        'A dispatch to a stopped seat fails the build.</span></p>'
        '<ul class="seat-strip" aria-label="Model seat status">'
        '<li class="seat-chip seat-chip-missing" data-state="down" data-seat="atlas-large">'
        '<span class="pill pill-fail"><span class="pill-glyph" aria-hidden="true">&#10007;</span>'
        '<span class="pill-label">Failed</span></span>'
        '<span class="seat-chip-name">atlas-large</span>'
        '<span class="seat-chip-state">down &middot; 3m</span>'
        '<span class="seat-chip-flag">needed now</span></li>'
        '<li class="seat-chip" data-state="live" data-seat="pebble-small">'
        '<span class="pill pill-pass"><span class="pill-glyph" aria-hidden="true">&#10003;</span>'
        '<span class="pill-label">OK</span></span>'
        '<span class="seat-chip-name">pebble-small</span>'
        '<span class="seat-chip-state">serving &middot; 12m</span></li>'
        '</ul>'
        '<p class="seat-note muted">Ages are since this server first saw the current state.</p>'
        '</section></main></body></html>'
    )
    report = WCAGValidator().validate(html)
    blocking = [
        i for i in report.issues
        if i.severity in {IssueSeverity.CRITICAL, IssueSeverity.HIGH}
    ]
    assert not blocking, [f"{i.criterion}: {i.message}" for i in blocking]
