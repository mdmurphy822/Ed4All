"""Unit tests for :mod:`lib.diagnostics.serving_window`.

Hermetic + network-free: ``httpx`` is replaced via ``sys.modules`` with a fake
client (the probe lazily imports it), or the probe itself is monkeypatched, so
the suite never touches an ollama server. Asserts the ``/api/show`` SERVED-window
resolution (Modelfile ``num_ctx`` → ``OLLAMA_CONTEXT_LENGTH`` → unset/ollama
default — the architectural ``model_info.context_length`` ceiling is reported
informationally and NEVER used as the served number), the assumed-vs-served
comparison logic, the INFO architectural-ceiling + detector-only notes, the
short-timeout contract, and the never-raises degraded path.
"""

from __future__ import annotations

import sys

import pytest

from lib.diagnostics import serving_window as sw
from lib.diagnostics.core import (
    CheckContext,
    Severity,
    clear_registry,
    registered_checks,
    resolve_exit_code,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.fixture(autouse=True)
def _clean_num_ctx_env(monkeypatch):
    # Pin the three assumed budgets to their documented defaults unless a test
    # overrides them, so comparisons are deterministic. OLLAMA_CONTEXT_LENGTH is
    # cleared so the served-window resolution is deterministic too.
    for var in (
        "ED4ALL_REWRITE_NUM_CTX",
        "ED4ALL_ANSWER_NUM_CTX",
        "ED4ALL_CONTENT_PAGE_NUM_CTX",
        "OLLAMA_CONTEXT_LENGTH",
        "LOCAL_SYNTHESIS_MODEL",
        "LOCAL_SYNTHESIS_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# --------------------------------------------------------------------------- #
# Fake httpx scaffolding
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, body, status_ok=True):
        self._body = body
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP 404 model not found")

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeClient:
    """Records the timeout + the POST call; returns a canned response."""

    last_timeout = None
    last_url = None
    last_json = None

    def __init__(self, timeout=None):
        _FakeClient.last_timeout = timeout
        self._response = _FakeClient._response
        self._raise_on_post = _FakeClient._raise_on_post

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None):
        _FakeClient.last_url = url
        _FakeClient.last_json = json
        if _FakeClient._raise_on_post is not None:
            raise _FakeClient._raise_on_post
        return self._response


def _install_fake_httpx(monkeypatch, response=None, raise_on_post=None):
    _FakeClient._response = response
    _FakeClient._raise_on_post = raise_on_post
    _FakeClient.last_timeout = None
    _FakeClient.last_url = None
    _FakeClient.last_json = None
    fake_module = type(sys)("httpx")
    fake_module.Client = _FakeClient
    monkeypatch.setitem(sys.modules, "httpx", fake_module)


def _body(*, num_ctx=None, ceiling=None, arch="qwen2", extra_params="stop <|im_end|>"):
    """Build a fake ``/api/show`` response body.

    ``num_ctx`` (when given) is baked into the ``parameters`` newline-string
    (the Modelfile override). ``ceiling`` (when given) is the architectural
    ``<arch>.context_length`` in ``model_info``.
    """
    model_info = {"general.architecture": arch}
    if ceiling is not None:
        model_info[f"{arch}.context_length"] = ceiling
        model_info[f"{arch}.embedding_length"] = 3584
    params_lines = []
    if num_ctx is not None:
        params_lines.append(f"num_ctx {num_ctx}")
    if extra_params:
        params_lines.append(extra_params)
    return _FakeResponse(
        {"model_info": model_info, "parameters": "\n".join(params_lines)}
    )


# --------------------------------------------------------------------------- #
# probe_served_window — served vs. ceiling resolution
# --------------------------------------------------------------------------- #


def test_probe_parameters_num_ctx_wins_over_arch_ceiling(monkeypatch):
    # Modelfile num_ctx=8192 present AND arch ceiling=32768 → served=8192 (the
    # Modelfile override wins; the ceiling is reported separately).
    _install_fake_httpx(monkeypatch, response=_body(num_ctx=8192, ceiling=32768))
    probe = sw.probe_served_window(model="qwen2.5:7b")
    assert probe.served == 8192
    assert probe.ceiling == 32768
    assert probe.source == sw._SOURCE_PARAMETERS
    assert probe.model == "qwen2.5:7b"
    assert probe.error is None
    # POST hit /api/show with the model name in the body.
    assert _FakeClient.last_url.endswith("/api/show")
    assert _FakeClient.last_json == {"name": "qwen2.5:7b"}


def test_probe_arch_ceiling_is_not_the_served_window(monkeypatch):
    # CORE REGRESSION GUARD: model_info ceiling=32768 but Modelfile num_ctx=4096
    # → served=4096 (NOT 32768). Reading the ceiling as served was the bug.
    _install_fake_httpx(monkeypatch, response=_body(num_ctx=4096, ceiling=32768))
    probe = sw.probe_served_window(model="m")
    assert probe.served == 4096
    assert probe.ceiling == 32768
    assert probe.source == sw._SOURCE_PARAMETERS
    assert probe.error is None


def test_probe_falls_back_to_ollama_context_length_env(monkeypatch):
    # No Modelfile num_ctx, but OLLAMA_CONTEXT_LENGTH=8192 env → served=8192.
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "8192")
    _install_fake_httpx(monkeypatch, response=_body(num_ctx=None, ceiling=32768))
    probe = sw.probe_served_window(model="m")
    assert probe.served == 8192
    assert probe.source == sw._SOURCE_ENV
    assert probe.ceiling == 32768
    assert probe.error is None


def test_probe_unset_when_neither_num_ctx_nor_env(monkeypatch):
    # No Modelfile num_ctx + no OLLAMA_CONTEXT_LENGTH → served unset (the model
    # serves ollama's built-in default). NOT a probe error.
    _install_fake_httpx(monkeypatch, response=_body(num_ctx=None, ceiling=32768))
    probe = sw.probe_served_window(model="m")
    assert probe.served is None
    assert probe.source == sw._SOURCE_UNSET
    assert probe.ceiling == 32768
    assert probe.error is None


def test_probe_num_ctx_dict_form(monkeypatch):
    # parameters as a dict (alternative ollama shape).
    body = _FakeResponse(
        {
            "model_info": {"qwen2.context_length": 32768},
            "parameters": {"num_ctx": 16384, "temperature": 0.7},
        }
    )
    _install_fake_httpx(monkeypatch, response=body)
    probe = sw.probe_served_window(model="m")
    assert probe.served == 16384
    assert probe.source == sw._SOURCE_PARAMETERS
    assert probe.ceiling == 32768


def test_probe_garbage_env_treated_as_unset(monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "not-a-number")
    _install_fake_httpx(monkeypatch, response=_body(num_ctx=None, ceiling=32768))
    probe = sw.probe_served_window(model="m")
    assert probe.served is None
    assert probe.source == sw._SOURCE_UNSET


def test_probe_short_timeout_used(monkeypatch):
    _install_fake_httpx(monkeypatch, response=_body(num_ctx=8192, ceiling=32768))
    sw.probe_served_window(model="m")
    assert _FakeClient.last_timeout == sw._API_SHOW_TIMEOUT_SECONDS
    assert sw._API_SHOW_TIMEOUT_SECONDS <= 2.0


# --------------------------------------------------------------------------- #
# probe_served_window — failure paths never raise
# --------------------------------------------------------------------------- #


def test_probe_non_200_returns_error(monkeypatch):
    _install_fake_httpx(monkeypatch, response=_FakeResponse({}, status_ok=False))
    probe = sw.probe_served_window(model="m")
    assert probe.served is None
    assert probe.model == "m"
    assert probe.error  # a reason string
    assert probe.source is None


def test_probe_thrown_httpx_error_does_not_raise(monkeypatch):
    _install_fake_httpx(
        monkeypatch, raise_on_post=ConnectionError("connection refused")
    )
    probe = sw.probe_served_window(model="m")
    assert probe.served is None
    assert "connection refused" in probe.error


def test_probe_bad_json_returns_error(monkeypatch):
    _install_fake_httpx(monkeypatch, response=_FakeResponse(ValueError("not json")))
    probe = sw.probe_served_window(model="m")
    assert probe.served is None
    assert probe.error


def test_probe_non_dict_body_is_error(monkeypatch):
    _install_fake_httpx(monkeypatch, response=_FakeResponse(["not", "a", "dict"]))
    probe = sw.probe_served_window(model="m")
    assert probe.served is None
    assert probe.error


def test_probe_resolves_default_model_when_unset(monkeypatch):
    _install_fake_httpx(monkeypatch, response=_body(num_ctx=8192, ceiling=32768))
    probe = sw.probe_served_window()
    assert probe.served == 8192
    assert probe.model == sw._DEFAULT_LOCAL_MODEL


# --------------------------------------------------------------------------- #
# serving_window_checks — comparison logic
# --------------------------------------------------------------------------- #


def _patch_probe(monkeypatch, *, served, ceiling=32768, source=None, model="qwen2.5:7b", error=None):
    if source is None and error is None:
        if served is None:
            source = sw._SOURCE_UNSET
        else:
            source = sw._SOURCE_PARAMETERS
    probe = sw.ServedWindow(served, ceiling, source, model, error)
    monkeypatch.setattr(
        sw, "probe_served_window", lambda base_url=None, model=None: probe
    )


def test_warn_when_assumed_above_served_modelfile(monkeypatch):
    # Modelfile num_ctx=8192 (served) even though arch ceiling=32768; rewrite
    # assumes 16384 → truncation WARN, and the ceiling INFO mentions 32768.
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "16384")
    _patch_probe(monkeypatch, served=8192, ceiling=32768, source=sw._SOURCE_PARAMETERS)
    results = sw.serving_window_checks(CheckContext())
    rewrite = next(r for r in results if r.name == "serving_window_fit_rewrite")
    assert rewrite.severity is Severity.WARN
    assert "16384" in rewrite.summary
    assert "8192" in rewrite.summary
    assert "HEAD-truncated" in rewrite.summary
    assert rewrite.remediation
    assert rewrite.data["assumed"] == 16384
    assert rewrite.data["effective_served_window"] == 8192

    ceiling = next(r for r in results if r.name == "serving_window_ceiling")
    assert ceiling.severity is Severity.INFO
    assert "32768" in ceiling.summary
    assert ceiling.data["architectural_ceiling"] == 32768


def test_env_served_window_used(monkeypatch):
    # OLLAMA_CONTEXT_LENGTH path: served=8192 from env, source flagged.
    _patch_probe(monkeypatch, served=8192, ceiling=32768, source=sw._SOURCE_ENV)
    results = sw.serving_window_checks(CheckContext())
    info = next(r for r in results if r.name == "serving_window_served")
    assert info.severity is Severity.OK
    assert info.data["served_window"] == 8192
    assert info.data["served_source"] == sw._SOURCE_ENV


def test_unset_served_window_warns_and_compares_against_4096(monkeypatch):
    # No num_ctx + no env → served unset → ollama default ~4096. rewrite=8192
    # assumed > 4096 → the previously-MASKED truncation WARN fires.
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "8192")
    _patch_probe(monkeypatch, served=None, ceiling=32768, source=sw._SOURCE_UNSET)
    results = sw.serving_window_checks(CheckContext())

    unset = next(r for r in results if r.name == "serving_window_unset_default")
    assert unset.severity is Severity.WARN
    assert "4096" in unset.summary
    assert unset.data["effective_served_window"] == sw._OLLAMA_DEFAULT_CONTEXT
    assert unset.data["served_window"] is None

    rewrite = next(r for r in results if r.name == "serving_window_fit_rewrite")
    assert rewrite.severity is Severity.WARN
    assert "8192" in rewrite.summary
    assert rewrite.data["effective_served_window"] == 4096
    # No OK "served" info result in the unset case.
    assert not any(r.name == "serving_window_served" for r in results)


def test_all_ok_when_assumed_within_served(monkeypatch):
    # Defaults: rewrite 8192, answer 4096, content_page 4096 — all <= 32768.
    _patch_probe(monkeypatch, served=32768, ceiling=32768, source=sw._SOURCE_PARAMETERS)
    results = sw.serving_window_checks(CheckContext())
    fit_results = [r for r in results if r.name.startswith("serving_window_fit_")]
    assert fit_results  # all three seats compared
    assert all(r.severity is Severity.OK for r in fit_results)
    # Served > 8192 → no detector-only note.
    assert not any(r.name == "serving_window_detector_only" for r in results)
    info = next(r for r in results if r.name == "serving_window_served")
    assert info.severity is Severity.OK
    assert info.data["served_window"] == 32768


def test_served_info_result_emitted(monkeypatch):
    _patch_probe(monkeypatch, served=8192, ceiling=32768, source=sw._SOURCE_PARAMETERS)
    results = sw.serving_window_checks(CheckContext())
    info = next(r for r in results if r.name == "serving_window_served")
    assert info.severity is Severity.OK
    assert "8192" in info.summary
    assert info.data["assumed"]  # dict of seat → assumed budget


# --------------------------------------------------------------------------- #
# Finding #7 — detector-only note is INFO, not WARN
# --------------------------------------------------------------------------- #


def test_detector_only_note_is_info(monkeypatch):
    # Healthy answer-only deployment: served=8192, all default budgets fit
    # (rewrite default 8192, answer/content_page 4096). The detector note must
    # be INFO and must NOT push the exit code up.
    _patch_probe(monkeypatch, served=8192, ceiling=32768, source=sw._SOURCE_PARAMETERS)
    results = sw.serving_window_checks(CheckContext())
    note = next(r for r in results if r.name == "serving_window_detector_only")
    assert note.severity is Severity.INFO
    assert "detector-only" in note.summary
    assert "ED4ALL_REWRITE_FIT_WINDOW" in note.detail
    # No truncation WARN at the default budgets → INFO note keeps exit at 0.
    assert not any(
        r.name.startswith("serving_window_fit_") and r.severity is Severity.WARN
        for r in results
    )
    assert resolve_exit_code(results) == 0


def test_detector_only_note_info_below_8192(monkeypatch):
    _patch_probe(monkeypatch, served=4096, ceiling=32768, source=sw._SOURCE_PARAMETERS)
    results = sw.serving_window_checks(CheckContext())
    note = next(r for r in results if r.name == "serving_window_detector_only")
    assert note.severity is Severity.INFO


def test_detector_note_does_not_inflate_exit_over_truncation(monkeypatch):
    # With a real truncation WARN present the exit is 1 (from the WARN), and the
    # INFO detector note neither adds to nor masks that.
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "16384")
    _patch_probe(monkeypatch, served=8192, ceiling=32768, source=sw._SOURCE_PARAMETERS)
    results = sw.serving_window_checks(CheckContext())
    assert any(
        r.name == "serving_window_fit_rewrite" and r.severity is Severity.WARN
        for r in results
    )
    assert resolve_exit_code(results) == 1


# --------------------------------------------------------------------------- #
# serving_window_checks — unknown window WARN path
# --------------------------------------------------------------------------- #


def test_unknown_window_single_warn(monkeypatch):
    _patch_probe(
        monkeypatch, served=None, ceiling=None, source=None, model="qwen2.5:7b",
        error="ollama down",
    )
    results = sw.serving_window_checks(CheckContext())
    assert len(results) == 1
    warn = results[0]
    assert warn.name == "serving_window_unknown"
    assert warn.severity is Severity.WARN
    assert "could not determine served window" in warn.summary
    assert "qwen2.5:7b" in warn.summary
    assert "ollama" in warn.remediation.lower()
    assert "LOCAL_SYNTHESIS_BASE_URL" in warn.remediation


def test_unknown_window_via_down_server_does_not_raise(monkeypatch):
    # End-to-end through the real probe with a down server (raise on post).
    _install_fake_httpx(
        monkeypatch, raise_on_post=ConnectionError("connection refused")
    )
    results = sw.serving_window_checks(CheckContext())
    assert len(results) == 1
    assert results[0].name == "serving_window_unknown"
    assert results[0].severity is Severity.WARN


# --------------------------------------------------------------------------- #
# Registration — no import-time side effect
# --------------------------------------------------------------------------- #


def test_register_serving_window_checks_no_import_side_effect():
    assert registered_checks() == []
    sw.register_serving_window_checks()
    pairs = registered_checks()
    assert [g for g, _ in pairs] == ["window"]
    assert pairs[0][1] is sw.serving_window_checks


def test_checks_never_raise_on_probe_exception(monkeypatch):
    # Belt-and-suspenders: even if the probe somehow raises, the public
    # serving_window_checks contract is no-raise — it degrades to a WARN.
    def boom(base_url=None, model=None):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(sw, "probe_served_window", boom)
    results = sw.serving_window_checks(CheckContext())  # must NOT raise
    assert len(results) == 1
    assert results[0].name == "serving_window_error"
    assert results[0].severity is Severity.WARN
    assert "unexpected" in results[0].summary


# --------------------------------------------------------------------- #
# P0-1: vLLM-seat topology branch — no ollama /api/show "unknown" WARN
# --------------------------------------------------------------------- #


class _Topo:
    def __init__(self, backend, base_url_root="http://localhost:8001", seat_name="spark-super"):
        self.backend = backend
        self.base_url_root = base_url_root
        self.seat_name = seat_name
        self.seat_registry_configured = backend == "vllm"


def test_serving_window_vllm_branch_info_no_warn(monkeypatch) -> None:
    monkeypatch.setattr(
        "lib.diagnostics.run_env.resolve_local_synthesis_topology",
        lambda: _Topo("vllm"),
    )
    monkeypatch.setattr(
        "lib.diagnostics.run_env.probe_v1_models",
        lambda root, **k: (True, ["super-120b"], None),
    )

    def _fail(*a, **k):
        raise AssertionError("/api/show must NOT be probed on a vLLM host")

    monkeypatch.setattr(sw, "probe_served_window", _fail)

    results = sw.serving_window_checks(CheckContext())
    names = {r.name for r in results}
    assert names == {"serving_window_vllm_seat"}
    res = results[0]
    assert res.severity is Severity.INFO
    assert res.data["backend"] == "vllm"
    # No false DEGRADED.
    assert resolve_exit_code(results) == 0


def test_serving_window_ollama_backend_takes_legacy_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "lib.diagnostics.run_env.resolve_local_synthesis_topology",
        lambda: _Topo("ollama"),
    )
    # Legacy path: probe returns a served window; no vllm result emitted.
    monkeypatch.setattr(
        sw,
        "probe_served_window",
        lambda base_url, model: sw.ServedWindow(32768, 32768, sw._SOURCE_PARAMETERS, "m", None),
    )
    monkeypatch.setattr(sw, "_resolve_assumed_budgets", lambda: [])
    results = sw.serving_window_checks(CheckContext())
    names = {r.name for r in results}
    assert "serving_window_vllm_seat" not in names
    assert "serving_window_served" in names
