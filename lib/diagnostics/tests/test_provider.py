"""Tests for the ``provider`` doctor group (:mod:`lib.diagnostics.provider`).

GPU-free, no real network: ``resolve_seats`` / ``key_status`` /
``applied_run_env`` / the ping client are all monkeypatched. We patch the
engine on the ``run_env`` MODULE (``lib.diagnostics.provider`` calls
``run_env.<fn>`` at call time, the same monkeypatch-honoring pattern
``environment.py`` uses for ``serving_window``), and the ping client via
``provider._make_ping_client``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional

import pytest

from lib.diagnostics import provider as prov
from lib.diagnostics import run_env
from lib.diagnostics.core import CheckContext, Severity
from lib.diagnostics.run_env import SeatResolution


# --------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------- #


def _seat(
    seat: str,
    effective_provider: Optional[str],
    *,
    source: str = "env",
    model: Optional[str] = None,
    provider_env: str = "SOME_PROVIDER",
    note: str = "",
) -> SeatResolution:
    return SeatResolution(
        seat=seat,
        provider_env=provider_env,
        effective_provider=effective_provider,
        model=model,
        source=source,
        note=note,
    )


def _key_table() -> Dict[str, Dict[str, Any]]:
    return {
        "nvidia": {
            "key_env": "NVIDIA_API_KEY",
            "key_required": True,
            "base_url_env": "NVIDIA_BASE_URL",
            "base_url_default": "https://integrate.api.nvidia.com/v1",
            "model_env": "NVIDIA_LARGE_MODEL",
            "default_model": "nvidia/nemotron",
        },
        "anthropic": {
            "key_env": "ANTHROPIC_API_KEY",
            "key_required": True,
            "base_url_env": None,
            "base_url_default": None,
            "model_env": "ANTHROPIC_SYNTHESIS_MODEL",
            "default_model": "claude-sonnet-4-6",
        },
        "local": {
            "key_env": "LOCAL_SYNTHESIS_API_KEY",
            "key_required": False,
            "base_url_env": "LOCAL_SYNTHESIS_BASE_URL",
            "base_url_default": "http://localhost:11434/v1",
            "model_env": "LOCAL_SYNTHESIS_MODEL",
            "default_model": "qwen2.5:7b",
        },
        "mock": {
            "key_env": None,
            "key_required": False,
            "base_url_env": None,
            "base_url_default": None,
            "model_env": None,
            "default_model": None,
        },
    }


def _key_status_factory(present: Dict[str, bool]):
    """Build a ``key_status`` stub from a {provider: key_present} map."""
    table = _key_table()

    def _key_status(p: str, key_table=None) -> Dict[str, Any]:
        row = table.get(p, {})
        required = bool(row.get("key_required", False))
        return {
            "provider": p,
            "key_env": row.get("key_env"),
            "key_required": required,
            "key_present": present.get(p, not required),
            "base_url": row.get("base_url_default"),
        }

    return _key_status


@pytest.fixture
def patch_engine(monkeypatch):
    """Return a configurator that wires resolve_seats / key_status / fanout."""

    def _configure(seats: List[SeatResolution], present: Dict[str, bool]):
        @contextlib.contextmanager
        def _noop_fanout(workflow, provider_hint=""):
            yield {}

        monkeypatch.setattr(run_env, "applied_run_env", _noop_fanout)
        monkeypatch.setattr(run_env, "resolve_seats", lambda two_pass: list(seats))
        monkeypatch.setattr(run_env, "load_provider_key_table", _key_table)
        monkeypatch.setattr(run_env, "key_status", _key_status_factory(present))

    return _configure


def _by_name(results) -> Dict[str, Any]:
    return {r.name: r for r in results}


def _run_ctx(**kw) -> CheckContext:
    rc = {"workflow": "textbook_to_course", "provider_hint": "", "mode": "local", "ping": False}
    rc.update(kw)
    return CheckContext(run_config=rc)


def _bare_ctx(**kw) -> CheckContext:
    rc = {"provider_hint": "", "mode": "local", "ping": False}
    rc.update(kw)
    return CheckContext(run_config=rc)


# --------------------------------------------------------------------- #
# RUN mode
# --------------------------------------------------------------------- #


def test_run_mode_missing_required_key_fails(patch_engine):
    patch_engine(
        [_seat("content", "nvidia", source="env")],
        present={"nvidia": False},
    )
    results = prov.provider_checks(_run_ctx())
    fails = [r for r in results if r.severity is Severity.FAIL]
    assert any(r.name == "seat_content_key_missing" for r in fails)
    fail = next(r for r in fails if r.name == "seat_content_key_missing")
    assert "NVIDIA_API_KEY" in fail.summary
    assert "export NVIDIA_API_KEY" in fail.remediation
    # the resolution INFO line is still present alongside the FAIL
    assert "seat_content" in _by_name(results)


def test_run_mode_split_brain_warns(patch_engine):
    patch_engine(
        [
            _seat("content", "nvidia", source="env"),
            _seat("textbook_synthesis", "local", source="fanout"),
            _seat("training_synthesis", "mock", source="class_default"),
        ],
        present={"nvidia": True},  # key present → no key FAIL noise
    )
    results = prov.provider_checks(_run_ctx())
    sb = _by_name(results).get("provider_split_brain")
    assert sb is not None
    assert sb.severity is Severity.WARN
    assert "authoring=nvidia" in sb.summary
    assert "textbook-synthesis=local" in sb.summary
    # #2 — remediation scopes to the textbook-synthesis seat ONLY and NEVER
    # advises routing the training corpus through cloud.
    assert "TEXTBOOK_SYNTHESIS_PROVIDER=nvidia" in sb.remediation
    assert "TRAINFORGE_SYNTHESIS_PROVIDER=nvidia" not in sb.remediation
    assert "LLM_PROVIDER=nvidia" not in sb.remediation


def test_run_mode_nvidia_authoring_local_training_no_split_brain(patch_engine):
    """#2 — authoring nvidia + textbook follows + training local → no WARN.

    A correctly-configured `--provider nvidia` run pins training_synthesis
    local for licensing. With textbook_synthesis following authoring, there is
    NO split-brain at all, and nothing ever flags the (correct) local training
    seat as a problem.
    """
    patch_engine(
        [
            _seat("content", "nvidia", source="env"),
            _seat("textbook_synthesis", "nvidia", source="env"),
            _seat("training_synthesis", "local", source="fanout"),
        ],
        present={"nvidia": True},
    )
    results = prov.provider_checks(_run_ctx())
    assert "provider_split_brain" not in _by_name(results)
    # No remediation anywhere advises routing training through cloud.
    assert not any(
        "TRAINFORGE_SYNTHESIS_PROVIDER=nvidia" in (r.remediation or "")
        for r in results
    )


def test_run_mode_unset_class_default_content_not_cloud_authoring(patch_engine):
    """#8 — an unset class-default `content` seat is not a cloud authoring seat.

    Defaulting to anthropic via class_default means the legacy no-LLM TEMPLATE
    path runs (no anthropic call), so it must NOT trip split-brain even though
    textbook_synthesis is local.
    """
    patch_engine(
        [
            _seat("content", "anthropic", source="class_default"),
            _seat("textbook_synthesis", "local", source="fanout"),
        ],
        present={"anthropic": False},
    )
    results = prov.provider_checks(_run_ctx())
    assert "provider_split_brain" not in _by_name(results)


def test_run_mode_training_synthesis_nvidia_licensing_warn(patch_engine):
    patch_engine(
        [_seat("training_synthesis", "nvidia", source="env")],
        present={"nvidia": True},
    )
    results = prov.provider_checks(_run_ctx())
    lic = _by_name(results).get("provider_training_synthesis_licensing")
    assert lic is not None
    assert lic.severity is Severity.WARN
    assert "corpus taint" in lic.summary
    assert "TRAINFORGE_SYNTHESIS_PROVIDER=local" in lic.remediation


def test_run_mode_training_synthesis_local_licensing_ok(patch_engine):
    patch_engine(
        [_seat("training_synthesis", "local", source="fanout")],
        present={},
    )
    results = prov.provider_checks(_run_ctx())
    lic = _by_name(results).get("provider_training_synthesis_licensing")
    assert lic is not None
    assert lic.severity is Severity.OK


def test_run_mode_cloud_seat_preflight_mapping(patch_engine):
    patch_engine([_seat("content", "local", source="fanout")], present={})
    npf = {
        "checks": [
            {"level": "PASS", "name": "key_present", "detail": "NVIDIA_API_KEY set"},
            {"level": "WARN", "name": "rate_limit", "detail": "no ceiling configured"},
            {"level": "FAIL", "name": "model_reachable", "detail": "401 from endpoint"},
        ]
    }
    # Legacy run_config key — provider checks read new-then-old, so an old
    # post-mortem sidecar carrying ``nvidia_preflight`` still maps through.
    results = prov.provider_checks(_run_ctx(nvidia_preflight=npf))
    by_name = _by_name(results)
    assert by_name["cloud_seat_preflight_key_present"].severity is Severity.OK
    assert by_name["cloud_seat_preflight_rate_limit"].severity is Severity.WARN
    assert by_name["cloud_seat_preflight_model_reachable"].severity is Severity.FAIL
    assert by_name["cloud_seat_preflight_model_reachable"].detail == "401 from endpoint"


def test_run_mode_cloud_seat_preflight_real_levels_and_unknown(patch_engine):
    """#1 — the REAL lowercase levels run.py emits map correctly.

    `cli/commands/run.py::_cloud_seat_preflight` emits per-check `level` ∈
    {'pass', 'warn', 'error'} + a top-level 'verdict'. 'error' MUST escalate
    to FAIL (the old map keyed only 'FAIL' so 'error' fell through to INFO and
    a real cloud-seat failure never escalated the exit code). An UNKNOWN level
    maps to WARN, never INFO.
    """
    patch_engine([_seat("content", "local", source="fanout")], present={})
    npf = {
        "verdict": "FAIL",
        "note": "routing resolved + asserted only",
        "checks": [
            {"level": "pass", "name": "nvidia_api_key", "detail": "present"},
            {"level": "warn", "name": "answer_loopback", "detail": "verify"},
            {"level": "error", "name": "two_pass", "detail": "not truthy"},
            {"level": "surprise", "name": "novel_check", "detail": "new level"},
        ],
    }
    # New run_config key (the current CLI emits it).
    results = prov.provider_checks(_run_ctx(cloud_seat_preflight=npf))
    by_name = _by_name(results)
    assert by_name["cloud_seat_preflight_nvidia_api_key"].severity is Severity.OK
    assert by_name["cloud_seat_preflight_answer_loopback"].severity is Severity.WARN
    # the critical fix: a real 'error' escalates to FAIL, not INFO
    assert by_name["cloud_seat_preflight_two_pass"].severity is Severity.FAIL
    # an unrecognized level surfaces as WARN (never silently INFO)
    assert by_name["cloud_seat_preflight_novel_check"].severity is Severity.WARN
    # the top-level verdict is surfaced and mapped through the same table
    assert by_name["cloud_seat_preflight_verdict"].severity is Severity.FAIL


def test_run_mode_class_default_missing_key_not_fail(patch_engine):
    """#5 — a class-default seat with no key is INFO, not FAIL (run mode).

    A workflow with no corpus-gen fanout (rag_training-like) keeps `content`
    on its anthropic class default with no key — but it may never use it, so
    it must NOT FAIL. Only an EXPLICITLY-set seat with a missing key FAILs.
    """
    patch_engine(
        [
            _seat("content", "anthropic", source="class_default"),
            _seat("rewrite", "nvidia", source="env"),
        ],
        present={"anthropic": False, "nvidia": False},
    )
    results = prov.provider_checks(_run_ctx())
    # class-default anthropic with no key → no FAIL for that seat
    assert not any(r.name == "seat_content_key_missing" for r in results)
    assert _by_name(results)["seat_content"].severity is Severity.INFO
    # the EXPLICIT env-set nvidia seat with no key → FAIL
    assert any(
        r.name == "seat_rewrite_key_missing" and r.severity is Severity.FAIL
        for r in results
    )


def test_run_mode_present_required_key_folds_to_ok(patch_engine):
    patch_engine([_seat("content", "nvidia", source="env")], present={"nvidia": True})
    results = prov.provider_checks(_run_ctx())
    res = _by_name(results)["seat_content"]
    assert res.severity is Severity.OK
    assert not any(r.name == "seat_content_key_missing" for r in results)


# --------------------------------------------------------------------- #
# BARE mode
# --------------------------------------------------------------------- #


def test_bare_mode_class_default_anthropic_no_key_not_fail(patch_engine):
    patch_engine(
        [_seat("content", "anthropic", source="class_default")],
        present={"anthropic": False},
    )
    results = prov.provider_checks(_bare_ctx())
    # No FAIL — a class-default seat that a run would flip is not gating.
    assert not any(r.severity is Severity.FAIL for r in results)
    res = _by_name(results)["seat_content"]
    assert res.severity is Severity.INFO
    # the "use --run" caveat is surfaced
    assert any("ed4all doctor --run" in (r.detail or "") for r in results)
    assert "provider_bare_caveat" in _by_name(results)


def test_bare_mode_explicit_nvidia_no_key_fails(patch_engine):
    patch_engine(
        [_seat("content", "nvidia", source="env")],
        present={"nvidia": False},
    )
    results = prov.provider_checks(_bare_ctx())
    assert any(
        r.name == "seat_content_key_missing" and r.severity is Severity.FAIL
        for r in results
    )


def test_bare_mode_skips_split_brain_and_licensing(patch_engine):
    patch_engine(
        [
            _seat("content", "nvidia", source="env"),
            _seat("training_synthesis", "nvidia", source="env"),
        ],
        present={"nvidia": True},
    )
    results = prov.provider_checks(_bare_ctx())
    names = set(_by_name(results))
    assert "provider_split_brain" not in names
    assert "provider_training_synthesis_licensing" not in names


# --------------------------------------------------------------------- #
# Ping
# --------------------------------------------------------------------- #


class _FakeClient:
    def __init__(self, *, raise_exc: Optional[Exception] = None, **kwargs):
        self._raise = raise_exc
        self.kwargs = kwargs

    def chat_completion(self, messages, **kwargs):
        if self._raise is not None:
            raise self._raise
        return "pong"


def test_ping_success_ok(patch_engine, monkeypatch):
    patch_engine([_seat("content", "local", source="fanout")], present={})
    monkeypatch.setattr(prov, "_make_ping_client", lambda **kw: _FakeClient())
    results = prov.provider_checks(_run_ctx(ping=True))
    ping = _by_name(results).get("provider_ping_local")
    assert ping is not None
    assert ping.severity is Severity.OK


def test_ping_synthesis_error_fails(patch_engine, monkeypatch):
    from Trainforge.generators.providers._openai_compatible_client import SynthesisProviderError

    patch_engine([_seat("content", "local", source="fanout")], present={})

    def _client(**kw):
        return _FakeClient(raise_exc=SynthesisProviderError("401 unauthorized", code="401"))

    monkeypatch.setattr(prov, "_make_ping_client", _client)
    results = prov.provider_checks(_run_ctx(ping=True))
    ping = _by_name(results).get("provider_ping_local")
    assert ping is not None
    assert ping.severity is Severity.FAIL
    assert "401" in ping.summary


def test_ping_output_truncated_is_reachable_ok(patch_engine, monkeypatch):
    """#4 — output_truncated (tiny max_tokens) PROVES reachability → OK.

    A healthy server hits finish_reason='length' on a tiny-max_tokens ping and
    the client raises SynthesisProviderError(code='output_truncated'). That is
    NOT a failure — the request was accepted and generation started, so the
    ping must report OK, not FAIL.
    """
    from Trainforge.generators.providers._openai_compatible_client import SynthesisProviderError

    patch_engine([_seat("content", "local", source="fanout")], present={})

    def _client(**kw):
        return _FakeClient(
            raise_exc=SynthesisProviderError(
                "response finish_reason='length'", code="output_truncated"
            )
        )

    monkeypatch.setattr(prov, "_make_ping_client", _client)
    results = prov.provider_checks(_run_ctx(ping=True))
    ping = _by_name(results).get("provider_ping_local")
    assert ping is not None
    assert ping.severity is Severity.OK
    assert "reachable (auth accepted)" in ping.summary


def test_ping_auth_failure_fails(patch_engine, monkeypatch):
    """#4 — a genuine 401 auth failure is still a FAIL."""
    from Trainforge.generators.providers._openai_compatible_client import SynthesisProviderError

    patch_engine([_seat("content", "local", source="fanout")], present={})

    def _client(**kw):
        return _FakeClient(raise_exc=SynthesisProviderError("401 unauthorized", code="401"))

    monkeypatch.setattr(prov, "_make_ping_client", _client)
    results = prov.provider_checks(_run_ctx(ping=True))
    ping = _by_name(results).get("provider_ping_local")
    assert ping is not None
    assert ping.severity is Severity.FAIL


def test_ping_transport_error_fails(patch_engine, monkeypatch):
    """#4 — a transport/connection error (no `code`) is a FAIL."""
    patch_engine([_seat("content", "local", source="fanout")], present={})

    def _client(**kw):
        return _FakeClient(raise_exc=ConnectionError("connection refused"))

    monkeypatch.setattr(prov, "_make_ping_client", _client)
    results = prov.provider_checks(_run_ctx(ping=True))
    ping = _by_name(results).get("provider_ping_local")
    assert ping is not None
    assert ping.severity is Severity.FAIL


def test_ping_anthropic_is_info_not_pinged(patch_engine, monkeypatch):
    patch_engine([_seat("content", "anthropic", source="env")], present={"anthropic": True})
    calls = []
    monkeypatch.setattr(
        prov, "_make_ping_client", lambda **kw: calls.append(kw) or _FakeClient()
    )
    results = prov.provider_checks(_run_ctx(ping=True))
    ping = _by_name(results).get("provider_ping_anthropic")
    assert ping is not None
    assert ping.severity is Severity.INFO
    assert "not pingable" in ping.summary
    # anthropic uses the SDK — the OpenAI-compatible client is never built
    assert calls == []


def test_ping_not_attempted_when_falsey(patch_engine, monkeypatch):
    patch_engine([_seat("content", "local", source="fanout")], present={})
    calls = []
    monkeypatch.setattr(
        prov, "_make_ping_client", lambda **kw: calls.append(kw) or _FakeClient()
    )
    results = prov.provider_checks(_run_ctx(ping=False))
    # client never constructed when ping is off
    assert calls == []
    assert not any(r.name == "provider_ping_local" for r in results)


# --------------------------------------------------------------------- #
# Never-raises
# --------------------------------------------------------------------- #


def test_never_raises_resolve_seats_throws(monkeypatch):
    @contextlib.contextmanager
    def _noop_fanout(workflow, provider_hint=""):
        yield {}

    def _boom(two_pass):
        raise RuntimeError("resolve boom")

    monkeypatch.setattr(run_env, "applied_run_env", _noop_fanout)
    monkeypatch.setattr(run_env, "resolve_seats", _boom)
    monkeypatch.setattr(run_env, "load_provider_key_table", _key_table)

    results = prov.provider_checks(_run_ctx())
    assert isinstance(results, list)
    assert results  # an error result, not an exception
    assert any(r.severity is Severity.WARN for r in results)


def test_never_raises_applied_run_env_throws(monkeypatch):
    @contextlib.contextmanager
    def _boom_fanout(workflow, provider_hint=""):
        raise RuntimeError("fanout boom")
        yield {}  # pragma: no cover

    monkeypatch.setattr(run_env, "applied_run_env", _boom_fanout)
    results = prov.provider_checks(_run_ctx())
    assert isinstance(results, list)
    assert any(r.severity is Severity.WARN for r in results)


def test_never_raises_ping_client_throws(patch_engine, monkeypatch):
    patch_engine([_seat("content", "local", source="fanout")], present={})

    def _boom(**kw):
        raise RuntimeError("client construct boom")

    monkeypatch.setattr(prov, "_make_ping_client", _boom)
    results = prov.provider_checks(_run_ctx(ping=True))
    assert isinstance(results, list)
    ping = _by_name(results).get("provider_ping_local")
    assert ping is not None
    assert ping.severity is Severity.FAIL


def test_register_provider_checks_registers(monkeypatch):
    from lib.diagnostics import core

    monkeypatch.setattr(core, "_REGISTRY", [])
    prov.register_provider_checks()
    groups = [g for g, _ in core.registered_checks()]
    assert "provider" in groups


def test_run_config_none_is_bare(patch_engine):
    patch_engine([_seat("content", "local", source="fanout")], present={})
    results = prov.provider_checks(CheckContext(run_config=None))
    assert "provider_bare_caveat" in _by_name(results)
