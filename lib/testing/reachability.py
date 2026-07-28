"""Live-backend reachability probes for tests — stdlib-only, no extra deps.

A small, shared set of helpers that let a test gate itself on whether a live
local model server (Ollama / vLLM / llama.cpp / LM Studio) is actually
listening. The contract is deliberately one-directional:

* When the backend IS reachable (a developer's box with Ollama up, or a CI
  lane that provisions one), the probe returns ``True`` and the test runs
  exactly as it would have before — same provider, same network call, same
  assertions. The probe NEVER stubs, mocks, or alters behavior.
* When the backend is genuinely absent (a clean CI runner with no model
  server on ``localhost:11434``), the probe returns ``False`` and the caller
  ``pytest.skip(...)`` with a precise reason. This is the difference between a
  red suite shaped by the environment and a green suite that still exercises
  the real path wherever the backend exists.

This exists because several Trainforge synthesis tests pass ``provider="mock"``
but the effective provider is overridden to ``local`` by a
``TRAINFORGE_SYNTHESIS_PROVIDER=local`` environment variable present on some
runners (the marketable-v1 license-clean default). When that override is in
force and no local server is listening, the run fails with
``SynthesisProviderError: ... [Errno 111] Connection refused`` rather than
skipping. These helpers detect exactly that condition.

Design constraints (mirrors ``lib/testing/no_network.py``):

* **stdlib-only.** ``socket`` connect with a short timeout; no ``requests`` /
  ``httpx`` import, so this stays importable in the slimmest CPU-only install.
* **Resolution matches production.** ``resolve_local_synthesis_base_url`` reads
  the same ``LOCAL_SYNTHESIS_BASE_URL`` env var + ``http://localhost:11434/v1``
  default that ``Trainforge/generators/_local_provider.py`` uses, so the probe
  targets the host:port the provider would actually dial.
* **Cheap + side-effect-free.** A single TCP connect, default 0.5s timeout,
  closed immediately. No DNS-less assumptions: hostnames resolve normally.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlsplit

__all__ = [
    "DEFAULT_LOCAL_SYNTHESIS_BASE_URL",
    "ENV_LOCAL_SYNTHESIS_BASE_URL",
    "ENV_TRAINFORGE_SYNTHESIS_PROVIDER",
    "endpoint_reachable",
    "resolve_local_synthesis_base_url",
    "effective_synthesis_provider",
    "local_synthesis_reachable",
    "skip_if_local_synthesis_unreachable",
    "is_local_connection_refused",
    "make_local_synthesis_skip_hook",
]

# Kept in sync with Trainforge/generators/_local_provider.py (DEFAULT_BASE_URL,
# ENV_BASE_URL) and Trainforge/synthesize_training.py (the
# TRAINFORGE_SYNTHESIS_PROVIDER override). Duplicated here rather than imported
# so the helper stays importable without pulling the generator stack.
DEFAULT_LOCAL_SYNTHESIS_BASE_URL = "http://localhost:8000/v1"
ENV_LOCAL_SYNTHESIS_BASE_URL = "LOCAL_SYNTHESIS_BASE_URL"
ENV_TRAINFORGE_SYNTHESIS_PROVIDER = "TRAINFORGE_SYNTHESIS_PROVIDER"

# Providers that dial a live OpenAI-compatible local server. ``mock`` is
# in-process/deterministic; ``together`` / ``anthropic`` hit cloud endpoints
# gated by their own API keys (a missing key fails loud before any socket).
_LIVE_LOCAL_PROVIDERS = frozenset({"local"})

_DEFAULT_PROBE_TIMEOUT = 0.5


def endpoint_reachable(base_url: str, timeout: float = _DEFAULT_PROBE_TIMEOUT) -> bool:
    """Return True iff a TCP connect to the host:port of ``base_url`` succeeds.

    Parses ``base_url`` for its host + port (defaulting the port from the URL
    scheme — 80 for http, 443 for https), then attempts a single short-timeout
    ``socket.create_connection``. Any failure (refused, timeout, DNS error,
    malformed URL) returns ``False`` — the caller treats that as "backend
    absent" and skips. A successful connect is closed immediately.
    """
    try:
        parts = urlsplit(base_url)
    except ValueError:
        return False
    host = parts.hostname
    if not host:
        return False
    port = parts.port
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_local_synthesis_base_url() -> str:
    """The base URL the ``local`` synthesis provider would dial.

    Mirrors ``_local_provider.py``: ``LOCAL_SYNTHESIS_BASE_URL`` env var if set
    (and non-empty after strip), else the canonical local
    OpenAI-compatible endpoint ``http://localhost:8000/v1``.
    """
    env = os.environ.get(ENV_LOCAL_SYNTHESIS_BASE_URL, "").strip()
    return env or DEFAULT_LOCAL_SYNTHESIS_BASE_URL


def effective_synthesis_provider(provider: str) -> str:
    """Resolve the provider a synthesis call will actually use.

    Mirrors ``Trainforge/synthesize_training.py::run_synthesis``: a non-empty
    ``TRAINFORGE_SYNTHESIS_PROVIDER`` env var overrides the caller-supplied
    ``provider`` kwarg. Used so a test that passes ``provider="mock"`` can ask
    whether an env override has silently re-routed it to ``local``.
    """
    env_provider = os.environ.get(ENV_TRAINFORGE_SYNTHESIS_PROVIDER, "").strip()
    return env_provider or provider


def local_synthesis_reachable(timeout: float = _DEFAULT_PROBE_TIMEOUT) -> bool:
    """True iff the resolved local-synthesis endpoint accepts a TCP connect."""
    return endpoint_reachable(resolve_local_synthesis_base_url(), timeout=timeout)


def skip_if_local_synthesis_unreachable(
    provider: str = "mock", timeout: float = _DEFAULT_PROBE_TIMEOUT
) -> None:
    """``pytest.skip`` iff the effective synthesis provider needs a live local
    server that isn't listening.

    Pass the ``provider`` the test hands to ``run_synthesis``. When an
    environment override (``TRAINFORGE_SYNTHESIS_PROVIDER=local``) re-routes a
    nominally-``mock`` test to the live local backend AND that backend is not
    reachable, the test is skipped with a precise reason. When the backend IS
    up — or no override is in force and the test stays ``mock`` — this is a
    no-op and the test runs unchanged.

    Imported lazily so this module has no hard pytest dependency at import.
    """
    effective = effective_synthesis_provider(provider)
    if effective not in _LIVE_LOCAL_PROVIDERS:
        return
    base_url = resolve_local_synthesis_base_url()
    if endpoint_reachable(base_url, timeout=timeout):
        return
    import pytest  # noqa: PLC0415 — lazy so non-test importers don't need pytest

    pytest.skip(
        f"local synthesis backend unreachable at {base_url} "
        f"(TRAINFORGE_SYNTHESIS_PROVIDER={effective!r} overrides "
        f"provider={provider!r}); start a local OpenAI-compatible server "
        f"(e.g. Ollama on :11434) or unset the override to run this test"
    )


def is_local_connection_refused(exc: BaseException) -> bool:
    """True iff ``exc`` (or any chained cause/context) is the local-synthesis
    endpoint refusing a connection.

    Matches a direct ``ConnectionRefusedError`` / ``OSError(ECONNREFUSED)`` in
    the chain, plus the local provider's wrapped error text
    (``SynthesisProviderError: local request ... transport attempts ...
    Connection refused``). Deliberately narrow so a genuine assertion failure,
    schema error, or any other exception is never misread as "backend absent".
    """
    import errno  # noqa: PLC0415

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ConnectionRefusedError):
            return True
        if isinstance(cur, OSError) and cur.errno == errno.ECONNREFUSED:
            return True
        text = str(cur)
        if "Connection refused" in text and (
            "local request" in text or "transport attempts" in text
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def make_local_synthesis_skip_hook():
    """Build a ``pytest_runtest_call`` hookwrapper that turns an "absent local
    synthesis backend" failure into a skip.

    A conftest that hosts synthesis tests aliases the returned callable as its
    own ``pytest_runtest_call`` so pytest discovers it::

        from lib.testing.reachability import make_local_synthesis_skip_hook
        pytest_runtest_call = make_local_synthesis_skip_hook()

    Precision contract (rules of engagement #3): the hook is a strict no-op
    unless the resolved local-synthesis endpoint (``LOCAL_SYNTHESIS_BASE_URL``,
    default Ollama ``:11434``) refuses a TCP connect AND the test failed
    *specifically* because a synthesis provider's call to that endpoint was
    refused. The connection-refused signature is narrow (a chained
    ``ConnectionRefusedError`` / ``OSError(ECONNREFUSED)``, or the provider's
    wrapped ``... local request ... transport attempts ... Connection refused``
    text, including when surfaced inside a result-dict ``error`` field that the
    test then asserts on) — so a genuine assertion / schema / logic failure is
    never misread as a skip.

    This covers EVERY surface that an env override can route to the local
    server (``TRAINFORGE_SYNTHESIS_PROVIDER`` for training-pair synthesis,
    ``TEXTBOOK_SYNTHESIS_PROVIDER`` for the three-stage concept pass, etc.)
    without enumerating each var: the gate is "endpoint down + the failure was
    a refused connect to it". When a local server IS up, no connection is
    refused, nothing is skipped, and every test runs against the live backend
    exactly as before.

    A hookwrapper (not a yield-fixture) is required: modern pytest does NOT
    throw test-body exceptions into yield-fixtures, so a fixture cannot see the
    raised error to translate it.
    """
    import pytest  # noqa: PLC0415

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(item):  # noqa: ANN001
        base_url = resolve_local_synthesis_base_url()
        # Probe ONCE up front: if the endpoint is reachable we never skip, so
        # there's no point inspecting the outcome.
        endpoint_down = not endpoint_reachable(base_url)
        outcome = yield
        if not endpoint_down:
            return
        excinfo = outcome.excinfo
        if excinfo is None:
            return
        if is_local_connection_refused(excinfo[1]):
            outcome.force_exception(
                pytest.skip.Exception(
                    f"local synthesis backend unreachable at {base_url}: a "
                    f"synthesis provider routed to it (likely a "
                    f"*_SYNTHESIS_PROVIDER=local override) and the connection "
                    f"was refused. Start a local OpenAI-compatible server "
                    f"(e.g. Ollama on :11434) or unset the override to run "
                    f"this test."
                )
            )

    return pytest_runtest_call
