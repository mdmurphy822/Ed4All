"""Offline-mode verification guard — stdlib socket-blocking, no extra deps.

A process-global, restore-on-exit context manager that patches the stdlib
``socket`` entry points so any attempt to open an ``AF_INET`` / ``AF_INET6``
network connection (or resolve a hostname) raises :class:`NetworkBlockedError`
instead of touching the network. It exists to *prove* a code path performs
zero network I/O — the lexical retrieval pipeline (BM25 over archived
``chunks.jsonl`` + citation anchoring) is pure file I/O and must need nothing
off-box.

Design constraints (deliberate, documented):

* **No ``pytest-socket`` dependency.** The default install stays slim; stdlib
  monkeypatching is sufficient and deterministic. There is no socket-patching
  precedent anywhere in the repo, so this is the canonical implementation.
* **Unix sockets stay open by default** (``allow_unix_sockets=True``).
  ``multiprocessing`` / ``asyncio`` internals and some test servers rely on
  ``AF_UNIX`` pairs; blocking them would break unrelated test infrastructure.
* **Loopback is blocked by default** (``allow_loopback=False``). The WS1
  retrieval path needs no localhost server. ``allow_loopback=True`` is the
  pre-declared escape hatch for a future query path that talks to a localhost
  model server — under it, a loopback connect to a closed port raises the
  normal ``ConnectionRefusedError`` / ``OSError`` rather than
  ``NetworkBlockedError`` (the guard simply doesn't intercept it).
* **Process-global patch, test-scope use only.** The patch is *not*
  thread-local. Use it within a context-manager scope (or the
  ``offline_guard`` fixture) so the originals are restored deterministically on
  exit. Re-entrant: nested ``no_network()`` blocks don't double-patch and the
  outermost exit restores the true originals.

The guard intercepts at four chokepoints so a network attempt fails *before*
any packet leaves the box:

* ``socket.socket.connect`` / ``connect_ex`` — instance-method connects.
* ``socket.create_connection`` — the high-level TCP helper.
* ``socket.getaddrinfo`` — DNS resolution (so even a name-lookup attempt
  raises, and the test itself never needs a working resolver).

No network, no LLM, no decision capture — this is pure test scaffolding.
"""

from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from typing import Any, Iterator

__all__ = ["NetworkBlockedError", "no_network"]


class NetworkBlockedError(RuntimeError):
    """Raised when code under a :func:`no_network` guard attempts a network call."""


# Re-entrancy bookkeeping. The patch is process-global; we track nesting depth
# and the captured originals so only the outermost ``no_network()`` installs /
# restores the patches. A lock guards the (rare, test-scope) case of two
# threads entering near-simultaneously — the depth/originals state must stay
# consistent even though the patch itself is global.
_state_lock = threading.Lock()
_depth = 0
_originals: dict[str, Any] = {}


def _is_inet_family(family: int) -> bool:
    """True for AF_INET / AF_INET6 (the families we block)."""
    return family in (socket.AF_INET, getattr(socket, "AF_INET6", socket.AF_INET))


def _address_is_inet(address: Any) -> bool:
    """Heuristic: a ``(host, port)`` tuple is an INET address; a str/bytes path
    is an AF_UNIX address. Used by the create_connection patch (which always
    takes a ``(host, port)`` tuple, so this is effectively always True there)
    and defensively elsewhere.
    """
    return isinstance(address, tuple) and len(address) >= 2


@contextmanager
def no_network(
    *, allow_unix_sockets: bool = True, allow_loopback: bool = False
) -> Iterator[None]:
    """Block INET network I/O for the duration of the ``with`` block.

    Args:
        allow_unix_sockets: When True (default), ``AF_UNIX`` socket connects
            are left untouched so multiprocessing / asyncio internals keep
            working. When False, every ``socket.connect`` is intercepted
            regardless of family.
        allow_loopback: When True, loopback (``127.0.0.1`` / ``::1`` /
            ``localhost``) connects are NOT intercepted — they proceed to the
            real stack (and raise the ordinary ``ConnectionRefusedError`` /
            ``OSError`` against a closed port). The escape hatch for a future
            localhost-model-server query path. Default False: loopback is
            blocked too, because the WS1 retrieval path needs nothing on-box.

    The patch restores the original ``socket`` callables on exit, even on
    exception, and is re-entrant.
    """
    global _depth

    loopback_hosts = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}

    def _is_loopback_address(address: Any) -> bool:
        if not (isinstance(address, tuple) and address):
            return False
        host = address[0]
        if not isinstance(host, str):
            return False
        return host in loopback_hosts or host.startswith("127.")

    with _state_lock:
        _depth += 1
        first = _depth == 1
        if first:
            _originals["connect"] = socket.socket.connect
            _originals["connect_ex"] = socket.socket.connect_ex
            _originals["create_connection"] = socket.create_connection
            _originals["getaddrinfo"] = socket.getaddrinfo

    if first:
        orig_connect = _originals["connect"]
        orig_connect_ex = _originals["connect_ex"]
        orig_create_connection = _originals["create_connection"]
        orig_getaddrinfo = _originals["getaddrinfo"]

        def _guard_unix_ok(self: socket.socket, address: Any) -> bool:
            """True iff this connect should be allowed through untouched."""
            if allow_unix_sockets and not _is_inet_family(self.family):
                return True
            if allow_loopback and _is_loopback_address(address):
                return True
            return False

        def patched_connect(self: socket.socket, address: Any) -> Any:
            if _guard_unix_ok(self, address):
                return orig_connect(self, address)
            raise NetworkBlockedError(
                f"network blocked under no_network(): connect to {address!r}"
            )

        def patched_connect_ex(self: socket.socket, address: Any) -> Any:
            if _guard_unix_ok(self, address):
                return orig_connect_ex(self, address)
            raise NetworkBlockedError(
                f"network blocked under no_network(): connect_ex to {address!r}"
            )

        def patched_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
            if allow_loopback and _is_loopback_address(address):
                return orig_create_connection(address, *args, **kwargs)
            raise NetworkBlockedError(
                f"network blocked under no_network(): create_connection to {address!r}"
            )

        def patched_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if allow_loopback and isinstance(host, (str, bytes)):
                h = host.decode() if isinstance(host, bytes) else host
                if h in loopback_hosts or h.startswith("127."):
                    return orig_getaddrinfo(host, *args, **kwargs)
            raise NetworkBlockedError(
                f"network blocked under no_network(): getaddrinfo for {host!r}"
            )

        socket.socket.connect = patched_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = patched_connect_ex  # type: ignore[method-assign]
        socket.create_connection = patched_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = patched_getaddrinfo  # type: ignore[assignment]

    try:
        yield
    finally:
        with _state_lock:
            _depth -= 1
            if _depth == 0:
                socket.socket.connect = _originals["connect"]  # type: ignore[method-assign]
                socket.socket.connect_ex = _originals["connect_ex"]  # type: ignore[method-assign]
                socket.create_connection = _originals["create_connection"]  # type: ignore[assignment]
                socket.getaddrinfo = _originals["getaddrinfo"]  # type: ignore[assignment]
                _originals.clear()
