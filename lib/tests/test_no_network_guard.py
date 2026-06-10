"""Unit tests for the offline-mode socket guard (``lib.testing.no_network``).

These are the guard's own self-tests: they prove the guard actually blocks a
socket attempt (so the offline retrieval suite's green result is meaningful),
that the escape hatch behaves as documented, and that the patch restores
cleanly + nests re-entrantly. No network is ever touched: the guard fires
before any packet leaves the box, so every "connect attempt" here is a
no-op that raises immediately.
"""

from __future__ import annotations

import socket

import pytest

from lib.testing.no_network import NetworkBlockedError, no_network


def test_blocks_inet_connect():
    """A loopback INET connect under the default guard raises NetworkBlockedError.

    This is the load-bearing self-test: it proves the guard genuinely blocks
    a socket attempt. If this passed for the wrong reason (guard not installed)
    the offline suite would be vacuously green.
    """
    with no_network():
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("127.0.0.1", 1), timeout=0.01)


def test_blocks_instance_connect():
    """``socket.socket.connect`` (the instance method) is also intercepted."""
    with no_network():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkBlockedError):
                s.connect(("127.0.0.1", 1))
        finally:
            s.close()


def test_blocks_getaddrinfo():
    """DNS resolution raises — the guard fires before any packet, so the test
    itself needs no working resolver / network."""
    with no_network():
        with pytest.raises(NetworkBlockedError):
            socket.getaddrinfo("example.invalid", 80)


def test_allow_loopback_escape_hatch():
    """With allow_loopback=True a loopback connect to a closed port raises the
    ordinary OSError (NOT NetworkBlockedError) — the guard steps aside."""
    with no_network(allow_loopback=True):
        with pytest.raises(OSError) as exc_info:
            socket.create_connection(("127.0.0.1", 1), timeout=0.05)
        assert not isinstance(exc_info.value, NetworkBlockedError)


def test_unix_sockets_allowed_by_default():
    """AF_UNIX socket connects are left untouched (multiprocessing/asyncio).

    Connecting to a non-existent unix path raises a plain OSError from the real
    stack — NOT NetworkBlockedError — proving the guard didn't intercept it.
    """
    if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - non-posix
        pytest.skip("AF_UNIX not available on this platform")
    with no_network():
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError) as exc_info:
                s.connect("/tmp/this_socket_does_not_exist_ed4all_ws1.sock")
            assert not isinstance(exc_info.value, NetworkBlockedError)
        finally:
            s.close()


def test_restores_socket_on_exit():
    """After the context exits, the socket callables are the originals."""
    orig_connect = socket.socket.connect
    orig_getaddrinfo = socket.getaddrinfo
    orig_create_connection = socket.create_connection
    with no_network():
        assert socket.socket.connect is not orig_connect
    assert socket.socket.connect is orig_connect
    assert socket.getaddrinfo is orig_getaddrinfo
    assert socket.create_connection is orig_create_connection


def test_restores_socket_on_exception():
    """Originals are restored even when the body raises."""
    orig_connect = socket.socket.connect
    with pytest.raises(ValueError):
        with no_network():
            raise ValueError("boom")
    assert socket.socket.connect is orig_connect


def test_reentrant():
    """Nested guards don't double-patch and the outermost exit restores."""
    orig_connect = socket.socket.connect
    with no_network():
        patched_outer = socket.socket.connect
        with no_network():
            # Re-entry must not re-wrap: the inner patch object is identical.
            assert socket.socket.connect is patched_outer
            with pytest.raises(NetworkBlockedError):
                socket.create_connection(("127.0.0.1", 1), timeout=0.01)
        # After the inner block exits, still patched (outer is active).
        assert socket.socket.connect is patched_outer
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("127.0.0.1", 1), timeout=0.01)
    # Outermost exit restores the true original.
    assert socket.socket.connect is orig_connect
