"""Deterministic test-support utilities (stdlib-only, no extra deps).

Currently exposes :mod:`lib.testing.no_network`, the offline-mode socket
guard used by the retrieval offline-verification suite. Kept dependency-free
on purpose: importing this package must never pull a heavy or optional
third-party module so it stays usable in slim CPU-only dev installs.
"""

from lib.testing.no_network import NetworkBlockedError, no_network

__all__ = ["NetworkBlockedError", "no_network"]
