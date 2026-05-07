"""SHA-256 helpers for file + string content.

Replaces 7 inline ``_sha256_file`` / ``_compute_sha256`` / ``_hash_file`` /
``_compute_file_checksum`` near-duplicates across :mod:`lib.validators`,
:mod:`lib.aggregators`, :mod:`lib.replay_engine`, :mod:`lib.run_finalizer`,
and :mod:`Trainforge.training.runner`. See plan
``plans/wave-D6-lib-utils-package-2026-05-07.md`` Section 3.1 for the
migration table + chunk-size rationale (resolved to 65536 across the
board).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

__all__ = ["sha256_file", "sha256_text"]


def sha256_file(path: Union[str, Path], *, chunk: int = 65536) -> str:
    """Stream a file through SHA-256 and return the hex digest.

    Canonical signature for content-addressing manifest gates, promotion
    chain reports, and training-run provenance hashes. Pure stdlib;
    propagates :class:`OSError` on missing / unreadable file. Callers
    that want best-effort semantics (e.g. promotion_chain_report's
    chain-stage walk) wrap this in their own ``try/except``.

    Args:
        path: File to hash. Accepts ``str`` or :class:`pathlib.Path`.
        chunk: Read-buffer size in bytes (keyword-only). Default 64 KiB.

    Returns:
        Lower-case hex digest (64 chars).

    Raises:
        OSError: When the file cannot be opened.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 encoded string.

    Pulled out of :mod:`lib.aggregators.promotion_chain_report` so it can
    be reused by any caller that hashes a canonicalised JSON payload
    (e.g. ``contentHash`` in ``Block.to_jsonld_entry()``). Pure stdlib.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
