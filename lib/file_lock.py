"""Cross-process advisory file locking helper.

A single, minimal ``flock``-based context manager reused by the LibV2 catalog
read-modify-write path (``LibV2/tools/libv2/catalog.py``) and any other
multi-process write critical section that needs lost-update protection.

POSIX-only: ``fcntl`` is unavailable on Windows and unreliable on WSL2 DrvFS
(``/mnt/c/``) and some NFS mounts. The helper degrades GRACEFULLY rather than
crashing — when ``fcntl`` is missing or the ``flock`` syscall raises ``OSError``
the body still runs (unlocked), matching the documented fallback posture of the
decision-capture write path. The lock is therefore best-effort: it serializes
concurrent OS processes on filesystems that honor ``flock`` (the common case),
and no-ops elsewhere instead of blocking progress.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

try:  # pragma: no cover - exercised on POSIX
    import fcntl as _fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - non-POSIX (Windows) fallback
    _fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = ["file_lock"]


@contextmanager
def file_lock(lock_path: Union[str, Path]) -> Iterator[None]:
    """Acquire an exclusive advisory lock on ``lock_path`` for the block.

    Creates ``lock_path`` (and parents) if absent, opens it, and holds an
    exclusive ``LOCK_EX`` for the duration of the ``with`` block. The lock
    file itself is a sentinel — it is never written to — so it can be safely
    reused across calls.

    Graceful degradation:

    * ``fcntl`` unavailable (Windows) → the body runs unlocked.
    * ``flock`` raises ``OSError`` (WSL2 DrvFS / NFS / unsupported FS) → the
      body runs unlocked after a one-line warning.
    * The sentinel file cannot be created/opened → the body runs unlocked
      after a warning (never blocks the caller's real work).
    """
    lock_path = Path(lock_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+")
    except OSError as exc:
        logger.warning(
            "file_lock: cannot open lock sentinel %s (%s); proceeding unlocked.",
            lock_path,
            exc,
        )
        yield
        return

    locked = False
    try:
        if _fcntl is not None:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
                locked = True
            except OSError as exc:
                logger.warning(
                    "file_lock: flock unavailable on %s (%s); proceeding "
                    "unlocked — concurrent writers may race.",
                    lock_path,
                    exc,
                )
        yield
    finally:
        if locked:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)  # type: ignore[union-attr]
            except OSError:
                pass
        try:
            fh.close()
        except OSError:
            pass
