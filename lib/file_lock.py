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
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Union

try:  # pragma: no cover - exercised on POSIX
    import fcntl as _fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - non-POSIX (Windows) fallback
    _fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = ["file_lock"]


def _read_holder_stamp(lock_path: Path) -> str:
    """Best-effort read of the holder stamp a timeout-mode acquirer wrote.

    Returns a short human-readable description of the current/most-recent
    holder, or ``"unknown holder"`` when the sentinel carries no stamp.
    """
    try:
        stamp = lock_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "unknown holder"
    return stamp or "unknown holder"


def _write_holder_stamp(fh) -> None:
    """Best-effort stamp of this process's identity into the held sentinel.

    Only called while the exclusive lock is held, and only on the
    timeout-mode path — blocking-mode callers (e.g. the LibV2 catalog) keep
    the historical never-written sentinel. The stamp lets a later contender
    that times out name the other holder in its warning.
    """
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(
            f"pid={os.getpid()} cmd={os.path.basename(sys.argv[0]) if sys.argv else '?'} "
            f"acquired={datetime.now().isoformat()}\n"
        )
        fh.flush()
    except (OSError, ValueError):
        pass


@contextmanager
def file_lock(
    lock_path: Union[str, Path],
    timeout: Optional[float] = None,
) -> Iterator[None]:
    """Acquire an exclusive advisory lock on ``lock_path`` for the block.

    Creates ``lock_path`` (and parents) if absent, opens it, and holds an
    exclusive ``LOCK_EX`` for the duration of the ``with`` block. In the
    default (blocking) mode the lock file itself is a sentinel — it is never
    written to — so it can be safely reused across calls.

    ``timeout``: when ``None`` (default) the acquire blocks indefinitely —
    the historical behavior. When a positive number of seconds is given the
    acquire is a non-blocking (``LOCK_NB``) retry loop bounded by that many
    seconds; on expiry a LOUD warning naming the other holder (when knowable
    from the sentinel's holder stamp) is logged and the body runs UNLOCKED.
    For a caller that pairs the lock with an atomic temp+replace write this
    is strictly safe: a lost lock can at worst lose one update, never
    corrupt the file. Timeout-mode acquirers stamp their pid/cmd into the
    sentinel so contenders can name them.

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
            if timeout is None:
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
            else:
                deadline = time.monotonic() + max(timeout, 0.0)
                flock_broken = False
                while True:
                    try:
                        _fcntl.flock(
                            fh.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB
                        )
                        locked = True
                        break
                    except (BlockingIOError, PermissionError):
                        # Held by another process — retry until the deadline.
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(0.1)
                    except OSError as exc:
                        # flock itself unsupported on this FS.
                        logger.warning(
                            "file_lock: flock unavailable on %s (%s); "
                            "proceeding unlocked — concurrent writers may "
                            "race.",
                            lock_path,
                            exc,
                        )
                        flock_broken = True
                        break
                if locked:
                    _write_holder_stamp(fh)
                elif not flock_broken:
                    logger.warning(
                        "file_lock: TIMED OUT after %.1fs waiting for %s "
                        "(held by %s); proceeding WITHOUT the lock — the "
                        "paired atomic replace cannot corrupt the file, but "
                        "one concurrent update may be lost.",
                        timeout,
                        lock_path,
                        _read_holder_stamp(lock_path),
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
