"""GPU + filesystem lock for Qwen LoRA training.

Two interlocked guards, both hard:

    1. ``nvidia-smi`` poll — refuse to start if any python process
       (other than ours) is currently using GPU memory. The 8 GB 3060
       is too small to co-tenant; a second trainer poisons the CUDA
       allocator and both jobs OOM-loop without surfacing a clean
       error. The poll is two-stage:
         a. List PIDs touching the GPU.
         b. Filter out our own PID (so the test harness can call this
            from inside a process that itself owns GPU memory — useful
            for resume-from-checkpoint flows).

    2. ``fcntl.flock`` on ``/tmp/semantik_qwen_train.lock`` — a process-level
       mutex. The flock is non-blocking; conflict raises immediately
       instead of stalling. Released on context exit (including SIGINT,
       because Python tears down the file descriptor in the signal
       handler path).

Why both: the GPU poll catches "another job started before we did";
the flock catches "another job started in the same second as we did,
and nvidia-smi hasn't seen its allocation yet." Together they are
sufficient for the local 3060 setup. On a multi-GPU box this would be
replaced with a real scheduler.
"""
from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


_LOCK_PATH = Path("/tmp/semantik_qwen_train.lock")


class TrainingSlotBusy(RuntimeError):
    """Raised when the GPU or lockfile is already in use."""


def _resolve_smi() -> str | None:
    """Find ``nvidia-smi``. WSL2 ships it under /usr/lib/wsl/lib but
    doesn't put it on PATH by default."""
    p = shutil.which("nvidia-smi")
    if p:
        return p
    wsl = "/usr/lib/wsl/lib/nvidia-smi"
    return wsl if Path(wsl).exists() else None


def _gpu_pids_in_use() -> list[tuple[int, str]]:
    """Return ``[(pid, process_name), ...]`` for processes touching GPU 0.

    Returns an empty list if ``nvidia-smi`` is missing (CPU-only host)
    or if the query succeeds but reports no processes.

    NOTE: WSL2 nvidia-smi does NOT expose compute-app PIDs through
    ``--query-compute-apps`` (only the Xwayland graphics process). So
    this returns an empty list under WSL2 even when the GPU is fully
    saturated. The memory-utilization fallback in ``_gpu_memory_used_mb``
    catches the WSL2 case.
    """
    smi = _resolve_smi()
    if smi is None:
        logger.warning(
            "nvidia-smi not found; skipping GPU-busy check (CPU-only host?)"
        )
        return []
    try:
        out = subprocess.check_output(
            [
                smi,
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("nvidia-smi probe failed (%s); proceeding without GPU check", exc)
        return []
    pids: list[tuple[int, str]] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # CSV: "12345, python"
        parts = [p.strip() for p in line.split(",", 1)]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        name = parts[1] if len(parts) > 1 else ""
        pids.append((pid, name))
    return pids


def _gpu_memory_used_mb() -> int | None:
    """Return GPU 0 used memory in MiB, or None if probe fails.

    This is the WSL2 fallback: even though we can't see PIDs, we CAN
    see memory utilisation. Memory > 1 GiB without our process
    accounting for it means someone else is on the GPU.
    """
    smi = _resolve_smi()
    if smi is None:
        return None
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    try:
        return int(out.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


# Threshold for the WSL2 memory-only fallback. Below this, we assume
# nothing material is on the GPU (Xwayland + driver overhead is ~200
# MiB; conservative headroom 1 GiB).
_WSL_FALLBACK_BUSY_MB = 1024


def _check_gpu_free(allow_self: bool = True) -> None:
    """Raise :class:`TrainingSlotBusy` if any non-self process owns GPU memory."""
    self_pid = os.getpid()
    pids = _gpu_pids_in_use()
    busy = [
        (pid, name) for pid, name in pids
        if not (allow_self and pid == self_pid)
    ]
    if busy:
        msg = "GPU is in use by: " + ", ".join(
            f"pid={pid} ({name})" for pid, name in busy
        )
        raise TrainingSlotBusy(msg)
    # WSL2 fallback: PID list can be empty even when memory is taken.
    if not pids:
        used = _gpu_memory_used_mb()
        if used is not None and used > _WSL_FALLBACK_BUSY_MB:
            raise TrainingSlotBusy(
                f"GPU memory utilization {used} MiB exceeds {_WSL_FALLBACK_BUSY_MB} MiB "
                f"(WSL2 hides PIDs; can't name the holder). Check `nvidia-smi` "
                f"manually before retrying."
            )


@contextlib.contextmanager
def acquire_training_slot(
    *,
    skip_gpu_check: bool = False,
    lock_path: Path | None = None,
) -> Iterator[None]:
    """Acquire the cross-process training slot.

    Blocking only on the lock release path: the GPU poll and the
    ``flock`` are both non-blocking. A failure on either raises
    :class:`TrainingSlotBusy` with a diagnostic message.

    The context releases the lock on exit (normal or exceptional).
    SIGINT also unwinds the context manager because Python's signal
    handler runs ``KeyboardInterrupt`` through normal exception
    machinery — the ``finally`` clause closes the fd and ``flock``
    releases automatically when the fd closes.

    Parameters
    ----------
    skip_gpu_check
        If True, skip the ``nvidia-smi`` poll. Used by the smoke-test
        harness to verify the lock-only path on a CPU box. Trainer
        proper always runs with ``skip_gpu_check=False``.
    lock_path
        Override the lock file location (tests).
    """
    path = Path(lock_path) if lock_path is not None else _LOCK_PATH

    if not skip_gpu_check:
        _check_gpu_free(allow_self=True)

    # Open in append mode so we don't truncate a stale lock file.
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TrainingSlotBusy(
                f"another trainer holds {path}; refusing to start"
            ) from exc
        # Stamp our pid so anyone running `cat <lockpath>` can see who's
        # holding it. Best-effort.
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.fsync(fd)
        except OSError:
            pass
        logger.info("acquired training slot (lock=%s, pid=%d)", path, os.getpid())
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass


__all__ = ["acquire_training_slot", "TrainingSlotBusy"]
