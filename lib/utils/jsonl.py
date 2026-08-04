"""JSONL I/O with canonical defaults.

Replaces 16+ inline JSONL writer / reader near-duplicates across
:mod:`Trainforge` and :mod:`Courseforge`. Defaults:

- ``ensure_ascii=False`` so non-ASCII content survives a tail-f read.
- ``sort_keys=True`` so re-runs produce byte-stable output.
- ``write_jsonl`` is atomic (tmp+rename) so a partial write never
  clobbers the canonical artifact.
- ``append_jsonl`` is per-record streaming for in-progress sidecars
  (resume checkpoints, observability tails) -- caller controls the
  open file handle for explicit lifetime management.

See plan ``plans/wave-D6-lib-utils-package-2026-05-07.md`` Section 3.2 for
the migration table + flag-default rationale (resolved to ensure_ascii=False
+ sort_keys=True across the board).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterable, List, Optional, Union

__all__ = ["read_jsonl", "write_jsonl", "append_jsonl"]


def read_jsonl(
    path: Union[str, Path],
    *,
    skip_blank: bool = True,
    skip_invalid: bool = False,
) -> List[Dict[str, Any]]:
    """Read a JSONL file and return its records as a list of dicts.

    Args:
        path: JSONL file. Returns ``[]`` when path doesn't exist
            (matches :func:`Trainforge.training.compute_backend._read_jsonl`).
        skip_blank: When True, skip empty lines. Default True.
        skip_invalid: When True, skip lines that fail :func:`json.loads`
            (matches ``pilot_report_helpers.count_property_coverage_from_jsonl``).
            Default False -- propagates :class:`json.JSONDecodeError` so a
            corrupt file fails loudly during ingest.

    Returns:
        List of decoded records (typed as ``List[Dict[str, Any]]`` to match
        the dominant call-site shape; callers reading non-dict rows can
        ``cast`` at the call site).
    """
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if skip_blank and not stripped:
                continue
            try:
                out.append(json.loads(stripped if skip_blank else line))
            except json.JSONDecodeError:
                if skip_invalid:
                    continue
                raise
    return out


def write_jsonl(
    path: Union[str, Path],
    records: Iterable[Dict[str, Any]],
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    default: Optional[Callable[[Any], Any]] = None,
    atomic: bool = True,
) -> int:
    """Atomically write an iterable of dict records to a JSONL file.

    Mirrors :func:`Trainforge.synthesize_training._write_jsonl` --
    writes to ``{path}.tmp`` then ``rename(path)`` so a partial write
    never clobbers the existing artifact.

    Args:
        path: Target JSONL path. Parent dir is created if missing.
        records: Iterable of records. Empty iterable produces an empty file.
        ensure_ascii: Default False (canonical project default).
        sort_keys: Default True (canonical project default).
        default: Optional ``json.dumps(default=...)`` fallback (set to ``str``
            when records carry datetimes; matches
            :mod:`Trainforge.training.runner` line 1023).
        atomic: When True (default), write via tmp+rename. Set False to
            stream directly (legacy callers, e.g. ``align_chunks.write_corpus``).

    Returns:
        Count of records written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    target = p.with_suffix(p.suffix + ".tmp") if atomic else p
    count = 0
    with target.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(
                json.dumps(
                    rec,
                    ensure_ascii=ensure_ascii,
                    sort_keys=sort_keys,
                    default=default,
                )
                + "\n"
            )
            count += 1
    if atomic:
        target.replace(p)
    return count


def append_jsonl(
    fh: IO[str],
    record: Dict[str, Any],
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    flush: bool = True,
) -> None:
    """Append one record to an open file handle (no path management).

    Designed for in-progress sidecars + resume checkpoints where the
    caller owns the file lifetime (open at start of run, append per
    record, close at end). Mirrors the inline pattern at
    :mod:`Trainforge.synthesize_training` line 1180, 1725, 1809, etc.

    Args:
        fh: File handle (text mode, encoding="utf-8" expected).
        record: Single dict record.
        ensure_ascii: Default False.
        sort_keys: Default True. Set False to preserve key order
            (matches inline ``json.dumps(record)`` behavior at
            :mod:`Trainforge.alignment.align_chunks` -- emit-order can be
            load-bearing for some checkpoint formats).
        flush: When True (default), flush after write so a tail-f reader
            sees the line immediately.
    """
    fh.write(
        json.dumps(record, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
        + "\n"
    )
    if flush:
        fh.flush()
