"""Process-local fail-closed guard for synthesis provider dispatch."""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock
from typing import Mapping, Optional


class SynthesisContractDrift(RuntimeError):
    """A sealed synthesis implementation changed before provider dispatch."""


_LOCK = RLock()
_EXPECTED: Optional[dict[str, str]] = None


def register_contract_files(files: Mapping[str, str]) -> None:
    global _EXPECTED
    with _LOCK:
        _EXPECTED = dict(files)


def clear_contract_files() -> None:
    global _EXPECTED
    with _LOCK:
        _EXPECTED = None


def assert_contract_files_unchanged() -> None:
    with _LOCK:
        expected = dict(_EXPECTED or {})
    changed: list[str] = []
    for raw_path, expected_sha in expected.items():
        path = Path(raw_path)
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            actual = "missing"
        if actual != expected_sha:
            changed.append(raw_path)
    if changed:
        raise SynthesisContractDrift(
            "synthesis contract drift blocked provider dispatch; changed: "
            + ", ".join(sorted(changed))
        )


__all__ = [
    "SynthesisContractDrift",
    "assert_contract_files_unchanged",
    "clear_contract_files",
    "register_contract_files",
]
