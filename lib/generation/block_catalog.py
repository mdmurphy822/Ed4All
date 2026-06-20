"""Block catalog loader (Wave-2 block-variety redesign, Part 1).

Loads ``Courseforge/config/block_catalog.yaml`` — the machine-readable
catalog with one discriminating entry per canonical block type in
``Courseforge/scripts/blocks.py::BLOCK_TYPES`` — and exposes it as a list
of plain dicts for a content planner to consult when choosing the RIGHT
block type for a given content / objective shape.

Surface:

    load_block_catalog() -> List[Dict]   # the catalog's ``blocks`` list

Fail-loud contract: the catalog ships in-tree, so a missing or malformed
file is a packaging bug, not a recoverable runtime state — the loader
raises rather than returning an empty list (mirroring the fail-closed
posture of ``lib/ontology/taxonomy.py``). Import-light: only ``yaml`` +
``pathlib`` + ``lib.paths`` (no Courseforge renderer dependency tree).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml

from lib.paths import get_project_root

__all__ = ["load_block_catalog", "block_catalog_path"]


def block_catalog_path() -> Path:
    """Resolve the in-tree catalog YAML path relative to the repo root."""
    return get_project_root() / "Courseforge" / "config" / "block_catalog.yaml"


@lru_cache(maxsize=1)
def _load_raw() -> Dict:
    """Read + parse the catalog YAML once (cached). Fail-loud."""
    path = block_catalog_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"block_catalog.yaml not found at {path} — the catalog ships "
            "in-tree; a missing file is a packaging bug."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "blocks" not in data:
        raise ValueError(
            f"block_catalog.yaml at {path} is malformed — expected a "
            "top-level mapping with a 'blocks' key."
        )
    blocks = data["blocks"]
    if not isinstance(blocks, list):
        raise ValueError(
            f"block_catalog.yaml at {path} 'blocks' must be a list, got "
            f"{type(blocks).__name__}."
        )
    return data


def load_block_catalog() -> List[Dict]:
    """Return the catalog ``blocks`` list (defensive copy per entry).

    Each list item is a dict carrying ``block_type`` / ``label`` /
    ``use_when`` / ``conveys`` / ``bloom_fit`` / ``format_summary``.
    """
    blocks = _load_raw()["blocks"]
    # Defensive shallow copy so a caller mutating an entry can't poison
    # the lru_cache singleton.
    return [dict(entry) for entry in blocks]
