"""Map ar5iv catalog references to their local dataset asset paths."""

from __future__ import annotations

import re
from pathlib import Path


def resolve_ar5iv_asset_path(
    output_root: Path,
    source: str,
    doc_id: str,
    ref: str,
) -> Path:
    """Return the fetched-image path for a valid ar5iv asset reference."""
    tail = (
        ref.split("/assets/", 1)[-1]
        if "/assets/" in ref
        else ref.rsplit("/", 1)[-1]
    )
    tail = re.sub(r"[^A-Za-z0-9._/-]", "_", tail).lstrip("/")
    safe_doc_id = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)
    return output_root / source / safe_doc_id / tail
