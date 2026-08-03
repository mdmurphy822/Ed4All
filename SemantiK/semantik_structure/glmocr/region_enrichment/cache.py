"""Persistent cache for GLM-OCR region results.

Keyed by sha256(pdf_sha256 | page_num | bbox_rounded | prompt). Values
are small JSON blobs with the OCR text (or an error record). Greedy
decoding makes the result deterministic, so re-asking the same question
always returns the same answer — and therefore caching is safe.

Storage layout:
    data/glm_ocr_cache/<first2>/<hash>.json

Small files per region keep reads cheap and avoid a central DB that
would serialize concurrent workers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ... import paths as _semantik_paths

# Anchored under the SemantiK cache root (CWD-independent) instead of the legacy
# CWD-relative ``data/glm_ocr_cache`` literal. Byte-stable to the historic
# ``<root>/data/glm_ocr_cache`` layout when no SEMANTIK_* env var is set.
DEFAULT_CACHE_DIR = _semantik_paths.resolve_cache("glm_ocr_cache")


@dataclass(frozen=True)
class CacheKey:
    pdf_sha: str
    page_num: int
    bbox: tuple[float, float, float, float]
    prompt: str

    def _as_string(self) -> str:
        bbox_r = tuple(round(float(v), 2) for v in self.bbox)
        return f"{self.pdf_sha}|{self.page_num}|{bbox_r}|{self.prompt}"

    def digest(self) -> str:
        return hashlib.sha256(self._as_string().encode("utf-8")).hexdigest()


def _path(key: CacheKey, cache_dir: Path) -> Path:
    h = key.digest()
    return cache_dir / h[:2] / f"{h}.json"


def get(key: CacheKey, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict | None:
    p = _path(key, cache_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def put(key: CacheKey, payload: dict,
        cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    p = _path(key, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(p)


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stable sha256 of a PDF so the cache key survives file moves."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
