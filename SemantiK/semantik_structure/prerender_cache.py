"""Disk cache for HTML→PDF renders used by the data-gen pipeline.

Pairs from Wikipedia / OpenStax / Gutenberg / Federal Register don't
carry a local_pdf — their PDF is produced by Playwright rendering the
`output_html` field. That render is the slowest part of data generation
(2-10s per pair, multi-MB browser process overhead).

This module caches rendered PDFs so subsequent data-gen runs skip the
renderer entirely. Cache key: (sha256 of html, render_config_version).
Change the version constant to invalidate every cached entry.

Usage:
    from semantik_structure.prerender_cache import cache_path_for, get_or_render

    p = cache_path_for(html)
    if p.exists():
        shared = extract_shared_cached(p)
    else:
        get_or_render(html, validator, p)  # renders + stores
        shared = extract_shared_cached(p)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import paths as _semantik_paths


# Bump when the render config changes (page size, margins, print_background, ...)
# so the cache auto-invalidates for any entry produced by an older config.
RENDER_CONFIG_VERSION = "v1"
# What we actually render with. `HtmlValidator.render_pdf` hardcodes
# Letter + print_background=True; this constant documents that so future
# tweaks know to bump the version above.
RENDER_CONFIG_DESCRIPTION = "format=Letter;print_background=True"

# Anchored under the SemantiK cache root (CWD-independent) instead of the legacy
# CWD-relative ``data/prerender_cache`` literal. Byte-stable to the historic
# ``<root>/data/prerender_cache`` layout when no SEMANTIK_* env var is set.
DEFAULT_CACHE_DIR = _semantik_paths.resolve_cache("prerender_cache")


def _key(html: str, config_version: str = RENDER_CONFIG_VERSION) -> str:
    h = hashlib.sha256()
    h.update(config_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(html.encode("utf-8"))
    return h.hexdigest()


def cache_path_for(html: str,
                   *,
                   cache_dir: Path = DEFAULT_CACHE_DIR,
                   config_version: str = RENDER_CONFIG_VERSION) -> Path:
    """Return the deterministic cache path for `html` under `cache_dir`.

    The path is a function of (config_version, sha256(html)). Does NOT
    render or touch disk — caller checks `.exists()` before deciding.
    """
    return cache_dir / f"{_key(html, config_version)}.pdf"


def get_or_render(html: str, validator, out_path: Path | None = None,
                  *, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return a path to a rendered PDF for `html`, using the cache when hot.

    If `out_path` is omitted, writes to the deterministic cache path.
    `validator` must be an HtmlValidator instance (so callers inside a
    `with HtmlValidator() as v` block can reuse its Chromium context).
    """
    path = out_path if out_path is not None else cache_path_for(html, cache_dir=cache_dir)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Render to a temp sibling first, then atomically rename — avoids
    # a partial file being mistaken for a hit on concurrent renders.
    tmp = path.with_suffix(path.suffix + ".tmp")
    validator.render_pdf(html, tmp)
    tmp.replace(path)
    return path
