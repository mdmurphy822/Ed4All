"""pyld document loader for the Courseforge JSON-LD @context (Wave 64).

The Courseforge emit carries ``@context: https://ed4all.dev/ns/courseforge/v1``.
That URL is the canonical logical identifier for the vocabulary — persistent
across any future hosting changes. Until ed4all.dev actually serves the
context document, a JSON-LD processor pointed at that URL would fail to
dereference it.

This module ships a drop-in :func:`pyld.jsonld.set_document_loader` hook
that serves the bundled repo copy at ``schemas/context/courseforge_v1.jsonld``
when a consumer asks for the canonical URL, and delegates to pyld's default
loader for anything else (so external vocabularies like Schema.org still
resolve over HTTP as normal).

Usage in consumer code::

    from pyld import jsonld
    from lib.ontology.jsonld_context_loader import register_local_loader

    register_local_loader()  # now ed4all.dev context resolves locally

    payload = {...}  # a Courseforge JSON-LD emit
    expanded = jsonld.expand(payload)  # works offline, no network

Usage in tests: prefer the fixture form (``register_local_loader`` is
idempotent, so calling it per-module in ``conftest.py`` is safe).

Design:

* No global mutation at import time — the caller opts in via
  :func:`register_local_loader`, preserving pyld's default behavior
  for processes that don't want the override.
* The bundled loader chain-calls the previous loader on cache miss, so
  the local override is additive and composable with any other custom
  loaders already registered.
* The URL set is data-driven (``_LOCAL_CONTEXT_BINDINGS`` dict) so
  future vocabulary files (e.g. a SKOS Bloom concept scheme, a
  Courseforge v2 context) land by adding one entry rather than forking
  the loader logic.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pyld import jsonld

__all__ = [
    "CANONICAL_COURSEFORGE_CONTEXT_URL",
    "register_local_loader",
    "load_courseforge_context",
]


#: The canonical logical identifier for the Courseforge JSON-LD context.
#: This URL is what ``generate_course.py`` stamps on every emitted payload's
#: ``@context`` key, and what the JSON Schema's ``@context`` property
#: enforces as a ``const``. Bound to the local file below.
CANONICAL_COURSEFORGE_CONTEXT_URL = "https://ed4all.dev/ns/courseforge/v1"


_REPO_ROOT = Path(__file__).resolve().parents[2]


#: URL → relative-path-from-repo-root bindings for every vocabulary file
#: we ship. Future waves (e.g., v2 context, a SKOS concept scheme for
#: Bloom verbs) extend this dict rather than fork the loader.
_LOCAL_CONTEXT_BINDINGS: Dict[str, Path] = {
    CANONICAL_COURSEFORGE_CONTEXT_URL: _REPO_ROOT
    / "schemas"
    / "context"
    / "courseforge_v1.jsonld",
}


@lru_cache(maxsize=1)
def load_courseforge_context() -> Dict[str, Any]:
    """Return the parsed Courseforge @context document.

    Cached — the context file is immutable for the lifetime of the
    process, and pyld may call the loader many times per validation run.
    """
    path = _LOCAL_CONTEXT_BINDINGS[CANONICAL_COURSEFORGE_CONTEXT_URL]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _raise_no_fallback(url: str, options: Optional[Dict[str, Any]] = None) -> None:
    """Sentinel fallback — raises with a clear message for unbound URLs."""
    raise jsonld.JsonLdError(
        f"No document loader registered for {url!r}. The Courseforge local "
        f"loader only serves URLs listed in _LOCAL_CONTEXT_BINDINGS; install "
        f"requests and call register_local_loader(preserve_existing=True) "
        f"with a prior network loader if you need HTTP resolution, or add the "
        f"URL to _LOCAL_CONTEXT_BINDINGS.",
        "jsonld.LoadDocumentError",
        code="loading document failed",
    )


def _make_loader(
    previous_loader: Optional[Callable[..., Any]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Build a pyld document loader that serves local bindings first.

    The returned callable matches pyld's loader contract:
    ``loader(url, options) -> {"contextUrl", "documentUrl", "document"}``.
    Unknown URLs fall through to ``previous_loader``; if no previous
    loader is supplied, unknown URLs raise a clear ``JsonLdError`` rather
    than silently failing or requiring ``requests`` as an implicit dep.
    Consumers who want HTTP fallback must install ``requests`` and pass
    a pre-built loader (e.g. ``jsonld.requests_document_loader()``) via
    :func:`register_local_loader`'s ``preserve_existing=True`` path.
    """
    fallback = previous_loader or _raise_no_fallback

    def _loader(url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if url in _LOCAL_CONTEXT_BINDINGS:
            if url == CANONICAL_COURSEFORGE_CONTEXT_URL:
                document = load_courseforge_context()
            else:
                with open(_LOCAL_CONTEXT_BINDINGS[url], encoding="utf-8") as f:
                    document = json.load(f)
            return {
                "contextUrl": None,
                "documentUrl": url,
                "document": document,
            }
        return fallback(url, options or {})

    return _loader


def register_local_loader(
    *, preserve_existing: bool = True
) -> Callable[..., Dict[str, Any]]:
    """Install the local-first document loader on pyld.

    Args:
        preserve_existing: When True (default), any currently-registered
            pyld document loader is chain-called for URLs not in our
            local bindings, so this override composes with other custom
            loaders already in place. When False, unknown URLs raise
            ``JsonLdError`` rather than attempting HTTP — keeps tests
            hermetic without pulling in ``requests``.

    Returns:
        The loader that was just installed.
    """
    existing = None
    if preserve_existing:
        try:
            existing = jsonld._default_document_loader  # type: ignore[attr-defined]
        except AttributeError:
            existing = None
        # If the current default IS already a local-binding loader
        # (register_local_loader called twice), skip chaining to avoid
        # infinite wrapping.
        if existing is not None and getattr(existing, "_cf_local_loader", False):
            existing = None
    loader = _make_loader(previous_loader=existing)
    loader._cf_local_loader = True  # type: ignore[attr-defined]
    jsonld.set_document_loader(loader)
    return loader
