"""Intent-dispatch tool wrapper (Wave 78 Worker C).

Thin wrapper around :func:`LibV2.tools.intent_router.dispatch` that
matches the shape downstream MCP-tool callers expect:

    intent_dispatch_query(slug, query, top_k=5) -> Dict[str, Any]

Mirrors ``ed4all libv2 ask`` (CLI surface) so a tool-call integration
gets the same routing behavior as a human invocation. The router does
heuristic intent classification (no LLM dependency), and dispatches
to the right Wave 77 backend (chunk_query / tutoring_tools /
pedagogy_graph walk / chunk-text BM25).

Like ``MCP/tools/tutoring_tools.py``, this module is *not* registered
with ``@mcp.tool()`` — it's a pure-Python helper available to tool-call
integrations and CLI wrappers. Decorating it would require deciding on
a stable schema; the routing envelope is rich enough that we want
direct callers to bind to the typed dict shape rather than a marshalled
JSON-RPC schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from LibV2.tools.intent_router import dispatch as _dispatch


__all__ = ["intent_dispatch_query"]


def intent_dispatch_query(
    slug: str,
    query: str,
    top_k: int = 5,
    *,
    courses_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Classify and dispatch ``query`` against the LibV2 archive ``slug``.

    Returns the canonical envelope from
    :func:`LibV2.tools.intent_router.dispatch`::

        {
            "query": str,
            "slug": str,
            "intent_class": str,   # one of INTENT_CLASSES
            "confidence": float,
            "route": str,          # human-readable backend descriptor
            "source_path": str,    # alias of ``route`` for ChatGPT-review parity
            "entities": dict,      # full extract_entities output
            "results": list,       # backend-specific shape
        }

    Empty ``query`` or unknown slug return an envelope with
    ``results=[]`` rather than raising — callers can detect "intent
    classified, but archive missing" without exception handling.
    """
    return _dispatch(query or "", slug, top_k=top_k, courses_root=courses_root)
