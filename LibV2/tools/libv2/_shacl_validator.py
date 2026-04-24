"""Wave 70 — vendored SHACL validator for LibV2 import gate.

LibV2 is sandboxed from ``lib/`` (see ``LibV2/CLAUDE.md``) so we can't
reach into ``lib.ontology.jsonld_context_loader`` here. This module
vendors a minimal version of the Wave 64 loader — just enough to serve
the Courseforge JSON-LD @context locally when pyld asks for it — plus a
thin ``validate_payload`` helper that converts a JSON-LD-shaped dict to
RDF and runs it through ``schemas/context/courseforge_v1.shacl.ttl``.

The vendored form is deliberately tiny: one URL binding, one loader,
one ``validate_payload`` call. When the Wave 64 loader changes we
reconcile by hand — LibV2's sandbox is the feature, not an
inconvenience.

All heavy deps (pyld, pyshacl, rdflib) are imported lazily inside the
validator, and the caller is expected to handle ``ImportError`` so a
bare LibV2 install can still run ``libv2 import`` without the RDF
toolchain present.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "CANONICAL_COURSEFORGE_CONTEXT_URL",
    "ShaclDepsMissing",
    "register_local_loader",
    "validate_manifest_shacl",
]

#: Canonical logical identifier for the Courseforge JSON-LD context.
#: Mirrors ``lib.ontology.jsonld_context_loader.CANONICAL_COURSEFORGE_CONTEXT_URL``.
CANONICAL_COURSEFORGE_CONTEXT_URL = "https://ed4all.dev/ns/courseforge/v1"


# LibV2/tools/libv2/_shacl_validator.py → repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTEXT_PATH = _REPO_ROOT / "schemas" / "context" / "courseforge_v1.jsonld"
_SHAPES_PATH = _REPO_ROOT / "schemas" / "context" / "courseforge_v1.shacl.ttl"


class ShaclDepsMissing(ImportError):
    """Raised when pyld / pyshacl / rdflib aren't importable.

    Callers should catch this and degrade gracefully — the import flow
    must not hard-fail when the RDF toolchain is absent.
    """


@lru_cache(maxsize=1)
def _load_context_document() -> Dict[str, Any]:
    with open(_CONTEXT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _make_loader(previous_loader: Optional[Callable[..., Any]] = None):
    """Build a pyld document loader that serves the local @context file.

    Mirrors the chain-calling behavior of the Wave 64 loader: the local
    binding always wins for the canonical URL; anything else falls
    through to ``previous_loader`` (or raises a clear JsonLdError).
    """
    from pyld import jsonld  # lazy; this whole module is skippable

    def _fallback(url: str, options: Optional[Dict[str, Any]] = None):
        if previous_loader is not None:
            return previous_loader(url, options or {})
        raise jsonld.JsonLdError(
            f"No document loader registered for {url!r}. LibV2's vendored "
            "loader only serves the Courseforge @context URL; install a "
            "requests-backed loader if you need HTTP resolution.",
            "jsonld.LoadDocumentError",
            code="loading document failed",
        )

    def _loader(url: str, options: Optional[Dict[str, Any]] = None):
        if url == CANONICAL_COURSEFORGE_CONTEXT_URL:
            return {
                "contextUrl": None,
                "documentUrl": url,
                "document": _load_context_document(),
            }
        return _fallback(url, options or {})

    return _loader


def register_local_loader(*, preserve_existing: bool = True):
    """Install the vendored local-first document loader on pyld.

    Idempotent — calling twice is safe. Returns the installed loader so
    tests can assert on it.
    """
    from pyld import jsonld

    existing = None
    if preserve_existing:
        try:
            existing = jsonld._default_document_loader  # type: ignore[attr-defined]
        except AttributeError:
            existing = None
        # Don't chain-wrap ourselves.
        if existing is not None and getattr(existing, "_libv2_local_loader", False):
            existing = None
    loader = _make_loader(previous_loader=existing)
    loader._libv2_local_loader = True  # type: ignore[attr-defined]
    jsonld.set_document_loader(loader)
    return loader


def _ensure_deps():
    """Import pyld, pyshacl, rdflib or raise ``ShaclDepsMissing``.

    Kept in one place so the import flow and tests can share the same
    skip-on-missing behavior.
    """
    try:
        import pyld  # noqa: F401
        import pyshacl  # noqa: F401
        import rdflib  # noqa: F401
    except ImportError as exc:
        raise ShaclDepsMissing(str(exc)) from exc


def validate_manifest_shacl(
    manifest_dict: Dict[str, Any],
    *,
    context_url: str = CANONICAL_COURSEFORGE_CONTEXT_URL,
) -> Tuple[bool, str]:
    """SHACL-validate a manifest-shaped dict against the Courseforge shapes.

    Args:
        manifest_dict: The manifest payload. The dict is copied; a
            ``@context`` key is added (pointing at ``context_url``) so
            JSON-LD expansion has something to resolve.
        context_url: Override of the canonical Courseforge context URL.
            Exposed for test isolation — production callers use the
            default.

    Returns:
        ``(conforms, report_text)``. ``conforms=True`` means the RDF
        graph produced by expanding the payload through our @context
        satisfies every shape in ``courseforge_v1.shacl.ttl``.

    Raises:
        ShaclDepsMissing: If pyld / pyshacl / rdflib aren't importable.
            Callers should catch this and skip validation gracefully.
    """
    _ensure_deps()

    from pyld import jsonld
    import pyshacl
    from rdflib import Graph

    # Make sure the @context resolves locally even in a hermetic run.
    register_local_loader()

    payload = dict(manifest_dict)
    payload["@context"] = context_url

    # JSON-LD → N-Quads → rdflib Graph. Mirrors the pipeline in
    # schemas/tests/test_courseforge_shacl_shapes.py.
    nq = jsonld.to_rdf(payload, {"format": "application/n-quads"})
    data_graph = Graph()
    data_graph.parse(data=nq, format="nquads")

    shapes_graph = Graph()
    shapes_graph.parse(_SHAPES_PATH, format="turtle")

    conforms, _results_graph, results_text = pyshacl.validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        js=False,
        debug=False,
    )
    return bool(conforms), str(results_text)
