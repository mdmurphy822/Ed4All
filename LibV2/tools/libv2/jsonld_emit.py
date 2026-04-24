"""Wave 70 — JSON-LD emit for retrieval results.

Provides an RDF-compatible projection of a LibV2 ``RetrievalResult`` so
downstream Pearson / LRMI / CASE consumers can pipe results into an RDF
tool without writing a custom mapping layer.

Predicate alignment (see ``schemas/context/courseforge_v1.jsonld``):

=====================  =====================  =================================
RetrievalResult field   JSON-LD key            IRI (expanded)
=====================  =====================  =================================
``chunk_id``            ``identifier``         ``schema:identifier``
``text``                ``text``               ``schema:text``
``score``               ``retrievalScore``     ``ed4all:retrievalScore``
``bloom_level``         ``bloomLevel``         ``ed4all:bloomLevel``
``concept_tags``        ``keywords``           ``schema:keywords``
``learning_outcome_refs`` ``derivedFromObjective`` ``ed4all:derivedFromObjective``
``source``              ``isBasedOn``          ``schema:isBasedOn``
``course_slug``         ``courseSlug``         ``ed4all:courseSlug``
``domain``              ``domain``             ``ed4all:domain``
``chunk_type``          ``chunkType``          ``ed4all:chunkType``
``difficulty``          ``difficulty``         ``ed4all:difficulty``
``tokens_estimate``     ``tokensEstimate``     ``ed4all:tokensEstimate``
=====================  =====================  =================================

The ``@type`` is ``ed4all:RetrievalResult`` — a custom class, simpler
than overloading ``schema:QuantitativeValue`` for the whole envelope and
more expressive for downstream SHACL shapes that want to target the
retrieval surface specifically.

The default ``@context`` URL is the canonical Courseforge context. All
terms used by the retrieval emit are defined there (Wave 62+67) — we
piggyback on that vocabulary instead of minting a retrieval-specific
one so a single document loader resolves the whole emit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .retriever import RetrievalResult  # circular at runtime, fine at type

# Default canonical context URL — same URL Courseforge stamps on its
# page metadata. Consumers with a local loader installed (e.g. via
# ``_shacl_validator.register_local_loader``) resolve it offline.
DEFAULT_CONTEXT_URL = "https://ed4all.dev/ns/courseforge/v1"


def retrieval_result_to_jsonld(
    result: "RetrievalResult",
    *,
    context_url: str = DEFAULT_CONTEXT_URL,
) -> Dict[str, Any]:
    """Project a ``RetrievalResult`` into a JSON-LD document.

    The emit is additive — ``RetrievalResult.to_dict()`` stays as the
    back-compat wire format for existing Trainforge / libv2_bridge
    consumers. This function produces a parallel shape suitable for
    feeding pyld / rdflib.

    Args:
        result: The result to serialize.
        context_url: Override of the canonical context URL.

    Returns:
        A dict with ``@context``, ``@type``, and Schema.org / ed4all:
        predicates for every populated field on the result. ``None``
        fields are omitted so the emit stays compact and JSON-LD
        expansion doesn't see empty literals.
    """
    # Build the payload. Order mirrors ``to_dict`` for readability.
    out: Dict[str, Any] = {
        "@context": context_url,
        "@type": "ed4all:RetrievalResult",
        "identifier": result.chunk_id,
        "text": result.text,
        "retrievalScore": result.score,
        "courseSlug": result.course_slug,
        "domain": result.domain,
        "chunkType": result.chunk_type,
        "tokensEstimate": result.tokens_estimate,
    }
    if result.difficulty is not None:
        out["difficulty"] = result.difficulty
    # concept_tags and learning_outcome_refs — always emit when non-empty
    # so consumers see the @container:@set shape even for empty lists.
    if result.concept_tags:
        # ``keywords`` is already a @set container in the Courseforge
        # context; passing a list keeps that semantics.
        out["keywords"] = list(result.concept_tags)
    if result.learning_outcome_refs:
        # Mint stable IRIs for LO refs so ``derivedFromObjective`` is an
        # IRI predicate (schema:competencyRequired / ed4all:derivedFromObjective).
        # A short ed4all:lo/ prefix keeps the refs opaque but linkable.
        out["derivedFromObjective"] = [
            _lo_ref_to_iri(ref) for ref in result.learning_outcome_refs
        ]
    if result.bloom_level:
        # The ``bloomLevel`` term maps to @type: @vocab over
        # https://ed4all.dev/vocab/bloom# — pyld expands bare tokens
        # ("apply", "remember", ...) to the full IRI automatically.
        out["bloomLevel"] = str(result.bloom_level).lower()
    if result.source:
        # ``isBasedOn`` on schema.org accepts a node (not just a URL);
        # keep the nested source dict as-is. Rdflib lifts it into a
        # blank node when expanded.
        out["isBasedOn"] = dict(result.source)

    # retrievalScore doesn't have a term in the Courseforge context —
    # inject an inline key binding so pyld can expand it. The context
    # merge is additive: consumers can still resolve the main @context
    # URL, we just augment with the retrieval-specific predicate.
    # Do this by replacing the plain-URL @context with a wrapper.
    out["@context"] = [
        context_url,
        {
            "retrievalScore": {
                "@id": "https://ed4all.dev/ns/courseforge/v1#retrievalScore",
                "@type": "http://www.w3.org/2001/XMLSchema#decimal",
            },
            "tokensEstimate": {
                "@id": "https://ed4all.dev/ns/courseforge/v1#tokensEstimate",
                "@type": "http://www.w3.org/2001/XMLSchema#integer",
            },
            "courseSlug": "https://ed4all.dev/ns/courseforge/v1#courseSlug",
            "domain": "https://ed4all.dev/ns/courseforge/v1#domain",
            "chunkType": "https://ed4all.dev/ns/courseforge/v1#chunkType",
            "difficulty": "https://ed4all.dev/ns/courseforge/v1#difficulty",
            "derivedFromObjective": {
                "@id": "https://ed4all.dev/ns/courseforge/v1#derivedFromObjective",
                "@type": "@id",
                "@container": "@set",
            },
            "text": "http://schema.org/text",
            "keywords": {"@id": "http://schema.org/keywords", "@container": "@set"},
            "identifier": "http://schema.org/identifier",
            "isBasedOn": {"@id": "http://schema.org/isBasedOn", "@type": "@id"},
            "ed4all": "https://ed4all.dev/ns/courseforge/v1#",
            "RetrievalResult": "ed4all:RetrievalResult",
        },
    ]
    return out


def _lo_ref_to_iri(lo_ref: str) -> str:
    """Turn an LO id like ``TO-03`` / ``co-03`` into a stable IRI.

    Uses the ed4all:lo/ namespace. Case is preserved (the @context
    doesn't downcase) but the canonical pattern matches
    ``courseforge_v1.shacl.ttl``'s parentObjective check.
    """
    ref = str(lo_ref).strip()
    # Handle already-IRI inputs gracefully.
    if ref.startswith(("http://", "https://")):
        return ref
    return f"https://ed4all.dev/ns/courseforge/v1/lo/{ref}"
