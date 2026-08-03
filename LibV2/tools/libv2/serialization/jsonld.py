"""JSON-LD serialization for LibV2 retrieval results.

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

The default ``@context`` URL is the canonical Courseforge context. The
retrieval projection reuses that vocabulary so one document loader resolves
the complete output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..retriever import RetrievalResult

# Default canonical context URL — same URL Courseforge stamps on its
# page metadata. Consumers with a local loader installed (e.g. via
# ``_shacl_validator.register_local_loader``) resolve it offline.
DEFAULT_CONTEXT_URL = "https://ed4all.dev/ns/courseforge/v1"


def retrieval_result_to_jsonld(
    result: RetrievalResult,
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
    # Keep field order aligned with ``RetrievalResult.to_dict`` for readability.
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
    # Emit list-backed semantic fields only when they contain values.
    if result.concept_tags:
        # ``keywords`` is already a @set container in the Courseforge
        # context; passing a list keeps that semantics.
        out["keywords"] = list(result.concept_tags)
    if result.learning_outcome_refs:
        # Mint linkable IRIs while preserving opaque learning-objective IDs.
        out["derivedFromObjective"] = [
            _lo_ref_to_iri(ref) for ref in result.learning_outcome_refs
        ]
    if result.bloom_level:
        # Normalize the vocabulary token before JSON-LD expansion.
        out["bloomLevel"] = str(result.bloom_level).lower()
    if result.source:
        # Preserve structured provenance as the node behind ``isBasedOn``.
        out["isBasedOn"] = dict(result.source)

    # Extend the canonical context with retrieval-specific predicates.
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
    """Turn a learning-objective ID into a stable IRI.

    Uses the ed4all:lo/ namespace. Case is preserved (the @context
    doesn't downcase) but the canonical pattern matches
    ``courseforge_v1.shacl.ttl``'s parentObjective check.
    """
    ref = str(lo_ref).strip()
    # Handle already-IRI inputs gracefully.
    if ref.startswith(("http://", "https://")):
        return ref
    return f"https://ed4all.dev/ns/courseforge/v1/lo/{ref}"
