"""HTTP QUERY method support for the retrieval-shaped GUI endpoints.

QUERY is the IETF-standardized request method that is BOTH *safe* and
*idempotent* yet carries a request body — exactly the shape of our
retrieval-style endpoints (learner ``/ask`` and operator ``/query``). Those ride
POST today only because a GET cannot carry a structured query body; semantically
they read, they do not mutate. QUERY is the correct method for them, so QUERY is
now the CANONICAL method and POST is kept as a deprecated back-compat alias that
carries a ``Deprecation`` response header.

IETF status (pinned 2026-07-19): the QUERY method is defined by **RFC 10008**
("The HTTP QUERY Method", Standards Track, June 2026) — it graduated from the
``draft-ietf-httpbis-safe-method-w-body`` working-group draft. Per the abstract,
"A QUERY requests that the request target process the enclosed content ... in a
safe and idempotent manner and then respond with the result of that processing."
Because QUERY is a non-safelisted method, a CROSS-ORIGIN QUERY triggers a CORS
preflight (``Access-Control-Allow-Methods`` must list it); the same-origin GUI
SPA is unaffected. Front-end callers therefore send QUERY and downgrade ONCE to
POST for the session if a client/proxy rejects the method (405/501 or a
network-layer method block).

This module is the single place the QUERY⇄POST contract is declared, imported by
every retrieval-shaped router so the method set and the deprecation signalling
never drift between endpoints.
"""

from __future__ import annotations

from typing import Any, List

from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# QUERY is canonical; POST is the deprecated-but-accepted alias. A single ordered
# list wired into every retrieval-shaped ``api_route(methods=...)`` so the two
# endpoints expose an identical method surface.
QUERY_METHODS: List[str] = ["QUERY", "POST"]


def apply_deprecation_if_post(request: Request, result: Any) -> Any:
    """Return ``result``; tag a ``Deprecation`` header when the method is POST.

    A QUERY response is returned unchanged. A POST response (the deprecated
    alias) gains ``Deprecation: true`` plus a ``Link; rel="successor-version"``
    pointing clients at the QUERY method — the widely-deployed deprecation signal
    (RFC 9745 "The Deprecation HTTP Response Header Field"; the boolean token
    predates the RFC's dated form and stays the interoperable choice when no
    deprecation date is asserted). A dict/primitive return is normalized to a
    ``JSONResponse`` so the header can attach; a return that is already a
    ``Response`` (the typed-error path) is tagged in place.
    """
    if request.method != "POST":
        return result
    resp = (
        result
        if isinstance(result, Response)
        else JSONResponse(content=jsonable_encoder(result))
    )
    resp.headers["Deprecation"] = "true"
    resp.headers.setdefault(
        "Link",
        '</api>; rel="successor-version"; title="send this request with the HTTP QUERY method"',
    )
    return resp


__all__ = ["QUERY_METHODS", "apply_deprecation_if_post"]
