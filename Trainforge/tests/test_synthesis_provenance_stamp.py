"""W9.1 regression: cloud seats stamp a truthful (non-``local``) provenance.

Licensing-correctness bug: a hosted-cloud Llama-3.3 / DeepSeek seat
(``groq`` / ``fireworks`` / ``deepseek``) previously recorded its output
provenance as ``local`` — the license-clean value a licensing auditor
reads as "Apache-2.0 local OSS, training-permitted, clean". Recording a
cloud-API-generated artifact as ``local`` is a licensing LIE.

Two layers are asserted:

1. The registry (``config/endpoints.yaml`` via ``lib.llm.endpoints``):
   every hosted-cloud seat's ``provenance_provider`` is a truthful
   non-``local`` value; the genuine ``local`` seat still stamps ``local``.
2. The Trainforge synthesis provenance-stamping code
   (``SynthesisProvider``): the value written to ``out["provider"]`` is the
   canonical registry provenance value, so a cloud seat never stamps
   ``local`` and the local seat still does.

The frozen closed provenance set
{anthropic, claude_session, deterministic, local, nvidia, together} is
UNCHANGED by the fix (cloud seats map onto the existing cloud ``together``
value), so the Touch.provider codegen/drift contract stays green.
"""

from __future__ import annotations

import httpx
import pytest

from lib.llm.endpoints import (
    load_endpoint_registry,
    provenance_provider_names,
)


def _provenance(name: str) -> str:
    # The declared provenance stamp — a pure row read (no key resolution,
    # so it works for api_key_required cloud seats in CI without keys).
    return str(load_endpoint_registry()[name]["provenance_provider"])


# Hosted-cloud OpenAI-compatible seats: these must NEVER stamp "local".
_CLOUD_SEATS = ("groq", "fireworks", "deepseek", "together", "nvidia")


@pytest.mark.parametrize("name", _CLOUD_SEATS)
def test_cloud_seat_provenance_is_not_local(name):
    assert _provenance(name) != "local", (
        f"cloud seat {name!r} stamps the license-clean 'local' provenance — "
        f"licensing LIE (W9.1)"
    )


def test_cloud_llama_seats_map_to_cloud_together_provenance():
    # groq/fireworks host Llama-3.3-70B, deepseek hosts deepseek-chat — all
    # hosted-cloud OSS teachers, mapped onto the existing cloud "together"
    # provenance value so the frozen set stays intact.
    for name in ("groq", "fireworks", "deepseek"):
        assert _provenance(name) == "together"


def test_local_seat_still_stamps_local():
    assert _provenance("local") == "local"


def test_frozen_provenance_set_unchanged():
    # The W9.1 remap onto an existing value must NOT grow the closed set
    # (that would force codegen into blocks.py + the JSON-LD / SHACL sites).
    assert set(provenance_provider_names()) == {
        "anthropic",
        "claude_session",
        "deterministic",
        "local",
        "nvidia",
        "together",
    }


# ---------------------------------------------------------------------------
# SynthesisProvider stamps the canonical provenance value onto pairs.
# Construction with an injected client is network-free (no HTTP, no key).
# ---------------------------------------------------------------------------


def _fake_client() -> httpx.Client:
    # A bare client is enough: construction resolves identity from the
    # registry and never makes a request. No transport handler needed
    # because no HTTP is issued during __init__.
    return httpx.Client()


def test_synthesis_provider_local_stamps_local():
    from Trainforge.generators._synthesis_provider import SynthesisProvider

    p = SynthesisProvider(provider="local", client=_fake_client())
    assert p._provenance_provider == "local"


@pytest.mark.parametrize("name", ("groq", "fireworks", "deepseek"))
def test_synthesis_provider_cloud_stamps_non_local(name):
    from Trainforge.generators._synthesis_provider import SynthesisProvider

    p = SynthesisProvider(provider=name, client=_fake_client())
    assert p._provenance_provider != "local"
    assert p._provenance_provider == "together"
    # The raw endpoint name is preserved for audit fidelity (rationale
    # + error messages), distinct from the stamped provenance value.
    assert p._provider_name == name


def test_synthesis_provider_together_and_nvidia_are_identity():
    from Trainforge.generators._synthesis_provider import SynthesisProvider

    assert (
        SynthesisProvider(provider="together", client=_fake_client())._provenance_provider
        == "together"
    )
    assert (
        SynthesisProvider(provider="nvidia", client=_fake_client())._provenance_provider
        == "nvidia"
    )


def test_unknown_name_falls_back_to_raw_name():
    # A name absent from the registry (a test double) keeps the raw name so
    # non-registry callers never crash.
    from Trainforge.generators._synthesis_provider import (
        _resolve_provenance_provider,
    )

    assert "no-such-endpoint-xyz" not in load_endpoint_registry()
    assert _resolve_provenance_provider("no-such-endpoint-xyz") == "no-such-endpoint-xyz"
