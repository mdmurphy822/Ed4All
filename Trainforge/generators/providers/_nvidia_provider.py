#!/usr/bin/env python3
"""NVIDIA hosted OpenAI-compatible inference provider — module constants.

NVIDIA's hosted inference API (``https://integrate.api.nvidia.com/v1``)
is an OpenAI-compatible ``/chat/completions`` endpoint, exactly the wire
shape the existing ``together`` / ``local`` providers already speak via
:class:`Trainforge.generators.providers._openai_compatible_client.OpenAICompatibleClient`.

This module is the registry-entry sibling of
:mod:`Trainforge.generators.providers._together_provider` — it carries only the
provider-identity constants (default base URL, env-var names, default
model). The Courseforge generator base
(:class:`Courseforge.generators._base._BaseLLMProvider`) imports these
constants to wire its ``nvidia`` dispatch branch through the shared
OpenAI-compatible client, so adding NVIDIA is a registry-entry change,
not a new dispatch subclass (per the project's "new providers are
registry entries, not subclasses" contract).

Usage surface: NVIDIA is selected as the Courseforge **rewrite LARGE /
escalation tier** via the capability-tier chain in
``Courseforge/config/block_routing.license_clean.yaml`` (a YAML tier
with ``provider: nvidia``). The small + medium tiers stay on the local
7B/14B models — only blocks that escalate to the ``large`` rewrite tier
send a request to the NVIDIA endpoint.

The NVIDIA chat-completion returns a CLEAN ``message.content`` (the
requested HTML) with the model's chain-of-thought isolated in a separate
``message.reasoning_content`` field, so the standard OpenAI-compatible
``_extract_text`` (which reads ``message.content``) returns clean output
with no ``<think>`` pollution — no special handling required.

**Licensing note:** NVIDIA's hosted API here generates COURSE CONTENT
(published HTML), which is generated *product*, NOT the training-data
corpus. The training-data synthesis surfaces (instruction / preference
pairs) keep their own license-clean provider posture (local Qwen /
Together OSS). See ``docs/LICENSING.md`` § "Content-generation
providers".
"""

from __future__ import annotations

# Default OpenAI-compatible base URL for NVIDIA's hosted inference API.
# The ``/chat/completions`` suffix is appended at request time by
# ``OpenAICompatibleClient`` / the provider's ``api_url`` property.
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Backwards-compat: full chat-completions URL constant, mirroring the
# ``TOGETHER_API_URL`` convention for callers that want the literal.
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Default large-tier model (exists in the NVIDIA catalog). Returns clean
# ``message.content`` with reasoning isolated in ``reasoning_content``.
DEFAULT_SYNTHESIS_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

# Env-var contract. The API key is REQUIRED (the hosted endpoint
# authenticates every request) — never hardcoded. Base URL + model are
# env-overridable so an operator can point at a different NVIDIA-hosted
# model or a self-hosted NIM endpoint without editing the YAML.
ENV_API_KEY = "NVIDIA_API_KEY"
ENV_BASE_URL = "NVIDIA_BASE_URL"
ENV_MODEL = "NVIDIA_LARGE_MODEL"


__all__ = [
    "DEFAULT_BASE_URL",
    "NVIDIA_API_URL",
    "DEFAULT_SYNTHESIS_MODEL",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
]
