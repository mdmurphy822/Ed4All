"""Phase 2: golden-output PARITY tests for the LLM-agnostic SynthesisProvider.

The safety proof for the registry-driven provider: feeding the SAME
(draft, chunk) + the SAME well-formed JSON completion to BOTH the new
:class:`SynthesisProvider` and the per-vendor leaf it supersedes yields
a byte-identical ``out`` dict.

- ``SynthesisProvider(provider="local", terse_prompts=True)`` vs
  :class:`LocalSynthesisProvider` (incl. a preserve-token +
  length/preserve-remediation case).
- ``SynthesisProvider(provider="together", terse_prompts=False)`` vs
  :class:`TogetherSynthesisProvider` (well-formed). On a MALFORMED /
  fenced response the agnostic provider RECOVERS (lenient parse) where
  strict-together would have retried/exhausted — asserted as an
  improvement, not a match.

Plus: the ``out["provider"]`` tag equals the constructor provider
(local/together/nvidia); the constructor resolves the client BY NAME
from the endpoint registry (base_url flows from the row); the preserve
gate is a no-op when ``preserve_tokens`` is empty (hosted parity).

All tests use ``httpx.MockTransport`` — no real network call fires.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._synthesis_provider import (  # noqa: E402
    DEFAULT_PROVIDER,
    SynthesisProvider,
    SynthesisProviderError,
    build_synthesis_provider,
)
from Trainforge.generators._local_provider import (  # noqa: E402
    DEFAULT_SYNTHESIS_MODEL,
    LocalSynthesisProvider,
)
from Trainforge.generators._together_provider import (  # noqa: E402
    DEFAULT_SYNTHESIS_MODEL as TOGETHER_DEFAULT_MODEL,
    TogetherSynthesisProvider,
)


# ---------------------------------------------------------------------------
# Env hygiene — pin a deterministic registry resolution per test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "LOCAL_SYNTHESIS_API_KEY",
        "LOCAL_SYNTHESIS_BASE_URL",
        "LOCAL_SYNTHESIS_MODEL",
        "TOGETHER_API_KEY",
        "TOGETHER_SYNTHESIS_MODEL",
        "TRAINFORGE_SYNTHESIS_MODEL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "NVIDIA_LARGE_MODEL",
        "ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Helpers (mirror the local / together suites)
# ---------------------------------------------------------------------------


def _success_body(content: str) -> dict:
    return {
        "id": "cmpl-agnostic-test",
        "model": DEFAULT_SYNTHESIS_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 180,
            "completion_tokens": 90,
            "total_tokens": 270,
        },
    }


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _client_yielding(*responses: httpx.Response) -> httpx.Client:
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] = i + 1
        return responses[i]

    return _make_client(handler)


def _fixed_client(content: str) -> httpx.Client:
    """A client that returns the SAME success body for every request."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(content))

    return _make_client(handler)


def _instruction_draft() -> dict:
    return {
        "prompt": "Define the central concept behind topic X.",
        "completion": (
            "Topic X is a central idea in this material. Learners should be "
            "able to recall and restate it."
        ),
        "chunk_id": "chunk_001",
        "lo_refs": ["TO-01"],
        "bloom_level": "remember",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "",
        "template_id": "remember.explanation",
        "provider": "mock",
        "schema_version": "v1",
    }


def _preference_draft() -> dict:
    return {
        "prompt": "Explain topic X clearly enough to avoid the misconception.",
        "chosen": (
            "Topic X is a foundational idea in the course material; the "
            "correct framing emphasises its grounding rules."
        ),
        "rejected": (
            "Topic X is mostly a theoretical curiosity; you can safely "
            "ignore the formal definition for everyday work."
        ),
        "misconception_id": "mc_abc",
        "chunk_id": "chunk_001",
        "lo_refs": ["TO-01"],
        "seed": 17,
        "decision_capture_id": "",
        "rejected_source": "misconception",
        "provider": "mock",
        "schema_version": "v1",
    }


def _chunk() -> dict:
    return {
        "id": "chunk_001",
        "text": (
            "Topic X is the foundational concept in chapter one. Learners "
            "encounter it in every subsequent chapter, and the correct "
            "framing anchors all later examples."
        ),
        "learning_outcome_refs": ["TO-01"],
    }


_INSTRUCTION_COMPLETION = json.dumps({
    "prompt": "Recall the central idea introduced for topic X in this material.",
    "completion": (
        "Topic X is the foundational concept of the chapter; recalling its "
        "formal definition is the first step toward applying it in "
        "subsequent material."
    ),
})

_PREFERENCE_COMPLETION = json.dumps({
    "prompt": (
        "Briefly explain topic X to a learner about to encounter the "
        "common misconception."
    ),
    "chosen": (
        "Topic X is the foundational concept of the chapter; the course "
        "material grounds every later example in its formal definition."
    ),
    "rejected": (
        "Topic X is essentially optional; my experience says you can skip "
        "the formal definition without losing much."
    ),
})


# ---------------------------------------------------------------------------
# LOCAL parity (terse prompts)
# ---------------------------------------------------------------------------


def test_local_instruction_parity_byte_identical():
    """SynthesisProvider(local) == LocalSynthesisProvider on a well-formed
    instruction response."""
    agnostic = SynthesisProvider(
        provider="local", terse_prompts=True,
        client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    leaf = LocalSynthesisProvider(client=_fixed_client(_INSTRUCTION_COMPLETION))

    out_a = agnostic.paraphrase_instruction(_instruction_draft(), _chunk())
    out_l = leaf.paraphrase_instruction(_instruction_draft(), _chunk())

    assert out_a == out_l
    assert out_a["provider"] == "local"


def test_local_preference_parity_byte_identical():
    agnostic = SynthesisProvider(
        provider="local", terse_prompts=True,
        client=_fixed_client(_PREFERENCE_COMPLETION),
    )
    leaf = LocalSynthesisProvider(client=_fixed_client(_PREFERENCE_COMPLETION))

    out_a = agnostic.paraphrase_preference(_preference_draft(), _chunk())
    out_l = leaf.paraphrase_preference(_preference_draft(), _chunk())

    assert out_a == out_l
    assert out_a["provider"] == "local"


def test_local_parity_with_preserve_tokens_and_remediation():
    """The preserve-token + preserve-remediation path matches local exactly:
    strict floor (1.0), first response drops the token, second carries it —
    both providers retry-then-accept identically."""
    bad = json.dumps({
        "prompt": "Explain how the node shape concept applies to topic X.",
        "completion": (
            "The node-shape idea constrains node-typed entities; a learner "
            "applies it by validating instances against the declared "
            "constraint and checking the datatype on each property."
        ),
    })
    good = json.dumps({
        "prompt": "Explain how sh:NodeShape applies to topic X.",
        "completion": (
            "sh:NodeShape constrains node-typed nodes; learners apply it by "
            "validating instances against the declared shape constraint and "
            "the sh:datatype rule on each property."
        ),
    })
    seq = (
        httpx.Response(200, json=_success_body(bad)),
        httpx.Response(200, json=_success_body(good)),
    )
    preserve = ["sh:NodeShape", "sh:datatype"]

    agnostic = SynthesisProvider(
        provider="local", terse_prompts=True, min_preserve_rate=1.0,
        client=_client_yielding(*seq),
    )
    leaf = LocalSynthesisProvider(
        min_preserve_rate=1.0, client=_client_yielding(*seq),
    )

    out_a = agnostic.paraphrase_instruction(
        _instruction_draft(), _chunk(), preserve_tokens=preserve,
    )
    out_l = leaf.paraphrase_instruction(
        _instruction_draft(), _chunk(), preserve_tokens=preserve,
    )

    assert out_a == out_l
    assert "sh:NodeShape" in out_a["prompt"]


def test_local_length_remediation_parity():
    """A below-floor prompt triggers a length-retry on both providers; the
    accepted long-prompt response is byte-identical."""
    short = json.dumps({
        "prompt": "short",
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    long_prompt = (
        "Recall the foundational shacl concept introduced for topic X in "
        "chapter one of the source material."
    )
    long = json.dumps({
        "prompt": long_prompt,
        "completion": (
            "Topic X anchors every later chapter; recall its formal "
            "definition before attempting application questions."
        ),
    })
    seq = (
        httpx.Response(200, json=_success_body(short)),
        httpx.Response(200, json=_success_body(long)),
    )
    agnostic = SynthesisProvider(
        provider="local", terse_prompts=True, client=_client_yielding(*seq),
    )
    leaf = LocalSynthesisProvider(client=_client_yielding(*seq))

    out_a = agnostic.paraphrase_instruction(_instruction_draft(), _chunk())
    out_l = leaf.paraphrase_instruction(_instruction_draft(), _chunk())

    assert out_a == out_l
    assert out_a["prompt"] == long_prompt


# ---------------------------------------------------------------------------
# TOGETHER parity (verbose prompts)
# ---------------------------------------------------------------------------


def test_together_instruction_parity_byte_identical():
    """SynthesisProvider(together, terse_prompts=False) ==
    TogetherSynthesisProvider on a well-formed response."""
    agnostic = SynthesisProvider(
        provider="together", terse_prompts=False,
        client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    leaf = TogetherSynthesisProvider(
        api_key="tg-test", client=_fixed_client(_INSTRUCTION_COMPLETION),
    )

    out_a = agnostic.paraphrase_instruction(_instruction_draft(), _chunk())
    out_l = leaf.paraphrase_instruction(_instruction_draft(), _chunk())

    assert out_a == out_l
    assert out_a["provider"] == "together"


def test_together_preference_parity_byte_identical():
    agnostic = SynthesisProvider(
        provider="together", terse_prompts=False,
        client=_fixed_client(_PREFERENCE_COMPLETION),
    )
    leaf = TogetherSynthesisProvider(
        api_key="tg-test", client=_fixed_client(_PREFERENCE_COMPLETION),
    )

    out_a = agnostic.paraphrase_preference(_preference_draft(), _chunk())
    out_l = leaf.paraphrase_preference(_preference_draft(), _chunk())

    assert out_a == out_l
    assert out_a["provider"] == "together"


def test_agnostic_recovers_fenced_where_strict_together_retries():
    """Lenient-parse improvement: a markdown-fenced response that strict
    -together cannot parse on the first attempt is RECOVERED by the agnostic
    provider's lenient extractor — it succeeds in ONE call.

    (strict-together would consume a retry on the fenced body via its
    ``_parse_json`` fence-strip; the point here is the agnostic provider
    recovers, not that it matches together's retry count.)"""
    inner = json.loads(_INSTRUCTION_COMPLETION)
    fenced = f"```json\n{json.dumps(inner)}\n```\n\nHope that helps!"
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(fenced))

    agnostic = SynthesisProvider(
        provider="together", terse_prompts=False, client=_make_client(handler),
    )
    out = agnostic.paraphrase_instruction(_instruction_draft(), _chunk())
    assert out["provider"] == "together"
    assert "Recall the central idea" in out["prompt"]
    # Recovered in a single round-trip — no retry burned.
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Provider tag == constructor provider (local / together / nvidia)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["local", "together", "nvidia"])
def test_out_provider_tag_matches_constructor(provider):
    agnostic = SynthesisProvider(
        provider=provider,
        terse_prompts=(provider == "local"),
        client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    out = agnostic.paraphrase_instruction(_instruction_draft(), _chunk())
    assert out["provider"] == provider


# ---------------------------------------------------------------------------
# Constructor resolves the client BY NAME from the registry
# ---------------------------------------------------------------------------


def test_constructor_resolves_local_base_url_by_name():
    p = SynthesisProvider(
        provider="local", client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    assert p.base_url == "http://localhost:11434/v1"
    assert p.api_url == "http://localhost:11434/v1/chat/completions"


def test_constructor_resolves_together_base_url_by_name():
    p = SynthesisProvider(
        provider="together", terse_prompts=False,
        client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    # base_url flows from the `together` registry row, NOT a kwarg.
    assert p.base_url == "https://api.together.xyz/v1"


def test_constructor_resolves_nvidia_base_url_by_name():
    p = SynthesisProvider(
        provider="nvidia", client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    assert p.base_url == "https://integrate.api.nvidia.com/v1"


def test_request_hits_resolved_endpoint_url():
    """The request is POSTed to the registry-resolved endpoint, proving the
    transport flows from the named row (not the network)."""
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_INSTRUCTION_COMPLETION))

    p = SynthesisProvider(provider="together", terse_prompts=False,
                          client=_make_client(handler))
    p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert str(seen[0].url) == "https://api.together.xyz/v1/chat/completions"


# ---------------------------------------------------------------------------
# Preserve gate is a no-op when preserve_tokens is empty (hosted parity)
# ---------------------------------------------------------------------------


def test_preserve_gate_noop_when_no_tokens():
    """No preserve_tokens + default floor 0.0 → the gate never fires; a single
    well-formed response is accepted in one round-trip (hosted parity)."""
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_INSTRUCTION_COMPLETION))

    p = SynthesisProvider(provider="together", terse_prompts=False,
                          client=_make_client(handler))
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert out["provider"] == "together"
    assert len(seen) == 1  # no preserve retry


def test_default_floor_accepts_dropped_tokens_without_retry():
    """Default min_preserve_rate (0.0) accepts a paraphrase that drops every
    preserve_token — no retry, no exception (force-injection owns anchoring)."""
    paraphrase = json.dumps({
        "prompt": "Explain how the node shape concept applies to topic X.",
        "completion": (
            "The node-shape idea constrains node-typed entities; a learner "
            "applies it by validating instances against the declared "
            "constraint and a per-property type rule."
        ),
    })
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(paraphrase))

    p = SynthesisProvider(provider="local", client=_make_client(handler))
    out = p.paraphrase_instruction(
        _instruction_draft(), _chunk(),
        preserve_tokens=["sh:NodeShape", "sh:datatype"],
    )
    assert "sh:NodeShape" not in out["prompt"]
    assert len(seen) == 1  # accepted first try


# ---------------------------------------------------------------------------
# Decision capture fires + provider validation
# ---------------------------------------------------------------------------


def test_decision_capture_fires_once_with_provider_name():
    captured: List[dict] = []

    class _Capture:
        def log_decision(self, **kwargs):
            captured.append(kwargs)

    p = SynthesisProvider(
        provider="together", terse_prompts=False, capture=_Capture(),
        client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert len(captured) == 1
    event = captured[0]
    assert event["decision_type"] == "synthesis_provider_call"
    assert len(event["rationale"]) >= 20
    assert "chunk_id=chunk_001" in event["rationale"]
    assert "'together'" in event["rationale"]


def test_min_preserve_rate_validates_range():
    with pytest.raises(ValueError, match="min_preserve_rate"):
        SynthesisProvider(provider="local", min_preserve_rate=1.5,
                          client=_fixed_client(_INSTRUCTION_COMPLETION))


def test_default_provider_is_local():
    assert DEFAULT_PROVIDER == "local"
    p = SynthesisProvider(client=_fixed_client(_INSTRUCTION_COMPLETION))
    out = p.paraphrase_instruction(_instruction_draft(), _chunk())
    assert out["provider"] == "local"


def test_surface_form_preservation_failed_after_retry_exhaustion():
    """Strict floor (1.0) with every response dropping the token raises the
    distinguished ``surface_form_preservation_failed`` code (parity with the
    local leaf's exhaustion contract)."""
    bad = json.dumps({
        "prompt": "Explain how the node shape applies to topic X.",
        "completion": (
            "The node-shape concept constrains node-typed entities; a "
            "learner applies it by validating instances against the declared "
            "constraint and the per-property type rule."
        ),
    })
    p = SynthesisProvider(
        provider="local", min_preserve_rate=1.0, max_parse_retries=3,
        client=_client_yielding(
            *[httpx.Response(200, json=_success_body(bad)) for _ in range(3)]
        ),
    )
    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(
            _instruction_draft(), _chunk(), preserve_tokens=["sh:NodeShape"],
        )
    assert excinfo.value.code == "surface_form_preservation_failed"


# ---------------------------------------------------------------------------
# Bug #1 — together NEVER injects a preserve directive (preserve_tokens_enabled
# is the leaf-parity switch). With it False, preserve_tokens are IGNORED so the
# rendered request is byte-identical to the 2-arg TogetherSynthesisProvider.
# ---------------------------------------------------------------------------


def _user_message(request: httpx.Request) -> str:
    return json.loads(request.content)["messages"][-1]["content"]


def test_together_preserve_disabled_ignores_tokens_byte_identical_to_2arg_leaf():
    preserve = ["sh:NodeShape", "sh:datatype"]
    seen_a: List[httpx.Request] = []
    seen_l: List[httpx.Request] = []

    def handler_a(request: httpx.Request) -> httpx.Response:
        seen_a.append(request)
        return httpx.Response(200, json=_success_body(_INSTRUCTION_COMPLETION))

    def handler_l(request: httpx.Request) -> httpx.Response:
        seen_l.append(request)
        return httpx.Response(200, json=_success_body(_INSTRUCTION_COMPLETION))

    agnostic = SynthesisProvider(
        provider="together", terse_prompts=False, preserve_tokens_enabled=False,
        client=_make_client(handler_a),
    )
    leaf = TogetherSynthesisProvider(
        api_key="tg-test", client=_make_client(handler_l),
    )

    # Pass preserve_tokens to the agnostic provider; the 2-arg leaf can't take
    # them. With preserve disabled the agnostic provider drops them entirely.
    out_a = agnostic.paraphrase_instruction(
        _instruction_draft(), _chunk(), preserve_tokens=preserve,
    )
    out_l = leaf.paraphrase_instruction(_instruction_draft(), _chunk())

    user_a = _user_message(seen_a[0])
    user_l = _user_message(seen_l[0])
    # Bug #1: NO "MUST appear verbatim" preserve directive injected on the
    # together seat (the 2-arg leaf never received one) — and the leaf, which
    # cannot take preserve_tokens, never renders one either.
    assert "PRESERVE THESE TOKENS VERBATIM" not in user_a
    assert "PRESERVE THESE TOKENS VERBATIM" not in user_l
    # The CURIE surface forms do not leak into the together request at all.
    for tok in preserve:
        assert tok not in user_a
    # Emitted pair is byte-identical to the 2-arg leaf's.
    assert out_a == out_l


def test_local_preserve_enabled_does_inject_directive():
    """The local seat keeps preserve_tokens_enabled=True (default): the
    directive IS rendered when preserve_tokens are passed."""
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_INSTRUCTION_COMPLETION))

    p = SynthesisProvider(
        provider="local", terse_prompts=True, client=_make_client(handler),
    )
    p.paraphrase_instruction(
        _instruction_draft(), _chunk(), preserve_tokens=["sh:NodeShape"],
    )
    assert "PRESERVE THESE TOKENS VERBATIM" in _user_message(seen[0])


# ---------------------------------------------------------------------------
# Bug #2 — the constructor does NOT auto-read TRAINFORGE_SYNTHESIS_MODEL; the
# resolved model is the per-provider registry default (model_env), matching
# the leaves byte-for-byte.
# ---------------------------------------------------------------------------


def test_together_ignores_trainforge_synthesis_model_override(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_MODEL", "some/override-model-99b")
    p = SynthesisProvider(
        provider="together", terse_prompts=False, preserve_tokens_enabled=False,
        client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    # The TRAINFORGE_SYNTHESIS_MODEL override is NOT honored — the together
    # registry default (TOGETHER_SYNTHESIS_MODEL / row default) wins.
    assert p._model == TOGETHER_DEFAULT_MODEL
    assert p._model != "some/override-model-99b"


def test_together_honors_its_own_model_env(monkeypatch):
    """Sanity: the per-provider model_env DOES resolve (proving only the
    TRAINFORGE_SYNTHESIS_MODEL auto-read was removed, not all overrides)."""
    monkeypatch.setenv("TOGETHER_SYNTHESIS_MODEL", "Qwen/Qwen2.5-72B-Instruct-Turbo")
    p = SynthesisProvider(
        provider="together", terse_prompts=False, preserve_tokens_enabled=False,
        client=_fixed_client(_INSTRUCTION_COMPLETION),
    )
    assert p._model == "Qwen/Qwen2.5-72B-Instruct-Turbo"


# ---------------------------------------------------------------------------
# Bug #3 — the cutover builder pins the leaf's hard 60s timeout even when
# ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS is set (often 300 in production).
# ---------------------------------------------------------------------------


def test_builder_pins_60s_timeout_over_env(monkeypatch):
    monkeypatch.setenv("ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS", "300")
    # local needs no key + makes no HTTP call at construction.
    p = build_synthesis_provider("local")
    assert p._oa_client._timeout == 60.0


# ---------------------------------------------------------------------------
# #5 — the shared cutover builder constructs the leaf-exact provider.
# ---------------------------------------------------------------------------


def test_build_synthesis_provider_local_mapping():
    p = build_synthesis_provider("local")
    assert p._terse_prompts is True
    assert p._preserve_tokens_enabled is True
    assert p._oa_client._timeout == 60.0
    assert p._provider_name == "local"


def test_build_synthesis_provider_together_mapping(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tg-test")
    p = build_synthesis_provider("together")
    assert p._terse_prompts is False
    assert p._preserve_tokens_enabled is False
    assert p._oa_client._timeout == 60.0
    assert p._provider_name == "together"


def test_build_synthesis_provider_threads_local_knobs():
    """The local-path knobs (max_parse_retries / min_preserve_rate /
    kind_bounds) flow through the builder verbatim."""
    bounds = {
        "prompt": (10, 4000), "completion": (10, 4000),
        "chosen": (10, 4000), "rejected": (10, 4000),
    }
    p = build_synthesis_provider(
        "local", max_parse_retries=2, min_preserve_rate=1.0, kind_bounds=bounds,
    )
    assert p._max_parse_retries == 2
    assert p._min_preserve_rate == 1.0
    assert p._kind_bounds == bounds


def test_build_synthesis_provider_local_sets_local_user_directives():
    p = build_synthesis_provider("local")
    assert p._local_user_directives is True


def test_build_synthesis_provider_together_clears_local_user_directives(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tg-test")
    p = build_synthesis_provider("together")
    assert p._local_user_directives is False


# ---------------------------------------------------------------------------
# P3 FULL-PROMPT PARITY — capture the ACTUAL rendered messages the provider
# sends (system + user) via the injected MockTransport and assert BYTE-EQUALITY
# with the leaf renderer for EVERY pair kind, INCLUDING a definition-kind chunk
# (the case that exposed the local-only ``definition_directive`` leak onto the
# hosted/together path). The output-dict parity tests above are mock-insensitive
# and could not see this divergence.
# ---------------------------------------------------------------------------


def _system_message(request: httpx.Request) -> str:
    return json.loads(request.content)["messages"][0]["content"]


def _definition_instruction_draft() -> dict:
    """A definition-kind instruction draft (content_type=definition AND
    bloom_level=remember) — the local renderer injects the Wave-126
    definition_directive on it; the lean together renderer must NOT."""
    d = _instruction_draft()
    d["content_type"] = "definition"
    d["bloom_level"] = "remember"
    d["template_id"] = "remember.definition"
    return d


def _agnostic_together(handler):
    return SynthesisProvider(
        provider="together", terse_prompts=False,
        preserve_tokens_enabled=False, local_user_directives=False,
        client=_make_client(handler),
    )


def _together_leaf(handler):
    return TogetherSynthesisProvider(
        api_key="tg-test", client=_make_client(handler),
    )


def _agnostic_local(handler):
    return SynthesisProvider(
        provider="local", terse_prompts=True,
        preserve_tokens_enabled=True, local_user_directives=True,
        client=_make_client(handler),
    )


def _local_leaf(handler):
    return LocalSynthesisProvider(client=_make_client(handler))


def _run_and_capture(provider_factory, method_name, draft, chunk, **kwargs):
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        content = (
            _PREFERENCE_COMPLETION
            if method_name == "paraphrase_preference"
            else _INSTRUCTION_COMPLETION
        )
        return httpx.Response(200, json=_success_body(content))

    provider = provider_factory(handler)
    getattr(provider, method_name)(draft, chunk, **kwargs)
    return _system_message(seen[0]), _user_message(seen[0])


# --- together full-prompt parity (lean renderer; NO definition / preserve) ---


@pytest.mark.parametrize("method_name", ["paraphrase_instruction", "paraphrase_preference"])
def test_together_full_prompt_parity_byte_identical(method_name):
    draft = (
        _preference_draft() if method_name == "paraphrase_preference"
        else _instruction_draft()
    )
    sys_a, user_a = _run_and_capture(_agnostic_together, method_name, draft, _chunk())
    sys_l, user_l = _run_and_capture(_together_leaf, method_name, draft, _chunk())

    assert sys_a == sys_l
    assert user_a == user_l


def test_together_definition_kind_full_prompt_parity_byte_identical():
    """THE bug case: a definition-kind chunk. The lean together renderer never
    sent a definition_directive; the agnostic together path must match it
    byte-for-byte (system + user), and the directive text must be absent."""
    draft = _definition_instruction_draft()
    sys_a, user_a = _run_and_capture(
        _agnostic_together, "paraphrase_instruction", draft, _chunk(),
    )
    sys_l, user_l = _run_and_capture(
        _together_leaf, "paraphrase_instruction", draft, _chunk(),
    )

    assert sys_a == sys_l
    assert user_a == user_l
    # No local-only directives leaked onto the hosted path.
    assert "definition / recall chunk" not in user_a
    assert "PRESERVE THESE TOKENS VERBATIM" not in user_a


def test_together_definition_kind_with_preserve_tokens_still_lean():
    """Even when preserve_tokens are passed, the together path stays byte-
    identical to the 2-arg leaf (no preserve directive, no definition
    directive) on a definition-kind chunk."""
    draft = _definition_instruction_draft()
    sys_a, user_a = _run_and_capture(
        _agnostic_together, "paraphrase_instruction", draft, _chunk(),
        preserve_tokens=["sh:NodeShape", "sh:datatype"],
    )
    sys_l, user_l = _run_and_capture(
        _together_leaf, "paraphrase_instruction", draft, _chunk(),
    )
    assert sys_a == sys_l
    assert user_a == user_l
    assert "definition / recall chunk" not in user_a
    assert "PRESERVE THESE TOKENS VERBATIM" not in user_a
    for tok in ("sh:NodeShape", "sh:datatype"):
        assert tok not in user_a


# --- local full-prompt parity (superset renderer; definition + preserve) ---


@pytest.mark.parametrize("method_name", ["paraphrase_instruction", "paraphrase_preference"])
def test_local_full_prompt_parity_byte_identical(method_name):
    draft = (
        _preference_draft() if method_name == "paraphrase_preference"
        else _instruction_draft()
    )
    sys_a, user_a = _run_and_capture(_agnostic_local, method_name, draft, _chunk())
    sys_l, user_l = _run_and_capture(_local_leaf, method_name, draft, _chunk())

    assert sys_a == sys_l
    assert user_a == user_l


def test_local_definition_kind_full_prompt_parity_with_directive():
    """The local superset renderer DOES inject the definition_directive on a
    definition-kind chunk; the agnostic local path matches byte-for-byte AND
    the directive text is present."""
    draft = _definition_instruction_draft()
    sys_a, user_a = _run_and_capture(
        _agnostic_local, "paraphrase_instruction", draft, _chunk(),
    )
    sys_l, user_l = _run_and_capture(
        _local_leaf, "paraphrase_instruction", draft, _chunk(),
    )

    assert sys_a == sys_l
    assert user_a == user_l
    # The local-leaf definition directive IS present where the leaf has it.
    assert "definition / recall chunk" in user_a


def test_local_definition_kind_with_preserve_full_prompt_parity():
    """Both local-only user directives (definition + preserve) present and
    byte-identical to the local leaf on a definition-kind chunk + tokens."""
    draft = _definition_instruction_draft()
    preserve = ["sh:NodeShape", "sh:datatype"]
    sys_a, user_a = _run_and_capture(
        _agnostic_local, "paraphrase_instruction", draft, _chunk(),
        preserve_tokens=preserve,
    )
    sys_l, user_l = _run_and_capture(
        _local_leaf, "paraphrase_instruction", draft, _chunk(),
        preserve_tokens=preserve,
    )

    assert sys_a == sys_l
    assert user_a == user_l
    assert "definition / recall chunk" in user_a
    assert "PRESERVE THESE TOKENS VERBATIM" in user_a
