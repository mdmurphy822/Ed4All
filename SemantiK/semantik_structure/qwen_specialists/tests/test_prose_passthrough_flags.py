"""Unit tests for SEMANTIK_STAGE6_PROSE_PASSTHROUGH (Stage-6 wall-time win).

CPU-pinned, no GPU, no local GGUF, no live endpoint. The runtime is a
generate_batch-observing MockRuntime so the tests assert WHICH regions were
sent to the model without ever loading a GGUF.

Contract under test:
  * flag OFF (default / malformed) -> prose regions GENERATE (byte-stable):
    the runtime IS invoked for prose and the candidate is a normal draft.
  * flag ON  -> AdapterID.PROSE regions receive a deterministic verbatim
    passthrough candidate (k=1) and the runtime is NOT invoked for them,
    while TABLE + MATH still generate.

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest \
    semantik_structure/qwen_specialists/tests/test_prose_passthrough_flags.py -q
"""

from __future__ import annotations

import types

import pytest

from semantik_structure.qwen_specialists.runner import (
    resolve_stage6_prose_passthrough,
    run_qwen_specialists,
)
from semantik_structure.qwen_specialists.runtime import MockRuntime
from semantik_structure.qwen_specialists.types import AdapterID


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _region(kind: str, text: str = "x"):
    """Minimal duck-typed Stage-5 Region the prompt builders + emit_fallback
    accept (empty FB indices -> emit_fallback falls back to payload text)."""
    return types.SimpleNamespace(
        kind=kind,
        payload={"text": text},
        aria_hints=(),
        feature_block_indices=(),
    )


class _GenSpyRuntime(MockRuntime):
    """MockRuntime that records every generate_batch prompt it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_prompts: list[str] = []

    def generate_batch(self, prompts, *, max_tokens, **kw):  # type: ignore[override]
        self.batch_prompts.extend(prompts)
        return super().generate_batch(prompts, max_tokens=max_tokens, **kw)


def _clear_env(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", raising=False)
    monkeypatch.delenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", raising=False)


# ---------------------------------------------------------------------------
# Resolver: parse-with-fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
def test_resolver_truthy(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", val)
    assert resolve_stage6_prose_passthrough() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "garbage", "  "])
def test_resolver_falsey_and_malformed(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", val)
    assert resolve_stage6_prose_passthrough() is False


def test_resolver_unset(monkeypatch):
    monkeypatch.delenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", raising=False)
    assert resolve_stage6_prose_passthrough() is False


# ---------------------------------------------------------------------------
# Flag OFF — prose GENERATES (byte-stable default)
# ---------------------------------------------------------------------------


def test_flag_off_prose_is_generated(monkeypatch):
    """Default: prose region goes through generate_batch; candidate is a
    normal local draft (NOT a passthrough)."""
    _clear_env(monkeypatch)

    regions = [_region("paragraph", "hello world")]
    rt = _GenSpyRuntime()
    out = run_qwen_specialists(regions, [], k=2, runtime=rt)

    # The prose prompt reached the model.
    assert any("paragraph" in p for p in rt.batch_prompts)
    # Candidate exists and is a generated draft, not a passthrough.
    assert set(out.keys()) == {0}
    assert len(out[0]) == 2  # k=2 drafts
    for cand in out[0]:
        assert cand.raw_metadata.get("stage6_phase") == "local"
        assert cand.raw_metadata.get("prose_passthrough") is not True


def test_malformed_env_generates_prose(monkeypatch):
    """A malformed env value is OFF -> prose still generates."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", "banana")

    regions = [_region("paragraph", "abc")]
    rt = _GenSpyRuntime()
    out = run_qwen_specialists(regions, [], k=1, runtime=rt)

    assert any("paragraph" in p for p in rt.batch_prompts)
    assert out[0][0].raw_metadata.get("stage6_phase") == "local"


# ---------------------------------------------------------------------------
# Flag ON — prose passthrough, table/math still generate
# ---------------------------------------------------------------------------


def test_flag_on_prose_passthrough_no_generation(monkeypatch):
    """Prose region: k=1 deterministic verbatim candidate, runtime NOT
    invoked for it. TABLE + MATH still generate."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", "1")

    regions = [
        _region("paragraph", "verbatim prose text"),
        _region("table", "tbl"),
        _region("math", "eqn"),
    ]
    rt = _GenSpyRuntime()
    out = run_qwen_specialists(regions, [], k=2, runtime=rt)

    # Every region produced candidates.
    assert set(out.keys()) == {0, 1, 2}

    # Prose (idx 0): exactly ONE deterministic passthrough candidate.
    prose = out[0]
    assert len(prose) == 1
    cand = prose[0]
    assert cand.adapter is AdapterID.PROSE
    assert cand.finish_reason == "stop"
    assert cand.raw_metadata.get("prose_passthrough") is True
    assert cand.raw_metadata.get("stage6_phase") == "passthrough"
    assert cand.raw_metadata.get("adapter_version") == "passthrough"
    # Verbatim source text preserved, wrapped as a well-formed <p>.
    assert cand.text == "<p>verbatim prose text</p>"

    # The prose prompt NEVER reached the model.
    assert not any("verbatim prose text" in p for p in rt.batch_prompts)
    assert not any('"kind": "paragraph"' in p for p in rt.batch_prompts)

    # TABLE + MATH DID reach the model (still generate) and got k=2 drafts.
    assert any("table" in p for p in rt.batch_prompts)
    assert any("math" in p for p in rt.batch_prompts)
    assert len(out[1]) == 2
    assert len(out[2]) == 2
    for cand in out[1] + out[2]:
        assert cand.raw_metadata.get("stage6_phase") == "local"


def test_flag_on_only_prose_kinds_passthrough(monkeypatch):
    """All AdapterID.PROSE kinds passthrough with kind-appropriate shape;
    NONE of them reach the model."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", "on")

    regions = [
        _region("paragraph", "para"),
        _region("heading", "My Heading"),
        _region("code_block", "print(1)"),
        _region("blockquote", "quoted"),
    ]
    rt = _GenSpyRuntime()
    out = run_qwen_specialists(regions, [], k=3, runtime=rt)

    # No generation happened at all (every region is prose).
    assert rt.batch_prompts == []

    # Kind-appropriate deterministic shapes (== emit_fallback), each k=1.
    assert out[0][0].text == "<p>para</p>"
    assert out[1][0].text == "<h2>My Heading</h2>"
    assert out[2][0].text == "<pre><code>print(1)</code></pre>"
    assert out[3][0].text == "<blockquote>quoted</blockquote>"
    for idx in out:
        assert len(out[idx]) == 1
        assert out[idx][0].raw_metadata.get("prose_passthrough") is True


def test_flag_on_all_prose_no_generation_calls(monkeypatch):
    """A document that is entirely prose fires ZERO generate_batch calls."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", "true")

    regions = [_region("paragraph", f"p{i}") for i in range(5)]
    rt = _GenSpyRuntime()
    out = run_qwen_specialists(regions, [], k=4, runtime=rt)

    assert rt.batch_prompts == []
    assert set(out.keys()) == {0, 1, 2, 3, 4}
    for idx in out:
        assert len(out[idx]) == 1  # k=1 passthrough regardless of requested k
