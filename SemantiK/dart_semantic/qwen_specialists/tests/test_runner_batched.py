"""Unit tests for the BATCHED two-phase Stage-6 driver.

CPU-pinned, no GPU, no local GGUF, no live endpoint. The runtime is a
swap-count-observing MockRuntime; the endpoint concurrency test MOCKS
``requests.post``.

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest dart_semantic/qwen_specialists/tests/test_runner_batched.py -q
"""

from __future__ import annotations

import threading
import time
import types
from unittest import mock

import pytest

from dart_semantic.qwen_specialists.endpoint_runtime import (
    EndpointBatchItemError,
    OpenAICompatibleRuntime,
    resolve_specialist_concurrency,
)
from dart_semantic.qwen_specialists.runner import (
    resolve_refine_mode,
    run_qwen_specialists,
)
from dart_semantic.qwen_specialists.runtime import MockRuntime
from dart_semantic.qwen_specialists.types import AdapterID


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _region(kind: str, text: str = "x"):
    """Minimal duck-typed Stage-5 Region the prompt builders accept."""
    return types.SimpleNamespace(
        kind=kind,
        payload={"text": text},
        aria_hints=(),
        feature_block_indices=(),
    )


class _BatchSpyRuntime(MockRuntime):
    """MockRuntime that also records generate_batch calls + load order.

    ``load_calls`` (from MockRuntime) records each adapter GGUF load. With
    every adapter at ``adapter_path: null`` in config.yaml, AdapterSwap
    SKIPS load(), so we instead count swap ENTRIES via ``swap_adapters``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[dict] = []

    def generate_batch(self, prompts, *, max_tokens, **kw):  # type: ignore[override]
        self.batch_calls.append(
            {"n_prompts": len(prompts), "prompts": list(prompts), "max_tokens": max_tokens}
        )
        return super().generate_batch(prompts, max_tokens=max_tokens, **kw)


# ---------------------------------------------------------------------------
# Phase 1 — batch BY ADAPTER, load each adapter ONCE
# ---------------------------------------------------------------------------


def test_phase1_groups_by_adapter_one_batch_per_adapter_per_slot(monkeypatch):
    """Three math + two table + one prose regions -> 3 distinct adapters.

    With k=2, generate_batch fires K times PER adapter group (one per
    candidate slot), NOT once per region. So the batch-call count is
    (#distinct adapters * k), and the prose group's single batch carries
    BOTH... no — prose has 1 region. The key proof: math's batch carries
    all 3 math prompts in ONE call (region count != call count)."""
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)

    regions = [
        _region("math", "a"),
        _region("paragraph", "b"),
        _region("math", "c"),
        _region("table", "d"),
        _region("math", "e"),
        _region("table", "f"),
    ]
    rt = _BatchSpyRuntime()
    out = run_qwen_specialists(regions, [], k=2, runtime=rt)

    # Every region produced k=2 candidates.
    assert set(out.keys()) == {0, 1, 2, 3, 4, 5}
    for idx, cands in out.items():
        assert len(cands) == 2, idx

    # 3 distinct adapters (PROSE, TABLE, MATH) x k=2 slots == 6 batch calls,
    # NOT 6 regions x 2 = 12. The math group's batch carried 3 prompts in
    # one call.
    assert len(rt.batch_calls) == 3 * 2

    # Find a math batch call (3 prompts) — proves region count != call count.
    math_calls = [c for c in rt.batch_calls if c["n_prompts"] == 3]
    assert len(math_calls) == 2  # k=2 slots, each batching all 3 math regions
    table_calls = [c for c in rt.batch_calls if c["n_prompts"] == 2]
    assert len(table_calls) == 2  # 2 table regions, k=2 slots
    prose_calls = [c for c in rt.batch_calls if c["n_prompts"] == 1]
    assert len(prose_calls) == 2  # 1 prose region, k=2 slots


def test_phase1_adapter_loaded_once_via_swap_count(monkeypatch):
    """AdapterSwap enters ONCE per distinct adapter, not once per region.

    Patch AdapterSwap to count __enter__ calls so the swap-thrash fix is
    asserted directly (adapter loads == distinct adapters)."""
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)

    enters: list[AdapterID] = []
    from dart_semantic.qwen_specialists import runner as runner_mod

    real_swap = runner_mod.AdapterSwap

    class _CountingSwap(real_swap):  # type: ignore[misc, valid-type]
        def __enter__(self):
            enters.append(self.adapter)
            return super().__enter__()

    monkeypatch.setattr(runner_mod, "AdapterSwap", _CountingSwap)

    regions = [
        _region("math", "a"),
        _region("math", "b"),
        _region("math", "c"),
        _region("paragraph", "d"),
        _region("paragraph", "e"),
    ]
    rt = _BatchSpyRuntime()
    run_qwen_specialists(regions, [], k=4, runtime=rt)

    # 2 distinct adapters touched (MATH, PROSE) — ONE swap each, despite
    # 3 math + 2 prose regions and k=4.
    assert enters == [AdapterID.PROSE, AdapterID.MATH]


def test_generate_batch_preserves_order():
    """MockRuntime.generate_batch returns one completion per prompt in order."""
    rt = MockRuntime()
    out = rt.generate_batch(
        ["kind=math one", "kind=table two", "kind=heading three"],
        max_tokens=64,
    )
    assert len(out) == 3
    assert "math" in out[0].lower() or "<math" in out[0]
    assert "<table" in out[1]
    assert "<h2" in out[2]


# ---------------------------------------------------------------------------
# Provider / REFINE mode routing
# ---------------------------------------------------------------------------


def test_provider_local_runs_phase1_only_no_endpoint(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "local")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    rt = _BatchSpyRuntime()
    out = run_qwen_specialists([_region("math", "a")], [], k=2, runtime=rt)
    # Phase 1 ran (batch calls present) and produced local candidates.
    assert rt.batch_calls
    assert out[0][0].raw_metadata["stage6_phase"] == "local"


def test_provider_endpoint_skips_phase1_runs_phase2(monkeypatch):
    """provider=nvidia, refine off -> Phase 1 SKIPPED, Phase 2 generates."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)

    # An endpoint-shaped spy: generate_batch echoes a per-prompt fragment.
    class _EndpointSpy(MockRuntime):
        def __init__(self):
            super().__init__()
            self.batch_prompts: list[list[str]] = []

        def generate_batch(self, prompts, *, max_tokens, **kw):  # type: ignore[override]
            self.batch_prompts.append(list(prompts))
            return [f"<p>endpoint {i}</p>" for i in range(len(prompts))]

    rt = _EndpointSpy()
    out = run_qwen_specialists([_region("math", "a"), _region("table", "b")], [], k=2, runtime=rt)
    # Phase 2 batched BOTH regions in one call per slot; refine off => no
    # DRAFT_FRAGMENT in the prompt.
    assert rt.batch_prompts, "endpoint generate_batch should have fired"
    assert all("DRAFT_FRAGMENT" not in p for batch in rt.batch_prompts for p in batch)
    # 2 regions x k=2 candidates, phase tagged "endpoint".
    assert out[0][0].raw_metadata["stage6_phase"] == "endpoint"
    assert len(out[0]) == 2 and len(out[1]) == 2


def test_refine_builds_prompt_plus_draft(monkeypatch):
    """provider=nvidia + REFINE=1 -> Phase 1 drafts feed Phase 2 prompts."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_REFINE", "1")

    captured: list[str] = []

    class _HybridSpy(MockRuntime):
        def generate_batch(self, prompts, *, max_tokens, **kw):  # type: ignore[override]
            captured.extend(prompts)
            return super().generate_batch(prompts, max_tokens=max_tokens, **kw)

    rt = _HybridSpy()
    out = run_qwen_specialists([_region("math", "a")], [], k=1, runtime=rt)

    # Phase 1 prompt(s) have NO draft; Phase 2 prompt(s) CARRY the draft +
    # refine directive.
    phase2 = [c for c in captured if "DRAFT_FRAGMENT" in c]
    assert phase2, "refine Phase 2 prompt must contain the local draft"
    assert "Improve and COMPLETE it" in phase2[0]
    # The Phase-1 draft fragment text is embedded in the refine prompt.
    assert "<math" in phase2[0]
    assert out[0][0].raw_metadata["stage6_phase"] == "refine"


def test_refine_default_off_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    assert resolve_refine_mode() is False
    monkeypatch.setenv("SEMANTIK_SPECIALIST_REFINE", "garbage")
    assert resolve_refine_mode() is False
    monkeypatch.setenv("SEMANTIK_SPECIALIST_REFINE", "yes")
    assert resolve_refine_mode() is True


# ---------------------------------------------------------------------------
# Endpoint generate_batch — concurrency + ordering + per-item failure
# ---------------------------------------------------------------------------


def _fake_response(content: str):
    r = mock.Mock()
    r.status_code = 200
    r.text = ""
    r.json.return_value = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return r


def test_endpoint_generate_batch_concurrent_and_ordered(monkeypatch):
    """N POSTs fire CONCURRENTLY (peak in-flight > 1) and re-order to inputs."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_CONCURRENCY", "4")
    rt = OpenAICompatibleRuntime(
        base_url="https://e.example/v1", api_key="UNIT-KEY", model="m", timeout=5.0
    )

    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def _fake_post(url, *, json, headers, timeout):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            time.sleep(0.05)  # hold so others pile up -> concurrency observable
            # Echo the user-turn content so we can verify ordering.
            user = json["messages"][1]["content"]
            return _fake_response(f"<p>{user}</p>")
        finally:
            with lock:
                in_flight -= 1

    prompts = [f"SYSTEM: r\nUSER: item-{i}" for i in range(4)]
    t0 = time.perf_counter()
    with mock.patch("requests.post", side_effect=_fake_post):
        out = rt.generate_batch(prompts, max_tokens=32)
    elapsed = time.perf_counter() - t0

    # Ordered: out[i] corresponds to prompts[i].
    assert out == [f"<p>item-{i}</p>" for i in range(4)]
    # Concurrency observed (>1 in-flight at once).
    assert peak > 1, f"expected concurrent posts, peak in-flight={peak}"
    # Wall-clock ~ one call (0.05s) not 4x serial (0.20s).
    assert elapsed < 0.15, f"expected concurrent wall-clock, got {elapsed:.3f}s"


def test_endpoint_generate_batch_one_item_failure_raises_indexed(monkeypatch):
    """A single failing POST raises EndpointBatchItemError with its index;
    the whole batch is not silently dropped/fabricated."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_CONCURRENCY", "4")
    rt = OpenAICompatibleRuntime(
        base_url="https://e.example/v1", api_key="UNIT-KEY", model="m", timeout=5.0
    )

    def _fake_post(url, *, json, headers, timeout):
        user = json["messages"][1]["content"]
        if user == "item-2":
            r = mock.Mock()
            r.status_code = 500
            r.text = "boom"
            return r
        return _fake_response(f"<p>{user}</p>")

    prompts = [f"SYSTEM: r\nUSER: item-{i}" for i in range(4)]
    with mock.patch("requests.post", side_effect=_fake_post):
        with pytest.raises(EndpointBatchItemError) as ei:
            rt.generate_batch(prompts, max_tokens=32)
    assert ei.value.index == 2


def test_concurrency_bound_resolver(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_CONCURRENCY", raising=False)
    assert resolve_specialist_concurrency() == 8
    monkeypatch.setenv("SEMANTIK_SPECIALIST_CONCURRENCY", "3")
    assert resolve_specialist_concurrency() == 3
    monkeypatch.setenv("SEMANTIK_SPECIALIST_CONCURRENCY", "0")
    assert resolve_specialist_concurrency() == 8
    monkeypatch.setenv("SEMANTIK_SPECIALIST_CONCURRENCY", "garbage")
    assert resolve_specialist_concurrency() == 8


def test_concurrency_bound_honored_capped_at_prompt_count(monkeypatch):
    """max_workers is min(concurrency, len(prompts)) — proven via peak."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_CONCURRENCY", "8")
    rt = OpenAICompatibleRuntime(
        base_url="https://e.example/v1", api_key="UNIT-KEY", model="m", timeout=5.0
    )
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def _fake_post(url, *, json, headers, timeout):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            time.sleep(0.03)
            return _fake_response("<p>ok</p>")
        finally:
            with lock:
                in_flight -= 1

    # Only 2 prompts -> peak in-flight capped at 2 even though bound is 8.
    prompts = ["SYSTEM: r\nUSER: a", "SYSTEM: r\nUSER: b"]
    with mock.patch("requests.post", side_effect=_fake_post):
        out = rt.generate_batch(prompts, max_tokens=32)
    assert len(out) == 2
    assert peak <= 2
