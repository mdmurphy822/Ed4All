"""Unit tests for the BATCHED two-phase Stage-6 driver.

CPU-pinned, no GPU, no local GGUF, no live endpoint. The runtime is a
swap-count-observing MockRuntime; the endpoint concurrency test MOCKS
``requests.post``.

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest semantik_structure/qwen_specialists/tests/test_runner_batched.py -q
"""

from __future__ import annotations

import threading
import time
import types
from unittest import mock

import pytest

from semantik_structure.qwen_specialists.endpoint_runtime import (
    EndpointBatchItemError,
    OpenAICompatibleRuntime,
    resolve_specialist_concurrency,
)
from semantik_structure.qwen_specialists.runner import (
    _RegionJob,
    _pack_batches,
    resolve_batch_mode,
    resolve_batched_endpoint_k,
    resolve_refine_mode,
    run_qwen_specialists,
)
from semantik_structure.qwen_specialists import runtime as runtime_mod
from semantik_structure.qwen_specialists.runtime import (
    LlamaCppRuntime,
    MockRuntime,
    any_phase_provider_is_endpoint,
    make_phase_runtime,
    phase_provider_is_endpoint,
    resolve_phase_model,
    resolve_phase_provider,
)
from semantik_structure.qwen_specialists.types import AdapterID


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
    from semantik_structure.qwen_specialists import runner as runner_mod

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


def test_provider_endpoint_alone_runs_local_specialists(monkeypatch):
    """DERELICTION FIX: provider=nvidia ALONE (no refine, no displace) now
    AUTHORS via the local specialists (Phase 1), NOT the endpoint.

    Selecting an endpoint provider no longer silently displaces the
    license-clean on-device specialists — the corrected default fires them."""
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", raising=False)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")

    rt = _BatchSpyRuntime()
    out = run_qwen_specialists(
        [_region("math", "a"), _region("table", "b")], [], k=2, runtime=rt
    )
    # Phase 1 (local, batched-by-adapter) ran and produced LOCAL candidates;
    # the endpoint was never the authoring tier.
    assert rt.batch_calls, "local Phase 1 should have fired"
    assert out[0][0].raw_metadata["stage6_phase"] == "local"
    assert len(out[0]) == 2 and len(out[1]) == 2


def test_provider_endpoint_displace_skips_phase1_runs_phase2(monkeypatch):
    """provider=nvidia + DISPLACE -> Phase 1 SKIPPED, Phase 2 (pure-endpoint).

    The explicit opt-in restores the pre-fix pure-endpoint behaviour."""
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")

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


def test_endpoint_displace_resolver_parse_with_fallback(monkeypatch):
    from semantik_structure.qwen_specialists.runtime import resolve_endpoint_displaces_local

    monkeypatch.delenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", raising=False)
    assert resolve_endpoint_displaces_local() is False
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "garbage")
    assert resolve_endpoint_displaces_local() is False
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "on")
    assert resolve_endpoint_displaces_local() is True


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


# ---------------------------------------------------------------------------
# Phase-2 per-region fail-soft degradation (the document-abort fix)
# ---------------------------------------------------------------------------


from semantik_structure.qwen_specialists.endpoint_runtime import EndpointRuntimeError


class _FailSoftEndpointSpy(MockRuntime):
    """Endpoint-shaped runtime that fails specific prompt indices.

    When ``fail_soft`` is passed (the runner Phase-2 path), failed indices
    return the ``None`` sentinel; successful indices echo a fragment. When
    ``fail_soft`` is False it RAISES on the first failing index (legacy
    fail-loud), mirroring the real endpoint runtime."""

    def __init__(self, fail_indices):
        super().__init__()
        self.fail_indices = set(fail_indices)
        self.batch_calls = 0

    def generate_batch(self, prompts, *, max_tokens, fail_soft=False, **kw):  # type: ignore[override]
        self.batch_calls += 1
        out = []
        for i, _p in enumerate(prompts):
            if i in self.fail_indices:
                if fail_soft:
                    out.append(None)
                else:
                    raise EndpointRuntimeError(f"item {i} failed", transient=True)
            else:
                out.append(f"<p>endpoint {i}</p>")
        return out


def test_phase2_hybrid_one_region_fails_keeps_local_draft(monkeypatch):
    """Scenario (a): one region times out in HYBRID (refine) mode -> the
    conversion CONTINUES, that region keeps its Phase-1 LOCAL draft, the
    document assembles."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_REFINE", "1")

    regions = [_region("math", "a"), _region("paragraph", "b")]
    # active order is PROSE-first then MATH (PASS_ORDER): paragraph=a_pos 0,
    # math=a_pos 1. Fail a_pos 1 (the math region) on every slot.
    rt = _FailSoftEndpointSpy(fail_indices={1})
    out = run_qwen_specialists(regions, [], k=2, runtime=rt)

    # Both regions produced candidates (document assembles, no abort).
    assert set(out.keys()) == {0, 1}
    for idx in out:
        assert len(out[idx]) == 2

    # The math region (region index 0) kept its LOCAL draft and is stamped
    # degraded. (region idx 0 is the math region in the input list.)
    math_cands = out[0]
    assert all(c.raw_metadata.get("endpoint_degraded") for c in math_cands)
    assert all(
        c.raw_metadata.get("degraded_reason")
        == "endpoint_refine_failed_kept_local_draft"
        for c in math_cands
    )
    # The draft text is real local content (a MathML fragment from the mock),
    # NOT empty / a skip.
    assert all(c.text and c.text.strip() for c in math_cands)
    assert all(c.raw_metadata["stage6_phase"] == "refine" for c in math_cands)

    # The healthy paragraph region got real endpoint output, NOT degraded.
    para_cands = out[1]
    assert all(not c.raw_metadata.get("endpoint_degraded") for c in para_cands)
    assert all("endpoint" in c.text for c in para_cands)


def test_phase2_pure_endpoint_one_region_fails_degraded_skip(monkeypatch):
    """Scenario (b): one region fails in PURE-ENDPOINT mode -> degraded/skip
    candidate for THAT region, the others are fine."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    # Pure-endpoint path now requires the explicit displace opt-in (the local
    # specialists author by default after the dereliction fix).
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")

    regions = [_region("math", "a"), _region("paragraph", "b")]
    # active: PROSE (paragraph) a_pos 0, MATH a_pos 1. Fail a_pos 1 (math).
    rt = _FailSoftEndpointSpy(fail_indices={1})
    out = run_qwen_specialists(regions, [], k=2, runtime=rt)

    assert set(out.keys()) == {0, 1}
    # Math region (region idx 0) -> single degraded skip candidate.
    math_cands = out[0]
    assert len(math_cands) == 1
    assert math_cands[0].text == ""
    assert math_cands[0].raw_metadata.get("endpoint_degraded") is True
    assert math_cands[0].raw_metadata.get("degraded_reason") == "endpoint_failed_all_slots"
    assert math_cands[0].finish_reason == "endpoint_degraded"

    # Paragraph region (idx 1) is fine — k=2 real endpoint candidates.
    para_cands = out[1]
    assert len(para_cands) == 2
    assert all("endpoint" in c.text for c in para_cands)


def test_phase2_pure_endpoint_all_regions_fail_raises(monkeypatch):
    """Scenario (c): ALL active regions fail the endpoint -> still RAISES
    (fail-loud). We do not ship an empty document on a total failure."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    # Pure-endpoint path now requires the explicit displace opt-in (the local
    # specialists author by default after the dereliction fix).
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")

    regions = [_region("math", "a"), _region("paragraph", "b")]
    rt = _FailSoftEndpointSpy(fail_indices={0, 1})  # every active region fails
    with pytest.raises(EndpointRuntimeError) as exc:
        run_qwen_specialists(regions, [], k=2, runtime=rt)
    assert "ALL" in str(exc.value) and "empty document" in str(exc.value)


def test_phase2_pure_endpoint_partial_slot_failure_keeps_good_slots(monkeypatch):
    """A region with SOME slots failing but ≥1 succeeding keeps the good
    candidates (no degraded skip), proving partial-slot resilience."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    # Pure-endpoint path now requires the explicit displace opt-in (the local
    # specialists author by default after the dereliction fix).
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")

    # Fail the region only on EVEN slots: a stateful spy.
    class _AltSlotSpy(MockRuntime):
        def __init__(self):
            super().__init__()
            self.slot = -1

        def generate_batch(self, prompts, *, max_tokens, fail_soft=False, **kw):  # type: ignore[override]
            self.slot += 1
            fail = self.slot % 2 == 0  # fail slot 0, succeed slot 1
            out = []
            for i in range(len(prompts)):
                out.append(None if (fail and fail_soft) else f"<p>e{i}</p>")
            return out

    rt = _AltSlotSpy()
    out = run_qwen_specialists([_region("math", "a")], [], k=2, runtime=rt)
    # k=2: slot0 failed (None), slot1 succeeded -> exactly 1 kept candidate.
    assert len(out[0]) == 1
    assert out[0][0].text == "<p>e0</p>"


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


# ---------------------------------------------------------------------------
# Stage B — _pack_batches packing policy
# ---------------------------------------------------------------------------


def _job(idx: int, *, adapter: AdapterID, max_new_tokens: int) -> _RegionJob:
    return _RegionJob(
        idx=idx,
        adapter=adapter,
        request=types.SimpleNamespace(
            adapter=adapter, request_id=f"r{idx}", sampling_overrides={}
        ),
        defaults={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.0,
        },
        prompt=f"SYSTEM: r\nUSER: region-{idx}",
    )


def test_pack_batches_respects_max_regions():
    jobs = [_job(i, adapter=AdapterID.PROSE, max_new_tokens=100) for i in range(7)]
    batches = _pack_batches(jobs, max_regions=3, output_token_cap=10_000_000)
    # 7 jobs / 3 per batch -> [3, 3, 1]; order preserved.
    assert [len(b) for b in batches] == [3, 3, 1]
    flat = [j.idx for b in batches for j in b]
    assert flat == list(range(7))


def test_pack_batches_output_token_cap_forces_smaller_table_batches():
    # Big-output table jobs (4000 tokens each) with a 10000 cap -> 2 per batch
    # even though max_regions allows 12.
    jobs = [_job(i, adapter=AdapterID.TABLE, max_new_tokens=4000) for i in range(5)]
    batches = _pack_batches(jobs, max_regions=12, output_token_cap=10_000)
    # 4000+4000=8000 <= 10000, +4000=12000 > 10000 -> new batch. [2,2,1].
    assert [len(b) for b in batches] == [2, 2, 1]


def test_pack_batches_oversized_single_job_gets_own_batch():
    jobs = [_job(0, adapter=AdapterID.TABLE, max_new_tokens=99_999)]
    batches = _pack_batches(jobs, max_regions=12, output_token_cap=1000)
    assert len(batches) == 1 and len(batches[0]) == 1  # never dropped


def test_pack_batches_197_prose_jobs_collapse_to_about_17_batches():
    jobs = [_job(i, adapter=AdapterID.PROSE, max_new_tokens=200) for i in range(197)]
    batches = _pack_batches(jobs, max_regions=12, output_token_cap=10_000_000)
    # 197 / 12 = 16.4 -> 17 batches (the ~197 POSTs -> ~17 POSTs claim).
    assert len(batches) == 17


def test_batch_mode_resolver_default_on_falsey_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH", raising=False)
    assert resolve_batch_mode() is True  # default ON for endpoint lane
    for falsey in ("0", "false", "no", "off"):
        monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH", falsey)
        assert resolve_batch_mode() is False, falsey
    for truthy in ("1", "true", "yes", "garbage", ""):
        monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH", truthy)
        assert resolve_batch_mode() is True, truthy


def test_batched_endpoint_k_caps_at_2(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH_K", raising=False)
    assert resolve_batched_endpoint_k(4) == 2  # prose K=4 -> capped to 2
    assert resolve_batched_endpoint_k(1) == 1  # never raised above caller's k
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH_K", "3")
    assert resolve_batched_endpoint_k(4) == 3  # explicit pin honoured
    assert resolve_batched_endpoint_k(2) == 2  # still clamped to caller's k


# ---------------------------------------------------------------------------
# Stage C/D — Phase-2 BATCHED dispatch via generate_multi
# ---------------------------------------------------------------------------


class _BatchedEndpointSpy(MockRuntime):
    """Endpoint-shaped runtime exposing generate_multi (batched path).

    Records each generate_multi call's region ids. ``fail_ids`` (a set of
    ``r{idx}`` strings) come back as the None sentinel; others echo a
    fragment. ``raise_batch_with`` (a region id) makes the WHOLE batch
    containing that id raise, to exercise the per-batch degrade-to-None."""

    def __init__(self, fail_ids=None, raise_batch_with=None):
        super().__init__()
        self.fail_ids = set(fail_ids or [])
        self.raise_batch_with = raise_batch_with
        self.multi_calls: list[list[str]] = []
        self.batch_fallback_calls = 0

    def generate_multi(self, items, *, max_tokens, drafts=None, **kw):  # type: ignore[override]
        ids = [rid for rid, _ in items]
        self.multi_calls.append(ids)
        if self.raise_batch_with is not None and self.raise_batch_with in ids:
            raise RuntimeError("simulated batch dispatch error")
        out = {}
        for rid, _payload in items:
            out[rid] = None if rid in self.fail_ids else f"<p>endpoint {rid}</p>"
        return out

    def generate_batch(self, prompts, *, max_tokens, fail_soft=False, **kw):  # type: ignore[override]
        # The #41 envelope-miss FALLBACK lane. The fallback only ever hands this
        # the regions that already FAILED the batched (generate_multi) lane, so
        # this spy models a seat whose per-region lane ALSO fails for them —
        # BOTH lanes fail, so a degraded region STAYS degraded (the fallback
        # rescues only when the per-region GENERATE succeeds; that case is the
        # dedicated envelope-miss regression fixture below).
        self.batch_fallback_calls += 1
        return [None if fail_soft else "" for _ in prompts]


def test_phase2_batched_one_post_per_batch_default_on(monkeypatch):
    """Default (batch ON): regions are dispatched via generate_multi, one
    POST per batch — fewer calls than regions×K."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    # Pure-endpoint path now requires the explicit displace opt-in (the local
    # specialists author by default after the dereliction fix).
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH", raising=False)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH_REGIONS", "12")

    regions = [_region("paragraph", str(i)) for i in range(5)]
    rt = _BatchedEndpointSpy()
    out = run_qwen_specialists(regions, [], k=4, runtime=rt)

    # All 5 regions produced candidates via the batched path.
    assert set(out.keys()) == {0, 1, 2, 3, 4}
    # K capped to 2 on the batched lane -> 2 slots; 5 prose regions fit in ONE
    # batch each slot -> 2 generate_multi calls total (NOT 5×4 = 20 POSTs).
    assert len(rt.multi_calls) == 2
    assert all(c.raw_metadata["stage6_phase"] == "endpoint" for c in out[0])
    assert all("endpoint" in c.text for c in out[0])


def test_phase2_batched_failsoft_one_region_degrades_others_assemble(monkeypatch):
    """One region's batch fails (None) -> that region degrades, the rest
    assemble. ALL regions failing still fails loud."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    # Pure-endpoint path now requires the explicit displace opt-in (the local
    # specialists author by default after the dereliction fix).
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH", raising=False)

    regions = [_region("paragraph", "a"), _region("paragraph", "b")]
    # Region index 1 (id r1) fails every slot.
    rt = _BatchedEndpointSpy(fail_ids={"r1"})
    out = run_qwen_specialists(regions, [], k=4, runtime=rt)

    assert set(out.keys()) == {0, 1}
    # r0 assembled normally.
    assert all("endpoint" in c.text for c in out[0])
    # r1 degraded to a single skip candidate (pure-endpoint, no local draft).
    assert len(out[1]) == 1
    assert out[1][0].text == ""
    assert out[1][0].raw_metadata.get("endpoint_degraded") is True
    assert out[1][0].raw_metadata.get("degraded_reason") == "endpoint_failed_all_slots"


def test_phase2_batched_all_regions_fail_raises(monkeypatch):
    from semantik_structure.qwen_specialists.endpoint_runtime import EndpointRuntimeError

    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    # Pure-endpoint path now requires the explicit displace opt-in (the local
    # specialists author by default after the dereliction fix).
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH", raising=False)

    regions = [_region("paragraph", "a"), _region("paragraph", "b")]
    rt = _BatchedEndpointSpy(fail_ids={"r0", "r1"})
    with pytest.raises(EndpointRuntimeError) as exc:
        run_qwen_specialists(regions, [], k=4, runtime=rt)
    assert "ALL" in str(exc.value) and "empty document" in str(exc.value)


def test_phase2_batched_batch_dispatch_raise_degrades_that_batch(monkeypatch):
    """A generate_multi call that RAISES degrades its batch's regions to None
    (defensive) rather than aborting — the other batch still assembles."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    # Pure-endpoint path now requires the explicit displace opt-in (the local
    # specialists author by default after the dereliction fix).
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH", raising=False)
    # Force one region per batch so the raising region is isolated.
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH_REGIONS", "1")

    regions = [_region("paragraph", "a"), _region("paragraph", "b")]
    rt = _BatchedEndpointSpy(raise_batch_with="r1")
    out = run_qwen_specialists(regions, [], k=4, runtime=rt)
    # r0 fine; r1's batch raised -> degraded skip, document still assembles.
    assert "endpoint" in out[0][0].text
    assert out[1][0].raw_metadata.get("endpoint_degraded") is True


def test_phase2_byte_stable_when_off(monkeypatch):
    """SEMANTIK_SPECIALIST_BATCH=0 -> EXACT per-region generate_batch path.

    The off path uses generate_batch (not generate_multi), keeps full K, and
    produces the identical Candidate shape the legacy path emits."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    # Pure-endpoint path now requires the explicit displace opt-in (the local
    # specialists author by default after the dereliction fix).
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH", "0")

    # A runtime exposing BOTH generate_batch (used when off) and generate_multi
    # (must NOT be called when off).
    class _DualSpy(MockRuntime):
        def __init__(self):
            super().__init__()
            self.batch_calls = 0
            self.multi_calls = 0

        def generate_batch(self, prompts, *, max_tokens, fail_soft=False, **kw):  # type: ignore[override]
            self.batch_calls += 1
            return [f"<p>endpoint {i}</p>" for i in range(len(prompts))]

        def generate_multi(self, items, *, max_tokens, drafts=None, **kw):  # type: ignore[override]
            self.multi_calls += 1
            return {rid: f"<p>multi {rid}</p>" for rid, _ in items}

    regions = [_region("paragraph", "a"), _region("math", "b")]
    rt = _DualSpy()
    out = run_qwen_specialists(regions, [], k=4, runtime=rt)

    # Off path took generate_batch, NEVER generate_multi.
    assert rt.batch_calls > 0
    assert rt.multi_calls == 0
    # Full K=4 preserved (no batched-lane K cap when off).
    assert len(out[0]) == 4 and len(out[1]) == 4
    # Legacy Candidate shape: phase tagged "endpoint", real content.
    assert all(c.raw_metadata["stage6_phase"] == "endpoint" for c in out[0])
    assert all("endpoint" in c.text for c in out[0])


# ---------------------------------------------------------------------------
# Per-phase seat resolution — model-agnostic two-phase Stage 6
# ---------------------------------------------------------------------------
#
# Phase 1 (draft) and Phase 2 (refine/generate) each resolve their OWN
# {provider, model} seat. Phase 1 defaults to local (the dereliction default,
# NOT inheriting the single seat); Phase 2 falls back to the single seat. The
# hosted 70B is just one possible Phase-2 model, never a hardcoded tier.


def _clear_phase_env(monkeypatch):
    for k in (
        "SEMANTIK_SPECIALIST_PROVIDER",
        "SEMANTIK_SPECIALIST_MODEL",
        "SEMANTIK_SPECIALIST_PHASE1_PROVIDER",
        "SEMANTIK_SPECIALIST_PHASE1_MODEL",
        "SEMANTIK_SPECIALIST_PHASE2_PROVIDER",
        "SEMANTIK_SPECIALIST_PHASE2_MODEL",
        "SEMANTIK_SPECIALIST_REFINE",
        "SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE",
        "NVIDIA_LARGE_MODEL",
    ):
        monkeypatch.delenv(k, raising=False)


def test_phase_provider_defaults_local(monkeypatch):
    _clear_phase_env(monkeypatch)
    assert resolve_phase_provider("phase1") == "local"
    assert resolve_phase_provider("phase2") == "local"
    assert phase_provider_is_endpoint("phase1") is False
    assert any_phase_provider_is_endpoint() is False


def test_phase1_does_not_inherit_single_seat(monkeypatch):
    """DERELICTION default: the single seat selects Phase 2, NOT Phase 1.

    provider=nvidia alone leaves Phase 1 LOCAL; Phase 2 inherits it."""
    _clear_phase_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    assert resolve_phase_provider("phase1") == "local"
    assert resolve_phase_provider("phase2") == "nvidia"
    assert any_phase_provider_is_endpoint() is True


def test_phase_provider_per_phase_overrides(monkeypatch):
    """Each phase's PROVIDER env wins; PHASE2 overrides the single seat."""
    _clear_phase_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PHASE1_PROVIDER", "local-openai")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PHASE2_PROVIDER", "local")
    assert resolve_phase_provider("phase1") == "local-openai"
    # Explicit PHASE2=local overrides the single nvidia seat (local refine).
    assert resolve_phase_provider("phase2") == "local"
    # Phase 1 is the only endpoint here.
    assert phase_provider_is_endpoint("phase1") is True
    assert phase_provider_is_endpoint("phase2") is False
    assert any_phase_provider_is_endpoint() is True


def test_phase_model_fallback_chain(monkeypatch):
    _clear_phase_env(monkeypatch)
    # No envs -> the literal default.
    assert resolve_phase_model("phase2") == "meta/llama-3.3-70b-instruct"
    # Single seat model is the shared fallback for both phases.
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MODEL", "shared-model")
    assert resolve_phase_model("phase1") == "shared-model"
    assert resolve_phase_model("phase2") == "shared-model"
    # Per-phase model wins.
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PHASE2_MODEL", "qwen2.5:32b")
    assert resolve_phase_model("phase2") == "qwen2.5:32b"
    assert resolve_phase_model("phase1") == "shared-model"


def test_make_phase_runtime_mock_is_mockruntime(monkeypatch):
    _clear_phase_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PHASE2_PROVIDER", "nvidia")
    # mock mode never reads the provider -> deterministic MockRuntime.
    rt = make_phase_runtime("mock", "phase2")
    assert isinstance(rt, MockRuntime)


def test_make_phase_runtime_local_seat_is_local_runtime_not_endpoint(monkeypatch):
    """A LOCAL Phase-2 seat builds the local GGUF runtime — NO endpoint.

    The key model-agnostic case: local-draft -> local-large-refine makes zero
    hosted calls. Proven by: the runtime is LlamaCppRuntime (the local arm) and
    OpenAICompatibleRuntime is NEVER constructed."""
    _clear_phase_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PHASE2_PROVIDER", "local")

    # Report a present adapter so the local arm returns a runtime rather than
    # the strict "no adapters on disk" raise (hermetic across boxes).
    from pathlib import Path

    monkeypatch.setattr(
        runtime_mod,
        "_scan_adapter_artifacts",
        lambda cfg=None: ({"prose": Path("x.gguf")}, {"prose": True}),
    )
    # Detonate if the endpoint runtime is ever constructed on this path.
    import semantik_structure.qwen_specialists.endpoint_runtime as ep_mod

    def _boom(*a, **k):  # pragma: no cover - asserted not-called
        raise AssertionError("OpenAICompatibleRuntime must NOT be built for a local seat")

    monkeypatch.setattr(ep_mod, "OpenAICompatibleRuntime", _boom)

    rt = make_phase_runtime("real", "phase2")
    assert isinstance(rt, LlamaCppRuntime)


def test_make_phase_runtime_endpoint_seat_resolves_model(monkeypatch):
    """A non-local Phase-2 seat builds the OpenAI-compatible runtime pinned to
    the per-phase model (base_url/api_key from the shared env)."""
    _clear_phase_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PHASE2_PROVIDER", "nvidia")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PHASE2_MODEL", "phase2-model")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BASE_URL", "https://e.example/v1")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_API_KEY", "UNIT-KEY")

    rt = make_phase_runtime("real", "phase2")
    assert isinstance(rt, OpenAICompatibleRuntime)
    assert rt._model == "phase2-model"  # per-phase model resolved
    assert rt._base_url == "https://e.example/v1"


def test_runner_builds_separate_runtimes_per_phase_local_refine(monkeypatch):
    """local-draft -> local-refine: Phase 1 and Phase 2 use SEPARATE runtimes
    (the latent single-runtime bug fix), the refine prompt carrying the draft
    goes to the Phase-2 runtime, and NO hosted endpoint is ever constructed.

    This exercises the runner wiring with runtime=None so per-phase
    construction fires; make_phase_runtime is patched to hand back a distinct
    recording spy per phase (both LOCAL — the model-agnostic on-device case)."""
    _clear_phase_env(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_REFINE", "1")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PHASE2_PROVIDER", "local")  # local refine

    class _PhaseSpy(MockRuntime):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag
            self.batch_prompts: list[str] = []

        def generate_batch(self, prompts, *, max_tokens, **kw):  # type: ignore[override]
            self.batch_prompts.extend(prompts)
            return super().generate_batch(prompts, max_tokens=max_tokens, **kw)

    built: dict[str, _PhaseSpy] = {}
    calls: list[tuple] = []

    from semantik_structure.qwen_specialists import runner as runner_mod

    def _fake_make(mode, phase, *, config_path=None):
        calls.append((mode, phase))
        spy = _PhaseSpy(phase)
        built[phase] = spy
        return spy

    monkeypatch.setattr(runner_mod, "make_phase_runtime", _fake_make)

    # Detonate if the real endpoint runtime is constructed anywhere.
    import semantik_structure.qwen_specialists.endpoint_runtime as ep_mod

    def _boom(*a, **k):  # pragma: no cover - asserted not-called
        raise AssertionError("no hosted endpoint must be built for local refine")

    monkeypatch.setattr(ep_mod, "OpenAICompatibleRuntime", _boom)

    out = run_qwen_specialists(
        [_region("math", "a")], [], k=1, runtime=None, runtime_mode="mock"
    )

    # Both phases built their OWN runtime (not one shared instance).
    assert ("mock", "phase1") in calls and ("mock", "phase2") in calls
    assert built["phase1"] is not built["phase2"]
    # Phase 1 drafted with NO refine directive.
    assert built["phase1"].batch_prompts
    assert all("DRAFT_FRAGMENT" not in p for p in built["phase1"].batch_prompts)
    # Phase 2 refined on its OWN (local) runtime — the draft travelled to it.
    assert built["phase2"].batch_prompts
    assert any("DRAFT_FRAGMENT" in p for p in built["phase2"].batch_prompts)
    # Output is the refined candidate, phase tagged "refine".
    assert out[0][0].raw_metadata["stage6_phase"] == "refine"


# ---------------------------------------------------------------------------
# #41 — pure-endpoint DISPLACE: batched envelope MISS (all-None) is recovered
# by the per-region GENERATE fallback (no draft precondition, no llama_cpp).
# ---------------------------------------------------------------------------


class _EnvelopeMissSpy(MockRuntime):
    """Endpoint-shaped runtime whose generate_multi ALWAYS returns all-None
    (the nemotron-seat envelope non-compliance) but whose per-region
    generate_batch SUCCEEDS — so the #41 fallback recovers every region."""

    def __init__(self):
        super().__init__()
        self.multi_calls = 0
        self.fallback_batch_calls = 0

    def generate_multi(self, items, *, max_tokens, drafts=None, **kw):  # type: ignore[override]
        self.multi_calls += 1
        # Envelope parse recovered ZERO regions on a "200" — all None.
        return {rid: None for rid, _ in items}

    def generate_batch(self, prompts, *, max_tokens, fail_soft=False, **kw):  # type: ignore[override]
        self.fallback_batch_calls += 1
        # The per-region GENERATE lane works (no envelope, no draft needed).
        return [f"<p>recovered {i}</p>" for i in range(len(prompts))]


def test_phase2_displace_envelope_miss_recovered_by_per_region_fallback(
    monkeypatch, tmp_path, caplog
):
    import logging
    import sys

    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH", raising=False)  # batched ON
    # Drain prose so the table/math jobs carry SPARSE (gap-bearing) region ids.
    monkeypatch.setenv("SEMANTIK_STAGE6_PROSE_PASSTHROUGH", "1")

    # Detonate if the local GGUF backend is ever imported on the endpoint path.
    class _NoLlamaCpp:
        def find_spec(self, name, path=None, target=None):
            if name == "llama_cpp" or name.startswith("llama_cpp."):
                raise AssertionError("llama_cpp must NOT import on the endpoint path")
            return None

    guard = _NoLlamaCpp()
    sys.meta_path.insert(0, guard)
    try:
        # prose at 0,2,4 (drained → passthrough); table at 1,3; math at 5.
        regions = [
            _region("paragraph", "p0"),
            _region("table", "t1"),
            _region("paragraph", "p2"),
            _region("table", "t3"),
            _region("paragraph", "p4"),
            _region("math", "m5"),
        ]
        rt = _EnvelopeMissSpy()
        with caplog.at_level(logging.WARNING):
            out = run_qwen_specialists(regions, [], k=2, runtime=rt)
    finally:
        sys.meta_path.remove(guard)

    # generate_multi fired (batched lane) AND the fallback fired (envelope miss).
    assert rt.multi_calls >= 1
    assert rt.fallback_batch_calls >= 1

    # Every region produced a REAL candidate — the table/math regions were
    # recovered by the per-region fallback (NOT degraded), the prose regions
    # passthrough'd.
    assert set(out.keys()) == {0, 1, 2, 3, 4, 5}
    for idx in (1, 3, 5):  # table + math recovered
        cands = out[idx]
        assert cands and all("recovered" in c.text for c in cands)
        assert all(not c.raw_metadata.get("endpoint_degraded") for c in cands)
    # No degraded-skip anywhere, and the misleading '(no local draft)' text is gone.
    joined = " ".join(r.message for r in caplog.records)
    assert "no local draft" not in joined
    assert "endpoint_failed_all_slots" not in joined


def test_phase2_displace_both_lanes_fail_still_fail_loud(monkeypatch):
    """Total endpoint failure — BOTH the batched envelope AND the per-region
    fallback fail → the fail-loud total-failure guard is preserved (#41 must
    not weaken it)."""
    from semantik_structure.qwen_specialists.endpoint_runtime import EndpointRuntimeError

    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE", "1")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH", raising=False)

    class _DeadBothLanes(MockRuntime):
        def generate_multi(self, items, *, max_tokens, drafts=None, **kw):  # type: ignore[override]
            return {rid: None for rid, _ in items}

        def generate_batch(self, prompts, *, max_tokens, fail_soft=False, **kw):  # type: ignore[override]
            return [None for _ in prompts]  # per-region lane dead too

    regions = [_region("table", "a"), _region("math", "b")]
    rt = _DeadBothLanes()
    with pytest.raises(EndpointRuntimeError) as exc:
        run_qwen_specialists(regions, [], k=2, runtime=rt)
    assert "ALL" in str(exc.value) and "empty document" in str(exc.value)
