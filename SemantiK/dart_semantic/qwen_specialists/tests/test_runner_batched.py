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
    _RegionJob,
    _pack_batches,
    resolve_batch_mode,
    resolve_batched_endpoint_k,
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


# ---------------------------------------------------------------------------
# Phase-2 per-region fail-soft degradation (the document-abort fix)
# ---------------------------------------------------------------------------


from dart_semantic.qwen_specialists.endpoint_runtime import EndpointRuntimeError


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

    def generate_multi(self, items, *, max_tokens, drafts=None, **kw):  # type: ignore[override]
        ids = [rid for rid, _ in items]
        self.multi_calls.append(ids)
        if self.raise_batch_with is not None and self.raise_batch_with in ids:
            raise RuntimeError("simulated batch dispatch error")
        out = {}
        for rid, _payload in items:
            out[rid] = None if rid in self.fail_ids else f"<p>endpoint {rid}</p>"
        return out


def test_phase2_batched_one_post_per_batch_default_on(monkeypatch):
    """Default (batch ON): regions are dispatched via generate_multi, one
    POST per batch — fewer calls than regions×K."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
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
    from dart_semantic.qwen_specialists.endpoint_runtime import EndpointRuntimeError

    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
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
