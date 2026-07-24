"""GLM-OCR SDK page-dispatch tests — ROLLING page queue.

Pins the conversion-bottleneck fix in
``glmocr/sdk_client.py::SdkGlmOcrClient.parse_pages``: the static N-way
partition (split the page list into ``n`` equal contiguous chunks, one chunk
per worker) is replaced by a ROLLING per-page queue (``ex.map`` over the flat
page list) so a worker that finishes cheap pages immediately pulls the next
page instead of idling while the worker holding the expensive chunk grinds on
alone. Contracts pinned here:

  1. Page-order reassembly — output is sorted by ``page_no`` regardless of
     completion order.
  2. Parser thread-safety / reuse — ONE GlmOcr parser per WORKER (constructed
     once, reused across the pages that worker pulls); construction count ==
     worker count, NOT page count (the model is never rebuilt per page).
  3. VRAM/width — the worker count is ``min(workers, n_pages)`` (unchanged);
     exactly that many parsers (= resident models) are built.
  4. Error handling — an SDK per-page ``_error`` becomes ``GlmPage.error``
     exactly as the chunk path did.
  5. Determinism — the rolling output is byte-identical to a serial reference
     for a fixed fake input (only the dispatch schedule changes).

Fully hermetic: a FAKE GlmOcr parser with controllable per-page
latency/output/error and a per-thread construction counter — no real model, no
GPU, no seat, no network. The OP2 usage meter is stubbed to a no-op so the
dispatch is measured in isolation (its row shape is covered by
``test_glmocr_usage_metering.py``).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

_SEMANTIK_ROOT = Path(__file__).resolve().parents[2]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))

from semantik_structure import llm_usage_meter  # noqa: E402
from semantik_structure.glmocr import sdk_client as sdk_mod  # noqa: E402
from semantik_structure.glmocr.transform import GlmPage  # noqa: E402


@pytest.fixture(autouse=True)
def _no_metering(monkeypatch):
    """Isolate the dispatch from the best-effort disk metering tap."""
    monkeypatch.setattr(llm_usage_meter, "record_provider_usage", lambda **_: None)


class _FakeResult:
    """Mirrors the SDK PipelineResult surface the client reads."""

    def __init__(self, regions, error=None):
        self.json_result = regions
        self._error = error


class _FakeParser:
    """A controllable GlmOcr stand-in.

    ``parse`` is driven by the shared ``harness`` so a test can inject
    per-page latency / a blocking gate / an ``_error`` and observe live
    concurrency + which parser handled which page (parser reuse)."""

    def __init__(self, harness):
        self.harness = harness
        self.pages_seen: list[int] = []
        harness._on_construct(self)

    def parse(self, paths, save_layout_visualization=False):
        results = []
        for pstr in paths:
            page = self.harness._page_of(pstr)
            self.pages_seen.append(page)
            results.append(self.harness._run_page(page, self))
        return results

    def close(self):
        self.harness._on_close(self)


class _Harness:
    """Owns the fake-parser fleet + the page-behaviour injection."""

    def __init__(self, n_pages, *, latency=None, errors=None,
                 slow_page=None, slow_after=None, barrier_width=None):
        self.n_pages = n_pages
        self.paths = [Path(f"/tmp/pg/page-{i}.png") for i in range(n_pages)]
        self._by_path = {str(p): i for i, p in enumerate(self.paths)}
        self._latency = latency or {}
        self._errors = errors or {}

        # Construction / reuse tracking.
        self.parsers: list[_FakeParser] = []
        self.closed: list[_FakeParser] = []
        self._construct_lock = threading.Lock()

        # Live-concurrency tracking.
        self._active = 0
        self.max_active = 0
        self._active_lock = threading.Lock()

        # Ragged-cost gate: ``slow_page`` blocks until ``slow_after`` OTHER
        # pages have completed (deterministic proof the free worker keeps
        # pulling — a static partition would strand or deadlock here).
        self._slow_page = slow_page
        self._slow_after = slow_after
        self._gate = threading.Event()
        self._completed = 0
        self._completed_lock = threading.Lock()
        self.slow_gate_released = None

        # Optional barrier to force exactly ``barrier_width`` live threads
        # (so construction count == worker count is deterministic).
        self._barrier = (
            threading.Barrier(barrier_width) if barrier_width else None
        )
        self._tls = threading.local()

    def _page_of(self, pstr):
        return self._by_path[str(pstr)]

    def _on_construct(self, parser):
        with self._construct_lock:
            self.parsers.append(parser)

    def _on_close(self, parser):
        self.closed.append(parser)

    def _run_page(self, page, parser):
        # Force n live threads once per worker if a barrier is configured.
        if self._barrier is not None and not getattr(self._tls, "waited", False):
            self._tls.waited = True
            self._barrier.wait(timeout=10)

        with self._active_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if page == self._slow_page:
                self.slow_gate_released = self._gate.wait(timeout=10)
            else:
                lat = self._latency.get(page, 0.0)
                if lat:
                    time.sleep(lat)
                if self._slow_page is not None:
                    with self._completed_lock:
                        self._completed += 1
                        if (self._slow_after is not None
                                and self._completed >= self._slow_after):
                            self._gate.set()
            regions = [{"index": 0, "native_label": "text",
                        "bbox_2d": [0, 0, 5, 5], "content": f"p{page}"}]
            return _FakeResult(regions, error=self._errors.get(page))
        finally:
            with self._active_lock:
                self._active -= 1

    def client(self, workers):
        c = sdk_mod.SdkGlmOcrClient(
            base_url="http://localhost:8002/v1", model="glm-ocr",
            workers=workers,
        )
        c._make_parser = lambda: _FakeParser(self)  # type: ignore[method-assign]
        return c


def _serial_reference(harness):
    """Parse every page in order on a SINGLE parser — the determinism oracle."""
    parser = _FakeParser(harness)
    out = []
    for i, path in enumerate(harness.paths):
        res = parser.parse([str(path)])[0]
        out.append(GlmPage(
            page_no=i + 1,
            regions=sdk_mod._normalise_regions(res.json_result),
            image_path=str(path),
            error=str(res._error) if res._error else None,
        ))
    parser.close()
    return out


def _key(pages):
    return [(p.page_no, p.image_path, p.error, tuple(
        (r["index"], r["native_label"], tuple(r["bbox_2d"]), r["content"])
        for r in p.regions)) for p in pages]


# ── 1. page-order reassembly ─────────────────────────────────────────────────


def test_output_is_page_ordered_regardless_of_completion_order():
    # Reverse latency: high page indices finish FIRST, so completion order is
    # the reverse of page order — the output must still come back page-ordered.
    n = 12
    latency = {i: (n - i) * 0.004 for i in range(n)}
    h = _Harness(n, latency=latency)
    pages = h.client(4).parse_pages(h.paths)
    assert [p.page_no for p in pages] == list(range(1, n + 1))
    assert [p.image_path for p in pages] == [str(p) for p in h.paths]


# ── 2/3. one parser per WORKER, not per page (construction == worker count) ───


def test_one_parser_per_worker_not_per_page():
    n_pages, workers = 9, 3
    # The barrier forces all 3 workers live simultaneously → 3 constructions.
    h = _Harness(n_pages, barrier_width=workers)
    pages = h.client(workers).parse_pages(h.paths)

    assert len(pages) == n_pages
    # Exactly one parser per worker — NOT one per page.
    assert len(h.parsers) == workers
    assert len(h.parsers) < n_pages
    # Every page accounted for, and parsers were REUSED across pages.
    seen = sorted(pg for prs in h.parsers for pg in prs.pages_seen)
    assert seen == list(range(n_pages))
    assert max(len(prs.pages_seen) for prs in h.parsers) > 1  # reuse, not per-page
    # All resident models released (the close bookend the chunk path had).
    assert len(h.closed) == workers


def test_worker_count_capped_at_page_count():
    # n = min(workers, n_pages): fewer pages than workers → fewer parsers.
    h = _Harness(2, barrier_width=2)
    pages = h.client(8).parse_pages(h.paths)
    assert [p.page_no for p in pages] == [1, 2]
    assert len(h.parsers) == 2  # capped at n_pages, not workers=8


# ── rolling saturation: a slow page does NOT drain concurrency to 1 ───────────


def test_rolling_keeps_free_worker_pulling_under_ragged_cost():
    # n=2 workers, 10 pages. Page 0 blocks until 8 OTHER pages complete. Under
    # the OLD static partition worker-1 owns only pages[5:10] (5 pages) and
    # would STRAND at 5 completed < 8 → page 0 never releases (drain/deadlock).
    # Under ROLLING the free worker keeps pulling the whole queue, so 8 finish,
    # page 0 releases, and the run completes — with the work heavily skewed to
    # the non-blocked worker (proof it kept pulling past its static 1/n share).
    n_pages, workers = 10, 2
    h = _Harness(n_pages, slow_page=0, slow_after=n_pages - workers)
    pages = h.client(workers).parse_pages(h.paths)

    assert h.slow_gate_released is True  # released, not timed out (no deadlock)
    assert [p.page_no for p in pages] == list(range(1, n_pages + 1))
    assert len(h.parsers) == workers
    counts = sorted(len(prs.pages_seen) for prs in h.parsers)
    # The free worker did far MORE than its static n_pages/workers = 5 share.
    assert counts[-1] > n_pages // workers
    assert counts[0] < n_pages // workers  # the blocked worker did far fewer
    # Concurrency stayed at the full width while the slow page ran.
    assert h.max_active == workers


# ── 4. per-page OCR failure handled as today (SDK _error → GlmPage.error) ─────


def test_per_page_error_becomes_glmpage_error():
    h = _Harness(6, errors={3: "ocr exploded on this page"})
    pages = h.client(3).parse_pages(h.paths)
    assert [p.page_no for p in pages] == [1, 2, 3, 4, 5, 6]
    errored = [p for p in pages if p.error]
    assert len(errored) == 1
    assert errored[0].page_no == 4  # page index 3 → page_no 4
    assert errored[0].error == "ocr exploded on this page"
    assert all(p.error is None for p in pages if p.page_no != 4)


# ── 5. byte-identical vs a serial reference ──────────────────────────────────


def test_rolling_is_byte_identical_to_serial_reference():
    n = 11
    latency = {i: (i % 3) * 0.003 for i in range(n)}  # ragged completion order
    rolling = _Harness(n, latency=latency)
    serial = _Harness(n, latency=latency)
    got = rolling.client(4).parse_pages(rolling.paths)
    ref = _serial_reference(serial)
    assert _key(got) == _key(ref)


# ── single-worker + empty input keep the chunk path byte-stable ──────────────


def test_single_worker_uses_chunk_path():
    h = _Harness(3)
    pages = h.client(1).parse_pages(h.paths)
    assert [p.page_no for p in pages] == [1, 2, 3]
    assert len(h.parsers) == 1  # one chunk, one parser
    assert len(h.closed) == 1


def test_empty_input_returns_empty():
    h = _Harness(0)
    assert h.client(4).parse_pages([]) == []
    assert h.parsers == []
