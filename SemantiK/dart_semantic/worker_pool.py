"""Process-pool helper for pair-generation pipelines.

Each worker process holds its own HtmlValidator (one Chromium instance).
Validators live for the worker's lifetime — they aren't torn down per task —
because launching Chromium costs ~1s and we're typically processing thousands
of items per run.

Usage from a pipeline script:

    from dart_semantic.worker_pool import run_in_pool

    # task_fn is a top-level function: def task_fn(validator, item) -> dict
    for title, stats in run_in_pool(task_fn, work_items, workers=4):
        aggregate(stats)

`task_fn` must be importable from a module (not a closure or local function),
because it's dispatched through pickle to each worker.
"""
from __future__ import annotations

import atexit
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator

_DEFAULT_WORKER_MAX_TASKS = 50


def resolve_worker_max_tasks() -> int:
    """Number of tasks a pool worker handles before it is recycled.

    Recycling each worker every N tasks tears down its native allocator arenas
    (the glibc malloc fragmentation that pypdfium2/pdfplumber per-page renders
    leave behind and never return to the OS), its Chromium child, and its
    per-process ``lru_cache`` state — bounding RSS growth on long builds.

    Resolved from ``SEMANTIK_WORKER_MAX_TASKS`` (parse-with-fallback: blank /
    non-int / non-positive / garbage → the default 50).
    """
    raw = os.environ.get("SEMANTIK_WORKER_MAX_TASKS", "")
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_WORKER_MAX_TASKS
    if val <= 0:
        return _DEFAULT_WORKER_MAX_TASKS
    return val

# Per-worker state. `_VALIDATOR` is populated once by `_initializer` in each
# child process and reused across every task that process handles.
_VALIDATOR = None


def _initializer() -> None:
    global _VALIDATOR
    # Import inside the initializer so the parent process doesn't need a
    # Chromium context just to dispatch work.
    from dart_semantic.validate import HtmlValidator
    _VALIDATOR = HtmlValidator()
    _VALIDATOR.__enter__()
    atexit.register(_teardown)


def _teardown() -> None:
    # Runs at worker-process exit — i.e. at every pool-generation boundary in
    # the batched run_in_pool below. A graceful Playwright stop here can hang
    # the worker's interpreter shutdown (the sync driver's non-daemon dispatcher
    # thread + node subprocess block the join), which in turn blocks the pool's
    # shutdown(wait=True). So HARD-exit instead — the worker's results are
    # already on the result queue. But a bare os._exit ORPHANS the worker's
    # Playwright node driver + Chromium children (reparented to init, leaking
    # ~3 procs/worker across generations), so kill the child process tree first.
    try:
        import psutil  # available in the SemantiK runtime venv

        me = psutil.Process()
        kids = me.children(recursive=True)
        for c in kids:
            try:
                c.kill()
            except Exception:
                pass
        psutil.wait_procs(kids, timeout=3)
    except Exception:
        pass
    os._exit(0)


def _task_entry(module_name: str, func_name: str, item: Any) -> Any:
    """Dispatched to each worker. Resolves the task function by name, calls it
    with the worker-local validator and the work item."""
    from importlib import import_module
    fn = getattr(import_module(module_name), func_name)
    return fn(_VALIDATOR, item)


def run_in_pool(task_fn: Callable[[Any, Any], Any],
                items: Iterable[Any],
                *,
                workers: int = 1) -> Iterator[Any]:
    """Run `task_fn(validator, item)` for each item in parallel workers.

    Yields each task's return value as soon as it completes (in arbitrary
    order — ordering isn't preserved).

    If workers == 1, runs in the main process with one HtmlValidator —
    cheaper than spinning up a subprocess for small batches.
    """
    items = list(items)
    if not items:
        return

    if workers <= 1:
        from dart_semantic.validate import HtmlValidator
        with HtmlValidator() as v:
            for item in items:
                yield task_fn(v, item)
        return

    mod_name = task_fn.__module__
    fn_name = task_fn.__name__
    # Recycle workers every N-tasks-per-worker by processing items in
    # generations: a FRESH ProcessPoolExecutor per generation, torn down before
    # the next. Tearing the pool down frees each worker's native glibc malloc
    # arenas (the tracemalloc-invisible RSS ratchet that per-page
    # pypdfium2/pdfplumber render+extract churn leaves fragmented), its Chromium
    # child, and its per-process lru_cache — the primary fix for the
    # build_structure_data --aligner global OOM.
    #
    # Why not ProcessPoolExecutor(max_tasks_per_child=...)? It is forbidden with
    # the default 'fork' start method, and on this Python build the 'forkserver'
    # / 'spawn' variants stall after the first generation of workers (replacement
    # workers never pick up the remaining tasks — reproduced with a no-op
    # initializer, so it is not Playwright-specific). Generational pools achieve
    # the same recycling on the proven-working default (fork) path.
    per_worker = resolve_worker_max_tasks()
    generation = max(1, per_worker * workers)  # tasks per pool generation
    for start in range(0, len(items), generation):
        chunk = items[start:start + generation]
        with ProcessPoolExecutor(max_workers=workers, initializer=_initializer) as ex:
            futures = [ex.submit(_task_entry, mod_name, fn_name, item) for item in chunk]
            for fut in as_completed(futures):
                yield fut.result()
        # Pool closed here → this generation's workers (+ arenas + Chromium +
        # lru_cache) are gone before the next generation forks fresh.
