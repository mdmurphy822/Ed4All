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
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator

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
    global _VALIDATOR
    if _VALIDATOR is not None:
        try:
            _VALIDATOR.__exit__(None, None, None)
        except Exception:
            pass
        _VALIDATOR = None


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
    with ProcessPoolExecutor(max_workers=workers, initializer=_initializer) as ex:
        futures = [ex.submit(_task_entry, mod_name, fn_name, item) for item in items]
        for fut in as_completed(futures):
            yield fut.result()
