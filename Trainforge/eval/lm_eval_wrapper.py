"""Wave 101 - EleutherAI ``lm-evaluation-harness`` wrapper.

Optional generic-benchmark sanity floor for trained adapters. Domain
SLMs typically score below their base on generic benchmarks (ARC,
TruthfulQA, HellaSwag), so the value is the **no catastrophic
forgetting** signal: if the trained adapter craters on a generic
benchmark relative to base, that's a flag.

Behaviour:

* When the harness package is **not** installed, return ``None``
  and emit a warning. This is OPTIONAL telemetry; the rest of the
  Wave 101 pipeline must keep working on a CPU-only dev box.
* When the harness IS installed, call its ``simple_evaluate`` entry
  point with ``model="hf"`` plus ``model_args`` pointing at the
  adapter + base, then write the raw results JSON to
  ``<run_dir>/lm_eval_results/results.json``.

Wave 101 default tasks (cheap, ~5 min total on RTX 3070):

* ``arc_easy``       - elementary-school multiple-choice science
* ``truthfulqa_mc1`` - factual-truthfulness MC
* ``hellaswag``      - commonsense sentence completion

The ``[training]`` extra in ``pyproject.toml`` adds
``lm-eval>=0.4.0,<1.0.0``; users opt in via ``pip install
ed4all[training]`` then re-install (Wave 101 doesn't auto-install on
the running training box).
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


_HARNESS_PACKAGE = "lm_eval"
_DEFAULT_TASKS = ("arc_easy", "truthfulqa_mc1", "hellaswag")


def run_lm_eval(
    adapter_dir: Path,
    base_model_repo: str,
    run_dir: Path,
    tasks: Optional[List[str]] = None,
    *,
    batch_size: int = 4,
    device: Optional[str] = None,
) -> Optional[Path]:
    """Run the lm-evaluation-harness against a saved adapter.

    Args:
        adapter_dir: Where TRL's ``save_model()`` wrote the PEFT
            adapter. Passed via ``model_args=peft=<adapter_dir>``.
        base_model_repo: HF repo identifier of the base model. Passed
            via ``model_args=pretrained=<repo>``.
        run_dir: Directory the results JSON is written under
            (``<run_dir>/lm_eval_results/results.json``).
        tasks: Override the default benchmark task list.
        batch_size: Per-device evaluation batch size.
        device: Optional device override; ``None`` lets the harness
            pick.

    Returns:
        Path to the results JSON when the harness ran, else ``None``
        when the package is unavailable (graceful skip, not an error).
    """
    try:
        harness_mod = importlib.import_module(_HARNESS_PACKAGE)
    except ImportError:
        logger.warning(
            "Trainforge.eval.lm_eval_wrapper: ``%s`` package not "
            "installed. Skipping generic-benchmark sanity floor. "
            "Install with: pip install ed4all[training] (then re-run).",
            _HARNESS_PACKAGE,
        )
        return None

    # The harness module exposes ``simple_evaluate``. Resolve via
    # getattr so missing-symbol errors surface as AttributeError
    # rather than ImportError, distinguishable in tests.
    runner = getattr(harness_mod, "simple_evaluate", None)
    if runner is None:
        logger.warning(
            "Trainforge.eval.lm_eval_wrapper: ``%s.simple_evaluate`` "
            "not found. Skipping (package present but API mismatch).",
            _HARNESS_PACKAGE,
        )
        return None

    task_list = list(tasks) if tasks else list(_DEFAULT_TASKS)

    model_args = (
        f"pretrained={base_model_repo},"
        f"peft={adapter_dir},"
        "load_in_4bit=True"
    )
    logger.info(
        "Trainforge.eval.lm_eval_wrapper: running %d tasks (%s) "
        "against adapter=%s base=%s",
        len(task_list), ", ".join(task_list), adapter_dir, base_model_repo,
    )

    runner_kwargs: Dict[str, Any] = {
        "model": "hf",
        "model_args": model_args,
        "tasks": task_list,
        "batch_size": batch_size,
    }
    if device is not None:
        runner_kwargs["device"] = device

    raw_results = runner(**runner_kwargs)

    out_dir = Path(run_dir) / "lm_eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    # The harness returns a dict containing ``results`` plus
    # version/config metadata. Some entries (the cached model handle)
    # are not JSON-serialisable, so we strip down to the documented
    # leaderboard surface before writing.
    serialisable = _strip_for_json(raw_results)
    out_path.write_text(
        json.dumps(serialisable, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return out_path


def summarize_lm_eval(results_path: Path) -> Dict[str, Any]:
    """Extract the headline accuracy score per task from results.json.

    Used by the runner to fold generic-benchmark scores into the
    ``model_card.json::eval_scores.lm_eval_summary`` block.
    """
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    results = payload.get("results") or {}
    summary: Dict[str, Any] = {}
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        # Prefer ``acc,none`` then ``acc`` then ``acc_norm,none``.
        for key in ("acc,none", "acc", "acc_norm,none", "acc_norm"):
            if key in metrics:
                try:
                    summary[task_name] = round(float(metrics[key]), 4)
                except (TypeError, ValueError):
                    pass
                break
    return summary


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _strip_for_json(payload: Any) -> Any:
    """Drop non-serialisable handles from the harness payload.

    The harness embeds tokenizer / model objects in some fields;
    filter them out so ``json.dumps`` doesn't blow up. Standard
    primitives (numbers, strings, bools, lists, dicts) pass through.
    """
    if isinstance(payload, dict):
        return {
            k: _strip_for_json(v)
            for k, v in payload.items()
            if not _is_handle(v)
        }
    if isinstance(payload, list):
        return [_strip_for_json(item) for item in payload if not _is_handle(item)]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    # Fall back to repr for anything exotic (covers numpy scalars etc.).
    return repr(payload)


def _is_handle(value: Any) -> bool:
    """Heuristic: drop torch / transformers handles from the dump."""
    cls = type(value)
    mod = getattr(cls, "__module__", "") or ""
    if mod.startswith(("torch.", "transformers.", "peft.")):
        return True
    return False


__all__ = ["run_lm_eval", "summarize_lm_eval"]
