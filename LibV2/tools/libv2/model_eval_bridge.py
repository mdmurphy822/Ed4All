"""Fresh-eval bridge — saved LibV2 adapter → SLM eval harness.

Wave 92 deferred the "run a fresh evaluation from an imported adapter"
half of ``libv2 models eval`` (Wave 93 only surfaced the *cached*
``eval_report.json``). Wave 101 shipped the heavy machinery in
Trainforge — :class:`Trainforge.eval.retrieval.adapter_callable.AdapterCallable`
(loads the base model in 4-bit + applies the saved PEFT adapter,
exposing ``__call__(prompt) -> str``) and
:class:`Trainforge.eval.runners.slm_eval_harness.SLMEvalHarness` (takes any
``model_callable`` and scores it). This module wires those two together
against a LibV2 course tree: the imported model dir
(``courses/<slug>/models/<model_id>/``) is a copytree of the training
run dir, so it already carries ``adapter_model.safetensors`` +
``adapter_config.json`` + tokenizer files + ``model_card.json`` with
``base_model.{name, revision, huggingface_repo}`` — everything needed to
rebuild the callable is on disk.

Design constraints:

* **Lazy Trainforge imports** — the heavy ML deps (``torch`` /
  ``transformers`` / ``peft`` / ``bitsandbytes``) only load when a REAL
  :class:`AdapterCallable` is built (inside :meth:`AdapterCallable.__init__`).
  A bare ``import LibV2.tools.libv2.model_eval_bridge`` stays cheap on a
  CPU-only box; every Trainforge import is in-function.
* **Injectable ``model_callable``** — tests (and any caller that already
  holds a callable) pass ``model_callable=`` to skip the adapter load
  entirely, so the harness wiring is exercised with a fake callable and
  zero GPU dependency.
* **Non-destructive by default** — a fresh run writes
  ``eval_report.fresh-<ts>.json`` alongside the model card. The canonical
  ``eval_report.json`` (which ``get_model_eval_report`` /
  ``EvalGatingValidator`` consume) is only overwritten under an explicit
  ``replace=True``, and then only after a ``.bak`` backup — a fresh eval
  can legitimately differ from the training-time one (eval-set drift,
  gen-config drift), so the canonical report stays training-time unless
  the operator deliberately replaces it.

Real GPU runs must be wrapped in ``scripts/ops/gpu_guard.sh`` on the shared
box (``ED4ALL_GPU_LIFECYCLE`` only sweeps inside ``ed4all run``, not a
standalone CLI call). See ``LibV2/CLAUDE.md`` § models.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


# Actionable guidance surfaced when the heavy training deps are absent.
# Mirrors the runner's Wave-101 ImportError fallback contract + names the
# mandatory gpu_guard wrap for the shared-GPU box.
TRAINING_DEPS_GUIDANCE = (
    "A fresh evaluation loads the base model + saved adapter, which needs "
    "the training ML stack (torch / transformers / peft / bitsandbytes). "
    "Install it with:\n"
    "    pip install -e '.[training]'\n"
    "from the project root, then re-run behind the shared-GPU guard:\n"
    "    scripts/ops/gpu_guard.sh run --task libv2-fresh-eval -- \\\n"
    "        libv2 models eval <slug> <model_id> --fresh\n"
    "(ED4ALL_GPU_LIFECYCLE only sweeps the card inside `ed4all run`, not a "
    "standalone CLI call, so the gpu_guard wrap is mandatory on a box that "
    "shares the GPU with resident ollama models.)"
)


class FreshEvalError(RuntimeError):
    """Raised when the fresh-eval bridge cannot proceed (bad inputs)."""


def _load_model_card(model_dir: Path) -> dict:
    card_path = model_dir / "model_card.json"
    if not card_path.exists():
        raise FreshEvalError(
            f"model_card.json not found under {model_dir}. A fresh eval "
            f"needs the card's base_model.{{name,revision,huggingface_repo}} "
            f"to rebuild the adapter callable."
        )
    try:
        return json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshEvalError(
            f"Could not read model_card.json at {card_path}: {exc}"
        ) from exc


def build_adapter_callable(
    model_dir: Path,
    loaded_eval_config: Any,
) -> Callable[[str], str]:
    """Rebuild the :class:`AdapterCallable` for an imported model dir.

    Reads ``model_card.json::base_model`` for the HF repo / revision /
    short name and maps the per-course eval-config generation knobs
    (``max_new_tokens`` / ``temperature`` / ``top_p`` / ``seed``) into
    the callable — the exact recipe
    :meth:`Trainforge.training.runner.TrainingRunner._run_eval_harness`
    uses at training time, so a fresh eval matches the training-time
    generation envelope.

    Trainforge is imported lazily (in-function) so a bare import of this
    module stays CPU-cheap; the heavy ML deps only load inside
    :meth:`AdapterCallable.__init__`. Missing deps raise ``ImportError``
    (the CLI translates it into :data:`TRAINING_DEPS_GUIDANCE`).

    Args:
        model_dir: ``courses/<slug>/models/<model_id>/`` — the copytree of
            the training run dir carrying the adapter + card + tokenizer.
        loaded_eval_config: A ``LoadedEvalConfig`` from
            :func:`Trainforge.eval.eval_config.load_eval_config`.

    Returns:
        An :class:`AdapterCallable` (a ``Callable[[str], str]``).
    """
    from Trainforge.eval.retrieval.adapter_callable import AdapterCallable

    card = _load_model_card(Path(model_dir))
    base = card.get("base_model") or {}
    repo = base.get("huggingface_repo")
    if not repo:
        raise FreshEvalError(
            f"model_card.json under {model_dir} has no "
            f"base_model.huggingface_repo; cannot resolve the base model "
            f"for a fresh eval."
        )
    cfg = getattr(loaded_eval_config, "config", {}) or {}
    return AdapterCallable(
        adapter_dir=Path(model_dir),
        base_model_repo=repo,
        base_model_short_name=base.get("name"),
        revision=base.get("revision"),
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        seed=int(cfg.get("seed", 42)),
    )


def run_fresh_eval(
    course_slug: str,
    model_id: str,
    repo_root: Path,
    *,
    model_callable: Optional[Callable[[str], str]] = None,
    smoke: bool = False,
    output_path: Optional[Path] = None,
    replace: bool = False,
    capture: Any = None,
) -> Path:
    """Run a fresh SLM evaluation against an imported LibV2 adapter.

    Wires :class:`AdapterCallable` (rebuilt from the model dir) into
    :class:`SLMEvalHarness` (pointed at the course tree) and writes a
    fresh ``eval_report.json``. Non-destructive by default.

    Args:
        course_slug: Course slug under ``<repo_root>/courses/``.
        model_id: Model id under ``courses/<slug>/models/``.
        repo_root: LibV2 root (the dir that holds ``courses/``).
        model_callable: Optional pre-built callable. When ``None`` a real
            :class:`AdapterCallable` is built from the model dir (loads
            the heavy ML stack). Tests inject a fake callable here.
        smoke: When True, run the harness in smoke mode (N=3 probes/stage).
        output_path: Explicit output path. When ``None`` the report lands
            at ``models/<model_id>/eval_report.fresh-<ts>.json`` (default,
            non-destructive) or at the canonical ``eval_report.json`` when
            ``replace=True`` (after a ``.bak`` backup).
        replace: When True, overwrite the canonical ``eval_report.json``
            (backing the prior one up to ``eval_report.json.bak`` first) so
            ``get_model_eval_report`` / ``EvalGatingValidator`` pick up the
            fresh scores deliberately.
        capture: Optional :class:`lib.decision_capture.DecisionCapture`. The
            bridge logs one ``fresh_eval_invocation`` decision when supplied
            (best-effort — a capture failure never fails the eval).

    Returns:
        Path to the written report.
    """
    import shutil

    repo_root = Path(repo_root)
    course_dir = repo_root / "courses" / course_slug
    if not course_dir.exists():
        raise FreshEvalError(f"Course not found: {course_dir}")
    model_dir = course_dir / "models" / model_id
    if not model_dir.exists():
        raise FreshEvalError(f"Model not found: {model_dir}")

    from Trainforge.eval.eval_config import load_eval_config
    from Trainforge.eval.runners.slm_eval_harness import SLMEvalHarness

    loaded_eval_config = load_eval_config(course_dir)

    if model_callable is None:
        model_callable = build_adapter_callable(model_dir, loaded_eval_config)

    harness = SLMEvalHarness(
        course_path=course_dir,
        model_callable=model_callable,
        smoke_mode=smoke,
    )

    # Resolve the output target. Default is a timestamped, non-destructive
    # sidecar; --replace targets the canonical report after a .bak backup.
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    canonical_path = model_dir / "eval_report.json"
    if output_path is not None:
        target_path = Path(output_path)
    elif replace:
        target_path = canonical_path
    else:
        target_path = model_dir / f"eval_report.fresh-{ts}.json"

    backup_path: Optional[Path] = None
    if replace and canonical_path.exists() and target_path == canonical_path:
        backup_path = canonical_path.with_suffix(".json.bak")
        shutil.copy2(canonical_path, backup_path)
        logger.info("Backed up prior eval_report.json to %s", backup_path)

    cfg = getattr(loaded_eval_config, "config", {}) or {}
    # Best-effort decision capture BEFORE the (potentially long) run so the
    # invocation is recorded even if the harness later crashes. The eval is
    # the contract; the audit event is best-effort, so guard construction +
    # logging.
    if capture is not None:
        try:
            base_repo = "unknown"
            try:
                base_repo = (
                    (_load_model_card(model_dir).get("base_model") or {})
                    .get("huggingface_repo", "unknown")
                )
            except FreshEvalError:
                pass
            injected = model_callable is not None
            rationale = (
                f"Fresh SLM eval for model_id={model_id} (course={course_slug}) "
                f"base_repo={base_repo} profile={harness.profile_name} "
                f"smoke={smoke} injected_callable={injected} "
                f"gen(max_new_tokens={cfg.get('max_new_tokens', 256)},"
                f"temperature={cfg.get('temperature', 0.0)},"
                f"seed={cfg.get('seed', 42)}) "
                f"replace={replace} -> {target_path.name}"
            )
            capture.log_decision(
                decision_type="fresh_eval_invocation",
                decision=(
                    f"run_fresh_eval:{course_slug}/{model_id}"
                    f"{' [replace]' if replace else ''}"
                ),
                rationale=rationale,
                alternatives_considered=[
                    {
                        "option": "surface cached eval_report.json"
                        " (default models eval)",
                        "reason_rejected": "operator asked for a fresh"
                        " re-score of the saved adapter",
                    },
                    {
                        "option": "retrain to re-score"
                        " (Trainforge.train_course)",
                        "reason_rejected": "full retraining is not needed"
                        " to re-evaluate an existing adapter",
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001 — capture is best-effort.
            logger.warning("fresh_eval_invocation capture failed: %s", exc)

    report_path = harness.run_all(output_path=target_path)
    logger.info("Fresh eval report written to %s", report_path)
    return Path(report_path)


__all__ = [
    "TRAINING_DEPS_GUIDANCE",
    "FreshEvalError",
    "build_adapter_callable",
    "run_fresh_eval",
]
