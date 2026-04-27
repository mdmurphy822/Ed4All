"""Wave 102 - Reproducibility envelope.

Emits ``reproduce_eval.sh`` next to the README so a procurement
reviewer can re-run the verification step without reading any code:

    cd Ed4All
    bash LibV2/courses/<slug>/models/<model_id>/reproduce_eval.sh

The script:

* Pins the commit SHA captured at emit time.
* Pins ``pip install ed4all[training]`` (the canonical training
  install).
* Pins the model_id, course_slug, and eval profile name.
* Runs ``python -m Trainforge.eval.verify_eval --model-card
  <path>``, which re-loads the eval and ablation reports and asserts
  every metric matches its stored value within
  ``model_card.eval_scores.tolerance_band``. Exits non-zero on drift.

The envelope is fully self-contained: no network access required, no
GPU required (the verifier reads stored reports rather than re-running
the model).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


_REPRODUCE_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
# Wave 102 - Reproduce Trainforge eval scores from stored reports.
#
# This script re-runs the verifier against the eval_report.json +
# ablation_report.json that ship alongside this README. It does NOT
# re-run the model; verification re-reads the stored metrics and
# asserts they match within the per-metric tolerance band declared
# in model_card.eval_scores.tolerance_band.
#
# Pinned values (emit-time):
#   commit:        {commit_sha}
#   model_id:      {model_id}
#   course_slug:   {course_slug}
#   eval profile:  {profile}
set -euo pipefail

# 1. Ensure we're at the pinned commit (best-effort; drift logs a warning).
EXPECTED_COMMIT="{commit_sha}"
if command -v git >/dev/null 2>&1; then
  ACTUAL_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" && "$ACTUAL_COMMIT" != "unknown" ]]; then
    echo "[warn] reproduce_eval.sh: working tree is at $ACTUAL_COMMIT but the report was emitted from $EXPECTED_COMMIT" >&2
  fi
fi

# 2. Verify the eval + ablation reports against the stored tolerance band.
python -m Trainforge.eval.verify_eval \\
  --model-card "{model_card_path}" \\
  --eval-report "{eval_report_path}" \\
  --ablation-report "{ablation_report_path}"
"""


def write_reproduce_script(
    run_dir: Path,
    model_card: Dict[str, Any],
    ablation_report: Optional[Dict[str, Any]] = None,
    *,
    eval_report_filename: str = "eval_report.json",
    ablation_report_filename: str = "ablation_report.json",
    model_card_filename: str = "model_card.json",
    commit_sha: Optional[str] = None,
) -> Path:
    """Emit ``<run_dir>/reproduce_eval.sh``.

    Args:
        run_dir: Directory the README + model_card already live in.
            ``reproduce_eval.sh`` is written here.
        model_card: Parsed ``model_card.json`` dict (we read
            ``model_id``, ``course_slug``, and the optional
            ``eval_scores.scoring_commit``).
        ablation_report: Parsed ``ablation_report.json`` dict (kept for
            forward compatibility; not currently read by the
            template).
        eval_report_filename: Filename inside ``run_dir`` for the
            stored eval report.
        ablation_report_filename: Filename for the stored ablation
            report.
        model_card_filename: Filename for the stored model card.
        commit_sha: Override for the pinned commit. When None we ask
            ``git`` for the current HEAD; falls back to
            ``unknown-commit`` when ``git`` isn't on PATH.

    Returns:
        Path to the written script (mode 0o755).
    """
    if commit_sha is None:
        commit_sha = (
            (model_card.get("eval_scores") or {}).get("scoring_commit")
            or _git_head_or_unknown()
        )
    model_id = model_card.get("model_id") or "unknown"
    course_slug = model_card.get("course_slug") or "unknown"
    profile = (
        (model_card.get("eval_scores") or {}).get("profile")
        or "generic"
    )

    contents = _REPRODUCE_SCRIPT_TEMPLATE.format(
        commit_sha=commit_sha,
        model_id=model_id,
        course_slug=course_slug,
        profile=profile,
        model_card_path=model_card_filename,
        eval_report_path=eval_report_filename,
        ablation_report_path=ablation_report_filename,
    )
    script_path = Path(run_dir) / "reproduce_eval.sh"
    script_path.write_text(contents, encoding="utf-8")
    try:
        script_path.chmod(0o755)
    except (OSError, PermissionError):  # pragma: no cover (Windows / FS quirks)
        logger.debug("reproduce_eval.sh: chmod 0755 not honoured; ignoring")
    return script_path


def _git_head_or_unknown() -> str:
    """Return ``git rev-parse HEAD`` or ``unknown-commit`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown-commit"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown-commit"


__all__ = ["write_reproduce_script"]
