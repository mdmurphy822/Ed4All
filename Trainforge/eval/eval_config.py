"""Wave 103 - Per-course eval-config loader with hash-pinned variables.

ED4ALL-Bench locks the eval surface so two runs of the same model
against the same corpus produce comparable numbers. The locked
variables (top_k, temperature, top_p, max_new_tokens, seed,
prompt_template_file, rubric_file, benchmark, benchmark_version)
live in ``LibV2/courses/<slug>/eval/eval_config.yaml`` alongside the
prompt template and rubric. When that file is missing, the loader
falls back to ``schemas/eval/default_eval_config.yaml`` and emits a
warning so downstream callers see that this course has not been
customised.

Two SHA-256 hashes flow into ``model_card.json::eval_scores``:

* ``eval_prompt_template_hash`` over the bytes of the loaded prompt
  template file.
* ``eval_config_hash`` over the canonical-JSON serialisation of the
  loaded config dict.

``Trainforge.eval.verify_eval`` re-reads the on-disk template/config
and asserts that the hashes recorded on the model card still match.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


_REQUIRED_KEYS = (
    "benchmark",
    "benchmark_version",
    "top_k",
    "temperature",
    "top_p",
    "max_new_tokens",
    "seed",
    "prompt_template_file",
    "rubric_file",
)


# Resolved at import time so callers can find the default schema dir.
_SCHEMAS_EVAL_DIR = (
    Path(__file__).resolve().parents[2] / "schemas" / "eval"
)


@dataclass
class LoadedEvalConfig:
    """Outcome of :func:`load_eval_config`.

    Attributes:
        config: The dict loaded from ``eval_config.yaml`` (per-course
            or default).
        config_path: Path the config was loaded from. ``None`` when no
            on-disk file existed and the loader synthesised an empty
            dict (should never happen in practice; the default ships
            with the schema).
        prompt_template: Text of the loaded prompt template (per-course
            override or default).
        prompt_template_path: Path the prompt template was loaded from.
        eval_config_hash: SHA-256 over the canonical-JSON shape of the
            config dict.
        eval_prompt_template_hash: SHA-256 over the prompt-template bytes.
        is_default: True when the per-course config was missing and we
            fell back to ``schemas/eval/default_eval_config.yaml``. The
            ablation runner emits a warning in this case so the model
            card carries an "uncustomised" annotation.
    """

    config: Dict[str, Any]
    config_path: Optional[Path]
    prompt_template: str
    prompt_template_path: Optional[Path]
    eval_config_hash: str
    eval_prompt_template_hash: str
    is_default: bool


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_config(config: Dict[str, Any]) -> str:
    """Canonical-JSON SHA-256 of the loaded config dict."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_required_keys(config: Dict[str, Any], source: Path) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(
            f"eval_config.yaml at {source} is missing required keys: "
            f"{sorted(missing)}. Required: {sorted(_REQUIRED_KEYS)}."
        )


def load_eval_config(course_path: Path) -> LoadedEvalConfig:
    """Load the per-course eval config + prompt template.

    Args:
        course_path: Path to ``LibV2/courses/<slug>/``.

    Returns:
        :class:`LoadedEvalConfig` with the parsed config dict, prompt
        template text, and the two SHA-256 hashes that gate the
        reproducibility envelope.

    Behaviour:
        * Per-course path: ``<course_path>/eval/eval_config.yaml``.
        * Default fallback: ``schemas/eval/default_eval_config.yaml``.
          When this branch fires, ``is_default`` is True and a
          warning is logged so the operator knows the course has not
          been customised.
        * The prompt template is resolved relative to the
          eval_config's parent dir (so per-course configs reference
          per-course templates by filename).
    """
    course_path = Path(course_path)
    per_course_config = course_path / "eval" / "eval_config.yaml"

    if per_course_config.exists():
        config_path: Optional[Path] = per_course_config
        is_default = False
    else:
        default_config = _SCHEMAS_EVAL_DIR / "default_eval_config.yaml"
        if not default_config.exists():
            raise FileNotFoundError(
                f"eval_config.yaml missing both per-course "
                f"({per_course_config}) and default "
                f"({default_config}). Wave 103 ships the default; "
                f"check schemas/eval/."
            )
        logger.warning(
            "Wave 103 eval_config: course %s has no per-course "
            "eval_config.yaml; falling back to default %s. Run "
            "`libv2 eval init %s` to scaffold a customisable copy.",
            course_path.name, default_config, course_path.name,
        )
        config_path = default_config
        is_default = True

    raw_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(raw_text) or {}
    if not isinstance(config, dict):
        raise ValueError(
            f"eval_config.yaml at {config_path} did not parse to a dict; "
            f"got {type(config).__name__}."
        )
    _validate_required_keys(config, config_path)

    template_filename = config["prompt_template_file"]
    # Prompt template is resolved relative to the same dir the config
    # came from. Per-course configs ship a sibling prompt_template.txt;
    # the schema default ships default_prompt_template.txt next to
    # default_eval_config.yaml.
    template_candidates = [
        config_path.parent / template_filename,
    ]
    if is_default:
        template_candidates.append(
            _SCHEMAS_EVAL_DIR / "default_prompt_template.txt"
        )
    template_path: Optional[Path] = None
    for candidate in template_candidates:
        if candidate.exists():
            template_path = candidate
            break
    if template_path is None:
        raise FileNotFoundError(
            f"eval_config references prompt_template_file="
            f"{template_filename!r} but no template was found. "
            f"Looked in: {[str(c) for c in template_candidates]}."
        )
    prompt_template = template_path.read_text(encoding="utf-8")

    return LoadedEvalConfig(
        config=config,
        config_path=config_path,
        prompt_template=prompt_template,
        prompt_template_path=template_path,
        eval_config_hash=_hash_config(config),
        eval_prompt_template_hash=_hash_text(prompt_template),
        is_default=is_default,
    )


__all__ = ["LoadedEvalConfig", "load_eval_config"]
