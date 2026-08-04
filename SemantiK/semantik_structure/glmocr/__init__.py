"""GLM-OCR extraction lane for SemantiK (owner-adopted architecture).

The GLM-OCR lane is a WHOLE-DOCUMENT converter that swaps in for the default
omni v2 cascade behind the master flag ``SEMANTIK_GLMOCR_LANE`` (default OFF —
the omni cascade is the byte-identical default path). Pipeline:

    PDF --(pdftoppm 300 DPI)--> page renders
        --(GLM-OCR SDK: PP-DocLayoutV3 layout + per-region OCR on the vLLM
           seat)--> per-page MD+JSON region layout
        --(deterministic transform, NO LLM)--> region_provenance (the SemantiK
           wire contract) + heading_tree + Super-escalation records
        --(optional Qwen3-VL alt-text/caption pass)--> figure alt text
        --> the existing ``lib/semantik`` adapter renders the accessible HTML +
            data-semantik-*/``semantik:{slug}#{block_id}`` provenance contract.

Design rule (mirrors the rest of SemantiK): learned models are narrow candidate
generators; DETERMINISTIC code owns composition / classification / provenance.
The GLM-OCR seat and the alt-text seat PROPOSE region OCR + alt text; the
transform DECIDES structure, and the deterministic rules are auditable code.

Contract preservation: the lane emits ``region_provenance`` in the SAME shape
the omni cascade emits (see ``cascade.py::_build_region_provenance`` /
``lib/semantik/cascade_ir.py::_block_from_provenance``), so the downstream
adapter + chunker + staging path consume the lane's output UNCHANGED.

All env resolvers are parse-with-fallback: unset / blank / garbage → the
documented default; a run never crashes on a malformed flag (root ``ED4ALL_*``
posture).
"""

from __future__ import annotations

import os

__all__ = [
    "resolve_glmocr_lane_mode",
    "resolve_glmocr_base_url",
    "resolve_glmocr_model",
    "resolve_glmocr_workers",
    "resolve_glmocr_render_dpi",
    "resolve_glmocr_layout_device",
    "resolve_alttext_provider",
    "resolve_alttext_base_url",
    "resolve_alttext_model",
    "resolve_alttext_api_key",
    "resolve_alttext_concurrency",
    "resolve_alttext_timeout",
    "resolve_heading_judge_mode",
    "resolve_heading_judge_base_url",
    "resolve_heading_judge_model",
    "resolve_heading_judge_timeout",
    "resolve_heading_judge_checkpoint",
]

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in _TRUTHY if value is not None else False


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return val if val >= minimum else default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        val = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if val != val or val in (float("inf"), float("-inf")):  # NaN / Inf
        return default
    return val if val > minimum else default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def resolve_glmocr_lane_mode() -> bool:
    """Master gate: is the GLM-OCR lane active? Default OFF (omni cascade)."""
    return _truthy(os.environ.get("SEMANTIK_GLMOCR_LANE"))


def resolve_glmocr_base_url() -> str:
    """OpenAI-compatible GLM-OCR seat base URL (default localhost:8002/v1)."""
    return _env_str("SEMANTIK_GLMOCR_BASE_URL", "http://localhost:8002/v1")


def resolve_glmocr_model() -> str:
    """GLM-OCR seat model id (default ``glm-ocr``)."""
    return _env_str("SEMANTIK_GLMOCR_MODEL", "glm-ocr")


def resolve_glmocr_workers() -> int:
    """SDK parallel-worker count (default 4; measured 4,453 pages/hr)."""
    return _env_int("SEMANTIK_GLMOCR_WORKERS", 4)


def resolve_glmocr_render_dpi() -> int:
    """pdftoppm render DPI for page rasters (default 300)."""
    return _env_int("SEMANTIK_GLMOCR_RENDER_DPI", 300, minimum=72)


def resolve_glmocr_layout_device() -> str:
    """PP-DocLayoutV3 device for the SDK layout model (default ``cpu``)."""
    return _env_str("SEMANTIK_GLMOCR_LAYOUT_DEVICE", "cpu")


def resolve_alttext_provider() -> str:
    """Alt-text/caption generation provider: ``off`` (default) | ``qwen30``.

    Default ``off`` inside the lane until seat provisioning is wired — the lane
    ships harvested captions + honest placeholders, never fabricated alt text.
    """
    val = _env_str("SEMANTIK_ALTTEXT_PROVIDER", "off").lower()
    return val if val in {"off", "qwen30"} else "off"


def resolve_alttext_base_url() -> str:
    """OpenAI-compatible alt-text seat base URL (default localhost:8003/v1)."""
    return _env_str("SEMANTIK_ALTTEXT_BASE_URL", "http://localhost:8003/v1")


def resolve_alttext_model() -> str:
    """Alt-text seat model id (default ``qwen3-vl-30b``, Apache-2.0)."""
    return _env_str("SEMANTIK_ALTTEXT_MODEL", "qwen3-vl-30b")


def resolve_alttext_api_key() -> str | None:
    """Alt-text seat bearer token (loopback needs none)."""
    raw = os.environ.get("SEMANTIK_ALTTEXT_API_KEY")
    return str(raw).strip() if raw and str(raw).strip() else None


def resolve_alttext_concurrency() -> int:
    """Alt-text batch fan-out width (default 8)."""
    return _env_int("SEMANTIK_ALTTEXT_CONCURRENCY", 8)


def resolve_alttext_timeout() -> float:
    """Alt-text per-request HTTP timeout in seconds (default 120)."""
    return _env_float("SEMANTIK_ALTTEXT_TIMEOUT_SECONDS", 120.0)


def resolve_alttext_cache_dir() -> str:
    """Durable per-figure describe cache directory (empty = disabled).

    With a directory set, each successful describe persists a
    content-addressed JSON sidecar keyed by crop bytes + caption + model,
    and re-runs serve cached figures without a seat call.
    """
    return _env_str("SEMANTIK_ALTTEXT_CACHE_DIR", "")


# ── Super heading-level JUDGE pass (SEMANTIK_HEADING_JUDGE). ─────────────────
_CHECKPOINT_FAMILY_ENV = "ED4ALL_GENERATION_CHECKPOINT"
_CHECKPOINT_FALSEY = {"0", "false", "no", "off"}


#: Explicit opt-out tokens for the default-ON heading-judge gate (mirrors the
#: ED4ALL_GPU_LIFECYCLE default-ON house pattern).
_HEADING_JUDGE_FALSEY = {"0", "false", "no", "off"}


def resolve_heading_judge_mode() -> bool:
    """Master gate: run the Super heading-level judge pass? Default ON (deviation).

    Owner directive: the Super heading-level judge must not be optional. Only
    an explicit falsey token (``0`` / ``false`` / ``no`` / ``off``,
    case-insensitive) disables; unset / blank / garbage / truthy → on
    (parse-with-fallback, the ``ED4ALL_GPU_LIFECYCLE`` default-ON pattern).
    When explicitly off the ``heading_judge`` module is never imported by the
    lane, so ``region_provenance`` / ``heading_tree`` / escalations are
    byte-identical. A no-pending (born-digital / fully-anchored) chapter stays
    a natural no-op even when the gate is on.
    """
    raw = os.environ.get("SEMANTIK_HEADING_JUDGE")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in _HEADING_JUDGE_FALSEY


def resolve_heading_judge_base_url() -> str:
    """Heading-judge seat base URL (default the TRT-LLM Super seat :8123/v1)."""
    return _env_str("SEMANTIK_HEADING_JUDGE_BASE_URL", "http://localhost:8123/v1")


def resolve_heading_judge_model() -> str:
    """Heading-judge seat model id (default the served TRT-LLM Super id).

    ``trtllm-serve`` has no ``--served-model-name``, so it serves the checkpoint
    under its full HF path — the default matches what ``/v1/models`` reports."""
    return _env_str(
        "SEMANTIK_HEADING_JUDGE_MODEL",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    )


def resolve_heading_judge_timeout() -> float:
    """Heading-judge per-request HTTP timeout in seconds.

    Defaults to 300 with thinking off (completion capped near 4096 tokens)
    and 1200 with thinking on. ``SEMANTIK_HEADING_JUDGE_TIMEOUT`` beats
    both defaults.
    """
    thinking_enabled = _truthy(os.environ.get("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING"))
    default = 1200.0 if thinking_enabled else 300.0
    return _env_float("SEMANTIK_HEADING_JUDGE_TIMEOUT", default)


def resolve_heading_judge_checkpoint() -> bool:
    """Per-window resume-cache / stop-checkpoint gate (default ON).

    Site env ``SEMANTIK_HEADING_JUDGE_CHECKPOINT`` beats the
    ``ED4ALL_GENERATION_CHECKPOINT`` family; explicit falsey → off.
    """
    site = os.environ.get("SEMANTIK_HEADING_JUDGE_CHECKPOINT")
    if site is not None and str(site).strip():
        return str(site).strip().lower() not in _CHECKPOINT_FALSEY
    fam = os.environ.get(_CHECKPOINT_FAMILY_ENV)
    if fam is not None and str(fam).strip():
        return str(fam).strip().lower() not in _CHECKPOINT_FALSEY
    return True
