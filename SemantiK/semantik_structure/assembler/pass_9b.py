"""Stage 9b — Qwen-GapFill driver.

For each :class:`GapSlot` flagged by Stage 9a:

  1. Build a :class:`SpecialistRequest` via
     :func:`qwen_specialists.prompts.build_gap_fill_request`.
  2. Open a single :class:`AdapterSwap` (``GAP_FILL``) and ask the
     runtime to generate K candidates per gap.
  3. Wrap each completion as a :class:`Candidate`.
  4. Run each candidate through the per-region hard gate using a
     synthetic prose-shell :class:`Region` (paragraph kind, empty
     ``feature_block_indices``) so the text-preservation check
     defaults to "skipped:no_source_text" (verified by reading
     ``gates.hard_region._check_text_preservation`` line ~226).
  5. Return ``dict[GapSlot -> list[Candidate]]`` of survivors. Empty
     list means all K candidates failed the gate; Stage 9c falls
     back to :attr:`GapSlot.fallback_html`.

Mock runtime is the default. ``config.yaml`` now configures real GGUF
paths for all four adapters (prose/table/math/gap_fill), so the real
runtime loads the gap-fill adapter normally; it raises only if a
configured ``adapter_path`` is missing or points at an absent file.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from ..gates.hard_region import check_region
from ..gates.types import GateResult
from ..qwen_specialists.base import AdapterSwap, generate
from ..qwen_specialists.prompts import build_gap_fill_request
from ..qwen_specialists.runtime import (
    QwenRuntime,
    make_runtime,
    resolve_endpoint_displaces_local,
    specialist_provider_is_endpoint,
)
from ..qwen_specialists.types import AdapterID, Candidate
from ..structure_graph import Region
from ..validate import HtmlValidator
from .types import GapSlot

if TYPE_CHECKING:
    from .api import AssemblerConfig


logger = logging.getLogger(__name__)


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "qwen_specialists" / "config.yaml"
)


def _load_gap_fill_sampling(config_path: Path | None = None) -> dict[str, Any]:
    """Read the ``gap_fill`` sampling defaults from ``config.yaml``."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    entry = ((cfg.get("adapters") or {}).get("gap_fill") or {})
    return {
        "temperature": float(entry.get("temperature", 0.8)),
        "top_p": float(entry.get("top_p", 0.95)),
        "max_new_tokens": int(entry.get("max_new_tokens", 256)),
        "candidates_k": int(entry.get("candidates_k", 8)),
    }


def _gap_synthetic_region() -> Region:
    """Return a synthetic ``paragraph`` region with empty FB indices.

    The gap-fill candidate has no source FBs to measure preservation
    against — :func:`gates.hard_region._gather_source_text` returns
    "" and the text-preserve check short-circuits with
    ``passed=True, message="skipped:no_source_text"``. This avoids
    introducing a per-candidate sentinel into :class:`Region`.
    """
    return Region(
        kind="paragraph",
        feature_block_indices=(),
        payload={},
        provenance={"pass": "gap_fill_synthetic"},
    )


def run_pass_9b(
    gaps_found: Sequence[GapSlot],
    *,
    runtime_mode: Literal["mock", "real"] = "mock",
    config: "AssemblerConfig | None" = None,
    config_path: Path | None = None,
    runtime: QwenRuntime | None = None,
    validator: HtmlValidator | None = None,
) -> dict[int, list[Candidate]]:
    """Generate + gate K candidates per gap. Returns survivors per gap.

    Keyed by index into ``gaps_found`` (parallel positional dictionary
    so callers don't need to hash :class:`GapSlot` instances). Empty
    survivors list means all K candidates failed the per-region gate.
    """
    if not gaps_found:
        return {}

    if config is None:
        from .api import AssemblerConfig
        config = AssemblerConfig()

    sampling = _load_gap_fill_sampling(config_path)
    k = int(config.gap_fill_k or sampling["candidates_k"])

    # Dereliction fix (mirrors the Stage-6 runner): the local gap_fill GGUF is
    # the authoring tier by default. Force the local runtime when an endpoint
    # PROVIDER is configured but the operator has NOT explicitly opted into
    # pure-endpoint displacement, so gap_fill stays on the on-device
    # specialist rather than silently routing to the hosted 70B. gap_fill is a
    # single-pass synthesis (no hybrid refine), so only the DISPLACE opt-in
    # routes it to the endpoint.
    force_local = (
        specialist_provider_is_endpoint() and not resolve_endpoint_displaces_local()
    )
    rt = (
        runtime
        if runtime is not None
        else make_runtime(
            runtime_mode, config_path=config_path, force_local=force_local
        )
    )

    out: dict[int, list[Candidate]] = {gi: [] for gi in range(len(gaps_found))}

    # One AdapterSwap covers every gap (adapter loaded exactly once).
    with AdapterSwap(
        AdapterID.GAP_FILL, runtime=rt, config_path=config_path,
    ):
        for gi, gap in enumerate(gaps_found):
            request = build_gap_fill_request(gap)
            try:
                cands = generate(
                    request,
                    k=k,
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    max_tokens=sampling["max_new_tokens"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Stage 9b: gap-fill generate failed for %s: %s",
                    gap.kind, exc,
                )
                out[gi] = []
                continue
            out[gi] = cands

    # Gate each candidate. Reuse the caller's validator when provided
    # (the eval / smoke driver opens one for the whole cascade); only
    # open our own when the caller didn't. Two nested
    # ``sync_playwright().start()`` blocks crash with
    # "using Playwright Sync API inside the asyncio loop".
    survivors: dict[int, list[Candidate]] = {}
    synth_region = _gap_synthetic_region()

    def _gate_with(v: HtmlValidator | None) -> None:
        for gi, cands in out.items():
            kept: list[Candidate] = []
            for c in cands:
                gr: GateResult = check_region(
                    c.text,
                    region_id=-1,
                    region=synth_region,
                    feature_blocks=(),
                    validator=v,
                    tau=0.0,  # gap-fill has no source text; tau=0 is a no-op.
                )
                if gr.passed:
                    kept.append(c)
            survivors[gi] = kept

    if validator is not None:
        _gate_with(validator)
        return survivors

    try:
        with HtmlValidator() as v:
            _gate_with(v)
    except Exception as exc:  # noqa: BLE001
        # Validator boot failed — fall back to text-only well-formedness
        # via check_region with no validator (still cheap, deterministic).
        logger.warning(
            "Stage 9b: HtmlValidator failed (%s); using text-only gating", exc,
        )
        _gate_with(None)

    return survivors


__all__ = ["run_pass_9b"]
