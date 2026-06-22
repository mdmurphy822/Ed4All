"""Stage 6 driver — three-pass region → K-candidate generator.

Walks the regions list in three passes (PROSE → TABLE → MATH),
opening one :class:`AdapterSwap` per pass so the runtime loads the
adapter exactly once per pass. Within a pass, every region is
prompted, dispatched to the runtime, and the K completions are wrapped
as :class:`Candidate`s.

Entry point::

    from dart_semantic.qwen_specialists.runner import run_qwen_specialists

    candidates = run_qwen_specialists(
        regions, feature_blocks,
        k=4, runtime_mode="mock",
    )

The output is a ``dict[int, list[Candidate]]`` keyed by the region's
**index in the input list**. Keys are a subset of
``range(len(regions))`` — only regions that successfully produced
candidates appear (in v1 every routed region produces, but the
contract leaves room for a runtime exception to skip a region while
the rest of the document still runs).

Pass ordering rationale (architecture.md §4.2)
----------------------------------------------

PROSE first: the bulk of regions hit this adapter. TABLE next: it
tends to demand more output tokens per region (cell-level expansion)
so it benefits from a fresh CUDA context. MATH last: shortest
outputs, smallest adapter — the leftover headroom is fine.

Gap-fill (the 4th adapter) is NOT invoked here. It's driven from
Stage 9 once the assembler has flagged gaps; see architecture.md §4.3.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml

from .base import AdapterSwap, generate
from .prompts import build_math_request, build_prose_request, build_table_request
from .routing import adapter_for
from .runtime import QwenRuntime, make_runtime
from .types import AdapterID, Candidate


logger = logging.getLogger(__name__)


# Pass order is locked: PROSE → TABLE → MATH. See architecture.md §4.2.
_PASS_ORDER: tuple[AdapterID, ...] = (
    AdapterID.PROSE,
    AdapterID.TABLE,
    AdapterID.MATH,
)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_sampling(
    config_path: Path | None = None,
    *,
    lane: Literal["fast", "offline"] = "fast",
) -> dict[str, dict[str, Any]]:
    """Read per-adapter sampling defaults from ``config.yaml``.

    Returns a dict keyed by adapter id (string) with values
    ``{"temperature", "top_p", "max_new_tokens", "candidates_k"}``.
    Missing entries get conservative defaults so the driver never
    crashes on a stripped-down config.

    When ``lane="offline"``, each adapter's ``offline_overrides`` block
    is layered on top of its fast-lane defaults. Architecture intent
    (architecture.md §6.3) is looser temperature, larger K, longer
    max_new_tokens — so the offline retry produces a different sample
    rather than re-rolling the same fast-lane output.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in (cfg.get("adapters") or {}).items():
        merged = {
            "temperature": float(entry.get("temperature", 0.6)),
            "top_p": float(entry.get("top_p", 0.95)),
            "max_new_tokens": int(entry.get("max_new_tokens", 512)),
            "candidates_k": int(entry.get("candidates_k", 4)),
            # repetition_penalty: 1.0 == no penalty == llama-cpp-python's
            # own default, so a config omitting the key leaves generation
            # behaviour unchanged. 1.05 is the designated A/B arm for
            # long-target overshoot (see config.yaml math note).
            "repetition_penalty": float(entry.get("repetition_penalty", 1.0)),
        }
        if lane == "offline":
            overrides = entry.get("offline_overrides") or {}
            for ok, ov in overrides.items():
                if ok in {"temperature", "top_p", "repetition_penalty"}:
                    merged[ok] = float(ov)
                elif ok in {"max_new_tokens", "candidates_k"}:
                    merged[ok] = int(ov)
                else:
                    # Forward-compat: any future key is passed through verbatim.
                    merged[ok] = ov
        out[key] = merged
    return out


def _build_request(
    adapter: AdapterID,
    region: Any,
    feature_blocks: Sequence[Any],
) -> Any:
    """Dispatch to the right prompt builder for the given adapter."""
    if adapter is AdapterID.PROSE:
        return build_prose_request(region, feature_blocks)
    if adapter is AdapterID.TABLE:
        return build_table_request(region, feature_blocks)
    if adapter is AdapterID.MATH:
        return build_math_request(region, feature_blocks)
    raise ValueError(f"no prompt builder for adapter {adapter!r}")


def run_qwen_specialists(
    regions: Sequence[Any],
    feature_blocks: Sequence[Any],
    *,
    k: int = 4,
    runtime_mode: Literal["mock", "real"] = "mock",
    config_path: Path | None = None,
    seed: int | None = None,
    runtime: QwenRuntime | None = None,
    lane: Literal["fast", "offline"] = "fast",
) -> dict[int, list[Candidate]]:
    """Run the 3-pass Stage 6 driver.

    Parameters
    ----------
    regions
        Stage 5 typed Region list. The keys in the output dict are
        indices into this list.
    feature_blocks
        The Stage 2 FeatureBlock stream Stage 5 operated on. Used by
        the table prompt builder for caption text lookup.
    k
        How many candidates per region (default 4 per architecture.md
        §4.2). The config's per-adapter ``candidates_k`` is used as a
        fallback when ``k`` is None — but ``k`` is required at the
        public API for clarity.
    runtime_mode
        ``"mock"`` (default; deterministic test/smoke output) or
        ``"real"``. Real-mode requires every adapter referenced by
        the routed regions to have a configured GGUF on disk; if the
        config still has ``adapter_path: null`` (the v1 default),
        the swap will skip the load and the runtime will be asked to
        generate without a model — which raises in
        ``LlamaCppRuntime.generate``. That's by design.
    config_path
        Override path to the qwen_specialists config; useful for
        tests. Defaults to the package-shipped ``config.yaml``.
    seed
        Optional base seed; the runtime offsets per-candidate.
    runtime
        Optional pre-constructed runtime. If supplied,
        ``runtime_mode`` is ignored. Useful for tests that need to
        observe load/free/generate calls.
    lane
        ``"fast"`` (default; back-compat) or ``"offline"``. When
        ``"offline"``, per-adapter ``offline_overrides`` from
        ``config.yaml`` are merged over the fast-lane defaults
        (looser temperature, larger K, longer max_new_tokens). The
        Stage 13 retry orchestrator (`maybe_offline_retry`) calls
        this function a second time with ``lane="offline"`` when the
        fast lane triggered a retry condition.

    Returns
    -------
    dict[int, list[Candidate]]
        Keys are indices into ``regions``; values are length-``k``
        lists. Empty input returns an empty dict.
    """
    if not regions:
        return {}

    sampling = _load_sampling(config_path, lane=lane)

    # Bucket regions by adapter so each pass can run as a single
    # adapter swap. Preserve original input ordering within each pass.
    buckets: dict[AdapterID, list[tuple[int, Any]]] = {a: [] for a in _PASS_ORDER}
    for idx, region in enumerate(regions):
        try:
            adapter = adapter_for(region.kind)
        except ValueError:
            logger.warning(
                "Stage6: skipping region %d with unknown kind %r",
                idx,
                region.kind,
            )
            continue
        if adapter not in buckets:
            # gap_fill or a future adapter — Stage 6 doesn't fire it.
            continue
        buckets[adapter].append((idx, region))

    # Resolve runtime once: a single instance threads through all 3
    # passes (with a load/free per swap). This matches the orchestrator
    # behaviour for the council BERTs.
    rt = runtime if runtime is not None else make_runtime(runtime_mode)

    out: dict[int, list[Candidate]] = {}

    for adapter in _PASS_ORDER:
        bucket = buckets.get(adapter, [])
        if not bucket:
            logger.debug("Stage6 pass %s: no regions, skipping", adapter)
            continue
        defaults = sampling.get(
            adapter.value if isinstance(adapter, AdapterID) else str(adapter),
            {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": 512,
                "repetition_penalty": 1.0,
            },
        )
        logger.info(
            "Stage6 pass %s: %d region(s)",
            adapter,
            len(bucket),
        )
        with AdapterSwap(adapter, runtime=rt, config_path=config_path):
            for idx, region in bucket:
                request = _build_request(adapter, region, feature_blocks)
                # request_id labels candidates back to their region index
                # so downstream stages can trace candidate → region.
                request_with_id = request.__class__(
                    adapter=request.adapter,
                    payload=request.payload,
                    request_id=f"r{idx}",
                    sampling_overrides=request.sampling_overrides,
                )
                # Plans/06 §7 / task #15 + bn34glowl post-mortem (2026-05-30):
                # Two-layer guard against "Requested tokens exceed context
                # window" runtime errors.
                #
                # Layer 1 (build-time, cheap char proxy):
                #   build_table_request flags tables whose serialized prompt
                #   exceeds _LONG_TABLE_CHAR_LIMIT (8.5k chars). See
                #   prompts.py:_LONG_TABLE_CHAR_LIMIT for the empirical
                #   density basis.
                #
                # Layer 2 (runtime, authoritative tokenizer count):
                #   The char proxy underestimates dense tabular content
                #   (2.92 chars/token observed, not 4). For ALL adapters
                #   (not just table — math/prose can theoretically also
                #   overflow), tokenize the wrapped prompt with the real
                #   Qwen tokenizer and skip generation if
                #   prompt_tokens + max_new_tokens would exceed n_ctx.
                #
                # Both paths emit the same empty Candidate so Stage 7 drops
                # it and pass_9a's per-kind fallback fires — every kind has
                # a deterministic emitter with no length limit.
                _md = (request_with_id.sampling_overrides or {}).get("metadata") or {}
                skip_reason: str | None = None
                skip_meta: dict[str, Any] = {}

                if _md.get("skip_qwen_long_table"):
                    skip_reason = "long_table_char_proxy"
                    skip_meta["prompt_char_len"] = _md.get("prompt_char_len")

                # Layer 2: authoritative token check via the runtime's
                # already-loaded HF tokenizer. Skipped if the runtime
                # doesn't expose one (MockRuntime), or if anything in the
                # check itself raises — in which case llama-cpp will hit
                # its own n_ctx error and the per-PDF wrapper records it,
                # same as before this guard (no regression).
                if skip_reason is None and hasattr(rt, "_ensure_tokenizer"):
                    try:
                        from .chat_format import wrap_for_qwen  # noqa: WPS433

                        prompt_str = (
                            request_with_id.payload.get("prompt")
                            if isinstance(request_with_id.payload, dict)
                            else None
                        )
                        if prompt_str is not None:
                            _tok = rt._ensure_tokenizer()
                            _wrapped = wrap_for_qwen(
                                _tok,
                                prompt_str,
                                add_generation_prompt=True,
                            )
                            _n_prompt = len(
                                _tok.encode(
                                    _wrapped,
                                    add_special_tokens=False,
                                )
                            )
                            # n_ctx=4096 hard-coded in LlamaCppRuntime.load;
                            # 32-token safety margin covers BOS/EOS quirks.
                            _N_CTX = 4096
                            _SAFETY = 32
                            _budget = _N_CTX - defaults["max_new_tokens"] - _SAFETY
                            if _n_prompt > _budget:
                                skip_reason = "prompt_plus_generation_over_ctx"
                                skip_meta.update(
                                    {
                                        "prompt_token_count": _n_prompt,
                                        "max_new_tokens": defaults["max_new_tokens"],
                                        "n_ctx": _N_CTX,
                                        "budget_tokens": _budget,
                                        "adapter": adapter.value,
                                    }
                                )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Stage6 token-guard failed for r%d (%s): %s — "
                            "deferring to llama-cpp's own n_ctx check",
                            idx,
                            adapter,
                            exc,
                        )

                if skip_reason is not None:
                    logger.info(
                        "Stage6 skip r%d (%s): %s %s",
                        idx,
                        adapter,
                        skip_reason,
                        skip_meta,
                    )
                    cands = [
                        Candidate(
                            adapter=request_with_id.adapter,
                            request_id=request_with_id.request_id,
                            text="",
                            score=0.0,
                            sampling_seed=seed,
                            finish_reason="skip_long_table",
                            raw_metadata={"skip_reason": skip_reason, **skip_meta},
                        )
                    ]
                else:
                    cands = generate(
                        request_with_id,
                        k=k,
                        temperature=defaults["temperature"],
                        top_p=defaults["top_p"],
                        max_tokens=defaults["max_new_tokens"],
                        seed=seed,
                        repeat_penalty=defaults.get("repetition_penalty", 1.0),
                    )
                # Phase 1.5: tag every Candidate with the adapter
                # version pulled off the runtime (set in
                # LlamaCppRuntime.load via sibling .metadata.json).
                # Backward-compat: missing attr defaults to "unknown".
                version = getattr(rt, "_adapter_version", "unknown")
                tagged: list[Candidate] = []
                for c in cands:
                    new_meta = dict(c.raw_metadata)
                    new_meta["adapter_version"] = version
                    tagged.append(
                        Candidate(
                            adapter=c.adapter,
                            request_id=c.request_id,
                            text=c.text,
                            score=c.score,
                            sampling_seed=c.sampling_seed,
                            finish_reason=c.finish_reason,
                            raw_metadata=new_meta,
                        )
                    )
                out[idx] = tagged

    return out


__all__ = [
    "run_qwen_specialists",
]
