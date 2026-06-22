"""Stage 6 driver — BATCHED two-phase region → K-candidate generator.

The driver buckets regions by their routed adapter, then runs up to two
phases (provider/mode-gated) — replacing the old per-region swap loop:

    Phase 1 (local drafts, batched BY ADAPTER)
        For each adapter group, ONE :class:`AdapterSwap` load, then a
        single :meth:`QwenRuntime.generate_batch` over every region in
        that group (run K times, one per candidate slot). The math
        adapter loads ONCE for all math regions, etc. — the swap-thrash
        fix. SKIPPED entirely when the specialist provider is an endpoint
        (no local adapters to run).

    Phase 2 (70B refine / generate, batched CONCURRENTLY)
        Gated by ``SEMANTIK_SPECIALIST_PROVIDER`` + ``SEMANTIK_SPECIALIST_REFINE``:
          * provider=local (default): Phase 1 ONLY. No endpoint calls.
            Byte-stable behaviour + the swap-thrash win.
          * provider=<endpoint> (no refine): SKIP Phase 1; build per-region
            prompts; one concurrent :meth:`generate_batch` over ALL regions.
          * provider=<endpoint> + REFINE=1 (hybrid): Phase 1 local drafts
            THEN Phase 2 sends each region's (prompt + local draft) to the
            70B with a refine directive; the 70B output REPLACES the draft.

Entry point::

    from dart_semantic.qwen_specialists.runner import run_qwen_specialists

    candidates = run_qwen_specialists(
        regions, feature_blocks,
        k=4, runtime_mode="mock",
    )

The output is a ``dict[int, list[Candidate]]`` keyed by the region's
**index in the input list** — the SAME shape the assembler/reranker
consumes regardless of which phase produced each completion. Only the
ORDER and BATCHING of generation changed, not the per-region output
shape. Keys are a subset of ``range(len(regions))`` — only regions that
successfully produced candidates appear.

Adapter-group ordering (architecture.md §4.2)
---------------------------------------------

PROSE first: the bulk of regions hit this adapter. TABLE next: it
tends to demand more output tokens per region. MATH last: shortest
outputs, smallest adapter. The group order is unchanged from the
three-pass driver; only the within-group generation is now batched.

Gap-fill (the 4th adapter) is NOT invoked here. It's driven from
Stage 9 once the assembler has flagged gaps; see architecture.md §4.3.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .base import AdapterSwap
from .prompts import build_math_request, build_prose_request, build_table_request
from .routing import adapter_for
from .runtime import QwenRuntime, make_runtime, specialist_provider_is_endpoint
from .types import AdapterID, Candidate


logger = logging.getLogger(__name__)


# Adapter-group order is locked: PROSE → TABLE → MATH. See architecture.md §4.2.
_PASS_ORDER: tuple[AdapterID, ...] = (
    AdapterID.PROSE,
    AdapterID.TABLE,
    AdapterID.MATH,
)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Truthy strings for SEMANTIK_SPECIALIST_REFINE (parse-with-fallback; any
# other value is falsey/off).
_REFINE_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Directive prepended to a region's prompt when the 70B is asked to REFINE
# an existing local draft (the hybrid Phase 2 flow). Keeps the same bare-
# fragment output envelope the assembler parses — the endpoint runtime's
# own _ENVELOPE_DIRECTIVE still applies; this adds the refine instruction
# into the USER turn so the draft travels with the region.
_REFINE_DIRECTIVE = (
    "A draft fragment for this region was produced by a smaller local "
    "specialist. Improve and COMPLETE it: fix any malformed markup, fill "
    "gaps, and raise the quality, but keep it grounded in the region "
    "described above. Emit ONLY the corrected fragment — no commentary, no "
    "code fences."
)


def resolve_refine_mode() -> bool:
    """True when ``SEMANTIK_SPECIALIST_REFINE`` opts into the hybrid flow.

    Parse-with-fallback: only the truthy set enables refine; anything else
    (unset / blank / garbage) is off."""
    raw = (os.environ.get("SEMANTIK_SPECIALIST_REFINE") or "").strip().lower()
    return raw in _REFINE_TRUTHY


def _refine_prompt(region_prompt: str, draft: str) -> str:
    """Build the Phase-2 refine prompt = region prompt + draft + directive.

    The local region prompt (a ``SYSTEM: ...\\nUSER: <json>`` string) is
    kept intact so the endpoint runtime's :func:`split_specialist_prompt`
    still pulls the specialist role; the local DRAFT and the refine
    directive are appended to the USER turn so the 70B sees both the region
    spec and the fragment it must improve."""
    return (
        f"{region_prompt}\n\n"
        f"DRAFT_FRAGMENT:\n{draft}\n\n"
        f"{_REFINE_DIRECTIVE}"
    )


@dataclass
class _RegionJob:
    """Pre-computed per-region generation job.

    Built ONCE up-front (including the token/long-table skip guards) so the
    two-phase driver can batch by adapter without re-deriving prompts or
    re-running the guards. ``skip_reason`` is set when the region must NOT
    be generated (emits an empty Candidate, same as the legacy path)."""

    idx: int
    adapter: AdapterID
    request: Any  # SpecialistRequest with request_id set
    defaults: dict[str, Any]
    prompt: str
    skip_reason: str | None = None
    skip_meta: dict[str, Any] = None  # type: ignore[assignment]
    # Phase-1 local draft (one per candidate slot); filled when Phase 1 runs.
    drafts: list[str] = None  # type: ignore[assignment]


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


def _compute_skip_guard(
    *,
    idx: int,
    adapter: AdapterID,
    request: Any,
    defaults: dict[str, Any],
    prompt: str,
    rt: Any,
) -> tuple[str | None, dict[str, Any]]:
    """Return ``(skip_reason, skip_meta)`` for a region, or ``(None, {})``.

    Two-layer context-overflow guard (identical logic to the legacy
    per-region loop, lifted verbatim so behaviour is byte-stable):

    Layer 1 — build-time char proxy: the table builder flags long tables.
    Layer 2 — runtime tokenizer count: tokenize the wrapped prompt and skip
    if prompt + max_new_tokens would exceed n_ctx. Only runs when the
    runtime exposes ``_ensure_tokenizer`` (LlamaCppRuntime); MockRuntime /
    endpoint runtimes have no local n_ctx so the check is skipped."""
    _md = (request.sampling_overrides or {}).get("metadata") or {}
    skip_reason: str | None = None
    skip_meta: dict[str, Any] = {}

    if _md.get("skip_qwen_long_table"):
        skip_reason = "long_table_char_proxy"
        skip_meta["prompt_char_len"] = _md.get("prompt_char_len")

    if skip_reason is None and hasattr(rt, "_ensure_tokenizer"):
        try:
            from .chat_format import wrap_for_qwen  # noqa: WPS433

            if prompt is not None:
                _tok = rt._ensure_tokenizer()
                _wrapped = wrap_for_qwen(_tok, prompt, add_generation_prompt=True)
                _n_prompt = len(_tok.encode(_wrapped, add_special_tokens=False))
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
                "Stage6 token-guard failed for r%d (%s): %s — deferring to "
                "llama-cpp's own n_ctx check",
                idx,
                adapter,
                exc,
            )
    return skip_reason, skip_meta


def _skip_candidate(job: "_RegionJob", seed: int | None) -> list[Candidate]:
    """Build the single empty Candidate emitted for a skipped region.

    Identical shape to the legacy skip path so Stage 7 drops it and the
    pass_9a per-kind fallback fires."""
    return [
        Candidate(
            adapter=job.request.adapter,
            request_id=job.request.request_id,
            text="",
            score=0.0,
            sampling_seed=seed,
            finish_reason="skip_long_table",
            raw_metadata={"skip_reason": job.skip_reason, **(job.skip_meta or {})},
        )
    ]


def _wrap_candidates(
    job: "_RegionJob",
    texts: list[str],
    *,
    seed: int | None,
    adapter_version: str,
    phase: str,
    base_idx: int = 0,
) -> list[Candidate]:
    """Wrap raw completion strings into typed Candidates for a region.

    Mirrors :func:`base.generate`'s folding (builder metadata + per-
    candidate index) plus the runner's adapter-version tag, so the
    per-region completion shape the assembler consumes is IDENTICAL no
    matter which phase produced ``texts``. ``phase`` is recorded on
    ``raw_metadata`` for provenance only (assembler ignores it).

    ``base_idx`` offsets ``candidate_idx`` so a caller appending one slot
    at a time (the Phase-2 per-slot endpoint path) can keep the indices
    monotonic across calls."""
    builder_metadata = (job.request.sampling_overrides or {}).get("metadata") or {}
    out: list[Candidate] = []
    for i, text in enumerate(texts):
        cand_idx = base_idx + i
        out.append(
            Candidate(
                adapter=job.request.adapter,
                request_id=job.request.request_id,
                text=text,
                sampling_seed=(seed + i) if seed is not None else None,
                finish_reason="stop",
                raw_metadata={
                    **builder_metadata,
                    "candidate_idx": cand_idx,
                    "adapter_version": adapter_version,
                    "stage6_phase": phase,
                },
            )
        )
    return out


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

    # Resolve the runtime once: a single instance threads through both
    # phases. For provider=local this is the LlamaCppRuntime; for an
    # endpoint provider make_runtime("real") short-circuits to the
    # OpenAICompatibleRuntime (see runtime.make_runtime).
    rt = runtime if runtime is not None else make_runtime(runtime_mode)

    # Provider/mode routing: decide which phase set runs.
    #   provider_is_endpoint == False -> Phase 1 only (local, batched).
    #   provider_is_endpoint == True  -> Phase 2 over all regions; Phase 1
    #                                    runs FIRST only when REFINE is set
    #                                    (hybrid: local drafts -> 70B refine).
    provider_is_endpoint = specialist_provider_is_endpoint()
    refine = resolve_refine_mode()
    run_phase1 = (not provider_is_endpoint) or refine
    run_phase2 = provider_is_endpoint

    def _defaults_for(adapter: AdapterID) -> dict[str, Any]:
        return sampling.get(
            adapter.value if isinstance(adapter, AdapterID) else str(adapter),
            {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": 512,
                "repetition_penalty": 1.0,
            },
        )

    # ------------------------------------------------------------------
    # Build per-region jobs up-front, bucketed by adapter. The skip-guard
    # (long-table char proxy + tokenizer n_ctx check) is computed ONCE per
    # region here — identical logic to the legacy per-region loop (lifted
    # into _compute_skip_guard), so behaviour is byte-stable.
    # ------------------------------------------------------------------
    buckets: dict[AdapterID, list[_RegionJob]] = {a: [] for a in _PASS_ORDER}
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
        request = _build_request(adapter, region, feature_blocks)
        # request_id labels candidates back to their region index so
        # downstream stages can trace candidate -> region.
        request_with_id = request.__class__(
            adapter=request.adapter,
            payload=request.payload,
            request_id=f"r{idx}",
            sampling_overrides=request.sampling_overrides,
        )
        prompt = (
            request_with_id.payload.get("prompt")
            if isinstance(request_with_id.payload, dict)
            else ""
        ) or ""
        defaults = _defaults_for(adapter)
        skip_reason, skip_meta = _compute_skip_guard(
            idx=idx,
            adapter=adapter,
            request=request_with_id,
            defaults=defaults,
            prompt=prompt,
            rt=rt,
        )
        buckets[adapter].append(
            _RegionJob(
                idx=idx,
                adapter=adapter,
                request=request_with_id,
                defaults=defaults,
                prompt=prompt,
                skip_reason=skip_reason,
                skip_meta=skip_meta,
            )
        )

    out: dict[int, list[Candidate]] = {}

    # ------------------------------------------------------------------
    # Phase 1 — local drafts, batched BY ADAPTER (the swap-thrash fix).
    # For each adapter group: ONE AdapterSwap load, then generate_batch
    # over the whole group, run K times (one batched pass per candidate
    # slot) so every region still gets K distinct drafts. The math adapter
    # loads ONCE for all math regions, etc.
    #
    # When provider is an endpoint AND refine is off, Phase 1 is SKIPPED
    # (run_phase1 == False) — there are no local adapters to run.
    # When refine is on, the drafts are stashed on each job for Phase 2.
    # ------------------------------------------------------------------
    if run_phase1:
        version = getattr(rt, "_adapter_version", "unknown")
        for adapter in _PASS_ORDER:
            bucket = buckets.get(adapter, [])
            if not bucket:
                logger.debug("Stage6 phase1 %s: no regions, skipping", adapter)
                continue
            # Active (non-skipped) jobs are the ones we batch-generate for.
            active = [j for j in bucket if j.skip_reason is None]
            logger.info(
                "Stage6 phase1 %s: %d region(s) (%d active, %d skipped) "
                "-> 1 adapter load",
                adapter,
                len(bucket),
                len(active),
                len(bucket) - len(active),
            )
            defaults = _defaults_for(adapter)
            # ONE swap (one load/free) for the whole adapter group.
            with AdapterSwap(adapter, runtime=rt, config_path=config_path):
                version = getattr(rt, "_adapter_version", version)
                # K batched passes — slot j across all active regions.
                # per_slot[k_idx] is a list aligned to ``active``.
                per_slot: list[list[str]] = []
                if active:
                    prompts = [j.prompt for j in active]
                    for k_idx in range(k):
                        slot_seed = None if seed is None else int(seed + k_idx)
                        texts = rt.generate_batch(
                            prompts,
                            max_tokens=defaults["max_new_tokens"],
                            temperature=defaults["temperature"],
                            top_p=defaults["top_p"],
                            seed=slot_seed,
                            repeat_penalty=defaults.get("repetition_penalty", 1.0),
                        )
                        if len(texts) != len(prompts):
                            raise RuntimeError(
                                f"Stage6 phase1 {adapter}: generate_batch "
                                f"returned {len(texts)} != {len(prompts)} prompts"
                            )
                        per_slot.append(texts)
            # Re-assemble per-region: drafts[region] = [slot0, slot1, ...].
            for a_pos, job in enumerate(active):
                job.drafts = [per_slot[k_idx][a_pos] for k_idx in range(k)]

            if not run_phase2:
                # Local-only: Phase 1 produces the final candidates.
                for job in bucket:
                    if job.skip_reason is not None:
                        logger.info(
                            "Stage6 skip r%d (%s): %s %s",
                            job.idx,
                            adapter,
                            job.skip_reason,
                            job.skip_meta,
                        )
                        out[job.idx] = _skip_candidate(job, seed)
                    else:
                        out[job.idx] = _wrap_candidates(
                            job,
                            job.drafts or [],
                            seed=seed,
                            adapter_version=version,
                            phase="local",
                        )

    # ------------------------------------------------------------------
    # Phase 2 — 70B refine/generate, batched CONCURRENTLY over ALL regions.
    # The endpoint runtime's generate_batch fans the per-region POSTs out
    # through a ThreadPoolExecutor (SEMANTIK_SPECIALIST_CONCURRENCY) and
    # re-orders results to inputs. Run K times for K candidates per region.
    #   - no-refine: prompt is the region prompt (endpoint generates fresh).
    #   - refine:    prompt is (region prompt + Phase-1 draft + directive).
    # ------------------------------------------------------------------
    if run_phase2:
        # Flatten all active jobs across adapters; skipped regions emit the
        # empty Candidate and never hit the endpoint.
        all_jobs: list[_RegionJob] = [
            j for adapter in _PASS_ORDER for j in buckets.get(adapter, [])
        ]
        active = [j for j in all_jobs if j.skip_reason is None]
        for job in all_jobs:
            if job.skip_reason is not None:
                logger.info(
                    "Stage6 skip r%d (%s): %s %s",
                    job.idx,
                    job.adapter,
                    job.skip_reason,
                    job.skip_meta,
                )
                out[job.idx] = _skip_candidate(job, seed)

        if active:
            # Phase 2 uses one representative sampling profile (prose
            # defaults) for the shared endpoint; per-region max_new_tokens
            # stays per-adapter so long tables still get headroom.
            logger.info(
                "Stage6 phase2 (endpoint%s): %d region(s) -> concurrent batch x%d",
                " refine" if refine else "",
                len(active),
                k,
            )
            # Per-region prompt: refine appends the chosen draft slot.
            for k_idx in range(k):
                slot_seed = None if seed is None else int(seed + k_idx)
                if refine:
                    prompts = [
                        _refine_prompt(
                            j.prompt,
                            (j.drafts[k_idx] if j.drafts and k_idx < len(j.drafts) else ""),
                        )
                        for j in active
                    ]
                else:
                    prompts = [j.prompt for j in active]
                # Use each job's own max_new_tokens? generate_batch takes one
                # max_tokens for the whole call — use the max across the
                # active set so no region is truncated below its budget.
                batch_max = max(j.defaults["max_new_tokens"] for j in active)
                batch_temp = active[0].defaults["temperature"]
                batch_top_p = active[0].defaults["top_p"]
                batch_rp = active[0].defaults.get("repetition_penalty", 1.0)
                texts = rt.generate_batch(
                    prompts,
                    max_tokens=batch_max,
                    temperature=batch_temp,
                    top_p=batch_top_p,
                    seed=slot_seed,
                    repeat_penalty=batch_rp,
                )
                if len(texts) != len(active):
                    raise RuntimeError(
                        f"Stage6 phase2: generate_batch returned "
                        f"{len(texts)} != {len(active)} prompts"
                    )
                for a_pos, job in enumerate(active):
                    bucket_slot = out.setdefault(job.idx, [])
                    # Build/extend the candidate list slot-by-slot.
                    cand = _wrap_candidates(
                        job,
                        [texts[a_pos]],
                        seed=slot_seed,
                        adapter_version="endpoint",
                        phase="refine" if refine else "endpoint",
                        base_idx=k_idx,
                    )[0]
                    bucket_slot.append(cand)

    return out


__all__ = [
    "resolve_refine_mode",
    "run_qwen_specialists",
]
