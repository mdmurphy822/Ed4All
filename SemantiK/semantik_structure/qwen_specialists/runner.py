"""Stage 6 driver — BATCHED two-phase region → K-candidate generator.

The driver buckets regions by their routed adapter, then runs up to two
phases — replacing the old per-region swap loop. BOTH phases are
**model-agnostic**: each independently resolves its own {provider, model,
base_url, api_key} seat from the registry/env, so any composition works
(local-draft→local-large-refine fully on-device, local-draft→hosted-refine,
hosted-draft→hosted-refine, single-phase-any-provider, ...). The hosted 70B is
just ONE possible Phase-2 model, never a hardcoded tier.

    Phase 1 (draft, batched BY ADAPTER)
        Runs on the Phase-1 seat (default: the local GGUF specialists —
        :func:`~.runtime.resolve_phase_provider` ``"phase1"`` defaults to
        ``"local"``). For each adapter group, ONE :class:`AdapterSwap` load,
        then a single :meth:`QwenRuntime.generate_batch` over every region in
        that group (run K times, one per candidate slot). The math adapter
        loads ONCE for all math regions, etc. — the swap-thrash fix. SKIPPED
        entirely when Phase 2 is the sole authoring tier (DISPLACE).

    Phase 2 (refine / generate)
        Runs on the Phase-2 seat (default: the single ``SEMANTIK_SPECIALIST_*``
        seat). Enabled by ``SEMANTIK_SPECIALIST_REFINE`` (hybrid) or
        ``SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE`` (pure Phase-2). The Phase-2
        seat may be local OR hosted — whatever the Phase-2 provider resolves
        to (the refiner model, not a hardcoded 70B):
          * neither opt-in (default): Phase 1 ONLY. Byte-stable + swap-thrash
            win.
          * DISPLACE: SKIP Phase 1; build per-region prompts; one batched pass
            over ALL regions on the Phase-2 seat.
          * REFINE (hybrid): Phase 1 drafts on the Phase-1 seat THEN Phase 2
            sends each region's (prompt + draft) to the Phase-2 seat with a
            refine directive; the Phase-2 output REPLACES the draft.

Entry point::

    from semantik_structure.qwen_specialists.runner import run_qwen_specialists

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
from .endpoint_runtime import (
    EndpointRuntimeError,
    resolve_specialist_batch_regions,
    resolve_specialist_concurrency,
)
from .prompts import build_math_request, build_prose_request, build_table_request
from .routing import adapter_for
from ..stop_seam import StopPoller
from .runtime import (
    QwenRuntime,
    make_phase_runtime,
    resolve_endpoint_displaces_local,
)
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

# Directive prepended to a region's prompt when the Phase-2 seat is asked to
# REFINE an existing Phase-1 draft (the hybrid flow). Keeps the same bare-
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


# Truthy strings for SEMANTIK_STAGE6_PROSE_PASSTHROUGH (parse-with-fallback;
# any other value is falsey/off — mirrors _REFINE_TRUTHY).
_PROSE_PASSTHROUGH_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_stage6_prose_passthrough() -> bool:
    """True when ``SEMANTIK_STAGE6_PROSE_PASSTHROUGH`` skips GGUF generation
    for AdapterID.PROSE regions in favour of a deterministic verbatim
    passthrough candidate.

    **Default OFF — byte-stable.** When off, every prose region still runs
    the normal Stage-6 generation path. When on, regions routed to
    :attr:`AdapterID.PROSE` receive a single deterministic passthrough
    Candidate (the region's verbatim text rendered by the canonical
    per-kind :func:`~semantik_structure.assembler.fallbacks.emit_fallback`) and no
    GGUF completion is generated for them; TABLE and MATH regions are
    UNAFFECTED (still generate). This does NOT touch ``runtime_mode`` — a
    real run stays ``runtime_mode='real'`` (the R4 mock-trap is not tripped).

    Parse-with-fallback: only the truthy set enables it; anything else
    (unset / blank / garbage) is off."""
    raw = (
        os.environ.get("SEMANTIK_STAGE6_PROSE_PASSTHROUGH") or ""
    ).strip().lower()
    return raw in _PROSE_PASSTHROUGH_TRUTHY


# Falsey strings that disable the (default-ON) endpoint batched lane.
_BATCH_FALSEY = frozenset({"0", "false", "no", "off"})

# Default candidate-K ceiling on the BATCHED endpoint lane. The prose
# config default is K=4; running 4 candidate slots multiplies POSTs by 4.
# Capping the batched endpoint lane to 2 keeps total POSTs ~34 for a 16-page
# slice while still giving the soft reranker two candidates to choose from.
_DEFAULT_BATCHED_ENDPOINT_K = 2

# Default concurrency for the BATCHED endpoint lane. The per-region path used
# 8; with batching each POST already carries ~12 regions, so a LOW fan-out of
# batches keeps us well under the ~40-req/min cap. SEMANTIK_SPECIALIST_CONCURRENCY
# (if explicitly set) overrides this.
_DEFAULT_BATCHED_CONCURRENCY = 2

# Per-batch summed-output-token cap for _pack_batches. A bucket of big-output
# regions (tables) starts a new batch sooner so no single batched POST asks
# for more than the model can emit. 16384 matches the runtime's ceiling; the
# runtime additionally applies its own hard cap + safety margin.
_DEFAULT_OUTPUT_TOKEN_CAP = 16384


def resolve_batch_mode() -> bool:
    """True when the endpoint lane should use the BATCHED multi-region path.

    Gate ``SEMANTIK_SPECIALIST_BATCH``. **Default ON for the endpoint lane**
    (the per-region path 429s against the ~40-req/min cap). Parse-with-
    fallback, default-ON semantics: an explicit falsey value
    (``0``/``false``/``no``/``off``) selects the EXACT legacy per-region
    ``generate_batch`` path (byte-stable); unset / truthy / garbage → on.

    Only consulted on the endpoint lane (Phase 2). The local Phase-1 path is
    unaffected — it always batches by adapter regardless of this flag."""
    raw = (os.environ.get("SEMANTIK_SPECIALIST_BATCH") or "").strip().lower()
    return raw not in _BATCH_FALSEY


def resolve_batched_endpoint_k(requested_k: int) -> int:
    """Resolve the candidate-K used on the BATCHED endpoint lane.

    ``SEMANTIK_SPECIALIST_BATCH_K`` (positive int) pins the cap explicitly;
    otherwise the caller's ``requested_k`` is clamped down to
    ``_DEFAULT_BATCHED_ENDPOINT_K`` (2) so the batched lane does not multiply
    POSTs by the prose K=4. Never raises K above what the caller asked for
    (a caller passing k=1 keeps k=1). Parse-with-fallback."""
    raw = os.environ.get("SEMANTIK_SPECIALIST_BATCH_K")
    if raw:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = 0
        if val > 0:
            return min(requested_k, val)
    return min(requested_k, _DEFAULT_BATCHED_ENDPOINT_K)


def resolve_batched_concurrency() -> int:
    """Thread-pool size for dispatching BATCHES on the endpoint lane.

    An explicitly-set ``SEMANTIK_SPECIALIST_CONCURRENCY`` is honoured as an
    override; otherwise the batched lane uses a LOW default
    (``_DEFAULT_BATCHED_CONCURRENCY`` = 2) because each batch already packs
    many regions — a high fan-out would re-create the rate-limit pressure
    batching exists to remove."""
    if os.environ.get("SEMANTIK_SPECIALIST_CONCURRENCY"):
        return resolve_specialist_concurrency()
    return _DEFAULT_BATCHED_CONCURRENCY


def _pack_batches(
    active: list["_RegionJob"],
    *,
    max_regions: int,
    output_token_cap: int,
) -> list[list["_RegionJob"]]:
    """Pack a bucket's active jobs (IN ORDER) into batches.

    A new batch is started when EITHER the current batch reaches
    ``max_regions`` jobs OR adding the next job would push the batch's summed
    ``defaults["max_new_tokens"]`` over ``output_token_cap`` — so a bucket of
    big-output regions (tables) automatically gets smaller batches and never
    asks one POST for more output than the model can emit. A single job whose
    own budget already exceeds the cap still gets its own (one-job) batch
    rather than being dropped. Order within and across batches is preserved
    so the region->job mapping by ``r{idx}`` stays exact."""
    batches: list[list[_RegionJob]] = []
    current: list[_RegionJob] = []
    current_tokens = 0
    cap = max(1, int(max_regions))
    tok_cap = max(1, int(output_token_cap))
    for job in active:
        job_tokens = int(job.defaults.get("max_new_tokens", 512))
        would_overflow_tokens = current and (current_tokens + job_tokens > tok_cap)
        would_overflow_count = len(current) >= cap
        if would_overflow_count or would_overflow_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(job)
        current_tokens += job_tokens
    if current:
        batches.append(current)
    return batches


def _refine_prompt(region_prompt: str, draft: str) -> str:
    """Build the Phase-2 refine prompt = region prompt + draft + directive.

    The local region prompt (a ``SYSTEM: ...\\nUSER: <json>`` string) is
    kept intact so the endpoint runtime's :func:`split_specialist_prompt`
    still pulls the specialist role; the local DRAFT and the refine
    directive are appended to the USER turn so the Phase-2 refiner sees both
    the region spec and the fragment it must improve."""
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
    # Deterministic verbatim passthrough HTML (SEMANTIK_STAGE6_PROSE_PASSTHROUGH
    # only). Pre-rendered at build time via emit_fallback for a PROSE region so
    # the region skips GGUF generation entirely; None on every other job.
    passthrough_html: str | None = None


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


def _render_passthrough_html(region: Any, feature_blocks: Sequence[Any]) -> str:
    """Deterministic verbatim HTML for a PROSE region (no GGUF generation).

    Reuses the canonical per-kind :func:`emit_fallback` — the SAME emitter
    the Stage-9 assembler uses when a region's candidates all die at the
    gate — so the passthrough is a small, valid HTML5 fragment carrying the
    region's verbatim source text in exactly the shape downstream Stage-7
    (html5 + text_preserve) and Stage-9 assembly expect (heading → ``<h2>``,
    list → ``<ul><li>``, code_block → ``<pre><code>``, metadata_drop → ``""``,
    …). Lazy import keeps the assembler package off the runner's import path
    for the flag-off default."""
    from ..assembler.fallbacks import emit_fallback  # noqa: WPS433 (lazy)

    return emit_fallback(region, feature_blocks)


def _passthrough_candidates(job: "_RegionJob", seed: int | None) -> list[Candidate]:
    """Build the single deterministic passthrough Candidate (k=1) for a PROSE
    region under ``SEMANTIK_STAGE6_PROSE_PASSTHROUGH``.

    Mirrors :func:`_wrap_candidates`' shape (builder metadata + candidate
    index) so the per-region completion the assembler consumes is identical
    to a generated one, except the text is the pre-rendered verbatim
    :attr:`_RegionJob.passthrough_html` and ``adapter_version`` /
    ``stage6_phase`` are stamped ``"passthrough"`` for provenance/audit."""
    meta = {
        **((job.request.sampling_overrides or {}).get("metadata") or {}),
        "candidate_idx": 0,
        "adapter_version": "passthrough",
        "stage6_phase": "passthrough",
        "prose_passthrough": True,
    }
    return [
        Candidate(
            adapter=job.request.adapter,
            request_id=job.request.request_id,
            text=job.passthrough_html or "",
            sampling_seed=seed,
            finish_reason="stop",
            raw_metadata=meta,
        )
    ]


# Reason code stamped on a region's candidate/metadata when its Phase-2
# endpoint call failed (after retries) and we degraded instead of aborting
# the whole document. Visible in provenance/audit via raw_metadata.
_ENDPOINT_DEGRADED_MARKER = "endpoint_degraded"


def _degraded_skip_candidate(
    job: "_RegionJob", seed: int | None, *, reason: str
) -> list[Candidate]:
    """Build the single empty Candidate for a region whose endpoint call
    failed in PURE-ENDPOINT mode (no local draft to fall back on).

    Mirrors :func:`_skip_candidate`'s empty-Candidate shape (so Stage 7
    drops it and pass_9a per-kind fallback fires) but stamps the
    ``endpoint_degraded`` marker + the failure reason so the degradation is
    auditable rather than silent."""
    return [
        Candidate(
            adapter=job.request.adapter,
            request_id=job.request.request_id,
            text="",
            score=0.0,
            sampling_seed=seed,
            finish_reason=_ENDPOINT_DEGRADED_MARKER,
            raw_metadata={
                "skip_reason": _ENDPOINT_DEGRADED_MARKER,
                "endpoint_degraded": True,
                "degraded_reason": reason,
            },
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
    degraded_reason: str | None = None,
) -> list[Candidate]:
    """Wrap raw completion strings into typed Candidates for a region.

    Mirrors :func:`base.generate`'s folding (builder metadata + per-
    candidate index) plus the runner's adapter-version tag, so the
    per-region completion shape the assembler consumes is IDENTICAL no
    matter which phase produced ``texts``. ``phase`` is recorded on
    ``raw_metadata`` for provenance only (assembler ignores it).

    ``base_idx`` offsets ``candidate_idx`` so a caller appending one slot
    at a time (the Phase-2 per-slot endpoint path) can keep the indices
    monotonic across calls.

    ``degraded_reason`` (hybrid fail-soft path): when set, the candidate is
    a KEPT Phase-1 local draft standing in for a failed endpoint refine —
    stamp ``endpoint_degraded`` + the reason so the degradation is
    auditable (the text is still real content, not a skip)."""
    builder_metadata = (job.request.sampling_overrides or {}).get("metadata") or {}
    out: list[Candidate] = []
    for i, text in enumerate(texts):
        cand_idx = base_idx + i
        meta = {
            **builder_metadata,
            "candidate_idx": cand_idx,
            "adapter_version": adapter_version,
            "stage6_phase": phase,
        }
        if degraded_reason is not None:
            meta["endpoint_degraded"] = True
            meta["degraded_reason"] = degraded_reason
        out.append(
            Candidate(
                adapter=job.request.adapter,
                request_id=job.request.request_id,
                text=text,
                sampling_seed=(seed + i) if seed is not None else None,
                finish_reason="stop",
                raw_metadata=meta,
            )
        )
    return out


def _phase2_slot_texts_per_region(
    rt: Any,
    active: list["_RegionJob"],
    *,
    k_idx: int,
    slot_seed: int | None,
    refine: bool,
) -> list[str | None]:
    """Produce one slot's completions for the per-region (legacy) endpoint
    path: ONE ``generate_batch`` over ALL active jobs, returned aligned to
    ``active`` order (None sentinels passed through).

    This is the EXACT byte-stable pre-batching dispatch — selected when
    ``SEMANTIK_SPECIALIST_BATCH`` is falsey."""
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
        fail_soft=True,
    )
    if len(texts) != len(active):
        raise RuntimeError(
            f"Stage6 phase2: generate_batch returned "
            f"{len(texts)} != {len(active)} prompts"
        )
    return list(texts)


def _phase2_slot_texts_batched(
    rt: Any,
    active: list["_RegionJob"],
    *,
    k_idx: int,
    slot_seed: int | None,
    refine: bool,
) -> list[str | None]:
    """Produce one slot's completions for the BATCHED endpoint path.

    Packs the active jobs into batches (``_pack_batches``) and dispatches the
    BATCHES (not regions) through a low-concurrency ThreadPoolExecutor, each
    via :meth:`OpenAICompatibleRuntime.generate_multi` — ONE POST per batch
    (the rate-limit defeat). Each ``{r{idx}: fragment}`` result is mapped back
    to its job by id; a region missing from its batch (or whose whole batch
    POST failed) stays the None sentinel. Returns a ``list[str | None]``
    aligned to ``active`` order so the downstream degradation mapping is
    IDENTICAL to the per-region path."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: WPS433

    # Capability fallback: a runtime that does NOT expose generate_multi
    # (an older mock, or a non-endpoint runtime that somehow reaches Phase 2)
    # cannot be batched — fall back to the per-region generate_batch path so
    # it still produces real candidates rather than degrading to all-None.
    if not callable(getattr(rt, "generate_multi", None)):
        return _phase2_slot_texts_per_region(
            rt, active, k_idx=k_idx, slot_seed=slot_seed, refine=refine
        )

    max_regions = resolve_specialist_batch_regions()
    batches = _pack_batches(
        active,
        max_regions=max_regions,
        output_token_cap=_DEFAULT_OUTPUT_TOKEN_CAP,
    )
    # Map each region id -> its index in ``active`` so results land in order.
    id_to_pos = {f"r{job.idx}": a_pos for a_pos, job in enumerate(active)}
    texts: list[str | None] = [None] * len(active)

    def _run_batch(batch: list["_RegionJob"]) -> dict[str, str | None]:
        items: list[tuple[str, str]] = [(f"r{j.idx}", j.prompt) for j in batch]
        drafts: dict[str, str] | None = None
        if refine:
            drafts = {
                f"r{j.idx}": (
                    j.drafts[k_idx] if j.drafts and k_idx < len(j.drafts) else ""
                )
                for j in batch
            }
        # Summed per-region budget for this batch (the runtime caps it at the
        # model ceiling minus the safety margin).
        batch_max = sum(int(j.defaults.get("max_new_tokens", 512)) for j in batch)
        batch_temp = batch[0].defaults["temperature"]
        batch_top_p = batch[0].defaults["top_p"]
        batch_rp = batch[0].defaults.get("repetition_penalty", 1.0)
        return rt.generate_multi(
            items,
            max_tokens=batch_max,
            temperature=batch_temp,
            top_p=batch_top_p,
            seed=slot_seed,
            repeat_penalty=batch_rp,
            drafts=drafts,
        )

    if not batches:
        return texts
    max_workers = max(1, min(resolve_batched_concurrency(), len(batches)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_batch, b): b for b in batches}
        for future in futures:
            batch = futures[future]
            try:
                mapping = future.result()
            except Exception as exc:  # noqa: BLE001
                # A batch dispatch that raises (generate_multi already fails
                # soft internally, so this is defensive) degrades every region
                # in that batch to the None sentinel — never aborts the slot.
                logger.warning(
                    "Stage6 phase2 batched POST raised (%d regions): %s "
                    "— degrading those regions to None",
                    len(batch),
                    exc,
                )
                mapping = {f"r{j.idx}": None for j in batch}
            for rid, fragment in mapping.items():
                pos = id_to_pos.get(rid)
                if pos is not None:
                    texts[pos] = fragment
    return texts


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

    # GRACEFUL-STOP SEAM (c) — Stage-6 adapter-batch boundary. The Ed4All side
    # may hand in a SEMANTIK_STOP_SENTINEL (declared-env plumbing, cross-venv
    # twin of the GPU-lifecycle lease). This throttled poller is checked BETWEEN
    # adapter batches (Phase 1) and between candidate-slot passes (Phase 2), so
    # a stop requested mid-Stage-6 (the ~3h/chapter local-7B authoring tier)
    # raises CascadeStopRequested at the next batch boundary instead of running
    # the whole document to completion. No-op / fail-soft when no sentinel path
    # was provided; never crashes a run that passes none.
    stop_poller = StopPoller()

    sampling = _load_sampling(config_path, lane=lane)

    # Phase routing: decide which phase set runs.
    #
    # Phase 2 (refine/generate) is enabled by an EXPLICIT opt-in — REFINE
    # (hybrid: Phase-1 drafts THEN Phase-2 refine) or DISPLACE (Phase 2 is the
    # sole authoring tier). Without either, Phase 1 is the only tier (the
    # dereliction default — the license-clean Phase-1 seat authors). This is
    # decoupled from the provider: Phase 2 runs whenever enabled regardless of
    # whether its seat is local or hosted (the model-agnostic generalization).
    #
    # Truth table (the seats are resolved per-phase below):
    #   neither REFINE nor DISPLACE -> Phase 1 only.
    #   REFINE                      -> Phase 1 + Phase 2 (hybrid).
    #   DISPLACE                    -> Phase 2 only.
    refine = resolve_refine_mode()
    displace = resolve_endpoint_displaces_local()
    # PROSE passthrough (default off): skip GGUF generation for AdapterID.PROSE
    # regions and emit a deterministic verbatim candidate instead. Resolved once
    # here; consumed at build time (pre-render) and drained before Phase 1/2.
    # Does NOT touch runtime_mode (R4 mock-trap unaffected).
    prose_passthrough = resolve_stage6_prose_passthrough()
    phase2_active = refine or displace
    run_phase1 = (not phase2_active) or refine
    run_phase2 = phase2_active

    # Resolve a runtime PER PHASE from its own seat (model-agnostic). When the
    # caller injects an explicit ``runtime`` it is used for BOTH phases (the
    # byte-stable test/back-compat path). Otherwise Phase 1 builds from the
    # Phase-1 seat (default local GGUF specialists) and Phase 2 from the
    # Phase-2 seat (default the single ``SEMANTIK_SPECIALIST_*`` seat) — each
    # may independently be local or hosted, so e.g. local-draft→local-large-
    # refine makes zero hosted calls. Only the phases that run are built.
    if runtime is not None:
        rt_phase1: QwenRuntime | None = runtime
        rt_phase2: QwenRuntime | None = runtime
    else:
        rt_phase1 = (
            make_phase_runtime(runtime_mode, "phase1", config_path=config_path)
            if run_phase1
            else None
        )
        rt_phase2 = (
            make_phase_runtime(runtime_mode, "phase2", config_path=config_path)
            if run_phase2
            else None
        )
    # The skip-guard's local-n_ctx tokenizer check only applies to a local
    # LlamaCppRuntime; the relevant runtime is Phase 1's when it runs (Phase 2
    # endpoint runtimes have no _ensure_tokenizer so the check no-ops).
    guard_rt = rt_phase1 if rt_phase1 is not None else rt_phase2

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
        # PROSE passthrough: pre-render the verbatim HTML and SKIP the
        # (generation-only) context-overflow guard — there is no prompt to
        # overflow. The region is drained straight to a deterministic
        # candidate below. TABLE / MATH run the guard + generate normally.
        is_prose_passthrough = prose_passthrough and adapter is AdapterID.PROSE
        if is_prose_passthrough:
            skip_reason, skip_meta = None, {}
            passthrough_html: str | None = _render_passthrough_html(
                region, feature_blocks
            )
        else:
            skip_reason, skip_meta = _compute_skip_guard(
                idx=idx,
                adapter=adapter,
                request=request_with_id,
                defaults=defaults,
                prompt=prompt,
                rt=guard_rt,
            )
            passthrough_html = None
        buckets[adapter].append(
            _RegionJob(
                idx=idx,
                adapter=adapter,
                request=request_with_id,
                defaults=defaults,
                prompt=prompt,
                skip_reason=skip_reason,
                skip_meta=skip_meta,
                passthrough_html=passthrough_html,
            )
        )

    out: dict[int, list[Candidate]] = {}

    # ------------------------------------------------------------------
    # PROSE passthrough (SEMANTIK_STAGE6_PROSE_PASSTHROUGH, default off).
    # Drain the whole PROSE bucket into deterministic verbatim candidates and
    # CLEAR it so neither Phase 1 nor Phase 2 generates for prose regions —
    # the ~3h/chapter GGUF prose-generation cost is removed. TABLE + MATH are
    # untouched (still generate). The passthrough candidate == emit_fallback,
    # the same deterministic HTML the Stage-9 assembler emits for a prose
    # region whose candidates all die at the gate, so Stage 7/9 assembly is
    # well-formed and token-conservation holds.
    # ------------------------------------------------------------------
    if prose_passthrough:
        prose_bucket = buckets.get(AdapterID.PROSE, [])
        n_prose_pass = len(prose_bucket)
        for job in prose_bucket:
            out[job.idx] = _passthrough_candidates(job, seed)
        buckets[AdapterID.PROSE] = []
        n_gen = sum(
            len(buckets.get(a, [])) for a in _PASS_ORDER if a is not AdapterID.PROSE
        )
        logger.info(
            "Stage6 prose passthrough (SEMANTIK_STAGE6_PROSE_PASSTHROUGH on): "
            "%d prose region(s) passthrough'd (0 GGUF-generated), "
            "%d table/math region(s) still generate",
            n_prose_pass,
            n_gen,
        )

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
        version = getattr(rt_phase1, "_adapter_version", "unknown")
        for adapter in _PASS_ORDER:
            # Seam (c): between adapter groups — the coarsest Stage-6 batch
            # boundary. A stop requested after (e.g.) the PROSE group finishes
            # raises before the TABLE/MATH AdapterSwap loads. No-op when unset.
            stop_poller.check(f"stage6:phase1:{adapter}")
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
            with AdapterSwap(adapter, runtime=rt_phase1, config_path=config_path):
                version = getattr(rt_phase1, "_adapter_version", version)
                # K batched passes — slot j across all active regions.
                # per_slot[k_idx] is a list aligned to ``active``.
                per_slot: list[list[str]] = []
                if active:
                    prompts = [j.prompt for j in active]
                    for k_idx in range(k):
                        slot_seed = None if seed is None else int(seed + k_idx)
                        # fail_soft: a per-item failure returns the None
                        # sentinel rather than aborting the whole Phase-1
                        # batch (closes the residual 429 gap on the local-
                        # draft path). Local runtimes ignore the kwarg and
                        # never fail an item; the endpoint runtime (hybrid
                        # refine mode runs Phase 1 too) degrades only that
                        # slot. A None slot becomes an empty draft below —
                        # Stage 7 then drops it, same as a skip.
                        texts = rt_phase1.generate_batch(
                            prompts,
                            max_tokens=defaults["max_new_tokens"],
                            temperature=defaults["temperature"],
                            top_p=defaults["top_p"],
                            seed=slot_seed,
                            repeat_penalty=defaults.get("repetition_penalty", 1.0),
                            fail_soft=True,
                        )
                        if len(texts) != len(prompts):
                            raise RuntimeError(
                                f"Stage6 phase1 {adapter}: generate_batch "
                                f"returned {len(texts)} != {len(prompts)} prompts"
                            )
                        # Normalize None sentinels (fail-soft) to empty drafts
                        # so the per-region re-assembly stays str-typed.
                        per_slot.append([t if t is not None else "" for t in texts])
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
    # Phase 2 — refine/generate on the Phase-2 seat (whatever provider/model
    # it resolves to — local or hosted). When that seat is a hosted endpoint,
    # generate_batch fans the per-region POSTs out through a ThreadPoolExecutor
    # (SEMANTIK_SPECIALIST_CONCURRENCY) and re-orders results to inputs; a local
    # Phase-2 seat runs the same prompts through its generate_batch. Run K times
    # for K candidates per region.
    #   - no-refine: prompt is the region prompt (Phase-2 seat generates fresh).
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
            # Batched lane (default ON for the endpoint): pack many regions
            # into few large POSTs to survive the ~40-req/min rate cap. An
            # explicit SEMANTIK_SPECIALIST_BATCH=0 selects the EXACT legacy
            # per-region generate_batch path (byte-stable). The batched lane
            # also caps candidate-K to <=2 so K does not multiply POSTs.
            batched = resolve_batch_mode()
            phase2_k = resolve_batched_endpoint_k(k) if batched else k
            # Phase 2 uses one representative sampling profile (prose
            # defaults) for the shared endpoint; per-region max_new_tokens
            # stays per-adapter so long tables still get headroom.
            logger.info(
                "Stage6 phase2 (endpoint%s, %s): %d region(s) -> x%d candidate slot(s)",
                " refine" if refine else "",
                "batched" if batched else "per-region",
                len(active),
                phase2_k,
            )
            # Per-region degradation bookkeeping: count, per a_pos, how many
            # of the K slots the endpoint FAILED (after retries). The KEY
            # FIX — a per-region endpoint failure degrades only THAT region
            # (keep its local draft in hybrid mode / emit a degraded skip in
            # pure-endpoint mode) instead of raising and aborting the whole
            # document.
            slot_failures: list[int] = [0] * len(active)
            # Per-region prompt: refine appends the chosen draft slot.
            for k_idx in range(phase2_k):
                # Seam (c): between candidate-slot passes on the Phase-2 lane
                # (each slot is one batched fan-out over all active regions). A
                # stop mid-Phase-2 raises at the next slot boundary. No-op unset.
                stop_poller.check(f"stage6:phase2:slot{k_idx}")
                slot_seed = None if seed is None else int(seed + k_idx)
                # Both dispatch paths return a list[str | None] aligned to
                # ``active`` order, so the degradation mapping below is
                # identical regardless of batching.
                if batched:
                    texts = _phase2_slot_texts_batched(
                        rt_phase2, active, k_idx=k_idx, slot_seed=slot_seed, refine=refine
                    )
                else:
                    texts = _phase2_slot_texts_per_region(
                        rt_phase2, active, k_idx=k_idx, slot_seed=slot_seed, refine=refine
                    )
                for a_pos, job in enumerate(active):
                    text = texts[a_pos]
                    if text is None:
                        # Endpoint failed this region's slot (after retries).
                        slot_failures[a_pos] += 1
                        if refine:
                            # Hybrid: keep the region's Phase-1 LOCAL draft
                            # for this slot — we already paid for it; don't
                            # throw it away. Stamp a degradation marker.
                            draft = (
                                job.drafts[k_idx]
                                if job.drafts and k_idx < len(job.drafts)
                                else ""
                            )
                            logger.warning(
                                "Stage6 phase2 r%d slot %d: endpoint refine "
                                "failed -> keeping Phase-1 local draft",
                                job.idx,
                                k_idx,
                            )
                            cand = _wrap_candidates(
                                job,
                                [draft],
                                seed=slot_seed,
                                adapter_version="local",
                                phase="refine",
                                base_idx=k_idx,
                                degraded_reason="endpoint_refine_failed_kept_local_draft",
                            )[0]
                            out.setdefault(job.idx, []).append(cand)
                        else:
                            # Pure-endpoint: no local draft. Defer the
                            # degraded skip until all slots are tried — only
                            # emit it if EVERY slot failed for this region.
                            logger.warning(
                                "Stage6 phase2 r%d slot %d: endpoint generate "
                                "failed (no local draft)",
                                job.idx,
                                k_idx,
                            )
                        continue
                    # Success: build/extend the candidate list slot-by-slot.
                    cand = _wrap_candidates(
                        job,
                        [text],
                        seed=slot_seed,
                        adapter_version="endpoint",
                        phase="refine" if refine else "endpoint",
                        base_idx=k_idx,
                    )[0]
                    out.setdefault(job.idx, []).append(cand)

            # Post-pass: pure-endpoint regions that produced NO usable
            # candidate (every slot failed) get a degraded skip candidate so
            # the document still assembles around them.
            degraded_regions = 0
            for a_pos, job in enumerate(active):
                produced = out.get(job.idx) or []
                if not produced:
                    degraded_regions += 1
                    logger.warning(
                        "Stage6 phase2 r%d (%s): all %d endpoint slot(s) "
                        "failed -> degraded skip candidate",
                        job.idx,
                        job.adapter,
                        phase2_k,
                    )
                    out[job.idx] = _degraded_skip_candidate(
                        job,
                        seed,
                        reason="endpoint_failed_all_slots",
                    )
                elif slot_failures[a_pos]:
                    # Partial degradation (kept-draft in refine, or some
                    # slots succeeded in pure-endpoint).
                    degraded_regions += 1

            # Fail-loud guard: if EVERY active region was fully degraded
            # (endpoint down / bad key) we must NOT ship an empty document —
            # only PARTIAL failures degrade-and-continue. (In hybrid/refine
            # mode a region always keeps its local draft, so it is never
            # "fully degraded" here and this guard never fires — correct: we
            # still have real local content.)
            fully_failed = [
                job
                for a_pos, job in enumerate(active)
                if slot_failures[a_pos] >= phase2_k
                and not refine
            ]
            if active and len(fully_failed) == len(active):
                raise EndpointRuntimeError(
                    "Stage6 phase2: ALL "
                    f"{len(active)} active region(s) failed the endpoint "
                    "(every slot, after retries). Refusing to ship an empty "
                    "document — this is a total endpoint failure (endpoint "
                    "down / bad key / wrong base_url), not a partial "
                    "degradation. Fail-loud."
                )

            if degraded_regions:
                logger.warning(
                    "Stage6 phase2 summary: %d/%d region(s) degraded due to "
                    "endpoint failure (the rest assembled normally).",
                    degraded_regions,
                    len(active),
                )

    return out


__all__ = [
    "resolve_batch_mode",
    "resolve_batched_endpoint_k",
    "resolve_refine_mode",
    "resolve_stage6_prose_passthrough",
    "run_qwen_specialists",
]
