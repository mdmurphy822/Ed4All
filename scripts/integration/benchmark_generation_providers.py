#!/usr/bin/env python3
"""Generation-provider A/B throughput harness (single-stream vs concurrent).

Sweeps a configurable list of ``config/endpoints.yaml`` endpoint NAMES
(``local`` / ``spark-nano`` / ``spark-super`` / any ``openai_compatible``
row) and, for each, drives a FIXED set of representative synthesis prompts
through the SAME registry-resolved client the pipeline uses — so a
benchmark against a Spark seat exercises the real code path, not a
hand-rolled HTTP client.

Why this exists: on the DGX Spark the headline metric is not single-stream
decode speed — it's how far batched concurrency scales aggregate tokens/sec
(a well-fed vLLM server on a GB10 gives 25-120x over a single stream). This
harness reports single-stream tps, aggregate tps under ``--concurrency N``,
and the derived scaling factor, per provider.

Design constraints honored:

  * Provider resolution goes through ``lib.llm.endpoints.
    build_openai_compatible_client(name)`` (the ONE registry constructor).
    No hand-rolled HTTP. Adding a provider is a registry row, never a code
    edit here.
  * Token accounting prefers the server-reported ``completion_tokens``
    (``client.last_usage``); when the server omits usage we fall back to a
    whitespace word estimate, CLEARLY labelled ``estimated`` in the report.
  * Time-to-first-token needs token streaming, which the shared
    ``OpenAICompatibleClient`` does not expose — so ``ttft_seconds`` is
    recorded as ``null`` with an explicit note rather than fabricated.
  * Graceful degrade: an unreachable endpoint (Spark server not up yet)
    records a per-provider ``error`` and the sweep CONTINUES. One dead
    provider never aborts the run.
  * This is a benchmark tool, not a pipeline LLM call site — it wires NO
    DecisionCapture (and fabricates none).

Usage:
    python scripts/integration/benchmark_generation_providers.py \
        --providers local spark-nano spark-super --concurrency 32

    # Single provider, quick single-stream-only smoke (concurrency 1):
    python scripts/integration/benchmark_generation_providers.py \
        --providers local --concurrency 1

    # Custom output + generation cap:
    python scripts/integration/benchmark_generation_providers.py \
        --providers spark-super --concurrency 64 \
        --max-tokens 512 --output /tmp/spark_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root on sys.path (scripts/integration/<this>).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Fixed, deterministic prompt set (inline — no network fixtures).
# --------------------------------------------------------------------------- #
#
# Two representative synthesis shapes:
#   (a) a SHORT Courseforge-style content-block authoring prompt, and
#   (b) a LONGER objective/outline-style prompt.
# Both are self-contained and temperature-0 so the benchmark is repeatable
# and the decode-length is roughly stable across runs.

@dataclass(frozen=True)
class BenchPrompt:
    name: str
    system: str
    user: str
    # Suggested generation cap so the two shapes decode a comparable-ish
    # number of tokens (the LONG prompt gets a bigger budget). The CLI
    # ``--max-tokens`` overrides this when set.
    default_max_tokens: int


_CONTENT_BLOCK_SYSTEM = (
    "You are an expert instructional content author. Write clear, accurate, "
    "self-contained course prose. Do not add commentary or meta-text."
)
_CONTENT_BLOCK_USER = (
    "Write a single 'explanation' content block (2-3 short paragraphs) for a "
    "college-level course page. Topic: how a hash table achieves average "
    "O(1) lookup, and what a hash collision is. Define the key term 'load "
    "factor' inline. Audience: first-year computer-science students."
)

_OUTLINE_SYSTEM = (
    "You are a curriculum designer. Produce well-structured, measurable "
    "learning objectives and a course outline. Be concrete and specific."
)
_OUTLINE_USER = (
    "For a 4-week introductory unit on relational databases, produce:\n"
    "1. Four terminal learning objectives (one per week), each written with "
    "an observable Bloom's-taxonomy verb and a measurable criterion.\n"
    "2. For each terminal objective, three supporting chapter-level "
    "objectives.\n"
    "3. A one-sentence rationale per week explaining how that week's "
    "objectives build on the prior week's.\n"
    "Cover, across the four weeks: the relational model and keys; "
    "normalization (1NF-3NF); SQL SELECT/JOIN/GROUP BY; and transactions "
    "with ACID properties. Write the objectives so an assessment author "
    "could map each to a quiz item without further clarification."
)

_PROMPTS: Tuple[BenchPrompt, ...] = (
    BenchPrompt(
        name="content_block_short",
        system=_CONTENT_BLOCK_SYSTEM,
        user=_CONTENT_BLOCK_USER,
        default_max_tokens=384,
    ),
    BenchPrompt(
        name="objective_outline_long",
        system=_OUTLINE_SYSTEM,
        user=_OUTLINE_USER,
        default_max_tokens=1024,
    ),
)


# --------------------------------------------------------------------------- #
# Result records.
# --------------------------------------------------------------------------- #

@dataclass
class SingleCallResult:
    prompt_name: str
    ok: bool
    wall_seconds: float
    completion_tokens: int
    tokens_estimated: bool
    output_snippet: str = ""
    error: Optional[str] = None


@dataclass
class ProviderReport:
    provider: str
    resolved_model: Optional[str] = None
    resolved_base_url: Optional[str] = None
    ok: bool = False
    error: Optional[str] = None
    # Single-stream metrics (mean over the prompt set).
    single_stream_tps: Optional[float] = None
    single_stream_wall_seconds: Optional[float] = None
    single_stream_total_tokens: int = 0
    # Concurrency metrics.
    concurrency: int = 1
    aggregate_tps: Optional[float] = None
    aggregate_wall_seconds: Optional[float] = None
    aggregate_total_tokens: int = 0
    aggregate_ok_requests: int = 0
    aggregate_failed_requests: int = 0
    scaling_factor: Optional[float] = None
    # Diagnostics.
    ttft_seconds: Optional[float] = None
    ttft_note: str = (
        "unavailable: the shared OpenAICompatibleClient does not expose token "
        "streaming, so time-to-first-token is not measured (never fabricated)."
    )
    tokens_estimated: bool = False
    sample_output_snippet: str = ""
    per_prompt: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Token accounting.
# --------------------------------------------------------------------------- #

def _estimate_tokens(text: str) -> int:
    """Whitespace word-count * 1.3 fallback when the server omits usage.

    Deliberately crude — it only needs to be in the right ballpark, and
    every result that uses it is flagged ``tokens_estimated=True`` so the
    report never conflates an estimate with a server tally.
    """
    words = len((text or "").split())
    return int(round(words * 1.3))


def _completion_tokens(client: Any, response_text: str) -> Tuple[int, bool]:
    """Return (tokens, estimated?) preferring server-reported usage."""
    usage = getattr(client, "last_usage", None) or {}
    reported = 0
    try:
        reported = int(usage.get("completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        reported = 0
    if reported > 0:
        return reported, False
    return _estimate_tokens(response_text), True


# --------------------------------------------------------------------------- #
# One call through the registry client.
# --------------------------------------------------------------------------- #

def _run_one_call(
    client: Any,
    prompt: BenchPrompt,
    *,
    max_tokens: int,
) -> SingleCallResult:
    """Issue one chat completion and time it. Never raises — errors captured."""
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    t0 = time.monotonic()
    try:
        text = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        wall = time.monotonic() - t0
        toks, est = _completion_tokens(client, text)
        return SingleCallResult(
            prompt_name=prompt.name,
            ok=True,
            wall_seconds=wall,
            completion_tokens=toks,
            tokens_estimated=est,
            output_snippet=(text or "")[:280].replace("\n", " ").strip(),
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never abort the sweep
        wall = time.monotonic() - t0
        return SingleCallResult(
            prompt_name=prompt.name,
            ok=False,
            wall_seconds=wall,
            completion_tokens=0,
            tokens_estimated=False,
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


def _build_client(provider: str, *, timeout: float, max_tokens: int) -> Any:
    """Build a fresh registry client for ``provider``.

    A fresh client per use keeps the per-instance ``last_usage`` isolated so
    concurrent requests never race on each other's token tally.
    """
    from lib.llm.endpoints import build_openai_compatible_client

    # json_mode=False: these are free-prose synthesis prompts, not
    # JSON-grammar-constrained decodes, so we measure raw prose throughput.
    return build_openai_compatible_client(
        provider,
        json_mode=False,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# Per-provider benchmark.
# --------------------------------------------------------------------------- #

def _benchmark_provider(
    provider: str,
    *,
    concurrency: int,
    max_tokens_override: Optional[int],
    timeout: float,
) -> ProviderReport:
    report = ProviderReport(provider=provider, concurrency=concurrency)

    # --- Resolve identity (fail-soft; a bad registry name is a per-provider
    #     error, not a crash). ---
    try:
        from lib.llm.endpoints import resolve_endpoint

        resolved = resolve_endpoint(provider)
        report.resolved_model = resolved.model
        report.resolved_base_url = resolved.base_url
    except Exception as exc:  # noqa: BLE001
        report.error = f"resolve_failed: {type(exc).__name__}: {str(exc)[:200]}"
        return report

    def _max_tokens_for(p: BenchPrompt) -> int:
        return int(max_tokens_override) if max_tokens_override else p.default_max_tokens

    # ================= Single-stream pass ================= #
    # One client, one prompt at a time. Establishes the baseline decode tps
    # AND doubles as the reachability probe: if every prompt errors here, the
    # server is down / misconfigured and we record the error + skip the
    # (pointless) concurrency pass.
    single_results: List[SingleCallResult] = []
    try:
        client = _build_client(provider, timeout=timeout, max_tokens=1)
    except Exception as exc:  # noqa: BLE001
        report.error = f"client_build_failed: {type(exc).__name__}: {str(exc)[:200]}"
        return report

    for prompt in _PROMPTS:
        res = _run_one_call(client, prompt, max_tokens=_max_tokens_for(prompt))
        single_results.append(res)

    ok_singles = [r for r in single_results if r.ok]
    report.per_prompt = [asdict(r) for r in single_results]

    if not ok_singles:
        # Every single-stream prompt failed → provider unreachable / broken.
        first_err = next(
            (r.error for r in single_results if r.error), "unknown error"
        )
        report.error = f"all_prompts_failed: {first_err}"
        return report

    single_total_tokens = sum(r.completion_tokens for r in ok_singles)
    single_total_wall = sum(r.wall_seconds for r in ok_singles)
    report.single_stream_total_tokens = single_total_tokens
    report.single_stream_wall_seconds = round(single_total_wall, 4)
    report.single_stream_tps = (
        round(single_total_tokens / single_total_wall, 2)
        if single_total_wall > 0
        else None
    )
    report.tokens_estimated = any(r.tokens_estimated for r in ok_singles)
    report.sample_output_snippet = ok_singles[0].output_snippet
    report.ok = True

    # ================= Concurrency pass ================= #
    if concurrency <= 1:
        # No fan-out requested: aggregate == single-stream by definition.
        report.aggregate_tps = report.single_stream_tps
        report.aggregate_wall_seconds = report.single_stream_wall_seconds
        report.aggregate_total_tokens = single_total_tokens
        report.aggregate_ok_requests = len(ok_singles)
        report.scaling_factor = 1.0 if report.single_stream_tps else None
        return report

    # Fire N identical requests concurrently. We pick the SHORT content-block
    # prompt as the concurrency probe so the batch turns over quickly and the
    # aggregate number reflects steady-state batched decode, not a few slow
    # long-outline stragglers dominating wall-clock.
    probe = _PROMPTS[0]
    probe_max_tokens = _max_tokens_for(probe)

    def _worker(_idx: int) -> SingleCallResult:
        # Fresh client per worker → isolated last_usage, thread-safe.
        try:
            c = _build_client(provider, timeout=timeout, max_tokens=1)
        except Exception as exc:  # noqa: BLE001
            return SingleCallResult(
                prompt_name=probe.name,
                ok=False,
                wall_seconds=0.0,
                completion_tokens=0,
                tokens_estimated=False,
                error=f"client_build_failed: {type(exc).__name__}: {str(exc)[:160]}",
            )
        return _run_one_call(c, probe, max_tokens=probe_max_tokens)

    conc_results: List[SingleCallResult] = []
    wall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, i) for i in range(concurrency)]
        for fut in as_completed(futures):
            conc_results.append(fut.result())
    aggregate_wall = time.monotonic() - wall_start

    ok_conc = [r for r in conc_results if r.ok]
    report.aggregate_ok_requests = len(ok_conc)
    report.aggregate_failed_requests = len(conc_results) - len(ok_conc)
    agg_tokens = sum(r.completion_tokens for r in ok_conc)
    report.aggregate_total_tokens = agg_tokens
    report.aggregate_wall_seconds = round(aggregate_wall, 4)
    if any(r.tokens_estimated for r in ok_conc):
        report.tokens_estimated = True

    # Aggregate tps = total tokens decoded across ALL concurrent requests /
    # the wall-clock of the whole batch. This is the batched-throughput
    # number the Spark serving path is judged on.
    if aggregate_wall > 0 and agg_tokens > 0:
        report.aggregate_tps = round(agg_tokens / aggregate_wall, 2)

    # Scaling factor vs a single stream of the SAME probe prompt. We
    # recompute a single-probe tps (rather than reuse the prompt-set mean)
    # so the ratio is apples-to-apples on the same prompt.
    probe_single = next(
        (r for r in ok_singles if r.prompt_name == probe.name), None
    )
    single_probe_tps: Optional[float] = None
    if probe_single and probe_single.wall_seconds > 0:
        single_probe_tps = probe_single.completion_tokens / probe_single.wall_seconds
    if (
        report.aggregate_tps is not None
        and single_probe_tps
        and single_probe_tps > 0
    ):
        report.scaling_factor = round(report.aggregate_tps / single_probe_tps, 2)

    # Concurrency latency spread (informational).
    if ok_conc:
        walls = [r.wall_seconds for r in ok_conc]
        report.per_prompt.append({
            "prompt_name": f"{probe.name}__concurrent_x{concurrency}",
            "ok_requests": len(ok_conc),
            "failed_requests": report.aggregate_failed_requests,
            "wall_p50_seconds": round(statistics.median(walls), 4),
            "wall_max_seconds": round(max(walls), 4),
            "single_probe_tps": round(single_probe_tps, 2) if single_probe_tps else None,
        })

    return report


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #

def _print_table(reports: List[ProviderReport], concurrency: int) -> None:
    print("\n" + "=" * 92)
    print(f"GENERATION-PROVIDER BENCHMARK  (concurrency={concurrency})")
    print("=" * 92)
    header = (
        f"{'provider':14s} {'model':26s} {'single tps':>11s} "
        f"{'agg tps':>10s} {'scale':>7s} {'status':>10s}"
    )
    print(header)
    print("-" * 92)
    for r in reports:
        model = (r.resolved_model or "-")[:26]
        single = f"{r.single_stream_tps:.1f}" if r.single_stream_tps else "-"
        agg = f"{r.aggregate_tps:.1f}" if r.aggregate_tps else "-"
        scale = f"{r.scaling_factor:.1f}x" if r.scaling_factor else "-"
        status = "OK" if r.ok else "ERROR"
        print(
            f"{r.provider:14s} {model:26s} {single:>11s} "
            f"{agg:>10s} {scale:>7s} {status:>10s}"
        )
        if not r.ok and r.error:
            print(f"{'':14s} └─ {r.error[:74]}")
        elif r.tokens_estimated:
            print(f"{'':14s} └─ (token counts estimated — server omitted usage)")
    print("=" * 92)
    print(
        "TTFT is not measured (shared client has no token streaming); "
        "aggregate tps = total decoded tokens / batch wall-clock."
    )


def _default_output_path() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    root_env = os.environ.get("ED4ALL_ROOT")
    root = Path(root_env) if root_env else _REPO_ROOT
    out_dir = root / "state" / "benchmarks"
    return out_dir / f"generation_providers_{ts}.json"


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--providers",
        nargs="+",
        default=["local"],
        help="Endpoint NAMES from config/endpoints.yaml to sweep "
             "(e.g. local spark-nano spark-super).",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of identical requests to fire concurrently for the "
             "aggregate-throughput measurement (the key Spark metric). "
             "1 = single-stream only.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the per-prompt generation cap (default: per-prompt).",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request HTTP timeout in seconds (default 300).",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="JSON report path (default: runtime/state/benchmarks/"
             "generation_providers_<ts>.json).",
    )
    args = ap.parse_args()

    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")

    output_path = Path(args.output) if args.output else _default_output_path()

    print(
        f"Benchmarking providers={args.providers} concurrency={args.concurrency} "
        f"timeout={args.timeout}s",
        flush=True,
    )

    reports: List[ProviderReport] = []
    for provider in args.providers:
        print(f"\n--- provider: {provider} ---", flush=True)
        report = _benchmark_provider(
            provider,
            concurrency=args.concurrency,
            max_tokens_override=args.max_tokens,
            timeout=args.timeout,
        )
        reports.append(report)
        if report.ok:
            print(
                f"  single_stream_tps={report.single_stream_tps} "
                f"aggregate_tps={report.aggregate_tps} "
                f"scaling_factor={report.scaling_factor}",
                flush=True,
            )
        else:
            print(f"  ERROR: {report.error}", flush=True)

    _print_table(reports, args.concurrency)

    payload = {
        "schema": "ed4all.generation_benchmark/1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "concurrency": args.concurrency,
        "max_tokens_override": args.max_tokens,
        "timeout_seconds": args.timeout,
        "prompts": [
            {"name": p.name, "default_max_tokens": p.default_max_tokens}
            for p in _PROMPTS
        ],
        "providers": [asdict(r) for r in reports],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nReport written to: {output_path}", flush=True)

    # Exit non-zero only if EVERY provider errored (so CI can tell a total
    # miss from a partial sweep where at least one seat answered).
    any_ok = any(r.ok for r in reports)
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
