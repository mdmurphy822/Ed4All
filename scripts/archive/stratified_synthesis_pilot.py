#!/usr/bin/env python3
"""Deterministic, isolated synthesis regression pilot.

The harness never writes into the source course. ``prepare`` selects a stable
tail-corpus sample and materializes a small course-shaped workspace. ``run``
executes the normal synthesis implementation against that workspace while
recording request-level telemetry. ``compare`` reports metric deltas between
two runs made from the same sample manifest.

This is intentionally separate from ``Trainforge/scripts/ops/pilot_synthesis.py``:
that operator-facing smoke test samples for property coverage, whereas this
regression pilot targets the late-corpus assessment/explanation and upper-Bloom
content that previously exposed excessive retries and rejection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260726
DEFAULT_PER_STRATUM = 6
DEFAULT_TAIL_FRACTION = 0.5
DEFAULT_MAX_CHUNKS = 26

STRATA = (
    ("assessment_item", "chunk_type", "assessment_item"),
    ("explanation", "chunk_type", "explanation"),
    ("bloom_analyze", "bloom_level", "analyze"),
    ("bloom_evaluate", "bloom_level", "evaluate"),
    ("bloom_create", "bloom_level", "create"),
)

_RETRY_PATTERNS = {
    "leakage_retries": "leakage-retry",
    "truncation_retries": "compact-length retry",
    "parse_retries": "lenient parse retry",
    "short_field_retries": "length-retry",
    "preservation_retries": "preserve-retry",
}

_PROVENANCE_FILES = (
    "Trainforge/synthesize_training.py",
    "Trainforge/generators/_synthesis_provider.py",
    "Trainforge/generators/staged_synthesis_provider.py",
    "Trainforge/generators/instruction_factory.py",
    "Trainforge/generators/preference_factory.py",
    "lib/validators/pair/claim_support.py",
    "lib/validators/pair/objective_delivery.py",
    "lib/validators/pair/promotion.py",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "")


def _stable_rank(seed: int, stratum: str, chunk: Mapping[str, Any]) -> str:
    material = f"{seed}\0{stratum}\0{_chunk_id(chunk)}".encode()
    return hashlib.sha256(material).hexdigest()


def select_sample(
    chunks: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    per_stratum: int = DEFAULT_PER_STRATUM,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select each requested stratum independently from the corpus tail.

    A chunk may satisfy multiple strata. It is materialized once, while the
    manifest records every stratum it covers. This preserves scarce
    evaluate/create coverage instead of discarding it merely because the same
    chunk was already selected as an assessment or explanation.
    """
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if not chunks:
        raise ValueError("source chunkset is empty")

    first_tail_index = max(0, math.floor(len(chunks) * (1.0 - tail_fraction)))
    tail = list(chunks[first_tail_index:])
    selected_by_id: dict[str, dict[str, Any]] = {}
    selected_strata: MutableMapping[str, list[str]] = defaultdict(list)
    strata_report: dict[str, Any] = {}

    for name, field, expected in STRATA:
        candidates = [
            chunk for chunk in tail
            if str(chunk.get(field) or "").strip().lower() == expected
            and _chunk_id(chunk)
            and bool(chunk.get("learning_outcome_refs"))
        ]
        candidates.sort(key=lambda row: _stable_rank(seed, name, row))
        chosen = candidates[:per_stratum]
        chosen_ids = [_chunk_id(row) for row in chosen]
        strata_report[name] = {
            "field": field,
            "value": expected,
            "available_in_tail": len(candidates),
            "target": per_stratum,
            "selected": len(chosen),
            "chunk_ids": chosen_ids,
        }
        for row in chosen:
            cid = _chunk_id(row)
            selected_by_id[cid] = dict(row)
            selected_strata[cid].append(name)

    if not selected_by_id:
        raise ValueError("no eligible chunks matched the requested strata")

    # Preserve canonical corpus order so the production seed+index behavior is
    # deterministic and easy to compare between revisions.
    source_order = {_chunk_id(row): index for index, row in enumerate(chunks)}
    selected = sorted(
        selected_by_id.values(), key=lambda row: source_order[_chunk_id(row)]
    )
    manifest = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "seed": seed,
        "per_stratum": per_stratum,
        "tail_fraction": tail_fraction,
        "source_chunk_count": len(chunks),
        "tail_start_index_zero_based": first_tail_index,
        "tail_chunk_count": len(tail),
        "unique_selected_chunks": len(selected),
        "selected_chunk_ids_in_source_order": [_chunk_id(row) for row in selected],
        "chunk_strata": {
            cid: sorted(names) for cid, names in sorted(selected_strata.items())
        },
        "strata": strata_report,
    }
    return selected, manifest


def _copy_optional_course_inputs(source: Path, pilot_course: Path) -> None:
    """Copy only small deterministic inputs the normal synthesis path reads."""
    for relative in (
        "objectives.json",
        "assessments.json",
        "answer_key.json",
        "manifest.json",
        "course.json",
        "graph/pedagogy_graph.json",
        "pedagogy/pedagogy_graph.json",
        "pedagogy_graph.json",
        "training_specs/assessments.json",
        "training_specs/answer_key.json",
    ):
        src = source / relative
        if not src.is_file():
            continue
        dst = pilot_course / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_workspace(
    *,
    source_course: Path,
    workspace: Path,
    seed: int,
    per_stratum: int,
    tail_fraction: float,
) -> dict[str, Any]:
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(
            f"pilot workspace must be absent or empty: {workspace}"
        )
    from lib.libv2_storage import resolve_imscc_chunks_path

    source_course = source_course.resolve()
    chunks_path = resolve_imscc_chunks_path(source_course, "chunks.jsonl")
    chunks = _read_jsonl(chunks_path)
    objective_candidates = (
        source_course / "objectives.json",
        source_course / "course_planning" / "synthesized_objectives.json",
        source_course / "synthesized_objectives.json",
    )
    objective_path = next(
        (path for path in objective_candidates if path.is_file()), None,
    )
    if objective_path is None:
        raise FileNotFoundError(
            "v4 pilot requires an authoritative objectives.json artifact"
        )
    from lib.validators.pair.objective_delivery import (
        _load_synthesized_objectives_for_w4c,
    )
    from Trainforge.synthesis_eligibility import (
        focus_chunk_on_canonical_objective,
        pair_eligibility,
    )

    frozen_objectives = _load_synthesized_objectives_for_w4c(objective_path)
    if not frozen_objectives:
        raise ValueError("authoritative objective map is empty or malformed")
    admitted: list[dict[str, Any]] = []
    pre_dispatch: list[dict[str, Any]] = []
    for source_index, chunk in enumerate(chunks):
        focused = focus_chunk_on_canonical_objective(
            chunk, seed=seed + source_index, objectives=frozen_objectives,
        )
        sft = pair_eligibility(focused, kind="instruction")
        dpo = pair_eligibility(focused, kind="preference")
        if sft.eligible and dpo.eligible:
            # Freeze the exact canonical focus selected against the immutable
            # source ordering. The isolated run therefore cannot reselect a
            # different tied objective merely because pilot indices are dense.
            admitted.append(dict(focused))
            continue
        pre_dispatch.append({
            "chunk_id": _chunk_id(chunk),
            "source_index": source_index,
            "sft": {"eligible": sft.eligible, "reason": sft.reason},
            "dpo": {"eligible": dpo.eligible, "reason": dpo.reason},
            "source_record_sha256": hashlib.sha256(
                _stable_json_for_hash(chunk).encode("utf-8")
            ).hexdigest(),
        })
    selected, manifest = select_sample(
        admitted,
        seed=seed,
        per_stratum=per_stratum,
        tail_fraction=tail_fraction,
    )
    # Fill the bounded pilot to its intended 26 admitted chunks after scarce
    # strata have been secured. Backfill is deterministic and remains inside
    # the same immutable source chunkset/objective-map hashes.
    selected_ids = {_chunk_id(row) for row in selected}
    replacements = [
        row for row in admitted if _chunk_id(row) not in selected_ids
    ]
    replacements.sort(
        key=lambda row: _stable_rank(seed, "eligible_backfill", row)
    )
    backfill = replacements[:max(0, DEFAULT_MAX_CHUNKS - len(selected))]
    selected.extend(dict(row) for row in backfill)
    admitted_order = {
        _chunk_id(row): index for index, row in enumerate(admitted)
    }
    selected.sort(key=lambda row: admitted_order[_chunk_id(row)])
    backfill_ids = [_chunk_id(row) for row in backfill]
    manifest["unique_selected_chunks"] = len(selected)
    manifest["selected_chunk_ids_in_source_order"] = [
        _chunk_id(row) for row in selected
    ]
    manifest["eligible_backfill_chunk_ids"] = backfill_ids
    manifest["strata"]["eligible_backfill"] = {
        "field": "eligibility",
        "value": "sft_and_dpo",
        "available_in_tail": len(replacements),
        "target": DEFAULT_MAX_CHUNKS,
        "selected": len(backfill),
        "chunk_ids": backfill_ids,
    }
    for chunk_id in backfill_ids:
        manifest["chunk_strata"].setdefault(chunk_id, []).append(
            "eligible_backfill"
        )
    pilot_course = workspace / "course"
    _write_jsonl(pilot_course / "imscc_chunks" / "chunks.jsonl", selected)
    _copy_optional_course_inputs(source_course, pilot_course)
    manifest.update(
        {
            "source_course": str(source_course),
            "source_chunks_sha256": hashlib.sha256(
                chunks_path.read_bytes()
            ).hexdigest(),
            "source_chunk_count": len(chunks),
            "canonical_objective_map": frozen_objectives,
            "canonical_objective_map_sha256": hashlib.sha256(
                _stable_json_for_hash(frozen_objectives).encode("utf-8")
            ).hexdigest(),
            "pre_dispatch_ineligible": pre_dispatch,
            "eligible_source_chunks": len(admitted),
            "replacement_policy": (
                "stable-rank eligible candidates from the immutable source "
                "chunkset hash; no cross-corpus or stochastic replacement"
            ),
        }
    )
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _stable_json_for_hash(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


class _RetryLogCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        with self._lock:
            for metric, pattern in _RETRY_PATTERNS.items():
                if pattern in message:
                    self.counts[metric] += 1


class _RequestRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            self.rows.append(dict(row))


@contextmanager
def record_provider_requests(recorder: _RequestRecorder) -> Iterator[None]:
    """Wrap every current local-provider implementation at the wire boundary."""
    from Trainforge.generators._local_provider import LocalSynthesisProvider
    from Trainforge.generators._synthesis_provider import SynthesisProvider
    from Trainforge.generators._together_provider import TogetherSynthesisProvider

    classes = (SynthesisProvider, LocalSynthesisProvider, TogetherSynthesisProvider)
    originals: list[tuple[type[Any], Any]] = []
    for cls in classes:
        original = cls._chat_completion_raw
        originals.append((cls, original))

        def wrapped(
            self: Any,
            messages: Sequence[Mapping[str, Any]],
            *,
            _original: Any = original,
        ) -> Any:
            started = time.monotonic()
            prompt_chars = sum(
                len(str(message.get("content") or "")) for message in messages
            )
            raw_prompt = json.dumps(
                list(messages), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            )
            try:
                text, usage, http_retries = _original(self, messages)
            except Exception as exc:
                recorder.record(
                    {
                        "prompt_chars": prompt_chars,
                        "response_chars": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "http_retries": None,
                        "elapsed_seconds": time.monotonic() - started,
                        "error_code": str(getattr(exc, "code", "") or ""),
                        "error_type": type(exc).__name__,
                        "raw_prompt": raw_prompt,
                        "prompt_sha256": hashlib.sha256(
                            raw_prompt.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                raise
            recorder.record(
                {
                    "prompt_chars": prompt_chars,
                    "response_chars": len(text),
                    "prompt_tokens": int((usage or {}).get("prompt_tokens", 0)),
                    "completion_tokens": int(
                        (usage or {}).get("completion_tokens", 0)
                    ),
                    "http_retries": int(http_retries),
                    "elapsed_seconds": time.monotonic() - started,
                    "error_code": None,
                    "error_type": None,
                    # The pilot workspace is isolated/gitignored. Retaining
                    # exact wire evidence here permits a human to inspect a
                    # rejected candidate rather than infer it from counters.
                    "raw_prompt": raw_prompt,
                    "raw_response": text,
                    "prompt_sha256": hashlib.sha256(
                        raw_prompt.encode("utf-8")
                    ).hexdigest(),
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )
            return text, usage, http_retries

        cls._chat_completion_raw = wrapped
    try:
        yield
    finally:
        for cls, original in originals:
            cls._chat_completion_raw = original


@contextmanager
def isolated_runtime_environment(workspace: Path) -> Iterator[None]:
    """Keep capture/state side effects inside the pilot workspace.

    The pilot intentionally does not inherit a workflow run identity or the
    production seat-recovery coordinator. A transport failure is evidence for
    a pilot no-go and must not restart or otherwise mutate the shared seat.
    """
    overrides = {
        "ED4ALL_LIBV2_ROOT": str((workspace / "runtime" / "libv2").resolve()),
        "ED4ALL_TRAINING_CAPTURES_DIR": str(
            (workspace / "runtime" / "runtime/training-captures").resolve()
        ),
        "ED4ALL_STATE_RUNS_DIR": str(
            (workspace / "runtime" / "state-runs").resolve()
        ),
        "ED4ALL_SEAT_SCHEDULE": "false",
        # This harness is specifically the bounded v4 qualification. The
        # source workflow remains untouched and the generic bare synthesis API
        # remains legacy-default when this isolated override is absent.
        "TRAINFORGE_STAGED_SYNTHESIS_V4": "true",
    }
    removed = ("ED4ALL_RUN_ID", "RUN_ID")
    previous = {key: os.environ.get(key) for key in (*overrides, *removed)}
    os.environ.update(overrides)
    for key in removed:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def rejection_efficiency_ablation(name: str) -> Iterator[None]:
    """Apply one explicitly named rejection-efficiency ablation.

    The primary ``modeled-old-leakage-policy`` arm reconstructs the observable
    former policy for well-formed provider responses: leakage rewrites may
    consume the remaining nine attempts of the ten-attempt parse budget, and
    exhaustion is surfaced as ``paraphrase_invalid_after_retry`` so the
    factories exercise their deterministic fallback. The historical
    intermediate source blob is unavailable, so this is deliberately labelled
    *modeled*, not an exact historical checkout. Parse failures interleaved
    with leakage remain independently counted by the current implementation;
    the report provenance makes that limitation explicit.

    ``terminal-rejection-replay-disabled`` is a separate optional cache
    experiment and is not a before-arm for acceptance/call efficiency.
    """
    if name == "none":
        yield
        return
    from Trainforge import synthesize_training

    if name == "terminal-rejection-replay-disabled":
        original = synthesize_training._checkpoint_rejection_matches_contract
        synthesize_training._checkpoint_rejection_matches_contract = (
            lambda _record, _fingerprint: False
        )
        try:
            yield
        finally:
            synthesize_training._checkpoint_rejection_matches_contract = original
        return
    if name != "modeled-old-leakage-policy":
        raise ValueError(f"unknown ablation: {name}")

    from Trainforge.generators import _synthesis_provider as provider_module
    from Trainforge.generators._synthesis_common import SynthesisProviderError

    original_budget = provider_module.MAX_LEAKAGE_REWRITE_RETRIES
    provider_class = provider_module.SynthesisProvider
    original_instruction = provider_class.paraphrase_instruction
    original_preference = provider_class.paraphrase_preference

    def modeled_instruction(
        self: Any,
        draft: Mapping[str, Any],
        chunk: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            return original_instruction(self, draft, chunk, **kwargs)
        except SynthesisProviderError as exc:
            if str(getattr(exc, "code", "") or "") != (
                "provider_output_verbatim_leakage"
            ):
                raise
            fallback = dict(draft)
            fallback["paraphrase_fallback_reason"] = (
                "paraphrase_invalid_after_retry"
            )
            return fallback

    def modeled_preference(
        self: Any,
        draft: Mapping[str, Any],
        chunk: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            return original_preference(self, draft, chunk, **kwargs)
        except SynthesisProviderError as exc:
            if str(getattr(exc, "code", "") or "") != (
                "provider_output_verbatim_leakage"
            ):
                raise
            fallback = dict(draft)
            fallback["paraphrase_fallback_reason"] = (
                "paraphrase_invalid_after_retry"
            )
            return fallback

    # MAX_PARSE_RETRIES is ten in the exact current implementation. One
    # request is the initial attempt, leaving nine modeled shared-budget
    # retries on the dominant well-formed-response path.
    provider_module.MAX_LEAKAGE_REWRITE_RETRIES = 9
    provider_class.paraphrase_instruction = modeled_instruction
    provider_class.paraphrase_preference = modeled_preference
    try:
        yield
    finally:
        provider_class.paraphrase_instruction = original_instruction
        provider_class.paraphrase_preference = original_preference
        provider_module.MAX_LEAKAGE_REWRITE_RETRIES = original_budget


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_provenance() -> dict[str, Any]:
    try:
        git_head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    files = {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in _PROVENANCE_FILES
    }
    files[str(Path(__file__).resolve().relative_to(PROJECT_ROOT))] = _sha256(
        Path(__file__).resolve()
    )
    return {"git_head": git_head, "files_sha256": files}


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[rank])


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    numeric = [
        float(value) for value in values
        if value is not None and not isinstance(value, bool)
    ]
    if not numeric:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(numeric),
        "min": min(numeric),
        "median": float(statistics.median(numeric)),
        "p95": _percentile(numeric, 0.95),
        "max": max(numeric),
    }


def _accepted_pairs(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def read(name: str) -> list[dict[str, Any]]:
        path = output_dir / name
        return _read_jsonl(path) if path.exists() else []

    return read("instruction_pairs.jsonl"), read("preference_pairs.jsonl")


def _field_lengths(
    instruction: Sequence[Mapping[str, Any]],
    preference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "sft_prompt_chars": _distribution(len(str(row.get("prompt") or "")) for row in instruction),
        "sft_completion_chars": _distribution(
            len(str(row.get("completion") or "")) for row in instruction
        ),
        "dpo_prompt_chars": _distribution(len(str(row.get("prompt") or "")) for row in preference),
        "dpo_chosen_chars": _distribution(len(str(row.get("chosen") or "")) for row in preference),
        "dpo_rejected_chars": _distribution(
            len(str(row.get("rejected") or "")) for row in preference
        ),
    }


def _pair_quality(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    claim_rates = [
        float(row["claim_support_rate"])
        for row in pairs
        if row.get("claim_support_rate") is not None
    ]
    objective_rates = [
        float(row["pair_objective_alignment_pass_rate"])
        for row in pairs
        if row.get("pair_objective_alignment_pass_rate") is not None
    ]
    bloom = Counter(
        "pass" if row.get("bloom_alignment") is True
        else "fail" if row.get("bloom_alignment") is False
        else "unknown"
        for row in pairs
    )
    return {
        "claim_support_rate": _distribution(claim_rates),
        "objective_alignment_pass_rate": _distribution(objective_rates),
        "bloom_alignment": dict(sorted(bloom.items())),
    }


def fallback_provenance_summary(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count fallbacks only when the emitted pair carries explicit provenance."""
    reasons = Counter(
        str(row["paraphrase_fallback_reason"])
        for row in pairs
        if str(row.get("paraphrase_fallback_reason") or "").strip()
    )
    return {
        "explicit_fallback_pairs": sum(reasons.values()),
        "reasons": dict(sorted(reasons.items())),
    }


def _per_stratum(
    manifest: Mapping[str, Any],
    instruction: Sequence[Mapping[str, Any]],
    preference: Sequence[Mapping[str, Any]],
    rejected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    accepted_sft = Counter(str(row.get("source_chunk_id") or row.get("chunk_id") or "") for row in instruction)
    accepted_dpo = Counter(str(row.get("source_chunk_id") or row.get("chunk_id") or "") for row in preference)
    rejected: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    for row in rejected_rows:
        if row.get("disposition") != "rejected":
            continue
        rejected[str(row.get("chunk_id") or "")][
            f"{row.get('kind')}:{row.get('reason')}"
        ] += 1
    result: dict[str, Any] = {}
    for stratum, data in manifest["strata"].items():
        chunk_ids = list(data["chunk_ids"])
        reasons: Counter[str] = Counter()
        for cid in chunk_ids:
            reasons.update(rejected[cid])
        result[stratum] = {
            "selected_chunks": len(chunk_ids),
            "accepted_sft": sum(accepted_sft[cid] for cid in chunk_ids),
            "accepted_dpo": sum(accepted_dpo[cid] for cid in chunk_ids),
            "rejection_reasons": dict(sorted(reasons.items())),
        }
    return result


def build_report(
    *,
    label: str,
    server_batch: int,
    client_concurrency: int,
    ablation: str,
    workspace: Path,
    stats: Any,
    request_rows: Sequence[Mapping[str, Any]],
    retry_counts: Mapping[str, int],
    elapsed_seconds: float,
) -> dict[str, Any]:
    manifest = json.loads((workspace / "sample_manifest.json").read_text())
    output_dir = workspace / "output" / label
    instruction, preference = _accepted_pairs(output_dir)
    all_pairs = [*instruction, *preference]
    checkpoint_path = output_dir / ".synthesis_pairs_checkpoint.jsonl"
    rejected_rows = _read_jsonl(checkpoint_path) if checkpoint_path.exists() else []
    templates = Counter(str(row.get("template_id") or "unknown") for row in all_pairs)
    request_errors = Counter(
        str(row.get("error_code") or row.get("error_type") or "unknown")
        for row in request_rows if row.get("error_type")
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": label,
        "server_batch": server_batch,
        "client_concurrency": client_concurrency,
        "ablation": ablation,
        "implementation_provenance": implementation_provenance(),
        "sample_manifest_sha256": hashlib.sha256(
            (workspace / "sample_manifest.json").read_bytes()
        ).hexdigest(),
        "sample": manifest,
        "elapsed_seconds": elapsed_seconds,
        "model_calls": len(request_rows),
        "request_errors": dict(sorted(request_errors.items())),
        "retry_counts": {
            key: int(retry_counts.get(key, 0)) for key in _RETRY_PATTERNS
        },
        "truncated_responses": sum(
            str(row.get("error_code") or "") == "output_truncated"
            for row in request_rows
        ),
        "http_retries": sum(
            int(row.get("http_retries") or 0) for row in request_rows
        ),
        "request_lengths": {
            "prompt_chars": _distribution(row.get("prompt_chars") for row in request_rows),
            "response_chars": _distribution(row.get("response_chars") for row in request_rows),
            "prompt_tokens": _distribution(row.get("prompt_tokens") for row in request_rows),
            "completion_tokens": _distribution(
                row.get("completion_tokens") for row in request_rows
            ),
        },
        "raw_provider_calls": list(request_rows),
        "accepted": {
            "sft": len(instruction),
            "dpo": len(preference),
            "total": len(all_pairs),
        },
        "rejected": {
            "sft": int(stats.instruction_pairs_rejected),
            "dpo": int(stats.preference_pairs_rejected),
            "total": int(
                stats.instruction_pairs_rejected + stats.preference_pairs_rejected
            ),
            "reasons": dict(sorted(stats.rejected_reasons.items())),
        },
        "quality": _pair_quality(all_pairs),
        "fallback_provenance": fallback_provenance_summary(all_pairs),
        "template_diversity": {
            "distinct_templates": len(templates),
            "distribution": dict(sorted(templates.items())),
            "top_template_share": (
                max(templates.values()) / len(all_pairs) if all_pairs else None
            ),
            "top3_share": (
                sum(count for _, count in templates.most_common(3)) / len(all_pairs)
                if all_pairs else None
            ),
        },
        "accepted_field_lengths": _field_lengths(instruction, preference),
        "per_stratum": _per_stratum(
            manifest, instruction, preference, rejected_rows
        ),
        "synthesis_stats": stats.as_dict(),
    }


def run_pilot(
    *,
    workspace: Path,
    label: str,
    course_code: str,
    provider: str,
    server_batch: int,
    max_concurrent: int,
    ablation: str = "none",
    resume_existing: bool = False,
) -> dict[str, Any]:
    if not (workspace / "sample_manifest.json").is_file():
        raise FileNotFoundError("run prepare before run")
    output_dir = workspace / "output" / label
    if (
        not resume_existing
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(f"run output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from Trainforge.synthesize_training import run_synthesis

    recorder = _RequestRecorder()
    retry_handler = _RetryLogCounter()
    root_logger = logging.getLogger()
    root_logger.addHandler(retry_handler)
    started = time.monotonic()
    try:
        with (
            isolated_runtime_environment(workspace),
            rejection_efficiency_ablation(ablation),
            record_provider_requests(recorder),
        ):
            stats = run_synthesis(
                corpus_dir=workspace / "course",
                course_code=course_code,
                provider=provider,
                seed=int(json.loads(
                    (workspace / "sample_manifest.json").read_text()
                )["seed"]),
                output_dir=output_dir,
                synthesis_pairs_checkpoint_path=(
                    output_dir / ".synthesis_pairs_checkpoint.jsonl"
                ),
                max_concurrent=max_concurrent,
                smoke_mode="none",
            )
    finally:
        root_logger.removeHandler(retry_handler)
    report = build_report(
        label=label,
        server_batch=server_batch,
        client_concurrency=max_concurrent,
        ablation=ablation,
        workspace=workspace,
        stats=stats,
        request_rows=recorder.rows,
        retry_counts=retry_handler.counts,
        elapsed_seconds=time.monotonic() - started,
    )
    report_path = workspace / f"report-{label}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def compare_reports(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if before.get("sample_manifest_sha256") != after.get("sample_manifest_sha256"):
        raise ValueError("reports were not produced from the same sample manifest")

    def delta(path: Sequence[str]) -> dict[str, Any]:
        left: Any = before
        right: Any = after
        for key in path:
            left = left[key]
            right = right[key]
        return {"before": left, "after": right, "delta": right - left}

    return {
        "schema_version": 1,
        "sample_manifest_sha256": before["sample_manifest_sha256"],
        "before_label": before.get("label"),
        "after_label": after.get("label"),
        "metrics": {
            "model_calls": delta(("model_calls",)),
            "leakage_retries": delta(("retry_counts", "leakage_retries")),
            "truncated_responses": delta(("truncated_responses",)),
            "accepted_sft": delta(("accepted", "sft")),
            "accepted_dpo": delta(("accepted", "dpo")),
            "rejected_total": delta(("rejected", "total")),
            "distinct_templates": delta(
                ("template_diversity", "distinct_templates")
            ),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-course", type=Path, required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    prepare.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION)

    run = subparsers.add_parser("run")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--course-code", required=True)
    run.add_argument(
        "--provider", choices=("local", "together"), default="local"
    )
    run.add_argument(
        "--server-batch",
        type=int,
        required=True,
        help="Declared TRT server max-batch setting, recorded for comparison.",
    )
    run.add_argument("--max-concurrent", type=int, required=True)
    run.add_argument(
        "--ablation",
        choices=(
            "none",
            "modeled-old-leakage-policy",
            "terminal-rejection-replay-disabled",
        ),
        default="none",
    )
    run.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume the same output directory after seeding its checkpoint.",
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "prepare":
        result = prepare_workspace(
            source_course=args.source_course,
            workspace=args.workspace,
            seed=args.seed,
            per_stratum=args.per_stratum,
            tail_fraction=args.tail_fraction,
        )
    elif args.command == "run":
        if not 1 <= args.max_concurrent <= 48:
            raise ValueError("--max-concurrent must be in [1, 48]")
        if not 1 <= args.server_batch <= 48:
            raise ValueError("--server-batch must be in [1, 48]")
        result = run_pilot(
            workspace=args.workspace,
            label=args.label,
            course_code=args.course_code,
            provider=args.provider,
            server_batch=args.server_batch,
            max_concurrent=args.max_concurrent,
            ablation=args.ablation,
            resume_existing=args.resume_existing,
        )
    else:
        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        result = compare_reports(before, after)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
