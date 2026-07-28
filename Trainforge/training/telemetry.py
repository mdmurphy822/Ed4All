"""Live, run-scoped telemetry for the trusted TRL/HF training loop.

The JSONL stream is append-only and each event is also projected to an atomic
``latest`` snapshot for readers that must never observe a partial write.
Event IDs are stable across checkpoint resume, so replayed callback boundaries
do not duplicate progress records.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

SCHEMA_VERSION = 1
STREAM_FILENAME = "training_telemetry.v1.jsonl"
LATEST_FILENAME = "training_telemetry.latest.v1.json"


def _finite_number(value: Any) -> Optional[float | int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _json_metrics(logs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep JSON-safe measured values, dropping only non-finite numbers."""
    result: Dict[str, Any] = {}
    for key, value in (logs or {}).items():
        if value is None or isinstance(value, (str, bool)):
            result[str(key)] = value
            continue
        clean = _finite_number(value)
        if clean is not None:
            result[str(key)] = clean
    return result


def tokenized_dataset_metrics(
    *,
    tokenizer: Any,
    sequences: Sequence[str],
    max_seq_length: int,
    completion_sequences: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Measure sequence/truncation/mask capacity directly with the tokenizer.

    ``padding_fraction`` is the exact fraction of padding when these tokenized
    records are rectangularized to the configured maximum sequence length.
    Runtime dynamic-padding efficiency can differ; the name is retained as the
    stable telemetry contract and its basis is emitted alongside it.
    """
    maximum = int(max_seq_length)
    if maximum <= 0:
        raise ValueError("max_seq_length must be positive")
    if completion_sequences is not None and (
        len(completion_sequences) != len(sequences)
    ):
        raise ValueError("completion_sequences must align with sequences")
    if not callable(tokenizer):
        return {
            "dataset_token_metrics_available": False,
            "sequence_length_p50": None,
            "sequence_length_p95": None,
            "sequence_length_max": None,
            "truncated_examples": None,
            "padding_fraction": None,
            "padding_fraction_basis": "configured_max_seq_length",
            "completion_mask_fraction": None,
            "non_padding_tokens": None,
        }

    lengths: list[int] = []
    completion_lengths: list[int] = []
    for index, text in enumerate(sequences):
        encoded = tokenizer(
            str(text), add_special_tokens=False, truncation=False,
        )
        ids = encoded.get("input_ids", encoded)
        length = len(ids)
        lengths.append(length)
        if completion_sequences is not None:
            completion_encoded = tokenizer(
                str(completion_sequences[index]),
                add_special_tokens=False,
                truncation=False,
            )
            completion_ids = completion_encoded.get(
                "input_ids", completion_encoded,
            )
            completion_lengths.append(len(completion_ids))
    if not lengths:
        return {
            "sequence_length_p50": 0,
            "sequence_length_p95": 0,
            "sequence_length_max": 0,
            "truncated_examples": 0,
            "padding_fraction": 0.0,
            "padding_fraction_basis": "configured_max_seq_length",
            "completion_mask_fraction": None,
            "non_padding_tokens": 0,
        }
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        position = max(0, math.ceil(fraction * len(ordered)) - 1)
        return int(ordered[position])

    retained = [min(length, maximum) for length in lengths]
    retained_total = sum(retained)
    capacity = len(retained) * maximum
    completion_retained = 0
    if completion_lengths:
        completion_retained = sum(
            min(completion, retained_length)
            for completion, retained_length
            in zip(completion_lengths, retained)
        )
    return {
        "dataset_token_metrics_available": True,
        "sequence_length_p50": percentile(0.50),
        "sequence_length_p95": percentile(0.95),
        "sequence_length_max": max(lengths),
        "truncated_examples": sum(length > maximum for length in lengths),
        "padding_fraction": (
            (capacity - retained_total) / capacity if capacity else 0.0
        ),
        "padding_fraction_basis": "configured_max_seq_length",
        "completion_mask_fraction": (
            completion_retained / retained_total
            if completion_sequences is not None and retained_total
            else None
        ),
        "non_padding_tokens": retained_total,
    }


class TrainingTelemetryWriter:
    """Crash-tolerant, deduplicated writer for one model run directory."""

    def __init__(
        self,
        run_dir: Path,
        *,
        stage: str,
        run_id: Optional[str] = None,
        clock: Any = time.monotonic,
    ) -> None:
        normalized = str(stage).strip().lower()
        if normalized not in {"sft", "dpo"}:
            raise ValueError("training telemetry stage must be 'sft' or 'dpo'")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage = normalized
        self.run_id = (
            str(run_id or os.environ.get("ED4ALL_RUN_ID") or "").strip()
            or self.run_dir.name
        )
        self.stream_path = self.run_dir / STREAM_FILENAME
        self.latest_path = self.run_dir / LATEST_FILENAME
        self._clock = clock
        self._started = float(clock())
        self._known_ids, self._sequence = self._read_existing()

    def _read_existing(self) -> tuple[set[str], int]:
        ids: set[str] = set()
        highest = 0
        try:
            with self.stream_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A crash may leave only the final line incomplete.
                        continue
                    event_id = record.get("event_id")
                    if isinstance(event_id, str):
                        ids.add(event_id)
                    highest = max(highest, int(record.get("sequence", 0) or 0))
        except FileNotFoundError:
            pass
        return ids, highest

    def _event_id(self, event: str, global_step: int) -> str:
        identity = (
            f"v{SCHEMA_VERSION}\0{self.run_id}\0{self.stage}\0"
            f"{event}\0{int(global_step)}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def emit(
        self,
        event: str,
        *,
        status: str,
        global_step: int = 0,
        max_steps: int = 0,
        epoch: Optional[float] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        force_distinct: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append one event and atomically refresh the latest snapshot.

        ``force_distinct`` is reserved for terminal transitions that may occur
        at the same step in separate stages/sessions. Normal progress and
        checkpoint events deliberately deduplicate by stage/event/step.
        """
        event_id = self._event_id(
            f"{event}:{force_distinct}" if force_distinct else event,
            global_step,
        )
        elapsed = max(0.0, float(self._clock()) - self._started)
        clean_metrics = _json_metrics(metrics)
        clean_metrics.setdefault("elapsed_seconds", round(elapsed, 6))
        if global_step > 0 and max_steps > global_step and elapsed > 0:
            clean_metrics.setdefault(
                "eta_seconds",
                round(elapsed * (max_steps - global_step) / global_step, 6),
            )
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "sequence": 0,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": self.run_id,
            "stage": self.stage,
            "event": str(event),
            "status": str(status),
            "global_step": int(global_step),
            "max_steps": int(max_steps),
            "epoch": _finite_number(epoch),
            "metrics": clean_metrics,
        }
        # One O_APPEND write under an advisory lock prevents interleaving.
        import fcntl

        descriptor = os.open(
            self.stream_path,
            os.O_APPEND | os.O_CREAT | os.O_RDWR,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            # Re-read under the lock: two processes may have constructed
            # writers before either emitted. This makes deduplication and the
            # monotonically increasing sequence correct across processes.
            os.lseek(descriptor, 0, os.SEEK_SET)
            existing_ids: set[str] = set()
            highest = 0
            with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as reader:
                for line in reader:
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    existing_id = existing.get("event_id")
                    if isinstance(existing_id, str):
                        existing_ids.add(existing_id)
                    highest = max(
                        highest, int(existing.get("sequence", 0) or 0)
                    )
            if event_id in existing_ids:
                self._known_ids.add(event_id)
                self._sequence = max(self._sequence, highest)
                return None
            self._sequence = max(self._sequence, highest) + 1
            record["sequence"] = self._sequence
            encoded = (
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        temporary = self.latest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.latest_path)
        self._known_ids.add(event_id)
        return record


class _TrainingTelemetryMixin:
    """HF ``TrainerCallback`` behavior without importing transformers."""

    def _init_telemetry(
        self,
        *,
        run_dir: Path,
        stage: str,
        pair_count: int,
        configured_max_seq_length: int,
        torch_module: Any,
        resumed: bool,
        dataset_metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.writer = TrainingTelemetryWriter(run_dir, stage=stage)
        self.pair_count = int(pair_count)
        self.configured_max_seq_length = int(configured_max_seq_length)
        self.torch_module = torch_module
        self.resumed = bool(resumed)
        self.dataset_metrics = dict(dataset_metrics or {})

    def _state(self, state: Any) -> tuple[int, int, Optional[float]]:
        return (
            int(getattr(state, "global_step", 0) or 0),
            int(getattr(state, "max_steps", 0) or 0),
            _finite_number(getattr(state, "epoch", None)),
        )

    def _cuda_metrics(self) -> Dict[str, Any]:
        cuda = getattr(self.torch_module, "cuda", None)
        if not cuda or not cuda.is_available():
            return {
                "cuda_available": 0,
                "cuda_allocated_bytes": None,
                "cuda_reserved_bytes": None,
                "cuda_free_bytes": None,
                "cuda_total_bytes": None,
                "cuda_headroom_fraction": None,
                **self._host_metrics(),
            }
        metrics: Dict[str, Any] = {"cuda_available": 1}
        names = {
            "memory_allocated": "cuda_allocated_bytes",
            "memory_reserved": "cuda_reserved_bytes",
            "max_memory_allocated": "cuda_peak_allocated_bytes",
            "max_memory_reserved": "cuda_peak_reserved_bytes",
        }
        for name, normalized in names.items():
            getter = getattr(cuda, name, None)
            if callable(getter):
                metrics[normalized] = int(getter())
        mem_getter = getattr(cuda, "mem_get_info", None)
        if callable(mem_getter):
            free, total = mem_getter()
            metrics["cuda_free_bytes"] = int(free)
            metrics["cuda_total_bytes"] = int(total)
            metrics["cuda_headroom_fraction"] = (
                float(free) / float(total) if total else 0.0
            )
        return {**metrics, **self._host_metrics()}

    @staticmethod
    def _host_metrics() -> Dict[str, Any]:
        values: Dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(
                encoding="utf-8"
            ).splitlines():
                name, raw = line.split(":", 1)
                if name in {"MemTotal", "MemAvailable"}:
                    values[name] = int(raw.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        return {
            "host_memory_available_bytes": available,
            "host_memory_total_bytes": total,
            "host_memory_headroom_fraction": (
                available / total if available is not None and total else None
            ),
            # CUDA UVM has no portable process-local usage counter. Keep the
            # field explicit and unknown instead of equating it with host RAM.
            "uvm_memory_headroom_bytes": None,
        }

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any,
    ) -> Any:
        step, maximum, epoch = self._state(state)
        metrics = {
            "pair_count": self.pair_count,
            "configured_max_seq_length": self.configured_max_seq_length,
            "resumed_from_checkpoint": int(self.resumed),
            "consumed_examples": 0,
            "skipped_examples": 0,
            "non_padding_tokens_per_second": None,
            **self.dataset_metrics,
            **self._cuda_metrics(),
        }
        self.writer.emit(
            "stage_start", status="running", global_step=step,
            max_steps=maximum, epoch=epoch, metrics=metrics,
            force_distinct="resume" if self.resumed else "fresh",
        )
        return control

    def on_log(
        self, args: Any, state: Any, control: Any,
        logs: Optional[Mapping[str, Any]] = None, **kwargs: Any,
    ) -> Any:
        step, maximum, epoch = self._state(state)
        metrics = {**_json_metrics(logs), **self._cuda_metrics()}
        for source in (
            "non_padding_tokens_per_second", "train_tokens_per_second",
            "tokens_per_second",
        ):
            if source in metrics:
                metrics["non_padding_tokens_per_second"] = metrics[source]
                break
        if (
            metrics.get("non_padding_tokens_per_second") is None
            and metrics.get("train_runtime")
            and self.dataset_metrics.get("non_padding_tokens") is not None
            and epoch is not None
        ):
            metrics["non_padding_tokens_per_second"] = (
                float(self.dataset_metrics["non_padding_tokens"])
                * float(epoch)
                / float(metrics["train_runtime"])
            )
        metrics.setdefault("non_padding_tokens_per_second", None)
        nonfinite = sum(
            1 for value in (logs or {}).values()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        )
        metrics["nonfinite_metric_count"] = nonfinite
        if maximum > 0:
            metrics["progress_fraction"] = min(1.0, step / maximum)
        self.writer.emit(
            "progress", status="running", global_step=step,
            max_steps=maximum, epoch=epoch, metrics=metrics,
        )
        return control

    def on_save(
        self, args: Any, state: Any, control: Any, **kwargs: Any,
    ) -> Any:
        step, maximum, epoch = self._state(state)
        checkpoint_dir = (
            self.writer.run_dir / ".trainer_checkpoints" / self.writer.stage
            / f"checkpoint-{step}"
        )
        checkpoint_bytes = 0
        checkpoint_duration_seconds: Optional[float] = None
        if checkpoint_dir.is_dir():
            try:
                stats = [
                    path.stat() for path in checkpoint_dir.rglob("*")
                    if path.is_file()
                ]
                checkpoint_bytes = sum(stat.st_size for stat in stats)
                if stats:
                    checkpoint_duration_seconds = max(
                        stat.st_mtime for stat in stats
                    ) - min(stat.st_mtime for stat in stats)
            except OSError:
                checkpoint_bytes = 0
        self.writer.emit(
            "checkpoint", status="running", global_step=step,
            max_steps=maximum, epoch=epoch,
            metrics={
                **self._cuda_metrics(),
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_duration_seconds": checkpoint_duration_seconds,
                "checkpoint_path": str(
                    checkpoint_dir.relative_to(self.writer.run_dir)
                ),
            },
        )
        return control

    def on_train_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any,
    ) -> Any:
        step, maximum, epoch = self._state(state)
        paused = bool(getattr(control, "should_training_stop", False)) and (
            maximum <= 0 or step < maximum
        )
        self.writer.emit(
            "stage_end", status="paused" if paused else "completed",
            global_step=step, max_steps=maximum, epoch=epoch,
            metrics={
                **self._cuda_metrics(),
                "early_stop_triggered": int(paused),
                "consumed_examples": (
                    self.pair_count
                    if not paused and epoch is not None and epoch >= 1.0
                    else None
                ),
                "skipped_examples": 0,
            },
        )
        return control


def build_training_telemetry_callback(
    *,
    run_dir: Path,
    stage: str,
    pair_count: int,
    configured_max_seq_length: int,
    torch_module: Any,
    resumed: bool,
    dataset_metrics: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Create a callback compatible with both full and stubbed transformers.

    HF's callback handler is intentionally duck-typed.  The lightweight mixin
    instance therefore remains fully functional in hermetic tests and partial
    runtime probes where the base ``TrainerCallback`` symbol is absent.
    """
    try:
        from transformers import TrainerCallback  # type: ignore
    except ImportError:
        TrainingTelemetryCallback = _TrainingTelemetryMixin
    else:
        class TrainingTelemetryCallback(  # type: ignore[misc,no-redef]
            _TrainingTelemetryMixin, TrainerCallback
        ):
            pass

    callback = TrainingTelemetryCallback()
    callback._init_telemetry(
        run_dir=run_dir,
        stage=stage,
        pair_count=pair_count,
        configured_max_seq_length=configured_max_seq_length,
        torch_module=torch_module,
        resumed=resumed,
        dataset_metrics=dataset_metrics,
    )
    return callback


__all__ = [
    "LATEST_FILENAME",
    "SCHEMA_VERSION",
    "STREAM_FILENAME",
    "TrainingTelemetryWriter",
    "_TrainingTelemetryMixin",
    "build_training_telemetry_callback",
    "tokenized_dataset_metrics",
]
