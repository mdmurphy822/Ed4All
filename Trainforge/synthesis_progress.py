"""Authoritative run-scoped progress snapshot for training synthesis."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from lib.file_lock import file_lock


SCHEMA_VERSION = 2
PROGRESS_RELATIVE_PATH = Path("telemetry") / "training_synthesis.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_progress_path(run_id: Optional[str] = None) -> Optional[Path]:
    resolved = str(run_id or os.environ.get("ED4ALL_RUN_ID") or "").strip()
    if not resolved:
        return None
    runs = os.environ.get("ED4ALL_STATE_RUNS_DIR", "").strip()
    root = Path(runs) if runs else Path("runtime/state/runs")
    return root / resolved / PROGRESS_RELATIVE_PATH


class SynthesisProgressWriter:
    """Atomically maintain one resumable synthesis progress snapshot."""

    def __init__(
        self,
        *,
        run_id: Optional[str],
        total_units: int,
        max_concurrent: int,
        provider: str,
        model: str,
        fresh_start_id: Optional[str] = None,
        marker_digest: Optional[str] = None,
        holdout_identity: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.path = resolve_progress_path(run_id)
        self.run_id = str(
            run_id or os.environ.get("ED4ALL_RUN_ID") or ""
        ).strip()
        self.fresh_start_id = (
            str(fresh_start_id).strip() if fresh_start_id is not None else None
        )
        self.marker_digest = (
            str(marker_digest).strip() if marker_digest is not None else None
        )
        self.holdout_identity = (
            dict(holdout_identity) if holdout_identity is not None else None
        )
        if (self.fresh_start_id is None) != (self.marker_digest is None):
            raise ValueError(
                "fresh_start_id and marker_digest must be supplied together"
            )
        if self.fresh_start_id == "" or self.marker_digest == "":
            raise ValueError(
                "fresh_start_id and marker_digest must be non-empty when supplied"
            )
        self._started_monotonic = time.monotonic()
        self._prior_elapsed = 0.0
        self._lock = threading.RLock()
        prior = self._read_prior()
        cumulative_keys = (
            "cached_replays",
        )
        self._prior_counts = {
            key: max(int(prior.get(key, 0) or 0), 0)
            for key in cumulative_keys
        }
        self._payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "phase": "training_synthesis",
            "state": "running",
            "run_id": self.run_id,
            "started_at": prior.get("started_at") or _utcnow(),
            "updated_at": _utcnow(),
            "elapsed_seconds": 0.0,
            "total_units": max(int(total_units), 0),
            "completed_units": max(int(prior.get("completed_units", 0) or 0), 0),
            "terminal_units": max(int(prior.get("terminal_units", 0) or 0), 0),
            "accepted_count": max(int(prior.get("accepted_count", 0) or 0), 0),
            "rejected_count": max(int(prior.get("rejected_count", 0) or 0), 0),
            "transient_count": 0,
            "sft_count": max(int(prior.get("sft_count", 0) or 0), 0),
            "dpo_count": max(int(prior.get("dpo_count", 0) or 0), 0),
            "provider_results": max(int(prior.get("provider_results", 0) or 0), 0),
            "cached_replays": self._prior_counts["cached_replays"],
            "transient_attempts": max(int(prior.get("transient_attempts", 0) or 0), 0),
            "recovered_units": max(int(prior.get("recovered_units", 0) or 0), 0),
            "exhausted_units": max(int(prior.get("exhausted_units", 0) or 0), 0),
            "fatal_units": max(int(prior.get("fatal_units", 0) or 0), 0),
            "active_workers": 0,
            "max_concurrent": int(max_concurrent),
            "queued_units": 0,
            "in_flight": 0,
            "backpressure": False,
            "throughput_units_per_second": None,
            "eta_seconds": None,
            "checkpoint_timestamp": prior.get("checkpoint_timestamp"),
            "stop_requested": False,
            "gate_readiness": "pending",
            "provider": str(provider),
            "model": str(model),
            "rejection_reasons": dict(
                prior.get("rejection_reasons") or {}
            ),
        }
        if self.fresh_start_id is not None:
            self._payload["fresh_start_id"] = self.fresh_start_id
            self._payload["marker_digest"] = self.marker_digest
        if self.holdout_identity is not None:
            self._payload["synthesis_holdout_identity"] = self.holdout_identity
        try:
            self._prior_elapsed = max(
                float(prior.get("elapsed_seconds", 0.0) or 0.0), 0.0
            )
        except (TypeError, ValueError):
            self._prior_elapsed = 0.0
        self.write()

    def _read_prior(self) -> Dict[str, Any]:
        if self.path is None:
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            if self.fresh_start_id is not None:
                raise ValueError(
                    f"invalid prior synthesis progress snapshot at {self.path}"
                ) from exc
            return {}
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != SCHEMA_VERSION
            or str(value.get("run_id") or "") != self.run_id
        ):
            if self.fresh_start_id is not None:
                raise ValueError(
                    "prior synthesis progress snapshot does not match the "
                    "current schema and run identity"
                )
            return {}
        if self.fresh_start_id is not None:
            prior_fresh_start_id = value.get("fresh_start_id")
            prior_marker_digest = value.get("marker_digest")
            if prior_fresh_start_id is None or prior_marker_digest is None:
                raise ValueError(
                    "prior synthesis progress snapshot is missing its "
                    "fresh-start identity"
                )
            if (
                prior_fresh_start_id != self.fresh_start_id
                or prior_marker_digest != self.marker_digest
            ):
                raise ValueError(
                    "prior synthesis progress snapshot fresh-start identity "
                    "does not match the current synthesis attempt"
                )
        if (
            self.holdout_identity is not None
            and value.get("synthesis_holdout_identity") != self.holdout_identity
        ):
            raise ValueError(
                "prior synthesis progress snapshot holdout identity does not "
                "match the current immutable split registry"
            )
        return value

    def update(
        self,
        *,
        state: Optional[str] = None,
        completed_units: Optional[int] = None,
        terminal_units: Optional[int] = None,
        accepted_count: Optional[int] = None,
        rejected_count: Optional[int] = None,
        transient_count: Optional[int] = None,
        sft_count: Optional[int] = None,
        dpo_count: Optional[int] = None,
        provider_results: Optional[int] = None,
        cached_replays: Optional[int] = None,
        transient_attempts: Optional[int] = None,
        recovered_units: Optional[int] = None,
        exhausted_units: Optional[int] = None,
        fatal_units: Optional[int] = None,
        active_workers: Optional[int] = None,
        queued_units: Optional[int] = None,
        in_flight: Optional[int] = None,
        stop_requested: Optional[bool] = None,
        gate_readiness: Optional[str] = None,
        rejection_reasons: Optional[Mapping[str, int]] = None,
        checkpointed: bool = False,
    ) -> None:
        with self._lock:
            values = {
                "completed_units": completed_units,
                "terminal_units": terminal_units,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "transient_count": transient_count,
                "sft_count": sft_count,
                "dpo_count": dpo_count,
                "provider_results": provider_results,
                "cached_replays": cached_replays,
                "transient_attempts": transient_attempts,
                "recovered_units": recovered_units,
                "exhausted_units": exhausted_units,
                "fatal_units": fatal_units,
                "active_workers": active_workers,
                "queued_units": queued_units,
                "in_flight": in_flight,
            }
            absolute_keys = {
                "completed_units",
                "terminal_units",
                "accepted_count",
                "rejected_count",
                "sft_count",
                "dpo_count",
                "provider_results",
                "transient_attempts",
                "recovered_units",
                "exhausted_units",
                "fatal_units",
            }
            cumulative_keys = {
                "cached_replays",
            }
            for key, value in values.items():
                if value is not None:
                    resolved = max(int(value), 0)
                    if key in absolute_keys:
                        resolved = max(
                            resolved, int(self._payload.get(key, 0) or 0)
                        )
                    elif key in cumulative_keys:
                        resolved = self._prior_counts[key] + resolved
                    self._payload[key] = resolved
            if state is not None:
                self._payload["state"] = str(state)
            if stop_requested is not None:
                self._payload["stop_requested"] = bool(stop_requested)
            if gate_readiness is not None:
                self._payload["gate_readiness"] = str(gate_readiness)
            if rejection_reasons is not None:
                self._payload["rejection_reasons"] = {
                    str(k): max(int(v), 0)
                    for k, v in rejection_reasons.items()
                }
            if checkpointed:
                self._payload["checkpoint_timestamp"] = _utcnow()
            self.write()

    def write(self) -> None:
        if self.path is None:
            return
        with self._lock:
            elapsed = self._prior_elapsed + max(
                time.monotonic() - self._started_monotonic, 0.0
            )
            self._payload["elapsed_seconds"] = round(elapsed, 3)
            self._payload["updated_at"] = _utcnow()
            done = int(self._payload.get("completed_units", 0) or 0)
            total = int(self._payload.get("total_units", 0) or 0)
            if done > 0 and elapsed > 0:
                rate = done / elapsed
                self._payload["throughput_units_per_second"] = round(
                    rate, 6
                )
                self._payload["eta_seconds"] = (
                    round(max(total - done, 0) / rate, 3)
                    if total > 0
                    and done < total
                    and self._payload.get("state") == "running"
                    else None
                )
            else:
                self._payload["throughput_units_per_second"] = None
                self._payload["eta_seconds"] = None
            queue_depth = int(self._payload.get("queued_units", 0) or 0)
            maximum = int(self._payload.get("max_concurrent", 1) or 1)
            self._payload["backpressure"] = queue_depth >= maximum
            completed = int(self._payload.get("completed_units", 0) or 0)
            terminal = int(self._payload.get("terminal_units", 0) or 0)
            active = int(self._payload.get("active_workers", 0) or 0)
            queued = int(self._payload.get("queued_units", 0) or 0)
            in_flight = int(self._payload.get("in_flight", 0) or 0)
            if (
                completed > total
                or terminal > completed
                or active > maximum
                or in_flight > maximum
                or active + queued != in_flight
            ):
                raise ValueError(
                    "invalid synthesis telemetry relationships: "
                    f"completed={completed}/{total}, terminal={terminal}, "
                    f"active={active}, queued={queued}, in_flight={in_flight}, "
                    f"max={maximum}"
                )

            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_suffix(self.path.suffix + ".lock")
            tmp = self.path.with_name(
                f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            encoded = json.dumps(
                self._payload, indent=2, sort_keys=True
            ) + "\n"
            with file_lock(lock_path, timeout=5.0):
                try:
                    tmp.write_text(encoded, encoding="utf-8")
                    os.replace(tmp, self.path)
                finally:
                    try:
                        tmp.unlink()
                    except FileNotFoundError:
                        pass

    @property
    def payload(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._payload)


__all__ = [
    "PROGRESS_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "SynthesisProgressWriter",
    "resolve_progress_path",
]
