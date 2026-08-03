"""Build-cost metering aggregator — roadmap OP2.

Deterministic post-loop aggregator that sums the resource cost of a build
from the artifacts every run already drops under ``runtime/state/runs/<run_id>/``:

* **Per-phase wall-clock** — from ``checkpoints/*.json`` (``started_at`` /
  ``completed_at``). Always emitted (a run that got far enough to aggregate
  has checkpoints).
* **GPU residency** — from ``vram_trajectory.jsonl`` (each row carries a
  ``ts`` + ``phase`` + ``resident_models``). Trajectory samples are joined to
  the phase wall-clock windows; per phase we surface the observed residency
  span + peak resident VRAM. **Absent file → the section is omitted** (a run
  without ``ED4ALL_VRAM_DOCTOR`` writes no trajectory).
* **LLM calls / tokens** — from ``llm_usage.jsonl`` (one row per
  chat-completion call, written by the OP2 usage tap in
  ``Trainforge/generators/providers/_openai_compatible_client.py`` when ``ED4ALL_RUN_ID``
  is set). Tallied globally + per-provider + per-model. **Absent file → the
  section is omitted** (a run whose calls bypassed the shared client, or a
  library user with no run id).

No LLM decisions are made — this is pure metering — so there is no
decision-capture surface. Best-effort: aggregator failure logs a warning and
never alters ``final_status``.

Lands at ``<libv2_course>/build_cost_report.json`` with a trainforge-dir
fallback (mirrors the coverage-map / promotion-chain reports).

Schema: :file:`schemas/aggregators/build_cost.schema.json` (Draft 2020-12,
``additionalProperties: false``).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"


def _read_json(path: Path) -> Optional[Any]:
    """Best-effort JSON read; returns None on any failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("build_cost: cannot read %s: %s", path, exc)
        return None


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Best-effort JSONL read; skips malformed lines, returns [] on failure."""
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("build_cost: cannot read %s: %s", path, exc)
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile over ``values`` (Task #10 TTFT).

    Deterministic; sorts a copy. Returns None on an empty list, the single
    value for a one-element list, else the ``pct``-th percentile via the
    standard ``(n-1)*p`` fractional-rank interpolation. Rounded to 3 dp for a
    stable emit.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(ordered[int(rank)], 3)
    interpolated = ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return round(interpolated, 3)


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 instant to a naive-UTC datetime (comparable).

    Handles both the tz-aware trajectory ``ts`` (``...+00:00`` / ``Z``) and
    the naive checkpoint ``started_at`` / ``completed_at`` (assumed UTC).
    Returns None on any unparseable value so a single bad row can't crash the
    join.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class BuildCostAggregator:
    """Sum a build's wall-clock / GPU-residency / LLM cost from run artifacts.

    Parameters
    ----------
    run_dir:
        The ``runtime/state/runs/<run_id>/`` directory carrying ``checkpoints/``,
        ``vram_trajectory.jsonl``, and ``llm_usage.jsonl``. When unset it is
        resolved as ``get_state_runs_dir() / run_id``.
    course_code:
        Operator-facing course code so cross-run diffs key cleanly.
    run_id:
        Workflow run ID (also selects the default ``run_dir``).
    """

    def __init__(
        self,
        *,
        run_dir: Optional[Path] = None,
        course_code: str = "",
        run_id: str = "",
    ) -> None:
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        if run_dir is not None:
            self.run_dir: Optional[Path] = Path(run_dir)
        elif self.run_id:
            try:
                from lib.paths import get_state_runs_dir

                self.run_dir = get_state_runs_dir() / self.run_id
            except Exception as exc:  # noqa: BLE001 — best-effort resolution
                logger.debug("build_cost: run_dir resolution failed: %s", exc)
                self.run_dir = None
        else:
            self.run_dir = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Build the canonical build-cost report dict (deterministic)."""
        phases = self._build_phases()
        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "course_code": self.course_code,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "phases": phases,
            "totals": self._build_totals(phases),
        }

        gpu = self._build_gpu_residency(phases)
        if gpu is not None:
            report["gpu_residency"] = gpu

        llm = self._build_llm_usage()
        if llm is not None:
            report["llm_usage"] = llm

        return report

    def write(self, output_path: Path) -> Path:
        """Serialise :meth:`build` output to ``output_path`` (deterministic)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build()
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    # ------------------------------------------------------------------
    # Phases (wall-clock)
    # ------------------------------------------------------------------
    def _build_phases(self) -> List[Dict[str, Any]]:
        """Per-phase wall-clock rows from ``checkpoints/*.json``.

        Sorted by ``phase_index`` (falls back to start time then name) so the
        report reads in execution order.
        """
        rows: List[Dict[str, Any]] = []
        if self.run_dir is None:
            return rows
        ckpt_dir = self.run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return rows
        try:
            files = sorted(ckpt_dir.glob("*.json"))
        except OSError:
            return rows
        for f in files:
            payload = _read_json(f)
            if not isinstance(payload, Mapping):
                continue
            phase_name = str(payload.get("phase_name") or f.stem)
            started_at = payload.get("started_at")
            completed_at = payload.get("completed_at")
            start_dt = _parse_ts(started_at)
            end_dt = _parse_ts(completed_at)
            wall_clock_seconds: Optional[float] = None
            if start_dt is not None and end_dt is not None:
                delta = (end_dt - start_dt).total_seconds()
                # A negative delta means a clock skew / malformed pair; drop it
                # rather than emit a nonsense negative cost.
                wall_clock_seconds = round(delta, 3) if delta >= 0 else None
            rows.append(
                {
                    "phase": phase_name,
                    "phase_index": (
                        int(payload["phase_index"])
                        if isinstance(payload.get("phase_index"), int)
                        else None
                    ),
                    "status": (
                        str(payload["status"])
                        if payload.get("status") is not None
                        else None
                    ),
                    "started_at": started_at if isinstance(started_at, str) else None,
                    "completed_at": (
                        completed_at if isinstance(completed_at, str) else None
                    ),
                    "wall_clock_seconds": wall_clock_seconds,
                }
            )
        rows.sort(
            key=lambda r: (
                r["phase_index"] if r["phase_index"] is not None else 1_000_000,
                r["started_at"] or "",
                r["phase"],
            )
        )
        return rows

    @staticmethod
    def _build_totals(phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Roll per-phase wall-clock into a total."""
        total = 0.0
        measured = 0
        for row in phases:
            wc = row.get("wall_clock_seconds")
            if isinstance(wc, (int, float)):
                total += float(wc)
                measured += 1
        return {
            "phase_count": len(phases),
            "measured_phase_count": measured,
            "total_wall_clock_seconds": round(total, 3),
        }

    # ------------------------------------------------------------------
    # GPU residency
    # ------------------------------------------------------------------
    def _build_gpu_residency(
        self, phases: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Join ``vram_trajectory.jsonl`` samples to phase windows.

        Returns None (section omitted) when the trajectory file is absent —
        the doctor only writes it under ``ED4ALL_VRAM_DOCTOR``.
        """
        if self.run_dir is None:
            return None
        traj_path = self.run_dir / "vram_trajectory.jsonl"
        if not traj_path.exists():
            return None
        rows = _iter_jsonl(traj_path)

        # Pre-parse phase windows for the ts-join.
        windows: List[Dict[str, Any]] = []
        for p in phases:
            start_dt = _parse_ts(p.get("started_at"))
            end_dt = _parse_ts(p.get("completed_at"))
            windows.append(
                {
                    "phase": p["phase"],
                    "start": start_dt,
                    "end": end_dt,
                    "samples": [],  # list[(ts_dt, resident_mib, resident_bool)]
                }
            )

        total_samples = 0
        peak_resident_mib_global = 0
        for row in rows:
            ts_dt = _parse_ts(row.get("ts"))
            resident = row.get("resident_models") or []
            resident_mib = 0
            if isinstance(resident, list):
                for m in resident:
                    if isinstance(m, Mapping):
                        try:
                            resident_mib += int(m.get("vram_mib", 0) or 0)
                        except (TypeError, ValueError):
                            continue
            has_resident = bool(resident) if isinstance(resident, list) else False
            peak_resident_mib_global = max(peak_resident_mib_global, resident_mib)
            total_samples += 1
            if ts_dt is None:
                continue
            # Prefer the phase window that CONTAINS ts; fall back to the row's
            # own ``phase`` label so a sample outside every measured window
            # still attributes.
            target = None
            for w in windows:
                if w["start"] is not None and ts_dt < w["start"]:
                    continue
                if w["end"] is not None and ts_dt > w["end"]:
                    continue
                if w["start"] is not None or w["end"] is not None:
                    target = w
                    break
            if target is None:
                row_phase = str(row.get("phase") or "")
                target = next(
                    (w for w in windows if w["phase"] == row_phase), None
                )
            if target is not None:
                target["samples"].append((ts_dt, resident_mib, has_resident))

        per_phase: List[Dict[str, Any]] = []
        for w in windows:
            samples = w["samples"]
            if not samples:
                continue
            ts_list = [s[0] for s in samples]
            resident_ts = [s[0] for s in samples if s[2]]
            span = 0.0
            if len(resident_ts) >= 2:
                span = round(
                    (max(resident_ts) - min(resident_ts)).total_seconds(), 3
                )
            peak = max((s[1] for s in samples), default=0)
            per_phase.append(
                {
                    "phase": w["phase"],
                    "sample_count": len(samples),
                    "resident_sample_count": len(resident_ts),
                    "residency_span_seconds": span,
                    "peak_resident_mib": int(peak),
                    "observed_from": min(ts_list).isoformat() if ts_list else None,
                    "observed_to": max(ts_list).isoformat() if ts_list else None,
                }
            )
        per_phase.sort(key=lambda r: r["phase"])

        return {
            "total_samples": total_samples,
            "peak_resident_mib": int(peak_resident_mib_global),
            "per_phase": per_phase,
        }

    # ------------------------------------------------------------------
    # LLM usage
    # ------------------------------------------------------------------
    def _build_llm_usage(self) -> Optional[Dict[str, Any]]:
        """Tally ``llm_usage.jsonl`` globally + per-provider + per-model.

        Returns None (section omitted) when the file is absent.
        """
        if self.run_dir is None:
            return None
        usage_path = self.run_dir / "llm_usage.jsonl"
        if not usage_path.exists():
            return None
        rows = _iter_jsonl(usage_path)

        def _int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        def _float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        total_calls = 0
        total_prompt = 0
        total_completion = 0
        total_duration_ms = 0.0
        # Task #10 — TTFT (time-to-first-token) samples, present only on rows
        # written under ``ED4ALL_LLM_TTFT_METER``. Collected here so the report
        # can surface p50/p95 latency-to-first-token WITHOUT changing output
        # for a run whose rows carry no ``ttft_ms`` (additive, no version bump).
        ttft_samples: List[float] = []
        by_provider: Dict[str, Dict[str, Any]] = {}
        by_model: Dict[str, Dict[str, Any]] = {}

        def _accrue(
            bucket: Dict[str, Dict[str, Any]],
            key: str,
            prompt: int,
            completion: int,
            duration_ms: float,
        ) -> None:
            entry = bucket.setdefault(
                key,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "duration_ms": 0.0,
                },
            )
            entry["calls"] += 1
            entry["prompt_tokens"] += prompt
            entry["completion_tokens"] += completion
            entry["duration_ms"] += duration_ms

        for row in rows:
            prompt = _int(row.get("prompt_tokens"))
            completion = _int(row.get("completion_tokens"))
            duration_ms = _float(row.get("duration_ms"))
            provider = str(row.get("provider") or "unknown")
            model = str(row.get("model") or "unknown")
            total_calls += 1
            total_prompt += prompt
            total_completion += completion
            total_duration_ms += duration_ms
            _accrue(by_provider, provider, prompt, completion, duration_ms)
            _accrue(by_model, model, prompt, completion, duration_ms)
            # TTFT is present only on streaming-metered rows; skip non-finite /
            # missing values so a legacy corpus never fabricates a sample.
            if "ttft_ms" in row:
                ttft = _float(row.get("ttft_ms"))
                if math.isfinite(ttft) and ttft >= 0:
                    ttft_samples.append(ttft)

        # Round the accumulated float durations for a stable emit.
        for bucket in (by_provider, by_model):
            for entry in bucket.values():
                entry["duration_ms"] = round(entry["duration_ms"], 3)

        result: Dict[str, Any] = {
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_duration_ms": round(total_duration_ms, 3),
            "by_provider": by_provider,
            "by_model": by_model,
        }
        # Additive TTFT block — emitted ONLY when at least one row carried a
        # ``ttft_ms`` sample, so a run without TTFT metering produces a
        # byte-identical ``llm_usage`` section (no schema version bump).
        if ttft_samples:
            result["ttft_sample_count"] = len(ttft_samples)
            result["ttft_ms_p50"] = _percentile(ttft_samples, 50.0)
            result["ttft_ms_p95"] = _percentile(ttft_samples, 95.0)
        return result


__all__ = [
    "BuildCostAggregator",
    "SCHEMA_VERSION",
]
