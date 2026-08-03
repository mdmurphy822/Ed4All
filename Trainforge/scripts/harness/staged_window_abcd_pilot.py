"""Executable, counterbalanced A/B/C/D pilot for staged synthesis windows.

The pilot never manages a model seat.  ``--execute`` calls an already-served
OpenAI-compatible endpoint through the same LocalSynthesisProvider and
StagedSynthesisProvider used by production.  Without ``--execute`` it writes
the deterministic manifest only.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import threading
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from Trainforge.synthesis.verification.decision_audit_verifier import (
    verify_decision_audit,
    write_decision_audit_report,
)
from Trainforge.synthesis.verification.benchmark_artifact_verifier import (
    RECONCILIATION_REPORT_SCHEMA,
    TELEMETRY_REPORT_SCHEMA,
    read_call_intents,
    verified_report,
    verify_http_reconciliation,
    verify_telemetry_artifacts,
)
from Trainforge.generators.synthesis_window_contract import objective_card


COHORT_SIZE = 28
PILOT_VERSION = "ed4all.staged-window-abcd.v2"
LEGACY_SYNTHESIS_CONTRACT = "legacy"
MICRO_SYNTHESIS_CONTRACT = "ed4all.staged-synthesis-micro.v1"
SYNTHESIS_CONTRACT_ALIASES = {
    "legacy": LEGACY_SYNTHESIS_CONTRACT,
    "micro-v1": MICRO_SYNTHESIS_CONTRACT,
    MICRO_SYNTHESIS_CONTRACT: MICRO_SYNTHESIS_CONTRACT,
}
VARIANTS = (
    "A_raw_minimal_objective",
    "B_semantic_minimal_objective",
    "C_raw_rich_objective",
    "D_production_contract",
)
BENCHMARK_CONCURRENCY_CELLS = (1, 16, 20, 24, 28)
BENCHMARK_CELL_LIMIT_SECONDS = 12 * 60
CELL_PUBLICATION_SCHEMA = "ed4all.benchmark-cell-publication.v1"
MATRIX_SUMMARY_SCHEMA = "ed4all.staged-window-matrix-summary.v2"
FROZEN_D_C1_LOADER_VERSION = "ed4all.frozen-d-c1-loader.v1"
FROZEN_D_C1_ELIGIBILITY_SCHEMA = "ed4all.frozen-d-c1-eligibility.v1"
FROZEN_D_C1_LOADER_AUDIT_SCHEMA = "ed4all.frozen-d-c1-loader-audit.v1"
FROZEN_D_C1_ORDERED_IDENTITY_SCHEMA = (
    "ed4all.frozen-d-c1-ordered-identity.v1"
)
COHORT_SELECTION_VERSION = "ed4all.staged-cohort-selection.v3"
_FROZEN_D_C1_REQUIRED_ROW_KEYS = frozenset({
    "pilot_version", "repetition", "chunk_id", "stratum", "kind", "variant",
    "chunk_sha256", "focus_objective",
})
_FROZEN_D_C1_OPTIONAL_ROW_KEYS = {
    # Historical manifests carry these selection coordinates.  Absence and
    # explicit null are equivalent; when present and non-null they are ints.
    "cohort_index": (int, type(None)),
    "order_index": (int, type(None)),
}


def frozen_d_c1_ordered_identity_projection(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project the independently pinned, order-sensitive cohort identity."""
    return [
        {
            "position": position,
            "chunk_id": row.get("chunk_id"),
            "kind": row.get("kind"),
            "variant": row.get("variant"),
            "repetition": row.get("repetition"),
            "chunk_sha256": row.get("chunk_sha256"),
            "focus_objective_sha256": _digest(row.get("focus_objective")),
            "pilot_version": row.get("pilot_version"),
            "stratum": row.get("stratum"),
        }
        for position, row in enumerate(rows)
    ]


def frozen_d_c1_ordered_identity_sha256(
    rows: Iterable[Mapping[str, Any]],
) -> str:
    envelope = {
        "schema": FROZEN_D_C1_ORDERED_IDENTITY_SCHEMA,
        "rows": frozen_d_c1_ordered_identity_projection(rows),
    }
    return _digest(envelope)


def resolve_synthesis_contract(value: str) -> str:
    """Resolve only documented contract spellings; never infer from env."""
    try:
        return SYNTHESIS_CONTRACT_ALIASES[str(value)]
    except KeyError as exc:
        raise ValueError(
            "synthesis contract must be 'legacy' or "
            f"'{MICRO_SYNTHESIS_CONTRACT}'"
        ) from exc


def synthesis_contract_identity(value: str) -> dict[str, Any]:
    """Return the immutable behavior identity for the selected provider."""
    resolved = resolve_synthesis_contract(value)
    if resolved == LEGACY_SYNTHESIS_CONTRACT:
        return {
            "version": LEGACY_SYNTHESIS_CONTRACT,
            "component_hashes": {},
            "fingerprint": None,
        }
    from Trainforge.generators.staged_synthesis_micro import (
        micro_contract_components,
        micro_contract_fingerprint,
    )
    components = micro_contract_components()
    return {
        "version": resolved,
        "component_hashes": components,
        "fingerprint": micro_contract_fingerprint(),
    }


def _micro_stage_coverage_complete(kind: str, stages: Iterable[str]) -> bool:
    required = set("ABCD") if kind == "instruction" else set("ABCDEF")
    return kind in {"instruction", "preference"} and set(stages) == required


def _canonical_micro_draft_identity(
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the only resume-authoritative draft identity from the manifest."""
    kind = str(manifest_row.get("kind") or "")
    variant = str(manifest_row.get("variant") or "")
    chunk_id = str(manifest_row.get("chunk_id") or "")
    chunk_sha256 = str(manifest_row.get("chunk_sha256") or "")
    if (
        kind not in {"instruction", "preference"}
        or not variant
        or not chunk_id
        or not _SAFE_HASH.fullmatch(chunk_sha256)
    ):
        raise ValueError("micro manifest draft identity is incomplete")
    return {
        "chunk_id": chunk_id,
        "chunk_sha256": chunk_sha256,
        "kind": kind,
        "variant": variant,
        "repetition": int(manifest_row.get("repetition", 0)),
        "draft": {
            "provider": "local",
            "source_chunk_id": chunk_id,
        },
    }


_MICRO_PROVIDER_STAGE = re.compile(
    r"^staged_synthesis:micro_([A-F])"
    r"(?:_claim_(\d+))?(?:_attempt_(\d+))?(?:_[a-z0-9_]+)?$"
)


def _micro_provider_stage_identity(stage: Any) -> tuple[str, Optional[int], int]:
    """Parse the physical stage label into its logical family and slot."""
    match = _MICRO_PROVIDER_STAGE.fullmatch(str(stage or ""))
    if match is None:
        raise ValueError("micro provider stage identity is invalid")
    family = match.group(1)
    slot = int(match.group(2)) if match.group(2) is not None else None
    semantic_attempt = (
        int(match.group(3)) if match.group(3) is not None else 1
    )
    if (
        family in {"A", "C"}
        or semantic_attempt <= 0
        or (family == "B") != (slot is not None)
    ):
        raise ValueError("micro provider stage identity is invalid")
    return family, slot, semantic_attempt


def verify_micro_evidence_plane_rows(
    *,
    intent_rows: Iterable[Mapping[str, Any]],
    decision_contexts: Iterable[Mapping[str, Any]],
    http_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join logical calls to capture and transport without count conflation."""
    intents = [
        dict(row) for row in intent_rows
        if row.get("kind") != "dialect"
    ]
    parsed: list[tuple[str, Optional[int], int]] = []
    identities: list[tuple[Any, ...]] = []
    repair_groups: dict[tuple[str, str, Optional[int]], list[int]] = defaultdict(list)
    for row in intents:
        family, slot, semantic_attempt = _micro_provider_stage_identity(
            row.get("stage")
        )
        logical_attempt = int(row.get("logical_attempt", 0))
        suffix_attempt = re.search(
            r"_attempt_(\d+)(?:_|$)", str(row.get("stage") or "")
        )
        effective_attempt = (
            semantic_attempt if suffix_attempt is not None else logical_attempt
        )
        if (
            logical_attempt <= 0
            or (
                suffix_attempt is not None
                and logical_attempt != 1
            )
            or row.get("kind") != (
                "initial" if logical_attempt == 1 else "repair"
            )
            or not _SAFE_HASH.fullmatch(str(row.get("request_sha256") or ""))
        ):
            raise ValueError("micro logical intent identity is invalid")
        identity = (
            row.get("unit"), row.get("stage"), row.get("logical_attempt"),
            row.get("request_sha256"),
        )
        if identity in identities:
            raise ValueError("micro logical intent identity is duplicated")
        identities.append(identity)
        parsed.append((family, slot, semantic_attempt))
        repair_groups[(str(row.get("unit")), family, slot)].append(
            effective_attempt
        )
    if any(
        attempts != list(range(1, max(attempts) + 1))
        for attempts in repair_groups.values()
    ):
        raise ValueError("micro semantic repair sequence is not contiguous")

    decisions = []
    for context in decision_contexts:
        stage = context.get("intent_stage")
        if stage == "staged_synthesis:dialect_preflight":
            continue
        _micro_provider_stage_identity(stage)
        decisions.append((
            context.get("intent_unit"), stage,
            context.get("intent_logical_attempt"),
            context.get("intent_request_sha256"),
        ))
    if Counter(decisions) != Counter(identities):
        raise ValueError("micro DecisionCapture/intents are not bijective")

    started_rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    terminal_rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in http_rows:
        stage = row.get("stage")
        if stage == "staged_synthesis:dialect_preflight":
            continue
        _micro_provider_stage_identity(stage)
        identity = (
            row.get("unit"), stage, row.get("attempt"),
            row.get("request_sha256"),
        )
        target = (
            started_rows if row.get("event") == "http_attempt_started"
            else terminal_rows if row.get("event") == "http_attempt_terminal"
            else None
        )
        if target is None or identity in target:
            raise ValueError("micro HTTP attempt event is invalid or duplicated")
        target[identity] = row
    if set(started_rows) != set(identities) or set(terminal_rows) != set(identities):
        raise ValueError("micro HTTP/intents are not bijective")
    if any(family in {"A", "C"} for family, _, _ in parsed):
        raise ValueError("deterministic micro stage emitted provider traffic")

    by_stage = Counter(family for family, _, _ in parsed)
    repairs = Counter()
    for attempts_key, attempts in repair_groups.items():
        repairs[attempts_key[1]] += sum(value > 1 for value in attempts)
    stage_outcomes: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "logical_attempts": 0, "transport_started": 0,
            "transport_terminal": 0, "succeeded": 0,
            "failed": {
                "retryable_http": 0, "nonretryable_http": 0,
                "timeout": 0, "transport": 0, "abort": 0,
            },
            "repairs": 0, "recovered": 0, "unrecovered": 0,
            "client_usage": {
                "available": 0, "partial": 0, "unavailable": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0,
            },
            "latency": {
                "success_sum_seconds": 0.0,
                "failure_sum_seconds": 0.0,
            },
        }
    )
    group_outcomes: dict[tuple[str, str, Optional[int]], list[tuple[int, bool]]] = (
        defaultdict(list)
    )
    all_started_times = []
    all_terminal_times = []
    for index, identity in enumerate(identities):
        family, slot, semantic_attempt = parsed[index]
        intent = intents[index]
        suffix_attempt = re.search(r"_attempt_(\d+)(?:_|$)", intent["stage"])
        effective_attempt = (
            semantic_attempt if suffix_attempt is not None
            else int(intent["logical_attempt"])
        )
        values = stage_outcomes[family]
        values["logical_attempts"] += 1
        values["transport_started"] += 1
        values["transport_terminal"] += 1
        started = started_rows[identity]
        terminal = terminal_rows[identity]
        started_at = float(started.get("monotonic_seconds", 0.0))
        terminal_at = float(terminal.get("monotonic_seconds", 0.0))
        if terminal_at < started_at:
            raise ValueError("micro HTTP latency chronology is invalid")
        latency = terminal_at - started_at
        all_started_times.append(started_at)
        all_terminal_times.append(terminal_at)
        success = (
            isinstance(terminal.get("http_status"), int)
            and 200 <= int(terminal["http_status"]) < 300
            and not terminal.get("exception_class")
        )
        if success:
            values["succeeded"] += 1
            values["latency"]["success_sum_seconds"] += latency
        else:
            values["latency"]["failure_sum_seconds"] += latency
            exception = str(terminal.get("exception_class") or "").lower()
            status = terminal.get("http_status")
            if "timeout" in exception:
                reason = "timeout"
            elif "abort" in exception or "disconnect" in exception:
                reason = "abort"
            elif isinstance(status, int):
                reason = (
                    "retryable_http"
                    if status in {408, 409, 425, 429} or status >= 500
                    else "nonretryable_http"
                )
            else:
                reason = "transport"
            values["failed"][reason] += 1
        usage = terminal.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        present = [
            key for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(usage.get(key), int)
            and not isinstance(usage.get(key), bool)
            and usage.get(key) >= 0
        ]
        coverage = (
            "available" if len(present) == 3
            else "partial" if present else "unavailable"
        )
        values["client_usage"][coverage] += 1
        for key in present:
            values["client_usage"][key] += int(usage[key])
        group_outcomes[(str(intent["unit"]), family, slot)].append(
            (effective_attempt, success)
        )
    for family, count in repairs.items():
        stage_outcomes[family]["repairs"] = count
    for (_unit, family, _slot), outcomes in group_outcomes.items():
        ordered = [success for _, success in sorted(outcomes)]
        if any(not success for success in ordered):
            if ordered[-1]:
                stage_outcomes[family]["recovered"] += 1
            else:
                stage_outcomes[family]["unrecovered"] += 1
    body = {
        "schema_version": "ed4all.micro-evidence-plane.v1",
        "logical_calls": dict(sorted(by_stage.items())),
        "repairs": dict(sorted(repairs.items())),
        "transport_attempts": dict(sorted(by_stage.items())),
        "stages": {
            stage: values for stage, values in sorted(stage_outcomes.items())
        },
        "wall_seconds": (
            max(all_terminal_times) - min(all_started_times)
            if all_started_times else 0.0
        ),
    }
    return {**body, "report_sha256": _digest(body)}


def apply_micro_transport_outcomes(
    summary: Mapping[str, Any], evidence: Mapping[str, Any],
    *, results: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Project the joined attempt authority into summary telemetry."""
    projected = dict(summary)
    metrics = {
        str(stage): dict(values)
        for stage, values in (summary.get("microstage_metrics") or {}).items()
    }
    for stage, outcomes in (evidence.get("stages") or {}).items():
        values = metrics.setdefault(str(stage), {})
        usage = outcomes["client_usage"]
        latency = outcomes["latency"]
        logical = int(outcomes["logical_attempts"])
        values.update({
            # Backward-compatible field with an explicit, non-ambiguous
            # definition. New consumers should use logical_attempts.
            "calls": logical,
            "calls_semantics": "logical_attempts",
            "logical_attempts": logical,
            "transport_started": int(outcomes["transport_started"]),
            "transport_terminal": int(outcomes["transport_terminal"]),
            "succeeded": int(outcomes["succeeded"]),
            "failed": dict(outcomes["failed"]),
            "repairs": int(outcomes["repairs"]),
            "recovered": int(outcomes["recovered"]),
            "unrecovered": int(outcomes["unrecovered"]),
            "content_rejections": 0,
            "client_usage": dict(usage),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            "success_latency_sum_seconds": float(
                latency["success_sum_seconds"]
            ),
            "failure_latency_sum_seconds": float(
                latency["failure_sum_seconds"]
            ),
            "latency_seconds": float(latency["success_sum_seconds"])
            + float(latency["failure_sum_seconds"]),
        })
        all_usage = (
            int(usage["available"]) == int(outcomes["transport_terminal"])
            and int(usage["partial"]) == 0
            and int(usage["unavailable"]) == 0
        )
        values["client_sum_tokens_per_second"] = (
            values["total_tokens"] / values["latency_seconds"]
            if all_usage and values["latency_seconds"] else None
        )
        values["client_wall_tokens_per_second"] = (
            values["total_tokens"] / float(evidence["wall_seconds"])
            if all_usage and evidence.get("wall_seconds") else None
        )
    by_microstage = dict(summary.get("by_microstage") or {})
    for stage, outcomes in (evidence.get("stages") or {}).items():
        by_microstage[str(stage)] = int(outcomes["logical_attempts"])
    projected["by_microstage"] = dict(sorted(by_microstage.items()))
    for row in results or ():
        details = row.get("error_details")
        details = details if isinstance(details, Mapping) else {}
        if details.get("terminal_content_rejection") is not True:
            continue
        match = re.fullmatch(r"micro_([A-F])", str(details.get("stage") or ""))
        if match and match.group(1) in metrics:
            metrics[match.group(1)]["content_rejections"] += 1
    projected["microstage_metrics"] = metrics
    projected["transport_outcomes"] = {
        "schema_version": evidence.get("schema_version"),
        "evidence_report_sha256": evidence.get("report_sha256"),
        "wall_seconds": evidence.get("wall_seconds"),
        "stages": evidence.get("stages"),
    }
    return projected


def verify_micro_journals(
    cell_dir: Path, *, summary: Mapping[str, Any],
    preflight: Mapping[str, Any],
    results: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Verify every resume journal and bind authoritative micro telemetry."""
    from Trainforge.generators.staged_synthesis_micro import MicroResumeStore

    synthesis_identity = preflight.get("synthesis_contract")
    expected_contract = synthesis_contract_identity(MICRO_SYNTHESIS_CONTRACT)
    budget_contract = (
        expected_contract.get("component_hashes") or {}
    ).get("stage_token_budget")
    if (
        not isinstance(budget_contract, Mapping)
        or budget_contract.get("version") != "micro-stage-max-tokens.v1"
        or not isinstance(budget_contract.get("max_tokens"), Mapping)
    ):
        raise ValueError("micro stage token budgets are not fingerprinted")
    stage_token_budgets = dict(budget_contract["max_tokens"])
    expected = {
        "contract": expected_contract["fingerprint"],
        "execution_fingerprint": preflight.get("run_contract_sha256"),
        "model": (preflight.get("telemetry_identity") or {}).get(
            "expected_model"
        ),
        "pilot_run_id": preflight.get("pilot_run_id"),
        "cell_id": preflight.get("cell_id"),
        "manifest_sha256": preflight.get("manifest_sha256"),
        "eligibility_sha256": preflight.get("eligibility_sha256"),
    }
    if synthesis_identity != expected_contract or any(
        not isinstance(value, str) or not value
        for value in expected.values()
    ):
        raise ValueError("micro preflight identity is incomplete")
    manifest_rows = [
        json.loads(line) for line in (
            cell_dir.parent / "manifest.jsonl"
        ).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    manifest_drafts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        draft_identity = _canonical_micro_draft_identity(row)
        key = (draft_identity["chunk_sha256"], draft_identity["kind"])
        authoritative = {
            "identity": draft_identity,
            "sha256": _digest(draft_identity),
        }
        if authoritative in manifest_drafts[key]:
            raise ValueError("micro manifest draft identity is duplicated")
        manifest_drafts[key].append(authoritative)
    journal_root_value = preflight.get("micro_journal_root")
    if (
        not isinstance(journal_root_value, str)
        or not journal_root_value
        or Path(journal_root_value).is_absolute()
        or ".." in Path(journal_root_value).parts
    ):
        raise ValueError("micro preflight journal root is invalid")
    journal_root = cell_dir / journal_root_value
    journals = []
    terminal_counts: Counter[str] = Counter()
    terminal_slots: Counter[str] = Counter()
    terminal_kinds: Counter[str] = Counter()
    journal_result_identities: Counter[tuple[Any, ...]] = Counter()
    journal_rejections: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted(journal_root.glob("*.jsonl")):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not lines:
            raise ValueError("micro resume journal is empty")
        first = json.loads(lines[0])
        fingerprint = first.get("contract_fingerprint")
        store_identity = first.get("store_identity")
        if not isinstance(fingerprint, str) or not _SAFE_HASH.fullmatch(
            fingerprint
        ):
            raise ValueError("micro resume journal fingerprint is invalid")
        if (
            not isinstance(store_identity, Mapping)
            or any(store_identity.get(key) != value for key, value in expected.items())
            or store_identity.get("kind") not in {"instruction", "preference"}
            or (
                store_identity.get("chunk_sha256"),
                store_identity.get("kind"),
            ) not in manifest_drafts
            or not _SAFE_HASH.fullmatch(str(store_identity.get("draft_sha256") or ""))
        ):
            raise ValueError("micro resume journal store identity is foreign")
        authoritative_drafts = manifest_drafts[(
            store_identity["chunk_sha256"], store_identity["kind"],
        )]
        if not any(
            store_identity.get("draft_identity") == candidate["identity"]
            and store_identity.get("draft_sha256") == candidate["sha256"]
            for candidate in authoritative_drafts
        ):
            raise ValueError("micro resume journal draft identity is stale or foreign")
        input_identity = {
            key: store_identity.get(key) for key in (
                "manifest_sha256", "eligibility_sha256", "chunk_sha256",
                "draft_sha256",
            )
        }
        if store_identity.get("input_fingerprint") != _digest(input_identity):
            raise ValueError("micro resume journal input identity is invalid")
        if fingerprint != _digest(store_identity):
            raise ValueError("micro resume journal fingerprint is inconsistent")
        terminal = MicroResumeStore(
            path, fingerprint=fingerprint, store_identity=store_identity,
        ).load(allow_failure_outcomes=True)
        parsed = list(map(json.loads, lines))
        terminal_rows = [
            row for row in parsed if row.get("state") == "terminal"
        ]
        rejection_rows = [
            row for row in terminal_rows
            if (row.get("outcome") or "success") == "content_rejected"
        ]
        if len(rejection_rows) > 1:
            raise ValueError("micro journal has multiple terminal rejections")
        if rejection_rows and terminal_rows[-1] is not rejection_rows[0]:
            raise ValueError("micro terminal rejection is not the final intent")
        if any(
            (row.get("outcome") or "success")
            not in {"success", "content_rejected"}
            for row in terminal_rows
        ):
            raise ValueError("micro journal terminal outcome is invalid")
        stages = {str(row.get("stage")) for row in terminal_rows}
        if (
            not rejection_rows
            and not _micro_stage_coverage_complete(store_identity["kind"], stages)
        ):
            raise ValueError("micro journal terminal stage coverage is incomplete")
        seen_units: set[str] = set()
        for row in terminal_rows:
            stage = str(row.get("stage"))
            slot = row.get("slot")
            unit = str(row.get("unit") or "")
            expected_unit = f"{store_identity['kind']}:{stage}"
            if stage == "B":
                if not isinstance(slot, int) or slot < 0:
                    raise ValueError("micro B terminal slot identity is invalid")
                expected_unit += f":slot-{slot}"
                terminal_slots[str(slot)] += 1
            elif slot is not None:
                raise ValueError("non-B micro terminal cannot have a claim slot")
            if unit != expected_unit or unit in seen_units:
                raise ValueError("micro journal has duplicate or invalid terminal unit")
            seen_units.add(unit)
            terminal_counts[stage] += 1
        terminal_kinds[store_identity["kind"]] += 1
        draft_identity = store_identity["draft_identity"]
        journal_result_identities[tuple(
            draft_identity.get(field) for field in (
                "chunk_id", "kind", "variant", "repetition",
            )
        )] += 1
        result_identity = tuple(
            draft_identity.get(field) for field in (
                "chunk_id", "kind", "variant", "repetition",
            )
        )
        if rejection_rows:
            evidence = rejection_rows[0].get("terminal_evidence")
            if not isinstance(evidence, Mapping):
                raise ValueError("micro terminal rejection lacks evidence")
            journal_rejections[result_identity] = {
                "reason_code": evidence.get("reason_code"),
                "evidence_sha256": evidence.get("evidence_sha256"),
                "stage": rejection_rows[0].get("stage"),
            }
        journals.append({
            "path": path.relative_to(cell_dir).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "terminal_units": len(terminal_rows),
            "terminal_outcome": (
                "content_rejected" if rejection_rows else "success"
            ),
        })
    if not journals:
        raise ValueError("micro contract produced no resume journals")
    expected_stages = set(terminal_counts)
    # A and Q are deterministic logical journal stages. They have no provider
    # request, call intent, or transport-token metric and therefore must not be
    # fabricated into the transport telemetry projection.
    by_microstage = dict(summary.get("by_microstage") or {})
    by_claim_slot = dict(summary.get("by_claim_slot") or {})
    metrics = dict(summary.get("microstage_metrics") or {})
    expected_telemetry_stages = expected_stages - {"Q"}
    if "A" not in by_microstage and "A" not in metrics:
        expected_telemetry_stages.discard("A")
    if (
        set(by_microstage) != expected_telemetry_stages
        or set(metrics) != expected_telemetry_stages
    ):
        raise ValueError("micro telemetry does not match verified journal stages")
    intent_path = cell_dir / "call-intents.jsonl"
    if not intent_path.is_file():
        raise ValueError("micro authoritative call-intent manifest is missing")
    intent_counts: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    intent_rows: list[dict[str, Any]] = []
    for line in intent_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        intent = json.loads(line)
        intent_rows.append(intent)
        match = re.search(r"micro_([A-F])(?:_|$)", str(intent.get("stage") or ""))
        if not match:
            continue
        stage = match.group(1)
        if (
            stage == "C"
            or stage not in stage_token_budgets
            or int(intent.get("max_tokens", -1))
            != int(stage_token_budgets[stage])
        ):
            raise ValueError("micro intent token budget differs from contract")
        intent_counts[stage] += 1
        attempt_match = re.search(
            r"_attempt_(\d+)", str(intent.get("stage") or "")
        )
        if (
            int(intent.get("logical_attempt", 0)) > 1
            or (attempt_match and int(attempt_match.group(1)) > 1)
        ):
            repair_counts[stage] += 1
    evidence_plane = None
    http_path = cell_dir / "http_attempts.jsonl"
    capture_paths = sorted(
        (cell_dir / "audit" / "runtime/training-captures").rglob("*.jsonl")
    )
    if http_path.is_file() or capture_paths:
        if not http_path.is_file() or len(capture_paths) != 1:
            raise ValueError("micro evidence plane artifacts are incomplete")
        http_rows = [
            json.loads(line)
            for line in http_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        decision_contexts = []
        for line in capture_paths[0].read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("decision_type") != "synthesis_provider_call":
                continue
            try:
                context = json.loads(event.get("context") or "")
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "micro provider DecisionCapture context is invalid"
                ) from exc
            if not isinstance(context, Mapping):
                raise ValueError(
                    "micro provider DecisionCapture context is invalid"
                )
            decision_contexts.append(context)
        evidence_plane = verify_micro_evidence_plane_rows(
            intent_rows=intent_rows,
            decision_contexts=decision_contexts,
            http_rows=http_rows,
        )
        expected_transport = apply_micro_transport_outcomes(
            summary, evidence_plane
        )["transport_outcomes"]
        if summary.get("transport_outcomes") != expected_transport:
            raise ValueError(
                "micro transport outcome summary differs from joined evidence"
            )
    expected_intent_projection = set(intent_counts)
    if "C" in expected_telemetry_stages:
        expected_intent_projection.add("C")
    if expected_intent_projection != expected_telemetry_stages:
        raise ValueError("micro intent stages differ from terminal units")
    for stage, count in terminal_counts.items():
        if stage not in expected_telemetry_stages:
            continue
        values = metrics.get(stage)
        if not isinstance(values, Mapping):
            raise ValueError("microstage metrics are invalid")
        calls = int(values.get("calls", -1))
        completed_units = int(values.get("completed_units", -1))
        deterministic = int(values.get("deterministic_events", 0))
        if stage == "C":
            if (
                calls != 0 or deterministic != count or completed_units != count
                or count <= 0
            ):
                raise ValueError("deterministic C telemetry is invalid")
        elif (
            calls != intent_counts[stage]
            or calls < count
            or completed_units != count
            or deterministic != 0
        ):
            raise ValueError("transport microstage telemetry is invalid")
        if int(by_microstage.get(stage, -1)) != (
            count if stage == "C" else calls
        ):
            raise ValueError("microstage call counts are invalid")
    if by_claim_slot != dict(sorted(terminal_slots.items())):
        raise ValueError("micro claim-slot result linkage is invalid")
    if results is not None:
        result_rows = list(results)
        result_identities = Counter(tuple(row.get(field) for field in (
            "chunk_id", "kind", "variant", "repetition",
        )) for row in result_rows)
        if result_identities != journal_result_identities:
            raise ValueError("micro journal/result identity linkage is invalid")
        result_by_identity = {
            tuple(row.get(field) for field in (
                "chunk_id", "kind", "variant", "repetition",
            )): row
            for row in result_rows
        }
        for identity, rejection in journal_rejections.items():
            result = result_by_identity.get(identity)
            if (
                not isinstance(result, Mapping)
                or result.get("accepted") is not False
                or result.get("error_code") != rejection["reason_code"]
                or not _SAFE_HASH.fullmatch(
                    str(rejection["evidence_sha256"] or "")
                )
            ):
                raise ValueError(
                    "micro terminal rejection/result linkage is invalid"
                )
        recomputed = summarize(result_rows)
        if evidence_plane is not None:
            recomputed = apply_micro_transport_outcomes(
                recomputed, evidence_plane, results=result_rows,
            )
        for key in ("by_microstage", "by_claim_slot", "microstage_metrics"):
            if recomputed.get(key) != summary.get(key):
                raise ValueError("micro telemetry differs from source results")
    verified_metrics = {
        stage: {
            **dict(values),
            "repairs": int(repair_counts.get(stage, 0)),
            "repair_count_source": "call-intent-ledger",
        }
        for stage, values in metrics.items()
    }
    completion_order_evidence = None
    raw_checkpoint_path = cell_dir / "checkpoint.jsonl"
    if raw_checkpoint_path.is_file():
        observed_identities = []
        for ordinal, line in enumerate(
            raw_checkpoint_path.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            observed_identities.append({
                "completion_ordinal": ordinal,
                "identity": {
                    field: row.get(field) for field in (
                        "row_id", "chunk_id", "kind", "variant", "repetition",
                    )
                },
            })
        completion_order_evidence = {
            "authority": "telemetry_only",
            "raw_checkpoint_sha256": hashlib.sha256(
                raw_checkpoint_path.read_bytes()
            ).hexdigest(),
            "completion_count": len(observed_identities),
            "ordered_identity_sha256": _digest(observed_identities),
            "observed": observed_identities,
        }
    body = {
        "schema_version": "ed4all.staged-synthesis-micro-verification.v1",
        "contract": expected_contract,
        "journals": journals,
        "terminal_outcomes": dict(sorted(Counter(
            row["terminal_outcome"] for row in journals
        ).items())),
        "by_microstage": by_microstage,
        "completed_by_microstage": dict(sorted(terminal_counts.items())),
        "by_claim_slot": by_claim_slot,
        "microstage_metrics": verified_metrics,
        "evidence_plane": evidence_plane,
        "completion_order_evidence": completion_order_evidence,
    }
    if (
        not body["by_microstage"]
        or not body["by_claim_slot"]
        or not body["microstage_metrics"]
    ):
        raise ValueError("micro telemetry is incomplete")
    return {**body, "report_sha256": _digest(body)}


def planned_benchmark_cells(
    stop_after_concurrency: Optional[int] = None,
) -> list[tuple[int, str]]:
    """Return the legacy matrix or an explicitly bounded prefix."""
    cells = [
        (value, f"c{value}") for value in BENCHMARK_CONCURRENCY_CELLS
    ]
    if stop_after_concurrency is None:
        return cells
    if stop_after_concurrency not in BENCHMARK_CONCURRENCY_CELLS:
        raise ValueError(
            "stop_after_concurrency must name a planned benchmark cell"
        )
    stop_index = BENCHMARK_CONCURRENCY_CELLS.index(stop_after_concurrency)
    return cells[:stop_index + 1]


def qualification_route_binding(
    *,
    explicit_base_url: Any,
    explicit_model: Any,
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Require explicit qualification routing and exact shell corroboration."""
    base_url = str(explicit_base_url or "").rstrip("/")
    model = str(explicit_model or "")
    env_url = str(environ.get("LOCAL_SYNTHESIS_BASE_URL") or "").rstrip("/")
    env_model = str(environ.get("LOCAL_SYNTHESIS_MODEL") or "")
    split = urlsplit(base_url)
    if (
        not base_url
        or not model
        or split.scheme not in {"http", "https"}
        or not split.hostname
        or env_url != base_url
        or env_model != model
    ):
        raise ValueError(
            "completion-cap qualification route binding is missing or conflicts "
            "with the explicit shell route"
        )
    return {"base_url": base_url, "served_model": model}


def remaining_cell_budget(
    *, cell_started: float, global_deadline: float, now: float,
) -> float:
    """Budget including provider construction and serial dialect preflight."""
    return max(
        0.0,
        min(
            cell_started + BENCHMARK_CELL_LIMIT_SECONDS,
            global_deadline,
        ) - now,
    )


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    text = value if isinstance(value, str) else _stable(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verified_audit_report(report: Mapping[str, Any]) -> bool:
    """Validate the verifier's structured, self-hashed decision-audit report."""
    if not isinstance(report, Mapping) or report.get("status") != "accepted":
        return False
    report_hash = report.get("report_sha256")
    core = {key: value for key, value in report.items() if key != "report_sha256"}
    return (
        isinstance(report_hash, str)
        and _SAFE_HASH.fullmatch(report_hash) is not None
        and report_hash
        == hashlib.sha256(
            (json.dumps(
                core, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ) + "\n").encode("utf-8")
        ).hexdigest()
        and report.get("capture_closed") is True
        and report.get("provider_events_expected")
        == report.get("provider_events_observed")
        == report.get("provider_identities_verified")
        and not report.get("errors")
    )


_STAGE_VALUES = {
    "plan_sft", "plan_dpo", "sft_realization",
    "dpo_chosen_realization", "dpo_rejected_realization", "post_validation",
    "micro_A", "micro_B", "micro_C", "micro_D", "micro_E", "micro_F",
}
_SAFE_CODE = __import__("re").compile(r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,127}$")
_SAFE_REF = __import__("re").compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_HASH = __import__("re").compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER = __import__("re").compile(r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,127}$")
_UNSAFE_VALUE = __import__("re").compile(
    r"(?i)(authorization|bearer\s+\S+|api[_-]?key|secret|password|cookie|"
    r"(?:^|\s)(?:/[^\s]+|[A-Za-z]:\\\\|\\\\\\\\))"
)


def _sanitize_error_details(value: Any, *, _nested: bool = False) -> Any:
    """Validate each diagnostic field against its own non-secret schema."""
    if not isinstance(value, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key).lower()
        if key in {"stage", "validator_stage"}:
            if isinstance(item, str) and item in _STAGE_VALUES:
                cleaned[str(raw_key)] = item
        elif key in {"code", "exception_code"}:
            if isinstance(item, str) and _SAFE_CODE.fullmatch(item):
                cleaned[str(raw_key)] = item
        elif key in {"validation_error", "validator_reason"}:
            if (
                isinstance(item, str) and len(item) <= 500
                and not _UNSAFE_VALUE.search(item)
                and not any(ord(char) < 32 and char not in "\t\n" for char in item)
            ):
                cleaned[str(raw_key)] = item
        elif key in {"prompt_ref", "response_ref"}:
            if isinstance(item, str) and _SAFE_REF.fullmatch(item):
                cleaned[str(raw_key)] = item
        elif key.endswith("_sha256") or key in {"sha256", "contract_sha256"}:
            if isinstance(item, str) and _SAFE_HASH.fullmatch(item):
                cleaned[str(raw_key)] = item
        elif key in {
            "terminal_content_rejection", "integrity_failure",
            "quality_rejection", "matched",
        }:
            if isinstance(item, bool):
                cleaned[str(raw_key)] = item
        elif key in {
            "reasoning_bytes", "attempt", "count", "bytes", "claim_index",
            "unit_index", "validator_attempt",
        }:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                cleaned[str(raw_key)] = item
        elif key in {"entailment", "contradiction"}:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                cleaned[str(raw_key)] = float(item)
        elif key in {"configured_model", "payload_model"}:
            if isinstance(item, str) and _SAFE_IDENTIFIER.fullmatch(item):
                cleaned[str(raw_key)] = item
        elif key in {"required_keys", "present_keys"} and isinstance(item, list):
            identifiers = [
                entry for entry in item
                if isinstance(entry, str) and _SAFE_IDENTIFIER.fullmatch(entry)
            ]
            if len(identifiers) == len(item):
                cleaned[str(raw_key)] = identifiers
        elif key == "nli_scores" and isinstance(item, list):
            rows = []
            for row in item[:100]:
                if not isinstance(row, Mapping):
                    continue
                safe_row = _sanitize_error_details(row, _nested=True)
                if safe_row and set(safe_row) <= {
                    "claim_index", "unit_index", "entailment",
                    "contradiction", "matched",
                }:
                    rows.append(safe_row)
            cleaned[str(raw_key)] = rows
    return cleaned


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "").strip()


def _word_bucket(chunk: Mapping[str, Any]) -> str:
    words = len(str(chunk.get("text") or "").split())
    return "short" if words < 250 else ("medium" if words < 600 else "long")


def _stratum(chunk: Mapping[str, Any]) -> tuple[str, str, str]:
    text = str(chunk.get("text") or "").lower()
    misconception = any(
        marker in text for marker in ("misconception", "incorrect", "wrong turn")
    )
    return (
        str(chunk.get("bloom_level") or "unknown").lower(),
        _word_bucket(chunk),
        "misconception" if misconception else "ordinary",
    )


def select_cohort(chunks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select 28 real chunks by deterministic round-robin over content strata."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in chunks:
        chunk = dict(source)
        if _chunk_id(chunk) and str(chunk.get("text") or "").strip():
            groups[_stratum(chunk)].append(chunk)
    for rows in groups.values():
        rows.sort(key=lambda row: (_digest(row), _chunk_id(row)))
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < COHORT_SIZE and keys:
        remaining = []
        for key in keys:
            if groups[key] and len(selected) < COHORT_SIZE:
                selected.append(groups[key].pop(0))
            if groups[key]:
                remaining.append(key)
        keys = remaining
    if len(selected) != COHORT_SIZE:
        raise ValueError(f"pilot requires at least {COHORT_SIZE} eligible chunks")
    return selected


def load_objectives(path: Path) -> dict[str, dict[str, Any]]:
    """Load canonical objectives from a list, map, or common wrapper shape."""
    value = json.loads(path.read_text(encoding="utf-8"))

    def normalized_id(raw: Any) -> str:
        return str(raw or "").strip().lower()

    def from_records(records: Any, *, shape: str) -> dict[str, dict[str, Any]]:
        if not isinstance(records, list):
            raise ValueError(f"objective artifact {shape} must be a list")
        loaded: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(records):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"objective artifact {shape}[{index}] must be an object"
                )
            objective_id = normalized_id(
                item.get("id") or item.get("objective_id")
            )
            if not objective_id:
                raise ValueError(
                    f"objective artifact {shape}[{index}] has no objective id"
                )
            record = dict(item)
            prior = loaded.get(objective_id)
            if prior is not None and prior != record:
                raise ValueError(
                    f"objective artifact has conflicting duplicate {objective_id}"
                )
            loaded[objective_id] = record
        return loaded

    if isinstance(value, Mapping):
        grouped = []
        for key in ("terminal_outcomes", "component_objectives"):
            if key in value:
                if not isinstance(value[key], list):
                    raise ValueError(f"objective artifact {key} must be a list")
                grouped.extend(value[key])
        if grouped:
            return from_records(grouped, shape="terminal/component outcomes")
        else:
            for key in (
                "learning_outcomes", "objectives",
                "learning_objectives", "outcomes",
            ):
                if key in value:
                    return from_records(value[key], shape=key)
            else:
                loaded: dict[str, dict[str, Any]] = {}
                for key, item in value.items():
                    if not isinstance(item, Mapping):
                        raise ValueError(
                            "objective artifact map values must be objects"
                        )
                    objective_id = normalized_id(key)
                    if not objective_id:
                        raise ValueError("objective artifact map has empty id")
                    record = dict(item)
                    declared = normalized_id(
                        record.get("id") or record.get("objective_id")
                    )
                    if declared and declared != objective_id:
                        raise ValueError(
                            "objective artifact map key conflicts with record id"
                        )
                    prior = loaded.get(objective_id)
                    if prior is not None and prior != record:
                        raise ValueError(
                            "objective artifact has conflicting duplicate "
                            f"{objective_id}"
                        )
                    loaded[objective_id] = record
                return loaded
    return from_records(value, shape="list")


def apply_runtime_focus(
    chunk: Mapping[str, Any],
    objectives: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach the exact runtime objective used by production synthesis."""
    from Trainforge.synthesis.synthesis_eligibility import (
        focus_chunk_on_canonical_objective,
    )
    result = focus_chunk_on_canonical_objective(
        chunk, seed=0, objectives=objectives,
    )
    refs = [str(ref).lower() for ref in result.get("learning_outcome_refs") or []]
    existing = result.get("synthesis_focus_objective")
    focus_id = (
        str(existing.get("id")).lower()
        if isinstance(existing, Mapping) and existing.get("id")
        else (refs[0] if len(refs) == 1 else "")
    )
    focus = objectives.get(focus_id)
    if focus is None and isinstance(existing, Mapping):
        focus = existing
    if not focus_id or focus is None:
        reason = str(
            result.get("synthesis_focus_skip_reason")
            or "runtime_focus_unresolvable"
        )
        raise ValueError(reason)
    card = objective_card(focus)
    result["synthesis_focus_objective"] = deepcopy(dict(focus))
    result["learning_outcome_refs"] = [card["id"]]
    result["bloom_level"] = card["bloom_level"]
    return result


def variant_chunk(chunk: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Apply a 2×2 semantic-window × rich-objective ablation."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown pilot variant {variant!r}")
    result = deepcopy(dict(chunk))
    semantic = variant in {VARIANTS[1], VARIANTS[3]}
    rich = variant in {VARIANTS[2], VARIANTS[3]}
    if not semantic:
        result.pop("html", None)
    if not rich:
        focus = objective_card(result["synthesis_focus_objective"])
        result["synthesis_focus_objective"] = {
            key: focus[key] for key in ("id", "statement", "bloom_level")
        }
    return result


def build_pilot_rows(
    chunks: Iterable[Mapping[str, Any]],
    *,
    objectives: Optional[Mapping[str, Mapping[str, Any]]] = None,
    repetitions: int = 1,
) -> list[dict[str, Any]]:
    """Build a deterministic Latin-square execution manifest."""
    rows, _audit = build_pilot_manifest(
        chunks, objectives=objectives, repetitions=repetitions,
    )
    return rows


def build_pilot_manifest(
    chunks: Iterable[Mapping[str, Any]],
    *,
    objectives: Optional[Mapping[str, Mapping[str, Any]]] = None,
    repetitions: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build production-eligible rows plus an explicit exclusion audit."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    from Trainforge.synthesis.synthesis_eligibility import pair_eligibility
    from Trainforge.generators.staged_synthesis_micro import (
        micro_preference_eligibility,
    )
    exclusions: Counter[str] = Counter()
    by_kind: dict[str, list[dict[str, Any]]] = {
        "instruction": [], "preference": [],
    }
    preference_window_hashes: dict[str, str] = {}
    total_chunks = 0
    for raw_chunk in chunks:
        total_chunks += 1
        try:
            focused = apply_runtime_focus(raw_chunk, objectives or {})
        except ValueError as exc:
            exclusions[f"focus:{str(exc).rsplit(': ', 1)[-1]}"] += 1
            continue
        for kind in ("instruction", "preference"):
            eligibility = pair_eligibility(focused, kind=kind)
            if eligibility.eligible and kind == "preference":
                exact = micro_preference_eligibility(
                    focused, focus=focused["synthesis_focus_objective"],
                )
                if not exact["eligible"]:
                    exclusions[f"preference:{exact['reason']}"] += 1
                    continue
                preference_window_hashes[_chunk_id(focused)] = str(
                    exact["telemetry"]["evidence_window_sha256"]
                )
            if eligibility.eligible:
                by_kind[kind].append(focused)
            else:
                exclusions[f"{kind}:{eligibility.reason}"] += 1

    def take_stratified(
        candidates: list[dict[str, Any]], count: int, used: set[str],
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            if _chunk_id(candidate) not in used:
                groups[_stratum(candidate)].append(candidate)
        for values in groups.values():
            values.sort(key=lambda row: (_digest(row), _chunk_id(row)))
        selected = []
        keys = sorted(groups)
        while len(selected) < count and keys:
            remaining = []
            for key in keys:
                if groups[key] and len(selected) < count:
                    selected.append(groups[key].pop(0))
                if groups[key]:
                    remaining.append(key)
            keys = remaining
        if len(selected) != count:
            raise ValueError(
                f"pilot requires {count} unused production-eligible candidates"
            )
        used.update(_chunk_id(row) for row in selected)
        return selected

    # Preference eligibility is narrower, so reserve its unique chunks first.
    used_ids: set[str] = set()
    selected_by_kind = {
        "preference": take_stratified(by_kind["preference"], 14, used_ids),
        "instruction": take_stratified(by_kind["instruction"], 14, used_ids),
    }
    cohort = [
        (kind, chunk)
        for kind in ("instruction", "preference")
        for chunk in selected_by_kind[kind]
    ]
    cohort.sort(key=lambda item: (_digest(item[1]), item[0], _chunk_id(item[1])))
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for cohort_index, (kind, chunk) in enumerate(cohort):
            rotation = (cohort_index + repetition) % len(VARIANTS)
            for order_index in range(len(VARIANTS)):
                variant = VARIANTS[(rotation + order_index) % len(VARIANTS)]
                candidate = variant_chunk(chunk, variant)
                rows.append({
                    "pilot_version": PILOT_VERSION,
                    "cohort_index": cohort_index,
                    "repetition": repetition,
                    "order_index": order_index,
                    "chunk_id": _chunk_id(candidate),
                    "stratum": list(_stratum(candidate)),
                    "kind": kind,
                    "variant": variant,
                    "chunk_sha256": _digest(candidate),
                    "focus_objective": objective_card(
                        candidate["synthesis_focus_objective"]
                    ),
                    "_chunk": candidate,
                })
    audit = {
        "selection_version": COHORT_SELECTION_VERSION,
        "preference_predicate_sha256": hashlib.sha256(
            inspect.getsource(micro_preference_eligibility).encode("utf-8")
        ).hexdigest(),
        "preference_window_universe_sha256": _digest(
            dict(sorted(preference_window_hashes.items()))
        ),
        "input_chunks": total_chunks,
        "selected_unique_chunks": len(used_ids),
        "selected_by_kind": {
            kind: len(values) for kind, values in selected_by_kind.items()
        },
        "eligible_candidates_by_kind": {
            kind: len(values) for kind, values in by_kind.items()
        },
        "candidate_universe_sha256_by_kind": {
            kind: _digest([
                {
                    "chunk_id": _chunk_id(row),
                    "chunk_sha256": _digest(row),
                    "stratum": list(_stratum(row)),
                }
                for row in sorted(
                    values, key=lambda item: (_stratum(item), _digest(item))
                )
            ])
            for kind, values in by_kind.items()
        },
        "selected_universe_sha256_by_kind": {
            kind: _digest([
                {
                    "chunk_id": _chunk_id(row),
                    "chunk_sha256": _digest(row),
                    "stratum": list(_stratum(row)),
                }
                for row in values
            ])
            for kind, values in selected_by_kind.items()
        },
        "exclusions": dict(sorted(exclusions.items())),
    }
    return rows, audit


def build_benchmark_rows(
    chunks: Iterable[Mapping[str, Any]],
    *,
    objectives: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the fixed D-only 8-unit matrix cohort: four SFT and four DPO."""
    rows, audit = build_pilot_manifest(
        chunks, objectives=objectives, repetitions=1,
    )
    selected = []
    for kind in ("instruction", "preference"):
        candidates = [
            row for row in rows
            if row["variant"] == "D_production_contract"
            and row["kind"] == kind
        ]
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            groups[tuple(candidate["stratum"])].append(candidate)
        for values in groups.values():
            values.sort(key=lambda row: (_digest(row), row["chunk_id"]))
        balanced = []
        keys = sorted(groups)
        while len(balanced) < 4 and keys:
            remaining = []
            for key in keys:
                if groups[key] and len(balanced) < 4:
                    balanced.append(groups[key].pop(0))
                if groups[key]:
                    remaining.append(key)
            keys = remaining
        selected.extend(balanced)
    selected.sort(
        key=lambda row: (
            _digest(f"{row['chunk_id']}|{row['kind']}"),
            row["kind"],
        )
    )
    if len(selected) != 8:
        raise ValueError("benchmark requires four eligible SFT and four DPO rows")
    audit = {
        **audit,
        "benchmark_rows": len(selected),
        "benchmark_by_kind": dict(Counter(row["kind"] for row in selected)),
        "benchmark_variant": "D_production_contract",
    }
    return selected, audit


def load_frozen_d_c1_manifest(
    frozen_manifest_path: Path,
    *,
    expected_sha256: str,
    expected_ordered_identity_sha256: str,
    chunks: Iterable[Mapping[str, Any]],
    objectives: Mapping[str, Mapping[str, Any]],
    chunks_path: Path,
    objectives_path: Path,
    frozen_eligibility_path: Path,
    expected_eligibility_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rehydrate an already-adjudicated D-only C1 cohort, fail closed.

    This is deliberately selection-only: it neither reranks eligibility nor
    substitutes another source row.  The frozen canonical JSONL is authority,
    while current production focus and D projection must reproduce each
    payload byte-for-byte.
    """
    chunks = tuple(chunks)
    if not _SAFE_HASH.fullmatch(str(expected_sha256)):
        raise ValueError("frozen manifest expected SHA-256 must be 64 lowercase hex")
    if not _SAFE_HASH.fullmatch(str(expected_ordered_identity_sha256)):
        raise ValueError(
            "frozen ordered identity expected SHA-256 must be 64 lowercase hex"
        )
    frozen_bytes = frozen_manifest_path.read_bytes()
    frozen_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
    if frozen_sha256 != expected_sha256:
        raise ValueError("frozen manifest SHA-256 mismatch")
    try:
        text = frozen_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("frozen manifest must be UTF-8") from exc
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 8:
        raise ValueError("frozen D-C1 manifest must contain exactly eight rows")
    try:
        authoritative = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError("frozen D-C1 manifest is not valid JSONL") from exc
    if not all(isinstance(row, Mapping) for row in authoritative):
        raise ValueError("frozen D-C1 manifest rows must be objects")
    canonical_bytes = "".join(f"{_stable(row)}\n" for row in authoritative).encode()
    if canonical_bytes != frozen_bytes:
        raise ValueError("frozen D-C1 manifest must use canonical JSONL bytes")
    allowed_keys = (
        _FROZEN_D_C1_REQUIRED_ROW_KEYS
        | frozenset(_FROZEN_D_C1_OPTIONAL_ROW_KEYS)
    )
    for position, frozen in enumerate(authoritative):
        keys = frozenset(frozen)
        missing = _FROZEN_D_C1_REQUIRED_ROW_KEYS - keys
        extra = keys - allowed_keys
        if missing or extra:
            raise ValueError(
                f"frozen D-C1 manifest row schema mismatch at row {position}"
            )
        for key, accepted_types in _FROZEN_D_C1_OPTIONAL_ROW_KEYS.items():
            if key in frozen and (
                not isinstance(frozen[key], accepted_types)
                or isinstance(frozen[key], bool)
            ):
                raise ValueError(
                    f"frozen D-C1 optional field is invalid at row {position}: {key}"
                )
    ordered_projection = frozen_d_c1_ordered_identity_projection(authoritative)
    ordered_identity_sha256 = frozen_d_c1_ordered_identity_sha256(authoritative)
    if ordered_identity_sha256 != expected_ordered_identity_sha256:
        raise ValueError("frozen D-C1 ordered identity SHA-256 mismatch")

    eligibility_bytes = frozen_eligibility_path.read_bytes()
    eligibility_sha256 = hashlib.sha256(eligibility_bytes).hexdigest()
    if (
        not _SAFE_HASH.fullmatch(str(expected_eligibility_sha256))
        or eligibility_sha256 != expected_eligibility_sha256
    ):
        raise ValueError("frozen eligibility SHA-256 mismatch")
    try:
        eligibility = json.loads(eligibility_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen eligibility must be UTF-8 JSON") from exc
    canonical_eligibility_bytes = (
        f"{json.dumps(eligibility, indent=2, sort_keys=True)}\n".encode("utf-8")
    )
    if canonical_eligibility_bytes != eligibility_bytes:
        raise ValueError("frozen eligibility must use canonical JSON bytes")
    eligibility_keys = {
        "benchmark_by_kind", "benchmark_rows", "benchmark_variant",
        "eligible_candidates_by_kind", "exclusions", "input_chunks",
        "selected_by_kind", "selected_unique_chunks",
    }
    selection_identity_keys = {
        "selection_version", "preference_predicate_sha256",
        "preference_window_universe_sha256",
        "candidate_universe_sha256_by_kind",
        "selected_universe_sha256_by_kind",
    }
    observed_eligibility_keys = (
        frozenset(eligibility) if isinstance(eligibility, Mapping) else frozenset()
    )
    if (
        not isinstance(eligibility, Mapping)
        or observed_eligibility_keys not in {
            frozenset(eligibility_keys),
            frozenset(eligibility_keys | selection_identity_keys),
        }
    ):
        raise ValueError("frozen eligibility schema mismatch")
    count_maps = (
        "benchmark_by_kind", "eligible_candidates_by_kind",
        "selected_by_kind",
    )
    for key in count_maps:
        value = eligibility[key]
        if (
            not isinstance(value, Mapping)
            or set(value) != {"instruction", "preference"}
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in value.values()
            )
        ):
            raise ValueError(f"frozen eligibility count map is invalid: {key}")
    exclusions = eligibility["exclusions"]
    if (
        not isinstance(exclusions, Mapping)
        or not exclusions
        or any(
            not isinstance(reason, str) or not reason
            or isinstance(count, bool) or not isinstance(count, int) or count < 0
            for reason, count in exclusions.items()
        )
    ):
        raise ValueError("frozen eligibility exclusions are invalid")
    for key in ("benchmark_rows", "input_chunks", "selected_unique_chunks"):
        value = eligibility[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"frozen eligibility count is invalid: {key}")
    if (
        eligibility["benchmark_variant"] != "D_production_contract"
        or eligibility["benchmark_rows"] != len(authoritative)
        or eligibility["benchmark_by_kind"]
        != {"instruction": 4, "preference": 4}
        or eligibility["selected_unique_chunks"]
        != sum(eligibility["selected_by_kind"].values())
        or eligibility["input_chunks"] != len(source_by_id := {
            _chunk_id(chunk): chunk for chunk in chunks
        })
        or "" in source_by_id
        or len(source_by_id) != len(list(chunks))
    ):
        raise ValueError("frozen eligibility is not bound to the source cohort")

    source_by_id = {}
    for chunk in chunks:
        chunk_id = _chunk_id(chunk)
        if not chunk_id:
            raise ValueError("source chunk is missing its identity")
        if chunk_id in source_by_id:
            raise ValueError(f"source chunk identity is duplicated: {chunk_id}")
        source_by_id[chunk_id] = chunk

    identities: set[tuple[str, str, str, int]] = set()
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    rehydrated_hashes: list[str] = []
    for position, frozen in enumerate(authoritative):
        kind = str(frozen.get("kind") or "")
        variant = str(frozen.get("variant") or "")
        chunk_id = str(frozen.get("chunk_id") or "")
        repetition = frozen.get("repetition")
        if (
            kind not in {"instruction", "preference"}
            or variant != "D_production_contract"
            or repetition != 0
            or not chunk_id
        ):
            raise ValueError(
                "frozen D-C1 rows require instruction/preference, "
                "D_production_contract, and repetition 0"
            )
        identity = (chunk_id, kind, variant, repetition)
        if identity in identities:
            raise ValueError("frozen D-C1 row identity is duplicated")
        identities.add(identity)
        counts[kind] += 1
        source = source_by_id.get(chunk_id)
        if source is None:
            raise ValueError(f"frozen source chunk is missing: {chunk_id}")
        candidate = variant_chunk(
            apply_runtime_focus(source, objectives), variant,
        )
        candidate_sha256 = _digest(candidate)
        if (
            not _SAFE_HASH.fullmatch(str(frozen.get("chunk_sha256") or ""))
            or frozen["chunk_sha256"] != candidate_sha256
        ):
            raise ValueError(f"frozen canonical payload hash mismatch at row {position}")
        expected_card = objective_card(
            candidate["synthesis_focus_objective"]
        )
        if frozen.get("focus_objective") != expected_card:
            raise ValueError(f"frozen objective card mismatch at row {position}")
        if kind == "preference":
            from Trainforge.generators.staged_synthesis_micro import (
                micro_preference_eligibility,
            )
            exact = micro_preference_eligibility(
                candidate, focus=candidate["synthesis_focus_objective"],
            )
            if not exact["eligible"]:
                raise ValueError(
                    f"frozen preference row is no longer exact-eligible at "
                    f"row {position}: {exact['reason']}"
                )
        # These are execution identity, not decorative metadata.
        if frozen.get("pilot_version") != PILOT_VERSION:
            raise ValueError(f"frozen pilot version mismatch at row {position}")
        if frozen.get("stratum") != list(_stratum(candidate)):
            raise ValueError(f"frozen stratum mismatch at row {position}")
        row = dict(frozen)
        row["_chunk"] = candidate
        rows.append(row)
        rehydrated_hashes.append(candidate_sha256)
    if counts != Counter({"instruction": 4, "preference": 4}):
        raise ValueError("frozen D-C1 manifest requires four rows of each kind")
    if selection_identity_keys <= observed_eligibility_keys:
        _recomputed_rows, recomputed = build_pilot_manifest(
            chunks, objectives=objectives, repetitions=1,
        )
        del _recomputed_rows
        for key in selection_identity_keys:
            if eligibility.get(key) != recomputed.get(key):
                raise ValueError(
                    f"frozen selection identity mismatch: {key}"
                )
    return rows, {
        "schema_version": FROZEN_D_C1_LOADER_AUDIT_SCHEMA,
        "selection_mode": FROZEN_D_C1_LOADER_VERSION,
        "loader_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_manifest_sha256": frozen_sha256,
        "frozen_eligibility_schema": FROZEN_D_C1_ELIGIBILITY_SCHEMA,
        "frozen_eligibility_sha256": eligibility_sha256,
        "ordered_identity_schema": FROZEN_D_C1_ORDERED_IDENTITY_SCHEMA,
        "ordered_identity_sha256": ordered_identity_sha256,
        "ordered_identity_projection": ordered_projection,
        "chunks_source_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "objectives_source_sha256": hashlib.sha256(
            objectives_path.read_bytes()
        ).hexdigest(),
        "rehydrated_row_sha256": rehydrated_hashes,
        "benchmark_rows": 8,
        "benchmark_by_kind": dict(sorted(counts.items())),
        "benchmark_variant": "D_production_contract",
    }


def counterbalanced_cell_rows(
    rows: Iterable[Mapping[str, Any]], *, run_id: str, cell_id: str,
) -> list[dict[str, Any]]:
    """Return a deterministic Latin rotation of the canonical row order.

    The five planned benchmark cells use five distinct rotations.  Consequently
    each of the eight units occupies five distinct positions across the sweep,
    which is the most even possible positional coverage for five observations
    over eight positions.  Canonicalization happens before rotation so caller
    input order cannot influence the schedule.
    """
    canonical = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _digest(
                f"{run_id}|{row.get('chunk_id')}|{row.get('kind')}|"
                f"{row.get('repetition', 0)}"
            ),
            str(row.get("chunk_id")),
            str(row.get("kind")),
        ),
    )
    if not canonical:
        return []
    planned_cells = tuple(f"c{value}" for value in BENCHMARK_CONCURRENCY_CELLS)
    if cell_id in planned_cells:
        rotation = planned_cells.index(cell_id)
    elif cell_id.endswith("-repeat") and cell_id[:-7] in planned_cells:
        # A confirmatory repeat follows the planned Latin rotations instead of
        # duplicating the original cell's position assignment.
        rotation = len(planned_cells)
    else:
        rotation = int(_digest(f"{run_id}|{cell_id}")[:16], 16)
    offset = rotation % len(canonical)
    return canonical[offset:] + canonical[:offset]


class _TransportProbe:
    """Measure every production structured call without changing its payload."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.calls: list[dict[str, Any]] = []
        self.active_requests = 0
        self._local = threading.local()
        self._lock = threading.Lock()
        client = base._oa_client
        original = client.post_with_usage

        def measured(payload: Mapping[str, Any], *, task: str) -> Any:
            started = time.perf_counter()
            with self._lock:
                self.active_requests += 1
            try:
                result = original(payload, task=task)
            finally:
                elapsed = time.perf_counter() - started
                with self._lock:
                    self.active_requests -= 1
            body, retries = result
            usage = client._extract_usage(body)
            record = {
                "latency_seconds": elapsed,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "requests": int(retries) + 1,
                "stage": str(task),
            }
            with self._lock:
                self.calls.append(record)
            local_calls = getattr(self._local, "calls", None)
            if local_calls is not None:
                local_calls.append(record)
            return result

        client.post_with_usage = measured

    def begin_unit(self) -> None:
        self._local.calls = []

    def end_unit(self) -> list[dict[str, Any]]:
        calls = list(getattr(self._local, "calls", []))
        self._local.calls = None
        return calls


def _score_pair(
    pair: dict[str, Any], chunk: dict[str, Any], kind: str,
    objectives: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the same post-generation semantic gates used by production."""
    pair.setdefault("chunk_id", _chunk_id(chunk))
    pair.setdefault("source_chunk_id", _chunk_id(chunk))
    pair.setdefault("lo_refs", list(chunk.get("learning_outcome_refs") or []))
    pair.setdefault("bloom_level", chunk.get("bloom_level"))
    pair.setdefault("rationale", "Pilot pair generated from the validated staged evidence plan.")
    from lib.validators.pair.claim_support import PairClaimSupportValidator
    from lib.validators.pair.objective_delivery import PairObjectiveDeliveryValidator
    from lib.validators.pair.promotion import TrainingPairPromotionValidator

    metrics: dict[str, Any] = {}
    validators = (
        ("claim_support", PairClaimSupportValidator(), {
            "chunk": chunk,
            "chunk_id_to_text_map": {_chunk_id(chunk): str(chunk.get("text") or "")},
            "authorized_private_sidecar": pair.get(
                "_objective_execution_private_sidecar"
            ),
        }),
        ("objective_delivery", PairObjectiveDeliveryValidator(require_verifiable=True), {
            "chunk": chunk, "objectives": objectives,
        }),
        ("promotion", TrainingPairPromotionValidator(), {
            "chunk": chunk,
            "authorized_private_sidecar": pair.get(
                "_objective_execution_private_sidecar"
            ),
        }),
    )
    for name, validator, kwargs in validators:
        status, reason, fields = validator.validate_pair(
            pair, kind=kind, **kwargs,
        )
        pair.update(fields)
        metrics[name] = {"status": status, "reason": reason}
        if name == "claim_support":
            metrics["semantic_coverage"] = fields.get("claim_support_rate")
            metrics["nli"] = fields.get("per_claim_support")
        elif name == "objective_delivery":
            metrics["bloom_objective_delivery"] = fields.get(
                "pair_objective_alignment"
            )
    metrics["accepted"] = all(
        metrics[name]["status"] != "rejected"
        for name in ("claim_support", "objective_delivery", "promotion")
    )
    return metrics


def execute_pilot(
    rows: Iterable[Mapping[str, Any]],
    provider: Any,
    *,
    objectives: Mapping[str, Mapping[str, Any]],
    scorer: Callable[..., dict[str, Any]] = _score_pair,
    max_concurrent: int = 1,
    checkpoint_path: Optional[Path] = None,
    cell_deadline_seconds: Optional[float] = None,
    cell_deadline_monotonic: Optional[float] = None,
    request_timeout_seconds: float = 240.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute manifest rows through the production staged provider surface."""
    if not 1 <= int(max_concurrent) <= 48:
        raise ValueError("max_concurrent must be within 1..48")
    result_lock = threading.Lock()
    checkpoint_lock = threading.Lock()

    def execute_one(source_row: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(source_row)
        chunk = row.pop("_chunk")
        transport_probe = getattr(provider, "_pilot_transport_probe", None)
        before = len(getattr(provider, "_pilot_calls", []))
        if transport_probe is not None:
            transport_probe.begin_unit()
        started = time.perf_counter()
        error_code = None
        scoring_error = None
        pair = None
        ledger = getattr(provider, "_pilot_attempt_ledger", None)
        unit_context = (
            ledger.unit(
                f"{row['chunk_id']}:{row['kind']}:{row['variant']}:"
                f"r{row.get('repetition', 0)}"
            )
            if ledger is not None else __import__("contextlib").nullcontext()
        )
        try:
            with unit_context:
                canonical_draft = {
                    "provider": "local",
                    "source_chunk_id": row["chunk_id"],
                }
                draft = {
                    **canonical_draft,
                    "_micro_manifest_identity": {
                        "chunk_id": row["chunk_id"],
                        "chunk_sha256": row["chunk_sha256"],
                        "kind": row["kind"],
                        "variant": row["variant"],
                        "repetition": int(row.get("repetition", 0)),
                        "draft": canonical_draft,
                    },
                }
                if row["kind"] == "instruction":
                    pair = provider.paraphrase_instruction(draft, chunk)
                else:
                    pair = provider.paraphrase_preference(draft, chunk)
        except Exception as exc:  # pilot records failures; it does not retry units
            scores = {"accepted": False}
            error_code = str(getattr(exc, "code", "") or type(exc).__name__)
            raw_details = getattr(exc, "details", None)
            error_details = _sanitize_error_details(raw_details)
        else:
            error_details = None
            try:
                scores = scorer(pair, chunk, row["kind"], objectives)
            except Exception as exc:
                scores = {"accepted": False}
                error_code = "objective_validation_unavailable"
                scoring_error = (
                    f"{type(exc).__name__}: {str(exc)[:500]}"
                )
                error_details = {
                    "terminal_content_rejection": False,
                    "integrity_failure": True,
                    "validation_error": "objective_validation_unavailable",
                }
            else:
                if scores.get("accepted") is False:
                    rejected_metrics = [
                        (name, metric)
                        for name, metric in scores.items()
                        if isinstance(metric, Mapping)
                        and metric.get("status") == "rejected"
                    ]
                    reasons = [
                        str(metric.get("reason") or "").strip()
                        for _name, metric in rejected_metrics
                    ]
                    if "objective_validation_unavailable" in reasons:
                        error_code = "objective_validation_unavailable"
                        error_details = {
                            "terminal_content_rejection": False,
                            "integrity_failure": True,
                            "validation_error": error_code,
                        }
                    elif "objective_statement_undersupported" in reasons:
                        error_code = "objective_statement_undersupported"
                        error_details = {
                            "terminal_content_rejection": True,
                            "stage": "post_validation",
                            "validation_error": error_code,
                            "quality_rejection": True,
                        }
                    else:
                        stable_reason = next((reason for reason in reasons if reason), "")
                        error_code = (
                            f"post_validation:{stable_reason}"
                            if stable_reason else "post_validation_rejected"
                        )
                        error_details = {
                            "terminal_content_rejection": True,
                            "stage": "post_validation",
                            "validation_error": error_code,
                            "quality_rejection": True,
                        }
            if isinstance(pair, dict):
                pair.pop("_objective_execution_private_sidecar", None)
        latency = time.perf_counter() - started
        calls = (
            transport_probe.end_unit()
            if transport_probe is not None
            else list(getattr(provider, "_pilot_calls", []))[before:]
        )
        prompt_tokens = sum(item["prompt_tokens"] for item in calls)
        completion_tokens = sum(item["completion_tokens"] for item in calls)
        requests = sum(item["requests"] for item in calls)
        total_tokens = prompt_tokens + completion_tokens
        microstage = Counter()
        claim_slots: set[str] = set()
        microstage_metrics: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "latency_seconds": 0.0, "repairs": 0,
                "deterministic_events": 0,
            }
        )
        for item in calls:
            stage = str(item.get("stage") or "")
            match = __import__("re").search(r"micro_([A-F])(?:_|$)", stage)
            if match:
                stage_letter = match.group(1)
                microstage[stage_letter] += 1
                metrics = microstage_metrics[stage_letter]
                metrics["calls"] += 1
                metrics["prompt_tokens"] += int(item.get("prompt_tokens", 0))
                metrics["completion_tokens"] += int(
                    item.get("completion_tokens", 0)
                )
                metrics["total_tokens"] += (
                    int(item.get("prompt_tokens", 0))
                    + int(item.get("completion_tokens", 0))
                )
                metrics["latency_seconds"] += float(
                    item.get("latency_seconds", 0.0)
                )
                attempt = __import__("re").search(r"_attempt_(\d+)", stage)
                if attempt and int(attempt.group(1)) > 1:
                    metrics["repairs"] += 1
            slot = __import__("re").search(r"micro_B_claim_(\d+)", stage)
            if slot:
                claim_slots.add(slot.group(1))
        row.update({
            "accepted": bool(scores.get("accepted")),
            "stage_validity": error_code is None and bool(scores.get("accepted")),
            "leakage_passed": error_code != "provider_output_verbatim_leakage",
            "scores": scores,
            "error_code": error_code,
            "error_details": error_details,
            "scoring_error": scoring_error,
            "truncated": error_code in {
                "output_truncated", "staged_output_truncated",
                "staged_context_window_exceeded", "field_clamp_truncation",
            },
            "calls": len(calls),
            "requests": requests,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": latency,
            "tokens_per_second": total_tokens / latency if latency else None,
            "pair_sha256": _digest(pair) if pair is not None else None,
        })
        if (
            getattr(provider, "_pilot_synthesis_contract", {}).get("version")
            == MICRO_SYNTHESIS_CONTRACT
        ):
            # C is deterministic assembly. Record it as an observed assembly
            # event only after a downstream realization call proves C returned;
            # it must never fabricate transport or DecisionCapture activity.
            if any(stage in microstage for stage in ("D", "E", "F")):
                microstage["C"] += 1
                microstage_metrics["C"]["deterministic_events"] += 1
            row["microstage_calls"] = dict(sorted(microstage.items()))
            row["claim_slot_calls"] = {
                slot: 1 for slot in sorted(claim_slots)
            }
            for stage, values in microstage_metrics.items():
                values["completed_units"] = (
                    len(claim_slots) if stage == "B" else 1
                )
            row["microstage_metrics"] = {
                key: {
                    **value,
                    "tokens_per_second": (
                        value["total_tokens"] / value["latency_seconds"]
                        if value["latency_seconds"] else None
                    ),
                }
                for key, value in sorted(microstage_metrics.items())
            }
        gate_d_binding = getattr(provider, "_pilot_gate_d_binding", None)
        if gate_d_binding is not None:
            # Persist the complete projection and trusted transaction binding in
            # the result/checkpoint row.  The independent Gate-D verifier must
            # be able to validate both without trusting live provider memory.
            row["gate_d_binding"] = dict(gate_d_binding)
            row["result"] = pair
        return row

    source_rows = list(rows)
    results: list[dict[str, Any]] = []
    identity_fields = ("chunk_id", "kind", "variant", "repetition")

    def checkpoint_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(field) for field in identity_fields)

    completed_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        open_started: set[tuple[Any, ...]] = set()
        for line_number, line in enumerate(
            checkpoint_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                saved = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"checkpoint has torn row {line_number}"
                ) from exc
            identity = checkpoint_identity(saved)
            if any(value is None for value in identity):
                raise ValueError(
                    f"checkpoint row {line_number} lacks unit identity"
                )
            state = saved.get("_checkpoint_state", "terminal")
            if state == "started":
                if identity in open_started or identity in completed_by_identity:
                    raise ValueError("checkpoint has duplicate/reentered start")
                open_started.add(identity)
                continue
            if state != "terminal":
                raise ValueError("checkpoint state is invalid")
            saved.pop("_checkpoint_state", None)
            digest = _digest(saved)
            existing = completed_by_identity.get(identity)
            if existing is not None and _digest(existing) != digest:
                raise ValueError("checkpoint has conflicting duplicate terminal")
            completed_by_identity[identity] = saved
            open_started.discard(identity)
        if open_started:
            raise ValueError("checkpoint has started rows without terminal rows")

    manifest_identities = [checkpoint_identity(row) for row in source_rows]
    if len(set(manifest_identities)) != len(manifest_identities):
        raise ValueError("pilot manifest contains duplicate unit identities")
    unknown = set(completed_by_identity) - set(manifest_identities)
    if unknown:
        raise ValueError("checkpoint contains units outside the pilot manifest")
    pending_source_rows = [
        row for row in source_rows
        if checkpoint_identity(row) not in completed_by_identity
    ]
    resumed_results = [
        completed_by_identity[identity]
        for identity in manifest_identities
        if identity in completed_by_identity
    ]
    source_rows = pending_source_rows

    def persist(row: Mapping[str, Any]) -> None:
        if checkpoint_path is None:
            return
        parent_created = not checkpoint_path.parent.exists()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"{_stable({**row, '_checkpoint_state': 'terminal'})}\n".encode(
            "utf-8"
        )
        with checkpoint_lock:
            created = not checkpoint_path.exists()
            with checkpoint_path.open("ab") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            if created or parent_created:
                directory_fd = os.open(checkpoint_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

    critical_codes = {
        "max_retries_exceeded", "request_timeout", "transport_error",
        "cell_request_admission_closed",
        "staged_output_truncated", "staged_context_window_exceeded",
        "staged_context_window_unverified", "staged_thinking_not_disabled",
        "staged_structured_transport_unavailable",
    }
    executor = ThreadPoolExecutor(max_workers=int(max_concurrent))
    started_cell = time.monotonic()
    hard_deadline = (
        float(cell_deadline_monotonic)
        if cell_deadline_monotonic is not None
        else (
            started_cell + cell_deadline_seconds
            if cell_deadline_seconds is not None else None
        )
    )
    ledger = getattr(provider, "_pilot_attempt_ledger", None)
    if hard_deadline is not None and ledger is not None:
        ledger.configure_deadline(
            hard_deadline=hard_deadline,
            max_timeout_seconds=request_timeout_seconds,
            cleanup_seconds=30.0,
        )
    next_index = 0
    active: dict[Any, int] = {}
    indexed_results = []
    persisted_indices: set[int] = set()
    critical = None
    drained = True
    try:
        def consume_future(future: Any, index: int) -> Optional[dict[str, Any]]:
            if index in persisted_indices or future.cancelled():
                return None
            try:
                row = future.result()
            except Exception as exc:
                source = {
                    key: value for key, value in source_rows[index].items()
                    if key != "_chunk"
                }
                row = {
                    **source,
                    "accepted": False,
                    "stage_validity": False,
                    "error_code": str(
                        getattr(exc, "code", "") or type(exc).__name__
                    ),
                    "scoring_error": None,
                    "truncated": False,
                    "calls": 0,
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": 0.0,
                    "tokens_per_second": None,
                    "pair_sha256": None,
                }
            persist(row)
            persisted_indices.add(index)
            indexed_results.append((index, row))
            return row

        while next_index < len(source_rows) and len(active) < max_concurrent:
            future = executor.submit(execute_one, source_rows[next_index])
            active[future] = next_index
            next_index += 1
        while active:
            if hard_deadline is not None and time.monotonic() >= hard_deadline:
                critical = "cell_deadline_exceeded"
                break
            done, _ = wait(active, timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                index = active.pop(future)
                row = consume_future(future, index)
                if row is not None and row.get("error_code") in critical_codes:
                    critical = row["error_code"]
            if critical is not None:
                break
            while next_index < len(source_rows) and len(active) < max_concurrent:
                future = executor.submit(execute_one, source_rows[next_index])
                active[future] = next_index
                next_index += 1
        if critical is not None:
            for pending in active:
                pending.cancel()
            drain_remaining = (
                max(0.0, hard_deadline - time.monotonic())
                if hard_deadline is not None
                else request_timeout_seconds + 5.0
            )
            done_drain, not_done = wait(active, timeout=drain_remaining)
            for future in done_drain:
                consume_future(future, active[future])
            if not_done:
                drained = False
                raise RuntimeError(
                    f"pilot stopped fail-closed: {critical}; "
                    f"{len(not_done)} active requests did not drain within "
                    f"{drain_remaining}s"
                )
            raise RuntimeError(f"pilot stopped fail-closed: {critical}")
        fresh_results = [row for _, row in sorted(indexed_results)]
        fresh_by_identity = {
            checkpoint_identity(row): row for row in fresh_results
        }
        resumed_by_identity = {
            checkpoint_identity(row): row for row in resumed_results
        }
        results = [
            (fresh_by_identity.get(identity) or resumed_by_identity[identity])
            for identity in manifest_identities
        ]
    finally:
        executor.shutdown(wait=drained, cancel_futures=True)
    return results, summarize(results)


def summarize(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    results = list(results)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["variant"])].append(row)
    variants = {}
    for variant in VARIANTS:
        rows = grouped.get(variant, [])
        accepted = sum(bool(row.get("accepted")) for row in rows)
        total_tokens = sum(int(row.get("total_tokens", 0)) for row in rows)
        total_latency = sum(float(row.get("latency_seconds", 0)) for row in rows)
        errors = Counter(
            str(row["error_code"]) for row in rows if row.get("error_code")
        )
        variants[variant] = {
            "units": len(rows),
            "accepted": accepted,
            "acceptance_rate": accepted / len(rows) if rows else None,
            "stage_valid_rate": (
                sum(bool(row.get("stage_validity")) for row in rows) / len(rows)
                if rows else None
            ),
            "calls_per_accepted_pair": (
                sum(int(row.get("calls", 0)) for row in rows) / accepted
                if accepted else None
            ),
            "tokens_per_accepted_pair": total_tokens / accepted if accepted else None,
            "latency_per_accepted_pair": total_latency / accepted if accepted else None,
            "aggregate_tokens_per_second": (
                total_tokens / total_latency if total_latency else None
            ),
            "truncations": sum(bool(row.get("truncated")) for row in rows),
            "errors": dict(sorted(errors.items())),
        }
    by_microstage: Counter[str] = Counter()
    by_claim_slot: Counter[str] = Counter()
    microstage_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "latency_seconds": 0.0, "repairs": 0,
            "deterministic_events": 0, "completed_units": 0,
            "accepted_units": 0, "failed_units": 0,
        }
    )
    for row in results:
        by_microstage.update(row.get("microstage_calls") or {})
        by_claim_slot.update(row.get("claim_slot_calls") or {})
        for stage, values in (row.get("microstage_metrics") or {}).items():
            target = microstage_metrics[str(stage)]
            for key in (
                "calls", "prompt_tokens", "completion_tokens",
                "total_tokens", "repairs", "deterministic_events",
                "completed_units",
            ):
                target[key] += int(values.get(key, 0))
            target["latency_seconds"] += float(
                values.get("latency_seconds", 0.0)
            )
            if row.get("accepted"):
                target["accepted_units"] += 1
            elif values.get("calls"):
                target["failed_units"] += 1
    summary = {
        "pilot_version": PILOT_VERSION,
        "units": sum(len(rows) for rows in grouped.values()),
        "variants": variants,
    }
    if by_microstage or by_claim_slot:
        summary["by_microstage"] = dict(sorted(by_microstage.items()))
        summary["by_claim_slot"] = dict(sorted(by_claim_slot.items()))
        accepted_pairs = sum(bool(row.get("accepted")) for row in results)
        summary["microstage_metrics"] = {
            stage: {
                **values,
                "tokens_per_second": (
                    values["total_tokens"] / values["latency_seconds"]
                    if values["latency_seconds"] else None
                ),
                "repair_rate": (
                    values["repairs"] / values["calls"]
                    if values["calls"] else None
                ),
                "acceptance_rate": (
                    values["accepted_units"]
                    / (values["accepted_units"] + values["failed_units"])
                    if values["accepted_units"] + values["failed_units"] else None
                ),
                "tokens_per_completed_pair": (
                    values["total_tokens"] / accepted_pairs
                    if accepted_pairs else None
                ),
                "latency_per_completed_pair": (
                    values["latency_seconds"] / accepted_pairs
                    if accepted_pairs else None
                ),
            }
            for stage, values in sorted(microstage_metrics.items())
        }
    return summary


_CELL_IDENTITY_FIELDS = ("chunk_id", "kind", "variant", "repetition")
_CRITICAL_CELL_ERROR_CODES = {
    "max_retries_exceeded", "request_timeout", "transport_error",
    "cell_request_admission_closed", "cell_deadline_exceeded",
    "output_truncated", "staged_output_truncated",
    "staged_context_window_exceeded", "field_clamp_truncation",
    "staged_context_window_unverified", "staged_thinking_not_disabled",
    "staged_structured_transport_unavailable",
    "provider_output_verbatim_leakage",
}


def _cell_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in _CELL_IDENTITY_FIELDS)


def _row_invariant_errors(row: Mapping[str, Any]) -> list[str]:
    """Return fail-closed errors for the independently classified row state."""
    errors: list[str] = []
    for field in ("accepted", "stage_validity", "truncated", "leakage_passed"):
        if type(row.get(field)) is not bool:
            errors.append(f"{field}_not_boolean")
    if errors:
        return errors

    accepted = row["accepted"]
    stage_valid = row["stage_validity"]
    truncated = row["truncated"]
    leakage_passed = row["leakage_passed"]
    error_code = row.get("error_code")
    scoring_error = row.get("scoring_error")
    details = row.get("error_details")
    scores = row.get("scores")
    quality_rejected = (
        isinstance(scores, Mapping) and scores.get("accepted") is False
    )
    terminal_rejection = (
        isinstance(details, Mapping)
        and details.get("terminal_content_rejection") is True
    )
    pair_hash = row.get("pair_sha256") or row.get("artifact_sha256")
    valid_pair_hash = (
        isinstance(pair_hash, str) and _SAFE_HASH.fullmatch(pair_hash) is not None
    )

    if truncated:
        errors.append("truncated_row")
    if leakage_passed is not True:
        errors.append("leakage_not_passed")
    if accepted and (
        error_code is not None
        or scoring_error is not None
        or stage_valid is not True
        or truncated
        or leakage_passed is not True
        or not valid_pair_hash
        or terminal_rejection
    ):
        errors.append("accepted_state_inconsistent")
    if (
        error_code is not None
        or scoring_error is not None
        or terminal_rejection
        or quality_rejected
    ) and (
        accepted is not False or stage_valid is not False
    ):
        errors.append("failure_state_inconsistent")
    if (
        not accepted and error_code is None and scoring_error is None
        and not terminal_rejection and not quality_rejected
    ):
        errors.append("rejection_reason_missing")
    return errors


def classify_cell_outcome(
    results: Iterable[Mapping[str, Any]], *,
    expected_rows: Iterable[Mapping[str, Any]],
    publication_state: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    telemetry_summary: Mapping[str, Any],
    decision_audit_report: Mapping[str, Any],
    active_requests: int,
    synthesis_summary: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Classify a cell without treating evidenced quality rejection as outage.

    Advancement is an operational-integrity decision.  Raw quality outcomes
    remain visible and are never converted into acceptances.
    """
    rows = list(results)
    expected = [_cell_identity(row) for row in expected_rows]
    observed = [_cell_identity(row) for row in rows]
    identity_complete = (
        bool(expected)
        and len(observed) == len(expected)
        and Counter(expected) == Counter(observed)
        and all(count == 1 for count in Counter(observed).values())
    )
    reconciliation_valid = verified_report(
        reconciliation, RECONCILIATION_REPORT_SCHEMA,
    )
    telemetry_valid = verified_report(
        telemetry_summary, TELEMETRY_REPORT_SCHEMA,
    )
    terminal_authority = str(
        publication_state.get("state") or "invalid_missing_authority"
    )
    publication_integrity_valid = terminal_authority in {
        "committed", "committed_complete", "terminal_hold",
        "precommit_verified",
    }
    advancing_authority = terminal_authority in {
        "committed_complete", "precommit_verified",
    }

    critical_errors: list[str] = []
    for stage, outcomes in (
        ((synthesis_summary or {}).get("transport_outcomes") or {}).get(
            "stages", {}
        ).items()
    ):
        if int(outcomes.get("unrecovered", 0)) > 0:
            critical_errors.append(f"transport_failure:{stage}")
    quality_rejections: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    accepted_by_kind: Counter[str] = Counter()
    low_confidence = 0
    for row in rows:
        row_errors = _row_invariant_errors(row)
        if row_errors:
            critical_errors.extend(row_errors)
            continue
        code = str(row.get("error_code") or "")
        details = row.get("error_details")
        details = details if isinstance(details, Mapping) else {}
        if row.get("scoring_error"):
            critical_errors.append("scoring_error")
        if row.get("accepted"):
            accepted_by_kind[str(row.get("kind"))] += 1
        if "low_confidence" in code or any(
            "low_confidence" in str(metric.get("reason") or "")
            for metric in (row.get("scores") or {}).values()
            if isinstance(metric, Mapping)
        ):
            low_confidence += 1
        if not code:
            if not row.get("stage_validity"):
                critical_errors.append("unclassified_stage_invalid")
            continue
        is_staged_invalid = code.startswith("staged_") and code.endswith("_invalid")
        is_post_validation_quality = (
            code == "objective_statement_undersupported"
            or code.startswith("post_validation:")
            or code == "post_validation_rejected"
        )
        has_stage_evidence = (
            details.get("terminal_content_rejection") is True
            and details.get("stage") in _STAGE_VALUES
            and bool(
                details.get("validation_error")
                or details.get("validator_reason")
                or details.get("response_ref")
            )
        )
        if (
            (is_staged_invalid or is_post_validation_quality)
            and has_stage_evidence
            and not row.get("truncated")
            and row.get("leakage_passed") is not False
        ):
            quality_rejections[code] += 1
            stage_counts[str(details["stage"])] += 1
        else:
            critical_errors.append(
                code
                if (
                    code in _CRITICAL_CELL_ERROR_CODES
                    or code == "objective_validation_unavailable"
                )
                else f"unclassified:{code}"
            )

    kinds_present = all(accepted_by_kind[kind] >= 1 for kind in (
        "instruction", "preference",
    ))
    integrity_errors = []
    for valid, reason in (
        (identity_complete, "terminal_identity_incomplete"),
        (publication_integrity_valid, "publication_integrity_invalid"),
        (_verified_audit_report(decision_audit_report), "audit_invalid"),
        (reconciliation_valid, "reconciliation_invalid"),
        (telemetry_valid, "telemetry_invalid"),
        (active_requests == 0, "active_requests_nonzero"),
        (kinds_present, "accepted_kind_missing"),
    ):
        if not valid:
            integrity_errors.append(reason)
    nonadvancing_reasons = []
    if not advancing_authority:
        nonadvancing_reasons.append(
            "terminal_authority_hold"
            if terminal_authority == "terminal_hold"
            else "not_committed_complete"
        )
    return {
        "operationally_valid": (
            not critical_errors
            and not integrity_errors
            and not nonadvancing_reasons
        ),
        "publication_integrity_valid": publication_integrity_valid,
        "terminal_authority": terminal_authority,
        "nonadvancing_reasons": nonadvancing_reasons,
        "accepted_by_kind": dict(sorted(accepted_by_kind.items())),
        "quality_rejections": dict(sorted(quality_rejections.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "low_confidence": low_confidence,
        "critical_errors": critical_errors,
        "integrity_errors": integrity_errors,
        "units": len(rows),
    }


def migrate_matrix_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy matrix summaries to explicit terminal authority semantics."""
    migrated = deepcopy(dict(summary))
    outcomes = []
    for raw_outcome in migrated.get("outcomes", []):
        outcome = dict(raw_outcome)
        integrity_errors = list(outcome.get("integrity_errors") or [])
        legacy_not_committed = "cell_not_committed" in integrity_errors
        integrity_errors = [
            error for error in integrity_errors if error != "cell_not_committed"
        ]
        authority = outcome.get("terminal_authority")
        if not isinstance(authority, str):
            authority = (
                "committed_complete"
                if outcome.get("operationally_valid") and not legacy_not_committed
                else "unknown_legacy_authority"
            )
        publication_valid = outcome.get("publication_integrity_valid")
        if not isinstance(publication_valid, bool):
            publication_valid = not legacy_not_committed
        reasons = list(outcome.get("nonadvancing_reasons") or [])
        if legacy_not_committed and "not_committed_complete" not in reasons:
            reasons.append("not_committed_complete")
        if authority == "terminal_hold":
            publication_valid = True
            reasons = [
                reason for reason in reasons
                if reason not in {"cell_not_committed", "not_committed_complete"}
            ]
            if "terminal_authority_hold" not in reasons:
                reasons.append("terminal_authority_hold")
        outcome.update({
            "publication_integrity_valid": publication_valid,
            "terminal_authority": authority,
            "nonadvancing_reasons": reasons,
            "integrity_errors": integrity_errors,
        })
        outcome["operationally_valid"] = bool(
            outcome.get("operationally_valid")
            and publication_valid
            and authority == "committed_complete"
            and not reasons
        )
        outcomes.append(outcome)
    migrated["schema_version"] = MATRIX_SUMMARY_SCHEMA
    migrated["outcomes"] = outcomes
    return migrated


def _provider(
    *,
    timeout_seconds: float = 240.0,
    transport_attempts: int = 1,
    initial_backoff_seconds: float = 1.0,
    attempt_ledger_path: Optional[Path] = None,
    raw_audit_root: Optional[Path] = None,
    intent_manifest_path: Optional[Path] = None,
    intent_run_id: Optional[str] = None,
    intent_cell_id: Optional[str] = None,
    capture: Any = None,
    synthesis_contract: str = LEGACY_SYNTHESIS_CONTRACT,
    synthesis_seed: Optional[int] = None,
    completion_cap: int = 600,
) -> Any:
    from Trainforge.generators.providers._local_provider import LocalSynthesisProvider
    from Trainforge.generators.providers._openai_compatible_client import (
        RESPONSE_DIALECT_OPENAI_JSON_SCHEMA_STRICT,
    )
    from Trainforge.generators.staged_synthesis_provider import StagedSynthesisProvider
    if transport_attempts != 1:
        raise ValueError(
            "benchmark transport_attempts must be 1: reissue is prohibited "
            "until the prior server request is independently confirmed drained"
        )
    base = LocalSynthesisProvider(
        timeout=timeout_seconds,
        capture=capture,
        response_dialect=RESPONSE_DIALECT_OPENAI_JSON_SCHEMA_STRICT,
    )
    base._oa_client._max_retries = transport_attempts
    base._oa_client._initial_backoff_seconds = initial_backoff_seconds
    probe = _TransportProbe(base)
    resolved_contract = resolve_synthesis_contract(synthesis_contract)
    if resolved_contract == MICRO_SYNTHESIS_CONTRACT:
        from Trainforge.generators.staged_synthesis_micro import (
            MicroStagedSynthesisProvider,
        )
        if synthesis_seed is None:
            raise ValueError(
                "micro synthesis requires an explicit --synthesis-seed"
            )
        staged = MicroStagedSynthesisProvider(
            base, synthesis_seed=synthesis_seed,
            completion_cap=completion_cap,
        )
    else:
        staged = StagedSynthesisProvider(base)
    staged._pilot_synthesis_contract = synthesis_contract_identity(
        resolved_contract
    )
    staged._pilot_calls = probe.calls
    staged._pilot_transport_probe = probe
    if attempt_ledger_path is not None:
        from Trainforge.generators.providers.http_attempt_ledger import (
            DurableCallIntentManifest,
            DurableHttpAttemptLedger,
            install_on_client,
        )
        ledger = DurableHttpAttemptLedger(
            attempt_ledger_path, raw_audit_root=raw_audit_root,
        )
        intent_manifest = (
            DurableCallIntentManifest(
                intent_manifest_path,
                run_id=str(intent_run_id),
                cell_id=str(intent_cell_id),
                synthesis_contract_sha256=(
                    synthesis_contract_identity(
                        resolved_contract
                    )["fingerprint"]
                    if resolved_contract == MICRO_SYNTHESIS_CONTRACT else None
                ),
            )
            if intent_manifest_path is not None else None
        )
        install_on_client(
            base._oa_client, ledger, intent_manifest=intent_manifest,
        )
        staged._pilot_attempt_ledger = ledger
    return staged


def negotiate_dialect(provider: Any) -> dict[str, Any]:
    """Serially compile one tiny schema before admitting concurrent work."""
    messages = [
        {"role": "system", "content": "Return strict JSON only."},
        {"role": "user", "content": 'Return {"probe":"ready"}.'},
    ]
    rendered_prompt = _stable(messages)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["probe"],
        "properties": {"probe": {"type": "string", "const": "ready"}},
    }
    started = time.perf_counter()
    ledger = getattr(provider, "_pilot_attempt_ledger", None)
    context = (
        ledger.unit("dialect-preflight")
        if ledger is not None else __import__("contextlib").nullcontext()
    )
    with context:
        capture_call = getattr(provider, "_capture_call", None)
        dialect_max_tokens = int(getattr(provider._base, "_max_tokens", 0))
        try:
            raw, usage, _ = provider._structured_completion(
                messages,
                schema=schema,
                meter_task="staged_synthesis:dialect_preflight",
                max_output_tokens=dialect_max_tokens,
            )
        except BaseException as exc:
            if callable(capture_call):
                capture_call(
                    stage="dialect_preflight",
                    chunk_id="dialect-preflight",
                    attempt=1,
                    prompt=rendered_prompt,
                    raw_response="",
                    usage={},
                    validation_error=(
                        f"dialect_transport:{getattr(exc, 'code', type(exc).__name__)}"
                    ),
                    context_headroom_tokens=0,
                    max_output_tokens=dialect_max_tokens,
                    validation_evidence={
                        "response_schema_sha256": _digest(schema),
                        "required_keys": ["probe"],
                        "validator_stage": "dialect_preflight",
                        "validator_attempt": 1,
                        "validator_reason": "transport_rejected",
                        "nli_scores": [],
                    },
                )
            raise
        if callable(capture_call):
            capture_call(
                stage="dialect_preflight",
                chunk_id="dialect-preflight",
                attempt=1,
                prompt=rendered_prompt,
                raw_response=raw,
                usage=usage,
                validation_error=None,
                context_headroom_tokens=0,
                max_output_tokens=dialect_max_tokens,
                validation_evidence={
                    "response_schema_sha256": _digest(schema),
                    "required_keys": ["probe"],
                    "validator_stage": "dialect_preflight",
                    "validator_attempt": 1,
                    "validator_reason": None,
                    "nli_scores": [],
                },
            )
    parsed = provider._base._oa_client._extract_json_lenient(raw)
    if parsed != {"probe": "ready"}:
        raise RuntimeError("serial dialect preflight violated its strict schema")
    return {
        "outcome": "accepted",
        "response_dialect": getattr(
            provider._base._oa_client, "_response_dialect", None,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "schema_sha256": _digest(schema),
        "response_sha256": _digest(raw),
        "usage": usage,
        "ollama_format_supported": getattr(
            provider._base._oa_client, "_ollama_format_supported", None,
        ),
    }


def build_preflight_artifact(
    *,
    manifest_path: Path,
    eligibility_path: Path,
    base_url: str,
    model_snapshot: Mapping[str, Any],
    run_id: str,
    server_batch: int,
    client_concurrency: int,
    timeout_seconds: float,
    transport_attempts: int,
    initial_backoff_seconds: float,
    max_tokens: int,
    temperature: float,
    dialect: Mapping[str, Any],
    cell_id: Optional[str] = None,
    telemetry_identity: Optional[Mapping[str, Any]] = None,
    intent_manifest_path: Optional[Path] = None,
    stop_after_concurrency: Optional[int] = None,
    synthesis_contract: str = LEGACY_SYNTHESIS_CONTRACT,
    frozen_loader_audit_path: Optional[Path] = None,
    micro_journal_root: Optional[str] = None,
    qualification_route: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Bind immutable code, serving identity, manifest, and run contract."""
    root = Path(__file__).resolve().parents[3]
    contract_paths = (
        "Trainforge/generators/staged_synthesis_provider.py",
        "Trainforge/generators/synthesis_window_contract.py",
        "Trainforge/synthesis_eligibility.py",
        "Trainforge/scripts/harness/staged_window_abcd_pilot.py",
        "Trainforge/generators/providers/http_attempt_ledger.py",
        "Trainforge/generators/trtllm_benchmark_telemetry.py",
        "Trainforge/synthesis/verification/benchmark_artifact_verifier.py",
        "Trainforge/synthesis/verification/decision_audit_verifier.py",
    )
    resolved_contract = resolve_synthesis_contract(synthesis_contract)
    if resolved_contract == MICRO_SYNTHESIS_CONTRACT:
        contract_paths += (
            "Trainforge/generators/staged_synthesis_micro.py",
        )
    code_hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in contract_paths
    }
    split = urlsplit(base_url)
    host = split.hostname or ""
    if split.port is not None:
        host = f"{host}:{split.port}"
    canonical_base_url = urlunsplit(
        (split.scheme.lower(), host.lower(), split.path.rstrip("/"), "", "")
    )
    dialect_contract = {
        key: dialect.get(key)
        for key in (
            "outcome", "schema_sha256", "ollama_format_supported",
            "response_dialect",
        )
    }
    contract = {
        "pilot_run_id": run_id,
        "cell_id": cell_id,
        "base_url": canonical_base_url,
        "model_snapshot": dict(model_snapshot),
        "model_snapshot_sha256": _digest(model_snapshot),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "eligibility_sha256": hashlib.sha256(
            eligibility_path.read_bytes()
        ).hexdigest(),
        "code_hashes": code_hashes,
        "synthesis_contract": synthesis_contract_identity(resolved_contract),
        "server_batch": server_batch,
        "client_concurrency": client_concurrency,
        "stop_after_concurrency": stop_after_concurrency,
        "timeout_seconds": timeout_seconds,
        "transport_attempts": transport_attempts,
        "retry_policy": "no_reissue_without_confirmed_abort_and_drain",
        "initial_backoff_seconds": initial_backoff_seconds,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "dialect_preflight": dialect_contract,
        "telemetry_identity": dict(telemetry_identity or {}),
        "intent_manifest_preflight_sha256": (
            hashlib.sha256(intent_manifest_path.read_bytes()).hexdigest()
            if intent_manifest_path is not None and intent_manifest_path.is_file()
            else hashlib.sha256(b"").hexdigest()
        ),
        "intent_manifest_preflight_count": (
            len([
                line for line in intent_manifest_path.read_text(
                    encoding="utf-8"
                ).splitlines() if line.strip()
            ])
            if intent_manifest_path is not None and intent_manifest_path.is_file()
            else 0
        ),
    }
    if qualification_route is not None:
        if (
            qualification_route.get("base_url") != canonical_base_url
            or not qualification_route.get("served_model")
        ):
            raise ValueError("qualification route differs from preflight route")
        contract["qualification_route_binding"] = dict(qualification_route)
    if resolved_contract == MICRO_SYNTHESIS_CONTRACT:
        if (
            not isinstance(micro_journal_root, str)
            or not micro_journal_root
            or Path(micro_journal_root).is_absolute()
            or ".." in Path(micro_journal_root).parts
        ):
            raise ValueError(
                "micro synthesis requires a run-local relative journal root"
            )
        contract["micro_journal_root"] = micro_journal_root
        contract["checkpoint_path"] = "checkpoint.jsonl"
    if frozen_loader_audit_path is not None:
        contract["frozen_loader_audit_sha256"] = hashlib.sha256(
            frozen_loader_audit_path.read_bytes()
        ).hexdigest()
    eligibility_record = (
        json.loads(frozen_loader_audit_path.read_text(encoding="utf-8"))
        if frozen_loader_audit_path is not None
        else {}
    )
    if eligibility_record.get("selection_mode") == FROZEN_D_C1_LOADER_VERSION:
        contract["frozen_selection"] = {
            key: eligibility_record[key]
            for key in (
                "selection_mode",
                "schema_version",
                "loader_code_sha256",
                "frozen_manifest_sha256",
                "frozen_eligibility_schema",
                "frozen_eligibility_sha256",
                "ordered_identity_schema",
                "ordered_identity_sha256",
                "ordered_identity_projection",
                "chunks_source_sha256",
                "objectives_source_sha256",
                "rehydrated_row_sha256",
            )
        }
    return {
        **contract,
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc,
        ).isoformat(),
        "run_contract_sha256": _digest(contract),
    }


def _atomic_write_bytes(
    path: Path, payload: bytes, *, hard_deadline: Optional[float] = None,
) -> None:
    if hard_deadline is not None and time.monotonic() >= hard_deadline:
        raise TimeoutError("artifact finalization exceeded absolute cell deadline")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    if hard_deadline is not None and time.monotonic() >= hard_deadline:
        raise TimeoutError("artifact finalization crossed absolute cell deadline")


def _write_json_fsync(
    path: Path, value: Mapping[str, Any], *,
    hard_deadline: Optional[float] = None,
) -> None:
    payload = f"{json.dumps(value, indent=2, sort_keys=True)}\n".encode()
    _atomic_write_bytes(path, payload, hard_deadline=hard_deadline)


def _write_immutable_json(
    path: Path, value: Mapping[str, Any], *,
    hard_deadline: Optional[float] = None,
) -> None:
    payload = f"{json.dumps(value, indent=2, sort_keys=True)}\n".encode()
    if path.exists():
        raise FileExistsError(path)
    _atomic_write_bytes(path, payload, hard_deadline=hard_deadline)
    path.chmod(0o444)


def reconcile_http_audit(
    ledger_path: Path, *, report_path: Optional[Path] = None,
    intent_manifest_path: Optional[Path] = None,
    hard_deadline: Optional[float] = None,
) -> dict[str, Any]:
    """Prove manifest-authorized attempts have one started+terminal pair."""
    if intent_manifest_path is None:
        raise ValueError("HTTP reconciliation requires the intent manifest")
    verified = verify_http_reconciliation(
        ledger_path, intent_manifest_path=intent_manifest_path,
    )
    rows = [
        json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    key = lambda row: (
        row.get("unit"), row.get("stage"), row.get("attempt"),
        row.get("request_sha256"),
    )
    started = defaultdict(list)
    terminal = defaultdict(list)
    for row in rows:
        if row.get("event") == "http_attempt_started":
            started[key(row)].append(row)
        elif row.get("event") == "http_attempt_terminal":
            terminal[key(row)].append(row)
    errors = list(verified.get("errors") or [])
    for identity in sorted(set(started) | set(terminal), key=str):
        terminal_rows = terminal.get(identity, [])
        started_rows = started.get(identity, [])
        if len(terminal_rows) != 1 or len(started_rows) != 1:
            errors.append({"identity": identity, "reason": "non_bijective_attempt"})
            continue
        for ref_name, row in (
            ("request_raw_ref", started_rows[0]),
            ("response_raw_ref", terminal_rows[0]),
        ):
            ref = row.get(ref_name)
            path = Path(str(ref.get("path"))) if isinstance(ref, Mapping) else None
            if path is None or not path.is_file():
                errors.append({"identity": identity, "reason": f"missing_{ref_name}"})
                continue
            payload = path.read_bytes()
            if (
                hashlib.sha256(payload).hexdigest() != ref.get("sha256")
                or len(payload) != ref.get("bytes")
            ):
                errors.append({"identity": identity, "reason": f"invalid_{ref_name}"})
    report = {
        "status": "accepted" if verified_report(
            verified, RECONCILIATION_REPORT_SCHEMA,
        ) and not errors else "rejected",
        "started": sum(len(values) for values in started.values()),
        "terminal": sum(len(values) for values in terminal.values()),
        "errors": errors,
    }
    if report_path is not None:
        _write_json_fsync(
            report_path, report, hard_deadline=hard_deadline,
        )
    if errors:
        raise RuntimeError("HTTP attempt audit reconciliation failed")
    return report


def canonicalize_micro_publication_rows(
    *,
    checkpoint_rows: Iterable[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    manifest_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reconcile concurrent completion rows using frozen manifest authority."""
    base_fields = ("chunk_id", "kind", "variant", "repetition")
    manifest_list = [dict(row) for row in manifest_rows]
    row_id_presence = ["row_id" in row for row in manifest_list]
    if any(row_id_presence) and not all(row_id_presence):
        raise ValueError("micro publication manifest mixes row-id schemas")
    fields = (
        ("row_id", *base_fields) if row_id_presence and all(row_id_presence)
        else base_fields
    )

    def indexed(
        rows: Iterable[Mapping[str, Any]], *, label: str,
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        output: dict[tuple[Any, ...], dict[str, Any]] = {}
        for source in rows:
            row = dict(source)
            identity = tuple(row.get(field) for field in fields)
            if (
                any(value is None or value == "" for value in identity)
                or identity in output
            ):
                raise ValueError(
                    f"micro publication {label} identity is invalid or duplicated"
                )
            output[identity] = row
        return output

    checkpoint = indexed(checkpoint_rows, label="checkpoint")
    results = indexed(result_rows, label="result")
    manifest_order = []
    seen_manifest: set[tuple[Any, ...]] = set()
    for row in manifest_list:
        identity = tuple(row.get(field) for field in fields)
        if (
            any(value is None or value == "" for value in identity)
            or identity in seen_manifest
        ):
            raise ValueError("micro publication manifest identity is invalid")
        seen_manifest.add(identity)
        manifest_order.append(identity)
    if set(checkpoint) != set(results) or set(results) != seen_manifest:
        raise ValueError(
            "micro checkpoint/results differ from frozen manifest identities"
        )
    for identity in manifest_order:
        if _digest(checkpoint[identity]) != _digest(results[identity]):
            raise ValueError("micro checkpoint/result content hash differs")
    return [results[identity] for identity in manifest_order]


def publish_cell_success(
    *,
    cell_dir: Path,
    ledger_path: Path,
    results: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
    hard_deadline: float,
    decision_audit_report_path: Path,
    decision_audit_report_sha256: str,
    telemetry_report_path: Path,
    telemetry_report_sha256: str,
    reconciliation_report_path: Path,
    reconciliation_report_sha256: str,
    intent_manifest_path: Path,
    publication_identity: Optional[Mapping[str, Any]] = None,
    authority_status: str = "committed",
) -> dict[str, Any]:
    """Transactionally publish success; commit marker is the sole read authority."""
    result_rows = [dict(row) for row in results]
    marker = cell_dir / "finalization_in_progress.json"
    success_path = cell_dir / "success-commit.json"
    terminal_paths = (
        success_path,
        cell_dir / "committed_complete.json",
        cell_dir / "terminal_hold.json",
    )
    if marker.exists() or any(path.exists() for path in terminal_paths):
        raise FileExistsError(
            "cell finalization authority already exists; recovery required"
        )
    _write_json_fsync(marker, {
        "status": "finalization_in_progress",
        "terminal_if_uncommitted": True,
    }, hard_deadline=hard_deadline)
    staging = cell_dir / ".finalization-staging"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        reconcile_http_audit(
            ledger_path,
            intent_manifest_path=intent_manifest_path,
            report_path=staging / "http-reconciliation.json",
            hard_deadline=hard_deadline,
        )
        audit_payload = decision_audit_report_path.read_bytes()
        if (
            not _SAFE_HASH.fullmatch(decision_audit_report_sha256)
            or hashlib.sha256(audit_payload).hexdigest()
            != decision_audit_report_sha256
        ):
            raise ValueError("decision audit report digest does not match bytes")
        _atomic_write_bytes(
            staging / "decision-audit-report.json", audit_payload,
            hard_deadline=hard_deadline,
        )
        intent_payload = intent_manifest_path.read_bytes()
        _atomic_write_bytes(
            staging / "call-intents.jsonl", intent_payload,
            hard_deadline=hard_deadline,
        )
        for name, source, expected_sha, schema in (
            ("telemetry-verification.json", telemetry_report_path,
             telemetry_report_sha256, TELEMETRY_REPORT_SCHEMA),
            ("http-reconciliation-verification.json",
             reconciliation_report_path, reconciliation_report_sha256,
             RECONCILIATION_REPORT_SCHEMA),
        ):
            payload = source.read_bytes()
            if (
                not _SAFE_HASH.fullmatch(expected_sha)
                or hashlib.sha256(payload).hexdigest() != expected_sha
                or not verified_report(json.loads(payload), schema)
            ):
                raise ValueError(f"{name} is not a matching verified report")
            _atomic_write_bytes(staging / name, payload, hard_deadline=hard_deadline)
        identity = dict(publication_identity or {})
        synthesis_identity = identity.get("synthesis_contract")
        micro_checkpoint_payload = None
        if (
            isinstance(synthesis_identity, Mapping)
            and synthesis_identity.get("version") == MICRO_SYNTHESIS_CONTRACT
        ):
            preflight = json.loads(
                (cell_dir / "preflight.json").read_text(encoding="utf-8")
            )
            checkpoint_path = cell_dir / str(
                preflight.get("checkpoint_path") or ""
            )
            if (
                preflight.get("checkpoint_path") != "checkpoint.jsonl"
                or not checkpoint_path.is_file()
            ):
                raise ValueError("micro checkpoint authority is missing")
            completion_rows = []
            for line in checkpoint_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.pop("_checkpoint_state", "terminal") != "terminal":
                    raise ValueError(
                        "micro checkpoint contains nonterminal state"
                    )
                completion_rows.append(row)
            manifest_rows = [
                json.loads(line)
                for line in (cell_dir.parent / "manifest.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            result_rows = canonicalize_micro_publication_rows(
                checkpoint_rows=completion_rows,
                result_rows=result_rows,
                manifest_rows=manifest_rows,
            )
            micro_checkpoint_payload = "".join(
                f"{_stable({**row, '_checkpoint_state': 'terminal'})}\n"
                for row in result_rows
            ).encode()
        result_payload = "".join(
            f"{_stable(row)}\n" for row in result_rows
        ).encode()
        _atomic_write_bytes(
            staging / "results.jsonl", result_payload,
            hard_deadline=hard_deadline,
        )
        _write_json_fsync(
            staging / "summary.json", summary,
            hard_deadline=hard_deadline,
        )
        names = [
            "decision-audit-report.json",
            "http-reconciliation-verification.json",
            "http-reconciliation.json",
            "call-intents.jsonl",
            "results.jsonl",
            "summary.json",
            "telemetry-verification.json",
        ]
        if (
            isinstance(synthesis_identity, Mapping)
            and synthesis_identity.get("version") == MICRO_SYNTHESIS_CONTRACT
        ):
            preflight = json.loads(
                (cell_dir / "preflight.json").read_text(encoding="utf-8")
            )
            micro_report = verify_micro_journals(
                cell_dir, summary=summary, preflight=preflight,
                results=result_rows,
            )
            _write_json_fsync(
                staging / "micro-verification.json", micro_report,
                hard_deadline=hard_deadline,
            )
            _atomic_write_bytes(
                staging / "checkpoint.jsonl", micro_checkpoint_payload or b"",
                hard_deadline=hard_deadline,
            )
            names.extend(("checkpoint.jsonl", "micro-verification.json"))
        names = tuple(names)
        hashes = {
            name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
            for name in names
        }
        if authority_status not in {
            "committed", "committed_complete", "terminal_hold",
        }:
            raise ValueError("invalid terminal publication authority")
        commit_body = {
            "schema_version": CELL_PUBLICATION_SCHEMA,
            "status": authority_status,
            "artifacts": hashes,
            "model_id": identity.get("model_id"),
            "pilot_run_id": identity.get("pilot_run_id"),
            "cell_id": identity.get("cell_id"),
            "preflight_sha256": identity.get("preflight_sha256"),
            "decision_audit_report_sha256": decision_audit_report_sha256,
            "telemetry_report_sha256": telemetry_report_sha256,
            "reconciliation_report_sha256": reconciliation_report_sha256,
            "intent_manifest_sha256": hashlib.sha256(intent_payload).hexdigest(),
            "intent_count": len([
                line for line in intent_payload.decode("utf-8").splitlines()
                if line.strip()
            ]),
        }
        if synthesis_identity is not None:
            commit_body["synthesis_contract"] = synthesis_identity
        commit = {**commit_body, "commit_sha256": _digest(commit_body)}
        if all(identity.get(key) for key in (
            "model_id", "pilot_run_id", "cell_id", "preflight_sha256",
        )):
            try:
                verify_cell_publication(
                    cell_dir,
                    expected_model=str(identity["model_id"]),
                    expected_run_id=str(identity["pilot_run_id"]),
                    expected_preflight_sha256=str(identity["preflight_sha256"]),
                    classified_rows=result_rows,
                    candidate_commit=commit,
                    artifact_dir=staging,
                )
            except BaseException as exc:
                _write_json_fsync(
                    cell_dir / "publication-integrity-failure.json",
                    {
                        "status": "publication_integrity_failed",
                        "candidate_artifacts": hashes,
                        "candidate_commit_sha256": commit["commit_sha256"],
                        "verifier_report": {
                            "status": "rejected",
                            "error_code": type(exc).__name__,
                        },
                    },
                    hard_deadline=hard_deadline,
                )
                raise
        for name in names:
            if time.monotonic() >= hard_deadline:
                raise TimeoutError("cell publication exceeded absolute deadline")
            os.replace(staging / name, cell_dir / name)
            directory_fd = os.open(cell_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        prepared = cell_dir / ".success-commit-prepared.json"
        _write_json_fsync(
            prepared, commit, hard_deadline=hard_deadline,
        )
        if time.monotonic() >= hard_deadline:
            raise TimeoutError("commit preparation exceeded absolute deadline")
        # Marker remains the sole authority while its content transitions from
        # in-progress to a fully durable, recoverable commit payload.
        os.replace(prepared, marker)
        directory_fd = os.open(cell_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if time.monotonic() >= hard_deadline:
            raise TimeoutError("commit marker preparation crossed deadline")
        # Linearization point: exactly one rename removes the marker name and
        # creates the authoritative success name. Never raise for time after it.
        authority_path = (
            success_path if authority_status == "committed"
            else cell_dir / f"{authority_status}.json"
        )
        os.replace(marker, authority_path)
        try:
            directory_fd = os.open(cell_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The success name is already authoritative in this process and
            # contains the fsynced commit payload. Recovery re-fsyncs its dir.
            return {**commit, "publication_recovery": "directory_fsync_required"}
        return commit
    except BaseException:
        # Marker was durable before any public success artifact. Readers must
        # ignore partial files whenever the last commit marker is absent.
        raise


def read_cell_publication_state(cell_dir: Path) -> dict[str, Any]:
    """Mechanically classify crash state; exactly one authority may exist."""
    marker = cell_dir / "finalization_in_progress.json"
    authorities = {
        "committed": cell_dir / "success-commit.json",
        "committed_complete": cell_dir / "committed_complete.json",
        "terminal_hold": cell_dir / "terminal_hold.json",
    }
    present = [(state, path) for state, path in authorities.items() if path.exists()]
    if (marker.exists() and present) or len(present) > 1:
        return {"state": "invalid_dual_authority", "recoverable": False}
    if present:
        state, path = present[0]
        return {
            "state": state, "recoverable": True,
            "commit": json.loads(path.read_text(encoding="utf-8")),
        }
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return {
            "state": "marker_only",
            "recoverable": payload.get("status") in {
                "committed", "committed_complete", "terminal_hold",
            },
            "marker": payload,
        }
    return {"state": "invalid_missing_authority", "recoverable": False}


def verify_cell_publication(
    cell_dir: Path, *,
    expected_model: str,
    expected_run_id: str,
    expected_preflight_sha256: str,
    classified_rows: Iterable[Mapping[str, Any]],
    candidate_commit: Optional[Mapping[str, Any]] = None,
    artifact_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Verify a committed or staged candidate from bytes, never assertions."""
    if candidate_commit is None:
        state = read_cell_publication_state(cell_dir)
        if state.get("state") not in {
            "committed", "committed_complete", "terminal_hold",
        }:
            raise ValueError(f"cell publication authority is {state.get('state')}")
        commit = state.get("commit")
    else:
        commit = dict(candidate_commit)
    artifacts_root = Path(artifact_dir) if artifact_dir is not None else cell_dir
    if not isinstance(commit, Mapping):
        raise ValueError("success commit payload is not an object")
    required_commit = {
        "schema_version", "status", "artifacts", "model_id",
        "pilot_run_id", "cell_id", "preflight_sha256", "decision_audit_report_sha256",
        "telemetry_report_sha256", "reconciliation_report_sha256",
        "intent_manifest_sha256", "intent_count",
        "commit_sha256",
    }
    preflight_path = cell_dir / "preflight.json"
    if hashlib.sha256(preflight_path.read_bytes()).hexdigest() != (
        expected_preflight_sha256
    ):
        raise ValueError("preflight digest does not match committed identity")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    frozen_selection = preflight.get("frozen_selection")
    if frozen_selection is not None:
        if (
            hashlib.sha256(
                (cell_dir.parent / "manifest.jsonl").read_bytes()
            ).hexdigest() != preflight.get("manifest_sha256")
            or hashlib.sha256(
                (cell_dir.parent / "eligibility_report.json").read_bytes()
            ).hexdigest() != preflight.get("eligibility_sha256")
        ):
            raise ValueError(
                "published cohort authority bytes do not match preflight"
            )
        loader_audit_path = cell_dir.parent / "frozen_loader_audit.json"
        required_frozen = {
            "selection_mode", "schema_version", "loader_code_sha256",
            "frozen_manifest_sha256", "frozen_eligibility_schema",
            "frozen_eligibility_sha256",
            "ordered_identity_schema", "ordered_identity_sha256",
            "ordered_identity_projection", "chunks_source_sha256",
            "objectives_source_sha256", "rehydrated_row_sha256",
        }
        if (
            not isinstance(frozen_selection, Mapping)
            or set(frozen_selection) != required_frozen
            or frozen_selection.get("selection_mode")
            != FROZEN_D_C1_LOADER_VERSION
            or frozen_selection.get("schema_version")
            != FROZEN_D_C1_LOADER_AUDIT_SCHEMA
            or frozen_selection.get("frozen_eligibility_schema")
            != FROZEN_D_C1_ELIGIBILITY_SCHEMA
            or frozen_selection.get("frozen_eligibility_sha256")
            != preflight.get("eligibility_sha256")
            or not _SAFE_HASH.fullmatch(
                str(preflight.get("frozen_loader_audit_sha256") or "")
            )
            or not loader_audit_path.is_file()
            or hashlib.sha256(loader_audit_path.read_bytes()).hexdigest()
            != preflight.get("frozen_loader_audit_sha256")
            or {
                key: json.loads(
                    loader_audit_path.read_text(encoding="utf-8")
                )[key]
                for key in required_frozen
            } != frozen_selection
            or frozen_selection.get("ordered_identity_schema")
            != FROZEN_D_C1_ORDERED_IDENTITY_SCHEMA
            or frozen_selection.get("ordered_identity_sha256")
            != _digest({
                "schema": FROZEN_D_C1_ORDERED_IDENTITY_SCHEMA,
                "rows": frozen_selection.get("ordered_identity_projection"),
            })
        ):
            raise ValueError("frozen selection preflight identity is invalid")
    synthesis_identity = preflight.get("synthesis_contract")
    micro_selected = (
        isinstance(synthesis_identity, Mapping)
        and synthesis_identity.get("version") == MICRO_SYNTHESIS_CONTRACT
    )
    if micro_selected:
        required_commit.add("synthesis_contract")
    if set(commit) != required_commit:
        raise ValueError("success commit schema keys do not match")
    if (
        commit.get("schema_version") != CELL_PUBLICATION_SCHEMA
        or commit.get("status") not in {
            "committed", "committed_complete", "terminal_hold",
        }
        or commit.get("model_id") != expected_model
        or commit.get("pilot_run_id") != expected_run_id
        or commit.get("preflight_sha256") != expected_preflight_sha256
    ):
        raise ValueError("success commit identity does not match")
    artifacts = commit.get("artifacts")
    names = {
        "decision-audit-report.json",
        "http-reconciliation-verification.json",
        "http-reconciliation.json",
        "call-intents.jsonl",
        "results.jsonl",
        "summary.json",
        "telemetry-verification.json",
    }
    if micro_selected:
        names.update(("checkpoint.jsonl", "micro-verification.json"))
    if not isinstance(artifacts, Mapping) or set(artifacts) != names:
        raise ValueError("success commit artifact names do not match")
    actual_hashes = {
        name: hashlib.sha256((artifacts_root / name).read_bytes()).hexdigest()
        for name in sorted(names)
    }
    if dict(artifacts) != actual_hashes:
        raise ValueError("committed artifact digest does not match bytes")
    if frozen_selection is not None:
        expected_result_identity = {
            "selection_mode": FROZEN_D_C1_LOADER_VERSION,
            "ordered_identity_schema": FROZEN_D_C1_ORDERED_IDENTITY_SCHEMA,
            "ordered_identity_sha256": frozen_selection[
                "ordered_identity_sha256"
            ],
        }
        published_results = [
            json.loads(line)
            for line in (artifacts_root / "results.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        published_summary = json.loads(
            (artifacts_root / "summary.json").read_text(encoding="utf-8")
        )
        if (
            not published_results
            or any(
                row.get("frozen_selection_identity") != expected_result_identity
                for row in published_results
            )
            or published_summary.get("frozen_selection_identity")
            != expected_result_identity
        ):
            raise ValueError(
                "published results do not bind frozen ordered identity"
            )
    intent_path = artifacts_root / "call-intents.jsonl"
    intent_bytes = intent_path.read_bytes()
    if (
        commit.get("intent_manifest_sha256")
        != hashlib.sha256(intent_bytes).hexdigest()
        or commit.get("intent_count")
        != len([line for line in intent_bytes.decode("utf-8").splitlines() if line])
    ):
        raise ValueError("committed intent manifest identity does not match bytes")
    audit_path = artifacts_root / "decision-audit-report.json"
    if (
        commit.get("decision_audit_report_sha256")
        != hashlib.sha256(audit_path.read_bytes()).hexdigest()
    ):
        raise ValueError("committed decision audit digest does not match bytes")
    try:
        audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("committed decision audit report is not valid JSON") from exc
    if not _verified_audit_report(audit_report):
        raise ValueError("committed decision audit report is not verified")
    verified_reports = {}
    for field, name, schema in (
        ("telemetry_report_sha256", "telemetry-verification.json",
         TELEMETRY_REPORT_SCHEMA),
        ("reconciliation_report_sha256",
         "http-reconciliation-verification.json",
         RECONCILIATION_REPORT_SCHEMA),
    ):
        report_path = artifacts_root / name
        if commit.get(field) != hashlib.sha256(report_path.read_bytes()).hexdigest():
            raise ValueError(f"committed {name} digest does not match bytes")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        verified_reports[name] = report
        if not verified_report(report, schema):
            raise ValueError(f"committed {name} is not verified")
    commit_body = {
        key: commit[key] for key in required_commit if key != "commit_sha256"
    }
    if commit.get("commit_sha256") != _digest(commit_body):
        raise ValueError("success commit digest does not match artifact manifest")

    if micro_selected:
        if commit.get("synthesis_contract") != synthesis_identity:
            raise ValueError("micro synthesis contract identity does not match")
        micro_report = json.loads(
            (artifacts_root / "micro-verification.json").read_text(
                encoding="utf-8"
            )
        )
        report_hash = micro_report.get("report_sha256")
        report_body = {
            key: value for key, value in micro_report.items()
            if key != "report_sha256"
        }
        if (
            report_hash != _digest(report_body)
            or micro_report.get("contract") != synthesis_identity
            or not micro_report.get("journals")
            or not micro_report.get("by_microstage")
            or not micro_report.get("by_claim_slot")
            or not micro_report.get("microstage_metrics")
        ):
            raise ValueError("micro verification report is not authoritative")
        for journal in micro_report["journals"]:
            journal_path = cell_dir / str(journal.get("path") or "")
            if (
                not journal_path.is_file()
                or hashlib.sha256(journal_path.read_bytes()).hexdigest()
                != journal.get("sha256")
            ):
                raise ValueError("micro journal digest does not match bytes")
        recomputed_micro = verify_micro_journals(
            cell_dir,
            summary=json.loads(
                (artifacts_root / "summary.json").read_text(encoding="utf-8")
            ),
            preflight=preflight,
            results=[
                json.loads(line) for line in (
                    artifacts_root / "results.jsonl"
                ).read_text(encoding="utf-8").splitlines() if line.strip()
            ],
        )
        if recomputed_micro != micro_report:
            raise ValueError("micro verification report differs from journal bytes")
        checkpoint_rows = []
        for line in (artifacts_root / "checkpoint.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.pop("_checkpoint_state", "terminal") != "terminal":
                raise ValueError("committed micro checkpoint is not terminal")
            checkpoint_rows.append(row)
        committed_results = [
            json.loads(line) for line in (
                artifacts_root / "results.jsonl"
            ).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        manifest_rows = [
            json.loads(line)
            for line in (cell_dir.parent / "manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        canonical_rows = canonicalize_micro_publication_rows(
            checkpoint_rows=checkpoint_rows,
            result_rows=committed_results,
            manifest_rows=manifest_rows,
        )
        if checkpoint_rows != canonical_rows or committed_results != canonical_rows:
            raise ValueError(
                "committed micro artifacts are not in canonical manifest order"
            )
    model = (preflight.get("telemetry_identity") or {}).get("expected_model")
    cell_id = preflight.get("cell_id")
    if (
        preflight.get("pilot_run_id") != expected_run_id
        or model != expected_model
        or not isinstance(cell_id, str)
        or not cell_id
        or commit.get("cell_id") != cell_id
    ):
        raise ValueError("preflight model/run identity does not match")
    intent_rows, intent_errors = read_call_intents(intent_path)
    if intent_errors or len(intent_rows) != commit["intent_count"]:
        raise ValueError("committed intent manifest is invalid")
    micro_budgets = {}
    if micro_selected:
        budget_contract = (
            synthesis_identity.get("component_hashes") or {}
        ).get("stage_token_budget")
        if (
            not isinstance(budget_contract, Mapping)
            or budget_contract.get("version") != "micro-stage-max-tokens.v1"
            or not isinstance(budget_contract.get("max_tokens"), Mapping)
        ):
            raise ValueError("micro stage token budgets are not fingerprinted")
        micro_budgets = dict(budget_contract["max_tokens"])

    def expected_intent_tokens(row: Mapping[str, Any]) -> int:
        stage = str(row.get("stage") or "")
        match = re.search(r"micro_([A-F])(?:_|$)", stage)
        if micro_selected and match:
            family = match.group(1)
            if family == "C" or family not in micro_budgets:
                raise ValueError("micro intent stage token budget is invalid")
            return int(micro_budgets[family])
        return int(preflight.get("max_tokens"))

    if any(
        row["run_id"] != expected_run_id
        or row["cell_id"] != cell_id
        or row["model"] != expected_model
        or row["max_tokens"] != expected_intent_tokens(row)
        or row["temperature"] != preflight.get("temperature")
        or row["response_dialect"] != (
            preflight.get("dialect_preflight") or {}
        ).get("response_dialect")
        for row in intent_rows
    ):
        raise ValueError("intent manifest semantic identity does not match preflight")
    if micro_selected and any(
        row.get("intent_version") != "ed4all.provider-call-intent.v2"
        or row.get("synthesis_contract_sha256")
        != synthesis_identity.get("fingerprint")
        for row in intent_rows
    ):
        raise ValueError("micro intent contract binding does not match preflight")
    reconciliation_identity = verified_reports[
        "http-reconciliation-verification.json"
    ].get("intent_identity")
    expected_reconciliation_identity = {
        "run_ids": sorted({row["run_id"] for row in intent_rows}),
        "cell_ids": sorted({row["cell_id"] for row in intent_rows}),
        "models": sorted({row["model"] for row in intent_rows}),
        "contract_sha256s": sorted({
            row["contract_sha256"] for row in intent_rows
        }),
    }
    if reconciliation_identity != expected_reconciliation_identity:
        raise ValueError("reconciliation report intent identity does not match")
    prefix_count = preflight.get("intent_manifest_preflight_count")
    if (
        isinstance(prefix_count, bool)
        or not isinstance(prefix_count, int)
        or prefix_count < 0
        or prefix_count > commit["intent_count"]
    ):
        raise ValueError("preflight intent count is invalid")
    prefix_bytes = b"".join(intent_bytes.splitlines(keepends=True)[:prefix_count])
    if hashlib.sha256(prefix_bytes).hexdigest() != preflight.get(
        "intent_manifest_preflight_sha256"
    ):
        raise ValueError("preflight intent manifest prefix does not match")

    results_bytes = (artifacts_root / "results.jsonl").read_bytes()
    try:
        results = [
            json.loads(line) for line in results_bytes.decode("utf-8").splitlines()
            if line
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("committed results are not valid JSONL") from exc
    canonical_bytes = "".join(f"{_stable(row)}\n" for row in results).encode()
    if results_bytes != canonical_bytes:
        raise ValueError("committed results are not canonical JSONL bytes")
    classified = [dict(row) for row in classified_rows]
    if results != classified:
        raise ValueError("classified rows do not byte/key/state match results")
    for row in results:
        violations = _row_invariant_errors(row)
        if violations:
            raise ValueError(f"committed row invariant violation: {violations[0]}")
        code = str(row.get("error_code") or "")
        if code and code not in _CRITICAL_CELL_ERROR_CODES:
            details = row.get("error_details")
            details = details if isinstance(details, Mapping) else {}
            classified_quality = (
                code.startswith("staged_")
                and code.endswith("_invalid")
                and details.get("terminal_content_rejection") is True
                and details.get("stage") in _STAGE_VALUES
                and bool(
                    details.get("validation_error")
                    or details.get("validator_reason")
                    or details.get("response_ref")
                )
            )
            if not classified_quality:
                raise ValueError(
                    "committed row invariant violation: unclassified_error_code"
                )
    summary = json.loads((artifacts_root / "summary.json").read_text(encoding="utf-8"))
    if summary != summarize(results):
        raise ValueError("committed summary does not match committed results")
    reconciliation = json.loads(
        (artifacts_root / "http-reconciliation.json").read_text(encoding="utf-8")
    )
    if (
        reconciliation.get("status") != "accepted"
        or reconciliation.get("started") != reconciliation.get("terminal")
        or reconciliation.get("errors")
    ):
        raise ValueError("committed reconciliation is not accepted")
    return {
        "status": "verified",
        "units": len(results),
        "commit_sha256": commit["commit_sha256"],
    }


def _models_snapshot(
    base_url: str, *, hard_deadline: Optional[float] = None,
) -> Mapping[str, Any]:
    timeout = 10.0
    if hard_deadline is not None:
        timeout = min(timeout, hard_deadline - time.monotonic())
        if timeout <= 0:
            raise RuntimeError("model preflight exceeded absolute cell deadline")
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/models", timeout=timeout,
    ) as response:
        value = json.loads(response.read())
    if not isinstance(value, Mapping) or not value.get("data"):
        raise RuntimeError("benchmark /models preflight returned no model data")
    return value


def _benchmark_execution_cells(args: Any) -> tuple[list[tuple[int, str]], bool]:
    """Return an advancement sweep or one non-repeating cap arm."""
    if getattr(args, "completion_cap_qualification", False):
        fixed_concurrency = int(args.fixed_qualification_concurrency)
        return [(fixed_concurrency, f"qualification-c{fixed_concurrency}")], True
    return planned_benchmark_cells(
        getattr(args, "stop_after_concurrency", None)
    ), False


def _run_benchmark_matrix(
    *,
    args: Any,
    rows: list[dict[str, Any]],
    objectives: Mapping[str, Mapping[str, Any]],
    run_identity: str,
    manifest_path: Path,
    eligibility_path: Path,
) -> int:
    """Run the <=75-minute operational-validity advancement matrix."""
    matrix_started = time.monotonic()
    matrix_limit = 75 * 60
    global_deadline = matrix_started + matrix_limit
    qualification_route = None
    if getattr(args, "completion_cap_qualification", False):
        qualification_route = qualification_route_binding(
            explicit_base_url=args.qualification_base_url,
            explicit_model=args.qualification_served_model,
            environ=os.environ,
        )
        base_url = qualification_route["base_url"]
    else:
        base_url = os.environ.get(
            "LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8000/v1",
        ).rstrip("/")
    outcomes = []
    highest_operationally_valid: Optional[int] = None
    stop_after_concurrency = getattr(args, "stop_after_concurrency", None)
    cells, repeat_scheduled = _benchmark_execution_cells(args)
    cell_index = 0
    while cell_index < len(cells):
        concurrency, cell_id = cells[cell_index]
        cell_index += 1
        cell_started = time.monotonic()
        cell_deadline = min(
            cell_started + BENCHMARK_CELL_LIMIT_SECONDS, global_deadline,
        )
        initial_budget = remaining_cell_budget(
            cell_started=cell_started, global_deadline=global_deadline,
            now=time.monotonic(),
        )
        if initial_budget <= 30.0:
            break
        model_snapshot = _models_snapshot(base_url, hard_deadline=cell_deadline)
        cell_dir = args.output_dir / cell_id
        from lib.decision_capture import DecisionCapture
        capture_root = cell_dir / "audit" / "runtime/training-captures"
        prior_capture_root = os.environ.get("ED4ALL_TRAINING_CAPTURES_DIR")
        os.environ["ED4ALL_TRAINING_CAPTURES_DIR"] = str(capture_root)
        try:
            capture = DecisionCapture(
                course_code=f"{run_identity}-{cell_id}",
                phase="training-synthesis-window-pilot",
                tool="trainforge",
                streaming=True,
            )
        except BaseException:
            if prior_capture_root is None:
                os.environ.pop("ED4ALL_TRAINING_CAPTURES_DIR", None)
            else:
                os.environ["ED4ALL_TRAINING_CAPTURES_DIR"] = prior_capture_root
            raise
        capture_closed = False
        def close_capture_env() -> None:
            nonlocal capture_closed
            if not capture_closed:
                capture.close()
                capture_closed = True
            if prior_capture_root is None:
                os.environ.pop("ED4ALL_TRAINING_CAPTURES_DIR", None)
            else:
                os.environ["ED4ALL_TRAINING_CAPTURES_DIR"] = prior_capture_root
        try:
            provider = _provider(
                timeout_seconds=args.timeout_seconds,
                transport_attempts=args.transport_attempts,
                initial_backoff_seconds=args.initial_backoff_seconds,
                attempt_ledger_path=cell_dir / "http_attempts.jsonl",
                raw_audit_root=cell_dir / "audit" / "http-raw",
                intent_manifest_path=cell_dir / "call-intents.jsonl",
                intent_run_id=f"{run_identity}-{cell_id}",
                intent_cell_id=cell_id,
                capture=capture,
                synthesis_contract=args.synthesis_contract,
                synthesis_seed=getattr(args, "synthesis_seed", None),
                completion_cap=getattr(args, "completion_cap", 600),
            )
        except BaseException:
            close_capture_env()
            raise
        if args.synthesis_contract == MICRO_SYNTHESIS_CONTRACT:
            # DecisionCapture established its stream path during construction.
            # Micro resume state has a separate, explicit cell-local authority.
            provider._capture.output_dir = cell_dir / "micro-journals"
        provider._pilot_attempt_ledger.configure_deadline(
            hard_deadline=cell_deadline,
            max_timeout_seconds=args.timeout_seconds,
            cleanup_seconds=30.0,
        )
        served_ids = {
            str(item.get("id"))
            for item in model_snapshot.get("data", [])
            if isinstance(item, Mapping)
        }
        if provider._model not in served_ids:
            close_capture_env()
            raise RuntimeError(
                "configured benchmark model is absent from the bound /models "
                "snapshot"
            )
        if (
            qualification_route is not None
            and provider._model != qualification_route["served_model"]
        ):
            close_capture_env()
            raise RuntimeError(
                "configured benchmark model differs from qualification route"
            )
        try:
            dialect = negotiate_dialect(provider)
        except BaseException:
            close_capture_env()
            raise
        remaining_after_dialect = cell_deadline - time.monotonic()
        if remaining_after_dialect <= 30.0:
            close_capture_env()
            break
        from Trainforge.generators.trtllm_benchmark_telemetry import (
            TrtllmTelemetrySampler,
            telemetry_preflight,
        )
        try:
            telemetry_contract = telemetry_preflight(
                expected_model=provider._model,
                model_snapshot=model_snapshot,
                base_url=base_url,
                hard_deadline=cell_deadline,
            )
            provider.bind_verified_served_context(telemetry_contract)
        except BaseException:
            close_capture_env()
            raise
        try:
            preflight = build_preflight_artifact(
                manifest_path=manifest_path, base_url=base_url,
                eligibility_path=eligibility_path,
                model_snapshot=model_snapshot, run_id=f"{run_identity}-{cell_id}",
                server_batch=28, client_concurrency=concurrency,
                timeout_seconds=args.timeout_seconds,
                transport_attempts=args.transport_attempts,
                initial_backoff_seconds=args.initial_backoff_seconds,
                max_tokens=int(provider._base._max_tokens),
                temperature=float(getattr(provider._base, "_temperature", 0.0)),
                dialect=dialect,
                cell_id=cell_id,
                telemetry_identity=telemetry_contract,
                intent_manifest_path=cell_dir / "call-intents.jsonl",
                stop_after_concurrency=stop_after_concurrency,
                synthesis_contract=args.synthesis_contract,
                micro_journal_root=(
                    "micro-journals/micro_synthesis_state"
                    if args.synthesis_contract == MICRO_SYNTHESIS_CONTRACT
                    else None
                ),
                qualification_route=qualification_route,
                frozen_loader_audit_path=(
                    manifest_path.parent / "frozen_loader_audit.json"
                    if args.frozen_d_c1_manifest
                    else None
                ),
            )
            provider._pilot_execution_fingerprint = preflight[
                "run_contract_sha256"
            ]
            provider._pilot_run_id = preflight["pilot_run_id"]
            provider._pilot_cell_id = preflight["cell_id"]
            provider._pilot_manifest_sha256 = preflight["manifest_sha256"]
            provider._pilot_eligibility_sha256 = preflight[
                "eligibility_sha256"
            ]
            _write_immutable_json(
                cell_dir / "preflight.json", preflight,
                hard_deadline=cell_deadline,
            )
            _write_immutable_json(
                cell_dir / "telemetry-preflight.json", telemetry_contract,
                hard_deadline=cell_deadline,
            )
        except BaseException:
            close_capture_env()
            raise
        remaining_after_preflight = cell_deadline - time.monotonic()
        if remaining_after_preflight <= 30.0:
            _write_json_fsync(cell_dir / "failure.json", {
                "cell": cell_id,
                "error_code": "preflight_budget_exhausted",
            }, hard_deadline=cell_deadline)
            close_capture_env()
            break
        sampler = TrtllmTelemetrySampler(
            cell_dir / "telemetry", hard_deadline=cell_deadline - 5.0,
        )
        try:
            sampler.start()
        except BaseException:
            close_capture_env()
            raise
        telemetry_summary = None
        try:
            results, summary = execute_pilot(
                counterbalanced_cell_rows(
                    rows, run_id=run_identity, cell_id=cell_id,
                ), provider, objectives=objectives,
                max_concurrent=concurrency,
                checkpoint_path=cell_dir / "checkpoint.jsonl",
                cell_deadline_monotonic=cell_deadline,
                request_timeout_seconds=args.timeout_seconds,
            )
            frozen_identity = preflight.get("frozen_selection")
            if isinstance(frozen_identity, Mapping):
                result_identity = {
                    "selection_mode": frozen_identity["selection_mode"],
                    "ordered_identity_schema": frozen_identity[
                        "ordered_identity_schema"
                    ],
                    "ordered_identity_sha256": frozen_identity[
                        "ordered_identity_sha256"
                    ],
                }
                results = [
                    {**row, "frozen_selection_identity": result_identity}
                    for row in results
                ]
                summary = {
                    **summary,
                    "frozen_selection_identity": result_identity,
                }
        except Exception as exc:
            failure = {
                "cell": cell_id,
                "error_code": str(
                    getattr(exc, "code", "") or type(exc).__name__
                ),
                "exception_class": type(exc).__name__,
            }
            _write_json_fsync(
                cell_dir / "failure.json", failure,
                hard_deadline=cell_deadline,
            )
            outcomes.append(failure)
            remaining = global_deadline - time.monotonic()
            if (
                "did not drain" not in str(exc)
                and
                highest_operationally_valid is not None
                and not repeat_scheduled
                and remaining >= BENCHMARK_CELL_LIMIT_SECONDS
            ):
                cells = cells[:cell_index]
                cells.append((
                    highest_operationally_valid,
                    f"c{highest_operationally_valid}-repeat",
                ))
                repeat_scheduled = True
                continue
            break
        finally:
            try:
                telemetry_summary = sampler.stop()
            finally:
                close_capture_env()
        if not telemetry_summary or not telemetry_summary.get("iteration_samples"):
            raise RuntimeError(
                "benchmark cell captured no scheduled-token telemetry"
            )
        decision_artifact_path = capture._stream_path
        if not isinstance(decision_artifact_path, Path):
            raise RuntimeError("decision capture did not expose its JSONL artifact")
        decision_artifact_sha256 = hashlib.sha256(
            decision_artifact_path.read_bytes()
        ).hexdigest()
        decision_audit_report = verify_decision_audit(
            decision_artifact_path,
            expected_artifact_sha256=decision_artifact_sha256,
            intent_manifest_path=cell_dir / "call-intents.jsonl",
            capture_closed=capture_closed,
        )
        if not _verified_audit_report(decision_audit_report):
            raise RuntimeError("decision audit verification failed")
        if args.synthesis_contract == MICRO_SYNTHESIS_CONTRACT:
            decision_contexts = []
            for line in decision_artifact_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("decision_type") == "synthesis_provider_call":
                    context = json.loads(event.get("context") or "")
                    if not isinstance(context, Mapping):
                        raise ValueError(
                            "micro provider DecisionCapture context is invalid"
                        )
                    decision_contexts.append(context)
            intent_rows = [
                json.loads(line)
                for line in (cell_dir / "call-intents.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            http_rows = [
                json.loads(line)
                for line in (cell_dir / "http_attempts.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            evidence = verify_micro_evidence_plane_rows(
                intent_rows=intent_rows,
                decision_contexts=decision_contexts,
                http_rows=http_rows,
            )
            summary = apply_micro_transport_outcomes(
                summary, evidence, results=results,
            )
        decision_audit_report_path = cell_dir / "decision-audit-report.json"
        decision_audit_report_sha256 = write_decision_audit_report(
            decision_audit_report_path, decision_audit_report,
        )
        telemetry_report = verify_telemetry_artifacts(
            cell_dir / "telemetry",
            preflight=telemetry_contract,
            expected_preflight_sha256=hashlib.sha256(
                f"{_stable(telemetry_contract)}\n".encode()
            ).hexdigest(),
            expected_container=str(telemetry_contract["container"]),
            expected_port=int(telemetry_contract["published_port"]),
            expected_model=provider._model,
            expected_max_num_tokens=int(telemetry_contract["max_num_tokens"]),
            expected_interval_seconds=float(sampler.interval),
            sampler_state=telemetry_summary["sampler_state"],
        )
        if not verified_report(telemetry_report, TELEMETRY_REPORT_SCHEMA):
            raise RuntimeError("telemetry artifact verification failed")
        telemetry_report_path = cell_dir / "telemetry-verification.json"
        _write_immutable_json(
            telemetry_report_path, telemetry_report,
            hard_deadline=cell_deadline,
        )
        telemetry_report_sha256 = hashlib.sha256(
            telemetry_report_path.read_bytes()
        ).hexdigest()
        ledger_path = cell_dir / "http_attempts.jsonl"
        reconciliation_report = verify_http_reconciliation(
            ledger_path,
            intent_manifest_path=cell_dir / "call-intents.jsonl",
        )
        if not verified_report(
            reconciliation_report, RECONCILIATION_REPORT_SCHEMA,
        ):
            raise RuntimeError("HTTP reconciliation verification failed")
        reconciliation_report_path = (
            cell_dir / "http-reconciliation-verification.json"
        )
        _write_immutable_json(
            reconciliation_report_path,
            reconciliation_report,
            hard_deadline=cell_deadline,
        )
        reconciliation_report_sha256 = hashlib.sha256(
            reconciliation_report_path.read_bytes()
        ).hexdigest()
        expected_cell_rows = counterbalanced_cell_rows(
            rows, run_id=run_identity, cell_id=cell_id,
        )
        precommit_outcome = classify_cell_outcome(
            results,
            expected_rows=expected_cell_rows,
            publication_state={"state": "precommit_verified"},
            reconciliation=reconciliation_report,
            telemetry_summary=telemetry_report,
            decision_audit_report=decision_audit_report,
            active_requests=provider._pilot_transport_probe.active_requests,
            synthesis_summary=summary,
        )
        authority_status = (
            "committed_complete"
            if precommit_outcome["operationally_valid"]
            else "terminal_hold"
        )
        try:
            if time.monotonic() >= cell_deadline - 2.0:
                raise TimeoutError("cell finalization reserve reached")
            publish_cell_success(
                cell_dir=cell_dir,
                ledger_path=ledger_path,
                results=results,
                summary=summary,
                hard_deadline=cell_deadline,
                decision_audit_report_path=decision_audit_report_path,
                decision_audit_report_sha256=decision_audit_report_sha256,
                telemetry_report_path=telemetry_report_path,
                telemetry_report_sha256=telemetry_report_sha256,
                reconciliation_report_path=reconciliation_report_path,
                reconciliation_report_sha256=reconciliation_report_sha256,
                intent_manifest_path=cell_dir / "call-intents.jsonl",
                publication_identity={
                    "model_id": provider._model,
                    "pilot_run_id": f"{run_identity}-{cell_id}",
                    "cell_id": cell_id,
                    "preflight_sha256": hashlib.sha256(
                        (cell_dir / "preflight.json").read_bytes()
                    ).hexdigest(),
                    **({
                        "synthesis_contract": synthesis_contract_identity(
                            args.synthesis_contract
                        ),
                    } if args.synthesis_contract == MICRO_SYNTHESIS_CONTRACT
                    else {}),
                },
                authority_status=authority_status,
            )
        except TimeoutError:
            _write_json_fsync(cell_dir / "failure.json", {
                "cell": cell_id,
                "error_code": "cell_finalization_deadline_exhausted",
                "terminal": True,
            }, hard_deadline=cell_deadline)
            outcomes.append({
                "cell": cell_id,
                "error_code": "cell_finalization_deadline_exhausted",
            })
            break
        publication_state = read_cell_publication_state(cell_dir)
        verify_cell_publication(
            cell_dir,
            expected_model=provider._model,
            expected_run_id=f"{run_identity}-{cell_id}",
            expected_preflight_sha256=hashlib.sha256(
                (cell_dir / "preflight.json").read_bytes()
            ).hexdigest(),
            classified_rows=results,
        )
        outcome = classify_cell_outcome(
            results,
            expected_rows=expected_cell_rows,
            publication_state=publication_state,
            reconciliation=reconciliation_report,
            telemetry_summary=telemetry_report,
            decision_audit_report=decision_audit_report,
            active_requests=provider._pilot_transport_probe.active_requests,
            synthesis_summary=summary,
        )
        outcomes.append({
            "cell": cell_id, **outcome,
        })
        if concurrency == stop_after_concurrency:
            break
        if not outcome["operationally_valid"]:
            remaining = global_deadline - time.monotonic()
            if (
                highest_operationally_valid is not None
                and not repeat_scheduled
                and remaining >= BENCHMARK_CELL_LIMIT_SECONDS
            ):
                cells = cells[:cell_index]
                cells.append((
                    highest_operationally_valid,
                    f"c{highest_operationally_valid}-repeat",
                ))
                repeat_scheduled = True
                continue
            break
        highest_operationally_valid = concurrency
        # Once the planned sweep is exhausted, append one confirmatory repeat
        # of its highest operationally valid cell when the global cap permits it.
        remaining = global_deadline - time.monotonic()
        if (
            cell_index == len(cells)
            and not repeat_scheduled
            and remaining >= BENCHMARK_CELL_LIMIT_SECONDS
        ):
            cells.append((
                highest_operationally_valid,
                f"c{highest_operationally_valid}-repeat",
            ))
            repeat_scheduled = True
    _write_json_fsync(args.output_dir / "matrix_summary.json", migrate_matrix_summary({
        "pilot_run_id": run_identity,
        "elapsed_seconds": time.monotonic() - matrix_started,
        "highest_operationally_valid_concurrency": highest_operationally_valid,
        "stop_after_concurrency": stop_after_concurrency,
        "outcomes": outcomes,
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks_jsonl", type=Path)
    parser.add_argument("objectives_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--benchmark-matrix", action="store_true")
    parser.add_argument(
        "--completion-cap-qualification",
        action="store_true",
        help=(
            "run one frozen empirical completion-cap arm with full benchmark "
            "reconciliation; rows are benchmark-only and never published"
        ),
    )
    parser.add_argument(
        "--fixed-qualification-concurrency",
        type=int,
        default=19,
        choices=tuple(range(1, 33)),
        help="single completion-cap arm concurrency (maximum 32)",
    )
    parser.add_argument("--qualification-base-url")
    parser.add_argument("--qualification-served-model")
    parser.add_argument(
        "--stop-after-concurrency",
        type=int,
        choices=BENCHMARK_CONCURRENCY_CELLS,
        help=(
            "stop after committing/classifying this benchmark cell; default "
            "preserves the complete legacy advancement matrix"
        ),
    )
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--pilot-run-id")
    parser.add_argument(
        "--synthesis-contract",
        default=LEGACY_SYNTHESIS_CONTRACT,
        choices=tuple(SYNTHESIS_CONTRACT_ALIASES),
        help=(
            "explicit provider contract; aliases are intentionally limited to "
            "'legacy' and the authoritative micro contract ID"
        ),
    )
    parser.add_argument(
        "--synthesis-seed",
        type=int,
        help=(
            "explicit unsigned 64-bit micro-synthesis seed; required for "
            "executing the micro contract and intentionally has no default"
        ),
    )
    parser.add_argument(
        "--completion-cap",
        type=int,
        default=600,
        choices=(600, 800, 1000, 1200, 1600),
        help=(
            "temporary strict Stage-D completion ceiling for the bounded "
            "five-cap qualification; overflow is rejected, never truncated"
        ),
    )
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-eligibility-sha256")
    parser.add_argument(
        "--frozen-d-c1-manifest",
        type=Path,
        help=(
            "explicit selection-only canonical eight-row D-C1 manifest; "
            "requires both frozen manifest and ordered-identity SHA-256 pins"
        ),
    )
    parser.add_argument("--frozen-d-c1-manifest-sha256")
    parser.add_argument("--frozen-d-c1-ordered-identity-sha256")
    parser.add_argument(
        "--frozen-d-c1-eligibility", type=Path,
        help="historical frozen eligibility JSON authority",
    )
    parser.add_argument("--frozen-d-c1-eligibility-sha256")
    parser.add_argument(
        "--gate-d-go-artifact", type=Path,
        help=(
            "signed reviewed single-row Gate-D GO; requires micro-v1 and the "
            "fully pinned frozen-eight authority"
        ),
    )
    parser.add_argument(
        "--gate-d-functional-canary", action="store_true",
        help=(
            "standing-authorized v1.3 functional one-row canary; uses frozen "
            "input hashes and direct observable preflight, never crypto tickets"
        ),
    )
    parser.add_argument(
        "--gate-d-functional-production-repairs", action="store_true",
        help=(
            "functional canary only: exercise existing fingerprinted micro-v1 "
            "production semantic/leakage repair loops under a bounded ceiling"
        ),
    )
    parser.add_argument("--gate-d-functional-row-id")
    parser.add_argument("--gate-d-functional-row-sha256")
    parser.add_argument(
        "--gate-d-functional-authority", type=Path,
        help="explicit operator-owned functional Gate-D authority document",
    )
    parser.add_argument("--gate-d-signed-wrapper", type=Path)
    parser.add_argument("--gate-d-capability-proof", type=Path)
    parser.add_argument("--gate-d-wp11-public-key", type=Path)
    parser.add_argument("--gate-a-authority", type=Path)
    parser.add_argument("--gate-d-preflight-observe-only", action="store_true")
    parser.add_argument("--gate-d-control-endpoint")
    parser.add_argument("--gate-d-served-model")
    parser.add_argument("--gate-d-model-revision")
    parser.add_argument("--gate-d-backend")
    parser.add_argument("--gate-d-backend-version")
    parser.add_argument("--gate-d-schema-sha256")
    parser.add_argument("--gate-d-max-context-tokens", type=int)
    parser.add_argument("--gate-d-max-output-tokens", type=int)
    parser.add_argument("--gate-d-backend-config", type=Path)
    parser.add_argument("--gate-d-projection-schema", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--transport-attempts", type=int, default=1)
    parser.add_argument("--initial-backoff-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.benchmark_matrix and args.timeout_seconds != 240.0:
        parser.error("--benchmark-matrix requires --timeout-seconds 240")
    if args.benchmark_matrix and args.transport_attempts != 1:
        parser.error("--benchmark-matrix requires --transport-attempts 1")
    if args.stop_after_concurrency is not None and not args.benchmark_matrix:
        parser.error("--stop-after-concurrency requires --benchmark-matrix")
    if args.execute and args.completion_cap_qualification and not all((
        args.qualification_base_url, args.qualification_served_model,
    )):
        parser.error(
            "executed completion-cap qualification requires explicit base URL "
            "and served model route binding"
        )
    if args.execute and args.completion_cap_qualification:
        try:
            qualification_route_binding(
                explicit_base_url=args.qualification_base_url,
                explicit_model=args.qualification_served_model,
                environ=os.environ,
            )
        except ValueError as exc:
            parser.error(str(exc))
    frozen_args = (
        args.frozen_d_c1_manifest,
        args.frozen_d_c1_manifest_sha256,
        args.frozen_d_c1_ordered_identity_sha256,
        args.frozen_d_c1_eligibility,
        args.frozen_d_c1_eligibility_sha256,
    )
    if args.gate_d_functional_canary and args.gate_d_go_artifact:
        parser.error("functional Gate D cannot be combined with crypto Gate D")
    if (
        args.gate_d_functional_canary
        and args.gate_d_functional_authority is None
    ):
        parser.error(
            "--gate-d-functional-canary requires "
            "--gate-d-functional-authority"
        )
    if (
        args.gate_d_functional_production_repairs
        and not args.gate_d_functional_canary
    ):
        parser.error("production repairs require --gate-d-functional-canary")
    if any(bool(value) for value in frozen_args) and not all(
        bool(value) for value in frozen_args
    ):
        parser.error(
            "--frozen-d-c1-manifest, --frozen-d-c1-manifest-sha256, and "
            "--frozen-d-c1-ordered-identity-sha256, "
            "--frozen-d-c1-eligibility, and "
            "--frozen-d-c1-eligibility-sha256 must be supplied together"
        )
    if (
        args.frozen_d_c1_manifest
        and not args.benchmark_matrix
        and not args.gate_d_go_artifact
        and not args.gate_d_functional_canary
    ):
        parser.error(
            "--frozen-d-c1-manifest requires --benchmark-matrix or "
            "--gate-d-go-artifact"
        )
    if args.gate_d_go_artifact and (
        args.benchmark_matrix
        or not args.execute
        or args.synthesis_contract not in {
            MICRO_SYNTHESIS_CONTRACT, "micro-v1",
        }
        or not all(bool(value) for value in frozen_args)
        or args.max_concurrent != 1
        or args.transport_attempts != 1
    ):
        parser.error(
            "Gate D requires --execute, micro-v1, the complete frozen-eight "
            "pins, max-concurrent=1, transport-attempts=1, and no matrix"
        )
    if args.gate_d_functional_canary and (
        args.benchmark_matrix
        or not args.execute
        or args.synthesis_contract not in {
            MICRO_SYNTHESIS_CONTRACT, "micro-v1",
        }
        or not all(bool(value) for value in frozen_args)
        or args.max_concurrent != 1
        or args.transport_attempts != 1
        or args.synthesis_seed != 0
        or not args.gate_d_control_endpoint
        or not args.gate_d_served_model
        or not args.gate_d_backend_config
        or not args.gate_d_projection_schema
        or not args.gate_d_functional_row_id
        or not args.gate_d_functional_row_sha256
    ):
        parser.error(
            "functional Gate D requires --execute, micro-v1, seed 0, complete "
            "frozen-eight pins, explicit row ID/SHA, max-concurrent=1, one "
            "transport attempt, no matrix"
        )
    if args.gate_d_go_artifact and not all((
        args.gate_d_signed_wrapper, args.gate_d_capability_proof,
        args.gate_d_wp11_public_key,
        args.gate_a_authority,
    )):
        parser.error(
            "Gate D requires signed wrapper, capability proof, operator WP11 "
            "public-key path, and the immutable Gate A trust authority"
        )
    if args.gate_d_preflight_observe_only:
        required_observe = (
            args.gate_d_control_endpoint, args.gate_d_served_model,
            args.gate_d_model_revision, args.gate_d_backend,
            args.gate_d_backend_version, args.gate_d_schema_sha256,
            args.gate_d_max_context_tokens, args.gate_d_max_output_tokens,
        )
        if not all(value is not None for value in required_observe):
            parser.error("Gate D observe-only requires complete control-plane identity")
        from Trainforge.scripts.harness.gate_d_single_row import (
            collect_control_plane_evidence,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        collect_control_plane_evidence(
            endpoint=args.gate_d_control_endpoint,
            output_path=args.output_dir / "control-plane-evidence.json",
            served_model=args.gate_d_served_model,
            model_revision=args.gate_d_model_revision,
            backend=args.gate_d_backend,
            backend_version=args.gate_d_backend_version,
            schema_sha256=args.gate_d_schema_sha256,
            max_context_tokens=args.gate_d_max_context_tokens,
            max_output_tokens=args.gate_d_max_output_tokens,
            thinking_enabled=False,
        )
        return 0
    if (args.gate_d_go_artifact or args.gate_d_functional_canary) and (
        args.expected_manifest_sha256
        != args.frozen_d_c1_manifest_sha256
        or args.expected_eligibility_sha256
        != args.frozen_d_c1_eligibility_sha256
    ):
        parser.error(
            "Gate D expected manifest/eligibility hashes must exactly equal "
            "the frozen authority pins"
        )
    if (
        args.frozen_d_c1_manifest
        and args.expected_eligibility_sha256
        and args.expected_eligibility_sha256
        != args.frozen_d_c1_eligibility_sha256
    ):
        parser.error(
            "--expected-eligibility-sha256 must equal the frozen eligibility pin"
        )
    chunks = [
        json.loads(line) for line in args.chunks_jsonl.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    objectives = load_objectives(args.objectives_json)
    if args.frozen_d_c1_manifest:
        rows, eligibility_audit = load_frozen_d_c1_manifest(
            args.frozen_d_c1_manifest,
            expected_sha256=args.frozen_d_c1_manifest_sha256,
            expected_ordered_identity_sha256=(
                args.frozen_d_c1_ordered_identity_sha256
            ),
            chunks=chunks,
            objectives=objectives,
            chunks_path=args.chunks_jsonl,
            objectives_path=args.objectives_json,
            frozen_eligibility_path=args.frozen_d_c1_eligibility,
            expected_eligibility_sha256=args.frozen_d_c1_eligibility_sha256,
        )
    elif args.completion_cap_qualification:
        from Trainforge.scripts.harness.completion_cap_qualification import (
            build_empirical_set_cover_manifest,
            cap_manifests,
            qualification_runner_draft_identity,
            qualification_runner_preflight,
        )
        rows, eligibility_audit = build_empirical_set_cover_manifest(
            chunks, objectives=objectives,
        )
        runner_preflight = qualification_runner_preflight(rows)
        for row in rows:
            if (
                _canonical_micro_draft_identity(row)
                != qualification_runner_draft_identity(row)
            ):
                parser.error(
                    "canonical micro draft identity differs from qualification "
                    "runner identity"
                )
        if eligibility_audit.get("preflight_status") != "ready":
            parser.error(
                "completion-cap qualification cohort is not ready: "
                + ",".join(eligibility_audit.get("status_reasons") or [])
            )
        if args.fixed_qualification_concurrency > len(rows):
            parser.error(
                "fixed qualification concurrency exceeds frozen row count"
            )
        arm = cap_manifests(rows)[args.completion_cap]
        if (
            args.fixed_qualification_concurrency
            != arm["runner_identity"]["client_concurrency"]
        ):
            parser.error(
                "fixed qualification concurrency differs from frozen runner "
                "identity"
            )
        eligibility_audit = {
            **eligibility_audit,
            "qualification_only": True,
            "publication_denied": True,
            "completion_cap_arm": arm,
            "runner_preflight": runner_preflight,
        }
    elif args.benchmark_matrix:
        rows, eligibility_audit = build_benchmark_rows(
            chunks, objectives=objectives,
        )
    else:
        rows, eligibility_audit = build_pilot_manifest(
            chunks, objectives=objectives, repetitions=args.repetitions,
        )
    run_identity = args.pilot_run_id or (
        f"pilot-{PILOT_VERSION}-{_digest(rows)[:16]}"
    )
    args.synthesis_contract = resolve_synthesis_contract(
        args.synthesis_contract
    )
    gate_d_controller = None
    dispatched = None
    gate_d_secure_tree = None
    gate_d_prior_umask = None
    if args.gate_d_go_artifact:
        from Trainforge.scripts.harness.gate_d_single_row import authorize_single_row
        selected, dispatched = authorize_single_row(
            rows=rows,
            full_manifest_sha256=args.frozen_d_c1_manifest_sha256,
            eligibility_sha256=args.frozen_d_c1_eligibility_sha256,
            ordered_identity_sha256=args.frozen_d_c1_ordered_identity_sha256,
            synthesis_seed=args.synthesis_seed, run_id=run_identity,
            output_dir=args.output_dir, go_path=args.gate_d_go_artifact,
            wrapper_path=args.gate_d_signed_wrapper,
            capability_path=args.gate_d_capability_proof,
            trust_root_path=args.gate_d_wp11_public_key,
            gate_a_authority_path=args.gate_a_authority,
        )
        from Trainforge.scripts.harness.gate_d_single_row import (
            SecureOutputTree, gate_a_trusted_output_root,
        )
        gate_d_prior_umask = os.umask(0o077)
        real_output_dir = args.output_dir.absolute()
        gate_d_secure_tree = SecureOutputTree(
            trusted_root=gate_a_trusted_output_root(args.gate_a_authority),
            output_dir=real_output_dir / "transaction-v1",
        )
        args.output_dir = gate_d_secure_tree.path
        full_rows = rows
        rows = [selected]
    elif args.gate_d_functional_canary:
        from Trainforge.scripts.harness.gate_d_single_row import (
            authorize_functional_single_row,
        )
        selected, dispatched = authorize_functional_single_row(
            rows=rows,
            full_manifest_sha256=args.frozen_d_c1_manifest_sha256,
            eligibility_sha256=args.frozen_d_c1_eligibility_sha256,
            ordered_identity_sha256=args.frozen_d_c1_ordered_identity_sha256,
            synthesis_seed=args.synthesis_seed,
            run_id=run_identity,
            output_dir=args.output_dir,
            expected_chunk_id=args.gate_d_functional_row_id,
            expected_chunk_sha256=args.gate_d_functional_row_sha256,
            plan_path=args.gate_d_functional_authority,
        )
        # Ordinary local safety: private fresh directory plus atomic/fsynced
        # artifacts. No custom trust root or retained-dirfd security gate.
        args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        gate_d_prior_umask = os.umask(0o077)
        from Trainforge.scripts.harness.gate_d_single_row import (
            collect_functional_preflight,
        )
        functional_preflight = collect_functional_preflight(
            endpoint=args.gate_d_control_endpoint,
            backend_config_path=args.gate_d_backend_config,
            schema_path=args.gate_d_projection_schema,
            output_dir=args.output_dir / "functional-preflight",
            expected_model=args.gate_d_served_model,
        )
        dispatched.pop("subset_sha256")
        dispatched["strict_dialect_capability_sha256"] = _digest(
            functional_preflight
        )
        dispatched["subset_sha256"] = _digest(dispatched)
        full_rows = rows
        rows = [selected]
    else:
        full_rows = rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    manifest.write_text("".join(
        f"{_stable({key: value for key, value in row.items() if key != '_chunk'})}\n"
        for row in full_rows
    ), encoding="utf-8")
    if args.frozen_d_c1_manifest:
        (args.output_dir / "eligibility_report.json").write_bytes(
            args.frozen_d_c1_eligibility.read_bytes()
        )
        (args.output_dir / "frozen_loader_audit.json").write_text(
            f"{json.dumps(eligibility_audit, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    else:
        (args.output_dir / "eligibility_report.json").write_text(
            f"{json.dumps(eligibility_audit, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    if (
        args.execute
        and args.synthesis_contract == MICRO_SYNTHESIS_CONTRACT
        and args.synthesis_seed is None
    ):
        parser.error(
            "--synthesis-seed is required when executing the micro contract"
        )
    if args.synthesis_contract == MICRO_SYNTHESIS_CONTRACT:
        from Trainforge.generators.staged_synthesis_micro import (
            verify_expected_input_hashes,
        )
        verify_expected_input_hashes(
            manifest_path=manifest,
            eligibility_path=args.output_dir / "eligibility_report.json",
            expected_manifest_sha256=args.expected_manifest_sha256 or "",
            expected_eligibility_sha256=(
                args.expected_eligibility_sha256 or ""
            ),
        )
    if args.gate_d_go_artifact or args.gate_d_functional_canary:
        from Trainforge.scripts.harness.gate_d_single_row import (
            GateDCallController, write_unconsumed,
        )
        dispatched_path = args.output_dir / "gate-d-dispatched-subset.json"
        dispatched_path.write_text(
            json.dumps(dispatched, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state_path = args.output_dir / "gate-d-consumption.json"
        checkpoint_path = (
            args.output_dir / f"{run_identity}.checkpoint.jsonl"
        )
        if checkpoint_path.exists():
            parser.error("Gate D forbids resume/replay from an existing checkpoint")
        write_unconsumed(state_path, dispatched)
        gate_d_controller = GateDCallController(
            state_path=state_path, binding=dispatched,
            production_repairs=args.gate_d_functional_production_repairs,
            max_calls=50 if args.gate_d_functional_production_repairs else 7,
        )
    if not args.execute:
        return 0
    if args.benchmark_matrix or args.completion_cap_qualification:
        return _run_benchmark_matrix(
            args=args, rows=rows, objectives=objectives,
            run_identity=run_identity, manifest_path=manifest,
            eligibility_path=args.output_dir / "eligibility_report.json",
        )
    gate_d_capture = None
    gate_d_prior_capture_root = None
    if gate_d_controller is not None:
        from lib.decision_capture import DecisionCapture
        gate_d_prior_capture_root = os.environ.get(
            "ED4ALL_TRAINING_CAPTURES_DIR"
        )
        os.environ["ED4ALL_TRAINING_CAPTURES_DIR"] = str(
            args.output_dir / "audit/decision-capture"
        )
        gate_d_capture = DecisionCapture(
            course_code=run_identity,
            phase="training-synthesis-gate-d-single-row",
            tool="trainforge",
            streaming=True,
        )
    provider = _provider(
        timeout_seconds=args.timeout_seconds,
        transport_attempts=args.transport_attempts,
        initial_backoff_seconds=args.initial_backoff_seconds,
        attempt_ledger_path=args.output_dir / "http_attempts.jsonl",
        raw_audit_root=(
            args.output_dir / "audit/http-raw"
            if gate_d_controller is not None else None
        ),
        intent_manifest_path=(
            args.output_dir / "call-intents.jsonl"
            if gate_d_controller is not None else None
        ),
        intent_run_id=run_identity if gate_d_controller is not None else None,
        intent_cell_id="gate-d-single-row" if gate_d_controller is not None else None,
        capture=gate_d_capture,
        synthesis_contract=args.synthesis_contract,
        synthesis_seed=args.synthesis_seed,
        completion_cap=args.completion_cap,
    )
    if args.gate_d_functional_canary:
        effective_base_url = str(
            getattr(getattr(provider, "_base", None), "_base_url", "")
        ).rstrip("/")
        effective_model = str(getattr(provider, "_model", ""))
        expected_base_url = str(functional_preflight["endpoint"]).rstrip("/")
        expected_model = str(functional_preflight["served_model"])
        if (
            effective_base_url != expected_base_url
            or effective_model != expected_model
        ):
            raise RuntimeError(
                "functional Gate D provider identity does not match preflight: "
                f"provider_base_url={effective_base_url!r}, "
                f"preflight_endpoint={expected_base_url!r}, "
                f"provider_model={effective_model!r}, "
                f"preflight_served_model={expected_model!r}"
            )
    gate_d_sampler = None
    if gate_d_controller is not None:
        provider._capture.output_dir = args.output_dir / "micro-journals"
        from Trainforge.generators.trtllm_benchmark_telemetry import (
            TrtllmTelemetrySampler,
        )
        gate_d_sampler = TrtllmTelemetrySampler(
            args.output_dir / "telemetry"
        )
        gate_d_sampler.start()
    if gate_d_controller is None:
        negotiate_dialect(provider)
    else:
        if args.gate_d_functional_canary:
            served_context = int(functional_preflight["max_context_tokens"])
            provider.bind_verified_served_context({
                "status": "accepted",
                "served_context_tokens": served_context,
                "parser_probe": {
                    "static_kv_startup_facts": {
                        "max_seq_len": served_context,
                    },
                },
            })
        if args.gate_d_functional_production_repairs:
            policy_identity = provider.execution_policy_identity()
            focus = provider._focus(rows[0]["_chunk"])
            eligible = provider._routed_claim_blocks(
                rows[0]["_chunk"], focus=focus,
            )
            planned = [{
                "family": "A", "slot": None,
                "stage": "micro_A_task_design", "model_call": False,
            }]
            planned.extend({
                "family": "B", "slot": slot,
                "stage": f"micro_B_claim_{slot}_attempt_1",
                "model_call": True,
            } for slot in range(min(3, len(eligible))))
            planned.extend([
                {"family": "C", "slot": None,
                 "stage": "micro_C_assembly", "model_call": False},
                {"family": "D", "slot": None,
                 "stage": "micro_D_dpo_chosen", "model_call": True},
                {"family": "E", "slot": None,
                 "stage": "micro_E_misconception_selection",
                 "model_call": True},
                {"family": "F", "slot": None,
                 "stage": "micro_F_one_fault_rejected", "model_call": True},
            ])
        else:
            # Core accepts single-pass only through the verified exact subset.
            policy_identity = provider.bind_trusted_gate_d_single_pass(dispatched)
            planned = provider.planned_stage_identities(
                rows[0]["_chunk"], kind="preference",
            )
        model_stages = [
            item["stage"] for item in planned if item["model_call"]
        ]
        gate_d_controller = GateDCallController(
            state_path=args.output_dir / "gate-d-consumption.json",
            binding=dispatched, expected_stages=model_stages,
            production_repairs=args.gate_d_functional_production_repairs,
            max_calls=45 if args.gate_d_functional_production_repairs else 6,
        )
        gate_d_controller.wrap(provider)
        gate_d_controller.install_http_attempt_hook(
            provider._pilot_attempt_ledger
        )
        provider._pilot_gate_d_binding = gate_d_controller.binding
        provider._pilot_execution_fingerprint = gate_d_controller.binding[
            "subset_sha256"
        ]
        provider._pilot_run_id = run_identity
        provider._pilot_cell_id = "gate-d-single-row"
        provider._pilot_manifest_sha256 = gate_d_controller.binding[
            "full_manifest_sha256"
        ]
        provider._pilot_eligibility_sha256 = gate_d_controller.binding[
            "eligibility_sha256"
        ]
        _write_immutable_json(
            args.output_dir / "gate-d-request-schedule.json",
            {
                "binding_sha256": dispatched["subset_sha256"],
                "policy": policy_identity,
                "stages": planned,
                **({
                    "functional_version": "1.3.1",
                } if args.gate_d_functional_canary else {}),
                **({
                    "functional_policy_mode": "production-repair-loops",
                    "maximum_model_calls": 45,
                } if args.gate_d_functional_production_repairs else {}),
            },
        )
    try:
        results, summary = execute_pilot(
            rows, provider, objectives=objectives,
            max_concurrent=args.max_concurrent,
            checkpoint_path=args.output_dir / f"{run_identity}.checkpoint.jsonl",
        )
    except BaseException:
        if gate_d_sampler is not None:
            gate_d_sampler.stop()
        if gate_d_controller is not None:
            gate_d_controller.terminal(outcome="failed")
            gate_d_capture.close()
            if gate_d_prior_capture_root is None:
                os.environ.pop("ED4ALL_TRAINING_CAPTURES_DIR", None)
            else:
                os.environ["ED4ALL_TRAINING_CAPTURES_DIR"] = (
                    gate_d_prior_capture_root
                )
            os.umask(gate_d_prior_umask)
        raise
    if gate_d_controller is not None:
        gate_d_telemetry = gate_d_sampler.stop()
        if args.gate_d_functional_canary:
            ledger_rows = [
                json.loads(line) for line in (
                    args.output_dir / "http_attempts.jsonl"
                ).read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            terminal_http = [
                row for row in ledger_rows
                if row.get("event") == "http_attempt_terminal"
            ]
            from Trainforge.scripts.harness.gate_d_single_row import (
                collect_functional_postflight, functional_reasoning_bytes,
            )
            last_terminal = max(
                float(row["monotonic_seconds"]) for row in terminal_http
            )
            functional_postflight = collect_functional_postflight(
                endpoint=args.gate_d_control_endpoint,
                backend_config_path=args.gate_d_backend_config,
                preflight=functional_preflight,
                output_dir=args.output_dir / "functional-postflight",
                last_terminal_monotonic_seconds=last_terminal,
            )
            response_raw = [
                Path(row["response_raw_ref"]["path"]).read_bytes()
                for row in terminal_http
            ]
            system_path = args.output_dir / "telemetry" / "system.jsonl"
            system_samples = [
                json.loads(line) for line in system_path.read_text(
                    encoding="utf-8"
                ).splitlines() if line.strip()
            ]
            gpu_observations = [
                {
                    "timestamp": fields[0],
                    "gpu_utilization_percent": float(fields[1]),
                    "memory_utilization_percent": float(fields[2]),
                    "power_watts": float(fields[3]),
                    "temperature_c": float(fields[4]),
                }
                for sample in system_samples
                for raw_gpu in (
                    sample.get("gpu")
                    if isinstance(sample.get("gpu"), list) else []
                )
                for fields in ([part.strip() for part in raw_gpu.split(",")],)
                if len(fields) == 5
            ]
            token_observations = []
            for row in terminal_http:
                usage = dict(row.get("usage") or {})
                stage = str(row.get("stage") or "").removeprefix(
                    "staged_synthesis:"
                )
                family = __import__("re").search(
                    r"micro_([ABDEF])(?:_|$)", stage,
                ).group(1)
                cap = {
                    "A": 2048, "B": 1536, "D": 1536,
                    "E": 1280, "F": 1024,
                }[family]
                completion = int(usage.get("completion_tokens", -1))
                token_observations.append({
                    "stage": stage,
                    "prompt_tokens": int(usage.get("prompt_tokens", -1)),
                    "completion_tokens": completion,
                    "total_tokens": int(usage.get("total_tokens", -1)),
                    "max_output_tokens": cap,
                    "output_headroom_tokens": cap - completion,
                })
            raw_sampler_sources = {}
            for name in ("trtllm.log", "system.jsonl"):
                path = args.output_dir / "telemetry" / name
                raw_sampler_sources[name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            gate_d_telemetry.update({
                "request_count": len(terminal_http),
                "stage_request_counts": dict(Counter(
                    str(row.get("stage") or "").removeprefix(
                        "staged_synthesis:"
                    )
                    for row in terminal_http
                )),
                "active_clients_final": int(
                    functional_postflight["active_clients"]
                ),
                "postflight_observed_monotonic_seconds": (
                    functional_postflight["observed_monotonic_seconds"]
                ),
                "reasoning_bytes": functional_reasoning_bytes(response_raw),
                "finish_reason_counts": dict(Counter(
                    str(row.get("finish_reason") or "")
                    for row in terminal_http
                )),
                "token_observations": token_observations,
                "gpu_observations": gpu_observations,
                "kv_observations": [{
                    "source": "sampler_static_startup",
                    **dict(gate_d_telemetry.get("static_kv_startup_facts") or {}),
                    "peak_scheduled_token_usage": gate_d_telemetry.get(
                        "peak_scheduled_token_usage"
                    ),
                    "peak_scheduled_token_headroom": gate_d_telemetry.get(
                        "peak_scheduled_token_headroom"
                    ),
                }],
                "raw_sampler_sources": raw_sampler_sources,
                "verifier_accepted": all(
                    row.get("finish_reason") == "stop"
                    and row.get("exception_class") is None
                    for row in terminal_http
                ),
            })
            _atomic_write_bytes(
                args.output_dir / "telemetry" / "summary.json",
                (json.dumps(
                    gate_d_telemetry, indent=2, sort_keys=True,
                ) + "\n").encode(),
            )
        gate_d_outcome = gate_d_controller.terminal(
            outcome="completed", evidence_root=args.output_dir,
        )
        if gate_d_outcome != "completed":
            gate_d_capture.close()
            raise RuntimeError(
                "Gate D did not complete one exact A-B-D-E-F traversal"
            )
        summary["gate_d_binding"] = gate_d_controller.binding
        gate_d_capture.close()
        if gate_d_prior_capture_root is None:
            os.environ.pop("ED4ALL_TRAINING_CAPTURES_DIR", None)
        else:
            os.environ["ED4ALL_TRAINING_CAPTURES_DIR"] = (
                gate_d_prior_capture_root
            )
    summary["pilot_run_id"] = run_identity
    results_bytes = "".join(
        f"{_stable(row)}\n" for row in results
    ).encode()
    summary_bytes = (
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    ).encode()
    if gate_d_controller is not None:
        from Trainforge.scripts.harness.gate_d_single_row import (
            _offline_pair_validator, verify_gate_d_precommit,
            verify_full_gate_d_transaction,
        )
        _pair_validator, schema_registry = _offline_pair_validator(
            Path("schemas/knowledge/preference_pair.schema.json"),
        )
        consumption_path = args.output_dir / "gate-d-consumption.json"
        schedule_path = args.output_dir / "gate-d-request-schedule.json"
        candidate_path = args.output_dir / "gate-d-precommit-candidate.json"
        _write_immutable_json(candidate_path, {
            "schema": "ed4all.gate-d-precommit-candidate.v1",
            "binding_sha256": dispatched["subset_sha256"],
            "policy_sha256": policy_identity["sha256"],
            "schedule_sha256": hashlib.sha256(
                schedule_path.read_bytes()
            ).hexdigest(),
            "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "consumption_sha256": hashlib.sha256(
                consumption_path.read_bytes()
            ).hexdigest(),
            "schema_registry_sha256": schema_registry["sha256"],
        })
        verified = verify_gate_d_precommit(
            candidate_path,
            expected_binding_sha256=dispatched["subset_sha256"],
            consumption_path=consumption_path,
        )
        _atomic_write_bytes(args.output_dir / "results.jsonl", results_bytes)
        _atomic_write_bytes(args.output_dir / "summary.json", summary_bytes)
        full_verification = verify_full_gate_d_transaction(
            args.output_dir,
            expected_stages=model_stages,
            expected_binding_sha256=dispatched["subset_sha256"],
            candidate_path=candidate_path,
            authority_paths=(
                {
                    "go_path": args.gate_d_go_artifact,
                    "wrapper_path": args.gate_d_signed_wrapper,
                    "capability_path": args.gate_d_capability_proof,
                    "trust_root_path": args.gate_d_wp11_public_key,
                    "gate_a_authority_path": args.gate_a_authority,
                }
                if args.gate_d_go_artifact else None
            ),
        )
        _write_immutable_json(
            args.output_dir / "gate-d-full-verification.json",
            full_verification,
        )
        _write_immutable_json(args.output_dir / "publication.json", {
            "schema": "ed4all.gate-d-transactional-publication.v1",
            **verified,
            "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "projection": "ed4all-dpo-preference.v2",
            "transaction_sha256": full_verification["transaction_sha256"],
            "verified_artifact_count": full_verification["artifact_count"],
            "state": "committed_complete",
        })
        if gate_d_secure_tree is not None:
            gate_d_secure_tree.assert_identity()
            gate_d_secure_tree.close()
        os.umask(gate_d_prior_umask)
    else:
        (args.output_dir / "results.jsonl").write_bytes(results_bytes)
        (args.output_dir / "summary.json").write_bytes(summary_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
