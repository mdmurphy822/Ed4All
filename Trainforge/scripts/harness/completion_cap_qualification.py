"""Offline manifest and decision harness for completion-cap qualification.

This module does not call a model.  It freezes the exact eligible workload
used by every candidate cap and adjudicates completed observations using the
selection rule in ``plans/c1-worked-answer-completeness-schema-amendment.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

from Trainforge.generators.objective_execution_contract import (
    CANDIDATE_COMPLETION_CAPS,
    content_sha256,
    derive_objective_requirements,
    objective_requirement_contract_components,
)
from Trainforge.generators.staged_synthesis_micro import (
    micro_contract_fingerprint,
    micro_preference_eligibility,
)
from Trainforge.generators.synthesis_window_contract import (
    build_evidence_window,
    objective_card,
)
from Trainforge.scripts.maintenance.runtime_focus import apply_runtime_focus
from Trainforge.synthesis.synthesis_eligibility import pair_eligibility

CAP_QUALIFICATION_CONTRACT = "ed4all.completion-cap-qualification.v1"
BLOOM_LEVELS = (
    "remember", "understand", "apply", "analyze", "evaluate", "create",
)
REQUIREMENT_BUCKETS = ("1-2", "3-4", "5-6", "7+")
TASK_FAMILIES = ("declarative", "procedural", "analytical", "computational")
SOURCE_CARDINALITIES = ("single", "multi")
PAIR_KINDS = ("instruction", "preference")

_TASK_FAMILY = {
    "define": "declarative",
    "identify": "declarative",
    "explain": "declarative",
    "summarize": "declarative",
    "classify": "declarative",
    "apply": "procedural",
    "debug": "procedural",
    "construct": "procedural",
    "compare": "analytical",
    "analyze": "analytical",
    "predict": "analytical",
    "infer": "analytical",
    "critique": "analytical",
    "evaluate": "analytical",
    "compute": "computational",
}


class CapQualificationError(ValueError):
    """A frozen workload or observation set is not qualification-safe."""


def qualification_contract_components() -> dict[str, Any]:
    """Return every local behavior constant that affects cohort identity."""
    return {
        "contract": CAP_QUALIFICATION_CONTRACT,
        "candidate_caps": list(CANDIDATE_COMPLETION_CAPS),
        "bloom_levels": list(BLOOM_LEVELS),
        "requirement_buckets": list(REQUIREMENT_BUCKETS),
        "task_families": list(TASK_FAMILIES),
        "task_family_mapping": dict(sorted(_TASK_FAMILY.items())),
        "source_cardinalities": list(SOURCE_CARDINALITIES),
        "pair_kinds": list(PAIR_KINDS),
        "objective_requirement_contract": (
            objective_requirement_contract_components()
        ),
        "micro_contract_fingerprint": micro_contract_fingerprint(),
    }


def qualification_contract_fingerprint() -> str:
    return content_sha256(qualification_contract_components())


def _stable(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_stable(value).encode("utf-8")).hexdigest()


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "").strip()


def _requirement_bucket(count: int) -> str:
    if count < 1:
        raise CapQualificationError("objective has no execution requirements")
    if count <= 2:
        return "1-2"
    if count <= 4:
        return "3-4"
    if count <= 6:
        return "5-6"
    return "7+"


def _row_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        key: row[key] for key in (
            "contract", "row_id", "kind", "chunk_id", "chunk_sha256",
            "variant", "repetition", "cohort_index", "order_index",
            "focus_objective", "objective_contract_sha256",
            "requirement_count", "stratum", "evidence_window_sha256",
        )
    }
    if row.get("qualification_roles"):
        projection["qualification_roles"] = list(row["qualification_roles"])
    return projection


def _candidate(
    chunk: Mapping[str, Any],
    *,
    kind: str,
    objectives: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    focused = apply_runtime_focus(chunk, objectives)
    eligibility = pair_eligibility(focused, kind=kind)
    if not eligibility.eligible:
        raise CapQualificationError(f"{kind}:{eligibility.reason}")
    focus = objective_card(focused["synthesis_focus_objective"])
    requirements = derive_objective_requirements(focus)
    task_type = str(requirements.get("cognitive_task_type") or "")
    task_family = _TASK_FAMILY.get(task_type)
    if task_family is None:
        raise CapQualificationError(f"task_family:{task_type or 'unresolved'}")
    bloom = str(focus.get("bloom_level") or "").strip().lower()
    if bloom not in BLOOM_LEVELS:
        raise CapQualificationError(f"bloom:{bloom or 'unresolved'}")
    try:
        window = build_evidence_window(focused, focus)
    except ValueError as exc:
        raise CapQualificationError(f"evidence_window:{exc}") from exc
    if kind == "preference":
        preference = micro_preference_eligibility(focused, focus=focus)
        if not preference["eligible"]:
            raise CapQualificationError(
                f"preference:{preference['reason']}"
            )
    block_ids = {
        str(block.get("block_id") or "").strip()
        for block in window.get("blocks") or []
        if str(block.get("block_id") or "").strip()
    }
    if not block_ids:
        raise CapQualificationError("evidence_window:no_source_blocks")
    source_cardinality = "single" if len(block_ids) == 1 else "multi"
    requirement_count = len(requirements["requirements"])
    stratum = [
        bloom,
        _requirement_bucket(requirement_count),
        task_family,
        source_cardinality,
    ]
    chunk_id = _chunk_id(focused)
    chunk_sha256 = _sha(focused)
    identity = {
        "kind": kind,
        "variant": "D_production_contract",
        "repetition": 0,
        "chunk_id": chunk_id,
        "chunk_sha256": chunk_sha256,
        "objective_contract_sha256": requirements[
            "objective_contract_sha256"
        ],
        "stratum": stratum,
    }
    return {
        "contract": CAP_QUALIFICATION_CONTRACT,
        "row_id": f"cap-row-{_sha(identity)[:24]}",
        "kind": kind,
        "variant": "D_production_contract",
        "repetition": 0,
        "cohort_index": 0,
        "order_index": 0,
        "chunk_id": chunk_id,
        "chunk_sha256": chunk_sha256,
        "focus_objective": focus,
        "objective_contract_sha256": requirements[
            "objective_contract_sha256"
        ],
        "requirement_count": requirement_count,
        "stratum": stratum,
        "evidence_window_sha256": _sha(window),
        "_qualification_workload": {
            "requirement_count": requirement_count,
            "objective_chars": len(_stable(focus)),
            "evidence_chars": sum(
                len(str(block.get("text") or ""))
                for block in window.get("blocks") or []
            ),
            "source_blocks": len(block_ids),
            "chunk_chars": len(str(focused.get("text") or "")),
            "result_required": bool(requirements["result_required"]),
        },
        "_chunk": focused,
    }


def required_strata() -> tuple[tuple[str, str, str, str, str], ...]:
    """Return the complete, frozen qualification cross-product."""
    return tuple(product(
        PAIR_KINDS,
        BLOOM_LEVELS,
        REQUIREMENT_BUCKETS,
        TASK_FAMILIES,
        SOURCE_CARDINALITIES,
    ))


def _eligible_groups(
    chunks: Iterable[Mapping[str, Any]],
    *,
    objectives: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str, str, str, str], list[dict[str, Any]]],
    Counter[str],
    int,
]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    exclusions: Counter[str] = Counter()
    input_count = 0
    seen_ids: set[str] = set()
    for source in chunks:
        input_count += 1
        chunk_id = _chunk_id(source)
        if not chunk_id or chunk_id in seen_ids:
            raise CapQualificationError(
                "input chunks require unique nonempty identities"
            )
        seen_ids.add(chunk_id)
        for kind in PAIR_KINDS:
            try:
                row = _candidate(source, kind=kind, objectives=objectives)
            except (CapQualificationError, ValueError) as exc:
                exclusions[str(exc)] += 1
                continue
            groups[(kind, *row["stratum"])].append(row)
    return groups, exclusions, input_count


def build_frozen_manifest(
    chunks: Iterable[Mapping[str, Any]],
    *,
    objectives: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one exact-eligible row for every required stratum.

    Selection is content-addressed and input-order independent.  Missing
    strata fail closed; a nearby row is never substituted.
    """
    groups, exclusions, input_count = _eligible_groups(
        chunks, objectives=objectives,
    )

    missing = [list(key) for key in required_strata() if not groups.get(key)]
    if missing:
        raise CapQualificationError(
            "qualification corpus lacks required strata: " + _stable(missing)
        )
    selected = []
    for key in required_strata():
        candidates = sorted(
            groups[key],
            key=lambda row: (
                row["chunk_sha256"], row["chunk_id"], row["row_id"],
            ),
        )
        selected.append(candidates[0])
    for index, row in enumerate(selected):
        row["cohort_index"] = index
        row["order_index"] = index
    public_rows = [_row_projection(row) for row in selected]
    manifest_sha256 = content_sha256(public_rows)
    audit = {
        "contract": CAP_QUALIFICATION_CONTRACT,
        "contract_fingerprint": qualification_contract_fingerprint(),
        "input_chunks": input_count,
        "required_strata": len(required_strata()),
        "selected_rows": len(public_rows),
        "candidate_counts": {
            "|".join(key): len(groups[key]) for key in required_strata()
        },
        "exclusions": dict(sorted(exclusions.items())),
        "manifest_sha256": manifest_sha256,
        "ordered_row_ids_sha256": content_sha256(
            [row["row_id"] for row in public_rows]
        ),
    }
    return selected, audit


def _coverage_tokens(key: tuple[str, ...]) -> frozenset[str]:
    """Cover every observed axis value and pairwise combination."""
    names = ("kind", "bloom", "requirements", "task", "source")
    tokens = {
        f"{names[index]}={value}" for index, value in enumerate(key)
    }
    tokens.update(
        f"{names[left]}={key[left]}&{names[right]}={key[right]}"
        for left in range(len(key))
        for right in range(left + 1, len(key))
    )
    return frozenset(tokens)


def _workload_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    workload = row.get("_qualification_workload") or {}
    score = (
        int(workload.get("evidence_chars") or 0)
        + int(workload.get("objective_chars") or 0)
        + 64 * int(workload.get("requirement_count") or 0)
    )
    return score, str(row["chunk_sha256"]), str(row["row_id"])


def qualification_runner_draft_identity(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the exact canonical micro draft identity; infer nothing."""
    return {
        "chunk_id": row["chunk_id"],
        "chunk_sha256": row["chunk_sha256"],
        "kind": row["kind"],
        "variant": row["variant"],
        "repetition": row["repetition"],
        "draft": {
            "provider": "local",
            "source_chunk_id": row["chunk_id"],
        },
    }


def qualification_runner_preflight(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail before provider construction or output creation on identity drift."""
    if not 16 <= len(rows) <= 32:
        raise CapQualificationError("runner row count must be within 16..32")
    identities = []
    expected_order = sorted(
        rows,
        key=lambda row: (
            PAIR_KINDS.index(str(row.get("kind") or "")),
            *[
                list(axis).index(value)
                for axis, value in zip(
                    (
                        BLOOM_LEVELS, REQUIREMENT_BUCKETS, TASK_FAMILIES,
                        SOURCE_CARDINALITIES,
                    ),
                    row.get("stratum") or (),
                )
            ],
            _workload_key(row),
        ),
    )
    if [row.get("row_id") for row in rows] != [
        row.get("row_id") for row in expected_order
    ]:
        raise CapQualificationError("runner rows are not in canonical order")
    for index, row in enumerate(rows):
        if (
            row.get("variant") != "D_production_contract"
            or isinstance(row.get("repetition"), bool)
            or row.get("repetition") != 0
            or isinstance(row.get("order_index"), bool)
            or row.get("order_index") != 0
            or isinstance(row.get("cohort_index"), bool)
            or row.get("cohort_index") != index
        ):
            raise CapQualificationError(
                "runner variant/repetition/order/cohort identity is invalid"
            )
        identity = (
            str(row.get("chunk_id") or ""),
            str(row.get("kind") or ""),
            str(row.get("variant") or ""),
            row.get("repetition"),
        )
        if not identity[0] or identity[1] not in PAIR_KINDS:
            raise CapQualificationError("runner identity is incomplete")
        identities.append(identity)
        # Required-key access is deliberate: there is no default inference.
        qualification_runner_draft_identity(row)
    if len(set(identities)) != len(identities):
        raise CapQualificationError("runner identity is duplicated")
    projection = {
        "contract": CAP_QUALIFICATION_CONTRACT,
        "variant": "D_production_contract",
        "repetition": 0,
        "order_index": 0,
        "cell_id": f"qualification-c{len(rows)}",
        "client_concurrency": len(rows),
        "ordered_draft_identity_sha256": [
            content_sha256(qualification_runner_draft_identity(row))
            for row in rows
        ],
    }
    return {
        **projection,
        "runner_preflight_sha256": content_sha256(projection),
    }


def build_empirical_set_cover_manifest(
    chunks: Iterable[Mapping[str, Any]],
    *,
    objectives: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the smallest deterministic greedy cover of observed axes/pairs.

    This is deliberately distinct from :func:`build_frozen_manifest`: absent
    Cartesian strata are recorded, not fabricated.  The selected combined
    strata are the only empirically qualified production strata.
    """
    groups, exclusions, input_count = _eligible_groups(
        chunks, objectives=objectives,
    )
    if not groups:
        raise CapQualificationError("qualification corpus has no eligible rows")
    candidates = []
    for key, values in groups.items():
        selected = min(values, key=_workload_key)
        candidates.append((key, selected, _coverage_tokens(key)))
    candidates.sort(key=lambda item: (
        item[1]["chunk_sha256"], item[1]["chunk_id"], item[1]["row_id"],
    ))
    all_rows = [
        row for values in groups.values() for row in values
    ]
    stress_metrics = {
        "max_requirement_count": "requirement_count",
        "max_objective_chars": "objective_chars",
        "max_evidence_chars": "evidence_chars",
        "max_source_blocks": "source_blocks",
        "longest_real_source": "chunk_chars",
    }
    stress_rows: dict[str, dict[str, Any]] = {}
    for role, metric in stress_metrics.items():
        stress_rows[role] = min(
            all_rows,
            key=lambda row: (
                -int((row.get("_qualification_workload") or {}).get(metric) or 0),
                *_workload_key(row),
            ),
        )
    result_candidates = [
        row for row in all_rows
        if bool((row.get("_qualification_workload") or {}).get(
            "result_required"
        ))
    ]
    if result_candidates:
        stress_rows["result_required"] = min(
            result_candidates, key=_workload_key,
        )

    by_id = {str(row["row_id"]): row for row in all_rows}
    roles_by_id: dict[str, list[str]] = defaultdict(list)
    for role, row in stress_rows.items():
        roles_by_id[str(row["row_id"])].append(role)
    selected_ids = set(roles_by_id)
    selected_rows: list[dict[str, Any]] = [
        by_id[row_id] for row_id in sorted(
            selected_ids, key=lambda value: _workload_key(by_id[value]),
        )
    ]
    uncovered = set().union(*(tokens for _key, _row, tokens in candidates))
    for row in selected_rows:
        uncovered.difference_update(_coverage_tokens(
            (row["kind"], *tuple(row["stratum"]))
        ))
    while uncovered:
        ranked = sorted(
            [item for item in candidates if item[1]["row_id"] not in selected_ids],
            key=lambda item: (
                -len(item[2] & uncovered),
                *_workload_key(item[1]),
            ),
        )
        if not ranked:
            raise CapQualificationError("empirical set cover made no progress")
        key, row, tokens = ranked[0]
        gain = tokens & uncovered
        if not gain:
            raise CapQualificationError("empirical set cover made no progress")
        selected_rows.append(row)
        selected_ids.add(str(row["row_id"]))
        uncovered.difference_update(gain)

    remaining = sorted(
        (row for row in all_rows if str(row["row_id"]) not in selected_ids),
        key=_workload_key,
    )
    for kind in PAIR_KINDS:
        while sum(row["kind"] == kind for row in selected_rows) < 6:
            match = next(
                (row for row in remaining if row["kind"] == kind), None,
            )
            if match is None:
                break
            selected_rows.append(match)
            selected_ids.add(str(match["row_id"]))
            remaining.remove(match)
    while len(selected_rows) < 16 and remaining:
        row = remaining.pop(0)
        selected_rows.append(row)
        selected_ids.add(str(row["row_id"]))

    status = "ready"
    status_reasons = []
    if len(selected_rows) < 16:
        status = "insufficient"
        status_reasons.append("fewer_than_16_exact_eligible_rows")
    for kind in PAIR_KINDS:
        if sum(row["kind"] == kind for row in selected_rows) < 6:
            status = "insufficient"
            status_reasons.append(f"fewer_than_6_{kind}_rows")
    if len(selected_rows) > 32:
        status = "budget_exceeded"
        status_reasons.append("more_than_32_rows_or_160_five_cap_units")

    for row in selected_rows:
        row["qualification_roles"] = sorted(
            roles_by_id.get(str(row["row_id"]), [])
        )
    selected_rows.sort(
        key=lambda row: (
            PAIR_KINDS.index(row["kind"]),
            *[list(axis).index(value) for axis, value in zip(
                (
                    BLOOM_LEVELS, REQUIREMENT_BUCKETS, TASK_FAMILIES,
                    SOURCE_CARDINALITIES,
                ),
                row["stratum"],
            )],
            _workload_key(row),
        )
    )
    for index, row in enumerate(selected_rows):
        row["cohort_index"] = index
        row["order_index"] = 0
    selected_keys = [
        (row["kind"], *tuple(row["stratum"])) for row in selected_rows
    ]
    public_rows = [_row_projection(row) for row in selected_rows]
    observed = set(groups)
    absent = [list(key) for key in required_strata() if key not in observed]
    audit = {
        "contract": CAP_QUALIFICATION_CONTRACT,
        "coverage_mode": "observed-axis-risk-pairwise-irreducible-greedy",
        "minimality": "irreducible_greedy",
        "preflight_status": status,
        "status_reasons": status_reasons,
        "contract_fingerprint": qualification_contract_fingerprint(),
        "input_chunks": input_count,
        "eligible_combined_strata": len(observed),
        "selected_rows": len(public_rows),
        "five_cap_units": 5 * len(public_rows),
        "selected_combined_strata": [list(key) for key in selected_keys],
        "mandatory_stress_rows": {
            role: row["row_id"] for role, row in sorted(stress_rows.items())
        },
        "selection_objective": {
            "priority": ["row_count", "workload", "row_identity"],
            "row_count": len(public_rows),
            "workload": sum(_workload_key(row)[0] for row in selected_rows),
            "ordered_row_ids": [row["row_id"] for row in selected_rows],
        },
        "absent_required_strata": absent,
        "absent_required_strata_sha256": content_sha256(absent),
        "observed_candidate_counts": {
            "|".join(key): len(groups[key]) for key in sorted(groups)
        },
        "exclusions": dict(sorted(exclusions.items())),
        "manifest_sha256": content_sha256(public_rows),
        "ordered_row_ids_sha256": content_sha256(
            [row["row_id"] for row in public_rows]
        ),
    }
    return selected_rows, audit


def require_qualified_production_strata(
    qualification_audit: Mapping[str, Any],
    production_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Require requalification when production introduces a new stratum."""
    qualified = {
        tuple(item)
        for item in qualification_audit.get("selected_combined_strata") or []
        if isinstance(item, list)
    }
    if not qualified:
        raise CapQualificationError("qualification audit has no strata")
    introduced = sorted({
        (str(row.get("kind") or ""), *tuple(row.get("stratum") or ()))
        for row in production_rows
    } - qualified)
    if introduced:
        raise CapQualificationError(
            "production introduces unqualified strata: " + _stable(introduced)
        )


def cap_manifests(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Bind the identical frozen row identities to every candidate cap."""
    projections = [_row_projection(row) for row in rows]
    runner_preflight = qualification_runner_preflight(rows)
    expected = list(required_strata())
    observed = [
        (row["kind"], *tuple(row["stratum"])) for row in projections
    ]
    expected_index = {key: index for index, key in enumerate(expected)}
    if (
        not observed
        or len({row["row_id"] for row in projections}) != len(projections)
        or any(key not in expected_index for key in observed)
        or observed != sorted(observed, key=expected_index.__getitem__)
    ):
        raise CapQualificationError(
            "rows are not a row-unique canonical required-strata sequence"
        )
    base_sha256 = content_sha256(projections)
    runner_identity = {
        **runner_preflight,
        "ordered_row_ids": [row["row_id"] for row in projections],
        "ordered_chunk_sha256": [
            row["chunk_sha256"] for row in projections
        ],
    }
    return {
        cap: {
            "contract": CAP_QUALIFICATION_CONTRACT,
            "contract_fingerprint": qualification_contract_fingerprint(),
            "completion_cap": cap,
            "base_manifest_sha256": base_sha256,
            "runner_identity": runner_identity,
            "runner_identity_sha256": content_sha256(runner_identity),
            "ordered_row_ids": [row["row_id"] for row in projections],
            "cap_manifest_sha256": content_sha256({
                "completion_cap": cap,
                "base_manifest_sha256": base_sha256,
                "runner_identity_sha256": content_sha256(runner_identity),
                "ordered_row_ids": [row["row_id"] for row in projections],
            }),
        }
        for cap in CANDIDATE_COMPLETION_CAPS
    }


def _nearest_rank(values: Sequence[float | int], percentile: float) -> float:
    if not values:
        raise CapQualificationError("percentile requires observations")
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(percentile * len(ordered)) - 1)])


def _distribution(values: Sequence[float | int]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "maximum": max(values),
    }


def adjudicate_caps(
    rows: Sequence[Mapping[str, Any]],
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize five identical cap cells and select the smallest qualifier."""
    manifests = cap_manifests(rows)
    row_by_id = {str(row["row_id"]): row for row in rows}
    by_cap: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for observation in observations:
        cap = observation.get("completion_cap")
        row_id = str(observation.get("row_id") or "")
        if cap not in CANDIDATE_COMPLETION_CAPS or row_id not in row_by_id:
            raise CapQualificationError("observation has unknown cap or row")
        identity = (int(cap), row_id)
        if identity in seen:
            raise CapQualificationError("duplicate cap/row observation")
        seen.add(identity)
        if observation.get("base_manifest_sha256") != manifests[int(cap)][
            "base_manifest_sha256"
        ]:
            raise CapQualificationError("observation manifest binding mismatch")
        if observation.get("cap_manifest_sha256") != manifests[int(cap)][
            "cap_manifest_sha256"
        ]:
            raise CapQualificationError("observation cap binding mismatch")
        by_cap[int(cap)].append(observation)
    expected = {
        (cap, row_id)
        for cap in CANDIDATE_COMPLETION_CAPS
        for row_id in row_by_id
    }
    if seen != expected:
        raise CapQualificationError("five-cap observation matrix is incomplete")

    summaries: dict[int, dict[str, Any]] = {}
    control_unsupported = None
    control_contradiction = None
    selected = None
    for cap in CANDIDATE_COMPLETION_CAPS:
        cell = by_cap[cap]
        counts = Counter(str(item.get("status") or "") for item in cell)
        accepted = [item for item in cell if item.get("status") == "accepted"]
        reason_codes = Counter(
            str(item.get("reason_code") or "none") for item in cell
        )
        finish_reasons = Counter(
            str(item.get("finish_reason") or "none") for item in cell
        )
        if not accepted:
            summaries[cap] = {
                "qualifies": False,
                "failure_reasons": ["no_accepted_rows"],
                "counts": dict(sorted(counts.items())),
                "reason_codes": dict(sorted(reason_codes.items())),
                "finish_reasons": dict(sorted(finish_reasons.items())),
            }
            continue
        for item in accepted:
            for field in (
                "completion_chars", "completion_tokens", "latency_seconds",
                "output_tokens", "throughput_tokens_per_second",
                "requirement_pass_rate", "result_coverage_rate",
                "claim_support_rate", "contradiction_rate",
                "unsupported_added_claim_rate",
                "reconciliation_rate",
            ):
                if isinstance(item.get(field), bool) or not isinstance(
                    item.get(field), (int, float)
                ):
                    raise CapQualificationError(
                        f"accepted observation lacks numeric {field}"
                    )
        for item in cell:
            if not isinstance(item.get("critical_error"), bool):
                raise CapQualificationError(
                    "observation lacks boolean critical_error"
                )
        unsupported = sum(
            float(item["unsupported_added_claim_rate"]) for item in cell
        ) / len(cell)
        contradiction = sum(
            float(item["contradiction_rate"]) for item in cell
        ) / len(cell)
        if cap == CANDIDATE_COMPLETION_CAPS[0]:
            control_unsupported = unsupported
            control_contradiction = contradiction
        failures: list[str] = []
        if cap != CANDIDATE_COMPLETION_CAPS[0] and (
            control_unsupported is None or control_contradiction is None
        ):
            failures.append("control_unavailable")
        if any(item.get("finish_reason") == "length" for item in cell):
            failures.append("length_finish")
        if any(bool(item.get("truncated")) for item in cell):
            failures.append("truncation")
        critical_count = sum(bool(item["critical_error"]) for item in cell)
        if critical_count:
            failures.append("critical_error")
        if any(
            float(item["requirement_pass_rate"]) != 1.0
            or float(item["result_coverage_rate"]) != 1.0
            or float(item["reconciliation_rate"]) != 1.0
            for item in accepted
        ):
            failures.append("accepted_execution_incomplete")
        if (
            control_unsupported is not None
            and unsupported > control_unsupported
        ):
            failures.append("unsupported_claim_regression")
        if (
            control_contradiction is not None
            and contradiction > control_contradiction
        ):
            failures.append("contradiction_regression")
        chars = [int(item["completion_chars"]) for item in accepted]
        grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for item in accepted:
            grouped[tuple(row_by_id[str(item["row_id"])]["stratum"])].append(
                int(item["completion_chars"])
            )
        accepted_by_id = {
            str(item["row_id"]): item for item in accepted
        }
        required_tokens = set().union(*(
            _coverage_tokens((row["kind"], *tuple(row["stratum"])))
            for row in rows
        ))
        accepted_tokens = set().union(*(
            _coverage_tokens((row["kind"], *tuple(row["stratum"])))
            for row in rows if str(row["row_id"]) in accepted_by_id
        )) if accepted else set()
        if accepted_tokens != required_tokens:
            failures.append("accepted_risk_coverage_incomplete")
        if {row_by_id[row_id]["kind"] for row_id in accepted_by_id} != set(
            PAIR_KINDS
        ):
            failures.append("accepted_kind_coverage_incomplete")
        if any(value >= 0.90 * cap for value in chars):
            failures.append("accepted_utilization_not_below_90_percent")
        if _nearest_rank(chars, 0.95) > 0.80 * cap:
            failures.append("pooled_p95_above_80_percent")
        stress_ids = {
            str(row["row_id"]) for row in rows
            if row.get("qualification_roles")
        }
        stress_failures = [
            row_id for row_id in sorted(stress_ids)
            if row_id not in accepted_by_id
            or int(accepted_by_id[row_id]["completion_chars"]) >= 0.85 * cap
        ]
        if stress_failures:
            failures.append("stress_utilization_not_below_85_percent")
        summary = {
            "qualifies": not failures,
            "failure_reasons": failures,
            "counts": dict(sorted(counts.items())),
            "reason_codes": dict(sorted(reason_codes.items())),
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "completion_chars": _distribution(chars),
            "completion_cap_utilization": _distribution([
                value / cap for value in chars
            ]),
            "completion_tokens": _distribution([
                int(item["completion_tokens"]) for item in accepted
            ]),
            "at_or_above_90_percent": sum(
                value >= 0.90 * cap for value in chars
            ),
            "at_or_above_95_percent": sum(
                value >= 0.95 * cap for value in chars
            ),
            "at_or_above_100_percent": sum(value >= cap for value in chars),
            "unsupported_added_claim_rate": unsupported,
            "contradiction_rate": contradiction,
            "requirement_pass_rate": sum(
                float(item["requirement_pass_rate"]) for item in accepted
            ) / len(accepted),
            "result_coverage_rate": sum(
                float(item["result_coverage_rate"]) for item in accepted
            ) / len(accepted),
            "claim_support_rate": sum(
                float(item["claim_support_rate"]) for item in accepted
            ) / len(accepted),
            "reconciliation_rate": sum(
                float(item["reconciliation_rate"]) for item in accepted
            ) / len(accepted),
            "stress_headroom_failures": stress_failures,
            "critical_error_count": critical_count,
            "rule_of_three_zero_critical_upper_95": (
                3.0 / len(cell) if critical_count == 0 else None
            ),
            "latency_seconds": _distribution([
                float(item["latency_seconds"]) for item in accepted
            ]),
            "output_tokens": sum(
                int(item["output_tokens"]) for item in accepted
            ),
            "aggregate_throughput_tokens_per_second": sum(
                float(item["throughput_tokens_per_second"])
                for item in accepted
            ),
        }
        summaries[cap] = summary
        if selected is None and summary["qualifies"]:
            selected = cap
    return {
        "contract": CAP_QUALIFICATION_CONTRACT,
        "contract_fingerprint": qualification_contract_fingerprint(),
        "base_manifest_sha256": next(iter(manifests.values()))[
            "base_manifest_sha256"
        ],
        "caps": summaries,
        "selected_cap": selected,
        "decision": "selected" if selected is not None else "hold",
        "selection_sha256": content_sha256({
            "base_manifest_sha256": next(iter(manifests.values()))[
                "base_manifest_sha256"
            ],
            "caps": summaries,
            "selected_cap": selected,
        }),
    }
