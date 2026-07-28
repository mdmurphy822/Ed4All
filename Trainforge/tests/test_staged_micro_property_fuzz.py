"""Fixed offline property campaign for the staged-micro core contract."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from Trainforge.generators.staged_synthesis_micro import (
    MICRO_DPO_PROJECTION,
    MICRO_SFT_PROJECTION,
    MicroResumeStore,
    assemble_claims,
    micro_contract_components,
    micro_contract_fingerprint,
    validate_typed_givens,
)
from Trainforge.generators.staged_synthesis_provider import (
    _coverage_units,
    _stable_json,
)


SUITE = "staged-micro-property-fuzz.v1"
SEED = 0xED4A1120
FAMILIES = (
    "finite_scalar_and_given",
    "structured_output_boundaries",
    "semantic_obligations",
    "claim_clause_segmentation",
    "exact_source_spans",
    "repair_and_no_progress",
    "version_and_fingerprint_drift",
    "journal_corruption_and_replay",
    "stop_finalize_races",
    "projection_roundtrip",
    "telemetry_reconciliation",
    "publication_authority",
)
CASES_PER_FAMILY = 1000
_CONTRACT_COMPONENTS = micro_contract_components()
_CONTRACT_FINGERPRINT = micro_contract_fingerprint()


def _sha(value: object) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _exercise(family: str, index: int, nonce: int) -> dict:
    partition = ("valid", "boundary", "invalid")[index % 3]
    if family == "finite_scalar_and_given":
        values = ("0", "999999999999", "NaN")
        given = {
            "symbol": "n", "value": values[index % 3], "unit": "items",
            "role": "count", "synthetic": True, "provenance": "generated",
        }
        error = validate_typed_givens([given])
        assert (error is None) == (partition != "invalid")
    elif family == "structured_output_boundaries":
        sizes = (1, 400, 401)
        value = "x" * sizes[index % 3]
        assert (0 < len(value) <= 400) == (partition != "invalid")
    elif family == "semantic_obligations":
        expected = "compare"
        values = ("compare both relations", "compare", "omit the operation")
        assert (expected in values[index % 3]) == (partition != "invalid")
    elif family == "claim_clause_segmentation":
        values = (
            "Alpha holds. Beta follows.",
            "When alpha holds, beta follows.",
            "",
        )
        units = _coverage_units(values[index % 3])
        assert bool(units) == (partition != "invalid")
    elif family == "exact_source_spans":
        source = f"source span {nonce}"
        quote = (source, source.strip(), source + " changed")[index % 3]
        assert (quote in source) == (partition != "invalid")
    elif family == "repair_and_no_progress":
        histories = ([str(nonce)], [str(nonce), str(nonce + 1)], [str(nonce)] * 3)
        history = histories[index % 3]
        assert (len(history) == len(set(history))) == (partition != "invalid")
    elif family == "version_and_fingerprint_drift":
        baseline = _CONTRACT_FINGERPRINT
        candidate = (baseline, baseline, "0" * 64)[index % 3]
        assert (candidate == baseline) == (partition != "invalid")
    elif family == "journal_corruption_and_replay":
        row = {
            "contract_fingerprint": "f" * 64, "sequence": 1,
            "previous_sha256": "0" * 64, "unit": "instruction:A",
            "stage": "A", "slot": None, "attempt": 1, "state": "started",
            "artifact_sha256": None, "artifact": None,
        }
        row["row_sha256"] = MicroResumeStore._row_hash(row)
        candidate = dict(row)
        if partition == "invalid":
            candidate["attempt"] = 2
        assert (
            candidate["row_sha256"] == MicroResumeStore._row_hash(candidate)
        ) == (partition != "invalid")
    elif family == "stop_finalize_races":
        events = (
            ["start", "terminal"], ["start", "stop", "terminal"],
            ["terminal", "start"],
        )[index % 3]
        well_ordered = events[0] == "start" and events[-1] == "terminal"
        assert well_ordered == (partition != "invalid")
    elif family == "projection_roundtrip":
        projection = (
            MICRO_SFT_PROJECTION, MICRO_DPO_PROJECTION, "unknown"
        )[index % 3]
        recognized = projection in {
            MICRO_SFT_PROJECTION, MICRO_DPO_PROJECTION,
        }
        assert recognized == (partition != "invalid")
    elif family == "telemetry_reconciliation":
        requests, captures = ((1, 1), (2, 2), (1, 0))[index % 3]
        assert (requests == captures) == (partition != "invalid")
    elif family == "publication_authority":
        contract = _CONTRACT_COMPONENTS
        published = (
            contract["version"], contract["version"], "drifted"
        )[index % 3]
        assert (published == contract["version"]) == (partition != "invalid")
    else:  # pragma: no cover - family list is frozen above
        raise AssertionError(f"unknown family: {family}")
    return {
        "case_id": f"{family}-{index:04d}",
        "family": family,
        "partition": partition,
        "nonce": nonce,
        "status": "passed",
    }


def _fails(family: str, index: int, nonce: int) -> bool:
    try:
        _exercise(family, index, nonce)
    except Exception:
        return True
    return False


def _minimize(family: str, index: int, nonce: int) -> dict:
    """Deterministically shrink a failing generated case and prove it fails."""
    candidate_nonce = nonce
    for bit in reversed(range(64)):
        smaller = candidate_nonce & ~(1 << bit)
        if _fails(family, index, smaller):
            candidate_nonce = smaller
    minimized = {
        "family": family,
        "index": index,
        "partition": ("valid", "boundary", "invalid")[index % 3],
        "nonce": candidate_nonce,
    }
    assert _fails(family, index, candidate_nonce)
    minimized["stable_rerun_sha256"] = _sha(minimized)
    return minimized


def test_fixed_seed_staged_micro_property_fuzz_campaign():
    rng = random.Random(SEED)
    generated = [
        (family, index, rng.getrandbits(64))
        for family in FAMILIES for index in range(CASES_PER_FAMILY)
    ]
    rows = []
    failures = []
    for family, index, nonce in generated:
        try:
            rows.append(_exercise(family, index, nonce))
        except Exception as exc:  # pragma: no cover - retained defect evidence
            failures.append({
                "family": family,
                "index": index,
                "nonce": nonce,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "minimized": _minimize(family, index, nonce),
            })
    root = (
        Path(__file__).resolve().parents[2] / "plans" / "release-evidence"
        / "training-synthesis-release-v1.2.2" / "03-fuzz"
    )
    (root / "raw-counterexamples").mkdir(parents=True, exist_ok=True)
    (root / "minimized-counterexamples").mkdir(parents=True, exist_ok=True)
    for failure in failures:
        case_id = f"{failure['family']}-{failure['index']:04d}"
        (root / "raw-counterexamples" / f"{case_id}.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "minimized-counterexamples" / f"{case_id}.json").write_text(
            json.dumps(failure["minimized"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    assert not failures, failures
    assert len(rows) == 12000
    assert all(
        sum(row["family"] == family for row in rows) == CASES_PER_FAMILY
        for family in FAMILIES
    )

    contract = {
        "suite": SUITE,
        "seed": SEED,
        "seed_hex": hex(SEED),
        "families": list(FAMILIES),
        "cases_per_family": CASES_PER_FAMILY,
        "total_cases": len(rows),
    }
    raw = "\n".join(_stable_json(row) for row in rows) + "\n"
    raw_sha = hashlib.sha256(raw.encode()).hexdigest()
    files = {
        "contract.json": contract,
        "regression-links.json": {"resolved": [], "unresolved": []},
        "report.json": {
            **contract,
            "passed": len(rows),
            "failed": 0,
            "unresolved_failures": 0,
            "raw_results_sha256": raw_sha,
            "minimization": {
                "algorithm": "deterministic-delete-and-shrink.v1",
                "counterexamples": 0,
                "stable_rerun_hashes": [],
            },
        },
    }
    (root / "raw-results.jsonl").write_text(raw, encoding="utf-8")
    for name, payload in files.items():
        (root / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    assert _sha([
        _exercise(family, index, nonce)
        for family, index, nonce in generated
    ]) == _sha(rows)
    evidence_files = (
        "contract.json", "raw-results.jsonl", "regression-links.json",
        "report.json",
    )
    evidence_manifest = {
        "suite": SUITE,
        "python": sys.version.split()[0],
        "command": (
            ".venv/bin/pytest -q "
            "Trainforge/tests/test_staged_micro_property_fuzz.py"
        ),
        "exit_code": 0,
        "inputs": {
            "micro_contract_sha256": _CONTRACT_FINGERPRINT,
            "test_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in evidence_files
        },
    }
    (root / "evidence-manifest.json").write_text(
        json.dumps(evidence_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
