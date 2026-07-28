"""Rejected pairs must carry their NLI evidence into the operator artifact.

The per-pair resume checkpoint already stored a ``rejection_evidence`` blob,
but the projection into ``synthesis_dispositions.jsonl`` dropped it and then
unlinked the checkpoint on a clean exit. The consequence was measured: an audit
of 150 claim-support rejections could hand-adjudicate 14, because the
per-sentence entailment / contradiction scores that decided each verdict no
longer existed anywhere on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.synthesize_training import (  # noqa: E402
    _DISPOSITION_PROJECTED_FIELDS,
    _append_synthesis_pairs_checkpoint,
    _load_synthesis_pairs_checkpoint,
    _project_terminal_dispositions,
    _VERDICT_POLICY_FILES,
)
from lib.validators.pair.claim_support import (  # noqa: E402
    summarize_claim_support_rejection,
)


_EVIDENCE = {
    "stage": "claim_support",
    "rejection_reason": "unsupported_claim",
    "total_claims": 4,
    "failing_claims": [
        {
            "sentence": "An assertion the source does not make.",
            "entailment": 0.11,
            "contradiction": 0.07,
            "outcome": "unsupported",
        },
    ],
}


def test_projection_carries_rejection_evidence_through() -> None:
    rows = _project_terminal_dispositions([
        {
            "schema_version": 2,
            "chunk_id": "chunk-0001",
            "kind": "instruction",
            "variant_index": 0,
            "provider": "local",
            "seed": 11,
            "disposition": "rejected",
            "reason": "claim_support:unsupported_claim",
            "contract_fingerprint": "fp-1",
            "rejection_evidence": _EVIDENCE,
        },
    ])

    assert len(rows) == 1
    assert rows[0]["rejection_evidence"] == _EVIDENCE
    # The scores an adjudicator needs are actually present.
    failing = rows[0]["rejection_evidence"]["failing_claims"][0]
    assert failing["sentence"]
    assert "entailment" in failing and "contradiction" in failing


def test_a_row_without_evidence_is_unchanged() -> None:
    """Omitted, not nulled — existing readers see the same shape as before."""
    rows = _project_terminal_dispositions([
        {
            "schema_version": 2,
            "chunk_id": "chunk-0002",
            "kind": "preference",
            "variant_index": 0,
            "provider": "local",
            "seed": 12,
            "disposition": "ineligible",
            "reason": "eval_holdout_reserved",
            "contract_fingerprint": "fp-1",
        },
    ])
    assert "rejection_evidence" not in rows[0]
    assert set(rows[0]) == set(_DISPOSITION_PROJECTED_FIELDS)


def test_accepted_rows_are_still_excluded() -> None:
    assert _project_terminal_dispositions([
        {"disposition": "accepted", "chunk_id": "c", "pair": {"prompt": "p"}},
    ]) == []


def test_the_pair_body_is_never_projected() -> None:
    """The dispositions file records verdicts, not a second copy of the corpus."""
    rows = _project_terminal_dispositions([
        {
            "disposition": "rejected",
            "chunk_id": "chunk-0003",
            "reason": "claim_support:contradicted_claim",
            "pair": {"prompt": "SHOULD NOT APPEAR", "completion": "NOR THIS"},
            "rejection_evidence": _EVIDENCE,
        },
    ])
    assert "pair" not in rows[0]
    assert "SHOULD NOT APPEAR" not in json.dumps(rows[0])


def test_evidence_survives_a_checkpoint_write_read_project_round_trip(
    tmp_path: Path,
) -> None:
    """End-to-end on the real path: append -> load -> project."""
    checkpoint = tmp_path / "pairs.jsonl"
    evidence = summarize_claim_support_rejection(
        {
            "per_claim_support": [
                {
                    "sentence": "A claim the chunk does not support.",
                    "entailment": 0.12,
                    "contradiction": 0.09,
                    "outcome": "unsupported",
                },
                {
                    "sentence": "A claim the chunk entails.",
                    "entailment": 0.94,
                    "contradiction": 0.01,
                    "outcome": "entailed",
                },
            ],
            "claim_support_rate": 0.5,
            "claim_contradicted_rate": 0.0,
        },
        rejection_reason="unsupported_claim",
    )
    assert evidence is not None

    with checkpoint.open("w", encoding="utf-8") as handle:
        _append_synthesis_pairs_checkpoint(
            handle,
            chunk_id="chunk-0004",
            kind="instruction",
            variant_index=0,
            pair=None,
            provider="local",
            seed=3,
            disposition="rejected",
            reason="claim_support:unsupported_claim",
            contract_fingerprint="fp-1",
            rejection_evidence=evidence,
        )

    rows = _project_terminal_dispositions(
        _load_synthesis_pairs_checkpoint(checkpoint).values()
    )
    assert len(rows) == 1
    assert rows[0]["rejection_evidence"]["failing_claims"] == (
        evidence["failing_claims"]
    )
    assert rows[0]["reason"] == "claim_support:unsupported_claim"
    # JSON-serializable, since this is what gets written to the operator file.
    json.loads(json.dumps(rows[0]))


def test_the_math_normalizer_is_recorded_as_verdict_policy() -> None:
    """It now decides NLI verdicts, so it belongs in the recorded digest."""
    assert "lib/semantik/math_fold.py" in _VERDICT_POLICY_FILES
