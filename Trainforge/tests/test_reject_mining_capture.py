"""Capture layer for reject-mined DPO negatives: flag, bands, reader, pool.

Pure unit tests. No seat, no model, no embedder, no filesystem — which is the
point: every decision this layer makes is a function of already-persisted
values, so the whole thing is testable without a GPU.

The failure this feature exists to avoid is measured, not hypothetical: an
audit of the archived cohorts found that most rejects are DEGENERATE (a fixed
pedagogical tail with a database key spliced into the topic slot, entailment
0.028), and a DPO negative drawn from that population teaches DISTRIBUTION
discrimination rather than QUALITY discrimination. The capture layer's job is
to make near-miss text AVAILABLE; the admission bands here are the first,
coarsest cut.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.synthesis.synthesis_reject_mining import (  # noqa: E402
    MAX_CAPTURED_COMPLETION_CHARS,
    MAX_CAPTURED_PROMPT_CHARS,
    MINE_MODE_OFF,
    MINE_MODE_ON,
    MINE_MODE_SHADOW,
    MINE_REJECTS_ENV,
    MIN_CAPTURED_COMPLETION_CHARS,
    MIN_CAPTURED_PROMPT_CHARS,
    RejectCandidate,
    RejectPool,
    build_capture_payload,
    is_mineable_reason,
    iter_mineable_records,
    resolve_mine_rejects_mode,
)

_PROMPT = (
    "Explain how the stated conservation rule constrains the "
    "outcome of the described interaction."
)
_COMPLETION = (
    "The rule fixes the total before and after the interaction, so any "
    "redistribution among the participants must leave that total unchanged. "
    "That constraint is what makes the outcome predictable from the initial "
    "state alone."
)


def _pair(**overrides):
    pair = {
        "prompt": _PROMPT,
        "completion": _COMPLETION,
        "chunk_id": "chunk-0001",
        "lo_refs": ["CO-01"],
        "requires_source_citation": True,
        "provider": "local",
        "generating_seat": "seat-alpha",
        "claim_support_rate": 0.67,
        "claim_contradicted_rate": 0.0,
        "per_claim_support": [
            {"sentence": "A.", "entailment": 0.81, "outcome": "entailed"},
            {"sentence": "B.", "entailment": 0.77, "outcome": "entailed"},
            {"sentence": "C.", "entailment": 0.42, "outcome": "unsupported"},
        ],
    }
    pair.update(overrides)
    return pair


def _record(**overrides):
    record = {
        "schema_version": "v1",
        "chunk_id": "chunk-0001",
        "kind": "instruction",
        "variant_index": 1,
        "provider": "local",
        "seed": 41,
        "disposition": "rejected",
        "reason": "claim_support:unsupported_claim",
        "contract_fingerprint": "fp-current",
        "pair": _pair(),
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------
# flag
# --------------------------------------------------------------------------

def test_master_flag_parses_with_fallback_to_off() -> None:
    for raw in ("", "0", "false", "no", "off", "banana", "  ", "On shadow"):
        assert resolve_mine_rejects_mode({MINE_REJECTS_ENV: raw}) == (
            MINE_MODE_OFF
        ), raw
    assert resolve_mine_rejects_mode({}) == MINE_MODE_OFF
    for raw in ("shadow", "SHADOW", " Shadow "):
        assert resolve_mine_rejects_mode({MINE_REJECTS_ENV: raw}) == (
            MINE_MODE_SHADOW
        ), raw
    for raw in ("1", "true", "TRUE", "yes", "on", " On "):
        assert resolve_mine_rejects_mode({MINE_REJECTS_ENV: raw}) == (
            MINE_MODE_ON
        ), raw


def test_stage_admission_predicate() -> None:
    assert is_mineable_reason("claim_support:unsupported_claim")
    for reason in (
        "promotion:source_free_generation",
        "promotion:placeholder_residue",
        "promotion:unanswerable_stem",
        "lo_refs:missing_pair_lo_refs",
        "objective_delivery:objective_verb_absent",
        "duplicate_prompt",
        "gate_failed",
        "",
        None,
    ):
        assert not is_mineable_reason(reason), reason


# --------------------------------------------------------------------------
# capture payload — the admission + length bands
# --------------------------------------------------------------------------

def test_capture_is_a_no_op_when_the_flag_is_off() -> None:
    """Default OFF: the checkpoint row keeps its historical ``pair=None``."""
    assert build_capture_payload(
        _pair(), reason="claim_support:unsupported_claim", mode=MINE_MODE_OFF,
    ) is None


def test_capture_admits_a_claim_support_reject() -> None:
    payload = build_capture_payload(
        _pair(), reason="claim_support:unsupported_claim", mode=MINE_MODE_ON,
    )
    assert payload is not None
    assert payload["prompt"] == _PROMPT
    assert payload["completion"] == _COMPLETION
    assert payload["per_claim_support"][2]["outcome"] == "unsupported"


def test_shadow_still_captures() -> None:
    """Shadow measures the funnel on real data; it cannot measure nothing."""
    assert build_capture_payload(
        _pair(),
        reason="claim_support:unsupported_claim",
        mode=MINE_MODE_SHADOW,
    ) is not None


def test_capture_refuses_every_non_claim_support_stage() -> None:
    for reason in (
        "promotion:source_free_generation",
        "lo_refs:missing_pair_lo_refs",
        "objective_delivery:objective_verb_absent",
        "duplicate_prompt",
    ):
        assert build_capture_payload(
            _pair(), reason=reason, mode=MINE_MODE_ON,
        ) is None, reason


def test_capture_refuses_a_preference_unit() -> None:
    assert build_capture_payload(
        _pair(),
        reason="claim_support:unsupported_claim",
        mode=MINE_MODE_ON,
        kind="preference",
    ) is None


def test_capture_enforces_the_preference_pair_length_bands() -> None:
    short_prompt = "x" * (MIN_CAPTURED_PROMPT_CHARS - 1)
    long_prompt = "x" * (MAX_CAPTURED_PROMPT_CHARS + 1)
    short_completion = "y" * (MIN_CAPTURED_COMPLETION_CHARS - 1)
    long_completion = "y" * (MAX_CAPTURED_COMPLETION_CHARS + 1)
    for pair in (
        _pair(prompt=short_prompt),
        _pair(prompt=long_prompt),
        _pair(completion=short_completion),
        _pair(completion=long_completion),
        _pair(prompt="   "),
        _pair(completion=None),
        _pair(completion=123),
    ):
        assert build_capture_payload(
            pair, reason="claim_support:unsupported_claim", mode=MINE_MODE_ON,
        ) is None
    # Both bounds are inclusive.
    assert build_capture_payload(
        _pair(
            prompt="p" * MIN_CAPTURED_PROMPT_CHARS,
            completion="c" * MAX_CAPTURED_COMPLETION_CHARS,
        ),
        reason="claim_support:unsupported_claim",
        mode=MINE_MODE_ON,
    ) is not None


def test_capture_strips_private_execution_sidecars() -> None:
    """Mirrors the accepted-pair write, which pops these before appending."""
    payload = build_capture_payload(
        _pair(
            _objective_execution_private_sidecar={"secret": "value"},
            _objective_execution_candidate={"draft": "value"},
        ),
        reason="claim_support:unsupported_claim",
        mode=MINE_MODE_ON,
    )
    assert payload is not None
    assert not any(key.startswith("_") for key in payload)


def test_capture_does_not_mutate_the_source_pair() -> None:
    pair = _pair(_objective_execution_private_sidecar={"secret": "value"})
    before = dict(pair)
    build_capture_payload(
        pair, reason="claim_support:unsupported_claim", mode=MINE_MODE_ON,
    )
    assert pair == before


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

def test_candidate_reads_identity_and_scores_off_one_row() -> None:
    candidate = RejectCandidate.from_checkpoint_record(_record())
    assert candidate is not None
    assert candidate.key == ("chunk-0001", "instruction", 1)
    assert candidate.chunk_id == "chunk-0001"
    assert candidate.variant_index == 1
    assert candidate.provider == "local"
    assert candidate.seed == 41
    assert candidate.contract_fingerprint == "fp-current"
    assert candidate.generating_seat == "seat-alpha"
    assert candidate.lo_refs == ("CO-01",)
    assert candidate.requires_source_citation is True
    assert candidate.claim_support_rate == 0.67
    assert candidate.claim_contradicted_rate == 0.0
    assert candidate.per_claim_support is not None
    assert len(candidate.per_claim_support) == 3
    assert candidate.completion == _COMPLETION
    assert candidate.from_resume_cache is False


def test_candidate_reader_rejects_unmineable_rows() -> None:
    unmineable = [
        _record(disposition="accepted"),
        _record(disposition="ineligible"),
        _record(kind="preference"),
        _record(reason="promotion:source_free_generation"),
        # Every legacy row written before capture existed.
        _record(pair=None),
        _record(pair={}),
        _record(pair={"prompt": _PROMPT}),
        _record(chunk_id=""),
        _record(variant_index="not-an-int"),
    ]
    for record in unmineable:
        assert RejectCandidate.from_checkpoint_record(record) is None, record
    assert iter_mineable_records(unmineable) == []


def test_candidate_reader_tolerates_a_degraded_nli_row() -> None:
    """The NLI degrade arm stamps ``per_claim_support=None``; still readable.

    Capture does not adjudicate near-missness — dropping a scoreless row here
    would hide it from the funnel the selection pass has to report on.
    """
    candidate = RejectCandidate.from_checkpoint_record(
        _record(pair=_pair(
            per_claim_support=None,
            claim_support_rate=None,
            claim_contradicted_rate=None,
            deps_missing=True,
        ))
    )
    assert candidate is not None
    assert candidate.per_claim_support is None
    assert candidate.claim_support_rate is None


# --------------------------------------------------------------------------
# pool — identity binding, dedup, determinism, thread safety
# --------------------------------------------------------------------------

def _pool_with(*records) -> RejectPool:
    pool = RejectPool()
    pool.seed_from_checkpoint_cache({
        (
            str(record["chunk_id"]),
            str(record["kind"]),
            int(record["variant_index"]),
        ): record
        for record in records
    })
    return pool


def test_pool_seeds_only_mineable_rows_from_the_resume_cache() -> None:
    pool = _pool_with(
        _record(),
        _record(chunk_id="chunk-0002", variant_index=0, disposition="accepted"),
        _record(chunk_id="chunk-0003", variant_index=0, pair=None),
    )
    assert len(pool) == 1
    counters = pool.counters()
    assert counters["resume_cache_rows_scanned"] == 3
    assert counters["resume_cache_candidates"] == 1
    assert counters["pool_size"] == 1
    assert pool.candidates()[0].from_resume_cache is True


def test_pool_seeding_is_a_no_op_for_an_empty_or_missing_cache() -> None:
    pool = RejectPool()
    assert pool.seed_from_checkpoint_cache(None) == 0
    assert pool.seed_from_checkpoint_cache({}) == 0
    assert pool.candidates() == []


def test_pool_union_is_keyed_and_never_double_counts() -> None:
    """Resume completeness: prior-run rows + this-run rows, no duplicates."""
    pool = _pool_with(
        _record(chunk_id="chunk-0001", variant_index=1),
        _record(chunk_id="chunk-0002", variant_index=1),
    )
    for chunk_id in ("chunk-0003", "chunk-0004"):
        pool.record_rejection(
            chunk_id=chunk_id,
            kind="instruction",
            variant_index=1,
            pair=build_capture_payload(
                _pair(chunk_id=chunk_id),
                reason="claim_support:unsupported_claim",
                mode=MINE_MODE_ON,
            ),
            provider="local",
            seed=7,
            reason="claim_support:unsupported_claim",
            contract_fingerprint="fp-current",
        )
    assert len(pool) == 4
    assert [c.chunk_id for c in pool.candidates()] == [
        "chunk-0001", "chunk-0002", "chunk-0003", "chunk-0004",
    ]
    assert pool.counters()["captured_this_run"] == 2


def test_live_capture_supersedes_a_stale_resume_cache_row() -> None:
    """A regenerated unit's completion wins over the stale cached one."""
    pool = _pool_with(_record(contract_fingerprint="fp-stale"))
    fresh = build_capture_payload(
        _pair(completion=_COMPLETION + " Regenerated under a new contract."),
        reason="claim_support:unsupported_claim",
        mode=MINE_MODE_ON,
    )
    pool.record_rejection(
        chunk_id="chunk-0001",
        kind="instruction",
        variant_index=1,
        pair=fresh,
        provider="local",
        seed=41,
        reason="claim_support:unsupported_claim",
        contract_fingerprint="fp-current",
    )
    assert len(pool) == 1
    only = pool.candidates()[0]
    assert only.contract_fingerprint == "fp-current"
    assert only.from_resume_cache is False
    assert pool.counters()["superseded_resume_cache"] == 1


def test_pool_ignores_a_none_payload_so_off_never_pools() -> None:
    pool = RejectPool()
    assert pool.record_rejection(
        chunk_id="chunk-0001",
        kind="instruction",
        variant_index=0,
        pair=build_capture_payload(
            _pair(),
            reason="claim_support:unsupported_claim",
            mode=MINE_MODE_OFF,
        ),
        provider="local",
        seed=1,
        reason="claim_support:unsupported_claim",
        contract_fingerprint="fp-current",
    ) is None
    assert len(pool) == 0
    assert pool.counters()["captured_this_run"] == 0


def test_pool_order_is_total_and_independent_of_arrival_order() -> None:
    keys = [("chunk-0002", 1), ("chunk-0001", 3), ("chunk-0001", 0)]
    forward, backward = RejectPool(), RejectPool()
    for pool, order in ((forward, keys), (backward, list(reversed(keys)))):
        for chunk_id, variant in order:
            pool.record_rejection(
                chunk_id=chunk_id,
                kind="instruction",
                variant_index=variant,
                pair=build_capture_payload(
                    _pair(chunk_id=chunk_id),
                    reason="claim_support:unsupported_claim",
                    mode=MINE_MODE_ON,
                ),
                provider="local",
                seed=3,
                reason="claim_support:unsupported_claim",
                contract_fingerprint="fp-current",
            )
    assert [c.key for c in forward.candidates()] == [
        ("chunk-0001", "instruction", 0),
        ("chunk-0001", "instruction", 3),
        ("chunk-0002", "instruction", 1),
    ]
    assert [c.key for c in forward.candidates()] == [
        c.key for c in backward.candidates()
    ]


def test_pool_is_safe_under_concurrent_writers() -> None:
    pool = RejectPool()
    payload = build_capture_payload(
        _pair(), reason="claim_support:unsupported_claim", mode=MINE_MODE_ON,
    )
    barrier = threading.Barrier(8)

    def worker(offset: int) -> None:
        barrier.wait()
        for index in range(25):
            pool.record_rejection(
                chunk_id=f"chunk-{offset:02d}{index:02d}",
                kind="instruction",
                variant_index=index % 3,
                pair=dict(payload or {}),
                provider="local",
                seed=index,
                reason="claim_support:unsupported_claim",
                contract_fingerprint="fp-current",
            )

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(pool) == 200
    assert pool.counters()["captured_this_run"] == 200
    assert len(pool.candidates()) == 200
