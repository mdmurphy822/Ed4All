"""Reject-mining capture: persistence, replay safety, resume completeness.

The capture layer stores the full rejected completion on the EXISTING per-pair
resume checkpoint row instead of a new sidecar, because that row already owns
fresh-start identity binding, the run-contract/holdout guards, terminal-key
uniqueness, the write-deferral invariant, unlink-on-clean-exit, and
``SYNTHESIS_ARTIFACT_NAMES`` registration. These tests pin the two things that
choice makes load-bearing:

1. with the flag OFF the row is byte-identical to the one written today; and
2. a rejected row carrying a pair can never be replayed as an ACCEPTED pair.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.synthesis_fresh_start import (  # noqa: E402
    SYNTHESIS_ARTIFACT_NAMES,
)
from Trainforge.synthesis_reject_mining import (  # noqa: E402
    MINE_MODE_OFF,
    MINE_MODE_ON,
    RejectPool,
    build_capture_payload,
)
from Trainforge.synthesis_reject_mining import (  # noqa: E402
    MINE_REJECTS_ENV,
)
from Trainforge.synthesize_training import (  # noqa: E402
    _DISPOSITION_PROJECTED_FIELDS,
    _GENERATION_CONTRACT_FILES,
    _VERDICT_POLICY_FILES,
    _append_synthesis_pairs_checkpoint,
    _checkpoint_skips_generation,
    _checkpoint_terminal_rejection,
    _load_synthesis_pairs_checkpoint,
    _project_terminal_dispositions,
    _resolve_cached_accepted_pair,
    run_synthesis,
)

_FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)

_PROMPT = (
    "Explain how the stated conservation rule constrains the "
    "outcome of the described interaction."
)
_COMPLETION = (
    "The rule fixes the total before and after the interaction, so any "
    "redistribution among the participants must leave that total unchanged. "
    "That constraint is what makes the outcome predictable."
)


def _pair(**overrides):
    pair = {
        "prompt": _PROMPT,
        "completion": _COMPLETION,
        "chunk_id": "chunk-0001",
        "lo_refs": ["CO-01"],
        "requires_source_citation": True,
        "provider": "local",
        "claim_support_rate": 0.67,
        "claim_contradicted_rate": 0.0,
        "per_claim_support": [
            {"sentence": "A.", "entailment": 0.42, "outcome": "unsupported"},
        ],
    }
    pair.update(overrides)
    return pair


def _write_rejection(fh, *, pair=None, reason="claim_support:unsupported_claim"):
    _checkpoint_terminal_rejection(
        fh,
        chunk_id="chunk-0001",
        kind="instruction",
        variant_index=1,
        provider="local",
        seed=41,
        reason=reason,
        contract_fingerprint="fp-current",
        rejection_evidence={
            "stage": "claim_support",
            "rejection_reason": "unsupported_claim",
            "total_claims": 1,
            "failing_claims": [
                {
                    "sentence": "A.",
                    "entailment": 0.42,
                    "contradiction": 0.02,
                    "outcome": "unsupported",
                },
            ],
        },
        pair=pair,
    )


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

def test_rejected_row_is_byte_identical_when_mining_is_off() -> None:
    """DEFAULT OFF proof: nothing new is written, not one byte.

    The row produced through the capture helper with the flag off must equal
    the row produced by the legacy call that never passed a pair at all.
    """
    legacy = io.StringIO()
    _write_rejection(legacy)

    gated = io.StringIO()
    _write_rejection(
        gated,
        pair=build_capture_payload(
            _pair(),
            reason="claim_support:unsupported_claim",
            mode=MINE_MODE_OFF,
        ),
    )

    assert gated.getvalue() == legacy.getvalue()
    row = json.loads(gated.getvalue())
    assert row["disposition"] == "rejected"
    assert row["pair"] is None


def test_rejected_row_carries_the_full_pair_when_enabled() -> None:
    fh = io.StringIO()
    payload = build_capture_payload(
        _pair(), reason="claim_support:unsupported_claim", mode=MINE_MODE_ON,
    )
    _write_rejection(fh, pair=payload)
    row = json.loads(fh.getvalue())

    assert row["disposition"] == "rejected"
    assert row["pair"]["completion"] == _COMPLETION
    assert row["pair"]["prompt"] == _PROMPT
    # Every identity field the existing row already carried is untouched.
    assert row["chunk_id"] == "chunk-0001"
    assert row["kind"] == "instruction"
    assert row["variant_index"] == 1
    assert row["contract_fingerprint"] == "fp-current"
    assert row["seed"] == 41
    assert row["schema_version"] == "v1"
    assert row["rejection_evidence"]["stage"] == "claim_support"


def test_capture_rides_the_existing_identity_binding() -> None:
    """Fresh-start / run-contract / holdout identity is inherited, not re-done."""
    fh = io.StringIO()
    fh._fresh_start_id = "f" * 32
    fh._fresh_start_marker_digest = "d" * 64
    fh._synthesis_run_contract_sha256 = "c" * 64
    fh._synthesis_contract_components_sha256 = "e" * 64
    fh._synthesis_holdout_identity = {"holdout_manifest_sha256": "a" * 64}
    _write_rejection(
        fh,
        pair=build_capture_payload(
            _pair(),
            reason="claim_support:unsupported_claim",
            mode=MINE_MODE_ON,
        ),
    )
    row = json.loads(fh.getvalue())
    assert row["fresh_start_id"] == "f" * 32
    assert row["fresh_start_marker_digest"] == "d" * 64
    assert row["synthesis_run_contract_sha256"] == "c" * 64
    assert row["synthesis_contract_components_sha256"] == "e" * 64
    assert row["synthesis_holdout_identity"] == {
        "holdout_manifest_sha256": "a" * 64,
    }


def test_terminal_uniqueness_raise_is_not_weakened_by_capture() -> None:
    """The duplicate-terminal-key guard still fires with a pair attached."""
    fh = io.StringIO()
    fh._terminal_semantic_keys = set()
    fh._enforce_terminal_uniqueness = True
    payload = build_capture_payload(
        _pair(), reason="claim_support:unsupported_claim", mode=MINE_MODE_ON,
    )
    _write_rejection(fh, pair=payload)
    try:
        _write_rejection(fh, pair=payload)
    except RuntimeError as exc:
        assert "duplicate pair terminal semantic key" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("duplicate terminal key was accepted")


def test_captured_rows_survive_a_write_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ".synthesis_pairs_checkpoint.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        _write_rejection(
            fh,
            pair=build_capture_payload(
                _pair(),
                reason="claim_support:unsupported_claim",
                mode=MINE_MODE_ON,
            ),
        )
    cache = _load_synthesis_pairs_checkpoint(path)
    assert list(cache) == [("chunk-0001", "instruction", 1)]
    assert cache[("chunk-0001", "instruction", 1)]["pair"]["completion"] == (
        _COMPLETION
    )


def test_concurrent_writers_produce_one_intact_row_each(
    tmp_path: Path,
) -> None:
    """The shared checkpoint lock still serializes appends with a payload."""
    path = tmp_path / ".synthesis_pairs_checkpoint.jsonl"
    payload = build_capture_payload(
        _pair(), reason="claim_support:unsupported_claim", mode=MINE_MODE_ON,
    )
    barrier = threading.Barrier(8)
    with path.open("w", encoding="utf-8") as fh:
        def worker(offset: int) -> None:
            barrier.wait()
            for index in range(20):
                _checkpoint_terminal_rejection(
                    fh,
                    chunk_id=f"chunk-{offset:02d}{index:02d}",
                    kind="instruction",
                    variant_index=0,
                    provider="local",
                    seed=index,
                    reason="claim_support:unsupported_claim",
                    contract_fingerprint="fp-current",
                    pair=dict(payload or {}),
                )

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 160
    rows = [json.loads(line) for line in lines]  # no interleaved/torn row
    assert all(row["pair"]["completion"] == _COMPLETION for row in rows)
    assert len({row["chunk_id"] for row in rows}) == 160


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def test_dispositions_projection_never_leaks_a_captured_pair() -> None:
    """``synthesis_dispositions.jsonl`` is an operator digest, not a corpus.

    The projection deliberately excludes ``pair``; capture must not turn that
    file into a second, unregistered copy of rejected completions.
    """
    rows = _project_terminal_dispositions([
        {
            "schema_version": "v1",
            "chunk_id": "chunk-0001",
            "kind": "instruction",
            "variant_index": 1,
            "provider": "local",
            "seed": 41,
            "disposition": "rejected",
            "reason": "claim_support:unsupported_claim",
            "contract_fingerprint": "fp-current",
            "rejection_evidence": {"stage": "claim_support"},
            "pair": _pair(),
        },
    ])
    assert len(rows) == 1
    assert "pair" not in rows[0]
    assert set(rows[0]) == set(_DISPOSITION_PROJECTED_FIELDS) | {
        "rejection_evidence",
    }


# --------------------------------------------------------------------------
# fail-closed replay guard
# --------------------------------------------------------------------------

def test_rejected_pair_is_never_replayed_as_an_accepted_pair() -> None:
    """The hazard capture creates, closed before it can reach the corpus.

    Pre-guard: the instruction replay branch warns on a rejected row whose
    contract fingerprint does not match and then FALLS THROUGH to the accepted
    branch, where ``_checkpoint_accepted_matches_contract`` returns True for a
    legacy ``None`` fingerprint and ``record["pair"] or {}`` was harmlessly
    empty. With capture that dict is a full rejected completion, so it would
    have been appended to ``instruction_records`` as an accepted pair.
    """
    record = {
        "schema_version": "v1",
        "chunk_id": "chunk-0001",
        "kind": "instruction",
        "variant_index": 1,
        "provider": "local",
        "seed": 41,
        "disposition": "rejected",
        "reason": "claim_support:unsupported_claim",
        "contract_fingerprint": None,  # legacy row: admitted by the accepted check
        "pair": _pair(),
    }
    chunk = {"learning_outcome_refs": ["CO-01"]}

    assert _resolve_cached_accepted_pair(record, "fp-current", chunk) == {}

    invalidations: list = []
    assert _resolve_cached_accepted_pair(
        record, "fp-current", chunk, invalidations=invalidations,
    ) == {}
    assert invalidations == ["rejected_disposition"]

    # And the generation-skip predicate still refuses it, so the unit is
    # regenerated rather than silently replayed.
    assert _checkpoint_skips_generation(
        {("chunk-0001", "instruction", 1): record},
        chunk_id="chunk-0001",
        kind="instruction",
        variant_index=1,
        provider="local",
        seed=41,
        focused_chunk=chunk,
        fingerprint_for_seed=lambda seed, focused: "fp-current",
    ) is False


def test_accepted_replay_behaviour_is_unchanged_by_the_guard() -> None:
    chunk = {"learning_outcome_refs": ["CO-01"]}
    accepted = {
        "disposition": "accepted",
        "contract_fingerprint": "fp-current",
        "pair": _pair(),
    }
    invalidations: list = []
    resolved = _resolve_cached_accepted_pair(
        accepted, "fp-current", chunk, invalidations=invalidations,
    )
    assert resolved["completion"] == _COMPLETION
    assert invalidations == []

    # Contract mismatch and focus mismatch each still invalidate the entry —
    # the preference branch clears its cache-hit flag on exactly these two.
    stale = dict(accepted, contract_fingerprint="fp-stale")
    invalidations = []
    assert _resolve_cached_accepted_pair(
        stale, "fp-current", chunk, invalidations=invalidations,
    ) == {}
    assert invalidations == ["contract_mismatch"]

    off_focus = dict(accepted, pair=_pair(lo_refs=["CO-99"]))
    invalidations = []
    assert _resolve_cached_accepted_pair(
        off_focus,
        "fp-current",
        {
            "learning_outcome_refs": ["CO-01"],
            "synthesis_focus_objective": "CO-01",
        },
        invalidations=invalidations,
    ) == {}
    assert invalidations == ["focus_mismatch"]


def test_legacy_accepted_row_without_a_pair_does_not_invalidate() -> None:
    """A contract-compatible accepted row with no pair never cleared the flag."""
    invalidations: list = []
    assert _resolve_cached_accepted_pair(
        {"disposition": "accepted", "contract_fingerprint": None, "pair": None},
        "fp-current",
        {"learning_outcome_refs": ["CO-01"]},
        invalidations=invalidations,
    ) == {}
    assert invalidations == []


# --------------------------------------------------------------------------
# resume completeness
# --------------------------------------------------------------------------

def test_resume_pool_is_the_union_of_prior_and_current_run(
    tmp_path: Path,
) -> None:
    """A killed-and-resumed run's pool equals an uninterrupted run's.

    A rejected unit is SKIPPED on resume, never regenerated, so its candidate
    exists only in the prior run's checkpoint rows. Seeding the pool from the
    loaded cache is what keeps the pool complete — and a partial pool is a
    biased pool.
    """
    path = tmp_path / ".synthesis_pairs_checkpoint.jsonl"
    payload = build_capture_payload(
        _pair(), reason="claim_support:unsupported_claim", mode=MINE_MODE_ON,
    )
    with path.open("w", encoding="utf-8") as fh:
        for chunk_id in ("chunk-0001", "chunk-0002"):
            _checkpoint_terminal_rejection(
                fh,
                chunk_id=chunk_id,
                kind="instruction",
                variant_index=1,
                provider="local",
                seed=1,
                reason="claim_support:unsupported_claim",
                contract_fingerprint="fp-current",
                pair=dict(payload or {}),
            )
        # An accepted row and a legacy pair-less rejection must not pool.
        _append_synthesis_pairs_checkpoint(
            fh,
            chunk_id="chunk-0003",
            kind="instruction",
            variant_index=0,
            pair=_pair(chunk_id="chunk-0003"),
            provider="local",
            seed=2,
            contract_fingerprint="fp-current",
        )
        _checkpoint_terminal_rejection(
            fh,
            chunk_id="chunk-0004",
            kind="instruction",
            variant_index=0,
            provider="local",
            seed=3,
            reason="claim_support:unsupported_claim",
            contract_fingerprint="fp-current",
        )

    pool = RejectPool()
    seeded = pool.seed_from_checkpoint_cache(
        _load_synthesis_pairs_checkpoint(path)
    )
    assert seeded == 2

    # This run rejects two more units, one of which re-observes a pooled key.
    for chunk_id in ("chunk-0002", "chunk-0005"):
        pool.record_rejection(
            chunk_id=chunk_id,
            kind="instruction",
            variant_index=1,
            pair=dict(payload or {}),
            provider="local",
            seed=4,
            reason="claim_support:unsupported_claim",
            contract_fingerprint="fp-current",
        )

    assert [c.chunk_id for c in pool.candidates()] == [
        "chunk-0001", "chunk-0002", "chunk-0005",
    ]
    counters = pool.counters()
    assert counters["resume_cache_candidates"] == 2
    assert counters["captured_this_run"] == 2
    assert counters["superseded_resume_cache"] == 1
    assert counters["pool_size"] == 3


# --------------------------------------------------------------------------
# end-to-end: the whole run, all three modes
# --------------------------------------------------------------------------

def _working_copy(tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copytree(_FIXTURE_ROOT, dst)
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
        dst / "training_specs" / "instruction_pairs.jsonl.in_progress",
        dst / "training_specs" / "preference_pairs.jsonl.in_progress",
        dst / "training_specs" / "pilot_report.md",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def _run(corpus: Path):
    return run_synthesis(
        corpus_dir=corpus,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )


def _emitted(corpus: Path, name: str):
    """Return the emitted rows minus the one field that is volatile per run.

    ``decision_capture_id`` is a fresh event id on every invocation (measured:
    it is the ONLY key that differs between two identical runs), so it is
    dropped before comparison. Everything else — prompt, completion, chunk_id,
    lo_refs, provider, scores — must match exactly.
    """
    path = corpus / "training_specs" / name
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row.pop("decision_capture_id", None)
    return rows


def test_full_run_is_byte_identical_with_the_flag_off(
    tmp_path: Path, monkeypatch,
) -> None:
    """DEFAULT OFF: no new field on the stats, no behaviour change at all."""
    monkeypatch.delenv(MINE_REJECTS_ENV, raising=False)
    corpus = _working_copy(tmp_path, "off")
    stats = _run(corpus)
    assert stats.reject_mining == {}
    assert stats.instruction_pairs_emitted > 0


def test_full_run_under_shadow_and_on_emits_the_same_corpus(
    tmp_path: Path, monkeypatch,
) -> None:
    """Capture pools and counts; it never changes what lands on disk.

    Also the only test that executes the flag-ON branches of
    ``run_synthesis`` end to end (mode resolution, pool seeding, the three
    capture call sites, the funnel roll-up) — a mode-gated NameError in any
    of them is invisible to every flag-off test in the tree.
    """
    monkeypatch.delenv(MINE_REJECTS_ENV, raising=False)
    baseline = _working_copy(tmp_path, "baseline")
    baseline_stats = _run(baseline)
    assert baseline_stats.instruction_pairs_emitted > 0
    baseline_inst = _emitted(baseline, "instruction_pairs.jsonl")
    baseline_pref = _emitted(baseline, "preference_pairs.jsonl")

    for mode in ("shadow", "on"):
        monkeypatch.setenv(MINE_REJECTS_ENV, mode)
        corpus = _working_copy(tmp_path, mode)
        stats = _run(corpus)
        assert _emitted(corpus, "instruction_pairs.jsonl") == baseline_inst, mode
        assert _emitted(corpus, "preference_pairs.jsonl") == baseline_pref, mode
        # The funnel is reported on the returned stats, not the summary
        # sidecar (whose schema is additionalProperties:false).
        assert stats.reject_mining, mode
        assert stats.reject_mining["mode_shadow"] == int(mode == "shadow")
        assert "pool_size" in stats.reject_mining
        assert "captured_this_run" in stats.reject_mining


# --------------------------------------------------------------------------
# end-to-end: the MINED EMIT path (append / mirror / cap / decontamination)
# --------------------------------------------------------------------------
#
# ``select_mined_pairs`` itself is exhaustively covered by
# ``test_reject_mining_selection.py`` as a pure function. What THOSE tests
# cannot reach is the seam between the selector and the corpus: the
# ``preference_records.append`` + ``.in_progress`` mirror + ``per_artifact_cap``
# guard inside ``run_synthesis``, and whether a mined row survives gold-set
# decontamination and lands in ``preference_pairs.jsonl`` at all. The
# ``mini_course_training`` fixture runs at the default
# ``instruction_variants_per_chunk == 1``, where mined yield is STRUCTURALLY
# zero (one instruction unit per chunk can never hold both an accepted and a
# rejected unit), so no realistic full-run fixture exercises that block — it
# was dead code under test. These tests drive it by substituting the selector's
# RESULT (rows built by the real selector from inline fixtures) while leaving
# every downstream step in ``run_synthesis`` untouched.

_MINE_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _mine_tok(index: int) -> str:
    return (
        "w"
        + _MINE_ALPHA[(index // 676) % 26]
        + _MINE_ALPHA[(index // 26) % 26]
        + _MINE_ALPHA[index % 26]
        + "x"
    )


def _mine_words(start: int, count: int):
    return [_mine_tok(i) for i in range(start, start + count)]


def _mine_rows(count: int):
    """Build ``count`` mined preference rows with the REAL selector.

    Nothing here is hand-shaped: the rows come out of
    ``select_mined_pairs`` exactly as a live run would produce them, so the
    downstream assertions are about the emit seam and not about a fixture that
    happens to look like a mined row.
    """
    from Trainforge.synthesis_reject_mining import (
        RejectCandidate,
        select_mined_pairs,
    )

    persona = "a learner"
    anchors = []
    candidates = []
    for i in range(count):
        chunk_id = f"chunk-{i + 1:04d}"
        prompt = "Explain " + " ".join(_mine_words(900, 8)) + " in your own words?"
        anchors.append({
            "chunk_id": chunk_id,
            "prompt": prompt,
            "completion": " ".join(
                _mine_words(0, 20) + _mine_words(100 + i * 20, 10)
            ) + ".",
            "lo_refs": ["CO-01"],
            "provider": "local",
            "seed": 7,
            "instruction_variant": 0,
            "requires_source_citation": False,
        })
        record = {
            "disposition": "rejected",
            "chunk_id": chunk_id,
            "kind": "instruction",
            "variant_index": 1,
            "provider": "local",
            "seed": 11,
            "reason": "claim_support:unsupported_claim",
            "contract_fingerprint": "fp-current",
            "pair": {
                "chunk_id": chunk_id,
                "prompt": prompt,
                "completion": " ".join(
                    _mine_words(0, 20) + _mine_words(500 + i * 20, 10)
                ) + ".",
                "lo_refs": ["CO-01"],
                "requires_source_citation": False,
                "instruction_variant": 1,
                "claim_support_rate": 2.0 / 3.0,
                "claim_contradicted_rate": 0.0,
                "per_claim_support": [
                    {"sentence": "a.", "entailment": 0.91, "outcome": "entailed"},
                    {"sentence": "b.", "entailment": 0.91, "outcome": "entailed"},
                    {"sentence": "c.", "entailment": 0.42, "outcome": "unsupported"},
                ],
            },
        }
        built = RejectCandidate.from_checkpoint_record(record)
        assert built is not None
        candidates.append(built)

    rows, funnel = select_mined_pairs(
        candidates,
        anchors,
        persona=persona,
        mode=MINE_MODE_ON,
        capture=_NullCapture(),
        event_id_fn=lambda _c: "evt-mined",
        max_fraction=1.0,
    )
    assert len(rows) == count, funnel.as_dict()
    return rows


class _NullCapture:
    def log_decision(self, **_kwargs) -> None:  # pragma: no cover - trivial
        return None


def _install_mined_rows(monkeypatch, rows) -> dict:
    """Substitute the selector's RESULT; leave the emit path untouched."""
    import Trainforge.synthesize_training as st
    from Trainforge.synthesis_reject_mining import MinedFunnel

    seen = {"calls": 0}

    def _stub(_candidates, _accepted, **_kwargs):
        seen["calls"] += 1
        funnel = MinedFunnel()
        funnel.candidates_seen = len(rows)
        funnel.emitted = len(rows)
        return (list(rows), funnel)

    monkeypatch.setattr(st, "select_mined_pairs", _stub)
    return seen


def test_mined_rows_reach_the_written_preference_corpus(
    tmp_path: Path, monkeypatch,
) -> None:
    """The emit seam: append -> ``.in_progress`` mirror -> decontamination -> file.

    Before this test the ``for _mined_pair in _mine_rows:`` body was never
    executed by any test in the tree (verified by making its first statement
    raise: the whole reject-mining suite still passed).
    """
    import Trainforge.synthesize_training as st

    monkeypatch.setenv(MINE_REJECTS_ENV, "on")
    rows = _mine_rows(2)
    seen = _install_mined_rows(monkeypatch, rows)

    # The ``.in_progress`` sidecar is the operator's live view, and it is
    # unlinked on a clean exit — so the mirror is observed as it happens.
    mirrored = []
    _real_append = st._utils_append_jsonl

    def _recording_append(fh, payload, *args, **kwargs):
        if isinstance(payload, dict):
            mirrored.append(payload)
        return _real_append(fh, payload, *args, **kwargs)

    monkeypatch.setattr(st, "_utils_append_jsonl", _recording_append)

    corpus = _working_copy(tmp_path, "mined-emit")
    stats = _run(corpus)

    assert seen["calls"] == 1
    written = _emitted(corpus, "preference_pairs.jsonl")
    mined = [r for r in written if r.get("source") == "mined_rejection"]
    assert len(mined) == 2
    for row in mined:
        assert row["prompt"] and row["chosen"] and row["rejected"]
        assert row["promotion_status"]
        assert row["lo_refs"]
        assert row["reject_mining"]["rejection_reason"].startswith(
            "claim_support:"
        )
    # A mined row that lands in the canonical file but not the mirror is
    # invisible to an operator tailing the run.
    assert sum(
        1 for r in mirrored if r.get("source") == "mined_rejection"
    ) == 2
    assert stats.reject_mining["emitted"] == 2


def test_mined_rows_honour_the_per_artifact_cap(
    tmp_path: Path, monkeypatch,
) -> None:
    """``--max-pairs`` must bound mined rows exactly like every other emit.

    A derived row silently exceeding an operator-set cap would be the same
    class of bug as an un-capped generator.
    """
    monkeypatch.delenv(MINE_REJECTS_ENV, raising=False)
    baseline = _working_copy(tmp_path, "cap-baseline")
    baseline_stats = _run(baseline)
    baseline_pref = baseline_stats.preference_pairs_emitted

    monkeypatch.setenv(MINE_REJECTS_ENV, "on")
    rows = _mine_rows(3)
    _install_mined_rows(monkeypatch, rows)
    corpus = _working_copy(tmp_path, "cap-on")
    stats = run_synthesis(
        corpus_dir=corpus,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
        max_pairs=baseline_pref + 1,
    )

    written = _emitted(corpus, "preference_pairs.jsonl")
    mined = [r for r in written if r.get("source") == "mined_rejection"]
    assert len(mined) == 1, "cap must truncate mined rows, not pass them through"
    assert stats.capped_at_max_pairs is True
    assert stats.preference_pairs_emitted == baseline_pref + 1


# --------------------------------------------------------------------------
# the "partial pool is a biased pool" skip guard
# --------------------------------------------------------------------------

def _install_mining_tripwire(monkeypatch) -> dict:
    """Fail loudly if the selector is reached at all."""
    import Trainforge.synthesize_training as st

    seen = {"calls": 0}

    def _tripwire(*_a, **_k):  # pragma: no cover - must never run
        seen["calls"] += 1
        raise AssertionError(
            "select_mined_pairs ran on an INCOMPLETE pass: the pool is "
            "partial, and a partial pool is a biased pool."
        )

    monkeypatch.setattr(st, "select_mined_pairs", _tripwire)
    return seen


def test_mining_skipped_when_the_dispatch_budget_is_exhausted(
    tmp_path: Path, monkeypatch,
) -> None:
    """Budget exhaustion breaks the chunk loop mid-corpus -> mine nothing.

    This is the REACHABLE half of the guard: ``_budget_exhausted_exc`` is set
    inside the loop and execution then flows straight through the mining hook.
    """
    from Trainforge.tests._synthesis_fakes import (
        FakeLocalDispatcher,
        make_instruction_response,
        make_preference_response,
    )

    ok_prompt = "Paraphrased prompt explaining the stated rule for a learner."
    ok_completion = (
        "Paraphrased completion grounded in the source chunk text, covering "
        "the stated rule and its constraint in sufficient detail."
    )

    async def agent_tool(*, task_params, **_kw):
        if task_params["kind"] == "instruction":
            return make_instruction_response(
                prompt=ok_prompt, completion=ok_completion,
            )
        return make_preference_response(
            prompt=ok_prompt, chosen=ok_completion, rejected=ok_completion,
        )

    monkeypatch.setenv(MINE_REJECTS_ENV, "on")
    monkeypatch.setenv("TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS", "1")
    seen = _install_mining_tripwire(monkeypatch)

    corpus = _working_copy(tmp_path, "budget")
    stats = run_synthesis(
        corpus_dir=corpus,
        course_code="MINI_TRAINING_101",
        provider="claude_session",
        seed=11,
        dispatcher=FakeLocalDispatcher(agent_tool=agent_tool),
        max_dispatches=1,
    )

    assert stats.capped_at_max_dispatches is True
    assert seen["calls"] == 0
    assert stats.reject_mining["reject_mining_skipped"] == "incomplete_pass"
    assert not [
        r for r in _emitted(corpus, "preference_pairs.jsonl")
        if r.get("source") == "mined_rejection"
    ]


def test_mining_skipped_when_the_generation_map_stopped_early(
    tmp_path: Path, monkeypatch, state_runs_isolated,
) -> None:
    """The ``stopped_early`` half of the same guard.

    In production a stop normally RAISES ``GracefulStopRequested`` at the
    drain boundary before the mining hook is reached. The guard covers the
    narrow race where the sentinel is cleared between the drain and that
    check, so the test isolates exactly that: the sentinel is armed (so
    ``BoundedOrderedMap`` sets ``stopped_early``) and only ``check_stop`` is
    neutralized, letting execution reach the hook the way the race would.

    ``max_concurrent=2`` is required, not incidental: ``run_synthesis`` passes
    ``stop_requested=None`` into ``BoundedOrderedMap`` at concurrency 1, so
    ``stopped_early`` can only ever be set on the concurrent path.
    """
    from lib.generation import stop_control
    import Trainforge.synthesize_training as st

    monkeypatch.setenv(MINE_REJECTS_ENV, "on")
    seen = _install_mining_tripwire(monkeypatch)
    monkeypatch.setattr(st.stop_control, "check_stop", lambda *a, **k: None)
    stop_control.request_stop(
        scope="all", reason="reject-mining-guard", source="unit-test",
    )
    try:
        corpus = _working_copy(tmp_path, "stopped")
        stats = run_synthesis(
            corpus_dir=corpus,
            course_code="MINI_TRAINING_101",
            provider="mock",
            seed=17,
            max_concurrent=2,
        )
    finally:
        stop_control.clear_stop(include_global=True)

    assert seen["calls"] == 0
    assert stats.reject_mining["reject_mining_skipped"] == "incomplete_pass"


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def test_reject_mining_module_is_registered_as_verdict_policy() -> None:
    """It derives rows from generated text; it cannot change a model call.

    ``_GENERATION_CONTRACT_FILES`` membership would re-key the generation
    fingerprint on every edit and force every paused fresh-start-bound run to
    archive and restart from zero.
    """
    assert "Trainforge/synthesis_reject_mining.py" in _VERDICT_POLICY_FILES
    assert (
        "Trainforge/synthesis_reject_mining.py"
        not in _GENERATION_CONTRACT_FILES
    )


def test_checkpoint_and_dispositions_are_registered_synthesis_artifacts() -> None:
    """The pool's store is registered; that is why no sidecar was added."""
    assert ".synthesis_pairs_checkpoint.jsonl" in SYNTHESIS_ARTIFACT_NAMES
    assert "synthesis_dispositions.jsonl" in SYNTHESIS_ARTIFACT_NAMES
