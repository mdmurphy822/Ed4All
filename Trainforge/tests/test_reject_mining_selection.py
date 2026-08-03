"""Selection layer for reject-mined DPO negatives.

Every test here runs with NO model seat, NO network, NO embedder and NO NLI —
that is the point of the design. ``select_mined_pairs`` is a pure function of
values already persisted on the pair checkpoint, so the entire near-miss
judgement is unit-testable and deterministic.

The population these tests are ultimately about is the measured one: across
three archived cohorts acceptance was 0/52, 2/52, 4/52 (~3.85%), and on 20 of
26 audited chunks EVERY content source was empty, so the completion collapsed
to a fixed pedagogical tail with a database key spliced into the topic slot
("...break this down and explain the parts of learning outcome co-117"),
entailment 0.028. Those are correct rejections and terrible DPO negatives:
trivially separable, so the preference teaches distribution discrimination
instead of quality discrimination. FLAME (arXiv:2405.01525) measured naive DPO
reducing factuality 44.7 -> 42.3 FActScore exactly that way.
"""
from __future__ import annotations

import random
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from Trainforge.synthesis_reject_mining import (
    DEFAULT_MAX_FRACTION,
    DEFAULT_MAX_SKELETON_FREQ,
    DEFAULT_MIN_FAIL_ENTAILMENT,
    DEFAULT_MIN_SUPPORT,
    MAX_FRACTION_ENV,
    MAX_SKELETON_FREQ_ENV,
    MIN_FAIL_ENTAILMENT_ENV,
    MIN_SUPPORT_ENV,
    MINE_MODE_OFF,
    MINE_MODE_ON,
    MINE_MODE_SHADOW,
    MINE_REJECTS_ENV,
    MINED_PAIR_SOURCE,
    SELECTION_DECISION_TYPE,
    RejectCandidate,
    _lo_id_tokens,
    completion_skeleton,
    resolve_max_fraction,
    resolve_max_skeleton_freq,
    resolve_min_fail_entailment,
    resolve_min_support,
    resolve_mine_rejects_mode,
    select_mined_pairs,
    token_jaccard,
    unframe_instruction_prompt,
)
from Trainforge.synthesize_training import _apply_instruction_variant

PERSONA = "a learner"
CHUNK = "chunk-0001"


# --------------------------------------------------------------------------- #
# Fixture builders (inline shapes only — no course slug, corpus path, or book)  #
# --------------------------------------------------------------------------- #

_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _tok(index: int) -> str:
    """A deterministic >=3-char alphabetic content token.

    Alphabetic on purpose so a fixture's identity never depends on how the D3
    skeleton normalizer treats digits (it does NOT normalize them — see
    ``test_numeric_answers_do_not_share_a_skeleton``).
    """
    return (
        "w"
        + _ALPHA[(index // 676) % 26]
        + _ALPHA[(index // 26) % 26]
        + _ALPHA[index % 26]
        + "x"
    )


def _words(start: int, count: int) -> List[str]:
    return [_tok(i) for i in range(start, start + count)]


def _base_prompt(offset: int = 900) -> str:
    """A >=40-char base question whose tokens never appear in a completion."""
    return "Explain " + " ".join(_words(offset, 8)) + " in your own words?"


def _completion(shared_start: int, uniq_start: int) -> str:
    """20 shared + 10 unique tokens -> pairwise Jaccard 0.5 with its sibling."""
    return " ".join(_words(shared_start, 20) + _words(uniq_start, 10)) + "."


def _per_claim(
    *,
    n_claims: int = 3,
    failing_entailment: float = 0.42,
    contradicted: bool = False,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for i in range(n_claims - 1):
        entries.append({
            "sentence": f"supported claim {i}",
            "entailment": 0.91,
            "contradiction": 0.01,
            "outcome": "entailed",
        })
    entries.append({
        "sentence": "the drifting claim",
        "entailment": failing_entailment,
        "contradiction": 0.40 if contradicted else 0.02,
        "outcome": "contradicted" if contradicted else "unsupported",
    })
    return entries


def _anchor(
    *,
    chunk_id: str = CHUNK,
    variant: int = 0,
    prompt: Optional[str] = None,
    completion: Optional[str] = None,
    lo_refs: Optional[List[str]] = None,
    provider: str = "local",
    requires_citation: Optional[bool] = None,
    generating_seat: Optional[str] = None,
    seed: int = 7,
) -> Dict[str, Any]:
    """One ACCEPTED instruction record, framed by the production helper."""
    pair: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "prompt": prompt if prompt is not None else _base_prompt(),
        "completion": (
            completion if completion is not None else _completion(0, 100)
        ),
        "lo_refs": list(lo_refs) if lo_refs is not None else ["CO-01"],
        "provider": provider,
        "seed": seed,
        "source_references": [],
    }
    _apply_instruction_variant(pair, variant, learner_persona=PERSONA)
    if requires_citation is not None:
        pair["requires_source_citation"] = bool(requires_citation)
    if generating_seat is not None:
        pair["generating_seat"] = generating_seat
    return pair


def _candidate(
    *,
    chunk_id: str = CHUNK,
    variant: int = 1,
    prompt: Optional[str] = None,
    completion: Optional[str] = None,
    lo_refs: Optional[List[str]] = None,
    provider: str = "local",
    reason: str = "claim_support:unsupported_claim",
    support_rate: Optional[float] = 2.0 / 3.0,
    contradicted_rate: float = 0.0,
    per_claim: Optional[List[Dict[str, Any]]] = None,
    requires_citation: Optional[bool] = None,
    generating_seat: Optional[str] = None,
    seed: int = 11,
    fingerprint: Optional[str] = "fp-current",
) -> RejectCandidate:
    """One pooled reject, read through the PRODUCTION checkpoint reader."""
    inner: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "prompt": prompt if prompt is not None else _base_prompt(),
        "completion": (
            completion if completion is not None else _completion(0, 200)
        ),
        "lo_refs": list(lo_refs) if lo_refs is not None else ["CO-01"],
        "claim_support_rate": support_rate,
        "claim_contradicted_rate": contradicted_rate,
        "per_claim_support": (
            per_claim if per_claim is not None else _per_claim()
        ),
    }
    if prompt is None:
        _apply_instruction_variant(inner, variant, learner_persona=PERSONA)
    else:
        inner["instruction_variant"] = variant
        inner["requires_source_citation"] = False
    if requires_citation is not None:
        inner["requires_source_citation"] = bool(requires_citation)
    if generating_seat is not None:
        inner["generating_seat"] = generating_seat
    record = {
        "disposition": "rejected",
        "chunk_id": chunk_id,
        "kind": "instruction",
        "variant_index": variant,
        "provider": provider,
        "seed": seed,
        "reason": reason,
        "contract_fingerprint": fingerprint,
        "pair": inner,
    }
    built = RejectCandidate.from_checkpoint_record(record)
    assert built is not None, "fixture is not a readable mineable reject"
    return built


def _raw_candidate(**overrides: Any) -> RejectCandidate:
    """Construct a candidate DIRECTLY, bypassing the reader's reason filter.

    The capture-side reader (:meth:`RejectCandidate.from_checkpoint_record`)
    admits only ``claim_support:`` rows, so a stage-admission test cannot build
    its fixture through it — the selector's own stage gate has to be exercised
    against a hand-built candidate.
    """
    template = _candidate()
    fields = {
        "chunk_id": template.chunk_id,
        "variant_index": template.variant_index,
        "reason": template.reason,
        "provider": template.provider,
        "seed": template.seed,
        "contract_fingerprint": template.contract_fingerprint,
        "generating_seat": template.generating_seat,
        "prompt": template.prompt,
        "completion": template.completion,
        "lo_refs": template.lo_refs,
        "requires_source_citation": template.requires_source_citation,
        "claim_support_rate": template.claim_support_rate,
        "claim_contradicted_rate": template.claim_contradicted_rate,
        "per_claim_support": template.per_claim_support,
        # Carried explicitly: the direct constructor defaults
        # ``contract_confirmed`` to False (fail-closed for a resume-cache row),
        # and this helper stands in for a LIVE capture, which the production
        # reader marks confirmed by construction. Leaving it False would make
        # every hand-built candidate short-circuit on ``stale_contract`` before
        # reaching the predicate under test.
        "from_resume_cache": template.from_resume_cache,
        "contract_confirmed": template.contract_confirmed,
    }
    fields.update(overrides)
    return RejectCandidate(**fields)


class _RecordingCapture:
    """Minimal DecisionCapture stand-in with the one method selection needs."""

    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        event = dict(kwargs)
        event["event_id"] = f"EVT_{len(self.decisions):04d}"
        self.decisions.append(event)


def _event_id(capture: Any) -> str:
    return str(capture.decisions[-1]["event_id"])


def _select(
    candidates: List[RejectCandidate],
    accepted: List[Dict[str, Any]],
    *,
    mode: str = MINE_MODE_ON,
    capture: Optional[_RecordingCapture] = None,
    **kwargs: Any,
):
    """Drive the selector.

    ``max_fraction`` defaults to 1.0 here so a one-anchor fixture is not
    silently zeroed by the production 25% ceiling (``int(0.25 * 1) == 0``).
    ``test_global_fraction_cap`` exercises the production value explicitly.
    """
    kwargs.setdefault("max_fraction", 1.0)
    return select_mined_pairs(
        candidates,
        accepted,
        persona=PERSONA,
        mode=mode,
        capture=capture if capture is not None else _RecordingCapture(),
        event_id_fn=_event_id,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 1-2. Flag parsing                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,expected", [
    (None, MINE_MODE_OFF),
    ("", MINE_MODE_OFF),
    ("0", MINE_MODE_OFF),
    ("false", MINE_MODE_OFF),
    ("banana", MINE_MODE_OFF),
    ("shadow", MINE_MODE_SHADOW),
    ("SHADOW", MINE_MODE_SHADOW),
    ("1", MINE_MODE_ON),
    ("true", MINE_MODE_ON),
    ("yes", MINE_MODE_ON),
    ("on", MINE_MODE_ON),
    ("ON", MINE_MODE_ON),
])
def test_master_flag_parse(raw, expected):
    env: Dict[str, str] = {} if raw is None else {MINE_REJECTS_ENV: raw}
    assert resolve_mine_rejects_mode(env) == expected


@pytest.mark.parametrize("raw", ["", "banana", "-0.5", "1.0", "5", "nan"])
def test_min_support_parse_with_fallback(raw):
    assert resolve_min_support({MIN_SUPPORT_ENV: raw}) == DEFAULT_MIN_SUPPORT


@pytest.mark.parametrize("raw", ["", "banana", "-1", "1.0", "99"])
def test_min_fail_entailment_parse_with_fallback(raw):
    assert (
        resolve_min_fail_entailment({MIN_FAIL_ENTAILMENT_ENV: raw})
        == DEFAULT_MIN_FAIL_ENTAILMENT
    )


@pytest.mark.parametrize("raw", ["", "banana", "0", "-3", "1.5"])
def test_max_skeleton_freq_parse_with_fallback(raw):
    assert (
        resolve_max_skeleton_freq({MAX_SKELETON_FREQ_ENV: raw})
        == DEFAULT_MAX_SKELETON_FREQ
    )


@pytest.mark.parametrize("raw", ["", "banana", "0", "0.0", "-0.1", "1.5"])
def test_max_fraction_parse_with_fallback(raw):
    assert (
        resolve_max_fraction({MAX_FRACTION_ENV: raw}) == DEFAULT_MAX_FRACTION
    )


def test_satellites_accept_in_band_overrides():
    assert resolve_min_support({MIN_SUPPORT_ENV: "0.8"}) == 0.8
    assert (
        resolve_min_fail_entailment({MIN_FAIL_ENTAILMENT_ENV: "0.05"}) == 0.05
    )
    assert resolve_max_skeleton_freq({MAX_SKELETON_FREQ_ENV: "7"}) == 7
    assert resolve_max_fraction({MAX_FRACTION_ENV: "1.0"}) == 1.0


def test_in_band_min_support_override_changes_behavior():
    """A raised floor must actually exclude a candidate the default admits."""
    anchor = _anchor()
    candidate = _candidate()
    rows, _ = _select([candidate], [anchor])
    assert len(rows) == 1
    rows, funnel = _select([candidate], [anchor], min_support=0.9)
    assert rows == []
    assert funnel.below_support_floor == 1


# --------------------------------------------------------------------------- #
# 3. The degenerate population                                                 #
# --------------------------------------------------------------------------- #


def _degenerate_completion(lo_id: str) -> str:
    """The MEASURED template-collapse shape: fixed tail + spliced LO key."""
    return (
        "Learners should be able to break this down and explain the parts of "
        f"learning outcome {lo_id} using what they already know."
    )


def test_measured_degenerate_shape_never_emitted():
    """The exact audited shape: 1 scored claim, entailment 0.028, rate 0.0.

    Pre-fix these were the dominant reject population (20 of 26 sampled chunks
    had every content source empty). They are correct rejections and must
    never become DPO negatives.
    """
    anchor = _anchor()
    candidates = [
        _candidate(
            variant=1,
            completion=_degenerate_completion(f"co-1{i:02d}"),
            support_rate=0.0,
            per_claim=[{
                "sentence": "learners should be able to break this down",
                "entailment": 0.028,
                "contradiction": 0.01,
                "outcome": "unsupported",
            }],
        )
        for i in range(5)
    ]
    rows, funnel = _select(candidates, [anchor])
    assert rows == []
    # A single scored claim carries no ratio signal, so the band stops it
    # before the support floor even gets a look at 0.0.
    assert funnel.too_few_claims == 5
    assert funnel.emitted == 0


def test_degenerate_template_collapse_excluded_by_lo_id_splice():
    """D1 is an INDEPENDENT net, not a consequence of the score band.

    These candidates are deliberately given a band-passing NLI verdict so the
    only thing standing between them and emission is the LO-identifier splice
    check — the exact ``"...learning outcome co-117"`` signature.
    """
    anchor = _anchor()
    candidates = [
        _candidate(
            variant=1,
            completion=(
                _completion(0, 200 + i)
                + " " + _degenerate_completion(f"CO-1{i:02d}")
            ),
        )
        for i in range(5)
    ]
    rows, funnel = _select(candidates, [anchor])
    assert rows == []
    assert funnel.lo_id_splice >= 1
    assert funnel.lo_id_splice == 5


def test_lo_id_splice_uses_the_canonical_helper_pattern():
    """Lowercase and uppercase LO ids are both caught."""
    from lib.ontology.learning_objectives import validate_lo_id

    assert validate_lo_id("CO-117")
    anchor = _anchor()
    lower = _candidate(
        variant=1, completion=_completion(0, 200) + " see outcome co-117.",
    )
    rows, funnel = _select([lower], [anchor])
    assert rows == []
    assert funnel.lo_id_splice == 1


def test_pool_skeleton_collapse_excluded():
    """D3 — repetition IS template collapse; only pooling can see it."""
    anchor = _anchor()
    body = _completion(0, 200)
    candidates = [
        _candidate(chunk_id=f"chunk-{i:04d}", variant=1, completion=body)
        for i in range(1, 4)
    ]
    assert len({completion_skeleton(c.completion) for c in candidates}) == 1
    rows, funnel = _select(candidates, [anchor])
    assert rows == []
    assert funnel.skeleton_collapse == 3


def test_low_novel_content_excluded():
    """D2 — a frame plus the prompt's own topic slot carries no new content."""
    prompt = _base_prompt()
    echo = " ".join(_words(900, 8)) * 4 + " matters here for the learner."
    anchor = _anchor(prompt=prompt)
    candidate = _candidate(variant=1, completion=echo)
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.low_novel_content == 1


def test_realistic_pedagogical_boilerplate_caught_by_novel_content_floor():
    """The degenerate population D1 and D3 do NOT catch.

    The two other degenerate tests each stop their fixture on a different net:
    ``test_measured_degenerate_shape_never_emitted`` carries one scored claim
    so the >=3-claim rule fires first, and
    ``test_degenerate_template_collapse_excluded_by_lo_id_splice`` bolts a
    boilerplate tail onto a near-miss BODY so D1 fires. Neither exercises the
    population that actually threatens the corpus: real pedagogical
    boilerplate that carries >=3 scored claims IN the near-miss band, carries
    NO learning-objective identifier (so D1 is silent), and is UNIQUE per
    candidate (so D3 is silent). Its only remaining net is D2, and this is the
    fixture that proves D2 holds it — with the measured prose shape, not a
    synthetic prompt echo.
    """
    prompt = _base_prompt()
    anchor = _anchor(prompt=prompt)
    # Content-free pedagogical frame. Every content-bearing token is lifted
    # from the prompt itself; the only "new" words are the frame's own
    # function words and a handful of generic pedagogy nouns.
    topic = " ".join(_words(900, 8))
    candidates = [
        _candidate(
            chunk_id=f"chunk-{i:04d}",
            variant=1,
            prompt=prompt,
            completion=(
                f"Learners should be able to break this down and explain the "
                f"parts of {topic} using what they already know. Learners who "
                f"already know the parts should be able to explain what "
                f"{topic} is, and should be able to break it down. "
                f"See section {i}."
            ),
        )
        for i in range(1, 4)
    ]
    # Independence check: neither of the other two degenerate nets can fire.
    assert all(not _lo_id_tokens(c.completion) for c in candidates)
    assert len({completion_skeleton(c.completion) for c in candidates}) == 3

    rows, funnel = _select(candidates, [anchor])
    assert rows == []
    assert funnel.low_novel_content == 3
    assert funnel.lo_id_splice == 0
    assert funnel.skeleton_collapse == 0


def test_numeric_answers_do_not_share_a_skeleton():
    """D3 must NOT normalize digits — numbers are the content of a math answer.

    An earlier revision collapsed every digit run to ``0`` in the skeleton, so
    three numerically distinct worked answers of a shared prose shape hashed
    identically and the default ``max_skeleton_freq`` of 3 rejected the whole
    numeric-answer population of a math corpus under a counter an operator
    reads as "template junk correctly detected".
    """
    worked = [
        "Multiply 3 by 14 to get 42, then subtract 7 for a result of 35.",
        "Multiply 5 by 22 to get 110, then subtract 9 for a result of 101.",
        "Multiply 8 by 31 to get 248, then subtract 4 for a result of 244.",
    ]
    assert len({completion_skeleton(text) for text in worked}) == 3
    # The LO-identifier splice it DOES exist to catch still collapses.
    spliced = [
        "Explain the parts of learning outcome co-117 in your own words.",
        "Explain the parts of learning outcome co-118 in your own words.",
    ]
    assert len({completion_skeleton(text) for text in spliced}) == 1


@pytest.mark.parametrize(
    "identifier",
    ["ISO-8601", "RFC-2119", "COVID-19", "H-1B", "IPv6-4291"],
)
def test_technical_identifiers_are_not_lo_id_splices(identifier: str):
    """D1 must key on the canonical LO PREFIX taxonomy, not the bare shape.

    ``validate_lo_id`` matches ``^[A-Z]{2,}-\\d{2,}$``; matching it
    case-insensitively on arbitrary tokens turns D1 into "any 2+ letters,
    hyphen, 2+ digits", which deletes legitimate near-misses on any standards,
    CS, or medical corpus.
    """
    assert _lo_id_tokens(f"As defined in {identifier}, the rule applies.") == []
    anchor = _anchor()
    candidate = _candidate(
        variant=1,
        completion=_completion(0, 200) + f" See {identifier} for the rule.",
    )
    rows, funnel = _select([candidate], [anchor])
    assert funnel.lo_id_splice == 0
    assert len(rows) == 1


@pytest.mark.parametrize("prefix", ["CO", "TO", "MO", "PO", "UO", "LO", "SO"])
def test_every_canonical_lo_prefix_is_still_a_splice(prefix: str):
    """Every prefix in the canonical hierarchy taxonomy still trips D1."""
    assert _lo_id_tokens(f"explain outcome {prefix.lower()}-117 fully") == [
        f"{prefix.lower()}-117"
    ]
    anchor = _anchor()
    candidate = _candidate(
        variant=1,
        completion=_completion(0, 200) + f" see outcome {prefix}-117.",
    )
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.lo_id_splice == 1


# --------------------------------------------------------------------------- #
# Generation-contract staleness                                                #
# --------------------------------------------------------------------------- #


def test_unconfirmed_resume_cache_candidate_is_dropped():
    """A prior-run reject this run never re-confirmed is never emitted.

    The reject pool is seeded from the loaded resume checkpoint, which admits
    rows regardless of their per-seed contract fingerprint. On an ordinary
    resume the loader has no expected-fingerprint to filter on, so a row that
    PREDATES a model-id / generation-parameter / generation-contract-file
    change lands in the pool; if its unit is regenerated and then ACCEPTED,
    the stale candidate survives and the six identity predicates (chunk /
    lo_refs / prompt / citation flag / provider / seat) move for none of those
    changes — so nothing else would catch it, and last contract's rejected
    text would be paired against this contract's ``chosen``.
    """
    anchor = _anchor()
    stale = _raw_candidate(from_resume_cache=True, contract_confirmed=False)
    rows, funnel = _select([stale], [anchor])
    assert rows == []
    assert funnel.stale_contract == 1
    # It is dropped BEFORE stage admission, so it never enters the D3
    # skeleton-frequency universe either.
    assert funnel.stage_excluded == 0
    assert funnel.emitted == 0


def test_confirmed_resume_cache_candidate_is_admitted():
    """Confirmation is what makes a resumed run's pool usable, not discarded."""
    anchor = _anchor()
    confirmed = _raw_candidate(
        from_resume_cache=True, contract_confirmed=True,
    )
    rows, funnel = _select([confirmed], [anchor])
    assert funnel.stale_contract == 0
    assert len(rows) == 1


def test_pool_confirm_contract_marks_only_resume_cache_rows():
    """``RejectPool.confirm_contract`` is safe to call unconditionally."""
    from Trainforge.synthesis_reject_mining import RejectPool, build_capture_payload

    payload = build_capture_payload(
        {
            "chunk_id": CHUNK,
            "prompt": _base_prompt(),
            "completion": _completion(0, 200),
            "lo_refs": ["CO-01"],
            "claim_support_rate": 2.0 / 3.0,
            "claim_contradicted_rate": 0.0,
            "per_claim_support": _per_claim(),
            "instruction_variant": 1,
        },
        reason="claim_support:unsupported_claim",
        mode=MINE_MODE_ON,
    )
    assert payload is not None
    pool = RejectPool()
    pool.seed_from_checkpoint_cache({
        (CHUNK, "instruction", 1): {
            "disposition": "rejected",
            "chunk_id": CHUNK,
            "kind": "instruction",
            "variant_index": 1,
            "provider": "local",
            "seed": 11,
            "reason": "claim_support:unsupported_claim",
            "contract_fingerprint": "fp-prior",
            "pair": dict(payload),
        },
    })
    only = pool.candidates()[0]
    assert only.from_resume_cache is True
    assert only.contract_confirmed is False

    assert pool.confirm_contract((CHUNK, "instruction", 1)) is True
    assert pool.candidates()[0].contract_confirmed is True
    assert pool.counters()["resume_cache_contract_confirmed"] == 1
    # Idempotent, and a no-op for an unknown key.
    assert pool.confirm_contract((CHUNK, "instruction", 1)) is False
    assert pool.confirm_contract(("chunk-9999", "instruction", 0)) is False
    assert pool.counters()["resume_cache_contract_confirmed"] == 1


# --------------------------------------------------------------------------- #
# 4-6. Admission                                                               #
# --------------------------------------------------------------------------- #


def test_near_miss_admitted():
    anchor = _anchor()
    candidate = _candidate()
    assert 0.30 <= token_jaccard(anchor["completion"], candidate.completion) <= 0.70
    rows, funnel = _select([candidate], [anchor])
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt"] == anchor["prompt"]
    assert row["chosen"] == anchor["completion"]
    assert row["rejected"] == candidate.completion
    assert funnel.emitted == 1
    assert funnel.candidates_seen == 1


def test_contradiction_excluded():
    anchor = _anchor()
    candidate = _candidate(
        contradicted_rate=0.5, per_claim=_per_claim(contradicted=True),
    )
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.contradicted == 1


def test_deps_missing_excluded():
    anchor = _anchor()
    candidate = _candidate(support_rate=None, per_claim=[])
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.deps_missing == 1


def test_below_fail_entailment_excluded():
    anchor = _anchor()
    candidate = _candidate(per_claim=_per_claim(failing_entailment=0.03))
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.below_fail_entailment == 1


@pytest.mark.parametrize("reason", [
    "promotion:source_free_generation",
    "promotion:placeholder_residue",
    "lo_refs:missing_pair_lo_refs",
    "objective_delivery:objective_verb_absent",
    "duplicate_prompt",
])
def test_stage_admission_excludes_non_claim_support(reason):
    """Only ``claim_support:`` carries the NLI scores a near-miss needs.

    ``promotion:`` is excluded twice over: it runs BEFORE claim_support so its
    rows have no scores at all, and its leaf reasons ARE the degenerate class.
    """
    anchor = _anchor()
    candidate = _raw_candidate(reason=reason)
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.stage_excluded == 1
    assert funnel.emitted == 0


def test_stage_admission_admits_the_claim_support_twin():
    anchor = _anchor()
    rows, funnel = _select([_candidate()], [anchor])
    assert len(rows) == 1
    assert funnel.stage_excluded == 0


# --------------------------------------------------------------------------- #
# 7-8. Anchor identity                                                         #
# --------------------------------------------------------------------------- #


def test_identity_i1_different_chunk():
    anchor = _anchor(chunk_id="chunk-0002")
    rows, funnel = _select([_candidate()], [anchor])
    assert rows == []
    assert funnel.no_anchor == 1


@pytest.mark.parametrize("anchor_kw,candidate_kw", [
    ({"lo_refs": ["CO-02"]}, {}),                              # I2
    ({"prompt": "Describe an entirely different question here?"}, {}),  # I3
    ({"requires_citation": True}, {}),                         # I4
    ({}, {"provider": "together"}),                            # I5 provider
    ({"generating_seat": "seat-alpha"}, {}),                   # I5 seat
])
def test_identity_gates(anchor_kw, candidate_kw):
    anchor = _anchor(**anchor_kw)
    candidate = _candidate(**candidate_kw)
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.identity_mismatch == 1


def test_identity_i6_same_variant_rejected():
    """A reject and an anchor cannot be the same unit."""
    anchor = _anchor(variant=1)
    candidate = _candidate(variant=1)
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.identity_mismatch == 1


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_unframe_roundtrip_against_the_real_frames(variant):
    """Pins the inverse to the production frame table.

    Breaks LOUDLY if ``_INSTRUCTION_PROMPT_FRAMES`` ever changes shape.
    """
    base = _base_prompt()
    pair = {"prompt": base}
    _apply_instruction_variant(pair, variant, learner_persona=PERSONA)
    assert unframe_instruction_prompt(pair["prompt"], persona=PERSONA) == base


def test_unframe_identity_on_the_no_frame_path():
    """>360 chars: the framed candidate is discarded, so the prompt is bare."""
    long_base = "Explain " + " ".join(_words(500, 80)) + "?"
    assert len(long_base) > 360
    pair = {"prompt": long_base}
    _apply_instruction_variant(pair, 1, learner_persona=PERSONA)
    assert pair["prompt"] == long_base
    assert (
        unframe_instruction_prompt(long_base, persona=PERSONA) == long_base
    )


# --------------------------------------------------------------------------- #
# 9-11. Caps, determinism, no-anchor                                           #
# --------------------------------------------------------------------------- #


def test_cap_one_per_chunk_and_deterministic_tiebreak():
    anchor = _anchor()
    # Variants 2, 5, ... carry requires_source_citation=True, a different
    # citation contract from the variant-0 anchor, so they are skipped here to
    # keep the fixture about the tie-break rather than about I4.
    candidates = [
        _candidate(
            variant=variant,
            completion=_completion(0, 200 + 40 * i),
            support_rate=rate,
        )
        for i, (variant, rate) in enumerate(
            ((1, 0.55), (3, 0.75), (4, 0.65), (6, 0.60)),
        )
    ]
    rows, funnel = _select(candidates, [anchor])
    assert len(rows) == 1
    assert rows[0]["reject_mining"]["claim_support_rate"] == 0.75
    assert funnel.per_chunk_capped == 3


def test_tiebreak_falls_through_to_entailment_then_digest():
    anchor = _anchor()
    equal_rate = [
        _candidate(
            variant=1,
            completion=_completion(0, 200),
            per_claim=_per_claim(failing_entailment=0.20),
        ),
        _candidate(
            variant=2,
            completion=_completion(0, 240),
            requires_citation=False,
            per_claim=_per_claim(failing_entailment=0.55),
        ),
    ]
    rows, _ = _select(equal_rate, [anchor])
    assert len(rows) == 1
    assert rows[0]["reject_mining"]["min_failing_entailment"] == 0.55


def test_selection_is_order_independent():
    """Shuffling the candidate list must not change one output byte."""
    anchors = [
        _anchor(chunk_id=f"chunk-{i:04d}", prompt=_base_prompt(900 + 20 * i))
        for i in range(6)
    ]
    # Each candidate is framed from ITS OWN chunk's base question, so the
    # exact-unframe anchor rule (I3) can match.
    candidates = [
        _candidate(
            chunk_id=f"chunk-{i:04d}",
            variant=1,
            prompt=_framed(_base_prompt(900 + 20 * i), 1),
            completion=_completion(0, 200 + 40 * i),
            support_rate=0.5 + 0.01 * i,
        )
        for i in range(6)
    ]
    baseline, _ = _select(list(candidates), list(anchors))
    assert baseline, "fixture must emit something for this test to mean anything"
    rng = random.Random(1234)
    for _ in range(5):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        shuffled_anchors = list(anchors)
        rng.shuffle(shuffled_anchors)
        again, _ = _select(shuffled, shuffled_anchors)
        assert again == baseline


def _framed(base: str, variant: int) -> str:
    pair = {"prompt": base}
    _apply_instruction_variant(pair, variant, learner_persona=PERSONA)
    return str(pair["prompt"])


def test_global_fraction_cap():
    anchors = []
    candidates = []
    for i in range(40):
        chunk = f"chunk-{i:04d}"
        base = _base_prompt(2000 + 20 * i)
        anchors.append(_anchor(chunk_id=chunk, prompt=base))
        candidates.append(_candidate(
            chunk_id=chunk,
            variant=1,
            prompt=_framed(base, 1),
            completion=_completion(0, 200 + 40 * i),
            support_rate=0.50 + 0.01 * i,
        ))
    rows, funnel = _select(candidates, anchors, max_fraction=0.25)
    assert len(rows) == 10
    assert funnel.global_capped == 30
    # The survivors are the 10 highest support rates.
    kept = sorted(r["reject_mining"]["claim_support_rate"] for r in rows)
    assert kept == pytest.approx([0.50 + 0.01 * i for i in range(30, 40)])
    assert [r["reject_mining"]["rank"] for r in rows] == list(range(1, 11))


def test_no_anchor_emits_nothing_and_never_synthesizes_a_chosen():
    candidates = [
        _candidate(chunk_id=f"chunk-{i:04d}", completion=_completion(0, 200 + 40 * i))
        for i in range(4)
    ]
    rows, funnel = _select(candidates, [])
    assert rows == []
    assert funnel.no_anchor == 4
    assert funnel.emitted == 0


# --------------------------------------------------------------------------- #
# 12. Emitted shape + REAL schema validation                                   #
# --------------------------------------------------------------------------- #


def _real_schema_validator():
    knowledge = Path("schemas/knowledge").resolve()
    schemas = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(knowledge.glob("*.schema.json"))
    ]
    resources = [
        (schema["$id"], Resource.from_contents(
            schema, default_specification=DRAFT202012,
        ))
        for schema in schemas
        if isinstance(schema.get("$id"), str)
    ]
    root = next(
        schema for schema in schemas
        if schema.get("$id")
        == "https://ed4all.dev/ns/knowledge/v1/preference_pair.schema.json"
    )
    return Draft202012Validator(root, registry=Registry().with_resources(resources))


def test_emitted_shape():
    anchor = _anchor()
    candidate = _candidate()
    rows, _ = _select([candidate], [anchor])
    row = rows[0]

    for key in ("prompt", "chosen", "rejected", "chunk_id", "lo_refs", "seed",
                "decision_capture_id"):
        assert key in row, key
    assert isinstance(row["seed"], int)
    assert row["decision_capture_id"]
    assert row["lo_refs"] == ["CO-01"]
    # Non-empty promotion_status is the critical ``pair_promotion`` contract.
    assert row["promotion_status"] == "validated"
    assert row["source"] == MINED_PAIR_SOURCE
    assert row["source_chunk_id"] == CHUNK
    assert "rejected_source" not in row
    assert row["misconception_id"] is None
    assert 40 <= len(row["prompt"]) <= 400
    assert 50 <= len(row["chosen"]) <= 1200
    assert 50 <= len(row["rejected"]) <= 1200

    block = row["reject_mining"]
    assert block["rejection_reason"].startswith("claim_support:")
    assert block["claim_contradicted_rate"] == 0.0
    assert block["anchor_variant_index"] == 0
    assert block["rejected_variant_index"] == 1
    assert block["n_claims"] == 3
    assert block["rank"] == 1
    assert block["contract_fingerprint"] == "fp-current"
    assert 0.30 <= block["completion_jaccard"] <= 0.70


def test_emitted_row_validates_against_the_real_preference_pair_schema():
    """Validated against ``schemas/knowledge/`` itself, not a copy."""
    validator = _real_schema_validator()
    anchor = _anchor()
    rows, _ = _select([_candidate()], [anchor])
    errors = sorted(validator.iter_errors(rows[0]), key=lambda e: e.path)
    assert errors == [], [e.message for e in errors]


def test_length_out_of_band_rejected():
    anchor = _anchor(completion="short.")
    rows, funnel = _select([_candidate()], [anchor])
    assert rows == []
    assert funnel.length_out_of_band == 1


def test_jaccard_out_of_band_rejected():
    """A negative from a different lexical universe teaches separability."""
    anchor = _anchor(completion=_completion(0, 100))
    candidate = _candidate(completion=_completion(600, 700))
    assert token_jaccard(anchor["completion"], candidate.completion) < 0.30
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.jaccard_out_of_band == 1


# --------------------------------------------------------------------------- #
# 13-15. Shadow, licensing, decision capture                                   #
# --------------------------------------------------------------------------- #


def test_shadow_emits_nothing_but_measures_the_funnel():
    anchor = _anchor()
    rows, funnel = _select([_candidate()], [anchor], mode=MINE_MODE_SHADOW)
    assert rows == []
    # The candidate was SELECTED, not suppressed — that distinction is the
    # entire point of shadow mode.
    assert funnel.emitted == 1


def test_off_mode_does_nothing():
    anchor = _anchor()
    rows, funnel = _select([_candidate()], [anchor], mode=MINE_MODE_OFF)
    assert rows == []
    assert funnel.as_dict() == {k: 0 for k in funnel.as_dict()}


def test_unregistered_teacher_dropped():
    from lib.licensing.teacher_roster import PROVENANCE_VERDICTS, license_for_model

    seat = "unregistered-seat-for-test"
    assert seat not in PROVENANCE_VERDICTS
    assert license_for_model(seat) is None
    anchor = _anchor(generating_seat=seat)
    candidate = _candidate(generating_seat=seat)
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.unregistered_teacher == 1


def test_barred_teacher_dropped():
    """A barred provider must never ride a mined row into the corpus.

    ``assert_export_licenses`` fail-closes the whole training run on one, and a
    mined row must never be the thing that bricks a multi-hour build.
    """
    anchor = _anchor(provider="claude_session")
    candidate = _candidate(provider="claude_session")
    rows, funnel = _select([candidate], [anchor])
    assert rows == []
    assert funnel.unregistered_teacher == 1


def test_licensing_stamped_on_emitted_row():
    anchor = _anchor()
    rows, _ = _select([_candidate()], [anchor])
    assert rows[0]["license"] == "local/provenance/safe"


def test_duplicate_prompt_respects_the_shared_emitted_set():
    anchor = _anchor()
    seen = {anchor["prompt"]}
    rows, funnel = _select(
        [_candidate()], [anchor], emitted_pref_prompts=seen,
    )
    assert rows == []
    assert funnel.duplicate_prompt == 1


def test_emitted_prompt_is_added_to_the_shared_set():
    anchor = _anchor()
    seen: set = set()
    rows, _ = _select([_candidate()], [anchor], emitted_pref_prompts=seen)
    assert len(rows) == 1
    assert anchor["prompt"] in seen


def test_decision_capture_rationale():
    capture = _RecordingCapture()
    anchor = _anchor()
    rows, _ = _select([_candidate()], [anchor], capture=capture)
    assert len(rows) == 1
    events = [
        e for e in capture.decisions
        if e["decision_type"] == SELECTION_DECISION_TYPE
    ]
    # One event per emitted row + one pass summary.
    assert len(events) == 2
    per_row = events[0]
    assert len(per_row["rationale"]) >= 20
    # Real per-unit signals, not a static boilerplate rationale.
    for signal in (
        CHUNK, "claim_support_rate=", "min_failing_entailment=",
        "anchor_variant=", "rank=", "skeleton_frequency=",
        "completion_jaccard=", "n_claims=",
    ):
        assert signal in per_row["rationale"], signal
    assert rows[0]["decision_capture_id"] == per_row["event_id"]

    summary = events[1]
    assert len(summary["rationale"]) >= 20
    for signal in ("candidates_seen=", "no_anchor=", "emitted=", "min_support="):
        assert signal in summary["rationale"], signal


def test_shadow_still_captures_the_summary():
    capture = _RecordingCapture()
    anchor = _anchor()
    _select([_candidate()], [anchor], mode=MINE_MODE_SHADOW, capture=capture)
    events = [
        e for e in capture.decisions
        if e["decision_type"] == SELECTION_DECISION_TYPE
    ]
    assert len(events) == 1
    assert "mode=shadow" in events[0]["context"]


def test_deterministic_generator_rows_are_never_anchors():
    """Their chunk_id resolves against no chunk and they carry no variant."""
    fake = {
        "chunk_id": CHUNK,
        "prompt": _base_prompt(),
        "completion": _completion(0, 100),
        "lo_refs": ["CO-01"],
        "provider": "local",
        "template_id": "kg_metadata.membership",
    }
    rows, funnel = _select([_candidate()], [fake])
    assert rows == []
    assert funnel.no_anchor == 1


# --------------------------------------------------------------------------- #
# Wiring: the hook actually runs inside run_synthesis                          #
# --------------------------------------------------------------------------- #


_FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)


def _working_copy(tmp_path: Path, name: str) -> Path:
    import shutil

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


@pytest.mark.parametrize("mode", ["shadow", "on"])
def test_selection_pass_runs_inside_run_synthesis(tmp_path, monkeypatch, mode):
    """The selection funnel reaches ``stats.reject_mining`` on a real run.

    This is the only test that executes the mode-gated hook in
    ``run_synthesis`` (persona resolution, satellite resolution, the funnel
    roll-up, the append loop). A NameError in any of them is invisible to
    every flag-off test in the tree.
    """
    from Trainforge.synthesize_training import run_synthesis

    monkeypatch.setenv(MINE_REJECTS_ENV, mode)
    corpus = _working_copy(tmp_path, f"wired-{mode}")
    stats = run_synthesis(
        corpus_dir=corpus,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )
    # Capture counters (pool) AND selection counters (funnel) are both present.
    assert "pool_size" in stats.reject_mining
    assert "candidates_seen" in stats.reject_mining
    assert "no_anchor" in stats.reject_mining
    assert stats.reject_mining["mode_shadow"] == int(mode == "shadow")


def test_flag_off_run_reports_no_funnel(tmp_path, monkeypatch):
    from Trainforge.synthesize_training import run_synthesis

    monkeypatch.delenv(MINE_REJECTS_ENV, raising=False)
    corpus = _working_copy(tmp_path, "wired-off")
    stats = run_synthesis(
        corpus_dir=corpus,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
    )
    assert stats.reject_mining == {}
