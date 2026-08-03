"""Capture layer for reject-mined DPO negatives.

WHAT THIS MODULE IS
===================

Training-pair synthesis rejects far more instruction pairs than it accepts
(measured acceptance across three archived cohorts: 0/52, 2/52, 4/52). Every
rejected pair is a candidate DPO ``rejected`` side that has already been paid
for in GPU time: same prompt, same chunk, same objective. Turning one into a
preference row needs the *full* rejected completion.

That text is currently discarded. The rejection evidence persisted by
``lib.validators.pair.claim_support.summarize_claim_support_rejection`` is
capped at six failing sentences of 400 characters — it is adjudication
evidence, deliberately bounded, and it is NOT a completion.

This module owns the capture half of the feature only:

- the master flag (:func:`resolve_mine_rejects_mode`),
- the admission + length bands that decide whether a rejected pair is worth
  persisting at all (:func:`build_capture_payload`),
- the in-memory pool (:class:`RejectPool`) that a later selection pass reads,
  seeded from the resume checkpoint so a killed-and-resumed run keeps a
  COMPLETE reject pool.

The selection rule (which rejects become preference rows, the near-miss score
band, the anti-degenerate filters, the anchor identity match, the caps) is a
separate, later pass. Nothing here emits a training pair.

WHY THERE IS NO NEW SIDECAR FILE
================================

The persisted store is the EXISTING per-pair resume checkpoint
(``.synthesis_pairs_checkpoint.jsonl``). A ``rejected`` row there already
carries ``chunk_id`` / ``kind`` / ``variant_index`` / ``provider`` / ``seed`` /
``reason`` / ``contract_fingerprint`` plus the fresh-start identity block, and
its ``pair`` field has simply always been ``null``. Populating it inherits, at
zero new surface area:

- fresh-start identity binding + the run-contract / holdout identity guards
  (``_load_synthesis_pairs_checkpoint``),
- the terminal-uniqueness raise and the ``FreshStartError`` contract,
- the write-deferral invariant (buffered in a ``StringIO`` until every identity
  check has passed),
- unlink-on-clean-exit / preserve-on-every-other-exit,
- ``SYNTHESIS_ARTIFACT_NAMES`` registration,
- and decisively, RESUME COMPLETENESS: a rejected unit is *skipped* on resume,
  never regenerated, so without reading the prior run's rows the pool would be
  silently partial — and a partial pool is a BIASED pool.

A dedicated sidecar would have to re-implement all six.

CAPS
====

Three, all structural, none a count cap:

1. stage admission — only ``claim_support:`` rejections carry the per-claim NLI
   scores a near-miss judgement needs;
2. length bands — the ``preference_pair.schema.json`` bounds, applied at
   capture time so an unusable completion is never written;
3. one row per ``(chunk_id, kind, variant_index)`` terminal key, enforced by
   the checkpoint writer that already owns that invariant.

There is deliberately NO cap on the number of captured rejects. An artificial
ceiling would truncate the pool in generation order and silently bias it toward
early chunks; the file is unlinked on a clean exit anyway.

FLAG
====

``TRAINFORGE_DPO_MINE_REJECTS`` — three-valued ``off`` (default) / ``shadow`` /
``on``, parse-with-fallback. ``off`` means no pair is stored on any rejected
row and no pool is built: byte-identical to the legacy path.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from lib.licensing.teacher_roster import (
    classify_pair_teacher,
    provenance_license_tag,
    stamp_pair_license,
)
from lib.ontology.learning_objectives import hierarchy_from_id, validate_lo_id

__all__ = [
    "MINE_REJECTS_ENV",
    "MINE_MODE_OFF",
    "MINE_MODE_SHADOW",
    "MINE_MODE_ON",
    "REJECT_MINING_SCHEMA_VERSION",
    "MINEABLE_KIND",
    "MINEABLE_REASON_PREFIX",
    "MIN_CAPTURED_PROMPT_CHARS",
    "MAX_CAPTURED_PROMPT_CHARS",
    "MIN_CAPTURED_COMPLETION_CHARS",
    "MAX_CAPTURED_COMPLETION_CHARS",
    "MINED_PAIR_SOURCE",
    "MIN_SUPPORT_ENV",
    "MIN_FAIL_ENTAILMENT_ENV",
    "MAX_SKELETON_FREQ_ENV",
    "MAX_FRACTION_ENV",
    "DEFAULT_MIN_SUPPORT",
    "DEFAULT_MIN_FAIL_ENTAILMENT",
    "DEFAULT_MAX_SKELETON_FREQ",
    "DEFAULT_MAX_FRACTION",
    "MIN_SCORED_CLAIMS",
    "MIN_NOVEL_CONTENT_TOKENS",
    "MIN_COMPLETION_JACCARD",
    "MAX_COMPLETION_JACCARD",
    "SELECTION_DECISION_TYPE",
    "resolve_mine_rejects_mode",
    "resolve_min_support",
    "resolve_min_fail_entailment",
    "resolve_max_skeleton_freq",
    "resolve_max_fraction",
    "is_mineable_reason",
    "build_capture_payload",
    "unframe_instruction_prompt",
    "completion_skeleton",
    "token_jaccard",
    "RejectCandidate",
    "RejectPool",
    "MinedFunnel",
    "select_mined_pairs",
    "iter_mineable_records",
]


#: Version of the reject-mining capture contract. Bumping this does NOT bump
#: ``_SYNTHESIS_CHECKPOINT_SCHEMA_VERSION``: ``pair`` is an existing key whose
#: value was previously always ``null`` on a rejected row, and every consumer
#: of that file dispatches on ``disposition`` before reading ``pair``.
REJECT_MINING_SCHEMA_VERSION = "v1"

MINE_REJECTS_ENV = "TRAINFORGE_DPO_MINE_REJECTS"

MINE_MODE_OFF = "off"
MINE_MODE_SHADOW = "shadow"
MINE_MODE_ON = "on"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Only instruction units are mined. A preference unit's rejected side is a
#: different contract and is explicitly out of scope for v1.
MINEABLE_KIND = "instruction"

#: Only ``claim_support`` rejections are captured. They are the only stage that
#: stamps ``per_claim_support`` / ``claim_support_rate`` /
#: ``claim_contradicted_rate`` onto the pair, so they are the only stage whose
#: rejects can be scored for near-missness without a fresh model call. The
#: ``promotion:`` stage is excluded twice over: it runs BEFORE claim_support so
#: its rows carry no NLI scores at all, and its leaf reasons
#: (``placeholder_residue`` / ``source_free_generation`` / ``unanswerable_stem``)
#: ARE the degenerate template-collapse class that must never become a DPO
#: negative.
MINEABLE_REASON_PREFIX = "claim_support:"

# Length bands, from ``schemas/knowledge/preference_pair.schema.json``. Applied
# at CAPTURE time so a completion that could never legally become a preference
# row is not written to the checkpoint at all.
MIN_CAPTURED_PROMPT_CHARS = 40
MAX_CAPTURED_PROMPT_CHARS = 400
MIN_CAPTURED_COMPLETION_CHARS = 50
MAX_CAPTURED_COMPLETION_CHARS = 1200

#: Value stamped on ``pair["source"]`` for every mined row. Read by
#: ``Trainforge/training/compute_backend.py::_filter_dpo_pairs`` and
#: ``Trainforge/training/runner.py::_count_dpo_eligible_records`` (which MUST
#: be widened in lockstep — extending one alone means the count says "run DPO"
#: while the filter drops everything, and ``dpo_fail_hard=True`` then kills the
#: run AFTER SFT already succeeded).
MINED_PAIR_SOURCE = "mined_rejection"

#: Decision type emitted per selected row + once per pass summary.
SELECTION_DECISION_TYPE = "reject_mined_preference_selection"

# --- selection satellites --------------------------------------------------
MIN_SUPPORT_ENV = "TRAINFORGE_DPO_MINE_MIN_SUPPORT"
MIN_FAIL_ENTAILMENT_ENV = "TRAINFORGE_DPO_MINE_MIN_FAIL_ENTAILMENT"
MAX_SKELETON_FREQ_ENV = "TRAINFORGE_DPO_MINE_MAX_SKELETON_FREQ"
MAX_FRACTION_ENV = "TRAINFORGE_DPO_MINE_MAX_FRACTION"

#: Floor on ``claim_support_rate``. The measured template-collapse population
#: scored ~0.0 (its single content claim entailed at 0.028), so this encodes
#: "the MAJORITY of this completion's claims were supported and a minority
#: drifted" — the definition of a near-miss. UNCALIBRATED: it brackets exactly
#: one measured value. Measure with ``shadow`` before moving it.
DEFAULT_MIN_SUPPORT = 0.50

#: Floor on the WORST failing claim's entailment. An order of magnitude above
#: the measured degenerate value (0.028) and well below the gate's own floor,
#: so it provably removes the measured degenerate population without asserting
#: where "near" begins. UNCALIBRATED for the same reason.
DEFAULT_MIN_FAIL_ENTAILMENT = 0.15

#: A normalized-completion skeleton recurring at least this many times across
#: the run's candidate pool is template collapse. Repetition IS the failure, so
#: no per-item threshold can replace this pool-wide check.
DEFAULT_MAX_SKELETON_FREQ = 3

#: Ceiling on mined rows as a fraction of the accepted SFT corpus. Bounds the
#: FLAME failure mode (naive DPO measured 44.7 -> 42.3 FActScore when the
#: preference signal is not factuality-balanced).
DEFAULT_MAX_FRACTION = 0.25

#: Fewer scored claims than this carries no ratio signal at all.
MIN_SCORED_CLAIMS = 3

#: D2 — distinct content tokens the completion must carry that its own prompt
#: does not. The degenerate form is a fixed pedagogical frame plus the prompt's
#: own topic slot, so it cannot clear a novel-content floor.
MIN_NOVEL_CONTENT_TOKENS = 25

#: Lower bound restates the HA-DPO minimal-difference discipline: a negative
#: drawn from a different lexical universe teaches SEPARABILITY, not quality.
#: Deliberately satellite-free so it cannot be tuned into meaninglessness
#: without a code change.
MIN_COMPLETION_JACCARD = 0.30

#: Upper bound is the ``preference_pair.schema.json`` constraint restated
#: ("must differ from chosen with token-Jaccard delta >= 0.3").
MAX_COMPLETION_JACCARD = 0.70


def resolve_mine_rejects_mode(
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve ``TRAINFORGE_DPO_MINE_REJECTS`` (parse-with-fallback, OFF).

    ``"shadow"`` -> pool and count, emit nothing.
    Truthy (``1``/``true``/``yes``/``on``, case-insensitive) -> ``"on"``.
    Everything else — unset, empty, falsey, garbage -> ``"off"``.

    Read at call time, never at import, so a test can drive it with an
    injected mapping and a process never caches a stale value.
    """
    source = os.environ if env is None else env
    raw = str(source.get(MINE_REJECTS_ENV, "") or "").strip().lower()
    if raw == MINE_MODE_SHADOW:
        return MINE_MODE_SHADOW
    return MINE_MODE_ON if raw in _TRUTHY else MINE_MODE_OFF


def _resolve_float(
    env: Optional[Mapping[str, str]],
    name: str,
    default: float,
    *,
    low: float,
    high: float,
    include_low: bool = True,
    include_high: bool = False,
) -> float:
    """Parse-with-fallback float resolver shared by the selection satellites.

    Out-of-band or unparseable values fall back to ``default`` — a partly
    garbage override never yields a half-configured band.
    """
    source = os.environ if env is None else env
    raw = str(source.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    if value < low or (not include_low and value == low):
        return default
    if value > high or (not include_high and value == high):
        return default
    return value


def resolve_min_support(env: Optional[Mapping[str, str]] = None) -> float:
    """Resolve ``TRAINFORGE_DPO_MINE_MIN_SUPPORT`` (float in ``[0.0, 1.0)``)."""
    return _resolve_float(
        env, MIN_SUPPORT_ENV, DEFAULT_MIN_SUPPORT,
        low=0.0, high=1.0, include_low=True, include_high=False,
    )


def resolve_min_fail_entailment(
    env: Optional[Mapping[str, str]] = None,
) -> float:
    """Resolve ``TRAINFORGE_DPO_MINE_MIN_FAIL_ENTAILMENT`` (``[0.0, 1.0)``)."""
    return _resolve_float(
        env, MIN_FAIL_ENTAILMENT_ENV, DEFAULT_MIN_FAIL_ENTAILMENT,
        low=0.0, high=1.0, include_low=True, include_high=False,
    )


def resolve_max_skeleton_freq(env: Optional[Mapping[str, str]] = None) -> int:
    """Resolve ``TRAINFORGE_DPO_MINE_MAX_SKELETON_FREQ`` (positive int)."""
    source = os.environ if env is None else env
    raw = str(source.get(MAX_SKELETON_FREQ_ENV, "") or "").strip()
    if not raw:
        return DEFAULT_MAX_SKELETON_FREQ
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_SKELETON_FREQ
    return value if value > 0 else DEFAULT_MAX_SKELETON_FREQ


def resolve_max_fraction(env: Optional[Mapping[str, str]] = None) -> float:
    """Resolve ``TRAINFORGE_DPO_MINE_MAX_FRACTION`` (float in ``(0.0, 1.0]``)."""
    return _resolve_float(
        env, MAX_FRACTION_ENV, DEFAULT_MAX_FRACTION,
        low=0.0, high=1.0, include_low=False, include_high=True,
    )


def is_mineable_reason(reason: Any) -> bool:
    """Return whether a terminal rejection reason is capture-eligible."""
    return str(reason or "").startswith(MINEABLE_REASON_PREFIX)


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_capture_payload(
    pair: Any,
    *,
    reason: Any,
    mode: str,
    kind: str = MINEABLE_KIND,
) -> Optional[Dict[str, Any]]:
    """Return the pair dict to persist on a rejected row, or ``None``.

    ``None`` — the caller writes ``pair=None``, i.e. exactly today's row — for
    every one of:

    - mining is ``off`` (the default): nothing is ever captured;
    - the unit is not an instruction unit;
    - the rejection did not come from the ``claim_support`` stage;
    - there is no usable prompt/completion text;
    - either side falls outside the ``preference_pair`` length band.

    ``shadow`` DOES capture. Shadow's whole purpose is to measure the funnel on
    real data before any row is emitted, and it cannot measure what was never
    written.

    Private execution sidecars (leading-underscore keys) are stripped, exactly
    as the accepted-pair checkpoint write does by popping them before the
    append. The result is a shallow copy: the caller has already abandoned the
    pair by the time this returns (every rejection site ``continue``s), so no
    nested value is subsequently mutated.
    """
    if mode == MINE_MODE_OFF:
        return None
    if str(kind) != MINEABLE_KIND:
        return None
    if not is_mineable_reason(reason):
        return None
    if not isinstance(pair, Mapping):
        return None
    prompt = _clean_text(pair.get("prompt"))
    completion = _clean_text(pair.get("completion"))
    if not prompt or not completion:
        return None
    if not (
        MIN_CAPTURED_PROMPT_CHARS <= len(prompt) <= MAX_CAPTURED_PROMPT_CHARS
    ):
        return None
    if not (
        MIN_CAPTURED_COMPLETION_CHARS
        <= len(completion)
        <= MAX_CAPTURED_COMPLETION_CHARS
    ):
        return None
    return {
        str(key): value
        for key, value in pair.items()
        if not str(key).startswith("_")
    }


@dataclass(frozen=True)
class RejectCandidate:
    """One rejected instruction unit with its full completion and NLI scores.

    Everything a selection pass needs to decide near-missness and to bind the
    negative to an accepted anchor, read off already-persisted values. No
    embedding, no model call, no re-derivation.

    ``contract_fingerprint`` rides along because a candidate loaded from a
    prior run may predate the current generation contract; the selection pass
    decides what to do with that, capture does not silently drop it.

    ``contract_confirmed`` is that decision, made by the ONLY code that can
    make it. The per-seed rejection fingerprint is a function of the seed and
    the focused chunk, so this module cannot recompute it; the run loop can,
    and does, at the point where it either replays a cached rejection (the
    fingerprint matched -> the candidate belongs to the CURRENT contract) or
    regenerates the unit (it did not). A live capture from this run is
    confirmed by construction. An UNCONFIRMED resume-cache candidate is
    dropped by :func:`select_mined_pairs` (counter ``stale_contract``) rather
    than paired against a current-contract ``chosen``: the six identity
    predicates compare chunk / lo_refs / prompt / citation flag / provider /
    seat, NONE of which moves when only the model id, a generation parameter,
    or a generation-contract file changed, so nothing else would catch it.
    Fail-closed on purpose — an unvisited chunk's candidate is unverifiable,
    and an unverifiable negative is worth less than no negative.
    """

    chunk_id: str
    variant_index: int
    reason: str
    provider: str
    seed: Optional[int]
    contract_fingerprint: Optional[str]
    generating_seat: Optional[str]
    prompt: str
    completion: str
    lo_refs: Tuple[str, ...]
    requires_source_citation: bool
    claim_support_rate: Optional[float]
    claim_contradicted_rate: Optional[float]
    per_claim_support: Optional[Tuple[Mapping[str, Any], ...]]
    from_resume_cache: bool = False
    contract_confirmed: bool = False

    @property
    def key(self) -> Tuple[str, str, int]:
        """The terminal semantic key this candidate came from."""
        return (self.chunk_id, MINEABLE_KIND, self.variant_index)

    @classmethod
    def from_checkpoint_record(
        cls,
        record: Mapping[str, Any],
        *,
        from_resume_cache: bool = False,
    ) -> Optional["RejectCandidate"]:
        """Build a candidate from one checkpoint row, or ``None``.

        This is the READER. It is the same code path for a row written earlier
        in this process and for a row loaded out of a prior run's checkpoint,
        so a resumed run's pool is indistinguishable from an uninterrupted
        one's.

        Returns ``None`` for anything that is not a mineable rejected
        instruction row carrying a length-in-band pair — including every legacy
        row written before capture existed, which has ``pair: null``.
        """
        if not isinstance(record, Mapping):
            return None
        if record.get("disposition") != "rejected":
            return None
        if str(record.get("kind") or "") != MINEABLE_KIND:
            return None
        reason = record.get("reason")
        if not is_mineable_reason(reason):
            return None
        pair = record.get("pair")
        if not isinstance(pair, Mapping) or not pair:
            return None
        prompt = _clean_text(pair.get("prompt"))
        completion = _clean_text(pair.get("completion"))
        if not prompt or not completion:
            return None
        chunk_id = str(record.get("chunk_id") or "").strip()
        if not chunk_id:
            return None
        try:
            variant_index = int(record.get("variant_index") or 0)
        except (TypeError, ValueError):
            return None
        seed = record.get("seed")
        try:
            seed_value = None if seed is None else int(seed)
        except (TypeError, ValueError):
            seed_value = None
        fingerprint = record.get("contract_fingerprint")
        per_claim = pair.get("per_claim_support")
        per_claim_tuple: Optional[Tuple[Mapping[str, Any], ...]]
        if isinstance(per_claim, Sequence) and not isinstance(per_claim, str):
            per_claim_tuple = tuple(
                entry for entry in per_claim if isinstance(entry, Mapping)
            )
        else:
            per_claim_tuple = None
        return cls(
            chunk_id=chunk_id,
            variant_index=variant_index,
            reason=str(reason),
            provider=str(record.get("provider") or ""),
            seed=seed_value,
            contract_fingerprint=(
                fingerprint if isinstance(fingerprint, str) else None
            ),
            generating_seat=(
                str(pair.get("generating_seat"))
                if pair.get("generating_seat")
                else None
            ),
            prompt=prompt,
            completion=completion,
            lo_refs=tuple(
                str(ref).strip()
                for ref in (pair.get("lo_refs") or [])
                if str(ref).strip()
            ),
            requires_source_citation=bool(
                pair.get("requires_source_citation")
            ),
            claim_support_rate=_as_float(pair.get("claim_support_rate")),
            claim_contradicted_rate=_as_float(
                pair.get("claim_contradicted_rate")
            ),
            per_claim_support=per_claim_tuple,
            from_resume_cache=bool(from_resume_cache),
            # A candidate captured live in THIS run was generated under the
            # current contract by construction. One loaded from a prior run's
            # checkpoint has not been verified yet -- see ``contract_confirmed``.
            contract_confirmed=not bool(from_resume_cache),
        )


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RejectPool:
    """Thread-safe, in-memory pool of mineable rejected instruction units.

    Keyed by the terminal semantic key ``(chunk_id, "instruction",
    variant_index)`` — the same key the resume checkpoint enforces uniqueness
    on — so a resumed run cannot double-count a unit it inherited from the
    prior run's file and also re-observed this run.

    Live capture WINS over a resume-cache row for the same key. It cannot
    happen for a contract-compatible rejection (that unit is skipped on
    resume, never regenerated), but it CAN happen when the cached row's
    contract fingerprint no longer matches and the loop regenerates the unit.
    In that case the freshly generated completion is the one that belongs to
    the current contract.

    Mutated from the source-order writer, and guarded by its own lock anyway
    so a future concurrent writer cannot interleave a partial update.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._candidates: Dict[Tuple[str, str, int], RejectCandidate] = {}
        self._counters: Dict[str, int] = {
            "resume_cache_rows_scanned": 0,
            "resume_cache_candidates": 0,
            "captured_this_run": 0,
            "superseded_resume_cache": 0,
            "resume_cache_contract_confirmed": 0,
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._candidates)

    def seed_from_checkpoint_cache(
        self,
        cache: Optional[Mapping[Tuple[str, str, int], Mapping[str, Any]]],
    ) -> int:
        """Ingest every mineable rejected row from a loaded resume cache.

        Returns the number of candidates admitted. The rows have already
        passed the loader's schema-version, fresh-start identity, holdout
        identity, run-contract and duplicate-key checks — this method does not
        re-implement any of them, which is the whole reason the pool lives on
        the pair checkpoint rather than in a sidecar of its own.
        """
        if not cache:
            return 0
        admitted = 0
        for record in cache.values():
            with self._lock:
                self._counters["resume_cache_rows_scanned"] += 1
            candidate = RejectCandidate.from_checkpoint_record(
                record, from_resume_cache=True,
            )
            if candidate is None:
                continue
            with self._lock:
                self._candidates[candidate.key] = candidate
                self._counters["resume_cache_candidates"] += 1
            admitted += 1
        return admitted

    def confirm_contract(self, key: Tuple[str, str, int]) -> bool:
        """Mark a pooled resume-cache candidate as CURRENT-contract.

        Called by the run loop at the one place that can decide it: the branch
        where a cached rejected row's per-seed contract fingerprint matched the
        fingerprint recomputed for this run, so the row is replayed as a
        rejection rather than regenerated. This module cannot recompute that
        fingerprint (it is a function of the seed and the focused chunk), which
        is exactly why confirmation is pushed out to the caller.

        Returns ``True`` when a pooled resume-cache candidate was confirmed.
        Anything else — no such key, or a key already superseded by a live
        capture (already confirmed by construction) — is a no-op returning
        ``False``, so a caller can invoke it unconditionally.
        """
        with self._lock:
            candidate = self._candidates.get(key)
            if candidate is None or candidate.contract_confirmed:
                return False
            self._candidates[key] = replace(candidate, contract_confirmed=True)
            self._counters["resume_cache_contract_confirmed"] += 1
            return True

    def record_rejection(
        self,
        *,
        chunk_id: Any,
        kind: str,
        variant_index: Any,
        pair: Optional[Mapping[str, Any]],
        provider: Any,
        seed: Any,
        reason: Any,
        contract_fingerprint: Any,
    ) -> Optional[RejectCandidate]:
        """Ingest one rejection observed in THIS run.

        ``pair`` is the payload :func:`build_capture_payload` returned — i.e.
        exactly the dict persisted on the checkpoint row, so the pool and the
        file can never disagree. ``None`` (mining off, wrong stage, out of
        band) is a no-op returning ``None``.
        """
        if pair is None:
            return None
        candidate = RejectCandidate.from_checkpoint_record({
            "disposition": "rejected",
            "chunk_id": chunk_id,
            "kind": kind,
            "variant_index": variant_index,
            "provider": provider,
            "seed": seed,
            "reason": reason,
            "contract_fingerprint": contract_fingerprint,
            "pair": pair,
        })
        if candidate is None:
            return None
        with self._lock:
            prior = self._candidates.get(candidate.key)
            if prior is not None and prior.from_resume_cache:
                self._counters["superseded_resume_cache"] += 1
            self._candidates[candidate.key] = candidate
            self._counters["captured_this_run"] += 1
        return candidate

    def candidates(self) -> List[RejectCandidate]:
        """Return every pooled candidate in a total, stable order.

        Sorted by the terminal semantic key so downstream selection is
        deterministic regardless of the order units completed in — which, with
        concurrent generation, is not reproducible.
        """
        with self._lock:
            values = list(self._candidates.values())
        return sorted(values, key=lambda c: (c.chunk_id, c.variant_index))

    def counters(self) -> Dict[str, int]:
        """Return a snapshot of the capture counters plus the pool size."""
        with self._lock:
            snapshot = dict(self._counters)
            snapshot["pool_size"] = len(self._candidates)
        return snapshot


def iter_mineable_records(
    records: Iterable[Mapping[str, Any]],
) -> List[RejectCandidate]:
    """Convenience reader over raw checkpoint rows (postmortem / tests)."""
    out: List[RejectCandidate] = []
    for record in records:
        candidate = RejectCandidate.from_checkpoint_record(record)
        if candidate is not None:
            out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# SELECTION
# ---------------------------------------------------------------------------
#
# Everything below is a PURE function of already-persisted values: the rejected
# completion, its per-claim NLI verdict, and the accepted anchor pair. No
# embedding, no NLI, no model call, no network, no clock, no RNG. Same input ->
# same output, byte for byte, which is the whole reason the miner can be
# unit-tested without a seat.
#
# THE TRAP THIS EXISTS TO AVOID
# =============================
#
# Most current rejects are DEGENERATE: a fixed pedagogical tail with a database
# key spliced into the topic slot ("...break this down and explain the parts of
# learning outcome co-117", entailment 0.028). On 20 of 26 audited chunks every
# content source was empty, so those are CORRECT rejections. They are also
# terrible DPO negatives — trivially separable, so the preference teaches
# DISTRIBUTION discrimination ("template junk vs real prose") instead of QUALITY
# discrimination ("grounded vs subtly unsupported"). FLAME (arXiv:2405.01525)
# measured naive DPO REDUCING factuality 44.7 -> 42.3 FActScore exactly this
# way. The valuable negative is the NEAR-MISS: fluent, on-topic, plausible
# prose that drifts just past what the source supports.
#
# Degenerates are excluded FOUR independent ways, every one of which fires on
# the measured population:
#
#   1. stage admission     — the ``promotion:`` stage, whose leaf reasons ARE
#                            that class, is never mined (and carries no NLI
#                            scores anyway);
#   2. the score band      — ``claim_support_rate >= 0.50`` and worst failing
#                            entailment ``>= 0.15``; the measured degenerates
#                            sit at ~0.0 and 0.028;
#   3. D1 LO-id splice     — an LO identifier token in the completion is the
#                            exact measured signature;
#   4. D3 skeleton collapse— a normalized completion recurring across the pool
#                            IS template collapse by definition.
#
# D2 (novel-content floor) is a fifth, weaker net for the same class.


def unframe_instruction_prompt(prompt: Any, *, persona: str) -> str:
    """Invert ``_apply_instruction_variant``'s three prompt frames.

    ``synthesize_training._INSTRUCTION_PROMPT_FRAMES`` applies one of three
    PURE wrappers to a single base prompt::

        "{prompt}"
        "For {persona}, {prompt_lc}"
        "Give a source-grounded answer: {prompt}"

    so recovering the base is exact and provable, and two variants of the same
    question compare EQUAL. That is why the anchor rule (I3) is exact equality
    rather than an embedding cosine: a 0.80 cosine gate admits genuinely
    different questions that merely share a frame and an objective noun phrase,
    and the resulting negative teaches "don't answer question B when asked
    question A" — precisely the distribution-discrimination trap above.

    Anything that carries no recognised frame (including the ``len > 360``
    no-frame path, where the framed candidate was discarded) is returned
    unchanged. The citation instruction that ``_attach_source_grounding``
    appends is NOT stripped: I4 holds ``requires_source_citation`` equal across
    the two sides, so that suffix is identical on both and survives
    prefix-stripping.
    """
    text = prompt if isinstance(prompt, str) else ""
    citation_frame = "Give a source-grounded answer: "
    if text.startswith(citation_frame):
        return text[len(citation_frame):]
    persona_frame = f"For {persona}, "
    if persona and text.startswith(persona_frame):
        rest = text[len(persona_frame):]
        # ``prompt_lc`` lowered exactly the first character; restore it.
        return rest[:1].upper() + rest[1:] if rest else rest
    return text


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s+")


def _word_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def _is_lo_id_token(token: str) -> bool:
    """Whether ``token`` is a canonical learning-objective identifier.

    Delegates to the canonical helpers in ``lib.ontology.learning_objectives``
    rather than re-deriving a regex. TWO checks, both required:

    1. :func:`validate_lo_id` — the canonical ``^[A-Z]{2,}-\\d{2,}$`` shape.
       Matched case-insensitively because the measured splice was lowercased
       prose ("...learning outcome co-117").
    2. :func:`hierarchy_from_id` — the PREFIX must be a recognized LO
       hierarchy (sourced from ``schemas/taxonomies/lo_hierarchy.json``:
       TO/CO/MO/PO/UO/LO/SO).

    Check 2 is not redundant. Case-insensitive matching on shape alone turns
    this into "any 2+ letter token, a hyphen, 2+ digits", which accepts
    ordinary technical identifiers — ``ISO-8601``, ``RFC-2119``, ``COVID-19``.
    Those are legitimate near-miss content, and treating them as the degenerate
    ``co-117`` splice signature silently deletes candidates on any standards,
    CS, or medical corpus. Gating on the canonical prefix taxonomy keeps the
    measured signature while making the detector corpus-agnostic — and it stays
    data-driven, so a course family adopting a new prefix needs no code edit.
    """
    upper = token.upper()
    if not validate_lo_id(upper):
        return False
    try:
        hierarchy_from_id(upper)
    except ValueError:
        return False
    return True


def _lo_id_tokens(text: str) -> List[str]:
    """Return the LO-identifier tokens present in ``text``.

    ``_WORD_RE`` splits on the hyphen, so the id is reassembled from adjacent
    ``word-word`` pairs before validation.
    """
    out: List[str] = []
    for match in re.finditer(r"[A-Za-z]{2,}-\d{2,}", text or ""):
        if _is_lo_id_token(match.group(0)):
            out.append(match.group(0))
    return out


def completion_skeleton(completion: str) -> str:
    """D3 — the pool-frequency key for a completion.

    casefold -> drop LO-identifier tokens -> collapse whitespace -> sha256.
    Two completions that differ only in the identifier spliced into the topic
    slot (the measured degenerate shape) therefore share one skeleton.

    NUMBERS ARE SIGNIFICANT and are deliberately NOT normalized. An earlier
    revision also collapsed every digit run to ``0``, which made three
    numerically distinct worked answers of a shared prose shape
    ("Multiply 3 by 14 to get 42 ...", "Multiply 5 by 22 to get 110 ...")
    hash identically. On a math corpus — the primary corpus class here — that
    is the whole numeric-answer population, and three such candidates were
    enough to trip the default ``max_skeleton_freq`` of 3 and reject all of
    them under a counter (``skeleton_collapse``) an operator reads as
    "template junk correctly detected". The degenerate signature this filter
    exists for varies in its LO identifier, which the preceding line already
    removes, so the digit collapse bought nothing and cost the corpus.
    """
    text = str(completion or "").casefold()
    for token in _lo_id_tokens(text):
        text = text.replace(token.casefold(), " ")
    text = _WS_RE.sub(" ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_jaccard(left: str, right: str) -> float:
    """Token-set Jaccard of two completions (0.0 when both are empty)."""
    a: Set[str] = {t.lower() for t in _word_tokens(left)}
    b: Set[str] = {t.lower() for t in _word_tokens(right)}
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _novel_content_tokens(completion: str, prompt: str) -> int:
    """D2 — distinct >=3-char content tokens in the completion, not in the prompt.

    LO-identifier tokens are dropped first so the spliced key cannot itself
    count as novel content.
    """
    text = str(completion or "")
    for token in _lo_id_tokens(text):
        text = text.replace(token, " ")
    prompt_tokens = {t.lower() for t in _word_tokens(str(prompt or ""))}
    novel = {
        t.lower()
        for t in _word_tokens(text)
        if len(t) >= 3 and t.lower() not in prompt_tokens
    }
    return len(novel)


@dataclass
class MinedFunnel:
    """Per-stage counters for one mining pass.

    Populated identically in ``shadow`` and ``on`` — shadow's entire purpose is
    to MEASURE the funnel on real data before a single row is emitted, so it
    runs the whole selection and then returns nothing.
    """

    candidates_seen: int = 0
    stale_contract: int = 0
    stage_excluded: int = 0
    deps_missing: int = 0
    too_few_claims: int = 0
    contradicted: int = 0
    below_support_floor: int = 0
    below_fail_entailment: int = 0
    lo_id_splice: int = 0
    low_novel_content: int = 0
    skeleton_collapse: int = 0
    no_anchor: int = 0
    identity_mismatch: int = 0
    jaccard_out_of_band: int = 0
    length_out_of_band: int = 0
    unregistered_teacher: int = 0
    duplicate_prompt: int = 0
    per_chunk_capped: int = 0
    global_capped: int = 0
    emitted: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "candidates_seen": self.candidates_seen,
            "stale_contract": self.stale_contract,
            "stage_excluded": self.stage_excluded,
            "deps_missing": self.deps_missing,
            "too_few_claims": self.too_few_claims,
            "contradicted": self.contradicted,
            "below_support_floor": self.below_support_floor,
            "below_fail_entailment": self.below_fail_entailment,
            "lo_id_splice": self.lo_id_splice,
            "low_novel_content": self.low_novel_content,
            "skeleton_collapse": self.skeleton_collapse,
            "no_anchor": self.no_anchor,
            "identity_mismatch": self.identity_mismatch,
            "jaccard_out_of_band": self.jaccard_out_of_band,
            "length_out_of_band": self.length_out_of_band,
            "unregistered_teacher": self.unregistered_teacher,
            "duplicate_prompt": self.duplicate_prompt,
            "per_chunk_capped": self.per_chunk_capped,
            "global_capped": self.global_capped,
            "emitted": self.emitted,
        }


@dataclass(frozen=True)
class _Qualified:
    """One (anchor, candidate) pair that cleared every selection predicate."""

    chunk_id: str
    anchor_variant: int
    candidate: RejectCandidate
    anchor: Mapping[str, Any]
    support_rate: float
    min_fail_entailment: float
    n_claims: int
    jaccard: float
    skeleton_freq: int
    completion_digest: str

    @property
    def order_key(self) -> Tuple[float, float, str, str, int]:
        """Total order: best near-miss first, ties broken by content digest.

        ``chunk_id`` + ``anchor_variant`` extend the contract's three-part key
        so the order is total even for two candidates with identical scores AND
        identical completions — no set/dict iteration order can leak in.
        """
        return (
            -self.support_rate,
            -self.min_fail_entailment,
            self.completion_digest,
            self.chunk_id,
            self.anchor_variant,
        )


def _anchor_index(
    accepted_records: Iterable[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    """Group accepted INSTRUCTION pairs by chunk id, in a stable order.

    Deterministic-generator rows (kg_metadata / violation_detection /
    abstention / schema_translation) never carry ``instruction_variant`` and
    their ``chunk_id`` is a synthetic anchor that resolves against no chunk, so
    they are excluded here rather than being offered as anchors.
    """
    index: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {}
    for position, record in enumerate(accepted_records):
        if not isinstance(record, Mapping):
            continue
        if "instruction_variant" not in record:
            continue
        chunk_id = str(record.get("chunk_id") or "").strip()
        completion = str(record.get("completion") or "").strip()
        prompt = str(record.get("prompt") or "").strip()
        if not chunk_id or not completion or not prompt:
            continue
        index.setdefault(chunk_id, []).append((position, record))
    return {
        chunk_id: [
            rec for _, rec in sorted(
                rows,
                key=lambda item: (
                    int(item[1].get("instruction_variant") or 0), item[0],
                ),
            )
        ]
        for chunk_id, rows in index.items()
    }


def _score_band(
    candidate: RejectCandidate,
    funnel: MinedFunnel,
    *,
    min_support: float,
    min_fail_entailment: float,
) -> Optional[Tuple[float, float, int]]:
    """Apply the near-miss score band. Returns ``(support, min_fail, n)``.

    Each failure increments its own counter and stops — the funnel is the
    diagnostic that tells an operator WHY a shadow run mined nothing.
    """
    per_claim = candidate.per_claim_support
    if not per_claim:
        # The NLI degrade arm stamps ``per_claim_support=None``; a row with no
        # scores cannot be judged for near-missness at all.
        funnel.deps_missing += 1
        return None
    if candidate.claim_support_rate is None:
        funnel.deps_missing += 1
        return None
    if len(per_claim) < MIN_SCORED_CLAIMS:
        # One or two scored claims carry no ratio signal.
        funnel.too_few_claims += 1
        return None
    if (candidate.claim_contradicted_rate or 0.0) != 0.0:
        # A contradicted claim is a blunt factual error, not a near-miss — and
        # this project has MEASURED the contradiction channel mislabelling a
        # true paraphrase at c=0.92, so it is the least trustworthy NLI output.
        funnel.contradicted += 1
        return None
    support = float(candidate.claim_support_rate)
    # ``support >= 1.0`` is definitional dead code (a fully supported
    # completion could not have failed the gate); it is banded anyway so a
    # malformed row can never be admitted on a vacuous score.
    if not (min_support <= support < 1.0):
        funnel.below_support_floor += 1
        return None
    failing = [
        _as_float(entry.get("entailment"))
        for entry in per_claim
        if str(entry.get("outcome") or "") == "unsupported"
    ]
    failing_values = [value for value in failing if value is not None]
    if not failing_values:
        # No scored claim actually drifted -> nothing near-miss about it.
        funnel.below_fail_entailment += 1
        return None
    worst = min(failing_values)
    if worst < min_fail_entailment:
        funnel.below_fail_entailment += 1
        return None
    return (support, worst, len(per_claim))


def _identity_matches(
    anchor: Mapping[str, Any],
    candidate: RejectCandidate,
    *,
    persona: str,
) -> bool:
    """I1-I6 — six EXACT predicates, no embeddings.

    I1 same chunk; I2 same focused objective set; I3 same unframed base
    question; I4 same citation contract; I5 one teacher per row; I6 the two
    sides are genuinely different units.
    """
    anchor_lo = sorted(
        str(ref).strip()
        for ref in (anchor.get("lo_refs") or [])
        if str(ref).strip()
    )
    candidate_lo = sorted(candidate.lo_refs)
    if not candidate_lo or anchor_lo != candidate_lo:
        # An empty lo_refs set could not satisfy the schema's ``minItems: 1``
        # either, so it is an identity failure rather than a silent emit.
        return False
    if str(anchor.get("chunk_id") or "") != candidate.chunk_id:
        return False
    anchor_prompt = unframe_instruction_prompt(
        anchor.get("prompt"), persona=persona,
    )
    reject_prompt = unframe_instruction_prompt(
        candidate.prompt, persona=persona,
    )
    if not anchor_prompt or anchor_prompt != reject_prompt:
        return False
    if bool(anchor.get("requires_source_citation")) != bool(
        candidate.requires_source_citation
    ):
        return False
    if str(anchor.get("provider") or "") != candidate.provider:
        return False
    anchor_seat = anchor.get("generating_seat")
    anchor_seat = str(anchor_seat) if anchor_seat else None
    if anchor_seat != candidate.generating_seat:
        return False
    if int(anchor.get("instruction_variant") or 0) == candidate.variant_index:
        return False
    return True


def _build_mined_pair(
    qualified: _Qualified,
    *,
    rank: int,
    event_id: str,
) -> Dict[str, Any]:
    """Assemble one ``preference_pairs.jsonl`` row from an anchor + a reject."""
    anchor = qualified.anchor
    candidate = qualified.candidate
    chunk_id = qualified.chunk_id
    lo_refs = [
        str(ref).strip()
        for ref in (anchor.get("lo_refs") or [])
        if str(ref).strip()
    ]
    seed = candidate.seed
    if seed is None:
        seed = anchor.get("seed")
    try:
        seed_value = int(seed) if seed is not None else 0
    except (TypeError, ValueError):
        seed_value = 0
    pair: Dict[str, Any] = {
        "prompt": str(anchor.get("prompt") or ""),
        "chosen": str(anchor.get("completion") or ""),
        "rejected": candidate.completion,
        "chunk_id": chunk_id,
        "lo_refs": lo_refs,
        "seed": seed_value,
        "decision_capture_id": event_id,
        "schema_version": "v1",
        "misconception_id": None,
        # OVERCLAIM, stated rather than silently chosen: the ``chosen`` side
        # genuinely IS a validated accepted completion (it passed promotion,
        # claim_support, lo_refs and objective_delivery). The row AS A
        # PREFERENCE PAIR was never put through a preference promotion
        # validator. "candidate" is more honest but is not a tier the DPO path
        # expects; "trainable" is worse (it is earned by passing gates).
        # "validated" is the least-wrong of three imperfect options.
        "promotion_status": "validated",
        "provider": str(anchor.get("provider") or ""),
        "source_chunk_id": chunk_id,
        # ``rejected_source`` is DELIBERATELY ABSENT: its schema enum is
        # {misconception, rule_synthesized} and this row is neither. Widening
        # that enum would edit a _GENERATION_CONTRACT_FILES schema and re-key
        # the generation fingerprint a second time; both pair schemas are
        # ``additionalProperties: true``, so ``source`` + ``reject_mining``
        # carry the full provenance without it.
        "source": MINED_PAIR_SOURCE,
        "source_references": list(anchor.get("source_references") or []),
        "source_citation": f"[{chunk_id}]",
        "reject_mining": {
            "schema_version": REJECT_MINING_SCHEMA_VERSION,
            "rejection_reason": candidate.reason,
            "rejected_variant_index": candidate.variant_index,
            "anchor_variant_index": qualified.anchor_variant,
            "claim_support_rate": qualified.support_rate,
            "claim_contradicted_rate": 0.0,
            "min_failing_entailment": qualified.min_fail_entailment,
            "n_claims": qualified.n_claims,
            "completion_jaccard": qualified.jaccard,
            "skeleton_frequency": qualified.skeleton_freq,
            "rank": rank,
            "contract_fingerprint": candidate.contract_fingerprint,
        },
    }
    # Canonical helper, once per row. When the teacher carried no fine-grained
    # seat id (today's accepted pairs do not stamp one), the coarse provider
    # provenance resolves the tag instead of writing an empty ``generating_seat``
    # that would read as a missing-teacher signal downstream.
    if candidate.generating_seat:
        stamp_pair_license(
            pair,
            generating_seat=candidate.generating_seat,
            provider=pair["provider"],
        )
    else:
        pair["license"] = provenance_license_tag(provider=pair["provider"])
    return pair


def select_mined_pairs(
    candidates: Iterable[RejectCandidate],
    accepted_records: Iterable[Mapping[str, Any]],
    *,
    persona: str,
    mode: str,
    capture: Any,
    event_id_fn: Callable[[Any], str],
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_fail_entailment: float = DEFAULT_MIN_FAIL_ENTAILMENT,
    max_skeleton_freq: int = DEFAULT_MAX_SKELETON_FREQ,
    max_fraction: float = DEFAULT_MAX_FRACTION,
    emitted_pref_prompts: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], MinedFunnel]:
    """Turn pooled rejects into DPO preference rows. Pure and deterministic.

    Returns ``(rows, funnel)``. In ``shadow`` the whole funnel runs and
    ``rows`` is empty — ``funnel.emitted`` still reports how many rows WOULD
    have landed, so an operator can read the real yield before changing a
    single byte of the corpus. In ``off`` nothing runs at all.

    ``emitted_pref_prompts`` is READ AND MUTATED (a prompt that lands is added)
    so mined rows share the preference arm's cross-chunk prompt-collision
    guard. The misconception-DPO block bypasses that set; this deliberately
    does not copy that shape.
    """
    funnel = MinedFunnel()
    if mode == MINE_MODE_OFF:
        return ([], funnel)

    pool = sorted(
        (c for c in candidates if isinstance(c, RejectCandidate)),
        key=lambda c: (c.chunk_id, c.variant_index),
    )
    funnel.candidates_seen = len(pool)
    anchors = _anchor_index(accepted_records)
    accepted_total = sum(len(rows) for rows in anchors.values())
    seen_prompts = (
        emitted_pref_prompts if emitted_pref_prompts is not None else set()
    )

    # --- pass 1: stage admission (the pool-frequency universe) --------------
    staged: List[RejectCandidate] = []
    for candidate in pool:
        # Contract gate FIRST, ahead of stage admission, so a stale-contract
        # row never even enters the D3 skeleton-frequency universe. A candidate
        # inherited from a prior run whose per-seed contract fingerprint this
        # run did not positively re-confirm may predate a model-id / generation
        # -parameter / generation-contract-file change; the six identity
        # predicates below compare none of those, so this is the only place it
        # can be caught. Emitting it would pair last contract's rejected text
        # against this contract's ``chosen`` -- exactly the "stale pairs survive
        # a change that should have invalidated them" failure the module's
        # _VERDICT_POLICY_FILES registration argument depends on being
        # impossible.
        if not candidate.contract_confirmed:
            funnel.stale_contract += 1
            continue
        if not is_mineable_reason(candidate.reason):
            funnel.stage_excluded += 1
            continue
        staged.append(candidate)
    # D3's frequency map is built over every STAGE-ADMITTED candidate, not only
    # the score-band survivors: a skeleton that recurs among degenerates is
    # still evidence that the skeleton is a template, and a near-miss sharing
    # it is not distinguishable from the template on the emitted text.
    skeleton_freq: Dict[str, int] = {}
    skeletons: Dict[Tuple[str, int], str] = {}
    for candidate in staged:
        digest = completion_skeleton(candidate.completion)
        skeletons[(candidate.chunk_id, candidate.variant_index)] = digest
        skeleton_freq[digest] = skeleton_freq.get(digest, 0) + 1

    # --- pass 2: per-candidate qualification --------------------------------
    qualified: List[_Qualified] = []
    for candidate in staged:
        banded = _score_band(
            candidate, funnel,
            min_support=min_support,
            min_fail_entailment=min_fail_entailment,
        )
        if banded is None:
            continue
        support, worst, n_claims = banded

        # D1 — the measured "...learning outcome co-117" splice signature.
        if _lo_id_tokens(candidate.completion):
            funnel.lo_id_splice += 1
            continue
        # D2 — a fixed pedagogical frame plus the prompt's own topic slot
        # cannot clear a novel-content floor.
        if _novel_content_tokens(
            candidate.completion, candidate.prompt,
        ) < MIN_NOVEL_CONTENT_TOKENS:
            funnel.low_novel_content += 1
            continue
        # D3 — repetition IS template collapse.
        digest = skeletons[(candidate.chunk_id, candidate.variant_index)]
        freq = skeleton_freq.get(digest, 0)
        if freq >= max_skeleton_freq:
            funnel.skeleton_collapse += 1
            continue

        chunk_anchors = anchors.get(candidate.chunk_id) or []
        if not chunk_anchors:
            # A chunk with rejects but NO accepted positive emits nothing. No
            # ``chosen`` is ever generated, templated, extracted, or borrowed
            # from another chunk — every one of those manufactures a worse
            # negative than none. The count is reported, not swallowed.
            funnel.no_anchor += 1
            continue

        matched = [
            anchor for anchor in chunk_anchors
            if _identity_matches(anchor, candidate, persona=persona)
        ]
        if not matched:
            funnel.identity_mismatch += 1
            continue
        anchor = matched[0]

        chosen = str(anchor.get("completion") or "")
        prompt = str(anchor.get("prompt") or "")
        if not (
            MIN_CAPTURED_PROMPT_CHARS <= len(prompt)
            <= MAX_CAPTURED_PROMPT_CHARS
        ) or not (
            MIN_CAPTURED_COMPLETION_CHARS <= len(chosen)
            <= MAX_CAPTURED_COMPLETION_CHARS
        ) or not (
            MIN_CAPTURED_COMPLETION_CHARS <= len(candidate.completion)
            <= MAX_CAPTURED_COMPLETION_CHARS
        ):
            funnel.length_out_of_band += 1
            continue

        jaccard = token_jaccard(chosen, candidate.completion)
        if not (MIN_COMPLETION_JACCARD <= jaccard <= MAX_COMPLETION_JACCARD):
            funnel.jaccard_out_of_band += 1
            continue

        # Licence resolution happens HERE, before the caps, so an unusable row
        # never consumes its chunk's single slot. ``assert_export_licenses``
        # fail-closes a whole multi-hour training run on a bad teacher, and a
        # mined row must never be the thing that bricks it. Uses the canonical
        # classifier (which also catches barred / claude-tagged teachers, not
        # only unregistered ones) rather than a second implementation.
        status, _verdict, _label = classify_pair_teacher({
            "generating_seat": candidate.generating_seat,
            "provider": candidate.provider,
        })
        if status != "ok":
            funnel.unregistered_teacher += 1
            continue

        if prompt in seen_prompts:
            funnel.duplicate_prompt += 1
            continue

        qualified.append(_Qualified(
            chunk_id=candidate.chunk_id,
            anchor_variant=int(anchor.get("instruction_variant") or 0),
            candidate=candidate,
            anchor=anchor,
            support_rate=support,
            min_fail_entailment=worst,
            n_claims=n_claims,
            jaccard=jaccard,
            skeleton_freq=freq,
            completion_digest=hashlib.sha256(
                candidate.completion.encode("utf-8"),
            ).hexdigest(),
        ))

    # --- pass 3: caps + deterministic tie-break -----------------------------
    qualified.sort(key=lambda q: q.order_key)
    chosen_by_chunk: Dict[str, _Qualified] = {}
    chosen_by_anchor: Set[Tuple[str, int]] = set()
    for item in qualified:
        anchor_key = (item.chunk_id, item.anchor_variant)
        if item.chunk_id in chosen_by_chunk or anchor_key in chosen_by_anchor:
            funnel.per_chunk_capped += 1
            continue
        chosen_by_chunk[item.chunk_id] = item
        chosen_by_anchor.add(anchor_key)
    selected = sorted(chosen_by_chunk.values(), key=lambda q: q.order_key)

    ceiling = int(max_fraction * accepted_total)
    if len(selected) > ceiling:
        funnel.global_capped += len(selected) - ceiling
        selected = selected[:ceiling]

    funnel.emitted = len(selected)

    # --- pass 4: emit --------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    if mode == MINE_MODE_ON:
        for rank, item in enumerate(selected, start=1):
            capture.log_decision(
                decision_type=SELECTION_DECISION_TYPE,
                decision=(
                    f"Mine rejected instruction variant "
                    f"{item.candidate.variant_index} of chunk "
                    f"{item.chunk_id} as the DPO rejected side against "
                    f"accepted variant {item.anchor_variant}."
                ),
                # Real per-unit signals, not a static boilerplate rationale.
                rationale=(
                    f"chunk={item.chunk_id}; "
                    f"rejection_reason={item.candidate.reason}; "
                    f"claim_support_rate={item.support_rate:.4f} "
                    f"(floor {min_support:.2f}); "
                    f"min_failing_entailment={item.min_fail_entailment:.4f} "
                    f"(floor {min_fail_entailment:.2f}); "
                    f"n_claims={item.n_claims}; "
                    f"contradicted_rate=0.0; "
                    f"completion_jaccard={item.jaccard:.4f}; "
                    f"skeleton_frequency={item.skeleton_freq} "
                    f"(max {max_skeleton_freq}); "
                    f"anchor_variant={item.anchor_variant}; "
                    f"rejected_variant={item.candidate.variant_index}; "
                    f"rank={rank}/{len(selected)}. Near-miss: the majority of "
                    "this completion's claims were supported and a minority "
                    "drifted, so it teaches grounded-vs-unsupported rather "
                    "than template-junk-vs-prose."
                ),
                alternatives_considered=[
                    "emit nothing for this chunk (no near-miss reject)",
                    "synthesize or template a chosen side (forbidden: "
                    "inverts the trap — DPO would learn to prefer templates)",
                ],
                context=(
                    f"chunk_id={item.chunk_id}; "
                    f"anchor_variant={item.anchor_variant}; "
                    f"rejected_variant={item.candidate.variant_index}"
                ),
            )
            rows.append(_build_mined_pair(
                item, rank=rank, event_id=event_id_fn(capture),
            ))
            seen_prompts.add(str(item.anchor.get("prompt") or ""))

    counters = funnel.as_dict()
    capture.log_decision(
        decision_type=SELECTION_DECISION_TYPE,
        decision=(
            f"Reject-mining pass ({mode}) selected {funnel.emitted} DPO "
            f"negative(s) from {funnel.candidates_seen} pooled reject(s) "
            f"against {accepted_total} accepted instruction anchor(s); "
            f"emitted {len(rows)}."
        ),
        rationale=(
            "Deterministic reject-mining funnel: "
            + "; ".join(f"{k}={v}" for k, v in counters.items())
            + f". Bands: min_support={min_support}, "
            f"min_fail_entailment={min_fail_entailment}, "
            f"max_skeleton_freq={max_skeleton_freq}, "
            f"max_fraction={max_fraction} (ceiling {ceiling}). "
            "Degenerate template-collapse rejects are excluded by stage "
            "admission, the score band, the LO-identifier splice check and "
            "the pool skeleton-frequency check; no chosen side is ever "
            "synthesized for a chunk with no accepted positive."
        ),
        context=f"mode={mode}; accepted_anchors={accepted_total}",
    )
    return (rows, funnel)
