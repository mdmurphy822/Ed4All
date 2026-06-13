"""Three-arm eval scorecard — what the model knows alone, what retrieval
surfaces alone, and what the combined grounded system delivers.

The three arms answer three different questions over the SAME frozen gold set,
scored with the SAME key-point machinery
(:func:`lib.retrieval.answer_scoring.score_key_point_coverage`) so the
comparison is honest:

  1. **BASE** (qwen only, no retrieval) — ask the raw local model with a minimal
     "answer this question" prompt, NO retrieved passages and NO refusal
     scaffolding. Measures what the model knows unaided. Promotes the prior
     one-off ``/tmp/base_ablation.py`` logic into the harness: per-question error
     isolation, ``max_tokens=900``, scored via ``score_key_point_coverage``.
     There is NO citation axis (N/A by construction) and NO refusal probe — the
     base model answers everything. The base answer IS, however, scored on the
     HALLUCINATION axis: after composing each answer the arm retrieves the SAME
     top-k passages the grounded pipeline would see (the RETRIEVAL arm's entry)
     and scores it with ``score_groundedness`` (scorer v2 — IDENTICAL machinery
     to the grounded arm), yielding an ``unsupported_claim_rate`` that is
     directly comparable to the grounded arm's. This is unsupported-vs-COURSE-
     CORPUS (a true-but-extra-course claim counts as unsupported) — the
     product's hallucination definition, NOT a general factual-error rate. When
     NLI is unavailable the axis reads n/a with a recorded reason (never a
     fabricated rate).

  2. **RETRIEVAL** (retrieval only, no LLM) — retrieve top-k for each question
     and score the EXTRACTIVE CEILING: ``score_key_point_coverage`` over the
     concatenated retrieved passage bodies (does retrieval even surface the
     expected content?), plus primary-relevant hit@k and top-1 hit against the
     gold ``relevant_passages``. No model call; latency is retrieval latency.

  3. **GROUNDED** (qwen + retrieval) — the existing full pipeline, delegated to
     :func:`lib.retrieval.grounded_eval.run_grounded_eval` UNCHANGED (it still
     writes its own ``grounded_answer_eval_<ts>.json`` report + review sample,
     so the staleness test that reads that artifact keeps working exactly as
     now).

The scorecard assembles the requested arms into ONE
``retrieval_eval/eval_scorecard_<ts>.json`` with per-arm blocks + a
``comparison`` block on the shared axes (key_point_coverage, answered/declined
counts, latency p50/p95). Grounded-only axes (citations, groundedness, refusal)
stay inside the grounded block. The default CLI arm set is ``grounded`` alone,
so every existing report artifact + the staleness test are byte-for-byte
unaffected.

Decision-capture contract: the BASE arm is a NEW LLM call path. Per the project
contract (docs/architecture/decision-capture.md) each per-question base call
emits a ``base_model_eval_call`` decision whose rationale interpolates dynamic
signals (course slug, question id, model id, max_tokens, key-point count); the
``OpenAICompatibleClient`` ALSO emits its own per-call ``llm_chat_call`` event
when threaded the same capture handle. The RETRIEVAL arm makes no LLM call (no
capture).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from lib.retrieval.answer_scoring import DISCLAIMER_PHRASES, score_key_point_coverage
from lib.retrieval.grounded_eval import (
    RETRIEVAL_EVAL_SUBDIR,
    _course_dir,
    _gold_questions,
    _load_probes,
    _percentile,
    _relevant_chunk_ids,
    _utcnow_iso,
    run_grounded_eval,
)
from lib.retrieval.gold_set import has_critical_issues, load_gold_set

#: Scorecard artifact schema. Additive conventions: new arm blocks / axes are
#: added without bumping unless a field's MEANING changes. Disjoint filename
#: (``eval_scorecard_*.json``) from the grounded report + review sample.
#: 1.0 → 1.1 (additive only — every 1.0 key unchanged): the BASE arm gained a
#: ``groundedness`` block (hallucination axis: unsupported_claim_rate,
#: groundedness_rate_mean, claims_scored, contradicted/computational/filtered
#: counts), the comparison gained a per-arm ``unsupported_claim_rate`` + a
#: derived ``hallucination_reduction`` entry, and per-base-question rows gained
#: a ``groundedness`` sub-block.
#: 1.1 → 1.2 (additive only — every 1.1 key unchanged): the BASE arm gained a
#: ``probe_confabulation`` block (OUT-OF-SCOPE CONFABULATION axis: the raw model
#: is asked every refusal probe and its answered-vs-declined split is counted as
#: a ``confabulation_rate``), its ``groundedness`` block gained the UNDILUTED
#: ``claim_level_unsupported_rate`` + ``questions_with_scorable_claims`` fields
#: (alongside the unchanged per-question-mean ``unsupported_claim_rate``), the
#: comparison gained an ``out_of_scope_confabulation_rate`` per-arm row + a
#: ``_note`` on ``hallucination_reduction`` naming the dilution convention, and
#: per-base-probe rows were added.
SCORECARD_SCHEMA_VERSION = "1.2"

SCORECARD_FILENAME_PREFIX = "eval_scorecard_"

#: The three arm names, in display order.
ARM_BASE = "base"
ARM_RETRIEVAL = "retrieval"
ARM_GROUNDED = "grounded"
ALL_ARMS = (ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED)

#: Token cap for the BASE arm composition, matching the one-off ablation script
#: (a base model with no passages tends to ramble; 900 is generous headroom).
_BASE_MAX_TOKENS = 900
#: Deterministic decoding for the eval (the one-off used temperature=0.0).
_BASE_TEMPERATURE = 0.0

#: Minimal, retrieval-free, refusal-free base prompt. The whole point of the
#: BASE arm is to probe the model's unaided knowledge, so there is NO system
#: grounding, NO passages, and NO "answer only from the course" instruction.
_BASE_SYSTEM_PROMPT = (
    "You are a knowledgeable tutor. Answer the question as accurately and "
    "completely as you can from your own knowledge."
)


def _question_key_points(question: Dict[str, Any]) -> List[str]:
    kps = question.get("expected_key_points")
    return (
        [str(k) for k in kps if str(k).strip()]
        if isinstance(kps, list)
        else []
    )


def _coverage_fields(cov: Any) -> Dict[str, Any]:
    """Project a ``KeyPointCoverage`` (or ``None``) into a flat dict."""
    if cov is None:
        return {"total": 0, "covered": 0, "coverage_rate": None}
    return {
        "total": cov.total,
        "covered": cov.covered,
        "coverage_rate": cov.coverage_rate,
    }


def _load_verified_gold(
    repo_root: Path, course_slug: str
) -> Dict[str, Any]:
    """Load + verify the gold set (WS1 fail-closed contract, shared by arms).

    Raises ``RuntimeError`` on any critical gold-set issue so a base / retrieval
    arm never measures against an unverified gold set — mirrors
    :func:`run_grounded_eval`'s gate exactly.
    """
    course_dir = _course_dir(repo_root, course_slug)
    gold, issues = load_gold_set(course_dir, verify=True)
    if has_critical_issues(issues):
        codes = sorted({i.code for i in issues if i.severity == "critical"})
        raise RuntimeError(
            f"gold set for {course_slug!r} has critical issues {codes}; "
            f"refusing to run the eval scorecard on an unverified gold set."
        )
    return gold


# --------------------------------------------------------------------------- #
# BASE arm — qwen only, no retrieval
# --------------------------------------------------------------------------- #

#: Note stamped on the base block's hallucination axis so a reader never
#: mistakes it for a general factual-error rate. The product's definition of a
#: hallucination is "ungrounded in the COURSE corpus" — a claim that is true in
#: the world but absent from this course's material STILL counts as unsupported
#: here. That is deliberate (the grounded arm is held to the same bar), so the
#: BASE-vs-GROUNDED comparison is apples-to-apples.
_BASE_GROUNDEDNESS_NOTE = (
    "unsupported_claim_rate is unsupported-vs-COURSE-CORPUS (the SAME top-k "
    "passages the grounded pipeline would retrieve, scored with the IDENTICAL "
    "score_groundedness v2 machinery: artifact filter + computational exemption "
    "+ windowed rescue). A true-but-extra-course claim counts as unsupported — "
    "this is the product's hallucination definition (ungrounded-in-course "
    "content), deliberately NOT a general factual-error rate."
)


#: Note stamped on the base block's OUT-OF-SCOPE CONFABULATION axis. The base
#: model has NO refusal machinery, so it answers the deliberately-unanswerable
#: refusal probes with pure invention — the most product-relevant hallucination
#: behavior. The grounded pipeline, by contrast, REFUSES / flags these (its
#: ``headline.refusal`` block); even when the grounded arm "answers" a probe the
#: answer still passes the citation gate + disclaimer scaffolding, whereas a base
#: "answer" is unanchored invention. The two are NOT the same failure and the
#: comparison labels them as such (``answered-instead-of-refused``).
_PROBE_CONFABULATION_NOTE = (
    "confabulation_rate is answered / n_probes over the refusal-probe set "
    "(questions deliberately unanswerable from the course corpus). The BASE arm "
    "has NO refusal machinery, so a 'declined' probe means the raw answer text "
    "tripped a does-not-cover disclaimer heuristic (reused from the gold key-"
    "point scorer) — a raw model nearly always answers, so this rate is "
    "expected to be near 1.0. A base 'answer' on a probe is pure invention; the "
    "grounded pipeline's comparable axis (answered-instead-of-refused = "
    "1 - refusal_recall) counts probes it failed to refuse, but even those "
    "grounded 'answers' still pass citation gates / disclaimers — NOT the same "
    "as base invention."
)


def _base_answer_declines(answer_text: Optional[str]) -> bool:
    """Decline heuristic for a raw base-model probe answer.

    Reuses the gold key-point scorer's :data:`DISCLAIMER_PHRASES` (the SAME
    does-not-cover phrase bank the grounded part-coverage flagging uses) so the
    base arm's "declined" notion is consistent with the rest of the harness — no
    bespoke phrase list. A raw model with NO refusal scaffolding nearly always
    answers, so ``True`` (declined) is the rare case: the answer text explicitly
    disclaims coverage / scope. Empty / whitespace-only answers count as
    declined (the model produced no real answer).
    """
    text = (answer_text or "").strip().lower()
    if not text:
        return True
    return any(phrase in text for phrase in DISCLAIMER_PHRASES)


def _resolve_base_groundedness_nli(nli: Optional[Any]) -> tuple:
    """Resolve the NLI singleton ONCE for the whole base arm.

    Reuses :func:`lib.retrieval.groundedness._resolve_nli` (the SAME resolution
    the grounded arm uses) so the comparison is apples-to-apples. Returns
    ``(resolved_nli_or_None, reason_or_None)``: when the model is unavailable
    (missing extras / load failure) the second element is a recorded reason and
    the hallucination axis reads n/a — scores are NEVER fabricated. The import
    is lazy so the ~750 MB DeBERTa stack is never imported when the groundedness
    deps are absent.
    """
    try:
        from lib.retrieval.groundedness import _resolve_nli
    except Exception:  # noqa: BLE001 — groundedness module absent → degrade
        return None, "groundedness_module_unavailable"
    resolved = _resolve_nli(nli)
    if resolved is None:
        return None, "nli_unavailable"
    return resolved, None


def _base_retrieve_fn(
    repo_root: Path, retrieve_fn: Optional[Any]
) -> Any:
    """Build the base arm's retrieval closure (same entry the RETRIEVAL arm uses).

    When ``retrieve_fn`` is injected (tests), it is returned unchanged. Otherwise
    the live :func:`lib.retrieval.grounded_answer._retrieve` is bound to the
    resolved LibV2 root — the SAME engine/limit-arg signature the RETRIEVAL arm
    uses, so the base arm sees exactly the passages the grounded pipeline would.
    """
    if retrieve_fn is not None:
        return retrieve_fn

    from lib.retrieval.grounded_answer import _libv2_root, _retrieve

    libv2_root = _libv2_root(repo_root)

    def _fn(_root, slug, query, *, engine, limit):
        return _retrieve(libv2_root, slug, query, engine=engine, limit=limit)

    return _fn


def _load_base_probes(
    repo_root: Path,
    course_slug: str,
    *,
    probes: Optional[Sequence[Dict[str, Any]]] = None,
    refusal_probes_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Resolve the refusal-probe set for the base arm's confabulation pass.

    Injected ``probes`` win (tests). Otherwise the same on-disk path the grounded
    arm reads (``<course>/retrieval_eval/refusal_probes.json``), via the SHARED
    :func:`lib.retrieval.grounded_eval._load_probes` loader — so the base arm and
    the grounded arm probe the IDENTICAL set. Missing / malformed file → empty
    list (the confabulation axis then reads n/a, never fabricated).
    """
    if probes is not None:
        return [p for p in probes if isinstance(p, dict)]
    probes_path = (
        Path(refusal_probes_path)
        if refusal_probes_path is not None
        else _course_dir(repo_root, course_slug)
        / RETRIEVAL_EVAL_SUBDIR
        / "refusal_probes.json"
    )
    return _load_probes(probes_path)


def _run_base_probe_pass(
    probes: Sequence[Dict[str, Any]],
    *,
    client: Any,
    max_tokens: int,
) -> Dict[str, Any]:
    """Ask the raw base model every refusal probe; count answered vs declined.

    The OUT-OF-SCOPE CONFABULATION axis. Each probe is a question deliberately
    unanswerable from the course corpus; the base model — with NO refusal
    scaffolding — answers it with invented content. Drives each probe through the
    SAME minimal base prompt the gold pass uses, then classifies the answer with
    :func:`_base_answer_declines` (the shared does-not-cover disclaimer
    heuristic). Per-probe error isolation: a client exception on one probe is
    recorded (``errored``) and the pass keeps going.

    Returns the ``probe_confabulation`` block:
    ``{n_probes, answered, declined, errored, confabulation_rate, probes:[...]}``.
    ``confabulation_rate = answered / (answered + declined)`` (errored probes are
    excluded from the denominator — they never produced an answer to classify);
    ``None`` when nothing was classifiable (no probes / all errored), never a
    fabricated rate.
    """
    rows: List[Dict[str, Any]] = []
    answered = 0
    declined = 0
    errored = 0
    for probe in probes:
        pid = str(probe.get("probe_id", ""))
        ptext = str(probe.get("question_text", ""))
        category = str(probe.get("category", ""))
        try:
            answer_text = client.chat_completion(
                [
                    {"role": "system", "content": _BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": ptext},
                ],
                max_tokens=max_tokens,
                temperature=_BASE_TEMPERATURE,
            )
        except Exception as exc:  # noqa: BLE001 — per-probe isolation
            errored += 1
            rows.append(
                {
                    "probe_id": pid,
                    "category": category,
                    "answered": None,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            )
            continue
        declined_here = _base_answer_declines(answer_text)
        if declined_here:
            declined += 1
        else:
            answered += 1
        rows.append(
            {
                "probe_id": pid,
                "category": category,
                "answered": not declined_here,
            }
        )

    classifiable = answered + declined
    return {
        "n_probes": len(probes),
        "answered": answered,
        "declined": declined,
        "errored": errored,
        # answered / classifiable — None when nothing was classifiable (no probes
        # or all errored), so the axis reads n/a rather than a fabricated 0.0.
        "confabulation_rate": (
            (answered / classifiable) if classifiable else None
        ),
        "_note": _PROBE_CONFABULATION_NOTE,
        "probes": rows,
    }


def run_base_arm(
    repo_root: Path,
    course_slug: str,
    *,
    client: Optional[Any] = None,
    capture: Optional[Any] = None,
    gold: Optional[Dict[str, Any]] = None,
    max_tokens: int = _BASE_MAX_TOKENS,
    engine: str = "semantic",
    limit: int = 8,
    retrieve_fn: Optional[Any] = None,
    nli: Optional[Any] = None,
    probes: Optional[Sequence[Dict[str, Any]]] = None,
    refusal_probes_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Probe the raw local model's unaided knowledge over the gold set.

    For each gold question carrying ``expected_key_points``, ask the model with
    a minimal prompt (NO retrieved passages, NO refusal scaffolding) and score
    key-point coverage with the shared scorer. Per-question error isolation: a
    client exception on one question is recorded (``error`` flag) and the arm
    keeps going, exactly like the one-off ablation script.

    **Hallucination axis (the hallucination-reduction comparison's BASE leg).**
    After composing each base answer, retrieve the SAME top-k passages the
    grounded pipeline would see (the RETRIEVAL arm's entry — same engine/limit
    args) and score the answer with
    :func:`lib.retrieval.groundedness.score_groundedness` (scorer v2: artifact
    filter + computational exemption + windowed rescue — IDENTICAL machinery to
    the grounded arm so the base-vs-grounded comparison is apples-to-apples).
    The NLI singleton is resolved ONCE outside the loop; when it is unavailable
    the axis reads n/a with a recorded ``reason`` — rates are NEVER fabricated.

    SEMANTICS: ``unsupported_claim_rate`` is unsupported-vs-COURSE-CORPUS — a
    true-but-extra-course claim counts as unsupported. That is the product's
    hallucination definition (ungrounded-in-course content), deliberately NOT a
    general factual-error rate (see :data:`_BASE_GROUNDEDNESS_NOTE`).

    **Out-of-scope confabulation axis (the ``probe_confabulation`` block).** The
    gold set ships refusal probes — questions deliberately unanswerable from the
    course. The grounded pipeline refuses / flags them; the base model, with NO
    refusal machinery, answers them with invented content (the most product-
    relevant hallucination behavior). This arm asks each probe through the SAME
    minimal base prompt and counts answered vs declined (decline = the shared
    does-not-cover disclaimer heuristic, :func:`_base_answer_declines`), yielding
    ``confabulation_rate``. The probe pass runs BEFORE the gold pass so the
    per-question decision-capture rationale can interpolate the probe-pass counts
    (the contract requires NO new call site). ``probes`` / ``refusal_probes_path``
    are injectable for tests; otherwise the same on-disk probe set the grounded
    arm reads is used.

    ``client`` is injectable for tests; when ``None`` the real loopback-enforced
    answer backend is built (:func:`lib.retrieval.answer_backend.build_answer_client`).
    ``retrieve_fn`` / ``nli`` are injectable for tests (same shapes the grounded
    + retrieval arms use). No citations are scored (axis N/A by construction).
    """
    repo_root = Path(repo_root)
    if gold is None:
        gold = _load_verified_gold(repo_root, course_slug)
    questions = _gold_questions(gold)

    if client is None:
        from lib.retrieval.answer_backend import (
            build_answer_client,
            resolve_answer_backend,
        )

        resolved = resolve_answer_backend()
        model_id = resolved.model_id
        client = build_answer_client(resolved, capture=capture)
    else:
        # Duck-typed model id from an injected fake (tests).
        model_id = getattr(client, "model", None) or getattr(
            client, "model_id", None
        )

    # Resolve the NLI singleton ONCE (outside the loop) + the shared retrieval
    # entry. When NLI is absent the hallucination axis degrades to n/a with a
    # reason; scoring is skipped per-question (never fabricated).
    resolved_nli, nli_unavailable_reason = _resolve_base_groundedness_nli(nli)
    base_retrieve_fn = _base_retrieve_fn(repo_root, retrieve_fn)

    # OUT-OF-SCOPE CONFABULATION axis (GAP 1): ask the raw model every refusal
    # probe and count answered (= confabulated) vs declined. Run BEFORE the gold
    # loop so the per-question decision rationale can carry the probe counts (the
    # contract forbids a NEW capture call site). Resolved from the SAME on-disk
    # probe set the grounded arm reads (or an injected set in tests).
    probe_list = _load_base_probes(
        repo_root,
        course_slug,
        probes=probes,
        refusal_probes_path=refusal_probes_path,
    )
    probe_block = _run_base_probe_pass(
        probe_list, client=client, max_tokens=max_tokens
    )

    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    kp_total = 0
    kp_covered = 0
    answered = 0
    errored = 0
    scored_questions = 0
    # Hallucination-axis aggregates — per-question rates averaged the SAME way
    # the grounded arm does (mean-of-rates), so the comparison is apples-to-apples.
    groundedness_rates: List[float] = []
    unsupported_rates: List[float] = []
    claims_scored_total = 0
    contradicted_total = 0
    computational_total = 0
    filtered_total = 0
    # UNDILUTED claim-level aggregates (GAP 2). The per-question-mean convention
    # above credits an answer whose claims are ALL computational/filtered as a
    # 0.0 unsupported rate; these totals roll up the raw claim counts across the
    # arm so the undiluted ``claim_level_unsupported_rate`` can be reported
    # additively alongside (NOT replacing) the diluted mean.
    unsupported_count_total = 0
    questions_with_scorable_claims = 0

    for q in questions:
        qid = str(q.get("question_id", ""))
        qtext = str(q.get("question_text", ""))
        key_points = _question_key_points(q)
        if not key_points:
            # No completeness slice to measure (a v1.0 question) — skip, like
            # the one-off (which `continue`d on empty expected_key_points).
            continue
        scored_questions += 1

        _emit_base_decision(
            capture,
            course_slug=course_slug,
            question_id=qid,
            model_id=model_id,
            max_tokens=max_tokens,
            n_key_points=len(key_points),
            nli_available=resolved_nli is not None,
            nli_unavailable_reason=nli_unavailable_reason,
            probe_block=probe_block,
        )

        t0 = time.monotonic()
        try:
            answer_text = client.chat_completion(
                [
                    {"role": "system", "content": _BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": qtext},
                ],
                max_tokens=max_tokens,
                temperature=_BASE_TEMPERATURE,
            )
        except Exception as exc:  # noqa: BLE001 — per-question isolation
            latency_ms = (time.monotonic() - t0) * 1000.0
            latencies.append(latency_ms)
            errored += 1
            rows.append(
                {
                    "question_id": qid,
                    "answered": False,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                    "key_point_coverage": _coverage_fields(None),
                    "groundedness": None,
                    "answer_len": 0,
                    "latency_ms": latency_ms,
                }
            )
            continue

        latency_ms = (time.monotonic() - t0) * 1000.0
        latencies.append(latency_ms)
        answered += 1
        cov = score_key_point_coverage(answer_text, key_points)
        if cov is not None:
            kp_total += cov.total
            kp_covered += cov.covered

        # Hallucination axis: score this answer against the SAME top-k passages
        # the grounded pipeline would retrieve. Per-question error isolation:
        # a retrieval / scoring failure on one question is recorded and the arm
        # keeps going (mirrors the answer-call isolation above).
        grounded_block = _score_base_groundedness(
            answer_text,
            repo_root=repo_root,
            course_slug=course_slug,
            qtext=qtext,
            engine=engine,
            limit=limit,
            retrieve_fn=base_retrieve_fn,
            nli=resolved_nli,
            nli_unavailable_reason=nli_unavailable_reason,
        )
        if grounded_block.get("available"):
            scored = int(grounded_block.get("scored_count", 0) or 0)
            unsupported = int(grounded_block.get("unsupported_count", 0) or 0)
            g_rate = float(grounded_block.get("groundedness_rate", 0.0) or 0.0)
            groundedness_rates.append(g_rate)
            unsupported_rates.append((unsupported / scored) if scored else 0.0)
            claims_scored_total += scored
            unsupported_count_total += unsupported
            if scored:
                questions_with_scorable_claims += 1
            contradicted_total += int(
                grounded_block.get("contradicted_count", 0) or 0
            )
            computational_total += int(
                grounded_block.get("computational_count", 0) or 0
            )
            filtered_total += int(grounded_block.get("filtered_count", 0) or 0)

        rows.append(
            {
                "question_id": qid,
                "answered": True,
                "key_point_coverage": _coverage_fields(cov),
                "groundedness": grounded_block,
                "answer_len": len(answer_text or ""),
                "latency_ms": latency_ms,
            }
        )

    # Hallucination axis rollup. ``available`` is True iff at least one answer
    # was scored against the corpus; when NLI was unavailable the whole axis is
    # n/a with the recorded reason (never a fabricated 0.0 rate).
    hallucination_available = bool(groundedness_rates)
    groundedness_block: Dict[str, Any] = {
        "available": hallucination_available,
        # n/a when NLI is unavailable / nothing was scored — NEVER fabricated.
        "unsupported_claim_rate": (
            (sum(unsupported_rates) / len(unsupported_rates))
            if unsupported_rates
            else None
        ),
        "groundedness_rate_mean": (
            (sum(groundedness_rates) / len(groundedness_rates))
            if groundedness_rates
            else None
        ),
        # UNDILUTED view (GAP 2): total unsupported claims / total scored claims
        # over the whole arm — does NOT credit an all-computational answer as a
        # 0.0 (the per-question mean ``unsupported_claim_rate`` above does). None
        # when no question had a scorable claim (never a fabricated 0.0). The
        # diluted mean stays the pinned-grounded-basis-comparable metric.
        "claim_level_unsupported_rate": (
            (unsupported_count_total / claims_scored_total)
            if claims_scored_total
            else None
        ),
        # How many scored questions actually contributed a scorable claim (i.e.
        # were NOT all-computational/filtered) — the undiluted denominator's
        # question count, surfaced so the dilution gap is legible.
        "questions_with_scorable_claims": questions_with_scorable_claims,
        "claims_scored": claims_scored_total,
        "contradicted_count": contradicted_total,
        "computational_count": computational_total,
        "filtered_count": filtered_total,
        "questions_scored": len(groundedness_rates),
        "engine": engine,
        "limit": limit,
        "_note": _BASE_GROUNDEDNESS_NOTE,
    }
    if not hallucination_available:
        groundedness_block["reason"] = (
            nli_unavailable_reason or "no_scorable_answers"
        )

    return {
        "arm": ARM_BASE,
        "model_id": model_id,
        "retrieval": False,
        "refusal_scaffolding": False,
        "questions_scored": scored_questions,
        "answered": answered,
        "errored": errored,
        # Out-of-scope confabulation axis: the raw model answered N of the
        # deliberately-unanswerable refusal probes (it has no refusal machinery).
        "probe_confabulation": probe_block,
        "key_point_coverage": {
            "total_key_points": kp_total,
            "covered_key_points": kp_covered,
            "coverage_rate": (kp_covered / kp_total) if kp_total else None,
        },
        # Hallucination axis: the base model's answers scored against the course
        # corpus with the SAME groundedness machinery the grounded arm uses.
        "groundedness": groundedness_block,
        # Base answers everything by construction — it has no refusal machinery.
        # Surfaced as a cheap honesty axis so the scorecard can say "the base
        # model produced an answer for every question, including content it may
        # not actually know" (it never declines).
        "declined": 0,
        "answers_everything": True,
        "citations": "n/a",  # no citation axis by construction
        "latency_ms": {
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
        },
        "questions": rows,
    }


def _score_base_groundedness(
    answer_text: Optional[str],
    *,
    repo_root: Path,
    course_slug: str,
    qtext: str,
    engine: str,
    limit: int,
    retrieve_fn: Any,
    nli: Optional[Any],
    nli_unavailable_reason: Optional[str],
) -> Dict[str, Any]:
    """Score one base answer against the SAME top-k passages the grounded
    pipeline would retrieve, with the IDENTICAL groundedness machinery.

    Returns a flat ``GroundednessReport.to_dict()`` (so the base row carries the
    same shape the grounded arm's per-claim block does). When NLI is unavailable
    the report is ``available=False`` with the recorded reason — never a
    fabricated rate. Per-question error isolation: a retrieval / scoring
    exception degrades THIS question's block to ``available=False`` with the
    error reason and the arm keeps going.
    """
    if nli is None:
        return {
            "available": False,
            "reason": nli_unavailable_reason or "nli_unavailable",
        }
    from lib.retrieval.groundedness import score_groundedness

    try:
        results = list(
            retrieve_fn(
                repo_root, course_slug, qtext, engine=engine, limit=limit
            )
        )
    except Exception as exc:  # noqa: BLE001 — per-question isolation
        return {
            "available": False,
            "reason": f"retrieve_error: {type(exc).__name__}: {exc}"[:200],
        }

    # The retrieved passages ARE the evidence pool — the base model emitted no
    # citations, so there is no cited/uncited split (cited_chunk_ids omitted).
    try:
        report = score_groundedness(answer_text or "", results, nli=nli)
    except Exception as exc:  # noqa: BLE001 — per-question isolation
        return {
            "available": False,
            "reason": f"score_error: {type(exc).__name__}: {exc}"[:200],
        }
    return report.to_dict()


def _probe_decision_signal(probe_block: Optional[Dict[str, Any]]) -> str:
    """Render the out-of-scope confabulation-axis signal for the base rationale.

    Interpolates the refusal-probe pass's answered / declined / errored counts +
    confabulation_rate so the per-question capture records the probe outcome
    without a separate call site. An empty / absent probe set reads as such (the
    axis was n/a — no fabricated rate).
    """
    if not isinstance(probe_block, dict) or not probe_block.get("n_probes"):
        return (
            "out-of-scope confabulation axis n/a (no refusal probes loaded for "
            "this course)"
        )
    rate = probe_block.get("confabulation_rate")
    rate_str = f"{rate:.4f}" if isinstance(rate, (int, float)) else "n/a"
    return (
        "out-of-scope confabulation axis: the raw model (no refusal machinery) "
        f"answered {probe_block.get('answered', 0)} / declined "
        f"{probe_block.get('declined', 0)} / errored "
        f"{probe_block.get('errored', 0)} of {probe_block.get('n_probes', 0)} "
        f"deliberately-unanswerable refusal probes (confabulation_rate "
        f"{rate_str})"
    )


def _emit_base_decision(
    capture: Optional[Any],
    *,
    course_slug: str,
    question_id: str,
    model_id: Optional[str],
    max_tokens: int,
    n_key_points: int,
    nli_available: bool,
    nli_unavailable_reason: Optional[str],
    probe_block: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit one ``base_model_eval_call`` decision per base-arm question.

    Rationale interpolates dynamic per-call signals (course, question id, model,
    max_tokens, key-point count, the hallucination-axis signal: whether NLI
    groundedness scoring is wired and, if not, why, AND the out-of-scope
    confabulation-axis signal: the refusal-probe pass's answered/declined/errored
    counts + confabulation_rate) so the capture is replayable post-hoc — never a
    static boilerplate string. NO new call site: the probe-pass counts ride on
    this existing per-question decision. Best-effort: a capture failure is
    swallowed so the eval is not aborted by a logging error (the underlying
    client also emits its own ``llm_chat_call`` event).
    """
    if capture is None:
        return
    if nli_available:
        groundedness_signal = (
            "the answer is ALSO scored for hallucination (unsupported-vs-course-"
            "corpus) against the same top-k passages the grounded pipeline would "
            "retrieve, via the shared score_groundedness v2 NLI machinery"
        )
    else:
        groundedness_signal = (
            "the hallucination axis is n/a for this run (NLI unavailable: "
            f"{nli_unavailable_reason or 'nli_unavailable'}) — no rates "
            "fabricated"
        )
    probe_signal = _probe_decision_signal(probe_block)
    try:
        capture.log_decision(
            decision_type="base_model_eval_call",
            decision=(
                f"base-arm probe course={course_slug} q={question_id} "
                f"model={model_id} max_tokens={max_tokens}"
            ),
            rationale=(
                f"BASE eval arm (no retrieval, no refusal scaffolding): probing "
                f"unaided model knowledge for question {question_id!r} of course "
                f"{course_slug!r} against {n_key_points} expected key point(s); "
                f"model={model_id}, max_tokens={max_tokens}, temperature="
                f"{_BASE_TEMPERATURE}. Scored with the shared "
                f"score_key_point_coverage machinery for honest cross-arm "
                f"comparison; {groundedness_signal}; {probe_signal}."
            ),
            alternatives_considered=[
                "grounded arm (retrieval + refusal): measures the combined "
                "system, not unaided knowledge",
                "retrieval-only arm: measures the extractive ceiling, no model "
                "knowledge",
            ],
        )
    except Exception:  # noqa: BLE001 — capture must never abort the eval
        pass


# --------------------------------------------------------------------------- #
# RETRIEVAL arm — retrieval only, no LLM
# --------------------------------------------------------------------------- #

def run_retrieval_arm(
    repo_root: Path,
    course_slug: str,
    *,
    engine: str = "semantic",
    limit: int = 8,
    retrieve_fn: Optional[Any] = None,
    gold: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure the extractive ceiling: does retrieval surface the expected
    content, and does it rank the gold-relevant passage into top-k?

    For each gold question: retrieve top-k (same engine/limit as the grounded
    run), score ``score_key_point_coverage`` over the CONCATENATED retrieved
    passage bodies (the extractive ceiling — no LLM), and record hit@k / top-1
    hit of the gold ``relevant_passages`` chunk ids. Latency is retrieval
    latency only. NO model call (no capture).

    ``retrieve_fn`` is injectable for tests with signature
    ``(libv2_root, course_slug, query, *, engine, limit) -> [results]`` where
    each result duck-types ``.chunk_id`` + ``.text`` (the live
    :func:`lib.retrieval.grounded_answer._retrieve` shape). When ``None`` the
    real retriever is used.
    """
    repo_root = Path(repo_root)
    if gold is None:
        gold = _load_verified_gold(repo_root, course_slug)
    questions = _gold_questions(gold)

    if retrieve_fn is None:
        from lib.retrieval.grounded_answer import _libv2_root, _retrieve

        libv2_root = _libv2_root(repo_root)

        def retrieve_fn(  # type: ignore[misc]
            _root, slug, query, *, engine, limit
        ):
            return _retrieve(libv2_root, slug, query, engine=engine, limit=limit)

    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    kp_total = 0
    kp_covered = 0
    scored_questions = 0
    hit_at_k = 0
    hit_top1 = 0
    primary_questions = 0  # questions carrying a primary relevant passage

    for q in questions:
        qid = str(q.get("question_id", ""))
        qtext = str(q.get("question_text", ""))
        key_points = _question_key_points(q)
        all_rel, primary_rel = _relevant_chunk_ids(q)
        # Hit metrics use the PRIMARY relevant set when present, else all
        # relevant — the gold-set "the passage that answers this" pin.
        target_ids = primary_rel or all_rel

        t0 = time.monotonic()
        try:
            results = list(
                retrieve_fn(
                    repo_root, course_slug, qtext, engine=engine, limit=limit
                )
            )
        except Exception as exc:  # noqa: BLE001 — per-question isolation
            latency_ms = (time.monotonic() - t0) * 1000.0
            latencies.append(latency_ms)
            rows.append(
                {
                    "question_id": qid,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                    "n_retrieved": 0,
                    "hit_at_k": False,
                    "hit_top1": False,
                    "key_point_coverage": _coverage_fields(None),
                    "latency_ms": latency_ms,
                }
            )
            continue
        latency_ms = (time.monotonic() - t0) * 1000.0
        latencies.append(latency_ms)

        retrieved_ids = [str(getattr(r, "chunk_id", "")) for r in results]
        passage_texts = [str(getattr(r, "text", "") or "") for r in results]
        concatenated = "\n\n".join(t for t in passage_texts if t)

        q_hit_at_k = bool(target_ids) and any(
            cid in target_ids for cid in retrieved_ids
        )
        q_hit_top1 = bool(target_ids) and (
            retrieved_ids[0] in target_ids if retrieved_ids else False
        )
        if target_ids:
            primary_questions += 1
            if q_hit_at_k:
                hit_at_k += 1
            if q_hit_top1:
                hit_top1 += 1

        cov = None
        if key_points:
            scored_questions += 1
            cov = score_key_point_coverage(concatenated, key_points)
            if cov is not None:
                kp_total += cov.total
                kp_covered += cov.covered

        rows.append(
            {
                "question_id": qid,
                "n_retrieved": len(retrieved_ids),
                "hit_at_k": q_hit_at_k,
                "hit_top1": q_hit_top1,
                "key_point_coverage": _coverage_fields(cov),
                "latency_ms": latency_ms,
            }
        )

    return {
        "arm": ARM_RETRIEVAL,
        "model_id": None,  # no LLM
        "retrieval": True,
        "engine": engine,
        "limit": limit,
        "questions_scored": scored_questions,
        # Extractive ceiling: key-point coverage over concatenated passages.
        "key_point_coverage": {
            "total_key_points": kp_total,
            "covered_key_points": kp_covered,
            "coverage_rate": (kp_covered / kp_total) if kp_total else None,
        },
        "primary_relevant_hit": {
            "questions": primary_questions,
            "hit_at_k": hit_at_k,
            "hit_top1": hit_top1,
            "hit_at_k_rate": (
                (hit_at_k / primary_questions) if primary_questions else None
            ),
            "hit_top1_rate": (
                (hit_top1 / primary_questions) if primary_questions else None
            ),
        },
        "latency_ms": {
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
        },
        "questions": rows,
    }


# --------------------------------------------------------------------------- #
# GROUNDED arm — delegate to the existing harness, unchanged
# --------------------------------------------------------------------------- #

def run_grounded_arm(
    repo_root: Path,
    course_slug: str,
    *,
    engine: str = "semantic",
    limit: int = 8,
    write: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the GROUNDED arm by delegating to ``run_grounded_eval`` unchanged.

    Passes ``write`` straight through so the grounded arm still writes its own
    ``grounded_answer_eval_<ts>.json`` report + review sample in addition to the
    scorecard (the staleness test reads that artifact, so it must keep landing).
    Returns the full grounded report dict (the scorecard embeds it verbatim).
    """
    return run_grounded_eval(
        repo_root,
        course_slug,
        engine=engine,
        limit=limit,
        write=write,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Comparison block — shared axes only
# --------------------------------------------------------------------------- #

def _grounded_out_of_scope_rate(grounded: Dict[str, Any]) -> Optional[float]:
    """Derive the GROUNDED arm's answered-instead-of-refused rate from its
    headline refusal block.

    The grounded comparable to the base ``confabulation_rate``: the share of
    refusal probes the pipeline FAILED to refuse, i.e.
    ``(n_probes - refused) / n_probes = 1 - refusal_recall``. Labeled clearly
    (answered-instead-of-refused) because even a grounded "answer" on a probe
    still passes citation gates / disclaimers — NOT the pure invention a base
    answer is. ``None`` when the refusal block carries no probes / no recall
    (never a fabricated rate).
    """
    headline = grounded.get("headline", {}) if isinstance(grounded, dict) else {}
    refusal = headline.get("refusal", {}) or {}
    n_probes = refusal.get("n_probes")
    recall = refusal.get("refusal_recall")
    if not isinstance(n_probes, int) or n_probes <= 0:
        return None
    if not isinstance(recall, (int, float)):
        return None
    return 1.0 - float(recall)


def _grounded_shared_axes(grounded: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the shared comparison axes out of a grounded report dict."""
    headline = grounded.get("headline", {}) if isinstance(grounded, dict) else {}
    kp = headline.get("key_point_coverage", {}) or {}
    questions = grounded.get("questions", []) or []
    answered = sum(
        1
        for r in questions
        if str(r.get("status", "")) in ("answered", "answered_with_warnings")
    )
    declined = sum(
        1
        for r in questions
        if str(r.get("status", "")).startswith("refused")
    )
    return {
        "key_point_coverage_rate": kp.get("coverage_rate"),
        "answered": answered,
        "declined": declined,
        # Hallucination axis: the grounded arm's per-question mean unsupported-
        # vs-corpus rate (the SAME metric the BASE arm now computes). None on a
        # report where NLI was unavailable / nothing was scored.
        "unsupported_claim_rate": headline.get("unsupported_claim_rate"),
        # Out-of-scope confabulation axis (answered-instead-of-refused), derived
        # from the headline refusal block as 1 - refusal_recall. Labeled clearly:
        # a grounded probe "answer" still passes citation gates / disclaimers,
        # unlike a base answer (pure invention).
        "out_of_scope_confabulation_rate": _grounded_out_of_scope_rate(grounded),
        # UNDILUTED claim-level rate is NOT derivable from the persisted grounded
        # report (per-question rows carry groundedness_rate but NOT per-question
        # unsupported_count / scored_count), so it reads None with a reason rather
        # than an approximation — anti-silent-degradation.
        "claim_level_unsupported_rate": None,
        "claim_level_unsupported_rate_reason": (
            "not derivable from persisted grounded report (per-question rows "
            "lack unsupported_count / scored_count); base arm carries the "
            "undiluted aggregate in its groundedness block"
        ),
        "latency_ms": headline.get("latency_ms", {}),
    }


def _arm_unsupported_rate(arm_block: Dict[str, Any]) -> Optional[float]:
    """Pull a base / grounded arm block's unsupported-vs-corpus rate (or None).

    BASE: from its ``groundedness`` sub-block (the hallucination axis added by
    :func:`run_base_arm`). GROUNDED: from its ``headline``. Returns ``None`` when
    the axis is n/a (NLI unavailable / nothing scored) so the comparison reads
    n/a rather than fabricating a rate. The RETRIEVAL arm has no answer claims
    by construction — it is handled separately ("—").
    """
    if not isinstance(arm_block, dict):
        return None
    if arm_block.get("arm") == ARM_GROUNDED or "headline" in arm_block:
        headline = arm_block.get("headline", {}) or {}
        rate = headline.get("unsupported_claim_rate")
        return float(rate) if isinstance(rate, (int, float)) else None
    grounded = arm_block.get("groundedness", {}) or {}
    if not grounded.get("available"):
        return None
    rate = grounded.get("unsupported_claim_rate")
    return float(rate) if isinstance(rate, (int, float)) else None


def _hallucination_reduction(
    base_rate: Optional[float], grounded_rate: Optional[float]
) -> Dict[str, Any]:
    """Derive the BASE→GROUNDED hallucination-reduction entry.

    ``absolute_reduction = base - grounded``; ``relative_reduction =
    (base - grounded) / base`` guarded for ``base in (0, None)`` and a
    ``None`` grounded rate (either ⇒ ``relative_reduction = None``, the honest
    "can't compute" value — never a fabricated 0/1). Lower rate is better, so a
    POSITIVE reduction means the grounded pipeline fabricates LESS than the base
    model — the headline win this axis exists to show.
    """
    absolute: Optional[float] = None
    relative: Optional[float] = None
    if base_rate is not None and grounded_rate is not None:
        absolute = base_rate - grounded_rate
        if base_rate:  # guard base == 0 (no division) — relative is undefined
            relative = absolute / base_rate
    return {
        "base_rate": base_rate,
        "grounded_rate": grounded_rate,
        "absolute_reduction": absolute,
        "relative_reduction": relative,
        # GAP 2: name the per-question-mean dilution convention the rates use, and
        # point at the undiluted aggregate that lives on the base groundedness
        # block. Both arms' unsupported_claim_rate credit an all-computational
        # answer as a 0.0; that convention is KEPT (the pinned grounded basis
        # depends on it) and the undiluted view is additive, not a replacement.
        "_note": (
            "per-question mean incl. all-computational answers as 0.0; "
            "claim_level_* fields carry the undiluted aggregate (base arm "
            "groundedness.claim_level_unsupported_rate; grounded claim-level is "
            "None — not derivable from the persisted report)"
        ),
    }


def _base_confabulation_rate(base_block: Dict[str, Any]) -> Any:
    """Pull the base arm's out-of-scope ``confabulation_rate`` (or None).

    From the ``probe_confabulation`` sub-block added by :func:`run_base_arm`.
    Returns ``None`` when no probes were classifiable (axis n/a) so the
    comparison reads n/a rather than fabricating a rate.
    """
    if not isinstance(base_block, dict):
        return None
    probe = base_block.get("probe_confabulation", {}) or {}
    rate = probe.get("confabulation_rate")
    return float(rate) if isinstance(rate, (int, float)) else None


def _base_claim_level_rate(base_block: Dict[str, Any]) -> Any:
    """Pull the base arm's UNDILUTED ``claim_level_unsupported_rate`` (or None)."""
    if not isinstance(base_block, dict):
        return None
    grounded = base_block.get("groundedness", {}) or {}
    rate = grounded.get("claim_level_unsupported_rate")
    return float(rate) if isinstance(rate, (int, float)) else None


def _build_comparison(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Roll the requested arms up onto the shared axes for a side-by-side view.

    Shared axes (every arm that ran): key_point_coverage rate, answered /
    declined counts, latency p50/p95, AND the hallucination axis
    (``unsupported_claim_rate``, unsupported-vs-corpus) — BASE and GROUNDED both
    carry it (apples-to-apples, same scorer), RETRIEVAL prints the sentinel
    ``"—"`` (no answer claims by construction). When BOTH base + grounded ran a
    derived ``hallucination_reduction`` entry ({base_rate, grounded_rate,
    absolute_reduction, relative_reduction, _note}) headlines the fabrication
    delta.

    Two further axes (additive, schema 1.2): ``out_of_scope_confabulation_rate``
    — BASE = its probe ``confabulation_rate`` (raw invention), GROUNDED =
    ``1 - refusal_recall`` (answered-instead-of-refused; still gated/disclaimed),
    RETRIEVAL = ``"—"``; and ``claim_level_unsupported_rate`` — the UNDILUTED
    aggregate, present on BASE (from its groundedness block), ``None`` on GROUNDED
    (not derivable from the persisted report), ``"—"`` on RETRIEVAL.

    Citation-precision + refusal axes deliberately stay in the grounded block —
    they have no base / retrieval counterpart, so surfacing them in the
    comparison would invite an apples-to-oranges read.
    """
    comparison: Dict[str, Dict[str, Any]] = {}

    if ARM_BASE in arms:
        base = arms[ARM_BASE]
        comparison[ARM_BASE] = {
            "key_point_coverage_rate": base["key_point_coverage"][
                "coverage_rate"
            ],
            "answered": base["answered"],
            # Base answers everything — declined is 0 by construction.
            "declined": base["declined"],
            # Hallucination axis: unsupported-vs-corpus (None ⇒ n/a, NLI absent).
            "unsupported_claim_rate": _arm_unsupported_rate(base),
            # UNDILUTED claim-level rate (GAP 2): additive view that does NOT
            # credit all-computational answers as 0.0. None ⇒ n/a (no scored
            # claim).
            "claim_level_unsupported_rate": _base_claim_level_rate(base),
            # Out-of-scope confabulation axis (GAP 1): the raw model's
            # answered-instead-of-refused rate over the refusal probes. None ⇒
            # n/a (no probes classifiable).
            "out_of_scope_confabulation_rate": _base_confabulation_rate(base),
            "latency_ms": base["latency_ms"],
            "note": (
                "qwen only, no retrieval; answers everything by construction "
                "(no refusal machinery); out_of_scope_confabulation_rate is the "
                "share of refusal probes it answered (pure invention)"
            ),
        }

    if ARM_RETRIEVAL in arms:
        retr = arms[ARM_RETRIEVAL]
        comparison[ARM_RETRIEVAL] = {
            "key_point_coverage_rate": retr["key_point_coverage"][
                "coverage_rate"
            ],
            # Retrieval has no "answered" notion (no LLM) — it surfaces passages.
            "answered": None,
            "declined": None,
            # No model claims to score → no hallucination axis by construction.
            # Sentinel "—" (NOT None) so the table prints a dash and a reader
            # never confuses it with "NLI unavailable" (which is None / n/a).
            "unsupported_claim_rate": "—",
            # No answer claims → no undiluted claim-level rate, no probe answers
            # to confabulate. Both print the sentinel "—" (NOT None) by
            # construction, mirroring the hallucination axis.
            "claim_level_unsupported_rate": "—",
            "out_of_scope_confabulation_rate": "—",
            "primary_relevant_hit_at_k_rate": retr["primary_relevant_hit"][
                "hit_at_k_rate"
            ],
            "latency_ms": retr["latency_ms"],
            "note": (
                "retrieval only, no LLM; key_point_coverage is the extractive "
                "ceiling over concatenated passages; no answer claims so no "
                "hallucination / confabulation axis (—)"
            ),
        }

    if ARM_GROUNDED in arms:
        comparison[ARM_GROUNDED] = {
            **_grounded_shared_axes(arms[ARM_GROUNDED]),
            "note": "qwen + retrieval (full pipeline)",
        }

    # Derived hallucination-reduction entry (BASE→GROUNDED). Only meaningful when
    # both arms ran; otherwise omitted (no fabricated comparison).
    if ARM_BASE in arms and ARM_GROUNDED in arms:
        comparison["hallucination_reduction"] = _hallucination_reduction(
            _arm_unsupported_rate(arms[ARM_BASE]),
            _arm_unsupported_rate(arms[ARM_GROUNDED]),
        )

    return comparison


# --------------------------------------------------------------------------- #
# Scorecard assembly
# --------------------------------------------------------------------------- #

def run_scorecard(
    repo_root: Path,
    course_slug: str,
    *,
    arms: Sequence[str] = (ARM_GROUNDED,),
    engine: str = "semantic",
    limit: int = 8,
    base_client: Optional[Any] = None,
    retrieve_fn: Optional[Any] = None,
    capture: Optional[Any] = None,
    write: bool = True,
    output_path: Optional[Path] = None,
    grounded_kwargs: Optional[Dict[str, Any]] = None,
    base_probes: Optional[Sequence[Dict[str, Any]]] = None,
    base_refusal_probes_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the requested ``arms`` and assemble ONE scorecard dict.

    ``arms`` defaults to ``("grounded",)`` so a default invocation is
    behaviourally identical to the legacy grounded-only eval (the grounded arm
    still writes its own report + review sample). When ``write`` is True the
    scorecard is written to ``retrieval_eval/eval_scorecard_<ts>.json``.

    Raises ``RuntimeError`` on any critical gold-set issue (shared fail-closed
    contract) BEFORE any arm runs, so a bad gold set never burns a base-arm LLM
    pass. Raises ``ValueError`` on an unknown / empty arm name.
    """
    repo_root = Path(repo_root)
    requested = _validate_arms(arms)

    # Load + verify the gold set ONCE (fail-closed) and share it across the
    # deterministic arms; the grounded arm re-verifies internally (its own
    # contract) — cheap and keeps run_grounded_eval untouched.
    gold = _load_verified_gold(repo_root, course_slug)

    arm_results: Dict[str, Dict[str, Any]] = {}

    if ARM_BASE in requested:
        arm_results[ARM_BASE] = run_base_arm(
            repo_root,
            course_slug,
            client=base_client,
            capture=capture,
            gold=gold,
            # Hallucination axis: score base answers against the SAME top-k
            # passages the retrieval / grounded arms see (apples-to-apples).
            engine=engine,
            limit=limit,
            retrieve_fn=retrieve_fn,
            # Out-of-scope confabulation axis: injectable probe set (tests);
            # otherwise the base arm resolves the on-disk refusal_probes.json.
            probes=base_probes,
            refusal_probes_path=base_refusal_probes_path,
        )

    if ARM_RETRIEVAL in requested:
        arm_results[ARM_RETRIEVAL] = run_retrieval_arm(
            repo_root,
            course_slug,
            engine=engine,
            limit=limit,
            retrieve_fn=retrieve_fn,
            gold=gold,
        )

    if ARM_GROUNDED in requested:
        gk = dict(grounded_kwargs or {})
        gk.setdefault("capture", capture)
        arm_results[ARM_GROUNDED] = run_grounded_arm(
            repo_root,
            course_slug,
            engine=engine,
            limit=limit,
            write=write,  # grounded arm writes its own report (staleness test)
            **gk,
        )

    scorecard: Dict[str, Any] = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "course_slug": course_slug,
        "engine": engine,
        "limit": limit,
        "arms_run": [a for a in ALL_ARMS if a in arm_results],
        "arms": arm_results,
        "comparison": _build_comparison(arm_results),
        "generated_at": _utcnow_iso(),
    }

    if write:
        course_dir = _course_dir(repo_root, course_slug)
        eval_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        eval_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = (
            Path(output_path)
            if output_path is not None
            else eval_dir / f"{SCORECARD_FILENAME_PREFIX}{ts}.json"
        )
        out.write_text(
            json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8"
        )
        scorecard["_written"] = {"scorecard_path": str(out)}

    return scorecard


def _validate_arms(arms: Sequence[str]) -> List[str]:
    """Normalize + validate the requested arm list (order-stable, de-duped)."""
    requested: List[str] = []
    for a in arms:
        name = str(a).strip().lower()
        if name not in ALL_ARMS:
            raise ValueError(
                f"unknown eval arm {a!r}; valid arms: {', '.join(ALL_ARMS)}"
            )
        if name not in requested:
            requested.append(name)
    if not requested:
        raise ValueError("no eval arms requested")
    return requested


# --------------------------------------------------------------------------- #
# Aligned stdout table
# --------------------------------------------------------------------------- #

def _fmt_rate(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_count(value: Any) -> str:
    return "—" if value is None else str(value)


def format_scorecard_table(scorecard: Dict[str, Any]) -> str:
    """Render the comparison block as an aligned, three-arm side-by-side table.

    Columns are the arms that ran; rows are the shared axes. Grounded-only axes
    are summarised in a trailing note line, not inlined into the comparison grid
    (they have no base / retrieval counterpart).
    """
    comparison = scorecard.get("comparison", {}) or {}
    arms_order = [a for a in ALL_ARMS if a in comparison]
    headers = {
        ARM_BASE: "BASE (qwen)",
        ARM_RETRIEVAL: "RETRIEVAL",
        ARM_GROUNDED: "GROUNDED",
    }

    axis_rows = [
        ("key_point_coverage", "key_point_coverage_rate", _fmt_rate),
        ("answered", "answered", _fmt_count),
        ("declined", "declined", _fmt_count),
        ("retrieval_hit@k", "primary_relevant_hit_at_k_rate", _fmt_rate),
        # Hallucination axis: lower is better. RETRIEVAL prints "—" (its
        # comparison value is the literal sentinel); BASE / GROUNDED print the
        # rate, or "—" when n/a (NLI unavailable / nothing scored → None).
        (
            "hallucination (unsupported-vs-corpus)",
            "unsupported_claim_rate",
            _fmt_rate,
        ),
        # Out-of-scope confabulation axis (GAP 1): lower is better. BASE = raw
        # invention rate on the refusal probes; GROUNDED = answered-instead-of-
        # refused (1 - refusal_recall, still gated/disclaimed); RETRIEVAL = "—".
        (
            "out_of_scope_confab (answered-instead-of-refused)",
            "out_of_scope_confabulation_rate",
            _fmt_rate,
        ),
        ("latency_p50_ms", None, None),
        ("latency_p95_ms", None, None),
    ]

    label_w = max(len(label) for label, _, _ in axis_rows)
    col_w = max(12, *(len(headers[a]) for a in arms_order)) if arms_order else 12

    lines: List[str] = []
    title = f"Eval scorecard — {scorecard.get('course_slug', '?')} " \
            f"(engine={scorecard.get('engine', '?')})"
    lines.append(title)
    lines.append("=" * len(title))

    header_cells = " ".join(headers[a].ljust(col_w) for a in arms_order)
    lines.append(f"{'axis'.ljust(label_w)}  {header_cells}")
    lines.append(f"{'-' * label_w}  {' '.join('-' * col_w for _ in arms_order)}")

    for label, key, fmt in axis_rows:
        cells: List[str] = []
        for a in arms_order:
            block = comparison.get(a, {})
            if key is None:
                lat = block.get("latency_ms", {}) or {}
                pct = "p50" if "p50" in label else "p95"
                cells.append(_fmt_rate(lat.get(pct)).ljust(col_w))
            else:
                cells.append((fmt or _fmt_rate)(block.get(key)).ljust(col_w))
        lines.append(f"{label.ljust(label_w)}  {' '.join(cells)}")

    # One-line hallucination-reduction summary (BASE→GROUNDED), when derived.
    reduction = comparison.get("hallucination_reduction")
    if isinstance(reduction, dict):
        base_r = reduction.get("base_rate")
        grounded_r = reduction.get("grounded_rate")
        abs_r = reduction.get("absolute_reduction")
        rel_r = reduction.get("relative_reduction")
        rel_str = (
            f"{rel_r * 100:.1f}%" if isinstance(rel_r, (int, float)) else "n/a"
        )
        lines.append("")
        lines.append(
            "hallucination reduction (BASE→GROUNDED, unsupported-vs-corpus): "
            f"{_fmt_rate(base_r)} → {_fmt_rate(grounded_r)}  "
            f"(absolute {_fmt_rate(abs_r)}, relative {rel_str})"
        )

    # Trailing per-arm notes + grounded-only axes pointer.
    lines.append("")
    for a in arms_order:
        note = (comparison.get(a, {}) or {}).get("note")
        if note:
            lines.append(f"  {headers[a]}: {note}")
    if ARM_GROUNDED in scorecard.get("arms", {}):
        lines.append(
            "  grounded-only axes (citations / groundedness / refusal) are in "
            "the grounded arm block"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI (__main__)
# --------------------------------------------------------------------------- #

def _parse_arms(raw: str) -> List[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.retrieval.eval_arms",
        description=(
            "Three-arm eval scorecard (BASE: qwen only / RETRIEVAL: retrieval "
            "only / GROUNDED: qwen + retrieval). Writes a timestamped "
            "retrieval_eval/eval_scorecard_<ts>.json + prints an aligned table. "
            "Default --arms grounded is byte-compatible with the legacy eval."
        ),
    )
    parser.add_argument("--course", required=True, help="course slug")
    parser.add_argument(
        "--arms",
        default=ARM_GROUNDED,
        help=(
            "comma-separated arms to run: base,retrieval,grounded "
            "(default: grounded)"
        ),
    )
    parser.add_argument(
        "--engine",
        default="semantic",
        help="retrieval engine (lexical | semantic | hybrid-rrf)",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--repo-root",
        default=None,
        help="repo root (default: auto-detect from this file)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    args = _build_arg_parser().parse_args(argv)
    repo_root = (
        Path(args.repo_root)
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    try:
        arms = _validate_arms(_parse_arms(args.arms))
    except ValueError as exc:
        print(f"invalid --arms: {exc}", file=sys.stderr)
        return 2

    # Import-guard the grounded pipeline error type only when the grounded arm
    # is requested (the base / retrieval arms don't need it).
    from lib.retrieval.grounded_eval import PipelineUnavailable

    try:
        scorecard = run_scorecard(
            repo_root,
            args.course,
            arms=arms,
            engine=args.engine,
            limit=args.limit,
            write=True,
        )
    except PipelineUnavailable as exc:
        print(f"grounded-answer pipeline unavailable: {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"eval scorecard refused: {exc}", file=sys.stderr)
        return 2

    print(format_scorecard_table(scorecard))
    written = scorecard.get("_written", {})
    if written:
        print(f"\nscorecard: {written.get('scorecard_path')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCORECARD_SCHEMA_VERSION",
    "SCORECARD_FILENAME_PREFIX",
    "ALL_ARMS",
    "ARM_BASE",
    "ARM_RETRIEVAL",
    "ARM_GROUNDED",
    "run_base_arm",
    "run_retrieval_arm",
    "run_grounded_arm",
    "run_scorecard",
    "format_scorecard_table",
    "main",
]
