"""Worker W2.E — TrainingPairPromotionValidator.

GPT Feedback v2 Wave 2 net-new validator. Per-pair, post-emit, pre-write
filter that gates every accepted instruction / preference pair before it
lands in ``training_specs/instruction_pairs.jsonl`` /
``preference_pairs.jsonl``. Closes the seven hard-rejection paths cited
in the GPT critique (lines 225-233):

1. **placeholder_residue** — re-applies the 13-regex
   ``ASSESSMENT_PLACEHOLDER_PATTERNS`` from
   :mod:`lib.validators.assessment` against every prompt / completion /
   chosen / rejected surface. Zero tolerance.
2. **unsupported_answer** — cosine similarity between the answer surface
   (``completion`` for instruction pairs, ``chosen`` for preference
   pairs) and the cited source-chunk text. Default floor
   ``min_answer_support_score=0.40``.
3. **weak_distractor** (preference only) — cosine semantic distinctness
   between ``chosen`` and ``rejected``. Default floor
   ``dpo_min_distractor_distinctness=0.40``.
4. **unanswerable_stem** — content-token Jaccard overlap of the
   ``prompt`` against the cited chunk text below
   ``min_prompt_chunk_jaccard``. Reuses the canonical
   ``_content_tokens`` + ``_jaccard`` helpers from
   :mod:`lib.validators.assessment_retrieval_grounding` so the two
   grounding axes share one tokenisation contract. As of the
   2026-06-09 RDF/SHACL calibration corpus recalibration the default floor is
   ``0.0``: the strict ``<`` comparison can never fire, so the reject
   arm is retired and ``prompt_chunk_jaccard`` is now an audit-stamp-
   only signal (paraphrase prompts are instructed to reword the
   source, so this axis was measured non-separating on the real
   corpus). The criterion stays wired so an operator can re-arm it by
   passing an explicit positive ``min_prompt_chunk_jaccard``.
5. **source_free_generation** — ``source_chunk_id`` is missing or empty.
   The schema's ``trainable``-conditional already requires the field;
   the validator catches it loudly before the pair is checkpointed.
6. **low_bloom_alignment** — Bloom classification of
   ``prompt + completion`` (or ``prompt + chosen`` for preference pairs)
   is below the declared minimum ``bloom_level``, AND the classifier
   winner's confidence exceeds ``bloom_alignment_min_confidence``. A
   higher observed level satisfies a lower declared level.

   **The reject arm is RETIRED**, exactly as criterion 4 above is: the
   default floor is ``1.01`` and scores are softmax probabilities bounded
   by 1.0, so the strict ``>`` can never fire. ``observed_bloom`` /
   ``bloom_alignment`` are still stamped on every pair for audit and
   calibration. The backing classifier is not trustworthy enough to
   discard training data — the three-member BERT ensemble degrades to the
   ``unknown`` sentinel (per-member dispatch is an unfinished
   placeholder) and classification falls through to the DeBERTa zero-shot
   head, which is itself pending replacement. Re-arm by passing an
   explicit ``bloom_alignment_min_confidence`` in [0.0, 1.0] once a
   purpose-trained Bloom classifier lands.

   NB: ``bloom_alignment`` is still computed against
   :data:`lib.validators.bloom.classifier_disagreement._DISAGREEMENT_CONFIDENCE_FLOOR`
   (0.40) — that governs the *audit stamp*, not the reject.
7. **generic_rationale** — heuristic richness score on the ``rationale``
   field (unique-token-count / total-token-count). Default floor
   ``min_rationale_richness_score=0.30``.

Filter mechanics (call site, in
:mod:`Trainforge.synthesize_training`)::

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk,
    )
    pair.update(new_fields)
    if status == "rejected":
        stats.<kind>_pairs_rejected += 1
        stats.rejected_reasons[f"<kind>:promotion:{reason}"] += 1
        stats.dropped_count += 1
        continue  # do NOT append to records, sidecar, or checkpoint
    pair["promotion_status"] = status   # "validated"

Graceful degrade. When the optional ``[embedding]`` extras are absent,
criteria 2 + 3 (which need a sentence embedder for cosine similarity)
fall back to a Jaccard surrogate. The validator emits exactly one
warning-severity ``EMBEDDING_DEPS_MISSING`` :class:`GateIssue` per
:meth:`validate_pair` invocation when the fallback fires, so the audit
trail records the silent-degrade. Strict mode via
``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` re-enables fail-closed (the
embedder loader raises and the validator surfaces the failure as
``unsupported_answer`` / ``weak_distractor`` per the existing reject
path). Mirrors the exact pattern at
:mod:`lib.validators.objective_assessment_similarity` line 307.

Decision capture. Every :meth:`validate_pair` invocation fires exactly
one ``training_pair_promotion_check`` event. Rationale interpolates the
pair's ``chunk_id``, the resolved ``promotion_status``, the matched
``rejection_reason`` (when present), the threshold of every criterion
that fired, and the kind ("instruction" / "preference"). Captures are
replayable post-hoc via the ``runtime/training-captures/`` JSONL stream.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.validators.assessment import ASSESSMENT_PLACEHOLDER_PATTERNS
from lib.validators.assessment_retrieval_grounding import (
    _content_tokens,
    _jaccard,
)
from lib.validators.bloom.classifier_disagreement import (
    _DISAGREEMENT_CONFIDENCE_FLOOR,
)
from lib.ontology.bloom import BLOOM_LEVELS
from Trainforge.generators.staged.objective_contract import (
    content_sha256 as _objective_content_sha256,
    reconcile_completion_execution,
    release_content_sha256,
)

_ANSWER_SUPPORT_AUTHORITY_CONTRACT = (
    "ed4all.complete-claim-proof-answer-support.v1"
)
_OBJECTIVE_EXECUTION_AUTHORITY_CONTRACT = (
    "ed4all.objective-execution-answer-support.v1"
)


def _objective_execution_answer_support_proof(
    pair: Dict[str, Any], *, answer: str, source_chunk_id: str,
    authorized_private_sidecar: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Reconcile the private worked-step proof and return a public authority."""
    candidate = pair.get("_objective_execution_candidate")
    sidecar = authorized_private_sidecar
    if not isinstance(candidate, dict) or not isinstance(sidecar, dict):
        return None
    audits = candidate.get("pair_objective_execution")
    records = candidate.get("execution_records")
    requirements = sidecar.get("requirements")
    objective_card = sidecar.get("objective_card")
    if (
        candidate.get("pair_objective_execution_pass_rate") != 1.0
        or candidate.get("claim_support_rate") != 1.0
        or candidate.get("claim_contradicted_rate") != 0.0
        or not isinstance(audits, list)
        or len(audits) != 1
        or not isinstance(audits[0], dict)
        or not isinstance(records, list)
        or not isinstance(requirements, list)
        or not isinstance(objective_card, dict)
    ):
        return None
    audit = audits[0]
    if (
        audit.get("completion_only") is not True
        or audit.get("status") != "delivered"
        or audit.get("requirement_pass_rate") != 1.0
        or audit.get("sidecar_sha256") != sidecar.get("sidecar_sha256")
    ):
        return None
    source_bindings = sidecar.get("source_bindings")
    claims = sidecar.get("claims")
    if not isinstance(source_bindings, list) or not isinstance(claims, list):
        return None
    if source_chunk_id not in {
        item.get("source_chunk_id")
        for item in source_bindings if isinstance(item, dict)
    }:
        return None
    if any(
        not isinstance(item, dict)
        or source_chunk_id not in (item.get("source_chunk_ids") or [])
        for item in claims
    ):
        return None
    requirement_contract = {
        "contract_version": (
            "ed4all.objective-requirement-normalization.v1"
        ),
        "objective_id": str(objective_card.get("id") or "").lower(),
        "cognitive_task_type": None,
        "result_required": any(
            isinstance(item, dict) and item.get("kind") == "result"
            for item in requirements
        ),
        "requirements": requirements,
        "objective_contract_sha256": sidecar.get(
            "objective_contract_sha256"
        ),
    }
    try:
        recomputed = reconcile_completion_execution(
            completion=answer,
            requirement_contract=requirement_contract,
            execution_records=records,
            sidecar=sidecar,
            release_pair=pair,
            validator_fingerprint=str(
                audit.get("validator_fingerprint") or ""
            ),
        )
    except ValueError:
        return None
    if recomputed != audit:
        return None
    proof_material = {
        "contract": _OBJECTIVE_EXECUTION_AUTHORITY_CONTRACT,
        "answer_sha256": _sha256_text(answer),
        "source_chunk_id": source_chunk_id,
        "release_content_sha256": release_content_sha256(pair),
        "objective_contract_sha256": audit["objective_contract_sha256"],
        "sidecar_sha256": audit["sidecar_sha256"],
        "validator_fingerprint": audit["validator_fingerprint"],
        "requirements": [{
            "requirement_id": item.get("requirement_id"),
            "completion_spans": item.get("completion_spans"),
            "proof_sha256": item.get("proof_sha256"),
        } for item in records],
    }
    return {
        **proof_material,
        "proof_sha256": _objective_content_sha256(proof_material),
        "claim_count": len(claims),
        "requirement_count": len(requirements),
        "claim_support_rate": 1.0,
        "claim_contradicted_rate": 0.0,
    }


def _replay_objective_execution_public_authority(
    pair: Dict[str, Any], *, answer: str, source_chunk_id: str,
) -> Optional[Dict[str, Any]]:
    """Replay the release-safe projection after private sidecar stripping."""
    candidate = pair.get("_objective_execution_candidate")
    if not isinstance(candidate, dict):
        return None
    authority = candidate.get("answer_support_authority")
    audits = candidate.get("pair_objective_execution")
    records = candidate.get("execution_records")
    if (
        not isinstance(authority, dict)
        or authority.get("contract")
        != _OBJECTIVE_EXECUTION_AUTHORITY_CONTRACT
        or not isinstance(audits, list)
        or len(audits) != 1
        or not isinstance(records, list)
    ):
        return None
    audit = audits[0]
    proof_material = {
        "contract": _OBJECTIVE_EXECUTION_AUTHORITY_CONTRACT,
        "answer_sha256": _sha256_text(answer),
        "source_chunk_id": source_chunk_id,
        "release_content_sha256": release_content_sha256(pair),
        "objective_contract_sha256": audit.get(
            "objective_contract_sha256"
        ),
        "sidecar_sha256": audit.get("sidecar_sha256"),
        "validator_fingerprint": audit.get("validator_fingerprint"),
        "requirements": [{
            "requirement_id": item.get("requirement_id"),
            "completion_spans": item.get("completion_spans"),
            "proof_sha256": item.get("proof_sha256"),
        } for item in records if isinstance(item, dict)],
    }
    if (
        proof_material != {
            key: authority.get(key) for key in proof_material
        }
        or authority.get("proof_sha256")
        != _objective_content_sha256(proof_material)
    ):
        return None
    return authority


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _complete_claim_proof(
    pair: Dict[str, Any],
    *,
    answer: str,
    source_chunk_id: str,
) -> Optional[Dict[str, Any]]:
    """Return authenticated stronger answer-support proof, or ``None``.

    This deliberately recognizes only the staged v2 projection's complete,
    same-byte realization chain.  Partial NLI, bare aggregate scores, and
    legacy rows never acquire authority and continue through the existing
    answer-support threshold.
    """
    if pair.get("projection_contract") not in {
        "ed4all-dpo-preference.v2",
        "ed4all-sft-chat.v2",
    }:
        return None
    if pair.get("deps_missing") is not False:
        return None
    if pair.get("claim_support_rate") != 1.0:
        return None
    if pair.get("claim_contradicted_rate") != 0.0:
        return None

    provenance = pair.get("provenance")
    if not isinstance(provenance, dict):
        return None
    if provenance.get("source_chunk_id") != source_chunk_id:
        return None
    source_refs = provenance.get("source_refs")
    if (
        not isinstance(source_refs, list)
        or not source_refs
        or any(not isinstance(ref, str) or not ref.strip() for ref in source_refs)
    ):
        return None
    sealed_provenance = dict(provenance)
    declared_provenance_sha = sealed_provenance.pop("provenance_sha256", None)
    if declared_provenance_sha != hashlib.sha256(
        json.dumps(
            sealed_provenance, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest():
        return None
    realization_map = provenance.get("claim_realizations")
    assembly = provenance.get("assembled_realization")
    if not isinstance(realization_map, dict) or not realization_map:
        return None
    if not isinstance(assembly, dict):
        return None
    if (
        assembly.get("contract_version")
        != "ed4all.micro-stage-d-claim-realizations.v3"
    ):
        return None
    sealed_assembly = dict(assembly)
    declared_assembly_sha = sealed_assembly.pop("provenance_sha256", None)
    if declared_assembly_sha != hashlib.sha256(
        json.dumps(
            sealed_assembly, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest():
        return None
    ordered_ids = assembly.get("ordered_claim_ids")
    item_hashes = assembly.get("item_sha256")
    if (
        not isinstance(ordered_ids, list)
        or not isinstance(item_hashes, list)
        or len(ordered_ids) != len(item_hashes)
        or len(ordered_ids) != len(realization_map)
        or len(set(ordered_ids)) != len(ordered_ids)
        or set(ordered_ids) != set(realization_map)
    ):
        return None
    realizations = [realization_map.get(claim_id) for claim_id in ordered_ids]
    if any(not isinstance(text, str) or not text.strip() for text in realizations):
        return None
    if item_hashes != [_sha256_text(text) for text in realizations]:
        return None
    if assembly.get("assembled_sha256") != _sha256_text(answer):
        return None
    if answer != " ".join(realizations):
        return None
    private_sha = assembly.get("private_artifact_sha256")
    if (
        not isinstance(private_sha, str)
        or len(private_sha) != 64
        or provenance.get("synthesis_plan_sha256") != private_sha
        or pair.get("synthesis_plan_sha256") != private_sha
    ):
        return None

    per_claim = pair.get("per_claim_support")
    if not isinstance(per_claim, list) or len(per_claim) != len(realizations):
        return None
    for expected, proof in zip(realizations, per_claim):
        if not isinstance(proof, dict):
            return None
        if proof.get("sentence") != expected or proof.get("outcome") != "entailed":
            return None
        entailment = proof.get("entailment")
        contradiction = proof.get("contradiction")
        if (
            isinstance(entailment, bool)
            or isinstance(contradiction, bool)
            or not isinstance(entailment, (int, float))
            or not isinstance(contradiction, (int, float))
            or not math.isfinite(float(entailment))
            or not math.isfinite(float(contradiction))
            or not 0.0 <= float(entailment) <= 1.0
            or not 0.0 <= float(contradiction) <= 1.0
            or entailment < 0.70
            or contradiction >= 0.50
        ):
            return None

    proof_material = {
        "contract": _ANSWER_SUPPORT_AUTHORITY_CONTRACT,
        "source_chunk_id": source_chunk_id,
        "source_refs": source_refs,
        "ordered_claim_ids": ordered_ids,
        "item_sha256": item_hashes,
        "assembled_sha256": assembly["assembled_sha256"],
        "private_artifact_sha256": private_sha,
        "per_claim_support": per_claim,
    }
    proof_sha256 = hashlib.sha256(
        json.dumps(
            proof_material, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "contract": _ANSWER_SUPPORT_AUTHORITY_CONTRACT,
        "proof_sha256": proof_sha256,
        "claim_count": len(ordered_ids),
        "ordered_claim_ids": list(ordered_ids),
        "assembled_sha256": assembly["assembled_sha256"],
        "private_artifact_sha256": private_sha,
        "claim_support_rate": 1.0,
        "claim_contradicted_rate": 0.0,
        "min_entailment": min(
            float(proof["entailment"]) for proof in per_claim
        ),
        "max_contradiction": max(
            float(proof["contradiction"]) for proof in per_claim
        ),
    }

# W-D7 T7.6: thresholds + strict-mode helper extracted into the
# ``_pair_promotion_stages`` private subpackage. Re-exported here so
# existing ``from lib.validators.training_pair_promotion import
# DEFAULT_*`` (and the canonical ``from lib.validators.pair.promotion
# import DEFAULT_*``) keep resolving without change.
from lib.validators._pair_promotion_stages.thresholds import (  # noqa: F401
    DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS,
    DEFAULT_MIN_ANSWER_SUPPORT_SCORE,
    DEFAULT_MIN_PROMPT_CHUNK_JACCARD,
    DEFAULT_BLOOM_ALIGNMENT_MIN_CONFIDENCE,
    DEFAULT_MIN_RATIONALE_RICHNESS_SCORE,
    _FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS,
    _FALLBACK_MIN_ANSWER_SUPPORT_SCORE,
    _FALLBACK_MIN_PROMPT_CHUNK_JACCARD,
    _RATIONALE_MIN_TOKENS_FOR_AUDIT,
    _RATIONALE_TOKEN_RE,
    _STRICT_EMBEDDINGS_ENV_VAR,
    _TRUTHY_VALUES,
    _is_strict_embeddings_mode,
)

logger = logging.getLogger(__name__)


def _emit_decision(
    capture: Any,
    *,
    pair_kind: str,
    chunk_id: str,
    promotion_status: str,
    rejection_reason: Optional[str],
    answer_support_score: Optional[float],
    distractor_distinctness: Optional[float],
    prompt_chunk_jaccard: Optional[float],
    observed_bloom: Optional[str],
    declared_bloom: Optional[str],
    bloom_alignment: Optional[bool],
    rationale_richness_score: Optional[float],
    placeholder_match: Optional[str],
    bloom_winner_score: Optional[float],
    thresholds: Dict[str, float],
    embedder_strict: bool,
    fallback_to_jaccard: bool,
    answer_support_authority: Optional[Dict[str, Any]],
    answer_support_outcome: Optional[str],
) -> None:
    """Emit one ``training_pair_promotion_check`` decision per pair.

    Rationale interpolates every signal that fed the verdict so an
    operator replaying the audit log can attribute the pass/fail to a
    specific criterion + threshold combination.
    """
    if capture is None:
        return

    decision = (
        promotion_status
        if promotion_status != "rejected"
        else f"rejected:{rejection_reason or 'unknown'}"
    )

    def _fmt(value: Optional[float]) -> str:
        return f"{value:.4f}" if isinstance(value, float) else "n/a"

    rationale = (
        f"Training-pair promotion check ({pair_kind}) on chunk_id="
        f"{chunk_id!r}: status={promotion_status}, "
        f"rejection_reason={rejection_reason or 'none'}, "
        f"answer_support_score={_fmt(answer_support_score)} "
        f"(min={thresholds['min_answer_support_score']:.4f}), "
        f"distractor_distinctness={_fmt(distractor_distinctness)} "
        f"(min={thresholds['dpo_min_distractor_distinctness']:.4f}), "
        f"prompt_chunk_jaccard={_fmt(prompt_chunk_jaccard)} "
        f"(min={thresholds['min_prompt_chunk_jaccard']:.4f}), "
        f"observed_bloom={observed_bloom or 'n/a'}, "
        f"declared_bloom={declared_bloom or 'n/a'}, "
        f"bloom_alignment={bloom_alignment}, "
        f"bloom_winner_score={_fmt(bloom_winner_score)} "
        f"(disagreement_floor="
        f"{_DISAGREEMENT_CONFIDENCE_FLOOR:.4f}), "
        f"rationale_richness_score={_fmt(rationale_richness_score)} "
        f"(min={thresholds['min_rationale_richness_score']:.4f}), "
        f"placeholder_match={placeholder_match or 'none'}, "
        f"embedder_strict_mode={embedder_strict}, "
        f"fallback_to_jaccard={fallback_to_jaccard}."
        f" answer_support_outcome={answer_support_outcome or 'not_scored'}, "
        f"answer_support_authority_contract="
        f"{(answer_support_authority or {}).get('contract', 'none')}, "
        f"answer_support_authority_sha256="
        f"{(answer_support_authority or {}).get('proof_sha256', 'none')}."
    )
    metrics: Dict[str, Any] = {
        "pair_kind": pair_kind,
        "chunk_id": chunk_id,
        "promotion_status": promotion_status,
        "rejection_reason": rejection_reason,
        "answer_support_score": answer_support_score,
        "distractor_distinctness": distractor_distinctness,
        "prompt_chunk_jaccard": prompt_chunk_jaccard,
        "observed_bloom": observed_bloom,
        "declared_bloom": declared_bloom,
        "bloom_alignment": bloom_alignment,
        "bloom_winner_score": bloom_winner_score,
        "rationale_richness_score": rationale_richness_score,
        "placeholder_match": placeholder_match,
        "fallback_to_jaccard": fallback_to_jaccard,
        "embedder_strict_mode": embedder_strict,
        "answer_support_outcome": answer_support_outcome,
        "answer_support_authority": answer_support_authority,
        "thresholds": dict(thresholds),
    }
    try:
        capture.log_decision(
            decision_type="training_pair_promotion_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "training_pair_promotion_check: %s",
            exc,
        )


def _placeholder_match(text: str) -> Optional[str]:
    """Return the regex source of the first matching placeholder pattern,
    or ``None`` when no pattern hits. Single source of truth lives in
    :data:`lib.validators.assessment.ASSESSMENT_PLACEHOLDER_PATTERNS`."""
    if not text:
        return None
    for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def _rationale_richness(text: str) -> Tuple[Optional[float], int]:
    """Compute ``unique_tokens / total_tokens`` on lowercased alphabetic
    content tokens. Returns ``(score, token_count)``. ``score`` is None
    when the token count is below
    :data:`_RATIONALE_MIN_TOKENS_FOR_AUDIT` so the caller can short-
    circuit the gate (rationales that short don't carry enough signal
    to discriminate richness)."""
    if not text:
        return None, 0
    tokens = [t.lower() for t in _RATIONALE_TOKEN_RE.findall(text)]
    if len(tokens) < _RATIONALE_MIN_TOKENS_FOR_AUDIT:
        return None, len(tokens)
    return len(set(tokens)) / len(tokens), len(tokens)


def _cosine_similarity_safe(vec_a: Any, vec_b: Any) -> Optional[float]:
    """Cosine similarity that degrades to ``None`` on any unexpected
    runtime error (e.g. a numpy version mismatch in a very slim
    install). Caller falls back to Jaccard."""
    try:
        from lib.embedding._math import cosine_similarity

        return float(cosine_similarity(vec_a, vec_b))
    except Exception as exc:  # noqa: BLE001
        logger.debug("cosine_similarity raised: %s", exc)
        return None


def _jaccard_overlap(text_a: str, text_b: str) -> float:
    """Jaccard overlap surrogate when embeddings are unavailable.

    Reuses the canonical ``_content_tokens`` + ``_jaccard`` helpers
    from :mod:`lib.validators.assessment_retrieval_grounding` (the
    answer-side grounding gate) so the same tokenisation contract
    drives both gates. The value lives in [0, 1] just like cosine
    similarity, so the same threshold can be applied symmetrically.
    """
    return float(_jaccard(_content_tokens(text_a), _content_tokens(text_b)))


def _semantic_distinctness_jaccard(text_a: str, text_b: str) -> float:
    """Distinctness surrogate ``1 - jaccard_overlap``. Mirrors the
    schema's ``distractor_quality.semantic_distinctness`` definition
    (``1 - cosine_similarity`` in the embedding world; here we
    substitute Jaccard when extras are absent)."""
    return float(1.0 - _jaccard_overlap(text_a, text_b))


class _ZeroShotBloomClassifier:
    """Bloom classifier backed by the pinned process-singleton NLI judge."""

    def classify(self, text: str) -> Dict[str, Any]:
        from lib.classifiers.bloom_zero_shot import zero_shot_bloom

        result = zero_shot_bloom(text)
        if result is None:
            return {
                "winner_level": "unknown",
                "winner_score": 0.0,
                "dispersion": 0.0,
                "per_member": [],
            }
        level, score, distribution = result
        return {
            "winner_level": level,
            "winner_score": score,
            "dispersion": None,
            "per_member": [("deberta-zero-shot", score)],
            "distribution": distribution,
        }


def _classify_bloom(
    *,
    text: str,
    classifier: Any,
) -> Tuple[Optional[str], Optional[float]]:
    """Run ``classifier.classify(text)`` and return
    ``(winner_level, winner_score)``. Both fields are ``None`` when the
    classifier returns the unknown sentinel or when classification
    raises (silent-degrade per the ensemble's contract)."""
    if classifier is None:
        return None, None
    try:
        result = classifier.classify(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("BloomBertEnsemble.classify raised: %s", exc)
        return None, None
    level = result.get("winner_level")
    score = result.get("winner_score")
    if level == "unknown" or not isinstance(level, str):
        # The legacy three-member ensemble may be unavailable (or intentionally
        # retired). Use the already-loaded, SHA-pinned DeBERTa NLI singleton as
        # the production classifier instead of silently stamping Bloom as
        # unverifiable. No additional model weights are introduced.
        try:
            from lib.classifiers.bloom_zero_shot import zero_shot_bloom

            zero_shot = zero_shot_bloom(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Zero-shot Bloom classification failed: %s", exc)
            zero_shot = None
        if zero_shot is None:
            return None, None
        zs_level, zs_score, _ = zero_shot
        return zs_level, float(zs_score)
    return level, float(score) if isinstance(score, (int, float)) else None


class TrainingPairPromotionValidator:
    """Per-pair, post-emit, pre-write filter for synthesis pairs.

    Two surfaces:

    - :meth:`validate_pair` — the in-process call site invoked by
      :mod:`Trainforge.synthesize_training` between
      ``_attach_source_grounding`` success and the buffer / sidecar /
      checkpoint append. Returns
      ``(promotion_status, rejection_reason, new_fields_dict)`` so the
      caller can stamp the new fields on the pair before deciding to
      keep or drop it.
    - :meth:`validate` — the gate-driven call invoked by
      :class:`MCP.hardening.validation_gates.ValidationGateRunner`. Walks
      the on-disk ``instruction_pairs.jsonl`` /
      ``preference_pairs.jsonl`` and confirms every pair carries
      ``promotion_status``. Lightweight post-hoc audit, NOT a re-run of
      the per-pair filter (that would require the chunks-lookup map).

    Constructor mirrors :class:`DistractorPlausibilityValidator`'s
    signature so the gate-runner instantiation pattern in
    :mod:`MCP.hardening.validation_gates` finds the expected kwargs.
    """

    name = "pair_promotion"
    version = "1.0.0"

    def __init__(
        self,
        *,
        min_answer_support_score: float = DEFAULT_MIN_ANSWER_SUPPORT_SCORE,
        dpo_min_distractor_distinctness: float = (
            DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS
        ),
        min_prompt_chunk_jaccard: float = DEFAULT_MIN_PROMPT_CHUNK_JACCARD,
        bloom_alignment_min_confidence: float = (
            DEFAULT_BLOOM_ALIGNMENT_MIN_CONFIDENCE
        ),
        min_rationale_richness_score: float = (
            DEFAULT_MIN_RATIONALE_RICHNESS_SCORE
        ),
        embedder: Optional[Any] = None,
        bloom_classifier: Optional[Any] = None,
    ) -> None:
        self._min_answer_support_score = min_answer_support_score
        self._dpo_min_distractor_distinctness = dpo_min_distractor_distinctness
        self._min_prompt_chunk_jaccard = min_prompt_chunk_jaccard
        self._bloom_alignment_min_confidence = bloom_alignment_min_confidence
        self._min_rationale_richness_score = min_rationale_richness_score
        # Lazy-load on first use so a slim install + test injection
        # both work; mirrors the
        # ``ObjectiveAssessmentSimilarityValidator.__init__`` pattern.
        self._embedder_override = embedder
        self._embedder_resolved: Any = None
        self._embedder_resolution_attempted: bool = False
        self._bloom_classifier_override = bloom_classifier
        self._bloom_classifier_resolved: Any = None
        self._bloom_classifier_resolution_attempted: bool = False

    # ------------------------------------------------------------------ #
    # Lazy resolvers — embedder + bloom classifier
    # ------------------------------------------------------------------ #

    def _resolve_embedder(self) -> Tuple[Any, bool]:
        """Return ``(embedder, raised_strict_error)``.

        ``embedder`` is the resolved :class:`SentenceEmbedder` (or the
        injected override) when extras are present, or ``None`` when the
        loader returned None (extras absent, default mode). When strict
        mode is on and the loader raises :class:`EmbeddingDepsMissing`,
        the error bubbles up and ``raised_strict_error=True`` so the
        caller can surface the failure as a hard reject path. Default
        mode swallows the error silently.
        """
        if self._embedder_override is not None:
            return self._embedder_override, False
        if self._embedder_resolution_attempted:
            return self._embedder_resolved, False
        self._embedder_resolution_attempted = True
        try:
            from lib.embedding.sentence_embedder import try_load_embedder

            self._embedder_resolved = try_load_embedder()
        except Exception as exc:  # noqa: BLE001
            # Strict mode raises EmbeddingDepsMissing; default mode
            # never raises. The exception type lives in the same module
            # so we don't need to import it eagerly.
            if _is_strict_embeddings_mode():
                # Re-raise so the caller fails closed in strict mode.
                raise
            logger.debug(
                "try_load_embedder raised in default mode "
                "(swallowed): %s",
                exc,
            )
            self._embedder_resolved = None
        return self._embedder_resolved, False

    def _resolve_bloom_classifier(self) -> Any:
        """Return the pinned zero-shot Bloom classifier or an override."""
        if self._bloom_classifier_override is not None:
            return self._bloom_classifier_override
        if self._bloom_classifier_resolution_attempted:
            return self._bloom_classifier_resolved
        self._bloom_classifier_resolution_attempted = True
        try:
            self._bloom_classifier_resolved = _ZeroShotBloomClassifier()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Zero-shot Bloom classifier construction raised: %s", exc
            )
            self._bloom_classifier_resolved = None
        return self._bloom_classifier_resolved

    # ------------------------------------------------------------------ #
    # Per-pair filter — call site in synthesize_training.py
    # ------------------------------------------------------------------ #

    def validate_pair(
        self,
        pair: Dict[str, Any],
        *,
        kind: str,
        chunk: Optional[Dict[str, Any]] = None,
        decision_capture: Any = None,
        authorized_private_sidecar: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Audit one pair against the seven hard-rejection criteria.

        Returns ``(promotion_status, rejection_reason, new_fields)``:

        - ``promotion_status`` is ``"validated"`` on pass and
          ``"rejected"`` on any criterion fire.
        - ``rejection_reason`` is the canonical enum string (one of the
          seven ``rejection_reason`` values listed in this module's
          docstring) when ``promotion_status == "rejected"``; ``None``
          otherwise.
        - ``new_fields`` is a dict of all the audit-trail fields the
          validator computed (``answer_support_score``,
          ``observed_bloom``, ``bloom_alignment``,
          ``rationale_richness_score``, plus
          ``distractor_quality`` for preference pairs). The caller
          MUST ``pair.update(new_fields)`` so the audit signals land on
          the pair regardless of pass/fail. On reject, the caller also
          stamps ``rejection_reason``; on pass, the caller stamps
          ``promotion_status="validated"``.
        """
        # Initial thresholds — answer-support + distractor-distinctness
        # may be relaxed below to the Jaccard-fallback floors when the
        # embedder isn't loaded (signal sparser without cosine).
        thresholds: Dict[str, float] = {
            "min_answer_support_score": float(
                self._min_answer_support_score
            ),
            "dpo_min_distractor_distinctness": float(
                self._dpo_min_distractor_distinctness
            ),
            "min_prompt_chunk_jaccard": float(
                self._min_prompt_chunk_jaccard
            ),
            "bloom_alignment_min_confidence": float(
                self._bloom_alignment_min_confidence
            ),
            "min_rationale_richness_score": float(
                self._min_rationale_richness_score
            ),
        }

        new_fields: Dict[str, Any] = {}
        embedder_strict = _is_strict_embeddings_mode()
        chunk_id = str(pair.get("chunk_id") or "")

        prompt = str(pair.get("prompt") or "")
        if kind == "instruction":
            answer = str(pair.get("completion") or "")
            chosen = answer
            rejected = ""
        else:
            chosen = str(pair.get("chosen") or "")
            rejected = str(pair.get("rejected") or "")
            answer = chosen

        rationale_text = str(pair.get("rationale") or "")
        chunk_text = str(chunk.get("text") if isinstance(chunk, dict) else "") or ""

        # ---- Criterion 1: placeholder residue (zero-tolerance) ---- #
        # We check every text surface — placeholder strings sometimes
        # land on rejected/distractor surfaces too.
        placeholder_match: Optional[str] = None
        for surface in (prompt, answer, chosen, rejected, rationale_text):
            placeholder_match = _placeholder_match(surface)
            if placeholder_match is not None:
                break

        # ---- Criterion 5: source-free generation ---- #
        source_chunk_id = str(pair.get("source_chunk_id") or "").strip()

        # ---- Criterion 6: low Bloom alignment (best-effort, even on early reject) ---- #
        bloom_classifier = self._resolve_bloom_classifier()
        declared_bloom = pair.get("bloom_level")
        if not isinstance(declared_bloom, str):
            declared_bloom = None
        # Compose the surface used for classification. For DPO we use
        # ``prompt + chosen`` so the model sees the "right answer" text
        # alongside the question stem.
        bloom_surface = (
            f"{prompt} {answer}".strip()
            if kind == "instruction"
            else f"{prompt} {chosen}".strip()
        )
        observed_bloom, bloom_winner_score = _classify_bloom(
            text=bloom_surface, classifier=bloom_classifier
        )
        from lib.validators.pair.objective_delivery import (
            recompute_complete_objective_bloom_authority,
        )

        objective_bloom_authority = (
            recompute_complete_objective_bloom_authority(pair)
        )
        if objective_bloom_authority is not None:
            observed_bloom = objective_bloom_authority["observed_bloom"]
        bloom_alignment: Optional[bool]
        if (
            observed_bloom in BLOOM_LEVELS
            and declared_bloom in BLOOM_LEVELS
            and bloom_winner_score is not None
            and bloom_winner_score > _DISAGREEMENT_CONFIDENCE_FLOOR
        ):
            # Bloom is an ordered minimum-demand contract: demonstrating a
            # higher cognitive level satisfies a lower declared objective.
            bloom_alignment = (
                BLOOM_LEVELS.index(observed_bloom)
                >= BLOOM_LEVELS.index(declared_bloom)
            )
        else:
            bloom_alignment = None
        new_fields["observed_bloom"] = observed_bloom
        new_fields["bloom_alignment"] = bloom_alignment

        # ---- Criterion 7: rationale richness ---- #
        rationale_score, rationale_token_count = _rationale_richness(
            rationale_text
        )
        # Stamp on the pair regardless of pass/fail (None when too short
        # to score).
        new_fields["rationale_richness_score"] = rationale_score

        # Resolve the embedder once. In strict mode + extras absent,
        # ``_resolve_embedder`` re-raises and the validator MUST propagate.
        # In default mode, embedder=None and we fall back to Jaccard for
        # criteria 2 + 3.
        try:
            embedder, _ = self._resolve_embedder()
        except Exception:
            if embedder_strict:
                raise
            embedder = None
        fallback_to_jaccard = embedder is None
        if fallback_to_jaccard:
            # Relax the cosine-floors to the Jaccard-equivalent floors.
            # The Jaccard signal on short answer / distractor strings is
            # sparser than cosine, so the cosine threshold would
            # false-positive. Semantic intent of each criterion is
            # preserved; only the calibration changes. Same logic for
            # min_prompt_chunk_jaccard — the criterion fires only when
            # the cosine path is available; in fallback mode it
            # collapses to the same Jaccard signal as criterion 2 so
            # deactivating it avoids double-counting (and avoids
            # false-positive rejections of short on-topic prompts where
            # |prompt|=5 tokens, |chunk|=30 tokens cluster at
            # J≈0.03–0.10 even when on-topic).
            thresholds["min_answer_support_score"] = (
                _FALLBACK_MIN_ANSWER_SUPPORT_SCORE
            )
            thresholds["dpo_min_distractor_distinctness"] = (
                _FALLBACK_DPO_MIN_DISTRACTOR_DISTINCTNESS
            )
            thresholds["min_prompt_chunk_jaccard"] = (
                _FALLBACK_MIN_PROMPT_CHUNK_JACCARD
            )

        # ---- Criterion 2: unsupported answer ---- #
        answer_support_score: Optional[float] = None
        if chunk_text and answer:
            if embedder is not None:
                try:
                    vec_answer = embedder.encode(answer)
                    # A whole long chunk is a poor embedding target for a
                    # concise, correctly grounded answer. Reuse the claim
                    # validator's bounded lexical retrieval only to select
                    # candidate evidence windows, then retain the semantic
                    # cosine gate and its existing threshold. This improves
                    # evidence resolution without weakening the validator.
                    from lib.validators.pair.claim_support import (
                        _localized_source_premises,
                    )

                    support_surfaces = [chunk_text]
                    support_surfaces.extend(
                        _localized_source_premises(chunk_text, answer)
                    )
                    support_scores = [
                        _cosine_similarity_safe(
                            vec_answer, embedder.encode(surface)
                        )
                        for surface in support_surfaces
                    ]
                    valid_scores = [
                        score for score in support_scores if score is not None
                    ]
                    if valid_scores:
                        answer_support_score = max(valid_scores)
                    else:
                        answer_support_score = _jaccard_overlap(
                            answer, chunk_text
                        )
                        fallback_to_jaccard = True
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "embedder.encode raised on answer surface: %s",
                        exc,
                    )
                    answer_support_score = _jaccard_overlap(
                        answer, chunk_text
                    )
                    fallback_to_jaccard = True
            else:
                answer_support_score = _jaccard_overlap(answer, chunk_text)
        new_fields["answer_support_score"] = answer_support_score
        answer_support_authority = (
            _objective_execution_answer_support_proof(
                pair, answer=answer, source_chunk_id=source_chunk_id,
                authorized_private_sidecar=authorized_private_sidecar,
            )
            or _complete_claim_proof(
                pair,
                answer=answer,
                source_chunk_id=source_chunk_id,
            )
        )
        if (
            answer_support_authority is not None
            and answer_support_authority.get("contract")
            == _OBJECTIVE_EXECUTION_AUTHORITY_CONTRACT
            and isinstance(pair.get("_objective_execution_candidate"), dict)
        ):
            pair["_objective_execution_candidate"][
                "answer_support_authority"
            ] = dict(answer_support_authority)
        answer_support_outcome: Optional[str] = None
        if answer_support_score is not None:
            answer_support_outcome = (
                "meets_floor"
                if answer_support_score
                >= thresholds["min_answer_support_score"]
                else (
                    "below_floor_superseded"
                    if answer_support_authority is not None
                    else "below_floor"
                )
            )

        # ---- Criterion 3: weak distractor (preference only) ---- #
        distractor_distinctness: Optional[float] = None
        distractor_quality: Optional[Dict[str, Any]] = None
        if kind == "preference" and chosen and rejected:
            if embedder is not None:
                try:
                    vec_chosen = embedder.encode(chosen)
                    vec_rejected = embedder.encode(rejected)
                    cos_pair = _cosine_similarity_safe(
                        vec_chosen, vec_rejected
                    )
                    if cos_pair is not None:
                        # Distinctness is 1 - cosine similarity per the
                        # schema's distractor_quality.semantic_distinctness
                        # definition.
                        distractor_distinctness = float(
                            1.0 - cos_pair
                        )
                    else:
                        distractor_distinctness = (
                            _semantic_distinctness_jaccard(chosen, rejected)
                        )
                        fallback_to_jaccard = True
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "embedder.encode raised on chosen/rejected: %s",
                        exc,
                    )
                    distractor_distinctness = (
                        _semantic_distinctness_jaccard(chosen, rejected)
                    )
                    fallback_to_jaccard = True
            else:
                distractor_distinctness = (
                    _semantic_distinctness_jaccard(chosen, rejected)
                )
            distractor_quality = {
                "semantic_distinctness": distractor_distinctness,
            }
            new_fields["distractor_quality"] = distractor_quality

        # ---- Criterion 4: unanswerable stem ---- #
        # Always stamp prompt_chunk_jaccard for the audit trail. As of
        # the 2026-06-09 RDF/SHACL calibration corpus recalibration the default
        # ``min_prompt_chunk_jaccard`` floor is 0.0, so the strict ``<``
        # comparison in the reject-precedence block below can never fire
        # against the default — the reject arm is retired and this value
        # is audit-stamp-only. (Paraphrase prompts are instructed to
        # reword the source, so the signal was measured non-separating
        # on real pairs: 13/417 on-topic prompts scored exactly 0.) An
        # operator can re-arm the reject arm by passing an explicit
        # positive ``min_prompt_chunk_jaccard`` to the constructor.
        prompt_chunk_jaccard: Optional[float] = None
        if chunk_text and prompt:
            prompt_chunk_jaccard = _jaccard_overlap(prompt, chunk_text)

        # ---- Reject precedence ---- #
        # Order: placeholder (highest, zero-tolerance) → source-free →
        # unanswerable_stem → unsupported_answer → weak_distractor →
        # low_bloom_alignment → generic_rationale. Order chosen so the
        # most severe / cheapest-to-detect criteria fire first; once a
        # criterion fires we short-circuit (the audit-trail signals are
        # already stamped on ``new_fields`` above).
        rejection_reason: Optional[str] = None
        if placeholder_match is not None:
            rejection_reason = "placeholder_residue"
        elif not source_chunk_id:
            rejection_reason = "source_free_generation"
        elif (
            chunk_text
            and prompt_chunk_jaccard is not None
            and prompt_chunk_jaccard < thresholds["min_prompt_chunk_jaccard"]
        ):
            rejection_reason = "unanswerable_stem"
        elif (
            chunk_text
            and answer
            and answer_support_score is not None
            and answer_support_score
            < thresholds["min_answer_support_score"]
            and answer_support_authority is None
        ):
            rejection_reason = "unsupported_answer"
        elif (
            kind == "preference"
            and distractor_distinctness is not None
            and distractor_distinctness
            < thresholds["dpo_min_distractor_distinctness"]
        ):
            rejection_reason = "weak_distractor"
        elif (
            observed_bloom is not None
            and declared_bloom is not None
            and observed_bloom in BLOOM_LEVELS
            and declared_bloom in BLOOM_LEVELS
            and BLOOM_LEVELS.index(observed_bloom)
            < BLOOM_LEVELS.index(declared_bloom)
            and bloom_winner_score is not None
            # RETIRED reject arm. The default floor sits above 1.0, and
            # scores are softmax probabilities, so this can never fire —
            # `observed_bloom` / `bloom_alignment` are still stamped for
            # audit. See DEFAULT_BLOOM_ALIGNMENT_MIN_CONFIDENCE for why the
            # backing classifier is not trusted to discard training data.
            and bloom_winner_score
            > thresholds["bloom_alignment_min_confidence"]
        ):
            rejection_reason = "low_bloom_alignment"
        elif (
            rationale_score is not None
            and rationale_score
            < thresholds["min_rationale_richness_score"]
        ):
            rejection_reason = "generic_rationale"

        promotion_status = (
            "rejected" if rejection_reason is not None else "validated"
        )
        new_fields["promotion_status"] = promotion_status

        _emit_decision(
            decision_capture,
            pair_kind=kind,
            chunk_id=chunk_id,
            promotion_status=promotion_status,
            rejection_reason=rejection_reason,
            answer_support_score=answer_support_score,
            distractor_distinctness=distractor_distinctness,
            prompt_chunk_jaccard=prompt_chunk_jaccard,
            observed_bloom=observed_bloom,
            declared_bloom=declared_bloom,
            bloom_alignment=bloom_alignment,
            rationale_richness_score=rationale_score,
            placeholder_match=placeholder_match,
            bloom_winner_score=bloom_winner_score,
            thresholds=thresholds,
            embedder_strict=embedder_strict,
            fallback_to_jaccard=fallback_to_jaccard,
            answer_support_authority=answer_support_authority,
            answer_support_outcome=answer_support_outcome,
        )

        return promotion_status, rejection_reason, new_fields

    # ------------------------------------------------------------------ #
    # Gate-runner surface — walks training_specs/*.jsonl
    # ------------------------------------------------------------------ #

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Post-hoc audit that every pair on disk carries
        ``promotion_status``. NOT a re-run of the per-pair filter (that
        would require the chunks-lookup map and is the call-site's
        job). Mirrors the curie_anchoring gate's read-from-disk shape.

        Inputs:

        - ``course_dir`` (str) — preferred. Resolves to
          ``<course_dir>/training_specs/{instruction,preference}_pairs.jsonl``.
        - ``training_specs_dir`` (str) — sibling to course_dir.
        - ``instruction_pairs_path`` (str) +
          ``preference_pairs_path`` (str) — explicit overrides.

        Outputs:

        - ``passed=True`` when every pair on disk has
          ``promotion_status`` set to a non-empty string.
        - ``passed=False`` with ``MISSING_PROMOTION_STATUS`` issues
          (capped at 50) otherwise.
        - Graceful-degrade when the embedding extras are absent: emits
          a single ``EMBEDDING_DEPS_MISSING`` warning issue,
          ``passed=True``, ``action=None`` per the Wave N1 fallback
          contract. Strict mode re-enables fail-closed.
        """
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        embedder_strict = _is_strict_embeddings_mode()

        # Resolve paths.
        inst_path: Optional[Path] = None
        pref_path: Optional[Path] = None
        course_dir_raw = inputs.get("course_dir")
        training_specs_dir_raw = inputs.get("training_specs_dir")
        explicit_inst = inputs.get("instruction_pairs_path")
        explicit_pref = inputs.get("preference_pairs_path")

        if isinstance(explicit_inst, str) and explicit_inst:
            inst_path = Path(explicit_inst)
        if isinstance(explicit_pref, str) and explicit_pref:
            pref_path = Path(explicit_pref)
        if course_dir_raw and (inst_path is None or pref_path is None):
            cd = Path(course_dir_raw)
            if inst_path is None:
                inst_path = cd / "training_specs" / "instruction_pairs.jsonl"
            if pref_path is None:
                pref_path = cd / "training_specs" / "preference_pairs.jsonl"
        if (
            training_specs_dir_raw
            and (inst_path is None or pref_path is None)
        ):
            ts = Path(training_specs_dir_raw)
            if inst_path is None:
                inst_path = ts / "instruction_pairs.jsonl"
            if pref_path is None:
                pref_path = ts / "preference_pairs.jsonl"

        if inst_path is None and pref_path is None:
            issue = GateIssue(
                severity="critical",
                code="MISSING_INPUTS",
                message=(
                    "TrainingPairPromotionValidator requires one of: "
                    "course_dir, training_specs_dir, or explicit "
                    "{instruction,preference}_pairs_path."
                ),
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[issue],
                action="block",
            )

        # Walk both files; missing files are tolerated (a course may
        # have only instruction pairs or only preference pairs).
        issues: List[GateIssue] = []
        audited = 0
        missing_status_count = 0
        invalid_authority_count = 0
        for path in (inst_path, pref_path):
            if path is None or not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line_num, line in enumerate(fh, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        audited += 1
                        ps = row.get("promotion_status")
                        if not isinstance(ps, str) or not ps.strip():
                            missing_status_count += 1
                            if len(issues) < 50:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code="MISSING_PROMOTION_STATUS",
                                    message=(
                                        f"{path.name}:{line_num} pair "
                                        f"missing promotion_status; the "
                                        f"per-pair promotion validator "
                                        f"did not stamp the pair before "
                                        f"emit."
                                    ),
                                    location=str(path),
                                ))
                        score = row.get("answer_support_score")
                        if (
                            row.get("promotion_status") == "validated"
                            and isinstance(score, (int, float))
                            and score < self._min_answer_support_score
                            and (
                                isinstance(
                                    row.get(
                                        "_objective_execution_candidate"
                                    ),
                                    dict,
                                )
                                or (
                                    row.get("projection_contract") in {
                                        "ed4all-dpo-preference.v2",
                                        "ed4all-sft-chat.v2",
                                    }
                                    and isinstance(
                                        row.get("per_claim_support"), list,
                                    )
                                )
                            )
                        ):
                            answer = str(
                                row.get("completion")
                                if path == inst_path
                                else row.get("chosen")
                                or ""
                            )
                            recomputed = (
                                _replay_objective_execution_public_authority(
                                    row,
                                    answer=answer,
                                    source_chunk_id=str(
                                        row.get("source_chunk_id") or ""
                                    ),
                                )
                                if isinstance(
                                    row.get(
                                        "_objective_execution_candidate"
                                    ),
                                    dict,
                                )
                                else _complete_claim_proof(
                                    row,
                                    answer=answer,
                                    source_chunk_id=str(
                                        row.get("source_chunk_id") or ""
                                    ),
                                )
                            )
                            if (
                                recomputed is None
                            ):
                                invalid_authority_count += 1
                                if len(issues) < 50:
                                    issues.append(GateIssue(
                                        severity="critical",
                                        code=(
                                            "INVALID_ANSWER_SUPPORT_AUTHORITY"
                                        ),
                                        message=(
                                            f"{path.name}:{line_num} has "
                                            "a non-reproducible superseding "
                                            "answer-support proof."
                                        ),
                                        location=str(path),
                                    ))
            except OSError as exc:
                issues.append(GateIssue(
                    severity="critical",
                    code="PAIRS_FILE_READ_ERROR",
                    message=(
                        f"Failed to read {path}: {exc}"
                    ),
                    location=str(path),
                ))

        passed = (
            missing_status_count == 0
            and invalid_authority_count == 0
            and not any(
            i.severity == "critical" for i in issues
            )
        )
        action: Optional[str] = None if passed else "block"

        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="training_pair_promotion_check",
                    decision=(
                        "passed"
                        if passed
                        else f"failed:{missing_status_count}_missing"
                    ),
                    rationale=(
                        f"On-disk audit: {audited} pair(s) audited, "
                        f"{missing_status_count} missing "
                        f"promotion_status and {invalid_authority_count} "
                        f"invalid answer-support authorities. "
                        f"Strict embedder mode="
                        f"{embedder_strict}."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "DecisionCapture.log_decision raised on "
                    "training_pair_promotion_check (gate path): %s",
                    exc,
                )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=(
                1.0
                if audited == 0
                else round((audited - missing_status_count) / audited, 4)
            ),
            issues=issues,
            action=action,
        )


__all__ = [
    "TrainingPairPromotionValidator",
    "DEFAULT_MIN_ANSWER_SUPPORT_SCORE",
    "DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS",
    "DEFAULT_MIN_PROMPT_CHUNK_JACCARD",
    "DEFAULT_MIN_RATIONALE_RICHNESS_SCORE",
]
