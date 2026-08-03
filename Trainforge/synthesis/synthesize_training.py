#!/usr/bin/env python3
"""
Trainforge — Training Pair Synthesis Stage

Reads the enriched ``corpus/chunks.jsonl`` produced by the base pass (and,
when present, refined by ``align_chunks.py``), and emits two artifacts under
``training_specs/`` inside the same output directory:

    training_specs/instruction_pairs.jsonl   # SFT format
    training_specs/preference_pairs.jsonl    # DPO format

It also updates ``training_specs/dataset_config.json`` with counts under
``statistics.instruction_pairs`` and ``statistics.preference_pairs``.

This stage is invoked either:
    * programmatically: ``run_synthesis(corpus_dir=..., course_code=...)``
    * from the CLI via ``process_course.py --synthesize`` after base
      processing completes.

It uses the deterministic mock provider by default. An Anthropic provider
hook exists for future work but is not wired.

All generation decisions are captured via :class:`lib.decision_capture.DecisionCapture`
using two new decision types:

    * ``instruction_pair_synthesis``  (one event per instruction pair)
    * ``preference_pair_generation``  (one event per preference pair)

Each pair embeds the ``event_id`` of its own decision event in the
``decision_capture_id`` field so downstream consumers can join pairs to
their rationales.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import io
import json
import logging
import os
import random
import re
import sys
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, NoReturn, Optional,
    Sequence, Set, Tuple,
)

# Make project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture  # noqa: E402
from lib.ontology.curie_extraction import extract_curies  # noqa: E402
from lib.ontology.slugs import deslugify_concept  # noqa: E402
from lib.utils import append_jsonl as _utils_append_jsonl  # noqa: E402
from lib.utils import read_jsonl as _utils_read_jsonl  # noqa: E402
from lib.utils import write_jsonl as _utils_write_jsonl  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.ontology.template_prefixes import (  # noqa: E402
    DETERMINISTIC_TEMPLATE_PREFIXES as _DETERMINISTIC_TEMPLATE_PREFIXES,
)
from lib.validators.content_type import (  # noqa: E402
    assert_chunk_type,
    validate_chunk_type,
)
from Trainforge.generators.pairs.instruction import (  # noqa: E402
    synthesize_instruction_pair,
)
from Trainforge.generators.pairs.preference import (  # noqa: E402
    synthesize_preference_pair,
)
from Trainforge.generators.providers._synthesis_provider import (  # noqa: E402
    agnostic_synthesis_enabled,
)
from Trainforge.generators.providers._synthesis_common import SynthesisProviderError  # noqa: E402
from Trainforge.synthesis.synthesis_journal import (  # noqa: E402
    GenerationJournal,
    MAX_TRANSIENT_RESUME_ATTEMPTS,
    load_generation_journal,
    summarize_generation_journal,
)
from Trainforge.synthesis.curriculum import (  # noqa: E402
    DEFAULT_PREREQ_CONTEXT_TOKENS,
    build_curriculum_context,
    build_curriculum_manifest,
    build_prereq_recap,
    load_pedagogy_graph,
    order_pairs_by_curriculum,
)
from Trainforge.synthesis.synthesis_concurrency import (  # noqa: E402
    BoundedOrderedMap,
    resolve_synthesis_max_concurrent,
)
from Trainforge.synthesis.synthesis_progress import SynthesisProgressWriter  # noqa: E402
from Trainforge.synthesis.synthesis_reject_mining import (  # noqa: E402
    MINE_MODE_OFF,
    MINE_MODE_ON,
    MINE_REJECTS_ENV,
    MINED_PAIR_SOURCE,
    RejectPool,
    build_capture_payload,
    resolve_max_fraction,
    resolve_max_skeleton_freq,
    resolve_min_fail_entailment,
    resolve_min_support,
    resolve_mine_rejects_mode,
    select_mined_pairs,
)

logger = logging.getLogger(__name__)


DEFAULT_SEED = 17  # Arbitrary but stable; stage adds chunk-index for variety.


def _preflight_local_staged_model_identity(
    *,
    base_url: str,
    local_model: str,
    generic_model: str = "",
    get_json: Optional[Callable[[str], Mapping[str, Any]]] = None,
) -> str:
    """Prove the staged local seat serves the explicitly configured model.

    ``LOCAL_SYNTHESIS_MODEL`` is the canonical registry selector.
    ``TRAINFORGE_SYNTHESIS_MODEL`` is retained only as a conflict detector:
    it never overrides or substitutes for the registry selector.
    """
    expected = local_model.strip()
    generic = generic_model.strip()
    if not expected:
        raise RuntimeError(
            "LOCAL_SYNTHESIS_MODEL must be set explicitly for staged local "
            "synthesis; registry defaults are not accepted"
        )
    if generic and generic != expected:
        raise RuntimeError(
            "conflicting synthesis model selectors: "
            f"LOCAL_SYNTHESIS_MODEL={expected!r} but "
            f"TRAINFORGE_SYNTHESIS_MODEL={generic!r}; "
            "LOCAL_SYNTHESIS_MODEL is canonical"
        )

    models_url = f"{base_url.rstrip('/')}/models"

    def _default_get_json(url: str) -> Mapping[str, Any]:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"unable to verify staged synthesis model identity at {url}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"staged synthesis model identity response at {url} is not an object"
            )
        return payload

    payload = (get_json or _default_get_json)(models_url)
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"staged synthesis model identity response at {models_url} is not an object"
        )
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise RuntimeError(
            f"staged synthesis model identity response at {models_url} "
            "must contain data[]"
        )
    served_ids = {
        str(row.get("id", "")).strip()
        for row in rows
        if isinstance(row, Mapping) and str(row.get("id", "")).strip()
    }
    if expected not in served_ids:
        raise RuntimeError(
            f"configured staged synthesis model {expected!r} is not served by "
            f"{models_url}; served ids={sorted(served_ids)!r}"
        )
    return expected


# the Anthropic-SDK training-pair synthesis path (the
# ``AnthropicSynthesisProvider`` class + its SDK transport) was REMOVED
# entirely. ``provider="anthropic"`` is now UNCONDITIONALLY forbidden for
# training-pair synthesis: there is NO acknowledgment escape, because the code
# that could route the SLM training corpus through the Anthropic SDK no longer
# exists (the surface is license-clean by construction). This is distinct from
# the NVIDIA no-ack set below only in provenance — both fail closed with no
# escape. Canonical posture: docs/LICENSING.md § "Synthesis providers".
_REMOVED_SYNTHESIS_PROVIDERS = frozenset({"anthropic"})

# Marketable-v1 D4 — providers whose ToS restricts using outputs to train a
# derivative model. ``claude_session`` is a SEPARATE Claude-Code-session route
# (NOT the removed SDK path); its outputs are restricted under Anthropic
# Consumer Terms, so selecting it for TRAINING-PAIR synthesis (the corpus the
# SLM adapter is a derivative work of) stays a fail-loud opt-in gated behind
# ``TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true``. ``mock`` / ``together`` /
# ``local`` (and any registered OpenAI-compatible OSS provider) are
# license-clean and pass through ungated. Canonical posture: docs/LICENSING.md.
_ANTHROPIC_FAMILY_SYNTHESIS_PROVIDERS = frozenset({"claude_session"})

# Hosted-cloud providers whose ToS + underlying model license unconditionally
# restrict using outputs as SLM training data — NVIDIA's hosted Llama-3.3 tier.
# Unlike the Anthropic family (which has a documented ack-flag escape for an
# operator holding a separate written agreement), there is NO escape hatch here:
# the hosted Llama-3.3 corpus is never shippable as training data, so selecting
# this provider for TRAINING-PAIR synthesis fails closed unconditionally. This
# is the synthesis-side defense-in-depth companion to the workflow-runner's
# license-clean training-seat default (which covers the pipeline path; this
# covers a bare run_synthesis / MCP-tool / CLI call that bypasses corpus-gen).
# Canonical posture: docs/LICENSING.md § "Synthesis providers".
_RESTRICTED_NO_ACK_SYNTHESIS_PROVIDERS = frozenset({"nvidia"})


class SynthesisLicensingError(RuntimeError):
    """Raised when a training-pair synthesis run selects a ToS-unclean
    provider without the explicit ``TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS``
    acknowledgment. Fails closed before any LLM dispatch so a ToS-unclean
    corpus is never produced silently. See docs/LICENSING.md."""


@dataclass
class SynthesisStats:
    """Counts returned from :func:`run_synthesis`."""

    chunks_total: int = 0
    chunks_eligible: int = 0
    chunks_skipped_no_lo: int = 0
    instruction_pairs_emitted: int = 0
    instruction_pairs_rejected: int = 0
    preference_pairs_emitted: int = 0
    preference_pairs_rejected: int = 0
    instruction_pairs_ineligible: int = 0
    preference_pairs_ineligible: int = 0
    ineligible_reasons: Dict[str, int] = field(default_factory=dict)
    rejected_reasons: Dict[str, int] = field(default_factory=dict)
    # per-pair promotion-validator filter count. Increments
    # on every pair that the TrainingPairPromotionValidator rejects
    # before write — surfaces silently-dropped pairs in the audit trail.
    # See lib/validators/training_pair_promotion.py for the rejection
    # criteria. Distinct from instruction_pairs_rejected /
    # preference_pairs_rejected because those counters cover earlier
    # gate failures (template-recognizer, content-type, duplicate-prompt,
    # etc.) — dropped_count is the post-emit pre-write filter delta.
    dropped_count: int = 0
    # promotion-ladder counters surfaced into operator
    # telemetry. The same per-pair filter site that drives
    # ``dropped_count`` populates these counters too. Three monotonic
    # ladder steps + a parallel rejection-reason histogram so an
    # operator reading dataset_config.json::statistics.promotion_ladder
    # (and the W2.B aggregator's mirrored block) sees the funnel
    # candidate -> validated -> trainable + the per-reason histogram
    # without parsing decision-capture JSONL. The 7 canonical reason
    # keys mirror lib/validators/training_pair_promotion.py:
    # placeholder_residue, unsupported_answer, weak_distractor,
    # unanswerable_stem, source_free_generation, low_bloom_alignment,
    # generic_rationale. Invariant:
    #   rejected_promotion_pairs == sum(promotion_rejection_reasons.values())
    # and:
    #   candidate_pairs_total == validated_pairs_total + rejected_promotion_pairs
    # and (current pipeline shape):
    #   trainable_pairs_total == validated_pairs_total
    # (promotion validation IS the pre-write gate; no post-validation
    # write loss exists today, but trainable_pairs_total is a separate
    # counter so future write-side losses surface without conflating
    # the two ladder steps).
    candidate_pairs_total: int = 0
    validated_pairs_total: int = 0
    trainable_pairs_total: int = 0
    rejected_promotion_pairs: int = 0
    promotion_rejection_reasons: Dict[str, int] = field(default_factory=dict)
    # stratified-sampling additions. None when stratification
    # not active, so legacy callers keep the same payload shape.
    misconception_dpo_pairs_emitted: int = 0
    stratify_dimensions: List[str] = field(default_factory=list)
    stratify_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    capped_at_max_pairs: bool = False
    max_pairs_cap: Optional[int] = None
    difficulty_curriculum: bool = False
    # prerequisite-aware curriculum mode.
    curriculum_from_graph: bool = False
    prereq_windowed: bool = False
    prereq_context_tokens: int = DEFAULT_PREREQ_CONTEXT_TOKENS
    cycles_broken_count: int = 0
    pairs_without_concepts: int = 0
    concepts_without_pairs_count: int = 0
    pairs_with_prereq_recap: int = 0
    source_grounded_pairs: int = 0
    instruction_variants_per_chunk: int = 1
    # budget telemetry surfaced to callers.
    capped_at_max_dispatches: bool = False
    dispatched_count: int = 0
    cache_hits_count: int = 0
    # KG-metadata + violation-detection generators.
    kg_metadata_pairs_emitted: int = 0
    violation_pairs_emitted: int = 0
    # Abstention and schema-translation generators address zero-abstention and
    # schema-to-English bridge gaps. Counters surface the cohort sizes for the
    # post-run pilot report and the audit script.
    abstention_pairs_emitted: int = 0
    schema_translation_pairs_emitted: int = 0
    # Reject-mining capture counters (TRAINFORGE_DPO_MINE_REJECTS). Left EMPTY
    # when the flag is off, which is the default — an empty dict keeps every
    # existing consumer's payload shape unchanged. Populated with the capture
    # funnel (resume-cache rows scanned/admitted, captured this run, pool size)
    # when mining is on or in shadow; the selection pass adds its own keys.
    # Values are counts, with ONE deliberate exception: the selection pass
    # writes the string key ``reject_mining_skipped`` when an incomplete
    # generation pass made the reject pool biased, because "not evaluated" is
    # not expressible as a count and all-zero counters would misread as "no
    # candidates".
    reject_mining: Dict[str, Any] = field(default_factory=dict)
    # per-pair claim-support (W4.A) + LO-refs (W4.B) +
    # objective-delivery (W4.C) filter counters. All three validators
    # run AFTER TrainingPairPromotionValidator (W2.E) returns
    # ``validated`` and before the pair lands on disk; rejects
    # increment the matching counter here AND the existing
    # ``promotion_rejection_reasons`` dict (which is free-form and
    # admits the seven new reason keys ``unsupported_claim`` /
    # ``contradicted_claim`` / ``phantom_pair_lo_refs`` /
    # ``missing_pair_lo_refs`` / ``objective_statement_undersupported`` /
    # ``objective_bloom_undermet`` / ``objective_verb_absent`` without
    # a schema change). ``pair_validation_passed`` is the count of
    # pairs that survived ALL of W2.E + W4.A + W4.B + W4.C (i.e. the
    # W2.E ``validated_pairs_total`` minus the W4.A/W4.B/W4.C reject
    # deltas).
    # Invariant:
    #   pair_validation_passed
    #     == validated_pairs_total
    #        - claim_support_rejected
    #        - lo_refs_rejected
    #        - objective_delivery_rejected
    # Bookkeeping note: ``validated_pairs_total`` keeps the W2.E
    # promotion-validator semantics — it counts pairs that the
    # 7-criterion promotion filter accepted, regardless of whether
    # W4.A/W4.B/W4.C subsequently rejected them. The new counter
    # ``pair_validation_passed`` is the post-W4.A/W4.B/W4.C survivor
    # count. This preserves the W2.E counter invariant
    # (``candidate == validated + rejected_promotion``) while still
    # exposing the additional W4 filter delta to the audit trail.
    claim_support_rejected: int = 0
    lo_refs_rejected: int = 0
    objective_delivery_rejected: int = 0
    pair_validation_passed: int = 0
    # SFT data program (S3/S4/S5): assessment->SFT + concept-graph->SFT
    # generator cohorts, holdout-reduced-graph exclusions, and the layered
    # gold-set decontamination drop count.
    assessment_sft_pairs_emitted: int = 0
    graph_sft_pairs_emitted: int = 0
    holdout_edges_excluded: int = 0
    decontam_quarantined: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "chunks_eligible": self.chunks_eligible,
            "chunks_skipped_no_lo": self.chunks_skipped_no_lo,
            "instruction_pairs_emitted": self.instruction_pairs_emitted,
            "instruction_pairs_rejected": self.instruction_pairs_rejected,
            "preference_pairs_emitted": self.preference_pairs_emitted,
            "preference_pairs_rejected": self.preference_pairs_rejected,
            "instruction_pairs_ineligible": self.instruction_pairs_ineligible,
            "preference_pairs_ineligible": self.preference_pairs_ineligible,
            "ineligible_reasons": dict(self.ineligible_reasons),
            "rejected_reasons": dict(self.rejected_reasons),
            "dropped_count": self.dropped_count,
            # promotion-ladder counters projected into the
            # ``synthesis.last_run`` block of dataset_config.json so the
            # W2.B aggregator + the eval_report.json pass-through both
            # have a stable wire-shape to copy through. Reason histogram
            # keys are alphabetised on emit so a future histogram diff
            # (run-vs-run) is byte-stable regardless of insertion order.
            "candidate_pairs_total": self.candidate_pairs_total,
            "validated_pairs_total": self.validated_pairs_total,
            "trainable_pairs_total": self.trainable_pairs_total,
            "rejected_promotion_pairs": self.rejected_promotion_pairs,
            "promotion_rejection_reasons": {
                k: self.promotion_rejection_reasons[k]
                for k in sorted(self.promotion_rejection_reasons)
            },
            "misconception_dpo_pairs_emitted": self.misconception_dpo_pairs_emitted,
            "stratify_dimensions": list(self.stratify_dimensions),
            "stratify_distribution": {
                k: dict(v) for k, v in self.stratify_distribution.items()
            },
            "capped_at_max_pairs": self.capped_at_max_pairs,
            "max_pairs_cap": self.max_pairs_cap,
            "difficulty_curriculum": self.difficulty_curriculum,
            "curriculum_from_graph": self.curriculum_from_graph,
            "prereq_windowed": self.prereq_windowed,
            "prereq_context_tokens": self.prereq_context_tokens,
            "cycles_broken_count": self.cycles_broken_count,
            "pairs_without_concepts": self.pairs_without_concepts,
            "concepts_without_pairs_count": self.concepts_without_pairs_count,
            "pairs_with_prereq_recap": self.pairs_with_prereq_recap,
            "source_grounded_pairs": self.source_grounded_pairs,
            "instruction_variants_per_chunk": self.instruction_variants_per_chunk,
            # per-pair claim-support (W4.A) + LO-refs
            # (W4.B) + objective-delivery (W4.C) filter counters
            # projected into the ``synthesis.last_run`` block of
            # dataset_config.json so the W2.B aggregator + the
            # eval_report.json pass-through both see the post-W2.E
            # filter deltas without parsing decision-capture JSONL.
            "claim_support_rejected": self.claim_support_rejected,
            "lo_refs_rejected": self.lo_refs_rejected,
            "objective_delivery_rejected": self.objective_delivery_rejected,
            "pair_validation_passed": self.pair_validation_passed,
            # SFT data program (S3/S4/S5).
            "assessment_sft_pairs_emitted": self.assessment_sft_pairs_emitted,
            "graph_sft_pairs_emitted": self.graph_sft_pairs_emitted,
            "holdout_edges_excluded": self.holdout_edges_excluded,
            "decontam_quarantined": self.decontam_quarantined,
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {chunks_path}")
    chunks = _utils_read_jsonl(chunks_path)
    # The ONE seam both synthesis reads go through (preflight + main), so
    # eligibility and emission cannot see different misconceptions. Recovery
    # fills only an absent/empty field, from each chunk's own markup; an
    # authored list is returned untouched, so a corpus whose chunker already
    # lands structured misconceptions is byte-identical. See
    # ``synthesis_window_contract.resolve_chunk_misconceptions`` for why this
    # has to happen before the emitter rather than at the window layer.
    from Trainforge.generators.staged.window_contract import (
        resolve_chunk_key_terms,
        resolve_chunk_misconceptions,
    )
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if not chunk.get("misconceptions"):
            recovered = resolve_chunk_misconceptions(chunk)
            if recovered:
                chunk["misconceptions"] = recovered
        # Same recovery, different authored artifact. Empty `key_terms` is
        # what makes `_derive_topic` splice an LO id into the topic slot, so
        # this is the difference between grounding on a real topic and
        # grounding on an opaque learning-outcome identifier.
        if not chunk.get("key_terms"):
            terms = resolve_chunk_key_terms(chunk)
            if terms:
                chunk["key_terms"] = terms
    _backfill_topicless_concept_tags(chunks)
    return chunks


def _backfill_topicless_concept_tags(chunks: List[Dict[str, Any]]) -> None:
    """Give topic-less chunks a deterministic lexical concept tag.

    A chunk with NEITHER ``concept_tags`` NOR ``key_terms`` has no topic
    ``preference_factory._derive_topic`` can name, so it falls back to
    splicing the learning-outcome id into the topic slot. That produces
    content-free completions that the entailment gate correctly rejects.

    The derivation is the project's OWN canonical remedy for this defect,
    ``lib/ontology/lexical_concept_seeds.derive_lexical_concept_seeds`` — the
    same helper ``TRAINFORGE_PAGE_CONCEPT_FALLBACK`` uses at chunking time.
    It is purely lexical/statistical (no embeddings, no LLM), so nothing is
    invented: a tag is assigned only when the chunk's OWN text contains the
    derived term. Chunks already carrying a topic are untouched, so a corpus
    whose chunker populated concept_tags is unchanged.

    Applied here rather than at chunking because these chunksets are already
    archived: re-deriving at the read seam avoids a full re-chunk (which
    would move ``imscc_chunks_sha256`` and invalidate the model card's
    provenance hash) while giving the generator the same signal.
    """
    topicless = [
        chunk for chunk in chunks
        if isinstance(chunk, dict)
        and not chunk.get("concept_tags")
        and not chunk.get("key_terms")
        and str(chunk.get("text") or "").strip()
    ]
    if not topicless:
        return
    from lib.ontology.lexical_concept_seeds import (
        derive_lexical_concept_seeds,
    )
    try:
        seeds = derive_lexical_concept_seeds(topicless, min_doc_freq=2)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("lexical concept-tag backfill failed: %s", exc)
        return
    if not seeds:
        return
    # Longest first: a chunk gets its most specific matching seed.
    ordered = sorted(seeds, key=lambda s: (-len(s), s))
    tagged = 0
    for chunk in topicless:
        haystack = " ".join(str(chunk.get("text") or "").lower().split())
        matched = [
            seed for seed in ordered
            if seed.replace("-", " ") in haystack
        ][:5]
        if matched:
            chunk["concept_tags"] = matched
            tagged += 1
    if tagged:
        logger.info(
            "lexical concept-tag backfill: tagged %d of %d topic-less chunks "
            "from %d derived seeds",
            tagged, len(topicless), len(seeds),
        )


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Atomic JSONL writer (W-D6: thin wrapper around
    :func:`lib.utils.write_jsonl`). Defaults preserved:
    ``ensure_ascii=False, sort_keys=True``, atomic tmp+rename.
    """
    return _utils_write_jsonl(path, records)


def _eligible(chunk: Dict[str, Any]) -> bool:
    return bool(chunk.get("learning_outcome_refs")) and bool(chunk.get("id") or chunk.get("chunk_id"))


def staged_objective_contract_enabled() -> bool:
    """Return True when a staged contract owns canonical objective focus.

    BOTH staged contracts — ``TRAINFORGE_STAGED_SYNTHESIS_V4`` and
    ``TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1`` — consume the focused provider
    view produced by
    :func:`Trainforge.synthesis.synthesis_eligibility.focus_chunk_on_canonical_objective`,
    and ``pair_eligibility`` requires ``synthesis_focus_objective`` to be
    present on the chunk it is handed.

    This predicate exists so the FOCUS seam and the ELIGIBILITY seam can never
    disagree about which modes are staged.  They previously did: focus was
    gated on v4 alone while eligibility admitted v4-or-micro-v1, so a micro-v1
    run handed ``pair_eligibility`` an UNFOCUSED chunk and every chunk
    carrying ``learning_outcome_refs`` reported
    ``missing_canonical_objective_focus`` — a whole-corpus zero-pair emit with
    no error raised.
    """
    from Trainforge.generators.staged.micro import (
        staged_synthesis_micro_v1_enabled,
    )
    from Trainforge.generators.staged.provider import (
        staged_synthesis_v4_enabled,
    )

    return staged_synthesis_v4_enabled() or staged_synthesis_micro_v1_enabled()


# ---------------------------------------------------------------------------
# Synthesis-contract selection (``--synthesis-contract``).
#
# The three contracts are selected by two mutually-exclusive environment
# switches read deep inside ``build_synthesis_provider``.  The CLI flag is the
# documented operator entry, so it must RESOLVE to that env pair before any
# provider is constructed; leaving it unread meant an operator following the
# documented invocation silently got whatever the ambient environment said.
#
# Omitting the flag preserves the historical environment-driven path
# byte-for-byte — this is a resolver, not a new default.
# ---------------------------------------------------------------------------
ENV_STAGED_SYNTHESIS_V4 = "TRAINFORGE_STAGED_SYNTHESIS_V4"
ENV_STAGED_SYNTHESIS_MICRO_V1 = "TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1"

SYNTHESIS_CONTRACT_ENV = {
    "legacy": {ENV_STAGED_SYNTHESIS_V4: "false",
               ENV_STAGED_SYNTHESIS_MICRO_V1: "false"},
    "staged-v4": {ENV_STAGED_SYNTHESIS_V4: "true",
                  ENV_STAGED_SYNTHESIS_MICRO_V1: "false"},
    "micro-v1": {ENV_STAGED_SYNTHESIS_V4: "false",
                 ENV_STAGED_SYNTHESIS_MICRO_V1: "true"},
}
SYNTHESIS_CONTRACT_CHOICES = tuple(SYNTHESIS_CONTRACT_ENV)

# Mirrors the ``_TRUE`` frozenset both contract modules parse with, so this
# resolver cannot disagree with the switches it is setting.
_SYNTHESIS_CONTRACT_TRUE = frozenset({"1", "true", "yes", "on"})


class SynthesisContractConflict(RuntimeError):
    """An explicit --synthesis-contract disagrees with the ambient env."""


def resolve_synthesis_contract_env(selection: str) -> Dict[str, str]:
    """Return the env pair for one documented contract spelling."""
    try:
        return dict(SYNTHESIS_CONTRACT_ENV[str(selection)])
    except KeyError as exc:
        raise ValueError(
            "synthesis contract must be one of "
            f"{', '.join(SYNTHESIS_CONTRACT_CHOICES)}; got {selection!r}"
        ) from exc


def apply_synthesis_contract_selection(
    selection: Optional[str],
    environ: Optional[Any] = None,
) -> Optional[Dict[str, str]]:
    """Apply an explicit contract selection to the process environment.

    ``None`` leaves the environment untouched (historical env-driven path).
    An ambient switch that DISAGREES with the selection is a loud failure
    rather than a silent override: an operator who asked for one contract must
    never be given another, and an operator whose environment already pins a
    contract must never have it changed out from under them without noticing.
    """
    if selection is None:
        return None
    desired = resolve_synthesis_contract_env(selection)
    source = os.environ if environ is None else environ
    conflicts = []
    for name, value in desired.items():
        ambient = str(source.get(name, "")).strip()
        if not ambient:
            continue
        ambient_on = ambient.lower() in _SYNTHESIS_CONTRACT_TRUE
        if ambient_on != (value == "true"):
            conflicts.append(
                f"{name}={ambient!r} (selection requires {value!r})"
            )
    if conflicts:
        raise SynthesisContractConflict(
            f"--synthesis-contract {selection} conflicts with the ambient "
            "environment: " + "; ".join(conflicts) + ". Unset the conflicting "
            "variable(s) or drop the flag."
        )
    for name, value in desired.items():
        source[name] = value
    return desired


def _micro_generation_unit(kind: str, variant_index: int) -> Any:
    """Bind the micro contract's per-unit resume identity.

    ``MicroStagedSynthesisProvider`` keys its resume-store FILE PATH on a
    per-unit identity whose only caller-supplied field is the variant token.
    Production drafts carry no manifest stamp (only the pilot harness does), so
    the unit is bound here from ``(kind, variant_index)`` — the same key the
    checkpoint cache and generation journal already use — keeping the micro
    resume store 1:1 with the outer journal's units.  Returns a no-op context
    for every other contract, so the non-micro path is byte-identical.
    """
    from Trainforge.generators.staged.micro import (
        bind_micro_generation_unit,
        staged_synthesis_micro_v1_enabled,
    )

    if not staged_synthesis_micro_v1_enabled():
        return contextlib.nullcontext()
    return bind_micro_generation_unit(
        kind=kind, variant_index=variant_index,
    )


def _focus_chunk_on_objective(
    chunk: Dict[str, Any],
    *,
    seed: int,
    objectives: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Apply authoritative objective focus for every staged contract.

    The staged workflows are opt-in at the provider surface.  Keeping this
    guard at the earliest focus seam preserves the historical chunk view
    exactly when no staged contract is selected, including mock/fixture runs.
    """
    if not staged_objective_contract_enabled():
        return chunk
    from Trainforge.synthesis.synthesis_eligibility import (
        focus_chunk_on_canonical_objective,
    )

    return focus_chunk_on_canonical_objective(
        chunk,
        seed=seed,
        objectives=objectives or {},
    )


def _pair_eligibility_for_mode(
    focused_chunk: Mapping[str, Any],
    *,
    kind: str,
) -> Any:
    """Return staged eligibility or an unconditional legacy admission.

    The staged predicate is shared verbatim with ``_focus_chunk_on_objective``
    via :func:`staged_objective_contract_enabled`, because this function's
    ``pair_eligibility`` branch is only meaningful on a chunk that seam
    already focused.
    """
    from Trainforge.synthesis.synthesis_eligibility import (
        PairEligibility,
        content_gate_eligibility,
        pair_eligibility,
    )

    if focused_chunk.get("_eval_holdout_reserved") is True:
        return PairEligibility(False, "eval_holdout_reserved")
    # The content gate is a property of the SOURCE chunk, not of the staged
    # contract, so it runs on every mode — the legacy path is exactly where
    # the content-free pairs were being manufactured.
    content_gate = content_gate_eligibility(focused_chunk)
    if not content_gate.eligible:
        return content_gate
    if not staged_objective_contract_enabled():
        return PairEligibility(True)
    return pair_eligibility(focused_chunk, kind=kind)


# ---------------------------------------------------------------------------
# stratified-sampling + LibV2-archive helpers
# ---------------------------------------------------------------------------

# Canonical difficulty tiers, ordered foundational -> advanced. Used by the
# --difficulty-curriculum ordering. Unknown tiers sort last.
_DIFFICULTY_ORDER: Dict[str, int] = {
    "foundational": 0,
    "intermediate": 1,
    "advanced": 2,
}


# Recognised stratification dimensions. Anything else is rejected with a
# ValueError so typos don't silently degrade to a no-op.
_STRATIFY_DIMENSIONS = {"bloom", "chunk_type", "outcome", "difficulty"}


def _resolve_libv2_corpus_dir(slug: str, libv2_root: Optional[Path] = None) -> Path:
    """Return the directory under ``LibV2/courses/`` matching ``slug``.

    Accepts both the canonical slug (``<course-slug>``) and the doubled-up
    form some archival runs produce (``<course-slug>-<course-slug>``). The
    archived layout is ``LibV2/courses/<slug>/{corpus,objectives.json,...}``;
    this function locates that root so callers can read ``corpus/chunks.jsonl``
    and ``objectives.json`` directly without re-running the Trainforge pipeline.
    """
    root = libv2_root or (PROJECT_ROOT / "LibV2" / "courses")
    direct = root / slug
    if direct.exists():
        return direct
    doubled = root / f"{slug}-{slug}"
    if doubled.exists():
        return doubled
    # Last attempt: case-insensitive scan.
    if root.exists():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.lower() == slug.lower():
                return child
    raise FileNotFoundError(
        f"LibV2 archive for slug={slug!r} not found under {root}; "
        f"tried {direct} and {doubled}"
    )


# --------------------------------------------------------------------------- #
# dual-source DART block-text resolver
# --------------------------------------------------------------------------- #


import re as _w9_re  # local alias so the block-text helpers don't shadow re imports


# Mirrors :data:`lib.validators.content_grounding._DART_BLOCK_ID_RE`.
# Single source of truth would be a re-export from content_grounding,
# but the pattern is two lines + the import topology there is heavier
# than this module wants — copy is the smaller blast radius.
# DART->semantik purge Stage 3 (dual-READ): staged HTML may carry either attr
# spelling; both are harvested identically.
_W9_DART_BLOCK_ID_RE = _w9_re.compile(
    r'data-(?:dart|semantik)-block-id\s*=\s*(["\'])([^"\']+)\1',
    _w9_re.IGNORECASE,
)
# Captures the body of a <section ...> wrapper; used to scope each
# block_id's text extraction to JUST that section. <section> wrappers
# are the canonical DART-block boundary per
# ``DART/CLAUDE.md`` § "Source provenance".
_W9_DART_SECTION_RE = _w9_re.compile(
    r"<section\b([^>]*)>(.*?)</section>",
    _w9_re.IGNORECASE | _w9_re.DOTALL,
)
_W9_HTML_TAG_RE = _w9_re.compile(r"<[^>]+>")
_W9_WHITESPACE_RE = _w9_re.compile(r"\s+")


def _resolve_dart_block_text_map(
    staging_dir: Optional[Path],
) -> Dict[str, str]:
    """build the ``dart:<slug>#<block_id> -> text`` map.

    Mirrors the staging-DART HTML walk at
    ``lib/validators/content_grounding.py:343`` (the precedent the W9
    plan called out at §2 lines 195-200) but extracts per-section text
    in addition to the block_id. Returns ``{}`` when ``staging_dir`` is
    None or doesn't exist (legacy / non-textbook-to-course corpora —
    the dual-source check no-ops cleanly via the empty-map arm in
    :meth:`PairClaimSupportValidator.validate_pair`).

    The helper is intentionally regex-based rather than BeautifulSoup-
    based: we don't want to add a soft dependency to the synthesis path
    (parsers are already heavy enough), and the staging DART HTML is
    machine-emitted so the regex shape is stable. When the regex misses
    a block (malformed nesting, etc.), the dual-source check no-ops on
    that block — no false-positive disagreement signal can fire from a
    malformed parse.
    """
    if staging_dir is None:
        return {}
    staging_path = Path(staging_dir)
    if not staging_path.exists():
        return {}

    block_text_map: Dict[str, str] = {}
    for html_path in staging_path.rglob("*.html"):
        try:
            content = html_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        slug = html_path.stem.lower().replace(" ", "-")
        # Walk every <section> body. The block_id lives in the opening
        # tag's attributes (data-dart-block-id="...").
        for match in _W9_DART_SECTION_RE.finditer(content):
            attrs = match.group(1)
            body = match.group(2)
            id_match = _W9_DART_BLOCK_ID_RE.search(attrs)
            if id_match is None:
                continue
            raw_block_id = id_match.group(2).strip()
            # Strip nested HTML tags + collapse whitespace.
            text = _W9_HTML_TAG_RE.sub(" ", body)
            text = _W9_WHITESPACE_RE.sub(" ", text).strip()
            if not text:
                continue
            # DART->semantik purge Stage 3: key the lookup under BOTH sourceId
            # prefixes so a chunk carrying a freshly-minted ``semantik:`` ref
            # AND a legacy ``dart:`` ref both resolve into this text map.
            # First-write-wins on duplicate keys — mirrors the W2.F /
            # source_module_map.json precedent. Duplicates inside one
            # staging dir are typically the same block_id reused
            # across re-staged HTML files, so first-seen is fine.
            for key in (
                f"dart:{slug}#{raw_block_id}",
                f"semantik:{slug}#{raw_block_id}",
            ):
                if key not in block_text_map:
                    block_text_map[key] = text
    return block_text_map


def _stratify_key(chunk: Dict[str, Any], dimension: str) -> str:
    """Extract the stratification bucket key for one chunk on one dimension.

    Missing fields collapse to ``"unknown"`` so every chunk lands in some
    bucket rather than being silently dropped.
    """
    if dimension == "bloom":
        return str(chunk.get("bloom_level") or "unknown").lower()
    if dimension == "chunk_type":
        return str(chunk.get("chunk_type") or "unknown").lower()
    if dimension == "outcome":
        refs = chunk.get("learning_outcome_refs") or []
        return str(refs[0]).lower() if refs else "unknown"
    if dimension == "difficulty":
        return str(chunk.get("difficulty") or "unknown").lower()
    return "unknown"


def _composite_stratify_key(chunk: Dict[str, Any], dimensions: Sequence[str]) -> str:
    return "|".join(_stratify_key(chunk, d) for d in dimensions)


def _stratified_sample(
    chunks: List[Dict[str, Any]],
    dimensions: Sequence[str],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Round-robin draw across stratification buckets so the output
    distribution is uniform across the dimension(s).

    Each bucket donates one chunk per pass; passes continue until either
    ``target_count`` chunks have been emitted or every bucket is empty.
    Within a bucket the pre-existing order is preserved (after a one-time
    deterministic shuffle keyed by the rng) so two runs at the same seed
    return the same sequence.
    """
    if not chunks or target_count <= 0:
        return []
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        buckets[_composite_stratify_key(c, dimensions)].append(c)

    # Deterministic shuffle inside each bucket so we don't always pick the
    # earliest chunk on a tie; rng is seeded by the caller.
    for key in buckets:
        rng.shuffle(buckets[key])

    bucket_keys = sorted(buckets.keys())
    out: List[Dict[str, Any]] = []
    while len(out) < target_count:
        progressed = False
        for k in bucket_keys:
            if not buckets[k]:
                continue
            out.append(buckets[k].pop(0))
            progressed = True
            if len(out) >= target_count:
                break
        if not progressed:
            break
    return out


def _smoke_stratified_sample(
    chunks: List[Dict[str, Any]],
    manifest: Optional[Any],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """smoke-mode chunk sampler.

    Selects ~``target_count`` chunks for fast-feedback runs. Strategy:

    1. For each property in ``manifest`` (if any), pick up to 3 chunks
       whose text contains a declared surface form. This guarantees the
       smoke run exercises every property the full run would gate.
    2. Pad with random chunks (deterministic via ``rng``) until
       ``target_count`` is reached.

    No manifest -> just pick the first ``target_count`` eligible chunks
    in deterministic order. Empty corpus -> empty list.
    """
    if not chunks or target_count <= 0:
        return list(chunks)
    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()

    def _add(c: Dict[str, Any]) -> None:
        cid = id(c)
        if cid in selected_ids:
            return
        selected.append(c)
        selected_ids.add(cid)

    if manifest is not None:
        per_property_cap = 3
        for prop in manifest.properties:
            hits = [
                c for c in chunks
                if any(sf in str(c.get("text") or "") for sf in prop.surface_forms)
            ]
            for c in hits[:per_property_cap]:
                _add(c)
                if len(selected) >= target_count:
                    return selected

    remaining = [c for c in chunks if id(c) not in selected_ids]
    rng.shuffle(remaining)
    for c in remaining:
        if len(selected) >= target_count:
            break
        _add(c)
    return selected


def _curriculum_sort_key(chunk: Dict[str, Any]) -> Tuple[int, str]:
    diff = str(chunk.get("difficulty") or "").lower()
    rank = _DIFFICULTY_ORDER.get(diff, len(_DIFFICULTY_ORDER))
    cid = str(chunk.get("id") or chunk.get("chunk_id") or "")
    return (rank, cid)


def _build_misconception_dpo_pair(
    chunk: Dict[str, Any],
    misconception: Dict[str, Any],
    pair_index: int,
    capture: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Convert a single (misconception, correction) entry into a DPO pair.

    Returns None when either side is empty. The silent-drop path emits a
    ``misconception_pair_skipped`` audit event via ``capture.log_decision``
    so a corpus rebuild that quietly loses a property family is still visible
    in the decision-capture stream. ``capture`` is optional only so unit tests
    can exercise this helper in isolation — every production call site (the
    augmentation loop in ``run_synthesis``) passes one in.
    """
    from Trainforge.generators.pairs.preference import _misconception_id

    chunk_id_for_log = str(chunk.get("id") or chunk.get("chunk_id") or "")
    mc_text_for_id = str(misconception.get("misconception", "")).strip()
    correction_for_id = str(misconception.get("correction", "")).strip()
    if not mc_text_for_id or not correction_for_id:
        empty_field = "misconception" if not mc_text_for_id else "correction"
        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="misconception_pair_skipped",
                    decision="dropped",
                    rationale=(
                        f"empty {empty_field} on chunk {chunk_id_for_log}: "
                        f"editorial misconception entry at pair_index="
                        f"{pair_index} had a blank/whitespace-only "
                        f"{empty_field} field after strip(); the DPO pair "
                        f"would carry an empty chosen/rejected side and "
                        f"violate the preference_pair schema, so the entry "
                        f"is dropped before emit. Pre-Wave-112 this drop "
                        f"happened with no audit trail."
                    ),
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to log misconception_pair_skipped event for "
                    "chunk %s: %s", chunk_id_for_log, e,
                )
        return None
    mc_text = html.unescape(mc_text_for_id)
    correction = html.unescape(correction_for_id)
    if correction.rstrip().endswith(":"):
        return None

    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    chunk_bloom = str(chunk.get("bloom_level") or "").lower() or None
    # use the misconception's OWN bloom_level (not the chunk's)
    # for mc_id computation. This mirrors
    # ``CourseProcessor._build_misconceptions_for_graph`` which seeds the
    # hash off ``entry.get("bloom_level")``. Using the chunk's bloom level
    # produced misconception_ids that didn't match the pedagogy graph
    # nodes, breaking misconception-coverage audits and downstream KG
    # lookups.
    mc_bloom = str(misconception.get("bloom_level") or "").strip().lower() or None
    refs = chunk.get("learning_outcome_refs") or []
    primary_concept = ""
    tags = chunk.get("concept_tags") or []
    if tags:
        primary_concept = deslugify_concept(str(tags[0]))
    elif refs:
        primary_concept = f"learning outcome {refs[0]}"
    else:
        primary_concept = "the course topic"
    correction = _fit_pair_answer(correction, primary_concept)
    mc_text = _fit_pair_answer(mc_text, primary_concept)

    prompt = (
        f"Explain {primary_concept} clearly enough for a new learner to "
        f"avoid the most common misconception."
    )
    mc_id = _misconception_id(mc_text_for_id, correction_for_id, mc_bloom)
    pair = {
        "id": f"mcp_{chunk_id}_{pair_index:03d}",
        "chunk_id": chunk_id,
        "prompt": prompt,
        "chosen": correction,
        "rejected": mc_text,
        "source": "misconception_editorial",
        "misconception_id": mc_id,
        "bloom_level": chunk_bloom or "unknown",
        "lo_refs": list(refs),
        "learning_outcome_refs": list(refs),
        "seed": pair_index,
    }
    return pair


def _chunk_source_references(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    source = chunk.get("source") if isinstance(chunk.get("source"), dict) else {}
    refs = source.get("source_references") if isinstance(source, dict) else None
    if not isinstance(refs, list):
        return []
    return [dict(r) for r in refs if isinstance(r, dict)]


def _append_citation(text: str, chunk_id: str, *, max_len: int = 600) -> str:
    citation = f" [{chunk_id}]"
    text = str(text or "").strip()
    if not chunk_id or citation in text:
        return text
    if len(text) + len(citation) <= max_len:
        return text + citation

    budget = max(0, max_len - len(citation))
    trimmed = text[:budget].rstrip()
    boundary = trimmed.rfind(". ")
    if boundary >= 50:
        trimmed = trimmed[:boundary + 1].rstrip()
    return (trimmed + citation).strip()


def _append_citation_instruction(prompt: str, *, max_len: int = 400) -> str:
    tail = " Cite the source chunk in brackets."
    prompt = str(prompt or "").strip()
    if "cite the source chunk" in prompt.lower():
        return prompt
    if len(prompt) + len(tail) <= max_len:
        return prompt + tail
    return prompt


def _pad_short_answer(text: str, topic: str, *, min_len: int = 50) -> str:
    text = str(text or "").strip()
    if len(text) >= min_len:
        return text
    return (
        f"{text} This correction keeps the learner grounded in {topic} "
        f"rather than a misleading shortcut."
    ).strip()


def _fit_pair_answer(text: str, topic: str, *, max_len: int = 600) -> str:
    text = _pad_short_answer(text, topic)
    if len(text) <= max_len:
        return text
    hard = text[:max_len]
    boundary = hard.rfind(". ")
    if boundary >= 50:
        return hard[:boundary + 1].strip()
    return hard[: max_len - 3].rstrip() + "..."


def _attach_source_grounding(
    pair: Dict[str, Any],
    chunk: Dict[str, Any],
    *,
    cite: Optional[bool] = None,
) -> bool:
    """Attach source metadata, adding target citations only when requested."""
    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    if not chunk_id:
        return False

    pair["source_chunk_id"] = chunk_id
    pair["source_references"] = _chunk_source_references(chunk)
    pair["source_citation"] = f"[{chunk_id}]"
    if cite is None:
        cite = bool(pair.get("requires_source_citation"))
    if not cite:
        return True

    pair["prompt"] = _append_citation_instruction(str(pair.get("prompt") or ""))

    grounded = False
    if "completion" in pair:
        pair["completion"] = _append_citation(str(pair.get("completion") or ""), chunk_id)
        grounded = True
    if "chosen" in pair:
        pair["chosen"] = _append_citation(str(pair.get("chosen") or ""), chunk_id)
        grounded = True
    return grounded


# the persona-prefix variant is a frame template parametrized
# by ``{persona}``. Pre-Wave-132d this slot was hardcoded to
# "For an RDF/SHACL learner, {prompt_lc}", which was wrong for any
# non-SHACL course. The persona is sourced from
# ``PropertyManifest.learner_persona`` (default ``DEFAULT_LEARNER_PERSONA``
# = "a learner"); see ``lib/ontology/property_manifest.py``.
_INSTRUCTION_PROMPT_FRAMES = (
    "{prompt}",
    "For {persona}, {prompt_lc}",
    "Give a source-grounded answer: {prompt}",
)


def _default_learner_persona() -> str:
    """The persona slot ``_INSTRUCTION_PROMPT_FRAMES[1]`` interpolates.

    Lazy-imported for the same reason ``_apply_instruction_variant`` does it:
    the mock-provider call paths never need a property manifest.
    """
    from lib.ontology.property_manifest import DEFAULT_LEARNER_PERSONA
    return DEFAULT_LEARNER_PERSONA


def _apply_instruction_variant(
    pair: Dict[str, Any],
    variant_index: int,
    *,
    learner_persona: Optional[str] = None,
) -> None:
    pair["instruction_variant"] = int(variant_index)
    pair["requires_source_citation"] = (
        variant_index % len(_INSTRUCTION_PROMPT_FRAMES) == 2
    )
    if variant_index <= 0:
        return
    prompt = str(pair.get("prompt") or "").strip()
    if not prompt:
        return
    frame = _INSTRUCTION_PROMPT_FRAMES[
        variant_index % len(_INSTRUCTION_PROMPT_FRAMES)
    ]
    # lazy-import keeps the module-import cost flat for the
    # mock-provider call paths that don't need a manifest.
    if not learner_persona:
        from lib.ontology.property_manifest import DEFAULT_LEARNER_PERSONA
        learner_persona = DEFAULT_LEARNER_PERSONA
    candidate = frame.format(
        prompt=prompt,
        prompt_lc=prompt[:1].lower() + prompt[1:],
        persona=learner_persona,
    )
    # Leave room for the citation instruction appended later.
    if len(candidate) <= 360:
        pair["prompt"] = candidate


def _update_dataset_config(
    dataset_config_path: Path,
    stats: SynthesisStats,
    *,
    holdout_identity: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load existing dataset_config.json, update statistics, write back atomically.

    If the file does not exist, a minimal stub is created. Fields already set
    by the base pass are preserved (additive-only update).
    """
    if dataset_config_path.exists():
        with dataset_config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    else:
        config = {
            "format": "instruction-following",
            "target_models": ["claude-opus-4-6", "claude-sonnet-4-6"],
            "training_objectives": [],
            "statistics": {},
        }

    config.setdefault("statistics", {})
    config["statistics"]["instruction_pairs"] = stats.instruction_pairs_emitted
    config["statistics"]["preference_pairs"] = stats.preference_pairs_emitted
    # surface the promotion-ladder counters into a
    # dedicated, operator-readable block under ``statistics`` so
    # downstream readers (W2.B aggregator, eval_report.json
    # pass-through) have a stable wire-shape independent of the
    # ``synthesis.last_run`` payload (which carries the full SynthesisStats
    # dump). Reason histogram keys are alphabetised on emit so a
    # run-vs-run diff is byte-stable. Always present (zeros on a
    # legacy / empty run) so a missing block is unambiguously a
    # legacy-corpus signal in downstream consumers.
    config["statistics"]["promotion_ladder"] = {
        "candidate_pairs_total": stats.candidate_pairs_total,
        "validated_pairs_total": stats.validated_pairs_total,
        "trainable_pairs_total": stats.trainable_pairs_total,
        "rejected_promotion_pairs": stats.rejected_promotion_pairs,
        "promotion_rejection_reasons": {
            k: stats.promotion_rejection_reasons[k]
            for k in sorted(stats.promotion_rejection_reasons)
        },
    }
    config.setdefault("synthesis", {})
    config["synthesis"]["last_run"] = stats.as_dict()
    if holdout_identity is not None:
        config["synthesis"]["holdout_identity"] = dict(holdout_identity)

    tmp = dataset_config_path.with_suffix(dataset_config_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    tmp.replace(dataset_config_path)
    return config


# ---------------------------------------------------------------------------
# W3.H sub-task H5: synthesis_summary.json sidecar
# ---------------------------------------------------------------------------

_SYNTHESIS_SUMMARY_SCHEMA_VERSION = "v1"


def _emit_synthesis_summary_sidecar(
    sidecar_path: Path,
    *,
    stats: "SynthesisStats",
    provider: str,
    course_code: str,
) -> None:
    """Write the W3.H H5 ``training_specs/synthesis_summary.json`` sidecar.

    Sibling to ``dataset_config.json`` carrying the canonical
    ``source_coverage`` block for the assessment-items → training-pairs
    arrow. The sidecar is canonical (W3.G master aggregator consumes
    it); ``dataset_config.json::statistics.promotion_ladder`` carries
    the same numbers but in a less aggregator-friendly shape.

    Atomic write via tmpfile + rename so a partial sidecar never
    exists on disk. Schema: ``schemas/training/synthesis_summary.schema.json``.
    """
    from lib.governance.source_coverage import build_source_coverage

    # items_consumed: every chunk eligible for paraphrase synthesis
    # (the actual upstream-item denominator for this stage). Note the
    # plan §W3.H phrasing "items_consumed / pairs_emitted" treats
    # items as the denominator, not pairs as the numerator divided
    # by items — items are 1:N to pairs by construction.
    items_consumed = max(0, int(stats.chunks_eligible))

    # pairs_emitted: ALL pair surfaces summed. Deterministic
    # generators (kg_metadata / violation / abstention /
    # schema_translation) are independent of the chunk loop so they
    # contribute to the emitted count even when chunks_eligible is
    # zero (which would push coverage_pct above 1.0; bounded by the
    # build helper).
    pairs_emitted = (
        int(stats.instruction_pairs_emitted)
        + int(stats.preference_pairs_emitted)
        + int(stats.misconception_dpo_pairs_emitted)
        + int(stats.kg_metadata_pairs_emitted)
        + int(stats.violation_pairs_emitted)
        + int(stats.abstention_pairs_emitted)
        + int(stats.schema_translation_pairs_emitted)
    )

    # Drop reasons: union of the W2.E promotion rejection histogram
    # (post-W3.B, lives in stats.promotion_rejection_reasons) and the
    # legacy per-pair rejection histogram. Both keys are alphabetised
    # by build_source_coverage on emit so a byte-diff between runs is
    # stable regardless of insertion order.
    drop_reasons: Dict[str, int] = {}
    for k, v in (stats.promotion_rejection_reasons or {}).items():
        try:
            drop_reasons[str(k)] = drop_reasons.get(str(k), 0) + int(v)
        except (TypeError, ValueError):
            continue
    for k, v in (stats.rejected_reasons or {}).items():
        try:
            drop_reasons[str(k)] = drop_reasons.get(str(k), 0) + int(v)
        except (TypeError, ValueError):
            continue

    # Total drops: W2.E rejected_promotion_pairs + legacy
    # instruction_pairs_rejected + preference_pairs_rejected. The
    # build helper handles the silent-gaming check (drops attributed
    # without a reason fire INTERNAL_DROP_REASON_MISSING).
    dropped_count = (
        int(stats.rejected_promotion_pairs)
        + int(stats.instruction_pairs_rejected)
        + int(stats.preference_pairs_rejected)
    )

    coverage_block = build_source_coverage(
        consumed_count=items_consumed,
        emitted_count=pairs_emitted,
        drop_reasons=drop_reasons,
        dropped_count=dropped_count,
        label=f"synthesis_summary:{course_code}",
    )

    summary = {
        "schema_version": _SYNTHESIS_SUMMARY_SCHEMA_VERSION,
        "course_code": course_code,
        "provider": str(provider),
        "instruction_pairs_emitted": int(stats.instruction_pairs_emitted),
        "preference_pairs_emitted": int(stats.preference_pairs_emitted),
        "misconception_dpo_pairs_emitted": int(stats.misconception_dpo_pairs_emitted),
        "kg_metadata_pairs_emitted": int(stats.kg_metadata_pairs_emitted),
        "violation_pairs_emitted": int(stats.violation_pairs_emitted),
        "abstention_pairs_emitted": int(stats.abstention_pairs_emitted),
        "schema_translation_pairs_emitted": int(stats.schema_translation_pairs_emitted),
        "chunks_total": int(stats.chunks_total),
        "chunks_eligible": int(stats.chunks_eligible),
        "candidate_pairs_total": int(stats.candidate_pairs_total),
        "validated_pairs_total": int(stats.validated_pairs_total),
        "trainable_pairs_total": int(stats.trainable_pairs_total),
        "rejected_promotion_pairs": int(stats.rejected_promotion_pairs),
        "source_coverage": coverage_block,
    }

    # Reject-mining funnel (TRAINFORGE_DPO_MINE_REJECTS). Emitted ONLY when the
    # flag is on/shadow -- ``stats.reject_mining`` is left empty when mining is
    # off, so the key is absent and a legacy sidecar stays byte-identical.
    #
    # This is load-bearing for the documented rollout, not telemetry garnish:
    # the whole point of ``shadow`` mode is to measure yield before changing the
    # corpus, and an operator cannot act on a number that only ever existed in a
    # log line. ``synthesis_summary.json`` is where they read `emitted` to decide
    # whether the corpus would clear the trainer's ``min_dpo_pairs: 50`` floor.
    # Optional property on the schema; no ``schema_version`` bump (an
    # absent-or-present optional key cannot break a reader).
    if stats.reject_mining:
        summary["reject_mining"] = {
            key: (value if isinstance(value, str) else int(value))
            for key, value in sorted(stats.reject_mining.items())
        }

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(sidecar_path)


# ---------------------------------------------------------------------------
# Decision-capture helpers
# ---------------------------------------------------------------------------

def _last_event_id(capture: DecisionCapture) -> str:
    """Return the event_id of the most recent decision written via ``capture``.

    ``DecisionCapture.log_decision`` appends to ``capture.decisions``; we pull
    ``event_id`` off the tail.

    Do NOT fall back to ``""`` when ``capture.decisions`` is empty: that
    empty string rides into the emitted JSONL as ``decision_capture_id: ""``,
    a schema-violating value that breaks strict-mode pair validation
    downstream. Every production call site logs a decision before it asks for
    the event_id, so empty here is unambiguously a bug — fail loud rather than
    poisoning the corpus.
    """
    if not capture.decisions:
        raise RuntimeError(
            "no decisions logged: _last_event_id called against an empty "
            "DecisionCapture. The synthesis loop must capture.log_decision(...) "
            "before requesting the event_id; an empty fallback would emit a "
            "schema-violating decision_capture_id=\"\" in the training pair."
        )
    return str(capture.decisions[-1].get("event_id", ""))


# ---------------------------------------------------------------------------
# deterministic-pair audit-stamp helper
# ---------------------------------------------------------------------------
#
# The four deterministic generators (kg_metadata, violation, abstention,
# schema_translation) emit pairs whose ``chunk_id`` is a synthetic anchor
# (``concept_alpha``, ``module:week_01``, ``shacl_test_graph_001``, …) that
# does NOT resolve against ``imscc_chunks/chunks.jsonl``. The post-emit
# gate-runner walks (W2.E pair_promotion, W4.A pair_claim_support, W4.B
# pair_lo_refs, W4.C pair_objective_delivery) treat the absent audit
# fields as evidence the per-pair filters never ran and critical-fail.
#
# Bypassing the per-pair filters is structurally correct for these
# generators — every pair is oracle-grounded by construction (pyshacl
# for violation, graph-membership truth for kg_metadata, fixture-pinned
# for schema_translation, concept-absence for abstention). What's broken
# today is the missing audit fields. This helper stamps the audit shape
# directly so the gate-runner walks see "the per-pair filters ran and
# legitimately bypassed this pair" rather than "the per-pair filters
# never ran at all".
#
# The ``"skipped": "deterministic_template"`` discriminator on
# ``pair_lo_resolution`` is the key mechanism: the W4.B walk reads it
# and short-circuits the per-pair chunk-id resolution that would
# otherwise fire ``PAIR_CHUNK_NOT_FOUND`` on every synthetic chunk_id.

def _stamp_deterministic_pair_audit_fields(
    pair: Dict[str, Any],
    *,
    capture: DecisionCapture,
) -> None:
    """Stamp the W2.E + W4.A + W4.B + W4.C audit fields on a deterministic-
    template pair so the post-emit gate-runner walks recognise it as
    legitimately-bypassed rather than missing-status / phantom / unsupported.

    Mutates the pair in place. Emits one
    ``deterministic_pair_audit_stamp`` decision-capture event per pair so
    the audit trail records WHY each per-pair filter was bypassed.

    Audit fields stamped (mirroring the graceful-pass arms in the four
    per-pair filters):

    - ``promotion_status="validated"`` (W2.E bypass — oracle-grounded by
      construction, the 7-criterion check would always pass).
    - ``per_claim_support=[]`` (W4.A graceful-pass; empty list NOT None
      so the audit walk's ``"per_claim_support" in row`` check passes).
    - ``claim_support_rate=None``, ``claim_contradicted_rate=None``,
      ``deps_missing=False`` (W4.A graceful-pass shape at
      ``pair_claim_support.py:632-655``).
    - ``pair_lo_resolution={"declared_los": pair["lo_refs"],
      "chunk_los": [], "phantom_los": [], "skipped":
      "deterministic_template"}`` (W4.B per-pair filter audit shape at
      ``pair_lo_refs.py:222-228``; ``"skipped"`` discriminator is the
      mechanism the W4.B validate walk reads to short-circuit the
      per-pair chunk-id resolution).
    - ``pair_objective_alignment=None``,
      ``pair_objective_alignment_pass_rate=None`` (W4.C deps-missing
      arm shape at ``pair_objective_delivery.py:735-738``).
    """
    template_id = str(pair.get("template_id") or "")
    chunk_id = str(pair.get("chunk_id") or "")
    lo_refs_raw = pair.get("lo_refs") or pair.get("learning_outcome_refs") or []
    lo_refs = [str(r) for r in lo_refs_raw if isinstance(r, (str, int))]

    pair["promotion_status"] = "validated"
    pair["per_claim_support"] = []
    pair["claim_support_rate"] = None
    pair["claim_contradicted_rate"] = None
    pair["deps_missing"] = False
    pair["pair_lo_resolution"] = {
        "declared_los": list(lo_refs),
        "chunk_los": [],
        "phantom_los": [],
        "skipped": "deterministic_template",
    }
    pair["pair_objective_alignment"] = None
    pair["pair_objective_alignment_pass_rate"] = None

    capture.log_decision(
        decision_type="deterministic_pair_audit_stamp",
        decision="stamped audit fields on deterministic-template pair",
        rationale=(
            f"Stamped audit fields on deterministic-template pair "
            f"(template_id={template_id}, chunk_id={chunk_id}, "
            f"lo_refs={lo_refs}, kind=instruction). Bypassed W2.E + W4.A + "
            f"W4.B + W4.C per-pair filters because the four deterministic "
            f"generators are oracle-grounded by construction (pyshacl / "
            f"graph-membership / fixture-pinned / concept-absence) and "
            f"their synthetic chunk_ids do not resolve against "
            f"imscc_chunks/chunks.jsonl."
        ),
    )


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SFT data program (S3/S4/S5): env-flag resolvers + holdout-reduced-graph +
# assessment / concept-graph artifact resolution.
# ---------------------------------------------------------------------------

def _resolve_bool_env(name: str, kwarg_value: bool) -> bool:
    """Parse-with-fallback OR of a kwarg and an env flag.

    The kwarg wins when truthy; otherwise the env var (``1``/``true``/``yes``/
    ``on``, case-insensitive) enables the behavior. Garbage / unset -> kwarg.
    This lets the ``textbook_to_course`` pipeline drive the assessment/graph
    SFT arms via environment (the ``_synthesize_training`` seam passes no
    ``with_*`` kwarg for them) with byte-identical default-off behavior.
    """
    if bool(kwarg_value):
        return True
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _normalize_holdout_rel(rel: Any) -> str:
    """Normalize a relation label so pedagogy (``relation_type``) and concept
    (``type``) edge naming compare equal (``prerequisite_of`` -> ``prerequisite``,
    ``related_to`` -> ``related-to``)."""
    r = str(rel or "").strip().lower()
    for suf in ("_of", "-of"):
        if r.endswith(suf):
            r = r[: -len(suf)]
            break
    return r.replace("_", "-")


def _load_withheld_edge_index(
    corpus_dir: Path,
    holdout_path: Optional[Path] = None,
) -> Set[Tuple[str, str, str]]:
    """Load the pedagogy-graph holdout split's withheld edges as a normalized
    identity set ``{(source, target, rel_norm)}``.

    Returns an empty set when no ``eval/holdout_split.json`` exists (legacy /
    pre-holdout corpus) or it can't be parsed — reduction then no-ops
    (byte-identical). This is the pair-side enforcement of the S4 design rule:
    every graph->pair generator consumes the holdout-REDUCED graph.
    """
    p = Path(holdout_path) if holdout_path is not None else (
        Path(corpus_dir) / "eval" / "holdout_split.json"
    )
    if not p.exists():
        return set()
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("holdout-reduced-graph: failed to read %s: %s", p, exc)
        return set()
    index: Set[Tuple[str, str, str]] = set()
    for e in payload.get("withheld_edges", []) or []:
        if not isinstance(e, dict):
            continue
        s = e.get("source")
        t = e.get("target")
        if s is None or t is None:
            continue
        index.add((str(s), str(t), _normalize_holdout_rel(e.get("relation_type"))))
    return index


def _reduce_graph_by_holdout(
    graph: Dict[str, Any],
    withheld_index: Set[Tuple[str, str, str]],
) -> Tuple[Dict[str, Any], int]:
    """Return a shallow-copied graph with every withheld edge removed.

    Matches an edge by normalized ``(source, target, relation)`` identity,
    reading ``relation_type`` (pedagogy) OR ``type`` (concept). Byte-identical
    (returns the input object) when the index is empty or nothing matches.
    Guarantees a withheld edge can never reach a downstream pair generator.
    """
    if not withheld_index or not isinstance(graph, dict):
        return graph, 0
    edges = graph.get("edges")
    if not isinstance(edges, list):
        return graph, 0
    kept: List[Any] = []
    removed = 0
    for e in edges:
        if isinstance(e, dict):
            rel = e.get("relation_type")
            if rel is None:
                rel = e.get("type")
            identity = (
                str(e.get("source")),
                str(e.get("target")),
                _normalize_holdout_rel(rel),
            )
            if identity in withheld_index:
                removed += 1
                continue
        kept.append(e)
    if removed == 0:
        return graph, 0
    reduced = dict(graph)
    reduced["edges"] = kept
    return reduced, removed


def _ensure_holdout_split_for_graph_pairs(
    corpus_dir: Path,
    holdout_split_path: Optional[Path] = None,
) -> None:
    """SFT program S4 sequencing fix: pre-build ``eval/holdout_split.json``
    BEFORE any graph->pair generator emits.

    The S4 design rule ("a withheld edge can never train") is only
    enforceable when the holdout split exists at pair-emission time. The
    in-build ``training_synthesis`` phase runs BEFORE any eval, so on a
    fresh corpus the split did not exist yet, ``_load_withheld_edge_index``
    resolved EMPTY, and the graph->pair generators trained on 100% of the
    edges — including the exact edges the Tier-2 eval would LATER withhold
    (``slm_eval_harness.run_all`` lazily builds the split at eval time,
    after training). Building it here closes that train-on-test race:

    * deterministic — ``HoldoutBuilder`` pins ``seed=42`` and reruns over
      the same graph are byte-identical, so a downstream lazy rebuild over
      the archived (unchanged) graph reproduces the same split;
    * respected downstream — ``run_all`` only builds when the file is
      absent, so a pre-built split is consumed verbatim.

    No-ops when: an explicit ``holdout_split_path`` override was supplied
    (the caller owns that file), the split already exists, or no pedagogy
    graph is resolvable (legacy corpus — the reduction then no-ops exactly
    as before). Best-effort: any build failure logs a warning and falls
    through to the legacy empty-index path (never aborts synthesis).
    """
    if holdout_split_path is not None:
        return
    split_path = Path(corpus_dir) / "eval" / "holdout_split.json"
    if split_path.exists():
        return
    try:
        from Trainforge.eval.holdout_builder import HoldoutBuilder

        built = HoldoutBuilder(Path(corpus_dir)).build()
        logger.info(
            "SFT program S4: pre-built pedagogy-graph holdout split at %s "
            "before graph->pair emission (train-on-test guard).", built,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "SFT program S4: holdout-split pre-build skipped (%s); the "
            "graph->pair holdout reduction will no-op on this corpus.", exc,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, never abort emit
        logger.warning(
            "SFT program S4: holdout-split pre-build failed (%s); the "
            "graph->pair holdout reduction will no-op on this corpus.", exc,
        )


def _resolve_assessment_docs(
    corpus_dir: Path,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Locate the W10 ``assessments.json`` + instructor ``answer_key.json`` for
    the assessment->SFT emitter. Returns ``(assessments_doc, answer_key_doc)``;
    either may be ``None`` when absent."""
    def _load(cands: Sequence[Path]) -> Optional[Any]:
        for c in cands:
            if c.exists() and c.is_file():
                try:
                    return json.loads(c.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("assessment_sft: failed to read %s: %s", c, exc)
        return None

    corpus_dir = Path(corpus_dir)
    assessments = _load([
        corpus_dir / "training_specs" / "assessments.json",
        corpus_dir / "assessments.json",
        corpus_dir / "06_assessments" / "assessments.json",
    ])
    answer_key = _load([
        corpus_dir / "training_specs" / "answer_key.json",
        corpus_dir / "answer_key.json",
        corpus_dir / "06_assessments" / "answer_key.json",
    ])
    answer_key = answer_key if isinstance(answer_key, dict) else None
    return assessments, answer_key


def _resolve_concept_graph_path(corpus_dir: Path) -> Optional[Path]:
    """Locate ``concept_graph_semantic.json`` for the concept-graph->SFT
    generator (LibV2 archive ``graph/`` layout first)."""
    corpus_dir = Path(corpus_dir)
    for rel in (
        "graph/concept_graph_semantic.json",
        "concept_graph_semantic.json",
        "pedagogy/concept_graph_semantic.json",
    ):
        p = corpus_dir / rel
        if p.exists():
            return p
    return None


def _resolve_pedagogy_graph_path(
    corpus_dir: Path,
    explicit: Optional[Path] = None,
) -> Optional[Path]:
    """Locate ``pedagogy_graph.json`` next to a Trainforge corpus.

    Order tried (first hit wins):
      1. ``explicit`` if supplied (caller override / tests).
      2. ``<corpus_dir>/graph/pedagogy_graph.json`` (LibV2 archive layout).
      3. ``<corpus_dir>/pedagogy/pedagogy_graph.json`` (Trainforge run output).
      4. ``<corpus_dir>/pedagogy_graph.json``.
    Returns None when no graph is on disk (caller decides whether that is
    fatal — it is when ``--curriculum-from-graph`` is set).
    """
    if explicit is not None:
        return Path(explicit) if Path(explicit).exists() else None
    candidates = [
        corpus_dir / "graph" / "pedagogy_graph.json",
        corpus_dir / "pedagogy" / "pedagogy_graph.json",
        corpus_dir / "pedagogy_graph.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# per-pair resume checkpoint sidecar
# ---------------------------------------------------------------------------
#
# A long local-LLM rebuild (10+ hours on a 14B-Q4 paraphrase pass over a
# few hundred chunks) that crashes mid-loop loses every emitted pair on
# restart: the chunk loop re-paraphrases everything from scratch
# regardless of what the ``.jsonl.in_progress`` sidecar already holds. That
# sidecar is observability only — its file handle is opened in ``"w"`` mode,
# which truncates on restart.
#
# This sidecar is the resume cache. One JSON line per ACCEPTED pair
# (post-validation, post-decoration, ready to land in the final
# ``instruction_records`` / ``preference_records`` buffers). On restart
# the loader rebuilds a ``(chunk_id, kind, variant_index)`` → record
# cache and the chunk loop short-circuits past every cached key, never
# dispatching to the LLM for it.
#
# Mirrors the ``align_chunks.py::_load_teaching_role_checkpoint`` /
# ``_append_teaching_role_checkpoint`` pattern: tolerant load (malformed
# lines / schema-version mismatches drop without poisoning the run),
# append + flush per emit, unlink on clean exit, preserve on every
# exception path so postmortem inspection still works.

_SYNTHESIS_CHECKPOINT_SCHEMA_VERSION = "v1"
_PAIR_CHECKPOINT_LOCK = threading.Lock()
_SYNTHESIS_REJECTION_CONTRACT_VERSION = "v1"

# GENERATION_CONTRACT_FILES: files that decide MODEL CALL outcomes.
# Fingerprint keys accepted rows + generation journal.  Changing these
# files invalidates all resume-cached pairs. Verdict-policy changes
# (validators, thresholds, classifiers, embedding) do NOT regenerate
# previously-accepted pairs — only their verdict verdict digest changes.
_GENERATION_CONTRACT_FILES = (
    "Trainforge/synthesis/synthesize_training.py",
    "Trainforge/synthesis/synthesis_eligibility.py",
    "Trainforge/generators/providers/_synthesis_provider.py",
    "Trainforge/generators/providers/_synthesis_common.py",
    "Trainforge/generators/providers/_openai_compatible_client.py",
    "Trainforge/generators/staged/provider.py",
    # The micro contract's twin of staged/provider.py. It decides
    # model-call outcomes exactly as its V4 sibling does, AND it owns
    # ``micro_preference_eligibility`` — the preference-admission predicate
    # that synthesis_eligibility.py delegates to for BOTH contracts — so an
    # edit here changes which chunks generate preference pairs. Without this
    # entry a resumed run appended post-edit rows to a pre-edit corpus.
    "Trainforge/generators/staged/micro.py",
    "Trainforge/generators/providers/_local_provider.py",
    "Trainforge/generators/pairs/instruction.py",
    "Trainforge/generators/pairs/preference.py",
    "Trainforge/generators/staged/window_contract.py",
    "Trainforge/generators/staged/objective_contract.py",
    "Trainforge/synthesis/synthesis_contract_guard.py",
    "Trainforge/synthesis/synthesis_concurrency.py",
    "Trainforge/synthesis/synthesis_journal.py",
    "Trainforge/synthesis/synthesis_fresh_start.py",
    "lib/decision_capture.py",
    "lib/utils/jsonl.py",
    "schemas/knowledge/instruction_pair.schema.json",
    "schemas/knowledge/preference_pair.schema.json",
)

# VERDICT_POLICY_FILES: files that decide VERDICT outcomes (pass/fail).
# RECORDED on the row but NOT used to key fingerprint. Changing these
# files re-evaluates ALL rows' verdicts without regenerating any pair.
#
# The tuple's name predates a third case: a file that DERIVES rows from
# already-generated text (``synthesis_reject_mining``) rather than generating
# or judging. It belongs here, not in the generation tuple, because the
# operative question either tuple answers is "would resuming mix rows from two
# implementations into one corpus?" — reject mining is recomputed from scratch
# every run out of the full (resume-complete) reject pool and is never
# resume-cached, so a mid-run edit changes only the derived output and cannot
# mix generations. Putting it in _GENERATION_CONTRACT_FILES would re-key the
# generation fingerprint and force every paused run to archive-and-restart for
# a change that provably cannot alter a single model call.
_VERDICT_POLICY_FILES = (
    "lib/validators/pair/promotion.py",
    "lib/validators/_pair_promotion_stages/thresholds.py",
    "lib/validators/pair/claim_support.py",
    "lib/validators/pair/_claim_support_thresholds.py",
    "lib/validators/pair/lo_refs.py",
    "lib/validators/pair/objective_delivery.py",
    # The claim-support validator normalizes both NLI inputs through
    # ``lib/semantik/math_fold.py::normalize_math_notation``, so that file
    # decides verdicts too and belongs in the recorded policy digest.
    "lib/semantik/math_fold.py",
    "lib/classifiers/nli_classifier.py",
    "lib/classifiers/bloom_bert_ensemble.py",
    "lib/classifiers/bloom_zero_shot.py",
    "lib/embedding/sentence_embedder.py",
    "lib/ontology/bloom.py",
    "lib/utils/objective_delivery_axes/status.py",
    "lib/utils/objective_delivery_axes/nli.py",
    "lib/utils/objective_delivery_axes/bloom_gap.py",
    "lib/utils/objective_delivery_axes/verb_cooccurrence.py",
    "Trainforge/synthesis_reject_mining.py",
)

# Legacy alias for backward compatibility; used for digest recording only.
_SYNTHESIS_REJECTION_CONTRACT_FILES = (
    _GENERATION_CONTRACT_FILES + _VERDICT_POLICY_FILES
)

# Runtime policy is allowlisted by prefix because synthesis, validator,
# classifier, and embedding knobs can change generated content or terminal
# verdicts.  This denylist stays intentionally tiny: these exact values affect
# only scheduling/reporting, so hashing them would waste valid checkpoints
# when an operator tunes throughput between resumes.
_SYNTHESIS_FINGERPRINT_OPERATIONAL_ENV = frozenset({
    "TRAINFORGE_SYNTHESIS_MAX_CONCURRENT",
    "TRAINFORGE_EVAL_PROGRESS_EVERY",
    "TRAINFORGE_SYNTHESIS_RUN_CONTRACT_SHA256",
    "TRAINFORGE_SYNTHESIS_CONTRACT_COMPONENTS_SHA256",
})
_SYNTHESIS_FINGERPRINT_POLICY_PREFIXES = (
    "TRAINFORGE_",
    "ED4ALL_NLI_",
    "ED4ALL_EMBED",
    "ED4ALL_BLOOM_",
)


def _synthesis_runtime_policy_identity(
    environment: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return only output/verdict-affecting runtime policy.

    Operational concurrency and telemetry cadence are excluded explicitly.
    Transport controls such as ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` are not
    in the policy-prefix allowlist: they alter how long a request may wait,
    not its intended model request or validator contract. Model, prompt,
    synthesis, NLI, embedding, and Bloom-policy inputs remain represented
    here or in the fingerprint's explicit model/endpoint/generation fields.
    """

    source = os.environ if environment is None else environment
    return {
        key: value
        for key, value in sorted(source.items())
        if key not in _SYNTHESIS_FINGERPRINT_OPERATIONAL_ENV
        and key.startswith(_SYNTHESIS_FINGERPRINT_POLICY_PREFIXES)
    }


def _synthesis_static_contract_components(
    *,
    provider: str,
    model_id: str,
    judge_identity: Optional[Mapping[str, str]] = None,
    endpoint_identity: str = "",
    generation_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture the immutable, diagnostic synthesis contract components.

    ONLY GENERATION_CONTRACT_FILES feeds the fingerprint that keys accepted rows.
    Verdict-policy changes are RECORDED on the row but do NOT regenerate it.
    """

    contract_files: Dict[str, str] = {}
    for relative_path in _GENERATION_CONTRACT_FILES:
        path = PROJECT_ROOT / relative_path
        try:
            contract_files[relative_path] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        except OSError:
            # Fail safe: a missing contract file still produces a deterministic
            # distinct fingerprint rather than reusing an old rejection.
            contract_files[relative_path] = "missing"
    return {
        "version": _SYNTHESIS_REJECTION_CONTRACT_VERSION,
        "provider": str(provider),
        "model_id": str(model_id),
        "endpoint_sha256": _normalized_sha256(str(endpoint_identity)),
        "generation_params_sha256": _normalized_sha256(
            dict(generation_params or {})),
        # Runtime policy can alter a rejection without changing source. Fold
        # every synthesis/validator/classifier knob into the identity; values
        # remain inside the SHA input and are never persisted.
        "runtime_policy": _synthesis_runtime_policy_identity(),
        "judge_identity": dict(judge_identity or {
            "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            "nli_revision": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
            "sentence_embedder_model": "all-MiniLM-L6-v2",
        }),
        "files": contract_files,
    }


def _verdict_policy_digest() -> Dict[str, Any]:
    """Digest the verdict-policy files for the RECORD, never for the key.

    The split above stopped FINGERPRINTING these files; without this they
    would also stop being RECORDED, which is the failure that made the
    original coupling look necessary: an auditor holding an accepted pair
    could no longer tell WHICH verdict policy judged it, so the only way to
    prove a pair was still valid was to re-key and regenerate it (the
    archive-and-restart loop visible in ``runtime/state/backups/``).  The digest is
    folded into ``synthesis_run_contract_components`` — which is stamped on
    decision-capture metadata, the fresh-start marker, and (via its sha) every
    journal row — but deliberately NOT into ``static_contract``, the only
    component set ``_synthesis_rejection_contract_fingerprint`` hashes.  So a
    claim-support or threshold edit changes what is recorded and changes
    nothing that can invalidate an already-accepted pair.
    """

    policy_files: Dict[str, str] = {}
    for relative_path in _VERDICT_POLICY_FILES:
        path = PROJECT_ROOT / relative_path
        try:
            policy_files[relative_path] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        except OSError:
            # A missing verdict-policy file is recorded as such rather than
            # raising: this surface must never be able to fail a run.
            policy_files[relative_path] = "missing"
    return {
        "files": policy_files,
        "sha256": _normalized_sha256(policy_files),
    }


def _normalized_sha256(value: Any) -> str:
    normalized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _synthesis_rejection_contract_fingerprint(
    *,
    provider: str,
    model_id: str,
    pair_seed: int,
    judge_identity: Optional[Mapping[str, str]] = None,
    source_chunk: Optional[Mapping[str, Any]] = None,
    endpoint_identity: str = "",
    generation_params: Optional[Mapping[str, Any]] = None,
    static_components: Optional[Mapping[str, Any]] = None,
) -> str:
    """Fingerprint one unit from an immutable preflight contract."""

    components = dict(static_components or _synthesis_static_contract_components(
        provider=provider,
        model_id=model_id,
        judge_identity=judge_identity,
        endpoint_identity=endpoint_identity,
        generation_params=generation_params,
    ))
    payload = {
        "static_contract": components,
        "pair_seed": int(pair_seed),
        "focused_source_sha256": _normalized_sha256(dict(source_chunk or {})),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_synthesis_pairs_checkpoint(
    path: Optional[Path],
    *,
    expected_fresh_start_id: Optional[str] = None,
    expected_marker_digest: Optional[str] = None,
    expected_holdout_identity: Optional[Mapping[str, str]] = None,
    expected_run_contract_sha256: Optional[str] = None,
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    """Tolerant loader for the per-pair resume sidecar.

    Returns a ``{(chunk_id, kind, variant_index): record}`` map. Malformed
    JSON lines are skipped silently. Records whose ``schema_version``
    doesn't match :data:`_SYNTHESIS_CHECKPOINT_SCHEMA_VERSION` are skipped
    with a ``logger.warning`` so the operator can see why a stale
    checkpoint isn't being honoured. Returns empty dict when ``path`` is
    ``None`` or the file doesn't exist (back-compat — first-run case).

    The caller decides what to do with the loaded cache; this function
    only loads, never deletes or rewrites the file.
    """
    if path is None or not path.exists():
        return {}
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if expected_fresh_start_id is not None:
                    raise RuntimeError(
                        "pair checkpoint contains a malformed row in "
                        "fresh-start mode"
                    )
                continue
            if obj.get("schema_version") != _SYNTHESIS_CHECKPOINT_SCHEMA_VERSION:
                if expected_fresh_start_id is not None:
                    raise RuntimeError(
                        "pair checkpoint contains an incompatible row in "
                        "fresh-start mode"
                    )
                logger.warning(
                    "Synthesis checkpoint schema_version mismatch (expected %r, "
                    "got %r) — skipping entry",
                    _SYNTHESIS_CHECKPOINT_SCHEMA_VERSION,
                    obj.get("schema_version"),
                )
                continue
            if expected_fresh_start_id is not None and (
                obj.get("fresh_start_id") != expected_fresh_start_id
                or obj.get("fresh_start_marker_digest") != expected_marker_digest
            ):
                raise RuntimeError(
                    "pair checkpoint fresh-start identity mismatch; stale or "
                    "copied dispositions cannot be replayed"
                )
            if (
                expected_holdout_identity is not None
                and obj.get("synthesis_holdout_identity")
                != dict(expected_holdout_identity)
            ):
                raise RuntimeError(
                    "pair checkpoint holdout identity mismatch; stale or "
                    "copied dispositions cannot be replayed"
                )
            if (
                expected_run_contract_sha256 is not None
                and obj.get("synthesis_run_contract_sha256")
                != expected_run_contract_sha256
            ):
                raise RuntimeError(
                    "pair checkpoint synthesis run contract mismatch; "
                    "mixed-version dispositions cannot be replayed"
                )
            chunk_id = obj.get("chunk_id")
            kind = obj.get("kind")
            variant = obj.get("variant_index")
            if chunk_id and kind in ("instruction", "preference"):
                try:
                    variant_int = int(variant or 0)
                except (TypeError, ValueError):
                    continue
                key = (str(chunk_id), str(kind), variant_int)
                if key in cache and expected_run_contract_sha256 is not None:
                    raise RuntimeError(
                        "pair checkpoint contains duplicate terminal semantic "
                        f"key {key!r}"
                    )
                cache[key] = obj
    return cache


def _append_synthesis_pairs_checkpoint(
    fh: Optional[Any],
    *,
    chunk_id: str,
    kind: str,
    variant_index: int,
    pair: Optional[Dict[str, Any]],
    provider: str,
    seed: Optional[int] = None,
    disposition: str = "accepted",
    reason: Optional[str] = None,
    contract_fingerprint: Optional[str] = None,
    rejection_evidence: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append one terminal pair disposition with immediate flush.

    Always a no-op when ``fh`` is None (resume cache disabled). Schema
    version is pinned by :data:`_SYNTHESIS_CHECKPOINT_SCHEMA_VERSION`;
    bumping it loudly invalidates pre-bump checkpoints via the
    ``logger.warning`` path in :func:`_load_synthesis_pairs_checkpoint`.

    ``accepted`` records retain the emitted pair for replay. ``rejected``
    records are generated candidates that failed a quality validator.
    ``ineligible`` records are deterministic pre-dispatch exclusions and are
    deliberately counted separately. Legacy v1 records without
    ``disposition`` are interpreted as accepted by callers.
    """
    if fh is None:
        return
    record = {
        "schema_version": _SYNTHESIS_CHECKPOINT_SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "kind": kind,
        "variant_index": variant_index,
        "pair": pair,
        "provider": provider,
        "seed": seed,
        "disposition": disposition,
        "reason": reason,
        "contract_fingerprint": contract_fingerprint,
    }
    fresh_start_id = getattr(fh, "_fresh_start_id", None)
    if fresh_start_id is not None:
        record["fresh_start_id"] = fresh_start_id
        record["fresh_start_marker_digest"] = getattr(
            fh, "_fresh_start_marker_digest", None,
        )
    run_contract_sha256 = getattr(fh, "_synthesis_run_contract_sha256", None)
    if run_contract_sha256 is not None:
        record["synthesis_run_contract_sha256"] = run_contract_sha256
    component_manifest_sha256 = getattr(
        fh, "_synthesis_contract_components_sha256", None,
    )
    if component_manifest_sha256 is not None:
        record["synthesis_contract_components_sha256"] = (
            component_manifest_sha256
        )
    holdout_identity = getattr(fh, "_synthesis_holdout_identity", None)
    if holdout_identity is not None:
        record["synthesis_holdout_identity"] = dict(holdout_identity)
    if rejection_evidence:
        record["rejection_evidence"] = dict(rejection_evidence)
    with _PAIR_CHECKPOINT_LOCK:
        key = (str(chunk_id), str(kind), int(variant_index))
        terminal_keys = getattr(fh, "_terminal_semantic_keys", None)
        if terminal_keys is None:
            terminal_keys = set()
            fh._terminal_semantic_keys = terminal_keys
        if (
            getattr(fh, "_enforce_terminal_uniqueness", False)
            and key in terminal_keys
        ):
            raise RuntimeError(
                "refusing duplicate pair terminal semantic key "
                f"{key!r}"
            )
        _utils_append_jsonl(fh, record, sort_keys=False)
        terminal_keys.add(key)


#: Fields projected from a terminal checkpoint record into
#: ``synthesis_dispositions.jsonl``. Deliberately excludes ``pair`` — a
#: rejected row has none, and an ineligible row was never generated.
_DISPOSITION_PROJECTED_FIELDS = (
    "schema_version",
    "chunk_id",
    "kind",
    "variant_index",
    "provider",
    "seed",
    "disposition",
    "reason",
    "contract_fingerprint",
)


def _project_terminal_dispositions(
    records: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Project rejected / ineligible checkpoint records into the operator file.

    The checkpoint sidecar is unlinked on a clean exit, so whatever this
    projection drops is gone for good. It previously dropped
    ``rejection_evidence``, which is why a rejected pair left only a reason
    string on disk and an audit of 150 claim-support rejections could
    hand-adjudicate 14 of them: the per-sentence NLI scores that decided each
    verdict existed only inside a deleted file.

    ``rejection_evidence`` is carried through when present and OMITTED (not
    nulled) when absent, so a row that never had one is byte-identical to
    before and existing readers are unaffected. Bounding the evidence is the
    producer's job (see
    ``lib.validators.pair.claim_support.summarize_claim_support_rejection``);
    this projection copies what it is given.
    """
    projected: List[Dict[str, Any]] = []
    for record in records:
        if record.get("disposition") not in {"rejected", "ineligible"}:
            continue
        row: Dict[str, Any] = {
            field: record.get(field)
            for field in _DISPOSITION_PROJECTED_FIELDS
        }
        evidence = record.get("rejection_evidence")
        if evidence:
            row["rejection_evidence"] = evidence
        projected.append(row)
    return projected


def _checkpoint_terminal_rejection(
    fh: Optional[Any],
    *,
    chunk_id: str,
    kind: str,
    variant_index: int,
    provider: str,
    seed: int,
    reason: Optional[str],
    contract_fingerprint: str,
    rejection_evidence: Optional[Mapping[str, Any]] = None,
    pair: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a deterministic rejection so resume does not pay for it again.

    ``pair`` is ``None`` on every legacy path and whenever reject mining is
    off, which is the default — the row is then byte-identical to the one this
    function has always written. When
    ``Trainforge.synthesis.synthesis_reject_mining.build_capture_payload`` returns a
    payload (mining enabled, ``claim_support`` stage, both sides inside the
    ``preference_pair`` length band) the full rejected completion rides on the
    existing row instead of being discarded, so a later pass can mine it as a
    DPO negative. Callers must pass the payload that helper returned, never a
    raw pair, so the persisted row and the in-memory pool cannot disagree.
    """
    _append_synthesis_pairs_checkpoint(
        fh,
        chunk_id=chunk_id,
        kind=kind,
        variant_index=variant_index,
        pair=pair,
        provider=provider,
        seed=seed,
        disposition="rejected",
        reason=str(reason or "unspecified_rejection"),
        contract_fingerprint=contract_fingerprint,
        rejection_evidence=rejection_evidence,
    )


def _checkpoint_terminal_ineligible(
    fh: Optional[Any],
    *,
    chunk_id: str,
    kind: str,
    variant_index: int,
    provider: str,
    seed: int,
    reason: Optional[str],
    contract_fingerprint: str,
) -> None:
    """Persist a deterministic pre-dispatch exclusion for exact replay."""
    _append_synthesis_pairs_checkpoint(
        fh,
        chunk_id=chunk_id,
        kind=kind,
        variant_index=variant_index,
        pair=None,
        provider=provider,
        seed=seed,
        disposition="ineligible",
        reason=str(reason or "unspecified_ineligible"),
        contract_fingerprint=contract_fingerprint,
    )


def _checkpoint_pair_matches_focus(
    pair: Any,
    focused_chunk: Dict[str, Any],
) -> bool:
    """Return whether an accepted cache record matches current LO scoping."""
    if not isinstance(pair, dict):
        return False
    focus_active = any(
        key in focused_chunk
        for key in (
            "synthesis_focus_objective",
            "synthesis_focus_skip_reason",
            "synthesis_focus_primary_ref",
        )
    )
    if not focus_active:
        return True
    # Legacy v1 accepted records predate mandatory LO audit fields. Preserve
    # their established replay contract; the live stale records this fix must
    # invalidate carry an explicit (over-broad) ``lo_refs`` array.
    if "lo_refs" not in pair:
        return True
    pair_refs = [
        str(ref).strip().lower()
        for ref in (pair.get("lo_refs") or [])
        if str(ref).strip()
    ]
    focused_refs = [
        str(ref).strip().lower()
        for ref in (focused_chunk.get("learning_outcome_refs") or [])
        if str(ref).strip()
    ]
    return bool(pair_refs) and pair_refs == focused_refs


def _checkpoint_accepted_matches_contract(
    record: Mapping[str, Any], current_fingerprint: str
) -> bool:
    """Validate modern accepted records while preserving legacy replay.

    Accepted v1 rows written before contract fingerprints existed remain
    replay-compatible. New accepted rows carry the complete contract identity,
    so real model/prompt/validator/content changes invalidate them while
    operational scheduling changes do not.
    """

    stored = record.get("contract_fingerprint")
    return stored in (None, "") or (
        isinstance(stored, str) and stored == current_fingerprint
    )


def _checkpoint_rejection_matches_contract(
    record: Dict[str, Any], current_fingerprint: str
) -> bool:
    """Return True only for a terminal rejection from the exact contract."""
    return (
        record.get("disposition") == "rejected"
        and isinstance(record.get("contract_fingerprint"), str)
        and record["contract_fingerprint"] == current_fingerprint
    )


def _checkpoint_ineligible_matches_contract(
    record: Mapping[str, Any], current_fingerprint: str
) -> bool:
    return (
        record.get("disposition") == "ineligible"
        and isinstance(record.get("contract_fingerprint"), str)
        and record["contract_fingerprint"] == current_fingerprint
    )


def _resolve_cached_accepted_pair(
    cached_record: Mapping[str, Any],
    current_contract: str,
    focused_chunk: Dict[str, Any],
    *,
    cache_key: Any = None,
    invalidations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return the replayable accepted pair, or ``{}`` when it must regenerate.

    FAIL-CLOSED GUARD. A ``rejected`` checkpoint row can now carry a ``pair``
    (reject mining). The instruction replay branch warns on a rejected row
    whose contract fingerprint does not match and then FALLS THROUGH to the
    accepted branch, where :func:`_checkpoint_accepted_matches_contract`
    returns True for a legacy ``None``/``""`` fingerprint. Before capture that
    degraded harmlessly to ``{}`` because a rejected row's ``pair`` was always
    ``null``; the moment rejected rows carry one, a rejected completion would
    be replayed into ``instruction_records`` AS AN ACCEPTED PAIR. Refuse on
    disposition first, before any contract check can admit it.

    ``invalidations`` is an optional sink for the reasons the cached entry was
    refused. The preference branch uses it to clear its ``_pref_cache_hit``
    flag on exactly the two cases that cleared it before (contract mismatch,
    focus mismatch) and NOT on a contract-compatible row whose pair is simply
    absent — which never cleared it.
    """

    def _invalidate(reason: str) -> None:
        if invalidations is not None:
            invalidations.append(reason)

    if cached_record.get("disposition") == "rejected":
        stored_pair = cached_record.get("pair")
        if isinstance(stored_pair, Mapping) and stored_pair:
            # Only reachable with reject-mining capture on. Loud, because a
            # silent pass here poisons the SFT corpus with rejected text.
            logger.warning(
                "Refusing to replay a REJECTED checkpoint pair as accepted "
                "for %s — regenerating",
                cache_key,
            )
        _invalidate("rejected_disposition")
        return {}
    if not _checkpoint_accepted_matches_contract(
        cached_record, current_contract
    ):
        logger.warning(
            "Synthesis checkpoint accepted contract "
            "mismatch for %s — regenerating",
            cache_key,
        )
        _invalidate("contract_mismatch")
        cached_pair: Dict[str, Any] = {}
    else:
        cached_pair = cached_record.get("pair") or {}
    if not _checkpoint_pair_matches_focus(cached_pair, focused_chunk):
        logger.warning(
            "Synthesis checkpoint objective focus mismatch "
            "for %s: cached_lo_refs=%r current_lo_refs=%r "
            "— regenerating",
            cache_key,
            cached_pair.get("lo_refs"),
            focused_chunk.get("learning_outcome_refs"),
        )
        _invalidate("focus_mismatch")
        cached_pair = {}
    return cached_pair


def _replay_terminal_rejection_stats(
    stats: SynthesisStats,
    *,
    kind: str,
    reason: str,
) -> None:
    """Reconstruct the counters a cached terminal rejection originally hit."""
    if kind == "instruction":
        stats.instruction_pairs_rejected += 1
    else:
        stats.preference_pairs_rejected += 1
    stats.rejected_reasons[f"{kind}:{reason}"] = (
        stats.rejected_reasons.get(f"{kind}:{reason}", 0) + 1
    )

    stage, separator, leaf_reason = reason.partition(":")
    if not separator or stage not in {
        "promotion",
        "claim_support",
        "lo_refs",
        "objective_delivery",
    }:
        return
    stats.candidate_pairs_total += 1
    stats.dropped_count += 1
    stats.rejected_promotion_pairs += 1
    stats.promotion_rejection_reasons[leaf_reason] = (
        stats.promotion_rejection_reasons.get(leaf_reason, 0) + 1
    )
    if stage == "claim_support":
        stats.claim_support_rejected += 1
    elif stage == "lo_refs":
        stats.lo_refs_rejected += 1
    elif stage == "objective_delivery":
        stats.objective_delivery_rejected += 1


def _record_ineligible_stats(
    stats: SynthesisStats,
    *,
    kind: str,
    reason: str,
) -> None:
    if kind == "instruction":
        stats.instruction_pairs_ineligible += 1
    else:
        stats.preference_pairs_ineligible += 1
    key = f"{kind}:{reason}"
    stats.ineligible_reasons[key] = stats.ineligible_reasons.get(key, 0) + 1


def _record_ineligible_disposition(
    *,
    stats: SynthesisStats,
    checkpoint_fh: Optional[Any],
    capture: DecisionCapture,
    chunk_id: str,
    kind: str,
    variant_index: int,
    provider: str,
    seed: int,
    reason: str,
    contract_fingerprint: str,
    detail: Optional[str] = None,
) -> None:
    _record_ineligible_stats(stats, kind=kind, reason=reason)
    _checkpoint_terminal_ineligible(
        checkpoint_fh,
        chunk_id=chunk_id,
        kind=kind,
        variant_index=variant_index,
        provider=provider,
        seed=seed,
        reason=reason,
        contract_fingerprint=contract_fingerprint,
    )
    capture.log_decision(
        decision_type=(
            "instruction_pair_synthesis"
            if kind == "instruction"
            else "preference_pair_generation"
        ),
        decision=(
            f"Marked {kind} unit for chunk {chunk_id} ineligible before "
            f"provider dispatch: {reason}."
        ),
        rationale=(
            f"Deterministic per-kind eligibility evaluated chunk_id={chunk_id}, "
            f"kind={kind}, variant_index={variant_index}, seed={seed}, "
            f"provider={provider}, reason={reason}"
            + (f", signals[{detail}]" if detail else "")
            + "; no model request was made "
            "and the unit remains outside the quality-rejection denominator."
        ),
        context=(
            f"chunk_id={chunk_id}; kind={kind}; "
            f"eligibility_reason={reason}"
            + (f"; eligibility_detail={detail}" if detail else "")
        ),
    )


@dataclass(frozen=True)
class _ChunkGenerationBundle:
    """Immutable generation results for one chunk.

    Worker threads build only this value.  They never touch output JSONL,
    checkpoints, aggregate counters, or dedupe sets; the source-order loop in
    :func:`run_synthesis` remains the sole writer for all of that state.
    ``None`` means the corresponding unit has a valid resume-checkpoint
    disposition and therefore required no provider call.
    """

    instruction_results: Tuple[Optional[Any], ...]
    preference_result: Optional[Any]
    instruction_ineligible_reasons: Tuple[Optional[str], ...]
    preference_ineligible_reason: Optional[str]
    provider_results: int
    cached_replays: int
    fingerprint: str
    errors: Tuple["_GenerationError", ...] = ()
    transient_attempts: int = 0
    recovered_units: int = 0
    exhausted_units: int = 0
    fatal_units: int = 0


@dataclass(frozen=True)
class _GenerationError:
    chunk_id: str
    kind: str
    variant_index: int
    attempt: int
    transient: bool
    error_type: str
    message: str


@dataclass(frozen=True)
class _GenerationUnitOutcome:
    result: Optional[Any]
    provider_results: int = 0
    cached_replays: int = 0
    error: Optional[_GenerationError] = None
    transient_attempts: int = 0
    recovered_units: int = 0
    exhausted_units: int = 0
    fatal_units: int = 0


def _pause_for_failed_seat_recovery(
    recovery_coordinator: Any,
    exc: BaseException,
) -> NoReturn:
    """Pause loudly after the one bounded infrastructure window is exhausted."""

    from lib.generation.stop_control import (
        GracefulStopRequested,
        request_stop,
    )

    run_id = getattr(recovery_coordinator, "run_id", None)
    sentinel = (
        request_stop(
            run_id,
            scope="run",
            reason="seat_unhealthy",
            source="synthesis_recovery",
        )
        if run_id
        else None
    )
    raise GracefulStopRequested(
        "training_synthesis.seat_recovery",
        0,
        sentinel,
    ) from exc


def _is_transient_generation_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if not isinstance(exc, SynthesisProviderError):
        return False
    code = str(getattr(exc, "code", "") or "").lower()
    return (
        code in {
            "429",
            "max_retries_exceeded",
            "malformed_response",
        }
        or (code.isdigit() and 500 <= int(code) <= 599)
    )


def _deserialize_generation_result(kind: str, payload: Mapping[str, Any]) -> Any:
    if kind == "instruction":
        from Trainforge.generators.pairs.instruction import (
            InstructionSynthesisResult,
        )
        return InstructionSynthesisResult(**dict(payload))
    from Trainforge.generators.pairs.preference import PreferenceSynthesisResult
    return PreferenceSynthesisResult(**dict(payload))


def _run_generation_unit(
    *,
    chunk_id: str,
    kind: str,
    variant_index: int,
    fingerprint: str,
    generation_cache: Dict[Tuple[str, str, int], Dict[str, Any]],
    journal: GenerationJournal,
    call: Callable[[], Any],
    recovery_coordinator: Optional[Any] = None,
) -> _GenerationUnitOutcome:
    """Replay or execute one provider unit and durably record its outcome."""

    key = (chunk_id, kind, variant_index)
    prior = generation_cache.get(key)
    prior_attempt = int((prior or {}).get("attempt", 0) or 0)
    if (
        prior
        and prior.get("fingerprint") == fingerprint
        and prior.get("disposition") == "success"
        and isinstance(prior.get("result"), dict)
    ):
        return _GenerationUnitOutcome(
            result=_deserialize_generation_result(kind, prior["result"]),
            cached_replays=1,
        )
    attempt = (
        prior_attempt + 1
        if prior and prior.get("fingerprint") == fingerprint
        else 1
    )
    if (
        prior
        and prior.get("fingerprint") == fingerprint
        and prior.get("disposition") == "fatal"
    ):
        error = _GenerationError(
            chunk_id, kind, variant_index, attempt, False,
            str(prior.get("error_type") or "FatalGenerationError"),
            str(prior.get("message") or "fatal generation error"),
        )
        return _GenerationUnitOutcome(result=None, error=error)
    try:
        result = call()
    except Exception as exc:
        recovery_eligible = False
        if recovery_coordinator is not None:
            from Trainforge.synthesis.seat_recovery import (
                eligible_engine_transport_failure,
            )
            recovery_eligible = eligible_engine_transport_failure(exc)
        if recovery_eligible and recovery_coordinator.recover(
            exc,
            incident_context={
                "workflow_phase": "training_synthesis",
                "task_id": f"{chunk_id}:{kind}:{variant_index}",
                "chunk_id": chunk_id,
                "kind": kind,
                "variant_index": variant_index,
                "fingerprint": fingerprint,
            },
        ):
            try:
                result = call()
            except Exception as retry_exc:
                from Trainforge.synthesis.seat_recovery import (
                    eligible_engine_transport_failure,
                )
                if eligible_engine_transport_failure(retry_exc):
                    _pause_for_failed_seat_recovery(
                        recovery_coordinator,
                        retry_exc,
                    )
                exc = retry_exc
            else:
                journal.append({
                    "chunk_id": chunk_id,
                    "kind": kind,
                    "variant_index": variant_index,
                    "fingerprint": fingerprint,
                    "attempt": attempt,
                    "disposition": "success",
                    "result": asdict(result),
                    "recovered_engine_incident": (
                        recovery_coordinator.recovery_id
                    ),
                })
                return _GenerationUnitOutcome(
                    result=result,
                    provider_results=1,
                    recovered_units=1,
                )
        elif recovery_eligible:
            # A confirmed dead seat that could not be recovered is
            # infrastructure-unavailable, not bad content. Pause the run at
            # this durable unit boundary so retries cannot burn the semantic
            # three-attempt terminal lineage against the same dead engine.
            _pause_for_failed_seat_recovery(recovery_coordinator, exc)
        was_transient = _is_transient_generation_error(exc)
        transient = was_transient
        if transient and attempt >= MAX_TRANSIENT_RESUME_ATTEMPTS:
            transient = False
        disposition = "transient" if transient else "fatal"
        row = {
            "chunk_id": chunk_id,
            "kind": kind,
            "variant_index": variant_index,
            "fingerprint": fingerprint,
            "attempt": attempt,
            "disposition": disposition,
            "was_transient": was_transient,
            "error_type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
        error_code = str(getattr(exc, "code", "") or "").strip()
        if error_code:
            row["error_code"] = error_code
        if error_code in {
            "output_truncated",
            "staged_output_truncated",
            "staged_context_window_exceeded",
            "field_clamp_truncation",
        }:
            row["truncation_kind"] = error_code
        journal.append(row)
        error = _GenerationError(
            chunk_id, kind, variant_index, attempt, transient,
            type(exc).__name__, str(exc)
        )
        return _GenerationUnitOutcome(
            result=None,
            provider_results=1,
            error=error,
            transient_attempts=1,
            exhausted_units=int(
                was_transient
                and not transient
                and attempt >= MAX_TRANSIENT_RESUME_ATTEMPTS
            ),
            fatal_units=int(not transient),
        )
    row = {
        "chunk_id": chunk_id,
        "kind": kind,
        "variant_index": variant_index,
        "fingerprint": fingerprint,
        "attempt": attempt,
        "disposition": "success",
        "result": asdict(result),
    }
    journal.append(row)
    return _GenerationUnitOutcome(
        result=result,
        provider_results=1,
        recovered_units=int(
            bool(prior)
            and prior.get("fingerprint") == fingerprint
            and prior.get("disposition") == "transient"
        ),
    )


def _call_with_seat_recovery(
    call: Callable[[], Any],
    *,
    recovery_coordinator: Optional[Any],
    incident_context: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Run one sequential provider call with infrastructure-only recovery.

    The healthy and non-engine-failure paths are exactly one direct call.
    Exhausted timeout/5xx failures may consume the coordinator's single cold
    recovery window, then retry the same semantic unit once. Recovery failure
    pauses the run; it is never converted into content-attempt lineage.
    """

    try:
        return call()
    except Exception as exc:
        if recovery_coordinator is None:
            raise
        from Trainforge.synthesis.seat_recovery import eligible_engine_transport_failure

        if not eligible_engine_transport_failure(exc):
            raise
        if recovery_coordinator.recover(
            exc,
            incident_context=dict(incident_context or {}),
        ):
            try:
                return call()
            except Exception as retry_exc:
                if eligible_engine_transport_failure(retry_exc):
                    _pause_for_failed_seat_recovery(
                        recovery_coordinator,
                        retry_exc,
                    )
                raise
        _pause_for_failed_seat_recovery(recovery_coordinator, exc)


def _checkpoint_skips_generation(
    checkpoint_cache: Dict[Tuple[str, str, int], Dict[str, Any]],
    *,
    chunk_id: str,
    kind: str,
    variant_index: int,
    provider: str,
    seed: int,
    focused_chunk: Dict[str, Any],
    fingerprint_for_seed: Callable[[int, Mapping[str, Any]], str],
) -> bool:
    """Return whether a checkpoint disposition avoids a provider call.

    This deliberately mirrors the source-order writer's stricter replay
    checks.  Cross-pair prompt dedupe remains writer-owned; it can suppress an
    accepted cached pair without causing a fresh provider call.
    """

    key = (chunk_id, kind, variant_index)
    record = checkpoint_cache.get(key)
    if not record or record.get("provider") != provider:
        return False
    if record.get("disposition") == "rejected":
        return _checkpoint_rejection_matches_contract(
            record,
            fingerprint_for_seed(seed, focused_chunk),
        )
    if record.get("disposition") == "ineligible":
        return _checkpoint_ineligible_matches_contract(
            record,
            fingerprint_for_seed(seed, focused_chunk),
        )
    if not _checkpoint_accepted_matches_contract(
        record, fingerprint_for_seed(seed, focused_chunk)
    ):
        return False
    pair = record.get("pair")
    return bool(
        isinstance(pair, dict)
        and pair
        and _checkpoint_pair_matches_focus(pair, focused_chunk)
    )


def _build_chunk_generation_bundle(
    item: Tuple[int, Dict[str, Any]],
    *,
    seed: int,
    provider: str,
    instruction_variants: int,
    paraphrase_provider: Optional[Any],
    pilot_manifest: Optional[Any],
    checkpoint_cache: Dict[Tuple[str, str, int], Dict[str, Any]],
    generation_cache: Dict[Tuple[str, str, int], Dict[str, Any]],
    generation_journal: GenerationJournal,
    recovery_coordinator: Optional[Any],
    fingerprint_for_seed: Callable[[int, Mapping[str, Any]], str],
    objectives: Mapping[str, Mapping[str, Any]],
    capture: DecisionCapture,
) -> _ChunkGenerationBundle:
    """Generate one chunk's provider-backed candidates without side effects.

    Provider HTTP calls and their mandatory per-call DecisionCapture events may
    run concurrently.  Artifact mutation and final per-pair capture events are
    intentionally absent and remain in the ordered caller.
    """

    idx, chunk = item
    chunk_text = str(chunk.get("text") or "")
    preserve_tokens = (
        pilot_manifest.detect_surface_forms(chunk_text)
        if pilot_manifest is not None
        else []
    )
    extra_anchor_tokens = sorted(
        extract_curies(chunk_text) - set(preserve_tokens)
    )[:6]
    effective_preserve_tokens = preserve_tokens + extra_anchor_tokens
    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")

    focused_units: List[Tuple[int, Dict[str, Any]]] = []
    for variant_index in range(instruction_variants):
        unit_seed = seed + idx + (variant_index * 100_000)
        focused_units.append(
            (
                unit_seed,
                _focus_chunk_on_objective(
                    chunk, seed=unit_seed, objectives=objectives,
                ),
            )
        )
    preference_seed = seed + idx
    preference_focused = _focus_chunk_on_objective(
        chunk, seed=preference_seed, objectives=objectives,
    )
    instruction_results: List[Optional[Any]] = []
    instruction_ineligible_reasons: List[Optional[str]] = []
    errors: List[_GenerationError] = []
    provider_results = 0
    cached_replays = 0
    transient_attempts = 0
    recovered_units = 0
    exhausted_units = 0
    fatal_units = 0
    generation_iterator: Optional[Any] = None
    fingerprints: List[str] = []
    for variant_index, (pair_seed, focused) in enumerate(focused_units):
        unit_fingerprint = fingerprint_for_seed(pair_seed, focused)
        fingerprints.append(unit_fingerprint)
        eligibility = _pair_eligibility_for_mode(
            focused, kind="instruction",
        )
        if not eligibility.eligible:
            instruction_results.append(None)
            instruction_ineligible_reasons.append(eligibility.reason)
            continue
        instruction_ineligible_reasons.append(None)
        if _checkpoint_skips_generation(
            checkpoint_cache,
            chunk_id=chunk_id,
            kind="instruction",
            variant_index=variant_index,
            provider=provider,
            seed=pair_seed,
            focused_chunk=focused,
            fingerprint_for_seed=fingerprint_for_seed,
        ):
            instruction_results.append(None)
            continue
        with _micro_generation_unit("instruction", variant_index):
            outcome = _run_generation_unit(
                chunk_id=chunk_id,
                kind="instruction",
                variant_index=variant_index,
                fingerprint=unit_fingerprint,
                generation_cache=generation_cache,
                journal=generation_journal,
                recovery_coordinator=recovery_coordinator,
                call=lambda focused=focused, pair_seed=pair_seed: synthesize_instruction_pair(
                    focused,
                    seed=pair_seed,
                    provider=provider,
                    paraphrase_provider=paraphrase_provider,
                    preserve_tokens=effective_preserve_tokens or None,
                    capture=capture,
                ),
            )
        instruction_results.append(outcome.result)
        provider_results += outcome.provider_results
        cached_replays += outcome.cached_replays
        transient_attempts += outcome.transient_attempts
        recovered_units += outcome.recovered_units
        exhausted_units += outcome.exhausted_units
        fatal_units += outcome.fatal_units
        if outcome.error is not None:
            errors.append(outcome.error)

    focused = preference_focused
    preference_fingerprint = fingerprint_for_seed(preference_seed, focused)
    fingerprints.append(preference_fingerprint)
    preference_eligibility = _pair_eligibility_for_mode(
        focused, kind="preference",
    )
    preference_ineligible_reason = (
        None
        if preference_eligibility.eligible
        else preference_eligibility.reason
    )
    if preference_ineligible_reason is not None:
        preference_result = None
    elif _checkpoint_skips_generation(
        checkpoint_cache,
        chunk_id=chunk_id,
        kind="preference",
        variant_index=0,
        provider=provider,
        seed=preference_seed,
        focused_chunk=focused,
        fingerprint_for_seed=fingerprint_for_seed,
    ):
        preference_result = None
    else:
        with _micro_generation_unit("preference", 0):
            outcome = _run_generation_unit(
                chunk_id=chunk_id,
                kind="preference",
                variant_index=0,
                fingerprint=preference_fingerprint,
                generation_cache=generation_cache,
                journal=generation_journal,
                recovery_coordinator=recovery_coordinator,
                call=lambda: synthesize_preference_pair(
                    focused,
                    seed=preference_seed,
                    provider=provider,
                    paraphrase_provider=paraphrase_provider,
                    preserve_tokens=effective_preserve_tokens or None,
                    capture=capture,
                ),
            )
        preference_result = outcome.result
        provider_results += outcome.provider_results
        cached_replays += outcome.cached_replays
        transient_attempts += outcome.transient_attempts
        recovered_units += outcome.recovered_units
        exhausted_units += outcome.exhausted_units
        fatal_units += outcome.fatal_units
        if outcome.error is not None:
            errors.append(outcome.error)

    return _ChunkGenerationBundle(
        instruction_results=tuple(instruction_results),
        preference_result=preference_result,
        instruction_ineligible_reasons=tuple(
            instruction_ineligible_reasons
        ),
        preference_ineligible_reason=preference_ineligible_reason,
        provider_results=provider_results,
        cached_replays=cached_replays,
        fingerprint=hashlib.sha256(
            "|".join(fingerprints).encode("utf-8")
        ).hexdigest(),
        errors=tuple(errors),
        transient_attempts=transient_attempts,
        recovered_units=recovered_units,
        exhausted_units=exhausted_units,
        fatal_units=fatal_units,
    )


def run_synthesis(
    corpus_dir: Path,
    course_code: str,
    provider: str = "mock",
    seed: int = DEFAULT_SEED,
    capture: Optional[DecisionCapture] = None,
    *,
    stratify: Optional[Sequence[str]] = None,
    include_dpo_from_misconceptions: bool = False,
    difficulty_curriculum: bool = False,
    max_pairs: Optional[int] = None,
    output_dir: Optional[Path] = None,
    curriculum_from_graph: bool = False,
    prereq_windowed: bool = False,
    prereq_context_tokens: int = DEFAULT_PREREQ_CONTEXT_TOKENS,
    pedagogy_graph_path: Optional[Path] = None,
    slug: Optional[str] = None,
    instruction_variants_per_chunk: int = 1,
    dispatcher: Optional[Any] = None,
    cache_path: Optional[Path] = None,
    synthesis_pairs_checkpoint_path: Optional[Path] = None,
    max_dispatches: Optional[int] = None,
    telemetry_path: Optional[Path] = None,
    pilot_report_every: int = 20,
    smoke_mode: str = "none",
    with_kg_metadata: bool = False,
    kg_metadata_max_pairs: int = 2000,
    with_violation_detection: bool = False,
    violation_shapes_glob: Optional[str] = None,
    violation_detection_max_pairs: Optional[int] = None,
    with_abstention: bool = False,
    abstention_max_pairs: int = 1000,
    with_schema_translation: bool = False,
    schema_translation_max_pairs: int = 50,
    with_assessment_sft: bool = False,
    assessment_sft_max_pairs: Optional[int] = None,
    with_graph_sft: bool = False,
    graph_sft_max_pairs: Optional[int] = None,
    holdout_split_path: Optional[Path] = None,
    staging_dir: Optional[Path] = None,
    max_concurrent: Optional[int] = None,
    expected_fresh_start_id: Optional[str] = None,
    objectives_path: Optional[Path] = None,
) -> SynthesisStats:
    """Run the full synthesis stage for one course output directory.

    Args:
        corpus_dir: The course output directory (NOT the inner ``corpus/``).
            This is the dir that contains ``corpus/chunks.jsonl`` and
            ``training_specs/``.
        course_code: Course code used for decision capture.
        provider: Synthesis provider; ``"mock"`` (default) is the only one wired.
        seed: Base seed. Each chunk's effective seed is ``seed + chunk_index``.
        capture: Optional pre-built DecisionCapture. If None, one is created
            for the ``synthesize-training`` phase and saved at end of run.

    Keyword-only options (all default-off so existing callers --
    process_course.py, the MCP synthesize_training tool, the
    textbook_to_course pipeline phase -- keep their behaviour):

        stratify: List of dimensions in ``{"bloom","chunk_type","outcome",
            "difficulty"}``. When set, eligible chunks are sampled
            round-robin across the resulting buckets so the output pair
            distribution is uniform across that dimension.
        include_dpo_from_misconceptions: When True, every editorial
            ``chunk.misconceptions`` entry produces an additional DPO pair
            with ``chosen=correction`` and ``rejected=misconception``. These
            are appended to the standard preference_pairs.jsonl output and
            tagged with ``source="misconception_editorial"`` so downstream
            consumers can filter.
        difficulty_curriculum: When True, the emitted pairs are ordered
            foundational -> intermediate -> advanced (preserved in the
            output JSONL). Default ordering remains by ``chunk_id``.
        max_pairs: Hard cap on total emitted pairs (instruction +
            preference combined). The cap is applied to each artifact
            independently so neither file exceeds the cap on its own. None
            (default) means uncapped.
        output_dir: Optional override for the directory that receives
            ``instruction_pairs.jsonl`` and ``preference_pairs.jsonl``.
            Defaults to ``corpus_dir/training_specs``. The
            ``dataset_config.json`` is always written next to the JSONL
            outputs.
        max_concurrent: Maximum immutable chunk-generation workers. ``None``
            resolves ``TRAINFORGE_SYNTHESIS_MAX_CONCURRENT``; unset defaults
            to the byte-identical sequential path (1). Values 2-48 use a
            bounded pool while retaining a single source-order writer.
        objectives_path: Optional authoritative objectives document. ``None``
            resolves ``TRAINFORGE_SYNTHESIS_OBJECTIVES_PATH`` and otherwise
            retains the historical corpus-relative objective lookup.

    Returns:
        :class:`SynthesisStats` with counts.
    """
    # TRAINFORGE_SYNTHESIS_PROVIDER overrides the
    # caller-supplied ``provider`` kwarg when set. Mirrors the
    # ``TRAINFORGE_ASSESSMENT_PROVIDER`` / ``COURSEFORGE_PROVIDER`` /
    # ``COURSEPLANNER_PROVIDER`` env-var precedence pattern. Pairs with
    # the ``MCP/core/executor.py`` short-circuit that bypasses the
    # ``training-synthesizer`` subagent dispatch when this env var is
    # set: the executor falls through to the in-process registry path
    # (``MCP/tools/pipeline_tools.py::_synthesize_training``), which
    # invokes this function — and we honor the same env var here so an
    # operator can pin the synthesis backend (e.g. ``local`` for
    # license-clean Qwen, ``together`` for hosted OSS) once via the
    # environment without threading the provider through every caller.
    # Empty / whitespace values are treated as unset to match the
    # executor's predicate. See ``docs/LICENSING.md`` § "Synthesis
    # providers" for the licensing contract that motivates the override.
    _env_provider = os.environ.get("TRAINFORGE_SYNTHESIS_PROVIDER", "").strip()
    if _env_provider:
        provider = _env_provider
    resolved_max_concurrent = resolve_synthesis_max_concurrent(max_concurrent)

    # the Anthropic-SDK training-pair synthesis path was REMOVED.
    # ``provider="anthropic"`` is UNCONDITIONALLY forbidden here: there is no
    # acknowledgment escape, because the ``AnthropicSynthesisProvider`` class +
    # its SDK transport no longer exist, so the SLM training corpus can never
    # be routed through the Anthropic SDK (license-clean by construction).
    # Fails closed BEFORE any provider construction. See docs/LICENSING.md
    # § "Synthesis providers".
    if provider in _REMOVED_SYNTHESIS_PROVIDERS:
        raise SynthesisLicensingError(
            f"Training-pair synthesis provider {provider!r} was REMOVED: the "
            f"Anthropic-SDK training path (AnthropicSynthesisProvider) no longer "
            f"exists, so the SLM training corpus can never be routed through it "
            f"— there is no acknowledgment-flag escape. The documented "
            f"license-clean default is --provider local (Apache-2.0 Qwen) or "
            f"--provider together (hosted OSS). See docs/LICENSING.md "
            f"§ \"Synthesis providers\"."
        )

    # Marketable-v1 D4 — license-clean-by-default gate for TRAINING-PAIR
    # synthesis. The emitted instruction / preference pairs ARE the canonical
    # SLM training corpus (the trained adapter is a derivative work of them),
    # so per ``docs/LICENSING.md`` § "Synthesis providers" the ``claude_session``
    # route (a SEPARATE Claude-Code-session path, Anthropic Consumer Terms) is
    # NOT a clean default here. Selecting it for THIS surface (the kwarg default
    # is "mock" and the documented clean path is local / together) is therefore
    # an explicit, fail-loud opt-in: the operator must set
    # ``TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true`` to acknowledge they have a
    # separate written agreement with Anthropic permitting derivative training.
    # Without it we fail closed rather than silently producing a ToS-unclean
    # corpus. This is the synthesis-side companion to the executor's subagent
    # short-circuit and the A5 CLI ``TRAINFORGE_SYNTHESIS_PROVIDER``
    # default-to-local.
    if provider in _ANTHROPIC_FAMILY_SYNTHESIS_PROVIDERS:
        _ack = os.environ.get("TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS", "").strip().lower()
        if _ack not in ("1", "true", "yes", "on"):
            raise SynthesisLicensingError(
                f"Training-pair synthesis provider {provider!r} routes the SLM "
                f"training corpus through a Claude Code session whose ToS "
                f"(Anthropic Consumer Terms, Pro/Max) restricts using outputs "
                f"to train a derivative model. The documented license-clean "
                f"default is --provider local (Apache-2.0 Qwen) or --provider "
                f"together (hosted OSS). To proceed with {provider!r} anyway — "
                f"only valid if you hold a separate written agreement with "
                f"Anthropic permitting derivative training — set "
                f"TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true to acknowledge the "
                f"licensing posture. See docs/LICENSING.md "
                f"§ \"Synthesis providers\"."
            )
        logger.warning(
            "TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS acknowledged: training-pair "
            "synthesis is routing through Anthropic-family provider %r. The "
            "resulting corpus is NOT license-clean for derivative training "
            "absent a separate Anthropic agreement (docs/LICENSING.md).",
            provider,
        )

    # Marketable-v1 D4 (defense-in-depth) — unconditional fail-closed gate for
    # hosted-cloud providers whose ToS restricts training-data use with NO
    # documented escape (NVIDIA-hosted Llama-3.3). Unlike the Anthropic-family
    # gate above, there is no ack-flag: the hosted Llama-3.3 corpus is never
    # shippable as SLM training data, so this raises unconditionally before any
    # LLM dispatch. This covers a DIRECT TRAINFORGE_SYNTHESIS_PROVIDER=nvidia
    # selection (a bare run_synthesis / MCP-tool / CLI call) that bypasses the
    # workflow-runner's license-clean training-seat default.
    if provider in _RESTRICTED_NO_ACK_SYNTHESIS_PROVIDERS:
        raise SynthesisLicensingError(
            f"Training-pair synthesis provider {provider!r} routes the SLM "
            f"training corpus through a hosted-cloud backend whose ToS + "
            f"underlying model license (NVIDIA-hosted Llama-3.3) restrict using "
            f"outputs to train a derivative model. Unlike the Anthropic family, "
            f"there is NO acknowledgment-flag escape — the hosted Llama-3.3 "
            f"corpus is unconditionally restricted for the training-pair "
            f"surface. The documented license-clean default is --provider local "
            f"(Apache-2.0 Qwen) or --provider together (hosted OSS). See "
            f"docs/LICENSING.md § \"Synthesis providers\"."
        )

    if provider == "claude_session" and dispatcher is None:
        raise RuntimeError(
            "--provider claude_session requires a LocalDispatcher to be "
            "supplied. Invoke via the workflow runner ('ed4all run "
            "trainforge_train ...') or the synthesize_training MCP tool, "
            "both of which inject a dispatcher. Standalone CLI invocation "
            "has no Claude Code session to dispatch to."
        )

    # The staged production path must prove model identity before creating the
    # output directory, inspecting resume state, opening a cache/sidecar, or
    # constructing a provider.  This prevents a registry default (notably the
    # Nano default) from silently serving a run explicitly intended for a
    # different immutable snapshot.
    from Trainforge.generators.staged.provider import (
        staged_synthesis_v4_enabled,
    )
    preflight_model_id: Optional[str] = None
    if provider == "local" and staged_synthesis_v4_enabled():
        preflight_model_id = _preflight_local_staged_model_identity(
            base_url=os.environ.get(
                "LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8000/v1"
            ),
            local_model=os.environ.get("LOCAL_SYNTHESIS_MODEL", ""),
            generic_model=os.environ.get("TRAINFORGE_SYNTHESIS_MODEL", ""),
        )

    corpus_dir = Path(corpus_dir)
    _env_objectives_path = os.environ.get(
        "TRAINFORGE_SYNTHESIS_OBJECTIVES_PATH", ""
    ).strip()
    configured_objectives_path = (
        Path(objectives_path)
        if objectives_path is not None
        else (Path(_env_objectives_path) if _env_objectives_path else None)
    )
    if (
        configured_objectives_path is not None
        and not configured_objectives_path.is_file()
    ):
        raise FileNotFoundError(
            "Configured synthesis objectives file does not exist or is not "
            f"a regular file: {configured_objectives_path}"
        )
    resolved_objectives_path = (
        configured_objectives_path
        if configured_objectives_path is not None
        else corpus_dir / "objectives.json"
    )
    # prefer imscc_chunks/, fall back to legacy corpus/ via shim.
    from lib.libv2_storage import resolve_imscc_chunks_path
    chunks_path = resolve_imscc_chunks_path(corpus_dir, "chunks.jsonl")
    # Only the explicit holdout path moves corpus parsing ahead of output
    # creation.  With the flag off, preserve the historical ordering exactly:
    # output paths/sidecars/config preflight happen first and _read_chunks runs
    # at its original point below.  This also avoids importing/reading any new
    # contract or objective file on the legacy path.
    holdout_enabled = str(
        os.environ.get("TRAINFORGE_SYNTHESIS_HOLDOUT_EXCLUSION", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    preflight_chunks: Optional[List[Dict[str, Any]]] = None
    holdout_registry: Optional[Any] = None
    holdout_identity: Optional[Mapping[str, str]] = None
    if holdout_enabled:
        from Trainforge.synthesis.synthesis_holdout import load_synthesis_holdout_registry

        preflight_chunks = _read_chunks(chunks_path)
        holdout_registry = load_synthesis_holdout_registry(
            chunks_path=chunks_path,
            objectives_path=resolved_objectives_path,
            chunks=preflight_chunks,
        )
        if holdout_registry is None:  # pragma: no cover - guarded above
            raise RuntimeError("holdout enforcement did not produce a registry")
        holdout_identity = holdout_registry.identity
    if output_dir is not None:
        training_specs_dir = Path(output_dir)
    else:
        training_specs_dir = corpus_dir / "training_specs"
    training_specs_dir.mkdir(parents=True, exist_ok=True)

    # A workflow that requests a fresh synthesis attempt must prove that the
    # scoped reset completed before this process even inspects stale sidecars,
    # caches, journals, or checkpoints.  The identity is explicit so legacy
    # standalone/pilot callers retain their established resume behavior.
    requested_fresh_start_id = (
        expected_fresh_start_id
        or os.environ.get("TRAINFORGE_SYNTHESIS_FRESH_START_ID", "").strip()
        or None
    )
    fresh_start_marker: Optional[Mapping[str, Any]] = None
    if requested_fresh_start_id is not None:
        from Trainforge.synthesis.synthesis_fresh_start import require_fresh_start_marker

        fresh_start_marker = require_fresh_start_marker(
            training_specs_dir,
            expected_fresh_start_id=requested_fresh_start_id,
            expected_holdout_identity=holdout_identity,
            allow_resume_artifacts=True,
        )
    fresh_start_marker_digest = (
        str(fresh_start_marker.get("archive_manifest_sha256") or "")
        if fresh_start_marker is not None
        else None
    )
    if requested_fresh_start_id is not None and not fresh_start_marker_digest:
        raise RuntimeError("fresh synthesis marker digest is required")

    # Validate the measured served window once at orchestration time.  The
    # provider retains its per-call headroom check, but a missing/garbage
    # declaration must fail before any synthesis output or resume file opens.
    from Trainforge.generators.staged.provider import (
        ENV_SERVED_CONTEXT_TOKENS,
    )
    production_embedder: Optional[Any] = None
    if provider == "local" and staged_synthesis_v4_enabled():
        raw_served_window = os.environ.get(ENV_SERVED_CONTEXT_TOKENS, "").strip()
        try:
            served_context_tokens = int(raw_served_window)
        except (TypeError, ValueError):
            served_context_tokens = 0
        if served_context_tokens <= 0:
            raise RuntimeError(
                f"{ENV_SERVED_CONTEXT_TOKENS} must be a positive measured "
                "served model window before staged synthesis starts"
            )
        if os.environ.get("TRAINFORGE_REQUIRE_EMBEDDINGS", "").strip().lower() not in {
            "1", "true", "yes", "on",
        }:
            raise RuntimeError(
                "TRAINFORGE_REQUIRE_EMBEDDINGS=true is required for staged "
                "production synthesis; Jaccard fallback is not permitted"
            )
        from lib.embedding.sentence_embedder import try_load_embedder

        production_embedder = try_load_embedder()
        if production_embedder is None:
            raise RuntimeError(
                "configured synthesis embedder could not be loaded; Jaccard "
                "fallback is not permitted"
            )
        probe = production_embedder.encode(
            "Ed4All staged synthesis embedding readiness probe."
        )
        if probe is None or len(probe) == 0:
            raise RuntimeError(
                "configured synthesis embedder returned an empty readiness "
                "probe; staged synthesis cannot start"
            )

    # smoke modes route JSONL outputs to ``smoke_*`` siblings
    # so a smoke run never clobbers the canonical instruction_pairs.jsonl
    # / preference_pairs.jsonl from a prior full run. dataset_config.json
    # is left at the canonical path because the smoke run's stats are
    # still useful telemetry.
    if smoke_mode in ("deterministic", "paraphrase"):
        instruction_out = training_specs_dir / "smoke_instruction_pairs.jsonl"
        preference_out = training_specs_dir / "smoke_preference_pairs.jsonl"
    else:
        instruction_out = training_specs_dir / "instruction_pairs.jsonl"
        preference_out = training_specs_dir / "preference_pairs.jsonl"
    dataset_config_path = training_specs_dir / "dataset_config.json"

    # incremental sidecar write. Each emitted instruction /
    # preference pair is appended to a ``.jsonl.in_progress`` sibling
    # file with ``flush()`` after every write so an operator can
    # ``tail -f`` the synthesis run and so a killed run leaves
    # inspectable artifacts on disk. The atomic final ``_write_jsonl``
    # is unchanged; sidecars are unlinked on a clean exit and preserved
    # on a ``SynthesisBudgetExceeded`` early-exit (or any other
    # exception that propagates out) for postmortem.
    instruction_progress = instruction_out.with_suffix(".jsonl.in_progress")
    preference_progress = preference_out.with_suffix(".jsonl.in_progress")
    # Handles remain unopened until the complete run contract is validated.
    # A mismatched resume must preserve existing sidecar bytes exactly.
    # Deterministic pre-provider producers may append before the contract
    # preflight completes. Buffer those bytes in memory so a mismatch leaves
    # every existing sidecar byte-identical.
    inst_progress_fh: Optional[Any] = io.StringIO()
    pref_progress_fh: Optional[Any] = io.StringIO()

    # per-pair resume checkpoint sidecar. Hidden dotfile so
    # operators looking at ``training_specs/`` see the canonical artifacts
    # + the ``.in_progress`` observability sidecars without visual
    # noise from machinery they don't need to inspect by hand. Default
    # to checkpoint-on whenever the call site has a write target on disk
    # (i.e. always under the canonical ``run_synthesis`` invocation);
    # the CLI ``--no-checkpoint`` flag (see :func:`main`) routes the
    # disable case via a ``Path`` sentinel that ends in
    # ``.no_checkpoint`` — :func:`main` is the only caller that sets
    # the sentinel; programmatic callers wanting opt-out can pass any
    # ``Path`` whose suffix matches.
    #
    # NOTE: the existing ``cache_path`` kwarg is the CLAUDE-SESSION
    # content-addressed paraphrase cache, not this resume cache. They
    # are deliberately separate paths and serve different purposes;
    # ``cache_path`` is wired only when ``provider == "claude_session"``,
    # whereas this checkpoint covers every provider that writes to disk.
    _CHECKPOINT_DISABLE_SENTINEL = "<disable-synthesis-checkpoint>"
    if synthesis_pairs_checkpoint_path is None:
        checkpoint_path: Optional[Path] = (
            training_specs_dir / ".synthesis_pairs_checkpoint.jsonl"
        )
    elif (
        isinstance(synthesis_pairs_checkpoint_path, Path)
        and synthesis_pairs_checkpoint_path.name == _CHECKPOINT_DISABLE_SENTINEL
    ):
        checkpoint_path = None
    else:
        checkpoint_path = Path(synthesis_pairs_checkpoint_path)
    pair_checkpoint_cache: Dict[
        Tuple[str, str, int], Dict[str, Any]
    ] = {}
    checkpoint_fh: Optional[Any] = (
        io.StringIO() if checkpoint_path is not None else None
    )
    if checkpoint_fh is not None:
        checkpoint_fh._terminal_semantic_keys = set()

    generation_checkpoint_path: Optional[Path] = (
        None
        if checkpoint_path is None or resolved_max_concurrent == 1
        else training_specs_dir / ".synthesis_generation_checkpoint.jsonl"
    )
    generation_checkpoint_cache: Dict[
        Tuple[str, str, int], Dict[str, Any]
    ] = {}
    generation_journal = GenerationJournal(
        generation_checkpoint_path,
        fresh_start_id=requested_fresh_start_id,
        marker_digest=fresh_start_marker_digest,
        holdout_identity=holdout_identity,
    )
    recovery_coordinator: Optional[Any] = None
    if provider == "local":
        from lib.vllm_container_lifecycle import resolve_seat_schedule_mode

        if resolve_seat_schedule_mode():
            from Trainforge.synthesis.seat_recovery import (
                SynthesisSeatRecoveryCoordinator,
            )
            from lib.paths import get_state_runs_dir

            recovery_run_id = (
                os.environ.get("ED4ALL_RUN_ID")
                or os.environ.get("RUN_ID")
            )
            recovery_run_dir = (
                get_state_runs_dir() / str(recovery_run_id)
                if recovery_run_id
                else None
            )
            recovery_coordinator = SynthesisSeatRecoveryCoordinator(
                base_url=os.environ.get(
                    "LOCAL_SYNTHESIS_BASE_URL",
                    "http://localhost:8000/v1",
                ),
                run_dir=recovery_run_dir,
                marker_path=(
                    training_specs_dir / ".synthesis_seat_recovery.jsonl"
                ),
            )

    # load property manifest once. The manifest gates
    # ALL pilot-report writes (in-flight and final). pilot_report_every
    # only governs the in-flight cadence — the final write fires
    # whenever a manifest exists, regardless of pilot_report_every, so
    # an operator who set --pilot-report-every 0 still gets the
    # post-run summary on disk.
    pilot_manifest = None
    # smoke modes write to a sidecar so the canonical
    # pilot_report.md is never overwritten by a partial run.
    if smoke_mode in ("deterministic", "paraphrase"):
        pilot_report_path = training_specs_dir / "smoke_pilot_report.md"
    else:
        pilot_report_path = training_specs_dir / "pilot_report.md"
    pilot_slug = slug or course_code or corpus_dir.name
    if pilot_slug:
        try:
            from lib.ontology.property_manifest import load_property_manifest
            pilot_manifest = load_property_manifest(pilot_slug)
        except FileNotFoundError:
            logger.info(
                "Wave 117: no property manifest for course %r; skipping "
                "pilot_report.md.",
                pilot_slug,
            )
            pilot_manifest = None
    # smoke modes scale every property's ``min_pairs`` floor
    # so a 20-chunk smoke run can pass when the full corpus would.
    # Deterministic = floor 1 (one pair proves preservation through
    # the deterministic path); paraphrase = floor 2 (some chance the
    # provider drops a token on one pair, fallback covers the other).
    if pilot_manifest is not None and smoke_mode in ("deterministic", "paraphrase"):
        from lib.ontology.property_manifest import (
            PropertyEntry as _PE,
            PropertyManifest as _PM,
        )
        smoke_floor = 1 if smoke_mode == "deterministic" else 2
        pilot_manifest = _PM(
            family=pilot_manifest.family,
            properties=[
                _PE(
                    id=p.id, uri=p.uri, curie=p.curie, label=p.label,
                    surface_forms=list(p.surface_forms),
                    min_pairs=smoke_floor,
                    min_accuracy=p.min_accuracy,
                )
                for p in pilot_manifest.properties
            ],
            description=pilot_manifest.description,
        )

    # Validate stratification dimensions early so a typo fails loud rather
    # than silently degrading to no-op.
    stratify_dims: List[str] = []
    if stratify:
        for d in stratify:
            d_clean = str(d).strip().lower()
            if not d_clean:
                continue
            if d_clean not in _STRATIFY_DIMENSIONS:
                raise ValueError(
                    f"Unknown stratification dimension: {d_clean!r}. "
                    f"Valid choices: {sorted(_STRATIFY_DIMENSIONS)}"
                )
            stratify_dims.append(d_clean)

    chunks = (
        preflight_chunks
        if preflight_chunks is not None
        else _read_chunks(chunks_path)
    )
    if holdout_registry is not None:
        # Use copies so the private dispatch sentinel can never leak back into
        # the canonical corpus object or emitted training records.
        chunks = [
            {
                **chunk,
                **(
                    {"_eval_holdout_reserved": True}
                    if holdout_registry.reserves(chunk)
                    else {}
                ),
            }
            for chunk in chunks
        ]
    # smoke modes. "deterministic" forces provider=mock and
    # subsamples to ~20 stratified chunks; "paraphrase" keeps the
    # configured provider but applies the same subsampling. Both write
    # smoke_pilot_report.md as a sidecar so the canonical
    # pilot_report.md is never overwritten by a partial run.
    if smoke_mode == "deterministic":
        provider = "mock"
    if smoke_mode in ("deterministic", "paraphrase"):
        chunks = _smoke_stratified_sample(
            chunks, pilot_manifest, target_count=20, rng=random.Random(seed),
        )
    stats = SynthesisStats(chunks_total=len(chunks))
    stats.stratify_dimensions = list(stratify_dims)
    stats.max_pairs_cap = max_pairs
    stats.difficulty_curriculum = bool(difficulty_curriculum)
    stats.curriculum_from_graph = bool(curriculum_from_graph)
    stats.prereq_windowed = bool(prereq_windowed)
    stats.prereq_context_tokens = int(prereq_context_tokens)
    instruction_variants = max(1, int(instruction_variants_per_chunk))
    stats.instruction_variants_per_chunk = instruction_variants
    if instruction_variants > 1:
        from Trainforge.generators.staged.micro import (
            staged_synthesis_micro_v1_enabled,
        )

        if staged_synthesis_micro_v1_enabled():
            # micro-v1 has NO per-variant entropy. Measured on a 939-chunk
            # corpus: all 449 instruction-eligible chunks produce a
            # BYTE-IDENTICAL focused chunk for variant 0 and variant 1
            # (``_focus_chunk_on_objective``'s seed only breaks ties among
            # equally-ranked objectives, and after focus there is one), and
            # every micro stage keys on the RUN-level ``synthesis_seed``, not
            # on the per-variant ``pair_seed`` — ``paraphrase_instruction``
            # even overwrites the draft's ``seed`` with it. So variant 1 would
            # emit a functionally identical training row while consuming a
            # second full stage ladder of model calls.
            #
            # Refuse rather than silently duplicate. Giving micro-v1 genuine
            # per-variant entropy is a contract change (it would re-key the
            # micro fingerprint), so it is an owner decision, not a default.
            raise ValueError(
                "micro-v1 staged synthesis does not support "
                f"instruction_variants_per_chunk={instruction_variants}: every "
                "micro stage keys on the run-level synthesis seed and the "
                "focused chunk is identical across variants, so variants >1 "
                "emit duplicate rows for a second ladder of model calls. Pass "
                "--instruction-variants-per-chunk 1, or select --synthesis-"
                "contract staged-v4."
            )
    # Reject-mining CAPTURE. Off by default: ``_mine_mode == "off"`` makes
    # ``build_capture_payload`` return None at every rejection site, so the
    # persisted row is byte-identical to the legacy one and the pool stays
    # empty. Resolved once here, never re-read mid-run.
    _mine_mode = resolve_mine_rejects_mode()
    _reject_pool = RejectPool()
    if _mine_mode != MINE_MODE_OFF:
        logger.warning(
            "%s=%s: rejected claim_support pairs will retain their full "
            "completion on the pair checkpoint. This flag participates in the "
            "synthesis run contract, so flipping it on a PAUSED run raises "
            "FreshStartError and archives that run — it is a launch-time "
            "decision only.",
            MINE_REJECTS_ENV,
            _mine_mode,
        )
        if instruction_variants < 2:
            logger.warning(
                "%s is enabled with instruction_variants_per_chunk=%d: mined "
                "yield is STRUCTURALLY ZERO. One instruction unit per chunk "
                "means a chunk can never hold both an accepted and a rejected "
                "instruction unit, so no reject can ever be paired with an "
                "anchor. Raise --instruction-variants to 2+ or expect an "
                "empty pool.",
                MINE_REJECTS_ENV,
                instruction_variants,
            )
    # load the pedagogy graph eagerly when curriculum mode
    # is active so a missing graph fails loud instead of silently degrading
    # to chunk-id ordering. The build itself is cheap (sub-1k concept dict
    # build + Kahn's pass) so doing it once here is fine.
    #
    # graph is now REQUIRED by default. Workflow runs
    # default to ``--curriculum-from-graph=true`` so synthesis never
    # silently produces graph-less ordering. Set ``--no-graph`` (or
    # ``allow_no_graph=True`` programmatically) to opt out for legacy
    # corpora that lack a pedagogy graph.
    curriculum_ctx = None
    chunks_by_id: Dict[str, Dict[str, Any]] = {}
    if curriculum_from_graph or prereq_windowed:
        graph_path = _resolve_pedagogy_graph_path(corpus_dir, pedagogy_graph_path)
        if graph_path is None:
            raise FileNotFoundError(
                "--curriculum-from-graph / --prereq-windowed require "
                f"pedagogy_graph.json under {corpus_dir} (looked in graph/, "
                f"pedagogy/, and the corpus root). Pass --no-graph to "
                f"opt out of the Wave-91 graph-required default."
            )
        graph = load_pedagogy_graph(graph_path)
        curriculum_ctx = build_curriculum_context(graph, chunks)
        chunks_by_id = {
            str(c.get("id") or c.get("chunk_id") or ""): c for c in chunks
        }
        chunks_by_id.pop("", None)

    owns_capture = False
    if capture is None:
        capture = DecisionCapture(
            course_code=course_code,
            phase="synthesize-training",
            tool="trainforge",
            streaming=True,
        )
        owns_capture = True
    if requested_fresh_start_id is not None:
        capture.fresh_start_id = requested_fresh_start_id
        capture.fresh_start_marker_digest = fresh_start_marker_digest
    if holdout_identity is not None:
        capture.synthesis_holdout_identity = holdout_identity

    # construct the claude_session paraphrase provider once per
    # run. The factory layer dispatches to whichever object is passed via
    # paraphrase_provider when provider != "mock". Anthropic stays lazily
    # constructed inside the factory so its API-key precondition only fires
    # when there's actually an eligible chunk to paraphrase.
    paraphrase_provider: Optional[Any] = None
    if provider == "claude_session":
        from Trainforge.generators.providers._claude_session_provider import (
            ClaudeSessionProvider,
        )
        # default telemetry_path under training_specs/.
        # fall back to training_specs_dir even when no
        # explicit cache_path is set, so a session run always leaves a
        # telemetry trail for post-hoc analysis.
        effective_telemetry = telemetry_path
        if effective_telemetry is None:
            base_dir = cache_path.parent if cache_path is not None else training_specs_dir
            effective_telemetry = base_dir / ".synthesis_telemetry.jsonl"
        paraphrase_provider = ClaudeSessionProvider(
            dispatcher=dispatcher,
            run_id=course_code,
            capture=capture,
            cache_path=cache_path,
            max_dispatches=max_dispatches,
            telemetry_path=effective_telemetry,
        )
    elif provider == "together":
        # ToS-clean OSS-teacher paraphrase via Together
        # AI's hosted models. HTTP-driven (no SDK dependency); session-
        # budget tracking is unnecessary because the provider is paid-
        # per-call rather than rate-limited per Claude session.
        #
        # when TRAINFORGE_AGNOSTIC_SYNTHESIS is ON (default),
        # route through the LLM-agnostic SynthesisProvider (verbose hosted
        # prompts via terse_prompts=False) — golden-tested byte-identical
        # to TogetherSynthesisProvider on well-formed responses. The leaf
        # remains the rollback path when the flag is OFF.
        if agnostic_synthesis_enabled():
            from Trainforge.generators.providers._synthesis_provider import (
                build_synthesis_provider,
            )
            paraphrase_provider = build_synthesis_provider(
                "together", capture=capture, synthesis_seed=seed,
            )
        else:
            from Trainforge.generators.providers._together_provider import (
                TogetherSynthesisProvider,
            )
            paraphrase_provider = TogetherSynthesisProvider(capture=capture)
    elif provider == "local":
        # third synthesis path — a local OpenAI-compatible
        # model server (Ollama / vLLM / llama.cpp / LM Studio). Same
        # HTTP wire shape as Together, so the provider subclasses
        # ``TogetherSynthesisProvider`` and only overrides the
        # endpoint / model / auth-required hooks. Zero cost per call
        # after hardware setup, zero ToS exposure (fully offline /
        # air-gapped friendly). Like ``together``, no session-budget
        # tracking — the provider is HTTP-driven, not Claude-session
        # rate-limited.
        #
        # smoke-paraphrase caps parse retries at 1 so the
        # property-heavy stratified sample doesn't compound retry cost
        # × 20 chunks into an unbounded wall time. Production
        # (smoke_mode='none') keeps the default budget.
        local_kwargs: Dict[str, Any] = {"capture": capture}
        if smoke_mode == "paraphrase":
            local_kwargs["max_parse_retries"] = 1
        # when TRAINFORGE_AGNOSTIC_SYNTHESIS is ON (default),
        # route through the LLM-agnostic SynthesisProvider (terse local
        # prompts via terse_prompts=True) — golden-tested byte-identical
        # to LocalSynthesisProvider on well-formed responses. The leaf
        # remains the rollback path when the flag is OFF. The same
        # local_kwargs (capture + optional smoke max_parse_retries) flow
        # into either class — SynthesisProvider accepts the identical
        # knobs with identical defaults.
        if agnostic_synthesis_enabled():
            from Trainforge.generators.providers._synthesis_provider import (
                build_synthesis_provider,
            )
            paraphrase_provider = build_synthesis_provider(
                "local", synthesis_seed=seed, **local_kwargs,
            )
        else:
            from Trainforge.generators.providers._local_provider import (
                LocalSynthesisProvider,
            )
            paraphrase_provider = LocalSynthesisProvider(**local_kwargs)

    rejection_contract_model_id = str(
        getattr(paraphrase_provider, "_model", None)
        or getattr(paraphrase_provider, "model", None)
        or ("deterministic-mock-v1" if provider == "mock" else "unknown")
    )
    if preflight_model_id is not None and rejection_contract_model_id != preflight_model_id:
        raise RuntimeError(
            "staged synthesis provider model identity changed after preflight: "
            f"verified={preflight_model_id!r}, provider={rejection_contract_model_id!r}"
        )

    _generation_param_names = (
        "_temperature",
        "_max_tokens",
        "_max_parse_retries",
        "_reasoning_thinking_off",
        "_top_p",
        "_seed",
        "_template",
        "_template_kwargs",
        "_extra_body",
    )
    rejection_generation_params = {
        name.removeprefix("_"): getattr(paraphrase_provider, name)
        for name in _generation_param_names
        if hasattr(paraphrase_provider, name)
    }
    rejection_endpoint_identity = str(
        getattr(paraphrase_provider, "_base_url", None)
        or getattr(paraphrase_provider, "base_url", None)
        or ""
    )
    static_synthesis_contract_components = (
        _synthesis_static_contract_components(
            provider=provider,
            model_id=rejection_contract_model_id,
            endpoint_identity=rejection_endpoint_identity,
            generation_params=rejection_generation_params,
        )
    )

    def _rejection_fingerprint_for_seed(
        pair_seed: int,
        pair_chunk: Mapping[str, Any],
    ) -> str:
        contract = _synthesis_rejection_contract_fingerprint(
            provider=provider,
            model_id=rejection_contract_model_id,
            pair_seed=pair_seed,
            source_chunk=pair_chunk,
            endpoint_identity=rejection_endpoint_identity,
            generation_params=rejection_generation_params,
            static_components=static_synthesis_contract_components,
        )
        if requested_fresh_start_id is None:
            if holdout_identity is None:
                return contract
            return hashlib.sha256(
                (
                    json.dumps(
                        holdout_identity, sort_keys=True, separators=(",", ":")
                    )
                    + "|"
                    + contract
                ).encode("utf-8")
            ).hexdigest()
        return hashlib.sha256(
            (
                f"{requested_fresh_start_id}|{fresh_start_marker_digest}|"
                f"{json.dumps(holdout_identity or {}, sort_keys=True, separators=(',', ':'))}|"
                f"{contract}"
            ).encode("utf-8")
        ).hexdigest()

    instruction_records: List[Dict[str, Any]] = []
    preference_records: List[Dict[str, Any]] = []

    # Budget-exceeded sentinel — MUST be hoisted above the try-body so the
    # finally-block can reference it even when an exception propagates before
    # the loop assigns it. Imported eagerly so the symbol exists in the
    # finally scope.
    from Trainforge.generators.providers._session_budget import (
        SynthesisBudgetExceeded as _SBE,
    )
    _budget_exhausted_exc: Optional[_SBE] = None

    # gate sidecar deletion on a clean exit. The flag is
    # only set True after the entire try-body completes without any
    # exception (budget-exceeded or otherwise). The finally block
    # checks both this flag AND ``_budget_exhausted_exc is None`` so
    # an exception that propagates past the try-body leaves sidecars
    # in place for postmortem inspection.
    clean_exit = False
    progress_writer: Optional[SynthesisProgressWriter] = None
    generation_iterator: Optional[Any] = None
    prior_run_contract_env = os.environ.get(
        "TRAINFORGE_SYNTHESIS_RUN_CONTRACT_SHA256"
    )
    prior_component_contract_env = os.environ.get(
        "TRAINFORGE_SYNTHESIS_CONTRACT_COMPONENTS_SHA256"
    )
    run_contract_env_set = False
    provider_results_completed = 0
    cached_generation_replays = 0
    transient_attempts = 0
    recovered_units = 0
    exhausted_units = 0
    fatal_units = 0

    # Contract preflight precedes the try-body because optional deterministic
    # generators inside it also emit DecisionCapture events. Every event from
    # this invocation must carry the same sealed identity.
    from lib.validators.pair.objective_delivery import (
        _load_synthesized_objectives_for_w4c,
    )
    _objectives_candidates = (
        (resolved_objectives_path,)
        if configured_objectives_path is not None
        else (
            corpus_dir / "course_planning" / "synthesized_objectives.json",
            corpus_dir / "objectives.json",
            corpus_dir / "synthesized_objectives.json",
            corpus_dir / "01_learning_objectives"
            / "synthesized_objectives.json",
            corpus_dir.parent / "01_learning_objectives"
            / "synthesized_objectives.json",
            corpus_dir.parent.parent / "01_learning_objectives"
            / "synthesized_objectives.json",
        )
    )
    _objectives_path: Optional[Path] = next(
        (path for path in _objectives_candidates if path.exists()), None,
    )
    try:
        _objectives_map = (
            _load_synthesized_objectives_for_w4c(_objectives_path)
            if _objectives_path is not None else {}
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to load canonical synthesis objectives from %s: %s",
            _objectives_path,
            exc,
        )
        _objectives_map = {}
    # A staged contract binds every emitted pair to an objective resolved from
    # the authoritative artifact. Without that artifact the focus seam marks
    # every chunk ``authoritative_objectives_unavailable`` and the run emits
    # ZERO pairs while exiting 0 — a silent whole-corpus failure. Fail here
    # instead, BEFORE any output state or provider is created, and name the
    # artifact rather than defaulting an objective (which would manufacture
    # training data with no real objective binding).
    if staged_objective_contract_enabled() and not _objectives_map:
        raise RuntimeError(
            "staged training synthesis requires canonical objectives, but no "
            "usable objectives artifact was found (searched: "
            + ", ".join(str(path) for path in _objectives_candidates)
            + "). This artifact is produced by the ``course_planning`` phase "
            "(synthesized_objectives.json) and copied to the LibV2 archive as "
            "objectives.json by ``libv2_archival``. Re-run those phases, or "
            "point TRAINFORGE_SYNTHESIS_OBJECTIVES_PATH at the artifact."
        )
    synthesis_run_contract_sha256 = (
        _synthesis_rejection_contract_fingerprint(
            provider=provider,
            model_id=rejection_contract_model_id,
            pair_seed=0,
            source_chunk={
                "objectives": _objectives_map,
                "holdout_identity": dict(holdout_identity or {}),
            },
            endpoint_identity=rejection_endpoint_identity,
            generation_params=rejection_generation_params,
            static_components=static_synthesis_contract_components,
        )
    )
    synthesis_run_contract_components = {
        **static_synthesis_contract_components,
        "objectives_sha256": _normalized_sha256(_objectives_map),
        "holdout_identity_sha256": _normalized_sha256(
            dict(holdout_identity or {})
        ),
        # Recorded AFTER the fingerprint was taken from static components
        # above, so verdict policy is auditable without being able to
        # invalidate an accepted pair. See ``_verdict_policy_digest``.
        "verdict_policy": _verdict_policy_digest(),
    }
    synthesis_contract_components_sha256 = _normalized_sha256(
        synthesis_run_contract_components
    )
    if requested_fresh_start_id is not None:
        from Trainforge.synthesis.synthesis_fresh_start import (
            bind_fresh_start_run_contract,
        )
        bind_fresh_start_run_contract(
            training_specs_dir,
            expected_fresh_start_id=requested_fresh_start_id,
            run_contract_sha256=synthesis_run_contract_sha256,
            resume_artifacts_exist=any(
                path is not None and path.is_file() and path.stat().st_size > 0
                for path in (checkpoint_path, generation_checkpoint_path)
            ),
            contract_components=synthesis_run_contract_components,
        )
        from Trainforge.synthesis.synthesis_contract_guard import register_contract_files
        register_contract_files({
            str(PROJECT_ROOT / relative): digest
            for relative, digest in static_synthesis_contract_components[
                "files"
            ].items()
        })
    os.environ["TRAINFORGE_SYNTHESIS_RUN_CONTRACT_SHA256"] = (
        synthesis_run_contract_sha256
    )
    os.environ["TRAINFORGE_SYNTHESIS_CONTRACT_COMPONENTS_SHA256"] = (
        synthesis_contract_components_sha256
    )
    run_contract_env_set = True
    capture.synthesis_run_contract_sha256 = synthesis_run_contract_sha256
    capture.synthesis_run_contract_components = (
        synthesis_run_contract_components
    )
    capture.synthesis_contract_components_sha256 = (
        synthesis_contract_components_sha256
    )
    try:
        # split chunk traversal into "count eligible" and
        # "iterate emit-order" so stratified sampling can reorder the
        # emit-order without changing the eligibility tally.
        eligible_chunks: List[Tuple[int, Dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            if chunk.get("_eval_holdout_reserved") is True:
                # Reserved units deliberately enter the normal two-kind
                # traversal so each receives an explicit, resumable
                # eval_holdout_reserved disposition. Pair eligibility rejects
                # both branches before provider dispatch.
                stats.chunks_eligible += 1
                eligible_chunks.append((idx, chunk))
                continue
            if not _eligible(chunk):
                stats.chunks_skipped_no_lo += 1
                continue
            stats.chunks_eligible += 1
            eligible_chunks.append((idx, chunk))

        # Apply stratified sampling if requested. The original (idx, chunk)
        # tuples are preserved so each chunk keeps its original seed offset
        # -- otherwise idempotence under `--seed N` would break.
        if stratify_dims and eligible_chunks:
            rng = random.Random(seed)
            target = max_pairs if max_pairs is not None else len(eligible_chunks)
            target = min(target, len(eligible_chunks))
            picked = _stratified_sample(
                [c for _, c in eligible_chunks],
                stratify_dims,
                target_count=target,
                rng=rng,
            )
            picked_ids = {id(c): True for c in picked}
            iter_chunks = [(i, c) for (i, c) in eligible_chunks if id(c) in picked_ids]
            # Track the bucket distribution so callers (and tests) can
            # confirm the sampler actually balanced.
            for c in picked:
                for d in stratify_dims:
                    bucket = _stratify_key(c, d)
                    stats.stratify_distribution.setdefault(d, {})
                    stats.stratify_distribution[d][bucket] = (
                        stats.stratify_distribution[d].get(bucket, 0) + 1
                    )
        else:
            iter_chunks = list(eligible_chunks)

        if resolved_max_concurrent > 1:
            progress_writer = SynthesisProgressWriter(
                run_id=os.environ.get("ED4ALL_RUN_ID"),
                total_units=len(iter_chunks),
                max_concurrent=resolved_max_concurrent,
                provider=provider,
                model=rejection_contract_model_id,
                fresh_start_id=requested_fresh_start_id,
                marker_digest=fresh_start_marker_digest,
                holdout_identity=holdout_identity,
            )

        # Effective per-artifact cap. None -> unlimited. We apply it to
        # instruction and preference outputs independently so a request for
        # `--max-pairs 50` produces at most 50 of each (matches the tests'
        # expectation that capping is per-file, not the combined total).
        per_artifact_cap = max_pairs

        # Warn pre-flight when --max-pairs will clip the run before all
        # eligible chunks are visited: the cap stops the traversal at the
        # Nth pair, so chunks later in the order are never visited at all
        # (a low cap can miss every property-bearing chunk). Surfacing it
        # here lets the operator abort before paying for the full run.
        if (
            max_pairs is not None
            and max_pairs < len(iter_chunks)
            and smoke_mode == "none"
        ):
            logger.warning(
                "Wave 119: --max-pairs=%d will clip this run before all "
                "%d eligible chunks are visited. Property-coverage gates "
                "may underreport because surface forms anchored in late "
                "chunks will not be sampled. Remove --max-pairs (or set "
                "it above eligible-chunks) for a full-corpus run.",
                max_pairs, len(iter_chunks),
            )

        # graceful SynthesisBudgetExceeded handling.
        # When the claude_session provider hits its dispatch cap mid-loop,
        # we stop emitting + persist whatever we have so far so the
        # caller can write a pilot_progress.json snapshot and return
        # SynthesisStats with capped_at_max_dispatches=True.
        # ``_SBE`` and ``_budget_exhausted_exc`` are hoisted above the
        # try-block so the finally-block can reference them.

        # count chunks fully processed (post both instruction
        # + preference branches) so the periodic pilot_report.md writer
        # snapshots a consistent view of all pairs from each chunk.
        chunks_processed_counter = 0

        # Factory-side dedupe. The zero-tolerance ``duplicates`` gate flags
        # any cross-chunk paraphrase collision; tracking emitted prompts and
        # rejecting the second occurrence keeps the gate clean without
        # re-running the paraphrase. Distinct sets per artefact so an
        # instruction prompt can legitimately match a preference prompt.
        emitted_inst_prompts: set = set()
        emitted_pref_prompts: set = set()

        # hoist deterministic generators ABOVE the chunk loop.
        # The four generators (kg_metadata, violation_detection,
        # abstention, schema_translation) walk fixture catalogs / the
        # pedagogy graph / the property manifest — none of them iterate
        # over chunks. Running them upfront means their pairs land in
        # the .jsonl.in_progress sidecar within the first ~minute, so an
        # operator can ``tail -f`` and verify all ``--with-*`` flags
        # wired through without waiting for the multi-hour paraphrase
        # loop to finish. A killed run mid-paraphrase preserves the
        # deterministic output in the sidecar instead of losing it.

        # SFT data program (S3/S4/S5): resolve the assessment/graph arms
        # (kwarg OR env, so the ``_synthesize_training`` pipeline seam can
        # drive them without a pipeline_tools edit) and load the pedagogy-
        # graph holdout split ONCE. The withheld-edge index is the S4 design
        # rule enforcement: every graph->pair generator below consumes the
        # holdout-REDUCED graph so a withheld edge can never train.
        _with_assessment_sft = _resolve_bool_env(
            "ED4ALL_WITH_ASSESSMENT_SFT", with_assessment_sft,
        )
        _with_graph_sft = _resolve_bool_env(
            "ED4ALL_WITH_GRAPH_SFT", with_graph_sft,
        )
        # S4 sequencing fix: on a fresh (in-build) corpus the holdout split
        # does not exist yet at synthesis time — pre-build it BEFORE loading
        # the withheld-edge index so the reduction below has real edges to
        # withhold instead of silently no-oping (train-on-test race with the
        # Stage-B eval harness's lazy holdout build). Only fires when a
        # graph->pair generator is actually enabled.
        if _with_graph_sft or with_kg_metadata:
            _ensure_holdout_split_for_graph_pairs(corpus_dir, holdout_split_path)
        _withheld_edge_index = _load_withheld_edge_index(
            corpus_dir, holdout_split_path,
        )
        # SFT program S8/S9 (memorization probe): the held-out assessment-item id
        # set. Empty when no probe is configured (no
        # ``training_specs/.memorization_holdout.json``) -> byte-identical legacy
        # behaviour (no item withheld). Passed to the assessment->SFT generator so
        # a withheld item never trains, making the probe's held-out slice genuinely
        # unseen. Best-effort: a missing helper / unreadable file -> empty set.
        try:
            from Trainforge.training.memorization_probe import (
                load_holdout_exclusion as _load_holdout_exclusion,
            )
            _holdout_item_ids = _load_holdout_exclusion(corpus_dir)
        except Exception:  # noqa: BLE001 — holdout is best-effort; empty = no withholding
            _holdout_item_ids = set()

        # SFT data program S1/Phase-1: append assessment->SFT pairs (open-book,
        # rationale-augmented; solve+steps / error-diagnosis / explain-why /
        # grade-rubric / hint-no-reveal / verify-answer). Hoisted here so the
        # deterministic cohort lands in the .in_progress sidecar up front.
        if _with_assessment_sft:
            from Trainforge.generators.assessment_sft_generator import (
                generate_assessment_sft_pairs,
            )
            _assess_doc, _ak_doc = _resolve_assessment_docs(corpus_dir)
            if _assess_doc is None:
                logger.warning(
                    "with_assessment_sft=True but no assessments.json under "
                    "%s; skipping assessment->SFT generator.", corpus_dir,
                )
            else:
                _chunks_by_id = {
                    str(c.get("chunk_id") or c.get("id")): c
                    for c in chunks
                    if isinstance(c, dict) and (c.get("chunk_id") or c.get("id"))
                }
                _asft_count = 0
                for _pair in generate_assessment_sft_pairs(
                    _assess_doc,
                    _chunks_by_id,
                    capture,
                    answer_key_doc=_ak_doc,
                    max_pairs=assessment_sft_max_pairs,
                    holdout_item_ids=_holdout_item_ids,
                ):
                    # Stamp the deterministic-template audit fields so the
                    # post-emit W2.E + W4.A/B/C gate-runner walks recognise
                    # these open-book pairs as legitimately-bypassed.
                    _stamp_deterministic_pair_audit_fields(_pair, capture=capture)
                    instruction_records.append(_pair)
                    _utils_append_jsonl(inst_progress_fh, _pair, flush=False)
                    _asft_count += 1
                inst_progress_fh.flush()
                stats.assessment_sft_pairs_emitted = _asft_count
                stats.instruction_pairs_emitted += _asft_count
                logger.info(
                    "SFT data program S1: appended %d assessment->SFT pairs.",
                    _asft_count,
                )

        # SFT data program S5/Phase-2: append concept-graph->SFT pairs
        # (relation-QA / prereq study-path / concept verbalization), over the
        # holdout-REDUCED concept graph, consensus-filtered.
        if _with_graph_sft:
            from Trainforge.generators.graph_sft_generator import (
                generate_graph_sft_pairs,
            )
            _cg_path = _resolve_concept_graph_path(corpus_dir)
            if _cg_path is None:
                logger.warning(
                    "with_graph_sft=True but no concept_graph_semantic.json "
                    "under %s; skipping concept-graph->SFT generator.",
                    corpus_dir,
                )
            else:
                try:
                    _cg_payload = json.loads(_cg_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as _cg_exc:
                    logger.warning(
                        "with_graph_sft: failed to read %s (%s); skipping.",
                        _cg_path, _cg_exc,
                    )
                    _cg_payload = None
                if isinstance(_cg_payload, dict):
                    _cg_reduced, _cg_removed = _reduce_graph_by_holdout(
                        _cg_payload, _withheld_edge_index,
                    )
                    stats.holdout_edges_excluded += _cg_removed
                    _gsft_count = 0
                    for _pair in generate_graph_sft_pairs(
                        _cg_reduced,
                        capture,
                        max_pairs=graph_sft_max_pairs,
                        seed=seed,
                    ):
                        _stamp_deterministic_pair_audit_fields(
                            _pair, capture=capture,
                        )
                        instruction_records.append(_pair)
                        _utils_append_jsonl(inst_progress_fh, _pair, flush=False)
                        _gsft_count += 1
                    inst_progress_fh.flush()
                    stats.graph_sft_pairs_emitted = _gsft_count
                    stats.instruction_pairs_emitted += _gsft_count
                    logger.info(
                        "SFT data program S5: appended %d concept-graph->SFT "
                        "pairs (holdout-excluded %d edges).",
                        _gsft_count, _cg_removed,
                    )

        # append KG-metadata pairs (yes/no membership
        # probes mirroring faithfulness._RELATION_TEMPLATES). Closes the
        # zero-KG-metadata-recall regression —
        # the eval harness asks these questions, the corpus must teach
        # them.
        if with_kg_metadata:
            from Trainforge.generators.kg_metadata_generator import (
                generate_kg_metadata_pairs,
            )
            ped_path = _resolve_pedagogy_graph_path(
                corpus_dir, pedagogy_graph_path,
            )
            if ped_path is None:
                logger.warning(
                    "with_kg_metadata=True but no pedagogy_graph.json on "
                    "disk; skipping KG-metadata generator.",
                )
            else:
                ped_payload = json.loads(
                    ped_path.read_text(encoding="utf-8"),
                )
                # SFT data program S4: feed kg_metadata the holdout-REDUCED
                # pedagogy graph so a withheld edge can never become a
                # positive training pair (the exact leak the eval holds out).
                ped_payload, _ped_removed = _reduce_graph_by_holdout(
                    ped_payload, _withheld_edge_index,
                )
                stats.holdout_edges_excluded += _ped_removed
                kg_pairs, kg_stats = generate_kg_metadata_pairs(
                    ped_payload,
                    capture=capture,
                    max_pairs=int(kg_metadata_max_pairs),
                    seed=seed,
                )
                # stamp audit fields BEFORE the on-disk
                # emit so every kg_metadata pair on disk carries the
                # deterministic-template skip-discriminator that the
                # W2.E + W4.A + W4.B + W4.C gate-runner walks read.
                for _p in kg_pairs:
                    _stamp_deterministic_pair_audit_fields(_p, capture=capture)
                instruction_records.extend(kg_pairs)
                stats.kg_metadata_pairs_emitted = kg_stats.pairs_emitted
                stats.instruction_pairs_emitted += kg_stats.pairs_emitted
                # mirror to .in_progress sidecar.
                for _p in kg_pairs:
                    _utils_append_jsonl(inst_progress_fh, _p, flush=False)
                inst_progress_fh.flush()
                logger.info(
                    "Audit 2026-04-30: appended %d KG-metadata pairs "
                    "(positives=%d, negatives=%d, capped=%s) sourced from %s",
                    kg_stats.pairs_emitted,
                    kg_stats.positives_emitted,
                    kg_stats.negatives_emitted,
                    kg_stats.capped_at_max_pairs,
                    ped_path,
                )

        # append violation-detection pairs (pyshacl-
        # oracle-verified (graph, shape, valid?, reason) tuples). Closes
        # the zero-negative-grounding regression — the corpus must teach
        # the model to refuse a graph that violates a shape.
        #
        # gate SHACL-specific violation pairs by the manifest's
        # validation_kind field. The hand-curated catalog hardcodes sh:/
        # rdfs:/owl: shapes + pyshacl as the oracle, so a non-RDF/SHACL
        # course (e.g. JSON Schema) toggling --with-violation-detection
        # would silently get RDF/SHACL pairs polluting its training data.
        # When the manifest is missing OR validation_kind != "shacl",
        # short-circuit and warn so an operator sees the intentional skip.
        _vk = (
            pilot_manifest.validation_kind if pilot_manifest is not None else None
        )
        _family_for_log = (
            pilot_manifest.family if pilot_manifest is not None else None
        )
        if with_violation_detection and (
            pilot_manifest is None or _vk != "shacl"
        ):
            logger.warning(
                "violation_generator skipped: family=%s validation_kind=%s",
                _family_for_log,
                _vk,
            )
        if with_violation_detection and pilot_manifest is not None and _vk == "shacl":
            from Trainforge.generators.violation_generator import (
                built_in_shape_catalog,
                generate_violation_pairs,
            )
            # Build chunks_by_surface_form so violation pairs anchor to
            # a chunk that actually teaches the constraint type, when
            # one exists in the property manifest.
            chunks_by_form: Dict[str, List[str]] = {}
            if pilot_manifest is not None:
                for chunk in chunks:
                    cid = chunk.get("chunk_id")
                    if not cid:
                        continue
                    text = str(chunk.get("text") or "")
                    for sf in pilot_manifest.detect_surface_forms(text):
                        chunks_by_form.setdefault(sf, []).append(str(cid))
            try:
                vio_pairs, vio_stats = generate_violation_pairs(
                    capture=capture,
                    fixtures=built_in_shape_catalog(),
                    chunks_by_surface_form=chunks_by_form or None,
                    seed=seed,
                    max_pairs=violation_detection_max_pairs,
                )
            except RuntimeError as exc:
                # pyshacl is optional. A missing dep should warn, not
                # break the whole synthesis run.
                logger.warning(
                    "Audit 2026-04-30: violation generator skipped (%s)",
                    exc,
                )
                vio_pairs, vio_stats = [], None
            if vio_stats is not None:
                # stamp audit fields BEFORE the on-disk
                # emit so every violation pair on disk carries the
                # deterministic-template skip-discriminator.
                for _p in vio_pairs:
                    _stamp_deterministic_pair_audit_fields(_p, capture=capture)
                instruction_records.extend(vio_pairs)
                stats.violation_pairs_emitted = vio_stats.pairs_emitted
                stats.instruction_pairs_emitted += vio_stats.pairs_emitted
                # mirror to .in_progress sidecar.
                for _p in vio_pairs:
                    _utils_append_jsonl(inst_progress_fh, _p, flush=False)
                inst_progress_fh.flush()
                logger.info(
                    "Audit 2026-04-30: appended %d violation-detection "
                    "pairs (valid=%d, invalid=%d, oracle_disagreements=%d)",
                    vio_stats.pairs_emitted,
                    vio_stats.valid_pairs,
                    vio_stats.invalid_pairs,
                    vio_stats.oracle_disagreements,
                )

        # append abstention
        # probes ('the source does not establish X'). Closes the
        # abstention regression — the eval harness probes
        # for absent edges and the corpus must teach the model to
        # abstain rather than hallucinate yes-answers.
        if with_abstention:
            from Trainforge.generators.abstention_generator import (
                generate_abstention_pairs,
            )
            ped_path = _resolve_pedagogy_graph_path(
                corpus_dir, pedagogy_graph_path,
            )
            if ped_path is None:
                logger.warning(
                    "with_abstention=True but no pedagogy_graph.json "
                    "on disk; skipping abstention generator.",
                )
            else:
                ped_payload = json.loads(
                    ped_path.read_text(encoding="utf-8"),
                )
                ab_pairs, ab_stats = generate_abstention_pairs(
                    ped_payload,
                    capture=capture,
                    max_pairs=int(abstention_max_pairs),
                    seed=seed,
                )
                # stamp audit fields BEFORE the on-disk
                # emit so every abstention pair on disk carries the
                # deterministic-template skip-discriminator.
                for _p in ab_pairs:
                    _stamp_deterministic_pair_audit_fields(_p, capture=capture)
                instruction_records.extend(ab_pairs)
                stats.abstention_pairs_emitted = ab_stats.pairs_emitted
                stats.instruction_pairs_emitted += ab_stats.pairs_emitted
                # mirror to .in_progress sidecar.
                for _p in ab_pairs:
                    _utils_append_jsonl(inst_progress_fh, _p, flush=False)
                inst_progress_fh.flush()
                logger.info(
                    "Wave 124: appended %d abstention pairs (chunks_with_silent=%d, "
                    "skipped_no_concepts=%d, capped=%s) from %s",
                    ab_stats.pairs_emitted,
                    ab_stats.chunks_with_silent,
                    ab_stats.chunks_skipped_no_concepts,
                    ab_stats.capped_at_max_pairs,
                    ped_path,
                )

        # append schema-to-
        # English bridge pairs. Walks the property manifest's surface
        # forms (sh:datatype, rdfs:subClassOf, ...) and emits one
        # definition + one usage pair per CURIE. Closes the schema-
        # to-English gap that weakens faithfulness.
        if with_schema_translation:
            from Trainforge.generators.schema_translation_generator import (
                generate_schema_translation_pairs,
            )
            manifest_for_st = pilot_manifest
            if manifest_for_st is None:
                # pilot_manifest is loaded for the property-coverage
                # surface earlier. If no manifest is on disk for this
                # course, schema-translation has nothing to bridge.
                logger.warning(
                    "with_schema_translation=True but no property "
                    "manifest is on disk for this course; skipping "
                    "schema-translation generator.",
                )
            else:
                st_pairs, st_stats = generate_schema_translation_pairs(
                    manifest_for_st,
                    capture=capture,
                    max_pairs=int(schema_translation_max_pairs),
                    seed=seed,
                )
                # stamp audit fields BEFORE the on-disk
                # emit so every schema_translation pair on disk carries
                # the deterministic-template skip-discriminator.
                for _p in st_pairs:
                    _stamp_deterministic_pair_audit_fields(_p, capture=capture)
                instruction_records.extend(st_pairs)
                stats.schema_translation_pairs_emitted = st_stats.pairs_emitted
                stats.instruction_pairs_emitted += st_stats.pairs_emitted
                # mirror to .in_progress sidecar.
                for _p in st_pairs:
                    _utils_append_jsonl(inst_progress_fh, _p, flush=False)
                inst_progress_fh.flush()
                logger.info(
                    "Wave 124: appended %d schema-translation pairs "
                    "(surface_forms_used=%d, skipped_no_definition=%d, "
                    "capped=%s) from manifest_family=%s",
                    st_stats.pairs_emitted,
                    st_stats.surface_forms_used,
                    st_stats.surface_forms_skipped_no_definition,
                    st_stats.capped_at_max_pairs,
                    manifest_for_st.family,
                )

        # instantiate the per-pair promotion validator once.
        # Filters every accepted pair through the 7-criterion hard-
        # rejection chain (placeholder residue, unsupported answer,
        # weak distractor, unanswerable stem, source-free generation,
        # low Bloom alignment, generic rationale). Lazy-loads the
        # embedder + pinned zero-shot Bloom classifier on first use; degrades gracefully
        # to Jaccard when [embedding] extras are absent.
        from lib.validators.pair.promotion import (
            TrainingPairPromotionValidator,
        )
        _promotion_validator = TrainingPairPromotionValidator(
            embedder=production_embedder,
        )

        # per-pair claim-support (W4.A) + LO-refs (W4.B)
        # filters. Both run AFTER the W2.E promotion validator returns
        # ``validated``; rejects increment ``claim_support_rejected`` /
        # ``lo_refs_rejected`` and bump the matching key in the
        # free-form ``promotion_rejection_reasons`` dict. Imported here
        # (rather than at module top) to mirror the W2.E lazy-import
        # convention and to keep the import error surface scoped to
        # the synthesis call path. W4.A is warning-severity at the
        # gate-runner surface (NLI confidence is fuzzy day-1) but the
        # per-pair filter still drops the pair so the audit trail and
        # disk emit stay consistent. W4.B is critical-severity at the
        # gate-runner surface (phantom-LO is a structural mismatch).
        from lib.validators.pair.claim_support import (
            PairClaimSupportValidator,
            summarize_claim_support_rejection,
        )
        from lib.validators.pair.lo_refs import (
            PairLearningOutcomeRefsValidator,
        )
        _claim_support_validator = PairClaimSupportValidator()
        _lo_refs_validator = PairLearningOutcomeRefsValidator()

        # pre-compute the chunk_id -> text lookup map
        # consumed by ``PairClaimSupportValidator.validate_pair`` for
        # the per-claim attribution scoping path. Built once outside
        # the chunk loop so per-pair calls are O(1) lookups, not
        # O(n_chunks) rebuilds. When ``chunks_by_id`` is empty
        # (curriculum mode off — pedagogy graph not threaded), the
        # lookup is None and the validator graceful-degrades to
        # whole-chunk-text scoring (back-compat day-1; behaviour
        # byte-identical to the W4.A pre-W5.D path).
        _claim_support_chunk_text_map: Optional[Dict[str, str]] = (
            {
                cid: str(c.get("text") or "")
                for cid, c in chunks_by_id.items()
            }
            if chunks_by_id
            else None
        )

        # pre-compute the dart:<slug>#<block_id> -> text
        # lookup map consumed by ``PairClaimSupportValidator.validate_pair``
        # for the dual-source DART cross-check (chunker-drift slice).
        # Built once outside the chunk loop — per-pair calls are O(1)
        # lookups in the helper. When ``staging_dir`` is None or has no
        # extant DART HTML, the map is empty and the validator no-ops
        # cleanly via the empty-map arm in ``validate_pair`` (back-compat
        # day-1 — legacy / non-textbook-to-course corpora keep their
        # current behaviour). The severity dial flips to ``"warning"``
        # only when the map is non-empty so the per-pair LLM-cost
        # arithmetic in §3 of the dual-source plan stays accurate (the
        # extra NLI calls only fire when there's a DART universe to
        # check against).
        _dart_block_text_map: Dict[str, str] = (
            _resolve_dart_block_text_map(staging_dir)
        )
        _dual_source_severity: str = (
            "warning" if _dart_block_text_map else "off"
        )
        if _dart_block_text_map:
            logger.info(
                "Wave 9 TIGHT: dual-source DART cross-check enabled "
                "(staging_dir=%s, %d block-text entries loaded).",
                staging_dir, len(_dart_block_text_map),
            )

        # Per-pair tri-axis objective-delivery
        # filter. Runs AFTER W4.A/W4.B return ``validated``. Rejects
        # increment ``objective_delivery_rejected`` and the matching
        # key in the free-form ``promotion_rejection_reasons`` dict
        # (three new reason keys: ``objective_statement_undersupported``,
        # ``objective_bloom_undermet``, ``objective_verb_absent``).
        # Warning-severity day-1 at the gate-runner surface (NLI
        # confidence is fuzzy on a fresh corpus); the per-pair filter
        # still drops the pair so the audit trail and disk emit stay
        # consistent. ``_load_synthesized_objectives_for_w4c`` is a
        # thin helper around ``lib.validators.abcd_objective._flatten_objectives``
        # that returns ``{lo.id: {"statement", "bloom_level",
        # "bloom_verb"}}`` — accepts both Courseforge-form
        # (``terminal_objectives``/``chapter_objectives``) and LibV2-form
        # (``terminal_outcomes``/``component_objectives``) so the same
        # call site works whether the corpus is being rebuilt from a
        # fresh Courseforge run or replayed from a LibV2 archive.
        from lib.validators.pair.objective_delivery import (
            PairObjectiveDeliveryValidator,
        )
        # Staged-v4 makes canonical objective resolution part of the pair
        # contract and therefore fails closed when it cannot be verified.
        # The flag-off path retains the historical audit-only behavior so a
        # bare legacy run is byte-compatible with its prior admission policy.
        from Trainforge.generators.staged.provider import (
            staged_synthesis_v4_enabled,
        )
        _objective_delivery_validator = PairObjectiveDeliveryValidator(
            require_verifiable=staged_synthesis_v4_enabled(),
        )

        # Seal a fresh synthesis pass to one complete implementation/runtime
        # contract before the first provider call. The complete objective map
        # is available only after the W4.C preflight above.
        synthesis_run_contract_sha256 = (
            _synthesis_rejection_contract_fingerprint(
                provider=provider,
                model_id=rejection_contract_model_id,
                pair_seed=0,
                source_chunk={
                    "objectives": _objectives_map,
                    "holdout_identity": dict(holdout_identity or {}),
                },
                endpoint_identity=rejection_endpoint_identity,
                generation_params=rejection_generation_params,
                static_components=static_synthesis_contract_components,
            )
        )
        synthesis_run_contract_components = {
            **static_synthesis_contract_components,
            "objectives_sha256": _normalized_sha256(_objectives_map),
            "holdout_identity_sha256": _normalized_sha256(
                dict(holdout_identity or {})
            ),
            # Same record-not-gate contract as the staged site above.
            "verdict_policy": _verdict_policy_digest(),
        }
        synthesis_contract_components_sha256 = _normalized_sha256(
            synthesis_run_contract_components
        )
        os.environ["TRAINFORGE_SYNTHESIS_CONTRACT_COMPONENTS_SHA256"] = (
            synthesis_contract_components_sha256
        )
        os.environ["TRAINFORGE_SYNTHESIS_RUN_CONTRACT_SHA256"] = (
            synthesis_run_contract_sha256
        )
        run_contract_env_set = True
        capture.synthesis_run_contract_sha256 = synthesis_run_contract_sha256
        capture.synthesis_run_contract_components = (
            synthesis_run_contract_components
        )
        capture.synthesis_contract_components_sha256 = (
            synthesis_contract_components_sha256
        )
        capture.log_decision(
            decision_type="instruction_pair_synthesis",
            decision=(
                f"Starting instruction/preference synthesis over {len(chunks)} "
                f"chunks for course '{course_code}' using provider="
                f"'{provider}' seed={seed}."
            ),
            rationale=(
                "The sealed synthesis contract now binds model, prompts, "
                "schemas, validators, objectives, holdout, and implementation "
                f"components before dispatch; contract="
                f"{synthesis_run_contract_sha256}."
            ),
            alternatives_considered=[
                {
                    "option": "resume-with-unsealed-contract",
                    "reason_rejected": (
                        "could mix provider implementations across checkpoints"
                    ),
                },
            ],
        )
        if requested_fresh_start_id is not None:
            from Trainforge.synthesis.synthesis_contract_guard import (
                register_contract_files,
            )
            register_contract_files({
                str(PROJECT_ROOT / relative): digest
                for relative, digest in static_synthesis_contract_components[
                    "files"
                ].items()
            })
        if requested_fresh_start_id is not None:
            from Trainforge.synthesis.synthesis_fresh_start import (
                bind_fresh_start_run_contract,
            )

            resume_artifacts_exist = any(
                path is not None and path.is_file() and path.stat().st_size > 0
                for path in (checkpoint_path, generation_checkpoint_path)
            )
            bind_fresh_start_run_contract(
                training_specs_dir,
                expected_fresh_start_id=requested_fresh_start_id,
                run_contract_sha256=synthesis_run_contract_sha256,
                resume_artifacts_exist=resume_artifacts_exist,
                contract_components=synthesis_run_contract_components,
            )
            # Any absent/mixed row fails before worker construction/dispatch.
        pair_checkpoint_cache = _load_synthesis_pairs_checkpoint(
            checkpoint_path,
            expected_fresh_start_id=requested_fresh_start_id,
            expected_marker_digest=fresh_start_marker_digest,
            expected_holdout_identity=holdout_identity,
            expected_run_contract_sha256=(
                synthesis_run_contract_sha256
                if requested_fresh_start_id is not None else None
            ),
        )
        generation_checkpoint_cache = load_generation_journal(
            generation_checkpoint_path,
            expected_fresh_start_id=requested_fresh_start_id,
            expected_marker_digest=fresh_start_marker_digest,
            expected_holdout_identity=holdout_identity,
            expected_run_contract_sha256=(
                synthesis_run_contract_sha256
                if requested_fresh_start_id is not None else None
            ),
        )
        if pair_checkpoint_cache:
            logger.info(
                "Resuming from %s with %d contract-compatible pair(s)",
                checkpoint_path,
                len(pair_checkpoint_cache),
            )
        if _mine_mode != MINE_MODE_OFF:
            # Resume completeness: a rejected unit is SKIPPED on resume, never
            # regenerated, so the prior run's rejected rows are the only place
            # those candidates exist. Seeding from the already-identity-checked
            # cache is what makes a killed-and-resumed run's pool identical to
            # an uninterrupted one's — and a partial pool is a biased pool.
            _seeded = _reject_pool.seed_from_checkpoint_cache(
                pair_checkpoint_cache
            )
            if _seeded:
                logger.info(
                    "Reject mining: seeded %d mineable rejection(s) from the "
                    "resume checkpoint",
                    _seeded,
                )

        # No mutable resume/output file is opened until every identity and
        # contract check above has succeeded.
        for sidecar in (instruction_progress, preference_progress):
            if sidecar.exists() and sidecar.stat().st_size > 0:
                logger.warning("Overwriting resumable sidecar: %s", sidecar)
        instruction_progress.parent.mkdir(parents=True, exist_ok=True)
        buffered_instruction = inst_progress_fh.getvalue()
        buffered_preference = pref_progress_fh.getvalue()
        inst_progress_fh = instruction_progress.open("w", encoding="utf-8")
        pref_progress_fh = preference_progress.open("w", encoding="utf-8")
        if buffered_instruction:
            inst_progress_fh.write(buffered_instruction)
            inst_progress_fh.flush()
        if buffered_preference:
            pref_progress_fh.write(buffered_preference)
            pref_progress_fh.flush()
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            buffered_checkpoint = checkpoint_fh.getvalue()
            checkpoint_fh = checkpoint_path.open("a", encoding="utf-8")
            checkpoint_fh._terminal_semantic_keys = set(
                pair_checkpoint_cache
            )
            checkpoint_fh._enforce_terminal_uniqueness = (
                requested_fresh_start_id is not None
            )
            if buffered_checkpoint:
                for buffered_line in buffered_checkpoint.splitlines():
                    buffered_record = json.loads(buffered_line)
                    if requested_fresh_start_id is not None:
                        buffered_record["fresh_start_id"] = (
                            requested_fresh_start_id
                        )
                        buffered_record["fresh_start_marker_digest"] = (
                            fresh_start_marker_digest
                        )
                        buffered_record[
                            "synthesis_run_contract_sha256"
                        ] = synthesis_run_contract_sha256
                        buffered_record[
                            "synthesis_contract_components_sha256"
                        ] = synthesis_contract_components_sha256
                    if holdout_identity is not None:
                        buffered_record["synthesis_holdout_identity"] = dict(
                            holdout_identity
                        )
                    checkpoint_fh.write(
                        json.dumps(buffered_record, separators=(",", ":"))
                        + "\n"
                    )
                checkpoint_fh.flush()
            if requested_fresh_start_id is not None:
                checkpoint_fh._fresh_start_id = requested_fresh_start_id
                checkpoint_fh._fresh_start_marker_digest = (
                    fresh_start_marker_digest
                )
                checkpoint_fh._synthesis_run_contract_sha256 = (
                    synthesis_run_contract_sha256
                )
                checkpoint_fh._synthesis_contract_components_sha256 = (
                    synthesis_contract_components_sha256
                )
            if holdout_identity is not None:
                checkpoint_fh._synthesis_holdout_identity = holdout_identity
        generation_journal.run_contract_sha256 = (
            synthesis_run_contract_sha256
            if requested_fresh_start_id is not None else None
        )
        generation_journal.component_manifest_sha256 = (
            synthesis_contract_components_sha256
            if requested_fresh_start_id is not None else None
        )

        def _assert_live_synthesis_contract() -> None:
            if requested_fresh_start_id is None:
                return
            live = _synthesis_static_contract_components(
                provider=provider,
                model_id=rejection_contract_model_id,
                endpoint_identity=rejection_endpoint_identity,
                generation_params=rejection_generation_params,
            )
            if live != static_synthesis_contract_components:
                changed = sorted(
                    key for key in set(live) | set(
                        static_synthesis_contract_components
                    )
                    if live.get(key)
                    != static_synthesis_contract_components.get(key)
                )
                raise RuntimeError(
                    "synthesis implementation/runtime contract drifted during "
                    "the run; checkpoint preserved and provider dispatch "
                    f"blocked; changed components: {', '.join(changed)}"
                )

        def _build_checked_generation_bundle(
            item: Tuple[int, Dict[str, Any]],
        ) -> _ChunkGenerationBundle:
            _assert_live_synthesis_contract()
            return _build_chunk_generation_bundle(
                item,
                seed=seed,
                provider=provider,
                instruction_variants=instruction_variants,
                paraphrase_provider=paraphrase_provider,
                pilot_manifest=pilot_manifest,
                checkpoint_cache=pair_checkpoint_cache,
                generation_cache=generation_checkpoint_cache,
                generation_journal=generation_journal,
                recovery_coordinator=recovery_coordinator,
                fingerprint_for_seed=_rejection_fingerprint_for_seed,
                objectives=_objectives_map,
                capture=capture,
            )

        if resolved_max_concurrent > 1 and provider == "claude_session":
            raise ValueError(
                "Concurrent training synthesis is not supported for "
                "provider='claude_session'; its session dispatch budget and "
                "mailbox ordering require max_concurrent=1."
            )
        if resolved_max_concurrent > 1:
            logger.info(
                "training_synthesis: bounded concurrent generation enabled "
                "(max_concurrent=%d, queue_capacity=%d, ordered_writer=1)",
                resolved_max_concurrent,
                resolved_max_concurrent,
            )

        generation_map = BoundedOrderedMap(
            iter_chunks,
            (
                _build_checked_generation_bundle
                if resolved_max_concurrent > 1
                else (lambda _item: _ChunkGenerationBundle(
                    instruction_results=tuple(
                        None for _ in range(instruction_variants)
                    ),
                    preference_result=None,
                    instruction_ineligible_reasons=tuple(
                        None for _ in range(instruction_variants)
                    ),
                    preference_ineligible_reason=None,
                    provider_results=0,
                    cached_replays=0,
                    fingerprint="",
                ))
            ),
            max_concurrent=resolved_max_concurrent,
            stop_requested=(
                stop_control.stop_requested
                if resolved_max_concurrent > 1
                else None
            ),
        )

        generation_iterator = iter(generation_map)
        for ordered_generation in generation_iterator:
            _assert_live_synthesis_contract()
            idx, chunk = ordered_generation.item
            generation_bundle = ordered_generation.value
            provider_results_completed += generation_bundle.provider_results
            cached_generation_replays += generation_bundle.cached_replays
            transient_attempts += generation_bundle.transient_attempts
            recovered_units += generation_bundle.recovered_units
            exhausted_units += generation_bundle.exhausted_units
            fatal_units += generation_bundle.fatal_units
            if generation_bundle.errors:
                first_error = generation_bundle.errors[0]
                if progress_writer is not None:
                    active, queued, in_flight = generation_map.metrics_snapshot()
                    journal_counts = summarize_generation_journal(
                        generation_checkpoint_path
                    )
                    progress_writer.update(
                        transient_count=sum(
                            error.transient
                            for error in generation_bundle.errors
                        ),
                        provider_results=journal_counts["provider_results"],
                        cached_replays=cached_generation_replays,
                        transient_attempts=journal_counts["transient_attempts"],
                        recovered_units=journal_counts["recovered_units"],
                        exhausted_units=journal_counts["exhausted_units"],
                        fatal_units=journal_counts["fatal_units"],
                        active_workers=active,
                        queued_units=queued,
                        in_flight=in_flight,
                        gate_readiness="blocked",
                        checkpointed=True,
                    )
                if first_error.transient:
                    raise SynthesisProviderError(
                        "Concurrent synthesis has retriable generation failures; "
                        f"resume will retry only transient/missing units. First: "
                        f"{first_error.kind}[{first_error.variant_index}] "
                        f"chunk={first_error.chunk_id} attempt={first_error.attempt}: "
                        f"{first_error.message}",
                        code="concurrent_transient_generation",
                        chunk_id=first_error.chunk_id,
                    )
                raise RuntimeError(
                    "Concurrent synthesis encountered a deterministic fatal "
                    f"generation error in {first_error.kind}"
                    f"[{first_error.variant_index}] chunk={first_error.chunk_id}: "
                    f"{first_error.error_type}: {first_error.message}"
                )
            if _budget_exhausted_exc is not None:
                break
            # Graceful stop ("checkpoint on command"). This is the same
            # unit boundary as the budget-exhausted break above, but the
            # disposition differs: the budget path BREAKS and then falls
            # through to the post-loop telemetry + artifact persistence,
            # so run_synthesis RETURNS its SynthesisStats normally and the
            # caller marks the phase COMPLETE (capped_at_max_dispatches
            # is a success-shaped return). A completed phase would never
            # be re-dispatched on --resume, so mirroring the break here
            # would silently drop the un-synthesized tail. Instead we
            # RAISE GracefulStopRequested: the executor maps it to a
            # PAUSED (never failed, never retried) result and the runner
            # stamps the phase `paused` so a later --resume re-enters and
            # replays the resume sidecar.
            #
            # Every accepted pair for every ALREADY-processed chunk has
            # already been appended to the resume checkpoint sidecar at
            # :2621 / :2976 (flushed + fsync'd per emit), so the stop
            # loses ZERO completed work — the current chunk has not yet
            # dispatched. Raising here (before the post-loop artifact
            # `open("w")` at the "Persist artifacts" block) also keeps the
            # sidecar on disk: ``clean_exit`` stays False, so the finally
            # block never unlinks it (risk R5). This is a plain sequential
            # loop (NOT an asyncio.gather site), so the direct-raise
            # check_stop pattern applies — the STOP_MARKER return-value
            # discipline is only for gathered coroutines.
            if (
                resolved_max_concurrent == 1
                and stop_control.stop_requested()
            ):
                _stop_units = (
                    stats.instruction_pairs_emitted
                    + stats.preference_pairs_emitted
                )
                logger.warning(
                    "training_synthesis: graceful stop requested after %d "
                    "checkpointed pair(s) (%d instruction + %d preference); "
                    "raising GracefulStopRequested so the phase pauses "
                    "(resumable) rather than completing. Resume sidecar "
                    "preserved at %s.",
                    _stop_units,
                    stats.instruction_pairs_emitted,
                    stats.preference_pairs_emitted,
                    checkpoint_path,
                )
                if progress_writer is not None:
                    progress_writer.update(
                        state="paused",
                        completed_units=chunks_processed_counter,
                        terminal_units=chunks_processed_counter,
                        accepted_count=_stop_units,
                        rejected_count=(
                            stats.instruction_pairs_rejected
                            + stats.preference_pairs_rejected
                        ),
                        sft_count=stats.instruction_pairs_emitted,
                        dpo_count=stats.preference_pairs_emitted,
                        provider_results=provider_results_completed,
                        cached_replays=cached_generation_replays,
                        transient_attempts=transient_attempts,
                        recovered_units=recovered_units,
                        exhausted_units=exhausted_units,
                        fatal_units=fatal_units,
                        active_workers=0,
                        queued_units=0,
                        in_flight=0,
                        stop_requested=True,
                        gate_readiness="pending",
                        rejection_reasons=stats.rejected_reasons,
                        checkpointed=True,
                    )
                stop_control.check_stop(
                    "training_synthesis.pair_loop", _stop_units, run_id=None,
                )
            # detect property surface forms in this chunk so
            # the paraphrase provider preserves them verbatim. None ->
            # no manifest loaded; empty list -> chunk doesn't reference
            # any declared property.
            chunk_text = str(chunk.get("text") or "")
            chunk_preserve_tokens = (
                pilot_manifest.detect_surface_forms(chunk_text)
                if pilot_manifest is not None
                else []
            )
            # Force-injection covers the FULL chunk CURIE set, not just
            # manifest-declared surface forms: only manifest-declared CURIEs
            # land in chunk_preserve_tokens via detect_surface_forms, so
            # restricting to those lets the LLM silently strip every other
            # prefix in the corpus (prov:, dcat:, geo:, ...).
            #
            # Non-manifest CURIEs hit the degraded fallback path in the
            # factories (FORM_DATA only knows the 40 manifest CURIEs),
            # which makes the operator-visible WARN + decision-capture
            # surface accurate: they fall through to token-stuffing
            # explicitly rather than disappearing silently.
            #
            # Cap at 6 extras to bound the prompt-suffix budget on
            # CURIE-rich chunks; sort stably so the same chunk picks
            # the same 6 across runs.
            chunk_full_curies = extract_curies(chunk_text)
            extra_anchor_tokens = sorted(
                chunk_full_curies - set(chunk_preserve_tokens)
            )[:6]
            effective_preserve_tokens = (
                chunk_preserve_tokens + extra_anchor_tokens
            )
            # pull a cache-stable chunk_id once per chunk so
            # the per-variant resume-cache lookup uses the same value the
            # synthesize_instruction_pair / synthesize_preference_pair
            # path would have written into ``pair["chunk_id"]``. Falls
            # back to the same ``id`` / ``chunk_id`` precedence used
            # everywhere else in this module.
            chunk_id_for_checkpoint = str(
                chunk.get("id") or chunk.get("chunk_id") or ""
            )
            # --- Instruction pair ---
            for variant_index in range(instruction_variants):
                inst_capped = (
                    per_artifact_cap is not None
                    and stats.instruction_pairs_emitted >= per_artifact_cap
                )
                if inst_capped:
                    stats.capped_at_max_pairs = True
                    break
                pair_seed = seed + idx + (variant_index * 100_000)
                pair_chunk = _focus_chunk_on_objective(
                    chunk, seed=pair_seed, objectives=_objectives_map,
                )
                _inst_eligibility = _pair_eligibility_for_mode(
                    pair_chunk, kind="instruction",
                )
                if not _inst_eligibility.eligible:
                    _ineligible_reason = str(
                        _inst_eligibility.reason or "unspecified_ineligible"
                    )
                    _record_ineligible_disposition(
                        stats=stats,
                        checkpoint_fh=checkpoint_fh,
                        capture=capture,
                        chunk_id=chunk_id_for_checkpoint,
                        kind="instruction",
                        variant_index=variant_index,
                        provider=provider,
                        seed=pair_seed,
                        reason=_ineligible_reason,
                        detail=getattr(_inst_eligibility, "detail", None),
                        contract_fingerprint=(
                            _rejection_fingerprint_for_seed(
                                pair_seed, pair_chunk,
                            )
                        ),
                    )
                    continue
                # per-pair resume-cache check. If the prior
                # run already emitted this (chunk_id, "instruction",
                # variant_index) triple, replay the cached pair into
                # the canonical buffers + sidecars and skip the LLM
                # dispatch entirely. Provider mismatch invalidates the
                # entry so a ``local`` cache isn't honoured by a
                # ``together`` re-run.
                _cp_key = (
                    chunk_id_for_checkpoint, "instruction", variant_index,
                )
                if chunk_id_for_checkpoint and _cp_key in pair_checkpoint_cache:
                    cached_record = pair_checkpoint_cache[_cp_key]
                    cached_provider = cached_record.get("provider")
                    if cached_provider != provider:
                        logger.warning(
                            "Synthesis checkpoint provider mismatch for "
                            "%s: cached=%r, current=%r — discarding",
                            _cp_key, cached_provider, provider,
                        )
                    else:
                        if (
                            cached_record.get("disposition") == "rejected"
                            and not _checkpoint_rejection_matches_contract(
                                cached_record,
                                _rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                            )
                        ):
                            logger.warning(
                                "Synthesis checkpoint rejection contract "
                                "mismatch for %s — regenerating",
                                _cp_key,
                            )
                        elif cached_record.get("disposition") == "rejected":
                            cached_reason = str(
                                cached_record.get("reason")
                                or "unspecified_rejection"
                            )
                            # The cached rejection's per-seed contract
                            # fingerprint MATCHED the one recomputed for this
                            # run, so a candidate seeded from it belongs to the
                            # CURRENT generation contract. This is the only
                            # place that can decide it -- the fingerprint is a
                            # function of the seed and the focused chunk, which
                            # the pure mining module never sees. An unconfirmed
                            # resume-cache candidate is dropped by
                            # ``select_mined_pairs`` (funnel ``stale_contract``)
                            # rather than paired against a current-contract
                            # ``chosen``. No-op when mining is off (empty pool).
                            _reject_pool.confirm_contract(_cp_key)
                            _replay_terminal_rejection_stats(
                                stats,
                                kind="instruction",
                                reason=cached_reason,
                            )
                            continue
                        current_contract = _rejection_fingerprint_for_seed(
                            pair_seed, pair_chunk
                        )
                        # Fail-closed: a rejected row (which may now carry a
                        # pair) can reach here when its contract fingerprint
                        # mismatched above, and must never replay as accepted.
                        cached_pair = _resolve_cached_accepted_pair(
                            cached_record,
                            current_contract,
                            pair_chunk,
                            cache_key=_cp_key,
                        )
                        if not cached_pair:
                            pass
                        else:
                            cached_prompt = cached_pair.get("prompt", "")
                            if cached_prompt in emitted_inst_prompts:
                                # Cross-chunk dedupe still applies on resume
                                # — skip the cached record, don't re-emit.
                                stats.instruction_pairs_rejected += 1
                                stats.rejected_reasons[
                                    "instruction:duplicate_prompt"
                                ] = (
                                    stats.rejected_reasons.get(
                                        "instruction:duplicate_prompt", 0,
                                    ) + 1
                                )
                                continue
                            instruction_records.append(cached_pair)
                            emitted_inst_prompts.add(cached_prompt)
                            stats.instruction_pairs_emitted += 1
                            # Mirror the cached pair to the .in_progress sidecar
                            # so ``tail -f`` continues to surface every
                            # accepted pair, regardless of whether it came
                            # from the LLM or the resume cache.
                            _utils_append_jsonl(inst_progress_fh, cached_pair)
                            continue
                try:
                    if resolved_max_concurrent > 1:
                        inst_result = generation_bundle.instruction_results[
                            variant_index
                        ]
                        # A None slot means the prefetch planner found a valid
                        # checkpoint disposition. The source-order cache branch
                        # above must have consumed it; reaching here would imply
                        # checkpoint validation drift between planner and writer.
                        if inst_result is None:
                            raise RuntimeError(
                                "Concurrent synthesis checkpoint planner/writer "
                                f"drift for instruction key {_cp_key}"
                            )
                    else:
                        with _micro_generation_unit(
                            "instruction", variant_index,
                        ):
                            inst_result = _call_with_seat_recovery(
                                lambda: synthesize_instruction_pair(
                                    pair_chunk,
                                    seed=pair_seed,
                                    provider=provider,
                                    paraphrase_provider=paraphrase_provider,
                                    preserve_tokens=(
                                        effective_preserve_tokens or None
                                    ),
                                    capture=capture,
                                ),
                                recovery_coordinator=recovery_coordinator,
                                incident_context={
                                    "workflow_phase": "training_synthesis",
                                    "task_id": (
                                        f"{chunk_id_for_checkpoint}:"
                                        f"instruction:{variant_index}"
                                    ),
                                    "chunk_id": chunk_id_for_checkpoint,
                                    "kind": "instruction",
                                    "variant_index": variant_index,
                                    "fingerprint": (
                                        _rejection_fingerprint_for_seed(
                                            pair_seed, pair_chunk
                                        )
                                    ),
                                },
                            )
                        provider_results_completed += 1
                except _SBE as exc:
                    _budget_exhausted_exc = exc
                    break
                if inst_result.pair is None and inst_result.quality.get(
                    "ineligible"
                ):
                    # Deterministic pre-dispatch exclusion signalled by the
                    # factory (no derivable topic / no groundable completion
                    # content). Recorded as ineligible, NOT rejected: nothing
                    # was generated to judge, and no model call was spent.
                    _record_ineligible_disposition(
                        stats=stats,
                        checkpoint_fh=checkpoint_fh,
                        capture=capture,
                        chunk_id=chunk_id_for_checkpoint,
                        kind="instruction",
                        variant_index=variant_index,
                        provider=provider,
                        seed=pair_seed,
                        reason=str(
                            inst_result.quality.get("reason")
                            or "unspecified_ineligible"
                        ),
                        detail=(
                            f"content_sources="
                            f"{inst_result.quality.get('content_sources')}"
                        ),
                        contract_fingerprint=_rejection_fingerprint_for_seed(
                            pair_seed, pair_chunk,
                        ),
                    )
                elif inst_result.pair is None:
                    stats.instruction_pairs_rejected += 1
                    reason = inst_result.quality.get("reason") or "gate_failed"
                    stats.rejected_reasons[f"instruction:{reason}"] = (
                        stats.rejected_reasons.get(f"instruction:{reason}", 0) + 1
                    )
                    _checkpoint_terminal_rejection(
                        checkpoint_fh,
                        chunk_id=chunk_id_for_checkpoint,
                        kind="instruction",
                        variant_index=variant_index,
                        provider=provider,
                        seed=pair_seed,
                        contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                        reason=reason,
                        rejection_evidence=inst_result.quality.get(
                            "rejection_evidence"
                        ),
                    )
                else:
                    # Opt-in content_type enforcement against the ChunkType
                    # enum. Flag off -> no-op; flag on -> fail-closed. Mirrors
                    # the TRAINFORGE_VALIDATE_CHUNKS pattern in
                    # process_course.py.
                    ct_value = inst_result.pair.get("content_type", "")
                    if not validate_chunk_type(ct_value):
                        stats.instruction_pairs_rejected += 1
                        reason = "invalid_content_type"
                        stats.rejected_reasons[f"instruction:{reason}"] = (
                            stats.rejected_reasons.get(f"instruction:{reason}", 0) + 1
                        )
                        chunk_id = inst_result.pair.get("chunk_id", "<unknown>")
                        # Fail-closed: raise so the pipeline surfaces the bad vocabulary
                        # rather than silently rejecting. Caller sets the env var
                        # intentionally; silent drop would undermine that intent.
                        assert_chunk_type(
                            ct_value,
                            context=f"instruction_pair.chunk_id={chunk_id}",
                        )
                    _apply_instruction_variant(
                        inst_result.pair,
                        variant_index,
                        learner_persona=(
                            pilot_manifest.learner_persona
                            if pilot_manifest is not None
                            else None
                        ),
                    )
                    # Audit-log when the paraphrase emit chose the
                    # deterministic draft path rather than the LLM paraphrase,
                    # so post-hoc analysis can find chunks where the LLM fell
                    # short of the (opt-in) preservation floor. The event name
                    # deliberately asserts no failure: the deterministic-draft
                    # path is the EXPECTED outcome on ``degraded_placeholder``
                    # FORM_DATA entries, so it only reports which path emitted.
                    if inst_result.pair.get("paraphrase_fallback_reason"):
                        capture.log_decision(
                            decision_type="paraphrase_used_deterministic_draft",
                            decision=(
                                f"Used the grounded deterministic instruction "
                                f"draft for chunk "
                                f"{inst_result.pair['chunk_id']}; paraphrase "
                                f"failed with "
                                f"{inst_result.pair['paraphrase_fallback_reason']} "
                                f"while preserving {effective_preserve_tokens}."
                            ),
                            rationale=(
                                f"Provider '{provider}' exhausted its bounded "
                                f"paraphrase attempts with "
                                f"{len(effective_preserve_tokens)} nonempty "
                                "required preservation tokens; the already-"
                                "grounded draft retains those exact tokens "
                                "without adding unsupported content."
                            ),
                            context=f"chunk_id={inst_result.pair['chunk_id']}",
                        )
                    # Cross-chunk prompt-collision
                    # dedupe. Skip the emit if the final-shape prompt
                    # already landed for an earlier chunk; the rejected
                    # bucket gets a ``duplicate_prompt`` reason so the
                    # operator can grep telemetry without inspecting
                    # JSONL byte-for-byte.
                    final_prompt = inst_result.pair.get("prompt", "")
                    if final_prompt in emitted_inst_prompts:
                        stats.instruction_pairs_rejected += 1
                        stats.rejected_reasons["instruction:duplicate_prompt"] = (
                            stats.rejected_reasons.get("instruction:duplicate_prompt", 0) + 1
                        )
                        _checkpoint_terminal_rejection(
                            checkpoint_fh,
                            chunk_id=chunk_id_for_checkpoint,
                            kind="instruction",
                            variant_index=variant_index,
                            provider=provider,
                            seed=pair_seed,
                            contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                            reason="duplicate_prompt",
                        )
                        continue
                    capture.log_decision(
                        decision_type="instruction_pair_synthesis",
                        decision=(
                            f"Emit instruction pair for chunk {inst_result.pair['chunk_id']} "
                            f"(template={inst_result.template_id}, "
                            f"variant={variant_index}, "
                            f"bloom={inst_result.pair['bloom_level']})."
                        ),
                        rationale=inst_result.rationale,
                        alternatives_considered=inst_result.alternatives or None,
                        context=(
                            f"topic='{inst_result.topic}'; "
                            f"content_type='{inst_result.pair['content_type']}'; "
                            f"quality={inst_result.quality}"
                        ),
                    )
                    inst_result.pair["decision_capture_id"] = _last_event_id(capture)
                    if _attach_source_grounding(inst_result.pair, pair_chunk):
                        stats.source_grounded_pairs += 1
                    # per-pair, post-emit, pre-write filter.
                    # Stamps audit fields on the pair regardless of
                    # outcome; on reject, skips append + sidecar +
                    # checkpoint so a rejected pair never lands on disk
                    # and is not re-emitted on resume.
                    # increment candidate counter BEFORE
                    # the validator call — every factory-output pair
                    # that survived the per-template quality gate AND
                    # the cross-chunk dedupe is a candidate.
                    stats.candidate_pairs_total += 1
                    _promo_status, _promo_reason, _promo_fields = (
                        _promotion_validator.validate_pair(
                            inst_result.pair,
                            kind="instruction",
                            chunk=pair_chunk,
                            decision_capture=capture,
                        )
                    )
                    inst_result.pair.update(_promo_fields)
                    if _promo_status == "rejected":
                        inst_result.pair["promotion_status"] = "rejected"
                        inst_result.pair["rejection_reason"] = _promo_reason
                        stats.instruction_pairs_rejected += 1
                        _key = (
                            f"instruction:promotion:{_promo_reason}"
                        )
                        stats.rejected_reasons[_key] = (
                            stats.rejected_reasons.get(_key, 0) + 1
                        )
                        stats.dropped_count += 1
                        # promotion-ladder rejection counters.
                        stats.rejected_promotion_pairs += 1
                        _reason_key = _promo_reason or "unknown"
                        stats.promotion_rejection_reasons[_reason_key] = (
                            stats.promotion_rejection_reasons.get(
                                _reason_key, 0
                            ) + 1
                        )
                        _checkpoint_terminal_rejection(
                            checkpoint_fh,
                            chunk_id=chunk_id_for_checkpoint,
                            kind="instruction",
                            variant_index=variant_index,
                            provider=provider,
                            seed=pair_seed,
                            contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                            reason=f"promotion:{_promo_reason}",
                        )
                        continue
                    inst_result.pair["promotion_status"] = _promo_status
                    # per-pair claim-support (W4.A) +
                    # LO-refs (W4.B) filters run AFTER the W2.E
                    # promotion validator returns ``validated``. Each
                    # reject increments the matching W4 counter AND
                    # the existing W2.E ladder counters
                    # (``rejected_promotion_pairs`` +
                    # ``promotion_rejection_reasons``) so the
                    # ``candidate == validated + rejected_promotion``
                    # invariant from W2.E stays intact — we delay the
                    # ``validated_pairs_total`` increment until ALL
                    # three validators pass, rather than incrementing
                    # at W2.E pass and decrementing on W4 reject.
                    if _promo_status == "validated":
                        # thread the precomputed
                        # ``_claim_support_chunk_text_map`` so the
                        # claim-support validator can scope NLI
                        # premise per-claim when chunk["key_claims"]
                        # carries the W1.5 / W5.A structured shape.
                        # When the map is None (curriculum mode off —
                        # ``chunks_by_id`` empty), the validator
                        # graceful-degrades to whole-chunk-text
                        # scoring (back-compat day-1).
                        # thread the dual-source DART
                        # cross-check map + severity dial. When the
                        # map is empty (no staging_dir / non-textbook-
                        # to-course corpus), severity is "off" so the
                        # validator skips the DART pass entirely.
                        _claim_status, _claim_reason, _claim_fields = (
                            _claim_support_validator.validate_pair(
                                inst_result.pair,
                                kind="instruction",
                                chunk=pair_chunk,
                                chunk_id_to_text_map=(
                                    _claim_support_chunk_text_map
                                ),
                                decision_capture=capture,
                                dart_block_text_map=(
                                    _dart_block_text_map or None
                                ),
                                dual_source_severity=_dual_source_severity,
                            )
                        )
                        inst_result.pair.update(_claim_fields)
                        if _claim_status == "rejected":
                            inst_result.pair["promotion_status"] = "rejected"
                            inst_result.pair["rejection_reason"] = _claim_reason
                            stats.instruction_pairs_rejected += 1
                            _claim_key = (
                                f"instruction:claim_support:{_claim_reason}"
                            )
                            stats.rejected_reasons[_claim_key] = (
                                stats.rejected_reasons.get(_claim_key, 0) + 1
                            )
                            stats.dropped_count += 1
                            stats.rejected_promotion_pairs += 1
                            _claim_reason_key = _claim_reason or "unknown"
                            stats.promotion_rejection_reasons[
                                _claim_reason_key
                            ] = (
                                stats.promotion_rejection_reasons.get(
                                    _claim_reason_key, 0,
                                ) + 1
                            )
                            stats.claim_support_rejected += 1
                            # Reject-mining capture. Returns None (today's
                            # exact row) unless the flag is on AND both sides
                            # sit inside the preference_pair length band.
                            # claim_support is the only stage whose rejects
                            # carry the per-claim NLI scores a near-miss
                            # judgement needs.
                            _reject_reason = f"claim_support:{_claim_reason}"
                            _reject_fingerprint = (
                                _rejection_fingerprint_for_seed(
                                    pair_seed, pair_chunk,
                                )
                            )
                            _reject_payload = build_capture_payload(
                                inst_result.pair,
                                reason=_reject_reason,
                                mode=_mine_mode,
                                kind="instruction",
                            )
                            _reject_pool.record_rejection(
                                chunk_id=chunk_id_for_checkpoint,
                                kind="instruction",
                                variant_index=variant_index,
                                pair=_reject_payload,
                                provider=provider,
                                seed=pair_seed,
                                reason=_reject_reason,
                                contract_fingerprint=_reject_fingerprint,
                            )
                            _checkpoint_terminal_rejection(
                                checkpoint_fh,
                                chunk_id=chunk_id_for_checkpoint,
                                kind="instruction",
                                variant_index=variant_index,
                                provider=provider,
                                seed=pair_seed,
                                contract_fingerprint=_reject_fingerprint,
                                reason=_reject_reason,
                                pair=_reject_payload,
                                # Persist the bounded per-sentence NLI verdict.
                                # Without it a claim_support rejection leaves
                                # only a reason string, so "was the verifier
                                # right?" is unanswerable without a lucky
                                # leftover checkpoint.
                                rejection_evidence=(
                                    summarize_claim_support_rejection(
                                        _claim_fields,
                                        rejection_reason=_claim_reason,
                                    )
                                ),
                            )
                            continue
                        _lo_status, _lo_reason, _lo_fields = (
                            _lo_refs_validator.validate_pair(
                                inst_result.pair,
                                kind="instruction",
                                chunk=pair_chunk,
                                decision_capture=capture,
                            )
                        )
                        inst_result.pair.update(_lo_fields)
                        if _lo_status == "rejected":
                            inst_result.pair["promotion_status"] = "rejected"
                            inst_result.pair["rejection_reason"] = _lo_reason
                            stats.instruction_pairs_rejected += 1
                            _lo_key = f"instruction:lo_refs:{_lo_reason}"
                            stats.rejected_reasons[_lo_key] = (
                                stats.rejected_reasons.get(_lo_key, 0) + 1
                            )
                            stats.dropped_count += 1
                            stats.rejected_promotion_pairs += 1
                            _lo_reason_key = _lo_reason or "unknown"
                            stats.promotion_rejection_reasons[
                                _lo_reason_key
                            ] = (
                                stats.promotion_rejection_reasons.get(
                                    _lo_reason_key, 0,
                                ) + 1
                            )
                            stats.lo_refs_rejected += 1
                            # Routed through the same capture helper for
                            # uniformity; it returns None for any stage other
                            # than claim_support, so this row is unchanged. An
                            # lo_refs reject already PASSED claim_support (its
                            # prose is grounded) and failed on objective
                            # binding, which the miner's anchor rule holds
                            # constant — it would be an off-objective negative.
                            _reject_payload = build_capture_payload(
                                inst_result.pair,
                                reason=f"lo_refs:{_lo_reason}",
                                mode=_mine_mode,
                                kind="instruction",
                            )
                            _checkpoint_terminal_rejection(
                                checkpoint_fh,
                                chunk_id=chunk_id_for_checkpoint,
                                kind="instruction",
                                variant_index=variant_index,
                                provider=provider,
                                seed=pair_seed,
                                contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                                reason=f"lo_refs:{_lo_reason}",
                                pair=_reject_payload,
                            )
                            continue
                        # Per-pair tri-axis
                        # objective-delivery filter runs AFTER W4.B
                        # passes. Mirrors the W4.A/W4.B reject
                        # bookkeeping (W2.E ladder counters +
                        # W4-specific counter + ``rejected_reasons``
                        # bucket). Always stamps ``_obj_fields`` on
                        # the pair so ``pair_objective_alignment``
                        # lands on disk regardless of pass/fail —
                        # downstream consumers can replay the per-LO
                        # entailment / Bloom / verb signal even on
                        # passing pairs.
                        _obj_status, _obj_reason, _obj_fields = (
                            _objective_delivery_validator.validate_pair(
                                inst_result.pair,
                                kind="instruction",
                                chunk=pair_chunk,
                                objectives=_objectives_map,
                                decision_capture=capture,
                            )
                        )
                        inst_result.pair.update(_obj_fields)
                        if _obj_status == "rejected":
                            inst_result.pair["promotion_status"] = "rejected"
                            inst_result.pair["rejection_reason"] = _obj_reason
                            stats.instruction_pairs_rejected += 1
                            _obj_key = (
                                f"instruction:objective_delivery:{_obj_reason}"
                            )
                            stats.rejected_reasons[_obj_key] = (
                                stats.rejected_reasons.get(_obj_key, 0) + 1
                            )
                            stats.dropped_count += 1
                            stats.rejected_promotion_pairs += 1
                            _obj_reason_key = _obj_reason or "unknown"
                            stats.promotion_rejection_reasons[
                                _obj_reason_key
                            ] = (
                                stats.promotion_rejection_reasons.get(
                                    _obj_reason_key, 0,
                                ) + 1
                            )
                            stats.objective_delivery_rejected += 1
                            # Same as the lo_refs site: capture returns None
                            # for this stage (a delivery/metadata failure, not
                            # a grounding failure), so the row is unchanged.
                            _reject_payload = build_capture_payload(
                                inst_result.pair,
                                reason=f"objective_delivery:{_obj_reason}",
                                mode=_mine_mode,
                                kind="instruction",
                            )
                            _checkpoint_terminal_rejection(
                                checkpoint_fh,
                                chunk_id=chunk_id_for_checkpoint,
                                kind="instruction",
                                variant_index=variant_index,
                                provider=provider,
                                seed=pair_seed,
                                contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                                reason=f"objective_delivery:{_obj_reason}",
                                pair=_reject_payload,
                            )
                            continue
                        # All four validators (W2.E + W4.A + W4.B +
                        # W4.C) passed — surface on both the W2.E
                        # counter (preserves the invariant) and the
                        # new W4.C post-W4 survivor counter.
                        stats.pair_validation_passed += 1
                    # validated counter increments on
                    # promotion_status != "rejected".
                    stats.validated_pairs_total += 1
                    instruction_records.append(inst_result.pair)
                    emitted_inst_prompts.add(final_prompt)
                    stats.instruction_pairs_emitted += 1
                    # trainable counter — incremented after
                    # the pair lands in the in-memory records list (which
                    # is byte-equivalent to the JSONL emit a few lines
                    # below). Distinct counter from validated so a future
                    # post-validation write loss surfaces here without
                    # conflating the two ladder steps.
                    stats.trainable_pairs_total += 1
                    # mirror to .in_progress sidecar with
                    # flush() so ``tail -f`` and post-kill inspection
                    # see this pair without waiting on OS buffers.
                    _utils_append_jsonl(inst_progress_fh, inst_result.pair)
                    # append the accepted pair to the resume
                    # checkpoint. A subsequent run that loads this
                    # sidecar will skip the LLM dispatch for this
                    # (chunk_id, "instruction", variant_index) triple.
                    _append_synthesis_pairs_checkpoint(
                        checkpoint_fh,
                        chunk_id=str(inst_result.pair.get("chunk_id", "")),
                        kind="instruction",
                        variant_index=variant_index,
                        pair=inst_result.pair,
                        provider=provider,
                        seed=pair_seed,
                        contract_fingerprint=_rejection_fingerprint_for_seed(
                            pair_seed, pair_chunk
                        ),
                    )

            # --- Preference pair ---
            pair_seed = seed + idx
            pair_chunk = _focus_chunk_on_objective(
                chunk, seed=pair_seed, objectives=_objectives_map,
            )
            pref_capped = (
                per_artifact_cap is not None
                and stats.preference_pairs_emitted >= per_artifact_cap
            )
            _pref_eligibility = _pair_eligibility_for_mode(
                pair_chunk, kind="preference",
            )
            _pref_ineligible = (
                not pref_capped and not _pref_eligibility.eligible
            )
            if _pref_ineligible:
                _ineligible_reason = str(
                    _pref_eligibility.reason or "unspecified_ineligible"
                )
                _record_ineligible_disposition(
                    stats=stats,
                    checkpoint_fh=checkpoint_fh,
                    capture=capture,
                    chunk_id=chunk_id_for_checkpoint,
                    kind="preference",
                    variant_index=0,
                    provider=provider,
                    seed=pair_seed,
                    reason=_ineligible_reason,
                    detail=getattr(_pref_eligibility, "detail", None),
                    contract_fingerprint=(
                        _rejection_fingerprint_for_seed(pair_seed, pair_chunk)
                    ),
                )
            # per-pair resume-cache check for the preference
            # branch. Single-variant (only one preference pair per
            # chunk), so the variant_index is always 0.
            _pref_cp_key = (chunk_id_for_checkpoint, "preference", 0)
            _pref_cache_hit = (
                chunk_id_for_checkpoint
                and _pref_cp_key in pair_checkpoint_cache
                and not pref_capped
                and not _pref_ineligible
            )
            if _pref_cache_hit:
                cached_record = pair_checkpoint_cache[_pref_cp_key]
                cached_provider = cached_record.get("provider")
                if cached_provider != provider:
                    logger.warning(
                        "Synthesis checkpoint provider mismatch for "
                        "%s: cached=%r, current=%r — discarding",
                        _pref_cp_key, cached_provider, provider,
                    )
                    _pref_cache_hit = False
                else:
                    if (
                        cached_record.get("disposition") == "rejected"
                        and not _checkpoint_rejection_matches_contract(
                            cached_record,
                            _rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                        )
                    ):
                        logger.warning(
                            "Synthesis checkpoint rejection contract "
                            "mismatch for %s — regenerating",
                            _pref_cp_key,
                        )
                        _pref_cache_hit = False
                    elif cached_record.get("disposition") == "rejected":
                        cached_reason = str(
                            cached_record.get("reason")
                            or "unspecified_rejection"
                        )
                        _replay_terminal_rejection_stats(
                            stats,
                            kind="preference",
                            reason=cached_reason,
                        )
                    else:
                        current_contract = _rejection_fingerprint_for_seed(
                            pair_seed, pair_chunk
                        )
                        _pref_cache_invalidations: List[str] = []
                        cached_pair = _resolve_cached_accepted_pair(
                            cached_record,
                            current_contract,
                            pair_chunk,
                            cache_key=_pref_cp_key,
                            invalidations=_pref_cache_invalidations,
                        )
                        if _pref_cache_invalidations:
                            # Same two conditions that cleared the flag before
                            # (contract mismatch, focus mismatch). A
                            # contract-compatible row with no pair records no
                            # invalidation and leaves the flag alone, as it did.
                            _pref_cache_hit = False
                        cached_prompt = cached_pair.get("prompt", "")
                        if cached_pair and cached_prompt in emitted_pref_prompts:
                            stats.preference_pairs_rejected += 1
                            stats.rejected_reasons[
                                "preference:duplicate_prompt"
                            ] = (
                                stats.rejected_reasons.get(
                                    "preference:duplicate_prompt", 0,
                                ) + 1
                            )
                            cached_pair = {}
                        if cached_pair:
                            preference_records.append(cached_pair)
                            emitted_pref_prompts.add(cached_prompt)
                            stats.preference_pairs_emitted += 1
                            _utils_append_jsonl(pref_progress_fh, cached_pair)
            if pref_capped:
                stats.capped_at_max_pairs = True
            elif _pref_ineligible:
                # Deterministic pre-dispatch exclusion recorded above.
                pass
            elif _pref_cache_hit:
                # Cache hit handled above — fall through to the
                # post-pair pilot_report progress block at end of
                # the chunk loop without dispatching to the LLM.
                pass
            else:
                try:
                    if resolved_max_concurrent > 1:
                        pref_result = generation_bundle.preference_result
                        if pref_result is None:
                            raise RuntimeError(
                                "Concurrent synthesis checkpoint planner/writer "
                                f"drift for preference key {_pref_cp_key}"
                            )
                    else:
                        with _micro_generation_unit("preference", 0):
                            pref_result = _call_with_seat_recovery(
                                lambda: synthesize_preference_pair(
                                    pair_chunk,
                                    seed=pair_seed,
                                    provider=provider,
                                    paraphrase_provider=paraphrase_provider,
                                    preserve_tokens=(
                                        effective_preserve_tokens or None
                                    ),
                                    capture=capture,
                                ),
                                recovery_coordinator=recovery_coordinator,
                                incident_context={
                                    "workflow_phase": "training_synthesis",
                                    "task_id": (
                                        f"{chunk_id_for_checkpoint}:preference:0"
                                    ),
                                    "chunk_id": chunk_id_for_checkpoint,
                                    "kind": "preference",
                                    "variant_index": 0,
                                    "fingerprint": (
                                        _rejection_fingerprint_for_seed(
                                            pair_seed, pair_chunk
                                        )
                                    ),
                                },
                            )
                        provider_results_completed += 1
                except _SBE as exc:
                    _budget_exhausted_exc = exc
                    break
                if pref_result.pair is None and pref_result.quality.get(
                    "ineligible"
                ):
                    _record_ineligible_disposition(
                        stats=stats,
                        checkpoint_fh=checkpoint_fh,
                        capture=capture,
                        chunk_id=chunk_id_for_checkpoint,
                        kind="preference",
                        variant_index=0,
                        provider=provider,
                        seed=pair_seed,
                        reason=str(
                            pref_result.quality.get("reason")
                            or "unspecified_ineligible"
                        ),
                        detail=(
                            f"content_sources="
                            f"{pref_result.quality.get('content_sources')}"
                        ),
                        contract_fingerprint=_rejection_fingerprint_for_seed(
                            pair_seed, pair_chunk,
                        ),
                    )
                elif pref_result.pair is None:
                    stats.preference_pairs_rejected += 1
                    reason = pref_result.quality.get("reason") or "gate_failed"
                    stats.rejected_reasons[f"preference:{reason}"] = (
                        stats.rejected_reasons.get(f"preference:{reason}", 0) + 1
                    )
                    _checkpoint_terminal_rejection(
                        checkpoint_fh,
                        chunk_id=chunk_id_for_checkpoint,
                        kind="preference",
                        variant_index=0,
                        provider=provider,
                        seed=pair_seed,
                        contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                        reason=reason,
                        rejection_evidence=pref_result.quality.get(
                            "rejection_evidence"
                        ),
                    )
                else:
                    if pref_result.pair.get("paraphrase_fallback_reason"):
                        capture.log_decision(
                            decision_type="paraphrase_used_deterministic_draft",
                            decision=(
                                f"Used the grounded deterministic preference "
                                f"draft for chunk "
                                f"{pref_result.pair['chunk_id']}; paraphrase "
                                f"failed with "
                                f"{pref_result.pair['paraphrase_fallback_reason']} "
                                f"while preserving {effective_preserve_tokens} "
                                "in the chosen field."
                            ),
                            rationale=(
                                f"Provider '{provider}' exhausted its bounded "
                                f"paraphrase attempts with "
                                f"{len(effective_preserve_tokens)} nonempty "
                                "required preservation tokens; the already-"
                                "grounded chosen draft retains those exact "
                                "tokens without adding unsupported content."
                            ),
                            context=f"chunk_id={pref_result.pair['chunk_id']}",
                        )
                    # Cross-chunk dedupe (preference).
                    # Nested ``if`` rather than ``continue`` so the
                    # outer chunk loop still falls through to the
                    # pilot_report progress block below.
                    final_pref_prompt = pref_result.pair.get("prompt", "")
                    if final_pref_prompt in emitted_pref_prompts:
                        stats.preference_pairs_rejected += 1
                        stats.rejected_reasons["preference:duplicate_prompt"] = (
                            stats.rejected_reasons.get("preference:duplicate_prompt", 0) + 1
                        )
                        _checkpoint_terminal_rejection(
                            checkpoint_fh,
                            chunk_id=chunk_id_for_checkpoint,
                            kind="preference",
                            variant_index=0,
                            provider=provider,
                            seed=pair_seed,
                            contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                            reason="duplicate_prompt",
                        )
                    else:
                        capture.log_decision(
                            decision_type="preference_pair_generation",
                            decision=(
                                f"Emit preference pair for chunk {pref_result.pair['chunk_id']} "
                                f"(source={pref_result.source}, "
                                f"misconception_id={pref_result.misconception_id})."
                            ),
                            rationale=pref_result.rationale,
                            alternatives_considered=pref_result.alternatives or None,
                            context=f"quality={pref_result.quality}",
                        )
                        pref_result.pair["decision_capture_id"] = _last_event_id(capture)
                        if _attach_source_grounding(pref_result.pair, pair_chunk):
                            stats.source_grounded_pairs += 1
                        # per-pair promotion filter on the
                        # preference branch. Mirrors the instruction
                        # branch exactly. Nested if/else (rather than
                        # continue) so the outer chunk loop still falls
                        # through to the pilot_report progress block —
                        # matches the surrounding control-flow
                        # convention for the preference branch.
                        # candidate counter increments
                        # BEFORE the validator dispatch, matching the
                        # instruction-branch semantics — every pair that
                        # survived the per-template quality gate AND the
                        # cross-chunk dedupe is a candidate.
                        stats.candidate_pairs_total += 1
                        _pref_promo_status, _pref_promo_reason, _pref_promo_fields = (
                            _promotion_validator.validate_pair(
                                pref_result.pair,
                                kind="preference",
                                chunk=pair_chunk,
                                decision_capture=capture,
                            )
                        )
                        pref_result.pair.update(_pref_promo_fields)
                        if _pref_promo_status == "rejected":
                            pref_result.pair["promotion_status"] = "rejected"
                            pref_result.pair["rejection_reason"] = _pref_promo_reason
                            stats.preference_pairs_rejected += 1
                            _pref_key = (
                                f"preference:promotion:{_pref_promo_reason}"
                            )
                            stats.rejected_reasons[_pref_key] = (
                                stats.rejected_reasons.get(_pref_key, 0) + 1
                            )
                            stats.dropped_count += 1
                            # promotion-ladder rejection
                            # counters mirror the instruction branch.
                            stats.rejected_promotion_pairs += 1
                            _pref_reason_key = (
                                _pref_promo_reason or "unknown"
                            )
                            stats.promotion_rejection_reasons[_pref_reason_key] = (
                                stats.promotion_rejection_reasons.get(
                                    _pref_reason_key, 0
                                ) + 1
                            )
                            _checkpoint_terminal_rejection(
                                checkpoint_fh,
                                chunk_id=chunk_id_for_checkpoint,
                                kind="preference",
                                variant_index=0,
                                provider=provider,
                                seed=pair_seed,
                                contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                                reason=f"promotion:{_pref_promo_reason}",
                            )
                        else:
                            pref_result.pair["promotion_status"] = (
                                _pref_promo_status
                            )
                            # per-pair claim-support
                            # (W4.A) + LO-refs (W4.B) filters mirror
                            # the instruction-pair branch. ``_pref_w4_rejected``
                            # tracks whether either W4 validator
                            # short-circuited so the surrounding
                            # nested-if structure (preference branch
                            # uses if/else rather than ``continue``)
                            # falls through to the chunk-loop
                            # progress block without appending the
                            # rejected pair to ``preference_records``.
                            _pref_w4_rejected = False
                            if _pref_promo_status == "validated":
                                # thread the precomputed
                                # chunk_id -> text map (same as the
                                # instruction-pair branch). Mirrors
                                # the W4.A symmetric instruction /
                                # preference handling.
                                # thread the dual-source
                                # DART cross-check map + severity dial.
                                # Mirrors the instruction-pair branch.
                                _pref_claim_status, _pref_claim_reason, _pref_claim_fields = (
                                    _claim_support_validator.validate_pair(
                                        pref_result.pair,
                                        kind="preference",
                                        chunk=pair_chunk,
                                        chunk_id_to_text_map=(
                                            _claim_support_chunk_text_map
                                        ),
                                        decision_capture=capture,
                                        dart_block_text_map=(
                                            _dart_block_text_map or None
                                        ),
                                        dual_source_severity=_dual_source_severity,
                                    )
                                )
                                pref_result.pair.update(_pref_claim_fields)
                                if _pref_claim_status == "rejected":
                                    pref_result.pair["promotion_status"] = "rejected"
                                    pref_result.pair["rejection_reason"] = _pref_claim_reason
                                    stats.preference_pairs_rejected += 1
                                    _pref_claim_key = (
                                        f"preference:claim_support:{_pref_claim_reason}"
                                    )
                                    stats.rejected_reasons[_pref_claim_key] = (
                                        stats.rejected_reasons.get(
                                            _pref_claim_key, 0,
                                        ) + 1
                                    )
                                    stats.dropped_count += 1
                                    stats.rejected_promotion_pairs += 1
                                    _pref_claim_reason_key = (
                                        _pref_claim_reason or "unknown"
                                    )
                                    stats.promotion_rejection_reasons[
                                        _pref_claim_reason_key
                                    ] = (
                                        stats.promotion_rejection_reasons.get(
                                            _pref_claim_reason_key, 0,
                                        ) + 1
                                    )
                                    stats.claim_support_rejected += 1
                                    _checkpoint_terminal_rejection(
                                        checkpoint_fh,
                                        chunk_id=chunk_id_for_checkpoint,
                                        kind="preference",
                                        variant_index=0,
                                        provider=provider,
                                        seed=pair_seed,
                                        contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                                        reason=(
                                            "claim_support:"
                                            f"{_pref_claim_reason}"
                                        ),
                                        # Same bounded per-sentence NLI
                                        # verdict as the instruction branch.
                                        rejection_evidence=(
                                            summarize_claim_support_rejection(
                                                _pref_claim_fields,
                                                rejection_reason=(
                                                    _pref_claim_reason
                                                ),
                                            )
                                        ),
                                    )
                                    _pref_w4_rejected = True
                                else:
                                    _pref_lo_status, _pref_lo_reason, _pref_lo_fields = (
                                        _lo_refs_validator.validate_pair(
                                            pref_result.pair,
                                            kind="preference",
                                            chunk=pair_chunk,
                                            decision_capture=capture,
                                        )
                                    )
                                    pref_result.pair.update(_pref_lo_fields)
                                    if _pref_lo_status == "rejected":
                                        pref_result.pair["promotion_status"] = "rejected"
                                        pref_result.pair["rejection_reason"] = _pref_lo_reason
                                        stats.preference_pairs_rejected += 1
                                        _pref_lo_key = (
                                            f"preference:lo_refs:{_pref_lo_reason}"
                                        )
                                        stats.rejected_reasons[_pref_lo_key] = (
                                            stats.rejected_reasons.get(
                                                _pref_lo_key, 0,
                                            ) + 1
                                        )
                                        stats.dropped_count += 1
                                        stats.rejected_promotion_pairs += 1
                                        _pref_lo_reason_key = (
                                            _pref_lo_reason or "unknown"
                                        )
                                        stats.promotion_rejection_reasons[
                                            _pref_lo_reason_key
                                        ] = (
                                            stats.promotion_rejection_reasons.get(
                                                _pref_lo_reason_key, 0,
                                            ) + 1
                                        )
                                        stats.lo_refs_rejected += 1
                                        _checkpoint_terminal_rejection(
                                            checkpoint_fh,
                                            chunk_id=chunk_id_for_checkpoint,
                                            kind="preference",
                                            variant_index=0,
                                            provider=provider,
                                            seed=pair_seed,
                                            contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                                            reason=f"lo_refs:{_pref_lo_reason}",
                                        )
                                        _pref_w4_rejected = True
                                    else:
                                        # Per-pair
                                        # tri-axis objective-delivery
                                        # filter on the preference
                                        # branch. Mirrors the
                                        # instruction-pair branch
                                        # exactly. Uses the same
                                        # ``_pref_w4_rejected`` flag so
                                        # the surrounding nested-if
                                        # structure (preference branch
                                        # uses if/else rather than
                                        # ``continue``) falls through
                                        # to the chunk-loop progress
                                        # block without appending the
                                        # rejected pair to
                                        # ``preference_records``.
                                        # Always stamps ``_pref_obj_fields``
                                        # so ``pair_objective_alignment``
                                        # lands on disk regardless of
                                        # pass/fail.
                                        _pref_obj_status, _pref_obj_reason, _pref_obj_fields = (
                                            _objective_delivery_validator.validate_pair(
                                                pref_result.pair,
                                                kind="preference",
                                                chunk=pair_chunk,
                                                objectives=_objectives_map,
                                                decision_capture=capture,
                                            )
                                        )
                                        pref_result.pair.update(_pref_obj_fields)
                                        if _pref_obj_status == "rejected":
                                            pref_result.pair["promotion_status"] = "rejected"
                                            pref_result.pair["rejection_reason"] = _pref_obj_reason
                                            stats.preference_pairs_rejected += 1
                                            _pref_obj_key = (
                                                f"preference:objective_delivery:{_pref_obj_reason}"
                                            )
                                            stats.rejected_reasons[_pref_obj_key] = (
                                                stats.rejected_reasons.get(
                                                    _pref_obj_key, 0,
                                                ) + 1
                                            )
                                            stats.dropped_count += 1
                                            stats.rejected_promotion_pairs += 1
                                            _pref_obj_reason_key = (
                                                _pref_obj_reason or "unknown"
                                            )
                                            stats.promotion_rejection_reasons[
                                                _pref_obj_reason_key
                                            ] = (
                                                stats.promotion_rejection_reasons.get(
                                                    _pref_obj_reason_key, 0,
                                                ) + 1
                                            )
                                            stats.objective_delivery_rejected += 1
                                            _checkpoint_terminal_rejection(
                                                checkpoint_fh,
                                                chunk_id=chunk_id_for_checkpoint,
                                                kind="preference",
                                                variant_index=0,
                                                provider=provider,
                                                seed=pair_seed,
                                                contract_fingerprint=_rejection_fingerprint_for_seed(pair_seed, pair_chunk),
                                                reason=(
                                                    "objective_delivery:"
                                                    f"{_pref_obj_reason}"
                                                ),
                                            )
                                            _pref_w4_rejected = True
                                        else:
                                            # All four validators
                                            # (W2.E + W4.A + W4.B +
                                            # W4.C) passed.
                                            stats.pair_validation_passed += 1
                            if _pref_w4_rejected:
                                # Skip the emit branch — fall through
                                # to the chunk-loop progress block.
                                # Don't append to preference_records,
                                # don't bump preference_pairs_emitted,
                                # don't mirror to sidecar / checkpoint;
                                # the rejection counters were already
                                # bumped inside the W4 branches above.
                                pass
                            else:
                                # validated counter
                                # increments on promotion_status !=
                                # "rejected" AND post-W4 survival.
                                stats.validated_pairs_total += 1
                                preference_records.append(pref_result.pair)
                                emitted_pref_prompts.add(final_pref_prompt)
                                stats.preference_pairs_emitted += 1
                                # trainable counter — pair
                                # has landed in the in-memory records
                                # list and will be flushed to JSONL via
                                # _write_jsonl at end of run.
                                stats.trainable_pairs_total += 1
                                # mirror to .in_progress sidecar.
                                _utils_append_jsonl(pref_progress_fh, pref_result.pair)
                                # append the accepted preference
                                # pair to the resume checkpoint.
                                _append_synthesis_pairs_checkpoint(
                                    checkpoint_fh,
                                    chunk_id=str(
                                        pref_result.pair.get("chunk_id", "")
                                    ),
                                    kind="preference",
                                    variant_index=0,
                                    pair=pref_result.pair,
                                    provider=provider,
                                    seed=pair_seed,
                                    contract_fingerprint=(
                                        _rejection_fingerprint_for_seed(
                                            pair_seed, pair_chunk
                                        )
                                    ),
                                )

            # every N processed chunks, regenerate the
            # in-flight pilot_report.md so the operator running a
            # multi-hour rebuild has live property-coverage /
            # template-distribution visibility. Atomic tmp-and-rename
            # write keeps a concurrent ``cat`` / ``less`` from
            # observing a half-written file.
            chunks_processed_counter += 1
            if progress_writer is not None:
                active, queued, in_flight = generation_map.metrics_snapshot()
                journal_counts = summarize_generation_journal(
                    generation_checkpoint_path
                )
                _accepted = (
                    stats.instruction_pairs_emitted
                    + stats.preference_pairs_emitted
                )
                _rejected = (
                    stats.instruction_pairs_rejected
                    + stats.preference_pairs_rejected
                )
                progress_writer.update(
                    completed_units=chunks_processed_counter,
                    terminal_units=chunks_processed_counter,
                    accepted_count=_accepted,
                    rejected_count=_rejected,
                    sft_count=stats.instruction_pairs_emitted,
                    dpo_count=stats.preference_pairs_emitted,
                    provider_results=journal_counts["provider_results"],
                    cached_replays=cached_generation_replays,
                    transient_count=0,
                    transient_attempts=journal_counts["transient_attempts"],
                    recovered_units=journal_counts["recovered_units"],
                    exhausted_units=journal_counts["exhausted_units"],
                    fatal_units=journal_counts["fatal_units"],
                    active_workers=active,
                    queued_units=queued,
                    in_flight=in_flight,
                    stop_requested=stop_control.stop_requested(),
                    gate_readiness="pending",
                    rejection_reasons=stats.rejected_reasons,
                    checkpointed=True,
                )
            if (
                pilot_manifest is not None
                and pilot_report_every > 0
                and chunks_processed_counter % pilot_report_every == 0
            ):
                from Trainforge.scripts.maintenance.pilot_report_helpers import (
                    count_property_coverage_from_records,
                    format_pilot_report,
                    template_distribution_from_records,
                    write_pilot_report_atomic,
                )
                _counts = count_property_coverage_from_records(
                    instruction_records, pilot_manifest,
                )
                _templates = template_distribution_from_records(
                    instruction_records,
                )
                _report = format_pilot_report(
                    course_slug=pilot_slug,
                    provider=provider,
                    counts=_counts,
                    manifest=pilot_manifest,
                    templates=_templates,
                    total_pairs=len(instruction_records),
                    chunks_processed=chunks_processed_counter,
                    chunks_total=len(iter_chunks),
                    in_flight=True,
                    capped_at_max_pairs=stats.capped_at_max_pairs,
                    max_pairs_cap=stats.max_pairs_cap,
                )
                try:
                    write_pilot_report_atomic(pilot_report_path, _report)
                except OSError as exc:
                    # Don't kill the run for a report-write failure —
                    # the JSONL is the source of truth.
                    logger.warning(
                        "Wave 117: pilot_report.md write failed: %s", exc,
                    )

        # Under concurrent generation, a stop request closes the submission
        # window and drains every already-started unit through this sole
        # ordered writer before pausing. This bounds stop latency to at most
        # ``max_concurrent`` chunks while ensuring a successful provider call
        # is never abandoned and re-paid on resume.
        if generation_map.stopped_early:
            _stop_units = (
                stats.instruction_pairs_emitted
                + stats.preference_pairs_emitted
            )
            logger.warning(
                "training_synthesis: concurrent stop drained %d submitted "
                "chunk(s), checkpointed %d pair(s), and submitted no new "
                "work after the stop boundary; pausing resumably.",
                generation_map.submitted_count,
                _stop_units,
            )
            if progress_writer is not None:
                progress_writer.update(
                    state="paused",
                    active_workers=0,
                    queued_units=0,
                    in_flight=0,
                    stop_requested=True,
                    gate_readiness="pending",
                    checkpointed=True,
                )
            stop_control.check_stop(
                "training_synthesis.concurrent_pair_loop",
                _stop_units,
                run_id=None,
            )

        # --- misconception -> DPO pair augmentation --------------------------
        # Emit one DPO pair per editorial (misconception, correction) entry
        # found on the eligible chunks. These augment the standard preference
        # pairs and are subject to the same per-artifact cap. We iterate the
        # FULL eligible-chunk set (not the post-stratification subset) so the
        # editorial signal is preserved end-to-end -- stratified sampling is
        # about template-generated pairs, not editorial misconceptions.
        if include_dpo_from_misconceptions:
            mc_index = 0
            for _, chunk in eligible_chunks:
                misconceptions = chunk.get("misconceptions") or []
                if not isinstance(misconceptions, list):
                    continue
                for mc in misconceptions:
                    if not isinstance(mc, dict):
                        continue
                    if (
                        per_artifact_cap is not None
                        and stats.preference_pairs_emitted >= per_artifact_cap
                    ):
                        stats.capped_at_max_pairs = True
                        break
                    pair = _build_misconception_dpo_pair(
                        chunk, mc, mc_index, capture=capture,
                    )
                    mc_index += 1
                    if pair is None:
                        continue
                    capture.log_decision(
                        decision_type="preference_pair_generation",
                        decision=(
                            f"Emit misconception DPO pair for chunk {pair['chunk_id']} "
                            f"(misconception_id={pair['misconception_id']})."
                        ),
                        rationale=(
                            "Editorial misconception/correction pair from "
                            "chunk.misconceptions converted directly into a DPO "
                            "preference pair: chosen=correction, rejected=misconception. "
                            "These are the highest-fidelity preference signal in the "
                            "corpus because the alternatives were authored by the "
                            "course designer, not template-synthesized."
                        ),
                    )
                    pair["decision_capture_id"] = _last_event_id(capture)
                    if _attach_source_grounding(pair, chunk):
                        stats.source_grounded_pairs += 1
                    preference_records.append(pair)
                    stats.preference_pairs_emitted += 1
                    stats.misconception_dpo_pairs_emitted += 1
                    # mirror to .in_progress sidecar.
                    _utils_append_jsonl(pref_progress_fh, pair)

        # --- reject-mined DPO negatives -------------------------------------
        # Placement is load-bearing: AFTER the misconception block so mined
        # rows join the same preference buffer, but BEFORE the record sort
        # (whose key subscripts ``r["chunk_id"]`` unguarded), BEFORE the
        # mandatory gold-set decontamination, and BEFORE the JSONL write.
        _mine_funnel: Dict[str, int] = {}
        _mine_skipped: Optional[str] = None
        if _mine_mode != MINE_MODE_OFF:
            if generation_map.stopped_early or _budget_exhausted_exc is not None:
                # A partial pool is a BIASED pool: an interrupted pass has
                # rejects only for the chunks it reached, so mining it would
                # skew the negatives toward the front of the corpus. Skip
                # loudly rather than emitting a lopsided preference set.
                _mine_skipped = "incomplete_pass"
                logger.warning(
                    "Reject mining skipped (%s): the generation pass did not "
                    "complete, so the reject pool is partial and a partial "
                    "pool is a biased pool.",
                    _mine_skipped,
                )
            else:
                _mine_rows, _mine_funnel_obj = select_mined_pairs(
                    _reject_pool.candidates(),
                    instruction_records,
                    persona=(
                        pilot_manifest.learner_persona
                        if pilot_manifest is not None
                        else _default_learner_persona()
                    ),
                    mode=_mine_mode,
                    capture=capture,
                    event_id_fn=_last_event_id,
                    min_support=resolve_min_support(),
                    min_fail_entailment=resolve_min_fail_entailment(),
                    max_skeleton_freq=resolve_max_skeleton_freq(),
                    max_fraction=resolve_max_fraction(),
                    emitted_pref_prompts=emitted_pref_prompts,
                )
                _mine_funnel = _mine_funnel_obj.as_dict()
                for _mined_pair in _mine_rows:
                    # Mined rows honour --max-pairs exactly like every other
                    # emitted preference pair; a per-artifact cap the operator
                    # set must not be silently exceeded by a derived row.
                    if (
                        per_artifact_cap is not None
                        and stats.preference_pairs_emitted >= per_artifact_cap
                    ):
                        stats.capped_at_max_pairs = True
                        break
                    preference_records.append(_mined_pair)
                    stats.preference_pairs_emitted += 1
                    _utils_append_jsonl(pref_progress_fh, _mined_pair)
                if _mine_rows:
                    logger.info(
                        "Reject mining emitted %d DPO negative(s) with "
                        "source=%s.",
                        len(_mine_rows), MINED_PAIR_SOURCE,
                    )

        # surface budget telemetry on stats whether
        # the loop completed normally OR hit the dispatch cap.
        if paraphrase_provider is not None and hasattr(paraphrase_provider, "budget"):
            bsum = paraphrase_provider.budget.summary()
            stats.dispatched_count = int(bsum.get("dispatched", 0))
            stats.cache_hits_count = int(bsum.get("cache_hits", 0))

        if _budget_exhausted_exc is not None:
            stats.capped_at_max_dispatches = True
            stats.dispatched_count = _budget_exhausted_exc.dispatched
            stats.cache_hits_count = _budget_exhausted_exc.cache_hits
            progress_payload = {
                "dispatched": _budget_exhausted_exc.dispatched,
                "cache_hits": _budget_exhausted_exc.cache_hits,
                "max_dispatches": _budget_exhausted_exc.max_dispatches,
                "instruction_pairs_emitted": stats.instruction_pairs_emitted,
                "preference_pairs_emitted": stats.preference_pairs_emitted,
                "message": (
                    f"Hit max_dispatches={_budget_exhausted_exc.max_dispatches} "
                    f"after {_budget_exhausted_exc.dispatched} dispatches + "
                    f"{_budget_exhausted_exc.cache_hits} cache hits. Re-run "
                    f"with a higher --max-dispatches to resume from the "
                    f"cache; cached calls cost zero new dispatches."
                ),
            }
            progress_path = training_specs_dir / "pilot_progress.json"
            progress_path.write_text(
                json.dumps(progress_payload, indent=2), encoding="utf-8",
            )
            logger.warning(
                "run_synthesis hit max_dispatches=%s; wrote progress to %s",
                _budget_exhausted_exc.max_dispatches, progress_path,
            )
            # preserve sidecars on budget-exceeded so the
            # operator can inspect partial output and re-run with a
            # higher cap to resume from the cache.
            logger.warning(
                "Wave 116: synthesis stopped early; sidecars preserved at "
                "%s and %s",
                instruction_progress, preference_progress,
            )

        # finalize pilot_report.md (in_flight=False) on every
        # exit path (normal completion OR budget-cap break). The JSONL
        # is the source of truth; this is the human-readable companion
        # artifact.
        if pilot_manifest is not None:
            from Trainforge.scripts.maintenance.pilot_report_helpers import (
                count_property_coverage_from_records,
                format_pilot_report,
                template_distribution_from_records,
                write_pilot_report_atomic,
            )
            _final_counts = count_property_coverage_from_records(
                instruction_records, pilot_manifest,
            )
            _final_templates = template_distribution_from_records(
                instruction_records,
            )
            _final_report = format_pilot_report(
                course_slug=pilot_slug,
                provider=provider,
                counts=_final_counts,
                manifest=pilot_manifest,
                templates=_final_templates,
                total_pairs=len(instruction_records),
                chunks_processed=chunks_processed_counter,
                chunks_total=len(iter_chunks),
                in_flight=False,
                capped_at_max_pairs=stats.capped_at_max_pairs,
                max_pairs_cap=stats.max_pairs_cap,
            )
            try:
                write_pilot_report_atomic(pilot_report_path, _final_report)
            except OSError as exc:
                logger.warning(
                    "Wave 117: pilot_report.md final write failed: %s", exc,
                )

        # --- Persist artifacts ------------------------------------------------
        # Default ordering: by chunk_id (deterministic, byte-stable across runs).
        # --difficulty-curriculum overrides with a foundational -> advanced
        # rank, with chunk_id as the tiebreaker so byte-stability under same
        # seed is preserved.
        if difficulty_curriculum:
            chunk_diff_lookup = {
                str(c.get("id") or c.get("chunk_id") or ""): c
                for _, c in eligible_chunks
            }

            def _curriculum_record_key(rec: Dict[str, Any]) -> Tuple[int, str, int]:
                cid = str(rec.get("chunk_id") or "")
                src_chunk = chunk_diff_lookup.get(cid)
                if src_chunk is None:
                    rank = len(_DIFFICULTY_ORDER)
                else:
                    rank, _ = _curriculum_sort_key(src_chunk)
                return (rank, cid, int(rec.get("seed", 0)))

            instruction_records.sort(key=_curriculum_record_key)
            preference_records.sort(key=_curriculum_record_key)
        else:
            instruction_records.sort(key=lambda r: (r["chunk_id"], r.get("seed", 0)))
            preference_records.sort(key=lambda r: (r["chunk_id"], r.get("seed", 0)))

        # ------------------------------------------------------------------
        # prerequisite-aware curriculum reordering + recap
        # ------------------------------------------------------------------
        # Runs AFTER the difficulty-curriculum pass so the topo order wins
        # when both flags are set (the prerequisite graph encodes more
        # information than the difficulty tier label alone).
        manifest_doc: Optional[Dict[str, Any]] = None
        if curriculum_ctx is not None and curriculum_from_graph:
            (
                instruction_records,
                inst_pairs_by_pos,
                inst_concepts_no_pairs,
                inst_no_concept,
            ) = order_pairs_by_curriculum(
                instruction_records,
                chunks_by_id,
                curriculum_ctx.topo,
                curriculum_ctx.concept_lookup,
            )
            (
                preference_records,
                pref_pairs_by_pos,
                pref_concepts_no_pairs,
                pref_no_concept,
            ) = order_pairs_by_curriculum(
                preference_records,
                chunks_by_id,
                curriculum_ctx.topo,
                curriculum_ctx.concept_lookup,
            )
            # Merge per-artifact manifests: a concept reports pairs from
            # BOTH instruction and preference outputs.
            merged_pairs_by_position: Dict[str, List[Dict[str, Any]]] = {}
            for src in (inst_pairs_by_pos, pref_pairs_by_pos):
                for cid, items in src.items():
                    merged_pairs_by_position.setdefault(cid, []).extend(items)
            concepts_with_pairs = set(merged_pairs_by_position.keys())
            concepts_without_pairs = [
                cid for cid in curriculum_ctx.topo.order
                if cid not in concepts_with_pairs
            ]
            stats.cycles_broken_count = len(curriculum_ctx.topo.cycles_broken)
            stats.pairs_without_concepts = inst_no_concept + pref_no_concept
            stats.concepts_without_pairs_count = len(concepts_without_pairs)
            manifest_slug = slug or corpus_dir.name
            manifest_doc = build_curriculum_manifest(
                slug=manifest_slug,
                topo=curriculum_ctx.topo,
                pairs_by_concept_position=merged_pairs_by_position,
                concepts_without_pairs=concepts_without_pairs,
                pairs_without_concepts=stats.pairs_without_concepts,
            )
            capture.log_decision(
                decision_type="instruction_pair_synthesis",
                decision=(
                    f"Curriculum ordering applied via pedagogy_graph: "
                    f"{len(curriculum_ctx.topo.order)} concepts in topo order, "
                    f"{stats.cycles_broken_count} cycles broken."
                ),
                rationale=(
                    "Prerequisite-aware emit order anchors each pair at the "
                    "latest concept its chunk references. Pairs without "
                    "concept tags fall to the end so a learner sees graph-"
                    "anchored material first; cycle-break rule is "
                    "(first_seen_week, concept_id) ascending so the order "
                    "is stable across runs."
                ),
                context=(
                    f"pairs_without_concepts={stats.pairs_without_concepts}; "
                    f"concepts_without_pairs={stats.concepts_without_pairs_count}; "
                    f"prereq_windowed={prereq_windowed}; "
                    f"context_tokens={prereq_context_tokens}"
                ),
            )

        # Apply --prereq-windowed AFTER ordering so the recap reflects the
        # final emit shape. We mutate the prompt field in place (both
        # instruction and preference pair records use ``prompt``).
        # skip pairs from the four deterministic generators —
        # they were hoisted above the chunk loop, but their fixture-based
        # prompts (especially violation-detection's bounded-length TTL
        # graphs) shouldn't get prereq context prepended; doing so can
        # push a violation prompt past its 400-char schema limit. The prefix
        # tuple is imported from lib.ontology.template_prefixes — the single
        # source of truth shared with the curie_preservation validator.
        if curriculum_ctx is not None and prereq_windowed:
            for rec in instruction_records:
                if str(rec.get("template_id", "")).startswith(
                    _DETERMINISTIC_TEMPLATE_PREFIXES
                ):
                    continue
                recap = build_prereq_recap(
                    rec,
                    chunks_by_id,
                    curriculum_ctx.concept_lookup,
                    curriculum_ctx.predecessors,
                    curriculum_ctx.first_seen_chunk,
                    context_tokens=prereq_context_tokens,
                    label_lookup=curriculum_ctx.label_lookup,
                )
                if recap:
                    rec["prereq_recap"] = recap
                    original = rec.get("prompt", "")
                    rec["prompt"] = recap + "\n\n" + original
                    stats.pairs_with_prereq_recap += 1
            for rec in preference_records:
                if str(rec.get("template_id", "")).startswith(
                    _DETERMINISTIC_TEMPLATE_PREFIXES
                ):
                    continue
                recap = build_prereq_recap(
                    rec,
                    chunks_by_id,
                    curriculum_ctx.concept_lookup,
                    curriculum_ctx.predecessors,
                    curriculum_ctx.first_seen_chunk,
                    context_tokens=prereq_context_tokens,
                    label_lookup=curriculum_ctx.label_lookup,
                )
                if recap:
                    rec["prereq_recap"] = recap
                    original = rec.get("prompt", "")
                    rec["prompt"] = recap + "\n\n" + original
                    stats.pairs_with_prereq_recap += 1

        # deterministic generators (kg_metadata,
        # violation_detection, abstention, schema_translation) used to
        # run here, post-chunk-loop. They were hoisted to fire BEFORE
        # the chunk loop so an operator `tail -f`-ing the
        # .jsonl.in_progress sidecar can verify all `--with-*` flags
        # wired through within the first ~minute, instead of waiting
        # for the multi-hour paraphrase loop to finish. See the hoisted
        # blocks just above the `for idx, chunk in iter_chunks:` line.

        # SFT data program S3: layered gold-set decontamination pre-train
        # gate. Freeze/load the course's retrieval_eval gold_set.json and
        # screen EVERY emitted pair (exact -> sliding 8-gram -> optional
        # embedding top-k -> optional paraphrase hook). Hits are dropped +
        # quarantined; survivors carry decontam_checked=true. Fully offline
        # by default (layers 1-2 only) — the embedding/paraphrase seams are
        # injectable and off unless supplied. No-op (no drops) when the
        # course has no gold set, but survivors are still stamped so the
        # provenance field is honest.
        try:
            from Trainforge.generators.postprocessing.pair_decontamination import (
                decontaminate_pairs,
                load_gold_questions,
            )
            _gold_questions, _ = load_gold_questions(corpus_dir)
            instruction_records, _quarantined_inst = decontaminate_pairs(
                instruction_records, _gold_questions, capture=capture,
            )
            # ``capture=capture`` matches the instruction arm: a mined
            # ``rejected`` side is exactly the row that most needs a drop
            # trail, and the default text projection already screens
            # prompt + completion + chosen + rejected (see
            # every pair field, so a gold-set leak on the
            # rejected side is caught, not just on chosen.
            preference_records, _quarantined_pref = decontaminate_pairs(
                preference_records, _gold_questions, capture=capture,
            )
            _quarantined_total = len(_quarantined_inst) + len(_quarantined_pref)
            stats.decontam_quarantined = _quarantined_total
            if _quarantined_total:
                # Recount emitted totals so downstream stats reflect the drop.
                stats.instruction_pairs_emitted = max(
                    0, stats.instruction_pairs_emitted - len(_quarantined_inst),
                )
                stats.preference_pairs_emitted = max(
                    0, stats.preference_pairs_emitted - len(_quarantined_pref),
                )
                _quar_path = training_specs_dir / "decontam_quarantine.jsonl"
                try:
                    _write_jsonl(
                        _quar_path, _quarantined_inst + _quarantined_pref,
                    )
                except Exception as _q_exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "S3 decontam: quarantine sidecar emit failed: %s", _q_exc,
                    )
                logger.warning(
                    "SFT data program S3: gold-set decontamination quarantined "
                    "%d pair(s) (instruction=%d, preference=%d) against %d gold "
                    "questions.",
                    _quarantined_total, len(_quarantined_inst),
                    len(_quarantined_pref), len(_gold_questions),
                )
        except Exception as _decontam_exc:  # noqa: BLE001 — never abort emit
            logger.warning(
                "SFT data program S3: decontamination pass skipped (%s).",
                _decontam_exc,
            )

        _write_jsonl(instruction_out, instruction_records)
        _write_jsonl(preference_out, preference_records)
        _update_dataset_config(
            dataset_config_path, stats, holdout_identity=holdout_identity,
        )
        if _mine_mode != MINE_MODE_OFF:
            # Capture + selection funnel onto the returned SynthesisStats.
            # Assigned BEFORE _emit_synthesis_summary_sidecar below, which
            # copies it into synthesis_summary.json as the OPTIONAL
            # ``reject_mining`` block -- the only durable surface a shadow run
            # has. Left absent entirely when mining is off, so the sidecar is
            # byte-identical on a legacy run.
            stats.reject_mining = {
                "mode_shadow": int(_mine_mode != MINE_MODE_ON),
                **_reject_pool.counters(),
                **_mine_funnel,
            }
            if _mine_skipped:
                # A string in an otherwise int-valued dict, deliberately: the
                # reason a pass mined nothing is not expressible as a count,
                # and silently reporting all-zero counters would read as "no
                # candidates" rather than "not evaluated".
                stats.reject_mining["reject_mining_skipped"] = _mine_skipped
            logger.info(
                "Reject mining capture funnel: %s", stats.reject_mining,
            )
        # W3.H sub-task H5: emit ``training_specs/synthesis_summary.json``
        # sidecar carrying the canonical source_coverage block. items_consumed
        # = chunks_eligible (the upstream item denominator); pairs_emitted
        # = instruction_pairs_emitted + preference_pairs_emitted +
        # misconception_dpo_pairs_emitted + the four deterministic
        # generators (kg_metadata / violation / abstention /
        # schema_translation). Drop reasons pull from
        # SynthesisStats.promotion_rejection_reasons (W2.E
        # TrainingPairPromotionValidator counts) and from the legacy
        # rejected_reasons histogram. Best-effort write — failure logs
        # but doesn't abort the run (the JSONL artifacts are the
        # source of truth).
        try:
            _emit_synthesis_summary_sidecar(
                training_specs_dir / "synthesis_summary.json",
                stats=stats,
                provider=provider,
                course_code=course_code,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "W3.H H5: synthesis_summary sidecar emit failed (non-fatal): %s",
                exc,
            )
        if manifest_doc is not None:
            manifest_path = training_specs_dir / "curriculum_manifest.json"
            _tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            with _tmp.open("w", encoding="utf-8") as fh:
                json.dump(manifest_doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
            _tmp.replace(manifest_path)

        # Log a stage-complete decision so the summary lives alongside the per-pair events.
        capture.log_decision(
            decision_type="instruction_pair_synthesis",
            decision=(
                f"Completed synthesis: {stats.instruction_pairs_emitted} instruction pairs, "
                f"{stats.preference_pairs_emitted} preference pairs from "
                f"{stats.chunks_eligible}/{stats.chunks_total} eligible chunks."
            ),
            rationale=(
                f"Artifacts written to {instruction_out.name} and {preference_out.name}. "
                f"Rejected counts: instruction={stats.instruction_pairs_rejected}, "
                f"preference={stats.preference_pairs_rejected}. "
                f"dataset_config.json updated with statistics.instruction_pairs and "
                f"statistics.preference_pairs."
            ),
        )

        # try-body completed without raising. Mark the run
        # clean so the finally-block deletes the sidecars. A
        # SynthesisBudgetExceeded run reaches this line too (it is
        # caught above and produces ``pilot_progress.json``), so we
        # additionally check ``_budget_exhausted_exc`` in the finally
        # to keep the sidecars on cap-exhausted runs.
        clean_exit = True
        if progress_writer is not None:
            progress_writer.update(
                state="complete",
                active_workers=0,
                queued_units=0,
                in_flight=0,
                stop_requested=False,
                gate_readiness="ready",
                checkpointed=True,
            )

    finally:
        # A consumer-side exception occurs while the ordered iterator is
        # suspended at ``yield``. Explicitly closing it first enters
        # BoundedOrderedMap's unwind path, cancels not-started futures, and
        # waits for every already-running worker to finish its durable journal
        # append. No terminal snapshot may be written before this barrier.
        if generation_iterator is not None:
            close_iterator = getattr(generation_iterator, "close", None)
            if callable(close_iterator):
                close_iterator()
        # Contract context must outlive every joined worker and its final
        # DecisionCapture/usage append.
        if run_contract_env_set:
            from Trainforge.synthesis.synthesis_contract_guard import clear_contract_files
            clear_contract_files()
            if prior_run_contract_env is None:
                os.environ.pop(
                    "TRAINFORGE_SYNTHESIS_RUN_CONTRACT_SHA256", None,
                )
            else:
                os.environ["TRAINFORGE_SYNTHESIS_RUN_CONTRACT_SHA256"] = (
                    prior_run_contract_env
                )
            if prior_component_contract_env is None:
                os.environ.pop(
                    "TRAINFORGE_SYNTHESIS_CONTRACT_COMPONENTS_SHA256", None,
                )
            else:
                os.environ[
                    "TRAINFORGE_SYNTHESIS_CONTRACT_COMPONENTS_SHA256"
                ] = prior_component_contract_env
        if progress_writer is not None and not clean_exit:
            # A graceful concurrent stop stamps PAUSED before raising. Preserve
            # that state; every other exception is a loud failed snapshot.
            current = progress_writer.payload.get("state")
            journal_counts = summarize_generation_journal(
                generation_checkpoint_path
            )
            terminal_state = "paused" if current == "paused" else "failed"
            progress_writer.update(
                state=terminal_state,
                transient_count=journal_counts["transient_pending_units"],
                provider_results=journal_counts["provider_results"],
                transient_attempts=journal_counts["transient_attempts"],
                recovered_units=journal_counts["recovered_units"],
                exhausted_units=journal_counts["exhausted_units"],
                fatal_units=journal_counts["fatal_units"],
                active_workers=0,
                queued_units=0,
                in_flight=0,
                stop_requested=stop_control.stop_requested(),
                gate_readiness=(
                    "pending" if terminal_state == "paused" else "blocked"
                ),
                checkpointed=True,
            )
        # always close sidecar file handles, even on
        # exception. Delete only on a fully clean exit (no exception
        # propagated AND no budget cap hit). On budget-exceeded or
        # any other early exit, the sidecars stay on disk so the
        # operator has inspectable partial output.
        if inst_progress_fh is not None:
            try:
                inst_progress_fh.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Failed to close instruction sidecar: %s", e)
        if pref_progress_fh is not None:
            try:
                pref_progress_fh.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Failed to close preference sidecar: %s", e)
        # close the resume checkpoint handle, mirroring the
        # existing Wave-116 sidecar close logic. The append handle
        # always closes; the file itself is unlinked only on a
        # fully-clean exit so a SynthesisBudgetExceeded / unexpected
        # exception leaves the checkpoint on disk for the next run to
        # resume from.
        if checkpoint_fh is not None:
            try:
                checkpoint_fh.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to close synthesis checkpoint sidecar: %s", e,
                )
        if clean_exit and _budget_exhausted_exc is None:
            instruction_progress.unlink(missing_ok=True)
            preference_progress.unlink(missing_ok=True)
            if checkpoint_path is not None and checkpoint_path.exists():
                # Accepted records now live in the canonical output JSONL.
                # Preserve quality rejections and deterministic pre-dispatch
                # exclusions as distinct compact diagnostics.
                terminal_dispositions = _project_terminal_dispositions(
                    _load_synthesis_pairs_checkpoint(checkpoint_path).values()
                )
                _write_jsonl(
                    training_specs_dir / "synthesis_dispositions.jsonl",
                    terminal_dispositions,
                )
                checkpoint_path.unlink()
            if (
                generation_checkpoint_path is not None
                and generation_checkpoint_path.exists()
            ):
                generation_checkpoint_path.unlink()

        if owns_capture:
            try:
                capture.save()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Failed to save decision capture: %s", e)

    return stats


# ---------------------------------------------------------------------------
# LibV2-archive entry path
# ---------------------------------------------------------------------------

def run_synthesis_from_libv2(
    slug: str,
    course_code: Optional[str] = None,
    *,
    libv2_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    provider: str = "mock",
    seed: int = DEFAULT_SEED,
    stratify: Optional[Sequence[str]] = None,
    include_dpo_from_misconceptions: bool = False,
    difficulty_curriculum: bool = False,
    max_pairs: Optional[int] = None,
    curriculum_from_graph: bool = False,
    prereq_windowed: bool = False,
    prereq_context_tokens: int = DEFAULT_PREREQ_CONTEXT_TOKENS,
    pedagogy_graph_path: Optional[Path] = None,
    instruction_variants_per_chunk: int = 1,
    pilot_report_every: int = 20,
    smoke_mode: str = "none",
    with_kg_metadata: bool = False,
    kg_metadata_max_pairs: int = 2000,
    with_violation_detection: bool = False,
    violation_shapes_glob: Optional[str] = None,
    violation_detection_max_pairs: Optional[int] = None,
    with_abstention: bool = False,
    abstention_max_pairs: int = 1000,
    with_schema_translation: bool = False,
    schema_translation_max_pairs: int = 50,
    with_assessment_sft: bool = False,
    assessment_sft_max_pairs: Optional[int] = None,
    with_graph_sft: bool = False,
    graph_sft_max_pairs: Optional[int] = None,
    holdout_split_path: Optional[Path] = None,
    synthesis_pairs_checkpoint_path: Optional[Path] = None,
    staging_dir: Optional[Path] = None,
    max_concurrent: Optional[int] = None,
) -> SynthesisStats:
    """Run synthesis directly against a LibV2 course archive.

    Locates the course directory under ``LibV2/courses/<slug>/``, which
    already contains ``corpus/chunks.jsonl`` (the same shape the Trainforge
    pipeline emits) and ``objectives.json``. This avoids re-running the
    pipeline when the only goal is to (re-)synthesize training pairs from
    an already-archived corpus -- e.g. when iterating on stratification or
    misconception-DPO emission.

    Args:
        slug: LibV2 course slug, e.g. ``"<course-slug>"``.
        course_code: Course code for decision capture. Defaults to the
            ``course_code`` field on objectives.json, or the slug uppercased
            with hyphens replaced by underscores.
        libv2_root: Override for ``LibV2/courses/`` (testing).
        output_dir: Where to write ``instruction_pairs.jsonl`` /
            ``preference_pairs.jsonl``. Defaults to
            ``<archive>/training_specs/`` (overwriting the on-disk pairs).
        provider, seed, stratify, include_dpo_from_misconceptions,
            difficulty_curriculum, max_pairs: Forwarded to
            :func:`run_synthesis`.

    Returns:
        Same :class:`SynthesisStats` payload as the pipeline-based entry.
    """
    archive_dir = _resolve_libv2_corpus_dir(slug, libv2_root=libv2_root)

    if course_code is None:
        objectives_path = archive_dir / "objectives.json"
        if objectives_path.exists():
            try:
                with objectives_path.open("r", encoding="utf-8") as fh:
                    obj_data = json.load(fh)
                    course_code = str(obj_data.get("course_code") or "").strip()
            except (OSError, ValueError):
                course_code = ""
        if not course_code:
            course_code = slug.upper().replace("-", "_")

    return run_synthesis(
        corpus_dir=archive_dir,
        course_code=course_code,
        provider=provider,
        seed=seed,
        stratify=stratify,
        include_dpo_from_misconceptions=include_dpo_from_misconceptions,
        difficulty_curriculum=difficulty_curriculum,
        max_pairs=max_pairs,
        output_dir=output_dir,
        curriculum_from_graph=curriculum_from_graph,
        prereq_windowed=prereq_windowed,
        prereq_context_tokens=prereq_context_tokens,
        pedagogy_graph_path=pedagogy_graph_path,
        slug=slug,
        instruction_variants_per_chunk=instruction_variants_per_chunk,
        pilot_report_every=pilot_report_every,
        smoke_mode=smoke_mode,
        with_kg_metadata=with_kg_metadata,
        kg_metadata_max_pairs=kg_metadata_max_pairs,
        with_violation_detection=with_violation_detection,
        violation_shapes_glob=violation_shapes_glob,
        violation_detection_max_pairs=violation_detection_max_pairs,
        with_abstention=with_abstention,
        abstention_max_pairs=abstention_max_pairs,
        with_schema_translation=with_schema_translation,
        schema_translation_max_pairs=schema_translation_max_pairs,
        with_assessment_sft=with_assessment_sft,
        assessment_sft_max_pairs=assessment_sft_max_pairs,
        with_graph_sft=with_graph_sft,
        graph_sft_max_pairs=graph_sft_max_pairs,
        holdout_split_path=holdout_split_path,
        synthesis_pairs_checkpoint_path=synthesis_pairs_checkpoint_path,
        staging_dir=staging_dir,
        max_concurrent=max_concurrent,
    )


# ---------------------------------------------------------------------------
# CLI (standalone invocation)
# ---------------------------------------------------------------------------

def _parse_stratify_arg(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [d.strip().lower() for d in raw.split(",") if d.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Synthesize SFT and DPO training pairs from an already-processed "
            "Trainforge course output directory or LibV2 course archive."
        )
    )
    # Either --corpus (legacy Trainforge output dir) or --slug (LibV2
    # archive entry path). At least one must be provided.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--corpus",
        help="Course output directory (the one containing corpus/ and training_specs/).",
    )
    src.add_argument(
        "--slug",
        help=(
            "LibV2 course slug under LibV2/courses/<slug>/ "
            "(reads corpus/chunks.jsonl + objectives.json from the archive)."
        ),
    )
    p.add_argument(
        "--course-code",
        help=(
            "Course code for decision capture. "
            "Required when --corpus is used; optional with --slug "
            "(falls back to objectives.json:course_code)."
        ),
    )
    p.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "anthropic", "claude_session", "together", "local"],
        help=(
            "Synthesis provider. 'mock' = template factory (plumbing tests "
            "only — produces template-recognizer adapters). 'anthropic' = "
            "Anthropic SDK (requires ANTHROPIC_API_KEY). 'claude_session' = "
            "running Claude Code session via LocalDispatcher (Claude Max / "
            "no-API-key path; requires invocation through the workflow runner "
            "or MCP tool so a dispatcher is in-context). 'together' = "
            "Together AI's OpenAI-compatible chat-completions endpoint "
            "(default model meta-llama/Llama-3.3-70B-Instruct-Turbo, "
            "override via TOGETHER_SYNTHESIS_MODEL; requires TOGETHER_API_KEY). "
            "Together's ToS permits using the output as training data for "
            "another model — Anthropic's does not. 'local' = a local "
            "OpenAI-compatible model server (Ollama default "
            "http://localhost:11434/v1, override via LOCAL_SYNTHESIS_BASE_URL; "
            "default model qwen2.5:7b-instruct-q4_K_M, override via "
            "LOCAL_SYNTHESIS_MODEL). API key optional — local servers ignore "
            "auth. Zero cost per call, zero ToS exposure; tradeoff is local "
            "hardware requirement."
        ),
    )
    p.add_argument(
        "--synthesis-contract",
        choices=list(SYNTHESIS_CONTRACT_CHOICES),
        default=None,
        help=(
            "Explicit synthesis orchestration contract. micro-v1 selects "
            "ed4all.staged-synthesis-micro.v1. Resolves to the "
            "TRAINFORGE_STAGED_SYNTHESIS_V4 / "
            "TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1 pair for the process before "
            "any provider is constructed; an ambient value that DISAGREES "
            "with the selection is a loud failure, never a silent override. "
            "Omit to preserve the historical environment-driven "
            "legacy/staged-v4 path byte-for-byte."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base deterministic seed (default: {DEFAULT_SEED}).",
    )
    p.add_argument(
        "--max-dispatches",
        type=int,
        default=None,
        help=(
            "Wave 110 / Phase D: hard cap on Claude-Max session dispatches "
            "(claude_session provider only). When the cap is hit, raises "
            "SynthesisBudgetExceeded; partial output is preserved in "
            "<corpus>/training_specs/.synthesis_cache.jsonl and the next "
            "run resumes for free."
        ),
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=(
            "Bounded immutable generation workers (1-48). Unset resolves "
            "TRAINFORGE_SYNTHESIS_MAX_CONCURRENT, whose legacy default is 1. "
            "Results are committed by one deterministic source-order writer."
        ),
    )
    p.add_argument(
        "--pilot-report-every",
        type=int,
        default=20,
        help=(
            "Wave 117: regenerate training_specs/pilot_report.md every N "
            "processed chunks during the run, so the operator has live "
            "property-coverage / template-distribution visibility. Set "
            "to 0 to disable. No-op when the course has no property "
            "manifest. Default: 20 chunks."
        ),
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        default=False,
        help=(
            "Worker A: opt OUT of the per-pair resume checkpoint sidecar "
            "(default: checkpoint enabled). The sidecar at "
            "<training_specs>/.synthesis_pairs_checkpoint.jsonl records "
            "every accepted instruction / preference pair and is unlinked "
            "on a clean run. A subsequent run on the same corpus reads "
            "the sidecar and skips the LLM dispatch for any pair already "
            "present, so a 10-hour local-LLM rebuild that crashes at "
            "hour 9 doesn't re-pay for the first 9 hours' work."
        ),
    )
    # Stratified-sampling / LibV2-archive options
    p.add_argument(
        "--stratify",
        default="",
        help=(
            "Comma-separated stratification dimensions. Choices: "
            "bloom, chunk_type, outcome, difficulty. When set, eligible "
            "chunks are sampled round-robin across the resulting buckets "
            "so the output distribution is uniform across the dimension(s)."
        ),
    )
    p.add_argument(
        "--include-dpo-from-misconceptions",
        action="store_true",
        help=(
            "Emit one DPO pair per editorial chunk.misconceptions entry "
            "(chosen=correction, rejected=misconception)."
        ),
    )
    p.add_argument(
        "--difficulty-curriculum",
        action="store_true",
        help=(
            "Order emitted pairs foundational -> intermediate -> advanced "
            "(preserved in the output JSONL) for curriculum-style training."
        ),
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=1000,
        help="Cap total emitted pairs per artifact (default: 1000).",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output directory for instruction_pairs.jsonl + "
            "preference_pairs.jsonl. Defaults to "
            "<corpus>/training_specs/."
        ),
    )
    # Curriculum mode: default-on; --no-graph to opt out.
    p.add_argument(
        "--curriculum-from-graph",
        action="store_true",
        default=True,
        help=(
            "Order emitted pairs by topological sort over pedagogy_graph "
            "prerequisite_of edges (Wave 91: default ON). Each pair anchors "
            "at the latest concept its chunk references; pairs whose chunks "
            "reference no concepts go to the end. Cycle-break: "
            "(first_seen_week, concept_id) asc."
        ),
    )
    # opt-out flag for legacy corpora that lack a
    # pedagogy graph. Without it, missing graph raises FileNotFoundError
    # at run time so the silent-degrade-to-chunk-id-order regression is
    # impossible.
    p.add_argument(
        "--no-graph",
        action="store_true",
        default=False,
        help=(
            "Opt out of the Wave-91 graph-required default. Use only for "
            "legacy corpora that have no pedagogy_graph.json on disk."
        ),
    )
    p.add_argument(
        "--prereq-windowed",
        action="store_true",
        help=(
            "Prepend a 'Prerequisites recap' block to each pair's prompt, "
            "summarising depth-1 prerequisite_of predecessors of the "
            "pair's chunk concepts. Recap is capped at "
            "--prereq-context-tokens whitespace tokens."
        ),
    )
    p.add_argument(
        "--prereq-context-tokens",
        type=int,
        default=DEFAULT_PREREQ_CONTEXT_TOKENS,
        help=(
            "Token cap for the prerequisites recap block "
            f"(default: {DEFAULT_PREREQ_CONTEXT_TOKENS}). Applied as a "
            "whitespace-token approximation."
        ),
    )
    p.add_argument(
        "--pedagogy-graph",
        default=None,
        help=(
            "Override path to pedagogy_graph.json. By default the stage "
            "looks under <corpus>/graph/, <corpus>/pedagogy/, then the "
            "corpus root."
        ),
    )
    p.add_argument(
        "--instruction-variants-per-chunk",
        type=int,
        default=1,
        help=(
            "Emit this many SFT instruction variants per eligible chunk "
            "(default: 1). Preference pairs remain one per chunk plus any "
            "editorial misconception DPO pairs."
        ),
    )
    # smoke modes. Stratified ~20-chunk sample so every
    # property surface form gets at least 3 chunks of representation;
    # writes ``smoke_pilot_report.md`` (sidecar — never overwrites
    # the canonical ``pilot_report.md``); floors scaled down so a
    # smoke run can pass when the full run would.
    smoke = p.add_mutually_exclusive_group()
    smoke.add_argument(
        "--smoke-deterministic",
        action="store_true",
        help=(
            "Wave 120: forces provider='mock', stratified-samples ~20 "
            "chunks (every property-bearing chunk first, capped at 3 per "
            "surface form, padded to 20). No LLM call — completes in "
            "<60 s. Writes training_specs/smoke_pilot_report.md with "
            "scaled floors (1 pair per property). Use to validate "
            "schema, decision capture, gate wiring before paying for "
            "a full provider run."
        ),
    )
    smoke.add_argument(
        "--smoke-paraphrase",
        action="store_true",
        help=(
            "Wave 120: like --smoke-deterministic but keeps the "
            "configured --provider so the paraphrase path (and "
            "preserve_tokens preservation) is exercised on ~20 "
            "stratified chunks. Floors scaled to 2 pairs per property. "
            "Smoke mode caps the local provider's parse-retry budget at "
            "1 (production default: 3) so the property-heavy stratified "
            "sample doesn't compound retry cost into unbounded wall "
            "time. Local-server 14B ceiling: ~20 min."
        ),
    )
    # KG-metadata + violation-detection generators.
    # Both are off by default so existing callers / corpora keep their
    # current behaviour; flip on with --with-kg-metadata /
    # --with-violation-detection to teach the adapter the literal
    # KG-membership facts and SHACL-violation reasoning the eval
    # harness probes for.
    p.add_argument(
        "--with-kg-metadata",
        dest="with_kg_metadata",
        action="store_true",
        default=False,
        help=(
            "Audit 2026-04-30 fix: append KG-metadata yes/no probes to "
            "instruction_pairs.jsonl. Reads pedagogy_graph.json and "
            "emits one positive + 1-2 negative pairs per relation type, "
            "mirroring Trainforge.eval.faithfulness._RELATION_TEMPLATES. "
            "Closes the zero-KG-metadata-recall regression."
        ),
    )
    p.add_argument(
        "--no-kg-metadata",
        dest="with_kg_metadata",
        action="store_false",
        help="Explicitly disable the KG-metadata generator (default).",
    )
    p.add_argument(
        "--kg-metadata-max-pairs",
        type=int,
        default=2000,
        help=(
            "Cap on KG-metadata pair emissions (default: 2000). "
            "Distributed evenly across relation types so a graph-rich "
            "relation doesn't crowd out low-volume ones."
        ),
    )
    p.add_argument(
        "--with-violation-detection",
        dest="with_violation_detection",
        action="store_true",
        default=False,
        help=(
            "Audit 2026-04-30 fix: append SHACL-violation-detection "
            "pairs to instruction_pairs.jsonl. Runs pyshacl over a "
            "built-in shape catalog (or course-supplied TTL files) and "
            "emits (graph, valid?, reason) SFT pairs whose labels are "
            "verified by the same engine the eval harness uses. Requires "
            "pyshacl + rdflib (already in pyproject [training] extra)."
        ),
    )
    p.add_argument(
        "--no-violation-detection",
        dest="with_violation_detection",
        action="store_false",
        help="Explicitly disable the violation-detection generator (default).",
    )
    p.add_argument(
        "--violation-detection-shapes-glob",
        dest="violation_shapes_glob",
        default=None,
        help=(
            "Optional glob pattern that points at TTL shape files to use "
            "as fixtures for the violation-detection generator (defaults "
            "to the built-in 6-shape catalog when unset). Resolved "
            "relative to the corpus_dir; absolute paths are honoured."
        ),
    )
    p.add_argument(
        "--violation-detection-max-pairs",
        dest="violation_detection_max_pairs",
        type=int,
        default=None,
        help=(
            "Wave 125a: cap on emitted SHACL violation-detection pairs. "
            "When unset (default), the entire pyshacl-validated catalog "
            "(>= 800 pairs) is appended. Set this to balance the "
            "violation-detection share of the total corpus when running "
            "production rebuilds. Choose a cap that preserves the intended "
            "family balance. "
            "Truncation is family-balanced round-robin across surface "
            "forms so every form keeps representation up to the cap."
        ),
    )
    # abstention +
    # schema-translation generators. Both are off by default, parallel
    # to --with-kg-metadata / --with-violation-detection. Closes the
    # the abstention and schema-to-English bridge gaps the eval harness probes.
    p.add_argument(
        "--with-abstention",
        dest="with_abstention",
        action="store_true",
        default=False,
        help=(
            "Wave 124 fix: append abstention probes ('the source does "
            "not establish X') to instruction_pairs.jsonl. Reads "
            "pedagogy_graph.json, samples concepts the chunk does NOT "
            "address, and emits grounded 'no, no evidence' completions. "
            "Closes the abstention regression."
        ),
    )
    p.add_argument(
        "--no-abstention",
        dest="with_abstention",
        action="store_false",
        help="Explicitly disable the abstention generator (default).",
    )
    p.add_argument(
        "--abstention-max-pairs",
        type=int,
        default=1000,
        help=(
            "Cap on abstention pair emissions (default: 1000). "
            "Distributed across chunks so a chunk-rich graph "
            "doesn't crowd the cohort onto one chunk's silent set."
        ),
    )
    p.add_argument(
        "--with-schema-translation",
        dest="with_schema_translation",
        action="store_true",
        default=False,
        help=(
            "Wave 124 fix: append schema-to-English bridge pairs to "
            "instruction_pairs.jsonl. Walks the property manifest's "
            "surface forms (e.g. sh:datatype, rdfs:subClassOf) and "
            "emits one definition pair + one usage pair per CURIE from "
            "a hand-curated table. Closes the schema-to-English bridge "
            "gap that weakens faithfulness."
        ),
    )
    p.add_argument(
        "--no-schema-translation",
        dest="with_schema_translation",
        action="store_false",
        help=(
            "Explicitly disable the schema-translation generator (default)."
        ),
    )
    p.add_argument(
        "--schema-translation-max-pairs",
        type=int,
        default=50,
        help=(
            "Cap on schema-translation pair emissions (default: 50). "
            "12 base pairs (6 surface forms * 2 variants) leaves room "
            "for future variant expansion under the same cap."
        ),
    )
    # SFT data program (S1/S5): assessment->SFT + concept-graph->SFT arms.
    # Off by default (byte-identical); also drivable via env
    # ED4ALL_WITH_ASSESSMENT_SFT / ED4ALL_WITH_GRAPH_SFT for the pipeline seam.
    p.add_argument(
        "--with-assessment-sft",
        dest="with_assessment_sft",
        action="store_true",
        default=False,
        help=(
            "SFT data program S1: append open-book assessment->SFT pairs "
            "(solve+steps / error-diagnosis / explain-why / grade-rubric / "
            "hint-no-reveal / verify-answer) from assessments.json + "
            "answer_key.json to instruction_pairs.jsonl. Deterministic, no LLM."
        ),
    )
    p.add_argument(
        "--assessment-sft-max-pairs",
        type=int,
        default=None,
        help="Optional global cap on assessment->SFT pairs (default: unlimited).",
    )
    p.add_argument(
        "--with-graph-sft",
        dest="with_graph_sft",
        action="store_true",
        default=False,
        help=(
            "SFT data program S5: append open-book concept-graph->SFT pairs "
            "(relation-QA / prereq study-path / concept verbalization) from "
            "the holdout-REDUCED concept_graph_semantic.json, consensus-"
            "filtered. Deterministic, no LLM."
        ),
    )
    p.add_argument(
        "--graph-sft-max-pairs",
        type=int,
        default=None,
        help="Optional global cap on concept-graph->SFT pairs (default: unlimited).",
    )
    return p


def main(args: Optional[argparse.Namespace] = None) -> SynthesisStats:
    if args is None:
        args = build_parser().parse_args()

    # Resolve the explicit contract selector BEFORE anything reads the two
    # staged-synthesis switches (``staged_objective_contract_enabled`` at the
    # focus seam, ``build_synthesis_provider`` at construction). Leaving this
    # unread is what made the documented `--synthesis-contract micro-v1`
    # invocation a no-op that silently ran whatever the environment said.
    try:
        apply_synthesis_contract_selection(
            getattr(args, "synthesis_contract", None)
        )
    except (SynthesisContractConflict, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    stratify_dims = _parse_stratify_arg(getattr(args, "stratify", ""))
    include_dpo = bool(getattr(args, "include_dpo_from_misconceptions", False))
    diff_curriculum = bool(getattr(args, "difficulty_curriculum", False))
    max_pairs_cap = getattr(args, "max_pairs", None)
    output_dir = Path(args.output) if getattr(args, "output", None) else None
    # graph-required by default; --no-graph opts out.
    no_graph = bool(getattr(args, "no_graph", False))
    curriculum_graph = bool(getattr(args, "curriculum_from_graph", True))
    if no_graph:
        curriculum_graph = False
    prereq_windowed = bool(getattr(args, "prereq_windowed", False))
    prereq_ctx_tokens = int(
        getattr(args, "prereq_context_tokens", DEFAULT_PREREQ_CONTEXT_TOKENS)
    )
    pedagogy_path = Path(args.pedagogy_graph) if getattr(args, "pedagogy_graph", None) else None
    # --max-dispatches is meaningful only with claude_session.
    max_dispatches = getattr(args, "max_dispatches", None)
    if max_dispatches is not None and args.provider != "claude_session":
        raise SystemExit(
            "--max-dispatches is only meaningful with --provider claude_session"
        )
    # incremental pilot_report.md writes during the chunk loop.
    pilot_report_every = int(getattr(args, "pilot_report_every", 20) or 0)
    # smoke modes (mutex group, only one can be set).
    if getattr(args, "smoke_deterministic", False):
        smoke_mode = "deterministic"
    elif getattr(args, "smoke_paraphrase", False):
        smoke_mode = "paraphrase"
    else:
        smoke_mode = "none"

    # KG-metadata + violation-detection generators.
    with_kg_metadata = bool(getattr(args, "with_kg_metadata", False))
    kg_metadata_max_pairs = int(getattr(args, "kg_metadata_max_pairs", 2000))
    with_violation_detection = bool(
        getattr(args, "with_violation_detection", False)
    )
    violation_shapes_glob = getattr(args, "violation_shapes_glob", None)
    # optional cap on violation-detection emit count.
    violation_detection_max_pairs_arg = getattr(
        args, "violation_detection_max_pairs", None,
    )
    violation_detection_max_pairs = (
        int(violation_detection_max_pairs_arg)
        if violation_detection_max_pairs_arg is not None
        else None
    )
    # abstention + schema-translation generators.
    with_abstention = bool(getattr(args, "with_abstention", False))
    abstention_max_pairs = int(getattr(args, "abstention_max_pairs", 1000))
    with_schema_translation = bool(
        getattr(args, "with_schema_translation", False)
    )
    schema_translation_max_pairs = int(
        getattr(args, "schema_translation_max_pairs", 50)
    )
    # SFT data program (S1/S5): assessment->SFT + concept-graph->SFT arms.
    with_assessment_sft = bool(getattr(args, "with_assessment_sft", False))
    assessment_sft_max_pairs = getattr(args, "assessment_sft_max_pairs", None)
    with_graph_sft = bool(getattr(args, "with_graph_sft", False))
    graph_sft_max_pairs = getattr(args, "graph_sft_max_pairs", None)

    # --no-checkpoint opts out of the per-pair resume cache.
    # Default behaviour (flag absent) leaves
    # ``synthesis_pairs_checkpoint_path=None`` so ``run_synthesis``
    # auto-derives ``<training_specs>/.synthesis_pairs_checkpoint.jsonl``.
    # Disable by passing a Path sentinel that ``run_synthesis``
    # recognises and treats as "no checkpoint" (see the matching
    # _CHECKPOINT_DISABLE_SENTINEL block in run_synthesis).
    no_checkpoint = bool(getattr(args, "no_checkpoint", False))
    checkpoint_arg: Optional[Path] = (
        Path("<disable-synthesis-checkpoint>") if no_checkpoint else None
    )

    if getattr(args, "slug", None):
        stats = run_synthesis_from_libv2(
            slug=args.slug,
            course_code=args.course_code,
            provider=args.provider,
            seed=args.seed,
            smoke_mode=smoke_mode,
            stratify=stratify_dims,
            include_dpo_from_misconceptions=include_dpo,
            difficulty_curriculum=diff_curriculum,
            max_pairs=max_pairs_cap,
            output_dir=output_dir,
            curriculum_from_graph=curriculum_graph,
            prereq_windowed=prereq_windowed,
            prereq_context_tokens=prereq_ctx_tokens,
            pedagogy_graph_path=pedagogy_path,
            instruction_variants_per_chunk=args.instruction_variants_per_chunk,
            pilot_report_every=pilot_report_every,
            with_kg_metadata=with_kg_metadata,
            kg_metadata_max_pairs=kg_metadata_max_pairs,
            with_violation_detection=with_violation_detection,
            violation_shapes_glob=violation_shapes_glob,
            violation_detection_max_pairs=violation_detection_max_pairs,
            with_abstention=with_abstention,
            abstention_max_pairs=abstention_max_pairs,
            with_schema_translation=with_schema_translation,
            schema_translation_max_pairs=schema_translation_max_pairs,
            with_assessment_sft=with_assessment_sft,
            assessment_sft_max_pairs=assessment_sft_max_pairs,
            with_graph_sft=with_graph_sft,
            graph_sft_max_pairs=graph_sft_max_pairs,
            synthesis_pairs_checkpoint_path=checkpoint_arg,
            max_concurrent=args.max_concurrent,
        )
    else:
        if not args.course_code:
            raise SystemExit(
                "--course-code is required when --corpus is used "
                "(only optional with --slug)."
            )
        stats = run_synthesis(
            corpus_dir=Path(args.corpus),
            course_code=args.course_code,
            provider=args.provider,
            seed=args.seed,
            stratify=stratify_dims,
            include_dpo_from_misconceptions=include_dpo,
            difficulty_curriculum=diff_curriculum,
            max_pairs=max_pairs_cap,
            output_dir=output_dir,
            curriculum_from_graph=curriculum_graph,
            prereq_windowed=prereq_windowed,
            prereq_context_tokens=prereq_ctx_tokens,
            pedagogy_graph_path=pedagogy_path,
            instruction_variants_per_chunk=args.instruction_variants_per_chunk,
            max_dispatches=max_dispatches,
            pilot_report_every=pilot_report_every,
            smoke_mode=smoke_mode,
            with_kg_metadata=with_kg_metadata,
            kg_metadata_max_pairs=kg_metadata_max_pairs,
            with_violation_detection=with_violation_detection,
            violation_shapes_glob=violation_shapes_glob,
            violation_detection_max_pairs=violation_detection_max_pairs,
            with_abstention=with_abstention,
            abstention_max_pairs=abstention_max_pairs,
            with_schema_translation=with_schema_translation,
            schema_translation_max_pairs=schema_translation_max_pairs,
            with_assessment_sft=with_assessment_sft,
            assessment_sft_max_pairs=assessment_sft_max_pairs,
            with_graph_sft=with_graph_sft,
            graph_sft_max_pairs=graph_sft_max_pairs,
            synthesis_pairs_checkpoint_path=checkpoint_arg,
            max_concurrent=args.max_concurrent,
        )

    print("\n[Synthesis] Complete.")
    print(f"  Chunks eligible:    {stats.chunks_eligible}/{stats.chunks_total}")
    print(f"  Instruction pairs:  {stats.instruction_pairs_emitted} "
          f"(rejected {stats.instruction_pairs_rejected})")
    print(f"  Preference pairs:   {stats.preference_pairs_emitted} "
          f"(rejected {stats.preference_pairs_rejected})")
    if stats.rejected_reasons:
        print("  Rejected reasons:")
        for reason, count in sorted(stats.rejected_reasons.items()):
            print(f"    {reason}: {count}")
    # surface session-budget telemetry on
    # claude_session runs. Counts are 0 for non-session providers.
    if stats.dispatched_count or stats.cache_hits_count:
        print(
            f"  Session budget:     dispatched={stats.dispatched_count}, "
            f"cache_hits={stats.cache_hits_count}"
        )
    if stats.capped_at_max_dispatches:
        print(
            "\n[Synthesis] CAPPED at --max-dispatches. See "
            "training_specs/pilot_progress.json. Re-run with a higher "
            "--max-dispatches to resume from the cache."
        )

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
