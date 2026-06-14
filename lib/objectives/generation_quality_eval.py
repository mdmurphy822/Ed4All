"""Generation-quality eval harness (W5 §1).

Scores ONE course at ONE technique config and emits a `generation_quality_eval`
report. Two metric families: **objective metrics** (over
``synthesized_objectives.json``) and **block metrics** (over the emitted
instructional blocks). The grounding metric family is a single call to
``lib.retrieval.groundedness.score_groundedness`` per artifact — the tutor's
citation gate pointed at generated prose.

Built on the **measure-then-pin** discipline of ``lib/retrieval/grounded_eval.py``
(the ``MILESTONE_TARGETS`` + single-corpus-caveat + seeded operator-review-sample
pattern — mirrored, NOT imported). Every threshold below is a *floor to be
measured first, pinned second*; the ``success_condition`` ships
``pinned_at="MEASURE-FIRST"`` and a single-corpus caveat. Objective *correctness*
(the right objective at the right grain) is curriculum-design judgment — it routes
to the seeded operator-review sample, never a pinned "matches the cloud set"
floor (§1.10 reference-not-gold).

Graceful degrade (mirror ``grounded_eval``): NLI/embedding absent ⇒ the dependent
metric reads ``null`` with a ``reason``, never a fabricated 0.0/1.0 — UNLESS
``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` flips to fail-closed. The structural gates
(ABCD, hierarchy) still run NLI-free.

Boundary (do NOT re-spec): W4 owns the entailment floors + the validators; W3 owns
the manifest shape; W2 owns the synthesis. W5 reads their outputs, computes rates,
and pins the success thresholds.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Report schema version (§1.9).
SCHEMA_VERSION = "1.0"

#: Subdirectory under a LibV2 course where reports land.
GENERATION_EVAL_SUBDIR = "generation_eval"

#: §1.5 Bloom-collapse floors (MEASURE-THEN-PIN — the reference sample-course-a set
#: spans all six incl. ``create``). Start advisory.
BLOOM_MIN_DISTINCT = 3
BLOOM_MAX_SHARE = 0.80

#: §1.6 objective dedup threshold (MEASURE-THEN-PIN — calibrate against the
#: reference set's OWN intra-set similarity, not from thin air; provenance plan R6).
DEDUP_THRESHOLD = 0.92

#: §1.4 concept-coverage cosine floor (advisory; measured-then-pinned).
CONCEPT_COVERAGE_TAU = 0.55

#: §4.1 PROPOSED success thresholds (MEASURE-FIRST — NOT pinned in code; the first
#: real C5 run pins them against a measured clean run). Shipped as the
#: ``success_condition`` floors so W6 reads ONE artifact to declare success.
PROPOSED_BLOCK_GROUNDING_FLOOR = 0.90
PROPOSED_PROVENANCE_COMPLETENESS_FLOOR = 1.00
PROPOSED_MANIFEST_MATCHES_FLOOR = 0.95
PROPOSED_OBJECTIVE_GROUNDING_FLOOR = 0.85

#: The W4 scaffolding set NOT scored for prose grounding (§1.1). Mirrors
#: ``claim_support._SKIPPED_BLOCK_TYPES``; SCORE ``example`` + ``self_check_question``.
_SCAFFOLD_BLOCK_TYPES = frozenset({"assessment_item", "objective", "chrome", "recap"})


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_strict() -> bool:
    """True when ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is truthy (fail-closed mode)."""
    return str(os.environ.get("TRAINFORGE_REQUIRE_EMBEDDINGS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _block_field(block: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off a Block dataclass OR a plain dict (duck-typed)."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _block_id(block: Any) -> str:
    bid = _block_field(block, "block_id") or _block_field(block, "id")
    if bid:
        return str(bid)
    # Compose a stable fallback from page/type when no id present.
    return str(_block_field(block, "block_type", "block"))


def _block_text(block: Any) -> str:
    """Strip the block's HTML/content to plain text for NLI scoring."""
    from lib.utils.html_text import strip_html_to_text

    content = _block_field(block, "content")
    if isinstance(content, dict):
        # Outline-tier dict content: join the text-bearing fields.
        parts: List[str] = []
        for key in ("body", "prose", "text", "explanation", "definition", "example"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
        for val in content.get("key_claims", []) or []:
            if isinstance(val, str):
                parts.append(val)
        raw = "\n".join(parts) if parts else json.dumps(content, ensure_ascii=False)
    elif isinstance(content, str):
        raw = content
    else:
        raw = _block_field(block, "html") or _block_field(block, "text") or ""
    return strip_html_to_text(str(raw)) if raw else ""


def _block_source_ids(block: Any) -> List[str]:
    """Resolve the block's cited source-chunk ids (source_references + source_ids)."""
    ids: List[str] = []
    seen: set = set()
    for ref in _block_field(block, "source_references", ()) or ():
        sid = ref.get("sourceId") if isinstance(ref, dict) else None
        if isinstance(sid, str) and sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
    for sid in _block_field(block, "source_ids", ()) or ():
        if isinstance(sid, str) and sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return str(getattr(chunk, "text", "") or "")


def _resolve_passages(
    chunk_ids: Sequence[str], chunks_by_id: Dict[str, Any]
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Resolve chunk ids → ``[{chunk_id, text}]`` premises + unresolved ids."""
    passages: List[Dict[str, str]] = []
    unresolved: List[str] = []
    for cid in chunk_ids:
        chunk = chunks_by_id.get(cid)
        text = _chunk_text(chunk) if chunk is not None else ""
        if text.strip():
            passages.append({"chunk_id": cid, "text": text})
        else:
            unresolved.append(cid)
    return passages, unresolved


# --------------------------------------------------------------------------- #
# Objective shape normalization (reuse the W4 flattener)
# --------------------------------------------------------------------------- #

def _flatten(objectives_payload: Any) -> List[Dict[str, Any]]:
    """Flatten a tolerant objectives payload via the W4 normaliser."""
    from lib.validators.abcd_objective import _flatten_objectives

    return list(_flatten_objectives(objectives_payload))


def _objective_statement(lo: Dict[str, Any]) -> str:
    for key in ("statement", "behavioral_outcome", "text", "objective", "description"):
        val = lo.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _objective_level(lo: Dict[str, Any]) -> str:
    """Resolve an objective's hierarchy level for within-level dedup bucketing.

    Prefers the explicit ``hierarchy_level`` field (Courseforge synthesized
    form); falls back to inferring it from the canonical LO id prefix
    (``TO-`` terminal / ``CO-`` chapter); ``"unknown"`` when neither resolves.

    A terminal objective is the abstract umbrella for its child chapter
    objectives, so a TO sharing text with one of its COs is an
    abstraction smell on a *different* quality axis — NOT the redundant-CO
    duplication the in-synthesis dedup pass (and this metric) guard against.
    Bucketing by level keeps the dedup pillar measuring within-level
    redundancy only.
    """
    level = lo.get("hierarchy_level")
    if isinstance(level, str) and level.strip():
        return level.strip()
    lo_id = lo.get("id")
    if isinstance(lo_id, str) and lo_id:
        try:
            from lib.ontology.learning_objectives import hierarchy_from_id

            return hierarchy_from_id(lo_id)
        except Exception:
            pass
    return "unknown"


def _objective_cited_chunk_ids(lo: Dict[str, Any]) -> List[str]:
    """Pull structured ``source_refs[].chunk_ids[]`` (legacy list / empty skipped)."""
    ids: List[str] = []
    for ref in lo.get("source_refs", []) or []:
        if not isinstance(ref, dict):
            continue
        for cid in ref.get("chunk_ids", []) or []:
            if isinstance(cid, str) and cid:
                ids.append(cid)
    return ids


def _objective_bloom(lo: Dict[str, Any]) -> Optional[str]:
    bloom = lo.get("bloom_level") or lo.get("bloomLevel")
    if isinstance(bloom, str) and bloom.strip():
        return bloom.strip().lower()
    statement = _objective_statement(lo)
    if statement:
        from lib.ontology.bloom import detect_bloom_level

        level, _verb = detect_bloom_level(statement)
        return level
    return None


# --------------------------------------------------------------------------- #
# NLI / embedding resolution (injectable seams)
# --------------------------------------------------------------------------- #

def _resolve_nli(nli: Optional[Any]) -> Optional[Any]:
    if nli is not None:
        return nli
    try:
        from lib.classifiers.nli_classifier import NliClassifier
    except Exception:  # noqa: BLE001
        return None
    return NliClassifier.get_or_load()


def _resolve_embedder(embedder: Optional[Any]) -> Optional[Any]:
    if embedder is not None:
        return embedder
    try:
        from lib.embedding.providers import build_embedding_client

        return build_embedding_client(offline=True)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Block-prose grounding (§1.1)
# --------------------------------------------------------------------------- #

def _score_blocks(
    blocks: List[Any],
    chunks_by_id: Dict[str, Any],
    manifests: Optional[Dict[str, Any]],
    nli: Optional[Any],
    *,
    min_block_entailment_rate: float,
) -> Dict[str, Any]:
    """Score every scorable block's prose grounding via NLI (§1.1)."""
    from lib.retrieval.groundedness import score_groundedness

    rows: List[Dict[str, Any]] = []
    grounded_count = 0
    contradicted_count = 0
    scored_blocks = 0
    skipped_scaffold = 0
    entailment_rates: List[float] = []
    windowed_rescued = 0
    windowed_total = 0
    computational = 0
    filtered = 0
    grounded_chunk_ids: set = set()

    for block in blocks:
        btype = str(_block_field(block, "block_type", "") or "")
        bid = _block_id(block)
        if btype in _SCAFFOLD_BLOCK_TYPES:
            skipped_scaffold += 1
            continue
        scored_blocks += 1
        text = _block_text(block)
        # Prefer W3 manifest source_chunk_ids (the authoritative "what was used").
        manifest = (manifests or {}).get(bid) if manifests else None
        if manifest and manifest.get("source_chunk_ids"):
            cited_ids = [c for c in manifest.get("source_chunk_ids", []) if isinstance(c, str)]
        else:
            cited_ids = _block_source_ids(block)
        passages, _unresolved = _resolve_passages(cited_ids, chunks_by_id)

        if nli is None:
            rows.append(
                {
                    "block_id": bid,
                    "block_type": btype,
                    "entailment_rate": None,
                    "contradicted": False,
                    "manifest_complete": None,
                    "grounded": None,
                }
            )
            continue

        report = score_groundedness(text, passages, nli=nli)
        rate = float(report.groundedness_rate)
        contradicted = report.contradicted_count > 0
        grounded = (rate >= min_block_entailment_rate) and not contradicted
        entailment_rates.append(rate)
        windowed_total += report.scored_count
        windowed_rescued += sum(1 for c in report.claims if getattr(c, "windowed", False))
        computational += report.computational_count
        filtered += report.filtered_count
        if contradicted:
            contradicted_count += 1
        if grounded:
            grounded_count += 1
            for cid in cited_ids:
                grounded_chunk_ids.add(cid)
        rows.append(
            {
                "block_id": bid,
                "block_type": btype,
                "entailment_rate": rate,
                "contradicted": contradicted,
                "grounded": grounded,
            }
        )

    nli_scored = bool(nli is not None and scored_blocks)
    return {
        "rows": rows,
        "scored_blocks": scored_blocks,
        "skipped_scaffold": skipped_scaffold,
        "grounded_count": grounded_count,
        "contradicted_block_count": contradicted_count,
        "block_prose_grounding_rate": (grounded_count / scored_blocks)
        if nli_scored
        else None,
        "mean_block_entailment_rate": (sum(entailment_rates) / len(entailment_rates))
        if entailment_rates
        else None,
        "windowed_rescue_share": (windowed_rescued / windowed_total)
        if windowed_total
        else (0.0 if nli_scored else None),
        "computational_claim_count": computational,
        "filtered_claim_count": filtered,
        "grounded_chunk_ids": grounded_chunk_ids,
    }


# --------------------------------------------------------------------------- #
# Objective grounding (§1.2)
# --------------------------------------------------------------------------- #

def _score_objectives_grounding(
    objectives: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    nli: Optional[Any],
    *,
    objective_entailment_floor: float,
) -> Dict[str, Any]:
    from lib.retrieval.groundedness import score_groundedness

    rows: List[Dict[str, Any]] = []
    scored = 0
    grounded = 0
    contradicted = 0
    unscorable = 0
    for lo in objectives:
        lo_id = str(lo.get("id") or lo.get("objective_id") or "")
        statement = _objective_statement(lo)
        cited_ids = _objective_cited_chunk_ids(lo)
        passages, _unresolved = _resolve_passages(cited_ids, chunks_by_id)
        if not passages or not statement:
            unscorable += 1
            rows.append(
                {
                    "id": lo_id,
                    "grounded": None,
                    "entailment": None,
                    "contradicted": False,
                    "cited_chunk_ids": cited_ids,
                }
            )
            continue
        if nli is None:
            rows.append(
                {
                    "id": lo_id,
                    "grounded": None,
                    "entailment": None,
                    "contradicted": False,
                    "cited_chunk_ids": cited_ids,
                }
            )
            continue
        scored += 1
        report = score_groundedness(statement, passages, nli=nli)
        rate = float(report.groundedness_rate)
        is_contra = report.contradicted_count > 0
        is_grounded = (rate >= objective_entailment_floor) and not is_contra
        if is_contra:
            contradicted += 1
        if is_grounded:
            grounded += 1
        rows.append(
            {
                "id": lo_id,
                "grounded": is_grounded,
                "entailment": rate,
                "contradicted": is_contra,
                "cited_chunk_ids": cited_ids,
            }
        )

    nli_scored = bool(nli is not None and scored)
    return {
        "rows": rows,
        "scored_objectives": scored,
        "objectives_unscorable": unscorable,
        "objective_grounding_rate": (grounded / scored) if nli_scored else None,
        "objective_unsupported_rate": (1.0 - grounded / scored) if nli_scored else None,
        "objective_contradicted_count": contradicted,
    }


# --------------------------------------------------------------------------- #
# Provenance completeness (§1.3)
# --------------------------------------------------------------------------- #

_REQUIRED_MANIFEST_KEYS = ("source_chunk_ids", "model", "params", "concept_tags")


def _manifest_complete(manifest: Dict[str, Any]) -> bool:
    """A manifest is complete iff the required keys are present + non-empty.

    ``source_chunk_ids`` must be non-empty; ``model`` + ``params`` present;
    ``concept_tags`` present (possibly empty list — present, not absent). A
    ``template_id`` + ``slots`` OR the documented free-scaffold sentinel covers
    the template surface.
    """
    if not isinstance(manifest, dict):
        return False
    if not manifest.get("source_chunk_ids"):
        return False
    if "model" not in manifest or "params" not in manifest:
        return False
    if "concept_tags" not in manifest:
        return False
    has_template = manifest.get("template_id") is not None and "slots" in manifest
    has_scaffold = bool(manifest.get("free_scaffold"))
    return has_template or has_scaffold


def _resolve_to_chunk_id(source_id: str, source_resolver: Optional[Any]) -> str:
    """Map a block-cited source id to its chunk id via the manifest resolver.

    A block may cite by chunk id directly (W2 ``key_claims[].source_chunk_ids``)
    OR by DART slug (``dart:{slug}#{block_id}``, the outline ``source_refs``
    shape). The manifest projection resolves both to chunk ids, so the
    ``manifest ⊆ block_cited`` cross-check must compare in the SAME (chunk-id)
    namespace — otherwise a block that cited by slug while the manifest holds
    the resolved chunk id is a spurious mismatch. Resolves through the same
    callable ``Block.to_synthesis_manifest`` consumed; falls back to the raw id
    when no resolver is supplied or the id is unresolvable (already a chunk id).
    """
    if source_resolver is None:
        return source_id
    try:
        resolved = source_resolver(source_id)
    except Exception:  # noqa: BLE001
        return source_id
    if isinstance(resolved, dict):
        cid = resolved.get("id") or resolved.get("chunk_id")
        if isinstance(cid, str) and cid:
            return cid
    elif isinstance(resolved, str) and resolved:
        return resolved
    return source_id


def _score_provenance(
    blocks: List[Any],
    manifests: Optional[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    source_resolver: Optional[Any] = None,
) -> Dict[str, Any]:
    if manifests is None:
        return {
            "provenance_completeness_rate": None,
            "reason": "manifests_not_supplied",
            "manifest_chunk_resolution_rate": None,
            "manifest_matches_cited_rate": None,
            "per_block_complete": {},
        }
    # Provenance completeness scopes to the SYNTHESIZED (non-scaffold) blocks —
    # the W4 scaffolding set (assessment_item / objective / chrome / recap) is
    # template chrome with no source prose to manifest, so including it would
    # depress the rate for a structurally-correct course. This mirrors the
    # block-prose-grounding skip set (§1.1).
    scorable_blocks = [
        b for b in blocks
        if str(_block_field(b, "block_type", "") or "") not in _SCAFFOLD_BLOCK_TYPES
    ]
    total = len(scorable_blocks)
    complete = 0
    per_block: Dict[str, bool] = {}
    resolved_manifests = 0
    total_manifests = 0
    matches = 0
    for block in scorable_blocks:
        bid = _block_id(block)
        manifest = manifests.get(bid)
        is_complete = bool(manifest) and _manifest_complete(manifest)
        per_block[bid] = is_complete
        if is_complete:
            complete += 1
        if manifest:
            total_manifests += 1
            chunk_ids = [c for c in manifest.get("source_chunk_ids", []) if isinstance(c, str)]
            if chunk_ids and all(cid in chunks_by_id for cid in chunk_ids):
                resolved_manifests += 1
            # Cross-check: manifest source set ⊆ block's actual cited set.
            # Resolve the block's cited ids to chunk-id space first so a block
            # that cited by DART slug (while the manifest holds the resolved
            # chunk id) is not a spurious namespace mismatch.
            block_cited = {
                _resolve_to_chunk_id(sid, source_resolver)
                for sid in _block_source_ids(block)
            }
            if block_cited and set(chunk_ids) <= block_cited:
                matches += 1
            elif not block_cited and not chunk_ids:
                matches += 1
    return {
        "provenance_completeness_rate": (complete / total) if total else 0.0,
        "reason": None,
        "manifest_chunk_resolution_rate": (resolved_manifests / total_manifests)
        if total_manifests
        else None,
        "manifest_matches_cited_rate": (matches / total_manifests)
        if total_manifests
        else None,
        "per_block_complete": per_block,
    }


# --------------------------------------------------------------------------- #
# Bloom distribution (§1.5)
# --------------------------------------------------------------------------- #

#: Pedagogically-essential instructional block types. A "complete"
#: instructional unit explains a concept, illustrates it with a worked
#: example, and checks understanding with an assessment. The 7B smoke course
#: today emits only concept+explanation, so this diagnostic surfaces the
#: instructional-depth gap vs a full course (informational — NOT a
#: success_condition pillar).
_INSTRUCTIONAL_ESSENTIAL_TYPES = ("concept", "explanation", "example", "assessment_item")


def _score_block_type_diversity(blocks: List[Any]) -> Dict[str, Any]:
    """Block-type histogram + instructional-completeness diagnostic.

    ``instructional_completeness`` is the share of the pedagogically-essential
    types (:data:`_INSTRUCTIONAL_ESSENTIAL_TYPES`) present at least once.
    Treats concept/explanation as interchangeable for the "explain" slot, so
    the essentials reduce to {explain, example, assessment_item}. Purely
    informational: a grounded-but-thin course (concept+explanation only)
    reads low here without failing the W5 success condition.
    """
    import collections

    histogram = collections.Counter(
        str(_block_field(b, "block_type", "") or "") for b in blocks
    )
    present = set(histogram)
    has_explain = bool(present & {"concept", "explanation"})
    has_example = "example" in present
    has_assessment = "assessment_item" in present
    essentials_met = sum((has_explain, has_example, has_assessment))
    return {
        "histogram": dict(histogram),
        "distinct_types": len([t for t in present if t]),
        "has_explain": has_explain,
        "has_example": has_example,
        "has_assessment": has_assessment,
        "instructional_completeness": round(essentials_met / 3.0, 4),
    }


def _score_bloom(objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
    from lib.ontology.bloom import BLOOM_LEVELS

    histogram = {level: 0 for level in BLOOM_LEVELS}
    total = 0
    for lo in objectives:
        level = _objective_bloom(lo)
        if level in histogram:
            histogram[level] += 1
            total += 1
    distinct = sum(1 for v in histogram.values() if v > 0)
    max_share = (max(histogram.values()) / total) if total else 0.0
    collapse = bool(total) and (
        distinct < BLOOM_MIN_DISTINCT or max_share > BLOOM_MAX_SHARE
    )
    return {
        "histogram": histogram,
        "distinct_levels": distinct,
        "max_share": max_share if total else None,
        "collapse": collapse,
    }


# --------------------------------------------------------------------------- #
# Embedding metrics: dedup (§1.6) + concept coverage (§1.4)
# --------------------------------------------------------------------------- #

def _embed(embedder: Any, texts: List[str]):
    return embedder.encode_batch(texts)


def _score_dedup(
    objectives: List[Dict[str, Any]], embedder: Optional[Any]
) -> Dict[str, Any]:
    # Carry the hierarchy level alongside each statement so near-dup pairs
    # are counted WITHIN a level only (TO-vs-CO identity is a separate
    # abstraction axis, not redundant-objective duplication — see
    # ``_objective_level``).
    rows = [
        (_objective_statement(lo), _objective_level(lo))
        for lo in objectives
    ]
    rows = [(s, lvl) for (s, lvl) in rows if s.strip()]
    if embedder is None or len(rows) < 2:
        return {
            "max_pairwise_cosine": None,
            "near_dup_pairs": None,
            "dedup_clean": None,
            "cross_level_max_cosine": None,
        }
    import numpy as np

    statements = [s for (s, _lvl) in rows]
    levels = [lvl for (_s, lvl) in rows]
    vecs = _embed(embedder, statements)
    vecs = np.asarray(vecs, dtype=np.float32)
    sims = vecs @ vecs.T
    n = sims.shape[0]
    max_cos = 0.0  # max WITHIN-level pair cosine (the dedup-relevant signal)
    cross_level_max = 0.0  # transparency: highest cross-level pair (e.g. TO≈CO)
    near_dups = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = float(sims[i, j])
            if levels[i] == levels[j]:
                max_cos = max(max_cos, c)
                if c >= DEDUP_THRESHOLD:
                    near_dups += 1
            else:
                cross_level_max = max(cross_level_max, c)
    return {
        "max_pairwise_cosine": max_cos,
        "near_dup_pairs": near_dups,
        "dedup_clean": near_dups == 0,
        "cross_level_max_cosine": cross_level_max,
    }


def _score_concept_coverage(
    objectives: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    block_rows: List[Dict[str, Any]],
    blocks: List[Any],
    embedder: Optional[Any],
) -> Optional[float]:
    if embedder is None:
        return None
    # Harvest the concept/key-term set from chunk concept_tags.
    concepts: set = set()
    for chunk in chunks_by_id.values():
        for tag in (chunk.get("concept_tags") if isinstance(chunk, dict) else []) or []:
            if isinstance(tag, str) and tag.strip():
                concepts.add(tag.strip())
    if not concepts:
        return None
    targets: List[str] = [s for s in (_objective_statement(lo) for lo in objectives) if s.strip()]
    grounded_ids = {r["block_id"] for r in block_rows if r.get("grounded")}
    for block in blocks:
        if _block_id(block) in grounded_ids:
            txt = _block_text(block)
            if txt.strip():
                targets.append(txt)
    if not targets:
        return None
    import numpy as np

    concept_list = sorted(concepts)
    cvecs = np.asarray(_embed(embedder, concept_list), dtype=np.float32)
    tvecs = np.asarray(_embed(embedder, targets), dtype=np.float32)
    sims = cvecs @ tvecs.T  # [n_concepts, n_targets]
    covered = int(np.sum(np.max(sims, axis=1) >= CONCEPT_COVERAGE_TAU))
    return covered / len(concept_list)


# --------------------------------------------------------------------------- #
# Hierarchy gates (§1.7)
# --------------------------------------------------------------------------- #

def _run_hierarchy_gates(
    objectives_payload: Any,
    objectives: List[Dict[str, Any]],
    blocks: List[Any],
    textbook_structure: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    from lib.ontology.learning_objectives import validate_lo_id
    from lib.validators.abcd_objective import AbcdObjectiveValidator
    from lib.validators.block_objective_delivery import BlockObjectiveDeliveryValidator
    from lib.validators.chapter_objective_coverage import (
        ChapterObjectiveCoverageValidator,
    )
    from lib.validators.objective_source_refs import ObjectiveSourceRefValidator
    from lib.validators.terminal_objective_coverage import (
        TerminalObjectiveCoverageValidator,
    )

    obj_map = {
        str(lo.get("id") or lo.get("objective_id") or ""): lo
        for lo in objectives
        if (lo.get("id") or lo.get("objective_id"))
    }

    plan: List[Tuple[str, Any, Dict[str, Any], bool]] = [
        (
            "abcd_verb_alignment",
            AbcdObjectiveValidator(),
            {"objectives": objectives_payload},
            True,
        ),
        (
            "objective_source_refs",
            ObjectiveSourceRefValidator(),
            {"objectives": objectives_payload},
            False,
        ),
        (
            "chapter_objective_coverage",
            ChapterObjectiveCoverageValidator(),
            {
                "synthesized_objectives": objectives_payload,
                "textbook_structure": textbook_structure or {},
            },
            True,
        ),
        (
            "terminal_objective_coverage",
            TerminalObjectiveCoverageValidator(),
            {
                "synthesized_objectives": objectives_payload,
                "textbook_structure": textbook_structure or {},
            },
            True,
        ),
        (
            "block_objective_delivery",
            BlockObjectiveDeliveryValidator(),
            {"blocks": blocks, "objectives": obj_map},
            False,
        ),
    ]

    gate_rows: List[Dict[str, Any]] = []
    all_critical_pass = True
    for gate_id, validator, inputs, is_critical in plan:
        try:
            result = validator.validate(dict(inputs, gate_id=gate_id))
            passed = bool(result.passed)
            critical_count = sum(
                1 for issue in (result.issues or []) if getattr(issue, "severity", "") == "critical"
            )
            issue_codes = [getattr(i, "code", "") for i in (result.issues or [])]
            score = result.score
            action = result.action
        except Exception as exc:  # noqa: BLE001 — a broken validator is a hard finding
            passed = False
            critical_count = 1
            issue_codes = [f"VALIDATOR_ERROR:{type(exc).__name__}"]
            score = None
            action = None
        gate_rows.append(
            {
                "gate_id": gate_id,
                "passed": passed,
                "action": action,
                "critical_count": critical_count,
                "score": score,
                "issue_codes": issue_codes,
            }
        )
        if is_critical and not passed:
            all_critical_pass = False

    ids_conformant = all(
        validate_lo_id(str(lo.get("id") or lo.get("objective_id") or ""))
        for lo in objectives
        if (lo.get("id") or lo.get("objective_id"))
    )
    hierarchy_valid = all_critical_pass and ids_conformant
    return gate_rows, hierarchy_valid


# --------------------------------------------------------------------------- #
# Downstream usability (§1.8)
# --------------------------------------------------------------------------- #

def _block_lo_reference_rate(
    objectives: List[Dict[str, Any]], chunks_by_id: Dict[str, Any]
) -> Optional[float]:
    """Objectives that ≥1 chunk references via learning_outcome_refs / total."""
    from lib.ontology.learning_objectives import scan_lo_refs

    obj_ids = {
        str(lo.get("id") or lo.get("objective_id") or "").upper()
        for lo in objectives
        if (lo.get("id") or lo.get("objective_id"))
    }
    obj_ids.discard("")
    if not obj_ids:
        return None
    referenced: set = set()
    for chunk in chunks_by_id.values():
        if not isinstance(chunk, dict):
            continue
        for ref in chunk.get("learning_outcome_refs", []) or []:
            if isinstance(ref, str) and ref.upper() in obj_ids:
                referenced.add(ref.upper())
        # Also scan chunk text for embedded LO refs (back-fill robustness).
        for ref in scan_lo_refs(text=str(chunk.get("text") or ""), allowed_ids=obj_ids):
            referenced.add(ref.upper())
    return len(referenced) / len(obj_ids)


# --------------------------------------------------------------------------- #
# Review sample (§5.4 — deterministic, seeded; mirrors grounded_eval)
# --------------------------------------------------------------------------- #

def _sample_size(n_items: int) -> int:
    return min(max(10, int(round(0.20 * n_items))), n_items)


def _seeded_order(item_ids: Sequence[str], course_slug: str) -> List[str]:
    def _key(iid: str) -> str:
        return hashlib.sha256(f"{course_slug}::{iid}".encode("utf-8")).hexdigest()

    return sorted(item_ids, key=_key)


def _build_review_sample(
    course_slug: str,
    block_rows: List[Dict[str, Any]],
    objective_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Seeded operator-review sample (objective correctness is human judgment)."""
    items: Dict[str, Dict[str, Any]] = {}
    for row in block_rows:
        items[f"block::{row['block_id']}"] = {
            "item_id": f"block::{row['block_id']}",
            "kind": "block",
            "block_type": row.get("block_type"),
            "entailment_rate": row.get("entailment_rate"),
            "contradicted": row.get("contradicted"),
            "grounded": row.get("grounded"),
        }
    for row in objective_rows:
        items[f"objective::{row['id']}"] = {
            "item_id": f"objective::{row['id']}",
            "kind": "objective",
            "entailment": row.get("entailment"),
            "contradicted": row.get("contradicted"),
            "grounded": row.get("grounded"),
            "cited_chunk_ids": row.get("cited_chunk_ids", []),
        }
    ordered = _seeded_order(list(items.keys()), course_slug)
    n = _sample_size(len(ordered))
    samples = []
    for iid in ordered[:n]:
        row = dict(items[iid])
        row["reviewer_fields"] = {
            "objective_correct": None,
            "grounding_correct": None,
            "bloom_correct": None,
            "grain_correct": None,
            "notes": "",
        }
        samples.append(row)
    return {
        "schema_version": "1.0",
        "course_slug": course_slug,
        "sample_seed": "sha256(course_slug::item_id)",
        "n_sampled": len(samples),
        "n_items": len(ordered),
        "samples": samples,
    }


# --------------------------------------------------------------------------- #
# Success condition (§4)
# --------------------------------------------------------------------------- #

def _build_success_condition(headline: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
    """Compose the four-pillar success verdict (a null any-one ⇒ met=false)."""

    def _pillar(metric: str, value: Any, threshold: float, *, extra_ok: bool = True) -> Dict[str, Any]:
        met = (value is not None) and (float(value) >= threshold) and extra_ok
        return {"metric": metric, "value": value, "threshold": threshold, "met": met}

    bloom_ok = headline["bloom"]["collapse"] is False
    dedup = headline["objective_dedup"]
    dedup_clean = dedup.get("dedup_clean") is True
    hierarchy_ok = headline["hierarchy_valid"] is True

    p_grounded = _pillar(
        "block_prose_grounding_rate",
        headline["block_prose_grounding_rate"],
        PROPOSED_BLOCK_GROUNDING_FLOOR,
    )
    p_provenanced = _pillar(
        "provenance_completeness_rate",
        headline["provenance_completeness_rate"],
        PROPOSED_PROVENANCE_COMPLETENESS_FLOOR,
    )
    p_metadata = _pillar(
        "manifest_matches_cited_rate",
        audit["manifest_matches_cited_rate"],
        PROPOSED_MANIFEST_MATCHES_FLOOR,
        extra_ok=bloom_ok,
    )
    p_metadata["and"] = "bloom.collapse == false"
    p_objectives = _pillar(
        "objective_grounding_rate",
        headline["objective_grounding_rate"],
        PROPOSED_OBJECTIVE_GROUNDING_FLOOR,
        extra_ok=hierarchy_ok and dedup_clean,
    )
    p_objectives["and"] = "hierarchy_valid == true && objective_dedup.dedup_clean == true"

    hard_zeros = {
        "contradicted_block_count": headline["contradicted_block_count"],
        "objective_contradicted_count": headline["objective_contradicted_count"],
    }
    hard_zero_ok = (
        hard_zeros["contradicted_block_count"] == 0
        and hard_zeros["objective_contradicted_count"] == 0
    )
    met = (
        p_grounded["met"]
        and p_provenanced["met"]
        and p_metadata["met"]
        and p_objectives["met"]
        and hard_zero_ok
    )
    return {
        "met": met,
        "pillars": {
            "every_block_grounded": p_grounded,
            "every_block_provenanced": p_provenanced,
            "concept_metadata_correct": p_metadata,
            "objectives_grounded_valid": p_objectives,
        },
        "hard_zeros": hard_zeros,
        "pinned_at": "MEASURE-FIRST",
        "single_corpus_caveat": True,
    }


# --------------------------------------------------------------------------- #
# Loaders (default on-disk discovery)
# --------------------------------------------------------------------------- #

def _course_dir(repo_root: Path, course_slug: str) -> Path:
    libv2_root = os.environ.get("ED4ALL_LIBV2_ROOT")
    base = Path(libv2_root) if libv2_root else (Path(repo_root) / "LibV2")
    return base / "courses" / course_slug


def _load_chunks(repo_root: Path, course_slug: str) -> Dict[str, Any]:
    from lib.retrieval.gold_set import _load_chunks_by_id

    course_dir = _course_dir(repo_root, course_slug)
    for sub in ("dart_chunks", "imscc_chunks"):
        path = course_dir / sub / "chunks.jsonl"
        if path.exists():
            return _load_chunks_by_id(path)
    return {}


# --------------------------------------------------------------------------- #
# THE harness
# --------------------------------------------------------------------------- #

def run_generation_quality_eval(
    repo_root: Path,
    course_slug: str,
    *,
    objectives: Optional[Dict] = None,
    blocks: Optional[List[Dict]] = None,
    chunks_by_id: Optional[Dict] = None,
    textbook_structure: Optional[Dict] = None,
    manifests: Optional[Dict] = None,
    nli: Optional[Any] = None,
    embedder: Optional[Any] = None,
    technique_config: str = "C5",
    capture: Optional[Any] = None,
    write: bool = True,
    output_path: Optional[Path] = None,
    source_resolver: Optional[Any] = None,
) -> Dict[str, Any]:
    """Score ONE course at ONE technique config (W5 §1). Returns the report dict.

    Inputs default to on-disk discovery under the LibV2 course root when not
    supplied. NLI / embedding are injectable seams (CI passes fakes); when absent
    the dependent metric reads ``null`` + ``reason`` (honest degrade), UNLESS
    ``TRAINFORGE_REQUIRE_EMBEDDINGS`` flips to fail-closed. The success-condition
    treats a null provenance rate as **not met** (anti-silent-degradation).
    """
    from lib.validators.block_prose_entailment import _DEFAULT_MIN_BLOCK_ENTAILMENT_RATE
    from lib.validators.objective_entailment import _DEFAULT_OBJECTIVE_ENTAILMENT_RATE_FLOOR

    repo_root = Path(repo_root)
    objectives_payload = objectives if objectives is not None else {}
    if objectives is None:
        # Default discovery: synthesized_objectives.json under the LibV2 course.
        cand = _course_dir(repo_root, course_slug) / "synthesized_objectives.json"
        if cand.exists():
            objectives_payload = json.loads(cand.read_text(encoding="utf-8"))
    flat_objectives = _flatten(objectives_payload)
    block_list = list(blocks or [])
    chunks = dict(chunks_by_id) if chunks_by_id is not None else _load_chunks(repo_root, course_slug)

    resolved_nli = _resolve_nli(nli)
    if resolved_nli is None and _require_strict():
        raise RuntimeError(
            "run_generation_quality_eval requires the NLI model "
            "(install the embedding/NLI extras) but it is unavailable and "
            "TRAINFORGE_REQUIRE_EMBEDDINGS is set; no grounding scores fabricated."
        )
    resolved_embedder = _resolve_embedder(embedder)
    if resolved_embedder is None and _require_strict():
        raise RuntimeError(
            "run_generation_quality_eval requires an embedding backend but it is "
            "unavailable and TRAINFORGE_REQUIRE_EMBEDDINGS is set; install the "
            "[embedding] extras. No coverage/dedup scores fabricated."
        )

    # §1.1 block-prose grounding.
    block_metrics = _score_blocks(
        block_list,
        chunks,
        manifests,
        resolved_nli,
        min_block_entailment_rate=_DEFAULT_MIN_BLOCK_ENTAILMENT_RATE,
    )
    # §1.2 objective grounding.
    obj_metrics = _score_objectives_grounding(
        flat_objectives,
        chunks,
        resolved_nli,
        objective_entailment_floor=_DEFAULT_OBJECTIVE_ENTAILMENT_RATE_FLOOR,
    )
    # §1.3 provenance completeness.
    prov = _score_provenance(block_list, manifests, chunks, source_resolver=source_resolver)
    # §1.4 concept coverage.
    chunk_coverage = (
        (len(block_metrics["grounded_chunk_ids"]) / len(chunks))
        if (resolved_nli is not None and chunks)
        else None
    )
    concept_coverage = _score_concept_coverage(
        flat_objectives, chunks, block_metrics["rows"], block_list, resolved_embedder
    )
    # §1.5 bloom.
    bloom = _score_bloom(flat_objectives)
    # §1.6 dedup.
    dedup = _score_dedup(flat_objectives, resolved_embedder)
    # §1.7 hierarchy gates.
    gate_rows, hierarchy_valid = _run_hierarchy_gates(
        objectives_payload, flat_objectives, block_list, textbook_structure
    )
    # §1.8 downstream.
    block_lo_rate = _block_lo_reference_rate(flat_objectives, chunks)

    headline = {
        "block_prose_grounding_rate": block_metrics["block_prose_grounding_rate"],
        "objective_grounding_rate": obj_metrics["objective_grounding_rate"],
        "provenance_completeness_rate": prov["provenance_completeness_rate"],
        "mean_block_entailment_rate": block_metrics["mean_block_entailment_rate"],
        "contradicted_block_count": block_metrics["contradicted_block_count"],
        "objective_unsupported_rate": obj_metrics["objective_unsupported_rate"],
        "objective_contradicted_count": obj_metrics["objective_contradicted_count"],
        "chunk_coverage_rate": chunk_coverage,
        "concept_coverage_rate": concept_coverage,
        "bloom": bloom,
        "objective_dedup": dedup,
        "hierarchy_valid": hierarchy_valid,
        "block_lo_reference_rate": block_lo_rate,
        "windowed_rescue_share": block_metrics["windowed_rescue_share"],
        "block_type_diversity": _score_block_type_diversity(block_list),
    }
    provenance_audit = {
        "manifest_chunk_resolution_rate": prov["manifest_chunk_resolution_rate"],
        "manifest_matches_cited_rate": prov["manifest_matches_cited_rate"],
    }
    success_condition = _build_success_condition(headline, provenance_audit)

    review_sample = _build_review_sample(
        course_slug, block_metrics["rows"], obj_metrics["rows"]
    )

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "course_slug": course_slug,
        "technique_config": technique_config,
        "generated_at": _utcnow_iso(),
        "provenance": {
            "nli_model_revision": getattr(resolved_nli, "_revision", None),
            "nli_device": getattr(resolved_nli, "device", None),
            "embedding_provider": getattr(resolved_embedder, "provider_name", None),
            "embedding_model": getattr(resolved_embedder, "model_id", None),
            "thresholds_provenance": "claim_support_defaults",
            "single_corpus_caveat": True,
        },
        "counts": {
            "total_blocks": len(block_list),
            "scored_blocks": block_metrics["scored_blocks"],
            "skipped_scaffold_blocks": block_metrics["skipped_scaffold"],
            "total_objectives": len(flat_objectives),
            "scored_objectives": obj_metrics["scored_objectives"],
            "objectives_unscorable": obj_metrics["objectives_unscorable"],
        },
        "headline": headline,
        "hierarchy_gates": gate_rows,
        "provenance_audit": provenance_audit,
        "downstream": {"served_answer_eval": None},
        "diagnostics": {
            "computational_claim_count": block_metrics["computational_claim_count"],
            "filtered_claim_count": block_metrics["filtered_claim_count"],
        },
        "blocks": block_metrics["rows"],
        "objectives": obj_metrics["rows"],
        "success_condition": success_condition,
        "review_sample": review_sample,
    }
    if prov["reason"]:
        report["provenance"]["provenance_reason"] = prov["reason"]

    if capture is not None:
        try:
            capture.log_decision(
                decision_type="generation_quality_eval",
                decision=(
                    f"scored {course_slug} @ {technique_config}: "
                    f"block_grounding={headline['block_prose_grounding_rate']}"
                ),
                rationale=(
                    f"technique={technique_config}; scored "
                    f"{block_metrics['scored_blocks']} blocks + "
                    f"{obj_metrics['scored_objectives']} objectives; "
                    f"contradicted_blocks={headline['contradicted_block_count']}; "
                    f"success_met={success_condition['met']}; "
                    f"nli={'on' if resolved_nli else 'off'}"
                ),
            )
        except Exception:  # noqa: BLE001 — capture is best-effort
            pass

    if write:
        eval_dir = _course_dir(repo_root, course_slug) / GENERATION_EVAL_SUBDIR
        eval_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(output_path) if output_path is not None else (
            eval_dir / f"generation_quality_eval_{technique_config}_{ts}.json"
        )
        # A caller-supplied output_path may point outside eval_dir (which we
        # created above); ensure ITS parent exists so an explicit path never
        # crashes on a missing intermediate directory.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["_written"] = {"report_path": str(out)}

    return report


__all__ = [
    "run_generation_quality_eval",
    "SCHEMA_VERSION",
    "PROPOSED_BLOCK_GROUNDING_FLOOR",
    "PROPOSED_PROVENANCE_COMPLETENESS_FLOOR",
    "PROPOSED_MANIFEST_MATCHES_FLOOR",
    "PROPOSED_OBJECTIVE_GROUNDING_FLOOR",
    "BLOOM_MIN_DISTINCT",
    "BLOOM_MAX_SHARE",
    "DEDUP_THRESHOLD",
]
