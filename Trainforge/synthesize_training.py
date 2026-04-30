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
import html
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Make project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture  # noqa: E402
from lib.validators.content_type import (  # noqa: E402
    assert_chunk_type,
    validate_chunk_type,
)
from Trainforge.generators.instruction_factory import (  # noqa: E402
    synthesize_instruction_pair,
)
from Trainforge.generators.preference_factory import (  # noqa: E402
    synthesize_preference_pair,
)
from Trainforge.curriculum import (  # noqa: E402
    DEFAULT_PREREQ_CONTEXT_TOKENS,
    build_curriculum_context,
    build_curriculum_manifest,
    build_prereq_recap,
    load_pedagogy_graph,
    order_pairs_by_curriculum,
)

logger = logging.getLogger(__name__)


DEFAULT_SEED = 17  # Arbitrary but stable; stage adds chunk-index for variety.


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
    rejected_reasons: Dict[str, int] = field(default_factory=dict)
    # Wave 77: stratified-sampling additions. None when stratification
    # not active, so legacy callers keep the same payload shape.
    misconception_dpo_pairs_emitted: int = 0
    stratify_dimensions: List[str] = field(default_factory=list)
    stratify_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    capped_at_max_pairs: bool = False
    max_pairs_cap: Optional[int] = None
    difficulty_curriculum: bool = False
    # Wave 79 Worker B: prerequisite-aware curriculum mode.
    curriculum_from_graph: bool = False
    prereq_windowed: bool = False
    prereq_context_tokens: int = DEFAULT_PREREQ_CONTEXT_TOKENS
    cycles_broken_count: int = 0
    pairs_without_concepts: int = 0
    concepts_without_pairs_count: int = 0
    pairs_with_prereq_recap: int = 0
    source_grounded_pairs: int = 0
    instruction_variants_per_chunk: int = 1
    # Wave 111 / Phase E: budget telemetry surfaced to callers.
    capped_at_max_dispatches: bool = False
    dispatched_count: int = 0
    cache_hits_count: int = 0
    # Audit 2026-04-30: KG-metadata + violation-detection generators.
    kg_metadata_pairs_emitted: int = 0
    violation_pairs_emitted: int = 0
    # Wave 124 (audit 2026-04-30 follow-up): abstention +
    # schema-translation generators. cc07cc76 hallucination_rate=0.63
    # was driven by zero abstention pairs + zero schema-to-English
    # bridge pairs; counters here surface the cohort sizes for the
    # post-run pilot report and the audit script.
    abstention_pairs_emitted: int = 0
    schema_translation_pairs_emitted: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "chunks_eligible": self.chunks_eligible,
            "chunks_skipped_no_lo": self.chunks_skipped_no_lo,
            "instruction_pairs_emitted": self.instruction_pairs_emitted,
            "instruction_pairs_rejected": self.instruction_pairs_rejected,
            "preference_pairs_emitted": self.preference_pairs_emitted,
            "preference_pairs_rejected": self.preference_pairs_rejected,
            "rejected_reasons": dict(self.rejected_reasons),
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
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {chunks_path}")
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def _eligible(chunk: Dict[str, Any]) -> bool:
    return bool(chunk.get("learning_outcome_refs")) and bool(chunk.get("id") or chunk.get("chunk_id"))


# ---------------------------------------------------------------------------
# Wave 77: stratified-sampling + LibV2-archive helpers
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

    Accepts both the canonical slug (``rdf-shacl-550``) and the doubled-up
    form some archival runs produce (``rdf-shacl-550-rdf-shacl-550``). The
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
    """Wave 120: smoke-mode chunk sampler.

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

    Returns None when either side is empty. Wave 112 Task 6: the silent-drop
    path now emits a ``misconception_pair_skipped`` audit event via
    ``capture.log_decision`` so a corpus rebuild that quietly loses a
    property family is still visible in the decision-capture stream.
    ``capture`` is optional only so legacy unit-tests that exercise this
    helper in isolation still work — every production call site (Wave 77
    augmentation loop in ``run_synthesis``) passes one in.
    """
    from Trainforge.generators.preference_factory import _misconception_id

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
    # Wave 95: use the misconception's OWN bloom_level (not the chunk's)
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
        primary_concept = str(tags[0]).replace("-", " ").replace("_", " ")
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


_INSTRUCTION_PROMPT_FRAMES = (
    "{prompt}",
    "For an RDF/SHACL learner, {prompt_lc}",
    "Give a source-grounded answer: {prompt}",
)


def _apply_instruction_variant(pair: Dict[str, Any], variant_index: int) -> None:
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
    candidate = frame.format(
        prompt=prompt,
        prompt_lc=prompt[:1].lower() + prompt[1:],
    )
    # Leave room for the citation instruction appended later.
    if len(candidate) <= 360:
        pair["prompt"] = candidate


def _update_dataset_config(
    dataset_config_path: Path,
    stats: SynthesisStats,
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
    config.setdefault("synthesis", {})
    config["synthesis"]["last_run"] = stats.as_dict()

    tmp = dataset_config_path.with_suffix(dataset_config_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    tmp.replace(dataset_config_path)
    return config


# ---------------------------------------------------------------------------
# Decision-capture helpers
# ---------------------------------------------------------------------------

def _last_event_id(capture: DecisionCapture) -> str:
    """Return the event_id of the most recent decision written via ``capture``.

    ``DecisionCapture.log_decision`` appends to ``capture.decisions``; we pull
    ``event_id`` off the tail.

    Wave 112 Task 1: this used to fall back to ``""`` when ``capture.decisions``
    was empty. Empty-string fallback then rode into the emitted JSONL as
    ``decision_capture_id: ""`` -- a schema-violating value that broke
    strict-mode pair validation downstream. Every production call site logs a
    decision before it asks for the event_id, so empty here is unambiguously a
    bug. Fail loud rather than poisoning the corpus.
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
# Stage entry point
# ---------------------------------------------------------------------------

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
) -> SynthesisStats:
    """Run the full synthesis stage for one course output directory.

    Args:
        corpus_dir: The course output directory (NOT the inner ``corpus/``).
            This is the dir that contains ``corpus/chunks.jsonl`` and
            ``training_specs/``.
        course_code: Course code, e.g. ``"SAMPLE_101"``. Used for decision capture.
        provider: Synthesis provider; ``"mock"`` (default) is the only one wired.
        seed: Base seed. Each chunk's effective seed is ``seed + chunk_index``.
        capture: Optional pre-built DecisionCapture. If None, one is created
            for the ``synthesize-training`` phase and saved at end of run.

    Wave 77 keyword-only additions (all default-off so legacy callers --
    process_course.py, MCP synthesize_training tool, the textbook_to_course
    pipeline phase -- keep their existing behaviour):

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

    Returns:
        :class:`SynthesisStats` with counts.
    """
    if provider == "claude_session" and dispatcher is None:
        raise RuntimeError(
            "--provider claude_session requires a LocalDispatcher to be "
            "supplied. Invoke via the workflow runner ('ed4all run "
            "trainforge_train ...') or the synthesize_training MCP tool, "
            "both of which inject a dispatcher. Standalone CLI invocation "
            "has no Claude Code session to dispatch to."
        )

    corpus_dir = Path(corpus_dir)
    chunks_path = corpus_dir / "corpus" / "chunks.jsonl"
    if output_dir is not None:
        training_specs_dir = Path(output_dir)
    else:
        training_specs_dir = corpus_dir / "training_specs"
    training_specs_dir.mkdir(parents=True, exist_ok=True)

    # Wave 120: smoke modes route JSONL outputs to ``smoke_*`` siblings
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

    # Wave 116: incremental sidecar write. Each emitted instruction /
    # preference pair is appended to a ``.jsonl.in_progress`` sibling
    # file with ``flush()`` after every write so an operator can
    # ``tail -f`` the synthesis run and so a killed run leaves
    # inspectable artifacts on disk. The atomic final ``_write_jsonl``
    # is unchanged; sidecars are unlinked on a clean exit and preserved
    # on a ``SynthesisBudgetExceeded`` early-exit (or any other
    # exception that propagates out) for postmortem.
    instruction_progress = instruction_out.with_suffix(".jsonl.in_progress")
    preference_progress = preference_out.with_suffix(".jsonl.in_progress")
    for sidecar in (instruction_progress, preference_progress):
        if sidecar.exists() and sidecar.stat().st_size > 0:
            logger.warning(
                "Wave 116: overwriting stale sidecar from a prior killed run: %s",
                sidecar,
            )
    instruction_progress.parent.mkdir(parents=True, exist_ok=True)
    inst_progress_fh = instruction_progress.open("w", encoding="utf-8")
    pref_progress_fh = preference_progress.open("w", encoding="utf-8")

    # Wave 117 / 119: load property manifest once. The manifest gates
    # ALL pilot-report writes (in-flight and final). pilot_report_every
    # only governs the in-flight cadence — the final write fires
    # whenever a manifest exists, regardless of pilot_report_every, so
    # an operator who set --pilot-report-every 0 still gets the
    # post-run summary on disk.
    pilot_manifest = None
    # Wave 120: smoke modes write to a sidecar so the canonical
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
    # Wave 120: smoke modes scale every property's ``min_pairs`` floor
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

    chunks = _read_chunks(chunks_path)
    # Wave 120: smoke modes. "deterministic" forces provider=mock and
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

    # Wave 79 Worker B: load the pedagogy graph eagerly when curriculum mode
    # is active so a missing graph fails loud instead of silently degrading
    # to chunk-id ordering. The build itself is cheap (sub-1k concept dict
    # build + Kahn's pass) so doing it once here is fine.
    #
    # Wave 91 Action B: graph is now REQUIRED by default. Workflow runs
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

    # Wave 107: construct the claude_session paraphrase provider once per
    # run. The factory layer dispatches to whichever object is passed via
    # paraphrase_provider when provider != "mock". Anthropic stays lazily
    # constructed inside the factory so its API-key precondition only fires
    # when there's actually an eligible chunk to paraphrase.
    paraphrase_provider: Optional[Any] = None
    if provider == "claude_session":
        from Trainforge.generators._claude_session_provider import (
            ClaudeSessionProvider,
        )
        # Wave 110 / Phase D: default telemetry_path under training_specs/.
        # Wave 111 / Phase E: fall back to training_specs_dir even when no
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
        # Wave 113 prep: ToS-clean OSS-teacher paraphrase via Together
        # AI's hosted models. HTTP-driven (no SDK dependency); session-
        # budget tracking is unnecessary because the provider is paid-
        # per-call rather than rate-limited per Claude session.
        from Trainforge.generators._together_provider import (
            TogetherSynthesisProvider,
        )
        paraphrase_provider = TogetherSynthesisProvider(capture=capture)
    elif provider == "local":
        # Wave 113: third synthesis path — a local OpenAI-compatible
        # model server (Ollama / vLLM / llama.cpp / LM Studio). Same
        # HTTP wire shape as Together, so the provider subclasses
        # ``TogetherSynthesisProvider`` and only overrides the
        # endpoint / model / auth-required hooks. Zero cost per call
        # after hardware setup, zero ToS exposure (fully offline /
        # air-gapped friendly). Like ``together``, no session-budget
        # tracking — the provider is HTTP-driven, not Claude-session
        # rate-limited.
        from Trainforge.generators._local_provider import (
            LocalSynthesisProvider,
        )
        # Wave 120: smoke-paraphrase caps parse retries at 1 so the
        # property-heavy stratified sample doesn't compound retry cost
        # × 20 chunks into an unbounded wall time. Production
        # (smoke_mode='none') keeps the default budget.
        local_kwargs: Dict[str, Any] = {"capture": capture}
        if smoke_mode == "paraphrase":
            local_kwargs["max_parse_retries"] = 1
        paraphrase_provider = LocalSynthesisProvider(**local_kwargs)

    instruction_records: List[Dict[str, Any]] = []
    preference_records: List[Dict[str, Any]] = []

    # Wave 111 / Phase E: budget-exceeded sentinel — hoisted above the
    # try-body in Wave 116 so the finally-block can reference it
    # safely even if an exception propagates before the loop assigns
    # it. Imported eagerly so the symbol exists in the finally scope.
    from Trainforge.generators._session_budget import (
        SynthesisBudgetExceeded as _SBE,
    )
    _budget_exhausted_exc: Optional[_SBE] = None

    # Wave 116: gate sidecar deletion on a clean exit. The flag is
    # only set True after the entire try-body completes without any
    # exception (budget-exceeded or otherwise). The finally block
    # checks both this flag AND ``_budget_exhausted_exc is None`` so
    # an exception that propagates past the try-body leaves sidecars
    # in place for postmortem inspection.
    clean_exit = False

    try:
        # Log a stage-start decision so the capture file is never empty even if
        # the corpus contains zero eligible chunks.
        capture.log_decision(
            decision_type="instruction_pair_synthesis",
            decision=(
                f"Starting instruction/preference synthesis over {len(chunks)} chunks "
                f"for course '{course_code}' using provider='{provider}' seed={seed}."
            ),
            rationale=(
                "Synthesizing SFT and DPO training pairs from enriched chunks produces a "
                "training corpus that is both LO-aligned and bloom-aware. Pairs are "
                "generated deterministically so a course regenerated later is stable and "
                "reproducible for downstream fine-tuning."
            ),
            alternatives_considered=[
                {
                    "option": "emit-only-SFT",
                    "reason_rejected": "loses misconception signal that DPO encodes",
                },
                {
                    "option": "emit-only-DPO",
                    "reason_rejected": "SFT pairs still needed for instruction tuning",
                },
            ],
        )

        # Wave 77: split chunk traversal into "count eligible" and
        # "iterate emit-order" so stratified sampling can reorder the
        # emit-order without changing the eligibility tally.
        eligible_chunks: List[Tuple[int, Dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
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

        # Effective per-artifact cap. None -> unlimited. We apply it to
        # instruction and preference outputs independently so a request for
        # `--max-pairs 50` produces at most 50 of each (matches the tests'
        # expectation that capping is per-file, not the combined total).
        per_artifact_cap = max_pairs

        # Wave 119: warn pre-flight when --max-pairs will clip the run
        # before all eligible chunks are visited. This is the failure
        # mode that bit Wave 118's first 14B rerun: a 30-pair cap on a
        # 295-chunk corpus stopped at chunk 30, never visiting any of
        # the property-bearing chunks at index 46+. Surfacing it here
        # gives the operator a chance to abort and re-launch without
        # the cap before paying for the full run.
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

        # Wave 111 / Phase E: graceful SynthesisBudgetExceeded handling.
        # When the claude_session provider hits its dispatch cap mid-loop,
        # we stop emitting + persist whatever we have so far so the
        # caller can write a pilot_progress.json snapshot and return
        # SynthesisStats with capped_at_max_dispatches=True.
        # ``_SBE`` and ``_budget_exhausted_exc`` are hoisted above the
        # try-block (Wave 116) so the finally-block can reference them.

        # Wave 117: count chunks fully processed (post both instruction
        # + preference branches) so the periodic pilot_report.md writer
        # snapshots a consistent view of all pairs from each chunk.
        chunks_processed_counter = 0

        # Wave 122 follow-up: factory-side dedupe. The audit's zero-
        # tolerance ``duplicates`` gate flags any cross-chunk paraphrase
        # collision; tracking emitted prompts and rejecting the second
        # occurrence keeps the gate clean without re-running the
        # paraphrase. Distinct sets per artefact so an instruction
        # prompt can legitimately match a preference prompt.
        emitted_inst_prompts: set = set()
        emitted_pref_prompts: set = set()

        for idx, chunk in iter_chunks:
            if _budget_exhausted_exc is not None:
                break
            # Wave 120: detect property surface forms in this chunk so
            # the paraphrase provider preserves them verbatim. None ->
            # no manifest loaded; empty list -> chunk doesn't reference
            # any declared property.
            chunk_preserve_tokens = (
                pilot_manifest.detect_surface_forms(str(chunk.get("text") or ""))
                if pilot_manifest is not None
                else []
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
                try:
                    inst_result = synthesize_instruction_pair(
                        chunk,
                        seed=pair_seed,
                        provider=provider,
                        paraphrase_provider=paraphrase_provider,
                        preserve_tokens=chunk_preserve_tokens or None,
                    )
                except _SBE as exc:
                    _budget_exhausted_exc = exc
                    break
                if inst_result.pair is None:
                    stats.instruction_pairs_rejected += 1
                    reason = inst_result.quality.get("reason") or "gate_failed"
                    stats.rejected_reasons[f"instruction:{reason}"] = (
                        stats.rejected_reasons.get(f"instruction:{reason}", 0) + 1
                    )
                else:
                    # REC-VOC-03 Phase 2 (Worker T): opt-in content_type enforcement
                    # against ChunkType enum. Flag off -> no-op; flag on -> fail-closed.
                    # Matches Worker I's TRAINFORGE_VALIDATE_CHUNKS pattern at
                    # process_course.py:1987-2009.
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
                    _apply_instruction_variant(inst_result.pair, variant_index)
                    # Wave 120: audit-log when paraphrase preservation
                    # failed and the deterministic draft was used. Lets
                    # post-hoc analysis identify chunks where the LLM
                    # consistently dropped technical CURIEs.
                    if inst_result.pair.get("paraphrase_fallback_reason"):
                        capture.log_decision(
                            decision_type="surface_form_preservation_fallback",
                            decision=(
                                f"Fell back to deterministic draft for "
                                f"instruction pair on chunk "
                                f"{inst_result.pair['chunk_id']}; paraphrase "
                                f"dropped surface form(s) "
                                f"{chunk_preserve_tokens}."
                            ),
                            rationale=(
                                f"Provider '{provider}' could not preserve required "
                                f"property surface forms after retry exhaustion; "
                                f"using the deterministic template draft preserves "
                                f"property coverage at the cost of paraphrase "
                                f"variety on chunks containing "
                                f"{chunk_preserve_tokens}."
                            ),
                            context=f"chunk_id={inst_result.pair['chunk_id']}",
                        )
                    # Wave 122 follow-up: cross-chunk prompt-collision
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
                    if _attach_source_grounding(inst_result.pair, chunk):
                        stats.source_grounded_pairs += 1
                    instruction_records.append(inst_result.pair)
                    emitted_inst_prompts.add(final_prompt)
                    stats.instruction_pairs_emitted += 1
                    # Wave 116: mirror to .in_progress sidecar with
                    # flush() so ``tail -f`` and post-kill inspection
                    # see this pair without waiting on OS buffers.
                    inst_progress_fh.write(
                        json.dumps(
                            inst_result.pair,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    inst_progress_fh.flush()

            # --- Preference pair ---
            pair_seed = seed + idx
            pref_capped = (
                per_artifact_cap is not None
                and stats.preference_pairs_emitted >= per_artifact_cap
            )
            if pref_capped:
                stats.capped_at_max_pairs = True
            else:
                try:
                    pref_result = synthesize_preference_pair(
                        chunk,
                        seed=pair_seed,
                        provider=provider,
                        paraphrase_provider=paraphrase_provider,
                        preserve_tokens=chunk_preserve_tokens or None,
                    )
                except _SBE as exc:
                    _budget_exhausted_exc = exc
                    break
                if pref_result.pair is None:
                    stats.preference_pairs_rejected += 1
                    reason = pref_result.quality.get("reason") or "gate_failed"
                    stats.rejected_reasons[f"preference:{reason}"] = (
                        stats.rejected_reasons.get(f"preference:{reason}", 0) + 1
                    )
                else:
                    if pref_result.pair.get("paraphrase_fallback_reason"):
                        capture.log_decision(
                            decision_type="surface_form_preservation_fallback",
                            decision=(
                                f"Fell back to deterministic draft for "
                                f"preference pair on chunk "
                                f"{pref_result.pair['chunk_id']}; paraphrase "
                                f"dropped surface form(s) "
                                f"{chunk_preserve_tokens} from chosen field."
                            ),
                            rationale=(
                                f"Provider '{provider}' could not preserve required "
                                f"property surface forms in the chosen completion "
                                f"after retry exhaustion; deterministic draft "
                                f"preserves property coverage."
                            ),
                            context=f"chunk_id={pref_result.pair['chunk_id']}",
                        )
                    # Wave 122 follow-up: cross-chunk dedupe (preference).
                    # Nested ``if`` rather than ``continue`` so the
                    # outer chunk loop still falls through to the
                    # pilot_report progress block below.
                    final_pref_prompt = pref_result.pair.get("prompt", "")
                    if final_pref_prompt in emitted_pref_prompts:
                        stats.preference_pairs_rejected += 1
                        stats.rejected_reasons["preference:duplicate_prompt"] = (
                            stats.rejected_reasons.get("preference:duplicate_prompt", 0) + 1
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
                        if _attach_source_grounding(pref_result.pair, chunk):
                            stats.source_grounded_pairs += 1
                        preference_records.append(pref_result.pair)
                        emitted_pref_prompts.add(final_pref_prompt)
                        stats.preference_pairs_emitted += 1
                        # Wave 116: mirror to .in_progress sidecar.
                        pref_progress_fh.write(
                            json.dumps(
                                pref_result.pair,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        pref_progress_fh.flush()

            # Wave 117: every N processed chunks, regenerate the
            # in-flight pilot_report.md so the operator running a
            # multi-hour rebuild has live property-coverage /
            # template-distribution visibility. Atomic tmp-and-rename
            # write keeps a concurrent ``cat`` / ``less`` from
            # observing a half-written file.
            chunks_processed_counter += 1
            if (
                pilot_manifest is not None
                and pilot_report_every > 0
                and chunks_processed_counter % pilot_report_every == 0
            ):
                from Trainforge.scripts.pilot_report_helpers import (
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

        # --- Wave 77: misconception -> DPO pair augmentation -----------------
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
                    # Wave 116: mirror to .in_progress sidecar.
                    pref_progress_fh.write(
                        json.dumps(pair, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
                    pref_progress_fh.flush()

        # Wave 111 / Phase E: surface budget telemetry on stats whether
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
            # Wave 116: preserve sidecars on budget-exceeded so the
            # operator can inspect partial output and re-run with a
            # higher cap to resume from the cache.
            logger.warning(
                "Wave 116: synthesis stopped early; sidecars preserved at "
                "%s and %s",
                instruction_progress, preference_progress,
            )

        # Wave 117: finalize pilot_report.md (in_flight=False) on every
        # exit path (normal completion OR budget-cap break). The JSONL
        # is the source of truth; this is the human-readable companion
        # artifact.
        if pilot_manifest is not None:
            from Trainforge.scripts.pilot_report_helpers import (
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
        # Wave 79 Worker B: prerequisite-aware curriculum reordering + recap
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
        if curriculum_ctx is not None and prereq_windowed:
            for rec in instruction_records:
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

        # Audit 2026-04-30: append KG-metadata pairs (yes/no membership
        # probes mirroring faithfulness._RELATION_TEMPLATES). Closes the
        # zero-KG-metadata-recall regression in the cc07cc76 corpus —
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
                kg_pairs, kg_stats = generate_kg_metadata_pairs(
                    ped_payload,
                    capture=capture,
                    max_pairs=int(kg_metadata_max_pairs),
                    seed=seed,
                )
                instruction_records.extend(kg_pairs)
                stats.kg_metadata_pairs_emitted = kg_stats.pairs_emitted
                stats.instruction_pairs_emitted += kg_stats.pairs_emitted
                logger.info(
                    "Audit 2026-04-30: appended %d KG-metadata pairs "
                    "(positives=%d, negatives=%d, capped=%s) sourced from %s",
                    kg_stats.pairs_emitted,
                    kg_stats.positives_emitted,
                    kg_stats.negatives_emitted,
                    kg_stats.capped_at_max_pairs,
                    ped_path,
                )

        # Audit 2026-04-30: append violation-detection pairs (pyshacl-
        # oracle-verified (graph, shape, valid?, reason) tuples). Closes
        # the zero-negative-grounding regression — the corpus must teach
        # the model to refuse a graph that violates a shape.
        if with_violation_detection:
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
                instruction_records.extend(vio_pairs)
                stats.violation_pairs_emitted = vio_stats.pairs_emitted
                stats.instruction_pairs_emitted += vio_stats.pairs_emitted
                logger.info(
                    "Audit 2026-04-30: appended %d violation-detection "
                    "pairs (valid=%d, invalid=%d, oracle_disagreements=%d)",
                    vio_stats.pairs_emitted,
                    vio_stats.valid_pairs,
                    vio_stats.invalid_pairs,
                    vio_stats.oracle_disagreements,
                )

        # Wave 124 (audit 2026-04-30 follow-up): append abstention
        # probes ('the source does not establish X'). Closes the
        # cc07cc76 hallucination_rate=0.63 — the eval harness probes
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
                instruction_records.extend(ab_pairs)
                stats.abstention_pairs_emitted = ab_stats.pairs_emitted
                stats.instruction_pairs_emitted += ab_stats.pairs_emitted
                logger.info(
                    "Wave 124: appended %d abstention pairs (chunks_with_silent=%d, "
                    "skipped_no_concepts=%d, capped=%s) from %s",
                    ab_stats.pairs_emitted,
                    ab_stats.chunks_with_silent,
                    ab_stats.chunks_skipped_no_concepts,
                    ab_stats.capped_at_max_pairs,
                    ped_path,
                )

        # Wave 124 (audit 2026-04-30 follow-up): append schema-to-
        # English bridge pairs. Walks the property manifest's surface
        # forms (sh:datatype, rdfs:subClassOf, ...) and emits one
        # definition + one usage pair per CURIE. Closes the schema-
        # to-English gap behind faithfulness=0.37.
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
                instruction_records.extend(st_pairs)
                stats.schema_translation_pairs_emitted = st_stats.pairs_emitted
                stats.instruction_pairs_emitted += st_stats.pairs_emitted
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

        _write_jsonl(instruction_out, instruction_records)
        _write_jsonl(preference_out, preference_records)
        _update_dataset_config(dataset_config_path, stats)
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

        # Wave 116: try-body completed without raising. Mark the run
        # clean so the finally-block deletes the sidecars. A
        # SynthesisBudgetExceeded run reaches this line too (it is
        # caught above and produces ``pilot_progress.json``), so we
        # additionally check ``_budget_exhausted_exc`` in the finally
        # to keep the sidecars on cap-exhausted runs.
        clean_exit = True

    finally:
        # Wave 116: always close sidecar file handles, even on
        # exception. Delete only on a fully clean exit (no exception
        # propagated AND no budget cap hit). On budget-exceeded or
        # any other early exit, the sidecars stay on disk so the
        # operator has inspectable partial output.
        try:
            inst_progress_fh.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to close instruction sidecar: %s", e)
        try:
            pref_progress_fh.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to close preference sidecar: %s", e)
        if clean_exit and _budget_exhausted_exc is None:
            instruction_progress.unlink(missing_ok=True)
            preference_progress.unlink(missing_ok=True)

        if owns_capture:
            try:
                capture.save()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Failed to save decision capture: %s", e)

    return stats


# ---------------------------------------------------------------------------
# Wave 77: LibV2-archive entry path
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
) -> SynthesisStats:
    """Run synthesis directly against a LibV2 course archive.

    Locates the course directory under ``LibV2/courses/<slug>/``, which
    already contains ``corpus/chunks.jsonl`` (the same shape the Trainforge
    pipeline emits) and ``objectives.json``. This avoids re-running the
    pipeline when the only goal is to (re-)synthesize training pairs from
    an already-archived corpus -- e.g. when iterating on stratification or
    misconception-DPO emission.

    Args:
        slug: LibV2 course slug, e.g. ``"rdf-shacl-550"``.
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
    # Either --corpus (legacy: Trainforge output dir) or --slug (Wave 77:
    # LibV2 archive entry path). At least one must be provided.
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
            "Course code for decision capture, e.g. SAMPLE_101. "
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
            "default model qwen2.5:14b-instruct-q4_K_M, override via "
            "LOCAL_SYNTHESIS_MODEL). API key optional — local servers ignore "
            "auth. Zero cost per call, zero ToS exposure; tradeoff is local "
            "hardware requirement."
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
    # Wave 77 additions
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
    # Wave 79 Worker B (Wave 91: now default-on; --no-graph to opt out).
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
    # Wave 91 Action B: opt-out flag for legacy corpora that lack a
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
    # Wave 120: smoke modes. Stratified ~20-chunk sample so every
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
    # Audit 2026-04-30: KG-metadata + violation-detection generators.
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
            "Closes the zero-KG-metadata-recall gap behind the cc07cc76 "
            "adapter's faithfulness=0.37 / negative_grounding=0 result."
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
            "production rebuilds (e.g. 350 for the cc07cc76 retrain). "
            "Truncation is family-balanced round-robin across surface "
            "forms so every form keeps representation up to the cap."
        ),
    )
    # Wave 124 (audit 2026-04-30 follow-up): abstention +
    # schema-translation generators. Both are off by default, parallel
    # to --with-kg-metadata / --with-violation-detection. Closes the
    # cc07cc76 hallucination_rate=0.63 + zero schema-to-English bridge
    # gaps the eval harness probes for.
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
            "Closes the cc07cc76 hallucination_rate=0.63 regression."
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
            "gap behind the cc07cc76 adapter's faithfulness=0.37."
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
    return p


def main(args: Optional[argparse.Namespace] = None) -> SynthesisStats:
    if args is None:
        args = build_parser().parse_args()

    stratify_dims = _parse_stratify_arg(getattr(args, "stratify", ""))
    include_dpo = bool(getattr(args, "include_dpo_from_misconceptions", False))
    diff_curriculum = bool(getattr(args, "difficulty_curriculum", False))
    max_pairs_cap = getattr(args, "max_pairs", None)
    output_dir = Path(args.output) if getattr(args, "output", None) else None
    # Wave 91 Action B: graph-required by default; --no-graph opts out.
    no_graph = bool(getattr(args, "no_graph", False))
    curriculum_graph = bool(getattr(args, "curriculum_from_graph", True))
    if no_graph:
        curriculum_graph = False
    prereq_windowed = bool(getattr(args, "prereq_windowed", False))
    prereq_ctx_tokens = int(
        getattr(args, "prereq_context_tokens", DEFAULT_PREREQ_CONTEXT_TOKENS)
    )
    pedagogy_path = Path(args.pedagogy_graph) if getattr(args, "pedagogy_graph", None) else None
    # Wave 110 / Phase D: --max-dispatches is meaningful only with claude_session.
    max_dispatches = getattr(args, "max_dispatches", None)
    if max_dispatches is not None and args.provider != "claude_session":
        raise SystemExit(
            "--max-dispatches is only meaningful with --provider claude_session"
        )
    # Wave 117: incremental pilot_report.md writes during the chunk loop.
    pilot_report_every = int(getattr(args, "pilot_report_every", 20) or 0)
    # Wave 120: smoke modes (mutex group, only one can be set).
    if getattr(args, "smoke_deterministic", False):
        smoke_mode = "deterministic"
    elif getattr(args, "smoke_paraphrase", False):
        smoke_mode = "paraphrase"
    else:
        smoke_mode = "none"

    # Audit 2026-04-30: KG-metadata + violation-detection generators.
    with_kg_metadata = bool(getattr(args, "with_kg_metadata", False))
    kg_metadata_max_pairs = int(getattr(args, "kg_metadata_max_pairs", 2000))
    with_violation_detection = bool(
        getattr(args, "with_violation_detection", False)
    )
    violation_shapes_glob = getattr(args, "violation_shapes_glob", None)
    # Wave 125a: optional cap on violation-detection emit count.
    violation_detection_max_pairs_arg = getattr(
        args, "violation_detection_max_pairs", None,
    )
    violation_detection_max_pairs = (
        int(violation_detection_max_pairs_arg)
        if violation_detection_max_pairs_arg is not None
        else None
    )
    # Wave 124: abstention + schema-translation generators.
    with_abstention = bool(getattr(args, "with_abstention", False))
    abstention_max_pairs = int(getattr(args, "abstention_max_pairs", 1000))
    with_schema_translation = bool(
        getattr(args, "with_schema_translation", False)
    )
    schema_translation_max_pairs = int(
        getattr(args, "schema_translation_max_pairs", 50)
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
    # Wave 111 / Phase E: surface session-budget telemetry on
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
