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


def _curriculum_sort_key(chunk: Dict[str, Any]) -> Tuple[int, str]:
    diff = str(chunk.get("difficulty") or "").lower()
    rank = _DIFFICULTY_ORDER.get(diff, len(_DIFFICULTY_ORDER))
    cid = str(chunk.get("id") or chunk.get("chunk_id") or "")
    return (rank, cid)


def _build_misconception_dpo_pair(
    chunk: Dict[str, Any],
    misconception: Dict[str, Any],
    pair_index: int,
) -> Optional[Dict[str, Any]]:
    """Convert a single (misconception, correction) entry into a DPO pair.

    Returns None when either side is empty -- silently skipping malformed
    misconception entries is preferable to crashing the run.
    """
    from Trainforge.generators.preference_factory import _misconception_id

    mc_text_for_id = str(misconception.get("misconception", "")).strip()
    correction_for_id = str(misconception.get("correction", "")).strip()
    if not mc_text_for_id or not correction_for_id:
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
    ``event_id`` off the tail. Empty-string fallback if nothing logged.
    """
    if capture.decisions:
        return str(capture.decisions[-1].get("event_id", ""))
    return ""


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
    corpus_dir = Path(corpus_dir)
    chunks_path = corpus_dir / "corpus" / "chunks.jsonl"
    if output_dir is not None:
        training_specs_dir = Path(output_dir)
    else:
        training_specs_dir = corpus_dir / "training_specs"
    training_specs_dir.mkdir(parents=True, exist_ok=True)

    instruction_out = training_specs_dir / "instruction_pairs.jsonl"
    preference_out = training_specs_dir / "preference_pairs.jsonl"
    dataset_config_path = training_specs_dir / "dataset_config.json"

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

    instruction_records: List[Dict[str, Any]] = []
    preference_records: List[Dict[str, Any]] = []

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
                "emit-only-SFT (rejected: loses misconception signal that DPO encodes)",
                "emit-only-DPO (rejected: SFT pairs still needed for instruction tuning)",
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

        for idx, chunk in iter_chunks:
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
                inst_result = synthesize_instruction_pair(
                    chunk, seed=pair_seed, provider=provider
                )
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
                    stats.instruction_pairs_emitted += 1

            # --- Preference pair ---
            pair_seed = seed + idx
            pref_capped = (
                per_artifact_cap is not None
                and stats.preference_pairs_emitted >= per_artifact_cap
            )
            if pref_capped:
                stats.capped_at_max_pairs = True
            else:
                pref_result = synthesize_preference_pair(
                    chunk, seed=pair_seed, provider=provider
                )
                if pref_result.pair is None:
                    stats.preference_pairs_rejected += 1
                    reason = pref_result.quality.get("reason") or "gate_failed"
                    stats.rejected_reasons[f"preference:{reason}"] = (
                        stats.rejected_reasons.get(f"preference:{reason}", 0) + 1
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
                    stats.preference_pairs_emitted += 1

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
                    pair = _build_misconception_dpo_pair(chunk, mc, mc_index)
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

    finally:
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
        choices=["mock", "anthropic"],
        help="Synthesis provider (default: mock).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base deterministic seed (default: {DEFAULT_SEED}).",
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

    if getattr(args, "slug", None):
        stats = run_synthesis_from_libv2(
            slug=args.slug,
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

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
