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
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Make project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture  # noqa: E402

from Trainforge.generators.instruction_factory import (  # noqa: E402
    synthesize_instruction_pair,
)
from Trainforge.generators.preference_factory import (  # noqa: E402
    synthesize_preference_pair,
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

def run_synthesis(
    corpus_dir: Path,
    course_code: str,
    provider: str = "mock",
    seed: int = DEFAULT_SEED,
    capture: Optional[DecisionCapture] = None,
) -> SynthesisStats:
    """Run the full synthesis stage for one course output directory.

    Args:
        corpus_dir: The course output directory (NOT the inner ``corpus/``).
            This is the dir that contains ``corpus/chunks.jsonl`` and
            ``training_specs/``.
        course_code: Course code, e.g. ``"WCAG_201"``. Used for decision capture.
        provider: Synthesis provider; ``"mock"`` (default) is the only one wired.
        seed: Base seed. Each chunk's effective seed is ``seed + chunk_index``.
        capture: Optional pre-built DecisionCapture. If None, one is created
            for the ``synthesize-training`` phase and saved at end of run.

    Returns:
        :class:`SynthesisStats` with counts.
    """
    corpus_dir = Path(corpus_dir)
    chunks_path = corpus_dir / "corpus" / "chunks.jsonl"
    training_specs_dir = corpus_dir / "training_specs"
    training_specs_dir.mkdir(parents=True, exist_ok=True)

    instruction_out = training_specs_dir / "instruction_pairs.jsonl"
    preference_out = training_specs_dir / "preference_pairs.jsonl"
    dataset_config_path = training_specs_dir / "dataset_config.json"

    chunks = _read_chunks(chunks_path)
    stats = SynthesisStats(chunks_total=len(chunks))

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

        for idx, chunk in enumerate(chunks):
            if not _eligible(chunk):
                stats.chunks_skipped_no_lo += 1
                continue
            stats.chunks_eligible += 1

            pair_seed = seed + idx

            # --- Instruction pair ---
            inst_result = synthesize_instruction_pair(chunk, seed=pair_seed, provider=provider)
            if inst_result.pair is None:
                stats.instruction_pairs_rejected += 1
                reason = inst_result.quality.get("reason") or "gate_failed"
                stats.rejected_reasons[f"instruction:{reason}"] = (
                    stats.rejected_reasons.get(f"instruction:{reason}", 0) + 1
                )
            else:
                capture.log_decision(
                    decision_type="instruction_pair_synthesis",
                    decision=(
                        f"Emit instruction pair for chunk {inst_result.pair['chunk_id']} "
                        f"(template={inst_result.template_id}, bloom={inst_result.pair['bloom_level']})."
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
                instruction_records.append(inst_result.pair)
                stats.instruction_pairs_emitted += 1

            # --- Preference pair ---
            pref_result = synthesize_preference_pair(chunk, seed=pair_seed, provider=provider)
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
                preference_records.append(pref_result.pair)
                stats.preference_pairs_emitted += 1

        # --- Persist artifacts (deterministic ordering: by chunk_id) ---
        instruction_records.sort(key=lambda r: (r["chunk_id"], r.get("seed", 0)))
        preference_records.sort(key=lambda r: (r["chunk_id"], r.get("seed", 0)))

        _write_jsonl(instruction_out, instruction_records)
        _write_jsonl(preference_out, preference_records)
        _update_dataset_config(dataset_config_path, stats)

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
# CLI (standalone invocation)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Synthesize SFT and DPO training pairs from an already-processed "
            "Trainforge course output directory."
        )
    )
    p.add_argument(
        "--corpus",
        required=True,
        help="Course output directory (the one containing corpus/ and training_specs/).",
    )
    p.add_argument(
        "--course-code",
        required=True,
        help="Course code for decision capture, e.g. WCAG_201.",
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
    return p


def main(args: Optional[argparse.Namespace] = None) -> SynthesisStats:
    if args is None:
        args = build_parser().parse_args()

    stats = run_synthesis(
        corpus_dir=Path(args.corpus),
        course_code=args.course_code,
        provider=args.provider,
        seed=args.seed,
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
