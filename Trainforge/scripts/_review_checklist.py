"""Wave 137d-1 — review checklist helper.

Prevents reviewer fatigue at backfill scale by structuring review:
always-2 (definitions[0] + usage_examples[0][1]) + sample-3 (deterministic
random pick from remaining pool, seeded by curie + content hash so the
same CURIE+content always shows the same sample).

The checklist is auto-printed by the drafting CLI between the YAML
block and the operator next-steps banner. The backfill loop's YAML
slicer (``_extract_yaml_payload_from_drafting_stdout``) is updated to
cut at the first of either the checklist header or the next-steps
header so the YAML round-trips cleanly through ``yaml.safe_load``
even when the checklist sits between them.

Determinism contract: ``_seed_for(curie, entry)`` returns a stable
64-bit integer derived from a SHA-256 over a canonical JSON payload
of the entry's content fields plus the CURIE. Identical CURIE +
identical content always produces an identical sample; any content
change reshuffles. This makes review reproducible across re-runs of
the same drafting pass.

Always-review choice rationale: ``definitions[0]`` is the entry's
canonical definition (and the slot the Wave 135b force-injection
pulls into the training pair); ``usage_examples[0][1]`` is the first
usage answer (the first CURIE-anchored sentence the operator will
encounter post-injection). These two slots are load-bearing — a
semantic miss on either propagates to every paraphrase pair derived
from this CURIE.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from Trainforge.generators.schema_translation_generator import SurfaceFormData


def _truncate(text: str, n: int = 80) -> str:
    """Right-truncate ``text`` to ``n`` chars with an ellipsis suffix.

    Operator review reads the checklist on a terminal — long sentences
    wrap awkwardly. 80-char truncation keeps every line fit-on-screen
    while leaving enough text to flag obvious semantic problems.
    """
    if len(text) <= n:
        return text
    return text[:n] + "..."


def _seed_for(curie: str, entry: SurfaceFormData) -> int:
    """Deterministic 64-bit seed for the random sample.

    SHA-256 over a canonical JSON payload of the entry's content
    fields + the CURIE. Truncated to the first 8 bytes. Same CURIE +
    same content => same seed; any content change reshuffles.
    """
    payload = {
        "curie": curie,
        "definitions": list(entry.definitions),
        "usage_examples": [list(t) for t in entry.usage_examples],
        "reasoning_scenarios": [list(t) for t in entry.reasoning_scenarios],
        "pitfalls": [list(t) for t in entry.pitfalls],
        "comparison_targets": [list(t) for t in entry.comparison_targets],
        "combinations": [list(t) for t in entry.combinations],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()[:8]
    return int.from_bytes(digest, "big")


def _pick_sample(
    entry: SurfaceFormData,
    seed: int,
    k: int = 3,
) -> List[Tuple[str, int, str]]:
    """Pick ``k`` review candidates from the entry's non-anchor pool.

    Returns a list of ``(category, index, sentence)`` tuples beyond
    the always-2 anchors (``definitions[0]`` + ``usage_examples[0][1]``).

    Pool composition:
      * ``definitions[1:]`` — every definition past the first.
      * ``usage_examples[1:]`` — answer slot only (the answer is the
        load-bearing CURIE-anchored sentence; prompts are rarely the
        semantic-correctness risk).
      * ``reasoning_scenarios[*]`` — answer slot only (full pool).

    Sample is drawn by shuffling the pool with a ``random.Random``
    seeded by ``seed``, then taking the first ``k`` entries. Pool
    smaller than ``k`` returns the whole pool.
    """
    candidates: List[Tuple[str, int, str]] = []
    for i in range(1, len(entry.definitions)):
        candidates.append(("definitions", i, entry.definitions[i]))
    for i in range(1, len(entry.usage_examples)):
        candidates.append(("usage_examples", i, entry.usage_examples[i][1]))
    for i in range(len(entry.reasoning_scenarios)):
        candidates.append((
            "reasoning_scenarios",
            i,
            entry.reasoning_scenarios[i][1],
        ))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:k]


def build_review_checklist(
    curie: str,
    entry: SurfaceFormData,
    *,
    validator_score_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the operator review checklist for ``curie`` + ``entry``.

    Output structure:
      * Header banner (=== x60 + "REVIEW CHECKLIST for {curie}").
      * Always-2 review slots (definitions[0] + usage_examples[0][1]).
      * Sample-3 review slots (deterministic random pick).
      * Auto-checks block (what the validator already verified).
      * Operator decision banner.

    Args:
        curie: Target CURIE (e.g., ``sh:datatype``).
        entry: Drafted ``SurfaceFormData`` to render checklist for.
        validator_score_summary: Optional dict with auto-check
            metrics. When ``"diversity_score"`` is present, the
            checklist surfaces the actual score; otherwise it
            surfaces the 0.45 threshold.

    Returns:
        Multi-line string ready for ``print()``.
    """
    seed = _seed_for(curie, entry)
    sample = _pick_sample(entry, seed, k=3)

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"REVIEW CHECKLIST for {curie}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Always review (load-bearing):")
    if entry.definitions:
        lines.append(
            f"  [ ] definitions[0]: {_truncate(entry.definitions[0])}"
        )
    else:
        lines.append("  [ ] (no definition available)")
    if entry.usage_examples:
        lines.append(
            f"  [ ] usage_examples[0][1] (answer): "
            f"{_truncate(entry.usage_examples[0][1])}"
        )
    else:
        lines.append("  [ ] (no usage_example available)")
    lines.append("")
    lines.append(f"Random sample (deterministic, seed={seed}):")
    for category, idx, sentence in sample:
        suffix = "[1]" if category in ("usage_examples", "reasoning_scenarios") else ""
        lines.append(
            f"  [ ] {category}[{idx}]{suffix}: {_truncate(sentence)}"
        )
    if len(sample) < 3:
        for _ in range(3 - len(sample)):
            lines.append("  [ ] (no candidate available)")
    lines.append("")
    lines.append("Auto-checks ALREADY PASSED (validator):")
    lines.append("  [x] structural shape valid")
    lines.append("  [x] CURIE present verbatim in every sentence")
    lines.append("  [x] no suffix/list artifacts")
    lines.append("  [x] no placeholder leakage")
    lines.append("  [x] length bounds satisfied")
    lines.append("  [x] anchor-verb present in definitions")
    if validator_score_summary and "diversity_score" in validator_score_summary:
        d = validator_score_summary["diversity_score"]
        lines.append(
            f"  [x] diversity score = {d:.2f} (<= 0.45 max pairwise)"
        )
    else:
        lines.append("  [x] diversity score <= 0.45 max pairwise")
    lines.append("")
    lines.append(
        "Operator decision: review the 5 sentences above for SEMANTIC CORRECTNESS,"
    )
    lines.append("then choose y/n/e/q.")
    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = ["build_review_checklist"]
