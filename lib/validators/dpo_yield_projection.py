"""Bloom-ladder initiative addendum AD-02 — ``DpoYieldProjectionValidator``.

DPO-yield famine guard. The July famine failure class (see the
``docs/architecture/decision-capture.md`` precedent) is that **eligibility is not
admission**: a corpus can look "DPO-ready" at the ``synthesis_quota`` /
content-gate layer and still emit zero trainer-admissible preference pairs,
because ``Trainforge/training/compute_backend.py::is_dpo_editorial_record``
only admits a record carrying ``misconception_id`` or an editorial
``source`` — and that stamp is only ever minted from a chunk's *structured*
``misconceptions[]`` (see ``synthesis_window_contract.resolve_chunk_misconceptions``
+ ``Trainforge/CLAUDE.md`` § "Eligibility is not DPO admission"). The trainer
only discovers a below-``min_dpo_pairs`` corpus AFTER the multi-hour
paraphrase loop has already run (``dpo_fail_hard=true`` raises
``InsufficientPreferencePairsError`` post-hoc).

This module is the single source of truth for the DPO-yield PROJECTION —
walk the chunkset, recover misconception cards, and simulate exactly the
same admission chain the real synthesis run will exercise, all before any
provider dispatch:

1. :func:`Trainforge.generators.staged.window_contract.resolve_chunk_misconceptions`
   recovers each chunk's structured misconception cards (authored value wins;
   recovery reads only the chunk's own markup). Per-card Bloom rung is
   carried when ``TRAINFORGE_BLOOM_WINDOWS`` recovery populated one.
2. :func:`Trainforge.synthesis.synthesis_eligibility.focus_chunk_on_canonical_objective`
   + :func:`Trainforge.synthesis.synthesis_eligibility.pair_eligibility` (``kind=
   "instruction"``) resolve the SAME canonical-objective focus and run the
   SAME shared content/LO/evidence-window/length gates every real pair goes
   through, before the preference-specific arm.
3. :func:`Trainforge.generators.staged.micro.micro_preference_eligibility`
   simulates Arm A/B admission against the chunk's REAL text — the exact
   predicate ``synthesis_eligibility.pair_eligibility(kind="preference")``
   calls, contract-independent of ``staged-v4`` vs ``micro-v1``.
4. :func:`Trainforge.training.compute_backend.is_dpo_editorial_record` is
   called on the synthetic record a Arm-A-admitted candidate would produce
   (``source="misconception"``, the same stamp
   ``Trainforge/generators/pairs/preference.py`` mints), so the final
   "would the trainer actually admit this" count is the REAL trainer
   predicate, not a re-derived assumption.

Every predicate above is imported, never re-implemented — this module (and
``Trainforge/scripts/ops/model_dpo_yield.py``, which imports :func:`project_dpo_yield`
from here rather than duplicating it) is a pure orchestration/reporting layer
over the real admission chain.

Both the standalone CPU script and the ``dpo_yield_projection`` warning gate
share :func:`project_dpo_yield` as their single source of truth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.generation.bloom_ladder_blocks import resolve_bloom_ladder

logger = logging.getLogger(__name__)

DPO_YIELD_PROJECTION_CONTRACT_VERSION = "ed4all.dpo-yield-projection.v1"

#: Mirrors ``Trainforge.training.configs.TrainingConfig.min_dpo_pairs``
#: default — the trainer's own floor. Kept as a local literal (not an
#: import) so this CPU-only projector never pulls in the training config
#: dataclass module.
DEFAULT_MIN_DPO_PAIRS = 50

_CODE_BELOW_FLOOR = "DPO_YIELD_PROJECTION_BELOW_FLOOR"
_CODE_SKIPPED_NO_CARDS = "DPO_YIELD_PROJECTION_SKIPPED_NO_MISCONCEPTION_CARDS"
_CODE_SKIPPED_MISSING_INPUT = "DPO_YIELD_PROJECTION_SKIPPED_MISSING_INPUT"
_CODE_INVALID_CHUNKS = "DPO_YIELD_PROJECTION_INVALID_CHUNKS_INPUT"

_UNKNOWN_RUNG = "unknown"


def _normalize_rung(value: Any) -> str:
    rung = str(value or "").strip().lower()
    return rung if rung else _UNKNOWN_RUNG


def project_dpo_yield(
    chunks: Sequence[Mapping[str, Any]],
    objectives: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    min_dpo_pairs: int = DEFAULT_MIN_DPO_PAIRS,
    seed: int = 0,
) -> Dict[str, Any]:
    """Project per-course DPO yield from a real chunkset + canonical objectives.

    Walks ``chunks``, recovering each one's structured misconception cards
    (``resolve_chunk_misconceptions`` — authored value wins, recovery reads
    only the chunk's own markup) for the ``cards_found`` / ``cards_by_rung``
    tally, then re-runs the SAME focus + eligibility chain the real
    ``training_synthesis`` phase runs (see module docstring) to project how
    many chunks would actually clear ``is_dpo_editorial_record`` admission.

    ``objectives`` is the ``{lo_id: {"statement", "bloom_level", ...}}``
    canonical-objectives map (e.g.
    :func:`lib.validators.pair.objective_delivery._load_synthesized_objectives_for_w4c`
    output) — the SAME shape ``focus_chunk_on_canonical_objective`` consumes
    in production. An empty/missing map means no chunk can resolve a focus
    (mirrors the real synthesis contract: no canonical objectives, no
    focus, no pairs), so every chunk projects ineligible; callers that want
    a meaningful (non-trivially-zero) projection must supply it.

    Returns a plain dict (no dataclass — this is the wire shape both the
    CLI script's JSON emit and the gate's ``GateResult.metadata`` use
    directly):

    ``contract_version``, ``chunk_count``, ``cards_found``,
    ``cards_by_rung`` (``{rung: count}``), ``arm_a_admitted`` (chunks whose
    ``micro_preference_eligibility`` simulation came back eligible),
    ``projected_admissible`` (the subset of those that also clear
    ``is_dpo_editorial_record`` — the real trainer predicate),
    ``projected_admissible_by_rung``, ``min_dpo_pairs``, ``shortfall``
    (``max(0, min_dpo_pairs - projected_admissible)``), ``deficit`` (bool).
    """
    # Imported lazily so a caller that only wants the arithmetic (e.g. a
    # unit test driving synthetic candidates) never pays Trainforge's
    # heavier import chain, and so the CPU-only projector never imports a
    # GPU-capable module at collection time.
    from Trainforge.generators.staged.micro import (
        micro_preference_eligibility,
    )
    from Trainforge.generators.staged.window_contract import (
        resolve_chunk_misconceptions,
    )
    from Trainforge.synthesis.synthesis_eligibility import (
        focus_chunk_on_canonical_objective,
        pair_eligibility,
    )
    from Trainforge.training.compute_backend import is_dpo_editorial_record

    objectives = objectives or {}
    cards_by_rung: Dict[str, int] = {}
    admissible_by_rung: Dict[str, int] = {}
    cards_found = 0
    chunk_count = 0
    arm_a_admitted = 0
    projected_admissible = 0

    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        chunk_count += 1

        authored = chunk.get("misconceptions")
        if isinstance(authored, list) and any(
            isinstance(item, Mapping) for item in authored
        ):
            recovered = list(authored)
        else:
            recovered = resolve_chunk_misconceptions(chunk)
        cards_found += len(recovered)
        for card in recovered:
            rung = _normalize_rung(
                card.get("bloom_level") if isinstance(card, Mapping) else None
            )
            cards_by_rung[rung] = cards_by_rung.get(rung, 0) + 1

        chunk_for_eligibility = dict(chunk)
        chunk_for_eligibility["misconceptions"] = recovered

        focused = focus_chunk_on_canonical_objective(
            chunk_for_eligibility, seed=seed, objectives=objectives,
        )
        # The shared content/LO/evidence-window/length gates are identical
        # for "instruction" and "preference" kinds up to the
        # preference-specific Arm A/B step (see
        # ``synthesis_eligibility.pair_eligibility``); reusing the
        # "instruction" kind here gets that shared gate for free without
        # re-deriving it.
        shared_gate = pair_eligibility(focused, kind="instruction")
        if not shared_gate.eligible:
            continue
        focus = focused.get("synthesis_focus_objective")
        if not isinstance(focus, Mapping):
            continue

        result = micro_preference_eligibility(focused, focus=focus)
        if not result.get("eligible"):
            continue
        arm_a_admitted += 1

        candidates = result.get("candidates") or []
        top_candidate: Mapping[str, Any] = candidates[0] if candidates else {}
        # The same stamp ``preference_factory.synthesize_preference_pair``
        # mints for an Arm-A-admitted candidate — real predicate, real
        # input shape, not a re-derived assumption.
        synthetic_record = {
            "source": "misconception",
            "misconception_id": top_candidate.get("id"),
        }
        if is_dpo_editorial_record(synthetic_record):
            projected_admissible += 1
            rung = _normalize_rung(top_candidate.get("bloom_level"))
            admissible_by_rung[rung] = admissible_by_rung.get(rung, 0) + 1

    floor = int(min_dpo_pairs)
    return {
        "contract_version": DPO_YIELD_PROJECTION_CONTRACT_VERSION,
        "chunk_count": chunk_count,
        "cards_found": cards_found,
        "cards_by_rung": dict(sorted(cards_by_rung.items())),
        "arm_a_admitted": arm_a_admitted,
        "projected_admissible": projected_admissible,
        "projected_admissible_by_rung": dict(sorted(admissible_by_rung.items())),
        "min_dpo_pairs": floor,
        "shortfall": max(0, floor - projected_admissible),
        "deficit": projected_admissible < floor,
    }


# --------------------------------------------------------------------------- #
# I/O helpers shared by the validator and the CLI script
# --------------------------------------------------------------------------- #


def read_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a ``chunks.jsonl`` sidecar into a list of dicts.

    Malformed lines are skipped (mirrors the tolerant JSONL readers
    elsewhere on this contract, e.g. ``lib.validators.synthesis_quota``) —
    a single corrupt line must not sink the whole projection.
    """
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_canonical_objectives(path: Optional[Any]) -> Dict[str, Dict[str, Any]]:
    """Load the canonical ``{lo_id: {...}}`` objectives map at ``path``.

    Thin re-export of
    :func:`lib.validators.pair.objective_delivery._load_synthesized_objectives_for_w4c`
    — the SAME loader the ``pair_objective_delivery`` gate + the staged-v4
    synthesis preflight use, so a course's objectives project identically
    here and at real synthesis time. Missing / unreadable / empty ``path``
    degrades to ``{}`` (that loader's own contract) rather than raising.
    """
    from lib.validators.pair.objective_delivery import (
        _load_synthesized_objectives_for_w4c,
    )

    return _load_synthesized_objectives_for_w4c(path)


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class DpoYieldProjectionValidator:
    """Warning-severity DPO-yield famine guard (Bloom-ladder addendum AD-02).

    Inputs:
        chunks_path: str | Path
            The corpus chunkset (``imscc_chunks/chunks.jsonl`` or the
            equivalent staged-synthesis-input sidecar).
        objectives_path: str | Path, optional
            The canonical ``synthesized_objectives.json`` / ``objectives.json``.
            Missing -> structured skip (no meaningful projection is possible
            without it — see :func:`project_dpo_yield`).
        min_dpo_pairs: int, optional
            Trainer floor to project against (default
            :data:`DEFAULT_MIN_DPO_PAIRS` = 50, mirroring
            ``TrainingConfig.min_dpo_pairs``).

    Structured skip (``passed=True``, single INFO issue,
    ``metadata["skipped"]=True``) fires on:
      * ``chunks_path`` missing / unreadable / absent from inputs,
      * ``objectives_path`` missing / unreadable / resolves to an empty map,
      * the corpus carries ZERO recovered misconception cards AND
        ``ED4ALL_BLOOM_LADDER`` is off — legacy corpora that were never
        authored with structured misconception cards get no famine warning
        (untouched byte-for-byte); a ladder-enabled corpus with zero cards
        is a real signal and fires the warning normally.

    Otherwise: a single warning-severity ``DPO_YIELD_PROJECTION_BELOW_FLOOR``
    issue fires when ``projected_admissible < min_dpo_pairs``, carrying the
    per-rung breakdown in its message. ``passed`` honors only critical
    issues (none are ever emitted here), so this gate never blocks a build —
    it computes and reports, matching the day-1 posture of its
    ``synthesis_quota`` sibling on the same phase.
    """

    name = "dpo_yield_projection"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        thresholds = inputs.get("thresholds") or {}
        try:
            min_dpo_pairs = int(
                inputs.get("min_dpo_pairs")
                or thresholds.get("min_dpo_pairs")
                or DEFAULT_MIN_DPO_PAIRS
            )
        except (TypeError, ValueError):
            min_dpo_pairs = DEFAULT_MIN_DPO_PAIRS

        def _skip(reason: str) -> GateResult:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="info",
                        code=_CODE_SKIPPED_MISSING_INPUT,
                        message=(
                            f"dpo_yield_projection skipped: {reason}."
                        ),
                    )
                ],
                action=None,
                metadata={"skipped": True, "skip_reason": reason},
            )

        chunks_path_raw = inputs.get("chunks_path")
        if not chunks_path_raw:
            return _skip("chunks_path_missing")
        chunks_path = Path(chunks_path_raw)
        if not chunks_path.exists():
            return _skip("chunks_path_absent")
        try:
            chunks = read_chunks_jsonl(chunks_path)
        except OSError as exc:
            return _skip(f"chunks_read_error:{exc}")

        objectives_path_raw = inputs.get("objectives_path")
        if not objectives_path_raw:
            return _skip("objectives_path_missing")
        objectives = load_canonical_objectives(objectives_path_raw)
        if not objectives:
            return _skip("objectives_unavailable")

        report = project_dpo_yield(
            chunks, objectives, min_dpo_pairs=min_dpo_pairs,
        )

        if report["cards_found"] == 0 and not resolve_bloom_ladder():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="info",
                        code=_CODE_SKIPPED_NO_CARDS,
                        message=(
                            "dpo_yield_projection skipped: no misconception "
                            "cards recovered and ED4ALL_BLOOM_LADDER is "
                            "unset (legacy corpus)."
                        ),
                    )
                ],
                action=None,
                metadata={"skipped": True, "skip_reason": "no_cards_legacy_corpus", **report},
            )

        issues: List[GateIssue] = []
        if report["deficit"]:
            rung_detail = ", ".join(
                f"{rung}={count}"
                for rung, count in report["cards_by_rung"].items()
            ) or "none"
            issues.append(
                GateIssue(
                    severity="warning",
                    code=_CODE_BELOW_FLOOR,
                    message=(
                        f"Projected DPO-admissible pairs "
                        f"({report['projected_admissible']}) is below "
                        f"min_dpo_pairs ({min_dpo_pairs}) — shortfall "
                        f"{report['shortfall']}. cards_found="
                        f"{report['cards_found']} across {report['chunk_count']} "
                        f"chunks (by rung: {rung_detail}); arm_a_admitted="
                        f"{report['arm_a_admitted']}. The paraphrase loop "
                        "will run to completion before the trainer's own "
                        "min_dpo_pairs gate discovers this shortfall — "
                        "backfill misconception cards or lower "
                        "min_dpo_pairs before a long synthesis run."
                    ),
                    location=str(chunks_path),
                    suggestion=(
                        "Run `python -m Trainforge.scripts.ops.model_dpo_yield "
                        "--course-dir <dir>` for the full per-rung table, or "
                        "author more misconception cards at the "
                        "under-represented rungs."
                    ),
                )
            )

        critical = [issue for issue in issues if issue.severity == "critical"]
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=len(critical) == 0,
            score=(
                1.0
                if not report["deficit"]
                else round(
                    report["projected_admissible"] / max(1, min_dpo_pairs), 4
                )
            ),
            issues=issues,
            action=None,
            metadata={"skipped": False, **report},
        )


__all__ = [
    "DEFAULT_MIN_DPO_PAIRS",
    "DPO_YIELD_PROJECTION_CONTRACT_VERSION",
    "DpoYieldProjectionValidator",
    "load_canonical_objectives",
    "project_dpo_yield",
    "read_chunks_jsonl",
]
