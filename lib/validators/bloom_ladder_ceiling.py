"""Bloom-ladder ceiling/completeness gate — Bloom-ladder initiative WI-15.

Two structural checks over the rewrite-tier ``blocks_final.jsonl`` + the
canonical ``synthesized_objectives.json`` (config wiring lands separately in
WI-21; this module is importable + independently testable ahead of that):

(a) **NEVER-above-ceiling** — a block's own ``bloom_level`` (its emitted
    Bloom demand) or ``ladder_rung`` (WI-06's planner-stamped rung
    provenance, ``Courseforge/scripts/blocks.py::Block.ladder_rung``) must
    never rank above ANY of its target objectives' own synthesized
    ``bloom_level``. The ladder is capped at the objective's own ceiling by
    design (WI-06's planner selection + WI-07's outline-tier clamp both
    enforce this at authoring time); this gate is the independent,
    read-only-off-disk backstop that catches a regression in either of
    those upstream enforcement points.
(b) **Ladder completeness** — every objective with a resolvable ceiling
    should have at least one covering block at EACH rung from ``remember``
    up to its own ceiling (``lib.ontology.bloom_ladder.rungs_up_to`` via
    the ``lib.generation.bloom_ladder_blocks.ladder_plan_for_objective``
    choke point). A rung with zero covering blocks is a gap in the
    objective's instructional ladder.

Warning severity day-1 — ``passed`` honors only critical issues (none are
ever emitted by the checks above; critical is reserved for a genuinely
malformed input), so this gate never blocks a build; it computes and
reports. The eventual critical-severity promotion is a `config/workflows.yaml`
change, not a code change here.

Structured skip when ``ED4ALL_BLOOM_LADDER`` is off
(:func:`lib.generation.bloom_ladder_blocks.resolve_bloom_ladder`): a single
INFO-severity ``BLOOM_LADDER_DISABLED`` issue, ``metadata["skipped"] =
True`` (the pass-rate rollup convention every flag-gated validator on this
contract uses — see ``lib.validators.bloom.classifier_disagreement``'s
``BLOOM_TRIVOTE_INSUFFICIENT_VOTERS`` structured skip), no ceiling/
completeness walk at all. Legacy corpora built before this initiative carry
no ``ladder_rung`` provenance and no ladder-shaped objectives, so the flag
being off keeps every existing snapshot byte-identical.

Objective-ceiling resolution mirrors
``lib.generation.block_planner._bloom_ladder_objective_ceiling`` field-for-
field (itself documented as sharing the tolerant ``bloom_level`` /
``bloomLevel`` fallback + ``strip().lower()`` normalization used by the
synthesized-objectives projection at
``lib/validators/pair/objective_delivery.py:574-636``). Reading the
identical two fields the identical way is what keeps WI-06's planner
ceiling and this validator's ceiling from ever structurally disagreeing —
this module deliberately does NOT import the heavier
``lib.validators.pair.objective_delivery`` (it pulls in the NLI classifier
stack this structural gate has no use for); it re-derives the same
projection locally instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.generation.bloom_ladder_blocks import (
    ladder_plan_for_objective,
    resolve_bloom_ladder,
)
from lib.ontology.bloom import BLOOM_LEVELS
from lib.validators.abcd_objective import _flatten_objectives

logger = logging.getLogger(__name__)

_CODE_DISABLED = "BLOOM_LADDER_DISABLED"
_CODE_ABOVE_CEILING = "BLOOM_LADDER_ABOVE_CEILING"
_CODE_INCOMPLETE_RUNG = "BLOOM_LADDER_INCOMPLETE_RUNG"
_CODE_MISSING_BLOCKS = "MISSING_BLOCKS_INPUT"

#: 0-indexed rank of each canonical Bloom level (remember=0 .. create=5).
#: Self-consistent within this module only — never compared against
#: ``lib.ontology.bloom_ladder.RungEntry.ordinal`` (1-indexed).
_LEVEL_ORDINAL: Dict[str, int] = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}


# --------------------------------------------------------------------------- #
# Block-shape helpers (dict-or-dataclass tolerant)
# --------------------------------------------------------------------------- #


def _block_attr(block: Any, key: str) -> Any:
    """Get ``block.<key>`` for dataclass Blocks OR ``block[<key>]`` for dicts.

    Mirror of ``lib.validators.block_objective_delivery._block_attr``.
    """
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, dict):
        return block.get(key)
    return None


def _norm_level(raw: Any) -> Optional[str]:
    """Lowercase/strip ``raw`` and return it iff a canonical Bloom level."""
    if not isinstance(raw, str):
        return None
    level = raw.strip().lower()
    return level if level in _LEVEL_ORDINAL else None


def _resolve_block_level(block: Any, field: str) -> Optional[str]:
    """Read + normalize ``block.<field>`` (``bloom_level`` / ``ladder_rung``)."""
    return _norm_level(_block_attr(block, field))


def _resolve_block_objective_ids(block: Any) -> List[str]:
    """Pull the canonical objective_id list off a block.

    Mirrors ``lib.validators.block_objective_delivery._resolve_objective_ids``
    — prefers the structural ``Block.objective_ids`` tuple, falls back to
    ``content["objective_ids"]`` / ``content["objective_refs"]`` on
    outline-tier dicts.
    """
    structural = list(_block_attr(block, "objective_ids") or ())
    if structural:
        return [o for o in structural if isinstance(o, str) and o]
    content = _block_attr(block, "content")
    if isinstance(content, dict):
        raw = content.get("objective_ids") or content.get("objective_refs") or []
        return [o for o in raw if isinstance(o, str) and o]
    return []


def _coerce_blocks(inputs: Dict[str, Any]) -> Tuple[List[Any], Optional[GateIssue]]:
    """Resolve ``inputs['blocks']`` or hydrate from ``inputs['blocks_final_path']``.

    Mirrors ``lib.validators.worked_example_math._coerce_blocks``.
    """
    raw = inputs.get("blocks")
    if raw is not None:
        if not isinstance(raw, list):
            return [], GateIssue(
                severity="critical",
                code="INVALID_BLOCKS_INPUT",
                message=(
                    f"inputs['blocks'] must be a list; got {type(raw).__name__}."
                ),
            )
        return list(raw), None

    path_raw = inputs.get("blocks_final_path")
    if not path_raw:
        return [], GateIssue(
            severity="critical",
            code=_CODE_MISSING_BLOCKS,
            message=(
                "inputs requires 'blocks' (List[Block]) or 'blocks_final_path' "
                "(a JSONL path of rewrite-tier block dicts)."
            ),
        )
    try:
        path = Path(path_raw)
    except (TypeError, ValueError):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=f"blocks_final_path is not a path: {path_raw!r}.",
        )
    if not path.exists():
        return [], GateIssue(
            severity="critical",
            code=_CODE_MISSING_BLOCKS,
            message=f"blocks_final_path does not exist: {path}.",
        )
    blocks: List[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return blocks, None


# --------------------------------------------------------------------------- #
# Objective-ceiling resolution (mirrors block_planner._bloom_ladder_objective_ceiling)
# --------------------------------------------------------------------------- #


def _objective_ceiling(lo: Mapping[str, Any]) -> Optional[str]:
    """Resolve an objective's OWN synthesized ceiling.

    Byte-identical field-read to
    ``lib.generation.block_planner._bloom_ladder_objective_ceiling``:
    ``bloom_level`` / ``bloomLevel`` tolerant fallback, stripped +
    lowercased, ``None`` unless the result is one of the six canonical
    Bloom levels. Reading the identical two fields the identical way is
    what keeps the planner (WI-06) and this validator from ever
    structurally disagreeing about an objective's ceiling.
    """
    raw = lo.get("bloom_level") or lo.get("bloomLevel")
    return _norm_level(raw)


def _coerce_objectives(inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve ``{lo_id: {"ceiling": <bloom_level | None>}}`` from ``inputs``.

    Priority: ``inputs['objectives']`` (any container ``_flatten_objectives``
    accepts) > ``inputs['synthesized_objectives_path']`` (the canonical
    ``synthesized_objectives.json``). Missing / unreadable / empty input
    degrades to ``{}`` — an empty objectives map is not an error, it is a
    no-op (no ceiling is resolvable, so nothing is auditable); mirrors
    ``lib.validators.abcd_objective._coerce_objectives``'s "empty LO list is
    NOT an error" contract.
    """
    explicit = inputs.get("objectives")
    if explicit is not None:
        los = _flatten_objectives(explicit)
    else:
        path_raw = inputs.get("synthesized_objectives_path")
        if not path_raw:
            return {}
        try:
            path = Path(path_raw)
        except (TypeError, ValueError):
            return {}
        if not path.exists():
            logger.debug(
                "bloom_ladder_ceiling: synthesized_objectives_path %r does "
                "not exist; running with an empty objectives map.",
                str(path),
            )
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "bloom_ladder_ceiling: failed to parse "
                "synthesized_objectives_path %r: %s",
                str(path),
                exc,
            )
            return {}
        los = _flatten_objectives(data)

    out: Dict[str, Dict[str, Any]] = {}
    for lo in los:
        if not isinstance(lo, Mapping):
            continue
        lo_id_raw = lo.get("id") or lo.get("objective_id")
        if not isinstance(lo_id_raw, str):
            continue
        lo_id = lo_id_raw.strip()
        if not lo_id:
            continue
        out[lo_id] = {"ceiling": _objective_ceiling(lo)}
    return out


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class BloomLadderCeilingValidator:
    """Ladder ceiling/completeness gate — Bloom-ladder initiative WI-15.

    Inputs:
        blocks: List[Block] | List[dict]
            Rewrite-tier blocks (hydrated Block instances, or plain dict
            rows read off ``blocks_final.jsonl``).
        blocks_final_path: str | Path
            Alternative to ``blocks`` — a rewrite-tier blocks JSONL.
        objectives: list | dict
            Already-loaded synthesized-objectives container (any shape
            ``lib.validators.abcd_objective._flatten_objectives`` accepts).
        synthesized_objectives_path: str | Path
            Alternative to ``objectives`` — the canonical
            ``synthesized_objectives.json`` path.

    See the module docstring for the two checks + the flag-off structured
    skip.
    """

    name = "bloom_ladder_ceiling"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)

        if not resolve_bloom_ladder():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="info",
                        code=_CODE_DISABLED,
                        message=(
                            "ED4ALL_BLOOM_LADDER is unset — Bloom-ladder "
                            "ceiling/completeness auditing is skipped "
                            "(legacy blocks carry no ladder_rung "
                            "provenance; byte-stable)."
                        ),
                    )
                ],
                action=None,
                metadata={"skipped": True, "enabled": False},
            )

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        objectives = _coerce_objectives(inputs)

        issues: List[GateIssue] = []
        ceiling_audited = 0
        ceiling_passed = 0
        blocks_by_objective: Dict[str, List[Any]] = {}

        # ---- (a) NEVER-above-ceiling ------------------------------------
        for block in blocks:
            objective_ids = _resolve_block_objective_ids(block)
            for oid in objective_ids:
                blocks_by_objective.setdefault(oid, []).append(block)

            observed: Dict[str, str] = {}
            bloom_level = _resolve_block_level(block, "bloom_level")
            if bloom_level is not None:
                observed["bloom_level"] = bloom_level
            ladder_rung = _resolve_block_level(block, "ladder_rung")
            if ladder_rung is not None:
                observed["ladder_rung"] = ladder_rung
            if not observed or not objective_ids:
                continue

            ceilings = [
                objectives[oid]["ceiling"]
                for oid in objective_ids
                if oid in objectives and objectives[oid].get("ceiling") is not None
            ]
            if not ceilings:
                # None of this block's target objectives have a resolvable
                # ceiling — not auditable either way.
                continue

            ceiling_audited += 1
            block_id = _block_attr(block, "block_id") or "<unknown>"
            breaches: List[str] = []
            for oid in objective_ids:
                entry = objectives.get(oid)
                ceiling = entry.get("ceiling") if entry else None
                if ceiling is None:
                    continue
                for field_name, level in observed.items():
                    if _LEVEL_ORDINAL[level] > _LEVEL_ORDINAL[ceiling]:
                        breaches.append(
                            f"{field_name}={level!r} ranks above {oid}'s "
                            f"ceiling {ceiling!r}"
                        )

            if breaches:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code=_CODE_ABOVE_CEILING,
                        message=(
                            f"Block {block_id!r} ranks above its target "
                            f"objective's own Bloom-ladder ceiling: "
                            f"{'; '.join(breaches)}."
                        ),
                        location=str(block_id),
                        suggestion=(
                            "The Bloom-ladder caps every rung at the "
                            "target objective's own declared bloom_level "
                            "(never above it) — re-roll or re-clamp this "
                            "block down to that ceiling."
                        ),
                    )
                )
            else:
                ceiling_passed += 1

        # ---- (b) ladder completeness -------------------------------------
        completeness_audited = 0
        completeness_passed = 0

        for oid in sorted(objectives):
            ceiling = objectives[oid].get("ceiling")
            if ceiling is None:
                continue
            try:
                rungs = ladder_plan_for_objective(ceiling)
            except ValueError:
                # Defensive only — _objective_ceiling already validated
                # `ceiling` against the canonical six-level set.
                continue

            covered_levels: set = set()
            for block in blocks_by_objective.get(oid, []):
                for field_name in ("bloom_level", "ladder_rung"):
                    level = _resolve_block_level(block, field_name)
                    if level is not None:
                        covered_levels.add(level)

            for rung in rungs:
                completeness_audited += 1
                if rung.level in covered_levels:
                    completeness_passed += 1
                else:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=_CODE_INCOMPLETE_RUNG,
                            message=(
                                f"Objective {oid!r} (ceiling {ceiling!r}) "
                                f"has no covering block at ladder rung "
                                f"{rung.level!r} (ordinal {rung.ordinal})."
                            ),
                            location=oid,
                            suggestion=(
                                "Every rung from remember up to the "
                                "objective's own ceiling should carry "
                                ">=1 covering block — re-run the "
                                "ladder-selection planner pass for this "
                                "objective."
                            ),
                        )
                    )

        total_audited = ceiling_audited + completeness_audited
        total_passed = ceiling_passed + completeness_passed
        score = 1.0 if total_audited == 0 else round(total_passed / total_audited, 4)
        # Warning-severity day-1: `passed` honors only critical issues
        # (none are ever emitted by the checks above), so the gate never
        # blocks — it computes + reports. The deferred critical promotion
        # is a config/workflows.yaml change, not a code change here.
        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
            metadata={
                "skipped": False,
                "enabled": True,
                "ceiling_audited": ceiling_audited,
                "ceiling_passed": ceiling_passed,
                "completeness_audited": completeness_audited,
                "completeness_passed": completeness_passed,
            },
        )


__all__ = ["BloomLadderCeilingValidator"]
