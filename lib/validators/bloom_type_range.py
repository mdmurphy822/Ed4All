"""IB7.6c — BloomTypeRangeValidator.

Per-type Bloom-range ceiling enforcement — the advisory→gate promotion of
``Courseforge/config/block_catalog.yaml``'s ``bloom_ceiling`` field. The
framework's ``bloom-type-pyramid-range-enforcement`` gap (sev med): a block
type encodes a *stretch limit* (an exposition ``concept`` should top out at
``apply``; an ``activity`` / ``problem`` at ``analyze``; a ``scenario`` at
``evaluate``; only a graded ``assessment_item`` reaches ``create``). When a
block's declared/observed target Bloom EXCEEDS its type's ceiling, the block
is asking the wrong block type to carry higher-order cognition — it should be
re-routed to a higher-order type. This validator flags the over-ceiling
block so the planner / rewrite tier can re-route or re-target it.

Different surface from ``BloomAlignmentValidator`` (declared==observed) and
from the planner's ``_bloom_floor_for_catalog_entry`` (a FLOOR, never a cap):
this validator audits ONLY the per-block CEILING.

Per the IB7.6 spec (``plans/finegrain/ib7-planner-lifecycle-bloom-climb-
spacing-2026-06-22.md`` §IB7.6c): for each block, resolve its target Bloom
(``target_bloom`` if a valid ``BLOOM_LEVELS`` member, else ``bloom_level``)
and look up its type's ``bloom_ceiling``. If the resolved Bloom's index in
``BLOOM_LEVELS`` exceeds the ceiling's index, emit a warning-severity
GateIssue carrying ``action="regenerate"``. Blocks whose type has no ceiling,
or whose Bloom is unresolvable, skip-pass. Empty input gracefully passes.

NO embeddings — pure index comparison. Mirrors the ``example_completeness``
posture (capture-emit pattern, empty-input graceful pass,
``_ISSUE_LIST_CAP``, resolve-override helper).

Inputs (``inputs`` dict):

    blocks: List[Block]
        Rewrite- (or outline-) tier ``Courseforge.scripts.blocks.Block``
        instances. Only ``block_type`` + ``target_bloom`` / ``bloom_level``
        are read.

    bloom_ceilings: Optional[Dict[str, str]]
        Override the ``{block_type: bloom_ceiling}`` map. When provided it is
        used INSTEAD of loading from the catalog — so the validator is
        independently testable before the catalog gains ``bloom_ceiling``
        keys. Resolution: inputs > constructor > catalog-loaded.

    decision_capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance. One
        ``block_validation_action`` event is emitted per audited block (only
        blocks whose type carries a ceiling).

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"bloom_type_range"``).

References:
    - ``plans/finegrain/ib7-planner-lifecycle-bloom-climb-spacing-2026-06-22.md``
      §IB7.6c.
    - ``lib/validators/example_completeness.py`` — sibling CB5a gate
      (structure mirrored here).
    - ``lib/ontology/bloom.py`` — ``BLOOM_LEVELS`` single source of truth.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.bloom import BLOOM_LEVELS

# Import bridge for Block (mirror of example_completeness.py).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover - import guard
    from blocks import Block  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for partial installs
    Block = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


#: Higher-order block types a re-route can target (informational; named in
#: the issue suggestion).
_HIGHER_ORDER_TYPES = ("scenario", "problem", "assessment_item")

#: Cap on per-validate issue list to avoid runaway emit on many-block
#: corpora. Mirrors the cap used by sibling structural validators.
_ISSUE_LIST_CAP: int = 50

#: Index lookup over the canonical six-level Bloom ordering.
_BLOOM_INDEX: Dict[str, int] = {lvl: i for i, lvl in enumerate(BLOOM_LEVELS)}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_block_bloom(block: Any) -> Optional[str]:
    """Resolve a block's target Bloom level.

    Prefer ``target_bloom`` when it is a valid ``BLOOM_LEVELS`` member, else
    fall back to ``bloom_level``. Returns ``None`` when neither resolves to a
    canonical level (the block then skip-passes).
    """
    tb = getattr(block, "target_bloom", None)
    if isinstance(tb, str) and tb in _BLOOM_INDEX:
        return tb
    bl = getattr(block, "bloom_level", None)
    if isinstance(bl, str) and bl in _BLOOM_INDEX:
        return bl
    return None


def _load_catalog_ceilings() -> Dict[str, str]:
    """Build the ``{block_type: bloom_ceiling}`` map from the block catalog.

    Reads each catalog entry's OPTIONAL ``bloom_ceiling`` key; entries
    without one contribute no ceiling (that type skip-passes). Graceful: a
    catalog-load failure logs + returns an empty map (every block then
    skip-passes — the gate degrades to a no-op rather than crashing).
    """
    try:
        from lib.generation.block_catalog import load_block_catalog

        catalog = load_block_catalog()
    except Exception as exc:  # noqa: BLE001 - degrade to no-op on load failure
        logger.debug(
            "bloom_type_range: block-catalog load failed (%s); "
            "no ceilings resolved.",
            exc,
        )
        return {}
    ceilings: Dict[str, str] = {}
    for entry in catalog:
        if not isinstance(entry, dict):
            continue
        bt = entry.get("block_type")
        ceiling = entry.get("bloom_ceiling")
        if (
            isinstance(bt, str)
            and isinstance(ceiling, str)
            and ceiling in _BLOOM_INDEX
        ):
            ceilings[bt] = ceiling
    return ceilings


# ---------------------------------------------------------------------------
# Decision-capture emit
# ---------------------------------------------------------------------------


def _emit_decision(
    capture: Optional[Any],
    *,
    block_id: str,
    block_type: str,
    observed_bloom: str,
    bloom_ceiling: str,
    over_ceiling: bool,
    passed: bool,
) -> None:
    """Emit one ``block_validation_action`` event per audited (ceiling-
    carrying) block.

    Rationale interpolates the dynamic signals so post-hoc replay
    reconstructs the verdict without re-reading the gate result.
    """
    if capture is None:
        return
    verdict = "over_ceiling" if over_ceiling else "within_ceiling"
    rationale = (
        f"bloom_type_range audit of block {block_id!r} "
        f"(block_type={block_type}): observed_bloom={observed_bloom}, "
        f"bloom_ceiling={bloom_ceiling}, over_ceiling={over_ceiling} -> "
        f"{verdict}. A block whose target Bloom exceeds its type's stretch "
        f"ceiling should be re-routed to a higher-order block type."
    )
    ml_features = {
        "block_id": block_id,
        "block_type": block_type,
        "observed_bloom": observed_bloom,
        "bloom_ceiling": bloom_ceiling,
        "over_ceiling": bool(over_ceiling),
        "passed": bool(passed),
    }
    try:
        capture.log_decision(
            decision_type="block_validation_action",
            decision=verdict,
            rationale=rationale,
            ml_features=ml_features,
        )
    except Exception as exc:  # noqa: BLE001 - audit emit is best-effort
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "bloom_type_range block_validation_action: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class BloomTypeRangeValidator:
    """IB7.6c — per-type Bloom-range ceiling gate.

    Validator-protocol-compatible class. Flags every block whose resolved
    target Bloom exceeds its type's ``bloom_ceiling``
    (``BLOCK_BLOOM_OVER_CEILING``, warning, ``action="regenerate"``). Blocks
    whose type carries no ceiling, or whose Bloom is unresolvable, skip-pass.
    Empty input passes gracefully.
    """

    name = "bloom_type_range"
    version = "0.1.0"

    def __init__(
        self,
        *,
        bloom_ceilings: Optional[Dict[str, str]] = None,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self._bloom_ceilings_override = bloom_ceilings
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture
        ceilings = self._resolve_ceilings(inputs)

        raw = inputs.get("blocks")
        if raw is None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="MISSING_BLOCKS_INPUT",
                        message=(
                            "inputs['blocks'] is required; expected a list "
                            "of Courseforge Block instances."
                        ),
                    )
                ],
                action="regenerate",
            )
        if not isinstance(raw, list):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="INVALID_BLOCKS_INPUT",
                        message=(
                            f"inputs['blocks'] must be a list; got "
                            f"{type(raw).__name__}."
                        ),
                    )
                ],
                action="regenerate",
            )

        blocks: List[Any] = list(raw)

        issues: List[GateIssue] = []
        audited = 0
        over = 0
        for b in blocks:
            block_type = getattr(b, "block_type", None)
            if not isinstance(block_type, str):
                continue
            ceiling = ceilings.get(block_type)
            if not ceiling:
                # No ceiling for this type → skip-pass.
                continue
            observed = _resolve_block_bloom(b)
            if observed is None:
                # Unresolvable Bloom → skip-pass.
                continue
            audited += 1
            block_id = str(
                getattr(b, "block_id", None)
                or getattr(b, "stable_id", None)
                or "<unknown>"
            )
            over_ceiling = _BLOOM_INDEX[observed] > _BLOOM_INDEX[ceiling]
            passed_block = not over_ceiling
            _emit_decision(
                capture,
                block_id=block_id,
                block_type=block_type,
                observed_bloom=observed,
                bloom_ceiling=ceiling,
                over_ceiling=over_ceiling,
                passed=passed_block,
            )
            if passed_block:
                continue
            over += 1
            if len(issues) >= _ISSUE_LIST_CAP:
                continue
            issues.append(
                GateIssue(
                    severity="warning",
                    code="BLOCK_BLOOM_OVER_CEILING",
                    message=(
                        f"block {block_id!r} (block_type={block_type}) targets "
                        f"Bloom level {observed!r} which exceeds its type's "
                        f"stretch ceiling {ceiling!r}."
                    ),
                    location=block_id,
                    suggestion=(
                        f"Re-route this block to a higher-order block type "
                        f"that admits {observed!r} (e.g. one of "
                        f"{', '.join(_HIGHER_ORDER_TYPES)}), preserving its "
                        f"target objective ids; or re-target the block down to "
                        f"<= {ceiling!r}."
                    ),
                )
            )

        passed = over == 0
        score = round((audited - over) / audited, 4) if audited else 1.0
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )

    # ---------------------------------------------------------------- #
    # Ceiling-map resolution helper
    # ---------------------------------------------------------------- #

    def _resolve_ceilings(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Resolve the ``{block_type: bloom_ceiling}`` map.

        Resolution: inputs['bloom_ceilings'] > constructor override >
        catalog-loaded. An override (inputs or constructor) is used VERBATIM
        in place of the catalog, so the gate is independently testable before
        the catalog gains ``bloom_ceiling`` keys.
        """
        override = inputs.get("bloom_ceilings")
        if isinstance(override, dict):
            return self._sanitize(override)
        if isinstance(self._bloom_ceilings_override, dict):
            return self._sanitize(self._bloom_ceilings_override)
        return _load_catalog_ceilings()

    @staticmethod
    def _sanitize(mapping: Dict[str, Any]) -> Dict[str, str]:
        """Keep only ``{str: <valid BLOOM_LEVELS member>}`` entries."""
        out: Dict[str, str] = {}
        for k, v in mapping.items():
            if isinstance(k, str) and isinstance(v, str) and v in _BLOOM_INDEX:
                out[k] = v
        return out


__all__ = [
    "BloomTypeRangeValidator",
]
