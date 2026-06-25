"""IB4.5 — UDL multiple-means coverage validator (≥2 representations + autonomy floor).

Audits the UDL coverage of the Block batch (QA-13 / 4.5 Rule 6 / D7):

* Per PAGE/module (grouped by ``page_id``, skip ``chrome`` / ``objective``
  blocks): assert the UNION of representation modes across the page's
  content-bearing blocks is ``>= 2`` (the Representation/Recognition floor).
  UDL multiple-means is a per-PAGE property, NOT a per-block one — an
  intrinsically single-mode block (table, checklist, callout, scenario, ...)
  is expected to be single-modal in isolation; only the page aggregate is
  judged. A page below the floor emits a single ``UDL_SINGLE_REPRESENTATION``
  (warning).
* Per MODULE/week (grouped by ``page_id`` week prefix): assert ≥1 block carries
  an engagement/autonomy affordance — a non-empty ``response_formats`` OR a
  non-``None`` ``engagement_affordance`` (the Action/Expression +
  Engagement/Affective autonomy floor). A week with zero is
  ``UDL_NO_AUTONOMY_AFFORDANCE`` (warning).

The validator derives ``(n_representations, response_formats,
engagement_affordance)`` ON READ via ``blocks._derive_udl_coverage`` when the
Block's fields are empty (the default), so it works even before the emit-side
``ED4ALL_BLOCK_A11Y`` flag populates them — the populated fields win when set.

Statistical-tier graceful-degrade contract (documented, inherited by a FUTURE
embedding-backed autonomy check): v1 is fully DETERMINISTIC (pure string/HTML
inspection, no embeddings), so it NEVER short-circuits on missing ``[embedding]``
extras. A later embedding-backed autonomy variant MUST honor the
``EMBEDDING_DEPS_MISSING`` warning-``passed=True`` contract and the
``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` fail-closed flip (mirroring
``objective_assessment_similarity`` per the root CLAUDE.md "Statistical-tier
embedding validators" paragraph).

Wired warning-day-1 at ``inter_tier_validation`` + ``post_rewrite_validation``
in both ``course_generation`` + ``textbook_to_course``. Feeds IB6's Engagement
+ Accessibility/UDL quality dimensions. The deferred critical-flip
(``UDL_SINGLE_REPRESENTATION`` → critical) follows a ≥2-corpus FP measurement
(the WS3/W4 deferred-flip pattern) — see the ``# TODO(calibration)`` marker in
``config/workflows.yaml``.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Block import bridge (mirror of rewrite_html_shape / inter_tier_gates).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import (  # type: ignore[import-not-found]  # noqa: E402
    _derive_udl_coverage,
    _derive_udl_representation_modes,
)

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP: int = 50

# Block types that are NOT content-bearing for the representation floor.
_UDL_SKIP_BLOCK_TYPES: frozenset = frozenset({"chrome", "objective"})

# Intrinsically single-mode block types. UDL multiple-means is a per-PAGE /
# module property, NOT a per-block one: a table, checklist, callout, or
# scenario is *expected* to be single-modal in isolation, and only the PAGE
# aggregate matters. These types are documented here so future maintainers do
# not re-introduce a per-block failure for them. With the page-level floor
# they never produce a per-block failure (the page union is what is judged);
# they STILL contribute their own mode(s) to that page union.
_UDL_INTRINSIC_SINGLE_MODE_BLOCK_TYPES: frozenset = frozenset({
    "table", "checklist", "summary_takeaway", "callout", "scenario",
    "key_idea", "acronym", "vocab_card",
})

# Minimum distinct representation modes the PAGE as a whole must provide
# (QA-13). UDL multiple-means is a per-PAGE/module property — the floor is the
# UNION of representation modes across all content-bearing blocks sharing a
# page, NOT a per-block check.
_MIN_REPRESENTATIONS: int = 2

# Week-prefix extractor: ``week_03#...`` / ``week03_...`` etc. → the week key.
_WEEK_PREFIX_RE = re.compile(r"(?i)(week[_-]?\d+)")


def _emit_udl_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    metrics: Dict[str, Any],
) -> None:
    """Emit one ``udl_coverage_check`` decision per ``validate()`` (IB4.5)."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    metric_strs = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
    rationale = (
        f"UDL multiple-means coverage gate verdict={decision}, "
        f"failure_code={code or 'none'}, metrics=({metric_strs})."
    )
    try:
        capture.log_decision(
            decision_type="udl_coverage_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on udl_coverage_check: %s",
            exc,
        )


def _week_key(block: Any) -> str:
    """Group key for the week-level autonomy floor — week prefix of page_id."""
    page_id = getattr(block, "page_id", "") or ""
    m = _WEEK_PREFIX_RE.search(str(page_id))
    if m:
        return m.group(1).lower().replace("-", "_")
    return str(page_id)


def _page_key(block: Any) -> str:
    """Group key for the page-level representation floor — the page_id itself.

    The page is the UDL multiple-means granularity (the week key, used only for
    the coarser autonomy floor, would over-aggregate distinct pages).
    """
    return str(getattr(block, "page_id", "") or "")


def _resolve_mode_set(block: Any) -> "frozenset[str]":
    """Resolve the SET of representation-mode names a block contributes.

    Mirrors ``_resolve_coverage``'s populated-fields fast path but at SET
    granularity, so the page-level floor can union modes across blocks. When
    the block carries a populated ``n_representations`` but we cannot recover
    its concrete mode SET (the emit-side fields store only the count), we fall
    back to a synthetic placeholder set sized to that count — enough to let a
    block with populated ``n_representations >= 2`` clear the page floor on its
    own. Otherwise we derive the real mode set on read.
    """
    n = getattr(block, "n_representations", 0) or 0
    if n:
        # Try to recover the concrete modes from content anyway (keeps the
        # page union precise); fall back to a count-sized placeholder set when
        # the content can't be inspected.
        try:
            derived = _derive_udl_representation_modes(block)
        except Exception:  # noqa: BLE001
            derived = frozenset()
        if derived:
            return derived
        # Synthetic placeholder modes so a populated count still unions.
        return frozenset(f"_n{i}" for i in range(int(n)))
    return _derive_udl_representation_modes(block)


def _resolve_coverage(block: Any) -> Tuple[int, Tuple[str, ...], Optional[str]]:
    """Use populated Block UDL fields when present; else derive on read.

    Each of the three signals is resolved independently: a populated value
    wins, an empty one falls back to the deterministic derivation. This avoids
    a partially-populated block (e.g. only ``engagement_affordance`` set)
    reading ``n_representations`` as 0 when its content clearly has ≥2 modes.
    """
    n = getattr(block, "n_representations", 0) or 0
    rf = getattr(block, "response_formats", ()) or ()
    ea = getattr(block, "engagement_affordance", None)
    if n and rf and ea is not None:
        return int(n), tuple(rf), ea
    d_n, d_rf, d_ea = _derive_udl_coverage(block)
    return (
        int(n) if n else d_n,
        tuple(rf) if rf else d_rf,
        ea if ea is not None else d_ea,
    )


class UdlCoverageValidator:
    """IB4.5 — UDL ≥2-representations + per-week autonomy-floor gate.

    Deterministic v1 (no embeddings → never short-circuits on missing
    ``[embedding]`` extras; see module docstring for the graceful-degrade
    contract a future embedding-backed autonomy check inherits). Warning-day-1.
    """

    name = "udl_coverage"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        raw = inputs.get("blocks")
        if not isinstance(raw, list):
            _emit_udl_decision(
                capture, passed=True, code="MISSING_BLOCKS_INPUT",
                metrics={"block_count": 0},
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code="MISSING_BLOCKS_INPUT",
                    message=(
                        "inputs['blocks'] absent/not a list; skipping UDL audit."
                    ),
                )],
            )

        issues: List[GateIssue] = []
        content_bearing = 0
        # week_key -> bool(any autonomy affordance seen)
        week_autonomy: Dict[str, bool] = {}
        # page_key -> set(distinct representation modes across the page's
        # content-bearing blocks). The page-level UDL multiple-means floor is
        # judged on this UNION, NOT per block.
        page_modes: Dict[str, set] = {}

        for block in raw:
            block_type = getattr(block, "block_type", "")
            if block_type in _UDL_SKIP_BLOCK_TYPES:
                continue
            # Outline-tier blocks (dict content) still expose block_type +
            # page_id; _derive_udl_coverage handles dict content.
            content_bearing += 1
            _n_rep, rf, ea = _resolve_coverage(block)

            # Page-level representation floor: union this block's modes onto
            # its page. Intrinsically-single-mode block types (table,
            # checklist, ...) are EXPECTED to be single-modal and never produce
            # a per-block failure — they still contribute their mode(s) to the
            # page union (the page aggregate is the only thing judged).
            pk = _page_key(block)
            modes = _resolve_mode_set(block)
            page_modes.setdefault(pk, set()).update(modes)

            wk = _week_key(block)
            has_autonomy = bool(rf) or ea is not None
            week_autonomy[wk] = week_autonomy.get(wk, False) or has_autonomy

        # Per-PAGE representation floor — a page whose UNION of representation
        # modes across its content-bearing blocks is below the floor.
        pages_audited = len(page_modes)
        pages_single_representation = 0
        for pk, modes in sorted(page_modes.items()):
            if len(modes) < _MIN_REPRESENTATIONS:
                pages_single_representation += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="UDL_SINGLE_REPRESENTATION",
                        message=(
                            f"Page {pk!r} provides {len(modes)} distinct "
                            f"representation mode(s) across its content-bearing "
                            f"blocks; UDL multiple-means floor is "
                            f"{_MIN_REPRESENTATIONS} (QA-13)."
                        ),
                        location=pk,
                    ))

        # Per-week autonomy floor — a week with zero affordance flags.
        weeks_no_autonomy = 0
        for wk, seen in sorted(week_autonomy.items()):
            if not seen:
                weeks_no_autonomy += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="UDL_NO_AUTONOMY_AFFORDANCE",
                        message=(
                            f"Week {wk!r} carries zero engagement/autonomy "
                            f"affordance across its content-bearing blocks "
                            f"(no response_formats and no engagement_affordance)."
                        ),
                        location=wk,
                    ))

        # Warning-day-1: every issue is warning severity, so passed stays True.
        passed = True
        score = round(
            1.0 - (pages_single_representation / pages_audited)
            if pages_audited else 1.0,
            4,
        )
        _emit_udl_decision(
            capture,
            passed=passed,
            code=(
                "UDL_SINGLE_REPRESENTATION" if pages_single_representation else (
                    "UDL_NO_AUTONOMY_AFFORDANCE" if weeks_no_autonomy else None
                )
            ),
            metrics={
                "block_count": len(raw),
                "content_bearing": content_bearing,
                # Repurposed to count under-floor PAGES (the floor is now
                # per-page); aliased page-named keys added for clarity.
                "single_representation_count": pages_single_representation,
                "pages_audited": pages_audited,
                "pages_single_representation": pages_single_representation,
                "weeks_audited": len(week_autonomy),
                "weeks_no_autonomy": weeks_no_autonomy,
            },
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )


__all__ = ["UdlCoverageValidator"]
