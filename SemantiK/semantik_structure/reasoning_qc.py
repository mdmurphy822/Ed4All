"""Stage-9b reasoning-QC pass (``SEMANTIK_REASONING_QC``) — reasoning-model
reading-order + structure quality control over the converged ``capped`` region
list, applied ONLY as block-ID reconcile ops (never a free-text rewrite).

**Text-only, document-level (2026-07-12 pivot).** The QC pass reasons over the
ASSEMBLED accessible-HTML-side block sequence (the capped/assembled region text —
type, role, heading level, page annotation) of the WHOLE converted document, NOT
over per-page image rasters. A reasoning TEXT model (own seat, decoupled from the
VLM family — see :func:`reasoning_qc_vlm.resolve_reasoning_qc_seat`) judges the
combined output. The full document's ordered block list is partitioned into
``SEMANTIK_REASONING_QC_WINDOW_BLOCKS``-block windows with junction SEAM strips
between adjacent windows — so seams now cover PAGE BOUNDARIES too (cross-page
reading-order scrambles a per-page design could never see). Each block's page (if
known) rides as an informational per-block annotation.

Posture (three-valued, default byte-identical)
----------------------------------------------
``resolve_reasoning_qc_mode()`` → ``off`` / ``shadow`` / ``on``:

* **off** (DEFAULT, incl. unset/blank/garbage) — the caller never imports this
  module; there is no VLM call, no extra assemble round, no ``reasoning_qc``
  result key. BYTE-IDENTICAL.
* **shadow** — verify + full audit + DecisionCapture tally, apply NOTHING; the
  returned ``capped`` is the INPUT object list unchanged. This is the
  calibration posture that feeds the owner-gated shadow→on precision bar
  (≥95% applied-op precision over ≥2 corpora).
* **on** — apply the reconcile ops through the EXISTING block-ID op layer:
  re-type / drop-phantom / re-type-apparatus via
  :func:`reviewer.run_structure_review` (``restrict_to`` + ``feedback_by_idx``);
  merge / split / reorder(move) via
  :func:`block_resegment.apply_proposed_unit_fix` — reorder graduates ONLY when
  the orthogonal ``SEMANTIK_MOVE_OP == 'live'`` (else validated + audited only).
  Every applied op is followed by the conservation asserts with whole-revert on
  raise, under a never-ship-worse adopt gate.

Reconcile ops REUSE the reviewer's op vocabulary and conservation gates — this
module proposes; the existing layer applies. Decision channels reuse the
existing ``decision_type`` enums (``structure_detection`` for the VLM judgment
call, ``structure_review`` for re-type ops, ``block_resegment`` for merge/move
ops) — NO ``decision_event`` schema change.

ToC reconcile is CASCADE-LOCAL and advisory: the declared ``N.M`` spine is
harvested from the document's OWN ToC/front-matter regions with a SemantiK-local
regex (NOT the forbidden extractor, NOT the downstream ``textbook_structure.json``);
a declared-missing ordinal is warn-only (never invented) so a weak harvest can't
drop real content.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

QC_AUDIT_SCHEMA = "reasoning-qc/1.0"

# Default per-page VLM fan-out width — mirrors the endpoint runtime's
# _DEFAULT_CONCURRENCY (8) so a QC pass saturates the reasoning seat's
# continuous-batching window instead of judging pages one at a time.
_DEFAULT_QC_CONCURRENCY = 8


# ---------------------------------------------------------------------------
# Flag resolver — three-valued, default OFF (byte-identical).
# ---------------------------------------------------------------------------
_QC_OFF_TOKENS = frozenset({"0", "false", "no", "off"})


def resolve_reasoning_qc_concurrency() -> int:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_CONCURRENCY`` (default 8, min 1).

    Bounds the :class:`~concurrent.futures.ThreadPoolExecutor` that fans the
    per-UNIT QC judgment VLM calls out (see :func:`_fan_out_page_verifies`) —
    window sub-slices AND junction seam strips ride the same pool. Mirrors
    :func:`endpoint_runtime.resolve_specialist_concurrency`. Garbage / blank /
    non-positive → the default 8; a valid positive int → that value. Read at CALL
    time (never cached at import).
    """
    raw = os.environ.get("SEMANTIK_REASONING_QC_CONCURRENCY")
    if not raw:
        return _DEFAULT_QC_CONCURRENCY
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_QC_CONCURRENCY
    if val < 1:
        return _DEFAULT_QC_CONCURRENCY
    return val


# ---------------------------------------------------------------------------
# Primary-window PARTITION + junction SEAM knobs (root-cause prevention: a
# dense page's ordered block list is partitioned into non-overlapping windows,
# each judged thinking-on; a seam strip re-judges each junction where scan
# scrambles hide).
# ---------------------------------------------------------------------------
_QC_WINDOW_BLOCKS_ENV = "SEMANTIK_REASONING_QC_WINDOW_BLOCKS"
_QC_SEAM_BLOCKS_ENV = "SEMANTIK_REASONING_QC_SEAM_BLOCKS"
_DEFAULT_QC_WINDOW_BLOCKS = 30
_MIN_QC_WINDOW_BLOCKS = 4
_DEFAULT_QC_SEAM_BLOCKS = 5
_MIN_QC_SEAM_BLOCKS = 2


def resolve_reasoning_qc_window_blocks() -> int:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_WINDOW_BLOCKS`` (default 30, min 4).

    The maximum blocks in ONE primary QC judgment window: a page whose block
    list exceeds this is PARTITIONED into consecutive, NON-overlapping windows of
    at most this size (root-cause prevention — an unbounded dense page drives the
    reasoning seat to exhaust its completion window). Blank / non-int / garbage →
    the default 30; any value below the floor clamps up to 4 (a window must be
    large enough to judge local order). Read at CALL time."""
    raw = os.environ.get(_QC_WINDOW_BLOCKS_ENV)
    if not raw:
        return _DEFAULT_QC_WINDOW_BLOCKS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_QC_WINDOW_BLOCKS
    return max(_MIN_QC_WINDOW_BLOCKS, val)


def resolve_reasoning_qc_seam_blocks() -> int:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_SEAM_BLOCKS`` (default 5, min 2).

    Half-width ``S`` of the junction SEAM strip: for each adjacent primary-window
    pair the seam pass issues one extra thinking-on judgment over the last ``S``
    blocks of window K + the first ``S`` blocks of window K+1 (a contiguous strip
    straddling the junction), so cross-boundary scrambles that no single
    partition window can see are judged on their own evidence. Blank / non-int /
    garbage → the default 5; any value below the floor clamps up to 2. Read at
    CALL time."""
    raw = os.environ.get(_QC_SEAM_BLOCKS_ENV)
    if not raw:
        return _DEFAULT_QC_SEAM_BLOCKS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_QC_SEAM_BLOCKS
    return max(_MIN_QC_SEAM_BLOCKS, val)


# ---------------------------------------------------------------------------
# Unit-planning SCOPE knobs (owner-delegated: QC is an OUTPUT PROOFREADER, not a
# structural re-validator — the arranger lane's deterministic gates already carry
# structural validation, so QC judges a TARGETED slice instead of every window).
# ---------------------------------------------------------------------------
_QC_SCOPE_ENV = "SEMANTIK_REASONING_QC_SCOPE"
_QC_SAMPLE_PCT_ENV = "SEMANTIK_REASONING_QC_SAMPLE_PCT"
_DEFAULT_QC_SAMPLE_PCT = 15


def resolve_reasoning_qc_scope() -> str:
    """Return the QC unit-planning scope: ``"full"`` (DEFAULT) / ``"targeted"``.

    Reads ``SEMANTIK_REASONING_QC_SCOPE``. ``full`` (DEFAULT, incl. unset / blank /
    garbage / any non-``targeted`` value) → the historic behaviour: EVERY partition
    window + junction seam is judged (byte-identical). ``targeted`` → the unit plan
    KEEPS all seam strips + windows overlapping upstream-FLAGGED pages + a
    deterministic pseudo-random :func:`resolve_reasoning_qc_sample_pct` sample of the
    remaining windows; everything else is skipped with an honest per-window audit
    entry (no silent truncation). Read at CALL time (never cached at import)."""
    raw = (os.environ.get(_QC_SCOPE_ENV) or "").strip().lower()
    return "targeted" if raw == "targeted" else "full"


def resolve_reasoning_qc_sample_pct() -> int:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_SAMPLE_PCT`` (default 15, 0..100).

    The percentage of the NON-flagged, NON-seam windows the ``targeted`` scope
    keeps as a deterministic sample (seeded from the doc sha — stable across
    resumes, never wall-clock). Blank / non-int / garbage → the default 15; a value
    outside ``[0, 100]`` clamps into range (``0`` = seams + flagged pages only;
    ``100`` = keep every window, i.e. same coverage as ``full``). Read at CALL
    time. No-op unless the scope is ``targeted``."""
    raw = os.environ.get(_QC_SAMPLE_PCT_ENV)
    if not raw:
        return _DEFAULT_QC_SAMPLE_PCT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_QC_SAMPLE_PCT
    return max(0, min(100, val))


def resolve_reasoning_qc_mode() -> str:
    """Return the reasoning-QC mode: ``"off"`` / ``"shadow"`` / ``"on"``.

    Reads ``SEMANTIK_REASONING_QC``. Three-valued parse-with-fallback whose
    DEFAULT is ``off`` (byte-identical) — mirrors
    :func:`block_resegment.resolve_move_op_mode`'s shape but flips the default,
    because the whole point of this stage is that it is inert until an operator
    opts in:

      * ``on`` → ``on`` (apply reconcile ops);
      * ``shadow`` → ``shadow`` (verify + audit + capture, apply nothing);
      * everything else (``0``/``false``/``no``/``off`` / unset / blank /
        garbage) → ``off`` (the module is never imported by the caller).

    Read at CALL time (never cached at import).
    """
    raw = (os.environ.get("SEMANTIK_REASONING_QC") or "").strip().lower()
    if raw == "on":
        return "on"
    if raw == "shadow":
        return "shadow"
    return "off"


def resolve_reasoning_qc_model(default_model: str | None = None) -> str:
    """Thin re-export of :func:`reasoning_qc_vlm.resolve_reasoning_qc_model`.

    Kept here so the orchestrator + its audit can name the QC model id without
    importing the VLM transport module at their own top level.
    """
    from .reasoning_qc_vlm import resolve_reasoning_qc_model as _r

    return _r(default_model=default_model)


# ---------------------------------------------------------------------------
# Tiny FB accessors — replicated locally (NOT imported from cascade, which
# lazy-imports THIS module → a top-level cascade import would be circular).
# Byte-for-byte the cascade._fb_page / _fb_text semantics.
# ---------------------------------------------------------------------------
def _fb_page(feature_blocks: Sequence[Any], fb_idx: int) -> int | None:
    try:
        return int(getattr(feature_blocks[fb_idx].raw, "page"))
    except (IndexError, TypeError, AttributeError, ValueError):
        return None


def _fb_text(feature_blocks: Sequence[Any], fb_idx: int) -> str:
    try:
        return str(getattr(feature_blocks[fb_idx].raw, "text", "") or "")
    except (IndexError, TypeError, AttributeError):
        return ""


def _region_pages(region: Any, feature_blocks: Sequence[Any]) -> list[int]:
    pages: set[int] = set()
    for fb in getattr(region, "feature_block_indices", ()) or ():
        p = _fb_page(feature_blocks, fb)
        if p is not None:
            pages.add(p)
    return sorted(pages)


def _region_raw_text(region: Any, feature_blocks: Sequence[Any]) -> str:
    fb_idx = sorted(getattr(region, "feature_block_indices", ()) or ())
    return " ".join(t for fb in fb_idx if (t := _fb_text(feature_blocks, fb).strip()))


def _block_record(region: Any, feature_blocks: Sequence[Any]) -> dict[str, Any]:
    """One text-only QC block record from an assembled/capped region.

    Carries the structural signals the text-only judgment reads: ``type`` (the
    region ``kind``), ``role`` (the ``doc_role`` payload label, else kind),
    ``level`` (heading ``level_hint``), the primary physical ``page`` (or
    ``None``), and the region's verbatim ``text``. No image, no raster — the
    reasoning model judges order + structure from THIS."""
    kind = str(getattr(region, "kind", "") or "")
    payload = getattr(region, "payload", {}) or {}
    role = payload.get("doc_role") or kind
    level = payload.get("level_hint")
    pages = _region_pages(region, feature_blocks)
    page = pages[0] if pages else None
    return {
        "type": kind,
        "role": role,
        "level": level,
        "page": page,
        "text": _region_raw_text(region, feature_blocks),
    }


# ---------------------------------------------------------------------------
# ToC spine harvest (cascade-local; replicated N.M regex, advisory only).
# NOTE: these patterns are REPLICATED from the extractor's _NM_HEADING_RE /
# _NM_SPLIT_RE — deliberately NOT imported (the extractor is owned by a
# concurrent worker and off SemantiK's self-containment path).
# ---------------------------------------------------------------------------
_NM_HEADING_RE = re.compile(r"^\s*(\d+\.\d+)\s+(\S.*)$")
_NM_SPLIT_RE = re.compile(r"\b(\d+\.\d+)\b")

# Region kinds / role hints that plausibly carry a table-of-contents / outline.
_TOC_HINT_TOKENS = ("contents", "table of contents", "outline")


def _looks_like_toc_region(region: Any, text: str) -> bool:
    """Heuristic: a region whose text/role smells like a ToC / outline block."""
    lowered = text.lower()
    if any(tok in lowered[:400] for tok in _TOC_HINT_TOKENS):
        return True
    # A region carrying MANY N.M ordinals with little prose between them is a
    # ToC listing (the phantom-TOC signal the reviewer's cluster pass uses).
    ordinals = _NM_SPLIT_RE.findall(text)
    return len(ordinals) >= 3


def harvest_declared_toc_spine(
    regions: Sequence[Any], feature_blocks: Sequence[Any]
) -> dict[str, Any]:
    """Harvest the document's OWN declared ``N.M`` section spine.

    Scans each region's deterministic raw text for ToC/outline blocks and
    collects the ``N.M`` ordinals they declare, plus the ordinals that appear as
    an actual heading region opener. Returns::

        {
          "declared_ordinals": ["1.1", "1.2", ...],   # sorted, deduped
          "heading_ordinals":  ["1.1", ...],          # ordinals seen opening a heading region
          "toc_region_indices": [i, ...],
        }

    Advisory ONLY — the caller warns on a declared-but-absent ordinal and NEVER
    invents content for it. A weak harvest (empty ``declared_ordinals``) simply
    contributes no phantom-heading evidence.
    """
    declared: set[str] = set()
    heading_ordinals: set[str] = set()
    toc_idx: list[int] = []
    for i, region in enumerate(regions):
        text = _region_raw_text(region, feature_blocks)
        if not text:
            continue
        kind = str(getattr(region, "kind", ""))
        if _looks_like_toc_region(region, text):
            toc_idx.append(i)
            declared.update(_NM_SPLIT_RE.findall(text))
        if kind == "heading":
            m = _NM_HEADING_RE.match(text)
            if m:
                heading_ordinals.add(m.group(1))

    def _ord_key(s: str) -> tuple[int, int]:
        a, _, b = s.partition(".")
        try:
            return (int(a), int(b))
        except ValueError:
            return (0, 0)

    return {
        "declared_ordinals": sorted(declared, key=_ord_key),
        "heading_ordinals": sorted(heading_ordinals, key=_ord_key),
        "toc_region_indices": toc_idx,
    }


# ---------------------------------------------------------------------------
# Document-sequence QC unit (2026-07-12 pivot — document-level, not per-page).
# The WHOLE document's ordered block list is one sequence; _plan_page_units
# partitions THAT into windows + junction seams (seams now span page boundaries).
# ---------------------------------------------------------------------------
class QCWindow:
    """The document's ordered QC block sequence (one per converted document).

    ``block_records[k]`` is the text-only record (``{type, role, level, page,
    text}``) of the region at capped index ``region_indices[k]`` — so a judgment
    keyed by LOCAL block index ``k`` (within a partition window/seam) maps back to
    a ``capped`` region index for the reconcile op. ``page`` is a nominal
    representative (the first block's page) kept for logging/audit only — it no
    longer gates anything (there is no raster to render after the pivot).
    """

    __slots__ = ("page", "region_indices", "block_records", "emit_positions")

    def __init__(
        self,
        page: int,
        region_indices: list[int],
        block_records: list[dict[str, Any]],
        emit_positions: list[int],
    ) -> None:
        self.page = page
        self.region_indices = region_indices
        self.block_records = block_records
        self.emit_positions = emit_positions


def build_qc_windows(
    capped: Sequence[Any],
    feature_blocks: Sequence[Any],
    region_order: Sequence[int],
) -> list[QCWindow]:
    """Build the SINGLE document-level QC sequence (emission/reading order).

    ``region_order`` is the assembler's ``region_provenance`` — capped indices in
    emission (reading) order. The WHOLE document is one ordered block sequence
    (document-level pivot); each region contributes one text-only
    :func:`_block_record`. Returns a one-element list (the document sequence) so
    the downstream fan-out / stitch machinery — which partitions a window's block
    list into sub-windows + junction seams — operates over the whole document.
    Returns ``[]`` when no region resolves (nothing to judge)."""
    region_indices: list[int] = []
    block_records: list[dict[str, Any]] = []
    emit_positions: list[int] = []
    for emit_pos, ridx in enumerate(region_order):
        if not (0 <= ridx < len(capped)):
            continue
        region_indices.append(ridx)
        block_records.append(_block_record(capped[ridx], feature_blocks))
        emit_positions.append(emit_pos)
    if not region_indices:
        return []
    pages = [r["page"] for r in block_records if r["page"] is not None]
    page = pages[0] if pages else 0
    return [
        QCWindow(
            page=page,
            region_indices=region_indices,
            block_records=block_records,
            emit_positions=emit_positions,
        )
    ]


# ---------------------------------------------------------------------------
# Judgment → FlaggedBlock conversion (reuses reviewer.FlaggedBlock's 7-mode
# taxonomy; NO new failure-mode vocabulary).
# ---------------------------------------------------------------------------
# Re-type channel modes (partition-immutable; carry a per-block fix_hint).
_RETYPE_MODE = "wrong_semantic_class"
_PHANTOM_MODE = "example_as_heading"
_APPARATUS_MODE = "mistyped_component"
# Merge/move channel mode (partition-changing; carries proposed_regroup_run).
_MISORDER_MODE = "example_misordered_from_body"

# The re-type channel modes (partition-immutable) that route to the reviewer.
_RETYPE_FAILURE_MODES = frozenset({_RETYPE_MODE, _PHANTOM_MODE, _APPARATUS_MODE})

# Heading kinds this pass considers ToC-reconcilable (reviewer SoT; local
# fallback keeps toc_reconcile working even if the reviewer import is deferred).
_HEADING_KINDS_FALLBACK = frozenset({"heading"})

# reading_order-divergence floor (# of moved positions) at/above which
# page_order_verify SYNTHESIZES a misorder reorder candidate from a verdict's
# ``reading_order`` permutation (advisory; applied only under MOVE=live). A
# CALIBRATION constant — not a corpus target. See page_order_verify.
_MISORDER_MIN_POSITIONS = 2  # TODO(calibration): shadow->on precision bar


def judgments_to_flagged_blocks(
    verdict: dict[str, Any],
    window: QCWindow,
) -> list[Any]:
    """Convert one window's parsed VLM verdict into :class:`FlaggedBlock` records.

    Maps the JSON judgment onto the EXISTING reviewer taxonomy:

      * ``phantom_headings[*].index``  → ``example_as_heading`` (re-type/demote,
        ``fixable=True``, ``fix_hint`` carries the reason);
      * ``apparatus_retype[*].index``  → ``mistyped_component`` (re-type,
        ``fixable=True``);
      * ``misordered[*].run``          → ``example_misordered_from_body``
        (``fixable=False``, ``proposed_regroup_run`` = the run's CAPPED indices).

    Every emitted index is validated against ``window.region_indices`` — an
    out-of-range / unknown block index is dropped (the VLM was told to use only
    the given indices; a stray one is fail-soft ignored, never applied).
    """
    from .qwen_specialists.reviewer import FlaggedBlock

    n = len(window.region_indices)

    def _capped_of(block_idx: Any) -> int | None:
        try:
            k = int(block_idx)
        except (TypeError, ValueError):
            return None
        if 0 <= k < n:
            return window.region_indices[k]
        return None

    flagged: list[Any] = []

    for item in verdict.get("phantom_headings") or ():
        idx = _capped_of(item.get("index") if isinstance(item, dict) else item)
        if idx is None:
            continue
        reason = (item.get("reason") if isinstance(item, dict) else "") or "phantom heading"
        flagged.append(
            FlaggedBlock(
                region_index=idx,
                failure_mode=_PHANTOM_MODE,
                fix_hint=f"reasoning-QC: demote phantom heading — {reason}",
                fixable=True,
            )
        )

    for item in verdict.get("apparatus_retype") or ():
        idx = _capped_of(item.get("index") if isinstance(item, dict) else item)
        if idx is None:
            continue
        reason = (item.get("reason") if isinstance(item, dict) else "") or "apparatus mistyped"
        flagged.append(
            FlaggedBlock(
                region_index=idx,
                failure_mode=_APPARATUS_MODE,
                fix_hint=f"reasoning-QC: re-type apparatus — {reason}",
                fixable=True,
            )
        )

    for item in verdict.get("misordered") or ():
        run_raw = item.get("run") if isinstance(item, dict) else item
        if not isinstance(run_raw, (list, tuple)):
            continue
        capped_run = [c for c in (_capped_of(x) for x in run_raw) if c is not None]
        if len(capped_run) < 2:
            continue
        reason = (item.get("reason") if isinstance(item, dict) else "") or "order divergence"
        # Anchor the flag on the run's FIRST member (reviewer keys by
        # region_index); the full capped run rides in proposed_regroup_run.
        flagged.append(
            FlaggedBlock(
                region_index=capped_run[0],
                failure_mode=_MISORDER_MODE,
                fix_hint=f"reasoning-QC: reorder — {reason}",
                fixable=False,
                proposed_regroup_run=tuple(capped_run),
            )
        )

    return flagged


def _window_divergence(verdict: dict[str, Any], window: QCWindow) -> int:
    """Count of order positions the VLM's proposed reading_order moves.

    A cheap, replayable divergence signal for the audit + capture rationale
    (0 = the VLM agrees with the emitted order). Fail-soft on a malformed
    ``reading_order``.
    """
    proposed = verdict.get("reading_order")
    if not isinstance(proposed, (list, tuple)) or not proposed:
        return 0
    try:
        seq = [int(x) for x in proposed]
    except (TypeError, ValueError):
        return 0
    baseline = list(range(len(window.region_indices)))
    # Truncate/pad the compared prefix to the shorter length.
    m = min(len(seq), len(baseline))
    return sum(1 for i in range(m) if seq[i] != baseline[i])


# ---------------------------------------------------------------------------
# CHECK (A) — ToC reconcile (deterministic, cascade-local, advisory).
# ---------------------------------------------------------------------------
def _heading_kinds() -> frozenset[str]:
    """The reviewer's heading-kind SoT, with a local fallback."""
    try:
        from .qwen_specialists.reviewer import HEADING_KINDS

        return HEADING_KINDS
    except Exception:  # noqa: BLE001 — never fatal
        return _HEADING_KINDS_FALLBACK


def _is_apparatus_banner(text: str) -> bool:
    """Reuse the deterministic apparatus-opener SoT (never re-derive)."""
    try:
        from .qwen_specialists.deterministic_structure import _matches_apparatus_opener

        return bool(_matches_apparatus_opener(text))
    except Exception:  # noqa: BLE001 — fail-soft: no apparatus signal
        return False


def _body_follows(capped: Sequence[Any], idx: int, heading_kinds: frozenset[str]) -> bool:
    """Whether a body-bearing region immediately follows ``capped[idx]``.

    A phantom-TOC heading sits in a RUN of headings with no content between
    them, so "no body follows" == the next region is another heading (or the
    heading is the last region). Conservative — a real section heading is
    followed by its content.
    """
    nxt = idx + 1
    if nxt >= len(capped):
        return False
    return str(getattr(capped[nxt], "kind", "")) not in heading_kinds


def toc_reconcile(
    capped: Sequence[Any],
    feature_blocks: Sequence[Any],
    toc: dict[str, Any],
    *,
    log: Callable[[str], None] | None = None,
) -> list[Any]:
    """CHECK (A): compare extracted heading regions against the declared ToC spine.

    Emits :class:`FlaggedBlock` reconcile ops (with evidence in ``fix_hint``)
    that route to the partition-IMMUTABLE re-type channel:

      * an ``N.M`` heading whose ordinal is NOT in the document's own declared
        ToC spine AND has no body following it → ``example_as_heading`` (demote
        the phantom heading);
      * a heading whose text is WHOLLY an exercise/answer-key apparatus banner
        (reusing the deterministic ``_matches_apparatus_opener`` SoT) →
        ``mistyped_component`` (re-type the furniture heading).

    Degrades gracefully: an EMPTY declared spine (no ToC harvested) → skip with
    a logged reason, return ``[]`` (never crash). It NEVER invents an op for a
    declared-but-absent ordinal (warn-only — that lives in the orchestrator's
    ``declared_missing`` audit); a weak harvest simply contributes no ops.
    Deterministic — the VLM adjudicates ambiguous phantom-vs-real, not this pass.
    """
    _log = log or (lambda msg: logger.debug(msg))
    from .qwen_specialists.reviewer import FlaggedBlock

    declared = set(toc.get("declared_ordinals") or ())
    if not declared:
        _log("[cascade] reasoning-QC toc_reconcile: no declared ToC spine → skip (advisory)")
        return []

    heading_kinds = _heading_kinds()
    declared_preview = sorted(declared, key=lambda s: _ord_sort_key(s))[:8]
    flagged: list[Any] = []
    for idx, region in enumerate(capped):
        if str(getattr(region, "kind", "")) not in heading_kinds:
            continue
        text = _region_raw_text(region, feature_blocks).strip()
        if not text:
            continue
        if _is_apparatus_banner(text):
            flagged.append(
                FlaggedBlock(
                    region_index=idx,
                    failure_mode=_APPARATUS_MODE,
                    fix_hint=(
                        f"reasoning-QC ToC: heading {text[:60]!r} is an "
                        f"exercise/apparatus banner, not a section (declared "
                        f"spine={declared_preview}) → re-type furniture"
                    ),
                    fixable=True,
                )
            )
            continue
        m = _NM_HEADING_RE.match(text)
        if not m:
            continue
        ordv = m.group(1)
        if ordv in declared:
            continue
        if _body_follows(capped, idx, heading_kinds):
            # Declared-absent BUT has real content → likely a real section the
            # weak ToC harvest missed. Leave it (never demote real content).
            continue
        flagged.append(
            FlaggedBlock(
                region_index=idx,
                failure_mode=_PHANTOM_MODE,
                fix_hint=(
                    f"reasoning-QC ToC: heading ordinal {ordv} not in the "
                    f"declared spine {declared_preview} and no body follows "
                    f"→ demote phantom"
                ),
                fixable=True,
            )
        )
    if flagged:
        _log(
            f"[cascade] reasoning-QC toc_reconcile: {len(flagged)} reconcile op(s) "
            f"proposed against {len(declared)} declared ordinal(s)"
        )
    return flagged


def _ord_sort_key(s: str) -> tuple[int, int]:
    a, _, b = str(s).partition(".")
    try:
        return (int(a), int(b))
    except ValueError:
        return (0, 0)


# ---------------------------------------------------------------------------
# CHECK (B) — page reading-order verify (VLM, thinking-on).
# ---------------------------------------------------------------------------
def _synthesize_misorder_flag(verdict: dict[str, Any], window: QCWindow) -> Any | None:
    """Turn a ``reading_order`` permutation into a misorder reorder candidate.

    When the VLM returns a ``reading_order`` that diverges from the emitted
    order by ≥ :data:`_MISORDER_MIN_POSITIONS` positions AND no explicit
    ``misordered`` runs were given, synthesize ONE
    ``example_misordered_from_body`` :class:`FlaggedBlock` whose
    ``proposed_regroup_run`` is the window's CAPPED indices in the VLM's
    proposed order (bounded, ≥2). Advisory — the reorder is applied ONLY under
    ``SEMANTIK_MOVE_OP == 'live'`` (else audited-only). Returns ``None`` when
    there is no actionable divergence or the permutation is malformed.
    """
    proposed = verdict.get("reading_order")
    if not isinstance(proposed, (list, tuple)) or not proposed:
        return None
    try:
        seq = [int(x) for x in proposed]
    except (TypeError, ValueError):
        return None
    n = len(window.region_indices)
    # A valid permutation of the window's block indices only (out-of-range or
    # non-permutation reading_order is ignored — fail-soft).
    if sorted(seq) != list(range(n)) or n < 2:
        return None
    if _window_divergence(verdict, window) < _MISORDER_MIN_POSITIONS:
        return None
    from .qwen_specialists.reviewer import FlaggedBlock

    capped_run = [window.region_indices[k] for k in seq]
    reason = "reading_order divergence (VLM-proposed order)"
    return FlaggedBlock(
        region_index=capped_run[0],
        failure_mode=_MISORDER_MODE,
        fix_hint=f"reasoning-QC: reorder — {reason}",
        fixable=False,
        proposed_regroup_run=tuple(capped_run),
    )


# ---------------------------------------------------------------------------
# Primary-window PARTITION + junction SEAM plan + verdict STITCH.
#
# A page whose block list exceeds SEMANTIK_REASONING_QC_WINDOW_BLOCKS is
# partitioned into consecutive, NON-overlapping windows; each adjacent pair adds
# ONE seam strip (last S of window K + first S of window K+1). Every window and
# seam is a VLM UNIT over a contiguous block slice; single-window pages emit
# ZERO seams. Stitch precedence (owner amendment): window verdicts own INTRA
# findings (re-type + intra order), seam verdicts are AUTHORITATIVE for
# cross-boundary order; assembly order is all windows (in window order) then all
# seams (in seam order); any window/seam flag ⇒ page flagged.
# ---------------------------------------------------------------------------
def _plan_page_units(n_blocks: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return ``(windows, seams)`` — each a list of ``[start, end)`` block ranges.

    ``windows`` PARTITION ``[0, n_blocks)`` exactly (consecutive, non-overlapping,
    ≤ ``SEMANTIK_REASONING_QC_WINDOW_BLOCKS``). ``seams`` has one contiguous strip
    per junction — ``[junction - S, junction + S)`` clamped to the two adjacent
    windows — so ``len(seams) == max(0, len(windows) - 1)``. A page that fits one
    window (``n_blocks <= WINDOW_BLOCKS``) → one window, ZERO seams."""
    win_size = resolve_reasoning_qc_window_blocks()
    if n_blocks <= win_size:
        return [(0, n_blocks)], []
    windows: list[tuple[int, int]] = []
    s = 0
    while s < n_blocks:
        e = min(s + win_size, n_blocks)
        windows.append((s, e))
        s = e
    seam_half = resolve_reasoning_qc_seam_blocks()
    seams: list[tuple[int, int]] = []
    for k in range(len(windows) - 1):
        junction = windows[k][1]  # == windows[k + 1][0]
        seam_start = max(windows[k][0], junction - seam_half)
        seam_end = min(windows[k + 1][1], junction + seam_half)
        seams.append((seam_start, seam_end))
    return windows, seams


# ---------------------------------------------------------------------------
# TARGETED-scope planning (SEMANTIK_REASONING_QC_SCOPE == 'targeted').
#
# QC is an OUTPUT PROOFREADER. In targeted mode the unit plan KEEPS: (a) ALL
# junction seam strips (cross-boundary reading order is QC's unique value the
# deterministic gates can't carry); (b) windows overlapping pages FLAGGED by
# upstream signals available AT QC TIME (arranger interventions + structure-review
# block changes — see derive_flagged_pages); (c) a deterministic pseudo-random
# doc-sha-seeded sample of the remaining windows. Everything else is skipped with
# an explicit per-window audit entry (no silent truncation).
# ---------------------------------------------------------------------------
def _qc_document_sha(window: QCWindow) -> str:
    """A stable, content-addressed document sha for the deterministic sample seed.

    Hashes the whole document sequence's block texts in reading order, so the
    per-window sample draw is STABLE across resumes + re-runs of the same input
    (never wall-clock). Independent of scope / sample_pct — those salt the
    DOCUMENT-level audit only, never a unit fingerprint, so cached unit verdicts
    stay valid across a scope change."""
    h = hashlib.sha256()
    for rec in window.block_records:
        h.update((rec.get("text") or "").encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _page_of_region(capped: Sequence[Any], feature_blocks: Sequence[Any], ridx: int) -> int | None:
    """The representative (first) physical page of ``capped[ridx]`` (or ``None``)."""
    if not (0 <= ridx < len(capped)):
        return None
    pages = _region_pages(capped[ridx], feature_blocks)
    return pages[0] if pages else None


def derive_flagged_pages(
    capped: Sequence[Any],
    feature_blocks: Sequence[Any],
    *,
    arranger_audit: dict[str, Any] | None = None,
    review_verdicts: Any = None,
) -> set[int]:
    """Collect the physical pages upstream signals FLAGGED as worth full QC.

    Threadable signals available AT QC TIME (Stage-9b) — documented honestly
    because the obvious one is NOT reachable here:

    * **unit_coverage is NOT threadable.** The ``SEMANTIK_UNIT_COVERAGE_GATE``
      report is produced at the cascade EXIT — strictly AFTER Stage-9b QC in
      cascade order — so its warn/floor pages do not yet exist when QC plans its
      units. It is therefore intentionally NOT consulted here.
    * **Page-arranger audit** (``arranger_audit['page_rows']``, in cascade scope
      at the QC seam). A page is flagged when the arranger STRUGGLED there — a
      failed-page fallback (``status != 'ok'``), a heading-sanity intervention
      (``heading_sanity > 0``), an arrangement retry (``attempts > 1``), or a
      coercion / repair (``coercions`` / ``repairs`` > 0). These are exactly the
      "pages with heading-sanity interventions / arrangement retries / failed-page
      fallbacks" the owner named.
    * **Structure-review verdicts** (``review_verdicts``) — the closest threadable
      "blocks changed" signal. The literal Stage-6 authoring diff is NOT surfaced
      per-page at QC time (the QC input is FB-derived verbatim text; Stage-6 HTML
      candidates feed only internal gate/theta signals, never a per-page ledger),
      so the block-change signal used is the reviewer / Stage-9 second-pass region
      RE-TYPE set: a region whose ``kind_before != kind_after`` and was not
      reverted has its page flagged.

    Coarse by design — a page-level union that ADMITS a window to the full QC
    slice; over-flagging is safe (more proofreading), under-flagging is covered by
    the deterministic sample. Returns a set of int page numbers (empty when no
    signal is threaded)."""
    flagged: set[int] = set()
    for row in (arranger_audit or {}).get("page_rows") or ():
        if not isinstance(row, dict):
            continue
        struggled = (
            row.get("status") not in (None, "ok")
            or _as_int(row.get("heading_sanity")) > 0
            or _as_int(row.get("attempts")) > 1
            or _as_int(row.get("coercions")) > 0
            or _as_int(row.get("repairs")) > 0
        )
        if not struggled:
            continue
        try:
            flagged.add(int(row.get("page")))
        except (TypeError, ValueError):
            continue
    for v in review_verdicts or ():
        try:
            if bool(getattr(v, "reverted_for_invariant", False)) or bool(
                getattr(v, "reverted_for_endpoint_failure", False)
            ):
                continue
            before = getattr(v, "kind_before", None)
            after = getattr(v, "kind_after", None)
            if before is None or after is None or before == after:
                continue
            ridx = int(getattr(v, "region_index"))
        except (TypeError, ValueError, AttributeError):
            continue
        p = _page_of_region(capped, feature_blocks, ridx)
        if p is not None:
            flagged.add(p)
    return flagged


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _targeted_window_pages(window: QCWindow, s: int, e: int) -> list[int]:
    """The sorted distinct pages a ``[s, e)`` sub-window's block records cover."""
    pages: set[int] = set()
    for rec in window.block_records[s:e]:
        p = rec.get("page")
        if p is not None:
            pages.add(p)
    return sorted(pages)


def _targeted_keep_window(
    window: QCWindow,
    s: int,
    e: int,
    *,
    flagged_pages: frozenset[int] | set[int],
    sample_pct: int,
    doc_sha: str,
) -> tuple[bool, str, list[int]]:
    """Decide whether a ``[s, e)`` sub-window survives ``targeted`` scope.

    Returns ``(keep, reason, pages)`` — ``reason`` ∈ ``{"flagged", "sample",
    "targeted_scope"}`` (the last = SKIP). A window overlapping any flagged page is
    kept (``"flagged"``); otherwise a deterministic pseudo-random draw seeded from
    ``doc_sha`` + the window's ``[s, e)`` identity keeps ``sample_pct``% of the rest
    (``"sample"``); else skip (``"targeted_scope"``). The draw is content-addressed
    (never wall-clock) so it is identical across resumes."""
    pages = _targeted_window_pages(window, s, e)
    if flagged_pages and (set(pages) & set(flagged_pages)):
        return True, "flagged", pages
    draw = int(hashlib.sha256(f"{doc_sha}|{s}|{e}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if draw < sample_pct:
        return True, "sample", pages
    return False, "targeted_scope", pages


def _qc_item_index(item: Any) -> int | None:
    raw = item.get("index") if isinstance(item, dict) else item
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _qc_item_reason(item: Any, default: str) -> str:
    if isinstance(item, dict):
        return str(item.get("reason") or default)
    return default


def _qc_run_indices(r: Any) -> list[int] | None:
    raw = r.get("run") if isinstance(r, dict) else r
    if not isinstance(raw, (list, tuple)):
        return None
    out: list[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            return None
    return out


def _qc_as_conf(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _qc_int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for x in value:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _qc_reading_order_run(v: dict[str, Any], unit_len: int) -> list[int] | None:
    """A unit's ``reading_order`` as a divergent permutation of its local range.

    Returns the proposed local order (list) when it is a valid permutation of
    ``range(unit_len)`` diverging by ≥ :data:`_MISORDER_MIN_POSITIONS`; else
    ``None`` (fail-soft on a malformed / non-divergent / non-permutation order).
    """
    proposed = v.get("reading_order")
    if not isinstance(proposed, (list, tuple)) or not proposed:
        return None
    try:
        seq = [int(x) for x in proposed]
    except (TypeError, ValueError):
        return None
    if unit_len < 2 or sorted(seq) != list(range(unit_len)):
        return None
    if sum(1 for i in range(unit_len) if seq[i] != i) < _MISORDER_MIN_POSITIONS:
        return None
    return seq


def _qc_local_divergence(v: dict[str, Any], unit_len: int) -> int:
    """The unit's reading_order position-move count vs its local baseline order."""
    proposed = v.get("reading_order")
    if not isinstance(proposed, (list, tuple)) or not proposed:
        return 0
    try:
        seq = [int(x) for x in proposed]
    except (TypeError, ValueError):
        return 0
    m = min(len(seq), unit_len)
    return sum(1 for i in range(m) if seq[i] != i)


def _absorb_window_findings(
    v: dict[str, Any], start: int, end: int, merged: dict[str, list]
) -> None:
    """Fold a WINDOW unit's verdict (INTRA findings) into the page-merged verdict.

    Window verdicts own the per-block re-type (phantom / apparatus) findings AND
    intra-window order. Indices are remapped from unit-local (0-based) to page
    positions (``+start``)."""
    unit_len = end - start
    for key in ("phantom_headings", "apparatus_retype"):
        for item in v.get(key) or ():
            idx = _qc_item_index(item)
            if idx is None or not (0 <= idx < unit_len):
                continue
            merged[key].append({"index": start + idx, "reason": _qc_item_reason(item, "")})
    explicit = False
    for r in v.get("misordered") or ():
        run = _qc_run_indices(r)
        if run is None:
            continue
        remapped = [start + i for i in run if 0 <= i < unit_len]
        if len(remapped) >= 2:
            merged["misordered"].append(
                {"run": remapped, "reason": _qc_item_reason(r, "order divergence")}
            )
            explicit = True
    if not explicit:
        seq = _qc_reading_order_run(v, unit_len)
        if seq is not None:
            merged["misordered"].append(
                {
                    "run": [start + k for k in seq],
                    "reason": "reasoning-QC window: reading_order divergence",
                }
            )


def _absorb_seam_findings(
    v: dict[str, Any], start: int, end: int, merged: dict[str, list]
) -> None:
    """Fold a SEAM unit's verdict (CROSS-boundary ORDER only) into the merge.

    Seam verdicts are authoritative for cross-boundary order (reading-order
    across the junction) — the thing no single partition window can see. A seam's
    per-block NON-order finding (phantom / apparatus) is DISCARDED: the window
    that fully contains the block saw more surrounding context and wins."""
    unit_len = end - start
    explicit = False
    for r in v.get("misordered") or ():
        run = _qc_run_indices(r)
        if run is None:
            continue
        remapped = [start + i for i in run if 0 <= i < unit_len]
        if len(remapped) >= 2:
            merged["misordered"].append(
                {"run": remapped, "reason": _qc_item_reason(r, "cross-boundary order divergence")}
            )
            explicit = True
    if not explicit:
        seq = _qc_reading_order_run(v, unit_len)
        if seq is not None:
            merged["misordered"].append(
                {
                    "run": [start + k for k in seq],
                    "reason": "reasoning-QC seam: cross-boundary reading_order divergence",
                }
            )


def _stitch_from_raw(
    raw_by_unit: dict[tuple, dict[str, Any]],
    w_idx: int,
    window: QCWindow,
    subwins: list[tuple[int, int]],
    seams: list[tuple[int, int]],
) -> tuple[dict[str, Any], int]:
    """Stitch a page's window + seam unit verdicts into ONE page verdict.

    Single-window pages return the raw window verdict UNCHANGED (byte-identical
    to the legacy single-call path, so ``reading_order`` synthesis + divergence
    downstream are preserved). Multi-window pages assemble window findings first
    (window order) then seam findings (seam order); ``reading_order`` is dropped
    (no global permutation across the partition) and the synthesized per-unit
    misordered runs carry order divergence instead. ``confidence`` is the min
    across units; ``_qc_incomplete`` is the union (remapped to page positions).
    Returns ``(merged_verdict, aggregate_divergence)``."""
    if len(subwins) == 1 and not seams:
        v = raw_by_unit.get((w_idx, "window", subwins[0][0], subwins[0][1])) or {}
        return v, _window_divergence(v, window)
    merged: dict[str, Any] = {"phantom_headings": [], "apparatus_retype": [], "misordered": []}
    confs: list[float] = []
    incomplete: list[int] = []
    total_div = 0
    for (s, e) in subwins:
        v = raw_by_unit.get((w_idx, "window", s, e)) or {}
        _absorb_window_findings(v, s, e, merged)
        c = _qc_as_conf(v.get("confidence"))
        if c is not None:
            confs.append(c)
        incomplete.extend(s + i for i in _qc_int_list(v.get("_qc_incomplete")))
        total_div += _qc_local_divergence(v, e - s)
    for (s, e) in seams:
        v = raw_by_unit.get((w_idx, "seam", s, e)) or {}
        _absorb_seam_findings(v, s, e, merged)
        c = _qc_as_conf(v.get("confidence"))
        if c is not None:
            confs.append(c)
        incomplete.extend(s + i for i in _qc_int_list(v.get("_qc_incomplete")))
        total_div += _qc_local_divergence(v, e - s)
    if confs:
        merged["confidence"] = min(confs)
    if incomplete:
        merged["_qc_incomplete"] = sorted(set(incomplete))
    merged = {k: val for k, val in merged.items() if not (isinstance(val, list) and not val)}
    return merged, total_div


def _stitch_and_flag(
    raw_by_unit: dict[tuple, dict[str, Any]],
    w_idx: int,
    window: QCWindow,
    subwins: list[tuple[int, int]],
    seams: list[tuple[int, int]],
) -> tuple[dict[str, Any], list[Any], int]:
    """Stitch the page's units, then convert to ``(verdict, flagged, divergence)``."""
    merged, divergence = _stitch_from_raw(raw_by_unit, w_idx, window, subwins, seams)
    if not merged:
        return {}, [], 0
    flagged = judgments_to_flagged_blocks(merged, window)
    # Synthesize a reorder candidate from a WHOLE-PAGE reading_order ONLY when the
    # verdict did not already carry a misordered run (single-window pages keep the
    # legacy synth; multi-window pages already carry per-unit synthesized runs).
    if not (merged.get("misordered") or ()):
        synth = _synthesize_misorder_flag(merged, window)
        if synth is not None:
            flagged.append(synth)
    return merged, flagged, divergence


def order_verify(
    seat: Any,
    pdf_path: Any,
    window: QCWindow,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[Any], int]:
    """CHECK (B): judge the document sequence's reading order + structure (text).

    PARTITIONS the document's ordered block list into ≤``SEMANTIK_REASONING_QC_WINDOW_BLOCKS``
    non-overlapping windows + junction seam strips (seams now straddle page
    boundaries too), judges each unit thinking-on via :func:`_run_qc_judgment`
    (SEQUENTIALLY here — the concurrent fan-out is :func:`_fan_out_page_verifies`),
    and STITCHES the unit verdicts into ``(verdict, flagged, divergence)``:

      * ``verdict`` — the merged judgment dict (``{}`` on a fail-soft miss);
      * ``flagged`` — :class:`FlaggedBlock` records (phantom / apparatus re-type
        + explicit-``misordered`` reorder + a reading_order-derived reorder);
      * ``divergence`` — the reading_order position-move count (audit signal).

    Degrades gracefully: a transport/parse failure inside :func:`_run_qc_judgment`
    degrades to an empty verdict (the reasoning client is fail-soft) — this NEVER
    raises into the cascade. ``pdf_path`` is retained for provenance/logging only
    (no page render after the 2026-07-12 text-only pivot).
    """
    _log = log or (lambda msg: logger.debug(msg))
    subwins, seams = _plan_page_units(len(window.block_records))
    raw_by_unit: dict[tuple, dict[str, Any]] = {}
    for (s, e) in subwins:
        raw_by_unit[(0, "window", s, e)] = _run_qc_judgment(
            seat, pdf_path, window.page, list(window.block_records[s:e])
        )
    for (s, e) in seams:
        raw_by_unit[(0, "seam", s, e)] = _run_qc_judgment(
            seat, pdf_path, window.page, list(window.block_records[s:e])
        )
    return _stitch_and_flag(raw_by_unit, 0, window, subwins, seams)


# Back-compat alias — the historic per-page name.
page_order_verify = order_verify


# ---------------------------------------------------------------------------
# DecisionCapture wiring (best-effort, lazy — mirrors figure_captioner).
# One decision per QC unit (window), decision_type 'structure_detection'.
# ---------------------------------------------------------------------------
def _qc_course_code() -> str:
    raw = (
        os.environ.get("SEMANTIK_COURSE_CODE")
        or os.environ.get("ED4ALL_COURSE_CODE")
        or ""
    ).strip()
    return raw or "SEMANTIK"


def _build_reasoning_qc_capture():
    """Construct a best-effort DecisionCapture for the QC VLM call site.

    Returns ``None`` (QC proceeds unaffected) when ``lib.decision_capture`` is
    unavailable or construction fails — mirroring the ``structure_review``
    DecisionCapture best-effort posture in ``MCP/tools/pipeline_tools.py`` and
    the figure-captioner capture. Lands under the canonical ``dart`` tool /
    ``dart-conversion`` phase.
    """
    try:
        from lib.decision_capture import DecisionCapture

        return DecisionCapture(
            course_code=_qc_course_code(),
            phase="dart-conversion",
            tool="dart",
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("reasoning-QC DecisionCapture unavailable (non-fatal): %s", exc)
        return None


# Per-document memoized capture — one capture per run_reasoning_qc invocation,
# reused across every window's decision (set/cleared by run_reasoning_qc).
_document_capture: Any | None = None


def _log_qc_decision(
    capture,
    *,
    window: QCWindow,
    model: str,
    verdict: dict[str, Any],
    flagged: Sequence[Any],
    divergence: int,
) -> None:
    """Emit ONE ``structure_detection`` decision for the document QC judgment.

    Rationale is dynamic + replayable (page span, model id, a BOUNDED region-index
    sample, block count, divergence, model confidence, per-mode flag counts) so a
    post-hoc replay can attribute the judgment to its exact input. Best-effort:
    never raises into the QC path.
    """
    if capture is None:
        return
    try:
        conf = verdict.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        modes = sorted({str(getattr(f, "failure_mode", "?")) for f in flagged})
        region_set = list(window.region_indices)
        n = len(region_set)
        # Bound the region dump — the document sequence can carry thousands of
        # regions; a replayable rationale keeps head + tail + a count.
        if n <= 20:
            region_repr = str(region_set)
        else:
            region_repr = (
                f"[{', '.join(str(x) for x in region_set[:10])}, "
                f"…(+{n - 20} more)…, "
                f"{', '.join(str(x) for x in region_set[-10:])}]"
            )
        pages = [b.get("page") for b in window.block_records if b.get("page") is not None]
        page_span = f"{min(pages)}-{max(pages)}" if pages else "n/a"
        # Honestly record any blocks the split ladder could not judge thinking-on
        # (qc_incomplete) — post-hoc analysis must never mistake a skipped block
        # for a clean pass. There is NO thinking-off fallback discriminator: the
        # QC judgment is thinking-on by construction (OWNER DIRECTIVE).
        incomplete = _qc_int_list(verdict.get("_qc_incomplete"))
        incomplete_tag = (
            f" qc_incomplete after split ladder exhausted (block_positions={incomplete})"
            if incomplete
            else ""
        )
        rationale = (
            f"reasoning-QC document judgment over pages {page_span} "
            f"(n_blocks={n}, regions={region_repr}): "
            f"model={model} confidence={conf} order_divergence={divergence}{incomplete_tag}; "
            f"flagged {len(flagged)} block(s) modes={modes}; "
            f"phantom={len(verdict.get('phantom_headings') or ())} "
            f"apparatus={len(verdict.get('apparatus_retype') or ())} "
            f"misordered={len(verdict.get('misordered') or ())}."
        )
        capture.log_decision(
            decision_type="structure_detection",
            decision=(
                f"reasoning-QC document (pages {page_span}): {len(flagged)} flag(s), "
                f"divergence={divergence}"
            ),
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug(
            "reasoning-QC DecisionCapture log failed (non-fatal) on page %s: %s",
            window.page, exc,
        )


# ---------------------------------------------------------------------------
# Call seams (monkeypatched in tests — no torch/ollama/weights).
# ---------------------------------------------------------------------------
def _resolve_qc_seat():
    """Resolve + require_ready the QC seat (delegates to the VLM module)."""
    from .reasoning_qc_vlm import resolve_reasoning_qc_seat

    return resolve_reasoning_qc_seat()


def _run_qc_judgment(seat, pdf_path, page_num, blocks) -> dict[str, Any]:
    """POST one unit's TEXT judgment (delegates to the reasoning-QC client).

    ``blocks`` is the list of text-only block RECORDS for the unit; ``pdf_path``
    /``page_num`` are provenance/logging only (no render after the 2026-07-12
    text-only pivot)."""
    from .reasoning_qc_vlm import run_qc_judgment

    return run_qc_judgment(seat, pdf_path, page_num, blocks)


# ---------------------------------------------------------------------------
# Op-apply seams (monkeypatched in STEP-2 tests — no LLM / GPU). Each delegates
# to the EXISTING block-ID op layer; the QC module PROPOSES, that layer APPLIES.
# ---------------------------------------------------------------------------
def _run_structure_review(regions, feature_blocks, runtime, *, restrict_to, feedback_by_idx):
    from .qwen_specialists.reviewer import run_structure_review

    return run_structure_review(
        regions,
        feature_blocks,
        runtime,
        restrict_to=restrict_to,
        feedback_by_idx=feedback_by_idx,
    )


def _apply_proposed_unit_fix(regions, feature_blocks, proposed_runs):
    from .qwen_specialists.block_resegment import apply_proposed_unit_fix

    return apply_proposed_unit_fix(regions, feature_blocks, proposed_runs)


def _assert_partition_conservation(in_regions, out_regions, *, move_ops=()):
    from .qwen_specialists.block_resegment import assert_partition_conservation

    assert_partition_conservation(in_regions, out_regions, move_ops=move_ops)


def _assert_token_conservation(original_regions, corrected_regions, feature_blocks):
    from .qwen_specialists.reviewer import assert_token_conservation

    assert_token_conservation(original_regions, corrected_regions, feature_blocks)


def _build_resegment_audit_rows(ops):
    from .qwen_specialists.block_resegment import build_resegment_audit_rows

    return build_resegment_audit_rows(list(ops))


def _resolve_move_op_mode() -> str:
    from .qwen_specialists.block_resegment import resolve_move_op_mode

    return resolve_move_op_mode()


# ---------------------------------------------------------------------------
# Op-apply-with-conservation channels (STEP 2).
#
# Each channel PROPOSES nothing of its own — it takes the FlaggedBlock records
# CHECK (A)/(B) produced and drives the EXISTING block-ID op layer, then guards
# every applied op with the conservation asserts and WHOLE-REVERTS on any raise
# (fail-closed). Never raises into the cascade.
# ---------------------------------------------------------------------------
def _region_kinds(regions: Sequence[Any]) -> list[str]:
    return [str(getattr(r, "kind", "")) for r in regions]


def _apply_retype_ops(
    capped: list[Any],
    feature_blocks: Sequence[Any],
    review_verdicts: Any,
    flagged: Sequence[Any],
    *,
    review_runtime: Any,
    log: Callable[[str], None],
) -> tuple[list[Any], Any, dict[str, Any]]:
    """Apply the partition-IMMUTABLE re-type / drop-phantom / re-type-apparatus ops.

    Collects the fixable re-type flags (``wrong_semantic_class`` /
    ``example_as_heading`` / ``mistyped_component``) into ``restrict_to`` +
    ``feedback_by_idx`` and drives :func:`reviewer.run_structure_review`, which
    re-types the flagged regions in place (verbatim source rides deterministic
    assembly). Fail-closed ladder:

      * no ``review_runtime`` (the reviewer seat is unavailable) → applied=0
        (the flags stay advisory — never a silent stub);
      * ``run_structure_review`` raises → whole-revert to the input snapshot;
      * the region COUNT drifts (re-type must be partition-immutable) →
        whole-revert;
      * :func:`reviewer.assert_token_conservation` raises → whole-revert.

    ``applied`` counts the flagged regions whose ``kind`` ACTUALLY changed (a
    reviewer that DECLINED a correction contributes 0 — the never-ship-worse
    property: an un-acted round ships the snapshot).
    """
    fixable = [
        f
        for f in flagged
        if getattr(f, "fixable", False)
        and getattr(f, "failure_mode", None) in _RETYPE_FAILURE_MODES
    ]
    # Dedup by region_index (ToC + VLM can both flag the same heading); keep the
    # first hint, so restrict_to is a clean set.
    seen: set[int] = set()
    deduped: list[Any] = []
    for f in fixable:
        ridx = getattr(f, "region_index", None)
        if ridx is None or ridx in seen:
            continue
        seen.add(int(ridx))
        deduped.append(f)
    proposed = len(deduped)
    if not proposed:
        return capped, review_verdicts, {"applied": 0, "proposed": 0}
    if review_runtime is None:
        log(
            f"[cascade] reasoning-QC re-type channel: {proposed} proposed op(s), "
            f"NO reviewer runtime → advisory (applied=0)"
        )
        return capped, review_verdicts, {"applied": 0, "proposed": proposed}

    restrict_to = frozenset(int(getattr(f, "region_index")) for f in deduped)
    feedback = {int(getattr(f, "region_index")): getattr(f, "fix_hint", "") or "" for f in deduped}
    snapshot = list(capped)
    snapshot_kinds = _region_kinds(snapshot)

    try:
        new_capped, new_verdicts = _run_structure_review(
            capped, feature_blocks, review_runtime, restrict_to=restrict_to, feedback_by_idx=feedback
        )
    except Exception as exc:  # noqa: BLE001 — fail-closed: keep the snapshot.
        log(f"[cascade] reasoning-QC re-type channel: reviewer raised ({exc}) → revert")
        return snapshot, review_verdicts, {"applied": 0, "proposed": proposed}

    new_capped = list(new_capped)
    if len(new_capped) != len(snapshot):
        log(
            f"[cascade] reasoning-QC re-type channel: region count drifted "
            f"({len(snapshot)}→{len(new_capped)}) → revert (partition-immutable)"
        )
        return snapshot, review_verdicts, {"applied": 0, "proposed": proposed}

    try:
        _assert_token_conservation(snapshot, new_capped, feature_blocks)
    except Exception as exc:  # noqa: BLE001 — TokenConservationError → fail-closed.
        log(f"[cascade] reasoning-QC re-type channel: token conservation FAILED ({exc}) → revert")
        return snapshot, review_verdicts, {"applied": 0, "proposed": proposed}

    new_kinds = _region_kinds(new_capped)
    applied = sum(
        1 for i in restrict_to if 0 <= i < len(new_kinds) and new_kinds[i] != snapshot_kinds[i]
    )
    if applied:
        log(f"[cascade] reasoning-QC re-type channel: {applied}/{proposed} op(s) applied")
        return new_capped, new_verdicts, {"applied": applied, "proposed": proposed}
    # Reviewer declined every correction → ship the snapshot (never-ship-worse).
    return snapshot, review_verdicts, {"applied": 0, "proposed": proposed}


def _apply_merge_move_ops(
    capped: list[Any],
    feature_blocks: Sequence[Any],
    flagged: Sequence[Any],
    *,
    log: Callable[[str], None],
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Apply the partition-CHANGING merge / reorder(move) ops.

    Collects the ``example_misordered_from_body`` flags' ``proposed_regroup_run``
    tuples and drives :func:`block_resegment.apply_proposed_unit_fix`. Reorder is
    DOUBLY gated: this whole channel is a NO-OP unless the orthogonal
    ``SEMANTIK_MOVE_OP == 'live'`` (default ``shadow`` → audited-only, byte
    identical). Under ``live`` the fix is applied, then guarded by BOTH
    :func:`block_resegment.assert_partition_conservation` (move-aware) +
    :func:`reviewer.assert_token_conservation`, with a WHOLE-REVERT to the input
    snapshot on any raise. Returns ``(new_capped, audit_rows, stats)``.
    """
    runs = [
        list(getattr(f, "proposed_regroup_run", ()) or ())
        for f in flagged
        if getattr(f, "failure_mode", None) == _MISORDER_MODE
        and (getattr(f, "proposed_regroup_run", ()) or ())
    ]
    proposed = len(runs)
    if not proposed:
        return capped, [], {"applied": 0, "proposed": 0}

    if _resolve_move_op_mode() != "live":
        # Default shadow / off → reorder is audit-only, never applied.
        log(
            f"[cascade] reasoning-QC merge/move channel: {proposed} proposed reorder(s), "
            f"SEMANTIK_MOVE_OP!=live → audited-only (applied=0)"
        )
        return capped, [], {"applied": 0, "proposed": proposed}

    snapshot = list(capped)
    try:
        new_capped, ops = _apply_proposed_unit_fix(capped, feature_blocks, runs)
    except Exception as exc:  # noqa: BLE001 — apply is documented never-raise; belt.
        log(f"[cascade] reasoning-QC merge/move channel: apply raised ({exc}) → revert")
        return snapshot, [], {"applied": 0, "proposed": proposed}

    new_capped = list(new_capped)
    ops = list(ops or ())
    if not ops:
        # Every proposed run dropped by the gates → byte-identical no-op.
        log(f"[cascade] reasoning-QC merge/move channel: {proposed} reorder(s) dropped by gates")
        return snapshot, [], {"applied": 0, "proposed": proposed}

    move_ops = [o for o in ops if getattr(o, "op", None) == "move"]
    try:
        _assert_partition_conservation(snapshot, new_capped, move_ops=move_ops)
        _assert_token_conservation(snapshot, new_capped, feature_blocks)
    except Exception as exc:  # noqa: BLE001 — conservation raise → fail-closed.
        log(f"[cascade] reasoning-QC merge/move channel: conservation FAILED ({exc}) → revert")
        return snapshot, [], {"applied": 0, "proposed": proposed}

    audit_rows = _build_resegment_audit_rows(ops)
    log(f"[cascade] reasoning-QC merge/move channel: {len(ops)} op(s) applied")
    return new_capped, audit_rows, {"applied": len(ops), "proposed": proposed}


# ---------------------------------------------------------------------------
# Per-UNIT resume cache/sidecar (stop-on-command checkpointing contract).
#
# Every QC judgment UNIT (window sub-slice OR junction seam) is a content-
# addressed sidecar: the fingerprint keys the EXACT reasoning input (the rendered
# user text that would be POSTed + the model id + the prompt-contract version +
# the thinking-disabled bool + the unit kind). A completed unit persists its
# verdict, so worst-case loss on an `ed4all stop` / crash is only the in-flight
# calls — a resume re-fans-out and every already-judged unit is a cache HIT (no
# re-POST). Mirrors the vlm_extract per-page disk cache (single-writer,
# atomic-ish temp+rename, sharded by key[:2]).
# ---------------------------------------------------------------------------
_REASONING_QC_CACHE_BASENAME = "reasoning_qc_cache"

# Site checkpoint flag (default ON) → falls back to the ED4ALL_GENERATION_CHECKPOINT
# family. Mirrors lib.generation.llm_checkpoint.checkpoint_enabled's semantics,
# re-implemented locally (SemantiK's self-containment boundary forbids a lib/
# import — the cross-venv conversion subprocess has no Ed4All lib on its path).
_QC_CHECKPOINT_ENV = "SEMANTIK_REASONING_QC_CHECKPOINT"
_QC_CHECKPOINT_FAMILY_ENV = "ED4ALL_GENERATION_CHECKPOINT"
_QC_CHECKPOINT_FALSEY = frozenset({"0", "false", "no", "off"})


def resolve_reasoning_qc_checkpoint() -> bool:
    """Resolve the per-unit resume-cache gate (default ON). Read at CALL time.

    Precedence: the site env ``SEMANTIK_REASONING_QC_CHECKPOINT``, WHEN SET
    (non-blank), wins (falsey token → off, anything else → on); otherwise the
    family ``ED4ALL_GENERATION_CHECKPOINT`` decides (falsey token → off, unset /
    garbage → on). Off → byte-identical: the fan-out neither reads nor writes the
    sidecar (no cache dir is ever created)."""
    site = os.environ.get(_QC_CHECKPOINT_ENV, "")
    if site.strip():
        return site.strip().lower() not in _QC_CHECKPOINT_FALSEY
    return os.environ.get(_QC_CHECKPOINT_FAMILY_ENV, "").strip().lower() not in _QC_CHECKPOINT_FALSEY


def _qc_cache_root() -> Path:
    """The content-addressed reasoning-QC sidecar root (CWD-independent).

    Mirrors :func:`vlm_extract._cache_root` — resolves via
    ``semantik_structure.paths.resolve_cache`` so it honours ``SEMANTIK_HOME`` /
    ``SEMANTIK_CACHE_DIR`` and is stable regardless of the subprocess CWD."""
    from . import paths as _semantik_paths

    return _semantik_paths.resolve_cache(_REASONING_QC_CACHE_BASENAME)


def _qc_unit_fingerprint(
    blocks: Sequence[Any], *, model: str, kind: str
) -> str:
    """Content-address one QC unit → a sha256 hex key.

    The fingerprint hashes the unit's SEMANTIC identity: the sha256 of the SAME
    rendered user text ``reasoning_qc_vlm._build_qc_user_text`` would POST, the
    model id, the ``QC_PROMPT_VERSION`` prompt-contract int, the
    thinking-disabled bool, the EFFECTIVE sampling params (temperature / top_p /
    max_tokens / reasoning_budget — sampling changes the verdict, so a cache HIT
    across different sampling would be wrong), and the unit ``kind``
    (``"window"``/``"seam"``).
    Any change to the block texts, model, prompt, thinking mode, sampling, or
    unit kind moves the key, so a stale sidecar is never served for a changed
    input."""
    from .reasoning_qc_vlm import (
        QC_PROMPT_VERSION,
        _build_qc_user_text,
        resolve_reasoning_qc_disable_thinking,
        resolve_reasoning_qc_sampling,
    )

    user_text = _build_qc_user_text(blocks)
    text_sha = hashlib.sha256(user_text.encode("utf-8")).hexdigest()
    thinking_off = int(bool(resolve_reasoning_qc_disable_thinking()))
    s = resolve_reasoning_qc_sampling()
    sampling_tag = (
        f"{s['temperature']}|{s['top_p']}|{s['max_tokens']}|{s.get('reasoning_budget')}"
    )
    raw = f"{text_sha}|{model}|{QC_PROMPT_VERSION}|{thinking_off}|{sampling_tag}|{kind}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _qc_cache_path(key: str, root: Path) -> Path:
    return root / key[:2] / f"{key}.json"


def _qc_cache_get(path: Path) -> dict | None:
    """Read a cached verdict (``None`` on miss / corrupt / IO error — fail-soft)."""
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else None
    except Exception as exc:  # noqa: BLE001 — a corrupt/unreadable sidecar → miss
        logger.debug("reasoning-QC cache read failed (non-fatal) %s: %s", path, exc)
        return None


def _qc_cache_put(path: Path, verdict: dict[str, Any]) -> None:
    """Persist a verdict atomically (temp+rename). Best-effort — never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(verdict, ensure_ascii=False))
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 — a cache write must never break QC
        logger.debug("reasoning-QC cache write failed (non-fatal) %s: %s", path, exc)


def _qc_verdict_cacheable(verdict: dict[str, Any]) -> bool:
    """Only a genuine judgment is persisted.

    An empty ``{}`` verdict is the fail-soft sentinel (transport/parse miss) — it
    must re-run next time, never be cached as "nothing flagged". A verdict that
    carries ``_qc_incomplete`` is a split-ladder EXHAUSTION (the unit could not
    be judged thinking-on) — also a failure to re-run. Everything else (including
    a clean split-ladder RECOVERY, which carries real findings and no
    ``_qc_incomplete``) is cacheable."""
    return bool(verdict) and "_qc_incomplete" not in verdict


# ---------------------------------------------------------------------------
# Terminal-failure LEDGER (run-scoped negative-result sidecar).
#
# A split-ladder EXHAUSTION (a verdict carrying ``_qc_incomplete``) is a TERMINAL
# failure for that unit THIS run. Historically such verdicts were NOT cached
# (``_qc_verdict_cacheable`` → False), so on every scheduling round that re-drives
# the pass — a killed run's ``--resume``, an offline-retry re-entry — each of the
# (often hundreds of) terminally-failed units RE-POSTs its full, expensive split
# ladder against the SAME still-broken endpoint. Against a persistently-null seat
# that is an unbounded re-attempt of the same window family across rounds (the
# 4h/170-qc_incomplete pathology).
#
# The ledger persists the terminal verdict as a NEGATIVE record tagged with the
# RUN SCOPE + a timestamp, and honours it ONLY within the same run scope and TTL.
# So a ``--resume`` of the SAME run (same ``ED4ALL_RUN_ID``) skips the re-POST
# (fast, bounded), while a genuinely NEW run — new run id, presumably a healthy
# endpoint — recomputes the same fingerprint, sees a cross-scope negative, treats
# it as a MISS, and RETRIES. Never a permanent "this unit is unjudgeable" record.
# ---------------------------------------------------------------------------
_QC_NEGATIVE_KEY = "_qc_terminal_negative"
# TTL belt so an ancient run-scoped negative (e.g. a same-run-id resume days later
# after the operator repaired the seat) is not honoured forever.
_QC_NEGATIVE_TTL_SECONDS = 7 * 24 * 3600.0
# Per-PROCESS run token — the fallback run scope when no orchestrator run id is in
# the environment. A fresh process (a real ``--resume`` without a stable run id)
# gets a new token, so its cross-process negatives are NOT honoured → it re-tries
# (errs toward re-attempt with a possibly-healthy endpoint, never a silent skip).
_PROCESS_RUN_TOKEN = uuid.uuid4().hex


def _qc_run_scope() -> str:
    """Resolve the current run scope for the terminal-failure ledger.

    Prefers a stable orchestrator run id (``ED4ALL_RUN_ID`` — reused verbatim by
    ``ed4all run --resume``, so a resumed run honours its own negatives) then a
    SemantiK-native override, falling back to a per-process token. Read at CALL
    time so tests / a resumed subprocess pick up the ambient run id."""
    for env in ("ED4ALL_RUN_ID", "SEMANTIK_RUN_ID"):
        val = (os.environ.get(env) or "").strip()
        if val:
            return val
    return _PROCESS_RUN_TOKEN


def _qc_is_terminal_incomplete(verdict: Any) -> bool:
    """True iff ``verdict`` is a split-ladder EXHAUSTION (carries ``_qc_incomplete``)."""
    return isinstance(verdict, dict) and bool(verdict.get("_qc_incomplete"))


def _qc_cache_put_negative(path: Path, verdict: dict[str, Any]) -> None:
    """Persist a run-scoped NEGATIVE (terminal-failure) record. Best-effort.

    The record wraps the terminal verdict alongside the run scope + timestamp so
    :func:`_qc_negative_honored` can gate re-use to the same run within the TTL.
    The wrapped ``verdict`` is returned verbatim on an honoured hit, so a partial
    verdict's real findings are preserved."""
    record = {
        _QC_NEGATIVE_KEY: {"run_scope": _qc_run_scope(), "ts": time.time()},
        "verdict": verdict,
    }
    _qc_cache_put(path, record)


def _qc_negative_honored(neg: Any) -> bool:
    """Honour a negative record only within the SAME run scope AND the TTL.

    A cross-scope (new run) or stale (past-TTL) negative returns False → the
    caller treats the sidecar as a MISS and re-attempts the unit, so a later run
    with a healthy endpoint is never permanently starved by a stale failure."""
    if not isinstance(neg, dict):
        return False
    if neg.get("run_scope") != _qc_run_scope():
        return False
    try:
        age = time.time() - float(neg.get("ts", 0.0))
    except (TypeError, ValueError):
        return False
    return 0.0 <= age <= _QC_NEGATIVE_TTL_SECONDS


def _qc_cache_lookup(path: Path) -> dict | None:
    """Read the sidecar and resolve positive / negative / miss semantics.

    Returns the cached verdict on a HONOURED hit (a positive finding record, or a
    run-scoped-and-fresh negative terminal record — unwrapped to its verdict), and
    ``None`` on a miss OR a cross-run / stale negative (re-attempt). Fail-soft."""
    cached = _qc_cache_get(path)
    if cached is None:
        return None
    neg = cached.get(_QC_NEGATIVE_KEY) if isinstance(cached, dict) else None
    if neg is not None:
        if _qc_negative_honored(neg):
            v = cached.get("verdict")
            return v if isinstance(v, dict) else {}
        return None  # cross-run / stale terminal failure → MISS, re-attempt.
    return cached  # positive finding hit.


def _qc_stop_requested() -> bool:
    """Non-raising probe of the handed-in stop sentinel (fail-soft → False).

    The complement of :func:`_check_stop` (which RAISES): used inside the fan-out
    to decide whether to stop SUBMITTING new units without aborting the in-flight
    ones. No wiring (``SEMANTIK_STOP_SENTINEL`` unset) → always False (byte-
    identical)."""
    try:
        from .stop_seam import stop_sentinel_present

        return bool(stop_sentinel_present())
    except Exception:  # noqa: BLE001 — a probe error is never a stop
        return False


# ---------------------------------------------------------------------------
# Per-UNIT fan-out (bounded ThreadPoolExecutor — mirrors
# endpoint_runtime.generate_batch). ONLY the reasoning judgment calls
# parallelize; the caller processes/applies the collected verdicts in ORIGINAL
# window order.
# ---------------------------------------------------------------------------
def _fan_out_page_verifies(
    seat: Any,
    pdf_path: Any,
    windows: Sequence[QCWindow],
    *,
    log: Callable[[str], None],
    stop_site_id: str = "cascade:reasoning-qc-unit",
    scope: str = "full",
    flagged_pages: frozenset[int] | set[int] | None = None,
    sample_pct: int = 0,
    doc_sha: str = "",
    scope_audit_sink: dict[str, Any] | None = None,
) -> dict[int, tuple[dict[str, Any], list[Any], int]]:
    """Fan the per-UNIT QC TEXT judgment calls out concurrently, then stitch.

    The document sequence (:func:`build_qc_windows` returns one) is PLANNED into
    partition sub-windows + junction seam strips (:func:`_plan_page_units`); every
    sub-window AND seam is an independent judgment unit that rides the SAME bounded
    thread pool. Returns ``{window_index: (verdict, flagged, divergence)}`` — the
    units are STITCHED (window findings then seam findings) into a single verdict
    in ORIGINAL window order, so completion order can never scramble downstream
    ordering. Width is :func:`resolve_reasoning_qc_concurrency`
    (``SEMANTIK_REASONING_QC_CONCURRENCY``, default 8), floored at 1 and capped at
    the number of units.

    **Resume cache (``SEMANTIK_REASONING_QC_CHECKPOINT``, default ON).** Each unit
    is a content-addressed sidecar: a cache HIT returns the persisted verdict with
    NO POST; a genuine judgment is persisted (:func:`_qc_verdict_cacheable`). Off
    → no reads, no writes (byte-identical).

    **Stop-on-command.** Units are NOT all submitted up front — the pool is kept
    to ``max_workers`` in flight (submit-one-consume-one), and the stop sentinel is
    probed (non-raising) before EACH new submission. On a stop request the fan-out
    stops SUBMITTING, lets the in-flight units finish (each persists to its
    sidecar), then RAISES via :func:`_check_stop` so the runner's pause path
    engages — worst-case loss is the in-flight units, not the whole stage.

    Fail-soft per UNIT: a worker that RAISES degrades to the empty verdict ``{}``
    for THAT unit (the other units still contribute); it never propagates into
    the cascade (EXCEPT a stop, which propagates by design). A document that fits
    one window emits exactly one unit → byte-identical to the single-call path.
    """
    unit_specs: list[tuple[int, str, int, int]] = []
    plans: dict[int, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {}
    targeted = scope == "targeted"
    _flagged = frozenset(flagged_pages or ())
    skipped_entries: list[dict[str, Any]] = []
    kept_windows = total_windows = total_seams = 0
    for w_idx, w in enumerate(windows):
        subwins, seams = _plan_page_units(len(w.block_records))
        plans[w_idx] = (subwins, seams)
        total_seams += len(seams)
        for (s, e) in subwins:
            total_windows += 1
            if targeted:
                keep, reason, pages = _targeted_keep_window(
                    w, s, e, flagged_pages=_flagged, sample_pct=sample_pct, doc_sha=doc_sha
                )
                if not keep:
                    # Skipped windows are simply NOT judged — the stitch treats a
                    # missing unit as an empty verdict (fail-soft), so no findings
                    # are fabricated. The audit records the honest non-coverage.
                    skipped_entries.append(
                        {"window": w_idx, "range": [s, e], "pages": pages, "skipped": reason}
                    )
                    continue
            kept_windows += 1
            unit_specs.append((w_idx, "window", s, e))
        # ALL seams are ALWAYS judged (cross-boundary order is QC's unique value).
        for (s, e) in seams:
            unit_specs.append((w_idx, "seam", s, e))

    if scope_audit_sink is not None:
        scope_audit_sink.update(
            {
                "scope": scope,
                "sample_pct": sample_pct,
                "total_windows": total_windows,
                "kept_windows": kept_windows,
                "skipped_windows": len(skipped_entries),
                "seams": total_seams,
                "flagged_pages": sorted(_flagged),
                "skipped": skipped_entries,
            }
        )
    if targeted and skipped_entries:
        # Loud (info-level) — the QC report must never silently under-cover.
        log(
            f"[cascade] reasoning-QC (targeted scope): kept {kept_windows}/{total_windows} "
            f"window(s) + ALL {total_seams} seam(s); SKIPPED {len(skipped_entries)} "
            f"window(s) [targeted_scope] (sample_pct={sample_pct}, "
            f"flagged_pages={sorted(_flagged)[:12]})"
        )

    results: dict[int, tuple[dict[str, Any], list[Any], int]] = {}
    if not unit_specs:
        # Everything skipped (targeted, no flag/sample hit, no seams) — still stitch
        # so each document window returns its empty verdict.
        for w_idx, w in enumerate(windows):
            subwins, seams = plans[w_idx]
            results[w_idx] = _stitch_and_flag({}, w_idx, w, subwins, seams)
        return results

    max_workers = max(1, min(resolve_reasoning_qc_concurrency(), len(unit_specs)))
    use_cache = resolve_reasoning_qc_checkpoint()
    unit_model = getattr(seat, "model", None) or resolve_reasoning_qc_model()

    def _work(spec: tuple[int, str, int, int]) -> dict[str, Any]:
        w_idx, kind, s, e = spec
        window = windows[w_idx]
        blocks = list(window.block_records[s:e])
        cache_path: Path | None = None
        if use_cache:
            try:
                key = _qc_unit_fingerprint(blocks, model=str(unit_model), kind=kind)
                root = _qc_cache_root()
                cache_path = _qc_cache_path(key, root)
                # A run-scoped-fresh terminal-failure negative counts as a HIT
                # (skip the re-POST); a cross-run / stale negative is a MISS.
                cached = _qc_cache_lookup(cache_path)
                if cached is not None:
                    return cached
            except Exception as exc:  # noqa: BLE001 — cache infra never breaks QC
                logger.debug("reasoning-QC cache lookup failed (non-fatal): %s", exc)
                cache_path = None
        # Log/judge under the unit's OWN representative page (its first block's
        # page), NOT the document window's page (always page 1) — so a failing
        # sub-slice's split-ladder log names the real page and the advancing
        # iterator is visible in the log stream (see build_qc_windows / _work).
        unit_page = next(
            (b.get("page") for b in blocks if b.get("page") is not None),
            window.page,
        )
        try:
            verdict = _run_qc_judgment(seat, pdf_path, unit_page, blocks)
        except Exception as exc:  # noqa: BLE001 — per-unit fail-soft
            log(
                f"[cascade] reasoning-QC unit verify raised fail-soft "
                f"({kind} [{s}:{e}] page {unit_page}): {exc}"
            )
            return {}
        if use_cache and cache_path is not None:
            if _qc_verdict_cacheable(verdict):
                _qc_cache_put(cache_path, verdict)
            elif _qc_is_terminal_incomplete(verdict):
                # Durable, run-scoped negative so a resume does not re-POST this
                # terminal failure (a new run re-tries — see _qc_negative_honored).
                _qc_cache_put_negative(cache_path, verdict)
        return verdict

    from concurrent.futures import (  # noqa: WPS433
        FIRST_COMPLETED,
        ThreadPoolExecutor,
        wait,
    )

    raw_by_unit: dict[tuple, dict[str, Any]] = {}
    stop_requested = False
    next_idx = 0
    n_specs = len(unit_specs)

    def _collect(future, spec) -> None:
        try:
            raw_by_unit[spec] = future.result()
        except BaseException as exc:  # noqa: BLE001 — belt: never propagate
            log(f"[cascade] reasoning-QC unit future raised fail-soft ({spec}): {exc}")
            raw_by_unit[spec] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        in_flight: dict[Any, tuple[int, str, int, int]] = {}
        # Prime up to max_workers — probing the stop sentinel before each submit.
        while next_idx < n_specs and len(in_flight) < max_workers:
            if _qc_stop_requested():
                stop_requested = True
                break
            spec = unit_specs[next_idx]
            next_idx += 1
            in_flight[pool.submit(_work, spec)] = spec
        # Submit-one-consume-one: as each unit finishes, top the pool back up
        # (bounded) unless a stop has been requested.
        while in_flight:
            done, _pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                spec = in_flight.pop(future)
                _collect(future, spec)
            while not stop_requested and next_idx < n_specs and len(in_flight) < max_workers:
                if _qc_stop_requested():
                    stop_requested = True
                    break
                spec = unit_specs[next_idx]
                next_idx += 1
                in_flight[pool.submit(_work, spec)] = spec

    # A stop was observed: every completed unit is persisted; propagate the stop
    # exactly as the pass-boundary probe would (the runner maps it to a pause).
    if stop_requested:
        log(
            "[cascade] reasoning-QC: stop sentinel observed mid-fan-out — "
            "in-flight units drained + checkpointed; pausing."
        )
        _check_stop(stop_site_id)

    # Stitch the document's units (window order then seam order) into its verdict.
    for w_idx, w in enumerate(windows):
        subwins, seams = plans[w_idx]
        results[w_idx] = _stitch_and_flag(raw_by_unit, w_idx, w, subwins, seams)
    return results


# ---------------------------------------------------------------------------
# Orchestrator entrypoint.
# ---------------------------------------------------------------------------
def run_reasoning_qc(
    capped: list[Any],
    feature_blocks: Sequence[Any],
    assembled: Any,
    region_order: Sequence[int] | None = None,
    *,
    pdf_path: Any,
    review_runtime: Any = None,
    review_verdicts: Any = None,
    arranger_audit: dict[str, Any] | None = None,
    run_inner: Callable[..., Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Any], Any, dict[str, Any]]:
    """Run the Stage-9b reasoning-QC pass over the converged ``capped`` list.

    Parameters
    ----------
    capped:
        The converged Stage-9 region list (``capped`` index space).
    feature_blocks:
        The Stage-2 FeatureBlock list (pages + raw text source).
    assembled:
        The fast-lane assembled doc; ``assembled.region_provenance`` supplies
        the emission order when ``region_order`` is not passed explicitly.
    region_order:
        Optional emission-order capped-index list (defaults to
        ``assembled.region_provenance``).
    pdf_path:
        The source PDF path — the QC pass re-renders ONLY the specific pages
        each window covers (no persistence, no extract-time coupling).
    review_runtime / review_verdicts / run_inner:
        Threaded to the STEP-2 apply channels + the re-assemble adopt gate.
    arranger_audit:
        Optional page-arranger audit dict (``result['page_arranger']`` shape). Read
        ONLY under ``SEMANTIK_REASONING_QC_SCOPE=targeted`` to flag pages the
        arranger struggled on (see :func:`derive_flagged_pages`); ignored in the
        default ``full`` scope.
    log:
        Diagnostic sink (defaults to the module logger).

    Returns
    -------
    ``(new_capped, new_review_verdicts, qc_audit)``. In ``shadow`` mode
    ``new_capped is capped`` (byte-identical); in ``off`` mode this function is
    never called. ``qc_audit`` is the ``result['reasoning_qc']`` arm.
    """
    global _document_capture

    _log = log or (lambda msg: logger.info(msg))
    mode = resolve_reasoning_qc_mode()
    if mode == "off":
        # Defensive: the caller gates on mode != 'off' before importing us, so
        # this only fires on a direct call. Byte-identical no-op.
        return capped, review_verdicts, {"schema": QC_AUDIT_SCHEMA, "ran": False, "mode": "off"}

    if region_order is None:
        region_order = list(getattr(assembled, "region_provenance", None) or [])

    model = resolve_reasoning_qc_model()
    windows = build_qc_windows(capped, feature_blocks, region_order)
    toc = harvest_declared_toc_spine(capped, feature_blocks)

    # SCOPE (owner-delegated): 'full' (default, every window) vs 'targeted' (seams
    # + upstream-flagged pages + a deterministic doc-sha sample). Resolved once.
    scope = resolve_reasoning_qc_scope()
    sample_pct = resolve_reasoning_qc_sample_pct()
    flagged_pages: set[int] = set()
    doc_sha = ""
    scope_audit_sink: dict[str, Any] = {}
    if scope == "targeted" and windows:
        flagged_pages = derive_flagged_pages(
            capped,
            feature_blocks,
            arranger_audit=arranger_audit,
            review_verdicts=review_verdicts,
        )
        doc_sha = _qc_document_sha(windows[0])

    # Declared-but-absent ordinals — warn-only (never invented).
    missing_declared = [
        o for o in toc["declared_ordinals"] if o not in set(toc["heading_ordinals"])
    ]

    qc_audit: dict[str, Any] = {
        "schema": QC_AUDIT_SCHEMA,
        "mode": mode,
        "ran": True,
        "model": model,
        "scope": scope,
        "n_windows": len(windows),
        "toc": {
            "declared_ordinals": toc["declared_ordinals"],
            "heading_ordinals": toc["heading_ordinals"],
            "declared_missing": missing_declared,
        },
        "windows": [],
        "flagged": [],
        "retype": {"applied": 0, "proposed": 0},
        "merge_move": {"applied": 0, "proposed": 0},
        "toc_reconcile": {"proposed": 0},
        "capture_fired": 0,
    }
    if missing_declared:
        _log(
            f"[cascade] reasoning-QC ToC: {len(missing_declared)} declared ordinal(s) "
            f"with no heading region (advisory, not invented): {missing_declared[:8]}"
        )

    all_flagged: list[Any] = []

    # CHECK (A) — ToC reconcile (deterministic, cascade-local; degrades to [] on
    # an empty declared spine). Runs once over the whole region list, BEFORE the
    # per-window VLM pass, and seeds the re-type channel's flag set.
    toc_flags = toc_reconcile(capped, feature_blocks, toc, log=_log)
    qc_audit["toc_reconcile"] = {"proposed": len(toc_flags)}
    for f in toc_flags:
        all_flagged.append(f)
        qc_audit["flagged"].append(
            {
                "region_index": getattr(f, "region_index", None),
                "failure_mode": getattr(f, "failure_mode", None),
                "fixable": bool(getattr(f, "fixable", False)),
                "proposed_regroup_run": list(getattr(f, "proposed_regroup_run", ()) or ()),
                "source": "toc",
                "applied": False,
            }
        )

    # Resolve the seat once (fail-loud on a misconfigured hosted seat). A seat
    # error is NOT swallowed here — it is the operator's "you asked for QC but
    # the seat is unusable" signal (the per-window VLM call is separately
    # fail-soft on transient/parse errors).
    seat = _resolve_qc_seat()

    try:
        _document_capture = _build_reasoning_qc_capture()
    except Exception as exc:  # noqa: BLE001 — a raising builder is never fatal
        logger.debug("reasoning-QC capture build raised (non-fatal): %s", exc)
        _document_capture = None
    try:
        # Stop-on-command: probe once at the pass boundary (cheap early-out).
        # The fan-out ALSO probes the sentinel before each unit submission and
        # persists every completed unit to its resume sidecar, so worst-case loss
        # on a stop mid-fan-out is only the in-flight UNIT judgments (≤ CONCURRENCY
        # window sub-slices + junction seams) — a resume re-fans-out and the
        # already-judged units are cache HITS (no re-POST).
        _check_stop("cascade:reasoning-qc-unit")
        # CHECK (B) — page reading-order verify (VLM, thinking-on) fanned out
        # concurrently per UNIT (SEMANTIK_REASONING_QC_CONCURRENCY, default 8).
        # ONLY the VLM calls parallelize; a page's window + seam units are
        # STITCHED and processed in ORIGINAL window order below so downstream
        # ordering stays deterministic.
        verdicts_by_index = _fan_out_page_verifies(
            seat,
            pdf_path,
            windows,
            log=_log,
            scope=scope,
            flagged_pages=flagged_pages,
            sample_pct=sample_pct,
            doc_sha=doc_sha,
            scope_audit_sink=(scope_audit_sink if scope == "targeted" else None),
        )
        if scope == "targeted":
            qc_audit["scope_plan"] = scope_audit_sink
            _log(
                f"[cascade] reasoning-QC scope=targeted: "
                f"{scope_audit_sink.get('kept_windows', 0)}/"
                f"{scope_audit_sink.get('total_windows', 0)} window(s) judged, "
                f"{scope_audit_sink.get('seams', 0)} seam(s), "
                f"{scope_audit_sink.get('skipped_windows', 0)} skipped "
                f"[targeted_scope] (sample_pct={sample_pct})"
            )

        for w_idx, window in enumerate(windows):
            # Fail-soft: an empty verdict returns ([], 0) and flags nothing (the
            # per-unit fail-soft contract is preserved inside the fan-out).
            verdict, flagged, divergence = verdicts_by_index[w_idx]
            _log_qc_decision(
                _document_capture,
                window=window,
                model=model,
                verdict=verdict,
                flagged=flagged,
                divergence=divergence,
            )
            if _document_capture is not None:
                qc_audit["capture_fired"] += 1
            all_flagged.extend(flagged)
            _pages = [b.get("page") for b in window.block_records if b.get("page") is not None]
            win_entry: dict[str, Any] = {
                "page": window.page,
                "pages": [min(_pages), max(_pages)] if _pages else None,
                "n_regions": len(window.region_indices),
                "confidence": verdict.get("confidence"),
                "order_divergence": divergence,
                "n_flagged": len(flagged),
            }
            # Honest record of any junction/window the split ladder left
            # unverified — mapped back to the capped region indices.
            incomplete_positions = _qc_int_list(verdict.get("_qc_incomplete"))
            incomplete_regions = [
                window.region_indices[p]
                for p in incomplete_positions
                if 0 <= p < len(window.region_indices)
            ]
            if incomplete_regions:
                win_entry["qc_incomplete"] = incomplete_regions
            qc_audit["windows"].append(win_entry)
            for f in flagged:
                qc_audit["flagged"].append(
                    {
                        "region_index": getattr(f, "region_index", None),
                        "failure_mode": getattr(f, "failure_mode", None),
                        "fixable": bool(getattr(f, "fixable", False)),
                        "proposed_regroup_run": list(getattr(f, "proposed_regroup_run", ()) or ()),
                        "source": "vlm",
                        "applied": False,  # advisory in shadow; may flip in ON below
                    }
                )
    finally:
        # Best-effort GPU hand-back (the QC VLM must not squat the card).
        _unload_seat(seat)

    # SHADOW: verify + audit + capture already done; apply NOTHING → the
    # returned capped is the INPUT object list, byte-identical.
    if mode == "shadow":
        _log(
            f"[cascade] reasoning-QC (shadow): {len(all_flagged)} flag(s) over "
            f"{len(windows)} window(s), applied=0"
        )
        return capped, review_verdicts, qc_audit

    # ON: apply the reconcile ops through the EXISTING block-ID op layer under a
    # never-ship-worse snapshot+adopt gate. The re-type channel is
    # partition-immutable (verbatim source rides assembly); the merge/move
    # channel is doubly-gated on SEMANTIK_MOVE_OP=='live'. Every applied op is
    # already conservation-guarded with a whole-revert on raise, so a failed op
    # can only leave ``capped`` byte-identical to the input.
    snapshot_capped = list(capped)
    new_capped, new_verdicts, retype_stats = _apply_retype_ops(
        capped,
        feature_blocks,
        review_verdicts,
        all_flagged,
        review_runtime=review_runtime,
        log=_log,
    )
    new_capped, move_audit_rows, merge_stats = _apply_merge_move_ops(
        new_capped,
        feature_blocks,
        all_flagged,
        log=_log,
    )
    qc_audit["retype"] = {"applied": retype_stats["applied"], "proposed": retype_stats["proposed"]}
    qc_audit["merge_move"] = {"applied": merge_stats["applied"], "proposed": merge_stats["proposed"]}
    qc_audit["applied_ops"] = move_audit_rows

    total_applied = retype_stats["applied"] + merge_stats["applied"]
    if total_applied == 0:
        # Nothing landed → never-ship-worse: return the snapshot verbatim
        # (byte-identical to the input capped list).
        _log(
            f"[cascade] reasoning-QC (on): {len(all_flagged)} flag(s), 0 op(s) applied "
            f"→ ship snapshot (never-ship-worse)"
        )
        return snapshot_capped, review_verdicts, qc_audit

    # Ops landed → re-assemble the reconciled list so the fast-lane assembled doc
    # + region_provenance reflect the reconcile (mirrors _verify_refine_loop's
    # run_inner re-assemble). Best-effort: a re-assemble error is non-fatal — the
    # reconciled region LIST is still returned (the caller owns re-assembly too).
    if run_inner is not None:
        try:
            run_inner("fast", regions=new_capped)
        except Exception as exc:  # noqa: BLE001 — re-assemble is best-effort.
            _log(f"[cascade] reasoning-QC (on): re-assemble raised ({exc}) → return reconciled list")
    qc_audit["applied"] = total_applied
    _log(
        f"[cascade] reasoning-QC (on): {total_applied} op(s) applied "
        f"(retype={retype_stats['applied']}, merge_move={merge_stats['applied']})"
    )
    return new_capped, new_verdicts, qc_audit


def _check_stop(site_id: str) -> None:
    """Best-effort stop-seam probe (no-op if the seam is unavailable)."""
    try:
        from .stop_seam import check_cascade_stop

        check_cascade_stop(site_id)
    except ImportError:
        pass


def _unload_seat(seat) -> None:
    try:
        from .reasoning_qc_vlm import unload_qc_model

        unload_qc_model(seat)
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("reasoning-QC seat unload failed (non-fatal): %s", exc)
