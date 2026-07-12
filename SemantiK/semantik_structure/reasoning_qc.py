"""Stage-9b reasoning-QC pass (``SEMANTIK_REASONING_QC``) — VLM-adjudicated
reading-order + structure quality control over the converged ``capped`` region
list, applied ONLY as block-ID reconcile ops (never a free-text rewrite).

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

import logging
import os
import re
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

QC_AUDIT_SCHEMA = "reasoning-qc/1.0"


# ---------------------------------------------------------------------------
# Flag resolver — three-valued, default OFF (byte-identical).
# ---------------------------------------------------------------------------
_QC_OFF_TOKENS = frozenset({"0", "false", "no", "off"})


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
# Page-window derivation — one window per physical page, keyed by
# region_provenance pages (re-render ONLY the pages a window covers).
# ---------------------------------------------------------------------------
class QCWindow:
    """One per-page QC unit: the emitted regions on a physical page, in order.

    ``block_texts[k]`` is the raw text of the region at capped index
    ``region_indices[k]`` — so a VLM judgment keyed by block index ``k`` maps
    back to a ``capped`` region index for the reconcile op.
    """

    __slots__ = ("page", "region_indices", "block_texts", "emit_positions")

    def __init__(
        self,
        page: int,
        region_indices: list[int],
        block_texts: list[str],
        emit_positions: list[int],
    ) -> None:
        self.page = page
        self.region_indices = region_indices
        self.block_texts = block_texts
        self.emit_positions = emit_positions


def build_qc_windows(
    capped: Sequence[Any],
    feature_blocks: Sequence[Any],
    region_order: Sequence[int],
) -> list[QCWindow]:
    """Group emitted regions into per-physical-page windows.

    ``region_order`` is the assembler's ``region_provenance`` — capped indices
    in emission (reading) order. Each region is assigned to its PRIMARY (min)
    physical page so the pass re-renders exactly one raster per window and never
    the whole book. Regions with no resolvable page are grouped under page 0
    (their text is still judged for order, just without a raster page number to
    render — the VLM call is skipped for a page-0 window).
    """
    by_page: dict[int, QCWindow] = {}
    for emit_pos, ridx in enumerate(region_order):
        if not (0 <= ridx < len(capped)):
            continue
        region = capped[ridx]
        pages = _region_pages(region, feature_blocks)
        page = pages[0] if pages else 0
        win = by_page.get(page)
        if win is None:
            win = QCWindow(page=page, region_indices=[], block_texts=[], emit_positions=[])
            by_page[page] = win
        win.region_indices.append(ridx)
        win.block_texts.append(_region_raw_text(region, feature_blocks))
        win.emit_positions.append(emit_pos)
    return [by_page[p] for p in sorted(by_page)]


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


def page_order_verify(
    seat: Any,
    pdf_path: Any,
    window: QCWindow,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[Any], int]:
    """CHECK (B): judge one page-window's reading order + structure via the VLM.

    Renders the window's physical page + POSTs (page raster + ordered block
    texts) to the multimodal reasoning client (thinking-on unless
    ``SEMANTIK_VLM_DISABLE_THINKING`` is set) and returns
    ``(verdict, flagged, divergence)``:

      * ``verdict`` — the parsed VLM judgment dict (``{}`` on a fail-soft miss);
      * ``flagged`` — :class:`FlaggedBlock` records (phantom / apparatus re-type
        + explicit-``misordered`` reorder + a reading_order-derived reorder);
      * ``divergence`` — the reading_order position-move count (audit signal).

    Degrades gracefully: a non-renderable page (``window.page <= 0``) is skipped
    with a logged reason and returns ``({}, [], 0)``; a render/transport/parse
    failure inside :func:`_run_qc_judgment` degrades to an empty verdict (the
    VLM client is fail-soft) — this NEVER raises into the cascade.
    """
    _log = log or (lambda msg: logger.debug(msg))
    if window.page <= 0:
        _log(
            f"[cascade] reasoning-QC page_order_verify: window has no renderable "
            f"page (regions={list(window.region_indices)}) → skip"
        )
        return {}, [], 0
    verdict = _run_qc_judgment(seat, pdf_path, window.page, window.block_texts)
    if not verdict:
        # Fail-soft empty verdict (render miss / transport error / parse miss).
        return {}, [], 0
    flagged = judgments_to_flagged_blocks(verdict, window)
    divergence = _window_divergence(verdict, window)
    # Synthesize a reorder candidate from reading_order ONLY when the verdict did
    # not already carry an explicit misordered run (avoid double-flagging).
    if not (verdict.get("misordered") or ()):
        synth = _synthesize_misorder_flag(verdict, window)
        if synth is not None:
            flagged.append(synth)
    return verdict, flagged, divergence


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
    """Emit ONE ``structure_detection`` decision for a QC window.

    Rationale is dynamic + replayable (page number, model id, the window's
    region-index set, divergence, model confidence, per-mode flag counts) so a
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
        rationale = (
            f"reasoning-QC judgment on page {window.page} "
            f"(regions={region_set}, n_blocks={len(region_set)}): "
            f"model={model} confidence={conf} order_divergence={divergence}; "
            f"flagged {len(flagged)} block(s) modes={modes}; "
            f"phantom={len(verdict.get('phantom_headings') or ())} "
            f"apparatus={len(verdict.get('apparatus_retype') or ())} "
            f"misordered={len(verdict.get('misordered') or ())}."
        )
        capture.log_decision(
            decision_type="structure_detection",
            decision=(
                f"reasoning-QC page {window.page}: {len(flagged)} flag(s), "
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


def _run_qc_judgment(seat, pdf_path, page_num, block_texts) -> dict[str, Any]:
    """Render + POST one window's judgment (delegates to the VLM module)."""
    from .reasoning_qc_vlm import run_qc_judgment

    return run_qc_judgment(seat, pdf_path, page_num, block_texts)


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

    # Declared-but-absent ordinals — warn-only (never invented).
    missing_declared = [
        o for o in toc["declared_ordinals"] if o not in set(toc["heading_ordinals"])
    ]

    qc_audit: dict[str, Any] = {
        "schema": QC_AUDIT_SCHEMA,
        "mode": mode,
        "ran": True,
        "model": model,
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
        for window in windows:
            # Stop-on-command: worst-case loss is one window's judgment.
            _check_stop("cascade:reasoning-qc-unit")
            if window.page <= 0:
                # No renderable page — record the window but skip the VLM call.
                qc_audit["windows"].append(
                    {"page": window.page, "regions": list(window.region_indices), "skipped": True}
                )
                continue
            # CHECK (B) — page reading-order verify (VLM, thinking-on). Fail-soft:
            # an empty verdict / render miss returns ([], 0) and flags nothing.
            verdict, flagged, divergence = page_order_verify(
                seat, pdf_path, window, log=_log
            )
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
            qc_audit["windows"].append(
                {
                    "page": window.page,
                    "regions": list(window.region_indices),
                    "confidence": verdict.get("confidence"),
                    "order_divergence": divergence,
                    "n_flagged": len(flagged),
                }
            )
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
