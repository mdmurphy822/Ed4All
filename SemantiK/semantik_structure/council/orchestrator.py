"""Stage 3 council orchestrator — canonical PDF → CouncilState entry point.

Walks the static DAG declared in :mod:`semantik_structure.council.routing`
in topological order, swapping LoRA adapters between BERTs (the runner's
``LoRAAdapter.load()`` releases whichever adapter is currently bound to
the shared backbone, so VRAM usage stays at ~one adapter at a time).

Pipeline stages (order matches ``routing.topological_order()``):

    1. ``extract_shared_cached(pdf_path)`` — Stage 1+2 deterministic
       extract (pikepdf + pypdfium2 + pdfplumber + Tesseract).
    2. ``featurize_with_regions(shared)`` — Stage 2 reading-order
       FeatureBlocks plus typed table/math RegionCandidates.
    3. ``run_bert("merge_or_split", spans)`` — adjacent-pair signals.
    4. ``run_bert("structure", spans)`` — span-level structural-role
       and gating signals (used to derive cascade for Semantic AND to
       feed the cross-BERT reranker).
    5. ``run_bert("semantic", spans)`` — cascade-augmented spans;
       cascade is derived from Structure's per-span head distributions.
    6. ``run_bert("table_specialist", cells)`` — per cell of every
       confirmed (size-passing) TableCandidate. Skipped if
       :func:`~semantik_structure.council.table_cell_builder.build_cells`
       returns ``[]`` for the candidate.
    7. ``run_bert("math_specialist", math_candidates)`` — ungated for
       v1 (the cross-BERT reranker decides whether each math-region is
       actually math; see ``cross_reranker.py``).

Returns ``(CouncilState, regions, feature_blocks)`` where ``regions``
is the ``table_candidates + math_candidates`` list passed to Stage 4
(cross-BERT reranker) for arbitration, and ``feature_blocks`` is the
Stage-2 FeatureBlock stream Stage 5 consumes directly.

Errors are isolated per-BERT: if one BERT throws, the orchestrator logs
to stderr and continues. The arbiter is designed to fall back to prose
for any region that lacks a signal it expected (see
``cross_reranker._table_confirmed`` etc.).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

from ..types import FeatureBlock
from .routing import topological_order
from .structure import IS_HEADING_LABELS, ROLE_NAMES, TABLE_REGION_LABELS
from .types import (
    BertOutput,
    CouncilState,
    ImageCandidate,
    MathCandidate,
    RegionCandidate,
    TableCandidate,
)


# ---------------------------------------------------------------------------
# Cascade derivation — Structure → Semantic
# ---------------------------------------------------------------------------
#
# AXIS-1 role expansion vs. the Semantic head's cascade contract.
#
# ``council.structure.ROLE_NAMES`` was widened 6 → 9 (AXIS-1 appended
# ``definition_term`` / ``definition_def`` / ``caption``) and a 9-class
# structure adapter is deployed. But the Semantic head's cascade contract is
# still the LEGACY 8-dim = [P(role)x6, P(is_heading=1), P(table_region=1)]
# (``data.builders.build_semantic_data.CASCADE_DIM = 6 + 1 + 1``; validated at
# ``council.semantic.CASCADE_DIM``). The Semantic adapter was trained on the
# 6-role cascade, so the vector fed to it MUST stay 6-role until Semantic is
# retrained on the 11-dim shape. We therefore project the 9-role structural
# distribution back onto the 6 legacy slots by NAME before concatenation,
# folding the 3 AXIS-1 roles' probability mass into ``paragraph`` (under the
# 6-role era those spans were classified predominantly as paragraph, so the
# fold reproduces the training-time input distribution for the Semantic head).
# The 3 AXIS-1 roles still carry region-typing value ELSEWHERE in the council
# (structure_graph / reviewer); this fold is scoped to the Semantic cascade
# only and must not widen this vector until the Semantic head is retrained.
_LEGACY_ROLE_NAMES: tuple[str, ...] = ROLE_NAMES[:6]
_EXPECTED_LEGACY_ROLE_NAMES: tuple[str, ...] = (
    "paragraph",
    "heading",
    "list_item",
    "form_label",
    "blockquote",
    "code_block",
)
# Fail loud at import if the legacy 6-role slot order ever drifts — the fold
# below is name-keyed but the Semantic head consumes the vector POSITIONALLY,
# so the first-6 order of ROLE_NAMES is load-bearing.
if _LEGACY_ROLE_NAMES != _EXPECTED_LEGACY_ROLE_NAMES:
    raise RuntimeError(
        "council.structure.ROLE_NAMES[:6] changed order/contents: "
        f"{_LEGACY_ROLE_NAMES!r} != {_EXPECTED_LEGACY_ROLE_NAMES!r}. The "
        "Semantic cascade fold (_derive_cascades) consumes these 6 slots "
        "positionally; re-derive the fold before shipping."
    )
# AXIS-1 roles (present only when len(ROLE_NAMES) == 9) → the legacy slot
# their mass folds into. Under the 6-role era these spans classified
# predominantly as paragraph.
_AXIS1_ROLE_FOLD: dict[str, str] = {
    "definition_term": "paragraph",
    "definition_def": "paragraph",
    "caption": "paragraph",
}
_LEGACY_CASCADE_DIM = len(_LEGACY_ROLE_NAMES) + 2  # 6 role + is_heading + table_region


def _fold_role_to_legacy(role_full: list[float]) -> list[float]:
    """Project a full per-role distribution onto the 6 LEGACY role slots.

    ``role_full`` is a ``len(ROLE_NAMES)``-dim vector in canonical role
    order (from :func:`_signal_to_full`). Returns a 6-dim vector in
    ``_LEGACY_ROLE_NAMES`` order, folding each AXIS-1 role's probability
    mass into the legacy slot named by ``_AXIS1_ROLE_FOLD`` (paragraph).
    When ``len(ROLE_NAMES) == 6`` (an older checkout / a pre-AXIS-1
    adapter) this is a straight passthrough — the loop finds no AXIS-1
    roles and the first 6 entries copy 1:1.
    """
    legacy_idx = {name: i for i, name in enumerate(_LEGACY_ROLE_NAMES)}
    legacy = [0.0] * len(_LEGACY_ROLE_NAMES)
    for role_name, mass in zip(ROLE_NAMES, role_full):
        target = role_name if role_name in legacy_idx else _AXIS1_ROLE_FOLD.get(role_name)
        idx = legacy_idx.get(target) if target is not None else None
        if idx is None:
            # A role that is neither a legacy slot nor a known AXIS-1 fold
            # target — drop its mass loudly rather than silently mis-slot.
            raise ValueError(
                f"structural_role {role_name!r} has no legacy-slot mapping "
                f"(legacy slots={_LEGACY_ROLE_NAMES}, "
                f"AXIS-1 fold={_AXIS1_ROLE_FOLD})"
            )
        legacy[idx] += float(mass)
    return legacy


def _signal_to_full(
    sig: Any,
    label_names: tuple[str, ...],
) -> list[float]:
    """Slot a TypedSignal's (label, confidence) pairs into canonical
    class order, producing a full per-class probability vector.

    Assumes the signal carries the FULL distribution (orchestrator
    requests ``top_k=None`` from Structure for cascade-bound heads).
    Labels not present in the signal stay at 0.0 — callers should only
    pass signals known to be full-distribution serializations. For
    binary heads, top-3 (truncated to 2) is also full and works here.
    Returns all-zeros if ``sig`` is ``None``.
    """
    full = [0.0] * len(label_names)
    if sig is None:
        return full
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    for label, conf in zip(sig.top_k_labels, sig.top_k_confidences):
        idx = label_to_idx.get(label)
        if idx is None:
            continue
        full[idx] = float(conf)
    return full


def _derive_cascades(
    spans: list[Any],
    structure_out: BertOutput | None,
) -> list[list[float]]:
    """Build one 8-dim cascade vector per span:

        [P(role)x6, P(is_heading=1), P(table_region=1)]

    Mirrors ``data.builders.build_semantic_data`` / ``train_semantic`` cascade
    contract. Requires Structure to have been invoked with
    ``top_k_per_head={"structural_role": None, ...}`` so the per-class
    confidences are the full softmax (not a truncated top-k). If
    Structure didn't run (None) or didn't emit a signal for a span,
    that span's cascade is zero-filled — this will trip Semantic's
    loud cascade validator. Caller should skip Semantic entirely when
    Structure failed.

    9→6 role FOLD (AXIS-1). The deployed structure adapter emits a
    9-class ``structural_role`` distribution (AXIS-1 added
    ``definition_term`` / ``definition_def`` / ``caption``), but the
    Semantic head was trained on the LEGACY 6-role cascade and consumes
    this vector positionally at a fixed 8-dim = 6 role + 2 gates. We
    therefore project the full role distribution onto the 6 legacy slots
    by NAME (via :func:`_fold_role_to_legacy`), folding the 3 AXIS-1
    roles' probability mass into ``paragraph`` — reproducing the
    training-time input distribution (those spans were 6-role-era
    paragraphs) rather than widening the vector the Semantic head cannot
    consume. The 3 AXIS-1 roles keep their region-typing value elsewhere
    in the council; this fold must NOT widen the cascade until the
    Semantic head is retrained on the 11-dim shape. On an older /
    pre-AXIS-1 checkout (``len(ROLE_NAMES) == 6``) the fold is a straight
    passthrough (no AXIS-1 roles to fold).
    """
    n = len(spans)
    cascades: list[list[float]] = [[0.0] * _LEGACY_CASCADE_DIM for _ in range(n)]
    if structure_out is None:
        return cascades
    # Index Structure signals by (head_name, region_id) for O(1) lookup.
    by_head: dict[str, dict[int, Any]] = {}
    for sig in structure_out.signals:
        by_head.setdefault(sig.head_name, {})[sig.region_id] = sig
    role_sigs = by_head.get("structural_role", {})
    ih_sigs = by_head.get("is_heading", {})
    tr_sigs = by_head.get("table_region", {})
    for i in range(n):
        # Full role distribution in ROLE_NAMES order (9-dim on the deployed
        # adapter), then fold onto the 6 legacy slots the Semantic head expects.
        role_full = _signal_to_full(role_sigs.get(i), ROLE_NAMES)
        role_legacy = _fold_role_to_legacy(role_full)
        ih_full = _signal_to_full(ih_sigs.get(i), IS_HEADING_LABELS)
        tr_full = _signal_to_full(tr_sigs.get(i), TABLE_REGION_LABELS)
        # IS_HEADING_LABELS = ("not_heading", "heading") → index 1.
        # TABLE_REGION_LABELS = ("not_table_region", "table_region") → 1.
        cascades[i] = list(role_legacy) + [ih_full[1], tr_full[1]]
        if len(cascades[i]) != _LEGACY_CASCADE_DIM:
            # Descriptive failure — the bare ``assert len == 8`` this
            # replaced rendered as "AssertionError: 11" in phase logs and
            # cost a diagnosis round-trip when ROLE_NAMES widened to 9.
            raise ValueError(
                "cascade vector has wrong dim: "
                f"{len(cascades[i])} != {_LEGACY_CASCADE_DIM} "
                f"(ROLE_NAMES has {len(ROLE_NAMES)} roles, folded onto "
                f"{len(_LEGACY_ROLE_NAMES)} legacy slots via {_AXIS1_ROLE_FOLD}; "
                "the Semantic head expects the legacy 6-role cascade — "
                "retrain Semantic on the 11-dim shape before widening this)"
            )
    return cascades


# ---------------------------------------------------------------------------
# Cascade-attached span wrapper
# ---------------------------------------------------------------------------


class _CascadeSpan:
    """Lightweight wrapper that exposes ``.raw``, ``.in_table``, and
    ``.cascade`` for Semantic's runtime — proxies all other attribute
    access to the wrapped FeatureBlock.

    Why a wrapper rather than mutating FeatureBlock? FeatureBlock is a
    shared dataclass instance that the orchestrator does not own; any
    field we add risks colliding with future featurizer fields.
    """

    __slots__ = ("_fb", "cascade")

    def __init__(self, fb: Any, cascade: list[float]) -> None:
        self._fb = fb
        self.cascade = cascade

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fb, name)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[council] {msg}", file=sys.stderr, flush=True)


def _safe_run(
    name: str,
    runner_callable: Any,
    *,
    errors: dict[str, str] | None = None,
) -> BertOutput | None:
    """Wrap a per-BERT call so a single failure doesn't sink the run.

    When the call raises, the exception's repr is recorded into
    ``errors[name]`` (if provided) so the orchestrator can surface the
    failure in ``CouncilState.signal_coverage`` rather than silently
    leaving the head's entry absent.
    """
    try:
        return runner_callable()
    except Exception as exc:  # noqa: BLE001
        _log(f"BERT {name!r} raised: {exc}")
        traceback.print_exc()
        if errors is not None:
            errors[name] = repr(exc)
        return None


def _observed_region_count(out: BertOutput | None) -> int:
    """Count distinct ``region_id``s that have at least one signal.

    A head may emit multiple TypedSignals per region (one per head
    label). For coverage purposes we want the number of input items
    the head actually scored, so we de-dup by ``region_id``.
    """
    if out is None:
        return 0
    return len({sig.region_id for sig in out.signals})


def _coverage_entry(
    *,
    expected: int,
    out: BertOutput | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build one ``signal_coverage`` entry for one head."""
    observed = _observed_region_count(out)
    missing = max(0, expected - observed)
    entry: dict[str, Any] = {
        "expected": int(expected),
        "observed": int(observed),
        "missing": int(missing),
    }
    if error is not None:
        entry["error"] = error
    return entry


def run_council(
    pdf_path: Path | str,
    *,
    max_pages: int | None = None,
) -> tuple[CouncilState, list[RegionCandidate], list[FeatureBlock]]:
    """Run all five council BERTs on one PDF in topological order.

    Returns ``(state, regions, feature_blocks)`` so callers can pair
    per-region predictions with their geometry/metadata for downstream
    arbitration AND access the Stage-2 FeatureBlock stream that Stage 5
    consumes directly. ``regions`` is the concatenation of table
    candidates followed by math candidates.

    ``max_pages`` (if provided) truncates the shared-extract page list
    after extraction — useful for smoke testing on large PDFs without
    re-extracting from scratch.

    Returns
    -------
    (state, regions, feature_blocks)
        ``state`` is the aggregate :class:`CouncilState`. ``regions``
        is the list passed to Stage 4
        (:func:`~semantik_structure.council.cross_reranker.arbitrate`); it
        contains :class:`TableCandidate`s and :class:`MathCandidate`s
        only — there is **no flat-text wrapper**. Per-span Structure
        outputs (the prose-coverage signals) are accessible via
        ``state.outputs['structure'].signals`` (each signal's
        ``region_id`` is the FeatureBlock index); Stage 5 consumes
        those directly via ``feature_blocks`` to build the structure
        graph. ``feature_blocks`` is the Stage-2 reading-order
        FeatureBlock stream — its length equals the number of spans
        every council BERT was invoked on.
    """
    from ..extract_shared import extract_shared_cached
    from ..features import featurize_with_regions
    from .runner import run_bert

    pdf_path = Path(pdf_path)
    _log(f"extract_shared_cached({pdf_path})")
    shared = extract_shared_cached(pdf_path)
    if max_pages is not None and max_pages > 0:
        # Truncate page list so downstream stages only see N pages.
        shared = dict(shared)
        shared["pages"] = list(shared.get("pages", []))[:max_pages]

    feature_set = featurize_with_regions(shared)
    spans: list[Any] = list(feature_set.feature_blocks)
    table_candidates: list[TableCandidate] = list(feature_set.table_candidates)
    math_candidates: list[MathCandidate] = list(feature_set.math_candidates)
    image_candidates: list[ImageCandidate] = list(feature_set.image_candidates)
    # Part F — figures append after tables + math. Empty unless
    # SEMANTIK_DETECT_FIGURES is on (byte-stable ordering when off).
    regions: list[RegionCandidate] = (
        list(table_candidates) + list(math_candidates) + list(image_candidates)
    )

    _log(
        f"feature_blocks={len(spans)}  "
        f"table_candidates={len(table_candidates)}  "
        f"math_candidates={len(math_candidates)}  "
        f"image_candidates={len(image_candidates)}"
    )

    if not spans:
        _log("no feature blocks; returning empty CouncilState")
        return (CouncilState(outputs={}), regions, spans)

    outputs: dict[str, BertOutput] = {}
    structure_out: BertOutput | None = None
    # Per-head observability: how many region_ids each head was
    # expected to score vs. how many it actually emitted signals for.
    # Populated as each head runs (or fails) so a partial-failure run
    # is loudly visible in CouncilState.signal_coverage rather than
    # silently absent from `outputs`. See no-silent-fallbacks invariant.
    signal_coverage: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    for bert_name in topological_order():
        if bert_name == "merge_or_split":
            # merge_or_split scores adjacent pairs; expected = N-1 for N
            # spans (last span has no right neighbor). Falls back to N
            # when only one span is present so the math stays sane.
            # NOTE(benign coverage gap): expected is the GLOBAL N-1 pair
            # count, but the builder intentionally pairs spans within-page
            # only (merge_or_split.py "Build adjacent pairs WITHIN each
            # page only", matching training adjacency in
            # data/builders/build_merge_or_split_data.py). Aggregate "missing" is
            # therefore ~(non-empty pages - 1) per PDF — page boundaries,
            # not dropped signals (R10: 42 missing == 42 page boundaries).
            expected = max(0, len(spans) - 1) if len(spans) > 1 else len(spans)
            out = _safe_run(
                bert_name,
                lambda: run_bert("merge_or_split", spans),
                errors=errors,
            )
            if out is not None:
                outputs[bert_name] = out
            signal_coverage[bert_name] = _coverage_entry(
                expected=expected,
                out=out,
                error=errors.get(bert_name),
            )

        elif bert_name == "structure":
            # Request full per-class distributions for the cascade-bound
            # heads (structural_role, is_heading, table_region) so the
            # Semantic BERT receives the same shape it was trained on
            # (see data.builders.build_semantic_data). Other heads keep the
            # default top-3 to minimize signal-payload size.
            expected = len(spans)
            out = _safe_run(
                bert_name,
                lambda: run_bert(
                    "structure",
                    spans,
                    top_k_per_head={
                        "structural_role": None,
                        "is_heading": None,
                        "table_region": None,
                    },
                ),
                errors=errors,
            )
            if out is not None:
                outputs[bert_name] = out
                structure_out = out
            signal_coverage[bert_name] = _coverage_entry(
                expected=expected,
                out=out,
                error=errors.get(bert_name),
            )

        elif bert_name == "semantic":
            expected = len(spans)
            if structure_out is None:
                _log("skipping semantic (no Structure cascade available)")
                # Mark the skip: expected=N, observed=0, with an
                # explanation so eval can distinguish "Structure failed
                # so we couldn't run Semantic" from a genuine head crash.
                signal_coverage[bert_name] = _coverage_entry(
                    expected=expected,
                    out=None,
                    error="skipped: no Structure cascade available",
                )
                continue
            cascades = _derive_cascades(spans, structure_out)
            cascade_spans = [_CascadeSpan(fb, casc) for fb, casc in zip(spans, cascades)]
            out = _safe_run(
                bert_name,
                lambda: run_bert("semantic", cascade_spans),
                errors=errors,
            )
            if out is not None:
                outputs[bert_name] = out
            signal_coverage[bert_name] = _coverage_entry(
                expected=expected,
                out=out,
                error=errors.get(bert_name),
            )

        elif bert_name == "table_specialist":
            if not table_candidates:
                continue
            from .table_cell_builder import build_cells

            # Aggregate cells across all candidates so we make a single
            # adapter swap. The runner caches per-BERT runners; multiple
            # invocations would each rebuild heads + tokenizer.
            all_cells: list[Any] = []
            for k, tc in enumerate(table_candidates):
                cells = build_cells(tc, table_idx=k)
                all_cells.extend(cells)
            if not all_cells:
                _log("table_specialist: no cells built (size filter)")
                # Cells were filtered by build_cells (size). Record the
                # zero-expected entry so eval can tell "no cells" from
                # "head crashed" — both yield observed=0, but expected=0
                # means there was nothing to do.
                signal_coverage[bert_name] = _coverage_entry(
                    expected=0,
                    out=None,
                )
                continue
            expected = len(all_cells)
            out = _safe_run(
                bert_name,
                lambda: run_bert("table_specialist", all_cells),
                errors=errors,
            )
            if out is not None:
                outputs[bert_name] = out
            signal_coverage[bert_name] = _coverage_entry(
                expected=expected,
                out=out,
                error=errors.get(bert_name),
            )

        elif bert_name == "math_specialist":
            if not math_candidates:
                continue
            expected = len(math_candidates)
            out = _safe_run(
                bert_name,
                lambda: run_bert("math_specialist", math_candidates),
                errors=errors,
            )
            if out is not None:
                outputs[bert_name] = out
            signal_coverage[bert_name] = _coverage_entry(
                expected=expected,
                out=out,
                error=errors.get(bert_name),
            )

        else:
            _log(f"unknown BERT in routing DAG: {bert_name!r}")

    state = CouncilState(
        outputs=outputs,
        document_id=str(pdf_path),
        council_version="council-v1",
        signal_coverage=signal_coverage,
    )
    return (state, regions, spans)


__all__ = [
    "run_council",
]
