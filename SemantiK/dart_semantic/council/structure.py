"""BERT-Structure runtime — wraps the trained 5-head adapter.

Phase 3b/3f contract:
    * Inputs:  list of :class:`FeatureBlock` from
               :func:`featurize_with_regions` (or any list of dataclass-
               like objects exposing ``.raw.text``, ``.raw.bbox``,
               ``.raw.font_size``, ``.raw.is_bold``).
    * Outputs: a :class:`BertOutput` carrying five :class:`TypedSignal`s
               per SPAN — head names: ``structural_role``,
               ``is_heading``, ``table_region``, ``is_image_block``,
               ``list_nesting``. Per-span ``region_id`` = the span's
               index in the input list. Pre-Phase-3f checkpoints
               (4-head only) suppress the ``is_image_block`` signal at
               emission time so downstream consumers don't see random
               output from an untrained head.

Subsumes today's `dart_semantic/classify.py` DistilBERT classifier
(`models/classifier_v5/final`). The structural_role head is the direct
replacement (7 active classes only); is_heading, table_region, and
list_nesting are new heads the v1 pipeline didn't have.

Two specialist-gating signals mirror the same pattern:
    * is_heading=1   → HeadingSpecialist emits h1..h6.
    * table_region=1 → TableSpecialist parses cell-level role + scope.
The Structure model only DETECTS membership; it never parses tables
or assigns heading levels. structural_role itself is a non-
authoritative recommendation — downstream specialists override per
their domain.

The 20-dim numeric layout side-channel and the 64-dim layout MLP
mirror Phase 3a v4 MergeOrSplit. Order of layout dims MUST match
:data:`data.build_structure_data.LAYOUT_FEATURE_NAMES`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from .. import paths as _semantik_paths
from .base import LoRAAdapter, LoRAAdapterSpec
from .registry import register_adapter
from .runner import register_runner
from .types import BertOutput, TypedSignal


# Default ships the calibrated production head. The env override lets an eval
# (or a head-swap A/B) point the council at a candidate checkpoint WITHOUT
# touching the shipped dir — read at import so it flows into ADAPTER_SPEC below.
DEFAULT_ADAPTER_DIR = (
    Path(os.environ["DART_STRUCTURE_ADAPTER_DIR"])
    if os.environ.get("DART_STRUCTURE_ADAPTER_DIR", "").strip()
    else _semantik_paths.resolve_model("council/structure/final")
)

# 7-class structural_role head — must match
# data.build_structure_data.ROLE_NAMES exactly (same order, same
# active-class subset). Duplicated here so the runtime stays
# self-contained (same pattern as merge_or_split.py).
#
# This head is a NON-AUTHORITATIVE recommendation; downstream council
# logic should override it per-domain. table_region and is_heading are
# the two binary gating signals that route a span to its specialist
# (TableSpecialist / HeadingSpecialist) — the structural_role logits
# only describe span CONTENT shape, not table/heading membership.
ROLE_NAMES = (
    "paragraph",
    "heading",
    "list_item",
    "form_label",
    "blockquote",
    "code_block",
)
IS_HEADING_LABELS = ("not_heading", "heading")
TABLE_REGION_LABELS = ("not_table_region", "table_region")
IS_IMAGE_BLOCK_LABELS = ("not_image_block", "image_block")
LIST_NESTING_LABELS = ("depth_0", "depth_1", "depth_2", "depth_3plus")

ADAPTER_SPEC = LoRAAdapterSpec(
    bert_name="structure",
    adapter_path=DEFAULT_ADAPTER_DIR,
    head_kind="multi_head",
    head_specs=(
        ("structural_role", len(ROLE_NAMES)),
        ("is_heading", len(IS_HEADING_LABELS)),
        ("table_region", len(TABLE_REGION_LABELS)),
        ("is_image_block", len(IS_IMAGE_BLOCK_LABELS)),
        ("list_nesting", len(LIST_NESTING_LABELS)),
    ),
)


# ---------------------------------------------------------------------------
# Layout side-channel — mirror of data/build_structure_data.py
# ---------------------------------------------------------------------------

LAYOUT_FEATURE_DIM = 20

_LAYOUT_BBOX_CLAMP = 5000.0
_LAYOUT_FS_CLAMP = 200.0
_LAYOUT_OUTPUT_CLAMP = 10.0


def _safe_coord(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or abs(f) > _LAYOUT_BBOX_CLAMP:
        return 0.0
    return f


def _safe_fs(v: Any) -> float:
    try:
        f = float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or f < 0 or f > _LAYOUT_FS_CLAMP:
        return 0.0
    return f


def _clip_layout_vec(vec: list[float]) -> list[float]:
    lo, hi = -_LAYOUT_OUTPUT_CLAMP, _LAYOUT_OUTPUT_CLAMP
    out = []
    for x in vec:
        if not math.isfinite(x):
            out.append(0.0)
        elif x < lo:
            out.append(lo)
        elif x > hi:
            out.append(hi)
        else:
            out.append(x)
    return out


def _ends_period(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] in ".!?"


def _ends_colon(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] == ":"


def _starts_upper(text: str) -> bool:
    t = (text or "").lstrip()
    return bool(t) and t[0].isupper()


def _titlecase_frac(text: str, *, max_words: int = 6) -> float:
    words = (text or "").split()[:max_words]
    if not words:
        return 0.0
    return sum(1 for w in words if w and w[0].isupper()) / len(words)


def _caps_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _is_artifact_for_block(fb: Any) -> bool:
    raw = getattr(fb, "raw", None)
    if raw is None:
        return False
    bbox = getattr(raw, "bbox", None) or [0, 0, 0, 0]
    text = (getattr(raw, "text", "") or "").strip()
    page_h = float(getattr(raw, "page_height", 0) or 0)
    if not text or len(text) >= 100 or page_h <= 0 or len(bbox) < 4:
        return False
    return (bbox[1] / page_h) < 0.05 or (bbox[3] / page_h) > 0.95


def _compute_span_layout(
    fb: Any,
    *,
    page_w: float,
    page_h: float,
    page_median_fs: float,
    page_median_h: float,
    in_table: bool,
) -> list[float]:
    """Per-span layout vector. Mirrors
    data.build_structure_data.compute_span_layout_features. Order is
    fixed; output clamped to ±10."""
    raw = getattr(fb, "raw", fb)
    bbox = list(raw.bbox)
    text = (raw.text or "").strip()
    fs = _safe_fs(getattr(raw, "font_size", 0) or 0)

    page_w = max(1.0, _safe_coord(page_w))
    page_h = max(1.0, _safe_coord(page_h))
    page_median_fs = max(1.0, _safe_fs(page_median_fs or 12.0))
    page_median_h = max(1.0, _safe_coord(page_median_h or 12.0))

    x0 = _safe_coord(bbox[0])
    y0 = _safe_coord(bbox[1])
    x1 = _safe_coord(bbox[2])
    y1 = _safe_coord(bbox[3])
    h = max(0.0, y1 - y0)
    w = max(0.0, x1 - x0)

    raw_vec = [
        fs / 12.0,
        fs / page_median_fs,
        1.0 if getattr(raw, "is_bold", False) else 0.0,
        1.0 if getattr(raw, "is_italic", False) else 0.0,
        w / page_w,
        h / 12.0,
        h / page_median_h,
        x0 / page_w,
        x1 / page_w,
        y0 / page_h,
        1.0 if y0 < 0.05 * page_h else 0.0,
        1.0 if y1 > 0.95 * page_h else 0.0,
        1.0 if _is_artifact_for_block(fb) else 0.0,
        1.0 if in_table else 0.0,
        1.0 if _ends_period(text) else 0.0,
        1.0 if _ends_colon(text) else 0.0,
        1.0 if _starts_upper(text) else 0.0,
        _titlecase_frac(text),
        math.log1p(len(text)) / 7.0,
        _caps_ratio(text),
    ]
    return _clip_layout_vec(raw_vec)


# ---------------------------------------------------------------------------
# Head loading
# ---------------------------------------------------------------------------


def _build_layout_mlp(layout_dim: int, layout_hidden: int) -> Any:
    import torch.nn as nn  # noqa: WPS433

    return nn.Sequential(
        nn.Linear(layout_dim, layout_hidden),
        nn.ReLU(),
        nn.Linear(layout_hidden, layout_hidden),
        nn.ReLU(),
    )


def _load_heads(heads_path: Path, hidden_size: int) -> dict[str, Any]:
    """Load the five classification heads + the layout side-channel
    (LayerNorm + MLP). Returns a dict with keys ``role``, ``is_heading``,
    ``table_region``, ``is_image_block``, ``list_nesting``,
    ``layout_norm``, ``layout_mlp``, ``layout_dim``, ``layout_mlp_hidden``.

    Backwards-compat: pre-Phase-3f checkpoints have only 4 heads (no
    ``head_is_image_block.state_dict``). We construct the head module
    anyway and leave it at random init — callers that read
    ``is_image_block`` signals will get junk on those checkpoints, but
    callers that ignore the head (e.g., tests) work unchanged. The
    head_specs in ``ADAPTER_SPEC`` advertises all 5 heads, so any
    consumer that depends on ``is_image_block`` should retrain."""
    import torch.nn as nn  # noqa: WPS433
    import torch  # noqa: WPS433

    state = torch.load(str(heads_path), map_location="cpu")
    layout_dim = int(state.get("layout_dim", LAYOUT_FEATURE_DIM))
    layout_hidden = int(state.get("layout_mlp_hidden", 64))
    head_in = hidden_size + layout_hidden
    head_role = nn.Linear(head_in, len(ROLE_NAMES))
    head_is_heading = nn.Linear(head_in, len(IS_HEADING_LABELS))
    head_table_region = nn.Linear(head_in, len(TABLE_REGION_LABELS))
    head_is_image_block = nn.Linear(head_in, len(IS_IMAGE_BLOCK_LABELS))
    head_list_nesting = nn.Linear(head_in, len(LIST_NESTING_LABELS))
    head_role.load_state_dict(state["head_role.state_dict"])
    head_is_heading.load_state_dict(state["head_is_heading.state_dict"])
    head_table_region.load_state_dict(state["head_table_region.state_dict"])
    if "head_is_image_block.state_dict" in state:
        head_is_image_block.load_state_dict(state["head_is_image_block.state_dict"])
    head_list_nesting.load_state_dict(state["head_list_nesting.state_dict"])
    layout_norm = nn.LayerNorm(layout_dim)
    layout_norm.load_state_dict(state["layout_norm.state_dict"])
    layout_mlp = _build_layout_mlp(layout_dim, layout_hidden)
    layout_mlp.load_state_dict(state["layout_mlp.state_dict"])
    # Phase 3b post-hoc temperature calibration for the is_heading head.
    # Fit on val.jsonl by ``scripts/calibrate_structure_heads.py``. T>1
    # means the head was over-confident and needs softening; T==1.0 (the
    # default) is a no-op for pre-calibration checkpoints. The runtime
    # plumbs this through to ``structure_graph.py`` as a SECOND signal
    # ``is_heading_calibrated`` so the cascade vector going to Semantic
    # (built from the raw ``is_heading`` signal) is unaffected.
    is_heading_temperature = float(state.get("is_heading_temperature", 1.0))
    return {
        "role": head_role,
        "is_heading": head_is_heading,
        "table_region": head_table_region,
        "is_image_block": head_is_image_block,
        "list_nesting": head_list_nesting,
        "layout_norm": layout_norm,
        "layout_mlp": layout_mlp,
        "layout_dim": layout_dim,
        "layout_mlp_hidden": layout_hidden,
        "has_is_image_block_weights": "head_is_image_block.state_dict" in state,
        "is_heading_temperature": is_heading_temperature,
    }


def _softmax_signal(
    head_name: str,
    region_id: int,
    logits: Any,
    label_names: tuple[str, ...],
    *,
    top_k: int | None = 3,
    feature_provenance: dict[str, Any] | None = None,
) -> TypedSignal:
    """Convert per-class logits to a TypedSignal.

    ``top_k`` controls how many class probabilities are serialized:
        * ``int`` — emit top-k labels/confidences (clamped to the
          available class count). Default ``3`` preserves legacy
          behavior.
        * ``None`` — emit ALL classes in descending-confidence order.
          Used by the cascade-bound heads (``structural_role``,
          ``is_heading``, ``table_region``) so the downstream Semantic
          BERT receives the full distribution it was trained against
          (see ``data.build_semantic_data``).
    """
    import torch  # noqa: WPS433

    probs = torch.softmax(logits.float(), dim=-1)
    if top_k is None:
        effective_k = len(label_names)
    else:
        effective_k = min(top_k, len(label_names))
    top_probs, top_idx = probs.topk(effective_k)
    return TypedSignal(
        head_name=head_name,
        region_id=region_id,
        top_k_labels=[label_names[i] for i in top_idx.tolist()],
        top_k_confidences=[float(p) for p in top_probs.tolist()],
        feature_provenance=feature_provenance or {},
    )


# ---------------------------------------------------------------------------
# Per-page helpers (median fs, median h, in_table flag)
# ---------------------------------------------------------------------------


def _group_by_page(spans: list[Any]) -> list[tuple[int, list[int]]]:
    by_page: dict[int, list[int]] = {}
    for i, fb in enumerate(spans):
        raw = getattr(fb, "raw", None)
        page = int(getattr(raw, "page", 0) or 0)
        by_page.setdefault(page, []).append(i)
    return sorted(by_page.items())


def _page_medians(page_spans: list[Any]) -> tuple[float, float]:
    """Return (median_font_size, median_block_height) for one page."""
    fss = []
    hs = []
    for fb in page_spans:
        raw = getattr(fb, "raw", None)
        if raw is None:
            continue
        fs = getattr(raw, "font_size", None)
        if fs is not None and fs > 0:
            fss.append(float(fs))
        bbox = getattr(raw, "bbox", None) or [0, 0, 0, 0]
        if len(bbox) >= 4:
            h = bbox[3] - bbox[1]
            if h > 0:
                hs.append(float(h))
    median_fs = sorted(fss)[len(fss) // 2] if fss else 12.0
    median_h = sorted(hs)[len(hs) // 2] if hs else 12.0
    return median_fs, median_h


def _block_in_table(fb: Any) -> bool:
    """Check the FeatureBlock's `in_table` flag if present (set by
    featurize_with_regions / featurize_from_shared based on pdfplumber
    table bboxes)."""
    return bool(getattr(fb, "in_table", False))


# ---------------------------------------------------------------------------
# Runtime entry — span-level forward
# ---------------------------------------------------------------------------


def run_inputs(
    adapter: LoRAAdapter,
    inputs: Any,
    *,
    top_k_per_head: dict[str, int | None] | None = None,
) -> BertOutput:
    """Per-BERT runner registered with ``runner.register_runner``.

    Parameters
    ----------
    adapter
        :class:`LoRAAdapter` already loaded onto the shared backbone.
    inputs
        A list of :class:`FeatureBlock`-like objects (each exposes
        ``.raw.text``, ``.raw.bbox``, ``.raw.page``, ``.raw.font_size``,
        ``.raw.page_width``, ``.raw.page_height``, ``.raw.is_bold``;
        optionally ``.in_table`` from upstream featurizer).
    top_k_per_head
        Optional override for how many class confidences each head
        serializes. Mapping of head-name → int (top-k) or ``None``
        (full distribution). Heads not present in the mapping default
        to top-3 (legacy behavior). Used by the orchestrator to
        request full distributions for the cascade-bound heads
        (``structural_role``, ``is_heading``, ``table_region``) so the
        Semantic BERT consumes the same shape it was trained on.
    """
    import torch  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    if not isinstance(inputs, (list, tuple)):
        spans = [inputs]
    else:
        spans = list(inputs)

    backbone = adapter.backbone
    spec = adapter.spec
    name = "structure"

    if not spans:
        return BertOutput(
            bert_name=name,
            signals=[],
            backbone_version=f"{backbone.name}@{backbone.revision}",
            adapter_version=f"{name}@{spec.adapter_path}",
        )

    tok_dir = spec.adapter_path / "tokenizer"
    if tok_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
    else:
        tok = AutoTokenizer.from_pretrained(backbone.name)

    heads_bundle = _load_heads(
        spec.adapter_path / "heads.pt",
        hidden_size=backbone.hidden_size,
    )
    device = backbone.device or "cpu"
    head_role = heads_bundle["role"].to(device)
    head_is_heading = heads_bundle["is_heading"].to(device)
    head_table_region = heads_bundle["table_region"].to(device)
    head_is_image_block = heads_bundle["is_image_block"].to(device)
    head_list_nesting = heads_bundle["list_nesting"].to(device)
    layout_norm = heads_bundle["layout_norm"].to(device)
    layout_mlp = heads_bundle["layout_mlp"].to(device)
    has_image_weights = bool(heads_bundle.get("has_is_image_block_weights"))
    # Post-hoc calibration scalar for is_heading. Defaults to 1.0 on
    # pre-calibration checkpoints — same shape, no behavior change.
    is_heading_T = float(heads_bundle.get("is_heading_temperature", 1.0))

    peft_model = adapter.peft_model
    peft_model.eval()

    # Build per-span text + layout vector. Group by page only to compute
    # per-page medians; emit signals in input-order.
    span_texts: list[str] = []
    span_layouts: list[list[float]] = []
    for page, idxs in _group_by_page(spans):
        page_spans = [spans[i] for i in idxs]
        first_raw = getattr(page_spans[0], "raw", None)
        page_w = float(getattr(first_raw, "page_width", 612.0) or 612.0)
        page_h = float(getattr(first_raw, "page_height", 792.0) or 792.0)
        median_fs, median_h = _page_medians(page_spans)
        for i in idxs:
            fb = spans[i]
            raw = getattr(fb, "raw", None)
            text = (getattr(raw, "text", "") or "").strip()
            in_table = _block_in_table(fb)
            layout_vec = _compute_span_layout(
                fb,
                page_w=page_w,
                page_h=page_h,
                page_median_fs=median_fs,
                page_median_h=median_h,
                in_table=in_table,
            )
            # Stash by input index so we emit in input-order regardless
            # of the page-grouping iteration order.
            while len(span_texts) <= i:
                span_texts.append("")
                span_layouts.append([0.0] * LAYOUT_FEATURE_DIM)
            span_texts[i] = text
            span_layouts[i] = layout_vec

    enc = tok(span_texts, padding=True, truncation=True, max_length=192, return_tensors="pt").to(
        device
    )
    layout_t = torch.tensor(span_layouts, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = peft_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        pooled = out.last_hidden_state[:, 0, :].float()
        layout_h = layout_mlp(layout_norm(layout_t))
        h = torch.cat([pooled, layout_h], dim=-1)
        logits_role = head_role(h)
        logits_h = head_is_heading(h)
        logits_tr = head_table_region(h)
        logits_ib = head_is_image_block(h)
        logits_ln = head_list_nesting(h)

    # Resolve per-head top_k. Missing keys default to 3 (legacy).
    tkph: dict[str, int | None] = dict(top_k_per_head or {})

    def _tk(head_name: str) -> int | None:
        return tkph.get(head_name, 3)

    signals: list[TypedSignal] = []
    for i, fb in enumerate(spans):
        raw = getattr(fb, "raw", None)
        prov = {"page": int(getattr(raw, "page", 0) or 0)}
        signals.append(
            _softmax_signal(
                "structural_role",
                i,
                logits_role[i],
                ROLE_NAMES,
                top_k=_tk("structural_role"),
                feature_provenance=prov,
            )
        )
        signals.append(
            _softmax_signal(
                "is_heading",
                i,
                logits_h[i],
                IS_HEADING_LABELS,
                top_k=_tk("is_heading"),
                feature_provenance=prov,
            )
        )
        # Calibrated twin used ONLY by Stage-5 heading-region gating (see
        # dart_semantic/structure_graph.py). The cascade vector going to
        # Semantic is built from the raw ``is_heading`` signal above; this
        # extra signal must NOT be consumed there. ``is_heading_T == 1.0``
        # on pre-calibration checkpoints, in which case this signal is
        # numerically identical to the raw one (still emitted for shape
        # parity so downstream code sees a consistent contract).
        signals.append(
            _softmax_signal(
                "is_heading_calibrated",
                i,
                logits_h[i] / is_heading_T,
                IS_HEADING_LABELS,
                top_k=_tk("is_heading_calibrated"),
                feature_provenance={
                    **prov,
                    "is_heading_temperature": is_heading_T,
                },
            )
        )
        signals.append(
            _softmax_signal(
                "table_region",
                i,
                logits_tr[i],
                TABLE_REGION_LABELS,
                top_k=_tk("table_region"),
                feature_provenance=prov,
            )
        )
        # is_image_block: only emit a signal if the loaded checkpoint
        # actually carries trained weights. Pre-Phase-3f adapters give
        # random predictions on this head; emitting them would mislead
        # downstream gating (e.g., ImageSpecialist routing).
        if has_image_weights:
            signals.append(
                _softmax_signal(
                    "is_image_block",
                    i,
                    logits_ib[i],
                    IS_IMAGE_BLOCK_LABELS,
                    top_k=_tk("is_image_block"),
                    feature_provenance=prov,
                )
            )
        signals.append(
            _softmax_signal(
                "list_nesting",
                i,
                logits_ln[i],
                LIST_NESTING_LABELS,
                top_k=_tk("list_nesting"),
                feature_provenance=prov,
            )
        )

    return BertOutput(
        bert_name=name,
        signals=signals,
        backbone_version=f"{backbone.name}@{backbone.revision}",
        adapter_version=f"{name}@{spec.adapter_path}",
    )


# Self-register on import.
register_adapter(ADAPTER_SPEC)
register_runner("structure", run_inputs)


__all__ = [
    "ADAPTER_SPEC",
    "DEFAULT_ADAPTER_DIR",
    "IS_HEADING_LABELS",
    "IS_IMAGE_BLOCK_LABELS",
    "LAYOUT_FEATURE_DIM",
    "LIST_NESTING_LABELS",
    "ROLE_NAMES",
    "TABLE_REGION_LABELS",
    "run_inputs",
]
