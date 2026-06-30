"""BERT-MergeOrSplit runtime — wraps the trained 3-head adapter.

Phase 3a contract:
    * Inputs:  list of :class:`FeatureBlock` from
               :func:`featurize_with_regions` (or any list of dataclass-
               like objects exposing ``.raw.text``, ``.raw.bbox``,
               ``.raw.font_size``).
    * Outputs: a :class:`BertOutput` carrying three :class:`TypedSignal`s
               per ADJACENT PAIR — head names: ``same_logical_block``,
               ``join_type``, ``hyphen_repair``. Per-pair ``region_id`` =
               index of span_a in the input list (so region_id k
               corresponds to the pair (k, k+1)).

Heading detection lives in BERT-Structure (Phase 3b) as a span-level
``is_heading`` head — the pair-level framing tried here struggled with
extreme class imbalance (~2.6% positives) and high false-positive
rates from the layout-jump signal. The span-level head has 30-50× more
positives.

The deterministic geometry features (``column_id``, ``is_artifact``,
font/bbox prefix) are computed at this layer to match the training-time
``_format_input`` exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths as _semantik_paths
from ..reading_order import column_ids_for_x0s as _column_ids_for_x0s
from .base import LoRAAdapter, LoRAAdapterSpec, load_local_tokenizer
from .registry import register_adapter
from .runner import register_runner
from .structure import batched_cls_pooled
from .types import BertOutput, TypedSignal


# Disk pointer + head shape — kept tight for the registry.
DEFAULT_ADAPTER_DIR = _semantik_paths.resolve_model("council/merge_or_split/final")

SAME_LABELS = ("not_same", "same")
JOIN_LABELS = ("space", "newline_within_p", "list_continuation")
HYPHEN_LABELS = ("no_repair", "repair")

ADAPTER_SPEC = LoRAAdapterSpec(
    bert_name="merge_or_split",
    adapter_path=DEFAULT_ADAPTER_DIR,
    head_kind="multi_head",
    head_specs=(
        ("same_logical_block", len(SAME_LABELS)),
        ("join_type", len(JOIN_LABELS)),
        ("hyphen_repair", len(HYPHEN_LABELS)),
    ),
)


# ---------------------------------------------------------------------------
# Deterministic per-span features (mirrors data/build_merge_or_split_data.py)
# ---------------------------------------------------------------------------


def _ends_with_hyphen(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] in {"-", "‐", "‑", "–", "—"}


def _is_artifact_for_block(fb: Any) -> bool:
    """Replicate the data builder's artifact heuristic: short text in
    top/bottom 5% of page.

    Operates on a FeatureBlock-like object — pulls ``raw.bbox``,
    ``raw.text``, ``raw.page_height`` from it.
    """
    raw = getattr(fb, "raw", None)
    if raw is None:
        return False
    bbox = getattr(raw, "bbox", None) or [0, 0, 0, 0]
    text = (getattr(raw, "text", "") or "").strip()
    page_h = float(getattr(raw, "page_height", 0) or 0)
    if not text or len(text) >= 100 or page_h <= 0 or len(bbox) < 4:
        return False
    return (bbox[1] / page_h) < 0.05 or (bbox[3] / page_h) > 0.95


def _column_ids_for_page(spans: list[Any], page_w: float) -> list[int]:
    """Per-span column index via the shared gutter-clustering core.

    Delegates to :func:`dart_semantic.reading_order.column_ids_for_x0s` (the
    lifted single-linkage ``0.06 * page_width`` gutter clustering) so the
    Fix-A reading-order reorder and this trained-time BERT feature share ONE
    algorithm. Output is byte-identical to the prior inline bin-walk.
    """
    if not spans:
        return []
    x0s = [
        float((getattr(getattr(fb, "raw", None), "bbox", None) or [0, 0, 0, 0])[0])
        for fb in spans
    ]
    return _column_ids_for_x0s(x0s, page_w)


def _bucket_size(fs: float) -> str:
    if fs <= 0:
        return "?"
    if fs < 9: return "xs"
    if fs < 11: return "sm"
    if fs < 13: return "md"
    if fs < 16: return "lg"
    return "xl"


def _bucket_gap(g: float) -> str:
    if g <= 0:
        return "0"
    if g < 6:
        return "sm"
    if g < 18:
        return "md"
    return "lg"


# ---------------------------------------------------------------------------
# Layout side-channel — mirror of data/build_merge_or_split_data.py
# ---------------------------------------------------------------------------
#
# Order MUST stay aligned with LAYOUT_FEATURE_NAMES in the data builder.
# To add a feature: append to the end and increase LAYOUT_FEATURE_DIM in
# both files. Re-train required.

LAYOUT_FEATURE_DIM = 24


def _ends_with_period(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] in ".!?"


def _ends_with_colon(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] == ":"


def _b_starts_upper(text: str) -> bool:
    t = (text or "").lstrip()
    return bool(t) and t[0].isupper()


def _b_titlecase_frac(text: str, *, max_words: int = 6) -> float:
    words = (text or "").split()[:max_words]
    if not words:
        return 0.0
    return sum(1 for w in words if w and w[0].isupper()) / len(words)


_LAYOUT_BBOX_CLAMP = 5000.0
_LAYOUT_FS_CLAMP = 200.0
_LAYOUT_OUTPUT_CLAMP = 10.0


def _safe_coord(v: Any) -> float:
    import math  # noqa: WPS433
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or abs(f) > _LAYOUT_BBOX_CLAMP:
        return 0.0
    return f


def _safe_fs(v: Any) -> float:
    import math  # noqa: WPS433
    try:
        f = float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or f < 0 or f > _LAYOUT_FS_CLAMP:
        return 0.0
    return f


def _clip_layout_vec(vec: list[float]) -> list[float]:
    import math  # noqa: WPS433
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


def _compute_pair_layout(
    fb_a: Any, fb_b: Any,
    *, col_a: int, col_b: int,
    art_a: bool, art_b: bool,
    median_h: float,
) -> list[float]:
    """Per-pair layout vector. Mirrors data.build_merge_or_split_data
    .compute_pair_layout_features(). Order is fixed; output is clamped
    element-wise to +/-10 to defend against pathological PDF coords."""
    import math
    raw_a = getattr(fb_a, "raw", fb_a)
    raw_b = getattr(fb_b, "raw", fb_b)
    bbox_a = list(raw_a.bbox)
    bbox_b = list(raw_b.bbox)
    text_a = (raw_a.text or "").strip()
    text_b = (raw_b.text or "").strip()
    fs_a = _safe_fs(raw_a.font_size)
    fs_b = _safe_fs(raw_b.font_size)
    bold_a = bool(raw_a.is_bold or False)
    bold_b = bool(raw_b.is_bold or False)
    italic_a = bool(getattr(raw_a, "is_italic", False) or False)
    italic_b = bool(getattr(raw_b, "is_italic", False) or False)

    x0_a = _safe_coord(bbox_a[0])
    y0_a = _safe_coord(bbox_a[1])
    x1_a = _safe_coord(bbox_a[2])
    y1_a = _safe_coord(bbox_a[3])
    x0_b = _safe_coord(bbox_b[0])
    y0_b = _safe_coord(bbox_b[1])
    x1_b = _safe_coord(bbox_b[2])
    y1_b = _safe_coord(bbox_b[3])
    h_a = max(0.0, y1_a - y0_a)
    w_a = max(0.0, x1_a - x0_a)
    w_b = max(0.0, x1_b - x0_b)
    y_gap = max(0.0, y0_b - y1_a)
    median_h = max(1.0, _safe_coord(median_h))

    overlap_lo = max(x0_a, x0_b)
    overlap_hi = min(x1_a, x1_b)
    overlap = max(0.0, overlap_hi - overlap_lo)
    union = max(x1_a, x1_b) - min(x0_a, x0_b)
    overlap_frac = (overlap / union) if union > 0 else 0.0

    raw = [
        fs_a / 12.0,
        fs_b / 12.0,
        (fs_b - fs_a) / 12.0,
        (fs_a / max(fs_b, 1.0)) if fs_b > 0 else 1.0,
        1.0 if bold_a else 0.0,
        1.0 if bold_b else 0.0,
        1.0 if bold_a != bold_b else 0.0,
        1.0 if italic_a else 0.0,
        1.0 if italic_b else 0.0,
        math.log1p(y_gap),
        y_gap / max(h_a, 1.0),
        h_a / median_h,
        (x0_b - x0_a) / 100.0,
        (x1_b - x1_a) / 100.0,
        w_b / max(w_a, 1.0),
        overlap_frac,
        1.0 if int(col_a) == int(col_b) else 0.0,
        1.0 if art_a else 0.0,
        1.0 if art_b else 0.0,
        1.0 if _ends_with_hyphen(text_a) else 0.0,
        1.0 if _ends_with_period(text_a) else 0.0,
        1.0 if _ends_with_colon(text_a) else 0.0,
        1.0 if _b_starts_upper(text_b) else 0.0,
        _b_titlecase_frac(text_b),
    ]
    return _clip_layout_vec(raw)


def _format_pair(
    fb_a: Any, fb_b: Any,
    *, cid_a: int, cid_b: int,
    art_a: bool, art_b: bool,
    median_h: float,
) -> str:
    """Format an adjacent-pair input string identical to the trainer."""
    raw_a = getattr(fb_a, "raw", fb_a)
    raw_b = getattr(fb_b, "raw", fb_b)
    bbox_a = list(raw_a.bbox)
    bbox_b = list(raw_b.bbox)
    text_a = (raw_a.text or "").strip()
    text_b = (raw_b.text or "").strip()
    fs_a = float(raw_a.font_size or 0)
    fs_b = float(raw_b.font_size or 0)
    is_bold_a = bool(raw_a.is_bold or False)
    is_bold_b = bool(raw_b.is_bold or False)

    pa = (
        f"[col:{int(cid_a)} art:{int(bool(art_a))} "
        f"fs:{_bucket_size(fs_a)} bold:{int(is_bold_a)} "
        f"hy:{int(_ends_with_hyphen(text_a))}]"
    )
    pb = (
        f"[col:{int(cid_b)} art:{int(bool(art_b))} "
        f"fs:{_bucket_size(fs_b)} bold:{int(is_bold_b)} "
        f"hy:{int(_ends_with_hyphen(text_b))}]"
    )
    y_gap = max(0.0, bbox_b[1] - bbox_a[3])
    height_a = max(1.0, bbox_a[3] - bbox_a[1])
    lhr = height_a / median_h if median_h > 0 else 1.0
    return (
        f"{pa} {text_a} [SEP] [gap:{_bucket_gap(y_gap)} "
        f"lhr:{lhr:.2f}] {pb} {text_b}"
    )


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
    """Load the three classification heads + the layout side-channel
    (LayerNorm + MLP). Returns a dict with keys ``same``, ``join``,
    ``hyphen``, ``layout_norm``, ``layout_mlp``, ``layout_dim``,
    ``layout_mlp_hidden``."""
    import torch.nn as nn  # noqa: WPS433
    import torch  # noqa: WPS433
    state = torch.load(str(heads_path), map_location="cpu")
    layout_dim = int(state.get("layout_dim", LAYOUT_FEATURE_DIM))
    layout_hidden = int(state.get("layout_mlp_hidden", 64))
    head_in = hidden_size + layout_hidden
    head_same = nn.Linear(head_in, len(SAME_LABELS))
    head_join = nn.Linear(head_in, len(JOIN_LABELS))
    head_hyphen = nn.Linear(head_in, len(HYPHEN_LABELS))
    head_same.load_state_dict(state["head_same.state_dict"])
    head_join.load_state_dict(state["head_join.state_dict"])
    head_hyphen.load_state_dict(state["head_hyphen.state_dict"])
    layout_norm = nn.LayerNorm(layout_dim)
    layout_norm.load_state_dict(state["layout_norm.state_dict"])
    layout_mlp = _build_layout_mlp(layout_dim, layout_hidden)
    layout_mlp.load_state_dict(state["layout_mlp.state_dict"])
    return {
        "same": head_same,
        "join": head_join,
        "hyphen": head_hyphen,
        "layout_norm": layout_norm,
        "layout_mlp": layout_mlp,
        "layout_dim": layout_dim,
        "layout_mlp_hidden": layout_hidden,
    }


def _softmax_signal(
    head_name: str,
    region_id: int,
    logits: Any,
    label_names: tuple[str, ...],
    *,
    top_k: int = 3,
    feature_provenance: dict[str, Any] | None = None,
) -> TypedSignal:
    import torch  # noqa: WPS433
    probs = torch.softmax(logits.float(), dim=-1)
    k = min(top_k, len(label_names))
    top_probs, top_idx = probs.topk(k)
    return TypedSignal(
        head_name=head_name,
        region_id=region_id,
        top_k_labels=[label_names[i] for i in top_idx.tolist()],
        top_k_confidences=[float(p) for p in top_probs.tolist()],
        feature_provenance=feature_provenance or {},
    )


# ---------------------------------------------------------------------------
# Runtime entry — adjacent-pair forward
# ---------------------------------------------------------------------------


def _group_by_page(spans: list[Any]) -> list[tuple[int, list[int]]]:
    """Return [(page_num, [span indices in input order])]."""
    by_page: dict[int, list[int]] = {}
    for i, fb in enumerate(spans):
        raw = getattr(fb, "raw", None)
        page = int(getattr(raw, "page", 0) or 0)
        by_page.setdefault(page, []).append(i)
    return sorted(by_page.items())


def run_inputs(adapter: LoRAAdapter, inputs: Any) -> BertOutput:
    """Per-BERT runner registered with ``runner.register_runner``.

    Parameters
    ----------
    adapter
        :class:`LoRAAdapter` already loaded onto the shared backbone.
    inputs
        A list of :class:`FeatureBlock`-like objects (each exposes
        ``.raw.text``, ``.raw.bbox``, ``.raw.page``, ``.raw.font_size``,
        ``.raw.page_width``, ``.raw.page_height``, ``.raw.is_bold``).
    """
    import torch  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    if not isinstance(inputs, (list, tuple)):
        spans = [inputs]
    else:
        spans = list(inputs)

    backbone = adapter.backbone
    spec = adapter.spec
    name = "merge_or_split"

    if len(spans) < 2:
        return BertOutput(
            bert_name=name,
            signals=[],
            backbone_version=f"{backbone.name}@{backbone.revision}",
            adapter_version=f"{name}@{spec.adapter_path}",
        )

    tok_dir = spec.adapter_path / "tokenizer"
    if tok_dir.exists():
        tok = load_local_tokenizer(tok_dir, fallback_name=backbone.name)
    else:
        tok = AutoTokenizer.from_pretrained(backbone.name)

    heads_bundle = _load_heads(
        spec.adapter_path / "heads.pt",
        hidden_size=backbone.hidden_size,
    )
    device = backbone.device or "cpu"
    head_same = heads_bundle["same"].to(device)
    head_join = heads_bundle["join"].to(device)
    head_hyphen = heads_bundle["hyphen"].to(device)
    layout_norm = heads_bundle["layout_norm"].to(device)
    layout_mlp = heads_bundle["layout_mlp"].to(device)

    peft_model = adapter.peft_model
    peft_model.eval()

    # Build adjacent pairs WITHIN each page only.
    pair_texts: list[str] = []
    pair_layouts: list[list[float]] = []
    pair_meta: list[tuple[int, int]] = []   # (region_id, page) — region_id == idx_a
    for page, idxs in _group_by_page(spans):
        if len(idxs) < 2:
            continue
        # Match training-time adjacency exactly: sort spans by (y0, x0)
        # within each page before pair-forming. The data builder applies
        # the same sort (data/build_merge_or_split_data.py:579), so this
        # makes the train/inference adjacency graph identical regardless
        # of the upstream FeatureBlock order.
        def _yx(i: int) -> tuple[float, float]:
            raw = getattr(spans[i], "raw", None)
            bbox = getattr(raw, "bbox", None) or [0.0, 0.0, 0.0, 0.0]
            return (float(bbox[1]), float(bbox[0]))
        idxs = sorted(idxs, key=_yx)
        # Per-page deterministic features.
        page_spans = [spans[i] for i in idxs]
        first_raw = getattr(page_spans[0], "raw", None)
        page_w = float(getattr(first_raw, "page_width", 612.0) or 612.0)
        col_ids = _column_ids_for_page(page_spans, page_w)
        artifact_flags = [_is_artifact_for_block(s) for s in page_spans]
        heights = []
        for fb in page_spans:
            raw = getattr(fb, "raw", None)
            bbox = getattr(raw, "bbox", None) or [0, 0, 0, 0]
            heights.append(max(0.0, bbox[3] - bbox[1]))
        positive_h = [h for h in heights if h > 0]
        median_h = sorted(positive_h)[len(positive_h) // 2] if positive_h else 12.0
        if median_h <= 0:
            median_h = 12.0

        for j in range(len(idxs) - 1):
            i_a = idxs[j]
            i_b = idxs[j + 1]
            txt = _format_pair(
                spans[i_a], spans[i_b],
                cid_a=col_ids[j], cid_b=col_ids[j + 1],
                art_a=artifact_flags[j], art_b=artifact_flags[j + 1],
                median_h=median_h,
            )
            layout_vec = _compute_pair_layout(
                spans[i_a], spans[i_b],
                col_a=col_ids[j], col_b=col_ids[j + 1],
                art_a=artifact_flags[j], art_b=artifact_flags[j + 1],
                median_h=median_h,
            )
            pair_texts.append(txt)
            pair_layouts.append(layout_vec)
            pair_meta.append((i_a, page))

    if not pair_texts:
        return BertOutput(
            bert_name=name,
            signals=[],
            backbone_version=f"{backbone.name}@{backbone.revision}",
            adapter_version=f"{name}@{spec.adapter_path}",
        )

    # Backbone forward in VRAM-bounded batches (SEMANTIK_COUNCIL_BATCH_SIZE);
    # CLS-pooled output is identical to the un-chunked path.
    pooled = batched_cls_pooled(peft_model, tok, pair_texts, device)
    layout_t = torch.tensor(pair_layouts, dtype=torch.float32, device=device)
    with torch.no_grad():
        layout_h = layout_mlp(layout_norm(layout_t))
        h = torch.cat([pooled, layout_h], dim=-1)
        logits_same = head_same(h)
        logits_join = head_join(h)
        logits_hyphen = head_hyphen(h)

    signals: list[TypedSignal] = []
    for k, (region_id, page) in enumerate(pair_meta):
        prov = {"page": int(page), "pair_b_index": region_id + 1}
        signals.append(_softmax_signal(
            "same_logical_block", region_id, logits_same[k],
            SAME_LABELS, feature_provenance=prov,
        ))
        signals.append(_softmax_signal(
            "join_type", region_id, logits_join[k],
            JOIN_LABELS, feature_provenance=prov,
        ))
        signals.append(_softmax_signal(
            "hyphen_repair", region_id, logits_hyphen[k],
            HYPHEN_LABELS, feature_provenance=prov,
        ))

    return BertOutput(
        bert_name=name,
        signals=signals,
        backbone_version=f"{backbone.name}@{backbone.revision}",
        adapter_version=f"{name}@{spec.adapter_path}",
    )


# Self-register on import.
register_adapter(ADAPTER_SPEC)
register_runner("merge_or_split", run_inputs)


__all__ = [
    "ADAPTER_SPEC",
    "DEFAULT_ADAPTER_DIR",
    "HYPHEN_LABELS",
    "JOIN_LABELS",
    "SAME_LABELS",
    "run_inputs",
]
