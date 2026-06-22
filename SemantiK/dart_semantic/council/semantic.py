"""BERT-Semantic runtime — wraps the trained 2-head adapter.

Phase 3c contract:
    * Inputs:  list of :class:`FeatureBlock`-like objects (each exposes
               ``.raw.text``, ``.raw.bbox``, ``.raw.font_size``,
               ``.raw.is_bold``, ``.raw.is_italic``, ``.raw.page``,
               ``.raw.page_width``, ``.raw.page_height``; optionally
               ``.in_table`` and ``.cascade`` — see below).
    * Outputs: a :class:`BertOutput` carrying two :class:`TypedSignal`s
               per SPAN — head names ``doc_role`` (7-class) and
               ``boilerplate`` (binary). Per-span ``region_id`` = the
               span's index in the input list.

Cascade vectors (teacher-forced from BERT-Structure):
    Each span MUST carry an 8-dim ``cascade`` vector via ``.cascade`` or
    a ``"cascade"`` dict-key:
        [P(role)x6, P(is_heading=1), P(table_region=1)]
    The Phase 3c trainer baked cascade into the JSONL at data-build
    time, and the council orchestrator computes cascade by running
    Structure first and attaching the result to each span. Semantic's
    runtime does NOT invoke Structure inline — keeping a single LoRA
    swap per BERT visit. If a span lacks cascade, the runtime raises
    :class:`ValueError`. (Drop-in zero vectors silently degrade
    accuracy; making it loud is the right call here.)

Positional vector (Semantic-only, 3-dim):
    Computed from input order + per-span page/bbox:
        block_index_norm    i / max(1, len(spans) - 1)
        page_index_norm     (page - first_page) / max(1, total_pages - 1)
        doc_position_norm   ((page - first_page) * page_h + y0) /
                            max(1, total_pages * page_h)
    Mirrors :func:`data.build_semantic_data.compute_positional_features`.
    Positional is appended ONLY to Semantic's side-channel; Structure
    stays on its 20-dim layout contract.

Side-channel layout (31-dim):
    layout(20) || cascade(8) || positional(3) → LayerNorm → MLP(31→64→64)
    → concatenated with ModernBERT pooled (CLS) → 2 heads.

Mirrors the shape declared in :mod:`train_semantic`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .. import paths as _semantik_paths
from .base import LoRAAdapter, LoRAAdapterSpec
from .registry import register_adapter
from .runner import register_runner
from .structure import (
    LAYOUT_FEATURE_DIM,
    _block_in_table,
    _compute_span_layout,
    _group_by_page,
    _page_medians,
)
from .types import BertOutput, TypedSignal


DEFAULT_ADAPTER_DIR = _semantik_paths.resolve_model("council/semantic/final")

# 7-class doc_role head — must match
# data.build_semantic_data.DOC_ROLE_NAMES exactly (same order, same
# active-class subset). Duplicated here so the runtime stays self-
# contained (same pattern as structure.py / merge_or_split.py).
#
# `abstract` was dropped at 0.16% prevalence (Phase 3c task #45);
# `metadata` is the catch-all for DOIs / arxiv ids / pub-date strings.
DOC_ROLE_NAMES = (
    "title",
    "author",
    "body",
    "citation",
    "footer",
    "legal",
    "metadata",
)
BOILERPLATE_LABELS = ("not_boilerplate", "boilerplate")

CASCADE_DIM = 8           # [P(role)x6, P(is_heading=1), P(table_region=1)]
POSITIONAL_FEATURE_DIM = 3
POSITIONAL_FEATURE_NAMES = (
    "block_index_norm",
    "page_index_norm",
    "doc_position_norm",
)
SIDE_CHANNEL_DIM = LAYOUT_FEATURE_DIM + CASCADE_DIM + POSITIONAL_FEATURE_DIM

ADAPTER_SPEC = LoRAAdapterSpec(
    bert_name="semantic",
    adapter_path=DEFAULT_ADAPTER_DIR,
    head_kind="multi_head",
    head_specs=(
        ("doc_role", len(DOC_ROLE_NAMES)),
        ("boilerplate", len(BOILERPLATE_LABELS)),
    ),
)


# ---------------------------------------------------------------------------
# Cascade + positional helpers
# ---------------------------------------------------------------------------


def _read_cascade(span: Any) -> list[float]:
    """Pull the cascade vector off a span. Accepts both ``.cascade``
    attribute and ``span["cascade"]`` dict-key forms (different
    upstream featurizers use different shapes). Raises ``ValueError``
    if the cascade is missing or malformed — see module docstring for
    why we don't silently zero-fill."""
    cascade = getattr(span, "cascade", None)
    if cascade is None and isinstance(span, dict):
        cascade = span.get("cascade")
    if cascade is None:
        raise ValueError(
            "BERT-Semantic span is missing its cascade vector. The "
            "council orchestrator must run BERT-Structure first and "
            "attach an 8-dim cascade vector to each span before "
            "invoking Semantic. See dart_semantic/council/semantic.py "
            "module docstring."
        )
    vec = list(cascade)
    if len(vec) != CASCADE_DIM:
        raise ValueError(
            f"cascade vector has wrong dim: {len(vec)} != {CASCADE_DIM}"
        )
    out: list[float] = []
    for x in vec:
        try:
            f = float(x)
        except (TypeError, ValueError):
            f = 0.0
        if not math.isfinite(f):
            f = 0.0
        out.append(f)
    return out


def _clip01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def compute_positional_features(
    spans: list[Any],
) -> list[list[float]]:
    """Build a 3-dim positional vector per span.

    Mirrors :func:`data.build_semantic_data.compute_positional_features`
    but operates on a span list rather than a single block + scalar
    document totals. Uses input order as ``block_idx`` (so the caller's
    span ordering is what defines positional coordinates — the same
    convention the data builder uses, where blocks are sorted by
    (page, y0, x0) before alignment).
    """
    pages = []
    for fb in spans:
        raw = getattr(fb, "raw", fb)
        page = getattr(raw, "page", None)
        if page is None and isinstance(raw, dict):
            page = raw.get("page")
        try:
            pages.append(int(page or 0))
        except (TypeError, ValueError):
            pages.append(0)
    total_blocks = len(spans)
    if total_blocks == 0:
        return []
    first_page_num = min(pages)
    total_pages = max(1, max(pages) - first_page_num + 1)

    out: list[list[float]] = []
    for i, fb in enumerate(spans):
        raw = getattr(fb, "raw", fb)
        page_num = pages[i]
        page_h = float(getattr(raw, "page_height", None)
                       or (raw.get("page_height") if isinstance(raw, dict) else 0)
                       or 792.0)
        bbox = getattr(raw, "bbox", None)
        if bbox is None and isinstance(raw, dict):
            bbox = raw.get("bbox")
        bbox = list(bbox or [0.0, 0.0, 0.0, 0.0])
        y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0

        block_index_norm = i / max(1, total_blocks - 1)
        page_index_norm = (
            (page_num - first_page_num) / max(1, total_pages - 1)
        )
        doc_position_norm = (
            ((page_num - first_page_num) * page_h + y0)
            / max(1.0, total_pages * page_h)
        )
        out.append([
            _clip01(block_index_norm),
            _clip01(page_index_norm),
            _clip01(doc_position_norm),
        ])
    return out


# ---------------------------------------------------------------------------
# Head loading
# ---------------------------------------------------------------------------


def _build_side_mlp(side_dim: int, side_hidden: int) -> Any:
    import torch.nn as nn  # noqa: WPS433
    return nn.Sequential(
        nn.Linear(side_dim, side_hidden),
        nn.ReLU(),
        nn.Linear(side_hidden, side_hidden),
        nn.ReLU(),
    )


def _load_heads(heads_path: Path, hidden_size: int) -> dict[str, Any]:
    """Load the 2 classification heads + the 31-dim side-channel
    (LayerNorm + MLP). Returns a dict with keys ``doc_role``,
    ``boilerplate``, ``side_norm``, ``side_mlp``, ``side_channel_dim``,
    ``side_mlp_hidden``.

    The checkpoint stores ``side_channel_dim`` so a runtime built on
    a different SIDE_CHANNEL_DIM can detect the mismatch loudly rather
    than silently mis-shape the side channel."""
    import torch  # noqa: WPS433
    import torch.nn as nn  # noqa: WPS433

    state = torch.load(str(heads_path), map_location="cpu")
    side_dim = int(state.get("side_channel_dim", SIDE_CHANNEL_DIM))
    side_hidden = int(state.get("side_mlp_hidden", 64))
    if side_dim != SIDE_CHANNEL_DIM:
        raise ValueError(
            f"semantic heads.pt side_channel_dim={side_dim} but "
            f"runtime SIDE_CHANNEL_DIM={SIDE_CHANNEL_DIM} "
            f"(layout={LAYOUT_FEATURE_DIM} + cascade={CASCADE_DIM} + "
            f"positional={POSITIONAL_FEATURE_DIM}); rebuild Semantic "
            f"with the matching feature contract"
        )
    head_in = hidden_size + side_hidden
    head_doc_role = nn.Linear(head_in, len(DOC_ROLE_NAMES))
    head_boilerplate = nn.Linear(head_in, len(BOILERPLATE_LABELS))
    head_doc_role.load_state_dict(state["head_doc_role.state_dict"])
    head_boilerplate.load_state_dict(state["head_boilerplate.state_dict"])
    side_norm = nn.LayerNorm(side_dim)
    side_norm.load_state_dict(state["side_norm.state_dict"])
    side_mlp = _build_side_mlp(side_dim, side_hidden)
    side_mlp.load_state_dict(state["side_mlp.state_dict"])
    return {
        "doc_role": head_doc_role,
        "boilerplate": head_boilerplate,
        "side_norm": side_norm,
        "side_mlp": side_mlp,
        "side_channel_dim": side_dim,
        "side_mlp_hidden": side_hidden,
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
# Runtime entry — span-level forward
# ---------------------------------------------------------------------------


def run_inputs(adapter: LoRAAdapter, inputs: Any) -> BertOutput:
    """Per-BERT runner registered with ``runner.register_runner``.

    Parameters
    ----------
    adapter
        :class:`LoRAAdapter` already loaded onto the shared backbone.
    inputs
        A list of :class:`FeatureBlock`-like objects with ``.cascade``
        already attached. See module docstring.
    """
    import torch  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    if not isinstance(inputs, (list, tuple)):
        spans = [inputs]
    else:
        spans = list(inputs)

    backbone = adapter.backbone
    spec = adapter.spec
    name = "semantic"

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
    head_doc_role = heads_bundle["doc_role"].to(device)
    head_boilerplate = heads_bundle["boilerplate"].to(device)
    side_norm = heads_bundle["side_norm"].to(device)
    side_mlp = heads_bundle["side_mlp"].to(device)

    peft_model = adapter.peft_model
    peft_model.eval()

    # Build per-span text + 20-dim layout vector (page-grouped to
    # compute per-page medians the same way Structure does), then the
    # cascade and positional pieces, then concatenate into a 31-dim
    # side channel in input-order.
    span_texts: list[str] = [""] * len(spans)
    span_layouts: list[list[float]] = [
        [0.0] * LAYOUT_FEATURE_DIM for _ in spans
    ]
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
            span_texts[i] = text
            span_layouts[i] = _compute_span_layout(
                fb,
                page_w=page_w, page_h=page_h,
                page_median_fs=median_fs,
                page_median_h=median_h,
                in_table=in_table,
            )

    cascades = [_read_cascade(fb) for fb in spans]
    positionals = compute_positional_features(spans)
    side_channel = [
        layout + cascade + positional
        for layout, cascade, positional
        in zip(span_layouts, cascades, positionals)
    ]

    enc = tok(span_texts, padding=True, truncation=True, max_length=192,
              return_tensors="pt").to(device)
    side_t = torch.tensor(side_channel, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = peft_model(input_ids=enc["input_ids"],
                         attention_mask=enc["attention_mask"])
        pooled = out.last_hidden_state[:, 0, :].float()
        side_h = side_mlp(side_norm(side_t))
        h = torch.cat([pooled, side_h], dim=-1)
        logits_dr = head_doc_role(h)
        logits_bp = head_boilerplate(h)

    signals: list[TypedSignal] = []
    for i, fb in enumerate(spans):
        raw = getattr(fb, "raw", None)
        prov = {"page": int(getattr(raw, "page", 0) or 0)}
        signals.append(_softmax_signal(
            "doc_role", i, logits_dr[i],
            DOC_ROLE_NAMES, feature_provenance=prov,
        ))
        signals.append(_softmax_signal(
            "boilerplate", i, logits_bp[i],
            BOILERPLATE_LABELS, feature_provenance=prov,
        ))

    return BertOutput(
        bert_name=name,
        signals=signals,
        backbone_version=f"{backbone.name}@{backbone.revision}",
        adapter_version=f"{name}@{spec.adapter_path}",
    )


# Self-register on import.
register_adapter(ADAPTER_SPEC)
register_runner("semantic", run_inputs)


__all__ = [
    "ADAPTER_SPEC",
    "BOILERPLATE_LABELS",
    "CASCADE_DIM",
    "DEFAULT_ADAPTER_DIR",
    "DOC_ROLE_NAMES",
    "POSITIONAL_FEATURE_DIM",
    "POSITIONAL_FEATURE_NAMES",
    "SIDE_CHANNEL_DIM",
    "compute_positional_features",
    "run_inputs",
]
