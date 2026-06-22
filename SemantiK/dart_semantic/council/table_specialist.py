"""BERT-TableSpecialist runtime — wraps the trained 1-head adapter.

Phase 4 contract:
    * Inputs:  list of cell-like objects, each exposing ``text`` and
               ``layout`` (15-dim per data.build_table_specialist_data
               LAYOUT_FEATURE_NAMES). Accepts both dataclass-style
               objects (``cell.text`` / ``cell.layout``) and dict-style
               (``cell["text"]`` / ``cell["layout"]``).
    * Outputs: a :class:`BertOutput` carrying one :class:`TypedSignal`
               per CELL — head name ``cell_role_scope`` (4-class). Per
               cell ``region_id`` = the cell's index in the input list.

Layout side-channel (15-dim — same shape as the data builder):
    layout(15) → LayerNorm → MLP(15→64→64) → concatenated with
    ModernBERT pooled (CLS) → 1 head.

This BERT does NOT consume Structure cascade vectors in v1. The
architecture allows a soft-hint slot for ``Structure top-k``, but v1
relies on positional/text-shape signals only — see
data.build_table_specialist_data module docstring.

Mirrors the shape declared in :mod:`train_table_specialist`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths as _semantik_paths
from .base import LoRAAdapter, LoRAAdapterSpec
from .registry import register_adapter
from .runner import register_runner
from .types import BertOutput, TypedSignal


DEFAULT_ADAPTER_DIR = _semantik_paths.resolve_model("council/table_specialist/final_v5")

CELL_ROLE_SCOPE_NAMES: tuple[str, ...] = (
    "data",
    "header_col",
    "header_row",
    "span",
)
LAYOUT_FEATURE_DIM = 17
LAYOUT_MLP_HIDDEN = 64

ADAPTER_SPEC = LoRAAdapterSpec(
    bert_name="table_specialist",
    adapter_path=DEFAULT_ADAPTER_DIR,
    head_kind="multi_head",
    head_specs=(
        ("cell_role_scope", len(CELL_ROLE_SCOPE_NAMES)),
    ),
)


# ---------------------------------------------------------------------------
# Cell input helpers
# ---------------------------------------------------------------------------


def _cell_text(cell: Any) -> str:
    text = getattr(cell, "text", None)
    if text is None and isinstance(cell, dict):
        text = cell.get("text")
    return (text or "").strip()


def _cell_layout(cell: Any) -> list[float]:
    layout = getattr(cell, "layout", None)
    if layout is None and isinstance(cell, dict):
        layout = cell.get("layout")
    if layout is None:
        raise ValueError(
            "BERT-TableSpecialist cell is missing its layout vector. "
            "The cell-grid builder must compute the 15-dim layout per "
            "data.build_table_specialist_data.compute_cell_layout "
            "before invoking the runtime."
        )
    vec = [float(x) for x in layout]
    if len(vec) != LAYOUT_FEATURE_DIM:
        raise ValueError(
            f"cell layout has wrong dim: {len(vec)} != {LAYOUT_FEATURE_DIM}"
        )
    return vec


# ---------------------------------------------------------------------------
# Head loading
# ---------------------------------------------------------------------------


def _build_layout_mlp(layout_dim: int, mlp_hidden: int) -> Any:
    import torch.nn as nn  # noqa: WPS433
    return nn.Sequential(
        nn.Linear(layout_dim, mlp_hidden),
        nn.ReLU(),
        nn.Linear(mlp_hidden, mlp_hidden),
        nn.ReLU(),
    )


def _load_heads(heads_path: Path, hidden_size: int) -> dict[str, Any]:
    import torch  # noqa: WPS433
    import torch.nn as nn  # noqa: WPS433

    state = torch.load(str(heads_path), map_location="cpu")
    layout_dim = int(state.get("layout_dim", LAYOUT_FEATURE_DIM))
    mlp_hidden = int(state.get("layout_mlp_hidden", LAYOUT_MLP_HIDDEN))
    if layout_dim != LAYOUT_FEATURE_DIM:
        raise ValueError(
            f"table_specialist heads.pt layout_dim={layout_dim} but "
            f"runtime LAYOUT_FEATURE_DIM={LAYOUT_FEATURE_DIM}; rebuild "
            f"with the matching feature contract"
        )
    head_in = hidden_size + mlp_hidden
    head = nn.Linear(head_in, len(CELL_ROLE_SCOPE_NAMES))
    head.load_state_dict(state["head_cell_role_scope.state_dict"])
    layout_norm = nn.LayerNorm(layout_dim)
    layout_norm.load_state_dict(state["layout_norm.state_dict"])
    layout_mlp = _build_layout_mlp(layout_dim, mlp_hidden)
    layout_mlp.load_state_dict(state["layout_mlp.state_dict"])
    return {
        "cell_role_scope": head,
        "layout_norm": layout_norm,
        "layout_mlp": layout_mlp,
        "layout_dim": layout_dim,
        "layout_mlp_hidden": mlp_hidden,
    }


def _softmax_signal(
    head_name: str, region_id: int, logits: Any,
    label_names: tuple[str, ...], *, top_k: int = 3,
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
# Runtime entry — cell-level forward
# ---------------------------------------------------------------------------


def run_inputs(adapter: LoRAAdapter, inputs: Any) -> BertOutput:
    """Per-BERT runner registered with ``runner.register_runner``."""
    import torch  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    if not isinstance(inputs, (list, tuple)):
        cells = [inputs]
    else:
        cells = list(inputs)

    backbone = adapter.backbone
    spec = adapter.spec
    name = "table_specialist"

    if not cells:
        return BertOutput(
            bert_name=name, signals=[],
            backbone_version=f"{backbone.name}@{backbone.revision}",
            adapter_version=f"{name}@{spec.adapter_path}",
        )

    tok_dir = spec.adapter_path / "tokenizer"
    if tok_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
    else:
        tok = AutoTokenizer.from_pretrained(backbone.name)

    heads = _load_heads(
        spec.adapter_path / "heads.pt",
        hidden_size=backbone.hidden_size,
    )
    device = backbone.device or "cpu"
    head = heads["cell_role_scope"].to(device)
    layout_norm = heads["layout_norm"].to(device)
    layout_mlp = heads["layout_mlp"].to(device)

    peft_model = adapter.peft_model
    peft_model.eval()

    texts = [_cell_text(c) for c in cells]
    layouts = [_cell_layout(c) for c in cells]

    enc = tok(
        texts, padding=True, truncation=True, max_length=192,
        return_tensors="pt",
    ).to(device)
    layout_t = torch.tensor(layouts, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = peft_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        pooled = out.last_hidden_state[:, 0, :].float()
        layout_h = layout_mlp(layout_norm(layout_t))
        h = torch.cat([pooled, layout_h], dim=-1)
        logits = head(h)

    signals: list[TypedSignal] = []
    for i, _ in enumerate(cells):
        signals.append(_softmax_signal(
            "cell_role_scope", i, logits[i],
            CELL_ROLE_SCOPE_NAMES,
            feature_provenance={"cell_idx": i},
        ))

    return BertOutput(
        bert_name=name, signals=signals,
        backbone_version=f"{backbone.name}@{backbone.revision}",
        adapter_version=f"{name}@{spec.adapter_path}",
    )


register_adapter(ADAPTER_SPEC)
register_runner("table_specialist", run_inputs)


__all__ = [
    "ADAPTER_SPEC",
    "CELL_ROLE_SCOPE_NAMES",
    "DEFAULT_ADAPTER_DIR",
    "LAYOUT_FEATURE_DIM",
    "LAYOUT_MLP_HIDDEN",
    "run_inputs",
]
