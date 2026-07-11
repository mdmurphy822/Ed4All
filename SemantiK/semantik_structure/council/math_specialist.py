"""BERT-MathSpecialist runtime — wraps the trained adapter for inference.

Phase 2 contract:
    * Inputs:  list of :class:`MathCandidate` (Phase 1 output).
    * Outputs: :class:`BertOutput` carrying two :class:`TypedSignal`s
               per candidate — ``math_type`` and ``eq_num_assoc``.

Adapter loading is delegated to ``runner.run_bert("math_specialist",
candidates)`` which calls into :func:`run_inputs` registered below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths as _semantik_paths
from .base import LoRAAdapter, LoRAAdapterSpec, load_local_tokenizer
from .registry import register_adapter
from .runner import register_runner
from .types import BertOutput, MathCandidate, TypedSignal


# Disk pointer + head spec — kept tight for the registry. The actual
# trained weights live under ``models/council/math_specialist/final``;
# the trainer (``train_math_specialist.py``) writes them there. Tests
# that don't need the trained adapter monkey-patch this Path.
DEFAULT_ADAPTER_DIR = _semantik_paths.resolve_model("council/math_specialist/final")

MATH_TYPES = ("inline", "display", "numbered", "multiline", "matrix")
EQ_NUM_ASSOC = ("none", "attached_left", "attached_right", "separate")

ADAPTER_SPEC = LoRAAdapterSpec(
    bert_name="math_specialist",
    adapter_path=DEFAULT_ADAPTER_DIR,
    head_kind="multi_head",
    head_specs=(
        ("math_type", len(MATH_TYPES)),
        ("eq_num_assoc", len(EQ_NUM_ASSOC)),
    ),
)


def _candidate_to_text(cand: MathCandidate) -> str:
    """Build the BERT input string for one MathCandidate.

    Mirrors the format used by the trainer (``train_math_specialist.py
    :: _format_input``): ``"[math] {alttext_or_text} [ctx] {context}"``.
    Phase-1 candidates carry ``evidence_features['src_text']`` (the
    geometric region's raw text) — we use that as the math content,
    and concatenate any neighbouring feature-block text as ``[ctx]``.
    """
    src = (cand.evidence_features or {}).get("src_text", "")
    src = (src or "").strip()
    parts = []
    if src:
        parts.append(f"[math] {src}")
    # Phase 1 candidates do NOT yet carry surrounding-paragraph text
    # (that lives in the FeatureBlock stream, indexed by
    # member_block_indices). Until the runner is given the full
    # FeatureSet, we fall back to the candidate's evidence string.
    return " ".join(parts) or "[math]"


def _load_heads(heads_path: Path, hidden_size: int) -> Any:
    import torch.nn as nn  # noqa: WPS433
    import torch  # noqa: WPS433
    state = torch.load(str(heads_path), map_location="cpu")
    head_mt = nn.Linear(hidden_size, len(MATH_TYPES))
    head_en = nn.Linear(hidden_size, len(EQ_NUM_ASSOC))
    head_mt.load_state_dict(state["head_math_type.state_dict"])
    head_en.load_state_dict(state["head_eq_num.state_dict"])
    return head_mt, head_en


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
    top_idx_list = top_idx.tolist()
    return TypedSignal(
        head_name=head_name,
        region_id=region_id,
        top_k_labels=[label_names[i] for i in top_idx_list],
        top_k_confidences=[float(p) for p in top_probs.tolist()],
        feature_provenance=feature_provenance or {},
    )


def run_inputs(adapter: LoRAAdapter, inputs: Any) -> BertOutput:
    """Per-BERT runner registered with ``runner.register_runner``.

    Parameters
    ----------
    adapter
        The :class:`LoRAAdapter` already loaded onto the shared backbone.
    inputs
        A list of :class:`MathCandidate` instances. Each becomes one
        ``region_id`` in the resulting ``BertOutput``.
    """
    import torch  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    if not isinstance(inputs, (list, tuple)):
        candidates = [inputs]
    else:
        candidates = list(inputs)

    backbone = adapter.backbone
    spec = adapter.spec

    if not candidates:
        # No work — return an empty BertOutput without paying for tokenizer
        # / head load. Callers commonly pass [] when the page has no math.
        return BertOutput(
            bert_name="math_specialist",
            signals=[],
            backbone_version=f"{backbone.name}@{backbone.revision}",
            adapter_version=f"math_specialist@{spec.adapter_path}",
        )

    # Tokenizer is colocated with the saved adapter under
    # ``<adapter_dir>/tokenizer/``.
    tok_dir = spec.adapter_path / "tokenizer"
    if tok_dir.exists():
        tok = load_local_tokenizer(tok_dir, fallback_name=backbone.name)
    else:
        tok = AutoTokenizer.from_pretrained(backbone.name)

    head_mt, head_en = _load_heads(
        spec.adapter_path / "heads.pt",
        hidden_size=backbone.hidden_size,
    )
    device = backbone.device or "cpu"
    head_mt = head_mt.to(device)
    head_en = head_en.to(device)

    peft_model = adapter.peft_model
    peft_model.eval()

    signals: list[TypedSignal] = []
    texts = [_candidate_to_text(c) for c in candidates]
    enc = tok(texts, padding=True, truncation=True, max_length=192,
              return_tensors="pt").to(device)
    with torch.no_grad():
        out = peft_model(input_ids=enc["input_ids"],
                         attention_mask=enc["attention_mask"])
        pooled = out.last_hidden_state[:, 0, :].float()
        logits_mt = head_mt(pooled)
        logits_en = head_en(pooled)

    for region_id, cand in enumerate(candidates):
        prov = {
            "bbox": list(cand.bbox),
            "pages": list(cand.pages),
        }
        if isinstance(cand, MathCandidate):
            prov["glyph_density_features"] = dict(cand.glyph_density_features)
        signals.append(_softmax_signal(
            "math_type", region_id, logits_mt[region_id],
            MATH_TYPES, feature_provenance=prov,
        ))
        signals.append(_softmax_signal(
            "eq_num_assoc", region_id, logits_en[region_id],
            EQ_NUM_ASSOC, feature_provenance=prov,
        ))

    return BertOutput(
        bert_name="math_specialist",
        signals=signals,
        backbone_version=f"{backbone.name}@{backbone.revision}",
        adapter_version=f"math_specialist@{spec.adapter_path}",
    )


# Register on import — runner._ensure_runner_imported triggers this.
register_adapter(ADAPTER_SPEC)
register_runner("math_specialist", run_inputs)


__all__ = [
    "ADAPTER_SPEC",
    "DEFAULT_ADAPTER_DIR",
    "EQ_NUM_ASSOC",
    "MATH_TYPES",
    "run_inputs",
]
