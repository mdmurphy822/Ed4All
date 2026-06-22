"""Multi-head wrapper over the shared backbone.

A council BERT runs the backbone (or its LoRA-wrapped adapter) once and
dispatches the resulting CLS / mean-pooled hidden state to N small
classification heads. The heads themselves are cheap — single
``nn.Linear`` projections from ``hidden_size`` to ``num_classes``. LoRA
matrices live on the backbone, not on the heads.

This module is intentionally tiny: it carries no training logic (that
lives in ``train_math_specialist.py``); its job is to be the runtime
forward path the council scheduler calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HeadSpec:
    """One head's name + label cardinality.

    ``label_names`` is optional metadata threaded through to ``BertOutput``
    so the council scheduler can label the predicted indices without
    re-loading the dataset.
    """
    head_name: str
    num_classes: int
    label_names: tuple[str, ...] = ()


def build_multi_head_model(
    backbone: Any,
    head_specs: list[HeadSpec] | list[tuple[str, int]],
    *,
    pooling: str = "cls",
) -> "MultiHeadModel":
    """Construct a :class:`MultiHeadModel` over ``backbone``.

    ``head_specs`` accepts either ``HeadSpec`` instances or bare
    ``(name, num_classes)`` tuples for ergonomic call sites.
    """
    normed: list[HeadSpec] = []
    for spec in head_specs:
        if isinstance(spec, HeadSpec):
            normed.append(spec)
        else:
            head_name, num_classes = spec
            normed.append(HeadSpec(head_name=head_name, num_classes=int(num_classes)))
    return MultiHeadModel(backbone, normed, pooling=pooling)


class MultiHeadModel:
    """Wrap a backbone in N parallel classification heads.

    Imports torch/nn lazily so ``import dart_semantic.council.multihead``
    stays cheap. The actual ``nn.Linear`` heads are built on first
    ``forward`` call (or via :meth:`build_heads`).
    """

    def __init__(
        self,
        backbone: Any,
        head_specs: list[HeadSpec],
        *,
        pooling: str = "cls",
    ) -> None:
        if not head_specs:
            raise ValueError("MultiHeadModel requires at least one head spec")
        self.backbone = backbone
        self.head_specs = list(head_specs)
        if pooling not in ("cls", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}; use 'cls' or 'mean'")
        self.pooling = pooling
        self._heads: dict[str, Any] | None = None
        self._module: Any | None = None  # nn.Module wrapper for state_dict

    # -- head construction --------------------------------------------------

    def build_heads(self) -> None:
        """Materialise the per-head ``nn.Linear`` projections.

        Idempotent — a second call leaves the already-built heads in
        place. Tests can call this explicitly to force the heavy import
        before exercising shape-only properties.
        """
        if self._heads is not None:
            return
        import torch.nn as nn  # noqa: WPS433

        # Prefer the LoRA-wrapped peft model's hidden size when one is
        # mounted; fall back to backbone.hidden_size otherwise.
        hidden = getattr(self.backbone, "hidden_size", None)
        if hidden is None:
            cfg = getattr(getattr(self.backbone, "model", None), "config", None)
            hidden = int(getattr(cfg, "hidden_size", 768)) if cfg is not None else 768

        heads: dict[str, nn.Module] = {}
        for spec in self.head_specs:
            heads[spec.head_name] = nn.Linear(hidden, spec.num_classes)

        # Bundle in an nn.ModuleDict so save_pretrained / state_dict
        # round-trips capture the heads alongside any future bookkeeping.
        module = nn.ModuleDict(heads)
        # Move heads onto the same device as the backbone, if any.
        device = getattr(self.backbone, "device", None) or getattr(
            getattr(self.backbone, "model", None), "device", None,
        )
        if device is not None:
            module = module.to(device)
        self._heads = dict(heads)
        self._module = module

    @property
    def heads(self) -> dict[str, Any]:
        if self._heads is None:
            self.build_heads()
        assert self._heads is not None
        return self._heads

    @property
    def module(self) -> Any:
        """The ``nn.ModuleDict`` carrying every head — for ``state_dict``."""
        if self._module is None:
            self.build_heads()
        return self._module

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        *,
        active_model: Any | None = None,
    ) -> dict[str, Any]:
        """Run a single backbone forward and return per-head logits.

        ``active_model`` overrides ``self.backbone.model`` — pass the
        ``peft.PeftModel`` returned by :class:`LoRAAdapter` when a LoRA
        is mounted, so the LoRA matrices are exercised on this pass.
        """
        import torch  # noqa: WPS433

        model = active_model if active_model is not None else self.backbone.model
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # (batch, seq, hidden)
        if self.pooling == "cls":
            pooled = last_hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom

        if self._heads is None:
            self.build_heads()
        assert self._heads is not None
        outputs: dict[str, Any] = {}
        for name, head in self._heads.items():
            outputs[name] = head(pooled.float())
        return outputs

    # -- introspection ------------------------------------------------------

    @property
    def head_names(self) -> list[str]:
        return [s.head_name for s in self.head_specs]

    def num_classes(self, head_name: str) -> int:
        for s in self.head_specs:
            if s.head_name == head_name:
                return s.num_classes
        raise KeyError(head_name)


__all__ = [
    "HeadSpec",
    "MultiHeadModel",
    "build_multi_head_model",
]
