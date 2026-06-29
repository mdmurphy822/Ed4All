"""BERT-ImageSpecialist runtime — STUB / follow-on (AXIS-3).

Status: **STUB, intentionally NOT registered.** This module declares the
membership→detail contract for the figure/image specialist (the third council
gating pair, mirroring ``table_specialist.py``) but does NOT
``register_adapter`` / ``register_runner`` on import, because no trained
``council/image_specialist`` adapter exists yet. Importing it is harmless (no
side effects, no model load); a real build must (1) author a
``data/build_image_specialist_data.py`` membership→detail labeler, (2) add a
``training/train_image_specialist.py`` trainer, (3) train the adapter, and (4)
flip ``_REGISTERED`` on / call :func:`register` at the bottom.

Gating contract (the half that IS live today):
    * Structure's ``is_image_block`` binary head (``council/structure.py``)
      already DETECTS figure/image-block membership end-to-end — it is labeled
      by ``data.build_structure_data`` (``labels.is_image_block``), trained by
      ``training.train_structure`` (``head_is_image_block``), and emitted by the
      council runtime whenever the loaded checkpoint carries trained weights
      (``has_is_image_block_weights``). On a v3-trained head the signal
      auto-emits; the runtime guard only suppresses random output from a
      pre-Phase-3f 4-head checkpoint. So ``is_image_block`` is the "membership"
      half — this specialist would own the "detail" half (below).

Detail head this specialist WOULD own (membership=1 → parse):
    * ``caption_role``   : {none, figure_caption, table_caption}
    * ``caption_position``: {none, above, below}
    * ``is_alt_candidate``: binary — whether the block is alt-text-bearing.

These mirror the ``is_heading``→HeadingSpecialist and
``table_region``→TableSpecialist gating: Structure detects membership, the
specialist parses the cell/caption-level detail. The label names below are the
authored vocabulary the data builder + trainer must agree on.

This file is a placeholder so the contract is checked in and the follow-on is
unambiguous; it is NOT wired into the council DAG.
"""
from __future__ import annotations

from typing import Any

# Detail-head label vocabularies (the contract for the follow-on data builder
# + trainer). NOT consumed by anything yet.
CAPTION_ROLE_NAMES: tuple[str, ...] = (
    "none",
    "figure_caption",
    "table_caption",
)
CAPTION_POSITION_NAMES: tuple[str, ...] = (
    "none",
    "above",
    "below",
)
IS_ALT_CANDIDATE_LABELS: tuple[str, ...] = (
    "not_alt_candidate",
    "alt_candidate",
)

# Mirror TableSpecialist's geometric side-channel width as the planned input
# contract (the follow-on data builder pins the real dim).
LAYOUT_FEATURE_DIM = 17
LAYOUT_MLP_HIDDEN = 64

# Flip to True and call register() once a trained adapter ships.
_REGISTERED = False


def run_inputs(adapter: Any, inputs: Any) -> Any:  # pragma: no cover - stub
    """STUB. The detail-parsing forward pass is a follow-on; see module docstring.

    Raises so a premature wiring fails loud rather than silently emitting junk
    (mirrors the anti-poisoning posture of the rest of the council)."""
    raise NotImplementedError(
        "ImageSpecialist is a follow-on stub (AXIS-3): no trained "
        "council/image_specialist adapter exists. Build "
        "data/build_image_specialist_data.py + training/train_image_specialist.py, "
        "train the adapter, then register() this runner. The is_image_block "
        "MEMBERSHIP head (council/structure.py) is already live."
    )


def register() -> None:  # pragma: no cover - stub
    """Follow-on hook: register the adapter + runner once weights exist."""
    raise NotImplementedError(
        "ImageSpecialist adapter is not trained yet — see module docstring."
    )


__all__ = [
    "CAPTION_ROLE_NAMES",
    "CAPTION_POSITION_NAMES",
    "IS_ALT_CANDIDATE_LABELS",
    "LAYOUT_FEATURE_DIM",
    "LAYOUT_MLP_HIDDEN",
    "run_inputs",
    "register",
]
