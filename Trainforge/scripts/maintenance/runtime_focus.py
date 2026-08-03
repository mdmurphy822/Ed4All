"""Canonical objective-focus helper for synthesis measurement harnesses."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from Trainforge.generators.synthesis_window_contract import objective_card
from Trainforge.synthesis.synthesis_eligibility import (
    focus_chunk_on_canonical_objective,
)


def apply_runtime_focus(
    chunk: Mapping[str, Any],
    objectives: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach the exact canonical objective used by production synthesis."""
    result = focus_chunk_on_canonical_objective(
        chunk, seed=0, objectives=objectives,
    )
    refs = [str(ref).lower() for ref in result.get("learning_outcome_refs") or []]
    existing = result.get("synthesis_focus_objective")
    focus_id = (
        str(existing.get("id")).lower()
        if isinstance(existing, Mapping) and existing.get("id")
        else (refs[0] if len(refs) == 1 else "")
    )
    focus = objectives.get(focus_id)
    if focus is None and isinstance(existing, Mapping):
        focus = existing
    if not focus_id or focus is None:
        reason = str(
            result.get("synthesis_focus_skip_reason")
            or "runtime_focus_unresolvable"
        )
        raise ValueError(reason)
    card = objective_card(focus)
    result["synthesis_focus_objective"] = deepcopy(dict(focus))
    result["learning_outcome_refs"] = [card["id"]]
    result["bloom_level"] = card["bloom_level"]
    return result


__all__ = ["apply_runtime_focus"]
