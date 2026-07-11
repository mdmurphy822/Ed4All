"""Deterministic assembler package (Stage 9 of the v2 pipeline).

Stitches per-region top-1 candidates into a coherent document, normalizes
the heading tree, wires landmarks, resolves cross-references, and emits
the GapSlot list that Stage 9b (Qwen gap-fill) consumes.

Phase 0 lands the type contract per
Plans/04_assembler_layer_investigation.md §10. Implementation lands in
Phase 9.
"""

from .api import AssemblerConfig, assemble_document
from .types import (
    AssembledDoc,
    CertificationStatus,
    DocCandidate,
    ExitDecision,
    GapKind,
    GapSlot,
)

__all__ = [
    "AssembledDoc",
    "AssemblerConfig",
    "CertificationStatus",
    "DocCandidate",
    "ExitDecision",
    "GapKind",
    "GapSlot",
    "assemble_document",
]
