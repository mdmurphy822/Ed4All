"""Stage 9 — back-compat shim. Implementation lives in :mod:`.api`.

The Phase-0 ``assemble(top1_regions, council_state)`` stub has been
retired. New callers should import :func:`assemble_document` from
:mod:`semantik_structure.assembler.api` (or the package root).
"""
from __future__ import annotations

from .api import AssemblerConfig, assemble_document


__all__ = ["AssemblerConfig", "assemble_document"]
