"""Wave 137d-2 — pure helper for FORM_DATA coverage metrics.

Consumed by ``EvalGatingValidator``'s checkpoint-emission step
(landed in the same wave). The helper computes:

  * ``manifest_coverage_pct`` — proportion of manifest-declared CURIEs
    whose form_data entry is ``anchored_status="complete"``.
  * ``complete_count`` / ``degraded_count`` — absolute counts.
  * ``family_coverage_map`` — per-family completeness map via
    Wave 137b's :func:`compute_family_coverage`. Empty when the
    family has no declared family map.

Pure: when both ``form_data`` and ``manifest`` are passed in, the
helper does no I/O. The default lookup paths fall back to the
canonical loaders so call-sites that don't already hold these
references can pass ``family`` only.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def compute_coverage_metrics(
    family: str,
    *,
    form_data: Optional[Dict[str, Any]] = None,
    manifest: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute coverage % + family map + per-CURIE complete/degraded counts.

    Args:
        family: Family slug (e.g., ``rdf_shacl``). Used as the fallback
            key for the loaders when ``form_data`` / ``manifest`` are
            ``None``.
        form_data: Optional pre-loaded form_data dict. When ``None``,
            falls back to ``schema_translation_generator._load_form_data(family)``.
        manifest: Optional pre-loaded property manifest object exposing
            ``.properties[].curie``. When ``None``, attempts to load
            via the family slug; the ``load_property_manifest_by_family``
            symbol is looked up best-effort because it doesn't exist
            yet on the canonical loader (which keys off course slug).
            Failure to resolve a manifest yields ``None`` coverage
            metrics (caller can render the row as "no manifest").

    Returns:
        Dict with keys:
          * ``manifest_coverage_pct``: float in ``[0, 1]`` or ``None``
            when no manifest was resolvable.
          * ``complete_count``: int or ``None``.
          * ``degraded_count``: int or ``None``.
          * ``family_coverage_map``: dict from
            :func:`compute_family_coverage`. Empty dict when the
            family has no declared family map.
    """
    if form_data is None:
        from Trainforge.generators.schema_translation_generator import (
            _load_form_data,
        )
        form_data = _load_form_data(family)
    if manifest is None:
        from lib.ontology import property_manifest as _pm
        # ``load_property_manifest`` keys off course slug — it accepts
        # the family slug as a fallback (the slug normaliser is a
        # noop on family-shaped strings). The ``load_property_manifest_by_family``
        # symbol may exist on the loader in some forks; try it first,
        # fall back to the canonical loader otherwise.
        loader = getattr(_pm, "load_property_manifest_by_family", None)
        try:
            if loader is not None:
                manifest = loader(family)
            else:
                manifest = _pm.load_property_manifest(family)
        except (FileNotFoundError, AttributeError, ImportError) as exc:
            logger.debug(
                "form_data_coverage: no manifest for family=%r (%s); "
                "returning empty coverage row.",
                family,
                exc,
            )
            return {
                "manifest_coverage_pct": None,
                "complete_count": None,
                "degraded_count": None,
                "family_coverage_map": {},
            }

    manifest_curies = [p.curie for p in manifest.properties]
    complete_count = sum(
        1 for c in manifest_curies
        if c in form_data
        and getattr(form_data[c], "anchored_status", "complete") == "complete"
    )
    degraded_count = len(manifest_curies) - complete_count
    coverage_pct = (
        round(complete_count / len(manifest_curies), 4)
        if manifest_curies
        else 0.0
    )

    family_coverage_map: Dict[str, Dict[str, Any]] = {}
    try:
        from lib.ontology.family_map import (
            compute_family_coverage,
            load_family_map,
        )
        fm = load_family_map(family)
        if fm is not None:
            family_coverage_map = compute_family_coverage(fm, form_data)
    except (ImportError, FileNotFoundError) as exc:
        logger.debug(
            "form_data_coverage: family_map load skipped for family=%r (%s)",
            family,
            exc,
        )

    return {
        "manifest_coverage_pct": coverage_pct,
        "complete_count": complete_count,
        "degraded_count": degraded_count,
        "family_coverage_map": family_coverage_map,
    }


__all__ = ["compute_coverage_metrics"]
