"""Wave 137b - CURIE family map loader.

Cross-CURIE coupling: families are clusters that exhibit asymmetric
anchoring failure when partially complete (e.g. ``sh:minCount`` complete
plus ``sh:maxCount`` degraded -> adapter learns one side of cardinality).
The :class:`FamilyMap` exposes per-family CURIE lists + a reverse-index
``family_of`` mapping; downstream consumers (the
``FamilyCompletenessValidator`` post-training gate, the family-clustered
backfill ordering in ``Trainforge/scripts/backfill_form_data.py``, and
Plan D's checkpoint emitter via :func:`compute_family_coverage`) all read
the same loader output.

Map files live at ``schemas/training/family_map.<family>.yaml`` and are
schema-validated against
``schemas/training/family_map.schema.json`` on load. The loader cross-
checks the declared CURIEs against the family's property manifest when
one is reachable; manifest-missing CURIEs raise. Missing map files
return ``None`` so the validator can no-op cleanly on families without
a declared family map.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_DIR = PROJECT_ROOT / "schemas" / "training"


@dataclass(frozen=True)
class FamilyMap:
    """Frozen snapshot of one family map file.

    Attributes:
        family: Family slug (matches the YAML's ``family`` field).
        families: Mapping ``family_name -> [curies]``; every list has
            at least 2 CURIEs.
        singletons: CURIEs that belong to no family. Evaluated
            independently by the family-completeness gate.
        family_of: Reverse index ``curie -> family_name`` (or the
            sentinel ``"<singleton>"`` for entries in ``singletons``).
    """

    family: str
    families: Dict[str, List[str]]
    singletons: List[str]
    family_of: Dict[str, str]


def _validate_partition(
    family: str,
    families: Dict[str, List[str]],
    singletons: List[str],
    manifest_curies: Optional[List[str]],
) -> None:
    """Enforce the partition invariants.

    Raises:
        ValueError: when any family declares <2 CURIEs, when a CURIE
            appears in two families, when a CURIE is in both a family
            and the singletons list, or when a manifest cross-check is
            requested and the family map references CURIEs that aren't
            in the manifest.
    """
    seen: Dict[str, str] = {}
    for fam_name, curies in families.items():
        if len(curies) < 2:
            raise ValueError(
                f"family_map[{family}]: family '{fam_name}' has <2 CURIEs "
                f"(got {len(curies)}); a 'family' with <2 members is a "
                "singleton and must move to the singletons list."
            )
        for c in curies:
            if c in seen:
                raise ValueError(
                    f"family_map[{family}]: CURIE '{c}' appears in two "
                    f"families: '{seen[c]}' and '{fam_name}'. CURIEs must "
                    "partition into exactly one family or singletons."
                )
            seen[c] = fam_name
    for c in singletons:
        if c in seen:
            raise ValueError(
                f"family_map[{family}]: CURIE '{c}' is in family "
                f"'{seen[c]}' AND singletons; pick exactly one."
            )
        seen[c] = "<singleton>"
    if manifest_curies is not None:
        manifest_set = set(manifest_curies)
        for c in seen:
            if c not in manifest_set:
                raise ValueError(
                    f"family_map[{family}] references CURIE '{c}' that is "
                    f"not declared in the property manifest. Add the CURIE "
                    "to the manifest or remove it from the family map."
                )


def _try_load_manifest_curies(family: str) -> Optional[List[str]]:
    """Best-effort manifest cross-check inputs.

    Reads the property manifest YAML directly so we don't trigger the
    manifest's own schema validation (which would need ``properties[]``
    populated for a fresh family map being authored against a manifest
    in flux). Returns ``None`` on any failure so the partition check
    skips the manifest cross-check.
    """
    candidate = _SCHEMA_DIR / f"property_manifest.{family}.yaml"
    if not candidate.exists():
        return None
    try:
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.debug(
            "family_map[%s]: property manifest parse failed (%s); "
            "skipping manifest cross-check.",
            family,
            exc,
        )
        return None
    props = payload.get("properties") or []
    out: List[str] = []
    for p in props:
        if isinstance(p, dict) and isinstance(p.get("curie"), str):
            out.append(p["curie"])
    return out or None


@lru_cache(maxsize=8)
def load_family_map(family: str) -> Optional[FamilyMap]:
    """Load + validate ``family_map.<family>.yaml``.

    Returns:
        :class:`FamilyMap` on success, or ``None`` when the map file
        does not exist (the validator no-ops cleanly so families
        without a declared family map don't break the gate).

    Raises:
        ValueError: when the YAML parses but the partition invariants
            fail (see :func:`_validate_partition`).
        ``jsonschema.ValidationError``: when the JSON Schema validation
            fails. Cached results never leak partial state — the
            ``@lru_cache`` decorator caches the dataclass only on
            successful return.
    """
    yaml_path = _SCHEMA_DIR / f"family_map.{family}.yaml"
    if not yaml_path.exists():
        return None
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.error(
            "family_map[%s]: malformed YAML at %s: %s",
            family,
            yaml_path,
            exc,
        )
        return None
    if not isinstance(data, dict):
        raise ValueError(
            f"family_map[{family}] at {yaml_path} did not parse to an "
            "object."
        )

    schema_path = _SCHEMA_DIR / "family_map.schema.json"
    if schema_path.exists():
        try:
            import jsonschema  # noqa: WPS433 - optional dep, lazy import
        except ImportError:  # pragma: no cover - dev installs always have it
            logger.warning(
                "jsonschema not installed; skipping family_map schema "
                "validation for family=%s",
                family,
            )
        else:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(data, schema)

    families_raw = data.get("families") or {}
    singletons_raw = data.get("singletons") or []
    families: Dict[str, List[str]] = {
        str(name): list(curies)
        for name, curies in families_raw.items()
    }
    singletons: List[str] = list(singletons_raw)

    manifest_curies = _try_load_manifest_curies(family)
    _validate_partition(
        data.get("family", family),
        families,
        singletons,
        manifest_curies,
    )

    family_of: Dict[str, str] = {}
    for fam_name, curies in families.items():
        for c in curies:
            family_of[c] = fam_name
    for c in singletons:
        family_of[c] = "<singleton>"

    return FamilyMap(
        family=str(data.get("family", family)),
        families=families,
        singletons=singletons,
        family_of=family_of,
    )


def compute_family_coverage(
    fm: FamilyMap,
    form_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-family complete / partial / untouched coverage.

    Pure helper exported for Plan D's checkpoint emitter. Reads each
    family's CURIE list and counts how many entries in ``form_data``
    have ``anchored_status="complete"``. Entries missing from
    ``form_data`` count as not-complete.

    Returns:
        Mapping ``family_name -> {complete, total, status, curies}``
        where ``status`` is ``"complete"`` (all family CURIEs complete),
        ``"untouched"`` (none complete), or ``"partial"`` (mixed).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for fam_name, curies in fm.families.items():
        complete_count = 0
        for c in curies:
            entry = form_data.get(c)
            if entry is None:
                continue
            status = getattr(entry, "anchored_status", "complete")
            if status == "complete":
                complete_count += 1
        total = len(curies)
        if complete_count == total:
            family_status = "complete"
        elif complete_count == 0:
            family_status = "untouched"
        else:
            family_status = "partial"
        out[fam_name] = {
            "complete": complete_count,
            "total": total,
            "status": family_status,
            "curies": list(curies),
        }
    return out


__all__ = ["FamilyMap", "load_family_map", "compute_family_coverage"]
