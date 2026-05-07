"""Cached Draft 2020-12 validator builders with referencing.Registry support.

Replaces 2 independently-built validator paths:

- :mod:`Courseforge.scripts.generate_course._build_jsonld_validator` --
  curated sibling schemas + 5 taxonomy files.
- :mod:`Trainforge.process_course._load_chunk_validator` --
  rglob over a schemas root.

Both followed identical Draft 2020-12 + ``referencing.Registry``
construction; W-D6 collapses to one helper that supports either an
explicit ``extra_schemas`` list (Site A pattern) or a ``registry_root``
glob (Site B pattern).

The deprecated ``RefResolver`` fallback in Site B is dropped -- the
project depends on ``referencing`` everywhere else, so the Site B
fallback was defensive coding that's no longer load-bearing.

See plan ``plans/wave-D6-lib-utils-package-2026-05-07.md`` Section 3.3 for
the migration table + resolver-registry shape rationale.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = ["build_validator"]


def build_validator(
    schema_path: Union[str, Path],
    *,
    registry_root: Optional[Union[str, Path]] = None,
    extra_schemas: Optional[Iterable[Union[str, Path]]] = None,
    return_schema: bool = False,
) -> Optional[Any]:
    """Build a Draft 2020-12 validator with a referencing.Registry resolver.

    Args:
        schema_path: The primary schema file; loaded as the validator's root.
        registry_root: When set, rglob the directory for every ``*.json`` file
            and add every schema with a ``$id`` field to the registry.
            (Site B pattern -- full transitive ``$ref`` graph.)
        extra_schemas: When set, treat as a curated list of sibling schema
            paths to add. Mutually compatible with ``registry_root`` --
            both lists merge (curated wins on ``$id`` collision).
            (Site A pattern -- explicit taxonomies + source_reference.)
        return_schema: When True, return ``(validator, schema_dict)`` so the
            caller can inspect the schema (Site A's pattern). Default
            False -- return the validator only.

    Returns:
        The validator, or ``None`` when ``jsonschema`` / ``referencing``
        aren't installed (graceful-degrade path; matches both Site A
        and Site B's no-deps behavior). When ``return_schema=True`` and
        construction succeeds, returns a 2-tuple ``(validator, schema)``;
        on deps-missing, returns ``None`` regardless.

    Notes:
        Callers that need module-level caching should wrap the result in
        their own ``_VALIDATOR = None`` + lazy-load -- caching is
        intentionally NOT inside this helper because cache invalidation
        + thread safety are caller concerns. (Site A and Site B both
        cached at module scope; same pattern continues post-extraction.)
    """
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012
    except ImportError:
        logger.debug(
            "build_validator: jsonschema / referencing not installed; "
            "returning None (graceful-degrade path)."
        )
        return None

    schema_p = Path(schema_path)
    with schema_p.open(encoding="utf-8") as fh:
        schema: Dict[str, Any] = json.load(fh)

    id_to_schema: Dict[str, Dict[str, Any]] = {}
    sid = schema.get("$id")
    if sid:
        id_to_schema[sid] = schema

    if registry_root is not None:
        root_p = Path(registry_root)
        for p in root_p.rglob("*.json"):
            try:
                with p.open(encoding="utf-8") as fh:
                    s = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            s_id = s.get("$id")
            if s_id and s_id not in id_to_schema:
                id_to_schema[s_id] = s

    if extra_schemas:
        for p in extra_schemas:
            try:
                with Path(p).open(encoding="utf-8") as fh:
                    s = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            s_id = s.get("$id")
            if s_id:
                id_to_schema[s_id] = s  # curated wins on collision

    resources = [
        (s_id, Resource.from_contents(s, default_specification=DRAFT202012))
        for s_id, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)
    validator = Draft202012Validator(schema, registry=registry)

    if return_schema:
        return validator, schema
    return validator
