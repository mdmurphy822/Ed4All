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

__all__ = ["build_validator", "validate_pair_record", "build_pair_validator"]


# W-D4 follow-up: cross-schema $ref registry for pair schemas.
#
# As of W-D4, ``schemas/knowledge/instruction_pair{,.strict}.schema.json`` and
# ``schemas/knowledge/preference_pair.schema.json`` reference
# ``https://ed4all.dev/ns/knowledge/v1/pair_audit_fields.schema.json#/$defs/...``
# via ``$ref`` for the per-claim-support / pair-LO-resolution / pair-objective-
# delivery embedded blocks. A bare ``jsonschema.validate(record, pair_schema)``
# call attempts to HTTP-fetch that URL and dies with ``socket.gaierror`` in
# offline environments. The fix is a Draft202012Validator built against a
# ``referencing.Registry`` populated with every ``*.schema.json`` under
# ``schemas/knowledge/`` so the cross-schema reference resolves locally.
#
# This module owns the canonical pattern; per-call sites stay small.

_PAIR_VALIDATOR_CACHE: Dict[str, Any] = {}


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


def _resolve_knowledge_dir(knowledge_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the ``schemas/knowledge`` directory.

    Defaults to the repo's ``schemas/knowledge`` (walking up from this file:
    ``lib/utils/jsonschema.py`` → repo root → ``schemas/knowledge``). Callers
    can override for fixtures or out-of-tree schema mirrors.
    """
    if knowledge_dir is not None:
        return Path(knowledge_dir)
    # lib/utils/jsonschema.py → repo_root
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "schemas" / "knowledge"


def build_pair_validator(
    pair_schema_path: Union[str, Path],
    *,
    knowledge_dir: Optional[Union[str, Path]] = None,
):
    """Build a Draft 2020-12 validator for a pair schema with the cross-schema
    registry populated from every ``*.schema.json`` under ``schemas/knowledge``.

    This is the W-D4 cross-schema-$ref-aware companion to ``build_validator``,
    specialized for the pair schemas that ``$ref`` the canonical
    ``pair_audit_fields.schema.json`` envelope. Cached per
    ``(pair_schema_path, knowledge_dir)`` key so repeated test invocations
    don't re-load the directory.

    Args:
        pair_schema_path: Path to the pair schema (instruction_pair /
            instruction_pair.strict / preference_pair).
        knowledge_dir: Override the ``schemas/knowledge`` discovery root.
            Defaults to the repo's canonical location.

    Returns:
        A ``Draft202012Validator`` instance, or ``None`` when ``jsonschema`` /
        ``referencing`` aren't installed (graceful-degrade path).
    """
    pair_p = Path(pair_schema_path).resolve()
    knowledge_p = _resolve_knowledge_dir(knowledge_dir).resolve()
    cache_key = f"{pair_p}|{knowledge_p}"
    cached = _PAIR_VALIDATOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    validator = build_validator(pair_p, registry_root=knowledge_p)
    if validator is not None:
        _PAIR_VALIDATOR_CACHE[cache_key] = validator
    return validator


def validate_pair_record(
    record: Dict[str, Any],
    pair_schema_path: Union[str, Path],
    *,
    knowledge_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Validate a single pair record against a pair schema with cross-schema
    ``$ref`` resolution.

    Drop-in replacement for ``jsonschema.validate(record, pair_schema)`` that
    survives the W-D4 cross-schema ``$ref`` to
    ``pair_audit_fields.schema.json``. Raises ``jsonschema.ValidationError``
    on shape failure (matching the bare-call contract); is a no-op when
    ``jsonschema`` / ``referencing`` aren't installed.

    Args:
        record: The pair dict to validate.
        pair_schema_path: Path to the pair schema file.
        knowledge_dir: Override the ``schemas/knowledge`` discovery root.

    Raises:
        jsonschema.ValidationError: When ``record`` fails the pair schema.
    """
    validator = build_pair_validator(
        pair_schema_path, knowledge_dir=knowledge_dir
    )
    if validator is None:
        return  # graceful-degrade: deps missing
    validator.validate(record)
