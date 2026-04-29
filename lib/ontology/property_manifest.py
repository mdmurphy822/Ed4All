"""Wave 109 — Phase C: property-manifest loader.

A property manifest declares the canonical surface forms (URIs / CURIEs /
text patterns) a course-FAMILY's SLM adapters must teach. The synthesis +
eval surfaces both consume the same manifest so coverage gates and
per-property eval line up. A single manifest applies to every course in
the family (e.g. ``property_manifest.rdf_shacl.yaml`` covers every
``rdf-shacl-*`` course).

Manifests live at ``schemas/training/property_manifest.<family>.yaml``
and are validated against
``schemas/training/property_manifest.schema.json`` on load. The manifest
declares its own ``family`` field; the loader resolves it from a course
slug via ``_family_slug`` (``rdf-shacl-551-2`` -> ``rdf_shacl``).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOT = PROJECT_ROOT / "schemas" / "training"
DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "training" / "property_manifest.schema.json"
)


@dataclass
class PropertyEntry:
    id: str
    uri: str
    curie: str
    label: str
    surface_forms: List[str]
    min_pairs: int
    min_accuracy: float = 0.40

    def matches(self, text: str) -> bool:
        """True when ``text`` contains any declared surface form."""
        if not text:
            return False
        return any(sf in text for sf in self.surface_forms)


@dataclass
class PropertyManifest:
    family: str
    properties: List[PropertyEntry]
    description: Optional[str] = None

    @property
    def by_id(self) -> Dict[str, PropertyEntry]:
        return {p.id: p for p in self.properties}

    def detect_surface_forms(self, text: str) -> List[str]:
        """Wave 120: return the deduplicated list of declared surface
        forms that appear in ``text``. Used by the synthesis pipeline to
        instruct the paraphrase provider to preserve technical CURIEs
        (``sh:NodeShape``, ``rdfs:subClassOf``, etc.) verbatim — without
        this, 14B-class local models silently rewrite them as prose,
        which is what bit Wave 119's 295-chunk run (5/6 properties below
        floor despite 295 instruction pairs emitted)."""
        if not text:
            return []
        seen: Dict[str, None] = {}
        for prop in self.properties:
            for sf in prop.surface_forms:
                if sf and sf in text and sf not in seen:
                    seen[sf] = None
        return list(seen.keys())


def _family_slug(course_slug: str) -> str:
    """Pick the corpus family from a course slug.

    ``rdf-shacl-551-2`` -> ``rdf_shacl`` (first two hyphen-separated
    tokens, joined by underscore). Single-token slugs return the
    slug unchanged.
    """
    parts = course_slug.split("-")
    if len(parts) < 2:
        return course_slug
    return f"{parts[0]}_{parts[1]}"


def load_property_manifest(
    course_slug: str,
    *,
    search_root: Optional[Path] = None,
    schema_path: Optional[Path] = None,
) -> PropertyManifest:
    """Load + schema-validate a property manifest for a course.

    Resolution: looks for
    ``<search_root>/property_manifest.<family>.yaml`` where ``<family>``
    is derived from the course slug (``rdf-shacl-551-2`` ->
    ``rdf_shacl``). Falls back to the literal slug when the family
    form isn't present.
    """
    root = Path(search_root) if search_root is not None else DEFAULT_SEARCH_ROOT
    family = _family_slug(course_slug)
    candidates = [
        root / f"property_manifest.{family}.yaml",
        root / f"property_manifest.{course_slug}.yaml",
    ]
    manifest_path: Optional[Path] = None
    for c in candidates:
        if c.exists():
            manifest_path = c
            break
    if manifest_path is None:
        raise FileNotFoundError(
            f"No property manifest for course '{course_slug}'. "
            f"Looked in: {[str(c) for c in candidates]}"
        )

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    schema_p = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    if schema_p.exists():
        try:
            import jsonschema
            schema = json.loads(schema_p.read_text(encoding="utf-8"))
            jsonschema.validate(payload, schema)
        except ImportError:
            logger.warning(
                "jsonschema not installed; skipping property manifest validation"
            )

    properties = [
        PropertyEntry(
            id=p["id"],
            uri=p["uri"],
            curie=p["curie"],
            label=p["label"],
            surface_forms=list(p["surface_forms"]),
            min_pairs=int(p["min_pairs"]),
            min_accuracy=float(p.get("min_accuracy", 0.40)),
        )
        for p in payload["properties"]
    ]
    return PropertyManifest(
        family=str(payload["family"]),
        properties=properties,
        description=payload.get("description"),
    )


__all__ = ["PropertyEntry", "PropertyManifest", "load_property_manifest"]
