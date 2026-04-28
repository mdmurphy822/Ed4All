"""Wave 109 / Phase C: property-manifest loader tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from lib.ontology.property_manifest import (
    PropertyEntry,
    PropertyManifest,
    load_property_manifest,
)


def test_load_rdf_shacl_manifest_returns_six_properties() -> None:
    manifest = load_property_manifest("rdf-shacl-551-2")
    assert isinstance(manifest, PropertyManifest)
    assert manifest.family == "rdf_shacl"
    assert len(manifest.properties) == 6
    ids = {p.id for p in manifest.properties}
    assert ids == {
        "sh_datatype", "sh_class", "sh_nodeshape",
        "sh_propertyshape", "rdfs_subclassof", "owl_sameas",
    }


def test_property_entry_match_text_against_surface_forms() -> None:
    manifest = load_property_manifest("rdf-shacl-551-2")
    sh_dt = next(p for p in manifest.properties if p.id == "sh_datatype")
    assert sh_dt.matches("ex:Shape sh:datatype xsd:string .") is True
    assert sh_dt.matches("ex:Shape sh:class :Person .") is False


def test_unknown_course_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_property_manifest("does-not-exist", search_root=tmp_path)


def test_manifest_by_id_lookup() -> None:
    manifest = load_property_manifest("rdf-shacl-551-2")
    by_id = manifest.by_id
    assert "sh_datatype" in by_id
    assert by_id["sh_datatype"].curie == "sh:datatype"
