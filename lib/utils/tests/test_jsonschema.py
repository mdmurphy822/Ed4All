"""Unit tests for :mod:`lib.utils.jsonschema`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.utils.jsonschema import build_validator


# Skip whole module if jsonschema/referencing aren't installed.
pytest.importorskip("jsonschema")
pytest.importorskip("referencing")


def test_build_validator_simple_schema(tmp_path: Path) -> None:
    schema_path = tmp_path / "s.json"
    schema_path.write_text(
        json.dumps(
            {
                "$id": "https://test/example.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["x"],
            }
        ),
        encoding="utf-8",
    )
    v = build_validator(schema_path)
    assert v is not None
    assert v.is_valid({"x": 1})
    assert not v.is_valid({})


def test_build_validator_with_extra_schemas(tmp_path: Path) -> None:
    sibling_path = tmp_path / "sibling.json"
    sibling_path.write_text(
        json.dumps(
            {
                "$id": "https://test/sibling.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "string",
                "minLength": 1,
            }
        ),
        encoding="utf-8",
    )
    primary_path = tmp_path / "primary.json"
    primary_path.write_text(
        json.dumps(
            {
                "$id": "https://test/primary.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "name": {"$ref": "https://test/sibling.schema.json"},
                },
                "required": ["name"],
            }
        ),
        encoding="utf-8",
    )
    v = build_validator(primary_path, extra_schemas=[sibling_path])
    assert v is not None
    assert v.is_valid({"name": "ok"})
    assert not v.is_valid({"name": ""})


def test_build_validator_with_registry_root(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    (schemas_dir / "sibling.json").write_text(
        json.dumps(
            {
                "$id": "https://test/sibling2.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "integer",
                "minimum": 0,
            }
        ),
        encoding="utf-8",
    )
    primary = tmp_path / "primary.json"
    primary.write_text(
        json.dumps(
            {
                "$id": "https://test/primary2.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "n": {"$ref": "https://test/sibling2.schema.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    v = build_validator(primary, registry_root=schemas_dir)
    assert v is not None
    assert v.is_valid({"n": 5})
    assert not v.is_valid({"n": -1})


def test_build_validator_return_schema_tuple(tmp_path: Path) -> None:
    schema_path = tmp_path / "s.json"
    schema_dict = {
        "$id": "https://test/ex.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
    }
    schema_path.write_text(json.dumps(schema_dict), encoding="utf-8")
    result = build_validator(schema_path, return_schema=True)
    assert result is not None
    validator, schema = result
    assert schema == schema_dict
    assert validator.is_valid({})


def test_build_validator_corrupt_extra_skipped(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not-json", encoding="utf-8")
    schema_path = tmp_path / "s.json"
    schema_path.write_text(
        json.dumps(
            {
                "$id": "https://test/ok.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
            }
        ),
        encoding="utf-8",
    )
    v = build_validator(schema_path, extra_schemas=[bad])
    assert v is not None
    assert v.is_valid({})


def test_build_validator_curated_wins_on_id_collision(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    # registry_root version: requires "v": "rootversion"
    (schemas_dir / "shared.json").write_text(
        json.dumps(
            {
                "$id": "https://test/shared.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["root_only"],
            }
        ),
        encoding="utf-8",
    )
    # Curated extras version: same $id but requires "extra_only"
    extra = tmp_path / "shared_curated.json"
    extra.write_text(
        json.dumps(
            {
                "$id": "https://test/shared.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["extra_only"],
            }
        ),
        encoding="utf-8",
    )
    primary = tmp_path / "primary.json"
    primary.write_text(
        json.dumps(
            {
                "$id": "https://test/primary3.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$ref": "https://test/shared.schema.json",
            }
        ),
        encoding="utf-8",
    )
    v = build_validator(primary, registry_root=schemas_dir, extra_schemas=[extra])
    assert v is not None
    # extras-version version wins -> requires extra_only, not root_only.
    assert v.is_valid({"extra_only": 1})
    assert not v.is_valid({"root_only": 1})


def test_build_validator_no_deps_returns_none(monkeypatch, tmp_path: Path) -> None:
    """Simulate missing jsonschema deps via import error."""
    import builtins

    real_import = builtins.__import__

    def faux_import(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError("simulated missing jsonschema")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", faux_import)
    schema_path = tmp_path / "s.json"
    schema_path.write_text("{}", encoding="utf-8")
    assert build_validator(schema_path) is None
