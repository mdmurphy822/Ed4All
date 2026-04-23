"""Regression tests for lib.ontology.teaching_roles.

Covers REC-VOC-02 (Wave 2, Worker K):
  * Loader exposes the canonical six-role enum.
  * `map_role` returns the expected role for every declared
    (component, purpose) pair in `x-component-mapping`.
  * `map_role` returns None for unmapped, partial, or empty inputs.
  * `get_valid_roles()` matches `Trainforge.align_chunks.VALID_ROLES`
    byte-for-byte — pins the schema ↔ consumer canonical set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_constants_are_six_values():
    """TEACHING_ROLES is the canonical six-tuple in schema order."""
    from lib.ontology.teaching_roles import TEACHING_ROLES

    assert TEACHING_ROLES == (
        "introduce",
        "elaborate",
        "reinforce",
        "assess",
        "transfer",
        "synthesize",
    )
    assert len(TEACHING_ROLES) == 6
    assert len(set(TEACHING_ROLES)) == 6  # no duplicates


def test_valid_roles_is_six_values():
    """get_valid_roles() returns the canonical six roles as a Set[str].

    Pins alignment with Trainforge/align_chunks.py:33 VALID_ROLES — if
    this assertion fails, one side has drifted and the schema is no
    longer authoritative.
    """
    from lib.ontology.teaching_roles import get_valid_roles

    roles = get_valid_roles()
    assert isinstance(roles, set)
    assert roles == {
        "introduce",
        "elaborate",
        "reinforce",
        "assess",
        "transfer",
        "synthesize",
    }

    # Cross-check against the Trainforge consumer constant.
    from Trainforge.align_chunks import VALID_ROLES as _VALID_ROLES

    assert roles == _VALID_ROLES, (
        "teaching_role schema drift vs Trainforge/align_chunks.py:33 VALID_ROLES"
    )


def test_get_valid_roles_returns_fresh_copy():
    """get_valid_roles() is safe to mutate — doesn't pollute the cache."""
    from lib.ontology.teaching_roles import get_valid_roles

    first = get_valid_roles()
    first.add("__scratch__")
    second = get_valid_roles()
    assert "__scratch__" not in second


def test_map_known_pairs():
    """Every declared (component, purpose) entry round-trips to the mapped role.

    Covers all three currently-emitted pairs from generate_course.py.
    """
    from lib.ontology.teaching_roles import map_role

    assert map_role("flip-card", "term-definition") == "introduce"
    assert map_role("self-check", "formative-assessment") == "assess"
    assert map_role("activity", "practice") == "transfer"


def test_mapping_covers_declared_emit_sites():
    """Read the schema directly and assert the mapper agrees with every
    declared x-component-mapping entry. Guards against the mapper going
    stale if the schema gains a new component/purpose entry that the
    mapping code doesn't pick up.
    """
    from lib.ontology.teaching_roles import map_role

    schema_path = _REPO_ROOT / "schemas" / "taxonomies" / "teaching_role.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    entries = schema.get("x-component-mapping", [])
    assert entries, "schema has no x-component-mapping entries to test"
    for entry in entries:
        assert map_role(entry["component"], entry["purpose"]) == entry["teaching_role"], (
            f"map_role disagreed with schema entry: {entry!r}"
        )


def test_map_unknown_returns_none():
    """Unmapped (component, purpose) pairs return None so callers fall back."""
    from lib.ontology.teaching_roles import map_role

    # Unknown component, valid purpose shape.
    assert map_role("accordion", "progressive-disclosure") is None
    assert map_role("timeline", "sequential-display") is None
    # Known component, wrong purpose.
    assert map_role("flip-card", "bogus-purpose") is None
    assert map_role("self-check", "practice") is None  # wrong pair
    # Fully unknown pair.
    assert map_role("bogus-component", "bogus-purpose") is None


def test_map_partial_returns_none():
    """None/empty inputs for either side return None without raising."""
    from lib.ontology.teaching_roles import map_role

    assert map_role(None, "term-definition") is None
    assert map_role("flip-card", None) is None
    assert map_role(None, None) is None
    assert map_role("", "term-definition") is None
    assert map_role("flip-card", "") is None
    assert map_role("", "") is None


def test_load_teaching_roles_returns_schema_dict():
    """load_teaching_roles() returns the raw schema as a dict."""
    from lib.ontology.teaching_roles import load_teaching_roles

    schema = load_teaching_roles()
    assert isinstance(schema, dict)
    assert "$defs" in schema
    assert "TeachingRole" in schema["$defs"]
    assert "x-component-mapping" in schema
