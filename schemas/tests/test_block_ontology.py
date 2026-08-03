"""Consistency tests for the universal block-label ontology data contract.

The ontology is three layers of data under ``schemas/taxonomies/``:

* L1 ``block_kinds.json``       — closed structural kinds (DocLayNet-mapped).
* L2 ``genre_profile_*.json``   — functional roles that attach to L1 kinds.
* L3 ``*_lexicon.json``         — publisher marker -> L2 role tables.

plus ``block_relations.json`` (structural + profile block relations).

These tests enforce the mechanically checkable parts of
``docs/architecture/block-ontology.md`` under "Governance" and
"Invariants any change must preserve":

1. Every ontology file loads via ``lib/ontology/taxonomy.py::load_taxonomy``.
2. Every L2 role ``attaches_to`` >=1 L1 kind (and only valid L1 kinds).
3. Every L1 kind declares a DocLayNet mapping key (value may be ``null``).
4. Lexicon files carry only marker->role data (and roles resolve to a real
   role of the lexicon's declared profile).
5. Every profile relation names >=1 real L2 role endpoint.
"""
from __future__ import annotations

import pytest

from lib.ontology.taxonomy import load_taxonomy

# --- ontology file inventory (bare stems under schemas/taxonomies/) ---------

L1_FILE = "block_kinds"
RELATIONS_FILE = "block_relations"
PROFILE_FILES = {
    "instructional": "genre_profile_instructional",
    "encyclopedic": "genre_profile_encyclopedic",
    "forms": "genre_profile_forms",
    "legal_regulatory": "genre_profile_legal_regulatory",
    "scholarly": "genre_profile_scholarly",
    "literary": "genre_profile_literary",
    "statistical": "genre_profile_statistical",
    "technical_manual": "genre_profile_technical_manual",
}
LEXICON_FILES = [
    "openstax_lexicon",
    "generic_instructional_lexicon",
    "federal_register_lexicon",
    "ansi_z535_lexicon",
]

# --- L1 snapshot guard: changes to the closed vocabulary require review -----
# Any accidental addition/removal of an L1 kind must fail loudly — L1 is a
# closed, schema-review-gated vocabulary (block_kinds.json governance).
EXPECTED_L1_KINDS = frozenset({
    "heading",
    "paragraph",
    "list_item",
    "table",
    "figure",
    "chart",
    "caption",
    "math_block",
    "code_block",
    "blockquote",
    "aside",
    "footnote",
    "form_field",
    "title_block",
    "separator",
    "furniture",
})

ALL_FILES = [L1_FILE, RELATIONS_FILE, *PROFILE_FILES.values(), *LEXICON_FILES]

_LEXICON_MARKER_KEYS = {"marker", "role", "notes"}


# --- helpers ---------------------------------------------------------------


def _l1_kinds() -> set:
    doc = load_taxonomy(L1_FILE)
    return {entry["kind"] for entry in doc["x-block-kinds"]}


# --- invariant 1: every file loads -----------------------------------------


@pytest.mark.parametrize("name", ALL_FILES)
def test_ontology_file_loads(name):
    doc = load_taxonomy(name)
    assert isinstance(doc, dict) and doc, f"{name}.json loaded empty"


# --- invariant 3: every L1 kind declares a DocLayNet mapping key ------------


def test_block_kinds_declare_doclaynet_key():
    doc = load_taxonomy(L1_FILE)
    entries = doc["x-block-kinds"]
    assert entries, "block_kinds.json has no x-block-kinds entries"
    # closed enum must match the detailed entry list exactly
    enum = set(doc["$defs"]["BlockKind"]["enum"])
    assert {e["kind"] for e in entries} == enum
    for entry in entries:
        # key MUST be present; value may be null (an ours-only kind with no
        # DocLayNet parent) — the honest-gap contract.
        assert "doclaynet" in entry, f"{entry['kind']} missing doclaynet key"


# --- L1 snapshot guard: pin the schema-review-gated closed vocabulary -------


def test_l1_kind_enum_matches_expected_snapshot():
    # Pin the exact closed set so an accidental kind add/remove fails loudly.
    enum = frozenset(load_taxonomy(L1_FILE)["$defs"]["BlockKind"]["enum"])
    assert enum == EXPECTED_L1_KINDS, (
        "block_kinds.json BlockKind enum drifted from the owner-gated "
        f"snapshot; added={sorted(enum - EXPECTED_L1_KINDS)} "
        f"removed={sorted(EXPECTED_L1_KINDS - enum)}"
    )
    assert len(EXPECTED_L1_KINDS) == 16


# --- invariant 2: every L2 role attaches_to >=1 valid L1 kind ---------------


@pytest.mark.parametrize("profile_id,name", sorted(PROFILE_FILES.items()))
def test_roles_attach_to_valid_l1_kinds(profile_id, name):
    kinds = _l1_kinds()
    doc = load_taxonomy(name)
    assert doc["profile"]["id"] == profile_id
    roles = doc["x-roles"]
    assert roles, f"{name}.json declares no roles"
    for role in roles:
        attaches = role.get("attaches_to") or []
        assert attaches, f"{profile_id}/{role['role']} attaches_to nothing"
        unknown = set(attaches) - kinds
        assert not unknown, (
            f"{profile_id}/{role['role']} attaches_to unknown L1 kind(s): "
            f"{sorted(unknown)}"
        )


# --- invariant 4: lexicons carry only marker->role data ---------------------


@pytest.mark.parametrize("name", LEXICON_FILES)
def test_lexicon_is_marker_to_role_only(name):
    doc = load_taxonomy(name)
    profile_id = doc["lexicon"]["profile"]
    profile_roles = {
        r["role"] for r in load_taxonomy(PROFILE_FILES[profile_id])["x-roles"]
    }
    markers = doc["x-markers"]
    assert markers, f"{name}.json has no markers"
    for entry in markers:
        extra = set(entry) - _LEXICON_MARKER_KEYS
        assert not extra, f"{name} marker carries non-marker/role keys: {extra}"
        assert entry.get("marker"), f"{name} marker missing 'marker'"
        assert entry.get("role") in profile_roles, (
            f"{name} marker '{entry.get('marker')}' -> unknown "
            f"{profile_id} role '{entry.get('role')}'"
        )


# --- invariant 5: every profile relation names >=1 real L2 role -------------


def test_profile_relations_name_real_roles():
    all_roles = set()
    for name in PROFILE_FILES.values():
        all_roles |= {r["role"] for r in load_taxonomy(name)["x-roles"]}
    # forms field_label/field_value pairing also references answer roles that
    # live on the instructional side via block_relations role_endpoints; the
    # union above already covers every profile.
    relations = load_taxonomy(RELATIONS_FILE)["x-relations"]
    profile_rels = [r for r in relations if r["family"] == "profile"]
    assert profile_rels, "no profile relations declared"
    for rel in profile_rels:
        endpoints = rel.get("role_endpoints") or []
        assert endpoints, f"profile relation {rel['relation']} names no roles"
        assert set(endpoints) & all_roles, (
            f"profile relation {rel['relation']} endpoints {endpoints} "
            "match no declared L2 role"
        )


# --- relation enum <-> x-relations parity (mirror of the L1 kind check) ------


def test_relation_enums_match_entries_by_family():
    doc = load_taxonomy(RELATIONS_FILE)
    relations = doc["x-relations"]
    assert relations, "block_relations.json has no x-relations entries"
    by_family = {
        "structural": {"StructuralRelation"},
        "profile": {"ProfileRelation"},
    }
    for family, def_names in (
        ("structural", "StructuralRelation"),
        ("profile", "ProfileRelation"),
    ):
        enum = set(doc["$defs"][def_names]["enum"])
        entry_relations = {
            r["relation"] for r in relations if r["family"] == family
        }
        assert entry_relations == enum, (
            f"{def_names} enum drifted from family=={family!r} x-relations "
            f"entries; enum_only={sorted(enum - entry_relations)} "
            f"entries_only={sorted(entry_relations - enum)}"
        )
    # every relation belongs to exactly one of the two enum families
    assert {r["family"] for r in relations} == set(by_family)
