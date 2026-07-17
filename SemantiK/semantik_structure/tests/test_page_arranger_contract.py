"""Unit tests for the pure ARRANGE contract (page_arranger_contract).

No env, no HTTP, no filesystem — the contract half is fully deterministic.
Covers: alias coercion (incl. the 4 ch01 failure-taxonomy extensions), list-family
context sensitivity both ways, duplicate auto-repair + emptied-block removal,
validation (coverage / unknown-id / duplicate), the rung-3 message restating the
full legal-id set, derive_relations_v2 shapes, and extract_json tolerance.
"""
from __future__ import annotations

from semantik_structure import page_arranger_contract as c


def _units(*ids_texts):
    return [{"id": i, "text": t} for i, t in ids_texts]


# --- alias coercion --------------------------------------------------------
def test_alias_coercion_four_ch01_extensions():
    unit_by_id = {"a": {"id": "a", "text": "Foo"}}
    for raw, canon in [
        ("section", "heading"),
        ("section-header", "heading"),
        ("section_header", "heading"),
        ("table_caption", "figure_caption"),
    ]:
        arr = {"blocks": [{"ids": ["a"], "type": raw}]}
        log = c.normalize_block_types(arr, unit_by_id)
        assert arr["blocks"][0]["type"] == canon
        assert log == [{"from": raw, "to": canon, "block_idx": 0}]


def test_exercise_alias_and_case_normalization():
    unit_by_id = {"a": {"id": "a", "text": "1. do this"}}
    arr = {"blocks": [{"ids": ["a"], "type": "Exercise"}]}
    log = c.normalize_block_types(arr, unit_by_id)
    assert arr["blocks"][0]["type"] == "exercise_list"
    assert log[0]["from"] == "Exercise" and log[0]["to"] == "exercise_list"


def test_enum_case_only_coercion_logged():
    arr = {"blocks": [{"ids": ["a"], "type": "Paragraph"}]}
    log = c.normalize_block_types(arr, {"a": {"id": "a", "text": "x"}})
    assert arr["blocks"][0]["type"] == "paragraph"
    assert log == [{"from": "Paragraph", "to": "paragraph", "block_idx": 0}]


def test_list_family_context_sensitive_both_ways():
    # exercise-looking text → exercise_list
    ex = {"a": {"id": "a", "text": "1. solve x\n2. solve y\n3. solve z"}}
    arr = {"blocks": [{"ids": ["a"], "type": "list"}]}
    c.normalize_block_types(arr, ex)
    assert arr["blocks"][0]["type"] == "exercise_list"
    # prose text → paragraph
    prose = {"a": {"id": "a", "text": "This is a plain sentence of running prose."}}
    arr2 = {"blocks": [{"ids": ["a"], "type": "list"}]}
    c.normalize_block_types(arr2, prose)
    assert arr2["blocks"][0]["type"] == "paragraph"


# --- duplicate repair ------------------------------------------------------
def test_repair_duplicate_ids_keep_first_and_drop_emptied_block():
    arr = {
        "blocks": [
            {"ids": ["a", "b"], "type": "paragraph"},
            {"ids": ["b"], "type": "paragraph"},  # pure-duplicate → emptied → dropped
            {"ids": ["c"], "type": "paragraph"},
        ]
    }
    repairs = c.repair_duplicate_ids(arr)
    assert repairs == [{"id": "b", "kept_block_idx": 0, "dropped_from_block_idx": 1}]
    # emptied block removed; a/b in first, c in last
    assert [b["ids"] for b in arr["blocks"]] == [["a", "b"], ["c"]]


# --- validation ------------------------------------------------------------
def test_validate_clean_arrangement():
    units = _units(("a", "x"), ("b", "y"))
    arr = {"blocks": [{"ids": ["a", "b"], "type": "paragraph"}]}
    assert c.validate_arrangement(arr, units) == []


def test_validate_detects_missing_unknown_and_duplicate():
    units = _units(("a", "x"), ("b", "y"))
    arr = {"blocks": [{"ids": ["a", "a", "zzz"], "type": "paragraph"}]}
    problems = c.validate_arrangement(arr, units)
    assert any("unknown id 'zzz'" in p for p in problems)
    assert any("more than once" in p for p in problems)
    assert any("missing from output" in p and "b" in p for p in problems)


def test_validate_bad_type():
    units = _units(("a", "x"))
    arr = {"blocks": [{"ids": ["a"], "type": "bogus"}]}
    problems = c.validate_arrangement(arr, units)
    assert any("invalid type" in p for p in problems)


# --- rung-3 message --------------------------------------------------------
def test_rung3_message_restates_full_legal_id_set():
    legal = ["p1_b00", "p1_b01", "p1_b02"]
    msg = c.build_violation_msg(["ids missing"], legal_ids=legal)
    assert "COMPLETE set of legal unit ids" in msg
    for lid in legal:
        assert lid in msg
    # rung-2 (no legal ids) does NOT restate the set
    msg2 = c.build_violation_msg(["ids missing"], legal_ids=None)
    assert "COMPLETE set of legal unit ids" not in msg2
    assert "furniture" in msg2


# --- relations -------------------------------------------------------------
def test_derive_relations_v2_shapes():
    blocks = [
        {"ids": ["a", "b"], "type": "example"},
        {"ids": ["c"], "type": "solution"},
        {"ids": ["d"], "type": "figure_caption"},
    ]
    rels = c.derive_relations_v2(blocks)
    kinds = {r["type"] for r in rels}
    assert "same_unit" in kinds  # block 0 has >1 id
    assert "solution_of" in kinds  # solution follows example
    assert "caption_of" in kinds  # figure_caption -> neighbor


def test_last_content_unit_id_skips_furniture():
    blocks = [
        {"ids": ["a"], "type": "paragraph"},
        {"ids": ["z"], "type": "furniture"},
    ]
    assert c.last_content_unit_id(blocks, {"a", "z"}) == "a"


# --- extract_json ----------------------------------------------------------
def test_extract_json_strips_fence_and_grabs_outermost_braces():
    assert c.extract_json('```json\n{"blocks":[]}\n```') == {"blocks": []}
    assert c.extract_json('noise {"a":1} trailing') == {"a": 1}


# --- repair_mechanical_ids (SEMANTIK_ARRANGER_ID_REPAIR, P2) ----------------
def test_repair_mechanical_ids_all_three_classes_at_once():
    units = _units(("u0", "a"), ("u1", "b"), ("u2", "c"), ("u3", "d"))
    arr = {
        "blocks": [
            {"type": "paragraph", "ids": ["u0", "u0"]},   # dup u0
            {"type": "heading", "ids": ["u1", "zz"]},     # unknown zz; u2 missing
            {"type": "paragraph", "ids": ["u3"]},
        ]
    }
    log = c.repair_mechanical_ids(arr, units)
    ops = {r["op"] for r in log}
    assert ops == {"dup_drop", "unknown_drop", "missing_insert"}
    assert c.validate_arrangement(arr, units) == []  # coverage restored
    ids = [uid for blk in arr["blocks"] for uid in blk["ids"]]
    assert ids.count("u0") == 1 and "zz" not in ids
    # missing u2 inserted adjacent to its source-order neighbor (after u1)
    assert ids.index("u2") == ids.index("u1") + 1


def test_repair_mechanical_ids_missing_chains_in_source_order():
    units = _units(("u0", "a"), ("u1", "b"), ("u2", "c"))
    arr = {"blocks": [{"type": "paragraph", "ids": ["u0"]}]}  # u1, u2 both missing
    c.repair_mechanical_ids(arr, units)
    ids = [uid for blk in arr["blocks"] for uid in blk["ids"]]
    assert ids == ["u0", "u1", "u2"]  # inserted contiguously in source order
    assert c.validate_arrangement(arr, units) == []


def test_repair_mechanical_ids_missing_before_successor_when_no_predecessor():
    units = _units(("u0", "a"), ("u1", "b"))
    arr = {"blocks": [{"type": "paragraph", "ids": ["u1"]}]}  # u0 missing, no pred
    c.repair_mechanical_ids(arr, units)
    ids = [uid for blk in arr["blocks"] for uid in blk["ids"]]
    assert ids == ["u0", "u1"]  # u0 inserted BEFORE its successor u1


def test_repair_mechanical_ids_inserts_as_paragraph_never_furniture():
    units = _units(("u0", "a"), ("u1", "b"))
    arr = {"blocks": [{"type": "heading", "ids": ["u0"]}]}
    c.repair_mechanical_ids(arr, units)
    inserted = [b for b in arr["blocks"] if b["ids"] == ["u1"]][0]
    assert inserted["type"] == "paragraph"  # content-preserving default
