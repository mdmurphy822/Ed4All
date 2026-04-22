"""Regression tests for lib.ontology.bloom loader and migrated callsites.

Covers REC-BL-01 (Wave 1.2, Worker H):
  * Loader exposes canonical shapes (set, list, object, flat set).
  * Every migrated callsite exposes the apply-level verb set of the
    canonical taxonomy.
  * The richest callsite (bloom_taxonomy_mapper) preserves its local
    Dict[BloomLevel, List[BloomVerb]] shape after migration.
  * detect_bloom_level returns the documented (level, verb) tuples.
  * LibV2 vendored copy is byte-identical to the authoritative schema.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_by_path(module_name: str, path: Path):
    """Load a Python module from an absolute path (for scripts that are not
    normally imported as packages)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_shapes():
    """Loader exposes three shapes over the same data."""
    from lib.ontology.bloom import (
        BLOOM_LEVELS,
        BloomVerb,
        get_all_verbs,
        get_verb_objects,
        get_verbs,
        get_verbs_list,
    )

    assert BLOOM_LEVELS == (
        "remember",
        "understand",
        "apply",
        "analyze",
        "evaluate",
        "create",
    )

    sets = get_verbs()
    lists = get_verbs_list()
    objs = get_verb_objects()

    for level in BLOOM_LEVELS:
        verb_set = sets[level]
        verb_list = lists[level]
        verb_objs = objs[level]

        assert isinstance(verb_set, set), f"{level}: expected set"
        assert isinstance(verb_list, list), f"{level}: expected list"
        assert all(isinstance(o, BloomVerb) for o in verb_objs), (
            f"{level}: expected BloomVerb instances"
        )
        assert verb_set == set(verb_list), (
            f"{level}: set/list shape mismatch"
        )
        assert {o.verb for o in verb_objs} == verb_set, (
            f"{level}: objects/set shape mismatch"
        )

    # Canonical schema has 60 unique verbs.
    flat = get_all_verbs()
    assert len(flat) == 60, f"Expected 60 canonical verbs, got {len(flat)}"


def test_defensive_copy_semantics():
    """Callers must be able to mutate returned structures without polluting cache."""
    from lib.ontology.bloom import get_verb_objects, get_verbs, get_verbs_list

    sets_a = get_verbs()
    sets_a["remember"].add("__mutated__")
    sets_b = get_verbs()
    assert "__mutated__" not in sets_b["remember"]

    lists_a = get_verbs_list()
    lists_a["remember"].append("__mutated__")
    lists_b = get_verbs_list()
    assert "__mutated__" not in lists_b["remember"]

    objs_a = get_verb_objects()
    objs_a["remember"].clear()
    objs_b = get_verb_objects()
    assert len(objs_b["remember"]) > 0


def test_migrated_sites_match_canonical_apply_level():
    """Every migrated callsite's apply-level verb set equals canonical."""
    from lib.ontology.bloom import get_verbs_list

    canonical_apply = set(get_verbs_list()["apply"])

    # 1. lib.validators.bloom — Set[str] shape
    from lib.validators.bloom import BLOOM_VERBS as validators_bloom
    assert set(validators_bloom["apply"]) == canonical_apply, (
        "lib.validators.bloom.apply drift"
    )

    # 2. Trainforge.parsers.html_content_parser — List[str] shape, class attr
    from Trainforge.parsers.html_content_parser import HTMLContentParser
    assert set(HTMLContentParser.BLOOM_VERBS["apply"]) == canonical_apply, (
        "html_content_parser.apply drift"
    )

    # 3. Courseforge.scripts.generate_course — List[str] shape, module attr.
    # Path-loaded because the script isn't a package.
    gc_path = _REPO_ROOT / "Courseforge" / "scripts" / "generate_course.py"
    gc = _load_by_path("generate_course_wh_test", gc_path)
    assert set(gc.BLOOM_VERBS["apply"]) == canonical_apply, (
        "generate_course.apply drift"
    )

    # 4. Trainforge.generators.assessment_generator — nested; verbs only
    from Trainforge.generators.assessment_generator import (
        BLOOM_LEVELS as ag_levels,
    )
    assert set(ag_levels["apply"]["verbs"]) == canonical_apply, (
        "assessment_generator.apply drift"
    )


# Wave 28f: test_bloom_taxonomy_mapper_shape_preserved removed alongside the
# textbook-objective-generator/ subtree deletion. The canonical bloom module
# + its shape contracts are covered by test_canonical_shapes and
# test_migrated_sites_match_canonical_apply_level above.


def test_libv2_vendor_loader_matches_canonical():
    """LibV2's internal loader returns the same verbs as the canonical loader."""
    # Add LibV2/tools to path so the internal loader imports cleanly.
    libv2_tools = str(_REPO_ROOT / "LibV2" / "tools")
    if libv2_tools not in sys.path:
        sys.path.insert(0, libv2_tools)
    from libv2._bloom_verbs import get_verbs_list as libv2_get_verbs_list

    from lib.ontology.bloom import get_verbs_list as canonical_get_verbs_list

    canonical = canonical_get_verbs_list()
    vendored = libv2_get_verbs_list()

    assert set(vendored.keys()) == set(canonical.keys())
    for level in canonical:
        assert set(vendored[level]) == set(canonical[level]), (
            f"LibV2 vendored verbs drift at level={level}"
        )


@pytest.mark.parametrize(
    "text,expected_level,expected_verb",
    [
        ("design a system to handle high load", "create", "design"),
        ("list the steps of photosynthesis", "remember", "list"),
        ("Evaluate the effectiveness of the plan", "evaluate", "evaluate"),
        ("Apply the formula to each scenario", "apply", "apply"),
        ("no verbs here at all whatsoever", None, None),
        ("", None, None),
    ],
)
def test_detect_bloom_level(text, expected_level, expected_verb):
    from lib.ontology.bloom import detect_bloom_level

    level, verb = detect_bloom_level(text)
    assert level == expected_level, (
        f"detect_bloom_level({text!r}) level: expected {expected_level}, got {level}"
    )
    assert verb == expected_verb, (
        f"detect_bloom_level({text!r}) verb: expected {expected_verb}, got {verb}"
    )


def test_libv2_vendor_hash_sync():
    """LibV2/vendor/bloom_verbs.json must be byte-identical to the source."""
    auth = _REPO_ROOT / "schemas" / "taxonomies" / "bloom_verbs.json"
    vendored = _REPO_ROOT / "LibV2" / "vendor" / "bloom_verbs.json"

    assert auth.exists(), f"Authoritative copy missing: {auth}"
    assert vendored.exists(), f"Vendored copy missing: {vendored}"

    h_auth = hashlib.sha256(auth.read_bytes()).hexdigest()
    h_vendored = hashlib.sha256(vendored.read_bytes()).hexdigest()

    assert h_auth == h_vendored, (
        f"Hash drift:\n  auth:     {h_auth}\n  vendored: {h_vendored}"
    )
