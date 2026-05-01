"""Wave 137b - tests for the family-clustered backfill ordering.

Nine tests pin the clustered-sort contract:

1. ``test_alphabetical_bypasses_clustering`` - by="alphabetical" still
   returns ascending CURIE order (preserves Wave 136d).
2. ``test_no_family_map_falls_back_to_flat`` - family_map=None falls
   back to flat freq-desc + alpha tie-break (Wave 136d behavior).
3. ``test_clusters_grouped_together`` - all CURIEs from one family
   appear contiguously in the output (no interleaving).
4. ``test_family_buckets_ordered_by_aggregate_frequency`` - the
   highest-aggregate-freq family appears first.
5. ``test_within_family_ordered_by_individual_frequency`` - within a
   family bucket, CURIEs are freq-desc.
6. ``test_singletons_appended_at_end`` - singletons are emitted after
   every family bucket.
7. ``test_singletons_ordered_by_frequency`` - singletons are
   freq-desc among themselves.
8. ``test_family_filter_restricts_to_one_cluster`` - family_filter=
   "cardinality" returns ONLY cardinality CURIEs.
9. ``test_family_filter_singletons_returns_only_singletons`` -
   family_filter="singletons" returns ONLY singleton CURIEs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.family_map import FamilyMap  # noqa: E402
from Trainforge.scripts import backfill_form_data as cli  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_family_map() -> FamilyMap:
    families = {
        "cardinality": ["sh:minCount", "sh:maxCount"],
        "domain_range": ["rdfs:domain", "rdfs:range"],
        "shape_kinds": ["sh:NodeShape", "sh:PropertyShape"],
    }
    singletons = ["sh:datatype", "sh:nodeKind", "sh:path"]
    family_of: Dict[str, str] = {}
    for fam_name, curies in families.items():
        for c in curies:
            family_of[c] = fam_name
    for c in singletons:
        family_of[c] = "<singleton>"
    return FamilyMap(
        family="rdf_shacl",
        families=families,
        singletons=singletons,
        family_of=family_of,
    )


def _all_curies(fm: FamilyMap) -> List[str]:
    out: List[str] = []
    for curies in fm.families.values():
        out.extend(curies)
    out.extend(fm.singletons)
    return out


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_alphabetical_bypasses_clustering():
    """by='alphabetical' returns plain ascending CURIE order regardless of map."""
    fm = _make_family_map()
    curies = _all_curies(fm)
    counts = {c: 0 for c in curies}
    ordered = cli._sort_targets_clustered(
        curies, counts, family_map=fm, by="alphabetical",
    )
    visited = [c for c, _ in ordered]
    assert visited == sorted(curies), (
        f"alphabetical mode should sort lexicographically; got {visited}"
    )


def test_no_family_map_falls_back_to_flat():
    """family_map=None preserves Wave 136d flat freq-desc + alpha tie-break."""
    curies = ["test:A", "test:B", "test:C", "test:D"]
    counts = {"test:A": 5, "test:B": 5, "test:C": 1, "test:D": 10}
    ordered = cli._sort_targets_clustered(
        curies, counts, family_map=None, by="frequency",
    )
    visited = [c for c, _ in ordered]
    # D=10, A/B=5 (alpha tie), C=1
    assert visited == ["test:D", "test:A", "test:B", "test:C"]


def test_clusters_grouped_together():
    """Every CURIE from one family appears contiguously."""
    fm = _make_family_map()
    curies = _all_curies(fm)
    # Equal frequencies so we test the clustering, not the freq sort.
    counts = {c: 1 for c in curies}
    ordered = cli._sort_targets_clustered(
        curies, counts, family_map=fm, by="frequency",
    )
    visited = [c for c, _ in ordered]
    # For each family, the indices of its CURIEs must be contiguous.
    for fam_name, family_curies in fm.families.items():
        indices = sorted(visited.index(c) for c in family_curies)
        # Indices should be a run of consecutive integers.
        assert indices == list(range(indices[0], indices[0] + len(indices))), (
            f"family '{fam_name}' CURIEs interleaved with another family; "
            f"got visit order {visited}"
        )


def test_family_buckets_ordered_by_aggregate_frequency():
    """Family with highest aggregate freq appears first."""
    fm = _make_family_map()
    counts = {
        # cardinality: aggregate 100
        "sh:minCount": 50, "sh:maxCount": 50,
        # domain_range: aggregate 30
        "rdfs:domain": 15, "rdfs:range": 15,
        # shape_kinds: aggregate 200 (highest)
        "sh:NodeShape": 100, "sh:PropertyShape": 100,
        # singletons
        "sh:datatype": 5, "sh:nodeKind": 1, "sh:path": 0,
    }
    ordered = cli._sort_targets_clustered(
        _all_curies(fm), counts, family_map=fm, by="frequency",
    )
    visited = [c for c, _ in ordered]
    # First two CURIEs should be from shape_kinds (200 agg).
    assert set(visited[0:2]) == {"sh:NodeShape", "sh:PropertyShape"}
    # Next two: cardinality (100 agg).
    assert set(visited[2:4]) == {"sh:minCount", "sh:maxCount"}
    # Then domain_range (30 agg).
    assert set(visited[4:6]) == {"rdfs:domain", "rdfs:range"}


def test_within_family_ordered_by_individual_frequency():
    """Inside a cluster bucket, CURIEs are freq-desc with alpha tie-break."""
    fm = _make_family_map()
    counts = {
        "sh:minCount": 10, "sh:maxCount": 50,  # maxCount wins
        "rdfs:domain": 20, "rdfs:range": 20,   # tie -> alpha (domain < range)
        "sh:NodeShape": 5, "sh:PropertyShape": 5,
        "sh:datatype": 0, "sh:nodeKind": 0, "sh:path": 0,
    }
    ordered = cli._sort_targets_clustered(
        _all_curies(fm), counts, family_map=fm, by="frequency",
    )
    visited = [c for c, _ in ordered]
    # cardinality block (agg 60): maxCount before minCount.
    cardinality_idx = visited.index("sh:maxCount")
    assert visited[cardinality_idx + 1] == "sh:minCount"
    # domain_range block (agg 40): domain before range (alpha tie).
    domain_idx = visited.index("rdfs:domain")
    assert visited[domain_idx + 1] == "rdfs:range"


def test_singletons_appended_at_end():
    """Singletons land after every family bucket regardless of freq."""
    fm = _make_family_map()
    counts = {
        # Family CURIEs with low frequencies.
        "sh:minCount": 1, "sh:maxCount": 1,
        "rdfs:domain": 1, "rdfs:range": 1,
        "sh:NodeShape": 1, "sh:PropertyShape": 1,
        # Singletons with very high frequencies — still emitted last.
        "sh:datatype": 1000, "sh:nodeKind": 999, "sh:path": 998,
    }
    ordered = cli._sort_targets_clustered(
        _all_curies(fm), counts, family_map=fm, by="frequency",
    )
    visited = [c for c, _ in ordered]
    # The last 3 entries should be exactly the singletons.
    assert set(visited[-3:]) == set(fm.singletons), (
        f"singletons not at end; visit order: {visited}"
    )


def test_singletons_ordered_by_frequency():
    """Among singletons, the order is freq-desc with alpha tie-break."""
    fm = _make_family_map()
    counts = {
        "sh:minCount": 0, "sh:maxCount": 0,
        "rdfs:domain": 0, "rdfs:range": 0,
        "sh:NodeShape": 0, "sh:PropertyShape": 0,
        # Distinct frequencies on the singletons.
        "sh:datatype": 5, "sh:nodeKind": 50, "sh:path": 100,
    }
    ordered = cli._sort_targets_clustered(
        _all_curies(fm), counts, family_map=fm, by="frequency",
    )
    visited = [c for c, _ in ordered]
    # Singletons are the last 3 entries.
    assert visited[-3:] == ["sh:path", "sh:nodeKind", "sh:datatype"]


def test_family_filter_restricts_to_one_cluster():
    """family_filter='cardinality' returns ONLY cardinality CURIEs."""
    fm = _make_family_map()
    counts = {c: 1 for c in _all_curies(fm)}
    ordered = cli._sort_targets_clustered(
        _all_curies(fm),
        counts,
        family_map=fm,
        by="frequency",
        family_filter="cardinality",
    )
    visited = [c for c, _ in ordered]
    assert set(visited) == set(fm.families["cardinality"]), (
        f"family_filter='cardinality' should yield only cardinality "
        f"CURIEs; got {visited}"
    )


def test_family_filter_singletons_returns_only_singletons():
    """family_filter='singletons' yields ONLY singleton CURIEs."""
    fm = _make_family_map()
    counts = {c: 1 for c in _all_curies(fm)}
    ordered = cli._sort_targets_clustered(
        _all_curies(fm),
        counts,
        family_map=fm,
        by="frequency",
        family_filter="singletons",
    )
    visited = [c for c, _ in ordered]
    assert set(visited) == set(fm.singletons), (
        f"family_filter='singletons' should yield only singletons; "
        f"got {visited}"
    )
