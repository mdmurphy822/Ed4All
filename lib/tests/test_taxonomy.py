"""Regression tests for lib.ontology.taxonomy (REC-TAX-01, Wave 2 Worker J)."""

from __future__ import annotations


def test_load_taxonomy_basic():
    """Loader returns a dict with the canonical top-level keys."""
    from lib.ontology.taxonomy import load_taxonomy

    data = load_taxonomy()
    assert isinstance(data, dict), "taxonomy must be a dict"
    assert "divisions" in data, "missing 'divisions' key"
    assert "version" in data, "missing 'version' key"
    assert isinstance(data["divisions"], dict)


def test_valid_divisions():
    """Divisions are STEM and ARTS (both present)."""
    from lib.ontology.taxonomy import get_valid_divisions

    divs = get_valid_divisions()
    assert "STEM" in divs, f"STEM missing from {divs}"
    assert "ARTS" in divs, f"ARTS missing from {divs}"
    # Exactly these two for the current taxonomy; if a third is added the
    # assertion below is the reminder to update this test consciously.
    assert divs == {"STEM", "ARTS"}, f"unexpected divisions: {divs}"


def test_get_valid_domains_stem_contains_cs():
    """Spot-check: computer-science is a STEM domain."""
    from lib.ontology.taxonomy import get_valid_domains

    domains = get_valid_domains("STEM")
    assert "computer-science" in domains
    assert "mathematics" in domains
    assert "biology" in domains


def test_get_valid_domains_arts_contains_design():
    """Spot-check: design is an ARTS domain."""
    from lib.ontology.taxonomy import get_valid_domains

    domains = get_valid_domains("ARTS")
    assert "design" in domains
    assert "history" in domains


def test_get_valid_domains_bad_division_returns_empty():
    """Unknown division returns the empty set (not an exception)."""
    from lib.ontology.taxonomy import get_valid_domains

    assert get_valid_domains("BOGUS") == set()
    assert get_valid_domains("") == set()


def test_get_valid_subdomains_spot_check():
    """Spot-check: software-engineering under STEM/computer-science."""
    from lib.ontology.taxonomy import get_valid_subdomains

    subs = get_valid_subdomains("STEM", "computer-science")
    assert "software-engineering" in subs
    assert "algorithms" in subs


def test_get_valid_subdomains_bad_path_returns_empty():
    from lib.ontology.taxonomy import get_valid_subdomains

    assert get_valid_subdomains("STEM", "bogus-domain") == set()
    assert get_valid_subdomains("BOGUS", "computer-science") == set()


def test_get_valid_topics_spot_check():
    """Topics are leaf strings under subdomains."""
    from lib.ontology.taxonomy import get_valid_topics

    topics = get_valid_topics("STEM", "computer-science", "software-engineering")
    assert "design-patterns" in topics
    assert "testing" in topics


def test_validate_classification_valid():
    """Well-formed classification returns empty error list."""
    from lib.ontology.taxonomy import validate_classification

    errors = validate_classification({
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": ["software-engineering"],
        "topics": [],
    })
    assert errors == [], f"expected no errors, got: {errors}"


def test_validate_classification_valid_with_topics():
    """Classification with valid topics passes."""
    from lib.ontology.taxonomy import validate_classification

    errors = validate_classification({
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": ["software-engineering"],
        "topics": ["design-patterns", "testing"],
    })
    assert errors == [], f"expected no errors, got: {errors}"


def test_validate_classification_invalid_division():
    """Unknown division returns error list."""
    from lib.ontology.taxonomy import validate_classification

    errors = validate_classification({
        "division": "BOGUS",
        "primary_domain": "computer-science",
        "subdomains": [],
        "topics": [],
    })
    assert errors, "expected errors for bogus division"
    assert any("division" in e.lower() for e in errors), errors


def test_validate_classification_wrong_domain_for_division():
    """STEM-only domain under ARTS returns error."""
    from lib.ontology.taxonomy import validate_classification

    # computer-science is STEM, so it must not be valid under ARTS
    errors = validate_classification({
        "division": "ARTS",
        "primary_domain": "computer-science",
        "subdomains": [],
        "topics": [],
    })
    assert errors, "expected errors when domain belongs to wrong division"
    assert any("primary_domain" in e for e in errors), errors


def test_validate_classification_bad_subdomain():
    """Subdomain not under declared domain returns error."""
    from lib.ontology.taxonomy import validate_classification

    errors = validate_classification({
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": ["bogus-subdomain"],
        "topics": [],
    })
    assert errors, "expected errors for bogus subdomain"
    assert any("subdomain" in e.lower() for e in errors), errors


def test_validate_classification_bad_topic():
    """Topic not in any allowed subdomain returns error."""
    from lib.ontology.taxonomy import validate_classification

    errors = validate_classification({
        "division": "STEM",
        "primary_domain": "computer-science",
        "subdomains": ["software-engineering"],
        "topics": ["bogus-topic-slug"],
    })
    assert errors, "expected errors for bogus topic"
    assert any("topic" in e.lower() for e in errors), errors


def test_validate_classification_empty():
    """Empty dict surfaces the missing-field error."""
    from lib.ontology.taxonomy import validate_classification

    errors = validate_classification({})
    assert errors, "expected errors for empty classification"


def test_validate_classification_none():
    """None classification returns an error instead of raising."""
    from lib.ontology.taxonomy import validate_classification

    errors = validate_classification(None)
    assert errors, "expected errors for None classification"


def test_validate_classification_missing_primary_domain():
    """Division without primary_domain returns error."""
    from lib.ontology.taxonomy import validate_classification

    errors = validate_classification({
        "division": "STEM",
        "subdomains": [],
    })
    assert errors
    assert any("primary_domain" in e for e in errors), errors


def test_defensive_copy_semantics():
    """Mutating returned sets does not pollute the cache."""
    from lib.ontology.taxonomy import get_valid_divisions, get_valid_domains

    a = get_valid_divisions()
    a.add("__mutated__")
    b = get_valid_divisions()
    assert "__mutated__" not in b

    c = get_valid_domains("STEM")
    c.add("__x__")
    d = get_valid_domains("STEM")
    assert "__x__" not in d
