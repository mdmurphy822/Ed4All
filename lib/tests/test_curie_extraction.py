"""Wave 135b — tests for the hoisted ``lib.ontology.curie_extraction``
module.

The module owns the open-prefix CURIE regex + URL-scheme exclusion list
that Wave 131 introduced. Tests here mirror the slice of behavior
``test_curie_preservation_validator.py`` already exercises via the
validator wrapper, plus dedicated coverage for the URL-scheme
exclusion path. Wave 135b's force-injection in
``Trainforge/generators/instruction_factory.py`` /
``preference_factory.py`` consumes ``extract_curies`` directly, so a
silent regex regression here would shift the trained adapter's CURIE
distribution — the unit coverage is load-bearing.
"""
from __future__ import annotations

from lib.ontology.curie_extraction import (
    CURIE_REGEX,
    EXCLUDED_PREFIXES,
    extract_curies,
)


def test_extracts_canonical_eight_prefixes() -> None:
    """The 8 prefixes the original Wave 130b allowlist covered all
    extract cleanly through the open-prefix regex."""
    text = (
        "sh:NodeShape constrains rdf:type assertions; rdfs:subClassOf "
        "and owl:sameAs interact with xsd:string literals; skos:Concept "
        "and dcterms:title are common in foaf:Agent profiles."
    )
    found = extract_curies(text)
    expected = {
        "sh:NodeShape",
        "rdf:type",
        "rdfs:subClassOf",
        "owl:sameAs",
        "xsd:string",
        "skos:Concept",
        "dcterms:title",
        "foaf:Agent",
    }
    assert expected.issubset(found)


def test_extracts_new_prefixes_open_regex_admits() -> None:
    """Wave 131 open-prefix detection — prefixes outside the original
    8-prefix allowlist (prov, dcat, geo, vcard, void, ex) extract
    because the regex is open and only URL schemes are excluded."""
    text = (
        "prov:wasDerivedFrom and dcat:Dataset show up in geo:lat / "
        "geo:long pairs; vcard:hasEmail and void:Dataset round out "
        "the W3C-vocabulary surface; ex:WorkedExample is the convention."
    )
    found = extract_curies(text)
    for curie in [
        "prov:wasDerivedFrom",
        "dcat:Dataset",
        "geo:lat",
        "geo:long",
        "vcard:hasEmail",
        "void:Dataset",
        "ex:WorkedExample",
    ]:
        assert curie in found, f"missing {curie} from {found}"


def test_excludes_url_schemes() -> None:
    """URL schemes (http, https, ftp, mailto, …) are filtered.
    Anything matching ``scheme:LocalName`` where ``scheme`` is in
    ``EXCLUDED_PREFIXES`` does NOT round-trip as a CURIE."""
    text = (
        "Read https:Example and mailto:Address and ftp:Server alongside "
        "sh:NodeShape — only the last is a CURIE."
    )
    found = extract_curies(text)
    assert "sh:NodeShape" in found
    for excluded in [
        "https:Example",
        "mailto:Address",
        "ftp:Server",
    ]:
        assert excluded not in found, f"unexpected {excluded} in {found}"


def test_rejects_digit_local_names() -> None:
    """The local-name's first character must be a letter; this
    mathematically rejects ``localhost:8080`` / ``10:30`` / ``8:00 AM``
    without an explicit exclusion list for digit-led local names."""
    text = "Connect to localhost:8080 by 10:30 or 8:00 AM. Use sh:NodeShape."
    found = extract_curies(text)
    assert "sh:NodeShape" in found
    assert all(":" not in c or c.split(":")[1][0].isalpha() for c in found)


def test_empty_input_returns_empty_set() -> None:
    """Falsy input returns an empty set rather than raising."""
    assert extract_curies("") == set()
    assert extract_curies(None) == set()  # type: ignore[arg-type]


def test_module_constants_exist() -> None:
    """Pin the module's public surface so the validator's re-import
    keeps resolving."""
    assert isinstance(EXCLUDED_PREFIXES, frozenset)
    assert "http" in EXCLUDED_PREFIXES
    assert "https" in EXCLUDED_PREFIXES
    assert CURIE_REGEX.pattern
