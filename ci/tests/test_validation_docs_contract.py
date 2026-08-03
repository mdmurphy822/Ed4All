"""Protect the public Bloom-classifier status and workflow wiring contract."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_PATH = PROJECT_ROOT / "config" / "workflows.yaml"
VALIDATORS_DOC = PROJECT_ROOT / "docs" / "validation" / "validators.md"
ARCHITECTURE_DOC = (
    PROJECT_ROOT / "docs" / "architecture" / "validation-architecture.md"
)

EXPECTED_DISAGREEMENT_GATES = {
    "outline_bloom_classifier_disagreement",
    "rewrite_bloom_classifier_disagreement",
}
CANONICAL_VALIDATOR = (
    "lib.validators.bloom.classifier_disagreement."
    "BloomClassifierDisagreementValidator"
)


def _walk_mappings(value: Any) -> Iterator[dict[str, Any]]:
    """Yield every mapping in a nested YAML value."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _paragraph_containing(text: str, marker: str) -> str:
    """Return normalized paragraphs surrounding a contract marker."""
    paragraphs = re.split(r"\n\s*\n", text)
    matches = [paragraph for paragraph in paragraphs if marker in paragraph]
    assert matches, f"missing documentation marker: {marker}"
    return " ".join(" ".join(paragraph.lower().split()) for paragraph in matches)


def _assert_terms(paragraph: str, *terms: str) -> None:
    """Require status terms within the paragraph that owns a marker."""
    missing = [term for term in terms if term.lower() not in paragraph]
    assert not missing, f"missing nearby contract terms {missing}: {paragraph}"


def test_bloom_disagreement_gates_are_advisory_and_canonical() -> None:
    config = yaml.safe_load(WORKFLOWS_PATH.read_text(encoding="utf-8"))
    gates = {
        mapping["gate_id"]: mapping
        for mapping in _walk_mappings(config)
        if mapping.get("gate_id") in EXPECTED_DISAGREEMENT_GATES
    }

    assert gates.keys() == EXPECTED_DISAGREEMENT_GATES
    for gate in gates.values():
        assert gate["validator"] == CANONICAL_VALIDATOR
        assert gate["severity"] == "warning"
        assert gate["behavior"] == {"on_fail": "warn", "on_error": "warn"}


def test_public_docs_preserve_bloom_classifier_accuracy_markers() -> None:
    for path in (VALIDATORS_DOC, ARCHITECTURE_DOC):
        text = path.read_text(encoding="utf-8")
        normalized = text.lower()

        trivote = _paragraph_containing(text, "ED4ALL_BLOOM_TRIVOTE")
        _assert_terms(trivote, "trained", "provisioned", "abst", "heuristic")

        heads = _paragraph_containing(text, "ED4ALL_BLOOM_TRIVOTE_HEADS")
        _assert_terms(heads, "ship", "local artifact", "abst")
        assert any(status in heads for status in ("fallback", "falls back", "continues"))

        strict = _paragraph_containing(text, "TRAINFORGE_REQUIRE_BERT_ENSEMBLE")
        _assert_terms(strict, "strict", "provision")

        deberta = _paragraph_containing(text, "DeBERTa")
        _assert_terms(deberta, "active", "nli", "entailment")
        assert any(
            distinction in deberta
            for distinction in ("distinct", "separate", "not a trained bloom")
        )

        assert "gates.md" in normalized

    assert "validation-architecture.md" in VALIDATORS_DOC.read_text(
        encoding="utf-8"
    ).lower()
    assert "validators.md" in ARCHITECTURE_DOC.read_text(encoding="utf-8").lower()
