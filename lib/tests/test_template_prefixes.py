"""Wave 131 — single-source-of-truth regression for the deterministic
generator template_id prefix tuple.

Both consumers (lib.validators.curie_preservation and
Trainforge.synthesize_training) MUST import the canonical tuple from
lib.ontology.template_prefixes. Drift between local copies = silent
under-enforcement (e.g. a future fifth deterministic generator added
to one file but not the other).
"""
from __future__ import annotations


def test_deterministic_template_prefixes_single_source_of_truth() -> None:
    """Both consumers import the same canonical tuple via identity check."""
    from lib.ontology.template_prefixes import (
        DETERMINISTIC_TEMPLATE_PREFIXES as canonical,
    )
    from lib.validators.curie_preservation import (
        DETERMINISTIC_TEMPLATE_PREFIXES as cp,
    )
    from Trainforge.synthesize_training import (
        _DETERMINISTIC_TEMPLATE_PREFIXES as st,
    )

    assert cp is canonical
    assert st is canonical
    assert canonical == (
        "kg_metadata.",
        "violation_detection.",
        "abstention.",
        "schema_translation.",
    )
