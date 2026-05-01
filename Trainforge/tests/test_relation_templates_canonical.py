"""Wave 132a regression: bytewise alignment of train + eval relation templates.

The kg_metadata generator (`Trainforge/generators/kg_metadata_generator.py`)
emits training pairs whose prompts mirror the eval-time probes the
faithfulness evaluator (`Trainforge/eval/faithfulness.py`) asks. Drift
between the two would desync the adapter's training signal from the
eval probe — kg_metadata_generator's docstring explicitly notes that
"keeping the wording bytewise-aligned is load-bearing".

Wave 132a consolidated the two duplicate `_RELATION_TEMPLATES` copies
into `lib/ontology/relation_templates.py`. This test pins the
contract: every relation that faithfulness probes must have a matching
canonical entry, and the bytewise positive form must agree.
"""
from __future__ import annotations


def test_kg_metadata_and_faithfulness_share_relation_template_text():
    """Bytewise-identical positive forms across train/eval is load-bearing."""
    from lib.ontology.relation_templates import RELATION_TEMPLATES
    from Trainforge.eval.faithfulness import _RELATION_TEMPLATES as fp

    # Each relation in faithfulness must equal the positive form in canonical.
    for rel, fp_text in fp.items():
        assert rel in RELATION_TEMPLATES, f"missing canonical: {rel}"
        canonical_pos, _ = RELATION_TEMPLATES[rel]
        assert fp_text == canonical_pos, f"drift on {rel}"


def test_kg_metadata_generator_imports_canonical_map():
    """kg_metadata generator's _RELATION_TEMPLATES is the canonical map."""
    from lib.ontology.relation_templates import RELATION_TEMPLATES
    from Trainforge.generators.kg_metadata_generator import (
        _RELATION_TEMPLATES as kg,
    )

    # Same object identity proves no rewrap / partial copy.
    assert kg is RELATION_TEMPLATES, (
        "kg_metadata_generator._RELATION_TEMPLATES must alias the canonical "
        "map directly, not copy it — copies invite drift."
    )


def test_canonical_map_has_assesses_assessment_wording():
    """The pre-Wave-132a drift was on 'assesses' specifically.

    Pin the eval-aligned wording ("assessment") so a future edit that
    re-introduces the kg_metadata-side "chunk" wording fails loud.
    """
    from lib.ontology.relation_templates import RELATION_TEMPLATES

    pos, _neg = RELATION_TEMPLATES["assesses"]
    assert "assessment" in pos, (
        f"'assesses' relation must use 'assessment' (faithfulness wording), "
        f"got: {pos!r}"
    )
