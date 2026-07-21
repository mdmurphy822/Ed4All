"""SemantiK semantic structure pipeline — shared library code.

The 8-stage pipeline (see docs/refactor_plan.md):

    stage 1 extract       extract.py + extract_shared.py  no model
    stage 2 features      features.py                     no model
    stage 3a classify     classify.py (+ trained adapter) distilbert
    stage 3b reason       reason.py (+ trained adapter)   Qwen 3 4B
    stage 4 hierarchy     hierarchy.py                    no model
    stage 5 ontology_map  ontology_map.py                 no model
    stage 6 enrich        enrich.py                       local tools
    stage 7 validate      validate.py                     axe-core
    stage 8 escalate      escalate.py                     no model

Legacy modules retained for the data-gen scripts only:
    ir.py, emit_html.py — used by scripts/pair_from_*.py to produce
    the ground-truth HTML the classifier trains against. Slated for
    deletion once the ingesters are rewritten to emit HTML directly.
"""
