"""Shared implementation for the SemantiK document-conversion pipeline.

The package extracts and featurizes source documents, combines deterministic
layout analysis with the specialist council and enrichment stages, assembles
accessible HTML, and validates the result. The compatibility pipeline remains
available through :mod:`semantik_structure.pipeline` for explicit v1 runs.

``ir.py`` and ``emit_html.py`` provide the intermediate representation and HTML
emission used by training-data builders and pair-generation utilities.
"""
