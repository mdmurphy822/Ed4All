"""Ontology loaders for Ed4All taxonomy schemas.

Provides canonical accessors over `schemas/taxonomies/*.json`, the authoritative
source of truth for Bloom verbs, question types, assessment methods, content
types, cognitive domains, teaching roles, and module types.

Callers should prefer `lib.ontology` modules over hardcoded enum/verb lists
so the taxonomy schema is the single source of truth.
"""
