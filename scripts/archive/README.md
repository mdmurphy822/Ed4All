# Archived one-shot migration scripts

The remaining files are unsupported historical one-shot migrations retained
temporarily while they are reviewed and moved to the ignored regression shelf.
They are not production tools, supported CLI entry points, or templates for new
repair work.

Two files remain here because tracked references must be resolved first:

- `wave75_classify_concept_graph.py` is imported directly by
  `Trainforge/tests/test_concept_graph_classification.py`.
- `wave81_reclassify_chunks.py` still has tracked legacy references that must be
  removed or redirected before it is shelved.

New supported repair behavior belongs in a purpose-named directory under
`Trainforge/scripts/ops/`, `Trainforge/scripts/maintenance/`, or
`Trainforge/scripts/harness/`, according to its role. Historical scripts belong
in the recursively ignored regression directory, not in tracked source.
