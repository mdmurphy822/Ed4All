# Temporary migration compatibility

`wave81_reclassify_chunks.py` is the only remaining temporary exception. It is
not a production tool, supported CLI entry point, or template for new repair
work. This archive directory will disappear after its tracked references are
removed or redirected and the script moves to the ignored regression shelf.

New supported repair behavior belongs in a purpose-named directory under
`Trainforge/scripts/ops/`, `Trainforge/scripts/maintenance/`, or
`Trainforge/scripts/harness/`, according to its role. Historical scripts belong
in the recursively ignored regression directory, not in tracked source.
