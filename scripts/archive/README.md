# Archived one-shot migration scripts

These are executed, one-shot LibV2-archive migration scripts kept for
provenance. Each was run once against on-disk LibV2 course archives to
retroactively backfill a field, reclassify nodes, or rebuild a graph as
the chunk / concept-graph / pedagogy-graph emit pipeline evolved. The
behaviors they performed are now part of the emit pipeline itself, so a
fresh run no longer needs them.

They are not wired into production workflows or the CLI. Their behavior is
frozen, but focused regression tests remain in CI so the historical migrations
stay reproducible and auditable. Do not extend them; if a new migration is
needed, write a new script rather than reviving one of these.

`test_wave76_clean_concept_graph.py` is the companion regression test for
`wave76_clean_concept_graph.py` and is kept alongside its script. It runs
against a synthetic stub graph and pulls in no real course data.

`Trainforge/tests/test_concept_graph_classification.py` likewise preserves the
classification contract exercised by `wave75_classify_concept_graph.py`.
