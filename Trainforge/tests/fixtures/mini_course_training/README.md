# mini_course_training — SFT/DPO training-pair synthesis fixture

Pre-built corpus of 15 enriched chunks used by Worker C's
`Trainforge.synthesize_training` stage. This fixture is NOT a raw HTML course
— it is pre-chunked and pre-enriched, because the synthesis stage consumes
`corpus/chunks.jsonl` (the output of the base + alignment passes) rather than
raw source HTML.

## Layout

```
mini_course_training/
├── README.md
├── corpus/
│   └── chunks.jsonl          # 15 enriched chunks, JSON one per line
└── training_specs/
    └── dataset_config.json   # Minimal stub; updated by the stage
```

## Shape of a chunk

Every chunk has the full enriched shape the stage depends on:

- `id` — unique chunk id
- `text` — full chunk text (used for the verbatim-leakage check)
- `learning_outcome_refs` — at least one ref, except where noted below
- `bloom_level` — one of the six Bloom levels, or `null`
- `content_type_label` — `explanation`, `procedure`, `example`, `comparison`, or `null`
- `key_terms` — list of `{term, definition}` dicts (present on most chunks)
- `misconceptions` — list of `{misconception, correction}` dicts (present on ~half)
- `concept_tags` — list of short tag strings

## What this fixture exercises

### Eligibility filter

Chunk `chunk_orphan_01` intentionally has an empty `learning_outcome_refs`
list. The stage must skip it entirely — zero instruction pairs, zero
preference pairs, counted in `stats.chunks_skipped_no_lo`.

### Misconception-backed preference pairs

Chunks `chunk_mc_01` … `chunk_mc_06` each carry at least one explicit
misconception. The preference factory must pull from the chunk's
`misconceptions[0]` and mark `rejected_source == "misconception"`.

### Rule-synthesized preference pairs

Chunks `chunk_rule_01` … `chunk_rule_05` have no misconceptions. The
preference factory must fall back to the deterministic negation-swap
distractor and mark `rejected_source == "rule_synthesized"`.

### Bloom × content-type template coverage

Chunks are spread across four Bloom levels (`remember`, `understand`,
`apply`, `analyze`) and four content types (`explanation`, `procedure`,
`example`, `comparison`) so the template-selection branch is exercised.

## CI assertions on this fixture

The test module `Trainforge/tests/test_training_synthesis.py` asserts:

- Exactly 14 eligible chunks (15 minus the orphan)
- At least 20 instruction pairs emitted across two runs with different seeds
  (single-run emission is 14; double-run emission sharing a seed is 14 by
  idempotence, but the integration assertion in the test doubles a seed-spread
  call to prove ≥20 is attainable)
- At least 5 preference pairs emitted (one per misconception-bearing chunk)
- Every emitted pair validates against the JSON schemas in
  `schemas/knowledge/instruction_pair.schema.json` and `schemas/knowledge/preference_pair.schema.json`
- No prompt contains a 50+-char verbatim span from its source chunk text
- Stage idempotence: two runs with the same seed produce byte-identical
  `instruction_pairs.jsonl` and `preference_pairs.jsonl`

If any of these fail, the Worker C contract is broken.
