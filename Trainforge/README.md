# Trainforge

**Turn a packaged course into a retrieval corpus, a typed knowledge graph, and a course-pinned SLM adapter.**

Trainforge is the back half of the Ed4All pipeline. It consumes IMSCC course packages (from Courseforge or any supported LMS) and does four things:

1. **Chunks + enriches.** Splits course content into pedagogical units tagged with Bloom's level, content type, key terms, and references back to the source region. The canonical chunker (`Trainforge/chunker/`) is shared with the conversion and IMSCC paths — one chunker, one contract.
2. **Builds the concept graph.** A typed graph over the corpus: three taxonomic relations (is-a, prerequisite, related-to) plus five pedagogical ones (assesses, exemplifies, misconception-of, derived-from-objective, defined-by).
3. **Synthesizes training pairs.** SFT (`instruction_pairs.jsonl`) and DPO (`preference_pairs.jsonl`) from the chunkset, through a license-clean local or hosted-OSS teacher seat — never through the orchestrating assistant.
4. **Trains and evaluates an adapter.** A course-pinned LoRA adapter with a model card pinning seven SHA-256 provenance hashes back to the LibV2 paths that produced it.

Assessment generation also lives here, but in the current `textbook_to_course` pipeline the graded artifacts are emitted upstream by Courseforge's `assessment_synthesis` phase; Trainforge's optional `trainforge_assessment` phase harvests and extends them.

## Quick example

```bash
# As part of the full pipeline (steps 1-3; training is opt-in via --with-training):
ed4all run textbook-to-course --corpus <CORPUS_PATH> --course-name <COURSE_NAME>

# Standalone RAG training on an existing IMSCC:
ed4all run rag_training --corpus <IMSCC_PATH> --course-name <COURSE_NAME>

# Train an adapter for an already-imported course (step 4):
ed4all run trainforge_train --course-name <course-slug> --base-model <name>
```

Pipeline output lands under `LibV2/courses/<slug>/` — `imscc_chunks/`, `graph/`, `training_specs/`, `models/`. (`Trainforge/output/` is only the default for a direct `python -m Trainforge.process_course --output ...` invocation.) Decision JSONL lands under `runtime/training-captures/trainforge/<COURSE_CODE>/`.

## More

- [`Trainforge/CLAUDE.md`](CLAUDE.md) — chunk shape, metadata extraction priority chain, Bloom's rubric, concept-graph edge taxonomy, decision-capture contract, and the behavior-flag table. Read **§ "Training-pair synthesis — what actually runs"** before touching synthesis: it documents which entry point reaches which pair program, and which capabilities exist but are currently unreachable.
- [`Trainforge/architecture.md`](architecture.md) — module map.
- [`docs/operations/nemotron-lora-canary.md`](../docs/operations/nemotron-lora-canary.md) — the qualified training environment (`scripts/ops/bootstrap-training-env.sh` + `scripts/ops/ed4all-training`) and the required canary preflight before a production fit.
- [`docs/LICENSING.md`](../docs/LICENSING.md) — which teacher seats may author a shippable training corpus.

## License

MIT
