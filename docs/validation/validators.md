# Canonical Helpers & Validators

> Long-form home for per-validator detail, BERT ensemble member detail, and pyproject extras. Root `CLAUDE.md § Canonical Helpers` carries one-line bullets per helper; this file carries the canonical paragraph-bullet for each validator.

## Single-source-of-truth loaders (`lib/ontology/`)

- `lib/ontology/bloom.py` — Bloom verb / level / cognitive-domain detection.
- `lib/ontology/slugs.py::canonical_slug` — unified slug helper.
- `lib/ontology/teaching_roles.py` — `(component, purpose) → role` mapper.
- `lib/ontology/taxonomy.py::load_taxonomy(name)` — generic JSON-taxonomy loader, reads from `schemas/taxonomies/`.

## Validators (`lib/validators/`)

See `docs/validation/gates.md` for the wiring (workflow → phase → gate_id → validator).

- `lib/validators/page_objectives.py` — objective coverage per page.
- `lib/validators/content_type.py` — content_type enum enforcement (gated).
- `lib/validators/evidence.py` — per-rule evidence discriminator loader; strict mode drops FallbackProvenance.
- `lib/validators/assessment_objective_alignment.py` — fail-loud gate keeping every assessment question's `objective_id` covered by at least one chunk's `learning_outcome_refs`.
- `lib/validators/source_refs.py` — verifies every emitted Courseforge `sourceId` resolves against the DART staging manifest.
- `lib/validators/libv2_manifest.py` — validates LibV2 manifest JSON, scaffold completeness, and on-disk artifact hash/size agreement.
- `lib/validators/libv2_model.py` — validates emitted `model_card.json` against `schemas/models/model_card.schema.json`. Critical: schema match, weights file presence + size + sha256 agreement, `pedagogy_graph_hash` resolves to extant graph in same course. Warning: missing eval scores, missing license, malformed HF repo regex. Wired as the `libv2_model` gate.
- `lib/validators/kg_quality.py` — KG-quality report (completeness / consistency / accuracy / coverage); thin wrapper over `Trainforge/rag/kg_quality_report.py::KGQualityReporter`. Thresholds: 0.95 / 0.95 / 0.95 / 0.5.
- `lib/validators/min_edge_count.py` — Pre-synthesis gate: critical-fails on pedagogy graph with <100 edges, <4 distinct edge types, or concept graph with <50 nodes. Closes the silent zero-edge regression class for the synthesis surface.
- `lib/validators/synthesis_diversity.py` — Post-synthesis gate: critical-fails when top-3 templates >60% of pairs, single template >35%, or distinct templates <8. Warns when total pairs <100.
- `lib/validators/synthesis_leakage.py` — Post-synthesis gate covering two contamination vectors: (a) verbatim-span leakage from `chunk.text` (default 5% rate / 50-char span); (b) assessment-outline scaffolding patterns like `Question N (XX-NN, Bloom: ...)` (default 0% — zero tolerance, structural contamination). Tunable via gate `config.thresholds.leak_rate_threshold`, `leak_span_chars`, `assessment_scaffold_rate_threshold`.
- `lib/validators/objective_assessment_similarity.py` — Cosine-similarity floor between every assessment-item block stem and its referenced learning-objective text. Default `min_cosine = 0.55` (calibrated against the embedder's intrinsic similarity floor — topically-related but not semantically-aligned pairs cluster below ~0.40). Below threshold emits `action="regenerate"`. Wired symmetrically as `outline_objective_assessment_similarity` (inter_tier_validation) and `rewrite_objective_assessment_similarity` (post_rewrite_validation).
- `lib/validators/concept_example_similarity.py` — Cosine-similarity floor between every concept-block definition and its illustrating example. Default `min_cosine = 0.50` — strictly looser than the objective↔assessment gate's 0.55 because examples are intentionally more concrete than the abstract concept they illustrate.
- `lib/validators/objective_roundtrip_similarity.py` — Cosine-similarity floor between the rewrite-tier learning-objective paraphrase and the source objective. Default `min_cosine = 0.70` — strictly tighter than the prior two gates because a paraphrase MUST preserve meaning; below 0.70 indicates semantic drift, not just surface-form variation.
- `lib/validators/courseforge_outline_shacl.py` — Statistical-tier wrapper around the `schemas/context/courseforge_v1.shacl-rules.ttl` shape constraints, applied to outline-tier Block emit before the rewrite tier sees it.
- `lib/validators/bloom_classifier_disagreement.py` — Wraps `lib/classifiers/bloom_bert_ensemble.py::BloomBertEnsemble`. Fires `action="regenerate"` on (a) majority-vote disagreement with the block's declared `bloomLevel` (`bert_ensemble_disagreement` decision event) or (b) ensemble dispersion above default `bert_dispersion = 0.7` (`bert_ensemble_dispersion_high` decision event; entropy of normalised per-level scores, range `[0, 1]`).

## BERT ensemble members

`lib/classifiers/bloom_bert_ensemble.py::_DEFAULT_ENSEMBLE_MEMBERS`:

1. `kabir5297/bloom_taxonomy_classifier` — purpose-built 6-class Bloom classifier; natively aligned with the canonical `BLOOM_LEVELS` enum (`remember` / `understand` / `apply` / `analyze` / `evaluate` / `create`).
2. `distilbert-base-uncased-finetuned-sst-2-english` — sentiment model contributing dispersion via the low-resolution `_SST2_TO_BLOOM` heuristic mapping (POSITIVE → `evaluate`, NEGATIVE → `remember`); intentionally a low-confidence vote whose role is dispersion contribution, not majority dominance.
3. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` — zero-shot NLI classifier; given a candidate text + the six Bloom-level labels as hypotheses, picks the highest-entailment level.

Each member's `revision` field is pinned to a concrete HuggingFace commit SHA, and each resolved revision is captured in the `bert_ensemble_member_loaded` decision event so the audit trail records which revision produced each classification.

## Optional pyproject extras

`pyproject.toml::[project.optional-dependencies]`:

- `embedding` — `sentence-transformers>=2.5,<4`, `transformers>=4.49,<4.50`, `torch>=2`, `numpy>=1.24`. Required for the four statistical-tier validators above. `pip install -e '.[embedding]'`. Kept out of the default install so CPU-only dev boxes don't pull in torch + transformers wheels just to run the orchestrator. There is no separate `[bert]` extras group: the BERT ensemble reuses the `transformers` pin from `[embedding]`, so `pip install -e '.[embedding]'` enables both the embedding-similarity validators AND the BERT ensemble disagreement gate. Missing extras degrade gracefully (warning-severity GateIssue, `passed=True`) unless `TRAINFORGE_REQUIRE_EMBEDDINGS=true` flips to fail-closed.

## Canonical LO helper

`lib/ontology/learning_objectives.py` owns the single source of truth for LO identity (`mint_lo_id`, `validate_lo_id`, `hierarchy_from_id`, `split_terminal_chapter`). Pattern `^[A-Z]{2,}-\\d{2,}$` mirrors `schemas/knowledge/courseforge_jsonld_v1.schema.json`. `schemas/knowledge/course.schema.json` is the canonical shape for Trainforge-emitted `course.json` consumed by LibV2.
