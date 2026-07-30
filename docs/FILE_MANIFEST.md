# Ed4All File Manifest

This is the canonical map of **what's used vs. what's extra/trash** across the
Ed4All repository. It was produced by a per-area file-hygiene audit (read-only)
and is meant as a fast orientation aid: for each area it names the productive
core (source, tests, docs, config), then flags the subtrees that are
**DATA / RUNTIME / OUTPUT** — regenerable working data that is `.gitignore`d and
never ships to GitHub.

Companion doc: [`file-audit-cleanup.md`](file-audit-cleanup.md) — the actionable
gitignore + trash cleanup list.

## How to read this

- **Productive core** = tracked source/tests/docs/config that is actively used.
- **DATA / RUNTIME (gitignored)** = large regenerable working trees. A live
  pipeline writes to several of these; they are excluded from git by design.
- Every area below was verified with
  `git status --porcelain --untracked-files=all` — a **zero-gap** result means no
  untracked, non-ignored files exist in that area.

---

## SemantiK — PDF → accessible-HTML conversion engine

The sole PDF → accessible-HTML conversion path. `semantik_structure/` is the
live v2 semantic cascade; `data/ training/ scripts/ eval/` are the model-training
+ evaluation harness that produces its LoRA adapters and classifiers.

| Kind | Paths |
|------|-------|
| Live conversion engine | `semantik_structure/` (205 tracked files): `pipeline_v2.py`, cascade, `region_detection.py`, `reading_order.py`, `region_order.py`, `structure_graph.py`, `pedagogical_units.py`, `vlm_extract.py`/`vlm_fusion.py`/`vlm_furniture.py` |
| Cascade subpackages | `assembler/` (27), `council/` (23), `qwen_specialists/` (47), `gates/` (13), `theta/` (8), `math_reconstruct/` (7), `soft_reranker/` (4), `tests/` (33) |
| Training/eval harness | `training/` (8 `train_*.py`), `data/` (27 dataset builders), `scripts/` (69 eval/smoke/calibrate/infer utils incl. `run_cascade_json.py`, `infer_pdf.py`, `pdf_to_html.py`), `eval/` (4 metric scripts) |
| Docs/config | `CLAUDE.md`, `architecture.md`, `README.semantic.md`, `pyproject.semantic.toml`, `LICENSE`, `docs/` |
| **DATA / RUNTIME (gitignored)** | `models/`, `eval_reports/`, `data/*` (~55 dataset/cache/logs subdirs), `Plans/`, `MANIFEST.md`, `**/__pycache__/`, plus corpus-provenance **harvester source** deliberately kept out of the repo (`scripts/pair_from_*`, `fetch_*`, `crawl_*`, `mine_*`, `semantik_structure/parse_*.py`) |

> The `_`-prefixed tracked scripts in `scripts/` (`_aggregate_v7_eval.py`,
> `_compare_*_adapter_on_test.py`, `_diag_merge_structure.py`, etc.) are private
> eval/diagnostic helpers in the active restructure/eval workflow — **not** scratch.

## Courseforge — course content generation & packaging

Turns staged SemantiK HTML into modules, blocks, learning-objective pages,
and IMSCC packages.

| Kind | Paths |
|------|-------|
| Productive core (230 tracked) | `generators/` (block emitters, providers), `scripts/` (`generate_course.py`, `package_multifile_imscc.py`, `qti_emitter.py`, `blocks.py`, accessibility-validator, imscc-extractor, component-applier), `router/`, `schemas/` (IMSCC XSDs, block schemas), `imscc-standards/`, `templates/`, `config/`, `agents/`, `docs/` |
| Tests | `scripts/tests/` (real pytest suite + `fixtures/sample_html` + `sample_imscc`, all tracked) |
| Docs | `CLAUDE.md`, `README`, `CHANGELOG` |
| **DATA / RUNTIME (gitignored)** | `exports/*` (per-run `PROJ-*` export dirs), `inputs/textbooks/*` (corpus PDFs), `inputs/course-data`, `inputs/exam-objectives`, `__pycache__/`, `.pytest_cache/`. Tracked `.gitkeep` sentinels keep the skeleton. |

## Trainforge — assessment/RAG training + canonical chunker

Parses IMSCC / SemantiK HTML into chunks, builds concept/pedagogy graphs, synthesizes
assessments and instruction/preference training pairs, runs PEFT/adapter training.

| Kind | Paths |
|------|-------|
| Pipeline core | `process_course.py`, `synthesize_training.py`, `instruction_pair_extractor.py`, `pedagogy_graph_builder.py`, `align_chunks.py`, `curriculum.py`, `retag_outcomes.py`, `train_course.py` (all referenced 3–27×) |
| Subsystems | `chunker/` (canonical shared chunker), `generators/`, `parsers/`, `rag/`, `eval/`, `training/` (PEFT runner + `base_models` + `compute_backend`), `scripts/`, `agents/`, `tests/`, `examples/sample_assessment.json` |
| Docs | `CLAUDE.md`, `architecture.md`, `README.md` |
| **DATA / RUNTIME (gitignored)** | `output/` (per-course build artifacts — chunks, graphs, manifests, quality reports), `**/__pycache__/`. Tracked `output/.gitkeep` placeholder. |

## LibV2 — course content repository / library layer

Stores archived per-course corpora + catalog metadata, plus the `libv2` tooling
package.

| Kind | Paths |
|------|-------|
| Tooling package (94 tracked) | `tools/libv2/`: `importer.py`, `indexer.py`, `retriever.py` (+ semantic/multi/result_fusion retrievers), `vector_index.py`, `catalog.py`, `eval_generator`/`eval_harness`, `backup.py`, `migrate.py`, `remove.py`, `jsonld_emit`/`rdf_export`, `models/`, `scripts/` (legacy-corpus chunk backfill tooling), `cli.py` + `tests/` |
| Top-level helpers | `tools/chunk_query.py`, `intent_router.py`, `study_pack_renderer.py`; `LibV2/tests/` integration tests; `vendor/bloom_verbs.json` |
| Docs | `CLAUDE.md`, `README.md`, `instructions.md`, `requirements.txt` |
| **DATA / RUNTIME (gitignored)** | `courses/*` (archived per-course corpora), `catalog/*` (per-course metadata dirs), `tools/**/__pycache__`. Only two `.gitkeep` sentinels tracked under the data trees. |

## MCP — control-plane core (FastMCP server + orchestrator)

Dispatches workflow phases; hosts the executor, IPC, task mailbox, LLM backend,
and tool registry.

| Kind | Paths |
|------|-------|
| Server + core | `server.py`; `core/` (`executor.py`, `workflow_runner.py`, `config.py`, `param_mapper.py`, `tool_schemas.py` + tests) |
| Orchestrator | `orchestrator/` (`pipeline_orchestrator.py`, `local_dispatcher.py`, `task_mailbox.py`, `llm_backend.py`, `worker_contracts.py`) |
| IPC + hardening | `ipc/status_tracker.py`; `hardening/` (`checkpoint.py`, `error_classifier.py`, `lockfile.py`, `validation_gates.py`, `gate_input_routing.py` + tests) |
| Tools | `tools/` (`pipeline_tools.py`, `orchestrator_tools.py`, `courseforge_tools.py`, `trainforge_tools.py`, `gui_tools.py`, `quiz_generator.py`, `tutoring_tools.py`) |
| Tests | 165 files under `MCP/tests` + per-subdir `tests/` |
| **DATA / RUNTIME (gitignored)** | `**/__pycache__/`, `*.cpython-312.pyc`. All 244 tracked files are `.py`. |

## lib + cli — shared libraries + `ed4all` CLI

| Kind | Paths |
|------|-------|
| lib/ (742 tracked) | `validators/` (gate validators + `shacl/*.ttl` SHACL shape), `aggregators/`, `ontology/` (bloom, slugs, learning_objectives, taxonomy loaders), `objectives/`, `retrieval/`, `embedding/`, `generation/`, `classifiers/`, `governance/`, `diagnostics/`, `semantic_structure_extractor/`, `llm/`, `utils/`, `testing/` |
| cli/ (51 tracked) | `commands/` (backup, convert, doctor, run, stop, import-docs, support-bundle, objectives, …) + co-located `cli/tests/` and per-package `tests/` |
| **DATA / RUNTIME (gitignored)** | `lib/**/__pycache__/`, `cli/**/__pycache__/` (30+ dirs) |

## config + schemas

| Kind | Paths |
|------|-------|
| config/ (5 tracked) | `workflows.yaml`, `agents.yaml`, `endpoints.yaml`, `config/tests/` |
| schemas/ (168 tracked) | JSON Schemas across 20 domain subdirs (knowledge, aggregators, config, events, governance, training, taxonomies, …) + test fixtures under `schemas/tests/fixtures` and `schemas/taxonomies/tests` |
| **DATA / RUNTIME (gitignored)** | `schemas/training/schema_translation_catalog.rdf_shacl.yaml` — a **generated** catalog output (its sibling `.schema.json` is tracked). Correctly ignored, not a leak. |

## ci + scripts

| Kind | Paths |
|------|-------|
| ci/ | `integrity_check.py` + `validator_test_allowlist.txt` — the CI gate (referenced by `.github/` workflows; **USED**, do not flag) |
| scripts/ core | `build_demo_course.py`, `calibration_harness.py`, `calibrate_phase4_thresholds.py`, `mailbox_servicer.py`, `gpu_guard.sh`, `render_audit.py`, `semantik_rerender.py`, `structure_scorecard.py`, `gold_compare.py`, `retrieval_smoke.py`, `repair_partial_resume_state.py`, `run_post_courseforge_tail.py`, OCR/raster probes, `codegen/sync_provenance_enum.py`, `scripts/integration/*`, `scripts/tests/*` |
| Provenance-only (tracked, dead-by-design) | `scripts/archive/` (14 `wave*`/`test_wave*` one-shot LibV2 migration scripts + `README.md`) — kept for audit per documented rationale; **not** trash |
| **DATA / RUNTIME (gitignored)** | `runtime/shots/` (rendered PNGs from `shoot_pages.py`), `**/__pycache__/` |

## docs + plans + examples

| Kind | Paths |
|------|-------|
| docs/ (tracked, published) | 32 files across accessibility, architecture, compliance, concept-graph, contributing, libv2, metrics, operations, schema, validation — the canonical operator/architecture references. **Only docs/ ships to GitHub.** |
| **plans/ (fully gitignored)** | dated roadmap/design markdown + feedback notes driving `/loop` work (incl. `FRAMEWORK.pdf`, `.docx` roadmaps) — intentional working scratch |
| **examples/ (fully gitignored)** | `sample_course_data.json`, `sample_objectives.json`, `README.md` — runnable quick-start corpus |

## gui — opt-in control-plane web GUI

| Kind | Paths |
|------|-------|
| Backend (124 tracked) | `app.py`, `server.py`, `launch.py`, `auth.py`, `env_catalog.py`, `models.py`, `settings_store.py`, `shared_state.py`; `routers/` (courses, learn, library, retrieval, runs, settings, uploads); `services/` (11 modules) |
| Frontend | `static/` (`studio/`, `learn/`, `dev/`, `shared/` component library) |
| Tests + docs | `tests/` (46-test suite), `README.md`, `LAUNCH.md` |
| **DATA / RUNTIME (gitignored)** | `**/__pycache__/` |

## tests — top-level cross-project suite

| Kind | Paths |
|------|-------|
| Test modules (38 tracked) | `test_pipeline_integration.py`, `test_w10_assessment_e2e.py`, `test_endpoint_registry_drift_guard.py`; `integration/` (5 tests + conftest); `offline/`; `decision_capture/` |
| Fixtures | `fixtures/pipeline/` (`build_fixture_pdf.py`, `build_reference_week.py`, `fixture_corpus.pdf` 4591 B, `reference_week_01/`, `reference_libv2/`); `fixtures/retrieval/mini_course/` (`build_mini_course.py` + chunkset/source/eval subdirs + `mini.imscc` 1608 B) |
| Note | Binary fixtures are well under the 1 MB threshold and each has a committed regenerable builder — satisfies the fixture-hygiene contract. |

## inputs — corpus data (NOT source)

| Kind | Paths |
|------|-------|
| **DATA / OUTPUT (fully gitignored)** | `inputs/` (corpus PDFs + per-corpus src/build/import scratch + `calib` / `contentgen` working dirs). Conversion output (accessible HTML + `cascade_ir`/`quality`/`synthesized` JSON) is written under each corpus's own gitignored working tree. |
| Only tracked item | `inputs/.gitkeep` |

## runtime + state + captures — pipeline scratch, run state, decision captures

| Kind | Paths |
|------|-------|
| Tracked core | Exactly 14 `.gitkeep` placeholders preserving the dir skeleton (`runtime/state/{gui,locks,logs,progress,runs,status,workflows}`, `runtime/training-captures/{courseforge,semantik,decisions,libv2,orchestrator,textbook-pipeline,trainforge}`) |
| **DATA / RUNTIME (gitignored)** | `runtime/` (pure scratch); `runtime/state/runs/` + `runtime/state/workflows/*.json` (live run state, actively written); `runtime/state/GENERATION_PROGRESS.md`; `runtime/state/gui/`; `runtime/state/{locks,status,progress}`; `runtime/state/benchmarks/`; `runtime/training-captures/**/*.jsonl` (15,802 decision-capture files — the mandated capture sink) |

> The `.gitignore` here is exemplary: `runtime/` wholesale; `runtime/state/<dir>/*`
> per-dir rules **plus** a defensive `runtime/state/*/*` catch-all with
> `!state/*/.gitkeep`, so a new state subdir cannot leak.

## root — top-level docs, packaging, container/build

| Kind | Paths |
|------|-------|
| Tracked (all in use) | `README.md`, `ARCHITECTURE.md`, `CLAUDE.md`, `VERSIONING.md`, `CONTRIBUTING.md`, `AGENTS.md`; `pyproject.toml`, `Makefile`; `conftest.py`, `pytest.ini`; `Dockerfile`, `Dockerfile.gui`, `docker-compose.yml`, `.dockerignore`, `run-gui.sh`, `run-gui.bat`; `LICENSE`, `NOTICE`, `.gitignore` |
| **DATA / RUNTIME (gitignored)** | `__pycache__/`, `.pytest_cache/`, `ed4all.egg-info/`, `.venv/`, `docker-compose.override.yml` (local-override convention) |
