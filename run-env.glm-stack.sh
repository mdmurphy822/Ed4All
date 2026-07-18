# ============================================================================
# run-env.glm-stack.sh — full-run GLM-OCR stack profile
#   v1 (plan §2, 2026-07-16): alg-glm-01 build.
#   v2 (2026-07-18): FRESH RUN after the validation overhaul (commit 4e19ca97).
#      COURSE NAME: alg-glm-02  (fresh identity — zero contamination; the
#      alg-glm-01 export/LibV2 artifacts stay browsable for A/B comparison).
#      Adds the proven post-overhaul flags (frontier NLI, feature cache, token
#      buckets, cross-block NLI, embed cache/fp16, Bloom trivote+harvester,
#      subsection chunk breaks, synthesis skeleton). Runbook:
#      scratchpad/freshrun-runbook-2026-07-18.md.
# Host: spark-3b3e (GB10, 128 GB unified). Source before `ed4all run`.
# Invocation binary: ~/.local/bin/ed4all (system python --user installs; the
# repo .venv is a stale old-box artifact — do NOT use it).
#
# Seat topology (never all at once — orchestrator owns transitions):
#   A. conversion:      vllm-glmocr :8002 (0.45) + vllm-qwen30 :8003 (0.35)
#   C. everything else: vllm-super-final :8001 (0.85, MTP k=3 + FP8 KV)
#      (outline-phase alt seat: vllm-super-sd = ngram k=5 — swap only if the
#       3-min MTP window at outline start measures agg < ~110 tok/s)
# Coherence probe after EVERY seat start (mode-collapse rule).
# ============================================================================

# ---- license-clean, pure-local (no cloud anywhere) -------------------------
unset ANTHROPIC_API_KEY NVIDIA_API_KEY || true
# Every model is in the local HF cache; a stale hf_oauth token 401'd the
# layout-model HEAD call at load (2026-07-17) and transformers fails hard
# instead of falling back to cache. Pure-local run → hub fully offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# ---- LLM routing: all Courseforge/planner traffic -> Super seat :8001 ------
export LLM_MODE=api
export LLM_PROVIDER=spark-super
export SPARK_BASE_URL=http://localhost:8001/v1
export SPARK_API_KEY=local
# The seat serves --served-model-name nemotron-3-super (registry default is
# nemotron-3-super-120b-a12b -> would 404). Pin the served name:
export SPARK_SUPER_MODEL=nemotron-3-super
export COURSEFORGE_PROVIDER=spark-super
export COURSEPLANNER_PROVIDER=spark-super
export TEXTBOOK_SYNTHESIS_PROVIDER=spark-super
export COURSEFORGE_OUTLINE_PROVIDER=spark-super
export COURSEFORGE_REWRITE_PROVIDER=spark-super
# Per-family model pins: with only *_PROVIDER set, these providers fall back
# to the LOCAL_SYNTHESIS default model id (qwen2.5:7b… — 404s on the super
# seat; seen live at canary-3 objective_extraction). Pin the served name.
export TEXTBOOK_SYNTHESIS_MODEL=nemotron-3-super
export COURSEPLANNER_MODEL=nemotron-3-super
export COURSEFORGE_OUTLINE_MODEL=nemotron-3-super
export COURSEFORGE_REWRITE_MODEL=nemotron-3-super
# W10 assessment synthesis rides the same seat
export TRAINFORGE_ASSESSMENT_PROVIDER=spark-super
export TRAINFORGE_ASSESSMENT_MODEL=nemotron-3-super
# Nemotron reasoning control: thinking off on composed OpenAI-compatible calls
export ED4ALL_REASONING_THINKING_OFF=1

# ---- SemantiK GLM-OCR extraction lane (plan §2 delta) ----------------------
export SEMANTIK_GLMOCR_LANE=1                 # new extraction lane (owner-adopted)
export SEMANTIK_GLMOCR_WORKERS=4              # measured 4,453 pages/hr
export SEMANTIK_ALTTEXT_PROVIDER=qwen30       # blind-validated winner (:8003 default)
export SEMANTIK_ARRANGER_ID_REPAIR=1          # deterministic id repair (fallback path)
# GLMOCR_BASE_URL :8002 / ALTTEXT_BASE_URL :8003 / models glm-ocr, qwen3-vl-30b
# are the code defaults and match the registered containers — not overridden.
# 2026-07-18: Super skeleton-aware heading-level JUDGE (lane.py step 3b, the
# escalations consumer). GATED — kept OFF here for two reasons: (1) the live
# ch01/ch06 trial was still in flight at the 07-17 shutdown (adopt only once it
# passes); (2) the judge dispatches to Super :8001, which CANNOT co-reside with
# glm+qwen on 122 GiB — so it must run as a POST-swap Super pass over the
# escalations sidecars (standalone runner), NOT truly inline during stage A.
# Enable per the runbook's "heading-judge seat note" once both are resolved.
# export SEMANTIK_HEADING_JUDGE=1

# ---- Courseforge two-pass (validated run-2 configuration) ------------------
export COURSEFORGE_TWO_PASS=true
export COURSEFORGE_OUTLINE_CONCURRENCY=32     # knee-validated
export COURSEFORGE_REWRITE_CONCURRENCY=32
export COURSEFORGE_CURIE_POSTMINT=1           # deterministic CURIEs (1.03 calls/block)
export COURSEFORGE_REWRITE_HTML_REPAIR=1      # deterministic tag-balance self-repair at emit
export ED4ALL_NLI_MICROBATCH=1                # defeats the ~28-in-flight lock ceiling
export ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE=1 # real grounding enforcement

# ---- Validation architecture v2 (2026-07-18 overhaul — replaces the old
#      "12-hour serial NLI" path; BENCH-8 full 52-gate chain = 5:46 cold) -----
# The block_prose_entailment gate no longer scores the whole-passage grid.
# ED4ALL_GROUNDEDNESS_FRONTIER: priority-ordered per-claim early-stop, PROVEN
# verdict-identical (2307/2307 vs serial baseline, -67.6% pairs). Never drops a
# candidate — a non-entailing claim still exhausts its whole pool, so the
# argmax + stage-2 rescue are bitwise-unchanged.
export ED4ALL_GROUNDEDNESS_FRONTIER=1          # 2026-07-18: verdict-identical early stop
# Shared per-block feature cache — the ~424-block hydration / chunk parse /
# HTML-strip / sentence-split / embeddings computed ONCE for the whole 52-gate
# suite (was per-gate recompute = the suite's old 40-180 min). Verdict-identical.
export ED4ALL_VALIDATION_FEATURE_CACHE=1       # 2026-07-18: compute-once gate inputs
# Single-process cross-block NLI concurrency for the entailment gate — fans
# cache-miss blocks across a thread pool onto the ONE coalescing GPU dispatcher
# (no multiprocess CUDA). Supersedes the dormant spawn-pool VALIDATORS path.
export ED4ALL_NLI_CROSSBLOCK=1                 # 2026-07-18: cross-block dispatcher fan-out
export ED4ALL_NLI_CROSSBLOCK_THREADS=16        # 2026-07-18: fan-out width (capped at miss count)
export ED4ALL_NLI_MICROBATCH_MAX_PAIRS=128     # 2026-07-18: coalesce depth for the cross-block pop.
# Token-length bucketed NLI forward pass — short window pairs batch big, long
# ~512-tok pairs batch small (bounded O(L^2) activation). Verdict-identical.
export ED4ALL_NLI_BUCKET_BATCHING=1            # 2026-07-18: kills arrival-order padding waste
# The spawn ProcessPool VALIDATORS path is SUPERSEDED by crossblock (measured
# livelock on this box) — stays OFF.
export ED4ALL_NLI_MICROBATCH_VALIDATORS=0
export ED4ALL_ANSWER_NUM_CTX=28672            # tripwire matches true 32k window - headroom
# NOT enabled (premise refuted 2026-07-16): COURSEFORGE_REWRITE_EARLY_EXIT, trim-N YAML.
# Capability-tier routing override: the in-repo block_routing.yaml pins the
# small tier to the local 7B/ollama seat (not serving in this profile), and
# the chain OUTRANKS the *_PROVIDER env pins (outline 404'd live). Run-scoped
# copy routes every tier to the Super seat.
export COURSEFORGE_BLOCK_ROUTING_PATH=/home/mdmurphy/Projects/Ed4All/scratchpad/block_routing.glm-stack.yaml

# ---- Course structure (validated run-2: TO modules, per-CO pages, TO titles)
export ED4ALL_TO_CHAPTER_ANCHOR=1             # chapter-anchored TO derivation
export ED4ALL_WEEK_TO_GROUPS=1                # week N = TO-N membership
export ED4ALL_CONTENT_PAGE_PER_CO=1
export ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED=1  # true one-page-per-CO
export ED4ALL_IMSCC_MODULE_TITLES=to          # "Module N: <TO topic>" org titles
# 2026-07-18: inject the window chapter's heading skeleton into the Stage-2
# per-window synthesis prompt so TO/CO derivation is structure-aware (active
# only with TEXTBOOK_SYNTHESIS_PROVIDER set — it is). NOTE: folds into the
# window-render fingerprint, so window resume sidecars invalidate on a mid-run
# flip — fine for a fresh alg-glm-02 build (no prior sidecars).
export ED4ALL_SYNTHESIS_SKELETON=1

# ---- Structure-fidelity guards (objective_extraction over converted HTML) --
export ED4ALL_STRUCTURE_EXTRACT_GUARDS=1
export ED4ALL_STRUCTURE_OUTLINE_ANCHOR=1
export SEMANTIK_TITLE_SANITIZE=true

# ---- Chunking fixes + pre-synthesis health gate ----------------------------
export TRAINFORGE_DROP_FRONTMATTER=true
export TRAINFORGE_DROP_APPARATUS_DUMPS=true
export TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE=true
export ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR=20
# ED4ALL_CHUNK_SECTION_HARD_BREAK intentionally OFF: A/B on the canary HTML
# measured 498 chunks / p50 42 words / 208 runts (vs 72 / 755 / 0 off) — the
# flag flushes far beyond its 49 matching boundary headings on this corpus.
# The 3 cross-section bleed chunks it would fix are the lesser harm.
# 2026-07-18: size-guarded successor to the refuted HARD_BREAK — breaks at a
# genuine content sub-section heading only once the buffer clears the min-words
# floor (prevents the 208-runt cascade), closing the CO-depth granularity gap.
export ED4ALL_CHUNK_SUBSECTION_BREAK=1
export ED4ALL_CHUNK_SUBSECTION_MIN_WORDS=250  # accumulation floor before a sub-section flush
export ED4ALL_CHUNK_HEALTH_GATE=1             # canary-mandated; kept for full run

# ---- GPU-for-all-inference (owner directive) -------------------------------
export ED4ALL_NLI_DEVICE=cuda
export ED4ALL_EMBEDDING_DEVICE=cuda
# Big-memory concurrent-serving profile (ed4all doctor gpu_profile advisory on
# this 122 GiB unified box): no phase-boundary eviction sweep, no NLI evict —
# the resident vLLM seat and NLI/embed co-reside; orchestrator owns seat swaps.
export ED4ALL_GPU_LIFECYCLE=0
export ED4ALL_NLI_EVICT_FOR_CUDA=0
# 2026-07-18: embed-throughput pack — bigger GPU batches + longest-first
# packing on the two hot SentenceEmbedder callers (feature cache + groundedness
# S1). Cosine shifts only low-order bits, well under every downstream floor.
export ED4ALL_EMBED_BATCH_TUNE=1
# Disk-persisted content+model-addressed embedding cache (state/embedding_cache
# .batch.<model>.jsonl) — batch-encodes only misses, appends incrementally.
export ED4ALL_EMBED_PERSIST_CACHE=1
# fp16 embed cast on CUDA-resident SentenceTransformer (CPU models left fp32;
# best-effort). Cosine ~1e-3 shift, under thresholds.
export ED4ALL_EMBED_FP16=1

# ---- Seat registry (P5) ----------------------------------------------------
# Lifecycle lease OFF for this run: the orchestrator owns every seat
# transition (plan §1 — cold stop/start + coherence probe), and release_all
# at workflow end kept stopping the seats after each canary failure (10 min
# re-warm per incident). The registry line stays for documentation/tooling.
export ED4ALL_VLLM_CONTAINER_LIFECYCLE=0
# ED4ALL_SEAT_SCHEDULE stays OFF (default): the orchestrator + this operator own
# every seat transition by hand. Deliberate — a ~30 h build is not where the
# declarative seat-schedule automation gets its first live exercise.
# qwen30 seat is the -u60 clone: util 0.60 (0.35 under-budgets on unified
# memory once the glm seat's ~24 GB counts as non-torch overhead → negative KV)
export ED4ALL_VLLM_CONTAINERS="http://localhost:8001=vllm-super-final,http://localhost:8002=vllm-glmocr,http://localhost:8003=vllm-qwen30-u60"

# ---- Bloom classification v2 (2026-07-18 overhaul) -------------------------
# Re-found the bloom_classifier_disagreement gate on 3 interpretable voters
# (generator's own asserted level + zero-shot DeBERTa on the ALREADY-licensed
# MNLI singleton + verb ontology); retires the unlicensed cip29 + sentiment
# members. Zero new model weights (HF-offline).
export ED4ALL_BLOOM_TRIVOTE=1
# Post-build hook: deterministically (no LLM) harvest every artifact-asserted
# Bloom label into state/bloom_labels/labels.jsonl (the trivote voter-1 corpus).
# Best-effort — never alters final_status.
export ED4ALL_HARVEST_BLOOM_LABELS=1

# ---- Observability + checkpoints -------------------------------------------
export ED4ALL_VRAM_DOCTOR=1
export ED4ALL_LLM_TTFT_METER=1
export ED4ALL_GENERATION_CHECKPOINT=on
export ED4ALL_LOG_LEVEL=INFO                   # 2026-07-18: per-gate + dispatcher stats visibility

# ---- Timeouts (generous; graceful-stop turns timeouts into pauses) ---------
export ED4ALL_TASK_TIMEOUT_MINUTES=600
export ED4ALL_BATCH_TIMEOUT_MINUTES=3000
export ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS=600

# Gate-failure retry (owner spec 2026-07-17): course_planning re-rolls the TO
# stage with a fresh salt on a gate failure instead of killing the run; after
# 10 failed attempts it fails OPEN with a LOUD PLANNING_GATE_RETRIES_EXHAUSTED
# warning. (Landed after this run's course_planning already passed on a manual
# re-roll — active for any future course_planning execution.)
export ED4ALL_PLANNING_GATE_RETRIES=10

# ED4ALL_RUN_ID is set at launch time by the launch wrapper so every usage tap
# (incl. the new SemantiK taps) lands in one state/runs/<id>/llm_usage.jsonl.
