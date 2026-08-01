# ============================================================================
# run-env.example.sh — environment template for a full `textbook_to_course` run
#
# Source this BEFORE `ed4all run`:
#
#     source ./run-env.example.sh
#     ed4all run textbook-to-course --corpus ./inputs/<DIR> --course-name <NAME>
#
# This file is a TEMPLATE, not a tuned profile. It is organised into three
# clearly separated sections, and they are NOT interchangeable:
#
#   §1 PORTABLE / REQUIRED    — needed by any run on any box. Copy as-is.
#   §2 HARDWARE-PROFILE       — concurrency, device pins, timeouts, GPU
#                               lifecycle. These values were tuned for a
#                               SINGLE-GPU BIG-MEMORY host (~128 GB unified
#                               memory, one card). On a small discrete GPU they
#                               are actively WRONG and will OOM. Every entry
#                               carries scaling guidance. READ BEFORE COPYING.
#   §3 VALIDATED MEASUREMENTS — observations recorded from completed runs.
#                               Provenance notes, NOT prescriptions.
#
# Per-flag semantics live in the flag tables, not here:
#   docs/operations/behavior-flags.md   (ED4ALL_* cross-cutting)
#   SemantiK/CLAUDE.md                  (SEMANTIK_*)
#   Courseforge/CLAUDE.md               (COURSEFORGE_* / COURSEPLANNER_* /
#                                        TEXTBOOK_SYNTHESIS_*)
#   Trainforge/CLAUDE.md                (TRAINFORGE_* / seat registry keys)
#   docs/operations/pipeline-invocation.md  (per-stage invocation, resume, stop)
#   docs/operations/full-run-playbook.md    (the end-to-end run procedure)
#
# No absolute paths, hostnames, or credentials are baked in: paths are derived
# from this file's own location, and every host/port is a shell variable you can
# override before sourcing. The §2 VALUES are still hardware-specific — see the
# warning on that section.
# ============================================================================

# Repo root, derived from this script — works from any checkout and any cwd.
_ED4ALL_PROFILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Canonical training-pair workflow: separate evidence planning from SFT and
# DPO realization. An explicit operator value wins; unset uses the validated
# staged path for this full-run profile.
export TRAINFORGE_STAGED_SYNTHESIS_V4="${TRAINFORGE_STAGED_SYNTHESIS_V4:-true}"

# ---- Staged synthesis preconditions (REQUIRED when TRAINFORGE_STAGED_SYNTHESIS_V4 is on)
#
# Staged synthesis validates THREE preconditions at orchestration time, before
# any synthesis output is created. All three must be satisfied, and they cannot
# be defaulted by code — they are operator-declared environment variables only.
# Missing or invalid values fail the run LOUDLY before the first generation call.
#
# 1. The served model context-window size (tokens). YOU MUST SET THIS — this
#    template deliberately supplies NO default. The value is per-deployment
#    (it is whatever window the seat was actually launched with, e.g. vLLM
#    --max-model-len / TRT-LLM --max_seq_len); read it off the running seat's
#    serve command or /v1/models. A guessed value silently satisfies a
#    fail-closed guard with the wrong window, which defeats the guard, so an
#    unset value is left to fail LOUDLY before the first generation call.
#    Synthesis validates it is positive and uses it to compute per-generation
#    completion budgets.
#    Failure message: "TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS must be a
#    positive measured served model window before staged synthesis starts"
export TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS="${TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS:-}"
if [ -z "${TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS}" ]; then
    echo "run-env.example.sh: WARNING — TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS is unset." >&2
    echo "  Staged synthesis will fail closed before its first generation call." >&2
    echo "  Set it to your seat's measured context window and re-source:" >&2
    echo "    TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS=<tokens> source ./run-env.example.sh" >&2
fi

# 2. An embedding backend (sentence-transformers or HuggingFace) is required
#    for evidence-grounded SFT pair selection. Synthesis requires this to be
#    explicitly enabled (no automatic fallback). The embedder is loaded once
#    at orchestration time — a failed load fails the run before synthesis.
#    Failure message: "TRAINFORGE_REQUIRE_EMBEDDINGS=true is required for
#    staged production synthesis; Jaccard fallback is not permitted"
export TRAINFORGE_REQUIRE_EMBEDDINGS="${TRAINFORGE_REQUIRE_EMBEDDINGS:-true}"

# 3. The exact model ID served at LOCAL_SYNTHESIS_BASE_URL. Staged synthesis
#    validates that LOCAL_SYNTHESIS_MODEL is present in the id list reported by
#    the server's OpenAI-compatible /v1/models endpoint. This prevents silent
#    model-mismatch 404s mid-phase.
#    Failure message (verbatim from
#    Trainforge/synthesize_training.py::_preflight_local_staged_model_identity):
#    "configured staged synthesis model <id> is not served by <base>/models;
#    served ids=[...]"
#    A seat launched WITHOUT --served-model-name reports its checkpoint path as
#    the id, not a friendly name — e.g. the TRT-LLM seat in
#    runtime/seats/launch-super-trtllm.sh serves the snapshot directory. Read the id off
#    /v1/models and set it here; a guessed friendly name fails the preflight.
#    NB: the two values below MUST describe the SAME seat as the context tokens
#    above. The defaults name the Nano seat; if you point base_url at another
#    seat (e.g. the Super seat on :8123), set the model id AND the context
#    tokens above to THAT seat's launched window — they differ by more than an
#    order of magnitude between seats, so a copied number is a wrong number.
export LOCAL_SYNTHESIS_BASE_URL="${LOCAL_SYNTHESIS_BASE_URL:-http://127.0.0.1:8000/v1}"
export LOCAL_SYNTHESIS_MODEL="${LOCAL_SYNTHESIS_MODEL:-nemotron-3-nano-30b-a3b}"

# Hugging Face executes pinned repository-local model code from a generated
# Python module cache. Some system/container installs leave the library's
# default ~/.cache/huggingface/modules tree owned by another user; a later
# trust_remote_code load then fails only after the multi-hour pipeline reaches
# training. An explicit operator value always wins. Otherwise use an Ed4All-
# owned XDG cache sibling, separate from the model-weight cache.
if [ -z "${HF_MODULES_CACHE:-}" ]; then
    _ED4ALL_XDG_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}"
    export HF_MODULES_CACHE="${_ED4ALL_XDG_CACHE_ROOT}/ed4all/huggingface/modules"
fi
if ! mkdir -p "${HF_MODULES_CACHE}" || [ ! -w "${HF_MODULES_CACHE}" ]; then
    echo "run-env.example.sh: ERROR — HF_MODULES_CACHE is not writable:" >&2
    echo "  ${HF_MODULES_CACHE}" >&2
    echo "  Set HF_MODULES_CACHE to a writable directory and re-source." >&2
    return 1 2>/dev/null || exit 1
fi

# Model-seat endpoints. Ports are DEFAULTS, not reservations — override any of
# these in your shell before sourcing if they collide with something else on
# the box. All three may be the same URL if you serve one seat.
_SEAT_MAIN_URL="${_SEAT_MAIN_URL:-http://localhost:8001/v1}"   # authoring/planning seat
_SEAT_OCR_URL="${_SEAT_OCR_URL:-http://localhost:8002/v1}"     # document-extraction seat
_SEAT_VLM_URL="${_SEAT_VLM_URL:-http://localhost:8003/v1}"     # vision / alt-text seat

# The served-model-name your main seat answers to. A vLLM seat launched with
# `--served-model-name X` 404s on any other id, and several providers fall back
# to their own default model id when only *_PROVIDER is set — so the model pins
# below are mandatory, not decorative.
#
# There is no safe default here: a wrong model id does not fail at source time,
# it 404s on the first dispatch of a multi-hour phase. So this warns LOUDLY at
# source time rather than letting a placeholder propagate into five pins.
_SEAT_MAIN_MODEL="${_SEAT_MAIN_MODEL:-}"
if [ -z "${_SEAT_MAIN_MODEL}" ]; then
    _SEAT_MAIN_MODEL="UNSET-set-_SEAT_MAIN_MODEL-before-sourcing"
    echo "run-env.example.sh: WARNING — _SEAT_MAIN_MODEL is unset." >&2
    echo "  Every model pin below is now a placeholder that will 404 at the seat." >&2
    echo "  Set it to your seat's --served-model-name and re-source:" >&2
    echo "    _SEAT_MAIN_MODEL=<served-model-name> source ./run-env.example.sh" >&2
fi


# ============================================================================
# §1  PORTABLE / REQUIRED
# ============================================================================

# ---- License-clean, fully local: no cloud provider reachable ---------------
# Training-pair synthesis is a derivative work of whatever model produced it
# (docs/LICENSING.md). Unsetting these guarantees no run silently reaches a
# hosted seat.
unset ANTHROPIC_API_KEY NVIDIA_API_KEY || true

# Hub fully offline. Models must already be in the local HF cache. This is not
# only a privacy posture: `transformers` fails hard on a stale/expired hub
# token rather than falling back to the cache, so a half-online box breaks
# model load at run time. Seed the cache first, then run offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# ---- LLM routing ----------------------------------------------------------
# `api` mode dispatches phase workers as in-process coroutines against an
# OpenAI-compatible endpoint. Providers are rows in config/endpoints.yaml —
# never subclasses; a new seat is a registry entry.
export LLM_MODE=api
export LLM_PROVIDER=spark-super
export SPARK_BASE_URL="${_SEAT_MAIN_URL}"
export SPARK_API_KEY=local
export SPARK_SUPER_MODEL="${_SEAT_MAIN_MODEL}"

# Per-family provider pins. Each of these families resolves its provider
# independently; leaving one unset drops that family onto the default provider.
export COURSEFORGE_PROVIDER=spark-super
export COURSEPLANNER_PROVIDER=spark-super
export TEXTBOOK_SYNTHESIS_PROVIDER=spark-super
export COURSEFORGE_OUTLINE_PROVIDER=spark-super
export COURSEFORGE_REWRITE_PROVIDER=spark-super
export TRAINFORGE_ASSESSMENT_PROVIDER=spark-super

# Per-family MODEL pins. REQUIRED alongside the provider pins: with only
# *_PROVIDER set these families fall back to the LOCAL_SYNTHESIS default model
# id, which 404s against a seat serving a different `--served-model-name`.
export TEXTBOOK_SYNTHESIS_MODEL="${_SEAT_MAIN_MODEL}"
export COURSEPLANNER_MODEL="${_SEAT_MAIN_MODEL}"
export COURSEFORGE_OUTLINE_MODEL="${_SEAT_MAIN_MODEL}"
export COURSEFORGE_REWRITE_MODEL="${_SEAT_MAIN_MODEL}"
export TRAINFORGE_ASSESSMENT_MODEL="${_SEAT_MAIN_MODEL}"

# Capability-tier routing. Precedence in the router (Courseforge/router/
# router.py::_resolve_spec) is: per-call override > tier env vars > YAML policy
# > hardcoded default. The tier env vars win — BUT they carry only `provider`
# and `model`, and only for the outline/rewrite tiers. Every OTHER field in a
# policy entry survives the env overlay, `base_url` included. So a policy file
# whose tiers pin a base_url at a seat you are not serving still routes there
# despite the env pins above, and 404s mid-phase.
#
# If you serve ONE seat, make all three capability tiers name that seat's URL
# and model. Copy a tracked profile and edit the three tiers:
#     Courseforge/config/block_routing.yaml               (canonical)
#     Courseforge/config/block_routing.license_clean.yaml (local-only large tier)
# then point at your copy. Path derived from this file, so it stays portable:
# export COURSEFORGE_BLOCK_ROUTING_PATH="${_ED4ALL_PROFILE_ROOT}/Courseforge/config/block_routing.license_clean.yaml"

# Reasoning-model control. Only relevant when your seat serves a model that
# emits reasoning tokens: it injects a thinking-off directive plus
# `chat_template_kwargs.enable_thinking=false` so reasoning output does not
# consume the completion budget and trip the finish_reason=length guard.
# Harmless on a non-reasoning seat.
export ED4ALL_REASONING_THINKING_OFF=1

# ---- Document conversion lane (SemantiK) -----------------------------------
export SEMANTIK_GLMOCR_LANE=1            # OCR-extraction lane
export SEMANTIK_ALTTEXT_PROVIDER=qwen30  # VLM alt text; `off` (the default)
                                         # ships harvested captions + honest
                                         # placeholders, never fabricated text
export SEMANTIK_ARRANGER_ID_REPAIR=1     # deterministic block-id repair
export SEMANTIK_TITLE_SANITIZE=true
# Seat endpoints/models for the two conversion seats. Defaults in code are
# localhost:8002 / localhost:8003; set explicitly so a port override propagates.
export SEMANTIK_GLMOCR_BASE_URL="${_SEAT_OCR_URL}"
export SEMANTIK_ALTTEXT_BASE_URL="${_SEAT_VLM_URL}"

# Heading-level judge over the extraction lane's layout sidecars. Skip-with-pass
# when off or when the corpus is born-digital; per-chapter FAIL-OPEN, so it can
# never block a build. It dispatches to the MAIN seat, which on a single-GPU box
# cannot co-reside with the two conversion seats — sequence it as a pass after
# the conversion seats are released, not concurrently with them.
export SEMANTIK_HEADING_JUDGE=1

# ---- Structure fidelity over the converted HTML ----------------------------
export ED4ALL_STRUCTURE_EXTRACT_GUARDS=1
export ED4ALL_STRUCTURE_OUTLINE_ANCHOR=1

# ---- Chunking --------------------------------------------------------------
export TRAINFORGE_DROP_FRONTMATTER=true
export TRAINFORGE_DROP_APPARATUS_DUMPS=true
export TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE=true
export ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR=20
# Size-guarded sub-section breaks: split at a genuine content sub-section
# heading only once the buffer clears the min-words floor. The unguarded
# variant (ED4ALL_CHUNK_SECTION_HARD_BREAK) is deliberately NOT set — see §3.
export ED4ALL_CHUNK_SUBSECTION_BREAK=1
export ED4ALL_CHUNK_SUBSECTION_MIN_WORDS=250
# Pre-synthesis health gate: audits the chunkset + textbook structure for
# synthesis-poisoning defects BEFORE course_planning spends hours on them.
export ED4ALL_CHUNK_HEALTH_GATE=1
# Thin-chunkset floor on the chunkset_manifest gate. Off by default, which
# means a corpus that silently collapses to a handful of chunks still passes.
# Set it to a per-corpus floor you would be alarmed to fall below — it is the
# only signal that makes a THINNED chunkset visible, and it is what you want in
# place before ever enabling ED4ALL_CHUNK_DEDUP (which deletes content).
# export ED4ALL_MIN_CHUNKS=250

# Within-package exact-normalized dedup. DEFAULT OFF and left off here: it
# deletes duplicate content before chunk IDs are minted, so enabling it is an
# owner decision gated on a measured drop rate plus a held-out retrieval A/B
# (plans/cuda-html-libv2-retrieval-spark-optimization-2026-07.md § B5). Drops
# are recorded in source_coverage.drop_reasons + a dedup_ledger.jsonl sidecar.
# export ED4ALL_CHUNK_DEDUP=1
# export ED4ALL_CHUNK_DEDUP_MIN_TOKENS=8

# ---- Course structure ------------------------------------------------------
export ED4ALL_TO_CHAPTER_ANCHOR=1            # one source module -> one terminal objective
export ED4ALL_WEEK_TO_GROUPS=1               # week N = TO-N membership
export ED4ALL_IMSCC_MODULE_TITLES=to         # "Module N: <TO topic>" org titles
export ED4ALL_CONTENT_PAGE_PER_CO=1
export ED4ALL_CONTENT_PAGE_PER_CO_UNCAPPED=1 # true one-page-per-CO; authoring
                                             # cost is O(total COs) — this is
                                             # the single biggest lever on
                                             # total run time
export ED4ALL_SYNTHESIS_SKELETON=1           # inject the chapter heading
                                             # skeleton into per-window
                                             # objective synthesis
export ED4ALL_OBJECTIVE_WINDOW_PER_SECTION=1 # per-SECTION synthesis windows
# NOTE: both of the two flags above fold into the synthesis-window fingerprint.
# Flipping either mid-course invalidates the window resume sidecars and those
# windows re-run on --resume. Harmless on a fresh build.

# ---- Two-pass content generation ------------------------------------------
export COURSEFORGE_TWO_PASS=true
export COURSEFORGE_REWRITE_HTML_REPAIR=1     # deterministic tag-balance repair
                                             # at rewrite emit
export COURSEFORGE_CURIE_PRESERVE_SKIP_WHEN_POSTMINT=1  # skip the CURIE
                                             # preservation re-roll ladder; the
                                             # deterministic postmint already
                                             # anchors exactly the enforceable set

# ---- Validation architecture (all verdict-identical to the serial path) ----
export ED4ALL_GROUNDEDNESS_FRONTIER=1        # priority-ordered per-claim
                                             # early-stop in the NLI scorer
export ED4ALL_VALIDATION_FEATURE_CACHE=1     # compute gate inputs once for the
                                             # whole gate suite instead of per gate
export ED4ALL_NLI_BUCKET_BATCHING=1          # token-length bucketed NLI batches
export ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE=1 # resolve cited provenance refs to
                                             # premises, so the entailment gate
                                             # actually has something to score
# The spawn-ProcessPool validator path is superseded by the cross-block
# dispatcher in §2 and is left OFF deliberately.
export ED4ALL_NLI_MICROBATCH_VALIDATORS=0

# ---- Bloom classification --------------------------------------------------
export ED4ALL_BLOOM_TRIVOTE=1                # 3 interpretable voters; no new
                                             # model weights
export ED4ALL_HARVEST_BLOOM_LABELS=1         # post-build, no-LLM label harvest

# ---- Checkpoints, observability, graceful degradation ----------------------
export ED4ALL_GENERATION_CHECKPOINT=on       # per-unit resume sidecars — this
                                             # is what makes `ed4all stop` and
                                             # `--resume` near-lossless
export ED4ALL_VRAM_DOCTOR=1
export ED4ALL_LLM_TTFT_METER=1
export ED4ALL_LOG_LEVEL=INFO
export ED4ALL_CAPTURE_BUFFER=1               # batch decision-capture writes
# course_planning re-rolls the objective stage with a fresh salt on a gate
# failure instead of killing a multi-hour run; after the budget is exhausted it
# fails OPEN with a loud PLANNING_GATE_RETRIES_EXHAUSTED warning.
export ED4ALL_PLANNING_GATE_RETRIES=10


# ============================================================================
# §2  HARDWARE-PROFILE  —  READ BEFORE COPYING
#
# Tuned for ONE GPU with ~128 GB unified memory. On a small discrete card
# (8-24 GB) these values will OOM or thrash. Scale each one as noted.
# `ed4all doctor` reports the box's GPU profile and flags small-box defaults
# that are still on; run it before a long build.
# ============================================================================

# ---- Inference device ------------------------------------------------------
# Any GPU-capable inference belongs on the GPU; CPU here is a misconfiguration,
# not a conservative choice. Set both to `cpu` ONLY on a box with no CUDA
# device — expect the validation phases to take multiples of the GPU timing.
#
# ED4ALL_EMBEDDING_DEVICE now defaults to `cuda` in CODE as well (index builds,
# query encoding, AND the validator-tier embedder); it is kept explicit here so
# the recorded provenance names the operator's choice. There is NO automatic
# CUDA->CPU fallback: on a GPU-less box you MUST set `cpu` yourself or the
# embedding paths raise. ED4ALL_NLI_DEVICE still degrades gracefully; the
# embedding knob does not.
export ED4ALL_NLI_DEVICE=cuda
export ED4ALL_EMBEDDING_DEVICE=cuda

# ---- Encoder precision (default fp32) --------------------------------------
# ED4ALL_EMBEDDING_DTYPE selects the ENCODER compute precision (fp32|bf16|fp16);
# the persisted matrix stays float32 either way. Non-fp32 also enables TF32
# matmul PROCESS-GLOBALLY, so it affects every torch consumer in the
# interpreter. Left at the fp32 default until the staged bf16-vs-fp32 drift
# benchmark runs (plans/cuda-html-libv2-retrieval-spark-optimization-2026-07.md
# § B2). Uncomment only after that measurement:
# export ED4ALL_EMBEDDING_DTYPE=bf16

# ---- Staged-HTML parse pool ------------------------------------------------
# Parse-pool size for the chunking phases. Pure-stdlib html.parser work — no
# torch, no GPU — so it scales with cores, not VRAM. The code default is 10 so
# two concurrent `ed4all run` invocations cannot put 40 processes on 20 cores;
# a single quiescent build on this 20-core profile measured its optimum at 20.
#
# SCALING: set this to the box's core count for a single build; drop it to
# `os.cpu_count() / 2` when two builds share the host, and to 1 (or 0) for the
# byte-identical serial path.
#
# STOP-LOSS: with a pool active, `ed4all stop` during a parse phase drains
# roughly 3 x this many in-flight files instead of one unit. Nothing is lost
# but re-parse time.
export ED4ALL_HTML_PARSE_WORKERS=20

# ---- GPU lifecycle ---------------------------------------------------------
# BOTH of these default to ON in code, and ON is the correct setting on a small
# card: the phase-boundary sweep releases the card between stages, and the NLI
# path evicts a resident serving model so the classifier can load.
#
# They are turned OFF here because a big-memory box can hold a resident serving
# seat AND the NLI/embedding models simultaneously, and eviction would force a
# multi-minute model reload at every phase boundary.
#
# SCALING: on anything under ~48 GB, DELETE these two lines (restore the ON
# defaults). Leaving them at 0 on a small card means the serving seat is never
# released and the NLI/embedding load hits an already-full card — an OOM, not a
# slowdown.
export ED4ALL_GPU_LIFECYCLE=0
export ED4ALL_NLI_EVICT_FOR_CUDA=0

# ---- Authoring concurrency -------------------------------------------------
# Threads dispatching blocks at the authoring seat. 32 was measured at the knee
# on a seat with a large KV budget.
#
# SCALING: this is bounded by the SEAT's concurrent-request capacity (KV cache),
# not by the client. A small seat serving a 7B model at a 4-8k window typically
# saturates at 4-8. Over-setting does not OOM the client — it queues at the seat
# and inflates per-request latency until requests time out.
export COURSEFORGE_OUTLINE_CONCURRENCY=32
export COURSEFORGE_REWRITE_CONCURRENCY=32

# ---- Conversion lane workers ----------------------------------------------
# Parallel workers against the extraction seat (code default is also 4).
# SCALING: raise only if the extraction seat has spare capacity; each worker
# holds page rasters in host memory, so on a memory-tight box drop to 1-2.
export SEMANTIK_GLMOCR_WORKERS=4

# ---- NLI entailment concurrency -------------------------------------------
# Cross-block fan-out for the prose-entailment gate: cache-miss blocks are
# spread across a thread pool onto ONE coalescing GPU dispatcher (no extra CUDA
# contexts). Verdict-identical; the dispatcher halves its drain on CUDA OOM.
#
# SCALING: THREADS is capped internally at the scorable-miss count, so it is
# safe to leave. MAX_PAIRS is the real memory lever — it sets how many pairs are
# coalesced into one forward pass. On a small card drop MAX_PAIRS to 32 or 16.
export ED4ALL_NLI_MICROBATCH=1
export ED4ALL_NLI_CROSSBLOCK=1
export ED4ALL_NLI_CROSSBLOCK_THREADS=16
export ED4ALL_NLI_MICROBATCH_MAX_PAIRS=128

# Per-bucket NLI forward-pass batch sizes, shortest-first, aligned to the
# <=128 / 256 / 384 / 512-token buckets. Must parse to exactly four positive
# ints or the whole default list ("256,128,64,32") is used.
#
# SCALING: activation cost is O(L^2) per pair, so the LAST entry (the 512-token
# bucket) dominates peak VRAM. On a small card halve every entry, e.g.
# "128,64,32,16", and re-measure before raising.
export ED4ALL_NLI_BUCKET_BATCH="256,128,64,64"

# ---- Embedding throughput --------------------------------------------------
# BATCH_TUNE raises the encode batch and packs longest-first; FP16 casts a
# CUDA-resident embedder to half precision (CPU models are left fp32). Both
# shift cosine only in low-order bits, well under every downstream threshold.
# PERSIST_CACHE is a content+model-addressed disk cache — it is pure win on any
# box and costs only disk.
#
# SCALING: on a small card drop ED4ALL_EMBED_BATCH_TUNE (the larger batch is the
# memory cost). Keep FP16 and PERSIST_CACHE.
export ED4ALL_EMBED_BATCH_TUNE=1
export ED4ALL_EMBED_FP16=1
export ED4ALL_EMBED_PERSIST_CACHE=1

# ---- Serving window --------------------------------------------------------
# Token budget assumed when packing retrieval/authoring prompts. This must sit
# BELOW your seat's true context window with headroom, or prompts silently
# head-truncate at the seat.
#
# SCALING: set it from YOUR seat. The code default is 4096. The value below
# assumes a 32k-window seat with headroom. A seat launched at 8k needs roughly
# 7000 here, not this number.
export ED4ALL_ANSWER_NUM_CTX=28672

# ---- Timeouts --------------------------------------------------------------
# Deliberately generous. Graceful stop turns a batch timeout into a checkpointed
# pause rather than a failure, so an over-long timeout costs little; an
# under-set one kills a phase mid-flight.
#
# SCALING: a slower box needs LARGER values, not smaller. The dominant single
# phase on a full build is validation, which can run hours.
export ED4ALL_TASK_TIMEOUT_MINUTES=600
export ED4ALL_BATCH_TIMEOUT_MINUTES=3000
export ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS=600

# ---- Seat lifecycle --------------------------------------------------------
# Both OFF here: the operator owns every seat transition by hand on this
# profile. Automated, pipeline-owned seat lifecycle (stop/start per phase, with
# liveness + coherence probes and cold-recreate self-heal) is configured
# separately — source docs/operations/seat-schedule.env.example alongside this
# file. Do not enable it mid-build.
export ED4ALL_VLLM_CONTAINER_LIFECYCLE=0
# ED4ALL_SEAT_SCHEDULE is left at its OFF default.
#
# SEAT TOPOLOGY on a single-GPU box: the conversion seats (extraction + VLM)
# and the large authoring seat CANNOT co-reside. Run conversion first with the
# two conversion seats up, release them, then bring up the authoring seat for
# everything downstream. GPU-memory-utilization fractions are vLLM LAUNCH
# arguments, not env vars, so they are not settable here — budget them so the
# co-resident pair sums to well under 1.0, and remember that a co-resident
# seat's footprint counts as non-torch overhead against the other seat's
# fraction on unified memory.
#
# ALWAYS content-probe a seat after starting it. A seat can answer /v1/models
# with 200 and still be mode-collapsed; liveness is not coherence.


# ============================================================================
# §3  VALIDATED MEASUREMENTS
#
# OBSERVATIONS from completed runs on the big-memory host described in §2.
# They are recorded to explain WHY a setting above is what it is. They are NOT
# targets, NOT guarantees, and they will not reproduce on different hardware or
# a different corpus. Do not treat any number here as a threshold.
# ============================================================================
#
# Extraction lane
#   - The extraction lane at 4 workers was measured at ~4,453 pages/hr. This is
#     also the source of the code default of 4.
#
# Chunking
#   - The unguarded section-hard-break flag was A/B'd against off on one
#     converted chapter: 498 chunks / p50 42 words / 208 runt chunks with it on,
#     vs 72 chunks / p50 755 words / 0 runts with it off. It flushed far past
#     the boundary headings it was meant to match. That is why §1 sets the
#     size-guarded successor instead.
#
# Rewrite emit
#   - Deterministic HTML tag-balance repair, measured on one 424-block export:
#     the HTML-shape gate score moved 0.3519 -> 0.9612 and all 262 unbalanced
#     blocks were repaired.
#   - Postmint-aware CURIE skip, observed on one run: 0 preservation re-rolls
#     vs 803 with the retry ladder active.
#
# Validation
#   - Frontier early-stop in the groundedness scorer was compared against the
#     serial baseline on one export: verdict-identical across all scored pairs,
#     with ~68% fewer pair scorings.
#   - On a torch NLI microbenchmark, 512-token forwards ran ~80.8 pairs/s at
#     batch 64 vs ~68.1 pairs/s at batch 32 (~+19%). That is the basis for the
#     raised long-pair bucket in §2.
#
# Whole-run shape (one completed production build: 21 declared phases, 19
# executed, roughly a dozen source documents, two-pass authoring, training
# synthesis skipped)
#   - Sum of per-phase durations: ~7 hours. Wall-clock span was substantially
#     longer because of operator gaps between resumed segments.
#   - The four dominant phases, in order: post-rewrite validation (~1h51m),
#     outline tier (~1h44m), rewrite tier (~1h16m), document conversion
#     (~1h09m). Assessment synthesis was ~49m. Everything else was minutes or
#     seconds.
#   - Budget accordingly: over 80% of a full build is spent in four phases, and
#     the largest single one is validation, not generation.
