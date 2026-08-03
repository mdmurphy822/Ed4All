#!/bin/sh
# Public environment template for an Ed4All textbook-to-course run.
#
# Copy this file to the ignored local profile below, fill in values for the
# model service you operate, and source that copy before invoking Ed4All:
#
#   cp docs/operations/run-env.example.sh docs/operations/run-env.local.sh
#   . docs/operations/run-env.local.sh
#   ed4all run textbook-to-course \
#     --corpus <PRIVATE_CORPUS_PATH> \
#     --course-name <PRIVATE_COURSE_NAME>
#
# Keep endpoints, credentials, model paths, course names, and host tuning
# in the ignored local copy. Do not commit runtime inputs or outputs.
#
# Configuration references:
#   docs/operations/pipeline-invocation.md  Invocation, resume, and stop
#   docs/operations/full-run-playbook.md    End-to-end operating procedure
#   docs/operations/behavior-flags.md       Cross-cutting behavior flags
#   docs/LICENSING.md                       Provider and model licensing
#   SemantiK/CLAUDE.md                      Conversion configuration
#   Courseforge/CLAUDE.md                   Course generation configuration
#   Trainforge/CLAUDE.md                    Synthesis configuration

# ---------------------------------------------------------------------------
# Required local model-service identity
# ---------------------------------------------------------------------------

# Set these in docs/operations/run-env.local.sh. The URL must be an
# OpenAI-compatible endpoint operated for this private workflow; the model ID
# must exactly match an ID returned by that service's models endpoint.
export LOCAL_SYNTHESIS_BASE_URL="${LOCAL_SYNTHESIS_BASE_URL:-<LOCAL_MODEL_BASE_URL>}"
export LOCAL_SYNTHESIS_MODEL="${LOCAL_SYNTHESIS_MODEL:-<SERVED_MODEL_ID>}"

# Leave the key empty only when the local service does not require one.
export LOCAL_SYNTHESIS_API_KEY="${LOCAL_SYNTHESIS_API_KEY:-}"

# API mode routes the public workflow through the configured local provider.
# Agent-session execution can instead be selected with `--mode local` without
# changing this template.
export LLM_MODE="${LLM_MODE:-api}"
export LLM_PROVIDER="${LLM_PROVIDER:-local}"
export LLM_MODEL="${LLM_MODEL:-${LOCAL_SYNTHESIS_MODEL}}"

# Keep every content-producing family on the explicitly configured local
# provider. See docs/LICENSING.md before choosing a different provider.
export COURSEFORGE_PROVIDER="${COURSEFORGE_PROVIDER:-local}"
export COURSEPLANNER_PROVIDER="${COURSEPLANNER_PROVIDER:-local}"
export TEXTBOOK_SYNTHESIS_PROVIDER="${TEXTBOOK_SYNTHESIS_PROVIDER:-local}"
export COURSEFORGE_OUTLINE_PROVIDER="${COURSEFORGE_OUTLINE_PROVIDER:-local}"
export COURSEFORGE_REWRITE_PROVIDER="${COURSEFORGE_REWRITE_PROVIDER:-local}"
export TRAINFORGE_ASSESSMENT_PROVIDER="${TRAINFORGE_ASSESSMENT_PROVIDER:-local}"
export TRAINFORGE_SYNTHESIS_PROVIDER="${TRAINFORGE_SYNTHESIS_PROVIDER:-local}"

# Explicit model pins prevent provider-family defaults from selecting a model
# that the configured service does not expose.
export TEXTBOOK_SYNTHESIS_MODEL="${TEXTBOOK_SYNTHESIS_MODEL:-${LOCAL_SYNTHESIS_MODEL}}"
export COURSEPLANNER_MODEL="${COURSEPLANNER_MODEL:-${LOCAL_SYNTHESIS_MODEL}}"
export COURSEFORGE_OUTLINE_MODEL="${COURSEFORGE_OUTLINE_MODEL:-${LOCAL_SYNTHESIS_MODEL}}"
export COURSEFORGE_REWRITE_MODEL="${COURSEFORGE_REWRITE_MODEL:-${LOCAL_SYNTHESIS_MODEL}}"
export TRAINFORGE_ASSESSMENT_MODEL="${TRAINFORGE_ASSESSMENT_MODEL:-${LOCAL_SYNTHESIS_MODEL}}"

# ---------------------------------------------------------------------------
# Staged training-pair synthesis
# ---------------------------------------------------------------------------

# Staged synthesis is the production workflow for evidence planning, SFT
# realization, and DPO realization.
export TRAINFORGE_STAGED_SYNTHESIS_V4="${TRAINFORGE_STAGED_SYNTHESIS_V4:-true}"
export TRAINFORGE_REQUIRE_EMBEDDINGS="${TRAINFORGE_REQUIRE_EMBEDDINGS:-true}"

# Set this to the context window actually served by the configured model. The
# staged preflight intentionally rejects a missing or non-positive value.
export TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS="${TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS:-<SERVED_CONTEXT_TOKENS>}"

# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------

# Enable the GLM-OCR extraction lane. Configure its private endpoint and model
# in the ignored local profile; consult SemantiK/CLAUDE.md for the full lane.
export SEMANTIK_GLMOCR_LANE="${SEMANTIK_GLMOCR_LANE:-1}"
export SEMANTIK_GLMOCR_BASE_URL="${SEMANTIK_GLMOCR_BASE_URL:-<LOCAL_OCR_BASE_URL>}"
export SEMANTIK_GLMOCR_MODEL="${SEMANTIK_GLMOCR_MODEL:-<SERVED_OCR_MODEL_ID>}"

# ---------------------------------------------------------------------------
# Portable privacy and recovery defaults
# ---------------------------------------------------------------------------

# Prevent model hubs from fetching content during an intentionally offline run.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# Persist per-unit checkpoints so stop and resume preserve completed work.
export ED4ALL_GENERATION_CHECKPOINT="${ED4ALL_GENERATION_CHECKPOINT:-on}"
export ED4ALL_LOG_LEVEL="${ED4ALL_LOG_LEVEL:-INFO}"

# Host-specific performance and service-lifecycle values belong only in
# docs/operations/run-env.local.sh. Start from documented defaults, run
# `ed4all doctor`, and tune the ignored profile for the host.
