"""Curated campaign env-overlay allowlist for the self-orchestrating assistant.

Posture (LAW — do not weaken):

* The ``ed4all assistant`` campaign surface ORGANIZES, ARRANGES and MONITORS
  multi-book campaign runs. It NEVER authors scripts, shell strings, or
  arbitrary files. The only way it influences a run's environment is by
  composing a *validated JSON DATA overlay* (``pending-runs/<name>.json``)
  whose keys and values pass this module — no code, no interpolation, no
  free-form env.
* :func:`validate_overlay` is the single choke point. Every key must be an
  explicit member of :data:`ALLOWED_CAMPAIGN_FLAGS` (a curated set of
  build-behavior *tuning* flags — no wildcards) and every value must match
  the conservative :data:`VALUE_RE` charset (no whitespace, no shell
  metacharacters, bounded length). An unknown/forbidden key or a bad value
  is a LOUD :class:`CampaignFlagError` that names EVERY offending entry —
  nothing is ever silently dropped or coerced past the gate.
* :data:`FORBIDDEN_CAMPAIGN_FLAGS` is a documented deny-list of the things
  the assistant must never touch: path/root vars, seat/registry/launch vars,
  the ``ED4ALL_ASSISTANT_*`` seat-config vars, every ``*_API_KEY`` /
  ``*_KEY`` / ``*_BASE_URL`` secret, and the safety-critical escape hatches
  (``ED4ALL_GATE_ADVISORY``, ``LOCAL_DISPATCHER_ALLOW_STUB``,
  ``ED4ALL_EMBEDDING_ALLOW_FAKE``, ``TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS``).
  It is asserted disjoint from the allowlist by a test; membership here is
  belt-and-suspenders (any non-allowed key is rejected regardless).

The allowlist is curated from the operator campaign env template and
the documented flag tables (root ``CLAUDE.md`` cross-cutting index +
subsystem tables): build-behavior tuning only. Provider/model/base-url/key
selection stays out of the assistant's reach — Claude (dev sessions) owns
seat wiring and licensing decisions.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping


class CampaignFlagError(ValueError):
    """Raised when a campaign env overlay contains a disallowed key or value.

    The message names EVERY offending entry so a single validation surfaces
    all problems at once (never one-at-a-time whack-a-mole).
    """


# Env-var name shape: one of the four owning prefixes, then UPPER_SNAKE.
KEY_RE = re.compile(r"^(ED4ALL|TRAINFORGE|COURSEFORGE|SEMANTIK)_[A-Z0-9_]{2,64}$")

# Conservative value charset: must start alphanumeric, then a bounded run of
# a safe set (word chars, ``. / + : = , -``). No whitespace, no shell
# metacharacters (``; $ ` | & > < ( ) * ? ! \\ ' " { } [ ]``), <= 200 chars.
VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./+:=,-]{0,199}$")


# --------------------------------------------------------------------------- #
# Allowlist — curated build-behavior TUNING flags (explicit names, no wildcards)
# --------------------------------------------------------------------------- #
ALLOWED_CAMPAIGN_FLAGS: "frozenset[str]" = frozenset(
    {
        # --- two-pass generation + dynamic block planner + floors -----------
        "COURSEFORGE_TWO_PASS",
        "ED4ALL_GENERATION_TECHNIQUE",
        "ED4ALL_DYNAMIC_BLOCK_PLAN",
        # registry seat KEY for the planner (e.g. "spark-super") — a local
        # logical seat name resolved via ED4ALL_SEAT_BASE_URLS, not a cloud
        # provider/model id; explicitly campaign-tunable.
        "ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER",
        "ED4ALL_WORKED_EXAMPLE_FLOOR",
        "ED4ALL_BLOOM_SPREAD_FLOOR",
        "ED4ALL_TRIANGLE_FLOOR",
        "ED4ALL_RETRIEVAL_INTERLEAVE",
        # --- block / framework emit + quality -------------------------------
        "ED4ALL_BLOCK_ANATOMY",
        "ED4ALL_BLOCK_A11Y",
        "ED4ALL_CALLOUT_TYPED",
        "ED4ALL_NEW_BLOCK_TYPES",
        "ED4ALL_BLOCK_QUALITY_RUBRIC",
        "ED4ALL_BLOCK_QUALITY_SHADOW",
        "ED4ALL_KEY_TERMS_PAGE",
        "ED4ALL_FAQ_PAGE",
        "ED4ALL_KEYTERM_DEF_QUALITY",
        # --- Bloom distribution / classification ----------------------------
        "ED4ALL_BLOOM_DISTRIBUTION",
        "ED4ALL_BLOOM_TRIVOTE",
        "ED4ALL_ALIGNMENT_VERB_TRIPLE",
        # --- objective synthesis + dedup tuning -----------------------------
        "ED4ALL_OBJECTIVE_CITATION_RESELECT",
        "ED4ALL_OBJECTIVE_SANITIZE_CITATIONS",
        "ED4ALL_OBJECTIVE_DEDUP_THRESHOLD",
        "ED4ALL_OBJECTIVE_DEDUP_LEXICAL",
        "ED4ALL_OBJECTIVE_SPECIFICITY",
        "ED4ALL_OBJECTIVE_WINDOW_PER_SECTION",
        "ED4ALL_OBJECTIVE_WINDOW_MAX_CANDIDATES",
        "ED4ALL_OBJECTIVE_SOURCE_BACKFILL",
        "ED4ALL_OBJECTIVE_BLOOM_RELEVEL",
        "ED4ALL_SYNTHESIS_SKELETON",
        # --- terminal-objective derivation ----------------------------------
        "ED4ALL_TO_CHAPTER_ANCHOR",
        "ED4ALL_TO_CHAPTER_MIN_MODULES",
        "ED4ALL_TO_CHAPTER_MIN_CO_COVERAGE",
        "ED4ALL_TO_SOURCE_GROUNDING",
        "ED4ALL_TO_CLUSTER_K",
        "ED4ALL_TO_COS_PER_CLUSTER",
        # --- pacing / week grouping / content pages -------------------------
        "ED4ALL_COS_PER_WEEK_CAP",
        "ED4ALL_WEEK_TO_GROUPS",
        "ED4ALL_IMSCC_MODULE_TITLES",
        "ED4ALL_CONTENT_PAGE_PER_CO",
        # --- prerequisite sequencing / KG health ----------------------------
        "ED4ALL_PREREQ_SEQUENCING",
        "ED4ALL_KG_PREREQ_HEALTH",
        "ED4ALL_KG_REAL_FLOORS",
        # --- chunking tuning + chunk-health thresholds ----------------------
        "ED4ALL_CHUNK_ROLE_DIVERSIFY",
        "ED4ALL_CHUNK_SUBSECTION_BREAK",
        "ED4ALL_CHUNK_SUBSECTION_MIN_WORDS",
        "ED4ALL_CHUNK_COVERAGE_FLOOR",
        "ED4ALL_MIN_CHUNKS",
        "ED4ALL_CHUNK_HEALTH_GATE",
        "ED4ALL_CHUNK_HEALTH_CHAPTER_RATIO_FAIL",
        "ED4ALL_CHUNK_HEALTH_CHAPTER_RATIO_WARN",
        # --- resegmentation -------------------------------------------------
        "ED4ALL_RESEGMENT_COLLAPSED",
        "ED4ALL_RESEGMENT_SECTIONS_PER_CHAPTER",
        # --- NLI batching / validation performance --------------------------
        "ED4ALL_NLI_MICROBATCH",
        "ED4ALL_NLI_CROSSBLOCK",
        "ED4ALL_NLI_BUCKET_BATCHING",
        "ED4ALL_VALIDATION_FEATURE_CACHE",
        "ED4ALL_GROUNDEDNESS_FRONTIER",
        # --- embedding batching / overflow (NOT provider/base-url/key) ------
        "ED4ALL_EMBED_BATCH_TUNE",
        "ED4ALL_EMBED_PERSIST_CACHE",
        "ED4ALL_EMBED_OVERFLOW_GUARD",
        # --- read-only aggregators + coverage floors ------------------------
        "ED4ALL_CONCEPT_COVERAGE",
        "ED4ALL_INTELLIGENCE_RUBRIC",
        "ED4ALL_HARVEST_BLOOM_LABELS",
        "ED4ALL_COVERAGE_FLOOR",
        # --- timeouts / checkpoints / gate-retry ----------------------------
        "ED4ALL_TASK_TIMEOUT_MINUTES",
        "ED4ALL_PLANNING_GATE_RETRIES",
        "ED4ALL_GENERATION_CHECKPOINT",
        "ED4ALL_VALIDATION_CHECKPOINT",
        # --- authoring serving-window budgets -------------------------------
        "ED4ALL_REWRITE_NUM_CTX",
        # --- archival strictness --------------------------------------------
        "ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE",
        "ED4ALL_REQUIRE_ARCHIVED_OBJECTIVES",
        # --- SemantiK heading judge -----------------------------------------
        "SEMANTIK_HEADING_JUDGE",
        # --- Trainforge tuning (non-provider / non-key) ---------------------
        "TRAINFORGE_REQUIRE_EMBEDDINGS",
        "TRAINFORGE_PREREQ_LO_ADJACENT_ONLY",
        "TRAINFORGE_EDGE_NLI",
        "TRAINFORGE_CONTRADICTED_EDGE_POLICY",
    }
)


# --------------------------------------------------------------------------- #
# Deny-list — never in ALLOWED, asserted disjoint by a test.
# --------------------------------------------------------------------------- #
FORBIDDEN_CAMPAIGN_FLAGS: "frozenset[str]" = frozenset(
    {
        # path / root data-dir relocation ------------------------------------
        "ED4ALL_HOME",
        "ED4ALL_ROOT",
        "ED4ALL_STATE_RUNS_DIR",
        "ED4ALL_LIBV2_ROOT",
        "ED4ALL_TRAINING_CAPTURES_DIR",
        "ED4ALL_MAILBOX_BASE_DIR",
        # seat / registry / launch orchestration -----------------------------
        "ED4ALL_SEAT_BASE_URLS",
        "ED4ALL_SEAT_LAUNCH_SPECS",
        "ED4ALL_SEAT_SCHEDULE",
        "ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS",
        "ED4ALL_SEAT_COHERENCE_ATTEMPTS",
        "ED4ALL_VLLM_CONTAINERS",
        "ED4ALL_VLLM_CONTAINER_LIFECYCLE",
        "ED4ALL_ASSISTANT_SEAT_PRIORITY",
        # ED4ALL_ASSISTANT_* seat-config -------------------------------------
        "ED4ALL_ASSISTANT_BASE_URL",
        "ED4ALL_ASSISTANT_MODEL",
        "ED4ALL_ASSISTANT_AUTOSTART",
        "ED4ALL_ASSISTANT_MAX_TOKENS",
        "ED4ALL_ASSISTANT_TIMEOUT_SECONDS",
        "ED4ALL_ASSISTANT_SEAT",
        "ED4ALL_ASSISTANT_DEBUG_ON_FAILURE",
        # secrets: *_API_KEY / *_KEY / *_BASE_URL ----------------------------
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "TOGETHER_API_KEY",
        "ANTHROPIC_API_KEY",
        "LOCAL_SYNTHESIS_API_KEY",
        "ED4ALL_EMBEDDING_API_KEY",
        "ED4ALL_EMBEDDING_BASE_URL",
        "ED4ALL_ANSWER_MODEL",
        "ED4ALL_EMBEDDING_PROVIDER",
        # safety-critical escape hatches -------------------------------------
        "ED4ALL_GATE_ADVISORY",
        "LOCAL_DISPATCHER_ALLOW_STUB",
        "ED4ALL_EMBEDDING_ALLOW_FAKE",
        "TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS",
        "ED4ALL_PRODUCTION",
        "MCP_ORCHESTRATOR_LLM_MODEL",
    }
)


def is_allowed_flag(key: str) -> bool:
    """True iff ``key`` is an explicit allowlist member (and not forbidden)."""
    return key in ALLOWED_CAMPAIGN_FLAGS and key not in FORBIDDEN_CAMPAIGN_FLAGS


def validate_overlay(overlay: Mapping[str, object]) -> Dict[str, str]:
    """Validate a campaign env overlay, returning a normalized ``{str: str}`` dict.

    Every key must be an allowlist member and every value must match
    :data:`VALUE_RE`. Raises :class:`CampaignFlagError` naming EVERY offending
    entry (unknown/forbidden key -> ``"unknown or disallowed key: X"``; bad
    value -> ``"invalid value for X"``). NOTHING is silently dropped. An empty
    overlay ``{}`` is valid and round-trips to ``{}``.
    """
    if not isinstance(overlay, Mapping):
        raise CampaignFlagError(
            f"env overlay must be a mapping, got {type(overlay).__name__}"
        )

    errors = []
    normalized: Dict[str, str] = {}
    for key, value in overlay.items():
        if not isinstance(key, str) or not is_allowed_flag(key):
            errors.append(f"unknown or disallowed key: {key!r}")
            continue
        svalue = value if isinstance(value, str) else str(value)
        if not VALUE_RE.match(svalue):
            errors.append(f"invalid value for {key}: {svalue!r}")
            continue
        normalized[key] = svalue

    if errors:
        raise CampaignFlagError("; ".join(errors))
    return normalized
