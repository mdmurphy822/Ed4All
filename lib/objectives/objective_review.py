"""Grounding-safe objective-review pass (``course_planning`` phase).

After the local 7B synthesizes course objectives in the ``course_planning``
phase, a strong hosted model (NVIDIA Nemotron, wired as the ``nvidia``
content-generation provider) REVIEWS and ADJUSTS the synthesized objectives'
*quality* IN PLACE — improving statement clarity, repairing ``bloom_level`` /
``bloom_verb`` mismatches, completing the ``abcd`` fields, and making each TO
statement genuinely cover its child COs.

This is a QUALITY review, **NOT a re-synthesis**. Hard guardrails:

1. **Editable surface only.** The model receives only
   ``{id, statement, bloom_level, bloom_verb, abcd, terminal_id}`` per
   objective (``terminal_id`` is the CO->TO MAPPING, sent for COs only).
   ``source_refs`` / ``chunk_ids`` are NEVER sent and NEVER mutated.
2. **Identity + structure are IMMUTABLE — but the CO->TO MAPPING is now
   re-pointable (Wave-2 Part 2 relaxation).** The pass never adds, removes,
   reorders, merges, or re-ids objectives — downstream chunk
   ``learning_outcome_refs[]`` continuity depends on the id SET (not on which
   TO a CO rolls up to). A returned id not in the original set is dropped; an
   omitted id keeps its original objective verbatim. The reviewer MAY change a
   CO's ``terminal_id`` to a DIFFERENT *existing* TO id (fixing weak CO->TO
   alignment); a ``terminal_id`` naming a non-existent TO is dropped (CO keeps
   its current parent). Only the editable text fields + the guardrailed remap
   are merged back by id.
3. **Grounding guardrail (load-bearing).** For every CO whose ``statement``
   the model changed, the pass recomputes ``cosine(adjusted_CO.statement,
   assigned_TO.statement)`` (reusing the SAME ``co_terminal_alignment`` floor
   / ``backlink_cos_to_tos`` cosine signal). If the adjusted statement scores
   BELOW the floor OR materially worse than the original, the CO is REVERTED
   to its original statement / bloom / abcd. A bloom change is reverted when
   ``abcd.behavior.verb`` falls outside ``BLOOMS_VERBS[bloom_level]`` (the
   ``abcd_verb_alignment`` gate rule). TOs carry no ``source_refs``, so their
   statement / bloom edits are accepted after only the Bloom-verb check.
   **Remap grounding-revert:** a CO re-pointed to a new TO is re-scored
   ``cosine(co.statement, new_TO.statement)`` vs ``cosine(co.statement,
   old_TO.statement)`` (on the CO's FINAL statement, post statement-edit) and
   REVERTED unless the remap IMPROVES alignment (strictly higher cosine, or it
   lifts a below-floor link to >= the floor). A remap is never allowed to make
   alignment worse, and every CO always keeps SOME ``terminal_id``
   (never-unset). With no embedder the remap is conservatively reverted (we
   never re-point blind).
4. **ANTI-FABRICATION.** The pass only edits text fields that already exist
   and re-points ``terminal_id`` among EXISTING TO ids. It never synthesizes
   new ``source_refs`` / ``chunk_ids`` and never invents provenance or a TO.
5. **Graceful + chunked.** The review is split into SMALL per-call chunks:
   one call per ``chapter_objectives`` group (that group's COs are the review
   set; the terminal objectives ride along as read-only context), plus one
   call for the terminal-objective set (the COs ride along as compact
   read-only context). Each call asks the model to return an adjusted entry
   for EVERY id in that chunk's review set and ONLY those — so the model
   never treats the review items as context (the single-shot bug where it
   returned only the TOs and zero COs). A per-chunk failure (timeout, parse
   error, provider error) is fail-SOFT for THAT chunk only: that group keeps
   its original 7B objectives, a warning is logged, and the other chunks
   continue. Counters are aggregated across all chunks into a SINGLE
   ``objective_review`` decision event. The review client uses a generous
   request timeout (``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS``, default ~300s —
   NOT the 60s client default) since a reasoning model authoring a chunk of
   edits is slow.

Env gate (default OFF): ``ED4ALL_OBJECTIVE_REVIEW_PROVIDER`` (unset/empty =
off; ``nvidia`` = run via the NVIDIA seat) + optional
``ED4ALL_OBJECTIVE_REVIEW_MODEL`` (chain: explicit arg → that env var →
``NVIDIA_LARGE_MODEL`` → the review-pass strong default
``meta/llama-3.3-70b-instruct``). The 70B is the review-specific default
(Wave-2 Part 2) — a stronger reviewer than the shared ``nvidia`` endpoint's
content-gen ``default_model`` (a 30B nemotron), which is NOT changed. The
NVIDIA client is built the SAME way the ``nvidia`` content-gen provider builds
it (reusing ``OpenAICompatibleClient`` — no hand-rolled second HTTP path).

Decision capture: one ``objective_review`` event per review call, with a
dynamic rationale interpolating objectives reviewed, # statements adjusted, #
reverted-for-grounding, # bloom edits applied / rejected, model id, provider.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env contract (parse-with-fallback, mirroring the other ED4ALL_* knobs).
# ---------------------------------------------------------------------------

ENV_REVIEW_PROVIDER = "ED4ALL_OBJECTIVE_REVIEW_PROVIDER"
ENV_REVIEW_MODEL = "ED4ALL_OBJECTIVE_REVIEW_MODEL"

#: Providers the review pass knows how to construct. Only ``nvidia`` today
#: (the strong hosted reviewer); the set is the validation allowlist. Every
#: entry MUST be a registered endpoint name in ``config/endpoints.yaml`` —
#: the review pass attaches BY NAME via the unified endpoint registry, so a
#: typo / non-endpoint here fails loud at import rather than at dispatch.
_SUPPORTED_REVIEW_PROVIDERS = ("nvidia",)


def _assert_review_providers_are_endpoints() -> None:
    """Fail loud at import if a review provider isn't a registry endpoint."""
    try:
        from lib.llm.endpoints import endpoint_names  # noqa: PLC0415

        known = set(endpoint_names())
    except Exception:  # noqa: BLE001 — registry load issues surface elsewhere
        return
    unknown = [p for p in _SUPPORTED_REVIEW_PROVIDERS if p not in known]
    if unknown:
        raise RuntimeError(
            f"objective-review providers not in the endpoint registry: "
            f"{unknown}; add a row to config/endpoints.yaml or fix "
            f"_SUPPORTED_REVIEW_PROVIDERS."
        )


_assert_review_providers_are_endpoints()

#: Generous output budget for the LEGACY single-shot path (kept for the
#: degenerate "everything in one chunk" case + back-compat). The chunked
#: path uses the smaller per-chunk budget below.
_REVIEW_MAX_TOKENS = 24576

#: Per-chunk output budget. Each chunked call echoes adjusted fields for only
#: a SMALL set (~6-12 objectives × ~90 tokens ≈ 1k) of editable content — but
#: the Nemotron reasoning model still emits a non-trivial residual chain of
#: thought even under "detailed thinking off" (see _SYSTEM_PROMPT). A live
#: 13-chunk run measured ONE 6-CO group truncating at 16384
#: (finish_reason='length' → fail-soft, group kept its originals), so the
#: budget is sized to the proven single-shot headroom (24576) — each chunk is
#: still one small group, so the larger CAP doesn't slow the common case (the
#: model stops at its natural JSON end) but gives the occasional reasoning-
#: heavy chunk room to finish rather than getting dropped. Sizing the budget
#: to the chunk (not the whole 83-objective payload) is still what keeps each
#: call's WALL TIME bounded enough to avoid the read-timeout the single-shot
#: path hit.
_REVIEW_CHUNK_MAX_TOKENS = 24576

#: Max chapter-objectives a SINGLE review call may carry. The chunked path is
#: sized for small (~6-12 CO) groups, but the call site passes the objectives
#: as ONE ``'all'`` group BEFORE week-slicing — so a large course (observed
#: 2026-06-21: 97 COs in one ``'all'`` group after the I3 distinct-skill split
#: + source backfill) otherwise becomes ONE 97-CO call that blows the read
#: timeout (3 transport attempts → fail-soft → the whole course gets NO 70B
#: review). Any group larger than this is sub-chunked into ``<=N``-CO calls so
#: each call's wall time stays bounded. A group already within the cap makes
#: exactly one call (byte-stable to the prior per-week behaviour).
_MAX_COS_PER_REVIEW_CHUNK = 12

#: Default per-request HTTP timeout (seconds) for the review client when
#: ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` is unset. Mirrors the Courseforge
#: outline / rewrite providers' generous posture — a single chunk of objective
#: edits can take well over the 60s ``OpenAICompatibleClient`` default on a
#: reasoning model, so we source a 300s floor rather than the bare client
#: default.
_REVIEW_DEFAULT_TIMEOUT_SECONDS = 300.0

#: Sampling temperature — low so the reviewer edits conservatively.
_REVIEW_TEMPERATURE = 0.2

#: A CO statement edit that stays above the cosine floor is reverted only when
#: its CO->TO cosine drops more than this margin below the original — a real
#: grounding regression, not a minor wording change. The below-floor check is
#: the primary guardrail; this catches edits that drift WITHIN the safe band
#: but still meaningfully weaken the roll-up.
_MATERIALLY_WORSE_MARGIN = 0.10

#: Decision-capture event type (NEW enum member, registered in
#: ``schemas/events/decision_event.schema.json``).
_DECISION_TYPE = "objective_review"

#: Editable fields sent to / merged back from the reviewer. ``source_refs``
#: and every other field are NEVER in this set. ``terminal_id`` is the CO->TO
#: MAPPING the reviewer may RE-POINT (Wave-2 Part 2 relaxation) — it is the
#: only id-shaped field the model may return, and it is guardrailed (the
#: returned value MUST be an existing TO id, and the remap must not WORSEN the
#: CO->TO cosine; see ``merge_reviewed_objectives``). It is NOT a quality
#: text edit, so it rides alongside ``_EDITABLE_FIELDS`` in the editable view
#: but is merged by the remap-specific guardrail, not ``_apply_bloom_fields``.
_EDITABLE_FIELDS = ("statement", "bloom_level", "bloom_verb", "abcd")

#: The CO->TO mapping field the reviewer may RE-POINT. Sent to the model for
#: COs (so it sees the current assignment) and accepted back under the remap
#: guardrail. Never sent / accepted for TOs (a TO has no parent terminal).
_REMAP_FIELD = "terminal_id"

#: Default objective-review model for the ``nvidia`` provider seat when neither
#: an explicit arg, ``ED4ALL_OBJECTIVE_REVIEW_MODEL``, nor ``NVIDIA_LARGE_MODEL``
#: is set. Wave-2 Part 2 points the review pass at the strong 70B
#: (``meta/llama-3.3-70b-instruct``) — a far stronger reviewer than the shared
#: ``nvidia`` endpoint's content-gen default (a 30B nemotron). We do NOT change
#: the endpoint's ``default_model`` (the rewrite tier shares that seat); the
#: 70B is the review-pass-specific default only, slotted into the resolution
#: chain ABOVE the endpoint default but BELOW every operator override.
_NVIDIA_REVIEW_DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"


def resolve_objective_review_provider(
    provider: Optional[str] = None,
) -> Optional[str]:
    """Resolve the review provider: explicit arg → env → off.

    Returns the lower-cased provider name when it is a recognised review
    provider, else ``None`` (the default-OFF no-op). Garbage / unknown /
    empty values resolve to ``None`` (parse-with-fallback — an unknown
    provider never crashes a run; it just disables the pass).
    """
    raw = provider if provider is not None else os.environ.get(
        ENV_REVIEW_PROVIDER
    )
    if not raw:
        return None
    candidate = str(raw).strip().lower()
    if candidate in _SUPPORTED_REVIEW_PROVIDERS:
        return candidate
    logger.warning(
        "%s=%r is not a supported objective-review provider (%s); "
        "objective-review pass disabled.",
        ENV_REVIEW_PROVIDER,
        raw,
        list(_SUPPORTED_REVIEW_PROVIDERS),
    )
    return None


def resolve_objective_review_model(
    provider: str,
    model: Optional[str] = None,
) -> Optional[str]:
    """Resolve the review model: explicit arg → ENV_REVIEW_MODEL → provider default.

    For ``nvidia`` the chain is: explicit arg → ``ED4ALL_OBJECTIVE_REVIEW_MODEL``
    → ``NVIDIA_LARGE_MODEL`` → the review-pass-specific 70B default
    (:data:`_NVIDIA_REVIEW_DEFAULT_MODEL`, ``meta/llama-3.3-70b-instruct``) —
    a STRONGER reviewer than the shared ``nvidia`` endpoint's content-gen
    ``default_model`` (a 30B nemotron). The endpoint default is the final
    fallback for any other provider. ``None`` lets the client fall back to its
    own resolution.
    """
    if model:
        return str(model).strip()
    env_model = os.environ.get(ENV_REVIEW_MODEL)
    if env_model and env_model.strip():
        return env_model.strip()
    # Provider default flows from the unified endpoint registry: the row's
    # ``model_env`` (e.g. ``NVIDIA_LARGE_MODEL``) → review default →
    # ``default_model``. Read the row directly (NOT ``resolve_endpoint``) so
    # model resolution never depends on the API key — an ``api_key_required``
    # endpoint must still report its model id on the key-unset failure path.
    from lib.llm.endpoints import load_endpoint_registry  # noqa: PLC0415

    row = load_endpoint_registry().get(provider)
    env_var = row.get("model_env") if row else None
    if env_var:
        env_default = os.environ.get(env_var)
        if env_default and env_default.strip():
            return env_default.strip()
    # Review-pass-specific strong-model default (Wave-2 Part 2): the ``nvidia``
    # review seat resolves to the 70B, NOT the endpoint's 30B content-gen
    # default, when no operator override is set. Slotted ABOVE the endpoint
    # default but BELOW every override (explicit arg / both env vars).
    if provider == "nvidia":
        return _NVIDIA_REVIEW_DEFAULT_MODEL
    if not row:
        return None
    default_model = row.get("default_model")
    return str(default_model) if default_model else None


def resolve_review_timeout() -> float:
    """Return the per-request HTTP timeout for the review client.

    Honors ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` (the build sets 300),
    falling back to :data:`_REVIEW_DEFAULT_TIMEOUT_SECONDS` (300s) — NOT the
    bare 60s ``OpenAICompatibleClient`` default — when the env var is unset /
    garbage / non-positive. Mirrors the Courseforge outline / rewrite
    providers' posture (generous default for a slow reasoning model authoring
    a chunk of objective edits). Parse-with-fallback, like the other
    ``ED4ALL_*`` timeout knobs.
    """
    import math as _math  # noqa: PLC0415

    raw = os.environ.get("ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS")
    if not raw or not str(raw).strip():
        return _REVIEW_DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return _REVIEW_DEFAULT_TIMEOUT_SECONDS
    if not _math.isfinite(parsed) or parsed <= 0:
        return _REVIEW_DEFAULT_TIMEOUT_SECONDS
    return parsed


def _build_review_client(
    provider: str,
    *,
    model: Optional[str],
    client: Optional[Any] = None,
) -> Tuple[Any, str]:
    """Construct the OpenAI-compatible review client (returns ``(client, model)``).

    Built via the unified endpoint registry's generic attach point
    (``lib.llm.endpoints.build_openai_compatible_client``) — the SAME
    constructor the ``nvidia`` content-gen provider routes through in
    ``Courseforge/generators/_base.py``. The endpoint NAME (the review
    provider, ``nvidia``) flows the base_url / key-env (``NVIDIA_API_KEY``,
    never hardcoded) / model from ``config/endpoints.yaml``. The
    ``api_key_required`` fail-loud is produced by the registry resolver
    (``EndpointKeyRequired``, a ``RuntimeError`` subclass) when the key is
    unset and no test client is injected — the caller catches it and no-ops.
    ``provider_label`` / ``json_mode`` / ``timeout`` are preserved verbatim.
    """
    from lib.llm.endpoints import (  # noqa: PLC0415
        build_openai_compatible_client,
    )

    # Resolve the model from the registry's env-or-default chain (no key
    # needed for this) so the returned model id is correct even on the
    # key-required failure path. ``build_*`` re-applies the same override.
    resolved_model = resolve_objective_review_model(provider, model)
    oa_client = build_openai_compatible_client(
        provider,
        model=resolved_model,
        capture=None,
        provider_label=provider,
        client=client,
        json_mode=True,
        timeout=resolve_review_timeout(),
    )
    return oa_client, resolved_model


# ---------------------------------------------------------------------------
# Editable-surface extraction + prompt
# ---------------------------------------------------------------------------


def _editable_view(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return the editable-only projection of an objective.

    Includes ``id`` (identity, immutable — sent so the model can key its
    response) + the four editable fields when present, PLUS the CO's CURRENT
    ``terminal_id`` (the CO->TO mapping the reviewer may re-point; resolved via
    any of the parent-terminal back-pointer keys so the model always sees the
    live assignment). NEVER includes ``source_refs`` / ``chunk_ids`` / any
    other field. TOs carry no parent-terminal key, so ``terminal_id`` is
    simply absent from a TO's view.
    """
    view: Dict[str, Any] = {"id": obj.get("id")}
    for field in _EDITABLE_FIELDS:
        if field in obj:
            view[field] = obj[field]
    assigned = _resolve_assigned_terminal(obj)
    if assigned:
        # Echo the live mapping back to the model so it can correct it. Use the
        # canonical lower-cased id resolved from whichever back-pointer key the
        # CO carries; the remap guardrail re-resolves the same way on merge.
        view[_REMAP_FIELD] = assigned
    return view


# The NVIDIA Nemotron model is a REASONING model: left to its defaults it
# emits a long chain-of-thought (in message.reasoning_content) BEFORE the
# answer JSON, which on an 83-objective payload overran the completion budget
# and truncated the JSON (finish_reason="length") — the pass then fail-soft
# kept the unreviewed 7B objectives. "detailed thinking off" is Nemotron's
# system-level reasoning toggle (verified to collapse reasoning_content to a
# handful of tokens); objective EDITING is a direct task that does not need
# extended CoT, so disabling it makes the call fit the budget reliably while a
# 30B editor still applies its knowledge. Paired with the raised
# _REVIEW_MAX_TOKENS below.
def _system_prompt(review_count: int) -> str:
    """Build the per-call system prompt.

    ``review_count`` is the number of ids the model MUST echo back (the size
    of THIS chunk — one chapter group's COs, or the TO set). The explicit
    "return ALL N / ONLY those" framing is what stops the model from treating
    the review set as read-only context (the bug where it returned only the
    12 TOs and zero COs).
    """
    return (
        "detailed thinking off\n\n"
        "You are an expert instructional designer reviewing a SMALL set of "
        "course learning objectives for QUALITY. You improve each objective's "
        "clarity and specificity, correct the Bloom level/verb when the verb "
        "does not match the cognitive level, complete or repair the ABCD "
        "fields (audience, behavior{verb,action_object}, condition, degree), "
        "and (for terminal objectives) make each statement genuinely cover "
        "its child chapter objectives. For a chapter objective you may also "
        "RE-MAP its 'terminal_id' to a BETTER-fitting EXISTING terminal "
        "objective when its current parent is wrong. You MUST NOT add, remove, "
        "rename, reorder, or merge objectives, and you MUST NOT invent a "
        "terminal id — any 'terminal_id' you return must already exist.\n\n"
        f"The input has a 'review' list of {review_count} objectives and a "
        "'context' list. You MUST return an adjusted entry for ALL "
        f"{review_count} ids listed under 'review' and ONLY those — echo "
        "every id, omit none. The 'context' objectives are READ-ONLY "
        "reference: use them to align wording, but do NOT edit them and do "
        "NOT return them. Return ONLY a JSON object mapping each 'review' id "
        "to its adjusted {statement, bloom_level, bloom_verb, abcd}. Do not "
        "invent ids. Output JSON only."
    )


def _render_co_group_prompt(
    *,
    course_name: str,
    chapter_label: Any,
    group_cos_view: List[Dict[str, Any]],
    terminals_view: List[Dict[str, Any]],
) -> str:
    """Render the user prompt for ONE chapter-objective group review call.

    The group's COs are the ``review`` set (the model returns an adjusted
    entry for each); the terminal objectives are passed as READ-ONLY
    ``context`` so the model can align CO wording to its parent terminal,
    NOT edit them.
    """
    payload = {
        "course_name": course_name,
        "chapter": chapter_label,
        "review": group_cos_view,
        "context_terminal_objectives_readonly": terminals_view,
    }
    n = len(group_cos_view)
    return (
        f"Review the {n} chapter objectives in 'review' for quality. The "
        "'context_terminal_objectives_readonly' are course-wide terminals — "
        "each chapter objective rolls up to ONE terminal (its current "
        "'terminal_id'). Improve clarity/specificity, fix Bloom level/verb "
        "mismatches, and complete ABCD fields. Do NOT edit or return the "
        "context terminals.\n\n"
        "RE-MAPPING (terminal_id): if a chapter objective is assigned to the "
        "WRONG terminal — i.e. another terminal in "
        "'context_terminal_objectives_readonly' is a clearly better thematic "
        "parent — set 'terminal_id' in your adjusted entry to that better "
        "terminal's EXISTING id. You may ONLY choose an id that already "
        "appears in the context terminals; never invent a terminal id. If the "
        "current mapping is already best, KEEP the existing 'terminal_id'.\n\n"
        f"Return an adjusted entry for ALL {n} ids in 'review' and ONLY "
        "those. Use the JSON form:\n"
        '{"adjusted": {"<id>": {"statement": "...", "bloom_level": "...", '
        '"bloom_verb": "...", "abcd": {"audience": "...", "behavior": '
        '{"verb": "...", "action_object": "..."}, "condition": "...", '
        '"degree": "..."}, "terminal_id": "<existing TO id>"}, ...}}\n\n'
        "Output JSON only.\n\n"
        "INPUT:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _render_terminals_prompt(
    *,
    course_name: str,
    terminals_view: List[Dict[str, Any]],
    co_context: List[Dict[str, Any]],
) -> str:
    """Render the user prompt for the SINGLE terminal-objective review call.

    The terminals are the ``review`` set; a compact summary (id + statement)
    of every CO is passed as READ-ONLY ``context`` so each terminal can be
    made to genuinely cover its child chapter objectives.
    """
    payload = {
        "course_name": course_name,
        "review": terminals_view,
        "context_chapter_objectives_readonly": co_context,
    }
    n = len(terminals_view)
    return (
        f"Review the {n} terminal objectives in 'review' for quality. The "
        "'context_chapter_objectives_readonly' are the child chapter "
        "objectives (id + statement) — use them to make each terminal "
        "statement genuinely COVER its children. Improve clarity, fix Bloom "
        "level/verb mismatches, and complete ABCD fields. Do NOT edit or "
        "return the context chapter objectives.\n\n"
        f"Return an adjusted entry for ALL {n} ids in 'review' and ONLY "
        "those. Use the JSON form:\n"
        '{"adjusted": {"<id>": {"statement": "...", "bloom_level": "...", '
        '"bloom_verb": "...", "abcd": {"audience": "...", "behavior": '
        '{"verb": "...", "action_object": "..."}, "condition": "...", '
        '"degree": "..."}}, ...}}\n\n'
        "Output JSON only.\n\n"
        "INPUT:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _parse_adjusted(raw_text: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Parse the reviewer response into ``{id: {editable fields}}`` or None.

    Tolerant: accepts a top-level ``{"adjusted": {...}}`` wrapper or a bare
    ``{id: {...}}`` map; strips ```` ```json ```` fences; extracts the first
    balanced JSON object when there is surrounding prose. Returns ``None`` on
    any unparseable shape (the caller then keeps the originals).
    """
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    # Strip code fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        # Salvage the first balanced { ... } object.
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        try:
            parsed = json.loads(text[start:end])
        except (ValueError, TypeError):
            return None

    if not isinstance(parsed, dict):
        return None
    adjusted = parsed.get("adjusted") if "adjusted" in parsed else parsed
    if not isinstance(adjusted, dict):
        return None
    # Coerce to {str_id: dict}.
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in adjusted.items():
        if isinstance(v, dict) and isinstance(k, str):
            out[k] = v
    return out or None


# ---------------------------------------------------------------------------
# Grounding guardrail
# ---------------------------------------------------------------------------


def _statement(obj: Dict[str, Any]) -> str:
    val = obj.get("statement")
    return str(val).strip() if isinstance(val, str) else ""


#: Lazy module-level reverse index ``verb (lowercase) -> sorted tuple of the
#: canonical Bloom levels that verb belongs to``. Built once from
#: ``BLOOMS_VERBS`` (which carries no verb→level helper). Used by the JOINT
#: Bloom reconciliation in ``_apply_bloom_fields`` to drive level + verb into
#: mutual agreement instead of all-or-nothing reverting. The values are sorted
#: (stable) so a verb valid for multiple levels resolves deterministically.
_VERB_TO_LEVELS_CACHE: Optional[Dict[str, Tuple[str, ...]]] = None


def _verb_to_levels() -> Dict[str, Tuple[str, ...]]:
    """Return the lazily-built ``verb -> (levels...)`` reverse index.

    Built once from ``BLOOMS_VERBS`` (lowercase-normalized). A verb maps to the
    sorted tuple of every canonical level whose verb-set contains it (the
    taxonomy currently has no multi-level verbs, but the reverse index handles
    that case so reconciliation stays deterministic).
    """
    global _VERB_TO_LEVELS_CACHE
    if _VERB_TO_LEVELS_CACHE is not None:
        return _VERB_TO_LEVELS_CACHE
    from lib.ontology.learning_objectives import BLOOMS_VERBS  # noqa: PLC0415

    rev: Dict[str, set] = {}
    for level, verbs in BLOOMS_VERBS.items():
        for verb in verbs:
            rev.setdefault(str(verb).strip().lower(), set()).add(level)
    _VERB_TO_LEVELS_CACHE = {
        v: tuple(sorted(levels)) for v, levels in rev.items()
    }
    return _VERB_TO_LEVELS_CACHE


def _canonical_level(level: Any) -> Optional[str]:
    """Return the lower-cased level iff it is a key of ``BLOOMS_VERBS``, else None."""
    from lib.ontology.learning_objectives import BLOOMS_VERBS  # noqa: PLC0415

    if not isinstance(level, str) or not level.strip():
        return None
    norm = level.strip().lower()
    return norm if norm in BLOOMS_VERBS else None


def _verb_in_level(verb: Optional[str], level: Optional[str]) -> bool:
    """True iff ``verb`` (case-insensitive) is canonical for ``level``."""
    from lib.ontology.learning_objectives import BLOOMS_VERBS  # noqa: PLC0415

    if not verb or not level:
        return False
    verbs = BLOOMS_VERBS.get(level)
    return bool(verbs) and verb.strip().lower() in verbs


def _level_for_verb(
    verb: Optional[str], *, prefer: Optional[str] = None
) -> Optional[str]:
    """Resolve the canonical level a verb belongs to.

    A verb valid for MULTIPLE levels resolves to: ``prefer`` when the verb is
    valid there, else the first (sorted, stable) of its levels. Returns None
    for an unknown / non-canonical verb (no resolvable level).
    """
    if not verb:
        return None
    levels = _verb_to_levels().get(verb.strip().lower())
    if not levels:
        return None
    if prefer and prefer in levels:
        return prefer
    return levels[0]


def _first_verb_for_level(level: str) -> Optional[str]:
    """Return the FIRST (sorted, stable) canonical verb of ``level``, or None."""
    from lib.ontology.learning_objectives import BLOOMS_VERBS  # noqa: PLC0415

    verbs = BLOOMS_VERBS.get(level)
    if not verbs:
        return None
    ordered = sorted(verbs)
    return ordered[0] if ordered else None


def _abcd_verb(abcd: Any) -> Optional[str]:
    """Return ``abcd.behavior.verb`` (stripped, original casing) or None."""
    if isinstance(abcd, dict):
        behavior = abcd.get("behavior")
        if isinstance(behavior, dict):
            v = behavior.get("verb")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _clone_abcd(abcd: Any) -> Dict[str, Any]:
    """Deep-ish copy an abcd dict so we don't alias the caller's structure.

    Copies the top-level dict + the nested ``behavior`` sub-dict (the only
    nested dict we mutate). Non-dict input yields a fresh empty dict.
    """
    if not isinstance(abcd, dict):
        return {}
    out: Dict[str, Any] = dict(abcd)
    behavior = out.get("behavior")
    if isinstance(behavior, dict):
        out["behavior"] = dict(behavior)
    return out


def _set_abcd_verb(abcd: Dict[str, Any], verb: str) -> None:
    """Write ``verb`` into ``abcd.behavior.verb`` (creating ``behavior`` as needed)."""
    behavior = abcd.get("behavior")
    if not isinstance(behavior, dict):
        behavior = {}
        abcd["behavior"] = behavior
    behavior["verb"] = verb


def _bloom_verb_aligned(bloom_level: Any, abcd: Any) -> bool:
    """Return True when ``abcd.behavior.verb`` is in ``BLOOMS_VERBS[bloom_level]``.

    Mirrors the ``abcd_verb_alignment`` gate rule. Missing bloom_level /
    abcd.behavior.verb is treated as ALIGNED (the bloom change is not the
    thing breaking — a separate gate handles missing abcd), so this only
    REVERTS a bloom change that introduces a real verb/level mismatch.
    """
    from lib.ontology.learning_objectives import BLOOMS_VERBS  # noqa: PLC0415

    if not isinstance(bloom_level, str) or not bloom_level.strip():
        return True
    level = bloom_level.strip().lower()
    verbs = BLOOMS_VERBS.get(level)
    if not verbs:
        # Unknown level → can't validate; treat as misaligned so an edit to
        # a non-canonical level is reverted (conservative).
        return False
    verb = None
    if isinstance(abcd, dict):
        behavior = abcd.get("behavior")
        if isinstance(behavior, dict):
            v = behavior.get("verb")
            if isinstance(v, str) and v.strip():
                verb = v.strip().lower()
    if verb is None:
        return True
    return verb in verbs


class _GroundingScorer:
    """CO→TO cosine scorer reusing the ``co_terminal_alignment`` signal.

    Memoizes TO-statement encodes (62 COs typically share ~3 TOs). When no
    embedder is available the scorer reports ``available=False`` and the
    caller adopts CO statement edits without the cosine guardrail (the bloom
    guardrail still applies) — matching the embedding-validator graceful
    degrade contract.
    """

    def __init__(self, embedder: Optional[Any]) -> None:
        self._embedder = embedder
        self._to_vec_cache: Dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return self._embedder is not None

    def cosine(self, co_statement: str, to_statement: str) -> Optional[float]:
        if self._embedder is None or not co_statement or not to_statement:
            return None
        from lib.embedding._math import cosine_similarity  # noqa: PLC0415

        to_vec = self._to_vec_cache.get(to_statement)
        if to_vec is None:
            try:
                to_vec = self._embedder.encode(to_statement)
            except Exception as exc:  # noqa: BLE001
                logger.warning("review: TO encode failed: %s", exc)
                return None
            self._to_vec_cache[to_statement] = to_vec
        try:
            co_vec = self._embedder.encode(co_statement)
        except Exception as exc:  # noqa: BLE001
            logger.warning("review: CO encode failed: %s", exc)
            return None
        return cosine_similarity(co_vec, to_vec)


def _resolve_assigned_terminal(co: Dict[str, Any]) -> Optional[str]:
    """Return the lower-cased parent-terminal id of a CO, or None."""
    from lib.ontology.terminal_coverage import (  # noqa: PLC0415
        _PARENT_TERMINAL_KEYS,
    )

    for key in _PARENT_TERMINAL_KEYS:
        val = co.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return None


def _set_assigned_terminal(co: Dict[str, Any], to_id: str) -> None:
    """Re-point a CO to ``to_id`` by writing EVERY back-pointer key it carries.

    The remap preserves the CO's identity + provenance; it only changes which
    terminal the CO rolls up to. To stay consistent for every downstream
    reader (the archival gate reads ``parent_terminal``; ``terminal_id`` is the
    generic alias; the RDF corpus uses ``parent_to``), we write the new id to
    EVERY parent-terminal key the CO already carries, falling back to
    ``terminal_id`` when the CO carried none under a non-default key — so the
    never-unset contract holds and no stale back-pointer survives the remap.
    The casing of the matched TO id (``to_id`` here is the TO's on-disk id) is
    preserved verbatim.
    """
    from lib.ontology.terminal_coverage import (  # noqa: PLC0415
        _PARENT_TERMINAL_KEYS,
    )

    wrote_any = False
    for key in _PARENT_TERMINAL_KEYS:
        if key in co:
            co[key] = to_id
            wrote_any = True
    if not wrote_any:
        # CO had no recognised back-pointer key (shouldn't happen for a CO that
        # resolved an assigned terminal, but stay safe) — set the generic alias.
        co["terminal_id"] = to_id


# ---------------------------------------------------------------------------
# Merge + guardrail (pure — unit-testable without network)
# ---------------------------------------------------------------------------


def _apply_bloom_fields(target: Dict[str, Any], adj: Dict[str, Any]) -> int:
    """Apply bloom_level / bloom_verb / abcd from ``adj`` to ``target`` in place.

    P2(a) JOINT Bloom reconciliation. ``statement`` is handled separately by
    the caller (it carries the cosine guardrail). Returns the count of
    bloom-ish fields changed (so the ``bloom_edits_applied`` /
    ``bloom_edits_rejected`` counters in ``merge_reviewed_objectives`` keep
    working: a reconciled edit returns > 0; a true no-coherent-pairing revert
    returns 0).

    Instead of the old all-or-nothing revert (which left the ORIGINAL mismatch
    in place whenever the reviewer offered a PARTIAL Bloom fix — e.g. a level
    edit with a stale abcd verb, or a verb edit with a stale level), this
    drives ``bloom_level`` + ``bloom_verb`` + ``abcd.behavior.verb`` into
    MUTUAL agreement, preferring the reviewer's intent:

    1. candidate LEVEL = ``adj.bloom_level`` (if a valid canonical level) else
       ``target.bloom_level``.
    2. candidate VERB = first present of ``adj.abcd.behavior.verb`` →
       ``adj.bloom_verb`` → ``target.abcd.behavior.verb`` → ``target.bloom_verb``.
    3. If the candidate VERB is canonical for the candidate LEVEL → coherent:
       apply the level + set both ``bloom_verb`` and ``abcd.behavior.verb`` to
       that verb.
    4. Else RECONCILE rather than revert:
       a. The verb has a resolvable level AND the reviewer changed the LEVEL →
          honor the reviewer's LEVEL, REPLACE the verb with a canonical verb
          for that level (reuse target/adj's bloom_verb if it is valid there,
          else the first sorted verb of the level).
       b. The reviewer only changed the VERB (level unchanged) AND the verb has
          a resolvable level → drive the LEVEL to that verb's level, adopt the
          verb.
       c. Neither yields a coherent (level, verb) — e.g. a non-canonical verb
          with no resolvable level — keep the ORIGINAL (return 0).

    Anti-fabrication: only verbs from ``BLOOMS_VERBS`` and levels from its keys
    are ever written; the rest of the abcd object (audience/condition/degree/
    action_object) is preserved verbatim via a non-aliasing copy.
    """
    raw_new_level = adj.get("bloom_level")
    new_level_canon = _canonical_level(raw_new_level)
    reviewer_changed_level = (
        new_level_canon is not None
        and new_level_canon != _canonical_level(target.get("bloom_level"))
    )

    target_level_canon = _canonical_level(target.get("bloom_level"))
    candidate_level = new_level_canon or target_level_canon

    # candidate VERB priority: adj.abcd verb → adj.bloom_verb → target.abcd
    # verb → target.bloom_verb. ``adj``-sourced verbs signal the reviewer
    # intent; the target-sourced ones are the fallback when the reviewer
    # offered no verb.
    adj_abcd_verb = _abcd_verb(adj.get("abcd"))
    adj_bloom_verb = (
        adj.get("bloom_verb").strip()
        if isinstance(adj.get("bloom_verb"), str) and adj.get("bloom_verb").strip()
        else None
    )
    target_abcd_verb = _abcd_verb(target.get("abcd"))
    target_bloom_verb = (
        target.get("bloom_verb").strip()
        if isinstance(target.get("bloom_verb"), str)
        and target.get("bloom_verb").strip()
        else None
    )
    candidate_verb = (
        adj_abcd_verb
        or adj_bloom_verb
        or target_abcd_verb
        or target_bloom_verb
    )
    reviewer_changed_verb = bool(adj_abcd_verb or adj_bloom_verb)

    final_level: Optional[str] = candidate_level
    final_verb: Optional[str] = candidate_verb

    if candidate_level is None:
        # No canonical level to anchor on (neither adj nor target carries one).
        # If the candidate verb resolves a level, adopt it; else revert.
        resolved = _level_for_verb(candidate_verb)
        if resolved is None:
            return 0
        final_level = resolved
        final_verb = candidate_verb
    elif _verb_in_level(candidate_verb, candidate_level):
        # (3) coherent as-is.
        final_level = candidate_level
        final_verb = candidate_verb
    else:
        # (4) reconcile rather than revert.
        verb_level = _level_for_verb(candidate_verb)
        if reviewer_changed_level and candidate_level is not None:
            # (4a) honor the reviewer's LEVEL; replace the verb with one
            # canonical for that level. Reuse an existing bloom_verb that is
            # valid there, else the first sorted verb of the level.
            replacement = None
            for cand in (target_bloom_verb, adj_bloom_verb, target_abcd_verb):
                if _verb_in_level(cand, candidate_level):
                    replacement = cand.strip()
                    break
            if replacement is None:
                replacement = _first_verb_for_level(candidate_level)
            if replacement is None:
                return 0
            final_level = candidate_level
            final_verb = replacement
        elif reviewer_changed_verb and verb_level is not None:
            # (4b) the reviewer changed only the verb → drive the level to the
            # verb's level, adopt the verb.
            final_level = verb_level
            final_verb = candidate_verb
        elif verb_level is not None:
            # No explicit reviewer level/verb change but the candidate verb
            # still resolves a level (e.g. only an abcd dict was offered whose
            # verb mismatches the original level): drive the level to agree.
            final_level = verb_level
            final_verb = candidate_verb
        else:
            # (4c) no coherent pairing — keep the original.
            return 0

    if final_level is None or final_verb is None:
        return 0

    # Apply the reconciled (level, verb) + abcd into the target in place,
    # counting each field that actually changed.
    applied = 0
    if final_level != target_level_canon:
        target["bloom_level"] = final_level
        applied += 1

    if final_verb != target_bloom_verb:
        target["bloom_verb"] = final_verb
        applied += 1

    # Build the abcd to write: prefer the reviewer's offered abcd (preserving
    # its audience/condition/degree/action_object), else the target's existing
    # abcd; then force the behavior verb to the reconciled verb. Copy so we
    # never alias the caller's nested dicts.
    offered_abcd = adj.get("abcd") if isinstance(adj.get("abcd"), dict) else None
    base_abcd = offered_abcd if offered_abcd is not None else target.get("abcd")
    if isinstance(base_abcd, dict) or offered_abcd is not None:
        new_abcd = _clone_abcd(base_abcd)
        _set_abcd_verb(new_abcd, final_verb)
        if new_abcd != target.get("abcd"):
            target["abcd"] = new_abcd
            applied += 1

    return applied


def _apply_co_remap(
    *,
    co: Dict[str, Any],
    adj: Dict[str, Any],
    to_by_id: Dict[str, Dict[str, Any]],
    scorer: "_GroundingScorer",
    threshold: float,
    counters: Dict[str, int],
) -> None:
    """Apply the reviewer's CO->TO remap to ``co`` in place, under guardrails.

    Reads ``adj['terminal_id']`` (the reviewer's proposed new parent). No-op
    when the field is absent or names the current parent. Guardrails:

    * **id-set immutable** — the proposed id must be an EXISTING TO id
      (``to_by_id``); otherwise it is dropped (``remaps_unknown_to_dropped``)
      and the CO keeps its current mapping.
    * **grounding-revert** — recompute ``cosine(co.statement, new_to.statement)``
      vs ``cosine(co.statement, old_to.statement)``. Accept the remap only when
      it IMPROVES alignment: the new cosine is strictly greater than the old,
      OR the old link was below ``threshold`` and the new link reaches it. A
      remap that does not improve (equal / worse) is REVERTED
      (``remaps_reverted``). With no embedder (cosine unavailable) the remap is
      conservatively reverted — we never re-point blind, since a remap with no
      grounding evidence could only make alignment worse.
    * **never-unset** — the CO always keeps SOME terminal_id (either the new
      accepted parent or its original).
    """
    raw_new = adj.get(_REMAP_FIELD)
    if not isinstance(raw_new, str) or not raw_new.strip():
        return
    new_to_id = raw_new.strip().lower()
    current = _resolve_assigned_terminal(co)
    if current is not None and new_to_id == current:
        return  # already assigned there — no remap
    if new_to_id not in to_by_id:
        # Proposed a non-existent terminal — drop, keep current mapping.
        counters["remaps_unknown_to_dropped"] += 1
        return

    co_stmt = _statement(co)
    new_to_stmt = _statement(to_by_id[new_to_id])
    old_to_stmt = (
        _statement(to_by_id[current])
        if current is not None and current in to_by_id
        else ""
    )

    # Grounding-revert: only accept a remap that improves alignment.
    accept = False
    if scorer.available and co_stmt and new_to_stmt:
        new_cos = scorer.cosine(co_stmt, new_to_stmt)
        old_cos = (
            scorer.cosine(co_stmt, old_to_stmt) if old_to_stmt else None
        )
        if new_cos is not None:
            if old_cos is None:
                # No measurable old link (missing / unresolved old TO) — accept
                # only when the new link clears the floor on its own.
                accept = new_cos >= threshold
            else:
                strictly_better = new_cos > old_cos
                lifts_below_floor = old_cos < threshold <= new_cos
                accept = strictly_better or lifts_below_floor
    # else: no embedder → cannot prove improvement → conservatively revert.

    if accept:
        # Preserve the TO's on-disk id casing (not the lower-cased match key).
        to_id_on_disk = to_by_id[new_to_id].get("id")
        _set_assigned_terminal(
            co,
            str(to_id_on_disk) if isinstance(to_id_on_disk, str) else new_to_id,
        )
        counters["remaps_applied"] += 1
    else:
        counters["remaps_reverted"] += 1


def merge_reviewed_objectives(
    *,
    terminals: List[Dict[str, Any]],
    chapter_objectives: List[Dict[str, Any]],
    adjusted: Dict[str, Dict[str, Any]],
    scorer: "_GroundingScorer",
    threshold: float,
) -> Dict[str, int]:
    """Merge the reviewer's editable-field adjustments back by id, in place.

    Mutates ``terminals`` and the CO dicts inside ``chapter_objectives`` in
    place. Identity + ``source_refs`` are preserved verbatim — only the
    editable fields are merged, and only when the grounding guardrail accepts
    the change. Returns a counters dict for the decision-capture rationale.

    Args:
        terminals: TO list (each ``{id, statement, ...}``); ``source_refs``
            empty by contract.
        chapter_objectives: the on-disk ``chapter_objectives`` groups
            (``[{chapter, objectives: [CO,...]}]``); CO dicts mutated in place.
        adjusted: ``{id: {editable fields}}`` parsed from the reviewer; ids
            not in the original set are dropped here.
        scorer: CO→TO cosine scorer (``available=False`` skips the cosine
            guardrail but keeps the bloom guardrail).
        threshold: per-CO cosine floor (default the ``co_terminal_alignment``
            0.45). A CO statement edit is reverted when the adjusted cosine is
            below this floor OR materially worse than the original.
    """
    counters = {
        "reviewed": 0,
        "statements_adjusted": 0,
        "statements_reverted": 0,
        "bloom_edits_applied": 0,
        "bloom_edits_rejected": 0,
        "unknown_ids_dropped": 0,
        # Wave-2 Part 2: CO->TO remap counters.
        "remaps_applied": 0,
        "remaps_reverted": 0,
        "remaps_unknown_to_dropped": 0,
    }

    # Lower-cased id → adjusted entry, so casing drift doesn't drop a match.
    adj_by_id = {str(k).strip().lower(): v for k, v in adjusted.items()}

    # Original id set (lower-cased) — drop any returned id outside it.
    to_by_id: Dict[str, Dict[str, Any]] = {}
    for t in terminals:
        tid = t.get("id")
        if isinstance(tid, str) and tid.strip():
            to_by_id[tid.strip().lower()] = t
    co_ids = set()
    for group in chapter_objectives:
        for co in group.get("objectives", []) or []:
            cid = co.get("id")
            if isinstance(cid, str) and cid.strip():
                co_ids.add(cid.strip().lower())
    original_ids = set(to_by_id) | co_ids
    counters["unknown_ids_dropped"] = sum(
        1 for k in adj_by_id if k not in original_ids
    )

    # ---- Terminals: statement + bloom edits accepted after bloom check ----
    for t in terminals:
        tid = t.get("id")
        if not isinstance(tid, str) or not tid.strip():
            continue
        adj = adj_by_id.get(tid.strip().lower())
        if not isinstance(adj, dict):
            continue
        counters["reviewed"] += 1
        new_stmt = adj.get("statement")
        if (
            isinstance(new_stmt, str)
            and new_stmt.strip()
            and new_stmt.strip() != _statement(t)
        ):
            t["statement"] = new_stmt.strip()
            counters["statements_adjusted"] += 1
        before = _apply_bloom_fields(t, adj)
        counters["bloom_edits_applied"] += before
        if _has_bloom_intent(adj) and before == 0:
            counters["bloom_edits_rejected"] += 1

    # ---- Chapter objectives: statement edit carries cosine guardrail ----
    for group in chapter_objectives:
        for co in group.get("objectives", []) or []:
            cid = co.get("id")
            if not isinstance(cid, str) or not cid.strip():
                continue
            adj = adj_by_id.get(cid.strip().lower())
            if not isinstance(adj, dict):
                continue
            counters["reviewed"] += 1

            new_stmt = adj.get("statement")
            original_stmt = _statement(co)
            statement_changed = (
                isinstance(new_stmt, str)
                and new_stmt.strip()
                and new_stmt.strip() != original_stmt
            )
            if statement_changed:
                assigned = _resolve_assigned_terminal(co)
                to_stmt = (
                    _statement(to_by_id[assigned])
                    if assigned and assigned in to_by_id
                    else ""
                )
                accept = True
                if scorer.available and to_stmt:
                    orig_cos = scorer.cosine(original_stmt, to_stmt)
                    new_cos = scorer.cosine(new_stmt.strip(), to_stmt)
                    if new_cos is not None:
                        below_floor = new_cos < threshold
                        # "materially worse" — a MEANINGFUL semantic drop vs
                        # the original (not a minor wording dilution). The
                        # margin is deliberately generous (0.10) so a genuine
                        # clarity edit that stays above the floor ships; only
                        # a real grounding regression is reverted.
                        materially_worse = (
                            orig_cos is not None
                            and new_cos < orig_cos - _MATERIALLY_WORSE_MARGIN
                        )
                        if below_floor or materially_worse:
                            accept = False
                if accept:
                    co["statement"] = new_stmt.strip()
                    counters["statements_adjusted"] += 1
                else:
                    counters["statements_reverted"] += 1

            # ---- CO->TO remap (the Wave-2 Part 2 relaxation) ----
            # The reviewer may re-point a CO to a BETTER-fitting EXISTING
            # terminal. Guardrails: (1) the returned id MUST be in the TO set
            # (else dropped — CO keeps original mapping); (2) GROUNDING-REVERT —
            # the remap must IMPROVE alignment (new cosine strictly better than
            # old, OR lift a below-floor CO to >= floor); a remap that does not
            # improve / makes it worse is reverted; (3) never-unset — the CO
            # always keeps SOME terminal_id. The cosine uses the CO's FINAL
            # statement (post statement-edit above), matching what the
            # co_terminal_alignment validator will recompute on disk.
            _apply_co_remap(
                co=co,
                adj=adj,
                to_by_id=to_by_id,
                scorer=scorer,
                threshold=threshold,
                counters=counters,
            )

            before = _apply_bloom_fields(co, adj)
            counters["bloom_edits_applied"] += before
            if _has_bloom_intent(adj) and before == 0:
                counters["bloom_edits_rejected"] += 1

    return counters


def _has_bloom_intent(adj: Dict[str, Any]) -> bool:
    """True when the adjustment carries any bloom/abcd field (so a 0-applied
    result means the guardrail rejected it, not that nothing was offered)."""
    return any(
        k in adj and adj.get(k) not in (None, "", {})
        for k in ("bloom_level", "bloom_verb", "abcd")
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def review_objectives(
    *,
    terminals: List[Dict[str, Any]],
    chapter_objectives: List[Dict[str, Any]],
    course_name: str,
    capture: Optional[Any] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    threshold: float = 0.45,
    embedder: Optional[Any] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the grounding-safe objective-review pass IN PLACE.

    Mutates ``terminals`` + the CO dicts inside ``chapter_objectives`` in
    place (when enabled and the call succeeds). Returns a result dict
    ``{"enabled": bool, "applied": bool, "counters": {...}, "provider": str,
    "model": str}``. NEVER raises — any error logs a warning and leaves the
    objectives byte-identical to the 7B input.

    Default-OFF: when ``ED4ALL_OBJECTIVE_REVIEW_PROVIDER`` (or the explicit
    ``provider`` arg) is unset/empty/unknown, returns ``{"enabled": False,
    "applied": False}`` WITHOUT constructing a client or touching the
    objectives.

    Args:
        terminals: TO list — ``source_refs`` empty by contract.
        chapter_objectives: on-disk ``chapter_objectives`` groups; CO dicts
            carry the chunk grounding in ``source_refs`` (NEVER mutated).
        course_name: for the prompt + rationale.
        capture: ``DecisionCapture`` — one ``objective_review`` event per
            review call.
        provider / model: explicit overrides (else env-resolved).
        threshold: per-CO cosine floor for the grounding guardrail.
        embedder: test seam for the CO→TO scorer; defaults to
            ``try_load_embedder()``.
        client: test seam for the OpenAI-compatible HTTP client.
    """
    resolved_provider = resolve_objective_review_provider(provider)
    if resolved_provider is None:
        return {"enabled": False, "applied": False, "counters": {}}

    resolved_model = resolve_objective_review_model(resolved_provider, model)

    # Snapshot original statements so a parse/dispatch failure mid-merge never
    # leaves a half-applied edit (we only mutate after a successful parse, but
    # the snapshot also lets the rationale report net adjustments honestly).
    try:
        oa_client, resolved_model = _build_review_client(
            resolved_provider, model=resolved_model, client=client
        )
    except Exception as exc:  # noqa: BLE001 — key unset / construction failure
        logger.warning(
            "objective-review pass disabled — client construction failed "
            "(provider=%s): %s. Keeping original 7B objectives.",
            resolved_provider,
            exc,
        )
        return {
            "enabled": True,
            "applied": False,
            "counters": {},
            "provider": resolved_provider,
            "model": resolved_model,
        }

    # Lazy-load the embedder for the CO→TO guardrail (test seam wins).
    scorer_embedder = embedder
    if scorer_embedder is None:
        try:
            from lib.embedding.sentence_embedder import (  # noqa: PLC0415
                try_load_embedder,
            )
            scorer_embedder = try_load_embedder()
        except Exception as exc:  # noqa: BLE001
            logger.warning("review: embedder load failed: %s", exc)
            scorer_embedder = None
    scorer = _GroundingScorer(scorer_embedder)

    terminals_view = [_editable_view(t) for t in terminals if isinstance(t, dict)]
    # Compact CO context (id + statement only) for the TO review call so a
    # terminal can be made to cover its children without bloating the prompt.
    co_context: List[Dict[str, Any]] = []
    total_cos = 0
    for group in chapter_objectives:
        for co in group.get("objectives", []) or []:
            if isinstance(co, dict):
                total_cos += 1
                co_context.append(
                    {"id": co.get("id"), "statement": _statement(co)}
                )

    total_objectives = len(terminals_view) + total_cos
    if total_objectives == 0:
        return {
            "enabled": True,
            "applied": False,
            "counters": {},
            "provider": resolved_provider,
            "model": resolved_model,
        }

    # ---- Build the explicit per-chunk work list ----
    # One chunk per chapter_objectives group (its COs are the review set, the
    # TOs ride along as read-only context), plus one chunk for the TO set
    # (the COs ride along as compact read-only context).
    chunks: List[Tuple[str, List[Dict[str, Any]]]] = []  # (label, messages)
    for group in chapter_objectives:
        group_cos_view = [
            _editable_view(co)
            for co in group.get("objectives", []) or []
            if isinstance(co, dict)
        ]
        if not group_cos_view:
            continue
        chapter_label = group.get("chapter")
        # Sub-chunk an oversized group (e.g. the pre-week-slicing ``'all'``
        # group of 97 COs) into ``<=_MAX_COS_PER_REVIEW_CHUNK``-CO calls so each
        # call finishes under the read timeout. A within-cap group makes
        # exactly one call labelled with its plain chapter label (byte-stable
        # to the prior per-week behaviour); only an oversized group gets the
        # ``#N`` sub-chunk suffix.
        n_group = len(group_cos_view)
        multi = n_group > _MAX_COS_PER_REVIEW_CHUNK
        for _sub in range(0, n_group, _MAX_COS_PER_REVIEW_CHUNK):
            sub_cos = group_cos_view[_sub:_sub + _MAX_COS_PER_REVIEW_CHUNK]
            sub_label = (
                f"{chapter_label}#{_sub // _MAX_COS_PER_REVIEW_CHUNK + 1}"
                if multi else chapter_label
            )
            prompt = _render_co_group_prompt(
                course_name=course_name,
                chapter_label=chapter_label,
                group_cos_view=sub_cos,
                terminals_view=terminals_view,
            )
            chunks.append(
                (
                    f"co_group:{sub_label!r}({len(sub_cos)})",
                    [
                        {"role": "system",
                         "content": _system_prompt(len(sub_cos))},
                        {"role": "user", "content": prompt},
                    ],
                )
            )
    if terminals_view:
        prompt = _render_terminals_prompt(
            course_name=course_name,
            terminals_view=terminals_view,
            co_context=co_context,
        )
        chunks.append(
            (
                f"terminals({len(terminals_view)})",
                [
                    {"role": "system",
                     "content": _system_prompt(len(terminals_view))},
                    {"role": "user", "content": prompt},
                ],
            )
        )

    # ---- Dispatch each chunk; fail-soft per chunk, aggregate counters ----
    agg = {
        "reviewed": 0,
        "statements_adjusted": 0,
        "statements_reverted": 0,
        "bloom_edits_applied": 0,
        "bloom_edits_rejected": 0,
        "unknown_ids_dropped": 0,
        "remaps_applied": 0,
        "remaps_reverted": 0,
        "remaps_unknown_to_dropped": 0,
    }
    chunks_attempted = 0
    chunks_failed = 0
    for label, messages in chunks:
        chunks_attempted += 1
        adjusted: Optional[Dict[str, Dict[str, Any]]] = None
        try:
            raw_text = oa_client.chat_completion(
                messages,
                max_tokens=_REVIEW_CHUNK_MAX_TOKENS,
                temperature=_REVIEW_TEMPERATURE,
            )
            adjusted = _parse_adjusted(raw_text)
        except Exception as exc:  # noqa: BLE001 — provider/network failure
            logger.warning(
                "objective-review chunk %s dispatch failed (provider=%s, "
                "model=%s): %s. Keeping this group's original 7B "
                "objectives; continuing.",
                label, resolved_provider, resolved_model, exc,
            )
            chunks_failed += 1
            continue
        if not adjusted:
            logger.warning(
                "objective-review chunk %s response did not parse into an "
                "adjusted-id map (provider=%s, model=%s); keeping this "
                "group's originals; continuing.",
                label, resolved_provider, resolved_model,
            )
            chunks_failed += 1
            continue
        # Reuse the SAME guardrail engine. It iterates all terminals + CO
        # groups but only mutates ids present in this chunk's adjusted map,
        # so a per-chunk call applies exactly this chunk's edits.
        chunk_counters = merge_reviewed_objectives(
            terminals=terminals,
            chapter_objectives=chapter_objectives,
            adjusted=adjusted,
            scorer=scorer,
            threshold=threshold,
        )
        for k in agg:
            agg[k] += chunk_counters.get(k, 0)

    # Applied if at least one chunk landed (some edits merged). When every
    # chunk failed, the objectives are byte-identical to the 7B input.
    applied = chunks_failed < chunks_attempted

    _emit_review_decision(
        capture,
        counters=agg,
        total_objectives=total_objectives,
        provider=resolved_provider,
        model=resolved_model,
        applied=applied,
        note=None if applied else "all_chunks_failed",
        scorer_available=scorer.available,
        chunks_attempted=chunks_attempted,
        chunks_failed=chunks_failed,
    )

    return {
        "enabled": True,
        "applied": applied,
        "counters": agg,
        "provider": resolved_provider,
        "model": resolved_model,
        "chunks_attempted": chunks_attempted,
        "chunks_failed": chunks_failed,
    }


def _emit_review_decision(
    capture: Optional[Any],
    *,
    counters: Dict[str, int],
    total_objectives: int,
    provider: str,
    model: Optional[str],
    applied: bool,
    note: Optional[str] = None,
    scorer_available: Optional[bool] = None,
    chunks_attempted: Optional[int] = None,
    chunks_failed: Optional[int] = None,
) -> None:
    """Emit one ``objective_review`` decision per review pass (AGGREGATED).

    A single ``objective_review`` event is emitted per ``review_objectives``
    call, with counters aggregated across all chunked sub-calls. Rationale
    interpolates dynamic signals (≥20 chars): objectives reviewed, #
    statements adjusted, # reverted-for-grounding, # bloom edits
    applied/rejected, unknown ids dropped, model id, provider, plus the new
    ``chunks_attempted`` / ``chunks_failed`` pair — so captures are replayable.
    """
    if capture is None:
        return
    reviewed = counters.get("reviewed", 0)
    decision = (
        f"objective_review:{provider}:"
        f"{counters.get('statements_adjusted', 0)}adj/"
        f"{counters.get('statements_reverted', 0)}revert/"
        f"{counters.get('remaps_applied', 0)}remap/"
        f"{chunks_failed if chunks_failed is not None else 0}of"
        f"{chunks_attempted if chunks_attempted is not None else 0}fail"
        if applied
        else f"objective_review:{provider}:no-op:{note or 'unknown'}"
    )
    rationale = (
        f"Grounding-safe CHUNKED objective-review pass over "
        f"{total_objectives} objectives (course-planning); provider="
        f"{provider}, model={model}; "
        f"chunks_attempted={chunks_attempted}, "
        f"chunks_failed={chunks_failed}, "
        f"reviewed={reviewed}, "
        f"statements_adjusted={counters.get('statements_adjusted', 0)}, "
        f"statements_reverted_for_grounding="
        f"{counters.get('statements_reverted', 0)}, "
        f"bloom_edits_applied={counters.get('bloom_edits_applied', 0)}, "
        f"bloom_edits_rejected={counters.get('bloom_edits_rejected', 0)}, "
        f"unknown_ids_dropped={counters.get('unknown_ids_dropped', 0)}, "
        f"co_to_remaps_applied={counters.get('remaps_applied', 0)}, "
        f"co_to_remaps_reverted_for_grounding="
        f"{counters.get('remaps_reverted', 0)}, "
        f"co_to_remaps_unknown_to_dropped="
        f"{counters.get('remaps_unknown_to_dropped', 0)}, "
        f"cosine_guardrail_active={scorer_available}, "
        f"applied={applied}, note={note or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the phase
        logger.debug(
            "DecisionCapture.log_decision raised on objective_review: %s", exc
        )


__all__ = [
    "review_objectives",
    "merge_reviewed_objectives",
    "resolve_objective_review_provider",
    "resolve_objective_review_model",
    "resolve_review_timeout",
    "ENV_REVIEW_PROVIDER",
    "ENV_REVIEW_MODEL",
]
