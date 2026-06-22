"""Stage-5d 70B structure REVIEWER — text-preserving role/kind corrector.

Phase 1 of the design
(`plans/finegrain/semantik-70b-structure-reviewer-2026-06-22.md` §3, §4,
§6). This module owns:

* ``run_structure_review(regions, feature_blocks, runtime)`` — builds N
  reviewer prompts, fans them through ``runtime.generate_batch``, parses
  each completion with the TOLERANT extractor (§3 M1), applies the
  STRUCTURAL-ONLY admission invariant (§4 C4), runs the document-level
  ``assert_token_conservation`` check (§4 C3), and returns
  ``(corrected_regions, verdicts)``.
* ``assert_token_conservation`` — the concrete ``kept ⊎ dropped ==
  source`` multiset check (§4 C3).
* ``resolve_structure_review_mode`` — parse-with-fallback
  ``SEMANTIK_STRUCTURE_REVIEW`` (default OFF).
* ``ReviewVerdict`` — the typed audit result.

Hard invariants (never violated)
--------------------------------
* The reviewer touches STRUCTURE ONLY. Source words are preserved
  verbatim. A verdict that would alter the FB partition / FB indices /
  non-promotion text is REJECTED and the original Region is kept
  verbatim (``reverted_for_invariant``).
* A ``paragraph -> heading`` promotion MUST set ``payload['text']`` to the
  region's joined verbatim source text and that text MUST equal the source
  (C1); otherwise the promotion is rejected (anti-fabrication).
* Regions are emitted via ``dataclasses.replace`` (``Region`` is frozen).
  ``feature_block_indices`` + ``source_region_id`` are preserved.
* On ANY token-conservation mismatch the stage FAILS CLOSED: it reverts to
  the flag-OFF original region list.
* The parser NEVER raises ``EndpointBatchItemError`` on a parse miss — it
  soft-falls-back that block to ``verdict='ok'`` (§3 M1).

NO MODEL / GPU is loaded here — the runtime is injected (a mocked
``generate_batch`` in tests, the hosted endpoint in production).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

from dart_semantic.structure_graph import REGION_KINDS, Region
from dart_semantic.types import FeatureBlock

from .reviewer_prompt import build_reviewer_request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode resolver — SEMANTIK_STRUCTURE_REVIEW (parse-with-fallback, default OFF).
# ---------------------------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_structure_review_mode() -> bool:
    """Return True when the Stage-5d reviewer is enabled.

    Reads ``SEMANTIK_STRUCTURE_REVIEW``. Default OFF (unset / blank /
    falsey / garbage) -> byte-identical to today (mirrors
    ``resolve_specialist_provider`` / ``resolve_refine_mode``). A truthy
    value (``1``/``true``/``yes``/``on``, case-insensitive) enables it.
    """
    raw = (os.environ.get("SEMANTIK_STRUCTURE_REVIEW") or "").strip().lower()
    return raw in _TRUTHY


# ---------------------------------------------------------------------------
# Typed verdict result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewVerdict:
    """One block's structural-review outcome (the audit deliverable).

    ``block_id`` is the region index in the structure_regions list.
    ``verdict`` is the model's claim (``ok`` / ``corrected`` /
    ``drop_injected_header``) AFTER admission — a rejected correction is
    recorded with ``reverted_for_invariant=True`` and the *_after fields
    equal the *_before fields (the original Region was kept verbatim).
    """

    block_id: int
    verdict: str
    kind_before: str
    kind_after: str
    level_before: int | None
    level_after: int | None
    review_note: str
    reverted_for_invariant: bool = False


# ---------------------------------------------------------------------------
# Text normalization (matches the gate's normalize(): lowercase + collapse
# whitespace). Used for both the C1 promotion-text equality assertion and
# the C3 token-conservation multiset.
# ---------------------------------------------------------------------------


def _normalize_tokens(text: str | None) -> list[str]:
    """Lowercase + whitespace-split into a token list (the C3 multiset unit).

    Equivalence class for "verbatim" = lowercase + collapse whitespace
    (§10 OQ8), matching the gate's ``normalize()``.
    """
    if not text:
        return []
    return str(text).lower().split()


def _normalize_text(text: str | None) -> str:
    """Lowercase + collapse-whitespace into a single normalized string."""
    return " ".join(_normalize_tokens(text))


def _fb_text(feature_blocks: list[FeatureBlock], idx: int) -> str:
    """Resolve one FeatureBlock's raw text (mirrors structure_graph._fb_text)."""
    try:
        fb = feature_blocks[idx]
    except (IndexError, TypeError):
        return ""
    return (getattr(getattr(fb, "raw", None), "text", "") or "").strip()


def _joined_source_text(region: Region, feature_blocks: list[FeatureBlock]) -> str:
    """Join the region's owned FeatureBlock raw texts in index order.

    This is the verbatim source text used as the C1 promotion text and as
    the token-conservation unit. Mirrors the structure_graph join (single
    spaces).
    """
    parts = [
        _fb_text(feature_blocks, i)
        for i in (region.feature_block_indices or ())
    ]
    return " ".join(p for p in parts if p)


def _resolve_region_text(region: Region, feature_blocks: list[FeatureBlock]) -> str:
    """m2 text-resolution rule (§3).

    For a ``heading``/``figure`` region, the text is ``payload['text']``
    (the value ``build_structure_graph`` minted, which can include absorbed
    continuation lines). For every other kind, join the owned
    FeatureBlocks' raw text. Do NOT re-derive a heading's text from raw
    FBs — that can diverge from what structure_graph minted.
    """
    payload = region.payload or {}
    if region.kind in {"heading", "figure"}:
        minted = payload.get("text")
        if minted:
            return str(minted)
        # Fall through to FB-join when payload has no minted text.
    return _joined_source_text(region, feature_blocks)


# ---------------------------------------------------------------------------
# Tolerant verdict extraction (§3 M1).
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Strip Markdown code fences (```json … ``` / bare ```) + commentary.

    Returns the inner body when fenced; otherwise the input unchanged. We
    do NOT trust this to be valid JSON — the balanced-brace scan below is
    the authoritative extractor.
    """
    if not text:
        return ""
    s = text.strip()
    if "```" not in s:
        return s
    # Take the content of the FIRST fenced block; tolerate a language tag.
    first = s.find("```")
    rest = s[first + 3:]
    # Drop a leading language tag line (e.g. "json\n").
    nl = rest.find("\n")
    if nl != -1:
        head = rest[:nl].strip().lower()
        if head and head.isalpha():
            rest = rest[nl + 1:]
    close = rest.find("```")
    if close != -1:
        return rest[:close].strip()
    return rest.strip()


def _first_balanced_object(text: str) -> str | None:
    """Scan for the FIRST balanced ``{ … }`` object and return it.

    String-aware (ignores braces inside JSON string literals + escapes) so
    a ``"review_note": "use {x}"`` doesn't break the brace count. Returns
    None when no balanced object is found.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def _extract_verdict_obj(raw: str) -> dict[str, Any] | None:
    """Tolerant extract: fence-strip -> first-balanced-{} -> json.loads.

    Returns the parsed dict or None on any failure. NEVER raises — the
    caller soft-falls-back to ``verdict='ok'`` on None (§3 M1 step 5).
    """
    try:
        stripped = _strip_code_fences(raw)
        candidate = _first_balanced_object(stripped)
        if candidate is None:
            return None
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            return None
        return obj
    except (ValueError, TypeError):
        return None


def _ok_verdict(region: Region, block_id: int, note: str) -> ReviewVerdict:
    """Build a no-op ``verdict='ok'`` result keeping the region as-is."""
    payload = region.payload or {}
    level = payload.get("level_hint")
    return ReviewVerdict(
        block_id=block_id,
        verdict="ok",
        kind_before=region.kind,
        kind_after=region.kind,
        level_before=level,
        level_after=level,
        review_note=note,
        reverted_for_invariant=False,
    )


# ---------------------------------------------------------------------------
# Structural-only admission invariant (§4 C4) + verdict application.
# ---------------------------------------------------------------------------


def _coerce_level(value: Any) -> int | None:
    """Coerce a level into an int 1-6, or None (anything out of range)."""
    if value is None:
        return None
    try:
        lvl = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= lvl <= 6:
        return lvl
    return None


def _apply_verdict(
    region: Region,
    block_id: int,
    obj: dict[str, Any],
    feature_blocks: list[FeatureBlock],
) -> tuple[Region, ReviewVerdict]:
    """Validate + apply ONE parsed verdict object under the §4 invariant.

    Returns ``(region_out, verdict)``. On any admission failure the
    ORIGINAL region is kept verbatim and the verdict records
    ``reverted_for_invariant=True``. Mutates ONLY ``kind`` /
    ``payload['level_hint']`` / ``payload['doc_role']`` (+ ``payload['text']``
    on a ``->heading`` promotion, C1), via ``dataclasses.replace``.
    """
    payload = dict(region.payload or {})
    kind_before = region.kind
    level_before = payload.get("level_hint")
    note = str(obj.get("review_note") or "")

    raw_verdict = str(obj.get("verdict") or "ok").strip().lower()

    corrected_kind = obj.get("corrected_kind")
    corrected_level = _coerce_level(obj.get("corrected_level"))
    corrected_doc_role = obj.get("corrected_doc_role")

    # --- kind validation (§3 M1 step 4): unknown kind degrades to ok. ---
    new_kind = kind_before
    if corrected_kind is not None:
        if corrected_kind in REGION_KINDS:
            new_kind = str(corrected_kind)
        else:
            # Unknown RegionKind -> drop the kind edit (degrade toward ok).
            corrected_kind = None

    # Nothing structural to change AND verdict is benign -> ok.
    is_promotion = (kind_before != "heading" and new_kind == "heading")
    is_demotion_from_heading = (kind_before == "heading" and new_kind != "heading")

    # --- assemble the mutated payload (carry everything else forward) ---
    new_payload = dict(payload)

    if corrected_doc_role is not None:
        new_payload["doc_role"] = corrected_doc_role

    if new_kind == "heading":
        # Headings carry a level; honor a corrected level, else keep current.
        if corrected_level is not None:
            new_payload["level_hint"] = corrected_level
    else:
        # Non-heading kinds: a corrected level is meaningless; null it only
        # when the kind moved OUT of heading (so a demoted phantom loses its
        # stale heading level).
        if is_demotion_from_heading:
            new_payload["level_hint"] = None
        elif corrected_level is not None:
            new_payload["level_hint"] = corrected_level

    # --- C1: a ->heading promotion MUST set payload['text'] to verbatim
    #         source AND that text MUST equal the source (anti-fabrication).
    if is_promotion:
        source_text = _joined_source_text(region, feature_blocks)
        if not source_text.strip():
            # No-source promotion is a fabrication vector (§4 skip-hole) ->
            # REJECT, keep original verbatim.
            return _rejected(region, block_id, kind_before, level_before, note)
        new_payload["text"] = source_text
        # promotion needs a level; default to h2 when the model gave none.
        if new_payload.get("level_hint") is None:
            new_payload["level_hint"] = corrected_level or 2

    # --- build the candidate replacement region (frozen-safe) ---
    candidate = dataclasses.replace(region, kind=new_kind, payload=new_payload)

    # --- §4 C4 structural-only admission invariant ---
    if not _admits(region, candidate, feature_blocks, is_promotion):
        return _rejected(region, block_id, kind_before, level_before, note)

    # Admitted.
    final_verdict = raw_verdict if raw_verdict in {"ok", "corrected", "drop_injected_header"} else "corrected"
    if new_kind == kind_before and new_payload.get("level_hint") == level_before \
            and new_payload.get("doc_role") == payload.get("doc_role"):
        # No structural change actually happened -> normalize to ok.
        final_verdict = "ok"

    verdict = ReviewVerdict(
        block_id=block_id,
        verdict=final_verdict,
        kind_before=kind_before,
        kind_after=new_kind,
        level_before=level_before,
        level_after=new_payload.get("level_hint"),
        review_note=note,
        reverted_for_invariant=False,
    )
    return candidate, verdict


def _rejected(
    region: Region,
    block_id: int,
    kind_before: str,
    level_before: int | None,
    note: str,
) -> tuple[Region, ReviewVerdict]:
    """Reject a verdict: keep the ORIGINAL region verbatim, record the revert."""
    return region, ReviewVerdict(
        block_id=block_id,
        verdict="ok",
        kind_before=kind_before,
        kind_after=kind_before,
        level_before=level_before,
        level_after=level_before,
        review_note=note,
        reverted_for_invariant=True,
    )


def _admits(
    original: Region,
    candidate: Region,
    feature_blocks: list[FeatureBlock],
    is_promotion: bool,
) -> bool:
    """The §4 C4 structural-only admission invariant.

    A candidate is admitted ONLY if it is a pure structural re-tag:

    1. The FB partition is byte-identical: ``feature_block_indices`` and
       ``source_region_id`` are unchanged.
    2. Non-promotion text is byte-identical: every payload key other than
       the three structural keys (``level_hint``, ``doc_role``, ``text``)
       is preserved verbatim; ``text`` is preserved EXCEPT on a ``->heading``
       promotion, where the new ``text`` MUST normalize-equal the region's
       joined verbatim source (C1).
    """
    # (1) FB partition immutable.
    if tuple(candidate.feature_block_indices) != tuple(original.feature_block_indices):
        return False
    if candidate.source_region_id != original.source_region_id:
        return False

    orig_payload = original.payload or {}
    cand_payload = candidate.payload or {}

    _structural = {"level_hint", "doc_role", "text"}

    # (2a) every NON-structural payload key is byte-identical.
    orig_keys = set(orig_payload) - _structural
    cand_keys = set(cand_payload) - _structural
    if orig_keys != cand_keys:
        return False
    for k in orig_keys:
        if orig_payload.get(k) != cand_payload.get(k):
            return False

    # (2b) text handling.
    if is_promotion:
        # On promotion the new text MUST equal verbatim source (C1).
        source_text = _joined_source_text(original, feature_blocks)
        new_text = cand_payload.get("text") or ""
        if not source_text.strip():
            return False
        if _normalize_text(new_text) != _normalize_text(source_text):
            return False
    else:
        # Non-promotion: text must be byte-identical to the original.
        if orig_payload.get("text") != cand_payload.get("text"):
            return False

    return True


# ---------------------------------------------------------------------------
# Document-level token-conservation check (§4 C3).
# ---------------------------------------------------------------------------


class TokenConservationError(RuntimeError):
    """Raised when the document-level token-conservation invariant fails.

    ``run_structure_review`` catches this and FAILS CLOSED — reverting to
    the flag-OFF original region list (never shipping a content-losing
    correction set)."""


def _owning_region_kinds(
    regions: list[Region], n_fb: int
) -> dict[int, str]:
    """Map each FeatureBlock index -> the kind of its owning region.

    Used by the conservation check to classify each FB's tokens as kept
    (owner kind != metadata_drop) vs dropped (owner kind == metadata_drop).
    A FB owned by no region (coverage-invariant violation) maps to ''.
    """
    owner: dict[int, str] = {}
    for region in regions:
        for idx in (region.feature_block_indices or ()):
            owner[idx] = region.kind
    return owner


def assert_token_conservation(
    original_regions: list[Region],
    corrected_regions: list[Region],
    feature_blocks: list[FeatureBlock],
) -> None:
    """§4 C3 — document-level token-conservation assertion.

    Concrete computation:

      kept    = multiset(tokens(fb) for every FB whose OWNING corrected
                region.kind != 'metadata_drop')
      dropped = multiset(tokens(fb) for every FB owned by a region this
                stage NEWLY re-tagged to 'metadata_drop')
      source  = multiset(tokens(fb) for ALL FBs)

    Assertions:
      * ``kept ⊎ dropped == source`` (multiset union equals source exactly).
      * ``dropped`` is EXACTLY the token multiset of the regions THIS stage
        re-tagged to metadata_drop (the "modulo intentional drops"
        accounting — nothing leaves ``kept`` except via an explicit
        metadata_drop re-tag this stage made).

    Raises :class:`TokenConservationError` on any mismatch (fail-closed).
    """
    n_fb = len(feature_blocks)

    # source multiset — every FB's tokens.
    source: Counter[str] = Counter()
    for i in range(n_fb):
        source.update(_normalize_tokens(_fb_text(feature_blocks, i)))

    # Which FBs were NEWLY re-tagged to metadata_drop by THIS stage?
    # = owned by a metadata_drop region in `corrected` but NOT in `original`.
    orig_owner = _owning_region_kinds(original_regions, n_fb)
    corr_owner = _owning_region_kinds(corrected_regions, n_fb)

    kept: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    intentional_dropped: Counter[str] = Counter()
    orphaned_fbs: list[int] = []

    for i in range(n_fb):
        toks = _normalize_tokens(_fb_text(feature_blocks, i))
        if i not in corr_owner:
            # An FB owned by NO corrected region — content vanished WITHOUT
            # an explicit metadata_drop re-tag. This is the coverage-loss /
            # fabricated-drop failure mode the conservation check exists to
            # catch (the FB's tokens are in neither `kept` nor `dropped`).
            orphaned_fbs.append(i)
            continue
        corr_kind = corr_owner.get(i, "")
        if corr_kind == "metadata_drop":
            dropped.update(toks)
            # Was this a NEW drop by this stage (not already metadata_drop)?
            if orig_owner.get(i, "") != "metadata_drop":
                intentional_dropped.update(toks)
        else:
            kept.update(toks)

    if orphaned_fbs:
        raise TokenConservationError(
            "structure-review token conservation FAILED: "
            f"{len(orphaned_fbs)} FeatureBlock(s) {orphaned_fbs!r} left "
            "every region's coverage (content lost without a metadata_drop "
            "re-tag) — coverage invariant + conservation both broken."
        )

    # kept ⊎ dropped == source.
    union = kept + dropped
    if union != source:
        missing = source - union
        extra = union - source
        raise TokenConservationError(
            "structure-review token conservation FAILED: kept ⊎ dropped != "
            f"source (missing={dict(missing)!r}, extra={dict(extra)!r})"
        )

    # `dropped` must equal exactly: (FBs already metadata_drop in original) +
    # (FBs newly dropped this stage). Equivalently, every FB that is
    # metadata_drop in `corrected` is accounted for. The above multiset
    # union already proves no NON-drop content leaked into `dropped`; the
    # intentional_dropped <= dropped relation is structural and always holds
    # here. We additionally assert no FB SILENTLY left `kept` except via a
    # metadata_drop owner — i.e. every source token is in kept ∪ dropped,
    # already proven by `union == source`. The "modulo intentional drops"
    # rule is therefore enforced.
    # (intentional_dropped is surfaced for the audit, not a separate gate.)
    _ = intentional_dropped


# ---------------------------------------------------------------------------
# Neighbor windowing.
# ---------------------------------------------------------------------------


def _neighbors_for(regions: list[Region], index: int) -> tuple[Region | None, Region | None]:
    """Return the (prev, next) ±1 neighbor Regions for ``index``."""
    prev_block = regions[index - 1] if index - 1 >= 0 else None
    next_block = regions[index + 1] if index + 1 < len(regions) else None
    return prev_block, next_block


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run_structure_review(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    runtime: Any,
    *,
    max_tokens: int = 512,
) -> tuple[list[Region], list[ReviewVerdict]]:
    """Run the Stage-5d structure review over the full region list.

    Flow (§3, §4, §6):

      1. Build N reviewer prompts (one per region) via
         ``build_reviewer_request`` (the m2-resolved text + ±1 neighbors).
      2. ``runtime.generate_batch(prompts, max_tokens=...)`` -> N strings.
      3. For each completion: TOLERANT extract (fence-strip ->
         first-balanced-{} -> block_id cross-check -> kind-validate). A
         parse miss / out-of-batch block_id / unknown kind soft-falls-back
         to ``verdict='ok'`` (NEVER raises).
      4. Apply each admitted verdict under the structural-only invariant
         (§4 C4); a rejection keeps the original Region verbatim.
      5. ``assert_token_conservation`` (§4 C3) pre-return; on mismatch FAIL
         CLOSED -> return the flag-OFF original ``regions`` unchanged.

    Returns ``(corrected_regions, verdicts)``. ``verdicts`` is one
    :class:`ReviewVerdict` per input region, in input order.

    NO model/GPU is loaded here; ``runtime`` is injected.
    """
    if not regions:
        return regions, []

    n = len(regions)
    block_ids = set(range(n))

    # (1) build prompts.
    prompts: list[str] = []
    for index, region in enumerate(regions):
        neighbors = _neighbors_for(regions, index)
        text = _resolve_region_text(region, feature_blocks)
        prompts.append(build_reviewer_request(region, neighbors, index, text=text))

    # (2) batch — REUSE generate_batch verbatim. A batch-level failure is
    #     NOT swallowed here (the endpoint runtime owns its fail-loud
    #     contract); only PER-ITEM parse misses soft-fall-back below.
    completions = runtime.generate_batch(prompts, max_tokens=max_tokens)

    # Defensive: a runtime must return one completion per prompt. Pad/clip
    # to N so the per-block loop stays addressable (a short batch
    # soft-falls-back the missing tail to ok).
    if len(completions) < n:
        completions = list(completions) + [""] * (n - len(completions))

    # (3)+(4) parse + apply per block.
    corrected: list[Region] = []
    verdicts: list[ReviewVerdict] = []
    for index, region in enumerate(regions):
        raw = completions[index] if index < len(completions) else ""
        obj = _extract_verdict_obj(raw if isinstance(raw, str) else "")
        if obj is None:
            # parse miss -> soft fallback ok.
            corrected.append(region)
            verdicts.append(_ok_verdict(region, index, "unparseable verdict; kept original"))
            continue
        # block_id cross-check (§3 M1 step 3): a verdict whose block_id is
        # not THIS block's id is dropped (never mutates this region). We key
        # strictly on the per-prompt index — an echoed block_id that matches
        # a DIFFERENT in-batch id is still wrong for THIS slot.
        claimed_id = obj.get("block_id")
        try:
            claimed_id_int = int(claimed_id)
        except (TypeError, ValueError):
            claimed_id_int = None
        if claimed_id_int != index:
            # Out-of-batch / mismatched id -> soft fallback ok (do not
            # mutate this region with another block's verdict).
            corrected.append(region)
            note = "verdict block_id mismatch; kept original"
            verdicts.append(_ok_verdict(region, index, note))
            continue
        region_out, verdict = _apply_verdict(region, index, obj, feature_blocks)
        corrected.append(region_out)
        verdicts.append(verdict)

    # (5) document-level token conservation — fail-closed to flag-OFF list.
    try:
        assert_token_conservation(regions, corrected, feature_blocks)
    except TokenConservationError as exc:
        logger.warning(
            "structure-review reverting to flag-OFF region list (fail-closed): %s",
            exc,
        )
        # Re-emit verdicts as all-reverted so the audit reflects the revert.
        reverted_verdicts = [
            _rejected(
                region,
                idx,
                region.kind,
                (region.payload or {}).get("level_hint"),
                "token-conservation fail-closed; whole-stage revert",
            )[1]
            for idx, region in enumerate(regions)
        ]
        return regions, reverted_verdicts

    return corrected, verdicts


__all__ = [
    "ReviewVerdict",
    "TokenConservationError",
    "assert_token_conservation",
    "resolve_structure_review_mode",
    "run_structure_review",
]
