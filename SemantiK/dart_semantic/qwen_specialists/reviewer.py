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
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

from dart_semantic.structure_graph import REGION_KINDS, Region, _get_signal, _top1
from dart_semantic.types import FeatureBlock

from .endpoint_runtime import EndpointRuntimeError
from .reviewer_prompt import (
    build_edge_input,
    build_reviewer_request,
    build_windowed_reviewer_request,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cluster-level structural signals (Phase 4 root-cause fix).
#
# The reviewer's per-region ±neighbor window cannot tell a phantom-TOC /
# chapter-index entry (a "Chapter 5" wedged between "Chapter 3" and
# "Chapter 7") from a real chapter opener — locally they are identical. The
# DISCRIMINATOR is cluster-level:
#   (a) a phantom-TOC/index entry is part of a RUN of consecutive same-level
#       headings with NO content-bearing region between consecutive entries;
#   (b) a real section heading is FOLLOWED BY content (paragraph/list/table/…)
#       before the next heading.
# We compute four deterministic per-region signals over the FULL ordered
# region list (cheap, no model) and feed them into the reviewer prompt so the
# 70B can reason about the cluster, not just the neighbor text.
#
# The content-bearing predicate + the trailing-page-number TOC pattern are
# kept in lock-step with ``lib/semantik/toc_frontmatter_detector.py`` — the
# deterministic backstop — so prompt-signal and backstop agree on what counts
# as content / a TOC line.
# ---------------------------------------------------------------------------

# The only heading kind in REGION_KINDS; isolated as a frozenset so the
# heading-scoping in run_structure_review and the run computation share one
# definition (and a future second heading kind is a one-line change).
HEADING_KINDS: frozenset[str] = frozenset({"heading"})

# Kinds that are NEITHER content-bearing NOR headings — they do not break a
# same-level heading run (a stray running header / dropped furniture between
# two index entries keeps the run open), mirroring the detector's treatment
# of ``heading`` / ``metadata_drop`` / "" in ``_is_content_bearing``.
_NON_CONTENT_NON_HEADING_KINDS: frozenset[str] = frozenset({"metadata_drop"})

# Trailing bare-page-number pattern (the TOC line shape) — a title with >=1
# alphabetic word followed by leader (spaces / dots / ellipsis) + a trailing
# integer. Byte-identical to ``toc_frontmatter_detector._TOC_ENTRY_RE`` so the
# ``trailing_pagenum`` prompt-signal agrees with the deterministic backstop.
_TRAILING_PAGENUM_RE = re.compile(
    r"^\s*(?P<title>.*?[A-Za-z].*?)"  # title with >=1 letter
    r"[\s\.…]+"                  # leader: spaces / dots / ellipsis
    r"(?P<page>\d{1,5})\s*$"           # trailing page number
)


def _region_is_content_bearing(region: Region) -> bool:
    """Whether a region carries real teaching content (paragraph / list /
    table / code / blockquote / figure / form / math / definition_list).

    A heading or a dropped metadata region is NOT content-bearing. This is the
    discriminator between a rendered chapter/section INDEX (headings
    back-to-back with no content between them) and real openers (each followed
    by content). Mirrors ``toc_frontmatter_detector._is_content_bearing`` —
    NOT content-bearing iff kind in {heading, metadata_drop, ""}.
    """
    kind = str(region.kind or "")
    if kind in HEADING_KINDS or kind in _NON_CONTENT_NON_HEADING_KINDS or kind == "":
        return False
    return True


def _region_heading_level(region: Region) -> int | None:
    """The heading level for a heading region (``payload['level_hint']``)."""
    if str(region.kind) not in HEADING_KINDS:
        return None
    return (region.payload or {}).get("level_hint")


def _has_trailing_pagenum(text: str | None) -> bool:
    """Whether ``text`` ends with a bare page number (the TOC-line shape)."""
    if not text:
        return False
    return _TRAILING_PAGENUM_RE.match(str(text)) is not None


@dataclass(frozen=True)
class ClusterSignals:
    """The four deterministic cluster-level signals for ONE region.

    Computed over the full ordered region list (see
    :func:`compute_cluster_signals`). Only meaningful for heading regions;
    a non-heading region carries the inert defaults
    (``same_level_run_len=0``, ``run_position=0``, ``trailing_pagenum=False``)
    and its ``content_blocks_following`` is not consulted by the reviewer.
    """

    same_level_run_len: int
    run_position: int
    content_blocks_following: int
    trailing_pagenum: bool


# Default level used when a heading region carries no/None ``level_hint`` — a
# deep-but-safe sentinel so a level-less heading is treated as the LOWEST in the
# hierarchy: it never closes another heading's content scope (only a
# same-or-higher heading does, and a sentinel-6 heading is same-or-higher only
# to another sentinel/level-6), and its OWN scope is closed by the very next
# heading. This is the conservative direction — a level-less heading can never
# spuriously inflate a neighbor's content_blocks_following.
_DEFAULT_HEADING_LEVEL = 6


def _heading_level_or_default(region: Region) -> int:
    """The heading region's level, defaulting missing/None to the deep
    sentinel ``_DEFAULT_HEADING_LEVEL`` (lowest in hierarchy). Only meaningful
    for heading regions; callers gate on ``kind in HEADING_KINDS`` first."""
    level = _region_heading_level(region)
    if level is None:
        return _DEFAULT_HEADING_LEVEL
    try:
        return int(level)
    except (TypeError, ValueError):
        return _DEFAULT_HEADING_LEVEL


def _content_blocks_following(regions: list[Region], index: int) -> int:
    """Count content-bearing regions after ``index``, up to the next
    SAME-OR-HIGHER-level heading.

    A heading whose level number is ``<=`` this heading's level (lower level
    number = higher in the hierarchy: h1=chapter, h2=section, h3=subsection)
    CLOSES this heading's content scope. A strictly-LOWER-in-hierarchy
    sub-heading (a larger level number — e.g. a level-3 "Learning Objectives"
    nested under a level-2 "1.1" opener) is TRANSPARENT: the content under it
    counts toward THIS opener.

    This closes the heading-nested under-protection: a real level-2 "1.1"
    opener whose body is laid out under level-3 sub-headings ("Learning
    Objectives" / "EXAMPLE" / "Solution") — never a paragraph DIRECTLY beneath
    it — now sees content_blocks_following >= 1 (PROTECTED), while a phantom
    level-1 chapter-index entry wedged in a same-level run with no content
    before the next level-1 entry still scores 0 (flagged phantom).

    A non-heading ``index`` keeps the old neighbor-window semantics: count
    content-bearing regions up to the next heading of ANY level (it has no
    level to scope against).
    """
    own_kind = str(regions[index].kind)
    own_is_heading = own_kind in HEADING_KINDS
    own_level = _heading_level_or_default(regions[index]) if own_is_heading else None

    count = 0
    for j in range(index + 1, len(regions)):
        if str(regions[j].kind) in HEADING_KINDS:
            if not own_is_heading:
                # Non-heading anchor: the next heading of any level closes it.
                break
            other_level = _heading_level_or_default(regions[j])
            if other_level <= own_level:
                # Same-or-higher in the hierarchy -> closes this scope.
                break
            # Strictly-lower sub-heading -> transparent; keep counting.
            continue
        if _region_is_content_bearing(regions[j]):
            count += 1
    return count


def compute_cluster_signals(regions: list[Region]) -> list[ClusterSignals]:
    """Compute the four cluster-level signals for every region (model-free).

    Scans the FULL ordered region list (not just heading regions — the runs
    must be computed correctly across the whole document) and returns one
    :class:`ClusterSignals` per region, in input order.

    Signals (per region):
      * ``same_level_run_len`` — length of the maximal contiguous run of
        same-``level`` heading regions that this heading belongs to, where a
        run is broken ONLY by a content-bearing region between two consecutive
        entries (a non-content non-heading region — e.g. metadata_drop — does
        NOT break the run, mirroring the detector). 0 for a non-heading.
      * ``run_position`` — this heading's 1-based position within that run.
        0 for a non-heading.
      * ``content_blocks_following`` — count of content-bearing regions
        after this heading, up to the next SAME-OR-HIGHER-level heading (a
        heading whose level number is ``<=`` this one). A strictly-lower-in-
        hierarchy sub-heading (larger level number) is TRANSPARENT — content
        nested under it counts toward this opener (protects a real "1.1"
        section opener whose body is laid out under level-3 sub-headings,
        NOT directly under a paragraph).
      * ``trailing_pagenum`` — does the heading text end with a bare page
        number (the TOC pattern)?
    """
    n = len(regions)
    signals: list[ClusterSignals] = [
        ClusterSignals(0, 0, 0, False) for _ in range(n)
    ]
    if n == 0:
        return signals
    if n == 0:
        return signals

    # Precompute content_blocks_following + trailing_pagenum for every region.
    cbf = [_content_blocks_following(regions, i) for i in range(n)]
    trailing = [
        _has_trailing_pagenum(_region_payload_text(regions[i]))
        if str(regions[i].kind) in HEADING_KINDS
        else False
        for i in range(n)
    ]

    # Compute same-level heading runs. Walk the ordered list; a run is a
    # maximal contiguous sequence of HEADING regions at the SAME level with no
    # content-bearing region between consecutive entries (non-content
    # non-heading regions are transparent — they neither extend nor break it).
    i = 0
    while i < n:
        if str(regions[i].kind) not in HEADING_KINDS:
            i += 1
            continue
        run_level = _region_heading_level(regions[i])
        run_indices = [i]
        j = i + 1
        while j < n:
            kind_j = str(regions[j].kind)
            if _region_is_content_bearing(regions[j]):
                break  # real content between entries -> run ends.
            if kind_j in HEADING_KINDS:
                if _region_heading_level(regions[j]) == run_level:
                    run_indices.append(j)
                    j += 1
                    continue
                # A different-level heading breaks this same-level run.
                break
            # A non-content non-heading region (metadata_drop) is transparent.
            j += 1
        run_len = len(run_indices)
        for pos, idx in enumerate(run_indices, start=1):
            signals[idx] = ClusterSignals(
                same_level_run_len=run_len,
                run_position=pos,
                content_blocks_following=cbf[idx],
                trailing_pagenum=trailing[idx],
            )
        i = max(run_indices[-1] + 1, i + 1)

    # Non-heading regions keep their inert default but still carry the
    # (rarely-consulted) content_blocks_following for completeness.
    for idx in range(n):
        if str(regions[idx].kind) not in HEADING_KINDS:
            signals[idx] = ClusterSignals(0, 0, cbf[idx], False)

    return signals


def _region_payload_text(region: Region) -> str:
    """The heading text from payload (``payload['text']``) for run/pagenum
    signal computation; empty string when absent."""
    return str((region.payload or {}).get("text") or "")


# ---------------------------------------------------------------------------
# Mode resolver — SEMANTIK_STRUCTURE_REVIEW (parse-with-fallback, default OFF).
# ---------------------------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on"})


# Anti-crawl short-circuit threshold. If at least this FRACTION of the first
# batch's heading clusters fail their endpoint call (transient errors, after
# the bounded retries), the endpoint is treated as effectively dead and the
# WHOLE review degrades to unreviewed — we do NOT keep grinding remaining
# clusters through timeout×retries (that was the ~46-minute crawl on the real
# down-endpoint run). Because generate_batch already fans every cluster
# concurrently in ONE call, this fires on that single batch's failure ratio.
_ENDPOINT_DEAD_FAILURE_RATIO = 0.8


def resolve_structure_review_mode() -> bool:
    """Return True when the Stage-5d reviewer is enabled.

    Reads ``SEMANTIK_STRUCTURE_REVIEW``. Default OFF (unset / blank /
    falsey / garbage) -> byte-identical to today (mirrors
    ``resolve_specialist_provider`` / ``resolve_refine_mode``). A truthy
    value (``1``/``true``/``yes``/``on``, case-insensitive) enables it.
    """
    raw = (os.environ.get("SEMANTIK_STRUCTURE_REVIEW") or "").strip().lower()
    return raw in _TRUTHY


# Default sampling temperature for the Stage-5d structure-reviewer dispatch.
# 0.0 = greedy / deterministic decoding: a structure-correction pass should
# NOT introduce run-to-run noise in chapter/section decisions (measured
# heading-set Jaccard 0.91-0.94 across re-runs at the old 0.6 default). An
# operator can opt back into sampling by setting the env > 0. NOTE: greedy
# decoding makes the DISPATCH deterministic at the sampling layer; hosted-
# endpoint / float non-determinism can still cause rare token ties, so this
# is "deterministic decoding (temperature 0)", not an absolute guarantee.
_DEFAULT_STRUCTURE_REVIEW_TEMPERATURE = 0.0


def resolve_structure_review_temperature() -> float:
    """Parse-with-fallback ``SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE`` (default 0.0).

    Returns the sampling temperature threaded into the Stage-5d reviewer's
    ``generate_batch`` dispatch. Default ``0.0`` is greedy / deterministic
    decoding; a positive float opts back into sampling. Garbage / non-float /
    negative / NaN values fall back to ``0.0`` (mirrors ``_resolve_timeout`` /
    ``resolve_specialist_max_retries``)."""
    raw = os.environ.get("SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE")
    if not raw:
        return _DEFAULT_STRUCTURE_REVIEW_TEMPERATURE
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_STRUCTURE_REVIEW_TEMPERATURE
    if val < 0 or val != val:  # negative / NaN
        return _DEFAULT_STRUCTURE_REVIEW_TEMPERATURE
    return val


# ---------------------------------------------------------------------------
# Mode resolvers — SEMANTIK_BLOCK_REVIEW quartet (Phase 0 scaffolding).
#
# The full-block structural-editor reviewer (re-type / merge / split / drop
# over block IDs). These resolvers are DEAD-BUT-IMPORTABLE until Phase 2+
# wires them into the cascade; landing them here keeps later phases to a pure
# gate flip. Bodies are copied verbatim from resolve_structure_review_mode
# (bool) and resolve_specialist_batch_regions (int parse-with-fallback).
# ---------------------------------------------------------------------------

# Falsey tokens for the default-ON cache gate (mirrors
# deterministic_structure._FALSEY / runner._BATCH_FALSEY).
_FALSEY = frozenset({"0", "false", "no", "off"})

# Default edge-windowing budgets for the block reviewer.
_DEFAULT_BLOCK_REVIEW_WINDOW = 24
_DEFAULT_BLOCK_REVIEW_EDGE_TOKENS = 12

# Default council structural_role top-1 confidence floor below which a
# ``code_block`` content region is escalated to the reviewer (§6.2
# deterministic-first). Mirrors structure_graph._CODE_CONFIDENCE_FLOOR (0.4):
# the council emits a code_block at >= 0.4, so a sub-floor code_block is a
# low-confidence type the LLM should re-judge.
_DEFAULT_BLOCK_REVIEW_CONF_FLOOR = 0.4

# Content (non-heading) region kinds the full-block reviewer may re-type when
# SEMANTIK_BLOCK_REVIEW is on. A SEPARATE scope set from HEADING_KINDS so the
# heading-vs-content discriminator (compute_cluster_signals /
# _region_is_content_bearing / _content_blocks_following) keeps reading the
# UNCHANGED HEADING_KINDS — widening that frozenset corrupts cluster signals
# even flag-off. Every member is validated against REGION_KINDS; a token that
# is not a real RegionKind is dropped (never invent a kind).
CONTENT_REVIEW_KINDS: frozenset[str] = frozenset(
    k
    for k in ("paragraph", "code_block", "table", "blockquote", "list")
    if k in REGION_KINDS
)


def resolve_block_review_mode() -> bool:
    """Return True when the Stage-5d full-block structural-editor reviewer is enabled.

    Reads ``SEMANTIK_BLOCK_REVIEW``. Default OFF (unset / blank / falsey /
    garbage) -> byte-identical to today (mirrors
    ``resolve_structure_review_mode``). A truthy value
    (``1``/``true``/``yes``/``on``, case-insensitive) enables it.
    """
    raw = (os.environ.get("SEMANTIK_BLOCK_REVIEW") or "").strip().lower()
    return raw in _TRUTHY


def resolve_block_review_window() -> int:
    """Parse-with-fallback ``SEMANTIK_BLOCK_REVIEW_WINDOW`` (default 24).

    The maximum number of blocks packed into ONE windowed block-review POST.
    Garbage / non-positive values fall back to the default, mirroring
    :func:`resolve_specialist_batch_regions`. Consumed in Phase 4 (windowed
    dispatch); a pure read until then."""
    raw = os.environ.get("SEMANTIK_BLOCK_REVIEW_WINDOW")
    if not raw:
        return _DEFAULT_BLOCK_REVIEW_WINDOW
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BLOCK_REVIEW_WINDOW
    if val <= 0:
        return _DEFAULT_BLOCK_REVIEW_WINDOW
    return val


def resolve_block_review_edge_tokens() -> int:
    """Parse-with-fallback ``SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS`` (default 12).

    The number of head / tail tokens kept per block in the edge-input record
    fed to the reviewer (furniture-deduped, tractable on a 7B). Garbage /
    non-positive values fall back to the default, mirroring
    :func:`resolve_specialist_batch_regions`. Consumed in Phase 1 (edge-input
    builder); a pure read until then."""
    raw = os.environ.get("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS")
    if not raw:
        return _DEFAULT_BLOCK_REVIEW_EDGE_TOKENS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BLOCK_REVIEW_EDGE_TOKENS
    if val <= 0:
        return _DEFAULT_BLOCK_REVIEW_EDGE_TOKENS
    return val


def resolve_block_review_cache_mode() -> bool:
    """Return True when the block-review content-hash window cache is enabled.

    Reads ``SEMANTIK_BLOCK_REVIEW_CACHE``. Default ON (pure memoization,
    output-identical) — explicit falsey (``0``/``false``/``no``/``off``,
    case-insensitive) disables it; unset / blank / truthy / garbage -> on
    (mirrors the project default-on parse-with-fallback pattern,
    ``deterministic_structure.resolve_structure_clean_mode`` /
    ``runner.resolve_batch_mode``). Consumed in Phase 4b; a pure read until
    then."""
    raw = (os.environ.get("SEMANTIK_BLOCK_REVIEW_CACHE") or "").strip().lower()
    return raw not in _FALSEY


def resolve_block_review_conf_floor() -> float:
    """Parse-with-fallback ``SEMANTIK_BLOCK_REVIEW_CONF`` (default 0.4).

    The council ``structural_role`` top-1 confidence below which a
    ``code_block`` content region is escalated to the reviewer (§6.2
    deterministic-first). Mirrors ``_CODE_CONFIDENCE_FLOOR``. Float
    parse-with-fallback (mirrors the Phase-0 int resolvers): blank / non-float
    / NaN / Inf / out-of-``[0.0, 1.0]`` -> the 0.4 default. Consumed by the
    content-block dispatch gate (a pure read until then). Makes the Phase-6
    calibration a no-code env change."""
    raw = os.environ.get("SEMANTIK_BLOCK_REVIEW_CONF")
    if not raw:
        return _DEFAULT_BLOCK_REVIEW_CONF_FLOOR
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BLOCK_REVIEW_CONF_FLOOR
    # NaN / Inf / out-of-range -> default (NaN comparisons are False, so the
    # range guard rejects it too).
    if not (0.0 <= val <= 1.0):
        return _DEFAULT_BLOCK_REVIEW_CONF_FLOOR
    return val


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

    ``reverted_for_endpoint_failure`` records the OTHER safe-revert reason:
    the cluster's endpoint call failed (after the bounded retries) so the
    region was degraded to UNREVIEWED rather than reviewed. It is the
    endpoint-failure sibling of ``reverted_for_invariant`` — both keep the
    original Region verbatim, but this one means "we never got a verdict"
    (a slow/down 70B), not "we got a verdict and rejected it". Defaults
    False (serialization-safe; an off / healthy-endpoint run is byte-stable).
    """

    block_id: int
    verdict: str
    kind_before: str
    kind_after: str
    level_before: int | None
    level_after: int | None
    review_note: str
    reverted_for_invariant: bool = False
    reverted_for_endpoint_failure: bool = False
    # Phase 2 (SEMANTIK_BLOCK_REVIEW): the region's doc_role AFTER a
    # block-review re-type, mirrored from the already-written
    # ``payload['doc_role']``. Optional-default-None + audit-EXCLUDED when
    # None (see cascade.py structure_review_audit) so the heading-only /
    # flag-off path's audit dict stays byte-identical to today. Populated
    # ONLY when block-review is on (no content re-types are produced with
    # the flag off, so this is None on every flag-off verdict).
    role_after: str | None = None


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

    # Phase 2: mirror the already-written ``doc_role`` onto ``role_after``,
    # but ONLY when block-review is on. The kind change itself already flows
    # through the admission path above (``dataclasses.replace`` + ``_admits``);
    # this is the sole net-new apply work. Gating on the flag keeps the
    # heading-only / flag-off path's audit byte-identical to today — a heading
    # whose payload carries a Semantic ``doc_role`` (or a heading verdict that
    # sets ``corrected_doc_role``) does NOT leak a non-None ``role_after``
    # while the flag is off, so no content re-types ⇒ ``role_after`` is None
    # on every flag-off verdict.
    role_after = new_payload.get("doc_role") if resolve_block_review_mode() else None
    verdict = ReviewVerdict(
        block_id=block_id,
        verdict=final_verdict,
        kind_before=kind_before,
        kind_after=new_kind,
        level_before=level_before,
        level_after=new_payload.get("level_hint"),
        review_note=note,
        reverted_for_invariant=False,
        role_after=role_after,
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


def _endpoint_failure_verdict(
    region: Region,
    block_id: int,
    note: str,
) -> ReviewVerdict:
    """Degrade ONE region to UNREVIEWED because its endpoint call failed.

    The endpoint-failure sibling of :func:`_rejected`: the cluster's 70B
    call failed (after the bounded retries) so we never got a verdict for
    this heading. Keep the ORIGINAL Region verbatim (do NOT alter text /
    kind / level) and record the degradation so the audit shows the block
    as endpoint-degraded, NOT silently "reviewed=ok". Maps an endpoint
    failure onto the same safe outcome the token-conservation /
    admission-invariant reverts use."""
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
        reverted_for_endpoint_failure=True,
    )


def _degrade_whole_review_unreviewed(
    regions: list[Region], note: str
) -> tuple[list[Region], list[ReviewVerdict]]:
    """Degrade the ENTIRE review to UNREVIEWED (the byte-stable floor).

    Every region is kept verbatim; heading regions get an endpoint-failure
    verdict and non-headings an ok verdict, so the audit shows the whole
    review as endpoint-degraded rather than silently "reviewed=ok". Shared by
    the anti-crawl dead-endpoint short-circuit and the fail_soft-unsupported
    fallback (a runtime that cannot fail-soft must NOT be re-run fail-loud —
    that re-creates the per-item timeout crawl the 429 fix removes)."""
    verdicts: list[ReviewVerdict] = []
    for index, region in enumerate(regions):
        if str(region.kind) in HEADING_KINDS:
            verdicts.append(_endpoint_failure_verdict(region, index, note))
        else:
            verdicts.append(_ok_verdict(region, index, "non-heading; not reviewed"))
    return regions, verdicts


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
# Phase 3 — review-scope predicate + deterministic-first gate + ambiguity
# escalation. All no-ops when SEMANTIK_BLOCK_REVIEW is off (the heading-only,
# byte-stable path).
# ---------------------------------------------------------------------------

# Pedagogical-label head tokens (OpenStax "TRY IT" / "EXAMPLE 2" / "EXERCISE")
# — a SMALL, documented allowlist. A content block whose first word is one of
# these is a likely mis-typed pedagogical unit and is always sent to the
# reviewer regardless of council confidence (the §6.2 OR-arm). Heading
# pedagogical-label demotion is owned by the always-on deterministic
# clean_structure pass — this set only routes CONTENT-block re-types.
_PEDAGOGICAL_LABEL_PREFIXES: frozenset[str] = frozenset({"TRY", "EXAMPLE", "EXERCISE"})

# §3 single-shot escalation: an ambiguous content-block re-type is re-asked
# ONCE with the FULL verbatim source (token-capped to fit n_ctx) instead of
# the head/tail edge. No recursion (the re-ask is not itself re-escalated).
_ESCALATION_SOURCE_TOKEN_CAP = 2048


def _in_review_scope(region: Region) -> bool:
    """True when ``region`` is eligible for a Stage-5d reviewer prompt.

    Heading regions are ALWAYS in scope (the byte-stable heading-only path).
    Content regions (``CONTENT_REVIEW_KINDS``) are in scope ONLY when
    ``SEMANTIK_BLOCK_REVIEW`` is on. SEPARATE from ``HEADING_KINDS`` so the
    heading-vs-content discriminator keeps reading the unchanged
    ``HEADING_KINDS`` (flag-off reduces this to the exact old predicate
    ``region.kind in HEADING_KINDS``).
    """
    if str(region.kind) in HEADING_KINDS:
        return True
    return resolve_block_review_mode() and str(region.kind) in CONTENT_REVIEW_KINDS


def _has_pedagogical_label_prefix(text: str) -> bool:
    """True when ``text``'s first word is a pedagogical label (TRY/EXAMPLE/EXERCISE)."""
    if not text:
        return False
    parts = text.split(None, 1)
    if not parts:
        return False
    head = parts[0].strip(string.punctuation).upper()
    return head in _PEDAGOGICAL_LABEL_PREFIXES


def _content_block_dispatch_gate(
    region: Region,
    council_state: Any | None,
    feature_blocks: list[FeatureBlock],
) -> bool:
    """Deterministic-first (§6.2): send a content block to the LLM only when uncertain.

    Dispatch iff EITHER
      * the block carries a pedagogical-label prefix (TRY/EXAMPLE/EXERCISE), OR
      * the council kind is ``code_block`` AND the council ``structural_role``
        top-1 confidence is below ``resolve_block_review_conf_floor()``.

    A high-confidence content block returns ``False`` -> NO prompt (the
    deterministic council kind is trusted). ``RoutingDecision.low_confidence``
    is NOT consulted: ``run_structure_review`` is handed a ``CouncilState``
    (per-head signals); RoutingDecisions live on a separate cross-reranker
    object that is not threaded here, so the gate uses council confidence only
    (documented). A missing structural_role signal reads as confidence 0.0
    (``_top1`` default) -> treated as low-confidence -> dispatch (when-in-doubt
    review).
    """
    text = _joined_source_text(region, feature_blocks) or str(
        (region.payload or {}).get("text") or ""
    )
    if _has_pedagogical_label_prefix(text):
        return True
    fb_indices = region.feature_block_indices or ()
    first_idx = fb_indices[0] if fb_indices else None
    if str(region.kind) == "code_block" and council_state is not None and first_idx is not None:
        _, conf = _top1(_get_signal(council_state, "structure", "structural_role", first_idx))
        if conf < resolve_block_review_conf_floor():
            return True
    return False


def _is_ambiguous_verdict(obj: dict[str, Any]) -> bool:
    """True when the model FLAGGED this re-type as genuinely ambiguous (§3).

    The verdict contract carries an OPTIONAL ``ambiguous`` boolean (absent ->
    False, backward-compatible with every existing completion). When truthy, a
    CONTENT-block re-type is escalated ONCE to a fuller-text re-ask. Heading
    verdicts never escalate (the heading path stays byte-stable)."""
    return bool(obj.get("ambiguous"))


def _content_edge_text(
    region: Region,
    index: int,
    feature_blocks: list[FeatureBlock],
    council_state: Any | None,
) -> str:
    """First-pass content-block prompt text: the Phase-1 head/tail edge record.

    Reuses ``build_edge_input`` (default ``edge_tokens`` resolver) so a long
    block is sent head/tail-truncated; a short block carries its full verbatim
    ``text``. The ambiguity escalation re-asks with the fuller capped source.
    """
    rec = build_edge_input(
        region,
        block_id=index,
        feature_blocks=feature_blocks,
        council_state=council_state,
    )
    if "text" in rec:
        return str(rec["text"])
    head = str(rec.get("head", ""))
    tail = str(rec.get("tail", ""))
    return f"{head} … {tail}".strip()


def _capped_source_text(region: Region, feature_blocks: list[FeatureBlock]) -> str:
    """Full verbatim joined source, token-capped to fit n_ctx (escalation text)."""
    src = _joined_source_text(region, feature_blocks)
    tokens = src.split()
    if len(tokens) > _ESCALATION_SOURCE_TOKEN_CAP:
        return " ".join(tokens[:_ESCALATION_SOURCE_TOKEN_CAP])
    return src


# ---------------------------------------------------------------------------
# Phase 4 — windowed dispatch + idx-keyed op-list parse + 1:1 output contract.
# All reached ONLY when SEMANTIK_BLOCK_REVIEW is on (the flag-off heading-only
# one-block-per-prompt path stays byte-identical — see run_structure_review).
# ---------------------------------------------------------------------------

# §11 window overlap: each window shares its FIRST block with the PREVIOUS
# window's LAST block so a stitch group straddling a window boundary is seen
# whole in at least one window. Stride = max(1, window - overlap); ops are
# deduped on idx (FIRST window that yields an op for an idx wins) so the shared
# block is never double-applied.
_BLOCK_REVIEW_WINDOW_OVERLAP = 1

# max_tokens budget per windowed op-list member. A re-type op object
# (``{"idx":12,"verdict":"corrected","corrected_kind":"paragraph",...,"review_note":"..."}``)
# is ~40-60 tokens; 96 leaves headroom so a max-size window's full op-list is
# never silently truncated (the tolerant parser returns FEWER ops with no
# fail-closed signal). Mirrors the resegment LLM's larger max_tokens
# (block_resegment._llm_propose_ops uses 1024 for a whole-doc op-list).
_PER_MEMBER_OP_MAX_TOKENS = 96


def _build_windows(
    indices: list[int], window: int, overlap: int
) -> list[list[int]]:
    """Group ``indices`` into overlapping windows of <= ``window`` members.

    Stride = ``max(1, window - overlap)``; consecutive windows share
    ``overlap`` members so a boundary-straddling group is whole in one window.
    The final (partial) window covers the tail. Returns ``[]`` for empty input.
    """
    if not indices:
        return []
    w = max(1, int(window))
    stride = max(1, w - max(0, int(overlap)))
    n = len(indices)
    windows: list[list[int]] = []
    start = 0
    while start < n:
        windows.append(indices[start : start + w])
        if start + w >= n:
            break
        start += stride
    return windows


def _window_max_tokens(base_max_tokens: int, windows: list[list[int]]) -> int:
    """Scale the dispatch ``max_tokens`` to the LARGEST window's op-list size.

    Budget = ``max(base_max_tokens, _PER_MEMBER_OP_MAX_TOKENS * max_window_len)``
    so a fat window (~24 members) gets room for a full idx-keyed op-list and a
    size-1 window keeps the byte-stable 512 default (96*1 < 512). One value
    applies to the single ``generate_batch`` call, so it is sized for the
    biggest window."""
    if not windows:
        return base_max_tokens
    biggest = max((len(w) for w in windows), default=1)
    return max(int(base_max_tokens), _PER_MEMBER_OP_MAX_TOKENS * biggest)


def _first_balanced_array(text: str) -> str | None:
    """Scan for the FIRST balanced ``[ … ]`` array and return it (string-aware).

    Mirrors :func:`_first_balanced_object` for the windowed op-LIST completion.
    Ignores brackets inside JSON string literals + escapes. Returns None when
    no balanced array is found."""
    if not text:
        return None
    start = text.find("[")
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
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_op_list(raw: str) -> list[dict[str, Any]] | None:
    """Tolerant extract of an idx-keyed op LIST from a window completion.

    Accepts (in order): a bare JSON array ``[{idx,...}, ...]``; a
    ``{"ops": [...]}`` wrapper object; or a SINGLE verdict object (the size-1
    window degenerate case, keyed by ``block_id`` or ``idx``) treated as a
    1-op list. Returns the list of op dicts, or None on any parse miss (the
    caller soft-falls-back to keeping the window's members verbatim). NEVER
    raises — mirrors :func:`_extract_verdict_obj`."""
    if not raw:
        return None
    try:
        stripped = _strip_code_fences(raw)
        arr = _first_balanced_array(stripped)
        # Prefer an array ONLY when it appears before any object (so a
        # ``{"ops":[...]}`` wrapper is read as the wrapper, not its inner array).
        obj_pos = stripped.find("{")
        arr_pos = stripped.find("[")
        if arr is not None and (obj_pos == -1 or 0 <= arr_pos < obj_pos):
            parsed = json.loads(arr)
            if isinstance(parsed, list):
                return [o for o in parsed if isinstance(o, dict)]
        obj = _extract_verdict_obj(raw)
        if obj is None:
            return None
        ops = obj.get("ops")
        if isinstance(ops, list):
            return [o for o in ops if isinstance(o, dict)]
        # A single verdict object (size-1 window legacy shape).
        return [obj]
    except (ValueError, TypeError):
        return None


def _op_idx(op: dict[str, Any]) -> int | None:
    """The op's target region index — ``idx`` (windowed) or ``block_id`` (legacy).

    Returns None when neither key coerces to an int."""
    for key in ("idx", "block_id"):
        if key in op:
            try:
                return int(op[key])
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run_structure_review(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    runtime: Any,
    *,
    max_tokens: int = 512,
    council_state: Any | None = None,
) -> tuple[list[Region], list[ReviewVerdict]]:
    """Run the Stage-5d structure review over the full region list.

    Flow (§3, §4, §6):

      0. Compute the four CLUSTER-LEVEL signals over the FULL ordered region
         list (``compute_cluster_signals`` — model-free) so the reviewer can
         distinguish a phantom-TOC / chapter-index entry (a RUN of same-level
         headings with no content between them) from a real section heading
         (FOLLOWED BY content). These signals are threaded into the prompt.
      1. Build a reviewer prompt for every HEADING region only (the
         heading-SCOPING that cuts cost — see below). The cluster pre-pass
         scanned the full list so the runs are correct, but only headings get
         a 70B call; every non-heading region passes through UNTOUCHED with
         ``verdict='ok'`` (indices preserved).
      2. ``runtime.generate_batch(heading_prompts, max_tokens=...)`` -> M
         strings (M = number of heading regions, M <= N).
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

    Heading-scoping (cost)
    ----------------------
    Only regions whose ``kind`` is a heading kind (``HEADING_KINDS``) get a
    70B call — the reviewer's three corrections (phantom-heading demotion,
    re-level, paragraph->heading promotion) all hinge on a heading's identity,
    and the phantom-TOC defect lives entirely in the heading stream. A
    non-heading region cannot become a phantom heading, so reviewing it is
    pure cost. Non-heading regions pass through verbatim with ``verdict='ok'``
    and their indices are preserved (the returned ``corrected`` / ``verdicts``
    lists stay 1:1 with the input). The cluster pre-pass STILL scans the full
    ordered list so same-level runs are computed correctly.

    NO model/GPU is loaded here; ``runtime`` is injected.
    """
    if not regions:
        return regions, []

    # Phase 4: SEMANTIK_BLOCK_REVIEW ON -> the windowed structural-editor path
    # (one idx-keyed op-list prompt per window of in-scope blocks,
    # O(blocks/window) calls). OFF -> the byte-stable heading-only,
    # one-block-per-prompt path below is UNCHANGED. The split keeps flag-off
    # byte-identical (this body is untouched) while collapsing the per-block
    # dispatch behind the flag.
    if resolve_block_review_mode():
        return _run_windowed_block_review(
            regions,
            feature_blocks,
            runtime,
            max_tokens=max_tokens,
            council_state=council_state,
        )

    n = len(regions)

    # (0) cluster-level signals over the FULL ordered list (model-free).
    cluster_signals = compute_cluster_signals(regions)

    # (1) build prompts — in-scope regions only. HEADINGS are always in scope
    # (byte-stable); CONTENT blocks join only when SEMANTIK_BLOCK_REVIEW is on
    # AND the deterministic-first confidence gate fires (§6.2). Track the
    # mapping from the dense prompt list back to the sparse region indices.
    prompts: list[str] = []
    dispatched_indices: list[int] = []
    for index, region in enumerate(regions):
        if not _in_review_scope(region):
            continue
        neighbors = _neighbors_for(regions, index)
        if str(region.kind) in HEADING_KINDS:
            text = _resolve_region_text(region, feature_blocks)
        else:
            # Content block (flag on). Deterministic-first: high-confidence
            # blocks get NO prompt; only uncertain ones are dispatched. The
            # first-pass prompt carries the head/tail edge record (Phase 1).
            if not _content_block_dispatch_gate(region, council_state, feature_blocks):
                continue
            text = _content_edge_text(region, index, feature_blocks, council_state)
        prompts.append(
            build_reviewer_request(
                region,
                neighbors,
                index,
                text=text,
                cluster_signals=cluster_signals[index],
            )
        )
        dispatched_indices.append(index)

    # No headings -> nothing to review; pass everything through as ok.
    if not prompts:
        return regions, [
            _ok_verdict(region, idx, "non-heading; not reviewed")
            for idx, region in enumerate(regions)
        ]

    # (2) batch — REUSE generate_batch with fail_soft=True (the Stage-6
    #     robustness contract). A per-cluster endpoint failure (after the
    #     bounded transient retries in OpenAICompatibleRuntime) comes back as
    #     the None SENTINEL in that slot instead of raising
    #     EndpointBatchItemError, so one slow/down cluster degrades only
    #     itself — never hanging the document. PER-ITEM parse misses still
    #     soft-fall-back below; a None sentinel degrades that cluster to
    #     UNREVIEWED (reverted_for_endpoint_failure) further down.
    #
    #     Anti-crawl: generate_batch already fans EVERY cluster concurrently
    #     in one call, so there is no cluster-after-cluster serial crawl here.
    #     The crawl the real run hit was per-item timeout×retries; fail_soft
    #     bounds each item, and the all-/high-failed short-circuit below stops
    #     us from doing any further work on a clearly-dead endpoint.
    #
    #     Defensive: a runtime that does NOT accept fail_soft (e.g. an older
    #     scripted mock) falls back to the legacy call — its own fail-loud
    #     contract then applies, unchanged from before this fix.
    # Greedy / deterministic decoding by default (temperature 0.0) so the
    # structure-correction pass does not introduce run-to-run noise in
    # chapter/section decisions; an operator can opt back into sampling via
    # SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE > 0.
    review_temperature = resolve_structure_review_temperature()
    try:
        completions = runtime.generate_batch(
            prompts,
            max_tokens=max_tokens,
            temperature=review_temperature,
            fail_soft=True,
        )
    except TypeError:
        # The runtime does NOT accept fail_soft (e.g. an older scripted mock).
        # Re-run WITHOUT the kwarg so a deterministic mock still produces its
        # crafted completions — BUT guard the call: a runtime that lacks
        # fail_soft AND raises a real endpoint failure must NOT propagate
        # (that re-creates the per-item 429 crawl the fix removes). On any
        # endpoint error here, degrade the WHOLE review to the byte-stable
        # UNREVIEWED floor instead of re-raising.
        try:
            completions = runtime.generate_batch(
                prompts, max_tokens=max_tokens, temperature=review_temperature
            )
        except EndpointRuntimeError as exc:
            logger.warning(
                "structure-review runtime lacks fail_soft and the fail-loud "
                "call failed (%s) -> degrading WHOLE review to unreviewed "
                "(no fail-loud crawl)",
                exc,
            )
            return _degrade_whole_review_unreviewed(
                regions,
                "endpoint failure without fail_soft; whole-review degraded to "
                "unreviewed",
            )

    # Defensive: a runtime must return one completion per prompt. Pad/clip
    # to M so the per-block loop stays addressable (a short batch
    # soft-falls-back the missing tail to ok).
    m = len(prompts)
    completions = list(completions)
    if len(completions) < m:
        completions = completions + [""] * (m - len(completions))

    # Anti-crawl whole-document short-circuit: if the endpoint is clearly
    # dead (>= _ENDPOINT_DEAD_FAILURE_RATIO of clusters came back as the None
    # sentinel), degrade the ENTIRE review to unreviewed rather than parsing /
    # re-attempting cluster after cluster. This is the byte-stable UNREVIEWED
    # floor — every region kept verbatim, the audit shows the whole review as
    # endpoint-degraded.
    none_count = sum(1 for c in completions[:m] if c is None)
    if m > 0 and none_count >= max(1, int(round(_ENDPOINT_DEAD_FAILURE_RATIO * m))):
        logger.warning(
            "structure-review endpoint appears dead (%d/%d clusters failed) "
            "-> degrading WHOLE review to unreviewed (anti-crawl short-circuit)",
            none_count,
            m,
        )
        return _degrade_whole_review_unreviewed(
            regions,
            "endpoint unavailable; whole-review degraded to unreviewed",
        )

    # Map heading region index -> its completion (None sentinel = endpoint
    # failure for that cluster; str = a completion to parse).
    completion_by_index: dict[int, str | None] = {}
    for prompt_pos, region_index in enumerate(dispatched_indices):
        raw = completions[prompt_pos] if prompt_pos < len(completions) else ""
        if raw is None:
            completion_by_index[region_index] = None
        else:
            completion_by_index[region_index] = raw if isinstance(raw, str) else ""

    # (3)+(4) parse + apply per block. Out-of-scope / un-gated regions pass
    # through. Content blocks whose re-type is flagged ``ambiguous`` are
    # DEFERRED for a single fuller-text re-ask (§3) collected below.
    block_review_on = resolve_block_review_mode()
    escalation_indices: list[int] = []
    corrected: list[Region] = []
    verdicts: list[ReviewVerdict] = []
    endpoint_failure_count = 0
    for index, region in enumerate(regions):
        if index not in completion_by_index:
            # Out-of-scope / un-gated region — not reviewed, kept verbatim.
            corrected.append(region)
            verdicts.append(_ok_verdict(region, index, "non-heading; not reviewed"))
            continue
        raw = completion_by_index[index]
        if raw is None:
            # Per-cluster endpoint failure (None sentinel from fail_soft) ->
            # degrade THIS cluster to unreviewed; keep the region verbatim.
            corrected.append(region)
            verdicts.append(
                _endpoint_failure_verdict(
                    region, index, "endpoint call failed; cluster kept unreviewed"
                )
            )
            endpoint_failure_count += 1
            continue
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
        # §3 ambiguity escalation: a CONTENT-block re-type the model flagged
        # ``ambiguous`` is DEFERRED — keep the original verbatim for now and
        # collect it for ONE fuller-text re-ask (single-shot, no recursion).
        # Headings never escalate (byte-stable). The first (ambiguous) result
        # is NOT applied; the escalated fuller-text result replaces it below.
        if (
            block_review_on
            and str(region.kind) not in HEADING_KINDS
            and _is_ambiguous_verdict(obj)
        ):
            escalation_indices.append(index)
            corrected.append(region)
            verdicts.append(
                _ok_verdict(region, index, "ambiguous re-type; escalating to fuller text")
            )
            continue
        region_out, verdict = _apply_verdict(region, index, obj, feature_blocks)
        corrected.append(region_out)
        verdicts.append(verdict)

    # (4b) §3 single-shot fuller-text escalation. For every deferred ambiguous
    # content block, re-ask ONCE with the FULL capped source (one extra batched
    # call total — NOT per-block, NO recursion). The escalated result is applied
    # through the SAME _apply_verdict / _admits invariant; a parse miss /
    # endpoint failure / out-of-idx re-ask keeps the placeholder original (never
    # regresses). The re-ask is itself NOT re-escalated.
    if escalation_indices:
        esc_prompts = [
            build_reviewer_request(
                regions[idx],
                _neighbors_for(regions, idx),
                idx,
                text=_capped_source_text(regions[idx], feature_blocks),
                cluster_signals=cluster_signals[idx],
            )
            for idx in escalation_indices
        ]
        try:
            esc_completions = runtime.generate_batch(
                esc_prompts,
                max_tokens=max_tokens,
                temperature=review_temperature,
                fail_soft=True,
            )
        except TypeError:
            try:
                esc_completions = runtime.generate_batch(
                    esc_prompts, max_tokens=max_tokens, temperature=review_temperature
                )
            except EndpointRuntimeError:
                esc_completions = [None] * len(esc_prompts)
        esc_completions = list(esc_completions)
        for pos, idx in enumerate(escalation_indices):
            raw = esc_completions[pos] if pos < len(esc_completions) else None
            if not isinstance(raw, str):
                continue  # endpoint failure -> keep placeholder original.
            obj2 = _extract_verdict_obj(raw)
            if obj2 is None:
                continue  # parse miss -> keep placeholder original.
            claimed2 = obj2.get("block_id")
            try:
                claimed2_int = int(claimed2)
            except (TypeError, ValueError):
                claimed2_int = None
            if claimed2_int != idx:
                continue  # mismatched id -> keep placeholder original.
            region_out, verdict = _apply_verdict(regions[idx], idx, obj2, feature_blocks)
            corrected[idx] = region_out
            verdicts[idx] = verdict

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

    # One-line summary: how many heading clusters reviewed vs degraded to
    # unreviewed because their endpoint call failed (visible in the audit via
    # each verdict's reverted_for_endpoint_failure flag).
    reviewed_count = m - endpoint_failure_count
    logger.info(
        "structure-review complete: %d/%d heading clusters reviewed, "
        "%d reverted-for-endpoint-failure",
        reviewed_count,
        m,
        endpoint_failure_count,
    )

    return corrected, verdicts


def _run_windowed_block_review(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    runtime: Any,
    *,
    max_tokens: int,
    council_state: Any | None,
) -> tuple[list[Region], list[ReviewVerdict]]:
    """Phase-4 windowed full-block reviewer (SEMANTIK_BLOCK_REVIEW on path).

    Collapses the per-block calls into ONE op-list prompt per window of
    ``resolve_block_review_window()`` in-scope blocks (O(blocks/window) calls).
    Reuses the SAME scoping (``_in_review_scope`` / ``_content_block_dispatch_gate``),
    apply (``_apply_verdict`` / ``_admits``), fail-soft (``_endpoint_failure_verdict``),
    anti-crawl (``_degrade_whole_review_unreviewed``), escalation
    (``_capped_source_text``), and token-conservation machinery as the
    heading-only path — only the dispatch SHAPE (windowed op-list) is new.

    LOAD-BEARING 1:1 output contract: returns EXACTLY one corrected Region +
    one ReviewVerdict per input region, index-stable (the cascade Stage-5d
    ``zip(pre_review_regions, reviewed_regions)`` metadata-drop exemption and
    the C2 ``_assert_fb_survival_through_cap`` baseline both zip positionally).
    Asserted as a post-condition before return.
    """
    n = len(regions)

    # (0) cluster-level signals over the FULL ordered list (model-free) — the
    # SAME pre-pass the heading path runs (threaded into multi-member prompts).
    cluster_signals = compute_cluster_signals(regions)

    # (1) Scope: dispatched indices + per-member prompt material. Headings are
    # always in scope; content blocks join only through the deterministic-first
    # confidence gate. text_by_idx feeds the size-1 (byte-stable single-block)
    # prompt; record_by_idx feeds the >=2 windowed envelope.
    dispatched_indices: list[int] = []
    text_by_idx: dict[int, str] = {}
    record_by_idx: dict[int, dict[str, Any]] = {}
    for index, region in enumerate(regions):
        if not _in_review_scope(region):
            continue
        if str(region.kind) in HEADING_KINDS:
            text_by_idx[index] = _resolve_region_text(region, feature_blocks)
        else:
            if not _content_block_dispatch_gate(region, council_state, feature_blocks):
                continue
            text_by_idx[index] = _content_edge_text(
                region, index, feature_blocks, council_state
            )
        record_by_idx[index] = build_edge_input(
            region,
            block_id=index,
            feature_blocks=feature_blocks,
            council_state=council_state,
        )
        dispatched_indices.append(index)

    # Nothing in scope -> pass everything through as ok (1:1 preserved).
    if not dispatched_indices:
        return regions, [
            _ok_verdict(region, idx, "non-heading; not reviewed")
            for idx, region in enumerate(regions)
        ]

    # (2) Window the in-scope indices (overlap so a boundary-straddling group is
    # whole in one window) and build ONE prompt per window. A single-member
    # window degenerates to the byte-stable single-block reviewer prompt; a
    # >=2-member window uses the idx-keyed op-list envelope.
    windows = _build_windows(
        dispatched_indices, resolve_block_review_window(), _BLOCK_REVIEW_WINDOW_OVERLAP
    )
    window_prompts: list[str] = []
    for win in windows:
        if len(win) == 1:
            idx = win[0]
            window_prompts.append(
                build_reviewer_request(
                    regions[idx],
                    _neighbors_for(regions, idx),
                    idx,
                    text=text_by_idx[idx],
                    cluster_signals=cluster_signals[idx],
                )
            )
        else:
            window_prompts.append(
                build_windowed_reviewer_request(
                    [record_by_idx[i] for i in win],
                    cluster_signals_by_idx={i: cluster_signals[i] for i in win},
                )
            )

    win_max_tokens = _window_max_tokens(max_tokens, windows)
    review_temperature = resolve_structure_review_temperature()

    # (3) Dispatch — ONE generate_batch over the window prompts (fail_soft so a
    # down window comes back as the None sentinel, not a raise). Same TypeError
    # fallback + fail-loud-degrade contract as the heading path.
    try:
        completions = runtime.generate_batch(
            window_prompts,
            max_tokens=win_max_tokens,
            temperature=review_temperature,
            fail_soft=True,
        )
    except TypeError:
        try:
            completions = runtime.generate_batch(
                window_prompts, max_tokens=win_max_tokens, temperature=review_temperature
            )
        except EndpointRuntimeError as exc:
            logger.warning(
                "block-review runtime lacks fail_soft and the fail-loud call "
                "failed (%s) -> degrading WHOLE review to unreviewed",
                exc,
            )
            return _degrade_whole_review_unreviewed(
                regions,
                "endpoint failure without fail_soft; whole-review degraded to "
                "unreviewed",
            )

    w = len(window_prompts)
    completions = list(completions)
    if len(completions) < w:
        completions = completions + [""] * (w - len(completions))

    # Anti-crawl: a clearly-dead endpoint (>= _ENDPOINT_DEAD_FAILURE_RATIO of
    # WINDOWS came back as the None sentinel) degrades the WHOLE review to the
    # byte-stable unreviewed floor rather than parsing window after window.
    none_windows = sum(1 for c in completions[:w] if c is None)
    if w > 0 and none_windows >= max(1, int(round(_ENDPOINT_DEAD_FAILURE_RATIO * w))):
        logger.warning(
            "block-review endpoint appears dead (%d/%d windows failed) -> "
            "degrading WHOLE review to unreviewed (anti-crawl short-circuit)",
            none_windows,
            w,
        )
        return _degrade_whole_review_unreviewed(
            regions,
            "endpoint unavailable; whole-review degraded to unreviewed",
        )

    # (4) Apply. Build the 1:1 index-stable output: every region keeps its slot.
    # Non-dispatched regions pass through ok; dispatched regions take their
    # window's op (first window to yield an op for an idx wins — overlap dedup).
    corrected: list[Region] = list(regions)
    verdicts: list[ReviewVerdict | None] = [None] * n
    for idx, region in enumerate(regions):
        if idx not in record_by_idx:
            verdicts[idx] = _ok_verdict(region, idx, "non-heading; not reviewed")

    applied: set[int] = set()
    escalation_indices: list[int] = []
    endpoint_failure_count = 0
    for win, raw in zip(windows, completions[:w]):
        member_set = set(win)
        if raw is None:
            # Window endpoint failure -> its not-yet-applied members degrade to
            # UNREVIEWED (kept verbatim, reverted_for_endpoint_failure).
            for idx in win:
                if idx in applied:
                    continue
                applied.add(idx)
                verdicts[idx] = _endpoint_failure_verdict(
                    regions[idx], idx, "endpoint call failed; window kept unreviewed"
                )
                endpoint_failure_count += 1
            continue
        ops = _extract_op_list(raw if isinstance(raw, str) else "")
        op_by_idx: dict[int, dict[str, Any]] = {}
        if ops:
            for op in ops:
                oid = _op_idx(op)
                # idx-bounds check: an op for an idx OUTSIDE this window's
                # member set is DROPPED (anti-cross-contamination) — mirrors
                # block_resegment._parse_llm_ops' bounds drop.
                if oid is None or oid not in member_set:
                    continue
                op_by_idx.setdefault(oid, op)  # first op for an idx wins
        for idx in win:
            if idx in applied:
                continue
            applied.add(idx)
            op = op_by_idx.get(idx)
            if op is None:
                # No op for this in-window member -> soft fallback ok.
                verdicts[idx] = _ok_verdict(
                    regions[idx], idx, "no op returned for in-window block; kept original"
                )
                continue
            # §3 ambiguity escalation: an ambiguous CONTENT re-type is deferred
            # for ONE fuller-text re-ask (headings never escalate — byte-stable).
            if (
                str(regions[idx].kind) not in HEADING_KINDS
                and _is_ambiguous_verdict(op)
            ):
                escalation_indices.append(idx)
                verdicts[idx] = _ok_verdict(
                    regions[idx], idx, "ambiguous re-type; escalating to fuller text"
                )
                continue
            region_out, verdict = _apply_verdict(regions[idx], idx, op, feature_blocks)
            corrected[idx] = region_out
            verdicts[idx] = verdict

    # (4b) §3 single-shot fuller-text escalation — ONE extra batched call total
    # (NOT per-block, NO recursion). Each deferred ambiguous content block is
    # re-asked with its FULL capped source; the result is applied through the
    # SAME _apply_verdict / _admits invariant. A parse miss / endpoint failure /
    # out-of-idx re-ask keeps the placeholder original (never regresses).
    if escalation_indices:
        esc_prompts = [
            build_reviewer_request(
                regions[idx],
                _neighbors_for(regions, idx),
                idx,
                text=_capped_source_text(regions[idx], feature_blocks),
                cluster_signals=cluster_signals[idx],
            )
            for idx in escalation_indices
        ]
        try:
            esc_completions = runtime.generate_batch(
                esc_prompts,
                max_tokens=win_max_tokens,
                temperature=review_temperature,
                fail_soft=True,
            )
        except TypeError:
            try:
                esc_completions = runtime.generate_batch(
                    esc_prompts, max_tokens=win_max_tokens, temperature=review_temperature
                )
            except EndpointRuntimeError:
                esc_completions = [None] * len(esc_prompts)
        esc_completions = list(esc_completions)
        for pos, idx in enumerate(escalation_indices):
            raw = esc_completions[pos] if pos < len(esc_completions) else None
            if not isinstance(raw, str):
                continue  # endpoint failure -> keep placeholder original.
            ops = _extract_op_list(raw)
            if not ops:
                continue  # parse miss -> keep placeholder original.
            op = next((o for o in ops if _op_idx(o) == idx), None)
            if op is None:
                continue  # mismatched id -> keep placeholder original.
            region_out, verdict = _apply_verdict(regions[idx], idx, op, feature_blocks)
            corrected[idx] = region_out
            verdicts[idx] = verdict

    # (4c) Post-condition: 1:1 index-stable output (LOAD-BEARING — the cascade
    # zip + cap-safety baseline both depend on it). Every slot filled.
    assert len(corrected) == n, "windowed review changed the region count"
    final_verdicts = [
        v if v is not None else _ok_verdict(regions[i], i, "not reviewed")
        for i, v in enumerate(verdicts)
    ]

    # (5) document-level token conservation — fail-closed to the flag-OFF list.
    try:
        assert_token_conservation(regions, corrected, feature_blocks)
    except TokenConservationError as exc:
        logger.warning(
            "block-review reverting to flag-OFF region list (fail-closed): %s",
            exc,
        )
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

    reviewed_count = len(applied) - endpoint_failure_count
    logger.info(
        "block-review complete: %d windows, %d/%d in-scope blocks reviewed, "
        "%d reverted-for-endpoint-failure",
        w,
        reviewed_count,
        len(dispatched_indices),
        endpoint_failure_count,
    )
    return corrected, final_verdicts


__all__ = [
    "CONTENT_REVIEW_KINDS",
    "ClusterSignals",
    "HEADING_KINDS",
    "ReviewVerdict",
    "TokenConservationError",
    "assert_token_conservation",
    "compute_cluster_signals",
    "resolve_block_review_cache_mode",
    "resolve_block_review_conf_floor",
    "resolve_block_review_edge_tokens",
    "resolve_block_review_mode",
    "resolve_block_review_window",
    "resolve_structure_review_mode",
    "resolve_structure_review_temperature",
    "run_structure_review",
]
