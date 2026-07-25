"""Deterministic prose-stutter validator (``block_prose_stutter``).

Book-1 canary keystone fix: the rewrite tier's local model emits
phrase-repetition slop ("stutter") into otherwise-valid prose — e.g.
``"GET and HEAD are safe and HEAD are safe and idempotent"`` or a
four-stage list duplicated verbatim inside one sentence. On the canary
corpus roughly 150-190 of 613 packaged prose chunks carried the defect,
and NLI entailment does not catch it (a stuttered sentence still entails
its source). This module rejects stutter AT AUTHORING TIME: it is wired
into the rewrite router's per-candidate validator chain (a stuttered
candidate fires ``action="regenerate"`` and the best-of-N loop re-rolls)
and doubles as the ``block_prose_stutter`` observability gate at
``post_rewrite_validation`` (severity warning day-1 per the calibration
convention — the in-chain rejection is the enforcement surface).

Detector design (pure, deterministic, no models, no I/O):

Text is stripped of hidden ``data-cf-curie`` spans, ``pre``/``code``/
``script``/``style`` bodies, HTML tags, and broken-markup remnants —
each replaced by a NEWLINE so element boundaries become hard segment
boundaries (this is what keeps flip-card front/back term repetition and
heading+body term restates out of scope). Segments split on
``[\\n.!?;:]``; within one segment five rules fire:

- **adjacent_repeat** — an n-gram (n >= 3, normalized tokens) repeated
  immediately (gap 0). Guards: a short (n <= 4) repeat whose second copy
  starts a capitalised phrase is term-label/heading fusion, and a second
  copy opening with ``(`` is a deliberate parenthetical restate — both
  skipped.
- **window_repeat** — an n-gram (n >= 4, >= 60% content tokens) repeated
  with start-to-start distance <= 30 tokens inside the same segment.
  Guards: either occurrence starting a capitalised phrase (heading
  fusion), or the second occurrence directly preceded by a parallelism
  marker ("another", "respectively", ...) — skipped.
- **near_adjacent_repeat** — a pure-alphabetic all-content 2/3-gram
  repeated with 1..3 intervening tokens (``"caching, rate limiting,
  authentication, authorization, and rate limiting"``). Guards: any
  capitalised / non-alphabetic gram or gap token, or a determiner /
  contrast conjunction in the gap ("...with its local X" / "X, while
  write X...") — legitimate parallel constructions, skipped.
- **echo_word** — a content word (>= 4 chars) repeated at distance 2
  around a lowercase content pivot where the second copy runs into a
  function word (``"adds delay adds to"``). Heavily guarded against
  noun-compound chains and enumerations ("dirty writes, dirty reads").
- **label_dup** — an adjacent duplicated raw-token run ending with
  ``:`` (``"Key Idea: Idea:"``), digit-free, alphabetic-initial. Runs on
  the raw stream (labels legitimately never duplicate adjacently).

Calibration (book-1 canary imscc chunkset, 2026-07-22, reported in the
landing session — no corpus data is embedded here): flags 243/613 prose
chunks (the canary's independent heuristic flagged ~247, ~150-190
confirmed), catches all five owner-confirmed stutter chunks verbatim,
and flags only 7/765 (0.9%) of the QTI-path assessment_item chunks (the
clean-population control). Hand audit of 40 flagged chunks: ~87%
precision, with the residual dominated by chunk-boundary fusion
artifacts that cannot occur on block HTML (where tags are hard
boundaries).

Wiring:

- In-chain: ``MCP/tools/pipeline_tools.py::_run_content_generation_rewrite``
  passes ``[ProseStutterValidator()]`` into
  ``CourseforgeRouter.route_rewrite_with_remediation(validators=...)``.
- Gate: ``config/workflows.yaml`` ``block_prose_stutter`` at
  ``post_rewrite_validation`` (textbook_to_course + course_generation);
  input builder registered in ``MCP/hardening/gate_input_routing.py``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

#: Function words excluded from "content token" status. Deliberately small
#: and domain-agnostic (SemantiK wide-net contract: no corpus-specific vocab).
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have if in into is it its of on or "
    "that the their there these this to was were will with which when what how "
    "can may not no nor so such than then too very we you your they he she".split()
)
#: Function words that may legitimately FOLLOW the second copy of a verb echo
#: ("adds delay adds TO ..."); a content continuation marks a noun-compound
#: chain ("read scaling read replicas") instead.
_ECHO_FOLLOW = frozenset("to of in on for with by at as and or the a an".split())
#: Echo pivots that mark legitimate "X <prep> X" constructions ("time over
#: time", "flight via flight service").
_ECHO_PIVOT_BLOCK = frozenset(
    "over per after before versus upon within by about against via through "
    "from with".split()
)
#: Contrast conjunctions marking legitimate parallel constructions
#: ("...read locks, while write locks...").
_CONTRAST = frozenset("while whereas but versus vs than unlike".split())
#: Tokens that, immediately before a windowed repeat's second occurrence,
#: mark deliberate parallel enumeration ("...and another only ever queries...").
_PARALLEL_MARKERS = frozenset(
    "another other either respectively similarly likewise conversely".split()
)
#: Determiners/possessives in a near-adjacent gap mark a legitimate
#: two-referent construction ("merges the received clock with its local clock").
_DETERMINERS = frozenset(
    "the a an its their his her another each every this that those these your "
    "our my one".split()
)

_HIDDEN_CURIE_RE = re.compile(
    r"<span\b[^>]*\bdata-cf-curie\b[^>]*>.*?</span>", re.IGNORECASE | re.DOTALL
)
_CODE_RE = re.compile(
    r"<(pre|code|script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")
#: Broken-markup remnants (attribute fragments leaked into visible text,
#: e.g. ``detail="flip-card-front">``) are boundaries, never prose.
_MARKUP_REMNANT_RE = re.compile(r'\S*(?:">|=")\S*')
_SEGMENT_RE = re.compile(r"[\n.!?;:]+")
_TOKEN_RE = re.compile(r"\S+")
_NORM_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")
_PURE_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z-]*$")

#: Cap on issues attached to one GateResult (mirrors block_prose_entailment).
_MAX_ISSUES = 50


@dataclass
class StutterHit:
    """One detected repetition. ``repeated_span`` is the verbatim repeated
    phrase (first occurrence, raw text); ``segment_excerpt`` shows both
    occurrences in context."""

    rule: str
    repeated_span: str
    n_tokens: int
    segment_excerpt: str


def strip_html_for_stutter(html: str) -> str:
    """HTML -> plain text where every element boundary is a hard newline.

    Hidden ``data-cf-curie`` spans (postminted CURIE carriers) and
    ``pre``/``code``/``script``/``style`` bodies are removed entirely —
    code legitimately repeats and CURIE tokens legitimately duplicate.
    """
    if not html:
        return ""
    text = _HIDDEN_CURIE_RE.sub("\n", html)
    text = _CODE_RE.sub("\n", text)
    text = _TAG_RE.sub("\n", text)
    text = _MARKUP_REMNANT_RE.sub("\n", text)
    return text


def _norm(tok: str) -> str:
    return _NORM_STRIP_RE.sub("", tok).lower()


def _is_content(tok: str) -> bool:
    return (
        len(tok) >= 3
        and tok not in _STOPWORDS
        and not tok.isdigit()
        and any(c.isalpha() for c in tok)
    )


def find_stutters(
    text: str,
    *,
    adjacent_min_n: int = 3,
    window_min_n: int = 4,
    window_tokens: int = 30,
    near_gap_max: int = 3,
    max_hits: int = 20,
) -> List[StutterHit]:
    """Scan plain text (see :func:`strip_html_for_stutter`) for stutter.

    Pure and deterministic; returns at most ``max_hits`` hits. Thresholds
    are calibrated against the book-1 canary corpus (module docstring).
    """
    hits: List[StutterHit] = []

    # label_dup — duplicated ':'-terminated label run (raw stream, global;
    # crosses segment boundaries deliberately: the ':' is itself a boundary).
    raws_all = list(_TOKEN_RE.finditer(text))
    seen_d: set = set()
    for run_n in (3, 2, 1):
        for i in range(len(raws_all) - 2 * run_n + 1):
            r1 = [x.group() for x in raws_all[i : i + run_n]]
            r2 = [x.group() for x in raws_all[i + run_n : i + 2 * run_n]]
            if r1 != r2:
                continue
            if not r1[-1].endswith(":") or len(_norm(r1[-1])) < 3:
                continue
            if not r1[0][:1].isalpha():
                continue
            if any(_norm(t).isdigit() for t in r1):
                continue
            if any(k in seen_d for k in range(i, i + 2 * run_n)):
                continue
            seen_d.update(range(i, i + 2 * run_n))
            hits.append(StutterHit(
                rule="label_dup",
                repeated_span=" ".join(r1),
                n_tokens=run_n,
                segment_excerpt=text[
                    raws_all[i].start() : raws_all[i + 2 * run_n - 1].end()
                ][:200],
            ))

    for seg in _SEGMENT_RE.split(text):
        if len(hits) >= max_hits:
            break
        raws = list(_TOKEN_RE.finditer(seg))
        toks = [_norm(x.group()) for x in raws]
        n_tok = len(toks)
        if n_tok < 5:
            continue
        claimed: set = set()

        def _add(rule: str, i1: int, i2: int, n: int) -> None:
            # Extend the match maximally so the reported span is the whole
            # repeated phrase, and claim its tokens so overlapping smaller
            # matches don't double-report.
            while i2 + n < n_tok and toks[i1 + n] == toks[i2 + n] and toks[i1 + n]:
                n += 1
            if all(k in claimed for k in range(i2, i2 + n)):
                return
            claimed.update(range(i1, i2 + n))
            hits.append(StutterHit(
                rule=rule,
                repeated_span=seg[raws[i1].start() : raws[i1 + n - 1].end()],
                n_tokens=n,
                segment_excerpt=seg[raws[i1].start() : raws[i2 + n - 1].end()][:200],
            ))

        # adjacent_repeat — gap-0 n-gram duplication, n >= adjacent_min_n.
        for n in range(min(12, n_tok // 2), adjacent_min_n - 1, -1):
            for i in range(0, n_tok - 2 * n + 1):
                if any(k in claimed for k in range(i, i + 2 * n)):
                    continue
                g1 = toks[i : i + n]
                if (
                    g1 == toks[i + n : i + 2 * n]
                    and all(g1)
                    and any(_is_content(t) for t in g1)
                ):
                    # term-label / heading fusion: short repeat whose second
                    # copy starts a fresh capitalised phrase.
                    if n <= 4 and raws[i + n].group()[:1].isupper():
                        continue
                    # parenthetical restate: "X (X ..." is authorial.
                    if raws[i + n].group()[:1] == "(":
                        continue
                    _add("adjacent_repeat", i, i + n, n)

        # window_repeat — same-segment n-gram repeat within window_tokens.
        n = window_min_n
        last_seen: Dict[Tuple[str, ...], int] = {}
        for i in range(0, n_tok - n + 1):
            g = tuple(toks[i : i + n])
            if not all(g):
                continue
            if sum(1 for t in g if _is_content(t)) < max(3, int(0.6 * n)):
                continue
            prev = last_seen.get(g)
            if (
                prev is not None
                and n <= i - prev <= window_tokens
                and not raws[i].group()[:1].isupper()
                and not raws[prev].group()[:1].isupper()
                and not (i >= 1 and toks[i - 1] in _PARALLEL_MARKERS)
                # parenthetical restate: "X (X ..." is authorial, not stutter.
                and "(" not in raws[i].group()
                and not (i >= 1 and "(" in raws[i - 1].group())
                and not any(k in claimed for k in range(i, i + n))
            ):
                _add("window_repeat", prev, i, n)
            last_seen[g] = i

        # near_adjacent_repeat — content 2/3-gram with a 1..3 token gap.
        for n in (3, 2):
            for i in range(0, n_tok - n):
                if any(k in claimed for k in range(i, i + n)):
                    continue
                g1 = toks[i : i + n]
                if not all(_is_content(t) for t in g1):
                    continue
                if not all(
                    _PURE_WORD_RE.match(raws[i + k].group().strip(".,"))
                    for k in range(n)
                ):
                    continue
                if any(raws[i + k].group()[:1].isupper() for k in range(n)):
                    continue
                for gap in range(1, near_gap_max + 1):
                    j = i + n + gap
                    if j + n > n_tok:
                        break
                    if toks[j : j + n] != g1:
                        continue
                    if any(raws[j + k].group()[:1].isupper() for k in range(n)):
                        break
                    gap_ok = True
                    for x in range(i + n, j):
                        rg = raws[x].group()
                        if not rg[:1].isalpha() or rg[:1].isupper():
                            gap_ok = False
                            break
                        if _norm(rg) in _DETERMINERS or _norm(rg) in _CONTRAST:
                            gap_ok = False
                            break
                    if gap_ok and not any(
                        k2 in claimed for k2 in range(i, j + n)
                    ):
                        _add("near_adjacent_repeat", i, j, n)
                    break

        # echo_word — "X y X <function-word>" verb echo ("adds delay adds to").
        for i in range(0, n_tok - 2):
            if i in claimed or (i + 2) in claimed:
                continue
            x, y = toks[i], toks[i + 1]
            rx, ry, rx2 = raws[i].group(), raws[i + 1].group(), raws[i + 2].group()
            if (
                x
                and x == toks[i + 2]
                and x != y
                and len(x) >= 4
                and _is_content(x)
                and _is_content(y)
                and y not in _ECHO_PIVOT_BLOCK
                # list-rhythm guard: commas/parens mark enumeration
                # ("dirty writes, dirty reads"), never a stutter.
                and _PURE_WORD_RE.match(rx)
                and _PURE_WORD_RE.match(ry)
                and _PURE_WORD_RE.match(rx2.rstrip(".,;"))
                and ry[:1].islower()
                and not rx2[:1].isupper()
                and (i + 3 >= n_tok or toks[i + 3] in _ECHO_FOLLOW)
            ):
                claimed.update((i, i + 1, i + 2))
                hits.append(StutterHit(
                    rule="echo_word",
                    repeated_span=rx,
                    n_tokens=1,
                    segment_excerpt=seg[raws[i].start() : raws[i + 2].end()][:200],
                ))

    return hits[:max_hits]


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


def _block_field(block: Any, field: str) -> Any:
    """Read ``field`` from a Block instance OR a plain dict (JSONL row)."""
    if isinstance(block, dict):
        return block.get(field)
    return getattr(block, field, None)


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Resolve ``inputs['blocks']`` or hydrate from a blocks JSONL path.

    Mirrors the ``near_dup_example`` coercion contract: the gate-input
    router surfaces ``blocks``; tests / standalone callers may pass
    ``blocks_final_path`` / ``blocks_outline_path`` instead.
    """
    raw = inputs.get("blocks")
    if raw is not None:
        if not isinstance(raw, list):
            return [], GateIssue(
                severity="critical",
                code="INVALID_BLOCKS_INPUT",
                message=(
                    f"inputs['blocks'] must be a list; got {type(raw).__name__}."
                ),
            )
        return list(raw), None

    path_raw = (
        inputs.get("blocks_final_path")
        or inputs.get("blocks_outline_path")
        or inputs.get("blocks_path")
    )
    if not path_raw:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs requires 'blocks' (List[Block]) or a blocks JSONL "
                "path ('blocks_final_path' / 'blocks_outline_path')."
            ),
        )
    try:
        path = Path(path_raw)
    except (TypeError, ValueError):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=f"blocks path is not a path: {path_raw!r}.",
        )
    if not path.exists():
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=f"blocks path does not exist: {path}.",
        )
    blocks: List[Any] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                blocks.append(entry)
    except OSError as exc:
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=f"failed reading {path}: {exc}.",
        )
    return blocks, None


class ProseStutterValidator:
    """Deterministic phrase-repetition (stutter) gate over block prose.

    Scans every str-content block's HTML-stripped text with
    :func:`find_stutters`. Dict-content (outline-tier) blocks are skipped
    — stutter is a prose-emit defect. A failing result carries
    ``action="regenerate"`` so the rewrite router's per-candidate chain
    re-rolls the candidate; at the ``post_rewrite_validation`` seam the
    gate runs severity=warning (observability only).
    """

    name = "prose_stutter"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        start = time.monotonic()
        gate_id = str(inputs.get("gate_id") or "block_prose_stutter")

        def _result(
            passed: bool,
            issues: List[GateIssue],
            *,
            action: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> GateResult:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=passed,
                issues=issues,
                action=action,
                metadata=metadata,
                execution_time_ms=int((time.monotonic() - start) * 1000),
            )

        blocks, coerce_issue = _coerce_blocks(inputs)
        if coerce_issue is not None:
            return _result(False, [coerce_issue], action="block")
        if not blocks:
            return _result(True, [], metadata={"blocks_scanned": 0})

        thresholds = inputs.get("thresholds") or {}
        if not isinstance(thresholds, dict):
            thresholds = {}
        detector_kwargs: Dict[str, Any] = {}
        for key in (
            "adjacent_min_n", "window_min_n", "window_tokens", "near_gap_max",
        ):
            val = thresholds.get(key)
            if isinstance(val, int) and val > 0:
                detector_kwargs[key] = val

        issues: List[GateIssue] = []
        rule_counts: Dict[str, int] = {}
        scanned = 0
        stuttered_block_ids: List[str] = []
        for block in blocks:
            content = _block_field(block, "content")
            if not isinstance(content, str) or not content.strip():
                # Outline-tier dict content / empty content: nothing to scan.
                continue
            scanned += 1
            block_id = str(_block_field(block, "block_id") or "<unknown>")
            hits = find_stutters(
                strip_html_for_stutter(content), **detector_kwargs
            )
            if not hits:
                continue
            stuttered_block_ids.append(block_id)
            for hit in hits:
                rule_counts[hit.rule] = rule_counts.get(hit.rule, 0) + 1
            if len(issues) < _MAX_ISSUES:
                top = hits[0]
                issues.append(GateIssue(
                    severity="warning",
                    code="BLOCK_PROSE_STUTTER",
                    message=(
                        f"block {block_id}: {len(hits)} repetition(s); "
                        f"[{top.rule}] repeated span: {top.repeated_span!r} "
                        f"(context: {top.segment_excerpt!r})"
                    ),
                    location=block_id,
                    suggestion=(
                        "Re-roll the block: the emitted prose duplicates a "
                        "phrase (model stutter). The repeated span must "
                        "appear exactly once."
                    ),
                ))

        stuttered = len(stuttered_block_ids)
        metadata = {
            "blocks_scanned": scanned,
            "blocks_stuttered": stuttered,
            "stutter_rate": round(stuttered / scanned, 4) if scanned else 0.0,
            "rule_counts": rule_counts,
        }
        if stuttered:
            return _result(
                False, issues, action="regenerate", metadata=metadata,
            )
        return _result(True, [], metadata=metadata)
