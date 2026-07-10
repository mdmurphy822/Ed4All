"""Deterministic rewrite-tier content lint — the STRUCTURAL-shape control for
leaked authoring artifacts that the semantic gates (NLI entailment, numeric
grounding, symbolic math) are blind to.

Motivation (real defects, hand-found in 7B-authored vendor-HTML builds)
-----------------------------------------------------------------------
A rewrite-tier block occasionally ships published HTML that is grounded,
number-correct, AND math-correct, yet still leaks authoring-machinery text a
human reviewer would immediately reject:

* **Pseudo-markup leaks** — a CURIE / concept-slug that should have been
  rendered as prose instead surfaces as literal pseudo-tag text: an escaped
  ``&lt;solution&gt;`` / ``&lt;decimal&gt;`` entity, a namespaced
  ``<course:concept>`` CURIE token, a custom-element-shaped
  ``<associative-property>`` slug, or a ``\\text{associative_property}`` LaTeX
  leftover carrying an underscore/colon/angle-bracket slug.
* **Publisher apparatus leaks** — a source textbook's numbered apparatus copied
  verbatim into course prose: ``Try It 3.2``, ``Example 3.2`` (a *dotted*
  section-number cross-ref, distinct from the legitimate ``Example 3.`` label),
  a parenthetical ``(see Figure 3`` / ``(see Table 4``, or a
  ``see Section 3.2`` cross-reference that points at a section number the
  generated course does not have.
* **Slug-glue** — a term slug welded straight into prose: a *doubled term*
  (``the discriminant, discriminant``) where a slug was appended to its own
  gloss, or a *definition sentence starting with a bare lowercase slug*
  (``decimal provides a way to …``) where a concept slug was used as the
  sentence subject with no leading article.

None of these is catchable by a semantic gate: the numbers are grounded, the
prose entails the source, and any embedded math is correct. Only a
structural-shape scan of the rendered bytes separates leaked machinery from
clean prose.

Wide-net discipline (root ``CLAUDE.md`` — SemantiK "wide net" feedback)
-----------------------------------------------------------------------
Every pattern matches a domain-agnostic, course-agnostic STRUCTURAL shape — an
escaped/namespaced/custom-element angle-bracket pseudo-tag, a ``\\text{}`` slug
leftover, a generic *numbered*-apparatus shape, a repeated-word comma glue, or a
lowercase-sentence-start definitional verb. No pattern hardcodes a specific
course's vocabulary or slug. The single-word escaped-tag arm is guarded by a
universal HTML-tag denylist (``&lt;p&gt;`` in a code sample is NOT a leak) so a
programming course that legitimately shows escaped HTML is not flagged.

Precision over recall (mirrors ``worked_example_math``)
-------------------------------------------------------
Ambiguous shapes are SKIPPED rather than risk a false positive: a hyphenated
word inside ``\\text{}`` (``\\text{well-known}``) is not a slug signal (only an
underscore / colon / angle-bracket inside is); a single-hyphen concept phrase in
prose is not flagged (only the doubled-term and pseudo-tag shapes are).

Severity + shadow (mirrors ``worked_example_math`` / the numeric gates)
-----------------------------------------------------------------------
Warning day-1 in ``config/workflows.yaml`` (calibration posture). ``shadow`` is
accepted for parity with the sibling gates + the deferred critical flip; until
the flip both are warning. Issues carry the block id in ``location`` so
``courseforge-rewrite --block-ids`` can consume the failure list and re-roll
exactly the leaking blocks.

Decision-capture
----------------
One ``rewrite_content_lint`` event per SCORED block (a block with linkable str
content). Rationale interpolates dynamic per-call signals (>=20 chars): block_id,
block_type, per-category hit counts, and the first flagged snippet. Clean blocks
still emit a passed event (matches the per-block audit discipline); a block with
no str content emits nothing.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Block lives at Courseforge/scripts/blocks.py — same import bridge as the
# sibling block validators (worked_example_math / numeric_literal_grounding).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover — Block always present in this repo
    from blocks import Block  # type: ignore[import-not-found]  # noqa: E402
except Exception:  # noqa: BLE001
    Block = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Canonical GateIssue codes + decision type
# --------------------------------------------------------------------------- #

_CODE_PSEUDO_MARKUP: str = "REWRITE_PSEUDO_MARKUP_LEAK"
_CODE_APPARATUS: str = "REWRITE_APPARATUS_LEAK"
_CODE_SLUG_GLUE: str = "REWRITE_SLUG_GLUE"
_DECISION_TYPE: str = "rewrite_content_lint"

#: Cap per-block finding list + overall issue list (mirrors sibling validators).
_PER_BLOCK_CAP: int = 20
_ISSUE_LIST_CAP: int = 100

#: Match length of the snippet carried in a GateIssue message.
_SNIPPET_LEN: int = 80


# --------------------------------------------------------------------------- #
# Pattern (1) — pseudo-markup leaks (escaped/namespaced/custom-element tags,
# \text{} slug leftovers)
# --------------------------------------------------------------------------- #

_RE_TAG = re.compile(r"<[^>]+>")

#: Universal HTML element names — the escaped single-word arm excludes these so
#: a legitimately-escaped ``&lt;p&gt;`` / ``&lt;div&gt;`` in a code sample is NOT
#: flagged (structural, course-agnostic; HTML tag names are universal).
_HTML_TAGS: frozenset = frozenset(
    {
        "a", "abbr", "address", "area", "article", "aside", "audio", "b",
        "base", "bdi", "bdo", "blockquote", "body", "br", "button", "canvas",
        "caption", "cite", "code", "col", "colgroup", "data", "datalist", "dd",
        "del", "details", "dfn", "dialog", "div", "dl", "dt", "em", "embed",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
        "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "i",
        "iframe", "img", "input", "ins", "kbd", "label", "legend", "li", "link",
        "main", "map", "mark", "menu", "meta", "meter", "nav", "noscript",
        "object", "ol", "optgroup", "option", "output", "p", "param", "picture",
        "pre", "progress", "q", "rp", "rt", "ruby", "s", "samp", "script",
        "section", "select", "slot", "small", "source", "span", "strong",
        "style", "sub", "summary", "sup", "svg", "table", "tbody", "td",
        "template", "textarea", "tfoot", "th", "thead", "time", "title", "tr",
        "track", "u", "ul", "var", "video", "wbr",
    }
)

#: A namespaced pseudo-tag ``<prefix:local>`` — a colon in a tag name is never
#: valid emitted HTML5, so a course-prefixed CURIE leaking as a literal tag is
#: an unambiguous leak (raw or escaped form).
_RE_NS_TAG_RAW = re.compile(r"</?[a-z][a-z0-9]*:[a-z0-9][a-z0-9_-]*>")
_RE_NS_TAG_ESC = re.compile(r"&lt;/?[a-z][a-z0-9]*:[a-z0-9][a-z0-9_-]*&gt;")

#: A custom-element-shaped hyphenated pseudo-tag ``<associative-property>`` — the
#: pipeline never emits custom elements, so a hyphenated angle-tag is a slug
#: leaked as literal markup (raw or escaped form).
_RE_HYPHEN_TAG_RAW = re.compile(r"</?[a-z][a-z0-9]*-[a-z0-9][a-z0-9-]*>")
_RE_HYPHEN_TAG_ESC = re.compile(r"&lt;/?[a-z][a-z0-9]*-[a-z0-9][a-z0-9-]*&gt;")

#: A single-word ESCAPED pseudo-tag ``&lt;solution&gt;`` / ``&lt;decimal&gt;`` —
#: only flagged when the word is NOT a universal HTML element (denylist above).
_RE_WORD_TAG_ESC = re.compile(r"&lt;/?([a-z][a-z0-9]*)&gt;")

#: ``\text{...}`` LaTeX leftover carrying a slug signal — an underscore
#: (slug-glue; subscript syntax outside \text, never natural word text inside),
#: a colon (namespace), or an angle-bracketed pseudo-tag. A plain / hyphenated
#: word inside \text{} is NOT a signal (skipped — precision over recall).
_RE_TEXT_BRACE = re.compile(r"\\text\s*\{([^{}]*)\}")
_RE_TEXT_SLUG_SIGNAL = re.compile(r"[_:]|<[^>]+>|&lt;[^&]+&gt;")


def _find_pseudo_markup(raw: str) -> List[str]:
    """Return snippet matches for pseudo-markup leaks in the raw block HTML."""
    hits: List[str] = []
    for regex in (
        _RE_NS_TAG_RAW,
        _RE_NS_TAG_ESC,
        _RE_HYPHEN_TAG_RAW,
        _RE_HYPHEN_TAG_ESC,
    ):
        for m in regex.finditer(raw):
            hits.append(m.group(0))
    for m in _RE_WORD_TAG_ESC.finditer(raw):
        if m.group(1) not in _HTML_TAGS:
            hits.append(m.group(0))
    for m in _RE_TEXT_BRACE.finditer(raw):
        inner = m.group(1)
        if _RE_TEXT_SLUG_SIGNAL.search(inner):
            hits.append(m.group(0)[:_SNIPPET_LEN])
    return hits


# --------------------------------------------------------------------------- #
# Pattern (2) — publisher apparatus leaks (generic numbered-apparatus shapes)
# --------------------------------------------------------------------------- #

#: ``Try It 3.2`` — numbered practice-apparatus cross-ref (generic shape).
_RE_TRY_IT = re.compile(r"\bTry It\s+\d+\.\d+")
#: ``Example 3.2`` — a DOTTED section-number cross-ref (distinct from the
#: legitimate ``Example 3.`` worked-example label the contract emits).
_RE_DOTTED_EXAMPLE = re.compile(r"\bExample\s+\d+\.\d+")
#: ``(see Figure 3`` / ``(see Table 4`` / ``(see Example 2`` — a parenthetical
#: cross-reference to a source figure/table the generated course lacks.
_RE_SEE_PAREN = re.compile(r"\(\s*see\s+(?:Figure|Table|Example|Section)\s+\d")
#: ``see Section 3.2`` / ``in Figure 3.1`` — a numbered section/figure cross-ref.
_RE_SECTION_XREF = re.compile(
    r"\b(?:see|in|from)\s+(?:Section|Figure|Table)\s+\d+\.\d+"
)

_APPARATUS_PATTERNS: Tuple[re.Pattern, ...] = (
    _RE_TRY_IT,
    _RE_DOTTED_EXAMPLE,
    _RE_SEE_PAREN,
    _RE_SECTION_XREF,
)


def _find_apparatus(text: str) -> List[str]:
    """Return snippet matches for publisher-apparatus leaks in tag-stripped text."""
    hits: List[str] = []
    for regex in _APPARATUS_PATTERNS:
        for m in regex.finditer(text):
            hits.append(m.group(0))
    return hits


# --------------------------------------------------------------------------- #
# Pattern (3) — slug-glue (doubled term, bare-slug definition sentence)
# --------------------------------------------------------------------------- #

#: A DOUBLED term ``the discriminant, discriminant`` — a slug appended straight
#: after its own gloss. Backreference: a >=4-char lowercase word, comma, the
#: SAME word. >=4 chars avoids ``no, no`` / ``so, so`` noise.
_RE_DOUBLED_TERM = re.compile(r"\b([a-z]{4,})\b\s*,\s+\1\b", re.IGNORECASE)

#: A DEFINITION sentence starting with a BARE lowercase slug + a definitional
#: verb: ``decimal provides …`` / ``discriminant is defined …``. Rendered prose
#: sentences begin capitalized, so a lowercase sentence-start with a single
#: token subject and a definitional verb (and NO leading article) is the slug
#: leak. Anchored at a sentence boundary (start, ``.``/``!``/``?``, or ``<p>``).
_RE_BARE_SLUG_DEF = re.compile(
    r"(?:^|[.!?]\s+|<p[^>]*>\s*)"
    r"([a-z][a-z]{2,})\s+"
    r"(?:provides|is defined|refers to|denotes|represents|means)\b"
)


def _find_slug_glue(raw: str, text: str) -> List[str]:
    """Return snippet matches for slug-glue leaks (doubled term / bare-slug def)."""
    hits: List[str] = []
    for m in _RE_DOUBLED_TERM.finditer(text):
        hits.append(m.group(0)[:_SNIPPET_LEN])
    # The bare-slug-definition sentence anchor is checked against raw HTML so a
    # leading ``<p>`` boundary is visible; the captured subject is lowercase.
    for m in _RE_BARE_SLUG_DEF.finditer(raw):
        snippet = m.group(0).lstrip("<p >").strip()
        hits.append(snippet[:_SNIPPET_LEN])
    return hits


# --------------------------------------------------------------------------- #
# Block coercion (mirrors worked_example_math._coerce_blocks)
# --------------------------------------------------------------------------- #


def _coerce_blocks(
    inputs: Dict[str, Any]
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Resolve ``inputs['blocks']`` or hydrate from ``inputs['blocks_final_path']``.

    The gate-input router surfaces ``blocks`` (rewrite-tier hydrated Block list);
    tests / standalone callers may pass ``blocks_final_path`` (a JSONL of block
    dicts) instead.
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

    path_raw = inputs.get("blocks_final_path")
    if not path_raw:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs requires 'blocks' (List[Block]) or 'blocks_final_path' "
                "(a JSONL path of rewrite-tier block dicts)."
            ),
        )
    try:
        path = Path(path_raw)
    except (TypeError, ValueError):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=f"blocks_final_path is not a path: {path_raw!r}.",
        )
    if not path.exists():
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=f"blocks_final_path does not exist: {path}.",
        )
    blocks: List[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return blocks, None


def _block_field(block: Any, field: str) -> Any:
    """Read ``field`` from a Block instance OR a plain dict (JSONL row)."""
    if isinstance(block, dict):
        return block.get(field)
    return getattr(block, field, None)


def _emit_decision(
    capture: Any,
    *,
    block_id: str,
    block_type: str,
    pseudo: int,
    apparatus: int,
    slug_glue: int,
    passed: bool,
    flagged_snippet: Optional[str],
    shadow: bool,
) -> None:
    """Emit one ``rewrite_content_lint`` decision per SCORED block.

    Rationale interpolates dynamic per-call signals (>=20 chars) so the capture
    is replayable post-hoc (root ``CLAUDE.md`` § "LLM call-site instrumentation").
    Swallows ``log_decision`` exceptions (capture must not break the gate).
    """
    if capture is None:
        return
    decision = "passed" if passed else "failed:rewrite_content_lint"
    rationale = (
        f"Deterministic rewrite-tier content lint on Block {block_id!r}: "
        f"block_type={block_type!r}, pseudo_markup_hits={pseudo}, "
        f"apparatus_hits={apparatus}, slug_glue_hits={slug_glue}, "
        f"shadow={shadow}, first_flagged={flagged_snippet or 'none'}. "
        f"Regex-only structural-shape scan (no LLM) catches leaked authoring "
        f"machinery — escaped/namespaced/custom-element pseudo-tags, \\text{{}} "
        f"slug leftovers, generic numbered-publisher-apparatus cross-refs, and "
        f"doubled-term / bare-slug definition glue — that grounding / NLI / "
        f"numeric / symbolic-math gates are blind to."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on %s: %s", _DECISION_TYPE, exc
        )


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class RewriteContentLintValidator:
    """Deterministic rewrite-tier content lint (structural-shape control).

    For each rewrite-tier block carrying str content, scans the rendered HTML
    for three domain-agnostic leak shapes — pseudo-markup (escaped / namespaced
    / custom-element angle-tags + ``\\text{}`` slug leftovers), publisher
    apparatus (generic numbered cross-refs), and slug-glue (doubled term /
    bare-slug definition sentence). Fires warning-severity GateIssues carrying
    the block id in ``location`` so ``courseforge-rewrite --block-ids`` can
    re-roll exactly the leaking blocks. No LLM, no external dependency.

    Inputs:
        blocks: List[Block] | List[dict]
            Rewrite-tier blocks (hydrated Block instances from the router, or
            plain dict rows from a JSONL).
        blocks_final_path: str | Path
            Alternative to ``blocks`` — a rewrite-tier blocks JSONL.
        shadow: bool
            Forces emitted issues to warning-severity (parity with the sibling
            gates; the gate is warning day-1 regardless).
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per scored block.
    """

    name = "rewrite_content_lint"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        shadow = bool(inputs.get("shadow", False))

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Warning day-1 regardless of the shadow knob (parity with
        # worked_example_math + the numeric gates; the deferred critical flip
        # would use shadow to keep issues at warning while a critical YAML
        # severity gates). Until the flip, both are warning.
        fail_severity = "warning"
        _ = shadow  # reserved for the deferred critical-flip severity split

        issues: List[GateIssue] = []
        scored_blocks = 0
        passed_blocks = 0

        for block in blocks:
            content = _block_field(block, "content")
            # Rewrite-tier content is str HTML; dict content (outline tier) and
            # empty content are skipped silently (no event).
            if not isinstance(content, str) or not content.strip():
                continue

            block_type = _block_field(block, "block_type")
            block_id = _block_field(block, "block_id") or "<unknown>"
            scored_blocks += 1

            text = _RE_TAG.sub(" ", content)
            pseudo_hits = _find_pseudo_markup(content)
            apparatus_hits = _find_apparatus(text)
            slug_hits = _find_slug_glue(content, text)

            findings: List[Tuple[str, str]] = []
            for snippet in pseudo_hits:
                findings.append((_CODE_PSEUDO_MARKUP, snippet))
            for snippet in apparatus_hits:
                findings.append((_CODE_APPARATUS, snippet))
            for snippet in slug_hits:
                findings.append((_CODE_SLUG_GLUE, snippet))

            if not findings:
                passed_blocks += 1
                _emit_decision(
                    capture, block_id=block_id, block_type=block_type,
                    pseudo=0, apparatus=0, slug_glue=0, passed=True,
                    flagged_snippet=None, shadow=shadow,
                )
                continue

            for code, snippet in findings[:_PER_BLOCK_CAP]:
                if len(issues) >= _ISSUE_LIST_CAP:
                    break
                issues.append(
                    GateIssue(
                        severity=fail_severity,
                        code=code,
                        message=(
                            f"Rewrite-tier block {block_id!r} "
                            f"(block_type={block_type!r}) leaks authoring "
                            f"machinery — {code}: {snippet!r}. This is a "
                            f"STRUCTURAL-shape leak (pseudo-markup / publisher "
                            f"apparatus / slug-glue) that grounding / NLI / "
                            f"numeric / symbolic-math gates cannot catch."
                        ),
                        location=str(block_id),
                        suggestion=(
                            "Re-author the block as clean prose. Render a "
                            "concept slug / CURIE as the words it names (never "
                            "a literal pseudo-tag or \\text{} leftover), drop "
                            "the source's numbered cross-references (Try It / "
                            "Example N.M / see Figure N), and remove the "
                            "doubled-term / bare-slug definition glue. Re-roll "
                            "this block via `courseforge-rewrite --block-ids "
                            f"{block_id}`."
                        ),
                    )
                )
            _emit_decision(
                capture, block_id=block_id, block_type=block_type,
                pseudo=len(pseudo_hits), apparatus=len(apparatus_hits),
                slug_glue=len(slug_hits), passed=False,
                flagged_snippet=findings[0][1][:_SNIPPET_LEN], shadow=shadow,
            )

        # Warning-severity day-1: ``passed`` honors only critical issues (there
        # are none day-1), so the gate never blocks — it COMPUTES + CAPTURES.
        # The deferred critical flip promotes the three leak codes to critical.
        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        score = (
            1.0 if scored_blocks == 0 else round(passed_blocks / scored_blocks, 4)
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
            metadata={
                "scored_blocks": scored_blocks,
                "passed_blocks": passed_blocks,
                "flagged_blocks": scored_blocks - passed_blocks,
            },
        )


__all__ = [
    "RewriteContentLintValidator",
    "_CODE_PSEUDO_MARKUP",
    "_CODE_APPARATUS",
    "_CODE_SLUG_GLUE",
    "_DECISION_TYPE",
]
