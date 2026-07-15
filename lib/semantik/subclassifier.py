r"""Tier-3 model-assisted subclassifier for SemantiK composite units (Build #23).

The Tier-2 composite-unit planner (:mod:`lib.semantik.composite_units`) coagulates
a run of adjacent sibling blocks into one ``<section class="dart-unit" data-dart-unit
="<type>">`` pedagogical whole (worked_example / exercise_set / definition_group /
procedure / section_opener / figure_group). Tier-3 asks a light local model to pick
a finer SUBCLASS for each unit — e.g. a ``worked_example`` is an
``application-problem`` vs a ``symbolic-manipulation`` — so downstream retrieval /
authoring can distinguish a drill set from a mixed-review, or an algorithm-style
procedure from a strategy.

Safety contract (from the June reviewer-redesign + the gold-semantic-class lessons):

* **Payload-only, HTML-only.** This is a post-cascade ADAPTER pass over already-
  rendered chapter HTML — consistent with every other post-cascade seam. It NEVER
  alters text or structure: it adds exactly two things to a unit's ``<section>``
  opening tag — ``data-dart-subclass="<label>"`` and an extra CSS class
  ``dart-sub-<label>``. The sidecar / raw_text are untouched. A before/after HTML
  diff differs ONLY in those two attributes (asserted by the payload-only test).
* **Strict parse.** The model is asked for ONE subclass from the seed vocabulary OR
  a new kebab-case label. The response is parsed strictly — a single-token
  kebab-case string of ``<= MAX_LABEL_CHARS`` chars; anything else (multi-word,
  punctuation, prose) is rejected and the unit stays unsubclassed.
* **Fold-on-ingestion.** A model-proposed NEW label within a small edit distance
  (Levenshtein ``<= FOLD_MAX_EDIT``) or sharing a stem with an existing vocabulary
  entry folds INTO that entry (no near-duplicate proliferation). A genuinely-new
  label is ACCEPTED onto the HTML but LOGGED to a review sidecar — never silently
  canonized into the lexicon taxonomy file.
* **Page-anomaly guard.** A unit whose member blocks' ``data-dart-pages`` are
  non-monotonic (out of document page order) is SKIPPED — the Tier-2 association
  may be unreliable there, so we don't put a confident subclass on it.
* **DecisionCapture.** One ``unit_subclass_assignment`` decision per unit, rationale
  interpolating the unit id / type / label / confidence / page-range (dynamic per
  call, per the CLAUDE.md LLM-call-site contract).

Provider seat: the model is reached through a provider-agnostic ``client`` callable
(``prompt -> completion text``). :func:`build_default_subclass_client` resolves the
existing license-clean local reviewer seat (qwen2.5-7b via ``LOCAL_SYNTHESIS_*`` on
an Ollama-style OpenAI-compatible server) with env-resolved base_url / model — no
hardcoded endpoints. Tests inject a mock callable, so the whole module is exercisable
CPU-only with no GPU / network.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "SEMANTIK_SUBCLASS_ENV",
    "SEMANTIK_SUBCLASS_SAMPLES_ENV",
    "MAX_LABEL_CHARS",
    "FOLD_MAX_EDIT",
    "TEXT_EXCERPT_CHARS",
    "SubclassAssignment",
    "resolve_semantic_subclass_enabled",
    "resolve_subclass_samples",
    "normalize_label",
    "fold_label",
    "load_seed_vocab",
    "annotate_html_subclasses",
    "build_default_subclass_client",
]

#: Env flag gating the whole pass. Default OFF (unset) — legacy corpora / the
#: byte-identical render path are unaffected. Parse-with-fallback truthy set.
SEMANTIK_SUBCLASS_ENV = "SEMANTIK_SEMANTIC_SUBCLASS"

#: Env knob for self-consistency sampling. Int >= 1, default 1. N=1 keeps the
#: single-call behavior byte-identical; N>1 draws N sampled calls per unit and
#: majority-votes the (parsed + folded) labels for a more robust subclass +
#: agreement-fraction confidence. Parse-with-fallback (garbage / <1 -> 1).
SEMANTIK_SUBCLASS_SAMPLES_ENV = "SEMANTIK_SUBCLASS_SAMPLES"

#: Sampling temperature used ONLY on the N>1 self-consistency path (passed to a
#: client that accepts it). The N=1 path never sets a temperature, so a
#: temperature-0.0 default seat stays deterministic + byte-identical.
SUBCLASS_SAMPLE_TEMPERATURE = 0.7

#: A parsed subclass label is at most this many chars (strict parse ceiling).
MAX_LABEL_CHARS = 32
#: A model-proposed new label within this Levenshtein distance of an existing
#: vocab entry folds INTO it (near-duplicate merge).
FOLD_MAX_EDIT = 2
#: First N chars of the unit's tag-stripped text handed to the model.
TEXT_EXCERPT_CHARS = 400

#: A well-formed subclass label: one kebab-case token, lowercase alnum groups.
_LABEL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_TRUTHY = {"1", "true", "yes", "on"}

# Tag scanner — matches BOTH ``<section ...>`` and ``</section>`` so spans can be
# recovered with a depth stack (a unit section can nest child dart-sections).
_SECTION_TAG_RE = re.compile(r"<(/?)section\b([^>]*)>", re.IGNORECASE)
# DART->semantik purge Stage 3 (dual-READ): the rendered HTML this locator
# scans may be freshly-emitted ``data-semantik-*`` OR a legacy ``data-dart-*``
# corpus, so both attr spellings are harvested identically.
_DATA_DART_UNIT_RE = re.compile(r'data-(?:dart|semantik)-unit="([^"]*)"')
_DATA_DART_SUBCLASS_RE = re.compile(r'data-(?:dart|semantik)-subclass="')
_CLASS_ATTR_RE = re.compile(r'class="([^"]*)"')
_HEADING_RE = re.compile(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PAGES_RE = re.compile(r'data-(?:dart|semantik)-pages="([^"]*)"')

_DATA_DART_PAGE_KIND = "physical"


# ---------------------------------------------------------------------------
# Flag + label helpers.
# ---------------------------------------------------------------------------
def resolve_semantic_subclass_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Whether the Tier-3 subclass pass is enabled (``SEMANTIK_SEMANTIC_SUBCLASS``)."""
    src = env if env is not None else os.environ
    return (src.get(SEMANTIK_SUBCLASS_ENV) or "").strip().lower() in _TRUTHY


def resolve_subclass_samples(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve the self-consistency sample count (``SEMANTIK_SUBCLASS_SAMPLES``).

    Parse-with-fallback: a positive int wins; an empty / non-integer / <1 value
    (or the unset default) resolves to ``1`` — the byte-identical single-call
    path. Never raises.
    """
    src = env if env is not None else os.environ
    raw = (src.get(SEMANTIK_SUBCLASS_SAMPLES_ENV) or "").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 1
    return n if n >= 1 else 1


def normalize_label(raw: Optional[str]) -> Optional[str]:
    """Strictly parse a model response into a single kebab-case subclass label.

    Accepts either a bare token (``"application-problem"``) or a JSON object
    ``{"subclass": "...", ...}`` — the caller extracts the token via
    :func:`_extract_label_and_confidence`, this fn is the final gate. Lowercases,
    trims surrounding quotes / whitespace, and returns the label ONLY when it is a
    single kebab-case token of ``<= MAX_LABEL_CHARS`` chars. Anything else — an
    empty string, a multi-word phrase, punctuation, an over-long token — returns
    ``None`` (the unit stays unsubclassed).
    """
    if not raw:
        return None
    lab = str(raw).strip().strip("\"'").strip().lower()
    # Collapse internal whitespace to a hyphen ONLY if it's otherwise clean —
    # but a phrase with multiple spaces is a prose answer, reject it. We do NOT
    # silently kebab-ify multi-word prose; a single interior space is treated as
    # a hyphen-typo, more than one space is a reject.
    if lab.count(" ") == 1 and "-" not in lab:
        lab = lab.replace(" ", "-")
    if not lab or len(lab) > MAX_LABEL_CHARS:
        return None
    if not _LABEL_RE.match(lab):
        return None
    return lab


def _extract_label_and_confidence(
    response: str,
) -> Tuple[Optional[str], Optional[float]]:
    """Pull ``(raw_label, confidence)`` out of a model completion.

    Tries JSON first (``{"subclass": "...", "confidence": 0.8}``); falls back to
    the bare stripped completion as the label token. Confidence is only surfaced
    when the model returned a parseable JSON number in ``[0, 1]``.
    """
    text = (response or "").strip()
    if not text:
        return None, None
    # JSON object arm.
    if text[:1] == "{":
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            label = obj.get("subclass") or obj.get("label") or obj.get("subclassification")
            conf = obj.get("confidence")
            conf_f: Optional[float] = None
            try:
                if conf is not None:
                    conf_f = float(conf)
                    if not (0.0 <= conf_f <= 1.0):
                        conf_f = None
            except (ValueError, TypeError):
                conf_f = None
            return (str(label) if label is not None else None), conf_f
    # Bare-token arm: first line only (models sometimes append prose).
    first_line = text.splitlines()[0].strip()
    return first_line, None


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein edit distance (pure, small strings)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _stem(label: str) -> str:
    """Very light stem for same-stem folding: drop a trailing plural / gerund.

    Domain-agnostic and conservative — only strips well-known English suffixes so
    ``diagrams`` folds to ``diagram`` and ``manipulating`` to ``manipulat``; it is
    used only as a folding SIGNAL alongside edit distance, never as output.
    """
    lab = label.replace("-", "")
    for suf in ("ies", "es", "ing", "ed", "s"):
        if lab.endswith(suf) and len(lab) - len(suf) >= 3:
            return lab[: -len(suf)]
    return lab


def fold_label(
    candidate: str, vocab: Tuple[str, ...]
) -> Tuple[str, bool, Optional[str]]:
    """Fold a proposed label into the seed vocabulary when near-duplicate.

    Returns ``(resolved_label, is_new, folded_into)``:

    * exact vocab hit -> ``(candidate, False, None)``.
    * near-duplicate (Levenshtein ``<= FOLD_MAX_EDIT`` OR same stem) of a vocab
      entry -> ``(vocab_entry, False, vocab_entry)`` (folded).
    * otherwise a genuinely-new label -> ``(candidate, True, None)``.

    Determinism: vocab is scanned in order, so the FIRST near-duplicate wins.
    """
    if candidate in vocab:
        return candidate, False, None
    cand_stem = _stem(candidate)
    for entry in vocab:
        if _levenshtein(candidate, entry) <= FOLD_MAX_EDIT or _stem(entry) == cand_stem:
            return entry, False, entry
    return candidate, True, None


def load_seed_vocab(
    profile_spec: Optional[str] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Load the per-unit-type subclass seed vocabulary from the lexicon taxonomy."""
    from lib.ontology.taxonomy import get_lexicon_subclasses

    return get_lexicon_subclasses(profile_spec)


def load_seed_glosses(
    profile_spec: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Load the per-label subclass GLOSSES (one-line definitions) from the lexicon.

    Empty strings for legacy list-form profiles; the prompt falls back to the
    bare label list when no gloss is available for a unit type's seeds.
    """
    from lib.ontology.taxonomy import get_lexicon_subclass_glosses

    return get_lexicon_subclass_glosses(profile_spec)


# ---------------------------------------------------------------------------
# HTML span scanning (payload-only — reads inner content, rewrites open tags).
# ---------------------------------------------------------------------------
@dataclass
class _UnitSpan:
    """One ``<section data-dart-unit>`` unit located in the rendered HTML."""

    unit_type: str
    open_start: int
    open_end: int
    close_start: int
    open_tag: str
    attrs: str
    inner: str


def _iter_unit_spans(html: str) -> List[_UnitSpan]:
    """Locate every ``data-dart-unit`` composite-unit section with its inner slice.

    Uses a depth stack over ``<section>`` / ``</section>`` so a unit wrapper that
    nests child ``dart-section`` blocks is recovered correctly. Only sections
    carrying a ``data-dart-unit`` attribute are returned. Malformed / unbalanced
    markup degrades gracefully (unmatched opens are ignored).
    """
    stack: List[Tuple[int, int, str, str]] = []  # (open_start, open_end, attrs, tag)
    spans: List[_UnitSpan] = []
    for m in _SECTION_TAG_RE.finditer(html):
        is_close = m.group(1) == "/"
        if not is_close:
            stack.append((m.start(), m.end(), m.group(2), m.group(0)))
        else:
            if not stack:
                continue
            open_start, open_end, attrs, tag = stack.pop()
            um = _DATA_DART_UNIT_RE.search(attrs)
            if not um or not um.group(1).strip():
                continue
            spans.append(
                _UnitSpan(
                    unit_type=um.group(1).strip(),
                    open_start=open_start,
                    open_end=open_end,
                    close_start=m.start(),
                    open_tag=tag,
                    attrs=attrs,
                    inner=html[open_end : m.start()],
                )
            )
    # Document order (outer-first): sort by open position.
    spans.sort(key=lambda s: s.open_start)
    return spans


def _lead_heading(inner: str) -> str:
    """First heading text inside the unit (tag-stripped, whitespace-collapsed)."""
    m = _HEADING_RE.search(inner)
    if not m:
        return ""
    return _WS_RE.sub(" ", _TAG_STRIP_RE.sub(" ", m.group(1))).strip()


def _text_excerpt(inner: str, limit: int = TEXT_EXCERPT_CHARS) -> str:
    """First ``limit`` chars of the unit's tag-stripped text content."""
    text = _WS_RE.sub(" ", _TAG_STRIP_RE.sub(" ", inner)).strip()
    return text[:limit]


def _parse_pages(value: str) -> List[int]:
    """Parse a ``data-dart-pages`` value (``"3"`` / ``"3-5"`` / ``"3,5,7"``)."""
    out: List[int] = []
    for part in (value or "").split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                out.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        elif part.isdigit():
            out.append(int(part))
    return out


def _member_page_firsts(inner: str) -> List[int]:
    """First page of each member block's ``data-dart-pages``, in document order."""
    firsts: List[int] = []
    for m in _PAGES_RE.finditer(inner):
        pages = _parse_pages(m.group(1))
        if pages:
            firsts.append(min(pages))
    return firsts


def _pages_monotonic(firsts: List[int]) -> bool:
    """True when the member first-pages are non-decreasing (in document order)."""
    return all(firsts[i] <= firsts[i + 1] for i in range(len(firsts) - 1))


def _unit_page_range(inner: str) -> Optional[Tuple[int, int]]:
    """``(lo, hi)`` spanning every member page, or ``None`` when unknown."""
    all_pages: List[int] = []
    for m in _PAGES_RE.finditer(inner):
        all_pages.extend(_parse_pages(m.group(1)))
    if not all_pages:
        return None
    return min(all_pages), max(all_pages)


def _annotate_open_tag(open_tag: str, attrs: str, label: str) -> str:
    """Return the rewritten unit ``<section>`` open tag carrying the subclass.

    Adds ``data-dart-subclass="<label>"`` and appends ``dart-sub-<label>`` to the
    existing ``class`` value. Payload-only: nothing else in the tag changes.
    """
    new_attrs = _CLASS_ATTR_RE.sub(
        lambda m: f'class="{m.group(1)} dart-sub-{label}"', attrs, count=1
    )
    if new_attrs == attrs:  # no class attr (defensive) — add one
        new_attrs = f'{attrs} class="dart-sub-{label}"'
    new_attrs = f'{new_attrs} data-semantik-subclass="{label}"'
    return f"<section{new_attrs}>"


# ---------------------------------------------------------------------------
# Assignment record + the main pass.
# ---------------------------------------------------------------------------
@dataclass
class SubclassAssignment:
    """One unit's Tier-3 outcome (assigned / folded / new / skipped / rejected)."""

    unit_index: int
    unit_id: str
    unit_type: str
    status: str  # assigned | folded | new | skipped_anomaly | parse_reject | no_seed_vocab | error
    label: Optional[str] = None
    folded_into: Optional[str] = None
    confidence: Optional[float] = None
    page_range: Optional[Tuple[int, int]] = None
    raw_response: str = ""
    lead_heading: str = ""


def _build_prompt(
    unit_type: str,
    lead_heading: str,
    excerpt: str,
    seeds: Tuple[str, ...],
    glosses: Optional[Dict[str, str]] = None,
) -> str:
    """Compact classification prompt: pick one seed OR propose a kebab label.

    When the lexicon carries per-label ``glosses`` (the ``{label: gloss}``
    subclass form), each seed is rendered with its one-line definition and the
    model is told to match on the DEFINITION — a bare label list mode-locks a
    small model onto the most generic-sounding label (the ch06/ch09 pilot
    collapsed 146/148 worked examples to ``symbolic-manipulation``, including
    obvious real-world application problems).
    """
    if seeds and glosses and any(glosses.get(s) for s in seeds):
        seed_lines = "\n".join(
            f"- {s}: {glosses.get(s) or '(no definition)'}" for s in seeds
        )
        seed_block = (
            f"Choose EXACTLY ONE subclass from this list:\n{seed_lines}\n"
            "Match on the DEFINITION after the colon, not the label wording; "
            "when two definitions both fit, prefer the more specific, "
            "context-aware label over the generic one. "
        )
    else:
        seed_list = ", ".join(seeds) if seeds else "(none)"
        seed_block = f"Choose EXACTLY ONE subclass from this list: [{seed_list}]. "
    return (
        "You are labelling the pedagogical SUBCLASS of one textbook unit.\n"
        f"Unit type: {unit_type}\n"
        f"Lead heading: {lead_heading or '(none)'}\n"
        f"Text (first {TEXT_EXCERPT_CHARS} chars): {excerpt}\n\n"
        + seed_block
        + "If none fits, propose a new single-token kebab-case label "
        "(lowercase, hyphenated, no spaces). "
        'Respond with ONLY a JSON object: {"subclass": "<label>", "confidence": <0-1>}.'
    )


@dataclass
class _VoteOutcome:
    """The result of a self-consistency vote over N sampled subclass calls."""

    status: str  # assigned | folded | new | parse_reject | error
    resolved: Optional[str] = None
    is_new: bool = False
    folded_into: Optional[str] = None
    confidence: Optional[float] = None
    raw_response: str = ""
    dist: Dict[str, int] = field(default_factory=dict)
    valid: int = 0
    samples: int = 0


def _supported_sample_kwargs(
    client: Callable[..., str],
    *,
    max_tokens: int,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Best-effort kwargs a ``client`` accepts (``max_tokens`` / ``temperature`` / ``seed``).

    Introspects the callable's signature so a mock that only takes ``max_tokens``
    is called exactly as today, while the real seat (which additionally accepts
    ``temperature`` / ``seed``) gets varied sampling params on the N>1 path. A
    client exposing ``**kwargs`` accepts all of them.
    """
    try:
        import inspect

        params = inspect.signature(client).parameters
    except (TypeError, ValueError):
        return {}
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    out: Dict[str, Any] = {}
    if "max_tokens" in params or has_var_kw:
        out["max_tokens"] = max_tokens
    if temperature is not None and ("temperature" in params or has_var_kw):
        out["temperature"] = temperature
    if seed is not None and ("seed" in params or has_var_kw):
        out["seed"] = seed
    return out


def _vote_subclass(
    client: Callable[..., str],
    prompt: str,
    seeds: Tuple[str, ...],
    samples_n: int,
    max_tokens: int,
) -> _VoteOutcome:
    """Draw ``samples_n`` sampled subclass calls and majority-vote the labels.

    Each sample runs the SAME strict parse (:func:`normalize_label`) + fold
    normalization (:func:`fold_label`) as the single-call path; a failed parse
    (or a raising call) drops out of the vote. The dominant folded label wins
    with confidence = ``dominant_count / valid_samples`` (agreement fraction,
    replacing the model-reported confidence). Ties resolve to the first winner
    seen in sample order. Zero valid samples -> ``parse_reject`` (some response
    came back but none parsed) or ``error`` (every call raised).
    """
    from collections import Counter

    labels: List[str] = []  # folded/resolved labels, in sample order
    fold_info: Dict[str, Tuple[bool, Optional[str]]] = {}
    any_response = False
    first_raw = ""
    for s in range(samples_n):
        kwargs = _supported_sample_kwargs(
            client, max_tokens=max_tokens,
            temperature=SUBCLASS_SAMPLE_TEMPERATURE, seed=s,
        )
        try:
            raw = client(prompt, **kwargs) if kwargs else client(prompt)
        except Exception as exc:  # noqa: BLE001 — a bad sample drops out of the vote
            logger.debug("subclassifier: sample %d client error: %s", s, exc)
            continue
        any_response = True
        if not first_raw:
            first_raw = str(raw)[:120]
        raw_label, _conf = _extract_label_and_confidence(raw)
        norm = normalize_label(raw_label)
        if norm is None:
            continue
        resolved, is_new, folded_into = fold_label(norm, seeds)
        labels.append(resolved)
        fold_info[resolved] = (is_new, folded_into)

    valid = len(labels)
    if valid == 0:
        return _VoteOutcome(
            status="parse_reject" if any_response else "error",
            raw_response=first_raw, samples=samples_n,
        )
    counts = Counter(labels)
    top_n = max(counts.values())
    # First label (in sample order) reaching the top count — deterministic tie-break.
    winner = next(lab for lab in labels if counts[lab] == top_n)
    is_new, folded_into = fold_info[winner]
    status = "new" if is_new else ("folded" if folded_into else "assigned")
    return _VoteOutcome(
        status=status, resolved=winner, is_new=is_new, folded_into=folded_into,
        confidence=top_n / valid, raw_response=first_raw,
        dist=dict(counts), valid=valid, samples=samples_n,
    )


def _vote_rationale_note(outcome: _VoteOutcome) -> str:
    """Dynamic sample-distribution note interpolated into the DecisionCapture rationale."""
    dist = ",".join(f"{lab}={n}" for lab, n in sorted(outcome.dist.items()))
    return f"vote [{dist}] valid={outcome.valid}/{outcome.samples}"


def _make_capture(course_code: str, capture: Optional[Any]) -> Any:
    """Return the injected capture, else construct the canonical semantik_conversion one."""
    if capture is not None:
        return capture
    from lib.decision_capture import DecisionCapture

    return DecisionCapture(course_code, phase="semantik_conversion", tool="semantik")


def annotate_html_subclasses(
    html: str,
    *,
    client: Callable[[str], str],
    course_code: str = "SEMANTIK_000",
    capture: Optional[Any] = None,
    profile_spec: Optional[str] = None,
    review_sidecar_path: Optional[str] = None,
    max_tokens: int = 64,
) -> Tuple[str, Dict[str, Any]]:
    """Annotate every composite unit in ``html`` with a Tier-3 subclass.

    Payload-only: the returned HTML differs from the input ONLY by the two
    ``data-dart-subclass`` / ``dart-sub-<label>`` attributes on subclassed unit
    ``<section>`` open tags. ``client`` is a ``prompt -> completion`` callable
    (mock in tests, the local reviewer seat in production). Returns
    ``(annotated_html, report)`` where ``report`` carries per-unit assignments,
    the per-type label distribution, fold events, the new-label log, and skip /
    reject counts. New labels are additionally appended to
    ``review_sidecar_path`` (JSONL) when supplied — never written back into the
    lexicon taxonomy.
    """
    seeds_by_type = load_seed_vocab(profile_spec)
    glosses_by_type = load_seed_glosses(profile_spec)
    spans = _iter_unit_spans(html)
    dc = _make_capture(course_code, capture)
    samples_n = resolve_subclass_samples()

    assignments: List[SubclassAssignment] = []
    edits: List[Tuple[int, int, str]] = []  # (open_start, open_end, new_open_tag)
    new_labels: List[Dict[str, Any]] = []

    for idx, span in enumerate(spans):
        unit_id = f"unit-{idx:04d}"
        # Idempotency: never re-annotate an already-subclassed unit.
        if _DATA_DART_SUBCLASS_RE.search(span.attrs):
            continue

        page_range = _unit_page_range(span.inner)

        # Page-anomaly guard: skip units whose member pages are non-monotonic.
        firsts = _member_page_firsts(span.inner)
        if not _pages_monotonic(firsts):
            assignments.append(
                SubclassAssignment(
                    unit_index=idx,
                    unit_id=unit_id,
                    unit_type=span.unit_type,
                    status="skipped_anomaly",
                    page_range=page_range,
                )
            )
            _log_decision(
                dc, unit_id, span.unit_type, None, None, page_range,
                status="skipped_anomaly",
            )
            continue

        seeds = seeds_by_type.get(span.unit_type, ())
        lead = _lead_heading(span.inner)
        excerpt = _text_excerpt(span.inner)
        prompt = _build_prompt(
            span.unit_type, lead, excerpt, seeds,
            glosses=glosses_by_type.get(span.unit_type),
        )

        # Common per-unit outcome locals populated by the single-call OR the
        # self-consistency (N>1) path below, then consumed by the shared
        # assignment / edit / decision-capture section.
        resolved: Optional[str] = None
        is_new = False
        folded_into: Optional[str] = None
        confidence: Optional[float] = None
        status = ""
        raw_response = ""
        vote_note: Optional[str] = None

        if samples_n <= 1:
            # Single-call path — byte-identical to the pre-sampling behavior.
            try:
                raw = client(prompt, max_tokens=max_tokens) if _accepts_max_tokens(client) else client(prompt)
            except Exception as exc:  # noqa: BLE001 — a model error never breaks render
                logger.warning("subclassifier: client error on %s: %s", unit_id, exc)
                assignments.append(
                    SubclassAssignment(
                        unit_index=idx, unit_id=unit_id, unit_type=span.unit_type,
                        status="error", page_range=page_range, lead_heading=lead,
                    )
                )
                _log_decision(
                    dc, unit_id, span.unit_type, None, None, page_range, status="error",
                )
                continue

            raw_label, confidence = _extract_label_and_confidence(raw)
            label = normalize_label(raw_label)
            if label is None:
                assignments.append(
                    SubclassAssignment(
                        unit_index=idx, unit_id=unit_id, unit_type=span.unit_type,
                        status="parse_reject", raw_response=str(raw)[:120],
                        page_range=page_range, lead_heading=lead,
                    )
                )
                _log_decision(
                    dc, unit_id, span.unit_type, None, confidence, page_range,
                    status="parse_reject",
                )
                continue

            resolved, is_new, folded_into = fold_label(label, seeds)
            status = "new" if is_new else ("folded" if folded_into else "assigned")
            raw_response = str(raw)[:120]
        else:
            # Self-consistency path — N sampled calls, majority vote, confidence
            # = agreement fraction. Failed samples drop out of the vote.
            outcome = _vote_subclass(client, prompt, seeds, samples_n, max_tokens)
            if outcome.status in ("error", "parse_reject"):
                if outcome.status == "error":
                    logger.warning(
                        "subclassifier: all %d samples failed on %s", samples_n, unit_id
                    )
                assignments.append(
                    SubclassAssignment(
                        unit_index=idx, unit_id=unit_id, unit_type=span.unit_type,
                        status=outcome.status, raw_response=outcome.raw_response,
                        page_range=page_range, lead_heading=lead,
                    )
                )
                _log_decision(
                    dc, unit_id, span.unit_type, None, None, page_range,
                    status=outcome.status,
                )
                continue
            resolved = outcome.resolved
            is_new = outcome.is_new
            folded_into = outcome.folded_into
            confidence = outcome.confidence
            status = outcome.status
            raw_response = outcome.raw_response
            vote_note = _vote_rationale_note(outcome)

        assignments.append(
            SubclassAssignment(
                unit_index=idx, unit_id=unit_id, unit_type=span.unit_type,
                status=status, label=resolved, folded_into=folded_into,
                confidence=confidence, page_range=page_range,
                raw_response=raw_response, lead_heading=lead,
            )
        )
        if is_new:
            new_labels.append(
                {"unit_id": unit_id, "unit_type": span.unit_type, "label": resolved}
            )
        edits.append(
            (span.open_start, span.open_end, _annotate_open_tag(span.open_tag, span.attrs, resolved))
        )
        _log_decision(
            dc, unit_id, span.unit_type, resolved, confidence, page_range,
            status=status, folded_into=folded_into, vote_note=vote_note,
        )

    annotated = _apply_edits(html, edits)
    if review_sidecar_path and new_labels:
        _append_review_sidecar(review_sidecar_path, new_labels)

    report = _build_report(assignments, seeds_by_type)
    return annotated, report


def _accepts_max_tokens(client: Callable[..., str]) -> bool:
    """True when ``client`` accepts a ``max_tokens`` kwarg (best-effort introspect)."""
    try:
        import inspect

        sig = inspect.signature(client)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "max_tokens" in params:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in params.values())


def _apply_edits(html: str, edits: List[Tuple[int, int, str]]) -> str:
    """Apply open-tag rewrites left-to-right; payload-only splice."""
    if not edits:
        return html
    edits.sort(key=lambda e: e[0])
    out: List[str] = []
    cursor = 0
    for start, end, new_tag in edits:
        out.append(html[cursor:start])
        out.append(new_tag)
        cursor = end
    out.append(html[cursor:])
    return "".join(out)


def _log_decision(
    dc: Any,
    unit_id: str,
    unit_type: str,
    label: Optional[str],
    confidence: Optional[float],
    page_range: Optional[Tuple[int, int]],
    *,
    status: str,
    folded_into: Optional[str] = None,
    vote_note: Optional[str] = None,
) -> None:
    """Emit one ``unit_subclass_assignment`` decision with a dynamic rationale."""
    if dc is None:
        return
    conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
    pr = f"{page_range[0]}-{page_range[1]}" if page_range else "unknown"
    fold_str = f" folded_into={folded_into}" if folded_into else ""
    vote_str = f"; {vote_note}" if vote_note else ""
    rationale = (
        f"Tier-3 subclass {status} for {unit_id} (type={unit_type}, "
        f"label={label or 'none'}, confidence={conf_str}, pages={pr}){fold_str}{vote_str}"
    )
    try:
        dc.log_decision(
            decision_type="unit_subclass_assignment",
            decision=f"{unit_type}:{label or status}",
            rationale=rationale,
            confidence=confidence,
        )
    except Exception as exc:  # noqa: BLE001 — capture must never break render
        logger.debug("subclassifier: decision capture failed: %s", exc)


def _append_review_sidecar(path: str, new_labels: List[Dict[str, Any]]) -> None:
    """Append genuinely-new labels to the operator review sidecar (JSONL)."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            for row in new_labels:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning("subclassifier: review sidecar write failed (%s): %s", path, exc)


def _build_report(
    assignments: List[SubclassAssignment],
    seeds_by_type: Dict[str, Tuple[str, ...]],
) -> Dict[str, Any]:
    """Build the pilot/apply report: assignments + per-type distribution + audits."""
    from collections import Counter

    dist: Dict[str, Counter] = {}
    folds: List[Dict[str, Any]] = []
    new_labels: List[Dict[str, Any]] = []
    counts = Counter(a.status for a in assignments)
    for a in assignments:
        if a.status in ("assigned", "folded", "new") and a.label:
            dist.setdefault(a.unit_type, Counter())[a.label] += 1
        if a.status == "folded":
            folds.append(
                {"unit_id": a.unit_id, "unit_type": a.unit_type, "folded_into": a.folded_into}
            )
        if a.status == "new":
            new_labels.append(
                {"unit_id": a.unit_id, "unit_type": a.unit_type, "label": a.label}
            )

    distribution = {ut: dict(c) for ut, c in dist.items()}
    return {
        "total_units": len(assignments),
        "status_counts": dict(counts),
        "label_distribution": distribution,
        "fold_events": folds,
        "new_labels": new_labels,
        "bucket_collapse": detect_bucket_collapse(distribution),
        "assignments": [
            {
                "unit_id": a.unit_id,
                "unit_type": a.unit_type,
                "status": a.status,
                "label": a.label,
                "folded_into": a.folded_into,
                "confidence": a.confidence,
                "page_range": list(a.page_range) if a.page_range else None,
            }
            for a in assignments
        ],
    }


#: A single label owning > this share within a type (n >= min) is bucket-collapse.
BUCKET_COLLAPSE_SHARE = 0.80
BUCKET_COLLAPSE_MIN_N = 10


def detect_bucket_collapse(
    distribution: Dict[str, Dict[str, int]],
) -> List[Dict[str, Any]]:
    """Flag per-type distributions where ONE label owns > 80% (n >= 10).

    The local-classifier-audit lesson: operational success (the pass ran) is not
    semantic accuracy — a bucket collapse means the model is rubber-stamping one
    label. Returns one row per collapsed type (empty when healthy).
    """
    collapsed: List[Dict[str, Any]] = []
    for unit_type, labels in distribution.items():
        total = sum(labels.values())
        if total < BUCKET_COLLAPSE_MIN_N:
            continue
        top_label, top_n = max(labels.items(), key=lambda kv: kv[1])
        share = top_n / total if total else 0.0
        if share > BUCKET_COLLAPSE_SHARE:
            collapsed.append(
                {
                    "unit_type": unit_type,
                    "dominant_label": top_label,
                    "share": round(share, 3),
                    "n": total,
                }
            )
    return collapsed


# ---------------------------------------------------------------------------
# Default provider seat (license-clean local reviewer — env-resolved, no hardcode).
# ---------------------------------------------------------------------------
def build_default_subclass_client() -> Callable[..., str]:
    """Build the default ``prompt -> completion`` callable over the local seat.

    Reuses the existing license-clean local reviewer seat: an OpenAI-compatible
    server (Ollama-style) at ``LOCAL_SYNTHESIS_BASE_URL`` running the
    ``LOCAL_SYNTHESIS_MODEL`` (default qwen2.5-7b, Apache-2.0). No endpoint is
    hardcoded — base_url / model / api_key resolve from env via the shared
    ``OpenAICompatibleClient`` (same wire client every provider composes). The
    returned closure issues one low-temperature JSON-mode chat completion per
    unit and returns the assistant text.
    """
    from Courseforge.generators._base import (
        LOCAL_DEFAULT_BASE_URL,
        LOCAL_DEFAULT_MODEL,
    )
    from Trainforge.generators._openai_compatible_client import (
        OpenAICompatibleClient,
    )

    base_url = os.environ.get("LOCAL_SYNTHESIS_BASE_URL") or LOCAL_DEFAULT_BASE_URL
    model = os.environ.get("LOCAL_SYNTHESIS_MODEL") or LOCAL_DEFAULT_MODEL
    api_key = os.environ.get("LOCAL_SYNTHESIS_API_KEY") or "local"
    oa = OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        provider_label="semantik_subclass",
        json_mode=True,
    )

    def _client(
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> str:
        # temperature defaults 0.0 so the single-call (N=1) path stays
        # deterministic + byte-identical; the self-consistency (N>1) path passes
        # a non-zero temperature + per-sample seed so the 7B seat varies.
        extra = {"seed": seed} if seed is not None else None
        return oa.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_payload=extra,
        )

    return _client
