"""Super-120B fused-page RE-SEGMENTATION stage (``SEMANTIK_SUPER_RESEGMENT``).

Productionized from the validated ``runtime/scratchpad/super_pages.py`` prototype
(owner-approved 2026-07-15). Given ONE structureless FUSED page — the run-on
paragraph blob SemantiK's per-page arranger emits when it cannot recover element
boundaries — this stage reasons it back into logically-correct, typed accessible
elements.

**Standalone by design (Step 1 of the refactor).** This module is importable and
unit-tested but wired into NOTHING yet — the cascade seam is a later step — so it
cannot regress any existing behaviour. It is a pure transformation over a single
page's text.

The finalized method is a CLEAN SEPARATION of concerns, each measured:

* **Super does PROSE ONLY.** A SIMPLE split+label prompt cuts a fused page into
  typed elements (``heading`` / ``para`` / ``item`` / ``callout`` / ``example``).
  Super is deliberately NOT asked to assign heading LEVELS inline (measured: that
  dropped conservation from 9/10 to 7/14) and NOT asked to rebuild TABLES
  (measured: it dropped 886-1225 chars). Heading levels are a deterministic
  post-hoc pass owned elsewhere.
* **Tables are DETERMINISTIC, never Super.** The upstream VLM already emits a
  table as a markdown pipe grid (``| a | b |`` with a ``| --- |`` separator row)
  or a LaTeX ``\\begin{tabular}`` block. A fused page that IS a table is converted
  deterministically (reusing ``lib.semantik.table_structure`` for the markdown
  path; :func:`latex_tabular_to_rows` for the LaTeX path) and Super is never
  called — the safety invariant is that a table region is left verbatim.

TRANSPORT: tagged lines, NEVER JSON. A backslash that is an illegal JSON escape
kills the parse; a legal one (``\\f`` ``\\t`` ``\\n``) SILENTLY decodes the LaTeX
away. Tags need no escaping, so the math survives verbatim. Every character is
conserved: the accept gate is the char-exact whitespace-normalized equality
``norm(block) == norm(" ".join(texts_of(elements)))``, matching the prototype.

The three-valued ``SEMANTIK_SUPER_RESEGMENT`` flag (``off`` / ``shadow`` / ``on``)
mirrors :func:`reasoning_qc.resolve_reasoning_qc_mode`. This module carries NO
DecisionCapture wiring — that lands at the call site in a later step.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seat + flag resolution (mirrors the resolve_*_seat / three-valued-mode
# conventions in reasoning_qc / reasoning_qc_vlm).
# ---------------------------------------------------------------------------
_SUPER_BASE_URL_ENV = "SEMANTIK_SUPER_BASE_URL"
_SUPER_MODEL_ENV = "SEMANTIK_SUPER_MODEL"
_SUPER_API_KEY_ENV = "SEMANTIK_SUPER_API_KEY"
_SUPER_RESEGMENT_ENV = "SEMANTIK_SUPER_RESEGMENT"

_DEFAULT_SUPER_BASE_URL = "http://localhost:8001/v1"
_DEFAULT_SUPER_MODEL = "nemotron-3-super"

# Sampling shape, verbatim from the validated prototype's ``one()`` / ``call()``:
# temperature 0.6 + top_p 0.95 (a reasoning model in thinking mode needs ~0.6 to
# TERMINATE deliberation); "reasoning effort: low" rides as a SYSTEM-PROMPT text
# directive (the ``reasoning_budget`` chat_template kwarg was measured INERT on
# this seat, so it is not sent).
_TEMPERATURE = 0.6
_TOP_P = 0.95
# A fused page must be re-emitted VERBATIM (split + label only), so leave ample
# completion room — mirrors the prototype's MAXTOK.
_MAX_TOKENS = 8000


class SuperSeat(NamedTuple):
    """Resolved Super-120B endpoint config (OpenAI-compatible, pure data)."""

    base_url: str
    model: str
    api_key: Optional[str]


def resolve_super_seat() -> SuperSeat:
    """Resolve the Super seat from the ``SEMANTIK_SUPER_*`` env family.

    Parse-with-fallback throughout: ``SEMANTIK_SUPER_BASE_URL`` (default
    ``http://localhost:8001/v1``), ``SEMANTIK_SUPER_MODEL`` (default
    ``nemotron-3-super``), ``SEMANTIK_SUPER_API_KEY`` (default ``None`` — the
    local / loopback seat needs no credential). Read at CALL time.
    """
    base_url = (os.environ.get(_SUPER_BASE_URL_ENV) or "").strip() or _DEFAULT_SUPER_BASE_URL
    model = (os.environ.get(_SUPER_MODEL_ENV) or "").strip() or _DEFAULT_SUPER_MODEL
    api_key = (os.environ.get(_SUPER_API_KEY_ENV) or "").strip() or None
    return SuperSeat(base_url=base_url, model=model, api_key=api_key)


def resolve_super_resegment_mode() -> str:
    """Return the Super-resegment mode: ``"off"`` / ``"shadow"`` / ``"on"``.

    Reads ``SEMANTIK_SUPER_RESEGMENT``. Three-valued parse-with-fallback whose
    DEFAULT is ``off`` (byte-identical / inert) — mirrors
    :func:`reasoning_qc.resolve_reasoning_qc_mode`:

      * ``on`` → ``on`` (compute + apply);
      * ``shadow`` → ``shadow`` (compute the result, caller applies nothing);
      * everything else (``0`` / ``false`` / ``no`` / ``off`` / unset / blank /
        garbage) → ``off``.

    Read at CALL time (never cached at import).
    """
    raw = (os.environ.get(_SUPER_RESEGMENT_ENV) or "").strip().lower()
    if raw == "on":
        return "on"
    if raw == "shadow":
        return "shadow"
    return "off"


# ---------------------------------------------------------------------------
# Prompt + tagged-line transport (proven bits copied from the prototype).
# ---------------------------------------------------------------------------
_SYS_PROSE = (
    "You are given the raw text of ONE page of a textbook whose true element "
    "boundaries were lost when it was fused into one run-on blob.\n"
    "Recover them. Emit ONE LINE PER ELEMENT, each starting with a tag:\n"
    "<<heading>>text\n"
    "<<para>>text\n"
    "<<item>>text      (one per list item; consecutive items form a list)\n"
    '<<callout>>text   (boxed asides: "BE PREPARED", "TRY IT", notes)\n'
    "<<example>>text   (worked examples)\n"
    "TEXT CONSERVATION: every character must appear verbatim exactly once, in "
    "order. Copy the text EXACTLY -- including all backslashes, LaTeX and OCR "
    "errors. Do NOT escape, correct, summarise, reformat, or assign heading "
    "levels, and do NOT rebuild tables. You may ONLY split and label.\n"
    "Output ONLY the tagged lines, nothing else.\n"
    "reasoning effort: low"
)

# The tag grammar the prototype's parse() consumes.
_TAG_RE = re.compile(r"^\s*<<(heading|para|item|callout|example|list)>>\s*(.*)$")


def _norm(s: str) -> str:
    """Whitespace-normalized, lowercased text — the char-exact conservation key.

    Byte-for-byte the prototype's ``norm`` (``re.sub(r"\\s+","",s).lower()``).
    """
    return re.sub(r"\s+", "", s or "").lower()


def parse(content: str) -> Optional[List[Dict[str, Any]]]:
    """Parse Super's tagged-line output into typed elements (prototype ``parse``).

    Consecutive ``<<item>>`` tags coalesce into one ``{"type": "list", "items":
    [...]}`` element; every other tag becomes ``{"type": kind, "text": text}``.
    Empty-text lines and untagged lines are dropped. Returns ``None`` when no
    element survives.
    """
    els: List[Dict[str, Any]] = []
    for line in (content or "").splitlines():
        m = _TAG_RE.match(line)
        if not m:
            continue
        kind, text = m.group(1), m.group(2).strip()
        if not text:
            continue
        if kind == "item":
            if els and els[-1]["type"] == "list":
                els[-1]["items"].append(text)
            else:
                els.append({"type": "list", "items": [text]})
        elif kind != "list":
            els.append({"type": kind, "text": text})
    return els or None


def texts_of(els: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Flatten elements to their ordered verbatim text fragments (prototype).

    Used for the conservation check: ``" ".join(texts_of(els))`` re-linearizes
    what Super emitted so it can be compared to the source page.
    """
    out: List[str] = []
    for e in els or []:
        if e.get("text"):
            out.append(str(e["text"]))
        out.extend(str(i) for i in (e.get("items") or []))
    return out


# ---------------------------------------------------------------------------
# Deterministic table detection + conversion (Super must LEAVE tables verbatim).
# ---------------------------------------------------------------------------
# A markdown table SEPARATOR run: a pipe-delimited run of dash cells (optional
# alignment colons), e.g. ``| --- | --- |``. Matched anywhere in the text with a
# SEARCH (not a per-line scan): the upstream VLM emits the whole pipe grid as one
# single-line blob inside ``region_provenance[].raw_text`` (the separator row is
# inline, not on its own physical line), which is exactly the shape
# lib.semantik.table_structure reads.
_MD_SEP_RUN_RE = re.compile(r"\|(?:\s*:?-{2,}:?\s*\|)+")

# A complete LaTeX tabular block; the optional {colspec} after \begin{tabular}
# is consumed so it never leaks into the row body.
_TABULAR_RE = re.compile(
    r"\\begin\{tabular\}(?:\{[^{}]*\})?(.*?)\\end\{tabular\}", re.DOTALL
)


def _has_md_separator_row(text: str) -> bool:
    """Whether ``text`` carries a markdown pipe-table header-separator run.

    Works on both the single-line fused blob (the production shape) and a
    newline-formatted grid — a data row (``| = | equals |``) never matches
    because its cells are not dash runs, and hyphenated prose never matches
    (no pipes).
    """
    return bool(_MD_SEP_RUN_RE.search(text or ""))


def is_table_text(text: str) -> bool:
    """Whether a fused block is actually a TABLE (leave it to deterministic code).

    True when the block carries a markdown pipe-table separator row
    (``| --- | --- |``) OR a LaTeX ``\\begin{tabular}`` environment; False on
    ordinary prose. A True verdict means :func:`resegment_page` must NOT call
    Super — the table is converted deterministically and left verbatim.
    """
    if not text:
        return False
    if "\\begin{tabular}" in text:
        return True
    return _has_md_separator_row(text)


def latex_tabular_to_rows(text: str) -> Optional[List[List[str]]]:
    """Parse a ``\\begin{tabular}...\\end{tabular}`` block into rows of cells.

    Rows split on ``\\\\``; cells split on ``&``; each cell is stripped but
    otherwise VERBATIM — inline LaTeX (``\\(...\\)``) is kept intact. Whitespace-
    only rows (e.g. a trailing ``\\\\``) are dropped. Returns ``None`` when no
    complete tabular environment is present or it yields no non-empty row.

    This is the LaTeX analogue of the markdown path already owned by
    ``lib.semantik.table_structure`` (not reimplemented here).
    """
    m = _TABULAR_RE.search(text or "")
    if not m:
        return None
    body = m.group(1)
    rows: List[List[str]] = []
    for raw_row in re.split(r"\\\\", body):
        if not raw_row.strip():
            continue
        cells = [c.strip() for c in raw_row.split("&")]
        rows.append(cells)
    return rows or None


def _markdown_table_rows(text: str) -> Optional[List[List[str]]]:
    """Deterministic markdown pipe-grid → rows, reusing lib.semantik.table_structure.

    Lazy-imports ``parse_table_topology`` (the strict, self-verifying cell-
    topology reconstructor) so this module imports cleanly even where ``lib`` is
    unavailable (e.g. the cross-venv bridge). Returns the header rows followed by
    the body rows, or ``None`` when the grid does not parse (the block is still a
    table — it is simply left verbatim, never sent to Super).
    """
    try:
        from lib.semantik.table_structure import parse_table_topology
    except Exception:  # noqa: BLE001 — lib absent → no structured rows, still a table
        logger.debug("super_resegment: lib.semantik.table_structure unavailable")
        return None
    try:
        topo = parse_table_topology(text)
    except Exception:  # noqa: BLE001 — a parser fault degrades, never raises
        logger.debug("super_resegment: parse_table_topology raised", exc_info=True)
        return None
    if topo is None:
        return None
    rows = [list(r) for r in topo.header_rows] + [list(r) for r in topo.body_rows]
    return rows or None


# ---------------------------------------------------------------------------
# The isolated Super HTTP call — a network error is caught by the caller and
# turned into a keep_fused result, never propagated.
# ---------------------------------------------------------------------------
def _super_completion(
    seat: SuperSeat, system_prompt: str, user_text: str, *, timeout: float
) -> Dict[str, Any]:
    """POST one prose page to the Super seat; return ``{content, finish}``.

    Mirrors the prototype's ``call()`` HTTP shape (tagged-line transport,
    temperature 0.6, top_p 0.95). This is the SOLE network boundary of the
    module, isolated so :func:`resegment_page` can wrap it in one try/except —
    tests monkeypatch THIS function.
    """
    body = json.dumps(
        {
            "model": seat.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "top_p": _TOP_P,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if seat.api_key:
        headers["Authorization"] = f"Bearer {seat.api_key}"
    url = seat.base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=body, headers=headers)
    import time as _time

    _t0 = _time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    _duration_ms = (_time.monotonic() - _t0) * 1000.0
    # P3 usage tap — best-effort per-call metering (no-op / swallowed on any
    # failure; never perturbs the call). Mirrors the Trainforge OP2 tap.
    try:
        from . import llm_usage_meter

        llm_usage_meter.record_llm_usage(
            site="super_resegment",
            model=seat.model,
            data=payload,
            duration_ms=_duration_ms,
        )
    except Exception:  # noqa: BLE001 — metering must never crash a call
        pass
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    return {
        "content": message.get("content") or "",
        "finish": choice.get("finish_reason") or "?",
    }


# ---------------------------------------------------------------------------
# The core stage entry point.
# ---------------------------------------------------------------------------
def resegment_page(
    fused_text: str,
    *,
    seat: Optional[SuperSeat] = None,
    mode: str = "on",
    timeout: float = 1400,
) -> Dict[str, Any]:
    """Re-segment ONE fused page into typed elements (or convert it as a table).

    Actions returned in the result ``dict``:

    * **table** — ``fused_text`` IS a table (:func:`is_table_text`): Super is NOT
      called; the block is converted deterministically (``latex_tabular_to_rows``
      for a ``\\begin{tabular}`` block, else the reused markdown topology parser).
      ``{"action": "table", "rows": <rows | None>, "conserved": True,
      "elements": None, "shadow": <bool>}`` — conserves by construction (the
      table is left verbatim).
    * **split** — a prose page Super re-segmented AND the char-exact conservation
      check passed: ``{"action": "split", "elements": [...], "conserved": True,
      "finish": ..., "shadow": <bool>}``.
    * **keep_fused** — a network error, an empty/untagged completion, OR a failed
      conservation check: ``{"action": "keep_fused", "elements": None,
      "conserved": False, "why": ..., "finish": ..., "shadow": <bool>}``. The
      caller keeps the fused block UNCHANGED — the safety invariant.

    ``mode`` is three-valued (:func:`resolve_super_resegment_mode`). ``"on"`` and
    ``"shadow"`` compute the result identically; ``"shadow"`` sets
    ``result["shadow"] = True`` so a caller can audit-without-applying. ``"off"``
    must never reach this function — it is guarded with a ``ValueError``. A Super
    HTTP failure is caught here and returned as ``keep_fused`` (never raised).
    """
    if mode == "off":
        raise ValueError(
            "resegment_page called with mode='off'; the caller must gate on "
            "resolve_super_resegment_mode() and skip this stage entirely when off."
        )
    if mode not in ("on", "shadow"):
        raise ValueError(f"resegment_page: unknown mode {mode!r} (want on/shadow/off)")

    shadow = mode == "shadow"

    # --- Table branch: deterministic, Super is NEVER called. ---
    if is_table_text(fused_text):
        if "\\begin{tabular}" in fused_text:
            rows = latex_tabular_to_rows(fused_text)
        else:
            rows = _markdown_table_rows(fused_text)
        return {
            "action": "table",
            "rows": rows,
            "conserved": True,
            "elements": None,
            "shadow": shadow,
        }

    # --- Prose branch: the isolated Super call, fully fail-soft. ---
    if seat is None:
        seat = resolve_super_seat()

    try:
        result = _super_completion(seat, _SYS_PROSE, fused_text, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — a network fault → keep_fused, never raise
        return {
            "action": "keep_fused",
            "elements": None,
            "conserved": False,
            "why": f"super http error: {str(exc)[:120]}",
            "finish": "-",
            "shadow": shadow,
        }

    content = result.get("content") or ""
    finish = result.get("finish") or "?"
    els = parse(content)
    if not els:
        why = (
            "truncated"
            if finish == "length"
            else ("empty" if not content.strip() else "untagged")
        )
        return {
            "action": "keep_fused",
            "elements": None,
            "conserved": False,
            "why": why,
            "finish": finish,
            "shadow": shadow,
        }

    conserved = _norm(fused_text) == _norm(" ".join(texts_of(els)))
    if not conserved:
        return {
            "action": "keep_fused",
            "elements": None,
            "conserved": False,
            "why": "not conserved (text lost or altered)",
            "finish": finish,
            "shadow": shadow,
        }

    return {
        "action": "split",
        "elements": els,
        "conserved": True,
        "finish": finish,
        "shadow": shadow,
    }


# ---------------------------------------------------------------------------
# Cascade-facing snap + entry (Step 2 of the refactor — the split-and-snap
# arranger seam). ``resegment_page`` splits the JOINED page text into typed
# elements; ``snap_elements_to_units`` re-anchors those splits to the parent's
# member-unit boundaries so the resulting sub-regions partition the parent's
# ``fb_idxs`` EXACTLY (the arranger's coverage invariant holds by construction).
# Both are PURE (no network); the network lives only in ``resegment_page``.
# ---------------------------------------------------------------------------
def snap_elements_to_units(
    elements: Optional[List[Dict[str, Any]]],
    member_units: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Snap Super's split ``elements`` onto the parent's ``member_units``.

    ``member_units`` is an ordered list of ``{"fb_index": int, "text": str}``.
    Because the fused page text IS ``" ".join(member unit texts)``, snapping ==
    walking the member units in order, greedily assigning consecutive units to
    the current element until the whitespace-normalized concatenation of the
    assigned units equals that element's normalized text, then advancing.

    Returns one descriptor per element —
    ``{"type": <element type>, "fb_idxs": [int...], "text": <joined member
    texts>, "items": [...]?}`` — such that the ``fb_idxs`` across all descriptors
    are EXACTLY the parent's member fb_idxs, in order, partitioned with no
    overlap and none dropped.

    Returns **None** (fail-closed → the caller keeps the single fused region)
    when the snap cannot cover the units exactly: a Super boundary that does not
    align to any unit boundary (an assigned concatenation OVERSHOOTS the element
    text), an element that consumes zero units, or leftover units after the last
    element. Pure, no network.
    """
    units_norm = [_norm(u.get("text") or "") for u in member_units]
    n = len(member_units)
    out: List[Dict[str, Any]] = []
    ui = 0
    for el in elements or []:
        target = _norm(" ".join(texts_of([el])))
        if not target:
            # A degenerate (empty) element can own no unit boundary.
            return None
        acc = ""
        start = ui
        while ui < n and len(acc) < len(target):
            acc += units_norm[ui]
            ui += 1
        if ui == start or acc != target:
            # No unit consumed, or the unit boundary overshoots / misaligns.
            return None
        assigned = member_units[start:ui]
        fb_idxs = [int(u["fb_index"]) for u in assigned]
        text = " ".join(
            (u.get("text") or "").replace("\n", " ").strip() for u in assigned
        ).strip()
        entry: Dict[str, Any] = {
            "type": el.get("type"),
            "fb_idxs": fb_idxs,
            "text": text,
        }
        items = el.get("items")
        if items:
            entry["items"] = list(items)
        out.append(entry)
    if ui != n:
        # Leftover unclaimed member units — cannot partition exactly.
        return None
    return out or None


def resegment_arranger_paragraph(
    member_units: List[Dict[str, Any]],
    joined_text: str,
    *,
    mode: str,
    seat: Optional[SuperSeat] = None,
    timeout: float = 1400,
    audit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Combine :func:`resegment_page` + :func:`snap_elements_to_units`.

    Returns the snapped sub-region descriptors when Super produced a conserved
    ``split`` that snaps cleanly to the member units AND ``mode == "on"``;
    returns **None** in every other case (the caller keeps the single fused
    region):

    * ``action == "table"`` → None (tables are handled by the deterministic
      table lane elsewhere; leave the fused region for now);
    * ``keep_fused`` / snap failure / ``mode == "shadow"`` / any error → None.

    In ``shadow`` mode everything is computed (the snap is attempted for
    calibration evidence) but **None is always returned** (apply nothing). The
    optional ``audit`` callback is invoked with a summary dict in EVERY mode
    (including shadow) so a caller can record the conservation / snap result
    without mutating regions. ``audit`` failures are swallowed (best-effort).
    """
    if mode == "off":
        return None
    try:
        result = resegment_page(joined_text, seat=seat, mode=mode, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — never break the arranger
        logger.debug("super_resegment: resegment_page raised: %s", exc)
        return None

    action = result.get("action")
    snapped: Optional[List[Dict[str, Any]]] = None
    if action == "split" and result.get("conserved"):
        snapped = snap_elements_to_units(result.get("elements") or [], member_units)

    if audit is not None:
        try:
            audit(
                {
                    "action": action,
                    "conserved": bool(result.get("conserved")),
                    "shadow": bool(result.get("shadow")),
                    "n_elements": len(result.get("elements") or []),
                    "snapped": snapped is not None,
                    "n_regions": len(snapped) if snapped else 0,
                    "why": result.get("why"),
                }
            )
        except Exception:  # noqa: BLE001 — audit is best-effort, never fatal
            logger.debug("super_resegment: audit callback raised", exc_info=True)

    if mode == "shadow":
        return None
    if action == "table":
        return None
    return snapped
