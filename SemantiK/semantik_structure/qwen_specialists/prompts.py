"""Stage 6 prompt builders — region → SpecialistRequest.

Three builders, one per real (non-gap-fill) adapter:

    build_prose_request(region, feature_blocks) -> SpecialistRequest
    build_table_request(region, feature_blocks) -> SpecialistRequest
    build_math_request(region, feature_blocks) -> SpecialistRequest

The Stage 6 driver (``runner.py``) calls these to produce a
``SpecialistRequest``; the runner is responsible for assigning
``request_id`` (typically ``f"r{idx}"`` where ``idx`` indexes the
region in the input list).

Prompt envelope (the trained-in contract)
-----------------------------------------

Every prompt is a chat-style two-turn exchange:

    SYSTEM: <one-line role specification>
    USER:   <single-line JSON payload>

The JSON payload has a strictly-typed shape per adapter; it is what
each Qwen LoRA is (will be) trained against, so any drift in the
keyset here invalidates the trained adapter. The shapes:

    PROSE
        {
          "kind": "paragraph"|"heading"|"list"|...,
          "level_hint": int|null,        # heading regions only
          "doc_role": str|null,          # Semantic doc_role override, paragraph only
          "ordered": bool|null,          # list regions only
          "items": [...]|null,           # list regions only
          "aria_hints": [str, ...],
          "text": str
        }

    TABLE
        {
          "kind": "table",
          "n_rows": int,
          "n_cols": int,
          "bordered": bool,
          "header_row_indices": [int, ...],
          "caption_text": str|null,
          "cell_grid": [[
              {"text": str, "rowspan": int, "colspan": int, "role": str},
              ...
          ], ...]
        }

    MATH
        {
          "kind": "math",
          "src_text": str,
          "display": bool,
          "glyph_density_features": {str: float}
        }

The system prompt fixes the deliverable: HTML5 + ARIA-in-HTML for
prose/table, MathML 4.0 + ``alttext`` for math. WCAG 2.2 AA — see
``docs/ontology.md`` §2.5/§2.10.

Provenance + metadata
---------------------

Every ``SpecialistRequest.metadata`` carries:

    {"region_kind": str, "fb_indices": tuple[int, ...]}

so candidates emerging from the runtime can be tied back to their
source region without the runner re-deriving it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .types import AdapterID, SpecialistRequest


# ---------------------------------------------------------------------------
# Legal/copyright sub-flag heuristics — SHARED contract surface.
#
# Both the runtime detector (assembler.pass_9a) and the dataset builder
# (data/builders/build_gap_fill_qwen_data.py) import these so the copyright-vs-
# disclaimer split and the ``legal_subkind`` discriminator can never
# drift between training rows and inference GapSlots. They live here
# (not in the assembler package) because the dataset-build environment
# must not transitively import the assembler's inference-only deps
# (axe_playwright_python) — same reason the builder keeps a local
# GapKind enum.
# ---------------------------------------------------------------------------

# Plans/04 §2 copyright sub-flag. The spec sketch is
# ``r"\b(©|copyright|\(c\)|all rights reserved)\b"i``; implemented with
# two precision fixes: ``\b©`` never matches at string start (© is a
# non-word char, so the spec regex misses a leading ©), and bare
# ``(c)`` only counts when followed by a digit (otherwise every
# ``(c) the party shall…`` enumeration item in legal prose would flip
# a disclaimer into a copyright_block).
COPYRIGHT_SUBFLAG_RE = re.compile(
    r"(©|\(c\)\s*(?=\d)|\bcopyright\b|\ball rights reserved\b)",
    re.IGNORECASE,
)

# ``has_copyright_year_match`` context flag (Plans/04 §2 copyright_block
# context shape): a ©/copyright marker with an adjacent plausible year.
COPYRIGHT_YEAR_RE = re.compile(
    r"(?:©|\(c\)|\bcopyright\b)[\s:]*(?:1[5-9]|20)\d{2}",
    re.IGNORECASE,
)

# ``legal_subkind`` discriminator (Plans/04 §2 legal_disclaimer context
# shape). Deterministic first-match phrase classifier; "disclaimer" is
# the catch-all.
_LEGAL_SUBKIND_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("license", re.compile(r"\blicen[cs]e[ds]?\b|\blicensing\b", re.IGNORECASE)),
    (
        "terms",
        re.compile(
            r"\bterms\s+(?:of\s+(?:use|service)|and\s+conditions)\b",
            re.IGNORECASE,
        ),
    ),
    ("public_domain", re.compile(r"\bpublic\s+domain\b", re.IGNORECASE)),
    (
        "warranty",
        re.compile(
            r"\bwarrant(?:y|ies)\b|\bliabilit(?:y|ies)\b",
            re.IGNORECASE,
        ),
    ),
]


def classify_legal_subkind(text: str) -> str:
    """Return the ``legal_subkind`` discriminator for a legal region."""
    for name, pattern in _LEGAL_SUBKIND_PATTERNS:
        if pattern.search(text or ""):
            return name
    return "disclaimer"


# ---------------------------------------------------------------------------
# System prompts — the contract surface for trained-adapter behaviour.
# ---------------------------------------------------------------------------

_SYSTEM_PROSE = (
    "You convert one PDF region into accessible HTML5 conforming to "
    "ARIA-in-HTML and WCAG 2.2 AA. Emit one HTML5 fragment only — no "
    "Markdown, no commentary, no code fences."
)

_SYSTEM_TABLE = (
    "You convert one tabular PDF region into a single accessible "
    "HTML5 <table> with <caption>, <thead>/<tbody>, and <th scope=...> "
    "as required by WCAG 2.2 AA. Per-cell roles already labelled as "
    "header_row / header_col / span / data must be honoured. Emit the "
    "<table> fragment only — no Markdown, no commentary."
)

_SYSTEM_MATH = (
    "You convert one math expression into MathML 4.0 with an alttext "
    "attribute matching the source. Emit one <math display=...>...</math> "
    "fragment only — no Markdown, no commentary, no LaTeX."
)

# NOTE: this SYSTEM line is the trained-in contract surface shared by ALL
# gap kinds and is pinned byte-for-byte by the shipped gap_fill dataset
# (tests/test_qwen_lora_format_roundtrip.py). Do NOT edit it without
# rebuilding the dataset — per-kind guidance belongs in the USER payload
# (see the citation_unresolved branch's "instruction" field below), which
# only appears on the rows that need it and leaves the V1 missing_title /
# author_block prompts byte-identical.
_SYSTEM_GAP_FILL = (
    "You produce an HTML5 fragment to fill a flagged document-level gap. "
    "Output must be a single HTML element (or small fragment) conforming to "
    "ARIA-in-HTML and WCAG 2.2 AA. No Markdown, no commentary."
)


# Cheap character-count proxy for table prompts that risk truncation. Tabular
# numeric content tokenizes densely, so the 8.5K-character limit leaves
# completion headroom inside the configured context window. The runner also
# performs the authoritative check with the real tokenizer.
_LONG_TABLE_CHAR_LIMIT = 8_500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_prompt(system: str, user_json: dict[str, Any]) -> str:
    """Serialize the SYSTEM + USER turns into a single prompt string.

    The format is intentionally simple (two labelled lines, one JSON
    payload) so it works against any Qwen chat template — the trained
    adapter will use whatever wrapping its tokenizer prefers, but the
    line content stays stable. ``ensure_ascii=False`` so unicode in the
    region text round-trips exactly.
    """
    payload = json.dumps(user_json, ensure_ascii=False, separators=(",", ": "))
    return f"SYSTEM: {system}\nUSER: {payload}"


def _region_text(region: Any) -> str:
    """Pull payload text robustly for prose/list/figure/etc."""
    payload = region.payload or {}
    text = payload.get("text")
    if text is not None:
        return text
    # Lists store individual item text; concatenate for the envelope.
    items = payload.get("items")
    if items:
        return "\n".join((item.get("text") or "") for item in items)
    return ""


def _caption_text(region: Any, feature_blocks: Sequence[Any]) -> str | None:
    """Look up the caption FB the table-passthrough pass marked.

    Returns ``None`` if no caption was identified or if the index is
    out of range (defensive — the structure_graph caption pick is a
    best-effort heuristic).
    """
    payload = region.payload or {}
    cap_idx = payload.get("caption_fb_index")
    if cap_idx is None or cap_idx < 0 or cap_idx >= len(feature_blocks):
        return None
    fb = feature_blocks[cap_idx]
    raw = getattr(fb, "raw", None)
    text = getattr(raw, "text", None) if raw is not None else None
    return (text or "").strip() or None


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_prose_request(
    region: Any,
    feature_blocks: Sequence[Any],
) -> SpecialistRequest:
    """Build a SpecialistRequest for a prose-routed region (paragraph,
    heading, list, code_block, blockquote, figure, form,
    metadata_drop, definition_list).
    """
    payload = region.payload or {}
    user: dict[str, Any] = {
        "kind": region.kind,
        "level_hint": payload.get("level_hint"),
        "doc_role": payload.get("doc_role"),
        "ordered": payload.get("ordered"),
        "items": payload.get("items"),
        "aria_hints": list(region.aria_hints or ()),
        "text": _region_text(region),
    }
    return SpecialistRequest(
        adapter=AdapterID.PROSE,
        payload={"prompt": _format_prompt(_SYSTEM_PROSE, user)},
        request_id="",
        sampling_overrides={
            "metadata": {
                "region_kind": region.kind,
                "fb_indices": tuple(region.feature_block_indices),
            },
        },
    )


def build_table_request(
    region: Any,
    feature_blocks: Sequence[Any],
) -> SpecialistRequest:
    """Build a SpecialistRequest for a table region.

    Per-cell roles are pulled from ``region.payload["cell_roles"]``
    (populated by structure_graph.py from the BERT-TableSpecialist
    output). When ``cell_roles`` is absent or ``None``, every cell
    falls back to ``"data"`` and the user envelope carries
    ``cell_roles_source = "qwen_inferred"`` — a marker the trained
    adapter's prompt template MUST surface as the legacy
    ``data-dart-cell-roles="qwen-inferred"`` attribute (a pre-SemantiK
    weights artifact) on the emitted ``<table>``
    so Stage 12 (theta) and any HTML consumer can see the table's
    header semantics were guessed without a verified upstream signal
    (no-silent-fallbacks policy; see
    feedback_no_silent_fallbacks.md). The mock runtime applies the
    marker directly. When ``cell_roles`` is present the
    ``cell_roles_source`` key is OMITTED (verified is the implicit
    happy-path default) so existing dataset rows — built with ground-
    truth ar5iv roles before this audit field existed — round-trip
    byte-for-byte through the contract test.

    Stage 7's per-region gate does NOT structurally validate
    ``<th scope=...>`` presence (only HTML5 well-formedness, text-
    preservation [skipped for tables], MathML, and axe-core under
    WCAG 2.2 AA). axe will catch some but not all missing-header
    cases (e.g. a 1xN table without ``<th>`` typically passes axe
    serious/critical), so the marker is the auditable trail.
    """
    payload = region.payload or {}
    grid: list[list[Any]] = list(payload.get("cell_grid") or [])
    cell_roles = payload.get("cell_roles")
    n_rows = int(payload.get("n_rows") or len(grid))
    n_cols = int(payload.get("n_cols") or max((len(r) for r in grid), default=0))
    header_row_indices = list(payload.get("header_row_indices") or [])

    # Build the per-cell envelope, defaulting role to "data" everywhere
    # cell_roles doesn't carry an entry. This keeps the envelope shape
    # stable whether or not BERT-TableSpecialist ran for this document.
    cell_envelope: list[list[dict[str, Any]]] = []
    for i, row in enumerate(grid):
        envelope_row: list[dict[str, Any]] = []
        for j, raw_cell in enumerate(row):
            role = "data"
            if cell_roles is not None and i < len(cell_roles) and j < len(cell_roles[i]):
                role = cell_roles[i][j] or "data"
            envelope_row.append(
                {
                    "text": (str(raw_cell) if raw_cell is not None else ""),
                    "rowspan": 1,
                    "colspan": 1,
                    "role": role,
                }
            )
        cell_envelope.append(envelope_row)

    user: dict[str, Any] = {
        "kind": "table",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "bordered": bool(payload.get("bordered")),
        "header_row_indices": header_row_indices,
        "caption_text": _caption_text(region, feature_blocks),
        "cell_grid": cell_envelope,
    }
    # Audit marker — present ONLY when cell_roles is None (no BERT-
    # TableSpecialist signal). Absent (= implicitly verified) on the
    # happy path so existing dataset rows — built before this field
    # existed and always with verified ground-truth roles from ar5iv —
    # round-trip identically through the contract test. The trained
    # adapter's prompt template MUST treat the absence as "verified" and
    # the presence of "qwen_inferred" as the trigger to emit the legacy
    # ``data-dart-cell-roles="qwen-inferred"`` attribute (a pre-SemantiK
    # weights artifact) on the outer ``<table>``.
    if cell_roles is None:
        user["cell_roles_source"] = "qwen_inferred"
    # Stage 5b GLM-OCR enrichment, when present. The trained adapter
    # gets a second view of the same cells from the vision model — the
    # prompt template should treat this as an *advisory secondary
    # signal* (better OCR for low-confidence cells), not as the source
    # of truth (the structured cell_grid is). Absent on PDFs where
    # Stage 5b didn't run or the table_region head didn't confirm.
    glm_ocr_text = payload.get("glm_ocr_text")
    if glm_ocr_text:
        user["glm_ocr_text"] = str(glm_ocr_text)
    # Compute the prompt once so we can also check its length for the long-
    # table routing flag (see _LONG_TABLE_CHAR_LIMIT above + Plans/06 §7).
    prompt = _format_prompt(_SYSTEM_TABLE, user)
    metadata: dict[str, Any] = {
        "region_kind": "table",
        "fb_indices": tuple(region.feature_block_indices),
    }
    if len(prompt) > _LONG_TABLE_CHAR_LIMIT:
        metadata["skip_qwen_long_table"] = True
        metadata["prompt_char_len"] = len(prompt)
    return SpecialistRequest(
        adapter=AdapterID.TABLE,
        payload={"prompt": prompt},
        request_id="",
        sampling_overrides={"metadata": metadata},
    )


def build_math_request(
    region: Any,
    feature_blocks: Sequence[Any],
) -> SpecialistRequest:
    """Build a SpecialistRequest for a math region.

    The ``glyph_density_features`` are the same dict the math
    detector emitted (math_font_frac, cid_token_count, ...); they
    give the adapter a soft hint about how confidently the source
    is "real math" vs. mis-detected prose.
    """
    payload = region.payload or {}
    user: dict[str, Any] = {
        "kind": "math",
        "src_text": payload.get("src_text") or "",
        "display": bool(payload.get("display", True)),
        "glyph_density_features": dict(payload.get("glyph_density_features") or {}),
    }
    return SpecialistRequest(
        adapter=AdapterID.MATH,
        payload={"prompt": _format_prompt(_SYSTEM_MATH, user)},
        request_id="",
        sampling_overrides={
            "metadata": {
                "region_kind": "math",
                "fb_indices": tuple(region.feature_block_indices),
            },
        },
    )


def build_gap_fill_request(gap_slot: Any) -> SpecialistRequest:
    """Build a SpecialistRequest for the gap-fill adapter.

    The user payload is a JSON envelope shaped per
    Plans/04 §2 — different keys per :class:`GapKind`. The mock runtime
    sniffs the leading ``kind=<name>`` line to decide which fragment
    shape to emit; the trained adapter consumes the full JSON payload.
    """
    kind = getattr(gap_slot, "kind", None)
    kind_value = getattr(kind, "value", None) or str(kind) if kind is not None else "unknown"
    context = getattr(gap_slot, "context", None) or {}
    if kind_value == "missing_title":
        target = "<title>...</title> + <h1>...</h1>"
        payload: dict[str, Any] = {
            "task": "missing_title",
            "first_paragraph_text": (str(context.get("first_paragraph_text", ""))[:500]),
            "doc_lang": str(context.get("doc_lang", "en")),
        }
    elif kind_value == "author_block":
        target = "<address>...</address>"
        payload = {
            "task": "author_block",
            "raw_text": str(context.get("raw_text", "")),
            "surrounding_context": (str(context.get("surrounding_context", ""))[:300]),
        }
    elif kind_value == "citation_unresolved":
        # Restore the anchor around a broken inline reference, OR confirm
        # "leave as plain text" when none of the candidate targets is the
        # real referent. The adapter must NOT invent an anchor id outside
        # ``candidate_targets`` (no-silent-fallbacks): the only valid
        # outputs are an <a href="#<one of the candidate anchor_ids>">
        # wrapping the match_text, or the verbatim plain-text span.
        target = '<a href="#...">[N]</a> | leave-as-plain-text'
        payload = {
            "task": "citation_unresolved",
            "instruction": (
                'Wrap match_text in <a href="#..."> using ONE candidate_targets '
                "anchor_id, or output match_text verbatim as plain text if none "
                "is the true referent. Never invent an anchor id."
            ),
            "match_text": str(context.get("match_text", "")),
            "surrounding_50_chars": (str(context.get("surrounding_50_chars", ""))[:120]),
            "candidate_targets": list(context.get("candidate_targets", []) or []),
            "reference_kind": str(context.get("reference_kind", "")),
        }
    elif kind_value == "copyright_block":
        # Wrap the detected copyright/rights text in the assembler's
        # canonical <aside class="copyright"> landmark (Plans/04 §2/§4.1;
        # architecture.md: the assembler emits <aside> for legal/copyright
        # and does NOT wire body-scope <footer>). The text must survive
        # verbatim — a copyright line is a rights claim, not prose to
        # paraphrase (no-silent-fallbacks: invented years/names are
        # hallucinated legal facts).
        target = '<aside class="copyright">...</aside>'
        payload = {
            "task": "copyright_block",
            "instruction": (
                'Wrap raw_text in <aside class="copyright"> (text verbatim, '
                "inside a <p>). Never invent or alter names, years, or "
                "rights claims."
            ),
            "raw_text": str(context.get("raw_text", ""))[:1000],
            "has_copyright_year_match": bool(context.get("has_copyright_year_match", False)),
            "surrounding_context": (str(context.get("surrounding_context", ""))[:300]),
        }
    elif kind_value == "legal_disclaimer":
        # Same shape as copyright_block plus the legal_subkind
        # discriminator (Plans/04 §2). Canonical wrap is
        # <aside class="legal">.
        target = '<aside class="legal">...</aside>'
        payload = {
            "task": "legal_disclaimer",
            "instruction": (
                'Wrap raw_text in <aside class="legal"> (text verbatim, '
                "inside a <p>). Never invent or alter legal terms."
            ),
            "legal_subkind": str(context.get("legal_subkind", "disclaimer")),
            "raw_text": str(context.get("raw_text", ""))[:1000],
            "surrounding_context": (str(context.get("surrounding_context", ""))[:300]),
        }
    else:
        target = "<p>...</p>"
        payload = {"task": kind_value, "context": context}
    user_str = f"kind={kind_value}\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ": ")
    )
    prompt = f"SYSTEM: {_SYSTEM_GAP_FILL}\nUSER: {user_str}\n"
    return SpecialistRequest(
        adapter=AdapterID.GAP_FILL,
        payload={"prompt": prompt, "target": target, "kind": kind_value},
        request_id=f"gap_{kind_value}",
        sampling_overrides={
            "metadata": {
                "gap_kind": kind_value,
                "gap_region_index": getattr(gap_slot, "region_index", -1),
            },
        },
    )


__all__ = [
    "COPYRIGHT_SUBFLAG_RE",
    "COPYRIGHT_YEAR_RE",
    "build_gap_fill_request",
    "build_math_request",
    "build_prose_request",
    "build_table_request",
    "classify_legal_subkind",
]
