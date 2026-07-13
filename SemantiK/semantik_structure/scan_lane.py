"""SEMANTIK_SCAN_LANE_DEBERT — the OCR/scan-lane BERT de-poisoning gate (task #43).

The council BERTs (Structure / Semantic / MergeOrSplit) are tuned on
born-digital textbook pages; the task-#25 oracle A/B established that OFF the
tuning domain — an inaccessible image-only *scan* — those heads COLLAPSE
(section recall ~0.25, ~20 false-positive apparatus/pedagogical-label
"headings" burying a couple of real ones), poisoning the deterministic
synthesis windows downstream. The deterministic furniture/apparatus regexes,
the VLM structure hints, and the outline/ToC anchor are the signals that HOLD
off-domain.

This gate is the switch that makes that observation actionable: when the
document is on the **OCR/scan lane** (its FeatureBlocks are predominantly
``tesseract``-provenance) AND the flag is truthy, EVERY council-BERT
*structural binding* becomes NON-BINDING — the enumerated head signals are
dropped from the ``CouncilState`` before the cross-reranker (Stage 4) and the
structure graph (Stage 5) read them, so each typing decision falls to its
DETERMINISTIC / geometric / default arm (which already exists). The BERT votes
are recorded as a per-region ``typing_authority`` provenance HINT, never a
kind authority.

Contract (mirrors ``SEMANTIK_STRUCTURE_ROUTER`` / ``resolve_bert_authoritative``):

* **Gate ``SEMANTIK_SCAN_LANE_DEBERT`` (default OFF).** Off → the mask never
  runs, the ``CouncilState`` passed to Stages 4/5 is the real one, and every
  output byte + ``region_provenance`` entry is byte-identical to today.
* **Natural NO-OP on the born-digital lane.** Even truthy, the mask fires only
  when :func:`is_ocr_scan_lane` — a born-digital PDF (pypdfium2/pdfplumber
  provenance) is unaffected (byte-identical).
* **KEPT fully active in both modes:** the deterministic furniture detection
  (``_detect_running_headers`` etc.), the ``deterministic_structure``
  pedagogical/apparatus regexes, ``vlm_furniture``, and the outline-anchor +
  structure-router paths — none of them read the neutralized ``state`` signals,
  so this gate cannot touch them.

NO new LLM call site is added (a pure deterministic mask), so — per the root
no-silent-LLM-call contract — it carries NO ``DecisionCapture`` obligation
(flagged loudly here: pure functions).
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any

__all__ = [
    "SEMANTIK_SCAN_LANE_DEBERT_ENV",
    "NEUTRALIZED_HEADS",
    "resolve_scan_lane_debert_mode",
    "is_ocr_scan_lane",
    "scan_lane_debert_active",
    "mask_council_state",
    "typing_authority_for_payload",
]

SEMANTIK_SCAN_LANE_DEBERT_ENV = "SEMANTIK_SCAN_LANE_DEBERT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# The council ``(bert_name, head_name)`` STRUCTURAL bindings neutralized on the
# OCR/scan lane when the gate is active. Exactly the heads the cross-reranker /
# structure-graph typing floors read via ``_get_signal`` — the enumerated
# task-#43 sites: structural_role + is_heading(_calibrated) + table_region +
# is_image_block + list_nesting + pedagogical_role (Structure), doc_role
# (Semantic), and same_logical_block (MergeOrSplit — DOWNGRADED to a hint, not
# removed: its vote no longer drives the partition). Text-repair heads
# (``hyphen_repair``) and any non-enumerated head pass through UNCHANGED — this
# gate is about *structural* authority, not text.
NEUTRALIZED_HEADS: frozenset[tuple[str, str]] = frozenset(
    {
        ("structure", "structural_role"),
        ("structure", "pedagogical_role"),
        ("structure", "is_heading"),
        ("structure", "is_heading_calibrated"),
        ("structure", "table_region"),
        ("structure", "is_image_block"),
        ("structure", "list_nesting"),
        ("semantic", "doc_role"),
        ("merge_or_split", "same_logical_block"),
    }
)

# Minimum share of TEXT FeatureBlocks that must carry ``tesseract`` provenance
# for the document to count as ON the OCR/scan lane. A born-digital PDF's blocks
# are pypdfium2/pdfplumber; a scanned one's are OCR (and the VLM-fusion lane
# forces ``tesseract`` provenance, so a fused scan still counts). Calibration
# constant, NOT a corpus target.
_SCAN_LANE_MIN_OCR_SHARE = 0.5  # TODO(calibration)


def resolve_scan_lane_debert_mode() -> bool:
    """Whether the scan-lane BERT de-poisoning gate may fire (default OFF).

    Parse-with-fallback truthy set (``1`` / ``true`` / ``yes`` / ``on``,
    case-insensitive); unset / blank / falsey / garbage → off (byte-identical
    pass-through — the mask never runs). Read at CALL time so tests / Docker can
    flip it without a re-import. Mirrors
    ``structure_router.resolve_structure_router_mode``.
    """
    return (os.environ.get(SEMANTIK_SCAN_LANE_DEBERT_ENV) or "").strip().lower() in _TRUTHY


def _fb_source(fb: Any) -> str:
    """The extraction-source string of a FeatureBlock (``provenance`` first,
    else the underlying RawBlock ``source``)."""
    src = getattr(fb, "provenance", None)
    if not src:
        src = getattr(getattr(fb, "raw", None), "source", None)
    return str(src or "")


def is_ocr_scan_lane(feature_blocks: Any) -> bool:
    """True iff the document was predominantly OCR-extracted (the scan lane).

    Deterministic, model-free: the share of TEXT FeatureBlocks whose extraction
    provenance mentions ``tesseract`` at/above :data:`_SCAN_LANE_MIN_OCR_SHARE`.
    Synthetic image FeatureBlocks (``is_image``) are excluded. An empty stream
    is NOT the scan lane (fail-open — never mask a document with no evidence).
    """
    ocr = 0
    total = 0
    for fb in feature_blocks or ():
        if getattr(fb, "is_image", False):
            continue
        total += 1
        if "tesseract" in _fb_source(fb).lower():
            ocr += 1
    if total == 0:
        return False
    return (ocr / total) >= _SCAN_LANE_MIN_OCR_SHARE


def scan_lane_debert_active(feature_blocks: Any) -> bool:
    """The load-bearing verdict: gate truthy AND the document on the OCR lane."""
    return resolve_scan_lane_debert_mode() and is_ocr_scan_lane(feature_blocks)


def mask_council_state(state: Any) -> Any:
    """Return a ``CouncilState`` with the NEUTRALIZED structural head signals
    dropped, so every consumer that reads them via ``_get_signal`` (the
    cross-reranker table/image confirm + ``doc_role`` override, the
    structure-graph Pass-6 typing floors + heading gates, the ``merge_or_split``
    boundary vote) sees them ABSENT and falls to its deterministic / geometric /
    default arm.

    A pure, allocation-light transform: it ``dataclasses.replace``s only the
    ``BertOutput``s that actually carry a neutralized signal (others pass through
    by identity), and leaves the original ``state`` untouched (the caller keeps
    it for Stage-6 authoring + ``signal_coverage`` observability). Fail-open: a
    ``state`` without a dict ``outputs`` is returned unchanged.
    """
    outputs = getattr(state, "outputs", None)
    if not isinstance(outputs, dict):
        return state
    new_outputs: dict[str, Any] = {}
    for bert_name, bert_out in outputs.items():
        signals = getattr(bert_out, "signals", None)
        if not signals:
            new_outputs[bert_name] = bert_out
            continue
        kept = [
            s
            for s in signals
            if (bert_name, getattr(s, "head_name", None)) not in NEUTRALIZED_HEADS
        ]
        new_outputs[bert_name] = (
            bert_out if len(kept) == len(signals) else dataclasses.replace(bert_out, signals=kept)
        )
    return dataclasses.replace(state, outputs=new_outputs)


def typing_authority_for_payload(payload: dict[str, Any] | None) -> str:
    """The per-region ``typing_authority`` value stamped when the gate is active.

    ``"vlm"`` when a VLM/router heading hint promoted the region (the region
    carries the ``vlm_corroborated`` audit flag), else ``"deterministic"``. The
    legacy value ``"bert"`` is never emitted under the gate — its ABSENCE (the
    key omitted, byte-identical) is the "council authoritative" default, so a
    downstream assert of "zero regions typed by a collapsed BERT on the scan
    lane" is ``all(e.get("typing_authority") != "bert")``.
    """
    payload = payload or {}
    if payload.get("vlm_corroborated"):
        return "vlm"
    return "deterministic"
