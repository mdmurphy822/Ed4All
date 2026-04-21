"""DART ontology-aware HTML converter package (Wave 12 foundation).

Four-phase pipeline that replaces the monolithic regex-driven converter:

    1. segment  - split raw text into RawBlocks (block_segmenter)
    2. classify - assign a BlockRole to each block (heuristic_classifier
                  or llm_classifier when DART_LLM_CLASSIFICATION=true)
    3. template - render each classified block via role template
    4. assemble - stitch rendered blocks into a full HTML document

Wave 12 shipped the foundation: enum, dataclasses, segmenter, heuristic
classifier (ports existing regex logic from pipeline_tools), minimal
template registry, and minimal assembler.

Subsequent waves:

    Wave 13 - expand templates with DPUB-ARIA + schema.org + rich HTML
    Wave 14 - add Claude-backed classifier behind LLMBackend (this wave)
    Wave 15 - document-level decoration (Dublin Core, JSON-LD, WCAG CSS)
    Wave 16 - dual-extraction + MathML + figure pipeline

The existing ``_raw_text_to_accessible_html`` in ``MCP/tools/pipeline_tools.py``
is left untouched. Wave 15 will flip the production path to this package.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, Union

from DART.converter.block_roles import (
    BlockRole,
    ClassifiedBlock,
    RawBlock,
)
from DART.converter.block_segmenter import (
    segment_extracted_document,
    segment_pdftotext_output,
)
from DART.converter.block_templates import TEMPLATE_REGISTRY, render_block
from DART.converter.document_assembler import assemble_html
from DART.converter.extractor import (
    ExtractedDocument,
    ExtractedFigure,
    ExtractedLink,
    ExtractedTOCEntry,
    ExtractedTable,
    ExtractedTextSpan,
    PageChrome,
    extract_document,
    median_body_font_size,
)
from DART.converter.heuristic_classifier import HeuristicClassifier
from DART.converter.llm_classifier import LLMClassifier
from DART.converter.mathml import detect_formulas, render_mathml
from DART.converter.page_chrome import detect_page_chrome, strip_page_chrome


def default_classifier(
    llm: Optional[Any] = None,
    *,
    text_spans: Optional[list] = None,
    median_body_font_size: Optional[float] = None,
) -> Union[LLMClassifier, HeuristicClassifier]:
    """Return the configured classifier for the current environment.

    Routing rules:

    * ``DART_LLM_CLASSIFICATION=true`` **and** ``llm`` provided →
      :class:`LLMClassifier` wrapping the injected backend.
    * Otherwise (flag off, flag on without a backend, or any other
      combination) → :class:`HeuristicClassifier`.

    This factory keeps callers flag-agnostic: set the env var once and
    every call-site picks up the new classifier, while unit tests that
    don't opt in keep the deterministic heuristic path.

    Wave 18: ``text_spans`` / ``median_body_font_size`` flow into
    :class:`HeuristicClassifier` so font-size heading promotion runs
    when PyMuPDF layout info is available. The LLM classifier ignores
    these kwargs — its prompt-based classification already uses text
    context directly. Extra kwargs are only consumed when the resolved
    classifier is the heuristic one.
    """
    flag = os.environ.get("DART_LLM_CLASSIFICATION", "").strip().lower()
    if flag == "true" and llm is not None:
        return LLMClassifier(llm=llm)
    return HeuristicClassifier(
        text_spans=text_spans,
        median_body_font_size=median_body_font_size,
    )


def convert_pdftotext_to_html(
    raw_text: str,
    title: str,
    metadata: dict | None = None,
    *,
    llm: Optional[Any] = None,
) -> str:
    """Full 4-phase pipeline: segment -> classify -> template -> assemble.

    Wave 14: adds an optional ``llm`` parameter. When the
    ``DART_LLM_CLASSIFICATION`` flag is set and ``llm`` is provided,
    classification is routed through Claude via :class:`LLMClassifier`;
    otherwise the heuristic classifier is used.
    """
    blocks = segment_pdftotext_output(raw_text)
    classifier = default_classifier(llm=llm)
    classified = _run_classifier_sync(classifier, blocks)
    return assemble_html(classified, title, metadata or {})


async def aconvert_pdftotext_to_html(
    raw_text: str,
    title: str,
    metadata: dict | None = None,
    *,
    llm: Optional[Any] = None,
) -> str:
    """Async variant of :func:`convert_pdftotext_to_html`.

    Prefer this from async contexts (pytest-asyncio, notebooks, async
    web workers). Awaits the classifier directly instead of bouncing the
    coroutine through a worker thread.
    """
    blocks = segment_pdftotext_output(raw_text)
    classifier = default_classifier(llm=llm)
    if isinstance(classifier, HeuristicClassifier):
        classified = classifier.classify_sync(blocks)
    else:
        classified = await classifier.classify(blocks)
    return assemble_html(classified, title, metadata or {})


def _run_classifier_sync(classifier, blocks):
    # HeuristicClassifier.classify_sync avoids touching the loop; prefer
    # it when the classifier is the heuristic (the vast majority case).
    if isinstance(classifier, HeuristicClassifier):
        return classifier.classify_sync(blocks)

    coro = classifier.classify(blocks)
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is None:
        return asyncio.run(coro)

    # A loop is already running (pytest-asyncio, notebooks, async workers).
    # Drive the coroutine on a dedicated thread so we don't deadlock the
    # caller's loop and don't crash with "asyncio.run() cannot be called
    # from a running event loop".
    import threading

    result: list = []
    error: list = []

    def _runner():
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


__all__ = [
    "BlockRole",
    "ClassifiedBlock",
    "ExtractedDocument",
    "ExtractedFigure",
    "ExtractedLink",
    "ExtractedTOCEntry",
    "ExtractedTable",
    "ExtractedTextSpan",
    "HeuristicClassifier",
    "LLMClassifier",
    "PageChrome",
    "RawBlock",
    "TEMPLATE_REGISTRY",
    "aconvert_pdftotext_to_html",
    "assemble_html",
    "convert_pdftotext_to_html",
    "default_classifier",
    "detect_formulas",
    "detect_page_chrome",
    "extract_document",
    "median_body_font_size",
    "render_block",
    "render_mathml",
    "segment_extracted_document",
    "segment_pdftotext_output",
    "strip_page_chrome",
]
