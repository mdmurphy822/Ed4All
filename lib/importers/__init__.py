"""Deterministic corpus importers → clean accessible HTML.

Importers here turn an already-digital source corpus (Markdown / MDX / docs
site) into a directory of clean, semantic ``*.html`` pages that the
``semantik_conversion`` phase ingests first-class through its vendor-ingest seam
(``_detect_conversion_input_type`` → ``_run_vendor_ingest_conversion``). No
PDF/OCR step, no LLM, no forged provenance sidecars: the pipeline re-normalizes
the HTML into the canonical ``{stem}_accessible.html`` contract itself.

``_markdown`` holds the shared, brand-neutral Markdown/MDX/notebook → HTML
helpers (heading hygiene, fenced code, embedded-HTML sanitation, section
slicing). ``docs_corpus`` is the docs-site importer built on top of them.
"""

from .docs_corpus import DocsImportResult, import_docs_corpus

__all__ = ["import_docs_corpus", "DocsImportResult"]
