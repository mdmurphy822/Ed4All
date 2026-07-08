#!/usr/bin/env python3
"""Convert the NVIDIA "Building RAG Agents" study materials into DART-style
``*_accessible.html`` pages + ``*_accessible_synthesized.json`` provenance
sidecars, so ``ed4all run textbook-to-course --skip-dart`` can ingest them.

The source materials are already clean digital text (Markdown, MDX, Jupyter
notebooks, one nbconvert HTML), so there is no PDF / OCR step to run — DART's
deterministic pdftotext/pymupdf extraction is exactly what we reproduce here:
one ``<h1>`` chapter per source file, ``<h2>/<h3>`` sections, prose paragraphs
and fenced code preserved. ``SemanticStructureExtractor`` keys off the heading
hierarchy, and the chunker keys off the rendered HTML body.

The brand-neutral Markdown / MDX / notebook → HTML machinery lives in the
shared :mod:`lib.importers._markdown` helpers; this script is only the
NVIDIA-specific SOURCES spine + the forged-sidecar writer. Each emitted page
mirrors the conversion-output synthesized-sidecar contract:
    {slug, title, source_pdf, sections[], document_provenance, metadata}

For a general docs tree, prefer ``ed4all import-docs`` (no forged sidecars).
"""
from __future__ import annotations

import argparse
import html as _html
import json
import sys
from pathlib import Path

# Standalone invocation (`python scripts/nvidia_corpus_to_dart.py`) needs the
# repo root on sys.path to resolve the shared importer helpers.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse the shared, brand-neutral importer helpers (moved out of this script).
from lib.importers._markdown import (  # noqa: E402
    build_sections,
    ipynb_to_html,
    md_to_html,
    nbconvert_html_body,
    scan_leaks,
    slugify,
    _strip_mdx,
)

# Provenance label + corpus-specific leak marker passed to the shared helpers
# so the reusable code stays brand-neutral.
_PROVENANCE_SOURCE = "nvidia-course-material"
_EXTRA_LEAK_MARKERS = ("nvidia-logo.png",)

# The source corpus directory is machine-specific and must be passed via
# --src; output defaults into the repo's DART output tree.
_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "DART" / "output" / "nvidia"

# (relative source path, human title). Order = reading order.
#
# Spine = the actual gated NVIDIA DLI "Building RAG Agents with LLMs" course
# notebooks (notebooks.zip). The open-source study packs / mdx lessons /
# downloaded cookbook notebooks are supplementary alternative framings that
# enrich the concept graph.
_CN = "learn/materials/course_notebooks/notebooks"
SOURCES = [
    # ---- Primary: the real NVIDIA course notebooks (00–09, 64, 99) ----
    (f"{_CN}/00_jupyterlab.ipynb",
     "NVIDIA Course 00: The JupyterLab Environment"),
    (f"{_CN}/01_microservices.ipynb",
     "NVIDIA Course 01: Microservices and the Course Architecture"),
    (f"{_CN}/02_llms.ipynb",
     "NVIDIA Course 02: LLM Services and NIM Endpoints"),
    (f"{_CN}/03_langchain_intro.ipynb",
     "NVIDIA Course 03: Introduction to LangChain — Chains and Runnables"),
    (f"{_CN}/04_running_state.ipynb",
     "NVIDIA Course 04: Running State Chains and Conversation Memory"),
    (f"{_CN}/05_documents.ipynb",
     "NVIDIA Course 05: Working with Documents and Long-Form Context"),
    (f"{_CN}/06_embeddings.ipynb",
     "NVIDIA Course 06: Embeddings for Semantic Retrieval"),
    (f"{_CN}/07_vectorstores.ipynb",
     "NVIDIA Course 07: Vector Stores and RAG for Conversation History"),
    (f"{_CN}/08_evaluation.ipynb",
     "NVIDIA Course 08: Evaluating RAG Pipelines (RAGAS)"),
    (f"{_CN}/09_langserve.ipynb",
     "NVIDIA Course 09: Serving RAG Agents with LangServe"),
    (f"{_CN}/64_guardrails.ipynb",
     "NVIDIA Course 64: Guardrails and Semantic Filtering"),
    (f"{_CN}/99_table_of_contents.ipynb",
     "NVIDIA Course: Table of Contents and Course Map"),
    # ---- Primary supplement: exercise solutions ----
    (f"{_CN}/solutions/03_solutions.ipynb", "NVIDIA Course Solutions 03: LangChain"),
    (f"{_CN}/solutions/04_solutions.ipynb", "NVIDIA Course Solutions 04: Running State"),
    (f"{_CN}/solutions/05_solution.ipynb", "NVIDIA Course Solutions 05: Documents"),
    (f"{_CN}/solutions/06_solutions.ipynb", "NVIDIA Course Solutions 06: Embeddings"),
    (f"{_CN}/solutions/07_solutions.ipynb", "NVIDIA Course Solutions 07: Vector Stores"),
    (f"{_CN}/solutions/08_solutions.ipynb", "NVIDIA Course Solutions 08: Evaluation"),
    (f"{_CN}/solutions/64_solutions.ipynb", "NVIDIA Course Solutions 64: Guardrails"),
    # ---- Supplementary study packs ----
    ("learn/materials/01_embeddings_and_vectorstores.md",
     "Embeddings and Vector Stores"),
    ("learn/materials/02_langchain_vocabulary.md",
     "LangChain Vocabulary: Runnables, LCEL, Retrievers"),
    ("learn/materials/03_langgraph_agents.md",
     "LangGraph Agents"),
    ("learn/downloaded/03_langgraph_agents/course_lessons/introduction.mdx",
     "Introduction to LangGraph"),
    ("learn/downloaded/03_langgraph_agents/course_lessons/when_to_use_langgraph.mdx",
     "When to Use LangGraph"),
    ("learn/downloaded/03_langgraph_agents/course_lessons/building_blocks.mdx",
     "LangGraph Building Blocks"),
    ("learn/downloaded/03_langgraph_agents/course_lessons/first_graph.mdx",
     "Building Your First Graph (StateGraph from Scratch)"),
    ("learn/downloaded/03_langgraph_agents/course_lessons/document_analysis_agent.mdx",
     "A Document Analysis Agent"),
    ("learn/downloaded/03_langgraph_agents/course_lessons/quiz1.mdx",
     "LangGraph Knowledge Check"),
    ("learn/downloaded/03_langgraph_agents/course_lessons/conclusion.mdx",
     "LangGraph Conclusion"),
    ("learn/downloaded/03_langgraph_agents/langgraph_agentic_rag.ipynb",
     "Agentic RAG with LangGraph (Retrieval as a Tool)"),
    ("learn/downloaded/03_langgraph_agents/langgraph_self_rag.ipynb",
     "Self-RAG: Grading Your Own Retrieval"),
    ("learn/downloaded/03_langgraph_agents/langgraph_crag.ipynb",
     "Corrective RAG (CRAG)"),
    ("learn/downloaded/03_langgraph_agents/agent_rag.ipynb",
     "Agentic RAG (Alternate Framing)"),
    ("learn/downloaded/03_langgraph_agents/multiagent_rag_system.ipynb",
     "Multi-Agent RAG Systems"),
    ("learn/downloaded/03_langgraph_agents/agents.ipynb",
     "Agents Fundamentals"),
    ("learn/downloaded/01_embeddings_and_rag/advanced_rag.ipynb",
     "Advanced RAG: Chunking, Embeddings, FAISS, Reranking"),
    ("learn/downloaded/01_embeddings_and_rag/rag_evaluation.ipynb",
     "Evaluating RAG Quality"),
    ("learn/downloaded/01_embeddings_and_rag/rag_with_unstructured_data.ipynb",
     "RAG over Unstructured Data"),
    ("learn/downloaded/01_embeddings_and_rag/semantic_reranking_elasticsearch.ipynb",
     "Semantic Reranking with Elasticsearch"),
    ("learn/downloaded/01_embeddings_and_rag/vector_search_with_hub_as_backend.ipynb",
     "Vector Search Mechanics"),
    ("learn/downloaded/02_langchain_in_practice/rag_zephyr_langchain.ipynb",
     "A Full LangChain RAG Pipeline (Zephyr)"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Root directory of the NVIDIA study-materials corpus",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output directory for the DART-style pages (default: DART/output/nvidia)",
    )
    args = parser.parse_args()
    src: Path = args.src
    out: Path = args.out

    out.mkdir(parents=True, exist_ok=True)
    written = []
    leaks: list[tuple[str, list[str]]] = []
    for rel, title in SOURCES:
        path = src / rel
        if not path.exists():
            print(f"  SKIP (missing): {rel}")
            continue
        ext = path.suffix.lower()
        if ext == ".ipynb":
            body = ipynb_to_html(json.loads(path.read_text(encoding="utf-8")))
        elif ext in (".md", ".mdx"):
            raw = path.read_text(encoding="utf-8")
            if ext == ".mdx":
                raw = _strip_mdx(raw)
            body = md_to_html(raw)
        elif ext in (".html", ".htm"):
            body = nbconvert_html_body(path.read_text(encoding="utf-8"))
        else:
            print(f"  SKIP (unsupported): {rel}")
            continue

        slug = slugify(title)
        page_html = (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\"/>\n"
            f"<title>{_html.escape(title)}</title>\n</head>\n<body>\n"
            f"<h1>{_html.escape(title)}</h1>\n{body}\n</body>\n</html>\n"
        )
        html_path = out / f"{slug}_accessible.html"
        html_path.write_text(page_html, encoding="utf-8")

        page_leaks = scan_leaks(page_html, extra_markers=_EXTRA_LEAK_MARKERS)
        if page_leaks:
            leaks.append((slug, page_leaks))

        sections = build_sections(body, provenance_source=_PROVENANCE_SOURCE)
        sidecar = {
            "slug": slug.replace("_", "-"),
            "title": title,
            "source_pdf": str(rel),
            "sections": sections,
            "document_provenance": {
                "extractors_used": [_PROVENANCE_SOURCE],
                "figures_extracted": 0,
                "tables_extracted": 0,
                "toc_entries": len(sections),
            },
            "metadata": {
                "source": "NVIDIA DLI - Building RAG Agents with LLMs (study corpus)",
                "license_note": "Open-source study materials; see learn/downloaded/SOURCES.md",
                "original_format": ext.lstrip("."),
            },
        }
        (out / f"{slug}_accessible_synthesized.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8")
        written.append((slug, len(sections), len(page_html)))
        print(f"  OK  {slug:48} sections={len(sections):3} html={len(page_html):>7}b")

    print(f"\nWrote {len(written)} pages to {out}")

    # Regression guard: no emitted page may carry escaped raw-HTML / JSX
    # markers inside a <p> body (would surface verbatim in chunk text).
    if leaks:
        print("\nLEAK CHECK FAILED — escaped markup inside <p> bodies:")
        for slug, marks in leaks:
            print(f"  {slug}: {', '.join(marks)}")
        raise SystemExit(1)
    print("LEAK CHECK OK — no escaped raw-HTML/JSX in any <p> body.")


if __name__ == "__main__":
    main()
