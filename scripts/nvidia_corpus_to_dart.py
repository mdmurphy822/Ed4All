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

Each emitted page mirrors the contract learned from
``DART/output/think_python_accessible_synthesized.json``:
    {slug, title, source_pdf, sections[], document_provenance, metadata}
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
from pathlib import Path

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


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "page"


# Scaffolding prefixes to strip from the FRONT of a heading, keeping any
# meaningful remainder. "Part 1: What Is LangChain?" -> "What Is LangChain?".
_SCAFFOLD_PREFIX = re.compile(
    r"^\s*(part|step|task|notebook|section|chapter|exercise|lesson|unit|module)\b"
    r"\s*\d*\s*[:.\)\-–]*\s*",
    re.IGNORECASE,
)
# Headings that are PURE scaffolding (nothing domain-bearing remains).
_PURE_GENERIC = {
    "questions to think about", "environment setup", "considering your models",
    "pros", "cons", "summary", "recap", "overview", "conclusion",
    "introduction", "assumption", "plan of action", "plan of attack",
    "considerations", "notes", "wrap-up", "wrap up", "takeaways",
    "key takeaways", "imports", "setup", "installation", "install",
    "appendix", "table of contents", "deeper dive", "advanced exercise",
    "timing solutions", "hint", "hints", "solution", "solutions", "answer",
    "answers", "your turn", "try it", "next steps", "references",
    "further reading", "learning objectives:",
}
_KEEP_HEADINGS = ("learning objectives",)


def _clean_heading(text: str) -> tuple[str, bool]:
    """Return (display_text, keep_as_heading).

    Strips a leading scaffolding token ("Part 3:", "Task 2", "Notebook 6")
    and decides whether the remainder is a real, concept-bearing heading.
    """
    # Strip any embedded presentational HTML tags from heading text, e.g. a
    # Markdown heading like `#### <font color=...>Great Job!</font>` — keep the
    # inner text ("Great Job!"), drop the markup so it never escapes to entities.
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    raw = re.sub(r"[*_`#]", "", text).strip()
    low = raw.lower().rstrip(":")
    if any(low == k or low.startswith(k) for k in _KEEP_HEADINGS):
        return raw, True
    stripped = _SCAFFOLD_PREFIX.sub("", raw).strip()
    s_low = stripped.lower().rstrip(":")
    # Nothing meaningful left, or remainder is itself pure scaffolding.
    if not stripped or s_low in _PURE_GENERIC or len(s_low) < 3:
        return raw, False
    if low in _PURE_GENERIC:
        return raw, False
    return stripped, True


# ---------------------------------------------------------------------------
# Minimal, dependency-free Markdown -> HTML (headings, code fences, lists,
# blockquotes, paragraphs, inline code/bold/italic/links). Heading levels are
# shifted down by one so the synthetic page <h1> is the sole chapter and all
# Markdown headings become <h2>+ sections.
# ---------------------------------------------------------------------------
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![\*])\*([^*\n]+)\*(?![\*])")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline(text: str) -> str:
    text = _html.escape(text, quote=False)
    text = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", text)
    text = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", text)
    text = _ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", text)
    text = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    return text


# Paired JSX component block: <Question ...> ... </Question>. The opening tag
# may span multiple lines (e.g. choices={[ {...}, {...} ]}). DOTALL so the
# whole block — opening tag, JS-object children, closing tag — is removed.
_MDX_PAIRED_BLOCK = re.compile(
    r"<([A-Z][A-Za-z0-9]*)\b[^>]*?>.*?</\1\s*>",
    re.DOTALL,
)
# Self-closing JSX component whose attributes may span multiple lines, e.g.
# <Question\n choices={[ ... ]}\n/>.
_MDX_SELF_CLOSING = re.compile(
    r"<[A-Z][A-Za-z0-9]*\b[^>]*?/>",
    re.DOTALL,
)


def _strip_mdx(md: str) -> str:
    """Drop MDX frontmatter, import/export lines, and JSX component blocks.

    Handles multi-line JSX: a ``<Question choices={[ ... ]}/>`` block whose
    opening tag spans many lines (and the paired ``<Tip>…</Tip>`` form) is
    removed wholesale so its raw JS object literal never leaks into chunk text.
    """
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            md = md[end + 4:]
    # Remove paired + self-closing capitalized JSX component blocks first,
    # across line boundaries, before the per-line scan handles bare tags.
    md = _MDX_PAIRED_BLOCK.sub("", md)
    md = _MDX_SELF_CLOSING.sub("", md)
    out_lines = []
    for ln in md.splitlines():
        st = ln.strip()
        if st.startswith(("import ", "export ")):
            continue
        # Drop any remaining standalone JSX tags like <Tip>, </Tip>, <Quiz />
        if re.fullmatch(r"</?[A-Z][A-Za-z0-9]*(\s[^>]*)?/?>", st):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)


# A line that opens / is / continues a raw embedded-HTML block. NVIDIA
# notebook markdown cells paste a presentational logo header straight in:
#   <br> <a href=...><div style=...><img src=.../nvidia-logo.png/></div></a>
#   <h1 style=...><font color=...><b>Building RAG Agents with LLMs</b></font></h1>
#   <h2><b>Notebook 3: </b>LangChain Expression Language</h2>
# These are real tags, not Markdown, so escaping them as literal text leaks
# `&lt;br&gt;`/`&lt;div`/`&lt;font` into chunk bodies. Detect & sanitize via bs4.
_RAW_HTML_TAG = re.compile(
    r"<\s*/?\s*(br|div|img|font|a|span|center|h[1-6]|table|tr|td|th|thead|tbody)"
    r"(\s|>|/)",
    re.IGNORECASE,
)


def _looks_like_raw_html(text: str) -> bool:
    """True when a paragraph blob is embedded presentational/structural HTML."""
    return bool(_RAW_HTML_TAG.search(text))


def _sanitize_raw_html_block(raw: str) -> str:
    """Turn an embedded raw-HTML blob into clean DART content.

    Drops presentational wrappers (``<div style>``, ``<img>``, ``<font>``,
    ``<a>`` logo links, inline ``style=``) but PRESERVES real heading text.
    ``<h1>Building RAG Agents with LLMs</h1>`` and
    ``<h2>Notebook 3: LangChain Expression Language</h2>`` survive as clean
    section headings (run through ``_clean_heading`` like Markdown headings),
    and any remaining prose becomes a paragraph. Returns rendered HTML with
    zero presentational markup and zero escaped entities.
    """
    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(raw, "html.parser")
    # Strip logo/image/link chrome outright — keep only their text (usually none).
    for tag in soup(["img", "a", "font", "br", "center"]):
        tag.unwrap() if tag.name in ("font", "a", "center") else tag.decompose()

    out: list[str] = []
    leftover: list[str] = []

    def flush_leftover():
        if leftover:
            txt = " ".join(t for t in (s.strip() for s in leftover) if t)
            leftover.clear()
            if txt:
                out.append(f"<p>{_inline(txt)}</p>")

    # Walk top-level nodes; headings become clean <h2>+ sections, rest is prose.
    for el in soup.children:
        name = getattr(el, "name", None)
        if name and re.fullmatch(r"h[1-6]", name):
            flush_leftover()
            htext = el.get_text(" ", strip=True)
            display, keep = _clean_heading(htext)
            if keep and display:
                out.append(f"<h2>{_inline(display)}</h2>")
            elif display:
                leftover.append(display)
        else:
            txt = el.get_text(" ", strip=True) if name else str(el).strip()
            if txt:
                leftover.append(txt)
    flush_leftover()
    return "\n".join(out)


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def md_to_html(md: str) -> str:
    md = md.replace("\r\n", "\n")
    # Strip HTML comments outright — NVIDIA cells park alternate <img> markup
    # inside <!-- ... --> blocks that would otherwise leak as escaped text.
    md = _HTML_COMMENT.sub("", md)
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while i < len(lines):
        line = lines[i]
        # fenced code
        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            close_list()
            lang = m.group(1)
            buf = []
            i += 1
            while i < len(lines) and not re.match(r"^```\s*$", lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = _html.escape("\n".join(buf), quote=False)
            cls = f' class="language-{lang}"' if lang else ""
            out.append(f"<pre><code{cls}>{code}</code></pre>")
            continue
        # heading (shift down one level, cap at h6)
        hm = re.match(r"^(#{1,6})\s+(.*)$", line)
        if hm:
            close_list()
            htext = hm.group(2).strip()
            display, keep = _clean_heading(htext)
            if keep:
                lvl = min(len(hm.group(1)) + 1, 6)
                out.append(f"<h{lvl}>{_inline(display)}</h{lvl}>")
            # else: pure-scaffolding heading ("Questions To Think About",
            # "Environment Setup", "Pros"…) — DROP the heading line entirely
            # so the concept-graph builder never harvests it as a node. The
            # content *under* the heading still flows through as paragraphs.
            i += 1
            continue
        # table row -> flatten to paragraph text (keeps prose for chunking)
        if "|" in line and line.strip().startswith("|"):
            if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
                i += 1
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            close_list()
            out.append(f"<p>{_inline(' — '.join(c for c in cells if c))}</p>")
            i += 1
            continue
        # blockquote
        if line.startswith(">"):
            close_list()
            qtext = line.lstrip("> ").strip()
            # A blockquote whose body is embedded raw HTML (e.g. `> <img .../>`)
            # must be sanitized, not escaped, or it leaks `&lt;img` into chunks.
            if _looks_like_raw_html(qtext):
                sanitized = _sanitize_raw_html_block(qtext)
                if sanitized:
                    out.append(f"<blockquote>{sanitized}</blockquote>")
            else:
                out.append(f"<blockquote><p>{_inline(qtext)}</p></blockquote>")
            i += 1
            continue
        # list item
        lm = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if lm:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(lm.group(1).strip())}</li>")
            i += 1
            continue
        om = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if om:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(om.group(1).strip())}</li>")
            i += 1
            continue
        # blank
        if not line.strip():
            close_list()
            i += 1
            continue
        # paragraph (accumulate consecutive non-blank, non-special lines)
        close_list()
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,6}\s|```|\s*[-*+]\s|\s*\d+\.\s|>|\|)", lines[i]
        ):
            para.append(lines[i].strip())
            i += 1
        para_text = " ".join(para)
        # Embedded raw HTML (e.g. the NVIDIA logo header) — sanitize via bs4
        # instead of escaping it to literal `&lt;...&gt;` text. Preserves real
        # heading text, drops presentational wrappers.
        if _looks_like_raw_html(para_text):
            sanitized = _sanitize_raw_html_block(para_text)
            if sanitized:
                out.append(sanitized)
            continue
        # Drop standalone pure-scaffolding bold labels ("**Questions To
        # Think About:**", "**Environment Setup:**") that the NVIDIA
        # notebooks use in lieu of headings — else the concept-graph
        # builder harvests them as high-frequency noise nodes.
        bare = re.sub(r"[*_`:]", "", para_text).strip().lower()
        if bare in _PURE_GENERIC or bare.rstrip(":") in _PURE_GENERIC:
            continue
        out.append(f"<p>{_inline(para_text)}</p>")
    close_list()
    return "\n".join(out)


def ipynb_to_html(nb_json: dict) -> str:
    out: list[str] = []
    for cell in nb_json.get("cells", []):
        ctype = cell.get("cell_type")
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = src.strip()
        if not src:
            continue
        if ctype == "markdown":
            out.append(md_to_html(src))
        elif ctype == "code":
            code = _html.escape(src, quote=False)
            out.append(f'<pre><code class="language-python">{code}</code></pre>')
            # keep short text outputs (stream/text), drop images & long dumps
            for o in cell.get("outputs", []) or []:
                txt = ""
                if o.get("output_type") == "stream":
                    txt = o.get("text", "")
                elif o.get("output_type") in ("execute_result", "display_data"):
                    data = o.get("data", {})
                    txt = data.get("text/plain", "")
                if isinstance(txt, list):
                    txt = "".join(txt)
                txt = txt.strip()
                if txt and len(txt) < 600:
                    out.append(f"<pre class=\"output\"><code>{_html.escape(txt, quote=False)}</code></pre>")
    return "\n".join(out)


def nbconvert_html_body(raw: str) -> str:
    """Extract the rendered body from an nbconvert HTML export (bs4)."""
    from bs4 import BeautifulSoup  # type: ignore
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    body = soup.body or soup
    # Promote markdown headings already present; strip jupyter chrome divs
    return body.decode_contents()


def build_sections(body_html: str) -> list[dict]:
    """Slice the body HTML into sections at <h2> boundaries for the sidecar."""
    from bs4 import BeautifulSoup  # type: ignore
    soup = BeautifulSoup(body_html, "html.parser")
    sections: list[dict] = []
    cur_title = None
    cur_buf: list[str] = []
    sid = 0

    def flush():
        nonlocal sid, cur_title, cur_buf
        text = BeautifulSoup("".join(cur_buf), "html.parser").get_text(" ", strip=True)
        if not text.strip():
            return
        sid += 1
        sections.append({
            "section_id": f"s{sid}",
            "section_title": cur_title or "Overview",
            "section_type": "section",
            "page_range": [sid, sid],
            "provenance": {"sources": ["nvidia-course-material"],
                           "strategy": "heuristic", "confidence": 1.0},
            "data": {"text": text[:20000]},
        })
        cur_buf = []

    for el in soup.children:
        name = getattr(el, "name", None)
        if name in ("h2", "h3"):
            flush()
            cur_title = el.get_text(" ", strip=True)
            cur_buf = []
        else:
            cur_buf.append(str(el))
    flush()
    if not sections:
        text = soup.get_text(" ", strip=True)
        sections.append({
            "section_id": "s1", "section_title": "Overview",
            "section_type": "section", "page_range": [1, 1],
            "provenance": {"sources": ["nvidia-course-material"],
                           "strategy": "heuristic", "confidence": 1.0},
            "data": {"text": text[:20000]},
        })
    return sections


# Escaped tag entities that must NEVER appear inside an emitted <p> body —
# their presence means raw HTML/MDX leaked through _inline()'s escape().
_LEAK_MARKERS = ("&lt;br&gt;", "&lt;br/&gt;", "&lt;div", "&lt;font",
                 "&lt;img", "&lt;Question", "nvidia-logo.png")
_P_BODY = re.compile(r"<p>(.*?)</p>", re.DOTALL)


def _scan_leaks(html: str) -> list[str]:
    """Return the leak markers found inside any <p> body of ``html``."""
    found: set[str] = set()
    for body in _P_BODY.findall(html):
        for mark in _LEAK_MARKERS:
            if mark in body:
                found.add(mark)
    return sorted(found)


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

        page_leaks = _scan_leaks(page_html)
        if page_leaks:
            leaks.append((slug, page_leaks))

        sections = build_sections(body)
        sidecar = {
            "slug": slug.replace("_", "-"),
            "title": title,
            "source_pdf": str(rel),
            "sections": sections,
            "document_provenance": {
                "extractors_used": ["nvidia-course-material"],
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
