"""Build ``fixture_corpus.pdf`` — a tiny hand-crafted PDF for the end-to-end
pipeline integration test.

The fixture is a 3-page PDF about **photosynthesis basics**. Topic chosen to
be generic and non-overlapping with any Ed4All research corpus. The PDF is
laid out so ``pdftotext`` (used by DART) extracts clean paragraph text with
the following structure:

Page 1 — "Introduction to Photosynthesis"
Page 2 — "The Two Stages of Photosynthesis"
Page 3 — "Common Misconceptions"

Content is sized for ≥ 10 chunks after Trainforge chunking.

This script is idempotent: re-running regenerates the same bytes. The output
PDF is committed alongside the builder so CI doesn't need to run this — but
the builder is kept for transparency (and to let devs tweak the fixture).

Why hand-crafted PDF rather than ``reportlab``?
- ``reportlab`` is not in ``requirements.txt``.
- ``PyMuPDF`` / ``fitz`` not present in the dev sandbox.
- ``pdftotext`` is guaranteed (it's a hard dep of DART's
  ``_extract_and_convert_pdf`` at ``MCP/tools/pipeline_tools.py:1053``).

The PDF uses the standard ``Helvetica`` font and a single content stream per
page. Text is tokenized into one ``Tj`` operator per line with ``T*`` for
line breaks. The object graph is the minimum valid PDF 1.4 shape.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------- #
# Page content (plain text — one list per page)
# ---------------------------------------------------------------------- #

PAGE_1_LINES = [
    "Introduction to Photosynthesis",
    "",
    "Photosynthesis is the biological process by which plants,",
    "algae, and some bacteria convert light energy into chemical",
    "energy stored as glucose. This fundamental process sustains",
    "nearly all life on Earth by producing the oxygen we breathe",
    "and forming the base of most food webs.",
    "",
    "The overall chemical equation for photosynthesis can be",
    "written as: 6 CO2 plus 6 H2O plus light energy yields",
    "glucose (C6H12O6) plus 6 O2. Carbon dioxide and water are",
    "the raw inputs; glucose and oxygen are the products.",
    "",
    "Photosynthesis occurs primarily in chloroplasts, specialized",
    "organelles found in the cells of plant leaves. Chloroplasts",
    "contain chlorophyll, a green pigment that absorbs light",
    "energy most effectively in the red and blue portions of the",
    "visible spectrum.",
    "",
    "Understanding photosynthesis is essential for biology",
    "students because it explains energy flow through ecosystems,",
    "the oxygen cycle, and the basis of agricultural productivity.",
]

PAGE_2_LINES = [
    "The Two Stages of Photosynthesis",
    "",
    "Photosynthesis proceeds in two interconnected stages: the",
    "light-dependent reactions and the Calvin cycle, also known",
    "as the light-independent reactions.",
    "",
    "The light-dependent reactions occur in the thylakoid",
    "membranes of the chloroplast. Here, chlorophyll absorbs",
    "photons, which excites electrons that pass through an",
    "electron transport chain. This process splits water",
    "molecules, releases oxygen as a byproduct, and generates",
    "the energy-carrying molecules ATP and NADPH.",
    "",
    "The Calvin cycle takes place in the stroma, the fluid-filled",
    "space surrounding the thylakoids. In this stage, the ATP and",
    "NADPH produced earlier are used to fix carbon dioxide into",
    "glucose. The enzyme RuBisCO catalyzes the first step of",
    "carbon fixation.",
    "",
    "The two stages are tightly coupled: the light reactions",
    "supply chemical energy, and the Calvin cycle consumes it to",
    "build organic molecules. Neither stage can proceed without",
    "the other, though only the first stage requires light",
    "directly.",
]

PAGE_3_LINES = [
    "Common Misconceptions",
    "",
    "Several widespread misconceptions complicate student",
    "understanding of photosynthesis. This section addresses three",
    "of the most common.",
    "",
    "Misconception: Plants get their food from the soil.",
    "Correction: Plants produce their own food through",
    "photosynthesis. Soil provides water and mineral nutrients",
    "but not the carbon that makes up plant biomass; that carbon",
    "comes from atmospheric carbon dioxide.",
    "",
    "Misconception: Plants only photosynthesize during the day.",
    "Correction: The light-dependent reactions require light, but",
    "the Calvin cycle can continue for a short time in darkness",
    "using the ATP and NADPH already produced. Plants also",
    "respire continuously, using oxygen and releasing carbon",
    "dioxide around the clock.",
    "",
    "Misconception: Green plants absorb green light.",
    "Correction: Plants appear green because they reflect green",
    "light. Chlorophyll absorbs most strongly in the red and",
    "blue ranges and poorly in green, which is why the reflected",
    "green light reaches our eyes.",
]

PAGES = [PAGE_1_LINES, PAGE_2_LINES, PAGE_3_LINES]


# ---------------------------------------------------------------------- #
# PDF builder
# ---------------------------------------------------------------------- #


def _escape_pdf_text(text: str) -> str:
    """Escape parentheses and backslashes for a PDF string literal."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_content_stream(lines: list[str]) -> bytes:
    """Return a PDF content stream that lays out ``lines`` as a single
    text block.

    Layout: Helvetica 12pt, starting at (72, 720) with 16pt line height.
    That fits ~36 lines on a letter-sized page (612 x 792). All our pages
    are ~22 lines, comfortably within budget.
    """
    out = ["BT", "/F1 12 Tf", "16 TL", "72 720 Td"]
    for line in lines:
        if line:
            out.append(f"({_escape_pdf_text(line)}) Tj T*")
        else:
            out.append("() Tj T*")
    out.append("ET")
    return ("\n".join(out)).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal PDF 1.4 document from a list of page line-lists."""
    # Object layout:
    #   1: catalog
    #   2: pages root
    #   3: font (Helvetica)
    #   4..N: page objects (one per page)
    #   N+1..2N: content streams (one per page)
    n_pages = len(pages)
    page_obj_ids = [4 + i for i in range(n_pages)]
    stream_obj_ids = [4 + n_pages + i for i in range(n_pages)]

    objects: dict[int, bytes] = {}

    # 1: catalog
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    # 2: pages root
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[2] = (
        f"<< /Type /Pages /Count {n_pages} /Kids [{kids}] "
        f"/MediaBox [0 0 612 792] >>"
    ).encode("latin-1")

    # 3: font
    objects[3] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding "
        b"/WinAnsiEncoding >>"
    )

    # 4..N: page objects
    for i, (pid, sid) in enumerate(zip(page_obj_ids, stream_obj_ids)):
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /Resources "
            f"<< /Font << /F1 3 0 R >> >> /Contents {sid} 0 R >>"
        ).encode("latin-1")

    # N+1..2N: content streams
    for sid, lines in zip(stream_obj_ids, pages):
        stream_bytes = _build_content_stream(lines)
        header = f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
        footer = b"\nendstream"
        objects[sid] = header + stream_bytes + footer

    # Assemble PDF
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode("latin-1")
        out += objects[obj_id]
        out += b"\nendobj\n"

    # xref
    xref_offset = len(out)
    n_objs = max(offsets)
    out += f"xref\n0 {n_objs + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, n_objs + 1):
        offset = offsets.get(obj_id, 0)
        out += f"{offset:010d} 00000 n \n".encode("latin-1")

    # trailer
    out += (
        f"trailer\n<< /Size {n_objs + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")

    return bytes(out)


def main() -> Path:
    pdf_bytes = build_pdf(PAGES)
    target = Path(__file__).resolve().parent / "fixture_corpus.pdf"
    target.write_bytes(pdf_bytes)
    print(f"Wrote {target} ({len(pdf_bytes)} bytes)")
    return target


if __name__ == "__main__":
    main()
