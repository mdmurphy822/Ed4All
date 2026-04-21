# DART

**Convert PDF documents into accessible, semantic HTML.**

DART (Digital Accessibility Remediation Tool) takes a PDF and produces HTML that meets WCAG 2.2 AA: proper heading hierarchy, alt text for images, ARIA landmarks, keyboard skip links, accessible tables, MathML for equations, and per-block source attribution that ties every claim back to its page in the original PDF. It combines text-layer extraction, layout analysis, optional OCR, and optional LLM classification into one multi-source pipeline — pdftotext is the only hard dependency, everything else degrades gracefully.

## Quick example

```bash
# From the repo root, with Ed4All installed (pip install -e ".[full]")
python DART/convert.py path/to/document.pdf -o ./accessible_output/
```

DART is also wired into the Ed4All pipeline. Running `ed4all run textbook-to-course` invokes it automatically as the first stage, so most users never call it directly.

## Optional system dependencies

For best results on scanned PDFs and image-heavy documents:

```bash
# Ubuntu / Debian
sudo apt install poppler-utils tesseract-ocr

# macOS
brew install poppler tesseract
```

## More

See [`DART/CLAUDE.md`](CLAUDE.md) for the architecture, extractor reconciliation rules, source-provenance contract, and WCAG 2.2 AA compliance matrix.

## License

MIT
