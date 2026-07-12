# Visual Review — the vision-sweep checklist

Repeatable procedure for eyeballing assembled end-user course pages and
catching render-time defects that no schema or gate can see. Pairs the
manual PNG review (Claude reads the screenshots) with the automated
`scripts/render_audit.py` browser check (GAP 3) so the human sweep only
has to adjudicate *look-and-feel*, not *is the math broken*.

This is the productionized version of the round-8 browser check: instead
of eyeballing one page in a live browser, we render a manifest of pages
across color schemes and scroll depths, run the automated audit, and Read
the PNGs against a fixed exemplar reference.

## When to run

- After any `semantik_conversion` / assembly change that touches HTML, CSS,
  MathJax config, or the accessibility template.
- Before signing off a converted corpus for Courseforge.
- Whenever a page "looks wrong" and you need a reproducible record.

## RAM guard (mandatory)

Both `scripts/shoot_pages.py` and `scripts/render_audit.py` refuse to
launch Chromium when `MemAvailable < 6 GB` or when an `ed4all run
textbook` synthesis is in flight (typesetting thousands of MathJax
equations can spike multiple GB and OOM the box). Check first:

```bash
awk '/MemAvailable/{print int($2/1024)" MB"}' /proc/meminfo
pgrep -f "ed4all run textbook" && echo "SYNTHESIS RUNNING — wait"
```

## Procedure

### 1. Re-render the pages

Regenerate the assembled end-user HTML from the current pipeline output
(the exact command depends on what you changed — conversion output lands
as `*_conv.html`, final-assembled pages as `*_final.html`). Stage the
pages you want to review under a working dir, e.g. `state/qa/visual/`.

### 2. Assemble the shoot manifest

Pick the review matrix — **chapters × schemes × scroll depths**. A good
default is 2-3 representative chapters (one math-heavy, one table-heavy,
one TOC/front-matter), both light + dark schemes, at several scroll
depths so callouts, tables, and the TOC all land in frame.
`scripts/shoot_pages.py` already sweeps light/dark and a fixed set of
scroll fractions per page:

```bash
cd state/qa/visual
python ../../../scripts/shoot_pages.py \
  ch02=ch02_conv.html ch09=ch09_conv.html ch04=ch04_conv.html
# → shots/<name>_<scheme>_<depth>.png
```

### 3. Run the automated render audit

Before reading a single pixel, let the browser check the mechanical
defects so the human sweep can focus on aesthetics:

```bash
python scripts/render_audit.py \
  ch02_conv=state/qa/visual/ch02_conv.html \
  ch09_conv=state/qa/visual/ch09_conv.html \
  --json-out state/qa/visual/render_audit.json
```

Per page it flags (non-zero exit on any failure):

| Finding | Severity | Meaning |
|---------|----------|---------|
| `MJX_MERROR` | failure | MathJax error node(s) — math failed to typeset. |
| `LITERAL_DELIMITERS` | failure | Visible `\(` `\)` `\[` `\]` `$$` outside math containers (un-typeset source). |
| `DUPLICATE_IDS` | failure | Repeated element ids (breaks anchors + a11y). |
| `MISSING_MAIN` | failure | Not exactly one `<main>` landmark. |
| `MISSING_SKIP_LINK` | failure | Not exactly one skip link. |
| `IMG_MISSING_ALT` | warning | Image(s) with no alt attribute. |

Fix any failures (or, if reviewing conversion quality, log them) before
proceeding — a page that fails the audit is not worth a human's eyes yet.

### 4. Read the PNGs against the exemplar reference

Read the shot PNGs and check each against the exemplar criteria below.
This is the part only a human/vision model can do — the audit already
guaranteed the math typeset and the landmarks are sound.

**Exemplar reference criteria** (a good page has all of these):

- [ ] **Dark background with a visible `<h1>`** — the page title reads
      clearly in dark scheme (no dark-on-dark, no invisible heading).
- [ ] **Boxed callouts** — callout/aside blocks render as distinct boxed
      cards (border/background), not as undifferentiated body text.
- [ ] **Tables render as tables** — bordered, aligned columns; no raw
      pipe-delimited text, no collapsed single-column mush.
- [ ] **TOC present and navigable** — a table of contents with working
      in-page anchors.
- [ ] **Currency is not italic** — `$` amounts render upright; a stray
      `$` did not open a MathJax math span and italicize the rest of the
      sentence. (This is the classic "one `$` ate the paragraph" defect;
      the audit's `LITERAL_DELIMITERS`/`MJX_MERROR` checks catch the
      worst cases, but subtle single-`$` italic runs still need eyes.)
- [ ] **No literal delimiters** — no visible `\(`, `\)`, `\[`, `\]`, or
      `$$` anywhere in the prose (belt-and-suspenders with the audit).

### 5. Record the verdict

Keep the `render_audit.json` and the shot manifest alongside the review
notes so the sweep is reproducible and diffable across runs.

## Related tooling

- `scripts/shoot_pages.py` — the screenshot shooter (light/dark ×
  scroll-depth sweep, RAM-guarded).
- `scripts/render_audit.py` — the automated DOM audit (GAP 3); pure
  scanners are unit-tested in `scripts/tests/test_render_audit.py`.
- `scripts/retrieval_smoke.py` — the retrieval-side smoke harness (GAP 2)
  for the "is the built course askable" question, orthogonal to the
  visual sweep.
