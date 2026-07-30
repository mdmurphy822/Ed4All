# `ed4all convert` — the accessible-HTML remediation slice

`ed4all convert` is the thin, standalone conversion verb: it turns a PDF (or a
directory of PDFs, or a directory of publisher HTML) into the canonical
accessible-HTML contract and writes it into a directory you choose. Nothing
else — no course, no run directory, no library archive.

```bash
ed4all convert INPUT --output DIR [--doc-title TITLE] \
    [--figures-dir DIR] [--reuse-conversion]
```

## What it does

* **Detects the input type** using the same contract the full pipeline uses
  (`_detect_conversion_input_type`):
  * a `.pdf` file, or a directory containing PDFs → the **SemantiK cascade**
    seam, emitting one accessible-HTML document per PDF;
  * a `.html`/`.htm` file, or a directory of publisher HTML pages (with no PDF
    present) → the **vendor-ingest** seam, which assembles the directory into a
    single accessible-HTML document;
  * anything else → a fail-closed-clear error.
  A PDF present anywhere in a mixed directory wins (the cascade is the
  authoritative converter; vendor-ingest is the already-accessible fast path).
* **Writes the standard conversion contract** into `--output`: for each
  converted unit, `{stem}_accessible.html` plus its two sidecars
  (`{stem}_accessible_synthesized.json` and `{stem}_accessible.quality.json`).
  The stem is the PDF stem, or the directory name for a vendor HTML directory.
* **Prints a concise summary**: how many inputs converted, each output path,
  and any per-file failures with the seam's own reason.

## What it does NOT do

* **No course.** There is no `--course-name`, no learning objectives, no
  modules, no IMSCC packaging.
* **No run directory / run id.** It does not create workflow run state under
  `runtime/state/runs/`.
* **No LibV2 writes and no vector index.** The output lands only in the
  directory you pass to `--output`; nothing is archived or indexed. To build a
  course from converted HTML, feed the output directory to the full pipeline:

  ```bash
  ed4all run textbook-to-course --corpus DIR --course-name NAME
  ```

## Where output lands

Everything is written under the `--output` directory you provide (created if it
does not exist). For a directory of PDFs, one `{stem}_accessible.html` +
sidecar set is written per PDF (PDFs that share a stem across sub-directories
will collide on filename — the later one wins). For a vendor HTML directory,
one `{dirname}_accessible.html` + sidecar set is written.

## Exit codes

| Code | Meaning |
|-----:|---------|
| `0`  | Every input converted successfully. |
| `1`  | Total failure — unrecognized input, no PDFs found, or every unit failed. |
| `2`  | Partial success — some units converted, some failed (per-file failures are printed). |

## Graceful stop

The SemantiK cascade seam polls the run-scoped and **global** stop sentinels at
its seam boundaries, and `convert` always publishes the global sentinel path to
the cascade even though it mints no run id. So `ed4all stop --all` reaches an
in-flight PDF conversion and lets it stop at a seam boundary; clear it with
`ed4all stop --clear-all`.

The vendor-ingest seam is a fast, deterministic single pass and does not poll
for a stop sentinel — interrupt it with Ctrl-C.
