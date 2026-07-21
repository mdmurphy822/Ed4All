# Building the bundled demo course (B3)

The demo course is a small, license-clean, end-to-end example course minted from
the tracked pipeline fixture PDF. It exists so a fresh checkout (or a
prospective user) can see a real Ed4All course — chunks, objectives, IMSCC,
manifest, `NOTICE` — without supplying their own textbook. This doc is the
mint → freeze → ship runbook.

## What it is built from

The single source is the tracked fixture at
`tests/fixtures/pipeline/fixture_corpus.pdf` — a hand-built 3-page PDF about
**photosynthesis basics** (see `tests/fixtures/pipeline/build_fixture_pdf.py`).
The content is original synthetic prose authored for this repo, so it carries
**no upstream rights holder**: the demo ships under a `CC0-1.0` public-domain
dedication with a plain attribution back to the fixture builder.

Because the content is original + license-clean, the demo bundle is safe to
commit and distribute (unlike any real-textbook corpus, which stays gitignored).

## Minting the bundle

The pinned, documented invocation is `scripts/build_demo_course.py`. It wraps
`ed4all run textbook-to-course` with the fixed course name + license +
attribution, refuses to run when the fixture is missing, and refuses a `fake`
embedding provider in `--full` mode.

```bash
# Show the exact ed4all command without running it:
python scripts/build_demo_course.py --full --print-only

# Retrieval-ready slice (fast; stops after imscc_chunking):
#   NOTE: this slice does NOT emit the license/attribution/NOTICE — those land
#   at the libv2_archival phase, which this slice skips.
python scripts/build_demo_course.py

# Full shippable bundle (manifest license + attribution + NOTICE + real
# vector index):
python scripts/build_demo_course.py --full
```

The full run mints the course under `LibV2/courses/demo-photosynthesis/` with:

* `manifest.json` carrying the B3 `license` (`{"spdx_or_name": "CC0-1.0"}`) and
  `attribution` (`{"statement": "..."}`) blocks plus the OP4
  `library_format_version` stamp;
* a human-readable `NOTICE` file (generated from the manifest license +
  attribution) to redistribute with the bundle;
* the standard course scaffold (`semantik_chunks/`, `imscc_chunks/`,
  `concept_graph/`, `course.json`, `vector_index/`, …).

### Flags the builder pins

| Flag | Value | Why |
|------|-------|-----|
| `--course-name` | `demo-photosynthesis` | Deterministic slug `demo-photosynthesis`. |
| `--license-note` | `CC0-1.0` | Public-domain dedication → `license.spdx_or_name`. |
| `--attribution` | fixture attribution statement | → `attribution.statement`; mirrored into `NOTICE`. |
| `--skip-training` + `--stop-after imscc_chunking` | default (non-`--full`) mode only | Fast retrieval-ready slice. |

## Embedding-provider pinning caveat

A shipped bundle MUST carry a **real** vector index. The query path refuses a
`provider="fake"` index unless `ED4ALL_EMBEDDING_ALLOW_FAKE=true` — so a demo
built with the fake provider would be un-askable on any consumer that has not
set that escape hatch. The builder therefore **refuses `--full` when
`ED4ALL_EMBEDDING_PROVIDER=fake`**.

Pin a real provider before a full build:

```bash
export ED4ALL_EMBEDDING_PROVIDER=st          # in-process sentence-transformers (default)
# export ED4ALL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5   # optional model pin
python scripts/build_demo_course.py --full
```

Determinism note: the same machine + venv + provider + model + `device=cpu` +
batch size produces a byte-identical `embeddings.npy` / `id_map.json`, so a
frozen bundle's index reproduces on a rebuild.

## Size expectations

The fixture PDF is ~4.5 KB. A full minted demo course dir is typically **~4–35
MB** depending on how many artifacts (source HTML, IMSCC, chunks, concept
graph, vector index) are retained. The dominant contributors are the packaged
IMSCC and the `embeddings.npy` (float32 `[N, dim]`). This is small enough to
commit as a bundle, but large binaries over 1 MB should follow the repo's
fixture policy: ship a regenerable builder (this script) rather than the raw
bytes when practical.

## Freezing + shipping

1. Mint the full bundle (`--full`) with a real embedding provider pinned.
2. Verify it: `libv2 validate course demo-photosynthesis` and
   `libv2 vector-index verify --course demo-photosynthesis` (exit 1 on drift).
3. Confirm the `NOTICE` file and the `manifest.json` `license` / `attribution`
   / `library_format_version` fields are present.
4. Freeze: the bundle is reproducible from this script + the tracked fixture, so
   record the exact `ED4ALL_EMBEDDING_PROVIDER` / `ED4ALL_EMBEDDING_MODEL` used
   (they are the only non-pinned inputs) alongside the shipped bundle.

The `library_format_version` on the manifest lets a future consumer know which
on-disk layout the frozen bundle was built against — see
[`library-versioning.md`](library-versioning.md) for the upgrade contract.
