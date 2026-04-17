# v0.1.0 Baseline Archive — Slot

## Why this directory exists

The pipeline state shipped on commit **`7403cf2`** (the parent of whatever
commit you read this from) is the v0.1.x cutover. Before any further
pipeline change ships, the **v0.1.0 real-domain artifact** — the first
real knowledge package the pipeline produced (held locally by the
maintainer, not shipped in this repo) and the package the nine-finding
defect analysis was performed against — must be snapshotted here,
verbatim, with no re-scoring, no re-chunking, no re-anything.

This is the single immutable evidence the NSF TechAccess "before / after"
narrative depends on. Once the v1.0 branch starts moving, regenerating the
v0.1.0 artifact becomes structurally impossible: the chunker, the metrics,
the canonicalisation, the orphan rule, the pedagogy graph split, every one
of those changes is already on `main` (or its successor branch) by then.

## What this directory must contain

When populated, the layout mirrors a standard Trainforge output dir:

```
archive/v0.1.0-baseline/
├── ARCHIVE_README.md          ← this file
├── manifest.json              ← course manifest as shipped on v0.1.0
├── course.json                ← outcome model as shipped (flat IDs only)
├── corpus/
│   ├── chunks.jsonl           ← every chunk as shipped (footer-contaminated etc.)
│   └── corpus_stats.json
├── graph/
│   └── concept_graph.json     ← the v0.1.0 mixed-tag graph
├── pedagogy/
│   └── pedagogy_model.json
├── quality/
│   └── quality_report.json    ← original v0.1.0 self-scores (the dishonest ones)
├── training_specs/
│   └── dataset_config.json
└── IMPORT_SUMMARY.md
```

Optionally, a separate `quality_report_rescored_v2.json` may be added that
re-runs the v0.1.x self-trust metrics against the *unchanged* corpus chunks
to produce the honest scores on the same data. This is the cleanest
side-by-side comparator for the grant narrative — same input, two metric
generations.

## Status as of this commit

**EMPTY.** The v0.1.0 real-domain artifact is not present in this checkout.
Possible reasons:

1. The package lives outside this repo (a separate output directory the
   maintainer holds locally).
2. The package was generated, assessed, and not committed — and the
   maintainer hasn't yet copied it here.
3. The package was committed earlier in branch history and removed.

## Action required before any further pipeline work

The repository owner (`mdmurphy822`) must either:

- (a) Copy the v0.1.0 real-domain outputs they assessed into this directory,
  matching the layout above, and commit them with a message of the form
  `archive: v0.1.0 real-domain baseline (frozen as of <date>)`. Note that
  the real-domain corpus itself is deliberately held outside this public
  repo; only the committed snapshot of the pipeline output structure
  belongs here. **OR**
- (b) If the artifact is genuinely lost, regenerate it by checking out
  commit `18c6613` (the last commit on `main` before the self-trust work
  shipped), running the pipeline against the original source document,
  and committing the outputs here. This is the lossy fallback — the
  regenerated package will not be byte-identical to the originally-assessed
  one. Note the regeneration in a sub-section here so the divergence is
  documented.

Until either (a) or (b) lands, the v1.0 severity flip described in
`VERSIONING.md` §3 is contingent on a proxy (the synthetic clean fixture)
rather than a real-domain regeneration. That weaker bar is acknowledged
in `VERSIONING.md` §3 and §6(b).

## Why an empty scaffold is committed

If this directory didn't exist, the obligation to capture the baseline
would migrate from "structural commitment in the repo" to "informal task
on someone's todo list." Empty scaffolds with explicit READMEs make the
gap visible to every contributor who reads the tree, and to every
external reviewer (NSF or otherwise) who looks at the repository
structure.
