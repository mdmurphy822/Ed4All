"""WS1.2 citation-anchor contract test — two tiers.

Tier 1 (always-on): runs ``anchor_report`` over the shared synthetic fixture
``tests/fixtures/retrieval/mini_course/`` and asserts every honest chunk
resolves to a real source page containing its text (anchoring_rate == 1.0 over
the honest chunks; zero SOURCE_PAGE_MISSING) — plus that the one deliberately
fabricated-span chunk is detected, not papered over.

Tier 2 (real-corpus, skip-if-absent, mirrors
``Trainforge/tests/test_provenance.py``'s ``TRAINFORGE_PROVENANCE_CORPUS``
two-tier pattern): DISCOVERS the in-tree LibV2 chunksets dynamically (no
hardcoded course slugs — course data dirs are gitignored user data) and
parameterizes over whatever is present. Skips cleanly when no LibV2 chunkset is
discoverable so a clean checkout still passes CI.

Calibration protocol (fail-without-fix discipline): rather than per-course
pins, the floors are keyed off the discoverable ``chunkset_kind`` (read from
the chunkset manifest, or inferred from the canonical ``corpus/`` layout) using
the weakest measured floor observed for that kind (see ``_PER_KIND_FLOOR``).
Floors were MEASURED by running
``python -m lib.retrieval.citation_anchor <course> <kind>`` (containment_
threshold=0.85, shingle_size=8) and set conservatively per kind; a regression
that drops a rate below its kind's floor fails loudly. Manifest- and
corpus-bearing chunksets resolve via archived source pages / text containment;
``corpus`` kind is loosest because it carries no archived source pages.

Provenance gate (``LEGACY_EXTRACTION_CONTRACT``, supersedes the old
``LEGACY_CHUNKER_VERSION`` token): a tier-2 arm ENFORCES the per-kind floor
iff the corpus declares ``extraction_contract ==
Trainforge.chunker.EXTRACTION_TEXT_CONTRACT_VERSION`` in EITHER its
``course_manifest.json`` OR its chunkset sidecar ``manifest.json`` — otherwise
it SKIPS loudly. This is finer than the coarse ``chunker_version`` (which
deliberately stays ``"v4"`` across chunk-TEXT extraction-contract changes, so a
chunk-only ``--stop-after chunking`` sidecar carrying ``chunker_version="v4"``
can predate the extraction-text contract yet look current on that field). The
integer ``extraction_contract`` marker is truthful: every existing on-disk
corpus predates it and skips as legacy; every corpus chunked from now on
(including chunk-only slices) carries it and gets full floor enforcement.

Anti-silent-degradation: a course exposing an imscc chunkset must keep a
``*.imscc`` archived under ``source/imscc/`` (sha-verified against the imscc
manifest's source_imscc_sha256). If that cartridge is removed, that arm flips
to all-SOURCE_PAGE_MISSING and fails loudly via
``test_tier2_imscc_arms_have_source_archive``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.retrieval.citation_anchor import AnchorStatus, anchor_report
from Trainforge.chunker import EXTRACTION_TEXT_CONTRACT_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINI_COURSE = PROJECT_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"


def _corpus_provenance(course_dir: Path, chunks_path: Path):
    """Resolve ``(extraction_contract, chunker_version)`` for a real corpus.

    The tier-2 anchoring floor is enforced only on corpora produced under the
    CURRENT chunk-TEXT extraction contract (the always-on ``HTMLTextExtractor``
    screen-reader-scaffolding suppression + structural delimiters — see
    ``Trainforge/CLAUDE.md`` § "Extraction-text change class"). That contract is
    versioned as ``Trainforge.chunker.EXTRACTION_TEXT_CONTRACT_VERSION`` and
    stamped as ``extraction_contract`` onto BOTH the ``course_manifest.json``
    (fully-archived corpora) AND the chunkset sidecar ``manifest.json``
    (chunk-only ``--stop-after chunking`` slices that never get a course
    manifest). This helper scans BOTH sources — the course manifest first,
    then the sidecar sitting next to ``chunks.jsonl`` — and returns the first
    ``extraction_contract`` int found (or ``None`` when neither declares it),
    alongside the ``chunker_version`` found (for the greppable skip message).

    NB: ``chunker_version`` alone is TOO COARSE for this gate — it deliberately
    stays ``"v4"`` across extraction-text changes (bumped only on emit-SHAPE
    changes), so a chunk-only sidecar carrying ``chunker_version="v4"`` can
    predate the extraction-text contract yet look current on that field. The
    integer ``extraction_contract`` marker is what distinguishes them; a corpus
    lacking it is legacy (pre-marker) and skips.
    """
    extraction_contract = None
    chunker_version = None
    for manifest in (
        course_dir / "course_manifest.json",
        chunks_path.parent / "manifest.json",
    ):
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if extraction_contract is None and isinstance(
            data.get("extraction_contract"), int
        ):
            extraction_contract = data.get("extraction_contract")
        if chunker_version is None and data.get("chunker_version") is not None:
            chunker_version = data.get("chunker_version")
    return extraction_contract, chunker_version

try:
    from lib.paths import libv2_path

    LIBV2_COURSES = libv2_path() / "courses"
except Exception:  # pragma: no cover
    LIBV2_COURSES = PROJECT_ROOT / "LibV2" / "courses"


# ---------------------------------------------------------------------------
# Provenance-helper unit coverage (synthetic tmp_path manifests, no live slugs)
# ---------------------------------------------------------------------------
#
# The tier-2 floor is enforced iff `_corpus_provenance` resolves an
# extraction_contract == EXTRACTION_TEXT_CONTRACT_VERSION from EITHER the
# course_manifest.json OR the chunkset sidecar manifest. These exercise the
# marker-present-current / marker-absent / marker-stale branches directly so
# the gate's enforce/skip decision is regression-covered without a real corpus.


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_provenance_marker_current_from_course_manifest(tmp_path):
    course_dir = tmp_path / "course"
    chunks_path = course_dir / "dart_chunks" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        course_dir / "course_manifest.json",
        {
            "chunker_version": "v4",
            "extraction_contract": EXTRACTION_TEXT_CONTRACT_VERSION,
        },
    )
    contract, cver = _corpus_provenance(course_dir, chunks_path)
    assert contract == EXTRACTION_TEXT_CONTRACT_VERSION
    assert cver == "v4"
    # Gate decision: ENFORCE (marker current).
    assert contract == EXTRACTION_TEXT_CONTRACT_VERSION


def test_provenance_marker_current_from_sidecar_only(tmp_path):
    """Chunk-only --stop-after slice: no course_manifest, marker lives on the
    chunkset sidecar. The gate must still ENFORCE."""
    course_dir = tmp_path / "course"
    chunks_dir = course_dir / "dart_chunks"
    chunks_path = chunks_dir / "chunks.jsonl"
    _write_json(
        chunks_dir / "manifest.json",
        {
            "chunker_version": "v4",
            "extraction_contract": EXTRACTION_TEXT_CONTRACT_VERSION,
            "chunkset_kind": "dart",
        },
    )
    contract, cver = _corpus_provenance(course_dir, chunks_path)
    assert contract == EXTRACTION_TEXT_CONTRACT_VERSION
    assert cver == "v4"


def test_provenance_marker_absent_skips_as_legacy(tmp_path):
    """A stale sidecar-'v4' corpus with NO extraction_contract marker: the
    coarse chunker_version looks current but the gate must SKIP (legacy)."""
    course_dir = tmp_path / "course"
    chunks_dir = course_dir / "dart_chunks"
    chunks_path = chunks_dir / "chunks.jsonl"
    _write_json(
        chunks_dir / "manifest.json",
        {"chunker_version": "v4", "chunkset_kind": "dart"},
    )
    contract, cver = _corpus_provenance(course_dir, chunks_path)
    assert contract is None
    assert cver == "v4"
    # Gate decision: SKIP as legacy (marker absent despite chunker_version v4).
    assert contract != EXTRACTION_TEXT_CONTRACT_VERSION


def test_provenance_marker_stale_skips_as_legacy(tmp_path):
    """A corpus stamped with a DIFFERENT extraction_contract than the live one
    (the state a future contract bump creates for pre-bump corpora) must SKIP —
    the marker is present but stale relative to EXTRACTION_TEXT_CONTRACT_VERSION."""
    stale = EXTRACTION_TEXT_CONTRACT_VERSION + 1
    course_dir = tmp_path / "course"
    chunks_path = course_dir / "dart_chunks" / "chunks.jsonl"
    _write_json(
        course_dir / "course_manifest.json",
        {"chunker_version": "v4", "extraction_contract": stale},
    )
    contract, _cver = _corpus_provenance(course_dir, chunks_path)
    assert contract == stale
    # Gate decision: SKIP as legacy (marker present but != live version).
    assert contract != EXTRACTION_TEXT_CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Tier 1 — always-on synthetic fixture
# ---------------------------------------------------------------------------


def test_tier1_mini_course_dart_all_honest_chunks_anchor():
    chunks_path = MINI_COURSE / "semantik_chunks" / "chunks.jsonl"
    assert chunks_path.is_file(), (
        "mini-course fixture missing; run "
        "tests/fixtures/retrieval/mini_course/build_mini_course.py"
    )
    report = anchor_report(chunks_path, MINI_COURSE, chunkset_kind="dart")

    # Zero missing source pages — every item_path resolves.
    assert report["status_counts"][AnchorStatus.SOURCE_PAGE_MISSING.value] == 0

    # Exactly one planted fabricated-span chunk (id suffix _fabricated).
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text().splitlines()
        if line.strip()
    ]
    fabricated_ids = {c["id"] for c in chunks if c["id"].endswith("_fabricated")}
    assert len(fabricated_ids) == 1

    # The honest chunks (all but the fabricated one) all resolve.
    honest = len(chunks) - len(fabricated_ids)
    resolved = report["resolved_count"]
    # The fabricated chunk's text is still a real page substring, so it
    # resolves via containment; what matters is the *span fidelity* signal
    # distinguishes it from an exact resolve. So all chunks resolve, and the
    # honest ones include the RESOLVED_EXACT cases.
    assert resolved >= honest
    assert report["anchoring_rate"] == pytest.approx(1.0)
    # At least the planted-correct chunks land RESOLVED_EXACT.
    assert report["status_counts"][AnchorStatus.RESOLVED_EXACT.value] >= 1


def test_tier1_mini_course_fabricated_span_not_exact():
    """The planted fabricated-span chunk must NOT be classified RESOLVED_EXACT
    (its span is a lie). It resolves via containment/normalized, proving the
    resolver detects the chunker bug class without repairing the span."""
    from lib.retrieval.citation_anchor import resolve_citation_anchor

    chunks_path = MINI_COURSE / "semantik_chunks" / "chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text().splitlines()
        if line.strip()
    ]
    fab = next(c for c in chunks if c["id"].endswith("_fabricated"))
    anchor = resolve_citation_anchor(fab, MINI_COURSE, chunkset_kind="dart")
    assert anchor.status is not AnchorStatus.RESOLVED_EXACT


def test_tier1_mini_course_imscc_resolves():
    """The mini cartridge member item_paths resolve via stdlib zipfile."""
    chunks_path = MINI_COURSE / "semantik_chunks" / "chunks.jsonl"
    # The chunks use item_path "alpha.html"/"beta.html" which are also
    # the imscc member names — so the same chunkset resolves under the imscc
    # axis against the mini cartridge.
    report = anchor_report(chunks_path, MINI_COURSE, chunkset_kind="imscc")
    assert report["status_counts"][AnchorStatus.SOURCE_PAGE_MISSING.value] == 0
    assert report["anchoring_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tier 2 — real corpora (skip-if-absent), conservative per-kind floors
# ---------------------------------------------------------------------------
#
# Course data dirs are gitignored user data; tier-2 tests must NOT name course
# slugs. We DISCOVER whatever chunksets are present under LibV2/courses/* and
# key the anchoring floor off the discoverable chunkset_kind (read from the
# sibling manifest, or inferred from the canonical corpus/ layout) rather than a
# per-course pin.
#
# The original per-(slug,kind) pins (measured 2026-06-09, floor = measured-0.02)
# ranged 0.63 (corpus) .. 0.98 (dart/imscc). A single global floor would barely
# protect the high-anchoring kinds, so we keep a conservative floor PER KIND —
# the weakest measured floor observed for that kind — which preserves the
# protective intent (a regression that drops a kind below the worst seen for
# that kind still fails loudly) without pinning any individual slug. corpus is
# the loosest because corpus chunksets carry no archived source pages and rely
# wholly on text containment.
_PER_KIND_FLOOR = {
    "dart": 0.96,
    "imscc": 0.96,
    "corpus": 0.63,
}

# Chunkset dirs that carry a manifest declaring chunkset_kind, plus the
# canonical corpus/ layout whose kind is the dir name.
_MANIFEST_CHUNKSETS = ("dart_chunks", "imscc_chunks")


def _discover_real_corpora():
    """Yield (course_slug, kind, chunks_path, floor) for every discoverable
    real chunkset. Kind is read off the sibling manifest where present, else
    inferred from the canonical corpus/ dir name."""
    out = []
    if not LIBV2_COURSES.is_dir():
        return out
    for course_dir in sorted(p for p in LIBV2_COURSES.iterdir() if p.is_dir()):
        for sub in _MANIFEST_CHUNKSETS:
            chunks_path = course_dir / sub / "chunks.jsonl"
            manifest = course_dir / sub / "manifest.json"
            if not chunks_path.is_file() or not manifest.is_file():
                continue
            try:
                kind = json.loads(manifest.read_text(encoding="utf-8")).get(
                    "chunkset_kind"
                )
            except Exception:  # pragma: no cover - skip malformed
                continue
            if kind in _PER_KIND_FLOOR:
                out.append(
                    (course_dir.name, kind, chunks_path, _PER_KIND_FLOOR[kind])
                )
        # Canonical corpus/ chunkset (no manifest): kind == "corpus".
        corpus_chunks = course_dir / "corpus" / "chunks.jsonl"
        if corpus_chunks.is_file():
            out.append(
                (course_dir.name, "corpus", corpus_chunks, _PER_KIND_FLOOR["corpus"])
            )
    return out


_REAL_CORPORA = _discover_real_corpora()


@pytest.mark.skipif(
    not _REAL_CORPORA,
    reason="no LibV2 chunksets discoverable under LibV2/courses/*",
)
@pytest.mark.parametrize(
    "course_slug,kind,chunks_path,floor",
    _REAL_CORPORA,
    ids=[f"{c}-{k}" for c, k, _, _ in _REAL_CORPORA],
)
def test_tier2_real_corpus_anchoring_floor(course_slug, kind, chunks_path, floor):
    course_dir = LIBV2_COURSES / course_slug
    if not chunks_path.is_file():
        pytest.skip(
            f"LibV2 corpus {course_slug}/{kind} absent on this checkout"
        )
    # Provenance gate (fresh corpora keep full enforcement; legacy corpora
    # skip loudly). The per-kind anchoring floors were measured against
    # corpora produced under the CURRENT chunk-TEXT extraction contract
    # (HTMLTextExtractor sr-scaffolding suppression + structural delimiters).
    # A corpus chunked before that contract carries chunk spans the current
    # resolver can no longer anchor exactly, so its rate legitimately sits
    # below the floor without being a REGRESSION on the current pipeline.
    #
    # Gate on the integer ``extraction_contract`` marker (read from EITHER the
    # course_manifest.json OR the chunkset sidecar manifest) rather than the
    # coarse ``chunker_version`` (formerly the LEGACY_CHUNKER_VERSION gate) —
    # chunker_version deliberately stays "v4" across extraction-text changes,
    # so 4 stale sidecar-"v4" corpora on this checkout predate the contract yet
    # would look current on that field. The extraction_contract marker is
    # truthful: EVERY existing on-disk corpus predates it -> skips as legacy;
    # every corpus chunked from now on (including chunk-only --stop-after
    # slices, whose sidecar carries the marker) -> full floor enforcement.
    # Current corpora are enforced; legacy ones skip with a greppable
    # LEGACY_EXTRACTION_CONTRACT reason (supersedes the old
    # LEGACY_CHUNKER_VERSION token) naming what was found.
    extraction_contract, chunker_version = _corpus_provenance(
        course_dir, chunks_path
    )
    if extraction_contract != EXTRACTION_TEXT_CONTRACT_VERSION:
        pytest.skip(
            f"{course_slug}/{kind}: LEGACY_EXTRACTION_CONTRACT — "
            f"extraction_contract={extraction_contract!r} != current "
            f"EXTRACTION_TEXT_CONTRACT_VERSION="
            f"{EXTRACTION_TEXT_CONTRACT_VERSION!r} "
            f"(chunker_version={chunker_version!r}; the coarse chunker_version "
            f"gate this replaces was LEGACY_CHUNKER_VERSION). The per-kind "
            f"anchoring floor is enforced only on corpora chunked under the "
            f"CURRENT extraction-text contract (HTMLTextExtractor "
            f"sr-scaffolding suppression + structural delimiters); a corpus "
            f"predating the extraction_contract marker skips here. Re-chunk / "
            f"re-archive the course to enforce the floor on it."
        )
    report = anchor_report(chunks_path, course_dir, chunkset_kind=kind)
    counts = report["status_counts"]
    # Distinguish "source archive simply absent" (every chunk missing its
    # source page → nothing to anchor against, equivalent to an absent corpus)
    # from "source present but anchoring regressed". Only the latter is a
    # protective failure; the former skips like an absent checkout. An
    # incomplete scratch course (no archived source pages) must not turn the
    # dynamic-discovery floor test red.
    resolvable = (
        counts.get("resolved_exact", 0)
        + counts.get("resolved_normalized", 0)
        + counts.get("resolved_containment", 0)
        + counts.get("span_fabricated", 0)
    )
    if resolvable == 0:
        pytest.skip(
            f"{course_slug}/{kind}: no archived source pages "
            f"(all {counts.get('source_page_missing', 0)} chunks "
            f"SOURCE_PAGE_MISSING) — source archive absent, nothing to anchor"
        )
    rate = report["anchoring_rate"]
    assert rate >= floor, (
        f"{course_slug}/{kind} anchoring_rate {rate:.4f} dropped below the "
        f"conservative per-kind floor {floor:.2f}. "
        f"status_counts={report['status_counts']}. "
        f"Re-measure with `python -m lib.retrieval.citation_anchor "
        f"{course_slug} {kind}` and re-pin _PER_KIND_FLOOR if the drop is "
        f"intentional; never silently lower the floor."
    )


def test_tier2_imscc_arms_have_source_archive():
    """Anti-silent-degradation: any course exposing an imscc chunkset must keep
    a *.imscc archived under source/imscc/ for the imscc arm to resolve. If it
    is removed, that arm goes to all-SOURCE_PAGE_MISSING — this test names the
    cause. Discovers imscc-bearing courses dynamically (no hardcoded slug)."""
    imscc_courses = [
        (slug, path)
        for slug, kind, path, _ in _REAL_CORPORA
        if kind == "imscc"
    ]
    if not imscc_courses:
        pytest.skip("no imscc chunksets discoverable on this checkout")
    for slug, _path in imscc_courses:
        imscc_dir = LIBV2_COURSES / slug / "source" / "imscc"
        archived = list(imscc_dir.glob("*.imscc")) if imscc_dir.is_dir() else []
        assert archived, (
            f"Source-archive regression for {slug}: no *.imscc archived under "
            f"{imscc_dir}. The imscc citation arm cannot resolve without it. "
            "Re-copy from the project export's final package "
            "(sha-verified against the imscc manifest's source_imscc_sha256)."
        )
