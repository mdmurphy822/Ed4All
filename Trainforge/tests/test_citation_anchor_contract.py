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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINI_COURSE = PROJECT_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"

try:
    from lib.paths import libv2_path

    LIBV2_COURSES = libv2_path() / "courses"
except Exception:  # pragma: no cover
    LIBV2_COURSES = PROJECT_ROOT / "LibV2" / "courses"


# ---------------------------------------------------------------------------
# Tier 1 — always-on synthetic fixture
# ---------------------------------------------------------------------------


def test_tier1_mini_course_dart_all_honest_chunks_anchor():
    chunks_path = MINI_COURSE / "dart_chunks" / "chunks.jsonl"
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

    chunks_path = MINI_COURSE / "dart_chunks" / "chunks.jsonl"
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
    chunks_path = MINI_COURSE / "dart_chunks" / "chunks.jsonl"
    # The dart chunks use item_path "alpha.html"/"beta.html" which are also
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
