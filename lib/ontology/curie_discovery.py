"""Corpus-driven CURIE discovery (Wave 137 followup).

The property manifest at ``schemas/training/property_manifest.<family>.yaml``
hand-declares the CURIEs the SLM adapter must teach. That works when
the operator already knows the schema surface, but a new corpus
(textbook, RDF/SHACL course, OWL ontology dump) often exposes
vocabulary that nobody has hand-enumerated yet. This module discovers
the corpus's actual CURIE inventory by walking ``chunks.jsonl`` and
tallying every CURIE that the canonical
``lib.ontology.curie_extraction.extract_curies`` regex finds, ranked
by occurrence frequency.

Use cases:

  1. Authoring a manifest for a new course family — operator runs
     ``Trainforge/scripts/discover_curies.py --format manifest`` to
     emit a starter ``property_manifest.<family>.yaml``.
  2. Catching manifest drift — operator runs
     ``--exclude-known-manifest`` against an existing course to
     surface CURIEs the corpus uses that the manifest hasn't declared.
  3. Driving the backfill loop dynamically — when
     ``backfill_form_data --discover-from-corpus`` is set, the loop's
     target list is computed from the corpus rather than the static
     manifest, so the number of CURIEs the operator backfills scales
     with the corpus instead of being hand-pinned.

The primitive is deliberately layering-clean: it depends only on
``curie_extraction`` (the canonical regex) and stdlib JSON / Path. The
Wave 137 backfill loop and the standalone discovery CLI both consume
this single helper, so any future change to the extraction contract
(e.g. adding a `data-cf-curie` HTML attribute pass) lands in one place.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from lib.ontology.curie_extraction import EXCLUDED_PREFIXES, extract_curies
from lib.ontology.slugs import canonical_slug

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dynamic CURIE minting (v0.3.0 corpus-generalization initiative)
# --------------------------------------------------------------------------- #
#
# A prose corpus (e.g. an OpenStax algebra textbook) has zero RDF/SHACL
# CURIEs in its text, so ``BlockCurieAnchoringValidator`` 100%-fails the
# ``OUTLINE_BLOCK_MISSING_CURIES`` gate. The helpers below mint a stable,
# grammar-valid per-course CURIE for every entry in the Stage-3 domain-
# concept vocabulary (``domain_concept_vocabulary.json``) so prose blocks
# can carry anchoring CURIEs derived from their own domain vocabulary.
#
# CRITICAL — dual-grammar intersection. Two CURIE grammars exist and
# disagree on the local-name character class:
#
#   * The outline JSON schema (``_outline_provider._CURIE_PATTERN``)
#     allows hyphens in the local name: ``^[a-z][a-z0-9]*:[A-Za-z0-9_-]+$``.
#   * The canonical extractor (``curie_extraction.CURIE_REGEX``) does NOT:
#     its local name is ``[A-Za-z][A-Za-z0-9_]*`` — must be letter-led,
#     underscores OK, NO hyphens.
#
# ``extract_curies()`` (used by the anchoring validator) runs CURIE_REGEX,
# so a minted CURIE with a hyphen (``alg:slope-intercept-form``) would be
# silently truncated to ``alg:slope``. Every minted CURIE therefore MUST
# satisfy the INTERSECTION of both grammars:
#
#   * prefix     — ``[a-z][a-z0-9]*`` (lowercase, letter-led, alnum)
#   * local name — ``[A-Za-z][A-Za-z0-9_]*`` (letter-led, hyphens → ``_``)
#
# ``test_curie_discovery.py`` round-trips every minted CURIE through both
# ``_CURIE_PATTERN`` and ``extract_curies`` to pin this invariant.

# Maximum minted-prefix length. Keeps the prefix terse — a long course id
# like ``OPENSTAX_INTRODUCTORY_ALGEBRA_2E`` collapses to a readable head.
_MINTED_PREFIX_MAX_LEN: int = 16

# Disallowed-character strip for prefix derivation (anything not [a-z0-9]).
_PREFIX_STRIP_RE = re.compile(r"[^a-z0-9]")

# Disallowed-character strip for the local name. canonical_slug already
# lowercases + kebab-cases; we then convert hyphens to underscores and
# drop anything outside [A-Za-z0-9_] so the result satisfies CURIE_REGEX.
_LOCALNAME_STRIP_RE = re.compile(r"[^A-Za-z0-9_]")


def discover_curies_from_corpus(
    chunks_path: Path,
    *,
    min_frequency: int = 2,
    extra_excluded_prefixes: Optional[Iterable[str]] = None,
    text_fields: Tuple[str, ...] = ("text",),
) -> Dict[str, int]:
    """Walk ``chunks.jsonl`` and return ``{curie: chunk_count}`` for
    every CURIE appearing in at least ``min_frequency`` distinct chunks.

    The count is **chunk-level** (one chunk that mentions ``rdf:type``
    100 times still counts as 1 toward the frequency total). This
    matches the canonical "how widely used is this CURIE in the corpus"
    semantic the property manifest's tier system assumes (high-tier
    >50 chunks, mid-tier 10-50, low-tier 2-10).

    Args:
        chunks_path: Path to a LibV2 chunks.jsonl. Each line must be a
            JSON object with at least one of the ``text_fields`` set
            to a string.
        min_frequency: Drop CURIEs appearing in fewer chunks than this.
            Default 2 — matches the property manifest's lowest tier so
            single-occurrence noise is filtered.
        extra_excluded_prefixes: Optional iterable of additional CURIE
            prefixes to drop on top of the canonical
            ``EXCLUDED_PREFIXES`` (URL schemes). Useful for project-
            local pseudo-CURIEs the operator wants to filter.
        text_fields: Tuple of chunk-object keys whose string values are
            scanned. Default ``("text",)``. Pass
            ``("text", "title")`` etc. to widen the scan surface.

    Returns:
        Dict ``{curie: chunk_count}`` sorted by descending count, then
        alphabetical CURIE for tie-breaking. Empty dict when the file
        is missing / empty / has no extractable CURIEs.

    Raises:
        FileNotFoundError: when ``chunks_path`` doesn't exist.
        ValueError: when ``min_frequency`` is < 1.
    """
    if min_frequency < 1:
        raise ValueError(
            f"min_frequency must be >= 1; got {min_frequency!r}"
        )
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"chunks.jsonl not found at {chunks_path}"
        )

    excluded = set(EXCLUDED_PREFIXES)
    if extra_excluded_prefixes:
        excluded.update(p.lower() for p in extra_excluded_prefixes)
    excluded_frozen = frozenset(excluded)

    counts: Dict[str, int] = {}

    with chunks_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning(
                    "discover_curies: skipping malformed JSON at line %d "
                    "in %s",
                    line_no,
                    chunks_path,
                )
                continue
            chunk_curies: set = set()
            for field in text_fields:
                value = obj.get(field)
                if isinstance(value, str) and value:
                    found = extract_curies(value)
                    if excluded_frozen != EXCLUDED_PREFIXES:
                        # Apply the project-local exclusion overlay.
                        found = {
                            c for c in found
                            if c.split(":", 1)[0].lower() not in excluded_frozen
                        }
                    chunk_curies.update(found)
            for curie in chunk_curies:
                counts[curie] = counts.get(curie, 0) + 1

    filtered = {
        curie: count
        for curie, count in counts.items()
        if count >= min_frequency
    }
    return dict(
        sorted(
            filtered.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    )


def diff_against_manifest(
    discovered: Dict[str, int],
    manifest_curies: Iterable[str],
) -> Tuple[Dict[str, int], List[str]]:
    """Return ``(new, dropped)``: ``new`` are corpus-discovered CURIEs
    NOT in the manifest (sorted-by-frequency-desc), and ``dropped`` are
    manifest-declared CURIEs that the corpus does NOT actually use
    (sorted alphabetically).

    Both signals matter:

      - ``new`` flags vocabulary the corpus uses that the manifest
        hasn't hand-declared yet — author them into the manifest if
        their frequency is meaningful.
      - ``dropped`` flags manifest entries that the corpus doesn't
        actually exercise — possible stale declarations or a corpus
        that's missing content.
    """
    manifest_set = set(manifest_curies)
    new = {
        curie: count
        for curie, count in discovered.items()
        if curie not in manifest_set
    }
    dropped = sorted(c for c in manifest_set if c not in discovered)
    return new, dropped


def mint_curie_prefix(course_id: str) -> str:
    """Derive a stable, grammar-valid lowercase CURIE prefix from a course id.

    The prefix is deterministic: same ``course_id`` always yields the
    same prefix. Derivation pipeline:

      1. Lowercase the input.
      2. Delete every character outside ``[a-z0-9]`` (hyphens,
         underscores, dots, whitespace all fuse out).
      3. Prepend ``c`` when the result starts with a digit (CURIE
         prefixes must be letter-led).
      4. Truncate to :data:`_MINTED_PREFIX_MAX_LEN` characters.

    The result always matches ``^[a-z][a-z0-9]*$`` — the prefix arm of
    both CURIE grammars. An empty / all-punctuation ``course_id`` falls
    back to the literal ``"course"``.

        >>> mint_curie_prefix("OPENSTAX_ALG_9")
        'samplecoursea'
        >>> mint_curie_prefix("9to5-physics")
        'c9to5physics'
        >>> mint_curie_prefix("")
        'course'

    Args:
        course_id: The course identifier / slug. Falsy input falls back
            to ``"course"``.

    Returns:
        A lowercase, letter-led, ``[a-z0-9]``-only prefix string, at most
        :data:`_MINTED_PREFIX_MAX_LEN` characters long.
    """
    lowered = (course_id or "").lower()
    stripped = _PREFIX_STRIP_RE.sub("", lowered)
    if not stripped:
        return "course"
    if stripped[0].isdigit():
        stripped = "c" + stripped
    return stripped[:_MINTED_PREFIX_MAX_LEN]


def curie_for_concept(canonical: str, prefix: str) -> str:
    """Mint the CURIE for a single domain concept.

    Returns ``f"{prefix}:{localname}"`` where ``localname`` is derived
    from ``canonical``:

      1. Run ``canonical`` through :func:`lib.ontology.slugs.canonical_slug`
         (lowercase kebab-case).
      2. Convert every hyphen to an underscore — ``CURIE_REGEX`` rejects
         hyphens in the local name, so ``extract_curies`` would otherwise
         truncate the CURIE at the first hyphen.
      3. Drop any residual character outside ``[A-Za-z0-9_]``.
      4. Prepend ``c`` when the result is empty or starts with a non-letter
         (the local name must be letter-led).

    The result always matches the dual-grammar intersection: the
    ``_CURIE_PATTERN`` schema check AND a clean ``extract_curies``
    round-trip.

        >>> curie_for_concept("Slope-Intercept Form", "alg")
        'alg:slope_intercept_form'
        >>> curie_for_concept("2D vectors", "phys")
        'phys:c2d_vectors'

    Args:
        canonical: The concept's canonical name (already run through
            ``canonical_slug`` at vocabulary-emit time, but re-slugged
            here defensively so a raw string also works).
        prefix: The minted course prefix from :func:`mint_curie_prefix`.

    Returns:
        A grammar-valid ``prefix:localname`` CURIE string.
    """
    slug = canonical_slug(canonical or "")
    localname = slug.replace("-", "_")
    localname = _LOCALNAME_STRIP_RE.sub("", localname)
    if not localname or not localname[0].isalpha():
        # Local name must be letter-led for CURIE_REGEX. Prepend a
        # constant letter rather than dropping the leading token so two
        # concepts that differ only in a leading digit stay distinct.
        localname = "c" + localname
    return f"{prefix}:{localname}"


def build_minted_curie_map(
    vocabulary: Dict,
    course_id: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Build the minted-CURIE map from a parsed domain-concept vocabulary.

    Given a parsed ``domain_concept_vocabulary.json`` dict (schema:
    ``schemas/knowledge/domain_concept_vocabulary.schema.json``), returns
    a map ``{minted_curie: {"canonical": <canonical>, "surface_forms":
    [<canonical>, *aliases]}}``.

    ``surface_forms`` carries the canonical name first, then every alias,
    deduplicated while preserving order. The anchoring validator
    case-insensitively substring-matches these against the block's text
    blob, so a minted CURIE is "anchored" when the block's prose
    actually mentions the concept (under any of its surface forms).

    The course id used to mint the prefix is resolved in priority order:
    the explicit ``course_id`` argument, then ``vocabulary["course_id"]``,
    then ``vocabulary["course_slug"]``. When none resolve, the prefix
    falls back to ``"course"`` via :func:`mint_curie_prefix`.

    Empty / missing ``concepts`` returns ``{}`` — a corpus with no domain
    vocabulary mints nothing, and the caller leaves blocks untouched.

    Args:
        vocabulary: Parsed ``domain_concept_vocabulary.json`` dict.
        course_id: Optional explicit course id; overrides the dict's
            ``course_id`` / ``course_slug`` when supplied.

    Returns:
        ``{minted_curie: {"canonical": str, "surface_forms": List[str]}}``.
        Empty dict when the vocabulary carries no concepts.
    """
    if not isinstance(vocabulary, dict):
        return {}
    concepts = vocabulary.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        return {}

    resolved_id = (
        course_id
        or vocabulary.get("course_id")
        or vocabulary.get("course_slug")
        or ""
    )
    prefix = mint_curie_prefix(str(resolved_id))

    minted: Dict[str, Dict[str, object]] = {}
    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        canonical = concept.get("canonical")
        if not isinstance(canonical, str) or not canonical:
            continue
        curie = curie_for_concept(canonical, prefix)
        # Surface forms: canonical first, then aliases, deduped (case-
        # insensitive) while preserving first-seen order.
        surface_forms: List[str] = []
        seen_lower: set = set()
        for sf in [canonical, *(concept.get("aliases") or [])]:
            if not isinstance(sf, str):
                continue
            sf = sf.strip()
            if not sf:
                continue
            key = sf.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            surface_forms.append(sf)
        # Collision policy: a later concept that mints to an already-
        # claimed CURIE merges its surface forms into the existing
        # entry rather than clobbering it (two concepts whose canonical
        # slugs collide are effectively the same concept).
        if curie in minted:
            existing = minted[curie]["surface_forms"]
            assert isinstance(existing, list)
            for sf in surface_forms:
                if sf.lower() not in {e.lower() for e in existing}:
                    existing.append(sf)
            continue
        minted[curie] = {
            "canonical": canonical,
            "surface_forms": surface_forms,
        }
    return minted


def locate_domain_concept_vocabulary(
    *,
    threaded_path: Optional[str] = None,
    course_slug: Optional[str] = None,
    libv2_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the ``domain_concept_vocabulary.json`` path for a course.

    R5 — single canonical locator. Before this helper two resolvers
    diverged: ``MCP/hardening/gate_input_routing.py`` had NO on-disk
    fallback while ``MCP/tools/pipeline_tools.py`` did, so a resumed /
    stage-subcommand run (where ``phase_outputs.concept_extraction``
    isn't threaded) gave the same logical gate two different verdicts.
    Both call sites now delegate here.

    Resolution order (first hit wins):

      1. ``threaded_path`` — the canonical
         ``phase_outputs.concept_extraction.domain_concept_vocabulary_path``
         thread (caller extracts it from whatever shape it has).
      2. The conventional on-disk location, a sibling of
         ``concept_graph_semantic.json`` under
         ``<libv2_root>/courses/<course_slug>/concept_graph/``.

    Returns ``None`` when neither resolves — the natural backward-compat
    gate: an RDF / legacy corpus has no domain-concept vocabulary, so
    the consuming minting / anchoring step becomes a complete no-op.

    Args:
        threaded_path: Vocabulary path threaded through phase outputs;
            ``None`` / empty falls through to the on-disk fallback.
        course_slug: Course slug for the on-disk fallback path. The
            caller is responsible for the ``course_code -> slug``
            transform (lower + ``_``/space -> ``-``).
        libv2_root: LibV2 root directory; required for the on-disk
            fallback. ``None`` skips the fallback.
    """
    if isinstance(threaded_path, str) and threaded_path:
        p = Path(threaded_path)
        if p.is_file():
            return p
    if course_slug and libv2_root is not None:
        vocab_path = (
            Path(libv2_root)
            / "courses"
            / course_slug
            / "concept_graph"
            / "domain_concept_vocabulary.json"
        )
        if vocab_path.is_file():
            return vocab_path
    return None


def resolve_minted_curie_map(
    *,
    threaded_path: Optional[str] = None,
    course_id: Optional[str] = None,
    course_slug: Optional[str] = None,
    libv2_root: Optional[Path] = None,
) -> Optional[Dict[str, Dict[str, object]]]:
    """Build the minted-CURIE map from the per-course domain vocabulary.

    R5 — the single canonical resolver both the gate-input router
    (``MCP/hardening/gate_input_routing.py``) and the validation
    handlers (``MCP/tools/pipeline_tools.py``) call. Locates
    ``domain_concept_vocabulary.json`` via
    :func:`locate_domain_concept_vocabulary` (INCLUDING the on-disk
    fallback), loads it, and returns the
    ``{minted_curie: {"canonical", "surface_forms"}}`` map.

    Returns ``None`` when no vocabulary path resolves (RDF / legacy
    corpora) or the file is missing / unparseable — the consuming gate
    then runs its legacy literal-token anchoring.
    """
    vocab_path = locate_domain_concept_vocabulary(
        threaded_path=threaded_path,
        course_slug=course_slug,
        libv2_root=libv2_root,
    )
    if vocab_path is None:
        return None
    try:
        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover — defensive
        logger.warning(
            "resolve_minted_curie_map: failed to load "
            "domain_concept_vocabulary.json at %s (%s); minted-CURIE "
            "anchoring unavailable.",
            vocab_path, exc,
        )
        return None
    try:
        minted = build_minted_curie_map(vocabulary, course_id=course_id)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.warning(
            "resolve_minted_curie_map: build_minted_curie_map failed: %s",
            exc,
        )
        return None
    return minted or None


def minted_curie_by_canonical(
    minted_curie_map: Dict[str, Dict[str, object]],
) -> Dict[str, str]:
    """Return the inverse ``{canonical: minted_curie}`` lookup.

    The minting helpers key on the minted CURIE, but the outline-handler
    minting step matches concepts by canonical slug (the output of
    :func:`lib.ontology.concept_tagging.extract_concept_tags`). This
    helper inverts :func:`build_minted_curie_map`'s output so a matched
    canonical slug resolves to its CURIE in O(1).

    Args:
        minted_curie_map: The output of :func:`build_minted_curie_map`.

    Returns:
        ``{canonical: minted_curie}``. When two CURIEs claim the same
        canonical (should not happen — collisions merge in
        ``build_minted_curie_map``), last-wins.
    """
    inverse: Dict[str, str] = {}
    for curie, entry in (minted_curie_map or {}).items():
        canonical = entry.get("canonical") if isinstance(entry, dict) else None
        if isinstance(canonical, str) and canonical:
            inverse[canonical] = curie
    return inverse


__all__ = [
    "discover_curies_from_corpus",
    "diff_against_manifest",
    "mint_curie_prefix",
    "curie_for_concept",
    "build_minted_curie_map",
    "locate_domain_concept_vocabulary",
    "resolve_minted_curie_map",
    "minted_curie_by_canonical",
]
