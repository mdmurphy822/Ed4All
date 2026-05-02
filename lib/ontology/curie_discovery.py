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
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from lib.ontology.curie_extraction import EXCLUDED_PREFIXES, extract_curies

logger = logging.getLogger(__name__)


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


__all__ = [
    "discover_curies_from_corpus",
    "diff_against_manifest",
]
