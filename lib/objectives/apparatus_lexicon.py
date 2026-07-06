"""Shared exercise-/apparatus-detection lexicon (objective-synthesis fix W1).

Textbook "apparatus" — end-of-section exercise banners, readiness-check
callouts (``BE PREPARED`` / ``TRY IT``), answer-key glyph runs (ⓐⓑⓒ), and the
learning-objective section boilerplate ("By the end of this section, you will
be able to:") — is NON-instructional chrome. When it leaks into a per-window
objective seed or a sub-objective statement it produces vacuous / fragmentary
objectives. Several passes independently grew their own hardcoded marker
constants to fight this; this module is the single, data-driven source of truth.

Two surfaces:

* **Legacy constants (byte-identical re-exports).** ``EXERCISE_BANNER_RE``,
  ``FOLLOWING_EXERCISES_RE``, ``EXTRA_EXERCISE_BANNER_RE`` and ``JUNK_MARKERS``
  are the EXACT patterns previously defined in
  ``lib/chunk_heading_sanity`` (``_EXERCISE_BANNER_RE``) and
  ``lib/objectives/citation_reselect`` (``_FOLLOWING_EXERCISES_RE`` /
  ``_EXTRA_EXERCISE_BANNER_RE``) and ``lib/objectives/sub_objectives``
  (``_JUNK_MARKERS``). They are re-homed here so those call sites import them
  from one place instead of re-declaring them — a pure refactor, ZERO behavior
  change. Detection semantics at the old sites are unchanged.

* **Compiled profile (the superset).** :func:`compile_profile` reads
  ``schemas/taxonomies/exercise_apparatus_lexicon.json`` (via
  ``lib.ontology.taxonomy.load_taxonomy("exercise_apparatus_lexicon")``) and
  returns a :class:`CompiledProfile` bundling (a) the structural detector over
  the SemantiK ``composite_unit`` / ``unit_roles`` chunk metadata and (b) the
  compiled lexical marker regexes (union of every profile by default, because
  over-match is safe — downstream sanitation edits rendered/seed text only,
  never chunk ids or citability). This superset is what the flag-gated
  window-sanitation + seed-backstop passes (later waves) and the Defect-D
  ``lo_boilerplate`` prefix-strip consume.

Nothing here is gated by an env flag; it is inert vocabulary + helpers. The
behavior-changing wiring lives at the call sites behind their own flags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    # Legacy byte-identical re-exports (single source of truth).
    "EXERCISE_BANNER_RE",
    "FOLLOWING_EXERCISES_RE",
    "EXTRA_EXERCISE_BANNER_RE",
    "JUNK_MARKERS",
    # Compiled-profile surface.
    "CompiledProfile",
    "compile_profile",
    "TAXONOMY_NAME",
]

#: Taxonomy basename loaded via :func:`lib.ontology.taxonomy.load_taxonomy`.
TAXONOMY_NAME = "exercise_apparatus_lexicon"


# ---------------------------------------------------------------------------
# Legacy constants — byte-identical to the pre-consolidation definitions.
# Kept verbatim (NOT derived from the lexicon) so the re-export at the old call
# sites is provably behavior-preserving. See module docstring.
# ---------------------------------------------------------------------------

#: Was ``lib.chunk_heading_sanity._EXERCISE_BANNER_RE``.
EXERCISE_BANNER_RE = re.compile(
    r"\b(?:EXERCISES?\s+Practice\s+Makes\s+Perfect"
    r"|In\s+the\s+following\s+exercises)\b",
    re.IGNORECASE,
)

#: Was ``lib.objectives.citation_reselect._FOLLOWING_EXERCISES_RE``.
FOLLOWING_EXERCISES_RE = re.compile(r"In\s+the\s+following\s+exercises", re.I)

#: Was ``lib.objectives.citation_reselect._EXTRA_EXERCISE_BANNER_RE``.
EXTRA_EXERCISE_BANNER_RE = re.compile(
    r"\b(?:Practice\s+Makes\s+Perfect"
    r"|Section\s+Exercises"
    r"|Review\s+Exercises)\b",
    re.IGNORECASE,
)

#: Was ``lib.objectives.sub_objectives._JUNK_MARKERS``.
JUNK_MARKERS: Tuple[str, ...] = (
    "::", "ⓐ", "ⓑ", "ⓒ", "ⓓ", "ⓔ",
    "BE PREPARED", "TRY IT", "HOW TO", "LEARNING OBJECTIVES",
)


# ---------------------------------------------------------------------------
# Lexicon compilation (the superset)
# ---------------------------------------------------------------------------

#: Phrase-list group keys compiled into the generic marker set.
_MARKER_GROUPS = (
    "lo_boilerplate",
    "apparatus_banners",
    "callout_labels",
    "glyph_markers",
)

#: Subset safe to match against an OBJECTIVE STATEMENT. ``callout_labels``
#: is excluded: labels like "how to" / "learning objectives" are ordinary
#: English inside a legitimate CO statement, so matching them there would
#: drop real objectives. Callouts still count for render/seed-body scans.
_STATEMENT_MARKER_GROUPS = (
    "lo_boilerplate",
    "apparatus_banners",
    "glyph_markers",
)


def _phrase_to_regex(phrase: str) -> re.Pattern:
    """Compile a marker phrase to a case-insensitive regex.

    Word-internal whitespace becomes ``\\s+`` so a phrase still matches across
    an OCR line-break / doubled space; single-token markers (glyphs) compile to
    their escaped literal. Deliberately NOT word-boundary-anchored: over-match
    is safe for the render/seed-only sanitation consumers.
    """
    tokens = [re.escape(tok) for tok in str(phrase).split() if tok]
    if not tokens:
        return re.compile(r"(?!x)x")  # never-matches sentinel
    return re.compile(r"\s+".join(tokens), re.IGNORECASE)


@dataclass(frozen=True)
class CompiledProfile:
    """A compiled, ready-to-query apparatus-detection profile.

    ``profile_spec`` records the resolved profile selection (``"*"`` for the
    default union of every profile). ``marker_regexes`` is the union of every
    compiled lexical marker; ``lo_boilerplate_regexes`` is the subset compiled
    from the ``lo_boilerplate`` groups (anchored-at-start when used for the
    Defect-D prefix strip). The four ``*_units`` / ``*_roles`` frozensets are
    the structural signal enums.
    """

    profile_spec: str
    marker_regexes: Tuple[re.Pattern, ...]
    statement_marker_regexes: Tuple[re.Pattern, ...]
    lo_boilerplate_regexes: Tuple[re.Pattern, ...]
    apparatus_units: frozenset
    apparatus_roles: frozenset
    instructional_units: frozenset
    instructional_roles: frozenset

    # -- structural detection ------------------------------------------------

    def structural_signal(self, chunk: Any) -> Optional[bool]:
        """Tri-state apparatus signal from a chunk's pedagogical metadata.

        Returns ``True`` (apparatus / exercise material), ``False``
        (instructional prose — wins over apparatus when both are present), or
        ``None`` (no ``composite_unit`` / ``unit_roles`` metadata → caller
        defers to the lexical signal). Pure read; never raises.
        """
        if not isinstance(chunk, dict):
            return None
        unit = chunk.get("composite_unit")
        roles = chunk.get("unit_roles")
        role_set = {str(r) for r in roles} if isinstance(roles, list) else set()
        # Instructional wins over apparatus when both fire on one chunk.
        if unit in self.instructional_units or (
            role_set & self.instructional_roles
        ):
            return False
        if unit in self.apparatus_units or (role_set & self.apparatus_roles):
            return True
        return None

    def is_apparatus_chunk(self, chunk: Any, *, text: Optional[str] = None) -> bool:
        """Whether ``chunk`` is apparatus — structural signal first, lexical fallback.

        Uses :meth:`structural_signal` when the chunk carries the pedagogical
        metadata; otherwise (or when the caller passes explicit ``text``) falls
        back to the lexical marker scan over ``text`` (defaulting to the
        chunk's ``text`` / ``body``).
        """
        signal = self.structural_signal(chunk)
        if signal is not None:
            return signal
        if text is None and isinstance(chunk, dict):
            text = str(chunk.get("text") or chunk.get("body") or "")
        return self.text_has_marker(text or "")

    # -- lexical detection ---------------------------------------------------

    def text_has_marker(self, text: str) -> bool:
        """Whether ``text`` contains ANY compiled apparatus marker."""
        if not text:
            return False
        return any(rgx.search(text) for rgx in self.marker_regexes)

    def statement_has_apparatus_marker(self, text: str) -> bool:
        """Whether an objective STATEMENT carries an apparatus marker.

        Scans only the ``_STATEMENT_MARKER_GROUPS`` subset — callout labels
        ("how to", "learning objectives", …) are legitimate statement English
        and never flag here.
        """
        if not text:
            return False
        return any(rgx.search(text) for rgx in self.statement_marker_regexes)

    def leads_with_lo_boilerplate(self, text: str) -> bool:
        """Whether ``text`` (after leading whitespace) STARTS with LO boilerplate."""
        stripped = (text or "").lstrip()
        return any(rgx.match(stripped) for rgx in self.lo_boilerplate_regexes)

    def strip_lo_boilerplate_prefix(self, text: str) -> str:
        """Strip a leading learning-objective boilerplate preamble.

        When ``text`` leads with an ``lo_boilerplate`` marker (e.g. "By the end
        of this section, you will be able to:"), recover the tail AFTER the
        first colon (the real objective). No colon after the boilerplate → the
        line is pure preamble with no content → returns ``""``. When ``text``
        does not lead with boilerplate it is returned stripped-but-unchanged.
        """
        stripped = (text or "").strip()
        if not self.leads_with_lo_boilerplate(stripped):
            return stripped
        colon = stripped.find(":")
        if colon == -1:
            return ""
        return stripped[colon + 1:].strip()


def _select_profiles(
    lex: Dict[str, Any], profile_spec: Optional[str]
) -> List[str]:
    """Resolve a ``+``-joined profile spec to known profile keys, in order.

    ``None`` / ``"*"`` / empty → EVERY profile (the default union; over-match is
    safe). Unknown keys are skipped; an all-unknown spec falls back to the full
    union so a caller never gets a silently-empty lexicon.
    """
    known = list((lex.get("profiles") or {}).keys())
    if profile_spec is None or not str(profile_spec).strip() or profile_spec == "*":
        return known
    keys = [k.strip() for k in str(profile_spec).split("+") if k.strip()]
    keys = [k for k in keys if k in known]
    return keys or known


@lru_cache(maxsize=None)
def compile_profile(profile_spec: Optional[str] = None) -> CompiledProfile:
    """Compile the apparatus lexicon for ``profile_spec`` (default: union of all).

    Reads ``schemas/taxonomies/exercise_apparatus_lexicon.json`` through
    ``lib.ontology.taxonomy.load_taxonomy``. Cached per spec. ``profile_spec``
    is a ``+``-joined list of profile keys; ``None`` (default) unions every
    profile — the safe default, since detection over-match only affects
    render/seed text.
    """
    from lib.ontology.taxonomy import load_taxonomy  # noqa: PLC0415

    lex = load_taxonomy(TAXONOMY_NAME)
    profiles = lex.get("profiles") or {}
    keys = _select_profiles(lex, profile_spec)

    marker_seen: set = set()
    markers: List[re.Pattern] = []
    stmt_seen: set = set()
    stmt_markers: List[re.Pattern] = []
    lo_seen: set = set()
    lo_regexes: List[re.Pattern] = []
    for key in keys:
        prof = profiles.get(key) or {}
        for group in _MARKER_GROUPS:
            for phrase in prof.get(group) or []:
                norm = str(phrase).strip().lower()
                if not norm or norm in marker_seen:
                    continue
                marker_seen.add(norm)
                markers.append(_phrase_to_regex(phrase))
        for group in _STATEMENT_MARKER_GROUPS:
            for phrase in prof.get(group) or []:
                norm = str(phrase).strip().lower()
                if not norm or norm in stmt_seen:
                    continue
                stmt_seen.add(norm)
                stmt_markers.append(_phrase_to_regex(phrase))
        for phrase in prof.get("lo_boilerplate") or []:
            norm = str(phrase).strip().lower()
            if not norm or norm in lo_seen:
                continue
            lo_seen.add(norm)
            lo_regexes.append(_phrase_to_regex(phrase))

    structural = lex.get("structural") or {}

    def _fs(field: str) -> frozenset:
        return frozenset(
            str(v).strip() for v in (structural.get(field) or []) if str(v).strip()
        )

    return CompiledProfile(
        profile_spec=profile_spec or "*",
        marker_regexes=tuple(markers),
        statement_marker_regexes=tuple(stmt_markers),
        lo_boilerplate_regexes=tuple(lo_regexes),
        apparatus_units=_fs("apparatus_units"),
        apparatus_roles=_fs("apparatus_roles"),
        instructional_units=_fs("instructional_units"),
        instructional_roles=_fs("instructional_roles"),
    )
