"""H1-derived, run-timestamped course slugs (``--auto-name``).

Owner directive: "slugs can inherit the H1 title after semantik creates it,
combined with the Date/time the run was initialized."

The ``--auto-name`` flag on ``ed4all run textbook-to-course`` makes the CLI
``--course-name`` a PROVISIONAL identity only (run_id / state dir / logging).
Immediately after the conversion (+ heading_judge) phases complete — and
BEFORE ``staging``, the first phase that consumes course identity into
artifacts — the workflow runner resolves the FINAL slug from the accessible
HTML's ``<h1>`` (the real book title on multi-chapter books, guaranteed by the
chapter-ladder reconcile) plus the run-INIT timestamp:

    canonical_slug(h1_title)[whole-token cap ~60 chars] + "-YYYYMMDD-HHMM"

    e.g. "principles-of-sample-systems-20260722-0704"

Everything here is deterministic and HONEST — when the h1 cannot be resolved
(multi-file corpus, missing/garbage h1, no alphanumeric content) the provided
provisional name is KEPT and the reason recorded; a title is never fabricated.

The resolver is PURE (no workflow-state writes): the workflow-runner seam
(``MCP/core/workflow_runner.py::WorkflowRunner._maybe_apply_auto_name``) owns
persistence + the decision-capture event.
"""

from __future__ import annotations

import html as _htmllib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from lib.ontology.slugs import canonical_slug

logger = logging.getLogger(__name__)

__all__ = [
    "AUTO_SLUG_TITLE_MAX_CHARS",
    "AUTO_SLUG_TIMESTAMP_FORMAT",
    "AutoNameResolution",
    "extract_h1_title",
    "title_rejection_reason",
    "truncate_slug_whole_tokens",
    "compose_auto_slug",
    "resolve_run_init_timestamp",
    "resolve_auto_course_name",
]

#: Cap on the TITLE part of the composed slug (the timestamp suffix is
#: appended after the cap). Truncation is whole-token: a hyphen-delimited
#: token is never split mid-word.
AUTO_SLUG_TITLE_MAX_CHARS = 60

#: Run-init timestamp suffix format (the run INIT time per the owner
#: directive — never the resolution time).
AUTO_SLUG_TIMESTAMP_FORMAT = "%Y%m%d-%H%M"

#: Bounded read for h1 extraction — the h1 of an accessible-HTML conversion
#: output lands within the first few KB; 256 KiB is a generous honest bound
#: that never slurps a whole 50 MB book into memory.
_H1_READ_MAX_BYTES = 256 * 1024

#: Raw-title length ceiling: an "h1" longer than this is body text that leaked
#: into a heading, not a book title.
_TITLE_MAX_RAW_CHARS = 120

_H1_RE = re.compile(r"<h1\b[^>]*>(.*?)</h1\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Structural / apparatus headings that are h1-shaped but are NOT a book title
# (``Chapter 3``, ``Part IV``, ``Unit 2:``, ``Appendix A`` …). Domain-agnostic
# by construction (structural words + ordinals only — never publisher or
# corpus vocabulary).
_STRUCTURAL_TITLE_RE = re.compile(
    r"^(chapter|section|part|unit|module|week|appendix|lesson|volume)"
    r"[\s:.\-]*([0-9]+|[ivxlcdm]+|[a-z])?[\s:.\-]*$",
    re.IGNORECASE,
)

# Purely numeric / punctuation "titles" (page numbers, ``1.2.3`` …).
_NUMERIC_TITLE_RE = re.compile(r"^[\d\s.,:;\-–—()]+$")


@dataclass(frozen=True)
class AutoNameResolution:
    """Outcome of one auto-name resolution attempt.

    ``final_name`` is ALWAYS usable: the composed slug on success, the
    unchanged provisional name on any fallback arm.
    """

    final_name: str
    resolved: bool
    reason: str
    display_title: Optional[str] = None
    h1_title: Optional[str] = None


def extract_h1_title(
    html_path: Path, max_bytes: int = _H1_READ_MAX_BYTES
) -> Optional[str]:
    """Return the first ``<h1>`` text of ``html_path`` (bounded read).

    Reads at most ``max_bytes`` from the head of the file, finds the first
    ``<h1>`` element, strips nested tags, unescapes entities, and collapses
    whitespace. Returns ``None`` when the file is unreadable or carries no
    ``<h1>`` within the bound.
    """
    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(max_bytes)
    except OSError as exc:
        logger.warning("auto-name: cannot read %s (%s)", html_path, exc)
        return None
    match = _H1_RE.search(head)
    if not match:
        return None
    inner = _TAG_RE.sub(" ", match.group(1))
    title = _WS_RE.sub(" ", _htmllib.unescape(inner)).strip()
    return title or None


def title_rejection_reason(title: Optional[str]) -> Optional[str]:
    """Return a rejection reason for ``title``, or ``None`` when acceptable.

    The fallback matrix (each arm keeps the provisional name):

    * ``h1_missing``    — no/empty title.
    * ``h1_too_long``   — raw title over 120 chars (leaked body text).
    * ``h1_structural`` — a structural heading (``Chapter 3``, ``Part IV`` …).
    * ``h1_numeric``    — digits/punctuation only.
    * ``slug_empty``    — no alphanumeric content survives ``canonical_slug``.
    """
    if not title or not title.strip():
        return "h1_missing"
    stripped = title.strip()
    if len(stripped) > _TITLE_MAX_RAW_CHARS:
        return "h1_too_long"
    if _STRUCTURAL_TITLE_RE.match(stripped):
        return "h1_structural"
    if _NUMERIC_TITLE_RE.match(stripped):
        return "h1_numeric"
    if not canonical_slug(stripped):
        return "slug_empty"
    return None


def truncate_slug_whole_tokens(
    slug: str, max_chars: int = AUTO_SLUG_TITLE_MAX_CHARS
) -> str:
    """Truncate ``slug`` to ``max_chars`` without splitting a hyphen token.

    The first token is kept even if it alone exceeds ``max_chars`` (a slug
    must never truncate to ``""`` when it had content).
    """
    if len(slug) <= max_chars:
        return slug
    tokens = [t for t in slug.split("-") if t]
    if not tokens:
        return slug[:max_chars].rstrip("-")
    kept: List[str] = [tokens[0]]
    length = len(tokens[0])
    for token in tokens[1:]:
        if length + 1 + len(token) > max_chars:
            break
        kept.append(token)
        length += 1 + len(token)
    return "-".join(kept)


def compose_auto_slug(title: str, run_init: datetime) -> str:
    """Compose the final slug: capped ``canonical_slug(title)`` + timestamp.

        >>> from datetime import datetime
        >>> compose_auto_slug(
        ...     "Principles Of Sample Systems",
        ...     datetime(2026, 7, 22, 7, 4),
        ... )
        'principles-of-sample-systems-20260722-0704'
    """
    base = truncate_slug_whole_tokens(canonical_slug(title))
    return f"{base}-{run_init.strftime(AUTO_SLUG_TIMESTAMP_FORMAT)}"


def resolve_run_init_timestamp(
    created_at: Optional[str], run_id: Optional[str]
) -> Optional[datetime]:
    """Resolve the run-INIT timestamp from the workflow record.

    Preference order (both are stamped at run creation, so either honestly
    represents "the Date/time the run was initialized"):

    1. The workflow-state ``created_at`` ISO string.
    2. The ``TTC_<course>_<YYYYmmdd_HHMMSS>`` run_id suffix.

    Returns ``None`` when neither parses — the caller falls back to keeping
    the provisional name rather than stamping a fabricated time.
    """
    if created_at:
        try:
            return datetime.fromisoformat(str(created_at))
        except ValueError:
            pass
    if run_id:
        match = re.search(r"(\d{8})_(\d{6})$", str(run_id))
        if match:
            try:
                return datetime.strptime(
                    f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S"
                )
            except ValueError:
                pass
    return None


def resolve_auto_course_name(
    provisional_name: str,
    html_paths: Sequence[str],
    run_init: Optional[datetime],
) -> AutoNameResolution:
    """Resolve the final course name from the conversion output.

    Pure function — no filesystem writes, no state mutation. Fallback arms
    (all keep ``provisional_name``): ``no_conversion_output`` (empty path
    list), ``multi_file_corpus`` (>1 accessible HTML — no single h1 names the
    corpus), ``no_run_timestamp``, plus every :func:`title_rejection_reason`
    arm evaluated against the FIRST (only) HTML file's first ``<h1>``.
    """
    paths = [p for p in (html_paths or []) if p]
    if not paths:
        return AutoNameResolution(
            final_name=provisional_name,
            resolved=False,
            reason="no_conversion_output",
        )
    if len(paths) > 1:
        return AutoNameResolution(
            final_name=provisional_name,
            resolved=False,
            reason="multi_file_corpus",
        )
    if run_init is None:
        return AutoNameResolution(
            final_name=provisional_name,
            resolved=False,
            reason="no_run_timestamp",
        )
    title = extract_h1_title(Path(paths[0]))
    rejection = title_rejection_reason(title)
    if rejection is not None:
        return AutoNameResolution(
            final_name=provisional_name,
            resolved=False,
            reason=rejection,
            h1_title=title,
        )
    assert title is not None  # title_rejection_reason(None) != None
    return AutoNameResolution(
        final_name=compose_auto_slug(title, run_init),
        resolved=True,
        reason="h1_resolved",
        display_title=title.strip(),
        h1_title=title,
    )
