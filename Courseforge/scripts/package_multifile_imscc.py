#!/usr/bin/env python3
"""
Package multi-file weekly course content into an IMS Common Cartridge (IMSCC) file.

Walks 03_content_development/week_*/ directories and creates an IMSCC with a proper
imsmanifest.xml reflecting the week -> module hierarchy.

Per-week ``learningObjectives`` validation runs by default (Wave 2, Worker L
— REC-CTR-03). Every ``week_*/*.html`` page with JSON-LD is validated against
the canonical objectives registry before packaging; the packager refuses to
build when any page's ``learningObjectives`` lists an ID outside its week's
allowed set. This guards against the LO-fanout defect that shipped in
pre-Worker-H packages and capped Trainforge quality metrics.

Resolution order for the objectives file:

    1. Explicit ``--objectives PATH`` argument.
    2. Auto-discovery: ``<content_dir>/course.json`` if it exists.
    3. None available — log a warning and skip validation (backward-compat).

``--skip-validation`` remains as an explicit opt-out for emergencies.

Usage:
    python package_multifile_imscc.py <content_dir> <output_imscc>
    python package_multifile_imscc.py <content_dir> <output_imscc> \
        --objectives inputs/exam-objectives/SAMPLE_101_objectives.json
    python package_multifile_imscc.py <content_dir> <output_imscc> \
        --skip-validation  # escape hatch, not recommended for production
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# Match the ``<h1>Week N Overview: {real title}</h1>`` tag emitted by
# :func:`Courseforge.scripts.generate_course.generate_week`. The real
# chapter title — the part after ``"Overview:"`` / ``"Overview &mdash;"`` /
# ``"— Overview"`` — is what the manifest week item should surface so
# Brightspace / Canvas render a meaningful week label instead of a bare
# ``"Week 3"``.
_WEEK_OVERVIEW_H1_RE = re.compile(
    r"<h1[^>]*>\s*(.*?)\s*</h1>",
    re.IGNORECASE | re.DOTALL,
)
_OVERVIEW_TITLE_SEP_RE = re.compile(
    r"(?i)(?:overview\s*[:—–-]\s*|"      # "Overview: Title" / "Overview — Title"
    r"\s*[—–-]\s*overview\s*$)"          # "Title — Overview"
)
_BARE_OVERVIEW_RE = re.compile(r"(?i)^\s*overview\s*$")

# Matches a FLAT week page emitted directly under ``03_content_development/``
# by the two-pass rewrite tier (``MCP/tools/pipeline_tools.py`` writes
# ``{page_id}.html`` where ``page_id = week_NN_content_NN``). The legacy
# single-pass ``generate_course.py`` writes NESTED ``week_NN/<page>.html``
# subdirs instead. Both layouts must package identically.
_FLAT_WEEK_PAGE_RE = re.compile(r"(?i)^(week_(\d+))_.+\.html$")


def _iter_week_groups(content_dir: Path) -> List[Tuple[str, int, Path, List[Path]]]:
    """Yield one entry per week, handling BOTH content layouts.

    Returns a sorted list of ``(week_name, week_num, title_dir, html_files)``
    tuples where:

    * ``week_name`` is the canonical ``week_NN`` slug used as the in-zip
      directory prefix + manifest ``WEEK_n`` id stem,
    * ``week_num`` is the integer week number,
    * ``title_dir`` is the directory to pass to ``_extract_week_title``
      (the real subdir for nested layout; ``content_dir`` for flat layout —
      ``_extract_week_title`` looks for ``week_NN_overview.html`` under it,
      which IS where the flat overview lives),
    * ``html_files`` are the week's pages, sorted by pedagogical order.

    Nested layout (``week_NN/*.html`` subdirs) is detected first and wins;
    a directory whose name matches ``week_*`` and ``is_dir()`` is grouped
    by its contained ``*.html`` files. Any FLAT ``week_NN_*.html`` file
    sitting directly under ``content_dir`` is grouped by its ``week_NN``
    prefix. The two are merged so a hybrid layout (unlikely, but safe)
    still packages every page exactly once.
    """
    groups: Dict[str, List[Path]] = {}
    title_dirs: Dict[str, Path] = {}

    # Nested subdirs.
    for week_dir in sorted(content_dir.glob("week_*")):
        if not week_dir.is_dir():
            continue
        files = sorted(week_dir.glob("*.html"))
        if files:
            groups.setdefault(week_dir.name, []).extend(files)
            title_dirs.setdefault(week_dir.name, week_dir)

    # Flat files directly under content_dir.
    for hf in sorted(content_dir.glob("week_*.html")):
        if not hf.is_file():
            continue
        m = _FLAT_WEEK_PAGE_RE.match(hf.name)
        if not m:
            continue
        week_name = m.group(1)
        # Avoid double-counting a file already grouped via a nested subdir.
        existing = groups.setdefault(week_name, [])
        if hf not in existing:
            existing.append(hf)
        title_dirs.setdefault(week_name, content_dir)

    def _week_num(week_name: str) -> int:
        digits = week_name.replace("week_", "").lstrip("0") or "0"
        try:
            return int(digits)
        except ValueError:
            return 0

    order = {
        "overview": 0, "content": 1, "application": 2,
        "self_check": 3, "key_terms": 4, "faq": 5, "summary": 6,
        "discussion": 7,
    }

    def _sort_key(f: Path):
        name = f.stem
        for key, val in order.items():
            if key in name:
                return (val, name)
        return (99, name)

    out: List[Tuple[str, int, Path, List[Path]]] = []
    for week_name in sorted(groups, key=_week_num):
        files = sorted(groups[week_name], key=_sort_key)
        out.append(
            (week_name, _week_num(week_name), title_dirs[week_name], files)
        )
    return out


def _extract_week_title(week_dir: Path, week_num: int) -> str:
    """Derive a human-readable week title from the week's overview HTML.

    Looks at ``week_NN_overview.html`` and pulls the chapter-title portion
    out of the emitted ``<h1>`` (Courseforge generate_week wraps it as
    ``"Week {N} Overview: {title}"``). Returns ``"Week {N}"`` when:

      * the overview file is missing,
      * its ``<h1>`` has no chapter title (neutral "Overview" fallback), or
      * parsing fails for any I/O reason.

    Never raises — packager manifest building is best-effort on the title
    layer; the LO-contract validator is the real gate for package quality.
    """
    overview_path = week_dir / f"week_{week_num:02d}_overview.html"
    if not overview_path.exists():
        return f"Week {week_num}"
    try:
        html = overview_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return f"Week {week_num}"
    m = _WEEK_OVERVIEW_H1_RE.search(html)
    if not m:
        return f"Week {week_num}"
    raw = m.group(1).strip()
    # Strip HTML entities that commonly appear in the H1 ("&mdash;").
    raw = raw.replace("&mdash;", "—").replace("&ndash;", "–")
    # Strip inner tags the H1 might carry (span wrappers, etc.).
    raw = re.sub(r"<[^>]+>", "", raw).strip()

    # Split off the "Week N Overview" prefix/suffix to isolate the real title.
    # Try "Week N Overview: Title" first.
    m2 = re.match(
        rf"(?i)^week\s+{week_num}\s*(?:overview)?\s*[:—–-]\s*(.+)$",
        raw,
    )
    if m2:
        title = m2.group(1).strip()
    else:
        m3 = re.match(
            rf"(?i)^(.+?)\s*[—–-]\s*week\s+{week_num}\s*(?:overview)?\s*$",
            raw,
        )
        if m3:
            title = m3.group(1).strip()
        else:
            title = raw

    # Bare "Overview" / empty → neutral week label (content-gen emits this
    # when no topic binds to the week).
    if not title or _BARE_OVERVIEW_RE.match(title):
        return f"Week {week_num}"
    return f"Week {week_num}: {title}"


# ---------------------------------------------------------------------------
# ED4ALL_IMSCC_MODULE_TITLES — opt-in terminal-objective module titles.
#
# With ED4ALL_WEEK_TO_GROUPS=1 and duration_weeks == num_tos, group N's COs
# are exactly TO-N's children (book order, chapter-anchored), so the org tree
# can present "Module N: <topic>" instead of the calendar-flavored "Week N".
# The mapping assumption (group ordinal N ↔ terminal_objectives[N-1]) is only
# valid under TO-membership grouping; the flag is operator-set alongside
# ED4ALL_WEEK_TO_GROUPS and is NOT verified here.
#
# Parse-with-fallback: only the exact token "to" (case-insensitive,
# whitespace-stripped) activates the mode; unset / anything else → the legacy
# "Week N" titles, byte-identical. Every failure on the objectives side
# (missing file, unparseable JSON, wrong shape, terminal_objectives not
# covering group N, empty statement) falls back SILENTLY to the legacy title
# for that group — packaging never fails over a title.
# ---------------------------------------------------------------------------
_MODULE_TITLES_ENV = "ED4ALL_IMSCC_MODULE_TITLES"
# Word-boundary truncation ceiling for the .statement fallback topic.
_MODULE_TITLE_STATEMENT_MAX = 80


def _module_titles_to_mode() -> bool:
    """True iff ED4ALL_IMSCC_MODULE_TITLES resolves to the exact token 'to'."""
    return os.environ.get(_MODULE_TITLES_ENV, "").strip().lower() == "to"


def _load_terminal_objectives_for_titles(content_dir: Path) -> List[Dict]:
    """Best-effort load of ``terminal_objectives`` for module-title mapping.

    Reads ``<export_root>/01_learning_objectives/synthesized_objectives.json``
    where ``export_root`` is the parent of ``content_dir``
    (``03_content_development``) — the same export-root convention as
    ``_resolve_objectives_source``. Accepts both the Courseforge shape
    (``terminal_objectives``) and the LibV2 archive shape
    (``terminal_outcomes``). Returns ``[]`` on ANY failure (missing file,
    I/O error, bad JSON, wrong shape) — never raises.
    """
    objectives_path = (
        content_dir.parent
        / "01_learning_objectives"
        / "synthesized_objectives.json"
    )
    try:
        if not objectives_path.exists():
            return []
        doc = json.loads(objectives_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(doc, dict):
        return []
    terminals = (
        doc.get("terminal_objectives")
        or doc.get("terminal_outcomes")
        or []
    )
    if not isinstance(terminals, list):
        return []
    return [t for t in terminals if isinstance(t, dict)]


def _module_title_from_to(
    terminals: List[Dict], week_num: int
) -> Optional[str]:
    """``"Module {N}: {topic}"`` from ``terminals[N-1]``, or ``None``.

    Topic preference: ``anchor_module_title`` (the chapter-anchored module
    title stamped by the W3 Defect-A TO derivation), falling back to a
    word-boundary-truncated ``statement``. ``None`` (→ caller uses the
    legacy "Week N" title) when the ordinal isn't covered or no usable
    topic text exists.
    """
    if week_num < 1 or week_num > len(terminals):
        return None
    to = terminals[week_num - 1]
    topic = str(to.get("anchor_module_title") or "").strip()
    if not topic:
        statement = re.sub(r"\s+", " ", str(to.get("statement") or "")).strip()
        if not statement:
            return None
        if len(statement) > _MODULE_TITLE_STATEMENT_MAX:
            head = statement[:_MODULE_TITLE_STATEMENT_MAX]
            # Cut at the last word boundary so the label reads cleanly.
            if " " in head:
                head = head.rsplit(" ", 1)[0]
            statement = head.rstrip(" ,;:.") + "…"
        topic = statement
    return f"Module {week_num}: {topic}"


# Course-overview front-matter module (Learning Objectives Map). The
# packager surfaces a top-level "Course Overview" module BEFORE the week
# items when a ``course_overview/`` directory exists in the content dir.
# Its first item is the deterministic ``learning_objectives.html`` page so a
# learner (and a retrieval / Q&A system) gets the canonical TO/CO structure
# instead of a scraped-together answer. See
# ``render_learning_objectives_page`` for the renderer.
_COURSE_OVERVIEW_DIRNAME = "course_overview"
_COURSE_OVERVIEW_TITLE = "Course Overview"
# Item ordering within the course-overview module: the objectives map first.
_COURSE_OVERVIEW_ORDER = {"learning_objectives": 0}


def _course_overview_html_files(content_dir: Path) -> List[Path]:
    """Return the course-overview module's HTML files in display order.

    Objectives map (``learning_objectives.html``) sorts first; any other
    front-matter pages follow alphabetically. Empty when the directory is
    absent — the module is then elided from the manifest entirely.
    """
    overview_dir = content_dir / _COURSE_OVERVIEW_DIRNAME
    if not overview_dir.is_dir():
        return []

    def _key(f: Path) -> Tuple[int, str]:
        for stem_key, rank in _COURSE_OVERVIEW_ORDER.items():
            if stem_key in f.stem:
                return (rank, f.name)
        return (99, f.name)

    return sorted(overview_dir.glob("*.html"), key=_key)


# ---------------------------------------------------------------------------
# W10 — Assessment surface (QTI quizzes / discussions / assignments)
# ---------------------------------------------------------------------------
#
# The pre-packaging ``assessment_synthesis`` phase writes the synthesized
# assessment XML into a ``06_assessments/`` sibling of the content dir,
# mirroring the ``course_overview/`` front-matter convention. The packager
# discovers those XML files, classifies each by IMS CC resource type, and
# emits the canonical resource + organization items so the cartridge imports
# into Brightspace / Canvas as real quizzes / discussions / assignments rather
# than inert HTML.
#
# Discovery is robust to an ABSENT ``06_assessments/`` dir (no-op,
# byte-identical to a package built without assessments).
#
# ── 06_assessments/manifest.json sidecar contract (the shape this packager
#    consumes; the synthesis-phase worker in the next wave MUST emit exactly
#    this shape) ────────────────────────────────────────────────────────────
#
#   {
#     "schema_version": "v1",                # optional; informational
#     "assessments": [
#       {
#         "file": "week_03_quiz.xml",        # REQUIRED. Relative file name
#                                            #   under 06_assessments/ (basename;
#                                            #   no path traversal). Must exist.
#         "type": "qti",                     # OPTIONAL. One of
#                                            #   "qti" | "discussion" | "assignment".
#                                            #   When omitted/blank/unknown the
#                                            #   packager infers from the XML
#                                            #   root element.
#         "title": "Week 3 Quiz",            # OPTIONAL. Human-readable nav
#                                            #   label; defaults to a Title-Cased
#                                            #   form of the file stem.
#         "week": 3,                         # OPTIONAL int. Places the item
#                                            #   under that week's <item>; when
#                                            #   absent / unmappable the item
#                                            #   lands under a top-level
#                                            #   "Assessments" org item.
#         "identifier": "RES_week_03_quiz"   # OPTIONAL. Manifest resource id;
#                                            #   defaults to a sanitized
#                                            #   RES_assessment_<stem>.
#       }
#     ]
#   }
#
# The sidecar is OPTIONAL. When absent, every ``06_assessments/*.xml`` file is
# discovered loose and classified by its XML root element:
#   * ``questestinterop``        -> qti
#   * ``topic`` (imsdt)          -> discussion
#   * ``assignment``             -> assignment
# A file whose root element is none of these (and which carries no sidecar
# ``type`` override) is skipped with a warning — never packaged as the wrong
# resource type.
_ASSESSMENTS_DIRNAME = "06_assessments"
_ASSESSMENTS_MANIFEST_NAME = "manifest.json"
_ASSESSMENTS_TITLE = "Assessments"

# IMS CC 1.3 resource-type strings (Courseforge/docs/troubleshooting.md:64-66).
_ASSESSMENT_RES_TYPE: Dict[str, str] = {
    "qti": "imsqti_xmlv1p2/imscc_xmlv1p3/assessment",
    "discussion": "imsdt_xmlv1p3",
    "assignment": "associatedcontent/imscc_xmlv1p3/learning-application-resource",
}

# Root-element → assessment-type inference (namespace-stripped local name).
_ROOT_TO_TYPE: Dict[str, str] = {
    "questestinterop": "qti",
    "topic": "discussion",
    "imsdt": "discussion",
    "assignment": "assignment",
}

# Vendored XSD filenames per assessment type (best-effort schema-validate).
_ASSESSMENT_XSD: Dict[str, str] = {
    "qti": "ccv1p3_qtiasiv1p2p1.xsd",
    "discussion": "ccv1p3_imsdt_v1p3.xsd",
    "assignment": "cc_extresource_assignmentv1p0.xsd",
}


class _Assessment:
    """A discovered + classified assessment resource (one ``06_assessments/*.xml``)."""

    __slots__ = ("path", "rel_path", "kind", "title", "week", "res_id")

    def __init__(
        self,
        *,
        path: Path,
        rel_path: str,
        kind: str,
        title: str,
        week: Optional[int],
        res_id: str,
    ) -> None:
        self.path = path
        self.rel_path = rel_path
        self.kind = kind
        self.title = title
        self.week = week
        self.res_id = res_id


def _local_root_tag(xml_path: Path) -> Optional[str]:
    """Return the namespace-stripped root element local name, or ``None``.

    Best-effort + never raises — a malformed / unreadable XML returns
    ``None`` so the caller can drop it with a warning.
    """
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    tag = root.tag
    if isinstance(tag, str) and "}" in tag:
        tag = tag.rsplit("}", 1)[1]
    return tag


def _default_assessment_res_id(stem: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", f"RES_assessment_{stem}")


def _read_assessments_sidecar(assessments_dir: Path) -> Optional[List[dict]]:
    """Read the optional ``06_assessments/manifest.json`` sidecar.

    Returns the ``assessments`` list when the sidecar exists and parses;
    ``None`` when the sidecar is absent (loose-discovery mode). A malformed
    sidecar logs a warning and returns ``None`` (fall back to loose discovery —
    never abort the package on a bad sidecar).
    """
    sidecar = assessments_dir / _ASSESSMENTS_MANIFEST_NAME
    if not sidecar.exists():
        return None
    try:
        doc = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"[assessments] WARN: malformed {_ASSESSMENTS_DIRNAME}/"
            f"{_ASSESSMENTS_MANIFEST_NAME} ({exc}); falling back to loose discovery."
        )
        return None
    entries = doc.get("assessments") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        print(
            f"[assessments] WARN: {_ASSESSMENTS_DIRNAME}/"
            f"{_ASSESSMENTS_MANIFEST_NAME} missing an 'assessments' list; "
            "falling back to loose discovery."
        )
        return None
    return entries


def _classify_assessment(
    *,
    xml_path: Path,
    assessments_dir: Path,
    declared_type: Optional[str],
    declared_title: Optional[str],
    declared_week: Optional[object],
    declared_id: Optional[str],
) -> Optional[_Assessment]:
    """Build an ``_Assessment`` for one XML file, or ``None`` to drop it.

    Resolution: the sidecar-declared ``type`` wins; else the root element is
    inferred. An unclassifiable file (unknown declared type AND unknown root
    element) is dropped with a warning.
    """
    kind = (declared_type or "").strip().lower()
    if kind not in _ASSESSMENT_RES_TYPE:
        root_tag = _local_root_tag(xml_path)
        inferred = _ROOT_TO_TYPE.get((root_tag or "").lower())
        if inferred is None:
            print(
                f"[assessments] WARN: cannot classify {xml_path.name} "
                f"(declared type={declared_type!r}, root element={root_tag!r}); "
                "skipping."
            )
            return None
        kind = inferred

    stem = xml_path.stem
    title = (declared_title or "").strip() or stem.replace("_", " ").title()

    week: Optional[int] = None
    if declared_week is not None:
        try:
            week = int(declared_week)
        except (TypeError, ValueError):
            week = None
    if week is None:
        # Infer from a ``week_NN`` token in the file name when present.
        m = re.search(r"week[_-]?(\d+)", xml_path.name, re.IGNORECASE)
        if m:
            try:
                week = int(m.group(1))
            except ValueError:
                week = None

    res_id = re.sub(
        r"[^a-zA-Z0-9_]", "_", (declared_id or "").strip()
    ) or _default_assessment_res_id(stem)

    rel_path = f"{_ASSESSMENTS_DIRNAME}/{xml_path.name}"
    return _Assessment(
        path=xml_path,
        rel_path=rel_path,
        kind=kind,
        title=title,
        week=week,
        res_id=res_id,
    )


def _discover_assessments(content_dir: Path) -> Tuple[List[_Assessment], int]:
    """Discover + classify assessment XML in the ``06_assessments/`` sibling.

    Returns ``(classified, authored_count)`` where ``classified`` is the list
    of successfully-classified ``_Assessment`` objects (sorted by (week, file
    name) so emit order is deterministic) and ``authored_count`` is the number
    of candidate assessment files the synthesis phase intended to ship — every
    sidecar entry that names an existing file, or (loose mode) every
    ``*.xml`` under the dir. A file that fails CLASSIFICATION (unknown declared
    type AND unparseable / unknown root element) counts toward ``authored`` but
    not toward ``classified`` — so the coverage sidecar surfaces it as a drop
    rather than hiding it.

    When the directory is absent returns ``([], 0)`` — the no-op / byte-
    identical path.
    """
    assessments_dir = content_dir / _ASSESSMENTS_DIRNAME
    if not assessments_dir.is_dir():
        # Pipeline layout: the ``assessment_synthesis`` phase writes the
        # ``06_assessments/`` dir as a SIBLING of the content dir — in a real
        # run ``content_dir`` is ``<export>/03_content_development`` and the
        # synthesized assessments live at ``<export>/06_assessments``. Fall
        # back to the sibling when the child layout (used by the standalone
        # CLI + the packager's own fixtures) is absent. Still ``([], 0)`` when
        # neither exists — the no-op / byte-identical path.
        sibling_dir = content_dir.parent / _ASSESSMENTS_DIRNAME
        if sibling_dir.is_dir():
            assessments_dir = sibling_dir
        else:
            return [], 0

    sidecar_entries = _read_assessments_sidecar(assessments_dir)
    out: List[_Assessment] = []
    seen: set = set()
    authored = 0

    if sidecar_entries is not None:
        for entry in sidecar_entries:
            if not isinstance(entry, dict):
                continue
            fname = str(entry.get("file") or "").strip()
            if not fname:
                continue
            # Basename-only — defend against path traversal in the sidecar.
            fname = Path(fname).name
            xml_path = assessments_dir / fname
            if not xml_path.is_file():
                print(
                    f"[assessments] WARN: sidecar references missing file "
                    f"{fname}; skipping."
                )
                continue
            authored += 1
            assessment = _classify_assessment(
                xml_path=xml_path,
                assessments_dir=assessments_dir,
                declared_type=entry.get("type"),
                declared_title=entry.get("title"),
                declared_week=entry.get("week"),
                declared_id=entry.get("identifier"),
            )
            if assessment is not None and assessment.path not in seen:
                out.append(assessment)
                seen.add(assessment.path)
    else:
        for xml_path in sorted(assessments_dir.glob("*.xml")):
            if not xml_path.is_file():
                continue
            authored += 1
            assessment = _classify_assessment(
                xml_path=xml_path,
                assessments_dir=assessments_dir,
                declared_type=None,
                declared_title=None,
                declared_week=None,
                declared_id=None,
            )
            if assessment is not None and assessment.path not in seen:
                out.append(assessment)
                seen.add(assessment.path)

    out.sort(key=lambda a: (a.week if a.week is not None else 10**9, a.path.name))
    return out, authored


def _validate_assessment_xsd(assessment: _Assessment) -> bool:
    """Best-effort XSD-validate one assessment XML against its vendored schema.

    Returns ``True`` when valid OR when validation cannot run (``lxml`` absent /
    XSD file missing) — the schema check is WARNING-only (mirrors the
    objectives-map validate). Returns ``False`` ONLY when ``lxml`` is present,
    the XSD loads, and the document is genuinely malformed / schema-invalid —
    so the caller drops the item rather than packaging a cartridge that fails
    to import.
    """
    try:
        from lxml import etree  # type: ignore
    except ImportError:
        return True  # graceful degrade — no lxml, skip the schema dimension.

    xsd_name = _ASSESSMENT_XSD.get(assessment.kind)
    if not xsd_name:
        return True
    xsd_path = _HERE.parent / "schemas" / "imscc" / xsd_name
    if not xsd_path.exists():
        print(
            f"[assessments] WARN: vendored XSD {xsd_name} not found; "
            f"skipping schema validation for {assessment.path.name}."
        )
        return True

    try:
        schema = etree.XMLSchema(etree.parse(str(xsd_path)))
        doc = etree.parse(str(assessment.path))
    except etree.XMLSyntaxError as exc:
        print(
            f"[assessments] WARN: {assessment.path.name} is not well-formed XML "
            f"({exc}); dropping from package."
        )
        return False
    except (etree.XMLSchemaParseError, OSError) as exc:
        # The XSD itself failed to load (e.g. unresolved imports) — degrade to
        # well-formed-only rather than dropping a genuinely fine document.
        print(
            f"[assessments] WARN: could not load XSD {xsd_name} ({exc}); "
            f"skipping schema validation for {assessment.path.name}."
        )
        return True

    if not schema.validate(doc):
        reason = schema.error_log.last_error if schema.error_log else "schema-invalid"
        print(
            f"[assessments] WARN: {assessment.path.name} failed XSD validation "
            f"against {xsd_name} ({reason}); dropping from package."
        )
        return False
    return True


def _resolve_objectives_source(
    content_dir: Path, objectives_path: Optional[Path]
) -> Optional[Path]:
    """Pick the richest objectives JSON to feed the objectives-map renderer.

    Preference order:
      1. ``<project_root>/01_learning_objectives/synthesized_objectives.json``
         (full statements + Bloom + sections), where ``project_root`` is the
         parent of ``content_dir`` (``03_content_development``).
      2. The explicitly-resolved ``objectives_path`` (e.g. an auto-discovered
         ``course.json`` projection, which retains the same key shape).
      3. ``content_dir / "course.json"``.

    Returns the first that exists, or ``None``.
    """
    candidates: List[Path] = []
    synthesized = (
        content_dir.parent
        / "01_learning_objectives"
        / "synthesized_objectives.json"
    )
    candidates.append(synthesized)
    if objectives_path is not None:
        candidates.append(objectives_path)
    candidates.append(content_dir / "course.json")
    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue
    return None


def _maybe_render_objectives_page(
    *,
    content_dir: Path,
    objectives_path: Optional[Path],
    course_code: str,
    course_title: str,
) -> Optional[Path]:
    """Render the Learning Objectives Map page into ``course_overview/``.

    Best-effort + non-destructive:
      * Skips silently when no objectives JSON can be resolved.
      * Never overwrites an author-provided
        ``course_overview/learning_objectives.html``.
      * Any renderer/IO error is logged to stdout and swallowed — the
        objectives map is additive course content, not a build gate.

    Returns the written path, or ``None`` when skipped.
    """
    out_path = content_dir / _COURSE_OVERVIEW_DIRNAME / "learning_objectives.html"
    if out_path.exists():
        print(
            "[objectives-map] existing course_overview/learning_objectives"
            ".html found; preserving author copy (no overwrite)."
        )
        return out_path

    source = _resolve_objectives_source(content_dir, objectives_path)
    if source is None:
        print(
            "[objectives-map] no synthesized_objectives.json / course.json "
            "resolved; skipping Course Overview objectives map."
        )
        return None

    try:
        from render_learning_objectives_page import (
            write_learning_objectives_page,
        )
    except ImportError as exc:  # pragma: no cover - import wiring
        print(f"[objectives-map] WARN renderer unavailable: {exc}")
        return None

    try:
        write_learning_objectives_page(
            source,
            out_path,
            course_code=course_code,
            course_title=course_title,
        )
    except Exception as exc:  # noqa: BLE001 - additive content, never a gate
        print(
            f"[objectives-map] WARN failed to render objectives map from "
            f"{source}: {exc}"
        )
        return None
    print(f"[objectives-map] rendered Course Overview objectives map from {source.name}")
    return out_path


def build_manifest(
    content_dir: Path,
    course_code: str,
    course_title: str,
    *,
    outline_only: bool = False,
    assessments: Optional[List["_Assessment"]] = None,
) -> str:
    """Build imsmanifest.xml for multi-file weekly content.

    Phase 2 (Subtask 29):
      * ``outline_only`` — when ``True``, the per-week ``html_files`` walk
        filters to only ``*overview.html`` and ``*summary.html`` pages
        (drops content / application / self_check / discussion). The
        general LOM ``<description>`` text is also tagged with the prefix
        ``"[OUTLINE] "`` so LMS-side viewers see at a glance that the
        package carries outline-tier content only. The companion
        ``course_metadata.json`` stub augmentation (``blocks_summary
        .outline_only=true``) is written upstream by
        ``generate_course.py`` (Subtask 28); this function only consumes
        / surfaces that signal in the manifest.
    """
    ns = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
    lom_ns = "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource"
    lom_manifest_ns = "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"

    # Register namespaces for clean serialization
    ET.register_namespace("", ns)
    ET.register_namespace("lom", lom_ns)
    ET.register_namespace("lomimscc", lom_manifest_ns)

    # Helper to create elements in the default IMSCC namespace
    def cc(tag):
        return f"{{{ns}}}{tag}"

    def lm(tag):
        return f"{{{lom_manifest_ns}}}{tag}"

    manifest = ET.Element(cc("manifest"), {
        "identifier": f"{course_code}_manifest",
    })

    # Metadata
    metadata = ET.SubElement(manifest, cc("metadata"))
    ET.SubElement(metadata, cc("schema")).text = "IMS Common Cartridge"
    ET.SubElement(metadata, cc("schemaversion")).text = "1.3.0"
    lom_el = ET.SubElement(metadata, lm("lom"))
    general = ET.SubElement(lom_el, lm("general"))
    title_el = ET.SubElement(general, lm("title"))
    ET.SubElement(title_el, lm("string"), {"language": "en"}).text = f"{course_code}: {course_title}"
    desc_el = ET.SubElement(general, lm("description"))
    description_text = (
        "A 12-week graduate course covering learning theory, instructional design, "
        "cognitive load, blended teaching, assessment, and accessibility."
    )
    # Phase 2 (Subtask 29): tag the LOM description with an `[OUTLINE] `
    # prefix in outline-only packaging so LMS-side viewers can detect the
    # outline-tier deliverable shape without parsing
    # course_metadata.json::blocks_summary.
    if outline_only:
        description_text = "[OUTLINE] " + description_text
    ET.SubElement(desc_el, lm("string"), {"language": "en"}).text = description_text

    # Organizations
    organizations = ET.SubElement(manifest, cc("organizations"))
    org = ET.SubElement(organizations, cc("organization"), {
        "identifier": "ORG_1",
        "structure": "rooted-hierarchy",
    })
    root_item = ET.SubElement(org, cc("item"), {"identifier": "ROOT"})
    ET.SubElement(root_item, cc("title")).text = f"{course_code}: {course_title}"

    # Resources
    resources = ET.SubElement(manifest, cc("resources"))

    # Course-overview front-matter module (Learning Objectives Map).
    # Emitted BEFORE the week items so the LMS week list opens with a
    # "Course Overview" module whose first page is the objectives map.
    # Honoured in outline-only mode too (the objectives map IS outline-tier
    # content). Absent ``course_overview/`` directory → no module emitted.
    overview_files = _course_overview_html_files(content_dir)
    if overview_files:
        overview_item = ET.SubElement(
            root_item, cc("item"), {"identifier": "COURSE_OVERVIEW"}
        )
        ET.SubElement(overview_item, cc("title")).text = _COURSE_OVERVIEW_TITLE
        for html_file in overview_files:
            rel_path = f"{_COURSE_OVERVIEW_DIRNAME}/{html_file.name}"
            res_id = re.sub(
                r"[^a-zA-Z0-9_]",
                "_",
                f"RES_{_COURSE_OVERVIEW_DIRNAME}_{html_file.stem}",
            )
            file_item = ET.SubElement(overview_item, cc("item"), {
                "identifier": f"ITEM_{res_id}",
                "identifierref": res_id,
            })
            title_text = html_file.stem.replace("_", " ").title()
            ET.SubElement(file_item, cc("title")).text = title_text
            resource = ET.SubElement(resources, cc("resource"), {
                "identifier": res_id,
                "type": "webcontent",
                "href": rel_path,
            })
            ET.SubElement(resource, cc("file"), {"href": rel_path})

    # Walk week groups in order. ``_iter_week_groups`` yields a coherent
    # ``(week_name, week_num, title_dir, html_files)`` view over BOTH the
    # nested ``week_NN/*.html`` layout (legacy single-pass) and the FLAT
    # ``week_NN_*.html`` layout (two-pass rewrite tier) — the flat layout
    # previously produced a 1-page IMSCC because the bare ``glob("week_*")``
    # + ``is_dir()`` filter skipped every flat page.
    # Map week_num → its organization <item> so W10 assessment items can be
    # nested under the right week (or fall back to a top-level "Assessments"
    # item when no week mapping resolves).
    week_items_by_num: Dict[int, ET.Element] = {}

    # ED4ALL_IMSCC_MODULE_TITLES=to — load the terminal objectives ONCE for
    # the whole walk; [] (flag off / any load failure) keeps every group on
    # the legacy "Week N" title path, byte-identical.
    to_title_terminals: List[Dict] = (
        _load_terminal_objectives_for_titles(content_dir)
        if _module_titles_to_mode()
        else []
    )

    for week_name, week_num_int, title_dir, html_files in _iter_week_groups(
        content_dir
    ):
        week_num = str(week_num_int)
        week_id = f"WEEK_{week_num}"

        week_item = ET.SubElement(root_item, cc("item"), {"identifier": week_id})
        week_items_by_num[week_num_int] = week_item
        # Prefer the real chapter title captured by generate_week in the
        # overview H1 (e.g. "Week 1: Introduction to Core Concepts")
        # over the bare "Week N" label that earlier revisions emitted and
        # that produced an uninformative LMS week list. Under
        # ED4ALL_IMSCC_MODULE_TITLES=to the group is instead titled from
        # its terminal objective ("Module N: <topic>", no calendar concept);
        # any per-group miss falls back to the legacy title.
        week_title: Optional[str] = None
        if to_title_terminals:
            week_title = _module_title_from_to(to_title_terminals, week_num_int)
        if week_title is None:
            week_title = _extract_week_title(title_dir, week_num_int)
        ET.SubElement(week_item, cc("title")).text = week_title

        # Phase 2 (Subtask 29): in outline-only mode, drop every page
        # except the overview + summary deliverables (the outline-tier
        # surfaces). Content / application / self_check / discussion
        # pages are excluded from BOTH the manifest organization tree
        # and the resources section so the IMSCC payload itself stays
        # outline-shaped (no orphan resources, no ITEM entries pointing
        # at suppressed pages).
        if outline_only:
            html_files = [
                f for f in html_files
                if f.name.endswith("overview.html") or f.name.endswith("summary.html")
            ]

        for html_file in html_files:
            rel_path = f"{week_name}/{html_file.name}"
            res_id = re.sub(r"[^a-zA-Z0-9_]", "_", f"RES_{week_name}_{html_file.stem}")

            file_item = ET.SubElement(week_item, cc("item"), {
                "identifier": f"ITEM_{res_id}",
                "identifierref": res_id,
            })
            title_text = html_file.stem.replace(f"{week_name}_", "").replace("_", " ").title()
            ET.SubElement(file_item, cc("title")).text = title_text

            resource = ET.SubElement(resources, cc("resource"), {
                "identifier": res_id,
                "type": "webcontent",
                "href": rel_path,
            })
            ET.SubElement(resource, cc("file"), {"href": rel_path})

    # W10 — assessment surface (QTI quizzes / discussions / assignments).
    # Each discovered assessment gets a manifest <resource> carrying its
    # canonical IMS CC resource type + an organization <item> under its week
    # (or a top-level "Assessments" item when no week maps). Honoured in
    # full mode only — outline-only packages strip every non-overview page,
    # and an assessment surface is not an outline-tier deliverable.
    if assessments and not outline_only:
        assessments_item: Optional[ET.Element] = None  # lazy top-level fallback
        for assessment in assessments:
            res_type = _ASSESSMENT_RES_TYPE.get(assessment.kind)
            if res_type is None:  # defensive; classifier never yields others
                continue
            parent_item = (
                week_items_by_num.get(assessment.week)
                if assessment.week is not None
                else None
            )
            if parent_item is None:
                if assessments_item is None:
                    assessments_item = ET.SubElement(
                        root_item, cc("item"), {"identifier": "ASSESSMENTS"}
                    )
                    ET.SubElement(
                        assessments_item, cc("title")
                    ).text = _ASSESSMENTS_TITLE
                parent_item = assessments_item

            item_el = ET.SubElement(parent_item, cc("item"), {
                "identifier": f"ITEM_{assessment.res_id}",
                "identifierref": assessment.res_id,
            })
            ET.SubElement(item_el, cc("title")).text = assessment.title

            resource = ET.SubElement(resources, cc("resource"), {
                "identifier": assessment.res_id,
                "type": res_type,
                "href": assessment.rel_path,
            })
            ET.SubElement(resource, cc("file"), {"href": assessment.rel_path})

    ET.indent(manifest, space="  ")
    return ET.tostring(manifest, encoding="unicode", xml_declaration=True)


def validate_content_objectives(
    content_dir: Path, objectives_path: Path
) -> Tuple[bool, List[str]]:
    """Run `validate_page_objectives.validate_page` on every week_*/*.html page.

    Returns ``(ok, failure_messages)``. On success the failure list is empty.
    Pages without a JSON-LD block are passed over silently (validator's own
    rule). Imported lazily so packaging without --objectives incurs no cost.
    """
    from validate_page_objectives import (
        discover_html_pages,
        load_canonical_objectives,
        validate_page,
    )

    canonical = load_canonical_objectives(objectives_path)
    pages = discover_html_pages(content_dir)
    failures: List[str] = []
    for page in pages:
        # Only validate week_* pages; project docs and non-week HTML aren't
        # expected to carry LO metadata.
        if not any(part.startswith("week_") for part in page.parts):
            continue
        ok, msg = validate_page(page, canonical)
        if not ok:
            failures.append(msg)
    return (not failures, failures)


def package_imscc(
    content_dir: Path,
    output_path: Path,
    course_code: str,
    course_title: str,
    *,
    objectives_path: Optional[Path] = None,
    skip_validation: bool = False,
    outline_only: bool = False,
    coverage_sidecar_path: Optional[Path] = None,
    emit_objectives_page: bool = True,
):
    """Create the IMSCC zip package.

    Per-week learningObjectives validation runs by default (Wave 2, Worker L
    — REC-CTR-03). Resolution order for the objectives file:

    1. Explicit ``objectives_path`` argument (CLI ``--objectives PATH``).
    2. Auto-discovery: ``content_dir / "course.json"`` if it exists.
    3. None available → log a warning and skip validation (backward-compat
       for callers that never wired the flag).

    ``skip_validation=True`` (CLI ``--skip-validation``) is an explicit
    opt-out that bypasses validation even when an objectives file is
    available. Hard-fail (``SystemExit(2)``) only occurs on a genuine
    validation FAILURE — never on a missing objectives file alone.

    Phase 2 (Subtask 29):
      * ``outline_only`` — when ``True``, the per-week zip walk filters to
        only ``*overview.html`` and ``*summary.html`` pages so the IMSCC
        payload mirrors the manifest organization tree (no orphan
        resources). The manifest description gets a ``[OUTLINE] `` prefix.
        The companion ``course_metadata.json`` augmentation is upstream
        (``generate_course.py --emit-mode outline``); this packager only
        consumes / surfaces the marker.

    W3.H sub-task H3:
      * ``coverage_sidecar_path`` — when provided, write a canonical
        ``packaging_report.json`` sidecar (per
        ``schemas/library/packaging_report.schema.json``) carrying the
        ``source_coverage`` block that records pages_authored vs
        pages_packaged plus the per-reason exclusion histogram
        (``missing_lo`` / ``gate_block`` / ``outline_filter``). When the
        per-week LO contract validator fails (``SystemExit(2)``), the
        sidecar is still emitted before raising so a downstream
        consumer (W3.G master aggregator) sees the failure attribution.
    """
    # Coverage tracking — initialised pre-validation so a hard-fail
    # path can still emit the sidecar with attribution.
    _coverage_pages_authored = 0
    _coverage_drop_missing_lo = 0
    _coverage_drop_gate_block = 0
    _coverage_drop_outline_filter = 0
    # W10 — assessment coverage (authored vs packaged); set during discovery.
    _assessment_coverage: Optional[dict] = None

    def _emit_coverage_sidecar(*, pages_packaged: int) -> None:
        """Best-effort write of the W3.H H3 packaging_report sidecar.

        Failure path: log + continue. The sidecar is observability,
        not a build gate; the package_path itself is the source of
        truth for IMSCC integrity.
        """
        if coverage_sidecar_path is None:
            return
        try:
            from lib.governance.source_coverage import build_source_coverage
        except Exception as exc:  # noqa: BLE001
            print(f"[coverage] WARN: source_coverage helper unavailable: {exc}")
            return
        drops: dict = {}
        if _coverage_drop_missing_lo:
            drops["missing_lo"] = _coverage_drop_missing_lo
        if _coverage_drop_gate_block:
            drops["gate_block"] = _coverage_drop_gate_block
        if _coverage_drop_outline_filter:
            drops["outline_filter"] = _coverage_drop_outline_filter
        block = build_source_coverage(
            consumed_count=_coverage_pages_authored,
            emitted_count=pages_packaged,
            drop_reasons=drops,
            dropped_count=max(0, _coverage_pages_authored - pages_packaged),
            label="package_imscc",
        )
        report = {
            "schema_version": "v1",
            "course_code": course_code,
            "course_title": course_title,
            "package_path": str(output_path),
            "outline_only": outline_only,
            "source_coverage": block,
        }
        # W10 — assessment coverage block (authored vs packaged), mirroring
        # the source_coverage shape. Only present when a 06_assessments/ dir
        # was discovered (None on the byte-identical no-assessments path).
        if _assessment_coverage is not None:
            report["assessment_coverage"] = _assessment_coverage
        try:
            coverage_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            coverage_sidecar_path.write_text(
                _json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(
                f"[coverage] WARN: failed to write packaging_report sidecar "
                f"{coverage_sidecar_path}: {exc}"
            )
    # Auto-discover objectives if not explicitly provided (default-on behavior).
    if objectives_path is None and not skip_validation:
        candidate = content_dir / "course.json"
        if candidate.exists():
            objectives_path = candidate
            print(f"[validate] Auto-discovered objectives at {candidate}")

    # W3.H H3: count pages_authored = every week_*/*.html page on disk
    # BEFORE any filtering. This is the upstream denominator for the
    # source_coverage block; pages_packaged (computed below) is the
    # numerator. Running this scan here means the validator-failure
    # short-circuit below still gets attribution.
    _pre_walk_pages = []
    for _wname, _wnum, _tdir, _wfiles in _iter_week_groups(content_dir):
        _pre_walk_pages.extend(_wfiles)
    _coverage_pages_authored = len(_pre_walk_pages)

    if skip_validation:
        print("[validate] SKIPPED (per --skip-validation) — build will not be gated on LO correctness.")
    elif objectives_path is None:
        print(
            "[validate] WARNING: no objectives file found; skipping LO validation. "
            "Pass --objectives or place course.json at content root to enable."
        )
    else:
        print(f"[validate] Checking per-week learningObjectives against {objectives_path.name}...")
        ok, failures = validate_content_objectives(content_dir, objectives_path)
        if not ok:
            print(f"[validate] REFUSING TO PACKAGE — {len(failures)} page(s) violate per-week LO contract:")
            for msg in failures:
                print(f"  - {msg}")
            print("Fix the offending pages (or re-run generate_course.py with --objectives) then retry.")
            print("Override with --skip-validation if you really know what you're doing.")
            # W3.H H3: classify each failure. A failure message that
            # mentions "no learningObjectives" / "missing learning"
            # buckets to ``missing_lo``; everything else (out-of-week
            # IDs, malformed JSON-LD) buckets to ``gate_block``.
            for _msg in failures:
                _lower = _msg.lower()
                if "no learningobjectives" in _lower or "missing learning" in _lower:
                    _coverage_drop_missing_lo += 1
                else:
                    _coverage_drop_gate_block += 1
            # On hard-fail, no pages get packaged — emit the coverage
            # sidecar with pages_packaged=0 so the master aggregator
            # sees the attribution before SystemExit propagates.
            _emit_coverage_sidecar(pages_packaged=0)
            raise SystemExit(2)
        print("[validate] All week pages pass per-week LO contract.")

    # Course-overview Learning Objectives Map (default-on). Render the
    # deterministic objectives page into ``content_dir/course_overview/``
    # BEFORE the manifest is built so the new "Course Overview" module +
    # its resource land in the organization tree and the zip payload. The
    # richest objectives source wins: a sibling
    # ``01_learning_objectives/synthesized_objectives.json`` (full
    # statements + Bloom + sections) is preferred over the resolved
    # ``objectives_path`` / auto-discovered ``course.json``. When neither
    # is available, the module is silently skipped (no fabrication). An
    # author-provided ``course_overview/learning_objectives.html`` already
    # on disk is never overwritten (operator override).
    if emit_objectives_page:
        _maybe_render_objectives_page(
            content_dir=content_dir,
            objectives_path=objectives_path,
            course_code=course_code,
            course_title=course_title,
        )

    # W10 — discover + classify + (best-effort) XSD-validate the assessment
    # surface BEFORE the manifest is built so the QTI/discussion/assignment
    # resources + organization items land in the manifest and the XML files
    # land in the zip payload. Absent ``06_assessments/`` → no-op, byte-
    # identical to a package built without assessments. Malformed items are
    # logged + dropped (never abort the package). Skipped in outline-only mode
    # (an assessment surface is not an outline-tier deliverable).
    packaged_assessments: List[_Assessment] = []
    if not outline_only:
        discovered, authored = _discover_assessments(content_dir)
        if authored:
            for assessment in discovered:
                if _validate_assessment_xsd(assessment):
                    packaged_assessments.append(assessment)
            packaged = len(packaged_assessments)
            by_type: Dict[str, int] = {}
            for a in packaged_assessments:
                by_type[a.kind] = by_type.get(a.kind, 0) + 1
            _assessment_coverage = {
                "assessments_authored": authored,
                "assessments_packaged": packaged,
                "assessments_dropped": max(0, authored - packaged),
                "by_type": by_type,
            }
            print(
                f"[assessments] discovered {authored} assessment XML file(s); "
                f"packaged {packaged} ({by_type})."
            )

    manifest_xml = build_manifest(
        content_dir, course_code, course_title, outline_only=outline_only,
        assessments=packaged_assessments,
    )

    stub_included = False

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest_xml)

        # REC-TAX-01 cleanup (Wave 3, Worker M): bundle Worker J's
        # course_metadata.json classification stub at the zip root when
        # present. Trainforge consume already supports both zip-root and
        # sibling paths, but zip-root is the canonical self-contained
        # delivery — this closes the Wave 2 integration gap. Additive
        # only; absence is a no-op for backward-compat.
        stub_path = content_dir / "course_metadata.json"
        if stub_path.exists():
            zf.write(stub_path, stub_path.name)
            stub_included = True

        file_count = 0

        # Course-overview front-matter module (Learning Objectives Map):
        # mirror the manifest emission so the zip payload matches the
        # organization tree (no orphan resources). Emitted in both full and
        # outline-only modes — the objectives map is outline-tier content.
        for html_file in _course_overview_html_files(content_dir):
            zf.write(
                html_file,
                f"{_COURSE_OVERVIEW_DIRNAME}/{html_file.name}",
            )
            file_count += 1

        # Zip every week page under a ``week_NN/`` prefix so the archive
        # paths match the manifest org tree built above — for BOTH nested
        # (real subdir) and flat (``week_NN_*.html`` directly under
        # content_dir) layouts. ``_iter_week_groups`` returns the same
        # ordering the manifest used, so rel_paths line up exactly.
        for week_name, _wnum, _tdir, html_files in _iter_week_groups(
            content_dir
        ):
            for html_file in html_files:
                # Phase 2 (Subtask 29): mirror the manifest filter so the
                # zip payload matches the organization tree (no orphan
                # resources / no resources missing from manifest).
                if outline_only and not (
                    html_file.name.endswith("overview.html")
                    or html_file.name.endswith("summary.html")
                ):
                    # W3.H H3: outline-only filtering is a known
                    # exclusion class — track separately from
                    # gate-driven drops so the master aggregator can
                    # tell intentional outline pruning from quality
                    # failures.
                    _coverage_drop_outline_filter += 1
                    continue
                zf.write(html_file, f"{week_name}/{html_file.name}")
                file_count += 1

        # W10 — write each packaged assessment XML under ``06_assessments/``
        # so the zip payload matches the manifest <resource href> (the rel_path
        # the resource emit used). Mirrors the per-week page write above.
        for assessment in packaged_assessments:
            zf.write(assessment.path, assessment.rel_path)

    print(f"IMSCC created: {output_path}")
    if stub_included:
        total = file_count + 2
        print(
            f"  Files: {file_count} HTML + 1 manifest + 1 course_metadata.json "
            f"= {total} total"
        )
    else:
        total = file_count + 1
        print(f"  Files: {file_count} HTML + 1 manifest = {total} total")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

    # W3.H H3: emit the canonical packaging_report sidecar (no-op
    # when ``coverage_sidecar_path`` was not provided). pages_packaged
    # = file_count after all filters; pages_authored was captured
    # pre-validation; drop_reasons accumulated during the walk.
    _emit_coverage_sidecar(pages_packaged=file_count)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("content_dir", type=Path, help="Course content dir containing week_* subdirs")
    p.add_argument("output_imscc", type=Path, help="Output .imscc file path")
    p.add_argument("course_code", nargs="?", default="SAMPLE_101", help="Course code (default: SAMPLE_101)")
    p.add_argument("course_title", nargs="?", default="Sample Course",
                   help="Course title (default: Sample Course)")
    p.add_argument("--objectives", type=Path, default=None,
                   help=("Canonical objectives JSON to validate per-week LO "
                         "specificity before packaging. If omitted, auto-"
                         "discovered at <content_dir>/course.json when present."))
    p.add_argument("--skip-validation", action="store_true",
                   help=("Opt out of per-week LO validation (not recommended "
                         "for production builds)."))
    p.add_argument(
        "--no-objectives-page",
        action="store_true",
        help=(
            "Opt out of the default-on Course Overview Learning Objectives "
            "Map page. By default the packager renders a deterministic "
            "course_overview/learning_objectives.html from the course's "
            "synthesized objectives and adds a 'Course Overview' module to "
            "the IMSCC organization. Use this flag to suppress it."
        ),
    )
    p.add_argument(
        "--outline-only",
        action="store_true",
        help=(
            "Phase 2 (Subtask 29): package only outline-tier deliverables "
            "(overview + summary pages per week). The IMSCC manifest "
            "description gets an `[OUTLINE] ` prefix; content / "
            "application / self_check / discussion pages are dropped from "
            "both the manifest organization tree and the zip payload. "
            "Pair with `generate_course.py --emit-mode outline` upstream "
            "so course_metadata.json::blocks_summary.outline_only=true "
            "is bundled into the package."
        ),
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.output_imscc.parent.mkdir(parents=True, exist_ok=True)
    package_imscc(
        args.content_dir, args.output_imscc,
        args.course_code, args.course_title,
        objectives_path=args.objectives,
        skip_validation=args.skip_validation,
        outline_only=args.outline_only,
        emit_objectives_page=not args.no_objectives_page,
    )
