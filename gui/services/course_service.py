"""Course / topics service for the Ed4All control-plane GUI.

Real reads/writes against two on-disk corpora — no stubs:

* **Courseforge project exports** —
  ``Courseforge/exports/PROJ-*/project_config.json`` (project_id, course_name,
  duration_weeks, credit_hours, status, objectives_path,
  synthesized_objectives_path) + the synthesized objectives doc at
  ``01_learning_objectives/synthesized_objectives.json`` (course_name,
  mint_method, duration_weeks, learning_outcomes[], terminal_objectives[],
  chapter_objectives[]).
* **LibV2 courses** — ``LibV2/courses/<slug>/manifest.json`` (the
  ``classification`` block: division / primary_domain / secondary_domains /
  subdomains / topics / subtopics / tags) + a sibling ``objectives.json`` in the
  LibV2 archive form (terminal_outcomes[] / component_objectives[]).

This module is import-safe WITHOUT fastapi — the router (which owns the web
deps) wraps these functions. Atomic writes reuse
``gui.shared_state._atomic_write_json`` (tmpfile + ``os.replace``).

A ``course_id`` accepted by the read/write helpers may be a Courseforge
``project_id`` (``PROJ-...``), a ``course_name`` (e.g. ``OPENSTAX_ALG_9``), or a
LibV2 ``slug`` (e.g. ``openstax-alg-9``); resolution searches both corpora.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.ontology.learning_objectives import validate_lo_id
from lib.paths import COURSEFORGE_PATH
from lib.paths import LIBV2_PATH as _DEFAULT_LIBV2_PATH

# The reuse-compatible mint_method that ``ed4all run --reuse-objectives`` accepts
# for a user-authored objectives JSON (matches the CLAUDE.md contract).
USER_SUPPLIED_MINT_METHOD = "user_supplied_objectives_json"


# --------------------------------------------------------------------------- #
# Corpus roots
# --------------------------------------------------------------------------- #


def _libv2_root() -> Path:
    """Resolve the LibV2 root, honoring ``ED4ALL_LIBV2_ROOT`` at call time.

    Mirrors ``MCP/tools/pipeline_tools.py::_resolve_libv2_root`` priority:
    ``ED4ALL_LIBV2_ROOT`` env var > the in-tree ``lib.paths.LIBV2_PATH`` default.
    Read at call time (not import) so tests can monkeypatch the env var.
    """
    env_val = os.environ.get("ED4ALL_LIBV2_ROOT", "").strip()
    if env_val:
        return Path(env_val)
    return Path(_DEFAULT_LIBV2_PATH)


def _exports_root() -> Path:
    """Return ``Courseforge/exports/`` (the Courseforge project-export root)."""
    return Path(COURSEFORGE_PATH) / "exports"


def _libv2_courses_root() -> Path:
    """Return ``<libv2_root>/courses/``."""
    return _libv2_root() / "courses"


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read + parse a JSON object; return ``None`` on missing / malformed."""
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(target: Path, payload: Any) -> None:
    """Atomic JSON write — reuse the shared_state idiom (tmpfile + os.replace).

    Imported lazily so this module stays importable even if a future
    shared_state edit pulls in a heavier dep; the function itself is web-free.
    """
    from gui.shared_state import _atomic_write_json as _shared_atomic

    _shared_atomic(target, payload)


# --------------------------------------------------------------------------- #
# Courseforge export discovery
# --------------------------------------------------------------------------- #


def _iter_export_dirs() -> List[Path]:
    """Yield every ``Courseforge/exports/PROJ-*`` directory (newest-name last).

    Sorted by name so timestamp-suffixed re-runs of the same course are in
    chronological order; the *latest* export for a course_name wins on collision.
    """
    root = _exports_root()
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("PROJ-")
    )


def _objectives_path_for_export(export_dir: Path, config: Dict[str, Any]) -> Path:
    """Resolve the synthesized objectives JSON for an export.

    Prefers the explicit ``synthesized_objectives_path`` / ``objectives_path``
    fields in ``project_config.json``; falls back to the canonical on-disk
    location ``01_learning_objectives/synthesized_objectives.json``.
    """
    for key in ("synthesized_objectives_path", "objectives_path"):
        raw = config.get(key)
        if isinstance(raw, str) and raw.strip():
            return Path(raw)
    return export_dir / "01_learning_objectives" / "synthesized_objectives.json"


def _export_records() -> List[Dict[str, Any]]:
    """Return one record per Courseforge export that has a project_config.json.

    Each record: ``{export_dir, config, project_id, course_name, objectives_path}``.
    """
    out: List[Dict[str, Any]] = []
    for export_dir in _iter_export_dirs():
        config = _read_json(export_dir / "project_config.json")
        if config is None:
            continue
        project_id = str(config.get("project_id") or export_dir.name)
        out.append(
            {
                "export_dir": export_dir,
                "config": config,
                "project_id": project_id,
                "course_name": config.get("course_name"),
                "objectives_path": _objectives_path_for_export(export_dir, config),
            }
        )
    return out


def _latest_export_by_course_name(course_name: str) -> Optional[Dict[str, Any]]:
    """Return the newest export record whose ``course_name`` matches (case-fold)."""
    target = course_name.strip().casefold()
    match: Optional[Dict[str, Any]] = None
    for rec in _export_records():  # already name-sorted ascending
        cn = rec.get("course_name")
        if isinstance(cn, str) and cn.casefold() == target:
            match = rec  # keep last (newest timestamp suffix)
    return match


# --------------------------------------------------------------------------- #
# LibV2 course discovery
# --------------------------------------------------------------------------- #


def _iter_libv2_dirs() -> List[Path]:
    """Yield every ``<libv2_root>/courses/<slug>/`` directory (sorted)."""
    root = _libv2_courses_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _libv2_display_name(slug: str, manifest: Optional[Dict[str, Any]]) -> str:
    """Best-effort human course name for a LibV2 course.

    Looks at ``course_name`` / ``title`` then the embedded
    ``sourceforge_manifest`` (``course_title`` / ``course_id``); falls back to
    the slug itself. Never invents data — only reads what is on disk.
    """
    if manifest:
        for key in ("course_name", "title"):
            val = manifest.get(key)
            if isinstance(val, str) and val.strip():
                return val
        sf = manifest.get("sourceforge_manifest")
        if isinstance(sf, dict):
            for key in ("course_title", "course_id"):
                val = sf.get(key)
                if isinstance(val, str) and val.strip():
                    return val
    return slug


# --------------------------------------------------------------------------- #
# Objective normalization
# --------------------------------------------------------------------------- #


def _normalize_lo(item: Any) -> Dict[str, Any]:
    """Normalize a single LO entry to carry both ``id`` and ``text``.

    Accepts a dict (with ``statement`` OR ``text``) or a bare string. The
    original fields are PRESERVED — ``text`` is added (mapped from
    ``statement``) without dropping ``statement`` or any other key, so the
    frontend can read ``text`` while reuse tooling still sees ``statement``.
    """
    if isinstance(item, str):
        return {"id": "", "text": item}
    if not isinstance(item, dict):
        return {"id": "", "text": str(item)}
    out = dict(item)
    if "text" not in out:
        statement = out.get("statement")
        if isinstance(statement, str):
            out["text"] = statement
        else:
            out["text"] = ""
    out.setdefault("id", "")
    return out


def _normalize_objectives_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an objectives doc for the frontend.

    Surfaces ``terminal_objectives`` + ``chapter_objectives`` (each entry with
    ``id`` + ``text``) regardless of whether the on-disk doc used the
    Courseforge form (``terminal_objectives`` / ``chapter_objectives`` /
    ``learning_outcomes``) or the LibV2 archive form (``terminal_outcomes`` /
    ``component_objectives``). All original top-level fields are preserved.
    """
    out = dict(doc)

    terminal_src = doc.get("terminal_objectives")
    if not isinstance(terminal_src, list):
        terminal_src = doc.get("terminal_outcomes")
    chapter_src = doc.get("chapter_objectives")
    if not isinstance(chapter_src, list):
        chapter_src = doc.get("component_objectives")

    # Last resort: split a flat learning_outcomes list by hierarchy_level.
    if not isinstance(terminal_src, list) and not isinstance(chapter_src, list):
        los = doc.get("learning_outcomes")
        if isinstance(los, list):
            terminal_src = [
                lo for lo in los
                if isinstance(lo, dict) and lo.get("hierarchy_level") == "terminal"
            ]
            chapter_src = [
                lo for lo in los
                if isinstance(lo, dict) and lo.get("hierarchy_level") != "terminal"
            ]

    out["terminal_objectives"] = [
        _normalize_lo(x) for x in (terminal_src or [])
    ]
    out["chapter_objectives"] = [
        _normalize_lo(x) for x in (chapter_src or [])
    ]
    return out


def _count_objectives(doc: Dict[str, Any]) -> Tuple[int, int]:
    """Return ``(terminal_count, chapter_count)`` from any objectives doc form."""
    normalized = _normalize_objectives_doc(doc)
    return (
        len(normalized.get("terminal_objectives", [])),
        len(normalized.get("chapter_objectives", [])),
    )


# --------------------------------------------------------------------------- #
# course_id resolution
# --------------------------------------------------------------------------- #


def _resolve(course_id: str) -> Dict[str, Any]:
    """Resolve a ``course_id`` to its export and/or LibV2 record.

    ``course_id`` may be a Courseforge ``project_id`` (``PROJ-...``), a
    ``course_name``, or a LibV2 ``slug``. Returns a dict with optional keys:
    ``export`` (the export record), ``slug``, ``libv2_dir``, ``manifest``.
    A successful resolve carries at least one of ``export`` / ``slug``.
    """
    cid = (course_id or "").strip()
    result: Dict[str, Any] = {}
    if not cid:
        return result

    exports = _export_records()

    # 1. Exact project_id match.
    for rec in exports:
        if rec["project_id"] == cid:
            result["export"] = rec
            break

    # 2. course_name match (newest export wins) when not already matched.
    if "export" not in result:
        by_name = _latest_export_by_course_name(cid)
        if by_name is not None:
            result["export"] = by_name

    # 3. LibV2 slug — direct directory, then case-fold scan.
    libv2_dir = _libv2_courses_root() / cid
    if libv2_dir.is_dir():
        result["slug"] = cid
        result["libv2_dir"] = libv2_dir
    else:
        target = cid.casefold()
        for d in _iter_libv2_dirs():
            if d.name.casefold() == target:
                result["slug"] = d.name
                result["libv2_dir"] = d
                break

    # 4. If we resolved an export by name/id, try to bind a same-named LibV2
    #    course (slugified) so classification can ride along.
    if "slug" not in result and "export" in result:
        cn = result["export"].get("course_name")
        if isinstance(cn, str) and cn:
            slug_guess = _slugify(cn)
            cand = _libv2_courses_root() / slug_guess
            if cand.is_dir():
                result["slug"] = slug_guess
                result["libv2_dir"] = cand

    if "libv2_dir" in result:
        result["manifest"] = _read_json(result["libv2_dir"] / "manifest.json")

    return result


def _slugify(name: str) -> str:
    """Lowercase, hyphenate a course_name for a best-effort LibV2 slug guess."""
    out = []
    prev_dash = False
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


# --------------------------------------------------------------------------- #
# Public API — §5 course_service
# --------------------------------------------------------------------------- #


def list_courses() -> List[Dict[str, Any]]:
    """List every known course across both corpora.

    Each item::

        {course_name, project_id, slug, duration_weeks,
         terminal_count, chapter_count, topics?, subdomains?}

    Courseforge exports are deduplicated by ``course_name`` (newest export
    wins). LibV2-only courses (no matching export) are appended with their
    classification topics/subdomains and objective counts from a sibling
    ``objectives.json`` when present. Missing files yield empty fields, never
    invented data.
    """
    courses: List[Dict[str, Any]] = []
    seen_slugs: set = set()
    # course_name (casefold) -> index in ``courses`` so we can attach LibV2
    # classification to the export-derived row.
    name_index: Dict[str, int] = {}

    # --- Courseforge exports (deduped by course_name, newest wins) ---------
    latest_by_name: Dict[str, Dict[str, Any]] = {}
    nameless: List[Dict[str, Any]] = []
    for rec in _export_records():  # ascending by dir name → newest last
        cn = rec.get("course_name")
        if isinstance(cn, str) and cn:
            latest_by_name[cn.casefold()] = rec
        else:
            nameless.append(rec)

    for rec in list(latest_by_name.values()) + nameless:
        config = rec["config"]
        course_name = rec.get("course_name") or rec["project_id"]
        obj_doc = _read_json(rec["objectives_path"]) or {}
        terminal_count, chapter_count = _count_objectives(obj_doc)
        duration_weeks = config.get("duration_weeks")
        if not isinstance(duration_weeks, int):
            dw = obj_doc.get("duration_weeks")
            duration_weeks = dw if isinstance(dw, int) else None

        # Best-effort sibling LibV2 course for classification enrichment.
        slug_guess = _slugify(str(course_name))
        libv2_dir = _libv2_courses_root() / slug_guess
        topics: List[str] = []
        subdomains: List[str] = []
        slug_val: Optional[str] = None
        if libv2_dir.is_dir():
            slug_val = slug_guess
            seen_slugs.add(slug_guess)
            manifest = _read_json(libv2_dir / "manifest.json")
            classification = (manifest or {}).get("classification") or {}
            topics = list(classification.get("topics") or [])
            subdomains = list(classification.get("subdomains") or [])

        entry = {
            "course_name": course_name,
            "project_id": rec["project_id"],
            "slug": slug_val,
            "duration_weeks": duration_weeks,
            "terminal_count": terminal_count,
            "chapter_count": chapter_count,
            "topics": topics,
            "subdomains": subdomains,
        }
        if isinstance(course_name, str):
            name_index[course_name.casefold()] = len(courses)
        courses.append(entry)

    # --- LibV2-only courses (no matching export) ---------------------------
    for d in _iter_libv2_dirs():
        slug = d.name
        if slug in seen_slugs:
            continue
        manifest = _read_json(d / "manifest.json")
        if manifest is None:
            # A course dir without a manifest is not a real course surface.
            continue
        classification = manifest.get("classification") or {}
        course_name = _libv2_display_name(slug, manifest)
        # If this LibV2 course shares a name with an export row, enrich it
        # in place instead of adding a duplicate.
        idx = name_index.get(course_name.casefold())
        if idx is not None and courses[idx].get("slug") is None:
            courses[idx]["slug"] = slug
            courses[idx]["topics"] = list(classification.get("topics") or [])
            courses[idx]["subdomains"] = list(classification.get("subdomains") or [])
            continue

        obj_doc = _read_json(d / "objectives.json") or {}
        terminal_count, chapter_count = _count_objectives(obj_doc)
        courses.append(
            {
                "course_name": course_name,
                "project_id": None,
                "slug": slug,
                "duration_weeks": (
                    obj_doc.get("duration_weeks")
                    if isinstance(obj_doc.get("duration_weeks"), int)
                    else None
                ),
                "terminal_count": terminal_count,
                "chapter_count": chapter_count,
                "topics": list(classification.get("topics") or []),
                "subdomains": list(classification.get("subdomains") or []),
            }
        )

    courses.sort(key=lambda c: str(c.get("course_name") or "").casefold())
    return courses


def get_objectives(course_id: str) -> Dict[str, Any]:
    """Read the objectives doc for ``course_id`` (export or LibV2 archive form).

    Returns the on-disk doc with ``terminal_objectives`` + ``chapter_objectives``
    normalized to carry ``id`` + ``text`` (original fields preserved).

    Raises:
        FileNotFoundError: if the course can't be resolved or has no
            objectives doc on disk.
    """
    resolved = _resolve(course_id)
    doc: Optional[Dict[str, Any]] = None

    if "export" in resolved:
        doc = _read_json(resolved["export"]["objectives_path"])
    if doc is None and "libv2_dir" in resolved:
        doc = _read_json(resolved["libv2_dir"] / "objectives.json")

    if doc is None:
        if not resolved:
            raise FileNotFoundError(f"no course resolves to {course_id!r}")
        raise FileNotFoundError(
            f"no objectives document found for course {course_id!r}"
        )
    return _normalize_objectives_doc(doc)


def save_objectives(course_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Write a reuse-compatible ``synthesized_objectives.json`` for ``course_id``.

    The written doc matches the Courseforge on-disk shape that
    ``ed4all run --reuse-objectives`` accepts: ``terminal_objectives[]`` +
    ``chapter_objectives[]`` (each with ``id`` + ``statement``), a flat
    ``learning_outcomes[]`` mirror, and ``mint_method="user_supplied_objectives_json"``.
    The frontend sends ``text``; it is mapped to ``statement`` when writing.

    Every LO id is validated against the canonical pattern via
    ``lib.ontology.learning_objectives.validate_lo_id``; an invalid id raises
    ``ValueError`` (fail closed — no partial write).

    Args:
        course_id: project_id / course_name / slug.
        doc: ``{terminal_objectives:[{id,text}], chapter_objectives:[{id,text}],
             mint_method?}``.

    Returns:
        The persisted doc (Courseforge form).

    Raises:
        ValueError: on a non-dict ``doc`` or any invalid LO id.
        FileNotFoundError: if ``course_id`` resolves to no Courseforge export
            (we only author into a Courseforge export's
            ``01_learning_objectives/`` directory).
    """
    if not isinstance(doc, dict):
        raise ValueError("objectives doc must be a JSON object")

    resolved = _resolve(course_id)
    export = resolved.get("export")
    if export is None:
        raise FileNotFoundError(
            f"cannot save objectives: {course_id!r} resolves to no Courseforge "
            f"export (a PROJ-* export with project_config.json is required)"
        )

    course_name = export.get("course_name") or doc.get("course_name") or course_id

    terminal = _materialize_los(
        doc.get("terminal_objectives") or [], "terminal"
    )
    chapter = _materialize_los(
        doc.get("chapter_objectives") or [], "chapter"
    )

    learning_outcomes = terminal + chapter

    duration_weeks = doc.get("duration_weeks")
    if not isinstance(duration_weeks, int):
        cfg_dw = export["config"].get("duration_weeks")
        duration_weeks = cfg_dw if isinstance(cfg_dw, int) else _default_weeks(
            len(chapter)
        )

    persisted: Dict[str, Any] = {
        "course_name": course_name,
        "generated_from": "gui.course_service.save_objectives",
        "mint_method": USER_SUPPLIED_MINT_METHOD,
        "duration_weeks": duration_weeks,
        "learning_outcomes": learning_outcomes,
        "terminal_objectives": terminal,
        "chapter_objectives": chapter,
        "synthesized_at": _now_iso(),
    }

    target = export["objectives_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target, persisted)
    return _normalize_objectives_doc(persisted)


def get_classification(slug: str) -> Dict[str, Any]:
    """Return the ``classification`` block from a LibV2 course manifest.

    Args:
        slug: A LibV2 course slug (or a course_id that resolves to one).

    Raises:
        FileNotFoundError: if the slug resolves to no LibV2 manifest.
    """
    resolved = _resolve(slug)
    manifest = resolved.get("manifest")
    if manifest is None:
        raise FileNotFoundError(
            f"no LibV2 manifest found for course {slug!r}"
        )
    classification = manifest.get("classification")
    return dict(classification) if isinstance(classification, dict) else {}


def save_classification(slug: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``patch`` into a LibV2 manifest's ``classification`` block.

    Only the classification keys present in ``patch`` are updated; the rest of
    the manifest is preserved byte-for-byte (deep). The merged classification is
    validated best-effort against the ``classification`` subschema of
    ``schemas/library/course_manifest.schema.json`` (jsonschema is a default
    dep); a schema violation raises ``ValueError`` before any write.

    Args:
        slug: LibV2 course slug (or resolvable course_id).
        patch: ``{division?, primary_domain?, secondary_domains?, subdomains?,
             topics?, subtopics?, tags?}``.

    Returns:
        The merged classification block.

    Raises:
        ValueError: on a non-dict patch or a schema violation.
        FileNotFoundError: if the slug resolves to no LibV2 manifest.
    """
    if not isinstance(patch, dict):
        raise ValueError("classification patch must be a JSON object")

    resolved = _resolve(slug)
    libv2_dir = resolved.get("libv2_dir")
    manifest = resolved.get("manifest")
    if libv2_dir is None or manifest is None:
        raise FileNotFoundError(
            f"no LibV2 manifest found for course {slug!r}"
        )

    classification = dict(manifest.get("classification") or {})
    # Only known classification keys are accepted; unknown keys are ignored so
    # the manifest schema stays satisfied.
    allowed = {
        "division",
        "primary_domain",
        "secondary_domains",
        "subdomains",
        "topics",
        "subtopics",
        "tags",
    }
    for key, value in patch.items():
        if key in allowed:
            classification[key] = value

    _validate_classification(classification)

    manifest = dict(manifest)
    manifest["classification"] = classification
    target = libv2_dir / "manifest.json"
    _atomic_write_json(target, manifest)
    return classification


# --------------------------------------------------------------------------- #
# Internal write helpers
# --------------------------------------------------------------------------- #


def _materialize_los(items: Any, hierarchy: str) -> List[Dict[str, Any]]:
    """Turn frontend LO entries into canonical Courseforge LO dicts.

    Maps ``text`` → ``statement``, validates each ``id`` against the canonical
    pattern, and stamps ``hierarchy_level``. Preserves any extra fields the
    caller supplied (``bloom_level`` / ``cognitive_domain`` / ``chapter``).

    Raises:
        ValueError: on a missing/invalid id, or an entry that is not a
            dict/string.
    """
    if not isinstance(items, list):
        raise ValueError(f"{hierarchy}_objectives must be a list")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            raise ValueError(
                f"{hierarchy} objective #{idx} must be an object with an 'id'; "
                f"got a bare string"
            )
        if not isinstance(item, dict):
            raise ValueError(
                f"{hierarchy} objective #{idx} must be a JSON object"
            )
        lo_id = item.get("id")
        if not isinstance(lo_id, str) or not validate_lo_id(lo_id):
            raise ValueError(
                f"{hierarchy} objective #{idx} has an invalid LO id {lo_id!r} "
                f"(must match canonical [A-Z]{{2,}}-\\d{{2,}})"
            )
        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            text = item.get("text")
            statement = text if isinstance(text, str) else ""
        entry: Dict[str, Any] = dict(item)
        entry.pop("text", None)
        entry["id"] = lo_id
        entry["statement"] = statement
        entry["hierarchy_level"] = hierarchy
        out.append(entry)
    return out


def _validate_classification(classification: Dict[str, Any]) -> None:
    """Best-effort jsonschema validation of a classification block.

    Validates against the ``classification`` subschema embedded in
    ``schemas/library/course_manifest.schema.json``. jsonschema is a default
    dep; if it (or the schema file) is unavailable, validation is skipped
    silently (best-effort, per the spec). A real schema violation raises
    ``ValueError``.
    """
    try:
        import jsonschema  # noqa: PLC0415
    except ImportError:
        return
    from lib.paths import SCHEMAS_PATH  # noqa: PLC0415

    schema_path = Path(SCHEMAS_PATH) / "library" / "course_manifest.schema.json"
    schema = _read_json(schema_path)
    if not schema:
        return
    subschema = (schema.get("properties") or {}).get("classification")
    if not isinstance(subschema, dict):
        return
    try:
        jsonschema.validate(instance=classification, schema=subschema)
    except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
        raise ValueError(
            f"classification fails course_manifest schema: {exc.message}"
        ) from exc


def _default_weeks(chapter_count: int) -> int:
    """Objective-driven default pacing: ``max(8, ceil(chapters / 2))``.

    Mirrors the Wave 1.8 ``WAVE18_COS_PER_WEEK`` default (2) used by
    ``plan_course_structure`` so a GUI-authored objectives doc with no explicit
    ``duration_weeks`` paces consistently with the pipeline.
    """
    return max(8, math.ceil(chapter_count / 2)) if chapter_count else 8


def _now_iso() -> str:
    """UTC ISO-8601 timestamp (delegates to shared_state for one source)."""
    from gui.shared_state import now_iso  # noqa: PLC0415

    return now_iso()


__all__ = [
    "USER_SUPPLIED_MINT_METHOD",
    "list_courses",
    "get_objectives",
    "save_objectives",
    "get_classification",
    "save_classification",
]
