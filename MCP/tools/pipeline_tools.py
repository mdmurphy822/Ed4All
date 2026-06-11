"""
Ed4All Pipeline Tools

MCP tools for the unified textbook-to-course pipeline.
Chains: DART (PDF -> HTML) -> Courseforge (course generation) -> Trainforge (assessments)
"""

import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import PROJECT_ROOT  # noqa: E402
from lib.paths import courseforge_exports_dir as _lib_courseforge_exports_dir  # noqa: E402
from lib.secure_paths import validate_path_within_root  # noqa: E402
from Trainforge.chunker import CHUNKER_SCHEMA_VERSION  # noqa: E402

logger = logging.getLogger(__name__)

# Last-resort Bloom level for LO->targets-concept synthesis. When an LO carries
# key_concepts but neither an explicit canonical bloom_level nor a detectable
# verb in its statement, we fall back to this level so the LO's targetedConcepts
# (and the downstream targets-concept edges) are NOT silently dropped. "apply"
# is the median/most-common course-objective level and a safe neutral default.
_FALLBACK_BLOOM_LEVEL = "apply"

# Derived paths
DART_OUTPUT_DIR = PROJECT_ROOT / "DART" / "batch_output"
COURSEFORGE_INPUTS = PROJECT_ROOT / "Courseforge" / "inputs" / "textbooks"
TRAINING_CAPTURES = PROJECT_ROOT / "training-captures"


# Snapshot of the project root at import time so ``courseforge_exports_dir`` can
# detect which test seam (``PROJECT_ROOT`` / ``_PROJECT_ROOT``) was monkeypatched.
_IMPORT_PROJECT_ROOT = PROJECT_ROOT


def courseforge_exports_dir() -> Path:
    """Resolve the Courseforge exports dir for this module.

    Honors the ED4ALL_HOME relocatable data root via ``lib.paths`` when set;
    otherwise resolves against this module's project root. ``PROJECT_ROOT`` and
    ``_PROJECT_ROOT`` are identical at import time but are two distinct
    long-standing test seams — different phase-handler tests monkeypatch one or
    the other to redirect project exports into a ``tmp_path``. To honor both, we
    use whichever has been patched away from the import-time default
    (``_IMPORT_PROJECT_ROOT``); when neither is patched they're equal so the
    choice is moot. Byte-stable to ``<root> / "Courseforge" / "exports"`` when
    ED4ALL_HOME is unset.
    """
    from lib.paths import ed4all_home  # noqa: PLC0415

    if ed4all_home() is not None:
        return _lib_courseforge_exports_dir()
    # Prefer whichever seam a test redirected; default to PROJECT_ROOT.
    if PROJECT_ROOT != _IMPORT_PROJECT_ROOT:
        root = PROJECT_ROOT
    elif _PROJECT_ROOT != _IMPORT_PROJECT_ROOT:
        root = _PROJECT_ROOT
    else:
        root = PROJECT_ROOT
    return root / "Courseforge" / "exports"

# Backstop regex to scrub Qwen-invented `data-cf-objective-id` values.
# Byte-identical to `_DATA_CF_OBJECTIVE_ID_RE` in
# `Courseforge/router/inter_tier_gates.py` (the validator's read-side
# regex). The rewrite-phase loop substitutes the canonical
# `Block.objective_ids` into emitted content so packaging sees the
# upstream-supplied IDs.
_OBJ_ID_RE = re.compile(r'data-cf-objective-id=["\']([^"\']*)["\']')


def _ensure_directories():
    """Ensure required directories exist."""
    for path in [COURSEFORGE_INPUTS, TRAINING_CAPTURES]:
        path.mkdir(parents=True, exist_ok=True)


_ensure_directories()


# ---------------------------------------------------------------------------
# Wave 74 cleanup: pluggable staging modes.
#
# stage_dart_outputs originally deep-copied every DART HTML, *_synthesized.json,
# *.quality.json, and `{stem}_figures/` directory into
# ``Courseforge/inputs/textbooks/{run_id}/``. For an 8-PDF / 768-page corpus
# this cost ~70MB per run; the staging dir is gitignored and never garbage
# collected. Symlinks (or hardlinks on platforms that disallow user symlinks)
# preserve every downstream behaviour because all known consumers go through
# Path().read_text() / read_bytes() rather than os.path.realpath().
#
# Modes:
#   - "copy"     : shutil.copy2 / shutil.copytree (legacy behaviour)
#   - "symlink"  : os.symlink for files AND directories (single tree-symlink
#                  for the figures dir, NOT a deep walk)
#   - "hardlink" : os.link for files; for directories, recreate the tree and
#                  hardlink each file. Falls back when symlinks are blocked
#                  (e.g., Windows without SeCreateSymbolicLinkPrivilege).
#
# Default for runs with no override is ``symlink``: the staging tree is
# gitignored test infrastructure, the source DART output is the durable copy,
# and downstream phases never write to the staged paths so symlink rot is not
# a concern.
# ---------------------------------------------------------------------------

VALID_STAGE_MODES = ("copy", "symlink", "hardlink")
DEFAULT_STAGE_MODE = "symlink"


def _resolve_stage_mode(explicit: Optional[str] = None) -> str:
    """Resolve the active staging mode.

    Precedence:
        1. ``explicit`` parameter (passed through from the tool kwargs).
        2. ``ED4ALL_STAGE_MODE`` environment variable.
        3. :data:`DEFAULT_STAGE_MODE` (``"symlink"``).

    Unknown values fall back to the default with a warning so a typo never
    silently disables staging.
    """
    candidate = explicit or os.environ.get("ED4ALL_STAGE_MODE") or DEFAULT_STAGE_MODE
    candidate = candidate.strip().lower()
    if candidate not in VALID_STAGE_MODES:
        logger.warning(
            "Unknown stage_mode %r — falling back to %r. Valid: %s",
            candidate, DEFAULT_STAGE_MODE, VALID_STAGE_MODES,
        )
        candidate = DEFAULT_STAGE_MODE
    return candidate


def _stage_file(src: Path, dest: Path, mode: str) -> None:
    """Stage a single file at ``src`` into ``dest`` using the given mode.

    Always replaces an existing ``dest`` (file or symlink) so re-runs are
    idempotent. Falls back from symlink/hardlink to copy on OSError so a
    locked-down platform can never break a staging phase outright.
    """
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if mode == "copy":
        shutil.copy2(src, dest)
        return
    if mode == "symlink":
        try:
            os.symlink(os.fspath(src.resolve()), os.fspath(dest))
            return
        except OSError as e:
            logger.warning(
                "symlink failed for %s -> %s (%s); falling back to copy", src, dest, e,
            )
            shutil.copy2(src, dest)
            return
    if mode == "hardlink":
        try:
            os.link(os.fspath(src), os.fspath(dest))
            return
        except OSError as e:
            logger.warning(
                "hardlink failed for %s -> %s (%s); falling back to copy", src, dest, e,
            )
            shutil.copy2(src, dest)
            return
    # Should never hit — _resolve_stage_mode guards the enum.
    shutil.copy2(src, dest)


def _stage_tree(src_dir: Path, dest_dir: Path, mode: str) -> None:
    """Stage a directory tree from ``src_dir`` into ``dest_dir``.

    - ``copy``     : shutil.copytree (deep copy)
    - ``symlink``  : a single tree-level os.symlink at ``dest_dir`` pointing at
                     ``src_dir``. Cheap (one inode, no walk).
    - ``hardlink`` : recreate the directory structure and hardlink every file.
    """
    if dest_dir.exists() or dest_dir.is_symlink():
        if dest_dir.is_symlink() or dest_dir.is_file():
            dest_dir.unlink()
        else:
            shutil.rmtree(dest_dir)
    if mode == "copy":
        shutil.copytree(src_dir, dest_dir)
        return
    if mode == "symlink":
        try:
            os.symlink(
                os.fspath(src_dir.resolve()),
                os.fspath(dest_dir),
                target_is_directory=True,
            )
            return
        except OSError as e:
            logger.warning(
                "tree symlink failed for %s -> %s (%s); falling back to copytree",
                src_dir, dest_dir, e,
            )
            shutil.copytree(src_dir, dest_dir)
            return
    if mode == "hardlink":
        try:
            for src_path in src_dir.rglob("*"):
                rel = src_path.relative_to(src_dir)
                target = dest_dir / rel
                if src_path.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        target.unlink()
                    try:
                        os.link(os.fspath(src_path), os.fspath(target))
                    except OSError:
                        shutil.copy2(src_path, target)
            return
        except OSError as e:
            logger.warning(
                "hardlink tree failed for %s -> %s (%s); falling back to copytree",
                src_dir, dest_dir, e,
            )
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir)
            return
    shutil.copytree(src_dir, dest_dir)


def _course_chunk_id_prefix(course_name: str) -> str:
    """Return the ``{course_code}_chunk_`` prefix Trainforge writes.

    Mirrors ``Trainforge.process_course.CourseProcessor`` — the chunk
    prefix is ``f"{self.course_code.lower()}_chunk_"`` (see
    ``Trainforge/process_course.py:1106``). We lowercase here too so the
    archival gate matches the on-disk IDs exactly. Spaces / dashes get
    normalised to underscores so values that have already been slugified
    (e.g. ``"demo-101"``) still produce the right prefix
    (``"demo_101_chunk_"``).
    """
    code = (course_name or "").strip().lower()
    if not code:
        return ""
    # Trainforge keeps underscores in the prefix; if the caller passed a
    # slug-shaped name (``demo-101``), normalise back to underscores.
    code = code.replace("-", "_").replace(" ", "_")
    return f"{code}_chunk_"


def _check_chunks_freshness(
    *,
    chunks_path: Path,
    course_name: str,
    run_start_ts: float,
    had_prior_chunks: bool,
) -> dict:
    """Wave 74: classify chunks.jsonl at the archive destination.

    Returns a dict with ``status`` ∈ {``"absent"``, ``"fresh"``,
    ``"stale"``} plus diagnostic fields. The archival caller fails closed
    on ``"stale"``.

    Decision rules:

    * ``absent`` — no file at ``chunks_path``. Trainforge was
      intentionally skipped (DART-only batch) OR the copy block deleted
      the prior file and never wrote a fresh one. Either way, archival
      proceeds without chunks; feature flags fall back to ``false``.
    * ``fresh`` — file exists; either every line decodes to a chunk
      whose ``id`` starts with ``{course_code_lower()}_chunk_`` OR the
      file's ``mtime`` is at or after ``run_start_ts`` (mtime check is
      a fallback for callers that don't follow the prefix convention).
    * ``stale`` — file exists, but at least one chunk's ``id`` carries
      a prefix that doesn't match the current course AND the file
      pre-dates ``run_start_ts``. Caught the RDF/SHACL calibration corpus leak.

    Args:
        chunks_path: Where chunks.jsonl lives in the LibV2 archive
            (``course_dir / "imscc_chunks" / "chunks.jsonl"`` post-Phase
            7c, or legacy ``course_dir / "corpus" / "chunks.jsonl"``).
        course_name: The current run's course code / name. Used to
            derive the expected ``{prefix}_chunk_`` ID pattern.
        run_start_ts: ``time.time()`` captured at archival entry. Files
            with ``mtime >= run_start_ts`` are by definition fresh.
        had_prior_chunks: ``True`` when the destination already had a
            chunks file before the copy block ran. Used to disambiguate
            the ``absent`` outcome — when we deleted a prior file but
            never re-wrote, the absent-after-delete state is OK as long
            as Trainforge was intentionally absent. (We don't fail
            closed on it because the existing flow was always fine with
            no chunks for DART-only runs.)
    """
    if not chunks_path.exists() or not chunks_path.is_file():
        return {"status": "absent", "reason": "no chunks.jsonl present"}

    expected_prefix = _course_chunk_id_prefix(course_name)
    if not expected_prefix:
        # No course name → can't validate. Treat as absent so we don't
        # fail closed on a caller-side bug; the missing-course-name
        # branch above already short-circuits with a clearer error.
        return {"status": "absent", "reason": "no course_name to validate against"}

    # mtime check — files written this run pass unconditionally.
    try:
        mtime = chunks_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if mtime >= run_start_ts:
        return {
            "status": "fresh",
            "reason": "chunks.jsonl mtime is at/after run start",
        }

    # mtime predates run-start → inspect the IDs. We sample a bounded
    # number of lines so a multi-GB chunks.jsonl doesn't blow the
    # archival path's runtime.
    #
    # Decision rule: chunks landing in a LibV2 archive are produced by
    # ``Trainforge.process_course`` which writes IDs as
    # ``{course_code.lower()}_chunk_{N}`` (process_course.py:1106). The
    # ``_chunk_`` substring is the recognizable production signature.
    # When at least one chunk on disk has a recognizable course prefix
    # (i.e. ``{head}_chunk_...``) that DOESN'T match the current course,
    # we have positive evidence the chunks file is from a different
    # run/course → stale. When chunks have no recognizable
    # ``{head}_chunk_`` shape at all (synthetic test fixtures,
    # malformed inputs), we treat as unverifiable rather than stale —
    # the pre-Wave-74 behaviour was to write the archive anyway, and a
    # purely-synthetic IMSCC pipeline is allowed to keep working.
    observed_prefixes: dict[str, int] = {}
    matched = 0
    unrecognized = 0
    inspected = 0
    sample_limit = 50
    try:
        with open(chunks_path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                inspected += 1
                try:
                    chunk = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    unrecognized += 1
                    continue
                if not isinstance(chunk, dict):
                    unrecognized += 1
                    continue
                cid = chunk.get("id") or ""
                if not isinstance(cid, str):
                    unrecognized += 1
                    continue
                if cid.startswith(expected_prefix):
                    matched += 1
                elif "_chunk_" in cid:
                    # Recognizable production-shape ID but wrong course.
                    head = cid.split("_chunk_", 1)[0]
                    if head:
                        observed_prefixes[head] = (
                            observed_prefixes.get(head, 0) + 1
                        )
                    else:
                        unrecognized += 1
                else:
                    # No ``_chunk_`` marker at all — synthetic / minimal
                    # test fixture or unknown shape. No positive evidence
                    # of staleness; tolerate.
                    unrecognized += 1
                if inspected >= sample_limit:
                    break
    except OSError as exc:
        return {
            "status": "stale",
            "reason": f"could not read chunks.jsonl ({exc})",
            "expected_prefix": expected_prefix,
            "observed_prefixes": {},
        }

    if inspected == 0:
        # File exists but is empty / all blank lines — treat as absent
        # so DART-only smoke tests that touch an empty chunks file
        # don't get a false-positive failure.
        return {
            "status": "absent",
            "reason": "chunks.jsonl is empty",
        }

    if matched > 0:
        return {
            "status": "fresh",
            "reason": (
                f"found {matched}/{inspected} chunks matching prefix "
                f"{expected_prefix!r}"
            ),
        }

    if observed_prefixes:
        # Recognizable production IDs from a DIFFERENT course → stale.
        return {
            "status": "stale",
            "reason": (
                f"chunks.jsonl carries IDs from a different course "
                f"(expected prefix {expected_prefix!r}, observed "
                f"{sorted(observed_prefixes.items(), key=lambda x: -x[1])[:3]})"
            ),
            "expected_prefix": expected_prefix,
            "observed_prefixes": observed_prefixes,
        }

    # No recognizable course prefix at all (synthetic / minimal test
    # fixture). No positive evidence of staleness — tolerate so we
    # don't fail closed on perfectly fine non-production inputs.
    return {
        "status": "fresh",
        "reason": (
            f"chunks.jsonl carries unrecognized IDs ({unrecognized}/"
            f"{inspected} lines) — no positive staleness evidence"
        ),
    }


def _resolve_libv2_root(explicit: Optional[str] = None) -> Path:
    """Phase 8 ST 3: resolve the LibV2 root directory used by Phase 6/7
    helpers (`_run_concept_extraction`, `_run_dart_chunking`,
    `_run_imscc_chunking`) when persisting per-course artifacts under
    ``<libv2_root>/courses/<course_slug>/``.

    Resolution chain (high → low priority):
        1. Explicit ``libv2_root`` kwarg (typically threaded by the
           workflow runner via ``inputs_from`` from
           ``workflow_params.libv2_root`` — see
           ``config/workflows.yaml::textbook_to_course`` and
           ``MCP/core/workflow_runner.py::_LEGACY_PHASE_PARAM_ROUTING``).
        2. ``ED4ALL_LIBV2_ROOT`` env var (deployment-level override —
           useful for ops topologies that mount LibV2 at a non-default
           location, e.g. Docker volume / NFS / ConfigMap).
        3. ``_PROJECT_ROOT / "LibV2"`` legacy default (unchanged from
           pre-Phase-8 behaviour — every existing run continues to write
           to the in-tree ``LibV2/`` directory).

    Returns a ``Path`` (existence is NOT enforced — callers create the
    target ``courses/<slug>/...`` subdir as needed). Empty / whitespace-
    only ``explicit`` argument falls through as if unset, mirroring the
    ``or ""`` → falsy treatment in the call sites.
    """
    if explicit:
        cand = explicit.strip() if isinstance(explicit, str) else ""
        if cand:
            return Path(cand)
    env_val = os.environ.get("ED4ALL_LIBV2_ROOT", "").strip()
    if env_val:
        return Path(env_val)
    return _PROJECT_ROOT / "LibV2"


def _project_synthesized_objectives_to_course_json(
    synthesized_objectives_path: Path,
    course_json_path: Path,
    *,
    course_code: str,
    course_title: str,
) -> Optional[Dict[str, Any]]:
    """Project ``synthesized_objectives.json`` to a packaging-shaped
    ``course.json`` at ``<project>/03_content_development/course.json``.

    Wave2-I3 (Finding 3 of plans/dispatch-7-execution-inspection-2026-05.md):
    closes ``PAGE_OBJECTIVES_PATH_MISSING`` blocker. The
    ``PageObjectivesValidator`` (``lib/validators/page_objectives.py:192-249``)
    auto-discovers ``content_dir / "course.json"`` and fails closed
    critical-severity when absent; ``package_multifile_imscc.package_imscc``
    has the same auto-discovery contract via ``load_canonical_objectives``
    (``Courseforge/scripts/generate_course.py:615-646``), which reads
    ``terminal_objectives`` + ``chapter_objectives``.

    Projection is idempotent: when ``course_json_path`` already exists
    (e.g. ``--reuse-objectives`` ran or Trainforge's ``_build_course_json``
    emitted), this function logs INFO and returns ``None`` without
    overwriting. Returns the on-disk dict on a fresh emit.

    Source-shape handling is defensive: ``synthesized_objectives.json``
    normally carries the Courseforge synthesized form
    (``terminal_objectives[]`` + ``chapter_objectives[]`` per
    ``MCP/tools/pipeline_tools.py::_plan_course_structure`` at line
    ~4948); the root ``CLAUDE.md`` ``--reuse-objectives`` doc says the
    runner normalizes the LibV2 archive form
    (``terminal_outcomes[]`` + ``component_objectives[]``) to the
    former on disk, but we handle both forms here so a hand-edited
    legacy file doesn't silently emit an empty course.json.

    Args:
        synthesized_objectives_path: Path to the input JSON
            (``01_learning_objectives/synthesized_objectives.json``).
        course_json_path: Path to the target packaging-shaped
            ``course.json`` (``03_content_development/course.json``).
        course_code: Stable course identifier (e.g. ``PHYS_101``).
        course_title: Human-readable course title.

    Returns:
        The emitted course.json dict on a fresh write, or ``None`` if
        the target already exists (idempotent skip) or the synthesized
        objectives file is missing / malformed.
    """
    if course_json_path.exists():
        logger.info(
            "_project_synthesized_objectives_to_course_json: target "
            "%s already exists; skipping idempotent emit.",
            course_json_path,
        )
        return None
    if not synthesized_objectives_path.exists():
        logger.warning(
            "_project_synthesized_objectives_to_course_json: synthesized "
            "objectives not found at %s; cannot emit course.json. "
            "PageObjectivesValidator will fail closed with "
            "PAGE_OBJECTIVES_PATH_MISSING at the packaging gate.",
            synthesized_objectives_path,
        )
        return None
    try:
        synthesized = json.loads(
            synthesized_objectives_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "_project_synthesized_objectives_to_course_json: failed to "
            "parse %s (%s); skipping course.json emit.",
            synthesized_objectives_path, exc,
        )
        return None
    if not isinstance(synthesized, dict):
        logger.warning(
            "_project_synthesized_objectives_to_course_json: %s did not "
            "parse to a dict (got %s); skipping course.json emit.",
            synthesized_objectives_path, type(synthesized).__name__,
        )
        return None

    # Defensive: accept three shapes for ``chapter_objectives``:
    # (1) the Courseforge synthesized list-of-groups form,
    # (2) the LibV2 archive form (``component_objectives``), and
    # (3) the OpenStax dict-of-lists form (Wave2b — see
    # ``_normalize_chapter_objectives_to_groups`` docstring).
    # ``terminal_objectives`` accepts both the Courseforge name and
    # the LibV2 ``terminal_outcomes`` alias.
    terminal = (
        synthesized.get("terminal_objectives")
        or synthesized.get("terminal_outcomes")
        or []
    )
    chapter_groups_raw = (
        synthesized.get("chapter_objectives")
        or synthesized.get("component_objectives")
        or []
    )
    chapter_groups = _normalize_chapter_objectives_to_groups(chapter_groups_raw)
    # The pre-emitted ``learning_outcomes`` flat list (Courseforge
    # form) is the canonical Trainforge-shaped roll-up. Reuse when
    # present; reconstruct from terminal + chapter otherwise so the
    # LibV2 archive form (which doesn't pre-emit the flat list)
    # still produces a usable course.json.
    learning_outcomes_raw = synthesized.get("learning_outcomes")
    if not isinstance(learning_outcomes_raw, list) or not learning_outcomes_raw:
        learning_outcomes_raw = []
        for t in terminal if isinstance(terminal, list) else []:
            if isinstance(t, dict):
                lo = dict(t)
                lo.setdefault("hierarchy_level", "terminal")
                learning_outcomes_raw.append(lo)
        for grp in chapter_groups:
            for c in grp.get("objectives") or []:
                if isinstance(c, dict):
                    lo = dict(c)
                    lo.setdefault("hierarchy_level", "chapter")
                    learning_outcomes_raw.append(lo)

    duration_weeks = synthesized.get("duration_weeks")

    course_json: Dict[str, Any] = {
        # Required by schemas/knowledge/course.schema.json + consumed by
        # Trainforge's load_course_outcomes.
        "course_code": course_code,
        "title": course_title,
        # PageObjectivesValidator (lib/validators/page_objectives.py:192-249)
        # auto-discovers content_dir / "course.json" then routes to
        # load_canonical_objectives (Courseforge/scripts/generate_course.py:615)
        # which reads these two keys verbatim.
        "terminal_objectives": list(terminal)
        if isinstance(terminal, list) else [],
        # Persist the normalized list-of-groups shape so
        # ``load_canonical_objectives`` (which reads ``chapter`` +
        # ``objectives`` keys per group) sees a uniform structure
        # across input shapes — the OpenStax dict-of-lists shape
        # previously emitted as a dict on disk and dropped every CO
        # silently at the downstream walk.
        "chapter_objectives": chapter_groups,
        # LibV2 course.json shape (schemas/knowledge/course.schema.json:32-87):
        # flat learning_outcomes[] with id/statement/hierarchy_level. Carries
        # bloom_level/bloom_verb/key_concepts/cognitive_domain when synthesizer
        # populated them so downstream Trainforge consumers (process_course
        # _build_course_json) keep parity with the synthesized payload.
        "learning_outcomes": learning_outcomes_raw,
        # Provenance: chunker_version mirrors course_manifest.json so
        # downstream Trainforge consumers don't graceful-degrade on a
        # missing version field.
        "chunker_version": _resolve_chunker_version(),
    }
    if duration_weeks is not None:
        course_json["duration_weeks"] = duration_weeks

    course_json_path.parent.mkdir(parents=True, exist_ok=True)
    course_json_path.write_text(
        json.dumps(course_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "_project_synthesized_objectives_to_course_json: emitted %s "
        "with %d terminal_objectives + %d chapter_objective groups + "
        "%d learning_outcomes (closes PAGE_OBJECTIVES_PATH_MISSING "
        "blocker; Wave2-I3).",
        course_json_path,
        len(course_json["terminal_objectives"]),
        len(course_json["chapter_objectives"]),
        len(course_json["learning_outcomes"]),
    )
    return course_json


def _resolve_chunker_version() -> str:
    """Resolve the chunker-schema-contract version stamped on chunkset
    sidecar manifests + ``course_manifest.json``.

    Migration drift (post-Phase-8 review): the ``ed4all-chunker``
    workspace package was folded back into ``Trainforge/chunker/``;
    ``importlib.metadata.version("ed4all-chunker")`` is no longer the
    source of truth. The field's semantics also shifted: it used to
    carry the Python-package release version (e.g. ``"0.1.0"``); it
    now carries the chunker-schema-contract version
    (``Trainforge.chunker.CHUNKER_SCHEMA_VERSION``, currently
    ``"v4"``) — bumped only when the emit shape or semantics change,
    decoupled from any Python-package release cadence.

    Returns the in-repo constant directly: the chunker now lives
    inside Trainforge so the import is a hot-path-cheap module
    attribute lookup. Both the chunkset sidecar schema (`schemas/
    library/chunkset_manifest.schema.json::chunker_version`) and the
    course manifest schema (`schemas/library/course_manifest.schema
    .json::chunker_version`) accept BOTH the old ``^\\d+\\.\\d+\\.\\d+``
    semver shape and the new ``^v\\d+$`` schema-version shape, so any
    pre-migration manifest on disk continues to validate.
    """
    return CHUNKER_SCHEMA_VERSION


def _normalize_chapter_objectives_to_groups(raw: Any) -> List[Dict[str, Any]]:
    """Normalize ``chapter_objectives`` (any shape) to the canonical
    list-of-groups form: ``[{"chapter": <label>, "objectives": [...]}, ...]``.

    Wave2b (``plans/wave2-smoke-verification-2026-05.md`` "Surprises"):
    OpenStax-shaped ``synthesized_objectives.json`` carries
    ``chapter_objectives`` as a dict-of-lists keyed on chapter labels
    (e.g. ``{"1": [...], "2": [...]}``). Both Wave 2 helpers
    (``_project_synthesized_objectives_to_course_json`` from Wave2-I3
    + ``_collect_lo_ids`` inside ``_plan_course_structure`` from
    Wave2-I7) previously handled only the list-of-groups form and the
    flat-list form; the dict-of-lists branch silently dropped every
    CO-NN id from the projection + the ``objective_ids`` rollup.
    Normalizing once here keeps both call sites byte-stable on the
    legacy shapes and adds the dict-of-lists branch.

    Accepts three shapes:

    1. **List-of-groups** (Courseforge synthesized form, the canonical
       on-disk shape emitted by ``_plan_course_structure``):
       ``[{"chapter": N, "objectives": [{"id": "CO-NN", ...}, ...]}, ...]``.
       Returned verbatim.
    2. **Flat list** (LibV2 archive ``component_objectives`` form when
       not pre-grouped): ``[{"id": "CO-NN", "chapter": <opt>, ...}, ...]``.
       Wrapped in a single synthetic group keyed
       ``chapter="(ungrouped)"`` so the downstream walk still yields
       every CO-NN id.
    3. **Dict-of-lists** (OpenStax shape):
       ``{"1": [{"id": "CO-NN", ...}, ...], "2": [...], ...}``.
       Iterates keys in sorted order so the emitted group ordering is
       deterministic across runs; each value becomes the
       ``objectives`` array of a group whose ``chapter`` is the dict
       key.

    Any other shape returns ``[]`` — caller's responsibility to log
    + degrade gracefully.
    """
    if isinstance(raw, dict):
        groups: List[Dict[str, Any]] = []
        for chapter_key in sorted(raw.keys(), key=str):
            items = raw[chapter_key]
            if not isinstance(items, list):
                continue
            objectives: List[Dict[str, Any]] = []
            for item in items:
                if isinstance(item, dict):
                    objectives.append(dict(item))
            groups.append({
                "chapter": str(chapter_key),
                "objectives": objectives,
            })
        return groups
    if isinstance(raw, list):
        groups = []
        flat_buffer: List[Dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("objectives"), list):
                # List-of-groups shape — preserve group dict verbatim,
                # but coerce ``objectives`` entries to plain dicts.
                inner = [
                    dict(obj)
                    for obj in entry.get("objectives") or []
                    if isinstance(obj, dict)
                ]
                groups.append({
                    "chapter": entry.get("chapter"),
                    "objectives": inner,
                })
            else:
                # Flat-list shape — collect into a single synthetic group.
                flat_buffer.append(dict(entry))
        if flat_buffer:
            groups.append({
                "chapter": "(ungrouped)",
                "objectives": flat_buffer,
            })
        return groups
    return []


def _normalize_objectives_payload_to_course(
    payload: Any,
    course_code: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Normalize an objectives doc to the ``course.json`` graph-input shape.

    NVIDIA-KG item 1: ``build_semantic_graph``'s LO-dependent rules need a
    ``course`` dict carrying ``learning_outcomes: [{"id": "TO-NN"|"CO-NN",
    ...}, ...]`` in canonical order (the list POSITION is the LO-ordering
    signal ``prerequisite_from_lo_order::_lo_order_map`` consumes). This
    helper accepts any of:

    1. The canonical Trainforge ``course.json`` form — non-empty flat
       ``learning_outcomes[]`` (``schemas/knowledge/course.schema.json``).
       Used verbatim (order preserved).
    2. The Courseforge synthesized form — ``terminal_objectives[]`` +
       ``chapter_objectives`` (any of the three shapes handled by
       :func:`_normalize_chapter_objectives_to_groups`). Flattened
       terminal-first then chapter-groups in document order — the same
       canonical order :func:`_project_synthesized_objectives_to_course_json`
       emits.
    3. The LibV2 archive aliases — ``terminal_outcomes[]`` +
       ``component_objectives``.

    LO identity is validated via the canonical
    ``lib.ontology.learning_objectives.validate_lo_id`` helper (no local
    re-derivation of the ID pattern); entries with missing / non-canonical
    IDs are dropped. Duplicate IDs keep the first occurrence so the
    ordering signal stays stable.

    Also derives the ``objectives_metadata`` list consumed by the
    ``targets_concept_from_lo`` rule: one ``{"id", "targetedConcepts":
    [{"concept", "bloomLevel"}]}`` entry per LO that carries
    ``key_concepts`` — mirroring the canonical Courseforge JSON-LD emit
    (``Courseforge/scripts/blocks.py::Block._objective_jsonld`` slugifies
    ``key_terms`` and stamps the LO's Bloom level). Bloom level resolution:
    explicit ``bloom_level`` when canonical, else
    ``lib.ontology.bloom.detect_bloom_level`` over the statement, else the
    neutral ``_FALLBACK_BLOOM_LEVEL`` ("apply") with a logged warning — so an
    LO that carries key_concepts but no resolvable Bloom level still
    contributes its targetedConcepts instead of being silently dropped. LOs
    without concepts contribute no entry.

    Returns:
        ``(course, objectives_metadata)`` — both ``None`` when the payload
        yields zero canonical LOs. ``objectives_metadata`` may be ``None``
        while ``course`` is populated (no key_concepts anywhere).
    """
    from lib.ontology.bloom import BLOOM_LEVELS, detect_bloom_level
    from lib.ontology.learning_objectives import validate_lo_id
    from lib.ontology.slugs import canonical_slug

    if not isinstance(payload, dict):
        return None, None

    lo_entries_raw: List[Dict[str, Any]] = []
    flat = payload.get("learning_outcomes")
    if isinstance(flat, list) and flat:
        lo_entries_raw = [dict(e) for e in flat if isinstance(e, dict)]
    else:
        terminal = (
            payload.get("terminal_objectives")
            or payload.get("terminal_outcomes")
            or []
        )
        chapter_raw = (
            payload.get("chapter_objectives")
            or payload.get("component_objectives")
            or []
        )
        for t in terminal if isinstance(terminal, list) else []:
            if isinstance(t, dict):
                entry = dict(t)
                entry.setdefault("hierarchy_level", "terminal")
                lo_entries_raw.append(entry)
        for group in _normalize_chapter_objectives_to_groups(chapter_raw):
            for c in group.get("objectives") or []:
                if isinstance(c, dict):
                    entry = dict(c)
                    entry.setdefault("hierarchy_level", "chapter")
                    lo_entries_raw.append(entry)

    canonical_levels = set(BLOOM_LEVELS)
    learning_outcomes: List[Dict[str, Any]] = []
    objectives_metadata: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for entry in lo_entries_raw:
        lo_id = str(entry.get("id") or "").strip().upper()
        if not validate_lo_id(lo_id) or lo_id in seen_ids:
            continue
        seen_ids.add(lo_id)
        normalized = dict(entry)
        normalized["id"] = lo_id
        learning_outcomes.append(normalized)

        # targets-concept metadata (best-effort per LO).
        statement = str(
            entry.get("statement") or entry.get("description") or ""
        )
        key_concepts = entry.get("key_concepts") or entry.get("keyConcepts")
        if not isinstance(key_concepts, list) or not key_concepts:
            continue
        bloom_level = str(entry.get("bloom_level") or "").strip().lower()
        if bloom_level not in canonical_levels:
            bloom_level, _verb = detect_bloom_level(statement)  # type: ignore[assignment]
        if not bloom_level or bloom_level not in canonical_levels:
            # Neither an explicit canonical level nor a detectable verb. Rather
            # than drop the LO's targetedConcepts (and the targets-concept edges
            # that depend on them), fall back to a neutral default level. Loud
            # warning so an audit can see which LOs took the fallback.
            logger.warning(
                "objectives normalize: LO %s has no resolvable bloom_level "
                "(statement=%r, key_concepts=%d); falling back to %r so its "
                "targetedConcepts are not dropped",
                lo_id,
                statement[:80],
                len(key_concepts),
                _FALLBACK_BLOOM_LEVEL,
            )
            bloom_level = _FALLBACK_BLOOM_LEVEL
        targeted = [
            {"concept": canonical_slug(str(c)), "bloomLevel": bloom_level}
            for c in key_concepts
            if isinstance(c, str) and canonical_slug(c)
        ]
        if targeted:
            objectives_metadata.append(
                {"id": lo_id, "targetedConcepts": targeted}
            )

    if not learning_outcomes:
        return None, None
    course: Dict[str, Any] = {
        "course_id": course_code,
        "course_code": payload.get("course_code") or course_code,
        "learning_outcomes": learning_outcomes,
    }
    return course, (objectives_metadata or None)


def _resolve_course_objectives_for_graph(
    *,
    objectives_path_kw: str = "",
    synthesized_objectives_path_kw: str = "",
    project_path: Optional[Path] = None,
    libv2_course_dir: Optional[Path] = None,
    course_code: str = "",
) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], str]:
    """Resolve + normalize the course objectives doc for the semantic graph.

    Resolution chain (first parseable candidate that yields >=1 canonical
    LO wins):

    1. ``objectives_path_kw`` — explicit kwarg, routed from the
       ``reuse_objectives_path`` workflow param via the
       ``concept_extraction`` phase's ``inputs_from`` block. A reuse run
       pins the LO-dependent typed-edge rules to the operator's verbatim
       objectives doc.
    2. ``synthesized_objectives_path_kw`` — the
       ``synthesized_objectives.json`` emitted by ``course_planning``,
       routed via the ``concept_extraction`` phase's ``inputs_from``
       block. Phase-ordering fix (Option A1): ``concept_extraction`` now
       runs AFTER ``course_planning``, so this candidate exists on a
       FRESH ``textbook_to_course`` run (objectives are minted in the
       prior phase) — it is the fresh-run objectives source.
    3. ``<project export>/01_learning_objectives/synthesized_objectives.json``
       — present on re-runs / Phase-5 stage subcommands (also where
       candidate #2 normally points; kept as a path-independent fallback).
    4. ``<project export>/03_content_development/course.json`` — the
       packaging-shaped projection emitted by
       :func:`_project_synthesized_objectives_to_course_json`.
    5. ``LibV2/courses/<slug>/course.json`` — the canonical
       Trainforge-archived form.

    Fail-soft by contract: unreadable / malformed / LO-less candidates log
    a warning and fall through; an empty chain returns ``(None, None, "")``
    so the caller degrades to the legacy ``course=None`` build.

    Returns:
        ``(course, objectives_metadata, resolved_source_path)``.
    """
    candidates: List[Path] = []
    if objectives_path_kw:
        candidates.append(Path(objectives_path_kw))
    if synthesized_objectives_path_kw:
        candidates.append(Path(synthesized_objectives_path_kw))
    if project_path is not None:
        candidates.append(
            project_path / "01_learning_objectives"
            / "synthesized_objectives.json"
        )
        candidates.append(
            project_path / "03_content_development" / "course.json"
        )
    if libv2_course_dir is not None:
        candidates.append(libv2_course_dir / "course.json")

    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            payload = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "concept_extraction: objectives candidate %s is unreadable "
                "or malformed (%s); trying next candidate.",
                cand, exc,
            )
            continue
        course, objectives_metadata = _normalize_objectives_payload_to_course(
            payload, course_code,
        )
        if course is None:
            logger.warning(
                "concept_extraction: objectives candidate %s parsed but "
                "yielded zero canonical learning outcomes; trying next "
                "candidate.",
                cand,
            )
            continue
        return course, objectives_metadata, str(cand)
    return None, None, ""


_BLOOM_TO_DIFFICULTY: Dict[str, str] = {
    "remember": "foundational",
    "understand": "foundational",
    "apply": "intermediate",
    "analyze": "intermediate",
    "evaluate": "advanced",
    "create": "advanced",
}

# Resource types that cap difficulty at "foundational" (overviews / summaries
# never sit at advanced). Mirrors
# ``Trainforge/process_course.py::INTRODUCTORY_RESOURCE_TYPES``.
_INTRODUCTORY_RESOURCE_TYPES = {"overview", "summary"}


def _resolve_chunk_bloom_level(
    item: Dict[str, Any],
    text: str,
) -> Tuple[str, str]:
    """Resolve ``bloom_level`` for a chunk via the canonical cascade.

    Returns a ``(bloom_level, bloom_source)`` tuple. ``bloom_source`` is one
    of ``page_jsonld`` / ``lo_inherited`` / ``verbs`` / ``default`` (mirroring
    the values in ``schemas/knowledge/chunk_v4.schema.json::bloom_level_source``).
    The two highest-fidelity sources from the schema cascade
    (``section_jsonld`` and the per-section JSON-LD ``blocks[]`` match) are
    not reachable from this stripped-down callback — the chunking phase's
    parsed_item dict carries page-level ``courseforge_metadata`` but no
    section-keyed metadata index — so this helper picks up the cascade at
    ``page_jsonld`` and below. Callers that need section-level granularity
    should still go through ``CourseProcessor._create_chunk``.

    Cascade:
      1. ``item["courseforge_metadata"]["learningObjectives"][*]["bloomLevel"]``
         (page-level JSON-LD; first non-empty wins).
      2. ``item["learning_objectives"][*].bloom_level`` (parsed LO objects;
         falls through ``LearningObjective.bloom_level``).
      3. Text-verb heuristic via ``lib.ontology.bloom.detect_bloom_level``.
      4. Hardcoded ``"understand"`` default (per the schema's documented
         resolution cascade).
    """
    # 1. Page-level JSON-LD learningObjectives[].bloomLevel.
    cf_meta = item.get("courseforge_metadata")
    if cf_meta:
        for lo in (cf_meta.get("learningObjectives") or []):
            bl = (lo or {}).get("bloomLevel")
            if isinstance(bl, str) and bl.strip():
                return (bl.strip().lower(), "page_jsonld")

    # 2. Parsed LearningObjective dataclass list. Each entry exposes
    #    ``.bloom_level`` (the parser normalizes JSON-LD camelCase /
    #    data-cf-bloom-level / regex matches to this snake_case field).
    for lo in (item.get("learning_objectives") or []):
        bl = getattr(lo, "bloom_level", None)
        if bl is None and isinstance(lo, dict):
            bl = lo.get("bloom_level")
        if isinstance(bl, str) and bl.strip():
            return (bl.strip().lower(), "lo_inherited")

    # 3. Verb heuristic via the canonical detector.
    try:
        from lib.ontology.bloom import detect_bloom_level
    except Exception:  # noqa: BLE001 — import guard mirrors other lazy imports
        detect_bloom_level = None  # type: ignore[assignment]
    if detect_bloom_level is not None:
        level, _verb = detect_bloom_level(text or "")
        if level:
            return (level, "verbs")

    # 4. Default per the schema cascade.
    return ("understand", "default")


def _resolve_chunk_difficulty(
    item: Dict[str, Any],
    text: str,
    bloom_level: Optional[str] = None,
) -> str:
    """Resolve ``difficulty`` for a chunk via the canonical cascade.

    Mirrors ``Trainforge/process_course.py::_determine_difficulty``. Returns
    one of the chunk_v4 enum values: ``foundational`` / ``intermediate`` /
    ``advanced``.

    Cascade:
      1. JSON-LD ``learningObjectives[].bloomLevel`` via ``_BLOOM_TO_DIFFICULTY``
         (first hit wins).
      2. Parsed ``LearningObjective.bloom_level`` via ``_BLOOM_TO_DIFFICULTY``.
      3. Already-resolved ``bloom_level`` argument via ``_BLOOM_TO_DIFFICULTY``
         (the chunk's own bloom_level — closes the loop for chunks where
         the bloom signal came from the verb heuristic).
      4. Keyword heuristic over chunk text (introductory verbs ->
         foundational, advanced verbs -> advanced, else intermediate).

    Introductory resource types (``overview`` / ``summary``) cap at
    ``foundational`` regardless of cascade.
    """
    difficulty: Optional[str] = None

    # 1. Page-level JSON-LD learningObjectives.
    cf_meta = item.get("courseforge_metadata")
    if cf_meta:
        for lo in (cf_meta.get("learningObjectives") or []):
            bl = (lo or {}).get("bloomLevel")
            if isinstance(bl, str) and bl.strip().lower() in _BLOOM_TO_DIFFICULTY:
                difficulty = _BLOOM_TO_DIFFICULTY[bl.strip().lower()]
                break

    # 2. Parsed LearningObjective list.
    if difficulty is None:
        for lo in (item.get("learning_objectives") or []):
            bl = getattr(lo, "bloom_level", None)
            if bl is None and isinstance(lo, dict):
                bl = lo.get("bloom_level")
            if isinstance(bl, str) and bl.strip().lower() in _BLOOM_TO_DIFFICULTY:
                difficulty = _BLOOM_TO_DIFFICULTY[bl.strip().lower()]
                break

    # 3. Resolved bloom_level (closes the loop for verb-heuristic chunks).
    if (
        difficulty is None
        and isinstance(bloom_level, str)
        and bloom_level.strip().lower() in _BLOOM_TO_DIFFICULTY
    ):
        difficulty = _BLOOM_TO_DIFFICULTY[bloom_level.strip().lower()]

    # 4. Keyword heuristic over chunk text.
    if difficulty is None:
        text_lower = (text or "").lower()
        if any(
            kw in text_lower
            for kw in ("basic", "introduction", "overview", "what is", "define")
        ):
            difficulty = "foundational"
        elif any(
            kw in text_lower
            for kw in ("evaluate", "create", "design", "critique", "justify")
        ):
            difficulty = "advanced"
        else:
            difficulty = "intermediate"

    # Cap introductory resource types.
    if item.get("resource_type") in _INTRODUCTORY_RESOURCE_TYPES:
        if difficulty == "advanced":
            difficulty = "intermediate"
        elif difficulty == "intermediate":
            difficulty = "foundational"

    return difficulty


# Canonical sourceId shape: dart:{slug}#{block_id} (lowercase slug + block;
# mirrors schemas/knowledge/source_reference.schema.json). The validator owns
# the canonical pattern + the slug-derivation rule; we import both rather than
# re-spell them so the emitter and the source_refs validator can never drift
# (MCP/tools depending on lib/validators is the allowed layering direction).
# Used to drop any harvested block whose minted id wouldn't validate so a
# malformed DART attribute can't poison a chunk's source_references[].
from lib.validators.source_refs import (
    SOURCE_ID_RE as _DART_SOURCE_ID_RE,
    dart_slug_from_filename as _dart_slug_from_filename,
)


def _dart_block_source_references(
    dart_source_refs: Optional[List[Dict[str, Any]]],
    slug: str,
) -> List[Dict[str, Any]]:
    """Mint canonical SourceReference dicts from harvested DART block refs.

    ``dart_source_refs`` is the chunker's harvest output — an ordered list
    of ``{"block_id": str, "pages": List[int]}`` pairs read off
    ``data-dart-block-id`` / ``data-dart-pages`` attributes (see
    ``Trainforge.chunker.helpers.harvest_dart_source_refs``). ``slug`` is the
    staged-HTML file stem (passed as ``item["item_id"]`` by
    ``_run_dart_chunking``). We re-run the canonical
    ``dart_slug_from_filename`` rule over it here — DART's multi-source
    strategy emits ``{stem}_synthesized.html``, so ``item_id`` can carry a
    trailing ``_synthesized`` that the source_refs validator + source-router
    strip when they key ``dart:{slug}#...``. Stripping only at mint time
    keeps the sourceId join key resolvable WITHOUT mutating ``item_id``
    itself (it also flows into the chunk's ``module_id`` / ``lesson_id``,
    which must stay the literal file stem).

    Returns SourceReference dicts in the shape
    ``schemas/knowledge/source_reference.schema.json`` requires:
    ``{"sourceId": "dart:{slug}#{block_id}", "role": "primary",
    "extractor": "synthesized", "pages": [N, ...]}``. ``role`` is
    ``primary`` — a chunk built from a DART block IS that source. ``pages``
    is omitted when empty (schema requires ``pages`` items ``minimum: 1``).
    Any minted sourceId that fails the canonical pattern is dropped so a
    malformed attribute never produces an unresolvable ref.

    Returns ``[]`` when ``dart_source_refs`` or ``slug`` is empty — the
    additive contract: chunks from HTML without ``data-dart-*`` attributes
    keep ``source.source_references`` unset (legacy corpora byte-stable).
    """
    if not dart_source_refs or not slug:
        return []
    # Canonicalize the slug (strip ``_synthesized``, lowercase, spaces→hyphens)
    # so minted sourceIds match the validator's / source-router's join keys on
    # real multi-source corpora; ``item_id`` itself is left untouched upstream.
    slug = _dart_slug_from_filename(slug)
    if not slug:
        return []
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for ref in dart_source_refs:
        if not isinstance(ref, dict):
            continue
        block_id = str(ref.get("block_id") or "").strip()
        if not block_id:
            continue
        source_id = f"dart:{slug}#{block_id}"
        if source_id in seen or not _DART_SOURCE_ID_RE.match(source_id):
            continue
        seen.add(source_id)
        entry: Dict[str, Any] = {
            "sourceId": source_id,
            "role": "primary",
            "extractor": "synthesized",
        }
        pages = [int(p) for p in (ref.get("pages") or []) if int(p) > 0]
        if pages:
            entry["pages"] = sorted(set(pages))
        out.append(entry)
    return out


def _backfill_dart_chunk_lo_refs(
    *,
    course_slug: str,
    objective_ids: List[str],
    libv2_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Wave3-Anew3 — back-fill ``learning_outcome_refs`` on DART chunks.

    Closes auditor Finding F3 of
    ``plans/dispatch-7-final-product-audit-2026-05.md``: the DART
    ``_run_dart_chunking`` phase emits chunks BEFORE
    ``_plan_course_structure`` synthesizes the TO-NN / CO-NN ID set, so
    every chunk's ``learning_outcome_refs[]`` is empty on initial emit.
    This helper runs after course planning publishes ``objective_ids``
    (Wave2-I7 plumbing), re-opens the on-disk chunks JSONL, text-scans
    each chunk's ``text`` + ``html`` for canonical LO IDs via
    :func:`lib.ontology.learning_objectives.scan_lo_refs`, and writes
    matches back into ``learning_outcome_refs`` — but only IDs that
    appear in ``objective_ids`` (the false-positive guard the brief
    mandates).

    Args:
        course_slug: Course slug used to locate
            ``<libv2>/courses/<slug>/dart_chunks/chunks.jsonl``.
        objective_ids: Allowlist of canonical TO-NN / CO-NN IDs minted
            by course planning. Empty list → no-op (cannot back-fill
            without an allowlist; preserves false-positive guard).
        libv2_root: Optional override for the LibV2 root; resolution
            chain follows :func:`_resolve_libv2_root`.

    Returns:
        Summary dict with ``chunks_path``, ``chunks_scanned``,
        ``chunks_updated``, ``new_refs_total``, and (when the chunks
        file is missing) ``skipped`` + ``reason``.

    Best-effort: missing chunks file / unreadable JSONL line / write
    error each emit a logger.warning and continue rather than crash the
    workflow. Existing non-empty ``learning_outcome_refs`` are preserved
    (additive union semantics — never destructive).
    """
    from lib.ontology.learning_objectives import scan_lo_refs

    if not objective_ids:
        return {
            "chunks_path": None,
            "chunks_scanned": 0,
            "chunks_updated": 0,
            "new_refs_total": 0,
            "skipped": True,
            "reason": "empty_objective_ids_allowlist",
        }

    chunks_path = (
        _resolve_libv2_root(libv2_root)
        / "courses"
        / course_slug
        / "dart_chunks"
        / "chunks.jsonl"
    )
    if not chunks_path.exists() or not chunks_path.is_file():
        return {
            "chunks_path": str(chunks_path),
            "chunks_scanned": 0,
            "chunks_updated": 0,
            "new_refs_total": 0,
            "skipped": True,
            "reason": "chunks_jsonl_missing",
        }

    allow = list(objective_ids)
    chunks_scanned = 0
    chunks_updated = 0
    new_refs_total = 0
    updated_lines: List[str] = []

    try:
        raw_lines = chunks_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning(
            "Wave3-Anew3: failed to read %s (%s); skipping LO back-fill.",
            chunks_path, exc,
        )
        return {
            "chunks_path": str(chunks_path),
            "chunks_scanned": 0,
            "chunks_updated": 0,
            "new_refs_total": 0,
            "skipped": True,
            "reason": f"read_error:{exc}",
        }

    for line in raw_lines:
        if not line.strip():
            updated_lines.append(line)
            continue
        try:
            chunk = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Wave3-Anew3: malformed chunk JSONL line in %s (%s); "
                "preserving verbatim.",
                chunks_path, exc,
            )
            updated_lines.append(line)
            continue

        chunks_scanned += 1
        existing = chunk.get("learning_outcome_refs") or []
        existing_set = {str(r).upper() for r in existing if r}
        scanned = scan_lo_refs(
            text=chunk.get("text") or "",
            html=chunk.get("html") or "",
            allowed_ids=allow,
        )
        # Union semantics — preserve any pre-existing refs.
        merged = sorted(existing_set | set(scanned))
        added = len(merged) - len(existing_set)
        if added > 0:
            chunks_updated += 1
            new_refs_total += added
            chunk["learning_outcome_refs"] = merged
            updated_lines.append(json.dumps(chunk, ensure_ascii=False))
        else:
            # No new refs — keep original bytes / order intact.
            updated_lines.append(line)

    if chunks_updated > 0:
        try:
            chunks_path.write_text(
                "\n".join(updated_lines) + ("\n" if updated_lines else ""),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Wave3-Anew3: failed to write back-filled chunks to %s "
                "(%s); leaving original on disk.",
                chunks_path, exc,
            )
            return {
                "chunks_path": str(chunks_path),
                "chunks_scanned": chunks_scanned,
                "chunks_updated": 0,
                "new_refs_total": 0,
                "skipped": True,
                "reason": f"write_error:{exc}",
            }

    return {
        "chunks_path": str(chunks_path),
        "chunks_scanned": chunks_scanned,
        "chunks_updated": chunks_updated,
        "new_refs_total": new_refs_total,
        "skipped": False,
    }


def _detect_source_provenance(course_dir: Path) -> bool:
    """Wave 10: scan archived chunks.jsonl for chunks with source_references[].

    Returns True when at least one chunk in ``<course_dir>/corpus/chunks.jsonl``
    carries ``source.source_references[]`` populated with at least one entry.
    Returns False on missing file, read errors, malformed JSONL lines, or
    when no chunks carry refs (pre-Wave-9 corpus). The manifest then advertises
    ``features.source_provenance: false`` so LibV2 retrieval callers can
    fast-skip source-grounded queries.
    """
    # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/ via shim.
    from lib.libv2_storage import resolve_imscc_chunks_path
    chunks_path = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
    if not chunks_path.exists() or not chunks_path.is_file():
        return False
    try:
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(chunk, dict):
                    continue
                source = chunk.get("source")
                if not isinstance(source, dict):
                    continue
                refs = source.get("source_references")
                if isinstance(refs, list) and len(refs) > 0:
                    return True
    except OSError:
        return False
    return False


def _detect_evidence_source_provenance(course_dir: Path) -> bool:
    """Wave 11: scan archived concept_graph_semantic.json for evidence-level refs.

    Returns True when at least one edge in the archived concept graph's
    ``edges[].provenance.evidence`` carries a populated ``source_references[]``
    array. False on missing file, read errors, malformed JSON, or when no
    edges carry evidence refs. The manifest then advertises
    ``features.evidence_source_provenance: true/false`` so LibV2 retrieval
    callers can distinguish chunk-level (Wave 10) from evidence-level (Wave 11)
    provenance.

    The scan looks in candidate locations under ``<course_dir>``:
    ``graph/concept_graph_semantic.json``, ``imscc_chunks/concept_graph_semantic.json``,
    or legacy ``corpus/concept_graph_semantic.json``, or any ``*.json``
    file shaped like a semantic graph (``kind == "concept_semantic"``)
    sitting inside the chunkset dir. First match wins.
    """
    candidates = [
        course_dir / "graph" / "concept_graph_semantic.json",
        course_dir / "imscc_chunks" / "concept_graph_semantic.json",
        course_dir / "corpus" / "concept_graph_semantic.json",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    graph = json.load(f)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if _graph_has_evidence_refs(graph):
                return True
            # First readable candidate wins — don't fall through to others
            # if this one was valid shape but carried no refs.
            return False
    return False


def _graph_has_evidence_refs(graph: object) -> bool:
    """Return True iff the graph has at least one edge whose
    ``provenance.evidence.source_references`` is a non-empty list.

    Tolerates partial / legacy shapes: silently returns False on any
    structural surprise rather than raising.
    """
    if not isinstance(graph, dict):
        return False
    edges = graph.get("edges")
    if not isinstance(edges, list):
        return False
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        provenance = edge.get("provenance")
        if not isinstance(provenance, dict):
            continue
        evidence = provenance.get("evidence")
        if not isinstance(evidence, dict):
            continue
        refs = evidence.get("source_references")
        if isinstance(refs, list) and len(refs) > 0:
            return True
    return False


# Wave 32 Deliverable C: phase-level empty-content guard for
# content_generation. Runs inline at the end of _generate_course_content
# so a dispatcher that returned zero real body content fails the phase
# loudly rather than passing with ``gates=pass`` on template skeletons.
# Reuses the Wave 31 ContentGroundingValidator's 30-word floor for
# behavioural consistency with the content_grounding gate — this check
# catches the strict "every page is an empty skeleton" failure mode the
# gate considers a warning when partial (< 25 %). Independent of the
# gate: gates require routing + inputs to fire, and when routing skips
# we want a phase-level guarantee that the dispatcher produced at least
# one non-trivial page.
_CONTENT_BODY_TAGS = ("p", "li", "blockquote", "figcaption")
_CONTENT_NONTRIVIAL_WORD_FLOOR = 30


def _check_content_nonempty(page_paths: list) -> "Optional[str]":
    """Return an error message when every emitted page is an empty template.

    Parses each page and counts words in body-text tags
    (``<p>``/``<li>``/``<blockquote>``/``<figcaption>``) inside
    ``<main>`` (or the document body when no main wrapper exists).
    Returns ``None`` when at least one page clears
    :data:`_CONTENT_NONTRIVIAL_WORD_FLOOR` words — otherwise returns an
    actionable error string that mentions the LOCAL_DISPATCHER_ALLOW_STUB
    bypass and the missing agent_tool wiring.

    Contract:
      * Empty ``page_paths`` → returns ``None`` (nothing to check —
        upstream already bailed out with an error when it mattered).
      * Unreadable / missing files are counted as empty.
    """
    if not page_paths:
        return None

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:  # pragma: no cover — bs4 is a hard dep in this repo
        # Without BeautifulSoup we can't reliably parse body content, so
        # fall back to a plain word-count heuristic on the raw file.
        def _plain_word_count(text: str) -> int:
            import re as _re_inner
            return len(_re_inner.findall(r"\b\w+\b", text))

        for p in page_paths:
            try:
                raw = Path(p).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _plain_word_count(raw) >= _CONTENT_NONTRIVIAL_WORD_FLOOR * 2:
                return None
        return (
            "CONTENT_GENERATION_EMPTY: All "
            f"{len(page_paths)} generated pages have <"
            f"{_CONTENT_NONTRIVIAL_WORD_FLOOR} body words each. "
            "This indicates the content-gen dispatcher produced template "
            "skeletons without filling them. Likely cause: --mode local "
            "dispatcher not wired to an actual agent_tool. See "
            "LOCAL_DISPATCHER_ALLOW_STUB for the bypass."
        )

    total = len(page_paths)
    nonempty = 0
    for p in page_paths:
        try:
            raw = Path(p).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        try:
            soup = BeautifulSoup(raw, "html.parser")
        except Exception:  # noqa: BLE001
            continue
        # Scope to <main> when present; otherwise the whole body.
        scope = soup.find("main")
        if scope is None:
            scope = soup.find(attrs={"role": "main"})
        if scope is None:
            scope = soup.body or soup
        # Strip nav/header/footer from the scope so their paragraphs
        # don't pollute the count (mirrors ContentGroundingValidator).
        for tag in scope.find_all(["nav", "header", "footer"]):
            tag.decompose()
        for el in scope.find_all(_CONTENT_BODY_TAGS):
            text = el.get_text(separator=" ", strip=True)
            if len(text.split()) >= _CONTENT_NONTRIVIAL_WORD_FLOOR:
                nonempty += 1
                break
        else:
            continue
        # Early exit once we see any non-trivial page — the phase
        # guarantee is "at least one page with real content".
        if nonempty >= 1:
            return None

    if nonempty >= 1:
        return None
    return (
        "CONTENT_GENERATION_EMPTY: All "
        f"{total} generated pages have <"
        f"{_CONTENT_NONTRIVIAL_WORD_FLOOR} body words each. "
        "This indicates the content-gen dispatcher produced template "
        "skeletons without filling them. Likely cause: --mode local "
        "dispatcher not wired to an actual agent_tool. See "
        "LOCAL_DISPATCHER_ALLOW_STUB for the bypass."
    )


async def create_textbook_pipeline(
    pdf_paths: str,
    course_name: str,
    objectives_path: Optional[str] = None,
    duration_weeks: int = 12,
    generate_assessments: bool = True,
    assessment_count: int = 50,
    bloom_levels: str = "remember,understand,apply,analyze",
    priority: str = "normal",
    duration_weeks_explicit: bool = True,
    skip_dart: bool = False,
    dart_output_dir: Optional[str] = None,
    reuse_objectives_path: Optional[str] = None,
    courseforge_stage: Optional[str] = None,
    force_rerun: bool = False,
    skip_training: bool = False,
) -> str:
    """
    Create and orchestrate a textbook-to-course pipeline.

    Chains: DART (PDF->HTML) -> Courseforge (course generation) -> Trainforge (assessments)

    This is a standalone function importable by both the MCP server and CLI.

    Args:
        pdf_paths: Comma-separated PDF paths OR directory containing PDFs
        course_name: Course identifier (e.g., "PHYS_101")
        objectives_path: Optional external objectives file to merge
        duration_weeks: Course duration in weeks (default: 12)
        generate_assessments: Run Trainforge phase (default: True)
        assessment_count: Questions to generate (default: 50)
        bloom_levels: Target Bloom levels (default: remember,understand,apply,analyze)
        priority: Workflow priority (low/normal/high)
        duration_weeks_explicit: Wave 39 follow-up. When ``False`` (the
            caller did NOT pass ``--weeks``), the extractor phase
            (``_extract_textbook_structure``) auto-scales
            ``duration_weeks`` to ``max(8, chapter_count)`` once the
            textbook structure is known. Defaults to ``True`` so legacy
            callers keep the historical fixed-12 behaviour.

    Returns:
        JSON with workflow_id, run_id, and status
    """
    try:
        from MCP.tools.orchestrator_tools import create_workflow_impl

        # Courseforge stage subcommands work off an existing project
        # export — DART / staging / objective_extraction etc. are
        # pre-populated via _synthesize_outline_output, so corpus PDFs
        # are not consumed. Skip PDF validation entirely when
        # courseforge_stage is set.
        if courseforge_stage:
            pdfs = []
        else:
            # Parse PDF paths
            pdf_path = Path(pdf_paths)
            if pdf_path.is_dir():
                pdfs = list(pdf_path.glob("*.pdf"))
                if not pdfs:
                    return json.dumps({"error": f"No PDF files found in directory: {pdf_paths}"})
            else:
                pdfs = [Path(p.strip()) for p in pdf_paths.split(",")]

            # Validate PDF paths are within project root
            for pdf in pdfs:
                try:
                    validate_path_within_root(pdf.resolve(), PROJECT_ROOT)
                except ValueError as e:
                    return json.dumps({"error": f"PDF path validation failed: {e}"})

            # Validate inputs
            missing_pdfs = [str(p) for p in pdfs if not p.exists()]
            if missing_pdfs:
                return json.dumps({"error": f"PDF files not found: {missing_pdfs}"})

        if objectives_path and not Path(objectives_path).exists():
            return json.dumps({"error": f"Objectives file not found: {objectives_path}"})

        # Validate course name format
        if not course_name or len(course_name) < 2:
            return json.dumps({"error": "Course name must be at least 2 characters"})

        # Generate run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"TTC_{course_name}_{timestamp}"

        # Wave 29 Defect 5: compute the canonical course code ONCE
        # from ``course_name`` (the CLI-supplied / caller-supplied
        # value) and pin it on the params dict. Every DecisionCapture
        # instantiated anywhere in this run reads from this field
        # instead of re-deriving from a PDF name, a workflow_id hash,
        # or the workflow type. That keeps captures, archives, and
        # CF/TF phase data tagged with a single consistent code —
        # fixing the SIM_RUN_01 four-codes-in-one-run observation.
        from lib.decision_capture import normalize_course_code as _normalize_cc

        canonical_cc = _normalize_cc(course_name)

        # Build workflow parameters. Wave 39 follow-up: propagate the
        # ``duration_weeks_explicit`` flag so ``_extract_textbook_structure``
        # sees it via kwargs and auto-scales ``duration_weeks`` to
        # ``max(8, chapter_count)`` when the CLI caller omitted
        # ``--weeks``. Pre-Wave-39-follow-up, this function hard-coded
        # ``duration_weeks=12`` into the workflow state regardless of
        # intent, so the auto-scale branch in the extractor was
        # effectively dead code on the real run path.
        params = {
            "pdf_paths": [str(p.resolve()) for p in pdfs],
            "course_name": course_name,
            "canonical_course_code": canonical_cc,
            "objectives_path": str(Path(objectives_path).resolve()) if objectives_path else None,
            "duration_weeks": duration_weeks,
            "duration_weeks_explicit": bool(duration_weeks_explicit),
            "generate_assessments": generate_assessments,
            "assessment_count": assessment_count,
            "bloom_levels": [level.strip() for level in bloom_levels.split(",")],
            "run_id": run_id
        }
        # Wave 74 Session 3: forward --skip-dart so the workflow runner
        # can synthesize the dart_conversion phase_output from an
        # existing DART/output/ directory instead of re-running the
        # PDF->HTML conversion.
        if skip_dart:
            params["skip_dart"] = True
            if dart_output_dir:
                params["dart_output_dir"] = str(Path(dart_output_dir).resolve())

        # Wave 80 Worker A: forward --reuse-objectives so the workflow
        # runner can synthesize the course_planning phase_output from
        # the user-supplied objectives JSON instead of dispatching the
        # course-outliner subagent. Stable across re-runs (no LLM
        # nondeterminism), preserving chunk learning_outcome_refs
        # continuity.
        if reuse_objectives_path:
            params["reuse_objectives_path"] = str(
                Path(reuse_objectives_path).resolve()
            )

        # Phase 5 operator stage subcommands: restrict execution to the
        # named Courseforge stage whitelist, optionally force re-run of
        # checkpointed phases, and optionally skip training_synthesis.
        if courseforge_stage:
            params["courseforge_stage"] = courseforge_stage
        if force_rerun:
            params["force_rerun"] = True
        if skip_training:
            params["skip_training"] = True

        # Create workflow via orchestrator
        result = await create_workflow_impl(
            workflow_type="textbook_to_course",
            params=json.dumps(params),
            priority=priority
        )

        result_data = json.loads(result)

        if result_data.get("success"):
            # Add run_id to response
            result_data["run_id"] = run_id
            result_data["params"] = params

            # Create training captures directory for this run
            captures_dir = TRAINING_CAPTURES / "textbook-pipeline" / course_name
            captures_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Created textbook_to_course pipeline: {result_data.get('workflow_id')}")

        return json.dumps(result_data)

    except Exception as e:
        logger.error(f"Failed to create textbook pipeline: {e}")
        return json.dumps({"error": str(e)})


async def run_textbook_pipeline(workflow_id: str) -> str:
    """
    Execute a textbook-to-course pipeline that was previously created.

    Standalone function importable by both MCP server and CLI.

    Runs all phases in dependency order:
    DART conversion -> Staging -> Objective extraction -> Course planning ->
    Content generation -> IMSCC packaging -> Trainforge assessment ->
    LibV2 archival -> Finalization

    Args:
        workflow_id: The workflow ID returned by create_textbook_pipeline

    Returns:
        JSON with final status, phase results, and output paths
    """
    try:
        from MCP.core.config import OrchestratorConfig
        from MCP.core.executor import TaskExecutor
        from MCP.core.workflow_runner import WorkflowRunner

        # Load orchestrator config
        config = OrchestratorConfig.load()

        # Create executor with tool registry
        tool_registry = _build_tool_registry()

        executor = TaskExecutor(tool_registry=tool_registry)

        # Create and run the workflow runner
        runner = WorkflowRunner(executor, config)
        result = await runner.run_workflow(workflow_id)

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "workflow_id": workflow_id,
        })


def register_pipeline_tools(mcp):
    """Register pipeline tools with the MCP server."""

    # Wave 28f: create_textbook_pipeline_tool was removed.
    # External MCP clients now route through the workflow API
    # (``create_workflow(workflow_type='textbook_to_course', ...)``) or
    # ``ed4all run textbook-to-course``. The underlying non-tool
    # ``create_textbook_pipeline()`` coroutine above remains for
    # internal callers (e.g. cli/commands/run.py).

    @mcp.tool()
    async def stage_dart_outputs(
        run_id: str,
        dart_html_paths: str,
        course_name: str,
        stage_mode: Optional[str] = None,
    ) -> str:
        """
        Stage DART outputs to Courseforge inputs directory.

        Stages synthesized HTML and JSON files from DART output to the
        Courseforge staging area for course generation. The default
        ``stage_mode`` is ``symlink`` (zero-byte references back to DART
        outputs) which avoids duplicating ~70MB per textbook-to-course run.
        Set ``stage_mode="copy"`` for the legacy deep-copy behaviour.

        Args:
            run_id: Pipeline run identifier
            dart_html_paths: Comma-separated paths to DART HTML outputs
            course_name: Course identifier for staging subdirectory
            stage_mode: One of ``"copy"``, ``"symlink"``, ``"hardlink"``.
                Defaults to ``ED4ALL_STAGE_MODE`` env var, then ``"symlink"``.

        Returns:
            JSON with staging_dir, staged_files list, and stage_mode used.
        """
        try:
            mode = _resolve_stage_mode(stage_mode)

            # Create staging directory
            staging_dir = COURSEFORGE_INPUTS / run_id
            staging_dir.mkdir(parents=True, exist_ok=True)

            staged_files = []
            # Wave 8: role-tagged manifest entries for the downstream
            # Courseforge source-router and Trainforge parser. Roles:
            #   "content"             -> the rendered HTML page
            #   "provenance_sidecar"  -> *_synthesized.json with per-block provenance
            #   "quality_sidecar"     -> *.quality.json with WCAG + confidence aggregates
            staged_entries = []
            errors = []

            html_paths = [Path(p.strip()) for p in dart_html_paths.split(",")]

            for html_path in html_paths:
                if not html_path.exists():
                    errors.append(f"DART output not found: {html_path}")
                    continue

                # Stage HTML file (role=content)
                dest = staging_dir / html_path.name
                _stage_file(html_path, dest, mode)
                staged_files.append(str(dest))
                staged_entries.append({"path": html_path.name, "role": "content"})
                logger.info(f"Staged ({mode}): {html_path.name} -> {dest}")

                # Wave 19: stage the sibling ``{stem}_figures/`` directory
                # (persisted PyMuPDF figure bytes from Wave 17) so the
                # Courseforge generator renders ``<img src>`` paths that
                # actually resolve. Missing directory is silently skipped
                # for backward compat with pre-Wave-17 outputs.
                figures_dir_src = html_path.parent / f"{html_path.stem}_figures"
                if figures_dir_src.is_dir():
                    figures_dir_dest = staging_dir / figures_dir_src.name
                    _stage_tree(figures_dir_src, figures_dir_dest, mode)
                    staged_files.append(str(figures_dir_dest))
                    staged_entries.append({
                        "path": figures_dir_src.name,
                        "role": "figures_bundle",
                    })
                    logger.info(
                        f"Staged figures dir ({mode}): {figures_dir_src.name} -> {figures_dir_dest}"
                    )

                # Validate HTML structure
                if html_path.suffix.lower() in ('.html', '.htm'):
                    try:
                        content = dest.read_text(encoding='utf-8', errors='ignore')[:5000]
                        content_lower = content.lower()
                        if '<html' not in content_lower and '<body' not in content_lower:
                            errors.append(
                                f"Warning: {html_path.name} may not be valid HTML "
                                f"(missing <html> and <body> tags)"
                            )
                    except OSError:
                        pass  # File was staged, just can't validate

                # Stage accompanying JSON if exists (DART synthesized metadata)
                json_path = html_path.with_suffix(".json")
                if json_path.exists():
                    json_dest = staging_dir / json_path.name
                    _stage_file(json_path, json_dest, mode)
                    staged_files.append(str(json_dest))
                    staged_entries.append({
                        "path": json_path.name,
                        "role": "provenance_sidecar",
                    })
                    logger.info(f"Staged ({mode}): {json_path.name} -> {json_dest}")

                # Also check for _synthesized.json pattern
                synth_json_name = html_path.stem.replace("_synthesized", "") + "_synthesized.json"
                synth_json_path = html_path.parent / synth_json_name
                if synth_json_path.exists() and str(synth_json_path) != str(json_path):
                    synth_json_dest = staging_dir / synth_json_name
                    _stage_file(synth_json_path, synth_json_dest, mode)
                    staged_files.append(str(synth_json_dest))
                    staged_entries.append({
                        "path": synth_json_name,
                        "role": "provenance_sidecar",
                    })
                    logger.info(f"Staged ({mode}): {synth_json_name} -> {synth_json_dest}")

                # Wave 8: also stage the DART quality sidecar if one exists.
                # Convention: same stem as the HTML, suffix .quality.json.
                # E.g. "science_of_learning.html" -> "science_of_learning.quality.json".
                # The legacy stage_dart_outputs never copied this even though
                # DART's convert_single_pdf has been writing it all along.
                quality_name = html_path.stem + ".quality.json"
                quality_path = html_path.parent / quality_name
                if quality_path.exists():
                    quality_dest = staging_dir / quality_name
                    _stage_file(quality_path, quality_dest, mode)
                    staged_files.append(str(quality_dest))
                    staged_entries.append({
                        "path": quality_name,
                        "role": "quality_sidecar",
                    })
                    logger.info(f"Staged ({mode}): {quality_name} -> {quality_dest}")

            if errors and not staged_files:
                return json.dumps({
                    "success": False,
                    "error": "No files staged",
                    "errors": errors
                })

            # Create manifest (Wave 8: role-tagged entries under "files")
            manifest = {
                "run_id": run_id,
                "course_name": course_name,
                "staged_at": datetime.now().isoformat(),
                "staged_files": staged_files,            # back-compat flat list
                "files": staged_entries,                 # role-tagged entries
                "errors": errors if errors else None,
            }

            manifest_path = staging_dir / "staging_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            return json.dumps({
                "success": True,
                "staging_dir": str(staging_dir),
                "staged_files": staged_files,
                "files": staged_entries,
                "file_count": len(staged_files),
                "manifest_path": str(manifest_path),
                "stage_mode": mode,
                "warnings": errors if errors else None
            })

        except Exception as e:
            logger.error(f"Failed to stage DART outputs: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_pipeline_status(workflow_id: str) -> str:
        """
        Get status of a textbook-to-course pipeline.

        Args:
            workflow_id: The workflow ID returned by create_textbook_pipeline

        Returns:
            JSON with current phase, progress, and phase outputs
        """
        try:
            # Read workflow state directly (get_workflow_status is a closure
            # inside register_orchestrator_tools, not importable at module level)
            workflow_path = PROJECT_ROOT / "state" / "workflows" / f"{workflow_id}.json"
            if not workflow_path.exists():
                return json.dumps({"error": f"Workflow not found: {workflow_id}"})
            with open(workflow_path) as f:
                workflow = json.load(f)

            # Enhance with pipeline-specific information
            params = workflow.get("params", {})

            pipeline_status = {
                "workflow_id": workflow.get("id"),
                "workflow_type": workflow.get("type"),
                "status": workflow.get("status"),
                "run_id": params.get("run_id"),
                "course_name": params.get("course_name"),
                "progress": workflow.get("progress"),
                "created_at": workflow.get("created_at"),
                "updated_at": workflow.get("updated_at"),
                "phases": {
                    "dart_conversion": _get_phase_status(workflow, "dart_conversion"),
                    "staging": _get_phase_status(workflow, "staging"),
                    "objective_extraction": _get_phase_status(workflow, "objective_extraction"),
                    "course_planning": _get_phase_status(workflow, "course_planning"),
                    "content_generation": _get_phase_status(workflow, "content_generation"),
                    "packaging": _get_phase_status(workflow, "packaging"),
                    "trainforge_assessment": _get_phase_status(workflow, "trainforge_assessment"),
                    "libv2_archival": _get_phase_status(workflow, "libv2_archival"),
                    "finalization": _get_phase_status(workflow, "finalization")
                },
                "params": params
            }

            return json.dumps(pipeline_status)

        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def validate_dart_markers(html_path: str) -> str:
        """
        Validate that an HTML file has required DART markers.

        DART-processed HTML must have:
        - Skip link (<a class="skip-link">)
        - Main content area (<main role="main">)
        - Semantic sections (<section aria-labelledby="...">)

        Args:
            html_path: Path to HTML file to validate

        Returns:
            JSON with validation results
        """
        try:
            path = Path(html_path)
            if not path.exists():
                return json.dumps({"error": f"File not found: {html_path}"})

            with open(path, encoding="utf-8") as f:
                content = f.read()

            markers = {
                "skip_link": 'class="skip' in content or "class='skip" in content,
                "main_role": 'role="main"' in content or "role='main'" in content,
                "aria_sections": 'aria-labelledby="' in content or "aria-labelledby='" in content,
                "dart_semantic_classes": 'dart-section' in content or 'dart-document' in content
            }

            all_valid = all(markers.values())

            result = {
                "valid": all_valid,
                "file": str(path),
                "markers": markers,
                "missing": [k for k, v in markers.items() if not v]
            }

            if not all_valid:
                result["message"] = f"Missing DART markers: {result['missing']}"

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Failed to validate DART markers: {e}")
            return json.dumps({"error": str(e)})


    @mcp.tool()
    async def synthesize_training(
        corpus_dir: str,
        course_code: str,
        provider: str = "mock",
        seed: Optional[int] = None,
        with_kg_metadata: bool = False,
        kg_metadata_max_pairs: int = 2000,
        with_violation_detection: bool = False,
        violation_detection_max_pairs: Optional[int] = None,
        with_abstention: bool = False,
        abstention_max_pairs: int = 1000,
        with_schema_translation: bool = False,
        schema_translation_max_pairs: int = 50,
    ) -> str:
        """Generate SFT + DPO training pairs from a Trainforge corpus.

        Wave 30 Gap 3: exposes
        :func:`Trainforge.synthesize_training.run_synthesis` as an MCP
        tool so external clients + the textbook_to_course pipeline both
        route to the same backing implementation. Reads
        ``{corpus_dir}/corpus/chunks.jsonl`` and writes
        ``{corpus_dir}/training_specs/instruction_pairs.jsonl`` +
        ``{corpus_dir}/training_specs/preference_pairs.jsonl``.

        Args:
            corpus_dir: Trainforge output directory (the one containing
                ``corpus/`` and ``training_specs/``).
            course_code: Course identifier for decision capture.
            provider: Synthesis provider. Accepted values:

                * ``"mock"`` (default) — deterministic template factory.
                * ``"anthropic"`` — Anthropic SDK paraphrase pass
                  (requires ``ANTHROPIC_API_KEY``). Anthropic's ToS
                  forbids using the output as training data, so use only
                  for in-house evaluation, not for SLM training corpora.
                * ``"claude_session"`` — Claude Code session via
                  LocalDispatcher (Claude Max path).
                * ``"together"`` — Together AI's OpenAI-compatible
                  chat-completions endpoint, default model
                  ``meta-llama/Llama-3.3-70B-Instruct-Turbo``
                  (override via ``TOGETHER_SYNTHESIS_MODEL``; requires
                  ``TOGETHER_API_KEY``). Together's ToS explicitly
                  permits using the output as training data — this is
                  the ToS-clean teacher pass for SLM corpora.
                * ``"local"`` — a local OpenAI-compatible model server
                  (Ollama / vLLM / llama.cpp / LM Studio). Default base
                  URL ``http://localhost:11434/v1`` (override via
                  ``LOCAL_SYNTHESIS_BASE_URL``); default model
                  ``qwen2.5:14b-instruct-q4_K_M`` (override via
                  ``LOCAL_SYNTHESIS_MODEL``). API key optional. Zero
                  per-call cost, zero ToS exposure (fully offline /
                  air-gapped friendly); tradeoff is local hardware.
            seed: Optional base seed for determinism.
            with_kg_metadata: Enable the deterministic kg_metadata
                generator (Wave 124a). Reads pedagogy_graph.json. No-op
                when the graph is absent.
            kg_metadata_max_pairs: Cap on kg_metadata pairs (default
                2000).
            with_violation_detection: Enable the deterministic SHACL
                violation generator (Wave 125a; pyshacl-oracle-verified).
            violation_detection_max_pairs: Cap on violation pairs
                (default unset = unlimited; family-balanced round-robin
                trim when set).
            with_abstention: Enable the deterministic abstention
                generator (Wave 124). Emits "the source does not
                establish X" probes from concepts the chunk does NOT
                address. Reads pedagogy_graph.json.
            abstention_max_pairs: Cap on abstention pairs (default
                1000).
            with_schema_translation: Enable the deterministic
                schema-translation generator (Wave 125b). Emits 6
                families × 6 surface forms (definition / usage /
                comparison / reasoning / pitfall / combination).
            schema_translation_max_pairs: Cap on schema-translation
                pairs (default 50).

        Returns:
            JSON with ``success``, the two output paths, and a stats
            summary. When ``chunks.jsonl`` is missing the call returns
            ``{"success": true, "skipped": true}`` so callers never
            crash on the no-LLM-available / no-corpus path.
        """
        try:
            from Trainforge.synthesize_training import (
                DEFAULT_SEED,
                run_synthesis,
            )
        except Exception as exc:
            return json.dumps({
                "error": f"Failed to import synthesize_training: {exc}",
            })

        corpus_dir_path = Path(corpus_dir)
        # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
        from lib.libv2_storage import resolve_imscc_chunks_path
        chunks_path = resolve_imscc_chunks_path(corpus_dir_path, "chunks.jsonl")
        if not chunks_path.exists():
            logger.warning(
                "synthesize_training: chunks.jsonl missing at %s; skipping",
                chunks_path,
            )
            return json.dumps({
                "success": True,
                "skipped": True,
                "reason": "chunks_missing",
                "corpus_dir": str(corpus_dir_path),
            })

        if seed is None:
            seed = DEFAULT_SEED

        try:
            stats = run_synthesis(
                corpus_dir=corpus_dir_path,
                course_code=course_code,
                provider=provider,
                seed=int(seed),
                with_kg_metadata=with_kg_metadata,
                kg_metadata_max_pairs=kg_metadata_max_pairs,
                with_violation_detection=with_violation_detection,
                violation_detection_max_pairs=violation_detection_max_pairs,
                with_abstention=with_abstention,
                abstention_max_pairs=abstention_max_pairs,
                with_schema_translation=with_schema_translation,
                schema_translation_max_pairs=schema_translation_max_pairs,
            )
        except Exception as exc:
            return json.dumps({
                "error": f"synthesize_training failed: {exc}",
                "corpus_dir": str(corpus_dir_path),
            })

        return json.dumps({
            "success": True,
            "corpus_dir": str(corpus_dir_path),
            "instruction_pairs_path": str(
                corpus_dir_path / "training_specs" / "instruction_pairs.jsonl"
            ),
            "preference_pairs_path": str(
                corpus_dir_path / "training_specs" / "preference_pairs.jsonl"
            ),
            "instruction_pairs_count": stats.instruction_pairs_emitted,
            "preference_pairs_count": stats.preference_pairs_emitted,
            "chunks_eligible": stats.chunks_eligible,
            "chunks_total": stats.chunks_total,
            "stats": stats.as_dict(),
        })

    @mcp.tool()
    async def archive_to_libv2(
        course_name: str,
        domain: str,
        division: str = "STEM",
        pdf_paths: Optional[str] = None,
        html_paths: Optional[str] = None,
        imscc_path: Optional[str] = None,
        assessment_path: Optional[str] = None,
        subdomains: Optional[str] = None,
        concept_graph_sha256: Optional[str] = None,
        dart_chunks_sha256: Optional[str] = None,
        imscc_chunks_sha256: Optional[str] = None,
    ) -> str:
        """
        Archive all pipeline artifacts to LibV2 unified repository.

        Stores raw inputs (PDFs), DART outputs (HTML), course packages (IMSCC),
        and RAG corpus together under a single course slug.

        Args:
            course_name: Course identifier (e.g., "PHYS_101")
            domain: Primary domain (e.g., "physics", "computer-science")
            division: Division classification ("STEM" or "ARTS", default: "STEM")
            pdf_paths: Comma-separated paths to original PDF inputs
            html_paths: Comma-separated paths to DART HTML outputs
            imscc_path: Path to Courseforge IMSCC package
            assessment_path: Path to Trainforge assessment JSON
            subdomains: Comma-separated subdomains (e.g., "mechanics,thermodynamics")
            concept_graph_sha256: Phase 8 ST 1 — optional 64-hex SHA256 of the
                ``concept_graph_semantic.json`` produced by ``concept_extraction``.
                When well-formed, persisted to ``manifest.concept_graph_sha256``;
                malformed values silently dropped (mirrors registry variant's
                ``INVALID_*`` fall-through). Default ``None`` for back-compat
                with legacy MCP clients.
            dart_chunks_sha256: Phase 8 ST 1 — optional 64-hex SHA256 of the
                DART chunkset (``dart_chunks/chunks.jsonl``) produced by the
                ``chunking`` phase. Same emit + drop semantics as above.
            imscc_chunks_sha256: Phase 8 ST 1 — optional 64-hex SHA256 of the
                IMSCC chunkset (``imscc_chunks/chunks.jsonl``) produced by the
                ``imscc_chunking`` phase. Same emit + drop semantics as above.

        Returns:
            JSON with course_slug, storage paths, and archival status
        """
        try:
            libv2_root = PROJECT_ROOT / "LibV2"

            # Generate slug from course name
            slug = course_name.lower().replace("_", "-").replace(" ", "-")

            # Create course directory structure
            course_dir = libv2_root / "courses" / slug
            for subdir in [
                "source/pdf", "source/html", "source/imscc",
                "corpus", "graph", "pedagogy", "training_specs", "quality"
            ]:
                (course_dir / subdir).mkdir(parents=True, exist_ok=True)

            archived = {"pdfs": [], "html": [], "imscc": None, "assessment": None}

            # Archive raw PDFs
            if pdf_paths:
                for pdf_str in pdf_paths.split(","):
                    pdf = Path(pdf_str.strip())
                    if pdf.exists():
                        dest = course_dir / "source" / "pdf" / pdf.name
                        shutil.copy2(pdf, dest)
                        archived["pdfs"].append(str(dest))

            # Archive DART HTML outputs
            if html_paths:
                for html_str in html_paths.split(","):
                    html_file = Path(html_str.strip())
                    if html_file.exists():
                        dest = course_dir / "source" / "html" / html_file.name
                        shutil.copy2(html_file, dest)
                        archived["html"].append(str(dest))
                        # Also copy quality JSON if present
                        quality_json = html_file.with_suffix(".quality.json")
                        if quality_json.exists():
                            shutil.copy2(
                                quality_json,
                                course_dir / "quality" / quality_json.name
                            )
                        # Wave 19: archive ``{stem}_figures/`` sibling dir
                        # when it exists so LibV2 stores the portable
                        # bundle alongside the HTML.
                        figures_dir_src = (
                            html_file.parent / f"{html_file.stem}_figures"
                        )
                        if figures_dir_src.is_dir():
                            figures_dir_dest = (
                                course_dir / "source" / "html"
                                / figures_dir_src.name
                            )
                            if figures_dir_dest.exists():
                                shutil.rmtree(figures_dir_dest)
                            shutil.copytree(figures_dir_src, figures_dir_dest)

            # Archive IMSCC package
            if imscc_path:
                imscc = Path(imscc_path)
                if imscc.exists():
                    dest = course_dir / "source" / "imscc" / imscc.name
                    shutil.copy2(imscc, dest)
                    archived["imscc"] = str(dest)

            # Archive assessment / RAG corpus output
            # Phase 7c: write to imscc_chunks/ (canonical).
            if assessment_path:
                assess = Path(assessment_path)
                if assess.exists():
                    dest = course_dir / "imscc_chunks" / assess.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(assess, dest)
                    archived["assessment"] = str(dest)
                    # Wave 30 Gap 3: when the caller points us at an
                    # assessments.json (or its containing directory), also
                    # pick up the Wave 30 training_synthesis artifacts.
                    # Mirrors the registry variant's copy_map; we keep the
                    # probe cheap so a missing sibling dir stays silent.
                    assess_parent = assess.parent if assess.is_file() else assess
                    for sibling_name in ("training_specs",):
                        sibling_dir = assess_parent / sibling_name
                        if not sibling_dir.is_dir():
                            # trainforge_dir might be the parent of parent.
                            sibling_dir = assess_parent.parent / sibling_name
                        if not sibling_dir.is_dir():
                            continue
                        for fname in (
                            "instruction_pairs.jsonl",
                            "preference_pairs.jsonl",
                            "dataset_config.json",
                        ):
                            src = sibling_dir / fname
                            if src.exists() and src.is_file():
                                dest = (
                                    course_dir / "training_specs" / fname
                                )
                                try:
                                    shutil.copy2(src, dest)
                                    archived.setdefault(
                                        "training_specs", []
                                    ).append(str(dest))
                                except OSError as _exc:
                                    logger.debug(
                                        "archive_to_libv2: failed to copy %s: %s",
                                        src, _exc,
                                    )
                    # Wave 30 Gap 4: course.json is materialised alongside
                    # assessments.json (trainforge_dir / course.json).
                    for _course_root in (
                        assess_parent,
                        assess_parent.parent,
                    ):
                        _cj = _course_root / "course.json"
                        if _cj.exists() and _cj.is_file():
                            try:
                                shutil.copy2(_cj, course_dir / "course.json")
                                archived["course_json"] = str(
                                    course_dir / "course.json"
                                )
                                break
                            except OSError as _exc:
                                logger.debug(
                                    "archive_to_libv2: course.json copy failed: %s",
                                    _exc,
                                )

            # Build manifest
            import hashlib

            def _sha256(filepath: Path) -> str:
                h = hashlib.sha256()
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                return h.hexdigest()

            source_artifacts = {}
            if archived["pdfs"]:
                source_artifacts["pdf"] = [
                    {"path": p, "checksum": _sha256(Path(p)), "size": Path(p).stat().st_size}
                    for p in archived["pdfs"]
                ]
            if archived["html"]:
                source_artifacts["html"] = [
                    {"path": p, "checksum": _sha256(Path(p)), "size": Path(p).stat().st_size}
                    for p in archived["html"]
                ]
            if archived["imscc"]:
                imscc_p = Path(archived["imscc"])
                source_artifacts["imscc"] = {
                    "path": archived["imscc"],
                    "checksum": _sha256(imscc_p),
                    "size": imscc_p.stat().st_size,
                }

            # Wave 10: advisory feature flag — scan the archived corpus's
            # chunks.jsonl (if any) for chunks carrying
            # source.source_references[]. Lets LibV2 retrieval callers
            # fast-skip source-grounded queries on legacy corpora.
            # Defaults false when no chunks file is found, when it can't
            # be read, or when no chunks carry refs.
            source_provenance_flag = _detect_source_provenance(course_dir)

            # Wave 11: companion flag for evidence-arm source_references[].
            # True when the archived concept_graph_semantic.json carries at
            # least one edge with evidence.source_references[]. Lets
            # consumers distinguish chunk-level (Wave 10) from evidence-
            # level (Wave 11) provenance.
            evidence_source_provenance_flag = _detect_evidence_source_provenance(course_dir)

            manifest = {
                "libv2_version": "1.2.0",
                "chunker_version": _resolve_chunker_version(),
                "slug": slug,
                "import_timestamp": datetime.now().isoformat(),
                "classification": {
                    "division": division,
                    "primary_domain": domain,
                    "subdomains": [s.strip() for s in subdomains.split(",")] if subdomains else [],
                },
                "source_artifacts": source_artifacts,
                "provenance": {
                    "source_type": "textbook_to_course_pipeline",
                    "import_pipeline_version": "1.0.0",
                },
                "features": {
                    "source_provenance": source_provenance_flag,
                    "evidence_source_provenance": evidence_source_provenance_flag,
                },
            }

            # Phase 8 ST 1: persist the three SHA256 fields when callers thread
            # them via kwargs. Same ``^[0-9a-f]{64}$`` regex shape as the
            # registry variant at ``:5667-5687, :5720-5727`` — only emit when
            # the kwarg is well-formed so a malformed value falls through to
            # the validator's ``MISSING_*`` critical (the validator owns
            # ``INVALID_*`` shape diagnostics).
            #
            # Intentional asymmetry vs registry variant: this @mcp.tool()
            # surface is kwarg-only — no on-disk recompute fallback for
            # ``concept_graph_sha256``. The recompute path lives only in the
            # registry variant because that surface is workflow-runner-driven
            # and can resolve the on-disk artifact path; external MCP clients
            # call this @mcp.tool() variant directly and pass paths
            # explicitly. See plans/phase8_cleanup.md pre-resolved decision #1.
            import re as _re_sha
            _SHA_RE = r"^[0-9a-f]{64}$"
            if concept_graph_sha256 and _re_sha.match(_SHA_RE, concept_graph_sha256):
                manifest["concept_graph_sha256"] = concept_graph_sha256
            if dart_chunks_sha256 and _re_sha.match(_SHA_RE, dart_chunks_sha256):
                manifest["dart_chunks_sha256"] = dart_chunks_sha256
            if imscc_chunks_sha256 and _re_sha.match(_SHA_RE, imscc_chunks_sha256):
                manifest["imscc_chunks_sha256"] = imscc_chunks_sha256

            # GPT Feedback v2 (May 12 / item 3): lineage mirror. Same shape
            # as the registry variant — build a dedup'd index of every
            # source-document SHA from source_artifacts, then mirror the
            # graph-level lineage fields from concept_graph_semantic.json.
            _src_doc_shas: set = set()
            for entry in source_artifacts.get("pdf", []) or []:
                cs = entry.get("checksum") if isinstance(entry, dict) else None
                if isinstance(cs, str) and _re_sha.match(_SHA_RE, cs):
                    _src_doc_shas.add(cs)
            for entry in source_artifacts.get("html", []) or []:
                cs = entry.get("checksum") if isinstance(entry, dict) else None
                if isinstance(cs, str) and _re_sha.match(_SHA_RE, cs):
                    _src_doc_shas.add(cs)
            imscc_entry = source_artifacts.get("imscc")
            if isinstance(imscc_entry, dict):
                cs = imscc_entry.get("checksum")
                if isinstance(cs, str) and _re_sha.match(_SHA_RE, cs):
                    _src_doc_shas.add(cs)
            if _src_doc_shas:
                manifest["source_documents_sha256_index"] = sorted(_src_doc_shas)

            # Mirror graph-level lineage when graph already on disk.
            _cg_path = course_dir / "graph" / "concept_graph_semantic.json"
            if _cg_path.exists() and _cg_path.is_file():
                try:
                    _cg_json = json.loads(_cg_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    _cg_json = None
                if isinstance(_cg_json, dict):
                    _gbh = _cg_json.get("graph_build_hash")
                    if isinstance(_gbh, str) and _re_sha.match(_SHA_RE, _gbh):
                        manifest["graph_build_hash"] = _gbh
                    _rpv = _cg_json.get("rulepack_version")
                    if isinstance(_rpv, str) and _re_sha.match(r"^v[0-9a-f]+$", _rpv):
                        manifest["rulepack_version"] = _rpv
                    if "course_package_version" in _cg_json:
                        _cpv = _cg_json["course_package_version"]
                        if _cpv is None or (
                            isinstance(_cpv, str) and _cpv
                        ):
                            manifest["course_package_version"] = _cpv

            manifest_path = course_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # F6: register the course in the LibV2 master catalog so that
            # ``libv2 catalog list`` / ``libv2 info <slug>`` see it immediately
            # without a separate manual ``index rebuild`` step.
            try:
                from LibV2.tools.libv2.catalog import _register_course_in_catalog
                _register_course_in_catalog(slug, manifest, libv2_root)
            except Exception as _exc:
                logger.warning(
                    "archive_to_libv2: catalog registration failed for %s: %s",
                    slug, _exc,
                )

            return json.dumps({
                "success": True,
                "course_slug": slug,
                "course_dir": str(course_dir),
                "manifest_path": str(manifest_path),
                "archived": archived,
                "artifact_counts": {
                    "pdfs": len(archived["pdfs"]),
                    "html_files": len(archived["html"]),
                    "imscc": 1 if archived["imscc"] else 0,
                    "assessment": 1 if archived["assessment"] else 0,
                },
            })

        except Exception as e:
            logger.error(f"Failed to archive to LibV2: {e}")
            return json.dumps({"error": str(e)})

    # Wave 28f: run_textbook_pipeline_tool was removed. External MCP
    # clients now route through the workflow API. The underlying
    # non-tool ``run_textbook_pipeline()`` coroutine above remains for
    # internal callers.


def _raw_text_to_accessible_html(
    raw_text: str,
    title: str,
    metadata: Optional[dict] = None,
    *,
    source_pdf: Optional[str] = None,
    output_path: Optional[str] = None,
    figures_dir: Optional[str] = None,
    llm: Optional[object] = None,
    capture: Optional[object] = None,
    canonical_course_code: Optional[str] = None,
) -> str:
    """Wave 15+16+17 entry point: route raw pdftotext / PDF to DART.converter.

    Flags:

    * ``DART_LLM_CLASSIFICATION`` is respected transitively through
      ``DART.converter.default_classifier`` — when on AND a backend is
      provided, block classification goes through Claude.

    Wave 16: when ``source_pdf`` is provided, the converter reaches the
    full :func:`DART.converter.extractor.extract_document` path so
    pdfplumber tables, PyMuPDF figures, and Tesseract OCR text all
    survive into the HTML output. When ``source_pdf`` is ``None`` (the
    legacy raw-text-only call shape), behaviour is unchanged from Wave
    15 — the converter runs on ``raw_text`` alone.

    Wave 17: when ``output_path`` is provided and ``figures_dir`` is
    not overridden, the converter auto-derives a sibling figures
    directory (``<output_stem>_figures``) next to the output HTML, so
    persisted figure images stay relative to the HTML file and the
    bundle is portable. Explicit ``figures_dir=...`` overrides the
    sibling derivation.

    ``metadata`` carries Dublin Core fields (authors, date, language,
    rights, subject) that the new assembler emits as ``<meta>`` tags in
    ``<head>``.

    Wave 28f: the ``DART_LEGACY_CONVERTER`` safety fallback (and the
    ~620-LOC ``_raw_text_to_accessible_html_legacy`` regex path it
    gated) were removed after one release of grace. The Wave-15+
    ontology-aware converter is now the only path.
    """
    import os as _os

    # Wave 22 DC3: pipeline_run_attribution capture. One record per
    # _raw_text_to_accessible_html call so runs are replayable from
    # captures alone. When the caller doesn't supply a capture, we
    # build a short-lived DARTDecisionCapture keyed on the canonical
    # course code (Wave 29 Defect 5) so every capture in one run shares
    # the same course_id. When the caller didn't provide one — legacy
    # pathways that invoke the converter directly from a PDF without a
    # workflow_state — we fall back to the Wave 22 DC4 behaviour of
    # normalising the PDF stem, but log at DEBUG that we're on the
    # legacy path (Wave 29 Defect 5 contract).
    _owns_capture = False
    if capture is None and source_pdf:
        try:
            from lib.decision_capture import (
                DARTDecisionCapture,
                normalize_course_code,
            )

            _pdf_stem = Path(source_pdf).stem or "unknown"
            if canonical_course_code:
                _cc = canonical_course_code
            else:
                _cc = normalize_course_code(_pdf_stem)
                logger.debug(
                    "DC5 legacy fallback: no canonical_course_code supplied; "
                    "deriving from PDF stem %s -> %s",
                    _pdf_stem,
                    _cc,
                )
            capture = DARTDecisionCapture(
                course_code=_cc,
                pdf_name=_pdf_stem,
            )
            _owns_capture = True
        except Exception as _exc:  # noqa: BLE001 — capture is best-effort
            logger.debug("DC3 capture init failed (%s); continuing", _exc)
            capture = None

    if capture is not None:
        try:
            classifier_mode = (
                "llm"
                if _os.environ.get("DART_LLM_CLASSIFICATION", "").strip().lower() == "true"
                and llm is not None
                else "heuristic"
            )
            backend = "heuristic" if classifier_mode == "heuristic" else "claude"
            # W-D13: surface resolved DART provider names so the audit
            # trail records WHICH backend produced each DART artifact —
            # parallel to how the Trainforge synthesis providers
            # interpolate ``provider=...`` in their decision rationales.
            try:
                from DART.pdf_converter.claude_processor import (
                    _resolve_dart_provider as _dart_provider_resolver,
                    _resolve_dart_vision_provider as _dart_vision_provider_resolver,
                )

                _dart_text_provider = _dart_provider_resolver()
                _dart_vision_provider_name = _dart_vision_provider_resolver()
            except Exception:  # noqa: BLE001 — resolver is best-effort
                _dart_text_provider = "unknown"
                _dart_vision_provider_name = "unknown"
            rationale = (
                f"Ran DART pipeline against "
                f"{Path(source_pdf).name if source_pdf else 'raw_text_only'}; "
                f"backend={backend}; classifier_mode={classifier_mode}; "
                f"provider={_dart_text_provider}; "
                f"vision_provider={_dart_vision_provider_name}; "
                f"raw_text len={len(raw_text or '')} chars; "
                f"title={title!r}; "
                f"output_path={'set' if output_path else 'unset'}; "
                f"figures_dir={'set' if figures_dir else 'unset'}; "
                f"llm={'injected' if llm is not None else 'none'}"
            )
            capture.log_decision(
                decision_type="pipeline_run_attribution",
                decision=(
                    f"Ran DART pipeline against "
                    f"{Path(source_pdf).name if source_pdf else 'raw_text_only'}"
                ),
                rationale=rationale,
                context=(
                    f"source_pdf={source_pdf or ''}; "
                    f"output_path={output_path or ''}"
                ),
            )
        except Exception as _exc:  # noqa: BLE001 — capture is best-effort
            logger.debug(
                "DC3 pipeline_run_attribution log failed (%s); continuing",
                _exc,
            )

    # Wave 30 Gap 1: alt-text generation decision-capture + operator warning.
    # Emits exactly one ``alt_text_generation`` decision per pipeline run
    # summarising whether the run used a live LLM backend or fell back to
    # the WCAG-decorative placeholder. Previously AltTextGenerator only
    # fired per-figure captures when the LLM actually ran, so runs with
    # ``llm=None`` produced no alt-text-related trace at all.
    if source_pdf:
        _alt_text_mode = "llm_generation" if llm is not None else "decorative_fallback"
        if llm is None:
            logger.warning(
                "Alt-text generation skipped (no LLM backend); figures on %s "
                "will emit WCAG-decorative fallback (alt='' role='presentation')",
                Path(source_pdf).name,
            )
        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="alt_text_generation",
                    decision=(
                        f"Alt-text pipeline mode={_alt_text_mode} "
                        f"for {Path(source_pdf).name}"
                    ),
                    rationale=(
                        f"Run-level alt-text mode for "
                        f"{Path(source_pdf).name}: mode={_alt_text_mode}; "
                        f"llm={'injected' if llm is not None else 'none'}; "
                        f"per-figure decisions follow when mode=llm_generation; "
                        f"WCAG 1.1.1: empty alt + role=presentation emitted "
                        f"on every <figure> when mode=decorative_fallback"
                    ),
                    context=f"source_pdf={source_pdf}",
                )
            except Exception as _exc:  # noqa: BLE001 — capture is best-effort
                logger.debug(
                    "Wave 30 alt_text_generation summary log failed (%s); continuing",
                    _exc,
                )

    try:
        return _run_dart_pipeline_body(
            raw_text=raw_text,
            title=title,
            metadata=metadata,
            source_pdf=source_pdf,
            output_path=output_path,
            figures_dir=figures_dir,
            llm=llm,
            capture=capture,
        )
    finally:
        # Finalise an owned capture so the JSONL flushes before the
        # caller's process ends. Externally-supplied captures are the
        # caller's responsibility to close.
        if _owns_capture and capture is not None:
            try:
                if hasattr(capture, "save"):
                    capture.save()
                elif hasattr(capture, "close"):
                    capture.close()
            except Exception as _exc:  # noqa: BLE001
                logger.debug(
                    "DC3 capture finalise failed (%s); continuing", _exc
                )


def _run_dart_pipeline_body(
    *,
    raw_text: str,
    title: str,
    metadata: Optional[dict],
    source_pdf: Optional[str],
    output_path: Optional[str],
    figures_dir: Optional[str],
    llm: Optional[object],
    capture: Optional[object],
) -> str:
    """Actual conversion body for ``_raw_text_to_accessible_html``.

    Wave 22 DC3 split the outer entry point from this body so the
    pipeline_run_attribution capture can wrap the whole call with a
    single try/finally. Behaviour is byte-for-byte identical to the
    pre-Wave-22 monolithic function body — this is a pure extraction.
    """

    # Wave 16 enriched path: when a source PDF is available, go through
    # the dual-extraction layer so tables / figures / OCR contribute
    # structured blocks. Wrap extractor failures in a fall-through so a
    # broken optional extractor never blocks the raw-text conversion.
    if source_pdf:
        try:
            from DART.converter import default_classifier
            from DART.converter.block_segmenter import (
                segment_extracted_document,
            )
            from DART.converter.document_assembler import assemble_html
            from DART.converter.extractor import extract_document

            # Wave 17: derive a sibling figures dir from ``output_path``
            # so persisted figure bytes travel with the HTML. Explicit
            # ``figures_dir`` wins. Unset + unset → tempdir fallback
            # (plumbed through anyway so ``data.image_path`` still
            # points somewhere; the pipeline won't see the files but
            # tests / ad-hoc runs keep the full round-trip).
            resolved_figures_dir: Optional[Path] = None
            rel_figures_prefix = ""
            if figures_dir:
                resolved_figures_dir = Path(figures_dir)
                # A caller-supplied figures_dir is treated as relative
                # to output_path when output_path exists, else as an
                # absolute/cwd-relative path.
                if output_path:
                    out_parent = Path(output_path).resolve().parent
                    try:
                        rel = resolved_figures_dir.resolve().relative_to(
                            out_parent
                        )
                        rel_figures_prefix = str(rel) + "/"
                    except ValueError:
                        rel_figures_prefix = str(resolved_figures_dir) + "/"
                else:
                    rel_figures_prefix = str(resolved_figures_dir) + "/"
            elif output_path:
                out_path = Path(output_path)
                sibling_name = f"{out_path.stem}_figures"
                resolved_figures_dir = out_path.parent / sibling_name
                rel_figures_prefix = sibling_name + "/"
            else:
                # Neither output_path nor figures_dir provided. Fall
                # back to a tempdir so figures still materialise on
                # disk for downstream consumers that know how to find
                # them; ``<img src>`` references become absolute paths
                # which isn't portable but is better than empty ``src``.
                import tempfile as _tempfile

                resolved_figures_dir = Path(
                    _tempfile.mkdtemp(prefix="dart_figures_")
                )
                rel_figures_prefix = str(resolved_figures_dir) + "/"
                logger.debug(
                    "No output_path or figures_dir; using tempdir %s for figures",
                    resolved_figures_dir,
                )

            doc = extract_document(
                source_pdf,
                llm=llm,
                figures_dir=resolved_figures_dir,
                capture=capture,
            )

            # Rewrite each figure's ``image_path`` to include the
            # sibling-dir prefix so downstream blocks carry a relative
            # path that resolves from the HTML output location.
            if rel_figures_prefix:
                for fig in doc.figures:
                    if fig.image_path and "/" not in fig.image_path:
                        fig.image_path = rel_figures_prefix + fig.image_path

            # Wave 18: merge PyMuPDF-surfaced PDF metadata into the
            # caller's metadata dict. Only fill in blanks — never
            # override explicit caller-supplied values. ``creationDate``
            # is already normalised to ISO 8601 by the extractor.
            merged_metadata = dict(metadata or {})
            pdf_meta = getattr(doc, "pdf_metadata", None) or {}
            if pdf_meta:
                _META_FALLBACKS = {
                    "title": "title",
                    "author": "authors",
                    "subject": "subject",
                    "creationDate": "date",
                }
                for src_key, dest_key in _META_FALLBACKS.items():
                    if src_key not in pdf_meta:
                        continue
                    value = pdf_meta[src_key]
                    if not value:
                        continue
                    # Fill in blanks only — never stomp caller-provided
                    # values. We check against the merged dict after
                    # default copy so absent keys trigger the fill.
                    if not merged_metadata.get(dest_key):
                        merged_metadata[dest_key] = value

            blocks = segment_extracted_document(doc)
            # Wave 18: thread text_spans + median through the classifier
            # so font-size-based heading promotion fires when PyMuPDF
            # layout data is available.
            from DART.converter.extractor import (
                median_body_font_size as _median_font,
            )

            spans = list(getattr(doc, "text_spans", None) or [])
            median_fs = _median_font(spans) if spans else None
            classifier = default_classifier(
                llm=llm,
                text_spans=spans,
                median_body_font_size=median_fs,
                capture=capture,
                page_chrome=getattr(doc, "page_chrome", None),
            )
            from DART.converter.heuristic_classifier import HeuristicClassifier

            if isinstance(classifier, HeuristicClassifier):
                classified = classifier.classify_sync(blocks)
            else:
                # Use the same loop-safe bridge as convert_pdftotext_to_html.
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import threading

                    result: list = []
                    error: list = []

                    def _runner():
                        try:
                            result.append(asyncio.run(classifier.classify(blocks)))
                        except BaseException as exc:  # noqa: BLE001
                            error.append(exc)

                    thread = threading.Thread(target=_runner, daemon=True)
                    thread.start()
                    thread.join()
                    if error:
                        raise error[0]
                    classified = result[0]
                except RuntimeError:
                    classified = asyncio.run(classifier.classify(blocks))
            html_out = assemble_html(classified, title, merged_metadata)
            _emit_dart_sidecars_if_requested(
                classified_blocks=classified,
                html=html_out,
                title=title,
                output_path=output_path,
                source_pdf=source_pdf,
                metadata=merged_metadata,
                page_chrome=getattr(doc, "page_chrome", None),
            )
            return html_out
        except RuntimeError as exc:
            logger.debug(
                "Wave 16 extractor failed (%s); falling back to raw-text path",
                exc,
            )
        except Exception as exc:  # noqa: BLE001 — never block on optional path
            logger.debug(
                "Wave 16 extractor raised unexpectedly (%s); falling back",
                exc,
            )

    # Wave 15 path (raw text only): delegate to the 4-phase pipeline.
    # Wave 19: inline the raw-text path so we can emit the sidecars
    # alongside the HTML when ``output_path`` is set.
    from DART.converter import (
        HeuristicClassifier,
        default_classifier,
        segment_pdftotext_output,
    )
    from DART.converter.document_assembler import assemble_html

    raw_blocks = segment_pdftotext_output(raw_text)
    raw_classifier = default_classifier(llm=llm, capture=capture)
    if isinstance(raw_classifier, HeuristicClassifier):
        raw_classified = raw_classifier.classify_sync(raw_blocks)
    else:
        import asyncio as _asyncio

        try:
            _asyncio.get_running_loop()
            import threading as _threading

            raw_result: list = []
            raw_error: list = []

            def _raw_runner():
                try:
                    raw_result.append(
                        _asyncio.run(raw_classifier.classify(raw_blocks))
                    )
                except BaseException as exc:  # noqa: BLE001
                    raw_error.append(exc)

            raw_thread = _threading.Thread(target=_raw_runner, daemon=True)
            raw_thread.start()
            raw_thread.join()
            if raw_error:
                raise raw_error[0]
            raw_classified = raw_result[0]
        except RuntimeError:
            raw_classified = _asyncio.run(raw_classifier.classify(raw_blocks))

    html_out = assemble_html(raw_classified, title, metadata or {})
    _emit_dart_sidecars_if_requested(
        classified_blocks=raw_classified,
        html=html_out,
        title=title,
        output_path=output_path,
        source_pdf=source_pdf,
        metadata=metadata,
    )
    return html_out


def _emit_dart_sidecars_if_requested(
    *,
    classified_blocks,
    html: str,
    title: str,
    output_path: Optional[str],
    source_pdf: Optional[str],
    metadata: Optional[dict],
    page_chrome: Any = None,
) -> None:
    """Wave 19: write ``*_synthesized.json`` + ``*.quality.json`` sidecars.

    Preconditions: only emits when ``output_path`` is set (mirrors the
    figure-persistence pattern — tempdir callers skip). Failures are
    logged + swallowed so a sidecar write error never blocks the HTML
    return path.

    Wave 20: ``page_chrome`` (optional) is surfaced into the synthesized
    sidecar's ``document_provenance.page_chrome_detected`` block when
    provided. Pre-Wave-20 callers that omit it get the original shape.
    """
    if not output_path:
        return
    try:
        from DART.converter.sidecars import (
            build_quality_sidecar,
            build_synthesized_sidecar,
        )

        out_path = Path(output_path)
        base = out_path.with_suffix("")

        synth = build_synthesized_sidecar(
            classified_blocks,
            title=title,
            source_pdf=source_pdf,
            metadata=metadata or {},
            page_chrome=page_chrome,
        )
        synth_path = base.parent / f"{base.name}_synthesized.json"
        synth_path.write_text(
            json.dumps(synth, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        quality = build_quality_sidecar(
            html, title=title, source_pdf=source_pdf
        )
        quality_path = out_path.with_suffix(".quality.json")
        quality_path.write_text(
            json.dumps(quality, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug(
            "Wave 19 sidecars emitted: %s, %s",
            synth_path,
            quality_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Wave 19 sidecar emission failed (non-fatal): %s", exc
        )




# Phase 3 Subtask 6: Courseforge two-pass-router enable flag. Mirror
# of ``Courseforge/scripts/blocks.py::_emit_blocks_enabled`` (the
# Phase 2 emit-blocks toggle) — same truthy set, same case-insensitive
# read-each-call pattern so tests can toggle the flag inline. Used by
# the rewrite of ``_generate_course_content`` (Subtask 39-41 in the
# Phase 3 plan) to dispatch to the two-pass OutlineProvider +
# RewriteProvider path instead of the legacy single-pass renderer.
_COURSEFORGE_TWO_PASS_ENV = "COURSEFORGE_TWO_PASS"
_COURSEFORGE_TWO_PASS_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _courseforge_two_pass_enabled() -> bool:
    """Read ``COURSEFORGE_TWO_PASS`` each call so tests can toggle it.

    Default off — the Phase 3 two-pass router is purely additive and
    must not alter byte-stable legacy emits until the migration window
    closes.
    """
    return (
        os.environ.get(_COURSEFORGE_TWO_PASS_ENV, "").strip().lower()
        in _COURSEFORGE_TWO_PASS_TRUTHY
    )


# ---------------------------------------------------------------------------
# Phase 3.5 Subtask 13: post_rewrite_validation phase helper
# ---------------------------------------------------------------------------


async def _run_post_rewrite_validation(**kwargs) -> str:
    """Run the four shape-discriminating Block-input validators against
    rewrite-tier blocks loaded from ``blocks_final_path``.

    Phase 3.5 Subtask 13. Mirrors the shape-discriminator wiring that
    ``inter_tier_validation`` does on outline-tier blocks, but consumes
    the rewrite-tier ``blocks_final_path`` (which the rewrite tier emits
    after producing the final HTML body) so a rewrite-tier emit that
    drops a CURIE / content_type / objective_ref / source_id silently
    is caught before packaging consumes it.

    Inputs (kwargs):
        blocks_final_path: Path to the rewrite-tier blocks JSONL/JSON
            sidecar. Each entry is a dict carrying at minimum
            ``{block_id, block_type, page_id, sequence, content,
            objective_ids, source_ids, ...}``. Entries are deserialised
            via :class:`Courseforge.scripts.blocks.Block` field
            assignment; unknown / extra fields are tolerated by dropping
            them (mirrors the to_jsonld_entry() emit shape).
        project_id: Course project slug (used to locate the
            decision-capture sidecar + the synthesized_objectives.json
            that BlockPageObjectivesValidator consumes).

    Outputs (JSON envelope):
        blocks_validated_path: Path to a JSONL file carrying the
            rewrite-tier blocks that passed every gate (subset of input).
        blocks_failed_path: Path to a JSONL file carrying the
            rewrite-tier blocks that tripped at least one gate.
        gate_results: List of per-gate ``GateResult.to_dict()`` payloads
            so a downstream consumer (or operator) can introspect the
            failure distribution without re-running the validators.

    Decision-capture: emits one ``block_validation_action`` decision
    per failed validator with ``ml_features.tier="rewrite"`` so a
    postmortem reader can stratify outline-tier failures (which trigger
    regen / escalate / block in route_with_self_consistency) from
    rewrite-tier failures (which surface here, post-emit).

    Wave B scope: this helper is the standalone callable. The
    ``_PHASE_TOOL_MAPPING`` wiring that lets the executor dispatch
    on phase-name lands in Wave C; until that wave the helper is
    invoked directly by integration tests / operators.
    """
    from pathlib import Path as _Path

    blocks_final_path_raw = kwargs.get("blocks_final_path") or ""
    project_id = kwargs.get("project_id") or ""

    if not blocks_final_path_raw:
        return json.dumps({
            "success": False,
            "error": "blocks_final_path is required",
        })

    blocks_path = _Path(blocks_final_path_raw)
    if not blocks_path.exists():
        return json.dumps({
            "success": False,
            "error": f"blocks_final_path does not exist: {blocks_path}",
        })

    # Lazy-import the Block dataclass + validators so this helper can
    # land without forcing every callsite to import the router surface.
    from Courseforge.scripts.blocks import Block, Touch  # type: ignore[import-not-found]
    from Courseforge.router.inter_tier_gates import (
        BlockContentTypeValidator,
        BlockCurieAnchoringValidator,
        BlockPageObjectivesValidator,
        BlockSourceRefValidator,
    )

    # Deserialise blocks. Accept JSONL (one block per line) and JSON
    # (top-level list, or top-level object with a ``blocks`` key).
    raw_text = blocks_path.read_text(encoding="utf-8")
    raw_entries: list = []
    try:
        if blocks_path.suffix == ".jsonl":
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw_entries.append(json.loads(line))
        else:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                raw_entries = parsed
            elif isinstance(parsed, dict):
                raw_entries = parsed.get("blocks", []) or []
    except json.JSONDecodeError as exc:
        return json.dumps({
            "success": False,
            "error": f"failed to parse {blocks_path}: {exc}",
        })

    # M3 fix: instrument the silent-drop sites the docstring above flags.
    # When a JSON entry's CURIE / content_type_label / objective_ref /
    # source_id values are malformed, we previously dropped them silently
    # via list / tuple coercion. The post_rewrite_validation phase now
    # records each drop in ``metadata_drops`` and emits a
    # ``metadata_field_drop`` decision event so an operator sees the
    # silent-degradation class without re-running the rewrite tier.
    metadata_drops: List[Dict[str, Any]] = []
    _capture_meta = None
    try:
        from lib.decision_capture import DecisionCapture as _DC_meta
        _capture_meta = _DC_meta(
            course_code=project_id or "post_rewrite_validation",
            phase="post_rewrite_validation",
            tool="courseforge",
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning(
            "post_rewrite_validation: DecisionCapture init for "
            "metadata_field_drop failed (%s); drops will be tracked "
            "but not persisted.",
            exc,
        )
        _capture_meta = None

    _CURIE_RE = __import__("re").compile(r"^[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9_./#:-]+$")
    _LO_REF_RE = __import__("re").compile(r"^[A-Za-z]{2,}-\d{2,}$")
    _CONTENT_TYPE_ENUM = {
        "definition", "example", "procedure", "principle",
        "fact", "concept", "rule", "narrative",
    }

    def _record_metadata_drop(
        field_name: str, reason: str, **fields: Any,
    ) -> None:
        """Append a structured drop record + emit a decision event."""
        rec = {"field_name": field_name, "reason": reason, **fields}
        metadata_drops.append(rec)
        if _capture_meta is None:
            return
        try:
            _capture_meta.log_decision(
                decision_type="metadata_field_drop",
                decision=f"dropped {field_name} reason={reason}",
                rationale=(
                    f"post_rewrite_validation _entry_to_block dropped a "
                    f"malformed {field_name} on a rewrite-tier block. "
                    f"reason={reason}; context={fields!r}. Surfaced via "
                    f"metadata_drops_count in phase output so operators "
                    f"see the silent-degradation class instead of the "
                    f"downstream gate firing on the stripped block."
                ),
                ml_features={
                    "field_name": field_name,
                    "reason": reason,
                    **{
                        k: v for k, v in fields.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "post_rewrite_validation: log_decision("
                "metadata_field_drop) failed (%s); drop still recorded.",
                exc,
            )

    def _entry_to_block(entry: dict) -> Optional["Block"]:  # type: ignore[name-defined]
        """Project a JSON entry onto a frozen Block, dropping unknown keys.

        The blocks_final_path emit shape is the canonical
        :func:`Block.to_jsonld_entry` output PLUS the structural
        fields the rewrite tier persists for re-execution. We only
        consume the fields :class:`Block` accepts; everything else
        is dropped.
        """
        accepted = {
            "block_id", "block_type", "page_id", "sequence", "content",
            "template_type", "key_terms", "objective_ids",
            "bloom_level", "bloom_verb", "bloom_range",
            "bloom_levels", "bloom_verbs", "cognitive_domain",
            "teaching_role", "content_type_label", "purpose",
            "component", "source_ids", "source_primary",
            "source_references", "content_hash",
            "validation_attempts", "escalation_marker",
        }
        block_id_for_log = (entry or {}).get("block_id") or "<unknown>"
        kwargs_clean: dict = {}
        for k, v in (entry or {}).items():
            if k not in accepted:
                continue
            # M3: validate + filter the four silent-drop classes the
            # docstring at line ~2091 flags.
            if k == "objective_ids":
                if isinstance(v, list):
                    cleaned: List[str] = []
                    for ref in v:
                        if isinstance(ref, str) and _LO_REF_RE.match(ref):
                            cleaned.append(ref)
                        else:
                            _record_metadata_drop(
                                "objective_ref",
                                "malformed_lo_ref",
                                block_id=str(block_id_for_log),
                                value=str(ref)[:64],
                            )
                    v = tuple(cleaned)
            elif k == "source_ids":
                if isinstance(v, list):
                    cleaned_src: List[str] = []
                    for sid in v:
                        if isinstance(sid, str) and sid.strip():
                            cleaned_src.append(sid)
                        else:
                            _record_metadata_drop(
                                "source_id",
                                "empty_or_non_string",
                                block_id=str(block_id_for_log),
                                value=str(sid)[:64],
                            )
                    v = tuple(cleaned_src)
            elif k == "content_type_label":
                if v is not None and (
                    not isinstance(v, str) or v not in _CONTENT_TYPE_ENUM
                ):
                    _record_metadata_drop(
                        "content_type",
                        "out_of_enum",
                        block_id=str(block_id_for_log),
                        value=str(v)[:64],
                    )
                    continue  # drop the field; Block ctor accepts None
            elif k == "key_terms":
                if isinstance(v, list):
                    cleaned_kt: List[str] = []
                    for term in v:
                        if isinstance(term, str) and ":" in term and not _CURIE_RE.match(term):
                            # Looks-like-CURIE that fails CURIE shape.
                            _record_metadata_drop(
                                "curie",
                                "malformed_curie",
                                block_id=str(block_id_for_log),
                                value=str(term)[:64],
                            )
                            continue
                        cleaned_kt.append(term)
                    v = tuple(cleaned_kt)
            # Tuple-typed fields take tuple input; lists from JSON are
            # acceptable to dataclasses.replace via Block.__init__ but
            # the frozen dataclass coerces nothing — pass tuples.
            if k in {
                "bloom_levels", "bloom_verbs", "source_references",
            }:
                if isinstance(v, list):
                    v = tuple(v) if k != "source_references" else tuple(
                        dict(r) if isinstance(r, dict) else r for r in v
                    )
            kwargs_clean[k] = v
        # Required fields with sensible defaults when absent.
        if "block_id" not in kwargs_clean or "block_type" not in kwargs_clean:
            return None
        kwargs_clean.setdefault("page_id", kwargs_clean.get("block_id", ""))
        kwargs_clean.setdefault("sequence", 0)
        kwargs_clean.setdefault("content", "")
        try:
            return Block(**kwargs_clean)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "post_rewrite_validation: skipping malformed block entry "
                "block_id=%r: %s",
                entry.get("block_id"),
                exc,
            )
            return None

    blocks: list = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        blk = _entry_to_block(entry)
        if blk is not None:
            blocks.append(blk)

    if not blocks:
        return json.dumps({
            "success": False,
            "error": (
                f"no blocks parseable from {blocks_path}; expected JSON-LD "
                f"shape with at least one Block-shaped entry"
            ),
        })

    # Resolve the canonical objectives JSON for BlockPageObjectivesValidator.
    objectives_path: Optional[_Path] = None
    if project_id:
        candidate = (
            courseforge_exports_dir()
            / project_id
            / "01_learning_objectives"
            / "synthesized_objectives.json"
        )
        if candidate.exists():
            objectives_path = candidate

    # Instantiate the validators + run each one. We pass tier="rewrite"
    # to the decision capture (Subtask 26's ml_features.tier field) so
    # postmortem readers can stratify outline-tier vs rewrite-tier
    # failures.
    validators = [
        ("rewrite_curie_anchoring", BlockCurieAnchoringValidator()),
        ("rewrite_content_type", BlockContentTypeValidator()),
        ("rewrite_page_objectives", BlockPageObjectivesValidator()),
        ("rewrite_source_refs", BlockSourceRefValidator()),
    ]

    inputs: dict = {"blocks": blocks}
    if objectives_path is not None:
        inputs["objectives_path"] = str(objectives_path)

    # v0.3.0 dynamic CURIE minting: thread the minted-CURIE map into the
    # rewrite-tier gate inputs too so surface-form anchoring is available
    # at the post-rewrite seam. No-op for RDF / legacy corpora.
    minted_curie_map = _resolve_minted_curie_map_for_validation(
        project_id=project_id, kwargs=kwargs,
    )
    if minted_curie_map:
        inputs["minted_curie_map"] = minted_curie_map

    gate_results: list = []
    failing_gate_ids: set = set()
    for gate_id, validator in validators:
        try:
            result = validator.validate({**inputs, "gate_id": gate_id})
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "post_rewrite_validation: validator %s raised: %s",
                gate_id, exc,
            )
            continue
        try:
            result_dict = result.to_dict()
        except Exception:  # pragma: no cover — defensive
            result_dict = {
                "gate_id": gate_id,
                "passed": getattr(result, "passed", False),
                "issues": [],
            }
        gate_results.append(result_dict)
        if not result.passed:
            failing_gate_ids.add(gate_id)

    # Decision-capture: emit per-failure ``block_validation_action`` events
    # so postmortem readers see the rewrite-tier failure chain. We
    # reuse the project_id-derived course code for the capture sidecar
    # and surface ``ml_features.tier="rewrite"`` per Subtask 26.
    if failing_gate_ids:
        try:
            from lib.decision_capture import DecisionCapture
            capture = DecisionCapture(
                course_code=project_id or "post_rewrite_validation",
                phase="post_rewrite_validation",
                tool="courseforge",
                streaming=True,
            )
            for result_dict in gate_results:
                if result_dict.get("passed"):
                    continue
                gate_id = result_dict.get("gate_id", "unknown_gate")
                issues = result_dict.get("issues", []) or []
                top_issues = issues[:3]
                summary = "; ".join(
                    f"{i.get('code','?')}({i.get('severity','?')}):"
                    f"{i.get('message','')}"
                    for i in top_issues
                ) or "no_issues"
                capture.log_decision(
                    decision_type="block_validation_action",
                    decision=(
                        f"post_rewrite_validation:{gate_id}:"
                        f"{result_dict.get('passed')}"
                    ),
                    rationale=(
                        f"Post-rewrite gate {gate_id} returned "
                        f"passed={result_dict.get('passed')} on "
                        f"{len(blocks)} rewrite-tier blocks; "
                        f"top_issues=[{summary}]"
                    ),
                    ml_features={
                        "gate_id": gate_id,
                        "passed": result_dict.get("passed"),
                        "issues_count": len(issues),
                        "block_count": len(blocks),
                        # Subtask 26: tier provenance — rewrite-tier
                        # post-emit validator failure (vs outline-tier
                        # in-loop failures).
                        "tier": "rewrite",
                    },
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "post_rewrite_validation: decision-capture emit failed: %s",
                exc,
            )

    # Persist blocks_validated_path + blocks_failed_path siblings to
    # blocks_final_path. The validated path carries the input list
    # untouched (validators are read-only); the failed path carries
    # block IDs + the gate they tripped so downstream consumers (and
    # the audit trail) can correlate without re-running the validators.
    out_dir = blocks_path.parent
    validated_path = out_dir / "blocks_validated.jsonl"
    failed_path = out_dir / "blocks_failed.jsonl"

    failed_block_ids: set = set()
    for result_dict in gate_results:
        if result_dict.get("passed"):
            continue
        for issue in result_dict.get("issues", []) or []:
            loc = issue.get("location") if isinstance(issue, dict) else None
            if isinstance(loc, str) and loc:
                failed_block_ids.add(loc)

    # Escalated blocks ride through to validated_path regardless of
    # validator outcome — they are marker-bearing by design (Wave-7
    # escalate_immediately) and the rewrite phase needs to see them to
    # author from scratch. Without this, blocks_validated.jsonl is empty
    # when the corpus is all-objective and the workflow halts.
    with validated_path.open("w", encoding="utf-8") as fh:
        for blk in blocks:
            if blk.block_id in failed_block_ids and blk.escalation_marker is None:
                continue
            fh.write(json.dumps(
                _block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")
    with failed_path.open("w", encoding="utf-8") as fh:
        for blk in blocks:
            if blk.block_id not in failed_block_ids:
                continue
            if blk.escalation_marker is not None:
                continue
            fh.write(json.dumps(
                _block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")

    return json.dumps({
        "success": True,
        "blocks_validated_path": str(validated_path),
        "blocks_failed_path": str(failed_path),
        "gate_results": gate_results,
        "block_count": len(blocks),
        "failed_block_count": len(failed_block_ids),
        # M3 fix: surface silent-drop counts from _entry_to_block. The
        # first 10 records are inlined for postmortem readability; the
        # count is authoritative.
        "metadata_drops_count": len(metadata_drops),
        "metadata_drops": metadata_drops[:10],
    })


# ---------------------------------------------------------------------------
# Phase 3.5 Subtask 28-30: snake_case Block JSONL shape helper
# ---------------------------------------------------------------------------
#
# ``Block.to_jsonld_entry()`` emits the legacy camelCase section /
# objective / misconception shape (``heading`` / ``contentType`` for
# explanation blocks, ``blockId`` / ``blockType`` for minimal-shape
# blocks, etc.) — that's the wire shape Trainforge / Courseforge
# JSON-LD consumers expect, but it's NOT round-trippable through
# ``Block(**fields)`` because the field names are camelCase and the
# block-type-specific dispatch drops dataclass fields the constructor
# would otherwise accept.
#
# The two-pass router phase helpers need a JSONL shape that (a)
# round-trips cleanly through ``Block(**snake_case_kwargs)`` so a
# downstream phase can rehydrate the Block, and (b) carries every
# field the Block dataclass exposes (block_id, block_type, page_id,
# sequence, content, objective_ids, source_ids, ...). The helper
# below projects a Block to that snake_case shape, mirroring the
# accepted-keys set ``_entry_to_block`` uses to reconstruct.


def _block_to_snake_case_entry(block: Any) -> Dict[str, Any]:
    """Project a :class:`Block` to a snake_case JSONL entry that
    round-trips through ``Block(**entry)``.

    Mirrors the accepted-keys set the helper-internal
    ``_entry_to_block`` consumers (in ``_run_post_rewrite_validation``,
    ``_run_inter_tier_validation``, ``_run_content_generation_rewrite``)
    use to rehydrate. Tuples are projected to lists for JSON
    serialisation; ``source_references`` entries are coerced to plain
    dicts.
    """
    entry: Dict[str, Any] = {
        "block_id": block.block_id,
        "block_type": block.block_type,
        "page_id": block.page_id,
        "sequence": block.sequence,
        "content": block.content,
    }
    optional_scalars = (
        "template_type", "bloom_level", "bloom_verb", "bloom_range",
        "cognitive_domain", "teaching_role", "content_type_label",
        "purpose", "component", "source_primary", "content_hash",
        "escalation_marker",
    )
    for name in optional_scalars:
        value = getattr(block, name, None)
        if value is not None:
            entry[name] = value
    optional_tuples = (
        "key_terms", "objective_ids", "bloom_levels", "bloom_verbs",
        "source_ids",
    )
    for name in optional_tuples:
        value = getattr(block, name, ()) or ()
        if value:
            entry[name] = list(value)
    refs = getattr(block, "source_references", ()) or ()
    if refs:
        entry["source_references"] = [
            dict(r) if isinstance(r, dict) else r for r in refs
        ]
    if getattr(block, "validation_attempts", 0):
        entry["validation_attempts"] = int(block.validation_attempts)
    return entry


# ---------------------------------------------------------------------------
# Phase 3.5 Subtasks 28-30: two-pass content_generation phase helpers
# ---------------------------------------------------------------------------
#
# These three helpers implement the outline -> validate -> rewrite seam the
# Phase 3 two-pass router introduces. Each is a thin async wrapper that:
#
#   * loads its upstream phase outputs (course_planning_path / staging_path
#     for outline; blocks_outline_path for validation; blocks_validated_path
#     for rewrite),
#   * dispatches the tier-appropriate work via :class:`CourseforgeRouter` or
#     the :mod:`Courseforge.router.inter_tier_gates` validator chain, and
#   * persists the canonical phase outputs declared in
#     ``MCP/core/workflow_runner.py::_LEGACY_PHASE_OUTPUT_KEYS`` (Phase 3
#     Subtask 5).
#
# Wave-N constraints (per Phase 3.5 Wave C scope):
# - Pre-filter the Block list before invoking ``CourseforgeRouter.route_all``
#   instead of widening that method's signature with a ``tier_filter`` kwarg.
#   Worker N2 is concurrently widening ``route_all`` itself for self-
#   consistency dispatch; pre-filtering keeps these helpers disjoint from
#   that change.
# - Decision-capture mirrors the ``_run_post_rewrite_validation`` shape:
#   each helper emits at least one ``phase`` provenance event and one
#   per-failure ``block_validation_action`` event with
#   ``ml_features.tier="outline"`` for inter_tier_validation failures.


def _resolve_inter_tier_validators(
    workflow_type: str,
    capture: Any = None,
) -> List[Any]:
    """Resolve the YAML-declared ``inter_tier_validation`` gate chain
    for ``workflow_type`` into a list of validator instances suitable
    for ``CourseforgeRouter.route_with_self_consistency(validators=...)``.

    Worker W2 (validation-wiring fix): Phase 3's self-consistency loop
    runs the inter-tier validators INSIDE the regen budget so failed
    outline candidates re-roll with remediation suffixes attached. Pre-
    fix, the outline phase called ``router.route()`` once per block and
    let the standalone ``inter_tier_validation`` phase catch failures
    after the fact — by which point the regen budget was already gone.

    Resolution chain:
      1. Load the workflow spec via ``OrchestratorConfig.load()``.
      2. Walk ``wf.phases`` for the ``inter_tier_validation`` phase.
      3. For each gate dict in ``phase.validation_gates``, import the
         dotted ``validator`` path and instantiate.
      4. Skip gates that fail to import; emit a structured warning per
         skip via ``capture`` (decision_type=
         ``inter_tier_validator_import_failed``).

    Returns ``[]`` when ``workflow_type`` is empty / unknown / has no
    ``inter_tier_validation`` phase / has no validation_gates declared
    — matching the pre-W2 ``route()`` semantics so legacy direct
    callers (tests, MCP-tool entry points without a workflow_runner-
    managed run) keep working unchanged.
    """
    import importlib

    if not workflow_type:
        return []

    try:
        from MCP.core.config import OrchestratorConfig  # noqa: PLC0415
        cfg = OrchestratorConfig.load()
        wf = cfg.get_workflow(workflow_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Worker W2: OrchestratorConfig.load() failed for "
            "workflow_type=%r: %s; falling back to empty validator list",
            workflow_type, exc,
        )
        return []

    if wf is None:
        logger.warning(
            "Worker W2: workflow_type=%r not registered; falling back "
            "to empty validator list",
            workflow_type,
        )
        return []

    target_phase = None
    for phase in getattr(wf, "phases", []) or []:
        if getattr(phase, "name", "") == "inter_tier_validation":
            target_phase = phase
            break
    if target_phase is None:
        return []

    gates = getattr(target_phase, "validation_gates", None) or []
    validators: List[Any] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        dotted = gate.get("validator") or ""
        gate_id = gate.get("gate_id") or dotted
        if not dotted:
            continue
        try:
            module_path, _, class_name = dotted.rpartition(".")
            if not module_path or not class_name:
                raise ImportError(f"malformed dotted path: {dotted!r}")
            module = importlib.import_module(module_path)
            validator_cls = getattr(module, class_name)
            validators.append(validator_cls())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Worker W2: failed to import inter-tier validator "
                "gate_id=%s validator=%s: %s",
                gate_id, dotted, exc,
            )
            if capture is not None:
                try:
                    capture.log_decision(
                        decision_type="inter_tier_validator_import_failed",
                        decision=(
                            f"Skipped inter-tier validator {gate_id!r} "
                            f"(dotted path {dotted!r}) — import error."
                        ),
                        rationale=(
                            f"OrchestratorConfig declared the gate at "
                            f"workflow_type={workflow_type!r} phase="
                            f"inter_tier_validation but the validator "
                            f"class did not import: {exc}. Falling "
                            f"through means this gate will NOT fire "
                            f"during the outline-tier self-consistency "
                            f"loop; check the validator dotted path."
                        ),
                        ml_features={
                            "workflow_type": workflow_type,
                            "gate_id": gate_id,
                            "dotted_path": dotted,
                            "error": str(exc),
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
            continue
    return validators


def _resolve_post_rewrite_validators(
    workflow_type: str,
    capture: Any = None,
) -> List[Any]:
    """Resolve the YAML-declared ``post_rewrite_validation`` gate chain
    for ``workflow_type`` into a list of validator instances suitable
    for ``CourseforgeRouter.route_rewrite_with_remediation(validators=...)``.

    Worker W3 (validation-wiring fix): symmetric mirror of
    :func:`_resolve_inter_tier_validators` — pulls the rewrite-tier
    validator chain from the YAML ``post_rewrite_validation`` phase so
    the rewrite-tier remediation loop runs the same gates that the
    standalone post-rewrite phase would catch (CURIE / content_type /
    page_objectives / source_refs / rewrite_html_shape /
    rewrite_source_grounding) — only INSIDE the regen budget so a
    failed rewrite re-rolls with a remediation suffix instead of being
    persisted as-is and caught downstream.

    Returns ``[]`` on missing workflow_type, missing phase, or absent
    validation_gates — matching pre-W3 ``route(..., tier="rewrite",
    source_chunks=[], objectives=[])`` semantics so legacy direct
    callers (tests, MCP-tool entry points without a workflow_runner-
    managed run) keep working unchanged.
    """
    import importlib

    if not workflow_type:
        return []

    try:
        from MCP.core.config import OrchestratorConfig  # noqa: PLC0415
        cfg = OrchestratorConfig.load()
        wf = cfg.get_workflow(workflow_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Worker W3: OrchestratorConfig.load() failed for "
            "workflow_type=%r: %s; falling back to empty validator list",
            workflow_type, exc,
        )
        return []

    if wf is None:
        logger.warning(
            "Worker W3: workflow_type=%r not registered; falling back "
            "to empty validator list",
            workflow_type,
        )
        return []

    target_phase = None
    for phase in getattr(wf, "phases", []) or []:
        if getattr(phase, "name", "") == "post_rewrite_validation":
            target_phase = phase
            break
    if target_phase is None:
        return []

    gates = getattr(target_phase, "validation_gates", None) or []
    validators: List[Any] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        dotted = gate.get("validator") or ""
        gate_id = gate.get("gate_id") or dotted
        if not dotted:
            continue
        try:
            module_path, _, class_name = dotted.rpartition(".")
            if not module_path or not class_name:
                raise ImportError(f"malformed dotted path: {dotted!r}")
            module = importlib.import_module(module_path)
            validator_cls = getattr(module, class_name)
            validators.append(validator_cls())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Worker W3: failed to import post-rewrite validator "
                "gate_id=%s validator=%s: %s",
                gate_id, dotted, exc,
            )
            if capture is not None:
                try:
                    capture.log_decision(
                        decision_type="post_rewrite_validator_import_failed",
                        decision=(
                            f"Skipped post-rewrite validator {gate_id!r} "
                            f"(dotted path {dotted!r}) — import error."
                        ),
                        rationale=(
                            f"OrchestratorConfig declared the gate at "
                            f"workflow_type={workflow_type!r} phase="
                            f"post_rewrite_validation but the validator "
                            f"class did not import: {exc}. Falling "
                            f"through means this gate will NOT fire "
                            f"during the rewrite-tier remediation "
                            f"loop; check the validator dotted path."
                        ),
                        ml_features={
                            "workflow_type": workflow_type,
                            "gate_id": gate_id,
                            "dotted_path": dotted,
                            "error": str(exc),
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
            continue
    return validators


def _load_outline_chunks(
    phase_outputs: Dict[str, Any],
    capture: Any = None,
) -> Dict[str, Any]:
    """Rehydrate the W2-persisted ``outline_chunks.json`` sidecar.

    Reads ``phase_outputs["content_generation_outline"]
    ["outline_chunks_path"]`` and returns the deserialized
    ``chunks_lookup`` dict (block_id -> list of chunk dicts) the outline
    phase fed into ``router.route_with_self_consistency(source_chunks=
    ...)``.

    On missing key OR missing file, emits a structured
    ``decision_type="rewrite_grounding_missing"`` warning capture and
    returns an empty dict. Pre-W3 the rewrite phase passed
    ``source_chunks=[]`` unconditionally; the empty-dict fallback
    preserves that behavior so the rewrite phase itself does not crash
    when the upstream sidecar is absent (legacy direct caller, mid-run
    crash before W2's sidecar emit, etc.).
    """
    outline_outputs = (phase_outputs or {}).get(
        "content_generation_outline"
    ) or {}
    chunks_path_raw = outline_outputs.get("outline_chunks_path")
    if not chunks_path_raw:
        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="rewrite_grounding_missing",
                    decision=(
                        "Rewrite phase falling through with empty "
                        "source_chunks: outline_chunks_path absent from "
                        "phase_outputs[content_generation_outline]."
                    ),
                    rationale=(
                        "Worker W2 persists outline_chunks.json sidecar "
                        "from the outline phase so W3's rewrite phase "
                        "can rehydrate per-block source chunks for the "
                        "remediation loop. Missing key indicates the "
                        "outline phase ran a pre-W2 build, the run was "
                        "resumed from a checkpoint without the sidecar, "
                        "or this is a legacy direct call without a "
                        "workflow_runner-managed phase_outputs dict. "
                        "Falling through preserves pre-W3 ``source_chunks="
                        "[]`` semantics; downstream gates will surface "
                        "the missing-grounding signal."
                    ),
                    ml_features={
                        "gate_id": "_run_content_generation_rewrite",
                        "missing_key": "outline_chunks_path",
                        "phase": "content_generation_outline",
                    },
                )
            except Exception:  # noqa: BLE001
                pass
        return {}
    try:
        chunks_path = Path(chunks_path_raw)
        if not chunks_path.exists():
            raise FileNotFoundError(str(chunks_path))
        return json.loads(chunks_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, FileNotFoundError) as exc:
        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="rewrite_grounding_missing",
                    decision=(
                        "Rewrite phase falling through with empty "
                        "source_chunks: outline_chunks.json failed "
                        f"to load ({exc})."
                    ),
                    rationale=(
                        "Sidecar path was present in phase_outputs but "
                        "the file did not load. Falling through preserves "
                        "pre-W3 ``source_chunks=[]`` semantics so the "
                        "rewrite phase itself does not crash; downstream "
                        "gates surface the missing-grounding signal."
                    ),
                    ml_features={
                        "gate_id": "_run_content_generation_rewrite",
                        "missing_key": "outline_chunks_path",
                        "sidecar_path": str(chunks_path_raw),
                        "error": str(exc),
                    },
                )
            except Exception:  # noqa: BLE001
                pass
        return {}


def _load_outline_objectives(
    phase_outputs: Dict[str, Any],
    capture: Any = None,
) -> List[Any]:
    """Rehydrate the W2-persisted ``outline_objectives.json`` sidecar.

    Symmetric mirror of :func:`_load_outline_chunks` for the
    ``objectives`` arg threaded into
    ``router.route_rewrite_with_remediation(objectives=...)``. Returns a
    list (canonical empty default) on missing key / missing file with a
    structured ``rewrite_grounding_missing`` warning capture.
    """
    outline_outputs = (phase_outputs or {}).get(
        "content_generation_outline"
    ) or {}
    objectives_path_raw = outline_outputs.get("outline_objectives_path")
    if not objectives_path_raw:
        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="rewrite_grounding_missing",
                    decision=(
                        "Rewrite phase falling through with empty "
                        "objectives: outline_objectives_path absent from "
                        "phase_outputs[content_generation_outline]."
                    ),
                    rationale=(
                        "Worker W2 persists outline_objectives.json "
                        "sidecar from the outline phase so W3's rewrite "
                        "phase can rehydrate the canonical TO-NN/CO-NN "
                        "objectives payload. Missing key indicates a "
                        "pre-W2 build / mid-run resume / legacy direct "
                        "call. Falling through preserves pre-W3 "
                        "``objectives=[]`` semantics; downstream gates "
                        "surface the missing-grounding signal."
                    ),
                    ml_features={
                        "gate_id": "_run_content_generation_rewrite",
                        "missing_key": "outline_objectives_path",
                        "phase": "content_generation_outline",
                    },
                )
            except Exception:  # noqa: BLE001
                pass
        return []
    try:
        objectives_path = Path(objectives_path_raw)
        if not objectives_path.exists():
            raise FileNotFoundError(str(objectives_path))
        payload = json.loads(objectives_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        return []
    except (OSError, ValueError, FileNotFoundError) as exc:
        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="rewrite_grounding_missing",
                    decision=(
                        "Rewrite phase falling through with empty "
                        "objectives: outline_objectives.json failed "
                        f"to load ({exc})."
                    ),
                    rationale=(
                        "Sidecar path was present in phase_outputs but "
                        "the file did not load. Falling through preserves "
                        "pre-W3 ``objectives=[]`` semantics so the "
                        "rewrite phase itself does not crash; downstream "
                        "gates surface the missing-grounding signal."
                    ),
                    ml_features={
                        "gate_id": "_run_content_generation_rewrite",
                        "missing_key": "outline_objectives_path",
                        "sidecar_path": str(objectives_path_raw),
                        "error": str(exc),
                    },
                )
            except Exception:  # noqa: BLE001
                pass
        return []


def _vocabulary_threaded_path(kwargs: Dict[str, Any]) -> Optional[str]:
    """Extract the threaded ``domain_concept_vocabulary_path``, if any.

    Reads ``kwargs['phase_outputs'].concept_extraction
    .domain_concept_vocabulary_path`` — the canonical thread the
    workflow runner sets when the Stage-3 textbook-synthesis pass ran.
    Returns ``None`` when absent (resumed / stage-subcommand runs).
    """
    phase_outputs = kwargs.get("phase_outputs") or {}
    if isinstance(phase_outputs, dict):
        ce = phase_outputs.get("concept_extraction") or {}
        candidate = ce.get("domain_concept_vocabulary_path")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _locate_domain_concept_vocabulary(
    course_code: str,
    kwargs: Dict[str, Any],
) -> Optional[Path]:
    """Resolve the ``domain_concept_vocabulary.json`` path for a course.

    R5 — delegates to the single canonical locator
    :func:`lib.ontology.curie_discovery.locate_domain_concept_vocabulary`
    so the threaded-path-then-on-disk-fallback resolution is identical
    across the gate-input router and the validation handlers. The
    ``course_code`` -> slug transform mirrors ``_run_concept_extraction``
    (lower + ``_``/space -> ``-``).

    Returns ``None`` when neither resolves — the natural backward-compat
    gate: an RDF / legacy corpus has no domain-concept vocabulary.
    """
    from lib.ontology.curie_discovery import locate_domain_concept_vocabulary

    course_slug = (course_code or "").lower().replace("_", "-").replace(
        " ", "-"
    )
    return locate_domain_concept_vocabulary(
        threaded_path=_vocabulary_threaded_path(kwargs),
        course_slug=course_slug or None,
        libv2_root=_resolve_libv2_root(kwargs.get("libv2_root")),
    )


def _resolve_minted_curie_map_for_validation(
    *,
    project_id: str,
    kwargs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build the minted-CURIE map for an inter-tier / post-rewrite gate.

    R5 — delegates to the single canonical resolver
    :func:`lib.ontology.curie_discovery.resolve_minted_curie_map`
    (which INCLUDES the on-disk fallback) so the gate-input router and
    the validation handlers resolve identically. The validation
    handlers carry a Courseforge ``project_id`` (the export folder
    name), not a course code; the course code is read from the
    project's ``project_config.json``.

    Returns ``None`` when no domain vocabulary exists (RDF / legacy
    corpora) or it is unparseable — the gate then runs legacy literal-
    token anchoring, byte-identical to the pre-minting contract.
    """
    from lib.ontology.curie_discovery import resolve_minted_curie_map

    course_code = project_id or ""
    if project_id:
        config_path = (
            courseforge_exports_dir()
            / project_id
            / "project_config.json"
        )
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                course_code = cfg.get("course_name") or course_code
            except (OSError, ValueError):
                pass
    course_slug = (course_code or "").lower().replace("_", "-").replace(
        " ", "-"
    )
    return resolve_minted_curie_map(
        threaded_path=_vocabulary_threaded_path(kwargs),
        course_id=course_code,
        course_slug=course_slug or None,
        libv2_root=_resolve_libv2_root(kwargs.get("libv2_root")),
    )


def _mint_outline_curies(
    *,
    outline_blocks: List[Any],
    course_code: str,
    kwargs: Dict[str, Any],
    capture: Any,
) -> None:
    """Mint per-course CURIEs onto curie-less outline blocks (v0.3.0).

    Locates the Stage-3 ``domain_concept_vocabulary.json``; when absent
    this is a complete no-op (RDF / legacy corpora unaffected — the key
    backward-compat contract). When present, builds the minted-CURIE
    map, compiles the vocabulary into ``domain_concept_seeds``, and for
    every outline block whose ``content`` is a dict with an EMPTY
    ``content["curies"]`` runs concept-tag extraction over the block's
    ``key_claims`` text surface and stamps the matching minted CURIEs in.

    Mutation is in-place via ``dataclasses.replace`` on each Block (the
    Block dataclass is frozen). Only EMPTY ``curies`` lists are touched
    — a block that already carries real CURIEs is never overwritten.
    """
    vocab_path = _locate_domain_concept_vocabulary(course_code, kwargs)
    if vocab_path is None:
        # No domain vocabulary — RDF / legacy corpus. No-op.
        return
    try:
        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "outline phase: failed to load domain_concept_vocabulary.json "
            "at %s (%s); skipping CURIE minting.",
            vocab_path, exc,
        )
        return

    from lib.ontology.curie_discovery import (
        build_minted_curie_map,
        minted_curie_by_canonical,
    )
    from lib.ontology.concept_tagging import extract_concept_tags
    from Trainforge.process_course import compile_domain_concept_seeds

    minted_map = build_minted_curie_map(vocabulary, course_id=course_code)
    if not minted_map:
        return
    by_canonical = minted_curie_by_canonical(minted_map)

    # Compile the vocabulary concepts into (canonical, [regex]) seed
    # pairs. ``compile_domain_concept_seeds`` expects ``[{id, aliases}]``;
    # the vocabulary's per-concept ``canonical`` field is the seed id.
    seed_input = [
        {
            "id": c.get("canonical"),
            "aliases": c.get("aliases") or [],
        }
        for c in (vocabulary.get("concepts") or [])
        if isinstance(c, dict) and c.get("canonical")
    ]
    domain_concept_seeds = compile_domain_concept_seeds(seed_input)

    import dataclasses as _dc

    minted_block_count = 0
    minted_curie_total = 0
    for idx, block in enumerate(outline_blocks):
        content = getattr(block, "content", None)
        if not isinstance(content, dict):
            continue
        existing = content.get("curies")
        # Only mint when the curies list is currently empty — never
        # overwrite real (RDF or already-minted) CURIEs.
        if existing:
            continue

        # Gather the block's text surface from ``key_claims``, handling
        # both the legacy ``List[str]`` shape and the structured
        # ``List[{claim, source_chunk_ids}]`` shape (mirrors
        # ``BlockCurieAnchoringValidator``'s text_blob assembly).
        claims_raw = content.get("key_claims") or []
        text_parts: List[str] = []
        for c in claims_raw:
            if isinstance(c, str):
                text_parts.append(c)
            elif isinstance(c, dict):
                claim_text = c.get("claim", "")
                if isinstance(claim_text, str):
                    text_parts.append(claim_text)
        text = "\n".join(text_parts)
        if not text.strip():
            continue

        # ``extract_concept_tags`` matches the block text against the
        # compiled domain-concept seeds; returns the canonical slugs of
        # the concepts the block discusses.
        matched_canonicals = extract_concept_tags(
            text, {}, domain_concept_seeds
        )
        minted_curies: List[str] = []
        for canonical in matched_canonicals:
            curie = by_canonical.get(canonical)
            if curie and curie not in minted_curies:
                minted_curies.append(curie)
        if not minted_curies:
            continue

        new_content = dict(content)
        new_content["curies"] = minted_curies
        outline_blocks[idx] = _dc.replace(block, content=new_content)
        minted_block_count += 1
        minted_curie_total += len(minted_curies)

        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="curie_minting",
                    decision=(
                        f"minted {len(minted_curies)} CURIE(s) onto "
                        f"block {block.block_id}"
                    ),
                    rationale=(
                        f"Block {block.block_id} (type={block.block_type}) "
                        f"carried empty content['curies']; its key_claims "
                        f"text matched {len(matched_canonicals)} of the "
                        f"{len(minted_map)} domain-concept-vocabulary "
                        f"concepts, minting {minted_curies} from the "
                        f"per-course CURIE map so the anchoring gate has a "
                        f"prose-corpus-valid CURIE to anchor against."
                    ),
                    ml_features={
                        "block_id": block.block_id,
                        "minted_curie_count": len(minted_curies),
                        "vocabulary_concept_count": len(minted_map),
                        "matched_concept_count": len(matched_canonicals),
                    },
                )
            except Exception:  # noqa: BLE001
                pass

    if minted_block_count:
        logger.info(
            "outline phase: minted CURIEs onto %d block(s) "
            "(%d CURIE(s) total) from %d-concept domain vocabulary at %s",
            minted_block_count, minted_curie_total, len(minted_map),
            vocab_path,
        )


async def _run_content_generation_outline(**kwargs) -> str:
    """Run the outline tier of the Phase 3 two-pass content pipeline.

    Phase 3.5 Subtask 28. For each (week, page) tuple derivable from the
    staging manifest + course_planning objectives, build a list of
    :class:`Block` stubs and dispatch them through
    :meth:`CourseforgeRouter.route_all` in outline-only mode (we
    pre-filter the input list here rather than widening route_all's
    signature; Worker N2's self-consistency widening is disjoint).

    Inputs (kwargs):
        project_id: Course project slug. Used to locate the project
            directory under ``Courseforge/exports/``.
        course_planning_path: Optional explicit path to
            ``synthesized_objectives.json``. Falls back to the project's
            canonical ``01_learning_objectives/synthesized_objectives.json``.
        staging_dir: Path to the per-run DART staging directory
            produced by ``stage_dart_outputs``.
        source_module_map_path: Optional path to ``source_module_map.json``
            for Wave 9 source-routing.
        duration_weeks: Optional weeks override (Wave 40 auto-scaling
            still respected via project_config.json).

    Outputs (JSON envelope):
        blocks_outline_path: Path to a JSONL file containing every
            outline-tier Block emitted across all weeks.
        project_id: Pass-through.
        weeks_prepared: Number of weeks for which at least one outline
            Block was emitted.
    """
    from Courseforge.scripts.blocks import Block, BLOCK_TYPES  # noqa: F401
    from MCP.tools import _content_gen_helpers as _cgh

    project_id = kwargs.get("project_id", "")
    if not project_id:
        return json.dumps({
            "success": False,
            "error": "_run_content_generation_outline requires project_id",
        })

    project_path = courseforge_exports_dir() / project_id
    out_dir = project_path / "01_outline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve duration_weeks via project_config (mirrors _generate_course_content).
    config_path = project_path / "project_config.json"
    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    duration_explicit = bool(kwargs.get("duration_weeks_explicit", False))
    kwarg_duration = kwargs.get("duration_weeks")
    if duration_explicit and kwarg_duration:
        duration_weeks = int(kwarg_duration)
    else:
        duration_weeks = int(
            config.get("duration_weeks") or kwarg_duration or 12
        )
    course_code = config.get("course_name") or project_id

    # Stage + objectives loading (mirrors _generate_course_content).
    staging_kwarg = kwargs.get("staging_dir")
    staging_dir = Path(staging_kwarg) if staging_kwarg else None
    html_files = _cgh.collect_staged_html(staging_dir, COURSEFORGE_INPUTS)
    topics = _cgh.parse_dart_html_files(html_files)
    objectives_path = (
        kwargs.get("course_planning_path")
        or config.get("objectives_path")
        or kwargs.get("objectives_path")
    )
    if not objectives_path:
        candidate = (
            project_path
            / "01_learning_objectives"
            / "synthesized_objectives.json"
        )
        if candidate.exists():
            objectives_path = str(candidate)
    terminal_objectives, chapter_objectives = _cgh.load_objectives_json(
        objectives_path
    )
    if not terminal_objectives and not chapter_objectives:
        terminal_objectives, chapter_objectives = (
            _cgh.synthesize_objectives_from_topics(topics, duration_weeks)
        )
    topics_by_week = _cgh._group_topics_by_week(topics, duration_weeks)

    # Decision capture for the outline phase.
    capture = None
    try:
        from lib.decision_capture import DecisionCapture
        capture = DecisionCapture(
            course_code=course_code,
            phase="courseforge-content-generator-outline",
            tool="courseforge",
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "DecisionCapture init failed in content_generation_outline: %s",
            exc,
        )
        capture = None

    # Instantiate the router. Fail loud on init errors — the operator
    # opted in via COURSEFORGE_TWO_PASS=true; silent fallback would
    # mask a misconfiguration.
    try:
        from Courseforge.router.router import CourseforgeRouter
        from Courseforge.router.policy import load_block_routing_policy
        router = CourseforgeRouter(
            capture=capture,
            policy=load_block_routing_policy(),
        )
    except Exception as exc:
        logger.exception(
            "CourseforgeRouter init failed in outline phase: %s", exc,
        )
        return json.dumps({
            "success": False,
            "error": f"router init failed: {exc}",
            "project_id": project_id,
        })

    # Build per-week per-page Block stubs. We mirror the page layout
    # ``_generate_course_content`` produces (overview / content_NN /
    # application / self_check / summary) at the Block level.
    all_blocks: List[Any] = []
    chunks_lookup: Dict[str, List[Any]] = {}
    weeks_with_blocks = 0

    from lib.ontology.slugs import canonical_slug as _slug

    for week_num in range(1, duration_weeks + 1):
        week_topics = (
            topics_by_week[week_num - 1]
            if (week_num - 1) < len(topics_by_week)
            else []
        )
        week_objectives = []
        if terminal_objectives:
            t_step = max(
                1,
                (len(terminal_objectives) + duration_weeks - 1)
                // duration_weeks,
            )
            t_start = (week_num - 1) * t_step
            week_objectives = list(
                terminal_objectives[t_start:t_start + t_step]
            )
        objective_ids: Tuple[str, ...] = tuple(
            str(o.get("id")) for o in week_objectives if o.get("id")
        )

        # One block stub per topic (or one minimum if no topics).
        topic_count = len(week_topics)
        page_count = max(topic_count, 1)
        week_block_added = False
        for i in range(page_count):
            topic = week_topics[i] if i < topic_count else None
            heading = (topic or {}).get("heading") or f"week_{week_num:02d}"
            page_id = f"week_{week_num:02d}_content_{i + 1:02d}"
            slug_value = _slug(heading or "content")
            # Follow-up #34 (W3.C interaction): the outline-tier emits
            # one stub Block per topic with no semantic discriminator
            # yet — the rewrite tier specialises it. Pre-fix this stub
            # was stamped ``block_type="explanation"`` which trips two
            # subtle issues: (a) the W3.C ``explanation`` matrix entry
            # in ``Courseforge/config/block_routing.yaml`` requires
            # ``source_ref`` + ``content_type`` (NOT ``curie_anchoring``),
            # so the in-loop validator chain is filtered down to a
            # subset that doesn't catch curie-less stub content, and
            # (b) every page emits at least one ``objective`` block
            # downstream anyway, so ``"objective"`` is a more honest
            # default for an empty-content stub. The ``objective``
            # matrix requires ``curie_anchoring`` + ``source_ref``,
            # which exercises the in-loop chain harder and fails-loud
            # against curie-less drafts.
            block_id = Block.stable_id(
                page_id=page_id,
                block_type="objective",
                slug=slug_value,
                idx=i,
            )
            try:
                stub = Block(
                    block_id=block_id,
                    block_type="objective",
                    page_id=page_id,
                    sequence=i,
                    content="",
                    objective_ids=objective_ids,
                    key_terms=tuple((topic or {}).get("key_terms") or []),
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "outline phase: skipping malformed Block stub "
                    "for week=%d, page=%d: %s",
                    week_num, i + 1, exc,
                )
                continue
            all_blocks.append(stub)
            chunks_lookup[block_id] = [
                {
                    "heading": heading,
                    "paragraphs": (topic or {}).get("paragraphs") or [],
                }
            ]
            week_block_added = True
        if week_block_added:
            weeks_with_blocks += 1

    # Worker W2: dispatch each block through
    # ``router.route_with_self_consistency`` with the resolved
    # validator chain so the inter-tier seam runs INSIDE the regen
    # budget (instead of catching failures after the fact). The chain
    # is resolved from the YAML ``inter_tier_validation`` phase via
    # ``_resolve_inter_tier_validators``; absent workflow_type (legacy
    # direct caller) returns []  → behaves identically to the prior
    # ``router.route()`` single-shot path.
    objectives_payload = [
        {"id": o.get("id"), "statement": o.get("statement")}
        for o in (terminal_objectives + chapter_objectives)
    ]
    workflow_type = kwargs.get("workflow_type") or ""
    validators = _resolve_inter_tier_validators(workflow_type, capture)
    outline_blocks: List[Any] = []
    for blk in all_blocks:
        block_chunks = chunks_lookup.get(blk.block_id, [])
        try:
            outlined = router.route_with_self_consistency(
                blk,
                validators=validators,
                source_chunks=block_chunks,
                objectives=objectives_payload,
            )
            outline_blocks.append(outlined)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "outline phase: route_with_self_consistency() failed "
                "for block_id=%s: %s",
                blk.block_id, exc,
            )
            # Persist a stub-with-marker so downstream sees the failure.
            try:
                import dataclasses as _dc
                outline_blocks.append(_dc.replace(
                    blk, escalation_marker="outline_budget_exhausted",
                ))
            except (TypeError, ValueError):
                continue

    # ------------------------------------------------------------------ #
    # Dynamic CURIE minting (v0.3.0 corpus-generalization initiative).
    # ------------------------------------------------------------------ #
    # A prose corpus (e.g. an OpenStax textbook) has zero RDF/SHACL
    # CURIEs in its text, so every outline block lands with an EMPTY
    # ``content["curies"]`` and ``BlockCurieAnchoringValidator`` 100%-
    # fails the ``OUTLINE_BLOCK_MISSING_CURIES`` gate. When the Stage-3
    # textbook-synthesis pass produced a ``domain_concept_vocabulary.json``
    # for this course, mint a per-course CURIE for every vocabulary
    # concept and stamp the matching minted CURIEs onto each outline
    # block whose ``curies`` is currently empty.
    #
    # Backward-compat contract: when NO vocabulary file exists (every
    # RDF / legacy corpus, every existing test fixture), this block is a
    # complete no-op — outline blocks are byte-identical to the
    # pre-minting emit. The absence of the vocabulary file is the
    # natural gate; no behavior flag is needed.
    _mint_outline_curies(
        outline_blocks=outline_blocks,
        course_code=course_code,
        kwargs=kwargs,
        capture=capture,
    )

    # Persist outline blocks to JSONL (one entry per Block via
    # to_jsonld_entry — same shape post_rewrite_validation consumes).
    blocks_outline_path = out_dir / "blocks_outline.jsonl"
    with blocks_outline_path.open("w", encoding="utf-8") as fh:
        for blk in outline_blocks:
            fh.write(json.dumps(
                _block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")

    # Worker W2: persist chunks_lookup + objectives_payload as sidecars
    # next to blocks_outline.jsonl so W3's rewrite phase can rehydrate
    # them without re-walking staging / synthesized_objectives. JSON
    # is canonicalised (indent=2, sort_keys=True) so on-disk diffs are
    # operator-readable.
    chunks_sidecar_path = out_dir / "outline_chunks.json"
    objectives_sidecar_path = out_dir / "outline_objectives.json"
    try:
        # W-D6 callout: ``ensure_ascii=False`` aligns with the canonical
        # ``lib.utils.write_jsonl`` default + matches the rest of the
        # JSONL emit chain (non-ASCII characters survive a tail-f read
        # instead of round-tripping through ``\uXXXX`` escapes).
        # Observable change for any sidecar carrying non-ASCII content.
        chunks_sidecar_path.write_text(
            json.dumps(chunks_lookup, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        objectives_sidecar_path.write_text(
            json.dumps(
                objectives_payload, indent=2, sort_keys=True, ensure_ascii=False
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning(
            "outline phase: failed to persist outline sidecars at %s: %s",
            out_dir, exc,
        )

    if capture is not None:
        try:
            capture.log_decision(
                decision_type="phase_start",
                decision=(
                    f"content_generation_outline: emitted "
                    f"{len(outline_blocks)} outline-tier blocks across "
                    f"{weeks_with_blocks} weeks."
                ),
                rationale=(
                    f"Dispatched route_with_self_consistency per block "
                    f"with {len(validators)} inter-tier validators "
                    f"resolved from workflow_type={workflow_type!r}; "
                    f"persisted to {blocks_outline_path.name} for the "
                    f"inter_tier_validation phase to consume."
                ),
                ml_features={
                    "block_count": len(outline_blocks),
                    "weeks_prepared": weeks_with_blocks,
                    "tier": "outline",
                    "validator_count": len(validators),
                },
            )
        except Exception:  # noqa: BLE001
            pass

    return json.dumps({
        "success": True,
        "blocks_outline_path": str(blocks_outline_path),
        "outline_chunks_path": str(chunks_sidecar_path),
        "outline_objectives_path": str(objectives_sidecar_path),
        "project_id": project_id,
        "weeks_prepared": weeks_with_blocks,
        "block_count": len(outline_blocks),
    })


async def _run_inter_tier_validation(**kwargs) -> str:
    """Run the four shape-discriminating Block-input validators against
    outline-tier blocks loaded from ``blocks_outline_path``.

    Phase 3.5 Subtask 29. Mirrors ``_run_post_rewrite_validation`` but
    consumes the outline-tier ``blocks_outline_path`` so a failed
    outline-tier emit (CURIE / content_type / objective_ref / source_id
    drop) is caught before the rewrite tier wastes cycles authoring HTML
    against the broken outline.

    Inputs (kwargs):
        blocks_outline_path: Path to the outline-tier JSONL file emitted
            by ``_run_content_generation_outline`` (Subtask 28).
        project_id: Course project slug (used to locate the
            ``synthesized_objectives.json`` for BlockPageObjectivesValidator).

    Outputs (JSON envelope):
        blocks_validated_path: JSONL of Blocks that passed every gate.
        blocks_failed_path: JSONL of Blocks that tripped at least one gate.
        gate_results: Per-gate ``GateResult.to_dict()`` payloads.

    Decision-capture: emits one ``block_validation_action`` event per
    failed validator with ``ml_features.tier="outline"`` so postmortem
    readers can stratify outline-tier vs rewrite-tier failures.
    """
    from pathlib import Path as _Path

    blocks_outline_path_raw = kwargs.get("blocks_outline_path") or ""
    project_id = kwargs.get("project_id") or ""

    if not blocks_outline_path_raw:
        return json.dumps({
            "success": False,
            "error": "blocks_outline_path is required",
        })

    blocks_path = _Path(blocks_outline_path_raw)
    if not blocks_path.exists():
        return json.dumps({
            "success": False,
            "error": f"blocks_outline_path does not exist: {blocks_path}",
        })

    from Courseforge.scripts.blocks import Block  # type: ignore[import-not-found]
    from Courseforge.router.inter_tier_gates import (
        BlockContentTypeValidator,
        BlockCurieAnchoringValidator,
        BlockPageObjectivesValidator,
        BlockSourceRefValidator,
    )

    raw_text = blocks_path.read_text(encoding="utf-8")
    raw_entries: list = []
    try:
        if blocks_path.suffix == ".jsonl":
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw_entries.append(json.loads(line))
        else:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                raw_entries = parsed
            elif isinstance(parsed, dict):
                raw_entries = parsed.get("blocks", []) or []
    except json.JSONDecodeError as exc:
        return json.dumps({
            "success": False,
            "error": f"failed to parse {blocks_path}: {exc}",
        })

    def _entry_to_block(entry: dict) -> Optional["Block"]:  # type: ignore[name-defined]
        accepted = {
            "block_id", "block_type", "page_id", "sequence", "content",
            "template_type", "key_terms", "objective_ids",
            "bloom_level", "bloom_verb", "bloom_range",
            "bloom_levels", "bloom_verbs", "cognitive_domain",
            "teaching_role", "content_type_label", "purpose",
            "component", "source_ids", "source_primary",
            "source_references", "content_hash",
            "validation_attempts", "escalation_marker",
        }
        kwargs_clean: dict = {}
        for k, v in (entry or {}).items():
            if k not in accepted:
                continue
            if k in {
                "key_terms", "objective_ids", "bloom_levels",
                "bloom_verbs", "source_ids", "source_references",
            }:
                if isinstance(v, list):
                    v = tuple(v) if k != "source_references" else tuple(
                        dict(r) if isinstance(r, dict) else r for r in v
                    )
            kwargs_clean[k] = v
        if "block_id" not in kwargs_clean or "block_type" not in kwargs_clean:
            return None
        kwargs_clean.setdefault("page_id", kwargs_clean.get("block_id", ""))
        kwargs_clean.setdefault("sequence", 0)
        kwargs_clean.setdefault("content", "")
        try:
            return Block(**kwargs_clean)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "inter_tier_validation: skipping malformed block entry "
                "block_id=%r: %s",
                entry.get("block_id"),
                exc,
            )
            return None

    blocks: list = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        blk = _entry_to_block(entry)
        if blk is not None:
            blocks.append(blk)

    if not blocks:
        return json.dumps({
            "success": False,
            "error": (
                f"no blocks parseable from {blocks_path}; expected JSON-LD "
                f"shape with at least one Block-shaped entry"
            ),
        })

    # Resolve canonical objectives JSON for BlockPageObjectivesValidator.
    objectives_path: Optional[_Path] = None
    if project_id:
        candidate = (
            courseforge_exports_dir()
            / project_id
            / "01_learning_objectives"
            / "synthesized_objectives.json"
        )
        if candidate.exists():
            objectives_path = candidate

    validators = [
        ("outline_curie_anchoring", BlockCurieAnchoringValidator()),
        ("outline_content_type", BlockContentTypeValidator()),
        ("outline_page_objectives", BlockPageObjectivesValidator()),
        ("outline_source_refs", BlockSourceRefValidator()),
    ]

    inputs: dict = {"blocks": blocks}
    if objectives_path is not None:
        inputs["objectives_path"] = str(objectives_path)

    # v0.3.0 dynamic CURIE minting: thread the per-course minted-CURIE
    # map into the gate inputs so BlockCurieAnchoringValidator can
    # anchor a minted (prose-corpus) CURIE via its vocabulary surface
    # forms. No-op when no domain_concept_vocabulary.json exists (RDF /
    # legacy corpora) — _resolve_minted_curie_map_for_validation returns
    # None and the validator runs legacy literal-token anchoring.
    minted_curie_map = _resolve_minted_curie_map_for_validation(
        project_id=project_id, kwargs=kwargs,
    )
    if minted_curie_map:
        inputs["minted_curie_map"] = minted_curie_map

    gate_results: list = []
    failing_gate_ids: set = set()
    for gate_id, validator in validators:
        try:
            result = validator.validate({**inputs, "gate_id": gate_id})
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "inter_tier_validation: validator %s raised: %s",
                gate_id, exc,
            )
            continue
        try:
            result_dict = result.to_dict()
        except Exception:  # pragma: no cover — defensive
            result_dict = {
                "gate_id": gate_id,
                "passed": getattr(result, "passed", False),
                "issues": [],
            }
        gate_results.append(result_dict)
        if not result.passed:
            failing_gate_ids.add(gate_id)

    # Decision-capture: emit per-failure block_validation_action events
    # with ml_features.tier="outline" so postmortem reader can stratify
    # outline-tier vs rewrite-tier failures (Subtask 26 surface).
    if failing_gate_ids:
        try:
            from lib.decision_capture import DecisionCapture
            capture = DecisionCapture(
                course_code=project_id or "inter_tier_validation",
                phase="inter_tier_validation",
                tool="courseforge",
                streaming=True,
            )
            for result_dict in gate_results:
                if result_dict.get("passed"):
                    continue
                gate_id = result_dict.get("gate_id", "unknown_gate")
                issues = result_dict.get("issues", []) or []
                top_issues = issues[:3]
                summary = "; ".join(
                    f"{i.get('code','?')}({i.get('severity','?')}):"
                    f"{i.get('message','')}"
                    for i in top_issues
                ) or "no_issues"
                capture.log_decision(
                    decision_type="block_validation_action",
                    decision=(
                        f"inter_tier_validation:{gate_id}:"
                        f"{result_dict.get('passed')}"
                    ),
                    rationale=(
                        f"Inter-tier gate {gate_id} returned "
                        f"passed={result_dict.get('passed')} on "
                        f"{len(blocks)} outline-tier blocks; "
                        f"top_issues=[{summary}]"
                    ),
                    ml_features={
                        "gate_id": gate_id,
                        "passed": result_dict.get("passed"),
                        "issues_count": len(issues),
                        "block_count": len(blocks),
                        # Subtask 26: tier provenance — outline-tier
                        # in-loop validator failure (vs rewrite-tier
                        # post-emit failures handled by post_rewrite_validation).
                        "tier": "outline",
                    },
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "inter_tier_validation: decision-capture emit failed: %s",
                exc,
            )

    out_dir = blocks_path.parent
    validated_path = out_dir / "blocks_validated.jsonl"
    failed_path = out_dir / "blocks_failed.jsonl"

    failed_block_ids: set = set()
    for result_dict in gate_results:
        if result_dict.get("passed"):
            continue
        for issue in result_dict.get("issues", []) or []:
            loc = issue.get("location") if isinstance(issue, dict) else None
            if isinstance(loc, str) and loc:
                failed_block_ids.add(loc)

    # Escalated blocks ride through to validated_path regardless of
    # validator outcome — they are marker-bearing by design (Wave-7
    # escalate_immediately) and the rewrite phase needs to see them to
    # author from scratch. Without this, blocks_validated.jsonl is empty
    # when the corpus is all-objective and the workflow halts.
    with validated_path.open("w", encoding="utf-8") as fh:
        for blk in blocks:
            if blk.block_id in failed_block_ids and blk.escalation_marker is None:
                continue
            fh.write(json.dumps(
                _block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")
    with failed_path.open("w", encoding="utf-8") as fh:
        for blk in blocks:
            if blk.block_id not in failed_block_ids:
                continue
            if blk.escalation_marker is not None:
                continue
            fh.write(json.dumps(
                _block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")

    return json.dumps({
        "success": True,
        "blocks_validated_path": str(validated_path),
        "blocks_failed_path": str(failed_path),
        "gate_results": gate_results,
        "block_count": len(blocks),
        "failed_block_count": len(failed_block_ids),
    })


async def _run_content_generation_rewrite(**kwargs) -> str:
    """Run the rewrite tier of the Phase 3 two-pass content pipeline.

    Phase 3.5 Subtask 30. Reads the validated outline-tier Blocks from
    ``blocks_validated_path`` (the inter_tier_validation phase output),
    then dispatches each block through :meth:`CourseforgeRouter.route`
    with ``tier="rewrite"``. The rewrite tier is responsible for the
    final HTML body emit; we persist both the per-block JSONL
    (``blocks_final.jsonl``, consumed by post_rewrite_validation) and
    the per-page HTML files (consumed by packaging + content_grounding).

    Inputs (kwargs):
        blocks_validated_path: JSONL of validated outline-tier blocks.
        project_id: Course project slug.
        source_module_map_path: Optional Wave 9 source-routing path.
        staging_dir: Staging directory from stage_dart_outputs.

    Outputs (JSON envelope):
        content_paths: Comma-joined string of emitted HTML paths
            (router canonical key for legacy gate-input parsers).
        page_paths: List of emitted HTML paths.
        content_dir: Directory where pages were written.
        blocks_final_path: JSONL of rewrite-tier Blocks (consumed by
            post_rewrite_validation).
        project_id: Pass-through.
    """
    from pathlib import Path as _Path

    blocks_validated_path_raw = kwargs.get("blocks_validated_path") or ""
    project_id = kwargs.get("project_id") or ""

    if not project_id:
        return json.dumps({
            "success": False,
            "error": "_run_content_generation_rewrite requires project_id",
        })
    if not blocks_validated_path_raw:
        return json.dumps({
            "success": False,
            "error": "blocks_validated_path is required",
        })

    blocks_path = _Path(blocks_validated_path_raw)
    if not blocks_path.exists():
        return json.dumps({
            "success": False,
            "error": f"blocks_validated_path does not exist: {blocks_path}",
        })

    project_path = courseforge_exports_dir() / project_id
    out_dir = project_path / "04_rewrite"
    out_dir.mkdir(parents=True, exist_ok=True)
    content_dir = project_path / "03_content_development"
    content_dir.mkdir(parents=True, exist_ok=True)

    config_path = project_path / "project_config.json"
    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    course_code = config.get("course_name") or project_id

    from Courseforge.scripts.blocks import Block  # noqa: F401

    raw_text = blocks_path.read_text(encoding="utf-8")
    raw_entries: list = []
    try:
        if blocks_path.suffix == ".jsonl":
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw_entries.append(json.loads(line))
        else:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                raw_entries = parsed
            elif isinstance(parsed, dict):
                raw_entries = parsed.get("blocks", []) or []
    except json.JSONDecodeError as exc:
        return json.dumps({
            "success": False,
            "error": f"failed to parse {blocks_path}: {exc}",
        })

    def _entry_to_block(entry: dict) -> Optional["Block"]:  # type: ignore[name-defined]
        accepted = {
            "block_id", "block_type", "page_id", "sequence", "content",
            "template_type", "key_terms", "objective_ids",
            "bloom_level", "bloom_verb", "bloom_range",
            "bloom_levels", "bloom_verbs", "cognitive_domain",
            "teaching_role", "content_type_label", "purpose",
            "component", "source_ids", "source_primary",
            "source_references", "content_hash",
            "validation_attempts", "escalation_marker",
        }
        kwargs_clean: dict = {}
        for k, v in (entry or {}).items():
            if k not in accepted:
                continue
            if k in {
                "key_terms", "objective_ids", "bloom_levels",
                "bloom_verbs", "source_ids", "source_references",
            }:
                if isinstance(v, list):
                    v = tuple(v) if k != "source_references" else tuple(
                        dict(r) if isinstance(r, dict) else r for r in v
                    )
            kwargs_clean[k] = v
        if "block_id" not in kwargs_clean or "block_type" not in kwargs_clean:
            return None
        kwargs_clean.setdefault("page_id", kwargs_clean.get("block_id", ""))
        kwargs_clean.setdefault("sequence", 0)
        kwargs_clean.setdefault("content", "")
        try:
            return Block(**kwargs_clean)
        except (TypeError, ValueError):
            return None

    outline_blocks: list = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        blk = _entry_to_block(entry)
        if blk is not None:
            outline_blocks.append(blk)

    capture = None
    try:
        from lib.decision_capture import DecisionCapture
        capture = DecisionCapture(
            course_code=course_code,
            phase="courseforge-content-generator-rewrite",
            tool="courseforge",
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "DecisionCapture init failed in content_generation_rewrite: %s",
            exc,
        )
        capture = None

    try:
        from Courseforge.router.router import CourseforgeRouter
        from Courseforge.router.policy import load_block_routing_policy
        router = CourseforgeRouter(
            capture=capture,
            policy=load_block_routing_policy(),
        )
    except Exception as exc:
        logger.exception(
            "CourseforgeRouter init failed in rewrite phase: %s", exc,
        )
        return json.dumps({
            "success": False,
            "error": f"router init failed: {exc}",
            "project_id": project_id,
        })

    # Rewrite-tier routing intentionally passes validators=[] (in-loop
    # remediation OFF; the standalone post_rewrite_validation phase owns
    # validation — operator decision 2026-06-09). Each block is dispatched
    # through ``router.route_rewrite_with_remediation`` for its sidecar-
    # rehydrated ``source_chunks`` / ``objectives`` grounding (loaded from
    # the W2-persisted ``outline_chunks.json`` / ``outline_objectives.json``
    # next to ``blocks_outline.jsonl``; the workflow_runner threads the
    # sidecar paths in as kwargs via the YAML ``inputs_from`` block), not
    # for an in-loop validator chain.
    # Reconstruct the phase_outputs sub-shape the loader helpers expect
    # from the resolved path kwargs the workflow_runner threaded in.
    _phase_outputs_proxy: Dict[str, Any] = {
        "content_generation_outline": {
            "outline_chunks_path": kwargs.get("outline_chunks_path"),
            "outline_objectives_path": kwargs.get("outline_objectives_path"),
        },
    }
    chunks_lookup = _load_outline_chunks(_phase_outputs_proxy, capture)
    objectives_payload = _load_outline_objectives(
        _phase_outputs_proxy, capture,
    )

    # Load source_module_map for page-level source attribution so the
    # rewrite-phase backstop can synthesize per-block source CURIEs both
    # at the block.content level AND at the per-page section wrapper.
    _source_map: Dict[str, Any] = {}
    _source_map_path = project_path / "source_module_map.json"
    if _source_map_path.exists():
        try:
            _source_map = json.loads(
                _source_map_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            _source_map = {}

    def _page_source_ids(page_id: str) -> list:
        parts = page_id.split("_")
        if len(parts) < 3:
            return []
        week_key = "_".join(parts[:2])
        rest = "_".join(parts[2:])
        week = _source_map.get(week_key, {})
        if not isinstance(week, dict):
            return []
        page_entry = week.get(rest, {})
        if not page_entry and "_" in rest:
            page_entry = week.get(rest.rsplit("_", 1)[0], {})
        if not isinstance(page_entry, dict):
            return []
        sids = list(page_entry.get("primary") or [])
        sids.extend(page_entry.get("contributing") or [])
        seen: set = set()
        out: list = []
        for sid in sids:
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
        return out

    # Route EVERY block through rewrite, including marker-bearing blocks
    # (Wave-7 escalate_immediately) — the rewrite tier authors them from
    # scratch as HTML. In-loop validators are skipped (validators=[]);
    # the backstop below post-processes the Qwen output (canonical
    # objective IDs + CURIE injection) and the post_rewrite_validation
    # phase runs the validator chain on the corrected content.
    rewrite_blocks: list = []
    import dataclasses as _dc
    for blk in outline_blocks:
        block_chunks = chunks_lookup.get(blk.block_id, []) if isinstance(
            chunks_lookup, dict
        ) else []
        try:
            rewritten = router.route_rewrite_with_remediation(
                blk,
                validators=[],
                source_chunks=block_chunks,
                objectives=objectives_payload,
            )
            # Backstop: scrub Qwen-invented objective IDs onto the
            # canonical Block.objective_ids, and inject page-level
            # source CURIEs as <cite> text so the rewrite-tier
            # CURIE-anchoring validator sees an anchored token.
            if isinstance(rewritten.content, str):
                new_content = rewritten.content
                if rewritten.objective_ids:
                    canonical_oids = ",".join(rewritten.objective_ids)
                    new_content = _OBJ_ID_RE.sub(
                        f'data-cf-objective-id="{canonical_oids}"',
                        new_content,
                    )
                block_sids = list(rewritten.source_ids or ())
                if not block_sids:
                    block_sids = _page_source_ids(rewritten.page_id)
                if block_sids:
                    new_content = (
                        f"{new_content}<cite class=\"source-attribution\">"
                        f"{'; '.join(block_sids)}</cite>"
                    )
                if new_content != rewritten.content:
                    rewritten = _dc.replace(rewritten, content=new_content)
            rewrite_blocks.append(rewritten)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "rewrite phase: route_rewrite_with_remediation() "
                "failed for block_id=%s: %s",
                blk.block_id, exc,
            )
            try:
                rewrite_blocks.append(_dc.replace(
                    blk, escalation_marker="validator_consensus_fail",
                ))
            except (TypeError, ValueError):
                continue

    # Persist final blocks JSONL (consumed by post_rewrite_validation).
    blocks_final_path = out_dir / "blocks_final.jsonl"
    with blocks_final_path.open("w", encoding="utf-8") as fh:
        for blk in rewrite_blocks:
            fh.write(json.dumps(
                _block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")

    # Group blocks by page_id and emit a minimal HTML page per group.
    # The two-pass router's HTML is the rewritten Block.content (string);
    # we wrap it in the minimum semantic structure the validators
    # require (an objectives section per page, plus the body text).
    pages_by_id: Dict[str, list] = {}
    for blk in rewrite_blocks:
        pages_by_id.setdefault(blk.page_id, []).append(blk)

    page_paths: list = []
    for page_id, page_blocks in pages_by_id.items():
        page_path = content_dir / f"{page_id}.html"
        objective_ids = []
        for b in page_blocks:
            for oid in b.objective_ids or ():
                if oid not in objective_ids:
                    objective_ids.append(oid)
        objective_lis = "".join(
            f'<li data-cf-objective-id="{oid}">{oid}</li>'
            for oid in objective_ids
        )
        body_parts = []
        for b in page_blocks:
            # W5: marker-bearing blocks (outline_budget_exhausted /
            # structural_unfixable / validator_consensus_fail) MUST NOT
            # ship into per-page HTML. They persist on disk in
            # blocks_final.jsonl for re-execution / audit, but they
            # never become part of the IMSCC body.
            if b.escalation_marker is not None and not (b.content or "").strip():
                if capture is not None:
                    try:
                        capture.log_decision(
                            decision_type="block_packaging_skipped",
                            decision=(
                                f"Skipped HTML emit for block_id="
                                f"{b.block_id} (block_type={b.block_type}, "
                                f"escalation_marker={b.escalation_marker})."
                            ),
                            rationale=(
                                "W5 packaging gate: blocks carrying a "
                                "non-null escalation_marker (consensus "
                                "failure / outline budget exhausted / "
                                "structural unfixable) MUST NOT ship "
                                "into the IMSCC. The block remains on "
                                "disk in blocks_final.jsonl for "
                                "re-execution; this skip prevents an "
                                "unvalidated block_id="
                                f"{b.block_id} from leaking into the "
                                "per-page HTML and bypassing the "
                                "post-rewrite validator chain."
                            ),
                            ml_features={
                                "gate_id": "_run_content_generation_rewrite",
                                "block_id": b.block_id,
                                "block_type": b.block_type,
                                "escalation_marker": b.escalation_marker,
                                "page_id": page_id,
                            },
                        )
                    except Exception:  # noqa: BLE001
                        pass
                continue
            content = b.content if isinstance(b.content, str) else ""
            if not content.strip():
                continue
            # Stamp canonical data-cf-source-ids / data-cf-objective-id
            # on the section wrapper from the Block dataclass fields,
            # falling back to the page-level source_module_map for
            # blocks whose source_ids were stripped upstream.
            block_sids = list(b.source_ids or ())
            if not block_sids:
                block_sids = _page_source_ids(b.page_id)
            source_ids_attr = ""
            cite_html = ""
            if block_sids:
                source_ids_attr = (
                    f' data-cf-source-ids="{",".join(block_sids)}"'
                )
                cite_html = (
                    '<cite class="source-attribution">'
                    + "; ".join(block_sids)
                    + "</cite>"
                )
            objective_attr = ""
            if b.objective_ids:
                objective_attr = (
                    f' data-cf-objective-id="{",".join(b.objective_ids)}"'
                )
            body_parts.append(
                f'<section data-cf-block-id="{b.block_id}"'
                f'{source_ids_attr}{objective_attr}>'
                f'{content}{cite_html}</section>'
            )
        page_html = (
            "<!DOCTYPE html>\n<html><head>"
            f"<title>{page_id}</title></head><body>"
            f'<section class="objectives"><h2>Objectives</h2>'
            f'<ul>{objective_lis}</ul></section>'
            f"<main>{''.join(body_parts)}</main>"
            "</body></html>"
        )
        try:
            page_path.write_text(page_html, encoding="utf-8")
            page_paths.append(str(page_path))
        except OSError as exc:
            logger.warning(
                "rewrite phase: failed to write %s: %s", page_path, exc,
            )

    if capture is not None:
        try:
            capture.log_decision(
                decision_type="phase_start",
                decision=(
                    f"content_generation_rewrite: emitted "
                    f"{len(rewrite_blocks)} rewrite-tier blocks across "
                    f"{len(page_paths)} pages."
                ),
                rationale=(
                    f"Pre-filtered route() per block with tier='rewrite'; "
                    f"persisted to {blocks_final_path.name} for "
                    f"post_rewrite_validation to consume."
                ),
                ml_features={
                    "block_count": len(rewrite_blocks),
                    "page_count": len(page_paths),
                    "tier": "rewrite",
                },
            )
        except Exception:  # noqa: BLE001
            pass

    return json.dumps({
        "success": True,
        "content_paths": ",".join(page_paths),
        "page_paths": page_paths,
        "content_dir": str(content_dir),
        "blocks_final_path": str(blocks_final_path),
        "project_id": project_id,
        "block_count": len(rewrite_blocks),
    })


def _build_tool_registry() -> dict:
    """
    Build a tool registry mapping tool names to callable async functions.

    Imports and wraps all MCP tool functions so the TaskExecutor
    can invoke them by name.
    """
    registry = {}

    # DART tools
    async def _extract_and_convert_pdf(**kwargs):
        """Extract text from PDF and convert to clean, accessible HTML.

        Strategy:
        1. Try multi-source synthesis if combined JSON exists
        2. Extract text via pdftotext
        3. Build clean semantic HTML from the extracted text
           (strips page numbers, TOC artifacts, headers/footers)
        """
        from lib.paths import DART_PATH

        pdf_path = kwargs.get("pdf_path", "")
        course_code = kwargs.get("course_code")
        output_dir_str = kwargs.get("output_dir")

        pdf = Path(pdf_path)
        out_dir = Path(output_dir_str) if output_dir_str else DART_PATH / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Output filename is keyed on the PDF basename so multi-PDF corpora
        # don't collide on a shared `course_code`. `code` is retained for
        # combined-JSON lookups + HTML title below.
        code = course_code or pdf.stem
        out_stem = pdf.stem

        sys.path.insert(0, str(DART_PATH))

        # Strategy 1: If combined JSON exists, use multi-source synthesis
        combined_dir = DART_PATH / "batch_output" / "combined"
        combined_json = combined_dir / f"{code}_combined.json"

        if combined_json.exists():
            try:
                from multi_source_interpreter import convert_single_pdf
                html_output = out_dir / f"{out_stem}_synthesized.html"
                convert_single_pdf(str(combined_json), str(html_output))
                # Wave 32 Deliverable B: surface html_path alongside
                # output_path (legacy alias) so DartMarkersValidator
                # gate builder picks it up as a canonical key.
                return json.dumps({
                    "success": True,
                    "output_path": str(html_output),
                    "html_path": str(html_output),
                    "method": "multi_source_synthesis",
                })
            except ImportError:
                pass

        # Strategy 2: Extract text via pdftotext, then build accessible HTML
        import re as _re
        import subprocess

        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf), "-"],
                capture_output=True, text=True, timeout=120,
            )
            raw_text = result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fallback: try pdf_converter
            try:
                from pdf_converter.converter import PDFToAccessibleHTML
                converter = PDFToAccessibleHTML()
                conv_result = converter.convert(str(pdf), str(out_dir))
                # Wave 32 Deliverable B: mirror html_path alongside
                # output_path (router canonical key).
                return json.dumps({
                    "success": conv_result.success,
                    "output_path": conv_result.html_path,
                    "html_path": conv_result.html_path,
                    "method": "pdf_converter",
                })
            except Exception as e2:
                return json.dumps({"error": f"DART conversion failed: {e2}"})

        if len(raw_text.strip()) < 100:
            return json.dumps({"error": "No meaningful text extracted from PDF"})

        # Build accessible HTML from raw extracted text.
        # Use the PDF stem (e.g. "demo_ontology_engineering") as the doc title
        # rather than the course_code — otherwise every PDF in a multi-PDF
        # corpus gets the same <h1>/<title> (the course code), which poisons
        # downstream objective extraction across the corpus.
        pretty_title = out_stem.replace("-", " ").replace("_", " ").strip()
        html_output = out_dir / f"{out_stem}_accessible.html"
        # Pass ``source_pdf`` so Wave 16 extraction enrichment kicks in
        # (pdfplumber tables + PyMuPDF figures + optional OCR). Pass
        # ``output_path`` so Wave 17 figure persistence auto-derives a
        # sibling ``{stem}_figures/`` directory; the caller override
        # (``kwargs["figures_dir"]``) still wins when set explicitly.
        # The extractor gracefully degrades when optional deps are
        # missing so this never regresses the raw-text-only path.
        # Wave 29 Defect 5: prefer the workflow-wide canonical course
        # code (derived from ``params.course_name`` via
        # :func:`normalize_course_code`) when the orchestrator threaded
        # it through. Falls back to the PDF-stem-derived code inside
        # ``_raw_text_to_accessible_html`` when absent (legacy path).
        # Wave 30 Gap 1: thread an LLM backend through so
        # ``AltTextGenerator.generate()`` actually runs on every figure.
        # Precedence: explicit ``kwargs["llm"]`` (tests / CLI override) >
        # env-resolved backend when ``ANTHROPIC_API_KEY`` is set + the
        # api-mode flag is on. Without a backend the figure template
        # falls back to the WCAG-decorative placeholder (alt='' +
        # role='presentation') and a single warning is logged — the
        # pipeline does not crash on the no-LLM-available path.
        #
        # Wave 73: also honor ``--mode local`` when ``ED4ALL_RUN_ID`` is
        # set — builds a ``MailboxBrokeredBackend`` so every LLM call
        # site (classifier, alt-text) routes through the TaskMailbox
        # to a Claude Code operator loop. Previously local mode
        # unconditionally produced ``_llm_backend=None``, which meant
        # alt-text / classifier silently dropped to heuristic / WCAG
        # decorative fallbacks even when the operator *could* service
        # real Claude completions.
        _llm_backend = kwargs.get("llm")
        if _llm_backend is None:
            try:
                import os as _os_inner
                _api_key_present = bool(_os_inner.environ.get("ANTHROPIC_API_KEY"))
                _mode = _os_inner.environ.get("LLM_MODE", "local").strip().lower()
                _run_id = _os_inner.environ.get("ED4ALL_RUN_ID", "").strip()
                from MCP.orchestrator.llm_backend import build_backend
                if _api_key_present and _mode == "api":
                    _llm_backend = build_backend()
                elif _mode == "local" and _run_id:
                    _llm_backend = build_backend()
            except Exception as _exc:  # noqa: BLE001 — never block on backend resolution
                logger.debug(
                    "Wave 30 Gap 1 / Wave 73: LLM backend auto-resolve "
                    "failed (%s); falling back to decorative alt-text",
                    _exc,
                )
                _llm_backend = None

        html_content = _raw_text_to_accessible_html(
            raw_text,
            pretty_title,
            source_pdf=str(pdf),
            output_path=str(html_output),
            figures_dir=kwargs.get("figures_dir"),
            canonical_course_code=kwargs.get("canonical_course_code"),
            llm=_llm_backend,
        )
        html_output.write_text(html_content, encoding="utf-8")

        word_count = len(_re.findall(r"\b\w+\b", html_content))

        # Wave 32 Deliverable B: surface html_path alongside the
        # legacy output_path alias so the DartMarkersValidator gate
        # builder stops reporting ``missing inputs: html_path``.
        return json.dumps({
            "success": True,
            "output_path": str(html_output),
            "html_path": str(html_output),
            "method": "pdftotext_to_html",
            "word_count": word_count,
            "html_length": len(html_content),
        })

    registry["extract_and_convert_pdf"] = _extract_and_convert_pdf

    # Pipeline tools - stage_dart_outputs
    # Registry variant now has full Wave 8 parity with the @mcp.tool() variant
    # (role-tagging, .quality.json copy, role-tagged manifest entries). The
    # MCP-tool variant at lines 316-451 remains the source of truth for the
    # copy/role logic; this wrapper just adapts kwargs into the Wave 8
    # staging pipeline.
    async def _stage_dart_outputs(**kwargs):
        """Stage DART outputs to Courseforge inputs with Wave 8 role-tagging.

        Stages HTML (role=content), *_synthesized.json provenance sidecars
        (role=provenance_sidecar), and *.quality.json confidence sidecars
        (role=quality_sidecar) to ``COURSEFORGE_INPUTS/{run_id}/`` and
        emits a role-tagged ``staging_manifest.json``. Kept in parity with
        the @mcp.tool() variant so pipeline-dispatch runs do not silently
        drop Wave 8 metadata (audit Q4 finding).

        Wave 74 cleanup: honours ``stage_mode`` kwarg (or ``ED4ALL_STAGE_MODE``
        env) — defaults to ``symlink`` to skip 70MB/run of duplicated DART
        output. ``copy`` preserves legacy behaviour; ``hardlink`` is a Windows
        fallback when symlinks are blocked.
        """
        run_id = kwargs.get("run_id", "")
        dart_html_paths = kwargs.get("dart_html_paths", "")
        course_name = kwargs.get("course_name", "")
        stage_mode = kwargs.get("stage_mode")

        try:
            mode = _resolve_stage_mode(stage_mode)
            staging_dir = COURSEFORGE_INPUTS / run_id
            staging_dir.mkdir(parents=True, exist_ok=True)

            staged_files: list = []
            staged_entries: list = []
            errors: list = []

            html_paths = [Path(p.strip()) for p in dart_html_paths.split(",") if p.strip()]

            for html_path in html_paths:
                if not html_path.exists():
                    errors.append(f"DART output not found: {html_path}")
                    continue

                # Stage HTML file (role=content)
                dest = staging_dir / html_path.name
                _stage_file(html_path, dest, mode)
                staged_files.append(str(dest))
                staged_entries.append({"path": html_path.name, "role": "content"})

                # Wave 19: also stage ``{stem}_figures/`` when present.
                figures_dir_src = html_path.parent / f"{html_path.stem}_figures"
                if figures_dir_src.is_dir():
                    figures_dir_dest = staging_dir / figures_dir_src.name
                    _stage_tree(figures_dir_src, figures_dir_dest, mode)
                    staged_files.append(str(figures_dir_dest))
                    staged_entries.append({
                        "path": figures_dir_src.name,
                        "role": "figures_bundle",
                    })

                # Stage accompanying JSON if it exists (DART synthesized metadata).
                json_path = html_path.with_suffix(".json")
                if json_path.exists():
                    json_dest = staging_dir / json_path.name
                    _stage_file(json_path, json_dest, mode)
                    staged_files.append(str(json_dest))
                    staged_entries.append({
                        "path": json_path.name,
                        "role": "provenance_sidecar",
                    })

                # Also check for the _synthesized.json pattern.
                synth_json_name = html_path.stem.replace("_synthesized", "") + "_synthesized.json"
                synth_json_path = html_path.parent / synth_json_name
                if synth_json_path.exists() and str(synth_json_path) != str(json_path):
                    synth_json_dest = staging_dir / synth_json_name
                    _stage_file(synth_json_path, synth_json_dest, mode)
                    staged_files.append(str(synth_json_dest))
                    staged_entries.append({
                        "path": synth_json_name,
                        "role": "provenance_sidecar",
                    })

                # Wave 8: also stage the DART quality sidecar if one exists.
                quality_name = html_path.stem + ".quality.json"
                quality_path = html_path.parent / quality_name
                if quality_path.exists():
                    quality_dest = staging_dir / quality_name
                    _stage_file(quality_path, quality_dest, mode)
                    staged_files.append(str(quality_dest))
                    staged_entries.append({
                        "path": quality_name,
                        "role": "quality_sidecar",
                    })

            if errors and not staged_files:
                return json.dumps({
                    "success": False,
                    "error": "No files staged",
                    "errors": errors,
                })

            manifest = {
                "run_id": run_id,
                "course_name": course_name,
                "staged_at": datetime.now().isoformat(),
                "staged_files": staged_files,
                "files": staged_entries,
                "errors": errors if errors else None,
            }
            manifest_path = staging_dir / "staging_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            return json.dumps({
                "success": True,
                "staging_dir": str(staging_dir),
                "staged_files": staged_files,
                "files": staged_entries,
                "file_count": len(staged_files),
                "manifest_path": str(manifest_path),
                "stage_mode": mode,
                "warnings": errors if errors else None,
            })
        except Exception as e:
            logger.error(f"Registry _stage_dart_outputs failed: {e}")
            return json.dumps({"error": str(e)})

    registry["stage_dart_outputs"] = _stage_dart_outputs

    # Courseforge tools
    try:
        from MCP.tools.courseforge_tools import register_courseforge_tools as _cf  # noqa: F401
        # Import the tool functions from courseforge_tools module scope
        # These are registered as MCP tools but we need direct callables

        async def _create_course_project(**kwargs):
            logger.info(f"_create_course_project called with kwargs: {list(kwargs.keys())}")
            logger.info(f"  objectives_path raw: {repr(kwargs.get('objectives_path'))}")
            course_name = kwargs.get("course_name", "")
            objectives_path = kwargs.get("objectives_path") or ""
            duration_weeks = kwargs.get("duration_weeks", 12)
            credit_hours = kwargs.get("credit_hours", 3)

            # Use the project creation logic directly
            project_id = f"PROJ-{course_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            project_path = courseforge_exports_dir() / project_id

            project_path.mkdir(parents=True, exist_ok=True)
            for subdir in ["00_template_analysis", "01_learning_objectives",
                           "02_course_planning", "03_content_development",
                           "04_quality_validation", "05_final_package",
                           "agent_workspaces"]:
                (project_path / subdir).mkdir(exist_ok=True)

            config_path = project_path / "project_config.json"

            # If config already exists (from a prior phase), update rather than overwrite
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
                # Only update fields that have real values
                if course_name:
                    config_data["course_name"] = course_name
                if objectives_path:
                    config_data["objectives_path"] = str(objectives_path)
                if duration_weeks:
                    config_data["duration_weeks"] = duration_weeks
            else:
                config_data = {
                    "project_id": project_id,
                    "course_name": course_name,
                    "objectives_path": str(objectives_path) if objectives_path else None,
                    "duration_weeks": duration_weeks,
                    "credit_hours": credit_hours,
                    "created_at": datetime.now().isoformat(),
                    "status": "initialized",
                }

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            # Generate default objective IDs from course name and weeks
            duration = duration_weeks if isinstance(duration_weeks, int) else 12
            objective_ids = [
                f"{course_name}_OBJ_{i}" for i in range(1, duration + 1)
            ]

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "project_path": str(project_path),
                "objective_ids": ",".join(objective_ids),
                "config": config_data,
            })

        registry["create_course_project"] = _create_course_project

        # ============================================================================
        # Wave 24: _extract_textbook_structure — replaces the textbook-ingestor's
        # pre-Wave-24 stub dispatch (which routed to create_course_project and
        # produced an empty skeleton). Runs SemanticStructureExtractor.extract()
        # over every staged DART HTML file, merges per-file chapter/section
        # hierarchies into a single textbook_structure.json, and publishes the
        # path via phase_outputs.objective_extraction.textbook_structure_path.
        # ============================================================================
        async def _extract_textbook_structure(**kwargs):
            """Extract textbook structure from staged DART HTML.

            Called during the ``objective_extraction`` phase of
            ``textbook_to_course``. Reads every HTML file under
            ``staging_dir`` (the directory produced by the prior
            ``staging`` phase), runs the mature
            ``SemanticStructureExtractor`` over each, merges chapters
            across files into a single unified structure, and writes
            ``{project_path}/01_learning_objectives/textbook_structure.json``.

            Required kwargs: ``course_name`` (used to mint / locate the
            Courseforge export dir). Optional: ``staging_dir``,
            ``duration_weeks``, ``objectives_path`` (threaded through to
            project_config.json so downstream phases see them).
            """
            from lib.semantic_structure_extractor.semantic_structure_extractor import (
                SemanticStructureExtractor,
            )

            course_name = kwargs.get("course_name", "")
            if not course_name:
                return json.dumps({
                    "error": "extract_textbook_structure requires course_name",
                })
            duration_weeks = kwargs.get("duration_weeks", 12)
            duration_explicit = bool(kwargs.get("duration_weeks_explicit", True))
            objectives_path = kwargs.get("objectives_path") or ""
            staging_kwarg = kwargs.get("staging_dir")

            # Resolve or create the project path. We reuse the
            # create_course_project layout so downstream phases (which
            # accept project_id as an input) find the same structure.
            project_id = f"PROJ-{course_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            project_path = courseforge_exports_dir() / project_id
            project_path.mkdir(parents=True, exist_ok=True)
            for subdir in ("00_template_analysis", "01_learning_objectives",
                           "02_course_planning", "03_content_development",
                           "04_quality_validation", "05_final_package",
                           "agent_workspaces"):
                (project_path / subdir).mkdir(exist_ok=True)

            # Persist/refresh project_config.json so course_planning + later
            # phases (content_generation, trainforge_assessment) see a real
            # objectives_path once the planner emits synthesized_objectives.json.
            config_path = project_path / "project_config.json"
            config_data: Dict[str, Any] = {
                "project_id": project_id,
                "course_name": course_name,
                "duration_weeks": int(duration_weeks) if duration_weeks else 12,
                "credit_hours": kwargs.get("credit_hours", 3),
                "created_at": datetime.now().isoformat(),
                "status": "extracting_structure",
            }
            if objectives_path:
                config_data["objectives_path"] = str(objectives_path)
            config_path.write_text(
                json.dumps(config_data, indent=2), encoding="utf-8",
            )

            # Locate staged HTML. Prefer the explicit kwarg from the
            # workflow runner; fall back to the most-recent staging
            # manifest under Courseforge/inputs/textbooks when absent.
            staging_dir: Optional[Path] = None
            if staging_kwarg:
                staging_dir = Path(staging_kwarg)
            if staging_dir is None or not staging_dir.exists():
                # Fallback: the Courseforge inputs area.
                cf_inputs = _PROJECT_ROOT / "Courseforge" / "inputs" / "textbooks"
                if cf_inputs.exists():
                    # Use the most recent subdir as staging.
                    subdirs = sorted(
                        (p for p in cf_inputs.iterdir() if p.is_dir()),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if subdirs:
                        staging_dir = subdirs[0]

            html_files: List[Path] = []
            if staging_dir and staging_dir.exists():
                html_files = sorted(staging_dir.rglob("*.html"))

            # Run the extractor across every HTML file and merge.
            extractor = SemanticStructureExtractor()
            merged_chapters: List[Dict[str, Any]] = []
            per_file_results: List[Dict[str, Any]] = []
            extraction_errors: List[Dict[str, str]] = []
            for html_path in html_files:
                try:
                    content = html_path.read_text(encoding="utf-8", errors="ignore")
                    structure = extractor.extract(content, str(html_path), format="html")
                    per_file_results.append({
                        "source_file": str(html_path),
                        "chapters_count": len(structure.get("chapters", [])),
                    })
                    for ch in structure.get("chapters", []) or []:
                        if isinstance(ch, dict):
                            # Preserve source_file for downstream routing.
                            ch.setdefault("source_file", str(html_path))
                            merged_chapters.append(ch)
                except Exception as e:  # noqa: BLE001 - best-effort merge
                    extraction_errors.append({
                        "source_file": str(html_path),
                        "error": str(e),
                    })

            # De-duplicate chapter IDs across files: append a disambiguator
            # when two files emit the same synthesized ``chN`` id.
            seen_ids: set = set()
            for ch in merged_chapters:
                base_id = str(ch.get("id") or "").strip() or "ch"
                cand = base_id
                ctr = 1
                while cand in seen_ids:
                    ctr += 1
                    cand = f"{base_id}_{ctr}"
                ch["id"] = cand
                seen_ids.add(cand)

            # Wave 24 HIGH-6: when --weeks wasn't explicit, scale to
            # max(8, chapter_count) using the actual chapter count we
            # just extracted. Updates project_config so the planner
            # + content generator + trainforge_assessment all see the
            # same autoscaled value.
            if not duration_explicit and merged_chapters:
                auto_weeks = max(8, len(merged_chapters))
                duration_weeks = auto_weeks
                config_data["duration_weeks"] = auto_weeks
                config_path.write_text(
                    json.dumps(config_data, indent=2), encoding="utf-8",
                )

            textbook_structure = {
                "course_name": course_name,
                "source_files": [str(p) for p in html_files],
                "staging_dir": str(staging_dir) if staging_dir else "",
                "chapter_count": len(merged_chapters),
                "duration_weeks": duration_weeks,
                "duration_weeks_autoscaled": bool(
                    not duration_explicit and merged_chapters
                ),
                "chapters": merged_chapters,
                "per_file_results": per_file_results,
                "extraction_errors": extraction_errors,
                "extracted_at": datetime.now().isoformat(),
            }

            # Three-stage textbook synthesis, Wave B / Stage 1 (plan
            # §3.2): when TEXTBOOK_SYNTHESIS_PROVIDER is set, run the
            # large-LLM outline pass and fold its three enrichment keys
            # into textbook_structure BEFORE the JSON write. Default-off
            # (env unset) → byte-identical textbook_structure.json, no
            # new keys. Fail-loud on TextbookSynthesisProviderError per
            # plan §2.2 (Stage 1 is a single call → no per-chapter
            # degradation).
            if os.environ.get("TEXTBOOK_SYNTHESIS_PROVIDER", "").strip():
                from Courseforge.generators._textbook_synthesis_provider import (
                    TextbookSynthesisProvider,
                )
                from lib.decision_capture import DecisionCapture

                synthesis_capture = None
                try:
                    synthesis_capture = DecisionCapture(
                        course_code=course_name,
                        phase="textbook-ingestor",
                        tool="courseforge",
                        streaming=True,
                    )
                except Exception as exc:  # noqa: BLE001 — capture is observability
                    logger.warning(
                        "DecisionCapture init failed in "
                        "objective_extraction Stage-1 synthesis: %s",
                        exc,
                    )

                provider = TextbookSynthesisProvider(capture=synthesis_capture)
                enrichment = provider.synthesize_outline(
                    textbook_structure, course_name=course_name
                )
                textbook_structure["semantic_outline"] = enrichment[
                    "semantic_outline"
                ]
                textbook_structure["draft_terminal_objectives"] = enrichment[
                    "draft_terminal_objectives"
                ]
                textbook_structure["structure_enrichment"] = enrichment[
                    "structure_enrichment"
                ]

            structure_path = (
                project_path / "01_learning_objectives" / "textbook_structure.json"
            )
            structure_path.write_text(
                json.dumps(textbook_structure, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # C6 fix: silent-degradation closure. When `_extract_textbook_structure`
            # raises on N of M files (e.g. malformed HTML, encoding error,
            # SemanticStructureExtractor crash), the surviving (M-N) files
            # would still emit a graph that downstream gates (e.g.
            # ``min_edge_count``) could clear. The phase now fails closed
            # when ``extraction_errors`` is non-empty so an operator sees
            # the partial-extraction class before it propagates.
            extraction_errors_count = len(extraction_errors)
            phase_success = extraction_errors_count == 0
            envelope: Dict[str, Any] = {
                "success": phase_success,
                "project_id": project_id,
                "project_path": str(project_path),
                "textbook_structure_path": str(structure_path),
                "chapter_count": len(merged_chapters),
                "duration_weeks": duration_weeks,
                "duration_weeks_autoscaled": bool(
                    not duration_explicit and merged_chapters
                ),
                "source_file_count": len(html_files),
                "extraction_error_count": extraction_errors_count,
                "extraction_errors_count": extraction_errors_count,
            }
            if not phase_success:
                envelope["error"] = (
                    f"objective_extraction: {extraction_errors_count} of "
                    f"{len(html_files)} HTML files failed structure "
                    f"extraction; downstream phases would consume a "
                    f"partial graph. First {min(3, extraction_errors_count)} "
                    f"errors:"
                )
                envelope["extraction_error_summaries"] = [
                    {
                        "source_file": err.get("source_file", ""),
                        "error": err.get("error", ""),
                    }
                    for err in extraction_errors[:3]
                ]
            return json.dumps(envelope)

        registry["extract_textbook_structure"] = _extract_textbook_structure

        # ============================================================================
        # Wave 24: _plan_course_structure — synthesize TO-NN / CO-NN objectives
        # from the textbook structure (produced by _extract_textbook_structure)
        # and persist them as synthesized_objectives.json. This replaces the
        # pre-Wave-24 course_planning path which only called create_course_project
        # and emitted {COURSE}_OBJ_N placeholders — a scheme disjoint from the
        # TO-NN / CO-NN IDs actually emitted to HTML pages.
        # ============================================================================
        async def _plan_course_structure(**kwargs):
            """Plan course structure: synthesize real LOs + persist.

            Required kwargs: ``project_id`` or (``course_name`` +
            implicit location). When a textbook_structure.json exists in
            the project, chapters and sections drive the synthesizer;
            otherwise we fall back to whatever staged HTML we can find.

            Writes ``{project_path}/01_learning_objectives/synthesized_objectives.json``
            with a canonical shape, populates
            ``project_config.json::objectives_path`` so downstream
            phases pick it up automatically, and returns the real TO/CO
            IDs in ``objective_ids``.
            """
            from MCP.tools import _content_gen_helpers as _cgh

            project_id = kwargs.get("project_id") or ""
            course_name = kwargs.get("course_name") or ""

            # Resolve project path. Prefer explicit project_id; otherwise
            # the most recent export matching course_name.
            project_path: Optional[Path] = None
            if project_id:
                cand = courseforge_exports_dir() / project_id
                if cand.exists():
                    project_path = cand
            if project_path is None and course_name:
                exports_dir = courseforge_exports_dir()
                if exports_dir.exists():
                    matches = sorted(
                        (p for p in exports_dir.iterdir()
                         if p.is_dir() and course_name.lower() in p.name.lower()),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if matches:
                        project_path = matches[0]
            if project_path is None:
                return json.dumps({
                    "error": "plan_course_structure could not locate project directory",
                    "project_id": project_id,
                    "course_name": course_name,
                })
            if not project_id:
                project_id = project_path.name

            # Load project config.
            config_path = project_path / "project_config.json"
            config_data: Dict[str, Any] = {}
            if config_path.exists():
                try:
                    config_data = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    config_data = {}
            # Wave 40: honor the auto-scaled duration_weeks persisted by
            # _extract_textbook_structure. When the CLI didn't receive an
            # explicit --weeks, the extractor already computed max(8, N) and
            # wrote it to config; the stale kwargs value (default 12) must
            # NOT shadow it. duration_weeks_explicit=False => config wins.
            duration_explicit = bool(kwargs.get("duration_weeks_explicit", True))
            if not duration_explicit and config_data.get("duration_weeks"):
                duration_weeks = int(config_data["duration_weeks"])
            else:
                duration_weeks = int(
                    kwargs.get("duration_weeks") or config_data.get("duration_weeks") or 12
                )
            course_name = course_name or config_data.get("course_name") or project_id

            # Prefer real topics from staged HTML when available.
            staging_kwarg = kwargs.get("staging_dir") or config_data.get("staging_dir")
            staging_dir = Path(staging_kwarg) if staging_kwarg else None
            html_files = _cgh.collect_staged_html(staging_dir, COURSEFORGE_INPUTS)
            topics = _cgh.parse_dart_html_files(html_files) if html_files else []

            # If an objectives JSON already exists (supplied by the user),
            # use it verbatim — the planner's job is to surface + persist,
            # not to regenerate over user input.
            supplied_objectives = (
                kwargs.get("objectives_path") or config_data.get("objectives_path")
            )
            supplied_terminal, supplied_chapter = (
                _cgh.load_objectives_json(supplied_objectives)
            )

            # R7 §5.4 — per-chapter Stage-2 failure isolation list.
            # Declared at function scope so it is always defined when
            # the ``synthesized`` dict is assembled below, and is a
            # truthful ``[]`` on the deterministic / user-supplied /
            # COURSEPLANNER paths where Stage-2 never runs (no stale
            # value leaks into synthesized_objectives.json).
            chapter_synthesis_failures: List[str] = []

            if supplied_terminal or supplied_chapter:
                terminal = list(supplied_terminal)
                chapter = list(supplied_chapter)
                mint_method = "user_supplied_objectives_json"
            else:
                # W-D14: COURSEPLANNER_PROVIDER short-circuits the
                # deterministic ``synthesize_objectives_from_topics`` path
                # when the operator opts into the in-process LLM seam for
                # the course-outliner agent. Mirrors the W-D11.A
                # COURSEFORGE_PROVIDER pattern at ``_generate_course_content``.
                # Default unset => deterministic synthesizer fires
                # byte-stable for every existing run.
                _courseplanner_provider_env = os.environ.get(
                    "COURSEPLANNER_PROVIDER", ""
                ).strip()
                terminal: List[Dict[str, Any]] = []
                chapter: List[Dict[str, Any]] = []
                mint_method = ""

                # ================================================== #
                # Stage 2 — three-stage textbook synthesis (plan §5  #
                # + §6). Gated on TEXTBOOK_SYNTHESIS_PROVIDER. Runs   #
                # ABOVE the COURSEPLANNER_PROVIDER branch: when set,  #
                # it dispatches N per-chapter ``synthesize_chapter_   #
                # objectives`` calls (batched ≤10), mints globally-   #
                # sequential CO-NN ids over the FLATTENED course-     #
                # ordered list, then ONE ``reconcile_terminal_        #
                # objectives`` call to adjust the Stage-1 draft TO-NN.#
                # Default unset → deterministic path runs byte-stable.#
                # ================================================== #
                _textbook_synthesis_env = os.environ.get(
                    "TEXTBOOK_SYNTHESIS_PROVIDER", ""
                ).strip()
                if _textbook_synthesis_env:
                    import asyncio as _asyncio
                    from lib.ontology.learning_objectives import (
                        mint_lo_id as _mint_lo_id,
                    )

                    # Read textbook_structure.json off disk for the
                    # Stage-1 draft TO-NN + per-chapter chapter_text
                    # (Wave B folded both into the artifact).
                    _ts_path_local = (
                        project_path
                        / "01_learning_objectives"
                        / "textbook_structure.json"
                    )
                    _ts_structure: Dict[str, Any] = {}
                    if _ts_path_local.exists():
                        try:
                            _ts_structure = json.loads(
                                _ts_path_local.read_text(encoding="utf-8")
                            )
                        except (OSError, ValueError) as exc:
                            logger.warning(
                                "plan_course_structure: Stage-2 "
                                "textbook_structure read failed (%s); "
                                "falling back to deterministic path.",
                                exc,
                            )
                            _ts_structure = {}
                    _ts_chapters: List[Dict[str, Any]] = []
                    if isinstance(_ts_structure, dict):
                        _raw_ch = _ts_structure.get("chapters")
                        if isinstance(_raw_ch, list):
                            _ts_chapters = [
                                c for c in _raw_ch if isinstance(c, dict)
                            ]
                    _draft_tos: List[Dict[str, Any]] = []
                    if isinstance(_ts_structure, dict):
                        _raw_draft = _ts_structure.get(
                            "draft_terminal_objectives"
                        )
                        if isinstance(_raw_draft, list):
                            _draft_tos = [
                                d for d in _raw_draft if isinstance(d, dict)
                            ]

                    if not _ts_chapters:
                        # No chapter surface → Stage 2 cannot run;
                        # fall through to COURSEPLANNER/deterministic.
                        logger.warning(
                            "plan_course_structure: Stage-2 requested "
                            "(TEXTBOOK_SYNTHESIS_PROVIDER=%s) but "
                            "textbook_structure.json carries no "
                            "chapters[]; falling back.",
                            _textbook_synthesis_env,
                        )
                    else:
                        # R7: append into the function-scope
                        # ``chapter_synthesis_failures`` declared above
                        # (no local re-declaration — the list must be
                        # visible at the ``synthesized`` dict assembly).
                        chapters_synthesized = 0
                        _stage2_provider = None
                        try:
                            from Courseforge.generators._textbook_synthesis_provider import (  # noqa: E501
                                TextbookSynthesisProvider,
                                TextbookSynthesisProviderError,
                            )
                            _stage2_capture = None
                            try:
                                from lib.decision_capture import (
                                    DecisionCapture,
                                )
                                _stage2_capture = DecisionCapture(
                                    course_code=course_name,
                                    phase="course-outliner",
                                    tool="courseforge",
                                    streaming=True,
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "DecisionCapture init failed for "
                                    "Stage-2 course-outliner: %s", exc,
                                )
                            _stage2_provider = TextbookSynthesisProvider(
                                capture=_stage2_capture,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "plan_course_structure: Stage-2 provider "
                                "construction failed (%s); falling back "
                                "to deterministic path.", exc,
                            )
                            _stage2_provider = None

                        if _stage2_provider is not None:
                            logger.info(
                                "TEXTBOOK_SYNTHESIS_PROVIDER=%s; routing "
                                "course_planning Stage-2 through the "
                                "per-chapter synthesis provider "
                                "(%d chapters).",
                                _textbook_synthesis_env,
                                len(_ts_chapters),
                            )

                            def _one_chapter_objectives(
                                chapter_dict: Dict[str, Any],
                            ) -> Optional[Dict[str, Any]]:
                                """Synchronous per-chapter Stage-2 call."""
                                cid = str(chapter_dict.get("id") or "")
                                try:
                                    return (
                                        _stage2_provider
                                        .synthesize_chapter_objectives(
                                            chapter_dict,
                                            course_name=course_name,
                                            draft_terminal_objectives=(
                                                _draft_tos
                                            ),
                                        )
                                    )
                                except TextbookSynthesisProviderError as exc:
                                    # Plan §5.4 — per-chapter failure
                                    # isolation: record + continue.
                                    logger.warning(
                                        "plan_course_structure: Stage-2 "
                                        "chapter %r objective call "
                                        "exhausted (%s); degrading per "
                                        "§5.4.", cid, exc,
                                    )
                                    return None
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning(
                                        "plan_course_structure: Stage-2 "
                                        "chapter %r objective call raised "
                                        "(%s); degrading per §5.4.",
                                        cid, exc,
                                    )
                                    return None

                            # Flattened, course-ordered CO list — CO-NN
                            # ids are minted globally-sequential AFTER
                            # the loop, not per-chapter.
                            _flat_cos: List[Dict[str, Any]] = []
                            _loop = _asyncio.get_event_loop()
                            # ``batch_chapters`` is a staticmethod —
                            # access via the instance so a test that
                            # injects a provider resolves the helper.
                            _batches = _stage2_provider.batch_chapters(
                                _ts_chapters
                            )
                            for _batch in _batches:
                                # Dispatch each batch of ≤10 via
                                # run_in_executor, awaiting the whole
                                # batch before the next (plan §5.3).
                                _results = await _asyncio.gather(*[
                                    _loop.run_in_executor(
                                        None, _one_chapter_objectives, ch
                                    )
                                    for ch in _batch
                                ])
                                for _chapter_dict, _res in zip(
                                    _batch, _results
                                ):
                                    _cid = str(
                                        _chapter_dict.get("id") or ""
                                    )
                                    if _res is None:
                                        chapter_synthesis_failures.append(
                                            _cid
                                        )
                                        continue
                                    chapters_synthesized += 1
                                    for _co in (
                                        _res.get("chapter_objectives") or []
                                    ):
                                        if isinstance(_co, dict):
                                            _flat_cos.append(_co)

                            if chapters_synthesized > 0 and _flat_cos:
                                # ≥1 chapter produced objectives — Stage
                                # 2 succeeds (plan §5.4). Mint CO-NN ids
                                # globally-sequential over the flattened
                                # course-ordered list.
                                for _idx, _co in enumerate(
                                    _flat_cos, start=1
                                ):
                                    _co["id"] = _mint_lo_id("chapter", _idx)
                                chapter = _flat_cos

                                # Reconciliation (plan §6) — ONE call to
                                # adjust the Stage-1 draft TO-NN against
                                # the synthesized COs. Fail-loud on
                                # exhaustion (one call, §6).
                                try:
                                    _recon = (
                                        _stage2_provider
                                        .reconcile_terminal_objectives(
                                            _draft_tos,
                                            _flat_cos,
                                            course_name=course_name,
                                        )
                                    )
                                    terminal = list(
                                        _recon.get(
                                            "terminal_objectives"
                                        ) or []
                                    )
                                except TextbookSynthesisProviderError:
                                    logger.exception(
                                        "plan_course_structure: Stage-2 "
                                        "reconciliation exhausted "
                                        "(provider=%s); failing loud.",
                                        _textbook_synthesis_env,
                                    )
                                    raise
                                mint_method = (
                                    f"textbook_synthesis:"
                                    f"{_textbook_synthesis_env}"
                                )
                                logger.info(
                                    "plan_course_structure: Stage-2 "
                                    "synthesized %d CO(s) across %d/%d "
                                    "chapter(s); reconciled to %d TO(s).",
                                    len(_flat_cos),
                                    chapters_synthesized,
                                    len(_ts_chapters),
                                    len(terminal),
                                )
                            else:
                                # ALL chapters failed (plan §5.4): fall
                                # through to the deterministic
                                # synthesizer below; SKIP reconciliation
                                # — there are no LLM-authored COs to
                                # reconcile against.
                                logger.warning(
                                    "plan_course_structure: Stage-2 "
                                    "produced no chapter objectives "
                                    "(all %d chapter call(s) failed); "
                                    "falling back to deterministic "
                                    "synthesizer, reconciliation "
                                    "skipped.", len(_ts_chapters),
                                )

                if (terminal or chapter):
                    pass
                elif _courseplanner_provider_env:
                    try:
                        from Courseforge.generators._outliner_provider import (
                            OutlinerProvider,
                            OutlinerProviderError,
                        )
                        # Build a textbook_structure dict from disk when
                        # the extractor has emitted one; fall back to an
                        # empty stub so the provider's prompt block
                        # surfaces "(none)" without raising. ``structure_path``
                        # below in this function is recomputed; we reference
                        # the same canonical location here so both paths see
                        # the same on-disk artifact.
                        _structure_path_local = (
                            project_path
                            / "01_learning_objectives"
                            / "textbook_structure.json"
                        )
                        outliner_structure: Dict[str, Any] = {}
                        if _structure_path_local.exists():
                            try:
                                outliner_structure = json.loads(
                                    _structure_path_local.read_text(
                                        encoding="utf-8"
                                    )
                                )
                            except (OSError, ValueError) as exc:
                                logger.warning(
                                    "plan_course_structure: "
                                    "textbook_structure read failed (%s); "
                                    "calling OutlinerProvider with empty "
                                    "chapters[]", exc,
                                )
                        outliner_capture = None
                        try:
                            from lib.decision_capture import DecisionCapture
                            outliner_capture = DecisionCapture(
                                course_code=course_name,
                                phase="course-outliner",
                                tool="courseforge",
                                streaming=True,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "DecisionCapture init failed for "
                                "course-outliner: %s", exc,
                            )
                            outliner_capture = None
                        outliner_provider = OutlinerProvider(
                            capture=outliner_capture,
                        )
                        logger.info(
                            "COURSEPLANNER_PROVIDER=%s; routing "
                            "course-outliner through in-process provider.",
                            _courseplanner_provider_env,
                        )
                        synthesised = outliner_provider.synthesize_objectives(
                            outliner_structure,
                            course_name=course_name,
                            chapter_count=len(
                                outliner_structure.get("chapters") or []
                            ),
                            weeks=duration_weeks,
                        )
                        terminal = list(
                            synthesised.get("terminal_objectives") or []
                        )
                        # Provider returns chapter_objectives in the
                        # group-of-groups shape; flatten so this
                        # function's downstream assembly stays consistent
                        # with both the user-supplied and deterministic
                        # paths (which deliver flat dicts).
                        chapter_groups = (
                            synthesised.get("chapter_objectives") or []
                        )
                        for grp in chapter_groups:
                            if not isinstance(grp, dict):
                                continue
                            for entry in grp.get("objectives") or []:
                                if isinstance(entry, dict):
                                    chapter.append(entry)
                        mint_method = synthesised.get(
                            "mint_method"
                        ) or f"courseplanner_provider:{_courseplanner_provider_env}"
                    except OutlinerProviderError as exc:
                        # LLM tier exhausted parse — fail loud rather
                        # than silently falling back so the operator's
                        # opt-in intent is honored or surfaced clearly.
                        logger.exception(
                            "OutlinerProvider exhausted (provider=%s): %s",
                            _courseplanner_provider_env, exc,
                        )
                        raise
                    except Exception as exc:
                        logger.exception(
                            "OutlinerProvider init/dispatch failed "
                            "(provider=%s): %s",
                            _courseplanner_provider_env, exc,
                        )
                        raise
                if not terminal and not chapter:
                    terminal, chapter = _cgh.synthesize_objectives_from_topics(
                        topics, duration_weeks,
                    )
                    mint_method = mint_method or (
                        "synthesize_objectives_from_topics"
                    )

            # Wave 1.8 — objective-driven dynamic week count. The
            # extractor phase auto-scaled ``duration_weeks`` to
            # ``max(8, len(merged_chapters))`` based on chapter count
            # (a structural proxy). After the synthesizer emits real
            # CO-NN objectives we have the authoritative signal — the
            # actual pedagogical surface area the course must teach.
            # Re-scale to ``max(8, ceil(len(chapter) / _COS_PER_WEEK))``
            # so a textbook with 6 chapters but 30 chapter objectives
            # paces at 15 weeks (2 COs/week) instead of 8.
            #
            # ``_COS_PER_WEEK = 2`` is the typical CC course pace
            # (calibrated against the RDF/SHACL calibration corpus: 30 COs
            # / 15 weeks). Override via the ``WAVE18_COS_PER_WEEK``
            # env var when calibrating against a different course
            # family. ``not duration_explicit`` preserves operator
            # intent — when ``--weeks N`` is passed, the synthesizer
            # respects it regardless of objective count.
            #
            # Skipped for ``user_supplied_objectives_json``: the
            # operator's hand-curated LO list pre-encodes pacing
            # decisions that automatic re-scaling would clobber.
            if not duration_explicit and mint_method != "user_supplied_objectives_json":
                _COS_PER_WEEK = int(
                    os.environ.get("WAVE18_COS_PER_WEEK", "2") or "2"
                )
                if _COS_PER_WEEK > 0 and chapter:
                    objective_weeks = max(
                        8,
                        (len(chapter) + _COS_PER_WEEK - 1) // _COS_PER_WEEK,
                    )
                    if objective_weeks != duration_weeks:
                        logger.info(
                            "plan_course_structure: re-scaling "
                            "duration_weeks from %d (chapter-driven) to "
                            "%d (objective-driven; %d COs / %d per week)",
                            duration_weeks,
                            objective_weeks,
                            len(chapter),
                            _COS_PER_WEEK,
                        )
                        duration_weeks = objective_weeks

            # Detect textbook_structure_path to record provenance.
            structure_path = (
                project_path / "01_learning_objectives" / "textbook_structure.json"
            )
            generated_from = str(structure_path) if structure_path.exists() else ""

            # Canonical on-disk shape.
            #
            # Phase 6 ST 7: pass through the optional ``abcd`` sub-object
            # when the source LO carries one. Both the synthesizer
            # (``_cgh.synthesize_objectives_from_topics`` after Phase 6
            # ST 8 — emits ABCD for every LO with a Bloom level) and a
            # user-supplied ``--reuse-objectives`` payload (preserved by
            # ``_cgh._normalize_objective_entry`` after the Phase 6 ST 7
            # widening) attach ``abcd`` per LO. The ABCD dict is
            # deep-copied so downstream mutation of ``lo_entries`` doesn't
            # leak back into the in-memory terminal/chapter lists that
            # the ``terminal_objectives`` / ``chapter_objectives`` blocks
            # below also serialize. Cross-link: ``$defs.AbcdObjective``
            # in ``schemas/knowledge/courseforge_jsonld_v1.schema.json``;
            # validated downstream by ``AbcdObjectiveValidator``
            # (Phase 6 ST 4) wired as the ``abcd_verb_alignment`` gate
            # on the ``course_planning`` phase (Phase 6 ST 4.5).
            def _clone_lo(src: Dict[str, Any], hierarchy: str) -> Dict[str, Any]:
                cloned = dict(src)
                abcd_payload = src.get("abcd")
                if isinstance(abcd_payload, dict):
                    # Deep-copy the ABCD sub-object so the per-LO entry
                    # in ``lo_entries`` is independent of the
                    # ``terminal_objectives`` / ``chapter_objectives``
                    # serialised view below. Behavior is a nested dict,
                    # so a one-level deep copy is sufficient.
                    behavior = abcd_payload.get("behavior")
                    cloned_abcd: Dict[str, Any] = dict(abcd_payload)
                    if isinstance(behavior, dict):
                        cloned_abcd["behavior"] = dict(behavior)
                    cloned["abcd"] = cloned_abcd
                cloned["hierarchy_level"] = hierarchy
                return cloned

            lo_entries: List[Dict[str, Any]] = []
            for to in terminal:
                lo_entries.append(_clone_lo(to, "terminal"))
            for co in chapter:
                lo_entries.append(_clone_lo(co, "chapter"))

            # Phase-ordering fix (Option A1): the concept-objective linker
            # pass (lib/ontology/concept_objective_linker.py) that
            # populated ``LearningObjective.keyConcepts[]`` from the
            # concept graph USED to run here, gated on a
            # ``concept_graph_path`` kwarg. It has been relocated into
            # ``_run_concept_extraction`` because that phase now runs AFTER
            # course_planning — the concept graph does not exist yet at
            # planning time, so a linker pass here would always no-op on a
            # fresh run. A ``concept_graph_path`` kwarg passed to this
            # function is now inert (preserved for back-compat callers).

            # Layer A — terminal-coverage guarantee. Prune any terminal
            # objective with ZERO chapter objectives rolling up to it
            # (CO-bearing courses only). An orphan terminal is one that
            # downstream content generation cannot cover, so it trips the
            # critical UNCOVERED_TERMINAL_OUTCOME archival gate
            # (lib/validators/libv2/packet_integrity.py). Pruning by
            # construction here makes the objectives set internally
            # consistent before it is ever written to disk; the
            # course_planning ``terminal_objective_coverage`` gate is the
            # fail-fast backstop. CO-less courses (OpenStax / gui-design)
            # are returned untouched — their terminals are self-contained
            # teaching units whose coverage is content-driven, so they are
            # adjudicated by the gate (with the textbook structure), never
            # silently pruned here. Cross-link:
            # lib/ontology/terminal_coverage.py::prune_orphan_terminals.
            try:
                from lib.ontology.terminal_coverage import (
                    flatten_chapter_objectives as _flatten_cos,
                    prune_orphan_terminals as _prune_orphan_terminals,
                )

                _chapter_groups_for_prune = [
                    {"chapter": f"Week {idx}", "objectives": [dict(c)]}
                    for idx, c in enumerate(chapter, start=1)
                ]
                _kept_terminals, _pruned_terminal_ids = (
                    _prune_orphan_terminals(
                        terminal, _chapter_groups_for_prune,
                    )
                )
                if _pruned_terminal_ids:
                    logger.warning(
                        "Layer A: pruning %d orphan terminal(s) with no "
                        "rolling-up chapter objective from %s: %s",
                        len(_pruned_terminal_ids),
                        course_name,
                        ", ".join(_pruned_terminal_ids),
                    )
                    _pruned_set = {str(t) for t in _pruned_terminal_ids}
                    terminal = [
                        t for t in terminal
                        if str((t or {}).get("id")) not in _pruned_set
                    ]
                    # Keep lo_entries consistent with the pruned terminal
                    # set so the on-disk learning_outcomes array does not
                    # re-introduce the orphan IDs.
                    lo_entries = [
                        e for e in lo_entries
                        if not (
                            (e.get("hierarchy_level") == "terminal")
                            and str(e.get("id")) in _pruned_set
                        )
                    ]
            except Exception as exc:  # noqa: BLE001 — best-effort guarantee
                logger.warning(
                    "Layer A: terminal-coverage prune failed (%s); "
                    "proceeding without prune (the course_planning "
                    "terminal_objective_coverage gate remains the "
                    "fail-fast backstop).",
                    exc,
                )

            # Layer A (CO-less branch) — stamp a resolvable ``chapter``
            # back-pointer on every terminal of a CO-less course (empty
            # chapter_objectives). The deterministic synthesizer mints
            # terminals with no chapter back-pointer (the topic-side parser
            # frequently sees chapter_id=None even though the structure-side
            # extractor synthesized real ch1/ch2/... chapters), which leaves
            # the objectives set internally inconsistent: the CO-less
            # ``terminal_objective_coverage`` gate then critical-fails with
            # ORPHAN_TERMINAL_NO_CHAPTER_REF. Anchoring each terminal to a
            # real structure chapter here (round-robin, document order) makes
            # the set consistent BEFORE it is written — the same Layer-A
            # guarantee the prune provides for CO-bearing courses. No-op for
            # CO-bearing courses and for runs with no resolvable structure
            # chapters (the gate's TERMINAL_COVERAGE_UNVERIFIED path owns
            # that). Cross-link:
            # lib/ontology/terminal_coverage.py::attach_terminal_chapter_refs.
            try:
                from lib.ontology.terminal_coverage import (
                    attach_terminal_chapter_refs as _attach_terminal_chapter_refs,
                )

                _ts_for_attach: Dict[str, Any] = {}
                if structure_path.exists():
                    try:
                        _ts_for_attach = json.loads(
                            structure_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError) as exc:
                        logger.warning(
                            "Layer A (CO-less): textbook_structure read "
                            "failed (%s); skipping chapter back-pointer "
                            "attach.", exc,
                        )
                        _ts_for_attach = {}
                _chapter_groups_for_attach = [
                    {"chapter": f"Week {idx}", "objectives": [dict(c)]}
                    for idx, c in enumerate(chapter, start=1)
                ]
                _attached_terminals, _attached_ids = (
                    _attach_terminal_chapter_refs(
                        terminal,
                        _chapter_groups_for_attach,
                        textbook_structure=_ts_for_attach,
                    )
                )
                if _attached_ids:
                    logger.info(
                        "Layer A (CO-less): attached chapter back-pointer "
                        "to %d terminal(s) of %s: %s",
                        len(_attached_ids), course_name,
                        ", ".join(_attached_ids),
                    )
                    terminal = _attached_terminals
                    # Re-sync the terminal entries in lo_entries with the
                    # freshly-stamped ``chapter`` so the on-disk
                    # learning_outcomes array carries the back-pointer too.
                    _by_id = {str(t.get("id")): t for t in terminal}
                    for _e in lo_entries:
                        if _e.get("hierarchy_level") != "terminal":
                            continue
                        _src = _by_id.get(str(_e.get("id")))
                        if _src is not None and "chapter" in _src:
                            _e["chapter"] = _src["chapter"]
            except Exception as exc:  # noqa: BLE001 — best-effort guarantee
                logger.warning(
                    "Layer A (CO-less): terminal chapter back-pointer "
                    "attach failed (%s); proceeding (the course_planning "
                    "terminal_objective_coverage gate remains the "
                    "fail-fast backstop).",
                    exc,
                )

            synthesized = {
                "course_name": course_name,
                "generated_from": generated_from,
                "mint_method": mint_method,
                "duration_weeks": duration_weeks,
                "learning_outcomes": lo_entries,
                # Preserve the split-by-hierarchy shape the content
                # generator + CourseProcessor's load_objectives expect.
                "terminal_objectives": [dict(t) for t in terminal],
                "chapter_objectives": [{
                    "chapter": f"Week {idx}",
                    "objectives": [dict(c)],
                } for idx, c in enumerate(chapter, start=1)],
                # R7 §5.4 — per-chapter Stage-2 failure isolation.
                # Persisted so ChapterObjectiveCoverageValidator's
                # file-fallback (objectives.get("chapter_synthesis_
                # failures")) resolves and the expected-failure
                # cross-check is no longer dead code. Empty [] on the
                # deterministic / user-supplied / COURSEPLANNER paths.
                "chapter_synthesis_failures": list(
                    chapter_synthesis_failures
                ),
                "synthesized_at": datetime.now().isoformat(),
            }
            objectives_out_path = (
                project_path / "01_learning_objectives" / "synthesized_objectives.json"
            )
            objectives_out_path.write_text(
                json.dumps(synthesized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Thread the path back into project_config so
            # _generate_course_content + Trainforge's CourseProcessor
            # (_invoke_trainforge) pick it up automatically.
            config_data["objectives_path"] = str(objectives_out_path)
            config_data["synthesized_objectives_path"] = str(objectives_out_path)
            config_data["course_name"] = course_name
            config_data["duration_weeks"] = duration_weeks
            config_data["project_id"] = project_id
            config_data["status"] = "planned"
            config_path.write_text(
                json.dumps(config_data, indent=2), encoding="utf-8",
            )

            # Wave2-I7 (dispatch-7 Finding 8): build ``objective_ids``
            # as a canonical ``List[str]`` so the
            # ``trainforge_assessment.inputs_from`` route resolves to a
            # non-None list when the workflow resumes from this phase's
            # checkpoint. The runner's ``_route_params`` comma-joins
            # list values on the wire (workflow_runner.py:1687-1688) so
            # downstream ``_generate_assessments`` still sees the
            # comma-string shape it already handles. Canonical order:
            # terminals first, then chapter LOs (matches the
            # ``lo_entries`` assembly above and the on-disk
            # ``learning_outcomes`` array). Defensive: read all three
            # supported ``chapter_objectives`` shapes via the
            # module-level normalizer
            # ``_normalize_chapter_objectives_to_groups`` (Wave2b —
            # adds the OpenStax dict-of-lists shape on top of the
            # legacy list-of-groups + flat-list shapes). Terminals are
            # read from either ``terminal_objectives`` or the LibV2
            # ``terminal_outcomes`` alias.
            def _collect_lo_ids(payload: Dict[str, Any]) -> List[str]:
                ids: List[str] = []
                terminals = (
                    payload.get("terminal_objectives")
                    or payload.get("terminal_outcomes")
                    or []
                )
                for entry in terminals:
                    if isinstance(entry, dict) and entry.get("id"):
                        ids.append(str(entry["id"]))
                chapters_raw = (
                    payload.get("chapter_objectives")
                    or payload.get("component_objectives")
                    or []
                )
                for group in _normalize_chapter_objectives_to_groups(
                    chapters_raw,
                ):
                    for entry in group.get("objectives") or []:
                        if isinstance(entry, dict) and entry.get("id"):
                            ids.append(str(entry["id"]))
                return ids

            objective_ids: List[str] = _collect_lo_ids(synthesized)
            # Fallback: if the synthesized payload's id-bearing dicts
            # were stripped en route, fall back to ``lo_entries`` (same
            # canonical order: terminal-first).
            if not objective_ids:
                objective_ids = [
                    str(e["id"]) for e in lo_entries if e.get("id")
                ]

            # Wave3-Anew3 (Finding F3): back-fill the DART chunkset's
            # ``learning_outcome_refs`` against the just-minted objective_ids.
            # ``_run_dart_chunking`` ran BEFORE this phase, so its chunks
            # carry empty LO refs by construction. Text-scan + allowlist
            # filter populates the field deterministically; existing
            # non-empty refs (legacy corpora) are preserved via union
            # semantics. Best-effort — failure logs a warning, does not
            # block the phase output.
            course_slug = (course_name or "").lower().replace("_", "-").replace(" ", "-")
            backfill_summary: Dict[str, Any] = {}
            if course_slug and objective_ids:
                try:
                    backfill_summary = _backfill_dart_chunk_lo_refs(
                        course_slug=course_slug,
                        objective_ids=objective_ids,
                        libv2_root=kwargs.get("libv2_root"),
                    )
                    if backfill_summary.get("chunks_updated"):
                        logger.info(
                            "Wave3-Anew3: back-filled learning_outcome_refs on "
                            "%d/%d DART chunks (+%d refs total) at %s",
                            backfill_summary["chunks_updated"],
                            backfill_summary["chunks_scanned"],
                            backfill_summary["new_refs_total"],
                            backfill_summary["chunks_path"],
                        )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "Wave3-Anew3: DART chunk LO-ref back-fill failed "
                        "(%s); proceeding with empty refs.",
                        exc,
                    )

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "project_path": str(project_path),
                "synthesized_objectives_path": str(objectives_out_path),
                "objective_ids": objective_ids,
                "terminal_count": len(terminal),
                "chapter_count": len(chapter),
                "mint_method": mint_method,
                "duration_weeks": duration_weeks,
                "dart_chunk_lo_backfill": backfill_summary,
            })

        registry["plan_course_structure"] = _plan_course_structure

        # ============================================================================
        # BLOCK: Worker α edits ONLY below this line through the next END marker.
        # Scope: _generate_course_content replacement. See plans/pipeline-execution-
        # fixes/contracts.md § "Courseforge content-generator contract".
        # ============================================================================
        async def _generate_course_content(**kwargs):
            """Generate 5-page weekly course modules from DART outputs + objectives.

            Replaces the legacy single-page stub with a full Courseforge
            emission: overview, content, application, self_check, and
            summary pages per week. Every emitted page carries the full
            ``data-cf-*`` attribute surface and a JSON-LD
            ``CourseModule`` body that validates against
            ``schemas/knowledge/courseforge_jsonld_v1.schema.json``.

            Delegates the actual HTML rendering to
            ``Courseforge.scripts.generate_course.generate_week`` (the
            mature multi-file emitter) — this wrapper only adapts the
            pipeline's kwargs into the ``week_data`` payload that the
            emitter consumes, plus forwards the Wave 9 source-routing
            map when one is present on disk.
            """
            from Courseforge.scripts import generate_course as _gen
            from MCP.tools import _content_gen_helpers as _cgh

            project_id = kwargs.get("project_id", "")
            if not project_id:
                return json.dumps({"error": "generate_course_content requires project_id"})

            project_path = courseforge_exports_dir() / project_id
            content_dir = project_path / "03_content_development"
            content_dir.mkdir(parents=True, exist_ok=True)

            config_path = project_path / "project_config.json"
            if not config_path.exists():
                return json.dumps({"error": f"Project config not found: {config_path}"})
            with open(config_path) as f:
                config = json.load(f)

            course_code = config.get("course_name") or project_id
            # Wave 40: honor the auto-scaled duration_weeks persisted by
            # _extract_textbook_structure. Config is authoritative when the
            # CLI's --weeks wasn't explicit; only a truly explicit kwarg may
            # override the value the extractor committed to disk.
            duration_explicit = bool(kwargs.get("duration_weeks_explicit", False))
            kwarg_duration = kwargs.get("duration_weeks")
            if duration_explicit and kwarg_duration:
                duration_weeks = int(kwarg_duration)
            else:
                duration_weeks = int(config.get("duration_weeks") or kwarg_duration or 12)
            objectives_path = config.get("objectives_path") or kwargs.get("objectives_path")

            # ---------------------------------------------------------- #
            # Staged DART HTML — prefer the staging_dir passed by the    #
            # workflow runner; fall back to the most-recent staging run. #
            # ---------------------------------------------------------- #
            staging_kwarg = kwargs.get("staging_dir")
            staging_dir = Path(staging_kwarg) if staging_kwarg else None
            html_files = _cgh.collect_staged_html(staging_dir, COURSEFORGE_INPUTS)
            topics = _cgh.parse_dart_html_files(html_files)

            # ---------------------------------------------------------- #
            # Objectives: honor supplied JSON; synthesize from DART otherwise.
            # ---------------------------------------------------------- #
            terminal_objectives, chapter_objectives = _cgh.load_objectives_json(
                objectives_path
            )
            if not terminal_objectives and not chapter_objectives:
                terminal_objectives, chapter_objectives = (
                    _cgh.synthesize_objectives_from_topics(topics, duration_weeks)
                )

            all_objectives = list(terminal_objectives) + list(chapter_objectives)
            topics_by_week = _cgh._group_topics_by_week(topics, duration_weeks)

            # ---------------------------------------------------------- #
            # Source-routing map (Wave 9). Empty dict or missing file =>  #
            # backward-compat path: pages emit without sourceReferences.  #
            # ---------------------------------------------------------- #
            source_module_map: Dict[str, Any] = {}
            map_path_kwarg = kwargs.get("source_module_map_path")
            if map_path_kwarg:
                map_path = Path(map_path_kwarg)
            else:
                map_path = project_path / "source_module_map.json"
            if map_path.exists():
                try:
                    source_module_map = json.loads(
                        map_path.read_text(encoding="utf-8")
                    ) or {}
                except (OSError, ValueError):
                    source_module_map = {}

            # Wave 2 prerequisite map: each page prerequisites the prior
            # page in the 5-page week sequence.
            prerequisite_map: Dict[str, list] = {}
            for week_num in range(1, duration_weeks + 1):
                w = f"{week_num:02d}"
                prerequisite_map[f"week_{w}_application"] = [
                    f"week_{w}_overview"
                ]
                prerequisite_map[f"week_{w}_self_check"] = [
                    f"week_{w}_application"
                ]
                prerequisite_map[f"week_{w}_summary"] = [
                    f"week_{w}_self_check"
                ]

            # ---------------------------------------------------------- #
            # Decision capture — content-generator phase.                 #
            # ---------------------------------------------------------- #
            # Phase 1 ToS unblock: COURSEFORGE_PROVIDER env (anthropic /
            # together / local) opens an in-process LLM seam alongside
            # the Wave-74 subagent dispatch path. Default unset =>
            # deterministic DART-paragraph synthesis (legacy behavior).
            _courseforge_provider_env = os.environ.get(
                "COURSEFORGE_PROVIDER", ""
            ).strip()

            capture = None
            try:
                from lib.decision_capture import DecisionCapture
                capture = DecisionCapture(
                    course_code=course_code,
                    phase="content-generator",
                    tool="courseforge",
                    streaming=True,
                )
                capture.log_decision(
                    decision_type="content_structure",
                    decision=(
                        f"Emit 5-page weekly modules (overview, content, "
                        f"application, self_check, summary) for "
                        f"{duration_weeks} weeks via Courseforge generate_week."
                    ),
                    rationale=(
                        "The 5-page structure matches the Courseforge "
                        "pipeline contract (plans/pipeline-execution-fixes/"
                        "contracts.md) and ensures each weekly module "
                        "validates under the page_objectives + "
                        "content_structure gates."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DecisionCapture init failed in content-generator: %s", exc
                )
                capture = None

            # Instantiate the in-process LLM provider when the env var
            # is set. We fail-loud here (e.g. missing ANTHROPIC_API_KEY)
            # rather than silently falling back to deterministic so the
            # operator's intent is honored or surfaced clearly.
            content_provider = None
            if _courseforge_provider_env:
                try:
                    from Courseforge.generators._provider import (
                        ContentGeneratorProvider,
                    )
                    content_provider = ContentGeneratorProvider(
                        capture=capture,
                    )
                    logger.info(
                        "COURSEFORGE_PROVIDER=%s; routing content-generator "
                        "through in-process provider.",
                        _courseforge_provider_env,
                    )
                except Exception as exc:
                    logger.exception(
                        "ContentGeneratorProvider init failed (provider=%s): %s",
                        _courseforge_provider_env,
                        exc,
                    )
                    raise

            # Phase 3 Subtask 61: when ``COURSEFORGE_TWO_PASS=true`` is
            # set, instantiate a :class:`CourseforgeRouter` and pass it
            # through to :func:`build_week_data` so the per-page emit
            # dispatches via the two-pass outline → inter-tier validate
            # → rewrite pipeline instead of the legacy single-pass
            # ``ContentGeneratorProvider``. The router is fully additive
            # — when the flag is off, ``content_router`` stays ``None``
            # and the legacy ``content_provider`` path runs unchanged
            # (preserves Phase 1 byte-stable behavior). Construction
            # mirrors the ``content_provider`` block above: load the
            # YAML policy + DecisionCapture once at the call site, fail
            # loud on init exceptions so an operator's opt-in intent is
            # surfaced rather than silently downgraded to legacy.
            content_router = None
            if _courseforge_two_pass_enabled():
                try:
                    from Courseforge.router.router import CourseforgeRouter
                    from Courseforge.router.policy import (
                        load_block_routing_policy,
                    )
                    content_router = CourseforgeRouter(
                        capture=capture,
                        policy=load_block_routing_policy(),
                    )
                    logger.info(
                        "COURSEFORGE_TWO_PASS=true; routing "
                        "content-generator through two-pass "
                        "CourseforgeRouter (outline → validate → rewrite).",
                    )
                except Exception as exc:
                    logger.exception(
                        "CourseforgeRouter init failed under "
                        "COURSEFORGE_TWO_PASS=true: %s",
                        exc,
                    )
                    raise

            # ---------------------------------------------------------- #
            # Emit each week via generate_week.                           #
            # ---------------------------------------------------------- #
            generated_files: list = []
            weeks_prepared = 0
            # Anti-silent-template guard: per-page authorship tally folded
            # into the provenance below so a provider/router that degrades
            # to the template floor at runtime is caught (not just the
            # construct-time generator_mode).
            _authorship_stats: Dict[str, int] = {
                "llm_authored": 0, "template_fallback": 0,
            }
            for week_num in range(1, duration_weeks + 1):
                week_topics = (
                    topics_by_week[week_num - 1]
                    if (week_num - 1) < len(topics_by_week)
                    else []
                )
                # Per-week LO set: scope to this week's terminals + at most
                # two chapter objectives round-robin assigned by week.
                # Earlier revisions prepended ALL terminal_objectives to
                # every week, which over-connected the derived-from-
                # objective edges in the KG (O(N*D) instead of O(N)).
                # Now: each week gets only the terminal slice round-robin
                # assigned to it.
                week_chapter_cos = []
                if chapter_objectives:
                    step = max(1, len(chapter_objectives) // max(1, duration_weeks))
                    start = (week_num - 1) * step
                    week_chapter_cos = list(
                        chapter_objectives[start:start + step + 1]
                    )[:2] or [chapter_objectives[(week_num - 1) % len(chapter_objectives)]]

                # Scope terminals per week. With N terminals and D weeks,
                # each week claims ceil(N/D) terminals in source order.
                week_terminals: list = []
                if terminal_objectives:
                    t_step = max(
                        1,
                        (len(terminal_objectives) + duration_weeks - 1) // duration_weeks,
                    )
                    t_start = (week_num - 1) * t_step
                    week_terminals = list(
                        terminal_objectives[t_start:t_start + t_step]
                    )
                    # Guarantee at least one terminal per week when corpus
                    # has any terminals at all — round-robin fallback.
                    if not week_terminals:
                        week_terminals = [
                            terminal_objectives[(week_num - 1) % len(terminal_objectives)]
                        ]

                week_objectives = list(week_terminals) + week_chapter_cos
                seen: set = set()
                week_objectives_deduped = []
                for o in week_objectives:
                    if o["id"] in seen:
                        continue
                    seen.add(o["id"])
                    week_objectives_deduped.append(o)

                week_data = _cgh.build_week_data(
                    week_num=week_num,
                    duration_weeks=duration_weeks,
                    week_topics=week_topics,
                    week_objectives=week_objectives_deduped,
                    all_objectives=all_objectives,
                    course_code=course_code,
                    content_provider=content_provider,
                    content_router=content_router,
                    authorship_stats=_authorship_stats,
                )

                try:
                    count, files = _gen.generate_week(
                        week_data,
                        content_dir,
                        course_code,
                        canonical_objectives=None,  # week_data already has canonical ids
                        classification=None,
                        prerequisite_map=prerequisite_map,
                        source_module_map=source_module_map or None,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "generate_week failed for week %d: %s", week_num, exc
                    )
                    continue

                weeks_prepared += 1
                week_dir = content_dir / f"week_{week_num:02d}"
                for name in files:
                    page_path = week_dir / name
                    # Post-process: ensure every page carries an
                    # objectives <section>. Overview already has one
                    # from generate_week; the other four pages don't by
                    # default but the page_objectives gate + integration
                    # test require the data-cf-objective-id attribute on
                    # every page.
                    try:
                        body = page_path.read_text(encoding="utf-8")
                        updated = _cgh.ensure_objectives_on_page(
                            body, week_objectives_deduped,
                        )
                        if updated != body:
                            page_path.write_text(updated, encoding="utf-8")
                    except OSError as exc:
                        logger.warning(
                            "Failed to post-process %s: %s", page_path, exc,
                        )
                    generated_files.append(str(page_path))

                if capture is not None:
                    try:
                        source_stems = sorted({
                            t.get("source_file", "") for t in week_topics
                            if t.get("source_file")
                        })
                        primary_heading = (
                            week_topics[0]["heading"] if week_topics else "synthetic"
                        )
                        capture.log_decision(
                            decision_type="source_selection",
                            decision=(
                                f"Week {week_num}: ground content on "
                                f"{primary_heading!r} from sources "
                                f"{source_stems or ['(no DART staging found)']}."
                            ),
                            rationale=(
                                "Selected DART-derived topics whose parsed "
                                "headings align with the week's chapter "
                                "objectives; synthesized placeholder content "
                                "only when no DART topics were available."
                            ),
                        )
                    except Exception:  # noqa: BLE001
                        # Never let decision capture crash emission.
                        pass

            # Wave 32 Deliverable C: fail the phase when every
            # generated page is an empty template skeleton. Pre-Wave-32
            # ``content_generation`` silently passed even when the
            # dispatcher returned zero actual body content — each page
            # carried only ``<h1>Week N</h1><h2>Overview</h2>`` with no
            # paragraphs ≥ 30 words. The counts showed 12/12 complete
            # and gates rubber-stamped it. This check reuses the same
            # ``NON_TRIVIAL_WORD_FLOOR`` (30) as the Wave 31
            # ContentGroundingValidator for behavioural consistency:
            # parse every emitted page, count body words in
            # ``<p>/<li>/<blockquote>/<figcaption>`` within ``<main>``
            # (or the document body when no main wrapper is present),
            # and fail the phase when zero pages clear the floor.
            empty_error = _check_content_nonempty(generated_files)
            if empty_error is not None:
                return json.dumps({
                    "success": False,
                    "error_code": "CONTENT_GENERATION_EMPTY",
                    "error": empty_error,
                    "project_id": project_id,
                    "page_paths": generated_files,
                    "content_dir": str(content_dir),
                    "weeks_prepared": weeks_prepared,
                })

            # Wave 32 Deliverable B: surface page_paths + content_dir so
            # downstream gate input routing picks them up. Pre-Wave-32
            # ``content_paths`` landed as a plain list in phase_outputs,
            # but the router's builders inspect ``content_paths`` only
            # when it's a comma-joined ``str`` and otherwise flag
            # ``page_paths`` / ``content_dir`` as missing — every live
            # re-sim showed ``content_grounding`` + ``page_objectives``
            # silently skipping with ``missing inputs: *``. The fix is
            # purely on the emit side: surface the list as
            # ``page_paths`` (the router's canonical key) and also
            # surface ``content_paths`` as a comma-joined str for the
            # legacy parsers (_all_html_paths, _find_content_dir).
            # -------------------------------------------------------- #
            # Generation provenance (anti-silent-template guard).      #
            # The deterministic ``generate_week`` template emitter must #
            # NEVER silently stand in for real LLM content authoring in #
            # a production run. Record which path actually ran so the   #
            # ContentAuthorshipValidator gate can BLOCK a template-only #
            # result when LLM authoring was the operator's intent — and #
            # warn loudly here regardless. ``generate_course_content``  #
            # stamps its own honest provenance, so an operator who runs #
            # the template tool out-of-band (e.g. to service a mailbox  #
            # agent_task) cannot launder it as LLM-authored content.    #
            # -------------------------------------------------------- #
            if content_router is not None:
                generator_mode = "two_pass_router"
            elif content_provider is not None:
                generator_mode = "llm_provider"
            else:
                generator_mode = "template_deterministic"

            # Runtime degradation: a constructed provider/router that fell
            # back to the deterministic template floor on EVERY page authored
            # zero real LLM content — treat it as a template run regardless of
            # construct-time mode. Partial degradation is surfaced as a flag.
            _llm_authored = _authorship_stats.get("llm_authored", 0)
            _template_fallback = _authorship_stats.get("template_fallback", 0)
            llm_authoring_degraded = (
                generator_mode != "template_deterministic"
                and _template_fallback > 0
            )
            if (
                generator_mode != "template_deterministic"
                and _llm_authored == 0
                and _template_fallback > 0
            ):
                # Provider/router was wired but authored nothing — this is a
                # template run in disguise.
                generator_mode = "template_deterministic"

            template_fired = generator_mode == "template_deterministic"

            def _envflag(name: str) -> bool:
                return os.environ.get(name, "").strip().lower() in (
                    "1", "true", "yes", "on"
                )

            agent_dispatch_enabled = _envflag("ED4ALL_AGENT_DISPATCH")
            allow_template = _envflag("COURSEFORGE_ALLOW_TEMPLATE_EMITTER")
            llm_authoring_intended = bool(
                _courseforge_provider_env
                or _courseforge_two_pass_enabled()
                or agent_dispatch_enabled
            )

            if template_fired:
                _msg = (
                    "TEMPLATE EMITTER FIRED for content_generation "
                    f"(course={course_code}, weeks={weeks_prepared}): the "
                    "deterministic generate_week template produced these "
                    "pages — NOT a real LLM content-generator. This should "
                    "not happen in a real run. Configure COURSEFORGE_PROVIDER "
                    "(anthropic/together/local), COURSEFORGE_TWO_PASS=true, or "
                    "service the content-generator mailbox tasks with real "
                    f"subagents. (llm_authored_pages={_llm_authored}, "
                    f"template_fallback_pages={_template_fallback})"
                )
                if llm_authoring_intended and not allow_template:
                    logger.error(
                        "%s LLM authoring was intended (provider=%r "
                        "two_pass=%s agent_dispatch=%s); the content_authorship "
                        "gate will BLOCK this run.",
                        _msg, _courseforge_provider_env,
                        _courseforge_two_pass_enabled(), agent_dispatch_enabled,
                    )
                else:
                    logger.warning("%s", _msg)
            elif llm_authoring_degraded:
                # Provider/router authored SOME pages but silently fell back
                # to the template floor on others — flag loudly even though
                # the gate won't block a partially-authored run.
                logger.warning(
                    "content_generation PARTIAL TEMPLATE FALLBACK "
                    "(course=%s): %d page(s) authored by the LLM "
                    "provider/router, %d page(s) silently degraded to the "
                    "deterministic template floor. Inspect provider health "
                    "(network / budget / parse misses).",
                    course_code, _llm_authored, _template_fallback,
                )

            generation_provenance = {
                "schema_version": "1.1",
                "generator_mode": generator_mode,
                "template_fallback_fired": template_fired,
                "llm_authored_pages": _llm_authored,
                "template_fallback_pages": _template_fallback,
                "llm_authoring_degraded": llm_authoring_degraded,
                "courseforge_provider": _courseforge_provider_env,
                "two_pass_enabled": _courseforge_two_pass_enabled(),
                "agent_dispatch_enabled": agent_dispatch_enabled,
                "llm_authoring_intended": llm_authoring_intended,
                "allow_template_emitter": allow_template,
                "weeks_prepared": weeks_prepared,
                "page_count": len(generated_files),
                "course_code": course_code,
            }
            provenance_path = project_path / "content_generation_provenance.json"
            try:
                provenance_path.write_text(
                    json.dumps(generation_provenance, indent=2), encoding="utf-8"
                )
            except OSError as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to write content_generation_provenance.json: %s", exc
                )
                provenance_path = None

            content_paths_str = ",".join(generated_files)
            return json.dumps({
                "success": True,
                "project_id": project_id,
                "weeks_prepared": weeks_prepared,
                "content_paths": content_paths_str,
                "page_paths": generated_files,
                "content_dir": str(content_dir),
                "source_sections": len(topics),
                "content_selection": (
                    "source-grounded" if topics else "synthesized"
                ),
                "generator_mode": generator_mode,
                "template_fallback_fired": template_fired,
                "content_generation_provenance_path": (
                    str(provenance_path) if provenance_path else ""
                ),
            })

        registry["generate_course_content"] = _generate_course_content
        # END BLOCK: Worker α

        # Phase 3.5 Subtasks 28-30: register the three two-pass router
        # phase helpers so the executor's _PHASE_TOOL_MAPPING shim
        # (Subtask 31) can dispatch to them by phase name.
        registry["run_content_generation_outline"] = (
            _run_content_generation_outline
        )
        registry["run_inter_tier_validation"] = _run_inter_tier_validation
        registry["run_content_generation_rewrite"] = (
            _run_content_generation_rewrite
        )
        # Phase 3.5 Subtask 13 + Wave C: also register the post_rewrite_validation
        # helper so phase-name dispatch can route to it (Subtask 31).
        registry["run_post_rewrite_validation"] = _run_post_rewrite_validation

        async def _package_imscc(**kwargs):
            """Build a real IMS Common Cartridge package from generated content.

            ⚠  **Sync-parity with**
            ``MCP/tools/courseforge_tools.py::package_imscc`` (the
            ``@mcp.tool()`` variant) is required. Both wrappers delegate to
            ``Courseforge.scripts.package_multifile_imscc.package_imscc``
            and share the same JSON envelope shape. This registry variant
            omits the `project_config.status`/`package_path` side-effects
            that the MCP-decorated variant performs — phase tracking
            happens in the workflow runner here. Keep both surfaces in
            lockstep until a shared helper is extracted in a later wave.

            Wave 27 HIGH-2: delegates to the mature multi-file packager
            (``Courseforge.scripts.package_multifile_imscc.package_imscc``)
            rather than hand-rolling the ZIP. Consequences of the
            delegation:

            * Per-week ``learningObjectives`` validation runs by default
              (the mature packager refuses to build when any page's LO
              list references an out-of-week ID).
            * ``course_metadata.json`` is bundled at the zip root when
              present (the mature packager's Wave 3 REC-TAX-01 behavior).
            * Manifest uses IMS Common Cartridge v1.3 namespaces.
            * Resources are nested under per-week ``<item>`` wrappers in
              the organization tree — Brightspace / Canvas / Moodle
              render a week-grouped module list instead of a flat page
              dump.

            The legacy JSON envelope (``success``, ``package_path``,
            ``libv2_package_path``, ``html_modules``, ``package_size_bytes``)
            is preserved so callers see no contract change. LO-contract
            failure surfaces as ``{"success": false, "error": ...,
            "validation_failures": [...]}`` instead of silently falling
            through.
            """
            import sys as _sys
            from pathlib import Path as _Path

            project_id = kwargs.get("project_id", "")
            project_path = courseforge_exports_dir() / project_id
            content_dir = project_path / "03_content_development"
            final_dir = project_path / "05_final_package"
            final_dir.mkdir(parents=True, exist_ok=True)

            # Sanity: require the content dir + at least one HTML page.
            html_files = sorted(content_dir.rglob("*.html"))
            if not html_files:
                return json.dumps({
                    "error": "No HTML modules found in content directory",
                    "content_dir": str(content_dir),
                })

            config_path = project_path / "project_config.json"
            course_name = project_id
            course_title = project_id
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    course_name = cfg.get("course_name", project_id)
                    course_title = (
                        cfg.get("course_title")
                        or cfg.get("title")
                        or course_name
                    )
                except (OSError, json.JSONDecodeError):
                    pass

            # Optional: caller-provided objectives JSON used by the
            # mature packager's LO-contract validator. Falls back to the
            # packager's auto-discovery (content_dir/course.json).
            objectives_path_kw = kwargs.get("objectives_path")
            objectives_path = (
                _Path(objectives_path_kw) if objectives_path_kw else None
            )
            skip_validation = bool(kwargs.get("skip_validation", False))

            # Wave2-I3 (Finding 3, plans/dispatch-7-execution-inspection-
            # 2026-05.md): emit packaging-shaped course.json from
            # synthesized_objectives.json BEFORE the mature packager + the
            # PageObjectivesValidator gate fire. Pre-fix, no phase wrote
            # course.json at the Courseforge content root, so the
            # PageObjectivesValidator (lib/validators/page_objectives.py
            # :192-249) auto-discovery hit a missing file and fail-closed
            # critical-severity with PAGE_OBJECTIVES_PATH_MISSING — blocking
            # every textbook_to_course packaging gate. Now the synthesized
            # objectives from the course_planning phase are projected to
            # the canonical packaging shape (terminal_objectives +
            # chapter_objectives + learning_outcomes[]) at the location
            # both the mature packager and the validator auto-discover.
            # Idempotent: when course.json already exists (e.g.
            # --reuse-objectives ran or a future Trainforge phase emitted
            # one), the projection logs INFO and skips without overwrite.
            synthesized_objectives_path = (
                project_path
                / "01_learning_objectives"
                / "synthesized_objectives.json"
            )
            content_course_json_path = content_dir / "course.json"
            try:
                _project_synthesized_objectives_to_course_json(
                    synthesized_objectives_path,
                    content_course_json_path,
                    course_code=course_name,
                    course_title=course_title,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort projection
                logger.warning(
                    "_package_imscc: course.json projection raised %s; "
                    "packaging will continue but PageObjectivesValidator "
                    "may fail closed downstream.",
                    exc,
                )

            package_path = final_dir / f"{course_name}.imscc"
            # W3.H sub-task H3: emit a sibling packaging_report.json
            # carrying the canonical source_coverage block. Lives at
            # ``<project_root>/05_final_package/packaging_report.json``
            # (next to the .imscc archive) per plan §W3.H.
            packaging_report_path = final_dir / "packaging_report.json"

            # Import the mature packager. The module lives under
            # ``Courseforge/scripts/`` (no ``__init__.py``) so we prepend
            # the directory to ``sys.path`` before importing. Resolve the
            # directory relative to this module's real location (NOT
            # ``_PROJECT_ROOT``, which tests may monkeypatch to a tmp
            # workspace that doesn't ship the mature packager).
            cf_scripts = (
                _Path(__file__).resolve().parents[2]
                / "Courseforge" / "scripts"
            )
            if str(cf_scripts) not in _sys.path:
                _sys.path.insert(0, str(cf_scripts))
            try:
                import package_multifile_imscc as _pkg_mod  # noqa: E402
            except ImportError as exc:
                return json.dumps({
                    "success": False,
                    "error": f"Failed to import mature packager: {exc}",
                    "project_id": project_id,
                })

            # Run in an executor so the (synchronous) packager does not
            # block the event loop. SystemExit raised by the packager on
            # LO-contract failure surfaces as a ``SystemExit`` we convert
            # into a structured error response. Any other exception is
            # surfaced the same way so the caller sees a normal JSON
            # envelope rather than a crash.
            try:
                _pkg_mod.package_imscc(
                    content_dir,
                    package_path,
                    course_name,
                    course_title,
                    objectives_path=objectives_path,
                    skip_validation=skip_validation,
                    coverage_sidecar_path=packaging_report_path,
                )
            except SystemExit as exc:
                return json.dumps({
                    "success": False,
                    "error": (
                        "IMSCC packaging refused: per-week LO contract "
                        "validation failed. See logs for per-page details."
                    ),
                    "exit_code": (
                        exc.code if isinstance(exc.code, int) else 2
                    ),
                    "project_id": project_id,
                    "packaging_report_path": str(packaging_report_path)
                    if packaging_report_path.exists() else None,
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Mature packager raised for project %s: %s",
                    project_id, exc,
                )
                return json.dumps({
                    "success": False,
                    "error": f"Mature packager failed: {exc}",
                    "project_id": project_id,
                })

            # Wave 32 Deliverable B: surface imscc_path + content_dir
            # alongside the legacy package_path / libv2_package_path
            # aliases so the IMSCCValidator + PageObjectivesValidator
            # gate builders stop reporting ``missing inputs:
            # imscc_path / content_dir``.
            return json.dumps({
                "success": True,
                "project_id": project_id,
                "package_path": str(package_path),
                "libv2_package_path": str(package_path),
                "imscc_path": str(package_path),
                "content_dir": str(content_dir),
                "html_modules": len(html_files),
                "package_size_bytes": package_path.stat().st_size,
                "packaging_report_path": str(packaging_report_path)
                if packaging_report_path.exists() else None,
            })

        registry["package_imscc"] = _package_imscc
    except ImportError:
        pass

    # Trainforge tools
    try:
        async def _analyze_imscc_content(**kwargs):
            """Registry wrapper: real IMSCC analysis (parity with @mcp.tool() variant).

            Previously a zero-value stub (audit Q4). Now opens the zip,
            validates the manifest, counts HTML modules + existing
            assessments, and suggests assessment opportunities — matching
            the MCP variant at trainforge_tools.py:129.
            """
            import zipfile

            imscc_path = kwargs.get("imscc_path", "")
            try:
                imscc = Path(imscc_path)
                if not imscc.exists():
                    return json.dumps({"error": f"IMSCC not found: {imscc_path}"})

                analysis = {
                    "source": str(imscc),
                    "analyzed_at": datetime.now().isoformat(),
                    "content": {
                        "html_modules": 0,
                        "existing_assessments": 0,
                        "total_word_count": 0,
                    },
                    "learning_objectives": [],
                    "assessment_opportunities": [],
                }

                with zipfile.ZipFile(imscc, "r") as z:
                    if "imsmanifest.xml" not in z.namelist():
                        return json.dumps({
                            "error": (
                                f"Invalid IMSCC package: missing imsmanifest.xml "
                                f"in {imscc.name}"
                            ),
                            "hint": (
                                "A valid IMSCC package must contain an "
                                "imsmanifest.xml file"
                            ),
                        })
                    analysis["has_manifest"] = True

                    for name in z.namelist():
                        if name.endswith(".html"):
                            analysis["content"]["html_modules"] += 1
                            content = z.read(name).decode("utf-8", errors="ignore")
                            word_count = len(content.split())
                            analysis["content"]["total_word_count"] += word_count
                            if "objective" in content.lower():
                                analysis["learning_objectives"].append({
                                    "source_file": name,
                                    "detected": True,
                                })
                        elif name.endswith(".xml") and "assessment" in name.lower():
                            analysis["content"]["existing_assessments"] += 1

                if analysis["content"]["html_modules"] > 0:
                    analysis["assessment_opportunities"] = [
                        {
                            "type": "quiz",
                            "coverage": "per_module",
                            "estimated_questions": (
                                analysis["content"]["html_modules"] * 5
                            ),
                        },
                        {
                            "type": "exam",
                            "coverage": "comprehensive",
                            "estimated_questions": min(
                                50, analysis["content"]["html_modules"] * 3
                            ),
                        },
                    ]

                return json.dumps(analysis)
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry["analyze_imscc_content"] = _analyze_imscc_content

        # ============================================================================
        # BLOCK: Worker β edits ONLY below this line through the next END marker.
        # Scope: _generate_assessments replacement. See plans/pipeline-execution-
        # fixes/contracts.md § "Trainforge-execution contract".
        # ============================================================================
        async def _generate_assessments(**kwargs):
            """Run Trainforge's full corpus pipeline against the IMSCC and
            generate grounded assessments.

            Concrete steps:

            1. Invoke Trainforge's :class:`CourseProcessor` (the same code
               path ``python -m Trainforge.process_course`` uses) against
               the packaged IMSCC. Produces ``corpus/chunks.jsonl``,
               ``graph/concept_graph_semantic.json``, ``manifest.json``,
               and a ``quality/`` report, validating under chunk_v4 /
               typed-edge schemas when the opt-in flags are set.
            2. Aggregate inline ``chunk["misconceptions"]`` entries into
               a first-class ``graph/misconceptions.json`` document with
               content-hash IDs (``mc_[0-9a-f]{16}``), per REC-LNK-02 and
               the ``misconception.schema.json`` shape.
            3. Run :class:`AssessmentGenerator` honoring the workflow's
               ``question_count`` / ``bloom_levels`` / ``objective_ids``
               params against the generated chunks, writing a single
               well-formed ``assessments.json`` (NOT the legacy
               jsonl-then-concat pattern that produced "Extra data"
               errors).

            Output dir: ``{project_workspace}/trainforge/`` where
            ``project_workspace`` is derived from
            ``imscc_path.parent.parent`` (the Courseforge project dir)
            or, for standalone calls, from an explicit ``project_id``
            kwarg. Colocating with the Courseforge export dir keeps all
            per-run artifacts under one tree and lets the
            libv2-archival phase locate them without a cross-tree
            lookup.
            """
            import hashlib as _hashlib
            import os as _os
            import traceback as _traceback

            course_id = kwargs.get("course_id") or kwargs.get("course_code") or ""
            question_count = int(kwargs.get("question_count", 10))
            bloom_levels_str = kwargs.get("bloom_levels", "remember,understand,apply")
            objective_ids_str = kwargs.get("objective_ids", "")
            imscc_path_str = kwargs.get("imscc_path", "")
            project_id_kw = kwargs.get("project_id", "")
            domain = kwargs.get("domain") or "general"
            division = kwargs.get("division") or "STEM"
            # Phase 8 ST 2: pre-built IMSCC chunkset path threaded
            # through from the upstream ``imscc_chunking`` workflow
            # phase (Phase 7c ST 16,
            # ``MCP/tools/pipeline_tools.py::_run_imscc_chunking``).
            # When supplied + readable, ``CourseProcessor.process``
            # short-circuits its in-process ``_chunk_content`` call
            # and consumes the canonical chunks emitted upstream at
            # ``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl``.
            # Falls through to the legacy in-process build path on
            # absent / unreadable upstream chunks. Routed via
            # ``config/workflows.yaml::trainforge_assessment.inputs_from``
            # + ``MCP/core/workflow_runner.py::_LEGACY_PHASE_PARAM_ROUTING``.
            imscc_chunks_path_kw = kwargs.get("imscc_chunks_path") or ""

            # Normalize list-ish params.
            if isinstance(bloom_levels_str, list):
                bloom_levels = [str(b).strip() for b in bloom_levels_str if str(b).strip()]
            else:
                bloom_levels = [b.strip() for b in str(bloom_levels_str).split(",") if b.strip()]
            if not bloom_levels:
                bloom_levels = ["remember", "understand", "apply"]

            if isinstance(objective_ids_str, list):
                objective_ids = [str(o).strip() for o in objective_ids_str if str(o).strip()]
            else:
                objective_ids = [o.strip() for o in str(objective_ids_str).split(",") if o.strip()]
            if not objective_ids:
                objective_ids = [f"{course_id}_OBJ_{i}" for i in range(1, 7)]

            # Locate project workspace. Standard path: imscc is under
            # Courseforge/exports/<proj>/05_final_package/, so project_dir
            # is imscc.parent.parent. Explicit project_id kwarg wins if set.
            project_dir: Optional[Path] = None
            imscc_path = Path(imscc_path_str) if imscc_path_str else None
            if project_id_kw:
                candidate = courseforge_exports_dir() / project_id_kw
                if candidate.exists():
                    project_dir = candidate
            if project_dir is None and imscc_path and imscc_path.exists():
                candidate = imscc_path.parent.parent
                if candidate.exists():
                    project_dir = candidate
            if project_dir is None:
                # Last-resort fallback: most recent export dir matching course_id.
                exports_dir = courseforge_exports_dir()
                if exports_dir.exists():
                    matches = sorted(
                        (p for p in exports_dir.iterdir()
                         if p.is_dir() and course_id and course_id.lower() in p.name.lower()),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if matches:
                        project_dir = matches[0]
            if project_dir is None:
                return json.dumps({
                    "error": "Cannot locate project workspace for Trainforge output",
                    "imscc_path": imscc_path_str,
                    "course_id": course_id,
                })

            trainforge_dir = project_dir / "trainforge"
            # Wipe any prior run's output so a retry starts clean.
            if trainforge_dir.exists():
                shutil.rmtree(trainforge_dir, ignore_errors=True)
            trainforge_dir.mkdir(parents=True, exist_ok=True)

            if not imscc_path or not imscc_path.exists() or imscc_path.stat().st_size == 0:
                return json.dumps({
                    "error": "IMSCC package not found or empty; Trainforge requires the packaging phase to complete first",
                    "imscc_path": imscc_path_str,
                })

            # Invoke CourseProcessor. Writes:
            #   <trainforge_dir>/corpus/chunks.jsonl
            #   <trainforge_dir>/graph/concept_graph.json
            #   <trainforge_dir>/graph/concept_graph_semantic.json
            #   <trainforge_dir>/graph/pedagogy_graph.json
            #   <trainforge_dir>/manifest.json
            #   <trainforge_dir>/quality/quality_report.json
            try:
                from Trainforge.process_course import CourseProcessor
            except Exception as e:
                return json.dumps({
                    "error": f"Failed to import CourseProcessor: {e}",
                    "traceback": _traceback.format_exc(limit=4),
                })

            # Wave 24: thread objectives_path through to CourseProcessor
            # so Trainforge synthesizes self.objectives, populates
            # _build_valid_outcome_ids, and writes course.json. Before
            # Wave 24 this argument was missing, so every chunk's
            # learning_outcome_refs surfaced as broken.
            project_dir_objectives = None
            try:
                cfg_path = project_dir / "project_config.json"
                if cfg_path.exists():
                    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
                    project_dir_objectives = (
                        cfg_data.get("synthesized_objectives_path")
                        or cfg_data.get("objectives_path")
                    )
            except (OSError, ValueError):
                project_dir_objectives = None

            # Legacy / no-textbook path: no objectives JSON. Fall back
            # to CourseProcessor's pre-Wave-24 behavior (no course.json,
            # empty valid_outcome_ids) with a single warning log so the
            # gap is observable.
            if not project_dir_objectives:
                logger.warning(
                    "[Wave 24] CourseProcessor invoked without an "
                    "objectives_path (project %s). course.json will not "
                    "be written; chunk learning_outcome_refs may surface "
                    "as broken. Run plan_course_structure first to "
                    "populate synthesized_objectives.json.",
                    project_dir.name,
                )

            processor = CourseProcessor(
                imscc_path=str(imscc_path),
                output_dir=str(trainforge_dir),
                course_code=course_id,
                division=division,
                domain=domain,
                objectives_path=(
                    str(project_dir_objectives) if project_dir_objectives else None
                ),
                strict_mode=False,
                # Phase 8 ST 2: pass through the upstream IMSCC
                # chunkset path so ``process()`` consumes it instead
                # of re-running the chunker in-process. Empty string
                # resolves to None inside the constructor (preserves
                # backward compat with callers that don't thread the
                # path).
                imscc_chunks_path=(
                    imscc_chunks_path_kw if imscc_chunks_path_kw else None
                ),
            )

            # Wave 22 DC2: the historical strict-mode override here was a
            # landmine. process_course.py now uses the canonical phase
            # name ``"trainforge-content-analysis"`` (already fixed) and
            # Wave 22 adds the five previously-orphan decision_type
            # values (assessment_planning, question_type_selection,
            # assessment_generation, content_selection, boilerplate_strip)
            # to ``schemas/events/decision_event.schema.json``. With both
            # landmines cleared, the caller's configured strictness now
            # applies uniformly across CourseProcessor + downstream
            # AssessmentGenerator runs.
            try:
                summary = processor.process()
            except Exception as e:
                return json.dumps({
                    "error": f"CourseProcessor.process() failed: {e}",
                    "traceback": _traceback.format_exc(limit=6),
                    "output_dir": str(trainforge_dir),
                })

            # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
            from lib.libv2_storage import resolve_imscc_chunks_path
            chunks_path = resolve_imscc_chunks_path(trainforge_dir, "chunks.jsonl")
            semantic_graph_path = trainforge_dir / "graph" / "concept_graph_semantic.json"

            if not chunks_path.exists():
                return json.dumps({
                    "error": "CourseProcessor did not produce chunks.jsonl",
                    "output_dir": str(trainforge_dir),
                })

            # Aggregate first-class misconceptions.json. Pulls inline
            # misconceptions from each chunk, dedupes by content, and
            # assigns mc_<16-hex> content-hash IDs per
            # schemas/knowledge/misconception.schema.json.
            loaded_chunks: list = []
            with open(chunks_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        loaded_chunks.append(json.loads(_line))
                    except (json.JSONDecodeError, ValueError):
                        continue

            mc_entities: list = []
            mc_seen: set = set()
            for _c in loaded_chunks:
                for _mc in _c.get("misconceptions") or []:
                    if not isinstance(_mc, dict):
                        continue
                    mtext = str(_mc.get("misconception", "")).strip()
                    ctext = str(_mc.get("correction", "")).strip()
                    if not mtext:
                        continue
                    # Correction is minLength:1 under the schema. Supply
                    # a minimal placeholder when the source didn't carry
                    # one (common with regex-extracted prose).
                    if not ctext:
                        ctext = "Correction not captured in source; review instructor materials."
                    _digest = _hashlib.sha256(
                        f"{mtext}|{ctext}".encode()
                    ).hexdigest()[:16]
                    mc_id = f"mc_{_digest}"
                    if mc_id in mc_seen:
                        continue
                    mc_seen.add(mc_id)
                    entity: dict = {
                        "id": mc_id,
                        "misconception": mtext,
                        "correction": ctext,
                    }
                    tags = _c.get("concept_tags") or []
                    if isinstance(tags, list) and tags:
                        entity["concept_id"] = str(tags[0])
                    los = _c.get("learning_outcome_refs") or []
                    if isinstance(los, list) and los:
                        entity["lo_id"] = str(los[0])
                    mc_entities.append(entity)

            # Fallback: process_course.py surfaced zero misconceptions
            # but we have real chunks — try the regex extractor on chunk
            # text. Keeps the artifact shape honest while Courseforge
            # (Worker α) is still being brought online with JSON-LD
            # misconceptions.
            if not mc_entities and loaded_chunks:
                try:
                    from Trainforge.process_course import extract_misconceptions_from_text
                    for _c in loaded_chunks:
                        text = str(_c.get("text", ""))
                        for _mc in extract_misconceptions_from_text(text):
                            mtext = _mc.get("misconception", "").strip()
                            if not mtext:
                                continue
                            ctext = _mc.get("correction") or "Correction not captured in source; review instructor materials."
                            _digest = _hashlib.sha256(
                                f"{mtext}|{ctext}".encode()
                            ).hexdigest()[:16]
                            mc_id = f"mc_{_digest}"
                            if mc_id in mc_seen:
                                continue
                            mc_seen.add(mc_id)
                            mc_entities.append({
                                "id": mc_id,
                                "misconception": mtext,
                                "correction": ctext,
                            })
                        if mc_entities:
                            break
                except Exception:
                    pass

            misconceptions_path = trainforge_dir / "graph" / "misconceptions.json"
            misconceptions_path.parent.mkdir(parents=True, exist_ok=True)
            with open(misconceptions_path, "w", encoding="utf-8") as _f:
                json.dump({"misconceptions": mc_entities}, _f, indent=2, ensure_ascii=False)

            # Run AssessmentGenerator on the Trainforge chunks. Every
            # field the ContentExtractor reads (text, concept_tags,
            # source, id) is already present in the canonical chunk
            # shape. Decision capture via create_trainforge_capture
            # writes the rationale stream.
            try:
                from Trainforge.generators.assessment_generator import AssessmentGenerator
            except Exception as e:
                return json.dumps({
                    "error": f"Failed to import AssessmentGenerator: {e}",
                    "traceback": _traceback.format_exc(limit=4),
                    "chunks_path": str(chunks_path),
                })

            gen_capture = None
            try:
                from lib.trainforge_capture import create_trainforge_capture
                gen_capture = create_trainforge_capture(
                    course_code=course_id or "UNKNOWN",
                    imscc_source=str(imscc_path),
                )
            except Exception:
                gen_capture = None

            generator = AssessmentGenerator(capture=gen_capture, check_leaks=True)
            try:
                assessment = generator.generate(
                    course_code=course_id,
                    objective_ids=objective_ids,
                    bloom_levels=bloom_levels,
                    question_count=question_count,
                    source_chunks=loaded_chunks,
                )
            except Exception as e:
                return json.dumps({
                    "error": f"AssessmentGenerator.generate() failed: {e}",
                    "traceback": _traceback.format_exc(limit=6),
                    "chunks_path": str(chunks_path),
                })

            assessments_path = trainforge_dir / "assessments.json"
            assessment_doc = assessment.to_dict()
            # Single write, single well-formed JSON document. The legacy
            # "Extra data" bug came from calling json.dump then appending
            # additional text to the same handle; we guard against that
            # by using a fresh open() and exactly one dump call.
            with open(assessments_path, "w", encoding="utf-8") as _f:
                json.dump(assessment_doc, _f, indent=2, ensure_ascii=False)

            # Wave 26: graft the assessment dimension onto quality_report.json
            # so a reviewer can see which questions are broken without
            # re-running validators. Best-effort: on any error we preserve
            # the existing quality report unchanged.
            try:
                from Trainforge.generators.assessment_quality_report import (
                    build_assessment_dimension,
                )
                qr_path = trainforge_dir / "quality" / "quality_report.json"
                if qr_path.exists():
                    with open(qr_path, encoding="utf-8") as _qrf:
                        qr_doc = json.load(_qrf)
                    dim = build_assessment_dimension(assessment_doc)
                    if dim is not None:
                        qr_doc["assessments"] = dim
                        with open(qr_path, "w", encoding="utf-8") as _qrf:
                            json.dump(qr_doc, _qrf, indent=2, ensure_ascii=False)
            except Exception as _qr_err:
                logger.warning(
                    "Failed to graft assessment dimension onto "
                    "quality_report.json: %s", _qr_err,
                )

            if gen_capture is not None:
                try:
                    gen_capture.log_decision(
                        decision_type="content_selection",
                        decision=(
                            f"Trainforge phase wrote {len(loaded_chunks)} chunks, "
                            f"{len(mc_entities)} misconceptions, "
                            f"{len(assessment.questions)} assessment questions "
                            f"to {trainforge_dir}"
                        ),
                        rationale=(
                            "Ran CourseProcessor against the packaged IMSCC to produce the "
                            "canonical corpus + typed-edge graph, then synthesized misconception "
                            "entities with content-hash IDs. Colocated output under the Courseforge "
                            "project dir so downstream LibV2 archival can byte-copy without a "
                            "cross-tree lookup. Honored workflow params for bloom_levels "
                            f"({','.join(bloom_levels)}) and question_count ({question_count})."
                        ),
                    )
                except Exception:
                    pass

            mc_id_out = str(misconceptions_path) if mc_entities else None
            validated = (
                _os.getenv("TRAINFORGE_VALIDATE_CHUNKS", "").lower() == "true"
            )

            # Worker W1: surface skipped-item count from the killed
            # template-fallback path so the workflow phase output keeps
            # visibility on slots the generator refused to fill.
            skipped_total = len(getattr(assessment, "skipped_items", []) or [])
            skipped_summary = [
                s.to_dict() for s in (getattr(assessment, "skipped_items", []) or [])[:3]
            ]
            return json.dumps({
                "success": True,
                "assessment_id": assessment.assessment_id,
                "question_count": len(assessment.questions),
                "skipped_items_count": skipped_total,
                "skipped_items_summary": skipped_summary,
                "output_path": str(assessments_path),
                "assessments_path": str(assessments_path),
                "chunks_path": str(chunks_path),
                "concept_graph_path": (
                    str(semantic_graph_path) if semantic_graph_path.exists() else None
                ),
                "misconceptions_path": mc_id_out,
                "trainforge_dir": str(trainforge_dir),
                "chunks_count": len(loaded_chunks),
                "misconceptions_count": len(mc_entities),
                "strict_chunks_validated": validated,
                "processor_summary": {
                    "course_code": summary.get("course_code"),
                    "title": summary.get("title"),
                    "stats": summary.get("stats"),
                },
            })

        registry["generate_assessments"] = _generate_assessments
        # END BLOCK: Worker β
    except Exception:
        pass

    # Wave 30 Gap 3: training_synthesis phase
    # ============================================================================
    # Wraps ``Trainforge.synthesize_training.run_synthesis`` as a pipeline phase
    # so ``textbook_to_course`` runs now materialise ``training_specs/
    # instruction_pairs.jsonl`` + ``training_specs/preference_pairs.jsonl``
    # alongside ``assessments.json``. Pre-Wave-30 the synthesizer only ran
    # when a human invoked its CLI — no textbook-to-course run ever emitted
    # SFT / DPO pairs, so ``ed4all export-training ... --format dpo`` was
    # exporting decision captures instead of real Q&A pairs.
    # ============================================================================
    async def _synthesize_training(**kwargs):
        """Generate SFT + DPO training pairs from the Trainforge corpus.

        Required inputs (accepts both shapes so both the MCP-tool and
        pipeline-dispatch variants route here cleanly):

        * ``corpus_dir`` OR ``trainforge_dir`` — the Trainforge output
          directory that already holds ``corpus/chunks.jsonl``. Derived
          from ``assessments_path`` (its parent) when neither is given.
        * ``course_code`` OR ``course_name`` OR ``course_id`` — used for
          decision capture so the run is traceable.

        Optional:

        * ``provider`` — synthesis provider. Accepted values: ``"mock"``
          (default; deterministic template factory), ``"anthropic"``
          (Anthropic SDK; requires ``ANTHROPIC_API_KEY``),
          ``"claude_session"`` (Claude Code session via LocalDispatcher),
          ``"together"`` (Together AI's OpenAI-compatible endpoint;
          default model ``meta-llama/Llama-3.3-70B-Instruct-Turbo``,
          override via ``TOGETHER_SYNTHESIS_MODEL``; requires
          ``TOGETHER_API_KEY``), or ``"local"`` (a local
          OpenAI-compatible model server: Ollama / vLLM / llama.cpp /
          LM Studio. Default base URL ``http://localhost:11434/v1``,
          override via ``LOCAL_SYNTHESIS_BASE_URL``; default model
          ``qwen2.5:14b-instruct-q4_K_M``, override via
          ``LOCAL_SYNTHESIS_MODEL``; API key optional). Together's ToS
          permits training-data generation, unlike Anthropic's; the
          local provider is fully offline and ToS-free. When ``None``
          is explicitly set AND no LLM backend is resolvable, the
          function logs a skip warning and returns an empty-results
          shell rather than crashing.
        * ``seed`` (int, default ``DEFAULT_SEED`` from
          ``synthesize_training`` so re-runs are byte-identical).

        Returns a JSON string with ``instruction_pairs_path``,
        ``preference_pairs_path``, and the ``SynthesisStats`` dict.
        """
        # Resolve the corpus directory.
        corpus_dir = (
            kwargs.get("corpus_dir")
            or kwargs.get("trainforge_dir")
            or kwargs.get("output_dir")
        )
        if not corpus_dir:
            assessments_path = kwargs.get("assessments_path")
            if assessments_path:
                corpus_dir = str(Path(assessments_path).parent)
        if not corpus_dir:
            chunks_path = kwargs.get("chunks_path")
            if chunks_path:
                # chunks.jsonl lives at {corpus_dir}/corpus/chunks.jsonl, so
                # the Trainforge root is two parents up.
                corpus_dir = str(Path(chunks_path).parent.parent)
        if not corpus_dir:
            return json.dumps({
                "error": (
                    "synthesize_training requires corpus_dir / "
                    "trainforge_dir / assessments_path / chunks_path to "
                    "locate imscc_chunks/chunks.jsonl (or legacy "
                    "corpus/chunks.jsonl)"
                ),
            })

        corpus_dir_path = Path(corpus_dir)
        # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
        from lib.libv2_storage import resolve_imscc_chunks_path
        chunks_path = resolve_imscc_chunks_path(corpus_dir_path, "chunks.jsonl")
        if not chunks_path.exists():
            # Skip-with-warning: downstream archival can still run, we
            # just won't have new training pairs. This is the safe
            # no-LLM-available path the audit calls out.
            logger.warning(
                "synthesize_training: chunks.jsonl missing at %s; "
                "skipping training-pair synthesis. ",
                chunks_path,
            )
            return json.dumps({
                "success": True,
                "skipped": True,
                "reason": "chunks_missing",
                "corpus_dir": str(corpus_dir_path),
            })

        course_code = (
            kwargs.get("course_code")
            or kwargs.get("course_name")
            or kwargs.get("course_id")
            or "UNKNOWN"
        )

        provider = kwargs.get("provider", "mock")
        # Seed defaults to synthesize_training's DEFAULT_SEED so re-runs
        # are byte-identical. Callers can override for test determinism.
        seed = kwargs.get("seed")

        # Wave 129: forward Wave 124-127 deterministic-generator kwargs
        # so workflow-phase dispatch + external MCP clients can trigger
        # kg_metadata / violation_detection / abstention / schema_translation
        # without the CLI. Defaults mirror run_synthesis() at
        # Trainforge/synthesize_training.py:677-685.
        with_kg_metadata = bool(kwargs.get("with_kg_metadata", False))
        kg_metadata_max_pairs = int(kwargs.get("kg_metadata_max_pairs", 2000))
        with_violation_detection = bool(
            kwargs.get("with_violation_detection", False)
        )
        violation_detection_max_pairs = kwargs.get("violation_detection_max_pairs")
        with_abstention = bool(kwargs.get("with_abstention", False))
        abstention_max_pairs = int(kwargs.get("abstention_max_pairs", 1000))
        with_schema_translation = bool(
            kwargs.get("with_schema_translation", False)
        )
        schema_translation_max_pairs = int(
            kwargs.get("schema_translation_max_pairs", 50)
        )

        try:
            from Trainforge.synthesize_training import (
                DEFAULT_SEED,
                run_synthesis,
            )
        except Exception as exc:  # pragma: no cover — dependency error
            return json.dumps({
                "error": f"Failed to import synthesize_training: {exc}",
            })

        if seed is None:
            seed = DEFAULT_SEED

        try:
            stats = run_synthesis(
                corpus_dir=corpus_dir_path,
                course_code=str(course_code),
                provider=str(provider),
                seed=int(seed),
                with_kg_metadata=with_kg_metadata,
                kg_metadata_max_pairs=kg_metadata_max_pairs,
                with_violation_detection=with_violation_detection,
                violation_detection_max_pairs=violation_detection_max_pairs,
                with_abstention=with_abstention,
                abstention_max_pairs=abstention_max_pairs,
                with_schema_translation=with_schema_translation,
                schema_translation_max_pairs=schema_translation_max_pairs,
            )
        except Exception as exc:
            return json.dumps({
                "error": f"synthesize_training failed: {exc}",
                "corpus_dir": str(corpus_dir_path),
            })

        instruction_pairs_path = (
            corpus_dir_path / "training_specs" / "instruction_pairs.jsonl"
        )
        preference_pairs_path = (
            corpus_dir_path / "training_specs" / "preference_pairs.jsonl"
        )

        return json.dumps({
            "success": True,
            "corpus_dir": str(corpus_dir_path),
            "instruction_pairs_path": str(instruction_pairs_path),
            "preference_pairs_path": str(preference_pairs_path),
            "instruction_pairs_count": stats.instruction_pairs_emitted,
            "preference_pairs_count": stats.preference_pairs_emitted,
            "chunks_eligible": stats.chunks_eligible,
            "chunks_total": stats.chunks_total,
            "stats": stats.as_dict(),
        })

    registry["synthesize_training"] = _synthesize_training

    # LibV2 archival tool
    # ============================================================================
    # BLOCK: Worker γ edits ONLY below this line through the next END marker.
    # Scope: _archive_to_libv2 extension. See plans/pipeline-execution-fixes/
    # contracts.md § "LibV2-archival contract".
    # ============================================================================
    async def _archive_to_libv2(**kwargs):
        """Archive pipeline artifacts (sources + Trainforge outputs) to LibV2.

        Parity with the ``@mcp.tool()`` variant at ``pipeline_tools.py:556-726``
        (slug computation, source copying, manifest shape, feature-flag scans)
        plus Wave 15 Trainforge output copying into
        ``corpus/`` / ``graph/`` / ``training_specs/`` / ``quality/``.

        Trainforge output lookup order (first match wins):
          1. Explicit kwargs: ``project_workspace`` (str/Path), else
             ``project_id`` → ``Courseforge/exports/{project_id}/trainforge/``.
          2. Legacy ``assessment_path`` — when it points at a directory, used
             as the Trainforge output root; when it points at a file, copied
             into ``corpus/`` (preserves the MCP-tool variant's behavior so
             existing provenance-flag tests keep passing).
          3. Heuristic fallback — scan ``Courseforge/exports/*/trainforge/``
             and ``state/runs/*/trainforge/`` for the most recently modified
             ``chunks.jsonl``. Absence is not an error — features flags fall
             back to ``false`` with a warning.

        Wave 74 fail-closed gate: when ``chunks.jsonl`` is found at the
        archive destination but doesn't carry IDs from this run's
        ``course_code`` (pattern ``^{course_code_lower}_chunk_``), the
        archival call refuses to proceed and emits ``error_code =
        TRAINFORGE_OUTPUT_STALE``. This catches the case where a prior
        run's chunks under the same slug survived into a fresh archive
        (observed today: smoke_sample_rag_chunk_* IDs leaked into the
        RDF/SHACL calibration corpus archive after trainforge_assessment
        failed). When
        Trainforge was intentionally absent (no chunks file at all), the
        archival proceeds — feature flags fall back to false with a
        warning, matching the pre-Wave-74 behaviour for DART-only runs.
        """
        # Wave 74: capture run-start mtime *before* any writes. Used as a
        # cheap second guard alongside the ID-pattern check below.
        _run_start_ts = time.time()

        course_name = (
            kwargs.get("course_name")
            or kwargs.get("course_id")
            or kwargs.get("id")
            or ""
        )
        domain = kwargs.get("domain") or "general"
        division = kwargs.get("division", "STEM")
        pdf_paths_str = kwargs.get("pdf_paths", "") or ""
        html_paths_str = kwargs.get("html_paths", "") or ""
        imscc_path_str = kwargs.get("imscc_path", "") or ""
        assessment_path_str = kwargs.get("assessment_path", "") or ""
        subdomains_str = kwargs.get("subdomains", "") or ""
        project_workspace_kw = kwargs.get("project_workspace") or ""
        project_id_kw = kwargs.get("project_id") or ""
        # Phase 6 ST 18: concept-graph hash threaded from the
        # ``concept_extraction`` phase output via the workflow runner's
        # ``inputs_from`` chain. Optional — when absent we fall back to
        # recomputing from ``concept_graph/concept_graph_semantic.json``
        # on disk under the LibV2 course dir (Worker C-J's helper writes
        # the file there at ST 12), and finally to ``None`` when no graph
        # exists (legacy / DART-only runs). Schema field is optional in
        # Phase 6 (Wave 6-D); the manifest validator is permissive (warning
        # severity) until Phase 7c promotes to critical.
        concept_graph_sha256_kw = kwargs.get("concept_graph_sha256") or ""
        # Phase 7c.5 SHIPPING BLOCKER: chunkset hashes threaded from the
        # ``chunking`` (DART) and ``imscc_chunking`` phase outputs via
        # the workflow runner's ``inputs_from`` chain. Phase 7c ST 17
        # promoted both fields to required at the validator boundary;
        # without this kwarg plumbing, every end-to-end ``ed4all run``
        # would fail validation at the ``libv2_archival`` gate. Mirrors
        # the ``concept_graph_sha256`` pattern above (Phase 6 ST 18).
        # Optional in legacy callers (no chunking phase wired) — None
        # falls through to the validator's ``MISSING_*`` critical.
        dart_chunks_sha256_kw = kwargs.get("dart_chunks_sha256") or ""
        imscc_chunks_sha256_kw = kwargs.get("imscc_chunks_sha256") or ""

        if not course_name:
            return json.dumps({"error": "archive_to_libv2 requires course_name"})

        slug = course_name.lower().replace("_", "-").replace(" ", "-")
        libv2_root = PROJECT_ROOT / "LibV2"
        course_dir = libv2_root / "courses" / slug

        for subdir in [
            "source/pdf", "source/html", "source/imscc",
            "corpus", "graph", "pedagogy", "training_specs", "quality"
        ]:
            (course_dir / subdir).mkdir(parents=True, exist_ok=True)

        archived = {
            "pdfs": [],
            "html": [],
            "imscc": None,
            "assessment": None,
            "trainforge": {
                "chunks": None,
                "graph": None,
                "misconceptions": None,
                "assessments": None,
                "quality_report": None,
            },
        }

        # --- Copy raw PDFs -------------------------------------------------
        if pdf_paths_str:
            for p in pdf_paths_str.split(","):
                src = Path(p.strip())
                if src.exists():
                    dest = course_dir / "source" / "pdf" / src.name
                    shutil.copy2(src, dest)
                    archived["pdfs"].append(str(dest))

        # --- Copy DART HTML outputs (+ adjacent .quality.json) -------------
        if html_paths_str:
            for p in html_paths_str.split(","):
                src = Path(p.strip())
                if src.exists():
                    dest = course_dir / "source" / "html" / src.name
                    shutil.copy2(src, dest)
                    archived["html"].append(str(dest))
                    quality_json = src.with_suffix(".quality.json")
                    if quality_json.exists():
                        shutil.copy2(
                            quality_json, course_dir / "quality" / quality_json.name
                        )
                    # Wave 19 (hotfix): archive ``{stem}_figures/`` sibling
                    # so orchestrated / CLI runs keep figure image refs
                    # intact. Mirrors the @mcp.tool() variant at L645.
                    figures_dir_src = src.parent / f"{src.stem}_figures"
                    if figures_dir_src.is_dir():
                        figures_dir_dest = (
                            course_dir / "source" / "html" / figures_dir_src.name
                        )
                        if figures_dir_dest.exists():
                            shutil.rmtree(figures_dir_dest)
                        shutil.copytree(figures_dir_src, figures_dir_dest)

        # --- Copy IMSCC package -------------------------------------------
        if imscc_path_str:
            src = Path(imscc_path_str)
            if src.exists():
                dest = course_dir / "source" / "imscc" / src.name
                shutil.copy2(src, dest)
                archived["imscc"] = str(dest)

        # --- Resolve Trainforge workspace ---------------------------------
        trainforge_dir: Optional[Path] = None

        if project_workspace_kw:
            candidate = Path(project_workspace_kw)
            if candidate.name != "trainforge":
                candidate = candidate / "trainforge"
            if candidate.exists() and candidate.is_dir():
                trainforge_dir = candidate

        if trainforge_dir is None and project_id_kw:
            candidate = (
                courseforge_exports_dir() / project_id_kw / "trainforge"
            )
            if candidate.exists() and candidate.is_dir():
                trainforge_dir = candidate

        # Legacy assessment_path handling: keep parity with the MCP-tool
        # variant so existing provenance / evidence flag tests pass
        # (they pass assessment_path=<chunks.jsonl>). If the path points at
        # a directory, treat it as the trainforge workspace root.
        if assessment_path_str:
            ap = Path(assessment_path_str)
            if ap.exists():
                if ap.is_dir():
                    if trainforge_dir is None:
                        trainforge_dir = ap
                else:
                    # Phase 7c: write to imscc_chunks/ (canonical).
                    dest = course_dir / "imscc_chunks" / ap.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ap, dest)
                    archived["assessment"] = str(dest)

        # Heuristic fallback: scan well-known locations for chunks.jsonl.
        # Phase 7c: imscc_chunks/ is canonical; corpus/ retained for back-compat.
        if trainforge_dir is None:
            candidates: list[Path] = []
            exports_root = courseforge_exports_dir()
            if exports_root.exists():
                for project_dir in exports_root.iterdir():
                    if not project_dir.is_dir():
                        continue
                    tf = project_dir / "trainforge"
                    if (
                        (tf / "chunks.jsonl").exists()
                        or (tf / "imscc_chunks" / "chunks.jsonl").exists()
                        or (tf / "corpus" / "chunks.jsonl").exists()
                    ):
                        candidates.append(tf)
            runs_root = PROJECT_ROOT / "state" / "runs"
            if runs_root.exists():
                for run_dir in runs_root.iterdir():
                    if not run_dir.is_dir():
                        continue
                    tf = run_dir / "trainforge"
                    if (
                        (tf / "chunks.jsonl").exists()
                        or (tf / "imscc_chunks" / "chunks.jsonl").exists()
                        or (tf / "corpus" / "chunks.jsonl").exists()
                    ):
                        candidates.append(tf)
            if candidates:
                def _chunks_mtime(p):
                    # Support flat, new (imscc_chunks/) and legacy (corpus/) layouts.
                    new_nested = p / "imscc_chunks" / "chunks.jsonl"
                    legacy_nested = p / "corpus" / "chunks.jsonl"
                    flat = p / "chunks.jsonl"
                    if new_nested.exists():
                        return new_nested.stat().st_mtime
                    if legacy_nested.exists():
                        return legacy_nested.stat().st_mtime
                    if flat.exists():
                        return flat.stat().st_mtime
                    return 0.0
                trainforge_dir = max(candidates, key=_chunks_mtime)

        # --- Copy Trainforge outputs --------------------------------------
        # Worker β writes in CourseProcessor's native nested layout
        # (trainforge/imscc_chunks/chunks.jsonl post-Phase-7c, or legacy
        # trainforge/corpus/chunks.jsonl, plus trainforge/graph/*.json).
        # We also check the flat layout for backward-compat with any
        # caller that mirrors the older stub's expected paths.
        def _pick(*candidates):
            for c in candidates:
                if c.exists() and c.is_file():
                    return c
            return None

        # Wave 74 fail-closed: never silently preserve a prior run's
        # chunks.jsonl under the same slug. If the destination already
        # exists, drop it before the copy block so we either install
        # fresh chunks below or end up with no chunks file (which is
        # the correct state for DART-only / Trainforge-skipped runs).
        # Phase 7c: clean BOTH the new imscc_chunks/ path AND the legacy
        # corpus/ path so a re-run on a partially-migrated archive
        # doesn't leave stale chunks behind in either location.
        _dest_chunks_path = course_dir / "imscc_chunks" / "chunks.jsonl"
        _legacy_dest_chunks_path = course_dir / "corpus" / "chunks.jsonl"
        _had_prior_chunks = (
            _dest_chunks_path.exists() or _legacy_dest_chunks_path.exists()
        )
        for _stale in (_dest_chunks_path, _legacy_dest_chunks_path):
            if _stale.exists():
                try:
                    _stale.unlink()
                except OSError as _exc:
                    logger.warning(
                        "archive_to_libv2: failed to remove prior-run "
                        "chunks.jsonl at %s: %s",
                        _stale,
                        _exc,
                    )

        if trainforge_dir is not None and trainforge_dir.exists():
            copy_map = [
                # Phase 7c: prefer imscc_chunks/ in source AND destination.
                (_pick(trainforge_dir / "imscc_chunks" / "chunks.jsonl",
                       trainforge_dir / "corpus" / "chunks.jsonl",
                       trainforge_dir / "chunks.jsonl"),
                 course_dir / "imscc_chunks" / "chunks.jsonl", "chunks"),
                (_pick(trainforge_dir / "graph" / "concept_graph_semantic.json",
                       trainforge_dir / "concept_graph_semantic.json"),
                 course_dir / "graph" / "concept_graph_semantic.json", "graph"),
                (_pick(trainforge_dir / "graph" / "misconceptions.json",
                       trainforge_dir / "misconceptions.json"),
                 course_dir / "graph" / "misconceptions.json", "misconceptions"),
                (_pick(trainforge_dir / "training_specs" / "assessments.json",
                       trainforge_dir / "assessments.json"),
                 course_dir / "training_specs" / "assessments.json", "assessments"),
                # Wave 30 Gap 3: new training_synthesis phase outputs.
                # These land under training_specs/ alongside assessments.json
                # so LibV2 archives + downstream export tooling have real
                # instruction + preference pairs to surface.
                (_pick(trainforge_dir / "training_specs" / "instruction_pairs.jsonl"),
                 course_dir / "training_specs" / "instruction_pairs.jsonl", "instruction_pairs"),
                (_pick(trainforge_dir / "training_specs" / "preference_pairs.jsonl"),
                 course_dir / "training_specs" / "preference_pairs.jsonl", "preference_pairs"),
                (_pick(trainforge_dir / "training_specs" / "dataset_config.json"),
                 course_dir / "training_specs" / "dataset_config.json", "dataset_config"),
                # Wave 30 Gap 4: course.json is now written unconditionally
                # (including an empty-LOs shell) so LibV2 retrieval + joins
                # always have a file to look at.
                (_pick(trainforge_dir / "course.json"),
                 course_dir / "course.json", "course_json"),
                (_pick(trainforge_dir / "quality" / "quality_report.json"),
                 course_dir / "quality" / "quality_report.json", "quality_report"),
            ]
            for src, dest, label in copy_map:
                if src is not None and src.exists() and src.is_file():
                    try:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
                        archived["trainforge"][label] = str(dest)
                    except OSError as exc:
                        logger.warning(
                            f"archive_to_libv2: failed to copy {src} -> {dest}: {exc}"
                        )
        else:
            logger.warning(
                "archive_to_libv2: no Trainforge output dir located for "
                f"course {course_name} — features flags will default to false."
            )

        # --- Wave 74 fail-closed: chunks-freshness gate -------------------
        # When a chunks.jsonl exists at the archive destination, it MUST
        # carry IDs from this run's course_code. Otherwise we caught a
        # leak from a prior run under the same slug — refuse to write
        # the manifest and surface ``error_code = TRAINFORGE_OUTPUT_STALE``
        # to the caller. When the destination has no chunks file (the
        # Trainforge-intentionally-absent case — e.g. DART-only batches
        # gated by ``--no-assessments``), this check is a no-op.
        _chunks_check = _check_chunks_freshness(
            chunks_path=_dest_chunks_path,
            course_name=course_name,
            run_start_ts=_run_start_ts,
            had_prior_chunks=_had_prior_chunks,
        )
        if _chunks_check["status"] == "stale":
            logger.error(
                "archive_to_libv2: refusing to write manifest — "
                "chunks.jsonl at %s is stale for course %s (%s).",
                _dest_chunks_path,
                course_name,
                _chunks_check["reason"],
            )
            return json.dumps({
                "success": False,
                "error": _chunks_check["reason"],
                "error_code": "TRAINFORGE_OUTPUT_STALE",
                "course_name": course_name,
                "chunks_path": str(_dest_chunks_path),
                "expected_prefix": _chunks_check.get("expected_prefix"),
                "observed_prefixes": _chunks_check.get("observed_prefixes"),
            })

        # --- Build manifest (with source_artifacts checksums) -------------
        import hashlib

        def _sha256(filepath: Path) -> str:
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                for block in iter(lambda: f.read(8192), b""):
                    h.update(block)
            return h.hexdigest()

        source_artifacts: dict = {}
        if archived["pdfs"]:
            source_artifacts["pdf"] = [
                {"path": p, "checksum": _sha256(Path(p)), "size": Path(p).stat().st_size}
                for p in archived["pdfs"]
            ]
        if archived["html"]:
            source_artifacts["html"] = [
                {"path": p, "checksum": _sha256(Path(p)), "size": Path(p).stat().st_size}
                for p in archived["html"]
            ]
        if archived["imscc"]:
            imscc_p = Path(archived["imscc"])
            source_artifacts["imscc"] = {
                "path": archived["imscc"],
                "checksum": _sha256(imscc_p),
                "size": imscc_p.stat().st_size,
            }

        # Wave 10 / Wave 11 feature flags — scan the archived files.
        source_provenance_flag = _detect_source_provenance(course_dir)
        evidence_source_provenance_flag = _detect_evidence_source_provenance(course_dir)

        # Phase 6 ST 18: resolve concept_graph_sha256 (per plan §D).
        # Resolution order:
        #   1. Explicit kwarg (workflow runner threads via inputs_from).
        #   2. Recompute from on-disk concept_graph_semantic.json under the
        #      LibV2 course's concept_graph/ subdir (Worker C-J's
        #      _run_concept_extraction writes here at ST 12).
        #   3. None — legacy archive without a concept graph (warning gate
        #      handles this in libv2_manifest validator).
        import re as _re_cg
        concept_graph_sha256_resolved: Optional[str] = None
        if concept_graph_sha256_kw and _re_cg.match(
            r"^[0-9a-f]{64}$", concept_graph_sha256_kw,
        ):
            concept_graph_sha256_resolved = concept_graph_sha256_kw
        else:
            cg_path = (
                course_dir / "concept_graph" / "concept_graph_semantic.json"
            )
            if cg_path.exists() and cg_path.is_file():
                try:
                    concept_graph_sha256_resolved = hashlib.sha256(
                        cg_path.read_bytes()
                    ).hexdigest()
                except OSError as exc:
                    logger.warning(
                        "archive_to_libv2: failed to hash concept graph "
                        "at %s: %s",
                        cg_path,
                        exc,
                    )

        manifest = {
            "libv2_version": "1.2.0",
            "chunker_version": _resolve_chunker_version(),
            "slug": slug,
            "import_timestamp": datetime.now().isoformat(),
            "classification": {
                "division": division,
                "primary_domain": domain,
                "subdomains": [s.strip() for s in subdomains_str.split(",")]
                if subdomains_str else [],
            },
            "source_artifacts": source_artifacts,
            "provenance": {
                "source_type": "textbook_to_course_pipeline",
                "import_pipeline_version": "1.0.0",
            },
            "features": {
                "source_provenance": source_provenance_flag,
                "evidence_source_provenance": evidence_source_provenance_flag,
            },
        }
        if concept_graph_sha256_resolved is not None:
            manifest["concept_graph_sha256"] = concept_graph_sha256_resolved

        # Phase 7c.5 SHIPPING BLOCKER: persist the chunkset hashes routed
        # via the workflow runner's ``inputs_from`` chain (chunking →
        # dart_chunks_sha256, imscc_chunking → imscc_chunks_sha256). Same
        # 64-hex regex shape as ``concept_graph_sha256``; only emit when
        # the kwarg is well-formed so a malformed upstream value falls
        # through to the validator's ``MISSING_*`` critical (the
        # validator owns ``INVALID_*`` shape diagnostics) rather than
        # being silently propagated.
        if dart_chunks_sha256_kw and _re_cg.match(
            r"^[0-9a-f]{64}$", dart_chunks_sha256_kw,
        ):
            manifest["dart_chunks_sha256"] = dart_chunks_sha256_kw
        if imscc_chunks_sha256_kw and _re_cg.match(
            r"^[0-9a-f]{64}$", imscc_chunks_sha256_kw,
        ):
            manifest["imscc_chunks_sha256"] = imscc_chunks_sha256_kw

        # GPT Feedback v2 (May 12 / item 3): lineage mirror.
        # Build a sorted, deduped index of every upstream source-document
        # SHA from source_artifacts. Lets an auditor read the manifest
        # alone and see the full "what bytes produced this course" list
        # without walking source_artifacts.pdf[].checksum + .imscc.checksum
        # subtrees by hand.
        _src_doc_shas: set = set()
        for entry in source_artifacts.get("pdf", []) or []:
            cs = entry.get("checksum") if isinstance(entry, dict) else None
            if isinstance(cs, str) and _re_cg.match(r"^[0-9a-f]{64}$", cs):
                _src_doc_shas.add(cs)
        for entry in source_artifacts.get("html", []) or []:
            cs = entry.get("checksum") if isinstance(entry, dict) else None
            if isinstance(cs, str) and _re_cg.match(r"^[0-9a-f]{64}$", cs):
                _src_doc_shas.add(cs)
        imscc_entry = source_artifacts.get("imscc")
        if isinstance(imscc_entry, dict):
            cs = imscc_entry.get("checksum")
            if isinstance(cs, str) and _re_cg.match(r"^[0-9a-f]{64}$", cs):
                _src_doc_shas.add(cs)
        if _src_doc_shas:
            manifest["source_documents_sha256_index"] = sorted(_src_doc_shas)

        # GPT Feedback v2 (May 12 / item 3): copy lineage fields from
        # the on-disk concept_graph_semantic.json onto the manifest.
        # Best-effort: missing / unreadable / pre-May-2026 graph → fields
        # omitted (manifest tolerates absence). The graph itself is the
        # source of truth; the manifest mirror exists so an auditor
        # reading the manifest alone sees the lineage without opening
        # the graph artifact.
        _cg_path = course_dir / "graph" / "concept_graph_semantic.json"
        if _cg_path.exists() and _cg_path.is_file():
            try:
                _cg_json = json.loads(_cg_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as _exc:
                logger.debug(
                    "archive_to_libv2: failed to read concept graph for "
                    "lineage mirror (%s); skipping.",
                    _exc,
                )
            else:
                if isinstance(_cg_json, dict):
                    _gbh = _cg_json.get("graph_build_hash")
                    if isinstance(_gbh, str) and _re_cg.match(
                        r"^[a-f0-9]{64}$", _gbh,
                    ):
                        manifest["graph_build_hash"] = _gbh
                    _rpv = _cg_json.get("rulepack_version")
                    if isinstance(_rpv, str) and _re_cg.match(
                        r"^v[0-9a-f]+$", _rpv,
                    ):
                        manifest["rulepack_version"] = _rpv
                    if "course_package_version" in _cg_json:
                        _cpv = _cg_json["course_package_version"]
                        if _cpv is None or (
                            isinstance(_cpv, str) and _cpv
                        ):
                            manifest["course_package_version"] = _cpv

        manifest_path = course_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # F6: register the course in the LibV2 master catalog so that
        # ``libv2 catalog list`` / ``libv2 info <slug>`` see it immediately
        # without a separate manual ``index rebuild`` step.
        try:
            from LibV2.tools.libv2.catalog import _register_course_in_catalog
            _register_course_in_catalog(slug, manifest, libv2_root)
        except Exception as _exc:
            logger.warning(
                "archive_to_libv2: catalog registration failed for %s: %s",
                slug, _exc,
            )

        return json.dumps({
            "success": True,
            "course_slug": slug,
            "course_dir": str(course_dir),
            "manifest_path": str(manifest_path),
            "archived": archived,
            "features": {
                "source_provenance": source_provenance_flag,
                "evidence_source_provenance": evidence_source_provenance_flag,
            },
            "trainforge_workspace": (
                str(trainforge_dir) if trainforge_dir is not None else None
            ),
            "artifact_counts": {
                "pdfs": len(archived["pdfs"]),
                "html_files": len(archived["html"]),
                "imscc": 1 if archived["imscc"] else 0,
                "assessment": 1 if archived["assessment"] else 0,
                "trainforge": sum(
                    1 for v in archived["trainforge"].values() if v is not None
                ),
            },
        })

    registry["archive_to_libv2"] = _archive_to_libv2
    # END BLOCK: Worker γ

    async def _build_source_module_map(**kwargs):
        """Source-router (Wave 9 ``source_mapping`` phase) — real heuristic.

        Previously wrote an empty ``source_module_map.json``, which left
        every Courseforge page emitted without ``sourceReferences[]`` and
        pinned the ``source_provenance`` / ``evidence_source_provenance``
        feature flags to false (investigation Issue 7). This implementation
        routes DART source blocks to Courseforge pages via keyword-overlap
        scoring:

          1. Enumerate DART block IDs by scanning ``staging_dir`` for
             ``*_synthesized.json`` sidecars — each ``sections[]`` entry
             contributes ``section_id``, ``section_title``, and any
             keyword-bearing text in ``data`` / ``sources_used``.
          2. Load the textbook structure (when available) and the
             project's objectives to enumerate per-page target topics.
          3. For each week (1..duration_weeks) and each page role
             (overview, content_0K, application, self_check, summary),
             score DART blocks by keyword overlap with the page's
             dominant topic. Blocks above a stronger threshold become
             ``primary`` refs; blocks above a weaker threshold become
             ``contributing`` refs.
          4. Emit the map in the Wave 9 shape that
             ``Courseforge.scripts.generate_course._page_refs_for``
             consumes: ``{week_key: {page_id: {primary, contributing,
             confidence}}}`` using ``dart:{slug}#{block_id}`` source IDs.

        No LLM. Pure text overlap — imperfect but deterministic and
        better than an empty map for provenance propagation.
        """
        project_id = kwargs.get("project_id", "")
        staging_dir_kw = kwargs.get("staging_dir", "") or ""
        textbook_structure_path = kwargs.get("textbook_structure_path", "") or ""

        if not project_id:
            return json.dumps({"error": "source-router requires project_id"})

        project_path = courseforge_exports_dir() / project_id
        project_path.mkdir(parents=True, exist_ok=True)
        map_path = project_path / "source_module_map.json"

        # ------------------------------------------------------------- #
        # Load project config for duration_weeks + course_name.          #
        # ------------------------------------------------------------- #
        config_path = project_path / "project_config.json"
        duration_weeks = 12
        course_name = project_id
        objectives_path: Optional[str] = None
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                duration_weeks = int(cfg.get("duration_weeks") or 12)
                course_name = cfg.get("course_name") or project_id
                objectives_path = cfg.get("objectives_path") or None
            except (OSError, ValueError):
                pass

        # ------------------------------------------------------------- #
        # Enumerate DART source blocks from staging_dir sidecars.        #
        # Each entry: {block_id, slug, keywords(set[str]), title}.       #
        # ------------------------------------------------------------- #
        dart_blocks: list = []
        staging_dir = Path(staging_dir_kw) if staging_dir_kw else None
        if staging_dir is None or not staging_dir.exists():
            # Fallback: scan Courseforge inputs for any synthesized sidecars.
            staging_dir = COURSEFORGE_INPUTS

        def _tokenize(text: str) -> set:
            """Lowercase, strip punctuation, drop stopwords + short tokens."""
            if not text:
                return set()
            import re as _re
            cleaned = _re.sub(r"[^a-z0-9\s]", " ", text.lower())
            _stopwords = {
                "the", "and", "for", "with", "from", "that", "this", "are",
                "was", "were", "has", "have", "had", "but", "not", "all",
                "any", "may", "can", "one", "two", "its", "their", "they",
                "will", "been", "you", "your", "our", "his", "her", "which",
                "what", "who", "why", "how", "when", "where", "into", "out",
                "over", "such", "more", "most", "some", "about", "there",
                "these", "those", "than", "then", "also", "only", "used",
                "use", "see", "via", "per",
            }
            return {
                t for t in cleaned.split()
                if len(t) > 3 and t not in _stopwords
            }

        if staging_dir and staging_dir.exists():
            for sidecar in sorted(staging_dir.rglob("*_synthesized.json")):
                try:
                    doc = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                # Wave 36: match ContentGroundingValidator + Wave 35
                # content-generator slug rules (lowercase + space→hyphen).
                # Pre-Wave-36 a staging stem like ``XYZ_201_synthesized``
                # emitted router refs as ``dart:XYZ_201#...`` while the
                # validator + content-generator lowercased, so
                # uppercase-named corpora silently failed the source_refs
                # gate.
                slug = (
                    sidecar.stem.replace("_synthesized", "")
                    .lower()
                    .replace(" ", "-")
                )
                sections = doc.get("sections") or []
                if not isinstance(sections, list):
                    continue
                for section in sections:
                    if not isinstance(section, dict):
                        continue
                    block_id = str(section.get("section_id") or "").strip()
                    if not block_id:
                        continue
                    title = str(section.get("section_title") or "").strip()
                    section_type = str(section.get("section_type") or "").strip()
                    # Gather text for keyword extraction: title + any
                    # paragraph text + key-value block labels + data keys.
                    text_bits: list = [title, section_type]
                    data = section.get("data")
                    if isinstance(data, dict):
                        for k, v in data.items():
                            text_bits.append(str(k))
                            if isinstance(v, str):
                                text_bits.append(v)
                            elif isinstance(v, list):
                                for item in v[:20]:
                                    if isinstance(item, str):
                                        text_bits.append(item)
                                    elif isinstance(item, dict):
                                        for sub_v in item.values():
                                            if isinstance(sub_v, str):
                                                text_bits.append(sub_v)
                    keywords = _tokenize(" ".join(text_bits))
                    if not keywords:
                        # Fall back to splitting the block id so at least
                        # the title contributes a scoring signal.
                        keywords = _tokenize(title) or _tokenize(slug)
                    dart_blocks.append({
                        "block_id": block_id,
                        "slug": slug,
                        "title": title,
                        "keywords": keywords,
                        "source_id": f"dart:{slug}#{block_id}",
                    })

        # ------------------------------------------------------------- #
        # Enumerate per-week topics. Preference order:                   #
        #   1. textbook_structure_path chapters/sections                 #
        #   2. objectives_path chapter/terminal objective statements     #
        #   3. DART block titles themselves (round-robin by week)        #
        # ------------------------------------------------------------- #
        week_topics: dict = {}  # week_num -> {page_id: set[str]}

        def _set_week_page(week_num: int, page_id: str, kw: set):
            week_topics.setdefault(week_num, {})[page_id] = kw

        structure_chapters: list = []
        if textbook_structure_path:
            sp = Path(textbook_structure_path)
            if sp.exists():
                try:
                    structure_doc = json.loads(sp.read_text(encoding="utf-8"))
                    chapters = structure_doc.get("chapters") or []
                    if isinstance(chapters, list):
                        structure_chapters = chapters
                except (OSError, ValueError):
                    pass

        objective_statements: list = []
        if objectives_path:
            op = Path(objectives_path)
            if op.exists():
                try:
                    obj_doc = json.loads(op.read_text(encoding="utf-8"))
                    for group in ("chapter_objectives", "terminal_objectives",
                                  "course_objectives"):
                        for item in obj_doc.get(group, []) or []:
                            if isinstance(item, dict):
                                text = (
                                    item.get("statement")
                                    or item.get("description")
                                    or item.get("text")
                                    or ""
                                )
                                if text:
                                    objective_statements.append(text)
                except (OSError, ValueError):
                    pass

        # Assemble per-week keyword bags. Wave 24 HIGH-5 fix: page roles
        # now scale with the week's LO count via _page_roles_for_week.
        # When objectives aren't loaded yet (source-router runs before
        # course_planning in some paths), fall back to the legacy 5-tuple.
        from MCP.tools._content_gen_helpers import _page_roles_for_week  # noqa: E402
        # Derive a per-week LO count: prefer objective_statements when
        # synthesized, else use structure chapters, else default to 4
        # (yields the legacy 5-page shape via _page_roles_for_week).
        if objective_statements:
            base_lo_count = max(1, len(objective_statements) // max(1, duration_weeks))
        elif structure_chapters:
            base_lo_count = max(1, len(structure_chapters) // max(1, duration_weeks) + 1)
        else:
            base_lo_count = 4
        page_roles = _page_roles_for_week(base_lo_count)

        # Prefer chapters / objective statements when available.
        topic_pool: list = []
        for ch in structure_chapters:
            if isinstance(ch, dict):
                # SemanticStructureExtractor emits the heading under
                # ``headingText`` (only legacy callers used ``title``), plus
                # the full inter-heading prose under ``chapter_text`` /
                # ``section_text``. Reading the wrong key here silently
                # collapsed the entire router: every chapter bag tokenized
                # to ∅, so every page hit the degenerate empty-bag fallback
                # and was pinned to one alphabetically-first source file.
                ch_topics = [
                    str(ch.get("headingText") or ch.get("title") or ""),
                    str(ch.get("chapter_text") or "")[:2000],
                ]
                for sub in ch.get("sections") or []:
                    if isinstance(sub, dict):
                        ch_topics.append(
                            str(sub.get("headingText") or sub.get("title") or "")
                        )
                        ch_topics.append(str(sub.get("section_text") or "")[:1000])
                        for ss in sub.get("subsections") or []:
                            if isinstance(ss, dict):
                                ch_topics.append(
                                    str(ss.get("headingText")
                                        or ss.get("title") or "")
                                )
                    elif isinstance(sub, str):
                        ch_topics.append(sub)
                topic_pool.append(_tokenize(" ".join(ch_topics)))
        # Fall through when the structure produced NO usable keyword bags
        # (every chapter tokenized to nothing). ``not any(...)`` — not
        # ``not topic_pool`` — so a list of empty sets still degrades to the
        # objectives / DART-block signal instead of collapsing the router.
        if not any(topic_pool) and objective_statements:
            topic_pool = [_tokenize(stmt) for stmt in objective_statements]
        if not any(topic_pool) and dart_blocks:
            # Final fallback: let DART block titles/keywords drive topic
            # bags, one per block, so each week gets a real signal.
            topic_pool = [blk["keywords"] for blk in dart_blocks]

        # Distribute topic_pool across weeks by SLICING, not single-index
        # round-robin. With many chapters and few weeks, the old
        # ``topic_pool[(week-1) % len]`` only ever consumed the first
        # ``duration_weeks`` bags — clustering every week onto the opening
        # source file. Slicing unions each week's contiguous share of the
        # material so weeks track the corpus in reading order and every
        # source file contributes to some week.
        non_empty_pool = [b for b in topic_pool if b] or topic_pool
        n = len(non_empty_pool)
        per_week = (-(-n // max(1, duration_weeks))) if n else 0  # ceil div
        for week_num in range(1, duration_weeks + 1):
            if not non_empty_pool:
                primary_bag: set = set()
            else:
                start = (week_num - 1) * per_week
                week_slice = non_empty_pool[start:start + per_week]
                if not week_slice:
                    # Past the end of the pool — reuse a wrapped bag so
                    # trailing weeks still carry a real signal.
                    week_slice = [non_empty_pool[(week_num - 1) % n]]
                primary_bag = set().union(*week_slice)
            for page_id in page_roles:
                _set_week_page(week_num, page_id, set(primary_bag))

        # ------------------------------------------------------------- #
        # Score blocks per (week, page) and emit refs.                   #
        # ------------------------------------------------------------- #
        source_module_map: dict = {}
        chunk_ids: set = set()

        if dart_blocks:
            for week_num in range(1, duration_weeks + 1):
                week_key = f"week_{week_num:02d}"
                pages_for_week = week_topics.get(week_num, {})
                week_entries: dict = {}
                for page_id, target_bag in pages_for_week.items():
                    if not target_bag:
                        # Wave 84 fix: degenerate fallback (no topic bag)
                        # used to round-robin a DART block as PRIMARY at
                        # confidence 0.3. That stamped a low-confidence
                        # alphabetically-first block on every page in the
                        # course, masking actually-relevant sources from
                        # data-cf-source-ids. Now we emit it as
                        # ``contributing`` so any genuine primary from
                        # the content-generator's grounding takes precedence.
                        fallback = dart_blocks[(week_num - 1) % len(dart_blocks)]
                        week_entries[page_id] = {
                            "primary": [],
                            "contributing": [fallback["source_id"]],
                            "confidence": 0.3,
                        }
                        chunk_ids.add(fallback["source_id"])
                        continue
                    scored: list = []
                    for blk in dart_blocks:
                        overlap = len(target_bag & blk["keywords"])
                        if overlap == 0:
                            continue
                        # Jaccard-ish score for ranking stability.
                        union = max(1, len(target_bag | blk["keywords"]))
                        score = overlap / union
                        scored.append((score, overlap, blk))
                    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
                    primary_ids: list = []
                    contributing_ids: list = []
                    top_score = scored[0][0] if scored else 0.0
                    # Wave 84: only emit primary refs when the top-scoring
                    # block clears a confidence floor (0.15 Jaccard). Below
                    # that floor the router can't tell which block is
                    # primary, so it cedes the role to the content-
                    # generator's data-cf-source-primary attribute (which
                    # picks the actual source the LLM used). Lower-scoring
                    # blocks still ride along as contributing so the page
                    # has provenance breadth.
                    PRIMARY_CONFIDENCE_FLOOR = 0.15
                    for score, overlap, blk in scored:
                        if (
                            score >= max(PRIMARY_CONFIDENCE_FLOOR, top_score * 0.8)
                            and len(primary_ids) < 2
                        ):
                            primary_ids.append(blk["source_id"])
                        elif score >= 0.05 and len(contributing_ids) < 3:
                            contributing_ids.append(blk["source_id"])
                    if not primary_ids and scored:
                        # Top score is below the floor → emit the top match
                        # as CONTRIBUTING (not primary). That preserves
                        # provenance breadth without polluting the primary
                        # role with a guess.
                        candidate = scored[0][2]["source_id"]
                        if candidate not in contributing_ids and len(contributing_ids) < 3:
                            contributing_ids.append(candidate)
                    if not primary_ids and not contributing_ids:
                        # Wave 84: no overlap at all — round-robin a DART
                        # block as CONTRIBUTING (was primary) so a chunk
                        # always has some provenance for trace, but the
                        # primary slot stays open for content-generator
                        # grounding.
                        fallback = dart_blocks[(week_num - 1) % len(dart_blocks)]
                        contributing_ids.append(fallback["source_id"])
                        top_score = 0.2
                    for sid in primary_ids:
                        chunk_ids.add(sid)
                    for sid in contributing_ids:
                        chunk_ids.add(sid)
                    week_entries[page_id] = {
                        "primary": primary_ids,
                        "contributing": contributing_ids,
                        "confidence": round(max(top_score, 0.2), 2),
                    }
                if week_entries:
                    source_module_map[week_key] = week_entries

        map_path.write_text(
            json.dumps(source_module_map, indent=2),
            encoding="utf-8",
        )

        routing_mode = (
            "keyword_overlap_heuristic" if dart_blocks
            else "stub_empty_map"
        )

        # ------------------------------------------------------------- #
        # Anti-silent-degradation guard. A "collapsed" routing run is one #
        # where NO page earned a real ``primary`` ref — every page fell  #
        # back to a low-confidence round-robin ``contributing`` block,   #
        # pinning the whole course to one source file. Pre-fix this      #
        # happened silently (wrong structure key -> empty topic bags).   #
        # Also flag a thin run where almost every page lacks a primary.  #
        # ------------------------------------------------------------- #
        total_pages = sum(len(v) for v in source_module_map.values())
        pages_with_primary = sum(
            1
            for v in source_module_map.values()
            for e in v.values()
            if e.get("primary")
        )
        distinct_sources = len({
            sid.split("#", 1)[0]
            for v in source_module_map.values()
            for e in v.values()
            for sid in (list(e.get("primary") or []) + list(e.get("contributing") or []))
        })
        primary_rate = (pages_with_primary / total_pages) if total_pages else 0.0
        routing_collapsed = (
            bool(dart_blocks) and total_pages > 0 and pages_with_primary == 0
        )
        if routing_collapsed:
            logger.error(
                "SOURCE ROUTER COLLAPSE (course=%s): 0/%d pages earned a "
                "primary source ref; every page fell back to a low-confidence "
                "round-robin block, so the course will be mis-sourced. Likely "
                "cause: textbook_structure chapters carry no usable heading "
                "text (check the ``headingText`` key), or staging "
                "``*_synthesized.json`` sidecars are missing. "
                "dart_blocks_indexed=%d distinct_sources=%d.",
                course_name, total_pages, len(dart_blocks), distinct_sources,
            )
        elif total_pages and primary_rate < 0.25:
            logger.warning(
                "Source router thin coverage (course=%s): only %d/%d pages "
                "(%.0f%%) earned a primary source ref. Provenance may be weak.",
                course_name, pages_with_primary, total_pages, primary_rate * 100,
            )

        return json.dumps({
            "source_module_map_path": str(map_path),
            "source_chunk_ids": sorted(chunk_ids),
            "staging_dir": str(staging_dir) if staging_dir else "",
            "textbook_structure_path": textbook_structure_path,
            "routing_mode": routing_mode,
            "dart_blocks_indexed": len(dart_blocks),
            "weeks_routed": len(source_module_map),
            "course_name": course_name,
            "routing_collapsed": routing_collapsed,
            "routing_primary_rate": round(primary_rate, 3),
            "pages_with_primary": pages_with_primary,
            "total_pages": total_pages,
            "distinct_sources_routed": distinct_sources,
        })

    registry["build_source_module_map"] = _build_source_module_map

    # ============================================================================
    # Phase 6 Subtask 12: _run_concept_extraction — concept-graph builder
    #
    # New ``concept_extraction`` workflow phase (between ``source_mapping``
    # and ``course_planning`` per plan ST 11). The phase runs the
    # ``Trainforge.pedagogy_graph_builder.build_pedagogy_graph`` over a
    # canonical v4 chunkset (DART chunks emitted by the upstream
    # ``chunking`` phase, falling back to a minimal inline projection of
    # ``*_synthesized.json`` sidecars when the upstream chunkset is not
    # available), persists the resulting graph to
    # ``LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json``
    # plus a sibling ``manifest.json``, computes the SHA-256 of the graph
    # bytes, and surfaces both the path and hash through phase outputs.
    #
    # Phase 7b ST 14.5 architectural reconciliation: when the upstream
    # ``chunking`` phase (Phase 7b ST 11, ``_run_dart_chunking``) has
    # already emitted ``LibV2/courses/<slug>/dart_chunks/chunks.jsonl``,
    # this helper consumes that chunkset directly via the ``dart_chunks_path``
    # kwarg threaded through the workflow's ``inputs_from`` wiring. This
    # eliminates the divergence-risk surface of two parallel chunk-shaping
    # paths (the upstream chunker invokes ``Trainforge.chunker.chunk_content``
    # — the canonical surface; the legacy inline-projection here was a
    # workaround for the Phase 6 phase ordering, where no IMSCC existed
    # yet). The inline-projection path is preserved as a back-compat
    # fallback for legacy / pre-Phase-7b runs that bypass the ``chunking``
    # phase, and for unit tests that exercise ``_run_concept_extraction``
    # in isolation.
    # ============================================================================
    async def _run_concept_extraction(**kwargs):
        """Build the genuine semantic concept graph over staged DART output.

        Fix-2: this phase emits the genuine ``kind: "concept_semantic"``
        graph (``DomainConcept`` nodes + typed edges via
        ``build_semantic_graph``), replacing the prior lightweight
        ``kind: "pedagogy"`` graph from ``build_pedagogy_graph``.

        Required kwargs (resolved via inputs_from in workflows.yaml):
            project_id: Courseforge project (used only for course_name lookup)
            course_name: Canonical course name (defaults to project config)
            staging_dir: DART staging directory (sibling to objective_extraction)

        Optional kwargs (Phase 7b ST 14.5 — consume upstream chunkset):
            dart_chunks_path: Path to ``LibV2/courses/<slug>/dart_chunks/chunks.jsonl``
                emitted by the upstream ``chunking`` phase. When provided
                and readable, this helper loads chunks from the JSONL
                file and skips the legacy inline-projection. When absent
                or unreadable, the inline-projection path runs as a
                back-compat fallback.

        Optional kwargs (objectives resolution for LO-driven typed edges):
            objectives_path: the ``--reuse-objectives`` JSON (resolution
                candidate #1; a reuse run pins the typed-edge rules to the
                operator's verbatim objectives doc).
            synthesized_objectives_path: the
                ``synthesized_objectives.json`` emitted by
                ``course_planning`` (resolution candidate #2, the
                fresh-run source). Phase-ordering fix (Option A1):
                ``concept_extraction`` now runs AFTER ``course_planning``,
                so on a fresh ``textbook_to_course`` run this candidate
                exists and gives the LO-driven typed-edge rules
                (prerequisite_from_lo_order, targets_concept_from_lo) a
                real learning-outcomes ordering. The concept-objective
                linker (relocated here from ``_plan_course_structure``)
                then enriches ``LearningObjective.key_concepts[]`` from the
                concept graph before ``build_semantic_graph`` runs, and the
                enriched key_concepts are written back to the project-export
                ``synthesized_objectives.json`` (never to a reuse path).

        Outputs (returned + persisted):
            concept_graph_path: ``LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json``
            concept_graph_sha256: SHA-256 hex digest of the graph file bytes
        """
        import hashlib as _hashlib

        project_id = kwargs.get("project_id") or ""
        course_name = kwargs.get("course_name") or ""
        staging_dir_kw = kwargs.get("staging_dir") or ""
        dart_chunks_path_kw = kwargs.get("dart_chunks_path") or ""

        # Resolve project + course_name from project_config when present.
        config_data: Dict[str, Any] = {}
        project_path: Optional[Path] = None
        if project_id:
            cand = courseforge_exports_dir() / project_id
            if cand.exists():
                project_path = cand
                cfg_path = cand / "project_config.json"
                if cfg_path.exists():
                    try:
                        config_data = json.loads(
                            cfg_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError):
                        config_data = {}
        course_name = (
            course_name
            or config_data.get("course_name")
            or project_id
            or "UNKNOWN"
        )

        # Resolve staging dir. Honor explicit kwarg first; fall back to
        # any project-config-recorded staging dir; finally walk
        # COURSEFORGE_INPUTS for any *_synthesized.json sidecars (degraded
        # path for legacy fixtures that bypass the staging phase).
        staging_dir: Optional[Path] = None
        if staging_dir_kw:
            cand = Path(staging_dir_kw)
            if cand.exists():
                staging_dir = cand
        if staging_dir is None:
            cfg_staging = config_data.get("staging_dir")
            if cfg_staging:
                cand = Path(cfg_staging)
                if cand.exists():
                    staging_dir = cand
        if staging_dir is None and COURSEFORGE_INPUTS.exists():
            staging_dir = COURSEFORGE_INPUTS

        course_slug = course_name.lower().replace("_", "-").replace(" ", "-")
        course_code_lower = course_name.lower().replace("-", "_")
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 1

        # M2 fix: track inline-projection drops so the phase output
        # carries a ``projection_drops_count`` and a structured
        # ``concept_projection_drop`` decision event fires per skip.
        # Pre-fix the per-section / per-sidecar drops were logger-only
        # and operators could not tell from the workflow report whether
        # extraction silently dropped concepts before
        # ``build_pedagogy_graph`` was even called.
        projection_drops: List[Dict[str, Any]] = []
        _capture_concept = None
        try:
            from lib.decision_capture import DecisionCapture as _DC_concept
            _capture_concept = _DC_concept(
                course_code=course_name.upper() or "concept_extraction",
                phase="content-analysis",
                tool="trainforge",
                streaming=True,
            )
        except Exception as exc:  # noqa: BLE001 — capture is best-effort
            logger.warning(
                "concept_extraction: DecisionCapture init failed (%s); "
                "projection drops will be tracked but not persisted.",
                exc,
            )
            _capture_concept = None

        def _record_projection_drop(reason: str, **fields: Any) -> None:
            """Append a structured drop record + emit a decision event.

            ``reason`` is a stable enum-ish slug (``malformed_sidecar``,
            ``non_list_sections``, ``non_dict_section``,
            ``missing_section_id``, ``malformed_upstream_chunk_jsonl``).
            Extra context fields are stamped onto both the drop record
            and the decision event's ``ml_features`` so a postmortem
            reader can stratify by reason without re-running.
            """
            drop_record = {"reason": reason, **fields}
            projection_drops.append(drop_record)
            if _capture_concept is None:
                return
            try:
                _capture_concept.log_decision(
                    decision_type="concept_projection_drop",
                    decision=f"dropped_section reason={reason}",
                    rationale=(
                        f"concept_extraction inline projection dropped a "
                        f"section before build_pedagogy_graph could "
                        f"consume it. reason={reason}; context={fields!r}. "
                        f"Surfaced via projection_drops_count in phase "
                        f"output so downstream operators see the silent-"
                        f"degradation class instead of relying on the "
                        f"min_edge_count gate to catch a thin graph."
                    ),
                    ml_features={"reason": reason, **{
                        k: v for k, v in fields.items()
                        if isinstance(v, (str, int, float, bool))
                    }},
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "concept_extraction: log_decision(concept_projection_drop) "
                    "failed (%s); drop still recorded in projection_drops.",
                    exc,
                )

        # Phase 7b ST 14.5: consume upstream DART chunkset when present.
        # The ``chunking`` phase (Phase 7b ST 11) emits
        # ``LibV2/courses/<slug>/dart_chunks/chunks.jsonl`` via the
        # canonical ``Trainforge.chunker.chunk_content`` path; the workflow
        # YAML's ``inputs_from`` block threads that path here. When the
        # path is provided AND readable, load the canonical v4 chunks
        # and skip the legacy inline-projection below. When absent or
        # unreadable, fall through to the inline-projection so legacy
        # / pre-Phase-7b fixtures (and unit tests that exercise this
        # helper in isolation) keep working.
        upstream_chunks_loaded = False
        if dart_chunks_path_kw:
            cand_chunks_path = Path(dart_chunks_path_kw)
            if cand_chunks_path.exists() and cand_chunks_path.is_file():
                try:
                    with cand_chunks_path.open("r", encoding="utf-8") as fh:
                        for _line_no, line in enumerate(fh, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                chunks.append(json.loads(line))
                            except ValueError as _ve:
                                # M2 fix: surface the drop so the
                                # silent-skip is visible to operators.
                                _record_projection_drop(
                                    "malformed_upstream_chunk_jsonl",
                                    source=str(cand_chunks_path),
                                    line=_line_no,
                                    parse_error=str(_ve),
                                )
                                continue
                    upstream_chunks_loaded = True
                except OSError as exc:
                    logger.warning(
                        "Phase 7b ST 14.5: failed to read upstream dart_chunks_path "
                        "%s (%s); falling back to inline-projection.",
                        cand_chunks_path, exc,
                    )

        # Legacy inline-projection fallback. Builds a minimal v4 chunk
        # projection from each ``*_synthesized.json`` sidecar so the
        # co-occurrence + semantic graph builders have populated
        # ``concept_tags`` + ``source.module_id`` / ``item_path`` to walk.
        # One chunk per DART section keeps wall-time deterministic.
        # Preserved for back-compat with legacy / pre-Phase-7b runs that
        # bypass the ``chunking`` phase.
        def _tokenize_concepts(text: str, limit: int = 8) -> List[str]:
            """Lift bare-word concept slugs from a section's text bits."""
            if not text:
                return []
            import re as _re_inner
            cleaned = _re_inner.sub(r"[^a-z0-9\s]", " ", text.lower())
            stop = {
                "the", "and", "for", "with", "from", "that", "this", "are",
                "was", "were", "has", "have", "had", "but", "not", "all",
                "any", "may", "can", "one", "two", "its", "their", "they",
                "will", "been", "you", "your", "our", "into", "over", "such",
                "more", "most", "some", "about", "these", "those", "than",
                "also", "only", "used", "use", "see", "via", "per",
            }
            seen: set = set()
            out: List[str] = []
            for tok in cleaned.split():
                if len(tok) <= 4 or tok in stop or tok in seen:
                    continue
                seen.add(tok)
                out.append(tok)
                if len(out) >= limit:
                    break
            return out

        if (
            not upstream_chunks_loaded
            and staging_dir is not None
            and staging_dir.exists()
        ):
            for sidecar in sorted(staging_dir.rglob("*_synthesized.json")):
                try:
                    doc = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError) as _e:
                    _record_projection_drop(
                        "malformed_sidecar",
                        source=str(sidecar),
                        parse_error=str(_e),
                    )
                    continue
                slug = (
                    sidecar.stem.replace("_synthesized", "")
                    .lower()
                    .replace(" ", "-")
                )
                sections = doc.get("sections") or []
                if not isinstance(sections, list):
                    _record_projection_drop(
                        "non_list_sections",
                        source=str(sidecar),
                        sections_type=type(sections).__name__,
                    )
                    continue
                for _sec_idx, section in enumerate(sections):
                    if not isinstance(section, dict):
                        _record_projection_drop(
                            "non_dict_section",
                            source=str(sidecar),
                            section_index=_sec_idx,
                            section_type=type(section).__name__,
                        )
                        continue
                    section_id = str(section.get("section_id") or "").strip()
                    if not section_id:
                        _record_projection_drop(
                            "missing_section_id",
                            source=str(sidecar),
                            section_index=_sec_idx,
                        )
                        continue
                    title = str(section.get("section_title") or "").strip()
                    section_type = (
                        str(section.get("section_type") or "").strip().lower()
                    )
                    text_bits: List[str] = [title, section_type]
                    data = section.get("data")
                    if isinstance(data, dict):
                        for k, v in data.items():
                            text_bits.append(str(k))
                            if isinstance(v, str):
                                text_bits.append(v)
                            elif isinstance(v, list):
                                for item in v[:20]:
                                    if isinstance(item, str):
                                        text_bits.append(item)
                    chunk_text = " ".join(t for t in text_bits if t).strip()
                    concept_tags = _tokenize_concepts(chunk_text)
                    chunk_id = (
                        f"{course_code_lower}_chunk_{chunk_counter:05d}"
                    )
                    # Phase 8 ST 6: emit canonical 'id' key
                    # (build_pedagogy_graph at
                    # Trainforge/pedagogy_graph_builder.py:593 reads
                    # c.get('id'); pre-Phase-8 chunk_id-keyed emit was
                    # silently dropped). Forward path (upstream
                    # dart_chunks_path JSONL load at :6231-6243) already
                    # emits canonical id via the chunker package; this
                    # fixes the fallback path that runs when
                    # dart_chunks_path is absent or unreadable.
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "concept_tags": concept_tags,
                        "learning_outcome_refs": [],
                        "chunk_type": (
                            "assessment_item"
                            if section_type in ("assessment", "self_check")
                            else "content"
                        ),
                        "bloom_level": "understand",
                        "difficulty": "intermediate",
                        "source": {
                            "module_id": slug,
                            "item_path": f"{slug}#{section_id}",
                        },
                    })
                    chunk_counter += 1

        # ------------------------------------------------------------------
        # Stage-3 concept-synthesis helper (Wave C). Nested so it shares
        # the handler's ``_record_projection_drop`` / logger scope; pure
        # function of its kwargs otherwise.
        # ------------------------------------------------------------------
        async def _run_stage3_concept_synthesis(
            *,
            chunks: List[Dict[str, Any]],
            textbook_structure_path: str,
            course_name: str,
            course_slug: str,
            capture: Any,
        ) -> Optional[Dict[str, Any]]:
            """Stage 3 — synthesize a domain-concept vocabulary, re-tag
            chunks in-memory, return the vocabulary dict.

            Per plan §4.2:

            (a) read ``textbook_structure.json``, pull ``chapters[]``
                with ``chapter_text``;
            (b) N per-chapter ``synthesize_concepts`` calls, batched
                ≤10 (plan §5.3);
            (c) merge per-chapter concepts into a course vocabulary,
                de-dup on ``canonical_slug``;
            (d) compile via ``compile_domain_concept_seeds``;
            (e) re-tag each chunk: ``extract_concept_tags(text, item,
                domain_concept_seeds=seeds)``, UNION into
                ``chunk["concept_tags"]``;
            (g) per-chapter failure isolation (plan §5.4) — one
                chapter's call failing degrades, doesn't abort; all-fail
                → empty-seed fallback.

            Returns the vocabulary dict (persisted by the caller as a
            sibling of ``concept_graph_semantic.json``), or ``None`` when
            no textbook structure / no chapters are available.
            """
            import asyncio as _asyncio

            # --- (a) read textbook_structure.json --------------------------
            ts_path: Optional[Path] = None
            if textbook_structure_path:
                cand = Path(textbook_structure_path)
                if cand.exists() and cand.is_file():
                    ts_path = cand
            if ts_path is None:
                logger.warning(
                    "concept_extraction: Stage-3 requested "
                    "(TEXTBOOK_SYNTHESIS_PROVIDER set) but no readable "
                    "textbook_structure_path (%r); skipping Stage 3.",
                    textbook_structure_path,
                )
                return None
            try:
                textbook_structure = json.loads(
                    ts_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                logger.warning(
                    "concept_extraction: Stage-3 failed to parse "
                    "textbook_structure.json %s (%s); skipping Stage 3.",
                    ts_path, exc,
                )
                return None
            chapters = []
            if isinstance(textbook_structure, dict):
                raw_chapters = textbook_structure.get("chapters")
                if isinstance(raw_chapters, list):
                    chapters = [c for c in raw_chapters if isinstance(c, dict)]
            if not chapters:
                logger.warning(
                    "concept_extraction: Stage-3 textbook_structure.json "
                    "carries no chapters[]; skipping Stage 3.",
                )
                return None

            # --- construct the provider -----------------------------------
            try:
                from Courseforge.generators._textbook_synthesis_provider import (
                    TextbookSynthesisProvider,
                    TextbookSynthesisProviderError,
                )
            except Exception as exc:  # noqa: BLE001 — import failure → skip
                logger.warning(
                    "concept_extraction: Stage-3 provider import failed "
                    "(%s); skipping Stage 3.",
                    exc,
                )
                return None
            try:
                provider = TextbookSynthesisProvider(capture=capture)
            except Exception as exc:  # noqa: BLE001 — construction failure
                logger.warning(
                    "concept_extraction: Stage-3 provider construction "
                    "failed (%s); skipping Stage 3.",
                    exc,
                )
                return None

            # --- (b) N per-chapter calls, batched ≤10 ----------------------
            per_chapter_concepts: List[Dict[str, Any]] = []
            chapter_synthesis_failures: List[str] = []
            chapters_synthesized = 0

            def _one_chapter(chapter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """Synchronous per-chapter call; runs in an executor."""
                cid = str(chapter.get("id") or "")
                try:
                    return provider.synthesize_concepts(
                        chapter, course_name=course_name
                    )
                except TextbookSynthesisProviderError as exc:
                    # Plan §5.4 — per-chapter failure isolation. Record
                    # the failed chapter, continue with the rest.
                    logger.warning(
                        "concept_extraction: Stage-3 chapter %r concept "
                        "call exhausted (%s); degrading per §5.4.",
                        cid, exc,
                    )
                    return None
                except Exception as exc:  # noqa: BLE001 — isolate any raise
                    logger.warning(
                        "concept_extraction: Stage-3 chapter %r concept "
                        "call raised (%s); degrading per §5.4.",
                        cid, exc,
                    )
                    return None

            loop = _asyncio.get_event_loop()
            # ``batch_chapters`` is a staticmethod on the provider class;
            # access it via the constructed instance so a test that
            # injects a provider FACTORY (not the class itself) still
            # resolves the helper. Plan §5.3 — batches of ≤10.
            batches = provider.batch_chapters(chapters)
            for batch in batches:
                # Dispatch each batch of ≤10 chapters via run_in_executor,
                # awaiting the whole batch before the next (plan §5.3 —
                # "wait for ALL batch completions before next batch").
                results = await _asyncio.gather(*[
                    loop.run_in_executor(None, _one_chapter, ch)
                    for ch in batch
                ])
                for chapter, res in zip(batch, results):
                    cid = str(chapter.get("id") or "")
                    if res is None:
                        chapter_synthesis_failures.append(cid)
                        continue
                    chapters_synthesized += 1
                    for concept in res.get("concepts") or []:
                        if isinstance(concept, dict):
                            per_chapter_concepts.append(concept)

            # --- (c) merge + de-dup on canonical_slug ----------------------
            from lib.ontology.slugs import canonical_slug as _canonical_slug

            merged: Dict[str, Dict[str, Any]] = {}
            for concept in per_chapter_concepts:
                canonical_raw = str(concept.get("canonical") or "").strip()
                if not canonical_raw:
                    continue
                slug = _canonical_slug(canonical_raw)
                if not slug:
                    continue
                aliases = [
                    str(a).strip()
                    for a in (concept.get("aliases") or [])
                    if isinstance(a, str) and str(a).strip()
                ]
                # Fix-1 (merge half): preserve the LLM's raw canonical surface
                # form as an alias before it's overwritten by the slug below,
                # so the compiled seed can also match the natural-language form
                # the model emitted (e.g. "Visual Hierarchy") in prose.
                if canonical_raw and canonical_raw not in aliases:
                    aliases.append(canonical_raw)
                chapter_ids = [
                    str(c)
                    for c in (concept.get("chapter_ids") or [])
                    if isinstance(c, (str, int))
                ]
                hint = str(concept.get("definition_hint") or "").strip()
                if slug in merged:
                    existing = merged[slug]
                    for a in aliases:
                        if a not in existing["aliases"]:
                            existing["aliases"].append(a)
                    for c in chapter_ids:
                        if c not in existing["chapter_ids"]:
                            existing["chapter_ids"].append(c)
                    if not existing.get("definition_hint") and hint:
                        existing["definition_hint"] = hint
                else:
                    merged[slug] = {
                        "canonical": slug,
                        "aliases": aliases,
                        "chapter_ids": chapter_ids,
                        "definition_hint": hint,
                    }

            concepts_out = list(merged.values())
            vocabulary: Dict[str, Any] = {
                "schema_version": "v1",
                "course_id": course_name.upper(),
                "course_slug": course_slug,
                "provider": getattr(provider, "_provider", ""),
                "model": getattr(provider, "_model", "") or "",
                "chapter_call_count": len(chapters),
                "chapter_synthesis_failures": chapter_synthesis_failures,
                "concept_count": len(concepts_out),
                "concepts": concepts_out,
            }

            # --- all-fail → empty-seed fallback (plan §5.4) ----------------
            if chapters_synthesized == 0 or not concepts_out:
                logger.warning(
                    "concept_extraction: Stage-3 produced no concepts "
                    "(chapters_synthesized=%d, concepts=%d); falling back "
                    "to empty seeds — graph stays at status-quo.",
                    chapters_synthesized, len(concepts_out),
                )
                # Still return the (empty) vocabulary so the artifact +
                # gate surface the degraded provenance.
                return vocabulary

            # --- (d) compile into (canonical, [regex]) seed pairs ----------
            try:
                from Trainforge.process_course import (
                    compile_domain_concept_seeds,
                )
            except Exception as exc:  # noqa: BLE001 — import failure → skip
                logger.warning(
                    "concept_extraction: Stage-3 compile_domain_concept_seeds "
                    "import failed (%s); skipping re-tag.",
                    exc,
                )
                return vocabulary
            # Fix-1: de-slug the canonical id into a surface-form alias.
            # The merge step above overwrote ``canonical`` with the
            # ``canonical_slug`` (``visual-hierarchy``), which
            # ``compile_domain_concept_seeds`` regex-escapes into a literal
            # ``\bvisual\-hierarchy\b`` pattern that never matches the spaced
            # prose form "visual hierarchy". Append the hyphen-replaced
            # surface form as an alias (mirroring the lexical path below) so
            # the compiled pattern actually fires on the textbook text. Dedupe
            # against existing aliases; skip the alias when it collapses back
            # to the id (single-word slugs like ``accessibility`` where
            # ``replace("-", " ")`` is a no-op).
            _stage3_seed_specs: List[Dict[str, Any]] = []
            for c in concepts_out:
                _cid = c["canonical"]
                _aliases = list(c.get("aliases") or [])
                _deslug = _cid.replace("-", " ")
                if _deslug != _cid and _deslug not in _aliases:
                    _aliases.append(_deslug)
                _stage3_seed_specs.append({"id": _cid, "aliases": _aliases})
            seeds = compile_domain_concept_seeds(_stage3_seed_specs)

            # --- (e) re-tag each loaded chunk in-memory --------------------
            # IN-MEMORY only — chunks.jsonl on disk is NOT rewritten, so
            # dart_chunks_sha256 stays byte-stable (plan §4.2 Risk-6a).
            try:
                from lib.ontology.concept_tagging import extract_concept_tags
            except Exception as exc:  # noqa: BLE001 — import failure → skip
                logger.warning(
                    "concept_extraction: Stage-3 extract_concept_tags "
                    "import failed (%s); skipping re-tag.",
                    exc,
                )
                return vocabulary

            retagged = 0
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                text = str(chunk.get("text") or "")
                if not text:
                    continue
                try:
                    new_tags = extract_concept_tags(
                        text, chunk, domain_concept_seeds=seeds
                    )
                except Exception as exc:  # noqa: BLE001 — isolate per chunk
                    logger.warning(
                        "concept_extraction: Stage-3 re-tag raised on "
                        "chunk %r (%s); leaving its tags untouched.",
                        chunk.get("id"), exc,
                    )
                    continue
                existing = chunk.get("concept_tags")
                existing = existing if isinstance(existing, list) else []
                # UNION — preserve any tags the chunk already carried.
                union = list(existing)
                for tag in new_tags:
                    if tag not in union:
                        union.append(tag)
                if union != existing:
                    retagged += 1
                chunk["concept_tags"] = union

            # The provider already emits one ``textbook_concept_call``
            # decision event per per-chapter call (plan §7) via
            # ``_emit_per_call_decision`` — the handler does not mint an
            # additional aggregate event. ``retagged`` is surfaced in the
            # phase-output envelope below for operator visibility.
            vocabulary["_chunks_retagged"] = retagged
            return vocabulary

        # ------------------------------------------------------------------
        # Three-stage textbook synthesis — Stage 3 (Wave C).
        #
        # plan: plans/textbook-llm-synthesis-3stage-2026-05.md §4.
        #
        # When ``TEXTBOOK_SYNTHESIS_PROVIDER`` is set, dispatch N
        # per-chapter ``synthesize_concepts`` LLM calls (batched ≤10 per
        # plan §5.3) over ``chapters[].chapter_text`` from
        # ``textbook_structure.json``, merge the per-chapter concepts
        # into a course-level vocabulary de-duped on ``canonical_slug``,
        # compile it into ``(canonical, [regex])`` seed pairs, and
        # re-tag the already-loaded chunks IN-MEMORY via
        # ``extract_concept_tags(..., domain_concept_seeds=...)``. The
        # re-tag UNIONs into each chunk's ``concept_tags`` so the
        # downstream ``build_cooccurrence_graph`` emits real
        # ``DomainConcept`` nodes for general (non-RDF) textbook prose —
        # the end-to-end payoff that closes the OpenStax 0-node failure.
        #
        # CRITICAL: the re-tag is in-memory only — ``chunks.jsonl`` on
        # disk is NOT rewritten, so ``dart_chunks_sha256`` stays
        # byte-stable (plan §4.2, Risk-6a). Per-chapter failure
        # isolation (plan §5.4): one chapter's call failing degrades
        # rather than aborts; all-fail → empty-seed fallback (status
        # quo). Default-off (env unset) → this block is a no-op and the
        # phase behaves exactly as today.
        # ------------------------------------------------------------------
        domain_concept_vocabulary: Optional[Dict[str, Any]] = None
        textbook_structure_path_kw = kwargs.get("textbook_structure_path") or ""
        _ts_provider_env = os.environ.get(
            "TEXTBOOK_SYNTHESIS_PROVIDER", ""
        ).strip()
        if _ts_provider_env:
            try:
                domain_concept_vocabulary = await _run_stage3_concept_synthesis(
                    chunks=chunks,
                    textbook_structure_path=textbook_structure_path_kw,
                    course_name=course_name,
                    course_slug=course_slug,
                    capture=_capture_concept,
                )
            except Exception as exc:  # noqa: BLE001 — Stage 3 is best-effort
                logger.warning(
                    "concept_extraction: Stage-3 concept synthesis raised "
                    "(%s); falling back to empty seeds (status quo).",
                    exc,
                )
                domain_concept_vocabulary = None

        # ------------------------------------------------------------------
        # Lexical concept-seed fallback (TRAINFORGE_LEXICAL_CONCEPT_SEEDS).
        #
        # Corpus-generalization fix: on a general (non-RDF/SHACL) textbook
        # corpus the Stage-3 LLM vocabulary pass (TEXTBOOK_SYNTHESIS_PROVIDER)
        # frequently does not run, leaving the per-course
        # ``domain_concept_seeds`` empty. Chunk tagging then matches only the
        # pedagogy-term CONCEPT_PATTERNS, every chunk lands ~2 tags, and the
        # co-occurrence + semantic graph collapses to a degenerate ~2-node
        # graph. This block derives domain-concept seeds from the loaded
        # chunk text using LEXICAL / statistical signals ONLY (frequency,
        # acronyms, multi-word noun phrases; no embeddings — SBIR posture),
        # compiles them, and re-tags the chunks IN-MEMORY (chunks.jsonl on
        # disk is NOT rewritten, so dart_chunks_sha256 stays byte-stable)
        # before the graph build below consumes them.
        #
        # CRITICAL — fires ONLY when:
        #   (a) TRAINFORGE_LEXICAL_CONCEPT_SEEDS is truthy, AND
        #   (b) Stage-3 produced no usable concept vocabulary (the empty-seed
        #       path that yields the degenerate graph).
        # When the flag is unset (default) OR Stage-3 seeds already exist,
        # this block is a no-op and behavior is byte-identical to today — the
        # RDF/SHACL calibration corpus (which has Stage-3 seeds and does not
        # set the flag) is unaffected.
        # ------------------------------------------------------------------
        lexical_seed_count = 0
        lexical_chunks_retagged = 0
        _lexical_flag = os.environ.get(
            "TRAINFORGE_LEXICAL_CONCEPT_SEEDS", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        _stage3_has_seeds = (
            isinstance(domain_concept_vocabulary, dict)
            and int(domain_concept_vocabulary.get("concept_count", 0) or 0) > 0
        )
        if _lexical_flag and not _stage3_has_seeds:
            try:
                from lib.ontology.lexical_concept_seeds import (
                    derive_lexical_concept_seeds,
                )
                from Trainforge.process_course import (
                    compile_domain_concept_seeds,
                )
                from lib.ontology.concept_tagging import extract_concept_tags
            except Exception as exc:  # noqa: BLE001 — import failure → skip
                logger.warning(
                    "concept_extraction: lexical-seed fallback import failed "
                    "(%s); leaving chunks un-retagged (status quo).",
                    exc,
                )
            else:
                lexical_seeds = derive_lexical_concept_seeds(chunks)
                lexical_seed_count = len(lexical_seeds)
                if lexical_seeds:
                    compiled = compile_domain_concept_seeds([
                        {"id": s, "aliases": [s.replace("-", " ")]}
                        for s in lexical_seeds
                    ])
                    for chunk in chunks:
                        if not isinstance(chunk, dict):
                            continue
                        text = str(chunk.get("text") or "")
                        if not text:
                            continue
                        try:
                            new_tags = extract_concept_tags(
                                text, chunk, domain_concept_seeds=compiled
                            )
                        except Exception as exc:  # noqa: BLE001 — per-chunk
                            logger.warning(
                                "concept_extraction: lexical re-tag raised on "
                                "chunk %r (%s); leaving its tags untouched.",
                                chunk.get("id"), exc,
                            )
                            continue
                        existing = chunk.get("concept_tags")
                        existing = existing if isinstance(existing, list) else []
                        union = list(existing)
                        for tag in new_tags:
                            if tag not in union:
                                union.append(tag)
                        if union != existing:
                            lexical_chunks_retagged += 1
                        chunk["concept_tags"] = union
                if _capture_concept is not None:
                    try:
                        _capture_concept.log_decision(
                            decision_type="content_selection",
                            decision=(
                                f"lexical_concept_seed_fallback "
                                f"seeds={lexical_seed_count} "
                                f"retagged={lexical_chunks_retagged}"
                            ),
                            rationale=(
                                f"TRAINFORGE_LEXICAL_CONCEPT_SEEDS is on and "
                                f"Stage-3 produced no domain_concept_vocabulary "
                                f"seeds (the empty-seed path that yields a "
                                f"degenerate concept graph on general corpora). "
                                f"Derived {lexical_seed_count} domain-concept "
                                f"seeds from {len(chunks)} chunks via lexical / "
                                f"statistical signals only (frequency + "
                                f"acronyms + multi-word noun phrases; no "
                                f"embeddings per SBIR posture) and re-tagged "
                                f"{lexical_chunks_retagged} chunks in-memory "
                                f"before the co-occurrence graph build."
                            ),
                            alternatives_considered=[
                                "leave seeds empty (status-quo degenerate graph)",
                                "run Stage-3 LLM vocabulary synthesis",
                            ],
                        )
                    except Exception as exc:  # noqa: BLE001 — best-effort
                        logger.warning(
                            "concept_extraction: lexical-seed decision-capture "
                            "log_decision failed (%s).",
                            exc,
                        )

        # ------------------------------------------------------------------
        # Fix-2: emit the genuine ``kind: "concept_semantic"`` graph.
        #
        # Pre-Fix-2 this phase called ``build_pedagogy_graph`` and wrote a
        # ``kind: "pedagogy"`` graph to a file *named*
        # ``concept_graph_semantic.json`` — the file name lied about its
        # content. The genuine semantic graph (``DomainConcept`` nodes +
        # typed edges) was rebuilt only later inside
        # ``process_course.py::_generate_semantic_concept_graph``.
        #
        # Now the phase: (1) builds the co-occurrence concept graph from the
        # loaded chunks' ``concept_tags`` via the instance-free
        # ``build_cooccurrence_graph`` helper, then (2) feeds it to
        # ``build_semantic_graph`` (whose ``_build_nodes`` copies its node
        # set verbatim from the co-occurrence graph). ``course=None`` is
        # explicitly supported; ``objectives_metadata=None`` is fine
        # (Risk 1 — only the deferred ``targets-concept`` edge rule is
        # inert, an accepted gap). ``misconceptions`` / ``questions`` are
        # derived inline below, mirroring
        # ``process_course.py::_build_misconceptions_for_graph`` /
        # ``_build_questions_for_graph`` — this phase has no
        # ``CourseProcessor`` instance.
        # ------------------------------------------------------------------
        def _empty_semantic_shell() -> Dict[str, Any]:
            """A ``kind: "concept_semantic"`` empty-graph shell.

            Fix-2: the empty-input shell is no longer a pedagogy graph.
            """
            return {
                "kind": "concept_semantic",
                "course_id": course_name.upper(),
                "nodes": [],
                "edges": [],
                "generated_at": datetime.now().isoformat(),
                "stats": {
                    "node_count": 0,
                    "edge_count": 0,
                    "nodes_by_class": {},
                    "edges_by_relation": {},
                },
            }

        def _derive_misconceptions(
            chunk_list: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            """Mirror ``CourseProcessor._build_misconceptions_for_graph``.

            Derives misconception entities from chunk ``misconceptions[]``
            so the ``misconception-of`` rule can fire. Concept routing
            precedence: explicit ``concept_id`` → token-overlap match →
            (implicitly) none.
            """
            try:
                from Trainforge.rag.typed_edge_inference import _make_concept_id
                from Trainforge.process_course import _route_misconception_to_tag
                from lib.ontology.misconception_id import canonical_mc_id
            except Exception:  # noqa: BLE001 — best-effort; rule self-skips on []
                return []
            entities: List[Dict[str, Any]] = []
            seen: set = set()
            cid_course = course_name.upper() or ""
            for chunk in chunk_list:
                raw = chunk.get("misconceptions") or []
                if not raw:
                    continue
                tags = [t for t in (chunk.get("concept_tags") or []) if t]
                for entry in raw:
                    if isinstance(entry, dict):
                        statement = (entry.get("misconception") or "").strip()
                        correction = (entry.get("correction") or "").strip()
                        explicit_cid = (
                            (entry.get("concept_id") or "").strip() or None
                        )
                        bloom_level = (
                            (entry.get("bloom_level") or "").strip().lower()
                        )
                        cognitive_domain = (
                            entry.get("cognitive_domain") or ""
                        ).strip()
                    elif isinstance(entry, str):
                        statement = entry.strip()
                        correction = ""
                        explicit_cid = None
                        bloom_level = ""
                        cognitive_domain = ""
                    else:
                        continue
                    if not statement:
                        continue
                    mc_id = canonical_mc_id(statement, correction, bloom_level)
                    if mc_id in seen:
                        continue
                    seen.add(mc_id)
                    entity: Dict[str, Any] = {
                        "id": mc_id,
                        "misconception": statement,
                        "correction": correction or statement,
                    }
                    if bloom_level:
                        entity["bloom_level"] = bloom_level
                    if cognitive_domain:
                        entity["cognitive_domain"] = cognitive_domain
                    concept_id = explicit_cid
                    if not concept_id and tags:
                        routed_tag = _route_misconception_to_tag(
                            statement, tags
                        )
                        if routed_tag:
                            concept_id = _make_concept_id(
                                routed_tag, cid_course
                            )
                    if concept_id:
                        entity["concept_id"] = concept_id
                    entities.append(entity)
            return entities

        def _derive_questions(
            chunk_list: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            """Mirror ``CourseProcessor._build_questions_for_graph``.

            One question entity per ``learning_outcome_ref`` on every
            ``assessment_item`` chunk so the ``assesses`` rule can fire.
            """
            questions: List[Dict[str, Any]] = []
            for chunk in chunk_list:
                if chunk.get("chunk_type") != "assessment_item":
                    continue
                chunk_id = chunk.get("id")
                if not chunk_id:
                    continue
                for ref in chunk.get("learning_outcome_refs") or []:
                    if not ref:
                        continue
                    questions.append({
                        "id": f"q_{chunk_id}_{ref}",
                        "objective_id": ref,
                        "source_chunk_id": chunk_id,
                    })
            return questions

        # ------------------------------------------------------------------
        # NVIDIA-KG item 1: thread the synthesized course objectives into
        # the typed-edge build. Pre-fix this phase passed ``course=None``
        # + ``objectives_metadata=None`` unconditionally, so the
        # LO-dependent rules (``prerequisite_from_lo_order``,
        # ``targets_concept_from_lo``) early-returned with zero edges on
        # every course (one calibration corpus measured 982 edges, zero prerequisite).
        # Resolution chain: explicit ``objectives_path`` kwarg (routed
        # from the ``reuse_objectives_path`` workflow param) → project
        # export ``01_learning_objectives/synthesized_objectives.json``
        # → project export ``03_content_development/course.json`` →
        # ``LibV2/courses/<slug>/course.json``. Missing objectives
        # degrade to the legacy ``course=None`` path with a warning —
        # never fail the phase. Deterministic (pure file reads, no LLM).
        # ------------------------------------------------------------------
        course_for_graph: Optional[Dict[str, Any]] = None
        objectives_meta_for_graph: Optional[List[Dict[str, Any]]] = None
        objectives_source = ""
        # Phase-ordering fix (Option A1): count of LOs the relocated
        # concept-objective linker enriched with key_concepts[] (0 until
        # the linker runs below; surfaced in the envelope).
        key_concepts_linked = 0
        try:
            course_for_graph, objectives_meta_for_graph, objectives_source = (
                _resolve_course_objectives_for_graph(
                    objectives_path_kw=str(
                        kwargs.get("objectives_path") or ""
                    ),
                    # Phase-ordering fix (Option A1): the fresh-run
                    # objectives source, routed from course_planning's
                    # synthesized_objectives_path (this phase now runs
                    # AFTER course_planning). Candidate #2 — the reuse
                    # kwarg above still wins when supplied.
                    synthesized_objectives_path_kw=str(
                        kwargs.get("synthesized_objectives_path") or ""
                    ),
                    project_path=project_path,
                    libv2_course_dir=(
                        _resolve_libv2_root(kwargs.get("libv2_root"))
                        / "courses"
                        / course_slug
                    ),
                    course_code=course_name.upper(),
                )
            )
        except Exception as exc:  # noqa: BLE001 — resolution is best-effort
            logger.warning(
                "concept_extraction: objectives resolution raised (%s); "
                "degrading to course=None.",
                exc,
            )
        if course_for_graph is None:
            logger.warning(
                "concept_extraction: no synthesized objectives resolvable "
                "for course %s (reuse objectives_path kwarg / "
                "course_planning synthesized_objectives_path / project "
                "export / LibV2 course.json all missing); LO-driven "
                "typed-edge rules (prerequisite_from_lo_order, "
                "targets_concept_from_lo) will emit zero edges. Phase-"
                "ordering fix (Option A1) makes course_planning run BEFORE "
                "this phase, so on a fresh textbook_to_course run this "
                "indicates course_planning did NOT emit objectives "
                "(synthesized_objectives.json missing/empty) — investigate "
                "the upstream phase rather than treating this as expected.",
                course_name,
            )
        else:
            logger.info(
                "concept_extraction: resolved %d learning outcomes from %s "
                "(+%d targets-concept metadata entries) for the typed-edge "
                "build.",
                len(course_for_graph.get("learning_outcomes") or []),
                objectives_source,
                len(objectives_meta_for_graph or []),
            )

        # ------------------------------------------------------------------
        # Fix-2: seed the resolved objectives' key_concepts into the chunk
        # retag BEFORE the co-occurrence graph build.
        #
        # The synthesized objectives carry per-LO ``key_concepts`` as slugs
        # (``design-system``, ``visual-hierarchy``, ...). Those slugs drive
        # the ``targets_concept_from_lo`` typed-edge rule, whose target
        # endpoint is the SAME slug. When that slug isn't already a real
        # co-occurrence node, ``_materialize_endpoint_nodes`` mints a
        # ``frequency=0, node_provenance="lo_key_concept"`` orphan — a node
        # with NO chunk anchoring, which drags down the chunk-anchored
        # coverage floor. By compiling each key_concept slug (with its
        # de-slugged surface-form alias, mirroring the Stage-3 / lexical
        # paths) into a domain-concept seed and union-retagging the in-memory
        # chunks here, a key_concept that actually appears in prose lands in
        # ≥2 chunks → mints a real co-occurrence node → the targets-concept
        # edge resolves to a grounded node instead of a frequency=0 orphan.
        # A key_concept ABSENT from the prose still falls through to the
        # legacy lo_key_concept materialization (behavior preserved).
        #
        # IN-MEMORY only — chunks.jsonl on disk is NOT rewritten (same
        # contract as the Stage-3 / lexical retag blocks above), so
        # dart_chunks_sha256 stays byte-stable.
        # ------------------------------------------------------------------
        lo_key_concept_seed_count = 0
        lo_key_concept_chunks_retagged = 0
        if course_for_graph is not None:
            try:
                from Trainforge.process_course import (
                    compile_domain_concept_seeds as _compile_lo_seeds,
                )
                from lib.ontology.concept_tagging import (
                    extract_concept_tags as _extract_lo_tags,
                )
                from lib.ontology.slugs import canonical_slug as _kc_slug
            except Exception as exc:  # noqa: BLE001 — import failure → skip
                logger.warning(
                    "concept_extraction: LO key_concept seed import failed "
                    "(%s); skipping LO-concept retag.",
                    exc,
                )
            else:
                _kc_slugs: List[str] = []
                _seen_kc: set = set()
                for _lo in course_for_graph.get("learning_outcomes") or []:
                    if not isinstance(_lo, dict):
                        continue
                    _kcs = _lo.get("key_concepts") or _lo.get("keyConcepts")
                    if not isinstance(_kcs, list):
                        continue
                    for _kc in _kcs:
                        if not isinstance(_kc, str):
                            continue
                        _slug = _kc_slug(_kc)
                        if _slug and _slug not in _seen_kc:
                            _seen_kc.add(_slug)
                            _kc_slugs.append(_slug)
                if _kc_slugs:
                    _lo_seed_specs: List[Dict[str, Any]] = []
                    for _slug in _kc_slugs:
                        _aliases = []
                        _deslug = _slug.replace("-", " ")
                        if _deslug != _slug:
                            _aliases.append(_deslug)
                        _lo_seed_specs.append(
                            {"id": _slug, "aliases": _aliases}
                        )
                    try:
                        _lo_seeds = _compile_lo_seeds(_lo_seed_specs)
                    except Exception as exc:  # noqa: BLE001 — compile failure
                        logger.warning(
                            "concept_extraction: LO key_concept seed compile "
                            "failed (%s); skipping LO-concept retag.",
                            exc,
                        )
                        _lo_seeds = []
                    lo_key_concept_seed_count = len(_lo_seeds)
                    for _chunk in chunks:
                        if not isinstance(_chunk, dict):
                            continue
                        _text = str(_chunk.get("text") or "")
                        if not _text or not _lo_seeds:
                            continue
                        try:
                            _new_tags = _extract_lo_tags(
                                _text, _chunk, domain_concept_seeds=_lo_seeds
                            )
                        except Exception as exc:  # noqa: BLE001 — per-chunk
                            logger.warning(
                                "concept_extraction: LO key_concept retag "
                                "raised on chunk %r (%s); leaving its tags "
                                "untouched.",
                                _chunk.get("id"), exc,
                            )
                            continue
                        _existing = _chunk.get("concept_tags")
                        _existing = (
                            _existing if isinstance(_existing, list) else []
                        )
                        _union = list(_existing)
                        for _tag in _new_tags:
                            if _tag not in _union:
                                _union.append(_tag)
                        if _union != _existing:
                            lo_key_concept_chunks_retagged += 1
                        _chunk["concept_tags"] = _union
                    if lo_key_concept_seed_count:
                        logger.info(
                            "concept_extraction: LO key_concept retag seeded "
                            "%d key_concepts and re-tagged %d chunks in-memory "
                            "before the co-occurrence graph build.",
                            lo_key_concept_seed_count,
                            lo_key_concept_chunks_retagged,
                        )

        try:
            from lib.ontology.cooccurrence_graph import build_cooccurrence_graph
            from Trainforge.rag.typed_edge_inference import build_semantic_graph
        except Exception as exc:  # noqa: BLE001 — import failures shouldn't crash phase
            logger.warning(
                "concept_extraction: semantic-graph builder import failed "
                "(%s); emitting empty concept_semantic graph shell.",
                exc,
            )
            graph: Dict[str, Any] = _empty_semantic_shell()
        else:
            try:
                cooccurrence_graph = build_cooccurrence_graph(
                    chunks,
                    course_name.upper() or "",
                    graph_kind="concept",
                )

                # Phase-ordering fix (Option A1): concept-objective linker
                # pass (relocated here from _plan_course_structure). With
                # this phase now running AFTER course_planning, the concept
                # graph (cooccurrence_graph) exists at the SAME time as the
                # synthesized objectives, so the linker can populate each
                # LO's key_concepts[] from concept-graph nodes BEFORE
                # build_semantic_graph runs — which lets the
                # targets_concept_from_lo typed-edge rule (driven off the
                # objectives_metadata derived from key_concepts) emit real
                # edges on a fresh run. We then re-normalize the enriched
                # LO list so course_for_graph + objectives_meta_for_graph
                # reflect the linked key_concepts. Fail-soft: a linker
                # error leaves the un-enriched objectives in place.
                if course_for_graph is not None:
                    try:
                        from lib.ontology.concept_objective_linker import (
                            link_concepts_to_objectives,
                        )

                        _enriched_los = link_concepts_to_objectives(
                            course_for_graph.get("learning_outcomes") or [],
                            cooccurrence_graph,
                        )
                        _relinked_course, _relinked_meta = (
                            _normalize_objectives_payload_to_course(
                                {
                                    "learning_outcomes": _enriched_los,
                                    "course_code": course_name.upper(),
                                },
                                course_name.upper(),
                            )
                        )
                        if _relinked_course is not None:
                            course_for_graph = _relinked_course
                            objectives_meta_for_graph = _relinked_meta
                            key_concepts_linked = sum(
                                1
                                for _lo in _enriched_los
                                if (
                                    _lo.get("key_concepts")
                                    or _lo.get("keyConcepts")
                                )
                            )
                            logger.info(
                                "concept_extraction: concept-objective "
                                "linker enriched key_concepts on %d/%d "
                                "learning outcomes from the concept graph.",
                                key_concepts_linked,
                                len(_enriched_los),
                            )
                    except Exception as exc:  # noqa: BLE001 — fail-soft
                        logger.warning(
                            "concept_extraction: concept-objective linker "
                            "pass failed (%s); proceeding with un-enriched "
                            "objectives.",
                            exc,
                        )

                misconceptions = _derive_misconceptions(chunks)
                questions = _derive_questions(chunks)
                graph = build_semantic_graph(
                    chunks,
                    course=course_for_graph,
                    concept_graph=cooccurrence_graph,
                    misconceptions=misconceptions or None,
                    questions=questions or None,
                    objectives_metadata=objectives_meta_for_graph,
                )
                # ``build_semantic_graph`` stamps ``kind: "concept_semantic"``;
                # add ``course_id`` for parity with the legacy shell shape.
                if isinstance(graph, dict):
                    graph.setdefault("course_id", course_name.upper())

                # KG-quality post-build passes (all default-OFF for byte
                # stability). Order is load-bearing: scaffolding-node prune
                # (Change B) already ran upstream at co-occurrence build
                # time; here we (C) merge duplicate concept nodes, then
                # (A) cap related-to fan-out on the post-merge edge set.
                # Each is an independent, fail-soft dict->dict rewrite.
                if isinstance(graph, dict):
                    if os.getenv(
                        "TRAINFORGE_MERGE_DUPLICATE_CONCEPTS", ""
                    ).lower() == "true":
                        try:
                            from lib.ontology.concept_node_merge import (
                                merge_duplicate_concept_nodes,
                            )
                            graph = merge_duplicate_concept_nodes(graph)
                        except Exception as exc:  # noqa: BLE001 — fail-soft
                            logger.warning(
                                "concept_extraction: concept-merge pass "
                                "failed (%s); leaving graph unmerged.", exc,
                            )
                    # Intra-chunk concept linking (after merge, before cap):
                    # connect concepts co-located in the same chunk with a
                    # low-confidence related-to, so single-section topics
                    # (e.g. LangGraph) don't shatter into degree-1 singletons.
                    if os.getenv(
                        "TRAINFORGE_INTRA_CHUNK_LINKS", ""
                    ).lower() == "true":
                        try:
                            from lib.ontology.intra_chunk_linker import (
                                link_intra_chunk_concepts,
                            )
                            graph = link_intra_chunk_concepts(graph)
                        except Exception as exc:  # noqa: BLE001 — fail-soft
                            logger.warning(
                                "concept_extraction: intra-chunk linking pass "
                                "failed (%s); leaving graph unlinked.", exc,
                            )
                    _fanout_k = os.getenv(
                        "TRAINFORGE_RELATED_FANOUT_CAP", ""
                    ).strip()
                    if _fanout_k.isdigit() and int(_fanout_k) > 0:
                        try:
                            from lib.ontology.related_edge_cap import (
                                cap_related_fanout,
                            )
                            graph = cap_related_fanout(graph, int(_fanout_k))
                        except Exception as exc:  # noqa: BLE001 — fail-soft
                            logger.warning(
                                "concept_extraction: related-to fan-out cap "
                                "failed (%s); leaving edges uncapped.", exc,
                            )
            except Exception as exc:  # noqa: BLE001 — fail-soft on builder error
                logger.warning(
                    "concept_extraction: build_semantic_graph raised (%s); "
                    "emitting empty concept_semantic graph shell.",
                    exc,
                )
                graph = _empty_semantic_shell()

        # Phase-ordering fix (Option A1): write the linker-enriched
        # key_concepts back to the project-export synthesized_objectives.json
        # so the persisted LO doc carries the same keyConcepts the
        # _plan_course_structure linker used to write before the phase move.
        # Guard rails:
        #   * Only when the linker actually enriched something
        #     (key_concepts_linked > 0) and a course was resolved.
        #   * NEVER mutate the reuse kwarg path (a user-supplied file) — we
        #     only write back when objectives_source resolved to the project
        #     export's synthesized_objectives.json (i.e. NOT the
        #     reuse_objectives_path).
        # Best-effort: a merge/write error logs a warning and does not fail
        # the phase.
        reuse_objectives_kwarg = str(kwargs.get("objectives_path") or "")
        if (
            key_concepts_linked > 0
            and course_for_graph is not None
            and objectives_source
            and objectives_source.endswith("synthesized_objectives.json")
            and objectives_source != reuse_objectives_kwarg
        ):
            try:
                _enriched_by_id = {
                    str(lo.get("id")): lo
                    for lo in (course_for_graph.get("learning_outcomes") or [])
                    if isinstance(lo, dict) and lo.get("id")
                }
                _src_path = Path(objectives_source)
                _doc = json.loads(_src_path.read_text(encoding="utf-8"))
                _merged = 0
                for _disk_lo in (_doc.get("learning_outcomes") or []):
                    if not isinstance(_disk_lo, dict):
                        continue
                    _enr = _enriched_by_id.get(str(_disk_lo.get("id")))
                    if _enr is None:
                        continue
                    _kc = _enr.get("key_concepts") or _enr.get("keyConcepts")
                    if isinstance(_kc, list) and _kc:
                        # Preserve the disk LO's existing casing if present,
                        # else default to key_concepts (the runtime form
                        # _plan_course_structure emits).
                        _field = (
                            "keyConcepts"
                            if "keyConcepts" in _disk_lo
                            else "key_concepts"
                        )
                        _disk_lo[_field] = list(_kc)
                        _merged += 1
                if _merged:
                    _src_path.write_text(
                        json.dumps(_doc, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    logger.info(
                        "concept_extraction: merged linker-enriched "
                        "key_concepts back into %d LO entries of %s.",
                        _merged, _src_path,
                    )
            except Exception as exc:  # noqa: BLE001 — fail-soft
                logger.warning(
                    "concept_extraction: key_concepts write-back to "
                    "synthesized_objectives.json failed (%s); the on-disk "
                    "objectives doc keeps its prior key_concepts.", exc,
                )

        # Edge-consensus stamping (GPT-fb-12-may item 2). Runs
        # UNCONDITIONALLY — not behind a behavior flag — because the
        # root CLAUDE.md § Aggregators documents EdgeConsensusAggregator
        # as the source of the per-edge ``edge_status`` +
        # ``consensus_signals[]`` fields on the SEMANTIC graph. Pre-fix,
        # this aggregator only ran post-loop in workflow_runner for the
        # ``graph/`` artifact (process_course path) and was NEVER wired
        # into the ``concept_extraction`` phase, so every course archived
        # via the textbook_to_course pipeline shipped a
        # concept_graph_semantic.json with edge_status=None on all edges
        # and no sibling edge_consensus_report.json (verified on a
        # calibration corpus: 982 edges, all edge_status=None, no report).
        # Stamping here — at the authoring point — closes the silent gap
        # for every course rather than patching one corpus. Deterministic
        # (cross-rule matrix only; no LLM, no NLI unless TRAINFORGE_EDGE_NLI),
        # so the graph stays byte-reproducible for fixed inputs. Fail-soft:
        # a stamping error leaves the graph un-stamped rather than failing
        # the phase, matching the best-effort posture of the post-loop
        # aggregators in workflow_runner.
        edge_consensus_report_path: str = ""
        if isinstance(graph, dict) and graph.get("edges"):
            try:
                from lib.aggregators.edge_consensus import (
                    EdgeConsensusAggregator,
                )

                _run_id = (
                    kwargs.get("run_id")
                    or os.getenv("ED4ALL_RUN_ID", "")
                    or ""
                )
                _consensus = EdgeConsensusAggregator(
                    semantic_graph_path=None,
                    course_slug=course_slug,
                    run_id=str(_run_id),
                )
                # Stamp edge_status + consensus_signals[] in place on the
                # freshly-built graph dict before it is serialized.
                _consensus.apply_to_graph(graph)
            except Exception as exc:  # noqa: BLE001 — fail-soft
                logger.warning(
                    "concept_extraction: edge-consensus stamping failed "
                    "(%s); graph edges left without edge_status.", exc,
                )

        # KG-quality stamping (NVIDIA-KG item 2). Runs AFTER edge-consensus
        # stamping (above) so the consistency axis's
        # (1 - contradiction_rate) attenuation reads the freshly-stamped
        # per-edge ``edge_status`` — composing with, not duplicating, the
        # gate-time attenuation that ``KGQualityValidator.validate`` applies
        # via the EdgeConsensusAggregator. Pre-fix, every textbook_to_course
        # corpus shipped a concept_graph_semantic.json with kg_quality=None
        # (verified on a calibration corpus: 197 nodes / 982 edges, kg_quality
        # null, no sibling quality/ report) because the KG-quality reporter
        # only ran as a gate at libv2_archival and never stamped the field
        # at authoring time. Uses the report-less ``compute_metrics_only``
        # entry point — no SHACL ValidationReport (none exists at this
        # phase), no LLM, deterministic. Fail-soft: any error logs a warning
        # and leaves ``kg_quality`` null, matching the consensus-stamping
        # posture (never fails the phase). The report dict is captured here
        # and written to ``quality/kg_quality_report.json`` after the LibV2
        # course dir is resolved below.
        kg_quality_report: Optional[Dict[str, Any]] = None
        if isinstance(graph, dict):
            try:
                from Trainforge.rag.kg_quality_report import KGQualityReporter

                _run_id_kgq = (
                    kwargs.get("run_id")
                    or os.getenv("ED4ALL_RUN_ID", "")
                    or ""
                )
                _kgq_reporter = KGQualityReporter(
                    course_slug=course_slug,
                    run_id=str(_run_id_kgq),
                    output_dir=Path("."),  # overridden at write time below
                )
                kg_quality_report = _kgq_reporter.compute_metrics_only(graph)
                # Stamp the compact four-dimension score block onto the
                # graph so the in-graph kg_quality field is no longer null.
                _kgq_dims = kg_quality_report.get("dimensions") or {}
                graph["kg_quality"] = {
                    dim: float(
                        (_kgq_dims.get(dim) or {}).get("score", 0.0)
                    )
                    for dim in (
                        "completeness", "consistency", "accuracy", "coverage"
                    )
                }
            except Exception as exc:  # noqa: BLE001 — fail-soft
                logger.warning(
                    "concept_extraction: kg-quality computation failed "
                    "(%s); graph kg_quality left null.", exc,
                )

        # Persist graph to LibV2/courses/<slug>/concept_graph/.
        # Phase 8 ST 3: route through `_resolve_libv2_root` so ops
        # topologies that mount LibV2 at a non-default location can
        # override via `ED4ALL_LIBV2_ROOT` env var or per-call
        # `libv2_root` kwarg threaded by the workflow runner.
        course_dir = (
            _resolve_libv2_root(kwargs.get("libv2_root"))
            / "courses"
            / course_slug
        )
        graph_dir = course_dir / "concept_graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / "concept_graph_semantic.json"
        graph_bytes = json.dumps(graph, indent=2, ensure_ascii=False).encode(
            "utf-8"
        )
        graph_path.write_bytes(graph_bytes)
        sha256 = _hashlib.sha256(graph_bytes).hexdigest()

        # Sibling manifest.json with provenance fields the LibV2 manifest
        # validator can later cross-check (Phase 6 ST 17 wires the field
        # into the canonical course manifest schema; ST 12 emits it here
        # eagerly so the per-phase output is self-describing).
        manifest = {
            "course_id": course_name.upper(),
            "course_slug": course_slug,
            "concept_graph_path": str(graph_path),
            "concept_graph_sha256": sha256,
            "generated_at": datetime.now().isoformat(),
            "source_chunks": len(chunks),
            "phase": "concept_extraction",
        }
        manifest_path = graph_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Sibling edge_consensus_report.json (GPT-fb-12-may item 2). Written
        # next to the just-serialized concept_graph_semantic.json so a LibV2
        # audit can read the corpus-wide consensus rollup (per-rule
        # confirmed/contradicted/pending counts + contradiction_rate) without
        # re-walking the edge list. Reads from the on-disk graph (which now
        # carries the stamped edge_status) so the report and the graph agree.
        # Best-effort: a report-write failure does not fail the phase.
        if isinstance(graph, dict) and graph.get("edges"):
            try:
                from lib.aggregators.edge_consensus import (
                    EdgeConsensusAggregator,
                )

                _run_id = (
                    kwargs.get("run_id")
                    or os.getenv("ED4ALL_RUN_ID", "")
                    or ""
                )
                _report_path = graph_dir / "edge_consensus_report.json"
                EdgeConsensusAggregator(
                    semantic_graph_path=graph_path,
                    course_slug=course_slug,
                    run_id=str(_run_id),
                ).write(_report_path)
                edge_consensus_report_path = str(_report_path)
            except Exception as exc:  # noqa: BLE001 — fail-soft
                logger.warning(
                    "concept_extraction: edge_consensus_report.json write "
                    "failed (%s); the consensus rollup was not persisted "
                    "(the graph's per-edge edge_status is unaffected).", exc,
                )

        # Sibling kg_quality_report.json under the LibV2 course's quality/
        # dir (NVIDIA-KG item 2). Follows the dir convention documented in
        # root CLAUDE.md § Aggregators for the assessment aggregator
        # (``<libv2_course>/quality/trainforge_assessment_quality_report.json``).
        # The full four-dimension report (computed above via
        # compute_metrics_only) is persisted here so a LibV2 audit reads the
        # numerator/denominator + contradiction_rate detail without
        # re-walking the graph. Best-effort: a write failure does not fail
        # the phase, and the in-graph kg_quality block is unaffected.
        kg_quality_report_path: str = ""
        if kg_quality_report is not None:
            try:
                quality_dir = course_dir / "quality"
                quality_dir.mkdir(parents=True, exist_ok=True)
                _kgq_path = quality_dir / "kg_quality_report.json"
                _kgq_path.write_text(
                    json.dumps(
                        kg_quality_report, indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
                kg_quality_report_path = str(_kgq_path)
            except Exception as exc:  # noqa: BLE001 — fail-soft
                logger.warning(
                    "concept_extraction: kg_quality_report.json write failed "
                    "(%s); the KG-quality rollup was not persisted (the "
                    "graph's kg_quality block is unaffected).", exc,
                )

        # Three-stage textbook synthesis — Stage 3 (Wave C, plan §4.3).
        # Persist the merged domain-concept vocabulary as a sibling of
        # concept_graph_semantic.json. Written only when Stage 3 ran
        # (TEXTBOOK_SYNTHESIS_PROVIDER set + a usable textbook structure)
        # — a default-off run leaves no domain_concept_vocabulary.json,
        # and the domain_concept_vocabulary gate skips-with-pass.
        domain_concept_vocabulary_path: str = ""
        domain_concept_count: int = 0
        chunks_retagged: int = 0
        if isinstance(domain_concept_vocabulary, dict):
            chunks_retagged = int(
                domain_concept_vocabulary.pop("_chunks_retagged", 0) or 0
            )
            domain_concept_count = int(
                domain_concept_vocabulary.get("concept_count", 0) or 0
            )
            vocab_path = graph_dir / "domain_concept_vocabulary.json"
            try:
                vocab_path.write_text(
                    json.dumps(
                        domain_concept_vocabulary,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                domain_concept_vocabulary_path = str(vocab_path)
            except OSError as exc:
                logger.warning(
                    "concept_extraction: failed to persist "
                    "domain_concept_vocabulary.json (%s); the Stage-3 "
                    "vocabulary artifact was not written.",
                    exc,
                )

        envelope: Dict[str, Any] = {
            "success": True,
            "concept_graph_path": str(graph_path),
            "concept_graph_sha256": sha256,
            "manifest_path": str(manifest_path),
            "course_slug": course_slug,
            "chunk_count": len(chunks),
            "node_count": len(graph.get("nodes") or []),
            "edge_count": len(graph.get("edges") or []),
            # M2 fix: surface inline-projection drop counts so workflow
            # consumers see the silent-degradation signal without parsing
            # logs. ``projection_drops`` carries the first 10 drop records
            # for postmortem readability; the count is authoritative.
            "projection_drops_count": len(projection_drops),
            "projection_drops": projection_drops[:10],
            # NVIDIA-KG item 1: surface where the LO ordering came from
            # (empty string = course=None degraded path) so operators can
            # tell from the phase output whether the LO-driven typed-edge
            # rules had a chance to fire.
            "objectives_source": objectives_source,
            "learning_outcome_count": len(
                (course_for_graph or {}).get("learning_outcomes") or []
            ),
            # Phase-ordering fix (Option A1): how many LOs the relocated
            # concept-objective linker enriched with key_concepts[] from
            # the concept graph this run.
            "key_concepts_linked": key_concepts_linked,
            # Merge-vs-LO-edges fix: count of graph nodes materialized by
            # typed_edge_inference._materialize_endpoint_nodes for an
            # unresolved targets-concept (LO key_concepts) target. > 0 means
            # the LO author named concepts the corpus chunks never tagged, now
            # carried as provenance-flagged DomainConcept nodes instead of
            # dropped targets-concept edges.
            "lo_concept_nodes_materialized": sum(
                1
                for n in (graph.get("nodes") or [])
                if isinstance(n, dict)
                and n.get("node_provenance") == "lo_key_concept"
            ),
        }
        # Edge-consensus rollup sibling (GPT-fb-12-may item 2). Present
        # whenever the graph carried edges and the report wrote cleanly.
        if edge_consensus_report_path:
            envelope["edge_consensus_report_path"] = (
                edge_consensus_report_path
            )
        # KG-quality rollup sibling (NVIDIA-KG item 2). Present whenever the
        # report computed + wrote cleanly.
        if kg_quality_report_path:
            envelope["kg_quality_report_path"] = kg_quality_report_path
        # Stage-3 (Wave C) output keys. Empty / zero on a default-off run.
        if domain_concept_vocabulary_path:
            envelope["domain_concept_vocabulary_path"] = (
                domain_concept_vocabulary_path
            )
            envelope["domain_concept_count"] = domain_concept_count
            envelope["chunks_retagged"] = chunks_retagged
        # Lexical-seed fallback output keys (TRAINFORGE_LEXICAL_CONCEPT_SEEDS).
        # Present only when the flag fired (off by default → absent).
        if lexical_seed_count:
            envelope["lexical_concept_seed_count"] = lexical_seed_count
            envelope["lexical_chunks_retagged"] = lexical_chunks_retagged
        # Fix-2 LO key_concept retag output keys. Present only when the
        # resolved objectives carried key_concepts to seed (absent otherwise).
        if lo_key_concept_seed_count:
            envelope["lo_key_concept_seed_count"] = lo_key_concept_seed_count
            envelope["lo_key_concept_chunks_retagged"] = (
                lo_key_concept_chunks_retagged
            )
        return json.dumps(envelope)

    registry["run_concept_extraction"] = _run_concept_extraction

    # ============================================================================
    # Phase 7b Subtask 11: _run_dart_chunking — DART chunkset emit
    #
    # New ``chunking`` workflow phase (between ``staging`` and
    # ``objective_extraction`` per Phase 7b ST 10). The phase invokes
    # ``Trainforge.chunker.chunk_content`` over staged DART HTML files,
    # persists the resulting chunks to
    # ``LibV2/courses/<slug>/dart_chunks/chunks.jsonl`` plus a sibling
    # ``manifest.json`` (matching the canonical
    # ``schemas/library/chunkset_manifest.schema.json`` shape), computes
    # the SHA-256 of the chunks file, and surfaces both the path and
    # hash through phase outputs.
    #
    # Mirrors the ``_run_concept_extraction`` template above (Phase 6
    # ST 12 commit ``e0ea640``): async helper, ``**kwargs`` resolved via
    # the workflow YAML's ``inputs_from``, registered in
    # ``_build_tool_registry`` as ``registry["run_dart_chunking"]``,
    # returns a JSON envelope with ``dart_chunks_path`` +
    # ``dart_chunks_sha256`` keys for downstream phase consumption.
    #
    # Unlike ``_run_concept_extraction`` (which inline-projects v4 chunks
    # for ``build_pedagogy_graph`` because no IMSCC exists at that
    # phase), this helper goes through the canonical
    # ``Trainforge.chunker.chunk_content`` path. DART HTML is parsed via
    # ``Trainforge/parsers/html_content_parser.HTMLContentParser`` into
    # ``ContentSection`` objects (the same shape Trainforge's IMSCC path
    # consumes), wrapped into a parsed-item dict, and threaded into the
    # chunker with a thin ``ChunkerContext`` whose ``create_chunk``
    # callback emits a minimal-but-canonical v4 chunk dict (no
    # CourseProcessor instance state — concept_tags / objective_refs are
    # empty here; downstream alignment / tagging is the synthesis
    # surface's responsibility).
    #
    # Per the Phase 7b ST 14.5 reconciliation plan, the upstream
    # ``concept_extraction`` phase will refactor to consume this
    # chunkset (eliminating the inline projection above). For now both
    # surfaces coexist — Phase 7b lands the producer; ST 14.5 lands the
    # consumer-side refactor.
    # ============================================================================
    async def _run_dart_chunking(**kwargs):
        """Run the canonical chunker over staged DART HTML.

        Required kwargs (resolved via ``inputs_from`` in workflows.yaml's
        ``chunking`` phase):
            course_name: Canonical course name (used for course slug + chunk-ID prefix).
            staging_dir: DART staging directory (sibling to objective_extraction).

        Outputs (returned as JSON envelope + persisted):
            dart_chunks_path: ``LibV2/courses/<slug>/dart_chunks/chunks.jsonl``
            dart_chunks_sha256: SHA-256 hex digest of the chunks file bytes

        Sibling ``manifest.json`` is also emitted at the same directory,
        carrying the canonical ``chunkset_manifest`` schema shape (per
        Phase 7b ST 12) so the ``ChunksetManifestValidator`` (ST 13)
        gate has something to validate against.
        """
        import hashlib as _hashlib

        course_name = kwargs.get("course_name") or ""
        staging_dir_kw = kwargs.get("staging_dir") or ""

        course_name = course_name or "UNKNOWN"
        course_slug = course_name.lower().replace("_", "-").replace(" ", "-")
        course_code = course_name.upper().replace("-", "_")

        # Resolve staging dir from the explicit kwarg only. We deliberately
        # do NOT fall back to the global COURSEFORGE_INPUTS dir: on a
        # workflow resume that drops the carried-forward staging path, that
        # fallback resolved to a non-empty cross-course aggregate and made
        # the Wave1-I2 preserve-or-fail-closed guard unreachable, silently
        # overwriting this course's chunks.jsonl. An unresolvable/empty
        # staging input must route to the Wave1-I2 guard below.
        staging_dir: Optional[Path] = None
        if staging_dir_kw:
            cand = Path(staging_dir_kw)
            if cand.exists():
                staging_dir = cand

        # Walk staging_dir for DART HTML files. Filter out
        # ``*_synthesized.json`` neighbours (those are sidecars, not the
        # canonical HTML the chunker consumes).
        html_files: List[Path] = []
        if staging_dir is not None and staging_dir.exists():
            html_files = sorted(staging_dir.rglob("*.html"))

        # Wave1-I2: resume-safety guard. If we have no real input to
        # chunk (no staging_dir resolved, OR staging_dir exists but
        # produced zero HTML files), do NOT overwrite a previously-emitted
        # chunks.jsonl with an empty-bytes shell. On a workflow resume
        # that drops the input path from carried-forward phase context,
        # the prior fail-soft empty-shell behaviour silently destroyed
        # real chunks (see plans/dispatch-7-execution-inspection-2026-05.md
        # Finding 2; cost ~92 chunks during the 2026-05-12 OpenStax run).
        # Preserve when an existing artifact is found; fail-closed when
        # there's nothing to preserve.
        if not html_files:
            existing_chunks_path = (
                _resolve_libv2_root(kwargs.get("libv2_root"))
                / "courses"
                / course_slug
                / "dart_chunks"
                / "chunks.jsonl"
            )
            if existing_chunks_path.exists():
                existing_sha = _hashlib.sha256(
                    existing_chunks_path.read_bytes()
                ).hexdigest()
                existing_count = sum(
                    1
                    for line in existing_chunks_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
                logger.warning(
                    "Wave1-I2: _run_dart_chunking input missing or empty "
                    "(staging_dir_kw=%r, resolved=%r); preserving existing "
                    "chunks.jsonl at %s (%d chunks, sha256=%s) instead of "
                    "overwriting with empty shell.",
                    staging_dir_kw,
                    str(staging_dir) if staging_dir else None,
                    existing_chunks_path,
                    existing_count,
                    existing_sha,
                )
                manifest_path = existing_chunks_path.parent / "manifest.json"
                return json.dumps({
                    "success": True,
                    "preserved_existing": True,
                    "dart_chunks_path": str(existing_chunks_path),
                    "dart_chunks_sha256": existing_sha,
                    "manifest_path": str(manifest_path),
                    "course_slug": course_slug,
                    "chunks_count": existing_count,
                    "source_html_count": 0,
                    "chunker_version": _resolve_chunker_version(),
                })
            raise RuntimeError(
                "Wave1-I2: _run_dart_chunking refusing to emit empty "
                "chunks.jsonl shell — no DART HTML input found "
                f"(staging_dir_kw={staging_dir_kw!r}, "
                f"resolved={str(staging_dir) if staging_dir else None!r}) "
                f"and no pre-existing artifact at {existing_chunks_path}. "
                "On a workflow resume, ensure the staging_dir kwarg is "
                "carried forward via phase_outputs."
            )

        # Compute aggregated source SHA-256 over the DART HTML inputs.
        # Per the schema's ``source_dart_html_sha256`` description, we
        # emit a deterministic merkle-ish digest: sort by filename,
        # concatenate per-file SHA-256 bytes, then SHA-256 the
        # concatenation. Stable across re-runs as long as the input
        # filename + content tuple is stable.
        def _file_sha256(p: Path) -> bytes:
            h = _hashlib.sha256()
            try:
                with p.open("rb") as fh:
                    for blk in iter(lambda: fh.read(65536), b""):
                        h.update(blk)
            except OSError:
                # Unreadable file — record a zero-bytes sentinel so the
                # aggregate still computes deterministically rather than
                # crashing the phase.
                return b"\x00" * 32
            return h.digest()

        if html_files:
            agg = _hashlib.sha256()
            for f in html_files:
                agg.update(_file_sha256(f))
            source_dart_html_sha256 = agg.hexdigest()
        else:
            # Empty-input shell: SHA of empty bytes. Schema-valid
            # (64-char lowercase hex) and deterministic. Lets a
            # pre-staging dry-run still emit a manifest.
            source_dart_html_sha256 = _hashlib.sha256(b"").hexdigest()

        # Parse DART HTML into ContentSection-bearing parsed_items.
        # Trainforge's HTMLContentParser produces the duck-typed
        # ContentSection objects ``Trainforge.chunker.chunk_content`` walks
        # via the ``merge_small_sections`` path. Lazy import keeps this
        # helper's import-time cost low.
        parsed_items: List[Dict[str, Any]] = []
        try:
            from Trainforge.parsers.html_content_parser import HTMLContentParser
        except Exception as exc:  # noqa: BLE001 — import failures shouldn't crash phase
            logger.warning(
                "Phase 7b ST 11: HTMLContentParser import failed (%s); "
                "emitting empty chunks shell.",
                exc,
            )
            html_parser = None
        else:
            html_parser = HTMLContentParser()

        if html_parser is not None:
            for idx, html_path in enumerate(html_files):
                try:
                    html_text = html_path.read_text(encoding="utf-8")
                except OSError as exc:
                    logger.warning(
                        "Phase 7b ST 11: failed to read %s (%s); skipping.",
                        html_path, exc,
                    )
                    continue
                try:
                    parsed = html_parser.parse(html_text)
                except Exception as exc:  # noqa: BLE001 — parser errors fail soft
                    logger.warning(
                        "Phase 7b ST 11: HTML parse failed for %s (%s); "
                        "skipping.",
                        html_path, exc,
                    )
                    continue
                slug = html_path.stem.lower().replace(" ", "-")
                # The chunker reads ``module_id``, ``item_id``,
                # ``raw_html``, ``resource_type``, ``sections``,
                # ``title``, ``misconceptions`` off each item. Other
                # keys are read defensively via ``.get(...)`` so a
                # minimal payload works.
                parsed_items.append({
                    "item_id": slug,
                    "item_path": str(html_path.relative_to(staging_dir))
                    if staging_dir and html_path.is_relative_to(staging_dir)
                    else html_path.name,
                    "title": parsed.title or slug,
                    "resource_type": "page",
                    "module_id": slug,
                    "module_title": parsed.title or slug,
                    "week_num": 0,
                    "word_count": parsed.word_count,
                    "sections": parsed.sections,
                    "learning_objectives": parsed.learning_objectives,
                    "key_concepts": parsed.key_concepts,
                    "interactive_components": parsed.interactive_components,
                    "raw_html": html_text,
                    "page_id": parsed.page_id,
                    "misconceptions": parsed.misconceptions,
                    "suggested_assessment_types": parsed.suggested_assessment_types,
                    "courseforge_metadata": parsed.metadata.get("courseforge"),
                    "objective_refs": parsed.objective_refs,
                    "source_references": parsed.source_references,
                })

        # Minimal create_chunk callback — emits a v4-shaped chunk dict
        # without CourseProcessor's deep instance state. The chunker
        # delegates per-chunk materialisation here so the surface
        # parameters (chunk_id, text, html, item, section_heading,
        # chunk_type, follows_chunk_id, position_in_module, html_xpath,
        # char_span, section_source_ids, merged_headings) are all that
        # we have to thread through.
        #
        # ``concept_tags`` IS populated here (previously ``[]``): the
        # empty-tag substrate starved the downstream concept-graph +
        # CURIE machinery, which has nothing to work from when DART
        # chunks carry no concept tags. The instance-free
        # ``lib.ontology.concept_tagging.extract_concept_tags`` helper
        # (the same logic ``CourseProcessor._extract_concept_tags``
        # delegates to) tags from the chunk text + the HTML parser's
        # ``key_concepts``. ``domain_concept_seeds`` is empty here: the
        # ``chunking`` phase runs before ``course_planning``, so no
        # synthesized objectives (the source of per-course seeds) exist
        # yet — the pedagogy-pattern + tech-anchor + key-concept paths
        # carry the load. objective-ref enrichment remains the synthesis
        # surface's responsibility downstream.
        from lib.ontology.concept_tagging import extract_concept_tags
        from Trainforge.chunker import (
            extract_learning_outcome_refs as _extract_learning_outcome_refs,
        )

        # FIX #3c: chunk-local concept_tags. Default-OFF behind
        # ``TRAINFORGE_CHUNK_LOCAL_TAGS``. When OFF, every chunk on a page
        # tags from the SAME page-level ``item["key_concepts"]`` set, so
        # all chunks of a page collapse onto a byte-identical tag list
        # (legacy behaviour, byte-stable). When ON, each chunk's
        # ``concept_tags`` derive from ITS OWN text: the page-level
        # ``key_concepts`` are filtered down to those whose surface form
        # actually appears in THIS chunk's text, and the text-pattern
        # signals in ``extract_concept_tags`` fire on the chunk's own text.
        # We re-route through ``extract_concept_tags`` (not around it) so
        # the ``TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS`` droppable-class +
        # scaffolding-noise filtering still applies to chunk-local tags.
        chunk_local_tags = (
            os.getenv("TRAINFORGE_CHUNK_LOCAL_TAGS", "").lower() == "true"
        )
        # FIX #3c (extract, not just filter): when the chunk-local flag is
        # ON, the per-chunk tag set is the UNION of (a) the filtered
        # page-level key_concepts that survive ``extract_concept_tags`` and
        # (b) concepts EXTRACTED from THIS chunk's own text via the shared
        # lexical machinery (``derive_lexical_concept_seeds`` — multi-word
        # noun phrases + acronyms, frequency/fragment/function-word filtered,
        # no embeddings). The filter-only path thinned coverage: a content-
        # rich chunk whose page-level key_concepts didn't surface in its text
        # ended up with one or zero tags, starving the concept graph. Adding
        # the chunk-local extraction recovers per-chunk coverage while
        # staying chunk-LOCAL (the seeds come from the chunk's own text).
        # Default-OFF preserves byte-identical legacy emit.
        _CHUNK_LOCAL_TAG_CAP = 12
        if chunk_local_tags:
            from lib.ontology.lexical_concept_seeds import (
                derive_lexical_concept_seeds as _derive_lexical_concept_seeds,
                is_fragment_phrase as _is_fragment_phrase,
            )
            from lib.ontology.concept_classifier import (
                is_scaffolding_noise as _is_scaffolding_noise,
            )

            def _chunk_local_extracted_tags(text: str) -> List[str]:
                """Extract chunk-local concept-tag slugs from ``text`` only.

                Routes the chunk text through the shared lexical seed
                deriver (``min_doc_freq=1`` so a single-chunk corpus still
                yields multi-word phrases; the function's built-in
                function-word / generic / denylist filters still apply),
                then re-applies the same quality gates the rest of the
                pipeline uses: ``is_fragment_phrase`` on multi-word slugs
                (sentence-fragment reject) and ``is_scaffolding_noise``
                (domain-agnostic scaffolding-noise reject). Deterministic:
                ``derive_lexical_concept_seeds`` already returns a stable
                ``(-doc_freq, slug)`` ordering, preserved here.
                """
                if not text:
                    return []
                try:
                    seeds = _derive_lexical_concept_seeds(
                        [{"text": text}], min_doc_freq=1
                    )
                except Exception:  # noqa: BLE001 — extraction is best-effort
                    return []
                out: List[str] = []
                for slug in seeds:
                    if not slug:
                        continue
                    # Multi-word slugs get the sentence-fragment gate; the
                    # deriver's own n-gram filter already drops most junk,
                    # but is_fragment_phrase is the canonical filter the
                    # bold-span harvest uses, so apply it for parity.
                    if "-" in slug and _is_fragment_phrase(
                        slug.replace("-", " ")
                    ):
                        continue
                    if _is_scaffolding_noise(slug):
                        continue
                    out.append(slug)
                return out
        # Tokeniser for chunk-local key-concept membership: split a concept
        # surface form into alphanumeric tokens (keeps internal hyphens /
        # apostrophes glued) so "Knowledge Base" -> ["knowledge", "base"].
        _CONCEPT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

        def _key_concept_in_text(concept: str, text_lower: str) -> bool:
            """True iff every alphanumeric token of ``concept`` appears in
            ``text_lower`` (case-insensitive). Tokenises on non-alphanumeric
            runs so "Knowledge Base" matches "...the knowledge base stores...".
            A concept with no alphanumeric tokens never matches.
            """
            toks = [t for t in _CONCEPT_TOKEN_RE.findall(concept.lower()) if t]
            if not toks:
                return False
            return all(t in text_lower for t in toks)

        def _filter_item_key_concepts(
            item: Dict[str, Any], text: str
        ) -> Dict[str, Any]:
            """Return a shallow-copied ``item`` whose ``key_concepts`` is
            restricted to concepts whose words appear in ``text``. Other
            keys are shared by reference (read-only downstream)."""
            page_concepts = item.get("key_concepts") or []
            if not page_concepts:
                return item
            text_lower = (text or "").lower()
            local_concepts = [
                c for c in page_concepts
                if _key_concept_in_text(str(c), text_lower)
            ]
            local_item = dict(item)
            local_item["key_concepts"] = local_concepts
            return local_item

        def _create_chunk(
            *,
            chunk_id: str,
            text: str,
            html: str,
            item: Dict[str, Any],
            section_heading: str,
            chunk_type: str,
            follows_chunk_id: Optional[str] = None,
            position_in_module: int = 0,
            html_xpath: Optional[str] = None,
            char_span: Optional[List[int]] = None,
            section_source_ids: Optional[List[str]] = None,
            merged_headings: Optional[List[str]] = None,
            dart_source_refs: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            words = text.split()
            word_count = len(words)
            tokens_estimate = int(word_count * 1.3)
            source: Dict[str, Any] = {
                "course_id": course_code,
                "module_id": item.get("module_id") or "",
                "module_title": item.get("module_title") or "",
                "lesson_id": item.get("item_id") or "",
                "lesson_title": item.get("title") or "",
                "resource_type": item.get("resource_type") or "page",
                "section_heading": section_heading,
                "position_in_module": position_in_module,
            }
            if html_xpath:
                source["html_xpath"] = html_xpath
            if char_span is not None:
                source["char_span"] = list(char_span)
            if item.get("item_path"):
                source["item_path"] = item["item_path"]
            # GPT Feedback v2 (May 12 / item 3): thread the aggregate
            # source-document SHA from the sidecar manifest into every
            # chunk's source block. Same value across every chunk from
            # this chunking run; gives downstream consumers a byte-stable
            # join key from chunk → upstream-source without a sidecar
            # lookup. See schemas/knowledge/chunk_v4.schema.json::
            # $defs.Source.source_document_sha256.
            if source_dart_html_sha256:
                source["source_document_sha256"] = source_dart_html_sha256
            # B1+B2: mint canonical source_references[] from the
            # ``{block_id, pages}`` pairs the chunker harvested off
            # ``data-dart-block-id`` / ``data-dart-pages`` on the source
            # HTML. sourceId = ``dart:{slug}#{block_id}`` where the slug is
            # the staged-HTML file stem (``item["item_id"]``, already in the
            # ``path.stem.lower().replace(" ", "-")`` form the source_refs
            # validator + source-router key on). Role auto-assigns to
            # ``primary`` (a DART block IS the source for a chunk built from
            # it). Empty/absent on HTML without ``data-dart-*`` attributes
            # (legacy corpora keep ``source_references`` unset, byte-stable).
            dart_refs = _dart_block_source_references(
                dart_source_refs, item.get("item_id") or ""
            )
            if dart_refs:
                source["source_references"] = dart_refs
            # Wave3-Anew2: bloom_level + difficulty resolution via the
            # canonical JSON-LD > data-cf > heuristic cascade.
            bloom_level, bloom_source = _resolve_chunk_bloom_level(item, text)
            difficulty = _resolve_chunk_difficulty(item, text, bloom_level)
            # Real domain-concept tags from the chunk text + the HTML
            # parser's bold-term / definition ``key_concepts``. No
            # per-course domain seeds at this phase (objectives not yet
            # synthesized); helper defaults ``domain_concept_seeds`` to
            # empty. Defensive: a tagging failure must not abort the
            # chunking phase — fall back to ``[]`` (legacy behaviour).
            try:
                tag_item = (
                    _filter_item_key_concepts(item, text)
                    if chunk_local_tags
                    else item
                )
                concept_tags = extract_concept_tags(text, tag_item)
                if chunk_local_tags:
                    # Union (a) filtered page-level tags (above) with (b)
                    # tags extracted from THIS chunk's own text. First-seen-
                    # stable ordering (page-level survivors first, then
                    # chunk-local extractions); deduped; capped so a long
                    # chunk can't explode the tag list.
                    seen = set(concept_tags)
                    merged = list(concept_tags)
                    for slug in _chunk_local_extracted_tags(text):
                        if slug not in seen:
                            seen.add(slug)
                            merged.append(slug)
                    concept_tags = merged[:_CHUNK_LOCAL_TAG_CAP]
            except Exception:  # noqa: BLE001 — tagging is best-effort
                logger.warning(
                    "concept-tag extraction failed for chunk %s; "
                    "emitting empty concept_tags",
                    chunk_id,
                    exc_info=True,
                )
                concept_tags = []
            # LO-anchoring fix: harvest the LO ids the source page carried
            # (JSON-LD learningObjectives[].id + section data-cf-objective-id
            # / -ref) and anchor this chunk to its section's LOs. DART HTML
            # without LO metadata yields [] (byte-identical to the prior
            # hardcoded behaviour). Best-effort: a harvest failure must not
            # abort the chunking phase.
            try:
                lo_refs = _extract_learning_outcome_refs(item, section_heading)
            except Exception:  # noqa: BLE001 — LO harvest is best-effort
                logger.warning(
                    "learning_outcome_refs extraction failed for chunk %s; "
                    "emitting empty learning_outcome_refs",
                    chunk_id,
                    exc_info=True,
                )
                lo_refs = []
            chunk: Dict[str, Any] = {
                "id": chunk_id,
                "schema_version": "v4",
                "chunk_type": chunk_type,
                "text": text,
                "html": html,
                "follows_chunk": follows_chunk_id,
                "source": source,
                "concept_tags": concept_tags,
                "learning_outcome_refs": lo_refs,
                "difficulty": difficulty,
                "tokens_estimate": tokens_estimate,
                "word_count": word_count,
                "bloom_level": bloom_level,
            }
            # Per the schema's bloom_level_source contract: only emit the
            # provenance tag on low-confidence sources (verbs / default).
            if bloom_source in ("verbs", "default"):
                chunk["bloom_level_source"] = bloom_source
            return chunk

        # W3.H sub-task H1: count blocks_seen + per-reason drops BEFORE
        # dispatching to the chunker so we can emit the canonical
        # source_coverage block in the manifest. Walk parsed_items and
        # tally one "block" per ContentSection (or one per item when
        # the item has no sections); attribute drops via lightweight
        # heuristics applied to the same fields the chunker reads
        # (text content + section html). The numbers are conservative:
        # the chunker may also collapse short adjacent sections via
        # merge_small_sections, which is captured under the `dedup`
        # bucket below as the difference between attributable drops
        # and total dropped blocks.
        import re as _re
        blocks_seen = 0
        drop_boilerplate = 0
        drop_image_only = 0
        for _it in parsed_items:
            sections_list = _it.get("sections") or []
            if not sections_list:
                # Item with no sections counts as a single block emitted
                # via the unsectioned chunk_text_block path.
                blocks_seen += 1
                _raw_html = (_it.get("raw_html") or "").strip()
                _text_probe = _raw_html
                # Strip tags for a coarse "is there any text" probe.
                _text_probe = _re.sub(r"<[^>]+>", " ", _text_probe).strip()
                if _raw_html and not _text_probe and (
                    "<img" in _raw_html.lower()
                    or "<figure" in _raw_html.lower()
                ):
                    drop_image_only += 1
                continue
            for section in sections_list:
                blocks_seen += 1
                _content = (
                    getattr(section, "content", None)
                    if not isinstance(section, dict)
                    else section.get("content")
                ) or ""
                _wc = (
                    getattr(section, "word_count", None)
                    if not isinstance(section, dict)
                    else section.get("word_count")
                ) or 0
                _components = (
                    getattr(section, "components", None)
                    if not isinstance(section, dict)
                    else section.get("components")
                ) or []
                _content_str = str(_content).strip()
                if not _content_str and _components:
                    # Section that carries only an interactive component
                    # (image / figure / flip-card) without textual prose:
                    # the chunker's `if not text.strip(): continue` drops
                    # it.
                    drop_image_only += 1

        # Dispatch to the canonical chunker. ``chunk_content`` is
        # fail-soft on empty input (returns empty result, no ctx
        # required); for non-empty input it requires ``ctx`` per the
        # ChunkerContextRequired contract.
        chunks: List[Dict[str, Any]] = []
        chunker_version = _resolve_chunker_version()
        try:
            from Trainforge.chunker import ChunkerContext, chunk_content
        except Exception as exc:  # noqa: BLE001 — import failures shouldn't crash phase
            logger.warning(
                "Phase 7b ST 11: Trainforge.chunker import failed (%s); "
                "emitting empty chunks shell.",
                exc,
            )
        else:
            try:
                ctx = ChunkerContext(create_chunk=_create_chunk) if parsed_items else None
                result = chunk_content(
                    parsed_items,
                    course_code,
                    boilerplate_spans=None,
                    ctx=ctx,
                )
                chunks = list(result.chunks)
            except Exception as exc:  # noqa: BLE001 — fail-soft on chunker error
                logger.warning(
                    "Phase 7b ST 11: chunk_content raised (%s); emitting "
                    "empty chunks shell.",
                    exc,
                )
                chunks = []

        # Persist chunks + manifest to LibV2/courses/<slug>/dart_chunks/.
        # Phase 8 ST 3: route through `_resolve_libv2_root` (see helper
        # docstring for resolution chain). Default behaviour unchanged.
        course_dir = (
            _resolve_libv2_root(kwargs.get("libv2_root"))
            / "courses"
            / course_slug
        )
        chunks_dir = course_dir / "dart_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = chunks_dir / "chunks.jsonl"

        # Stream chunks as JSONL (one chunk per line). Use a hashing
        # writer pattern so the SHA-256 reflects exactly the bytes on
        # disk (no double-read).
        chunks_sha = _hashlib.sha256()
        with chunks_path.open("wb") as fh:
            for chunk in chunks:
                line = (json.dumps(chunk, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                )
                fh.write(line)
                chunks_sha.update(line)
        chunks_sha256 = chunks_sha.hexdigest()

        # W3.H sub-task H1: build the canonical source_coverage block.
        # consumed_count = blocks_seen, emitted_count = len(chunks).
        # Per-reason histogram pulls from the heuristics tracked above
        # (boilerplate / image_only). Any remaining drop delta —
        # blocks merged into adjacent sections via merge_small_sections,
        # or empty after sentence-splitting — falls into the `dedup`
        # bucket. The build_source_coverage helper enforces the
        # dropped_count == sum(drop_reasons.values()) invariant and
        # fires INTERNAL_DROP_REASON_MISSING when the math doesn't
        # balance (silent-gaming check).
        from lib.governance.source_coverage import build_source_coverage
        _attributable_drops = drop_boilerplate + drop_image_only
        _dropped_total = max(0, blocks_seen - len(chunks))
        _drop_reasons: Dict[str, int] = {}
        if drop_boilerplate:
            _drop_reasons["boilerplate"] = drop_boilerplate
        if drop_image_only:
            _drop_reasons["image_only"] = drop_image_only
        _dedup_delta = max(0, _dropped_total - _attributable_drops)
        if _dedup_delta:
            _drop_reasons["dedup"] = _dedup_delta
        source_coverage_block = build_source_coverage(
            consumed_count=blocks_seen,
            emitted_count=len(chunks),
            drop_reasons=_drop_reasons,
            dropped_count=_dropped_total,
            label="dart_chunking",
        )

        # Sibling manifest.json — must validate against
        # ``schemas/library/chunkset_manifest.schema.json`` per ST 12.
        # Required: chunks_sha256, chunker_version, chunkset_kind,
        # source_dart_html_sha256 (conditional on chunkset_kind=dart).
        # Optional: chunks_count, generated_at, source_coverage (W3.H H1).
        manifest = {
            "chunks_sha256": chunks_sha256,
            "chunker_version": chunker_version,
            "chunkset_kind": "dart",
            "source_dart_html_sha256": source_dart_html_sha256,
            "chunks_count": len(chunks),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_coverage": source_coverage_block,
        }
        manifest_path = chunks_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return json.dumps({
            "success": True,
            "dart_chunks_path": str(chunks_path),
            "dart_chunks_sha256": chunks_sha256,
            "manifest_path": str(manifest_path),
            "course_slug": course_slug,
            "chunks_count": len(chunks),
            "source_html_count": len(html_files),
            "chunker_version": chunker_version,
        })

    registry["run_dart_chunking"] = _run_dart_chunking

    # ============================================================================
    # Phase 7c Subtask 16: _run_imscc_chunking — IMSCC chunkset emit
    #
    # New ``imscc_chunking`` workflow phase (between ``packaging`` and
    # ``training_synthesis`` per Phase 7c ST 16). The phase invokes
    # ``Trainforge.chunker.chunk_content`` over the HTML files extracted from
    # the packaged IMSCC zip emitted by the upstream ``packaging`` phase,
    # persists the resulting chunks to
    # ``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl`` plus a sibling
    # ``manifest.json`` (matching the canonical
    # ``schemas/library/chunkset_manifest.schema.json`` shape with
    # ``chunkset_kind="imscc"``), computes the SHA-256 of the chunks file,
    # and surfaces both the path and hash through phase outputs.
    #
    # Mirrors the ``_run_dart_chunking`` template above (Phase 7b
    # ST 11 commit ``5ccbf0c``); the structural differences are:
    #
    # - Source artifact: a packaged ``.imscc`` zip (``imscc_path`` kwarg
    #   resolved from ``phase_outputs.packaging.package_path``) rather
    #   than a staging directory of loose HTML files.
    # - ``source_imscc_sha256`` (SHA-256 of the .imscc archive bytes)
    #   replaces ``source_dart_html_sha256`` in the manifest emit.
    # - ``chunkset_kind="imscc"`` discriminates downstream consumers
    #   (the schema's conditional source-SHA branch fires accordingly).
    # - HTML files are read from inside the zip in-memory via
    #   ``zipfile.ZipFile`` (no on-disk extraction required).
    #
    # Per Phase 7c ST 15 (commit ``090d286``), the directory was renamed
    # from ``corpus/`` to ``imscc_chunks/`` symmetrically with the new
    # ``dart_chunks/`` directory. Trainforge's in-process chunker
    # invocation in ``Trainforge/process_course.py`` is preserved for
    # legacy callers (no churn this subtask) — this phase provides a
    # workflow-level entry point that downstream orchestrator runs use
    # in lieu of re-running the in-process chunker.
    # ============================================================================
    async def _run_imscc_chunking(**kwargs):
        """Run the canonical chunker over the packaged IMSCC archive.

        Required kwargs (resolved via ``inputs_from`` in workflows.yaml's
        ``imscc_chunking`` phase):
            course_name: Canonical course name (used for course slug).
            imscc_path: Path to the packaged ``.imscc`` archive emitted by
                the upstream ``packaging`` phase
                (``phase_outputs.packaging.package_path``).

        Outputs (returned as JSON envelope + persisted):
            imscc_chunks_path: ``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl``
            imscc_chunks_sha256: SHA-256 hex digest of the chunks file bytes

        Sibling ``manifest.json`` is also emitted at the same directory,
        carrying the canonical ``chunkset_manifest`` schema shape
        (``chunkset_kind="imscc"``, ``source_imscc_sha256``) so the
        ``ChunksetManifestValidator`` gate has something to validate
        against.
        """
        import hashlib as _hashlib
        import zipfile as _zipfile

        course_name = kwargs.get("course_name") or ""
        imscc_path_kw = kwargs.get("imscc_path") or kwargs.get("package_path") or ""

        course_name = course_name or "UNKNOWN"
        course_slug = course_name.lower().replace("_", "-").replace(" ", "-")
        course_code = course_name.upper().replace("-", "_")

        # Resolve IMSCC path. Honor explicit kwarg first; emit an empty
        # shell + log a warning when the upstream phase didn't surface
        # a real path so the manifest is still schema-valid (mirroring
        # ``_run_dart_chunking``'s fail-soft empty-input shell).
        imscc_path: Optional[Path] = None
        if imscc_path_kw:
            cand = Path(imscc_path_kw)
            if cand.exists() and cand.is_file():
                imscc_path = cand

        if imscc_path is None:
            # Wave1-I2: resume-safety guard. Do NOT overwrite a
            # previously-emitted chunks.jsonl with an empty-bytes shell
            # on a workflow resume that drops imscc_path from carried-
            # forward phase context. The prior fail-soft empty-shell
            # behaviour silently destroyed real chunks (see
            # plans/dispatch-7-execution-inspection-2026-05.md Finding 2;
            # cost ~92 chunks during the 2026-05-12 OpenStax run).
            # Preserve when an existing artifact is found; fail-closed
            # when there's nothing to preserve.
            existing_chunks_path = (
                _resolve_libv2_root(kwargs.get("libv2_root"))
                / "courses"
                / course_slug
                / "imscc_chunks"
                / "chunks.jsonl"
            )
            if existing_chunks_path.exists():
                existing_sha = _hashlib.sha256(
                    existing_chunks_path.read_bytes()
                ).hexdigest()
                existing_count = sum(
                    1
                    for line in existing_chunks_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
                logger.warning(
                    "Wave1-I2: _run_imscc_chunking imscc_path %r missing "
                    "or not a file; preserving existing chunks.jsonl at "
                    "%s (%d chunks, sha256=%s) instead of overwriting "
                    "with empty shell.",
                    imscc_path_kw,
                    existing_chunks_path,
                    existing_count,
                    existing_sha,
                )
                manifest_path = existing_chunks_path.parent / "manifest.json"
                # Read source_imscc_sha256 from the existing manifest if
                # available; otherwise emit None so callers know it's
                # not derivable without the archive on hand.
                preserved_source_sha: Optional[str] = None
                if manifest_path.exists():
                    try:
                        preserved_manifest = json.loads(
                            manifest_path.read_text(encoding="utf-8")
                        )
                        preserved_source_sha = preserved_manifest.get(
                            "source_imscc_sha256"
                        )
                    except (OSError, json.JSONDecodeError):
                        preserved_source_sha = None
                return json.dumps({
                    "success": True,
                    "preserved_existing": True,
                    "imscc_chunks_path": str(existing_chunks_path),
                    "imscc_chunks_sha256": existing_sha,
                    "manifest_path": str(manifest_path),
                    "course_slug": course_slug,
                    "chunks_count": existing_count,
                    "source_html_count": 0,
                    "source_imscc_sha256": preserved_source_sha,
                    "chunker_version": _resolve_chunker_version(),
                })
            raise RuntimeError(
                "Wave1-I2: _run_imscc_chunking refusing to emit empty "
                "chunks.jsonl shell — imscc_path "
                f"{imscc_path_kw!r} is missing or not a file, and no "
                f"pre-existing artifact at {existing_chunks_path}. "
                "On a workflow resume, ensure imscc_path is carried "
                "forward via phase_outputs.packaging.package_path."
            )

        # Compute source SHA-256 over the .imscc archive bytes. Per
        # the schema's ``source_imscc_sha256`` description: a single
        # SHA-256 over the zip bytes (the archive is one file, unlike
        # the multi-file DART HTML aggregate). Empty input shell uses
        # SHA of empty bytes for determinism.
        if imscc_path is not None:
            agg = _hashlib.sha256()
            try:
                with imscc_path.open("rb") as fh:
                    for blk in iter(lambda: fh.read(65536), b""):
                        agg.update(blk)
                source_imscc_sha256 = agg.hexdigest()
            except OSError as exc:
                logger.warning(
                    "Phase 7c ST 16: failed to read imscc %s (%s); "
                    "using empty-bytes SHA sentinel.",
                    imscc_path, exc,
                )
                source_imscc_sha256 = _hashlib.sha256(b"").hexdigest()
        else:
            source_imscc_sha256 = _hashlib.sha256(b"").hexdigest()

        # Extract HTML files from inside the IMSCC zip. We walk the
        # archive in-memory rather than calling
        # ``IMSCCParser.extract_to_directory`` so the helper has no
        # filesystem side effects beyond the ``imscc_chunks/`` write.
        html_entries: List[Dict[str, str]] = []  # [{"path": ..., "content": ...}, ...]
        if imscc_path is not None:
            try:
                with _zipfile.ZipFile(imscc_path, "r") as zf:
                    for name in sorted(zf.namelist()):
                        if name.endswith(".html") or name.endswith(".htm"):
                            try:
                                content = zf.read(name).decode(
                                    "utf-8", errors="ignore"
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "Phase 7c ST 16: failed to read %s "
                                    "from imscc (%s); skipping.",
                                    name, exc,
                                )
                                continue
                            html_entries.append({"path": name, "content": content})
            except _zipfile.BadZipFile as exc:
                logger.warning(
                    "Phase 7c ST 16: %s is not a valid zip (%s); "
                    "emitting empty chunks shell.",
                    imscc_path, exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Phase 7c ST 16: zip-walk failed on %s (%s); "
                    "emitting empty chunks shell.",
                    imscc_path, exc,
                )

        # Parse each HTML payload via Trainforge's HTMLContentParser
        # into ContentSection-bearing parsed_items shaped exactly like
        # the IMSCC consumer in ``Trainforge/process_course.py``.
        parsed_items: List[Dict[str, Any]] = []
        try:
            from Trainforge.parsers.html_content_parser import HTMLContentParser
        except Exception as exc:  # noqa: BLE001 — import failures shouldn't crash phase
            logger.warning(
                "Phase 7c ST 16: HTMLContentParser import failed (%s); "
                "emitting empty chunks shell.",
                exc,
            )
            html_parser = None
        else:
            html_parser = HTMLContentParser()

        if html_parser is not None and html_entries:
            for entry in html_entries:
                html_text = entry["content"]
                inner_path = entry["path"]
                try:
                    parsed = html_parser.parse(html_text)
                except Exception as exc:  # noqa: BLE001 — parser errors fail soft
                    logger.warning(
                        "Phase 7c ST 16: HTML parse failed for %s (%s); "
                        "skipping.",
                        inner_path, exc,
                    )
                    continue
                slug = Path(inner_path).stem.lower().replace(" ", "-")
                parsed_items.append({
                    "item_id": slug,
                    "item_path": inner_path,
                    "title": parsed.title or slug,
                    "resource_type": "page",
                    "module_id": slug,
                    "module_title": parsed.title or slug,
                    "week_num": 0,
                    "word_count": parsed.word_count,
                    "sections": parsed.sections,
                    "learning_objectives": parsed.learning_objectives,
                    "key_concepts": parsed.key_concepts,
                    "interactive_components": parsed.interactive_components,
                    "raw_html": html_text,
                    "page_id": parsed.page_id,
                    "misconceptions": parsed.misconceptions,
                    "suggested_assessment_types": parsed.suggested_assessment_types,
                    "courseforge_metadata": parsed.metadata.get("courseforge"),
                    "objective_refs": parsed.objective_refs,
                    "source_references": parsed.source_references,
                })

        # ``concept_tags`` IS populated here (previously hardcoded ``[]``):
        # the empty-tag substrate starved every downstream concept-graph +
        # CURIE consumer (the IMSCC chunkset feeds the same machinery the
        # DART chunkset does). Mirror ``_run_dart_chunking`` exactly: tag
        # from the chunk text + the HTML parser's bold-term / definition
        # ``key_concepts`` via the instance-free
        # ``lib.ontology.concept_tagging.extract_concept_tags`` helper (the
        # same logic ``CourseProcessor._extract_concept_tags`` delegates
        # to). ``extract_concept_tags`` honors the
        # ``TRAINFORGE_SEED_TECH_CONCEPTS`` (tech-anchor seeding) and
        # ``TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS`` (scaffolding prune)
        # flags internally; the fragment-filter
        # (``TRAINFORGE_FILTER_FRAGMENT_CONCEPTS``) + content-aware
        # chunk_type (``TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE``) flags fire in
        # the parser / chunker the IMSCC path already routes through.
        # ``domain_concept_seeds`` is empty here (post-packaging IMSCC
        # chunking carries no per-course seed compiler instance), matching
        # the DART path's default-empty seeds.
        from lib.ontology.concept_tagging import extract_concept_tags
        from Trainforge.chunker import (
            extract_learning_outcome_refs as _extract_learning_outcome_refs,
        )

        # FIX #3c (mirrored from ``_run_dart_chunking``): chunk-local
        # concept_tags behind ``TRAINFORGE_CHUNK_LOCAL_TAGS``. When OFF
        # (default), every chunk on a page tags from the SAME page-level
        # ``item["key_concepts"]`` set. When ON, each chunk's tags derive
        # from ITS OWN text: page-level ``key_concepts`` are filtered to
        # those whose surface form appears in this chunk, unioned with
        # concepts extracted from the chunk's own text via the shared
        # lexical-seed machinery (re-routed through ``extract_concept_tags``
        # so scaffolding-noise filtering still applies).
        chunk_local_tags = (
            os.getenv("TRAINFORGE_CHUNK_LOCAL_TAGS", "").lower() == "true"
        )
        _CHUNK_LOCAL_TAG_CAP = 12
        if chunk_local_tags:
            from lib.ontology.lexical_concept_seeds import (
                derive_lexical_concept_seeds as _derive_lexical_concept_seeds,
                is_fragment_phrase as _is_fragment_phrase,
            )
            from lib.ontology.concept_classifier import (
                is_scaffolding_noise as _is_scaffolding_noise,
            )

            def _chunk_local_extracted_tags(text: str) -> List[str]:
                """Extract chunk-local concept-tag slugs from ``text`` only.

                Routes the chunk text through the shared lexical seed
                deriver (``min_doc_freq=1`` so a single-chunk corpus still
                yields multi-word phrases), then re-applies the same
                quality gates the rest of the pipeline uses
                (``is_fragment_phrase`` + ``is_scaffolding_noise``).
                """
                if not text:
                    return []
                try:
                    seeds = _derive_lexical_concept_seeds(
                        [{"text": text}], min_doc_freq=1
                    )
                except Exception:  # noqa: BLE001 — extraction is best-effort
                    return []
                out: List[str] = []
                for slug in seeds:
                    if not slug:
                        continue
                    if "-" in slug and _is_fragment_phrase(
                        slug.replace("-", " ")
                    ):
                        continue
                    if _is_scaffolding_noise(slug):
                        continue
                    out.append(slug)
                return out

        # Tokeniser for chunk-local key-concept membership (mirrors
        # ``_run_dart_chunking``): split a concept surface form into
        # alphanumeric tokens so "Knowledge Base" -> ["knowledge", "base"].
        _CONCEPT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

        def _key_concept_in_text(concept: str, text_lower: str) -> bool:
            """True iff every alphanumeric token of ``concept`` appears in
            ``text_lower`` (case-insensitive)."""
            toks = [t for t in _CONCEPT_TOKEN_RE.findall(concept.lower()) if t]
            if not toks:
                return False
            return all(t in text_lower for t in toks)

        def _filter_item_key_concepts(
            item: Dict[str, Any], text: str
        ) -> Dict[str, Any]:
            """Return a shallow-copied ``item`` whose ``key_concepts`` is
            restricted to concepts whose words appear in ``text``."""
            page_concepts = item.get("key_concepts") or []
            if not page_concepts:
                return item
            text_lower = (text or "").lower()
            local_concepts = [
                c for c in page_concepts
                if _key_concept_in_text(str(c), text_lower)
            ]
            local_item = dict(item)
            local_item["key_concepts"] = local_concepts
            return local_item

        # Minimal create_chunk callback — emits a v4-shaped chunk dict
        # without CourseProcessor's deep instance state. Mirrors
        # ``_run_dart_chunking``'s callback exactly so DART + IMSCC
        # chunks share a single chunk shape (the same v4 contract).
        def _create_chunk(
            *,
            chunk_id: str,
            text: str,
            html: str,
            item: Dict[str, Any],
            section_heading: str,
            chunk_type: str,
            follows_chunk_id: Optional[str] = None,
            position_in_module: int = 0,
            html_xpath: Optional[str] = None,
            char_span: Optional[List[int]] = None,
            section_source_ids: Optional[List[str]] = None,
            merged_headings: Optional[List[str]] = None,
            dart_source_refs: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            words = text.split()
            word_count = len(words)
            tokens_estimate = int(word_count * 1.3)
            source: Dict[str, Any] = {
                "course_id": course_code,
                "module_id": item.get("module_id") or "",
                "module_title": item.get("module_title") or "",
                "lesson_id": item.get("item_id") or "",
                "lesson_title": item.get("title") or "",
                "resource_type": item.get("resource_type") or "page",
                "section_heading": section_heading,
                "position_in_module": position_in_module,
            }
            if html_xpath:
                source["html_xpath"] = html_xpath
            if char_span is not None:
                source["char_span"] = list(char_span)
            if item.get("item_path"):
                source["item_path"] = item["item_path"]
            # GPT Feedback v2 (May 12 / item 3): thread the aggregate
            # IMSCC zip SHA from the sidecar manifest into every chunk's
            # source block. See schemas/knowledge/chunk_v4.schema.json::
            # $defs.Source.source_document_sha256.
            if source_imscc_sha256:
                source["source_document_sha256"] = source_imscc_sha256
            # B1+B2 parity with the DART path: when an imscc-chunked page
            # happens to carry DART provenance attributes, mint the same
            # canonical source_references[]. No-op for ordinary IMSCC /
            # Courseforge HTML (no ``data-dart-*`` attrs → harvest is empty).
            dart_refs = _dart_block_source_references(
                dart_source_refs, item.get("item_id") or ""
            )
            if dart_refs:
                source["source_references"] = dart_refs
            # Wave3-Anew2: bloom_level + difficulty resolution via the
            # canonical JSON-LD > data-cf > heuristic cascade.
            bloom_level, bloom_source = _resolve_chunk_bloom_level(item, text)
            difficulty = _resolve_chunk_difficulty(item, text, bloom_level)
            # Real domain-concept tags from the chunk text + the HTML
            # parser's bold-term / definition ``key_concepts`` (previously
            # hardcoded ``[]``, which starved every downstream concept
            # consumer). Mirrors ``_run_dart_chunking`` exactly: empty
            # per-course ``domain_concept_seeds``; honors the chunk-local
            # flag. Defensive: a tagging failure must not abort the chunking
            # phase — fall back to ``[]`` (legacy behaviour).
            try:
                tag_item = (
                    _filter_item_key_concepts(item, text)
                    if chunk_local_tags
                    else item
                )
                concept_tags = extract_concept_tags(text, tag_item)
                if chunk_local_tags:
                    seen = set(concept_tags)
                    merged = list(concept_tags)
                    for slug in _chunk_local_extracted_tags(text):
                        if slug not in seen:
                            seen.add(slug)
                            merged.append(slug)
                    concept_tags = merged[:_CHUNK_LOCAL_TAG_CAP]
            except Exception:  # noqa: BLE001 — tagging is best-effort
                logger.warning(
                    "concept-tag extraction failed for chunk %s; "
                    "emitting empty concept_tags",
                    chunk_id,
                    exc_info=True,
                )
                concept_tags = []
            # LO-anchoring fix: harvest the LO ids the source page carried
            # (JSON-LD learningObjectives[].id + section data-cf-objective-id
            # / -ref) and anchor this chunk to its section's LOs. Empty for
            # legacy / non-Courseforge IMSCC pages (byte-identical to the
            # prior hardcoded []). Best-effort: a harvest failure must not
            # abort the chunking phase.
            try:
                lo_refs = _extract_learning_outcome_refs(item, section_heading)
            except Exception:  # noqa: BLE001 — LO harvest is best-effort
                logger.warning(
                    "learning_outcome_refs extraction failed for chunk %s; "
                    "emitting empty learning_outcome_refs",
                    chunk_id,
                    exc_info=True,
                )
                lo_refs = []
            chunk: Dict[str, Any] = {
                "id": chunk_id,
                "schema_version": "v4",
                "chunk_type": chunk_type,
                "text": text,
                "html": html,
                "follows_chunk": follows_chunk_id,
                "source": source,
                "concept_tags": concept_tags,
                "learning_outcome_refs": lo_refs,
                "difficulty": difficulty,
                "tokens_estimate": tokens_estimate,
                "word_count": word_count,
                "bloom_level": bloom_level,
            }
            if bloom_source in ("verbs", "default"):
                chunk["bloom_level_source"] = bloom_source
            return chunk

        # Dispatch to the canonical chunker. ``chunk_content`` is
        # fail-soft on empty input.
        chunks: List[Dict[str, Any]] = []
        chunker_version = _resolve_chunker_version()
        try:
            from Trainforge.chunker import ChunkerContext, chunk_content
        except Exception as exc:  # noqa: BLE001 — import failures shouldn't crash phase
            logger.warning(
                "Phase 7c ST 16: Trainforge.chunker import failed (%s); "
                "emitting empty chunks shell.",
                exc,
            )
        else:
            try:
                ctx = ChunkerContext(create_chunk=_create_chunk) if parsed_items else None
                result = chunk_content(
                    parsed_items,
                    course_code,
                    boilerplate_spans=None,
                    ctx=ctx,
                )
                chunks = list(result.chunks)
            except Exception as exc:  # noqa: BLE001 — fail-soft on chunker error
                logger.warning(
                    "Phase 7c ST 16: chunk_content raised (%s); emitting "
                    "empty chunks shell.",
                    exc,
                )
                chunks = []

        # Persist chunks + manifest to LibV2/courses/<slug>/imscc_chunks/.
        # Phase 8 ST 3: route through `_resolve_libv2_root` (see helper
        # docstring for resolution chain). Default behaviour unchanged.
        course_dir = (
            _resolve_libv2_root(kwargs.get("libv2_root"))
            / "courses"
            / course_slug
        )
        chunks_dir = course_dir / "imscc_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = chunks_dir / "chunks.jsonl"

        # Stream chunks as JSONL with hashing writer pattern so the
        # SHA-256 reflects exactly the bytes on disk.
        chunks_sha = _hashlib.sha256()
        with chunks_path.open("wb") as fh:
            for chunk in chunks:
                line = (json.dumps(chunk, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                )
                fh.write(line)
                chunks_sha.update(line)
        chunks_sha256 = chunks_sha.hexdigest()

        # W3.H sub-task H1 (mirrored to IMSCC chunkset for symmetry):
        # build the canonical source_coverage block. Same heuristics
        # as the DART path — count one block per ContentSection (or
        # one block per item with no sections) and attribute drops via
        # boilerplate / image_only checks; remaining drop delta lands
        # in the `dedup` bucket. Symmetric so a downstream consumer
        # (W3.G master aggregator) sees the same shape regardless of
        # chunkset_kind.
        import re as _re_imscc
        from lib.governance.source_coverage import (
            build_source_coverage as _build_source_coverage_imscc,
        )
        _imscc_blocks_seen = 0
        _imscc_drop_boilerplate = 0
        _imscc_drop_image_only = 0
        for _it in parsed_items:
            sections_list = _it.get("sections") or []
            if not sections_list:
                _imscc_blocks_seen += 1
                _raw_html = (_it.get("raw_html") or "").strip()
                _text_probe = _re_imscc.sub(r"<[^>]+>", " ", _raw_html).strip()
                if _raw_html and not _text_probe and (
                    "<img" in _raw_html.lower()
                    or "<figure" in _raw_html.lower()
                ):
                    _imscc_drop_image_only += 1
                continue
            for section in sections_list:
                _imscc_blocks_seen += 1
                _content = (
                    getattr(section, "content", None)
                    if not isinstance(section, dict)
                    else section.get("content")
                ) or ""
                _components = (
                    getattr(section, "components", None)
                    if not isinstance(section, dict)
                    else section.get("components")
                ) or []
                if not str(_content).strip() and _components:
                    _imscc_drop_image_only += 1
        _imscc_attributable = _imscc_drop_boilerplate + _imscc_drop_image_only
        _imscc_dropped_total = max(0, _imscc_blocks_seen - len(chunks))
        _imscc_drop_reasons: Dict[str, int] = {}
        if _imscc_drop_boilerplate:
            _imscc_drop_reasons["boilerplate"] = _imscc_drop_boilerplate
        if _imscc_drop_image_only:
            _imscc_drop_reasons["image_only"] = _imscc_drop_image_only
        _imscc_dedup_delta = max(0, _imscc_dropped_total - _imscc_attributable)
        if _imscc_dedup_delta:
            _imscc_drop_reasons["dedup"] = _imscc_dedup_delta
        _imscc_source_coverage = _build_source_coverage_imscc(
            consumed_count=_imscc_blocks_seen,
            emitted_count=len(chunks),
            drop_reasons=_imscc_drop_reasons,
            dropped_count=_imscc_dropped_total,
            label="imscc_chunking",
        )

        # Sibling manifest.json — must validate against
        # ``schemas/library/chunkset_manifest.schema.json`` per Phase
        # 7b ST 12 (symmetric across DART + IMSCC). Required:
        # chunks_sha256, chunker_version, chunkset_kind,
        # source_imscc_sha256 (conditional on chunkset_kind=imscc).
        # Optional: chunks_count, generated_at, source_coverage (W3.H H1).
        manifest = {
            "chunks_sha256": chunks_sha256,
            "chunker_version": chunker_version,
            "chunkset_kind": "imscc",
            "source_imscc_sha256": source_imscc_sha256,
            "chunks_count": len(chunks),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_coverage": _imscc_source_coverage,
        }
        manifest_path = chunks_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return json.dumps({
            "success": True,
            "imscc_chunks_path": str(chunks_path),
            "imscc_chunks_sha256": chunks_sha256,
            "manifest_path": str(manifest_path),
            "course_slug": course_slug,
            "chunks_count": len(chunks),
            "source_html_count": len(html_entries),
            "source_imscc_sha256": source_imscc_sha256,
            "chunker_version": chunker_version,
        })

    registry["run_imscc_chunking"] = _run_imscc_chunking

    # ============================================================================
    # WS2 — run_vector_indexing: deterministic on-device vector-index build.
    #
    # Backs the ``rag-indexer`` agent (``rag_training`` ``indexing`` phase) via
    # the AGENT_TOOL_MAPPING flip from ``analyze_imscc_content`` (which only
    # counted HTML files and never produced an index) to this real tool.
    #
    # Deterministic transformation (no LLM dispatch, no DecisionCapture
    # obligation — precedent: ``_run_dart_chunking``). It resolves the course
    # dir under the LibV2 root, builds an EmbeddingClient from the env-configured
    # provider, embeds the resolved chunkset, and persists
    # ``vector_index/{embeddings.npy,id_map.json,manifest.json}``.
    #
    # FAIL-CLOSED CONTRACT (anti-silent-degradation): when the embedding
    # backend is unavailable (weights not cached, server down, extras missing)
    # the tool returns ``{"success": false, "error": ...}`` and the phase FAILS.
    # There is NO file-counting / lexical fallback — the indexing phase can no
    # longer "succeed" without an actual index.
    # ============================================================================
    async def _run_vector_indexing(**kwargs):
        """Build the per-course vector index. Fail-closed; deterministic.

        Resolved kwargs (workflows.yaml ``inputs_from`` / param mapping):
            course_name (aliases: course / name / course_code): the course
                whose chunkset is embedded (resolves the LibV2 course slug).
            chunkset (optional): ``imscc`` | ``dart`` | ``corpus-legacy`` pin;
                defaults to the canonical resolver precedence.
            provider (optional): embedding provider registry key; defaults to
                ``ED4ALL_EMBEDDING_PROVIDER`` or ``st``.
            model (optional): embedding model id override.
            text_field_policy (optional): defaults to ``text+heading``.
            force (optional): rebuild a still-fresh index.
            libv2_root (optional): LibV2 root override.

        Envelope (success):
            {"success": true, "manifest_path", "embeddings_path",
             "id_map_path", "vector_index_dir", "model_fingerprint",
             "embedding_model_id", "embedding_provider", "embedding_dim",
             "chunks_count", "chunkset_kind", "source_chunks_sha256",
             "course_slug"}

        Envelope (fail-closed): {"success": false, "error", "error_type",
            "course_slug"} — the phase fails; no partial index, no fallback.
        """
        course_name = (
            kwargs.get("course_name")
            or kwargs.get("course")
            or kwargs.get("name")
            or kwargs.get("course_code")
            or ""
        )
        course_name = course_name or "UNKNOWN"
        course_slug = course_name.lower().replace("_", "-").replace(" ", "-")

        chunkset = kwargs.get("chunkset") or None
        provider = kwargs.get("provider") or None
        model_id = kwargs.get("model") or kwargs.get("model_id") or None
        text_field_policy = kwargs.get("text_field_policy") or "text+heading"
        force = bool(kwargs.get("force", False))

        course_dir = (
            _resolve_libv2_root(kwargs.get("libv2_root"))
            / "courses"
            / course_slug
        )

        # Lazy imports so this module has no import-time coupling to the
        # embedding / index packages (slim installs stay importable).
        try:
            from lib.embedding.providers import (
                EmbeddingBackendUnavailable,
                build_embedding_client,
            )
            from LibV2.tools.libv2.vector_index import (
                SemanticIndexMissing,
                build_vector_index,
            )
        except Exception as exc:  # noqa: BLE001 — missing index/embedding deps
            return json.dumps({
                "success": False,
                "error": (
                    f"vector-index dependencies unavailable: {exc}. Install the "
                    f"[embedding] extra and ensure LibV2 is on the path."
                ),
                "error_type": "dependency_missing",
                "course_slug": course_slug,
            })

        if not course_dir.exists():
            return json.dumps({
                "success": False,
                "error": (
                    f"course directory not found at {course_dir}; archive the "
                    f"course to LibV2 (and run the chunking phase) before "
                    f"indexing."
                ),
                "error_type": "course_missing",
                "course_slug": course_slug,
            })

        # Build the embedding client from the env-configured provider. The
        # build path is provision-time, so offline=False (a real provider may
        # download weights here — never on the query path).
        try:
            client = build_embedding_client(
                provider, model_id, offline=False
            )
            manifest = build_vector_index(
                course_dir,
                client=client,
                chunkset=chunkset,
                text_field_policy=text_field_policy,
                force=force,
            )
        except EmbeddingBackendUnavailable as exc:
            # Honest fail-closed: NO file-counting fallback.
            return json.dumps({
                "success": False,
                "error": (
                    f"embedding backend unavailable; the indexing phase fails "
                    f"closed (no lexical/file-count fallback): {exc}"
                ),
                "error_type": "embedding_backend_unavailable",
                "course_slug": course_slug,
            })
        except SemanticIndexMissing as exc:
            return json.dumps({
                "success": False,
                "error": str(exc),
                "error_type": "chunkset_missing",
                "course_slug": course_slug,
            })
        except FileExistsError as exc:
            return json.dumps({
                "success": False,
                "error": str(exc),
                "error_type": "fresh_index_exists",
                "course_slug": course_slug,
            })
        except Exception as exc:  # noqa: BLE001 — surface, never silently degrade
            return json.dumps({
                "success": False,
                "error": f"vector-index build failed: {exc}",
                "error_type": "build_error",
                "course_slug": course_slug,
            })

        index_dir = course_dir / "vector_index"
        try:
            fingerprint = dict(client.model_fingerprint())
        except Exception:  # noqa: BLE001 — fingerprint is best-effort surface
            fingerprint = {}

        return json.dumps({
            "success": True,
            "vector_index_dir": str(index_dir),
            "manifest_path": str(index_dir / "manifest.json"),
            "embeddings_path": str(index_dir / "embeddings.npy"),
            "id_map_path": str(index_dir / "id_map.json"),
            "model_fingerprint": fingerprint,
            "embedding_model_id": manifest.embedding_model_id,
            "embedding_provider": manifest.embedding_provider,
            "embedding_dim": manifest.embedding_dim,
            "chunks_count": manifest.chunks_count,
            "chunkset_kind": manifest.chunkset_kind,
            "source_chunks_sha256": manifest.source_chunks_sha256,
            "course_slug": course_slug,
        })

    registry["run_vector_indexing"] = _run_vector_indexing

    # ================================================================= #
    # Runtime registry stubs for the 7 tools that AGENT_TOOL_MAPPING     #
    # routes but _build_tool_registry previously skipped (MCP audit      #
    # Q1 critical finding). Each wrapper imports the @mcp.tool()         #
    # implementation at call time (register_* functions create closures  #
    # — we extract them into a capturing MCP stand-in the same way       #
    # test_stage_dart_outputs.py::_CapturingMCP does).                   #
    # ================================================================= #
    class _CapturingMCP:
        """Minimal stand-in for FastMCP: captures decorated tools by name."""
        def __init__(self) -> None:
            self.tools: dict = {}

        def tool(self):  # noqa: D401 - mimics FastMCP's .tool() decorator
            def _decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return _decorator

    def _capture_dart_tools() -> dict:
        try:
            from MCP.tools.dart_tools import register_dart_tools
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"DART tool capture failed: {exc}")
            return {}
        mcp_cap = _CapturingMCP()
        register_dart_tools(mcp_cap)
        return mcp_cap.tools

    def _capture_courseforge_tools() -> dict:
        try:
            from MCP.tools.courseforge_tools import register_courseforge_tools
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Courseforge tool capture failed: {exc}")
            return {}
        mcp_cap = _CapturingMCP()
        register_courseforge_tools(mcp_cap)
        return mcp_cap.tools

    def _capture_trainforge_tools() -> dict:
        try:
            from MCP.tools.trainforge_tools import register_trainforge_tools
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Trainforge tool capture failed: {exc}")
            return {}
        mcp_cap = _CapturingMCP()
        register_trainforge_tools(mcp_cap)
        return mcp_cap.tools

    async def _get_courseforge_status(**kwargs):
        """Registry wrapper: delegates to courseforge_tools.get_courseforge_status."""
        tools = _capture_courseforge_tools()
        tool = tools.get("get_courseforge_status")
        if tool is None:
            return json.dumps({"error": "get_courseforge_status tool unavailable"})
        return await tool()

    registry["get_courseforge_status"] = _get_courseforge_status

    async def _validate_wcag_compliance(**kwargs):
        """Registry wrapper: delegates to dart_tools.validate_wcag_compliance."""
        html_path = kwargs.get("html_path") or kwargs.get("path") or ""
        tools = _capture_dart_tools()
        tool = tools.get("validate_wcag_compliance")
        if tool is None:
            return json.dumps({"error": "validate_wcag_compliance tool unavailable"})
        return await tool(html_path=html_path)

    registry["validate_wcag_compliance"] = _validate_wcag_compliance

    async def _batch_convert_multi_source(**kwargs):
        """Registry wrapper: delegates to dart_tools.batch_convert_multi_source."""
        combined_dir = kwargs.get("combined_dir") or kwargs.get("input") or ""
        output_zip = kwargs.get("output_zip")
        output_dir = kwargs.get("output_dir")
        tools = _capture_dart_tools()
        tool = tools.get("batch_convert_multi_source")
        if tool is None:
            return json.dumps({"error": "batch_convert_multi_source tool unavailable"})
        return await tool(
            combined_dir=combined_dir,
            output_zip=output_zip,
            output_dir=output_dir,
        )

    registry["batch_convert_multi_source"] = _batch_convert_multi_source

    async def _convert_pdf_multi_source(**kwargs):
        """Registry wrapper: delegates to dart_tools.convert_pdf_multi_source."""
        combined_json_path = (
            kwargs.get("combined_json_path")
            or kwargs.get("combined_json")
            or kwargs.get("source")
            or ""
        )
        output_path = kwargs.get("output_path")
        course_code = kwargs.get("course_code")
        tools = _capture_dart_tools()
        tool = tools.get("convert_pdf_multi_source")
        if tool is None:
            return json.dumps({"error": "convert_pdf_multi_source tool unavailable"})
        return await tool(
            combined_json_path=combined_json_path,
            output_path=output_path,
            course_code=course_code,
        )

    registry["convert_pdf_multi_source"] = _convert_pdf_multi_source

    async def _intake_imscc_package(**kwargs):
        """Registry wrapper: delegates to courseforge_tools.intake_imscc_package."""
        imscc_path = kwargs.get("imscc_path") or kwargs.get("package") or ""
        output_dir = kwargs.get("output_dir") or kwargs.get("extract_to") or ""
        remediate = kwargs.get("remediate", True)
        tools = _capture_courseforge_tools()
        tool = tools.get("intake_imscc_package")
        if tool is None:
            return json.dumps({"error": "intake_imscc_package tool unavailable"})
        return await tool(
            imscc_path=imscc_path,
            output_dir=output_dir,
            remediate=remediate,
        )

    registry["intake_imscc_package"] = _intake_imscc_package

    async def _remediate_course_content(**kwargs):
        """Registry wrapper: delegates to courseforge_tools.remediate_course_content."""
        project_id = kwargs.get("project_id") or ""
        remediation_types = kwargs.get("remediation_types")
        tools = _capture_courseforge_tools()
        tool = tools.get("remediate_course_content")
        if tool is None:
            return json.dumps({"error": "remediate_course_content tool unavailable"})
        return await tool(
            project_id=project_id,
            remediation_types=remediation_types,
        )

    registry["remediate_course_content"] = _remediate_course_content

    async def _validate_assessment(**kwargs):
        """Registry wrapper: delegates to trainforge_tools.validate_assessment."""
        assessment_id = (
            kwargs.get("assessment_id")
            or kwargs.get("assessment")
            or kwargs.get("id")
            or ""
        )
        tools = _capture_trainforge_tools()
        tool = tools.get("validate_assessment")
        if tool is None:
            return json.dumps({"error": "validate_assessment tool unavailable"})
        return await tool(assessment_id=assessment_id)

    registry["validate_assessment"] = _validate_assessment

    return registry


def _get_phase_status(workflow: dict, phase_name: str) -> dict:
    """Extract status for a specific phase from workflow tasks."""
    tasks = workflow.get("tasks", [])

    # Map phase names to agent types
    phase_agents = {
        "dart_conversion": ["dart-converter"],
        "staging": ["textbook-stager"],
        "objective_extraction": ["textbook-ingestor"],
        "source_mapping": ["source-router"],
        # Phase 6 ST 11: concept_extraction phase backed by
        # pedagogy-graph-builder agent.
        "concept_extraction": ["pedagogy-graph-builder"],
        "course_planning": ["course-outliner"],
        "content_generation": ["content-generator"],
        "packaging": ["brightspace-packager"],
        "trainforge_assessment": ["assessment-generator"],
        "libv2_archival": ["libv2-archivist"],
        "finalization": ["brightspace-packager"]
    }

    agents = phase_agents.get(phase_name, [])

    phase_tasks = [t for t in tasks if t.get("agent_type") in agents]

    if not phase_tasks:
        return {"status": "PENDING", "tasks": 0}

    statuses = [t.get("status") for t in phase_tasks]

    if all(s == "COMPLETE" for s in statuses):
        phase_status = "COMPLETE"
    elif any(s == "ERROR" for s in statuses):
        phase_status = "ERROR"
    elif any(s == "IN_PROGRESS" for s in statuses):
        phase_status = "IN_PROGRESS"
    else:
        phase_status = "PENDING"

    return {
        "status": phase_status,
        "tasks": len(phase_tasks),
        "completed": sum(1 for s in statuses if s == "COMPLETE"),
        "errors": sum(1 for s in statuses if s == "ERROR")
    }
