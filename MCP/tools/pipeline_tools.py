"""
Ed4All Pipeline Tools

MCP tools for the unified textbook-to-course pipeline.
Chains: DART (PDF -> HTML) -> Courseforge (course generation) -> Trainforge (assessments)
"""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import PROJECT_ROOT  # noqa: E402
from lib.secure_paths import validate_path_within_root  # noqa: E402

logger = logging.getLogger(__name__)

# Derived paths
DART_OUTPUT_DIR = PROJECT_ROOT / "DART" / "batch_output"
COURSEFORGE_INPUTS = PROJECT_ROOT / "Courseforge" / "inputs" / "textbooks"
TRAINING_CAPTURES = PROJECT_ROOT / "training-captures"


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
    (e.g. ``"rdf-shacl-550"``) still produce the right prefix
    (``"rdf_shacl_550_chunk_"``).
    """
    code = (course_name or "").strip().lower()
    if not code:
        return ""
    # Trainforge keeps underscores in the prefix; if the caller passed a
    # slug-shaped name (``rdf-shacl-550``), normalise back to underscores.
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
      pre-dates ``run_start_ts``. Caught the rdf-shacl-550 leak.

    Args:
        chunks_path: Where chunks.jsonl lives in the LibV2 archive
            (``course_dir / "corpus" / "chunks.jsonl"``).
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


def _detect_source_provenance(course_dir: Path) -> bool:
    """Wave 10: scan archived chunks.jsonl for chunks with source_references[].

    Returns True when at least one chunk in ``<course_dir>/corpus/chunks.jsonl``
    carries ``source.source_references[]`` populated with at least one entry.
    Returns False on missing file, read errors, malformed JSONL lines, or
    when no chunks carry refs (pre-Wave-9 corpus). The manifest then advertises
    ``features.source_provenance: false`` so LibV2 retrieval callers can
    fast-skip source-grounded queries.
    """
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
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

    The scan looks in three candidate locations under ``<course_dir>``:
    ``graph/concept_graph_semantic.json``, ``corpus/concept_graph_semantic.json``,
    or any ``*.json`` file shaped like a semantic graph (``kind ==
    "concept_semantic"``) sitting inside the corpus dir. First match wins.
    """
    candidates = [
        course_dir / "graph" / "concept_graph_semantic.json",
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
        # fixing the OLSR_SIM_01 four-codes-in-one-run observation.
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
        chunks_path = corpus_dir_path / "corpus" / "chunks.jsonl"
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
            if assessment_path:
                assess = Path(assessment_path)
                if assess.exists():
                    dest = course_dir / "corpus" / assess.name
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

            manifest_path = course_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

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
            rationale = (
                f"Ran DART pipeline against "
                f"{Path(source_pdf).name if source_pdf else 'raw_text_only'}; "
                f"backend={backend}; classifier_mode={classifier_mode}; "
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
        # Use the PDF stem (e.g. "keet_ontology_engineering") as the doc title
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
            project_path = _PROJECT_ROOT / "Courseforge" / "exports" / project_id

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
            project_path = _PROJECT_ROOT / "Courseforge" / "exports" / project_id
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

            structure_path = (
                project_path / "01_learning_objectives" / "textbook_structure.json"
            )
            structure_path.write_text(
                json.dumps(textbook_structure, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "project_path": str(project_path),
                "textbook_structure_path": str(structure_path),
                "chapter_count": len(merged_chapters),
                "duration_weeks": duration_weeks,
                "duration_weeks_autoscaled": bool(
                    not duration_explicit and merged_chapters
                ),
                "source_file_count": len(html_files),
                "extraction_error_count": len(extraction_errors),
            })

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
                cand = _PROJECT_ROOT / "Courseforge" / "exports" / project_id
                if cand.exists():
                    project_path = cand
            if project_path is None and course_name:
                exports_dir = _PROJECT_ROOT / "Courseforge" / "exports"
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

            if supplied_terminal or supplied_chapter:
                terminal = list(supplied_terminal)
                chapter = list(supplied_chapter)
                mint_method = "user_supplied_objectives_json"
            else:
                terminal, chapter = _cgh.synthesize_objectives_from_topics(
                    topics, duration_weeks,
                )
                mint_method = "synthesize_objectives_from_topics"

            # Detect textbook_structure_path to record provenance.
            structure_path = (
                project_path / "01_learning_objectives" / "textbook_structure.json"
            )
            generated_from = str(structure_path) if structure_path.exists() else ""

            # Canonical on-disk shape.
            lo_entries: List[Dict[str, Any]] = []
            for to in terminal:
                entry = dict(to)
                entry["hierarchy_level"] = "terminal"
                lo_entries.append(entry)
            for co in chapter:
                entry = dict(co)
                entry["hierarchy_level"] = "chapter"
                lo_entries.append(entry)

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

            # Real TO/CO ids for downstream phase_outputs.
            objective_ids = [str(e["id"]) for e in lo_entries if e.get("id")]

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "project_path": str(project_path),
                "synthesized_objectives_path": str(objectives_out_path),
                "objective_ids": ",".join(objective_ids),
                "terminal_count": len(terminal),
                "chapter_count": len(chapter),
                "mint_method": mint_method,
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

            project_path = _PROJECT_ROOT / "Courseforge" / "exports" / project_id
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

            # ---------------------------------------------------------- #
            # Emit each week via generate_week.                           #
            # ---------------------------------------------------------- #
            generated_files: list = []
            weeks_prepared = 0
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
            })

        registry["generate_course_content"] = _generate_course_content
        # END BLOCK: Worker α

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
            project_path = _PROJECT_ROOT / "Courseforge" / "exports" / project_id
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

            package_path = final_dir / f"{course_name}.imscc"

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
                candidate = _PROJECT_ROOT / "Courseforge" / "exports" / project_id_kw
                if candidate.exists():
                    project_dir = candidate
            if project_dir is None and imscc_path and imscc_path.exists():
                candidate = imscc_path.parent.parent
                if candidate.exists():
                    project_dir = candidate
            if project_dir is None:
                # Last-resort fallback: most recent export dir matching course_id.
                exports_dir = _PROJECT_ROOT / "Courseforge" / "exports"
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

            chunks_path = trainforge_dir / "corpus" / "chunks.jsonl"
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

            return json.dumps({
                "success": True,
                "assessment_id": assessment.assessment_id,
                "question_count": len(assessment.questions),
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
                    "locate corpus/chunks.jsonl"
                ),
            })

        corpus_dir_path = Path(corpus_dir)
        chunks_path = corpus_dir_path / "corpus" / "chunks.jsonl"
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
        (observed today: smoke_hifi_rag_chunk_* IDs leaked into the
        rdf-shacl-550 archive after trainforge_assessment failed). When
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
                PROJECT_ROOT / "Courseforge" / "exports" / project_id_kw / "trainforge"
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
                    dest = course_dir / "corpus" / ap.name
                    shutil.copy2(ap, dest)
                    archived["assessment"] = str(dest)

        # Heuristic fallback: scan well-known locations for chunks.jsonl.
        if trainforge_dir is None:
            candidates: list[Path] = []
            exports_root = PROJECT_ROOT / "Courseforge" / "exports"
            if exports_root.exists():
                for project_dir in exports_root.iterdir():
                    if not project_dir.is_dir():
                        continue
                    tf = project_dir / "trainforge"
                    if (tf / "chunks.jsonl").exists() or (tf / "corpus" / "chunks.jsonl").exists():
                        candidates.append(tf)
            runs_root = PROJECT_ROOT / "state" / "runs"
            if runs_root.exists():
                for run_dir in runs_root.iterdir():
                    if not run_dir.is_dir():
                        continue
                    tf = run_dir / "trainforge"
                    if (tf / "chunks.jsonl").exists() or (tf / "corpus" / "chunks.jsonl").exists():
                        candidates.append(tf)
            if candidates:
                def _chunks_mtime(p):
                    # Support both flat and nested (CourseProcessor-native) layouts.
                    nested = p / "corpus" / "chunks.jsonl"
                    flat = p / "chunks.jsonl"
                    if nested.exists():
                        return nested.stat().st_mtime
                    if flat.exists():
                        return flat.stat().st_mtime
                    return 0.0
                trainforge_dir = max(candidates, key=_chunks_mtime)

        # --- Copy Trainforge outputs --------------------------------------
        # Worker β writes in CourseProcessor's native nested layout
        # (trainforge/corpus/chunks.jsonl, trainforge/graph/*.json). We
        # also check the flat layout for backward-compat with any caller
        # that mirrors the older stub's expected paths.
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
        _dest_chunks_path = course_dir / "corpus" / "chunks.jsonl"
        _had_prior_chunks = _dest_chunks_path.exists()
        if _had_prior_chunks:
            try:
                _dest_chunks_path.unlink()
            except OSError as _exc:
                logger.warning(
                    "archive_to_libv2: failed to remove prior-run "
                    "chunks.jsonl at %s: %s",
                    _dest_chunks_path,
                    _exc,
                )

        if trainforge_dir is not None and trainforge_dir.exists():
            copy_map = [
                (_pick(trainforge_dir / "corpus" / "chunks.jsonl",
                       trainforge_dir / "chunks.jsonl"),
                 course_dir / "corpus" / "chunks.jsonl", "chunks"),
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

        manifest = {
            "libv2_version": "1.2.0",
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

        manifest_path = course_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

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

        project_path = PROJECT_ROOT / "Courseforge" / "exports" / project_id
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
                ch_title = str(ch.get("title") or "")
                ch_topics = [ch_title]
                for sub in ch.get("sections") or []:
                    if isinstance(sub, dict):
                        ch_topics.append(str(sub.get("title") or ""))
                    elif isinstance(sub, str):
                        ch_topics.append(sub)
                topic_pool.append(_tokenize(" ".join(ch_topics)))
        if not topic_pool and objective_statements:
            for stmt in objective_statements:
                topic_pool.append(_tokenize(stmt))
        if not topic_pool and dart_blocks:
            # Final fallback: let DART block titles drive topic bags, one
            # per block, so each week gets at least a nominal signal.
            for blk in dart_blocks:
                topic_pool.append(blk["keywords"])

        # Distribute topic_pool across weeks (round-robin).
        for week_num in range(1, duration_weeks + 1):
            if not topic_pool:
                primary_bag: set = set()
            else:
                # Pick the topic whose index matches (week_num-1) mod len.
                primary_bag = topic_pool[(week_num - 1) % len(topic_pool)]
            for page_id in page_roles:
                # Application / self_check / summary share week bag;
                # content_0N gets the same bag plus a blend across
                # neighbor weeks so content doesn't duplicate overview.
                bag = set(primary_bag)
                if page_id.startswith("content") and len(topic_pool) > 1:
                    neighbor = topic_pool[(week_num) % len(topic_pool)]
                    bag = bag.union(neighbor)
                _set_week_page(week_num, page_id, bag)

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

        return json.dumps({
            "source_module_map_path": str(map_path),
            "source_chunk_ids": sorted(chunk_ids),
            "staging_dir": str(staging_dir) if staging_dir else "",
            "textbook_structure_path": textbook_structure_path,
            "routing_mode": routing_mode,
            "dart_blocks_indexed": len(dart_blocks),
            "weeks_routed": len(source_module_map),
            "course_name": course_name,
        })

    registry["build_source_module_map"] = _build_source_module_map

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
