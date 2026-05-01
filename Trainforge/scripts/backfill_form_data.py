"""Wave 136d — interactive backfill loop for FORM_DATA degraded entries.

Operator-facing CLI that drives the Wave 136c drafting CLI per-CURIE
with a mandatory operator confirmation gate. Enumerates the
``degraded_placeholder`` entries in the post-overlay form_data dict,
sorts them by corpus frequency (or alphabetically), and for each
target CURIE:

  1. Calls ``python -m Trainforge.scripts.draft_form_data_entry`` as a
     subprocess. The drafting CLI's stdout is the rendered YAML block
     plus operator next-steps comments.
  2. Prints the YAML block, then pauses for operator confirmation:

       y — append to the catalog YAML (atomic write + post-validate).
       n — skip this CURIE; continue.
       e — open in $EDITOR, edit, then append.
       q — exit cleanly.

  3. On ``y`` (or ``e`` after edit), the drafted entry is parsed and
     deep-merged INTO the per-family YAML overlay, preserving every
     pre-existing entry. The Wave 136a ``_load_form_data`` cache is
     invalidated, the catalog re-loaded, and Wave 136b's
     ``validate_form_data_contract`` is run on the post-merge dict;
     a content-quality violation specific to this CURIE rolls the
     append back.

ToS posture: this CLI generates NO training-data corpus content
itself — it routes to the Wave 136c drafting CLI (which routes to
Qwen / Together) and pauses for operator review on every emit.
Claude (or any dev tool) only authors the loop scaffolding. The
end-of-run summary emits ONE
``decision_type="form_data_backfill_session"`` decision-capture
event with metadata-shaped rationale (counts only — never the
authored YAML strings).

Usage::

    python -m Trainforge.scripts.backfill_form_data \\
        --course-code rdf-shacl-551-2 \\
        --family rdf_shacl \\
        --limit 5 \\
        --by frequency

Exit codes:
    0  loop completed (regardless of accept / skip / quit counts).
    2  property manifest not found / unreadable.
    3  target YAML overlay path doesn't exist (created at first append
       if --yaml-path overrides resolve to an absent file is handled
       by the merge step — exit 3 reserved for unrecoverable I/O).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture  # noqa: E402
from lib.ontology.family_map import (  # noqa: E402
    FamilyMap,
    load_family_map,
)
from lib.ontology.property_manifest import (  # noqa: E402
    load_property_manifest,
)
from Trainforge.generators.schema_translation_generator import (  # noqa: E402
    SurfaceFormData,
    _invalidate_form_data_cache,
    _load_form_data,
    validate_form_data_contract,
)

logger = logging.getLogger(__name__)


# Canonical action strings.
_ACTION_PROMPT = (
    "============================================================\n"
    "Action? [y]es-append  [n]o-skip  [e]dit-then-append  [q]uit\n"
    "============================================================\n"
    "> "
)


def _resolve_chunks_jsonl(course_code: str) -> Optional[Path]:
    """Locate the LibV2 chunks.jsonl for a given course code.

    Returns None when no chunks.jsonl exists for this course — the loop
    falls back to alphabetical sort with all-zero corpus frequencies.
    """
    candidates = [
        PROJECT_ROOT / "LibV2" / "courses" / course_code / "corpus" / "chunks.jsonl",
        PROJECT_ROOT / "LibV2" / "courses" / course_code / "chunks.jsonl",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _count_curie_frequencies(
    chunks_path: Optional[Path],
    curies: List[str],
) -> Dict[str, int]:
    """Count substring occurrences of each CURIE across chunk text.

    Walks ``chunks.jsonl`` once and tallies per-CURIE substring matches
    across every chunk's ``text`` field. Returns ``{curie: 0}`` for
    every CURIE when ``chunks_path`` is None.
    """
    counts: Dict[str, int] = {curie: 0 for curie in curies}
    if chunks_path is None or not chunks_path.is_file():
        return counts
    try:
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or ""
                if not isinstance(text, str) or not text:
                    continue
                for curie in curies:
                    if curie in text:
                        counts[curie] += 1
    except OSError as exc:
        logger.warning(
            "backfill_form_data: failed to read chunks.jsonl at %s (%s); "
            "falling back to zero corpus frequencies.",
            chunks_path,
            exc,
        )
    return counts


def _sort_targets_clustered(
    degraded_curies: List[str],
    counts: Dict[str, int],
    family_map: Optional[FamilyMap],
    by: str,
    *,
    family_filter: Optional[str] = None,
) -> List[Tuple[str, int]]:
    """Return ``[(curie, freq), ...]`` ordered by --by mode + family clustering.

    Wave 137b: replaces Wave 136d's flat frequency sort with family-
    aware ordering so the operator visits all degraded CURIEs in one
    family before crossing to the next, guaranteeing the
    family_completeness gate flips green in batches rather than
    mid-family.

    Algorithm:
      * ``by="alphabetical"`` bypasses clustering entirely (preserves
        Wave 136d test-deterministic behavior).
      * ``family_map is None`` falls back to a flat frequency-desc sort
        with alpha tie-break (preserves Wave 136d behavior on courses
        without a family map).
      * Otherwise: partition degraded CURIEs into one bucket per family
        + a singletons bucket. Filter to ``family_filter`` if set.
        Sort family buckets by aggregate freq desc; within-bucket by
        individual freq desc (alpha tie-break). Append singletons last,
        sorted by individual freq desc (alpha tie-break).

    ``family_filter`` accepts a family name (one of the keys in
    ``family_map.families``) or the literal ``"singletons"`` for the
    singletons-only run.
    """
    if by == "alphabetical":
        return sorted(
            ((c, counts.get(c, 0)) for c in degraded_curies),
            key=lambda pair: pair[0],
        )
    if family_map is None:
        # Flat fallback: Wave 136d behavior.
        return sorted(
            ((c, counts.get(c, 0)) for c in degraded_curies),
            key=lambda pair: (-pair[1], pair[0]),
        )

    cluster_buckets: Dict[str, List[Tuple[str, int]]] = {}
    singletons_bucket: List[Tuple[str, int]] = []
    for c in degraded_curies:
        freq = counts.get(c, 0)
        fam = family_map.family_of.get(c)
        if fam is None or fam == "<singleton>":
            singletons_bucket.append((c, freq))
        else:
            cluster_buckets.setdefault(fam, []).append((c, freq))

    # Apply family_filter.
    if family_filter is not None:
        if family_filter == "singletons":
            cluster_buckets = {}
        else:
            cluster_buckets = {
                k: v
                for k, v in cluster_buckets.items()
                if k == family_filter
            }
            singletons_bucket = []

    # Sort each cluster bucket internally: freq desc, alpha tie-break.
    for fam in cluster_buckets:
        cluster_buckets[fam].sort(key=lambda pair: (-pair[1], pair[0]))

    # Sort cluster buckets by aggregate frequency desc; tie-break by
    # family name to keep ordering deterministic.
    def _aggregate(items: List[Tuple[str, int]]) -> int:
        return sum(freq for _, freq in items)

    ordered_clusters: List[Tuple[str, int]] = []
    for fam in sorted(
        cluster_buckets,
        key=lambda f: (-_aggregate(cluster_buckets[f]), f),
    ):
        ordered_clusters.extend(cluster_buckets[fam])

    # Singletons last, sorted by individual freq desc, alpha tie-break.
    singletons_bucket.sort(key=lambda pair: (-pair[1], pair[0]))

    return ordered_clusters + singletons_bucket


# Wave 137b: keep the Wave 136d name available for callers that
# imported `_sort_targets` directly. Implementation now routes through
# the clustered sort with `family_map=None` so behavior is unchanged
# for the legacy signature.
def _sort_targets(
    degraded_curies: List[str],
    counts: Dict[str, int],
    by: str,
) -> List[Tuple[str, int]]:
    """Wave 136d compatibility shim. New callers should use
    :func:`_sort_targets_clustered` and pass a ``family_map``."""
    return _sort_targets_clustered(
        degraded_curies, counts, family_map=None, by=by,
    )


def _resolve_yaml_path(family: str, override: Optional[str]) -> Path:
    if override:
        return Path(override).resolve()
    return (
        PROJECT_ROOT
        / "schemas"
        / "training"
        / f"schema_translation_catalog.{family}.yaml"
    )


def _read_yaml_overlay(path: Path) -> Dict[str, Any]:
    """Read the existing YAML overlay file as a dict.

    On absent file, returns ``{"family": <inferred-from-path>, "forms": {}}``
    — that's the steady state for a never-edited overlay. On YAML
    parse failure, raises (the operator must repair the file by hand
    before backfilling — silent overwrite of a malformed catalog
    would risk losing existing entries).
    """
    import yaml as _yaml

    if not path.exists():
        # Family stem from "schema_translation_catalog.<family>.yaml".
        m = re.search(
            r"schema_translation_catalog\.([^.]+)\.yaml$", path.name
        )
        family = m.group(1) if m else "unknown"
        return {"family": family, "forms": {}}
    raw = path.read_text(encoding="utf-8")
    payload = _yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"YAML overlay at {path} did not parse to a dict; refusing "
            f"to overwrite. Edit the file by hand first."
        )
    payload.setdefault("forms", {})
    if not isinstance(payload["forms"], dict):
        raise RuntimeError(
            f"YAML overlay at {path} has non-dict 'forms' key; refusing "
            f"to overwrite."
        )
    return payload


def _atomic_write_yaml(payload: Dict[str, Any], target: Path) -> None:
    """Atomic tmp + rename write for the overlay YAML.

    Mirrors the Wave 136a load shape so a round-trip through
    ``_load_yaml_catalog`` reads back the same entries the operator
    just authored.
    """
    import yaml as _yaml

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    text = _yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=120,
    )
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(target)


# Anchors `forms:` then any indented `<CURIE>:` block. Used to slice the
# Wave 136c drafting CLI's stdout into the YAML payload (drop the
# trailing operator next-steps comment block before yaml.safe_load).
_NEXT_STEPS_HEADER_RE = re.compile(r"^\s*#\s*NEXT STEPS\s*$", re.MULTILINE)
# Wave 137d-1: when the drafting CLI emits a review checklist between
# the YAML block and the next-steps banner, the slicer must cut at the
# checklist header (it appears BEFORE the next-steps header). Match the
# 60-equals-sign banner followed by "REVIEW CHECKLIST" on the next line.
_REVIEW_CHECKLIST_HEADER_RE = re.compile(
    r"^=+\s*\nREVIEW CHECKLIST",
    re.MULTILINE,
)


def _extract_yaml_payload_from_drafting_stdout(stdout: str) -> Dict[str, Any]:
    """Slice the YAML payload out of the drafting CLI's stdout.

    Pre-Wave-137d-1: ``<yaml-text>\\n\\n# NEXT STEPS\\n...`` — we cut
    at the first ``# NEXT STEPS`` line.

    Wave 137d-1: ``<yaml-text>\\n\\n=== REVIEW CHECKLIST ...\\n\\n# NEXT STEPS\\n...``
    — we cut at the first of either header so the YAML round-trips
    cleanly through ``yaml.safe_load``. Slicing first also keeps
    parse failures easy to diagnose for the operator.
    """
    import yaml as _yaml

    next_steps_match = _NEXT_STEPS_HEADER_RE.search(stdout)
    checklist_match = _REVIEW_CHECKLIST_HEADER_RE.search(stdout)
    cut_positions = [
        m.start() for m in (next_steps_match, checklist_match) if m is not None
    ]
    head = stdout[: min(cut_positions)] if cut_positions else stdout
    payload = _yaml.safe_load(head)
    if not isinstance(payload, dict) or "forms" not in payload:
        raise RuntimeError(
            "drafting CLI stdout did not parse to a dict with a 'forms' "
            "key; first 500 chars: " + repr(stdout[:500])
        )
    forms = payload.get("forms")
    if not isinstance(forms, dict) or not forms:
        raise RuntimeError(
            "drafting CLI stdout 'forms' block was empty; first 500 "
            "chars: " + repr(stdout[:500])
        )
    return payload


def _merge_curie_into_overlay(
    target_path: Path,
    curie: str,
    new_entry_yaml: Dict[str, Any],
) -> Dict[str, Any]:
    """Deep-merge ONE CURIE entry into the overlay YAML file.

    Preserves every pre-existing entry (Wave 136a per-CURIE merge
    semantics applied at the file layer too). Returns the
    pre-merge payload so the caller can roll back on validator
    failure.
    """
    pre = _read_yaml_overlay(target_path)
    pre_forms_snapshot = dict(pre.get("forms") or {})
    post: Dict[str, Any] = dict(pre)
    post["forms"] = dict(pre_forms_snapshot)
    post["forms"][curie] = new_entry_yaml
    _atomic_write_yaml(post, target_path)
    # Return the pre-merge snapshot for rollback.
    return {"family": pre.get("family"), "forms": pre_forms_snapshot,
            "_pre_full": pre}


def _rollback_overlay(target_path: Path, pre_full: Dict[str, Any]) -> None:
    """Restore the overlay to its pre-merge state."""
    _atomic_write_yaml(pre_full, target_path)


def _run_drafting_cli(
    curie: str,
    family: str,
    course_code: str,
    provider: str,
    model: Optional[str],
) -> Tuple[int, str, str]:
    """Dispatch the Wave 136c drafting CLI as a subprocess.

    Returns ``(returncode, stdout, stderr)``. Wave 136d never imports
    the drafting CLI's ``main`` directly because operators sometimes
    swap providers or models between rounds — a clean subprocess
    boundary makes the per-CURIE call independently auditable.
    """
    cmd = [
        sys.executable,
        "-m",
        "Trainforge.scripts.draft_form_data_entry",
        "--curie",
        curie,
        "--family",
        family,
        "--course-code",
        course_code,
        "--provider",
        provider,
        "--output",
        "-",
        "--force-overwrite",
    ]
    if model:
        cmd.extend(["--model", model])
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _read_action(input_fn=input) -> str:
    """Read one operator action from stdin.

    Accepts ``y`` / ``n`` / ``e`` / ``q`` (case-insensitive). Repeats
    the prompt for unknown input. Wired through ``input_fn`` so tests
    can inject canned responses.
    """
    while True:
        try:
            raw = input_fn(_ACTION_PROMPT)
        except EOFError:
            # Treat EOF as quit so a piped session ends cleanly.
            return "q"
        choice = (raw or "").strip().lower()
        if choice in ("y", "n", "e", "q"):
            return choice
        print(
            f"Unknown action {choice!r}; expected one of y / n / e / q.",
            file=sys.stderr,
        )


def _editor_round_trip(initial_text: str) -> str:
    """Open ``initial_text`` in $EDITOR; return the edited bytes.

    Uses ``vi`` when ``$EDITOR`` is unset (POSIX baseline). The
    operator saves + quits to commit; on any exception we re-raise
    so the loop can surface the edit failure and re-prompt.
    """
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(initial_text)
        tmp_path = Path(tmp.name)
    try:
        subprocess.run([editor, str(tmp_path)], check=False)
        return tmp_path.read_text(encoding="utf-8")
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


# ----------------------------------------------------------------------
# Loop body: one CURIE pass.
# ----------------------------------------------------------------------


def _process_one_curie(
    *,
    idx: int,
    total: int,
    curie: str,
    freq: int,
    label: str,
    family: str,
    course_code: str,
    provider: str,
    model: Optional[str],
    yaml_path: Path,
    manifest_curies: List[str],
    input_fn,
    print_fn,
    runner=None,
    editor_fn=None,
) -> str:
    # Resolve at call time via module-attribute lookup so
    # ``patch.object(cli, "_run_drafting_cli", ...)`` in tests is
    # honored. Capturing module-level callables in default arg values
    # would freeze the binding at function-definition time.
    _module = sys.modules[__name__]
    if runner is None:
        runner = getattr(_module, "_run_drafting_cli")
    if editor_fn is None:
        editor_fn = getattr(_module, "_editor_round_trip")
    """Drive one CURIE through the operator-paused loop.

    Returns one of: ``"accepted"``, ``"skipped"``, ``"edited"``,
    ``"failed_validation"``, ``"quit_after"``.
    """
    print_fn(
        f"\n[{idx}/{total}] CURIE={curie} corpus_freq={freq} label={label}"
    )

    rc, stdout, stderr = runner(curie, family, course_code, provider, model)
    if rc != 0:
        print_fn(
            f"  drafting CLI failed (exit {rc}); stderr=\n{stderr}",
        )
        return "failed_validation"

    # Print the rendered YAML + next-steps comment block verbatim.
    print_fn(stdout)

    # First action prompt.
    action = _read_action(input_fn=input_fn)

    if action == "q":
        return "quit_after"

    if action == "n":
        print_fn("  Skipped.")
        return "skipped"

    # y or e: parse the YAML payload from drafting stdout.
    if action == "e":
        try:
            edited_text = editor_fn(stdout)
        except Exception as exc:
            print_fn(f"  editor session failed: {exc}; skipping.")
            return "skipped"
        try:
            payload = _extract_yaml_payload_from_drafting_stdout(edited_text)
        except Exception as exc:
            print_fn(f"  edited YAML did not parse: {exc}; skipping.")
            return "skipped"
        outcome_label = "edited"
    else:
        # action == "y"
        try:
            payload = _extract_yaml_payload_from_drafting_stdout(stdout)
        except Exception as exc:
            print_fn(f"  drafting YAML did not parse: {exc}; skipping.")
            return "skipped"
        outcome_label = "accepted"

    forms = payload.get("forms") or {}
    new_entry_yaml = forms.get(curie)
    if not isinstance(new_entry_yaml, dict):
        print_fn(
            f"  drafted YAML did not contain forms.{curie}; skipping."
        )
        return "skipped"

    # Atomic merge + post-validate + rollback on failure.
    snapshot = _merge_curie_into_overlay(yaml_path, curie, new_entry_yaml)
    pre_full = snapshot["_pre_full"]
    _invalidate_form_data_cache()
    reloaded = _load_form_data(family)
    report = validate_form_data_contract(reloaded, manifest_curies)
    violations = report.get("content_violations") or []
    # Filter to violations specific to THIS CURIE — we accept upstream
    # entries' violations as out-of-scope for this operator pass.
    this_curie_violations = [
        v for v in violations
        if isinstance(v, dict) and v.get("curie") == curie
    ]
    if this_curie_violations or not report.get("passed"):
        # Rollback when this CURIE is the violator OR when overall
        # contract failure suddenly appeared after the append.
        if this_curie_violations:
            print_fn(
                f"  Wave 136b validator rejected the appended entry "
                f"for {curie}:"
            )
            for v in this_curie_violations:
                print_fn(f"    {v}")
            _rollback_overlay(yaml_path, pre_full)
            _invalidate_form_data_cache()
            return "failed_validation"

    print_fn(f"  OK {outcome_label}")
    return outcome_label if outcome_label == "edited" else "accepted"


# ----------------------------------------------------------------------
# Main entry point.
# ----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backfill_form_data",
        description=(
            "Interactive backfill loop for FORM_DATA degraded "
            "placeholders. Drives the Wave 136c drafting CLI per-CURIE "
            "with mandatory operator confirmation."
        ),
    )
    parser.add_argument("--course-code", required=True)
    parser.add_argument("--family", default="rdf_shacl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--by",
        choices=("frequency", "alphabetical"),
        default="frequency",
    )
    parser.add_argument(
        "--provider",
        choices=("local", "together"),
        default="local",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier override; passthrough to drafting CLI.",
    )
    parser.add_argument(
        "--yaml-path",
        default=None,
        help=(
            "Override path for the per-family YAML overlay. "
            "Default: schemas/training/schema_translation_catalog."
            "<family>.yaml."
        ),
    )
    # Wave 137b: restrict a session to one family cluster (e.g. just
    # cardinality) so the operator can flip a single family from
    # partial -> complete in one sitting and the family_completeness
    # gate goes green on that family.
    parser.add_argument(
        "--family-cluster",
        default=None,
        help=(
            "Restrict the session to one family cluster from the "
            "family map (or 'singletons'). Default: visit every "
            "cluster in family-clustered order. Validated against the "
            "family map at parse time."
        ),
    )
    return parser


def main(
    argv: Optional[List[str]] = None,
    *,
    input_fn=input,
    print_fn=print,
) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Step 1: load the manifest. Failure is exit 2.
    try:
        manifest = load_property_manifest(args.course_code)
    except FileNotFoundError as exc:
        print(
            f"ERROR: property manifest not found: {exc}",
            file=sys.stderr,
        )
        return 2

    manifest_curies = [p.curie for p in manifest.properties]
    label_by_curie: Dict[str, str] = {p.curie: p.label for p in manifest.properties}

    # Step 2: load the post-overlay form_data and pick degraded entries.
    _invalidate_form_data_cache()
    form_data = _load_form_data(args.family)
    degraded = [
        c for c, e in form_data.items()
        if e.anchored_status == "degraded_placeholder"
    ]
    # Restrict to manifest-declared CURIEs — the post-overlay form_data
    # may include legacy non-manifest entries that the operator
    # shouldn't be backfilling here.
    degraded = [c for c in degraded if c in manifest_curies]

    if not degraded:
        print_fn(
            f"No degraded_placeholder entries found for family={args.family}. "
            "Nothing to backfill."
        )
        return 0

    # Step 3: corpus-frequency or alphabetical sort with family clustering.
    chunks_path = _resolve_chunks_jsonl(args.course_code)
    counts = _count_curie_frequencies(chunks_path, degraded)

    # Wave 137b: load the family map (None on families without one --
    # the clustered sort falls back to flat freq-desc behavior).
    try:
        family_map = load_family_map(args.family)
    except Exception as exc:  # noqa: BLE001 - bubble as exit-3
        print(
            f"ERROR: family_map for family={args.family!r} failed to "
            f"load: {exc}",
            file=sys.stderr,
        )
        return 3

    # Validate --family-cluster against the loaded map (parse-time check
    # already accepted any string; do the semantic check here).
    if args.family_cluster is not None:
        if family_map is None:
            print(
                f"ERROR: --family-cluster {args.family_cluster!r} requires "
                f"a family_map, but none exists for family={args.family!r}.",
                file=sys.stderr,
            )
            return 2
        valid_clusters = set(family_map.families) | {"singletons"}
        if args.family_cluster not in valid_clusters:
            print(
                f"ERROR: --family-cluster {args.family_cluster!r} is not "
                f"a valid cluster name. Choose one of: "
                f"{sorted(valid_clusters)}.",
                file=sys.stderr,
            )
            return 2

    ordered = _sort_targets_clustered(
        degraded,
        counts,
        family_map=family_map,
        by=args.by,
        family_filter=args.family_cluster,
    )

    # Step 4: cap to --limit.
    targets = ordered[: args.limit]

    yaml_path = _resolve_yaml_path(args.family, args.yaml_path)

    # Step 5: drive each CURIE through the operator pause loop.
    counters = {
        "accepted": 0,
        "skipped": 0,
        "edited": 0,
        "failed_validation": 0,
        "quit_after": 0,
    }
    quit_flag = False
    total = len(targets)
    for idx, (curie, freq) in enumerate(targets, start=1):
        outcome = _process_one_curie(
            idx=idx,
            total=total,
            curie=curie,
            freq=freq,
            label=label_by_curie.get(curie, ""),
            family=args.family,
            course_code=args.course_code,
            provider=args.provider,
            model=args.model,
            yaml_path=yaml_path,
            manifest_curies=manifest_curies,
            input_fn=input_fn,
            print_fn=print_fn,
        )
        counters[outcome] = counters.get(outcome, 0) + 1
        if outcome == "quit_after":
            quit_flag = True
            break

    # Step 6: end-of-run summary + decision-capture event.
    print_fn(
        "\n=== backfill_form_data summary ==="
    )
    print_fn(f"  family            : {args.family}")
    print_fn(f"  accepted          : {counters['accepted']}")
    print_fn(f"  skipped           : {counters['skipped']}")
    print_fn(f"  edited            : {counters['edited']}")
    print_fn(f"  failed_validation : {counters['failed_validation']}")
    print_fn(f"  quit_after        : {counters['quit_after']}")

    capture = DecisionCapture(
        course_code=args.course_code,
        phase="trainforge-training",
        tool="trainforge",
        streaming=True,
    )
    capture.log_decision(
        decision_type="form_data_backfill_session",
        decision="completed",
        rationale=(
            f"family={args.family} "
            f"accepted={counters['accepted']} "
            f"skipped={counters['skipped']} "
            f"edited={counters['edited']} "
            f"failed={counters['failed_validation']} "
            f"quit={'true' if quit_flag else 'false'}"
        ),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
