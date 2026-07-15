"""LibV2 library-format migration framework (OP4 stage 2).

Stage 1 (shipped earlier) added only the ``library_format_version`` *stamp* on
freshly-emitted ``course_manifest.json`` (``LIBRARY_FORMAT_VERSION`` in
``MCP/tools/pipeline_tools.py``) plus the report-only contract doc
(``docs/operations/library-versioning.md``). This module is stage 2: the actual
in-place *upgrader* the contract deferred — version detection, an ordered
registry of migration steps, and a plan/apply engine that chains steps to bring
an old-format course dir to the current layout version.

Design contract (mirrors ``docs/operations/library-versioning.md``):

* **Version detection.** A manifest with no ``library_format_version`` predates
  the OP4 stamp and is treated as the pre-1.0 baseline (the ``LEGACY_VERSION``
  sentinel). Any present value is the on-disk layout contract version.
* **Ordered registry.** Migration steps are keyed ``from_version -> to_version``.
  The engine chains them from the detected version to the registry's terminal
  (current) version. There is exactly one outgoing step per version, so the
  chain is deterministic and cycle-free.
* **Dry-run by default; apply is safe.** ``plan_course_migration`` never writes.
  ``apply_course_migration`` backs up ``manifest.json`` (timestamped ``.bak``
  sibling) BEFORE writing, then re-runs the existing LibV2 validate check on the
  migrated course and **rolls the manifest back** if validation fails — never a
  silent half-migrated course.

The baseline step registered here is ``legacy -> 1.0``: it stamps
``library_format_version = "1.0"`` after the manifest otherwise conforms
(post-write validation is the enforcement seam). ``1.0 -> 1.0`` is expressed as
the empty "already current" plan.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Sentinel version for a manifest that carries no ``library_format_version``
# key at all — the pre-OP4-stamp, pre-1.0 baseline layout. Not a real
# ``^\d+\.\d+$`` version string on purpose so it can never collide with a
# stamped value.
LEGACY_VERSION = "legacy"


class MigrationError(Exception):
    """Typed migration failure carrying a ``(code, detail)`` pair.

    ``code`` is a short machine-stable slug (``UNKNOWN_VERSION``,
    ``MANIFEST_NOT_FOUND``, ``MANIFEST_INVALID_JSON``, ``VALIDATION_FAILED``);
    the CLI renders ``detail`` and exits non-zero.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


# A migration transform takes the parsed manifest dict and returns a NEW dict
# (it must not mutate its argument in place — the caller keeps the original for
# rollback comparison).
ManifestTransform = Callable[[dict], dict]

# A corpus transform operates on the whole course DIRECTORY (not just its
# manifest dict) — it rewrites on-disk artifact BODIES (chunks.jsonl sourceIds,
# ``data-*`` HTML attrs) and renames chunkset dirs, then re-derives the content
# hashes. It takes ``(course_dir, apply)`` and returns a
# :class:`CorpusMigrationReport`. ``apply=False`` MUST NOT write (it only
# computes the plan/counts); ``apply=True`` writes with the caller's
# backup/rollback wrapper already in force. Steps that only touch the manifest
# leave this ``None``.
CorpusTransform = Callable[[Path, bool], "CorpusMigrationReport"]


@dataclass(frozen=True)
class MigrationStep:
    """A single ordered layout-version migration.

    ``transform`` receives a deep-ish copy of the manifest and returns the
    migrated manifest. It MUST stamp ``library_format_version = to_version`` (or
    the engine will refuse the result as a no-op that never advanced).

    ``corpus_transform`` (optional) additionally rewrites the on-disk corpus
    BODIES for steps that are not manifest-neutral (e.g. the DART->semantik
    naming purge). It is ``None`` for pure manifest-stamp steps.
    """

    from_version: str
    to_version: str
    description: str
    transform: ManifestTransform
    corpus_transform: Optional[CorpusTransform] = None


def detect_version(manifest: dict) -> str:
    """Return the on-disk layout version declared by a manifest dict.

    A missing/empty ``library_format_version`` => :data:`LEGACY_VERSION` (the
    pre-1.0 baseline). Any present value is returned verbatim (as ``str``).
    """
    value = manifest.get("library_format_version")
    if not value:
        return LEGACY_VERSION
    return str(value)


class MigrationRegistry:
    """An ordered registry of layout-migration steps, keyed by ``from_version``.

    Exactly one step may be registered per ``from_version`` (a version has a
    single deterministic successor). The registry's *target* (current) version
    is the terminus of the chain that starts at :data:`LEGACY_VERSION`.
    """

    def __init__(self) -> None:
        self._steps: Dict[str, MigrationStep] = {}

    def register(self, step: MigrationStep) -> None:
        if step.from_version in self._steps:
            raise MigrationError(
                "DUPLICATE_STEP",
                f"a migration step is already registered from "
                f"{step.from_version!r}",
            )
        if step.from_version == step.to_version:
            raise MigrationError(
                "SELF_STEP",
                f"a migration step cannot go from {step.from_version!r} to "
                f"itself",
            )
        self._steps[step.from_version] = step

    def get(self, from_version: str) -> Optional[MigrationStep]:
        return self._steps.get(from_version)

    def target_version(self) -> str:
        """The current layout version — the terminus reachable from legacy.

        Follows the chain from :data:`LEGACY_VERSION` through the registered
        steps until a version with no outgoing step is reached. With no steps
        registered the target is :data:`LEGACY_VERSION` itself.
        """
        cur = LEGACY_VERSION
        seen = {cur}
        while True:
            step = self._steps.get(cur)
            if step is None:
                return cur
            cur = step.to_version
            if cur in seen:  # pragma: no cover — registration guards prevent cycles
                raise MigrationError(
                    "CYCLE", f"migration cycle detected at {cur!r}"
                )
            seen.add(cur)

    def plan_steps(self, current_version: str) -> List[MigrationStep]:
        """Ordered steps to advance ``current_version`` to the target version.

        Empty list => already current. Raises :class:`MigrationError`
        (``UNKNOWN_VERSION``) when ``current_version`` cannot reach the target
        (an unrecognized version with no registered outgoing step).
        """
        target = self.target_version()
        if current_version == target:
            return []
        steps: List[MigrationStep] = []
        seen = {current_version}
        cur = current_version
        while cur != target:
            step = self._steps.get(cur)
            if step is None:
                raise MigrationError(
                    "UNKNOWN_VERSION",
                    f"no migration path from library_format_version "
                    f"{current_version!r} to current {target!r} "
                    f"(no step registered from {cur!r})",
                )
            steps.append(step)
            cur = step.to_version
            if cur in seen:  # pragma: no cover — registration guards prevent cycles
                raise MigrationError("CYCLE", f"migration cycle detected at {cur!r}")
            seen.add(cur)
        return steps


# --------------------------------------------------------------------------- #
# Baseline migration: legacy (missing field) -> 1.0                            #
# --------------------------------------------------------------------------- #

def _stamp_v1_0(manifest: dict) -> dict:
    """legacy -> 1.0: stamp ``library_format_version = "1.0"``.

    Layout-neutral by construction — the pre-1.0 baseline directory layout IS
    the 1.0 layout; only the explicit stamp was missing. The transform copies
    the manifest and adds the field; the manifest "otherwise conforming" is
    enforced by the post-write LibV2 validate pass (rollback on failure), not
    by mutating any other key here.
    """
    migrated = dict(manifest)
    migrated["library_format_version"] = "1.0"
    return migrated


def build_default_registry() -> MigrationRegistry:
    """The canonical registry of shipped layout migrations."""
    registry = MigrationRegistry()
    registry.register(
        MigrationStep(
            from_version=LEGACY_VERSION,
            to_version="1.0",
            description=(
                "Stamp library_format_version=1.0 on a pre-OP4 manifest "
                "(pre-1.0 baseline layout; no directory-layout change)."
            ),
            transform=_stamp_v1_0,
        )
    )
    return registry


DEFAULT_REGISTRY = build_default_registry()


# --------------------------------------------------------------------------- #
# Plan / apply engine                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class MigrationPlan:
    """A dry-run plan for one course — computed, never written."""

    slug: str
    course_dir: Path
    from_version: str
    to_version: str
    steps: List[MigrationStep] = field(default_factory=list)

    @property
    def already_current(self) -> bool:
        return not self.steps

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "course_dir": str(self.course_dir),
            "from_version": self.from_version,
            "to_version": self.to_version,
            "already_current": self.already_current,
            "steps": [
                {
                    "from_version": s.from_version,
                    "to_version": s.to_version,
                    "description": s.description,
                }
                for s in self.steps
            ],
        }


@dataclass
class MigrationApplyResult:
    """The outcome of an :func:`apply_course_migration` run."""

    slug: str
    course_dir: Path
    from_version: str
    to_version: str
    applied: bool
    already_current: bool = False
    steps_applied: List[str] = field(default_factory=list)
    backup_path: Optional[Path] = None
    rolled_back: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "course_dir": str(self.course_dir),
            "from_version": self.from_version,
            "to_version": self.to_version,
            "applied": self.applied,
            "already_current": self.already_current,
            "steps_applied": self.steps_applied,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "rolled_back": self.rolled_back,
            "validation_errors": self.validation_errors,
        }


def _manifest_path(course_dir: Path) -> Path:
    return course_dir / "manifest.json"


def _load_manifest(course_dir: Path) -> dict:
    path = _manifest_path(course_dir)
    if not path.exists():
        raise MigrationError(
            "MANIFEST_NOT_FOUND", f"manifest.json not found in {course_dir}"
        )
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise MigrationError(
            "MANIFEST_INVALID_JSON", f"invalid JSON in manifest.json: {exc}"
        ) from exc


def plan_course_migration(
    course_dir: Path,
    slug: Optional[str] = None,
    registry: MigrationRegistry = DEFAULT_REGISTRY,
) -> MigrationPlan:
    """Compute (never write) the migration plan for one course dir."""
    slug = slug or course_dir.name
    manifest = _load_manifest(course_dir)
    from_version = detect_version(manifest)
    target = registry.target_version()
    steps = registry.plan_steps(from_version)
    return MigrationPlan(
        slug=slug,
        course_dir=course_dir,
        from_version=from_version,
        to_version=target,
        steps=steps,
    )


# The default post-migration verification is the existing per-course LibV2
# validate pass (structure + manifest-schema + taxonomy). It is injectable so
# callers/tests can substitute a lighter or stricter check (e.g. the whole-repo
# ``lib.libv2_fsck.run_fsck`` is heavier and repo-scoped; validate_course is the
# right per-course granularity for an in-place manifest migration).
Validator = Callable[[Path, Path], "object"]


def _default_validator(course_dir: Path, repo_root: Path):
    from .validator import validate_course  # local import: keep module import cheap

    return validate_course(course_dir, repo_root)


def apply_course_migration(
    course_dir: Path,
    repo_root: Path,
    slug: Optional[str] = None,
    registry: MigrationRegistry = DEFAULT_REGISTRY,
    validator: Optional[Validator] = None,
    backup: bool = True,
) -> MigrationApplyResult:
    """Migrate one course in place, with backup + validate + rollback safety.

    Steps:

    1. Load + detect the current layout version and compute the plan.
    2. If already current, return a no-op result (nothing written).
    3. Back up ``manifest.json`` to a timestamped ``.bak`` sibling.
    4. Apply the chained transforms in memory, then write the migrated manifest.
    5. Run the LibV2 validate check on the migrated course. On failure, restore
       the manifest from the backup and mark ``rolled_back=True``.
    """
    slug = slug or course_dir.name
    validator = validator or _default_validator

    manifest = _load_manifest(course_dir)
    from_version = detect_version(manifest)
    target = registry.target_version()
    steps = registry.plan_steps(from_version)

    if not steps:
        return MigrationApplyResult(
            slug=slug,
            course_dir=course_dir,
            from_version=from_version,
            to_version=target,
            applied=False,
            already_current=True,
        )

    manifest_path = _manifest_path(course_dir)

    # (3) Back up BEFORE any write.
    backup_path: Optional[Path] = None
    if backup:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = manifest_path.with_name(f"manifest.json.{ts}.bak")
        shutil.copy2(manifest_path, backup_path)

    # (4) Apply transforms in memory.
    migrated = manifest
    steps_applied: List[str] = []
    for step in steps:
        migrated = step.transform(migrated)
        if detect_version(migrated) != step.to_version:
            # A transform that failed to advance the stamp is a programming
            # error, not an operator error — restore and fail loudly.
            if backup_path is not None:
                shutil.copy2(backup_path, manifest_path)
            raise MigrationError(
                "STEP_NO_ADVANCE",
                f"migration step {step.from_version}->{step.to_version} did "
                f"not stamp library_format_version to {step.to_version!r}",
            )
        steps_applied.append(f"{step.from_version}->{step.to_version}")

    # Write the migrated manifest.
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(migrated, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # (5) Validate; roll back on failure.
    result = validator(course_dir, repo_root)
    valid = bool(getattr(result, "valid", True))
    if not valid:
        errors = list(getattr(result, "errors", []))
        if backup_path is not None:
            shutil.copy2(backup_path, manifest_path)
        return MigrationApplyResult(
            slug=slug,
            course_dir=course_dir,
            from_version=from_version,
            to_version=target,
            applied=False,
            steps_applied=steps_applied,
            backup_path=backup_path,
            rolled_back=True,
            validation_errors=errors,
        )

    return MigrationApplyResult(
        slug=slug,
        course_dir=course_dir,
        from_version=from_version,
        to_version=target,
        applied=True,
        steps_applied=steps_applied,
        backup_path=backup_path,
    )


def discover_courses(repo_root: Path) -> List[Path]:
    """Every course dir under ``<repo_root>/courses`` (sorted, dotfiles skipped)."""
    courses_dir = repo_root / "courses"
    if not courses_dir.exists():
        return []
    return sorted(
        cd
        for cd in courses_dir.iterdir()
        if cd.is_dir() and not cd.name.startswith(".")
    )


# --------------------------------------------------------------------------- #
# DART -> semantik naming purge (Stage S2 — corpus-body migration)            #
# --------------------------------------------------------------------------- #
#
# The naming purge (ratified 2026-07-11) renames the on-disk DART-staged
# chunkset from the legacy ``dart`` naming to the ratified ``semantik`` naming,
# WITHOUT touching the ``{slug}#{block_id}`` payload of any sourceId or the
# VALUE of any ``data-*`` attribute — only the ``dart:`` / ``data-dart-`` /
# ``dart_chunks`` naming tokens flip. Concretely, per course dir this migration:
#
#   * rewrites ``dart_chunks/chunks.jsonl`` sourceIds ``dart:`` -> ``semantik:``
#     (only the ``"sourceId": "dart:`` prefix; the rest of the id is untouched)
#     AND any ``data-dart-`` attr names embedded in chunk ``html`` fields;
#   * rewrites ``data-dart-`` -> ``data-semantik-`` attr NAMES in the course's
#     archived HTML files (``source/**/*.html``), leaving attr VALUES intact
#     (e.g. ``data-dart-source="dart_converter"`` -> the reader keys on the attr
#     NAME, so the provenance value ``dart_converter`` is preserved verbatim);
#   * recomputes the chunkset content SHA over the rewritten ``chunks.jsonl``
#     and writes it under the semantik-named key ``semantik_chunks_sha256``
#     (top-level manifest) + the sidecar ``chunks_sha256``, flips the sidecar
#     ``chunkset_kind`` ``dart`` -> ``semantik``, and RENAMES the sidecar
#     provenance key ``source_dart_html_sha256`` -> ``source_semantik_html_sha256``
#     PRESERVING its value (see note below);
#   * renames the directory ``dart_chunks/`` -> ``semantik_chunks/`` to match the
#     reader's ratified preference (``lib/libv2_storage.py::SEMANTIK_CHUNKS_DIRNAME``
#     wins over ``dart_chunks/`` in ``resolve_imscc_chunks_dir``).
#
# ``source_dart_html_sha256`` value is PRESERVED (key renamed only): that hash
# is the "aggregate Merkle of the staged DART HTML inputs" computed at chunking
# time over the TRANSIENT staging dir, which no longer exists on disk. It is NOT
# derivable from the archived ``source/html/`` copies (which may differ from the
# staged inputs and, post-migration, carry the flipped attr names). Recomputing
# it from ``source/html/`` would corrupt the provenance anchor, so the migration
# renames the key while keeping the historical value. No on-disk validator
# recomputes this field (only the chunkset ``chunks_sha256`` and the top-level
# ``*_chunks_sha256`` are hash-checked), so the anchor stays honest.
#
# NOTE ON THE VERSION BUMP: this step is expressed as ``1.0 -> 1.1`` in the
# dedicated :func:`build_dart_purge_registry`, NOT in :data:`DEFAULT_REGISTRY`.
# The DEFAULT registry's terminus is drift-guarded against
# ``MCP.tools.pipeline_tools.LIBRARY_FORMAT_VERSION`` (still ``"1.0"``), and the
# emitter still WRITES ``dart_chunks/`` — bumping DEFAULT now would stamp a
# layout version that contradicts what the pipeline emits. The DEFAULT bump
# lands with the emitter flip (a later, separately-gated stage); this stage
# ships only the corpus-migration machinery + dry-run.

DART_SOURCEID_RE = re.compile(r'("sourceId"\s*:\s*")dart:')
_DART_ATTR_TOKEN = "data-dart-"
_SEMANTIK_ATTR_TOKEN = "data-semantik-"
_DART_CHUNKS_DIRNAME = "dart_chunks"
_SEMANTIK_CHUNKS_DIRNAME = "semantik_chunks"
_CHUNKS_FILENAME = "chunks.jsonl"
_SIDECAR_FILENAME = "manifest.json"


@dataclass
class CorpusMigrationReport:
    """Per-course plan/outcome for the DART->semantik corpus-body migration.

    In dry-run (``apply=False``) every ``*_changed`` count is the number that
    WOULD change and no file is written. On apply the same counts describe what
    was written; ``applied`` / ``rolled_back`` / ``backup_dir`` record the
    outcome of the backup+validate+rollback wrapper.
    """

    slug: str
    course_dir: Path
    applicable: bool = False          # a dart_chunks/chunks.jsonl was found
    already_migrated: bool = False    # semantik_chunks/ present, no dart_chunks/
    # chunks.jsonl body
    chunks_lines_total: int = 0
    chunks_lines_changed: int = 0
    sourceids_changed: int = 0
    chunk_html_attrs_changed: int = 0
    # archived HTML files
    html_files_total: int = 0
    html_files_changed: int = 0
    html_attrs_changed: int = 0
    # sidecar manifest
    sidecar_present: bool = False
    sidecar_kind_flipped: bool = False
    sidecar_source_key_renamed: bool = False
    # top-level manifest
    top_manifest_present: bool = False
    top_manifest_key_renamed: bool = False
    # hashes
    old_chunks_sha256: Optional[str] = None
    new_chunks_sha256: Optional[str] = None
    dir_renamed: bool = False         # dart_chunks -> semantik_chunks (planned/done)
    # diagnostics
    nonconforming_reasons: List[str] = field(default_factory=list)
    advisories: List[str] = field(default_factory=list)
    # apply-only outcome
    applied: bool = False
    rolled_back: bool = False
    backup_dir: Optional[Path] = None
    validation_errors: List[str] = field(default_factory=list)

    @property
    def would_change_anything(self) -> bool:
        return bool(
            self.chunks_lines_changed
            or self.html_files_changed
            or self.sidecar_kind_flipped
            or self.sidecar_source_key_renamed
            or self.top_manifest_key_renamed
            or self.dir_renamed
        )

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "course_dir": str(self.course_dir),
            "applicable": self.applicable,
            "already_migrated": self.already_migrated,
            "chunks_lines_total": self.chunks_lines_total,
            "chunks_lines_changed": self.chunks_lines_changed,
            "sourceids_changed": self.sourceids_changed,
            "chunk_html_attrs_changed": self.chunk_html_attrs_changed,
            "html_files_total": self.html_files_total,
            "html_files_changed": self.html_files_changed,
            "html_attrs_changed": self.html_attrs_changed,
            "sidecar_present": self.sidecar_present,
            "sidecar_kind_flipped": self.sidecar_kind_flipped,
            "sidecar_source_key_renamed": self.sidecar_source_key_renamed,
            "top_manifest_present": self.top_manifest_present,
            "top_manifest_key_renamed": self.top_manifest_key_renamed,
            "old_chunks_sha256": self.old_chunks_sha256,
            "new_chunks_sha256": self.new_chunks_sha256,
            "dir_renamed": self.dir_renamed,
            "nonconforming_reasons": list(self.nonconforming_reasons),
            "advisories": list(self.advisories),
            "applied": self.applied,
            "rolled_back": self.rolled_back,
            "backup_dir": str(self.backup_dir) if self.backup_dir else None,
            "validation_errors": list(self.validation_errors),
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rename_manifest_key(manifest: dict, old_key: str, new_key: str, new_value):
    """Return a NEW manifest dict with ``old_key`` renamed to ``new_key`` (value
    ``new_value``), preserving key order. If ``old_key`` is absent the manifest
    is returned unchanged (a copy)."""
    out: dict = {}
    renamed = False
    for k, v in manifest.items():
        if k == old_key:
            out[new_key] = new_value
            renamed = True
        else:
            out[k] = v
    return out, renamed


def _iter_html_files(course_dir: Path) -> List[Path]:
    """Archived HTML files eligible for attr rewrite.

    Scoped to ``source/**/*.html`` (where archived accessible HTML lives) and
    any top-level ``*.html``. Skips backup siblings (``*.bak`` and
    ``*.<ts>.bak/`` dirs) and the chunkset dirs (their JSONL is handled
    separately)."""
    html: List[Path] = []
    source_dir = course_dir / "source"
    roots = []
    if source_dir.is_dir():
        roots.append(source_dir)
    for root in roots:
        for p in sorted(root.rglob("*.html")):
            if ".bak" in p.name or any(part.endswith(".bak") for part in p.parts):
                continue
            html.append(p)
    for p in sorted(course_dir.glob("*.html")):
        html.append(p)
    return html


def _rewrite_chunks_text(text: str) -> Tuple[str, int, int, int]:
    """Rewrite chunks.jsonl text. Returns
    ``(new_text, lines_changed, sourceids_changed, attrs_changed)``."""
    out_lines: List[str] = []
    lines_changed = sourceids_changed = attrs_changed = 0
    # splitlines(keepends=True) preserves the exact trailing-newline layout.
    for line in text.splitlines(keepends=True):
        new_line, n_src = DART_SOURCEID_RE.subn(r"\1semantik:", line)
        n_attr = new_line.count(_DART_ATTR_TOKEN)
        if n_attr:
            new_line = new_line.replace(_DART_ATTR_TOKEN, _SEMANTIK_ATTR_TOKEN)
        if n_src or n_attr:
            lines_changed += 1
            sourceids_changed += n_src
            attrs_changed += n_attr
        out_lines.append(new_line)
    return "".join(out_lines), lines_changed, sourceids_changed, attrs_changed


def migrate_course_corpus(
    course_dir: Path,
    apply: bool = False,
    *,
    slug: Optional[str] = None,
    backup: bool = True,
    validator: Optional["Validator"] = None,
    repo_root: Optional[Path] = None,
) -> CorpusMigrationReport:
    """Plan (``apply=False``) or execute (``apply=True``) the DART->semantik
    corpus-body migration for ONE course dir.

    Dry-run is side-effect-free: it reads the artifacts, computes every count
    and the post-rewrite ``chunks_sha256``, and writes NOTHING. Apply performs a
    full backup of every mutated artifact (the ``dart_chunks/`` dir + touched
    HTML + top-level ``manifest.json``) to a single timestamped ``.migrate-bak``
    sibling BEFORE writing, then — if ``validator``/``repo_root`` are supplied —
    re-runs the LibV2 validate pass and ROLLS BACK all artifacts on failure
    (never a half-migrated course).

    Manifest-optional: many DART-staged courses carry no top-level
    ``manifest.json`` (the sidecar ``dart_chunks/manifest.json`` is the chunkset
    contract). The migration keys off the sidecar's ``chunkset_kind == "dart"``
    and the presence of ``dart_chunks/chunks.jsonl``; the top-level manifest key
    rename is applied only when that manifest exists.
    """
    slug = slug or course_dir.name
    report = CorpusMigrationReport(slug=slug, course_dir=course_dir)

    dart_dir = course_dir / _DART_CHUNKS_DIRNAME
    semantik_dir = course_dir / _SEMANTIK_CHUNKS_DIRNAME
    chunks_path = dart_dir / _CHUNKS_FILENAME

    # Already-migrated detection: the semantik dir exists and the dart dir is
    # gone -> nothing to do (idempotent no-op).
    if semantik_dir.is_dir() and not dart_dir.is_dir():
        report.already_migrated = True
        return report

    if not chunks_path.is_file():
        # Not a DART-staged course (or the chunkset is absent) — not applicable.
        report.applicable = False
        if dart_dir.is_dir():
            report.nonconforming_reasons.append(
                "dart_chunks/ exists but has no chunks.jsonl"
            )
        return report

    report.applicable = True

    # --- sidecar manifest ---
    sidecar_path = dart_dir / _SIDECAR_FILENAME
    sidecar: Optional[dict] = None
    if sidecar_path.is_file():
        report.sidecar_present = True
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.nonconforming_reasons.append(f"sidecar manifest invalid JSON: {exc}")
            sidecar = None
    else:
        report.nonconforming_reasons.append("dart_chunks/manifest.json missing")

    if sidecar is not None:
        kind = sidecar.get("chunkset_kind")
        if kind != "dart":
            report.nonconforming_reasons.append(
                f"sidecar chunkset_kind={kind!r} (expected 'dart')"
            )

    # --- chunks.jsonl body (read + plan rewrite) ---
    old_bytes = chunks_path.read_bytes()
    report.old_chunks_sha256 = _sha256_bytes(old_bytes)
    old_text = old_bytes.decode("utf-8")
    report.chunks_lines_total = old_text.count("\n") + (
        0 if old_text.endswith("\n") or not old_text else 1
    )
    new_text, lines_changed, src_changed, attr_changed = _rewrite_chunks_text(old_text)
    report.chunks_lines_changed = lines_changed
    report.sourceids_changed = src_changed
    report.chunk_html_attrs_changed = attr_changed
    new_bytes = new_text.encode("utf-8")
    report.new_chunks_sha256 = _sha256_bytes(new_bytes)

    if report.sourceids_changed == 0 and report.chunk_html_attrs_changed == 0:
        report.advisories.append(
            "chunks.jsonl carries no dart: sourceIds or data-dart- attrs "
            "(nothing to rewrite in the chunkset body)"
        )

    # --- archived HTML files ---
    html_files = _iter_html_files(course_dir)
    report.html_files_total = len(html_files)
    html_plan: List[Tuple[Path, str]] = []  # (path, new_text) for files that change
    for hp in html_files:
        try:
            htext = hp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        n = htext.count(_DART_ATTR_TOKEN)
        if n:
            report.html_files_changed += 1
            report.html_attrs_changed += n
            html_plan.append((hp, htext.replace(_DART_ATTR_TOKEN, _SEMANTIK_ATTR_TOKEN)))

    # --- sidecar rewrite plan ---
    new_sidecar: Optional[dict] = None
    if sidecar is not None:
        new_sidecar = dict(sidecar)
        if new_sidecar.get("chunkset_kind") == "dart":
            new_sidecar["chunkset_kind"] = "semantik"
            report.sidecar_kind_flipped = True
        new_sidecar["chunks_sha256"] = report.new_chunks_sha256
        if "source_dart_html_sha256" in new_sidecar:
            new_sidecar, renamed = _rename_manifest_key(
                new_sidecar,
                "source_dart_html_sha256",
                "source_semantik_html_sha256",
                new_sidecar["source_dart_html_sha256"],  # value preserved
            )
            report.sidecar_source_key_renamed = renamed

    # --- top-level manifest rewrite plan ---
    top_manifest_path = course_dir / "manifest.json"
    new_top: Optional[dict] = None
    if top_manifest_path.is_file():
        report.top_manifest_present = True
        try:
            top = json.loads(top_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.nonconforming_reasons.append(
                f"top-level manifest invalid JSON: {exc}"
            )
            top = None
        if top is not None:
            if "dart_chunks_sha256" in top:
                old_top_sha = top.get("dart_chunks_sha256")
                if old_top_sha and old_top_sha != report.old_chunks_sha256:
                    report.advisories.append(
                        "top-level dart_chunks_sha256 was already STALE vs the "
                        "on-disk chunks.jsonl before migration (pre-existing "
                        "inconsistency; migration writes the correct fresh sha)"
                    )
                new_top, renamed = _rename_manifest_key(
                    top,
                    "dart_chunks_sha256",
                    "semantik_chunks_sha256",
                    report.new_chunks_sha256,
                )
                report.top_manifest_key_renamed = renamed

    # dir rename is always part of the plan when applicable.
    report.dir_renamed = True

    if not apply:
        return report

    # ------------------------------------------------------------------- #
    # APPLY: backup everything, write, validate, roll back on failure.    #
    # ------------------------------------------------------------------- #
    backup_dir: Optional[Path] = None
    if backup:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir = course_dir / f".migrate-bak-{ts}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        shutil.copytree(dart_dir, backup_dir / _DART_CHUNKS_DIRNAME)
        if html_plan:
            html_bak = backup_dir / "html"
            for hp, _ in html_plan:
                rel = hp.relative_to(course_dir)
                dest = html_bak / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(hp, dest)
        if new_top is not None:
            shutil.copy2(top_manifest_path, backup_dir / "manifest.json")
        report.backup_dir = backup_dir

    def _rollback() -> None:
        if backup_dir is None:
            return
        # Restore chunkset dir (whether or not it was renamed yet).
        if semantik_dir.exists():
            shutil.rmtree(semantik_dir)
        if dart_dir.exists():
            shutil.rmtree(dart_dir)
        shutil.copytree(backup_dir / _DART_CHUNKS_DIRNAME, dart_dir)
        # Restore HTML.
        html_bak = backup_dir / "html"
        if html_bak.is_dir():
            for hp, _ in html_plan:
                rel = hp.relative_to(course_dir)
                src = html_bak / rel
                if src.is_file():
                    shutil.copy2(src, hp)
        # Restore top-level manifest.
        if new_top is not None and (backup_dir / "manifest.json").is_file():
            shutil.copy2(backup_dir / "manifest.json", top_manifest_path)

    try:
        # 1. chunks.jsonl body
        chunks_path.write_bytes(new_bytes)
        # 2. sidecar
        if new_sidecar is not None:
            sidecar_path.write_text(
                json.dumps(new_sidecar, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        # 3. HTML files
        for hp, htext in html_plan:
            hp.write_text(htext, encoding="utf-8")
        # 4. top-level manifest
        if new_top is not None:
            top_manifest_path.write_text(
                json.dumps(new_top, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        # 5. rename dir dart_chunks -> semantik_chunks (last: bodies are in place)
        if semantik_dir.exists():
            raise MigrationError(
                "SEMANTIK_DIR_EXISTS",
                f"{semantik_dir} already exists; refusing to overwrite",
            )
        dart_dir.rename(semantik_dir)
    except Exception:
        _rollback()
        raise

    # 6. validate + rollback on failure
    if validator is not None and repo_root is not None:
        result = validator(course_dir, repo_root)
        if not bool(getattr(result, "valid", True)):
            report.validation_errors = list(getattr(result, "errors", []))
            _rollback()
            report.rolled_back = True
            report.applied = False
            return report

    report.applied = True
    return report


def _purge_dart_naming_manifest(manifest: dict) -> dict:
    """Manifest transform for the ``1.0 -> 1.1`` dart-purge step.

    Renames the top-level ``dart_chunks_sha256`` key -> ``semantik_chunks_sha256``
    (value preserved here; :func:`migrate_course_corpus` writes the freshly
    recomputed sha over the migrated chunks.jsonl) and stamps
    ``library_format_version = "1.1"``. The corpus BODY rewrite is carried by the
    step's ``corpus_transform`` (:func:`migrate_course_corpus`), not here."""
    migrated, _ = _rename_manifest_key(
        dict(manifest),
        "dart_chunks_sha256",
        "semantik_chunks_sha256",
        manifest.get("dart_chunks_sha256"),
    )
    migrated["library_format_version"] = "1.1"
    return migrated


def build_dart_purge_registry() -> MigrationRegistry:
    """Registry whose terminus is ``1.1`` — the DART->semantik naming purge.

    Kept SEPARATE from :data:`DEFAULT_REGISTRY` on purpose: the DEFAULT
    terminus is drift-guarded against the emitter's still-``1.0``
    ``LIBRARY_FORMAT_VERSION``, so the DEFAULT bump lands with the on-disk
    dir-rename migration in a later gated stage. The chunkset SIDECAR emit has
    already flipped (task #19): new conversions write ``semantik_chunks/`` with
    ``chunkset_kind="semantik"`` + ``source_semantik_html_sha256``, so this
    registry's corpus_transform only upgrades pre-flip ``dart_chunks/`` archives.
    This registry drives the corpus-migration tooling + dry-run in the meantime."""
    registry = MigrationRegistry()
    registry.register(
        MigrationStep(
            from_version=LEGACY_VERSION,
            to_version="1.0",
            description=(
                "Stamp library_format_version=1.0 on a pre-OP4 manifest "
                "(pre-1.0 baseline layout; no directory-layout change)."
            ),
            transform=_stamp_v1_0,
        )
    )
    registry.register(
        MigrationStep(
            from_version="1.0",
            to_version="1.1",
            description=(
                "DART->semantik naming purge: rewrite dart: sourceIds -> "
                "semantik:, data-dart-* -> data-semantik-* attrs, rename "
                "dart_chunks/ -> semantik_chunks/, recompute chunkset sha under "
                "semantik-named keys, flip chunkset_kind dart -> semantik."
            ),
            transform=_purge_dart_naming_manifest,
            corpus_transform=migrate_course_corpus,
        )
    )
    return registry


DART_PURGE_REGISTRY = build_dart_purge_registry()


def plan_dart_purge(repo_root: Path) -> List[CorpusMigrationReport]:
    """Dry-run the DART->semantik corpus migration across every course.

    Side-effect-free: runs :func:`migrate_course_corpus` with ``apply=False``
    over each discovered course and returns the per-course reports (sorted by
    slug via :func:`discover_courses`)."""
    reports: List[CorpusMigrationReport] = []
    for course_dir in discover_courses(repo_root):
        reports.append(migrate_course_corpus(course_dir, apply=False, slug=course_dir.name))
    return reports
