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

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

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


@dataclass(frozen=True)
class MigrationStep:
    """A single ordered layout-version migration.

    ``transform`` receives a deep-ish copy of the manifest and returns the
    migrated manifest. It MUST stamp ``library_format_version = to_version`` (or
    the engine will refuse the result as a no-op that never advanced).
    """

    from_version: str
    to_version: str
    description: str
    transform: ManifestTransform


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
