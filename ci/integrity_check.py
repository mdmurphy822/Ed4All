#!/usr/bin/env python3
"""
CI Integrity Check Hook

Validates system integrity as part of CI/CD pipeline.
Runs comprehensive checks on schemas, hash chains, tool registry,
and sample finalization.

Phase 0.5 Enhancement: CI Integrity Test Hook (E3)

Usage:
    python ci/integrity_check.py [--verbose] [--fix] [--runs-path PATH]

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
    2 - Configuration error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_RUNS_PATH = PROJECT_ROOT / "state" / "runs"
DEFAULT_SCHEMAS_PATH = PROJECT_ROOT / "schemas"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CheckResult:
    """Result of a single integrity check."""
    name: str
    passed: bool
    message: str
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fixed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrityReport:
    """Complete integrity check report."""
    timestamp: str
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    total_duration_seconds: float
    checks: List[CheckResult] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "total_duration_seconds": self.total_duration_seconds,
            "checks": [c.to_dict() for c in self.checks],
            "environment": self.environment,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Integrity Check {status}: "
            f"{self.passed_checks}/{self.total_checks} checks passed "
            f"({self.total_duration_seconds:.2f}s)"
        )


# ============================================================================
# CHECK FUNCTIONS
# ============================================================================

def check_schemas(schemas_path: Path, verbose: bool = False) -> CheckResult:
    """
    Validate all JSON schemas are valid.

    Args:
        schemas_path: Path to schemas directory
        verbose: Enable verbose output

    Returns:
        CheckResult
    """
    start_time = time.time()
    result = CheckResult(
        name="schema_validation",
        passed=False,
        message="",
    )

    if not schemas_path.exists():
        result.message = f"Schemas directory not found: {schemas_path}"
        result.errors.append(result.message)
        result.duration_seconds = time.time() - start_time
        return result

    schema_files = list(schemas_path.glob("**/*.json"))
    result.details["schema_count"] = len(schema_files)

    if len(schema_files) == 0:
        result.message = "No schema files found"
        result.warnings.append(result.message)
        result.passed = True  # Not a failure, just nothing to validate
        result.duration_seconds = time.time() - start_time
        return result

    try:
        import jsonschema
    except ImportError:
        result.message = "jsonschema library not installed - skipping validation"
        result.warnings.append(result.message)
        result.passed = True
        result.duration_seconds = time.time() - start_time
        return result

    valid_count = 0
    for schema_file in schema_files:
        try:
            with open(schema_file) as f:
                schema = json.load(f)

            # Validate it's a valid JSON Schema
            jsonschema.Draft7Validator.check_schema(schema)
            valid_count += 1

            if verbose:
                logger.info(f"  Valid: {schema_file.name}")

        except json.JSONDecodeError as e:
            result.errors.append(f"{schema_file.name}: Invalid JSON - {e}")
        except jsonschema.SchemaError as e:
            result.errors.append(f"{schema_file.name}: Invalid schema - {e.message}")
        except Exception as e:
            result.errors.append(f"{schema_file.name}: Error - {e}")

    result.details["valid_count"] = valid_count
    result.passed = len(result.errors) == 0
    result.message = f"Validated {valid_count}/{len(schema_files)} schemas"
    result.duration_seconds = time.time() - start_time
    return result


def check_hash_chains(runs_path: Path, verbose: bool = False) -> CheckResult:
    """
    Verify integrity of all hash chains.

    Args:
        runs_path: Path to runs directory
        verbose: Enable verbose output

    Returns:
        CheckResult
    """
    start_time = time.time()
    result = CheckResult(
        name="hash_chain_integrity",
        passed=False,
        message="",
    )

    if not runs_path.exists():
        result.message = "Runs directory not found"
        result.warnings.append(result.message)
        result.passed = True  # Not a failure if no runs exist
        result.duration_seconds = time.time() - start_time
        return result

    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    result.details["run_count"] = len(run_dirs)

    if len(run_dirs) == 0:
        result.message = "No runs found"
        result.passed = True
        result.duration_seconds = time.time() - start_time
        return result

    try:
        from lib.replay_engine import ReplayEngine
        engine = ReplayEngine(runs_path=runs_path)
    except ImportError:
        # Fall back to direct verification
        engine = None

    verified_count = 0
    for run_dir in run_dirs:
        run_id = run_dir.name
        chain_path = run_dir / "hash_chain.jsonl"

        if not chain_path.exists():
            if verbose:
                logger.info(f"  {run_id}: No hash chain (skipped)")
            continue

        try:
            if engine:
                integrity = engine.verify_run_integrity(run_id)
                valid = integrity.get("checks", {}).get("hash_chain_valid", False)
            else:
                valid = _verify_hash_chain_direct(chain_path)

            if valid:
                verified_count += 1
                if verbose:
                    logger.info(f"  {run_id}: Valid")
            else:
                result.errors.append(f"{run_id}: Hash chain integrity failed")
                if verbose:
                    logger.warning(f"  {run_id}: INVALID")

        except Exception as e:
            result.errors.append(f"{run_id}: Error - {e}")

    result.details["verified_count"] = verified_count
    result.passed = len(result.errors) == 0
    result.message = f"Verified {verified_count}/{len(run_dirs)} hash chains"
    result.duration_seconds = time.time() - start_time
    return result


def _verify_hash_chain_direct(chain_path: Path) -> bool:
    """Direct hash chain verification without ReplayEngine."""
    import hashlib

    prev_hash = None
    with open(chain_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                stored_prev = entry.get("prev_hash")

                if prev_hash is None:
                    if stored_prev and stored_prev != "":
                        return False
                else:
                    if stored_prev != prev_hash:
                        return False

                entry_copy = entry.copy()
                entry_copy.pop("entry_hash", None)
                content = json.dumps(entry_copy, sort_keys=True)
                prev_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            except (json.JSONDecodeError, KeyError):
                return False

    return True


def check_tool_registry(verbose: bool = False) -> CheckResult:
    """
    Verify tool registry configuration.

    Args:
        verbose: Enable verbose output

    Returns:
        CheckResult
    """
    start_time = time.time()
    result = CheckResult(
        name="tool_registry",
        passed=False,
        message="",
    )

    try:
        from lib.tool_registry import ToolRegistry, get_registry

        registry = get_registry()

        # Check required tools
        validation = registry.validate_required_tools()
        result.details["required_tools_valid"] = validation.valid
        result.details["registered_tools"] = registry.list_tools()
        result.details["required_tools"] = registry.required_tools

        if not validation.valid:
            result.errors.extend(validation.errors)
        result.warnings.extend(validation.warnings)

        # Create and verify snapshot
        snapshot = registry.snapshot()
        result.details["snapshot_hash"] = snapshot.get("snapshot_hash", "")[:12]
        result.details["tool_count"] = snapshot.get("tool_count", 0)

        result.passed = validation.valid
        result.message = f"Registry valid: {len(registry.list_tools())} tools registered"

        if verbose:
            for tool in registry.list_tools():
                logger.info(f"  Registered: {tool}")

    except ImportError as e:
        result.message = f"Tool registry not available: {e}"
        result.warnings.append(result.message)
        result.passed = True  # Not a failure if module not available

    except Exception as e:
        result.errors.append(f"Registry error: {e}")
        result.message = f"Registry check failed: {e}"

    result.duration_seconds = time.time() - start_time
    return result


def check_config_files(config_path: Path, verbose: bool = False) -> CheckResult:
    """
    Validate configuration files.

    Args:
        config_path: Path to config directory
        verbose: Enable verbose output

    Returns:
        CheckResult
    """
    start_time = time.time()
    result = CheckResult(
        name="config_validation",
        passed=False,
        message="",
    )

    if not config_path.exists():
        result.message = f"Config directory not found: {config_path}"
        result.errors.append(result.message)
        result.duration_seconds = time.time() - start_time
        return result

    # Check workflows.yaml
    workflows_path = config_path / "workflows.yaml"
    if workflows_path.exists():
        try:
            import yaml
            with open(workflows_path) as f:
                config = yaml.safe_load(f)

            result.details["workflows_valid"] = True
            result.details["has_hardening_section"] = "hardening" in config

            if "hardening" in config:
                hardening = config["hardening"]
                result.details["hardening_keys"] = list(hardening.keys())

            if verbose:
                logger.info(f"  workflows.yaml: Valid")
                if "hardening" in config:
                    logger.info(f"  Hardening section present")

        except Exception as e:
            result.errors.append(f"workflows.yaml: {e}")
            result.details["workflows_valid"] = False
    else:
        result.warnings.append("workflows.yaml not found")

    # Check agents.yaml
    agents_path = config_path / "agents.yaml"
    if agents_path.exists():
        try:
            import yaml
            with open(agents_path) as f:
                config = yaml.safe_load(f)

            result.details["agents_valid"] = True
            result.details["agent_count"] = len(config.get("agents", {}))

            if verbose:
                logger.info(f"  agents.yaml: Valid ({result.details['agent_count']} agents)")

        except Exception as e:
            result.errors.append(f"agents.yaml: {e}")
            result.details["agents_valid"] = False
    else:
        result.warnings.append("agents.yaml not found")

    result.passed = len(result.errors) == 0
    result.message = "Config files validated"
    result.duration_seconds = time.time() - start_time
    return result


def check_sample_finalization(runs_path: Path, verbose: bool = False) -> CheckResult:
    """
    Test finalization on a sample run (if available).

    Args:
        runs_path: Path to runs directory
        verbose: Enable verbose output

    Returns:
        CheckResult
    """
    start_time = time.time()
    result = CheckResult(
        name="sample_finalization",
        passed=False,
        message="",
    )

    if not runs_path.exists():
        result.message = "No runs directory for finalization test"
        result.warnings.append(result.message)
        result.passed = True
        result.duration_seconds = time.time() - start_time
        return result

    # Find a run to test
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    if not run_dirs:
        result.message = "No runs available for finalization test"
        result.passed = True
        result.duration_seconds = time.time() - start_time
        return result

    # Pick the most recent run
    test_run = sorted(run_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]
    result.details["test_run"] = test_run.name

    try:
        from lib.run_finalizer import RunFinalizer

        finalizer = RunFinalizer(test_run)
        report = finalizer.verify_only()

        result.details["finalization_valid"] = report.success
        result.details["hash_chain_valid"] = report.hash_chain_valid
        result.details["artifact_count"] = report.artifact_count

        if not report.success:
            result.errors.extend(report.errors)

        result.passed = report.success
        result.message = f"Finalization test on {test_run.name}: {'PASSED' if report.success else 'FAILED'}"

        if verbose:
            logger.info(f"  Test run: {test_run.name}")
            logger.info(f"  Hash chain valid: {report.hash_chain_valid}")
            logger.info(f"  Artifacts: {report.artifact_count}")

    except ImportError as e:
        result.message = f"Run finalizer not available: {e}"
        result.warnings.append(result.message)
        result.passed = True

    except Exception as e:
        result.errors.append(f"Finalization error: {e}")
        result.message = f"Finalization test failed: {e}"

    result.duration_seconds = time.time() - start_time
    return result


def check_path_security(verbose: bool = False) -> CheckResult:
    """
    Verify path security constants are properly configured.

    Args:
        verbose: Enable verbose output

    Returns:
        CheckResult
    """
    start_time = time.time()
    result = CheckResult(
        name="path_security",
        passed=False,
        message="",
    )

    try:
        from lib.path_constants import (
            MAX_PATH_LENGTH,
            DISALLOW_PARENT_TRAVERSAL,
            get_project_root,
            load_hardening_config,
        )

        result.details["max_path_length"] = MAX_PATH_LENGTH
        result.details["disallow_parent_traversal"] = DISALLOW_PARENT_TRAVERSAL

        # Check project root resolution
        try:
            project_root = get_project_root()
            result.details["project_root"] = str(project_root)
            result.details["project_root_exists"] = project_root.exists()
        except Exception as e:
            result.warnings.append(f"Project root resolution: {e}")

        # Check hardening config
        try:
            config = load_hardening_config()
            result.details["hardening_config_loaded"] = True
            result.details["config_keys"] = list(config.keys()) if config else []
        except Exception as e:
            result.warnings.append(f"Hardening config: {e}")

        result.passed = True
        result.message = "Path security constants verified"

        if verbose:
            logger.info(f"  MAX_PATH_LENGTH: {MAX_PATH_LENGTH}")
            logger.info(f"  DISALLOW_PARENT_TRAVERSAL: {DISALLOW_PARENT_TRAVERSAL}")

    except ImportError as e:
        result.message = f"Path constants not available: {e}"
        result.warnings.append(result.message)
        result.passed = True

    except Exception as e:
        result.errors.append(f"Path security error: {e}")
        result.message = f"Path security check failed: {e}"

    result.duration_seconds = time.time() - start_time
    return result


def check_write_facade(verbose: bool = False) -> CheckResult:
    """
    Verify WriteFacade is properly configured.

    Args:
        verbose: Enable verbose output

    Returns:
        CheckResult
    """
    start_time = time.time()
    result = CheckResult(
        name="write_facade",
        passed=False,
        message="",
    )

    try:
        from lib.write_facade import WriteFacade, WriteResult
        from lib.path_constants import is_write_facade_enforced

        result.details["write_facade_available"] = True
        result.details["enforcement_enabled"] = is_write_facade_enforced()

        # Test basic functionality
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            facade = WriteFacade(allowed_paths=[tmppath])

            # Test write
            test_file = tmppath / "test.txt"
            write_result = facade.write(test_file, "test content")

            result.details["test_write_success"] = write_result.success
            if not write_result.success:
                result.errors.append(f"Test write failed: {write_result.error}")

            # Test transaction
            facade.begin_transaction()
            facade.write(tmppath / "tx_test.txt", "transaction test")
            tx_result = facade.commit_transaction()

            result.details["transaction_success"] = tx_result.success

        result.passed = len(result.errors) == 0
        result.message = "WriteFacade functional"

        if verbose:
            logger.info(f"  Enforcement enabled: {result.details['enforcement_enabled']}")
            logger.info(f"  Test write: {'PASS' if result.details['test_write_success'] else 'FAIL'}")

    except ImportError as e:
        result.message = f"WriteFacade not available: {e}"
        result.warnings.append(result.message)
        result.passed = True

    except Exception as e:
        result.errors.append(f"WriteFacade error: {e}")
        result.message = f"WriteFacade check failed: {e}"

    result.duration_seconds = time.time() - start_time
    return result


def check_validator_test_coverage(
    validators_path: Path,
    allowlist_path: Optional[Path] = None,
    verbose: bool = False,
) -> CheckResult:
    """Detect validator modules without a corresponding test file.

    Wave W-D10 T10.3 — closes the gate-coverage feedback gap.

    Walks every ``*.py`` under ``validators_path`` (recursively, post-W-D10
    subpackages) and confirms a matching test file exists at one of:

    - ``lib/validators/tests/test_{stem}.py``
    - ``lib/validators/tests/test_{stem}_validator.py``
    - ``lib/validators/tests/test_{family}_{stem}.py`` (subpackage case)
    - ``lib/validators/tests/test_{stem}_*.py`` (prefix-match)
    - same in ``lib/tests/``, ``schemas/tests/``, ``Trainforge/tests/``,
      ``MCP/tests/``
    - if the validator is in a subpackage and a flat-name shim still
      exists, tests for the shim's stem also count as coverage.

    Skips:
    - ``__init__.py``, ``__pycache__``, ``tests/`` subdirs, and shim
      modules (modules whose docstring begins with ``DEPRECATED`` and
      that re-export from a sibling subpackage path).
    - Modules whose stem appears in ``allowlist_path`` (one stem per
      line, ``#``-comments allowed).

    Returns ``CheckResult`` with ``passed=False`` and an explicit error
    per missing-test validator when any non-allow-listed validator
    lacks a test file. ``passed=True`` otherwise.
    """
    import re

    start_time = time.time()
    result = CheckResult(
        name="validator_test_coverage",
        passed=False,
        message="",
    )

    if not validators_path.exists():
        result.message = f"Validators directory not found: {validators_path}"
        result.warnings.append(result.message)
        result.passed = True
        result.duration_seconds = time.time() - start_time
        return result

    # Canonical test dirs.
    test_dirs = [
        validators_path / "tests",
        PROJECT_ROOT / "lib" / "tests",
        PROJECT_ROOT / "schemas" / "tests",
        PROJECT_ROOT / "Trainforge" / "tests",
        PROJECT_ROOT / "MCP" / "tests",
    ]

    # Allow-list.
    allowlisted_stems = set()
    if allowlist_path is None:
        allowlist_path = PROJECT_ROOT / "ci" / "validator_test_allowlist.txt"
    if allowlist_path.exists():
        for raw in allowlist_path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                allowlisted_stems.add(line)
    result.details["allowlist_count"] = len(allowlisted_stems)

    # Walk validator modules. Skip __init__.py, tests subdirs,
    # __pycache__, and private modules (anything starting with "_" —
    # by convention private helper packages or modules extracted from
    # a public validator are not separately tested; the parent
    # validator's test exercises them transitively).
    all_validators = []
    for py in validators_path.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        if "tests" in py.parts:
            continue
        if "__pycache__" in py.parts:
            continue
        rel_parts = py.relative_to(validators_path).parts
        if any(part.startswith("_") for part in rel_parts):
            continue
        all_validators.append(py)

    # Detect shim files: modules whose docstring opens with "DEPRECATED"
    # AND that re-export from a sibling subpackage. Captures both the
    # `DEPRECATED:` colon form and the ``Wave NNN — DEPRECATED.`` form.
    def _is_shim(p: Path) -> bool:
        try:
            head = p.read_text()[:600]
        except OSError:
            return False
        if "DEPRECATED" not in head:
            return False
        body = p.read_text()
        return (
            "re-export" in head
            or "shim" in head.lower()
            or "warnings.warn" in body[:1500]
        )

    shims = [p for p in all_validators if _is_shim(p)]
    real_validators = [p for p in all_validators if p not in shims]

    # Build (family, modname) -> shim_stem map for cross-coverage:
    # tests for the legacy flat-shim name cover the new subpackage path.
    shim_for_subpackage: Dict[Tuple[str, str], str] = {}
    for s in shims:
        text = s.read_text()
        m = re.search(r"from lib\.validators\.([\w_]+)\.([\w_]+) import", text)
        if m:
            shim_for_subpackage[(m.group(1), m.group(2))] = s.stem

    # Aggregate test stems across canonical dirs. Recurse so tests placed
    # in a subpackage-mirroring subdir (e.g. lib/validators/tests/alignment/)
    # count as coverage for the corresponding validator subpackage.
    all_test_stems: set = set()
    for d in test_dirs:
        if not d.exists():
            continue
        for t in d.rglob("test_*.py"):
            if "__pycache__" in t.parts:
                continue
            all_test_stems.add(t.stem)

    def _has_test(validator: Path) -> bool:
        rel = validator.relative_to(validators_path)
        parts = list(rel.parts)
        parts[-1] = parts[-1].replace(".py", "")
        stem = parts[-1]
        family = parts[0] if len(parts) > 1 else None

        candidates = {
            f"test_{stem}",
            f"test_{stem}_validator",
        }
        if family:
            candidates.add(f"test_{family}_{stem}")
            if (family, stem) in shim_for_subpackage:
                shim_flat = shim_for_subpackage[(family, stem)]
                candidates.add(f"test_{shim_flat}")
                candidates.add(f"test_{shim_flat}_validator")

        if candidates & all_test_stems:
            return True

        # Prefix match: any test stem starting with test_<stem>_<...>
        # or test_<shim_flat>_<...>.
        prefixes = [f"test_{stem}_"]
        if family and (family, stem) in shim_for_subpackage:
            prefixes.append(f"test_{shim_for_subpackage[(family, stem)]}_")
        for tname in all_test_stems:
            if any(tname.startswith(p) for p in prefixes):
                return True
        return False

    missing = []
    for v in real_validators:
        rel = v.relative_to(validators_path)
        parts = list(rel.parts)
        parts[-1] = parts[-1].replace(".py", "")
        stem = parts[-1]
        if stem in allowlisted_stems:
            continue
        if not _has_test(v):
            missing.append(v)

    result.details["validator_count"] = len(real_validators)
    result.details["shim_count"] = len(shims)
    result.details["missing_count"] = len(missing)

    if missing:
        for v in missing:
            result.errors.append(
                f"validator without test: {v.relative_to(PROJECT_ROOT)} "
                f"(no matching test_*.py under lib/validators/tests/, "
                f"lib/tests/, schemas/tests/, Trainforge/tests/, MCP/tests/; "
                f"add a test or list the stem in "
                f"ci/validator_test_allowlist.txt)"
            )
        result.passed = False
        result.message = (
            f"{len(missing)} validator(s) without tests "
            f"({len(real_validators)} real validators, "
            f"{len(shims)} shims skipped, "
            f"{len(allowlisted_stems)} allow-listed)"
        )
    else:
        result.passed = True
        result.message = (
            f"All {len(real_validators)} real validators have tests "
            f"({len(shims)} shims skipped, "
            f"{len(allowlisted_stems)} allow-listed)"
        )

    if verbose:
        logger.info(f"  Real validators: {len(real_validators)}")
        logger.info(f"  Shims (skipped): {len(shims)}")
        logger.info(f"  Allow-listed:    {len(allowlisted_stems)}")
        logger.info(f"  Missing tests:   {len(missing)}")
        for v in missing:
            logger.warning(f"    - {v.relative_to(PROJECT_ROOT)}")

    result.duration_seconds = time.time() - start_time
    return result


def check_libv2_vendor_sync(verbose: bool = False) -> CheckResult:
    """Verify LibV2/vendor/bloom_verbs.json matches the authoritative copy.

    LibV2 is sandboxed from importing Ed4All's lib/ package (cross-package
    caveat documented in LibV2/CLAUDE.md). Instead of reaching across the
    package boundary, LibV2 reads a byte-identical vendored copy of
    schemas/taxonomies/bloom_verbs.json at LibV2/vendor/bloom_verbs.json.

    This check ensures the vendored copy has not drifted from the source.
    """
    import hashlib

    start_time = time.time()
    result = CheckResult(name="LibV2 Vendor Sync", passed=True, message="")

    auth_path = PROJECT_ROOT / "schemas" / "taxonomies" / "bloom_verbs.json"
    vendored_path = PROJECT_ROOT / "LibV2" / "vendor" / "bloom_verbs.json"

    if not auth_path.exists():
        result.errors.append(f"Authoritative schema missing: {auth_path}")
        result.passed = False
        result.message = "Authoritative bloom_verbs.json missing"
        result.duration_seconds = time.time() - start_time
        return result

    if not vendored_path.exists():
        result.errors.append(f"Vendored copy missing: {vendored_path}")
        result.passed = False
        result.message = "LibV2 vendored bloom_verbs.json missing"
        result.duration_seconds = time.time() - start_time
        return result

    auth_hash = hashlib.sha256(auth_path.read_bytes()).hexdigest()
    vendored_hash = hashlib.sha256(vendored_path.read_bytes()).hexdigest()

    result.details["auth_sha256"] = auth_hash
    result.details["vendored_sha256"] = vendored_hash

    if auth_hash != vendored_hash:
        result.errors.append(
            f"Hash drift between {auth_path.name} and {vendored_path}: "
            f"auth={auth_hash[:16]}... vendored={vendored_hash[:16]}..."
        )
        result.passed = False
        result.message = "LibV2 vendored bloom_verbs.json has drifted"
    else:
        result.message = f"LibV2 vendored copy in sync (sha256={auth_hash[:16]}...)"

    if verbose:
        logger.info(f"  auth sha256:     {auth_hash}")
        logger.info(f"  vendored sha256: {vendored_hash}")

    result.duration_seconds = time.time() - start_time
    return result


def check_course_slug_leak(verbose: bool = False) -> CheckResult:
    """Fail when a built-course slug is hardcoded in a tracked file.

    Courses this pipeline produces are working data, not product; their
    slugs must not appear in tracked code/tests/config/docs. Detection and
    the allowlist live in ``ci/course_slug_guard.py`` (scheme regexes, not a
    literal blocklist, so future members of a family are caught). Scans
    ``git ls-files`` only. An unscannable tree (no git) is a warning + pass,
    not a silent skip; any violation fails the check with file:line:token.
    """
    start_time = time.time()
    result = CheckResult(name="course_slug_leak", passed=False, message="")

    try:
        from ci import course_slug_guard
    except ImportError as e:
        result.errors.append(f"course_slug_guard import failed: {e}")
        result.message = "Course-slug guard unavailable"
        result.duration_seconds = time.time() - start_time
        return result

    try:
        violations = course_slug_guard.scan_repository(PROJECT_ROOT)
    except RuntimeError as e:
        result.message = f"Course-slug scan skipped: {e}"
        result.warnings.append(result.message)
        result.passed = True
        result.duration_seconds = time.time() - start_time
        return result

    result.details["violation_count"] = len(violations)
    if violations:
        for v in violations:
            result.errors.append(v.format())
        result.passed = False
        result.message = f"{len(violations)} built-course slug(s) in tracked files"
    else:
        result.passed = True
        result.message = "No built-course slugs in tracked files"

    if verbose:
        logger.info(f"  {result.message}")
        for v in violations:
            logger.warning(f"    {v.format()}")

    result.duration_seconds = time.time() - start_time
    return result


def check_legacy_token_leak(verbose: bool = False) -> CheckResult:
    """Fail when the retired legacy 'dart' engine name is in a tracked file.

    ``DART`` was the AGPL PDF-conversion engine that ``SemantiK`` replaced;
    the public repo must carry no reference to it except in the narrow set
    of dual-read compat / migration / legacy-compat-test files enumerated
    in ``ci/legacy_token_allowlist.txt``. Detection (boundary-aware, so
    ``standard`` never fires) and the allowlist live in
    ``ci/legacy_token_guard.py``. Scans ``git ls-files`` only. An
    unscannable tree (no git) is a warning + pass, not a silent skip; any
    violation fails the check with file:line:token.
    """
    start_time = time.time()
    result = CheckResult(name="legacy_token_leak", passed=False, message="")

    try:
        from ci import legacy_token_guard
    except ImportError as e:
        result.errors.append(f"legacy_token_guard import failed: {e}")
        result.message = "Legacy-token guard unavailable"
        result.duration_seconds = time.time() - start_time
        return result

    try:
        violations = legacy_token_guard.scan_repository(PROJECT_ROOT)
    except RuntimeError as e:
        result.message = f"Legacy-token scan skipped: {e}"
        result.warnings.append(result.message)
        result.passed = True
        result.duration_seconds = time.time() - start_time
        return result

    result.details["violation_count"] = len(violations)
    if violations:
        for v in violations:
            result.errors.append(v.format())
        result.passed = False
        result.message = f"{len(violations)} legacy 'dart' token(s) in tracked files"
    else:
        result.passed = True
        result.message = "No legacy 'dart' tokens in tracked files"

    if verbose:
        logger.info(f"  {result.message}")
        for v in violations:
            logger.warning(f"    {v.format()}")

    result.duration_seconds = time.time() - start_time
    return result


def check_provenance_enum_sync(verbose: bool = False) -> CheckResult:
    """Verify the closed Touch.provider enum is in sync across all sites.

    The provenance set is derived from the unified endpoint registry
    (``config/endpoints.yaml``). Runs
    ``scripts/codegen/sync_provenance_enum.py --check``, which exits
    non-zero when the JSON-LD schema enum or the SHACL ``sh:in`` list has
    drifted from the derived set. Closes the governance hole where the
    three provenance sites stayed in sync only by hand.
    """
    import subprocess

    start_time = time.time()
    result = CheckResult(
        name="provenance_enum_sync",
        passed=False,
        message="",
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.codegen.sync_provenance_enum",
                "--check",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        result.details["returncode"] = proc.returncode
        if proc.returncode == 0:
            result.passed = True
            result.message = "Touch.provider enum in sync across all sites"
        else:
            result.passed = False
            result.message = "Touch.provider enum drift detected"
            for line in (proc.stderr or proc.stdout or "").splitlines():
                if line.strip():
                    result.errors.append(line.strip())
        if verbose:
            logger.info(f"  {result.message}")
    except Exception as e:
        result.errors.append(f"Provenance enum sync error: {e}")
        result.message = f"Provenance enum sync check failed: {e}"
        result.passed = False

    result.duration_seconds = time.time() - start_time
    return result


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_integrity_checks(
    runs_path: Path = DEFAULT_RUNS_PATH,
    schemas_path: Path = DEFAULT_SCHEMAS_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    verbose: bool = False,
    fix: bool = False,
) -> IntegrityReport:
    """
    Run all integrity checks.

    Args:
        runs_path: Path to runs directory
        schemas_path: Path to schemas directory
        config_path: Path to config directory
        verbose: Enable verbose output
        fix: Attempt to fix issues (not implemented)

    Returns:
        IntegrityReport with all results
    """
    start_time = time.time()

    report = IntegrityReport(
        timestamp=datetime.now().isoformat(),
        passed=False,
        total_checks=0,
        passed_checks=0,
        failed_checks=0,
        total_duration_seconds=0.0,
        environment={
            "python_version": sys.version,
            "runs_path": str(runs_path),
            "schemas_path": str(schemas_path),
            "config_path": str(config_path),
        },
    )

    # Run all checks
    checks = [
        ("Schema Validation", lambda: check_schemas(schemas_path, verbose)),
        ("Config Files", lambda: check_config_files(config_path, verbose)),
        ("Path Security", lambda: check_path_security(verbose)),
        ("Write Facade", lambda: check_write_facade(verbose)),
        ("Tool Registry", lambda: check_tool_registry(verbose)),
        ("Hash Chains", lambda: check_hash_chains(runs_path, verbose)),
        ("Sample Finalization", lambda: check_sample_finalization(runs_path, verbose)),
        ("LibV2 Vendor Sync", lambda: check_libv2_vendor_sync(verbose)),
        ("Course Slug Leak", lambda: check_course_slug_leak(verbose)),
        ("Legacy Token Leak", lambda: check_legacy_token_leak(verbose)),
        ("Provenance Enum Sync", lambda: check_provenance_enum_sync(verbose)),
        (
            "Validator Test Coverage",
            lambda: check_validator_test_coverage(
                PROJECT_ROOT / "lib" / "validators",
                verbose=verbose,
            ),
        ),
    ]

    for name, check_func in checks:
        if verbose:
            logger.info(f"Running: {name}")

        try:
            result = check_func()
            report.checks.append(result)
            report.total_checks += 1

            if result.passed:
                report.passed_checks += 1
                if verbose:
                    logger.info(f"  Result: PASSED - {result.message}")
            else:
                report.failed_checks += 1
                if verbose:
                    logger.warning(f"  Result: FAILED - {result.message}")
                    for error in result.errors:
                        logger.error(f"    Error: {error}")

        except Exception as e:
            logger.error(f"Check '{name}' crashed: {e}")
            report.checks.append(CheckResult(
                name=name.lower().replace(" ", "_"),
                passed=False,
                message=f"Check crashed: {e}",
                errors=[str(e)],
            ))
            report.total_checks += 1
            report.failed_checks += 1

    report.total_duration_seconds = time.time() - start_time
    report.passed = report.failed_checks == 0

    return report


def main() -> int:
    """
    Main entry point for CI integrity checks.

    Returns:
        Exit code (0 = success, 1 = failure, 2 = config error)
    """
    parser = argparse.ArgumentParser(
        description="Run Ed4All integrity checks for CI/CD"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues (not implemented)",
    )
    parser.add_argument(
        "--runs-path",
        type=Path,
        default=DEFAULT_RUNS_PATH,
        help="Path to runs directory",
    )
    parser.add_argument(
        "--schemas-path",
        type=Path,
        default=DEFAULT_SCHEMAS_PATH,
        help="Path to schemas directory",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON report to file",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("Ed4All Integrity Check")
    print("=" * 60)
    print()

    try:
        report = run_integrity_checks(
            runs_path=args.runs_path,
            schemas_path=args.schemas_path,
            config_path=args.config_path,
            verbose=args.verbose,
            fix=args.fix,
        )

        # Print summary
        print()
        print("=" * 60)
        print(f"Results: {report.summary()}")
        print("=" * 60)

        if not report.passed:
            print("\nFailed checks:")
            for check in report.checks:
                if not check.passed:
                    print(f"  - {check.name}: {check.message}")
                    for error in check.errors:
                        print(f"      Error: {error}")

        # Write JSON report if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nReport written to: {args.output}")

        return 0 if report.passed else 1

    except Exception as e:
        logger.error(f"Integrity check failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
