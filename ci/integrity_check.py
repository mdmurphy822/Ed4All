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

DEFAULT_RUNS_PATH = PROJECT_ROOT / "runs"
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
