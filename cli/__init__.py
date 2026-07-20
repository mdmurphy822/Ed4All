"""
Ed4All CLI - Integrity checking and run management tools.

Phase 0 Hardening - Requirement 9: CLI Integrity Checks
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# VERSIONING.md names ``pyproject.toml::project.version`` the single source of
# truth for the release version. Reading it back from the installed
# distribution keeps that promise -- a hardcoded literal here is a second copy
# that silently drifts the moment only one of the two is bumped (it had already
# drifted from the branch line before this was derived).
#
# Falls back to "0.0.0+unknown" when the package is not installed (running
# straight from a source checkout). The sentinel is deliberately not a plausible
# release number: an unset version must be obviously unset, never mistaken for
# a real one in a provenance stamp or a bug report.
try:
    __version__ = _pkg_version("ed4all")
except PackageNotFoundError:  # pragma: no cover - source-checkout fallback
    __version__ = "0.0.0+unknown"
