"""Wave 137c — operator-only gold-set hash recompute.

The 6 pre-Wave-135a complete FORM_DATA entries are the calibration
reference for Plan A's diversity / style / anchor-verb thresholds.
``Trainforge/tests/test_form_data_gold_set.py`` snapshots the SHA-256
of ``rdf_shacl_gold_set.yaml`` to catch accidental drift; this script
exists ONLY to recompute that hash when the gold set changes
intentionally.

**Operator contract**: any commit that updates the hash via this
script MUST also recalibrate Plan A's diversity / style / anchor-verb
thresholds in the same commit, and link the recalibration commit in
the recompute commit's message. Drift without recalibration silently
invalidates the threshold floors.

Usage::

    # Diff-only mode (default, exits 1 without --yes):
    python -m Trainforge.scripts.recompute_gold_set_hash

    # Commit the new hash:
    python -m Trainforge.scripts.recompute_gold_set_hash --yes
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional


_DEFAULT_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "form_data_gold_set"
)
_YAML_NAME = "rdf_shacl_gold_set.yaml"
_HASH_NAME = "rdf_shacl_gold_set_hash.txt"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recompute_gold_set_hash",
        description=(
            "Recompute the SHA-256 of the FORM_DATA gold-set YAML and "
            "(with --yes) overwrite the locked hash file. Operator-only "
            "utility — any committed hash update MUST land alongside a "
            "Plan A threshold recalibration."
        ),
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=_DEFAULT_FIXTURE_DIR,
        help=(
            "Directory containing rdf_shacl_gold_set.yaml + "
            "rdf_shacl_gold_set_hash.txt. Default: "
            f"{_DEFAULT_FIXTURE_DIR}"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Required to overwrite the hash file. Without --yes, the "
            "script prints the diff and exits 1."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    fixture_dir: Path = args.fixture_dir
    yaml_path = fixture_dir / _YAML_NAME
    hash_path = fixture_dir / _HASH_NAME

    if not yaml_path.exists():
        print(
            f"ERROR: gold-set YAML not found at {yaml_path}",
            file=sys.stderr,
        )
        return 2
    if not hash_path.exists():
        print(
            f"ERROR: gold-set hash file not found at {hash_path}",
            file=sys.stderr,
        )
        return 2

    yaml_bytes = yaml_path.read_bytes()
    new_hash = hashlib.sha256(yaml_bytes).hexdigest()
    old_hash = hash_path.read_text(encoding="utf-8").strip()

    print(f"Gold-set YAML: {yaml_path}")
    print(f"  bytes: {len(yaml_bytes)}")
    print(f"  old hash: {old_hash}")
    print(f"  new hash: {new_hash}")

    if old_hash == new_hash:
        print("No drift detected; hash is current.")
        return 0

    if not args.yes:
        print(
            "\nDrift detected. Re-run with --yes to overwrite the locked "
            "hash. REMINDER: any committed hash update MUST land alongside "
            "a Plan A threshold recalibration in the same commit.",
            file=sys.stderr,
        )
        return 1

    hash_path.write_text(new_hash, encoding="utf-8")
    print(
        f"\nWrote new hash to {hash_path}. Reminder: recalibrate Plan A "
        "diversity / style / anchor-verb thresholds in the same commit, "
        "and link the recalibration commit in the recompute commit's "
        "message."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
