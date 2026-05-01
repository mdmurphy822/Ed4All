"""Wave 137c — gold-set drift sentinel.

The first 6 complete FORM_DATA entries are the calibration reference
for Plan A's diversity / style / anchor-verb thresholds. Changes to
their content invalidate every Wave 137 threshold; this test catches
accidental drift.

Drift recovery:
  1. python -m Trainforge.scripts.recompute_gold_set_hash --yes
  2. Document threshold recalibration in the same commit.
  3. Link recalibration commit in this commit's message.
"""
import hashlib
from pathlib import Path

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "form_data_gold_set"


def test_gold_set_hash_is_locked() -> None:
    yaml_bytes = (_FIXTURE_DIR / "rdf_shacl_gold_set.yaml").read_bytes()
    expected = (_FIXTURE_DIR / "rdf_shacl_gold_set_hash.txt").read_text().strip()
    actual = hashlib.sha256(yaml_bytes).hexdigest()
    assert actual == expected, (
        f"Gold set drifted. Expected {expected[:12]}..., got {actual[:12]}.... "
        "If intentional, run `python -m Trainforge.scripts.recompute_gold_set_hash --yes` "
        "and document threshold recalibration in the same commit."
    )
