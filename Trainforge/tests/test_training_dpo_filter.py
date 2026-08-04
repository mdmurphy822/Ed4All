"""The DPO preference filter and its pre-flight count must move in LOCKSTEP.

``compute_backend._filter_dpo_pairs`` decides which preference rows reach TRL;
``runner._count_dpo_eligible_records`` decides whether DPO runs at all (``min_dpo_pairs``). Widening one alone gives
the worst failure available in this subsystem: the count says "run DPO", the
filter then drops everything, and ``dpo_fail_hard=True`` kills the run AFTER
SFT already succeeded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from Trainforge.synthesis.synthesis_reject_mining import MINED_PAIR_SOURCE
from Trainforge.training.compute_backend import (
    DPO_EDITORIAL_SOURCES,
    _filter_dpo_pairs,
    is_dpo_editorial_record,
)
from Trainforge.training.runner import _count_dpo_eligible_records

MODE = "editorial_or_misconception"


def _records() -> List[Dict[str, Any]]:
    return [
        {"prompt": "p0", "chosen": "c0", "rejected": "r0",
         "source": MINED_PAIR_SOURCE},
        {"prompt": "p1", "chosen": "c1", "rejected": "r1",
         "source": "misconception"},
        {"prompt": "p2", "chosen": "c2", "rejected": "r2",
         "source": "misconception_editorial"},
        {"prompt": "p3", "chosen": "c3", "rejected": "r3",
         "misconception_id": "mc_0123456789abcdef"},
        # Neither editorial nor mined: must NOT be admitted.
        {"prompt": "p4", "chosen": "c4", "rejected": "r4",
         "source": "rule_synthesized"},
        {"prompt": "p5", "chosen": "c5", "rejected": "r5"},
    ]


def test_mined_rejection_is_in_the_shared_accept_set():
    assert MINED_PAIR_SOURCE in DPO_EDITORIAL_SOURCES


def test_filter_admits_mined_rejection():
    kept = _filter_dpo_pairs(_records(), MODE)
    sources = [r.get("source") for r in kept]
    assert MINED_PAIR_SOURCE in sources
    assert "rule_synthesized" not in sources
    assert len(kept) == 4


def test_filter_and_count_agree_on_the_same_records(tmp_path: Path):
    records = _records()
    path = tmp_path / "preference_pairs.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8",
    )
    kept = _filter_dpo_pairs(records, MODE)
    counted = _count_dpo_eligible_records(path, MODE)
    # Extending one site alone fails HERE, before it can kill a real run.
    assert counted == len(kept) == 4


@pytest.mark.parametrize("mode", ["all", ""])
def test_all_mode_is_unfiltered(mode: str, tmp_path: Path):
    records = _records()
    path = tmp_path / "preference_pairs.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8",
    )
    assert len(_filter_dpo_pairs(records, mode)) == len(records)
    assert _count_dpo_eligible_records(path, mode) == len(records)


def test_unknown_mode_still_raises():
    with pytest.raises(ValueError):
        _filter_dpo_pairs(_records(), "not_a_mode")


def test_predicate_is_the_single_source_of_truth():
    for rec in _records():
        expected = rec in _filter_dpo_pairs([rec], MODE)
        assert is_dpo_editorial_record(rec) is expected
