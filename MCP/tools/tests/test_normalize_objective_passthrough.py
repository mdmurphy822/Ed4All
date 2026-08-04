"""Round-trip tests: terminal-coverage back-pointers survive normalization.

Regression for the Layer-A orphan-terminal prune bug. The root cause was
that ``_normalize_objective_entry`` STRIPPED the CO→TO roll-up back-pointers
(``parent_to`` / ``parent_terminal`` / ``parent_terminal_id`` /
``terminal_id``) and the chapter attribution (``chapter`` / ``chapter_id``)
from every objective it normalized. With those keys gone:

* ``rolled_up_terminal_ids`` returned an empty set, so the Layer-A prune
  (``prune_orphan_terminals``) deleted EVERY terminal objective; and
* the Layer-B CO-less branch saw no ``chapter`` and raised
  ``ORPHAN_TERMINAL_NO_CHAPTER_REF`` false positives.

Downstream consumers that also depend on these keys surviving:
``Trainforge/rag/graphs/pedagogy_graph_builder.py`` (CO→TO edge validation) and
``lib/validators/libv2/packet_integrity.py`` (CO→TO expansion).

These pins assert the keys round-trip through both
``_normalize_objective_entry`` and ``load_objectives_json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from MCP.tools._content_gen_helpers import (  # noqa: E402
    _normalize_objective_entry,
    load_objectives_json,
)


# --- _normalize_objective_entry --------------------------------------------

def test_normalize_preserves_parent_terminal_backpointer() -> None:
    entry = _normalize_objective_entry(
        {
            "id": "CO-01",
            "statement": "Explain the bonding model in covalent compounds.",
            "parent_terminal": "TO-03",
        }
    )
    assert entry is not None
    assert entry["parent_terminal"] == "TO-03"


@pytest.mark.parametrize(
    "key", ["parent_to", "parent_terminal", "parent_terminal_id", "terminal_id"]
)
def test_normalize_preserves_all_parent_keys(key: str) -> None:
    entry = _normalize_objective_entry(
        {
            "id": "CO-02",
            "statement": "Describe the reaction mechanism for substitution.",
            key: "TO-07",
        }
    )
    assert entry is not None
    assert entry[key] == "TO-07"


def test_normalize_preserves_chapter_attribution() -> None:
    entry = _normalize_objective_entry(
        {
            "id": "TO-05",
            "statement": "Apply the gas laws to closed-system problems.",
            "chapter": "ch4",
            "chapter_id": "ch4",
        }
    )
    assert entry is not None
    assert entry["chapter"] == "ch4"
    assert entry["chapter_id"] == "ch4"


def test_normalize_omits_absent_passthrough_keys() -> None:
    # When the back-pointers are absent they must NOT be invented.
    entry = _normalize_objective_entry(
        {
            "id": "TO-09",
            "statement": "Summarize the stoichiometry of a balanced equation.",
        }
    )
    assert entry is not None
    for key in (
        "parent_to",
        "parent_terminal",
        "parent_terminal_id",
        "terminal_id",
        "chapter",
        "chapter_id",
    ):
        assert key not in entry


# --- load_objectives_json --------------------------------------------------

def test_load_objectives_json_round_trips_backpointers(tmp_path: Path) -> None:
    payload = {
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Analyze the periodic trends across a group.",
                "chapter": "ch2",
            }
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {
                        "id": "CO-01",
                        "statement": "Recall the noble-gas electron config.",
                        "parent_terminal": "TO-01",
                    }
                ],
            }
        ],
    }
    obj_path = tmp_path / "synthesized_objectives.json"
    obj_path.write_text(json.dumps(payload), encoding="utf-8")

    terminal, chapter = load_objectives_json(str(obj_path))

    assert len(terminal) == 1
    assert terminal[0]["chapter"] == "ch2"
    assert len(chapter) == 1
    assert chapter[0]["parent_terminal"] == "TO-01"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
