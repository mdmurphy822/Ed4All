"""GPT Feedback v2 Wave 1 (W1.C) — course-status helper tests.

Pins the public surface of :mod:`lib.governance.course_status`:

  * :func:`load_course_status_schema` returns a dict carrying all five
    canonical enum values.
  * :func:`compose_course_status` returns ``"failed"`` when any arrow has
    ``promotion_decision == "fail"``.
  * :func:`compose_course_status` returns ``"non_certified_archive"`` when
    arrows 1-5 pass and arrow 6 is ``"missing"``.
  * :func:`compose_course_status` raises :class:`NotImplementedError` for
    the three ``certified_*`` branches with a message pointing at Wave 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Repo-root sys.path bootstrap (mirror lib/tests/test_decision_capture.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.governance.course_status import (  # noqa: E402
    compose_course_status,
    load_course_status_schema,
)

EXPECTED_ENUM = [
    "certified_accessible",
    "certified_instructional",
    "certified_trainable",
    "non_certified_archive",
    "failed",
]


# ---------------------------------------------------------------------------
# load_course_status_schema
# ---------------------------------------------------------------------------


def test_load_course_status_schema_returns_dict_with_full_enum():
    schema = load_course_status_schema()
    assert isinstance(schema, dict)
    assert "enum" in schema
    assert schema["enum"] == EXPECTED_ENUM
    assert schema["type"] == "string"
    # Defensive-copy contract: mutating the returned dict must not pollute
    # subsequent loads (mirrors lib.ontology.taxonomy semantics).
    schema["enum"].append("totally_made_up")
    again = load_course_status_schema()
    assert again["enum"] == EXPECTED_ENUM


# ---------------------------------------------------------------------------
# compose_course_status — failed branch
# ---------------------------------------------------------------------------


def test_compose_course_status_returns_failed_on_any_fail():
    assert compose_course_status({"arrow1": "fail"}) == "failed"


def test_compose_course_status_returns_failed_when_one_of_many_fails():
    decisions = {
        "arrow1": "pass",
        "arrow2": "pass",
        "arrow3": "fail",
        "arrow4": "pass",
        "arrow5": "pass",
    }
    assert compose_course_status(decisions) == "failed"


# ---------------------------------------------------------------------------
# compose_course_status — non_certified_archive branch
# ---------------------------------------------------------------------------


def test_compose_course_status_returns_non_certified_archive_when_arrows_1_5_pass_and_6_missing():
    decisions = {
        "arrow1": "pass",
        "arrow2": "pass",
        "arrow3": "pass",
        "arrow4": "pass",
        "arrow5": "pass",
        "arrow6": "missing",
    }
    assert compose_course_status(decisions) == "non_certified_archive"


def test_compose_course_status_non_certified_archive_when_only_arrows_1_5_present():
    # When the trainable slice is entirely absent (no arrow6+ keys at all),
    # the helper still routes to non_certified_archive — arrows 1-5 pass
    # and there is no passing trainable arrow.
    decisions = {
        "arrow1": "pass",
        "arrow2": "warn",
        "arrow3": "pass",
        "arrow4": "pass",
        "arrow5": "pass",
    }
    assert compose_course_status(decisions) == "non_certified_archive"


# ---------------------------------------------------------------------------
# compose_course_status — certified_* (Wave 3 NotImplementedError)
# ---------------------------------------------------------------------------


def test_compose_course_status_raises_not_implemented_for_full_chain_pass():
    decisions = {f"arrow{i}": "pass" for i in range(1, 10)}
    with pytest.raises(NotImplementedError) as excinfo:
        compose_course_status(decisions)
    message = str(excinfo.value)
    assert "Wave 3" in message
    assert "certified" in message.lower()


def test_compose_course_status_not_implemented_message_points_at_plan():
    decisions = {f"arrow{i}": "pass" for i in range(1, 10)}
    with pytest.raises(NotImplementedError) as excinfo:
        compose_course_status(decisions)
    # Pin the deferral citation so a future drift on the message is loud.
    assert "gpt-feedback-2-wave1-schemas-2026-05" in str(excinfo.value)
