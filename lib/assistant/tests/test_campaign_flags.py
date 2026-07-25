"""Tests for the campaign env-overlay allowlist (:mod:`lib.assistant.campaign_flags`).

Load-bearing contracts: an unknown or forbidden key is a LOUD error that
names the key; a value carrying whitespace / shell metacharacters / an
over-long payload is rejected; every offending entry appears in one error
message (nothing silently dropped); the allowlist and deny-list are disjoint
and every allowlist member is a well-shaped env-var name. Hermetic — no
network, no subprocess, no filesystem.
"""

from __future__ import annotations

import pytest

from lib.assistant.campaign_flags import (
    ALLOWED_CAMPAIGN_FLAGS,
    FORBIDDEN_CAMPAIGN_FLAGS,
    KEY_RE,
    VALUE_RE,
    CampaignFlagError,
    is_allowed_flag,
    validate_overlay,
)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_valid_overlay_round_trips():
    overlay = {
        "COURSEFORGE_TWO_PASS": "true",
        "ED4ALL_PLANNING_GATE_RETRIES": "2",
        "ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER": "spark-super",
    }
    result = validate_overlay(overlay)
    assert result == overlay
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in result.items())


def test_empty_overlay_is_valid():
    assert validate_overlay({}) == {}


def test_non_string_value_is_normalized_to_string():
    # A JSON overlay may carry a bool/int; it normalizes to a VALUE_RE-clean str.
    result = validate_overlay({"ED4ALL_REWRITE_NUM_CTX": 16384})
    assert result == {"ED4ALL_REWRITE_NUM_CTX": "16384"}


# --------------------------------------------------------------------------- #
# Spot-checks on allow / deny membership
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "flag",
    [
        "COURSEFORGE_TWO_PASS",
        "ED4ALL_PLANNING_GATE_RETRIES",
        "ED4ALL_DYNAMIC_BLOCK_PLAN",
        "ED4ALL_TRIANGLE_FLOOR",
        "SEMANTIK_HEADING_JUDGE",
        "TRAINFORGE_REQUIRE_EMBEDDINGS",
    ],
)
def test_known_good_flags_are_allowed(flag):
    assert is_allowed_flag(flag) is True
    assert validate_overlay({flag: "1"}) == {flag: "1"}


@pytest.mark.parametrize(
    "flag",
    [
        "ED4ALL_SEAT_BASE_URLS",
        "NVIDIA_API_KEY",
        "ED4ALL_HOME",
        "ED4ALL_GATE_ADVISORY",
        "LOCAL_DISPATCHER_ALLOW_STUB",
        "ED4ALL_EMBEDDING_ALLOW_FAKE",
        "TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS",
        "ED4ALL_ASSISTANT_MODEL",
    ],
)
def test_known_bad_flags_are_not_allowed(flag):
    assert is_allowed_flag(flag) is False


# --------------------------------------------------------------------------- #
# Rejection paths
# --------------------------------------------------------------------------- #


def test_unknown_key_raises_and_names_the_key():
    with pytest.raises(CampaignFlagError) as excinfo:
        validate_overlay({"TOTALLY_MADE_UP_FLAG": "1"})
    assert "TOTALLY_MADE_UP_FLAG" in str(excinfo.value)


def test_forbidden_key_raises_and_names_the_key():
    with pytest.raises(CampaignFlagError) as excinfo:
        validate_overlay({"ED4ALL_SEAT_BASE_URLS": "spark=http://x"})
    assert "ED4ALL_SEAT_BASE_URLS" in str(excinfo.value)


def test_every_forbidden_member_is_rejected():
    for flag in FORBIDDEN_CAMPAIGN_FLAGS:
        assert is_allowed_flag(flag) is False
        with pytest.raises(CampaignFlagError):
            validate_overlay({flag: "1"})


@pytest.mark.parametrize(
    "bad_value",
    [
        "has space",
        "a;b",
        "a$b",
        "a`b",
        "a|b",
        "a&b",
        "$(whoami)",
        "a" * 201,
    ],
)
def test_bad_values_raise(bad_value):
    with pytest.raises(CampaignFlagError) as excinfo:
        validate_overlay({"COURSEFORGE_TWO_PASS": bad_value})
    assert "COURSEFORGE_TWO_PASS" in str(excinfo.value)


def test_value_at_max_length_is_accepted():
    ok = "a" * 200
    assert validate_overlay({"COURSEFORGE_TWO_PASS": ok}) == {
        "COURSEFORGE_TWO_PASS": ok
    }


def test_multiple_bad_entries_all_appear_in_one_error():
    with pytest.raises(CampaignFlagError) as excinfo:
        validate_overlay(
            {
                "UNKNOWN_ONE": "1",
                "ED4ALL_SEAT_BASE_URLS": "x",  # forbidden key
                "COURSEFORGE_TWO_PASS": "bad value",  # bad value
            }
        )
    message = str(excinfo.value)
    assert "UNKNOWN_ONE" in message
    assert "ED4ALL_SEAT_BASE_URLS" in message
    assert "COURSEFORGE_TWO_PASS" in message


def test_non_mapping_overlay_raises():
    with pytest.raises(CampaignFlagError):
        validate_overlay(["ED4ALL_TRIANGLE_FLOOR=1"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Set invariants
# --------------------------------------------------------------------------- #


def test_allowed_and_forbidden_are_disjoint():
    assert ALLOWED_CAMPAIGN_FLAGS.isdisjoint(FORBIDDEN_CAMPAIGN_FLAGS)


def test_every_allowed_member_matches_key_re():
    for flag in ALLOWED_CAMPAIGN_FLAGS:
        assert KEY_RE.match(flag), flag


def test_allowlist_is_curated_size():
    # ~40-80 curated build-tuning flags (guards against wildcard drift).
    assert 40 <= len(ALLOWED_CAMPAIGN_FLAGS) <= 80


def test_value_re_rejects_whitespace_and_metacharacters():
    assert VALUE_RE.match("clean-value_1.0") is not None
    for bad in (" x", "x y", "x;y", "x$y", "x`y", "x\ty", "x\ny"):
        assert VALUE_RE.match(bad) is None
