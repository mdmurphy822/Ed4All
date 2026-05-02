"""Wave 137 follow-up: tests for the FORM_DATA backfill loop.

Wave 136d originally shipped an interactive operator-paused contract
(y/n/e/q). The Wave 137 follow-up converted that to a fully-automatic
loop: each CURIE is auto-drafted, auto-validated, and on validator
failure auto-redrafted up to MAX_REDRAFTS=10 times with cumulative
violation feedback. No operator interaction; review happens at git
diff time.

Tests pin the automatic contract:

1. ``test_backfill_sorts_by_frequency_descending`` — synthetic
   chunks.jsonl with known CURIE frequencies; the loop visits them
   in descending-frequency order.
2. ``test_backfill_auto_accepts_on_validator_pass`` — mock subprocess
   returns valid YAML; validator passes; the overlay gets the new
   entry while every pre-existing entry survives. No input_fn needed.
3. ``test_backfill_rejects_when_validator_fails_after_append`` —
   subprocess returns valid-looking YAML that trips Wave 136b's
   content validator on append; the loop auto-redrafts MAX_REDRAFTS
   times, returns ``max_redrafts_exceeded``, and the YAML file ends
   byte-identical to pre-merge.
4. ``test_backfill_max_redrafts_exhausted_returns_dedicated_outcome``
   — every draft fails; expect MAX_REDRAFTS=10 + 1 initial = 11
   runner calls and a ``max_redrafts_exceeded`` outcome.
5. ``test_backfill_accumulates_violations_across_redrafts`` — each
   redraft's prompt receives the cumulative dedup'd set of prior
   violations.

All test fixtures use synthetic ``test:Foo``-style CURIEs because
this CLI must work generically across families. The property
manifest is monkey-patched to return synthetic CURIEs only;
``_load_form_data`` is monkey-patched so we don't depend on the
shipping rdf_shacl YAML either.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.scripts import backfill_form_data as cli  # noqa: E402
from Trainforge.generators.schema_translation_generator import (  # noqa: E402
    SurfaceFormData,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


_SYNTHETIC_CURIES = [
    ("test:Alpha", "Alpha label"),
    ("test:Beta", "Beta label"),
    ("test:Gamma", "Gamma label"),
]


def _build_synthetic_manifest():
    """Return a dataclass-shaped manifest stub.

    Mirrors the duck-typed surface ``cli.main`` reads from the real
    ``PropertyManifest``: ``.properties`` is a list of objects with
    ``.curie``, ``.label``, ``.surface_forms``, ``.min_pairs``.
    """
    return SimpleNamespace(
        family="test_family",
        properties=[
            SimpleNamespace(
                curie=curie,
                label=label,
                surface_forms=[curie],
                min_pairs=1,
            )
            for curie, label in _SYNTHETIC_CURIES
        ],
    )


def _build_synthetic_form_data() -> Dict[str, SurfaceFormData]:
    """All three CURIEs as degraded_placeholder."""
    out: Dict[str, SurfaceFormData] = {}
    for curie, _label in _SYNTHETIC_CURIES:
        out[curie] = SurfaceFormData(
            curie=curie,
            short_name=curie.split(":")[-1],
            anchored_status="degraded_placeholder",
            definitions=["[degraded: placeholder for " + curie + "]"],
            usage_examples=[
                ("[degraded: usage prompt]", "[degraded: usage answer]")
            ],
        )
    return out


def _build_valid_yaml_payload(curie: str) -> str:
    """Render a YAML payload + next-steps block matching the drafting CLI's stdout shape.

    The merged definition + usage_example strings clear Wave 136b's
    content-quality rules: 50-400 char definitions, contains the
    literal CURIE, no forbidden prefixes, no placeholder leak tokens.
    """
    short = curie.split(":")[-1]
    text_block = (
        f"family: test_family\n"
        f"forms:\n"
        f"  {curie}:\n"
        f"    short_name: {short}\n"
        f"    anchored_status: complete\n"
        f"    definitions:\n"
    )
    # 7 definitions, each >50 chars and containing the CURIE literally.
    for i in range(7):
        text_block += (
            f"    - {curie} is a synthetic test definition number {i} "
            f"crafted to satisfy the structural floor with concrete content.\n"
        )
    text_block += "    usage_examples:\n"
    for i in range(7):
        text_block += (
            f"    - - Synthetic prompt number {i} probing the use of "
            f"{curie} in a structurally-sound way.\n"
            f"      - When the test fixture references {curie}, the "
            f"answer demonstrates how {curie} appears in a paired body.\n"
        )
    next_steps = (
        "\n# NEXT STEPS\n"
        f"# 1. Append the `forms.{curie}:` block above to:\n"
        f"#    schemas/training/schema_translation_catalog.test_family.yaml\n"
    )
    return text_block + next_steps


def _build_invalid_yaml_payload(curie: str) -> str:
    """YAML payload that satisfies the YAML parse but trips Wave 136b's
    content-quality check via the placeholder-leak token.

    The definition line carries the literal ``"[degraded:"`` prefix
    which Wave 136b classifies as ``PLACEHOLDER_LEAK`` for any entry
    marked ``anchored_status="complete"``.
    """
    short = curie.split(":")[-1]
    text_block = (
        f"family: test_family\n"
        f"forms:\n"
        f"  {curie}:\n"
        f"    short_name: {short}\n"
        f"    anchored_status: complete\n"
        f"    definitions:\n"
    )
    # Seed the leak token (fails Wave 136b PLACEHOLDER_LEAK).
    text_block += (
        f"    - {curie} is a synthetic test definition that wrongly "
        f"includes the literal [degraded: token to trip the leak rule.\n"
    )
    for i in range(6):
        text_block += (
            f"    - {curie} is a structurally-valid replacement definition "
            f"line number {i} that is otherwise fine.\n"
        )
    text_block += "    usage_examples:\n"
    for i in range(7):
        text_block += (
            f"    - - Synthetic prompt number {i} probing the use of "
            f"{curie} in a structurally-sound way.\n"
            f"      - When the test fixture references {curie}, the "
            f"answer demonstrates how {curie} appears in a paired body.\n"
        )
    next_steps = (
        "\n# NEXT STEPS\n"
        f"# 1. Append the `forms.{curie}:` block above to:\n"
        f"#    schemas/training/schema_translation_catalog.test_family.yaml\n"
    )
    return text_block + next_steps


def _make_chunks_jsonl(tmp_path: Path, freq_map: Dict[str, int]) -> Path:
    """Write a chunks.jsonl whose chunk text contains each CURIE exactly N times.

    Each chunk is one line; ``text`` is "<curie> " * N for clear
    substring counting. The CLI's frequency counter walks the file
    once and tallies substring matches.
    """
    out = tmp_path / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        idx = 0
        for curie, n in freq_map.items():
            for _ in range(n):
                fh.write(json.dumps({"chunk_id": f"c{idx}",
                                     "text": curie + " "}) + "\n")
                idx += 1
    return out


# ----------------------------------------------------------------------
# Test 1: frequency-descending sort
# ----------------------------------------------------------------------


def test_backfill_sorts_by_frequency_descending(tmp_path):
    """High-freq CURIE visited before low-freq CURIE (default --by frequency)."""
    chunks_path = _make_chunks_jsonl(
        tmp_path,
        {
            "test:Alpha": 1,
            "test:Beta": 5,  # highest
            "test:Gamma": 3,
        },
    )
    counts = cli._count_curie_frequencies(
        chunks_path, ["test:Alpha", "test:Beta", "test:Gamma"]
    )
    assert counts == {"test:Alpha": 1, "test:Beta": 5, "test:Gamma": 3}

    ordered = cli._sort_targets(
        ["test:Alpha", "test:Beta", "test:Gamma"], counts, by="frequency"
    )
    visited_curies = [c for c, _ in ordered]
    assert visited_curies == ["test:Beta", "test:Gamma", "test:Alpha"], (
        f"expected freq-desc order; got {visited_curies}"
    )

    # And alphabetical comparison.
    ordered_alpha = cli._sort_targets(
        ["test:Beta", "test:Gamma", "test:Alpha"], counts, by="alphabetical"
    )
    assert [c for c, _ in ordered_alpha] == [
        "test:Alpha", "test:Beta", "test:Gamma"
    ]


# ----------------------------------------------------------------------
# Test 2: validator passes -> auto-accept + preserve existing entries
# ----------------------------------------------------------------------


def test_backfill_auto_accepts_on_validator_pass(tmp_path):
    """Wave 137 followup: fully-automatic mode. Mock subprocess returns
    valid YAML, validator passes, loop auto-accepts (no operator
    prompt). YAML overlay gets the new entry while preserving the
    pre-existing entry."""
    yaml_path = tmp_path / "schema_translation_catalog.test_family.yaml"
    # Pre-seed an existing complete entry that must NOT be erased.
    pre_text = (
        "family: test_family\n"
        "forms:\n"
        "  test:Existing:\n"
        "    short_name: Existing\n"
        "    anchored_status: complete\n"
        "    definitions:\n"
        "    - test:Existing is a pre-seeded entry that must survive a "
        "round-trip through the merge step intact for safety.\n"
        "    usage_examples:\n"
        "    - - Pre-seeded prompt about test:Existing usage example.\n"
        "      - Pre-seeded answer demonstrating test:Existing in a body.\n"
    )
    yaml_path.write_text(pre_text, encoding="utf-8")

    target_curie = "test:Beta"

    fake_stdout_yaml = _build_valid_yaml_payload(target_curie)

    def fake_runner(curie, family, course_code, provider, model, timeout=None, prior_violations=None, semantic_profile_name=None):
        assert curie == target_curie
        return 0, fake_stdout_yaml, ""

    # Stub validator to always pass — Wave 136b's full validator is
    # exercised in its own test file; here we just want to assert the
    # merge works.
    def fake_validator(form_data, manifest_curies, semantic_profile=None):
        return {"passed": True, "content_violations": []}

    fake_form_data = {
        target_curie: SurfaceFormData(
            curie=target_curie,
            short_name="Beta",
            anchored_status="degraded_placeholder",
            definitions=["[degraded: stub]"],
            usage_examples=[("[degraded: prompt]", "[degraded: answer]")],
        )
    }

    sink = io.StringIO()

    def captured_print(*args, **kwargs):
        kwargs["file"] = sink
        print(*args, **kwargs)

    with patch.object(cli, "_run_drafting_cli", side_effect=fake_runner), \
         patch.object(cli, "validate_form_data_contract",
                      side_effect=fake_validator), \
         patch.object(cli, "_load_form_data", return_value=fake_form_data), \
         patch.object(cli, "load_property_manifest",
                      return_value=_build_synthetic_manifest()), \
         patch.object(cli, "_resolve_chunks_jsonl", return_value=None):
        rc = cli._process_one_curie(
            idx=1,
            total=1,
            curie=target_curie,
            freq=0,
            label="Beta label",
            family="test_family",
            course_code="test-course",
            provider="local",
            model=None,
            yaml_path=yaml_path,
            manifest_curies=[c for c, _ in _SYNTHETIC_CURIES],
            print_fn=captured_print,
        )
    assert rc == "accepted", f"expected 'accepted'; got {rc!r}"

    # Read the merged file: pre-existing entry preserved, new entry added.
    import yaml as _yaml
    merged = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    forms = merged["forms"]
    assert "test:Existing" in forms, (
        f"pre-existing entry was erased; got keys: {list(forms.keys())}"
    )
    assert target_curie in forms, (
        f"new entry not appended; got keys: {list(forms.keys())}"
    )
    new_entry = forms[target_curie]
    assert new_entry["anchored_status"] == "complete"
    assert len(new_entry["definitions"]) == 7
    assert all(target_curie in d for d in new_entry["definitions"])


# ----------------------------------------------------------------------
# Test 3: validator failure after append -> rollback + failed_validation
# ----------------------------------------------------------------------


def test_backfill_rejects_when_validator_fails_after_append(tmp_path):
    """Wave 136b validator fails on every appended entry. Wave 137
    follow-up (fully-automatic): the loop auto-redrafts MAX_REDRAFTS=10
    times then returns ``max_redrafts_exceeded``; the YAML file ends
    byte-identical to pre-merge state because every append is rolled
    back."""
    yaml_path = tmp_path / "schema_translation_catalog.test_family.yaml"
    pre_text = (
        "family: test_family\n"
        "forms:\n"
        "  test:Existing:\n"
        "    short_name: Existing\n"
        "    anchored_status: complete\n"
        "    definitions:\n"
        "    - test:Existing is a pre-seeded entry that must survive a "
        "rollback round-trip through the validator-failure code path.\n"
        "    usage_examples:\n"
        "    - - Pre-seeded prompt about test:Existing.\n"
        "      - Pre-seeded answer demonstrating test:Existing usage.\n"
    )
    yaml_path.write_text(pre_text, encoding="utf-8")
    pre_bytes = yaml_path.read_bytes()

    target_curie = "test:Beta"
    bad_yaml = _build_invalid_yaml_payload(target_curie)

    def fake_runner(curie, family, course_code, provider, model, timeout=None, prior_violations=None, semantic_profile_name=None):
        return 0, bad_yaml, ""

    def fake_validator(form_data, manifest_curies, semantic_profile=None):
        return {
            "passed": False,
            "content_violations": [
                {"curie": target_curie, "rule": "PLACEHOLDER_LEAK",
                 "detail": "[degraded: token in definition"},
            ],
            "missing_curies": [],
            "incomplete_curies": [],
        }

    sink = io.StringIO()

    def captured_print(*args, **kwargs):
        kwargs["file"] = sink
        print(*args, **kwargs)

    fake_form_data = {
        target_curie: SurfaceFormData(
            curie=target_curie,
            short_name="Beta",
            anchored_status="degraded_placeholder",
            definitions=["[degraded: stub]"],
            usage_examples=[("[degraded: prompt]", "[degraded: answer]")],
        )
    }

    with patch.object(cli, "_run_drafting_cli", side_effect=fake_runner), \
         patch.object(cli, "validate_form_data_contract",
                      side_effect=fake_validator), \
         patch.object(cli, "_load_form_data", return_value=fake_form_data), \
         patch.object(cli, "load_property_manifest",
                      return_value=_build_synthetic_manifest()), \
         patch.object(cli, "_resolve_chunks_jsonl", return_value=None):
        outcome = cli._process_one_curie(
            idx=1,
            total=1,
            curie=target_curie,
            freq=0,
            label="Beta label",
            family="test_family",
            course_code="test-course",
            provider="local",
            model=None,
            yaml_path=yaml_path,
            manifest_curies=[c for c, _ in _SYNTHETIC_CURIES],
            print_fn=captured_print,
        )
    assert outcome == "max_redrafts_exceeded", (
        f"expected 'max_redrafts_exceeded' (auto-redraft chain exhausted); got {outcome!r}"
    )
    assert yaml_path.read_bytes() == pre_bytes, (
        "rollback failed — YAML file did not return to pre-merge state"
    )


def test_backfill_max_redrafts_exhausted_returns_dedicated_outcome(tmp_path):
    """Wave 137 follow-up (fully-automatic): when every draft fails
    append-time validation, the loop runs MAX_REDRAFTS=10 attempts and
    then returns ``max_redrafts_exceeded`` (not ``skipped`` or
    ``failed_validation``) so the main() exit code can flag the
    failure."""
    yaml_path = tmp_path / "schema_translation_catalog.test_family.yaml"
    pre_text = "family: test_family\nforms: {}\n"
    yaml_path.write_text(pre_text, encoding="utf-8")

    target_curie = "test:Beta"
    bad_yaml = _build_invalid_yaml_payload(target_curie)

    runner_calls = []

    def fake_runner(curie, family, course_code, provider, model, timeout=None, prior_violations=None, semantic_profile_name=None):
        runner_calls.append(curie)
        return 0, bad_yaml, ""

    def fake_validator(form_data, manifest_curies, semantic_profile=None):
        return {
            "passed": False,
            "content_violations": [
                {"curie": target_curie, "rule": "PLACEHOLDER_LEAK", "detail": "..."},
            ],
            "missing_curies": [],
            "incomplete_curies": [],
        }

    fake_form_data = {
        target_curie: SurfaceFormData(
            curie=target_curie,
            short_name="Beta",
            anchored_status="degraded_placeholder",
            definitions=["[degraded: stub]"],
            usage_examples=[("[degraded: prompt]", "[degraded: answer]")],
        )
    }

    with patch.object(cli, "_run_drafting_cli", side_effect=fake_runner), \
         patch.object(cli, "validate_form_data_contract",
                      side_effect=fake_validator), \
         patch.object(cli, "_load_form_data", return_value=fake_form_data), \
         patch.object(cli, "load_property_manifest",
                      return_value=_build_synthetic_manifest()), \
         patch.object(cli, "_resolve_chunks_jsonl", return_value=None):
        outcome = cli._process_one_curie(
            idx=1,
            total=1,
            curie=target_curie,
            freq=0,
            label="Beta label",
            family="test_family",
            course_code="test-course",
            provider="local",
            model=None,
            yaml_path=yaml_path,
            manifest_curies=[c for c, _ in _SYNTHETIC_CURIES],
            print_fn=lambda *a, **kw: None,
        )
    assert outcome == "max_redrafts_exceeded", (
        f"expected 'max_redrafts_exceeded' after MAX_REDRAFTS auto-redrafts; got {outcome!r}"
    )
    # MAX_REDRAFTS=10 means 10 attempts total: 1 initial + 9 redrafts
    # (the 10th rejection trips the cap before the 11th call would fire).
    assert len(runner_calls) == 10, (
        f"expected 10 runner calls (MAX_REDRAFTS attempts); got {len(runner_calls)}"
    )


def test_backfill_accumulates_violations_across_redrafts(tmp_path):
    """Wave 137 follow-up: each redraft's prompt receives the
    CUMULATIVE set of unique violations across all prior attempts,
    not just the immediately-previous attempt's. Persistent failure
    modes reinforce across the chain."""
    yaml_path = tmp_path / "schema_translation_catalog.test_family.yaml"
    yaml_path.write_text("family: test_family\nforms: {}\n", encoding="utf-8")

    target_curie = "test:Beta"
    bad_yaml = _build_invalid_yaml_payload(target_curie)

    runner_calls: List[Tuple[Optional[List[str]], ...]] = []

    def fake_runner(curie, family, course_code, provider, model,
                    timeout=None, prior_violations=None,
                    semantic_profile_name=None):
        # Capture every prior_violations payload threaded into the
        # subprocess so we can assert cumulative growth.
        runner_calls.append(tuple(prior_violations or []))
        return 0, bad_yaml, ""

    # Each attempt produces a DIFFERENT violation, so the cumulative
    # set should grow across attempts.
    call_index = {"i": 0}

    def fake_validator(form_data, manifest_curies, semantic_profile=None):
        call_index["i"] += 1
        return {
            "passed": False,
            "content_violations": [
                {
                    "curie": target_curie,
                    "code": f"VIOLATION_CODE_{call_index['i']}",
                    "detail": f"detail for attempt {call_index['i']}",
                },
            ],
            "missing_curies": [],
            "incomplete_curies": [],
        }

    fake_form_data = {
        target_curie: SurfaceFormData(
            curie=target_curie,
            short_name="Beta",
            anchored_status="degraded_placeholder",
            definitions=["[degraded: stub]"],
            usage_examples=[("[degraded: prompt]", "[degraded: answer]")],
        )
    }

    with patch.object(cli, "_run_drafting_cli", side_effect=fake_runner), \
         patch.object(cli, "validate_form_data_contract",
                      side_effect=fake_validator), \
         patch.object(cli, "_load_form_data", return_value=fake_form_data), \
         patch.object(cli, "load_property_manifest",
                      return_value=_build_synthetic_manifest()), \
         patch.object(cli, "_resolve_chunks_jsonl", return_value=None):
        cli._process_one_curie(
            idx=1,
            total=1,
            curie=target_curie,
            freq=0,
            label="Beta label",
            family="test_family",
            course_code="test-course",
            provider="local",
            model=None,
            yaml_path=yaml_path,
            manifest_curies=[c for c, _ in _SYNTHETIC_CURIES],
            print_fn=lambda *a, **kw: None,
        )

    # Initial attempt: no prior_violations.
    assert runner_calls[0] == ()
    # Auto-redraft 1: should carry the violation from attempt 1.
    assert "VIOLATION_CODE_1" in " ".join(runner_calls[1])
    # Auto-redraft 2: should carry violations from attempts 1 + 2.
    assert "VIOLATION_CODE_1" in " ".join(runner_calls[2])
    assert "VIOLATION_CODE_2" in " ".join(runner_calls[2])
    # Auto-redraft 3: should carry attempts 1 + 2 + 3.
    assert "VIOLATION_CODE_1" in " ".join(runner_calls[3])
    assert "VIOLATION_CODE_2" in " ".join(runner_calls[3])
    assert "VIOLATION_CODE_3" in " ".join(runner_calls[3])
    # Cumulative count grows monotonically (no amnesia).
    assert len(runner_calls[3]) > len(runner_calls[2]) > len(runner_calls[1])
