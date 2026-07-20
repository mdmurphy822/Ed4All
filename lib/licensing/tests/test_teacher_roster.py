"""SFT-C S6/S7 — teacher-roster + fail-closed license-guard tests.

Fully offline: pure Python logic, no network, no model load.
"""

from __future__ import annotations

import pytest

from lib.licensing import (
    GENERAL_NVIDIA_OML_IDENTITY,
    NEMOTRON_PINNED_LICENSE_IDENTITY,
    NEMOTRON_ROSTER_KEY,
    TEACHER_ROSTER,
    LicenseGuardError,
    assert_checkpoint_license,
    assert_export_licenses,
    assert_nemotron_pin,
    classify_pair_teacher,
    license_for_model,
    provenance_license_tag,
    provider_verdict_roster,
    stamp_pair_license,
)
from lib.licensing.teacher_roster import LicenseRecord


def test_provider_verdict_roster_shape_for_apply_arm():
    roster = provider_verdict_roster()
    # Consumed by assessment_generator._apply_arm_provider_allowed, which keys
    # on the coarse provider name and refuses a `barred` verdict.
    assert roster["local"]["license_verdict"] == "safe"
    assert roster["nvidia"]["verdict"] == "barred"
    assert roster["anthropic"]["license_verdict"] == "barred"
    assert roster["together"]["verdict"] == "conditional"


def test_provider_verdict_roster_integrates_with_apply_arm_guard():
    """SFT-C flow-down: the roster shape feeds the assessment apply-arm license
    guard, which refuses a barred teacher and admits a safe one."""
    from Trainforge.generators.assessment_generator import (
        _apply_arm_provider_allowed,
    )

    roster = provider_verdict_roster()
    assert _apply_arm_provider_allowed("local", roster) is True
    assert _apply_arm_provider_allowed("nvidia", roster) is False
    assert _apply_arm_provider_allowed("anthropic", roster) is False


def test_pipeline_tools_wires_provider_verdict_roster():
    """SFT-C stitch: BOTH AssessmentGenerator construction sites in
    pipeline_tools thread the S6 teacher-license roster so the apply-arm license
    guard activates. Source-level regression against silent removal."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    src = (root / "MCP" / "tools" / "pipeline_tools.py").read_text(encoding="utf-8")
    assert "provider_verdict_roster" in src
    # Both construction sites pass the resolved roster.
    assert src.count("teacher_roster=_teacher_roster") == 2


# --------------------------------------------------------------------------- #
# Roster resolution                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model_id,expected_key,expected_verdict",
    [
        ("qwen2.5:7b-instruct-q4_K_M", "qwen2.5-7b", "safe"),
        ("Qwen/Qwen2.5-14B-Instruct", "qwen2.5-14b", "safe"),
        ("qwen2.5-32b", "qwen2.5-32b", "safe"),
        ("Qwen/Qwen2.5-72B-Instruct-Turbo", "qwen2.5-72b", "conditional"),
        ("qwen2.5-3b-instruct", "qwen2.5-3b", "barred"),
        ("Qwen/Qwen2.5-1.5B", "qwen2.5-1.5b", "safe"),
        ("nvidia/nemotron-3-super-120b-a12b", NEMOTRON_ROSTER_KEY, "safe"),
        ("mistral-large-2411", "mistral-mrl", "barred"),
        ("codestral-22b", "mistral-mrl", "barred"),
        ("mistralai/Mixtral-8x7B-Instruct", "mistral-apache", "safe"),
        ("mistral-nemo-12b", "mistral-apache", "safe"),
        ("microsoft/Phi-3.5-mini-instruct", "phi", "safe"),
        ("glm-4.6", "glm", "safe"),
        ("allenai/OLMo-2-1124-7B", "olmo", "safe"),
        ("deepseek-ai/DeepSeek-V3", "deepseek", "safe"),
        ("HuggingFaceTB/SmolLM2-1.7B", "smollm2", "safe"),
        ("google/gemma-2-9b", "gemma", "conditional"),
        ("meta-llama/Llama-3.3-70B-Instruct", "llama", "barred"),
        ("claude-sonnet-4-6", "anthropic", "barred"),
    ],
)
def test_license_for_model_resolution(model_id, expected_key, expected_verdict):
    rec = license_for_model(model_id)
    assert rec is not None
    assert rec.name == expected_key
    assert rec.verdict == expected_verdict


def test_license_for_model_unregistered_returns_none():
    assert license_for_model("some-unknown-model-9000") is None
    assert license_for_model("") is None
    assert license_for_model(None) is None


def test_all_supported_base_models_resolve():
    # Every Trainforge SUPPORTED_BASES short name must resolve so the
    # runner's per-checkpoint ingest assertion never false-bars a base.
    for short_name in (
        "qwen2.5-1.5b",
        "llama-3.2-1b",
        "llama-3.2-3b",
        "smollm2-1.7b",
        "phi-3.5-mini",
    ):
        assert license_for_model(short_name) is not None, short_name


# --------------------------------------------------------------------------- #
# Per-pair provenance helpers                                                  #
# --------------------------------------------------------------------------- #


def test_provenance_license_tag_generating_seat_wins():
    tag = provenance_license_tag(
        generating_seat="Qwen/Qwen2.5-32B-Instruct", provider="local"
    )
    assert tag == "qwen2.5-32b/Apache-2.0/safe"


def test_provenance_license_tag_provider_fallback():
    assert provenance_license_tag(provider="local") == "local/provenance/safe"
    assert provenance_license_tag(provider="together") == "together/provenance/conditional"


def test_provenance_license_tag_unregistered_and_unknown():
    assert provenance_license_tag(generating_seat="mystery-model") == "unregistered"
    assert provenance_license_tag() == "unknown"


def test_stamp_pair_license_is_additive():
    pair = {"prompt": "p", "completion": "c", "provider": "local"}
    out = stamp_pair_license(pair, generating_seat="qwen2.5:32b", provider="local")
    assert out is pair  # in place
    assert pair["provider"] == "local"  # closed-enum field untouched
    assert pair["generating_seat"] == "qwen2.5:32b"
    assert pair["license"] == "qwen2.5-32b/Apache-2.0/safe"


# --------------------------------------------------------------------------- #
# Pair classification                                                          #
# --------------------------------------------------------------------------- #


def test_classify_legacy_pair_is_ok():
    status, verdict, teacher = classify_pair_teacher({"prompt": "x", "completion": "y"})
    assert status == "ok"
    assert verdict == "legacy"


def test_classify_local_provider_ok():
    assert classify_pair_teacher({"provider": "local"})[0] == "ok"
    assert classify_pair_teacher({"provider": "together"})[0] == "ok"


def test_classify_claude_and_barred_and_unregistered():
    assert classify_pair_teacher({"provider": "anthropic"})[0] == "claude"
    assert classify_pair_teacher({"provider": "claude_session"})[0] == "claude"
    assert classify_pair_teacher({"provider": "nvidia"})[0] == "barred"
    assert classify_pair_teacher({"provider": "made_up"})[0] == "unregistered"


def test_classify_generating_seat_overrides_provider():
    # A pair tagged provider=local but generating_seat=llama is caught.
    status, _, teacher = classify_pair_teacher(
        {"provider": "local", "generating_seat": "meta-llama/Llama-3.3-70B"}
    )
    assert status == "barred"
    assert teacher == "llama"


# --------------------------------------------------------------------------- #
# Export-time fail-closed filter                                              #
# --------------------------------------------------------------------------- #


def test_assert_export_clean_corpus_passes():
    pairs = [
        {"prompt": "a", "completion": "b"},                       # legacy
        {"provider": "local", "prompt": "c"},                     # safe
        {"provider": "together", "prompt": "d"},                  # conditional
        {"provider": "local", "generating_seat": "qwen2.5:32b"},  # safe fine
        {"provider": "deterministic", "template_id": "kg.1"},     # deterministic
    ]
    assert assert_export_licenses(pairs) is None


def test_assert_export_barred_teacher_raises_naming_pair():
    pairs = [
        {"provider": "local", "chunk_id": "c0"},
        {"provider": "local", "generating_seat": "meta-llama/Llama-3.3-70B", "chunk_id": "c1"},
    ]
    with pytest.raises(LicenseGuardError) as exc:
        assert_export_licenses(pairs, source_desc="unit/corpus")
    msg = str(exc.value)
    assert "pair #1" in msg
    assert "llama" in msg
    assert "unit/corpus" in msg


def test_assert_export_claude_tagged_raises():
    with pytest.raises(LicenseGuardError, match="Claude/Anthropic"):
        assert_export_licenses([{"provider": "anthropic", "chunk_id": "c"}])
    with pytest.raises(LicenseGuardError, match="Claude/Anthropic"):
        assert_export_licenses([{"provider": "claude_session"}])


def test_assert_export_unregistered_raises():
    with pytest.raises(LicenseGuardError, match="UNREGISTERED"):
        assert_export_licenses([{"generating_seat": "unknown-teacher-x"}])


def test_assert_export_hosted_nvidia_barred():
    with pytest.raises(LicenseGuardError, match="BARRED"):
        assert_export_licenses([{"provider": "nvidia", "chunk_id": "c"}])


# --------------------------------------------------------------------------- #
# Per-checkpoint ingest assertion                                             #
# --------------------------------------------------------------------------- #


def test_checkpoint_base_model_allows_commercial_bases():
    for base in ("qwen2.5-1.5b", "meta-llama/Llama-3.2-1B", "smollm2-1.7b", "phi-3.5-mini"):
        rec = assert_checkpoint_license(base, role="base_model")
        assert rec is not None


def test_checkpoint_base_model_bars_noncommercial():
    with pytest.raises(LicenseGuardError, match="non-commercial"):
        assert_checkpoint_license("qwen2.5-3b", role="base_model")
    with pytest.raises(LicenseGuardError, match="non-commercial"):
        assert_checkpoint_license("mistral-large", role="base_model")


def test_checkpoint_unregistered_raises():
    with pytest.raises(LicenseGuardError, match="UNREGISTERED"):
        assert_checkpoint_license("nonexistent-model", role="base_model")


def test_checkpoint_teacher_role_bars_barred_verdict():
    # Llama is commercial-permitted as a base but BARRED as a teacher.
    with pytest.raises(LicenseGuardError, match="BARRED as a teacher"):
        assert_checkpoint_license("meta-llama/Llama-3.3-70B", role="teacher")
    # Nemotron is a safe teacher.
    assert assert_checkpoint_license("nemotron-3-super-120b", role="teacher") is not None


# --------------------------------------------------------------------------- #
# Nemotron license-pin guard                                                   #
# --------------------------------------------------------------------------- #


def test_nemotron_pin_default_roster_passes():
    assert assert_nemotron_pin() is None


def test_nemotron_pin_roster_identity_matches_constant():
    rec = TEACHER_ROSTER[NEMOTRON_ROSTER_KEY]
    assert rec.identity == NEMOTRON_PINNED_LICENSE_IDENTITY


def test_nemotron_pin_general_oml_repin_fails_build():
    with pytest.raises(LicenseGuardError, match="license pin violation"):
        assert_nemotron_pin(GENERAL_NVIDIA_OML_IDENTITY)


def test_nemotron_pin_arbitrary_drift_fails():
    with pytest.raises(LicenseGuardError):
        assert_nemotron_pin("Some Other License v2")


def test_nemotron_pin_guard_catches_roster_repin(monkeypatch):
    # Simulate an operator re-pinning the roster's Nemotron record to the
    # general NVIDIA OML — the build must fail.
    drifted = LicenseRecord(
        NEMOTRON_ROSTER_KEY,
        "LicenseRef-NVIDIA-OML",
        "https://example",
        "safe",
        license_identity=GENERAL_NVIDIA_OML_IDENTITY,
    )
    patched = dict(TEACHER_ROSTER)
    patched[NEMOTRON_ROSTER_KEY] = drifted
    monkeypatch.setattr("lib.licensing.teacher_roster.TEACHER_ROSTER", patched)
    with pytest.raises(LicenseGuardError, match="pin violation"):
        assert_nemotron_pin()
