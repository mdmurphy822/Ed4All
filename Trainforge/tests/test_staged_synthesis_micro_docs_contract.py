"""Documentation and routing contract for staged synthesis micro-v1.

No provider method is called here: provider construction is sufficient to
prove that the default path remains the unwrapped legacy implementation.
"""
from __future__ import annotations

from pathlib import Path

from Trainforge.generators.providers._synthesis_provider import (
    SynthesisProvider,
    build_synthesis_provider,
)
from Trainforge.generators.staged.micro import (
    MICRO_CONTRACT_VERSION,
    micro_contract_components,
)
from Trainforge.generators.staged.provider import (
    StagedSynthesisProvider,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINFORGE_GUIDE = ROOT / "Trainforge" / "CLAUDE.md"
LICENSING_GUIDE = ROOT / "docs" / "LICENSING.md"


def test_micro_contract_path_and_cli_are_documented_in_owning_guides():
    trainforge = TRAINFORGE_GUIDE.read_text(encoding="utf-8")
    licensing = LICENSING_GUIDE.read_text(encoding="utf-8")

    assert MICRO_CONTRACT_VERSION == "ed4all.staged-synthesis-micro.v1"
    assert MICRO_CONTRACT_VERSION in trainforge
    assert MICRO_CONTRACT_VERSION in licensing
    assert "--synthesis-contract micro-v1" in trainforge
    assert "--synthesis-contract micro-v1" in licensing
    assert "TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1" in trainforge
    assert "TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1" in licensing


def test_documented_micro_contract_preserves_thresholds_and_operational_seams():
    components = micro_contract_components()
    trainforge = TRAINFORGE_GUIDE.read_text(encoding="utf-8")

    assert components["entailment_floor"] == 0.70
    assert components["contradiction_ceiling"] == 0.50
    for phrase in (
        "entailment `0.70`",
        "contradiction `0.50`",
        "synthesis_contract_conflict",
        "committed_complete",
        "terminal_hold",
        "stop sentinel",
        "resume sidecar",
        "gates remain unchanged",
        "byte-identical legacy route",
    ):
        assert phrase in trainforge


def test_flag_off_constructs_unwrapped_legacy_provider(monkeypatch):
    monkeypatch.delenv("TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1", raising=False)
    monkeypatch.delenv("TRAINFORGE_STAGED_SYNTHESIS_V4", raising=False)

    provider = build_synthesis_provider("local")

    assert type(provider) is SynthesisProvider
    assert not isinstance(provider, StagedSynthesisProvider)


def test_micro_docs_contain_no_course_or_device_specific_data():
    documented_rows = "\n".join(
        line
        for path in (TRAINFORGE_GUIDE, LICENSING_GUIDE)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "STAGED_SYNTHESIS_MICRO_V1" in line
        or "synthesis-contract micro-v1" in line
    )
    # Assemble sentinels so the sanitation test does not itself introduce the
    # machine/course tokens it is charged with detecting.
    forbidden = tuple(
        "".join(parts)
        for parts in (
            ("/", "home", "/"),
            ("/", "Users", "/"),
            ("127", ".0.0.1"),
            ("192", ".168."),
            ("W", "F-"),
            ("open", "stax"),
        )
    )
    assert not any(token in documented_rows for token in forbidden)
