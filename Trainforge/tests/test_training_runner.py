"""Wave 90 — TrainingRunner + base-model registry + config loader tests.

All tests are dry-run / CPU-only. No GPU required, no heavy ML
dependencies imported. The Wave 89 → Wave 90 contract test
(``test_dry_run_emits_valid_model_card``) asserts the runner's emitted
``model_card.json`` validates against
:class:`lib.validators.libv2_model.LibV2ModelValidator`.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.training import (  # noqa: E402
    BaseModelRegistry,
    LocalBackend,
    RunPodBackend,
    TrainingConfig,
    TrainingRunner,
    format_instruction,
    load_config,
)
from Trainforge.training.compute_backend import TrainingJobSpec  # noqa: E402
from lib.validators.libv2_model import LibV2ModelValidator  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixtures                                                                #
# ---------------------------------------------------------------------- #


def _build_libv2_course(tmp_path: Path, slug: str = "tst-101") -> Path:
    """Synthesize a minimal LibV2 course tree under tmp_path/courses/<slug>.

    Layout matches the v0.3.0 archive:

        courses/<slug>/
          corpus/chunks.jsonl
          graph/pedagogy_graph.json
          graph/concept_graph_semantic.json
          graph/courseforge_v1.vocabulary.ttl
          training_specs/instruction_pairs.jsonl
          training_specs/preference_pairs.jsonl
          training_specs/dataset_config.json
    """
    libv2_root = tmp_path / "courses"
    course_dir = libv2_root / slug
    (course_dir / "corpus").mkdir(parents=True)
    (course_dir / "graph").mkdir(parents=True)
    (course_dir / "training_specs").mkdir(parents=True)

    (course_dir / "corpus" / "chunks.jsonl").write_text(
        '{"id": "c1", "text": "fixture", "learning_outcome_refs": ["TO-01"]}\n',
        encoding="utf-8",
    )
    (course_dir / "graph" / "pedagogy_graph.json").write_text(
        '{"nodes": [], "edges": []}',
        encoding="utf-8",
    )
    (course_dir / "graph" / "concept_graph_semantic.json").write_text(
        '{"concepts": []}',
        encoding="utf-8",
    )
    (course_dir / "graph" / "courseforge_v1.vocabulary.ttl").write_text(
        "@prefix : <http://example.com/> .\n",
        encoding="utf-8",
    )
    (course_dir / "training_specs" / "instruction_pairs.jsonl").write_text(
        json.dumps({
            "prompt": "Define the central concept of fixtures.",
            "completion": "A fixture is a small reusable test setup.",
            "chunk_id": "c1",
        }) + "\n",
        encoding="utf-8",
    )
    (course_dir / "training_specs" / "preference_pairs.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    (course_dir / "training_specs" / "dataset_config.json").write_text(
        '{"format": "instruction-following", "statistics": {}}',
        encoding="utf-8",
    )
    return libv2_root


@pytest.fixture
def libv2_root(tmp_path: Path) -> Path:
    return _build_libv2_course(tmp_path)


# ---------------------------------------------------------------------- #
# 1. Dry-run produces a valid model card                                  #
# ---------------------------------------------------------------------- #


def test_dry_run_emits_valid_model_card(libv2_root: Path):
    runner = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=True,
    )
    result = runner.run()

    # Card on disk
    assert result.model_card_path.exists()
    card = json.loads(result.model_card_path.read_text(encoding="utf-8"))

    # Wave 89 → Wave 90 contract: card validates against the
    # LibV2ModelValidator with no critical issues (warnings only).
    # We synthesize a stub adapter file so the weights-presence
    # critical check doesn't trip.
    (result.run_dir / "adapter.safetensors").write_bytes(b"stub-adapter-bytes")

    validator = LibV2ModelValidator()
    gate_result = validator.validate({
        "model_card_path": str(result.model_card_path),
        "model_dir": str(result.run_dir),
        "course_dir": str(libv2_root / "tst-101"),
    })
    critical = [i for i in gate_result.issues if i.severity == "critical"]
    assert not critical, (
        f"Expected zero critical issues from emitted card; got: "
        f"{[(i.code, i.message) for i in critical]}"
    )

    # Sanity: card top-level keys.
    for required in (
        "model_id", "course_slug", "base_model", "adapter_format",
        "training_config", "provenance", "created_at",
    ):
        assert required in card


# ---------------------------------------------------------------------- #
# 2. Provenance hashes match expected SHA-256 over the source files       #
# ---------------------------------------------------------------------- #


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_provenance_hashes_match_source_files(libv2_root: Path):
    runner = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=True,
    )
    result = runner.run()
    card = json.loads(result.model_card_path.read_text(encoding="utf-8"))

    course_dir = libv2_root / "tst-101"
    expected = {
        "chunks_hash": _sha256_path(course_dir / "corpus" / "chunks.jsonl"),
        "pedagogy_graph_hash": _sha256_path(
            course_dir / "graph" / "pedagogy_graph.json"
        ),
        "instruction_pairs_hash": _sha256_path(
            course_dir / "training_specs" / "instruction_pairs.jsonl"
        ),
        "preference_pairs_hash": _sha256_path(
            course_dir / "training_specs" / "preference_pairs.jsonl"
        ),
        "concept_graph_hash": _sha256_path(
            course_dir / "graph" / "concept_graph_semantic.json"
        ),
        "vocabulary_ttl_hash": _sha256_path(
            course_dir / "graph" / "courseforge_v1.vocabulary.ttl"
        ),
        # Wave 92: holdout split is optional at runtime; absent in this
        # fixture so the runner substitutes the empty-bytes sha256.
        "holdout_graph_hash": hashlib.sha256(b"").hexdigest(),
    }
    assert card["provenance"] == expected


# ---------------------------------------------------------------------- #
# 3. DecisionCapture fires the 4 required event types                     #
# ---------------------------------------------------------------------- #


def test_decision_capture_fires_required_events(libv2_root: Path):
    runner = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=True,
    )
    result = runner.run()
    assert result.decision_capture_path.exists()
    records: List[Dict[str, Any]] = []
    with result.decision_capture_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    decision_types = {rec["decision_type"] for rec in records}
    required = {
        "training_run_planning",
        "base_model_selection",
        "hyperparameter_selection",
        "eval_run_decision",
    }
    assert required.issubset(decision_types), (
        f"Missing required decision_type values: {required - decision_types}. "
        f"Got: {sorted(decision_types)}"
    )

    # Every rationale must be ≥20 chars (project-wide quality bar).
    for rec in records:
        rationale = rec.get("rationale") or ""
        assert len(rationale) >= 20, (
            f"Rationale too short on {rec['decision_type']}: {rationale!r}"
        )


# ---------------------------------------------------------------------- #
# 4. BaseModelRegistry resolves all 5 supported names                     #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [
    "qwen2.5-1.5b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "smollm2-1.7b",
    "phi-3.5-mini",
])
def test_base_model_registry_resolves(name: str):
    spec = BaseModelRegistry.resolve(name)
    assert spec.name == name
    assert "/" in spec.huggingface_repo
    assert spec.chat_template in {"chatml", "llama3", "phi3"}
    assert spec.recommended_max_seq_length > 0
    assert spec.recommended_lora_rank > 0


def test_base_model_registry_unknown_raises():
    with pytest.raises(KeyError):
        BaseModelRegistry.resolve("not-a-real-model")


def test_list_supported_returns_at_least_5():
    supported = BaseModelRegistry.list_supported()
    assert len(supported) >= 5
    assert "qwen2.5-1.5b" in supported


# ---------------------------------------------------------------------- #
# 5. format_instruction templates render correctly                        #
# ---------------------------------------------------------------------- #


def test_format_instruction_chatml():
    spec = BaseModelRegistry.resolve("qwen2.5-1.5b")
    out = format_instruction(spec, {"prompt": "Q?", "completion": "A."})
    assert "<|im_start|>user" in out
    assert "<|im_end|>" in out
    assert "<|im_start|>assistant" in out
    assert "Q?" in out and "A." in out


def test_format_instruction_llama3():
    spec = BaseModelRegistry.resolve("llama-3.2-1b")
    out = format_instruction(spec, {"prompt": "Q?", "completion": "A."})
    assert "<|begin_of_text|>" in out
    assert "<|start_header_id|>user<|end_header_id|>" in out
    assert "<|eot_id|>" in out


def test_format_instruction_phi3():
    spec = BaseModelRegistry.resolve("phi-3.5-mini")
    out = format_instruction(spec, {"prompt": "Q?", "completion": "A."})
    assert "<|user|>" in out
    assert "<|assistant|>" in out
    assert "<|end|>" in out


def test_format_instruction_missing_keys():
    spec = BaseModelRegistry.resolve("qwen2.5-1.5b")
    with pytest.raises(KeyError):
        format_instruction(spec, {"prompt": "no completion"})


# ---------------------------------------------------------------------- #
# 6. load_config merges base YAML with course overrides                   #
# ---------------------------------------------------------------------- #


def test_load_config_qwen_defaults():
    cfg = load_config("qwen2.5-1.5b")
    assert isinstance(cfg, TrainingConfig)
    assert cfg.base_model == "qwen2.5-1.5b"
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32


def test_load_config_with_overrides(tmp_path: Path):
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        "learning_rate: 5.0e-5\nepochs: 1\n",
        encoding="utf-8",
    )
    cfg = load_config("qwen2.5-1.5b", course_overrides=overrides)
    assert cfg.learning_rate == 5e-5
    assert cfg.epochs == 1
    # Untouched fields stay at the per-base default.
    assert cfg.lora_rank == 16


def test_load_config_unknown_override_key_raises(tmp_path: Path):
    overrides = tmp_path / "bad.yaml"
    overrides.write_text("totally_made_up_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config("qwen2.5-1.5b", course_overrides=overrides)


def test_load_config_unknown_base_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nope-1.0b")


# ---------------------------------------------------------------------- #
# 7. LocalBackend without a GPU raises a clear error                      #
# ---------------------------------------------------------------------- #


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(
    _torch_cuda_available(),
    reason="GPU is present; the no-GPU error path can't be reached here.",
)
def test_local_backend_raises_without_gpu(libv2_root: Path):
    backend = LocalBackend(allow_no_gpu=False)
    spec = TrainingJobSpec(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        instruction_pairs_path=(
            libv2_root / "tst-101" / "training_specs" / "instruction_pairs.jsonl"
        ),
        preference_pairs_path=(
            libv2_root / "tst-101" / "training_specs" / "preference_pairs.jsonl"
        ),
        training_config={},
        output_dir=libv2_root / "tst-101" / "models" / "stub",
    )
    with pytest.raises(RuntimeError) as excinfo:
        backend.run(spec)
    assert "GPU" in str(excinfo.value) or "training" in str(excinfo.value)


# ---------------------------------------------------------------------- #
# 8. RunPodBackend stub raises NotImplementedError                        #
# ---------------------------------------------------------------------- #


def test_runpod_backend_raises_not_implemented(libv2_root: Path):
    backend = RunPodBackend()
    spec = TrainingJobSpec(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        instruction_pairs_path=Path("/dev/null"),
        preference_pairs_path=Path("/dev/null"),
        training_config={},
        output_dir=Path("/dev/null"),
    )
    with pytest.raises(NotImplementedError) as excinfo:
        backend.run(spec)
    assert "Wave 90" in str(excinfo.value) or "stub" in str(excinfo.value).lower()


# ---------------------------------------------------------------------- #
# Bonus: model_id is stable across re-runs over identical input           #
# ---------------------------------------------------------------------- #


def test_model_id_stable_across_reruns(libv2_root: Path):
    r1 = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=True,
    ).run()
    r2 = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=True,
    ).run()
    assert r1.model_id == r2.model_id


def test_missing_training_specs_fails_loud(tmp_path: Path):
    libv2_root = tmp_path / "courses"
    course = libv2_root / "tst-101"
    (course / "corpus").mkdir(parents=True)
    (course / "graph").mkdir(parents=True)
    (course / "corpus" / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (course / "graph" / "pedagogy_graph.json").write_text("{}", encoding="utf-8")
    # Note: NO training_specs/ dir.
    with pytest.raises(FileNotFoundError) as excinfo:
        TrainingRunner(
            course_slug="tst-101",
            base_model="qwen2.5-1.5b",
            libv2_root=libv2_root,
            dry_run=True,
        ).run()
    assert "training" in str(excinfo.value).lower()
