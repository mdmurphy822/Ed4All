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
import re
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


# ---------------------------------------------------------------------- #
# Wave 96 — vocabulary_ttl_hash falls back to project-root canonical copy #
# ---------------------------------------------------------------------- #


SHA256_EMPTY = hashlib.sha256(b"").hexdigest()


def _build_libv2_course_no_vocab(tmp_path: Path, slug: str = "tst-101-novocab") -> Path:
    """Same fixture as ``_build_libv2_course`` but without graph/*.vocabulary.ttl.

    Mirrors production LibV2 courses (e.g. rdf-shacl-551-2) where no
    course-local vocab file is materialized — the canonical TTL lives at
    project-root ``schemas/context/courseforge_v1.vocabulary.ttl`` and
    Wave 96 wires it as a fallback so the model card pins a non-empty
    SHA-256.
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
    # NOTE: no graph/courseforge_v1.vocabulary.ttl on purpose.
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


def test_vocabulary_ttl_hash_is_non_empty_in_dry_run(tmp_path: Path):
    """The canonical project-root vocabulary.ttl must be hashed when no
    course-local copy exists (Wave 96 fix).

    Pre-Wave-96 the runner logged a "substituting empty-bytes sha256"
    warning and emitted the empty-bytes SHA-256 for vocabulary_ttl_hash,
    even though the canonical TTL was sitting in
    ``schemas/context/courseforge_v1.vocabulary.ttl``. The model card
    then claimed an unhashed vocab and broke replayability.
    """
    canonical_ttl = (
        Path(__file__).resolve().parents[2]
        / "schemas" / "context" / "courseforge_v1.vocabulary.ttl"
    )
    assert canonical_ttl.exists(), (
        "Wave 96 fix relies on the canonical vocabulary.ttl being "
        f"present at {canonical_ttl}. If this file moved, update the "
        "_VOCABULARY_TTL_CANONICAL constant in Trainforge/training/runner.py."
    )
    expected_hash = _sha256_path(canonical_ttl)
    assert expected_hash != SHA256_EMPTY, (
        "Canonical vocabulary.ttl is empty — fixture is broken."
    )

    libv2_root = _build_libv2_course_no_vocab(tmp_path)
    runner = TrainingRunner(
        course_slug="tst-101-novocab",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=True,
    )
    result = runner.run()
    card = json.loads(result.model_card_path.read_text(encoding="utf-8"))

    vocab_hash = card["provenance"]["vocabulary_ttl_hash"]
    assert vocab_hash != SHA256_EMPTY, (
        "vocabulary_ttl_hash is the empty-bytes sha256 — Wave 96 fallback "
        "didn't fire."
    )
    assert vocab_hash == expected_hash, (
        f"vocabulary_ttl_hash {vocab_hash!r} doesn't match canonical "
        f"project-root TTL {expected_hash!r}."
    )


# ---------------------------------------------------------------------- #
# Wave 96 — decision rationales clear the lib.decision_capture quality   #
# gate (no 'developing' ratings)                                          #
# ---------------------------------------------------------------------- #


def test_decision_rationales_pass_quality_gate(libv2_root: Path):
    """Each of the four mandatory training decisions must clear the
    ``lib.decision_capture`` quality gate (proficient or better).

    The gate logic lives in ``lib/decision_capture.py::_build_record``
    via ``lib/quality.py::assess_decision_quality`` — proficient
    requires rationale ≥50 chars AND (inputs_ref OR alternatives).
    Wave 96 fix added alternatives_considered to the three failing
    decisions and richer dynamic-signal interpolation. This test
    proxies the gate by asserting:

      1. Every rationale interpolates ≥1 numeric value (a regex match
         on ``\\d+``) — proves dynamic interpolation per project policy.
      2. Every rationale references at least one identifier from the
         run (course slug, base model, or HF repo).
      3. Every record has alternatives_considered or inputs_ref to
         reach 'proficient' on the centralized gate.
      4. The metadata.quality_gate_passed flag is True for all four.
    """
    runner = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=True,
    )
    result = runner.run()

    records: List[Dict[str, Any]] = []
    with result.decision_capture_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    required_types = {
        "training_run_planning",
        "base_model_selection",
        "hyperparameter_selection",
        "eval_run_decision",
    }
    by_type = {r["decision_type"]: r for r in records}
    missing = required_types - by_type.keys()
    assert not missing, f"Missing decision types: {missing}"

    base_model_short = "qwen2.5-1.5b"
    course_slug = "tst-101"
    hf_repo_fragment = "Qwen"  # canonical for qwen2.5-1.5b

    for dt in required_types:
        rec = by_type[dt]
        rationale = rec.get("rationale") or ""

        # 1. ≥1 numeric value (proves dynamic-signal interpolation).
        assert re.search(r"\d+", rationale), (
            f"{dt}: rationale lacks any numeric signal. Rationale={rationale!r}"
        )

        # 2. ≥1 run-specific identifier.
        identifiers_present = sum(
            1 for ident in (course_slug, base_model_short, hf_repo_fragment)
            if ident in rationale
        )
        assert identifiers_present >= 1, (
            f"{dt}: rationale doesn't reference any run identifier "
            f"(slug={course_slug!r}, base={base_model_short!r}, "
            f"hf_repo_fragment={hf_repo_fragment!r}). Rationale={rationale!r}"
        )

        # 3. alternatives_considered or inputs_ref present.
        has_alts = bool(rec.get("alternatives_considered"))
        has_inputs = bool(rec.get("inputs_ref"))
        assert has_alts or has_inputs, (
            f"{dt}: neither alternatives_considered nor inputs_ref is "
            f"populated; the centralized quality gate will rate this "
            f"'developing'."
        )

        # 4. Quality gate flag present + true.
        meta = rec.get("metadata") or {}
        assert meta.get("quality_gate_passed") is True, (
            f"{dt}: metadata.quality_gate_passed is "
            f"{meta.get('quality_gate_passed')!r} "
            f"(quality_level={meta.get('quality_level')!r}, "
            f"reason={meta.get('quality_gate_reason')!r}). Rationale="
            f"{rationale!r}"
        )


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


# ---------------------------------------------------------------------- #
# Wave 100 — runner emits model_card + training_run.jsonl when eval is   #
# unwired (Bug 5)                                                         #
# ---------------------------------------------------------------------- #


def test_runner_writes_card_when_eval_unwired(libv2_root: Path, monkeypatch):
    """Bug 5: training success ≠ eval success. The runner must write
    model_card.json + training_run.jsonl AFTER training succeeds but
    BEFORE attempting eval, so a failure from ``_run_eval_harness``
    doesn't void the provenance card and decision log.

    Wave 101: the eval bridge is now wired (no longer raises
    NotImplementedError unconditionally), but it can still fail on a
    CPU-only test machine (no transformers / peft installed) or when
    the adapter dir is missing fragments. The runner's broadened
    except clause catches NotImplementedError, ImportError, and
    FileNotFoundError so the no-eval-scores fallback still fires.
    """
    from Trainforge.training.compute_backend import TrainingJobResult

    runner = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=False,  # IMPORTANT: real-run path, not dry-run
    )

    # Stub _dispatch_training to "succeed" without invoking the GPU
    # backend. We materialise the adapter file on disk so the
    # adapter-presence guard doesn't trip.
    def _fake_dispatch(run_dir: Path, run_dpo: bool) -> TrainingJobResult:
        adapter = run_dir / "adapter_model.safetensors"
        adapter.write_bytes(b"stub-trained-bytes")
        return TrainingJobResult(
            adapter_path=adapter,
            metrics={"backend": "stub", "final_train_loss": 1.37},
        )

    monkeypatch.setattr(runner, "_dispatch_training", _fake_dispatch)

    # Wave 101: force the eval bridge to fail with NotImplementedError
    # so the test exercises the runner's fallback path regardless of
    # whether the test box has transformers/peft installed. In real
    # CPU-only runs, ImportError gets caught the same way.
    def _eval_unwired(run_dir: Path, adapter_path):
        raise NotImplementedError("test-stub: eval bridge skipped")

    monkeypatch.setattr(runner, "_run_eval_harness", _eval_unwired)

    result = runner.run()

    # 1. Adapter persisted.
    assert result.adapter_path is not None
    assert result.adapter_path.exists()

    # 2. model_card.json on disk (the central Bug 5 assertion).
    assert result.model_card_path.exists(), (
        "model_card.json must be written even when eval is unwired."
    )
    card = json.loads(result.model_card_path.read_text(encoding="utf-8"))
    # eval_scores is absent (or empty) since the harness was skipped.
    assert "eval_scores" not in card or not card["eval_scores"], (
        "Eval was unwired, so eval_scores must be absent from the card."
    )

    # 3. training_run.jsonl on disk with the 4 required decisions.
    assert result.decision_capture_path.exists(), (
        "training_run.jsonl must be written even when eval is unwired."
    )
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
        f"Missing required decision types: {required - decision_types}. "
        f"Got: {sorted(decision_types)}"
    )


# ---------------------------------------------------------------------- #
# Wave 100 — fit_sft returns the actual TRL-written filename (Bug 2)     #
# ---------------------------------------------------------------------- #


def test_fit_sft_adapter_filename(tmp_path: Path, monkeypatch):
    """Bug 2: TRL's ``save_model()`` writes
    ``adapter_model.safetensors`` (with underscore), not
    ``adapter.safetensors``. ``fit_sft`` must return the actual on-disk
    path so the runner's adapter-presence guard finds the file.
    """
    pytest.importorskip("trl", reason="Bug 2 test exercises TRL save path")
    pytest.importorskip("peft", reason="fit_sft requires peft")

    # Stub the heavy ML imports so we don't actually load a 1.5B model
    # weight set on a CPU-only CI box. The test only cares about the
    # filename round-trip: we install fakes for SFTTrainer +
    # AutoTokenizer + LoraConfig + Dataset + torch + bitsandbytes that
    # let fit_sft execute the save_model side-effect without touching
    # GPU/network.
    from Trainforge.training.peft_trainer import PEFTTrainer

    output_dir = tmp_path / "stub_run"
    output_dir.mkdir()

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self):
            pass

        def save_model(self, path: str):
            # TRL writes the file with the underscore form.
            (Path(path) / "adapter_model.safetensors").write_bytes(b"x")

    class _FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _FakeBitsAndBytesConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeDataset:
        @classmethod
        def from_dict(cls, mapping):
            return cls()

    class _FakeLoraConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeSFTConfig:
        def __init__(self, *args, **kwargs):
            pass

    # Bypass _require_training_deps so the test runs even without
    # bitsandbytes installed (CI may not have CUDA wheels).
    monkeypatch.setattr(
        "Trainforge.training.peft_trainer._require_training_deps",
        lambda: None,
    )

    # Patch the modules pulled in by fit_sft. Use sys.modules patching
    # because fit_sft does ``from peft import LoraConfig`` etc. at call
    # time.
    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: False,
    )
    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = _FakeLoraConfig
    fake_peft.prepare_model_for_kbit_training = lambda model: model
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _FakeModel
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.BitsAndBytesConfig = _FakeBitsAndBytesConfig
    fake_trl = types.ModuleType("trl")
    fake_trl.SFTTrainer = _FakeTrainer
    fake_trl.SFTConfig = _FakeSFTConfig
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _FakeDataset

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={"epochs": 1, "batch_size": 1},
    )
    pairs = [{"prompt": "Q?", "completion": "A.", "chunk_id": "c1"}]
    returned = trainer.fit_sft(pairs, output_dir)

    # The returned path must equal the actual file TRL writes.
    expected = output_dir / "adapter_model.safetensors"
    assert returned == expected, (
        f"fit_sft returned {returned!r} but TRL writes {expected!r}. "
        f"This drift would trip the runner's adapter-presence guard."
    )
    assert expected.exists(), (
        "Stub TRL save_model didn't materialise the expected file."
    )


# ---------------------------------------------------------------------- #
# Wave 100 — Wave 96 quality_gate_passed assertion against ON-DISK JSONL #
# (Bug 4 flavour)                                                        #
# ---------------------------------------------------------------------- #


def test_decision_rationales_pass_quality_gate_on_disk(libv2_root: Path):
    """Variant of ``test_decision_rationales_pass_quality_gate`` that
    explicitly reads from ``training_run.jsonl`` on disk (not the
    in-memory ``capture.decisions`` list) — proves the on-disk emit
    path carries the same ``metadata.quality_gate_passed=True`` flag
    Wave 96 asserted against in-memory.

    Bug 4 of Wave 100: the Wave 99 worker reported finding
    ``quality_gate_passed=None`` on disk; the actual root cause was a
    pre-Wave-96 build that emitted thinner rationales (``developing``
    quality, gate=False). This test pins the contract that
    ``training_run.jsonl`` records the True flag for all four mandatory
    decisions in the current code tree.
    """
    runner = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=True,
    )
    result = runner.run()

    # ON-DISK read — this is the surface real consumers see.
    on_disk_path = result.decision_capture_path
    assert on_disk_path.exists()
    assert on_disk_path.name == "training_run.jsonl"

    records: List[Dict[str, Any]] = []
    with on_disk_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    required_types = {
        "training_run_planning",
        "base_model_selection",
        "hyperparameter_selection",
        "eval_run_decision",
    }
    by_type = {r["decision_type"]: r for r in records}
    missing = required_types - by_type.keys()
    assert not missing, f"Missing decision types on disk: {missing}"

    for dt in required_types:
        rec = by_type[dt]
        meta = rec.get("metadata") or {}
        assert meta.get("quality_gate_passed") is True, (
            f"{dt} (on disk): metadata.quality_gate_passed is "
            f"{meta.get('quality_gate_passed')!r} "
            f"(quality_level={meta.get('quality_level')!r}, "
            f"reason={meta.get('quality_gate_reason')!r})."
        )


# ---------------------------------------------------------------------- #
# Wave 101 — eval bridge happy path with fully mocked harness            #
# ---------------------------------------------------------------------- #


def test_runner_eval_bridge_wired_in_dry_run(libv2_root: Path, monkeypatch):
    """Wave 101 happy path: when _run_eval_harness returns a real
    eval_scores dict, the runner folds it into a SECOND model_card.json
    write so the card on disk carries the canonical eval keys."""
    from Trainforge.training.compute_backend import TrainingJobResult

    runner = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=False,
    )

    def _fake_dispatch(run_dir: Path, run_dpo: bool) -> TrainingJobResult:
        adapter = run_dir / "adapter_model.safetensors"
        adapter.write_bytes(b"stub-trained-bytes")
        return TrainingJobResult(
            adapter_path=adapter,
            metrics={"backend": "stub", "final_train_loss": 0.42},
        )

    monkeypatch.setattr(runner, "_dispatch_training", _fake_dispatch)

    # Mock the eval bridge to return canonical scores AND drop a
    # gate-shaped eval_report.json on disk so the inline
    # EvalGatingValidator (audit 2026-04-30 fix) finds something and
    # passes. Real eval harness always writes this file; the test
    # mocks both surfaces.
    def _fake_eval(run_dir: Path, adapter_path):
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval_report.json").write_text(
            json.dumps({
                "faithfulness": 0.75,
                "coverage": 0.62,
                "baseline_delta": 0.13,
                "yes_rate": 0.30,
                "metrics": {"hallucination_rate": 0.25},
            }),
            encoding="utf-8",
        )
        return {
            "faithfulness": 0.75,
            "coverage": 0.62,
            "baseline_delta": 0.13,
        }

    monkeypatch.setattr(runner, "_run_eval_harness", _fake_eval)
    # Skip the ablation runner — torch unavailable on CPU-only CI.
    monkeypatch.setattr(runner, "_run_ablation", lambda **kw: None)

    result = runner.run()

    # Card on disk carries the eval scores (filtered to canonical
    # keys per the schema's additionalProperties=false guard).
    assert result.model_card_path.exists()
    card = json.loads(result.model_card_path.read_text(encoding="utf-8"))
    assert "eval_scores" in card
    assert card["eval_scores"]["faithfulness"] == 0.75
    assert card["eval_scores"]["coverage"] == 0.62
    assert card["eval_scores"]["baseline_delta"] == 0.13


def test_runner_eval_bridge_uses_eval_config_generation_settings(
    libv2_root: Path,
    tmp_path: Path,
    monkeypatch,
):
    """The adapter eval bridge must use the locked per-course
    eval_config.yaml instead of silently relying on AdapterCallable
    constructor defaults.
    """
    course_dir = libv2_root / "tst-101"
    eval_dir = course_dir / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "prompt_template.txt").write_text(
        "{context_section}\n{question}\n",
        encoding="utf-8",
    )
    (eval_dir / "rubric.md").write_text("# rubric\n", encoding="utf-8")
    (eval_dir / "eval_config.yaml").write_text(
        "\n".join([
            "benchmark: ED4ALL-Bench",
            "benchmark_version: '1.0'",
            "top_k: 5",
            "temperature: 0.7",
            "top_p: 0.9",
            "max_new_tokens: 64",
            "seed: 123",
            "prompt_template_file: prompt_template.txt",
            "rubric_file: rubric.md",
        ]) + "\n",
        encoding="utf-8",
    )

    runner = TrainingRunner(
        course_slug="tst-101",
        base_model="qwen2.5-1.5b",
        libv2_root=libv2_root,
        dry_run=False,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    adapter_path = run_dir / "adapter_model.safetensors"
    adapter_path.write_bytes(b"adapter")
    (run_dir / "model_card.json").write_text("{}", encoding="utf-8")

    captured: Dict[str, Any] = {}

    class _FakeAdapterCallable:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, prompt: str) -> str:
            return "yes"

    class _FakeHarness:
        def __init__(self, *, course_path, model_callable):
            self.course_path = course_path
            self.model_callable = model_callable

        def run_all(self, *, output_path):
            Path(output_path).write_text(
                json.dumps({
                    "faithfulness": 0.5,
                    "coverage": 0.25,
                    "baseline_delta": 0.1,
                }),
                encoding="utf-8",
            )
            return Path(output_path)

    adapter_module = importlib.import_module("Trainforge.eval.adapter_callable")
    harness_module = importlib.import_module("Trainforge.eval.slm_eval_harness")
    hf_index_module = importlib.import_module("Trainforge.eval.hf_model_index")
    monkeypatch.setattr(adapter_module, "AdapterCallable", _FakeAdapterCallable)
    monkeypatch.setattr(harness_module, "SLMEvalHarness", _FakeHarness)
    monkeypatch.setattr(hf_index_module, "write_hf_readme", lambda **kwargs: None)

    scores = runner._run_eval_harness(run_dir, adapter_path)

    assert scores["faithfulness"] == 0.5
    assert captured["max_new_tokens"] == 64
    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.9
    assert captured["seed"] == 123
    assert captured["revision"] == runner.spec.default_revision
