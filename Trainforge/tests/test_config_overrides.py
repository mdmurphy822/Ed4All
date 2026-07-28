"""Regression — the canonical ``--config-overrides`` parser + its consumers.

``Trainforge/training/configs/__init__.py`` owns the ONE definition of what a
legal per-run TrainingConfig override is. The CLI validates through it at
parse time, ``MCP/tools/pipeline_tools.py::_run_training`` re-normalizes
through it at dispatch, and ``TrainingRunner`` coerces through it again — so
the three can never disagree about a key or a range.

The contract these tests pin:

* unknown keys raise, naming the supported set (never dropped, never a new
  attribute on the dataclass);
* values are coerced to the field's declared type and range-checked at PARSE
  time, not six hours into a run;
* ``base_model`` is locked (it is owned by ``--base-model``, and letting an
  override rewrite it would desync the model card from the loaded weights);
* no overrides => the resolved config and the emitted model card are
  byte-identical to before the feature existed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.training.configs import (  # noqa: E402
    ConfigOverrideError,
    TrainingConfig,
    coerce_config_overrides,
    load_config,
    parse_config_overrides,
)


# --------------------------------------------------------------------- #
# Parsing                                                               #
# --------------------------------------------------------------------- #


def test_empty_specs_parse_to_no_overrides() -> None:
    for spec in (None, "", "   ", {}):
        assert parse_config_overrides(spec) == {}


def test_inline_key_value_pairs() -> None:
    assert parse_config_overrides("dpo_learning_rate=5e-7,epochs=2") == {
        "dpo_learning_rate": 5e-7,
        "epochs": 2,
    }


def test_inline_json_object() -> None:
    assert parse_config_overrides('{"epochs": 2, "use_4bit": false}') == {
        "epochs": 2,
        "use_4bit": False,
    }


def test_yaml_file_path(tmp_path: Path) -> None:
    path = tmp_path / "overrides.yaml"
    path.write_text("dpo_learning_rate: 5.0e-7\nepochs: 2\n", encoding="utf-8")
    assert parse_config_overrides(str(path)) == {
        "dpo_learning_rate": 5e-7,
        "epochs": 2,
    }
    assert parse_config_overrides(path) == {
        "dpo_learning_rate": 5e-7,
        "epochs": 2,
    }


def test_a_mistyped_path_raises_rather_than_meaning_no_overrides() -> None:
    with pytest.raises(ConfigOverrideError, match="neither an existing file"):
        parse_config_overrides("configs/typo.yaml")


def test_list_field_uses_pipe_inline_and_a_real_list_in_a_file(
    tmp_path: Path,
) -> None:
    assert parse_config_overrides("target_modules=q_proj|k_proj") == {
        "target_modules": ["q_proj", "k_proj"],
    }
    path = tmp_path / "o.yaml"
    path.write_text("target_modules:\n  - q_proj\n  - k_proj\n", encoding="utf-8")
    assert parse_config_overrides(path) == {
        "target_modules": ["q_proj", "k_proj"],
    }


def test_parsing_is_idempotent() -> None:
    """The CLI validates, then the handler re-validates the same value."""
    once = parse_config_overrides("dpo_learning_rate=5e-7,epochs=2")
    assert parse_config_overrides(once) == once


# --------------------------------------------------------------------- #
# Validation                                                            #
# --------------------------------------------------------------------- #


def test_unknown_key_raises_and_names_the_supported_set() -> None:
    with pytest.raises(ConfigOverrideError) as exc:
        coerce_config_overrides({"dpo_lernin_rate": 5e-7})
    message = str(exc.value)
    assert "dpo_lernin_rate" in message
    assert "dpo_learning_rate" in message


def test_unknown_key_is_never_silently_dropped() -> None:
    with pytest.raises(ConfigOverrideError):
        coerce_config_overrides({"epochs": 2, "not_a_field": 1})


def test_base_model_is_locked_to_its_own_flag() -> None:
    with pytest.raises(ConfigOverrideError, match="--base-model"):
        coerce_config_overrides({"base_model": "qwen2.5-1.5b"})


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ({"dpo_learning_rate": -1}, "greater than 0"),
        ({"dpo_learning_rate": 0}, "greater than 0"),
        ({"learning_rate": 0}, "greater than 0"),
        ({"epochs": 0}, "greater than 0"),
        ({"max_steps": -5}, "greater than 0"),
        ({"seed": -1}, ">= 0"),
        ({"weight_decay": -0.1}, ">= 0"),
        ({"lora_dropout": 1.5}, "[0.0, 1.0]"),
        ({"warmup_ratio": -0.1}, "[0.0, 1.0]"),
        ({"dpo_preference_filter": "bogus"}, "not a supported value"),
        ({"checkpoint_selection_metric": "loss"}, "not a supported value"),
    ],
)
def test_out_of_range_values_raise_at_parse_time(
    spec: Dict[str, Any], needle: str,
) -> None:
    with pytest.raises(ConfigOverrideError) as exc:
        coerce_config_overrides(spec)
    assert needle in str(exc.value)


@pytest.mark.parametrize(
    "spec",
    [
        {"dpo_learning_rate": "fast"},
        {"epochs": 1.5},
        {"epochs": True},
        {"use_4bit": "maybe"},
        {"target_modules": []},
        {"target_modules": ["q_proj", ""]},
        {"dpo_preference_filter": ["all"]},
    ],
)
def test_bad_types_raise_at_parse_time(spec: Dict[str, Any]) -> None:
    with pytest.raises(ConfigOverrideError):
        coerce_config_overrides(spec)


def test_values_are_coerced_to_the_declared_field_type() -> None:
    coerced = coerce_config_overrides({
        "dpo_learning_rate": "5e-7",
        "epochs": "2",
        "use_4bit": "false",
        "save_total_limit": 4.0,
    })
    assert isinstance(coerced["dpo_learning_rate"], float)
    assert coerced["epochs"] == 2 and isinstance(coerced["epochs"], int)
    assert coerced["use_4bit"] is False
    assert coerced["save_total_limit"] == 4


def test_optional_fields_accept_an_explicit_null() -> None:
    assert coerce_config_overrides({"max_steps": None}) == {"max_steps": None}
    assert parse_config_overrides("max_steps=null") == {"max_steps": None}


def test_every_overridable_field_round_trips_its_own_resolved_value() -> None:
    """No field is accidentally un-overridable (or wrongly coerced).

    Feeding each field the value the per-base YAML already resolved to must
    validate cleanly and come back unchanged — the parser's type table has to
    stay in step with the dataclass as fields are added.
    """
    from dataclasses import fields

    resolved = load_config("nemotron3-nano-30b").to_dict()
    overridable = {f.name for f in fields(TrainingConfig)} - {"base_model"}
    for name in sorted(overridable):
        assert coerce_config_overrides({name: resolved[name]}) == {
            name: resolved[name]
        }, f"{name} did not round-trip its own resolved value"


# --------------------------------------------------------------------- #
# load_config / card emit                                               #
# --------------------------------------------------------------------- #


def test_load_config_applies_overrides_over_the_per_base_yaml() -> None:
    base = load_config("nemotron3-nano-30b")
    assert base.dpo_learning_rate is None, (
        "the checked-in YAML must keep shipping a null DPO rate — this route "
        "exists precisely because the trainer refuses to invent one"
    )
    merged = load_config(
        "nemotron3-nano-30b", overrides={"dpo_learning_rate": 5e-7},
    )
    assert merged.dpo_learning_rate == 5e-7
    # Everything else is untouched.
    assert merged.epochs == base.epochs
    assert merged.learning_rate == base.learning_rate


def test_load_config_without_overrides_is_byte_identical() -> None:
    assert (
        load_config("nemotron3-nano-30b").to_dict()
        == load_config("nemotron3-nano-30b", overrides=None).to_dict()
        == load_config("nemotron3-nano-30b", overrides={}).to_dict()
    )


def test_card_dict_omits_an_unset_dpo_learning_rate() -> None:
    """No override => the card is byte-identical to a pre-feature card."""
    card = load_config("qwen2.5-1.5b").to_card_dict()
    assert "dpo_learning_rate" not in card


def test_card_dict_records_a_supplied_dpo_learning_rate() -> None:
    """An adapter trained at a hand-picked rate must be reproducible."""
    card = load_config(
        "nemotron3-nano-30b", overrides={"dpo_learning_rate": 5e-7},
    ).to_card_dict()
    assert card["dpo_learning_rate"] == 5e-7


# --------------------------------------------------------------------- #
# TrainingRunner                                                        #
# --------------------------------------------------------------------- #


def _course(tmp_path: Path) -> Path:
    root = tmp_path / "libv2"
    course = root / "demo-course"
    (course / "training_specs").mkdir(parents=True, exist_ok=True)
    (course / "imscc_chunks").mkdir(exist_ok=True)
    (course / "imscc_chunks" / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    specs = course / "training_specs"
    specs.joinpath("instruction_pairs.jsonl").write_text(
        json.dumps({"prompt": "p", "completion": "c", "provider": "local"}) + "\n",
        encoding="utf-8",
    )
    specs.joinpath("preference_pairs.jsonl").write_text(
        json.dumps(
            {"prompt": "p", "chosen": "a", "rejected": "b", "provider": "local"}
        ) + "\n",
        encoding="utf-8",
    )
    specs.joinpath("dataset_config.json").write_text("{}", encoding="utf-8")
    return root


def _run_card(tmp_path: Path, **kwargs: Any) -> Dict[str, Any]:
    from Trainforge.training import TrainingRunner

    runner = TrainingRunner(
        course_slug="demo-course",
        base_model=kwargs.pop("base_model", "nemotron3-nano-30b"),
        dry_run=True,
        libv2_root=_course(tmp_path),
        **kwargs,
    )
    return json.loads(
        Path(runner.run().model_card_path).read_text(encoding="utf-8")
    )


def test_runner_records_the_override_set_on_the_model_card(
    tmp_path: Path,
) -> None:
    card = _run_card(tmp_path, config_overrides={"dpo_learning_rate": "5e-7"})
    assert card["config_overrides"] == {"dpo_learning_rate": 5e-7}
    assert card["training_config"]["dpo_learning_rate"] == 5e-7


def test_runner_omits_the_provenance_block_without_overrides(
    tmp_path: Path,
) -> None:
    card = _run_card(tmp_path, base_model="qwen2.5-1.5b")
    assert "config_overrides" not in card
    assert "dpo_learning_rate" not in card["training_config"]


def test_emitted_cards_validate_against_the_schema(tmp_path: Path) -> None:
    import jsonschema

    schema = json.loads(
        (PROJECT_ROOT / "schemas" / "models" / "model_card.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.validate(
        _run_card(tmp_path, config_overrides={"dpo_learning_rate": 5e-7}),
        schema,
    )
    jsonschema.validate(_run_card(tmp_path, base_model="qwen2.5-1.5b"), schema)


def test_runner_rejects_a_bad_override_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ConfigOverrideError):
        _run_card(tmp_path, config_overrides={"dpo_learning_rate": -1})


def test_the_override_reaches_the_dict_peft_trainer_reads(
    tmp_path: Path,
) -> None:
    """The last link: ``PeftTrainer.training_config['dpo_learning_rate']``.

    ``runner._dispatch_training`` passes ``self.config.to_dict()`` (the FULL
    dataclass, not ``to_card_dict``) as ``TrainingJobSpec.training_config``,
    and ``peft_trainer.fit_dpo`` reads ``dpo_learning_rate`` straight off it —
    raising when it is None on ``nemotron3-nano-30b``. If this key is absent
    or None here, DPO on that base cannot start.
    """
    from Trainforge.training import TrainingRunner

    runner = TrainingRunner(
        course_slug="demo-course",
        base_model="nemotron3-nano-30b",
        dry_run=True,
        libv2_root=_course(tmp_path),
        config_overrides={"dpo_learning_rate": 5e-7},
    )
    assert runner.config.to_dict()["dpo_learning_rate"] == 5e-7
    # ...and without the override it is still None, i.e. the trainer's raise
    # is the correct, unchanged behaviour.
    bare = TrainingRunner(
        course_slug="demo-course",
        base_model="nemotron3-nano-30b",
        dry_run=True,
        libv2_root=_course(tmp_path),
    )
    assert bare.config.to_dict()["dpo_learning_rate"] is None


def test_runner_applies_overrides_over_an_explicit_config(
    tmp_path: Path,
) -> None:
    """An explicitly-supplied config must not swallow the operator's flag."""
    from Trainforge.training import TrainingRunner

    runner = TrainingRunner(
        course_slug="demo-course",
        base_model="nemotron3-nano-30b",
        config=load_config("nemotron3-nano-30b"),
        dry_run=True,
        libv2_root=_course(tmp_path),
        config_overrides={"dpo_learning_rate": 5e-7},
    )
    assert runner.config.dpo_learning_rate == 5e-7
