"""Wave 90 — training-config loader for ``Trainforge.training``.

YAML files in this package map base-model short names (e.g.
``qwen2.5-1.5b.yaml``) to a :class:`TrainingConfig` dataclass. The
loader merges optional course-level overrides on top so a single course
can pin a non-default LR / rank / epochs without forking the per-base
defaults.

Per-base configs ship for ``qwen2.5-1.5b``, ``llama-3.2-1b``,
``smollm2-1.7b`` and ``nemotron3-nano-30b``. ``llama-3.2-3b`` and
``phi-3.5-mini`` registry entries exist but have no per-base config —
``load_config`` raises ``FileNotFoundError`` with a clear message so
the gap is loud, not silent.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml


@dataclass
class TrainingConfig:
    """Per-run training hyperparameters.

    Mirrors the ``training_config`` block of
    ``schemas/models/model_card.schema.json`` so the runner can pass
    the dataclass directly into ``model_card.training_config`` via
    :meth:`to_dict`.
    """

    base_model: str
    learning_rate: float
    epochs: int
    lora_rank: int
    lora_alpha: int
    max_seq_length: int
    batch_size: int
    seed: int
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    # SFT program recipe (runtime/scratchpad/sft_data_program.md §C): bf16 LoRA is
    # the DEFAULT (use_4bit=False) — the shipped adapter is a small 1.5B
    # course-tutor; QLoRA (4-bit) stays reachable behind this flag for
    # memory-constrained boxes. NB: LoRA already leaks less than full-FT;
    # keep epochs light (2-3) with early-stopping on.
    use_4bit: bool = False
    # Activation/gradient checkpointing: trade compute for memory on large
    # bases (e.g. the 30B nemotron3-nano-30b bf16 LoRA fit). Threaded into
    # SFTConfig as ``gradient_checkpointing=True`` +
    # ``gradient_checkpointing_kwargs={"use_reentrant": False}`` via the
    # accepted-kwargs shim, so older trl/transformers bands that lack a
    # kwarg degrade gracefully. Orchestration knob — NOT a CARD_FIELD.
    gradient_checkpointing: bool = False
    min_dpo_pairs: int = 50
    dpo_preference_filter: str = "editorial_or_misconception"
    dpo_fail_hard: bool = True
    # DPO needs its own calibrated rate.  ``None`` preserves legacy sibling
    # recipes, but the 30B Nano path fails loud until a measured canary selects
    # an override; reusing the SFT rate implicitly is not acceptable there.
    dpo_learning_rate: Optional[float] = None
    # Operator-only canary bound. ``None`` is the production path.
    max_steps: Optional[int] = None

    # ------------------------------------------------------------------ #
    # S8 training-recipe orchestration knobs (NOT persisted to the model  #
    # card — the schema's training_config is additionalProperties:false;  #
    # the runner filters to the canonical schema keys before emit).       #
    # ------------------------------------------------------------------ #
    # Completion-only loss masking: train the loss over the assistant
    # completion only, never the prompt (runtime/scratchpad/sft_data_program.md §C).
    completion_only_loss: bool = True
    # Sample a batch and fail LOUD if the prompt isn't masked before the
    # full run (peft_trainer._verify_completion_only_loss_mask).
    verify_loss_mask: bool = True
    # Per-epoch checkpoint retention so the downstream-probe selector can
    # pick the best epoch out-of-band. Defaults to `epochs` when None.
    save_total_limit: Optional[int] = None
    # Early-stopping patience (epochs) for the narrow/repetitive-corpus
    # overfit guard; consumed by the checkpoint-selection scaffolding.
    early_stopping_patience: int = 1
    # Empty preserves the legacy final-epoch path.  A per-base/course config
    # explicitly enables selection with "gold_keypoint_coverage",
    # "sympy_correctness", or "composite". Pair/held-out loss is only an
    # overfit tripwire, never a production selector.
    checkpoint_selection_metric: str = ""


    # Fields the model_card schema's ``training_config`` object accepts
    # (schemas/models/model_card.schema.json, additionalProperties:false).
    # The S8 orchestration knobs above are deliberately NOT in this set —
    # they steer the run but don't belong in the provenance card.
    CARD_FIELDS = (
        "seed", "learning_rate", "epochs", "lora_rank", "lora_alpha",
        "max_seq_length", "batch_size", "lora_dropout",
        "gradient_accumulation_steps", "warmup_ratio", "weight_decay",
        "target_modules", "use_4bit", "min_dpo_pairs",
        "dpo_preference_filter", "dpo_fail_hard",
        # The EFFECTIVE DPO learning rate. Recorded because an adapter whose
        # DPO stage ran at a hand-picked rate is not reproducible from the
        # card without it — and on ``nemotron3-nano-30b`` the rate is
        # REQUIRED to be operator-selected (peft_trainer raises when it is
        # None), so it is never derivable from the checked-in per-base YAML.
        # Omitted from the emitted card when None (see ``to_card_dict``), so
        # every legacy base's card stays byte-identical.
        "dpo_learning_rate",
    )

    #: ``CARD_FIELDS`` members that are dropped from the emitted card when
    #: their value is ``None``. Every other card field is non-Optional on the
    #: dataclass, so a None there is a real defect and must reach the schema
    #: validator rather than being quietly swallowed.
    CARD_OPTIONAL_FIELDS = ("dpo_learning_rate",)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_card_dict(self) -> Dict[str, Any]:
        """``to_dict`` filtered to the model_card-schema-accepted keys.

        The runner writes this into ``model_card.json::training_config`` so
        the S8 orchestration knobs never trip the schema's
        ``additionalProperties:false`` guard.
        """
        full = self.to_dict()
        return {
            k: full[k]
            for k in self.CARD_FIELDS
            if k in full
            and not (k in self.CARD_OPTIONAL_FIELDS and full[k] is None)
        }

    def merged(self, overrides: Dict[str, Any]) -> "TrainingConfig":
        """Return a new :class:`TrainingConfig` with ``overrides`` applied.

        Unknown keys in ``overrides`` raise ``ValueError`` so a typo
        fails loud rather than silently no-op-ing.
        """
        if not overrides:
            return self
        valid_fields = {f.name for f in fields(self)}
        unknown = set(overrides.keys()) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown TrainingConfig override key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_fields)}"
            )
        merged = self.to_dict()
        merged.update(overrides)
        # Re-cast so YAML floats aren't accidentally promoted to int etc.
        return TrainingConfig(**merged)


_CONFIG_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------- #
# Per-run config overrides (``--config-overrides``)                       #
# ---------------------------------------------------------------------- #


class ConfigOverrideError(ValueError):
    """An override spec that cannot be applied to :class:`TrainingConfig`.

    Raised at PARSE time — before any workflow state is written and long
    before a multi-hour trainer starts — so an unknown key or a garbage
    value costs seconds, not a training run.
    """


#: Values ``dpo_preference_filter`` may take. Fail-FAST echo of the runtime
#: enforcement in ``Trainforge/training/compute_backend.py::_filter_dpo_pairs``
#: and ``runner._count_dpo_eligible_records`` (both treat "" / "all" / None as
#: "admit everything"), NOT a second allowlist: a value that somehow gets past
#: here still meets those checks. Same pattern as
#: ``cli/commands/run.py::_validate_base_model`` echoing ``BaseModelRegistry``.
DPO_PREFERENCE_FILTERS = frozenset({"", "all", "editorial_or_misconception"})

#: Values ``checkpoint_selection_metric`` may take. ``""`` selects the legacy
#: final-epoch path; the three named metrics are the canonical set consumed by
#: ``Trainforge/training/checkpoint_probe.py``.
CHECKPOINT_SELECTION_METRICS = frozenset({
    "",
    "gold_keypoint_coverage",
    "sympy_correctness",
    "composite",
})

#: Fields an override may never set, mapped to the flag that owns them. The
#: base model is resolved through ``BaseModelRegistry`` and stamped on the
#: model card; letting an override rewrite it here would desync the card from
#: the weights that were actually loaded.
_LOCKED_OVERRIDE_FIELDS: Dict[str, str] = {"base_model": "--base-model"}

#: Numeric fields that are meaningless at or below zero (a rate, a count, a
#: window). ``dpo_learning_rate`` is the reason this whole route exists.
_POSITIVE_FIELDS = frozenset({
    "learning_rate",
    "dpo_learning_rate",
    "epochs",
    "lora_rank",
    "lora_alpha",
    "max_seq_length",
    "batch_size",
    "gradient_accumulation_steps",
    "save_total_limit",
    "max_steps",
})

#: Numeric fields where zero is legal but negatives are not.
_NON_NEGATIVE_FIELDS = frozenset({
    "seed",
    "weight_decay",
    "min_dpo_pairs",
    "early_stopping_patience",
})

#: Fields constrained to ``[0.0, 1.0]`` (mirrors the model_card schema's
#: ``minimum``/``maximum`` on the same keys).
_UNIT_INTERVAL_FIELDS = frozenset({"lora_dropout", "warmup_ratio"})

_ENUM_FIELDS: Dict[str, frozenset] = {
    "dpo_preference_filter": DPO_PREFERENCE_FILTERS,
    "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRICS,
}

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
_NULL_TOKENS = frozenset({"", "none", "null", "~"})


def _override_field_types() -> Dict[str, Any]:
    """Resolved (non-string) annotations for every ``TrainingConfig`` field.

    ``from __future__ import annotations`` makes ``dataclasses.fields()[i].type``
    a string, so coercion has to go through ``get_type_hints`` to see the real
    ``float`` / ``Optional[int]`` / ``List[str]`` objects.
    """
    hints = get_type_hints(TrainingConfig)
    return {f.name: hints[f.name] for f in fields(TrainingConfig) if f.name in hints}


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ConfigOverrideError(
        f"{name}: expected a boolean, got {value!r}. Accepted: "
        f"{sorted(_TRUE_TOKENS | _FALSE_TOKENS)}."
    )


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ConfigOverrideError(f"{name}: expected an integer, got {value!r}.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ConfigOverrideError(
            f"{name}: expected an integer, got the fractional value {value!r}."
        )
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigOverrideError(
            f"{name}: expected an integer, got {value!r}."
        ) from None


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ConfigOverrideError(f"{name}: expected a number, got {value!r}.")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigOverrideError(
            f"{name}: expected a number, got {value!r}."
        ) from None


def _coerce_str_list(name: str, value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    elif isinstance(value, str):
        # ``|`` first so an inline ``k=v`` spec (whose pairs are split on
        # ``,``) can still carry a list; a YAML/JSON file supplies a real
        # list and never reaches this branch.
        raw = value.strip()
        parts = raw.split("|") if "|" in raw else raw.split(",")
        items = [p.strip() for p in parts]
    else:
        raise ConfigOverrideError(
            f"{name}: expected a list of strings, got {value!r}."
        )
    if not items or any(not item for item in items):
        raise ConfigOverrideError(
            f"{name}: expected a non-empty list of non-empty strings, "
            f"got {value!r}."
        )
    return items


def _coerce_override(name: str, value: Any, target: Any) -> Any:
    """Coerce one override value to the field's declared type."""
    if get_origin(target) is Union:
        args = [a for a in get_args(target) if a is not type(None)]  # noqa: E721
        nullable = len(args) != len(get_args(target))
        if nullable and (
            value is None
            or (isinstance(value, str) and value.strip().lower() in _NULL_TOKENS)
        ):
            return None
        if len(args) != 1:
            raise ConfigOverrideError(
                f"{name}: unsupported override type {target!r}."
            )
        target = args[0]

    origin = get_origin(target)
    if origin in (list, List):
        return _coerce_str_list(name, value)
    if target is bool:
        return _coerce_bool(name, value)
    if target is int:
        return _coerce_int(name, value)
    if target is float:
        return _coerce_float(name, value)
    if target is str:
        if isinstance(value, (list, tuple, dict)):
            raise ConfigOverrideError(
                f"{name}: expected a string, got {value!r}."
            )
        return str(value).strip()
    raise ConfigOverrideError(f"{name}: unsupported override type {target!r}.")


def _range_check(name: str, value: Any) -> Any:
    """Reject an in-type but out-of-range value.

    A garbage value has to die here — six hours into a training run is the
    wrong place to discover ``dpo_learning_rate=-1``.
    """
    if value is None:
        return value
    if name in _ENUM_FIELDS and value not in _ENUM_FIELDS[name]:
        raise ConfigOverrideError(
            f"{name}: {value!r} is not a supported value. "
            f"Supported: {sorted(_ENUM_FIELDS[name])}."
        )
    if name in _POSITIVE_FIELDS and value <= 0:
        raise ConfigOverrideError(
            f"{name}: must be greater than 0, got {value!r}."
        )
    if name in _NON_NEGATIVE_FIELDS and value < 0:
        raise ConfigOverrideError(
            f"{name}: must be >= 0, got {value!r}."
        )
    if name in _UNIT_INTERVAL_FIELDS and not (0.0 <= value <= 1.0):
        raise ConfigOverrideError(
            f"{name}: must be within [0.0, 1.0], got {value!r}."
        )
    return value


def coerce_config_overrides(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate + type-coerce an override mapping against ``TrainingConfig``.

    Every key must name a real ``TrainingConfig`` field; every value is cast
    to that field's declared type and range-checked. Unknown keys raise
    :class:`ConfigOverrideError` naming the supported set — they are NEVER
    dropped, and a new attribute is never invented on the dataclass.

    Idempotent: re-running it over an already-coerced dict is a no-op, so the
    CLI can validate at parse time and the handler can re-validate whatever
    reached it off persisted state without the two disagreeing.
    """
    if not isinstance(raw, Mapping):
        raise ConfigOverrideError(
            f"config overrides must be a mapping of TrainingConfig fields, "
            f"got {type(raw).__name__}."
        )
    types = _override_field_types()
    supported = sorted(set(types) - set(_LOCKED_OVERRIDE_FIELDS))
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if name in _LOCKED_OVERRIDE_FIELDS:
            raise ConfigOverrideError(
                f"{name!r} cannot be set through config overrides; use "
                f"{_LOCKED_OVERRIDE_FIELDS[name]} instead so the resolved "
                f"base and the model card cannot disagree."
            )
        if name not in types:
            raise ConfigOverrideError(
                f"Unknown TrainingConfig override key {name!r}. "
                f"Supported keys: {supported}."
            )
        out[name] = _range_check(name, _coerce_override(name, value, types[name]))
    return out


def parse_config_overrides(spec: Any) -> Dict[str, Any]:
    """Parse a ``--config-overrides`` spec into a validated override dict.

    Four accepted shapes, resolved in this order:

    1. an already-built mapping (validated + coerced, nothing else);
    2. a path to an EXISTING ``.yaml`` / ``.yml`` / ``.json`` file whose
       top-level keys are ``TrainingConfig`` fields;
    3. an inline JSON object (a string starting with ``{``);
    4. an inline ``key=value[,key=value]`` string; values are read as YAML
       scalars so ``5e-7`` is a float and ``true`` is a bool. A list-valued
       field (``target_modules``) uses ``|`` between items, since ``,``
       already separates the pairs.

    A string that is neither an existing file nor an inline spec raises —
    a mistyped path must not be silently read as "no overrides".
    """
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        return coerce_config_overrides(spec)
    if isinstance(spec, Path):
        return coerce_config_overrides(_load_override_file(spec))
    if not isinstance(spec, str):
        raise ConfigOverrideError(
            f"config overrides must be a mapping, a file path, or an inline "
            f"spec string; got {type(spec).__name__}."
        )

    text = spec.strip()
    if not text:
        return {}

    candidate = Path(text).expanduser()
    if candidate.is_file():
        return coerce_config_overrides(_load_override_file(candidate))

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ConfigOverrideError(
                f"config overrides looked like inline JSON but did not parse: "
                f"{exc}."
            ) from None
        return coerce_config_overrides(payload)

    if "=" not in text:
        raise ConfigOverrideError(
            f"config overrides {spec!r} is neither an existing file nor an "
            f"inline spec. Pass a YAML/JSON file path, an inline JSON object, "
            f"or key=value pairs (e.g. 'dpo_learning_rate=5e-7')."
        )

    payload: Dict[str, Any] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ConfigOverrideError(
                f"config overrides: {token!r} is not a key=value pair."
            )
        key, _, value = token.partition("=")
        key = key.strip()
        if not key:
            raise ConfigOverrideError(
                f"config overrides: {token!r} has an empty key."
            )
        if key in payload:
            raise ConfigOverrideError(
                f"config overrides: {key!r} supplied more than once."
            )
        # YAML scalar rules give ints / floats / bools / null for free and
        # leave anything else a string.
        try:
            payload[key] = yaml.safe_load(value.strip())
        except yaml.YAMLError:
            payload[key] = value.strip()
    return coerce_config_overrides(payload)


def _load_override_file(path: Path) -> Mapping[str, Any]:
    """Read a YAML (or JSON — YAML is a superset) override file."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigOverrideError(
            f"config overrides file {path} could not be read: {exc}."
        ) from None
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigOverrideError(
            f"config overrides file {path} must hold a mapping of "
            f"TrainingConfig fields, got {type(payload).__name__}."
        )
    return payload


def _yaml_filename_for(base_model: str) -> Path:
    """Return the canonical YAML path for a base model short-name.

    Uses the short name verbatim with ``.yaml`` appended. Wave 90's
    five registered bases all map cleanly with this rule.
    """
    return _CONFIG_DIR / f"{base_model}.yaml"


def load_config(
    base_model: str,
    course_overrides: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> TrainingConfig:
    """Load the per-base default config and merge any course overrides.

    Args:
        base_model: Short name registered in
            :data:`Trainforge.training.base_models._REGISTRY`. Typo
            here surfaces as ``FileNotFoundError`` listing the dir.
        course_overrides: Optional path to a YAML file whose top-level
            keys override the per-base defaults. Unknown keys fail loud
            via :meth:`TrainingConfig.merged`.
        overrides: Optional PER-RUN override mapping (the
            ``--config-overrides`` route). Validated + type-coerced through
            :func:`coerce_config_overrides` and applied AFTER
            ``course_overrides``, so a per-run value wins over a per-course
            file. ``None`` / empty leaves the merge chain byte-identical.

    Returns:
        The merged :class:`TrainingConfig`.

    Raises:
        FileNotFoundError: when no per-base YAML ships for the
            requested base. The error message points at the expected
            path so it's obvious where to drop the new config.
    """
    yaml_path = _yaml_filename_for(base_model)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No training config for base_model={base_model!r}. "
            f"Expected {yaml_path}. Configs ship for qwen2.5-1.5b, "
            f"llama-3.2-1b, smollm2-1.7b, nemotron3-nano-30b. Add a "
            f"YAML with the same shape to extend support."
        )
    base_payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    base_payload.setdefault("base_model", base_model)
    cfg = TrainingConfig(**base_payload)

    if course_overrides is not None:
        overrides_path = Path(course_overrides)
        if not overrides_path.exists():
            raise FileNotFoundError(
                f"course_overrides path does not exist: {overrides_path}"
            )
        override_payload = yaml.safe_load(
            overrides_path.read_text(encoding="utf-8")
        ) or {}
        cfg = cfg.merged(override_payload)

    if overrides:
        cfg = cfg.merged(coerce_config_overrides(overrides))
    return cfg


__all__ = [
    "CHECKPOINT_SELECTION_METRICS",
    "ConfigOverrideError",
    "DPO_PREFERENCE_FILTERS",
    "TrainingConfig",
    "coerce_config_overrides",
    "load_config",
    "parse_config_overrides",
]
