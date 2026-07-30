"""Train per-Bloom-level DeBERTa-v3 classifier head(s): one-vs-rest or multiclass.

Bloom-ladder initiative WI-13, operator-run precedent-mirror of
``SemantiK/training/train_classifier.py``, extended by addendum WI-AD-01
with a second ``--head`` mode.

**``--head one-vs-rest`` (default, back-compat).** Run six times (once per
:data:`lib.bloom_labels.dataset.BLOOM_LEVELS` entry via ``--bloom-level``) to
populate the checkpoint tree
:class:`lib.classifiers.bloom_deberta_heads.BloomDebertaHeads` loads at
``ED4ALL_BLOOM_HEADS_DIR`` (default ``models/bloom_classifiers``):
``<heads-dir>/<level>/final/`` holding the saved model + tokenizer +
``summary.json``. Task: binary text classification, one-vs-rest per Bloom
level (input: harvested block/objective/assessment-item text; output: 1
"this text IS at <bloom-level>" or 0, every other level).

**``--head multiclass``.** ONE run trains a single ``num_labels=6`` softmax
head over all six levels at once instead of six independent binary heads —
the addendum's answer to the thin-class tail (a measured harvest ran roughly
930 / 790 / 590 / 350 / 140 / 80 across understand..create — a one-vs-rest
evaluate/create head trains on only ~100-150 positives). Writes
``<heads-dir>/multiclass/final/`` by default.
``--bloom-level`` is forbidden with ``--head multiclass`` (argparse error) —
see ``docs/operations/bloom-deberta-training.md`` DECISION PROTOCOL for when
to prefer this mode over the six one-vs-rest runs.

Base: a pre-seeded LOCAL snapshot of the MIT ``microsoft/deberta-v3-base``
checkpoint (``models/base/deberta-v3-base`` by default — not fetched over
the network by this script; see :func:`validate_base_model_path`).

Splits come from :mod:`lib.bloom_labels.dataset` (WI-04): a seeded
per-level stratified train/val split, then either a per-split one-vs-rest
binary view for the requested level, or a per-split multiclass view over
all six levels. Class-weighted cross-entropy loss is always applied in
both modes — there is no opt-out flag, the same way the harvester's
``class_balance_report`` always surfaces the thin levels rather than
letting them silently disappear into an aggregate accuracy number.

Not registered in any ``@mcp.tool()`` surface, ``AGENT_TOOL_MAPPING``, or
``config/workflows.yaml`` phase — unreachable by agents; an operator runs
it directly:

    python -m lib.classifiers.training.train_bloom_deberta \\
        --bloom-level create

    python -m lib.classifiers.training.train_bloom_deberta \\
        --head multiclass

Heavy imports (torch / transformers / datasets / sklearn) are deferred
inside the training functions so this module stays importable on a
CPU-only box without the ``[training]`` extra — mirrors the layering
contract documented on :mod:`lib.bloom_labels.dataset` (pure-stdlib prep,
tensor-touching work stays in a separate, later-imported module).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.bloom_labels.dataset import (
    BLOOM_LEVEL_TO_ID,
    BLOOM_LEVELS,
    DEFAULT_SEED,
    ID_TO_BLOOM_LEVEL,
    MulticlassView,
    OneVsRestView,
    compute_binary_class_weights,
    compute_multiclass_class_weights,
    multiclass_view,
    one_vs_rest_view,
    prepare_dataset,
)

#: Base directory the six per-level checkpoints live under. Mirrors
#: ``lib.classifiers.bloom_deberta_heads._DEFAULT_HEADS_DIR`` verbatim but
#: is deliberately NOT imported from there -- that module pulls in
#: ``lib.classifiers.nli_classifier`` (a sibling import graph this trainer
#: has no other reason to depend on), and a plain string constant is cheap
#: to keep in sync (both are covered by the WI-04/WI-05 spec's frozen
#: ``<heads-dir>/<level>/final`` shape).
_DEFAULT_HEADS_DIR = "models/bloom_classifiers"

#: Pre-seeded local snapshot of the MIT ``microsoft/deberta-v3-base``
#: checkpoint. Never a bare HuggingFace hub id by default -- see
#: :func:`validate_base_model_path`.
DEFAULT_BASE_MODEL = "models/base/deberta-v3-base"

#: Env var HuggingFace's own stack honors to force offline mode. When
#: truthy, ``--base-model`` MUST resolve to a local directory: a bare hub
#: id would otherwise sail past argument parsing and only fail deep inside
#: ``from_pretrained`` with a confusing network-error traceback.
HF_HUB_OFFLINE_ENV = "HF_HUB_OFFLINE"

#: Parse-with-fallback truthy set -- mirrors the repo-wide convention
#: (e.g. ``lib.classifiers.nli_microbatch._TRUTHY``,
#: ``lib.classifiers.bloom_zero_shot._TRUTHY_VALUES``).
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

#: Binary label vocabulary for a one-vs-rest head. ``1`` = "this text IS
#: at the requested Bloom level", ``0`` = every other level.
LABELS = ["negative", "positive"]
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}


def _is_truthy(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY_VALUES


def default_output_dir(bloom_level: str) -> Path:
    """``<heads-dir>/<level>`` -- the exact shape the WI-05 registry reads.

    :class:`lib.classifiers.bloom_deberta_heads.BloomDebertaHeads`'s
    ``default_registry`` appends the ``final`` checkpoint subdir itself
    (``<heads_dir>/<level>/final``); this trainer does the same in
    :func:`main` so a default (unflagged) run lands its output exactly
    where the loader looks, with zero operator path-wiring required.
    """
    return Path(_DEFAULT_HEADS_DIR) / bloom_level


def default_multiclass_output_dir() -> Path:
    """``<heads-dir>/multiclass`` -- default ``--output-dir`` for ``--head multiclass``.

    Sibling of :func:`default_output_dir`;
    :meth:`lib.classifiers.bloom_deberta_heads.BloomDebertaHeads.get_or_load`'s
    auto-detect (WI-AD-01c) looks for exactly ``<heads_dir>/multiclass/final``
    BEFORE falling back to the six-way one-vs-rest ladder, so a default
    (unflagged) multiclass run lands its output exactly where the loader
    checks first.
    """
    return Path(_DEFAULT_HEADS_DIR) / "multiclass"


def validate_base_model_path(
    base_model: str, *, hf_hub_offline: Optional[str] = None
) -> None:
    """Reject a HuggingFace hub id for ``--base-model`` under HF_HUB_OFFLINE.

    ``hf_hub_offline`` defaults to the live ``HF_HUB_OFFLINE`` env (read at
    call time, not import time -- an explicit arg is test-only). When that
    value is truthy and ``base_model`` is not an existing local directory,
    raises :class:`ValueError` with an actionable message up front, rather
    than letting the run limp into ``from_pretrained`` and fail minutes
    later (after the dataset load + tokenization) with an opaque
    connection error.

    A falsy/unset ``HF_HUB_OFFLINE`` is a no-op regardless of what
    ``base_model`` looks like -- this project's default posture is
    license-clean LOCAL models (see module docstring), but this function
    only enforces the offline contract when the operator has explicitly
    opted into it.
    """
    env_value = (
        hf_hub_offline if hf_hub_offline is not None
        else os.environ.get(HF_HUB_OFFLINE_ENV)
    )
    if not _is_truthy(env_value):
        return
    if Path(base_model).is_dir():
        return
    raise ValueError(
        f"--base-model {base_model!r} is not a local directory, but "
        f"{HF_HUB_OFFLINE_ENV} is set -- refusing to fall through to a "
        f"HuggingFace hub id (that lookup would just fail offline anyway, "
        f"further downstream and less legibly). Point --base-model at a "
        f"pre-seeded local snapshot directory (e.g. {DEFAULT_BASE_MODEL!r}), "
        f"or unset {HF_HUB_OFFLINE_ENV} to allow a hub download for this run."
    )


def build_binary_examples(view: OneVsRestView) -> List[Dict[str, Any]]:
    """One-vs-rest binary examples for one split's :class:`OneVsRestView`.

    ``label=1`` for every row asserted AT ``view.level`` (``view.positive``),
    ``label=0`` for every row at any other level (``view.negative``) --
    plain dicts (``{"text": ..., "label": ...}``), the shape
    ``datasets.Dataset.from_list`` wants downstream. Pure stdlib -- no
    ``datasets`` import here, so this is directly testable without the
    ``[training]`` extra.
    """
    examples: List[Dict[str, Any]] = [
        {"text": row.text, "label": LABEL2ID["positive"]} for row in view.positive
    ]
    examples.extend(
        {"text": row.text, "label": LABEL2ID["negative"]} for row in view.negative
    )
    return examples


def build_multiclass_examples(view: MulticlassView) -> List[Dict[str, Any]]:
    """Six-way multiclass examples for one split's :class:`MulticlassView`.

    ``label`` is the row's own level mapped through the canonical
    :data:`lib.bloom_labels.dataset.BLOOM_LEVEL_TO_ID` (0-5, BLOOM_LEVELS
    order) -- the shape ``datasets.Dataset.from_list`` wants for a single
    6-way softmax head, mirroring :func:`build_binary_examples`'s
    one-vs-rest framing but with no positive/negative split.
    """
    return [
        {"text": row.text, "label": BLOOM_LEVEL_TO_ID[row.bloom_level]}
        for row in view.rows
    ]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train one per-Bloom-level DeBERTa-v3 one-vs-rest binary "
            "classifier head (operator-run; see module docstring)."
        )
    )
    ap.add_argument(
        "--labels-path", type=Path,
        default=Path("runtime/state/bloom_labels/labels.jsonl"),
        help="Harvested Bloom-label store (ed4all harvest-bloom-labels output).",
    )
    ap.add_argument(
        "--head", choices=["one-vs-rest", "multiclass"], default="one-vs-rest",
        help=(
            "one-vs-rest (default, back-compat): six independent per-level "
            "binary heads, one operator-run invocation per --bloom-level. "
            "multiclass: ONE num_labels=6 softmax head over all six levels "
            "in a single run -- --bloom-level is forbidden with this mode "
            "(see docs/operations/bloom-deberta-training.md DECISION "
            "PROTOCOL for when to prefer it)."
        ),
    )
    ap.add_argument(
        "--bloom-level", default=None, choices=list(BLOOM_LEVELS),
        help=(
            "Which of the six canonical Bloom levels this head votes for. "
            "Required for --head one-vs-rest (the default); forbidden for "
            "--head multiclass."
        ),
    )
    ap.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help="Local checkpoint directory (or hub id, if HF_HUB_OFFLINE is unset).",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Defaults to <models/bloom_classifiers>/<bloom-level> for "
            "--head one-vs-rest, or <models/bloom_classifiers>/multiclass "
            "for --head multiclass -- the exact paths the BloomDebertaHeads "
            "loader's auto-detect expects (a 'final' checkpoint subdir is "
            "appended automatically)."
        ),
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args(argv)
    if args.head == "one-vs-rest" and args.bloom_level is None:
        ap.error("--bloom-level is required for --head one-vs-rest")
    if args.head == "multiclass" and args.bloom_level is not None:
        ap.error(
            "--bloom-level is not allowed with --head multiclass -- one "
            "num_labels=6 head trains over all six levels in a single run"
        )
    return args


def _train_one_vs_rest(args: argparse.Namespace, output_dir: Path) -> None:
    """``--head one-vs-rest`` training path -- byte-identical to the
    pre-WI-AD-01 ``main()`` body (only lifted into its own function so
    :func:`main` can dispatch between this and :func:`_train_multiclass`).
    """
    print(f"[data] loading {args.labels_path} (level={args.bloom_level})")
    dataset = prepare_dataset(args.labels_path, seed=args.seed)
    train_view = one_vs_rest_view(dataset.split.train, args.bloom_level)
    val_view = one_vs_rest_view(dataset.split.val, args.bloom_level)
    print(f"  train: positive={train_view.n_positive} negative={train_view.n_negative}")
    print(f"  val:   positive={val_view.n_positive} negative={val_view.n_negative}")

    train_examples = build_binary_examples(train_view)
    val_examples = build_binary_examples(val_view)
    if not train_examples:
        raise RuntimeError(
            f"no training examples for bloom level {args.bloom_level!r} out "
            f"of {args.labels_path} -- confirm the harvester has run "
            f"(ed4all harvest-bloom-labels) and the store actually holds "
            f"rows for this level (positive) or any other level (negative)."
        )
    if not val_examples:
        raise RuntimeError(
            f"no validation examples for bloom level {args.bloom_level!r} "
            f"out of {args.labels_path} -- the corpus is too thin to hold "
            f"out a val split for this level yet (see "
            f"lib.bloom_labels.dataset.class_balance_report)."
        )

    class_weights_map = compute_binary_class_weights(train_view)
    print(
        f"[weights] negative={class_weights_map['negative']:.3f} "
        f"positive={class_weights_map['positive']:.3f}"
    )

    # -------------------------------------------------------------------
    # Heavy imports, deferred so this module stays CPU-importable without
    # the [training] extra (module docstring contract).
    # -------------------------------------------------------------------
    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    class WeightedLossTrainer(Trainer):
        """Trainer variant applying the fixed class-weighted CE loss above.

        Mirrors ``SemantiK/training/train_classifier.py::WeightedLossTrainer``
        minus the multiclass ``weight_cap`` / oversampling knobs -- a
        two-class one-vs-rest head only ever has two weights, so there is
        no runaway-skew case to cap.
        """

        def __init__(self, *args: Any, class_weights: "torch.Tensor", **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self, model: Any, inputs: Dict[str, Any],
            return_outputs: bool = False, num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = nn.functional.cross_entropy(
                logits, labels, weight=self.class_weights.to(logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    print(f"[gpu] cuda={torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def _tokenize(batch: Dict[str, List[str]]) -> Any:
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_list(train_examples).map(_tokenize, batched=True)
    val_ds = Dataset.from_list(val_examples).map(_tokenize, batched=True)

    class_weights = torch.tensor(
        [class_weights_map["negative"], class_weights_map["positive"]],
        dtype=torch.float32,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    def _compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_positive": f1_score(
                y_true, y_pred, pos_label=LABEL2ID["positive"],
                average="binary", zero_division=0,
            ),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # The thin tail (create=81, evaluate=142) is exactly where a
        # low-confidence positive vote does the most damage to the
        # trivote's abstention contract -- select on the POSITIVE-class
        # F1, not aggregate accuracy, which a 90%-negative corpus can
        # satisfy by predicting "not this level" every time.
        metric_for_best_model="f1_positive",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
    )

    print("[train] starting...")
    trainer.train()

    print("\n[val] final evaluation on the held-out val split")
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    for k, v in val_metrics.items():
        print(f"  {k:30} {v}")

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = np.array(preds.label_ids)
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    print("\n[val] per-class report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=LABELS,
        zero_division=0, digits=3,
    ))

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"\n[save] model + tokenizer -> {final_dir}")

    summary = {
        "bloom_level": args.bloom_level,
        "base_model": args.base_model,
        "seed": args.seed,
        "labels": LABELS,
        "label_counts": {
            "train": {
                "positive": train_view.n_positive,
                "negative": train_view.n_negative,
            },
            "val": {
                "positive": val_view.n_positive,
                "negative": val_view.n_negative,
            },
        },
        "class_weights": class_weights_map,
        "per_class_metrics": {
            "negative": report["negative"],
            "positive": report["positive"],
        },
        "val_metrics": val_metrics,
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] summary -> {final_dir / 'summary.json'}")


def _train_multiclass(args: argparse.Namespace, output_dir: Path) -> None:
    """``--head multiclass`` training path -- ONE ``num_labels=6`` softmax
    head over all six Bloom levels at once, instead of six independent
    one-vs-rest heads. Mirrors :func:`_train_one_vs_rest`'s shape (prep ->
    tokenize -> train -> evaluate -> save + summary.json) but selects the
    best checkpoint on macro-F1 across all six classes -- there is no
    single "positive class" the way a one-vs-rest head has one.
    """
    print(f"[data] loading {args.labels_path} (head=multiclass)")
    dataset = prepare_dataset(args.labels_path, seed=args.seed)
    train_view = multiclass_view(dataset.split.train)
    val_view = multiclass_view(dataset.split.val)
    print(f"  train: {train_view.label_counts}")
    print(f"  val:   {val_view.label_counts}")

    train_examples = build_multiclass_examples(train_view)
    val_examples = build_multiclass_examples(val_view)
    if not train_examples:
        raise RuntimeError(
            f"no training examples out of {args.labels_path} -- confirm "
            f"the harvester has run (ed4all harvest-bloom-labels) and the "
            f"store actually holds ANY labeled rows (see "
            f"lib.bloom_labels.dataset.class_balance_report)."
        )
    if not val_examples:
        raise RuntimeError(
            f"no validation examples out of {args.labels_path} -- the "
            f"corpus is too thin to hold out a val split yet (see "
            f"lib.bloom_labels.dataset.class_balance_report)."
        )

    class_weights_map = compute_multiclass_class_weights(train_view)
    print(
        "[weights] "
        + " ".join(f"{level}={class_weights_map[level]:.3f}" for level in BLOOM_LEVELS)
    )

    # -------------------------------------------------------------------
    # Heavy imports, deferred so this module stays CPU-importable without
    # the [training] extra (module docstring contract).
    # -------------------------------------------------------------------
    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    class WeightedLossTrainer(Trainer):
        """Trainer variant applying the fixed class-weighted CE loss above.

        Mirrors ``_train_one_vs_rest``'s local ``WeightedLossTrainer``
        (itself a mirror of
        ``SemantiK/training/train_classifier.py::WeightedLossTrainer`` minus
        the ``weight_cap`` / oversampling knobs) -- duplicated rather than
        shared so the deferred-heavy-import contract holds independently in
        both training paths (a module-level class here would need ``nn`` at
        import time).
        """

        def __init__(self, *args: Any, class_weights: "torch.Tensor", **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self, model: Any, inputs: Dict[str, Any],
            return_outputs: bool = False, num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = nn.functional.cross_entropy(
                logits, labels, weight=self.class_weights.to(logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    print(f"[gpu] cuda={torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def _tokenize(batch: Dict[str, List[str]]) -> Any:
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_list(train_examples).map(_tokenize, batched=True)
    val_ds = Dataset.from_list(val_examples).map(_tokenize, batched=True)

    class_weights = torch.tensor(
        [class_weights_map[level] for level in BLOOM_LEVELS], dtype=torch.float32,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(BLOOM_LEVELS),
        id2label=ID_TO_BLOOM_LEVEL, label2id=BLOOM_LEVEL_TO_ID,
    )

    def _compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # Multiclass has no single "positive class" -- select on macro-F1
        # across all six levels so the thin evaluate/create tail can't be
        # ignored by an aggregate-accuracy-friendly checkpoint the way a
        # 90%-negative one-vs-rest split can.
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
    )

    print("[train] starting...")
    trainer.train()

    print("\n[val] final evaluation on the held-out val split")
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    for k, v in val_metrics.items():
        print(f"  {k:30} {v}")

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = np.array(preds.label_ids)
    target_names = list(BLOOM_LEVELS)
    label_ids = list(range(len(BLOOM_LEVELS)))
    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    print("\n[val] per-class report:")
    print(classification_report(
        y_true, y_pred, labels=label_ids, target_names=target_names,
        zero_division=0, digits=3,
    ))

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"\n[save] model + tokenizer -> {final_dir}")

    summary = {
        "head_mode": "multiclass",
        "base_model": args.base_model,
        "seed": args.seed,
        "labels": target_names,
        "label_counts": {
            "train": train_view.label_counts,
            "val": val_view.label_counts,
        },
        "class_weights": class_weights_map,
        "per_class_metrics": {level: report[level] for level in BLOOM_LEVELS},
        "val_metrics": val_metrics,
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] summary -> {final_dir / 'summary.json'}")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    validate_base_model_path(args.base_model)
    if args.head == "multiclass":
        output_dir = (
            args.output_dir if args.output_dir is not None
            else default_multiclass_output_dir()
        )
        _train_multiclass(args, output_dir)
    else:
        output_dir = (
            args.output_dir if args.output_dir is not None
            else default_output_dir(args.bloom_level)
        )
        _train_one_vs_rest(args, output_dir)


if __name__ == "__main__":
    main()
