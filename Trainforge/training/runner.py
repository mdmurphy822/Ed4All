"""Wave 90 — :class:`TrainingRunner` orchestrates one end-to-end training run.

Reads the LibV2 course at ``LibV2/courses/<course_slug>/``:

    corpus/chunks.jsonl
    pedagogy/pedagogy_graph.json (or graph/pedagogy_graph.json)
    training_specs/instruction_pairs.jsonl
    training_specs/preference_pairs.jsonl
    training_specs/dataset_config.json
    graph/concept_graph_semantic.json (or .json fallback)
    graph/courseforge_v1.vocabulary.ttl (or vocabulary.ttl fallback)

…dispatches the trainer via a :class:`ComputeBackend`, and writes the
following back into the same course slug under
``models/<model_id>/``:

    adapter.safetensors
    model_card.json     (validates against schemas/models/model_card.schema.json)
    training_run.jsonl  (DecisionCapture stream — 4+ events guaranteed)

The runner is the **single Wave 89 → Wave 90 contract surface**: the
emitted card must validate against
:class:`lib.validators.libv2_model.LibV2ModelValidator` (Wave 89). When
``dry_run=True`` the runner skips the trainer and writes only the
model-card stub + decision capture, so tests can exercise the full
emit path on CPU-only CI.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.decision_capture import DecisionCapture
from lib.paths import LIBV2_COURSES, SCHEMAS_PATH

from Trainforge.training.base_models import BaseModelRegistry, BaseModelSpec
from Trainforge.training.compute_backend import (
    ComputeBackend,
    LocalBackend,
    TrainingJobResult,
    TrainingJobSpec,
)
from Trainforge.training.configs import TrainingConfig, load_config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Provenance contract                                                     #
# ---------------------------------------------------------------------- #
#
# The card pins SHA-256 hashes of these six artifacts. Each tuple is
# (provenance-key, [candidate, ...]) where each candidate is either a
# course-relative ``str`` (resolved against ``self.course_dir``) or an
# absolute ``Path`` (used as-is). First-existing wins so we accept both
# the v0.2.0 and v0.3.0 LibV2 layouts.

# Wave 96: the canonical Courseforge vocabulary lives under the
# project-root schemas/context/ tree, not inside each LibV2 course.
# Without this fallback the runner silently substitutes the empty-bytes
# sha256 for vocabulary_ttl_hash and emits a model card whose hash
# doesn't pin the actual TTL the synthesizer consumed.
_VOCABULARY_TTL_CANONICAL = SCHEMAS_PATH / "context" / "courseforge_v1.vocabulary.ttl"

_PROVENANCE_SOURCES = (
    ("chunks_hash", ["corpus/chunks.jsonl"]),
    ("pedagogy_graph_hash", [
        "graph/pedagogy_graph.json",
        "pedagogy/pedagogy_graph.json",
        "pedagogy/pedagogy_model.json",
    ]),
    ("instruction_pairs_hash", ["training_specs/instruction_pairs.jsonl"]),
    ("preference_pairs_hash", ["training_specs/preference_pairs.jsonl"]),
    ("concept_graph_hash", [
        "graph/concept_graph_semantic.json",
        "graph/concept_graph.json",
    ]),
    ("vocabulary_ttl_hash", [
        "graph/courseforge_v1.vocabulary.ttl",
        "graph/vocabulary.ttl",
        _VOCABULARY_TTL_CANONICAL,
    ]),
    # Wave 92: holdout split for Tier-2 eval. The runner emits a stub
    # split when the eval submodule has not pre-built one; the real
    # split is built by ``Trainforge.eval.holdout_builder.HoldoutBuilder``
    # before training. Empty-bytes hash is acceptable here because
    # the eval phase is gated below — a present-but-empty file means
    # "no holdout was built", not "the holdout was tampered with".
    ("holdout_graph_hash", [
        "eval/holdout_split.json",
        "training_specs/holdout_split.json",
    ]),
)

_REQUIRED_TRAINING_SPECS = (
    "instruction_pairs.jsonl",
    "preference_pairs.jsonl",
    "dataset_config.json",
)


@dataclass
class TrainingRunResult:
    """What :meth:`TrainingRunner.run` returns."""

    model_id: str
    run_dir: Path
    model_card_path: Path
    decision_capture_path: Path
    adapter_path: Optional[Path] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Runner                                                                  #
# ---------------------------------------------------------------------- #


class TrainingRunner:
    """Orchestrates one training run from LibV2 → adapter + model card.

    Designed so each step can be unit-tested independently. ``run()``
    is the public entry point; the helpers (``_compute_provenance``,
    ``_emit_model_card``, etc.) are deliberately small and pure.
    """

    def __init__(
        self,
        course_slug: str,
        base_model: str,
        output_dir: Optional[Path] = None,
        config: Optional[TrainingConfig] = None,
        backend: Optional[ComputeBackend] = None,
        *,
        dry_run: bool = False,
        libv2_root: Optional[Path] = None,
        config_overrides_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            course_slug: LibV2 course slug under ``LibV2/courses/``.
                The runner reads training_specs from this slug and
                writes ``models/<model_id>/`` back to the same slug
                (unless ``output_dir`` overrides).
            base_model: Short name resolved against
                :class:`BaseModelRegistry`.
            output_dir: Override for where ``<model_id>/`` is written.
                Defaults to ``LibV2/courses/<course_slug>/models/``.
            config: Optional pre-resolved :class:`TrainingConfig`.
                When None, the runner loads the per-base default and
                merges ``config_overrides_path`` if provided.
            backend: Optional :class:`ComputeBackend`. Defaults to
                ``LocalBackend(allow_no_gpu=dry_run)``.
            dry_run: Skip the actual trainer call. Card + decision
                capture are still emitted. Used by tests.
            libv2_root: Override for ``LIBV2_COURSES`` (testing).
            config_overrides_path: Optional path forwarded into
                :func:`load_config` when ``config`` is None.
        """
        self.course_slug = course_slug
        self.base_model = base_model
        self.dry_run = bool(dry_run)
        self.libv2_root = Path(libv2_root) if libv2_root else LIBV2_COURSES
        self.spec: BaseModelSpec = BaseModelRegistry.resolve(base_model)
        self.config: TrainingConfig = config or load_config(
            base_model, course_overrides=config_overrides_path,
        )
        self.backend: ComputeBackend = backend or LocalBackend(
            allow_no_gpu=self.dry_run,
        )

        self.course_dir = self._resolve_course_dir()
        self._models_root = (
            Path(output_dir) if output_dir else (self.course_dir / "models")
        )

    # ------------------------------------------------------------------ #
    # Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> TrainingRunResult:
        """Execute the run end-to-end.

        Returns the :class:`TrainingRunResult` with paths to the
        model card + decision capture (and adapter, when not dry-run).
        """
        self._assert_training_specs_present()

        provenance = self._compute_provenance()
        model_id = self._mint_model_id(provenance)
        run_dir = self._models_root / model_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Decision capture handle (Wave 89 added the trainforge-training
        # phase enum so this stream lives at:
        #   training-captures/trainforge/<COURSE>/phase_trainforge-training/
        # plus the LibV2-mirrored copy under the slug.
        capture = self._build_capture()

        try:
            self._log_planning_decision(capture, model_id, run_dir)
            self._log_base_model_decision(capture)
            self._log_hyperparam_decision(capture)
            should_run_dpo, dpo_rationale = self._decide_run_dpo()
            self._log_eval_decision(capture, should_run_dpo, dpo_rationale)

            adapter_path: Optional[Path] = None
            metrics: Dict[str, Any] = {}

            if not self.dry_run:
                job_result = self._dispatch_training(run_dir, should_run_dpo)
                adapter_path = job_result.adapter_path
                metrics = dict(job_result.metrics)
                if not adapter_path.exists():
                    raise RuntimeError(
                        f"Backend reported success but adapter file "
                        f"is missing: {adapter_path}"
                    )

            # Wave 92: run the eval harness BEFORE emitting the card
            # so the eval_scores block can be folded in. Skipped on
            # dry-run because there is no trained model to call.
            eval_scores: Optional[Dict[str, Any]] = None
            if not self.dry_run:
                eval_scores = self._run_eval_harness(run_dir, adapter_path)

            card_path = self._emit_model_card(
                run_dir=run_dir,
                model_id=model_id,
                provenance=provenance,
                adapter_path=adapter_path,
                eval_scores=eval_scores,
            )
        finally:
            capture.save()

        decisions_path = self._save_decision_run_log(capture, run_dir)
        return TrainingRunResult(
            model_id=model_id,
            run_dir=run_dir,
            model_card_path=card_path,
            decision_capture_path=decisions_path,
            adapter_path=adapter_path,
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # Resolution + integrity                                              #
    # ------------------------------------------------------------------ #

    def _resolve_course_dir(self) -> Path:
        """Return ``LibV2/courses/<course_slug>/``. Fails loud when missing."""
        candidate = self.libv2_root / self.course_slug
        if not candidate.exists():
            raise FileNotFoundError(
                f"LibV2 course slug not found: {candidate}. "
                f"Import the course first via ``ed4all run textbook_to_course`` "
                f"or ``libv2 import``."
            )
        return candidate

    def _assert_training_specs_present(self) -> None:
        """Refuse to start when the LibV2 course hasn't been synthesized."""
        specs_dir = self.course_dir / "training_specs"
        missing = [
            name for name in _REQUIRED_TRAINING_SPECS
            if not (specs_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"LibV2 course {self.course_slug!r} is missing training specs: "
                f"{missing}. Run "
                f"``python -m Trainforge.synthesize_training --slug "
                f"{self.course_slug}`` first."
            )

    # ------------------------------------------------------------------ #
    # Provenance + ID minting                                             #
    # ------------------------------------------------------------------ #

    def _compute_provenance(self) -> Dict[str, str]:
        """Hash each provenance source. Missing OPTIONAL sources are
        accepted for the v0.2.0-archive case — the model card schema
        requires all six, so we substitute the canonical
        ``e3b0c44…`` SHA-256-of-empty-bytes for missing artifacts and
        log a warning rather than fail-closed (the validator's
        pedagogy_hash check will catch a truly-broken pedagogy graph).

        Wave 90 keeps the empty-artifact substitution behind a strict
        check on chunks_hash + pedagogy_graph_hash + the two
        training-spec hashes — without those, the run is
        unreproducible and we refuse loudly.
        """
        sha_empty = hashlib.sha256(b"").hexdigest()
        out: Dict[str, str] = {}
        critical_missing: List[str] = []
        for key, candidates in _PROVENANCE_SOURCES:
            resolved: Optional[Path] = None
            for rel in candidates:
                # Absolute Path candidates (e.g. project-root canonical
                # vocabulary) are used as-is; str candidates are resolved
                # relative to the LibV2 course dir.
                if isinstance(rel, Path) and rel.is_absolute():
                    p = rel
                else:
                    p = self.course_dir / rel
                if p.exists():
                    resolved = p
                    break
            if resolved is None:
                if key in {
                    "chunks_hash",
                    "pedagogy_graph_hash",
                    "instruction_pairs_hash",
                    "preference_pairs_hash",
                }:
                    critical_missing.append(f"{key} (tried {candidates})")
                    continue
                logger.warning(
                    "TrainingRunner: optional provenance artifact missing "
                    "for %s; substituting empty-bytes sha256.",
                    key,
                )
                out[key] = sha_empty
            else:
                out[key] = _sha256_file(resolved)
        if critical_missing:
            raise FileNotFoundError(
                "TrainingRunner cannot mint model card; required "
                f"provenance artifacts missing: {critical_missing}"
            )
        return out

    def _mint_model_id(self, provenance: Dict[str, str]) -> str:
        """``<course-slug>-<base-short>-<8hex>``.

        ``<8hex>`` is the first 8 chars of SHA-256 over the sorted
        provenance hashes. Stable across re-runs over the same source
        artifacts — the same LibV2 course + same base model trained
        twice mints the same ``model_id``.
        """
        agg = hashlib.sha256()
        for key in sorted(provenance.keys()):
            agg.update(key.encode("utf-8"))
            agg.update(b"=")
            agg.update(provenance[key].encode("utf-8"))
            agg.update(b"\n")
        short_hash = agg.hexdigest()[:8]
        base_short = self.base_model.replace(".", "-").replace("/", "-").lower()
        slug = self.course_slug.lower()
        return f"{slug}-{base_short}-{short_hash}"

    # ------------------------------------------------------------------ #
    # Decision capture                                                    #
    # ------------------------------------------------------------------ #

    def _build_capture(self) -> DecisionCapture:
        """Construct the canonical ``trainforge-training`` capture.

        Wave 89 added ``trainforge-training`` to the canonical phase
        enum; without that, ``DECISION_VALIDATION_STRICT=true`` would
        fail-close on every event we log here.
        """
        return DecisionCapture(
            course_code=self.course_slug,
            phase="trainforge-training",
            tool="trainforge",
            streaming=True,
        )

    def _log_planning_decision(
        self,
        capture: DecisionCapture,
        model_id: str,
        run_dir: Path,
    ) -> None:
        capture.log_decision(
            decision_type="training_run_planning",
            decision=(
                f"Plan training run model_id={model_id!r} for "
                f"course_slug={self.course_slug!r} on base "
                f"{self.base_model!r} (dry_run={self.dry_run})."
            ),
            rationale=(
                f"Run dir resolved to {run_dir}. Backend "
                f"{type(self.backend).__name__} chosen; "
                f"epochs={self.config.epochs}, "
                f"learning_rate={self.config.learning_rate}, "
                f"lora_rank={self.config.lora_rank}, "
                f"max_seq_length={self.config.max_seq_length}. "
                f"Provenance hashes pin LibV2 artifacts so the card "
                f"is fully replayable post-hoc."
            ),
            alternatives_considered=[
                "Reuse an existing model_id (rejected: every retrain mints a "
                "new id keyed off provenance hashes)",
                "Skip dry-run scaffolding (rejected: tests need card emit "
                "without GPU)",
            ],
        )

    def _log_base_model_decision(self, capture: DecisionCapture) -> None:
        instr_count = _count_jsonl_records(
            self.course_dir / "training_specs" / "instruction_pairs.jsonl"
        )
        pref_count = _count_jsonl_records(
            self.course_dir / "training_specs" / "preference_pairs.jsonl"
        )
        capture.log_decision(
            decision_type="base_model_selection",
            decision=(
                f"Selected base model {self.base_model!r} "
                f"(huggingface_repo={self.spec.huggingface_repo!r}, "
                f"revision={self.spec.default_revision!r})."
            ),
            rationale=(
                f"Chose HF repo {self.spec.huggingface_repo!r} pinned at "
                f"revision={self.spec.default_revision!r} for course "
                f"{self.course_slug!r} ({instr_count} instruction pairs, "
                f"{pref_count} preference pairs). Chat template "
                f"{self.spec.chat_template!r} + recommended_max_seq_length="
                f"{self.spec.recommended_max_seq_length} + "
                f"recommended_lora_rank={self.spec.recommended_lora_rank} "
                f"match the per-base defaults loaded from "
                f"Trainforge/training/configs/{self.base_model}.yaml; "
                f"pinned revision keeps reruns byte-identical "
                f"(dry_run={self.dry_run})."
            ),
            alternatives_considered=[
                f"Larger base ({BaseModelRegistry.list_supported()}): "
                f"rejected — sub-2B {self.base_model!r} is the calibrated "
                f"baseline for {instr_count}-pair corpora at lora_rank="
                f"{self.config.lora_rank}.",
                f"Floating revision ('main' on {self.spec.huggingface_repo!r}): "
                f"rejected — would break the model_card provenance contract "
                f"on every HF repo update.",
            ],
        )

    def _log_hyperparam_decision(self, capture: DecisionCapture) -> None:
        instr_count = _count_jsonl_records(
            self.course_dir / "training_specs" / "instruction_pairs.jsonl"
        )
        # Effective steps are an order-of-magnitude signal of run cost;
        # the actual trainer scheduler may pad/truncate, but this is
        # the planned-step estimate the runner is committing to.
        effective_steps = max(
            1,
            (instr_count * self.config.epochs)
            // max(1, self.config.batch_size),
        )
        capture.log_decision(
            decision_type="hyperparameter_selection",
            decision=(
                f"Hyperparameters: lora_rank={self.config.lora_rank}, "
                f"lora_alpha={self.config.lora_alpha}, "
                f"learning_rate={self.config.learning_rate}, "
                f"epochs={self.config.epochs}, "
                f"batch_size={self.config.batch_size}, "
                f"seed={self.config.seed}."
            ),
            rationale=(
                f"Loaded per-base defaults from "
                f"Trainforge/training/configs/{self.base_model}.yaml for "
                f"{self.base_model!r}. With {instr_count} instruction pairs "
                f"× {self.config.epochs} epochs / batch_size="
                f"{self.config.batch_size} that yields ~{effective_steps} "
                f"effective SFT steps. lora_rank={self.config.lora_rank} / "
                f"lora_alpha={self.config.lora_alpha} = "
                f"{self.config.lora_alpha / max(1, self.config.lora_rank):.1f}x "
                f"scaling per the QLoRA stable recipe; "
                f"learning_rate={self.config.learning_rate} is the TRL SFT "
                f"baseline for sub-3B models; seed={self.config.seed} pinned "
                f"for reproducibility (max_seq_length="
                f"{self.config.max_seq_length})."
            ),
            alternatives_considered=[
                f"Higher rank (lora_rank={self.config.lora_rank * 2}): "
                f"rejected — doubles adapter weights without measurable gain "
                f"on a {instr_count}-pair corpus.",
                f"Higher LR (5x → "
                f"{self.config.learning_rate * 5:.0e}): rejected — "
                f"TRL/QLoRA literature reports loss instability above ~1e-3 "
                f"on small-corpus sub-3B fine-tunes.",
                f"More epochs (2x → {self.config.epochs * 2}): rejected — "
                f"~{effective_steps * 2} steps risks overfit on "
                f"{instr_count}-pair corpus; preferred regularization "
                f"is rank/alpha tuning.",
            ],
        )

    def _decide_run_dpo(self) -> tuple[bool, str]:
        """Gate the optional DPO chain on the size of preference_pairs.jsonl.

        Pre-Wave-91 we don't yet have an eval harness signaling whether
        DPO improves the run, so the decision is gated on data
        availability: <10 pairs is too few for stable DPO and we skip.
        """
        pref_path = self.course_dir / "training_specs" / "preference_pairs.jsonl"
        instr_path = self.course_dir / "training_specs" / "instruction_pairs.jsonl"
        pair_count = _count_jsonl_records(pref_path)
        instr_count = _count_jsonl_records(instr_path)
        if pair_count < 10:
            return False, (
                f"Preference pair count={pair_count} for course "
                f"{self.course_slug!r} is below the minimum 10 required for "
                f"stable DPO (SFT corpus={instr_count} pairs, base="
                f"{self.base_model!r}). Running SFT-only with "
                f"learning_rate={self.config.learning_rate}, epochs="
                f"{self.config.epochs}, lora_rank={self.config.lora_rank}; "
                f"DPO chain skipped (dry_run={self.dry_run})."
            )
        return True, (
            f"Preference pair count={pair_count} ≥ 10 for course "
            f"{self.course_slug!r} (SFT corpus={instr_count} pairs, base="
            f"{self.base_model!r}); chaining DPO after SFT at "
            f"learning_rate={self.config.learning_rate}, epochs="
            f"{self.config.epochs}, lora_rank={self.config.lora_rank} "
            f"to learn the misconception → correction preference signal "
            f"(dry_run={self.dry_run})."
        )

    def _log_eval_decision(
        self,
        capture: DecisionCapture,
        should_run_dpo: bool,
        rationale: str,
    ) -> None:
        pref_count = _count_jsonl_records(
            self.course_dir / "training_specs" / "preference_pairs.jsonl"
        )
        instr_count = _count_jsonl_records(
            self.course_dir / "training_specs" / "instruction_pairs.jsonl"
        )
        capture.log_decision(
            decision_type="eval_run_decision",
            decision=(
                f"Will{' ' if should_run_dpo else ' NOT '}chain DPO after SFT."
            ),
            rationale=rationale,
            alternatives_considered=[
                f"Force DPO regardless of pair count "
                f"(currently {pref_count}): rejected — TRL/DPO requires "
                f"≥10 pairs for stable preference gradients on "
                f"{self.base_model!r}.",
                f"Skip DPO unconditionally on {self.course_slug!r} "
                f"({instr_count} SFT pairs / {pref_count} preference pairs): "
                f"rejected — wastes the misconception → correction signal "
                f"the synthesizer emitted whenever pair_count ≥ 10.",
            ],
        )

    # ------------------------------------------------------------------ #
    # Dispatch + emit                                                     #
    # ------------------------------------------------------------------ #

    def _dispatch_training(
        self,
        run_dir: Path,
        run_dpo: bool,
    ) -> TrainingJobResult:
        spec = TrainingJobSpec(
            course_slug=self.course_slug,
            base_model=self.base_model,
            instruction_pairs_path=(
                self.course_dir / "training_specs" / "instruction_pairs.jsonl"
            ),
            preference_pairs_path=(
                self.course_dir / "training_specs" / "preference_pairs.jsonl"
            ),
            training_config=self.config.to_dict(),
            output_dir=run_dir,
            run_dpo=run_dpo,
        )
        return self.backend.run(spec)

    def _emit_model_card(
        self,
        run_dir: Path,
        model_id: str,
        provenance: Dict[str, str],
        adapter_path: Optional[Path],
        eval_scores: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write ``model_card.json`` validating against the Wave 89 schema.

        ``adapter_format`` defaults to ``safetensors`` (PEFT/LoRA
        adapter) — Wave 90 doesn't emit GGUF; that's a follow-up
        wave's concern.
        """
        card = {
            "model_id": model_id,
            "course_slug": self.course_slug,
            "base_model": {
                "name": self.spec.name,
                "revision": self.spec.default_revision,
                "huggingface_repo": self.spec.huggingface_repo,
            },
            "adapter_format": "safetensors",
            "training_config": self.config.to_dict(),
            "provenance": provenance,
            "created_at": _iso_now(),
        }
        # Drop the redundant base_model echo from training_config so
        # the schema's strict additionalProperties=false doesn't trip.
        card["training_config"].pop("base_model", None)

        if eval_scores is not None:
            # Filter to canonical keys the schema accepts so the
            # additionalProperties=false guard on eval_scores doesn't
            # reject the card.
            allowed_keys = {"faithfulness", "coverage", "baseline_delta"}
            card["eval_scores"] = {
                k: float(v) for k, v in eval_scores.items() if k in allowed_keys
            }

        card_path = run_dir / "model_card.json"
        # Atomic write so a crash mid-emit doesn't leave a half-card.
        tmp = card_path.with_suffix(card_path.suffix + ".tmp")
        tmp.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(card_path)
        return card_path

    # ------------------------------------------------------------------ #
    # Wave 92 — eval hook                                                 #
    # ------------------------------------------------------------------ #

    def _run_eval_harness(
        self,
        run_dir: Path,
        adapter_path: Optional[Path],
    ) -> Dict[str, Any]:
        """Invoke the SLM eval harness and return canonical eval scores.

        Wave 92: the eval harness is a hard dependency of a real
        (non-dry-run) training pass. If the harness fails to import
        or to run, we raise loudly rather than emit a card with empty
        eval_scores — a card without scores is worse than no card at
        all because it claims an unevaluated model is evaluated.
        """
        from Trainforge.eval.slm_eval_harness import SLMEvalHarness

        # Wire a model_callable from the adapter. Wave 92 leaves the
        # exact wiring (transformers.pipeline + PEFT load) to the
        # caller / future wave; for now the runner expects a backend
        # that has produced an adapter on disk and we surface a
        # NotImplementedError if no callable is configured. Tests
        # patch this method, so the production path being incomplete
        # doesn't block CI.
        raise NotImplementedError(
            "Wave 92: model_callable wiring from adapter to harness is "
            "deferred to a follow-up wave. Tests patch this method to "
            "exercise the integration. See plans/slm-training-2026-04-26.md "
            "Wave 92 deferred items."
        )

    def _save_decision_run_log(
        self,
        capture: DecisionCapture,
        run_dir: Path,
    ) -> Path:
        """Mirror the decision capture into the run dir as ``training_run.jsonl``.

        ``DecisionCapture`` already streams to ``training-captures/`` +
        the LibV2-mirrored capture dir; the run-dir mirror lets a
        consumer reading the model card find the rationale without
        knowing the project's capture conventions.
        """
        decisions_path = run_dir / "training_run.jsonl"
        with decisions_path.open("w", encoding="utf-8") as fh:
            for record in capture.decisions:
                fh.write(json.dumps(record, default=str) + "\n")
        return decisions_path


# ---------------------------------------------------------------------- #
# Module helpers                                                          #
# ---------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _iso_now() -> str:
    """ISO 8601 UTC timestamp with explicit ``Z`` suffix.

    The ``model_card.created_at`` schema uses ``format: date-time``
    which jsonschema accepts with the trailing ``Z``.
    """
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "TrainingRunner",
    "TrainingRunResult",
]
