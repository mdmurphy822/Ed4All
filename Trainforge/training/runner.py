"""Wave 90 — :class:`TrainingRunner` orchestrates one end-to-end training run.

Reads the LibV2 course at ``LibV2/courses/<course_slug>/``:

    imscc_chunks/chunks.jsonl  (Phase 7c rename of corpus/chunks.jsonl)
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
:class:`lib.validators.libv2.model.LibV2ModelValidator` (Wave 89). When
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
from typing import Any, Dict, List, Mapping, Optional

from lib.decision_capture import DecisionCapture
from lib.licensing import (
    assert_checkpoint_license,
    assert_export_licenses,
    assert_nemotron_pin,
)
from lib.paths import LIBV2_COURSES, SCHEMAS_PATH
from lib.utils import sha256_file, write_jsonl

from Trainforge.training.base_models import BaseModelRegistry, BaseModelSpec
from Trainforge.training.compute_backend import (
    ComputeBackend,
    LocalBackend,
    TrainingJobResult,
    TrainingJobSpec,
    is_dpo_editorial_record,
)
from Trainforge.training.configs import (
    TrainingConfig,
    coerce_config_overrides,
    load_config,
)


logger = logging.getLogger(__name__)


class InsufficientPreferencePairsError(RuntimeError):
    """The corpus cannot support the DPO stage the run requires.

    Raised BEFORE any weight is trained, so an operator who asked for a
    DPO-tuned adapter never discovers at promotion time that they were
    quietly handed SFT-only weights.
    """


# ---------------------------------------------------------------------- #
# Provenance contract                                                     #
# ---------------------------------------------------------------------- #
#
# The card pins SHA-256 hashes of these six artifacts. Each tuple is
# (provenance-key, [candidate, ...]) where each candidate is either a
# course-relative ``str`` (resolved against ``self.course_dir``) or an
# absolute ``Path`` (used as-is). First-existing wins so we accept both
# the v0.2.0 and v0.3.0 LibV2 layouts.

# The canonical Courseforge vocabulary lives under the project-root
# schemas/context tree, not inside each LibV2 course.
# Without this fallback the runner silently substitutes the empty-bytes
# sha256 for vocabulary_ttl_hash and emits a model card whose hash
# doesn't pin the actual TTL the synthesizer consumed.
_VOCABULARY_TTL_CANONICAL = SCHEMAS_PATH / "context" / "courseforge_v1.vocabulary.ttl"

_PROVENANCE_SOURCES = (
    # Phase 7c: imscc_chunks/ is canonical; corpus/ retained for back-compat.
    ("chunks_hash", ["imscc_chunks/chunks.jsonl", "corpus/chunks.jsonl"]),
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
    # Holdout split for Tier-2 evaluation. The runner emits a stub split when
    # the evaluation submodule has not pre-built one; the real
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
        config_overrides: Optional[Mapping[str, Any]] = None,
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
            config_overrides: Optional PER-RUN override mapping (the
                ``--config-overrides`` route). Validated + type-coerced via
                :func:`coerce_config_overrides` — an unknown key or an
                out-of-range value raises here rather than surviving into a
                multi-hour run. Applied on top of a pre-resolved ``config``
                too, so an explicitly-supplied config can never silently
                swallow the operator's overrides. Recorded on the model card
                (``config_overrides``) because an adapter trained at a
                hand-picked rate is unreproducible if the rate is nowhere on
                disk.
        """
        self.course_slug = course_slug
        self.base_model = base_model
        self.dry_run = bool(dry_run)
        self.libv2_root = Path(libv2_root) if libv2_root else LIBV2_COURSES
        self.spec: BaseModelSpec = BaseModelRegistry.resolve(base_model)
        self.config_overrides: Dict[str, Any] = (
            coerce_config_overrides(config_overrides) if config_overrides else {}
        )
        self.config: TrainingConfig
        if config is not None:
            self.config = (
                config.merged(self.config_overrides)
                if self.config_overrides
                else config
            )
        else:
            self.config = load_config(
                base_model,
                course_overrides=config_overrides_path,
                overrides=self.config_overrides or None,
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
        self._assert_licensing_preflight()
        # SFT-D B3/B4: deterministically normalize the course's graph/ layout
        # BEFORE hashing provenance so the CRITICAL pedagogy_graph_hash
        # resolves (B3 re-emit) and the concept graph is where downstream
        # eval + retrieval expect it (B4 copy). Idempotent + fail-soft: a
        # course that already carries both artifacts is left byte-identical.
        self._ensure_graph_layout()

        provenance = self._compute_provenance()
        model_id = self._mint_model_id(provenance)
        run_dir = self._models_root / model_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Decision-capture stream for the trainforge-training phase lives at:
        #   runtime/training-captures/trainforge/<COURSE>/phase_trainforge-training/
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
                self._training_decision_capture = capture
                job_result = self._dispatch_training(run_dir, should_run_dpo)
                adapter_path = job_result.adapter_path
                metrics = dict(job_result.metrics)
                if not adapter_path.exists():
                    raise RuntimeError(
                        f"Backend reported success but adapter file "
                        f"is missing: {adapter_path}"
                    )

            # Emit model_card.json and training_run.jsonl before invoking the
            # evaluation harness so downstream evaluation failure cannot erase
            # the successful training run's provenance.
            #
            # The flow is now:
            #   1. _dispatch_training succeeds → adapter on disk
            #   2. emit model_card.json (eval_scores omitted)
            #   3. emit training_run.jsonl
            #   4. attempt eval; on success, REWRITE model_card.json
            #      with eval_scores folded in. On NotImplementedError
            #      (eval-bridge unwired), log warning and exit cleanly.
            card_path = self._emit_model_card(
                run_dir=run_dir,
                model_id=model_id,
                provenance=provenance,
                adapter_path=adapter_path,
                eval_scores=None,
            )

            # Mirror the decision capture into the run dir as
            # training_run.jsonl. Even if eval blows up below, the
            # decision log is preserved on disk.
            decisions_path = self._save_decision_run_log(capture, run_dir)

            eval_scores: Optional[Dict[str, Any]] = None
            if not self.dry_run:
                try:
                    eval_scores = self._run_eval_harness(run_dir, adapter_path)
                except (
                    NotImplementedError,
                    ImportError,
                    FileNotFoundError,
                ) as exc:
                    # Evaluation-bridge errors fall through to the no-score
                    # path so a successful training run
                    # never voids its provenance card on a downstream
                    # eval failure.
                    #   * NotImplementedError - compatibility boundary for stubs.
                    #   * ImportError - heavy ML deps missing
                    #     (CPU-only dev box; ed4all[training] not
                    #     installed).
                    #   * FileNotFoundError - adapter dir or course
                    #     artifacts missing.
                    logger.warning(
                        "TrainingRunner: eval harness skipped (%s). "
                        "Model card emitted without eval_scores; "
                        "adapter at %s is still trained and persisted.",
                        exc, adapter_path,
                    )
                    eval_scores = None

            # If evaluation succeeds, fold scores into a second model-card
            # write. The
            # _emit_model_card helper does an atomic tmpfile + rename,
            # so a partial overwrite never leaves a half-card on disk.
            if eval_scores is not None:
                # A7 — wire AblationRunner so headline_delta lands on disk
                # alongside eval_report. Best-effort: ablation failure
                # never voids the eval pass.
                try:
                    ablation_delta = self._run_ablation(
                        run_dir=run_dir,
                        adapter_path=adapter_path,
                    )
                    if ablation_delta is not None:
                        eval_scores["headline_delta"] = ablation_delta
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Ablation run failed; eval_scores written without "
                        "headline_delta. Adapter at %s still trained "
                        "and persisted.",
                        adapter_path,
                    )

                # A6 — re-read provenance after eval. The harness wrote
                # eval/holdout_split.json during run_all(); the first
                # _compute_provenance() at line 196 saw a stale or
                # missing file, so the model_card pinned an inaccurate
                # holdout_graph_hash. Re-hashing here closes the race.
                provenance = self._compute_provenance()

                card_path = self._emit_model_card(
                    run_dir=run_dir,
                    model_id=model_id,
                    provenance=provenance,
                    adapter_path=adapter_path,
                    eval_scores=eval_scores,
                )

                # A2 — invoke EvalGatingValidator inline. Both
                # `python -m Trainforge.train_course` and `ed4all run
                # trainforge_train` now enforce the gate; previously
                # only the workflow-orchestrator path fired it.
                self._enforce_eval_gate(run_dir=run_dir, capture=capture)
        finally:
            capture.save()

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

    def _ensure_graph_layout(self) -> None:
        """SFT-D B3/B4: normalize ``course_dir/graph/`` (best-effort).

        Delegates to :func:`Trainforge.training.graph_layout.ensure_graph_layout`
        which (B4) copies ``concept_graph/concept_graph_semantic.json`` into
        ``graph/`` when missing and (B3) deterministically re-emits
        ``graph/pedagogy_graph.json`` from the chunkset. Fail-soft: any
        unexpected error is logged and swallowed — the subsequent
        ``_compute_provenance`` still fails LOUD if the CRITICAL
        pedagogy_graph_hash truly can't resolve.
        """
        try:
            from Trainforge.training.graph_layout import ensure_graph_layout

            report = ensure_graph_layout(self.course_dir)
            if report.get("concept_graph_copied") or report.get(
                "pedagogy_graph_emitted"
            ):
                logger.info(
                    "TrainingRunner: graph-layout normalized for %s "
                    "(concept_graph_copied=%s, pedagogy_graph_emitted=%s).",
                    self.course_slug,
                    report.get("concept_graph_copied"),
                    report.get("pedagogy_graph_emitted"),
                )
        except Exception as exc:  # noqa: BLE001 - best-effort normalization
            logger.warning(
                "TrainingRunner: graph-layout normalization failed for %s "
                "(%s); proceeding to provenance (which fails loud on a truly "
                "missing pedagogy graph).",
                self.course_slug, exc,
            )

    def _assert_licensing_preflight(self) -> None:
        """Fail-closed licensing gates before a single weight is trained.

        SFT-C S6/S7. Three build invariants, all fail-closed:

        1. **Nemotron license-pin** — the roster still pins the NVIDIA
           Nemotron Open Model License (Dec 15 2025); a drift to the
           general NVIDIA OML fails the build.
        2. **Per-checkpoint LICENSE assertion at ingest** — the student
           base checkpoint's license is asserted (non-commercial licenses
           refused for a commercially-shipped adapter).
        3. **Export-time teacher filter** — every instruction / preference
           pair's teacher is asserted non-barred / non-claude / registered.

        Byte-identical for a license-clean corpus: the export filter only
        fires on a *positively identified* barred / claude / unregistered
        teacher (a legacy pair with no ``provider`` / ``generating_seat``
        field passes), and every SUPPORTED base model resolves in the
        roster. Canonical posture: ``docs/LICENSING.md``.
        """
        # 1. Nemotron license-pin / FAIL-BUILD-ON-RE-PIN guard.
        assert_nemotron_pin()

        # 2. Per-checkpoint LICENSE assertion at ingest (student base weights).
        assert_checkpoint_license(self.base_model, role="base_model")

        # 3. Export-time teacher filter over the pair corpus.
        specs_dir = self.course_dir / "training_specs"
        for spec_name in ("instruction_pairs.jsonl", "preference_pairs.jsonl"):
            assert_export_licenses(
                self._iter_pair_records(specs_dir / spec_name),
                source_desc=f"{self.course_slug}/{spec_name}",
            )

    @staticmethod
    def _iter_pair_records(path: Path):
        """Yield JSON pair dicts from a JSONL file (skip blank / bad lines)."""
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj

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
                out[key] = sha256_file(resolved)
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
                f"Per-run config overrides: "
                f"{dict(sorted(self.config_overrides.items())) or 'none'}. "
                f"Provenance hashes pin LibV2 artifacts so the card "
                f"is fully replayable post-hoc."
            ),
            alternatives_considered=[
                {
                    "option": "Reuse an existing model identifier",
                    "reason_rejected": (
                        f"This run resolved model_id={model_id!r} from current "
                        f"provenance for course={self.course_slug!r}; reusing "
                        "another identifier would merge distinct training runs."
                    ),
                },
                {
                    "option": "Skip dry-run artifact scaffolding",
                    "reason_rejected": (
                        f"dry_run={self.dry_run} still requires replayable "
                        f"artifacts under {run_dir.name!r} so planning and card "
                        "emission can be verified without loading a model."
                    ),
                },
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
                {
                    "option": "Select a different registered base model",
                    "reason_rejected": (
                        f"The requested base={self.base_model!r} resolves to "
                        f"{self.spec.huggingface_repo!r} for {instr_count} "
                        f"instruction pairs at lora_rank={self.config.lora_rank}; "
                        f"supported alternatives are "
                        f"{BaseModelRegistry.list_supported()}."
                    ),
                },
                {
                    "option": "Use the repository's floating main revision",
                    "reason_rejected": (
                        f"The selected revision={self.spec.default_revision!r} "
                        f"pins {self.spec.huggingface_repo!r}; floating main "
                        "would make model-card provenance change upstream."
                    ),
                },
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
                {
                    "option": f"Double LoRA rank to {self.config.lora_rank * 2}",
                    "reason_rejected": (
                        f"The configured rank={self.config.lora_rank} and "
                        f"alpha={self.config.lora_alpha} match the selected "
                        f"base recipe for this {instr_count}-pair corpus."
                    ),
                },
                {
                    "option": (
                        f"Raise learning rate to "
                        f"{self.config.learning_rate * 5:.0e}"
                    ),
                    "reason_rejected": (
                        f"The base recipe selects learning_rate="
                        f"{self.config.learning_rate} for "
                        f"{self.base_model!r}; the higher value is outside the "
                        "pinned run configuration."
                    ),
                },
                {
                    "option": f"Double epochs to {self.config.epochs * 2}",
                    "reason_rejected": (
                        f"Configured epochs={self.config.epochs} yields about "
                        f"{effective_steps} steps from {instr_count} pairs; "
                        "doubling would exceed the selected replayable recipe."
                    ),
                },
            ],
        )

    def _decide_run_dpo(self) -> tuple[bool, str]:
        """Gate the optional DPO chain on high-signal preference pairs."""
        pref_path = self.course_dir / "training_specs" / "preference_pairs.jsonl"
        instr_path = self.course_dir / "training_specs" / "instruction_pairs.jsonl"
        pair_count = _count_jsonl_records(pref_path)
        filtered_count = _count_dpo_eligible_records(
            pref_path,
            str(self.config.dpo_preference_filter),
        )
        min_pairs = int(self.config.min_dpo_pairs)
        instr_count = _count_jsonl_records(instr_path)
        if filtered_count < min_pairs:
            # `dpo_fail_hard` means "a DPO problem kills the run rather than
            # silently shipping SFT-only weights". LocalBackend already
            # enforces that for the same below-floor condition
            # (compute_backend.py::_run_local raises when the filtered count
            # is short), but it only ever sees that branch when run_dpo is
            # True -- so returning False here short-circuited the loud guard
            # and degraded to SFT-only exactly where it was meant to stop.
            # Honour the flag on this branch too; set dpo_fail_hard=false to
            # opt in to SFT-only explicitly.
            if bool(self.config.dpo_fail_hard):
                raise InsufficientPreferencePairsError(
                    f"DPO is required (dpo_fail_hard=true) but the filtered "
                    f"preference-pair count={filtered_count} "
                    f"(raw={pair_count}, "
                    f"filter={self.config.dpo_preference_filter!r}) for "
                    f"course {self.course_slug!r} is below min_dpo_pairs="
                    f"{min_pairs}. Refusing to train an SFT-only adapter "
                    f"and label it complete. Either synthesize more "
                    f"admissible preference pairs, or opt in to SFT-only "
                    f"explicitly with "
                    f"--config-overrides dpo_fail_hard=false."
                )
            return False, (
                f"Filtered DPO preference pair count={filtered_count} "
                f"(raw={pair_count}, filter={self.config.dpo_preference_filter!r}) "
                f"for course {self.course_slug!r} is below min_dpo_pairs="
                f"{min_pairs} (SFT corpus={instr_count} pairs, base="
                f"{self.base_model!r}). Running SFT-only with "
                f"learning_rate={self.config.learning_rate}, epochs="
                f"{self.config.epochs}, lora_rank={self.config.lora_rank}; "
                f"DPO chain skipped (dry_run={self.dry_run})."
            )
        return True, (
            f"Filtered DPO preference pair count={filtered_count} "
            f"(raw={pair_count}, filter={self.config.dpo_preference_filter!r}) "
            f"meets min_dpo_pairs={min_pairs} for course {self.course_slug!r} "
            f"(SFT corpus={instr_count} pairs, base={self.base_model!r}); "
            f"chaining DPO after SFT at "
            f"learning_rate={self.config.learning_rate}, epochs="
            f"{self.config.epochs}, lora_rank={self.config.lora_rank} "
            f"to learn the curated misconception/correction preference signal "
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
        filtered_pref_count = _count_dpo_eligible_records(
            self.course_dir / "training_specs" / "preference_pairs.jsonl",
            str(self.config.dpo_preference_filter),
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
                {
                    "option": "Force DPO regardless of eligible pair count",
                    "reason_rejected": (
                        f"raw={pref_count} and filtered={filtered_pref_count} "
                        f"must be evaluated against min_dpo_pairs="
                        f"{self.config.min_dpo_pairs} with filter="
                        f"{self.config.dpo_preference_filter!r} for "
                        f"base={self.base_model!r}."
                    ),
                },
                {
                    "option": "Skip DPO unconditionally",
                    "reason_rejected": (
                        f"Course={self.course_slug!r} has "
                        f"filtered_count={filtered_pref_count}, "
                        f"raw_count={pref_count}, and SFT_count={instr_count}; "
                        "the configured threshold must decide whether the "
                        "curated preference signal is usable."
                    ),
                },
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
            extra={
                "course_dir": self.course_dir,
                "decision_capture": getattr(
                    self, "_training_decision_capture", None,
                ),
            },
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
            # ``to_card_dict`` filters to the schema-accepted keys — drops the
            # redundant base_model echo AND the S8 orchestration knobs
            # (completion_only_loss / save_total_limit / …) so the schema's
            # strict additionalProperties=false guard doesn't trip.
            "training_config": self.config.to_card_dict(),
            "provenance": provenance,
            "created_at": _iso_now(),
        }

        # PER-RUN overrides, recorded verbatim. ``training_config`` above
        # carries the EFFECTIVE values but not which of them the operator
        # chose by hand; without this block an adapter trained at a canary-
        # selected dpo_learning_rate is indistinguishable from one that
        # inherited a checked-in YAML, and the run is not reproducible.
        # Omitted entirely when empty, so an unflagged run's card is
        # byte-identical to every card written before this field existed.
        if self.config_overrides:
            card["config_overrides"] = dict(sorted(self.config_overrides.items()))

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
    # Evaluation hook                                                     #
    # ------------------------------------------------------------------ #

    def _run_eval_harness(
        self,
        run_dir: Path,
        adapter_path: Optional[Path],
    ) -> Dict[str, Any]:
        """Invoke the SLM eval harness and return canonical eval scores.

        Build an
        :class:`Trainforge.eval.retrieval.adapter_callable.AdapterCallable`
        around the saved adapter dir, hand it to
        :class:`SLMEvalHarness`, parse the resulting
        ``eval_report.json``, and (optionally) run the
        ``lm-evaluation-harness`` generic-benchmark sweep on the side.
        Returns a dict shaped to drop into
        ``model_card.json::eval_scores`` (the runner filters to the
        canonical keys before writing).

        Failure modes:
        * Adapter dir missing or unreadable -> ``FileNotFoundError``.
        * Heavy ML deps not installed -> ``ImportError``; the runner's
          calling try/except converts this into the same fall-back
          path as Wave 100's NotImplementedError did
          (model_card emitted without ``eval_scores``).
        """
        import os

        from Trainforge.eval.retrieval.adapter_callable import AdapterCallable
        from Trainforge.eval.eval_config import load_eval_config
        from Trainforge.eval.publication.hf_model_index import write_hf_readme
        from Trainforge.eval.runners.slm_eval_harness import SLMEvalHarness

        if adapter_path is None:
            raise FileNotFoundError(
                "Wave 101: cannot run eval harness without a saved "
                "adapter (adapter_path is None)."
            )
        # The adapter file lives at run_dir/adapter_model.safetensors;
        # AdapterCallable wants the directory.
        adapter_dir = Path(adapter_path).parent if Path(adapter_path).is_file() else Path(adapter_path)
        course_path = self.course_dir
        loaded_eval_config = load_eval_config(course_path)
        eval_cfg = loaded_eval_config.config

        callable_kwargs: Dict[str, Any] = {
            "base_model_short_name": self.spec.name,
            "max_new_tokens": int(eval_cfg.get("max_new_tokens", 256)),
            "temperature": float(eval_cfg.get("temperature", 0.0)),
            "top_p": float(eval_cfg.get("top_p", 1.0)),
            "seed": int(eval_cfg.get("seed", self.config.seed)),
            "revision": self.spec.default_revision,
            "use_4bit": bool(self.config.use_4bit),
        }
        logger.info(
            "TrainingRunner: eval generation config loaded from %s "
            "(max_new_tokens=%s, temperature=%s, top_p=%s, seed=%s).",
            loaded_eval_config.config_path,
            callable_kwargs["max_new_tokens"],
            callable_kwargs["temperature"],
            callable_kwargs["top_p"],
            callable_kwargs["seed"],
        )
        adapter_callable = AdapterCallable(
            adapter_dir=adapter_dir,
            base_model_repo=self.spec.huggingface_repo,
            **callable_kwargs,
        )

        harness = SLMEvalHarness(
            course_path=course_path,
            model_callable=adapter_callable,
        )
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        report_path = harness.run_all(
            output_path=eval_dir / "eval_report.json",
        )
        eval_report = json.loads(report_path.read_text(encoding="utf-8"))

        # Render the HF README alongside eval_report.json. Failures
        # here shouldn't void the eval pass - log + continue.
        try:
            # Read the just-emitted (no-eval-scores) model card for the
            # provenance + license fields the README pulls in.
            card_path = run_dir / "model_card.json"
            model_card = (
                json.loads(card_path.read_text(encoding="utf-8"))
                if card_path.exists() else {}
            )
            write_hf_readme(
                run_dir=run_dir,
                eval_report=eval_report,
                course_slug=self.course_slug,
                base_model=self.base_model,
                model_id=Path(run_dir).name,
                model_card=model_card,
                base_model_repo=self.spec.huggingface_repo,
                # Pass the LibV2 course directory so write_hf_readme reads
                # classification.tags from
                # manifest.json instead of substring-sniffing the slug.
                course_path=course_path,
            )
        except Exception:  # noqa: BLE001 - README is best-effort
            logger.exception(
                "write_hf_readme failed; eval_report.json "
                "still on disk."
            )

        # Optional: lm-eval generic-benchmark sweep when explicitly
        # opted in via env var. Default off because a 3-task sweep
        # stays bounded enough for a routine smoke check.
        lm_eval_summary: Optional[Dict[str, Any]] = None
        if os.environ.get("LM_EVAL_ENABLED", "").lower() == "true":
            try:
                from Trainforge.eval.lm_eval_wrapper import (
                    run_lm_eval,
                    summarize_lm_eval,
                )
                results_path = run_lm_eval(
                    adapter_dir=adapter_dir,
                    base_model_repo=self.spec.huggingface_repo,
                    run_dir=run_dir,
                )
                if results_path is not None:
                    lm_eval_summary = summarize_lm_eval(results_path)
            except Exception:  # noqa: BLE001 - optional telemetry
                logger.exception(
                    "Wave 101: lm-eval sweep failed; main eval scores "
                    "still recorded."
                )

        eval_scores: Dict[str, Any] = {}
        if "faithfulness" in eval_report and eval_report["faithfulness"] is not None:
            eval_scores["faithfulness"] = eval_report["faithfulness"]
        if "coverage" in eval_report and eval_report["coverage"] is not None:
            eval_scores["coverage"] = eval_report["coverage"]
        if "baseline_delta" in eval_report and eval_report["baseline_delta"] is not None:
            eval_scores["baseline_delta"] = eval_report["baseline_delta"]
        if lm_eval_summary:
            # The runner filters to the canonical keys before writing
            # the card so the schema's additionalProperties=false
            # guard isn't broken; lm_eval_summary is informational
            # and lands in the run dir's lm_eval_results/ instead.
            eval_scores["lm_eval_summary"] = lm_eval_summary
        return eval_scores

    def _run_ablation(
        self,
        run_dir: Path,
        adapter_path: Optional[Path],
    ) -> Optional[Dict[str, Any]]:
        """Wire A7: run AblationRunner so ablation_report.json + the
        headline_delta block (hallucination_reduction_pct, source-grounded
        lift, accuracy lift) land alongside eval_report.json.

        Returns the headline_delta dict (or None if ablation skipped /
        failed). Best-effort: an exception in here is caught by the
        caller and the run still ships its eval_scores card.
        """
        if adapter_path is None:
            return None
        if os.environ.get("ED4ALL_SKIP_ABLATION", "").lower() in ("1", "true"):
            logger.info(
                "TrainingRunner: ED4ALL_SKIP_ABLATION set; ablation "
                "skipped (no headline_delta on this run)."
            )
            return None

        from Trainforge.eval.runners.ablation_runner import AblationRunner, AblationSetup
        from Trainforge.eval.retrieval.adapter_callable import (
            AdapterCallable,
            AdapterDisabledCallable,
        )
        from Trainforge.eval.eval_config import load_eval_config
        from Trainforge.eval.retrieval.rag_callable import RAGCallable

        adapter_dir = (
            Path(adapter_path).parent
            if Path(adapter_path).is_file()
            else Path(adapter_path)
        )
        course_path = self.course_dir
        loaded_eval_config = load_eval_config(course_path)
        eval_cfg = loaded_eval_config.config

        callable_kwargs: Dict[str, Any] = {
            "base_model_short_name": self.spec.name,
            "max_new_tokens": int(eval_cfg.get("max_new_tokens", 256)),
            "temperature": float(eval_cfg.get("temperature", 0.0)),
            "top_p": float(eval_cfg.get("top_p", 1.0)),
            "seed": int(eval_cfg.get("seed", self.config.seed)),
            "revision": self.spec.default_revision,
            "use_4bit": bool(self.config.use_4bit),
        }

        adapter_callable = AdapterCallable(
            adapter_dir=adapter_dir,
            base_model_repo=self.spec.huggingface_repo,
            **callable_kwargs,
        )
        # Reuse one resident BF16 base for both arms so model weights,
        # activations, and backend workspaces stay within available memory.
        base_callable = AdapterDisabledCallable(adapter_callable)
        adapter_rag = RAGCallable(
            base_callable=adapter_callable,
            course_slug=self.course_slug,
            eval_config=loaded_eval_config,
        )
        base_rag = RAGCallable(
            base_callable=base_callable,
            course_slug=self.course_slug,
            eval_config=loaded_eval_config,
        )

        setups = [
            AblationSetup(setup="base", callable=base_callable),
            AblationSetup(setup="base+rag", callable=base_rag, rag_callable=base_rag),
            AblationSetup(setup="adapter", callable=adapter_callable),
            AblationSetup(
                setup="adapter+rag",
                callable=adapter_rag,
                rag_callable=adapter_rag,
            ),
        ]
        runner = AblationRunner(
            course_path=course_path,
            setups=setups,
            eval_config=loaded_eval_config,
        )
        ablation_path = runner.run(
            output_path=run_dir / "eval" / "ablation_report.json",
        )
        report = json.loads(ablation_path.read_text(encoding="utf-8"))
        return report.get("headline_delta")

    def _enforce_eval_gate(
        self,
        run_dir: Path,
        capture: DecisionCapture,
    ) -> None:
        """A2: run EvalGatingValidator inline so direct-CLI training
        runs are gated. Logs the result; raises on critical failure
        unless ``ED4ALL_GATE_ADVISORY=true`` is set.

        The gate already fail-louds on missing eval_report.json
        (EVAL_REPORT_NOT_FOUND), faithfulness regression, yes-bias,
        no-bias drop, and per-property accuracy floor. This wiring
        ensures the gate ACTUALLY FIRES regardless of how the runner
        was invoked.
        """
        from lib.validators.eval_gating import EvalGatingValidator

        validator = EvalGatingValidator()
        result = validator.validate({
            "gate_id": "eval_gating",
            "model_dir": str(run_dir),
            "capture": capture,
        })
        critical_count = sum(1 for i in result.issues if i.severity == "critical")
        warn_count = sum(1 for i in result.issues if i.severity == "warning")
        logger.info(
            "EvalGatingValidator: passed=%s critical=%d warning=%d",
            result.passed, critical_count, warn_count,
        )
        for issue in result.issues:
            logger.log(
                logging.ERROR if issue.severity == "critical" else logging.WARNING,
                "EvalGatingValidator: [%s] %s — %s",
                issue.severity, issue.code, issue.message,
            )
        if not result.passed:
            advisory = os.environ.get("ED4ALL_GATE_ADVISORY", "").lower() in ("1", "true")
            if not advisory:
                codes = ", ".join(
                    i.code for i in result.issues if i.severity == "critical"
                )
                raise RuntimeError(
                    f"EvalGatingValidator blocked promotion: {codes}. "
                    "Set ED4ALL_GATE_ADVISORY=true to log-only and ship "
                    "the run dir anyway."
                )

    def _save_decision_run_log(
        self,
        capture: DecisionCapture,
        run_dir: Path,
    ) -> Path:
        """Mirror the decision capture into the run dir as ``training_run.jsonl``.

        ``DecisionCapture`` already streams to ``runtime/training-captures/`` +
        the LibV2-mirrored capture dir; the run-dir mirror lets a
        consumer reading the model card find the rationale without
        knowing the project's capture conventions.
        """
        decisions_path = run_dir / "training_run.jsonl"
        write_jsonl(
            decisions_path,
            capture.decisions,
            default=str,
            sort_keys=False,
            atomic=False,
        )
        return decisions_path


# ---------------------------------------------------------------------- #
# Module helpers                                                          #
# ---------------------------------------------------------------------- #


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _count_dpo_eligible_records(path: Path, mode: str) -> int:
    if not path.exists():
        return 0
    if mode in ("", "all", None):
        return _count_jsonl_records(path)
    if mode != "editorial_or_misconception":
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # LOCKSTEP: the SAME predicate the filter uses
            # (compute_backend.is_dpo_editorial_record). Re-implementing the
            # membership test here is how the count and the filter drift, and
            # a drift means "run DPO" followed by an empty filtered set, which
            # with dpo_fail_hard=True kills the run after SFT succeeded.
            if is_dpo_editorial_record(rec):
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
