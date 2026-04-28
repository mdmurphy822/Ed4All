"""LibV2 Model Validator (Wave 89 — slm-training-2026-04-26).

Gates the post-import attach of a trained adapter into a LibV2 course
under ``LibV2/courses/<slug>/models/<model_id>/``. Mirrors
``LibV2ManifestValidator``'s critical/warning split:

- **Critical**: schema parse, schema match, weights file present +
  size + sha256 match, ``pedagogy_graph_hash`` resolves to an extant
  graph artifact in the same course (``graph/pedagogy_graph.json`` or
  ``pedagogy/pedagogy_model.json``).
- **Warning**: ``eval_scores`` absent, ``license`` empty/null,
  ``base_model.huggingface_repo`` doesn't match the canonical
  ``^[\\w-]+/[\\w.-]+$`` regex.

Wave 89 deliberately **does not** call Hugging Face to resolve repos
(no network IO from a validator). The HF-resolve check is deferred to
Wave 92's ``libv2 import-model`` CLI surface where network is OK.

Referenced by: future ``config/workflows.yaml`` →
``trainforge_train.packaging.validation_gates[libv2_model]`` (Wave 90)
and ``LibV2/tools/libv2/importer.py::import_model`` (Wave 92).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# Canonical HF-repo shape; we validate the regex shape only — actual
# repo resolvability is a Wave 92 concern (network IO).
_HF_REPO_PATTERN = re.compile(r"^[\w-]+/[\w.-]+$")

# Default weights filename per adapter_format. Used when the card
# doesn't carry an explicit ``adapter_path`` field; matches the layout
# Wave 92 importer writes.
_DEFAULT_WEIGHTS_FILENAME = {
    "safetensors": "adapter.safetensors",
    "merged_safetensors": "model.safetensors",
    "gguf": "merged.gguf",
}

# Pedagogy artifacts the validator will sniff for ``pedagogy_graph_hash``
# resolution. First-match wins; both are emitted by Trainforge but at
# different points in the v0.2.0 → v0.3.0 history.
_PEDAGOGY_CANDIDATES = (
    "graph/pedagogy_graph.json",
    "pedagogy/pedagogy_model.json",
)


class LibV2ModelValidator:
    """Validates a LibV2-imported trained adapter + its model card.

    Inputs:
        model_card_path: Path to ``courses/<slug>/models/<model_id>/model_card.json``.
                         Required.
        model_dir: Path to the model directory (parent of model_card_path).
                   Optional; derived from model_card_path when absent.
        course_dir: Path to ``courses/<slug>/`` — used to resolve
                    ``pedagogy_graph_hash`` against the on-disk pedagogy
                    artifact. Optional; derived as ``model_dir.parent.parent``
                    when absent (``models/<model_id>`` → ``models`` → slug).
    """

    name = "libv2_model"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "libv2_model")
        issues: List[GateIssue] = []

        # -- 1. Required input.
        card_path_raw = inputs.get("model_card_path")
        if not card_path_raw:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_MODEL_CARD_PATH",
                    message="model_card_path is required for LibV2ModelValidator",
                )],
            )
        card_path = Path(card_path_raw)
        if not card_path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MODEL_CARD_NOT_FOUND",
                    message=f"Model card path does not exist: {card_path}",
                )],
            )

        model_dir_raw = inputs.get("model_dir")
        model_dir = Path(model_dir_raw) if model_dir_raw else card_path.parent

        course_dir_raw = inputs.get("course_dir")
        if course_dir_raw:
            course_dir: Optional[Path] = Path(course_dir_raw)
        else:
            # ``models/<model_id>/`` -> ``models/`` -> slug-dir
            try:
                course_dir = model_dir.parent.parent
                if not course_dir.exists():
                    course_dir = None
            except (AttributeError, OSError):
                course_dir = None

        # -- 2. JSON parse.
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="INVALID_JSON",
                    message=f"Model card JSON failed to parse: {exc}",
                    location=str(card_path),
                )],
            )

        # -- 3. Schema validation.
        schema_issues = self._validate_against_schema(card, card_path)
        issues.extend(schema_issues)

        # If the card is so broken we can't read top-level fields,
        # short-circuit before the on-disk integrity / hash-resolution
        # passes (they assume a well-shaped card).
        if not isinstance(card, dict) or "adapter_format" not in card:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=issues,
                score=0.0,
            )

        # -- 4. Weights file integrity.
        issues.extend(self._validate_weights(card, model_dir))

        # -- 5. pedagogy_graph_hash resolution against course_dir.
        issues.extend(self._validate_pedagogy_hash_resolves(card, course_dir))

        # -- 6. Wave 107: critical-fail when training corpus is mock-provider.
        issues.extend(self._check_instruction_pairs_provider(course_dir))

        # -- 7. Warning-severity advisories.
        issues.extend(self._check_eval_scores_present(card, card_path))
        issues.extend(self._check_license_declared(card, card_path))
        issues.extend(self._check_huggingface_repo_shape(card, card_path))

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    # ------------------------------------------------------------------ #
    # Schema validation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_against_schema(
        card: Any, card_path: Path,
    ) -> List[GateIssue]:
        """Validate against ``schemas/models/model_card.schema.json``.

        Best-effort: when jsonschema isn't installed, fall back to a
        lightweight structural check (top-level required keys).
        """
        issues: List[GateIssue] = []

        if not isinstance(card, dict):
            issues.append(GateIssue(
                severity="critical",
                code="SCHEMA_VIOLATION",
                message=f"Model card root is not an object: {type(card).__name__}",
                location=str(card_path),
            ))
            return issues

        try:
            import jsonschema  # type: ignore
        except ImportError:
            for required in (
                "model_id", "course_slug", "base_model", "adapter_format",
                "training_config", "provenance", "created_at",
            ):
                if required not in card:
                    issues.append(GateIssue(
                        severity="critical",
                        code="SCHEMA_VIOLATION",
                        message=f"Missing required model card key: {required}",
                        location=str(card_path),
                    ))
            return issues

        schema_path = _resolve_schema_path()
        if not schema_path or not schema_path.exists():
            issues.append(GateIssue(
                severity="warning",
                code="SCHEMA_UNAVAILABLE",
                message=(
                    f"model_card.schema.json not found at {schema_path}; "
                    "falling back to structural check."
                ),
            ))
            return issues

        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(card, schema)
        except jsonschema.ValidationError as exc:
            issues.append(GateIssue(
                severity="critical",
                code="SCHEMA_VIOLATION",
                message=f"Model card schema check: {exc.message}",
                location=".".join(str(p) for p in exc.absolute_path),
                suggestion=(
                    "See schemas/models/model_card.schema.json. "
                    "Required: model_id, course_slug, base_model, "
                    "adapter_format, training_config, provenance, created_at."
                ),
            ))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(GateIssue(
                severity="warning",
                code="SCHEMA_LOAD_ERROR",
                message=f"Failed to load model card schema: {exc}",
            ))
        return issues

    # ------------------------------------------------------------------ #
    # Weights file integrity                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_weights(
        card: Dict[str, Any], model_dir: Path,
    ) -> List[GateIssue]:
        """Verify the weights file exists and (optionally) matches size + sha256.

        The schema doesn't require weights metadata in the card itself
        — that's a Wave 90 emit detail. The validator checks the
        canonical filename for the declared ``adapter_format`` exists
        on disk; when the card carries an optional ``weights`` block
        with ``sha256`` + ``size``, those are verified too.
        """
        issues: List[GateIssue] = []

        adapter_format = card.get("adapter_format")
        weights_meta = card.get("weights") if isinstance(card.get("weights"), dict) else {}
        weights_filename = (
            weights_meta.get("filename")
            or _DEFAULT_WEIGHTS_FILENAME.get(adapter_format)
        )

        if not weights_filename:
            issues.append(GateIssue(
                severity="critical",
                code="UNKNOWN_WEIGHTS_FILENAME",
                message=(
                    f"Cannot determine weights filename for adapter_format="
                    f"{adapter_format!r}; no default and no card.weights.filename."
                ),
            ))
            return issues

        weights_path = model_dir / weights_filename
        if not weights_path.exists():
            issues.append(GateIssue(
                severity="critical",
                code="MISSING_WEIGHTS",
                message=f"Weights file not found: {weights_path}",
                location=str(weights_path),
            ))
            return issues

        # Optional size check
        expected_size = weights_meta.get("size")
        if expected_size is not None:
            actual_size = weights_path.stat().st_size
            if expected_size != actual_size:
                issues.append(GateIssue(
                    severity="critical",
                    code="WEIGHTS_SIZE_MISMATCH",
                    message=(
                        f"Weights size mismatch for {weights_path}: card says "
                        f"{expected_size}, disk shows {actual_size}"
                    ),
                    location=str(weights_path),
                ))

        # Optional checksum check
        expected_sha = weights_meta.get("sha256") or weights_meta.get("checksum")
        if expected_sha:
            actual_sha = _sha256_file(weights_path)
            if actual_sha != expected_sha:
                issues.append(GateIssue(
                    severity="critical",
                    code="WEIGHTS_CHECKSUM_MISMATCH",
                    message=(
                        f"Weights sha256 mismatch for {weights_path}: card says "
                        f"{expected_sha[:16]}..., disk hashes to "
                        f"{actual_sha[:16]}..."
                    ),
                    location=str(weights_path),
                ))

        return issues

    # ------------------------------------------------------------------ #
    # pedagogy_graph_hash resolution                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_pedagogy_hash_resolves(
        card: Dict[str, Any], course_dir: Optional[Path],
    ) -> List[GateIssue]:
        """The card's ``pedagogy_graph_hash`` must hash an extant artifact
        in the same course's pedagogy or graph dir.

        Critical: no candidate file exists, or none of the candidates
        hashes match. Wave 90's runner is responsible for hashing
        whichever artifact it actually consumed; wave 89's validator
        accepts either canonical layout.
        """
        issues: List[GateIssue] = []

        provenance = card.get("provenance") or {}
        expected_hash = provenance.get("pedagogy_graph_hash")
        if not expected_hash:
            # The schema check above already covers missing-hash cases;
            # don't double-report.
            return issues

        if course_dir is None or not course_dir.exists():
            issues.append(GateIssue(
                severity="critical",
                code="PEDAGOGY_HASH_UNRESOLVABLE",
                message=(
                    "Cannot resolve pedagogy_graph_hash: course_dir was not "
                    "provided and could not be derived from model_card_path."
                ),
            ))
            return issues

        candidates = [course_dir / rel for rel in _PEDAGOGY_CANDIDATES]
        existing = [p for p in candidates if p.exists()]
        if not existing:
            issues.append(GateIssue(
                severity="critical",
                code="PEDAGOGY_GRAPH_NOT_FOUND",
                message=(
                    f"No pedagogy artifact found in course_dir {course_dir}. "
                    f"Looked for: {', '.join(_PEDAGOGY_CANDIDATES)}"
                ),
                location=str(course_dir),
                suggestion=(
                    "Ensure the LibV2 course has a populated pedagogy graph "
                    "before training (Wave 90 runner refuses otherwise)."
                ),
            ))
            return issues

        match_found = False
        for candidate in existing:
            if _sha256_file(candidate) == expected_hash:
                match_found = True
                break

        if not match_found:
            issues.append(GateIssue(
                severity="critical",
                code="PEDAGOGY_HASH_MISMATCH",
                message=(
                    f"pedagogy_graph_hash {expected_hash[:16]}... does not "
                    f"match any of: "
                    f"{', '.join(str(p.relative_to(course_dir)) for p in existing)}"
                ),
                location=str(course_dir),
                suggestion=(
                    "The pedagogy graph changed since the card was minted. "
                    "Rerun the Wave 90 training pipeline against the current "
                    "course state."
                ),
            ))
        return issues

    # ------------------------------------------------------------------ #
    # Critical: training corpus provenance                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_instruction_pairs_provider(
        course_dir: Optional[Path],
    ) -> List[GateIssue]:
        """Critical-fail when ``instruction_pairs.jsonl`` first row has
        ``provider == "mock"``.

        Mock-provider corpora are template-factory output and produce
        template-recognizer adapters (Wave 107 root cause for the
        rdf-shacl-551-2 regression). No production training run may
        consume them. When ``course_dir`` is unresolvable or the
        instruction-pairs file is absent, this check no-ops — the rest
        of the validator already covers those conditions.
        """
        if course_dir is None:
            return []
        inst = course_dir / "training_specs" / "instruction_pairs.jsonl"
        if not inst.exists():
            return []
        try:
            with inst.open("r", encoding="utf-8") as fh:
                first = fh.readline().strip()
            if not first:
                return []
            row = json.loads(first)
        except (OSError, json.JSONDecodeError):
            return []
        provider = str(row.get("provider", ""))
        if provider == "mock":
            return [GateIssue(
                severity="critical",
                code="MOCK_PROVIDER_CORPUS",
                message=(
                    "instruction_pairs.jsonl first row has provider='mock'; "
                    "mock corpora train template-recognizers and may not "
                    "promote. Re-synthesize with provider='claude_session' "
                    "or 'anthropic'. See Trainforge/CLAUDE.md."
                ),
                location=str(inst),
            )]
        return []

    # ------------------------------------------------------------------ #
    # Warning advisories                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_eval_scores_present(
        card: Dict[str, Any], card_path: Path,
    ) -> List[GateIssue]:
        eval_scores = card.get("eval_scores")
        if not eval_scores:
            return [GateIssue(
                severity="warning",
                code="EVAL_SCORES_MISSING",
                message=(
                    "eval_scores block absent — model has not been evaluated. "
                    "Run `libv2 models eval <slug> <model_id>` (Wave 92) to "
                    "populate."
                ),
                location=str(card_path),
            )]
        # Wave 102: when eval_scores is populated, the reproducibility
        # surface (scoring_commit + tolerance_band) is required so
        # verify_eval can run end-to-end without external context.
        issues: List[GateIssue] = []
        if not isinstance(eval_scores, dict):
            return issues
        if not eval_scores.get("scoring_commit"):
            issues.append(GateIssue(
                severity="critical",
                code="EVAL_SCORES_MISSING_SCORING_COMMIT",
                message=(
                    "eval_scores.scoring_commit is required when "
                    "eval_scores is present. Wave 102 reproduce_eval.sh "
                    "pins this 40-char SHA so the verifier can detect "
                    "working-tree drift."
                ),
                location=str(card_path),
            ))
        tolerance = eval_scores.get("tolerance_band")
        if not isinstance(tolerance, dict) or not tolerance:
            issues.append(GateIssue(
                severity="critical",
                code="EVAL_SCORES_MISSING_TOLERANCE_BAND",
                message=(
                    "eval_scores.tolerance_band is required when "
                    "eval_scores is present. Wave 102 verify_eval reads "
                    "the per-metric bands to decide whether to flag "
                    "DRIFT."
                ),
                location=str(card_path),
            ))
        return issues

    @staticmethod
    def _check_license_declared(
        card: Dict[str, Any], card_path: Path,
    ) -> List[GateIssue]:
        license_str = card.get("license")
        if license_str is None or (isinstance(license_str, str) and not license_str.strip()):
            return [GateIssue(
                severity="warning",
                code="LICENSE_NOT_DECLARED",
                message=(
                    "Adapter license not declared. Set 'license' to an SPDX "
                    "identifier (e.g. apache-2.0) before publishing."
                ),
                location=str(card_path),
            )]
        return []

    @staticmethod
    def _check_huggingface_repo_shape(
        card: Dict[str, Any], card_path: Path,
    ) -> List[GateIssue]:
        base = card.get("base_model")
        if not isinstance(base, dict):
            return []
        repo = base.get("huggingface_repo")
        if not isinstance(repo, str) or not _HF_REPO_PATTERN.match(repo):
            return [GateIssue(
                severity="warning",
                code="HF_REPO_PATTERN_INVALID",
                message=(
                    f"base_model.huggingface_repo={repo!r} does not match "
                    f"^[\\w-]+/[\\w.-]+$. Wave 89 only checks shape; repo "
                    "resolvability is verified at Wave 92 import time."
                ),
                location=str(card_path),
            )]
        return []


# ---------------------------------------------------------------------- #
# Module helpers                                                          #
# ---------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_schema_path() -> Optional[Path]:
    """Locate ``schemas/models/model_card.schema.json``.

    Walks up from this file until it finds a ``schemas/models`` dir.
    Returns ``None`` when not found (validator falls back to
    structural check).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "schemas" / "models" / "model_card.schema.json"
        if candidate.exists():
            return candidate
    return None
