"""Wave 101 - HuggingFace model-index converter for eval reports.

Converts the Wave 92 ``eval_report.json`` shape produced by
:class:`Trainforge.eval.slm_eval_harness.SLMEvalHarness` into the
HuggingFace Hub ``model-index`` results[] schema, then renders a
README.md with the corresponding YAML frontmatter so the upload-target
adapter shows leaderboard-readable scores on its HF model page.

Mapping (Wave 101):

* Tier-1 syntactic checks    -> task=text-generation, metric=accuracy
* Tier-2 invariants          -> task=text-generation, metric=accuracy
                                (one entry per invariant, descriptive
                                metric.name)
* Faithfulness               -> task=text-generation, metric=f1
                                (closest standard taxonomy match for
                                "answer-vs-graph alignment")
* Calibration ECE            -> metric.type=expected_calibration_error
                                (HF accepts custom string types)
* Baseline delta             -> metric.type=accuracy_delta (custom)

The ``write_hf_readme`` helper composes a complete README.md with:
* YAML frontmatter (model-index + base_model + library_name +
  tags + license).
* Body: Training Data / Evaluation / Limitations / Provenance.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


logger = logging.getLogger(__name__)


_DATASET_NAMESPACE = "ed4all"
_DATASET_SPLIT = "holdout"


def eval_report_to_model_index(
    eval_report: Dict[str, Any],
    course_slug: str,
    base_model: str,
    model_id: str,
) -> List[Dict[str, Any]]:
    """Convert a Trainforge eval_report dict into HF ``model-index`` results[].

    Args:
        eval_report: The dict shape emitted by
            :class:`SLMEvalHarness.run_all` (parsed from
            ``<run_dir>/eval_report.json``). Expected keys: any subset
            of ``faithfulness``, ``coverage``, ``baseline_delta``,
            ``calibration_ece``, ``per_tier``, ``per_invariant``.
        course_slug: LibV2 course slug -- used as the dataset name
            namespace (``ed4all/<slug>``).
        base_model: Short base name (e.g. ``"qwen2.5-1.5b"``).
        model_id: The minted model_id (used as ``name`` in the
            single result entry; HF wraps each result in this name).

    Returns:
        List of one dict shaped per the HF ``model-index`` schema:
        ``[{"name": ..., "results": [...]}]`` semantics expressed as
        a flat results list. Callers wrap this in
        ``[{"name": model_id, "results": <list>}]`` when emitting
        frontmatter (see :func:`write_hf_readme`).
    """
    results: List[Dict[str, Any]] = []

    dataset = {
        "type": f"{_DATASET_NAMESPACE}/{course_slug}",
        "name": f"{course_slug} (LibV2 holdout)",
        "split": _DATASET_SPLIT,
    }
    task = {"type": "text-generation", "name": "Text Generation"}

    # --- Faithfulness (mapped to f1; closest standard metric) ----- #
    if "faithfulness" in eval_report and eval_report["faithfulness"] is not None:
        results.append({
            "task": dict(task),
            "dataset": dict(dataset),
            "metrics": [
                {
                    "type": "f1",
                    "name": "Faithfulness (KG-anchored)",
                    "value": _round(eval_report["faithfulness"]),
                },
            ],
        })

    # --- Coverage (overall pass-rate proxy) ----------------------- #
    if "coverage" in eval_report and eval_report["coverage"] is not None:
        results.append({
            "task": dict(task),
            "dataset": dict(dataset),
            "metrics": [
                {
                    "type": "accuracy",
                    "name": "Coverage (Tier-1 x Tier-2 pass rate)",
                    "value": _round(eval_report["coverage"]),
                },
            ],
        })

    # --- Per-invariant (Tier-2) ----------------------------------- #
    per_inv = eval_report.get("per_invariant") or {}
    for inv_name, inv_payload in sorted(per_inv.items()):
        pass_rate = _extract_pass_rate(inv_payload)
        if pass_rate is None:
            continue
        results.append({
            "task": dict(task),
            "dataset": dict(dataset),
            "metrics": [
                {
                    "type": "accuracy",
                    "name": f"{inv_name}_pass_rate",
                    "value": _round(pass_rate),
                },
            ],
        })

    # --- Per-tier (Tier-1 syntactic + Tier-3 key-term precision) -- #
    per_tier = eval_report.get("per_tier") or {}
    if "syntactic_pass_rate" in per_tier and per_tier["syntactic_pass_rate"] is not None:
        results.append({
            "task": dict(task),
            "dataset": dict(dataset),
            "metrics": [
                {
                    "type": "accuracy",
                    "name": "syntactic_pass_rate",
                    "value": _round(per_tier["syntactic_pass_rate"]),
                },
            ],
        })

    kt = per_tier.get("key_term_precision") or {}
    if "required_element_precision" in kt and kt["required_element_precision"] is not None:
        results.append({
            "task": dict(task),
            "dataset": dict(dataset),
            "metrics": [
                {
                    "type": "accuracy",
                    "name": "key_term_required_element_precision",
                    "value": _round(kt["required_element_precision"]),
                },
            ],
        })

    # --- Calibration ECE (custom HF metric type) ------------------ #
    ece = eval_report.get("calibration_ece")
    if ece is None:
        ece = (per_tier.get("calibration") or {}).get("ece")
    if ece is not None:
        results.append({
            "task": dict(task),
            "dataset": dict(dataset),
            "metrics": [
                {
                    "type": "expected_calibration_error",
                    "name": "Calibration ECE",
                    "value": _round(ece),
                },
            ],
        })

    # --- Baseline delta (custom HF metric type) ------------------- #
    if "baseline_delta" in eval_report and eval_report["baseline_delta"] is not None:
        results.append({
            "task": dict(task),
            "dataset": dict(dataset),
            "metrics": [
                {
                    "type": "accuracy_delta",
                    "name": "Baseline delta (trained - base)",
                    "value": _round(eval_report["baseline_delta"]),
                },
            ],
        })

    logger.info(
        "eval_report_to_model_index: emitted %d result entries for %s "
        "(course=%s, base=%s)",
        len(results), model_id, course_slug, base_model,
    )
    return results


def write_hf_readme(
    run_dir: Path,
    eval_report: Dict[str, Any],
    course_slug: str,
    base_model: str,
    model_id: str,
    model_card: Dict[str, Any],
    *,
    base_model_repo: Optional[str] = None,
    extra_tags: Optional[List[str]] = None,
    ablation_report: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render ``<run_dir>/README.md`` with HF-format YAML frontmatter.

    Args:
        run_dir: ``LibV2/courses/<slug>/models/<model_id>/`` -- the
            directory the README is written into. Becomes the upload
            target for ``huggingface-cli upload``.
        eval_report: Wave 92 eval_report shape (see
            :func:`eval_report_to_model_index`).
        course_slug: LibV2 course slug.
        base_model: Short base name (e.g. ``"qwen2.5-1.5b"``) used for
            the ``base_model:`` frontmatter key.
        model_id: The minted ``<slug>-<base>-<8hex>`` identifier.
        model_card: The Wave 89 model_card.json dict; we read
            ``provenance`` (7 hashes) and ``license`` from here.
        base_model_repo: Optional HF repo identifier for the base
            (e.g. ``"Qwen/Qwen2.5-1.5B"``). When None the registry
            is used.
        extra_tags: Optional extra HF tags appended to the default
            ``["education", "qlora", "peft"]``.

    Returns:
        Path to the written README.md.
    """
    results = eval_report_to_model_index(
        eval_report=eval_report,
        course_slug=course_slug,
        base_model=base_model,
        model_id=model_id,
    )

    if base_model_repo is None:
        try:
            from Trainforge.training.base_models import BaseModelRegistry
            base_model_repo = BaseModelRegistry.resolve(base_model).huggingface_repo
        except (KeyError, ImportError):
            base_model_repo = base_model

    tags = ["education", "qlora", "peft", "trainforge", course_slug]
    # Heuristic: if the slug mentions semantic-web vocabularies tag for
    # discoverability on the HF Hub.
    slug_lower = course_slug.lower()
    if "rdf" in slug_lower or "shacl" in slug_lower or "sparql" in slug_lower:
        tags.extend(["rdf", "shacl"])
    if extra_tags:
        for t in extra_tags:
            if t not in tags:
                tags.append(t)

    license_value = model_card.get("license") or "apache-2.0"

    frontmatter: Dict[str, Any] = {
        "library_name": "peft",
        "base_model": base_model_repo,
        "tags": tags,
        "license": license_value,
        "model-index": [
            {
                "name": model_id,
                "results": results,
            },
        ],
    }

    body = _render_body(
        eval_report=eval_report,
        course_slug=course_slug,
        base_model=base_model,
        model_id=model_id,
        model_card=model_card,
        ablation_report=ablation_report,
    )

    yaml_block = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )

    content = "---\n" + yaml_block + "---\n\n" + body

    readme_path = Path(run_dir) / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _round(value: Any) -> float:
    """Coerce + round to 4 dp; HF accepts numeric metric values."""
    return round(float(value), 4)


def _extract_pass_rate(payload: Any) -> Optional[float]:
    """Extract a scalar pass_rate from per-invariant payload shapes.

    The Wave 92 invariants emit dicts like::

        {"pass_rate": 0.83, "scored": 30, "passed": 25, ...}

    Some Tier-3 evaluators (disambiguation) follow the same shape;
    others (key_term_precision) embed ``required_element_precision``
    instead. This helper returns ``pass_rate`` when present.
    """
    if not isinstance(payload, dict):
        return None
    for key in ("pass_rate", "accuracy"):
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return None
    return None


def _render_body(
    eval_report: Dict[str, Any],
    course_slug: str,
    base_model: str,
    model_id: str,
    model_card: Dict[str, Any],
    ablation_report: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the README body (post-frontmatter) sections."""
    provenance = model_card.get("provenance") or {}
    training_config = model_card.get("training_config") or {}

    lines: List[str] = []
    lines.append(f"# {model_id}")
    lines.append("")
    lines.append(
        f"QLoRA-fine-tuned **{base_model}** adapter trained on the "
        f"**{course_slug}** LibV2 course corpus via the Ed4All "
        f"Trainforge pipeline."
    )
    lines.append("")

    # --- Training Data ---------------------------------------- #
    lines.append("## Training Data")
    lines.append("")
    lines.append(
        f"- **LibV2 course slug**: `{course_slug}`"
    )
    lines.append(
        f"- **Base model**: `{base_model}`"
    )
    lines.append(
        f"- **Adapter format**: `{model_card.get('adapter_format', 'safetensors')}`"
    )
    lines.append(
        f"- **Training config**: lora_rank={training_config.get('lora_rank')}, "
        f"lora_alpha={training_config.get('lora_alpha')}, "
        f"epochs={training_config.get('epochs')}, "
        f"learning_rate={training_config.get('learning_rate')}, "
        f"batch_size={training_config.get('batch_size')}, "
        f"max_seq_length={training_config.get('max_seq_length')}, "
        f"seed={training_config.get('seed')}"
    )
    lines.append("")

    # --- Evaluation ------------------------------------------- #
    lines.append("## Evaluation")
    lines.append("")
    # Wave 102: open with the verbatim thesis statement so any reader
    # who only skims the README sees the procurement claim up front.
    lines.append(
        "> This benchmark evaluates whether a domain adapter improves "
        "grounded reasoning over structured educational knowledge "
        "packages, using held-out questions, expected evidence chunks, "
        "and reproducible scoring scripts."
    )
    lines.append("")
    lines.append(
        f"Evaluated on the held-out split of the {course_slug} corpus "
        f"using the Trainforge SLM eval harness "
        f"(profile: `{eval_report.get('profile', 'unknown')}`). "
        f"All scores below are also published in machine-readable "
        f"form via the `model-index` frontmatter."
    )
    lines.append("")

    score_keys = (
        ("faithfulness", "Faithfulness (KG-anchored)"),
        ("coverage", "Coverage"),
        ("baseline_delta", "Baseline delta (trained - base)"),
        ("calibration_ece", "Calibration ECE"),
        ("source_match", "Source-Match"),
    )
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, label in score_keys:
        if key in eval_report and eval_report[key] is not None:
            lines.append(f"| {label} | {_round(eval_report[key])} |")
    # Wave 102: surface hallucination_rate as its own row so it doesn't
    # have to be reconstructed from faithfulness by the reader.
    metrics_block = eval_report.get("metrics") or {}
    if "hallucination_rate" in metrics_block and metrics_block["hallucination_rate"] is not None:
        lines.append(
            f"| Hallucination rate (1 - faithfulness) | "
            f"{_round(metrics_block['hallucination_rate'])} |"
        )
    per_inv = eval_report.get("per_invariant") or {}
    for inv_name, payload in sorted(per_inv.items()):
        pr = _extract_pass_rate(payload)
        if pr is not None:
            lines.append(f"| {inv_name}_pass_rate | {_round(pr)} |")
    lines.append("")

    # Wave 102: render the headline + retrieval-method ablation tables.
    if ablation_report:
        lines.extend(_render_headline_table(ablation_report))
        lines.extend(_render_retrieval_method_table(ablation_report))

    # --- Limitations ------------------------------------------ #
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        f"- Scope: this adapter was trained exclusively on the "
        f"`{course_slug}` corpus. Out-of-domain prompts are not "
        f"covered by the eval scores above and may regress against "
        f"the base model."
    )
    lines.append(
        "- Determinism: all reported scores were computed with greedy "
        "decoding (temperature=0.0). Sampled generations may differ."
    )
    lines.append(
        "- Calibration: ECE is computed on a Bloom-stratified holdout "
        "split; rare-class confidence may be under-sampled."
    )
    # Wave 102: name the non-deterministic synthesis step so reviewers
    # know which dimension carries the +/- 0.03 paraphrase drift.
    instruction_pairs_hash = provenance.get("instruction_pairs_hash") or "<unset>"
    lines.append(
        "- Synthesis non-determinism: the instruction-pair corpus is "
        "paraphrased through a Claude Code subagent, which is the only "
        "non-deterministic stage in the pipeline. The pairs are pinned "
        f"by `instruction_pairs_hash={instruction_pairs_hash}`; "
        "re-paraphrasing under a different model snapshot would shift "
        "headline metrics by approximately +/- 0.03 (calibration "
        "baseline from the v1 corpus)."
    )
    lines.append("")

    # Wave 102: reproducibility envelope.
    lines.extend(_render_reproducing_section(model_card))

    # --- Provenance ------------------------------------------- #
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "The Trainforge runner pins SHA-256 hashes of every input "
        "artifact so this adapter is fully replayable post-hoc:"
    )
    lines.append("")
    prov_keys = (
        "chunks_hash",
        "pedagogy_graph_hash",
        "instruction_pairs_hash",
        "preference_pairs_hash",
        "concept_graph_hash",
        "vocabulary_ttl_hash",
        "holdout_graph_hash",
    )
    for key in prov_keys:
        if key in provenance:
            lines.append(f"- `{key}`: `{provenance[key]}`")
    created_at = model_card.get("created_at")
    if created_at:
        lines.append("")
        lines.append(f"Card created at: `{created_at}`")
    lines.append("")
    lines.append(
        "Generated by Wave 101 of Ed4All Trainforge. See "
        "[`Trainforge/eval/hf_model_index.py`](https://github.com/) "
        "for the converter."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Wave 102 helpers                                                        #
# ---------------------------------------------------------------------- #


def _render_headline_table(ablation_report: Dict[str, Any]) -> List[str]:
    """Render the 4-row headline ablation table (4 or 5 columns).

    The qualitative_score column is included only when at least one
    row carries a non-None value; otherwise we render a 4-column
    table.
    """
    rows = ablation_report.get("headline_table") or []
    if not rows:
        return []
    has_qualitative = any(
        r.get("qualitative_score") is not None for r in rows
    )

    lines: List[str] = ["### Headline Ablation", ""]
    headers = [
        "Setup", "Accuracy", "Faithfulness",
        "Hallucination rate", "Source-Match",
    ]
    if has_qualitative:
        headers.append("Qualitative (1-5)")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [
            str(row.get("setup", "?")),
            _fmt_or_dash(row.get("accuracy")),
            _fmt_or_dash(row.get("faithfulness")),
            _fmt_or_dash(row.get("hallucination_rate")),
            _fmt_or_dash(row.get("source_match")),
        ]
        if has_qualitative:
            cells.append(_fmt_or_dash(row.get("qualitative_score")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _render_retrieval_method_table(ablation_report: Dict[str, Any]) -> List[str]:
    """Render the 5-row retrieval-method comparison table."""
    rows = ablation_report.get("retrieval_method_table") or []
    if not rows:
        return []
    lines: List[str] = ["### Retrieval-Method Comparison", ""]
    lines.append(
        "| Method | Accuracy | Faithfulness | Source-Match | "
        "Mean latency (ms) |"
    )
    lines.append("|---|---|---|---|---|")
    for row in rows:
        cells = [
            str(row.get("method", "?")),
            _fmt_or_dash(row.get("accuracy")),
            _fmt_or_dash(row.get("faithfulness")),
            _fmt_or_dash(row.get("source_match")),
            _fmt_or_dash(row.get("mean_latency_ms"), digits=2),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _render_reproducing_section(model_card: Dict[str, Any]) -> List[str]:
    """Render the 'Reproducing These Numbers' subsection."""
    eval_scores = model_card.get("eval_scores") or {}
    scoring_commit = eval_scores.get("scoring_commit") or "<unset>"
    tolerance_band = eval_scores.get("tolerance_band") or {}
    provenance = model_card.get("provenance") or {}

    lines: List[str] = ["## Reproducing These Numbers", ""]
    lines.append(
        "Run `bash reproduce_eval.sh` from this directory. The script "
        "pins the commit SHA, model id, and eval profile, then invokes "
        "`python -m Trainforge.eval.verify_eval` against the stored "
        "`eval_report.json` + `ablation_report.json`. Verification "
        "re-reads the metrics rather than re-running the model, so no "
        "GPU is required."
    )
    lines.append("")
    lines.append(f"- Scoring commit: `{scoring_commit}`")
    if tolerance_band:
        bands_pretty = ", ".join(
            f"{k}={v}" for k, v in sorted(tolerance_band.items())
        )
        lines.append(f"- Tolerance band: `{bands_pretty}`")
    if provenance:
        lines.append(
            "- Pinned hashes: see the `Provenance` section below for "
            "the seven SHA-256 hashes covering chunks, pedagogy graph, "
            "training pairs, concept graph, vocabulary TTL, and the "
            "holdout split."
        )
    lines.append("")
    return lines


def _fmt_or_dash(value: Any, digits: int = 4) -> str:
    """Render a number to ``digits`` dp; return em dash for None."""
    if value is None:
        return "—"
    try:
        return f"{round(float(value), digits)}"
    except (TypeError, ValueError):
        return "—"


__all__ = ["eval_report_to_model_index", "write_hf_readme"]
