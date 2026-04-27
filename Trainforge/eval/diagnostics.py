"""Wave 103 - Auto-detection of qualitative findings from ablation reports.

Three rules fire over the headline ablation table + the per-probe
trace stream:

* ``adapter_tone_only``: the adapter changes faithfulness without
  meaningfully shifting accuracy (within +/- 0.05). Indicates the
  adapter is shifting *style* without delivering new knowledge - a
  common ablation artefact when the SFT corpus is heavy on phrasing
  and light on KG-anchored facts.
* ``prompting_failure``: more than 30% of RAG probes return zero
  citations even when retrieval found chunks. Indicates the prompt
  template / instruction-following pipeline is not driving the model
  to cite, regardless of corpus quality.
* ``dataset_too_easy``: the base model alone clears the 0.7 accuracy
  bar, leaving little headroom for the adapter / RAG to demonstrate
  lift. Procurement claims need a corpus where the bar is meaningful.

Findings render under a "## Diagnostic Findings" section in the HF
README so a human reviewer sees the qualitative signal alongside the
quantitative tables.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from Trainforge.eval.evidence_trace import EvidenceTrace


logger = logging.getLogger(__name__)


# Rule thresholds. Locked here so a regression test can pin them.
ADAPTER_TONE_ACCURACY_BAND = 0.05
ADAPTER_TONE_FAITHFULNESS_DELTA = 0.05
PROMPTING_FAILURE_EMPTY_CITATION_FRACTION = 0.30
DATASET_TOO_EASY_BASE_ACCURACY = 0.70


_HEADLINE_KEYS = ("base", "base+rag", "adapter", "adapter+rag")


def _index_headline_rows(headline_table: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return a setup-keyed view over the headline table, lower-cased."""
    return {
        str(row.get("setup", "")).lower(): dict(row)
        for row in headline_table
    }


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nan(x: float) -> bool:
    return x != x  # NaN check w/o math import


def _detect_adapter_tone_only(
    headline_rows: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base = headline_rows.get("base") or {}
    adapter = headline_rows.get("adapter") or {}
    base_acc = _safe_float(base.get("accuracy"))
    adapter_acc = _safe_float(adapter.get("accuracy"))
    base_faith = _safe_float(base.get("faithfulness"))
    adapter_faith = _safe_float(adapter.get("faithfulness"))
    if any(_nan(v) for v in (base_acc, adapter_acc, base_faith, adapter_faith)):
        return []
    accuracy_close = abs(adapter_acc - base_acc) <= ADAPTER_TONE_ACCURACY_BAND
    faithfulness_lift = (adapter_faith - base_faith) > ADAPTER_TONE_FAITHFULNESS_DELTA
    if accuracy_close and faithfulness_lift:
        return [{
            "finding": "adapter_tone_only",
            "severity": "warning",
            "rationale": (
                f"Adapter accuracy ({adapter_acc:.3f}) is within "
                f"+/-{ADAPTER_TONE_ACCURACY_BAND:.2f} of base "
                f"({base_acc:.3f}) but adapter faithfulness "
                f"({adapter_faith:.3f}) exceeds base "
                f"({base_faith:.3f}) by more than "
                f"{ADAPTER_TONE_FAITHFULNESS_DELTA:.2f}. The adapter "
                f"appears to be shifting tone or hedging behaviour "
                f"without delivering new factual knowledge."
            ),
        }]
    return []


def _detect_prompting_failure(traces: Sequence[EvidenceTrace]) -> List[Dict[str, Any]]:
    rag_traces = [t for t in traces if t.retrieval_method is not None]
    if not rag_traces:
        return []
    empty = sum(1 for t in rag_traces if not t.extracted_citations)
    fraction = empty / len(rag_traces)
    if fraction > PROMPTING_FAILURE_EMPTY_CITATION_FRACTION:
        return [{
            "finding": "prompting_failure",
            "severity": "warning",
            "rationale": (
                f"{empty}/{len(rag_traces)} RAG probes "
                f"({fraction*100:.1f}%) returned zero citations - "
                f"above the {PROMPTING_FAILURE_EMPTY_CITATION_FRACTION*100:.0f}% "
                f"threshold. The prompt template / instruction-following "
                f"path is not driving the model to cite the retrieved "
                f"chunks; consider tightening the citation directive or "
                f"adding a few-shot example."
            ),
        }]
    return []


def _detect_dataset_too_easy(
    headline_rows: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base = headline_rows.get("base") or {}
    base_acc = _safe_float(base.get("accuracy"))
    if _nan(base_acc):
        return []
    if base_acc > DATASET_TOO_EASY_BASE_ACCURACY:
        return [{
            "finding": "dataset_too_easy",
            "severity": "warning",
            "rationale": (
                f"Base-model accuracy ({base_acc:.3f}) already exceeds "
                f"{DATASET_TOO_EASY_BASE_ACCURACY:.2f}. The corpus may "
                f"be too easy for the headline ablation to demonstrate "
                f"meaningful adapter / RAG lift; consider a harder "
                f"holdout slice or a larger eval corpus before publishing "
                f"procurement numbers."
            ),
        }]
    return []


def detect_findings(
    ablation_report: Dict[str, Any],
    traces: Sequence[EvidenceTrace],
) -> List[Dict[str, Any]]:
    """Return the list of diagnostic findings that apply.

    Args:
        ablation_report: Output of
            :class:`Trainforge.eval.ablation_runner.AblationRunner.run`
            (the parsed ablation_report.json dict).
        traces: Iterable of :class:`EvidenceTrace` rows
            (typically read back via :func:`evidence_trace.load_traces`).

    Returns:
        List of ``{"finding": str, "severity": "warning"|"info",
        "rationale": str}`` dicts. Empty list when no rule fires.
    """
    headline_table = ablation_report.get("headline_table") or []
    headline_rows = _index_headline_rows(headline_table)

    findings: List[Dict[str, Any]] = []
    findings.extend(_detect_adapter_tone_only(headline_rows))
    findings.extend(_detect_prompting_failure(list(traces)))
    findings.extend(_detect_dataset_too_easy(headline_rows))
    return findings


__all__ = [
    "ADAPTER_TONE_ACCURACY_BAND",
    "ADAPTER_TONE_FAITHFULNESS_DELTA",
    "DATASET_TOO_EASY_BASE_ACCURACY",
    "PROMPTING_FAILURE_EMPTY_CITATION_FRACTION",
    "detect_findings",
]
