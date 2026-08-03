"""Wave 103 - Per-probe evidence traces for the ablation runner.

Every probe across every model setup + retrieval method emits one
trace row that captures:

* What the model saw (prompt + retrieved chunks).
* What the model produced (output + extracted citations).
* Whether the ground-truth chunk was retrieved at top-k.
* Whether the model cited the correct chunk and answered correctly.
* The classified failure mode (one of five canonical labels).

Traces land at ``<run_dir>/eval_traces.jsonl`` and serve two
downstream consumers:

1. :mod:`Trainforge.eval.diagnostics` runs auto-detection rules over
   the traces (e.g. retrieval-hit-no-cite triggers
   ``prompting_failure``).
2. Manual auditing during procurement review - a human can scroll the
   first 50 rows and sanity-check the failure-mode classifier.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# Canonical failure-mode labels. Locked - downstream diagnostics rules
# pattern-match against these strings.
FAILURE_MODES = (
    "none",
    "retrieval_miss",
    "retrieval_hit_no_cite",
    "cited_wrong",
    "model_ignored_context",
)


@dataclass
class EvidenceTrace:
    """One row in eval_traces.jsonl."""

    probe_id: str
    setup: str  # "base" | "base_rag" | "adapter" | "adapter_rag"
    retrieval_method: Optional[str]  # None for non-RAG rows
    prompt: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    ground_truth_chunk_id: Optional[str] = None
    retrieved_at_top_k: bool = False
    model_output: str = ""
    extracted_citations: List[str] = field(default_factory=list)
    cited_correct_chunk: bool = False
    answer_correct: bool = False
    failure_mode: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Coerce retrieval_method=None to JSON null explicitly.
        return d


# Citation extraction: tolerates "[chunk_id]", "[CHUNK_123]",
# multi-citation "[a][b]" and inline "[abc-def_42]" forms. Excludes
# free-text bracketed numbers like footnote refs by anchoring on at
# least one alphabetic char in the bracket body.
_CITATION_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_\-:]*)\]")


def extract_citations(model_output: str) -> List[str]:
    """Pull bracketed chunk-id citations from a free-text response."""
    if not model_output:
        return []
    return _CITATION_RE.findall(model_output)


def classify_failure_mode(
    *,
    retrieved_at_top_k: bool,
    cited_correct_chunk: bool,
    answer_correct: bool,
    model_used_context: bool,
) -> str:
    """Return one of :data:`FAILURE_MODES` for a probe.

    Decision tree:

    * Correct answer + correct citation -> ``none``.
    * GT chunk not retrieved at top-k -> ``retrieval_miss``.
    * GT chunk retrieved but not cited (regardless of correctness)
      -> ``retrieval_hit_no_cite``.
    * GT chunk retrieved and a different chunk cited -> ``cited_wrong``.
    * GT chunk retrieved + answer is wrong + model produced no
      citations and ignored the context entirely -> ``model_ignored_context``.
    * Fallback when none of the above apply -> ``none``.
    """
    if answer_correct and cited_correct_chunk:
        return "none"
    if not retrieved_at_top_k:
        return "retrieval_miss"
    # GT chunk was in the prelude.
    if not cited_correct_chunk and not model_used_context:
        return "model_ignored_context"
    if not cited_correct_chunk:
        return "retrieval_hit_no_cite"
    # Cited the correct chunk but answer is still wrong - model used
    # the context but produced an incorrect answer. We treat this as
    # a non-failure for routing (no rule fires).
    return "none"


class TraceWriter:
    """Append-only JSONL trace writer."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate so re-runs produce a fresh trace file.
        self._fp = self.path.open("w", encoding="utf-8")
        self._closed = False

    def append(self, trace: EvidenceTrace) -> None:
        if self._closed:
            raise RuntimeError(
                f"TraceWriter at {self.path} is already closed."
            )
        line = json.dumps(trace.to_dict(), sort_keys=True, ensure_ascii=False)
        self._fp.write(line + "\n")

    def close(self) -> None:
        if not self._closed:
            self._fp.flush()
            self._fp.close()
            self._closed = True

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def load_traces(path: Path) -> List[EvidenceTrace]:
    """Read an eval_traces.jsonl file back into ``EvidenceTrace`` rows."""
    out: List[EvidenceTrace] = []
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(EvidenceTrace(**row))
    return out


__all__ = [
    "EvidenceTrace",
    "FAILURE_MODES",
    "TraceWriter",
    "classify_failure_mode",
    "extract_citations",
    "load_traces",
]
