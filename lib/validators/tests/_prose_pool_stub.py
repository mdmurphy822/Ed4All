"""Importable, picklable stub-NLI factory for the process-pool prose-entailment
tests.

Lives as its OWN module (not inside a test file) so a spawn
``ProcessPoolExecutor`` worker can import it by its qualified name
(``lib.validators.tests._prose_pool_stub``) and build a deterministic,
GPU-free NLI in the child process — the same shape
``ED4ALL_NLI_VALIDATORS_FACTORY`` points production at
``NliClassifier.get_or_load``. Nothing here loads torch.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from lib.classifiers.nli_classifier import NliScore


class MarkerStubNli:
    """Deterministic NLI stub keyed on markers embedded in the hypothesis.

    ``[ENTAIL]`` → high entailment, ``[CONTRA]`` → high contradiction, else
    unsupported. Batch-position independent so batched and serial scores are
    identical (the property real attention-masked batching also guarantees to
    within low-order bits). Exposes ``_revision`` / ``device`` so
    ``score_groundedness`` records provenance identically to the real model.
    """

    _revision = "fake-nli-rev-mb"
    device = "cpu"

    def score_batch(
        self, *, pairs: List[Tuple[str, str]], batch_size: Optional[int] = None
    ) -> List[NliScore]:
        out: List[NliScore] = []
        for _premise, hypothesis in pairs:
            if "[ENTAIL]" in hypothesis:
                out.append(NliScore(entailment=0.95, neutral=0.03, contradiction=0.02))
            elif "[CONTRA]" in hypothesis:
                out.append(NliScore(entailment=0.05, neutral=0.10, contradiction=0.85))
            else:
                out.append(NliScore(entailment=0.10, neutral=0.85, contradiction=0.05))
        return out


def marker_stub_factory() -> MarkerStubNli:
    """Build the deterministic stub — the per-worker NLI-load the pool calls once.

    When ``_PROSE_POOL_STUB_INITFILE`` is set, appends the calling PID so a
    test can observe that a worker (a distinct process) actually ran.
    """
    initfile = os.environ.get("_PROSE_POOL_STUB_INITFILE")
    if initfile:
        try:
            with open(initfile, "a", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()}\n")
        except OSError:
            pass
    return MarkerStubNli()
