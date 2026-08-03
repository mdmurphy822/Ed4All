"""Course-aware downstream probes used to select SFT/DPO checkpoints.

The selector intentionally evaluates learner-facing generations rather than
training loss.  Gold key-point coverage reuses the canonical retrieval-answer
scorer; symbolic assessment correctness uses SymPy equivalence over authored
answer keys.  Progress is checkpointed after every generation so an operator
stop or process restart resumes without repeating completed model calls.
"""
from __future__ import annotations

import gc
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from lib.generation.stop_control import check_stop
from lib.retrieval.answer_scoring import score_key_point_coverage
from lib.utils import sha256_file

from Trainforge.eval.retrieval.adapter_callable import AdapterCallable
from Trainforge.training.base_models import BaseModelSpec


_METRICS = frozenset({"gold_keypoint_coverage", "sympy_correctness", "composite"})
_ANSWER_RE = re.compile(
    r"(?:answer\s*(?:is|:|=)\s*)?([-+]?(?:\d+(?:\.\d+)?|\d+/\d+))",
    re.IGNORECASE,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_gold_questions(course_dir: Path, limit: int) -> List[Dict[str, Any]]:
    path = course_dir / "retrieval_eval" / "gold_set.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint probe requires the verified retrieval gold set: {path}"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in doc.get("questions", [])
        if isinstance(row, dict)
        and str(row.get("question_text", "")).strip()
        and len(row.get("expected_key_points") or []) >= 1
    ]
    if not rows:
        raise RuntimeError(
            f"checkpoint probe found no questions with expected_key_points in {path}"
        )
    return rows[:limit]


def _walk_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _load_symbolic_items(course_dir: Path, limit: int) -> List[Dict[str, str]]:
    """Harvest authored numeric-answer assessment items from course artifacts."""
    candidates = [
        course_dir / "assessment_questions.json",
        course_dir / "assessments" / "questions.json",
        course_dir / "quality" / "assessment_questions.json",
    ]
    chunks = course_dir / "imscc_chunks" / "chunks.jsonl"
    docs: List[Any] = []
    for path in candidates:
        if path.is_file():
            docs.append(json.loads(path.read_text(encoding="utf-8")))
    if chunks.is_file():
        with chunks.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    docs.append(json.loads(line))

    out: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        for row in _walk_dicts(doc):
            question = next(
                (str(row.get(key, "")).strip() for key in
                 ("question_text", "question", "stem", "prompt")
                 if str(row.get(key, "")).strip()),
                "",
            )
            answer = next(
                (str(row.get(key, "")).strip() for key in
                 ("correct_answer", "answer_key", "answer", "expected_answer")
                 if isinstance(row.get(key), (str, int, float))
                 and str(row.get(key, "")).strip()),
                "",
            )
            if not question or not answer or not _ANSWER_RE.search(answer):
                continue
            key = (question, answer)
            if key in seen:
                continue
            seen.add(key)
            out.append({"question": question, "answer": answer})
            if len(out) >= limit:
                return out
    return out


def _numeric_expression(text: str) -> Any:
    try:
        import sympy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "checkpoint probe metric sympy_correctness requires SymPy; "
            "install the Ed4All embedding/training dependencies."
        ) from exc
    match = _ANSWER_RE.search(str(text))
    if not match:
        return None
    try:
        return sympy.sympify(match.group(1), evaluate=True)
    except (TypeError, ValueError, SyntaxError):
        return None


class CourseCheckpointProbe:
    """Callable that evaluates one adapter checkpoint against held-out probes."""

    def __init__(
        self,
        *,
        course_dir: Path,
        stage: str,
        state_root: Path,
        spec: BaseModelSpec,
        training_config: Mapping[str, Any],
        metric: str,
        capture: Optional[Any] = None,
        max_gold_questions: int = 110,
        max_sympy_questions: int = 40,
        model_callable_factory: Optional[Callable[[Path], Callable[[str], str]]] = None,
    ) -> None:
        if metric not in _METRICS:
            raise ValueError(
                f"unsupported downstream checkpoint metric {metric!r}; "
                f"expected one of {sorted(_METRICS)}"
            )
        if model_callable_factory is None and capture is None:
            raise RuntimeError(
                "production checkpoint probes require DecisionCapture; "
                "refusing uncaptured model-evaluation calls"
            )
        self.course_dir = Path(course_dir)
        self.stage = str(stage)
        self.state_root = Path(state_root)
        self.spec = spec
        self.training_config = dict(training_config)
        self.metric = metric
        self.capture = capture
        self.gold = _load_gold_questions(self.course_dir, max_gold_questions)
        self.symbolic = (
            _load_symbolic_items(self.course_dir, max_sympy_questions)
            if metric in {"sympy_correctness", "composite"}
            else []
        )
        if metric in {"sympy_correctness", "composite"} and not self.symbolic:
            raise RuntimeError(
                "checkpoint probe requires at least one authored, numeric "
                f"assessment answer for metric={metric!r}; none found under "
                f"{self.course_dir}"
            )
        self._factory = model_callable_factory or self._default_factory

    def _default_factory(self, checkpoint_dir: Path) -> Callable[[str], str]:
        return AdapterCallable(
            checkpoint_dir,
            self.spec.huggingface_repo,
            revision=self.spec.default_revision,
            base_model_short_name=self.spec.name,
            seed=int(self.training_config.get("seed", 42)),
            max_new_tokens=256,
            temperature=0.0,
            use_4bit=bool(self.training_config.get("use_4bit", False)),
        )

    def _state_path(self, checkpoint_dir: Path) -> Path:
        identity = hashlib.sha256(
            str(checkpoint_dir.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        return self.state_root / self.stage / f"{checkpoint_dir.name}-{identity}.json"

    def __call__(self, checkpoint_dir: Path) -> Dict[str, float]:
        state_path = self._state_path(checkpoint_dir)
        adapter_path = checkpoint_dir / "adapter_model.safetensors"
        if not adapter_path.is_file():
            raise FileNotFoundError(
                f"checkpoint probe adapter is missing: {adapter_path}"
            )
        fingerprint_payload = {
            "adapter_sha256": sha256_file(adapter_path),
            "base_model": self.spec.name,
            "revision": self.spec.default_revision,
            "metric": self.metric,
            "stage": self.stage,
            "gold": self.gold,
            "symbolic": self.symbolic,
            "max_new_tokens": 256,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        state: Dict[str, Any] = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "answers": {},
        }
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != 1:
                raise RuntimeError(
                    f"checkpoint probe resume schema mismatch at {state_path}"
                )
            if state.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "checkpoint probe resume fingerprint mismatch at "
                    f"{state_path}; checkpoint/model/prompts/scorer inputs "
                    "changed, so cached generations cannot be reused"
                )
        answers = state.setdefault("answers", {})
        model = self._factory(checkpoint_dir)
        try:
            gold_total = gold_covered = 0
            for index, row in enumerate(self.gold):
                qid = str(row.get("question_id") or f"gold-{index}")
                key = f"gold:{qid}"
                check_stop(
                    f"trainforge_train.checkpoint_probe.{self.stage}.gold",
                    index,
                )
                if key not in answers:
                    answers[key] = str(model(str(row["question_text"])))
                    _atomic_json(state_path, state)
                    self._capture(
                        checkpoint_dir, key, len(row["expected_key_points"])
                    )
                coverage = score_key_point_coverage(
                    answers[key], row["expected_key_points"]
                )
                if coverage is None:
                    raise RuntimeError(f"gold probe {qid!r} was not scorable")
                gold_total += coverage.total
                gold_covered += coverage.covered
            metrics: Dict[str, float] = {
                "gold_keypoint_coverage": gold_covered / gold_total
            }

            if self.symbolic:
                correct = 0
                for index, row in enumerate(self.symbolic):
                    key = f"sympy:{index}"
                    check_stop(
                        f"trainforge_train.checkpoint_probe.{self.stage}.sympy",
                        index,
                    )
                    if key not in answers:
                        answers[key] = str(model(row["question"]))
                        _atomic_json(state_path, state)
                        self._capture(checkpoint_dir, key, 1)
                    expected = _numeric_expression(row["answer"])
                    actual = _numeric_expression(answers[key])
                    if expected is None:
                        raise RuntimeError(
                            f"authored symbolic answer is not parseable: {row['answer']!r}"
                        )
                    if actual is not None and bool(
                        math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
                    ):
                        correct += 1
                metrics["sympy_correctness"] = correct / len(self.symbolic)
            return metrics
        finally:
            del model
            gc.collect()
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _capture(self, checkpoint_dir: Path, probe_id: str, points: int) -> None:
        if self.capture is None:
            return
        self.capture.log_decision(
            decision_type="eval_run_decision",
            decision=f"Evaluate {self.stage} {checkpoint_dir.name} on {probe_id}",
            rationale=(
                f"Downstream checkpoint probe {probe_id} evaluated pinned model "
                f"{self.spec.name} revision {self.spec.default_revision} with "
                f"max_new_tokens=256 and {points} authored scoring target(s); "
                f"stage={self.stage}, metric={self.metric}."
            ),
            context=str(self.course_dir),
            confidence=1.0,
            is_default=False,
        )


def preflight_checkpoint_probe(
    course_dir: Path,
    *,
    metric: str,
    max_gold_questions: int = 110,
    max_sympy_questions: int = 40,
) -> Dict[str, Any]:
    """Validate probe inputs without loading a model or touching CUDA."""
    if metric not in _METRICS:
        raise ValueError(
            f"unsupported downstream checkpoint metric {metric!r}; "
            f"expected one of {sorted(_METRICS)}"
        )
    gold = _load_gold_questions(Path(course_dir), max_gold_questions)
    symbolic = (
        _load_symbolic_items(Path(course_dir), max_sympy_questions)
        if metric in {"sympy_correctness", "composite"}
        else []
    )
    if metric in {"sympy_correctness", "composite"} and not symbolic:
        raise RuntimeError(
            f"metric={metric!r} requires authored numeric assessment items"
        )
    return {
        "course_dir": str(Path(course_dir)),
        "metric": metric,
        "gold_questions": len(gold),
        "gold_key_points": sum(
            len(row.get("expected_key_points") or []) for row in gold
        ),
        "sympy_questions": len(symbolic),
        "ready": True,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only preflight for downstream checkpoint selection"
    )
    parser.add_argument("--course-dir", type=Path, required=True)
    parser.add_argument("--metric", required=True, choices=sorted(_METRICS))
    args = parser.parse_args(argv)
    print(json.dumps(
        preflight_checkpoint_probe(args.course_dir, metric=args.metric),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CourseCheckpointProbe", "preflight_checkpoint_probe", "main"]
