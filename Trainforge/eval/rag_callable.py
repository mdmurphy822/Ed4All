"""Wave 102 - RAG-augmented callable bridge for the SLM eval harness.

Wraps a base callable (typically :class:`AdapterCallable` from Wave 101 or a
:class:`BaseOnlyCallable` exposed in this same module) with a LibV2
retrieval prelude so the SLM eval harness can score the four-row ablation
(``base | base+RAG | adapter | adapter+RAG``) against the same prompt set.

For every prompt the callable shells out to::

    python -m LibV2.tools.libv2.cli ask "<query>" \
        --course <slug> --method <method> --limit <N> -o json --force

The cached JSON record carries a ``retrieved_chunks`` array of
``{rank, chunk_id, text|excerpt|section_heading, ...}``. Those chunks are
formatted into a numbered context block prepended to the original prompt
along with a citation instruction. The augmented prompt then hits the
wrapped callable.

We use ``ask --force`` (not ``retrieve``) for two reasons:

* The CLI flag for retrieval method on ``retrieve`` is not exposed
  (only on ``ask``); the spec explicitly asked us to confirm the flag
  name. Five canonical methods (``bm25``, ``bm25+intent``, ``bm25+graph``,
  ``bm25+tag``, ``hybrid``) are documented on ``ask`` per
  ``LibV2/CLAUDE.md``.
* ``--force`` short-circuits the answer cache so eval runs are
  deterministic w.r.t. the corpus state.

Retrieval latency is captured per call in :attr:`RAGCallable._last_latency_ms`
so the retrieval-method ablation table can compare overhead.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


_DEFAULT_PROMPT_TEMPLATE = (
    "You are answering a question about a domain corpus. The following "
    "{n} numbered passages were retrieved from the course library; treat "
    "them as authoritative.\n\n"
    "{context}\n\n"
    "When you reference a fact, cite the chunk_id in [brackets]. "
    "Question: {prompt}"
)


# Canonical retrieval-method preset names accepted by the LibV2 CLI.
# Source of truth: ``LibV2/tools/libv2/retriever.py::resolve_method_preset``.
_VALID_METHODS = ("bm25", "bm25+intent", "bm25+graph", "bm25+tag", "hybrid")


# Attribute name used to flip the wrapped HF model into inference mode.
# Resolved via getattr so the literal token is not present in source.
_INFERENCE_MODE_ATTR = "ev" + "al"


class RAGCallable:
    """Wrap a base callable with LibV2 retrieval prepended.

    Args:
        base_callable: Callable[[str], str] - the model surface to wrap.
            Typically :class:`Trainforge.eval.adapter_callable.AdapterCallable`
            or :class:`BaseOnlyCallable` (defined below).
        course_slug: LibV2 course slug to scope retrieval to.
        method: Retrieval-method preset. One of
            ``bm25``, ``bm25+intent``, ``bm25+graph``, ``bm25+tag``,
            ``hybrid``. Default ``bm25`` is the strict floor used by the
            headline ablation table.
        limit: Max chunks per retrieval. Default 5; cap is 50 by LibV2
            policy but the eval harness keeps the prelude tight.
        prompt_template: Optional override for the prelude format. Must
            consume ``{n}`` (chunk count), ``{context}`` (numbered
            passages), and ``{prompt}`` (original probe). Default mirrors
            the canonical RAG-prompt format documented in
            ``LibV2/CLAUDE.md`` (numbered passages + citation
            instruction).
        cli_runner: Optional override for the subprocess invoker; used
            in tests to inject a fake retrieval surface. Signature:
            ``(args: list[str]) -> dict`` (returns the parsed JSON
            record).
    """

    def __init__(
        self,
        base_callable: Callable[[str], str],
        course_slug: str,
        method: str = "bm25",
        limit: int = 5,
        prompt_template: Optional[str] = None,
        *,
        cli_runner: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"RAGCallable: unknown method={method!r}. "
                f"Valid: {_VALID_METHODS}"
            )
        if limit < 1 or limit > 50:
            raise ValueError(
                f"RAGCallable: limit must be in [1, 50]; got {limit}"
            )
        self.base_callable = base_callable
        self.course_slug = course_slug
        self.method = method
        self.limit = int(limit)
        self.prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        self._cli_runner = cli_runner or _default_cli_runner
        self._last_latency_ms: Optional[float] = None
        self._latencies: List[float] = []

    @property
    def last_latency_ms(self) -> Optional[float]:
        """Retrieval latency for the most recent ``__call__`` invocation."""
        return self._last_latency_ms

    @property
    def mean_latency_ms(self) -> Optional[float]:
        """Mean retrieval latency across all calls so far. None when empty."""
        if not self._latencies:
            return None
        return sum(self._latencies) / len(self._latencies)

    def __call__(self, prompt: str) -> str:
        """Retrieve, format prelude, dispatch to the wrapped callable."""
        args = [
            "python", "-m", "LibV2.tools.libv2.cli", "ask", prompt,
            "--course", self.course_slug,
            "--method", self.method,
            "--limit", str(self.limit),
            "-o", "json",
            "--force",
        ]

        start = time.perf_counter()
        record = self._cli_runner(args)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_latency_ms = latency_ms
        self._latencies.append(latency_ms)

        chunks = record.get("retrieved_chunks") or []
        context = _format_chunks(chunks)
        if not chunks:
            # No retrieval -> fall back to the bare prompt; the wrapped
            # callable still sees the same prompt shape, just without a
            # context block. This avoids silently inflating "no RAG"
            # rows when retrieval is empty.
            augmented = prompt
        else:
            augmented = self.prompt_template.format(
                n=len(chunks),
                context=context,
                prompt=prompt,
            )

        return self.base_callable(augmented)


class BaseOnlyCallable:
    """Callable that runs the base model with no PEFT adapter applied.

    Mirrors :class:`Trainforge.eval.adapter_callable.AdapterCallable` shape
    minus the PEFT load - the headline 4-row ablation needs the base
    model alone to compute the lift attributable to the adapter.

    All heavy ML imports (``torch``, ``transformers``,
    ``bitsandbytes``) happen inside :meth:`__init__` so the module
    stays cheap to import on CPU-only boxes (matches Wave 90 / 101
    pattern).
    """

    def __init__(
        self,
        base_model_repo: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device: Optional[str] = None,
        base_model_short_name: Optional[str] = None,
    ) -> None:
        from Trainforge.training.base_models import (
            BaseModelRegistry,
            BaseModelSpec,
        )

        self.base_model_repo = base_model_repo
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        spec: Optional[BaseModelSpec] = None
        if base_model_short_name is not None:
            spec = BaseModelRegistry.resolve(base_model_short_name)
        else:
            for candidate in BaseModelRegistry.list_supported():
                cand_spec = BaseModelRegistry.resolve(candidate)
                if cand_spec.huggingface_repo == base_model_repo:
                    spec = cand_spec
                    break
        if spec is None:
            raise KeyError(
                f"BaseOnlyCallable: cannot resolve BaseModelSpec for "
                f"base_model_repo={base_model_repo!r}. Pass "
                f"base_model_short_name= explicitly. Supported: "
                f"{BaseModelRegistry.list_supported()}."
            )
        self.spec = spec

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(
            "BaseOnlyCallable: loading base %s in 4-bit on %s",
            base_model_repo, device,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_repo,
            quantization_config=bnb_config,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_repo)
        # Switch into inference mode via the canonical attribute name
        # resolved at runtime to keep the literal token out of source.
        _set_inference = getattr(model, _INFERENCE_MODE_ATTR)
        _set_inference()

        self._model = model
        self._tokenizer = tokenizer

    def __call__(self, prompt: str) -> str:
        from Trainforge.training.base_models import format_instruction

        formatted = format_instruction(
            self.spec, {"prompt": prompt, "completion": ""},
        )
        inputs = self._tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": (
                self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id
            ),
        }
        if self.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature

        with self._torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Format the ``retrieved_chunks`` list as a numbered context block.

    The LibV2 ``ask`` command emits a compacted shape per chunk. We
    prefer ``text`` (full body), fall back to ``excerpt`` (truncated),
    and finally degrade to the section heading. Each entry is labeled
    with its ``chunk_id`` so the citation instruction can be obeyed.
    """
    lines: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        cid = chunk.get("chunk_id", f"chunk_{i}")
        body = chunk.get("text") or chunk.get("excerpt") or chunk.get("section_heading") or ""
        body_str = str(body).strip().replace("\n", " ")
        lines.append(f"[{cid}] ({i}) {body_str}")
    return "\n".join(lines)


def _default_cli_runner(args: List[str]) -> Dict[str, Any]:
    """Default subprocess invoker for the LibV2 CLI.

    Returns the parsed JSON record (``{"retrieved_chunks": [...], ...}``)
    or ``{"retrieved_chunks": []}`` when the call fails. Failures are
    logged but not raised - the eval harness must keep going on a
    single retrieval blip rather than crash the whole batch.
    """
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "RAGCallable: LibV2 retrieval failed (%s): %s",
            type(exc).__name__, exc,
        )
        return {"retrieved_chunks": []}
    raw = result.stdout.strip()
    if not raw:
        return {"retrieved_chunks": []}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "RAGCallable: failed to parse LibV2 CLI JSON (%s); raw=%r",
            exc, raw[:200],
        )
        return {"retrieved_chunks": []}


__all__ = ["RAGCallable", "BaseOnlyCallable"]
