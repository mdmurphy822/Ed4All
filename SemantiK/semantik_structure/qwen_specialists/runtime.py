"""Qwen runtime backends for Stage 6 specialist generation.

Two implementations behind one Protocol so the driver doesn't care
whether it's running against a real GGUF or a deterministic mock:

    QwenRuntime         — Protocol (load / free / generate).
    MockRuntime         — deterministic stand-in for tests + smoke runs.
                          Returns synthetic strings; never touches GPU.
    LlamaCppRuntime     — wraps llama-cpp-python. Lazy import: tests +
                          mock-mode runs do not require the package to
                          be installed.

The factory ``make_runtime("mock"|"real")`` is the single entry point
the runner uses; it stays in this module so callers don't import the
backend names directly.

Why llama-cpp-python and not transformers + bnb-NF4? See architecture.md
§4.1 — Q4_K_M / Q5_K_M GGUFs leave more headroom on the 8 GB 3060 with
Chromium + axe-core concurrently, and per-token latency is lower for
batched generation. Council BERTs stay on transformers + PEFT.

Real-mode contract
------------------

``LlamaCppRuntime`` is a thin shim. ``load(gguf_path)`` will FAIL
LOUDLY in two cases — both deliberate, never silenced:

    1. ``llama-cpp-python`` not installed → RuntimeError with the
       install hint.
    2. GGUF path missing on disk → FileNotFoundError naming the path.

The "trained adapters don't exist yet" reality (Stage 6 v1) means in
practice every adapter in ``config.yaml`` carries ``adapter_path:
null``. ``AdapterSwap`` treats null as a no-op load, so the runtime
never gets a path. Calling ``LlamaCppRuntime.generate()`` without a
prior successful ``load()`` is a programming error and raises.

Phase 1.5 update — chat-template wrap + adapter version tag
-----------------------------------------------------------

Trained adapters expect their input wrapped in Qwen3's chat template;
``LlamaCppRuntime.generate`` calls
:func:`semantik_structure.qwen_specialists.chat_format.wrap_for_qwen` before
dispatching to ``self._handle``. The wrap matches the training-time
shape (see ``training.py``) so the model sees identical token boundaries
in both phases.

``load`` also reads a sibling ``<gguf>.metadata.json`` if present and
populates ``self._adapter_version``. The runner pulls that version onto
every emitted Candidate so a trained-adapter audit trail survives into
Stage 8 reranking.
"""

from __future__ import annotations

import json
import logging
import os
import re as _re_mod
from pathlib import Path
from typing import Any, Literal, Protocol

from .. import paths as _semantik_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Specialist-provider selector
# ---------------------------------------------------------------------------
#
# ``SEMANTIK_SPECIALIST_PROVIDER`` chooses WHERE the HEAVY Stage-6
# specialist generation runs. Only the qwen specialists are affected — the
# council/structure BERTs, region detection, and theta stay local.
#
#   unset / "local"                       -> local GGUF (LlamaCppRuntime),
#                                            current behaviour, byte-stable.
#   "nvidia" / "local-openai" / "endpoint"
#   / any other non-"local" value         -> hosted endpoint
#                                            (OpenAICompatibleRuntime).
#
# Parse-with-fallback: a blank / "local" value is the local arm; ANY other
# non-empty value selects the endpoint arm (mirrors the ED4ALL_ANSWER_PROVIDER
# registry posture — an opt-in provider string flips the backend).

# Provider values that explicitly mean "stay local on the GGUF".
_LOCAL_PROVIDER_VALUES = frozenset({"", "local", "gguf", "llama_cpp", "llamacpp"})

# Sane literal default model for the OpenAI-compatible seat — mirrors
# endpoint_runtime._DEFAULT_MODEL and resolve_structure_review_model. Kept as a
# literal (not imported) to avoid pulling endpoint_runtime in at module load.
_DEFAULT_SEAT_MODEL = "meta/llama-3.3-70b-instruct"


def _normalize_provider(raw: str | None) -> str:
    """Normalize a raw provider string → ``"local"`` or the lowercased value.

    ``""`` / ``"local"`` (and the GGUF aliases in ``_LOCAL_PROVIDER_VALUES``)
    collapse to ``"local"``; anything else is the lower-cased, stripped
    provider name (e.g. ``"nvidia"``, ``"local-openai"``, ``"endpoint"``)."""
    v = (raw or "").strip().lower()
    return "local" if v in _LOCAL_PROVIDER_VALUES else v


def resolve_specialist_provider() -> str:
    """Return the normalized single-seat specialist provider string.

    ``"local"`` when ``SEMANTIK_SPECIALIST_PROVIDER`` is unset / blank /
    ``"local"`` (and aliases); otherwise the lower-cased provider value
    (e.g. ``"nvidia"``, ``"local-openai"``, ``"endpoint"``).

    Post per-phase-seat redesign this is the **Phase-2 (refine/generate) seat
    default** — the single seat selects WHICH provider/model the Phase-2 tier
    (and the Stage-5d reviewer / Stage-5e resegment seats) use. Phase 1
    (draft) defaults to ``"local"`` regardless of this value (the dereliction
    default — see :func:`resolve_phase_provider`)."""
    return _normalize_provider(os.environ.get("SEMANTIK_SPECIALIST_PROVIDER"))


def specialist_provider_is_endpoint() -> bool:
    """True when the single-seat provider is an endpoint (NOT the local GGUF)."""
    return resolve_specialist_provider() != "local"


# ---------------------------------------------------------------------------
# Per-phase seat resolution (model-agnostic two-phase Stage-6)
# ---------------------------------------------------------------------------
#
# Stage 6 has two phases — Phase 1 (draft) and Phase 2 (refine/generate). Each
# phase resolves its OWN {provider, model} seat from the registry/env, so any
# composition works: local-draft→local-large-refine (fully on-device),
# local-draft→hosted-refine, hosted-draft→hosted-refine, single-phase-any-
# provider, etc. The hosted large seat is just ONE possible Phase-2 model,
# never a hardcoded tier. base_url / api_key are resolved by the OpenAI-compatible
# endpoint runtime from the shared SEMANTIK_SPECIALIST_BASE_URL/API_KEY (>
# NVIDIA_*) env — per-phase provider + model are the per-phase knobs.
#
# Resolution (parse-with-fallback everywhere):
#
#   Phase-1 provider : SEMANTIK_SPECIALIST_PHASE1_PROVIDER > "local"
#                      (does NOT inherit the single seat — the dereliction
#                       default keeps Phase 1 on the license-clean on-device
#                       specialists unless EXPLICITLY pointed elsewhere).
#   Phase-2 provider : SEMANTIK_SPECIALIST_PHASE2_PROVIDER
#                      > SEMANTIK_SPECIALIST_PROVIDER (single seat) > "local".
#   Phase-N model    : SEMANTIK_SPECIALIST_PHASE{N}_MODEL
#                      > SEMANTIK_SPECIALIST_MODEL > NVIDIA_LARGE_MODEL
#                      > the literal default (only consulted on a non-local
#                        seat — the local GGUF arm picks its model from the
#                        adapter config, not this string).

_VALID_PHASES = ("phase1", "phase2")


def resolve_phase_provider(phase: str) -> str:
    """Return the normalized provider for ``"phase1"`` / ``"phase2"``.

    Phase 1 defaults to ``"local"`` (the dereliction default — it does NOT
    inherit ``SEMANTIK_SPECIALIST_PROVIDER``); Phase 2 falls back to the
    single seat. See the module comment above for the full chain."""
    if phase == "phase1":
        # _normalize_provider("" / unset) -> "local"; explicit value honored.
        return _normalize_provider(os.environ.get("SEMANTIK_SPECIALIST_PHASE1_PROVIDER"))
    if phase == "phase2":
        raw = os.environ.get("SEMANTIK_SPECIALIST_PHASE2_PROVIDER")
        if raw and raw.strip():
            return _normalize_provider(raw)
        return resolve_specialist_provider()
    raise ValueError(f"unknown phase {phase!r} (expected one of {_VALID_PHASES})")


def resolve_phase_model(phase: str) -> str:
    """Return the model id for ``"phase1"`` / ``"phase2"`` on the endpoint seat.

    Only consulted when the phase's provider is non-local (the local GGUF arm
    resolves its model from the adapter config). Resolution: the per-phase env
    > the shared ``SEMANTIK_SPECIALIST_MODEL`` > ``NVIDIA_LARGE_MODEL`` > the
    literal default."""
    if phase == "phase1":
        per_phase = os.environ.get("SEMANTIK_SPECIALIST_PHASE1_MODEL")
    elif phase == "phase2":
        per_phase = os.environ.get("SEMANTIK_SPECIALIST_PHASE2_MODEL")
    else:
        raise ValueError(f"unknown phase {phase!r} (expected one of {_VALID_PHASES})")
    return (
        per_phase
        or os.environ.get("SEMANTIK_SPECIALIST_MODEL")
        or os.environ.get("NVIDIA_LARGE_MODEL")
        or _DEFAULT_SEAT_MODEL
    )


def phase_provider_is_endpoint(phase: str) -> bool:
    """True when the phase's resolved provider is an endpoint (not local GGUF)."""
    return resolve_phase_provider(phase) != "local"


def any_phase_provider_is_endpoint() -> bool:
    """True when EITHER phase's seat resolves to an endpoint provider.

    Used by the cascade entry to decide ``runtime_mode`` (an endpoint on
    either phase forces ``"real"``). Subsumes
    :func:`specialist_provider_is_endpoint` because the Phase-2 seat falls
    back to the single seat."""
    return any(phase_provider_is_endpoint(p) for p in _VALID_PHASES)


# Truthy strings for SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE (parse-with-
# fallback; any other value is falsey/off).
_ENDPOINT_DISPLACE_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_endpoint_displaces_local() -> bool:
    """True when a selected endpoint provider should DISPLACE the local
    specialists as the Stage-6 authoring tier (pure-endpoint generation).

    **Default OFF — the dereliction fix.** Selecting an endpoint provider via
    ``SEMANTIK_SPECIALIST_PROVIDER`` alone no longer silently skips the local
    GGUF specialists; the license-clean on-device specialists remain the
    authoring tier and the hosted large seat is reachable only on an EXPLICIT
    opt-in:

      * ``SEMANTIK_SPECIALIST_REFINE`` — the hybrid flow (local drafts →
        hosted-large polish), OR
      * ``SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE`` (this flag) — full
        pure-endpoint displacement (the pre-fix ``provider=<endpoint>``
        behaviour, now gated behind an intentional flag).

    Parse-with-fallback: only the truthy set enables displacement; anything
    else (unset / blank / garbage) is off. No provider/model selection — no
    ``docs/LICENSING.md`` row (the seat it unlocks is already covered by the
    ``SEMANTIK_SPECIALIST_PROVIDER`` licensing row)."""
    raw = (
        os.environ.get("SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE") or ""
    ).strip().lower()
    return raw in _ENDPOINT_DISPLACE_TRUTHY


def resolve_structure_review_model() -> str:
    """Return the model ID for the Stage-5d structure-REVIEWER seat.

    Mirrors :func:`resolve_specialist_provider`'s parse-with-fallback
    posture and REUSES the endpoint config (the reviewer rides the same
    hosted large seat as the Stage-6 endpoint specialists). Resolution chain
    (high -> low):

        SEMANTIK_STRUCTURE_REVIEW_MODEL  (reviewer-specific override)
        > SEMANTIK_SPECIALIST_MODEL      (shared specialist seat)
        > NVIDIA_LARGE_MODEL             (NVIDIA seat default)
        > "meta/llama-3.3-70b-instruct"  (sane literal default)

    The literal default matches the Stage-6 endpoint runtime's
    ``_DEFAULT_MODEL`` so a reviewer + specialist on the same unconfigured
    seat resolve to the same large model. Key is NEVER read here (env-only,
    resolved at POST time by the endpoint runtime).
    """
    return (
        os.environ.get("SEMANTIK_STRUCTURE_REVIEW_MODEL")
        or os.environ.get("SEMANTIK_SPECIALIST_MODEL")
        or os.environ.get("NVIDIA_LARGE_MODEL")
        or "meta/llama-3.3-70b-instruct"
    )


class QwenRuntime(Protocol):
    """Generic runtime contract Stage 6 dispatches against."""

    def load(self, gguf_path: Path) -> None:
        """Bind a GGUF (real) or record the path (mock). Idempotent
        only insofar as a second load with the same path is a no-op;
        switching paths must ``free()`` first or the implementation
        is free to raise."""

    def free(self) -> None:
        """Release any GPU/CPU memory the runtime is holding. Safe to
        call when nothing is loaded."""

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
    ) -> list[str]:
        """Return ``n`` completions. Implementations MUST return a list
        of length ``n`` even if some completions are empty strings.

        ``repeat_penalty`` maps to llama-cpp-python's ``repeat_penalty``
        kwarg (HF ``repetition_penalty``). Default 1.0 == no penalty,
        which is also llama-cpp-python's own default, so omitting it
        leaves generation behaviour unchanged."""

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
        fail_soft: bool = False,
    ) -> list[str] | list[str | None]:
        """Return one completion per prompt, IN INPUT ORDER.

        Unlike :meth:`generate` (K samples of ONE prompt), this maps N
        prompts → N completions one-to-one. Used by the Stage-6 two-phase
        batched driver so a whole adapter group (local) or a whole region
        set (endpoint) is generated in a single batched pass rather than a
        per-region swap/POST loop. Implementations MUST return a list the
        SAME LENGTH as ``prompts`` and in the SAME ORDER.

        ``fail_soft`` (endpoint runtime only): when ``True``, a per-item
        failure returns the ``None`` sentinel in that slot instead of
        raising, so the caller degrades only that region. Local runtimes
        ignore it (a broken local model is a fail-loud condition)."""


# ---------------------------------------------------------------------------
# Mock runtime
# ---------------------------------------------------------------------------


class MockRuntime:
    """Deterministic stand-in. Useful for:

    * unit tests (no GGUF on disk, no GPU).
    * Stage 6 wiring smoke runs that prove the driver shape but don't
      need real-quality HTML.

    The output strings are intentionally distinct per ``i`` so callers
    can assert "K candidates came back" without false-positive
    duplicate dedup hiding a bug.
    """

    def __init__(self) -> None:
        self._loaded_path: Path | None = None
        self.load_calls: list[Path] = []  # observable for tests
        self.free_calls: int = 0  # observable for tests
        self.generate_calls: list[dict] = []  # observable for tests

    def load(self, gguf_path: Path) -> None:
        self._loaded_path = Path(gguf_path)
        self.load_calls.append(self._loaded_path)

    def free(self) -> None:
        self._loaded_path = None
        self.free_calls += 1

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
    ) -> list[str]:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "prompt_head": prompt[:80],
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed,
                "repeat_penalty": repeat_penalty,
            }
        )
        # Sniff the kind out of the JSON envelope (or a kind=foo
        # short-form some tests use) so the emitted HTML5 fragment is
        # at least kind-appropriate. Stage 7's html5 well-formedness
        # check rejected the previous ``<mock-output .../>`` strings
        # because they used a custom-element name with a stray closing
        # form; the per-kind fragments below all parse as well-formed
        # HTML5 + (for math) MathML.
        #
        # Each fragment carries the candidate index ``{i}`` so the
        # K outputs remain distinct strings — assert in
        # ``test_mock_runtime_basic`` requires len(set(out)) == n.
        # Every emitted candidate carries ``data-semantik-mock="true"`` on
        # its outer tag so a developer reading the assembled HTML can
        # tell at a glance the content is fabricated, not real-OCR
        # output. See feedback_no_silent_fallbacks.md — placeholder
        # strings that look like real content are themselves a silent
        # fallback. Stage 9a's heading-level rewrite preserves the
        # attribute (it copies ``open_m.group(2)`` verbatim); 9c's
        # missing-title and author-block splices replace whole tags
        # from the candidate text, so the attribute survives there too.
        _MARK = ' data-semantik-mock="true"'
        p = prompt.lower()
        if '"kind": "table"' in p or "kind=table" in p:
            # Surface the cell_roles_source marker on the outer <table>
            # so a developer reading assembled.html — and the Stage 12
            # theta evaluator — can see when header semantics were
            # guessed by the adapter without a verified BERT-
            # TableSpecialist signal. See build_table_request docstring
            # and feedback_no_silent_fallbacks.md. We grep the lower-
            # cased prompt because the JSON envelope embeds the marker
            # verbatim ("cell_roles_source": "qwen_inferred"). The real
            # adapter's prompt template is responsible for the same
            # surfacing in production.
            cell_roles_mark = (
                ' data-semantik-cell-roles="qwen-inferred"'
                if '"cell_roles_source": "qwen_inferred"' in p
                else ""
            )
            body = (
                "<table" + _MARK + cell_roles_mark + "><caption>mock</caption>"
                '<thead><tr><th scope="col">a</th></tr></thead>'
                "<tbody><tr><td>cell {i}</td></tr></tbody></table>"
            )
        elif '"kind": "math"' in p or "kind=math" in p:
            body = (
                '<math xmlns="http://www.w3.org/1998/Math/MathML"' + _MARK + " "
                'alttext="x equals {i}">'
                "<mi>x</mi><mo>=</mo><mn>{i}</mn></math>"
            )
        elif '"kind": "heading"' in p or "kind=heading" in p:
            body = "<h2" + _MARK + ">mock heading {i}</h2>"
        elif '"kind": "definition_list"' in p or "kind=definition_list" in p:
            body = "<dl" + _MARK + "><dt>term {i}</dt><dd>def {i}</dd></dl>"
        elif '"kind": "list"' in p or "kind=list" in p:
            body = "<ul" + _MARK + "><li>mock item {i}</li></ul>"
        elif '"kind": "code_block"' in p or "kind=code_block" in p:
            body = "<pre" + _MARK + "><code>mock code {i}</code></pre>"
        elif '"kind": "blockquote"' in p or "kind=blockquote" in p:
            body = "<blockquote" + _MARK + ">mock quote {i}</blockquote>"
        elif '"kind": "figure"' in p or "kind=figure" in p:
            body = "<figure" + _MARK + "><figcaption>mock figure {i}</figcaption></figure>"
        elif '"kind": "form"' in p or "kind=form" in p:
            body = (
                "<form" + _MARK + '><label for="m{i}">m</label><input id="m{i}" type="text"></form>'
            )
        elif '"kind": "metadata_drop"' in p or "kind=metadata_drop" in p:
            body = '<p class="metadata"' + _MARK + ">mock metadata {i}</p>"
        elif "kind=missing_title" in p:
            # Both spliced tags get the marker — the gap-fill candidate
            # text becomes both the doc <title> AND the body <h1>.
            body = "<title" + _MARK + ">Mock Title {i}</title><h1" + _MARK + ">Mock Title {i}</h1>"
        elif "kind=author_block" in p:
            body = "<address" + _MARK + ">Mock Author {i}</address>"
        elif "kind=copyright_block" in p or "kind=legal_disclaimer" in p:
            # Echo the raw_text out of the JSON envelope inside the
            # canonical <aside class="copyright|legal"> wrap so the 9c
            # splice (which requires an aside/footer root and, for
            # copyright, a surviving ©/copyright keyword) and the
            # per-gap scorer's diff-from-context check both accept the
            # fragment. Braces are escaped so ``body.format(i=i)``
            # below cannot choke; ``data-i`` keeps the K candidates
            # distinct (test_mock_runtime_basic: len(set(out)) == n).
            is_copyright = "kind=copyright_block" in p
            tm = _re_mod.search(r'"raw_text":\s*"([^"]*)"', prompt)
            default_text = "© Mock Copyright" if is_copyright else "Mock legal disclaimer text."
            raw = (tm.group(1) if tm else "") or default_text
            raw = raw.replace("{", "{{").replace("}", "}}")
            aside_class = "copyright" if is_copyright else "legal"
            body = (
                "<aside" + _MARK + f' class="{aside_class}" data-i="{{i}}">'
                "<p>" + raw + "</p></aside>"
            )
        elif "kind=citation_unresolved" in p:
            # Restore an anchor around the (mock) reference. The first
            # candidate_targets anchor_id in the prompt is the natural
            # mock choice — we sniff it out of the JSON envelope so the
            # 9c splice (which validates the chosen id against the gap's
            # candidate_targets) accepts the fragment. Falls back to a
            # plain-text passthrough when no anchor_id is present. We
            # escape literal braces out of the sniffed text so the
            # ``body.format(i=i)`` distinctness pass below cannot choke.
            mm = _re_mod.search(r'"anchor_id":\s*"([^"]+)"', prompt)
            tm = _re_mod.search(r'"match_text":\s*"([^"]*)"', prompt)
            raw_match = (tm.group(1) if tm else "ref") or "ref"
            match_text = raw_match.replace("{", "{{").replace("}", "}}")
            if mm:
                anchor_id = mm.group(1).replace("{", "{{").replace("}", "}}")
                # ``data-i="{i}"`` keeps the K candidates distinct
                # (test_mock_runtime_basic requires len(set(out)) == n).
                body = (
                    "<a" + _MARK + ' data-i="{i}" href="#' + anchor_id + '">' + match_text + "</a>"
                )
            else:
                # No candidate target → confirm "leave as plain text".
                body = match_text
        else:
            body = "<p" + _MARK + ">mock paragraph {i}</p>"
        return [body.format(i=i) for i in range(n)]

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
        fail_soft: bool = False,
    ) -> list[str]:
        """One completion per prompt, in order — maps over :meth:`generate`.

        Each prompt is generated with ``n=1`` so the per-kind fragment
        sniffing in :meth:`generate` still produces kind-appropriate
        well-formed output. The single completion (index 0) is returned per
        prompt, preserving input order. ``fail_soft`` is accepted for
        signature parity with the endpoint runtime and ignored (the mock
        never fails an item)."""
        out: list[str] = []
        for prompt in prompts:
            res = self.generate(
                prompt,
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                repeat_penalty=repeat_penalty,
            )
            out.append(res[0])
        return out


# ---------------------------------------------------------------------------
# Real runtime — llama-cpp-python wrapper
# ---------------------------------------------------------------------------


class LlamaCppRuntime:
    """Real-mode wrapper. ``llama-cpp-python`` is a lazy dependency.

    Use ``make_runtime("real")`` to instantiate via the factory; the
    constructor is also public for tests that monkey-patch the lazy
    import.
    """

    def __init__(self) -> None:
        self._handle = None
        self._loaded_path: Path | None = None
        self._tokenizer: Any | None = None
        self._adapter_version: str = "unknown"

    def _ensure_tokenizer(self) -> Any:
        """Lazy-load the Qwen tokenizer for chat-template wrapping.

        We need the HF tokenizer (not llama.cpp's BPE) because
        ``apply_chat_template`` is the canonical wrap, and reproducing
        Qwen3's chat template by hand drifts on whitespace.
        """
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer  # noqa: WPS433

        self._tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            trust_remote_code=True,
        )
        return self._tokenizer

    def load(self, gguf_path: Path) -> None:
        try:
            from llama_cpp import Llama  # noqa: WPS433  (lazy import)
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python not installed; install with "
                "`pip install llama-cpp-python` to enable real-mode "
                "generation"
            ) from exc
        gguf_path = Path(gguf_path)
        if not gguf_path.exists():
            raise FileNotFoundError(
                f"GGUF artifact missing: {gguf_path}\n"
                "Strict-by-default refuses to silently fall back to MockRuntime "
                "(see feedback_no_silent_fallbacks.md).\n"
                "To fix: train + convert + register the adapter with\n"
                "  scripts/qwen_lora_to_gguf.py     (HF LoRA -> GGUF)\n"
                "  scripts/register_qwen_adapter.py (write adapter_path into config.yaml)\n"
                "To opt back into the deterministic mock, pass "
                "runtime_mode='mock' (Stage 6 runner / assembler default)."
            )
        # n_gpu_layers=-1 → offload everything; honoured iff the binary
        # was built with CUDA. verbose=False keeps stderr clean for the
        # smoke harness.
        #
        # n_ctx=4096: llama-cpp-python defaults to n_ctx=512, which silently
        # caps prompt+generation at 512 tokens. Real MathML targets reach
        # ~919 tokens and the math adapter generates up to max_new_tokens
        # 960 (1024 offline), so the 512 default truncates long display /
        # numbered / multiline equations the eval scored as complete (the
        # eval ran via llama-server with a larger context). 4096 holds the
        # region prompt + the largest configured generation with headroom;
        # KV cache at this ctx is a few hundred MB on the 8GB 3070.
        self._handle = Llama(
            model_path=str(gguf_path),
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        self._loaded_path = gguf_path
        self._adapter_version = self._read_adapter_version(gguf_path)

    @staticmethod
    def _read_adapter_version(gguf_path: Path) -> str:
        """Read the sibling ``.metadata.json`` for ``adapter_version``.

        Missing or unreadable metadata -> ``"unknown"`` and a WARNING.
        We log loudly because an unversioned adapter in production
        means a future regression has nothing to bisect against.
        """
        meta_path = gguf_path.with_suffix(gguf_path.suffix + ".metadata.json")
        if not meta_path.exists():
            # Try alternate sibling name: <stem>.metadata.json
            alt = gguf_path.with_suffix(".metadata.json")
            meta_path = alt if alt.exists() else meta_path
        if not meta_path.exists():
            logger.warning(
                "no adapter metadata at %s or %s; adapter_version=unknown",
                gguf_path.with_suffix(gguf_path.suffix + ".metadata.json"),
                gguf_path.with_suffix(".metadata.json"),
            )
            return "unknown"
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            version = data.get("adapter_version") or "unknown"
            return str(version)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to read adapter metadata %s: %s", meta_path, exc)
            return "unknown"

    def free(self) -> None:
        if self._handle is None:
            return
        del self._handle
        self._handle = None
        self._loaded_path = None
        # Don't drop _adapter_version here; downstream may want to
        # query it on the most-recently-loaded adapter even after free.
        # Best-effort GPU/CUDA cache release. We import gc + torch lazily
        # so the runtime itself doesn't carry torch as a hard dep.
        import gc  # noqa: WPS433

        gc.collect()
        try:
            import torch  # noqa: WPS433

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
    ) -> list[str]:
        if self._handle is None:
            raise RuntimeError(
                "LlamaCppRuntime.generate(): no GGUF loaded; call load(gguf_path) first"
            )
        # Chat-template wrap matches training (see chat_format.py).
        # Without this, the trained adapter's first generated token is
        # off-by-one from training and quality collapses silently.
        from .chat_format import wrap_for_qwen  # noqa: WPS433

        tok = self._ensure_tokenizer()
        wrapped = wrap_for_qwen(tok, prompt, add_generation_prompt=True)
        outputs: list[str] = []
        # llama-cpp-python's __call__ is single-completion. Loop K times.
        # Per-iteration seed offset keeps the K samples reproducible
        # without collapsing them onto identical text.
        for i in range(n):
            this_seed = -1 if seed is None else int(seed + i)
            res = self._handle(
                wrapped,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=this_seed,
                repeat_penalty=repeat_penalty,
            )
            outputs.append(res["choices"][0]["text"])
        return outputs

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
        repeat_penalty: float = 1.0,
        fail_soft: bool = False,
    ) -> list[str]:
        """One completion per prompt, sequentially, over the LOADED adapter.

        The GPU adapter is already resident (the Stage-6 Phase-1 driver
        loads the adapter ONCE via :class:`AdapterSwap`, then calls this
        once for the whole adapter group — no swap inside the batch). We
        loop ``generate(n=1)`` per prompt so each region gets one draft in
        input order; the 8 GB card cannot run a true parallel batch under
        llama-cpp-python, so sequential-over-resident-adapter is the win
        (one load amortized across the whole group) without a parallel
        decode the hardware can't support."""
        if self._handle is None:
            raise RuntimeError(
                "LlamaCppRuntime.generate_batch(): no GGUF loaded; call "
                "load(gguf_path) first"
            )
        outputs: list[str] = []
        for j, prompt in enumerate(prompts):
            # Per-prompt seed offset keeps drafts reproducible without
            # collapsing distinct regions onto identical text.
            this_seed = None if seed is None else int(seed + j)
            res = self.generate(
                prompt,
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=this_seed,
                repeat_penalty=repeat_penalty,
            )
            outputs.append(res[0])
        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_DEFAULT_QWEN_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _scan_adapter_artifacts(
    config_path: Path | None = None,
) -> tuple[dict[str, Path | None], dict[str, bool]]:
    """Read ``config.yaml`` and report each adapter's configured GGUF
    path + on-disk presence.

    Returns ``(paths, present)`` — ``paths[adapter] = Path | None`` (None
    when ``adapter_path: null``) and ``present[adapter] = bool``.

    Used by :func:`make_runtime` to fail loudly at construction when
    real-mode is requested but no trained adapter exists on disk —
    mirroring the strict-by-default theta sentinel pattern in
    :mod:`semantik_structure.theta._module_state`.
    """
    path = Path(config_path) if config_path else _DEFAULT_QWEN_CONFIG_PATH
    try:
        import yaml  # noqa: WPS433  (lazy import; mock mode shouldn't need yaml)

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        return ({}, {})
    adapters_cfg = (cfg.get("adapters") or {}) if isinstance(cfg, dict) else {}
    paths: dict[str, Path | None] = {}
    present: dict[str, bool] = {}
    for name, entry in adapters_cfg.items():
        raw = (entry or {}).get("adapter_path") if isinstance(entry, dict) else None
        if raw is None:
            paths[name] = None
            present[name] = False
        else:
            # Anchor the config-stored ``models/<...>.gguf`` literal through the
            # SemantiK model root so the presence check is CWD-independent and
            # honors SEMANTIK_MODEL_DIR (byte-stable to the legacy layout when
            # no env var is set). MUST match base.py::_gguf_path_for's resolution
            # so the preflight presence check and the live load agree.
            p = _semantik_paths.resolve_model_literal(raw)
            paths[name] = p
            present[name] = p.exists()
    return paths, present


def make_runtime(
    mode: Literal["mock", "real", "endpoint"],
    *,
    config_path: Path | None = None,
    model: str | None = None,
    force_local: bool = False,
) -> QwenRuntime:
    """Return a fresh runtime instance for the requested mode.

    The mode string is a simple discriminator — keeping the factory
    here means callers (the Stage 6 runner; future Stage 9 gap-fill)
    don't need to know the backend class names.

    **Endpoint provider short-circuit.** When the operator has opted into
    a hosted specialist endpoint (``SEMANTIK_SPECIALIST_PROVIDER`` set to
    a non-``local`` value), ``mode="real"`` (or explicit ``"endpoint"``)
    returns an :class:`~.endpoint_runtime.OpenAICompatibleRuntime` and the
    local-GGUF presence check is SKIPPED — there are no local weights to
    require. The strict-by-default GGUF check below applies ONLY to the
    local arm.

    **Optional ``model`` override (endpoint arm only).** When ``model`` is
    given, it is passed verbatim to
    :class:`~.endpoint_runtime.OpenAICompatibleRuntime` so the seat resolves
    to THAT model id instead of the Stage-6 specialist default — this is how
    the Stage-5d structure reviewer pins its dedicated
    ``SEMANTIK_STRUCTURE_REVIEW_MODEL`` seat (see
    :func:`resolve_structure_review_model`). ``None`` (the default, used by
    the Stage-6 specialist path) preserves the byte-stable env-resolved model
    (``SEMANTIK_SPECIALIST_MODEL`` > ``NVIDIA_LARGE_MODEL`` > default). It is
    ignored by the ``mock`` / local-``real`` arms (no hosted model to pin).

    **Strict-by-default for the LOCAL ``mode="real"`` arm.** If
    ``config.yaml`` has no adapter pointing at an on-disk GGUF,
    ``make_runtime("real")`` raises ``RuntimeError`` with build-it /
    opt-back-to-mock guidance. Silent fall-back to :class:`MockRuntime` is
    forbidden — see feedback_no_silent_fallbacks.md ("do NOT silently fall
    back to mock when real adapters are missing — let the runtime
    constructor fail").
    """
    if mode == "mock":
        return MockRuntime()
    # Endpoint arm: explicit "endpoint" mode, OR "real" mode while an
    # endpoint provider is selected. No local GGUF is needed or checked.
    #
    # ``force_local`` (dereliction fix): the Stage-6 caller sets this when the
    # local GGUF specialists are the active authoring tier even though an
    # endpoint PROVIDER is configured (the corrected default — see
    # :func:`resolve_endpoint_displaces_local`). It only suppresses the
    # ``mode="real"`` short-circuit; an explicit ``mode="endpoint"`` (the
    # Stage-5d reviewer / Stage-5e resegment seats) is never forced local.
    if mode == "endpoint" or (
        mode == "real" and not force_local and specialist_provider_is_endpoint()
    ):
        from .endpoint_runtime import OpenAICompatibleRuntime  # noqa: WPS433

        logger.info(
            "make_runtime(%r): routing Stage-6 specialists to hosted "
            "endpoint (provider=%s); local GGUF skipped",
            mode,
            resolve_specialist_provider(),
        )
        return OpenAICompatibleRuntime(model=model)
    if mode == "real":
        cfg_path = Path(config_path) if config_path else _DEFAULT_QWEN_CONFIG_PATH
        paths, present = _scan_adapter_artifacts(cfg_path)
        any_present = any(present.values())
        if not any_present:
            # Build a per-adapter status report so the operator can see
            # exactly which artifacts are missing vs un-configured.
            lines: list[str] = []
            if paths:
                for name in sorted(paths.keys()):
                    p = paths[name]
                    if p is None:
                        lines.append(f"  - {name}: adapter_path: null (un-configured)")
                    else:
                        status = "present" if present.get(name) else "MISSING"
                        lines.append(f"  - {name}: {p} ({status})")
            else:
                lines.append(f"  (config not readable at {cfg_path})")
            status_block = "\n".join(lines)
            raise RuntimeError(
                "make_runtime('real'): no Qwen LoRA adapters are on disk; "
                "strict-by-default refuses to silently fall back to MockRuntime "
                "(see feedback_no_silent_fallbacks.md).\n"
                f"Adapter status (from {cfg_path}):\n{status_block}\n"
                "To fix: train + convert + register adapters with\n"
                "  scripts/qwen_lora_to_gguf.py     (HF LoRA -> GGUF)\n"
                "  scripts/register_qwen_adapter.py (write adapter_path into config.yaml)\n"
                "To opt back into the deterministic mock for tests / smoke runs, "
                "pass runtime_mode='mock' (the assembler / Stage 6 runner default)."
            )
        return LlamaCppRuntime()
    raise ValueError(
        f"unknown runtime mode: {mode!r} (expected 'mock' | 'real' | 'endpoint')"
    )


def make_phase_runtime(
    mode: Literal["mock", "real"],
    phase: str,
    *,
    config_path: Path | None = None,
) -> QwenRuntime:
    """Build the runtime for a Stage-6 phase from its OWN resolved seat.

    Model-agnostic: each phase independently resolves ``{provider, model}``
    (:func:`resolve_phase_provider` / :func:`resolve_phase_model`) and the
    runtime is chosen from the SAME registry mechanism as everything else —

      * ``mode="mock"``           -> :class:`MockRuntime` (tests / smoke; no
        provider read — never a GPU/network touch).
      * provider ``"local"``      -> the local GGUF arm (:class:`LlamaCppRuntime`
        via ``make_runtime("real", force_local=True)``, including the strict
        on-disk-adapter presence check). This is the byte-stable default for
        Phase 1; a LOCAL Phase-2 seat is equally valid (local-draft→local-
        refine with NO hosted call).
      * any other provider        -> the OpenAI-compatible endpoint runtime
        (:class:`~.endpoint_runtime.OpenAICompatibleRuntime`) pointed at that
        seat's ``model`` (base_url / api_key resolved from the shared
        ``SEMANTIK_SPECIALIST_BASE_URL`` / ``_API_KEY`` > ``NVIDIA_*`` env). The
        hosted large seat is just one possible value of ``model`` here — never
        a special-cased tier.

    The phase's provider decides local-vs-endpoint; ``mode`` only chooses the
    deterministic mock for tests. No large-model id is hardcoded as a tier
    anywhere on this path."""
    if phase not in _VALID_PHASES:
        raise ValueError(f"unknown phase {phase!r} (expected one of {_VALID_PHASES})")
    if mode == "mock":
        return MockRuntime()
    provider = resolve_phase_provider(phase)
    if provider == "local":
        # Local GGUF arm — strict presence check, never the endpoint
        # short-circuit (force_local suppresses it).
        return make_runtime("real", config_path=config_path, force_local=True)
    # Any OpenAI-compatible provider: one backend, driven by the seat's model
    # (+ the shared base_url/api_key the endpoint runtime resolves from env).
    logger.info(
        "make_phase_runtime(%s): routing to endpoint seat (provider=%s, model=%s)",
        phase,
        provider,
        resolve_phase_model(phase),
    )
    return make_runtime(
        "endpoint", config_path=config_path, model=resolve_phase_model(phase)
    )


__all__ = [
    "LlamaCppRuntime",
    "MockRuntime",
    "QwenRuntime",
    "any_phase_provider_is_endpoint",
    "make_phase_runtime",
    "make_runtime",
    "phase_provider_is_endpoint",
    "resolve_endpoint_displaces_local",
    "resolve_phase_model",
    "resolve_phase_provider",
    "resolve_specialist_provider",
    "resolve_structure_review_model",
    "specialist_provider_is_endpoint",
]
