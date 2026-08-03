"""COURSEFORGE_CURIE_DETERMINISTIC — fully-deterministic rewrite-tier CURIEs.

Owner scope (supersedes the earlier post-mint-on-failure design): CURIEs must
be handled ENTIRELY deterministically — the rewrite model is never asked to
recite them. Under the flag:

  1. the model-facing CURIE directive is stripped from the system prompt AND
     the serialised outline payload the model sees (no CURIE instruction, no
     CURIE tokens in the prompt);
  2. the CURIE-preservation retry ladder is REMOVED (a CURIE miss never
     triggers a re-roll — structural dispatch / truncation retries stay);
  3. every vocabulary-resolvable outline CURIE is stamped onto the emit
     deterministically post-generation, so ``rewrite_curie_anchoring`` passes;
  4. anti-fabrication: only CURIEs whose token is a key in the course
     ``domain_concept_vocabulary`` minted-CURIE map are stamped.

Default OFF → byte-identical to the legacy CURIE-preservation retry ladder.

Mirrors the helper conventions in ``test_rewrite_provider.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._rewrite_provider import (  # noqa: E402
    ENV_PROVIDER,
    RewriteProvider,
    _TOUCH_PURPOSE_CURIE_DETERMINISTIC,
)
from Courseforge.generators._rewrite_fit_window import (  # noqa: E402
    ENV_CURIE_DETERMINISTIC,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-rewrite") -> dict:
    return {
        "id": "cmpl-rewrite-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        # A realistic non-truncated prompt-token tally so the default-ON
        # input-truncation tripwire does not (correctly) trip.
        "usage": {
            "prompt_tokens": 8800,
            "completion_tokens": 80,
            "total_tokens": 8880,
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _outline_block(*, curies: List[str] | None = None) -> Block:
    return Block(
        block_id="page#concept_intro_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content={
            "key_claims": ["The central concept is X."],
            "curies": list(curies or []),
            "source_refs": ["semantik:slug#blk1"],
            "objective_refs": ["TO-01"],
        },
    )


def _minted_map(*tokens: str) -> Dict[str, Any]:
    """A minimal minted-CURIE map keyed by token (the anti-fabrication
    universe). Surface forms are the localname humanised, so the gate's
    minted arm can also anchor via surface form if needed."""
    out: Dict[str, Any] = {}
    for tok in tokens:
        local = tok.split(":", 1)[1] if ":" in tok else tok
        out[tok] = {
            "surface_forms": [local.replace("_", " ").replace("-", " ")],
        }
    return out


def _curie_survives_validator_path(html: str, curie: str) -> bool:
    from Courseforge.router.inter_tier_gates import _strip_html
    from lib.ontology.curie_extraction import extract_curies

    return curie in extract_curies(_strip_html(html))


class _SpyCapture:
    """Minimal DecisionCapture stand-in: records every log_decision call."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


_DROP_CURIE_HTML = "<section><p>The node shape constrains.</p></section>"


def _local_env(monkeypatch) -> None:
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")


# ---------------------------------------------------------------------------
# (a) flag OFF → retry ladder byte-identical
# ---------------------------------------------------------------------------


def test_flag_off_curie_miss_still_runs_retry_ladder(monkeypatch):
    """Flag unset: a persistent CURIE drop runs the legacy ladder — initial
    dispatch + MAX_PARSE_RETRIES re-rolls = 3 total dispatches, then the
    exhaustion force-inject. Byte-identical to the pre-flag contract."""
    monkeypatch.delenv(ENV_CURIE_DETERMINISTIC, raising=False)
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_outline_block(curies=["sh:NodeShape"]))

    # Legacy ladder: 1 initial + 2 remediation retries = 3 dispatches.
    assert len(seen) == 3
    # A CURIE remediation directive was appended on a re-roll.
    assert any(
        "did not include the required" in r.read().decode("utf-8")
        for r in seen
    )
    # Exhaustion force-inject still lands the CURIE (unchanged contract).
    assert _curie_survives_validator_path(out.content, "sh:NodeShape")


# ---------------------------------------------------------------------------
# (b) flag ON + CURIE-only failure → ZERO CURIE retries, stamped, gate passes
# ---------------------------------------------------------------------------


def test_flag_on_zero_curie_retries_and_stamps(monkeypatch):
    """Flag on: the model drops the CURIE, but there is NO CURIE-driven
    re-roll — exactly ONE dispatch — and the vocabulary-resolvable CURIE is
    stamped deterministically so it survives the validator path."""
    monkeypatch.setenv(ENV_CURIE_DETERMINISTIC, "1")
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
        minted_curie_map=_minted_map("tstcourse101:node_shape"),
    )
    block = _outline_block(curies=["tstcourse101:node_shape"])
    out = p.generate_rewrite(block)

    assert len(seen) == 1, "flag on must never fire a CURIE-driven re-roll"
    assert _curie_survives_validator_path(out.content, "tstcourse101:node_shape")
    # Touch chain records the deterministic-stamp provenance.
    assert out.touched_by[-1].purpose == _TOUCH_PURPOSE_CURIE_DETERMINISTIC


def test_flag_on_stamped_output_passes_anchoring_gate(monkeypatch):
    """The deterministically-stamped output PASSES
    ``BlockCurieAnchoringValidator`` (with the minted map threaded, exactly
    as the post_rewrite_validation phase does)."""
    monkeypatch.setenv(ENV_CURIE_DETERMINISTIC, "1")
    _local_env(monkeypatch)

    minted = _minted_map("tstcourse101:node_shape")
    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))
        ),
        minted_curie_map=minted,
    )
    out = p.generate_rewrite(_outline_block(curies=["tstcourse101:node_shape"]))

    from Courseforge.router.inter_tier_gates import BlockCurieAnchoringValidator

    result = BlockCurieAnchoringValidator().validate(
        {"blocks": [out], "minted_curie_map": minted}
    )
    assert result.passed, [i.code for i in result.issues]


# ---------------------------------------------------------------------------
# flag ON → model prompt carries NO CURIE instruction / token
# ---------------------------------------------------------------------------


def test_flag_on_prompt_has_no_curie_instruction_or_token(monkeypatch):
    """Flag on: neither the system prompt nor the user prompt (POSTed to the
    server) mentions CURIEs or carries the CURIE token — the model is never
    asked to recite them."""
    monkeypatch.setenv(ENV_CURIE_DETERMINISTIC, "1")
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
        minted_curie_map=_minted_map("tstcourse101:node_shape"),
    )
    # The system prompt itself carries no CURIE mention.
    assert "curie" not in p._system_prompt.lower()

    p.generate_rewrite(_outline_block(curies=["tstcourse101:node_shape"]))
    assert len(seen) == 1
    wire = seen[0].read().decode("utf-8").lower()
    assert "curie" not in wire, "user prompt must not mention CURIEs"
    assert "node_shape" not in wire, "CURIE token must not leak into the prompt"


def test_flag_off_prompt_keeps_curie_directive(monkeypatch):
    """Flag off: the system prompt keeps its CURIE directive (byte-identical
    legacy behaviour)."""
    monkeypatch.delenv(ENV_CURIE_DETERMINISTIC, raising=False)
    _local_env(monkeypatch)
    p = RewriteProvider(provider="local")
    assert "curie" in p._system_prompt.lower()


# ---------------------------------------------------------------------------
# (c) flag ON + a STRUCTURAL (dispatch) failure → structural retry preserved
# ---------------------------------------------------------------------------


def test_flag_on_structural_transient_retry_preserved(monkeypatch):
    """Flag on removes only the CURIE ladder — a TRANSIENT dispatch failure
    still retries structurally. Two transient errors then success = 3
    dispatches, and the CURIE is still stamped deterministically on the
    surviving emit."""
    monkeypatch.setenv(ENV_CURIE_DETERMINISTIC, "1")
    _local_env(monkeypatch)

    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))
        ),
        minted_curie_map=_minted_map("tstcourse101:node_shape"),
    )

    call_count = {"n": 0}
    real_dispatch = p._dispatch_call

    def fake_dispatch(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise ConnectionError("connection reset by peer")
        return real_dispatch(*args, **kwargs)

    monkeypatch.setattr(p, "_dispatch_call", fake_dispatch)
    out = p.generate_rewrite(_outline_block(curies=["tstcourse101:node_shape"]))

    assert call_count["n"] == 3, "structural transient retries must survive"
    assert _curie_survives_validator_path(out.content, "tstcourse101:node_shape")


# ---------------------------------------------------------------------------
# (d) anti-fabrication: an unmintable token is skipped (never stamped)
# ---------------------------------------------------------------------------


def test_flag_on_unmintable_token_is_not_stamped(monkeypatch):
    """Flag on: only vocabulary-resolvable CURIEs are stamped. A token that
    is NOT a key in the minted map is silently skipped (anti-fabrication);
    the resolvable one is stamped. Still exactly one dispatch (no ladder)."""
    monkeypatch.setenv(ENV_CURIE_DETERMINISTIC, "1")
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
        # Only ``node_shape`` resolves; ``not_a_concept`` is unmintable.
        minted_curie_map=_minted_map("tstcourse101:node_shape"),
    )
    out = p.generate_rewrite(
        _outline_block(
            curies=["tstcourse101:node_shape", "tstcourse101:not_a_concept"],
        )
    )

    assert len(seen) == 1
    assert _curie_survives_validator_path(out.content, "tstcourse101:node_shape")
    assert not _curie_survives_validator_path(
        out.content, "tstcourse101:not_a_concept"
    ), "an unmintable token must never be fabricated onto the emit"


def test_flag_on_empty_minted_map_stamps_nothing(monkeypatch):
    """Flag on but NO vocabulary (RDF / legacy corpus) → nothing is
    stampable; still exactly one dispatch (no CURIE ladder)."""
    monkeypatch.setenv(ENV_CURIE_DETERMINISTIC, "1")
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    p = RewriteProvider(
        provider="local",
        client=_make_client(handler),
        minted_curie_map=None,
    )
    out = p.generate_rewrite(_outline_block(curies=["sh:NodeShape"]))
    assert len(seen) == 1
    assert not _curie_survives_validator_path(out.content, "sh:NodeShape")


# ---------------------------------------------------------------------------
# (e) decision capture fires with a dynamic rationale + the minted token list
# ---------------------------------------------------------------------------


def test_flag_on_capture_fires_curie_minting_with_dynamic_rationale(
    monkeypatch,
):
    """Flag on: a ``curie_minting`` decision fires carrying the stamped token
    list and per-call dynamic signals (block id + provider) in the rationale
    — no static boilerplate."""
    monkeypatch.setenv(ENV_CURIE_DETERMINISTIC, "1")
    _local_env(monkeypatch)

    spy = _SpyCapture()
    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))
        ),
        capture=spy,
        minted_curie_map=_minted_map("tstcourse101:node_shape"),
    )
    p.generate_rewrite(_outline_block(curies=["tstcourse101:node_shape"]))

    minting = [c for c in spy.calls if c.get("decision_type") == "curie_minting"]
    assert len(minting) == 1, "exactly one curie_minting decision expected"
    rationale = minting[0]["rationale"]
    assert len(rationale) >= 20
    assert "tstcourse101:node_shape" in rationale  # the minted token list
    assert "page#concept_intro_0" in rationale     # dynamic block id
    assert "COURSEFORGE_CURIE_DETERMINISTIC" in rationale
