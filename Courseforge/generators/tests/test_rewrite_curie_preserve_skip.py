"""COURSEFORGE_CURIE_PRESERVE_SKIP_WHEN_POSTMINT — postmint-aware CURIE skip.

The per-block ``generate_rewrite`` path runs a CURIE-preservation retry ladder:
a rewrite HTML that drops an enforceable CURIE token RE-ROLLS the whole block
up to ``MAX_PARSE_RETRIES`` times, then force-injects (postmints) the still-
missing set anyway at exhaustion. That trailing force-inject anchors EXACTLY
the enforceable set — exactly what the preservation gate + the downstream
``BlockCurieAnchoringValidator`` str path enforce — so the re-rolls buy nothing
the postmint doesn't already deliver.

Under this flag a missing-CURIE candidate is postminted + accepted immediately
(zero CURIE re-roll), subject to every OTHER (router-side) gate. Default OFF →
byte-identical to the legacy retry ladder.

Mirrors the harness conventions in ``test_rewrite_curie_deterministic.py``.
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
    ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT,
    ENV_PROVIDER,
    MAX_PARSE_RETRIES,
    RewriteProvider,
    _TOUCH_PURPOSE_CURIE_FORCED,
    resolve_curie_preserve_skip_when_postmint,
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
        # Realistic non-truncated prompt tally so the default-ON input-
        # truncation tripwire does not (correctly) trip.
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
            "source_refs": ["dart:slug#blk1"],
            "objective_refs": ["TO-01"],
        },
    )


def _curie_survives_validator_path(html: str, curie: str) -> bool:
    from Courseforge.router.inter_tier_gates import _strip_html
    from lib.ontology.curie_extraction import extract_curies

    return curie in extract_curies(_strip_html(html))


class _SpyCapture:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


# HTML that omits the ``sh:NodeShape`` TOKEN, so the preservation gate flags a
# miss on every dispatch (drives the retry ladder when the skip is off).
_DROP_CURIE_HTML = "<section><p>The constraint restricts values.</p></section>"


def _local_env(monkeypatch) -> None:
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv(ENV_CURIE_DETERMINISTIC, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")


# ---------------------------------------------------------------------------
# resolver parse-with-fallback
# ---------------------------------------------------------------------------


def test_resolver_parse_with_fallback(monkeypatch):
    for truthy in ("1", "true", "TRUE", "Yes", "on", "  on  "):
        monkeypatch.setenv(ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT, truthy)
        assert resolve_curie_preserve_skip_when_postmint() is True
    for falsey in ("0", "false", "no", "off", "garbage", ""):
        monkeypatch.setenv(ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT, falsey)
        assert resolve_curie_preserve_skip_when_postmint() is False
    monkeypatch.delenv(ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT, raising=False)
    assert resolve_curie_preserve_skip_when_postmint() is False


# ---------------------------------------------------------------------------
# (a) flag OFF → retry ladder byte-identical
# ---------------------------------------------------------------------------


def test_flag_off_curie_miss_runs_full_retry_ladder(monkeypatch):
    """Flag unset: a persistent CURIE drop runs the legacy ladder — initial
    dispatch + MAX_PARSE_RETRIES re-rolls = 3 total dispatches — then the
    exhaustion force-inject lands the token. Byte-identical pre-flag."""
    monkeypatch.delenv(ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT, raising=False)
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_outline_block(curies=["sh:NodeShape"]))

    assert len(seen) == MAX_PARSE_RETRIES + 1  # 1 initial + 2 re-rolls = 3
    assert _curie_survives_validator_path(out.content, "sh:NodeShape")


# ---------------------------------------------------------------------------
# (b) flag ON + CURIE-only miss → ZERO CURIE re-rolls, postminted, survives
# ---------------------------------------------------------------------------


def test_flag_on_curie_miss_skips_retry_and_postmints(monkeypatch):
    """Flag on: the model drops the CURIE token, but there is NO CURIE-driven
    re-roll — exactly ONE dispatch — and the enforceable CURIE is postminted so
    it survives the validator path (exactly what the retry ladder would have
    delivered at exhaustion, minus the churn)."""
    monkeypatch.setenv(ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT, "1")
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_outline_block(curies=["sh:NodeShape"]))

    # Exactly one dispatch — the CURIE-preservation retry was SKIPPED.
    assert len(seen) == 1
    # No CURIE remediation directive was ever appended (no re-roll happened).
    assert not any(
        "did not include the required" in r.read().decode("utf-8")
        for r in seen
    )
    # Postmint anchored the token — identical anchoring to the exhaustion path.
    assert _curie_survives_validator_path(out.content, "sh:NodeShape")
    # Touch provenance is the force-injected marker.
    assert out.touched_by[-1].purpose == _TOUCH_PURPOSE_CURIE_FORCED


def test_flag_on_records_skip_in_decision_capture(monkeypatch):
    """The skip is audit-captured: a ``block_rewrite_call`` decision names the
    postmint skip in its rationale."""
    monkeypatch.setenv(ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT, "on")
    _local_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    cap = _SpyCapture()
    p = RewriteProvider(
        provider="local", client=_make_client(handler), capture=cap
    )
    p.generate_rewrite(_outline_block(curies=["sh:NodeShape"]))

    rewrite_calls = [
        c for c in cap.calls
        if c.get("decision_type") == "block_rewrite_call"
    ]
    assert rewrite_calls
    assert any(
        "PRESERVE_SKIP_WHEN_POSTMINT" in c.get("rationale", "")
        for c in rewrite_calls
    )


def test_flag_on_no_curie_drop_is_normal_single_dispatch(monkeypatch):
    """Flag on but the model DID keep the token → the miss branch never fires;
    the normal single-dispatch success path runs (no behavioural change when
    there is nothing to skip)."""
    monkeypatch.setenv(ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT, "true")
    _local_env(monkeypatch)

    good_html = (
        "<section><p>The <code>sh:NodeShape</code> constrains nodes.</p>"
        "</section>"
    )
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(good_html))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_outline_block(curies=["sh:NodeShape"]))

    assert len(seen) == 1
    assert _curie_survives_validator_path(out.content, "sh:NodeShape")


def test_flag_on_no_curies_declared_is_byte_identical(monkeypatch):
    """A block declaring NO CURIEs never enters the preservation branch, so the
    flag is a strict no-op there (single dispatch, no postmint)."""
    monkeypatch.setenv(ENV_CURIE_PRESERVE_SKIP_WHEN_POSTMINT, "1")
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_DROP_CURIE_HTML))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_outline_block(curies=[]))

    assert len(seen) == 1
    assert isinstance(out.content, str)
