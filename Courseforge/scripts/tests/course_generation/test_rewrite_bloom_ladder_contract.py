"""Bloom-ladder WI-08 — rewrite-tier per-rung misconception contract +
fit-window registration.

``ED4ALL_BLOOM_LADDER`` (default OFF) gates an ADDITIVE suffix onto the
existing rewrite-tier misconception output contract
(``Courseforge/generators/_rewrite_provider.py::_BLOCK_TYPE_OUTPUT_CONTRACTS
["misconception"]``): when a ``misconception`` block carries a resolvable
``bloom_level`` (the ladder rung), the rewrite-tier prompt is told to (a)
stamp ``data-cf-bloom-level="<rung>"`` on the ``misconception-card`` wrapper
and (b) author the claim/correction from the rung's misconception-probe
strategy (``lib.ontology.bloom_ladder``, the WI-01 taxonomy), phrased from
source-adjacent distractor-style wording so it survives the Arm-A
``source_backs`` token-subset check
(``Trainforge/generators/staged_synthesis_micro.py::source_backs``). A
``misconception_card_authoring`` decision fires per successfully-authored
card via the inherited ``_base.py::_emit_decision`` seam.

Coverage:

- Flag OFF (default/unset/garbage): the output contract + rendered prompts
  are byte-identical to pre-WI-08, and ``generate_rewrite`` emits only the
  pre-existing ``block_rewrite_call`` decision — never
  ``misconception_card_authoring``.
- Flag ON, non-misconception block: no-op (byte-identical) — the suffix and
  decision are scoped to ``block_type == "misconception"`` only.
- Flag ON, misconception block with NO resolvable ``bloom_level``: no-op —
  never fabricates a rung.
- Flag ON, misconception block at a valid rung: the suffix + the rendered
  user prompt carry the ``data-cf-bloom-level`` stamp instruction and the
  rung's probe_strategy guidance; the existing ``misconception-card`` /
  ``misconception-claim`` / ``misconception-correction`` class-token
  contract stays byte-identical (additive only); a
  ``misconception_card_authoring`` decision fires with a dynamic (>=20
  char) rationale interpolating the block id, rung, and probe_strategy.
- Every one of the six canonical rungs resolves to its own probe_strategy
  (mirrors ``lib.ontology.bloom_ladder.misconception_probe_for``).
- ``_RELOCATED_SEGMENTS_BY_BLOCK_TYPE`` (the ``ED4ALL_REWRITE_FIT_WINDOW``
  relocation map) still has no "misconception" entry, and the fit-window
  scaffold-overflow guard still fires on an over-window misconception
  scaffold — the new guidance cannot silently re-trip the 8192-window
  overflow class.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from Courseforge.generators._rewrite_provider import (  # noqa: E402
    ENV_PROVIDER,
    RewriteProvider,
    _BLOCK_TYPE_OUTPUT_CONTRACTS,
    _RELOCATED_SEGMENTS_BY_BLOCK_TYPE,
    _block_type_output_contract,
    _misconception_ladder_probe_contract_suffix,
)
from lib.generation.bloom_ladder_blocks import ENV_BLOOM_LADDER  # noqa: E402
from lib.ontology.bloom_ladder import (  # noqa: E402
    PROBE_STRATEGIES,
    misconception_probe_for,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrors the RewriteProvider test conventions in
# Courseforge/generators/tests/test_rewrite_curie_deterministic.py)
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-rewrite") -> dict:
    return {
        "id": "cmpl-bloom-ladder-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        # A realistic non-truncated prompt-token tally so the default-ON
        # input-truncation tripwire never (correctly) trips on this stub.
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


def _misconception_block(*, bloom_level: str | None = "analyze") -> Block:
    return Block(
        block_id="week_01_content_02#misconception_carrying_0",
        block_type="misconception",
        page_id="week_01_content_02",
        sequence=0,
        content={
            "key_claims": ["Learners often cancel terms instead of factors."],
            "curies": [],
            "source_refs": ["semantik:slug#blk1"],
            "objective_refs": ["CO-01"],
        },
        bloom_level=bloom_level,
    )


def _concept_block(*, bloom_level: str | None = "analyze") -> Block:
    return Block(
        block_id="week_01_content_02#concept_ratio_0",
        block_type="concept",
        page_id="week_01_content_02",
        sequence=0,
        content={
            "key_claims": ["Ratios compare two quantities."],
            "curies": [],
            "source_refs": ["semantik:slug#blk1"],
            "objective_refs": ["CO-01"],
        },
        bloom_level=bloom_level,
    )


_MISCONCEPTION_CARD_HTML = (
    '<div class="misconception-card" data-cf-source-ids="semantik:slug#blk1">'
    '<p class="misconception-claim">Canceling terms instead of factors keeps '
    "the fraction's value.</p>"
    '<p class="misconception-correction">Only a common FACTOR of numerator '
    "and denominator may be canceled; canceling a term changes the value."
    "</p></div>"
)


class _SpyCapture:
    """Minimal DecisionCapture stand-in: records every log_decision call."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _local_env(monkeypatch) -> None:
    monkeypatch.delenv(ENV_PROVIDER, raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")


# ---------------------------------------------------------------------------
# (a) The output-contract suffix: flag + block-type + rung gating
# ---------------------------------------------------------------------------


def test_suffix_empty_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    assert _misconception_ladder_probe_contract_suffix(_misconception_block()) == ""


def test_suffix_empty_on_garbage_token(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "definitely-not-truthy")
    assert _misconception_ladder_probe_contract_suffix(_misconception_block()) == ""


def test_suffix_empty_for_non_misconception_block_even_when_flag_on(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    assert _misconception_ladder_probe_contract_suffix(_concept_block()) == ""


def test_suffix_empty_when_no_bloom_level(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    block = _misconception_block(bloom_level=None)
    assert _misconception_ladder_probe_contract_suffix(block) == ""


def test_suffix_empty_when_bloom_level_not_canonical(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    block = _misconception_block(bloom_level="not-a-real-level")
    assert _misconception_ladder_probe_contract_suffix(block) == ""


def test_suffix_none_block_is_safe(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    assert _misconception_ladder_probe_contract_suffix(None) == ""


def test_suffix_populated_when_flag_on_and_rung_resolves(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    block = _misconception_block(bloom_level="analyze")
    suffix = _misconception_ladder_probe_contract_suffix(block)
    assert suffix
    assert 'data-cf-bloom-level="analyze"' in suffix
    probe = misconception_probe_for("analyze")
    assert probe["probe_strategy"] in suffix
    # Probe strategy for "analyze" is "reversal" per the WI-01 taxonomy.
    assert probe["probe_strategy"] == "reversal"
    assert "REVERSE" in suffix
    assert "source_backs" in suffix or "assessment_item distractor" in suffix


def test_suffix_case_insensitive_rung(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    block = _misconception_block(bloom_level="ANALYZE")
    suffix = _misconception_ladder_probe_contract_suffix(block)
    assert 'data-cf-bloom-level="analyze"' in suffix


@pytest.mark.parametrize("level", ["remember", "understand", "apply", "analyze", "evaluate", "create"])
def test_suffix_every_canonical_rung_resolves_its_own_probe_strategy(
    monkeypatch, level
):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    block = _misconception_block(bloom_level=level)
    suffix = _misconception_ladder_probe_contract_suffix(block)
    expected_strategy = misconception_probe_for(level)["probe_strategy"]
    assert expected_strategy in PROBE_STRATEGIES
    assert expected_strategy in suffix
    assert f'data-cf-bloom-level="{level}"' in suffix


def test_truthy_tokens_all_enable_the_suffix(monkeypatch):
    for token in ("1", "TRUE", "yes", "On"):
        monkeypatch.setenv(ENV_BLOOM_LADDER, token)
        block = _misconception_block(bloom_level="apply")
        assert _misconception_ladder_probe_contract_suffix(block) != "", token


# ---------------------------------------------------------------------------
# (b) Existing class-token contract stays byte-identical (additive only)
# ---------------------------------------------------------------------------


def test_existing_misconception_class_tokens_untouched(monkeypatch):
    base_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["misconception"]
    assert "misconception-card" in base_contract
    assert "misconception-claim" in base_contract
    assert "misconception-correction" in base_contract

    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    block = _misconception_block(bloom_level="apply")
    contract = _block_type_output_contract("misconception", fit_window=False)
    # The base table entry is unmodified — the suffix is appended
    # separately at the prompt-render call sites, not baked into the table.
    assert contract == base_contract


def test_output_contract_table_entry_byte_identical_regardless_of_flag(monkeypatch):
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    off_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["misconception"]
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    on_contract = _BLOCK_TYPE_OUTPUT_CONTRACTS["misconception"]
    assert off_contract == on_contract


# ---------------------------------------------------------------------------
# (c) Fit-window registration: "misconception" stays unregistered, and the
#     scaffold-overflow guard still catches an over-window scaffold.
# ---------------------------------------------------------------------------


def test_misconception_has_no_relocation_map_entry():
    # See the comment above `_RELOCATED_SEGMENTS_BY_BLOCK_TYPE` in
    # _rewrite_provider.py: the ladder-probe suffix is per-BLOCK dynamic
    # content and can never live in that static, segment-index table.
    assert "misconception" not in _RELOCATED_SEGMENTS_BY_BLOCK_TYPE


def test_fit_window_scaffold_overflow_guard_still_fires_for_misconception(
    monkeypatch,
):
    """The dynamic whole-prompt budget measures the REAL rendered scaffold
    (ladder suffix included), so an over-window misconception scaffold is
    still caught — the new guidance cannot silently re-trip the 8192-window
    overflow class."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    monkeypatch.setenv("ED4ALL_REWRITE_FIT_WINDOW", "1")
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "50")  # tiny window
    _local_env(monkeypatch)

    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(_MISCONCEPTION_CARD_HTML))
        ),
    )
    block = _misconception_block(bloom_level="apply")
    out = p.generate_rewrite(block, source_chunks=[])
    assert out.escalation_marker == "rewrite_scaffold_overflow"


# ---------------------------------------------------------------------------
# (d) End-to-end via RewriteProvider.generate_rewrite — prompt + decision
# ---------------------------------------------------------------------------


def test_e2e_flag_off_prompt_and_decisions_byte_identical(monkeypatch):
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_MISCONCEPTION_CARD_HTML))

    capture = _SpyCapture()
    p = RewriteProvider(
        provider="local", client=_make_client(handler), capture=capture
    )
    block = _misconception_block(bloom_level="analyze")
    p.generate_rewrite(block, source_chunks=[])

    wire = seen[0].read().decode("utf-8")
    assert "data-cf-bloom-level=" not in wire
    assert "BLOOM-LADDER RUNG" not in wire

    decision_types = [c["decision_type"] for c in capture.calls]
    assert decision_types == ["block_rewrite_call"]


def test_e2e_flag_on_non_misconception_block_no_ladder_decision(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    _local_env(monkeypatch)

    capture = _SpyCapture()
    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(
                200, json=_success_body("<section><p>Ratios compare.</p></section>")
            )
        ),
        capture=capture,
    )
    block = _concept_block(bloom_level="analyze")
    p.generate_rewrite(block, source_chunks=[])

    decision_types = [c["decision_type"] for c in capture.calls]
    assert "misconception_card_authoring" not in decision_types


def test_e2e_flag_on_misconception_no_bloom_level_no_ladder_decision(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    _local_env(monkeypatch)

    capture = _SpyCapture()
    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(_MISCONCEPTION_CARD_HTML))
        ),
        capture=capture,
    )
    block = _misconception_block(bloom_level=None)
    p.generate_rewrite(block, source_chunks=[])

    decision_types = [c["decision_type"] for c in capture.calls]
    assert decision_types == ["block_rewrite_call"]


def test_e2e_flag_on_misconception_block_emits_ladder_decision_and_prompt(
    monkeypatch,
):
    """Flag ON + a misconception block with a resolvable rung: the wire
    prompt carries the rung stamp + probe-strategy guidance, and a
    ``misconception_card_authoring`` decision fires with a dynamic,
    >=20-char rationale interpolating block id / rung / probe_strategy."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    _local_env(monkeypatch)

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body(_MISCONCEPTION_CARD_HTML))

    capture = _SpyCapture()
    p = RewriteProvider(
        provider="local", client=_make_client(handler), capture=capture
    )
    block = _misconception_block(bloom_level="analyze")
    out = p.generate_rewrite(block, source_chunks=[])

    # Wire prompt carries the rung stamp + probe-strategy guidance.
    wire = seen[0].read().decode("utf-8")
    assert 'data-cf-bloom-level=\\"analyze\\"' in wire or (
        'data-cf-bloom-level="analyze"' in wire
    )
    assert "reversal" in wire

    # Output HTML is untouched by the ladder machinery (prompt-only feature).
    assert "misconception-card" in out.content

    decision_types = [c["decision_type"] for c in capture.calls]
    assert "block_rewrite_call" in decision_types
    assert "misconception_card_authoring" in decision_types

    ladder_call = next(
        c for c in capture.calls
        if c["decision_type"] == "misconception_card_authoring"
    )
    rationale = ladder_call["rationale"]
    assert len(rationale) >= 20
    assert block.block_id in rationale
    assert "analyze" in rationale
    assert "reversal" in rationale
    assert block.block_id in ladder_call["decision"]


def test_e2e_flag_on_exhaustion_path_still_emits_ladder_decision(monkeypatch):
    """The ladder decision fires from every successful-authoring exit path
    of ``generate_rewrite``, including the CURIE-exhaustion force-inject
    return — a misconception block whose outline declares an un-echoed
    CURIE still gets its card authored (and the ladder decision fired)."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    _local_env(monkeypatch)

    capture = _SpyCapture()
    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(_MISCONCEPTION_CARD_HTML))
        ),
        capture=capture,
    )
    block = Block(
        block_id="week_01_content_02#misconception_carrying_1",
        block_type="misconception",
        page_id="week_01_content_02",
        sequence=0,
        content={
            "key_claims": ["Learners often cancel terms instead of factors."],
            "curies": ["sh:NodeShape"],
            "source_refs": ["semantik:slug#blk1"],
            "objective_refs": ["CO-01"],
        },
        bloom_level="apply",
    )
    p.generate_rewrite(block, source_chunks=[])

    decision_types = [c["decision_type"] for c in capture.calls]
    assert "misconception_card_authoring" in decision_types
