"""Stage-6 per-unit resume sidecar (task #40).

CPU-pinned, no GPU / GGUF / endpoint. Pins:

  * the checkpoint resolver (default ON, site-falsey, family-falsey, site-beats-
    family) — mirrors reasoning_qc.resolve_reasoning_qc_checkpoint;
  * the content-address fingerprint moves on prompt / model / sampling /
    slot_tag / draft;
  * put/get round-trip + corrupt→miss + empty-never-cached;
  * the runner Phase-1 resume: run1 generates N + populates sidecars, run2
    generates 0 with identical candidates; flag-off ⇒ no cache dir, byte-stable.

Run:
  python3 -m pytest semantik_structure/qwen_specialists/tests/test_stage6_checkpoint.py -q
"""
from __future__ import annotations

import types

from semantik_structure.qwen_specialists import stage6_checkpoint as sc
from semantik_structure.qwen_specialists.runner import run_qwen_specialists
from semantik_structure.qwen_specialists.runtime import MockRuntime


def _region(kind: str, text: str = "x"):
    return types.SimpleNamespace(
        kind=kind, payload={"text": text}, aria_hints=(), feature_block_indices=()
    )


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #
def test_checkpoint_resolver_default_on(monkeypatch):
    monkeypatch.delenv("SEMANTIK_STAGE6_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    assert sc.resolve_stage6_checkpoint() is True


def test_checkpoint_resolver_site_falsey(monkeypatch):
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    for falsey in ("0", "false", "no", "off"):
        monkeypatch.setenv("SEMANTIK_STAGE6_CHECKPOINT", falsey)
        assert sc.resolve_stage6_checkpoint() is False, falsey


def test_checkpoint_resolver_family_falsey(monkeypatch):
    monkeypatch.delenv("SEMANTIK_STAGE6_CHECKPOINT", raising=False)
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "off")
    assert sc.resolve_stage6_checkpoint() is False
    # Family unset / garbage → on.
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "garbage")
    assert sc.resolve_stage6_checkpoint() is True


def test_checkpoint_resolver_site_beats_family(monkeypatch):
    # Site ON wins over family OFF.
    monkeypatch.setenv("SEMANTIK_STAGE6_CHECKPOINT", "1")
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    assert sc.resolve_stage6_checkpoint() is True
    # Site OFF wins over family ON.
    monkeypatch.setenv("SEMANTIK_STAGE6_CHECKPOINT", "0")
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "1")
    assert sc.resolve_stage6_checkpoint() is False


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def _fp(**over):
    base = dict(
        prompt="SYSTEM: r\nUSER: region",
        model="m1",
        temperature=0.6,
        top_p=0.95,
        max_tokens=512,
        seed=7,
        repeat_penalty=1.0,
        slot_tag="phase1:prose:slot0",
        draft=None,
    )
    base.update(over)
    prompt = base.pop("prompt")
    return sc.candidate_fingerprint(prompt, **base)


def test_fingerprint_moves_on_every_component():
    base = _fp()
    assert _fp(prompt="SYSTEM: r\nUSER: OTHER") != base
    assert _fp(model="m2") != base
    assert _fp(temperature=0.7) != base
    assert _fp(top_p=0.9) != base
    assert _fp(max_tokens=1024) != base
    assert _fp(seed=8) != base
    assert _fp(repeat_penalty=1.05) != base
    assert _fp(slot_tag="phase1:prose:slot1") != base
    assert _fp(draft="<p>d</p>") != base
    # Stable: identical inputs → identical key.
    assert _fp() == base


# --------------------------------------------------------------------------- #
# put / get
# --------------------------------------------------------------------------- #
def test_put_get_round_trip(tmp_path):
    root = tmp_path / "stage6"
    p = sc.cache_path("abcd1234", root)
    assert sc.get(p) is None  # cold
    sc.put(p, "<p>hello</p>")
    assert sc.get(p) == "<p>hello</p>"


def test_corrupt_entry_is_miss(tmp_path):
    root = tmp_path / "stage6"
    p = sc.cache_path("beef", root)
    sc.put(p, "<p>x</p>")
    p.write_text("}{ not json")
    assert sc.get(p) is None  # corrupt → miss (never raises)


def test_empty_and_none_never_cached(tmp_path):
    root = tmp_path / "stage6"
    assert sc.cacheable("") is False
    assert sc.cacheable(None) is False
    assert sc.cacheable("x") is True
    p = sc.cache_path("empty", root)
    sc.put(p, "")  # guarded no-op
    assert sc.get(p) is None
    assert not p.exists()


# --------------------------------------------------------------------------- #
# Runner Phase-1 resume (real-mode fake runtime)
# --------------------------------------------------------------------------- #
class _RealSpy(MockRuntime):
    """MockRuntime that records generate_batch invocations (prompt lists)."""

    def __init__(self) -> None:
        super().__init__()
        self.gen_prompts: list[list[str]] = []

    def generate_batch(self, prompts, *, max_tokens, **kw):  # type: ignore[override]
        self.gen_prompts.append(list(prompts))
        return super().generate_batch(prompts, max_tokens=max_tokens, **kw)


def _clear(monkeypatch):
    for k in (
        "SEMANTIK_SPECIALIST_PROVIDER",
        "SEMANTIK_SPECIALIST_REFINE",
        "SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE",
        "ED4ALL_GENERATION_CHECKPOINT",
    ):
        monkeypatch.delenv(k, raising=False)


def test_phase1_resume_second_run_serves_from_cache(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SEMANTIK_STAGE6_CHECKPOINT", "1")

    regions = [_region("paragraph", "a"), _region("paragraph", "b")]

    # Run 1 — cache cold: generate_batch fires (2 prompts), sidecars populate.
    rt1 = _RealSpy()
    out1 = run_qwen_specialists(regions, [], k=1, runtime=rt1, runtime_mode="real")
    assert rt1.gen_prompts, "run1 must generate (cold cache)"
    n_generated_1 = sum(len(p) for p in rt1.gen_prompts)
    assert n_generated_1 == 2  # both prose prompts generated
    cache_root = tmp_path / "stage6_candidate_cache"
    assert cache_root.exists()
    assert len(list(cache_root.rglob("*.json"))) == 2  # one sidecar per region

    # Run 2 — same input, FRESH spy: every region HITS → ZERO generation.
    rt2 = _RealSpy()
    out2 = run_qwen_specialists(regions, [], k=1, runtime=rt2, runtime_mode="real")
    assert rt2.gen_prompts == []  # all served from cache
    # Identical candidate text served from the sidecar.
    assert out2[0][0].text == out1[0][0].text
    assert out2[1][0].text == out1[1][0].text


def test_phase1_flag_off_no_cache_dir_byte_stable(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SEMANTIK_STAGE6_CHECKPOINT", "0")  # off

    regions = [_region("paragraph", "a"), _region("paragraph", "b")]
    rt1 = _RealSpy()
    out1 = run_qwen_specialists(regions, [], k=1, runtime=rt1, runtime_mode="real")
    rt2 = _RealSpy()
    out2 = run_qwen_specialists(regions, [], k=1, runtime=rt2, runtime_mode="real")

    # Flag off → both runs generate (no cache short-circuit), no cache dir made.
    assert rt1.gen_prompts and rt2.gen_prompts
    assert not (tmp_path / "stage6_candidate_cache").exists()
    assert out1[0][0].text == out2[0][0].text  # deterministic mock


def test_mock_mode_never_caches(monkeypatch, tmp_path):
    """runtime_mode='mock' keeps use_cache False even with the flag ON —
    every existing mock-mode test stays byte-identical (no cache dir)."""
    _clear(monkeypatch)
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SEMANTIK_STAGE6_CHECKPOINT", "1")

    regions = [_region("paragraph", "a")]
    rt = _RealSpy()
    run_qwen_specialists(regions, [], k=1, runtime=rt, runtime_mode="mock")
    assert not (tmp_path / "stage6_candidate_cache").exists()
