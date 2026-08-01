"""Encoder precision seam + the CUDA-default device contract (W9 / W10).

The encoder, not the persisted matrix, is where precision belongs: the index
build streams the full fp32 weight set per forward on a bandwidth-bound
encoder, so halving the weight and activation traffic is the dominant
index-build lever. The persisted ``embeddings.npy`` stays float32 either way —
``encode_batch`` casts on the way out — so nothing downstream changes shape.

Everything here runs with a stub ``sentence_transformers`` module: zero
weights, zero network, zero GPU. Run CPU-pinned:

    ED4ALL_EMBEDDING_DEVICE=cpu ED4ALL_NLI_DEVICE=cpu \
      pytest lib/embedding/tests/test_embedding_precision.py
"""
from __future__ import annotations

import sys
import types

import pytest

from lib.embedding.providers import (
    ENV_DEVICE,
    ENV_DTYPE,
    VALID_DTYPES,
    EmbeddingBackendUnavailable,
    EmbeddingClient,
    _EMBEDDING_PROVIDERS,
    _reset_embedding_client_cache_for_tests,
    resolve_embedding_provider,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    _reset_embedding_client_cache_for_tests()
    yield
    _reset_embedding_client_cache_for_tests()


def _install_capture_st(monkeypatch):
    """Stub SentenceTransformer that records kwargs then stops the load."""
    calls = []
    fake_mod = types.ModuleType("sentence_transformers")

    class _CaptureST:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            raise RuntimeError("stop after the kwargs snapshot")

    fake_mod.SentenceTransformer = _CaptureST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    return calls


# ---------------------------------------------------------------------------
# Registry seam — a registry FIELD, never a subclass.
# ---------------------------------------------------------------------------
def test_dtype_is_a_registry_field_on_the_st_entry():
    entry = _EMBEDDING_PROVIDERS["st"]
    assert entry["dtype_env"] == ENV_DTYPE
    assert entry["dtype_default"] == "fp32"
    # Threaded the same way trust_remote_code is: no per-precision subclass.
    assert entry["device_env"] == ENV_DEVICE
    assert entry["device_default"] == "cuda"


def test_valid_dtypes_are_exactly_the_documented_set():
    assert VALID_DTYPES == ("fp32", "bf16", "fp16")


# ---------------------------------------------------------------------------
# Default (fp32) load is byte-identical to the pre-seam load.
# ---------------------------------------------------------------------------
def test_fp32_default_adds_no_kwargs(monkeypatch):
    calls = _install_capture_st(monkeypatch)
    monkeypatch.setenv(ENV_DEVICE, "cpu")
    monkeypatch.delenv(ENV_DTYPE, raising=False)

    client = EmbeddingClient(resolve_embedding_provider(provider_name="st"))
    with pytest.raises(EmbeddingBackendUnavailable):
        client.encode_batch(["x"])

    kwargs = calls[-1]["kwargs"]
    assert kwargs == {"device": "cpu"}
    assert "model_kwargs" not in kwargs


def test_fp32_default_does_not_touch_tf32(monkeypatch):
    """TF32 is a process-global precision change; the default path must not
    make it on anyone's behalf."""
    torch = pytest.importorskip("torch")
    _install_capture_st(monkeypatch)
    monkeypatch.setenv(ENV_DEVICE, "cpu")
    monkeypatch.delenv(ENV_DTYPE, raising=False)

    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        client = EmbeddingClient(resolve_embedding_provider(provider_name="st"))
        with pytest.raises(EmbeddingBackendUnavailable):
            client.encode_batch(["x"])
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------------------
# Half precision: threaded to torch, TF32 on, CPU refused.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("token, attr", [("bf16", "bfloat16"), ("fp16", "float16")])
def test_half_precision_threaded_into_model_kwargs(monkeypatch, token, attr):
    torch = pytest.importorskip("torch")
    calls = _install_capture_st(monkeypatch)
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    monkeypatch.setenv(ENV_DTYPE, token)

    prior = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        client = EmbeddingClient(resolve_embedding_provider(provider_name="st"))
        with pytest.raises(EmbeddingBackendUnavailable):
            client.encode_batch(["x"])
        kwargs = calls[-1]["kwargs"]
        assert kwargs["device"] == "cuda"
        assert kwargs["model_kwargs"] == {"torch_dtype": getattr(torch, attr)}
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior[0]
        torch.backends.cudnn.allow_tf32 = prior[1]


def test_half_precision_on_cpu_is_refused_at_resolution(monkeypatch):
    """Resolution fails BEFORE any load: an ignored precision would be
    recorded as fp32 provenance the operator never asked for."""
    monkeypatch.setenv(ENV_DEVICE, "cpu")
    monkeypatch.setenv(ENV_DTYPE, "bf16")
    with pytest.raises(EmbeddingBackendUnavailable) as exc:
        resolve_embedding_provider(provider_name="st")
    message = str(exc.value)
    assert "ED4ALL_EMBEDDING_DTYPE=fp32" in message
    assert "cuda" in message


def test_half_precision_allowed_on_indexed_cuda_device(monkeypatch):
    monkeypatch.setenv(ENV_DEVICE, "cuda:1")
    monkeypatch.setenv(ENV_DTYPE, "bf16")
    resolved = resolve_embedding_provider(provider_name="st")
    assert resolved.device == "cuda:1"
    assert resolved.dtype == "bf16"


# ---------------------------------------------------------------------------
# The load-failure message is operator-actionable and never downgrades.
# ---------------------------------------------------------------------------
def test_load_failure_message_names_device_dtype_and_optout(monkeypatch):
    _install_capture_st(monkeypatch)
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    monkeypatch.setenv(ENV_DTYPE, "fp32")

    client = EmbeddingClient(
        resolve_embedding_provider(provider_name="st"), offline=True
    )
    with pytest.raises(EmbeddingBackendUnavailable) as exc:
        client.encode_batch(["x"])
    message = str(exc.value)
    assert "device=cuda" in message
    assert "dtype=fp32" in message
    assert "ED4ALL_EMBEDDING_DEVICE=cpu" in message
    assert "No automatic CUDA→CPU downgrade" in message


def test_providers_module_has_no_cuda_availability_probe():
    """Guard the anti-silent-degradation contract structurally.

    A ``torch.cuda.is_available()`` branch in this module is exactly how a
    "for safety" CPU downgrade gets reintroduced, and no behavioural test can
    catch its absence. Checked over the parsed AST rather than the raw text so
    the prose that *documents* the ban does not trip its own guard.
    """
    import ast
    from pathlib import Path

    import lib.embedding.providers as providers_mod

    tree = ast.parse(Path(providers_mod.__file__).read_text(encoding="utf-8"))
    probes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("is_available", "device_count")
    ]
    assert probes == [], [ast.unparse(p) for p in probes]
