"""Unit tests — council backbone batching (CPU, mocked forward, no model load).

Asserts that ``batched_cls_pooled`` chunks the backbone forward into
``SEMANTIK_COUNCIL_BATCH_SIZE`` slices and concatenates the per-slice CLS
vectors into a tensor byte-identical to a single un-chunked forward — the
"same logits -> same labels" equivalence the batch-size loop must preserve.

The real council backbone is too heavy for CI, so the forward is mocked with a
deterministic per-row encoder: each input text maps to a fixed hidden vector,
the mock ``peft_model`` returns ``last_hidden_state`` whose CLS row equals that
vector. Because the CLS row is independent of batch padding, the concatenated
batched output must equal the single-forward output exactly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dart_semantic.council.structure import batched_cls_pooled  # noqa: E402

_HIDDEN = 8


def _vec_for(text: str) -> list[float]:
    """A deterministic hidden vector for one text (stable, content-derived)."""
    h = abs(hash(text)) % 1000
    return [float((h + k) % 7) - 3.0 for k in range(_HIDDEN)]


class _Enc(dict):
    """A tok() return value — carries input_ids/attention_mask + .to()."""

    def __init__(self, texts):
        super().__init__()
        self.texts = list(texts)
        # Pad to the batch max so per-batch padding differs from doc-max.
        maxlen = max((len(t.split()) for t in self.texts), default=1)
        ids = []
        mask = []
        for t in self.texts:
            toks = t.split()
            row = [hash(w) % 100 + 1 for w in toks] + [0] * (maxlen - len(toks))
            m = [1] * len(toks) + [0] * (maxlen - len(toks))
            ids.append(row)
            mask.append(m)
        self["input_ids"] = torch.tensor(ids, dtype=torch.long)
        self["attention_mask"] = torch.tensor(mask, dtype=torch.long)

    def to(self, device):  # noqa: D401 - mock .to()
        return self


class _Tok:
    """A mock tokenizer: tok(texts, ...) -> _Enc. Stashes the order of texts."""

    def __init__(self):
        self.text_order: list[list[str]] = []

    def __call__(self, texts, **kwargs):
        self.text_order.append(list(texts))
        return _Enc(texts)


class _Out:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _Model:
    """A mock peft_model: maps each row's first input id back to its text vector.

    We can't recover the text from ids alone, so instead the model carries the
    text->vec map keyed by call order: each forward gets a contiguous slice of
    the master text list, in order, so it reproduces the right CLS rows.
    """

    def __init__(self, texts):
        self._texts = list(texts)
        self._cursor = 0
        self.forward_calls = 0

    def __call__(self, *, input_ids, attention_mask):
        self.forward_calls += 1
        n = input_ids.shape[0]
        slice_texts = self._texts[self._cursor : self._cursor + n]
        self._cursor += n
        seqlen = input_ids.shape[1]
        # Build last_hidden_state [n, seqlen, hidden] with the CLS (position 0)
        # row equal to the text's vector; other positions are junk (masked out).
        cls = torch.tensor([_vec_for(t) for t in slice_texts], dtype=torch.float32)
        lhs = torch.zeros((n, seqlen, _HIDDEN), dtype=torch.float32)
        lhs[:, 0, :] = cls
        # Pollute non-CLS positions so a wrong-position pool would diverge.
        if seqlen > 1:
            lhs[:, 1:, :] = 999.0
        return _Out(lhs)


def _run(texts, batch_size):
    tok = _Tok()
    model = _Model(texts)
    pooled = batched_cls_pooled(model, tok, texts, "cpu", batch_size=batch_size)
    return pooled, model, tok


def test_batched_equals_unbatched():
    texts = [f"span number {i} text" for i in range(37)]
    full, full_model, _ = _run(texts, batch_size=len(texts))
    batched, batch_model, _ = _run(texts, batch_size=8)
    assert full.shape == batched.shape == (37, _HIDDEN)
    assert torch.equal(full, batched)
    # Equivalence holds: 1 forward un-chunked vs ceil(37/8)=5 chunked forwards.
    assert full_model.forward_calls == 1
    assert batch_model.forward_calls == 5


@pytest.mark.parametrize("bs", [1, 2, 8, 16, 1000])
def test_various_batch_sizes_equal(bs):
    texts = [f"row {i}" for i in range(20)]
    ref, _, _ = _run(texts, batch_size=10000)
    out, _, _ = _run(texts, batch_size=bs)
    assert torch.equal(ref, out)


def test_input_order_preserved():
    texts = ["alpha", "beta one two", "gamma", "delta four five six", "epsilon"]
    out, _, _ = _run(texts, batch_size=2)
    expected = torch.tensor([_vec_for(t) for t in texts], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_empty_input_returns_empty():
    out, model, _ = _run([], batch_size=4)
    assert out.shape[0] == 0
    assert model.forward_calls == 0


def test_batch_size_from_env_used(monkeypatch):
    monkeypatch.setenv("SEMANTIK_COUNCIL_BATCH_SIZE", "5")
    texts = [f"r{i}" for i in range(12)]
    tok = _Tok()
    model = _Model(texts)
    # batch_size=None -> resolve from env (5) -> ceil(12/5)=3 forwards.
    batched_cls_pooled(model, tok, texts, "cpu")
    assert model.forward_calls == 3
