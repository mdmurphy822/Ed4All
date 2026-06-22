"""Version-tolerant council tokenizer loader — regression tests.

The council BERT tokenizers under ``models/council/*/.../tokenizer/`` were
SAVED by transformers 5.x: their ``tokenizer_config.json`` declares
``"tokenizer_class": "TokenizersBackend"``, which transformers 4.x (Ed4All's
installed 4.57.x) cannot resolve — a bare ``AutoTokenizer.from_pretrained``
raises ``ValueError: Tokenizer class TokenizersBackend does not exist ...``.
``load_local_tokenizer`` recovers via the colocated version-independent
``tokenizer.json``. These tests prove that recovery on whatever transformers
is installed (the 4.57.x path), and exercise the pure helpers with no model
weights so a CI box without weights still passes.

CPU-pinned + dependency-graceful: ``transformers`` and the model dir are both
optional (``importorskip`` / ``skip`` when absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dart_semantic.council.base import (
    _coerce_special_token,
    load_local_tokenizer,
)

# .../SemantiK/dart_semantic/council/tests/  -> parents[3] == SemantiK root.
_SEMANTIK_ROOT = Path(__file__).resolve().parents[3]
# Any council head's saved tokenizer dir exercises the same loader; structure
# is the canonical one named in the migration diagnosis.
_STRUCTURE_TOK_DIR = (
    _SEMANTIK_ROOT / "models" / "council" / "structure" / "final" / "tokenizer"
)


# --- pure-helper tests (no transformers, no weights) ----------------------


def test_coerce_special_token_bare_string():
    assert _coerce_special_token("[CLS]") == "[CLS]"


def test_coerce_special_token_added_token_dict():
    # transformers serializes some special tokens as AddedToken dicts.
    added = {"content": "[MASK]", "lstrip": False, "rstrip": False}
    assert _coerce_special_token(added) == "[MASK]"


def test_coerce_special_token_none_and_garbage():
    assert _coerce_special_token(None) is None
    assert _coerce_special_token(123) is None
    assert _coerce_special_token({"no_content_key": "x"}) is None


# --- the real 4.57.x recovery proof --------------------------------------


@pytest.mark.skipif(
    not _STRUCTURE_TOK_DIR.exists(),
    reason="council structure tokenizer weights not present on this box",
)
def test_load_local_tokenizer_recovers_on_installed_transformers():
    """Proves the saved 5.x tokenizer loads on the installed transformers.

    On transformers 4.57.x a bare ``AutoTokenizer.from_pretrained`` on this
    dir raises the ``TokenizersBackend`` skew; ``load_local_tokenizer`` must
    NOT raise and must return a working tokenizer.
    """
    pytest.importorskip("transformers")

    tok = load_local_tokenizer(_STRUCTURE_TOK_DIR)

    # The structure council tokenizer has a 50280-token vocab.
    assert tok.vocab_size == 50280

    # And it must actually encode text.
    ids = tok("Chapter 1")["input_ids"]
    assert isinstance(ids, list)
    assert len(ids) >= 1
    assert all(isinstance(i, int) for i in ids)


@pytest.mark.skipif(
    not _STRUCTURE_TOK_DIR.exists(),
    reason="council structure tokenizer weights not present on this box",
)
def test_load_local_tokenizer_carries_special_tokens():
    """Special tokens recorded in tokenizer_config.json survive the fallback."""
    pytest.importorskip("transformers")

    config = json.loads(
        (_STRUCTURE_TOK_DIR / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    tok = load_local_tokenizer(_STRUCTURE_TOK_DIR)

    # When the saved config declares a pad token, the loaded tokenizer must
    # expose the same surface string (str or AddedToken-dict shapes both ok).
    expected_pad = _coerce_special_token(config.get("pad_token"))
    if expected_pad is not None:
        assert tok.pad_token == expected_pad
