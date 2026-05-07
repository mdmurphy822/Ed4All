"""DEPRECATED: re-exports from :mod:`lib.validators.pair.claim_support`.

Wave W-D10 T10.1: the validator now lives in the ``pair/`` subpackage.
Re-exported here for back-compat with ``config/workflows.yaml``,
external MCP clients, and existing test imports. Will be removed in a
future minor version. Use the new path directly.
"""
import warnings

from lib.validators.pair.claim_support import *  # noqa: F401,F403
from lib.validators.pair.claim_support import (  # noqa: F401
    PairClaimSupportValidator,
    _DEFAULT_ENTAILMENT_FLOOR,
    _DEFAULT_CONTRADICTION_FLOOR,
    _DEFAULT_MAX_UNSUPPORTED_RATE,
    _DEFAULT_MAX_CONTRADICTED_RATE,
    _DEFAULT_DART_CONTRADICTION_FLOOR,
    _DART_DISAGREEMENT_RATE_WARN_CEILING,
    _MIN_SENTENCE_TOKENS,
    _PER_CLAIM_MATCH_COSINE_FLOOR,
    _REASON_UNSUPPORTED_CLAIM,
    _REASON_CONTRADICTED_CLAIM,
    _CODE_NLI_DEPS_MISSING,
    _CODE_MISSING_PER_CLAIM_SUPPORT,
    _CODE_MISSING_CLAIM_SUPPORT_RATE,
    _CODE_DART_DISAGREEMENT_RATE_HIGH,
    _CODE_MISSING_INPUTS,
    _CODE_PAIRS_FILE_READ_ERROR,
    _OUTCOME_DART_DISAGREEMENT,
    _CONTENT_TOKEN_RE,
    _SENTENCE_SPLIT_RE,
    _GATE_ISSUE_CAP,
    _ISSUE_LIST_CAP,
    _resolve_chunk_key_claims_with_attribution,
    _resolve_dart_block_texts_for_chunk,
    _try_load_embedder_safe,
    _cosine_safe,
    _match_sentence_to_claim,
    _decompose_sentences,
    _content_token_count,
    _emit_decision,
)

warnings.warn(
    "lib.validators.pair_claim_support is deprecated; "
    "use lib.validators.pair.claim_support",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PairClaimSupportValidator",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_DEFAULT_CONTRADICTION_FLOOR",
    "_DEFAULT_MAX_UNSUPPORTED_RATE",
    "_DEFAULT_MAX_CONTRADICTED_RATE",
    "_DEFAULT_DART_CONTRADICTION_FLOOR",
    "_DART_DISAGREEMENT_RATE_WARN_CEILING",
    "_MIN_SENTENCE_TOKENS",
    "_PER_CLAIM_MATCH_COSINE_FLOOR",
    "_REASON_UNSUPPORTED_CLAIM",
    "_REASON_CONTRADICTED_CLAIM",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_MISSING_PER_CLAIM_SUPPORT",
    "_CODE_MISSING_CLAIM_SUPPORT_RATE",
    "_CODE_DART_DISAGREEMENT_RATE_HIGH",
    "_CODE_MISSING_INPUTS",
    "_CODE_PAIRS_FILE_READ_ERROR",
    "_OUTCOME_DART_DISAGREEMENT",
    "_CONTENT_TOKEN_RE",
    "_SENTENCE_SPLIT_RE",
    "_GATE_ISSUE_CAP",
    "_ISSUE_LIST_CAP",
    "_resolve_chunk_key_claims_with_attribution",
    "_resolve_dart_block_texts_for_chunk",
    "_try_load_embedder_safe",
    "_cosine_safe",
    "_match_sentence_to_claim",
    "_decompose_sentences",
    "_content_token_count",
    "_emit_decision",
]
