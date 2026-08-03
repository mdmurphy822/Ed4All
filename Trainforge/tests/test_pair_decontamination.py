"""Regression net for the canonical pair-decontamination postprocessor.

Layered gold-set decontamination:
  * exact-match drop;
  * sliding 8-gram overlap drop;
  * embedding top-k drop through a deterministic embedder seam;
  * paraphrase hook drop (injected callable);
  * survivors stamped decontam_checked=True; quarantine carries reasons;
  * capture event fires; empty gold set => no drops (stamps only).

Offline / deterministic — no network, no model, no course slugs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.postprocessing.pair_decontamination import (  # noqa: E402
    REASON_EMBED,
    REASON_EXACT,
    REASON_NGRAM,
    REASON_PARAPHRASE,
    decontaminate_pairs,
    gold_question_texts,
)


class _RecordingCapture:
    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        kwargs = {**kwargs, "event_id": f"evt_{len(self.decisions):04d}"}
        self.decisions.append(kwargs)


_GOLD = [
    "What is the quadratic formula used to solve equations?",
    "How do you factor a difference of two squares completely?",
]


def _pair(prompt: str, completion: str = "A grounded, sufficiently long completion "
          "that comfortably clears the fifty character schema floor.") -> Dict[str, Any]:
    return {"prompt": prompt, "completion": completion}


def test_gold_question_texts_shapes():
    assert gold_question_texts({"questions": [{"question_text": "Q1?"}, {"question": "Q2?"}]}) == ["Q1?", "Q2?"]
    assert gold_question_texts(["a", "b", ""]) == ["a", "b"]
    assert gold_question_texts(None) == []


def test_exact_match_dropped():
    pairs = [_pair("What is the quadratic formula used to solve equations?")]
    survivors, quarantined = decontaminate_pairs(pairs, _GOLD)
    assert survivors == []
    assert len(quarantined) == 1
    assert quarantined[0]["_decontam_reason"] == REASON_EXACT


def test_ngram_overlap_dropped():
    # 8+ consecutive gold tokens embedded in a longer pair prompt.
    leaked = "In this lesson: what is the quadratic formula used to solve equations quickly today?"
    survivors, quarantined = decontaminate_pairs([_pair(leaked)], _GOLD)
    assert survivors == []
    assert quarantined[0]["_decontam_reason"] == REASON_NGRAM


def test_clean_pair_survives_and_is_stamped():
    clean = _pair("Explain why isolating a variable preserves the equality of an equation.")
    survivors, quarantined = decontaminate_pairs([clean], _GOLD)
    assert quarantined == []
    assert len(survivors) == 1
    assert survivors[0]["decontam_checked"] is True


def test_empty_gold_no_drops_but_stamps():
    pairs = [_pair("anything at all here"), _pair("another distinct prompt")]
    survivors, quarantined = decontaminate_pairs(pairs, [])
    assert quarantined == []
    assert all(p["decontam_checked"] is True for p in survivors)


class _FakeEmbedder:
    """Map topic markers to vectors for deterministic similarity tests."""

    def encode(self, text: str) -> List[float]:
        t = text.lower()
        if "quadratic" in t:
            return [1.0, 0.0, 0.0]
        if "factor" in t and "squares" in t:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_embedding_layer_catches_paraphrase():
    # The vector seam identifies semantic overlap without an 8-gram match.
    para = _pair("Which quadratic expression solves a second-degree polynomial?")
    survivors, quarantined = decontaminate_pairs(
        [para], _GOLD, embedder=_FakeEmbedder(), embed_floor=0.9,
    )
    assert survivors == []
    assert quarantined[0]["_decontam_reason"] == REASON_EMBED


def test_embedding_layer_off_by_default_offline():
    # Without an embedder, the offline pass applies only lexical layers 1-2.
    para = _pair("Which quadratic expression solves a second-degree polynomial?")
    survivors, _ = decontaminate_pairs([para], _GOLD)
    assert len(survivors) == 1


def test_paraphrase_hook_layer():
    def _hook(text: str, gold: Any) -> Any:
        return "matched-gold-0" if "polynomial" in text.lower() else None

    para = _pair("Which expression solves a second-degree polynomial cleanly?")
    survivors, quarantined = decontaminate_pairs(
        [para], _GOLD, paraphrase_check=_hook,
    )
    assert survivors == []
    assert quarantined[0]["_decontam_reason"] == REASON_PARAPHRASE


def test_preference_pair_text_projection():
    # Preference pairs use chosen/rejected, not completion.
    pref = {"prompt": "p", "chosen": "what is the quadratic formula used to solve equations?",
            "rejected": "no"}
    survivors, quarantined = decontaminate_pairs([pref], _GOLD)
    assert survivors == []
    assert quarantined[0]["_decontam_reason"] in (REASON_EXACT, REASON_NGRAM)


def test_capture_event_fires():
    cap = _RecordingCapture()
    decontaminate_pairs([_pair("clean unrelated prompt about arithmetic sums")], _GOLD, capture=cap)
    assert len(cap.decisions) == 1
    assert cap.decisions[0]["decision_type"] == "synthesis_leakage_check"
    assert len(cap.decisions[0]["rationale"]) >= 20


def test_deterministic():
    pairs1 = [_pair("clean one about slopes"), _pair("clean two about intercepts")]
    pairs2 = [_pair("clean one about slopes"), _pair("clean two about intercepts")]
    s1, q1 = decontaminate_pairs(pairs1, _GOLD)
    s2, q2 = decontaminate_pairs(pairs2, _GOLD)
    assert [p["prompt"] for p in s1] == [p["prompt"] for p in s2]
    assert len(q1) == len(q2)
