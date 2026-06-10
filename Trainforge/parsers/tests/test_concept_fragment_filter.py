"""FIX #3b: bold-span sentence-fragment rejection in concept harvest.

``HTMLContentParser._extract_concepts`` harvests bold/strong spans as
candidate concepts. Some source notebooks (e.g. NVIDIA's) bold whole
SENTENCES rather than key terms, leaking sentence-fragment "concepts"
("Your LLM Can Have", "Congratulations We Now Have") into the KG.

The fix gates a domain-agnostic fragment filter
(``lib.ontology.lexical_concept_seeds.is_fragment_phrase``) behind the
default-OFF ``TRAINFORGE_FILTER_FRAGMENT_CONCEPTS`` flag. These tests pin:

* flag OFF → byte-identical legacy harvest (every bold span kept);
* flag ON → sentence fragments dropped, real noun-phrase concepts kept;
* keep-set / reject-set regression on the helper directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402
from lib.ontology.lexical_concept_seeds import is_fragment_phrase  # noqa: E402


_HTML = (
    "<p>"
    "<strong>Your LLM can have a modifiable knowledge base</strong> "
    "and a <strong>Knowledge Base</strong> is central."
    "</p>"
)


def test_flag_off_keeps_both_legacy_parity(monkeypatch):
    """Flag OFF: both the sentence fragment and the real concept harvest."""
    monkeypatch.delenv("TRAINFORGE_FILTER_FRAGMENT_CONCEPTS", raising=False)
    parser = HTMLContentParser()
    concepts = parser._extract_concepts(_HTML)
    assert "Your LLM can have a modifiable knowledge base" in concepts
    assert "Knowledge Base" in concepts


def test_flag_on_rejects_fragment_keeps_concept(monkeypatch):
    """Flag ON: sentence fragment dropped, real concept kept."""
    monkeypatch.setenv("TRAINFORGE_FILTER_FRAGMENT_CONCEPTS", "1")
    parser = HTMLContentParser()
    concepts = parser._extract_concepts(_HTML)
    assert "Your LLM can have a modifiable knowledge base" not in concepts
    assert "Knowledge Base" in concepts


_KEEP = [
    "Semantic Guardrailing",
    "Vector Store",
    "Foundation Models",
    "FAISS Vector Store",
    "RunnableAssign",
    "Knowledge Base",
]

_REJECT = [
    "Congratulations We Now Have",
    "Passing Dictionaries Helps Us",
    "This Notebook Will Serve",
    "Try To Find Models",
    "Your LLM Can Have",
    "Specifically We Can Use",
    "Luckily For Us Llms",
    "One Tool Is Your",
]


@pytest.mark.parametrize("phrase", _KEEP)
def test_keep_set_passes_filter(phrase):
    assert is_fragment_phrase(phrase) is False, phrase


@pytest.mark.parametrize("phrase", _REJECT)
def test_reject_set_flagged_as_fragment(phrase):
    assert is_fragment_phrase(phrase) is True, phrase


# ---------------------------------------------------------------------------
# Leading-imperative-verb rejection (KG audit). Imperative verb-phrase
# "concepts" are instructions, not noun-concepts — they leaked through the
# pre-existing rules because none start with a function word / sentence-opener,
# carry < 2 function words, and stay under the token ceiling. The new
# leading-imperative-verb rule rejects them. The noun-concept keep-set proves
# real concepts (which never open with an action verb) still pass — including
# the gerund-NOUN edge case "running state chain" (base "run" is imperative,
# but "running" here is a noun modifier).
# ---------------------------------------------------------------------------

_IMPERATIVE_REJECT = [
    "apply to self host",
    "receive user input",
    "delegate tasks",
    "download the dataset",
    "embed the data",
    "define clear tools",
    "create a robust state",
    "maintain contextual awareness",
    "synchronize saved objects",
    "aggregating data",
    "run the cell below",
]

_NOUN_CONCEPT_KEEP = [
    "vector store",
    "knowledge base",
    "semantic guardrailing",
    "running state chain",
    "prompt engineering",
    "document summary base",
    "faiss vector store",
    "retrieval augmented generation",
]


@pytest.mark.parametrize("phrase", _IMPERATIVE_REJECT)
def test_imperative_verb_phrase_flagged_as_fragment(phrase):
    assert is_fragment_phrase(phrase) is True, phrase


@pytest.mark.parametrize("phrase", _NOUN_CONCEPT_KEEP)
def test_noun_concept_keep_set_passes_filter(phrase):
    assert is_fragment_phrase(phrase) is False, phrase


def test_single_action_word_not_rejected_by_imperative_rule():
    """A bare single action word is scaffolding handled elsewhere — the
    multi-word-only imperative rule must NOT fire on it here.
    """
    assert is_fragment_phrase("apply") is False
    assert is_fragment_phrase("run") is False


def test_keep_set_survives_full_parser_harvest(monkeypatch):
    """The keep-set concepts survive the flag-ON parser harvest end-to-end."""
    monkeypatch.setenv("TRAINFORGE_FILTER_FRAGMENT_CONCEPTS", "1")
    html = "".join(f"<strong>{c}</strong>" for c in _KEEP)
    parser = HTMLContentParser()
    concepts = parser._extract_concepts(html)
    for c in _KEEP:
        assert c in concepts, c


def test_reject_set_dropped_by_full_parser_harvest(monkeypatch):
    """The reject-set fragments are dropped by the flag-ON parser harvest."""
    monkeypatch.setenv("TRAINFORGE_FILTER_FRAGMENT_CONCEPTS", "1")
    html = "".join(f"<strong>{c}</strong>" for c in _REJECT)
    parser = HTMLContentParser()
    concepts = parser._extract_concepts(html)
    for c in _REJECT:
        assert c not in concepts, c
