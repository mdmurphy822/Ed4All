"""Smoke tests for Trainforge parser modules."""

import pytest

from Trainforge.parsers.html_content_parser import (
    ContentSection,
    HTMLContentParser,
    LearningObjective,
    ParsedHTMLModule,
)
from Trainforge.parsers.imscc_parser import ContentItem, IMSCCPackage, IMSCCParser
from Trainforge.parsers.qti_parser import QTIChoice, QTIParser, QTIQuestion


@pytest.mark.unit
class TestHTMLContentParser:
    """Test HTML content parser imports and construction."""

    def test_content_section_dataclass(self):
        section = ContentSection(heading="Intro", level=2, content="Welcome", word_count=1)
        assert section.heading == "Intro"
        assert section.level == 2

    def test_learning_objective_dataclass(self):
        obj = LearningObjective(id="LO-1", text="Understand testing", bloom_level="understand")
        assert obj.bloom_level == "understand"

    def test_parsed_html_module_dataclass(self):
        module = ParsedHTMLModule(title="Module 1", word_count=100)
        assert module.title == "Module 1"
        assert module.sections == []

    def test_parser_construction(self):
        parser = HTMLContentParser()
        assert parser is not None


@pytest.mark.unit
class TestIMSCCParser:
    """Test IMSCC parser imports and construction."""

    def test_content_item_dataclass(self):
        item = ContentItem(id="item1", title="Test Item", type="html", path="test.html")
        assert item.id == "item1"

    def test_imscc_package_dataclass(self):
        pkg = IMSCCPackage(
            source_path="/tmp/test.imscc",
            source_lms="generic",
            version="1.1",
            title="Test Course",
        )
        assert pkg.title == "Test Course"
        assert pkg.items == []

    def test_parser_construction(self):
        parser = IMSCCParser()
        assert parser is not None


@pytest.mark.unit
class TestQTIParser:
    """Test QTI parser imports and construction."""

    def test_qti_choice_dataclass(self):
        choice = QTIChoice(id="A", text="Option A", is_correct=True)
        assert choice.is_correct is True

    def test_qti_question_dataclass(self):
        q = QTIQuestion(id="q1", type="multiple_choice", stem="What is 1+1?")
        assert q.type == "multiple_choice"
        assert q.choices == []

    def test_parser_construction(self):
        parser = QTIParser()
        assert parser is not None
