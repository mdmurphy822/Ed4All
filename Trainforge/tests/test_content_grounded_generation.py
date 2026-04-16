"""
Tests for content-grounded assessment generation.

Verifies that:
- ContentExtractor correctly extracts terms, statements, relationships
- Question generators produce content-grounded output (not placeholders)
- Validators reject placeholder content
- LeakCheckValidator implements the Validator protocol
- QuestionQualityValidator scores grounded vs ungrounded questions
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------- Sample data ----------

SAMPLE_CHUNKS = [
    {
        "id": "chunk_001",
        "text": (
            "<p><strong>Cognitive Load Theory</strong> is defined as "
            "the framework describing how working memory capacity "
            "limits the amount of information a learner can process "
            "simultaneously. Cognitive Load Theory was developed by "
            "John Sweller in 1988.</p>"
            "<p><strong>Intrinsic load</strong> refers to the inherent "
            "difficulty of the material being learned. For example, "
            "learning basic arithmetic has lower intrinsic load than "
            "solving differential equations.</p>"
        ),
        "chunk_type": "explanation",
        "concept_tags": ["cognitive-load", "working-memory"],
        "difficulty": "intermediate",
    },
    {
        "id": "chunk_002",
        "text": (
            "<p>Extraneous load is caused by poorly designed instruction "
            "that does not contribute to learning. Unlike intrinsic load, "
            "extraneous load can always be reduced through better "
            "instructional design.</p>"
            "<p>Germane load involves the cognitive effort required to "
            "construct and automate schemas. Germane load increases "
            "when learners engage in meaningful processing.</p>"
        ),
        "chunk_type": "explanation",
        "concept_tags": ["cognitive-load", "instructional-design"],
        "difficulty": "intermediate",
    },
    {
        "id": "chunk_003",
        "text": (
            "<p>To apply cognitive load theory in course design:</p>"
            "<ol>"
            "<li>Identify the intrinsic complexity of the material</li>"
            "<li>Minimize extraneous cognitive load through clear layouts</li>"
            "<li>Maximize germane load via worked examples</li>"
            "<li>Sequence content from simple to complex</li>"
            "</ol>"
        ),
        "chunk_type": "example",
        "concept_tags": ["cognitive-load", "course-design"],
        "difficulty": "advanced",
    },
]

PLACEHOLDER_QUESTION = {
    "question_id": "Q-placeholder",
    "question_type": "multiple_choice",
    "stem": "<p>What is the concept from LO-001?</p>",
    "bloom_level": "remember",
    "objective_id": "LO-001",
    "choices": [
        {"text": "<p>Correct answer based on content</p>", "is_correct": True},
        {"text": "<p>Plausible distractor A</p>", "is_correct": False},
        {"text": "<p>Plausible distractor B</p>", "is_correct": False},
        {"text": "<p>Plausible distractor C</p>", "is_correct": False},
    ],
    "feedback": "<p>Review content for objective LO-001.</p>",
}


# ---------- ContentExtractor tests ----------

class TestContentExtractor:
    def setup_method(self):
        from Trainforge.generators.content_extractor import ContentExtractor
        self.extractor = ContentExtractor()

    def test_extract_key_terms_finds_definitions(self):
        terms = self.extractor.extract_key_terms(SAMPLE_CHUNKS)
        term_names = [t.term.lower() for t in terms]
        # Should find "Cognitive Load Theory" defined with "is defined as"
        assert any("cognitive load" in name for name in term_names), (
            f"Expected 'cognitive load' in terms, got: {term_names}"
        )

    def test_extract_key_terms_finds_bold_terms(self):
        terms = self.extractor.extract_key_terms(SAMPLE_CHUNKS)
        term_names = [t.term.lower() for t in terms]
        # Should find "Intrinsic load" from <strong> tag
        assert any("intrinsic" in name for name in term_names), (
            f"Expected 'intrinsic' in terms, got: {term_names}"
        )

    def test_extract_key_terms_includes_source_chunk_id(self):
        terms = self.extractor.extract_key_terms(SAMPLE_CHUNKS)
        assert all(t.source_chunk_id for t in terms)

    def test_extract_factual_statements(self):
        statements = self.extractor.extract_factual_statements(SAMPLE_CHUNKS)
        assert len(statements) > 0
        # All statements should be declarative (no questions)
        for s in statements:
            assert not s.statement.endswith("?")
            assert len(s.statement) >= 20

    def test_extract_factual_statements_have_subjects(self):
        statements = self.extractor.extract_factual_statements(SAMPLE_CHUNKS)
        for s in statements:
            assert s.key_subject, f"Statement missing subject: {s.statement}"

    def test_extract_relationships(self):
        rels = self.extractor.extract_relationships(SAMPLE_CHUNKS)
        # Should find "Unlike intrinsic load, extraneous load..."
        assert len(rels) > 0, "Expected at least one relationship"

    def test_extract_procedures(self):
        procs = self.extractor.extract_procedures(SAMPLE_CHUNKS)
        assert len(procs) > 0, "Expected at least one procedure from chunk_003"
        assert len(procs[0].steps) >= 2

    def test_extract_examples(self):
        examples = self.extractor.extract_examples(SAMPLE_CHUNKS)
        # chunk_001 has "For example, learning basic arithmetic..."
        assert len(examples) > 0

    def test_extract_all(self):
        result = self.extractor.extract_all(SAMPLE_CHUNKS)
        assert "key_terms" in result
        assert "factual_statements" in result
        assert "relationships" in result
        assert "procedures" in result
        assert "examples" in result


# ---------- Content-grounded generation tests ----------

class TestContentGroundedGeneration:
    def setup_method(self):
        from Trainforge.generators.assessment_generator import AssessmentGenerator
        self.generator = AssessmentGenerator(capture=None, check_leaks=False)

    def test_mcq_with_chunks_not_placeholder(self):
        question = self.generator._generate_multiple_choice(
            "Q-test-1", "LO-001", "remember",
            {"verbs": ["define"], "patterns": ["What is...?"],
             "question_types": ["multiple_choice"]},
            SAMPLE_CHUNKS,
        )
        # Should NOT contain placeholder text
        assert "Correct answer based on content" not in question.stem
        assert "TEMPLATE_FALLBACK" not in (question.generation_rationale or "")
        # Should reference actual content
        for choice in question.choices:
            assert "Plausible distractor" not in choice["text"]

    def test_mcq_without_chunks_falls_back(self):
        question = self.generator._generate_multiple_choice(
            "Q-test-2", "LO-001", "remember",
            {"verbs": ["define"], "patterns": ["What is...?"],
             "question_types": ["multiple_choice"]},
            None,
        )
        assert "TEMPLATE_FALLBACK" in (question.generation_rationale or "")

    def test_true_false_with_chunks_uses_content(self):
        question = self.generator._generate_true_false(
            "Q-test-3", "LO-001", "remember",
            {"verbs": ["identify"], "patterns": ["Which of the following...?"],
             "question_types": ["true_false"]},
            SAMPLE_CHUNKS,
        )
        stem_text = question.stem.replace("<p>", "").replace("</p>", "")
        assert len(stem_text) > 20
        assert "Statement about" not in stem_text
        assert "TEMPLATE_FALLBACK" not in (question.generation_rationale or "")

    def test_fill_in_blank_with_chunks(self):
        question = self.generator._generate_fill_in_blank(
            "Q-test-4", "LO-001", "remember",
            {"verbs": ["recall"], "patterns": ["What is...?"],
             "question_types": ["fill_in_blank"]},
            SAMPLE_CHUNKS,
        )
        if "TEMPLATE_FALLBACK" not in (question.generation_rationale or ""):
            assert "_______" in question.stem
            assert question.correct_answer != "concept term"

    def test_essay_with_chunks(self):
        question = self.generator._generate_essay(
            "Q-test-5", "LO-001", "evaluate",
            {"verbs": ["evaluate"], "patterns": ["Evaluate the effectiveness..."],
             "question_types": ["essay"]},
            SAMPLE_CHUNKS,
        )
        if "TEMPLATE_FALLBACK" not in (question.generation_rationale or ""):
            assert "concepts from LO-001" not in question.stem

    def test_short_answer_with_chunks(self):
        question = self.generator._generate_short_answer(
            "Q-test-6", "LO-001", "apply",
            {"verbs": ["apply"], "patterns": ["Apply X to..."],
             "question_types": ["short_answer"]},
            SAMPLE_CHUNKS,
        )
        if "TEMPLATE_FALLBACK" not in (question.generation_rationale or ""):
            assert "key points from LO-001" not in question.stem

    def test_negate_statement(self):
        assert "not" in self.generator._negate_statement(
            "The theory is based on evidence."
        ).lower() or "never" in self.generator._negate_statement(
            "The theory is based on evidence."
        ).lower()

    def test_negate_statement_swaps_qualifiers(self):
        result = self.generator._negate_statement(
            "Extraneous load always increases difficulty."
        )
        assert "never" in result.lower()

    def test_full_generate_with_chunks(self):
        assessment = self.generator.generate(
            course_code="TEST_101",
            objective_ids=["LO-001"],
            bloom_levels=["remember"],
            question_count=3,
            source_chunks=SAMPLE_CHUNKS,
        )
        assert len(assessment.questions) == 3
        # At least some questions should be content-grounded
        grounded = [
            q for q in assessment.questions
            if "TEMPLATE_FALLBACK" not in (q.generation_rationale or "")
        ]
        assert len(grounded) > 0, "Expected at least one content-grounded question"


# ---------- Placeholder detection tests ----------

class TestPlaceholderDetection:
    def test_validator_catches_placeholder_stem(self):
        from lib.validators.assessment import AssessmentQualityValidator
        validator = AssessmentQualityValidator()
        result = validator.validate({
            "assessment_data": {
                "questions": [PLACEHOLDER_QUESTION],
            },
        })
        placeholder_issues = [
            i for i in result.issues
            if i.code in ("PLACEHOLDER_QUESTION", "PLACEHOLDER_CHOICE", "PLACEHOLDER_ANSWER", "PLACEHOLDER_FEEDBACK")
        ]
        assert len(placeholder_issues) > 0, (
            f"Expected placeholder issues, got: {[i.code for i in result.issues]}"
        )

    def test_validator_passes_grounded_question(self):
        from lib.validators.assessment import AssessmentQualityValidator
        validator = AssessmentQualityValidator()
        grounded_question = {
            "question_id": "Q-grounded",
            "question_type": "multiple_choice",
            "stem": "<p>Which of the following best describes <em>Cognitive Load Theory</em>?</p>",
            "bloom_level": "remember",
            "objective_id": "LO-001",
            "choices": [
                {"text": "<p>The framework describing how working memory capacity limits information processing</p>", "is_correct": True},
                {"text": "<p>A method for designing multiple-choice assessments</p>", "is_correct": False},
                {"text": "<p>The process of automating procedural knowledge</p>", "is_correct": False},
                {"text": "<p>A technique for reducing test anxiety</p>", "is_correct": False},
            ],
            "feedback": "<p>Cognitive Load Theory was developed by John Sweller to describe working memory constraints.</p>",
        }
        result = validator.validate({
            "assessment_data": {"questions": [grounded_question]},
        })
        placeholder_issues = [
            i for i in result.issues
            if i.code in ("PLACEHOLDER_QUESTION", "PLACEHOLDER_CHOICE")
        ]
        assert len(placeholder_issues) == 0, (
            f"Grounded question should not trigger placeholder: {placeholder_issues}"
        )


# ---------- LeakCheckValidator protocol test ----------

class TestLeakCheckValidator:
    def test_implements_validator_protocol(self):
        from lib.validators.leak_check import LeakCheckValidator
        v = LeakCheckValidator()
        assert hasattr(v, "name")
        assert hasattr(v, "version")
        assert hasattr(v, "validate")

    def test_validates_clean_assessment(self):
        from lib.validators.leak_check import LeakCheckValidator
        v = LeakCheckValidator()
        result = v.validate({
            "assessment_data": {
                "assessment_id": "test-001",
                "questions": [
                    {
                        "question_id": "Q1",
                        "stem": "<p>What is the capital of France?</p>",
                        "choices": [
                            {"text": "Paris", "is_correct": True},
                            {"text": "London", "is_correct": False},
                        ],
                    }
                ],
            },
        })
        assert result.passed
        assert result.score is not None


# ---------- QuestionQualityValidator tests ----------

class TestQuestionQualityValidator:
    def test_scores_grounded_higher_than_placeholder(self):
        from lib.validators.question_quality import QuestionQualityValidator
        v = QuestionQualityValidator()

        grounded = v.validate({
            "assessment_data": {
                "questions": [{
                    "question_id": "Q1",
                    "question_type": "multiple_choice",
                    "stem": "<p>Which of the following best describes Cognitive Load Theory?</p>",
                    "bloom_level": "remember",
                    "choices": [
                        {"text": "<p>The framework describing how working memory capacity limits processing</p>", "is_correct": True},
                        {"text": "<p>A method for designing instructional materials</p>", "is_correct": False},
                        {"text": "<p>The process of automating knowledge schemas</p>", "is_correct": False},
                        {"text": "<p>A technique for reducing cognitive interference</p>", "is_correct": False},
                    ],
                    "feedback": "<p>Cognitive Load Theory describes working memory constraints on learning.</p>",
                }],
            },
            "source_chunks": SAMPLE_CHUNKS,
        })

        placeholder_result = v.validate({
            "assessment_data": {"questions": [PLACEHOLDER_QUESTION]},
            "source_chunks": SAMPLE_CHUNKS,
        })

        assert grounded.score > placeholder_result.score, (
            f"Grounded ({grounded.score}) should score higher than "
            f"placeholder ({placeholder_result.score})"
        )

    def test_rejects_no_questions(self):
        from lib.validators.question_quality import QuestionQualityValidator
        v = QuestionQualityValidator()
        result = v.validate({"assessment_data": {"questions": []}})
        assert not result.passed
