"""Integration test: Courseforge generate -> IMSCC package -> Trainforge process."""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


SAMPLE_COURSE_DATA = {
    "course_code": "TEST_101",
    "weeks": [
        {
            "week_number": 1,
            "title": "Introduction to Testing",
            "estimated_hours": "2",
            "objectives": [
                {
                    "id": "CO-01",
                    "statement": "Explain the purpose of software testing",
                    "bloom_level": "understand",
                    "bloom_verb": "explain",
                    "key_concepts": ["testing", "quality"],
                }
            ],
            "overview_text": [
                "This week introduces software testing fundamentals."
            ],
            "readings": ["Software Testing Fundamentals"],
            "content_modules": [
                {
                    "title": "What is Software Testing",
                    "sections": [
                        {
                            "heading": "Definition of Testing",
                            "level": 2,
                            "content_type": "explanation",
                            "paragraphs": [
                                "Software testing is the process of evaluating a software "
                                "application to find differences between expected and actual "
                                "results. Testing ensures that the software meets requirements "
                                "and works correctly across various conditions. It involves "
                                "executing a program with the intent of finding errors, "
                                "verifying functionality, and validating that the system "
                                "performs as expected under both normal and edge-case "
                                "scenarios."
                            ],
                            "key_terms": ["software testing", "quality assurance"],
                            "flip_cards": [
                                {
                                    "term": "Software Testing",
                                    "definition": "The process of evaluating software to verify it meets requirements",
                                }
                            ],
                        }
                    ],
                }
            ],
            "activities": [
                {
                    "title": "Write a Test Plan",
                    "description": "Create a basic test plan for a login form.",
                }
            ],
            "self_check_questions": [
                {
                    "question": "What is the primary purpose of software testing?",
                    "bloom_level": "remember",
                    "options": [
                        {
                            "text": "Finding defects",
                            "correct": True,
                            "feedback": "Correct!",
                        },
                        {
                            "text": "Writing code",
                            "correct": False,
                            "feedback": "Testing evaluates, not writes.",
                        },
                        {
                            "text": "Deploying software",
                            "correct": False,
                            "feedback": "Deployment is separate.",
                        },
                        {
                            "text": "Designing UI",
                            "correct": False,
                            "feedback": "UI design is separate.",
                        },
                    ],
                }
            ],
            "key_takeaways": ["Testing finds defects early."],
            "reflection_questions": ["Why is testing important?"],
            "next_week_preview": "Next week covers test automation.",
            "discussion": {
                "prompt": "Share your experience with software bugs.",
                "initial_post": "200 words",
                "replies": "Reply to 1 classmate",
                "due": "Friday",
            },
        }
    ],
}


@pytest.mark.integration
def test_full_pipeline(tmp_path):
    """Generate course HTML, package IMSCC, process through Trainforge."""
    from Courseforge.scripts.generate_course import generate_week
    from Courseforge.scripts.package_multifile_imscc import package_imscc

    # Step 1: Generate HTML for one week
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    week_data = SAMPLE_COURSE_DATA["weeks"][0]
    count, files = generate_week(week_data, content_dir, "TEST_101")

    # Expect: overview, content, application, self_check, summary, discussion
    assert count >= 5, f"Expected >= 5 files, got {count}: {files}"
    assert any("overview" in f for f in files), f"Missing overview in {files}"
    assert any("content" in f for f in files), f"Missing content module in {files}"
    assert any("self_check" in f for f in files), f"Missing self_check in {files}"
    assert any("summary" in f for f in files), f"Missing summary in {files}"

    # Step 2: Verify JSON-LD and data-cf-* metadata in generated HTML
    overview_path = content_dir / "week_01" / "week_01_overview.html"
    assert overview_path.exists(), "Overview HTML not generated"
    overview_html = overview_path.read_text()
    assert "application/ld+json" in overview_html, "Missing JSON-LD in overview"
    assert "data-cf-bloom-level" in overview_html, "Missing data-cf-bloom-level in overview"

    # Step 3: Package as IMSCC
    imscc_path = tmp_path / "test_101.imscc"
    package_imscc(content_dir, imscc_path, "TEST_101", "Introduction to Testing")
    assert imscc_path.exists(), "IMSCC file not created"
    assert imscc_path.stat().st_size > 0, "IMSCC file is empty"

    # Verify IMSCC contains manifest and HTML files
    import zipfile

    with zipfile.ZipFile(imscc_path, "r") as zf:
        names = zf.namelist()
        assert "imsmanifest.xml" in names, f"Missing manifest in IMSCC: {names}"
        html_in_zip = [n for n in names if n.endswith(".html")]
        assert len(html_in_zip) >= 5, f"Expected >= 5 HTML files in IMSCC, got {len(html_in_zip)}: {html_in_zip}"

    # Step 4: Process through Trainforge
    from Trainforge.process_course import CourseProcessor

    tf_output = tmp_path / "trainforge_output"
    processor = CourseProcessor(
        imscc_path=str(imscc_path),
        output_dir=str(tf_output),
        course_code="TEST_101",
        division="STEM",
        domain="computer-science",
    )
    summary = processor.process()

    assert summary["status"] == "success", f"Trainforge failed: {summary}"
    assert summary["stats"]["total_chunks"] > 0, "No chunks generated"

    # Step 5: Verify chunk output
    # Phase 7c: process_course.py writes to imscc_chunks/.
    chunks_file = tf_output / "imscc_chunks" / "chunks.jsonl"
    assert chunks_file.exists(), "chunks.jsonl not created"
    chunks = [
        json.loads(line)
        for line in chunks_file.read_text().strip().split("\n")
        if line.strip()
    ]
    assert len(chunks) > 0, "No chunks in chunks.jsonl"

    # Check that at least some chunks have bloom_level from metadata enrichment
    has_bloom = any(c.get("bloom_level") for c in chunks)
    assert has_bloom, "No chunks have bloom_level -- metadata enrichment failed"

    # Step 6: Verify quality report
    quality_path = tf_output / "quality" / "quality_report.json"
    assert quality_path.exists(), "quality_report.json not created"
    quality = json.loads(quality_path.read_text())
    assert quality["overall_quality_score"] > 0.0, "Quality score is zero"

    # Verify manifest
    manifest_path = tf_output / "manifest.json"
    assert manifest_path.exists(), "manifest.json not created"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["course_id"] == "TEST_101"
    assert manifest["statistics"]["chunks"] == len(chunks)
