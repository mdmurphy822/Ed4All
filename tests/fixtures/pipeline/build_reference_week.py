"""Build the 5 reference Week 01 HTML pages + the 3 reference LibV2 fixtures.

These fixtures show the **target shape** that workers α and γ must produce.
The integration test uses the Week 01 pages as a visual reference and the
LibV2 fixtures as strict-schema validation targets.

Topic: photosynthesis basics (same as ``fixture_corpus.pdf``).
Course code: ``TESTPIPE_101``.

This script is idempotent — re-running regenerates the same bytes. Output
files are committed so CI doesn't need to run this; the builder is kept
for transparency and dev-side tweaking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


COURSE_CODE = "TESTPIPE_101"
FIXTURE_DIR = Path(__file__).resolve().parent
REFERENCE_WEEK_DIR = FIXTURE_DIR / "reference_week_01"
REFERENCE_LIBV2_DIR = FIXTURE_DIR / "reference_libv2"


# ---------------------------------------------------------------------- #
# Shared content pool (objectives + sections + misconceptions)
# ---------------------------------------------------------------------- #

OBJECTIVES = [
    {
        "id": "CO-01",
        "statement": "Describe photosynthesis as a light-driven conversion of CO2 and water into glucose and oxygen.",
        "bloomLevel": "understand",
        "bloomVerb": "describe",
        "cognitiveDomain": "conceptual",
        "keyConcepts": ["photosynthesis", "chloroplast", "chlorophyll"],
        "assessmentSuggestions": ["multiple_choice", "short_answer", "fill_in_blank"],
    },
    {
        "id": "CO-02",
        "statement": "Differentiate the light-dependent reactions from the Calvin cycle.",
        "bloomLevel": "analyze",
        "bloomVerb": "differentiate",
        "cognitiveDomain": "conceptual",
        "keyConcepts": ["calvin-cycle", "light-dependent-reactions"],
        "assessmentSuggestions": ["multiple_choice", "essay", "short_answer"],
    },
    {
        "id": "CO-03",
        "statement": "Identify common misconceptions about photosynthesis and explain the correct science.",
        "bloomLevel": "evaluate",
        "bloomVerb": "identify",
        "cognitiveDomain": "metacognitive",
        "keyConcepts": ["misconception"],
        "assessmentSuggestions": ["essay", "multiple_choice", "short_answer"],
    },
]

MISCONCEPTIONS = [
    {
        "misconception": "Plants get their food from the soil.",
        "correction": "Plants produce their own food via photosynthesis; soil contributes water and minerals but not the carbon that builds plant biomass.",
    },
]


# ---------------------------------------------------------------------- #
# HTML helpers
# ---------------------------------------------------------------------- #


def _head(title: str, json_ld: dict) -> str:
    ld = json.dumps(json_ld, indent=2, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} &mdash; {COURSE_CODE}</title>
  <style>body{{font-family:system-ui,sans-serif;max-width:50em;margin:0 auto;padding:1em;line-height:1.6}}
.skip-link{{position:absolute;left:-9999px}}.skip-link:focus{{position:static}}
.flip-card,.self-check,.activity-card{{border:1px solid #ccc;padding:0.8em;margin:0.8em 0;border-radius:4px}}</style>
  <script type="application/ld+json">
{ld}
  </script>
</head>"""


def _body_open(week_num: int) -> str:
    return f"""<body>
  <a href="#main-content" class="skip-link" data-cf-role="template-chrome">Skip to main content</a>
  <header role="banner" data-cf-role="template-chrome">
    <p>{COURSE_CODE} &mdash; Week {week_num}</p>
  </header>
  <main id="main-content" role="main">"""


def _body_close() -> str:
    return """  </main>
  <footer role="contentinfo" data-cf-role="template-chrome">
    <p>&copy; 2026 """ + COURSE_CODE + """. All rights reserved.</p>
  </footer>
</body>
</html>"""


def _objectives_section(selected_ids: list[str]) -> str:
    items = []
    for obj in OBJECTIVES:
        if obj["id"] not in selected_ids:
            continue
        items.append(
            f'      <li data-cf-objective-id="{obj["id"]}"'
            f' data-cf-bloom-level="{obj["bloomLevel"]}"'
            f' data-cf-bloom-verb="{obj["bloomVerb"]}"'
            f' data-cf-cognitive-domain="{obj["cognitiveDomain"]}">'
            f'{obj["statement"]}</li>'
        )
    items_html = "\n".join(items)
    return f"""    <section id="objectives" class="objectives" aria-labelledby="objectives-heading">
      <h2 id="objectives-heading" data-cf-content-type="overview">Learning Objectives</h2>
      <ul>
{items_html}
      </ul>
    </section>"""


# ---------------------------------------------------------------------- #
# Per-page builders
# ---------------------------------------------------------------------- #


def build_overview() -> tuple[str, str]:
    """Return (filename, html) for the week 01 overview page."""
    json_ld = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": COURSE_CODE,
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "learningObjectives": OBJECTIVES,
        "sections": [
            {"heading": "Welcome to Photosynthesis", "contentType": "overview",
             "bloomRange": ["understand"]},
            {"heading": "This Week's Roadmap", "contentType": "overview",
             "bloomRange": ["remember", "understand"]},
        ],
    }
    html = _head("Week 1 Overview", json_ld) + "\n" + _body_open(1) + """
    <h1>Week 1: Photosynthesis &mdash; Overview</h1>
""" + _objectives_section(["CO-01", "CO-02", "CO-03"]) + """
    <section id="welcome" aria-labelledby="welcome-heading">
      <h2 id="welcome-heading" data-cf-content-type="overview"
          data-cf-key-terms="photosynthesis,chloroplast"
          data-cf-bloom-range="understand">Welcome to Photosynthesis</h2>
      <p>This week introduces <strong class="key-term">photosynthesis</strong> as
      the process by which plants convert light energy into chemical energy.</p>
    </section>
    <section id="roadmap" aria-labelledby="roadmap-heading">
      <h2 id="roadmap-heading" data-cf-content-type="overview"
          data-cf-bloom-range="remember,understand">This Week's Roadmap</h2>
      <p>You will move through five pages: this overview, a content explanation,
      an application activity, a self-check, and a summary.</p>
    </section>
""" + _body_close()
    return "week_01_overview.html", html


def build_content() -> tuple[str, str]:
    json_ld = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": COURSE_CODE,
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01_two_stages",
        "learningObjectives": [OBJECTIVES[0], OBJECTIVES[1]],
        "sections": [
            {
                "heading": "What is Photosynthesis?",
                "contentType": "definition",
                "keyTerms": [
                    {"term": "photosynthesis", "definition": "A light-driven conversion of CO2 and water into glucose and oxygen."},
                    {"term": "chloroplast", "definition": "The plant organelle where photosynthesis occurs."},
                ],
                "bloomRange": ["remember", "understand"],
            },
            {
                "heading": "The Two Stages",
                "contentType": "explanation",
                "keyTerms": [
                    {"term": "calvin-cycle", "definition": "The light-independent stage that fixes carbon into sugars."},
                ],
                "bloomRange": ["understand", "analyze"],
            },
        ],
        "prerequisitePages": ["week_01_overview"],
    }
    html = _head("Content: The Two Stages", json_ld) + "\n" + _body_open(1) + """
    <h1>Week 1 &mdash; The Two Stages of Photosynthesis</h1>
""" + _objectives_section(["CO-01", "CO-02"]) + """
    <section id="what-is" aria-labelledby="what-is-heading">
      <h2 id="what-is-heading" data-cf-content-type="definition"
          data-cf-key-terms="photosynthesis,chloroplast"
          data-cf-bloom-range="remember,understand">What is Photosynthesis?</h2>
      <p><strong class="key-term">Photosynthesis</strong> is the process by
      which plants convert light energy into stored chemical energy. It takes
      place inside <strong class="key-term">chloroplasts</strong>, which
      contain the pigment chlorophyll.</p>
      <div class="flip-card" data-cf-component="flip-card"
           data-cf-purpose="term-definition" data-cf-term="chloroplast">
        <h3>Chloroplast</h3>
        <p>The plant organelle where photosynthesis occurs.</p>
      </div>
    </section>
    <section id="two-stages" aria-labelledby="two-stages-heading">
      <h2 id="two-stages-heading" data-cf-content-type="explanation"
          data-cf-key-terms="calvin-cycle"
          data-cf-bloom-range="understand,analyze">The Two Stages</h2>
      <p>The <strong class="key-term">light-dependent reactions</strong> split
      water and generate ATP and NADPH in the thylakoid membranes. The
      <strong class="key-term">Calvin cycle</strong> then consumes that energy
      to fix CO2 into glucose in the stroma.</p>
    </section>
""" + _body_close()
    return "week_01_content_01_two_stages.html", html


def build_application() -> tuple[str, str]:
    json_ld = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": COURSE_CODE,
        "weekNumber": 1,
        "moduleType": "application",
        "pageId": "week_01_application",
        "learningObjectives": [OBJECTIVES[1]],
        "sections": [
            {"heading": "Apply: Trace the Carbon", "contentType": "exercise",
             "bloomRange": ["apply", "analyze"]},
        ],
        "prerequisitePages": ["week_01_content_01_two_stages"],
    }
    html = _head("Application", json_ld) + "\n" + _body_open(1) + """
    <h1>Week 1 &mdash; Application</h1>
""" + _objectives_section(["CO-02"]) + """
    <section id="apply" aria-labelledby="apply-heading">
      <h2 id="apply-heading" data-cf-content-type="exercise"
          data-cf-bloom-range="apply,analyze">Apply: Trace the Carbon</h2>
      <p>Work through this scenario to trace a single carbon atom from CO2 in
      the atmosphere all the way to glucose inside a plant cell.</p>
      <div class="activity-card" data-cf-component="activity"
           data-cf-purpose="practice" data-cf-bloom-level="apply"
           data-cf-objective-ref="CO-02">
        <h3>Activity 1: Carbon's Journey</h3>
        <p>Sketch a diagram showing the carbon atom moving through the
        light-dependent reactions and the Calvin cycle.</p>
      </div>
    </section>
""" + _body_close()
    return "week_01_application.html", html


def build_self_check() -> tuple[str, str]:
    json_ld = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": COURSE_CODE,
        "weekNumber": 1,
        # module_type enum lacks "self_check" — emit as "assessment" so the
        # JSON-LD validates. Documented in contracts.md § schema gaps.
        "moduleType": "assessment",
        "pageId": "week_01_self_check",
        "learningObjectives": [OBJECTIVES[0], OBJECTIVES[2]],
        "sections": [
            {"heading": "Self-Check Questions", "contentType": "exercise",
             "bloomRange": ["remember", "understand"]},
        ],
        "misconceptions": MISCONCEPTIONS,
        "prerequisitePages": ["week_01_application"],
    }
    html = _head("Self-Check", json_ld) + "\n" + _body_open(1) + """
    <h1>Week 1 &mdash; Self-Check</h1>
""" + _objectives_section(["CO-01", "CO-03"]) + """
    <section id="self-check" aria-labelledby="self-check-heading">
      <h2 id="self-check-heading" data-cf-content-type="exercise"
          data-cf-bloom-range="remember,understand">Self-Check Questions</h2>
      <div class="self-check" data-cf-component="self-check"
           data-cf-purpose="formative-assessment" data-cf-bloom-level="remember"
           data-cf-objective-ref="CO-01">
        <h3>Question 1</h3>
        <p>Where does photosynthesis primarily take place in a plant cell?</p>
        <label class="sc-option" data-correct="true"><input type="radio" name="q1"> Chloroplast</label>
        <label class="sc-option" data-correct="false"><input type="radio" name="q1"> Nucleus</label>
        <label class="sc-option" data-correct="false"><input type="radio" name="q1"> Mitochondrion</label>
      </div>
    </section>
""" + _body_close()
    return "week_01_self_check.html", html


def build_summary() -> tuple[str, str]:
    json_ld = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": COURSE_CODE,
        "weekNumber": 1,
        "moduleType": "summary",
        "pageId": "week_01_summary",
        "learningObjectives": OBJECTIVES,
        "sections": [
            {"heading": "Key Takeaways", "contentType": "summary",
             "bloomRange": ["remember", "understand"]},
        ],
        "prerequisitePages": ["week_01_self_check"],
    }
    html = _head("Summary", json_ld) + "\n" + _body_open(1) + """
    <h1>Week 1 &mdash; Summary</h1>
""" + _objectives_section(["CO-01", "CO-02", "CO-03"]) + """
    <section id="takeaways" aria-labelledby="takeaways-heading">
      <h2 id="takeaways-heading" data-cf-content-type="summary"
          data-cf-key-terms="photosynthesis,calvin-cycle"
          data-cf-bloom-range="remember,understand">Key Takeaways</h2>
      <p>Photosynthesis stores light energy as glucose in two coupled
      stages: the light-dependent reactions and the Calvin cycle.</p>
    </section>
""" + _body_close()
    return "week_01_summary.html", html


# ---------------------------------------------------------------------- #
# LibV2 reference fixtures
# ---------------------------------------------------------------------- #


def _mc_id(misconception: str, correction: str) -> str:
    """Content-hash ID per ``schemas/knowledge/misconception.schema.json``."""
    digest = hashlib.sha256(
        (misconception + "|" + correction).encode("utf-8")
    ).hexdigest()
    return f"mc_{digest[:16]}"


def _chunk_id(index: int) -> str:
    """Legacy chunk ID shape (position-based). Pattern:
    ``^[a-z][a-z0-9_]*_chunk_\\d{5}$`` per ``chunk_v4.schema.json``.
    """
    return f"testpipe_101_chunk_{index:05d}"


def build_libv2_chunks() -> list[dict]:
    base_source = {
        "course_id": "TESTPIPE_101",
        "module_id": "week_01",
    }
    return [
        {
            "id": _chunk_id(1),
            "schema_version": "v4",
            "chunk_type": "explanation",
            "text": "Photosynthesis is the biological process by which plants convert light into stored chemical energy as glucose.",
            "html": "<section><h2>What is Photosynthesis?</h2><p>Photosynthesis is the biological process by which plants convert light into stored chemical energy as glucose.</p></section>",
            "follows_chunk": None,
            "source": {
                **base_source,
                "lesson_id": "week_01_content_01_two_stages",
                "section_heading": "What is Photosynthesis?",
                "position_in_module": 0,
            },
            "concept_tags": ["photosynthesis", "chloroplast"],
            "learning_outcome_refs": ["co-01"],
            "difficulty": "foundational",
            "tokens_estimate": 28,
            "word_count": 17,
            "bloom_level": "understand",
            "content_type_label": "definition",
            "key_terms": [
                {"term": "photosynthesis", "definition": "A light-driven conversion of CO2 and water into glucose and oxygen."}
            ],
        },
        {
            "id": _chunk_id(2),
            "schema_version": "v4",
            "chunk_type": "explanation",
            "text": "The light-dependent reactions split water and generate ATP and NADPH in the thylakoid membranes of the chloroplast.",
            "html": "<section><h2>The Two Stages</h2><p>The light-dependent reactions split water and generate ATP and NADPH in the thylakoid membranes of the chloroplast.</p></section>",
            "follows_chunk": _chunk_id(1),
            "source": {
                **base_source,
                "lesson_id": "week_01_content_01_two_stages",
                "section_heading": "The Two Stages",
                "position_in_module": 1,
            },
            "concept_tags": ["light-dependent-reactions", "atp", "nadph"],
            "learning_outcome_refs": ["co-02"],
            "difficulty": "intermediate",
            "tokens_estimate": 30,
            "word_count": 19,
            "bloom_level": "analyze",
            "content_type_label": "explanation",
        },
        {
            "id": _chunk_id(3),
            "schema_version": "v4",
            "chunk_type": "example",
            "text": "The Calvin cycle consumes ATP and NADPH to fix CO2 into glucose in the stroma of the chloroplast.",
            "html": "<section><h2>The Calvin Cycle</h2><p>The Calvin cycle consumes ATP and NADPH to fix CO2 into glucose in the stroma of the chloroplast.</p></section>",
            "follows_chunk": _chunk_id(2),
            "source": {
                **base_source,
                "lesson_id": "week_01_content_01_two_stages",
                "section_heading": "The Calvin Cycle",
                "position_in_module": 2,
            },
            "concept_tags": ["calvin-cycle", "carbon-fixation"],
            "learning_outcome_refs": ["co-02"],
            "difficulty": "intermediate",
            "tokens_estimate": 26,
            "word_count": 18,
            "bloom_level": "understand",
            "content_type_label": "example",
        },
    ]


def build_libv2_graph() -> dict:
    return {
        "kind": "concept_semantic",
        "generated_at": "2026-04-20T00:00:00+00:00",
        "rule_versions": {
            "is_a_from_key_terms": 1,
            "related_from_cooccurrence": 1,
            "defined_by_from_first_mention": 1,
        },
        "nodes": [
            {"id": "photosynthesis", "label": "Photosynthesis", "frequency": 3,
             "occurrences": [_chunk_id(1), _chunk_id(2)]},
            {"id": "chloroplast", "label": "Chloroplast", "frequency": 2,
             "occurrences": [_chunk_id(1), _chunk_id(2)]},
            {"id": "calvin-cycle", "label": "Calvin Cycle", "frequency": 1,
             "occurrences": [_chunk_id(3)]},
            {"id": "light-dependent-reactions", "label": "Light Dependent Reactions",
             "frequency": 1, "occurrences": [_chunk_id(2)]},
            {"id": "atp", "label": "ATP", "frequency": 2,
             "occurrences": [_chunk_id(2), _chunk_id(3)]},
        ],
        "edges": [
            {
                "source": "chloroplast",
                "target": "photosynthesis",
                "type": "is-a",
                "confidence": 0.8,
                "provenance": {
                    "rule": "is_a_from_key_terms",
                    "rule_version": 1,
                    "evidence": {
                        "chunk_id": _chunk_id(1),
                        "term": "chloroplast",
                        "definition_excerpt": "The plant organelle where photosynthesis occurs.",
                        "pattern": "is an?",
                    },
                },
            },
            {
                "source": "photosynthesis",
                "target": "calvin-cycle",
                "type": "related-to",
                "confidence": 0.7,
                "provenance": {
                    "rule": "related_from_cooccurrence",
                    "rule_version": 1,
                    "evidence": {
                        "cooccurrence_weight": 2,
                        "threshold": 1,
                    },
                },
            },
            {
                "source": "calvin-cycle",
                "target": "photosynthesis",
                "type": "defined-by",
                "confidence": 0.9,
                "provenance": {
                    "rule": "defined_by_from_first_mention",
                    "rule_version": 1,
                    "evidence": {
                        "chunk_id": _chunk_id(3),
                        "concept_slug": "calvin-cycle",
                        "first_mention_position": 0,
                    },
                },
            },
        ],
    }


def build_libv2_misconceptions() -> dict:
    entries = []
    for m in MISCONCEPTIONS:
        entries.append({
            "id": _mc_id(m["misconception"], m["correction"]),
            "misconception": m["misconception"],
            "correction": m["correction"],
            "concept_id": "photosynthesis",
            "lo_id": "CO-03",
        })
    return {"misconceptions": entries}


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #


def main() -> None:
    REFERENCE_WEEK_DIR.mkdir(parents=True, exist_ok=True)
    # Phase 7c: write to imscc_chunks/ (canonical).
    (REFERENCE_LIBV2_DIR / "imscc_chunks").mkdir(parents=True, exist_ok=True)
    (REFERENCE_LIBV2_DIR / "graph").mkdir(parents=True, exist_ok=True)

    for builder in (
        build_overview, build_content, build_application,
        build_self_check, build_summary,
    ):
        name, html = builder()
        (REFERENCE_WEEK_DIR / name).write_text(html, encoding="utf-8")
        print(f"Wrote {REFERENCE_WEEK_DIR / name}")

    chunks_path = REFERENCE_LIBV2_DIR / "imscc_chunks" / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in build_libv2_chunks():
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {chunks_path}")

    graph_path = REFERENCE_LIBV2_DIR / "graph" / "concept_graph_semantic.json"
    graph_path.write_text(
        json.dumps(build_libv2_graph(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {graph_path}")

    mc_path = REFERENCE_LIBV2_DIR / "graph" / "misconceptions.json"
    mc_path.write_text(
        json.dumps(build_libv2_misconceptions(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {mc_path}")


if __name__ == "__main__":
    main()
