# Objective Synthesizer Agent

Generates comprehensive learning objectives from extracted textbook structure. Applies the Equal Treatment Principle - ALL extracted content receives objectives without AI filtering based on perceived importance.

## Core Function

Transform textbook structure JSON into hierarchical learning objectives (Course → Chapter → Section) with Bloom's taxonomy alignment, key concept mapping, and assessment suggestions.

## Equal Treatment Principle

**CRITICAL**: This agent does NOT filter or rank content based on perceived importance.

```
EQUAL TREATMENT REQUIREMENTS:
✅ Generate objective for EVERY definition
✅ Generate objective for EVERY key term
✅ Generate objective for EVERY procedure
✅ Generate objective for EVERY section
✅ Generate objective for EVERY explicit objective found
❌ DO NOT skip content deemed "less important"
❌ DO NOT filter based on word count thresholds
❌ DO NOT prioritize some topics over others
```

## Input Requirements

### Primary Input
Textbook structure JSON from `textbook-ingestor` agent:
- Location: `agent_workspaces/textbook_ingestor_workspace/structure_extraction/textbook_structure.json`
- Schema: `schemas/learning-objectives/textbook_structure_schema.json`

### Input Structure
```json
{
  "documentInfo": { ... },
  "tableOfContents": [ ... ],
  "chapters": [
    {
      "id": "ch1",
      "headingText": "Chapter Title",
      "explicitObjectives": [],
      "contentBlocks": [],
      "sections": []
    }
  ],
  "extractedConcepts": {
    "definitions": [],
    "keyTerms": [],
    "procedures": [],
    "examples": []
  },
  "reviewQuestions": []
}
```

## Workspace Structure

```
agent_workspaces/objective_synthesizer_workspace/
├── input/
│   └── textbook_structure.json
├── output/
│   ├── learning_objectives.json
│   └── learning_objectives.md
├── processing/
│   ├── bloom_mappings.json
│   └── concept_index.json
└── synthesizer_scratchpad.md
```

## Processing Pipeline

### Phase 1: Structure Analysis

```python
def analyze_structure(structure: Dict) -> StructureAnalysis:
    """
    Analyze textbook structure to plan objective generation.

    Counts all content that will receive objectives (Equal Treatment).
    """
    analysis = StructureAnalysis()

    # Count ALL definitions (will each get an objective)
    definitions = structure.get('extractedConcepts', {}).get('definitions', [])
    analysis.definition_count = len(definitions)

    # Count ALL key terms (will each get an objective)
    key_terms = structure.get('extractedConcepts', {}).get('keyTerms', [])
    analysis.key_term_count = len(key_terms)

    # Count ALL procedures (will each get an objective)
    procedures = structure.get('extractedConcepts', {}).get('procedures', [])
    analysis.procedure_count = len(procedures)

    # Count ALL sections (will each get at least one objective)
    for chapter in structure.get('chapters', []):
        analysis.chapter_count += 1
        analysis.section_count += count_sections_recursive(chapter.get('sections', []))

    # Count explicit objectives (will each be formalized)
    for chapter in structure.get('chapters', []):
        analysis.explicit_objective_count += len(chapter.get('explicitObjectives', []))

    return analysis
```

### Phase 2: Objective Generation

#### Course-Level Objectives
```python
def generate_course_objectives(structure: Dict) -> List[LearningObjective]:
    """
    Generate 8-12 course-level objectives from chapter topics.

    Course objectives span higher Bloom's levels:
    - Understand, Apply, Analyze, Evaluate
    """
    objectives = []

    for i, chapter in enumerate(structure.get('chapters', [])[:8]):
        # Rotate through higher-order levels
        levels = [BloomLevel.UNDERSTAND, BloomLevel.APPLY,
                  BloomLevel.ANALYZE, BloomLevel.EVALUATE]
        level = levels[i % len(levels)]

        verb = get_verb_for_level(level)

        objective = LearningObjective(
            objective_id=f"CO_{i+1}",
            statement=f"{verb.capitalize()} {chapter['headingText'].lower()}",
            bloom_level=level,
            bloom_verb=verb,
            hierarchy_level="course"
        )
        objectives.append(objective)

    return objectives
```

#### Chapter-Level Objectives
```python
def generate_chapter_objectives(chapter: Dict, concepts: Dict) -> List[LearningObjective]:
    """
    Generate objectives for a chapter.

    Includes:
    - Chapter summary objective
    - Explicit objectives (formalized)
    - Definition objectives
    - Key term objectives
    - Procedure objectives
    """
    objectives = []
    chapter_id = chapter.get('id', 'ch1')

    # 1. Chapter summary objective
    obj = format_chapter_objective(chapter)
    objectives.append(obj)

    # 2. Explicit objectives (EQUAL TREATMENT - all of them)
    for exp in chapter.get('explicitObjectives', []):
        obj = format_explicit_objective(exp['text'], chapter_id)
        objectives.append(obj)

    # 3. Definition objectives (EQUAL TREATMENT - all of them)
    chapter_definitions = [d for d in concepts.get('definitions', [])
                          if d.get('chapterId') == chapter_id
                          and not d.get('sectionId')]
    for defn in chapter_definitions:
        obj = format_definition_objective(defn, chapter_id)
        objectives.append(obj)

    # 4. Key term objectives (EQUAL TREATMENT - all of them)
    chapter_terms = [t for t in concepts.get('keyTerms', [])
                    if t.get('chapterId') == chapter_id
                    and not t.get('sectionId')]
    for term in chapter_terms:
        obj = format_key_term_objective(term, chapter_id)
        objectives.append(obj)

    # 5. Procedure objectives (EQUAL TREATMENT - all of them)
    chapter_procedures = [p for p in concepts.get('procedures', [])
                         if p.get('chapterId') == chapter_id
                         and not p.get('sectionId')]
    for proc in chapter_procedures:
        obj = format_procedure_objective(proc, chapter_id)
        objectives.append(obj)

    return objectives
```

#### Section-Level Objectives
```python
def generate_section_objectives(section: Dict, chapter_id: str, concepts: Dict) -> List[LearningObjective]:
    """
    Generate objectives for a section.

    EQUAL TREATMENT: Every section gets at least one objective.
    """
    objectives = []
    section_id = section.get('id', 's1')

    # 1. Section summary objective (EVERY section gets one)
    obj = format_section_objective(section, chapter_id, section_id)
    objectives.append(obj)

    # 2. Definition objectives in this section (ALL of them)
    section_definitions = [d for d in concepts.get('definitions', [])
                          if d.get('sectionId') == section_id]
    for defn in section_definitions:
        obj = format_definition_objective(defn, chapter_id, section_id)
        objectives.append(obj)

    # 3. Key term objectives in this section (ALL of them)
    section_terms = [t for t in concepts.get('keyTerms', [])
                    if t.get('sectionId') == section_id]
    for term in section_terms:
        obj = format_key_term_objective(term, chapter_id, section_id)
        objectives.append(obj)

    # 4. Procedure objectives in this section (ALL of them)
    section_procedures = [p for p in concepts.get('procedures', [])
                         if p.get('sectionId') == section_id]
    for proc in section_procedures:
        obj = format_procedure_objective(proc, chapter_id, section_id)
        objectives.append(obj)

    # 5. Process subsections recursively
    for subsection in section.get('subsections', []):
        sub_objectives = generate_section_objectives(
            subsection, chapter_id, concepts
        )
        objectives.extend(sub_objectives)

    return objectives
```

### Phase 3: Bloom's Taxonomy Mapping

```python
BLOOM_VERB_MAPPING = {
    BloomLevel.REMEMBER: [
        "define", "list", "recall", "identify", "name", "state",
        "label", "match", "recognize", "select"
    ],
    BloomLevel.UNDERSTAND: [
        "explain", "describe", "summarize", "classify", "compare",
        "interpret", "discuss", "paraphrase", "distinguish", "illustrate"
    ],
    BloomLevel.APPLY: [
        "apply", "demonstrate", "implement", "solve", "use",
        "execute", "compute", "calculate", "practice", "perform"
    ],
    BloomLevel.ANALYZE: [
        "analyze", "differentiate", "examine", "organize", "relate",
        "categorize", "deconstruct", "investigate", "contrast", "attribute"
    ],
    BloomLevel.EVALUATE: [
        "evaluate", "assess", "critique", "justify", "judge",
        "argue", "defend", "support", "recommend", "prioritize"
    ],
    BloomLevel.CREATE: [
        "create", "design", "construct", "develop", "formulate",
        "compose", "plan", "invent", "produce", "generate"
    ]
}
```

### Phase 4: Output Generation

#### JSON Output
```python
def generate_json_output(objectives: Dict) -> str:
    """
    Generate JSON conforming to learning_objectives_schema.json
    """
    output = {
        "documentMetadata": {
            "sourceType": "textbook",
            "sourcePath": structure['documentInfo']['sourcePath'],
            "sourceTitle": structure['documentInfo']['title'],
            "generationTimestamp": datetime.now().isoformat(),
            "toolVersion": "1.0.0",
            "extractionMethod": "semantic_structure"
        },
        "courseObjectives": [o.to_dict() for o in course_objectives],
        "chapters": [
            {
                "chapterId": ch.id,
                "chapterTitle": ch.title,
                "chapterObjectives": [o.to_dict() for o in ch.objectives],
                "sections": [s.to_dict() for s in ch.sections]
            }
            for ch in chapters
        ],
        "objectivesSummary": compute_summary(all_objectives)
    }
    return json.dumps(output, indent=2)
```

#### Markdown Output
```markdown
# Learning Objectives: [Source Title]

**Generated:** [timestamp]
**Source:** [source_path]

## Course-Level Objectives

1. **Analyze** network architectures and explain their use cases (Bloom's: Analyze)
2. **Apply** security protocols in various scenarios (Bloom's: Apply)

## Chapter 1: Introduction to Networking

### Chapter Objectives
- **Explain** computer networking fundamentals (Bloom's: Understand)
- **Define** key networking terminology (Bloom's: Remember)

### Section 1.1: What is a Network?
- LO 1.1.1: **Define** computer network (Bloom's: Remember)
- LO 1.1.2: **Identify** network components (Bloom's: Remember)

---

## Summary by Bloom's Level

- **Remember**: 24
- **Understand**: 18
- **Apply**: 32
- **Analyze**: 15
- **Evaluate**: 8
- **Create**: 5

**Total Objectives**: 102
```

## Output Schema

### Learning Objective Structure
```json
{
  "objectiveId": "ch1_s1_3",
  "statement": "Define computer network and explain its significance",
  "bloomLevel": "remember",
  "bloomVerb": "define",
  "keyConcepts": ["computer network", "nodes", "communication"],
  "sourceReference": {
    "type": "definition",
    "term": "computer network",
    "chapterId": "ch1",
    "sectionId": "ch1_s1"
  },
  "assessmentSuggestions": ["quiz", "exam", "matching"],
  "extractionSource": "definition"
}
```

## Integration Points

### Upstream
- Receives from: `textbook-ingestor` agent
- Input: `textbook_structure.json`

### Downstream
- Feeds into: `course-outliner` agent
- Output: `learning_objectives.json`

### Integration with Course-Outliner
```json
{
  "exam_objectives": { ... },         // From exam-research (if available)
  "textbook_objectives": { ... },     // From objective-synthesizer
  "merge_strategy": "union_with_dedup"
}
```

## Quality Validation

### Completeness Checks
```python
def validate_completeness(objectives: Dict, structure: Dict) -> ValidationResult:
    """
    Validate that all content received objectives (Equal Treatment).
    """
    issues = []

    # Check all definitions have objectives
    def_count = len(structure['extractedConcepts']['definitions'])
    def_obj_count = count_objectives_by_source(objectives, 'definition')
    if def_obj_count < def_count:
        issues.append(f"Missing {def_count - def_obj_count} definition objectives")

    # Check all key terms have objectives
    term_count = len(structure['extractedConcepts']['keyTerms'])
    term_obj_count = count_objectives_by_source(objectives, 'concept')
    if term_obj_count < term_count:
        issues.append(f"Missing {term_count - term_obj_count} key term objectives")

    # Check all sections have at least one objective
    for chapter in structure['chapters']:
        for section in chapter.get('sections', []):
            if not has_section_objective(objectives, section['id']):
                issues.append(f"Section {section['id']} has no objectives")

    return ValidationResult(valid=len(issues) == 0, issues=issues)
```

### Bloom's Distribution Validation
```python
def validate_bloom_distribution(objectives: List) -> ValidationResult:
    """
    Validate reasonable distribution across Bloom's levels.

    Target distribution:
    - Remember/Understand: 20-40%
    - Apply/Analyze: 40-60%
    - Evaluate/Create: 10-30%
    """
    total = len(objectives)
    by_level = count_by_bloom_level(objectives)

    foundation = by_level['remember'] + by_level['understand']
    core = by_level['apply'] + by_level['analyze']
    advanced = by_level['evaluate'] + by_level['create']

    issues = []

    foundation_pct = foundation / total * 100
    if foundation_pct > 50:
        issues.append(f"Foundation objectives too high: {foundation_pct:.1f}%")

    core_pct = core / total * 100
    if core_pct < 30:
        issues.append(f"Core objectives too low: {core_pct:.1f}%")

    return ValidationResult(valid=len(issues) == 0, issues=issues)
```

## Usage Examples

### Basic Usage
```python
# Generate objectives from structure
result = invoke_objective_synthesizer(
    structure_path="agent_workspaces/textbook_ingestor_workspace/structure_extraction/textbook_structure.json",
    workspace="agent_workspaces/objective_synthesizer_workspace/"
)
```

### With Existing Exam Objectives
```python
# Generate and merge with exam objectives
textbook_objectives = invoke_objective_synthesizer(
    structure_path="textbook_structure.json",
    workspace="workspace/"
)

# Merge with exam-research output
merged = merge_objectives(
    exam_objectives=exam_research_output,
    textbook_objectives=textbook_objectives,
    strategy="union_with_dedup"
)
```

## Success Criteria

### Equal Treatment Validation
- 100% of definitions receive objectives
- 100% of key terms receive objectives
- 100% of procedures receive objectives
- 100% of sections receive at least one objective

### Quality Metrics
- All objectives use valid Bloom's verbs
- All objectives have measurable statements
- All objectives include source references
- JSON output passes schema validation

### Output Completeness
- Course-level objectives: 8-12
- Chapter-level objectives: 3-10 per chapter
- Section-level objectives: 1+ per section
- Total objectives: Proportional to source content
