# Textbook Ingestor Agent

Entry point for supported educational source materials (textbooks, course
packages, and exam objectives) that extracts semantic structure for learning
objective generation.

## Core Function

Accept various input formats (PDF, IMSCC, HTML, text) and route to appropriate processing pipelines, producing SemantiK accessible HTML with semantic structure that feeds into the objective-synthesizer agent.

## Input Types Supported

| Input Type | Detection | Processing Route |
|------------|-----------|-----------------|
| PDF (textbook) | `.pdf` extension | SemantiK v2 cascade conversion |
| IMSCC Package | `.imscc` extension, ZIP with `imscc_manifest.xml` | IMSCC intake parser → SemantiK enhancement |
| HTML (SemantiK-processed) | Skip-link, ARIA landmarks, semantic sections | Direct to structure extraction |
| HTML (generic) | Missing SemantiK markers | WCAG enhancement → SemantiK-style formatting |
| Text/Markdown | `.txt`, `.md` extension | Convert to semantic HTML |
| Exam Objectives | PDF with objective patterns | Exam-research agent (existing workflow) |

## Workspace Structure

```
agent_workspaces/textbook_ingestor_workspace/
├── inputs/
│   └── [original_files]
├── semantik_processing/
│   └── [semantik_output_files]
├── structure_extraction/
│   └── [textbook_structure.json]
├── processing_log.json
└── ingestor_scratchpad.md
```

## Processing Pipeline

### Phase 1: Input Detection

```python
def detect_input_type(input_path: str) -> InputType:
    """
    Detect the type of input material.

    Detection Order:
    1. File extension analysis
    2. Content inspection (for ambiguous cases)
    3. Semantic marker detection
    """
    extension = Path(input_path).suffix.lower()

    if extension == '.pdf':
        # Check if exam objectives or textbook
        if contains_exam_objective_patterns(input_path):
            return InputType.EXAM_OBJECTIVES
        return InputType.PDF_TEXTBOOK

    elif extension == '.imscc':
        return InputType.IMSCC_PACKAGE

    elif extension in ['.html', '.htm']:
        if is_semantik_formatted(input_path):
            return InputType.SEMANTIK_HTML
        return InputType.GENERIC_HTML

    elif extension in ['.txt', '.md']:
        return InputType.TEXT

    else:
        raise UnsupportedInputType(f"Cannot process {extension} files")
```

### Phase 2: Format-Specific Processing

#### PDF Textbook Processing
```python
def process_pdf_textbook(pdf_path: str, workspace: str) -> ProcessingResult:
    """
    Convert a PDF textbook to accessible HTML via the SemantiK v2 cascade.

    SemantiK is the sole conversion engine. The single chokepoint is
    MCP/tools/pipeline_tools.py::_run_semantik_v2_conversion, which runs
    SemantiK.semantik_structure.cascade.run_pipeline_v2(pdf) -> build_chapters_ir
    -> normalize_cascade_to_ed4all, writes {stem}_accessible.html + two
    sidecars, and emits the data-semantik-* / semantik:{slug}#{block_id}
    source-provenance wire contract that downstream stages consume.
    """
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    stem = Path(pdf_path).stem
    output_path = Path(workspace) / 'semantik_processing' / f'{stem}_accessible.html'

    # Run the SemantiK v2 cascade (lazy-imports SemantiK's heavy runtime deps;
    # returns the Ed4All tool JSON contract — success / html_path / html_paths).
    result = _run_semantik_v2_conversion(
        pdf_path=str(pdf_path),
        output_path=str(output_path),
    )

    if result.get('success'):
        return ProcessingResult(
            status='success',
            output_path=result['html_path'],
            format='semantik_html'  # accessible HTML carrying the data-cf-*/data-semantik-* contract
        )
    else:
        return ProcessingResult(
            status='failed',
            error=result.get('error', 'SemantiK conversion failed')
        )
```

#### IMSCC Package Processing
```python
def process_imscc_package(imscc_path: str, workspace: str) -> ProcessingResult:
    """
    Extract and process IMSCC package.

    Uses: imscc-intake-parser agent (existing)
    Then: SemantiK enhancement for non-accessible HTML
    """
    # Invoke IMSCC intake parser
    extraction_result = invoke_imscc_intake_parser(imscc_path, workspace)

    # Get HTML content files
    html_files = extraction_result['content_inventory']['html_files']

    # Check each file for SemantiK compliance
    for html_file in html_files:
        if not is_semantik_formatted(html_file):
            # Enhance with WCAG compliance
            enhance_to_semantik_format(html_file)

    return ProcessingResult(
        status='success',
        output_paths=html_files,
        format='semantik_html',
        source_lms=extraction_result['detected_lms']
    )
```

#### Generic HTML Processing
```python
def process_generic_html(html_path: str, workspace: str) -> ProcessingResult:
    """
    Enhance generic HTML to SemantiK format.

    Adds:
    - Skip links
    - ARIA landmarks
    - Semantic section wrappers
    - Heading hierarchy normalization
    """
    with open(html_path, 'r') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Add skip link
    add_skip_link(soup)

    # Add ARIA landmarks
    wrap_main_content(soup)
    add_role_attributes(soup)

    # Normalize heading hierarchy
    normalize_headings(soup)

    # Wrap sections
    wrap_sections_by_heading(soup)

    # Write enhanced HTML
    output_path = workspace / 'semantik_processing' / f'{Path(html_path).stem}_enhanced.html'
    with open(output_path, 'w') as f:
        f.write(str(soup))

    return ProcessingResult(
        status='success',
        output_path=output_path,
        format='semantik_html'
    )
```

### Phase 3: Structure Extraction

After format-specific processing, invoke the semantic-structure-extractor:

```python
def extract_structure(processed_html_path: str, workspace: str) -> Dict:
    """
    Extract semantic structure from SemantiK-formatted HTML.

    Uses: scripts/semantic-structure-extractor/
    """
    from semantic_structure_extractor import extract_textbook_structure

    structure = extract_textbook_structure(
        html_path=processed_html_path,
        config_path='scripts/semantic-structure-extractor/config/extractor_config.json'
    )

    # Save structure to workspace
    output_path = workspace / 'structure_extraction' / 'textbook_structure.json'
    with open(output_path, 'w') as f:
        json.dump(structure, f, indent=2)

    return structure
```

## Output Format

The agent produces a textbook structure JSON conforming to:
`schemas/academic/textbook_structure.schema.json`

### Output JSON Structure
```json
{
  "documentInfo": {
    "title": "Document title",
    "sourcePath": "<SOURCE_PATH>",
    "sourceFormat": "<enum per schemas/academic/textbook_structure.schema.json>",
    "extractionTimestamp": "ISO timestamp",
    "metadata": {
      "authors": [],
      "description": "",
      "keywords": [],
      "language": "en"
    }
  },
  "tableOfContents": [
    {
      "level": 2,
      "text": "Chapter 1: Introduction",
      "id": "intro",
      "children": []
    }
  ],
  "chapters": [
    {
      "id": "ch1",
      "headingText": "Chapter 1",
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

## Integration with Pipeline

### Upstream Integration
- Receives raw source materials from orchestrator
- Accepts files from `inputs/textbooks/`, `inputs/existing-packages/`, or direct paths

### Downstream Integration
- Output feeds into `objective-synthesizer` agent
- Structure JSON compatible with `course-outliner` integration

### Pipeline Flow
```
textbook-ingestor → objective-synthesizer → course-outliner
       ↓                    ↓                     ↓
 textbook_structure.json  learning_objectives.json  course_structure.md
```

## Error Handling

### Common Errors and Recovery

| Error | Cause | Recovery |
|-------|-------|----------|
| `UnsupportedInputType` | Unknown file extension | Return error, suggest conversion |
| `SemantiKConversionFailed` | PDF extraction error | Retry with OCR fallback |
| `IMSCCExtractionFailed` | Corrupt package | Attempt partial extraction |
| `HTMLParsingError` | Malformed HTML | Use lenient parser, clean markup |

### Error Logging
```python
def log_processing_error(error: Exception, context: Dict):
    """Log processing errors with full context."""
    error_log = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "input_path": context.get("input_path"),
        "processing_stage": context.get("stage"),
        "recovery_attempted": context.get("recovery_attempted", False)
    }

    with open(workspace / 'processing_log.json', 'a') as f:
        f.write(json.dumps(error_log) + '\n')
```

## Quality Validation

### SemantiK Format Validation
```python
def is_semantik_formatted(html_path: str) -> bool:
    """Check if HTML has SemantiK formatting markers."""
    with open(html_path, 'r') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Check for SemantiK markers
    has_skip_link = soup.find('a', class_='skip-link') is not None
    has_main_role = soup.find('main', attrs={'role': 'main'}) is not None
    has_sections = len(soup.find_all('section', attrs={'aria-labelledby': True})) > 0

    return has_skip_link and has_main_role and has_sections
```

### Structure Completeness Validation
```python
def validate_extraction(structure: Dict) -> ValidationResult:
    """Validate extracted structure completeness."""
    issues = []

    if not structure.get('chapters'):
        issues.append("No chapters extracted")

    for chapter in structure.get('chapters', []):
        if not chapter.get('sections'):
            issues.append(f"Chapter {chapter.get('id')} has no sections")

        if not chapter.get('contentBlocks'):
            issues.append(f"Chapter {chapter.get('id')} has no content")

    return ValidationResult(
        valid=len(issues) == 0,
        issues=issues
    )
```

## Usage Examples

### Single Textbook Processing
```python
# Invoke for PDF textbook
result = invoke_textbook_ingestor(
    input_path="/inputs/textbooks/networking_fundamentals.pdf",
    workspace="/exports/20251208_120000_networking/agent_workspaces/"
)
```

### IMSCC Package Processing
```python
# Invoke for existing course package
result = invoke_textbook_ingestor(
    input_path="/inputs/existing-packages/canvas_course.imscc",
    workspace="/exports/20251208_120000_canvas_remediation/agent_workspaces/"
)
```

### Multiple Input Processing
```python
# Process multiple sources
sources = [
    "/inputs/textbooks/chapter1.pdf",
    "/inputs/textbooks/chapter2.pdf",
    "/inputs/textbooks/chapter3.pdf"
]

results = []
for source in sources:
    result = invoke_textbook_ingestor(
        input_path=source,
        workspace=workspace
    )
    results.append(result)

# Merge structures if needed
merged_structure = merge_textbook_structures(results)
```

## Success Criteria

- **Input detection**: supported formats are identified; unknown inputs fail loudly
- **SemantiK conversion**: inspect conversion status and validation evidence
- **Structure Extraction**: All chapters and sections identified
- **Content Completeness**: All definitions, terms, procedures extracted
- **Quality Validation**: All extracted structures pass schema validation
