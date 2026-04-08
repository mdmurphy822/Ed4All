# DART Batch Processor

Automated document-to-accessible HTML conversion using DART (Digital Accessibility Remediation Tool).

## Overview

The DART Batch Processor orchestrates parallel conversion of PDF and Office documents to WCAG 2.2 AA compliant accessible HTML. It integrates with the Courseforge remediation pipeline to ensure 100% accessible course content.

## Features

- **Parallel Processing**: Convert multiple documents simultaneously
- **Universal Document Support**: PDF, Word, PowerPoint, Excel
- **Progress Tracking**: Detailed logging and status reporting
- **Error Handling**: Automatic retries with fallback options
- **Pipeline Integration**: Works with remediation queue manifests

## Usage

### From Remediation Queue
```bash
python dart_batch_processor.py \
    --input-manifest /path/to/remediation_queue.json \
    --output-dir /path/to/accessible_content/
```

### Direct File List
```bash
python dart_batch_processor.py \
    --input-files syllabus.pdf lecture1.pptx handout.docx \
    --output-dir /path/to/output/
```

### Auto-Detect from Course Directory
```bash
python dart_batch_processor.py \
    --course-dir /path/to/extracted_course/ \
    --output-dir /path/to/accessible_content/ \
    --auto-detect
```

### With Options
```bash
python dart_batch_processor.py \
    --input-manifest queue.json \
    --output-dir ./output/ \
    --max-workers 6 \
    --json
```

## Programmatic Usage

```python
from dart_batch_processor import DARTBatchProcessor

# Initialize processor
processor = DARTBatchProcessor(
    output_dir=Path('/path/to/output/'),
    max_workers=4
)

# Add documents
processor.add_document(Path('syllabus.pdf'))
processor.add_documents_from_directory(Path('/course/resources/'))
processor.add_documents_from_manifest(Path('remediation_queue.json'))

# Process all
result = processor.process_all()

# Check results
print(f"Successful: {result.successful_conversions}")
print(f"Failed: {result.failed_conversions}")

# Get report
print(processor.generate_report())
```

## Supported Document Types

| Type | Extensions | Conversion Path |
|------|-----------|-----------------|
| **PDF** | `.pdf` | Direct DART conversion |
| **Word** | `.doc`, `.docx`, `.odt`, `.rtf` | LibreOffice → PDF → DART |
| **PowerPoint** | `.ppt`, `.pptx`, `.odp` | LibreOffice → PDF → DART |
| **Excel** | `.xls`, `.xlsx`, `.ods`, `.csv` | LibreOffice → PDF → DART |

## Output Structure

```
output_directory/
├── document1/
│   └── document1.html
├── document2/
│   └── document2.html
└── conversion_report.json
```

## Conversion Report (JSON)

```json
{
  "total_documents": 15,
  "successful_conversions": 14,
  "failed_conversions": 1,
  "total_duration_seconds": 45.3,
  "tasks": [
    {
      "source_path": "/path/to/syllabus.pdf",
      "document_type": "pdf",
      "status": "completed",
      "output_path": "/output/syllabus/syllabus.html",
      "duration_seconds": 3.2
    }
  ]
}
```

## Dependencies

- Python 3.8+
- DART installation (set `DART_PATH` environment variable)
- LibreOffice (for Office document conversion)

## Error Handling

### Retry Logic
- Maximum 2 retries per document
- 5 second delay between retries

### Fallback Behavior
- Office documents without LibreOffice: Creates placeholder HTML with download link
- Timeout after 5 minutes per PDF, 2 minutes for Office conversion

## Integration

This processor integrates with the Courseforge remediation pipeline:

```
imscc-intake-parser → content-analyzer →
    dart-batch-processor → accessibility-remediation →
    brightspace-packager
```

## Performance

- Default 4 parallel workers
- ~3-10 seconds per PDF page (varies by complexity)
- Office documents add ~10-30 seconds for LibreOffice conversion
