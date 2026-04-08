# Textbook Stager Agent

## Purpose

Stage DART-processed HTML outputs to Courseforge input directory for course generation. This agent bridges the DART and Courseforge components in the textbook-to-course pipeline.

## Responsibilities

1. **Detect DART Output Files**
   - Locate synthesized HTML files (`*_synthesized.html`)
   - Locate accompanying JSON metadata (`*_synthesized.json`)
   - Handle batch outputs from DART's `batch_output/` directory

2. **Validate DART Markers**
   - Verify skip-link presence (`<a class="skip-link">`)
   - Verify main content landmark (`<main role="main">`)
   - Verify semantic sections (`<section aria-labelledby="...">`)

3. **Stage Files**
   - Copy files to `Courseforge/inputs/textbooks/{run_id}/`
   - Preserve file metadata and timestamps
   - Create staging manifest

4. **Create Manifest**
   - Generate `staging_manifest.json` with file inventory
   - Record staging timestamp
   - Include validation results

## Input

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `run_id` | string | Yes | Pipeline run identifier |
| `dart_output_dir` | string | Yes | Directory containing DART outputs |
| `course_name` | string | Yes | Course identifier |

## Output

| Field | Type | Description |
|-------|------|-------------|
| `staging_dir` | string | Path to staging directory |
| `staged_files` | array | List of staged file paths |
| `manifest` | object | JSON manifest of staged content |
| `validation_results` | object | DART marker validation per file |

## MCP Tool Mapping

This agent maps to the `stage_dart_outputs` MCP tool in `MCP/tools/pipeline_tools.py`.

## Validation Requirements

Before staging, each HTML file MUST pass DART marker validation:

```
Required Markers:
- <a class="skip-link"> or <a class='skip-link'>  (Skip navigation)
- <main role="main"> or <main role='main'>        (Main content landmark)
- <section aria-labelledby="...">                  (Semantic sections)
```

Files failing validation will be logged but still staged (with warnings).

## Decision Capture

Log all staging decisions to:
```
training-captures/textbook-pipeline/{course_name}/phase_staging/decisions_{run_id}.jsonl
```

### Required Decision Events

| Event Type | When | Required Fields |
|------------|------|-----------------|
| `file_discovery` | When locating DART outputs | `files_found`, `directory_scanned` |
| `validation_result` | After validating each file | `file_path`, `markers_present`, `markers_missing` |
| `staging_decision` | When copying files | `source_path`, `dest_path`, `include_reason` |
| `manifest_creation` | After staging complete | `manifest_path`, `file_count` |

## Error Handling

| Error | Action |
|-------|--------|
| No DART outputs found | Fail with clear error message |
| Validation fails | Stage with warning, log to manifest |
| Copy fails | Retry once, then fail task |
| Permission denied | Fail with path information |

## Example Staging Manifest

```json
{
  "run_id": "TTC_PHYS_101_20250214_120000",
  "course_name": "PHYS_101",
  "staged_at": "2025-02-14T12:00:30Z",
  "staged_files": [
    "Courseforge/inputs/textbooks/TTC_PHYS_101_20250214_120000/chapter1_synthesized.html",
    "Courseforge/inputs/textbooks/TTC_PHYS_101_20250214_120000/chapter1_synthesized.json",
    "Courseforge/inputs/textbooks/TTC_PHYS_101_20250214_120000/chapter2_synthesized.html",
    "Courseforge/inputs/textbooks/TTC_PHYS_101_20250214_120000/chapter2_synthesized.json"
  ],
  "validation": {
    "chapter1_synthesized.html": {
      "valid": true,
      "markers": {"skip_link": true, "main_role": true, "aria_sections": true}
    },
    "chapter2_synthesized.html": {
      "valid": true,
      "markers": {"skip_link": true, "main_role": true, "aria_sections": true}
    }
  },
  "errors": null
}
```

## Integration Points

### Upstream: DART
- Reads from: `DART/batch_output/html/` and `DART/batch_output/synthesized/`
- Expects: WCAG 2.2 AA compliant HTML with DART semantic markers

### Downstream: Courseforge
- Writes to: `Courseforge/inputs/textbooks/{run_id}/`
- Produces: Staged files ready for `textbook-ingestor` agent
