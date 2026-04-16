# Ed4All Examples

## Quick Start: Generate a Course

### 1. Generate HTML modules
```bash
python Courseforge/scripts/generate_course.py examples/sample_course_data.json examples/output/
```

### 2. Package as IMSCC
```bash
python Courseforge/scripts/package_multifile_imscc.py examples/output/ examples/output/EXAMPLE_101.imscc
```

### 3. Process through Trainforge
```bash
python -m Trainforge.process_course \
  --imscc examples/output/EXAMPLE_101.imscc \
  --course-code EXAMPLE_101 \
  --objectives examples/sample_objectives.json \
  --division STEM --domain computer-science \
  --output examples/output/trainforge_output
```

## What to Expect
- Step 1 produces ~12 HTML files across 2 weeks
- Step 2 creates an IMSCC package importable into any LMS
- Step 3 creates a chunked corpus with Bloom's metadata, concept graphs, and quality reports
