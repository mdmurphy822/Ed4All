# LibV2 - SLM Model Graph Repository

A large-scale repository for Small Language Model (SLM) training data, organizing processed educational content with semantic categorization across STEM and Arts domains.

## Overview

LibV2 stores and organizes SLM model graphs produced by [TrainForge](../TrainForge). Each entry contains:
- **Corpus**: Chunked pedagogical content (explanations, examples, exercises)
- **Knowledge Graph**: Concept relationships and prerequisites
- **Pedagogy Model**: Teaching patterns and sequences
- **Training Specs**: ML training configurations

## Repository Structure

```
LibV2/
├── courses/                     # Flat course storage
│   └── [course-slug]/           # One directory per course
├── catalog/                     # Derived indexes
│   ├── master_catalog.json      # All courses with metadata
│   ├── by_division/             # STEM.json, ARTS.json
│   ├── by_domain/               # physics.json, etc.
│   └── cross_references/        # Shared concepts
├── ontology/                    # Classification systems
│   ├── taxonomy.json            # STEM/Arts hierarchy
│   ├── acm_ccs/                 # ACM Computing Classification
│   └── lcsh/                    # Library of Congress headings
├── schema/                      # JSON Schema definitions
├── docs/                        # Documentation
└── tools/                       # Management CLI
```

## Classification System

### Divisions
- **STEM**: Science, Technology, Engineering, Mathematics
- **ARTS**: Arts & Humanities

### STEM Domains
- Physics, Chemistry, Biology, Mathematics
- Computer Science, Engineering
- Medicine, Environmental Science, Data Science

### Hierarchy
`Division → Domain → Subdomain → Topic → Subtopic`

## Quick Start

### Install CLI Tools
```bash
cd tools
pip install -e .
```

### Import a Course
```bash
libv2 import /path/to/trainforge/output/course_name --domain physics --subdomain mechanics
```

### Query the Catalog
```bash
libv2 catalog list --division STEM
libv2 catalog search --domain computer-science --difficulty intermediate
```

### Validate Repository
```bash
libv2 validate --all
```

### Rebuild Indexes
```bash
libv2 index rebuild
```

### Retrieve Content
```bash
# Simple query
libv2 retrieve "learning objectives" --limit 10

# With filters
libv2 retrieve "Python functions" --domain computer-science --chunk-type example

# Multi-query with decomposition (for complex queries)
libv2 multi-retrieve "compare formative and summative assessment" --explain
libv2 multi-retrieve "how does scaffolding improve learning" --no-decompose
```

## Course Structure

Each course directory contains:
```
courses/[slug]/
├── manifest.json        # Extended metadata
├── corpus/
│   ├── chunks.json      # Pedagogical units
│   └── chunks.jsonl     # Streaming format
├── graph/
│   ├── concept_graph.json
│   └── concept_graph.graphml
├── pedagogy/
│   └── pedagogy_model.json
└── training_specs/
    └── dataset_config.json
```

## Cross-Domain Content

Courses spanning multiple domains use:
- `primary_domain`: Main classification
- `secondary_domains`: Additional relevant domains

Example: Bioinformatics
```json
{
  "primary_domain": "biology",
  "secondary_domains": ["computer-science"]
}
```

## License

MIT License - See LICENSE file for details.
