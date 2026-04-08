# Assessment Extractor Agent

## Purpose

Extracts learning objectives and content chunks from IMSCC packages for assessment generation.

## Responsibilities

1. **IMSCC Parsing**: Extract content from IMSCC package format
2. **Learning Objective Extraction**: Identify and extract learning objectives
3. **Content Chunking**: Break content into semantic chunks for RAG indexing

## Inputs

- IMSCC package path
- Course metadata

## Outputs

- Extracted learning objectives with metadata
- Content chunks with source tracking
- Extraction report

## Decision Points

- Determine chunk boundaries and types
- Identify learning objective patterns
- Decide on metadata extraction strategy

## Integration

Works with:
- rag-indexer agent (sends content chunks)
- assessment-generator agent (sends learning objectives)
