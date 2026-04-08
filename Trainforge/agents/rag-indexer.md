# RAG Indexer Agent

## Purpose

Generates embeddings and indexes content chunks for retrieval-augmented generation.

## Responsibilities

1. **Embedding Generation**: Create vector embeddings for content chunks
2. **Vector Indexing**: Index embeddings for efficient retrieval
3. **Metadata Management**: Maintain chunk metadata for filtering

## Inputs

- Content chunks from assessment-extractor
- Indexing configuration

## Outputs

- Vector index stored in LibV2
- Index statistics and health report

## Decision Points

- Select embedding strategy for content type
- Determine indexing parameters
- Decide on metadata to include in index

## Integration

Works with:
- assessment-extractor agent (receives content chunks)
- LibV2 storage system (stores index)
