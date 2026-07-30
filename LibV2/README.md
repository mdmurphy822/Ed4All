# LibV2

**A searchable archive of pedagogically structured course content.**

LibV2 is the final stage of the Ed4All pipeline and the long-term home for everything it produces. Each archived course carries chunked content, a concept graph, learning outcomes, pedagogy metadata, quality reports, and the original source artefacts, classified under a division → domain → subdomain → topic hierarchy spanning STEM and Arts. Courses are retrieved through three engines: `lexical` (BM25 + character n-grams), `semantic` (exact cosine search over a per-course on-device vector index), and `hybrid-rrf`, which fuses the two with reciprocal rank fusion — the benchmark-selected default, since pure semantic never beat the BM25 baseline. All three support metadata filters (concept tags, learning objectives, Bloom's levels, teaching role, content type, week) and return a structured rationale explaining why each result was ranked where it was. The vector index is pure-numpy and local, with no external vector service, so the whole retrieval path stays easy to understand and audit.

## Quick example

```bash
# Courses land here automatically at the end of `ed4all run textbook-to-course`.
# Retrieve content from the archive:
python -m LibV2.tools.libv2.cli retrieve "your query" --limit 10

# Filter by domain and chunk type:
python -m LibV2.tools.libv2.cli retrieve "your query" \
  --domain computer-science --chunk-type example --limit 10

# Browse the catalog without loading any chunk content:
python -m LibV2.tools.libv2.cli catalog list --division STEM
```

## More

See [`LibV2/CLAUDE.md`](CLAUDE.md) for the storage model, classification taxonomy, retrieval API, and import/validation workflows. Query-based retrieval is the only supported access pattern — never read `chunks.jsonl` files directly.

## License

MIT
