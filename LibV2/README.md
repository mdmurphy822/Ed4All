# LibV2

**A searchable archive of pedagogically structured course content.**

LibV2 is the final stage of the Ed4All pipeline and the long-term home for everything it produces. Each archived course carries chunked content, a concept graph, learning outcomes, pedagogy metadata, quality reports, and the original source artefacts, classified under a division → domain → subdomain → topic hierarchy spanning STEM and Arts. Courses are retrieved with a BM25 + character-n-gram engine that supports metadata filters (concept tags, learning objectives, Bloom's levels, teaching role, content type, week) and returns a structured rationale explaining why each result was ranked where it was — a reference retrieval implementation, not a production vector store, and intentionally bounded to stay easy to understand and audit.

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

See [`LibV2/CLAUDE.md`](CLAUDE.md) for the storage model, classification taxonomy, retrieval API, and import/validation workflows. Query-based retrieval is the only supported access pattern — never read `chunks.json` files directly.

## License

MIT
