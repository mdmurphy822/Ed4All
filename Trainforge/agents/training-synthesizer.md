# Training Synthesizer Agent

## Purpose

Paraphrase mock-drafted instruction or preference pairs into LLM-quality
training data without altering load-bearing metadata. Used by
`Trainforge/generators/_claude_session_provider.py` when
`synthesize_training.py --provider claude_session` is selected.

## Input

```json
{
  "kind": "instruction",
  "chunk_id": "rdf_shacl_551_chunk_00054",
  "chunk_text": "RDFS allows authors to declare classes and properties...",
  "draft": {
    "prompt": "Original mock-factory prompt",
    "completion": "Original mock-factory completion",
    "template_id": "understand.explanation",
    "bloom_level": "understand",
    "content_type": "explanation",
    "lo_refs": ["TO-01"],
    "chunk_id": "rdf_shacl_551_chunk_00054"
  },
  "expected_keys": ["prompt", "completion"]
}
```

For `kind: "preference"`, `draft` carries `prompt`, `chosen`, `rejected`,
and `expected_keys` is `["prompt", "chosen", "rejected"]`.

## Output

Return ONLY a JSON object whose keys match `expected_keys` exactly:

```json
{"prompt": "...", "completion": "..."}
```

or for preference:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Do not return prose, markdown, code fences, or explanation. The dispatcher
parses the response as raw JSON.

## Workflow

1. Read `chunk_text` carefully — it is the only source-of-truth.
2. Rewrite `draft.prompt` so it preserves the `template_id`'s pedagogical
   intent (e.g. an `understand.explanation` template stays an explain-it
   prompt; an `apply.example` stays an apply-an-example prompt).
3. Rewrite `draft.completion` (or `chosen`/`rejected` for preference) so
   it is grounded ONLY in `chunk_text`. Do not introduce facts absent
   from the chunk. Boilerplate phrases like
   "Learners should be able to use this in a new but similar situation"
   MUST be removed — they are mock-factory scaffolding.
4. For `kind: "instruction"` whose `draft.requires_source_citation` is
   `true`, the rewritten `completion` MUST end with `[<chunk_id>]` exactly
   (e.g. `[rdf_shacl_551_chunk_00054]`).
5. Keep `prompt` length 20–800 chars; keep `completion` / `chosen` /
   `rejected` length 60–1500 chars. The provider re-clamps but rejecting
   the draft outright by returning extreme lengths wastes a dispatch.
6. Return the JSON object. Nothing else.

## Quality bar

- Faithful: every claim traceable to `chunk_text`.
- Specific: avoid generic pedagogy meta-talk; commit to the domain content.
- Stable: the `chosen` should be technically correct and well-formed; the
  `rejected` (preference only) should be plausible-but-wrong, ideally
  echoing a known misconception when `draft.misconception_id` is set.
