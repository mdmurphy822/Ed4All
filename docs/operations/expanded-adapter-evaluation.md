# Expanded adapter evaluation

The final adapter evaluation uses immutable, disjoint splits. Checkpoint
selection is a development activity; it must not inspect the final test sets.

`Trainforge.eval.expanded_suite` builds the suite offline from the verified
assessment bank, answer key, objectives, and canonical IMSCC chunks. It does
not call an LLM or mutate workflow state.

The allocator gives the objective-complete held-out split first priority,
selecting one source-grounded item for every canonical terminal or component
objective. It then fills the checkpoint-development, grounding-stress,
pedagogy/misconception, and out-of-domain splits without reusing a fingerprint.
If the verified source pool is too small, the manifest records the exact
deficit and `ready` remains false. It never fills a quota by copying an item
between splits.

Every item carries its canonical objective and Bloom level, answer and
keypoints, chunk and citation anchors, content type, difficulty, provenance,
retrieval expectations, split, and a canonical fingerprint. The suite
manifest pins all input hashes and the ordered assessment-chunk exclusion
list. Those assessment chunks must be excluded from synthesis before the first
provider call; reserving only the emitted question text is insufficient.
Final-source exclusion is family-closed using the approved
`cf_block_id_or_contiguous_follows_v1` contract: chunks share a family only
when they carry the same explicit `data-cf-block-id`, or when a direct
`follows_chunk` edge has the same item path and a contiguous character
boundary (delta zero or one). Aggregate source-reference lists, whole item
paths, and generic follows links are deliberately not family keys. The
registry records every qualifying contiguous edge and hashes that evidence.

Final-set contamination checks cover exact and normalized text, fuzzy
similarity, semantic similarity, source-chunk identity, and answer/keypoint
containment. Semantic validation is performed with the production embedding
model after candidate authoring and again over final emitted SFT/DPO records.
Any finding quarantines the training pair or evaluation candidate; thresholds
must not be weakened to obtain a pass.

The runner projects each non-development item across the same six arms:

- base model without retrieval
- base model with retrieval
- SFT adapter without retrieval
- SFT adapter with retrieval
- DPO adapter without retrieval
- DPO adapter with retrieval

All arms share the same item fingerprint. This makes differences attributable
to retrieval and adapter stage rather than question sampling.

Example:

```bash
python -m Trainforge.eval.expanded_suite \
  --assessments <ASSESSMENTS_PATH> \
  --answer-key <ANSWER_KEY_PATH> \
  --objectives <OBJECTIVES_PATH> \
  --chunks <CHUNKS_PATH> \
  --authored-items <AUTHORED_ITEMS_PATH> \
  --ood-probes <OOD_PROBES_PATH> \
  --training-pairs <INSTRUCTION_PAIRS_PATH> \
  --training-pairs <PREFERENCE_PAIRS_PATH> \
  --output-dir <EVAL_OUTPUT_DIR>
```

Exit code `0` means the offline suite is complete and clean. Exit code `2`
means the report was written but at least one quota, objective, or
contamination requirement remains unresolved.

Authored deficit items are accepted only into their preassigned development,
grounding-stress, or pedagogy split and only after the independent structured
review marks every quality dimension true. OOD cases are imported only from a
foreign licensed course's operator-reviewed, verified refusal probes; they
cannot be fabricated from the target course. Neither input expands the frozen
target-course training exclusion set.

The final-set authoring queue rejects assessment-item chunks. Every queue also
rejects aggregate overview chunks with more than five objective references and
passages whose content-bearing body has insufficient lexical support for the
selected canonical objective. It records a source deficit instead of reusing a
held-out assessment or silently assigning the first objective ID.

The frozen source-reuse policy is `dev_final_disjoint_family_max2_v1`.
Development authoring may use only assessment surfaces assigned exclusively to
the development split. Final grounding and pedagogy authoring may use only
clean non-assessment families reserved for final evaluation. No family may
cross development and final, and all remain inside the frozen training
exclusion union. A final family contributes at most two items; a second item
must use a different objective, task, and failure mode. Normalized, fuzzy, and
semantic cross-split deduplication remains mandatory. The registry records the
exact dev/final family assignments, the unchanged exclusion hash, and a policy
fingerprint.

Automatic review is two bounded structured passes. The grounding pass sees
only source text plus the compact scoring fields; the alignment pass also sees
the full canonical objective statement, Bloom level, and target split. Neither
pass receives provenance fan-out or aggregate source-ID lists. Promotion
requires every grounding, scoreability, citation, objective, Bloom,
independence, and split-fidelity boolean to be exactly true. Raw author output,
both raw review objects, and deterministic aggregation are retained even for a
rejected candidate so an offline replay can reproduce the decision.

Before either model review can promote an item, deterministic verification is
authoritative. It independently rechecks supported fraction, coin, place-value,
numeric, and symbolic-algebra answers with the repository's safe SymPy parser;
an item containing mathematical claims that the verifier cannot establish is
rejected. It also checks question/answer operands and entities against chosen
keypoints, objective and Bloom semantics, source-independent transfer, exact
schema-boundary clipping, markup/control characters, and unexpected scripts.
Model review can never override a deterministic failure. A model false
negative may be recovered only when the independent deterministic transfer
contract is completely verified and recorded in the raw audit.

Author schemas are split-specific: development emits an objective probe;
grounding emits an explicit evidence-conflict, missing-evidence, citation, or
refusal challenge; pedagogy emits a plausible learner error plus correction
rationale. A normal question with unused stress or misconception metadata does
not satisfy either specialized split.
