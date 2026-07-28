# Reject-mined DPO negatives — operator guide

**Flag:** `TRAINFORGE_DPO_MINE_REJECTS` (three-valued: `off` default / `shadow` / `on`)
**Runs in:** the `training_synthesis` phase (`Trainforge/synthesize_training.py::run_synthesis`)
**Emits into:** `LibV2/courses/<slug>/training_specs/preference_pairs.jsonl`
**Costs:** zero extra GPU time, zero extra LLM calls, zero network

> **Read § "The failure mode" before you turn this on.** Published measurement
> (FLAME, arXiv:2405.01525) found that naive DPO on unfiltered negatives
> *reduced* factuality — 44.7 → 42.3 FActScore. This feature exists to be the
> targeted variant, not the naive one, and the thresholds that make it targeted
> are **not calibrated on this corpus**. That is what `shadow` mode is for.

---

## 1. What it does

Trainforge synthesizes SFT instruction pairs from course chunks. Measured
acceptance across three archived cohorts was 0/52, 2/52, 4/52 — about **3.85%**.
Every rejected pair is discarded, including its full completion.

That completion is the natural `rejected` side of a DPO preference pair: same
prompt, same chunk, same objective, same teacher — already paid for in GPU time.
Reject mining recovers it.

Two layers, both behind the one flag:

| Layer | What happens |
|---|---|
| **Capture** | A rejected `claim_support` instruction unit keeps its full pair on the *existing* `.synthesis_pairs_checkpoint.jsonl` row (whose `pair` field was previously always `null` on a rejected row). No new file, no new lock, no new loader. |
| **Selection** | After the generation loop, a deterministic pass pairs each *near-miss* reject against the accepted instruction pair for the **same chunk, same `lo_refs`, same unframed base prompt, same citation contract, same teacher**, and appends the survivors to `preference_pairs.jsonl`. |

Selection is a pure function of already-persisted NLI scores. **No model call, no
embedder, no network** — it is fully reproducible and unit-testable without a
seat, and it is byte-identical under input reordering.

### Why it matters beyond the row count

`micro_preference_eligibility` — the gate that governs normal preference-pair
synthesis — requires a chunk-level `misconceptions[]` array. **No chunk in the
measured corpus carries that field** (0 of 16,269; only 26 of 939 match its
fallback regex). Mining bypasses that gate entirely, so it can produce preference
pairs for chunks that currently cannot produce one at all. That is the real
headline, not the raw count.

---

## 2. The failure mode — read this before flipping to `on`

Most current rejects are **degenerate**, not near-misses. The audit found that on
20 of 26 sampled chunks *every content source was empty*, so the completion
collapsed to a fixed pedagogical tail with a database key spliced into the topic
slot — e.g. "Learners should be able to break this down and explain the parts of
learning outcome co-117", entailment **0.028**. Those are correct rejections.

Degenerate negatives are actively harmful as DPO training signal:

- They are trivially separable from a good completion, so the preference carries
  almost no gradient.
- Worse, if every negative comes from a visibly different distribution, DPO
  learns **distribution discrimination** ("template junk vs real prose") instead
  of **quality discrimination** ("grounded vs subtly unsupported"). The second is
  the behavior you want to teach; the first is not.

That is the FLAME result restated: dumping in all rejects indiscriminately is the
naive version that *lowered* factuality. The valuable negative is a **near-miss** —
fluent, on-topic, plausible prose that drifts just past what the source supports.
That is the error the trained model will actually make.

### The five nets that keep degenerates out

| # | Net | What it removes |
|---|---|---|
| 1 | **Stage admission** — only `claim_support:` rejects are minable | The `promotion:` stage runs earlier, carries **no NLI scores at all**, and its leaf reasons (`placeholder_residue`, `source_free_generation`, `unanswerable_stem`) *are* the degenerate class. Never mined. `lo_refs:` / `objective_delivery:` / `duplicate_prompt` are excluded too — those failed on metadata, not grounding. |
| 2 | **Near-miss score band** | `claim_contradicted_rate == 0.0`, `>= 3` scored claims, `claim_support_rate ∈ [0.50, 1.0)`, worst failing claim's entailment `>= 0.15`. The measured degenerate population sits at rate ≈ 0.0 / entailment 0.028 and is excluded by construction. |
| 3 | **LO-identifier splice (D1)** | Any completion containing an LO-id token — the exact "…learning outcome co-117" signature. Detected with BOTH canonical helpers: `lib/ontology/learning_objectives.validate_lo_id` for the `^[A-Z]{2,}-\d{2,}` shape AND `hierarchy_from_id` for a recognized LO prefix from `schemas/taxonomies/lo_hierarchy.json`. The prefix check is load-bearing: matching the shape alone case-insensitively would also delete `ISO-8601`, `RFC-2119`, `COVID-19` — legitimate near-miss content on a standards / CS / medical corpus. |
| 4 | **Novel-content floor (D2)** | Requires ≥ 25 distinct content tokens the prompt does not already carry. A fixed frame plus the prompt's own topic slot fails. |
| 5 | **Pool skeleton frequency (D3)** | A normalized-completion skeleton (casefold → drop LO-ids → collapse whitespace → sha256) recurring `>= 3` times across the run's pool. **Template collapse *is* repetition** — no per-item threshold can catch this, which is why the miner pools at all. **Digits are deliberately not normalized**: collapsing them made numerically distinct worked answers of a shared prose shape hash identically, which rejected the whole numeric-answer population of a math corpus as "template collapse". |

Plus a minimal-difference discipline (HA-DPO, arXiv:2311.16839): chosen/rejected
token-Jaccard must land in `[0.30, 0.70]`. A negative from a different lexical
universe teaches separability, not quality.

And a hard balance cap (FLAME again): mined rows are capped at
`TRAINFORGE_DPO_MINE_MAX_FRACTION` (default **25%**) of the accepted SFT corpus,
plus one negative per anchor and one per chunk.

### What it will never do

A chunk with rejects but **no accepted positive** emits nothing (counter
`no_anchor`). No `chosen` is ever synthesized by a model call, built from a
deterministic template, extracted verbatim from the source, or borrowed from
another chunk. Each of those manufactures a worse negative than none — a
templated `chosen` against a fluent model-prose `rejected` inverts the trap and
teaches "prefer templates".

---

## 3. When to turn it on

**Always run `shadow` first.** `shadow` executes the entire funnel — pools,
scores, ranks, caps, captures decisions — and emits **nothing**.

Two reasons this is not optional:

1. **The thresholds are uncalibrated.** They bracket exactly one measured value
   (entailment 0.028). The archived cohorts *cannot* calibrate them, because
   rejected rows carried `pair=null`, so the rejected text does not exist on disk
   for any past run. `shadow` is the first time the distribution is measurable.
2. **At the default `--instruction-variants-per-chunk 1`, yield is structurally
   ZERO.** One instruction unit per chunk means a chunk can never hold both an
   accepted and a rejected instruction unit, so no reject can ever find an
   anchor. `run_synthesis` logs a warning at startup when it sees this. You need
   `>= 2` variants for the feature to produce anything at all.

### The gate for flipping to `on`

Read `training_specs/synthesis_summary.json → reject_mining.emitted`.

**Do not enable `on` until that number is `>= 50`.** Below the trainer's
`min_dpo_pairs: 50` floor, `Trainforge/training/runner.py` skips DPO entirely and
the feature is inert — you would take the corpus change and the fingerprint
re-key for no training effect.

### Timing constraint — land at a run boundary

The flag lands in `runtime_policy` and re-keys `synthesis_run_contract_sha256`.
**Flipping it on a paused run raises `FreshStartError` and forces that run to
archive-and-restart from zero.** It is a launch-time decision only, deliberately
not denylisted (the guard is behaving correctly — the flag genuinely changes what
lands in `preference_pairs.jsonl`, so it is not a scheduling knob).

### Invocation

```bash
# 1. Measure. Emits nothing; changes no corpus byte.
export TRAINFORGE_DPO_MINE_REJECTS=shadow
ed4all run textbook-to-course --corpus ./inputs/<DIR> --course-name <NAME>

# 2. Read the funnel.
jq '.reject_mining' LibV2/courses/<slug>/training_specs/synthesis_summary.json

# 3. Only if `emitted >= 50` AND you are at a run boundary:
export TRAINFORGE_DPO_MINE_REJECTS=on
```

Optional threshold overrides (all parse-with-fallback; garbage or out-of-band
silently resolves to the default; all are no-ops when the master flag is `off`):

| Env | Band | Default |
|---|---|---|
| `TRAINFORGE_DPO_MINE_MIN_SUPPORT` | float `[0.0, 1.0)` | `0.50` |
| `TRAINFORGE_DPO_MINE_MIN_FAIL_ENTAILMENT` | float `[0.0, 1.0)` | `0.15` |
| `TRAINFORGE_DPO_MINE_MAX_SKELETON_FREQ` | positive int | `3` |
| `TRAINFORGE_DPO_MINE_MAX_FRACTION` | float `(0.0, 1.0]` | `0.25` |

---

## 4. What it emits

### The funnel — `synthesis_summary.json → reject_mining`

Two groups in one flat object: five **capture-pool** counters plus the nineteen
**selection-funnel** counters, and `mode_shadow` (`1` in shadow, `0` in `on`).

Capture pool — did anything get captured, and did resume work?

| Counter | Meaning |
|---|---|
| `captured_this_run` | Rejects pooled by this process. |
| `resume_cache_rows_scanned` / `resume_cache_candidates` | Rows read from the loaded pair checkpoint / how many became candidates. Non-zero only on a `--resume`. |
| `superseded_resume_cache` | A live rejection replaced a stale cached row for the same terminal key. |
| `pool_size` | Distinct candidates the selection pass actually saw (the union, de-duplicated by `(chunk_id, "instruction", variant_index)`). |

On a resumed run, `pool_size` should equal what an uninterrupted run would have
produced — that is the resume-completeness property the design buys by riding the
existing checkpoint row instead of a new sidecar.

Selection funnel — every candidate is accounted for by exactly one counter, so it
tells you *which* net is binding rather than just that yield was low:

```
candidates_seen  →  stale_contract  stage_excluded  deps_missing
                    too_few_claims  contradicted
                    below_support_floor  below_fail_entailment
                    lo_id_splice  low_novel_content  skeleton_collapse
                    no_anchor  identity_mismatch
                    jaccard_out_of_band  length_out_of_band
                    unregistered_teacher  duplicate_prompt
                    per_chunk_capped  global_capped
                                                    →  emitted
```

Reading it:

| You see | It means |
|---|---|
| `candidates_seen: 0` | Nothing was captured. Either the flag was `off` during generation, or no unit reached the `claim_support` stage. |
| `no_anchor` dominates | The chunks that produced rejects produced no *accepted* pair. Almost always `--instruction-variants-per-chunk 1`. Raise it. |
| `lo_id_splice` / `skeleton_collapse` dominate | Your rejects are the degenerate template-collapse population. The nets are working; there is no near-miss signal here to mine. Fix the upstream generator instead (§6). |
| `identity_mismatch` dominates | Variants exist but land on different focused objectives or different base prompts, so no reject shares an anchor. |
| `global_capped` non-zero | You hit the 25% balance ceiling. This is the FLAME guard, not a bug. |
| `stale_contract` non-zero after a `--resume` | Candidates inherited from the prior run whose per-seed generation-contract fingerprint this run did not re-confirm — i.e. the unit was regenerated under a changed contract, or its chunk was never revisited. They are refused rather than paired against a current-contract `chosen`. Expected after a model / generation-parameter / generation-contract-file change; unexpected otherwise. |
| `reject_mining_skipped: "incomplete_pass"` | The generation pass stopped early or exhausted its dispatch budget, so mining was skipped entirely. A partial pool is a *biased* pool (rejects only from the front of the corpus), so it is refused rather than mined. |

### The pairs — `preference_pairs.jsonl`

A mined row is a normal preference pair (it satisfies
`schemas/knowledge/preference_pair.schema.json` and every gate that reads it),
plus two additive provenance fields.

---

## 5. Telling a mined pair from a synthesized one

Both pair schemas are `additionalProperties: true`, so provenance rides along
without a schema edit.

**The one-field test:**

```bash
jq -c 'select(.source == "mined_rejection")' \
  LibV2/courses/<slug>/training_specs/preference_pairs.jsonl
```

**Counts by origin:**

```bash
jq -r '.source // .rejected_source // "synthesized"' \
  LibV2/courses/<slug>/training_specs/preference_pairs.jsonl | sort | uniq -c
```

| Signal | Mined row | Normal synthesized row |
|---|---|---|
| `source` | `"mined_rejection"` | absent, or `"misconception"` / `"misconception_editorial"` |
| `reject_mining` block | present | absent |
| `misconception_id` | `null` | set on misconception-derived rows |
| `rejected_source` | **deliberately absent** — its schema enum is `{misconception, rule_synthesized}` and this row is neither | set where applicable |
| `rejected` side origin | a real model completion that failed the `claim_support` gate | synthesized or editorial |
| `chosen` side origin | the accepted completion for the same chunk + objective, verbatim | synthesized |

The `reject_mining` block carries the full audit trail per row:

```json
{
  "schema_version": "v1",
  "rejection_reason": "claim_support:<leaf>",
  "rejected_variant_index": 2,
  "anchor_variant_index": 0,
  "claim_support_rate": 0.67,
  "claim_contradicted_rate": 0.0,
  "min_failing_entailment": 0.42,
  "n_claims": 3,
  "completion_jaccard": 0.51,
  "skeleton_frequency": 1,
  "rank": 0,
  "contract_fingerprint": "<the reject's fingerprint>"
}
```

Every emitted row also fires a `reject_mined_preference_selection` decision-capture
event whose rationale interpolates the chunk id, support rate, min failing
entailment, anchor variant, and rank — so a post-hoc audit can replay *why* each
negative was chosen, plus one pass-summary event carrying the whole funnel.

### Two honest caveats about provenance

- **`promotion_status: "validated"` overclaims.** The `chosen` side genuinely is a
  validated accepted completion (it passed promotion, claim_support, lo_refs and
  objective_delivery). The row *as a preference pair* was never put through a
  preference promotion validator. `"candidate"` is more honest but is not a tier
  the DPO path expects; `"trainable"` is worse. This is the least-wrong of three
  imperfect options and is commented as such in the code.
- **Provenance dies at the trainer.** `runner.py` hashes `preference_pairs.jsonl`
  into `preference_pairs_hash` on the model card, so a trained adapter is pinned
  to the exact mined corpus — but `peft_trainer.py` hands TRL only
  `prompt`/`chosen`/`rejected`. An A/B can attribute causality at
  **corpus-hash level only**, not per row.

### Licensing provenance

Each mined row is stamped through the canonical
`lib/licensing/teacher_roster.py::stamp_pair_license` with the teacher that
produced the completion being reused. A row whose teacher classifies as barred,
unregistered, or claude-tagged is **dropped** (counter `unregistered_teacher`)
rather than emitted — `assert_export_licenses` fail-closes the entire training
run on such a teacher, and a derived row must never be the thing that bricks a
multi-hour build.

---

## 6. The thing this feature does not fix

The audit's dominant finding was that **20 of 26 sampled chunks had every content
source empty**. That is the actual cause of the 3.85% acceptance rate.

This feature monetizes that symptom, and by design (§2 nets 1, 3 and 5) it
*filters away* the very population that is the evidence for it. Fixing evidence-
window viability and chunk content gating moves acceptance off 3.85% and makes
reject mining largely unnecessary.

**If there is capacity for exactly one of the two, do that one.**

---

## 7. Related

- Flag rows (master + four satellites): `Trainforge/CLAUDE.md § Opt-In Behavior Flags`
- Selection implementation: `Trainforge/synthesis_reject_mining.py`
- Pipeline hook: `Trainforge/synthesize_training.py::run_synthesis` (after the
  misconception-DPO block, before the record sort / gold-set decontamination /
  JSONL write)
- DPO admission predicate (one function, both call sites):
  `Trainforge/training/compute_backend.py::is_dpo_editorial_record`
- Regression nets: `Trainforge/tests/test_reject_mining_capture.py`,
  `test_reject_mining_checkpoint.py`, `test_reject_mining_selection.py`,
  `test_training_dpo_filter.py`, `test_h5_synthesis_source_coverage.py`
- Licensing posture: `docs/LICENSING.md § SFT teacher roster` (no synthesis-provider
  row — this feature selects no provider, model, or backend)
