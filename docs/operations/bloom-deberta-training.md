# Bloom DeBERTa heads — weight seeding + staged training

Operator runbook for producing the DeBERTa-v3 classifier head(s) that back
the `bloom_classifier_disagreement` gate's optional fourth voter
(`ED4ALL_BLOOM_TRIVOTE_HEADS`, meaningful only alongside
`ED4ALL_BLOOM_TRIVOTE`). `lib/classifiers/training/train_bloom_deberta.py`'s
`--head` flag selects between two trained-artifact shapes
(`lib/classifiers/bloom_deberta_heads.py`'s loader auto-detects whichever is
on disk — see §6):

* **`--head multiclass`** — ONE `num_labels=6` softmax head over all six
  Bloom levels, trained in a single GPU run. **Train this first** — see the
  Decision Protocol immediately below.
* **`--head one-vs-rest`** (the original WI-05 shape, default for
  back-compat) — six SEPARATE per-level binary heads, one GPU run each
  (`--bloom-level <level>`).

Mirrors the staged, never-agent-executed posture of
[`nemotron-lora-canary.md`](nemotron-lora-canary.md): every command below is
run BY HAND, by an operator, on an idle card. No pipeline phase, workflow, or
agent dispatches any of this — `train_bloom_deberta.py` is not registered in
`AGENT_TOOL_MAPPING`, any `@mcp.tool()` surface, or `config/workflows.yaml`.

---

## Decision protocol — train multiclass FIRST

The corpus is heavily imbalanced on the high end of the ladder (a measured
harvest ran roughly 930 / 790 / 590 / 350 / 140 / 80 across
understand..create) — a one-vs-rest `evaluate`/`create` head trains on only
a low-hundreds tail of positives. Before committing to six separate one-vs-rest runs,
train the single multiclass head first: it is ONE GPU run, not six, and it
directly answers whether the six-head ladder is worth the other five runs
at all.

1. Stages §1–§3 below still apply first regardless of which head shape you
   train (pre-seed the base checkpoint, idle-check the card, provision the
   venv) — they are shared infrastructure, not one-vs-rest-specific.
2. Run the staged multiclass command:

   ```bash
   python -m lib.classifiers.training.train_bloom_deberta \
     --labels-path state/bloom_labels/labels.jsonl \
     --head multiclass \
     --base-model models/base/deberta-v3-base \
     --output-dir models/bloom_classifiers/multiclass \
     --seed 42
   ```

3. Read `models/bloom_classifiers/multiclass/final/summary.json` —
   specifically `per_class_metrics.evaluate` and `per_class_metrics.create`
   (the thin tail) alongside `val_metrics.val_f1_macro` (the metric the
   multiclass run selects its best checkpoint on — see §5).
4. **Only run the six one-vs-rest commands (§4) if the multiclass
   per-class F1 is unsatisfactory** for a level you actually need voting
   on. Six extra GPU runs are not free — do not run them reflexively "to
   compare" once the multiclass number already answers the question.
5. **If `evaluate` / `create` F1 is weak under BOTH the multiclass run AND
   the one-vs-rest heads**, that is conclusive: the fix is MORE LABELS,
   not more heads. Neither framing can manufacture positives that were
   never harvested — class-weighted loss (applied unconditionally in both
   modes) cannot substitute for examples that don't exist (§5's thin-class
   caveat applies to both shapes identically). Re-run
   `ed4all harvest-bloom-labels` against more Courseforge exports — or let
   the post-build `ED4ALL_HARVEST_BLOOM_LABELS` hook accumulate labels
   automatically after every ladder-built course — and retrain once
   `class_balance_report`'s `counts["evaluate"]` / `counts["create"]` has
   meaningfully grown.
6. `BloomDebertaHeads.get_or_load()` (`lib/classifiers/bloom_deberta_heads.py`)
   auto-detects whichever artifact exists on disk, checking
   `models/bloom_classifiers/multiclass/final/` FIRST — if you later train
   the six one-vs-rest heads too, the multiclass head still wins (§6).
   Delete `models/bloom_classifiers/multiclass/` to force the loader onto
   the one-vs-rest ladder instead.

This mirrors the "start cheap, escalate only if the cheap path falls
short" posture the rest of this runbook already takes with the light venv
(§3) vs. `scripts/bootstrap-training-env.sh`.

---

Four stages, run in order:

1. One-time weight pre-seed (§1)
2. Idle-check every vLLM seat (§2)
3. A light training venv — NOT `scripts/bootstrap-training-env.sh` (§3)
4. The multiclass command above (recommended first — see Decision Protocol),
   OR the six per-level one-vs-rest training commands + reading
   `summary.json` (§4, §5)

Only after the head(s) you trained exist do you flip
`ED4ALL_BLOOM_TRIVOTE_HEADS` (§6).

---

## §1 — One-time weight pre-seed

The trainer (`lib/classifiers/training/train_bloom_deberta.py`) refuses a bare
HuggingFace hub id under `HF_HUB_OFFLINE` (`validate_base_model_path`) — by
design, this machine stays offline for every regular run ("no models phoning
home"). Seeding the one base checkpoint the six heads all fine-tune from is
therefore a deliberate, narrow, ONE-TIME exception: unset `HF_HUB_OFFLINE` for
a single pinned-revision `snapshot_download`, then immediately restore the
machine-wide offline posture (`docs/operations/run-env.example.sh:155`
precedent — `export HF_HUB_OFFLINE=1` is the steady-state default every other
command in this repo assumes).

```bash
# Resolve the base model's current commit SHA yourself first (e.g. the
# "Files and versions" tab on the microsoft/deberta-v3-base model page, or
# `git ls-remote https://huggingface.co/microsoft/deberta-v3-base`) and pin
# it explicitly below — never a floating "main". This mirrors the 40-hex-SHA
# `revision` pinning lib/classifiers/bloom_bert_ensemble.py already uses for
# every other DeBERTa checkpoint this repo loads.
DEBERTA_BASE_REVISION="<pinned-40-char-commit-sha>"

# Temporarily allow ONE hub reach — nothing else on this machine does.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/deberta-v3-base',
    revision='${DEBERTA_BASE_REVISION}',
    local_dir='models/base/deberta-v3-base',
)
"

# Restore the standing offline posture immediately — do not leave this
# window open for anything else in the same shell.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

`models/base/deberta-v3-base/` (the pre-seeded base snapshot) and
`models/bloom_classifiers/` (the six trained-head outputs, §4) are BOTH
gitignored (`.gitignore`, "Bloom-ladder initiative WI-13") — GPU training
artifacts never ship in-repo. After this step, every subsequent command in
this document runs fully offline; `validate_base_model_path` will accept
`models/base/deberta-v3-base` as-is (it is a local directory) with
`HF_HUB_OFFLINE=1` set.

---

## §2 — Idle-check every vLLM seat first

The trainer is a normal foreground GPU job — it does not go through
`ED4ALL_SEAT_SCHEDULE` or any pipeline-owned seat lifecycle, so nothing stops
a resident seat for you. Verify the card is actually free BEFORE step §4;
sharing the GPU with a live seat is a silent VRAM-contention hazard, not a
loud failure.

```bash
# Every container in the ED4ALL_VLLM_CONTAINERS registry should be absent
# or Exited — none Up.
docker ps --format '{{.Names}}: {{.Status}}'

# Confirm the card itself is idle (same posture the pipeline's own trainers
# enforce: the `training` phase declares `seats: []` in config/workflows.yaml,
# and `assistant_campaign_launch_training` stops every registered seat and
# verifies the card is free before launching).
nvidia-smi
```

If a seat is up, stop it first (`docker stop <container>` — never `rm`, so
the container survives for a later restart) and re-check `nvidia-smi` before
proceeding.

---

## §3 — Training venv (light path, NOT `scripts/bootstrap-training-env.sh`)

`scripts/bootstrap-training-env.sh` exists for the strict, version-banded
Nemotron/TRL/PEFT/Accelerate/Datasets fit (`docs/operations/nemotron-lora-canary.md`)
— overkill here. `train_bloom_deberta.py`'s heavy imports are deferred inside
`main()` precisely so it never needs that band: it imports only `torch`,
`transformers`, `datasets`, and `sklearn` (module docstring). Provision a
plain venv with just those, the same "install the actually-imported packages
directly" posture `SemantiK/CLAUDE.md § In-process install (Option A)` takes
for SemantiK's own classifier trainers — not the repository-managed training
environment.

```bash
python -m venv .venv-bloom-classifiers
. .venv-bloom-classifiers/bin/activate
pip install torch transformers datasets scikit-learn
```

Do **not** reach for `pip install -e '.[training,embedding]'` as a shortcut —
the two extras' `transformers` pins are mutually unsatisfiable
(`[training]` wants `>=4.57,<5`; `[embedding]` wants `>=4.49,<4.50`;
`pyproject.toml`), so that combined install fails to resolve. The plain
four-package venv above sidesteps the conflict entirely because it pins
nothing beyond what `train_bloom_deberta.py` actually imports.

---

## §4 — Six per-level training commands

Per the Decision Protocol above, run these ONLY if the single multiclass run
(also §1–§3 shared infra, then `--head multiclass` as shown above) turned
out unsatisfactory on a level you need. Run once per canonical Bloom level
(`lib.bloom_labels.dataset.BLOOM_LEVELS`), from the repo root, inside the §3
venv, with §2's idle-check still holding (one CUDA context at a time — do
not background a second one while the first runs):

```bash
python -m lib.classifiers.training.train_bloom_deberta \
  --labels-path state/bloom_labels/labels.jsonl \
  --bloom-level remember \
  --base-model models/base/deberta-v3-base \
  --output-dir models/bloom_classifiers/remember \
  --seed 42

python -m lib.classifiers.training.train_bloom_deberta \
  --labels-path state/bloom_labels/labels.jsonl \
  --bloom-level understand \
  --base-model models/base/deberta-v3-base \
  --output-dir models/bloom_classifiers/understand \
  --seed 42

python -m lib.classifiers.training.train_bloom_deberta \
  --labels-path state/bloom_labels/labels.jsonl \
  --bloom-level apply \
  --base-model models/base/deberta-v3-base \
  --output-dir models/bloom_classifiers/apply \
  --seed 42

python -m lib.classifiers.training.train_bloom_deberta \
  --labels-path state/bloom_labels/labels.jsonl \
  --bloom-level analyze \
  --base-model models/base/deberta-v3-base \
  --output-dir models/bloom_classifiers/analyze \
  --seed 42

python -m lib.classifiers.training.train_bloom_deberta \
  --labels-path state/bloom_labels/labels.jsonl \
  --bloom-level evaluate \
  --base-model models/base/deberta-v3-base \
  --output-dir models/bloom_classifiers/evaluate \
  --seed 42

python -m lib.classifiers.training.train_bloom_deberta \
  --labels-path state/bloom_labels/labels.jsonl \
  --bloom-level create \
  --base-model models/base/deberta-v3-base \
  --output-dir models/bloom_classifiers/create \
  --seed 42
```

`state/bloom_labels/labels.jsonl` is `ed4all harvest-bloom-labels`'s default
output store — if it does not exist yet (or is stale against the courses you
want represented), harvest first:

```bash
ed4all harvest-bloom-labels ./Courseforge/exports/<PROJECT-DIR> --dry-run
ed4all harvest-bloom-labels ./Courseforge/exports/<PROJECT-DIR>
```

`--output-dir` above is redundant with the trainer's own default
(`default_output_dir` already resolves `<heads-dir>/<level>` from
`--bloom-level` alone), but is passed explicitly so every command is legible
and reproducible standalone.

**Expected wall-clock / VRAM (single-GPU host, order-of-magnitude, not a
measured benchmark).** `deberta-v3-base` is a ~184M-parameter encoder; a
binary (one-vs-rest) classification head at `--max-length 256` /
`--batch-size 16` / 3 epochs with early stopping is a small fit — expect low
single-digit GiB of VRAM (fp16 on CUDA) and low tens of minutes per level on
a modern single GPU, with the thin tail (`create` / `evaluate`,
low-hundreds of rows) finishing fastest and the dense low-Bloom levels
(`understand` / `apply` / `remember`, high-hundreds each) taking longest. Time the first
(thinnest) level yourself and extrapolate — do not treat the figures above
as a promised number for your corpus or hardware; run label counts vary run
to run as the harvester ingests more courses.

---

## §5 — Reading `summary.json`, the thin-class caveat, and when to re-harvest

Each run writes `<output-dir>/final/summary.json` alongside the saved model +
tokenizer (`lib/classifiers/training/train_bloom_deberta.py`, the same
`<heads_dir>/.../final` shape `BloomDebertaHeads`'s loader expects). The two
`--head` modes write different (but parallel) shapes:

* **`--head multiclass`** (`models/bloom_classifiers/multiclass/final/summary.json`)
  — `head_mode: "multiclass"`, `base_model`, `seed`, `labels` (all six, in
  `BLOOM_LEVELS` order), `label_counts` (train/val per-level counts),
  `class_weights` (per-level), `per_class_metrics` (a classification-report
  dict for EACH of the six levels), and `val_metrics` (`val_accuracy`,
  `val_f1_macro`, …). Read `val_f1_macro` first — the trainer selects its
  best checkpoint on that metric specifically
  (`metric_for_best_model="f1_macro"`), because multiclass has no single
  "positive class" the way a one-vs-rest head has one; `per_class_metrics`
  broken out per level is the same signal at finer grain — check
  `evaluate` / `create` specifically (the thin tail).
* **`--head one-vs-rest`** (`models/bloom_classifiers/<level>/final/summary.json`)
  — `bloom_level`, `base_model`, `seed`, `label_counts` (train/val
  positive/negative), `class_weights`, `per_class_metrics` (`negative` /
  `positive` classification-report dicts), and `val_metrics`
  (`val_accuracy`, `val_f1_macro`, `val_f1_positive`, …). Read
  `val_f1_positive` first — the trainer selects its best checkpoint on that
  metric specifically (`metric_for_best_model="f1_positive"`), NOT
  accuracy, because a 90%-negative one-vs-rest split can satisfy aggregate
  accuracy by predicting "not this level" every time; `per_class_metrics.positive`
  is the same signal broken out by precision/recall.

**Thin-class caveat (applies to BOTH modes).** The corpus is heavily
imbalanced on the high end of the ladder — `class_balance_report`
(`lib/bloom_labels/dataset.py`) flags any level under half the balanced
"fair share" as `thin`, and on a measured harvest that was exactly `create`
and `evaluate` (low-hundreds of rows) against high-hundreds for the
low-Bloom levels. Class-weighted cross-entropy loss is always applied (no opt-out
flag, in either mode) to counteract this, but weighting a loss function
cannot manufacture examples that were never harvested — a `create` or
`evaluate` class trained on a few dozen positives will have a noisier
per-class F1 than the dense levels regardless of head shape, and a
near-zero positive count for either split fails the run outright (`main`
raises `RuntimeError` rather than silently training on zero examples — see
the "no training examples" / "no validation examples" checks in
`train_bloom_deberta.py`, present in both `_train_one_vs_rest` and
`_train_multiclass`).

**When to prefer re-harvesting over training.** If `summary.json` shows a
near-zero `create` / `evaluate` per-class F1 (`val_f1_positive` for
one-vs-rest, `per_class_metrics.create` / `.evaluate` for multiclass — or
the run never got there because `main` raised on an empty split), that is a
data problem, not a hyperparameter one — do not chase it with `--epochs` /
`--lr` tuning, and do not treat switching `--head` modes as a fix either
(see the Decision Protocol's step 5: weak on BOTH means re-harvest). Re-run
`ed4all harvest-bloom-labels` against more Courseforge exports first
(widening the artifact population the harvester walks is the only lever
that grows the thin tail's positive count) and re-train once
`class_balance_report`'s `counts["create"]` / `counts["evaluate"]` has
meaningfully grown. For one-vs-rest specifically, a partial re-train (only
the thin levels re-run) is fine — each level's `summary.json` is
independent and the six checkpoint subdirectories do not need to share a
training run; multiclass always retrains from scratch as one run since
there is only one head.

---

## §6 — Enabling `ED4ALL_BLOOM_TRIVOTE_HEADS`

`BloomDebertaHeads`'s loader (`lib/classifiers/bloom_deberta_heads.py`)
checks `models/bloom_classifiers/multiclass/final/` FIRST; if that single
artifact exists, it loads it and never touches the one-vs-rest ladder at
all. Only when the multiclass artifact is absent does it fall back to the
six-way one-vs-rest ladder, which is itself ALL-OR-NOTHING by design: if
any of the six `<heads-dir>/<level>/final` subdirectories exists but a
sibling is missing, `get_or_load()` returns `None` (the same "no fourth
voter" abstention as a completely empty `ED4ALL_BLOOM_HEADS_DIR`) — a
partial ladder is treated exactly like no ladder, never a silent partial
vote. Practically: do not flip the flag after training two or three
one-vs-rest levels "to see it work" — nothing observable changes until
either the multiclass head or the full six-level set exists, and there is
no partial-credit state to debug.

Checklist before enabling:

1. Either `models/bloom_classifiers/multiclass/final/` exists, OR all six
   `models/bloom_classifiers/<level>/final/` directories exist (or
   whatever `ED4ALL_BLOOM_HEADS_DIR` points at — default
   `models/bloom_classifiers`, `lib/classifiers/bloom_deberta_heads.py::resolve_bloom_heads_dir`).
   Both being present is fine — the loader's auto-detect always prefers
   the multiclass artifact (§ Decision Protocol, step 6).
2. Whichever artifact(s) you trained carry a `summary.json` you have
   actually read (§5) — training completing without a crash is not the
   same as the head being any good.
3. `ED4ALL_BLOOM_TRIVOTE` is ALSO truthy — `ED4ALL_BLOOM_TRIVOTE_HEADS` only
   swaps the voter-2 backend of the trivote re-founding
   (`lib/classifiers/bloom_zero_shot.py`); with `ED4ALL_BLOOM_TRIVOTE`
   unset/off, setting `ED4ALL_BLOOM_TRIVOTE_HEADS` alone is a silent no-op
   (the gate never reaches the code path that reads it).

```bash
export ED4ALL_BLOOM_TRIVOTE=true
export ED4ALL_BLOOM_TRIVOTE_HEADS=true
# Optional — only needed if the heads live somewhere other than the default:
# export ED4ALL_BLOOM_HEADS_DIR=/path/to/heads
```

Both flags default OFF; unset (or falsey / garbage) leaves
`bloom_classifier_disagreement` on its existing behavior byte-identically —
this is a pure opt-in, and flipping it back off at any time (missing weights,
a regression, an operator judgment call) degrades straight back to the
zero-shot backend with no other configuration change required.
