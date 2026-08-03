# Nemotron Nano LoRA qualification

Real SFT/DPO fitting must use the repository-managed training environment:

```bash
scripts/ops/bootstrap-training-env.sh
scripts/ops/ed4all-training --help
```

The bootstrap is offline-first and consumes
`$ED4ALL_TRAINING_WHEEL_DIR` (default: `~/wheel-cache/training-band`). The
launcher fails before model weight loading when Torch, Transformers, TRL,
PEFT, Accelerate, or Datasets falls outside the qualified version band.
Do not repair this by modifying the system Python environment.

On Linux aarch64, the automatic `gb10-cu130` profile requires NVIDIA GB10
(SM 12.1), Torch `2.13.0+cu130`, CUDA 13.0, Transformers 4.57.6, TRL 0.26.2,
PEFT 0.19.1, Accelerate 1.12.0, and Datasets 4.4.1. It also imports the exact
SSM fast-path symbols before any model weights are loaded. The cached
architecture-specific wheels are verified before installation:

| Wheel | SHA-256 |
|---|---|
| `causal_conv1d-1.6.2.post1-cp312-cp312-linux_aarch64.whl` | `0995ceb7e43deffc8860d357c02a0cc5ce8a0cc31018e35d3d2665e3ec4dd703` |
| `mamba_ssm-2.3.2.post1-cp312-cp312-linux_aarch64.whl` | `ef9b8c7d4363fd7486dc4c7eccf6c37e5e0d9bc0f2cf6bb1ad798d9e154c7abc` |

The default remains offline-first. If the cache lacks portable dependencies,
the bootstrap announces that it is resolving the same pinned constraints from
the configured Python index. Set `ED4ALL_TRAINING_OFFLINE_ONLY=true` to require
a complete air-gapped cache. Set `ED4ALL_TRAINING_PROFILE=gb10-cu130` to select
the profile explicitly; an incompatible CPU architecture, CUDA build, GPU, or
extension ABI fails loudly.

## Required preflight

Before a production run, use a copied course override file containing:

```yaml
base_model: nemotron3-nano-30b
max_steps: 1
dpo_learning_rate: 1.0e-6
```

`max_steps: 1` is a canary-only bound. Run the normal `trainforge_train`
invocation through `scripts/ops/ed4all-training`, inspect the SFT and DPO loss,
peak allocated/reserved GPU memory, host available memory, and wall time, then
repeat with the candidate DPO rates selected by the operator. Remove
`max_steps` and pin the measured DPO rate before production. The Nano recipe
deliberately refuses to reuse the SFT learning rate when
`dpo_learning_rate` is unset.

The production gates are:

1. SFT completes one optimizer step without OOM or non-finite loss.
2. DPO completes one optimizer step at an independently measured rate.
3. The selected DPO checkpoint meets or exceeds the selected SFT checkpoint
   on the same downstream probe. Regression refuses promotion.
4. Evaluation compares base and adapter from one loaded BF16 base using
   PEFT's adapter-disable context. It must not load a second 59-GiB base.
5. Prompts are rendered by the pinned tokenizer template with
   `enable_thinking=false`; hand-written ChatML is not an evaluation path.

No canary command is run automatically. GPU execution and go/no-go remain
operator decisions.

## Serving handoff

Do not assume dynamic LoRA support from a backend name alone. For an installed
TensorRT-LLM release that has not explicitly qualified Nemotron-H dynamic
adapters, export the promoted adapter as a merged BF16 Hugging Face checkpoint
with `Trainforge.training.adapter_export.merge_promoted_adapter`, then build
the TensorRT engine from that immutable directory. This preserves the learned
feature; it does not disable the adapter. Keep the original adapter alongside
the merged artifact for provenance and rollback.

An unmerged vLLM alternative is permitted only after the installed vLLM
release reports LoRA support for the exact architecture and a fixed-prompt
coherence check shows the same tokenization and materially equivalent output.
Unsupported combinations fail loud; there is no silent backend fallback.
