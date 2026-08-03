# Installation and Dependencies

Ed4All supports Python 3.10 or newer. The repository contains first-party
source, dependency declarations, constraints, and installation instructions.
It does **not** host installed packages, virtual environments, package caches,
browser binaries, model or tokenizer files, native wheels, training data,
generated course material, or other dependency payloads.

Choose the smallest installation that supports the task. Heavy machine-learning
stacks are optional and intentionally excluded from the base environment.

## Base installation

From the repository root, create an isolated environment and install the
project in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with its platform-provided activation
script before running the same `python -m pip` command.

The base install provides the CLI, shared contracts, orchestration foundations,
and core validation dependencies. Keep the environment outside version control;
never copy an environment or its package cache into the repository.

## Optional extras by task

Extras are additive capabilities declared in `pyproject.toml`:

| Task | Extra | What it adds |
|---|---|---|
| Run the MCP server | `server` | MCP server runtime |
| Develop and test | `dev` | Pytest, coverage, formatting, and linting tools |
| Run the browser control plane | `gui` | FastAPI, Uvicorn, form handling, and environment loading |
| Use CPU-light application surfaces | `full` | `server`, `dev`, and `gui` together |
| Build dense indexes and use hybrid RRF retrieval | `embedding` | Sentence embeddings, numeric tooling, clustering, and statistical validators |
| Convert PDF learning material with SemantiK | `semantik` | PDF, OCR integration, browser audit, language, and vision support |
| Train SFT/DPO LoRA adapters | `training` | PyTorch, Transformers, TRL, PEFT, quantization, datasets, and evaluation tools |
| Use explicitly selected Anthropic API mode | `anthropic` | Anthropic SDK only; provider use remains subject to licensing policy |
| Run isolated metric calibration | `eval-calibration` | Calibration-only evaluation tooling, not a runtime dependency |

Common public installations are:

```bash
python -m pip install -e '.[full]'
python -m pip install -e '.[full,embedding]'
python -m pip install -e '.[full,semantik]'
python -m pip install -e '.[full,training]'
```

Do not combine heavy extras automatically. Their accelerator and Transformers
requirements may target different qualified environments. Use the managed
training environment below for real adapter fitting, and use a separate
conversion or retrieval environment when resolver constraints require it.

Read [Licensing and ToS posture](../LICENSING.md) before enabling a hosted
provider or producing training data. Installing an SDK does not authorize its
outputs for training.

## Managed training environment

Real Trainforge fitting uses a repository-managed environment rather than the
general development environment:

```bash
scripts/ops/bootstrap-training-env.sh
scripts/ops/ed4all-training --help
```

The bootstrap script creates `.venv-training`, selects a supported runtime
profile, installs against the corresponding file under `config/`, and runs the
fail-loud runtime preflight before returning success. The wrapper refuses to run
when that managed environment is absent.

The bootstrap is offline-first. Set `ED4ALL_TRAINING_WHEEL_DIR` to an
operator-controlled wheel cache outside the repository. If the cache is
incomplete, the script may resolve the same constrained dependencies from the
configured Python index. Set `ED4ALL_TRAINING_OFFLINE_ONLY=true` to prohibit
network resolution; an incomplete cache then fails rather than degrading to a
source build or a different dependency band.

`ED4ALL_TRAINING_PROFILE` may select a profile supported by the script; leave it
at its default automatic selection unless the deployment documentation requires
an explicit profile. Do not edit system Python, bypass
`Trainforge.training.runtime_preflight`, or loosen the constraints to make an
unsupported environment start. Version, accelerator, and native-extension
mismatches must fail before model weights load.

The repository tracks constraints and verification logic, not the wheels or
their private cache metadata. Native binaries, local checksums, and downloaded
artifacts stay in the operator-managed cache.

## Platform dependencies

Some capabilities require software installed outside Python:

- SemantiK can use Tesseract OCR and Poppler for scanned or image-heavy PDF
  conversion. Install them with the operating system's supported package
  manager and verify their executables are on `PATH`.
- Playwright installs its browser separately from the Python package:

  ```bash
  python -m playwright install chromium
  ```

- GPU inference or training requires a compatible driver, runtime, and
  accelerator-specific PyTorch distribution. The managed training preflight is
  authoritative; do not infer compatibility merely because `import torch`
  succeeds.
- Local `llama-cpp-python` acceleration is an external native build. Build it
  for the target platform using the upstream project instructions and verify
  that the intended accelerator backend loaded. Do not commit the resulting
  wheel or shared library.

For the supported Studio container workflow, use the checked-in compose and
image definitions and follow [Docker deployment](docker.md):

```bash
docker compose config
docker compose build gui
```

Container images, layers, named volumes, pulled models, and service caches are
external runtime state and must not be added to Git.

## IMS Common Cartridge schemas

QTI and Common Cartridge conformance validation requires upstream 1EdTech and
W3C schema files. They are deliberately not distributed by this repository.
Follow the filename, provenance, placement, import-rewrite, and verification
instructions in
[`Courseforge/schemas/imscc/README.md`](../../Courseforge/schemas/imscc/README.md).

Missing or unreadable schemas produce blocking `QTI_XSD_MISSING` or
`CARTRIDGE_XSD_MISSING` findings. There is no partial-validation fallback and
no permission to substitute unrelated schema versions.

## External services and models

Ed4All can connect to separately operated local or hosted services. Install and
configure only the service required by the selected workflow:

- local OpenAI-compatible inference endpoints provide authoring or synthesis;
- the compose stack can provide its documented local answer service;
- hosted providers require their optional SDK, credentials supplied through
  private environment configuration, and an approved licensing posture; and
- embedding, reranking, NLI, vision, and training models are acquired through
  their upstream tooling into operator-controlled storage.

Installation does not define a default model identifier. Resolve model choices
from the active configuration and provider documentation, then verify the model
license, architecture compatibility, context requirements, and endpoint
identity before a run. Never commit credentials, provider responses, model
snapshots, tokenizers, adapters, or model caches.

## Offline operation and caches

Prepare offline installations before disconnecting:

1. Resolve packages, browser assets, schemas, models, and native extensions on
   an authorized connected system.
2. Verify provenance and licensing outside the repository.
3. Store payloads in operator-controlled caches or runtime volumes.
4. Point the relevant package manager or service at those caches.
5. Enable the component's offline-only setting where one is provided and run
   its preflight.

Common caches such as pip, Hugging Face, Playwright, model-server storage, and
the managed training wheel directory must remain outside tracked source. An
offline miss is an installation failure; do not silently fetch from an
unapproved service, switch models, select CPU, or skip validation.

## Verify the installation

Start with package and CLI checks:

```bash
python -m pip check
ed4all --help
ed4all doctor
```

`ed4all doctor` is a diagnostic command: a degraded or failed check returns a
nonzero status and should be resolved before the matching workflow runs. It
does not install packages or modify the environment.

With the `dev` extra installed, verify the installation contracts:

```bash
pytest -q cli/tests/test_doctor_command.py \
  Trainforge/tests/test_training_runtime_preflight.py \
  lib/validators/tests/test_qti_well_formed.py \
  lib/validators/tests/test_cartridge_conformance.py
```

For a configured workflow, run the workflow-aware doctor so provider and
runtime checks match the requested execution path:

```bash
ed4all doctor --run <workflow-name> --mode local
```

Replace the placeholder with a workflow declared in `config/workflows.yaml`;
`ed4all run --help` documents the command syntax and accepted options.

## Troubleshooting

- **`ed4all` is not found** — activate the environment used for installation
  and rerun `python -m pip install -e .`.
- **`python -m pip check` reports conflicts** — create a clean environment and
  install only the extras required for that task. Do not force incompatible
  heavy extras into one environment.
- **An embedding dependency is missing** — install `.[embedding]`. Strict
  embedding mode fails closed; a missing package is not a successful
  statistical validation.
- **An embedding device is unavailable** — explicitly select a supported
  device. Ed4All does not silently fall back from the requested CUDA device to
  CPU.
- **Playwright cannot launch Chromium** — rerun
  `python -m playwright install chromium` in the same environment and confirm
  its external cache is readable.
- **OCR or PDF conversion tools are missing** — verify Tesseract and Poppler on
  `PATH`; reinstall them through the operating system rather than copying
  binaries into the checkout.
- **Common Cartridge validation reports a missing schema** — complete the
  upstream schema installation exactly as described above and rerun the gate.
- **Training bootstrap reports an incomplete offline cache** — populate the
  external wheel cache with the constrained runtime or allow the configured
  index for that bootstrap. Do not disable the preflight.
- **Training preflight rejects versions or native extensions** — rebuild the
  managed environment from its supported constraints. Do not patch the version
  check or continue to weight loading.
- **A local service is unreachable or exposes the wrong model** — run
  `ed4all doctor --ping`, inspect the service's own health endpoint, and confirm
  the active endpoint and model configuration without publishing credentials.

When an installation dependency is absent, install and verify it, then rerun
the same command or gate. Never lower a validation threshold or reinterpret a
dependency failure as a pass.
