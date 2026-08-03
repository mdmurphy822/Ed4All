# Installation and local dependencies

Ed4All supports Python 3.10 or newer. The repository contains first-party
source and dependency manifests; it does not host downloaded packages, browser
binaries, model weights, tokenizer snapshots, training corpora, caches, or the
third-party IMS Common Cartridge schemas.

## Create an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Use a fresh virtual environment when changing heavy ML extras. Never copy a
virtual environment or package cache into the repository.

## Choose capabilities

The base install provides orchestration, schemas, and core validation. Extras
are additive:

| Extra | Capability |
|---|---|
| `server` | MCP server |
| `dev` | tests, coverage, and formatting tools |
| `gui` | browser control plane |
| `full` | CPU-light `server`, `dev`, and `gui` bundle |
| `embedding` | dense retrieval, hybrid RRF, and statistical validators |
| `training` | Transformers/TRL/PEFT SFT and DPO LoRA training |
| `semantik` | PDF-to-accessible-HTML conversion dependencies |
| `anthropic` | explicitly selected Anthropic SDK/API mode |
| `eval-calibration` | isolated RAGAS calibration only |

Typical installs:

```bash
pip install -e '.[full]'
pip install -e '.[full,embedding]'
pip install -e '.[full,embedding,semantik]'
pip install -e '.[full,training]'
```

Read [the licensing posture](../LICENSING.md) before enabling any hosted model
provider or training-data synthesis route.

## Platform dependencies

Tesseract OCR and Poppler improve scanned and image-heavy PDF conversion and
are installed through the operating system's package manager. GPU training and
local inference require a compatible driver, CUDA toolchain, and architecture-
appropriate PyTorch packages. The `training` extra is intentionally excluded
from `full` because its platform wheels are large and hardware-specific.

Local llama.cpp acceleration is an out-of-band install. Build
`llama-cpp-python` from source for the target CUDA architecture; do not accept a
CPU wheel when the intended deployment requires CUDA:

```bash
CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=<architecture>' \
  pip install --no-binary llama-cpp-python llama-cpp-python
```

## Playwright browser

The `semantik` extra installs the Python package, but Playwright manages its
Chromium binary separately:

```bash
playwright install chromium
```

The browser download remains in the operator's local Playwright cache. A
project-local `ms-playwright/` cache is ignored and must not be committed.

## IMS Common Cartridge schemas

QTI and cartridge validation require nine upstream IMS Global/1EdTech and W3C
XSD files. They are deliberately not distributed by this repository. Follow
the exact filename, provenance, placement, import-rewrite, and verification
instructions in
[`Courseforge/schemas/imscc/README.md`](../../Courseforge/schemas/imscc/README.md).

Missing or unreadable files produce blocking `QTI_XSD_MISSING` or
`CARTRIDGE_XSD_MISSING` findings. There is no partial-validation fallback.

## Models, tokenizers, and caches

Download models and tokenizers through the selected provider's documented
tooling into an operator-controlled cache outside Git. Generated embeddings,
course data, training pairs, adapters, evaluation artifacts, browser binaries,
package caches, and model files stay local. The repository ignores common
weight, cache, wheel, native-library, and package-install artifacts as a second
line of defense.

Model/provider licensing and the rules for training on generated outputs are
documented in [Licensing and ToS posture](../LICENSING.md). That document is
authoritative; installation does not imply that every optional provider is
appropriate for every workflow.

## Verify the environment

```bash
python -m pip check
ed4all --help
pytest -q lib/validators/tests/test_qti_well_formed.py \
  lib/validators/tests/test_cartridge_conformance.py
```

If a validation dependency is absent, install it and rerun the same gate. Do
not lower a threshold or treat a missing dependency as a successful check.
