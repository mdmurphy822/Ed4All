#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${repo_dir}/.venv-training"
wheel_dir="${ED4ALL_TRAINING_WHEEL_DIR:-${HOME}/wheel-cache/training-band}"
profile="${ED4ALL_TRAINING_PROFILE:-auto}"
if [[ "${profile}" == "auto" ]]; then
  if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]]; then
    profile="gb10-cu130"
  else
    profile="generic"
  fi
fi
case "${profile}" in
  gb10-cu130)
    constraints="${repo_dir}/config/training-runtime-gb10-cu130-constraints.txt"
    torch_requirement="torch==2.13.0"
    ;;
  generic)
    constraints="${repo_dir}/config/training-runtime-constraints.txt"
    torch_requirement="torch==2.9.0"
    ;;
  *)
    echo "unsupported ED4ALL_TRAINING_PROFILE=${profile}; expected auto, generic, or gb10-cu130" >&2
    exit 2
    ;;
esac

if [[ "${profile}" == "gb10-cu130" ]]; then
  (
    cd "${wheel_dir}"
    sha256sum --check --status <<'EOF'
0995ceb7e43deffc8860d357c02a0cc5ce8a0cc31018e35d3d2665e3ec4dd703  causal_conv1d-1.6.2.post1-cp312-cp312-linux_aarch64.whl
ef9b8c7d4363fd7486dc4c7eccf6c37e5e0d9bc0f2cf6bb1ad798d9e154c7abc  mamba_ssm-2.3.2.post1-cp312-cp312-linux_aarch64.whl
EOF
  ) || {
    echo "GB10 SSM wheel cache is missing or failed SHA-256 verification: ${wheel_dir}" >&2
    exit 1
  }
fi

python3 -m venv --clear "${venv_dir}"
index_required=false
if ! "${venv_dir}/bin/python" -m pip install \
  --no-index --find-links "${wheel_dir}" \
  --constraint "${constraints}" "setuptools==80.9.0"; then
  if [[ "${ED4ALL_TRAINING_OFFLINE_ONLY:-false}" == "true" ]]; then
    echo "offline training wheel cache is incomplete: ${wheel_dir}" >&2
    exit 1
  fi
  index_required=true
  echo "offline training wheel cache lacks the pinned build backend; resolving it from the configured Python index" >&2
  "${venv_dir}/bin/python" -m pip install \
    --constraint "${constraints}" "setuptools==80.9.0"
fi

pip_cmd=(
  "${venv_dir}/bin/python" -m pip install
  --find-links "${wheel_dir}"
  --constraint "${constraints}"
  --no-build-isolation
  -e "${repo_dir}[training]"
  "${torch_requirement}"
  "mamba-ssm==2.3.2.post1"
  "causal-conv1d==1.6.2.post1"
)

if [[ "${index_required}" == "true" ]]; then
  "${pip_cmd[@]}"
elif ! "${pip_cmd[@]}" --no-index; then
  if [[ "${ED4ALL_TRAINING_OFFLINE_ONLY:-false}" == "true" ]]; then
    echo "offline training wheel cache is incomplete: ${wheel_dir}" >&2
    exit 1
  fi
  echo "offline training wheel cache is incomplete; resolving the same pinned contract from the configured Python index" >&2
  "${pip_cmd[@]}"
fi
if [[ "${profile}" == "gb10-cu130" ]]; then
  "${venv_dir}/bin/python" -c \
    "from Trainforge.training.runtime_preflight import assert_gb10_cu130_training_runtime; print(assert_gb10_cu130_training_runtime())"
else
  "${venv_dir}/bin/python" -c \
    "from Trainforge.training.runtime_preflight import assert_supported_training_runtime; print(assert_supported_training_runtime())"
fi
