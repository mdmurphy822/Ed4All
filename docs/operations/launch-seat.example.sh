#!/usr/bin/env bash
# Portable Ed4All model-seat launcher template.
#
# Copy this file to gitignored runtime/seats/, set the required environment
# values in a private local configuration, and add deployment-specific device
# or vLLM arguments to the empty arrays below. Do not add credentials here.

set -euo pipefail

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf 'error: required setting %s is not set\n' "${name}" >&2
    exit 64
  fi
}

required=(
  ED4ALL_SEAT_CONTAINER
  ED4ALL_SEAT_HOST_PORT
  ED4ALL_SEAT_CONTAINER_PORT
  ED4ALL_SEAT_MODEL_PATH
  ED4ALL_SEAT_CONTAINER_MODEL_PATH
  ED4ALL_SEAT_SERVED_MODEL
  ED4ALL_SEAT_VLLM_IMAGE
)
for name in "${required[@]}"; do
  require_value "${name}"
done

for port_name in ED4ALL_SEAT_HOST_PORT ED4ALL_SEAT_CONTAINER_PORT; do
  port_value="${!port_name}"
  if [[ ! "${port_value}" =~ ^[0-9]+$ ]] ||
     (( port_value < 1 || port_value > 65535 )); then
    printf 'error: %s must be an integer from 1 to 65535\n' "${port_name}" >&2
    exit 64
  fi
done

if [[ ! -e "${ED4ALL_SEAT_MODEL_PATH}" ]]; then
  printf 'error: model path does not exist: %s\n' \
    "${ED4ALL_SEAT_MODEL_PATH}" >&2
  exit 66
fi

if ! command -v docker >/dev/null 2>&1; then
  printf 'error: docker is required to launch this seat\n' >&2
  exit 69
fi

# Add operator-selected device mounts and resource controls only in the private
# copied launcher. Keep each argument as its own array element.
device_args=()
vllm_args=()

# All validation occurs before the existing container is removed.
docker rm -f "${ED4ALL_SEAT_CONTAINER}" >/dev/null 2>&1 || true

exec docker run -d \
  --name "${ED4ALL_SEAT_CONTAINER}" \
  --ipc=host \
  --publish "127.0.0.1:${ED4ALL_SEAT_HOST_PORT}:${ED4ALL_SEAT_CONTAINER_PORT}" \
  --volume "${ED4ALL_SEAT_MODEL_PATH}:${ED4ALL_SEAT_CONTAINER_MODEL_PATH}:ro" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
  "${device_args[@]}" \
  "${ED4ALL_SEAT_VLLM_IMAGE}" \
  vllm serve "${ED4ALL_SEAT_CONTAINER_MODEL_PATH}" \
  --served-model-name "${ED4ALL_SEAT_SERVED_MODEL}" \
  "${vllm_args[@]}"
