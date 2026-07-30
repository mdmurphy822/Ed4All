#!/bin/bash
# =============================================================================
# Seat launch-spec TEMPLATE (sanitized — copy, fill the <PLACEHOLDERS>, keep
# your copy local-only; real seat scripts under seats/ are gitignored).
# =============================================================================
#
# A "seat" is one long-lived local vLLM container serving ONE model on a fixed
# loopback port. The pipeline addresses seats by LOGICAL name; the mapping from
# a logical name to a URL and to a launch script is data-driven via three envs:
#
#   ED4ALL_SEAT_BASE_URLS   <seat-name>=<loopback base_url>   (logical -> URL)
#   ED4ALL_VLLM_CONTAINERS  <base_url>=<container-name>       (URL -> container)
#   ED4ALL_SEAT_LAUNCH_SPECS <seat-name>=<abs path to THIS script>  (cold-recreate)
#
# The seat-schedule reconciler (ED4ALL_SEAT_SCHEDULE) COLD-RECREATES a seat by
# running its ED4ALL_SEAT_LAUNCH_SPECS script when the seat must (re)start or
# self-heal from a mode-collapse. See docs/operations/seat-schedule.env.example.
#
# RULES a seat script must honor:
#   * Cold rm + run, NEVER a warm `docker start` — a warm-restarted vLLM seat
#     can come up live-but-mode-collapsed (degenerate output). Always recreate.
#   * The schedule content-coherence-probes every (re)started seat; a launch
#     script only needs to bring the container up cleanly.
#   * NEVER co-resident: size --gpus / --gpu-memory-utilization so this seat and
#     any other simultaneously-scheduled seat fit the card. The default
#     small-box profile serves ONE heavy seat at a time (GPU-lifecycle lease);
#     only a large-unified-memory host runs several concurrently.
#   * Offline by construction: models are pre-seeded; pass HF_HUB_OFFLINE=1 so a
#     launch never phones home.
# =============================================================================

set -euo pipefail

CONTAINER="<CONTAINER_NAME>"          # e.g. vllm-myseat
HOST_PORT="<HOST_PORT>"              # loopback port this seat listens on, e.g. 8005
MODEL="<HF_MODEL_ID>"               # e.g. org/Model-Name
SERVED_NAME="<SERVED_MODEL_NAME>"    # must match the seat name your config expects
HF_CACHE="<ABS_PATH_TO_HF_CACHE>"    # e.g. /home/<user>/.cache/huggingface
VLLM_IMAGE="<VLLM_IMAGE>"           # e.g. nvcr.io/nvidia/vllm:<tag>

# Cold recreate: remove any stale container, then launch fresh.
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
sleep 3

docker run -d --name "${CONTAINER}" --gpus all --ipc=host \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -p "${HOST_PORT}:8000" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
  "${VLLM_IMAGE}" \
  vllm serve "${MODEL}" --served-model-name "${SERVED_NAME}" \
  --max-model-len <MAX_MODEL_LEN> \
  --gpu-memory-utilization <GPU_UTIL_0_TO_1> \
  --max-num-seqs <MAX_NUM_SEQS>
  # Add model-specific flags as needed, e.g.:
  #   --trust-remote-code
  #   --reasoning-parser <PARSER>
  #   --kv-cache-dtype fp8
  #   --allowed-local-media-path /      # vision seats reading local images
