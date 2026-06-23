#!/usr/bin/env bash
# GPU allocation guard — coordinates GPU-heavy work across the concurrent
# Ed4All and a sibling Semantic session that share one local 8GB GPU.
# Deliberately interoperable with that sibling session's conventions:
#
#   * Shared flock mutex on /tmp/dart_qwen_train.lock — the SAME file
#     Semantic's dart_semantic/qwen_specialists/training_lock.py flocks. Holding
#     it here makes Semantic's (non-blocking) flock refuse, and vice-versa.
#   * VRAM gate: refuse to start while memory.used > 1500 MiB — the SAME
#     threshold Semantic's scripts/run_gpu_queue.sh uses. Catches ANY hog
#     (e.g. SemantiK's training/train_structure.py, which does NOT take the flock).
#
# Primary entrypoint:
#   scripts/gpu_guard.sh run --task LABEL -- <command...>
#       Block until (used <= MAX_USED) AND the shared flock is free, then hold
#       the flock for the command's entire lifetime (auto-released on exit) and
#       exec it. This is what the build launcher wraps `ed4all run` with.
#   scripts/gpu_guard.sh status            # GPU + procs + lock + threshold
#   scripts/gpu_guard.sh wait              # VRAM-gate only (no flock); for ad-hoc use
#
# Env overrides (flags win):
#   ED4ALL_GPU_LOCK         lock path     (default /tmp/dart_qwen_train.lock — shared w/ Semantic)
#   ED4ALL_GPU_MAX_USED_MB  busy ceiling  (default 1500 — matches Semantic run_gpu_queue.sh)
#   ED4ALL_GPU_POLL_SECONDS poll interval (default 15)
#   ED4ALL_GPU_TIMEOUT_SECONDS  max wait, 0 = forever (default 0)
set -uo pipefail

LOCK="${ED4ALL_GPU_LOCK:-/tmp/dart_qwen_train.lock}"
MAX_USED="${ED4ALL_GPU_MAX_USED_MB:-1500}"
POLL="${ED4ALL_GPU_POLL_SECONDS:-15}"
TIMEOUT="${ED4ALL_GPU_TIMEOUT_SECONDS:-0}"
TASK="${ED4ALL_GPU_TASK:-gpu-task}"

cmd="${1:-status}"; shift || true
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --max-used-mb) MAX_USED="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    --) shift; ARGS=("$@"); break ;;
    *) echo "gpu_guard: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

_smi() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi "$@" 2>/dev/null || /usr/lib/wsl/lib/nvidia-smi "$@" 2>/dev/null; }
_used_mb() { _smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
_gpu_line() { _smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader | head -1; }
_procs() {
  _smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=',' read -r p m; do
    p="$(echo "$p" | tr -dc '0-9')"; [ -z "$p" ] && continue
    echo "    PID $p (${m// /}): $(ps -p "$p" -o args= 2>/dev/null | cut -c1-72)"
  done
}

case "$cmd" in
  status)
    echo "GPU: $(_gpu_line)"
    echo "  compute processes:"; _procs
    echo "  shared lock: $LOCK"
    if command -v flock >/dev/null 2>&1; then
      ( exec 9>"$LOCK"; if flock -n 9; then echo "  lock state: FREE"; else echo "  lock state: HELD (another GPU job)"; fi )
    fi
    used="$(_used_mb)"; echo "  used=${used}MiB  busy_ceiling=${MAX_USED}MiB  -> $([ "${used:-99999}" -le "$MAX_USED" ] && echo CLEAR || echo BUSY)"
    ;;

  wait)
    start="$(date +%s)"
    while :; do
      used="$(_used_mb)"
      if [ "${used:-99999}" -le "$MAX_USED" ]; then echo "gpu_guard: GPU clear (used=${used}MiB <= ${MAX_USED}MiB)"; exit 0; fi
      now="$(date +%s)"; [ "$TIMEOUT" -gt 0 ] && [ $((now-start)) -ge "$TIMEOUT" ] && { echo "gpu_guard: TIMEOUT (used=${used}MiB)" >&2; exit 1; }
      echo "gpu_guard: waiting — used=${used}MiB > ${MAX_USED}MiB"; _procs; sleep "$POLL"
    done
    ;;

  run)
    [ "${#ARGS[@]}" -gt 0 ] || { echo "gpu_guard: 'run' needs -- <command>" >&2; exit 2; }
    command -v flock >/dev/null 2>&1 || { echo "gpu_guard: flock missing" >&2; exit 1; }
    exec 9>"$LOCK" || { echo "gpu_guard: cannot open lock $LOCK" >&2; exit 1; }
    start="$(date +%s)"
    while :; do
      used="$(_used_mb)"
      if [ "${used:-99999}" -le "$MAX_USED" ] && flock -w 10 9; then
        used2="$(_used_mb)"
        if [ "${used2:-99999}" -le "$MAX_USED" ]; then
          echo "gpu_guard: acquired (used=${used2}MiB, flock=$LOCK, task='$TASK'); exec: ${ARGS[*]}"
          exec "${ARGS[@]}"            # fd 9 stays open -> flock held until the command exits
        fi
        flock -u 9                      # VRAM refilled between checks — release + retry
      fi
      now="$(date +%s)"; [ "$TIMEOUT" -gt 0 ] && [ $((now-start)) -ge "$TIMEOUT" ] && { echo "gpu_guard: TIMEOUT (used=${used}MiB)" >&2; exit 1; }
      echo "gpu_guard: waiting ($(( $(date +%s) - start ))s) — used=${used}MiB > ${MAX_USED}MiB or lock held"; _procs
      sleep "$POLL"
    done
    ;;

  *) echo "gpu_guard: unknown command '$cmd' (status|wait|run)" >&2; exit 2 ;;
esac
