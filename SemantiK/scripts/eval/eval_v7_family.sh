#!/usr/bin/env bash
# Run a family of PDFs through scripts/eval/make_pdf_vs_html.py, one process per
# PDF so VRAM is fully released between runs (the load-bearing fix for the
# math-heavy-short silent crash on 8GB VRAM after 3 sequential reasoner loads).
#
# Usage:
#   scripts/eval/eval_v7_family.sh PDF [PDF ...]
#
# At least one PDF argument is required. ADAPTER and CLASSIFIER are
# env-overridable; tracked code carries no corpus defaults.
#
# Outputs:
#   data/logs/v7_eval_<slug>.log         per-PDF stdout+stderr
#   data/logs/v7_family_eval_<ts>.log    top-level run log with status table
#   eval/side_by_side/<run_name>/...     per-PDF artifacts (input.pdf, output.html, result.json, index.html)
#   eval/results/v7_family_<ts>.json     aggregated summary
#
# Status detection is intentionally tolerant: a single failing PDF does
# not abort the rest of the batch — that's the whole point of running the
# family in one go.

set -u  # NOT -e: continue on per-PDF failures
cd "$(dirname "$0")/.."

if [[ $# -eq 0 ]]; then
  echo "usage: $0 PDF [PDF ...]" >&2
  exit 2
fi
PDFS=("$@")

# ENGINE: 'council' (default) runs the compatibility cascade (BERT council +
# Qwen GGUFs + theta v8) to evaluate newly trained council weights. It does not
# imply that the currently staged recipe is qualified. 'v1' runs the legacy
# classify+reason pipeline (uses ADAPTER).
ENGINE="${ENGINE:-council}"
ADAPTER="${ADAPTER:-models/reasoner_v8/final}"   # v1 engine only (reasoner_v7 retired -> v8)
CLASSIFIER="${CLASSIFIER:-models/classifier_v5/final}"
PY="${PY:-.venv/bin/python}"

TS="$(date +%Y%m%d-%H%M%S)"
SUMMARY_LOG="data/logs/v7_family_eval_${TS}.log"
RESULTS_JSON="eval/results/v7_family_${TS}.json"
SIDE_BY_SIDE_ROOT="eval/side_by_side"

mkdir -p "data/logs" "eval/results" "$SIDE_BY_SIDE_ROOT"

log() { echo "$@" | tee -a "$SUMMARY_LOG"; }

slugify() {
  "$PY" -c '
import re, sys
p = sys.argv[1]
print(re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_"))
' "$1"
}

read_json_field() {
  # $1=path, $2=key (top-level scalar)
  "$PY" -c '
import json, sys
try:
    print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))
except Exception:
    print("")
' "$1" "$2"
}

declare -a RUN_NAMES=()
declare -a STATUSES=()

log "=== v7 family eval START $(date '+%Y-%m-%d %H:%M:%S') ==="
log "  adapter:    $ADAPTER"
log "  classifier: $CLASSIFIER"
log "  pdfs:       ${#PDFS[@]}"
log ""

for pdf in "${PDFS[@]}"; do
  if [[ ! -f "$pdf" ]]; then
    log "[skip] $pdf: file not found"
    continue
  fi
  # The in-repo samples are all named input.pdf (eval/side_by_side/<X>/input.pdf),
  # so basename collides — fall back to the parent dir name when that happens.
  stem=$(basename "$pdf" .pdf)
  if [[ "$stem" == "input" ]]; then
    stem=$(basename "$(dirname "$pdf")")
  fi
  slug=$(slugify "$stem")
  run_name="${slug}_${ENGINE}"
  per_log="data/logs/v7_eval_${slug}_${ENGINE}.log"

  log "=== [$(date +%H:%M:%S)] ${run_name} ==="
  log "  pdf: $pdf"
  log "  log: $per_log"

  # Subshell — process exits on completion, kernel reclaims all VRAM
  # before the next iteration. This is the fix for the silent-crash-
  # on-third-load failure mode the previous ad-hoc loop hit.
  (
    "$PY" scripts/eval/make_pdf_vs_html.py "$pdf" \
        --engine "$ENGINE" \
        --adapter "$ADAPTER" \
        --classifier "$CLASSIFIER" \
        --run-name "$run_name"
  ) >"$per_log" 2>&1
  rc=$?

  rj="${SIDE_BY_SIDE_ROOT}/${run_name}/result.json"
  if [[ $rc -ne 0 ]]; then
    status="exit_${rc}"
  elif [[ ! -f "$rj" ]]; then
    status="missing_result_json"
  else
    n=$(read_json_field "$rj" n_classified_blocks)
    v=$(read_json_field "$rj" escalation_verdict)
    axe=$(read_json_field "$rj" axe_ok)
    if [[ -z "$n" || "$n" -lt 1 ]]; then
      status="zero_blocks"
    elif [[ "$ENGINE" == "council" ]]; then
      # council verdict = WCAG status; treat a real axe pass as ok.
      if [[ "$axe" == "True" || "$v" == "pass" || "$v" == "PASS" ]]; then
        status="ok"
      else
        status="wcag_${v:-empty}"
      fi
    elif [[ "$v" != "ship" && "$v" != "llm_fallback" ]]; then
      status="bad_verdict_${v:-empty}"
    else
      status="ok"
    fi
  fi
  log "  status: $status (rc=$rc)"
  log ""
  RUN_NAMES+=("$run_name")
  STATUSES+=("$status")
done

if [[ ${#RUN_NAMES[@]} -eq 0 ]]; then
  log "=== no runs were attempted ==="
  exit 1
fi

log "=== aggregating ==="
"$PY" scripts/eval/_aggregate_v7_eval.py \
    --out "$RESULTS_JSON" \
    --side-by-side-root "$SIDE_BY_SIDE_ROOT" \
    --adapter "$ADAPTER" \
    --classifier "$CLASSIFIER" \
    "${RUN_NAMES[@]}" 2>&1 | tee -a "$SUMMARY_LOG"

log ""
log "=== v7 family eval END $(date '+%Y-%m-%d %H:%M:%S') ==="
log "  results: $RESULTS_JSON"
log "  log:     $SUMMARY_LOG"

# Exit nonzero if any run failed — but only after writing the summary so
# the orchestrator's caller can still inspect partial output on failure.
for s in "${STATUSES[@]}"; do
  if [[ "$s" != "ok" ]]; then exit 2; fi
done
