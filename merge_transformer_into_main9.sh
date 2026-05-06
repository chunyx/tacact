#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/yaxin/datasets/TacAct-original}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-outputs_main_experiment_fastio}"
TRANSFORMER_RUN_ROOT="${TRANSFORMER_RUN_ROOT:-outputs_main_transformer_phase2}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRANSFORMER_RUN_ROOT}/merged_with_main9}"

TRANSFORMER_METRICS="${TRANSFORMER_RUN_ROOT}/gpu9/subject_seed${SEED}/metrics.csv"
if [[ ! -f "${TRANSFORMER_METRICS}" ]]; then
  echo "[ERROR] Missing Transformer metrics: ${TRANSFORMER_METRICS}" >&2
  exit 1
fi

MERGE_INPUTS=()
if [[ -f "${BASE_RUN_ROOT}/merged/metrics_merged.csv" ]]; then
  MERGE_INPUTS+=("${BASE_RUN_ROOT}/merged/metrics_merged.csv")
else
  for g in 0 1 2 3 4 5 6 7 8; do
    m="${BASE_RUN_ROOT}/gpu${g}/subject_seed${SEED}/metrics.csv"
    if [[ -f "${m}" ]]; then
      MERGE_INPUTS+=("${m}")
    fi
  done
fi
if [[ ! -f "${BASE_RUN_ROOT}/merged/metrics_merged.csv" ]] && [[ "${#MERGE_INPUTS[@]}" -lt 9 ]]; then
  echo "[ERROR] Base 9-model main run is not complete yet (${#MERGE_INPUTS[@]}/9 metrics found)." >&2
  exit 1
fi
MERGE_INPUTS+=("${TRANSFORMER_METRICS}")
MERGE_CSVS="$(IFS=,; echo "${MERGE_INPUTS[*]}")"

mkdir -p "${OUTPUT_DIR}"
python benchmark_data_loading/experiment_tacact.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --merge_metrics_csvs "${MERGE_CSVS}"

echo "[DONE] Merged output: ${OUTPUT_DIR}"
