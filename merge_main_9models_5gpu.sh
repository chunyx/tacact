#!/usr/bin/env bash
set -euo pipefail

# Usage:
# 1) Default: pass a RUN_ROOT that contains gpu0..gpu4 directories
#    ./merge_main_9models_5gpu.sh outputs_main_9models_5gpu/run_20260408_123000 /home/yaxin/datasets/TacAct-original
# 2) Or pass explicit metrics CSV list via METRICS_CSVS env var
#    METRICS_CSVS="a.csv,b.csv,c.csv,d.csv,e.csv" ./merge_main_9models_5gpu.sh <RUN_ROOT> <DATA_ROOT>

RUN_ROOT="${1:-}"
DATA_ROOT="${2:-${DATA_ROOT:-/home/yaxin/datasets/TacAct-original}}"
SEED="${SEED:-42}"
SPLIT_MODE="${SPLIT_MODE:-subject}"

if [[ -z "${RUN_ROOT}" ]]; then
  echo "Usage: $0 <RUN_ROOT> [DATA_ROOT]" >&2
  exit 1
fi

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "[ERROR] RUN_ROOT does not exist: ${RUN_ROOT}" >&2
  exit 1
fi

MERGED_DIR="${RUN_ROOT}/merged"
mkdir -p "${MERGED_DIR}"

if [[ -n "${METRICS_CSVS:-}" ]]; then
  METRICS_ARG="${METRICS_CSVS}"
else
  M0="${RUN_ROOT}/gpu0/${SPLIT_MODE}_seed${SEED}/metrics.csv"
  M1="${RUN_ROOT}/gpu1/${SPLIT_MODE}_seed${SEED}/metrics.csv"
  M2="${RUN_ROOT}/gpu2/${SPLIT_MODE}_seed${SEED}/metrics.csv"
  M3="${RUN_ROOT}/gpu3/${SPLIT_MODE}_seed${SEED}/metrics.csv"
  M4="${RUN_ROOT}/gpu4/${SPLIT_MODE}_seed${SEED}/metrics.csv"
  METRICS_ARG="${M0},${M1},${M2},${M3},${M4}"
fi

IFS=',' read -r -a paths <<< "${METRICS_ARG}"
MISSING=()
for p in "${paths[@]}"; do
  if [[ ! -f "${p}" ]]; then
    MISSING+=("${p}")
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "[ERROR] Missing metrics csv files:" >&2
  for p in "${MISSING[@]}"; do
    echo "  - ${p}" >&2
  done
  exit 1
fi

CMD=(
  python benchmark_data_loading/experiment_tacact.py
  --data_root "${DATA_ROOT}"
  --output_dir "${MERGED_DIR}"
  --merge_metrics_csvs "${METRICS_ARG}"
)

echo "[Merge] ${CMD[*]}"
"${CMD[@]}"

echo "[DONE] Merged outputs: ${MERGED_DIR}"
