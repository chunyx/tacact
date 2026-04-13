#!/usr/bin/env bash
set -euo pipefail

# ===============================
# Config (edit if needed)
# ===============================
DATA_ROOT="${DATA_ROOT:-/home/yaxin/datasets/TacAct-original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_main_9models_5gpu}"
SEED="${SEED:-42}"
SPLIT_MODE="${SPLIT_MODE:-subject}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_INFER="${AMP_INFER:-1}"            # 1 -> add --amp_infer
SKIP_CACHE_WARMUP="${SKIP_CACHE_WARMUP:-1}"  # 1 -> add --skip_cache_warmup
NO_PRELOAD="${NO_PRELOAD:-1}"          # 1 -> add --no_preload
BEST_CONFIG_PATH="${BEST_CONFIG_PATH:-}"   # optional JSON used by --best_config_path
HPO_ROOT="${HPO_ROOT:-}"                   # optional: auto-export from <HPO_ROOT>/phase2
HPO_META_PATH="${HPO_META_PATH:-}"         # optional: auto-export from hpo_pipeline_meta.json

# Conservative collision-avoidance: never overwrite previous runs silently.
# If OUTPUT_ROOT exists, create a timestamped subdir under it.
if [[ -e "${OUTPUT_ROOT}" ]]; then
  RUN_ROOT="${OUTPUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
else
  RUN_ROOT="${OUTPUT_ROOT}"
fi

LOG_DIR="${RUN_ROOT}/logs"
MERGED_DIR="${RUN_ROOT}/merged"
STATUS_DIR="${RUN_ROOT}/status"
mkdir -p "${LOG_DIR}" "${MERGED_DIR}" "${STATUS_DIR}"

# Optional best-config preparation:
# priority: BEST_CONFIG_PATH > HPO_ROOT > HPO_META_PATH
if [[ -z "${BEST_CONFIG_PATH}" ]] && [[ -n "${HPO_ROOT}" ]]; then
  AUTO_BEST_JSON="${RUN_ROOT}/best_model_configs.from_phase2.json"
  python benchmark_data_loading/export_phase2_best_configs.py \
    --hpo_root "${HPO_ROOT}" \
    --output "${AUTO_BEST_JSON}"
  BEST_CONFIG_PATH="${AUTO_BEST_JSON}"
fi
if [[ -z "${BEST_CONFIG_PATH}" ]] && [[ -n "${HPO_META_PATH}" ]]; then
  AUTO_BEST_JSON="${RUN_ROOT}/best_model_configs.from_meta.json"
  python benchmark_data_loading/export_phase2_best_configs.py \
    --meta_json "${HPO_META_PATH}" \
    --output "${AUTO_BEST_JSON}"
  BEST_CONFIG_PATH="${AUTO_BEST_JSON}"
fi

# ===============================
# GPU/model split
# ===============================
# NOTE:
# - Keep CNN_LSTM alone on one GPU as requested.
# - All jobs run deep-only via --run_mode deep.

GPU0_MODELS="CNN_LSTM"
GPU1_MODELS="ViT"
GPU2_MODELS="TCN,LeNet"
GPU3_MODELS="ResNet18,LSTM"
GPU4_MODELS="AlexNet,MobileNet_V2,EfficientNet_B0"

# Per-job output dirs
GPU0_OUT="${RUN_ROOT}/gpu0"
GPU1_OUT="${RUN_ROOT}/gpu1"
GPU2_OUT="${RUN_ROOT}/gpu2"
GPU3_OUT="${RUN_ROOT}/gpu3"
GPU4_OUT="${RUN_ROOT}/gpu4"
mkdir -p "${GPU0_OUT}" "${GPU1_OUT}" "${GPU2_OUT}" "${GPU3_OUT}" "${GPU4_OUT}"

# CLI optional flags
EXTRA_FLAGS=()
if [[ "${AMP_INFER}" == "1" ]]; then
  EXTRA_FLAGS+=("--amp_infer")
fi
if [[ "${SKIP_CACHE_WARMUP}" == "1" ]]; then
  EXTRA_FLAGS+=("--skip_cache_warmup")
fi
if [[ "${NO_PRELOAD}" == "1" ]]; then
  EXTRA_FLAGS+=("--no_preload")
fi

run_job() {
  local gpu_id="$1"
  local model_list="$2"
  local out_dir="$3"
  local log_file="$4"
  local status_file="${STATUS_DIR}/gpu${gpu_id}.json"
  local queue_total=0
  if [[ -n "${model_list}" ]]; then
    queue_total=$(awk -F',' '{print NF}' <<< "${model_list}")
  fi
  cat > "${status_file}" <<JSON
{
  "status": "queued",
  "gpu_id": "${gpu_id}",
  "pid": null,
  "queue_models": [$(awk -v s="${model_list}" 'BEGIN{n=split(s,a,","); for(i=1;i<=n;i++){gsub(/^ +| +$/,"",a[i]); printf "\"%s\"%s", a[i], (i<n?", ":"")}}')],
  "queue_total": ${queue_total},
  "queue_completed": 0,
  "current_model": null,
  "current_model_index": 0,
  "current_epoch": 0,
  "total_epochs": ${EPOCHS},
  "latest_val_f1": null,
  "last_update_ts": 0
}
JSON

  local cmd=(
    python benchmark_data_loading/experiment_tacact.py
    --data_root "${DATA_ROOT}"
    --output_dir "${out_dir}"
    --run_mode deep
    --deep_models "${model_list}"
    --seed "${SEED}"
    --split_mode "${SPLIT_MODE}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
  )
  if [[ -n "${BEST_CONFIG_PATH}" ]]; then
    cmd+=(--best_config_path "${BEST_CONFIG_PATH}")
  fi
  cmd+=("${EXTRA_FLAGS[@]}")

  echo "[Launch][GPU${gpu_id}] CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
  TACACT_STATUS_FILE="${status_file}" \
  TACACT_GPU_ID="${gpu_id}" \
  TACACT_QUEUE_MODELS="${model_list}" \
  TACACT_QUEUE_TOTAL="${queue_total}" \
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${LOG_DIR}/gpu${gpu_id}.pid"
  python - <<PY >/dev/null 2>&1
import json
from pathlib import Path
p=Path("${status_file}")
d=json.loads(p.read_text(encoding="utf-8"))
d["pid"]=${pid}
d["status"]="running"
import time
d["last_update_ts"]=time.time()
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  echo "[PID][GPU${gpu_id}] ${pid} (log: ${log_file})"
}

update_final_status() {
  local status_file="$1"
  local final_status="$2"
  python - <<PY >/dev/null 2>&1
import json
import time
from pathlib import Path
p=Path("${status_file}")
if p.exists():
    d=json.loads(p.read_text(encoding="utf-8"))
else:
    d={}
d["status"]="${final_status}"
d["last_update_ts"]=time.time()
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

echo "[Run Root] ${RUN_ROOT}"
echo "[Data Root] ${DATA_ROOT}"
echo "[Seed] ${SEED}"
echo "[Split Mode] ${SPLIT_MODE}"
echo "[Epochs] ${EPOCHS}"
echo "[Deep-Only] enabled via --run_mode deep"
echo "[Warmup Policy] SKIP_CACHE_WARMUP=${SKIP_CACHE_WARMUP}, NO_PRELOAD=${NO_PRELOAD}"
if [[ -n "${BEST_CONFIG_PATH}" ]]; then
  echo "[Best Config] Using: ${BEST_CONFIG_PATH}"
else
  echo "[Best Config] Not provided; experiment_tacact defaults will be used."
fi
echo "[GPU Split] GPU0: ${GPU0_MODELS}"
echo "[GPU Split] GPU1: ${GPU1_MODELS}"
echo "[GPU Split] GPU2: ${GPU2_MODELS}"
echo "[GPU Split] GPU3: ${GPU3_MODELS}"
echo "[GPU Split] GPU4: ${GPU4_MODELS}"
echo "[Status Dir] ${STATUS_DIR}"
echo "[Watcher] python watch_main_9models_5gpu.py --run_root ${RUN_ROOT}"

run_job 0 "${GPU0_MODELS}" "${GPU0_OUT}" "${LOG_DIR}/gpu0.log"
run_job 1 "${GPU1_MODELS}" "${GPU1_OUT}" "${LOG_DIR}/gpu1.log"
run_job 2 "${GPU2_MODELS}" "${GPU2_OUT}" "${LOG_DIR}/gpu2.log"
run_job 3 "${GPU3_MODELS}" "${GPU3_OUT}" "${LOG_DIR}/gpu3.log"
run_job 4 "${GPU4_MODELS}" "${GPU4_OUT}" "${LOG_DIR}/gpu4.log"

# Wait all jobs but continue collecting failures.
set +e
FAIL_WAIT=0
for g in 0 1 2 3 4; do
  pid_file="${LOG_DIR}/gpu${g}.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    wait "${pid}"
    rc=$?
    status_file="${STATUS_DIR}/gpu${g}.json"
    if [[ ${rc} -ne 0 ]]; then
      update_final_status "${status_file}" "failed"
      echo "[ERROR] GPU${g} job exited with code ${rc}" >&2
      FAIL_WAIT=1
    else
      update_final_status "${status_file}" "done"
      echo "[OK] GPU${g} job finished"
    fi
  else
    echo "[ERROR] Missing PID file for GPU${g}" >&2
    FAIL_WAIT=1
  fi
done
set -e

# Verify expected metrics.csv outputs
METRICS0="${GPU0_OUT}/${SPLIT_MODE}_seed${SEED}/metrics.csv"
METRICS1="${GPU1_OUT}/${SPLIT_MODE}_seed${SEED}/metrics.csv"
METRICS2="${GPU2_OUT}/${SPLIT_MODE}_seed${SEED}/metrics.csv"
METRICS3="${GPU3_OUT}/${SPLIT_MODE}_seed${SEED}/metrics.csv"
METRICS4="${GPU4_OUT}/${SPLIT_MODE}_seed${SEED}/metrics.csv"

MISSING=()
for p in "${METRICS0}" "${METRICS1}" "${METRICS2}" "${METRICS3}" "${METRICS4}"; do
  if [[ ! -f "${p}" ]]; then
    MISSING+=("${p}")
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "[ERROR] Missing metrics.csv from the following jobs:" >&2
  for p in "${MISSING[@]}"; do
    echo "  - ${p}" >&2
  done
  echo "[Hint] Check logs under: ${LOG_DIR}" >&2
  exit 1
fi

if [[ ${FAIL_WAIT} -ne 0 ]]; then
  echo "[ERROR] At least one job exited non-zero; abort merge." >&2
  exit 1
fi

# Merge all per-GPU metrics into one unified table + unified plots
MERGE_CMD=(
  python benchmark_data_loading/experiment_tacact.py
  --data_root "${DATA_ROOT}"
  --output_dir "${MERGED_DIR}"
  --merge_metrics_csvs "${METRICS0},${METRICS1},${METRICS2},${METRICS3},${METRICS4}"
)

echo "[Merge] ${MERGE_CMD[*]}"
"${MERGE_CMD[@]}"

echo "[DONE] Unified merged outputs: ${MERGED_DIR}"
echo "[DONE] Run root: ${RUN_ROOT}"
