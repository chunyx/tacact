#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/yaxin/datasets/TacAct-original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_main_experiment_fastio_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
SPLIT_MODE="${SPLIT_MODE:-subject}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
AMP_INFER="${AMP_INFER:-1}"
SKIP_CACHE_WARMUP="${SKIP_CACHE_WARMUP:-1}"
NO_PRELOAD="${NO_PRELOAD:-1}"
CACHE_DIR="${CACHE_DIR:-.cache_tacact_n80_weighted}"
BEST_CONFIG_PATH="${BEST_CONFIG_PATH:-}"
HPO_ROOT="${HPO_ROOT:-}"
HPO_META_PATH="${HPO_META_PATH:-}"
DEFAULT_COMBINED_BEST_CONFIG="outputs_best_configs_combined/best_model_configs_10models.json"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  RUN_ROOT="${OUTPUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
else
  RUN_ROOT="${OUTPUT_ROOT}"
fi

LOG_DIR="${RUN_ROOT}/logs"
MERGED_DIR="${RUN_ROOT}/merged"
STATUS_DIR="${RUN_ROOT}/status"
MASTER_LOG="${RUN_ROOT}/master_run.log"
mkdir -p "${LOG_DIR}" "${MERGED_DIR}" "${STATUS_DIR}"
touch "${MASTER_LOG}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(timestamp)" "${msg}" | tee -a "${MASTER_LOG}"
}

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
if [[ -z "${BEST_CONFIG_PATH}" ]] && [[ -f "${DEFAULT_COMBINED_BEST_CONFIG}" ]]; then
  BEST_CONFIG_PATH="${DEFAULT_COMBINED_BEST_CONFIG}"
fi

# 9-GPU split, AlexNet removed from final fair comparison.
GPU0_MODELS="CNN_LSTM"
GPU1_MODELS="TCN"
GPU2_MODELS="ResNet18"
GPU3_MODELS="EfficientNet_B0"
GPU4_MODELS="LSTM"
GPU5_MODELS="GRU"
GPU6_MODELS="MobileNet_V2"
GPU7_MODELS="LeNet"
GPU8_MODELS=""

for g in 0 1 2 3 4 5 6 7 8; do
  mkdir -p "${RUN_ROOT}/gpu${g}"
done

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
  local out_dir="${RUN_ROOT}/gpu${gpu_id}"
  local log_file="${LOG_DIR}/gpu${gpu_id}.log"
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
  "exit_code": null,
  "signal": null,
  "launch_ts": 0,
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
    --prefetch_factor "${PREFETCH_FACTOR}"
    --cache_dir "${CACHE_DIR}"
  )
  if [[ -n "${BEST_CONFIG_PATH}" ]]; then
    cmd+=(--best_config_path "${BEST_CONFIG_PATH}")
  fi
  cmd+=("${EXTRA_FLAGS[@]}")

  log "[Launch][GPU${gpu_id}] CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
  TACACT_STATUS_FILE="${status_file}" \
  TACACT_GPU_ID="${gpu_id}" \
  TACACT_QUEUE_MODELS="${model_list}" \
  TACACT_QUEUE_TOTAL="${queue_total}" \
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${LOG_DIR}/gpu${gpu_id}.pid"
  python - <<PY >/dev/null 2>&1
import json
import time
from pathlib import Path
p=Path("${status_file}")
d=json.loads(p.read_text(encoding="utf-8"))
d["pid"]=${pid}
d["status"]="running"
d["exit_code"]=None
d["signal"]=None
d["launch_ts"]=time.time()
d["last_update_ts"]=time.time()
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  log "[PID][GPU${gpu_id}] ${pid} (log: ${log_file})"
}

update_final_status() {
  local status_file="$1"
  local final_status="$2"
  local exit_code="${3:-}"
  local signal_name="${4:-}"
  STATUS_FILE_ENV="${status_file}" \
  FINAL_STATUS_ENV="${final_status}" \
  EXIT_CODE_ENV="${exit_code}" \
  SIGNAL_NAME_ENV="${signal_name}" \
  python - <<'PY' >/dev/null 2>&1
import json
import os
import time
from pathlib import Path
p=Path(os.environ["STATUS_FILE_ENV"])
if p.exists():
    d=json.loads(p.read_text(encoding="utf-8"))
else:
    d={}
d["status"]=os.environ["FINAL_STATUS_ENV"]
exit_code_raw=os.environ.get("EXIT_CODE_ENV","").strip()
d["exit_code"]=int(exit_code_raw) if exit_code_raw else None
signal_name=os.environ.get("SIGNAL_NAME_ENV","").strip()
d["signal"]=signal_name or None
d["last_update_ts"]=time.time()
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

terminate_all_children() {
  local signal_name="$1"
  local signal_num="$2"
  for g in 0 1 2 3 4 5 6 7 8; do
    local pid_file="${LOG_DIR}/gpu${g}.pid"
    local status_file="${STATUS_DIR}/gpu${g}.json"
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if kill -0 "${pid}" >/dev/null 2>&1; then
        log "[Signal] ${signal_name} received; forwarding SIGTERM to GPU${g} child PID=${pid}"
        kill "${pid}" >/dev/null 2>&1 || true
      fi
      update_final_status "${status_file}" "terminated" "$((128 + signal_num))" "${signal_name}"
    fi
  done
}

on_signal() {
  local signal_name="$1"
  local signal_num="$2"
  log "[Launcher] Caught ${signal_name}; marking active jobs terminated"
  terminate_all_children "${signal_name}" "${signal_num}"
  exit "$((128 + signal_num))"
}

trap 'on_signal SIGHUP 1' HUP
trap 'on_signal SIGINT 2' INT
trap 'on_signal SIGTERM 15' TERM

log "[Run Root] ${RUN_ROOT}"
log "[Master Log] ${MASTER_LOG}"
log "[Data Root] ${DATA_ROOT}"
log "[Seed] ${SEED}"
log "[Split Mode] ${SPLIT_MODE}"
log "[Epochs] ${EPOCHS}"
log "[Loader] NUM_WORKERS=${NUM_WORKERS}, PREFETCH_FACTOR=${PREFETCH_FACTOR}"
log "[Deep-Only] enabled via --run_mode deep"
log "[Cache Dir] ${CACHE_DIR}"
log "[Warmup Policy] SKIP_CACHE_WARMUP=${SKIP_CACHE_WARMUP}, NO_PRELOAD=${NO_PRELOAD}"
if [[ -n "${BEST_CONFIG_PATH}" ]]; then
  log "[Best Config] Using: ${BEST_CONFIG_PATH}"
else
  log "[Best Config] Not provided; experiment_tacact defaults will be used."
fi
log "[GPU Split] GPU0: ${GPU0_MODELS}"
log "[GPU Split] GPU1: ${GPU1_MODELS}"
log "[GPU Split] GPU2: ${GPU2_MODELS}"
log "[GPU Split] GPU3: ${GPU3_MODELS}"
log "[GPU Split] GPU4: ${GPU4_MODELS}"
log "[GPU Split] GPU5: ${GPU5_MODELS}"
log "[GPU Split] GPU6: ${GPU6_MODELS}"
log "[GPU Split] GPU7: ${GPU7_MODELS}"
log "[GPU Split] GPU8: ${GPU8_MODELS}"
log "[Status Dir] ${STATUS_DIR}"
log "[Watcher] python watch_main_9models_5gpu.py --run_root ${RUN_ROOT} --gpu_count 8"

if [[ -n "${GPU0_MODELS}" ]]; then run_job 0 "${GPU0_MODELS}"; fi
if [[ -n "${GPU1_MODELS}" ]]; then run_job 1 "${GPU1_MODELS}"; fi
if [[ -n "${GPU2_MODELS}" ]]; then run_job 2 "${GPU2_MODELS}"; fi
if [[ -n "${GPU3_MODELS}" ]]; then run_job 3 "${GPU3_MODELS}"; fi
if [[ -n "${GPU4_MODELS}" ]]; then run_job 4 "${GPU4_MODELS}"; fi
if [[ -n "${GPU5_MODELS}" ]]; then run_job 5 "${GPU5_MODELS}"; fi
if [[ -n "${GPU6_MODELS}" ]]; then run_job 6 "${GPU6_MODELS}"; fi
if [[ -n "${GPU7_MODELS}" ]]; then run_job 7 "${GPU7_MODELS}"; fi
if [[ -n "${GPU8_MODELS}" ]]; then run_job 8 "${GPU8_MODELS}"; fi

set +e
FAIL_WAIT=0
for g in 0 1 2 3 4 5 6 7 8; do
  pid_file="${LOG_DIR}/gpu${g}.pid"
  status_file="${STATUS_DIR}/gpu${g}.json"
  if [[ ! -f "${pid_file}" ]]; then
    log "[ERROR] Missing pid file for GPU${g}: ${pid_file}"
    FAIL_WAIT=1
    update_final_status "${status_file}" "failed" "1" ""
    continue
  fi
  pid="$(cat "${pid_file}")"
  wait "${pid}"
  rc=$?
  if [[ ${rc} -eq 0 ]]; then
    log "[Exit][GPU${g}] rc=0"
    update_final_status "${status_file}" "done" "0" ""
  elif [[ ${rc} -ge 128 ]]; then
    sig_num=$((rc - 128))
    case "${sig_num}" in
      1) sig_name="SIGHUP" ;;
      2) sig_name="SIGINT" ;;
      15) sig_name="SIGTERM" ;;
      *) sig_name="SIG${sig_num}" ;;
    esac
    log "[Exit][GPU${g}] rc=${rc} signal=${sig_name}"
    update_final_status "${status_file}" "terminated" "${rc}" "${sig_name}"
    FAIL_WAIT=1
  else
    log "[Exit][GPU${g}] rc=${rc}"
    update_final_status "${status_file}" "failed" "${rc}" ""
    FAIL_WAIT=1
  fi
done
set -e

if [[ ${FAIL_WAIT} -ne 0 ]]; then
  log "[Run Result] at least one GPU job failed/terminated"
  exit 1
fi

METRICS_CSVS=""
for g in 0 1 2 3 4 5 6 7 8; do
  metrics_path="${RUN_ROOT}/gpu${g}/${SPLIT_MODE}_seed${SEED}/metrics.csv"
  if [[ ! -f "${metrics_path}" ]]; then
    log "[ERROR] Missing metrics file: ${metrics_path}"
    exit 1
  fi
  if [[ -z "${METRICS_CSVS}" ]]; then
    METRICS_CSVS="${metrics_path}"
  else
    METRICS_CSVS="${METRICS_CSVS},${metrics_path}"
  fi
done

log "[Merge] python benchmark_data_loading/experiment_tacact.py --data_root ${DATA_ROOT} --output_dir ${MERGED_DIR} --merge_metrics_csvs ${METRICS_CSVS}"
python benchmark_data_loading/experiment_tacact.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${MERGED_DIR}" \
  --merge_metrics_csvs "${METRICS_CSVS}"

log "[Done] merged outputs at ${MERGED_DIR}"
