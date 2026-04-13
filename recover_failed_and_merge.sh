#!/usr/bin/env bash
set -euo pipefail

# One-command conservative recovery workflow:
# 1) verify reusable old successful outputs
# 2) rerun only failed models into NEW recovery dir
# 3) verify new metrics exist
# 4) auto-merge old+new metrics into one final merged output

DATA_ROOT="${DATA_ROOT:-/home/yaxin/datasets/TacAct-original}"
OLD_RUN_ROOT="${OLD_RUN_ROOT:-outputs_main_9models_5gpu}"
RECOVERY_ROOT_BASE="${RECOVERY_ROOT_BASE:-outputs_main_9models_5gpu_recovery}"
SEED="${SEED:-42}"
SPLIT_MODE="${SPLIT_MODE:-subject}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AUTO_WATCH="${AUTO_WATCH:-1}"
WATCH_REFRESH_SEC="${WATCH_REFRESH_SEC:-1.0}"

# Use any 3 free physical GPUs for recovery rerun.
GPU_A="${GPU_A:-2}"  # TCN
GPU_B="${GPU_B:-3}"  # ResNet18
GPU_C="${GPU_C:-4}"  # LSTM+LeNet

# Keep same best-config chain
BEST_CONFIG_PATH="${BEST_CONFIG_PATH:-${OLD_RUN_ROOT}/best_model_configs.from_phase2.json}"

# Cache policy:
# - default: reuse existing shared cache for speed
# - optional one-time serial prefill to avoid parallel write race
CACHE_MODE="${CACHE_MODE:-shared}"  # shared | isolated
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-.cache_tacact_n80_weighted}"
PREFILL_SHARED_CACHE="${PREFILL_SHARED_CACHE:-1}"
CACHE_ROOT="${CACHE_ROOT:-${RECOVERY_ROOT_BASE}_cache}"

TS="$(date +%Y%m%d_%H%M%S)"
RECOVERY_ROOT="${RECOVERY_ROOT_BASE}_${TS}"
LOG_DIR="${RECOVERY_ROOT}/logs"
MERGED_DIR="${RECOVERY_ROOT}/merged_final"
STATUS_DIR="${RECOVERY_ROOT}/status"
mkdir -p "${LOG_DIR}" "${MERGED_DIR}" "${STATUS_DIR}"

OLD_GPU0_METRICS="${OLD_RUN_ROOT}/gpu0/${SPLIT_MODE}_seed${SEED}/metrics.csv"
OLD_GPU1_METRICS="${OLD_RUN_ROOT}/gpu1/${SPLIT_MODE}_seed${SEED}/metrics.csv"
OLD_GPU4_METRICS="${OLD_RUN_ROOT}/gpu4/${SPLIT_MODE}_seed${SEED}/metrics.csv"

verify_old_group() {
  local group_name="$1"
  local run_dir="$2"
  local metrics="$3"
  local runtime_csv="${run_dir}/runtime_summary.csv"
  local per_class_csv="${run_dir}/per_class_metrics.csv"

  if [[ ! -f "${metrics}" ]]; then
    echo "[ERROR] Missing reusable old metrics for ${group_name}: ${metrics}" >&2
    exit 1
  fi
  if [[ ! -f "${runtime_csv}" ]]; then
    echo "[ERROR] Missing runtime_summary for ${group_name}: ${runtime_csv}" >&2
    exit 1
  fi
  if [[ ! -f "${per_class_csv}" ]]; then
    echo "[ERROR] Missing per_class_metrics for ${group_name}: ${per_class_csv}" >&2
    exit 1
  fi

  python - <<PY
import pandas as pd
from pathlib import Path
p = Path("${metrics}")
df = pd.read_csv(p)
required = {"model", "accuracy", "macro_f1"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"[ERROR] {p} missing required columns: {sorted(missing)}")
if len(df) == 0:
    raise SystemExit(f"[ERROR] {p} is empty")
print(f"[OK] reusable metrics verified: {p} rows={len(df)} models={','.join(df['model'].astype(str).tolist())}")
PY
}

echo "[Stage A] Verify old reusable outputs..."
verify_old_group \
  "gpu0(CNN_LSTM)" \
  "${OLD_RUN_ROOT}/gpu0/${SPLIT_MODE}_seed${SEED}" \
  "${OLD_GPU0_METRICS}"
verify_old_group \
  "gpu1(ViT)" \
  "${OLD_RUN_ROOT}/gpu1/${SPLIT_MODE}_seed${SEED}" \
  "${OLD_GPU1_METRICS}"
verify_old_group \
  "gpu4(AlexNet,MobileNet_V2,EfficientNet_B0)" \
  "${OLD_RUN_ROOT}/gpu4/${SPLIT_MODE}_seed${SEED}" \
  "${OLD_GPU4_METRICS}"

if [[ ! -f "${BEST_CONFIG_PATH}" ]]; then
  echo "[ERROR] best config file not found: ${BEST_CONFIG_PATH}" >&2
  exit 1
fi

echo "[Stage B] Create new recovery dirs..."
RERUN_A_OUT="${RECOVERY_ROOT}/gpuA_tcn"
RERUN_B_OUT="${RECOVERY_ROOT}/gpuB_resnet18"
RERUN_C_OUT="${RECOVERY_ROOT}/gpuC_lstm_lenet"
mkdir -p "${RERUN_A_OUT}" "${RERUN_B_OUT}" "${RERUN_C_OUT}"

if [[ "${CACHE_MODE}" == "shared" ]]; then
  RERUN_A_CACHE="${SHARED_CACHE_DIR}"
  RERUN_B_CACHE="${SHARED_CACHE_DIR}"
  RERUN_C_CACHE="${SHARED_CACHE_DIR}"
  mkdir -p "${SHARED_CACHE_DIR}"
else
  RERUN_A_CACHE="${CACHE_ROOT}_${TS}/gpuA_tcn"
  RERUN_B_CACHE="${CACHE_ROOT}_${TS}/gpuB_resnet18"
  RERUN_C_CACHE="${CACHE_ROOT}_${TS}/gpuC_lstm_lenet"
  mkdir -p "${RERUN_A_CACHE}" "${RERUN_B_CACHE}" "${RERUN_C_CACHE}"
fi

echo "[Recovery Root] ${RECOVERY_ROOT}"
echo "[Watcher] python watch_main_9models_5gpu.py --run_root ${RECOVERY_ROOT}"
echo "[Old Reuse] ${OLD_GPU0_METRICS}"
echo "[Old Reuse] ${OLD_GPU1_METRICS}"
echo "[Old Reuse] ${OLD_GPU4_METRICS}"
echo "[Rerun Split] GPU${GPU_A}: TCN"
echo "[Rerun Split] GPU${GPU_B}: ResNet18"
echo "[Rerun Split] GPU${GPU_C}: LSTM,LeNet"
echo "[Best Config] ${BEST_CONFIG_PATH}"
echo "[Cache Mode] ${CACHE_MODE}"
echo "[Cache Dir] ${RERUN_A_CACHE}"

init_status_file() {
  local gpu="$1"
  local queue_models="$2"
  local queue_total=0
  if [[ -n "${queue_models}" ]]; then
    queue_total=$(awk -F',' '{print NF}' <<< "${queue_models}")
  fi
  cat > "${STATUS_DIR}/gpu${gpu}.json" <<JSON
{
  "status": "queued",
  "gpu_id": "${gpu}",
  "pid": null,
  "queue_models": [$(awk -v s="${queue_models}" 'BEGIN{n=split(s,a,","); for(i=1;i<=n;i++){gsub(/^ +| +$/,"",a[i]); if(a[i]!="") printf "\"%s\"%s", a[i], (i<n?", ":"")}}')],
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
}

# Prepare watcher-compatible status files for gpu0..gpu4
for g in 0 1 2 3 4; do
  init_status_file "${g}" ""
done
init_status_file "${GPU_A}" "TCN"
init_status_file "${GPU_B}" "ResNet18"
init_status_file "${GPU_C}" "LSTM,LeNet"

if [[ "${CACHE_MODE}" == "shared" && "${PREFILL_SHARED_CACHE}" == "1" ]]; then
  echo "[Stage B.1] Prefill shared cache once (serial) to avoid parallel cache-write race..."
  python - <<PY
import sys
from pathlib import Path
pkg_parent = Path(".").resolve().parent
if str(pkg_parent) not in sys.path:
    sys.path.insert(0, str(pkg_parent))
from tacact.benchmark_common import create_optimized_dataset, warmup_cache

data_root = Path("${DATA_ROOT}")
cache_dir = Path("${SHARED_CACHE_DIR}")
ds = create_optimized_dataset(
    data_root,
    n_frames=80,
    clip_mode="weighted_center",
    cache_dir=cache_dir,
    preload_cache=False,
)
warmup_cache(
    ds,
    batch_size=128,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
    use_tqdm=True,
    tqdm_desc="Recovery Cache Warmup",
)
print(f"[OK] Shared cache prefill done: {cache_dir}")
PY
fi

run_rerun_job() {
  local gpu="$1"
  local models="$2"
  local out_dir="$3"
  local cache_dir="$4"
  local log_file="$5"
  local pid_var_name="$6"

  local cmd=(
    python benchmark_data_loading/experiment_tacact.py
    --data_root "${DATA_ROOT}"
    --output_dir "${out_dir}"
    --run_mode deep
    --deep_models "${models}"
    --seed "${SEED}"
    --split_mode "${SPLIT_MODE}"
    --epochs "${EPOCHS}"
    --num_workers "${NUM_WORKERS}"
    --best_config_path "${BEST_CONFIG_PATH}"
    --cache_dir "${cache_dir}"
    --skip_cache_warmup
    --no_preload
    --amp_infer
  )

  echo "[Launch][GPU${gpu}] ${cmd[*]}" >&2
  TACACT_STATUS_FILE="${STATUS_DIR}/gpu${gpu}.json" \
  TACACT_GPU_ID="${gpu}" \
  TACACT_QUEUE_MODELS="${models}" \
  TACACT_QUEUE_TOTAL="$(awk -F',' '{print NF}' <<< "${models}")" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log_file}" 2>&1 &
  local pid="$!"
  printf -v "${pid_var_name}" '%s' "${pid}"
}

echo "[Stage C] Rerun failed models only (3 parallel jobs)..."
PID_A=""
PID_B=""
PID_C=""
run_rerun_job "${GPU_A}" "TCN" "${RERUN_A_OUT}" "${RERUN_A_CACHE}" "${LOG_DIR}/gpu${GPU_A}_tcn.log" PID_A
run_rerun_job "${GPU_B}" "ResNet18" "${RERUN_B_OUT}" "${RERUN_B_CACHE}" "${LOG_DIR}/gpu${GPU_B}_resnet18.log" PID_B
run_rerun_job "${GPU_C}" "LSTM,LeNet" "${RERUN_C_OUT}" "${RERUN_C_CACHE}" "${LOG_DIR}/gpu${GPU_C}_lstm_lenet.log" PID_C

WATCHER_PID=""
cleanup_watcher() {
  if [[ -n "${WATCHER_PID}" ]] && kill -0 "${WATCHER_PID}" 2>/dev/null; then
    kill "${WATCHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_watcher EXIT

if [[ "${AUTO_WATCH}" == "1" ]]; then
  echo "[Watcher] Auto-start (refresh=${WATCH_REFRESH_SEC}s)"
  (
    python watch_main_9models_5gpu.py \
      --run_root "${RECOVERY_ROOT}" \
      --refresh_sec "${WATCH_REFRESH_SEC}"
  ) &
  WATCHER_PID="$!"
fi

set +e
wait "${PID_A}"; RC_A=$?
wait "${PID_B}"; RC_B=$?
wait "${PID_C}"; RC_C=$?
set -e

cleanup_watcher

if [[ ${RC_A} -ne 0 || ${RC_B} -ne 0 || ${RC_C} -ne 0 ]]; then
  echo "[ERROR] At least one rerun job failed." >&2
  echo "  GPU${GPU_A} (TCN) rc=${RC_A} log=${LOG_DIR}/gpu${GPU_A}_tcn.log" >&2
  echo "  GPU${GPU_B} (ResNet18) rc=${RC_B} log=${LOG_DIR}/gpu${GPU_B}_resnet18.log" >&2
  echo "  GPU${GPU_C} (LSTM,LeNet) rc=${RC_C} log=${LOG_DIR}/gpu${GPU_C}_lstm_lenet.log" >&2
  exit 1
fi

echo "[Stage D] Verify new recovery metrics..."
NEW_A_METRICS="${RERUN_A_OUT}/${SPLIT_MODE}_seed${SEED}/metrics.csv"
NEW_B_METRICS="${RERUN_B_OUT}/${SPLIT_MODE}_seed${SEED}/metrics.csv"
NEW_C_METRICS="${RERUN_C_OUT}/${SPLIT_MODE}_seed${SEED}/metrics.csv"

for p in "${NEW_A_METRICS}" "${NEW_B_METRICS}" "${NEW_C_METRICS}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing rerun metrics: ${p}" >&2
    exit 1
  fi
done

python - <<PY
import pandas as pd
checks = {
  "${NEW_A_METRICS}": {"TCN"},
  "${NEW_B_METRICS}": {"ResNet18"},
  "${NEW_C_METRICS}": {"LSTM", "LeNet"},
}
for p, expected in checks.items():
    df = pd.read_csv(p)
    got = set(df["model"].astype(str).tolist())
    if not expected.issubset(got):
        raise SystemExit(f"[ERROR] {p} missing expected models. expected={sorted(expected)} got={sorted(got)}")
    print(f"[OK] {p} models={sorted(got)}")
PY

echo "[Stage E] Final merge old success + new recovery..."
MERGE_LIST="${OLD_GPU0_METRICS},${OLD_GPU1_METRICS},${OLD_GPU4_METRICS},${NEW_A_METRICS},${NEW_B_METRICS},${NEW_C_METRICS}"
MERGE_CMD=(
  python benchmark_data_loading/experiment_tacact.py
  --data_root "${DATA_ROOT}"
  --output_dir "${MERGED_DIR}"
  --merge_metrics_csvs "${MERGE_LIST}"
)
echo "[Merge] ${MERGE_CMD[*]}"
"${MERGE_CMD[@]}"

echo "[DONE] Recovery + merge finished."
echo "[Final Merged Output] ${MERGED_DIR}"
echo "[Merged Inputs]"
echo "  - ${OLD_GPU0_METRICS}"
echo "  - ${OLD_GPU1_METRICS}"
echo "  - ${OLD_GPU4_METRICS}"
echo "  - ${NEW_A_METRICS}"
echo "  - ${NEW_B_METRICS}"
echo "  - ${NEW_C_METRICS}"
