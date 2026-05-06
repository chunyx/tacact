#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/yaxin/datasets/TacAct-original}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-outputs_main_experiment_fastio}"
HPO_ROOT="${HPO_ROOT:-outputs_hpo_transformer_tuning}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_main_transformer_phase2_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-9}"
SEED="${SEED:-42}"
SPLIT_MODE="${SPLIT_MODE:-subject}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
AMP_INFER="${AMP_INFER:-1}"
SKIP_CACHE_WARMUP="${SKIP_CACHE_WARMUP:-1}"
NO_PRELOAD="${NO_PRELOAD:-1}"
CACHE_DIR="${CACHE_DIR:-.cache_tacact_n80_weighted}"
BASE_BEST_CONFIG_PATH="${BASE_BEST_CONFIG_PATH:-outputs_best_configs_combined/best_model_configs_10models.json}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  RUN_ROOT="${OUTPUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
else
  RUN_ROOT="${OUTPUT_ROOT}"
fi

LOG_DIR="${RUN_ROOT}/logs"
STATUS_DIR="${RUN_ROOT}/status"
MERGED_DIR="${RUN_ROOT}/merged_with_main9"
MASTER_LOG="${RUN_ROOT}/master_run.log"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}" "${MERGED_DIR}"
touch "${MASTER_LOG}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(timestamp)" "${msg}" | tee -a "${MASTER_LOG}"
}

BEST_CONFIG_PATH="${RUN_ROOT}/best_model_configs_11models.json"
BEST_CONFIG_PATH_ENV="${BEST_CONFIG_PATH}" \
BASE_BEST_CONFIG_PATH_ENV="${BASE_BEST_CONFIG_PATH}" \
HPO_ROOT_ENV="${HPO_ROOT}" \
python - <<'PY'
from pathlib import Path
import os
import json
import pandas as pd

base = Path(os.environ["BASE_BEST_CONFIG_PATH_ENV"])
hpo = Path(os.environ["HPO_ROOT_ENV"]) / "phase2" / "Transformer" / "phase2_results.csv"
out = Path(os.environ["BEST_CONFIG_PATH_ENV"])

obj = json.loads(base.read_text(encoding="utf-8"))
df = pd.read_csv(hpo)
best = df.sort_values("best_val_f1", ascending=False).iloc[0]
obj.setdefault("deep", {})["Transformer"] = {
    "params": {
        "lr": float(best["lr"]),
        "batch_size": int(best["batch_size"]),
        "weight_decay": float(best["weight_decay"]),
        "d_model": int(best["d_model"]),
        "nhead": int(best["nhead"]),
        "num_layers": int(best["num_layers"]),
        "dim_feedforward": int(best["dim_feedforward"]),
        "dropout": float(best["dropout"]),
        "pooling": str(best["pooling"]),
        "norm_first": bool(best["norm_first"]),
    }
}
out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY

STATUS_FILE="${STATUS_DIR}/gpu${GPU_ID}.json"
cat > "${STATUS_FILE}" <<JSON
{
  "status": "queued",
  "gpu_id": "${GPU_ID}",
  "pid": null,
  "queue_models": ["Transformer"],
  "queue_total": 1,
  "queue_completed": 0,
  "current_model": null,
  "current_model_index": 0,
  "current_epoch": 0,
  "total_epochs": ${EPOCHS},
  "latest_val_f1": null,
  "last_update_ts": 0
}
JSON

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

CMD=(
  python benchmark_data_loading/experiment_tacact.py
  --data_root "${DATA_ROOT}"
  --output_dir "${RUN_ROOT}/gpu${GPU_ID}"
  --run_mode deep
  --deep_models Transformer
  --seed "${SEED}"
  --split_mode "${SPLIT_MODE}"
  --epochs "${EPOCHS}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --cache_dir "${CACHE_DIR}"
  --best_config_path "${BEST_CONFIG_PATH}"
)
CMD+=("${EXTRA_FLAGS[@]}")

log "[Run Root] ${RUN_ROOT}"
log "[Master Log] ${MASTER_LOG}"
log "[Base Main Run] ${BASE_RUN_ROOT}"
log "[HPO Root] ${HPO_ROOT}"
log "[GPU] ${GPU_ID}"
log "[Best Config] ${BEST_CONFIG_PATH}"
log "[Launch] CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD[*]}"

TACACT_STATUS_FILE="${STATUS_FILE}" \
TACACT_GPU_ID="${GPU_ID}" \
TACACT_QUEUE_MODELS="Transformer" \
TACACT_QUEUE_TOTAL="1" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" >"${LOG_DIR}/gpu${GPU_ID}.log" 2>&1 &
PID=$!
echo "${PID}" > "${LOG_DIR}/gpu${GPU_ID}.pid"
log "[PID] ${PID} (log: ${LOG_DIR}/gpu${GPU_ID}.log)"

python - <<PY >/dev/null 2>&1
import json, time
from pathlib import Path
p=Path("${STATUS_FILE}")
d=json.loads(p.read_text(encoding="utf-8"))
d["pid"]=${PID}
d["status"]="running"
d["last_update_ts"]=time.time()
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
PY

wait "${PID}"
RC=$?

if [[ "${RC}" -ne 0 ]]; then
  log "[ERROR] Transformer single-GPU run failed with exit code ${RC}"
  exit "${RC}"
fi

TRANSFORMER_METRICS="${RUN_ROOT}/gpu${GPU_ID}/subject_seed${SEED}/metrics.csv"
if [[ ! -f "${TRANSFORMER_METRICS}" ]]; then
  log "[ERROR] Missing Transformer metrics.csv: ${TRANSFORMER_METRICS}"
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
  log "[WARN] Base 9-model main run is not complete yet (${#MERGE_INPUTS[@]}/9 metrics found). Skip auto-merge for now."
  log "[Next Step] After the 9-GPU main run finishes, run ./merge_transformer_into_main9.sh"
  echo "[DONE] Transformer single-GPU run: ${RUN_ROOT}"
  echo "[NEXT] Merge later with: ./merge_transformer_into_main9.sh"
  exit 0
fi

MERGE_INPUTS+=("${TRANSFORMER_METRICS}")
MERGE_CSVS="$(IFS=,; echo "${MERGE_INPUTS[*]}")"

log "[Merge] python benchmark_data_loading/experiment_tacact.py --data_root ${DATA_ROOT} --output_dir ${MERGED_DIR} --merge_metrics_csvs ${MERGE_CSVS}"
python benchmark_data_loading/experiment_tacact.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${MERGED_DIR}" \
  --merge_metrics_csvs "${MERGE_CSVS}"
log "[Done] Transformer run completed and merged outputs written to ${MERGED_DIR}"
echo "[DONE] Transformer single-GPU run: ${RUN_ROOT}"
echo "[DONE] Merged output: ${MERGED_DIR}"
