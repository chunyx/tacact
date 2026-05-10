#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yaxin/tacact"
DATA_ROOT="/home/yaxin/datasets/TacAct-original"
BEST_CONFIG_PATH="/home/yaxin/tacact/outputs_hpo_three_stage_20260428_061937/best_model_configs.json"
CACHE_DIR="/home/yaxin/tacact/.cache_tacact_n80_weighted"
OUTPUT_ROOT="/home/yaxin/tacact/outputs/targeted_lenet_lstm_regularization_10seed_comparison/runs"
SUMMARY_ROOT="/home/yaxin/tacact/outputs/targeted_lenet_lstm_regularization_10seed_comparison"
ORIGINAL_RUNS_CSV="/home/yaxin/tacact/outputs/targeted_lenet_lstm_regularization_10seed_comparison/original_lenet_lstm_runs_found.csv"
SEEDS=(42 43 44 45 46 47 48 49 50 51)
EPOCHS=25
NUM_WORKERS=4
PREFETCH_FACTOR=2

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
mkdir -p "$MPLCONFIGDIR"

cd "$PROJECT_ROOT"
if [[ -f "/home/yaxin/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source /home/yaxin/miniconda3/etc/profile.d/conda.sh
  conda activate tacact_env
fi

echo "[Reg-only 10-seed] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[Reg-only 10-seed] Seeds: ${SEEDS[*]}"
echo "[Reg-only 10-seed] Output root: $OUTPUT_ROOT"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPUS=(0)
fi

NUM_GPUS="${#GPUS[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "[ERROR] No visible GPUs available."
  exit 1
fi

run_seed() {
  local seed="$1"
  local gpu="$2"
  local out_dir="${OUTPUT_ROOT}/reg_seed${seed}"
  echo
  echo "============================================================"
  echo "[Reg-only 10-seed] Launching seed ${seed} on GPU ${gpu}"
  echo "[Reg-only 10-seed] Output: ${out_dir}"
  echo "============================================================"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python benchmark_data_loading/experiment_tacact.py \
      --data_root "$DATA_ROOT" \
      --output_dir "$out_dir" \
      --run_mode deep \
      --deep_models "LeNet_LSTM" \
      --best_config_path "$BEST_CONFIG_PATH" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --split_mode subject \
      --num_workers "$NUM_WORKERS" \
      --prefetch_factor "$PREFETCH_FACTOR" \
      --cache_dir "$CACHE_DIR" \
      --skip_cache_warmup \
      --no_preload \
      --label_smoothing 0.05 \
      --weight_decay_override 3e-4
  )
}

FAIL=0
if [[ "$NUM_GPUS" -ge "${#SEEDS[@]}" ]]; then
  PIDS=()
  for idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[$idx]}"
    gpu="${GPUS[$idx]}"
    run_seed "$seed" "$gpu" &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      FAIL=1
    fi
  done
else
  echo "[Reg-only 10-seed] Fewer GPUs than seeds; running sequentially."
  for idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[$idx]}"
    gpu="${GPUS[$((idx % NUM_GPUS))]}"
    if ! run_seed "$seed" "$gpu"; then
      FAIL=1
    fi
  done
fi

echo
if [[ "$FAIL" -ne 0 ]]; then
  echo "[Reg-only 10-seed] One or more seed runs failed. Skipping summary."
  exit 1
fi

echo "[Reg-only 10-seed] Completed all seeds. Generating summary..."
python tools/summarize_lenet_lstm_reg_10seed_comparison.py \
  --original_runs_csv "$ORIGINAL_RUNS_CSV" \
  --improved_runs_root "$OUTPUT_ROOT" \
  --output_dir "$SUMMARY_ROOT"

echo "[Reg-only 10-seed] Summary completed under: $SUMMARY_ROOT"
