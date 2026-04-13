#!/usr/bin/env bash
set -euo pipefail

STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-90}"
STATUS_TIMEOUT_SEC="${STATUS_TIMEOUT_SEC:-180}"

echo "Starting training..."

TMP_RUN_LOG="$(mktemp -t tacact_run_main_8gpu_XXXX.log)"

cleanup() {
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
    echo "[Cleanup] Training process still running (PID=${TRAIN_PID}), stopping it..."
    kill "${TRAIN_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM HUP

bash run_main_10models_8gpu.sh >"${TMP_RUN_LOG}" 2>&1 &
TRAIN_PID=$!
echo "Training PID: ${TRAIN_PID}"
echo "Capturing startup logs at: ${TMP_RUN_LOG}"

run_root=""
start_ts="$(date +%s)"
while true; do
  if grep -q '\[Run Root\] ' "${TMP_RUN_LOG}"; then
    run_root="$(sed -n 's/^.*\[Run Root\] //p' "${TMP_RUN_LOG}" | head -n1 | tr -d '\r')"
    break
  fi
  if ! kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
    echo "[ERROR] Training exited before emitting [Run Root]."
    echo "----- training output -----"
    cat "${TMP_RUN_LOG}" || true
    exit 1
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts > STARTUP_TIMEOUT_SEC )); then
    echo "[ERROR] Could not parse [Run Root] within ${STARTUP_TIMEOUT_SEC}s."
    echo "----- partial training output -----"
    cat "${TMP_RUN_LOG}" || true
    kill "${TRAIN_PID}" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep 0.5
done

if [[ -z "${run_root}" ]]; then
  echo "[ERROR] Parsed RUN_ROOT is empty."
  echo "----- training output -----"
  cat "${TMP_RUN_LOG}" || true
  kill "${TRAIN_PID}" >/dev/null 2>&1 || true
  exit 1
fi

echo "Detected RUN_ROOT: ${run_root}"
status_dir="${run_root}/status"
echo "Master log: ${run_root}/master_run.log"
echo "Waiting for status files..."

wait_start_ts="$(date +%s)"
while true; do
  if [[ -d "${status_dir}" ]]; then
    break
  fi
  if ! kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
    echo "[ERROR] Training exited before status directory appeared: ${status_dir}"
    echo "----- training output -----"
    cat "${TMP_RUN_LOG}" || true
    exit 1
  fi
  now_ts="$(date +%s)"
  if (( now_ts - wait_start_ts > STATUS_TIMEOUT_SEC )); then
    echo "[ERROR] Status directory not found within ${STATUS_TIMEOUT_SEC}s: ${status_dir}"
    echo "----- training output -----"
    cat "${TMP_RUN_LOG}" || true
    kill "${TRAIN_PID}" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep 0.5
done

echo "Starting watcher..."
python watch_main_9models_5gpu.py --run_root "${run_root}" --gpu_count 8

wait "${TRAIN_PID}" || {
  rc=$?
  echo "[ERROR] Training finished with non-zero exit code: ${rc}"
  echo "----- training output (tail) -----"
  tail -n 200 "${TMP_RUN_LOG}" || true
  exit "${rc}"
}

echo "Training and monitoring finished successfully."
echo "Run root: ${run_root}"
echo "Master log: ${run_root}/master_run.log"
echo "Main logs:"
for g in 0 1 2 3 4 5 6 7; do
  echo "  ${run_root}/logs/gpu${g}.log"
done
echo "Startup capture log: ${TMP_RUN_LOG}"
