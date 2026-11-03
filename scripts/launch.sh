#!/usr/bin/env bash
# Usage: scripts/launch.sh <config.yaml> [<config.yaml> ...] [-- overrides...]
# Runs every config in parallel, assigning GPUs round-robin: one per GPU on Kaggle, several on one A100.
# Env: OUT (runs), RESUME_ROOT (a dir holding <run_name>/ckpt_*.pt), MAX_HOURS (11.5), AUTO_STOP (1 stops the RunPod pod at the end)
set -u
CONFIGS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do CONFIGS+=("$1"); shift; done
[ $# -gt 0 ] && shift
EXTRA=("$@")
OUT=${OUT:-runs}
MAX_HOURS=${MAX_HOURS:-11.5}
LOGS=$OUT/logs
mkdir -p "$LOGS"
GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$GPUS" -ge 1 ] || GPUS=1

PIDS=(); NAMES=()
i=0
for config in "${CONFIGS[@]}"; do
    name=$(basename "$(dirname "$config")")_$(basename "$config" .yaml)
    resume=()
    if [ -n "${RESUME_ROOT:-}" ] && [ -d "$RESUME_ROOT/$name" ]; then
        resume=(--resume_from "$RESUME_ROOT/$name")
    fi
    gpu=$((i % GPUS))
    CUDA_VISIBLE_DEVICES=$gpu nohup python -m cruxtok.cli.train_tokenizer --config "$config" --max_hours "$MAX_HOURS" "${resume[@]}" \
        run_name="$name" runtime.output_root="$OUT" "${EXTRA[@]}" >> "$LOGS/$name.log" 2>&1 &
    PIDS+=($!); NAMES+=("$name")
    echo "[launch] $name on GPU $gpu (pid ${PIDS[-1]}); log $LOGS/$name.log"
    i=$((i + 1))
done

STATUS=0
for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}"; code=$?
    echo "[launch] ${NAMES[$j]} exit $code"
    [ "$code" -eq 0 ] || STATUS=1
done
if [ "${AUTO_STOP:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
exit $STATUS
